# Physical-Field-Bridged Material Parameter Inference from Multi-View Deformation Videos

Linrui Shou<sup>1</sup> Jie Li<sup>2</sup> Yu Liu<sup>3</sup> Tao Huang<sup>1,\*</sup>

<sup>1</sup>Shanghai Jiao Tong University

<sup>2</sup>University of Science and Technology Beijing <sup>3</sup>Peking University

Corresponding author: t.huang@sjtu.edu.cn.

## Abstract

Estimating simulation-ready material parameters from visual observations is crucial for deformable-object manipulation, yet remains challenging because material properties are only indirectly expressed through physical responses such as contact, deformation, recovery, and stress concentration. We introduce physical-field-bridged material inference, a single-pass framework that treats dense physical responses as intermediate evidence between RGB deformation videos and a discrete constitutive type together with continuous material parameters. This design addresses a practical limitation of existing inverse-physics methods, which often rely on per-scene reconstruction, online tuning, or simulation-in-the-loop optimization, while feed-forward visual estimators can exploit appearance or category shortcuts. Given multi-view RGB videos, our model predicts force/contact masks, projected motion-flow fields, and projected stress-response maps, then conditions constitutive-type classification and parameter regression on these fields. To train and evaluate this formulation, we construct DeformFieldBench, a 3DGS–MPM learning environment with 5,000 simulations over 100 object shapes, five mechanically informative loading modes, and dense pixel-aligned physical supervision. Our method achieves the best median errors on all four continuous parameters, raises stress-hotspot recall from 0.0762 to 0.5121, and reaches 0.8676 SSIM/36.82 dB PSNR in same-simulator replay on unseen objects. Under a protocol-matched GSO evaluation, it performs inference and future-state prediction in approximately 30 seconds rather than the 15–120 minutes required by per-scene optimization; it also produces measurable replay agreement on five real-world sequences. These results support dense physical fields as an efective bridge for fast material inference, while cross-simulator and real-world tests expose remaining domain and identifiability gaps. The dataset is available on Hugging Face, and source code is available in the DeformFieldBench GitHub repository.

## Project and Code

Code repository

github.com/3365538768/DeformFieldBench

## Dataset

DeformFieldBench

huggingface.co/.../HandsomeHusky/DeformFieldBench

## Contents

1 Introduction 4   
2 Related Work 6   
3 DeformFieldBench: A Learning Environment for Material Inference   
3.1 Benchmark Design 8   
3.2 Dense Physical Field Annotations 9   
3.3 3DGS–MPM Simulation Procedure 9   
3.3.1 Input Representation 9   
3.3.2 Material Parameters 10   
3.3.3 Stress From Deformation 10   
3.3.4 One MPM Substep 11   
3.3.5 Stress Output 11   
3.3.6 Flow Output 12   
3.3.7 Force and Contact Mask Output 12   
3.3.8 Rendering Back to 3DGS 12   
3.4 Tasks and Evaluation Protocol 12   
3.5 Dataset Release and Usage Documentation 13   
4 Physical-Field-Bridged Material Inference 14   
4.1 Multi-view Visual Encoding 15   
4.2 Physics Bottleneck 15   
4.3 Dense Physical Field Prediction 16   
4.3.1 Dense Physical Field Decoder 16   
4.3.2 Sequential Stress Refinement 16   
4.4 Field-conditioned Constitutive and Parameter Prediction 16   
4.5 Training Objective 17   
4.5.1 Target-space Transformation and Validity Masking 18   
4.6 Staged Optimization 18   
5 Experimental Setup and Evaluation Protocol 18   
5.1 Data Organization 18   
5.2 Baselines 18   
5.3 Metrics Used in Tables 19   
6 Experiments 19   
6.1 Benchmark Comparison 20   
6.1.1 Parameter Prediction Visualization 20   
6.1.2 Material-wise Analysis 20   
6.2 Physical-Field Evidence 21   
6.2.1 Quantitative Field Evaluation 21   
6.2.2 Field Contribution Intervention 22   
6.2.3 Qualitative Field Prediction Results 22   
6.3 Object-level Generalization and Ablation Study 23   
6.4 Simulation-based Evaluation under Parameter Ambiguity . . . . . 23   
6.4.1 Simulation Replay Analysis 23   
6.5 Protocol-Matched Cross-Benchmark Evaluation 24   
6.5.1 GSO Future-State Protocol . 24   
6.5.2 Complete Spring-Gaus Synthetic Set 24   
6.5.3 Lightweight Domain Adaptation . 25   
6.5.4 Broader Robustness and Real-world Transfer . 25   
6.6 Expanded Object and Interaction Generalization 25   
6.6.1 Additional Unseen DifSplat Objects 25   
6.6.2 Dataset Scaling 26   
6.6.3 Held-Out Interactions 26   
6.7 Input Resolution and View Robustness 26   
6.7.1 Resolution Sensitivity 26   
6.7.2 Number and Configuration of Views 26   
6.8 Real-world Object Application 27   
7 Discussion and Limitations 28   
8 Conclusion 29   
References 29

## 1 Introduction

Embodied agents need to predict how objects respond to physical interaction. For deformable objects, this requires estimating not only geometry or semantic category, but also material parameters such as Young’s modulus, Poisson’s ratio, density, and yield stress. These parameters are simulation-ready: once inferred, they can be inserted into a physics simulator to forecast deformation under new actions. Recent progress in inverse physics, neural rendering, and physicsaware 3D representations has started to bridge visual perception and physical simulation, enabling material estimation from videos or reconstructed geometry [1–5]. Such a bridge is crucial for robotic manipulation, where perception should support actionable physical prediction rather than recognition alone.

However, current visual system-identification methods remain dificult to use as real-time perception modules. Many pipelines estimate material parameters through per-scene reconstruction [2, 6, 7] or simulation-in-the-loop optimization [1, 4, 5, 8], repeatedly rolling out a simulator, comparing rendered future states with observations, and updating parameters. These methods are physically grounded, but are often computationally expensive, scene-specific, and dependent on reconstructed geometry, camera information, or iterative test-time tuning. A more scalable alternative is amortized feed-forward inference from RGB images or videos [9–13]. Yet one-shot visual material inference is highly under-constrained: physical parameters are only indirectly expressed through appearance and motion, and existing feed-forward methods often rely on large vision-language models, semantic feature fields, interaction priors, or manually designed constraints. While useful, such priors can bias predictions toward object category, texture, or coarse material semantics. For deformable objects, a successful model should instead infer material properties from observed physical responses, such as contact, deformation propagation, recovery, and stress concentration.

This motivates a missing ingredient: dense physical supervision for learning the visual evidence behind material parameters. Existing deformable-object datasets and physical-reasoning benchmarks [14–20] often provide RGB videos, object states, or global labels, but rarely expose the intermediate physical fields that make material inference identifiable. Contact regions indicate where external forces are applied, motion fields reveal how deformation propagates, and stress responses highlight where force and deformation concentrate. Moreover, many benchmarks rely on limited geometries or action patterns dominated by dropping and impact, whereas compression, stretching, shearing, and bending reveal complementary properties such as stifness, plasticity, and stress localization. Without diverse loading modes and dense physical fields, feed-forward models can easily collapse into black-box video-to-parameter regression.

![](images/38e898e9a7b18eab0d6b6c7862311bc14f7733e792d8029bdfd44c6682158261.jpg)  
Figure 1. Overview of DeformFieldBench. Each sample combines three-view RGB observations with force/contact masks, projected flow, projected stress, object masks, a constitutive family, and continuous material parameters. The left panel illustrates pixel-aligned fields over loading and recovery, the center summarizes asset and parameter diversity relative to prior datasets, and the right shows the canonical views and five loading modes.

To address this gap, we construct DeformFieldBench, a 3DGS–MPM simulation environment for learning simulation-ready material inference from multi-view RGB videos. It contains 5,000 simulations over 100 everyday object shapes, five mechanically informative loading modes, multiple material categories, and broad parameter ranges. Each sample provides synchronized three-view RGB deformation videos, object masks, force/contact masks, projected motion-flow fields, projected stress-response maps, and ground-truth material parameters. By combining the rendering eficiency of 3D Gaussian Splatting with the physical interpretability of MPM, DeformFieldBench provides pixel-aligned physical supervision while keeping the inference-time input purely visual. We use this benchmark not merely as a dataset contribution, but to study a central modeling question: how should dense physical fields be used to improve feed-forward material-parameter inference?

We introduce a physical-field-bridged multi-view model to answer this question. Unlike direct video regression, which maps RGB observations to material parameters through a single opaque representation, our model predicts intermediate physical responses and uses them as evidence for parameter estimation. Given multi-view RGB deformation videos, it extracts static geometric cues and dynamic deformation cues, fuses them across views, and decomposes the representation through a physics bottleneck. It then predicts force/contact regions, projected motion-flow fields, and projected stress-response fields. These predictions are not auxiliary outputs only: they are encoded and fused back into the material regressor, so that parameter estimation is conditioned on where the object is loaded, how motion propagates, and where stress concentrates. Since stress is less directly observable than contact and motion, we adopt a staged design that first learns visually grounded contact and flow prediction, then infers stress response, and finally regresses material parameters from the predicted fields and bottleneck representation.

The resulting framework provides an amortized feed-forward alternative to online system identification. It predicts a constitutive type, continuous material parameters, and dense physicalresponse fields in a single test-time forward pass, without per-scene reconstruction, camera-pose input, online tuning, or simulation-in-the-loop optimization. Experiments on object-level generalization show that our method achieves lower median errors on all four continuous parameters than VLM-based estimators and direct supervised video-regression baselines. It also substantially improves dense physical-field prediction, especially stress-hotspot localization. When the predicted constitutive model and parameters are fed back into the simulator, our method produces dynamically plausible future deformation on unseen objects. Protocol-matched, held-out-interaction, input-robustness, cross-domain, and real-world evaluations further characterize when this fast amortized inference transfers and where domain gaps remain.

In summary, our contributions are three-fold:

• We construct DeformFieldBench, a 3DGS–MPM-based learning environment for feed-forward, simulation-ready material inference, with diverse objects, mechanically informative loading modes, multi-view RGB deformation videos, simulator-ready material labels, and dense physical-field supervision for force/contact, motion flow, and stress response.

• We propose a single-pass physical-field-bridged model that predicts dense contact, motion, and stress-response fields as intermediate evidence for constitutive-type classification and continuous material-parameter inference, avoiding per-scene reconstruction, camera-pose input, and simulation-in-the-loop optimization.

• We demonstrate improvements in median material-parameter accuracy, dense-field quality, object-level generalization, and same-simulator replay, and evaluate protocol-matched crossbenchmark transfer, unseen interactions, view and resolution sensitivity, lightweight domain adaptation, and quantitative real-world replay.

## 2 Related Work

Physics-aware Gaussian representations and dynamic scene modeling. Recent advances in 3D Gaussian representations have extended Gaussian primitives from static scene reconstruction to non-rigid dynamics and physically grounded modeling. Dynamic Gaussian manipulation methods study how Gaussian-based scene representations can support robotic interaction and non-rigid manipulation [21]. Generative Gaussian methods further extend this representation to 4D asset generation and dynamic content synthesis [22, 23]. These methods provide flexible visual representations for dynamic objects, but they do not directly target the recovery of simulator-ready material parameters from deformation observations.

Another line of work couples Gaussian representations with physical simulation. PhysGaussian and related physics-integrated systems connect 3D Gaussians with MPM-based solvers, making it possible to generate or simulate physically plausible deformable dynamics from Gaussian primitives [24, 25]. These systems show that Gaussian representations can be made compatible with continuum simulation, which motivates our use of a 3DGS–MPM pipeline for dataset construction. However, their main focus is on forward simulation, reconstruction, or physically grounded generation, whereas our benchmark targets feed-forward material-parameter inference from multi-view RGB deformation videos.

Several recent methods also incorporate semantic or multimodal physical priors into Gaussian representations. Property-aware Gaussian methods use large multimodal models to infer coarse physical attributes or semantic-level material information [9, 10]. More recent physical Gaussian models and real-video-to-physics pipelines explore feed-forward 4D synthesis and physics-aware demonstration generation [26, 27]. These methods suggest that high-level priors are useful for physical reasoning, but they often struggle to capture fine temporal cues such as rebound, oscillation, and localized stress concentration. Our work instead uses dense stress, flow, and contact fields as explicit intermediate supervision for material inference.

Inverse physics and simulation-ready material inference. System identification and inversephysics methods aim to recover physical parameters from observed motion, deformation, or reconstructed geometry. Representative approaches include unified constitutive modeling for inverse simulation [28], material-agnostic system identification from videos [1], learning neural constitutive laws from motion observations [29], and Gaussian-informed continuum modeling for physical property identification and simulation [2, 3]. These methods are physically grounded and have demonstrated strong reconstruction or simulation performance in controlled settings.

Despite their strengths, many existing inverse-physics pipelines depend on assumptions that limit their use as real-time perception modules. Some methods require known geometry, depth observations, or reconstructed object states; others rely on iterative test-time optimization, where a simulator is repeatedly rolled out and matched to observations. This per-scene optimization can be expensive when the goal is to quickly infer physical parameters for downstream robotic manipulation. In addition, many methods optimize global parameters through future-state matching, but do not explicitly supervise intermediate physical responses. Without field-level supervision, a learned model may rely on object appearance, texture, or semantic category shortcuts rather than material-discriminative deformation cues. Our method addresses this issue by predicting motion flow, force/contact regions, and stress distributions before classifying the constitutive family and regressing Young’s modulus, Poisson’s ratio, density, and yield stress.

Feed-forward physics inference. PIXIE predicts discrete material categories and continuous physical properties with supervised feed-forward learning from static multi-view appearance and geometry [30]. Our setting is complementary: it observes deformation videos under external interaction and explicitly predicts contact, motion-flow, and stress-response fields before material inference. This dynamic evidence is useful when objects with similar texture and geometry difer in stifness, plasticity, density, or internal structure, properties that static appearance alone may leave ambiguous.

Datasets and benchmarks for physical scene understanding. Datasets and simulation environments have played an important role in physical scene understanding. General-purpose environments such as Physion [14] and ThreeDWorld [15] provide standardized platforms for evaluating physical prediction and intuitive physical reasoning. However, these environments usually abstract away simulator-ready material parameters and internal physical response fields, which makes them less suitable for training models that infer numerical material properties.

Deformable-object benchmarks provide more relevant settings for manipulation and dynamics learning. SoftGym [16], Cloth3D [17], and DEDO [18] focus on control-oriented deformableobject tasks, garment dynamics, or dynamic environments with deformable objects. These datasets are valuable for learning interaction policies and deformation dynamics, but they typically do not pair visual observations with explicit Young’s modulus, Poisson’s ratio, density, yield stress, or dense stress-field annotations.

Real-world resources provide complementary signals. DOFS [19] and PokeFlex [20] contain real deformable-object observations, geometry, or force-related measurements. These datasets are important for grounding deformable-object research in real measurements, but their scale, parameter coverage, and dense field annotations are limited for training generalizable materialinference models. Recent simulation-aligned pipelines such as Vid2Sim [7] and PhysTwin [5] reconstruct geometry and physics from videos, but remain centered on per-scene reconstruction rather than large-scale benchmarking of feed-forward material inference.

DeformFieldBench is designed to complement these resources. It provides diverse object shapes, five mechanically informative loading modes, broad material-parameter ranges, and pixel-aligned physical fields for stress, flow, and force/contact regions. This enables systematic evaluation of whether models can infer simulation-ready material parameters from visual deformation evidence rather than from appearance alone.

## 3 DeformFieldBench: A Learning Environment for Material Inference

A central dificulty in feed-forward material-parameter inference is that the target variables are not directly observable from RGB appearance. Young’s modulus, density, Poisson’s ratio, and yield stress are expressed through how an object responds to external loading: where contact occurs, how deformation propagates, how the object recovers after release, and where stress concentrates. Existing datasets usually provide visual observations or reconstructed states, but rarely expose these intermediate physical responses in a form that can supervise learning. As a result, video-based models are often forced to learn a black-box mapping from appearance and motion to global material parameters, which makes them vulnerable to object-category or texture shortcuts.

We construct DeformFieldBench to study a diferent formulation: material inference as learning from dense physical responses. The benchmark is built on a 3DGS–MPM simulation pipeline and contains 5,000 simulations over 100 everyday object shapes, five loading modes (press, stretch, shear, bend, and drop), multiple material categories (jelly, metal, and plasticine), and broad material-parameter ranges. Each sample includes synchronized three-view RGB deformation videos, object masks, force/contact masks, projected motion-flow fields, projected stress-response maps, action labels, and ground-truth material parameters

$$
p = [ E , \nu , \rho , \sigma _ { y } ] ,\tag{1}
$$

where E is Young’s modulus, ν is Poisson’s ratio, $\rho$ is density, and $\sigma _ { y }$ is yield stress. During inference, models are given only multi-view RGB videos; dense physical fields are used as intermediate learning targets. This setting allows us to evaluate whether a model can infer simulator-ready material parameters from visual deformation while being guided by physically meaningful evidence. Figure 1 compares DeformFieldBench with related deformable-object and physical-reasoning datasets, highlighting its joint support for diverse object geometries, multiple mechanically informative loading modes, simulator-ready material labels, and dense projected physical-field supervision.

The current benchmark assumes one homogeneous constitutive description per object. Its dense fields provide spatial evidence for global system identification; they are not spatially varying material labels. The original videos are rendered at 512 × 512 resolution from three stationary canonical cameras distributed approximately horizontally around each normalized scene. Camera extrinsics are not provided to the model.

## 3.1 Benchmark Design

DeformFieldBench is built on a 3DGS–MPM simulation pipeline. Starting from a 3D Gaussian object representation, we convert visible Gaussian primitives into volumetric particles for MPM simulation, assign material parameters sampled from predefined ranges, and evolve the particles under external loading. The deformed particle states are then mapped back to Gaussian primitives and rendered into multi-view RGB videos and aligned physical-field videos. This design combines the rendering eficiency and appearance diversity of 3D Gaussian Splatting with the physical interpretability of MPM-based deformable simulation. A key design principle is to decouple visual appearance from material properties. By varying object geometry and surface style independently from $E , \nu , \rho ,$ and $\sigma _ { y { \mathrm { : } } }$ , the benchmark discourages models from predicting materials through category or texture shortcuts. Instead, models must rely on deformation behavior. To make such behavior informative, we include five loading modes: press, stretch, shear, bend, and drop. These actions expose complementary physical cues, including local contact, compressibility, stifness, tangential deformation, flexural recovery, inertia, and impact response. For actuator-driven actions, object-aware loading policies determine contact regions and trajectories from object geometry, and the simulation continues after release to capture rebound, oscillation, and residual deformation.

## 3.2 Dense Physical Field Annotations

The main distinction of DeformFieldBench is that RGB deformation videos are paired with dense, pixel-aligned physical fields. These fields are not privileged inputs at test time; they supervise the intermediate physical responses that should mediate material inference. Force/contact masks identify the regions afected by external actuation or contact, specifying where deformation is caused. Projected motion-flow fields are rendered from particle displacements between consecutive frames and describe how deformation propagates over the visible surface. Projected stress-response maps are computed from MPM stress and rendered into image space, revealing force concentration and yielding-related responses that may not be directly visible in RGB motion.

We export these annotations in image space rather than requiring full 3D particle trajectories or volumetric stress tensors. This keeps the learning problem aligned with visual perception, reduces storage and training cost, and still preserves the physical signals most relevant to material inference. The resulting supervision allows a model to learn not only what parameter values to predict, but also which physical evidence supports the prediction.

Eficient multi-modal field export. A practical challenge in building dense physical supervision is that full 3D particle trajectories and per-particle stress tensors are expensive to store and dificult to use for video-based learning. We therefore export physical responses directly as image-space fields through Gaussian rendering. Specifically, particle motion is projected and encoded into rendered flow maps, while stress magnitude is mapped into rendered stress-response images. This design preserves dense, high-frequency physical signals while keeping the dataset lightweight and naturally aligned with RGB observations. It also makes DeformFieldBench scalable: compared with prior deformable physical-reasoning resources, our benchmark jointly supports broader object diversity, more loading types, wider parameter ranges, and stronger pixel-aligned physical supervision [3, 20, 27, 28].

## 3.3 3DGS–MPM Simulation Procedure

## 3.3.1 Input Representation

The input object is represented by a set of 3D Gaussian primitives

$$
\mathcal { G } = \{ ( \mathbf { x } _ { p } ^ { 0 } , \pmb { \Sigma } _ { p } ^ { 0 } , \alpha _ { p } , \mathbf { c } _ { p } ) \} _ { p = 1 } ^ { N } ,\tag{2}
$$

where $\mathbf { x } _ { p } ^ { 0 } \in \mathbb { R } ^ { 3 }$ is the initial Gaussian center, $\Sigma _ { p } ^ { 0 }$ is its covariance, $\alpha _ { p }$ is opacity, and $\mathbf { c } _ { p }$ is the rendered color. Low-opacity Gaussians are discarded:

$$
\alpha _ { p } > \alpha _ { \mathrm { m i n } } .\tag{3}
$$

The remaining Gaussian centers are normalized into the MPM simulation domain:

$$
\mathbf { x } _ { p } = \mathbf { 1 } + s \mathbf { R } ( \mathbf { x } _ { p } ^ { 0 } - \bar { \mathbf { x } } ) ,\tag{4}
$$

where R is the preprocessing rotation, s is a scale factor, x¯ is the object center, and $\mathbf { 1 } = ( 1 , 1 , 1 ) ^ { T }$ shifts the object near the center of the MPM box.

Each Gaussian is treated as an MPM particle with state

$$
( { \bf x } _ { p } , { \bf v } _ { p } , { \bf F } _ { p } , { \bf F } _ { p } ^ { * } , { \bf C } _ { p } , V _ { p } , m _ { p } ) ,\tag{5}
$$

where $\mathbf { v } _ { p }$ is velocity, $\mathbf { F } _ { p }$ is the elastic deformation gradient, $\mathbf { F } _ { p } ^ { * }$ is the trial deformation gradient, $\mathbf { C } _ { p }$ is the APIC afine velocity matrix, $V _ { p }$ is particle volume, and $m _ { p }$ is particle mass.

## 3.3.2 Material Parameters

The material parameters used in simulation are

$$
E , \quad \nu , \quad \rho , \quad \sigma _ { y } , \quad \mathbf { g } ,\tag{6}
$$

where E is Young’s modulus, $\nu$ is Poisson’s ratio, $\rho$ is density, $\sigma _ { y }$ is yield stress, and $\mathbf { g }$ is gravity. The particle mass is

$$
m _ { p } = \rho V _ { p } .\tag{7}
$$

The Lamé parameters are computed from E and ν:

$$
\mu = \frac E { 2 ( 1 + \nu ) } , \qquad \lambda = \frac { E \nu } { ( 1 + \nu ) ( 1 - 2 \nu ) } .\tag{8}
$$

Here $\mu$ controls shear resistance, while λ controls volumetric resistance.

The material type determines the return mapping and stress law. In our setting, the main material behaviors are elastic jelly-like materials, metal-like von Mises plasticity, and plasticine-like plasticity with softening.

## 3.3.3 Stress From Deformation

At each substep, the solver first updates stress from the trial deformation gradient $\mathbf { F } _ { p } ^ { * } .$ . A materialdependent projection gives

$$
\mathbf { F } _ { p } = \Pi ( \mathbf { F } _ { p } ^ { * } ) ,\tag{9}
$$

where Π is identity for elastic materials and a plastic return mapping for plastic materials.

For elastic materials, the solver uses a corotated stress model. Let

$$
\mathbf { F } _ { p } = \mathbf { U } _ { p } \mathbf { S } _ { p } \mathbf { V } _ { p } ^ { T } , \qquad \mathbf { R } _ { p } = \mathbf { U } _ { p } \mathbf { V } _ { p } ^ { T } , \qquad J _ { p } = \operatorname* { d e t } ( \mathbf { F } _ { p } ) .\tag{10}
$$

Here $\mathbf { U } _ { p } , \mathbf { S } _ { p } , \mathbf { V } _ { p }$ come from singular value decomposition, $\mathbf { R } _ { p }$ is the local rotation, and $J _ { p }$ is the volume change. The Kirchhof stress is

$$
\boldsymbol { \tau } _ { p } = 2 \mu ( \mathbf { F } _ { p } - \mathbf { R } _ { p } ) \mathbf { F } _ { p } ^ { T } + \lambda J _ { p } ( J _ { p } - 1 ) \mathbf { I } .\tag{11}
$$

For plastic materials, the singular values of $\mathbf { F } _ { p } ^ { * }$ are mapped to logarithmic strain

$$
\begin{array} { r } { \epsilon _ { p } = \log ( { s _ { p } } ) , } \end{array}\tag{12}
$$

where $\boldsymbol { s } _ { p }$ contains the singular values. The deviatoric strain is

$$
\hat { \epsilon } _ { p } = \epsilon _ { p } - \frac { 1 } { 3 } \mathrm { t r } ( \epsilon _ { p } ) { \bf 1 } .\tag{13}
$$

Plastic yielding occurs when the deviatoric part exceeds the yield stress. A simple return step is

$$
\Delta \gamma = \lVert \hat { \epsilon } _ { p } \rVert - \frac { \sigma _ { y } } { 2 \mu } ,\tag{14}
$$

$$
\epsilon _ { p } \gets \epsilon _ { p } - \frac { \Delta \gamma } { \lVert \hat { \epsilon } _ { p } \rVert } \hat { \epsilon } _ { p } .\tag{15}
$$

Thus, E and ν determine stifness through $\mu , \lambda .$ , while $\sigma _ { y }$ controls when plastic deformation starts.

## 3.3.4 One MPM Substep

Let $\Delta x$ be the grid spacing and $\Delta t$ be the substep size. Each particle transfers mass and momentum to nearby grid nodes. Let $w _ { i p }$ be the interpolation weight between particle $p$ and grid node $i ,$ and $\nabla w _ { i p }$ its spatial gradient.

Particle to grid.

$$
m _ { i } = \sum _ { p } w _ { i p } m _ { p } ,\tag{16}
$$

$$
\mathbf { p } _ { i } = \sum _ { p } w _ { i p } m _ { p } \left( \mathbf { v } _ { p } + \mathbf { C } _ { p } ( \mathbf { x } _ { i } - \mathbf { x } _ { p } ) \right) - \Delta t \sum _ { p } V _ { p } \pmb { \tau } _ { p } \nabla w _ { i p } .\tag{17}
$$

The first term transfers particle momentum to the grid. The second term applies elastic internal force from stress.

Grid update.

$$
\mathbf { u } _ { i } = \frac { \mathbf { p } _ { i } } { m _ { i } } + \Delta t \mathbf { g } ,\tag{18}
$$

where $\mathbf { u } _ { i }$ is the grid velocity.

Contact and actuation. For a moving cuboid actuator with center $\mathbf { q } ( t )$ , half-size $\mathbf { a } ,$ and velocity ${ \bf u } _ { b } ,$ , grid nodes inside the cuboid satisfy

$$
| { \bf x } _ { i } - { \bf q } ( t ) | < { \bf a } .\tag{19}
$$

Their velocity is overwritten by

$$
\mathbf { u } _ { i }  \mathbf { u } _ { b } .\tag{20}
$$

The actuator moves as

$$
\mathbf { q } ( t + \Delta t ) = \mathbf { q } ( t ) + \Delta t \mathbf { u } _ { b } .\tag{21}
$$

This is how press, shear, stretch, and bend actions impose external motion. For drop, gravity and ground contact dominate the motion.

Grid to particle. The updated grid velocities are interpolated back:

$$
\mathbf { v } _ { p } ^ { n + 1 } = \sum _ { i } w _ { i p } \mathbf { u } _ { i } ,\tag{22}
$$

$$
\nabla \mathbf { v } _ { p } = \sum _ { i } \mathbf { u } _ { i } ( \nabla w _ { i p } ) ^ { T } ,\tag{23}
$$

$$
\mathbf { F } _ { p } ^ { * , n + 1 } = ( \mathbf { I } + \Delta t \nabla \mathbf { v } _ { p } ) \mathbf { F } _ { p } ^ { n } ,\tag{24}
$$

$$
\begin{array} { r } { \mathbf { x } _ { p } ^ { n + 1 } = \mathbf { x } _ { p } ^ { n } + \Delta t \mathbf { v } _ { p } ^ { n + 1 } . } \end{array}\tag{25}
$$

## 3.3.5 Stress Output

The solver stores Kirchhof stress $\tau _ { p }$ . For output, it is converted to Cauchy stress:

$$
{ \pmb \sigma } _ { p } = \frac { 1 } { J _ { p } } { \pmb \tau } _ { p } , \qquad J _ { p } = \operatorname* { d e t } ( \mathbf { F } _ { p } ^ { * } ) .\tag{26}
$$

The scalar stress visualization uses von Mises stress:

$$
\sigma _ { \mathrm { v m } } = \sqrt { \frac { 1 } { 2 } \left[ ( \sigma _ { 1 1 } - \sigma _ { 2 2 } ) ^ { 2 } + ( \sigma _ { 2 2 } - \sigma _ { 3 3 } ) ^ { 2 } + ( \sigma _ { 3 3 } - \sigma _ { 1 1 } ) ^ { 2 } + 6 ( \sigma _ { 1 2 } ^ { 2 } + \sigma _ { 2 3 } ^ { 2 } + \sigma _ { 1 3 } ^ { 2 } ) \right] } .\tag{27}
$$

This scalar is normalized to $t _ { p } \in [ 0 , 1 ]$ and used to color the Gaussian:

$$
\mathbf { c } _ { p } ^ { \mathrm { s t r e s s } } = \beta \mathbf { c } _ { p } + ( 1 - \beta ) \operatorname { C o l o r M a p } ( t _ { p } ) .\tag{28}
$$

## 3.3.6 Flow Output

For each rendered view, the current particle position is projected to image coordinates:

$$
\mathbf { u } _ { p } ( t ) = \Pi ( \mathbf { x } _ { p } ( t ) ) = ( u _ { p } ( t ) , v _ { p } ( t ) ) .\tag{29}
$$

The optical-flow-like signal is the screen displacement between two output frames:

$$
\Delta { \mathbf { u } } _ { p } ( t ) = { \mathbf { u } } _ { p } ( t ) - { \mathbf { u } } _ { p } ( t - \Delta t _ { \mathrm { o u t } } ) .\tag{30}
$$

Its direction and magnitude are

$$
\theta _ { p } = \mathrm { a t a n 2 } ( \Delta v _ { p } , \Delta u _ { p } ) , \qquad r _ { p } = \| \Delta \mathbf { u } _ { p } \| _ { 2 } .\tag{31}
$$

The direction $\theta _ { p }$ is encoded as hue, and the magnitude $r _ { p }$ is encoded as brightness/opacity before Gaussian rasterization.

## 3.3.7 Force and Contact Mask Output

The force mask is defined from the active moving cuboid actuators. A particle is marked as being in the force/contact region if

$$
| { \bf x } _ { p } ( t ) - { \bf q } ( t ) | < { \bf a } .\tag{32}
$$

Thus the binary force mask is

$$
M _ { p } ^ { \mathrm { f o r c e } } ( t ) = [ | \mathbf { x } _ { p } ( t ) - \mathbf { q } ( t ) | < \mathbf { a } ] .\tag{33}
$$

For single-channel mask rendering, only selected particles are visible:

$$
\alpha _ { p } ^ { \mathrm { f o r c e } } = M _ { p } ^ { \mathrm { f o r c e } } \alpha _ { p } , \qquad { \bf c } _ { p } ^ { \mathrm { f o r c e } } = ( 1 , 1 , 1 ) .\tag{34}
$$

The object mask uses all visible object Gaussians:

$$
\alpha _ { p } ^ { \mathrm { { o b j e c t } } } = \alpha _ { p } , \qquad { \bf c } _ { p } ^ { \mathrm { { o b j e c t } } } = ( 1 , 1 , 1 ) .\tag{35}
$$

## 3.3.8 Rendering Back to 3DGS

After each selected simulation frame, the MPM particle positions are transformed back to the original 3DGS coordinate system:

$$
\begin{array} { r } { { \bf x } _ { p } ^ { \mathrm { w o r l d } } ( t ) = \bar { \bf x } + s ^ { - 1 } { \bf R } ^ { - 1 } ( { \bf x } _ { p } ( t ) - { \bf 1 } ) . } \end{array}\tag{36}
$$

The Gaussian covariance can be updated by deformation:

$$
\begin{array} { r } { \Sigma _ { p } ( t ) = \mathbf { F } _ { p } ( t ) \Sigma _ { p } ^ { 0 } \mathbf { F } _ { p } ( t ) ^ { T } . } \end{array}\tag{37}
$$

The updated Gaussian centers, covariances, colors, and opacities are then rendered to produce RGB frames, stress frames, flow frames, force masks, and object masks.

## 3.4 Tasks and Evaluation Protocol

The primary task is to infer a constitutive type cˆ and continuous material parameters from synchronized three-view RGB videos: $\hat { p } = [ \hat { E } , \hat { \nu } , \hat { \rho } , \hat { \sigma } _ { y } ]$ . We evaluate $E , \rho ,$ and $\sigma _ { y }$ in log space due to their large dynamic ranges, and evaluate ν in raw space. Besides per-parameter errors, we report sample-wise parameter error to measure whether a method predicts a coherent parameter set for each simulation sample.

We use an object-level split, with 90 objects for training and 10 unseen objects for testing, to evaluate generalization beyond memorized shapes or appearances. We also evaluate dense field prediction: contact-region quality for force masks, coverage and velocity-aware quality for motion flow, and hotspot localization for stress response. Finally, we perform simulation replay by feeding the predicted constitutive type and parameters back into the simulator and comparing the rendered deformation with the ground truth. This replay protocol is important because diferent parameter combinations may induce similar visible motion; a useful estimate should therefore be not only numerically accurate, but also dynamically plausible. Together, parameter accuracy, field quality, object-level generalization, and replay performance provide the basis for evaluating the physical-field-bridged model introduced next. Details are reported in Section 5.

## 3.5 Dataset Release and Usage Documentation

Dataset name. DeformFieldBench: Physical-Field Material Parameter Dataset with 5,000 3DGS– MPM simulations.

Hosting and access. The dataset artifacts are available on Hugging Face. Source code, documentation, and release updates are available in the DeformFieldBench GitHub repository. The released files include rendered multi-view RGB deformation videos, projected physical-response fields, masks, action labels, material parameters, and train/test split metadata.

Dataset purpose. DeformFieldBench is designed for simulation-ready material-parameter inference of deformable objects from multi-view RGB deformation videos. It supports both global parameter regression and dense physical-response prediction.

Dataset scale. The dataset contains 5,000 simulation samples over 100 object shapes. Each object is simulated under five loading modes: press, stretch, shear, bend, and drop. Each sample contains 512×512 observations from three stationary canonical cameras distributed approximately horizontally around the normalized scene, together with aligned physical-field annotations.

Modalities and annotations. Each sample contains: (1) multi-view RGB deformation videos; (2) object masks; (3) force/contact masks; (4) projected motion-flow fields; (5) projected stressresponse maps; (6) action labels; (7) a ground-truth constitutive family; and (8) continuous material parameters, including Young’s modulus, Poisson’s ratio, density, and yield stress.

Generation pipeline. The dataset is generated using a 3DGS–MPM simulation pipeline. Starting from 3D Gaussian object representations, we densify the object volume into MPM particles, assign material parameters, apply controlled mechanical loading, and render RGB videos and image-space physical-response fields through the Gaussian rendering pipeline.

Splits. We use an object-level split to evaluate generalization. Specifically, 90 objects are used for training and 10 completely unseen objects are reserved for testing. This prevents identical object shapes from appearing in both training and test sets.

Evaluation tasks. The benchmark supports: (1) constitutive-type classification and materialparameter regression; (2) dense stress-response prediction; (3) projected motion-flow prediction; (4) force/contact mask prediction; (5) simulation replay evaluation using the predicted constitutive type and material parameters.

Known limitations. The dataset is synthetic and generated by one 3DGS–MPM pipeline. It assumes a single homogeneous material description per object and uses a canonical three-camera layout. The stress and flow annotations are image-space projections rather than full volumetric fields, and normalized stress maps do not provide absolute stress values. High-stifness metallic materials may therefore be dificult to distinguish when constant-velocity loading induces saturated permanent deformation. Cross-simulator results further show sensitivity to damping, filling, integration, contact, and boundary-condition implementations. Future versions could include heterogeneous materials, randomized camera and simulator configurations, calibrated external force, per-particle absolute stress, and tactile or force-torque signals.

License and usage. Users should refer to the Hugging Face dataset page for the license, terms of use, and download instructions.

## 4 Physical-Field-Bridged Material Inference

Our feed-forward model maps multi-view RGB deformation videos to simulation-ready material parameters through explicit physical evidence. Direct regression can entangle material cues with category and texture, while fields trained only as auxiliary outputs need not influence the final prediction. As shown in Fig. 2, our model instead fuses frame-, patch-, and multi-scale features, factorizes them through a physics bottleneck for contact, deformation, and stress, and supervises the corresponding dense fields. Encodings of the predicted fields are then fused back into the bottleneck, conditioning constitutive and parameter inference on where loading occurs, how motion propagates, and where stress concentrates.

Given synchronized RGB videos from V views, $X = \{ I _ { v , t } \mid v = 1 , \ldots , V , t = 1 , \ldots , T \}$ , our model predicts dense contact, motion, and stress fields, $\hat { \mathcal { V } } = \{ \hat { M } , \hat { U } , \hat { S } \}$ , a discrete constitutive type ${ \hat { c } } ,$ and global material parameters $\hat { p } = [ \hat { E } , \hat { \nu } , \hat { \rho } , \hat { \sigma } _ { y } ]$ . Here $\hat { M }$ denotes the force/contact mask, $\hat { U }$ denotes projected motion flow, and $\hat { S }$ denotes the projected stress-response map. Only RGB videos are required at inference time; dense fields supervise the internal representation during training.

![](images/94def5efa799a2b41a629fa73939daddc008d58a7d6fb498e7b0748aca4a239f.jpg)  
Figure 2. Overview of the proposed model. Multi-view RGB deformation videos are encoded into static and dynamic features, decomposed through a physics bottleneck, and decoded into dense motion-flow, force/contact, and stress fields. The predicted fields are encoded and fused with the bottleneck for material inference. The implemented constitutive-type classifier runs in parallel with the illustrated continuous parameter regressor and is omitted from the diagram for clarity.

## 4.1 Multi-view Visual Encoding

Material inference requires static geometry, which afects load distribution, and dynamic cues such as deformation, rebound, oscillation, and residual displacement. A pretrained DINOv2 backbone extracts frame features and patch tokens; a temporal adapter aggregates dynamics within each view, and a cross-view transformer produces a global multi-view condition. Patch tokens and multi-scale RGB features remain as local conditions for dense decoding, preserving details that global features alone would discard.

For each frame in each camera view, we use a pretrained DINOv2 backbone to extract framelevel visual features. Let ${ \boldsymbol { x } } _ { v , t }$ denote the RGB frame at view v and time t. The frame encoder produces

$$
f _ { v , t } = \Phi _ { \mathrm { e n c } } ( x _ { v , t } ) ,\tag{38}
$$

where $\Phi _ { \mathrm { e n c } }$ denotes the frozen visual backbone. In addition to the global frame token, patch-level tokens are also retained and later used as local spatial conditions for dense field prediction.

To model temporal deformation within each view, the frame features are passed through a lightweight Transformer temporal adapter with learnable temporal embeddings:

$$
g _ { v } = \mathrm { P o o l } _ { t } \left( \mathrm { T r a n s } _ { t } ( f _ { v , 1 : T } + e _ { 1 : T } ) \right) ,\tag{39}
$$

where $g _ { v }$ is the view-level dynamic representation. The resulting view features are further fused by a cross-view Transformer. Before fusion, a learnable view embedding is added to each view representation:

$$
z = \Psi _ { \mathrm { v i e w } } \left( \{ g _ { v } + a _ { v } \} _ { v = 1 } ^ { V } \right) .\tag{40}
$$

Here, $z$ is the fused multi-view representation, which serves as the global visual condition for subsequent physical reasoning. Meanwhile, multi-scale RGB features and patch-level DINO features are extracted to provide local geometry and deformation cues for dense field decoding.

## 4.2 Physics Bottleneck

Instead of sending the fused video representation directly to all output heads, we introduce a physics bottleneck that decomposes it into contact-, deformation-, and stress-related latent factors:

$$
h _ { c } , h _ { d } , h _ { s } = \Phi _ { \mathrm { p h y s } } ( z ) , \qquad z _ { \mathrm { p h y s } } = [ h _ { c } ; h _ { d } ; h _ { s } ] ,\tag{41}
$$

where $z$ is the fused multi-view visual feature. This bottleneck reflects the structure of the inference problem. Contact describes where external loading enters the object, deformation describes how the object moves under that loading, and stress describes the internal response that links visible motion to material parameters.

The decomposition organizes visual evidence for field-specific heads instead of exposing all outputs to one opaque representation: contact marks where loading enters, deformation describes the resulting motion, and stress connects that response to material behavior.

## 4.3 Dense Physical Field Prediction

A shared spatiotemporal decoder combines the global condition, physical latents, frame and patch features, and multi-scale spatial features; task-specific heads predict force/contact, motion flow, and stress response. The heads follow an observability hierarchy: contact and flow are grounded in actuator location and visible deformation, whereas stress is less directly observable and is therefore conditioned on both predicted responses. This sequential dependency ties stress to loading and deformation propagation rather than surface appearance alone. Object masks suppress background responses, especially for sparse stress and flow signals.

## 4.3.1 Dense Physical Field Decoder

Dense physical fields are predicted by a shared conditional spatiotemporal decoder. The decoder combines global multi-view information, frame-level features, temporal embeddings, physical latents, patch-level features, and multi-scale spatial RGB conditions. These conditions are first projected into a low-resolution seed feature map, then refined by temporal mixing blocks and upsampling layers to produce a shared dense feature tensor:

$$
G = D _ { \mathrm { s h a r e d } } ( z , f _ { v , t } , z _ { \mathrm { p h y s } } , c _ { v , t } , p _ { v , t } ) ,\tag{42}
$$

where $c _ { v , t }$ denotes multi-scale spatial RGB conditions and $p _ { v , t }$ denotes patch-token conditions.   
The shared tensor G is then used by task-adaptive heads to predict flow, force, and stress.

Each task head modulates the shared feature using its corresponding physical latent:

$$
\hat { Y } _ { k } = H _ { k } ( G , h _ { k } ) , \qquad k \in \{ f , m , s \} ,\tag{43}
$$

where $\hat { Y } _ { f } , \hat { Y } _ { m } .$ , and $\hat { Y } _ { s }$ denote the predicted flow, force/contact, and stress fields, respectively. This design allows the three dense prediction tasks to share common spatiotemporal features while preserving task-specific behavior.

## 4.3.2 Sequential Stress Refinement

Stress is more indirect and dificult to infer from RGB observations than motion or contact. Therefore, we impose a sequential physical bias in the stress branch. The model first predicts flow and force/contact fields, then feeds them back into the stress branch through a fusion module:

$$
G _ { s } = F _ { s } \left( [ G ; { \hat { Y } } _ { f } ; { \hat { Y } } _ { m } ] \right) , \qquad { \hat { Y } } _ { s } = H _ { s } ( G _ { s } , h _ { s } ) .\tag{44}
$$

This design encourages stress prediction to be grounded in visible motion and contact evidence. In addition, spatial enhancement blocks are applied to refine flow and stress features. When object masks are available, they are aligned to the prediction resolution and used to suppress background responses, making the predicted physical fields focus on object regions.

## 4.4 Field-conditioned Constitutive and Parameter Prediction

Lightweight spatiotemporal encoders summarize the predicted fields, which are fused with the physics bottleneck for constitutive-type classification and continuous-parameter regression. Thus, the fields are regressor inputs rather than auxiliary outputs alone. We regress in the transformed target space

$$
\begin{array} { r } { \tilde { p } = [ \mathrm { l o g } ( 1 + E ) , \nu , \mathrm { l o g } ( 1 + \rho ) , \mathrm { l o g } ( 1 + \sigma _ { y } ) ] . } \end{array}\tag{45}
$$

The logarithmic transform handles large dynamic ranges, while Poisson’s ratio remains in raw space. During replay, cˆ selects the elastic, von Mises, or plasticity-with-softening MPM law

configured by $\hat { p } ;$ an incorrect type is never replaced by ground truth. Yield stress is supervised and evaluated only for yielding ground-truth models and is ignored when the predicted replay model is non-yielding.

Instead of regressing material parameters only from the global visual feature, we further encode the predicted dense physical fields with compact 3D convolutional encoders:

$$
u _ { s } = \Phi _ { s } ( \hat { Y } _ { s } ) , \qquad u _ { f } = \Phi _ { f } ( \hat { Y } _ { f } ) , \qquad u _ { m } = \Phi _ { m } ( \hat { Y } _ { m } ) .\tag{46}
$$

The field summaries are fused with the physics bottleneck to obtain a field-aware parameter representation:

$$
z _ { p } = \Phi _ { \mathrm { f u s e } } \big ( \big [ z _ { \mathrm { p h y s } } ; u _ { s } ; u _ { f } ; u _ { m } \big ] \big ) .\tag{47}
$$

An MLP classification head predicts one of the three constitutive families—elasticjelly-like behavior, metal-like von Mises plasticity, or plasticine-like plasticity with softening—and a regression head predicts transformed material parameters:

$$
\hat { c } = C ( z _ { p } ) , \qquad \hat { \bar { p } } = R ( z _ { p } ) , \qquad \tilde { p } = [ \log ( 1 + E ) , \nu , \log ( 1 + \rho ) , \log ( 1 + \sigma _ { y } ) ] .\tag{48}
$$

Large-range quantities are handled in log space, while Poisson’s ratio is constrained to a physically valid range during prediction. During replay, cˆ selects the MPM stress law and return mapping, and $\hat { p }$ configures it. Ground-truth types are never substituted for predicted types in reported replay metrics.

## 4.5 Training Objective

The model is trained with a multi-task objective that combines constitutive classification, parameter regression, dense field supervision, and physics-consistency regularization:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { r e g } } + \lambda _ { c } \mathcal { L } _ { \mathrm { c l s } } + \lambda _ { s } \mathcal { L } _ { \mathrm { s t r e s s } } + \lambda _ { f } \mathcal { L } _ { \mathrm { f l o w } } + \lambda _ { m } \mathcal { L } _ { \mathrm { f o r c e } } + \lambda _ { p } \mathcal { L } _ { \mathrm { p h y s } } .\tag{49}
$$

Here $\mathcal { L } _ { \mathrm { c l s } }$ is cross-entropy over the three constitutive families. The regression loss is computed in the transformed parameter space. Invalid or undefined dimensions are masked: yield stress is supervised only for samples whose ground-truth law includes yielding, and purely elastic samples are excluded from yield-stress parameter-error statistics. During replay, yield stress is ignored if the predicted type is non-yielding; if an elastic sample is misclassified as yielding, the predicted yield stress is used and the resulting replay error remains included. For dense fields, stress and flow are supervised with foreground-weighted reconstruction losses and edge-consistency regularization. Force prediction is supervised using a combination of binary cross-entropy and Dice loss.

The physics-consistency term encourages agreement among flow, force, and stress responses. It is defined as a sum of three weak structural constraints over the predicted dense fields:

$$
\mathcal { L } _ { \mathrm { p h y s } } = \mathcal { L } _ { s f } + \mathcal { L } _ { f s } + \mathcal { L } _ { f f } ,\tag{50}
$$

where $\mathcal { L } _ { s f }$ encourages stress and flow fields to be positively correlated, while $\mathcal { L } _ { f s }$ and $\mathcal { L } _ { f f }$ encourage stress and flow responses to be stronger in force/contact regions than in non-contact regions.

Specifically, stress and flow maps are first converted to magnitude maps and normalized with a per-sample z-score. We then penalize low stress-flow correlation with a softplus margin loss:

$$
\mathcal { L } _ { s f } = \mathrm { s o f t p l u s } ( m _ { s f } - \mathrm { c o r r } ( \hat { S } , \hat { F } ) ) .\tag{51}
$$

For force-aware consistency, the force map is converted into a soft contact weight map, and we impose a contrastive margin between the average field magnitude inside and outside the contact region:

$$
\mathcal { L } _ { f s } = \mathrm { s o f t p l u s } \left( m _ { f s } - ( \mu _ { \mathrm { i n } } ( \hat { S } ) - \mu _ { \mathrm { o u t } } ( \hat { S } ) ) \right) ,\tag{52}
$$

$$
\mathcal { L } _ { f f } = \mathrm { s o f t p l u s } \left( m _ { f f } - \bigl ( \mu _ { \mathrm { i n } } ( \hat { F } ) - \mu _ { \mathrm { o u t } } ( \hat { F } ) \bigr ) \right) .\tag{53}
$$

## 4.5.1 Target-space Transformation and Validity Masking

We do not regress the raw physical values directly. Instead, we transform the target parameters as

$$
\tilde { \mathbf { p } } = [ \mathrm { l o g } ( 1 + E ) , \nu , \mathrm { l o g } ( 1 + \rho ) , \mathrm { l o g } ( 1 + \sigma _ { y } ) ] .\tag{54}
$$

This transformation compresses large dynamic ranges and improves optimization stability. We further construct a per-dimension validity mask so that only physically meaningful targets participate in the loss. Specifically, we require $E > 0 , \rho > 0 .$ , and $\nu \in ( 0 , 0 . 5 ]$ ; yield stress additionally requires $\sigma _ { y } > 0$ and a yielding ground-truth constitutive family. The transformed targets are then normalized using training-set statistics before regression loss is applied.

## 4.6 Staged Optimization

To reduce gradient competition among tasks, we adopt a staged training schedule. The model first learns flow and force prediction, which are more directly observable from RGB videos. It then trains the stress branch conditioned on the learned flow and force fields. After that, the full dense-field pipeline is optimized jointly. Finally, the dense-field components are frozen and the constitutive classification and parameter regression branches are refined using the predicted physical responses.

This training order follows the physical dependency encoded in the architecture: contact and motion provide low-level evidence, stress is inferred from these responses, and the constitutive family and continuous parameters are estimated from the resulting physical state.

## 5 Experimental Setup and Evaluation Protocol

## 5.1 Data Organization

Each training sample contains RGB, stress, flow, force-mask, and object-mask videos, together with action labels and four-dimensional material parameters. The dataset is organized as aligned multiview spatiotemporal tensors, ensuring eficient joint training for field prediction and parameter inference.

## 5.2 Baselines

We compare with (1) a VLM-based estimator, motivated by multimodal and zero-shot physical reasoning [9, 10, 31], that predicts material and motion types plus four parameters from ordered frames; and (2) a supervised video baseline with the same DINO backbone and multi-view fusion but no physics bottleneck or field-conditioned regression. The latter predicts fields and parameters from a shared representation, isolating the contribution of bridging through predicted fields.

## 5.3 Metrics Used in Tables

Material parameters have diferent units and dynamic ranges. We therefore evaluate $E ,$ density $\rho ,$ and yield stress $\sigma _ { y }$ in $\log ( 1 + x )$ space, and evaluate Poisson’s ratio ν in raw space. For each parameter, we report median absolute error (MedAE), which is more robust than mean error under heavy-tailed failures. We also report sample-wise parameter error (SPE) to measure the quality of the whole predicted parameter set for each simulation sample. For parameter $k ,$ we normalize its absolute error by the central 80% ground-truth range $r _ { k }$ , obtained after removing the lowest and highest 10% ground-truth values:

$$
\mathrm { S P E } ^ { ( i ) } = \frac { 1 } { \lvert \mathcal { K } _ { i } \rvert } \sum _ { k \in \mathcal { K } _ { i } } \frac { \lvert \hat { y } _ { i , k } - y _ { i , k } \rvert } { r _ { k } } ,\tag{55}
$$

where $\kappa _ { i }$ contains valid parameters for sample i. Yield stress is excluded for purely elastic ground-truth samples. Unfiltered metrics use all test samples. Top-20% filtered metrics remove the 20% samples with the largest SPE for each method. This filtering is used because the dataset contains metal objects whose stifness, especially $E ,$ is visually ambiguous and causes large errors for all methods. The filtered scores describe typical deformable-object inference, while the unfiltered scores remain the primary evidence of failure sensitivity. Because errors for $E , \rho ,$ and $\sigma _ { y }$ are measured after a log(1 + x) transform, a log-space MedAE such as 1.521 still represents substantial absolute uncertainty and should not be read as precise parameter recovery. The replay evaluation complements, rather than replaces, this parameter-space limitation.

For fields, stress is evaluated mainly by hotspot recall because stress localization is more important than exact full-field reconstruction. Let $S$ and $\hat { S }$ be the ground-truth and predicted stress magnitude maps inside the object mask. We define the hotspot region as the top-response region,

$$
\Omega _ { S } = \{ x \mid S ( x ) \geq q _ { \tau _ { S } } ( S ) \} , \quad \hat { \Omega } _ { S } = \{ x \mid \hat { S } ( x ) \geq q _ { \tau _ { S } } ( \hat { S } ) \} ,\tag{56}
$$

where $q _ { \tau _ { S } }$ denotes a high-percentile threshold. Hotspot recall is

$$
\mathrm { H o t s p o t R e c a l l } = \frac { | \hat { \Omega } _ { S } \cap \Omega _ { S } | } { | \Omega _ { S } | } .\tag{57}
$$

It measures whether the predicted stress field recovers the true concentrated stress region. Region IoU uses the same hotspot masks to measure overlap, while soft IoU computes overlap on normalized continuous stress responses. Flow is evaluated by moving-region coverage and velocity-direction agreement, summarized by motion quality. Force is a binary contact mask, so we directly evaluate masked reconstruction and region overlap.

## 6 Experiments

We evaluate parameter and constitutive inference, dense fields, simulation replay, and protocolmatched transfer; the evaluation also covers scaling, unseen interactions, domain adaptation, input robustness, and real videos. Unless noted otherwise, all supervised methods use three-view RGB videos and the cleaned object-level split of 5,000 samples: 496 valid samples for E, $\nu ,$ and density, and 164 yielding samples for yield stress.

Table 1. Comparison with VLM and direct regression baselines on simulation-ready material parameter inference. We report unfiltered parameter-level median absolute error (MedAE) and sample-wise parameter error (SPE), together with SPE after removing the top 20% highest-SPE samples for each method. $E ,$ density, and yield stress are evaluated in log-space, while ν is evaluated in raw space.
<table><tr><td rowspan="3">Method</td><td colspan="4">MedAE</td><td colspan="4">SPE</td></tr><tr><td>E↓</td><td>ν↓</td><td>Density ↓ Yield ↓</td><td></td><td>Unfilt. Med. ↓</td><td>Unfilt. Mean ↓</td><td>Med. ↓</td><td>Top-20% Filt. Top-20% Filt. Mean ↓</td></tr><tr><td>VLM baseline</td><td>2.601</td><td>0.079</td><td>1.301</td><td>3.028</td><td>0.645</td><td>0.677</td><td>0.600</td><td>0.576</td></tr><tr><td>Supervised baseline</td><td>1.901</td><td>0.048</td><td>0.408</td><td>1.797</td><td>0.304</td><td>0.314</td><td>0.275</td><td>0.270</td></tr><tr><td>Ours</td><td>1.521</td><td>0.045</td><td>0.389</td><td>1.585</td><td>0.299</td><td>0.334</td><td>0.251</td><td>0.266</td></tr></table>

Table 2. Material-wise parameter error analysis of our method. SPE summarizes the normalized error of the full parameter set for each sample. E error is reported in log-space.
<table><tr><td>Material</td><td>Count</td><td>Med. SPE↓</td><td>Mean SPE↓</td><td>E MedAE↓</td><td>E Mean AE↓</td></tr><tr><td>Jelly</td><td>170</td><td>0.286</td><td>0.317</td><td>1.137</td><td>1.869</td></tr><tr><td>Metal</td><td>164</td><td>0.382</td><td>0.383</td><td>3.195</td><td>3.354</td></tr><tr><td>Plasticine</td><td>162</td><td>0.252</td><td>0.300</td><td>1.226</td><td>1.359</td></tr></table>

## 6.1 Benchmark Comparison

Table 1 compares our method with the VLM and supervised baselines on simulation-ready material parameter inference. We report per-parameter MedAE as well as sample-wise parameter error (SPE), which normalizes each sample’s valid parameter errors by robust test-set ranges.

The VLM baseline is weak at numerical recovery, whereas supervised regression is competitive and has the best unfiltered mean SPE. Relative to that baseline, ours reduces MedAE from 1.901 to 1.521 for E, 0.048 to 0.045 for ν, 0.408 to 0.389 for density, and 1.797 to 1.585 for yield stress; it also has the best median and top-20%-filtered SPE. The supervised baseline’s better mean SPE (0.314 versus 0.334) exposes a heavier tail for ours, primarily on visually ambiguous high-stifness metal, rather than uniform superiority. Sections 6.1.1 and 6.1.2 provide the full distributions and material-wise analysis.

## 6.1.1 Parameter Prediction Visualization

Figure 3 provides additional visualizations for material-parameter prediction. The scatter plots compare predicted and ground-truth values for each parameter, while the sorted SPE curve shows the distribution of per-sample parameter-set errors and highlights the top-20% highest-error cases.

## 6.1.2 Material-wise Analysis

A material-wise breakdown further confirms that the long-tail behavior is mainly caused by highstifness metal samples. For our method, metal has a much higher SPE than the other material groups, with median/mean SPE of 0.382, compared with 0.286 for jelly and 0.252 for plasticine. The diference is especially pronounced for Young’s modulus: the metal subset has an E MedAE of 3.195 and mean AE of 3.354, while jelly and plasticine have much lower E MedAE values of 1.137 and 1.226, respectively. This supports our observation that visually estimating stifness for metal is intrinsically dificult under the available deformation cues. Therefore, the unfiltered mean SPE is strongly afected by a small number of high-stifness metal failure cases, whereas the top-20% filtered metrics better reflect performance on the common deformable-object regime.

![](images/163ff4b9a90a6d4288051533cfed842452e01879e522eb5f591c2cd512fc3be0.jpg)  
Figure 3. Material-parameter predictions from the VLM baseline, supervised baseline, and our model. The four scatter plots show prediction versus ground truth for density, $E , \nu ,$ and yield stress; the red diagonal is ideal agreement. The sorted SPE plot reports the full sample-wise error distribution, with dashed segments identifying each method’s highest-error 20%. It exposes both the improved typical-sample performance and the remaining long-tail failures.

## 6.2 Physical-Field Evidence

Physical-field bridging raises stress-hotspot recall from 0.0762 to 0.5121 and flow coverage IoU from 0.4054 to 0.6675 over direct field regression. Without retraining, masking stress raises E MedAE from 1.521 to 2.431 and yield-stress MedAE from 1.585 to 2.134; flow and contact are complementary. Sections 6.2.1 and 6.2.3 report the complete intervention and field results.

## 6.2.1 Quantitative Field Evaluation

Beyond parameter regression, we evaluate stress, flow, and force predictions in Table 3. These metrics are designed around the physical meaning of each field. For stress, exact full-field reconstruction is less important than identifying high-response regions, since stress hotspots reveal concentrated force and deformation-sensitive areas. We therefore emphasize hotspot recall and region-overlap metrics. For flow, correct coverage of moving regions is not suficient: the predicted motion should also carry meaningful velocity direction and magnitude. We therefore report both coverage IoU and velocity-aware motion quality. Force is a contact/force mask, so its evaluation is more direct and uses masked reconstruction and region overlap.

Our model improves stress hotspot recall, stress region overlap, flow coverage, velocity overlap, and temporal consistency over the supervised baseline. Compared with the supervised baseline, which directly predicts all outputs from shared video features, our method performs stress inference through a chained physical reasoning process. The predicted flow and force representations provide additional cues about motion discontinuities, contact regions, and force concentration, all of which are closely related to stress localization. Moreover, our model uses both static geometric features and dynamic deformation features, allowing stress prediction to depend on object structure as well as temporal response. These additional physical and spatiotemporal cues make stress hotspots easier to identify, explaining the gain over direct supervised field regression. Qualitative examples of the predicted flow, force/contact, and stress fields are provided in Section 6.2.3.

Table 3. Field prediction quality on the evaluation split. Metrics are computed inside the object mask. Our physical-field-bridged model achieves stronger hotspot localization and motion-field quality.
<table><tr><td rowspan="3">Method</td><td colspan="3">Stress</td><td colspan="4">Flow</td><td>Force</td></tr><tr><td>Hotspot Rec.↑</td><td>Region IoU↑</td><td>Soft IoU↑</td><td>Coverage IoU↑</td><td>Velocity Overlap↑</td><td>Motion Quality↑</td><td>Temporal Cons.↑</td><td>MSE↓</td></tr><tr><td>Supervised baseline</td><td>0.0762</td><td>0.0245</td><td>0.4815</td><td>0.4054</td><td>0.1232</td><td>0.2184</td><td>0.7141</td><td>0.0601</td></tr><tr><td>Ours</td><td>0.5121</td><td>0.1524</td><td>0.7719</td><td>0.6675</td><td>0.1725</td><td>0.3316</td><td>0.8135</td><td>0.0597</td></tr></table>

Table 4. Inference-time physical-field masking. Values are MedAE; lower is better.
<table><tr><td>Regressor input</td><td>E</td><td>ν</td><td>Density</td><td>Yield</td></tr><tr><td>Full model</td><td>1.521</td><td>0.045</td><td>0.389</td><td>1.585</td></tr><tr><td>Stress masked</td><td>2.431</td><td>0.043</td><td>0.394</td><td>2.134</td></tr><tr><td>Flow masked</td><td>2.035</td><td>0.058</td><td>0.405</td><td>1.913</td></tr><tr><td>Contact masked</td><td>1.980</td><td>0.057</td><td>0.406</td><td>1.878</td></tr><tr><td>All fields masked</td><td>2.525</td><td>0.045</td><td>0.391</td><td>2.149</td></tr></table>

## 6.2.2 Field Contribution Intervention

To test whether the predicted fields are actually used by the final regressor, we mask one field at inference time without retraining. When flow or contact is masked, the original predicted stress is retained so that the intervention measures the direct contribution of the masked field rather than its indirect efect through the stress branch. Table 4 shows complementary roles: stress is the strongest cue for E and yield stress, while flow and contact contribute to all four parameters.

## 6.2.3 Qualitative Field Prediction Results

Figure 4 shows qualitative examples of the predicted dense physical fields. Across diferent unseen objects, the model recovers the main motion-flow regions, contact/force regions, and stress-concentration patterns from multi-view RGB videos.

![](images/c3b42211ab7d922016289eb0e9cd41a6b2a58ac5443ec46ae8f54228ee4ff922.jpg)

![](images/545e29bb283638386b8540edb08547d4bec1003caac542043f3a4203203fae60.jpg)

![](images/2eb968a8570168b81627e5474f3fb332e087b96480a28545a7b4de58fc55ee7f.jpg)  
Figure 4. Qualitative physical-field predictions for three unseen objects over representative timesteps. Rows compare ground truth (green label) and prediction (red label) for projected motion flow, force/contact response, and projected stress. The model recovers the principal moving regions, localized contacts, and high-response stress structure despite changes in geometry and action.

Table 5. Ablation study on physical-field-bridged parameter inference. We report unfiltered sample-wise parameter error (SPE) and parameter mean MAE, together with the same metrics after removing the top 20% highest-SPE samples for each variant. Lower is better.
<table><tr><td rowspan="3">Variant</td><td colspan="4">SPE</td><td colspan="2">MAE</td></tr><tr><td>Unfilt. Med. ↓</td><td>Unfilt. Mean ↓</td><td>Filt. Med. ↓</td><td>Filt. Mean ↓</td><td>Unfilt. Param Mean ↓</td><td>Filt. Param Mean ↓</td></tr><tr><td>w/o boundary reasoning</td><td>0.321</td><td>0.345</td><td>0.272</td><td>0.279</td><td>1.197</td><td>1.013</td></tr><tr><td>w/o field auxiliary learning</td><td>0.312</td><td>0.349</td><td>0.266</td><td>0.274</td><td>1.204</td><td>1.023</td></tr><tr><td>w/o multi-stage training</td><td>0.326</td><td>0.349</td><td>0.275</td><td>0.285</td><td>1.118</td><td>0.946</td></tr><tr><td>Ours</td><td>0.299</td><td>0.334</td><td>0.251</td><td>0.266</td><td>1.139</td><td>0.940</td></tr></table>

Table 6. Prediction quality under parameter ambiguity. Results are averaged over 10 unseen objects, five loading types, and 5,000 parameter combinations.
<table><tr><td>Method</td><td>Metric</td><td>Bend</td><td>Drop</td><td>Press</td><td>Shear</td><td>Stretch</td><td>Mean</td></tr><tr><td>Supervised</td><td>SSIM ↑</td><td>0.736</td><td>0.628</td><td>0.762</td><td>0.732</td><td>0.765</td><td>0.725</td></tr><tr><td></td><td>PSNR↑</td><td>24.22</td><td>22.25</td><td>24.77</td><td>23.78</td><td>24.20</td><td>23.86</td></tr><tr><td>Ours</td><td>SSIM ↑</td><td>0.918</td><td>0.746</td><td>0.845</td><td>0.921</td><td>0.907</td><td>0.868</td></tr><tr><td></td><td>PSNR ↑</td><td>39.01</td><td>27.78</td><td>33.68</td><td>46.34</td><td>37.36</td><td>36.82</td></tr></table>

## 6.3 Object-level Generalization and Ablation Study

We use an object-level train/test split to test whether the model learns transferable physical rules rather than memorizing object-specific appearance. Under this setting, our method achieves the strongest robust sample-wise parameter accuracy. In particular, Table 5 shows that the full model obtains the best unfiltered median SPE and the best top-20% filtered SPE. Although the “w/o multi-stage training” variant has a slightly lower unfiltered Param Mean MAE, it performs worse in SPE, indicating that per-parameter averages alone do not fully capture whether a method predicts a coherent parameter set for each sample.

The ablations confirm the importance of the full design. Removing boundary reasoning weakens edge/contact-sensitive supervision. Removing field auxiliary learning degrades the physical intermediate representation. Removing multi-stage training reduces the stability of the field-bridged learning process. The full model achieves the best range-balanced parameter-set prediction, validating the role of explicit physical fields and staged reasoning in object-level generalization.

## 6.4 Simulation-based Evaluation under Parameter Ambiguity

Because distinct parameters can produce similar trajectories under limited actions, we replay the predicted constitutive type and parameters under the observed action and compare renderings. Misclassified types are not corrected. Table 6 shows higher replay SSIM and PSNR than the supervised baseline across all five modes, but these scores measure behavior rather than exact parameter recovery; Section 6.4.1 provides additional analysis.

## 6.4.1 Simulation Replay Analysis

Figure 5 provides additional quantitative analysis of the simulation-based replay results. This analysis complements Table 6 by showing the distribution of replay quality across unseen objects and loading modes.

![](images/6748d0c39ec86375e9f2d638003f66dc2fd74673f0fd31480df640c907dbc901.jpg)  
Figure 5. Simulation replay examples across unseen objects and loading modes, grouped from lower to higher SSIM. Each example compares ground-truth and predicted trajectories and lists the corresponding continuous parameter sets. Similar motion can arise from noticeably diferent parameters, illustrating why replay agreement complements but does not replace parameter-space evaluation.

Table 7. Protocol-matched evaluation on GSO.
<table><tr><td>Method</td><td>PSNR ↑</td><td>SSIM ↑</td><td>Runtime/sequence ↓</td></tr><tr><td>Vid2Sim [7]</td><td>25.07</td><td>0.945</td><td>~15-120 min</td></tr><tr><td>Spring-Gaus [32]</td><td>18.32</td><td>0.905</td><td>~15-120 min</td></tr><tr><td>Ours</td><td>17.15</td><td>0.882</td><td>~30 s</td></tr></table>

## 6.5 Protocol-Matched Cross-Benchmark Evaluation

Under Vid2Sim’s GSO protocol (16 inference and 8 prediction frames), Vid2Sim reaches 25.07 dB/0.945 PSNR/SSIM and Spring-Gaus reaches 18.32 dB/0.905, while ours reaches 17.15 dB/0.882 in about 30 seconds rather than 15–120 minutes. On Spring-Gaus [32], its native method reaches 17.06 dB/0.897 and ours 12.35 dB/0.791. This is an accuracy–eficiency trade-of, not cross-simulator superiority: Vid2Sim uses roughly 50K matched falling-object sequences, and interaction and simulator shifts reduce transfer. Section 6.5 gives the protocol, gap diagnosis, and adaptation.

## 6.5.1 GSO Future-State Protocol

Following Vid2Sim, we use the first 16 frames for material inference, align the simulated state with the final observed frame, and predict the subsequent 8 frames. Table 7 reports all methods on this common protocol. Runtimes are measured or reported on one NVIDIA RTX 4090.

Vid2Sim is trained on approximately 50K falling-object sequences closely aligned with the GSO evaluation, whereas our original training set spans multiple actively applied forces and is not specialized to dropping. Per-scene approaches can specialize reconstruction and parameters to each sequence; our method instead amortizes inference across scenes. The result should therefore be interpreted as an accuracy–eficiency trade-of rather than an accuracy claim over per-scene optimization.

## 6.5.2 Complete Spring-Gaus Synthetic Set

We retain every Spring-Gaus sequence rather than selecting simulator-compatible cases. Spring-Gaus obtains 17.06 dB PSNR and 0.897 SSIM, while our zero-shot model obtains 12.35 dB and 0.791. Directly replaying the reported Spring-Gaus parameters in our simulator does not reproduce all benchmark trajectories: several objects rebound above 60% of their initial drop height, whereas the rebound in our original configuration remains below approximately 5% even after sweeping the material ranges represented in DeformFieldBench. The discrepancy indicates that trajectories depend on damping, interior filling, time integration, contact implementation, and boundary-condition implementation in addition to nominal material parameters. Complete reproduction of all Spring-Gaus simulator settings is not possible from the available implementation details.

Table 8. Efect of 500-simulation domain adaptation.
<table><tr><td>Evaluation benchmark</td><td>Zero-shot PSNR/SSIM</td><td>Adapted PSNR/SSIM</td></tr><tr><td>Vid2Sim/GSO</td><td>17.15 / 0.882</td><td>20.21 / 0.923</td></tr><tr><td>Spring-Gaus</td><td>12.35 / 0.791</td><td>13.97 / 0.824</td></tr></table>

Table 9. Continuous-parameter accuracy on 50 additional unseen objects. Values are MedAE.
<table><tr><td> $E \downarrow$ </td><td> $\nu \downarrow$ </td><td>Density↓</td><td>Yield ↓</td></tr><tr><td>1.620</td><td>0.039</td><td>0.374</td><td>1.821</td></tr></table>

## 6.5.3 Lightweight Domain Adaptation

To separate method capacity from domain mismatch, we generate two domain-specific sets: 500 falling-object simulations following the Vid2Sim generation protocol, and 500 simulations configured to better reproduce Spring-Gaus dynamics. The original model is fine-tuned separately on each set.

Adaptation improves GSO by 3.06 dB and 0.041 SSIM and Spring-Gaus by 1.62 dB and 0.033 SSIM. These improvements indicate a substantial interaction- and simulator-domain component in the zero-shot gap, but they do not eliminate the gap or prove simulator invariance. Broader randomization over damping, object filling, time integration, contact, and boundary conditions is an important extension; it is not retroactively attributed to the original 5K training results.

## 6.5.4 Broader Robustness and Real-world Transfer

On 50 disjoint DifSplat objects, ours reaches 36.92 dB/0.8820 PSNR/SSIM; full-set training improves over 10 objects by 13.41 dB/0.1635. Without interaction-specific retraining, squeezing/lifting and force-controlled indentation reach 36.41/0.842 and 24.10/0.712, while five real sequences span 15.864–20.561 dB and 0.7714–0.8430. Sections 6.6–6.8 include complete results and view/resolution sensitivity. These tests measure behavior-level transfer under unknown real forces, heterogeneous composition, viewpoint shift, and simulator mismatch.

## 6.6 Expanded Object and Interaction Generalization

## 6.6.1 Additional Unseen DifSplat Objects

We generate 50 objects with DifSplat [23] that are disjoint from the original 100 assets. All objects, including severe failures, are retained in the aggregate results.

The corresponding future-state replay reaches 36.92 dB PSNR and 0.8820 SSIM. Approximately 10% of the new assets exhibit severe replay failure or geometric collapse, and these cases are not removed. Post-hoc inspection suggests that some failures resemble training geometries whose simulations occupy narrow parameter ranges. This is a diagnostic hypothesis rather than a confirmed causal explanation and may indicate residual geometry–parameter correlation.

Table 10. Replay performance as object diversity increases.
<table><tr><td>Training configuration</td><td>PSNR ↑</td><td>SSIM ↑</td></tr><tr><td>10 training objects</td><td>23.41</td><td>0.7041</td></tr><tr><td>Full training set</td><td>36.82</td><td>0.8676</td></tr><tr><td>Improvement</td><td>+13.41</td><td>+0.1635</td></tr></table>

Table 11. Generalization to held-out interactions.
<table><tr><td>Interaction</td><td>PSNR ↑</td><td>SSIM ↑</td></tr><tr><td>Gripper squeezing and lifting</td><td>36.41</td><td>0.842</td></tr><tr><td>Force-controlled indentation</td><td>24.10</td><td>0.712</td></tr></table>

## 6.6.2 Dataset Scaling

We train the same model with simulations from only 10 objects and compare it with the full training set.

The large gain confirms that the model benefits from increased diversity in object geometry and simulated trajectories.

## 6.6.3 Held-Out Interactions

We evaluate two interactions absent from the five original training modes, without interactionspecific retraining.

Squeezing and lifting transfer well because their contact and deformation patterns overlap with structures represented during training. Force-controlled indentation is more distinct and yields lower, but still meaningful, replay agreement. Transfer therefore depends on how well the new interaction is covered by the physical patterns in the training distribution.

## 6.7 Input Resolution and View Robustness

## 6.7.1 Resolution Sensitivity

We downsample the RGB videos at inference time without retraining or fine-tuning.

Full-resolution input is strongest overall, but the model remains functional at $1 2 8 \times 1 2 8$ without resolution-specific training.

## 6.7.2 Number and Configuration of Views

The original cameras are stationary within each sequence and share three canonical horizontal extrinsics across normalized scenes. Because the architecture expects three inputs, single-view inference repeats one available view three times; two-view inference duplicates one of the two views. The model does not receive camera extrinsics and does not impose camera-transformation equivariance.

Table 12. Resolution sensitivity of continuous-parameter prediction. Values are MedAE.
<table><tr><td>Resolution</td><td> $E \downarrow$ </td><td> $\nu \downarrow$ </td><td>Density ↓</td><td>Yield ↓</td></tr><tr><td> $5 1 2 \times 5 1 2$ </td><td>1.5210</td><td>0.0450</td><td>0.3890</td><td>1.5850</td></tr><tr><td> $2 5 6 \times 2 5 6$ </td><td>1.8365</td><td>0.0638</td><td>0.4059</td><td>1.7163</td></tr><tr><td> $1 2 8 \times 1 2 8$ </td><td>1.9200</td><td>0.0651</td><td>0.5083</td><td>1.6780</td></tr></table>

Table 13. Single-view inference without retraining. Values are MedAE.
<table><tr><td>Input configuration</td><td>E</td><td>ν</td><td>Density</td><td>Yield</td></tr><tr><td>Three canonical views</td><td>1.521</td><td>0.0450</td><td>0.389</td><td>1.585</td></tr><tr><td>First view repeated</td><td>2.026</td><td>0.0574</td><td>0.420</td><td>1.921</td></tr><tr><td>Second view repeated</td><td>1.670</td><td>0.0494</td><td>0.357</td><td>2.047</td></tr><tr><td>Third view repeated</td><td>1.580</td><td>0.0482</td><td>0.391</td><td>1.745</td></tr></table>

Table 14. Quantitative replay on real-world deformation videos.
<table><tr><td>Object</td><td>Interaction</td><td>PSNR ↑</td><td>SSIM↑</td></tr><tr><td>Wallet</td><td>Bending</td><td>15.864</td><td>0.7714</td></tr><tr><td>Wallet</td><td>Compression</td><td>16.427</td><td>0.7936</td></tr><tr><td>Wallet</td><td>Stretching</td><td>16.042</td><td>0.7851</td></tr><tr><td>Plastic bottle</td><td>Bending</td><td>17.642</td><td>0.8231</td></tr><tr><td>Plastic bottle</td><td>Compression</td><td>20.561</td><td>0.8430</td></tr></table>

With duplicated two-view inputs, E MedAE ranges from 1.833 to 2.144 depending on the pair and duplicated view. Across random three-view configurations, E MedAE ranges from 1.483 to 2.491 and replay PSNR ranges from 21.67 to 35.42 dB. Several configurations match or slightly improve on the canonical views, but top-down or bottom-up views can hide the dominant contact and deformation. These results show that observable evidence matters at least as much as nominal view count, while the performance range confirms meaningful camera sensitivity.

Future work should randomize camera angle, elevation, view count, view dropping, and ordering during training. A pose-conditioned set encoder that consumes camera extrinsics would support variable view counts more directly; these extensions are not part of the reported original model.

## 6.8 Real-world Object Application

We quantitatively evaluate five real deformation sequences involving two previously unseen objects: a wallet under bending, compression, and stretching, and a thin-walled plastic bottle under bending and compression. No real-data fine-tuning or ground-truth force, stress, material parameter, or constitutive-type label is used.

Replay protocol. For each interaction, we capture synchronized multi-view RGB videos, predict the constitutive type and continuous parameters, align the initial object geometry and cameras, and replay the observed interaction with the predicted material configuration. The same foreground mask is applied to the aligned real and rendered sequences. Masks are initialized with SAM and manually refined to remove background regions and areas occluded by the manipulating hand. We then compute image-space PSNR and SSIM between the real and rendered foreground sequences.

The plastic-bottle compression sequence has the strongest agreement, where interaction and boundary conditions can be aligned most reliably. Residual diferences arise from geometry and camera alignment, unknown contact forces, approximate boundary conditions, hand occlusion, trajectory mismatch, heterogeneous object composition, and constitutive behavior outside the current simulator. PSNR and SSIM therefore measure replay consistency at the deformationbehavior level; they do not establish that the inferred numerical parameters equal unknown real material properties.

## 7 Discussion and Limitations

Our results suggest that simulation-ready material inference benefits from explicitly decomposing visual observations into dense physical fields before predicting a constitutive family and global material parameters. Compared with direct video-to-parameter regression, force/contact, motion flow, and stress response provide more interpretable physical evidence, especially when rebound, oscillation, recovery, and stress concentration are subtle.

Despite these benefits, our study has several limitations. First, DeformFieldBench is primarily synthetic. The real-world replay study provides quantitative behavior-level evidence but lacks calibrated force sensing and ground-truth material parameters, so it cannot validate absolute recovery. Second, zero-shot performance drops across simulator and interaction domains. Damping, object filling, integration, contact, and boundary-condition implementations can change a trajectory even when nominal material parameters match. Third, the original model is trained with a canonical three-camera arrangement, does not consume extrinsics, and remains sensitive to whether a view exposes contact and deformation. Fourth, its flow and stress outputs are image-space projections rather than volumetric fields. Finally, continuous-parameter errors remain substantial even when replay appears plausible, reflecting both observation ambiguity and the many-to-one relationship between parameters and trajectories.

Homogeneous-material assumption. The current output is one global constitutive type and parameter set per object. Dense physical fields act as spatial evidence for global system identification; they do not recover regional material properties. For a composite object such as a wallet made of leather, fabric, stitching, and metal, the result should be interpreted as an efective global approximation of the dominant observed deformation, not as the physical property of each component. A spatial extension could lift multi-view and field features to surface points, volumetric elements, or MPM particles and predict region-level constitutive types and parameters. Such an extension requires heterogeneous training objects, material-boundary annotations, image-to-particle correspondence, region-level supervision, and simulators that handle material interfaces.

We observe that most long-tail parameter errors are concentrated on high-stifness metallic samples. This failure mode is partly caused by an identifiability limitation of the current loading protocol, rather than only by insuficient model capacity. In our simulator, press, stretch, shear, and bend actions are imposed by kinematic actuators with prescribed trajectories. For high-stifness metallic objects, this can resemble applying an efectively large external force until deformation occurs. Under such force-saturated loading, metallic objects with diferent Young’s moduli or yield stresses may exhibit similar visible behavior, including limited elastic recovery, permanent deformation, and saturated relative stress responses. Consequently, their RGB deformation trajectories, projected flow fields, and normalized image-space stress maps become less discriminative than those of more recoverable materials such as jelly-like or plasticine-like objects.

A direct way to reduce this ambiguity is to include absolute force or stress measurements. For example, a robotic force-torque sensor, tactile array, pressure sensor, or calibrated actuator-force signal could reveal the force magnitude at which deformation starts, which would be informative for distinguishing near-rigid metallic materials. Our current dataset does not include such absolute external-force measurements or per-particle absolute stress targets; instead, it provides rendered and normalized physical-response fields. This design keeps the benchmark lightweight, visually aligned, and suitable for dense pixel-level supervision, but it also limits the identifiability of high-stifness materials whose visible deformation responses are weak or saturated.

Future versions of DeformFieldBench could incorporate calibrated loading magnitudes, perpoint absolute stress, tactile or force-torque measurements, spatially varying materials, and broader randomization of simulator settings. Training with randomized camera angle, elevation, view count, view dropping, and view order would reduce reliance on the canonical arrangement; a pose-conditioned set encoder could explicitly support arbitrary cameras and variable view counts. These are future extensions rather than components of the original 5K-result model. The framework could also be integrated with vision-language priors and robotic sensing to support downstream reasoning, planning, and interaction.

## 8 Conclusion

We introduced physical-field-bridged material inference, which organizes observable contact, motion, and stress evidence before predicting a constitutive type and continuous parameters from multi-view RGB videos. DeformFieldBench supplies the aligned dense fields and simulator-ready supervision required to study this formulation. The physical fields improve median parameter accuracy, field quality, object-level generalization, and same-simulator replay; protocol-matched, held-out-interaction, input-robustness, adaptation, and real-world evaluations characterize transfer beyond the original split. The current scope remains bounded: absolute parameter recovery is dificult, image-space replay measures behavioral consistency rather than material truth, and performance is sensitive to camera evidence and simulator configuration. Within this scope, the results support physical-field bridging as a practical basis for fast material inference while motivating pose-aware aggregation, broader simulator randomization, calibrated sensing, and spatially varying material prediction.

## References

[1] Yizhou Zhao, Haoyu Chen, Chunjiang Liu, Zhenyang Li, Charles Herrmann, Junhwa Hur, Yinxiao Li, Ming-Hsuan Yang, Bhiksha Raj, and Min Xu. Masiv: Toward material-agnostic system identification from videos, 2025. URL https://arxiv.org/abs/2508.01112.

[2] Junhao Cai, Yuji Yang, Weihao Yuan, Yisheng He, Zilong Dong, Liefeng Bo, Hui Cheng, and Qifeng Chen. Gic: Gaussian-informed continuum for physical property identification and simulation, 2024. URL http://arxiv.org/abs/2406.14927.

[3] Shiqian Li, Ruihong Shen, Junfeng Ni, Chang Pan, Chi Zhang, and Yixin Zhu. Learning physicsgrounded 4d dynamics with neural gaussian force fields, 2026. URL http://arxiv.org/abs/2602.00148.

[4] Yunuo Chen, Yafei Hu, Lingfeng Sun, Tushar Kusnur, Laura Herlant, and Chenfanfu Jiang. Empm: Embodied mpm for modeling and simulation of deformable objects, 2026. URL http://arxiv.org/abs/ 2601.17251.

[5] Hanxiao Jiang, Hao-Yu Hsu, Kaifeng Zhang, Hsin-Ni Yu, Shenlong Wang, and Yunzhu Li. Phystwin: Physics-informed reconstruction and simulation of deformable objects from videos, 2025. URL http://arxiv.org/abs/2503.17973.

[6] Xuan Li, Yi-Ling Qiao, Peter Yichen Chen, Krishna Murthy Jatavallabhula, Ming Lin, Chenfanfu Jiang, and Chuang Gan. Pac-nerf: Physics augmented continuum neural radiance fields for geometry-agnostic system identification. arXiv preprint arXiv:2303.05512, 2023.

[7] Chuhao Chen, Zhiyang Dou, Chen Wang, Yiming Huang, Anjun Chen, Qiao Feng, Jiatao Gu, and Lingjie Liu. Vid2sim: Generalizable, video-based reconstruction of appearance, geometry and physics for mesh-free simulation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26545–26555, 2025.

[8] Tianyuan Zhang, Hong-Xing Yu, Rundi Wu, Brandon Y. Feng, Changxi Zheng, Noah Snavely, Jiajun Wu, and William T. Freeman. Physdreamer: Physics-based interaction with 3d objects via video generation, 2024. URL http://arxiv.org/abs/2404.13026.

[9] Xinli Xu, Wenhang Ge, Dicong Qiu, ZhiFei Chen, Dongyu Yan, Zhuoyun Liu, Haoyu Zhao, Hanfeng Zhao, Shunsi Zhang, Junwei Liang, and Ying-Cong Chen. Gaussianproperty: Integrating physical properties to 3d gaussians with lmms, 2024. URL http://arxiv.org/abs/2412.11258.

[10] Yinghao Shuai, Ran Yu, Yuantao Chen, Zijian Jiang, Xiaowei Song, Nan Wang, Jv Zheng, Jianzhu Ma, Meng Yang, Zhicheng Wang, Wenbo Ding, and Hao Zhao. Pugs: Zero-shot physical understanding with gaussian splatting, 2025. URL http://arxiv.org/abs/2502.12231.

[11] Albert J Zhai, Yuan Shen, Emily Y Chen, Gloria X Wang, Xinlei Wang, Sheng Wang, Kaiyu Guan, and Shenlong Wang. Physical property understanding from language-embedded feature fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 28296– 28305, 2024.

[12] Jiajun Wu, Joseph Lim, Hongyi Zhang, Joshua Tenenbaum, and William Freeman. Physics 101: Learning physical object properties from unlabeled videos. In Edwin R. Hancock Richard C. Wilson and William A. P. Smith, editors, Proceedings of the British Machine Vision Conference (BMVC), pages 39.1–39.12. BMVA Press, September 2016. ISBN 1-901725-59-6. doi: 10.5244/C.30.39. URL https://dx.doi.org/10.5244/C.30.39.

[13] Zhenjia Xu, Jiajun Wu, Andy Zeng, Joshua B. Tenenbaum, and Shuran Song. Densephysnet: Learning dense physical object representations via multi-step dynamic interactions. In Robotics: Science and Systems (RSS), 2019. URL http://www.zhenjiaxu.com/DensePhysNet/.

[14] Daniel M Bear, Elias Wang, Damian Mrowca, Felix J Binder, Hsiao-Yu Fish Tung, RT Pramod, Cameron Holdaway, Sirui Tao, Kevin Smith, Fan-Yun Sun, et al. Physion: Evaluating physical prediction from vision in humans and machines. arXiv preprint arXiv:2106.08261, 2021.

[15] Chuang Gan, Jeremy Schwartz, Seth Alter, Damian Mrowca, Martin Schrimpf, James Traer, Julian De Freitas, Jonas Kubilius, Abhishek Bhandwaldar, Nick Haber, et al. Threedworld: A platform for interactive multi-modal physical simulation. arXiv preprint arXiv:2007.04954, 2020.

[16] Xingyu Lin, Yufei Wang, Jake Olkin, and David Held. Softgym: Benchmarking deep reinforcement learning for deformable object manipulation. In Conference on Robot Learning, pages 432–448. PMLR, 2021.

[17] Hugo Bertiche, Meysam Madadi, and Sergio Escalera. Cloth3d: clothed 3d humans. In European Conference on Computer Vision, pages 344–359. Springer, 2020.

[18] Rika Antonova, Peiyang Shi, Hang Yin, Zehang Weng, and Danica Kragic Jensfelt. Dynamic environments with deformable objects. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021.

[19] Zhen Zhang, Xiangyu Chu, Yunxi Tang, and KW Au. Dofs: A real-world 3d deformable object dataset with full spatial information for dynamics model learning. arXiv preprint arXiv:2410.21758, 2024.

[20] Jan Obrist, Miguel Zamora, Hehui Zheng, Ronan Hinchet, Firat Ozdemir, Juan Zarate, Robert K. Katzschmann, and Stelian Coros. Pokeflex: A real-world dataset of volumetric deformable objects for robotics, 2025. URL http://arxiv.org/abs/2410.07688.

[21] Guanxing Lu, Shiyi Zhang, Ziwei Wang, Changliu Liu, Jiwen Lu, and Yansong Tang. Manigaussian: Dynamic gaussian splatting for multi-task robotic manipulation, 2024. URL http://arxiv.org/abs/24 03.08321.

[22] Jiawei Ren, Liang Pan, Jiaxiang Tang, Chi Zhang, Ang Cao, Gang Zeng, and Ziwei Liu. Dreamgaussian4d: Generative 4d gaussian splatting, 2024. URL http://arxiv.org/abs/2312.17142.

[23] Chenguo Lin, Panwang Pan, Bangbang Yang, Zeming Li, and Yadong Mu. Difsplat: Repurposing image difusion models for scalable gaussian splat generation, 2025. URL http://arxiv.org/abs/2501.16764.

[24] Tianyi Xie, Zeshun Zong, Yuxing Qiu, Xuan Li, Yutao Feng, Yin Yang, and Chenfanfu Jiang. Physgaussian: Physics-integrated 3d gaussians for generative dynamics. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4389–4398, Seattle, WA, USA, 2024. IEEE. ISBN 979-8-3503-5300-6. URL https://ieeexplore.ieee.org/document/10658121/.

[25] Ziang Cao, Zhaoxi Chen, Liang Pan, and Ziwei Liu. Physx-3d: Physical-grounded 3d asset generation, 2025. URL https://arxiv.org/abs/2507.12465.

[26] Chunji Lv, Zequn Chen, Donglin Di, Weinan Zhang, Hao Li, Wei Chen, and Changsheng Li. Physgm: Large physical gaussian model for feed-forward 4d synthesis, 2025. URL http://arxiv.org/abs/2508.1 3911.

[27] Yu Yang, Zhilu Zhang, Xiang Zhang, Yihan Zeng, Hui Li, and Wangmeng Zuo. Physworld: From real videos to world models of deformable objects via physics-aware demonstration synthesis, 2025. URL http://arxiv.org/abs/2510.21447.

[28] Himangi Mittal, Peiye Zhuang, Hsin-Ying Lee, and Shubham Tulsiani. Uniphy: Learning a unified constitutive model for inverse physics simulation, 2025. URL https://arxiv.org/abs/2505.16971.

[29] Pingchuan Ma, Peter Yichen Chen, Bolei Deng, Joshua B. Tenenbaum, Tao Du, Chuang Gan, and Wojciech Matusik. Learning neural constitutive laws from motion observations for generalizable pde dynamics, 2023. URL http://arxiv.org/abs/2304.14369.

[30] Long Le, Ryan Lucas, Chen Wang, Chuhao Chen, Dinesh Jayaraman, Eric Eaton, and Lingjie Liu. Pixie: Fast and generalizable supervised learning of 3d physics from pixels. arXiv preprint arXiv:2508.17437, 2025.

[31] Albert J. Zhai, Yuan Shen, Emily Y. Chen, Gloria X. Wang, Xinlei Wang, Sheng Wang, Kaiyu Guan, and Shenlong Wang. Physical property understanding from language-embedded feature fields, 2024. URL http://arxiv.org/abs/2404.04242.

[32] Licheng Zhong, Hong-Xing Yu, Jiajun Wu, and Yunzhu Li. Reconstruction and simulation of elastic objects with spring-mass 3d gaussians. In European Conference on Computer Vision, 2024.