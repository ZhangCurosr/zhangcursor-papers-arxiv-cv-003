![](images/2a7cc115fd16c66ea011537bc47e2bff87032cedb19c286a1f9287d2e4756adb.jpg)

# Structure from Collision via Hypothesis-Space Reasoning

Takuhiro Kaneko NTT, Inc.

Same visible surface Different hidden interiors

![](images/73e986fc0ba40d8c1996ea20cff7feae3bca3dc86a822d24ed6ad588d195fd38.jpg)  
(a)

![](images/7a3ac3abebaa98ea0fd1267b0575af353963a49b2411021c7ce0e96eec2bd7c0.jpg)  
(b)  
(i) SfC task

(1) Global hypothesis optimization  
![](images/efd6c811a6a56781b3648cc13fa75e4368b473e7598f34608ff2d15dfb71eb50.jpg)  
For (a)  
For (b)  
Infer hidden interior structure from collision observations  
(ii) Previous density-based SfC method Ambiguous cavity localization

![](images/2a146f3b7460e15aa430578335af529cbd87cb4a9f18eb2bb07714ded187311a.jpg)  
For (a)

![](images/c068dc9d0228a741e06b561219020bebc3ee52fa1c1474e759016b492a3a7700.jpg)  
For (b)  
(iii) Hypothesis-space reasoning (ours) Coherent interior reconstruction  
Figure 1. Overview of Structure from Collision (SfC) via hypothesis-space reasoning. (i) SfC infers the hidden interior structure from collision observations of objects with the same visible surface but different interiors. (ii) A previous density-based SfC method [16] uses local density updates without an explicit hypothesis space, leading to ambiguous cavity localization. (iii) Our method optimizes explicit structural hypotheses globally and refines the selected hypothesis as a Lagrangian particle system, yielding coherent interior reconstruction

## Abstract

Knowledge ofthe hidden interior structure ofan object, including the presence and location of cavities, is essential for predictive simulation, robotic manipulation, and non destructive inspection. However, such cavities leave no direct signature on the static visible surface, making collisioninduced deformation one ofthefew cues used to infer their presence. Structure from Collision (SfC) addresses this problem by recovering the hidden interior structure of an objectfrom multiview observations ofa collision event. SfC is challenging for two reasons: hidden cavities are only indirectly observable through surface motion, and voids correspond to the absence of material, making direct op timization using standard density representations difficult. The density-based SfC baseline inherits both difficulties. Surface-matching gradients do not directly indicate the cavity location, while density gradients can be weak or uninformative inside or near void regions. Therefore, we reformulate SfC as hypothesis-space reasoning, representing hidden cavities as explicit structural hypotheses whose variables encode available structural assumptions rather than relying on implicit density updates. A global stage performs gradient-free optimization over this hypothesis space using forward-only material point method (MPM) simulation. A local stage then refines the selected hypothesis as a Lagrangian particle system through gradient-based optimization using differentiable MPM simulation. On the SfC benchmark, our method improves hidden-structure reconstruction and future prediction over the density-based SfC baseline. Interior-topology stress tests and robustness analyses characterize its scope and limitations.<sup>1</sup>

## 1. Introduction

Vision-based 3D reconstruction is important for applications that require physical understanding, such as predictive simulation, robotic manipulation, and nondestructive inspection. Recent methods have recovered detailed visible geometry and appearance from images or videos [18, 28]. However, visible geometry alone is insufficient to explain the physical behavior governed by internal mass distribution. For example, as shown in Fig. 1(i), two objects with identical external surfaces but different hidden cavities are indistinguishable from static visual observations, yet can respond differently to the same collision.

Structure from Collision (SfC) [16] addresses this limitation by using collision-induced motion to infer the hidden interior structure from multiview observations. Because post-collision surface motion reflects the internal mass distribution, cavities indistinguishable from the static appearance can be inferred from the deformation. However, this inverse problem is difficult for two related reasons: hidden cavities are only indirectly observable, and void regions are difficult to optimize directly with standard density representations because they represent the absence of material. As shown in Fig. 2, the post-collision surface-motion discrepancy between a hidden-cavity object and its filled initialization can suggest an internal correction at a location different from the true cavity. A previous density-based SfC method [16] inherits both difficulties. Without an explicit hypothesis space, surface-matching gradients may suggest corrections away from the true cavity, while density updates inside or near void regions can be weak or uninformative, leading to ambiguous cavity localization (Fig. 1(ii)).

To address these limitations, we reformulate SfC as hypothesis-space reasoning, replacing local density updates with explicit structural hypotheses evaluated against the observed collision response. This representation directly encodes available structural assumptions in the hypothesis space rather than indirectly through a density field. We instantiate this formulation in two complementary stages (Fig. 1(iii)). The global stage performs gradient-free optimization over the hypothesis space using forward-only material point method (MPM) simulation [10, 37], while the local stage refines the selected hypothesis as a Lagrangian particle system through gradient-based optimization with differentiable MPM simulation [11], recovering details beyond the initial hypothesis parameterization. This two-stage design separates coarse structural reasoning from fine geometric refinement.

On the SfC benchmark, the proposed method improves hidden-structure reconstruction and future prediction over the density-based SfC baseline [16]. Interior-topology stress tests and robustness analyses further characterize its scope and limitations.

Our contributions are summarized as follows:

• We identify a structural failure mode of density-based SfC. Indirect surface cues and void regions make local density gradients unreliable for cavity localization.

• We reformulate SfC as hypothesis-space reasoning, representing hidden cavities as explicit structural hypotheses rather than relying on implicit density updates.

• We propose a two-stage framework that combines gradient-free global hypothesis optimization with gradient-based Lagrangian particle refinement, separating hypothesis-level structural selection from particle-level geometric refinement.

• We demonstrate improved reconstruction and future prediction over the density-based SfC baseline [16], with interior-topology stress tests and robustness analyses clarifying the scope and limitations of the method.

## 2. Related Work

3D reconstruction. Classical structure-from-motion [36, 40] and multiview stereo [6, 34] recover surface geometry from images. Neural 3D representations, such as neural radiance fields (NeRF) [28] and 3D Gaussian splatting [18], further improve the reconstruction of visible geometry and appearance. However, these methods primarily model visible properties and do not explicitly target the hidden interior structures that govern the physical response.

Dynamic 3D reconstruction. 3D reconstruction methods have been extended to dynamic and nonrigid scenes [7, 17, 21, 26, 27, 30, 31, 41, 45, 47–49]. These methods typically learn deformation fields, canonical representations, and time-varying Gaussian primitives for temporally coherent reconstruction and dynamic view synthesis. However, they primarily model visible motion and appearance, and do not explicitly target hidden interior structures revealed only indirectly through physical interaction.

Physics-informed modeling and system identification. Recent studies have incorporated physical constraints or differentiable simulators to enforce plausible dynamics or estimate physical quantities from visual or multimodal observations [1–5, 8, 12–15, 20, 23–25, 29, 32, 35, 42, 43, 46, 50–52]. However, these methods typically assume a fixed interior, even though internal mass distribution can strongly affect physical behavior.

Structure from Collision. SfC [16] aims to recover internal cavity structures from collision-induced deformation. The previous method, SfC-NeRF [16], optimizes volumetric densities in an Eulerian representation such that the simulated surface dynamics match the observed motion. Although effective in some settings, this density-based formulation relies on local density updates without an explicit structural hypothesis space, which can result in ambiguous cavity localization. In contrast, by representing hidden cavities as explicit structural hypotheses, we first optimize them globally and then refine the selected hypothesis as a Lagrangian particle system.

## 3. Problem Setup and Failure Analysis

## 3.1. Problem Definition

We consider the problem of inferring the hidden interior structure from visual observations of a collision event. Given calibrated multiview videos $\{ \mathcal { T } _ { i } ( t _ { j } ) ~ | ~ i =$

$1 , \ldots , M , ~ j ~ = ~ 1 , \ldots , N \}$ , where M is the number of views and N is the number of time steps, we obtain two observation streams: a time-varying visible surface ${ \cal { S } } ( t _ { j } )$ and image-space object masks $\boldsymbol { A } _ { i } ( t _ { j } )$ . The surface ${ \cal { S } } ( t _ { j } )$ can be reconstructed from multiview observations using an off-the-shelf dynamic 3D reconstruction method [3, 45, 48]. The masks $\boldsymbol { A } _ { i } ( t _ { j } )$ can be extracted using video matting [22] or video object segmentation methods [33]. These observations constrain the external geometry and motion of an object but do not directly reveal hidden cavities. Thus, SfC aims to recover the hidden interior structure that shapes the collision response of an object rather than merely explaining the visible surface evolution.

We represent the object as a particle-based continuum. Let $\mathcal { P } ( t _ { 1 } )$ denote the ground-truth initial particle configuration and $\hat { \mathcal { P } } ( t _ { 1 } )$ denote its estimate. Given $\hat { \mathcal { P } } ( t _ { 1 } )$ and physical material parameters $\Theta _ { \mathrm { p h y s } }$ , a physics simulator (e.g., MPM [10, 37]) predicts the evolving particle configurations $\hat { \mathcal { P } } ( t _ { j } )$ . From these simulated configurations, we extract predicted surfaces $\hat { S } ( t _ { j } )$ and render predicted masks $\hat { \mathcal { A } } _ { i } ( t _ { j } )$ Thus, SfC can be formulated as an inverse problem: estimating an initial particle configuration $\hat { \mathcal { P } } ( t _ { 1 } )$ that captures the hidden interior structure whose simulated surface and mask evolution match the observed collision dynamics.

Unless otherwise stated, we treat the physical material parameters $\Theta _ { \mathrm { p h y s } }$ , such as the Young’s modulus, Poisson’s ratio, and mass density, as given. This controlled setting isolates the structural ambiguity from material-parameter ambiguity. However, specifying these physical parameters does not specify the cavity; its location, shape, and topology remain unknown. To assess the sensitivity of this assumption, we evaluate representative robustness settings in Section 5.6, with additional relaxed assumptions and physicalparameter error analyses in Appendices B.5 and B.6.

Under this formulation, a natural approach is to optimize the internal representation so that the predicted surfaces and masks match the observations. However, density-based local optimization is structurally mismatched to hidden-cavity inference, as analyzed in the following subsection.

## 3.2. Why Density-Based Local Optimization Fails

Fig. 2 illustrates a representative failure mode. A hidden cavity can induce a post-collision response whose surfacemotion discrepancy points away from the true cavity location. Thus, matching visible motion does not localize the area from which the material should be removed.

This example reflects two coupled difficulties. First, hidden cavities are only indirectly observable; surface or silhouette motion does not directly indicate which internal region is responsible. Second, cavities are void regions (i.e., the absence of material), making it difficult to optimize them directly with density-based representations. At the gradient level, local updates are driven by surface- or mask-level discrepancies. Near the visible surface, gradients tend to adjust the external motion or silhouettes, while inside or near void regions, little material exists to support informative density updates. Consequently, local updates can reduce observation losses in directions misaligned with the true cavity, leading to ambiguous reconstructions and misleading minima.

![](images/1958713e752a0bffe014c4425ea92464a46fd2f31579d14e034633a7dc5efcc7.jpg)  
Figure 2. Misleading surface-motion cue for hidden-cavity localization. An object with a left-side cavity (a) produces a leftward post-collision response (b), whereas the filled initialization used by a previous density-based SfC method (c) moves upward (d). The resulting post-collision difference (e) suggests the removal of material from the right side, even though the true cavity lies on the left. Thus, surface-matching cues alone do not reliably localize hidden cavities and can mislead local density-based optimization.

SfC-NeRF [16] partially mitigates these issues by optimizing volumetric densities and applying volume annealing to escape the local minima. Nevertheless, its optimization remains tied to local density updates starting from the filled initialization and lacks explicit space for structural alternatives. Therefore, it can remain sensitive to initialization and may converge to cavity configurations that reduce local observation errors while remaining structurally inconsistent. Together, these limitations motivate our formulation: we represent candidate cavities as explicit structural hypotheses and evaluate them before local particle-level refinement.

## 4. Method: Hypothesis-Space Reasoning

## 4.1. Overview

We instantiate SfC as hypothesis-space reasoning by representing hidden cavities as explicit structural hypotheses rather than relying on implicit density updates. This shifts the optimization from local density updates to hypothesislevel evaluation in a low-dimensional space that can explicitly encode available structural assumptions.

Our method begins with a dense initialization, followed by two optimization stages (Fig. 3). Following Gaussian-Informed Continuum (GIC) [3], we first reconstruct a fully dense particle representation from multiview observations (Fig. 3(0)). Unlike GIC, which uses this representation for physical-parameter identification under a fixed dense interior, we use it as an initialization for hidden-cavity reasoning. Stage I performs gradient-free global hypothesis optimization by instantiating and evaluating cavity hypotheses with forward-only MPM simulation [10, 37] (Fig. 3(1)–(2)). Stage II performs gradient-based Lagrangian particle refinement of the selected hypothesis with differentiable MPM simulation [11] (Fig. 3(3)).

![](images/e1c6713136f4fff505d44e75b7603c15493adc73746b176244c20bf2e51eb5ab.jpg)  
Figure 3. Overview of hypothesis-space reasoning for SfC. (0) Dense particle initialization is reconstructed from multiview observations following GIC [3]. The initial rollout shows the dynamics before hypothesis optimization. (I) Global hypothesis optimization consists of (1) instantiating cavity hypotheses and (2) scoring them with forward-only MPM simulation using a collision-adjacent loss. (II) Lagrangian particle refinement: (3) the selected hypothesis is refined using differentiable MPM simulation with physical and geometric constraints. This two-stage framework separates coarse structural reasoning from fine geometric refinement.

Observation losses. For a set of time steps T, we define

$$
\mathcal { L } _ { \mathrm { 3 D } } ( \mathcal { T } ) = \frac { 1 } { | \mathcal { T } | } \sum _ { t _ { j } \in \mathcal { T } } d _ { \mathrm { C D } } \Big ( \mathcal { S } ( t _ { j } ) , \hat { \mathcal { S } } ( t _ { j } ) \Big ) ,\tag{1}
$$

$$
\mathcal { L } _ { \mathrm { 2 D } } ( \mathcal { T } ) = \frac { 1 } { M | \mathcal { T } | } \sum _ { i = 1 } ^ { M } \sum _ { t _ { j } \in \mathcal { T } } \Big \Vert \mathcal { A } _ { i } ( t _ { j } ) - \hat { \mathcal { A } } _ { i } ( t _ { j } ) \Big \Vert _ { 1 } .\tag{2}
$$

where $d _ { \mathrm { C D } }$ denotes the Chamfer distance (CD), and the predicted surfaces and masks are obtained by simulating and rendering the current particle configuration.

## 4.2. Structural Hypothesis Space

We define a structural hypothesis space H for candidate hidden cavities. A structural hypothesis denotes a candidate cavity configuration evaluated and selected during optimization. Starting from the dense particle initialization $\hat { \mathcal { P } } _ { \mathrm { d e n s e } } ( t _ { 1 } )$ (Fig. 3(0)), a hypothesis $h \in \mathcal H$ specifies an internal void to instantiate (Fig. 3(1)). We parameterize h using compact structural variables, such as a cavity shape representation τ (e.g., an exterior-derived shape or a template index), center position c, size s, and orientation R: $h = ( \tau , c , s , R ) \in \mathcal { H } .$ . Given h, we remove the specified material region from $\hat { \mathcal { P } } _ { \mathrm { d e n s e } } ( t _ { 1 } )$ , yielding the candidate particle configuration $\hat { \mathcal { P } } _ { h } ( t _ { 1 } )$ . Unlike per-particle or per-voxel density representations, H directly parameterizes structural hypothesis variables for the hidden-cavity inference.<sup>2</sup>

Structural assumptions define the hypothesis space by fixing or activating different variables. We consider three settings with different amounts of structural information.

(i) Known size. When the cavity size is specified, the search focuses on the cavity position. The cavity shape is initialized by offsetting inward from the observed external surfaces rather than selected from predefined topology templates.

(ii) A set of candidate templates and known center. When the cavity center and a set of candidate templates are available, the center is fixed, while the template index and remaining configuration variables, such as orientation, are searched.

(iii) Known template and center. When both the cavity template and center are specified, the hypothesis space is restricted to the remaining configuration variables, such as orientation.

These settings yield compact, low-dimensional hypothesis spaces for global exploration prior to particle-level refinement. Their dataset-specific instantiations are described in Section 5.1 and Appendix A.2.

## 4.3. Global Hypothesis Optimization

Given the structural hypothesis space H, the global stage selects a cavity hypothesis by evaluating candidate structures with forward-only simulation (Fig. 3(2)). Because each hypothesis removes material from the dense particle configuration, it induces a discrete structural change that is not well suited to local particle-gradient updates. Therefore, we use gradient-free hypothesis-level optimization, with coarse-to-fine enumeration as the default for stability and reproducibility.

In other words, for each evaluated hypothesis h, we instantiate $\hat { \mathcal { P } } _ { h } ( t _ { 1 } )$ and perform forward-only MPM simulation [10, 37]. We define the base observation score using the 2D mask loss restricted to collision-adjacent frames.

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { g l o b a l } } ( h ) = \mathcal { L } _ { \mathrm { 2 D } } ^ { h } ( \mathcal { T } _ { \mathrm { c o l } } ) , } \end{array}\tag{3}
$$

where $\mathcal { T } _ { \mathrm { c o l } }$ denotes the frames immediately before and after the collision, and superscript h indicates masks rendered from the simulation initialized by hypothesis $h .$ Restricting the observation loss to 2D masks and collision-adjacent frames keeps the global stage focused on structural plausibility and avoids penalizing long-horizon discrepancies that an unrefined hypothesis cannot resolve. For the position search, we augment this base score with a displacement regularizer normalized by the object extent to discourage disproportionately large cavity shifts, as detailed in $\mathsf { A p - }$ pendix A.2.3. We select the lowest-loss hypothesis $h ^ { * }$ and pass the corresponding particle configuration $\hat { \mathcal { P } } _ { h ^ { * } } ( t _ { 1 } )$ to the local-refinement stage.

## 4.4. Lagrangian Particle Refinement

Given the selected configuration $\hat { \mathcal { P } } _ { h ^ { * } } ( t _ { 1 } )$ , the local stage refines particle positions to recover geometric details beyond the hypothesis parameterization (Fig. 3(3)). Unlike the global stage, which evaluates discrete cavity hypotheses, this stage performs continuous particle-level optimization while preserving the selected cavity structure. Following Lagrangian particle optimization [15], we optimize the initial particle positions through differentiable MPM simulation [11], which propagates observation-loss gradients through simulated dynamics.

The local stage uses the full observed sequence $\mathcal { T } _ { \mathrm { a l l } } =$ $\{ t _ { 1 } , \ldots , t _ { N } \}$ :

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { l o c a l } } = \mathcal { L } _ { \mathrm { 3 D } } ( \mathcal { T } _ { \mathrm { a l l } } ) + \lambda _ { \mathrm { 2 D } } \mathcal { L } _ { \mathrm { 2 D } } ( \mathcal { T } _ { \mathrm { a l l } } ) } \\ & { ~ + ~ \lambda _ { \mathrm { r e p } } \mathcal { L } _ { \mathrm { r e p } } + \lambda _ { \mathrm { v o l } } \mathcal { L } _ { \mathrm { v o l } } . } \end{array}\tag{4}
$$

This full-sequence objective complements the collisionadjacent mask loss in the global stage by refining the particle-level geometry and dynamics over the entire trajectory.

To stabilize the refinement around the selected hypothesis, we use two geometry-aware regularization losses. The

repulsive loss is

$$
\mathcal { L } _ { \mathrm { r e p } } = \frac { 1 } { | \mathcal { E } _ { \mathrm { r e p } } | } \sum _ { ( i , j ) \in \mathcal { E } _ { \mathrm { r e p } } } \left( \frac { \left. \hat { p } _ { i } ( t _ { 1 } ) - \hat { p } _ { j } ( t _ { 1 } ) \right. } { d _ { \mathrm { g r i d } } } - 1 \right) ^ { 2 } ,\tag{5}
$$

where $\hat { p } _ { i } ( t _ { 1 } )$ denotes the position of the ith particle in $\hat { \mathcal { P } } ( t _ { 1 } ) , \mathcal { E } _ { \mathrm { r e p } }$ contains nearby particle pairs, and $d _ { \mathrm { g r i d } }$ is the grid spacing. This loss discourages particle collapse and excessive clustering. The volume-preservation loss is

$$
\mathcal { L } _ { \mathrm { v o l } } = \left( \frac { \mathcal { V } ( \hat { \mathcal { P } } ( t _ { 1 } ) ) + \epsilon } { \mathcal { V } ( \hat { \mathcal { P } } _ { \mathrm { r e f } } ( t _ { 1 } ) ) + \epsilon } - 1 \right) ^ { 2 } ,\tag{6}
$$

where $\mathcal { V } ( \cdot )$ denotes the estimated material volume of the particle configuration, $\hat { \mathcal { P } } _ { \mathrm { r e f } } ( t _ { 1 } )$ denotes the reference configuration, and ϵ denotes a small constant. The reference can either be fixed to the selected hypothesis or updated during refinement. We use the latter adaptive variant by default. Together, these losses allow local adjustment while preserving the coarse structure selected in the global stage. Loss-design ablations are provided in Section 5.3.

## 4.5. Implementation and Complexity

Implementation. We kept the hypothesis-grid resolution, MPM time step, refinement steps, and loss weights $\lambda _ { \mathrm { 2 D } }$ $\lambda _ { \mathrm { { r e p } } } .$ , and $\lambda _ { \mathrm { v o l } }$ fixed across experiments unless otherwise stated. We used the physical parameters $\Theta _ { \mathrm { p h y s } }$ from the previous SfC setting [16], such that the evaluation focuses on structural inference rather than material-parameter estimation. In relaxed settings, physical parameters can be included as additional variables, as evaluated in Section 5.6. Additional implementation details are provided in Appendix A.2.

Complexity. The global and local stages have different computational profiles. The global stage evaluates the cavity hypotheses with forward-only MPM simulation [10, 37] on collision-adjacent frames without backpropagation. For an exhaustive grid with n samples per dimension in a ddimensional hypothesis space, the cost scales as $O ( n ^ { d } )$ forward simulations. In our default coarse-to-fine setting, this stage used 250 forward simulations.

The local stage is more expensive per iteration because it performs forward–backward differentiable MPM simulation [11] over the full observed sequence. We used 2,000 Lagrangian particle refinement iterations in the default setting. In a representative timing measurement, the global search accounted for approximately 3% of the local-refinement runtime because each global evaluation is forward-only and uses fewer frames. We provide detailed runtime analysis in Appendix B.8 and optimizer ablations in Appendix B.1.

## 5. Experiments

## 5.1. Experimental Setup

Datasets. We evaluated our method on the standard SfC benchmark [16] and the Topology dataset. The SfC benchmark contains 40 sequences over five external shapes (sphere, cube, bicone, cylinder, and diamond) with four cavity sizes and four cavity positions. The Topology dataset is designed as an interior-topology stress test with 60 configurations grouped into symmetric, asymmetric, and complex subsets. Detailed dataset construction and representative examples are provided in Appendix A.1.

Baselines. Our primary baseline is SfC-NeRF [16], a density-based method that optimizes volumetric densities through differentiable physics simulation [11]. For future prediction, we also compared against GIC [3], which estimates physical parameters under a fixed dense interior, to test the benefit of explicit interior reconstruction.

Metrics. For reconstruction, we reported the CD and Fscore [39] between the reconstructed and ground-truth particle configurations. CD measures the geometric proximity, while the F-score evaluates the thresholded reconstruction consistency. We set the F-score threshold to 0.5% of the maximum bounding-box length and scale CD by 10<sup>3</sup> for readability. For future prediction, we reported the peak signal-to-noise ratio (PSNR) and structural similarity index (SSIM) [44] for rendered future frames.

Default configuration. Unless otherwise stated, we followed the default SfC protocol [16]: the cavity size is known, with the Young’s modulus $E = 1 0 ^ { 6 }$ and Poisson’s ratio $\nu = 0 . 3$ . Each sequence contains N = 14 observed frames rendered from $M = 1 1$ views around the collision event. For our method, we used the known-size setting (i) in Section 4.2, initializing the cavity from the observed external geometry without a ground-truth template. This initialization closely matches centered cavities under size variation, while under position variation, cavities can differ in both position and geometry. Hyperparameters are fixed across datasets, as described in Section 4.5.

## 5.2. Main Results on the SfC Benchmark

We first evaluated hidden-structure reconstruction on the SfC benchmark, which varies the cavity size and position across five external shapes.

Quantitative results. Table 1 reports CD and F-score for the size- and position-variation subsets. Our method outperforms SfC-NeRF in both settings, with larger gains under position variation, where cavity localization is more ambiguous. This supports the failure analysis in Section 3.2. Density-based local optimization can be misled when surface-motion discrepancies do not directly indicate the location of hidden cavities. By evaluating the explicit structural hypotheses through physical simulation, the global stage reduces this ambiguity before Lagrangian particle refinement.

<table><tr><td colspan="3">Size</td><td colspan="2">Position</td></tr><tr><td>Method</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td></tr><tr><td>SfC-NeRF [16]</td><td>0.165</td><td>0.135</td><td>0.281</td><td>0.128</td></tr><tr><td>Ours</td><td>0.063</td><td>0.151</td><td>0.109</td><td>0.144</td></tr></table>

Table 1. Main reconstruction results on the SfC benchmark. Chamfer distance (CD↓) and F-score (F↑) are reported for the cavity size and position variations. CD values are scaled by $1 0 ^ { 3 }$

<table><tr><td>Configuration</td><td>CD↓</td><td>F↑</td></tr><tr><td>(a) Framework stages</td><td></td><td></td></tr><tr><td>Instantiation only</td><td>0.204</td><td>0.136</td></tr><tr><td>+Local refine</td><td>0.156</td><td>0.138</td></tr><tr><td>+Global opt.</td><td>0.121</td><td>0.141</td></tr><tr><td>+Global opt. + Local refine (Full)</td><td>0.109</td><td>0.144</td></tr><tr><td>(b) Geometry-aware losses for local refinement</td><td></td><td></td></tr><tr><td>None</td><td>0.136</td><td>0.124</td></tr><tr><td>Rep. only</td><td>0.124</td><td>0.142</td></tr><tr><td>Rep. + Fixed volume</td><td>0.112</td><td>0.143</td></tr><tr><td>Rep. + Adaptive volume (Full)</td><td>0.109</td><td>0.144</td></tr></table>

Table 2. Ablation study on the SfC benchmark under cavity position variation. (a) Effect of framework stages. (b) Effect of geometry-aware losses during Lagrangian particle refinement. CD values are scaled by $1 0 ^ { 3 }$

Qualitative comparison. Fig. 4(i) shows a representative example of cavity position variation. SfC-NeRF reconstructs a less coherent internal cavity, which can lead to post-collision dynamics that deviate from the observation. In contrast, our method reconstructs a more coherent cavity structure and better matches the observed sequence. Additional qualitative results are available in Appendix C and on the project page.

## 5.3. Ablation Study

We ablated the main components on the SfC benchmark under cavity position variation, where cavity localization is the most ambiguous.

Framework stages. Table 2(a) evaluates the framework stages, starting from the exterior-derived cavity instantiation, which encodes a structural prior. Local refinement improves this initial configuration, while global hypothesis optimization yields a larger improvement by reducing ambiguity in cavity localization. The full pipeline performs the best, indicating that the structural prior, global ambiguity reduction, and local geometric refinement play complementary roles. Fig. 4(ii) shows the same trend.

Geometry-aware losses. Table 2(b) shows that both geometry-aware losses improve the Lagrangian refinement. The repulsive loss stabilizes the particle spacing, and the

![](images/e0ff7e3b20a2323dda0165cf73da34f62ef9e38e731b60ad0bfa0054ed902095.jpg)  
(a) Observation

![](images/b42a2de0754805e2d3c8a6d62eed7b5beee9b2c7a79a14265a02a3069e85a6f7.jpg)  
(b) Ground truth

![](images/680a716956899121f86c4612967e8faeb951a069042964d1256dd5c902227c7e.jpg)  
CD↓ 0.353

![](images/8c78eca638068d3f6f91e6e9e8bcf492cfbe3c72226bf2d32af888eca5a6fe1a.jpg)

![](images/326f775bcfdcbd3900d10eb8cf498693620663f8178f0be376a6511848563f39.jpg)  
CD↓ 0.122

![](images/70b149f2ff539e259c50e5d16ffe6d3fae751037193f1559073f180d4746f0ec.jpg)  
(c) SfC-NeRF

![](images/c9b63e82d3d7ba1b3978bf66838d6810af3c815d8f22ba287a96b5d7751d3b7b.jpg)

(i) Main comparison  
![](images/c350b5cd1c1e98b04c962e147225d29b1c1b01d8e76fbda17395324caedc48ec.jpg)

![](images/a4160b031b9d23ec5add4856ef371f5324b4a3c352473642ddf1a8de6f852c07.jpg)  
(d) Ours  
CD↓ 0.223

![](images/24c49b717b38b6e35af7481e52a5b39ed79a678d74cc263e5bb463df7a0d01b4.jpg)

![](images/f1dd2e37ffbcade3ad98796705aa0c2fd2438c5bc5c38b4be4912b119d10af3f.jpg)  
CD↓ 0.129

![](images/9ccc320914789076cea4f8c9863a6ed39a117d8711aec3c885f72db171354e2a.jpg)  
CD↓ 0.122

![](images/875a45f7afb7b213c17dd071c940c2876a6a11ba283c7d40186d4cbad9cf8699.jpg)

![](images/eb28617f0ddc4eb00a56905bb5b3b3d570728882f9891b2122a74a0e02ca40c5.jpg)

![](images/1a14cc53542aea1d751efc96b2c84d960da3272926bc713dc0c5d4902abd03f4.jpg)  
(e) Instantiation only

![](images/3a9a9d9ee12b20a3cdd59e2a21b60834b3addc2afa4a1107432302659144dd08.jpg)  
(f) +Global opt.  
(ii) Stage-wise ablation

![](images/c458e1c2b0ddce14684c2c319a3511fdc2fb7808b706f46c4a3427d401765f46.jpg)  
(g) +Global opt.  
+ Local refine (Full)

Figure 4. Qualitative results on the SfC benchmark under cavity position variation. (i) Main comparison. Our method reconstructs a more coherent internal cavity than SfC-NeRF and better matches the observed post-collision dynamics. (ii) Stage-wise ablation shows the effects of cavity instantiation, global hypothesis optimization, and Lagrangian particle refinement. The Chamfer distance (CD↓) is reported below each reconstruction. Ground-truth cross-sections are shown in gray.
<table><tr><td>Method</td><td>Setting</td><td>Sym</td><td>Asym</td><td>Comp</td></tr><tr><td>SfC-NeRF [16]</td><td>Size</td><td>0.174</td><td>0.287</td><td>0.728</td></tr><tr><td>Ours</td><td>Size</td><td>0.142</td><td>0.185</td><td>0.582</td></tr><tr><td>Ours</td><td>Cand. w/ center</td><td>0.111</td><td>0.066</td><td>0.083</td></tr><tr><td>Ours</td><td>Template w/ center</td><td>0.090</td><td>0.066</td><td>0.070</td></tr></table>

Table 3. Interior-topology stress tests under different hypothesisspace settings. We report CD↓ scaled by $1 0 ^ { 3 }$ on symmetric (Sym), asymmetric (Asym), and complex (Comp) subsets. Size denotes the known-size setting, Cand. w/ center uses a set of candidate templates and a known center, and Template w/ center uses a known template and center. The upper rows compare methods under the known-size setting, while the lower rows vary structural assumptions. F-score results are reported in Appendix B.2.1.

volume-preservation loss further improves the reconstruction, with the adaptive variant performing the best.

## 5.4. Interior-Topology Stress Tests

We used the Topology dataset to stress-test reconstruction of more complex and ambiguous interior topologies than those in the standard SfC benchmark. Table 3 shows how structural assumptions encoded in H affect reconstruction. The candidate-template setting includes the ground-truth template and assumes a known center, whereas the knowntemplate setting provides both the ground-truth template and center. These controlled settings are stress tests rather than evidence of arbitrary topology discovery.

Known-size comparison. Under the known-size setting, our method outperforms SfC-NeRF across all topology subsets. The error remains relatively high for Comp, indicating that the detailed reconstruction of complex cavities from collision observations alone is still difficult.

Effect of structural assumptions. The lower rows show how changing H affects the result. A set of candidate templates with a known center enables the global stage to compare structural alternatives before refinement, while a known template and center leave orientation as the main ambiguity. These rows are not directly comparable because the settings provide different amounts of structural information. Rather, they show how the same inference framework incorporates different structural assumptions through the hypothesis space.

## 5.5. Future Prediction

A recovered interior structure should support predictive simulation. We evaluated future prediction on the SfC benchmark under cavity position variation, where incorrect cavity localization strongly affects post-collision dynamics. Each method used N = 14 observed frames and was evaluated at +7 and +14 future frames. GIC and both variants of our method used the same GIC-based rendering pipeline, including the re-optimization of Gaussian attributes described in Appendix A.2.5, while SfC-NeRF used its original PAC-NeRF-based rendering pipeline. Additional results on the SfC and Topology datasets are provided in Appendix B.4.

Comparison with baselines. Table 4 shows that our full method outperforms GIC and SfC-NeRF at both prediction horizons, particularly at the longer horizon. The stronger long-horizon performance suggests a benefit from explicit hidden-structure reconstruction for predictive simulation. We also note that the comparison with SfC-NeRF reflects differences in the rendering pipeline.

Effect of local refinement. Lagrangian particle refinement corrects residual geometric discrepancies after the global stage and substantially improves prediction at both horizons. Larger gains in future-frame metrics than in CD may arise because CD measures only the initial reconstructed geometry, while future-frame metrics also reflect the simulated trajectory and can be more sensitive to local geometric differences. Together with the reconstruction ablation in Section 5.3, these results support the complementary roles of the global and local stages.

<table><tr><td rowspan="2">Method</td><td colspan="2">+7 frames</td><td colspan="2">+14 frames</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td></tr><tr><td>GIC [3]</td><td>23.1</td><td>0.977</td><td>22.1</td><td>0.974</td></tr><tr><td>SfC-NeRF [16]</td><td>27.1</td><td>0.982</td><td>25.2</td><td>0.978</td></tr><tr><td>Ours (Global only)</td><td>24.9</td><td>0.980</td><td>23.6</td><td>0.977</td></tr><tr><td>Ours (Full)</td><td>33.6</td><td>0.991</td><td>30.8</td><td>0.988</td></tr></table>

Table 4. Future prediction on the SfC benchmark under cavity position variation. PSNR and SSIM are reported at +7 and +14 frames beyond the observed window of N = 14 frames.

<table><tr><td rowspan="3">Condition</td><td>Size</td><td colspan="2">Position</td></tr><tr><td>CD↓ F↑</td><td>CD↓</td><td>F↑</td></tr><tr><td>0.067</td><td>0.151</td><td>0.087</td></tr><tr><td>SfC-NeRF [16] Ours (Clean)</td><td>0.059</td><td>0.093 0.097</td><td>0.100 0.093</td></tr><tr><td>(a) Optimizer</td><td></td><td></td><td></td></tr><tr><td>Ours (Random search)</td><td>0.059</td><td>0.098 0.097</td><td>0.101 0.093</td></tr><tr><td>Ours (CMA-ES [9])</td><td>0.059</td><td>0.102</td><td>0.093</td></tr><tr><td>(b) Physical parameters</td><td></td><td></td><td></td></tr><tr><td>Ours (E, ±10% error)</td><td>0.061</td><td>0.096</td><td>0.100 0.093</td></tr><tr><td>Ours (ν, ±10% error)</td><td>0.059</td><td>0.097 0.101</td><td>0.093</td></tr><tr><td>Ours (Unknown phys. params.) (c) Observations</td><td>0.060</td><td>0.097</td><td>0.119 0.091</td></tr><tr><td>Ours (Motion blur)</td><td></td><td></td><td></td></tr><tr><td>Ours (Half FPS)</td><td>0.064 0.061</td><td>0.095 0.096 0.096</td><td>0.093 0.112 0.091</td></tr></table>

Table 5. Robustness results on spherical objects under cavity size and position variations. (a) Optimizer variants use the same global objective and simulation budget as the default grid enumeration. (b) Physical-parameter errors are averaged over positive and negative perturbations; unknown physical parameters are jointly estimated. (c) Motion blur uses shutter angle 0.5, and half FPS halves the observation frame rate. CD values are scaled by 10<sup>3</sup>.

## 5.6. Robustness and Practical Considerations

We evaluated the robustness of spherical objects from the SfC benchmark under cavity size and position variations, isolating internal-structure ambiguity from external-shape differences. Table 5 summarizes the representative results for optimizer changes, physical-parameter uncertainty, and degraded observations.

Optimizer sensitivity. The random search and Covariance Matrix Adaptation Evolution Strategy (CMA-ES) match the default grid enumeration within 0.002 CD, with nearly identical F-scores, suggesting limited sensitivity to the choice of gradient-free optimizer. Additional ablations are provided in Appendix B.1.

Toward less controlled settings. Moderate errors in E or ν and motion blur have little effect. Joint physical-parameter estimation remains close to the clean setting for size variation but causes moderate degradation for position variation. Reducing the frame rate also moderately degrades performance in the position subset. Additional robustness analyses are provided in Appendices B.5–B.7.

## 6. Limitations and Discussion

This study has several limitations. First, our experiments used controlled synthetic data with calibrated cameras and a known simulation model. Although Section 5.6 evaluates motion blur and reduced frame rate, real-world capture can introduce additional errors, such as calibration noise and forward-model mismatch. The evaluation on in-the-wild collision sequences remains a topic for future work.

Second, the exact recovery of complex interior topology remains ill-posed. As shown in Section 5.4, collision observations leave residual ambiguities under weak structural assumptions, particularly for intricate geometries. Hypothesis-space reasoning makes this trade-off explicit but cannot resolve insufficient observations or structures that are not adequately represented in the hypothesis space. In the latter case, the global stage can only select the best available mismatched hypothesis, while Lagrangian particle refinement adjusts the geometry but preserves the cavity structure. Thus, substantial topology changes must be represented in the hypothesis space before refinement.

Future work will include richer structural variables, such as multiple disconnected cavities, anisotropic scaling, and affine transformations, possibly combined with learned shape priors. CMA-ES performs comparably to coarse-tofine enumeration in the current low-dimensional settings, while adaptive sampling may become useful as the hypothesis space grows. Extending the framework to interactions such as squeezing or shaking may also support internalstructure inference from everyday object dynamics.

## 7. Conclusion

We presented SfC as hypothesis-space reasoning, replacing implicit density updates with explicit structural hypotheses. Our two-stage framework combines global hypothesis optimization and Lagrangian particle refinement, separating coarse structural reasoning from fine geometric adjustment. Experiments showed improved reconstruction and future prediction over the density-based SfC baseline, while topology stress tests and robustness analyses clarified its scope and limitations. This framework supports the structured inference of the hidden interior structure from collisioninduced dynamics.

## References

[1] Jad Abou-Chakra, Feras Dayoub, and Niko Sunderhauf. Par-¨ ticleNeRF: A particle-based encoding for online neural radiance fields. In WACV, 2024. 2

[2] Jad Abou-Chakra, Krishan Rana, Feras Dayoub, and Niko Sunderhauf. Physically embodied Gaussian splatting: A vi-¨ sually learnt and physically grounded 3D representation for robotics. In CoRL, 2024. 2

[3] Junhao Cai, Yuji Yang, Weihao Yuan, Yisheng He, Zilong Dong, Liefeng Bo, Hui Cheng, and Qifeng Chen. GIC: Gaussian-informed continuum for physical property identification and simulation. In NeurIPS, 2024. 2, 3, 4, 6, 8, 11, 12, 14, 17

[4] Mengyu Chu, Lingjie Liu, Quan Zheng, Erik Franz, Hans-Peter Seidel, Christian Theobalt, and Rhaleb Zayer. Physics informed neural fields for smoke reconstruction with sparse data. ACM Trans. Graph., 41(4), 2022. 2

[5] Yutao Feng, Yintong Shang, Xuan Li, Tianjia Shao, Chenfanfu Jiang, and Yin Yang. PIE-NeRF: Physics-based interactive elastodynamics with NeRF. In CVPR, 2024. 2

[6] Yasutaka Furukawa and Jean Ponce. Accurate, dense, and robust multiview stereopsis. IEEE Trans. Pattern Anal. Mach. Intell., 32(8):1362–1376, 2009. 2

[7] Guy Gafni, Justus Thies, Michael Zollhofer, and Matthias Nießner. Dynamic neural radiance fields for monocular 4D facial avatar reconstruction. In CVPR, 2021. 2

[8] Shanyan Guan, Huayu Deng, Yunbo Wang, and Xiaokang Yang. NeuroFluid: Fluid dynamics grounding with particledriven neural radiance fields. In ICML, 2022. 2

[9] Nikolaus Hansen and Andreas Ostermeier. Completely derandomized self-adaptation in evolution strategies. Evol. Comput., 9(2):159–195, 2001. 8, 13, 14

[10] Yuanming Hu, Yu Fang, Ziheng Ge, Ziyin Qu, Yixin Zhu, Andre Pradhana, and Chenfanfu Jiang. A moving least squares material point method with displacement discontinuity and two-way rigid body coupling. ACM Trans. Graph., 37(4), 2018. 2, 3, 4, 5, 11, 13, 20

[11] Yuanming Hu, Luke Anderson, Tzu-Mao Li, Qi Sun, Nathan Carr, Jonathan Ragan-Kelley, and Fredo Durand. Diff-´ Taichi: Differentiable programming for physical simulation. In ICLR, 2020. 2, 4, 5, 6, 12, 14, 20

[12] Tianyu Huang, Haoze Zhang, Yihan Zeng, Zhilu Zhang, Hui Li, Wangmeng Zuo, and Rynson W. H. Lau. DreamPhysics: Learning physics-based 3D dynamics with video diffusion priors. In AAAI, 2025. 2

[13] Hanxiao Jiang, Hao-Yu Hsu, Kaifeng Zhang, Hsin-Ni Yu, Shenlong Wang, and Yunzhu Li. PhysTwin: Physicsinformed reconstruction and simulation of deformable objects from videos. In ICCV, 2025. 2

[14] Ying Jiang, Chang Yu, Tianyi Xie, Xuan Li, Yutao Feng, Huamin Wang, Minchen Li, Henry Lau, Feng Gao, Yin Yang, and Chenfanfu Jiang. VR-GS: A physical dynamicsaware interactive Gaussian splatting system in virtual reality. In SIGGRAPH, 2024. 2

[15] Takuhiro Kaneko. Improving physics-augmented continuum neural radiance field-based geometry-agnostic system iden-

tification with Lagrangian particle optimization. In CVPR, 2024. 2, 5, 13

[16] Takuhiro Kaneko. Structure from collision. In CVPR, 2025. 1, 2, 3, 5, 6, 7, 8, 11, 12, 15, 17, 18

[17] Kai Katsumata, Duc Minh Vo, and Hideki Nakayama. A compact dynamic 3D Gaussian representation for real-time dynamic view synthesis. In ECCV, 2024. 2

[18] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler,¨ and George Drettakis. 3D Gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4), 2023. 1, 2, 12, 16

[19] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015. 14

[20] Xuan Li, Yi-Ling Qiao, Peter Yichen Chen, Krishna Murthy Jatavallabhula, Ming Lin, Chenfanfu Jiang, and Chuang Gan. PAC-NeRF: Physics augmented continuum neural radiance fields for geometry-agnostic system identification. In ICLR, 2023. 2, 16, 18

[21] Zhengqi Li, Simon Niklaus, Noah Snavely, and Oliver Wang. Neural scene flow fields for space-time view synthesis of dy namic scenes. In CVPR, 2021. 2

[22] Shanchuan Lin, Andrey Ryabtsev, Soumyadip Sengupta, Brian L. Curless, Steven M. Seitz, and Ira Kemelmacher-Shlizerman. Real-time high-resolution background matting. In CVPR, 2021. 3

[23] Chunjiang Liu, Xiaoyuan Wang, Qingran Lin, Albert Xiao, Haoyu Chen, Shizheng Wen, Hao Zhang, Lu Qi, Ming-Hsuan Yang, Laszlo A. Jeni, Min Xu, and Yizhou Zhao. Multi-object system identification from videos. In ICLR, 2026. 2

[24] Fangfu Liu, Hanyang Wang, Shunyu Yao, Shengjun Zhang, Jie Zhou, and Yueqi Duan. Physics3D: Learning physical properties of 3D Gaussians via video diffusion. arXiv preprint arXiv:2406.04338, 2024. 2

[25] Zhuoman Liu, Weicai Ye, Yan Luximon, Pengfei Wan, and Di Zhang. Unleashing the potential of multi-modal foundation models and video diffusion for 4D dynamic physical scene simulation. In CVPR, 2025. 2

[26] Zhicheng Lu, Xiang Guo, Le Hui, Tianrui Chen, Min Yang, Xiao Tang, Feng Zhu, and Yuchao Dai. 3D geometry-aware deformable Gaussian splatting for dynamic view synthesis. In CVPR, 2024. 2

[27] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and Deva Ramanan. Dynamic 3D Gaussians: Tracking by per sistent dynamic view synthesis. In 3DV, 2024. 2

[28] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. NeRF: Representing scenes as neural radiance fields for view syn thesis. In ECCV, 2020. 1, 2

[29] Junfeng Ni, Yixin Chen, Bohan Jing, Nan Jiang, Bin Wang, Bo Dai, Puhao Li, Yixin Zhu, Song-Chun Zhu, and Siyuan Huang. PhyRecon: Physically plausible neural scene reconstruction. In NeurIPS, 2024. 2

[30] Keunhong Park, Utkarsh Sinha, Jonathan T. Barron, Sofien Bouaziz, Dan B. Goldman, Steven M. Seitz, and Ricardo Martin-Brualla. Nerfies: Deformable neural radiance fields. In ICCV, 2021. 2

[31] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-NeRF: Neural radiance fields for dynamic scenes. In CVPR, 2021. 2

[32] Ri-Zhao Qiu, Ge Yang, Weijia Zeng, and Xiaolong Wang. Feature Splatting: Language-driven physics-based scene synthesis and editing. In ECCV, 2024. 2

[33] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Radle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junt-¨ ing Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollar, and Christoph Feicht-´ enhofer. SAM 2: Segment anything in images and videos. In ICLR, 2025. 3

[34] Steven M. Seitz, Brian Curless, James Diebel, Daniel Scharstein, and Richard Szeliski. A comparison and evaluation of multi-view stereo reconstruction algorithms. In CVPR, 2006. 2

[35] Farhat Shaikh, Ayan Banerjee, and Sandeep Gupta. EMMA: Extracting multiple physical parameters from multimodal data. In CVPR, 2026. 2

[36] Noah Snavely, Steven M. Seitz, and Richard Szeliski. Photo tourism: Exploring photo collections in 3D. ACM Trans. Graph., 25(3):835–846, 2006. 2

[37] Deborah Sulsky, Zhen Chen, and Howard L. Schreyer. A particle method for history-dependent materials. Comput. Methods Appl. Mech. Eng., 118(1-2):179–196, 1994. 2, 3, 4, 5, 13, 20

[38] Cheng Sun, Min Sun, and Hwann-Tzong Chen. Direct voxel grid optimization: Super-fast convergence for radiance fields reconstruction. In CVPR, 2022. 16

[39] Maxim Tatarchenko, Stephan R. Richter, Rene Ranftl,´ Zhuwen Li, Vladlen Koltun, and Thomas Brox. What do single-view 3D reconstruction networks learn? In CVPR, 2019. 6

[40] Carlo Tomasi and Takeo Kanade. Shape and motion from image streams under orthography: a factorization method. Int. J. Comput. Vis., 9(2):137–154, 1992. 2

[41] Edgar Tretschk, Ayush Tewari, Vladislav Golyanik, Michael Zollhofer, Christoph Lassner, and Christian Theobalt. Non-¨ rigid neural radiance fields: Reconstruction and novel view synthesis of a dynamic scene from monocular video. In ICCV, 2021. 2

[42] Federico Vasile, Ri-Zhao Qiu, Lorenzo Natale, and Xiaolong Wang. Gaussian-augmented physics simulation and system identification with complex colliders. In NeurIPS, 2025. 2

[43] Norika Wada, Kohei Yamashita, Ryo Kawahara, and Ko Nishino. M-PhyGs: Multi-material object dynamics from video. In CVPR Findings, 2026. 2

[44] Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. Image quality assessment: From error visibility to structural similarity. IEEE Trans. Image Process., 13(4), 2004. 6, 12

[45] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4D Gaussian splatting for real-time dynamic scene rendering. In CVPR, 2024. 2, 3

[46] Tianyi Xie, Zeshun Zong, Yuxing Qiu, Xuan Li, Yutao Feng, Yin Yang, and Chenfanfu Jiang. PhysGaussian: Physicsintegrated 3D Gaussians for generative dynamics. In CVPR, 2024. 2

[47] Jinbo Yan, Rui Peng, Zhiyan Wang, Luyang Tang, Jiayu Yang, Jie Liang, Jiahao Wu, and Ronggang Wang. Instant Gaussian stream: Fast and generalizable streaming of dy namic scene reconstruction via Gaussian splatting. In CVPR, 2025. 2

[48] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3D Gaussians for high-fidelity monocular dynamic scene reconstruction. In CVPR, 2024. 2, 3

[49] Zeyu Yang, Hongye Yang, Zijie Pan, and Li Zhang. Real time photorealistic dynamic scene representation and render ing with 4D Gaussian splatting. In ICLR, 2024. 2

[50] Tianyuan Zhang, Hong-Xing Yu, Rundi Wu, Brandon Y. Feng, Changxi Zheng, Noah Snavely, Jiajun Wu, and William T. Freeman. PhysDreamer: Physics-based interaction with 3D objects via video generation. In ECCV, 2024. 2

[51] Yizhou Zhao, Haoyu Chen, Chunjiang Liu, Zhenyang Li, Charles Herrmann, Junhwa Hur, Yinxiao Li, Ming-Hsuan Yang, Bhiksha Raj, and Min Xu. Toward material-agnostic system identification from videos. In ICCV, 2025. 2

[52] Licheng Zhong, Hong-Xing Yu, Jiajun Wu, and Yunzhu Li. Reconstruction and simulation of elastic objects with spring mass 3D Gaussians. In ECCV, 2024. 2

## Supplementary Material

A. Implementation Details 11   
A.1. Dataset 11   
A.2. Model Details . 11   
B. Additional Experiments and Analysis 14   
B.1. Global Optimizer and Refinement Ablation . 14   
B.2. Additional Interior-Topology Results 15   
B.3. Effect of Initialization Source 16   
B.4. Additional Future Prediction Results . 16   
B.5. Robustness under Relaxed Assumptions . 16   
B.6. Robustness to Physical-Parameter Errors 18   
B.7. Robustness to Material Variations . 18   
B.8. Runtime Analysis . 20   
C. Qualitative Results 21   
C.1. SfC Benchmark . 21   
C.2. Interior-Topology Stress Tests . 21

## A. Implementation Details

## A.1. Dataset

## A.1.1. SfC Dataset

The SfC dataset [16] contains five external shapes: sphere, cube, bicone, cylinder, and diamond. The dataset includes two structural variation subsets used in our main experiments: cavity size and cavity position. In the size variation subset, cavity sizes vary over $s _ { c } \in$ $\left\{ 0 , \left( { \frac { 1 } { 2 } } \right) ^ { 3 } , \left( { \frac { 2 } { 3 } } \right) ^ { 3 } , \left( { \frac { 3 } { 4 } } \right) ^ { 3 } \right\}$ with the cavity center held fixed. In the position variation subset, cavity locations vary over $l _ { c } ~ \in ~ \{ 1 \mathrm { e f t } $ , right, up, down} with the cavity size fixed to $\begin{array} { l l l } { s _ { c } } & { = } & { \left( { \frac { 2 } { 3 } } \right) ^ { 3 } } \end{array}$ . All objects are composed of elastic material with Young’s modulus $E \ : = \ : 1 0 ^ { 6 }$ and Poisson’s ratio $\nu = 0 . 3 ,$ , following the original SfC simulation setting. In total, the structural benchmark comprises 5 external shapes × (4 size settings + 4 position settings) = 40 sequences. Representative examples are shown in Fig. 5.

Ground-truth dynamics are simulated using the Moving Least Squares Material Point Method (MLS-MPM) simulator [10], in which each object is dropped under gravity and allowed to collide with a ground plane. Each sequence contains 14 observed frames. For each deformation state, multiview images are rendered from 11 viewpoints arranged on the upper hemisphere surrounding the object. Images are rendered with diverse environmental lighting conditions and ground textures to increase visual diversity.

## A.1.2. Topology Dataset

We used the Topology dataset to stress-test reconstruction under more complex interior topologies. Unlike the SfC dataset, which varies cavity size and position, this dataset varies the internal cavity geometry and topology. It is designed to study how structural assumptions encoded in the hypothesis space affect reconstruction, rather than to evaluate arbitrary topology discovery. The dataset contains three subsets—symmetric, asymmetric, and complex—each consisting of four cavity shapes. Representative examples are shown in Fig. 6.

Symmetric set. The symmetric set contains cavity shapes with discrete rotational symmetry about the object center. It includes cube, cross, snowman, and sun. The cube defines an axis-aligned rectangular cavity. The cross is composed of three mutually orthogonal bars. The snowman consists of two equal spheres arranged along a single axis, exhibiting uniaxial symmetry. The sun extends conical protrusions in 26 discrete directions from a central sphere. These shapes provide controlled cases for evaluating the effect of symmetric cavity geometry on reconstruction.

Asymmetric set. The asymmetric set contains cavity shapes that break symmetry and introduce a preferred orientation. It includes pacman, crescent, unequal snowman, and frustum. The pacman is derived from a sphere by removing an angular wedge, while the crescent is obtained by subtracting a shifted overlapping sphere. The unequal snowman combines two spheres of different radii to form an elongated asymmetric structure, and the frustum defines a truncated cone with a tapered cross-section. These shapes test sensitivity to orientation cues in hidden cavity geometry.

Complex set. The complex set contains cavity shapes with non-trivial topology or spatial discontinuity. It includes Cshape, spherical ring, split sphere, and triple tunnel. The C-shape forms a partially enclosed shell; the spherical ring retains a cylindrical core within a spherical cavity; and the split sphere produces two disjoint half-spherical cavities displaced along a single axis. The triple tunnel embeds three orthogonal cylindrical channels within a spherical cavity, yielding a genus-3 topology. These shapes stresstest reconstruction under high curvature and topological complexity.

All configurations use a spherical exterior and fixed cavity size $s _ { c } \ = \ \left( { \textstyle { \frac { 2 } { 3 } } } \right) ^ { 3 }$ . We generated five random rotations for each shape, yielding 4 shapes × 3 subsets × 5 rotations = 60 configurations. Simulation and rendering follow the SfC dataset settings, with 14 observed frames and 11 upperhemisphere views per sequence.

## A.2. Model Details

An overview of the proposed pipeline is shown in Fig. 3 of the main paper. We describe the implementation details of each step below.

## A.2.1. Step (0). Dense Initialization

We initialized the object with a fully dense particle representation using GIC [3]. GIC first reconstructs a dynamic

![](images/a147511b3f2660a9e80bc43ad3338f2bcd442b0d7a2a330a149d25033457317b.jpg)  
(a) Sphere

![](images/cb36a6b274684820643a3224647f47b7d94c64ead24f5ce3a42624b2eb2ca4c2.jpg)

![](images/2d91bb1269f3e1cb6dede61d4eebce67fd8445261ce23e364217b2ca9903b818.jpg)  
(b) Cube

![](images/656a9c8b78fca1d2e07ab5b7aa31b685228a1ce4771b94c4368f426486ec4dfb.jpg)  
(c) Bicone

![](images/bb5033de4ea744ed13c451fdbbf0154727908c8fc8019e97f8b40f36d0b0ccd9.jpg)  
(d) Cylinder

![](images/93f956782b39fcdbe77f691417542d51510769bcb26f4b4b0b2e5d817067e12a.jpg)

![](images/532dd11dbb839cecd2f7f309677b7babaa13af54fea601b63c28be341622fad4.jpg)  
(e) Diamond

Figure 5. Overview of the SfC dataset. For each object, we show a representative observation image and the corresponding ground-truth cross-section. The dataset contains five external shapes ((a) sphere, (b) cube, (c) bicone, (d) cylinder, and (e) diamond) with cavity size and position variations.  
![](images/1f6930aeee3202bb4efe67be9eeeff3b0410a0c876ddf1cbd658bff9018c5b66.jpg)  
(a) Cube

![](images/8ba411ad1f1a6c58f2d2597dfb0e6abbb26a36d7dc2d0e30717d73c3f299fea4.jpg)  
(b) Cross

![](images/64ad5e68dc2aa169e724fd2283b053c5f78ab7ca66e835bf44cd8f8a16ff5fbb.jpg)  
(c) Snowman

![](images/008a897fd5ab05341c65bf6d4059582076d5fdf875688e9bc2d6b6c3cb7c1859.jpg)  
(d) Sun

![](images/8d7f396545e1102edae05bf5c768ade109ec14e2aa98abfa8bf72676653db460.jpg)  
(e) Pacman

![](images/bdec458b0a6a37d09f2fdedccdc8c00101a585ce5f63f58fa28192b06180e85a.jpg)  
(f) Crescent

(a) Symmetric  
![](images/311d0a2cf84eb9ad4113bc283edd765dff7303bdc69536323ad6cfb100d23456.jpg)  
(g) Unequal snowman

![](images/5942b6c3c26c1c9b4f75c3816876d2b360d7850062e480d96075f85e4590b83c.jpg)  
(h) Frustum  
(b) Asymmetric

(i) C-shape  
![](images/b6e3739ed60d260b339e5b3bf9e6d347d229876e1aee57e6cac982e5446876ed.jpg)

![](images/63e7353904c88e986d96c918f2b1d79921e07c98079ba29aafe09470f13fa015.jpg)  
(j) Spherical ring

(c) Complex  
![](images/0bac92b08fd2e5ff5eb0691cfc918bd441220a962660d050d2b996a6905782a2.jpg)  
(k) Split sphere

![](images/c43fbd8a7955b4e29fab963e89f14cea5af68eab3b3d8f2f5a438b0bb3204efb.jpg)  
(l) Triple tunnel  
Figure 6. Overview of the Topology dataset. For each shape, we show a representative observation image and the corresponding groundtruth cross-section. The dataset contains three subsets: (I) symmetric ((a) cube, (b) cross, (c) snowman, (d) sun), (II) asymmetric ((e) pac man, (f) crescent, (g) unequal snowman, (h) frustum), and (III) complex ((i) C-shape, (j) spherical ring, (k) split sphere, (l) triple tunnel).

3D Gaussian representation from multiview observations and then converts it into a continuum particle representation for MPM simulation. Each continuum particle is associated with a position, scale, and density value, which are used for simulation and mask rendering. We used the official GIC implementation and followed its default training configuration. Below, we describe the GIC-based initialization procedure, including dynamic 3D reconstruction, continuum generation, physical simulation, and velocity estimation.

This yielded a final grid spacing of $d _ { \mathrm { g r i d } } = 0 . 1 6 / 2 ^ { 4 } = 0 . 0 1$ This spacing matched the average inter-particle distance used in SfC-NeRF [16], so that the comparison was not affected by particle density. In practice, this resolution produced approximately 10<sup>5</sup> particles for a fully dense object before cavity removal.

Dynamic 3D reconstruction. GIC reconstructs dynamic scenes using a canonical set of 3D Gaussians and a learned deformation field that maps the canonical configuration to each time step. The model is trained with L1 and SSIM [44] losses, following the architecture and hyperparameters of the official GIC implementation.

Physical simulation. We used a differentiable MPM simulator implemented with DiffTaichi [11]. During simulation, object silhouettes are rendered as alpha masks using 3DGS [18]. These masks are compared with the observed image masks to compute the 2D observation loss. The simulation parameters, including the material model and timestep settings, follow those used in SfC-NeRF [16] for a fair comparison.

Continuum generation. From the reconstructed Gaussian representation, we generated a fully dense grid-based point set using a coarse-to-fine strategy. The initial grid size was set to $d _ { \mathrm { g r i d } } ^ { \mathrm { 0 } } = 0 . 1 6 ,$ , and the grid was upsampled four times.

Velocity estimation. We estimated the pre-collision falling velocity following the procedure used in GIC [3]. We used four pre-impact frames with the default hyperparameters from the official GIC implementation.

## A.2.2. Step (1). Hypothesis Instantiation

Starting from the dense particle initialization, we instantiated candidate cavity hypotheses using the structural parameterization described in Section 4.2 of the main paper. Each hypothesis is specified by compact structural variables, including a cavity shape representation, center position, size, and orientation. The cavity shape representation may be derived from the observed external geometry or selected from a set of templates, depending on the available structural assumptions. We removed particles inside the resulting cavity region to form the corresponding internal structure. Structural assumptions fix or restrict a subset of these variables, while the remaining variables are optimized in Step (2).

## A.2.3. Step (2). Global Hypothesis Optimization

The global stage evaluates candidate cavity hypotheses using forward-only MPM simulation [10, 37] and selects the lowest-loss hypothesis.

Default optimizer. Unless otherwise stated, we used coarse-to-fine enumeration as the default gradient-free optimizer. This choice is deterministic, reproducible, and consistent with the main experiments.

Position-search regularization. For position search, we augmented ${ \mathcal { L } } _ { \mathrm { g l o b a l } } ( h )$ with the displacement regularizer

$$
\mathcal { R } _ { \mathrm { d i s p } } ( h ) = e ^ { - k _ { \mathrm { d i s p } } r _ { h } } \left\| \widetilde { \boldsymbol { \Delta } } _ { h } \right\| _ { 2 } ^ { 2 } ,\tag{7}
$$

where $\widetilde { \Delta } _ { h }$ denotes the cavity displacement from its initial placement in voxel-index units, normalized element-wise by the index span of the object mask along each axis, with each span lower-bounded by one. The cavity-volume ratio $r _ { h }$ is the density volume inside the shifted cavity divided by that of the original object. Hypotheses whose shifted cavity reaches the object boundary are not rendered and are treated as invalid during hypothesis selection. We minimized ${ \mathcal { L } } _ { \mathrm { g l o b a l } } ( h ) + w _ { \mathrm { d i s p } } { \mathcal { R } } _ { \mathrm { d i s p } } ( h )$ with $k _ { \mathrm { d i s p } } = 5$ and $w _ { \mathrm { d i s p } } = 2$ This regularizer discourages cavity displacements that are disproportionately large relative to the object span. The penalty is measured from the initial cavity placement, with its weight decreasing as the cavity-volume ratio increases.

For the unknown-size experiment, we instead used $w _ { \mathrm { d i s p } } = 1$ and added ${ w _ { \mathrm { s i z e } } \mathcal { R } _ { \mathrm { s i z e } } ( h ) }$ to the objective. The size regularizer is

$$
\mathcal { R } _ { \mathrm { s i z e } } ( h ) = \frac { e ^ { k _ { \mathrm { s i z e } } r _ { h } } - 1 } { e ^ { k _ { \mathrm { s i z e } } } - 1 } ,\tag{8}
$$

with $k _ { \mathrm { s i z e } } = 5$ and $w _ { \mathrm { s i z e } } = 0 . 1$ . This regularizer penalizes large cavity-volume ratios and is normalized to [0, 1].

Setting-specific search variables. The optimized variables and search ranges depend on the hypothesis-space setting, as detailed below.

(i) Known size. In this setting, we constructed the cavity by offsetting inward from the observed external surface without using a ground-truth template. The resulting cavity shape and the specified size are fixed, and the default search optimizes only its position. The default grid optimizer first performs a coarse $5 \times 5 \times 5$ position search over the three Cartesian axes of the object interior, followed by a refined $5 \times 5 \times 5$ search around the best candidate. This evaluates 250 hypotheses in total (125 coarse and 125 refined), each requiring one forward-only simulation.

(ii) A set of candidate templates and known center. In this setting, the search considers the four templates used to construct the corresponding topology subset and therefore includes the ground-truth template for each evaluated object. The coarse stage jointly evaluates the template and orientation by combining all candidate templates with 125 orientations sampled on the sphere using Fibonacci sphere sampling. The fine stage fixes the template selected in the coarse stage and refines the orientation using 125 cone directions around the selected coarse direction.

(iii) Known template and center. In this setting, we provided the ground-truth template used to construct the evaluated object and its cavity center, leaving orientation as the remaining search variable. We used the same coarse-to-fine orientation search as in setting (ii): 125 Fibonacci-sphere samples in the coarse stage, followed by 125 cone-direction samples around the selected coarse direction.

Alternative global optimizers. We also evaluated alternative gradient-free optimizers for the global stage to investigate optimizer sensitivity (Section 5.6 and Appendix B.1). For random search in the position-search setting, the coarse grid enumeration is replaced with uniform random sampling over the same search ranges, while the fine search follows the default grid refinement around the best coarse candidate. For Covariance Matrix Adaptation Evolution Strategy (CMA-ES) [9], the optimizer proposes hypotheses within the same parameter ranges and uses the specified evaluation budget without a separate coarse/fine split. In the main-paper optimizer-sensitivity evaluation, all optimizer variants are followed by the same Lagrangian particle refinement stage.

Relaxed-assumption settings. In relaxed-assumption settings (Section 5.6 and Appendix B.5), we expanded the hypothesis space with additional continuous variables, such as cavity size or physical parameters, to evaluate robustness under relaxed assumptions. Table 6 summarizes their search ranges and grid resolutions.

## A.2.4. Step (3). Lagrangian Particle Refinement

After global hypothesis optimization selected a cavity hypothesis, we refined the corresponding particle configuration following Lagrangian particle optimization [15]. Starting from $\hat { \mathcal { P } } _ { h ^ { * } } ( t _ { 1 } )$ , the local stage treats the particles as a Lagrangian system and optimizes their initial positions through differentiable MPM simulation [11]. Observationloss gradients are backpropagated through the full simulated sequence, allowing particle-level refinement under the physical constraints of the simulator.

<table><tr><td>Parameter</td><td>Range</td><td>Coarse</td><td>Fine</td></tr><tr><td>Cavity size sc</td><td> $[ 0 , 0 . 8 ]$ </td><td>5</td><td>5</td></tr><tr><td>Young&#x27;s modulus E</td><td> $[ 5 \times \dot { 1 0 } ^ { 5 } , 5 \times \dot { 1 0 } ^ { 6 } ]$ </td><td>5</td><td>5</td></tr><tr><td>Poisson&#x27;s ratio ν</td><td>[0.1, 0.4]</td><td>5</td><td>5</td></tr></table>

Table 6. Search ranges and grid resolutions for additional continuous variables in relaxed-assumption experiments. Each parameter is explored using coarse-to-fine grid enumeration with the number of candidates shown for each stage. The candidate sets exclude the exact ground-truth values to avoid evaluating a trivially correct hypothesis.

Unless otherwise stated, the objective follows Eq. 4 in the main paper and combines the 3D surface loss, 2D mask loss, repulsive loss, and volume-preservation loss. For the repulsive loss, neighboring particle pairs in $\mathcal { L } _ { \mathrm { r e p } }$ are defined within a radius of $1 . 5 d _ { \mathrm { g r i d } }$ in the initial configuration. For the volume-preservation loss, $\mathcal { V } ( \cdot )$ in ${ \mathcal L } _ { \mathrm { v o l } }$ denotes the material volume estimated by Delaunay tetrahedralization, with $\epsilon = 1 0 ^ { - 6 }$ . We used $\lambda _ { \mathrm { 2 D } } = 1 , \lambda _ { \mathrm { r e p } } = 1$ , and $\lambda _ { \mathrm { v o l } } = 1 0 ^ { 4 }$ . For the adaptive volume-preservation loss, the reference configuration $\hat { \mathcal { P } } _ { \mathrm { r e f } } ( t _ { 1 } )$ is updated every five iterations. We performed optimization for 2,000 iterations using Adam [19], with learning rate $0 . 0 0 0 3 1 2 5 \ : ( = d _ { \mathrm { g r i d } } / 3 2 )$ and momentum parameters $\beta _ { 1 } = 0 . 9$ and $\beta _ { 2 } = 0 . 9 9 9$

## A.2.5. Re-optimizing Gaussian Attributes

The particle positions obtained by our method define the optimized geometry but do not directly provide the Gaussian appearance attributes required for novel-view rendering. We therefore ran a fixed-point-cloud 3DGS optimization following the official GIC implementation [3]: the particle positions were kept fixed, and only the Gaussian appearance-related attributes were optimized. This strategy enables novel-view rendering of the optimized and predicted states.

## B. Additional Experiments and Analysis

This appendix provides additional experiments and analysis to further characterize the proposed hypothesis-space reasoning framework. We first analyze global optimizers and Lagrangian particle refinement in more detail (Appendix B.1). We then provide supplementary results for the interior-topology stress tests, including F-score results and an additional ablation on the asymmetric subset (Appendix B.2). We also examine the effect of initialization source to separate the contribution of the optimization framework from that of the initialization representation (Appendix B.3). Additional future-prediction results on the SfC and Topology datasets are provided in Appendix B.4. Finally, we evaluate robustness under relaxed structural and physical assumptions (Appendix B.5), analyze sensitivity to physical-parameter errors and material variations (Appendices B.6 and B.7), and report runtime details (Appendix B.8).

## B.1. Global Optimizer and Refinement Ablation

We analyzed the contributions of global hypothesis optimization and local refinement under cavity position variation, where cavity localization is most ambiguous. This analysis extends the framework-stage ablation in Section 5.3 and the optimizer-sensitivity evaluation in Section 5.6 of the main paper.

We compared the default coarse-to-fine grid enumeration with random search and CMA-ES [9]. For grid and random search, the 125-evaluation setting uses only the coarse stage, whereas the 250-evaluation setting uses 125 coarse evaluations followed by 125 fine evaluations around the best coarse candidate. Random search replaces coarse grid enumeration with uniform sampling over the same search ranges. CMA-ES uses the same evaluation budgets without a separate coarse/fine split. All global optimizer configurations use the same objective and search ranges. For each configuration, we evaluated the selected hypothesis with and without Lagrangian particle refinement.

Table 7 shows three main trends:

(i) Limitations of local refinement. Local refinement alone improves over instantiation only, but remains limited by the initial cavity placement.

(ii) Effects of global hypothesis optimization. Adding global hypothesis optimization improves reconstruction before refinement, and combining global optimization with Lagrangian particle refinement gives the strongest results. For the default grid optimizer, the coarse-to-fine search improves over the coarse-only search, showing that the fine search stage is useful in this hypothesis space.

(iii) Optimizer sensitivity. Alternative gradient-free optimizers achieve comparable performance under matched or comparable evaluation budgets. CMA-ES reaches a strong solution with only 125 evaluations, but increasing its budget to 250 evaluations does not further improve performance in this low-dimensional hypothesis space. The default coarse-to-fine grid enumeration achieves comparable final accuracy, indicating that it is a reasonable and reproducible choice for the current hypothesis space. Overall, these results suggest that the performance gains primarily arise from hypothesis-level evaluation and subsequent Lagrangian particle refinement, rather than from the choice of global optimizer.

<table><tr><td>Configuration</td><td>Optimizer</td><td>Global evals</td><td>Fine stage</td><td>Local refine</td><td>CD↓</td><td>F↑</td></tr><tr><td>Instantiation only</td><td>一</td><td>0</td><td>一</td><td>No</td><td>0.204</td><td>0.136</td></tr><tr><td>+Local refine</td><td>一</td><td>0</td><td>一</td><td>Yes</td><td>0.156</td><td>0.138</td></tr><tr><td>+Global opt. (coarse)</td><td>Grid</td><td>125</td><td>No</td><td>No</td><td>0.155</td><td>0.138</td></tr><tr><td>+Global opt. (coarse) + Local refine</td><td>Grid</td><td>125</td><td>No</td><td>Yes</td><td>0.135</td><td>0.139</td></tr><tr><td>+Global opt. (coarse+fine)</td><td>Grid</td><td>250</td><td>Grid</td><td>No</td><td>0.121</td><td>0.141</td></tr><tr><td>+Global opt. (coarse+fine) + Local refine</td><td>Grid</td><td>250</td><td>Grid</td><td>Yes</td><td>0.109</td><td>0.144</td></tr><tr><td>+Random coarse</td><td>Random</td><td>125</td><td>No</td><td>No</td><td>0.152</td><td>0.139</td></tr><tr><td>+Random coarse + Local refine</td><td>Random</td><td>125</td><td>No</td><td>Yes</td><td>0.133</td><td>0.140</td></tr><tr><td>+Random coarse + grid fine</td><td>Random/Grid</td><td>250</td><td>Grid</td><td>No</td><td>0.129</td><td>0.141</td></tr><tr><td>+Random coarse + grid fine + Local refine</td><td>Random/Grid</td><td>250</td><td>Grid</td><td>Yes</td><td>0.112</td><td>0.142</td></tr><tr><td>+CMA-ES</td><td>CMA-ES</td><td>125</td><td>No</td><td>No</td><td>0.124</td><td>0.141</td></tr><tr><td>+CMA-ES + Local refine</td><td>CMA-ES</td><td>125</td><td>No</td><td>Yes</td><td>0.109</td><td>0.143</td></tr><tr><td>+CMA-ES</td><td>CMA-ES</td><td>250</td><td>No</td><td>No</td><td>0.125</td><td>0.141</td></tr><tr><td>+CMA-ES + Local refine</td><td>CMA-ES</td><td>250</td><td>No</td><td>Yes</td><td>0.110</td><td>0.142</td></tr></table>

Table 7. Ablation study of global optimizers and Lagrangian particle refinement on the SfC benchmark under cavity position variation Global evals denotes the number of forward-only MPM simulations used in the global stage. Fine stage indicates whether grid refinement is applied after the initial global search. Local refine indicates whether the selected hypothesis is refined with differentiable MPM simulation. All global optimizers use the same objective and search ranges.

## B.2. Additional Interior-Topology Results

We provide supplementary results for the interior-topology stress tests in Section 5.4 of the main paper. We first report F-score results corresponding to the main-paper CD results (Appendix B.2.1), and then provide an additional ablation on the asymmetric subset to analyze how the global and local stages behave when the dominant ambiguity lies in cavity shape (Appendix B.2.2).

## B.2.1. F-score Results for Interior-Topology Stress Tests

Table 8 reports F-score results for the interior-topology stress tests corresponding to the CD results in Section 5.4. Overall, the F-score results show a trend consistent with the CD results in the main paper: Under the known-size setting, our method improves over SfC-NeRF across all topology subsets. When stronger structural assumptions are available, the F-score remains stable or improves across the topology subsets.

Compared with CD, the difference between the templatecandidate setting and the known-template setting is smaller in F-score. This is consistent with F-score being a thresholded metric that is less sensitive than CD to fine geometric deviations. Although this thresholding compresses the gap between the two stronger-assumption settings, the overall trend remains consistent with the main-paper CD results. Thus, the F-score results provide complementary support for the same overall trend.

## B.2.2. Ablation on the Asymmetric Topology Subset

We further analyzed the asymmetric subset of the Topology dataset to complement the interior-topology stress tests in Section 5.4 of the main paper. We chose this subset because its cavities break symmetry and introduce preferred orientations, creating a clear shape mismatch with the innersurface initialization derived from the external geometry. Unlike the cavity-position setting in the SfC benchmark, this setting primarily changes the cavity shape rather than its location. This makes the experiment useful for examining how the global and local stages behave when the dominant ambiguity lies in cavity shape.

<table><tr><td>Method</td><td>Setting</td><td>Sym</td><td>Asym</td><td>Comp</td></tr><tr><td>SfC-NeRF [16]</td><td>Size</td><td>0.085</td><td>0.084</td><td>0.081</td></tr><tr><td>Ours</td><td>Size</td><td>0.089</td><td>0.090</td><td>0.085</td></tr><tr><td>Ours</td><td>Cand. w/ center</td><td>0.091</td><td>0.096</td><td>0.095</td></tr><tr><td>Ours</td><td>Template w/ center</td><td>0.092</td><td>0.096</td><td>0.095</td></tr></table>

Table 8. F-score results for the interior-topology stress tests. Results correspond to the CD results reported in Section 5.4 of the main paper. Higher values indicate better thresholded reconstruction consistency on the symmetric (Sym), asymmetric (Asym), and complex (Comp) subsets. Bold indicates the better result between SfC-NeRF and Ours under the known-size setting.

Table 9 provides an ablation complementary to the position-variation analysis in the main paper.

Stage-wise ablation. Local refinement alone gives a large improvement over instantiation only, suggesting that particle-level adjustment is particularly useful when the initial hypothesis has a shape mismatch. Global optimization alone also improves over instantiation only, although its effect is smaller than local refinement in this shape-mismatch setting. The full pipeline achieves the best CD, indicating that global hypothesis optimization and Lagrangian particle refinement remain complementary, while their relative contributions depend on the dominant source of ambiguity. Geometry-aware losses. The fixed volume-preservation loss slightly degrades CD, likely because asymmetric cavities require larger shape adjustment and a fixed reference volume can be overly restrictive. The adaptive volumepreservation loss improves reconstruction by updating the reference configuration during refinement. F-score changes only marginally across configurations, suggesting that it is less sensitive than CD to the geometric differences observed in this setting.

<table><tr><td>Configuration</td><td>CD↓</td><td>F↑</td></tr><tr><td>(a) Framework stages Instantiation only +Local refine +Global opt.</td><td>0.359 0.221 0.273</td><td>0.086 0.090 0.088</td></tr><tr><td>+Global opt. + Local refine (Full) (b) Geometry-aware losses</td><td>0.185</td><td>0.090</td></tr><tr><td>None</td><td>0.240</td><td>0.085</td></tr><tr><td>Rep. only</td><td>0.213</td><td>0.090</td></tr><tr><td>Rep. + Fixed volume Rep. + Adaptive volume (Full)</td><td>0.243 0.185</td><td>0.090 0.090</td></tr></table>

Table 9. Additional ablation results on the asymmetric subset of the Topology dataset. (a) Effect of framework stages. (b) Effect of geometry-aware losses during Lagrangian particle refinement. CD values are scaled by 10<sup>3</sup>.

## B.3. Effect of Initialization Source

SfC-NeRF uses Direct Voxel Grid Optimization (DVGO) [38], a voxel-based NeRF representation, for initialization, whereas our default implementation uses 3D Gaussian Splatting (3DGS) [18]. To isolate the influence of the initialization source, we additionally initialized our optimization from DVGO reconstructions by thresholding the learned alpha grid and sampling particle positions from occupied voxels. For this DVGO-based initialization variant, the deformation network used for surface extraction is unavailable, so we disabled the 3D surface loss.

Table 10 compares different combinations of initialization sources and optimization frameworks. Even when initialized from the same DVGO reconstruction used by SfC-NeRF, our optimization framework improves over SfC-NeRF. Using the default 3DGS initialization further improves reconstruction accuracy. These results indicate that the proposed optimization framework contributes substantially to the performance gain, while the initialization source provides additional improvement.

## B.4. Additional Future Prediction Results

We report additional quantitative results for future prediction on the SfC and Topology datasets. These results complement the analysis in Section 5.5 of the main paper. All methods are optimized using N = 14 observed frames and evaluated at +7 and +14 future frames. GIC and our variants use the GIC-based rendering pipeline described in Appendix A.2.5, whereas SfC-NeRF uses its original PAC-NeRF-based rendering pipeline [20].

<table><tr><td colspan="3"></td><td colspan="2">Size</td><td colspan="2">Position</td></tr><tr><td>Initialization</td><td>Optimization</td><td>CD↓</td><td>F↑</td><td></td><td>CD↓</td><td>F↑</td></tr><tr><td>DVGO</td><td>SfC-NeRF</td><td>0.165</td><td>0.135</td><td>0.281</td><td></td><td>0.128</td></tr><tr><td>DVGO</td><td>Ours</td><td>0.104</td><td>0.143</td><td></td><td>0.170</td><td>0.136</td></tr><tr><td>3DGS</td><td>Ours</td><td>0.063</td><td>0.151</td><td></td><td>0.109</td><td>0.144</td></tr></table>

Table 10. Reconstruction accuracy under different initialization sources and optimization frameworks. Even with the same DVGO initialization as SfC-NeRF, our hypothesis-space reasoning framework improves accuracy. CD values are scaled by 10<sup>3</sup>.

SfC benchmark. On the SfC benchmark, Table 11 reports results under cavity size and position variations. Our full model achieves the best performance for all settings and prediction horizons. The gains are substantial under both variations, demonstrating the strong end-to-end futureprediction performance of our full framework. For example, at +14 frames under position variation, our method improves PSNR by more than 5 dB over SfC-NeRF.

Topology dataset. Table 12 reports results on the Topology dataset under symmetric, asymmetric, and complex subsets. Our full model outperforms all baselines for every subset and prediction horizon. The gap between the global-only variant and the full model indicates that Lagrangian particle refinement is important for future prediction, especially for complex interior structures.

Together, these results show that global hypothesis optimization and Lagrangian particle refinement provide complementary benefits for future prediction under diverse geometry and topology conditions.

## B.5. Robustness under Relaxed Assumptions

We evaluated robustness by relaxing controlled assumptions used in the main experiments. Specifically, we considered two settings: unknown cavity size and unknown physical parameters. For each setting, global hypothesis optimization is extended to include the corresponding variables. The search ranges and grid resolutions are summarized in Table 6 in Appendix A. These relaxed settings test whether the framework remains effective when part of the structural or physical information is not provided.

## B.5.1. Unknown Cavity Size

We evaluated the unknown-size setting on the SfC benchmark under both cavity size and position variations. The cavity size is estimated jointly with cavity position by extending the global hypothesis space to include size as an additional variable. For SfC-NeRF, the mass loss term is removed to avoid relying on the known-size assumption. Table 13 summarizes the results. Compared with the knownsize setting, the unknown-size setting degrades reconstruction accuracy, but our method remains substantially better than SfC-NeRF under both assumption settings. These results indicate that the proposed framework can incorporate cavity size into hypothesis-level optimization without requiring it to be fixed in advance.

<table><tr><td rowspan="3">Method</td><td colspan="4">Size</td><td colspan="4">Position</td></tr><tr><td colspan="2">+7</td><td colspan="2">+14</td><td colspan="2">+7</td><td colspan="2">+14</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td></tr><tr><td>SfC-NeRF [16]</td><td>30.8</td><td>0.987</td><td>29.0</td><td>0.984</td><td>27.1</td><td>0.982</td><td>25.2</td><td>0.978</td></tr><tr><td>GIC [3]</td><td>28.4</td><td>0.985</td><td>26.7</td><td>0.982</td><td>23.1</td><td>0.977</td><td>22.1</td><td>0.974</td></tr><tr><td>Ours (Global only)</td><td>29.4</td><td>0.986</td><td>27.3</td><td>0.982</td><td>24.9</td><td>0.980</td><td>23.6</td><td>0.977</td></tr><tr><td>Ours (Full)</td><td>37.0</td><td>0.995</td><td>35.5</td><td>0.993</td><td>33.6</td><td>0.991</td><td>30.8</td><td>0.988</td></tr></table>

Table 11. Future prediction accuracy on the SfC benchmark under cavity size and position variations. All methods are optimized using 14 observed frames and evaluated at +7 and +14 future frames.

<table><tr><td rowspan="3">Method</td><td colspan="4">Symmetric</td><td colspan="4">Asymmetric</td><td colspan="4">Complex</td></tr><tr><td colspan="2">+7</td><td colspan="2">+14</td><td colspan="2">+7</td><td colspan="2">+14</td><td colspan="2">+7</td><td colspan="2">+14</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td><td>PSNR↑</td><td>SSIM↑</td></tr><tr><td>SfC-NeRF [16]</td><td>31.4</td><td>0.988</td><td>30.0</td><td>0.987</td><td>29.1</td><td>0.986</td><td>27.5</td><td>0.984</td><td>28.8</td><td>0.985</td><td>26.7</td><td>0.982</td></tr><tr><td>GIC [3]</td><td>24.3</td><td>0.981</td><td>23.9</td><td>0.980</td><td>22.1</td><td>0.977</td><td>21.5</td><td>0.975</td><td>21.4</td><td>0.975</td><td>20.5</td><td>0.973</td></tr><tr><td>Ours (Global only)</td><td>25.5</td><td>0.982</td><td>24.9</td><td>0.981</td><td>23.4</td><td>0.979</td><td>22.4</td><td>0.976</td><td>22.1</td><td>0.976</td><td>21.0</td><td>0.973</td></tr><tr><td>Ours (Full)</td><td>35.7</td><td>0.993</td><td>33.0</td><td>0.989</td><td>34.5</td><td>0.992</td><td>30.8</td><td>0.987</td><td>34.7</td><td>0.992</td><td>31.1</td><td>0.988</td></tr></table>

Table 12. Future prediction accuracy on the Topology dataset under symmetric, asymmetric, and complex subsets. All methods are optimized using 14 observed frames and evaluated at +7 and +14 future frames.

<table><tr><td></td><td></td><td colspan="2">Size</td><td colspan="2">Position</td></tr><tr><td>Method</td><td>Size</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td></tr><tr><td>SfC-NeRF</td><td>Known</td><td>0.165</td><td>0.135</td><td>0.281</td><td>0.128</td></tr><tr><td>Ours</td><td>Known</td><td>0.063</td><td>0.151</td><td>0.109</td><td>0.144</td></tr><tr><td>SfC-NeRF</td><td>Unknown</td><td>0.637</td><td>0.125</td><td>0.606</td><td>0.121</td></tr><tr><td>Ours</td><td>Unknown</td><td>0.086</td><td>0.147</td><td>0.128</td><td>0.141</td></tr></table>

Table 13. Results when relaxing the known-size assumption. Known indicates that the cavity size is provided, while Unknown estimates the cavity size jointly with cavity position. Bold indicates the better result between SfC-NeRF and Ours under each assumption setting. CD values are scaled by 10<sup>3</sup>.

## B.5.2. Unknown Physical Parameters

We evaluated robustness when the physical parameters, namely Young’s modulus and Poisson’s ratio, are unknown. The global hypothesis space is extended to jointly search cavity position and physical parameters, both of which are further optimized during local refinement. For SfC-NeRF, the physical parameters are optimized together with the reconstruction variables, using the ground-truth values as initialization to provide a favorable condition. Table 14 shows that our method exhibits only minor degradation compared with the known-parameter setting while outperforming SfC-NeRF under both assumption settings. Table 15 further shows that our method achieves lower estimation errors for both Young’s modulus and Poisson’s ratio.

<table><tr><td></td><td colspan="3"></td><td colspan="2">Position</td></tr><tr><td>Method</td><td>Phys. Params.</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td></tr><tr><td>SfC-NeRF</td><td>Known</td><td>0.165</td><td>0.135</td><td>0.281</td><td>0.128</td></tr><tr><td>Ours</td><td>Known</td><td>0.063</td><td>0.151</td><td>0.109</td><td>0.144</td></tr><tr><td>SfC-NeRF</td><td>Unknown</td><td>0.181</td><td>0.135</td><td>0.321</td><td>0.126</td></tr><tr><td>Ours</td><td>Unknown</td><td>0.063</td><td>0.151</td><td>0.121</td><td>0.142</td></tr></table>

Table 14. Results when relaxing the physical-parameter assumption. Known indicates that the physical parameters are provided, while Unknown estimates them jointly with cavity position. Bold indicates the better result between SfC-NeRF and Ours under each assumption setting. CD values are scaled by 10<sup>3</sup>.

<table><tr><td rowspan="2">Method</td><td colspan="2">Size</td><td colspan="2">Position</td></tr><tr><td> $\log _ { 1 0 } E \downarrow$ </td><td>ν↓</td><td> $\log _ { 1 0 } E \downarrow$ </td><td>ν↓</td></tr><tr><td>SfC-NeRF</td><td>0.052</td><td>0.201</td><td>0.106</td><td>0.180</td></tr><tr><td>Ours</td><td>0.023</td><td>0.119</td><td>0.068</td><td>0.162</td></tr></table>

Table 15. Physical-parameter estimation errors in the unknownparameter setting. We report the mean absolute errors of Young’s modulus $\log _ { 1 0 } E$ and Poisson’s ratio ν under the size- and position-variation settings of the SfC benchmark.

Summary. The proposed framework remains stable when cavity size or physical parameters are not provided. These results support the robustness of hypothesis-space reasoning under relaxed structural and physical assumptions.

## B.6. Robustness to Physical-Parameter Errors

We further evaluated sensitivity to physical-parameter misspecification, extending the representative ±10% error results reported in Table 5 of the main paper. Unlike the unknown-parameter setting in Appendix B.5, the physical parameters are provided to the method but are perturbed from the ground-truth values. We evaluated larger perturbations of ±20% and ±30% for Young’s modulus E and Poisson’s ratio ν on sphere objects under cavity size and position variations.

Table 16 summarizes the results. Errors in Young’s modulus lead to a gradual degradation as the perturbation magnitude increases, especially under cavity position variation. This is consistent with stiffness directly affecting postcollision deformation, so an incorrect stiffness value can make cavity localization from motion cues more difficult. In contrast, errors in Poisson’s ratio have little effect in this setting: even under ±30% perturbations, the results remain close to the clean setting for both size and position variations. Overall, these results indicate that the framework is robust to moderate physical-parameter errors, while large stiffness errors can degrade reconstruction when cavity localization is ambiguous.

<table><tr><td rowspan="2">Condition</td><td colspan="2">Size</td><td colspan="2">Position</td></tr><tr><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td></tr><tr><td>Ours (Clean)</td><td>0.059</td><td>0.097</td><td>0.100</td><td>0.093</td></tr><tr><td>Ours (E, ±10% error)</td><td>0.061</td><td>0.096</td><td>0.100</td><td>0.093</td></tr><tr><td>Ours (E, ±20% error)</td><td>0.069</td><td>0.095</td><td>0.120</td><td>0.092</td></tr><tr><td>Ours (E, ±30% error)</td><td>0.080</td><td>0.095</td><td>0.164</td><td>0.089</td></tr><tr><td>Ours (ν, ±10% error)</td><td>0.059</td><td>0.097</td><td>0.101</td><td>0.093</td></tr><tr><td>Ours (ν, ±20% error)</td><td>0.060</td><td>0.097</td><td>0.101</td><td>0.093</td></tr><tr><td>Ours (ν, ±30% error)</td><td>0.061</td><td>0.097</td><td>0.102</td><td>0.093</td></tr></table>

Table 16. Sensitivity to physical-parameter errors on sphere objects under cavity size and position variations. Results for each error level are averaged over positive and negative perturbations. CD values are scaled by 10<sup>3</sup>.

## B.7. Robustness to Material Variations

While the main experiments use the material-fixed subset of the SfC benchmark, the SfC dataset [16] also provides material variations under controlled geometry. Here, we used these variations to evaluate whether reconstruction remains stable under changes in material properties and models. This analysis differs from the unknown-parameter setting in Appendix B.5: the physical parameters are provided for each sequence rather than estimated, while either the parameter values or the material model differs from the default elastic setting. Thus, the goal is to evaluate robustness to material variations rather than physical-parameter estimation accuracy. We organized the evaluation along two axes: material variation and geometry setting.

Material axis. We evaluated three types of material variation from the SfC dataset: (i) Young’s modulus variation, where E is varied over $\{ 2 . 5 \times 1 0 ^ { 5 } , 5 . 0 \times 1 0 ^ { 5 }$ , 1.0 × $1 0 ^ { 6 } , 2 . 0 \times 1 0 ^ { 6 } , 4 . 0 \times 1 0 ^ { 6 } \}$ with the other simulation settings fixed; (ii) Poisson’s ratio variation, where ν is varied over {0.2, 0.25, 0.3, 0.35, 0.4} with the other simulation settings fixed; and (iii) material-model variation, where we evaluated seven material settings: two Newtonian fluids, two non-Newtonian fluids, two plasticine materials, and one sand model, with parameters following the PAC-NeRF dataset [20].

Geometry axis. For each material variation, we evaluated reconstruction performance under the following three geometry settings: (a) external-shape variation, where the five external shapes from the SfC dataset are used while the cavity is fixed at the center with size $\left( { \frac { 2 } { 3 } } \right) ^ { 3 } ;$ (b) cavity-size variation, where the external shape is fixed to a sphere and the cavity size follows the size-variation setting of the main SfC benchmark; and (c) cavity-position variation, where the external shape is fixed to a sphere and the cavity position follows the position-variation setting of the main SfC benchmark.

## B.7.1. Robustness to Young’s Modulus

Table 17 summarizes reconstruction results under Young’s modulus variations across the three geometry settings. Our method maintains stable performance across stiffness values and consistently improves over SfC-NeRF, indicating robustness to changes in material stiffness.

## B.7.2. Robustness to Poisson’s Ratio

Table 18 summarizes reconstruction results under Poisson’s ratio variations across the three geometry settings. Our method maintains stable performance across the evaluated values and consistently improves over SfC-NeRF, indicating robustness to changes in Poisson’s ratio.

## B.7.3. Robustness to Material Models

Table 19 summarizes reconstruction results under different material models across the three geometry settings. Our method consistently improves over SfC-NeRF across material categories and geometric variations, indicating that the framework remains effective beyond the elastic setting.

Summary. Our method consistently improves over SfC-NeRF across a wide range of elastic parameters and material models. These results indicate that the proposed hypothesis-space reasoning framework remains robust to material variations beyond the default elastic setting.

<table><tr><td></td><td colspan="2"> $2 . 5 \times 1 0 ^ { 5 }$ </td><td colspan="2"> $5 . 0 \times 1 0 ^ { 5 }$ </td><td colspan="2"> $1 . 0 \times 1 0 ^ { 6 }$ </td><td colspan="2"> $2 . 0 \times 1 0 ^ { 6 }$ </td><td colspan="2"> $4 . 0 \times 1 0 ^ { 6 }$ </td><td colspan="2">Avg.</td></tr><tr><td>Method</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td> $\operatorname { F } \uparrow$ </td><td>CD↓</td><td>F↑</td></tr><tr><td>SfC-NeRF Ours</td><td>0.206</td><td>0.132</td><td>0.196</td><td>0.132</td><td>0.195</td><td>0.133</td><td>0.215</td><td>0.132</td><td>0.230</td><td>0.132</td><td>0.208</td><td>0.132</td></tr><tr><td></td><td>0.066</td><td>0.150</td><td>0.062</td><td>0.148</td><td>0.068</td><td>0.150</td><td>0.069</td><td>0.149</td><td>0.072</td><td>0.147</td><td>0.068</td><td>0.149</td></tr><tr><td></td><td colspan="10">(a) External-shape variation</td><td colspan="2"></td></tr><tr><td>Method</td><td colspan="2"> $2 . 5 \times 1 0 ^ { 5 }$ </td><td colspan="2"> $5 . 0 \times 1 0 ^ { 5 }$ </td><td colspan="2"> $1 . 0 \times 1 0 ^ { 6 }$ </td><td colspan="2"> $2 . 0 \times 1 0 ^ { 6 }$ </td><td colspan="2"> $4 . 0 \times 1 0 ^ { 6 }$ </td><td colspan="2">Avg.</td></tr><tr><td></td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td></tr><tr><td>SfC-NeRF Ours</td><td>0.067 0.059</td><td>0.093 0.098</td><td>0.069 0.058</td><td>0.093 0.096</td><td>0.067 0.059</td><td>0.093 0.097</td><td>0.075 0.059</td><td>0.093 0.097</td><td>0.099 0.062</td><td>0.092 0.096</td><td>0.075 0.059</td><td>0.093 0.097</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>(b) Cavity-size variation</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $2 . 5 \times 1 0 ^ { 5 }$ </td><td colspan="10"></td><td colspan="2"></td></tr><tr><td>Method</td><td colspan="2"></td><td colspan="2"> $5 . 0 \times 1 0 ^ { 5 }$ </td><td colspan="2"> $1 . 0 \times 1 0 ^ { 6 }$ </td><td colspan="2"> $2 . 0 \times 1 0 ^ { 6 }$ </td><td colspan="2"> $4 . 0 \times 1 0 ^ { 6 }$ </td><td colspan="2">Avg.</td></tr><tr><td></td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td></tr><tr><td>SfC-NeRF</td><td>0.165</td><td>0.086</td><td>0.149</td><td>0.087</td><td>0.152</td><td>0.087</td><td>0.187</td><td>0.086</td><td>0.168</td><td>0.086</td><td>0.164</td><td>0.087</td></tr><tr><td>Ours</td><td>0.086</td><td>0.093</td><td>0.091</td><td>0.093</td><td>0.100</td><td>0.093</td><td>0.128</td><td>0.091</td><td>0.119</td><td>0.092</td><td>0.105</td><td>0.092</td></tr></table>

(c) Cavity-position variation  
Table 17. Robustness to Young’s modulus across geometry settings. We report CD↓ and F-score (F↑) under (a) external-shape, (b) cavitysize, and (c) cavity-position variations. Avg. represents the average over the five Young’s modulus settings. CD values are scaled by 10<sup>3</sup>.

<table><tr><td></td><td colspan="2">0.2</td><td colspan="2">0.25</td><td colspan="2">0.3</td><td colspan="2">0.35</td><td colspan="2">0.4</td><td colspan="2">Avg.</td></tr><tr><td>Method</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td></tr><tr><td>SfC-NeRF Ours</td><td>0.212 0.064</td><td>0.133 0.150</td><td>0.214</td><td>0.131</td><td>0.195</td><td>0.133</td><td>0.225</td><td>0.132 0.149</td><td>0.226 0.066</td><td>0.131</td><td>0.214</td><td>0.132</td></tr><tr><td></td><td></td><td></td><td>0.067</td><td>0.148</td><td>0.068</td><td>0.150</td><td>0.067 (a) External-shape variation</td><td></td><td></td><td>0.149</td><td>0.066</td><td>0.149</td></tr><tr><td colspan="9"></td><td colspan="2"></td><td colspan="2"></td></tr><tr><td rowspan="2">Method</td><td colspan="2">0.2</td><td colspan="2">0.25</td><td colspan="2">0.3</td><td colspan="2">0.35</td><td colspan="2">0.4</td><td colspan="2">Avg.</td></tr><tr><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td></tr><tr><td>SfC-NeRF Ours</td><td>0.070 0.059</td><td>0.093 0.097</td><td>0.076 0.059</td><td>0.092 0.096</td><td>0.067 0.059</td><td>0.093 0.097</td><td>0.073 0.058</td><td>0.093 0.097</td><td>0.066 0.058</td><td>0.092 0.098</td><td>0.070 0.058</td><td>0.093 0.097</td></tr><tr><td colspan="10">(b) Cavity-size variation</td><td colspan="2"></td></tr><tr><td colspan="2"></td><td colspan="2">0.2</td><td colspan="2">0.25</td><td colspan="2"></td><td colspan="2"></td><td colspan="2"></td><td colspan="2">Avg.</td></tr><tr><td>Method</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td></tr><tr><td>SfC-NeRF</td><td>0.202</td><td>0.087</td><td>0.159</td><td>0.086</td><td>0.152</td><td>0.087</td><td>0.199</td><td>0.087</td><td>0.176</td><td>0.086</td><td>0.178</td><td>0.087</td></tr><tr><td>Ours</td><td>0.104</td><td>0.093</td><td>0.100</td><td>0.094</td><td>0.100</td><td>0.093</td><td>0.100</td><td>0.094</td><td>0.098</td><td>0.093</td><td>0.101</td><td>0.093</td></tr></table>

(c) Cavity-position variation  
Table 18. Robustness to Poisson’s ratio across geometry settings. We report CD↓ and F-score (F↑) under (a) external-shape, (b) cavity-size, and (c) cavity-position variations. Avg. represents the average over the five Poisson’s ratio settings. CD values are scaled by 10<sup>3</sup>.

<table><tr><td></td><td colspan="2">Newtonian</td><td colspan="2">Non-Newtonian</td><td colspan="2">Plasticine</td><td colspan="2">Sand</td><td colspan="2">Avg.</td></tr><tr><td>Method</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td></tr><tr><td>SfC-NeRF</td><td>0.196</td><td>0.131</td><td>0.218</td><td>0.132</td><td>0.230</td><td>0.132</td><td>0.222</td><td>0.132</td><td>0.216</td><td>0.132</td></tr><tr><td>Ours</td><td>0.076</td><td>0.147</td><td>0.083</td><td>0.146</td><td>0.086</td><td>0.146</td><td>0.071</td><td>0.150</td><td>0.080</td><td>0.147</td></tr></table>

(a) External-shape variation
<table><tr><td></td><td colspan="2">Newtonian</td><td colspan="2">Non-Newtonian</td><td colspan="2">Plasticine</td><td colspan="2">Sand</td><td colspan="2">Avg.</td></tr><tr><td>Method</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td></tr><tr><td>SfC-NeRF</td><td>0.071</td><td>0.093</td><td>0.088</td><td>0.092</td><td>0.080</td><td>0.093</td><td>0.082</td><td>0.091</td><td>0.080</td><td>0.092</td></tr><tr><td>Ours</td><td>0.063</td><td>0.099</td><td>0.061</td><td>0.098</td><td>0.079</td><td>0.096</td><td>0.065</td><td>0.097</td><td>0.067</td><td>0.097</td></tr></table>

(b) Cavity-size variation
<table><tr><td></td><td colspan="2">Newtonian</td><td colspan="2">Non-Newtonian</td><td colspan="2">Plasticine</td><td colspan="2">Sand</td><td colspan="2">Avg.</td></tr><tr><td>Method</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td></tr><tr><td>SfC-NeRF</td><td>0.235</td><td>0.084</td><td>0.216</td><td>0.083</td><td>0.205</td><td>0.085</td><td>0.217</td><td>0.084</td><td>0.218</td><td>0.084</td></tr><tr><td>Ours</td><td>0.108</td><td>0.094</td><td>0.095</td><td>0.094</td><td>0.129</td><td>0.092</td><td>0.107</td><td>0.093</td><td>0.110</td><td>0.093</td></tr></table>

(c) Cavity-position variation

Table 19. Robustness across material models and geometry settings. We report CD↓ and F-score (F↑) under (a) external-shape, (b) cavitysize, and (c) cavity-position variations. Results for Newtonian, non-Newtonian, and plasticine are averaged over two parameter settings per material category. Avg. is computed over all evaluated material settings. CD values are scaled by 10<sup>3</sup>.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Global fwd. evals</td><td rowspan="2">Local grad. iters</td><td rowspan="2">Total budget</td><td colspan="2">Size</td><td colspan="2">Position</td></tr><tr><td>CD↓</td><td>F↑</td><td>CD↓</td><td>F↑</td></tr><tr><td>SfC-NeRF (default)</td><td>0</td><td>1,000</td><td>1,000</td><td>0.165</td><td>0.135</td><td>0.281</td><td>0.128</td></tr><tr><td>SfC-NeRF (extended)</td><td>0</td><td>2,300</td><td>2,300</td><td>0.167</td><td>0.136</td><td>0.288</td><td>0.128</td></tr><tr><td>Ours</td><td>250</td><td>2,000</td><td>2,250</td><td>0.063</td><td>0.151</td><td>0.109</td><td>0.144</td></tr></table>

Table 20. Computational budget and reconstruction accuracy on the SfC benchmark. Globalfwd. evals denotes the number of forward-only MPM simulations used by the global stage. Local grad. iters denotes the number of gradient-based differentiable-simulation iterations. For SfC-NeRF, this corresponds to density-optimization iterations; for ours, to Lagrangian particle refinement iterations. Total budget is reported as the sum of these counts for comparison, although forward-only global evaluations are cheaper than forward–backward gradien iterations and use fewer frames. CD values are scaled by 10<sup>3</sup>.

## B.8. Runtime Analysis

We provide additional details for the complexity discussion in Section 4.5 of the main paper. The proposed framework has two computationally different components: global hypothesis optimization uses forward-only MPM simulation [10, 37], whereas Lagrangian particle refinement uses forward–backward differentiable MPM simulation [11]. A single global evaluation requires less computation than a single local refinement iteration because it involves no backward pass and uses only collision-adjacent frames, whereas each local refinement iteration backpropagates through the full observed sequence. We therefore report the numbers of forward-only global evaluations and gradient-based optimization iterations separately.

SfC-NeRF optimizes densities through differentiable simulation for 1,000 iterations under its default setting. Our method uses 250 forward-only simulations for coarse-tofine global hypothesis optimization, followed by 2,000 iterations of Lagrangian particle refinement. To assess whether the improvement can be explained simply by a larger optimization budget, we also evaluated an extended SfC-NeRF baseline with 2,300 optimization iterations, giving a comparable count-based budget to our method.

Table 20 shows that increasing the number of SfC-NeRF iterations does not improve reconstruction accuracy. In contrast, our method achieves substantially lower CD and higher F-score with a comparable budget. This indicates that the improvement comes from changing the optimization structure—evaluating explicit structural hypotheses before local refinement—rather than from increasing the number of gradient-based iterations.

We also measured representative wall-clock runtime on a sphere sequence with cavity size $\left( { \frac { 2 } { 3 } } \right) ^ { 3 }$ using a single H100 GPU. A forward-only global evaluation on collisionadjacent frames takes about 12 s, whereas one forward– backward local refinement iteration over the full observed sequence takes about 48 s on this sequence. As a result, the 250-evaluation global search accounts for roughly 3% of the local-refinement runtime. This supports the practical feasibility of using a moderate $\mathbf { g }$ lobal search budget before local refinement.

## C. Qualitative Results

We provide additional qualitative comparisons that complement Fig. 4 of the main paper. These examples show how the proposed framework behaves under cavity position and size variations, as well as interior-topology stress tests. Supplementary videos for Fig. 4 and the examples below are available on the project page.

## C.1. SfC Benchmark

We first present qualitative reconstruction results on the SfC benchmark, which contains controlled variations in cavity position and size.

## C.1.1. Cavity Position Variations

Figure 7 shows representative results for two objects under cavity position variation. SfC-NeRF often reconstructs fragmented or irregular cavity structures, which can lead to post-collision dynamics that deviate from the observation. In contrast, our method reconstructs more coherent internal structures that better match the ground truth. The stage-wise results further show how each component contributes: cavity instantiation provides an initial structure, global hypothesis optimization improves the coarse cavity configuration, and Lagrangian particle refinement adjusts local geometry. These differences become more apparent in post-collision frames, where inaccurate cavity configurations lead to different deformation patterns.

## C.1.2. Cavity Size Variations

Figure 8 shows representative results for two objects under cavity size variation. In this setting, cavity instantiation already captures the coarse cavity extent, leading to smaller differences across stages than in the positionvariation setting. Nevertheless, Lagrangian particle refinement improves local geometric consistency and leads to more stable post-collision dynamics. These results suggest that, under cavity-size variation, refinement mainly improves physical consistency and local geometry rather than resolving a large positional ambiguity.

## C.2. Interior-Topology Stress Tests

Figures 9–11 show representative qualitative results for the interior-topology stress tests, covering symmetric (snowman), asymmetric (pacman), and complex (split sphere)

cavities. Following Table 3 of the main paper, we visualized the known-size setting and two stronger structuralassumption settings. Under the known-size setting, our method produces more coherent cavity structures than SfC-NeRF, although exact reconstruction remains challenging for complex cavities. With a set of candidate templates and a known center, the global stage can select among structural alternatives before refinement. Because the candidate set includes the ground-truth template, this setting can produce results close to the known-template setting, suggesting that hypothesis-level selection can identify a plausible structural alternative from the candidate set. With a known template and center, the remaining ambiguity is mainly orientation, further constraining the reconstruction problem. These results are consistent with the quantitative trends in Table 3 of the main paper: hypothesis-space reasoning improves reconstruction in the interior-topology stress tests, while exact recovery of complex interior topology remains difficult from collision observations alone.

Summary. These qualitative examples support the quantitative findings: global hypothesis optimization improves coarse cavity selection, and Lagrangian particle refinement improves local geometry and post-collision consistency. They also clarify the scope of the framework: stronger structural assumptions can improve reconstruction in the interior-topology stress tests, but complex cavities remain challenging when the observations do not provide sufficien evidence.

![](images/0fe0ca892cd2b4e0164e65dd1fd7e32cca6dfa9b6846b1d6b07bbc95314db24f.jpg)

![](images/f8d9ea9aeac243b705995bb35c26062d01e6c51fcd2d0431eaeb85cfcc2a8990.jpg)

![](images/69b9348faeb16e21785e4c1cf6cf55a5a9c03a9a4955d58d25c8278d6bc0c9c1.jpg)

![](images/dd8a4f15780452529c866e8920fabfb77139511411dbb9eec990a904691294b6.jpg)  
CD↓ 0.279  
CD↓ 0.075

![](images/7614ead3cb1aa0ffa837434110b2ace87bbabecb3b81aa6a6c14fcdbf99e42a7.jpg)

![](images/b0d7683a1822c0776d61baad591eb2b72ed79d686975aaf3bcc51fa2750e2b20.jpg)  
CD↓ 0.160

![](images/6cb226770dbcc25d75337dbba0653cfa7af37b4874de527114d94c5b8259381e.jpg)  
CD↓ 0.090

![](images/d34a3cc621c9a84c59c778961132dadfa9f95f7387a81e98d0f3b1026fd4b012.jpg)  
CD↓ 0.075

(a) Observation  
![](images/0abf563327cb04da60bbda874ca5f44888c973e18b761940c748c9f34d6fa2dc.jpg)

![](images/7da551ec6f8a1fb147594dfc28b548a691ce4205ef308be9a3597eb2a02e1cbe.jpg)

![](images/76453f445ffd5ed820ab071e21947ea927c05766540fb201dafb678f4cbcae9d.jpg)  
(b) Ground truth

![](images/f5ca032c7a714b6a4bce9e2d0e2ad42ba5c47e3a2139336062d602dbdc82cb96.jpg)

![](images/717182c0c15d138a462e28d0be0c018159dab5d7cd3e60e225543a50610efcca.jpg)  
(c) SfC-NeRF

![](images/7eca8eb416bcce5d225d6302b6a702b86cd61995c165d2c2dcafd3ababf02645.jpg)

![](images/4a08053c0a547cf8555420dc10388cc0fe9c932c19bfeb14a2fb8c610270a921.jpg)  
(d) Ours

![](images/4b199648189a705ea76cfe05e89d881c647fee812fb18f5440acbfd6d4aefdde.jpg)  
(e) Instantiation only

![](images/57cc99e40b057544b1ce9661ab495b4f055e8b221b0168494b80b86c3e54b042.jpg)  
(f) +Global opt.

![](images/84b8737b7efae91f0dd2756cec5ecfecc1dfcc07a896ad3dad92464b0ca54f99.jpg)  
(ii) Stage-wise ablation  
(g) +Global opt. + Local refine (Full)

![](images/35595165751a5d34bf24473c5dc7596a2657bd0ee8e01067e94df2114ce658d4.jpg)

(i) Main comparison (a) Bicone  
![](images/d6dee5a4e722c255a9986dae03c09db50df898f0f2e4198d7ef8e9cb407d6bdb.jpg)

![](images/ce33fb68d6e87a042015fe4c34413624033c6b94d37f547950d5151a59781aae.jpg)  
CD↓ 0.217

![](images/1791a84d0e4bc2e5eb8a6eba483627e8fb9ae79d94ba7a7ab31a6e7bfc911a91.jpg)  
CD↓ 0.112

![](images/b8059652828b5b22c510440e624cd9703451c2c6c4875ca705d82dfb21f7d140.jpg)  
CD↓ 0.240

![](images/31f4981ec5afd4b5bd53a6e69701d4d9d43ba09915121732f2ad16d0069f8ae4.jpg)  
CD↓ 0.150

![](images/b408b5576f51d8c6b7d440b8dd78b979976a2fc975c7d42a934435b66b689112.jpg)  
CD↓ 0.112

![](images/e07609974399acec06d80dcdcba5543eb0a2b69eb685834724dd02f18fb0bcf9.jpg)

![](images/02db7108c6dafdcbdf27ef024f1c18c0e617453660e9a9e1a31f41b988f5b12d.jpg)

![](images/80d31b14143e516b1ec7aaf3cae3400f44066ac7b186374eea654ab1532d256e.jpg)

![](images/e26f4e098d26c1b3cdef9372d43988832591951d5545227ad3a6c2cbb3f023cb.jpg)

![](images/b11458719346708acb8fd61096d0025a4f432b74a0c147d99e1b5ec411fc6a10.jpg)

![](images/160680d3f87a1a1570e22bd032908ca3827e2c94f9838e0436ec95b1c1533046.jpg)

![](images/1632f461345286d917cf7312cb015dea3dde135b2710b499303f301ee5443767.jpg)  
(a) Observation

![](images/492d7eb5fbce2e6f4e5ff66df30ee7e78a9d65e37c326501dad3564a995bbc1a.jpg)

![](images/3674880dd3a2d50802c6c6cb983301a624fda20319d815f36b1d5f085115c75f.jpg)  
(b) Ground truth

![](images/e637dca968219976525548e3315fa921f2138f7bdcf63b35e910212209198984.jpg)  
(c) SfC-NeRF

![](images/5a4d0e298b40ab46a68f8df541d693cd422fca5b120d25b09fe9901f922c5179.jpg)  
(d) Ours

(i) Main comparison  
![](images/a6cc8665c6fcb39b8fe8c3fe5755e12c29effc0b2b4c3e66bbe6499c960f873c.jpg)  
(e) Instantiation only

![](images/d835c742bb3cd3f3c7898d06ac731c646ad7fc469cbcc9a68d64a93d3df324d5.jpg)  
(f) +Global opt.  
(b) Sphere  
(ii) Stage-wise ablation

![](images/48f6b23d369c4fe06504a2368bc0c4bc718f85a87af602b59311ecb078963557.jpg)  
(g) +Global opt.  
+ Local refine (Full)

Figure 7. Qualitative reconstruction results under cavity position variation on the SfC benchmark. We show examples for bicone and sphere For each object, we show observed images, ground-truth cross-sections, and reconstructions from different methods and stages. Block (i) shows the main comparison between SfC-NeRF and our method, while block (ii) visualizes the stage-wise effects of cavity instantiation, global hypothesis optimization, and Lagrangian particle refinement. Rows correspond to the initial frame (t ), the collision moment, and a post-collision frame. CD is reported for the initial reconstruction. Ground-truth cross-sections are shown in gray. Corresponding supplementary videos are available on the project page.

![](images/0da963a45de1fbb22b843cd52cb381283178d5c6159cd7dec119d3ab71cad976.jpg)

(a) Observation  
![](images/0e161188cad102d2d3148c9a7aee92baa7b8a35f23fb28b780c3c3a7e5b3a59c.jpg)

![](images/18d8edbf9d0102de22c8b7fad9de4c41a12c3dfc898edcfb0dd2ad36b1b3f118.jpg)

![](images/bb15172be11915bc4aef7a19c1f117cb81afdab509b7b9beeaeec1cbd3f5ed5b.jpg)  
CD↓ 0.151

![](images/07e6971ae625bcb3966c18aaf9018a6040276da03a44d7ed96bfa69641b6507f.jpg)

![](images/47ca203591c7fb2421dd87f80bb500c710767596323174c7d7e63d2aafa78e5b.jpg)  
CD↓ 0.059

![](images/b891df53aa3f24af296afd8f76ca989dba0157e452778af3cc9b2381d5b0792c.jpg)  
CD↓ 0.057  
CD↓ 0.057

(b) Ground truth  
![](images/0893995780e80a65384fbdd858123677cfb3431fe3a52ddc2010898a961d1197.jpg)  
CD↓ 0.059

![](images/db366bdb94ec60891119f7e07bbd6c8ad803b022ea676bbf266947034f85e4f6.jpg)

![](images/cfcff4e10f07d150b3093082dd87e9af35c413e927e9252b7110e70c405912de.jpg)

![](images/a6fb6f6eb5c31c5a482b976ab5037a791d890761923cba59eaf8110a6a926176.jpg)

![](images/7b1be8ccdbfd990b09f1e556a4b505fb7cc5c6b99e1ab1ef5f75d2e540993d93.jpg)

![](images/80d5372452eac72a5c55b3a1d3333edc80ac9aa051a194cb074be580e9326833.jpg)

![](images/0855602e7646860fd4577e719481126a084e15d163f86ea00478cc7622309efb.jpg)

![](images/288f02a3faccdca3eca6096bc4fd58f7ceb0adf6db3443fe2d99e03d319caa27.jpg)  
(c) SfC-NeRF

![](images/f31e5223750b104f36d1fdaee023d72336f55d20304e0863bdb5fa2bb9299ad4.jpg)

![](images/bb831b9cfd40e3aed1e84620746fadf4cae4a88ee478aea0f6f135e024da0d60.jpg)  
(d) Ours

![](images/2b06dbe220589c48524176db5ab841337fad4b85bffb88b1d9cbce21c2f9ac58.jpg)  
(e) Instantiation only

![](images/e7cdc6b537b692d4dec8b058ea6b9fc5669596dd6f5018c4cd09b1cc1b9cad88.jpg)  
(f) +Global opt.

![](images/34fbf72ad36ab8c7769c9e3e07a416f08dc28d2bf7bccc190f7a60f79a643730.jpg)  
(g) +Global opt. + Local refine (Full)

(i) Main comparison (a) Cube  
(ii) Stage-wise ablation  
![](images/3ee8d854febb18daa39f079438210db69f9603bcae18ae555370bf802bdcc55c.jpg)

![](images/63bdf710dae20f1e40cb148702ab4c216fdf3c996bbfbc45ff4b785e564818ca.jpg)  
CD↓ 0.478

![](images/12b2881289bd83d219787f5ec7265d000af6d400da9dcf46caf4edccf59c97cf.jpg)  
CD↓ 0.062

![](images/785e586b627169ed83a7b98b53426fc881381634a86b8800c87acd8f7298beb8.jpg)  
CD↓ 0.061

![](images/97dea170ff64025117080dc864ab5592382a4a3c34826efc68672accf3aab3ae.jpg)  
CD↓ 0.061

![](images/4be454e371fd3539da424e22694e7ddc7a514049f715cd8f3ab05ea76ac60b8f.jpg)  
CD↓ 0.062

![](images/89fe05c0976bd631b0a2de0698516c3726d5a2602d074a6128755c17aeceb9be.jpg)

![](images/822648d41f51066c10cc98b12686681252ce1b182b5ea8cc02ca8996b5449d09.jpg)

![](images/38e36d1b57602f60da7bc3a309f7735af143fdc7bf27e187f2caeddcc2faa169.jpg)

![](images/f444142ffd8de98cdcf3b7b1bcc636ce5cd0d9d657a09ffffeaa7f2dd0498893.jpg)

![](images/91d3d879e26e690b8a140a2ed8ee211c4b0d74e927f39aa5278c1405275317fd.jpg)

![](images/a354ef613d8f742700c6d4938d51e3ab71e298a1990751c5bd037bef71f5a74a.jpg)

![](images/47ed75eadd052446058d1a08701f26593c5c06fba6e54ce05ce222270c4c4481.jpg)

![](images/ee5850540601ed532f802e4d179ce5694680ea1eb40b65f82cdc07a8f36ceddb.jpg)  
(a) Observation

![](images/d6bec2c8254950d28b2fa7607975917c5c96c822449125d0611fe0b712d84ee9.jpg)  
(b) Ground truth

![](images/3794ce9291a8862a8d2bd7b635ac294a6832efb2f7bc3dbf70f381c557830ce4.jpg)  
(c) SfC-NeRF

![](images/b3fce5de2df1fd460f6b69cb95a0909fe28785eb1f74d6cb5d522f322833175d.jpg)  
(i) Main comparison  
(d) Ours

![](images/016f721e21fb563c1da3d2a156daf861b40adf337b84d69629fbe06065503f0a.jpg)  
(b) Cylinder  
(e) Instantiation only

![](images/68126367d7338e5be0457ff47c6cc6e0f1826ee558d55b2be0cd29201b0d5d97.jpg)  
(f) +Global opt.  
(ii) Stage-wise ablation

![](images/050dbb25017f6dcc04054d5988840c385a41ffc7fea7d1858b4f445c646285ba.jpg)  
(g) +Global opt.  
+ Local refine (Full)

Figure 8. Qualitative reconstruction results under cavity size variation on the SfC benchmark. We show examples for cube and cylinder. For each object, we show observed images, ground-truth cross-sections, and reconstructions from different methods and stages. Block (i) shows the main comparison between SfC-NeRF and our method, while block (ii) visualizes the stage-wise effects of cavity instantiation, global hypothesis optimization, and Lagrangian particle refinement. Rows correspond to the initial frame (t ), the collision moment, and a post-collision frame. The coarse cavity geometry is largely captured after instantiation, while Lagrangian particle refinement improves local geometry and post-collision consistency. CD is reported for the initial reconstruction. Ground-truth cross-sections are shown in gray. Corresponding supplementary videos are available on the project page.

![](images/8039328d087d9a33dd91f3175045f7540c9256572117eef34c7cc06235fbb71f.jpg)

![](images/7f515c5a4ecb773a23498d61ccd0b58f8c593a0795d95af9df7b3420a3332c21.jpg)

![](images/f5615c7c8129e0a3a49c674eb21867a9f999f698f75c954c3f16c2f4b5023008.jpg)  
CD↓ 0.166

![](images/4844a9bbd3c7571b2c0fe7417df6edd2f091d1de3ea062a5c0a54ec6cb62c982.jpg)  
CD↓ 0.145

![](images/04ebed9e928c2f72dacccb9f470b7d39111488cc33ed91f34213db041b1dacb9.jpg)

![](images/3a4326c7fdd2b3ca33c486b6f0a3e5d796c9f483ab76eed94aad186641337271.jpg)  
CD↓ 0.058

![](images/46674f2832f58c445eb79a6d27e062709503e70e7a6164a857d2df5375d7eabf.jpg)  
CD↓ 0.058

![](images/f23c51cc500678378c4d3d45daa4921f54a3f7b7434a8ff88d2ffaa4662c90c4.jpg)

![](images/a009483ef8c5dbc388232ebcd63fd7f6fadbbe7a913023712972dc8e0c57b7a6.jpg)

![](images/776948a901687fad1dfdc36fcdc51993c937dd3ae0d80f2466a2463f6c2baf38.jpg)

![](images/41e13995ed2f657ba3491a01e28e68b4e5544edf06cf4acc11b81762d016653d.jpg)

![](images/c4fe2464bfd0285ff6c8797005534ac0d40201bd8115fa91e0f511cd5e525cf1.jpg)

![](images/f6f14ebbd7d2fbb5c6ea496797179d1dccb665546a7c17add6cf702e4dbf2685.jpg)  
(a) Observation

![](images/54f25a987d81efab41051a90cc231367321362a2841f689154fabac621cb8b6c.jpg)  
(b) Ground truth

![](images/cb9022a2f4819ec6a91b2773afa963aa892904a781cc8313e254ba429e345cd0.jpg)  
(c) SfC-NeRF  
(Size)

![](images/bbd9883e75e2b90e8135577902fd7b353b3136e3f809212ff1b693f033f90459.jpg)  
(d) Ours  
(Size)

![](images/d4acdc3bbcbde2b10c4b70a089d9e3853263b2089c6a5c80174dd56507a8f349.jpg)  
(e) Ours  
(Cand. w/ center)

![](images/ac168047f8ccd9857143474ecd0e9c8c2b4d3bcb77026f3ded7fb01e6561e8fc.jpg)  
(f) Ours  
(Template w/ center)

Figure 9. Qualitative results for the symmetric subset of the interior-topology stress tests. We show a representative snowman cavity. Columns correspond to (a) observation, (b) ground truth, (c) SfC-NeRF under the known-size setting, (d) our method under the knownsize setting, (e) our method with a set of candidate templates and a known center, and (f) our method with a known template and center. Rows correspond to the initial frame (t ), the collision moment, and a post-collision frame. The visualization follows the hypothesis space settings in Table 3 of the main paper. Stronger structural assumptions constrain the hypothesis space and improve reconstruction consistency. Corresponding supplementary videos are available on the project page.

![](images/5e69b5497d4f78d21d2392d8d222ec15f0ff3288f7c782ddb425292f84c3bfde.jpg)

![](images/8d0c2b043e2a7dc85ded731b162613642d00bc586b5d71fe3b895dd598ce7b53.jpg)

![](images/0d02462f45c594d0a5a0747593b396d0c93d6cc77a7ef5166a74171056bb924a.jpg)  
CD↓ 0.325

![](images/eed2a6b6368ba0dc2d03f75488ffd4b64f00faf2d20dcc21e30e3e771bafbf7b.jpg)  
CD↓ 0.204

![](images/7e1831b002ba6a5c5cad91d5d97ddebd4173d08b5cf4363b584af9ff1cda1fbd.jpg)  
CD↓ 0.063

![](images/3b9b07ea1be5e470790f35908b0aa4332bceb6f5e0eb9e9dc93859f0c0e3fe5c.jpg)  
CD↓ 0.063

![](images/99767acacb5cfce06e1909a69b104be415de3adde0dff978838d627dff59e320.jpg)

![](images/d3f755462760b561bff0e5c585d2e31b456751da482f11899c14019ce3c507ad.jpg)

![](images/530b17403d4d1391ceab9845304eb14f3d5197cc449c1ecdafc84f11c84aae87.jpg)

![](images/3f28f194033ab53f6d9d8d6e104f230b987f02725881ee599ac75e9ad80c3b89.jpg)

![](images/3a2335363044f5743bcd63a5aac0e07aa5023b9f54e1335a8c1a30360c613340.jpg)

![](images/1b81660e950da73022bb6965d107edb0afb9d7cfd978997b71e4573bb90a2793.jpg)

![](images/f62707d0b38a541a4868d6bc0ecac6a127725308d84eb7f95508f9326bcf531b.jpg)  
(a) Observation

![](images/57c76ecfcda9c682988f43880c14fc3abb24e343f02a3158991f450cc36eb7f4.jpg)  
(b) Ground truth

![](images/9ca7dd239b9e4999b6c993172d867959a978752cd30c0750d299c4987bfa29fe.jpg)  
(c) SfC-NeRF  
(Size)

![](images/49bcaa3d6e6b97623334df287906d5e0250fbd65973f843b43b90891d6b9c681.jpg)  
(d) Ours  
(Size)

![](images/a4553270503e15016b5c15e1f0d729cb073f4e78071b41f98454d0d776664044.jpg)  
(e) Ours  
(Cand. w/ center)

![](images/53e8ba23e7ed1fd9d9b32524cf9f301a2bbc618f8d1a37b502749a11e75616c5.jpg)  
(f) Ours  
(Template w/ center)

Figure 10. Qualitative results for the asymmetric subset of the interior-topology stress tests. We show a representative pacman cavity. Columns correspond to (a) observation, (b) ground truth, (c) SfC-NeRF under the known-size setting, (d) our method under the knownsize setting, (e) our method with a set of candidate templates and a known center, and (f) our method with a known template and center. Rows correspond to the initial frame (t<sub>1</sub>), the collision moment, and a post-collision frame. The visualization follows the hypothesis-space settings in Table 3 of the main paper. The asymmetric cavity introduces a preferred orientation, making this example useful for visualizing how candidate templates and orientation refinement affect reconstruction. Corresponding supplementary videos are available on the projec page.

![](images/81df5d5f3a233decb32e99de41783e57449259e9e587e218a54b06168d926f63.jpg)

![](images/a3c70de4d28d4f5304f0f0d0278f382a1c5130ffd203e5cf07f2e5b833c9247b.jpg)

![](images/a911e606ff22068ca64ea4da50677552d26832aca4aad5a2f5d1f80c89b85223.jpg)

![](images/16bb824386c3095ec8cf24617f40111755b48954f91edd1f5606c449cc6fb65d.jpg)  
CD↓ 0.527

![](images/dc41361d6c627f64bf42e070b23e67e804ce8bc96ed65a9f50fec9d8cb2b6f64.jpg)  
CD↓ 0.505

![](images/a1587ac5a2ccddd466d4eef5993cdc73a419c2eb251fe8225eba91654a6ad2ee.jpg)  
CD↓ 0.062

![](images/d0e22fcc8d53ae90042e2083bbc9c595144645bb2b12f81efd2ff8eb366bbe00.jpg)  
CD↓ 0.062

![](images/258d0b4ffa98d5feae04558b7f131a5b2a0046dd4460752a18d66f36387d40c1.jpg)

![](images/a319142592ab9e9f36e96a659db493be6184db1ea0cf31e3ec01162cd01021a7.jpg)

(a) Observation  
![](images/aa4cb04f9fc28312fe95d17fda0b6b387c9ff7b44e276750b0b578aef653d67b.jpg)

![](images/46682b018a97647ad35028ca4865611da61b57c9d5ec8ab08467fcc05db130fe.jpg)

(b) Ground truth  
![](images/b53fc6516554f1b0ce8447ca0d42e777d99869306792060d2e6ed161ab292ee3.jpg)  
(c) SfC-NeRF  
(Size)

![](images/bc225b8e2af53e17429dc0126f61caaf6a1346a5c39313b1981b0abbac55833e.jpg)

![](images/e8665ed9f4d0d80fa5b85a7cdb62b84393398fbd036d9d5009c36fd2e53187d1.jpg)

![](images/a9f7ce7cd3650bbaac4b3b2682e82d4b88a2718ec68d2838ba5efc56f703a49e.jpg)

(d) Ours  
(Size)  
![](images/d115412740b7d0cae93558570242b11d2dd8ca8f2a864d6f03b89b974330ab62.jpg)

![](images/cf8facf49253e77d87eedbeb29162e46abac1df5d3e7816a130404cd0a1ad465.jpg)

(e) Ours  
(Cand. w/ center)  
![](images/15f58b7b83fdace919337fffef4137b875bfbd021f893e8657ebf77c597f7fa8.jpg)  
(f) Ours  
(Template w/ center)

Figure 11. Qualitative results for the complex subset of the interior-topology stress tests. We show a representative split sphere cavity. Columns correspond to (a) observation, (b) ground truth, (c) SfC-NeRF under the known-size setting, (d) our method under the knownsize setting, (e) our method with a set of candidate templates and a known center, and (f) our method with a known template and center. Rows correspond to the initial frame $( t _ { 1 } ) ,$ , the collision moment, and a post-collision frame. The visualization follows the hypothesis-space settings in Table 3 of the main paper. Exact reconstruction remains challenging for complex cavities from collision observations alone, but hypothesis-space reasoning produces more coherent structures and more physically plausible deformation dynamics. Corresponding supplementary videos are available on the project page.