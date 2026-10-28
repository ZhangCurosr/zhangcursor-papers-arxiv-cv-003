# TriDifNet: A Unified Three-Task Diffusion Network for End-to-End RGB-T Perception

Qianchao He, Jing Liu, Senior Member, IEEE, Yunhao Li and Xiaoqing Tian

Abstract—Infrared–visible (IR–VIS) multi-modal perception is important for robust scene understanding under challenging illumination conditions. In practice, infrared and visible images are often misaligned due to platform motion, viewpoint changes, and modality-dependent distortions, causing registration errors to propagate into subsequent fusion and detection stages. To address this issue, we propose TriDifNet, a unified three-task diffusion network for end-to-end RGB-T perception. To the best of our knowledge, TriDifNet is the first diffusion-based network that jointly optimizes registration, fusion, and detection for misaligned RGB-T pedestrian perception.

TriDifNet consists of three diffusion-based modules, i.e., the Registration Diffusion Module (RegDM), the Fusion Diffusion Module (FusDM), and the Detection Diffusion Module (DetDM). RegDM aligns the unaligned infrared image to the optical reference by estimating a dense deformation field. FusDM performs residual conditional diffusion to generate a detectionoriented fused representation with optical structural cues and complementary infrared saliency. DetDM formulates pedestrian detection as an iterative denoising process in bounding-box space. To couple the three tasks, TriDifNet introduces a closedloop optimization mechanism, where detection semantics guide fusion and registration, and the fusion result further regularizes geometric alignment. A stage-wise progressive training strategy and a lightweight Nash multi-task learning (Nash-MTL)-based late-stage refinement are further adopted to balance registration, fusion, and detection optimization.

Extensive experiments on the KAIST multispectral pedestrian benchmark show that TriDifNet achieves superior downstream detection performance under misaligned RGB-T settings. Against a comprehensive set of strong system-level competitors, TriDifNet achieves 65.44% Recall, 57.55% mAP<sub>50</sub>, and 26.83% mAP<sub>50:95</sub>.

Index Terms—Multispectral pedestrian detection, misaligned RGB-T perception, infrared–visible image registration and fusion, diffusion models, multi-task learning.

## I. INTRODUCTION

NFRARED–VISIBLE(IR–VIS) multi-modal perception has become increasingly important in safety-critical scenarios such as night-time surveillance, autonomous driving, and search-and-rescue. Visible images provide rich texture and structural details, whereas infrared imagery highlights thermal saliency and remains robust under adverse illumination. In practice, however, the two modalities are often spatially misaligned due to platform motion, viewpoint variation, capture timing differences, and modality-dependent imaging distortions. The core challenge is therefore not merely the presence of cross-modal misalignment itself, but the fact that geometric alignment, cross-modal fusion, and task-level semantics are usually optimized in a decoupled manner. Recent studies have shown that registration and fusion should not be treated as entirely independent stages [1], and that bidirectional interaction between them can improve both alignment quality and fusion fidelity [2]. More recent efforts have further explored tighter registration–fusion integration within unified or single-stage methods [3], [4]. For example, IMF introduces a one-stage progressive dense registration scheme and jointly optimizes it with a Transformer–Conv fusion subnetwork to mitigate structural distortions and edge ghosts in misaligned infrared–visible fusion [4]. Nevertheless, object detection in most existing registration–fusion systems still remains an external downstream consumer of the fused result rather than an endogenous optimization target inside the perception loop.

A representative limitation appears in conventional sequential pipelines that perform registration first and fusion afterward. In such methods, registration is optimized before knowing which structures and regions are truly critical for downstream recognition, while fusion is forced to absorb the residual errors inherited from upstream geometric inaccuracies. This issue becomes more severe in practical misaligned IR–VIS scenarios, where explicit alignment is difficult [5] and cross-modal appearance discrepancy is large [6]. As a result, even mild misregistration can cause ghosting, double edges, structural distortion, or target deformation in the fused result [4], [7]. In complex real-world sensing conditions, such sensitivity to misalignment can further degrade the effectiveness of subsequent fusion-based perception [8]. Consequently, the fusion stage is often reduced to a compensatory process on imperfectly aligned inputs, which limits both semantic fidelity and downstream robustness.

Another line of work attempts to inject high-level semantics into fusion, for example by using segmentation or detection signals to guide the generation of fused representations [9]– [12]. Recent VIS–IR detection-oriented fusion methods further show that fusion and detection should be optimized in a taskconsistent manner. For instance, MMI-Det explores multilevel visible–infrared integration with contour enhancement, fusion-focused feature interaction, contrastive bridging, and information-guided optimization to better balance fusion and detection objectives [13]. EI2Det further introduces crossmodal interaction, illumination-aware modality weighting, and edge-guided fusion to improve detection robustness under varying lighting conditions [14]. In parallel, recent weakly aligned RGB-T detection studies have recognized that strict cross-modal alignment is often unrealistic in practice [15]– [18]. Uncertainty-aware cross-modal detectors also indicate that practical RGB–IR perception suffers from object-level uncertainty caused by low-light visible observations, confusing infrared responses, thermal ghost shadows, and pose-induced cross-modal misalignment [19]. Although these efforts demonstrate the value of task semantics, modality reliability, and alignment-aware detection, detection is still commonly treated either as an auxiliary guidance source for fusion or as a separate downstream task. As a result, it is rarely incorporated as a trainable component that jointly shapes alignment, fusion behavior, and final recognition. Consequently, what remains missing is a unified network in which geometric alignment, cross-modal fusion, and task-level detection semantics are optimized within a shared closed loop.

Recently, diffusion models have demonstrated strong capacity for high-fidelity generation and restoration [20]–[23], making them appealing for misaligned IR–VIS perception. More importantly, diffusion provides a unified iterative refinement view for the three tasks considered in this work. For registration, cross-modal matching under weak affine misalignment still involves spatially varying deformation estimation, which can be naturally cast as a progressive denoising process over deformation fields [2], [24]. For fusion, the optical image already provides a stable structural base, while the complementary infrared contribution is better modeled as a conditional residual restoration problem rather than direct fullimage generation [25], [26]. For detection, object localization can be reformulated as iterative denoising in the boundingbox space, enabling progressive refinement from noisy box hypotheses to final object predictions [27]. Therefore, instead of treating registration, fusion, and detection as heterogeneous modules coupled only at the loss level, diffusion offers a common optimization perspective in which geometry, crossmodal representation, and task semantics can all be refined through task-specific denoising processes. This shared formulation is particularly suitable for the proposed closed-loop design, as it makes the three tasks not only jointly trainable, but also conceptually unified under iterative error correction and mutual refinement.

Building on this unified denoising perspective, we propose TriDifNet, a unified three-task diffusion network that jointly optimizes registration, fusion, and detection in an end-toend manner. Instead of following the conventional “registration → fusion → detection” chain, TriDifNet explicitly couples the three tasks within one optimization system. First, a Registration Diffusion Module (RegDM) performs conditional diffusion to estimate a dense deformation field and warp the unaligned infrared image toward the optical reference. Then, a Fusion Diffusion Module (FusDM) performs residual conditional diffusion under the joint guidance of the optical image and the registered infrared image. This residual parameterization is adopted since the optical image already provides a stable structural base, while the diffusion model only needs to generate the complementary cross-modal component that injects infrared saliency into target-relevant regions.

To prevent detection from remaining a detached downstream head, we further elevate it into a trainable third task through a Detection Diffusion Module (DetDM), which formulates object detection as an iterative denoising process in boundingbox space. DetDM operates on the detection-oriented fusion representation and produces the final pedestrian predictions, while simultaneously feeding task semantics back into the upstream modules. In particular, the fusion branch is constrained by an ROI-guided task-driven supervision, denoted as Det2Fus, so that target-centered regions receive stronger infrared enhancement and more discriminative fusion behavior. In parallel, registration is regularized by infrared-domain object-level semantic consistency between the misaligned and registered infrared images, denoted as Det2Reg. For this purpose, we employ a frozen YOLO detector only during training as a stable infrared-domain semantic anchor, rather than reusing the trainable fusion-side detector. This allows the registration branch to receive consistent object-level supervision without being affected by the optimization drift of the fusion-domain detection process. Finally, the fusion result is also fed back to regularize registration through a structural fusion-to-registration constraint, denoted as Fus2Reg, forming a closed-loop optimization over the three tasks. To stabilize optimization across the three branches, TriDifNet is trained in a stage-wise progressive manner. After full closed-loop joint training, we further perform a lightweight Nash-MTLbased [28] late-stage refinement across the registration, fusion, and detection branches, so that the three task objectives can be dynamically balanced in the final optimization stage.

In this way, TriDifNet is not a simple concatenation of three modules, but a jointly optimized perception network in which geometry, fusion, and semantics are mutually constrained. RegDM improves the geometric basis for both fusion and recognition; FusDM builds a cross-modal representation that is explicitly shaped by downstream task demands; and DetDM not only produces the final predictions but also acts as a semantic driver for upstream learning. Through this closedloop design, the proposed network reduces early-stage error propagation and promotes coordinated improvement in alignment quality, cross-modal complementarity, and pedestrian detection performance.

The main contributions of this work are summarized as follows:

• We propose a unified three-task end-to-end network for misaligned RGB-T pedestrian perception, which jointly optimizes infrared–visible registration, fusion, and detection within a shared closed loop instead of treating them as isolated or sequential stages.

• We develop a diffusion-based multi-module formulation for the three tasks, in which RegDM performs conditional deformation estimation for cross-modal alignment, FusDM performs residual conditional diffusion for complementary infrared–visible fusion, and DetDM formulates pedestrian detection as an iterative denoising process in bounding-box space.

• We design explicit cross-task coupling mechanisms, including Det2Reg for detection-guided registration via a frozen infrared-domain semantic anchor, Det2Fus for ROI-guided task-driven fusion, and Fus2Reg for structural fusion-to-registration feedback, thereby establishing a genuine closed-loop optimization among geometry, fusion, and semantics.

• By integrating the above components into TriDifNet and employing a stage-wise training strategy with a lightweight late-stage three-branch Nash-MTL refinement, the proposed method achieves clear end-to-end detection improvements under misaligned RGB-T inputs. On the KAIST multispectral pedestrian benchmark with controlled RGB-T misalignment, TriDifNet reaches 65.44% Recall, 57.55% $\mathrm { \ m A P _ { 5 0 } }$ , and 26.83% $\mathrm { m A P _ { 5 0 : 9 5 } }$ . The results show superior system-level performance against strong competing pipelines under the same setting.

## II. BRIEF INTRODUCTION TO DIFFUSION MODEL

Diffusion models are likelihood-based generative models that learn a data distribution by simulating a gradual noising process and then learning its denoising reverse process [29]. This section briefly introduces the standard DDPM formulation, including forward diffusion, reverse diffusion, and conditional diffusion.

a) Forward diffusion: Let $\mathbf { x } _ { 0 } \sim q ( \mathbf { x } _ { 0 } )$ denote a clean target variable. A diffusion model defines a Markov forward process that gradually perturbs $\mathbf { x } _ { \mathrm { 0 } }$ into Gaussian noise:

$$
q ( \mathbf { x } _ { 1 : T } \mid \mathbf { x } _ { 0 } ) = \prod _ { t = 1 } ^ { T } q ( \mathbf { x } _ { t } \mid \mathbf { x } _ { t - 1 } )\tag{1}
$$

where each transition is commonly chosen as

$$
q ( \mathbf { x } _ { t } \mid \mathbf { x } _ { t - 1 } ) = \mathcal { N } \Big ( \mathbf { x } _ { t } ; \sqrt { 1 - \beta _ { t } } \mathbf { x } _ { t - 1 } , \beta _ { t } \mathbf { I } \Big )\tag{2}
$$

with $\{ \beta _ { t } \} _ { t = 1 } ^ { T }$ a variance schedule. Define $\alpha _ { t } = 1 - \beta _ { t }$ and $\begin{array} { r } { \bar { \alpha } _ { t } = \prod _ { s = 1 } ^ { t } \alpha _ { s } } \end{array}$ . Then the marginal distribution admits a closed form:

$$
\begin{array} { r l } & { q ( \mathbf { x } _ { t } \mid \mathbf { x } _ { 0 } ) = \mathcal { N } \Big ( \mathbf { x } _ { t } ; \sqrt { \bar { \alpha } _ { t } } \mathbf { x } _ { 0 } , ( 1 - \bar { \alpha } _ { t } ) \mathbf { I } \Big ) , } \\ & { \qquad \mathbf { x } _ { t } = \sqrt { \bar { \alpha } _ { t } } \mathbf { x } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \pmb \epsilon } \end{array}\tag{3}
$$

where $\mathbf { \epsilon } \epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$

b) Reverse diffusion: The generative reverse process learns to reconstruct $\mathbf { x } _ { \mathrm { 0 } }$ from $\mathbf { x } _ { T } \mathbf { \cdot }$ :

$$
p _ { \theta } ( \mathbf { x } _ { 0 : T } \mid \mathbf { x } _ { T } ) = \prod _ { t = 1 } ^ { T } p _ { \theta } ( \mathbf { x } _ { t - 1 } \mid \mathbf { x } _ { t } )\tag{4}
$$

where each reverse step is parameterized as a Gaussian

$$
p _ { \theta } ( \mathbf { x } _ { t - 1 } \mid \mathbf { x } _ { t } ) = \mathcal { N } \Big ( \mathbf { x } _ { t - 1 } ; \pmb { \mu } _ { \theta } ( \mathbf { x } _ { t } , t ) , \pmb { \sigma } _ { \theta } ^ { 2 } ( \mathbf { x } _ { t } , t ) \mathbf { I } \Big )\tag{5}
$$

Sampling starts from $\mathbf { x } _ { T } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ and iteratively applies Eq. (5) to obtain $\mathbf { x } _ { 0 } .$ . In practice, the denoiser is commonly trained to predict the injected noise ϵ, which leads to a simple and stable optimization objective.

c) Conditional diffusion and guidance: For vision tasks, diffusion is typically used in a conditional manner, where the reverse process is guided by an observation y:

$$
p _ { \theta } ( \mathbf { x } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { y } ) = \mathcal { N } \Big ( \mathbf { x } _ { t - 1 } ; \pmb { \mu } _ { \theta } ( \mathbf { x } _ { t } , t , \mathbf { y } ) , \pmb { \sigma } _ { \theta } ^ { 2 } ( \mathbf { x } _ { t } , t , \mathbf { y } ) \mathbf { I } \Big )\tag{6}
$$

Compared with unconditional diffusion, conditional diffusion introduces task-related observations into the denoising process, enabling the model to generate or restore target variables under specific visual constraints.

## III. METHODOLOGY

## A. Overview

This section describes the architecture and optimization of TriDifNet. As illustrated in Fig. 1, TriDifNet is organized as a modular closed-loop network, where registration, fusion, and detection are executed sequentially in the forward path and are further coupled through cross-task feedback during training.

TriDifNet consists of three core components:

1) Registration Diffusion Module (RegDM): estimates a dense deformation field to align the unaligned infrared image with the optical reference image, producing the registered infrared input for fusion.

2) Fusion Diffusion Module (FusDM): performs residual conditional diffusion to generate a detection-oriented fused representation from the optical image and the registered infrared image.

3) Detection Diffusion Module (DetDM): formulates object detection as a denoising diffusion process in bounding-box space and performs ROI-guided reinference for final prediction.

The three components are connected by explicit crosstask interactions. In the forward direction, RegDM provides aligned infrared guidance to FusDM (Reg-to-Fus), and FusDM supplies the detector-oriented representation to DetDM (Fusto-Det). In the feedback direction, the fusion result regularizes registration through structural consistency (Fus2Reg), a frozen infrared-domain detector provides semantic supervision for detection-guided registration (Det2Reg), and the detector feeds ROI-guided task semantics back to the fusion branch (Det2Fus). Through these interactions, the three tasks are jointly refined within a unified closed-loop network.

The remainder of this section is organized as follows. Sec. III-B defines the notation and formulates the three-task problem. Sec. III-C details the Registration Diffusion Module (RegDM). Sec. III-D presents the Fusion Diffusion Module (FusDM). Sec. III-E introduces the Detection Diffusion Module (DetDM). Sec. III-F describes the cross-task coupling paths among registration, fusion, and detection. Sec. III-G presents the stage-wise training strategy and the Nash-MTL-based latestage refinement. Finally, Sec. III-H summarizes the inference pipeline.

## B. Notation and Problem Formulation

Let $Y \in \mathbb { R } ^ { H \times W \times 3 }$ denote the optical (visible) image, and let $Z ^ { \prime } \in \mathbb { R } ^ { H \times W \times 1 }$ denote the unaligned infrared image. The goal of TriDifNet is to jointly optimize registration, fusion, and detection under the misaligned RGB-T setting.

![](images/bafbe9ae15780d8d7a0b66f708bcffa126fb1a8f15df47b8db1b7ff4b269d1e2.jpg)  
Fig. 1. Overall architecture of TriDifNet. (a) Task overview. Given a visible image $Y$ and an unaligned infrared image $Z ^ { \prime } ,$ TriDifNet sequentially produces the registered infrared image $\hat { Z } ,$ the fused image ${ \hat { X } } ,$ and the final detection result ${ \hat { D } } _ { 3 }$ . (b) Closed-loop tri-task diffusion network. RegDM estimates a dense deformation field through conditional deformation-field diffusion and spatial warping; FusDM generates a detection-oriented fused representation through residual conditional diffusion; and DetDM performs box-space conditional diffusion followed by ROI-guided re-inference. Black arrows denote the forward data flow, while red dashed arrows denote cross-task feedback paths, including Det2Reg, Fus2Reg, and Det2Fus. The frozen infrared-domain detector provides semantic guidance during training and is not updated with the TriDifNet parameters. (c) Symbol description.

In the current work, the cross-modal discrepancy is modeled as controlled weak affine misalignment. Specifically, the infrared input is perturbed by in-plane rotation and translation with respect to the visible reference. This setting provides a practical weakly misaligned RGB-T scenario for evaluating whether registration, fusion, and detection can be optimized consistently within a unified network.

Given $Y$ and $Z ^ { \prime } ,$ , the Registration Diffusion Module (RegDM) estimates an inverse deformation field

$$
\hat { \phi } _ { 0 } ^ { - 1 } = \mathrm { R e g D M } ( Y , Z ^ { \prime } )\tag{7}
$$

which is used to warp the unaligned infrared image into the optical reference coordinate system:

$$
\hat { Z } = W ( Z ^ { \prime } , \hat { \phi } _ { 0 } ^ { - 1 } )\tag{8}
$$

where $W ( \cdot , \cdot )$ denotes the spatial warping operation. Here, $\hat { Z } \in \mathbb { R } ^ { H \times \dot { W } \times 1 }$ is the registered infrared image.

Based on $Y$ and $\hat { Z } ,$ the Fusion Diffusion Module (FusDM) predicts a residual fusion component

$$
\hat { X } _ { \mathrm { r e s } } = \mathrm { F u s D M } ( Y , \hat { Z } )\tag{9}
$$

and the final fused output is obtained by residual reconstruction

$$
{ \hat { X } } = Y + { \hat { X } } _ { \mathrm { r e s } }\tag{10}
$$

where $\hat { X } \in \mathbb { R } ^ { H \times W \times C }$ denotes the fused representation.<sup>1</sup> The detection-oriented fusion representation that is first fed into the Detection Diffusion Module (DetDM) is denoted by $F _ { \mathrm { d e t } }$

which is obtained from the FusDM output after a visible– fusion cross-attention interaction module and serves as the first-pass detector input.

The Detection Diffusion Module (DetDM) performs object detection in the bounding-box space. Given $F _ { \mathrm { d e t } }$ , the detector first produces an initial prediction

$$
D _ { 3 } = \mathrm { D e t D M } ( F _ { \mathrm { d e t } } )\tag{11}
$$

where $D _ { 3 }$ denotes the first-pass detection result on the fusionside detector input. Using the ground-truth box prior and the prediction-derived ROI prior, a mixed ROI mask $M _ { \mathrm { m i x } }$ is constructed, and the detector-oriented representation is further refined into the task-driven detector representation $F _ { \mathrm { d e t } } ^ { \mathrm { t d } }$ , which is used for the task-driven re-inference stage of DetDM. Therefore, the final fusion-side detection can be written as

$$
\hat { D } _ { 3 } = \mathrm { D e t D M } \big ( F _ { \mathrm { d e t } } ^ { \mathrm { t d } } \big )\tag{12}
$$

In addition to fusion-side detection, TriDifNet also introduces infrared-domain detection for registration guidance. Specifically, a frozen infrared detector is applied to the unregistered and registered infrared images, yielding

$$
D _ { 1 } = { \mathrm { D e t } } _ { \mathrm { I R } } ( Z ^ { \prime } ) , \qquad D _ { 2 } = { \mathrm { D e t } } _ { \mathrm { I R } } ( { \hat { Z } } )\tag{13}
$$

where $D _ { 1 }$ and $D _ { 2 }$ provide stable semantic supervision for detection-guided registration.

Overall, TriDifNet can be summarized as a three-task closed-loop system:

$$
( Y , Z ^ { \prime } ) \xrightarrow { \mathrm { \ R e g D M } } \hat { Z } \xrightarrow { \mathrm { \ F u s D M } } \hat { X } \xrightarrow { \mathrm { \ D e t D M } } \hat { D } _ { 3 }\tag{14}
$$

with three cross-task coupling paths:

• Det2Reg: infrared-domain semantic consistency from $D _ { 1 }$ and $D _ { 2 }$ regularizes RegDM;

• Det2Fus: detection semantics derived from $D _ { 3 }$ and $M _ { \mathrm { m i x } }$ reshape the fusion-side detector representation from $F _ { \mathrm { d e t } }$ to $F _ { \mathrm { d e t } } ^ { \mathrm { t d } }$ and feed back to FusDM;

• Fus2Reg: the fused representation further regularizes registration through structural feedback.

Accordingly, the objective of this work is not to optimize registration, fusion, and detection independently, but to jointly learn them within a unified end-to-end network in which geometry, cross-modal fusion, and task-level semantics are mutually constrained.

## C. Registration Diffusion Module (RegDM)

1) Conditional Diffusion for Deformation Field Estimation: The Registration Diffusion Module (RegDM) formulates dense infrared-to-optical alignment as a conditional diffusion denoising process over the inverse deformation field. Given the optical reference image Y and the unaligned infrared image $Z ^ { \prime }$ , RegDM aims to estimate an inverse deformation field that maps the source infrared image into the optical reference coordinate system.

At diffusion step t, the model takes a noisy deformation representation $\phi _ { t } ^ { - 1 }$ together with a registration condition tensor con $\lfloor _ { R }$ and predicts the clean inverse deformation field. Under the current IR–optical setting, the condition tensor is constructed by concatenating the optical reference and the unaligned infrared input along the channel dimension:

$$
\mathrm { c o n d } _ { R } = \mathrm { C o n c a t } ( Y , Z ^ { \prime } ) \in \mathbb { R } ^ { H \times W \times 4 }\tag{15}
$$

The RegDM denoiser $f _ { \phi } ( \cdot )$ predicts the injected Gaussian noise in the inverse deformation diffusion process as

$$
\hat { \epsilon } _ { R } = f _ { \phi } ( \phi _ { t } ^ { - 1 } , \mathrm { c o n d } _ { R } , t )\tag{16}
$$

According to the standard diffusion reconstruction form, the clean inverse deformation field can then be recovered as

$$
\hat { \phi } _ { 0 } ^ { - 1 } = \frac { 1 } { \sqrt { \bar { \alpha } _ { t } } } \left( \phi _ { t } ^ { - 1 } - \sqrt { 1 - \bar { \alpha } _ { t } } \hat { \epsilon } _ { R } \right)\tag{17}
$$

where $\bar { \alpha } _ { t }$ denotes the cumulative noise coefficient defined in Sec. II.

The predicted inverse deformation field is then applied to the unaligned infrared image through a spatial warping operation:

$$
\hat { Z } = \mathcal { W } ( Z ^ { \prime } , \hat { \phi } _ { 0 } ^ { - 1 } )\tag{18}
$$

where $\mathcal { W } ( \cdot , \cdot )$ denotes the spatial warping operation. The resulting $\hat { Z }$ is expected to be geometrically aligned with the optical reference image $Y$ and subsequently serves as the infrared condition for FusDM.

In this way, RegDM provides the geometric foundation of the overall network: it first reduces cross-modal misalignment, then supplies the registered infrared image to the fusion branch, and meanwhile remains open to semantic and structural feedback from the downstream tasks through the closedloop optimization.

2) Registration Objective: RegDM is optimized with a composite objective that balances diffusion reconstruction, deformation smoothness, and cross-modal alignment consistency:

$$
L _ { R } = L _ { R } ^ { \mathrm { D i f f } } + \alpha L _ { R } ^ { \mathrm { s m o o t h } } + \beta L _ { R } ^ { \mathrm { a l i g n } }\tag{19}
$$

Here, $L _ { R } ^ { \mathrm { D i f f } }$ denotes the diffusion reconstruction loss for denoising the inverse deformation field. Following the standard diffusion training formulation, the denoiser is trained to predict the injected Gaussian noise:

$$
L _ { R } ^ { \mathrm { D i f f } } = \mathbb { E } _ { \phi _ { 0 } ^ { - 1 } , \epsilon , t } \left[ \left\| \epsilon - f _ { \phi } ( \phi _ { t } ^ { - 1 } , \mathrm { c o n d } _ { R } , t ) \right\| _ { 2 } ^ { 2 } \right]\tag{20}
$$

where $\epsilon \sim \mathcal { N } ( 0 , I )$ and $\phi _ { t } ^ { - 1 }$ is obtained by the forward diffusion process from the clean inverse deformation field.

The smoothness term regularizes the spatial continuity of the predicted deformation field:

$$
L _ { R } ^ { \mathrm { s m o o t h } } = \left. \nabla _ { x } \hat { \phi } _ { 0 } ^ { - 1 } \right. _ { 1 } + \left. \nabla _ { y } \hat { \phi } _ { 0 } ^ { - 1 } \right. _ { 1 }\tag{21}
$$

where $\nabla _ { x }$ and $\nabla _ { y }$ denote horizontal and vertical finitedifference operators, respectively. This term discourages unrealistic discontinuities and stabilizes the registration process.

The alignment term enforces cross-modal consistency between the registered infrared image and the optical reference:

$$
L _ { R } ^ { \mathrm { a l i g n } } = 1 - \mathrm { N C C } ( \hat { Z } , Y _ { \mathrm { g r a y } } )\tag{22}
$$

where $Y _ { \mathrm { g r a y } }$ denotes the grayscale projection of the optical image used for cross-modal similarity measurement, and $\operatorname { N C C } ( \cdot , \cdot )$ is the normalized cross-correlation operator. This term encourages the warped infrared image to be geometrically consistent with the visible reference while remaining robust to cross-modal intensity differences.

Therefore, the overall role of RegDM is twofold: it explicitly estimates the dense geometric correction needed for misaligned RGB-T inputs, and it provides the aligned infrared representation required by the subsequent fusion and detection processes.

## D. Fusion Diffusion Module (FusDM)

1) Residual Conditional Diffusion for Infrared–Visible Fusion: After RegDM produces the registered infrared image Z<sup>ˆ</sup>, the Fusion Diffusion Module (FusDM) performs conditional diffusion to generate a cross-modal residual representation under the joint guidance of the optical image and the aligned infrared image. Unlike direct full-image generation, FusDM adopts a residual formulation since the optical image Y already provides a stable structural and textural base, while the fusion branch mainly needs to inject complementary infrared saliency into target-relevant regions.

Formally, the fusion condition tensor is defined as

$$
\mathrm { c o n d } _ { F } = \mathrm { C o n c a t } ( Y , \hat { Z } )\tag{23}
$$

At diffusion step t, the FusDM denoiser $f _ { \psi } ( \cdot )$ takes the noisy residual fusion state $x _ { t }$ together with cond<sub>F</sub> and predicts the clean residual component:

$$
\hat { X } _ { \mathrm { r e s } } = f _ { \psi } ( x _ { t } , \mathrm { c o n d } _ { F } , t )\tag{24}
$$

The final fused representation is reconstructed by residual addition:

$$
{ \hat { X } } = Y + { \hat { X } } _ { \mathrm { r e s } }\tag{25}
$$

where $\hat { X } \in \mathbb { R } ^ { H \times W \times C }$ denotes the fused output of FusDM. This residual design explicitly preserves the dominant optical structures while allowing the diffusion model to focus on modality-complementary information from the infrared branch.

Therefore, FusDM serves as the cross-modal representation hub of the overall network: it receives the geometrically corrected infrared signal from RegDM, produces the fused representation for image-level perception, and simultaneously supplies the detector-oriented fusion cue for the downstream detection branch.

2) Detector-Oriented Fusion Representation: In addition to generating the fused output ${ \hat { X } } ,$ FusDM also provides a detector-oriented representation for DetDM. Rather than using a separate post-processing branch, this representation is built directly within the fusion pipeline, so that it can be refined by downstream detection semantics.

Specifically, let $\tilde { F } _ { \mathrm { d e t } } ~ = ~ \hat { X } ^ { ( 4 ) }$ denote the raw detectionoriented fusion channel extracted from the FusDM output, where $\hat { X } ^ { ( 4 ) }$ is the detector-side single-channel output before visible–fusion interaction. To preserve detector-friendly optical structural cues, we introduce a visible–fusion cross-attention module to generate the first-pass detector input $F _ { \mathrm { d e t } }$ . This helps keep the detector input closer to the original opticalimage setting of DiffusionDet [27], while complementing the subsequent ROI-guided infrared enhancement stage.

We first encode the optical image and the raw detectororiented fusion channel as

$$
E _ { \mathrm { v i s } } = \phi _ { \mathrm { v i s } } ( Y ) , \qquad E _ { \mathrm { f u s } } = \phi _ { \mathrm { f u s } } ( \mathrm { R e p 3 } ( \tilde { F } _ { \mathrm { d e t } } ) )\tag{26}
$$

where $\phi _ { \mathrm { v i s } } ( \cdot )$ and $\phi _ { \mathrm { f u s } } ( \cdot )$ denote the optical and fusion feature encoders, respectively, and $\mathrm { R e p 3 ( \cdot ) }$ denotes the channelreplication operator that expands a single-channel fusion map into a three-channel tensor for detector-side processing.

Based on these two feature embeddings, the visible-guided cross-attention is formulated as

$$
Q = W _ { Q } E _ { \mathrm { v i s } } , \qquad K = W _ { K } E _ { \mathrm { f u s } } , \qquad V = W _ { V } E _ { \mathrm { f u s } }\tag{27}
$$

where $W _ { Q } , \ W _ { K }$ , and $W _ { V }$ are learnable linear projection operators for query, key, and value, respectively. The attention map is computed by

$$
A _ { \mathrm { v f } } = \mathrm { S o f t m a x } \bigg ( \frac { Q K ^ { \top } } { \sqrt { d } } \bigg )\tag{28}
$$

where d is the feature dimension. The detector-oriented fusion representation after visible–fusion interaction is then obtained as

$$
F _ { \mathrm { d e t } } = { \mathcal { P } } ( E _ { \mathrm { v i s } } + A _ { \mathrm { v f } } V )\tag{29}
$$

where $\mathcal { P } ( \cdot )$ denotes the projection operator that maps the cross-attended feature representation back to the detector input representation space.

The resulting $F _ { \mathrm { d e t } }$ serves as the first-pass detector input for DetDM. Therefore, the fusion branch provides not only the fused image representation $\hat { X }$ for image-level perception, but

also a cross-attention-enhanced detector-oriented representation $F _ { \mathrm { d e t } }$ that bridges fusion and detection.

In this sense, FusDM plays two roles in TriDifNet:

• it produces the fused image representation $\hat { X }$ for crossmodal perception;

• it generates the detector-oriented representation $F _ { \mathrm { d e t } }$ that bridges fusion and detection.

This design ensures that the fusion branch is not only optimized for visual fusion quality, but also prepared for semantic feedback from the downstream detector.

3) Fusion Objective: The base optimization of FusDM is defined as

$$
L _ { F } = L _ { F } ^ { \mathrm { D i f f } } + \lambda _ { 1 } L _ { F } ^ { \mathrm { p i x } } + \lambda _ { 2 } L _ { F } ^ { \mathrm { s t r } }\tag{30}
$$

where $L _ { F } ^ { \mathrm { D i f f } }$ is the diffusion reconstruction loss for the residual fusion process, $L _ { F } ^ { \mathrm { p i x } }$ enforces pixel-level fidelity between the fused output and the fusion target, and $L _ { F } ^ { \mathrm { s t r } }$ preserves structural consistency in the fused representation.

The diffusion term is written as

$$
L _ { F } ^ { \mathrm { D i f f } } = \mathbb { E } _ { x _ { 0 } , \epsilon , t } \left[ \big | | \epsilon - f _ { \psi } ( x _ { t } , \mathrm { c o n d } _ { F } , t ) \big | \big | _ { 2 } ^ { 2 } \right]\tag{31}
$$

where $\epsilon \sim \mathcal { N } ( 0 , I )$ and $x _ { t }$ is the noisy residual state sampled from the forward diffusion process. The pixel-level fidelity term is defined as

$$
L _ { F } ^ { \mathrm { p i x } } = \Big \| \hat { X } - X _ { \mathrm { g t } } \Big \| _ { 1 }\tag{32}
$$

where $X _ { \mathrm { g t } }$ denotes the fusion supervision target. The structural regularization term is defined as

$$
L _ { F } ^ { \mathrm { s t r } } = 1 - \mathrm { S S I M } ( \hat { X } , X _ { \mathrm { g t } } )\tag{33}
$$

which encourages the fused output to preserve perceptually meaningful scene structures.

It should be noted that the above objective only defines the base fusion optimization of FusDM. In the full threetask network, this objective is further augmented by the taskdriven Det2Fus constraints and the closed-loop cross-task feedback. The corresponding cross-task coupling mechanisms are formulated in Sec. III-F. With these additional couplings, the fusion branch is ultimately optimized not only for imagelevel fusion quality but also for detection-oriented semantic utility.

## E. Detection Diffusion Module (DetDM)

1) Diffusion-Based Detection Formulation: To prevent detection from remaining a detached downstream task, TriDifNet introduces a Detection Diffusion Module (DetDM) as the third trainable component in the closed-loop system. Instead of directly predicting object boxes in a one-shot manner, DetDM formulates pedestrian detection as an iterative denoising process in the bounding-box space. Given a detector-oriented fusion representation, the detector starts from a set of noisy box proposals and progressively refines them into the final object predictions.

Given the first-pass detector input $F _ { \mathrm { d e t } }$ obtained from the cross-attention-enhanced fusion representation, DetDM first generates an initial fusion-side prediction:

$$
D _ { 3 } = \mathrm { D e t D M } ( F _ { \mathrm { d e t } } )\tag{34}
$$

![](images/3266dcac8ba4cfa97d234a938628dbccff21d7aa9b27b90fbf9459ad1953ad32.jpg)  
Fig. 2. ROI-guided re-inference in DetDM. First-pass predictions are converted into an ROI prior, which guides ROI-IR injection for second-pass detection.

where $D _ { 3 }$ contains the first-pass pedestrian predictions, including the box locations and confidence scores on the fusion-side detector representation.

2) ROI-Guided Re-inference: Unlike a conventional singlepass detector, the current DetDM does not directly use the firstpass prediction $D _ { 3 }$ as the final output. Instead, $D _ { 3 }$ is converted into an ROI prior and used to refine the detector-side fusion representation for second-pass detection.

As shown in Fig. 2, this re-inference process consists of four steps: D3 first-pass detection, ROI prior construction, ROI-IR injection, and second-pass detection. To construct the task-driven refinement prior, the ground-truth boxes are first converted into a soft ROI mask:

$$
M _ { \mathrm { { g t } } } = \mathcal { R } ( B _ { \mathrm { { g t } } } )\tag{35}
$$

where $B _ { g t }$ denotes the ground-truth bounding-box annotations, and $R ( \cdot )$ denotes the ROI mask generation operator that converts box annotations into a soft ROI mask. Similarly, the first-pass prediction $D _ { 3 }$ is converted into a prediction-derived ROI mask:

$$
M _ { d 3 } = \mathcal { R } ( D _ { 3 } )\tag{36}
$$

To balance supervision from the ground-truth prior and the detector prediction prior, a mixed ROI mask is constructed as

$$
M _ { \mathrm { { m i x } } } = \beta M _ { \mathrm { { g t } } } + ( 1 - \beta ) M _ { d 3 }\tag{37}
$$

where $\beta$ is an adaptive mixing coefficient determined by the agreement between the prediction-derived ROI and the groundtruth ROI. In practice, the agreement is measured by the masklevel IoU between $M _ { d 3 }$ and $M _ { \mathrm { { g t } } }$ , and $\beta$ is constrained within a GT-dominant interval to prevent the mixed ROI prior from being overly dominated by unstable early predictions.

The mixed ROI mask is then used to refine the first-pass detector input $F _ { \mathrm { d e t } }$ into a task-driven detector representation:

$$
F _ { \mathrm { d e t } } ^ { \mathrm { t d } } = { \mathcal { T } } ( F _ { \mathrm { d e t } } , { \hat { Z } } , M _ { \mathrm { m i x } } )\tag{38}
$$

where $\tau ( \cdot )$ denotes the ROI-guided task-driven rewriting operator. Intuitively, this operation preserves the original detectororiented fusion response in non-target regions while injecting stronger infrared saliency into target-centered ROI regions. As a result, the detector input is no longer a purely fusion-driven representation, but a task-aware representation explicitly reshaped by the object prior.

The refined representation is then used for the second-pass detection:

$$
\hat { D } _ { 3 } = \mathrm { D e t D M } ( F _ { \mathrm { d e t } } ^ { \mathrm { t d } } )\tag{39}
$$

where ${ \hat { D } } _ { 3 }$ denotes the final fusion-side detection output.

3) Detection Objective: The optimization of DetDM consists of two coupled levels. At the detector level, we adopt the box-space diffusion detection objective to supervise the final second-pass prediction ${ \hat { D } } _ { 3 } \colon$

$$
L _ { D } = L _ { D } ^ { \mathrm { c l s } } + L _ { D } ^ { \mathrm { b o x } } + L _ { D } ^ { \mathrm { g i o u } }\tag{40}
$$

where $L _ { D } ^ { \mathrm { c l s } }$ is the classification loss, $L _ { D } ^ { \mathrm { b o x } }$ is the box regression loss, and $\overline { { L _ { D } ^ { \mathrm { g i o u } } } }$ is the generalized intersection-over-union loss.

At the representation level, the detector is further coupled with the fusion branch through ROI-guided task-driven constraints. These constraints do not merely supervise the final boxes; instead, they act on the detector-oriented fusion representation itself through $M _ { \mathrm { m i x } }$ and $F _ { \mathrm { d e t } } ^ { \mathrm { t d } }$ . Consequently, the detector not only predicts pedestrian instances, but also provides semantic feedback that reshapes the fusion-side representation before final inference.

Therefore, the overall role of DetDM in TriDifNet is twofold:

• it serves as the final pedestrian detector operating on fusion-side representations;

• it acts as a task-semantic driver that constructs the ROIguided feedback used by the Det2Fus pathway.

In this way, detection is lifted from a passive downstream consumer to an endogenous component of the closed-loop optimization, enabling the final detector to jointly influence fusion behavior and, through the cross-task constraints, the overall RGB-T perception process.

## F. Cross-Task Coupling: Registration–Fusion–Detection

A key contribution of TriDifNet is to explicitly couple registration, fusion, and detection in a closed-loop manner, so that improvements in one task can directly constrain and enhance the others. The resulting interactions include a forward coupling from registration to fusion, a semantic feedback path from detection to registration, a task-driven feedback path from detection to fusion, and a structural feedback path from fusion to registration. These couplings are denoted as Reg-to-Fus, Fus2Reg, Det2Reg, Det2Fus, and Fus-to-Det, respectively.

1) Registration-to-Fusion Consistency $( R e g  – t o – F u s ) .$ : The registered infrared image $\hat { Z }$ produced by RegDM serves as the geometrically aligned infrared condition for FusDM. By replacing the raw unaligned infrared input with $\hat { Z } ,$ FusDM operates on a cross-modal pair with substantially reduced spatial discrepancy, which stabilizes the residual diffusion process and improves the structural consistency of the fused output. Accordingly, Reg-to-Fus is implemented by using $\hat { Z }$ as the infrared condition in the FusDM residual diffusion process defined in Sec. III-D. This forward coupling allows FusDM to operate on a geometrically corrected RGB-T pair and stabilizes the generation of the fused representation.

2) Fusion-to-Registration Structural Feedback (Fus2Reg): Beyond serving as the output of FusDM, the fused representation also feeds back to RegDM as a structural regularizer. The motivation is that the fusion result encodes cross-modal structural consistency after complementary information from both modalities has been integrated, and such fused structural cues can be used to further regularize geometric alignment.

We denote this feedback path as $F u s 2 R e g$ . To preserve geometry-sensitive structures, we implement Fus2Reg through a fractional-order derivative consistency between the registered infrared image and the structure extracted from the fused representation:

$$
L _ { \mathrm { F u s 2 R e g } } = \left\| D ^ { \alpha } ( { \hat { Z } } ) - D ^ { \alpha } \big ( \mathrm { C A } ( { \hat { X } } ) \big ) \right\| _ { 1 }\tag{41}
$$

where $D ^ { \alpha } ( \cdot )$ denotes a fractional-order derivative operator with order $\alpha ~ \in ~ ( 0 , 1 ]$ , and $\mathrm { C A } ( \cdot )$ denotes channel-wise averaging that maps the fused representation to a comparable structural map. This constraint encourages the registered infrared image to share consistent structural responses with the fusion-enhanced structure, thereby correcting deformation estimation using fusion-driven geometry cues.

3) Detection-Guided Registration (Det2Reg): Low-level alignment cues alone are insufficient for object-centric RGB-T perception, since registration should also preserve the semantic consistency of target regions across the unaligned and registered infrared domains. To this end, we introduce $D e t 2 R e g$ which uses a frozen infrared-domain detector as a stable semantic anchor during training.

Specifically, the frozen detector is applied to the unaligned infrared image $Z ^ { \prime }$ and the registered infrared image $\hat { Z } ,$ producing

$$
D _ { 1 } = { \mathrm { D e t } } _ { \mathrm { I R } } ( Z ^ { \prime } ) , \qquad D _ { 2 } = { \mathrm { D e t } } _ { \mathrm { I R } } ( { \hat { Z } } )\tag{42}
$$

These two predictions provide object-level semantic supervision for registration, encouraging RegDM to align objectcentric structures more accurately. Unlike using the trainable fusion-side detector, this frozen infrared-domain detector provides a stable semantic reference and avoids introducing optimization drift from the fusion-side detection branch into the registration supervision.

Accordingly, the Det2Reg constraint can be summarized as

$$
L _ { \mathrm { D e t 2 R e g } } = L _ { \mathrm { s e m } } ( D _ { 1 } , D _ { 2 } ; B _ { \mathrm { g t } } )\tag{43}
$$

where $L _ { \mathrm { s e m } } ( \cdot )$ denotes the infrared-domain semantic consistency objective induced by the frozen detector predictions and

the ground-truth bounding-box annotations $B _ { \mathrm { g t } }$ . Specifically, we formulate it as

$$
\begin{array} { r } { L _ { \mathrm { s e m } } ( D _ { 1 } , D _ { 2 } ; B _ { \mathrm { g t } } ) = L _ { \mathrm { d e t } } ( D _ { 2 } , B _ { \mathrm { g t } } ) } \\ { + \lambda _ { \mathrm { s c } } L _ { \mathrm { s c } } ( D _ { 1 } , D _ { 2 } ; B _ { \mathrm { g t } } ) } \end{array}\tag{44}
$$

where $\cal L _ { \mathrm { d e t } } ( D _ { 2 } , B _ { \mathrm { g t } } )$ supervises the registered-infrared detection prediction $D _ { 2 }$ with the ground-truth boxes, and $L _ { \mathrm { s c } } ( D _ { 1 } , D _ { 2 } ; B _ { \mathrm { g t } } )$ encourages object-level semantic responses before and after registration to remain consistent around annotated target regions. In TriDifNet, this term acts as a trainingtime semantic regularizer for RegDM and is not used during inference.

In practice, the semantic consistency term is computed on detector responses matched to the ground-truth boxes:

$$
L _ { \mathrm { s c } } ( D _ { 1 } , D _ { 2 } ; B _ { \mathrm { g t } } ) = \frac { 1 } { | B _ { \mathrm { g t } } | } \sum _ { j = 1 } ^ { | B _ { \mathrm { g t } } | } \left\| \mathbf { s } _ { 2 } ^ { j } - \mathrm { s g } ( \mathbf { s } _ { 1 } ^ { j } ) \right\| _ { 1 }\tag{45}
$$

where $\mathbf { s } _ { 1 } ^ { j }$ and $\mathbf { s } _ { 2 } ^ { j }$ denote the detector semantic responses associated with the j-th ground-truth box from $D _ { 1 }$ and $D _ { 2 }$ respectively, and $\operatorname { s g } ( \cdot )$ denotes the stop-gradient operation.

4) Detection-Aware Fusion (Det2Fus): To explicitly inject object-level semantics into the fusion process, we further design $D e t 2 F u s ,$ , which allows detection to reshape the detectororiented fusion representation before final inference. Unlike simply attaching a detection loss to the fused output, Det2Fus first constructs an ROI-guided prior and uses it to rewrite the detector-side fusion representation. The construction of the ROI masks $M _ { \mathrm { { g t } } }$ $M _ { \mathrm { d 3 } }$ , the mixed prior $M _ { \mathrm { m i x } } .$ , and the task-driven detector representation $F _ { \mathrm { d e t } } ^ { \mathrm { t d } }$ has been introduced in Sec. III-E2. Here, we focus on how this detector-side semantic prior is further fed back to FusDM through task-driven fusion constraints.

As shown in Fig. 3, we introduce a TDFusion-style learnable weighting branch to strengthen the detector-to-fusion feedback. This branch serves as a training-time loss-side guidance module rather than an additional inference-time fusion generator. Specifically, a learnable weighting generator $\mathcal { G } _ { \mathrm { t d } }$ takes the registered infrared image, the optical image, the firstpass detector input, and the mixed ROI prior as inputs:

![](images/72ad5c345af9393f88d16ec14a25a8a83692ad6fa614f5609f1064920ee0613d.jpg)  
Fig. 3. TDFusion-style loss-side guidance branch in Det2Fus. It predicts taskdriven spatial weights for ROI-aware fusion constraints during training.

$$
\begin{array} { r l r } & { } & { \left\{ w _ { \mathrm { i r } } , w _ { \mathrm { v i s } } , w _ { \mathrm { r o i } } , w _ { \mathrm { e d g e } } , w _ { \mathrm { b g } } , w _ { \mathrm { a l i g n } } \right\} = } \\ & { } & { \left. \mathcal { G } _ { \mathrm { t d } } ( \hat { Z } , Y , F _ { \mathrm { d e t } } ^ { \mathrm { t d } } , M _ { \mathrm { m i x } } ) \right. } \end{array}\tag{46}
$$

where the predicted weights are modality-aware and regionaware spatial maps that adaptively control infrared enhancement, visible suppression, ROI emphasis, edge preservation, background inhibition, and alignment relaxation.

To avoid a trivial visible-dominant solution, the TDFusion branch biases the effective weighting toward infrared responses inside ROI regions while suppressing unnecessary visible dominance in object-centric areas. Therefore, it regularizes FusDM through the detector-aware representation and the ROI-guided semantic prior without introducing an additional fused-image generation path during inference.

With the ROI-guided rewriting and the TDFusion-style weighting branch, the Det2Fus objective imposes task-driven constraints directly on the detector-oriented fusion representation:

$$
\begin{array} { r l } { L _ { \mathrm { D e t 2 F u s } } = L _ { \mathrm { i n t } } + \lambda _ { g } L _ { \mathrm { g r a d } } + \lambda _ { r } L _ { \mathrm { r o i - i r } } } & { } \\ { + \lambda _ { b } L _ { \mathrm { b g } } + \lambda _ { a } L _ { \mathrm { a l i g n } } } & { } \end{array}\tag{47}
$$

where all terms are computed on the ROI-rewritten detectorside fusion representation $F _ { \mathrm { d e t } } ^ { \mathrm { t d } } . \ L _ { \mathrm { i n t } }$ enforces weighted intensity consistency with the registered infrared image $\hat { Z }$ and the visible structural proxy $\operatorname { C A } ( Y )$ . $L _ { \mathrm { g r a d } }$ preserves the dominant Sobel-gradient responses from the two modalities. $L _ { \mathrm { r o i - i r } }$ applies an ROI-masked infrared consistency constraint to enhance target saliency. $L _ { \mathrm { b g } }$ suppresses unnecessary detectorchannel residual activation in background regions. $L _ { \mathrm { a l i g n } }$ maintains consistency with the teacher detector-oriented fusion target while relaxing this constraint inside ROI regions.

This design enables DetDM to shape the fusion behavior at the representation level while preserving global fusion fidelity outside object-centric regions. Therefore, Det2Fus is not a simple auxiliary detection loss, but a representationlevel semantic feedback mechanism that reshapes the detectororiented fusion branch before final detection.

5) Fusion-Enhanced Detection(Fus-to-Det): The fused representation also enhances detection in the forward direction. Specifically, FusDM provides the first-pass detector input $F _ { \mathrm { d e t } }$ while the refined representation $F _ { \mathrm { d e t } } ^ { \mathrm { t d } }$ is used for the final detector inference. In this way, the fusion branch not only receives semantic feedback from DetDM, but also produces the task-specialized detector input that drives the final diffusionbased detection process.

6) Overall Training Objective: The overall optimization of TriDifNet is jointly driven by registration, fusion, and detection objectives:

$$
{ \cal L } _ { \mathrm { t o t a l } } = { \cal L } _ { R } + { \cal L } _ { F } + { \cal L } _ { D }\tag{48}
$$

The registration objective is defined as

$$
\begin{array} { r } { L _ { R } = L _ { R } ^ { \mathrm { D i f f } } + \alpha L _ { R } ^ { \mathrm { s m o o t h } } + \beta L _ { R } ^ { \mathrm { a l i g n } } } \\ { + \lambda _ { \mathrm { d e t 2 r e g } } L _ { \mathrm { D e t 2 R e g } } + \lambda _ { \mathrm { f u s 2 r e g } } L _ { \mathrm { F u s 2 R e g } } } \end{array}\tag{49}
$$

where the first three terms are the base registration losses of RegDM, and the last two terms represent semantic feedback from Det2Reg and structural feedback from Fus2Reg.

The fusion objective is defined as

$$
{ \cal L } _ { F } = { \cal L } _ { F } ^ { \mathrm { D i f f } } + \lambda _ { 1 } { \cal L } _ { F } ^ { \mathrm { p i x } } + \lambda _ { 2 } { \cal L } _ { F } ^ { \mathrm { s t r } } + \lambda _ { \mathrm { d e t 2 f u s } } { \cal L } _ { \mathrm { D e t 2 F u s } }\tag{50}
$$

where the first three terms denote the base fusion optimization of FusDM, and the last term introduces the ROI-guided taskdriven semantic feedback from Det2Fus.

In the current implementation, $L _ { \mathrm { D e t 2 F u s } }$ already includes the ROI-guided infrared enhancement and the TDFusion-style learnable weighting mechanism. The corresponding TDFusion branch acts as a training-time loss-side guidance module, which shapes the optimization trajectory of FusDM without introducing an additional fused-image generation path at inference time.

The detection objective $L _ { D }$ is imposed on the final detector output driven by the refined detector representation $F _ { \mathrm { d e t } } ^ { \mathrm { t d } } .$ It follows the box-space diffusion detection loss defined in Sec. III-E, including classification, box regression, and generalized intersection-over-union terms.

Through this formulation, RegDM maintains geometric consistency, FusDM is jointly shaped by image-level supervision and detection-aware semantic feedback, and DetDM is optimized on the detector-oriented fusion representation refined by ROI-guided re-inference. As a result, the final training strategy consistently reflects the current three-task closedloop network: geometry, fusion, and semantics are no longer optimized in isolation, but are mutually constrained within one unified end-to-end network. The above formulation defines the branch-wise optimization targets of TriDifNet. The practical training schedule used to optimize these objectives, including the late-stage Nash-MTL-based refinement, is introduced next.

G. Training Strategy and Nash-MTL-based Late-Stage Refinement

TriDifNet is not trained with a single static schedule from the beginning. Instead, we adopt a stage-wise progressive training strategy to stabilize optimization across registration, fusion, and detection. Specifically, the training process consists of multiple stages. We first optimize the registration branch while freezing the fusion and detection branches, so that RegDM can establish a reliable geometric alignment basis. Next, we optimize the coupled RegDM–FusDM and RegDM– DetDM settings separately by freezing one branch at a time, which allows the fusion and detection branches to adapt to the registration-guided inputs under reduced optimization interference. After that, all three branches are jointly optimized in the full closed-loop setting, so that Det2Reg, Det2Fus, and Fus2Reg can simultaneously participate in the end-to-end interaction.

After the above joint training stage, we further introduce a late-stage refinement step to improve the optimization balance among the three task objectives. Unlike a fixed loss summation, Nash multi-task learning (Nash-MTL) is used as a dynamic task-coordination strategy to adaptively balance registration, fusion, and detection in the final training stage. Let

$$
\mathcal { L } _ { \mathrm { a c t } } = \{ L _ { R } , L _ { F } , L _ { D } \}\tag{51}
$$

denote the set of active trainable objectives in this refinement stage. We compute their gradients with respect to a shared intermediate representation s and form the gradient matrix

$$
\mathbf { G } = \left[ \nabla _ { \mathbf { s } } L _ { R } , \nabla _ { \mathbf { s } } L _ { F } , \nabla _ { \mathbf { s } } L _ { D } \right] ^ { \top }\tag{52}
$$

Following Nash-MTL, the dynamic task weights $\alpha \quad =$ $[ \alpha _ { R } , \alpha _ { F } , \alpha _ { D } ] ^ { \top }$ are obtained by solving

$$
\mathbf { G } \mathbf { G } ^ { \top } \alpha = \mathbf { 1 } / \alpha\tag{53}
$$

where $\mathbf { 1 } / \alpha$ denotes the element-wise reciprocal. The resulting weighted objective for the late-stage refinement is

$$
{ \cal L } _ { \mathrm { N a s h } } = \alpha _ { R } { \cal L } _ { R } + \alpha _ { F } { \cal L } _ { F } + \alpha _ { D } { \cal L } _ { D }\tag{54}
$$

This design is motivated by the observation that the final closed-loop optimization contains heterogeneous objectives from geometric alignment, cross-modal fusion, and task-level detection. Directly summing these losses with fixed weights may cause one task to dominate the optimization and suppress the others. Nash-MTL provides a dynamic balancing mechanism that coordinates the gradients of the three branches and encourages a more stable multi-task update. If a training batch does not contain valid detection supervision, the detector branch is excluded from the bargaining step, and the optimization falls back to the available active objective(s) rather than enforcing an invalid three-task update.

It should be emphasized that Nash-MTL is used only in this training-time late-stage refinement step. It improves the optimization balance among RegDM, FusDM, and DetDM, but does not introduce additional inference-time modules or change the structural closed-loop coupling of TriDifNet.

## H. Inference Pipeline

During inference, TriDifNet performs end-to-end RGB-T perception through a sequential forward pass over RegDM, FusDM, and DetDM, while preserving the detector-side ROIguided re-inference mechanism. Given an optical image $Y$ and an unaligned infrared image $Z ^ { \prime }$ , RegDM first estimates the inverse deformation field and warps the infrared image into the optical reference coordinate system:

$$
\hat { \phi } _ { 0 } ^ { - 1 } = \mathrm { R e g D M } ( Y , Z ^ { \prime } ) , \qquad \hat { Z } = { \mathcal W } ( Z ^ { \prime } , \hat { \phi } _ { 0 } ^ { - 1 } )\tag{55}
$$

The registered infrared image $\hat { Z }$ is then fed into FusDM together with the optical image $Y$ to produce the fused representation:

$$
\hat { X } _ { \mathrm { r e s } } = \mathrm { F u s D M } ( Y , \hat { Z } ) , \qquad \hat { X } = Y + \hat { X } _ { \mathrm { r e s } }\tag{56}
$$

Next, the visible–fusion cross-attention module converts the raw detector-side fusion channel into the cross-attentionenhanced detector-oriented representation $F _ { \mathrm { d e t } }$ , which serves as the first-pass detector input.

$$
F _ { \mathrm { d e t } } = { \mathcal { P } } ( E _ { \mathrm { v i s } } + A _ { \mathrm { v f } } V )\tag{57}
$$

Based on $F _ { \mathrm { d e t } }$ , DetDM performs an initial diffusion-based detection and produces the first-pass prediction

$$
D _ { 3 } = \mathrm { D e t D M } ( F _ { \mathrm { d e t } } )\tag{58}
$$

The first-pass prediction is then converted into a predictionderived ROI prior

$$
M _ { d 3 } = \mathcal { R } ( D _ { 3 } )\tag{59}
$$

where $\mathcal { R } ( \cdot )$ denotes the ROI mask generation operator from the detector prediction. Using this prediction-derived ROI prior, the detector-oriented representation is rewritten into a task-driven refined detector input:

$$
F _ { \mathrm { d e t } } ^ { \mathrm { t d } } = \mathcal { T } ( F _ { \mathrm { d e t } } , \hat { Z } , M _ { d 3 } )\tag{60}
$$

where $\tau ( \cdot )$ denotes the ROI-guided rewriting operator that enhances infrared saliency in target-centered regions while preserving the original fusion-driven detector response in nontarget regions. The final pedestrian detection is then obtained through ROI-guided re-inference:

$$
\hat { D } _ { 3 } = \mathrm { D e t D M } ( F _ { \mathrm { d e t } } ^ { \mathrm { t d } } )\tag{61}
$$

It is worth noting that the training-only semantic and structural supervision terms are not used during inference. In particular, the frozen infrared-domain YOLO detector employed by Det2Reg serves only as a semantic anchor during training and is removed at test time. Similarly, the loss terms associated with Det2Reg, Det2Fus, and Fus2Reg are used only for optimization and are not evaluated during inference. However, the detector-side ROI-guided refinement remains part of the forward inference pipeline, since it directly constructs the task-driven detector representation $F _ { \mathrm { d e t } } ^ { \mathrm { t d } }$ from the first-pass prediction prior $M _ { d 3 }$ for the second-pass detection.

Therefore, the inference stage of TriDifNet outputs three perceptually and semantically meaningful results: the registered infrared image ${ \hat { Z } } ,$ the fused representation ${ \hat { X } } .$ , and the final pedestrian detection result ${ \hat { D } } _ { 3 }$ . This inference pipeline consistently reflects the design principle of TriDifNet: geometry is first corrected by RegDM, cross-modal information is integrated by FusDM, and final task-level semantics are produced by DetDM through prediction-guided detection refinement.

## IV. EXPERIMENTS

## A. Datasets, Misalignment Setting, and Evaluation Metrics

a) Datasets: All experiments are conducted on the KAIST Multispectral Pedestrian Benchmark [30], which provides paired visible–thermal images for RGB–T pedestrian perception. Following a cleaned annotation setting, heavily occluded, severely truncated, extremely small, and ambiguous pedestrian annotations are removed or ignored. The remaining valid pedestrian-related annotations are merged into a single pedestrian class for training and evaluation.

TABLE I  
COMPARISON SETTINGS FOR DIFFERENT EVALUATION SETTINGS
<table><tr><td>Evaluation Setting</td><td>Input</td><td> $\overline { { \mathrm { \mathbf { O u t p u t } } } }$ </td><td>Primary Goal</td></tr><tr><td>Registration comparison</td><td>Misaligned RGB-T</td><td>Registered IR</td><td>Image-level registration accuracy</td></tr><tr><td>Fusion comparison</td><td>Aligned RGB-T</td><td>Fusion result</td><td>Image-level fusion effectiveness</td></tr><tr><td>End-to-end comparison</td><td>Misaligned RGB-T</td><td>Final detection</td><td>Full-system perception performance</td></tr></table>

TABLE II  
BASELINES USED IN THE EXPERIMENTS
<table><tr><td>Category</td><td>ID</td><td>Description</td><td>Selected</td></tr><tr><td>Registration</td><td></td><td>No-registration baseline</td><td>NoReg</td></tr><tr><td>Registration</td><td></td><td>Template-matching-based visible-thermal registration baseline</td><td>TWMM (2022) [31]</td></tr><tr><td>Registration</td><td>R3</td><td>Plug-and-play general multi-modal image registration baseline</td><td>PGMR (2025) [32]</td></tr><tr><td>Fusion</td><td>F1</td><td>Detection-driven fusion baseline</td><td>DetFusion (2022) [10]</td></tr><tr><td>Fusion</td><td>F2</td><td>Meta-feature embedding guided infrared-visible fusion baseline</td><td>MetaFusion (2023) [11]</td></tr><tr><td>Fusion</td><td>F3</td><td>Registration-fusion integrated infrared-visible fusion baseline</td><td>MulFS-CAP (2025) [3]</td></tr><tr><td>System</td><td>S1</td><td>Visible-only detection baseline with an external detector</td><td>YOLO11s</td></tr><tr><td>System</td><td>S2 S3</td><td>Infrared-only detection baseline with an external detector</td><td>YOLO11s</td></tr><tr><td>System</td><td></td><td>External registration + registered-IR detection pipeline</td><td>PGMR (2025) [32] + YOLO11s</td></tr><tr><td>System</td><td></td><td>External fusion + detection pipeline</td><td>LRRNet (2023) [33] + YOLO11s</td></tr><tr><td>System</td><td>S4 S5</td><td>Fully external sequential Reg→Fus→Det pipeline</td><td>PGMR (2025) [32] + LRRNet (2023) [33] + YOLO11s</td></tr><tr><td>System</td><td>S6</td><td>External registration preprocessing + fusion-detection integrated method</td><td>PGMR (2025) [32] + MetaFusion (2023) [11]</td></tr><tr><td>System</td><td>S7</td><td>Registration-fusion integrated method + external detector</td><td> $\mathrm { M u l F S \mathrm { - } C A P \ ( 2 0 2 5 ) \ [ 3 ] + Y O L O 1 1 s }$ </td></tr></table>

b) Patch-Based Data Preparation: Considering the memory cost of joint registration, fusion, and detection training, we adopt patch-based processing instead of full-frame training. Each visible–thermal pair is cropped into $2 5 6 \times 2 5 6$ patches with a stride of 128, and at most 50 ground-truth boxes are retained for each patch. The processed training patches are further split into training and validation subsets with a ratio of 0.8. For background-only patches with empty samples, only 20% are retained independently within the training/validation and testing splits to avoid overwhelming the optimization. For each retained patch, both reference-coordinate boxes and perturbed infrared-domain boxes are stored for branch-specific detection evaluation.

c) Teacher Fusion Target: Since KAIST does not provide real fusion ground truth, we use the fusion results produced by LRRNet [33] on the original KAIST data as a pseudo teacher target for training FusDM. The teacher output is used only to stabilize residual fusion learning and is not treated as exact fusion ground truth for final evaluation.

d) Misalignment Setting: To simulate weakly misaligned RGB-T perception, synthetic affine perturbations are applied to the infrared modality while the visible image is kept as the reference. In the current setting, the infrared patch is perturbed by an in-plane rotation up to $2 ^ { \circ }$ and a translation ratio of 0.05, corresponding to a maximum translation of approximately 12.8 pixels along each axis for a $2 5 6 \times 2 5 6$ patch. This controlled setting focuses on weak affine misalignment rather than severe non-rigid deformation, and is used to evaluate whether RegDM can provide reliable geometric correction for the target RGB-T perception scenario.

e) Evaluation Metrics: We evaluate TriDifNet from three perspectives. For registration, we report root mean squared error (RMSE), normalized cross-correlation / correlation coefficient (NCC/CC), and structural similarity (SSIM) [34] between the registered infrared image and the aligned infrared reference. For fusion, we use weak-reference image-level measures, including CC and UIQI [35], to evaluate structural consistency and cross-modal information preservation. For detection, we report Precision, Recall, ${ \mathrm { m A P } } _ { 5 0 } ,$ , and $\mathrm { m A P _ { 5 0 : 9 5 } }$ for the visible branch, the infrared or registered-infrared branch, and the fusion branch. Since all valid pedestrianrelated annotations are merged into a single pedestrian class, m ${ \bf A P } _ { 5 0 : 9 5 }$ is computed in a single-class COCO-style [36] manner over IoU thresholds from 0.50 to 0.95 with a step size of 0.05. The annotation domain used for each detection branch follows the comparison setting summarized in Table I. The final fusion-side detection result serves as the primary evaluation of the complete network.

## B. Experimental Settings and Baselines

a) Training and Inference Settings: TriDifNet is implemented in PyTorch and trained on a single NVIDIA RTX 5060 Ti GPU with 16 GB memory. All experiments are conducted with 256 × 256 input patches. The model is trained under the stage-wise three-task optimization strategy introduced in Sec. III-G, with a batch size of 1, 30,000 training iterations, and 500 diffusion timesteps. RegDM, FusDM, and DetDM are optimized within the unified closed-loop training pipeline using branch-specific optimizers, with learning rates of $2 \times 1 0 ^ { - 4 }$ $1 \times 1 0 ^ { - 4 }$ , and $2 . 5 \times 1 0 ^ { - 5 }$ , respectively.

At test time, the training-only auxiliary components are removed, and the deployed inference pipeline consists of RegDM, FusDM, and DetDM with ROI-guided re-inference. For diffusion inference, RegDM and FusDM use DDIM sampling [37] with 25 steps. For ROI-guided re-inference, the firstpass D3 confidence threshold is set to 0.10 to improve recall for ROI construction, whereas the final second-pass detection uses a confidence threshold of 0.30 for reporting Precision and Recall. No ground-truth boxes are used for ROI construction during inference. All FPS results are measured on the same platform with batch size 1, excluding data loading and file I/O.

b) Baseline Categories: To avoid ambiguity, the compared methods are grouped into registration, fusion, and system-level baselines, as summarized in Table II. Registration baselines evaluate geometric correction under the same misaligned RGB-T input. Fusion baselines evaluate image-level fusion behavior under aligned-input conditions. System-level baselines evaluate the final pedestrian detection performance under complete RGB-T perception pipelines. Since LRR-Net [33] is used as the pseudo teacher for FusDM training, it is not included as a direct image-level fusion baseline, and its downstream influence is evaluated through S4 and S5. For detector-based system baselines, most external pipelines use YOLO11s as the downstream detector, while S6 follows the original MetaFusion evaluation setup and uses YOLOv5s [11].

c) Comparison Settings: Different experiments follow different input settings. Registration comparisons start from the same misaligned RGB-T input and evaluate only the registered infrared output. Fusion comparisons are conducted under a unified aligned-input condition to isolate fusion behavior from registration errors. End-to-end comparisons start from the same misaligned RGB-T input and evaluate the final detection results with branch-specific annotations. Specifically, visible, registered-infrared, and fusion-side outputs are evaluated with reference-coordinate boxes, while unaligned-infrared detection is evaluated with perturbed infrared-domain boxes.

d) Training Settings for Baselines: For fair comparison, all baseline methods are trained and evaluated under the same overall data setting used by TriDifNet. For methods with official implementations, we follow their released configurations whenever applicable, initialize them from the official pretrained weights, and fine-tune them on the current training data rather than training from scratch. For module-level baselines, registration methods are fine-tuned on the misaligned RGB-T training set, while fusion methods are fine-tuned on the aligned RGB-T training set according to Table I. Each singletask baseline is fine-tuned for 50 epochs and evaluated on the corresponding unified test set.

For system-level baselines with sequential components, each downstream component is fine-tuned on the training-set outputs produced by its upstream component for 50 epochs before being evaluated on the corresponding test-set outputs. For the MulFS-CAP-based [3] baselines F3 and S7, we follow the official implementation and fine-tune it for 200 epochs using 500 randomly sampled training pairs and 100 validation pairs. All baselines, including MulFS-CAP-based pipelines, are evaluated on the same unified test set.

## C. Main Quantitative Results

a) Registration Comparison: We first compare RegDM with the registration baselines under the same misalignment setting. All methods take the same misaligned RGB-T pair as input and output a registered infrared image. Since the current benchmark focuses on controlled weak affine perturbations, this comparison is intended to evaluate whether RegDM can provide reliable geometric correction under the target weakly misaligned RGB-T setting, rather than to claim general superiority under severe non-rigid deformation scenarios.

TABLE III  
QUANTITATIVE COMPARISON OF REGISTRATION RESULTS
<table><tr><td>Method</td><td>RMSE↓</td><td>NCC/CC↑</td><td>SSIM↑</td></tr><tr><td>R1</td><td>10.976204</td><td>0.831235</td><td>0.854959</td></tr><tr><td>R2</td><td>45.652067</td><td>0.591024</td><td>0.543002</td></tr><tr><td>R3</td><td>10.457858</td><td>0.835497</td><td>0.853929</td></tr><tr><td>Proposed RegDM</td><td>10.425901</td><td>0.832831</td><td>0.856359</td></tr></table>

![](images/c0f3483cb92fdcf3866262b85fb06a557e16dbe9d3ddb37990c703b6f3f47456.jpg)  
Fig. 4. Qualitative comparison of registration results corresponding to Table III, including absolute-error heatmaps and checkerboard alignment visualizations.

As shown in Table III, the proposed RegDM achieves the lowest RMSE (10.425901) and the highest SSIM (0.856359) among the compared registration methods, while maintaining competitive NCC/CC. R3 obtains a slightly higher NCC/CC (0.835497), indicating that the image-level registration scores of different methods can be close under the controlled weakmisalignment setting. These results suggest that RegDM provides stable geometric correction for the subsequent fusion and detection stages, rather than an overwhelming single-task registration advantage over all baselines. The qualitative results in Fig. 4 further show that RegDM produces low-residual responses and coherent checkerboard alignment patterns. The downstream contribution of RegDM is further evaluated in the end-to-end detection comparison and the system-level ablations, where RegDM is coupled with Det2Reg and Fus2Reg.

b) Fusion Comparison: We next compare FusDM with the fusion baselines under a unified aligned-input condition. This setting isolates fusion behavior from registration errors and follows the corresponding setting in Table I. Since KAIST does not provide real fusion ground truth and the LRRNet output is used only as a pseudo teacher during training, the following comparison reports weak-reference image-level indicators rather than treating teacher-referenced scores as the primary evidence.

Table IV reports weak-reference image-level fusion scores under the aligned-input setting. F3 and F2 achieve the best CC and UIQI, respectively, while FusDM obtains competitive but not leading teacher-referenced scores. Since the KAIST benchmark lacks real fusion ground truth, these metrics are used only to characterize image-level fusion behavior rather than downstream perception utility. Fig. 5 shows that FusDM preserves visible structural cues and enhances target-related infrared responses. More detailed evidence for its task-level effectiveness is provided by the end-to-end detection comparison in Sec. IV-C and the cross-task ablations in Sec. IV-D.

TABLE IV  
QUANTITATIVE COMPARISON OF FUSION RESULTS
<table><tr><td>Method</td><td>CC↑</td><td>UIQI↑</td></tr><tr><td>F1</td><td>0.900771</td><td>0.808711</td></tr><tr><td>F2</td><td>0.903327</td><td>0.864656</td></tr><tr><td>F3</td><td>0.942980</td><td>0.862271</td></tr><tr><td>Proposed FusDM</td><td>0.926469</td><td>0.840299</td></tr></table>

![](images/332f52ded7b0b7c531169f5851c5e0ef31ce0cf6a0f67fc1d57f9270f64d96b9.jpg)  
Fig. 5. Qualitative comparison of fusion results corresponding to Table IV. TriDifNet emphasizes target-aware infrared saliency while maintaining visible structural cues for downstream detection.

c) End-to-End Detection Comparison: The most important comparison is the end-to-end pedestrian detection performance. All system-level baselines start from the same misaligned RGB-T input and are evaluated under the same detection benchmark. This comparison directly verifies whether the proposed closed-loop Reg–Fus–Det optimization improves downstream perception compared with single-modality settings, sequential external pipelines, and partially coupled alternatives.

TABLE V  
END-TO-END PEDESTRIAN DETECTION COMPARISON
<table><tr><td>Method</td><td>P(%)↑</td><td>R(%)↑</td><td> $\overline { { \mathrm { m A P } _ { 5 0 } ( \% ) \uparrow } }$ </td><td>mAP50:95(%)↑</td></tr><tr><td>S1</td><td>81.53</td><td>39.49</td><td>36.02</td><td>16.45</td></tr><tr><td>S2</td><td>79.39</td><td>44.80</td><td>40.33</td><td>20.88</td></tr><tr><td>S3</td><td>81.14</td><td>9.28</td><td>8.32</td><td>3.12</td></tr><tr><td>S4</td><td>80.22</td><td>25.22</td><td>22.90</td><td>10.15</td></tr><tr><td>S5</td><td>69.53</td><td>37.71</td><td>32.65</td><td>14.66</td></tr><tr><td>S6</td><td>41.62</td><td>52.84</td><td>38.51</td><td>18.76</td></tr><tr><td>S7</td><td>72.42</td><td>48.45</td><td>43.66</td><td>19.02</td></tr><tr><td>Proposed TriDifNet</td><td>73.85</td><td>65.44</td><td>57.55</td><td>26.83</td></tr></table>

Table V reports the system-level detection results. TriDifNet obtains the best scores on the three main detection metrics, i.e., 65.44% Recall, 57.55% mAP<sub>50</sub>, and 26.83% mAP<sub>50:95</sub>. Compared with the strongest external system-level baseline S7, TriDifNet improves these three metrics by 16.99%, 13.89%, and 7.81%, respectively. Although the visible-only baseline S1 obtains the highest precision, its recall and mAP remain limited, indicating that single-modality detection is insufficient under misaligned RGB-T inputs.

S3 shows a sharp degradation under the post-registration detection setting, with only 9.28% recall, 8.32% $\mathrm { \ m A P _ { 5 0 } } .$ , and $3 . 1 2 \% \mathrm { m A P } _ { 5 0 : 9 5 } .$ . Although PGMR obtains competitive imagelevel registration scores in Table III, its registered output may still contain object-level residual offsets with respect to the reference-domain detection annotations. Such offsets can be amplified into localization errors and missed detections, leading to a clear decrease in recall and mAP. This result shows that good single-task registration metrics do not necessarily guarantee detection-consistent alignment in a decoupled $\mathrm { R e g \mathrm {  D e t } }$ pipeline.

VIS  
S1  
Unaligned IR  
S2  
S4  
S5  
S3  
S6  
S7  
TriDifNet  
![](images/667672985258e5c5d5c141c578877597eac4af3e6fe133fc8815e01df0078e28.jpg)  
Fig. 6. Qualitative comparison of end-to-end pedestrian detection results corresponding to Table V. Case 1 shows daytime large targets; Cases 2 and 3 show daytime small targets; Case 4 shows nighttime small targets under low-light conditions; and Case 5 shows nighttime small targets with relatively stronger illumination. Yellow and green boxes denote ground truth and predictions, respectively.

TABLE VI  
SYSTEM-LEVEL ABLATION STUDY
<table><tr><td>Method</td><td colspan="4">P(%)↑ R(%)↑</td></tr><tr><td>RegDM FusDM</td><td></td><td></td><td> $\mathrm { m A P _ { 5 0 } ( \% ) } \uparrow$ </td><td> $\mathrm { m A P _ { 5 0 : 9 5 } } ( \mathcal { I } _ { O } ) \uparrow$ </td><td>FPS↑</td></tr><tr><td>一 一</td><td>48.77</td><td>49.57</td><td>37.90</td><td>13.71</td><td>11.02</td></tr><tr><td>√ 一</td><td>70.81</td><td>57.34</td><td>44.67</td><td>19.13</td><td>0.32</td></tr><tr><td>√</td><td>一 51.03</td><td>49.46</td><td>39.04</td><td>14.07</td><td>1.03</td></tr><tr><td>√</td><td>√</td><td>73.85 65.44</td><td>57.55</td><td>26.83</td><td>0.25</td></tr></table>

S4 also yields limited detection performance, with 25.22% recall and 22.90% $\mathrm { \ m A P _ { 5 0 } }$ . Since LRRNet is applied without explicit registration in this setting, the fused image can suffer from misalignment-induced ghosting, double edges, and target deformation. These artifacts degrade the detector input and further propagate to the final detection metrics. S5 performs the fully external $\mathrm { R e g {  } F u s {  } D e t }$ pipeline, but the sequential combination still suffers from the accumulation of registration and fusion errors. For the MetaFusion-based setting S6, we follow its original evaluation setup with YOLOv5s, so part of the performance difference may also come from the different detector configuration.

Overall, the large gains in recall and mAP indicate that TriDifNet reduces missed detections and achieves a better detection trade-off under misaligned RGB-T inputs. These results support the effectiveness of closed-loop optimization among registration, fusion, and detection over sequential or partially coupled pipelines.

Fig. 6 further provides qualitative evidence for the endto-end comparison. Compared with the system-level baselines, TriDifNet produces more complete and better localized pedestrian predictions under misaligned RGB-T inputs. This advantage is most evident in Cases 4 and 5, where several baselines suffer from missed detections under weak visible cues and low-contrast infrared responses, whereas TriDifNet still retains stable target predictions. The qualitative comparison is consistent with the quantitative gains reported in Table V.

## D. Ablation Studies

a) System-Level Ablation: We conduct system-level ablations to evaluate the contribution of RegDM and FusDM to RGB-T pedestrian perception. DetDM is retained in all variants, and each variant is trained under the same training schedule before testing. When both RegDM and FusDM are removed, detection is performed on the unaligned infrared input; when FusDM is removed, detection is performed on the registered infrared image; otherwise, detection is performed on the fusion-oriented representation.

As shown in Table VI, removing RegDM and FusDM gives limited detection performance, with 37.90% $\mathrm { \ m A P _ { 5 0 } }$ and $1 3 . 7 1 \% \mathrm { \ m A P { _ { 5 0 : 9 5 } } }$ . Using FusDM alone improves $\mathrm { \ m A P _ { 5 0 } }$ from 37.90% to 44.67% and $\mathrm { m A P _ { 5 0 : 9 5 } }$ from 13.71% to 19.13%, indicating that cross-modal fusion provides useful perception cues even under misalignment. Using RegDM alone produces a smaller improvement, increasing $\mathrm { \ m A P _ { 5 0 } }$ from 37.90% to 39.04% and $\mathrm { m A P _ { 5 0 : 9 5 } }$ from 13.71% to 14.07%. This suggests that registration alone is insufficient for the final perception task when cross-modal fusion is absent. The full TriDifNet achieves the best detection results, reaching 65.44% recall, $5 7 . 5 5 \% \mathrm { m A P _ { 5 0 } }$ , and $2 6 . 8 3 \% \mathrm { m A P } _ { 5 0 : 9 5 }$ . The large performance gap between the full TriDifNet and the single-module variants indicates that RegDM and FusDM are more effective when jointly optimized within the full TriDifNet model.

Table VI also reports the inference efficiency of different variants. The lower FPS of the full TriDifNet is mainly associated with the additional iterative computation introduced by diffusion-based registration, diffusion-based fusion, and ROI-guided re-inference. This efficiency cost is a common limitation of diffusion-based registration–fusion methods [2]. In this work, the main focus is to evaluate three-task closedloop optimization under misaligned RGB-T inputs. Improving the real-time efficiency of the diffusion-based inference pipeline will be investigated in future work.

b) Cross-Task and Internal Mechanism Ablation: We further evaluate the cross-task feedback paths and the key internal mechanisms in Table VII. Compared with the full TriDifNet, removing Det2Reg degrades both registration and detection, increasing RMSE from 10.4259 to 10.8720 and reducing $\mathrm { \ m A P _ { 5 0 } }$ from 57.55% to 54.11%. This indicates that the frozen infrared-domain detector provides useful object-level semantic guidance for registration. Removing Fus2Reg causes the largest registration degradation among the feedback-path variants, with RMSE increasing to 11.0282 and SSIM decreasing to 0.8426, suggesting that fusion-to-registration structural feedback contributes to geometric consistency.

The detection-aware fusion pathway has the most direct influence on final perception. Without Det2Fus, $\mathrm { \ m A P _ { 5 0 } }$ and $\mathrm { m A P _ { 5 0 : 9 5 } }$ decrease from 57.55% and 26.83% to 42.03% and 18.70%, respectively. When ROI-IR Injection is further disabled, the performance further drops to 36.60% and 14.54%.

TABLE VII  
ABLATION STUDY ON CROSS-TASK FEEDBACK
<table><tr><td>Index</td><td>Det2Reg</td><td>Fus2Reg</td><td>Det2Fus</td><td>ROI-IR Injection</td><td>Nash-MTL</td><td>RMSE↓</td><td>Registration NCC/CC↑</td><td>SSIM↑</td><td>CC↑</td><td>Fusion UIQI↑</td><td>P(%)↑</td><td>R(%)↑</td><td>Detection  $\mathrm { m A P } _ { 5 0 } ( \% ) \uparrow$ </td><td> $\mathrm { m A P _ { 5 0 : 9 5 } ( \% ) } \uparrow$ </td></tr><tr><td>1</td><td>一</td><td>√</td><td>√</td><td>√</td><td>√</td><td>10.8720</td><td>0.8215</td><td>0.8461</td><td>0.9258</td><td>0.8447</td><td>71.30</td><td>62.77</td><td>54.11</td><td>24.73</td></tr><tr><td>2</td><td>√</td><td>一</td><td>√</td><td>V</td><td>√</td><td>11.0282</td><td>0.8176</td><td>0.8426</td><td>0.9263</td><td>0.8427</td><td>73.80</td><td>61.48</td><td>55.59</td><td>26.75</td></tr><tr><td>3</td><td>√</td><td>V</td><td>一</td><td>√</td><td>√</td><td>10.9231</td><td>0.8188</td><td>0.8448</td><td>0.9665</td><td>0.9492</td><td>55.17</td><td>52.17</td><td>42.03</td><td>18.70</td></tr><tr><td>4</td><td>√</td><td>√</td><td>一</td><td>一</td><td>√</td><td>11.0814</td><td>0.8181</td><td>0.8424</td><td>0.9640</td><td>0.9406</td><td>54.92</td><td>47.01</td><td>36.60</td><td>14.54</td></tr><tr><td>5</td><td>√</td><td>√</td><td>√</td><td>√</td><td>一</td><td>10.9469</td><td>0.8196</td><td>0.8443</td><td>0.9312</td><td>0.8508</td><td>71.56</td><td>63.59</td><td>55.32</td><td>25.52</td></tr><tr><td>6</td><td>V</td><td>√</td><td>V</td><td>√</td><td>√</td><td>10.4259</td><td>0.8328</td><td>0.8563</td><td>0.9264</td><td>0.8402</td><td>73.85</td><td>65.44</td><td>57.55</td><td>26.83</td></tr></table>

The variants without Det2Fus or ROI-IR Injection obtain higher fusion scores in terms of CC and UIQI, but these higher image-level scores are accompanied by a clear decline in detection performance. This trend suggests that removing detection-oriented feedback shifts the fusion branch toward weak-reference image-level similarity, rather than target-aware representation learning. Therefore, weak-reference fusion metrics do not necessarily reflect downstream task utility in the misaligned RGB-T detection setting. The performance drops caused by removing Det2Fus and ROI-IR Injection support the importance of fusion–detection coupling and ROI-level infrared injection for constructing a task-discriminative fusion representation.

Removing Nash-MTL also reduces $\mathrm { \ m A P _ { 5 0 } }$ from 57.55% to 55.32%, indicating the benefit of late-stage three-branch optimization balancing among registration, fusion, and detection. Overall, the full TriDifNet achieves the best registration and detection results, supporting the complementary effects of the proposed closed-loop feedback paths and internal refinement mechanisms.

## V. CONCLUSION

In this paper, we propose TriDifNet, a unified diffusionbased network for misaligned RGB-T pedestrian perception. By optimizing registration, fusion, and detection in a closed loop, TriDifNet reduces sequential error propagation and enables geometric alignment, cross-modal fusion, and task-level semantics to reinforce each other. Experiments on the KAIST multispectral pedestrian benchmark show that TriDifNet provides reliable registration and task-oriented fusion, while achieving clear improvements in end-to-end pedestrian detection over single-modality, sequential, and partially coupled baselines. Ablation studies further support the effectiveness of the cross-task feedback paths, ROI-guided infrared injection, and late-stage multi-task balancing. Future work will focus on improving inference efficiency and extending the network to broader RGB-T perception scenarios.

## REFERENCES

[1] H. Xu, J. Ma, J. Yuan, Z. Le, and W. Liu, “Rfnet: Unsupervised network for mutually reinforcing multi-modal image registration and fusion,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 19 647–19 656.

[2] W. Diao, L. Hu, K. Zhang, and L. Xiao, “Dual-conditionally guided diffusion models for fusion of unregistered multisource remote sensing images,” IEEE Transactions on Geoscience and Remote Sensing, vol. 63, pp. 1–17, 2025.

[3] H. Li, Z. Yang, Y. Zhang, W. Jia, Z. Yu, and Y. Liu, “Mulfs-cap: Multimodal fusion-supervised cross-modality alignment perception for unregistered infrared-visible image fusion,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 5, pp. 3673–3690, 2025.

[4] D. Wang, J. Liu, L. Ma, R. Liu, and X. Fan, “Improving misaligned multi-modality image fusion with one-stage progressive dense registration,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 34, no. 11, pp. 10 944–10 958, 2024.

[5] J. Li, B. Gao, W. L. Woo, J. Xu, L. Liu, and Y. Zeng, “A novel multispectral fusion defect detection framework with coarse-to-fine multispectral registration,” IEEE Transactions on Instrumentation and Measurement, vol. 73, pp. 1–13, 2024.

[6] D. Wang, J. Liu, X. Fan, and R. Liu, “Unsupervised misaligned infrared and visible image fusion via cross-modality image generation and registration,” 2022. [Online]. Available: https://arxiv.org/abs/2205.11876

[7] Z. Huang, J. Liu, X. Fan, R. Liu, W. Zhong, and Z. Luo, “Reconet: Recurrent correction network for fast and efficient multi-modality image fusion,” in Computer Vision – ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XVIII. Berlin, Heidelberg: Springer-Verlag, 2022, pp. 539–555.

[8] K. Niu, C. Wang, J. Xu, J. Liang, X. Zhou, K. Wen, M. Lu, and C. Yang, “Early forest fire detection with uav image fusion: A novel deep learning method using visible and infrared sensors,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 18, pp. 6617–6629, 2025.

[9] L. Tang, J. Yuan, and J. Ma, “Image fusion in the loop of high-level vision tasks: A semantic-aware real-time infrared and visible image fusion network,” Information Fusion, vol. 82, pp. 28–42, 2022. [Online]. Available: https://www.sciencedirect.com/science/article/pii/ S1566253521002542

[10] Y. Sun, B. Cao, P. Zhu, and Q. Hu, “Detfusion: A detection-driven infrared and visible image fusion network,” in Proceedings of the 30th ACM International Conference on Multimedia, ser. MM ’22. New York, NY, USA: Association for Computing Machinery, 2022, p. 4003–4011. [Online]. Available: https://doi.org/10.1145/3503161.3547902

[11] W. Zhao, S. Xie, F. Zhao, Y. He, and H. Lu, “Metafusion: Infrared and visible image fusion via meta-feature embedding from object detection,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 13 955–13 965.

[12] L. Tang, H. Zhang, H. Xu, and J. Ma, “Rethinking the necessity of image fusion in high-level vision tasks: A practical infrared and visible image fusion network based on progressive semantic injection and scene fidelity,” Information Fusion, vol. 99, p. 101870, 2023. [Online]. Available: https://www.sciencedirect.com/science/article/pii/ S1566253523001860

[13] Y. Zeng, T. Liang, Y. Jin, and Y. Li, “Mmi-det: Exploring multi-modal integration for visible and infrared object detection,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 34, no. 11, pp. 11 198–11 213, 2024.

[14] K. Hu, Y. He, Y. Li, J. Zhao, S. Chen, and Y. Kang, “Ei²det: Edge-guided illumination-aware interactive learning for visible-infrared object detection,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 35, no. 7, pp. 7101–7115, 2025.

[15] H. Fu, J. Yuan, G. Zhong, X. He, J. Lin, and Z. Li, “Cfdeformable detr: An end-to-end alignment-free model for weakly aligned visible-infrared object detection,” in Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI-24, K. Larson, Ed. International Joint Conferences on Artificial Intelligence Organization, 8 2024, pp. 758–766, main Track. [Online]. Available: https://doi.org/10.24963/ijcai.2024/84

[16] M. Yuan, Y. Wang, and X. Wei, “Translation, scale and rotation: Cross-modal alignment meets rgb-infrared vehicle detection,” 2022. [Online]. Available: https://arxiv.org/abs/2209.13801

[17] Y. Cao, X. Luo, J. Yang, Y. Cao, and M. Y. Yang, “Locality guided cross-modal feature aggregation and pixel-level fusion for multispectral pedestrian detection,” Information Fusion, vol. 88, pp. 1–11, 2022. [Online]. Available: https://www.sciencedirect.com/science/ article/pii/S1566253522000549

[18] N. Chen, J. Xie, J. Nie, J. Cao, Z. Shao, and Y. Pang, “Attentive alignment network for multispectral pedestrian detection,” in Proceedings of the 31st ACM International Conference on Multimedia, ser. MM ’23. New York, NY, USA: Association for Computing Machinery, 2023, p. 3787–3795. [Online]. Available: https://doi.org/10.1145/3581783.3613444

[19] Y. Sun, B. Cao, P. Zhu, and Q. Hu, “Drone-based rgb-infrared crossmodality vehicle detection via uncertainty-aware learning,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 32, no. 10, pp. 6700–6713, 2022.

[20] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 10 674–10 685.

[21] C. Saharia, J. Ho, W. Chan, T. Salimans, D. J. Fleet, and M. Norouzi, “Image super-resolution via iterative refinement,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 4, pp. 4713– 4726, 2023.

[22] C. Saharia, W. Chan, H. Chang, C. A. Lee, J. Ho, T. Salimans, D. J. Fleet, and M. Norouzi, “Palette: Image-to-image diffusion models,” CoRR, vol. abs/2111.05826, 2021. [Online]. Available: https://arxiv.org/abs/2111.05826

[23] B. Kawar, M. Elad, S. Ermon, and J. Song, “Denoising diffusion restoration models,” in Advances in Neural Information Processing

Systems, S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, Eds., vol. 35. Curran Associates, Inc., 2022, pp. 23 593–23 606. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/ 2022/file/95504595b6169131b6ed6cd72eb05616-Paper-Conference.pdf

[24] B. Kim, I. Han, and J. C. Ye, “Diffusemorph: Unsupervised deformable image registration using diffusion model,” 2022. [Online]. Available: https://arxiv.org/abs/2112.05149

[25] Z. Zhao, H. Bai, Y. Zhu, J. Zhang, S. Xu, Y. Zhang, K. Zhang, D. Meng, R. Timofte, and L. Van Gool, “Ddfm: Denoising diffusion model for multi-modality image fusion,” in 2023 IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 8048–8059.

[26] Z. Cao, S. Cao, L.-J. Deng, X. Wu, J. Hou, and G. Vivone, “Diffusion model with disentangled modulations for sharpening multispectral and hyperspectral images,” Information Fusion, vol. 104, p. 102158, 2024. [Online]. Available: https://www.sciencedirect.com/science/article/pii/ S1566253523004748

[27] S. Chen, P. Sun, Y. Song, and P. Luo, “Diffusiondet: Diffusion model for object detection,” in 2023 IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 19 773–19 786.

[28] A. Navon, A. Shamsian, I. Achituve, H. Maron, K. Kawaguchi, G. Chechik, and E. Fetaya, “Multi-task learning as a bargaining game,” arXiv preprint arXiv:2202.01017, 2022.

[29] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in Advances in Neural Information Processing Systems, H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, Eds., vol. 33. Curran Associates, Inc., 2020, pp. 6840–6851. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/ 2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf

[30] C. Li, D. Song, R. Tong, and M. Tang, “Multispectral pedestrian detection via simultaneous detection and segmentation,” in British Machine Vision Conference (BMVC), 2018.

[31] Y. Mo, X. Kang, S. Zhang, P. Duan, and S. Li, “A robust infrared and visible image registration method for dual-sensor uav system,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1–13, 2023.

[32] T. Zheng, G. Dong, P. Zhang, X. He, and C. Ren, “Plug-and-play general image registration for misaligned multi-modal image fusion,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 35, no. 10, pp. 10 017–10 031, 2025.

[33] H. Li, T. Xu, X.-J. Wu, J. Lu, and J. Kittler, “Lrrnet: A novel representation learning guided fusion network for infrared and visible images,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 9, pp. 11 040–11 052, 2023.

[34] Z. Wang, A. Bovik, H. Sheikh, and E. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600–612, 2004.

![](images/6350fa33f63ee686c66ce22de2c64b898ba2ad5cd0c7d419b83014998cea315a.jpg)

![](images/1a9a369465dfc8bb00944c07bef3672032dbd7c95f6bdec11d0ef76d6c337516.jpg)

[35] Z. Wang and A. Bovik, “A universal image quality index,” IEEE Signal Processing Letters, vol. 9, no. 3, pp. 81–84, 2002.

[36] T. Lin, M. Maire, S. J. Belongie, L. D. Bourdev, R. B. Girshick, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick, “Microsoft´ COCO: common objects in context,” CoRR, vol. abs/1405.0312, 2014. [Online]. Available: http://arxiv.org/abs/1405.0312

Yunhao Li received the B.Eng. degree in automation from Xi’an Jiaotong University, Xi’an, China, in 2025. He is currently pursuing the M.Eng. degree in control science and engineering with the School of Automation Science and Engineering, Xi’an Jiaotong University, Xi’an, China.

Jing Liu (Senior Member, IEEE) received the B.Sc. and M.Sc. degrees in automatic control from Northwestern Polytechnical University, Xi’an, China, in 1998 and 2000, respectively, and the Ph.D. degree in electrical and computer engineering from the National University of Singapore, Singapore, in 2007.

and Information Engineering, Xi’an Jiaotong University, Xi’an. Her research interests include tensor decomposition, information fusion, signal processing, and applications in target detection, recognition, and tracking.

His research interests include image processing, target tracking and deep learning.

From 2006 to 2008, she was a Research Fellow with the Department of Electrical and Computer Engineering, National University of Singapore. She is currently a Professor with the School of Electronics

[37] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” CoRR, vol. abs/2010.02502, 2020. [Online]. Available: https://arxiv.org/abs/2010.02502

![](images/7c44df1287553afbcf270e9e58c82dd16284d9f9f5c10bfcce3db96b741de294.jpg)

Qianchao He received the B.Eng. degree in aircraft control and information engineering from Northwestern Polytechnical University, Xi’an, China, in 2025. He is currently pursuing the M.Eng. degree in control science and engineering with the School of Automation Science and Engineering, Xi’an Jiaotong University, Xi’an, China.

Xiaoqing Tian received the M.E. degree from Ningxia University, Yinchuan, China, in 2017, and the Ph.D. degree in control science and engineering from Xi’an Jiaotong University, Xi’an, China, in 2024. She is currently a Lecturer with the School of Electrical Engineering and Automation, Xi’an University of Science and Technology, Xi’an.

His research interests include computer vision and diffusion models.

![](images/7fabcdbbd257b6e36d51f09558bf52369f86190a62a2542157376956d53cfeed.jpg)

Her research interests include radar signal processing and target tracking.