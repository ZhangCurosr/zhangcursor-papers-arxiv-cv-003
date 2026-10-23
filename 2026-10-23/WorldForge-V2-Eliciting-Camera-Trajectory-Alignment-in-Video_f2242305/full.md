# WorldForge V2: Eliciting Camera-Trajectory Alignment in Video Diffusion Models

Chenxi Song, Yanming Yang, Tong Zhao, Ruibo Li,

Joey Tianyi Zhou, Senior Member, IEEE, and Chi Zhang, Member, IEEE

Abstract—Video diffusion models have learned rich 3D/4D priors, making them promising for scene generation and video re-rendering. However, their use in these spatial tasks is limited by imprecise camera control, spatial and temporal inconsistencies, and the entanglement of camera motion with scene dynamics. Existing approaches, such as task-specific fine-tuning and post-process warping, often introduce visual artifacts, generalize poorly, or incur high computational costs. We propose WorldForge V2, which aims to achieve accurate camera control while preserving the generative capabilities of pretrained video models. We first develop a training-free guidance framework using warped references to iteratively refine generation along a target camera path. Optical-flow analysis identifies motionrelated latent channels for selective guidance, helping separate motion control from appearance. A dual-path strategy further compares guided and unguided predictions to correct artifacts from imperfect references. Together, these components guide viewpoint changes while preserving scene appearance and object motion. To reduce refinement costs, v-cache approximates the correction using the preceding denoising prediction, avoiding an additional model evaluation. We further extend motion localization from latent channels to network blocks and introduce Trajectory LoRA, which concentrates adaptation on blocks most responsive to trajectory guidance. A two-stage training strategy teaches these adapters to denoise the states encountered during guided generation, using limited supervision effectively. This enables reliable camera control with very little training data, with a single training scene already supporting generalization to unseen scenes. Quantitative and qualitative evaluations across multiple datasets show that our method achieves state-of-the-art generation quality and trajectory accuracy in camera-controlled video generation.

Index Terms—Camera control, video diffusion models, 3D/4D generation.

## I. INTRODUCTION

R <sup>ECENT</sup> <sup>video</sup> <sup>diffusion</sup> <sup>models</sup> <sup>(VDMs)</sup> <sup>can</sup> <sup>synthesize</sup>videos with coherent motion, consistent scene content, and plausible viewpoint changes over time [1]–[3]. Trained on large-scale video data, these models acquire strong spatiotemporal priors, making them powerful backbones for spatial-intelligence tasks such as 3D/4D generation, video rerendering, and world modeling [4]–[11]. In these applications, the camera trajectory determines how a scene is observed over time, which regions become visible, and how viewpoint changes relate to scene geometry. Effective trajectory control is therefore essential. The generated video should follow a specified camera path while preserving scene content, object motion, temporal coherence, and visual quality.

![](images/a8befdc9ad9570eaea4e5c5a6db363a554062824e82fd68e235132902430b04d.jpg)  
Fig. 1. Comparison with existing methods. With limited training data and iterations, our method achieves state-of-the-art visual quality and trajectory accuracy in camera-controlled video generation.

To realize such control, existing approaches broadly follow two directions. One direction adapts VDMs for camera control through additional trainable modules, pose-aware conditioning interfaces, or model fine-tuning [5], [12]–[16]. These methods provide explicit camera control but often require largescale trajectory–video supervision or careful fine-tuning. With limited supervision, such adaptation may weaken the visual and temporal priors of the pretrained model. Recent work has therefore explored training-free alternatives that keep the pretrained VDM fixed and inject trajectory cues only during inference [6], [17], including the zero-shot variant of Warp-as-History [18]. A representative route is warping-and-repainting, which warps already observed content along the target camera path and uses the pretrained video model to synthesize newly revealed regions. This route enables direct camera control at inference time while preserving the pretrained model, making it well suited to downstream spatial-intelligence tasks.

Despite these advantages, warping-and-repainting can introduce a denoising-state mismatch during diffusion sampling. When warped visible regions are directly combined with model-generated regions in a noisy latent, the two parts may correspond to different noise states. The warped regions are associated with externally sampled noise, whereas the generated regions follow the model’s own denoising trajectory. We refer to this mismatch as a denoising-state domain gap. It can lead to boundary discontinuities, geometric artifacts, noisy patches, and texture degradation. The key challenge is therefore to inject useful trajectory evidence from warped references while reducing the denoising-state gap introduced by their fusion.

To address this problem, we propose WorldForge V2, a camera-trajectory-control framework that helps VDMs use imperfect warped trajectory signals more effectively while preserving generation quality. WorldForge V2 follows the warping-and-repainting route. First, it addresses the incompatibility introduced by warped-reference fusion: direct fusion in noisy latents mixes externally noised warped regions with generated regions following the model’s current denoising state. WorldForge V2 reduces this mismatch by fusing the warped reference into the model-predicted clean estimate and mapping the fused result back to the current denoising state, enabling training-free trajectory-aligned video generation. Second, clean-estimate fusion provides a reliable way to inject warped references into the denoising process by reducing noisy-state mismatch. To further improve its sampling efficiency, we introduce v-cache, which approximates the cleanestimate correction directly in noisy latents. In this way, warped-reference guidance can retain the benefit of cleanestimate fusion while reducing the extra sampling overhead. Finally, full-model fine-tuning and large trainable control modules involve optimizing a large number of parameters, making it difficult to use limited training data efficiently. To address this problem, we build on our pipeline and further investigate an optimization-based approach to camera-trajectory control. Since different layers in VDMs specialize in different functions, we extend the motion-channel localization used in FLF to model adaptation. We identify trajectory-sensitive blocks and apply LoRA only to these blocks. This selective adaptation provides a data-efficient alternative to broad model tuning.

First, we address the denoising-state incompatibility caused by warped-reference fusion. In direct noisy-space fusion, the warped visible regions and the model-generated regions may follow different denoising states, causing geometric artifacts, boundary discontinuities, and noisy patches. To mitigate this mismatch, Intra-Step Recursive Refinement (IRR) fuses the warped reference with the model-predicted clean estimate under the visibility mask and then re-noises the fused estimate to the current timestep. This fuse-then-renoise process associates all regions with the same Gaussian noise realization, thereby reducing their denoising-state gap. Camera-trajectory control also faces a typical motion–appearance disentanglement problem, as enforcing camera motion may inadvertently alter object and scene appearance. To address this problem, Flow-Gated Latent Fusion (FLF) uses optical-flow cues to locate channels that are highly correlated with motion and injects trajectory guidance only into these channels, thereby reducing appearance distortion. Finally, artifacts or misalignment in the warped reference may degrade the quality of the generated video. To mitigate this issue, we propose Dual-Path Self-Corrective Guidance (DSG). Inspired by classifierfree guidance [19], DSG compares the trajectory-guided and unguided denoising directions. Their difference captures the update introduced by trajectory guidance, which DSG uses to suppress unreliable deviations while retaining useful trajectory constraints. Together, IRR, FLF, and DSG enable trainingfree trajectory-aligned video generation while reducing the adverse effects of incompatible denoising states and unreliable reference signals.

Second, the fuse-then-renoise process in IRR provides an effective way to reduce the denoising-state gap introduced by warped-reference fusion. We further improve the efficiency of this process with v-cache. Our key observation is that the correction induced by clean-estimate fusion can be approximated directly in noisy latent space from a denoising prediction already available during sampling. Based on this observation, v-cache applies the approximated correction during warpedreference fusion, allowing trajectory guidance to remain on the original denoising path. This avoids an additional model evaluation and reduces inference latency.

Finally, we further develop an optimization-based extension of our pipeline for camera-trajectory control. Existing optimization-based methods often fine-tune the full model or introduce large trainable control modules, making inefficient use of limited supervision. Since camera control mainly depends on motion-related representations, we extend the motion-channel localization idea of FLF to model adaptation. Leveraging the functional specialization of different VDM layers [20], we identify trajectory-sensitive blocks, apply LoRA only to these blocks, and train the adapters with a two-stage optimization strategy tailored to our pipeline. This selective adaptation provides a data-efficient alternative to broad model tuning. In our experiments, we find that training data from a single scene is sufficient to enable effective trajectory following.

Our main contributions are summarized as follows:

• To address denoising-state mismatch, motion–appearance entanglement, and unreliable warped references, we develop a training-free guidance pipeline with IRR, FLF, and DSG, enabling trajectory-aligned video generation with fewer artifacts and less appearance distortion.

• To retain the benefit of clean-estimate correction with lower sampling cost, we introduce v-cache, which approximates the correction in noisy latent space and avoids an additional model evaluation, thereby reducing inference latency.

• To avoid data-inefficient full-model tuning, we identify trajectory-sensitive blocks, adapt only these blocks with LoRA, and optimize them using a two-stage strategy tailored to our pipeline; training data from a single scene is sufficient to enable effective trajectory following.

• Comparative experiments on multiple public datasets demonstrate accurate camera-path control and highquality video generation.

A preliminary version of this work was published at CVPR 2026 [21]. The source code is available at https://github.com/ Westlake-AGI-Lab/WorldForge.

![](images/5049388a3a8d763b4c814a514f1247a906e55275caeb7cccaf374970f79c9967.jpg)  
Fig. 2. Overview of WorldForge V2. Given a single image or video frame and a target camera trajectory, the scene geometry is reconstructed and rendered into trajectory-aligned warped frames. WorldForge V2 first performs training-free trajectory guidance in the predicted clean latent space using IRR, FLF, and DSG. Building on this framework, we further introduce an efficient route that directly fuses trajectory information in the noisy latent space with V-Cache, together with a trajectory-sensitive lightweight LoRA scheme for camera trajectory control using only limited training data. The resulting video follows the specified camera path while preserving the generative capability of the pretrained video diffusion model.

## II. RELATED WORK

We review four lines of work closely related to WorldForge V2: video diffusion priors for spatial generation, camera control in video diffusion models, guidance and correction strategies in generative models, and parameter-efficient adaptation for camera control.

## A. Video Diffusion Priors for Spatial Generation

Video diffusion models (VDMs) learn rich priors of realworld appearance, motion, temporal continuity, and viewpoint change from large-scale video data [1]–[3], [22]. Recent studies have transferred these priors to spatial generation tasks, including 3D/4D scene generation, novel-view video synthesis, and world modeling [4]–[6], [8]–[10], [14], [23]–[30]. This scene-level use of video priors complements explicit scene reconstruction based on NeRFs, Gaussian splatting, multi-view and light-field synthesis [31]–[37], as well as scene generation from panoramic images [38] and object-centric 3D generation based on diffusion models [39]–[43]. While reconstruction methods recover scene structure from observed views, VDMs can use their learned priors to generate plausible content beyond the observed regions. However, VDM pretraining does not explicitly align camera trajectories with scene content. As a result, these models can generate high-quality videos but cannot reliably follow a prescribed camera path, motivating growing research on trajectory-controlled video generation. WorldForge V2 combines warped geometric evidence with the generative prior of a pretrained VDM to control viewpoint changes.

## B. Camera Control in Video Diffusion Models

Existing camera-control methods broadly follow two directions. Training-based methods introduce pose-aware conditioning interfaces, additional control modules, or model finetuning to learn camera-controlled generation [5], [12]–[16], [44], [45]. These methods provide explicit camera control but usually require substantial trajectory–video supervision; when training data are limited, model adaptation may weaken the visual and temporal priors learned during pretraining. Recent work has therefore increasingly explored training-free inference-time control and lightweight adaptation, which preserve most or all of the pretrained model and inject trajectory cues through initial noise, latent transformations, rendered references, or history frames [6], [8], [17], [18], [46]–[48]. Across training-based and training-free methods, warping-andrepainting warps observed content along the target camera path and uses a VDM to generate newly revealed regions [49], [50]. This route provides explicit geometric correspondence and integrates naturally with depth estimation, point-based rendering, and visibility masks. WorldForge V2 adopts this route and studies how warped references can be used more effectively for camera-trajectory control while reducing the artifacts introduced during denoising. Related motion-control work also uses entity-specific 3D trajectories to control object motion [51].

## C. Guidance and Correction Strategies in Generative Models

Guidance methods steer diffusion or flow models toward desired outputs by modifying their sampling directions. Classifier-free guidance (CFG) combines conditional and unconditional predictions to improve alignment with semantic conditions [19]. Subsequent methods further adjust the guidance direction through negative or auxiliary branches, differential predictions, adaptive rescaling, resampling, and iterative repainting [52]–[58]. In 3D/4D generation, guidance is also widely used to enforce viewpoint consistency from rendered or warped observations [6], [8], [17], [59], [60].

However, conventional guidance does not directly address the artifacts introduced by noisy warped references. WorldForge V2 explores correction strategies for warping-guided generation, including correcting warped-reference fusion in noisy latent space and using the difference between guided and unguided predictions to refine the denoising process. Together, these strategies improve the quality of trajectory-controlled videos generated under warping guidance.

## D. Parameter-Efficient Adaptation for Camera Control

Model adaptation commonly uses trainable control branches or parameter-efficient fine-tuning. ControlNet and T2I-Adapter introduce lightweight conditioning modules, while LoRA adapts pretrained models through low-rank parameter updates [61]–[63]. These lightweight adaptation strategies have also been applied to motion learning, sparse-view synthesis, and trajectory-controlled generation with limited training data [18], [45], [64], [65]. Related studies further show that motion- and camera-related representations are concentrated in particular latent channels or network layers [20], [66]. Based on these findings, WorldForge V2 uses optical-flow cues in FLF to localize motion-related latent channels and reduce appearance interference. We further extend this idea to model adaptation by identifying trajectory-sensitive blocks and applying lightweight LoRA only to these blocks, enabling effective trajectory alignment with very limited training data.

## III. METHOD

WorldForge V2 (Fig. 2) aims to control camera trajectories while preserving the video priors of pretrained generation models. Given a source image or video and a target camera path, geometric warping first produces a trajectory reference and its visibility mask. WorldForge V2 then uses these signals to guide denoising and generate videos that follow the target path. Sec. III-A introduces the required notation and background. Sec. III-B presents the training-free guidance pipeline: IRR reduces the denoising-state gap through cleanestimate fusion, FLF restricts trajectory injection to motionrelated latent channels, and DSG corrects the guided prediction using the model’s original denoising path. Sec. III-C derives a noisy-latent correction that reduces the denoisingstate gap with fewer model evaluations. Finally, Sec. III-D identifies trajectory-sensitive network blocks and adapts them with LoRA using limited supervision, enabling data-efficient trajectory control for VDMs.

## A. Preliminaries

We introduce the notation and background required by WorldForge V2, including flow-based denoising, samplingtime guidance, low-rank adaptation, and geometry-based trajectory guidance.

Flow-Based Denoising. Diffusion and flow-based models both generate videos by iteratively transforming a noisy latent into a clean sample. A diffusion process has a corresponding probability-flow ODE and can be written in flow form through reparameterization [67]–[70]; the detailed relationship is provided in the supplementary material.

Our implementation uses flow-based video models [2], [71], so we describe sampling in velocity form. Let $\mathbf { x } _ { i }$ be the noisy video latent at noise level $\sigma _ { i } ,$ , and let ℓ denote the model condition. The pretrained model $f _ { \theta }$ predicts $\mathbf { v } _ { i } = f _ { \theta } ( \mathbf { x } _ { i } , \sigma _ { i } , \ell )$ A first-order solver updates the latent as

$$
\mathbf { x } _ { i - 1 } = \mathbf { x } _ { i } + ( \sigma _ { i - 1 } - \sigma _ { i } ) \mathbf { v } _ { i } , \qquad \sigma _ { i - 1 } < \sigma _ { i } .\tag{1}
$$

The same velocity also gives the one-step clean estimate

$$
\begin{array} { r } { \hat { \mathbf { x } } _ { 0 , i } = \mathbf { x } _ { i } - \sigma _ { i } \mathbf { v } _ { i } . } \end{array}\tag{2}
$$

Thus, $\mathbf { x } _ { i }$ is the current noisy state, whereas $\hat { \mathbf { x } } _ { 0 , i }$ is the clean content predicted from it. WorldForge V2 injects trajectory evidence by modifying either of these states before the next solver update.

Classifier-Free Guidance. Classifier-free guidance (CFG) [19] strengthens conditioning by comparing conditional and unconditional predictions. In the velocity formulation,

$$
\begin{array} { r } { \tilde { \mathbf { v } } _ { i } = f _ { \theta } \left( \mathbf { x } _ { i } , \sigma _ { i } , \ell \right) + \omega _ { \mathrm { C F G } } \left[ f _ { \theta } \left( \mathbf { x } _ { i } , \sigma _ { i } , \ell \right) - f _ { \theta } \left( \mathbf { x } _ { i } , \sigma _ { i } , \varnothing \right) \right] , } \end{array}\tag{3}
$$

where $\omega _ { \mathrm { C F G } }$ is the guidance weight and $\varnothing$ denotes the unconditional input. The prediction difference measures the update introduced by the condition. DSG later follows this principle by comparing denoising predictions with and without trajectory guidance.

Low-Rank Adaptation. Low-Rank Adaptation (LoRA) [63] keeps a pretrained weight $\mathbf { W } \in \mathbb { R } ^ { d _ { \mathrm { o u t } } \times d _ { \mathrm { i n } } }$ fixed and represents its trainable update with two low-rank matrices:

$$
\mathbf { W } ^ { \prime } = \mathbf { W } + \lambda _ { \mathrm { L } } \mathbf { B } \mathbf { A } ,\tag{4}
$$

where $\textbf { A } \in \mathbb { R } ^ { r \times d _ { \mathrm { i n } } }$ and $\mathbf { B } \in \mathbb { R } ^ { d _ { \mathrm { o u t } } \times r }$ are trainable, $r \ll$ min $( d _ { \mathrm { i n } } , d _ { \mathrm { o u t } } )$ , and $\lambda _ { \mathrm { L } }$ controls the adapter scale. LoRA therefore adapts a model with far fewer trainable parameters than full-model fine-tuning. Sec. III-D applies LoRA only to trajectory-sensitive network blocks.

Geometry-Based Warping and Trajectory Guidance. WorldForge V2 converts a target camera path into explicit trajectory evidence through geometric warping. Given a source image or video, we obtain depth and camera poses using available geometry estimators, such as VGGT and UniDepth [72], [73]. For source frame $\mathbf { I } _ { \mathrm { s r c } } ,$ source depth $\mathbf { D } _ { \mathrm { { s r c } } }$ , source pose $\mathbf { P _ { \mathrm { s r c } } } .$ , and target pose $\mathbf { P } _ { \mathrm { t a r } } .$ , the warping operator W produces a partial target view and its visibility mask:

$$
( \mathbf { I } _ { \mathrm { t a r } } ^ { \prime } , \mathbf { M } _ { \mathrm { t a r } } ) = \mathcal { W } ( \mathbf { I } _ { \mathrm { s r c } } , \mathbf { D } _ { \mathrm { s r c } } , \mathbf { P } _ { \mathrm { s r c } } , \mathbf { P } _ { \mathrm { t a r } } ) .\tag{5}
$$

We denote the VAE-encoded warped reference by $\mathbf { x } _ { \mathrm { t r a j } }$ and its latent-space mask by M. This partial reference follows the target path but contains holes, disocclusions, and depthboundary errors.

A common way to use this reference is to noise it to the current level and insert it into the visible region of $\mathbf { x } _ { i } .$ . Let $\epsilon _ { i }$ be the Gaussian noise used at level $\sigma _ { i } .$ Direct noisy-latent insertion is

$$
\mathbf { x } _ { i } ^ { \mathrm { { d i r } } } = ( 1 - \mathbf { M } ) \odot \mathbf { x } _ { i } + \mathbf { M } \odot [ ( 1 - \sigma _ { i } ) \mathbf { x } _ { \mathrm { { t r a j } } } + \sigma _ { i } \epsilon _ { i } ] .\tag{6}
$$

This operation combines an externally re-noised visible region with a model-generated region that follows the current denoising path. Their different noise states may create a denoising-state gap. Sec. III-B reduces this gap through cleanestimate fusion, while Sec. III-C derives a more efficient noisylatent approximation.

Camera trajectories can also be injected through other interfaces. Warp-as-History provides warped frames through the model’s history interface [18], whereas Go-with-the-Flow injects warped trajectory signals into the initial noise or initial denoising state [16]. We later use these interfaces to evaluate whether the trained LoRA transfers to different trajectory signals without additional fine-tuning.

## B. Inference-Time Trajectory Guidance

Geometric warping explicitly indicates how observed content should move along the target camera path. However, directly inserting the noised reference mixes the externally re-noised visible region with the model-generated region, producing a denoising-state gap that can cause noisy patches and structural errors. We therefore inject the reference through the model-predicted clean estimate and re-noise the fused result with a unified noise sample. We further restrict guidance to motion-related channels and calibrate the guided prediction against the model’s original denoising path. Together, these designs enable training-free trajectory control while preserving visual quality.

Intra-Step Recursive Refinement (IRR). Direct noisy-latent insertion assigns newly sampled noise to the warped visible region while retaining the model’s current denoising state elsewhere. To avoid directly combining these incompatible states, IRR first fuses the warped reference in the modelpredicted clean space and then re-noises the fused result with one Gaussian noise sample.

At denoising step i, IRR obtains the one-step clean estimate $\hat { \mathbf { x } } _ { 0 , i }$ from Eq. (2) and fuses the warped reference under the visibility mask:

$$
\begin{array} { r } { \tilde { \mathbf { x } } _ { 0 , i } = \mathbf { M } \odot \mathbf { x } _ { \mathrm { t r a j } } + ( 1 - \mathbf { M } ) \odot \hat { \mathbf { x } } _ { 0 , i } . } \end{array}\tag{7}
$$

The visible region follows the warped reference, while the remaining region retains the model-predicted clean content. IRR then samples $\epsilon _ { i } ,$ maps the fused estimate back to the current noise level, and predicts the corresponding velocity:

$$
\begin{array} { r l } & { \mathbf { x } _ { i } ^ { \mathrm { { i r r } } } = ( 1 - \sigma _ { i } ) \tilde { \mathbf { x } } _ { 0 , i } + \sigma _ { i } \epsilon _ { i } , } \\ & { \mathbf { v } _ { i } ^ { \mathrm { { i r r } } } = f _ { \theta } \left( \mathbf { x } _ { i } ^ { \mathrm { { i r r } } } , \sigma _ { i } , \ell \right) . } \end{array}\tag{8}
$$

Because all regions of ${ \bf x } _ { i } ^ { \mathrm { i r r } }$ are re-noised with the same $\epsilon _ { i } ,$ the warped and generated regions no longer come from directly stitched noisy states. IRR therefore preserves the trajectory evidence while reducing their denoising-state gap. Flow-Gated Latent Fusion (FLF). Camera-trajectory control should separate camera-induced motion from object appearance and intrinsic motion. However, VAE channels encode different types of information, so replacing all channels with the same trajectory signal may also distort appearance. This channel specialization is supported by MOFT, which uses PCA to identify motion-related channels [20]. Based on this observation, FLF uses optical flow to select channels whose temporal motion agrees with the target trajectory, requiring no gradient-based optimization during inference.

For denoising step i and channel $c ,$ FLF estimates optical flow between adjacent frames of $\hat { \mathbf { x } } _ { 0 , i } ^ { c }$ and $\mathbf { x } _ { \mathrm { t r a j } } ^ { c } .$ We use Farneback by default and evaluate RAFT as an alternative [74], [75]. Within the visible region, normalized endpoint error $E _ { i } ^ { c }$ angular error $A _ { i } ^ { c }$ , and outlier ratio $O _ { i } ^ { c }$ measure displacement, direction, and reliability differences. We combine them as

$$
S _ { i } ^ { c } = \gamma _ { E } \big ( 1 - E _ { i } ^ { c } \big ) + \gamma _ { A } \big ( 1 - A _ { i } ^ { c } \big ) + \gamma _ { O } \big ( 1 - O _ { i } ^ { c } \big ) ,\tag{9}
$$

where a higher $S _ { i } ^ { c }$ indicates stronger agreement with the warped trajectory.

FLF selects $\mathcal { C } _ { i } = \{ c \ | \ S _ { i } ^ { c } \ \ge \ \delta _ { i } \}$ with adaptive threshold $\delta _ { i } ~ = ~ \mu _ { i } ^ { S } - \lambda _ { i } \sigma _ { i } ^ { S }$ , where $\mu _ { i } ^ { S }$ and $\sigma _ { i } ^ { S }$ are the mean and standard deviation of channel scores. The schedule keeps more channels during early noisy steps to establish coarse motion and gradually tightens the selection as appearance details emerge. Detailed normalization, weights, and schedules are provided in the supplementary material.

FLF replaces full-channel fusion with

$$
\tilde { \mathbf { x } } _ { 0 , i } ^ { c } = \left\{ \begin{array} { l l } { \mathbf { M } \odot \mathbf { x } _ { \mathrm { t r a j } } ^ { c } + \left( 1 - \mathbf { M } \right) \odot \hat { \mathbf { x } } _ { 0 , i } ^ { c } , } & { c \in \mathcal { C } _ { i } , } \\ { \hat { \mathbf { x } } _ { 0 , i } ^ { c } , } & { c \not \in \mathcal { C } _ { i } . } \end{array} \right.\tag{10}
$$

The selected estimate $\tilde { \mathbf { x } } _ { 0 , i } ^ { c }$ is substituted for $\tilde { \mathbf { x } } _ { 0 , i }$ in Eq. (8) and then re-noised. Thus, trajectory evidence is injected into channels with clear motion agreement, while the remaining channels continue to follow the pretrained model, reducing texture and appearance distortion.

Dual-Path Self-Corrective Guidance (DSG). Warped trajectory references may contain distortions caused by depth errors, occlusions, or misalignment, and these errors can degrade the generated video. Inspired by CFG and AutoGuidance [19], [52], DSG compares a trajectory-guided prediction with the model’s original prediction and uses their difference to refine the guidance, allowing the result to follow the trajectory constraint while improving visual quality.

At step i, the base velocity $\mathbf { v } _ { i } = f _ { \theta } ( \mathbf { x } _ { i } , \sigma _ { i } , \ell )$ follows the pretrained denoising path and preserves visual fidelity, but does not enforce the target camera path. The guided velocity ${ \bf v } _ { i } ^ { \mathrm { i r r } }$ follows the trajectory but may inherit errors from the warped reference. DSG therefore uses $\mathbf { v } _ { i }$ as a quality reference to correct ${ \bf v } _ { i } ^ { \mathrm { i r r } }$

We empirically find that the cosine similarity between the two velocities is typically 0.3–0.6, corresponding to an angle of about 50<sup>◦</sup>–70<sup>◦</sup>, whereas the two branches in standard CFG are nearly parallel. This larger difference arises because IRR and FLF modify the latent itself rather than only the condition. Direct CFG extrapolation can therefore produce excessive updates. DSG instead matches the two velocity magnitudes and uses the trajectory-guided component orthogonal to the base direction, following the principle of APG [54]:

$$
\mathbf { v } _ { i } ^ { \mathrm { c o r r } } = \mathbf { v } _ { i } ^ { \mathrm { i r r } } + \rho \beta _ { i } \left[ \mathbf { v } _ { i } ^ { \mathrm { i r r } } - \alpha _ { i } \frac { \left\| \mathbf { v } _ { i } ^ { \mathrm { i r r } } \right\| _ { 2 } } { \left\| \mathbf { v } _ { i } \right\| _ { 2 } + \varepsilon } \mathbf { v } _ { i } \right] .\tag{11}
$$

Here, $\alpha _ { i } = \langle \mathbf { v } _ { i } ^ { \mathrm { i r r } } , \mathbf { v } _ { i } \rangle / ( \| \mathbf { v } _ { i } ^ { \mathrm { i r r } } \| _ { 2 } \| \mathbf { v } _ { i } \| _ { 2 } )$ measures directional agreement, $\beta _ { i } = \sqrt { 1 - \alpha _ { i } ^ { 2 } }$ reflects angular difference, and $\rho$ controls the correction strength. When the two paths agree, the correction remains small; when they diverge, the trajectoryspecific component receives more weight. The corrected velocity $\mathbf { v } _ { i } ^ { \mathrm { c o r r } }$ is then used for the solver update, preserving useful camera control while reducing artifacts from inaccurate warped references.

Overall, IRR injects the warped reference through cleanestimate fusion, FLF restricts guidance to motion-related channels, and DSG calibrates the guided prediction using the base denoising path. Together, they enable training-free camera-trajectory control while preserving scene appearance and reducing artifacts from imperfect warped references.

## C. Efficient Noisy-Latent Correction

The fuse-then-renoise process in IRR effectively reduces the denoising-state gap introduced by warped-reference fusion. Building on this, we further improve its sampling efficiency. IRR first evaluates the video model on the original latent to obtain the clean estimate and then evaluates the trajectoryguided latent to continue denoising. Our key observation is that the correction introduced by clean-estimate fusion can be written directly in noisy latent space. We therefore derive this correction and approximate it using the velocity cached from the preceding denoising step. This gives v-cache, a singleforward path for efficient trajectory guidance.

Noisy-Latent Correction. To identify the correction required by direct noisy-latent fusion, we compare the IRR state and direct-insertion state constructed from the same $\mathbf { x } _ { i } , \mathbf { x } _ { \mathrm { t r a j } } , \mathbf { M }$ and $\epsilon _ { i } .$ The IRR state first forms $\begin{array} { r } { \hat { \mathbf { x } } _ { 0 , i } = \mathbf { x } _ { i } - \sigma _ { i } \mathbf { v } _ { i } , } \end{array}$ fuses it with the warped reference, and re-noises the result. The directinsertion state instead inserts the noised reference in the visible region and retains $\mathbf { x } _ { i }$ elsewhere. Their difference is

$$
\begin{array} { r l } & { \mathbf { x } _ { i } ^ { \mathrm { i r r } } - \mathbf { x } _ { i } ^ { \mathrm { d i r } } } \\ & { \ = ( 1 - \sigma _ { i } ) \Big [ \mathbf { M } \odot \mathbf { x } _ { \mathrm { t r a j } } + \left( 1 - \mathbf { M } \right) \odot \left( \mathbf { x } _ { i } - \sigma _ { i } \mathbf { v } _ { i } \right) \Big ] + \sigma _ { i } \epsilon _ { i } } \\ & { \quad - \left\{ \left( 1 - \mathbf { M } \right) \odot \mathbf { x } _ { i } + \mathbf { M } \odot \left[ \left( 1 - \sigma _ { i } \right) \mathbf { x } _ { \mathrm { t r a j } } + \sigma _ { i } \epsilon _ { i } \right] \right\} } \\ & { \ = ( 1 - \mathbf { M } ) \odot \Big [ \big ( 1 - \sigma _ { i } \big ) \big ( \mathbf { x } _ { i } - \sigma _ { i } \mathbf { v } _ { i } \big ) + \sigma _ { i } \epsilon _ { i } - \mathbf { x } _ { i } \Big ] } \\ & { \ = ( 1 - \mathbf { M } ) \odot \sigma _ { i } \Big \{ \epsilon _ { i } - \underbrace { \left[ \mathbf { x } _ { i } + \left( 1 - \sigma _ { i } \right) \mathbf { v } _ { i } \right] } _ { \mathrm { o n e \ s t e p \ i n e r s i o n \ n o i s e } } \Big \} . } \end{array}\tag{12}
$$

The visible region cancels because both states use the same re-noised warped reference there. The remaining difference lies in the model-generated region. Under the flow formulation, $\mathbf { x } _ { i } + ( 1 - \sigma _ { i } ) \mathbf { v } _ { i }$ is the one-step inversion noise inferred from the current latent and velocity. Eq. (12) therefore shows that IRR differs from direct insertion by a model-dependent noise correction outside the visible mask, making the two regions more compatible at the current noise level.

Cached Approximation. The correction in Eq. (12) requires the current base velocity $\mathbf { v } _ { i } ,$ adding a model evaluation on $\mathbf { x } _ { i } .$ Diffusion caching methods reuse intermediate computations by exploiting the correlation between predictions at nearby denoising steps [76]. Following this idea, v-cache approximates $\mathbf { v } _ { i }$ with the velocity cached from the preceding step, denoted by $\bar { \bf v } _ { i } .$ . Starting from the second guided step, the corrected latent is

$$
\mathbf { x } _ { i } ^ { \mathrm { v c } } = \mathbf { x } _ { i } ^ { \mathrm { d i r } } + ( 1 - \mathbf { M } ) \odot \sigma _ { i } \left[ \epsilon _ { i } - ( \mathbf { x } _ { i } + ( 1 - \sigma _ { i } ) \bar { \mathbf { v } } _ { i } ) \right] .\tag{13}
$$

The first guided step has no cached velocity and therefore uses direct noisy-latent insertion. At subsequent steps, the model is evaluated once on $\mathbf { x } _ { i } ^ { \mathrm { v c } }$ to obtain $\mathbf { v } _ { i } ^ { \mathrm { v c } } = f _ { \theta } ( \mathbf { x } _ { i } ^ { \mathrm { v c } } , \sigma _ { i } , \boldsymbol { \ell } ) ;$ this velocity updates the sampler and is cached for the next step. Thus, v-cache approximates the IRR correction with one model evaluation per guided step. The experiments compare this approximation with the exact current-step correction in terms of noise statistics and generation quality.

## D. Trajectory-Sensitive LoRA Adaptation

Optimization-based camera control often relies on fullmodel fine-tuning or large trainable modules, requiring substantial trajectory–video data and computation. With limited supervision, broad parameter updates may also weaken pretrained visual and spatiotemporal priors, causing degraded geometry or implausible structures. We therefore seek a dataefficient adaptation strategy that strengthens trajectory control with limited supervision while restricting parameter updates to preserve the pretrained priors.

Our method combines trajectory-sensitive block selection with v-cache-aligned LoRA training. FLF analyzes how motion and appearance features are distributed across latent channels; we extend this localization idea to network blocks and apply LoRA only to blocks that are sensitive to trajectory signals. We then inject trajectories through v-cache and train these adapters on latent states produced by v-cache guidance. This design concentrates limited supervision on trajectoryrelevant components and enables data-efficient adaptation.

Trajectory-Supervised Data Construction. To construct paired trajectory references and target videos, we use the public DL3DV dataset [77], which provides multi-view observations and optimized 3D Gaussian Splatting representations [32]. For each scene, we select multiple source views and sample several target camera paths for each view. The 3DGS representation provides the source-view depth and renders RGB frames along each target path.

Using the source depth, we warp each source image along the sampled path according to Eq. (5), producing a warped reference and visibility mask. We render the same 3DGS scene along the identical path to obtain the clean target video. Each pair therefore shares the same trajectory: the warped video provides the trajectory condition, and the rendered video provides geometrically aligned supervision. Video-VAE encoding yields the trajectory latent $\mathbf { x } _ { \mathrm { t r a j } }$ and clean target latent $\mathbf { x } _ { 0 } ^ { \mathrm { g t } }$

Trajectory-Sensitive Block Identification. Prior analyses suggest that motion-related responses are not uniformly distributed across video-model layers [20], [66]. To identify blocks that are sensitive to trajectory control, we extend the motion–appearance localization idea of FLF from latent channels to network blocks. We treat all self-attention blocks as candidates and determine their sensitivity directly from their responses to trajectory supervision.

To measure this sensitivity, we attach a lightweight probe LoRA to every candidate block while keeping the pretrained model fixed. A small set of warped-reference–target pairs is encoded into latent space. At a sampled noise level $\sigma ,$ we interpolate between the clean target and trajectory latent and train the probe with

$$
\begin{array} { r l } & { \mathbf { x } _ { \sigma } ^ { \mathrm { p r o b e } } = ( 1 - \sigma ) \mathbf { x } _ { 0 } ^ { \mathrm { g t } } + \sigma \mathbf { x } _ { \mathrm { t r a j } } , } \\ & { \mathcal { L } _ { \mathrm { p r o b e } } = \mathbb { E } \left[ \left\| \mathbf { v } _ { \theta , \phi _ { \mathrm { p } } } \left( \mathbf { x } _ { \sigma } ^ { \mathrm { p r o b e } } , \sigma , \ell \right) - \frac { \mathbf { x } _ { \sigma } ^ { \mathrm { p r o b e } } - \mathbf { x } _ { 0 } ^ { \mathrm { g t } } } { \sigma } \right\| _ { 2 } ^ { 2 } \right] , } \end{array}\tag{14}
$$

where $\phi _ { \mathrm { p } }$ denotes the probe parameters. The probe learns the denoising direction from the warped trajectory state to the clean target at different noise levels. Brief probe training provides the responses needed to measure block sensitivity.

After probe training, we estimate each block’s contribution using Shapley analysis [78]. We score each block subset by the normalized projection of its velocity change onto the full probe LoRA response, as defined in the supplementary material. The Shapley value $\Phi _ { b }$ averages the marginal change in this score when the LoRA in block b is enabled across sampled block combinations. We rank blocks by $\Phi _ { b }$ and retain LoRA only in $B _ { \mathrm { t r a j } } = \mathrm { T o p K } _ { b } ( \Phi _ { b } )$ for final adaptation.

Data-Efficient Trajectory LoRA Training. The final LoRA is used with v-cache, so its training inputs should reflect the states encountered during v-cache inference. Standard flow matching constructs inputs by independently noising the clean target. In contrast, a v-cache state also contains the warped reference, visibility mask, and cached inversion-noise correction. Training only on independently noised targets would therefore leave a gap between training and inference states.

To construct v-cache-aligned inputs, we use two stages. The first records real intermediate states from v-cache-guided denoising and uses them to warm up the selected adapters. The second starts from these states and constructs additional inputs that retain the same warped-reference insertion and vcache correction at sampled noise levels. Both stages optimize

$$
\mathcal { L } _ { \mathrm { L o R A } } = \mathbb { E } \left[ \left. \mathbf { v } _ { \theta , \phi } \left( \tilde { \mathbf { x } } _ { \sigma } , \sigma , \ell \right) - \frac { \tilde { \mathbf { x } } _ { \sigma } - \mathbf { x } _ { 0 } ^ { \mathrm { g t } } } { \sigma } \right. _ { 2 } ^ { 2 } \right] ,\tag{15}
$$

where $\tilde { \mathbf { x } } _ { \sigma }$ is a v-cache-aligned state and $\phi$ denotes the selected LoRA parameters.

In the first stage, we run v-cache-guided sampling with a fixed noise schedule and save several intermediate states as anchors. For anchor $\mathbf { x } _ { a }$ at noise level $\sigma _ { a } ,$ we set $\tilde { \mathbf { x } } _ { \sigma _ { a } } = \mathbf { x } _ { a } .$ These anchors already contain the warped reference, visibility mask, and cached correction used during inference, allowing LoRA to first adapt to the actual v-cache input distribution.

The second stage constructs v-cache-consistent inputs at sampled noise levels. Starting from $\left( \mathbf { x } _ { a } , \sigma _ { a } \right)$ , the warmed-up model predicts $\mathbf { v } _ { a } = \mathbf { v } _ { \theta , \phi } ( \mathbf { x } _ { a } , \sigma _ { a } , \ell )$ and propagates the anchor to $\sigma _ { d } \mathbf { : }$

$$
\begin{array} { r } { \bar { \mathbf { x } } _ { d } = \mathbf { x } _ { a } - ( \sigma _ { a } - \sigma _ { d } ) \mathbf { v } _ { a } . } \end{array}\tag{16}
$$

We then apply the same warped-reference insertion, visibility mask, and v-cache correction used during inference to obtain $\tilde { \mathbf { X } } _ { \sigma _ { d } }$ . This state preserves the structure of a v-cache input while extending training beyond the recorded anchor levels.

The first stage provides real v-cache states, while the second constructs aligned states at additional noise levels, reducing the training–inference gap and extending adaptation beyond the saved anchors. Detailed training settings are provided in the supplementary material. At inference, only the LoRA adapters in $B _ { \mathrm { t r a j } }$ are loaded. Experiments show that training on one scene generalizes to unseen scenes. The trained LongCat adapter also transfers to other trajectory-injection methods without additional fine-tuning.

## IV. EXPERIMENTS

Our experiments examine how rendered-reference guidance combines camera control with the scene priors of pretrained video models. We first evaluate the training-free framework in static 3D generation and dynamic 4D re-rendering, then study V-Cache correction and data-efficient LoRA adaptation. Transfer experiments examine the reuse of the learned control, and ablations establish the roles of guidance, block selection, and training. We denote the IRR–FLF–DSG framework as Ours (Training-Free) and the adapted models as Ours (LongCat) and Ours (H3).

## A. Experimental Setup

a) Implementation: For training-free camera control, we use Wan2.1 I2V-14B [2], with IRR applied during early denoising. We implement Trajectory LoRA on LongCat-Video [71] and MiniMax-H3 by adapting selected selfattention projections while keeping the pretrained weights fixed. Both models use anchor warm-up followed by training on guided latent states. Detailed training and inference settings are provided in the supplementary material.

b) Data and evaluation protocol: The training-free evaluation covers static scenes and dynamic videos from diverse sources. The static 3D setting contains more than 70 source views from over 40 scenes, including LLFF [79], Tanks and Temples [80], Mip-NeRF 360 [81], and real or generated images. The dynamic 4D setting contains more than 50 trajectory-controlled outputs from over 30 source videos, including DAVIS [82], movie clips, and videos generated by video diffusion models.

For LoRA adaptation, we construct trajectory–video pairs from optimized 3D Gaussian Splatting representations [32] of DL3DV scenes [77]. Warping a source view along a target camera path provides the partial reference and its visibility mask; rendering the scene along the same path supplies the target video. The main comparison uses 200 test cases, with five random camera trajectories per scene across 40 scenes. Sequences contain 25–93 frames, with the length adapted to each method’s supported input format. Training and test scenes are disjoint.

c) Metrics: We evaluate visual quality and camera accuracy together. The training-free experiments use FID and CLIP similarity for static scenes, and FVD and CLIP-V similarity for dynamic videos. For Trajectory LoRA, we additionally evaluate target-view reconstruction with pixel-level metrics,

![](images/fff6b3f1dd489f23953c81cbc6daf8b93c358b104482706e0a3f725ee59f9656.jpg)  
Fig. 3. Qualitative comparison for 3D novel-view synthesis and 4D trajectory-controlled re-rendering. For single-image 3D generation (left), our method produces more coherent geometry and plausible newly revealed content. For dynamic re-rendering (right), it preserves human appearance and object motion while reducing floating structures and flattened faces. The training-free guidance yields more realistic results in both static and dynamic scenes.

PSNR and SSIM, and the perceptual metric LPIPS. Vis-LPIPS and Disocc-LPIPS assess preservation of visible content and completion of newly exposed regions, respectively. ATE measures overall trajectory agreement, while RPE-T and RPE-R measure local translation and rotation errors. Metric implementations are specified in the supplementary material.

## B. Inference-Time Camera-Trajectory Control

We evaluate whether warped-reference guidance can follow a prescribed camera path while retaining the scene priors of a frozen video backbone. For static 3D generation, we compare with See3D [49], ViewCrafter [4], ViewExtrapolator [17], TrajectoryAttention [15], TrajectoryCrafter [5], and NVS-Solver [6]. For dynamic 4D re-rendering, we compare with ViewExtrapolator, TrajectoryAttention, and TrajectoryCrafter using matched source videos and target trajectories. ReCam-Master [14] does not directly accept warped reference videos as trajectory inputs and is therefore included only in the qualitative comparison.

Tables I and II show that Ours (Training-Free) combines accurate camera motion with strong generation quality. It achieves the best results on all static metrics and leads in FVD, CLIP-V similarity, RPE-T, and RPE-R on dynamic videos. Under large viewpoint changes, our method preserves human identity, object motion, and scene structure (Fig. 3). Competing methods more often produce flattened geometry, floating structures, or implausible content in newly revealed regions. The warped references provide explicit camera constraints, while the video model supplies the scene content needed for realistic 3D and 4D generation.

## C. V-Cache Correction and Data-Efficient LoRA

The training-free results establish rendered references as an effective source of camera guidance. We next examine two

## TABLE I

QUANTITATIVE COMPARISON ON 3D STATIC SCENES USING OFFICIAL IMPLEMENTATIONS AND IDENTICAL SOURCE INPUTS. FID AND CLIP ASSESS GENERATION QUALITY; ATE, RPE-T, AND RPE-R ASSESS TRAJECTORY ACCURACY. OUR TRAINING-FREE METHOD LEADS ON ALL METRICS. ↑: HIGHER IS BETTER; ↓: LOWER IS BETTER. BOLD AND UNDERLINING MARK THE BEST AND SECOND-BEST RESULTS.

<table><tr><td>Method</td><td>FID↓ CLIP↑</td><td>ATE↓</td><td>RPE-T↓</td><td>RPE-R↓</td></tr><tr><td>See3D</td><td>123.26</td><td>0.941 0.091</td><td>0.089</td><td>0.250</td></tr><tr><td>ViewCrafter</td><td>117.50</td><td>0.930 0.236</td><td>0.315</td><td>0.728</td></tr><tr><td>ViewExtrapolator</td><td>125.50</td><td>0.930 0.183</td><td>0.260</td><td>0.882</td></tr><tr><td>TrajectoryAttention</td><td>122.37</td><td>0.920 0.159</td><td>0.238</td><td>0.532</td></tr><tr><td>TrajectoryCrafter</td><td>111.49</td><td>0.910 0.090</td><td>0.152</td><td>0.267</td></tr><tr><td>NVS-Solver</td><td>118.64</td><td>0.937 0.224</td><td>0.268</td><td>1.056</td></tr><tr><td>Ours (Training-Free)</td><td>96.08</td><td>0.948 0.077</td><td>0.086</td><td>0.221</td></tr></table>

questions: can the clean-estimate correction be approximated directly in noisy latent space to reduce sampling cost, and can limited trajectory supervision be concentrated in the network components most useful for camera control? V-Cache

## TABLE II

QUANTITATIVE COMPARISON ON 4D DYNAMIC RE-RENDERING USING OFFICIAL IMPLEMENTATIONS WITH MATCHED SOURCE VIDEOS AND TARGET TRAJECTORIES. FVD AND CLIP-V ASSESS VIDEO QUALITY; ATE, RPE-T, AND RPE-R ASSESS TRAJECTORY ACCURACY. OUR   
METHOD COMBINES THE BEST VIDEO QUALITY WITH THE LOWEST LOCAL   
CAMERA ERRORS. ↑: HIGHER IS BETTER; ↓: LOWER IS BETTER. BOLD AND UNDERLINING MARK THE BEST AND SECOND-BEST RESULTS.

<table><tr><td>Method</td><td>FVD↓</td><td>CLIP-V↑ ATE↓</td><td>RPE-T↓</td><td>RPE-R↓</td></tr><tr><td>ViewExtrapolator</td><td>108.48</td><td>0.913 1.040</td><td>1.208</td><td>4.750</td></tr><tr><td>TrajectoryAttention</td><td>106.94</td><td>0.911 0.605</td><td>1.238</td><td>3.560</td></tr><tr><td>TrajectoryCrafter</td><td>97.31</td><td>0.923 0.431</td><td>1.078</td><td>8.950</td></tr><tr><td>Ours (Training-Free)</td><td>93.17</td><td>0.938 0.527</td><td>0.826</td><td>2.690</td></tr></table>

![](images/e18f79466f24ba700b1a887fbb7d93d1c753bd083d9fc516ad5d847e8ab2ac98.jpg)

TABLE III  
QUANTITATIVE COMPARISON OF TRAJECTORY LORA WITH CAMERA-CONTROL BASELINES. PSNR, SSIM, AND LPIPS ASSESS TARGET-VIEW RECONSTRUCTION; VIS-LPIPS AND DISOCC-LPIPS ASSESS VISIBLE AND NEWLY EXPOSED REGIONS. BOTH ADAPTED BACKBONES IMPROVE RECONSTRUCTION ACROSS THESE METRICS, PRESERVING OBSERVED CONTENT AND COMPLETING NEW REGIONS MORE FAITHFULLY. H3 ALSO ACHIEVES THE LOWEST LOCAL CAMERA ERRORS. BOLD AND UNDERLINING MARK THE BEST AND SECOND-BEST RESULTS.
<table><tr><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>Vis-LPIPS↓</td><td>Disocc-LPIPS↓ ATE↓</td><td>RPE-T↓</td><td>RPE-R↓</td></tr><tr><td>UniWorld-View</td><td>17.15</td><td>0.5958</td><td>0.3370</td><td>0.2799</td><td>0.4431 0.00625</td><td>0.2333</td><td>0.1218</td></tr><tr><td>TrajectoryCrafter</td><td>17.41</td><td>0.5986</td><td>0.3558</td><td>0.3066</td><td>0.4512 0.00710</td><td>0.2013</td><td>0.0990</td></tr><tr><td>SymphoMotion</td><td>15.38</td><td>0.4760</td><td>0.4433</td><td>0.4215</td><td>0.4908 0.02292</td><td>0.4065</td><td>0.2196</td></tr><tr><td>Warp-as-History</td><td>15.48</td><td>0.4553</td><td>0.4206</td><td>0.3829</td><td>0.4939 0.03533</td><td>0.4856</td><td>0.2398</td></tr><tr><td>Ours (LongCat)</td><td>19.72</td><td>0.6663</td><td>0.2748</td><td>0.2231</td><td>0.3640 0.00905</td><td>0.2725</td><td>0.1324</td></tr><tr><td>Ours (H3)</td><td>20.21</td><td>0.6806</td><td>0.2698</td><td>0.2244</td><td>0.3459 0.00685</td><td>0.1647</td><td>0.0780</td></tr></table>

addresses the first question, while Trajectory LoRA uses this selective adaptation to learn control from very little data.

a) V-Cache correction: Direct reference insertion combines an externally noised observation with a latent following the model’s denoising path. To test their noise-state compatibility, we compare Direct Insertion, Only IRR, and V-Cache across denoising steps with the same inputs, target trajectories, and sampling settings (Fig. 4). For a latent $x _ { i }$ at noise level $\sigma _ { i } ,$ , the model velocity $v _ { i }$ gives the inferred noise endpoint $\hat { \epsilon } _ { i } = x _ { i } + ( 1 - \sigma _ { i } ) v _ { i }$ . We compute its mean and variance in the warped-visible and model-generated regions and plot the mean and log-variance gaps. Direct Insertion leaves persistent gaps, accompanied by noisy patches, discontinuous boundaries, and local structural errors. IRR reduces these gaps by fusing the reference in clean-estimate space and re-noising the result with one noise sample. V-Cache achieves a similar reduction without additional model evaluations (NFE), making correction more efficient.

![](images/96c2b878779f416e9fe97f69cb3494da6cf84493819a91c5bd3c309b7c83c491.jpg)  
Fig. 4. Noise-state consistency under different reference-insertion strategies. Direct Insertion leaves gaps in the inferred-noise mean and log-variance between visible and model-generated regions, accompanied by noisy patches and broken structure. IRR reduces these gaps, and V-Cache approximates its correction while retaining continuous surfaces and clear geometric detail.

b) Data-efficient LoRA: Trajectory LoRA adapts only selected trajectory-sensitive blocks to strengthen the model’s response to camera guidance. We use V-Cache to inject the target trajectory and compare the adapted models with UniWorld-View [83], TrajectoryCrafter [5], SymphoMotion [84], and Warp-as-History [18] on trajectory accuracy, preservation of observed content, and completion of newly revealed regions.

Both models use rank-128 LoRA and 50 trajectory–video training pairs. LongCat adapts its 36 most trajectory-sensitive blocks (75%) for 1,200 updates using eight scenes; H3 adapts its 38 most sensitive blocks (approximately 75%) for 1,500 updates using one scene. These main-comparison settings are the defaults for ablations, which vary the factor under study.

Both adapted models outperform the baselines on all five reconstruction metrics (Table III). The gains in Vis-LPIPS and Disocc-LPIPS show improved preservation of observed content and completion of newly exposed regions. H3 also achieves the lowest local camera errors, combining faithful scene reconstruction with accurate camera motion. Figure 5 illustrates these improvements in thin structures and partially revealed objects: our models retain material detail and coherent shapes where the warped reference is incomplete, producing fewer rendering artifacts and more continuous geometry along the requested path.

## D. Additional Results

a) Generalization and transfer: The inference-time guidance transfers directly to the U-Net-based SVD [1] and the Transformer-based LongCat-Video, and remains effective with different depth and optical-flow estimators (see the supplementary material). Trajectory LoRA offers a complementary form of transfer. The same LongCat adapter can be attached to Warp-as-History [18] and Go-with-the-Flow [16] without further fine-tuning. It improves the model’s response to the injected trajectory under both methods, reducing structural degradation and preserving more surface and object detail (Fig. 6). These results show that the selected blocks learn broader trajectory-control capabilities that transfer across injection interfaces.

b) Training and inference efficiency: Training resources and inference efficiency, including accelerated generation with Trajectory LoRA, are reported in Sec. 4.4 of the supplementary material (Table S4).

## E. Ablation Studies

Our ablations test how each design contributes to camera control. Following the noise-state analysis of IRR and V-Cache, we examine how FLF and DSG preserve generation quality during trajectory guidance. We then assess the effectiveness of the selected trajectory-sensitive blocks, how efficiently these blocks learn from limited scenes, and how the two training stages prepare the adapter for V-Cache-guided denoising.

![](images/7282e715e2cf84ef7fc2a25b6438c6ab3be660a26cd9a8552e41b455c6f20d1c.jpg)  
Fig. 5. Qualitative comparison of camera-controlled scene generation. Our LongCat and H3 models recover coherent object shapes and material detail from incomplete warped references, including thin structures and partially revealed objects. Compared with the baselines, they produce fewer rendering remnant and more continuous geometry, preserving scene appearance as the camera reveals new content.

![](images/c5f4b9655dc2ae531ca6a48593697ffbecd897b09447908916fdef7cd50f4dba.jpg)  
Fig. 6. Qualitative transfer of Trajectory LoRA across camera-control methods. Without additional fine-tuning, the same trained LongCat adapter improves surface continuity and object detail in WAH [18] and GWF [16], demonstrating transferable trajectory control.

a) Inference-time guidance: IRR injects the target trajectory through the warped reference; removing this injection leaves the model without camera guidance. FLF confines reference fusion to channels aligned with the target motion, allowing the others to retain the model’s appearance prediction. Removing this selection applies fusion to all channels and disrupts appearance. DSG corrects warped-reference distortions using the model’s original prediction. The guided and unguided velocities differ substantially in direction, making standard CFG unstable. Table IV shows that removing FLF or DSG reduces static and dynamic generation quality, while replacing DSG with its CFG form causes further degradation. The complete framework performs best on all four quality metrics, combining appearance-preserving fusion with correction of guidance artifacts. Qualitative component comparisons are provided in the supplementary material.

## TABLE IV

QUANTITATIVE ABLATION OF INFERENCE-TIME GUIDANCE. IRR SUPPLIES   
THE TRAJECTORY SIGNAL; REMOVING ITS REFERENCE INJECTION LEAVES THE MODEL WITHOUT CAMERA GUIDANCE. WITH IRR FIXED, WE REMOVE FLF, DSG, OR BOTH, OR REPLACE DSG WITH ITS CFG FORM. BOTH COMPONENTS IMPROVE GENERATION QUALITY, AND THEIR   
COMBINATION PERFORMS BEST ON STATIC AND DYNAMIC SCENES. BOLD AND UNDERLINING MARK THE BEST AND SECOND-BEST RESULTS.

<table><tr><td>Method</td><td>FID↓ CLIP↑</td><td>FVD↓</td><td>CLIP-V↑</td></tr><tr><td>w/o DSG</td><td>109.43</td><td>0.943</td><td>95.69 0.937</td></tr><tr><td>w/o FLF</td><td>112.69</td><td>0.945</td><td>99.79 0.932</td></tr><tr><td>w/o DSG and FLF</td><td>113.12</td><td>0.943</td><td>103.17 0.931</td></tr><tr><td>DSG (CFG form)</td><td>120.91</td><td>0.936 109.10</td><td>0.919</td></tr><tr><td>Ours (Training-Free)</td><td>96.08</td><td>0.948</td><td>93.17 0.938</td></tr></table>

b) Trajectory-sensitive block selection: We use Shapley analysis [78] to measure how strongly each self-attention block responds to trajectory supervision. A block’s score averages its marginal contribution to the full probe LoRA’s velocity response across different combinations of active adapters. After brief probe LoRA training on each backbone, Fig. 7

TABLE V

QUANTITATIVE COMPARISON OF LORA ADAPTATION WITH DIFFERENT BLOCK SELECTIONS. BLOCKS ARE RANKED BY SHAPLEY SENSITIVITY; PARENTHESES GIVE THE NUMBER ADAPTED. RANDOM USES HALF OF THE BLOCKS, WITH METRICS AVERAGED OVER FIVE TRAINING RUNS; LOW   
RESPONSE USES THE LEAST SENSITIVE HALF. OTHER TRAINING SETTINGS FOLLOW THE DEFAULT CONFIGURATION. SELECTING ROUGHLY THE MOST   
SENSITIVE THREE QUARTERS MATCHES FULL-BLOCK PERFORMANCE AND IMPROVES SEVERAL RECONSTRUCTION AND CAMERA METRICS. BOLD AND UNDERLINING MARK THE BEST AND SECOND-BEST RESULTS WITHIN EACH BACKBONE.

LongCat
<table><tr><td>Selection</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>ATE↓</td><td>RPE-T↓</td><td>RPE-R↓</td></tr><tr><td>All blocks (48)</td><td>18.98</td><td>0.6140</td><td>0.2847</td><td>0.01045</td><td>0.2904</td><td>0.1693</td></tr><tr><td>Selected (36)</td><td>19.03</td><td>0.6176</td><td>0.2822</td><td>0.01089</td><td>0.2806</td><td>0.1678</td></tr><tr><td>Selected (24)</td><td>18.91</td><td>0.6106</td><td>0.2866</td><td>0.01163</td><td>0.3311</td><td>0.1719</td></tr><tr><td>Selected (12)</td><td>18.96</td><td>0.6138</td><td>0.2845</td><td>0.01188</td><td>0.3127</td><td>0.1696</td></tr><tr><td>Random (24)</td><td>18.29</td><td>0.6041</td><td>0.2949</td><td>0.01241</td><td>0.3310</td><td>0.1753</td></tr><tr><td>Low response (24)</td><td>17.12</td><td>0.5778</td><td>0.3336</td><td>0.01832</td><td>0.4107</td><td>0.1924</td></tr></table>

H3
<table><tr><td>Selection</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>ATE↓</td><td>RPE-T↓</td><td>RPE-R↓</td></tr><tr><td>All blocks (50)</td><td>20.20</td><td>0.6753</td><td>0.2783</td><td>0.00740</td><td>0.1742</td><td>0.1015</td></tr><tr><td>Selected (38)</td><td>20.15</td><td>0.6741</td><td>0.2775</td><td>0.00726</td><td>0.1732</td><td>0.1003</td></tr><tr><td>Selected (25)</td><td>20.01</td><td>0.6725</td><td>0.2800</td><td>0.00757</td><td>0.1821</td><td>0.1101</td></tr><tr><td>Selected (13)</td><td>19.97</td><td>0.6747</td><td>0.2810</td><td>0.00751</td><td>0.1887</td><td>0.1151</td></tr><tr><td>Random (25)</td><td>19.44</td><td>0.6483</td><td>0.2967</td><td>0.00875</td><td>0.2256</td><td>0.1367</td></tr><tr><td>Low response (25)</td><td>17.27</td><td>0.5715</td><td>0.3603</td><td>0.02339</td><td>0.4081</td><td>0.2529</td></tr></table>

![](images/cadc1eafe69d05838549f1b97cba600d06790bf6db5b718fe67abedac290d72e.jpg)

![](images/1bd955c040f33575f9bee5400eeada5d8c1979586e250d665ae0ba0bd0a36eb1.jpg)  
Selected / Full Random Low response (e) H3 appearance (f) H3 camera controlFig. 7. Shapley analysis of block sensitivity to trajectory supervision. After 0.350 0.025<sub>approximately 400 steps of probe LoRA training, contributions are estimated</sub> from 25 trajectory pairs per backbone. Sensitive blocks are dispersed through P<sup>S</sup> Ethe middle and later layers of LongCat and show a more regular concentration 0.300<sup>P</sup> 0.015Ain the later layers of H3. This variation identifies the blocks to prioritize for adaptation.

EFFECT OF TRAINING-SCENE DIVERSITY ON CAMERA-CONTROLLED GENERATION. THE TOTAL NUMBER OF TRAINING PAIRS IS FIXED. BOTH BACKBONES ALREADY PRODUCE GOOD RESULTS AFTER SINGLE-SCENE TRAINING; COVERING TWO TO FOUR SCENES CAPTURES MOST GAINS

Retained blocks (%)   Retained blocks (%)<sup>reveals</sup> <sup>clear</sup> <sup>differences</sup> <sup>in</sup> <sup>sensitivity.</sup> <sup>LongCat’s</sup> <sup>responsive</sup> Random Low responseblocks are dispersed through the middle and later layers, while

TABLE VI  
LongCat  
FROM ADDED DIVERSITY. FURTHER EXPANSION BRINGS LIMITED BENEFIT, SHOWING THAT THE ADAPTER CAN LEARN RELIABLE CAMERA CONTROL FROM FEW TRAINING SCENES. BOLD AND UNDERLINING MARK THE BEST AND SECOND-BEST RESULTS WITHIN EACH BACKBONE.  
![](images/674f708047c78098f9299e8ba96df3dcc5ed07696cfa5ec2eee772526a4803bf.jpg)

<table><tr><td>Scenes</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>ATE↓</td><td>RPE-T↓</td><td>RPE-R↓</td></tr><tr><td colspan="7">LongCat</td></tr><tr><td>1</td><td>18.96</td><td>0.6108</td><td>0.2838</td><td>0.01087</td><td>0.2723</td><td>0.1886</td></tr><tr><td>2</td><td>19.00</td><td>0.6152</td><td>0.2851</td><td>0.01130</td><td>0.2834</td><td>0.1766</td></tr><tr><td>4</td><td>19.02</td><td>0.6167</td><td>0.2828</td><td>0.01048</td><td>0.2861</td><td>0.1638</td></tr><tr><td>8</td><td>19.03</td><td>0.6176</td><td>0.2822</td><td>0.01089</td><td>0.2806</td><td>0.1678</td></tr><tr><td colspan="7">H3</td></tr><tr><td>1</td><td>20.15</td><td>0.6741</td><td>0.2775</td><td>0.00726</td><td>0.1732</td><td>0.1003</td></tr><tr><td>2</td><td>20.06</td><td>0.6740</td><td>0.2817</td><td>0.00703</td><td>0.1730</td><td>0.1017</td></tr><tr><td>4</td><td>19.86</td><td>0.6625</td><td>0.2820</td><td>0.00816</td><td>0.1833</td><td>0.1018</td></tr><tr><td>8</td><td>19.91</td><td>0.6662</td><td>0.2761</td><td>0.00720</td><td>0.1801</td><td>0.1008</td></tr></table>

![](images/06f4d3d57413d2bd59b0e44723f3e000779f19eccb589c5254b02e4060d0de70.jpg)

H3  
![](images/f907f7bda6ae9886565b1747ca066d7d40091462b8432ddc890b5ec6c5853631.jpg)  
(f) H3 camera control<sub>1</sub> <sub>scene</sub> <sub>/</sub> <sub>50</sub> <sub>blocks</sub> <sub>(Full)</sub>

![](images/1a6830dd4ebf7fdbeec0a6d5933d46f73676395344c2ffce884355edd84e0800.jpg)  
Fig. 8. Effect of training updates on generation quality and camera control. LongCat trains more stably with eight scenes and reaches its best perceptual quality near 1,200 updates; longer training shows signs of overfitting. H3 learns reliably from one scene. Removing LoRA from its less sensitive blocks accelerates convergence and improves perceptual quality compared with full block adaptation. Additional updates yield diminishing gains.

H3 shows a more regular concentration in later layers.

To test whether this sensitivity guides effective adaptation, we train different proportions of the highest-scoring blocks and compare them with full-block adaptation. Controls use the least responsive half or a random half, with random-selection results averaged over five independent training runs. All LoRA ablations use the same 100 test cases: five random camera trajectories from each of 20 randomly selected DL3DV scenes, disjoint from all training scenes. Other training settings follow the default configuration.

Adapting roughly the most sensitive three quarters of the blocks matches or improves full-block performance (Table V). LongCat’s 36 selected blocks improve reconstruction and local camera accuracy, while H3’s 38 selected blocks retain comparable reconstruction and reduce camera errors. These gains support leaving low-sensitivity blocks unadapted to preserve pretrained texture and appearance priors. The selected H3 blocks also reach good performance sooner than full-block adaptation (Fig. 8). At the same block count, selecting the sensitive half improves reconstruction over random selection on both backbones, while low-response selection performs worst. These comparisons validate our block-selection strategy.

TABLE VII  
QUANTITATIVE ABLATION OF THE TWO-STAGE TRAINING STRATEGY AND TRAJECTORY LORA. STANDARD FM TRAINS ON INDEPENDENTLY NOISED TARGETS; ONLY WARM-UP AND W/O WARM-UP ISOLATE ADAPTATION TO RECORDED STATES AND TRAINING ACROSS ADDITIONAL NOISE LEVELS. FULL COMBINES BOTH STAGES AND IMPROVES RECONSTRUCTION OVER STANDARD FM ON BOTH BACKBONES, WITH THE BEST CAMERA ACCURACY ON H3. REMOVING LORA GIVES THE POOREST RECONSTRUCTION.  
LongCat
<table><tr><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>ATE↓</td><td>RPE-T↓</td><td>RPE-R↓</td></tr><tr><td>Standard FM</td><td>18.55</td><td>0.5477</td><td>0.3407</td><td>0.01012</td><td>0.2590</td><td>0.1692</td></tr><tr><td>Only warm-up</td><td>18.92</td><td>0.6062</td><td>0.2834</td><td>0.01476</td><td>0.2985</td><td>0.1879</td></tr><tr><td>w/o warm-up</td><td>18.90</td><td>0.6084</td><td>0.2857</td><td>0.01276</td><td>0.2931</td><td>0.1801</td></tr><tr><td>w/o LoRA</td><td>18.07</td><td>0.5271</td><td>0.3640</td><td>0.01683</td><td>0.2986</td><td>0.1964</td></tr><tr><td>Full</td><td>19.03</td><td>0.6176</td><td>0.2822</td><td>0.01089</td><td>0.2806</td><td>0.1678</td></tr></table>

H3
<table><tr><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>ATE↓</td><td>RPE-T↓</td><td>RPE-R↓</td></tr><tr><td>Standard FM</td><td>19.55</td><td>0.6219</td><td>0.2928</td><td>0.00767</td><td>0.1878</td><td>0.1215</td></tr><tr><td>Only warm-up</td><td>19.87</td><td>0.6416</td><td>0.2861</td><td>0.00898</td><td>0.1904</td><td>0.1182</td></tr><tr><td>w/o warm-up</td><td>20.35</td><td>0.6558</td><td>0.2771</td><td>0.00806</td><td>0.1894</td><td>0.1106</td></tr><tr><td>w/o LoRA</td><td>14.85</td><td>0.4840</td><td>0.4220</td><td>0.01064</td><td>0.2458</td><td>0.1494</td></tr><tr><td>Full</td><td>20.15</td><td>0.6741</td><td>0.2775</td><td>0.00726</td><td>0.1732</td><td>0.1003</td></tr></table>

![](images/f96971cc46673da1612eca5eefe4f6f15efe9ea82866272b6f9a069b1a16e80f.jpg)  
Fig. 9. Qualitative comparison of Trajectory LoRA training strategies. For LongCat (a) and H3 (b), two examples per backbone place the input and warpe reference beside No LoRA, Standard FM, and the full method. The full strategy reduces fragmented geometry and residual rendering artifacts, producing more continuous surfaces and finer scene detail than No LoRA and Standard FM. Red boxes and enlarged crops compare corresponding regions.

c) Training-scene diversity: With adaptation focused on trajectory-sensitive blocks, we examine how much scene diversity is needed for reliable camera control. We draw a fixed total of 50 trajectory–video pairs from one, two, four, or eight DL3DV scenes, keeping other training settings unchanged. Both backbones already achieve good reconstruction and control with one scene (Table VI). Added diversity improves LongCat’s reconstruction and rotation accuracy, while H3 gains mainly in translation accuracy. Most gains are reached with two to four scenes; extending coverage to eight brings limited further benefit. These results show that the training strategy uses scene supervision efficiently, enabling reliable trajectory control and high-quality generation on unseen scenes from few examples.

d) Training updates: We next vary training updates while fixing each backbone’s training data and other settings. Figure 8 tracks the eight-scene LongCat and single-scene H3 configurations, with additional scene-count comparisons and full-block adaptation on H3. LongCat trains more stably with eight scenes and reaches its best perceptual quality near 1,200 updates; longer training brings little benefit and eventually shows signs of overfitting. H3 learns reliably from one scene. Removing LoRA from its less sensitive blocks accelerates convergence and improves perceptual quality over full-block adaptation. Later updates refine camera accuracy as perceptual quality stabilizes, with diminishing gains. Thus, the sensitive blocks learn to use trajectory guidance within relatively few updates.

e) Training strategy: We compare Standard FM, Only warm-up, and w/o warm-up to isolate the effects of the training strategies, using 1,200 updates for LongCat and 1,500 for H3. The adapter must learn to denoise states modified by reference insertion and cached correction. Anchor warm-up first adapts it to intermediate states recorded during guided sampling; the second stage uses the warmed-up model and these states to construct guided inputs at additional noise levels. Table VII and Fig. 9 compare these variants with Full and w/o LoRA under fixed inference settings for each backbone. Standard FM uses independently noised targets. Only warm-up trains on recorded states without extending the noise range, while w/o warm-up skips initial adaptation but still constructs guided inputs from intermediate states at additional noise levels. The full strategy gives the best overall results across quantitative and qualitative comparisons, improving reconstruction over Standard FM on both backbones and achieving the best camera accuracy on H3. Removing LoRA gives the poorest reconstruction. Although Standard FM has low errors on some camera metrics, its training inputs differ substantially from inference-time guided states. This gap prevents reliable use of trajectory guidance, leaving holes, cracks, and rendering artifacts. Training on states that reproduce guided denoising restores continuous surfaces and finer detail.

## V. CONCLUSION

We presented WorldForge V2 for accurate camera control while preserving the generative capabilities of pretrained video models. Our training-free guidance uses warped references to iteratively refine generation along the requested camera path, with motion-aware channel selection and self-correction preserving scene appearance and object motion. Building on this guidance, v-cache reuses denoising predictions to approximate the refinement correction, reducing sampling cost. Trajectory LoRA extends motion localization from latent channels to network blocks and combines selective adaptation with two-stage training, enabling pretrained video models to learn reliable trajectory control from very few trajectory–video pairs. Quantitative and qualitative evaluations across multiple datasets demonstrate state-of-the-art generation quality and trajectory accuracy in camera-controlled video generation. Future work will explore long-horizon, real-time streaming generation with camera control to support the development of interactive world models.

## REFERENCES

[1] A. Blattmann et al., “Stable video diffusion: Scaling latent video diffusion models to large datasets,” arXiv preprint arXiv:2311.15127, 2023.

[2] Team Wan et al., “Wan: Open and advanced large-scale video generative models,” arXiv preprint arXiv:2503.20314, 2025.

[3] Z. Yang et al., “CogVideoX: Text-to-video diffusion models with an expert transformer,” in International Conference on Learning Representations (ICLR), 2025.

[4] W. Yu et al., “ViewCrafter: Taming video diffusion models for highfidelity novel view synthesis,” IEEE Transactions on Pattern Analysis and Machine Intelligence, early access, pp. 1–18, 2025, doi: 10.1109/TPAMI.2025.3613256.

[5] M. Yu, W. Hu, J. Xing, and Y. Shan, “TrajectoryCrafter: Redirecting camera trajectory for monocular videos via diffusion models,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025, pp. 100–111.

[6] M. You, Z. Zhu, H. Liu, and J. Hou, “NVS-Solver: Video diffusion model as zero-shot novel view synthesizer,” in International Conference on Learning Representations (ICLR), 2025.

[7] H. Wang, F. Liu, J. Chi, and Y. Duan, “VideoScene: Distilling video diffusion model to generate 3D scenes in one step,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 16 475–16 485.

[8] T. Liu et al., “Free4D: Tuning-free 4D scene generation with spatialtemporal consistency,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025, pp. 25 571–25 582.

[9] J. Bruce et al., “Genie: Generative interactive environments,” in Proceedings of the 41st International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 235, 2024, pp. 4603–4623.

[10] A. Bar, G. Zhou, D. Tran, T. Darrell, and Y. LeCun, “Navigation world models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 15 791–15 801.

[11] Y. Cao et al., “Reconstructing 4D spatial intelligence: A survey,” arXiv preprint arXiv:2507.21045, 2025.

[12] Z. Wang et al., “MotionCtrl: A unified and flexible motion controller for video generation,” in ACM SIGGRAPH 2024 Conference Papers, 2024, pp. 1–11.

[13] H. He et al., “CameraCtrl: Enabling camera control for video diffusion models,” in International Conference on Learning Representations (ICLR), 2025.

[14] J. Bai et al., “ReCamMaster: Camera-controlled generative rendering from a single video,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025, pp. 14 834–14 844.

[15] Z. Xiao et al., “Trajectory attention for fine-grained video motion control,” in International Conference on Learning Representations (ICLR), 2025.

[16] R. Burgert et al., “Go-with-the-Flow: Motion-controllable video diffusion models using real-time warped noise,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 13–23.

[17] K. Liu, L. Shao, and S. Lu, “Novel view extrapolation with video diffusion priors,” arXiv preprint arXiv:2411.14208, 2024.

[18] Y. Wang and T. He, “Warp-as-History: Generalizable cameracontrolled video generation from one training video,” arXiv preprint arXiv:2605.15182, 2026.

[19] J. Ho and T. Salimans, “Classifier-free diffusion guidance,” in NeurIPS Workshop on Deep Generative Models and Downstream Applications, 2021.

[20] Z. Xiao, Y. Zhou, S. Yang, and X. Pan, “Video diffusion models are training-free motion interpreter and controller,” in Advances in Neural Information Processing Systems, vol. 37, 2024, pp. 76 115–76 138.

[21] C. Song, Y. Yang, T. Zhao, R. Li, and C. Zhang, “Taming video models for 3D and 4D generation via zero-shot camera control,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026, pp. 40 352–40 363.

[22] W. Kong et al., “HunyuanVideo: A systematic framework for large video generative models,” arXiv preprint arXiv:2412.03603, 2024.

[23] S. Bahmani et al., “AC3D: Analyzing and improving 3D camera control in video diffusion transformers,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 22 875–22 889.

[24] S. Bahmani et al., “VD3D: Taming large video diffusion transformers for 3D camera control,” in International Conference on Learning Representations (ICLR), 2025.

[25] B. Van Hoorick et al., “Generative camera dolly: Extreme monocular dynamic novel view synthesis,” in European Conference on Computer Vision (ECCV), 2024, pp. 313–331.

[26] Robbyant Team et al., “Advancing open-source world models,” arXiv preprint arXiv:2601.20540, 2026.

[27] InSpatio Team et al., “INSPATIO-WORLD: A real-time 4D world simulator via spatiotemporal autoregressive modeling,” arXiv preprint arXiv:2604.07209, 2026.

[28] Team HY-World et al., “HY-World 2.0: A multi-modal world model for reconstructing, generating, and simulating 3D worlds,” arXiv preprint arXiv:2604.14268, 2026.

[29] W. Sun et al., “WorldPlay: Towards long-term geometric consistency for real-time interactive world modeling,” in Proceedings of the 43rd International Conference on Machine Learning (ICML), ser. Proceedings of Machine Learning Research, vol. 306, 2026, pp. 117 501–117 523.

[30] Z. Wang et al., “Matrix-Game 3.0: Real-time and streaming interactive world model with long-horizon memory,” arXiv preprint arXiv:2604.08995, 2026.

[31] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “NeRF: Representing scenes as neural radiance fields for view synthesis,” in European Conference on Computer Vision (ECCV), 2020, pp. 405–421.

[32] B. Kerbl, G. Kopanas, T. Leimkuehler, and G. Drettakis, “3D Gaussian splatting for real-time radiance field rendering,” ACM Transactions on Graphics (TOG), vol. 42, no. 4, Art. no. 139, pp. 1–14, 2023.

[33] Y. Yao, Z. Luo, S. Li, T. Fang, and L. Quan, “MVSNet: Depth inference for unstructured multi-view stereo,” in European Conference on Computer Vision (ECCV), 2018, pp. 767–783.

[34] Z. Yu, A. Chen, B. Huang, T. Sattler, and A. Geiger, “Mip-Splatting: Alias-free 3D Gaussian splatting,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 19 447–19 456.

[35] T. Muller, A. Evans, C. Schied, and A. Keller, “Instant neural graphics¨ primitives with a multiresolution hash encoding,” ACM Transactions on Graphics (TOG), vol. 41, no. 4, Art. no. 102, pp. 1–15, 2022.

[36] C. Song, S. Wang, J. Wei, and Y. Zhao, “FewarNet: An efficient few-shot view synthesis network based on trend regularization,” IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), vol. 34, no. 10, pp. 9264–9280, 2024.

[37] C. Song, S. Wang, J. Wei, Y. Zhao, and R. Zhang, “Wide-baseline view synthesis for light-field display based on plane-depth-fused sweep volume,” Displays, vol. 79, Art. no. 102503, 2023.

[38] G. Wang, P. Wang, Z. Chen, W. Wang, C. C. Loy, and Z. Liu, “PERF: Panoramic neural radiance field from a single panorama,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 10, pp. 6905–6918, 2024, doi: 10.1109/TPAMI.2024.3387307.

[39] B. Poole, A. Jain, J. T. Barron, and B. Mildenhall, “DreamFusion: Textto-3D using 2D diffusion,” in International Conference on Learning Representations (ICLR), 2023.

[40] M. Wei, J. Zhou, J. Sun, and X. Zhang, “Adversarial score distillation: When score distillation meets GAN,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 8131–8141.

[41] J. Xiang et al., “Structured 3D latents for scalable and versatile 3D generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 21 469–21 480.

[42] M.-S. Kwak, D. Ahn, I. H. Kim, J.-H. Kim, and S. Kim, “Geometryaware score distillation via 3D consistent noising and gradient consistency modeling,” arXiv preprint arXiv:2406.16695, 2024.

[43] Y. Yang et al., “DreamComposer++: Empowering diffusion models with multi-view conditions for 3D content generation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, early access, pp. 1–18, 2025, doi: 10.1109/TPAMI.2025.3568190.

[44] J. Bai et al., “SynCamMaster: Synchronizing multi-camera video generation from diverse viewpoints,” in International Conference on Learning Representations (ICLR), 2025.

[45] Y. Ma et al., “EasyCreator: Empowering 4D creation through video inpainting,” in International Conference on Learning Representations (ICLR), 2026.

[46] F. Tian, T. Ding, J. Luo, H. Min, and R. Vidal, “Voyaging into perpetual dynamic scenes from a single view,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025, pp. 7698– 7708.

[47] Z. Zhou, J. An, and J. Luo, “Latent-Reframe: Enabling camera control for video diffusion models without training,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025, pp. 12 779–12 789.

[48] C. Song, Y. Yang, and C. Zhang, “World in world: Explore the world with world models,” arXiv preprint arXiv:2609.11548, 2026.

[49] B. Ma et al., “You see it, you got it: Learning 3D creation on posefree videos at scale,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 2016– 2029.

[50] T. Huang et al., “Voyager: Long-range and world-consistent video diffusion for explorable 3D scene generation,” ACM Transactions on Graphics (TOG), vol. 44, no. 6, Art. no. 245, pp. 1–15, 2025.

[51] X. Fu et al., “3DTrajMaster: Mastering 3D trajectory for multi-entity motion in video generation,” in International Conference on Learning Representations (ICLR), 2025.

[52] T. Karras, M. Aittala, T. Kynka¨anniemi, J. Lehtinen, T. Aila, and¨ S. Laine, “Guiding a diffusion model with a bad version of itself,” Advances in Neural Information Processing Systems (NeurIPS), vol. 37, pp. 52 996–53 021, 2024.

[53] J. Hyung, K. Kim, S. Hong, M.-J. Kim, and J. Choo, “Spatiotemporal skip guidance for enhanced video diffusion sampling,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 11 006–11 015.

[54] S. Sadat, O. Hilliges, and R. M. Weber, “Eliminating oversaturation and artifacts of high guidance scales in diffusion models,” in International Conference on Learning Representations (ICLR), 2025.

[55] G. Li, Y. Yang, C. Song, X. Liu, and C. Zhang, “FlowDirector: Trainingfree flow steering for precise text-to-video editing,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026, pp. 7805–7815.

[56] Y. Xu, M. Deng, X. Cheng, Y. Tian, Z. Liu, and T. Jaakkola, “Restart sampling for improving generative processes,” Advances in Neural Information Processing Systems (NeurIPS), vol. 36, pp. 76 806–76 838, 2023.

[57] A. Lugmayr, M. Danelljan, A. Romero, F. Yu, R. Timofte, and L. Van Gool, “RePaint: Inpainting using denoising diffusion probabilistic models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 11 461–11 471.

[58] L. Bai et al., “Zigzag diffusion sampling: Diffusion models can selfimprove via self-reflection,” in International Conference on Learning Representations (ICLR), 2025.

[59] X. Cai et al., “Dust to tower: Prior-driven coarse-to-fine photo-realistic scene reconstruction from sparse uncalibrated images,” IEEE Transactions on Visualization and Computer Graphics (TVCG), vol. 32, no. 7, pp. 6129–6143, 2026.

[60] H. Wang, Y. Liu, Z. Liu, W. Wang, Z. Dong, and B. Yang, “VistaDream: Sampling multiview consistent images for single-view scene reconstruction,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025, pp. 26 772–26 782.

[61] L. Zhang, A. Rao, and M. Agrawala, “Adding conditional control to text-to-image diffusion models,” in Proceedings of the IEEE/CVF

International Conference on Computer Vision (ICCV), 2023, pp. 3836– 3847.

[62] C. Mou et al., “T2I-Adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models,” in Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), vol. 38, no. 5, 2024, pp. 4296–4304.

[63] E. Hu et al., “LoRA: Low-rank adaptation of large language models,” in International Conference on Learning Representations (ICLR), 2022.

[64] Y. Ren et al., “Customize-A-Video: One-shot motion customization of text-to-video diffusion models,” in European Conference on Computer Vision (ECCV), 2024, pp. 332–349.

[65] Q. Wu, K. Vuong, M. Jeon, S. Narasimhan, and D. Ramanan, “Frame-Crafter: Novel view synthesis as video completion,” in European Conference on Computer Vision (ECCV), 2026, pp. 73–91.

[66] Y. Zhang, C. Cao, C. Yu, and J. Zhu, “LiON-LoRA: Rethinking LoRA fusion to unify controllable spatial and temporal generation for video diffusion,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025, pp. 14 569–14 579.

[67] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in Neural Information Processing Systems (NeurIPS), vol. 33, pp. 6840–6851, 2020.

[68] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” in International Conference on Learning Representations (ICLR), 2023.

[69] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, “Score-based generative modeling through stochastic differential equations,” in International Conference on Learning Representations (ICLR), 2021.

[70] F.-A. Croitoru, V. Hondru, R. T. Ionescu, and M. Shah, “Diffusion models in vision: A survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 9, pp. 10 850–10 869, 2023, doi: 10.1109/TPAMI.2023.3261988.

[71] Meituan LongCat Team et al., “LongCat-Video technical report,” arXiv preprint arXiv:2510.22200, 2025.

[72] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and D. Novotny, “VGGT: Visual geometry grounded transformer,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 5294–5306.

[73] L. Piccinelli et al., “UniDepth: Universal monocular metric depth estimation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 10 106–10 116.

[74] G. Farneback, “Two-frame motion estimation based on polynomial¨ expansion,” in Scandinavian Conference on Image Analysis (SCIA), 2003, pp. 363–370.

[75] Z. Teed and J. Deng, “RAFT: Recurrent all-pairs field transforms for optical flow,” in European Conference on Computer Vision (ECCV), 2020, pp. 402–419.

[76] F. Liu et al., “Timestep embedding tells: It’s time to cache for video diffusion model,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 7353–7363.

[77] L. Ling et al., “DL3DV-10K: A large-scale scene dataset for deep learning-based 3D vision,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 22 160– 22 169.

[78] L. S. Shapley, “A value for n-person games,” in Contributions to the Theory of Games II, ser. Annals of Mathematics Studies, H. W. Kuhn and A. W. Tucker, Eds. Princeton, NJ: Princeton University Press, 1953, vol. 28, pp. 307–317.

[79] B. Mildenhall et al., “Local light field fusion: Practical view synthesis with prescriptive sampling guidelines,” ACM Transactions on Graphics (TOG), vol. 38, no. 4, Art. no. 29, pp. 1–14, 2019.

[80] A. Knapitsch, J. Park, Q.-Y. Zhou, and V. Koltun, “Tanks and temples: Benchmarking large-scale scene reconstruction,” ACM Transactions on Graphics (TOG), vol. 36, no. 4, Art. no. 78, pp. 1–13, 2017.

[81] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman, “Mip-NeRF 360: Unbounded anti-aliased neural radiance fields,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 5470–5479.

[82] F. Perazzi, J. Pont-Tuset, B. McWilliams, L. Van Gool, M. Gross, and A. Sorkine-Hornung, “A benchmark dataset and evaluation methodology for video object segmentation,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 724– 732.

[83] H. Zhou, W. Yu, C. Feng, X. Zhou, Y. Tian, and L. Yuan, “UniWorld-View: Large-baseline view synthesis via video diffusion models,” arXiv preprint arXiv:2608.04701, 2026.

[84] G. Zhang, Y. Chen, X. Xiang, J. Huang, Z. Wang, and L. Jiang, “SymphoMotion: Joint control of camera motion and object dynamics for coherent video generation,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026, pp. 11 127– 11 137.

[85] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “GANs trained by a two time-scale update rule converge to a local Nash equilibrium,” in Advances in Neural Information Processing Systems, vol. 30, 2017, pp. 6626–6637.

[86] A. Radford et al., “Learning transferable visual models from natural language supervision,” in Proceedings of the International Conference on Machine Learning (ICML), ser. Proceedings of Machine Learning Research, vol. 139, 2021, pp. 8748–8763.

[87] T. Unterthiner, S. van Steenkiste, K. Kurach, R. Marinier, M. Michalski, and S. Gelly, “Towards accurate generative models of video: A new metric & challenges,” arXiv preprint arXiv:1812.01717, 2018.

[88] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 586–595.

[89] L. Pan, D. Barath, M. Pollefeys, and J. L. Sch´ onberger, “Global¨ structure-from-motion revisited,” in European Conference on Computer Vision (ECCV), 2024, pp. 58–77.

[90] Y. Wang et al., “π<sup>3</sup>: Permutation-equivariant visual geometry learning,” in International Conference on Learning Representations (ICLR), 2026.

# WorldForge V2: Eliciting Camera-Trajectory Alignment in Video Diffusion Models Supplementary Material

## Contents

1 Equivalence between Diffusion and Flow Models . . . . 16   
2 Evaluation Protocol . . . . . 16   
2.1 Data and Comparison Protocol 16   
2.2 Static Scene Evaluation 17   
2.3 Dynamic Scene Evaluation 17   
2.4 Reconstruction and Completion 17   
2.5 Camera-Trajectory Accuracy . 17   
3 Implementation Details . . . 18   
3.1 FLF Motion Estimation and Channel Scoring . . . 18   
3.2 Hyperparameter Settings 18   
3.3 V-Cache . . 19   
3.4 Trajectory LoRA Adaptation . 19   
4 Additional Experiments . 20   
4.1 Component Behavior and Motion Selection . . 20   
4.2 Self-Corrective Guidance . 20   
4.3 Backbone and Geometry Compatibility 21   
4.4 Training Resources and Inference Efficiency 21   
4.5 Challenging Scenes and Control Limits . 21   
4.6 Wide-Range Control and Editing 22   
5 Additional Visual Comparisons 22

## 1. Equivalence between Diffusion and Flow Models

Our camera guidance uses the model’s estimate of the clean video latent. This estimate can be obtained from either a diffusion noise prediction or a Flow Matching velocity prediction. We derive their relationship by expressing the Flow Matching path [68] through the noise schedule of a diffusion model [69].

Let $\mathbf { x } _ { \mathrm { 0 } }$ be drawn from the data distribution $q _ { 0 } .$ . A diffusion model [67], [69] gradually adds Gaussian noise to the data, giving the conditional distribution

$$
q _ { t } ( \mathbf { x } _ { t } \mid \mathbf { x } _ { 0 } ) = { \mathcal { N } } ( \mathbf { x } _ { t } \mid \alpha _ { t } \mathbf { x } _ { 0 } , \sigma _ { t } ^ { 2 } \mathbf { I } ) .\tag{S1}
$$

Equivalently, a sample at time t can be written as [69]

$$
\begin{array} { r } { \mathbf { x } _ { t } = \alpha _ { t } \mathbf { x } _ { 0 } + \sigma _ { t } \mathbf { \epsilon } , \qquad \mathbf { \epsilon } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) . } \end{array}\tag{S2}
$$

The noise schedules $\alpha _ { t }$ and $\sigma _ { t }$ control the signal and noise levels: $\alpha _ { t }$ typically decreases as $\sigma _ { t }$ increases. Variancepreserving diffusion satisfies $\alpha _ { t } ^ { 2 } + \sigma _ { t } ^ { 2 } = 1 \ [ 6 7 ]$ , [69]. More generally, the Gaussian transitions in Eq. (S1) correspond to the forward stochastic differential equation (SDE)

$$
\mathrm { d } \mathbf { x } _ { t } = f ( t ) \mathbf { x } _ { t } \mathrm { d } t + g ( t ) \mathrm { d } \mathbf { w } _ { t } , \qquad \mathbf { x } _ { 0 } \sim q _ { 0 } ,\tag{S3}
$$

where $\mathbf { w } _ { t }$ is a standard Wiener process, and the drift and diffusion coefficients are determined by the noise schedules [69]:

$$
\begin{array} { l } { f ( t ) = \displaystyle \frac { \mathrm { d } \log \alpha _ { t } } { \mathrm { d } t } , } \\ { g ^ { 2 } ( t ) = \displaystyle \frac { \mathrm { d } \sigma _ { t } ^ { 2 } } { \mathrm { d } t } - 2 \frac { \mathrm { d } \log \alpha _ { t } } { \mathrm { d } t } \sigma _ { t } ^ { 2 } . } \end{array}\tag{S4}
$$

Generation reverses this process from noise to data. The associated probability-flow ordinary differential equation (ODE) follows the same marginal distributions as the SDE [69]. Denoting the marginal density of $\mathbf { x } _ { t }$ by $p _ { t } .$ , we have

$$
\frac { \mathrm { d } \mathbf { x } _ { t } } { \mathrm { d } t } = f ( t ) \mathbf { x } _ { t } - \frac { 1 } { 2 } g ^ { 2 } ( t ) \nabla _ { \mathbf { x } _ { t } } \log p _ { t } ( \mathbf { x } _ { t } ) .\tag{S5}
$$

The noise predictor $\epsilon _ { \theta }$ approximates the score as $\nabla _ { \mathbf x _ { t } }$ log $p _ { t } ( \mathbf { x } _ { t } ) \approx - \epsilon _ { \theta } ( \mathbf { x } _ { t } , t ) / \sigma _ { t }$ . Substituting it into Eq. (S5) gives the sampling ODE [69]:

$$
\frac { \mathrm { d } \mathbf { x } _ { t } } { \mathrm { d } t } = f ( t ) \mathbf { x } _ { t } + \frac { g ^ { 2 } ( t ) } { 2 \sigma _ { t } } \epsilon _ { \theta } ( \mathbf { x } _ { t } , t ) .\tag{S6}
$$

For Flow Matching [68], we consider the path obtained by linearly interpolating between a data sample and Gaussian noise:

$$
{ \bf x } _ { t } = ( 1 - t ) { \bf x } _ { 0 } + t \epsilon , \qquad \epsilon \sim \mathcal { N } ( { \bf 0 } , { \bf I } ) ,\tag{S7}
$$

where $t \in [ 0 , 1 ]$ . Comparing Eq. (S7) with Eq. (S2) gives $\alpha _ { t } = 1 - t$ and $\sigma _ { t } = t .$ Substitution into Eq. (S4) yields

$$
f _ { \mathrm { F M } } ( t ) = \frac { \mathrm { d } \log ( 1 - t ) } { \mathrm { d } t } = - \frac { 1 } { 1 - t } ,\tag{S8}
$$

$$
g _ { \mathrm { F M } } ^ { 2 } ( t ) = { \frac { \mathrm { d } ( t ^ { 2 } ) } { \mathrm { d } t } } + { \frac { 2 t ^ { 2 } } { 1 - t } } = { \frac { 2 t } { 1 - t } } .\tag{S9}
$$

With an ideal noise predictor, $\epsilon _ { \theta } ( \mathbf { x } _ { t } , t ) = \mathbb { E } [ \epsilon \mid \mathbf { x } _ { t } ]$ . Using the coefficients above in Eq. (S6), we obtain

$$
\begin{array} { r l } { \displaystyle \frac { \mathrm { d } \mathbf { x } _ { t } } { \mathrm { d } t } = \frac { \boldsymbol { \epsilon } _ { \theta } \left( \mathbf { x } _ { t } , t \right) - \mathbf { x } _ { t } } { 1 - t } } \\ { = \mathbb { E } [ \boldsymbol { \epsilon } - \mathbf { x } _ { 0 } \mid \mathbf { x } _ { t } ] , \quad } & { 0 < t < 1 . } \end{array}\tag{S10}
$$

The conditional path in Eq. (S7) has derivative $\epsilon - { \bf x } _ { 0 }$ whose conditional expectation gives the marginal Flow Matching velocity. Thus, the Flow Matching path considered here corresponds to the diffusion schedule $\alpha _ { t } = 1 - t , \sigma _ { t } = t \left[ 6 8 \right]$ For corresponding noise and velocity predictions, the clean latent used by our guidance is

$$
\begin{array} { r l } { \hat { \mathbf { x } } _ { 0 } = \frac { \mathbf { x } _ { t } - t \epsilon _ { \theta } ( \mathbf { x } _ { t } , t ) } { 1 - t } } \\ { = \mathbf { x } _ { t } - t \mathbf { v } _ { \theta } ( \mathbf { x } _ { t } , t ) , \quad } & { { } 0 < t < 1 . } \end{array}\tag{S11}
$$

## 2. Evaluation Protocol

Camera control requires the generated video to follow the target path while preserving the scene. We evaluate these two aspects through visual quality and trajectory accuracy on the training-free benchmarks, and use the rendered DL3DV targets to further examine reconstruction of observed content and completion of newly exposed regions.

## 2.1. Data and Comparison Protocol

The training-free static benchmark contains more than 70 source views from over 40 scenes, including LLFF, Tanks and Temples, Mip-NeRF 360, and real or generated images. The dynamic benchmark contains more than 50 trajectorycontrolled outputs from over 30 source videos, including DAVIS, cinematic sequences, and videos generated by video diffusion models.

The learned-control experiments use DL3DV [77], with disjoint training and test scenes. The main comparison contains 200 cases, with five random camera paths for each of 40 scenes. All LoRA ablations use the same 100 cases, comprising five random paths from each of 20 randomly selected scenes. Sequences contain 25–93 frames, with lengths adapted to each method’s supported input format. Across the benchmarks, methods within each comparison share the inputs, target trajectories, and evaluation cases.

## 2.2. Static Scene Evaluation

Frechet Inception Distance (FID) [85] measures visual quality´ by comparing the feature distributions of real and generated images. We resize images to $2 9 9 \times 2 9 9$ and extract 2048-dimensional pool3 features from an ImageNet-pretrained Inception-V3 model. The Frechet distance between the Gaus-´ sian distributions fitted to these features is

$$
\begin{array} { r l } & { \mathrm { F I D } = \Vert \mu _ { r } - \mu _ { g } \Vert _ { 2 } ^ { 2 } } \\ & { \qquad + \operatorname { T r } \Bigl ( \Sigma _ { r } + \Sigma _ { g } - 2 \bigl ( \Sigma _ { r } \Sigma _ { g } \bigr ) ^ { 1 / 2 } \Bigr ) , } \end{array}\tag{S12}
$$

where $\mu _ { r } , \mu _ { g }$ and $\Sigma _ { r } , \Sigma _ { g }$ are the means and covariances of the real and generated features, computed over the evaluation set. The reference distribution aggregates ground-truth images from the static scenes. For in-the-wild scenes without ground truth, we use manually curated reference sets.

CLIP similarity [86] complements FID by measuring semantic agreement between corresponding real and generated images. We use CLIP ViT-B/32, pretrained on 400 million image–text pairs, to extract L2-normalized 512-dimensional features. Their average cosine similarity over N image pairs is

$$
\mathrm { C L I P } _ { \mathrm { s i m } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \cos ( f _ { r , i } , f _ { g , i } ) ,\tag{S13}
$$

where $f _ { r , i }$ and $f _ { g , i }$ are the features of the i-th real and generated image pair.

## 2.3. Dynamic Scene Evaluation

For dynamic scenes, Frechet Video Distance (FVD) [87] com-´ pares real and generated videos using spatiotemporal features. We resize I3D inputs to 224 × 224 and extract a 1024- dimensional feature for each clip from the global average pooling layer of a Kinetics-pretrained I3D network. As with FID, we fit Gaussian distributions to the real and generated features and compute

$$
\begin{array} { r l } & { \mathrm { F V D } = \| \mu _ { r } - \mu _ { g } \| _ { 2 } ^ { 2 } } \\ & { \qquad + \operatorname { T r } \Bigl ( \Sigma _ { r } + \Sigma _ { g } - 2 ( \Sigma _ { r } \Sigma _ { g } ) ^ { 1 / 2 } \Bigr ) , } \end{array}\tag{S14}
$$

where the means and covariances are estimated from cliplevel I3D features over the evaluation set. Reference clips are extracted from DAVIS and cinematic sequences.

Video CLIP similarity $\mathrm { ( C L I P  – V _ { s i m } ) }$ measures frame-level semantic agreement. We average the CLIP similarity of corresponding frames within each video pair, then average over all M pairs:

$$
\mathrm { C L I P - V _ { s i m } } = \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \frac { 1 } { T _ { j } } \sum _ { t = 1 } ^ { T _ { j } } \cos ( f _ { r , j , t } , f _ { g , j , t } ) ,\tag{S15}
$$

where $T _ { j }$ is the number of evaluated frames in pair $j ,$ and $f _ { r , j , t } , f _ { g , j , t }$ are the corresponding CLIP features. Video evaluation uses 16 uniformly sampled frames, including the first and last frames.

## 2.4. Reconstruction and Completion

The rendered DL3DV targets allow us to compare generated content with the scene along each camera path. We use PSNR for pixel agreement, SSIM for structural agreement, and LPIPS [88] for perceptual similarity. Frames are evaluated on a common $6 4 0 \times 3 6 0$ image area, excluding the input frame.

To distinguish preservation of observed content from completion of newly exposed regions, we also report LPIPS for these regions separately. Let $D _ { i } ( p )$ be the spatial LPIPS error map for frame i and $M _ { i } ( p )$ its rendered visibility mask. The regional scores are

$$
\begin{array} { c } { { \mathrm { V i s } _ { i } = \displaystyle \frac { \sum _ { p } M _ { i } ( p ) D _ { i } ( p ) } { \sum _ { p } M _ { i } ( p ) } , } } \\ { { \mathrm { D i s o c c } _ { i } = \displaystyle \frac { \sum _ { p } [ 1 - M _ { i } ( p ) ] D _ { i } ( p ) } { \sum _ { p } [ 1 - M _ { i } ( p ) ] } . } } \end{array}\tag{S16}
$$

Vis-LPIPS evaluates the observed region, and Disocc-LPIPS evaluates the newly exposed region. Empty regions are omitted. We average scores over valid frames within each case, then over cases within each scene, and finally over scenes with equal weight.

## 2.5. Camera-Trajectory Accuracy

For the training-free benchmarks, we use evo to compute rootmean-square errors (RMSE). A global positive-scale similarity transform aligns the estimated trajectory to the target by correcting scale, rotation, and translation. We denote the aligned camera position and rotation at frame i by $\tilde { \mathbf { t } } _ { \mathrm { e s t } , i }$ and $\ddot { \mathbf { R } } _ { \mathrm { e s t } , i }$ Absolute Trajectory Error (ATE) measures global position agreement over the n camera poses:

$$
\begin{array} { l } { { \displaystyle \mathrm { A T E } _ { i } = \| { \bf t } _ { \mathrm { r e f } , i } - \tilde { { \bf t } } _ { \mathrm { e s t } , i } \| _ { 2 } } , } \\ { { \displaystyle \mathrm { A T E } = \sqrt { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathrm { A T E } _ { i } ^ { 2 } } } . } \end{array}\tag{S17}
$$

Local motion accuracy is measured by Relative Pose Error in translation (RPE-T). With an index gap of $\Delta \ : = \ : 1$ , the relative poses between adjacent frames are

$$
\begin{array} { r } { \Delta { \bf T } _ { \mathrm { r e f } , i } = { \bf T } _ { \mathrm { r e f } , i } ^ { - 1 } { \bf T } _ { \mathrm { r e f } , i + 1 } , } \\ { \Delta { \bf T } _ { \mathrm { e s t } , i } = \tilde { \bf T } _ { \mathrm { e s t } , i } ^ { - 1 } \tilde { \bf T } _ { \mathrm { e s t } , i + 1 } . } \end{array}\tag{S18}
$$

Denoting their translation components by $\Delta \mathbf { t } _ { \mathrm { r e f } , i }$ and $\Delta \mathbf { t } _ { \mathrm { e s t } , i }$ , we compute the per-step translation error and its RMSE as

$$
\begin{array} { l } { { \displaystyle { \mathrm { R P E  – T } } _ { i } = \| \Delta \mathbf { t } _ { \mathrm { r e f } , i } - \Delta \mathbf { t } _ { \mathrm { e s t } , i } \| _ { 2 } } , } \\ { { \displaystyle { \mathrm { R P E } } \mathrm { - } \mathrm { T } = \sqrt { \frac { 1 } { n - 1 } \sum _ { i = 1 } ^ { n - 1 } \mathrm { R P E } \mathrm { - } \mathrm { T } _ { i } ^ { 2 } } } . } \end{array}\tag{S19}
$$

Relative Pose Error in rotation (RPE-R) measures the corresponding error in orientation changes. The relative rotations between adjacent frames are

$$
\begin{array} { r } { \Delta \mathbf { R } _ { \mathrm { r e f } , i } = \mathbf { R } _ { \mathrm { r e f } , i } ^ { - 1 } \mathbf { R } _ { \mathrm { r e f } , i + 1 } , } \\ { \Delta \mathbf { R } _ { \mathrm { e s t } , i } = \tilde { \mathbf { R } } _ { \mathrm { e s t } , i } ^ { - 1 } \tilde { \mathbf { R } } _ { \mathrm { e s t } , i + 1 } . } \end{array}\tag{S20}
$$

We compute the angular error in degrees and take its RMSE over the trajectory:

$$
\begin{array} { r l } & { \mathrm { R P E - R } _ { i } = \operatorname { a r c c o s } \left( \frac { \mathrm { T r } ( \Delta \mathbf { R } _ { \mathrm { r e f } , i } ^ { \mathsf { T } } \Delta \mathbf { R } _ { \mathrm { e s t } , i } ) - 1 } { 2 } \right) \frac { 1 8 0 } { \pi } , } \\ & { \mathrm { R P E - R } = \sqrt { \displaystyle \frac { 1 } { n - 1 } \sum _ { i = 1 } ^ { n - 1 } \mathrm { R P E - R } _ { i } ^ { 2 } } . } \end{array}\tag{S21}
$$

For DL3DV, we also align recovered cameras to the target with a positive-scale similarity transform. To compare paths of different lengths, we normalize translation errors by the reference motion. ATE uses normalized position RMSE, RPE-T divides the sum of translation errors by the sum of reference translation magnitudes, and RPE-R averages angular errors over adjacent poses.

Let $\hat { \mathbf { c } } _ { i }$ and $\mathbf { c } _ { i }$ be the aligned estimated and target camera centers, and $\Delta \hat { \mathbf { T } } _ { i } , \ \Delta \mathbf { T } _ { i }$ their adjacent relative poses. With reference path length $\begin{array} { r } { L = \sum _ { i = 1 } ^ { n - 1 } \| \mathbf { c } _ { i + 1 } - \mathbf { c } _ { i } \| _ { 2 } } \end{array}$ , ATE is

$$
\mathrm { A T E } = \frac { 1 } { L } \sqrt { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \| \hat { \mathbf { c } } _ { i } - \mathbf { c } _ { i } \| _ { 2 } ^ { 2 } } .\tag{S22}
$$

Writing $\Delta \mathbf { t } _ { i }$ and $\Delta \mathbf { R } _ { i }$ for the relative translation and rotation, the relative pose errors are

$$
\mathrm { R P E - T } = \frac { \sum _ { i = 1 } ^ { n - 1 } \| \Delta \hat { \mathbf { t } } _ { i } - \Delta \mathbf { t } _ { i } \| _ { 2 } } { \sum _ { i = 1 } ^ { n - 1 } \| \Delta \mathbf { t } _ { i } \| _ { 2 } } ,\tag{S23}
$$

$$
\mathrm { R P E - R } = \frac { 1 } { n - 1 } \sum _ { i = 1 } ^ { n - 1 } \angle ( \Delta \mathbf { R } _ { i } ^ { \top } \Delta \hat { \mathbf { R } } _ { i } ) ,\tag{S24}
$$

with rotation angles in degrees. The input frame is excluded from alignment and scoring. We use GLOMAP [89] to recover poses for the main comparison and H3 studies, and Pi3X [90] for the LongCat training ablations. All methods within each comparison use the same pose estimator and aggregation.

## 3. Implementation Details

## 3.1. FLF Motion Estimation and Channel Scoring

Flow-Gated Latent Fusion (FLF) uses motion agreement to select the latent channels that receive trajectory guidance. At each denoising step i, we compare the motion in each predicted channel with that in the rendered reference and assign a score $S _ { i } ^ { c } .$ . Channels with high scores receive the reference fusion, while the others retain the model’s prediction to preserve appearance details.

For each channel c, we apply Farneback optical flow [74]¨ between adjacent frames of the predicted clean latent $\hat { \mathbf { x } } _ { 0 } ^ { ( i ) }$ and the trajectory latent $\mathbf { x } _ { \mathrm { t r a j } }$ . This gives predicted flow $\mathcal { F } _ { \mathrm { p r e d } } ^ { ( i , c ) }$ and reference flow $\mathcal { F } _ { \mathrm { r e f } } ^ { ( i , c ) }$ , with each flow vector describing horizontal and vertical displacement. For a latent video with $T _ { l }$ frames, the frame-pair index is $\tau = 1 , \dots , T _ { l } - 1$ . We compare the flows over valid locations in the rendered visibility mask, $\Omega ^ { ( i , c ) } = \{ ( x , y , \tau ) | \mathbf { M } ^ { ( c ) } ( x , y , \tau ) = 1 \}$ , where $( x , y )$ are spatial coordinates.

We measure motion agreement through displacement error, angular error, and the proportion of large mismatches. The masked endpoint error (M-EPE) averages the distance between the two flow vectors over valid locations:

$$
e ( x , y , \tau ) = \left. \mathcal { F } _ { \mathrm { p r e d } } ^ { ( i , c ) } ( x , y , \tau ) - \mathcal { F } _ { \mathrm { r e f } } ^ { ( i , c ) } ( x , y , \tau ) \right. _ { 2 } .\tag{S25}
$$

Table S1. Default guidance coefficients for the Wan2.1 training-free experiments. The flow weights combine displacement, direction, and outlier errors; λ controls the adaptive channel threshold in Eq. (S31), and $\rho$ sets the strength of DSG correction.
<table><tr><td>Coefficient</td><td>Value</td><td>Description</td></tr><tr><td> $\gamma _ { k }$ </td><td>(0.4, 0.4, 0.2)</td><td>Optical-flow metric weights</td></tr><tr><td>λi</td><td>0.65</td><td>FLF threshold coefficient</td></tr><tr><td>ρ</td><td>4.0</td><td>DSG guidance scale</td></tr></table>

$$
\mathrm { M \mathrm { \mathrm { - } E P E } } ^ { ( i , c ) } = \frac { 1 } { | \Omega ^ { ( i , c ) } | } \sum _ { ( x , y , \tau ) \in \Omega ^ { ( i , c ) } } e ( x , y , \tau ) .\tag{S26}
$$

To compare flow directions, we compute their cosine similarity at each valid location and average the corresponding angles in degrees to obtain the masked angular error (M-AE):

$$
\begin{array} { r l } & { \ s ( x , y , \tau ) = \frac { \mathcal { F } _ { \mathrm { p r e d } } ^ { ( i , c ) } ( x , y , \tau ) \cdot \mathcal { F } _ { \mathrm { r e f } } ^ { ( i , c ) } ( x , y , \tau ) } { \Vert \mathcal { F } _ { \mathrm { p r e d } } ^ { ( i , c ) } ( x , y , \tau ) \Vert _ { 2 } \Vert \mathcal { F } _ { \mathrm { r e f } } ^ { ( i , c ) } ( x , y , \tau ) \Vert _ { 2 } } . } \\ & { { \mathrm { M } } \mathrm { - A E } ^ { ( i , c ) } = \displaystyle \frac { 1 8 0 } { \pi \left. \Omega ^ { ( i , c ) } \right. } \sum _ { ( x , y , \tau ) \in \Omega ^ { ( i , c ) } } \operatorname { a r c c o s } s ( x , y , \tau ) . } \end{array}\tag{S27}
$$

(S28)

The outlier percentage $\mathrm { F } ^ { ( i , c ) }$ measures the proportion of valid locations whose endpoint error exceeds 3 pixels or whose relative error $e ( x , y , \tau ) / \| \mathcal { F } _ { \mathrm { r e f } } ^ { ( i , c ) } ( x , y , \tau ) \| _ { 2 }$ exceeds 5%. It captures large local mismatches alongside the average displacement and angular errors.

We normalize these three errors to [0, 1] before combining them:

$$
\begin{array} { r l } & { E _ { i } ^ { c } = \mathrm { m i n } ( \mathrm { M } \mathrm { - } \mathrm { E P E } ^ { ( i , c ) } / n _ { \mathrm { E } } , 1 ) , } \\ & { A _ { i } ^ { c } = \mathrm { m i n } ( \mathrm { M } \mathrm { - } \mathrm { A E } ^ { ( i , c ) } / n _ { \mathrm { A } } , 1 ) , } \\ & { O _ { i } ^ { c } = \mathrm { m i n } ( \mathrm { F } ^ { ( i , c ) } / n _ { \mathrm { F } } , 1 ) . } \end{array}\tag{S29}
$$

Subtracting each normalized error from one gives a score that increases with motion agreement. The weighted motion score is

$$
S _ { i } ^ { c } = \gamma _ { \mathrm { E } } \big ( 1 - E _ { i } ^ { c } \big ) + \gamma _ { \mathrm { A } } \big ( 1 - A _ { i } ^ { c } \big ) + \gamma _ { \mathrm { O } } \big ( 1 - O _ { i } ^ { c } \big ) .\tag{S30}
$$

We set $( n _ { \mathrm { E } } , n _ { \mathrm { A } } , n _ { \mathrm { F } } ) = ( 1 0 , 3 0 ^ { \circ } , 0 . 5 )$ and $( \gamma _ { \mathrm { E } } , \gamma _ { \mathrm { A } } , \gamma _ { \mathrm { O } } ) =$ $( 0 . 4 , 0 . 4 , 0 . 2 )$ , with $\textstyle \sum _ { k } \gamma _ { k } = 1$ . At each step, the selection threshold follows the distribution of channel scores:

$$
\mathcal C _ { i } = \{ c | S _ { i } ^ { c } \geq \delta _ { i } \} , \qquad \delta _ { i } = \mu _ { i } ^ { S } - \lambda _ { i } \sigma _ { i } ^ { S } ,\tag{S31}
$$

where $\mu _ { i } ^ { S }$ and $\sigma _ { i } ^ { S }$ are the mean and standard deviation of the channel scores, and $\lambda _ { i }$ controls the threshold. Channels in $\mathcal { C } _ { i }$ receive the reference fusion.

## 3.2. Hyperparameter Settings

For training-free control with Wan2.1 I2V-14B, we use a 50-step UniPC sampler and apply Intra-Step Recursive Refinement (IRR) during the first 20 steps by default. The FLF and Dual-Path Self-Corrective Guidance (DSG) coefficients are given in Table S1. Farneback flow estimation uses¨ pyr\_scale=0.5, levels=3, and winsize=15 [74].

Early latent states are too noisy for reliable flow estimation, so we disable channel filtering and inject the reference into all channels during steps 0–5. As the motion estimates become more reliable, we allow at most two of Wan2.1’s 16 channels to retain the original prediction during steps $6 \mathrm { - } 1 0 ;$ at least 14 channels therefore continue to receive the reference. During steps 11–19, we relax this limit to six channels, giving the model more freedom to recover texture while maintaining trajectory guidance in the remaining channels. These indices are zero-based and cover the first 20 guided steps.

## 3.3. V-Cache

V-Cache approximates IRR’s clean-state correction in noisy latent space using the velocity cached from the preceding denoising step. The first guided step uses direct reference insertion; subsequent steps within the correction interval use the cached velocity to correct the latent before model evaluation. The new prediction advances the sampler and updates the cache, avoiding the extra model evaluation used by IRR for correction.

To examine whether this approximation preserves IRR’s correction effect, we compare the inferred-noise statistics of the warped-visible and model-generated regions. Direct Insertion, Only IRR, and V-Cache are evaluated on three heldout cases using the same backbone, inputs, target trajectories, initial random seed, noise schedule, and guidance interval.

At denoising step $i , x _ { i }$ denotes the latent input to the final model evaluation of that step, and $v _ { i }$ is the corresponding velocity used to update the sampler. For the Flow Matching path $x _ { \sigma } = ( 1 - \sigma ) x _ { 0 } + \sigma \epsilon .$ , the inferred noise endpoint is

$$
\hat { \epsilon } _ { i } = x _ { i } + ( 1 - \sigma _ { i } ) v _ { i } .\tag{S32}
$$

We compute the regional mean and variance separately for each latent channel, excluding the first conditioning latent frame. Let p index the remaining spatiotemporal positions and $m _ { p } \in [ 0 , 1 ]$ be the warped-visibility mask resampled to the latent resolution. With weights $w _ { p } ^ { \mathrm { v i s } } = m _ { p }$ and $w _ { p } ^ { \mathrm { g e n } } = 1 - m _ { p } ,$ the weighted mean and population variance for channel c and region $r \in \{ \mathrm { v i s } , \mathrm { g e n } \}$ are

$$
\begin{array} { r l } & { \mu _ { i , c } ^ { r } = \frac { \sum _ { p } w _ { p } ^ { r } \hat { \epsilon } _ { i , c , p } } { \sum _ { p } w _ { p } ^ { r } } , } \\ & { \nu _ { i , c } ^ { r } = \frac { \sum _ { p } w _ { p } ^ { r } \left( \hat { \epsilon } _ { i , c , p } - \mu _ { i , c } ^ { r } \right) ^ { 2 } } { \sum _ { p } w _ { p } ^ { r } } . } \end{array}\tag{S33}
$$

For each case, we compute the absolute differences between the two regions within each channel, then average over the C latent channels:

$$
\begin{array} { r l r } & { } & { \Delta _ { \mu } ( i ) = \displaystyle \frac { 1 } { C } \sum _ { c = 1 } ^ { C } \left| \mu _ { i , c } ^ { \mathrm { v i s } } - \mu _ { i , c } ^ { \mathrm { g e n } } \right| , } \\ & { } & { \Delta _ { \mathrm { l o g } \mathrm { v a r } } ( i ) = \displaystyle \frac { 1 } { C } \sum _ { c = 1 } ^ { C } \left| \log \frac { \nu _ { i , c } ^ { \mathrm { v i s } } + \eta } { \nu _ { i , c } ^ { \mathrm { g e n } } + \eta } \right| . } \end{array}\tag{S34}
$$

Here, $\eta ~ = ~ 1 0 ^ { - 1 2 }$ stabilizes the variance ratio, and log denotes the natural logarithm. Each point in the main-paper curves averages the corresponding gap equally over the three cases at the same denoising step. Smaller gaps indicate closer agreement between the regional noise statistics.

Direct Insertion produces larger gaps, particularly in logvariance, while Only IRR and V-Cache reduce these gaps over much of denoising. These reductions accompany the more coherent structures shown in Fig. 4. V-Cache obtains this improvement using the cached velocity, avoiding IRR’s extra model evaluation for correction.

Table S2. Default training settings for Trajectory LoRA. Both backbones first train on recorded guided anchors, then use a mixture of anchors, constructed states, and directly noised targets.
<table><tr><td>Setting</td><td>LongCat</td><td>H3</td></tr><tr><td>Training scenes / pairs</td><td> $8 ~ / ~ 5 0$ </td><td>1 / 50</td></tr><tr><td>Selected blocks</td><td> $3 6 ~ / ~ 4 8$ </td><td>38 /  50</td></tr><tr><td>LoRA rank  $/ \alpha$ </td><td>128 /  128</td><td>128 /  128</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $2 \times 1 0 ^ { - 5 }$ </td><td> $2 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Warm-up iterations</td><td>200</td><td>200</td></tr><tr><td>Total training iterations</td><td>1,200</td><td>1,500</td></tr></table>

## 3.4. Trajectory LoRA Adaptation

Trajectory LoRA adapts selected blocks to follow rendered camera guidance while preserving scene content. We construct training pairs from optimized 3D Gaussian representations [32] of DL3DV scenes [77]. For a source view and target camera path, depth-based warping provides a partial reference and its visibility mask, while rendering the scene along the same path provides the target video. Video-VAE encoding gives the trajectory latent $x _ { \mathrm { t r a j } }$ and clean target latent $x _ { 0 } ^ { \mathrm { g t } }$ . The target video supervises both preservation of visible content and completion of newly exposed regions.

We identify trajectory-sensitive blocks by measuring their contributions to the velocity change learned by a probe LoRA. With the pretrained weights fixed, the probe learns to denoise interpolated trajectory–target states toward the clean target at sampled noise levels. After approximately 400 updates, we evaluate block contributions using 25 trajectory pairs per backbone.

For a fixed validation state $\boldsymbol { \xi } = ( x _ { \sigma } , \sigma , \ell )$ , let $v _ { S } ( \xi )$ be the predicted velocity with probe LoRA enabled in the subset $S \subseteq$ B of candidate blocks. Enabling all probe adapters changes the frozen backbone’s prediction by $d ( \xi ) = v _ { B } ( \xi ) - v _ { \emptyset } ( \xi )$ We measure a subset’s contribution along this full response with the normalized utility

$$
U _ { \xi } ( S ) = \frac { \langle v _ { S } ( \xi ) - v _ { \emptyset } ( \xi ) , d ( \xi ) \rangle } { \| d ( \xi ) \| _ { 2 } ^ { 2 } } .\tag{S35}
$$

A block’s contribution can depend on which other adapters are active. Shapley scores [78] account for this dependence by averaging the marginal change in $U _ { \xi }$ when a block is enabled across different block combinations. We estimate these scores using random permutations paired with their reverses. Let $S _ { \pi , b }$ contain the blocks preceding block b in permutation π, and let Π include the sampled permutations and their reverses. Averaging over Π and the validation states D gives

$$
\begin{array} { r } { \widehat { \phi } _ { b } = \displaystyle \frac { 1 } { \left| \mathscr { D } \right| \left| \Pi \right| } \displaystyle \sum _ { \xi \in \mathscr { D } } \displaystyle \sum _ { \pi \in \Pi } \Big [ U _ { \xi } \big ( S _ { \pi , b } \cup \{ b \} \big ) } \\ { - U _ { \xi } ( S _ { \pi , b } ) \Big ] . } \end{array}\tag{S36}
$$

We select the K highest-scoring self-attention blocks and train LoRA in their query, key, value, and output projections, keeping the pretrained weights fixed. Both backbones use rank $r = 1 2 8$ and scaling parameter $\alpha = 1 2 8$ , so the update multiplier $\alpha / r$ is one. H3 receives reference conditioning through its Ref2VA branch. Table S2 lists the default adaptation settings.

Reference insertion changes the latent states received by the pretrained model, so training begins with states recorded during V-Cache-guided sampling. For each training pair, we save five anchors $( x _ { a } , \sigma _ { a } )$ at approximately $\sigma _ { a } \in$ $\{ 0 . 8 7 , 0 . 9 2 , 0 . 9 3 , 0 . 9 5 , 0 . 9 7 \}$ . During warm-up, we sample these anchors uniformly, add a small noise-dependent perturbation, and train the selected LoRA parameters to recover the clean target. This initial adaptation supplies the velocity estimates used to construct additional training states in the second stage.

The second stage extends training beyond the saved anchors to cover additional noise levels. In our experiments, 10% of updates use saved anchors, 25% use directly noised targets, and 65% use states constructed from anchors at sampled noise levels. For directly noised targets, $\sigma$ is drawn from U(0.08, 0.30) with probability 0.6 and from $\mathcal { U } ( 0 . 3 0 , 0 . 7 0 )$ otherwise, covering the lower noise levels used to recover fine details. Noise levels for anchor propagation are sampled from the recorded schedule with a multiplicative jitter of up to 5%.

When a sampled schedule level $\sigma _ { d }$ is close to a saved anchor, the current model predicts $v _ { a } = v _ { \theta , \phi } ( x _ { a } , \sigma _ { a } , \ell )$ and propagates that anchor with one Euler step:

$$
\bar { x } _ { d } = x _ { a } - ( \sigma _ { a } - \sigma _ { d } ) \mathrm { s g } ( v _ { a } ) ,\tag{S37}
$$

where $\mathrm { s g }$ denotes stop-gradient. Within the active guidance interval, we apply reference insertion with the visibility mask and cached-velocity correction to obtain $\tilde { x } _ { \sigma _ { d } } .$ . State construction does not track gradients; the subsequent denoising prediction on the constructed input updates the LoRA parameters.

Both stages use the target velocity defined in the main paper:

$$
v _ { \sigma } ^ { * } = \frac { \tilde { x } _ { \sigma } - x _ { 0 } ^ { \mathrm { g t } } } { \sigma } .\tag{S38}
$$

Let $e _ { \sigma } ~ = ~ v _ { \theta , \phi } ( \tilde { x } _ { \sigma } , \sigma , \ell ) - v _ { \sigma } ^ { \ast } ,$ , evaluated over the nonconditioning latent elements. The loss covers both visible and newly exposed regions; the first conditioning latent frame remains fixed and is excluded. Both stages use the same squared-error objective as Eq. (15) in the main paper:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { L o R A } } = \mathbb { E } \left[ \| e _ { \sigma } \| _ { 2 } ^ { 2 } \right] , } \end{array}\tag{S39}
$$

where $\theta$ is the frozen backbone, ϕ contains the selected LoRA parameters, and ℓ denotes the model conditioning.

To study scene diversity, we vary the training scenes among one, two, four, and eight while keeping 50 pairs in total. The update-count study fixes each backbone’s training data. Blockselection controls compare equally sized sensitive, random, and low-response subsets, with random-half results averaged over five independent training runs. The training-strategy study fixes inference and compares Standard FM, anchor warm-up alone, guided-state training without warm-up, and the full twostage strategy, together with a no-LoRA baseline.

For standard sampling, LongCat and H3 use 40 and 50 steps, respectively, and generate videos at approximately 832 × 480 resolution. On each backbone, the trained trajectory LoRA can be combined with the released acceleration LoRA without retraining. Generation times and visual comparisons are provided in Sec. 4.4.

To examine transfer across camera-guidance methods, we reuse the trained LongCat LoRA with Warp-as-History [18] and Go-with-the-Flow [16] on the LongCat backbone. Each method provides its own trajectory condition, and the LoRA is applied without further fine-tuning.

## 4. Additional Experiments

## 4.1. Component Behavior and Motion Selection

In training-free WorldForge, introducing a rendered reference gives IRR direct control over camera motion, while FLF and DSG help preserve the scene during this change. In Fig. S1, FLF reduces appearance distortions from reference fusion, and DSG further corrects the guided prediction using the original prediction. Their combination produces coherent scene content along the requested camera path.

![](images/9f0ba17551ecae84a453a2ee8ab4029c7a253c27597a7a93e1099e54e5cf73a9.jpg)  
Figure S1. Contribution of IRR, FLF, and DSG to training-free camera control. IRR introduces the target camera motion; adding FLF reduces appearance distortions during reference fusion, and DSG further corrects the guided prediction. The complete method preserves coherent scene content along the requested camera path.

For training-free control, FLF selects channels by comparing their motion with the rendered reference. Figure S2 records how often each channel retains its original prediction across more than 40 static and 30 dynamic scenes. The stable differences in filtering frequency reflect differences in motion relevance: selected channels receive camera guidance, while the remaining channels preserve information from the model’s prediction.

![](images/edc096bccb9bf83fa131d182479c5766fcc530bab1e2c1f8cad0d5ebecbde5d2.jpg)  
Figure S2. Channel filtering frequencies for training-free control over more than 40 static and 30 dynamic scenes. A higher frequency means that a channel more often retains its original prediction. Stable differences across channels support selecting where to inject the camera guidance according to motion relevance.

## 4.2. Self-Corrective Guidance

The original and trajectory-guided predictions can point in substantially different directions, making their combination sensitive to the guidance rule. In Fig. S3, directly applying standard CFG introduces structural artifacts, and removing DSG’s adaptive angular weight weakens consistency. The complete DSG uses cosine-based projection and adaptive weighting to combine the predictions, preserving scene structure and fine detail as it follows the rendered guidance.

Table S3. Training-free camera control across three video backbones on the static benchmark of more than 40 scenes. All backbones use the same rendered-reference guidance. LongCat-Video leads on CLIP and RPE-T, while Wan2.1 leads on ATE and RPE-R; both improve all reported metrics over SVD. Bold and underlining mark the best and second-best results.
<table><tr><td>Backbone</td><td>CLIP↑</td><td>ATE↓</td><td>RPE-T↓</td><td>RPE-R↓</td></tr><tr><td>SVD</td><td>0.910</td><td>0.265</td><td>0.316</td><td>0.444</td></tr><tr><td>Wan2.1</td><td>0.948</td><td>0.077</td><td>0.086</td><td>0.221</td></tr><tr><td>LongCat-Video</td><td>0.949</td><td>0.095</td><td>0.076</td><td>0.230</td></tr></table>

![](images/cea95a82ab8cf554d0917d699abb99daee8bcc5414530f3c5dffb2b31f5a7272.jpg)  
Figure S3. Combining the original and trajectory-guided predictions for training-free camera control. Standard CFG introduces structural artifacts, and removing DSG’s adaptive angular weight weakens consistency. Cosine-based projection with adaptive weighting preserves scene structure and fine detail in the guided result.

## 4.3. Backbone and Geometry Compatibility

We apply the same training-free guidance to the U-Netbased SVD [1] and the transformer-based LongCat-Video [71]. Table S3 compares them with Wan2.1 on more than 40 static scenes. Wan2.1 and LongCat-Video improve both CLIP similarity and camera accuracy over SVD, showing that the shared control mechanism can benefit from stronger video backbones. Figure S4 compares guidance methods under a shared SVD backbone and shows the richer scene detail obtained with LongCat-Video.

For training-free guidance, FLF uses optical flow to compare predicted motion with the rendered reference. Replacing the lightweight Farneback estimator [74] with RAFT allows¨ us to examine whether this selection remains useful with a learned flow model. Figure S5(a) shows that FLF reduces rendering artifacts and improves generation quality with both estimators.

We also construct warped references with VGGT, UniDepth, Mega-SaM, and DepthCrafter to examine the effect of depth estimation (Fig. S5(b)). Although the estimated geometry differs, the generated views maintain recognizable layouts and plausible surfaces across these reference inputs.

## 4.4. Training Resources and Inference Efficiency

Trajectory LoRA learns camera control from 50 videos of 25–93 frames while keeping the pretrained backbone fixed. LongCat uses eight scenes and 1,200 updates, and H3 uses one scene and 1,500 updates. Table S4 compares these training resources and the resulting generation time with the other methods. Warp-as-History also uses a compact source set, deriving four training clips from one source video and expanding them through online clip and direction augmentation.

![](images/1552292335d4d1ad5ce900a2ae4d408308f00c67138160b930ea6b65e5d923a6.jpg)  
Figure S4. Training-free camera control across video backbones. Comparisons under a shared SVD backbone show the effect of the guidance method. WorldForge also produces richer scene detail with LongCat-Video, using the same inference-time guidance.

![](images/89ec28a5592f27a63f148ef0fae1df53fe49dd5ea9fa9688b0142e6855f6358e.jpg)  
Figure S5. Compatibility of training-free guidance with motion and geometry estimators. (a) FLF reduces rendering artifacts with both Farneback¨ and RAFT optical flow. (b) Warped references from VGGT, UniDepth, Mega-SaM, and DepthCrafter support coherent scene layouts and plausible surfaces despite differences in their estimated geometry.

For each backbone, the trained trajectory LoRA can be reused with a released acceleration LoRA, reducing sampling cost without further trajectory training. LongCat [71] reduces its sampling steps from 40 to 16; H3 uses MiniMax-H3 Turbo LoRA to reduce sampling from 50 to eight. Visual results are shown in Fig. S6.

## 4.5. Challenging Scenes and Control Limits

Fast motion, non-rigid motion, and heavy occlusion make reference fusion more difficult. Figure S8 compares generation with and without FLF under these conditions. Selecting the channels for fusion reduces the marked distortions in objects, water, and scene boundaries while retaining the requested viewpoint changes.

Table S4. Training resources and inference efficiency for camera-controlled generation. Our trajectory adapters learn from 50 videos with 1,200 or 1,500 updates. Combining them with released acceleration LoRAs gives speedups of 4.09× on LongCat and 5.09× on H3. Times are mean seconds per generated video over 200 test cases for each method, estimated for a single A100 from local runs with native video configurations. Parameter counts refer to the backbones. Training updates exclude backbone pretraining; a dash denotes an unreported total update count.
<table><tr><td>Method</td><td>Backbone</td><td>Backbone params.</td><td>Training data</td><td>Training updates</td><td>Inference time (s)</td></tr><tr><td>UniWorld-View [83]</td><td>Wan2.1-VACE</td><td>14B</td><td>100K triplets + 100K pairs</td><td>20,000</td><td>124.5</td></tr><tr><td>TrajectoryCrafter [5]</td><td>CogVideoX-Fun</td><td>5B</td><td>120K multi-view + 60K pairs</td><td>15,000</td><td>169.0</td></tr><tr><td>SymphoMotion [84]</td><td>Wan2.1-I2V</td><td>14B</td><td>25K videos</td><td></td><td>442.9</td></tr><tr><td>Warp-as-History [18]</td><td>Helios</td><td>14B</td><td>1 scene / 4 clips</td><td>1,000</td><td>39.3</td></tr><tr><td>Ours (LongCat)</td><td>LongCat-Video</td><td>13.6B</td><td>8 scenes / 50 videos</td><td>1,200</td><td>274.2</td></tr><tr><td>Ours (LongCat+acceleration LoRA)</td><td>LongCat-Video</td><td>13.6B</td><td>8 scenes / 50 videos</td><td>1,200</td><td>67.0</td></tr><tr><td>Ours (H3)</td><td>MiniMax-H3</td><td>33B</td><td>1 scene / 50 videos</td><td>1,500</td><td>707.3</td></tr><tr><td>Ours (H3+acceleration LoRA)</td><td>MiniMax-H3</td><td>33B</td><td>1 scene / 50 videos</td><td>1,500</td><td>139.0</td></tr></table>

![](images/b3218d73674d71f44fa99de64b5fbb7bf1023ecdaec5dd235501317a454001d2.jpg)

Figure S6. H3 generation with 50-step and eight-step sampling using the same trajectory LoRA. The accelerated results retain scene layout and object geometry in both examples. Enlargements compare corresponding details.  
![](images/b19257691d4b8c797b041951886d198c3e1b917b3766ade5da8773d4369e3c81.jpg)  
Figure S7. Camera-control limits under erroneous depth and complex scene dynamics. Distorted face geometry and inconsistencies around moving people show how reference errors remain visible in the generated views. Rows compare training-free WorldForge (top) with TrajectoryCrafter (bottom).

![](images/ed761a06c4a2b570e542eb53b21b778811458cf040536930484689a88f81f81c.jpg)  
Figure S8. FLF in training-free generation under fast motion, non-rigid motion, and heavy occlusion. Comparing w/o FLF with w/ FLF shows that selecting channels for reference fusion reduces distortions in objects, water, and scene boundaries while retaining the requested viewpoint changes.

![](images/4823f0a9e5033fcac3ec4561024fcac0c1c971e7c8eb4ba79813bf31a7374c34.jpg)  
Figure S9. Training-free generation of an object-centric 360<sup>◦</sup> orbit from one input image. The final frame of each clip initializes the next segment, extending camera control beyond the original field of view while retaining recognizable scene content.

Severe depth errors can give the rendered reference incorrect geometry, while complex scene dynamics can make its guidance inconsistent across frames. Figure S7 shows these errors in the outputs of training-free WorldForge and TrajectoryCrafter, including distorted faces and inconsistencies around moving people.

## 4.6. Wide-Range Control and Editing

To follow paths with large camera rotations, we generate successive clips and use the final frame of each clip to initialize the next. Figure S9 shows an object-centric 360<sup>◦</sup> orbit generated from one input image. The clips retain recognizable scene content as the camera moves beyond the original field of view.

The same rendered-reference guidance also supports camera freezing and stabilization (Fig. S10). For content editing, we segment the target region in each frame and edit its appearance in the first frame. This edited frame and the region masks guide the change through the video, while the rendered reference specifies the viewpoint. For an object absent from the source video, a bounding box specifies where it should be added.

## 5. Additional Visual Comparisons

Figure S11 adds six held-out DL3DV cases to the learnedcontrol comparison. Indoor layouts, outdoor structures, and finely textured objects allow us to examine how viewpoint changes affect structural boundaries, repeated patterns, and surface detail. Our LongCat and H3 results preserve scene layout and local appearance, while several baseline outputs contain distorted boundaries, texture loss, or rendering remnants.

![](images/12ec91487a87a8527989c94f5de31eea597f0e88e40344790697a786429e8437.jpg)  
Freeze Camera

![](images/b1c3007f0a90b1433ca48f74a5646def5b7879d77db46304e4605ed339c8cd9a.jpg)

![](images/8629e5d1b4b73740f15f8450f5cea07005d257f46ae90dd3af2c0060a5043ec8.jpg)  
Video Stabilization  
Source PathTarget Path

![](images/77a6de93aa12c9678d9db3dae6263f8efc8c444aef1ec6c79bcb0f1b1aebd8e9.jpg)  
Video Edit (Erasure)

Input & Warped  
![](images/b7f1ccf71d47d1906817ba1344446640c0c6ff8e2fed5f803d3a9781086616ad.jpg)  
UniWorld-View

![](images/0179c728fa8a5e9b695c9d92c1dd8e61e93a9d3791f6b978b930f6d70c0416de.jpg)  
Video Edit (Replacement)

GT  
![](images/36109c2793d46c6b51805551a1aee47f0b550e347edc12fadfadb055adfaa5ed.jpg)  
Figure S10. Camera and content editing with training-free WorldForge. Rendered references support camera freezing and stabilization while retaining scene content. An edited first frame and region masks propagate appearance changes through the video.

TrajectoryCrafter  
SymphoMotion  
Warp-as-History  
Ours (LongCat)  
Ours (H3)  
![](images/8a00209f48918f852cd5f9c77ff63fded8618e250c52f32713bc1e23221a7d2d.jpg)  
Figure S11. Additional camera-controlled generation on six held-out DL3DV cases beyond the main figure. Our LongCat and H3 results preserve scene layout and local appearance, while several baseline outputs exhibit distorted boundaries, texture loss, or rendering remnants. LongCat uses the maintable model; H3 uses the eight-step accelerated sampling evaluated in Sec. 4.4. Inputs and warped references appear on the left, and red boxes identify corresponding enlarged regions.