# OmniCam: Omni-Camera Trajectory Generation via Geometry-Grounded Pose Token Learning

Zhenyang Liu<sup>1,2,3</sup> Chenjie Cao<sup>3</sup> Yisu Zhang<sup>4,3</sup> Xuhui Zuo<sup>3</sup> Xiangyang Xue<sup>1,‡</sup> Yanwei Fu<sup>1,2,‡</sup> Tengfei Wang<sup>3,†</sup> Chunchao Guo<sup>3</sup>

<sup>1</sup>Fudan University <sup>2</sup>Shanghai Innovation Institute <sup>3</sup>Tencent Hunyuan <sup>4</sup>Zhejiang University Project Page: ZhenyangLiu.github.io/OmniCam-Website

![](images/a36dce13886a33a5d6d7b69af813286eaaa61408adbb2ff967be747852e266b9.jpg)

![](images/05a4cbc531cc1fbc0951863003cfbd02d30e9ee091944f54cbc4a2ec40d0b24c.jpg)

![](images/5f094d8802477a4eb7e8d964b08be3f56d8555b2ff210da829b0236eff62b07f.jpg)  
Figure 1. Overview of OmniCam. Given a single panorama and textual trajectory descriptions, OmniCam generates spatially aware camera trajectories (SE(3) pose sequences). OmniCam supports four camera behaviors that enable various downstream applications such as camera-controlled video generation, 3D reconstruction, and active perception in robotics.

## Abstract

Camera trajectories control viewpoint changes in video generation, scene reconstruction, and robotic perception. Generating them from language requires both scene geometry and target-awareframing. We introduce OmniCam, an autoregressive model that generates camera pose sequences from a single panorama and textual trajectory descriptions. Its geometry-grounded pose token learning combines three components: a panoramic point-cloud encoderfor omnidirectional geometric context; hybrid absolute-rotation and relative-translation tokenization with temporally consistent quaternion signs; and separate geometric and semantic conditioning streams with an explicit 3D target anchor. We also construct OmniCaT, containing 267,700 trajectories across four camera behaviors. On the reported OmniCaT evaluation, OmniCam reduces trajectory errors by 28–47% and collision rate by 65.8% relative to GenDoP retrained on OmniCaT. Against the best baselinefor each metric, the

ATE and collision reductions are 43.0% and 62.3%, respectively. Component ablations support the use of geometric and target-aware conditioning, while downstream experiments examine camera-controlled video generation and robotic active perception.

## 1. Introduction

Camera trajectories serve as a control interface for spatialintelligence systems, including world models [1], video generation [13, 31], and robotics [2, 16, 21]. Specifying a trajectory requires deciding where to move, how to orient, and what to attend to. A prescribed orbit or linear sweep may fail when it intersects scene geometry or loses the intended target. We study generating scene-aware camera trajectories from a panoramic observation and a textual instruction (Figure 1).

Recent trajectory generators [7, 18, 23, 34] motivate three design questions. (i) Scene coverage: perspective RGBD inputs [34] provide limited angular coverage, while geometry-based planners require explicit spatial representations [29, 30]. A panorama provides observations in all directions, although occluded surfaces remain unknown. (ii) Pose serialization: quantization trades precision against token vocabulary size. Quaternion sign ambiguity $( q \equiv - q )$ , coordinate scale, and the accumulation of relative increments can affect the resulting learning problem. (iii) Conditioning: geometric free-space information and language-referenced targets serve different roles in planning. Providing dedicated pathways for these cues is a useful architectural choice whose value should be tested empirically. We use geometrygrounded pose token learning to describe a design that connects geometric perception, pose serialization, and targetaware conditioning.

OmniCam implements this design with a panoramic point-cloud encoder, hybrid pose tokenization, and a dualbranch conditioning architecture. The point cloud, reconstructed from estimated panoramic depth, provides omnidirectional but partial geometry. Rotations are encoded absolutely with sequential quaternion sign alignment; translations are encoded as increments in a scene-normalized coordinate system. Sign alignment selects consistent representatives of the quaternion double-cover, while quantization remains lossy. Absolute rotations avoid composition of rotationquantization residuals; relative translations trade a smaller per-step range against accumulated displacement error. We analyze these distinctions in Section I and compare the encodings empirically.

Separate geometric and semantic pathways supply the decoder with structural and target-related features. An attention-derived 3D target centroid is injected through gated cross-attention to provide a spatial anchor for object-directed trajectories. The geometry pathway does not directly receive text; this routing does not imply statistical independence or guarantee the absence of interference after fusion. At inference, a format constraint restricts termination to complete pose-token groups.

Training requires trajectories paired with scene observations and instructions. Existing collections [24, 33, 34, 37] cover recorded object, scene, and cinematic motions. We construct OmniCaT by recovering geometry and semantic targets from panoramas and synthesizing four behaviors: Target, Surrounding, Reconstruction, and Wandering. The dataset contains 267,700 trajectories with hierarchical descriptions of task, target, and motion. Because supervision is generated on estimated geometry, the learned model may inherit reconstruction errors and planner preferences.

Our contributions are:

• Panorama-conditioned trajectory generation. Omni-Cam combines explicit point-cloud features, dedicated geometric and semantic pathways, and a 3D target anchor to generate camera trajectories from panoramic imagery and language.

• Hybrid pose serialization. We combine absolute rotations, relative translations, and consistent quaternion signs; provide qualified error and representation analyses; and compare the choices through ablations.

• OmniCaT and empirical evaluation. We introduce a 267,700-trajectory dataset spanning four behaviors. The reported OmniCaT results improve on retrained baselines, with component ablations and downstream video and robotic-perception evaluations.

## 2. Related Work

Camera Trajectory Generation. Early approaches rely on optimization-based planning [3, 11, 25, 28] or learningbased control [4, 10, 15, 17]. Recent methods integrate camera movement with scene dynamics: CCD [18] employs a diffusion model with text and keyframe conditioning in character-centric coordinates; E.T. [7] incorporates character trajectories in global coordinates; Director3D [23] adopts a DiT-based framework on multi-view data; Hou et al. [14] predict next-frame motion autoregressively for aerial videography; GenDoP [34] combines text with RGBD inputs in an autoregressive framework. These methods use different task-specific inputs and trajectory representations. Omni-Cam studies panoramic point-cloud conditioning and explicit target anchoring, together with the effect of absolute and relative pose serialization in an autoregressive generator.

Camera Trajectory Datasets. Existing trajectory datasets remain limited in spatial diversity. MVImgNet [33], RealEstate10K [37], and DL3DV-10K [24] provide calibrated trajectories but focus on simple motions around stationary objects or within static environments. CCD [18] and E.T. [7] offer character-centric tracking data that does not generalize to unconstrained spatial navigation. Data-DoP [34] pairs trajectories with depth maps and text for cinematic applications, yet relies on perspective-view depth covering only a limited field of view. None of these resources simultaneously provides omnidirectional scene geometry, diverse spatial behaviors, and hierarchical textual guidance. OmniCaT fills this gap by grounding trajectory generation in panoramic point-cloud geometry and pairing each trajectory with structured language annotations across four distinct navigation behaviors.

Pose Representation Learning. Rotation representation [19, 22] has a well-studied impact on learning stability. Zhou et al. [38] show that quaternions and Euler angles suffer from discontinuities under the topology of SO(3) and propose continuous 6D representations for single-frame rotation regression. Subsequent work considers sequential settings such as human motion prediction [5, 8] and robotic manipulation [21, 26]. OmniCam uses sequential quaternion sign alignment to reduce inconsistent discrete labels for equivalent rotations, combined with relative translation. This convention does not provide a globally continuous rotation parameterization or an injective finite quantizer.

![](images/0ec5ef95d5bbddd197438f8747e02d0aaca19f2eef43d5c11ec08aee69b13735.jpg)  
Figure 2. OmniCaT Dataset Overview. Left: (a) Construction pipeline comprising (1) Geometric and Semantic Grounding to extract nav-meshes and 3D targets, (2) Spatial Trajectory Synthesis to generate diverse camera behaviors, and (3) Rendering and Multimodal Annotation to produce text-trajectory pairs. Right: Dataset statistics showing (b) task distribution, (c) top object categories, (d) motion primitive diversity, and (e) target direction distribution.

## 3. The OmniCaT Dataset

Collecting camera trajectories with controlled behavior and target coverage is costly. OmniCaT synthesizes trajectories on geometry reconstructed from panoramic images, enabling controlled variation of spatial behaviors. Feasibility is assessed against the reconstructed geometry and does not by itself guarantee collision avoidance in the physical scene.

## 3.1. Dataset Construction

As illustrated in Figure 2, the pipeline consists of three phases. (1) Geometric and semantic grounding: panoramic depth maps are aligned into a global point cloud P<sup>pan</sup> [32], filtered for outliers, and augmented with semantic landmarks and 3D volumetric masks using language-assisted annotation and segmentation models [6, 20, 27]; a navigation mesh is then constructed to define the traversable space. (2) Spatial trajectory synthesis: four categories of camera behaviors (Target, Surrounding, Reconstruction, and Wandering) are generated on the navigation mesh, with waypoints and orientations refined via spline and spherical linear interpolation. (3) Rendering and multimodal annotation: a point-based renderer produces video sequences from the planned poses, and a hierarchical annotation scheme derives kinematic motion primitives from camera extrinsics before a vision-language model synthesizes motion-aware textual instructions. Full implementation details, including depth alignment optimization, NavMesh engineering, and trajectory planning heuristics, are provided in Section A.

## 3.2. Dataset Statistics

Scale and Annotations. As summarized in Figure 2, OmniCaT comprises 267,700 trajectories across 9,998 scenes (21.7M rendered frames), categorized into four spatial behaviors: Surrounding (97,702), Target (90,065), Reconstruction (43,316), and Wandering (36,617). Each trajectory is annotated with 3D semantic targets, directional bearings, and hierarchical textual descriptions, enabling joint geometricsemantic conditioning. Full annotation details are provided in Section A.

Scope of Synthetic Supervision. The planner uses reconstructed geometry, navigation meshes, and semantic targets to generate feasible examples. OmniCam learns to predict trajectories from a panorama and text without constructing a navigation mesh or running graph search at inference. Downstream evaluation tests the usefulness of this learned mapping. A gap to the planner does not establish freedom from planner bias; evaluation on independently measured geometry and independently specified tasks is needed to assess that limitation.

## 4. Method

## 4.1. Hybrid Pose Tokenization

We serialize continuous camera parameters using sceneaware calibration and a hybrid absolute-relative representation. Quaternion sign alignment addresses equivalent positive and negative representations; finite binning still introduces quantization error.

Scene-Aware Calibration. Panoramic depth is unprojected into a filtered 3D point cloud. Scene statistics, including a centroid c and a scale s, calibrate camera translations and the initial frame. Normalization reduces variation in coordinate magnitude across scenes but does not eliminate errors in estimated metric scale.

![](images/6f8f241a01ce3e06b4826788bef163ae3c67f1db22f8a4a341be50f0d83fb3ab.jpg)  
Figure 3. OmniCam Architecture. Three parallel branches encode the caption, panoramic image, and point cloud. A Language-Guided Resampler produces target-aware semantic tokens while a Geometry Resampler yields geometric descriptors; an attention-aggregated 3D target centroid provides an explicit spatial anchor for the decoder. The autoregressive decoder generates pose tokens from the concatenated condition, which are mapped to continuous SE(3) trajectories by a de-tokenizer. Right bottom: SE(3) tokenization with 9 tokens for each pose frame.

Hybrid Absolute-Relative Pose Encoding. Rotations are encoded as absolute unit quaternions with sequential sign alignment: each frame is aligned with the preceding aligned representative. The deterministic convention and its assumptions are stated in Proposition 2; implementation-specific tie handling must agree with that convention. Translations after the initial frame use relative increments, reducing the perstep dynamic range while allowing quantization residuals to accumulate. These components, together with normalized intrinsics, are quantized via uniform binning into per-frame token groups of 9 tokens: 4 rotation, 3 translation, 2 intrinsics (Figure 3, right bottom). The token sequence S is augmented with boundary tokens (BOS, EOS, PAD) and projected into a latent space through a learnable codebook.

Theoretical Motivation. The hybrid scheme balances different reconstruction-error tradeoffs (Section I). If each decoded rotation has angular quantization error at most ϵ, absolute encoding retains this per-frame bound, whereas integrating relative rotations permits a worst-case bound of min $\{ \pi , ( T - 1 ) \epsilon \}$ when the initial rotation is exact (Proposition 1). The value of ϵ depends on the component quantizer and decoder normalization. Relative translations can use a smaller displacement range and hence finer local resolution, but their errors can also accumulate. Proposition 2 shows that a specified initial sign rule followed by alignment to the already aligned preceding quaternion removes dependence on the arbitrary quaternion sign when adjacent rotations differ by less than π. This resolves double-cover ambiguity in the supervision; finite quantization remains lossy and many-to-one.

## 4.2. Decoupled Dual-Branch Conditioning

OmniCam encodes estimated panoramic geometry and language-referenced appearance in separate pathways. The point cloud supplies omnidirectional, partial scene structure; the semantic pathway supplies target-related features. This architecture separates input routing, without assuming that the learned representations are statistically independent.

Semantic Branch (Text & Vision Encoders). The semantic branch jointly processes the trajectory caption and the panoramic image to produce target-aware features. A pretrained text encoder maps the caption $T _ { t x t }$ into textual features $\mathbf { Z } _ { t x t } \in \mathbb { R } ^ { N _ { t x t } \times C _ { t x t } }$ , which also serve as cross-modal guidance for the vision pathway. In parallel, a vision encoder produces a visual feature map $\mathbf { F } _ { s e m } \in \mathbb { R } ^ { H \times W \times C _ { s e m } }$ from $\mathbf { I } _ { p a n o }$ . Text and the target descriptor $T _ { o b j }$ guide the subsequent semantic resampling stage. Structured caption dropout varies instruction composition during training (Section H).

Geometric Branch (Point Cloud Encoder). The geometric branch receives no direct text input. At inference time, a monocular depth estimator [32] reconstructs panoramic depth from the input RGB panorama; the estimated depth is then unprojected into a 3D point cloud, voxelized, and processed by a sparse-convolution encoder combined with a Transformer encoder to yield geometric descriptors ${ \bf F } _ { g e o } \in { \bf \Sigma }$ $\mathbb { R } ^ { M \times C _ { g e o } }$ . Its features provide structural context for trajectory generation. Architectural details for both branches are provided in Section C.

## 4.3. Spatial Grounding via Resampling and Target Centroid Injection

This section introduces two parallel resamplers that compress heterogeneous encoder outputs (text, vision, and point cloud features) into compact prefix tokens, and an attentionaggregated 3D centroid that serves as an explicit spatial anchor.

Language-Guided Semantic Resampler. Given the visual features from the vision encoder, we first project them into the 3D point-cloud space to obtain 3D semantic features $\mathbf { Z } _ { s e m } ^ { ( 0 ) }$ . The semantic resampler combines self-attention, spatial attention, and textual attention. The interaction is summarized schematically at layer l:

$$
\begin{array} { r l } & { \mathbf { Z } _ { s e m } ^ { ( l ) } = \mathrm { S e l f A t t n } \Big ( \underbrace { \mathbf { Z } _ { s e m } ^ { ( l - 1 ) } } _ { \mathrm { Q , K , V } } \Big ) } \\ & { ~ + ~ \mathrm { C r o s s A t t n } \Big ( \underbrace { \mathbf { Z } _ { s e m } ^ { ( l - 1 ) } } _ { \mathrm { Q } } , \underbrace { \big [ \mathbf { P } ; \mathbf { F } _ { s e m } \big ] } _ { \mathrm { K , V } } \Big ) } \\ & { ~ + ~ \mathrm { C r o s s A t t n } \Big ( \underbrace { \mathbf { Z } _ { s e m } ^ { ( l - 1 ) } } _ { \mathrm { Q } } , \underbrace { \mathbf { Z } _ { t x t } } _ { \mathrm { K , V } } \Big ) . } \end{array}\tag{1}
$$

where 3D semantic features serve as the common query; selfattention captures inter-query dependencies; spatial crossattention grounds queries in physical scene geometry aggregated by 3D point coordinates P concatenated with visual features ${ \bf F } _ { s e m }$ , and text-guided cross-attention aggregates textual guidance from $\mathbf { Z } _ { t x t }$ . The resulting $\mathbf { Z } _ { s e m } \in$ $\mathbb { R } ^ { M _ { s e m } \times C }$ serves as the semantic latent code.

Geometry Resampler. A text-agnostic geometry resampler compresses dense per-point descriptors ${ \bf F } _ { g e o }$ into a fixedlength latent sequence $\mathbf { \dot { Z } } _ { g e o } \in \mathbb { R } ^ { M _ { g e o } \times C }$ through a stack of Transformer decoder layers, providing a geometric condition without a direct text input.

Target Centroid Injection. A target centroid $\mathbf { c } _ { t g t }$ supplies an explicit spatial anchor. Let $a _ { j }$ denote the target-attention logit associated with point $\mathbf { p } _ { j }$ and let $\tau > 0$ be the temperature. The described soft-argmax aggregation has the form

$$
\alpha _ { j } = \frac { \exp ( a _ { j } / \tau ) } { \sum _ { k = 1 } ^ { N } \exp ( a _ { k } / \tau ) } , \qquad \mathbf { c } _ { t g t } = \sum _ { j = 1 } ^ { N } \alpha _ { j } \mathbf { p } _ { j } .\tag{2}
$$

This requires point-indexed weights; attention over text tokens alone does not directly define them. The centroid is positionally encoded into an auxiliary target memory $\mathbf { E } _ { t g t } ^ { \mathsf { ^ { * } } } \in \mathbb { R } ^ { M _ { t g t } \times C }$ . When caption dropout removes the target object tag, these tokens are masked.

The final multimodal condition Z is formed by concatenating the textual features with the resampled panoramic

latents:

$$
\mathbf { Z } = [ \mathbf { Z } _ { t x t } ; ~ \mathbf { Z } _ { s e m } ; ~ \mathbf { Z } _ { g e o } ] .\tag{3}
$$

The target token bank $\mathbf { E } _ { t g t }$ serves as auxiliary memory accessed by the decoder through a gated cross-attention module, as detailed in Section 4.4.

## 4.4. Format-Constrained Autoregressive Decoding

An autoregressive Transformer decoder D generates discrete trajectory tokens conditioned on the multimodal prefix Z and the target token bank $\mathbf { E } _ { t g t }$

Decoder Architecture. The condition Z is prepended to the input sequence before the BOS token. Each decoder block consists of causal self-attention, gated cross-attention over $\mathbf { E } _ { t g t } .$ , and a feed-forward network. The gate controls the contribution of the target memory. Its effect on target-directed trajectories is evaluated through the grounding ablations; per-behavior gate strength is not inferred from the aggregate results.

Format-Constrained Decoding. During inference, the EOS token is masked within each group of 9 tokens (rotation, translation, and intrinsics), so termination occurs only at a frame boundary. The initial frame is constrained by the calibrated origin. This is a sequence-format constraint: valid rotations require an appropriate quaternion decoding rule, and collision avoidance is assessed empirically rather than guaranteed by token masking. Top-k sampling controls the sampling support.

Training Objective. The model is optimized with a weighted cross-entropy loss Lce $\mathcal { L } _ { c e }$ over pose tokens, where rotation and translation channels receive higher weights to prioritize geometric accuracy. An auxiliary smooth $\ell _ { 1 }$ localization loss $\mathcal { L } _ { l o c }$ supervises the estimated centroid $\mathbf { c } _ { t g t }$ against the annotated 3D target position: $\mathcal { L } = \mathcal { L } _ { c e } + \lambda \cdot \mathcal { L } _ { l o c } .$

## 5. Experiments

## 5.1. Experimental Setup

Datasets. OmniCaT experiments are reported on disjoint training and evaluation splits. Exact scene-level counts and split manifests require verification. DataDoP [34] results are reported for OmniCam trained on OmniCaT without Data-DoP fine-tuning. The construction of OmniCam inputs and the evaluation geometry on DataDoP must be specified before interpreting these as a controlled zero-shot comparison. Baselines. Director3D [23] and GenDoP [34] are retrained on OmniCaT. CCD [18] and E.T. [7] use released checkpoints as out-of-domain references. Their character-centric conditioning differs from the available OmniCaT inputs, so these rows do not isolate model quality under matched conditioning.

Metrics. We report trajectory errors (ATE, FDE, and RPE-R/T, using the EVO toolkit [12] where applicable), collision against reconstructed navigation meshes, target visibility,

![](images/e702b9cbc0cefbf71c8baef1f9d0fec30fe5b3eb96d32929419812e31285cf36.jpg)

Table 1. Comparison with representative trajectory generation baselines on OmniCaT and the public DataDoP benchmark.
<table><tr><td>Method</td><td>Dataset</td><td>ATE↓</td><td>FDE↓</td><td>RPE-R↓</td><td>RPE-T↓</td><td>Coll. (%)↓</td><td>Vis. (%) ↑</td><td>Angle (°) ↓</td></tr><tr><td colspan="9">Evaluation on the OmniCaT benchmark</td></tr><tr><td>CCD [18]</td><td>Pre-trained</td><td>2.483</td><td>3.396</td><td>3.112</td><td>0.082</td><td>36.9</td><td>12.3</td><td>32.5</td></tr><tr><td>E.T. [7]</td><td>Pre-trained</td><td>1.983</td><td>2.368</td><td>2.151</td><td>0.069</td><td>46.2</td><td>21.3</td><td>40.2</td></tr><tr><td>Director3D [23]</td><td>Pre-trained</td><td>1.828</td><td>2.942</td><td>2.309</td><td>0.052</td><td>28.4</td><td>28.1</td><td>27.2</td></tr><tr><td>Director3D</td><td>OmniCaT</td><td>1.522</td><td>2.648</td><td>2.187</td><td>0.049</td><td>27.6</td><td>29.3</td><td>26.5</td></tr><tr><td>GenDoP [34]</td><td>Pre-trained</td><td>1.865</td><td>2.951</td><td>1.681</td><td>0.044</td><td>32.6</td><td>31.5</td><td>29.1</td></tr><tr><td>GenDoP</td><td>OmniCaT</td><td>1.624</td><td>2.483</td><td>1.577</td><td>0.038</td><td>30.4</td><td>33.6</td><td>27.6</td></tr><tr><td>OmniCam (Ours)</td><td>OmniCaT</td><td>0.868</td><td>1.462</td><td>1.135</td><td>0.027</td><td>10.4</td><td>52.0</td><td>10.9</td></tr></table>

Evaluation on the public DataDoP benchmark [34]
<table><tr><td>CCD [18]</td><td>Pre-trained</td><td>2.697</td><td>3.338</td><td>3.218</td><td>0.093</td><td>44.2</td><td>18.7</td><td>34.2</td></tr><tr><td>E.T. [7]</td><td>Pre-trained</td><td>2.568</td><td>3.142</td><td>3.086</td><td>0.084</td><td>45.9</td><td>22.5</td><td>33.8</td></tr><tr><td>Director3D [23]</td><td>Pre-trained</td><td>1.944</td><td>2.887</td><td>2.968</td><td>0.077</td><td>39.6</td><td>30.8</td><td>34.3</td></tr><tr><td>Director3D</td><td>DataDoP</td><td>1.688</td><td>2.762</td><td>2.819</td><td>0.062</td><td>41.7</td><td>32.2</td><td>29.5</td></tr><tr><td>GenDoP [34]</td><td>DataDoP</td><td>1.437</td><td>2.255</td><td>1.642</td><td>0.058</td><td>28.5</td><td>36.6</td><td>29.8</td></tr><tr><td>OmniCam (Ours)</td><td>OmniCaT</td><td>0.925</td><td>1.834</td><td>1.118</td><td>0.046</td><td>8.9</td><td>64.2</td><td>12.7</td></tr></table>

![](images/3804415aec2cb46b08c8557cad436bfd627d72a4c97d10697861f5e943920c4f.jpg)

![](images/1406285bee507315c639530fadd73fb11c8992d248dba3c318edbcc63bfe313a.jpg)

![](images/4a9212a0a1a78b736e3684764a5c36b1215c41f39733207115e6c019f8f230aa.jpg)

![](images/3fab5b4b8e36bf1f2f40620189662d37342e702e5e5beb26290907f5583ef21c.jpg)

![](images/520d1b8e603712ad4ae512a2090dc18008fa041694cd18e0c951fc2e2d76a829.jpg)  
<TASK: SURROUND> <TARGET\_OBJ: sculpture> <TARGET\_DIR: Back> The … executes an orbit left to gracefully circle around the central structure. The sequence concludes by arriving at the final <END\_OBJ: sculpture> position in the <END\_DIR: Back> sector.

![](images/720d1d60540b05112e6042f063029e3fb044f80dd24971e9f4b9cd86779bcae3.jpg)  
<TASK: TARGET> <TARGET\_OBJ: door> <TARGET\_DIR: Front> The intent is to draw attention to the Front door through a push in, and finally turn right. The sequence concludes by arriving at the final <END\_OBJ: door> position in the <END\_DIR: Front> sector.  
<TASK: RECONSTRUCT> <TARGET\_OBJ: computer workstation> <TARGET\_DIR: Back> Inspect the back of the workstation to examine its frame and lighting. The sequence … <END\_OBJ: computer workstation> position within the <END\_DIR: Back> sector of the panoramic scene.

![](images/0da9eeee5d215a98045fe5ece0a99b14f7b2031d1aaed3a77701616c4a7e01a4.jpg)

![](images/6714137d8053b21d5358bbeccfb1c0b973251bd7a27ad13f7217579517d62a41.jpg)

![](images/b96e6fd6163a40c2190a0df1f15cf19e615eb221329d52f52649edbdbf2ef05e.jpg)

![](images/676230d92988a34d2f692ade2b20dbf61b6087c7b45b2ca7df6d422d38b23950.jpg)

![](images/1aeb8e002c4ed1ae27884333f789ba2ae10c241097d9a3628252e07a57acb718.jpg)  
<TASK: TARGET> <TARGET\_OBJ: leather chair> <TARGET\_DIR: Front-Right> The intent …Front-Right through a push in, and finally turn right. The sequence concludes by arriving at the final <END\_OBJ: leather chair> position in the <END\_DIR: Front-Right> sector. Scenes CCD Director3D GenDoP OmniCam (Ours)

Figure 4. Qualitative comparison on representative panoramic scenes. Each column shows the trajectory produced by a specific method.   
The target object is indicated by a red bounding box.

Panorama

![](images/7840dabfd43d7de704f88adc8e5f7dbe3c1c51875f6f7fdd127c534c23c583cd.jpg)

![](images/310bff4aafe507c39800c40a32d40d7eb0248920f741c21899a9efd217afaab3.jpg)  
Camera-Controlled Video Generation with OmniCam Traj.  
3D Reconstruction 3D Reconstruction w/o OmniCam Traj. w/ OmniCam Traj.

Figure 5. Qualitative comparison on panoramic scenes. Each row displays an input panorama, generated video sequences, and 3D reconstruction results. The two rightmost columns compare reconstruction outputs without and with the OmniCam trajectory; no quantitative reconstruction metric is reported here.  
![](images/9388a171a26523695231867864b1af176b5f5a2535e71a06fda010fc3b215247.jpg)  
Locating and reaching an apple hidden behind a box

Figure 6. Real-robot demonstration of OmniCam-guided active perception. OmniCam plans a camera trajectory that actively repositions the observation camera to resolve target occlusion.

and angular deviation. These reference-trajectory errors measure agreement with the generated supervision; multiple distinct trajectories can satisfy the same instruction.

Implementation. OmniCam uses an OPT-style [35] decoder with FlashAttention-2 [9], trained in bf16 on 8×H20 GPUs for ∼100 epochs. The base configuration (hidden dim 1024, 8 heads, 12 layers; ∼500M trainable parameters) is used for the main comparisons; a larger variant (∼1B) is analyzed in Section E.2. Hyperparameters are provided in Section B.

## 5.2. Main Results

Quantitative Comparison. On OmniCaT (Table 1), OmniCam reduces ATE, FDE, RPE-R, and RPE-T by 46.6%, 41.1%, 28.0%, and 28.9% relative to retrained GenDoP, respectively. Collision decreases from 30.4% to 10.4% (65.8% relative reduction), and visibility rises from 33.6% to 52.0% (54.8% relative increase). Director3D is the best baseline for ATE and collision; relative to it, the reductions are 43.0% and 62.3%. The scene-bootstrap intervals for OmniCam and GenDoP are reported in Section M; component-specific effects are examined separately below.

Zero-Shot Cross-Dataset Transfer. The reported DataDoP results favor OmniCam over GenDoP trained on DataDoP: ATE 0.925 vs. 1.437, collision 8.9% vs. 28.5%, visibility 64.2% vs. 36.6%, and angle 12.7<sup>◦</sup> vs. 29.8<sup>◦</sup>. OmniCam is reported to use no DataDoP fine-tuning. Establishing the extent of transfer requires the input adaptation, data-overlap checks, and metric protocol identified above; these aggregate values alone do not rule out planner bias or differences in available scene information.

Table 2. Ablation studies on OmniCaT. The dual-branch panoramic encoder, target injection components, and pose tokenization schemes are investigated.
<table><tr><td></td><td>Variant</td><td>ATE↓</td><td>FDE↓</td><td>RPE-R↓</td><td>RPE-T↓</td><td>Coll. (%) ↓</td><td>Vis. (%) ↑</td><td>Angle (°) ↓</td></tr><tr><td rowspan="2">Encoder</td><td>w/o semantic branch</td><td>2.186</td><td>3.247</td><td>2.694</td><td>0.072</td><td>13.6</td><td>11.5</td><td>64.7</td></tr><tr><td>w/o geometric branch</td><td>1.774</td><td>3.084</td><td>2.881</td><td>0.049</td><td>45.8</td><td>33.6</td><td>19.7</td></tr><tr><td rowspan="2">Geo. Input</td><td>Panorama RGB only</td><td>1.536</td><td>2.687</td><td>2.143</td><td>0.044</td><td>38.7</td><td>36.2</td><td>18.4</td></tr><tr><td>Panorama RGB-D</td><td>1.247</td><td>2.118</td><td>1.685</td><td>0.036</td><td>22.6</td><td>42.8</td><td>14.7</td></tr><tr><td rowspan="3">Grounding</td><td>w/o target cross-attention</td><td>1.673</td><td>2.259</td><td>2.774</td><td>0.042</td><td>15.2</td><td>25.8</td><td>35.6</td></tr><tr><td>w/o target localization loss</td><td>1.582</td><td>2.148</td><td>2.661</td><td>0.038</td><td>14.8</td><td>32.4</td><td>28.7</td></tr><tr><td>w/o target cross-attn. &amp; loss</td><td>1.924</td><td>2.576</td><td>2.903</td><td>0.051</td><td>18.9</td><td>21.3</td><td>40.2</td></tr><tr><td rowspan="3">Tokenization</td><td>All absolute (rot. + trans.)</td><td>1.278</td><td>1.886</td><td>1.625</td><td>0.058</td><td>14.3</td><td>41.6</td><td>15.3</td></tr><tr><td>All relative (rot. + trans.)</td><td>1.449</td><td>1.927</td><td>1.984</td><td>0.062</td><td>16.6</td><td>43.2</td><td>16.4</td></tr><tr><td>Hybrid w/o hemisphere consistency</td><td>1.094</td><td>1.756</td><td>1.847</td><td>0.033</td><td>12.2</td><td>46.1</td><td>13.8</td></tr><tr><td>Full</td><td>OmniCam</td><td>0.868</td><td>1.462</td><td>1.135</td><td>0.027</td><td>10.4</td><td>52.0</td><td>10.9</td></tr></table>

Comparison with the Heuristic Planner. With access to reconstructed geometry, navigation meshes, and semantic annotations, the planner has 5.2% collision and 58.7% visibility, compared with OmniCam’s 10.4% and 52.0%. This is a privileged-information reference, not a mathematical performance bound. The reported runtimes are 0.84s and approximately 16.5s (Section J); matched timing conditions require verification.

Per-Behavior Analysis. Section D reports lower ATE and FDE for OmniCam in all four behavior categories. These results describe coverage of the evaluated behaviors without identifying which component causes each gain.

Qualitative Comparison. Figure 4 compares trajectories across representative scenes. OmniCam trajectories remain aligned with navigable regions, follow the intended viewpoint of the referenced object, and exhibit smoother transitions in both position and orientation compared to all baselines. Additional qualitative examples are provided in Section K.

## 5.3. Ablation Studies

Dual-Branch Encoder. Removing the semantic branch reduces visibility from 52.0% to 11.5%, while removing the geometric branch increases collision from 10.4% to 45.8%. This supports the usefulness of both pathways. Because branch removal also changes capacity and available information, a capacity-matched single-branch comparison is needed to isolate the effect of routing.

Geometric Input Modality. RGB-only, RGB-D, and pointcloud variants yield collision rates of 38.7%, 22.6%, and 10.4%, respectively. The results favor explicit 3D processing in these configurations. Since the point cloud is unprojected from depth, this comparison concerns representation and processing, rather than additional geometric information created by unprojection.

Target Centroid Injection. Removing target cross-attention changes visibility from 52.0% to 25.8% and angle from 10.9<sup>◦</sup> to 35.6<sup>◦</sup>. Removing the localization loss yields 32.4% and 28.7<sup>◦</sup>; removing both yields 21.3% and 40.2<sup>◦</sup>, the worst values among these grounding variants. The comparison supports explicit target anchoring in the tested architecture (Section G).

Hybrid Pose Encoding. Hybrid encoding obtains RPE-R 1.135 and RPE-T 0.027, compared with 1.625/0.058 for all-absolute and 1.984/0.062 for all-relative encoding. Removing sign alignment increases RPE-R to 1.847. The representation analysis in Section I concerns quantization and sign selection; it does not guarantee the accuracy of autoregressive predictions.

## 5.4. Downstream Application Validation

With WorldStereo [36], OmniCam trajectories improve the reported CLIP score by 14.1% and consecutive-frame SSIM by 15.4% relative to GenDoP. Figure 5 illustrates video and reconstruction outputs. Robotic active-perception results and a qualitative real-robot demonstration are presented in Section L and Fig. 6. The robot success-rate aggregation needs verification; the demonstration is evidence of feasibility rather than a systematic sim-to-real evaluation.

## 6. Conclusion

OmniCam generates camera trajectories from panoramic imagery and language using point-cloud conditioning, hybrid pose serialization, and explicit 3D target anchoring.

OmniCaT supplies 267,700 examples across four camera behaviors. The reported comparisons and ablations support these design choices within the evaluated settings, and downstream studies examine their practical use. The representation analysis distinguishes quaternion sign consistency from lossy quantization and identifies the error-accumulation tradeoff of relative encoding. Remaining limitations include estimated and incomplete geometry, planner-derived supervision, static-scene observations, and the need for independently verified evaluation protocols. Future work includes dynamic scenes and closed-loop planning.

## References

[1] Eloi Alonso, Adam Jelley, Vincent Micheli, Anssi Kanervisto, Amos Storkey, Tim Pearce, and François Fleuret. Diffusion for world modeling: Visual details matter in Atari. Advances in Neural Information Processing Systems, 37:58757–58791, 2024.

[2] Amir Bar, Gaoyue Zhou, Danny Tran, Trevor Darrell, and Yann LeCun. Navigation world models. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 15791–15801, 2025.

[3] Jim Blinn. Where am i? what am i looking at?(cinematography). IEEE Computer Graphics and Applications, 8(4):76–81, 2002.

[4] Rogerio Bonatti, Wenshan Wang, Cherie Ho, Aayush Ahuja, Mirko Gschwindt, Efe Camci, Erdal Kayacan, Sanjiban Choudhury, and Sebastian Scherer. Autonomous aerial cinematography in unstructured environments with learned artistic decision-making. Journal ofField Robotics, 37(4):606–641, 2020.

[5] Zhe Cao, Hang Gao, Karttikeya Mangalam, Qi-Zhi Cai, Minh Vo, and Jitendra Malik. Long-term human motion prediction with scene context. In European Conference on Computer Vision, pages 387–404. Springer, 2020.

[6] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. SAM 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719, 2025.

[7] Robin Courant, Nicolas Dufour, Xi Wang, Marc Christie, and Vicky Kalogeiton. ET the exceptional trajectories: Text-tocamera-trajectory generation with character awareness. In European Conference on Computer Vision, pages 464–480. Springer, 2024.

[8] Qiongjie Cui, Huaijiang Sun, and Fei Yang. Learning dynamic relationships for 3D human motion prediction. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 6519–6527, 2020.

[9] Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691, 2023.

[10] Steven M Drucker, Tinsley A Galyean, and David Zeltzer. Cinema: A system for procedural camera movements. In Proceedings of the 1992 symposium on Interactive 3D graphics, pages 67–70, 1992.

[11] Quentin Galvane, Marc Christie, Chrsitophe Lino, and Rémi Ronfard. Camera-on-rails: automated computation of constrained camera paths. In Proceedings of the 8th ACM SIG-GRAPH Conference on Motion in Games, pages 151–157, 2015.

[12] Michael Grupp. evo: Python package for the evaluation of odometry and SLAM. https://github.com/ MichaelGrupp/evo, 2017.

[13] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. CameraCtrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101, 2024.

[14] Yunzhong Hou, Liang Zheng, and Philip Torr. Learning camera movement control from real-world drone videos. arXiv preprint arXiv:2412.09620, 2024.

[15] Chong Huang, Chuan-En Lin, Zhenyu Yang, Yan Kong, Peng Chen, Xin Yang, and Kwang-Ting Cheng. Learning to film from professional human motion videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4244–4253, 2019.

[16] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. π<sub>0.5</sub>: a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.

[17] Hongda Jiang, Bin Wang, Xi Wang, Marc Christie, and Baoquan Chen. Example-driven virtual cinematography by learning camera behaviors. ACM Trans. Graph., 39(4):45, 2020.

[18] Hongda Jiang, Xi Wang, Marc Christie, Libin Liu, and Baoquan Chen. Cinematographic camera diffusion model. Computer Graphics Forum, 43(2):e15055, 2024.

[19] Michael Kazhdan, Thomas Funkhouser, and Szymon Rusinkiewicz. Rotation invariant spherical harmonic representation of 3 d shape descriptors. In Symposium on geometry processing, pages 156–164, 2003.

[20] Beomyoung Kim, Chanyong Shin, Joonhyun Jeong, Hyungsik Jung, Se-Yun Lee, Sewhan Chun, Dong-Hyun Hwang, and Joonsang Yu. ZIM: Zero-shot image matting for anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23828–23838, 2025.

[21] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. OpenVLA: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.

[22] Soohwan Kim and Minkyoung Kim. Rotation representations and their conversions. Ieee Access, 11:6682–6699, 2023.

[23] Xinyang Li, Zhangyu Lai, Linning Xu, Yansong Qu, Liujuan Cao, Shengchuan Zhang, Bo Dai, and Rongrong Ji. Director3D: Real-world camera trajectory and 3D scene generation from text. Advances in neural information processing systems, 37:75125–75151, 2024.

[24] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, et al. DL3DV-10K: A large-scale scene dataset for deep learningbased 3D vision. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22160– 22169, 2024.

[25] Christophe Lino and Marc Christie. Intuitive and efficient camera control with the toric space. ACM Transactions on Graphics (TOG), 34(4):1–12, 2015.

[26] Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, and Jun Zhu. RDT-1B: a diffusion foundation model for bimanual manipulation. arXiv preprint arXiv:2410.07864, 2024.

[27] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding DINO: Marrying DINO with grounded pre-training for open-set object detection. In European conference on computer vision, pages 38–55. Springer, 2024.

[28] Xinyi Liu, Tianyi Zhang, Matthew Johnson-Roberson, and Weiming Zhi. SplaTraj: Camera trajectory generation with semantic gaussian splatting. arXiv preprint arXiv:2410.06014, 2024.

[29] Sebastian Pütz, Thomas Wiemann, Jochen Sprickerhof, and Joachim Hertzberg. 3D navigation mesh generation for path planning in uneven terrain. Ifac-Papersonline, 49(15):212– 217, 2016.

[30] Fabio Ruetz, Emili Hernández, Mark Pfeiffer, Helen Oleynikova, Mark Cox, Thomas Lowe, and Paulo Borges. OVPC mesh: 3D free-space representation for local ground vehicle navigation. In 2019 International Conference on Robotics and Automation (ICRA), pages 8648–8654. IEEE, 2019.

[31] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

[32] Ruicheng Wang, Sicheng Xu, Yue Dong, Yu Deng, Jianfeng Xiang, Zelong Lv, Guangzhong Sun, Xin Tong, and Jiaolong Yang. MoGe-2: Accurate monocular geometry with metric scale and sharp details. arXiv preprint arXiv:2507.02546, 2025.

[33] Xianggang Yu, Mutian Xu, Yidan Zhang, Haolin Liu, Chongjie Ye, Yushuang Wu, Zizheng Yan, Chenming Zhu, Zhangyang Xiong, Tianyou Liang, et al. MVImgNet: A large-scale dataset of multi-view images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9150–9161, 2023.

[34] Mengchen Zhang, Tong Wu, Jing Tan, Ziwei Liu, Gordon Wetzstein, and Dahua Lin. GenDoP: Auto-regressive camera trajectory generation as a director of photography. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 18229–18239, 2025.

[35] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. OPT: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022.

[36] Yisu Zhang, Chenjie Cao, Tengfei Wang, Xuhui Zuo, Junta Wu, Jianke Zhu, and Chunchao Guo. WorldStereo: Bridging camera-guided video generation and scene reconstruction via 3D geometric memories. arXiv preprint arXiv:2603.02049, 2026.

[37] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: Learning view synthesis using multiplane images. arXiv preprint arXiv:1805.09817, 2018.

[38] Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, and Hao Li. On the continuity of rotation representations in neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5745–5753, 2019.

## A. Implementation Details of OmniCaT

## A.1. Geometric Initialization and Refinement

Depth Alignment Optimization. During geometry initialization, the ERP space is subdivided into 42 perspective views (an increase from the default 12) to ensure dense coverage. Depth maps are aligned via a GPU-accelerated Least-Squares Minimal Residual (LSMR) solver. To mitigate artifacts, a grounding pipeline is employed to mask sky regions and to remove depth discontinuities (edge floaters). NavMesh Engineering. The raw NavMesh generated by the Recast algorithm is configured with cell size 0.1 m, cell height 0.1 m, agent height 0.1 m, agent radius 0.05 m, maximum climb 0.05 m, and maximum slope $3 0 ^ { \circ }$ . Postprocessing consists of: (1) height filtering to remove polygons above a roof threshold; (2) connectivity filtering to retain only the largest connected component; (3) surface snapping through dense ray-casting to correct misaligned vertices that lie below the actual mesh surface; and (4) boundary erosion using KD-Tree accelerated searches to maintain a safety buffer from mesh edges. The resulting navigation graph is constructed by sampling points on the NavMesh surface at 0.05 m spacing (up to 600K nodes), connecting neighbors within 5× spacing with a height coherence constraint $( | \Delta y | \le 0 . 5 \mathrm { m } )$ , and computing shortest paths via Dijkstra from the camera origin.

## A.2. Trajectory Planning Heuristics

Scoring Function for Target Paths. For target trajectories, 72 candidate nodes are sampled on a circular boundary at radius $r = \mathrm { m i n } ( 1 . 5 \times s _ { o b j } , d _ { o b j } )$ , where $s _ { o b j }$ is the object scale and $d _ { o b j }$ is the distance from the camera origin to the target. The optimal node is selected by maximizing a scoring function: $S = 5 0 0 0 \cdot R _ { l o s } - 1 0 0 \cdot \Delta \theta .$ , where $R _ { l o s }$ is the Line-of-Sight ratio calculated via a 50-point projection test between the candidate position and the target center, and $\Delta \theta$ is the angular deviation from the ideal approach direction. Paths are computed via Dijkstra on the NavMesh graph, pruned of backtracking segments, and smoothed with Bspline interpolation (smoothing factor 0.5). All trajectories are resampled to 81 keyframes.

Surround Behavior. For surround trajectories, the orbit radius is computed as $r = \operatorname* { m i n } ( d _ { o b j } , 2 \times s _ { o b j } )$ , clamped by a scene-dependent threshold $( 4 \times$ global median depth). 72 points are sampled on the orbit circle; only those reachable via the NavMesh graph are connected to form continuous surround paths.

Exploration Behavior. For wander trajectories, 8 radial directions are explored from the origin along the NavMesh, each extending up to 4× global median depth in path length. The top 10 trajectories are selected via Farthest Point Sampling on endpoint positions, with a minimum angular separation of $4 0 ^ { \circ }$ and maximum overlap ratio of 0.5.

Reconstruction Behavior. Reconstruction targets are selected via the top-10 semantic objects ranked by segment area and viewpoint diversity (FPS in 3D space). For each target, multi-view paths are planned with a reduced smoothing factor (0.2) to preserve sharp viewpoint transitions optimal for 3D reconstruction coverage.

Path Smoothing and Orientation. All discrete waypoints are smoothed using B-splines and Gaussian filtering $( \sigma =$ 8.0). Camera orientations are managed via Spherical Linear Interpolation (SLERP). For “Wander” and “Target” modes, orientations follow the path tangent and target centroid, respectively.

## B. Training and Inference Details

The reported tokenizer uses $B = 2 5 6$ bins and a normalized displacement limit $\Delta t _ { \mathrm { m a x } } = 0 . 0 2 5$ . These settings determine precision and clipping tradeoffs; sign alignment does not remove quantization boundaries.

OmniCam is trained with the Accelerate framework under bf16 mixed precision. The autoregressive backbone is an OPT-style decoder equipped with FlashAttention-2. The optimizer is AdamW with $\beta = ( 0 . 9 , 0 . 9 5 )$ , a weight decay of 0.01, a peak learning rate of $3 \times 1 0 ^ { - 5 }$ , a linear warmup over the first one percent of the schedule, and cosine annealing for the remaining steps. The gradient is clipped to a global norm of 1.0 and gradient checkpointing is enabled throughout the decoder. The discrete bin size is $B = 2 5 6$ and the maximum translation delta is $\Delta t _ { \mathrm { m a x } } = 0 . 0 2 5$ in the normalized coordinate system. The hidden dimension, number of heads, and number of layers are configured as the base variant (hidden dim 1024, 8 heads, 12 layers). Training is performed for approximately one hundred effective epochs on a single node with eight NVIDIA H20 GPUs, with a per-GPU batch size of thirty-two. The localization loss weight is set to $\lambda = 0 . 5 .$ . Inference adopts the sample strategy with top-k truncation at $k = 1 0$ and with the format-constrained decoding procedure described in Section 4.4.

Weighted Cross-Entropy Loss. The per-token crossentropy loss is weighted by three multiplicative factors to emphasize the most critical pose channels and learning phases: (1) channel importance weights: rotation channels receive weight 2.0, translation channels 1.5, and intrinsic channels 0.5, reflecting the relative importance of orientation accuracy over positional precision; (2) temporal decay: a linear schedule assigns weight 2.0 to the first frame and decreases to 1.0 at the final frame, prioritizing early trajectory establishment; and (3) curriculum learning mask: during the initial training phase, only early frames contribute to the loss, with later frames gradually enabled as training progresses, preventing the model from attempting to learn long-horizon dynamics before mastering short-horizon patterns.

Caption Dropout Probabilities. The five-variant caption dropout strategy applies the following distribution: full caption (35%), prefix tags only (25%), body text only (25%), motion tag only (5%), and partial prefix (10%). This is the distribution used in the reported experiments.

Computational Cost. Table 3 reports training and inference costs for GenDoP and two OmniCam configurations. The original timing description includes depth estimation, pointcloud construction, encoding, and autoregressive decoding. The prefix condition is reused across decoding steps. These reported timings do not establish real-time operation under an unspecified deployment budget.

## C. Encoder Architecture Details

The dual-branch encoder routes appearance and text through a semantic pathway and point-cloud features through a geometry pathway. The outputs are fused by the decoder.

Semantic Branch. The panoramic image $\mathbf { I } _ { p a n o }$ is processed by a frozen SAM 3 vision encoder with 1024-dimensional features. The target descriptor $T _ { o b j }$ and caption guide the Language-Guided Resampler after visual feature extraction. This pathway provides target-related information, but it is not excluded from contributing to geometric decisions after decoder fusion.

Geometric Branch. Panoramic geometric priors are unprojected into a 3D point cloud, which is then structured into a voxel grid (0.02m resolution) to enable efficient feature extraction. A LitePT encoder uses positional and surface-normal features to extract per-voxel descriptors via sparse convolution, which are re-projected to yield ${ \bf F } _ { g e o } \in { \bf \Sigma }$ $\bf { \bar { \mathbb { R } } } ^ { M \times C _ { g e o } }$ . This branch has no direct text input. Removing it increases collision rate from 10.4% to 45.8% in Table 2; this supports the usefulness of the geometric pathway but does not isolate routing from capacity or input-information changes.

## D. Per-Behavior Decomposition

Table 4 decomposes the OmniCaT results. Compared with retrained GenDoP, OmniCam has lower ATE for Wander (1.150 vs. 2.421), Target (0.677 vs. 1.088), Surround (1.258 vs. 1.932), and Reconstruct (0.385 vs. 1.055). The corresponding Target and Reconstruct reductions are 37.8% and 63.5%. This comparison supports performance across the four evaluated behaviors; it does not isolate caption dropout, loss weighting, or geometry as the cause of individual gains.

## E. Scaling Analysis

We examine the reported dependence on training-data volume and model size. These finite comparisons show trends within the evaluated range; they do not establish a scaling law or the absence of representational bottlenecks.

## E.1. Data Scaling

Table 5 reports the base model at four nominal data scales, labeled 27K, 53K, 107K, and 267K in the original experiment record. Metrics improve across these settings: ATE decreases from 1.483 to 0.868, collision from 19.8% to 10.4%, and visibility increases from 34.6% to 52.0%. Four data points do not determine a power law or establish gains beyond the observed range.

## E.2. Model Scaling

Three model sizes are compared: Small (hidden dimension 512, 8 heads, 8 layers; approximately 120M trainable parameters), Base (1024, 8, 12; approximately 500M), and Large (1536, 16, 24; approximately 1B). The reported setup holds other hyperparameters constant while changing architecture dimensions. ATE decreases from 1.127 to 0.743, collision from 13.8% to 8.7%, and visibility increases from 44.6% to 56.3% (Table 6). These observations support gains at the tested sizes, without establishing which representational choices explain the trend.

## F. Failure Case Analysis

Understanding when and why the framework fails is essential for guiding future improvements. Five categories of failures are identified, each linked to a specific component or assumption of the pipeline:

• Noisy panoramic geometry / thin structures (geometric branch limitation): in large open environments or near thin structures (fences, poles, railings), the monocular depth estimator [32] produces noisy or incomplete reconstructions with floating artifacts. These propagate into the LitePT voxel features, causing the geometric branch to either miss collision boundaries or hallucinate obstacles, resulting in physically implausible paths.

• Ambiguous target phrases (centroid injection limitation): when multiple semantically similar objects coexist (e.g., “the wooden door” in a hallway with several doors, or “the pillar” among clustered columns), the attention-aggregated centroid may ground to an incorrect instance. Spatial descriptors in the textual guidance partially mitigate this, but the failure persists when such cues are absent or when objects are visually indistinguishable from the panoramic viewpoint.

• Dynamic and interactive scenes (static-scene assumption): the current framework assumes a static panoramic observation. In environments with moving objects (people, vehicles, opening doors), the generated trajectory cannot react to temporal changes, potentially producing paths that collide with dynamic obstacles or fail to track moving targets. Extending OmniCam to video-conditioned or closed-loop settings is an important direction for future work.

Table 3. Reported computational costs for GenDoP and OmniCam Base/Large.
<table><tr><td>Method</td><td>Training Time</td><td>Inference Latency</td><td>#Params</td></tr><tr><td>GenDoP</td><td>~18h (8×H20)</td><td>0.42s / trajectory</td><td>~380M</td></tr><tr><td>OmniCam (Base)</td><td>~24h (8×H20)</td><td>0.84s / trajectory</td><td>~500M</td></tr><tr><td>OmniCam (Large)</td><td>~40h (8×H20)</td><td>1.20s / trajectory</td><td>~1B</td></tr></table>

Table 4. Per-behavior decomposition of the main result on OmniCaT. Metrics are reported separately for Wander, Target, Surround, and Reconstruct trajectories.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Dataset</td><td colspan="2">Wander</td><td colspan="2">Target</td><td colspan="2">Surround</td><td colspan="2">Reconstruct</td></tr><tr><td>ATE↓</td><td>FDE↓</td><td>ATE↓</td><td>FDE↓</td><td>ATE↓</td><td>FDE↓</td><td>ATE↓</td><td>FDE↓</td></tr><tr><td>CCD [18]</td><td>Pre-trained</td><td>3.718</td><td>4.907</td><td>1.485</td><td>2.285</td><td>2.524</td><td>3.419</td><td>2.206</td><td>2.972</td></tr><tr><td>E.T. [7]</td><td>Pre-trained</td><td>3.151</td><td>4.083</td><td>1.735</td><td>1.945</td><td>1.580</td><td>1.640</td><td>1.467</td><td>1.804</td></tr><tr><td>Director3D [23]</td><td>Pre-trained</td><td>2.969</td><td>4.779</td><td>1.444</td><td>2.207</td><td>1.620</td><td>2.688</td><td>1.279</td><td>2.094</td></tr><tr><td>Director3D</td><td>OmniCaT</td><td>2.562</td><td>4.250</td><td>1.130</td><td>2.089</td><td>1.351</td><td>2.304</td><td>1.045</td><td>1.950</td></tr><tr><td>GenDoP [34]</td><td>Pre-trained</td><td>2.667</td><td>4.408</td><td>1.362</td><td>2.131</td><td>2.176</td><td>3.248</td><td>1.255</td><td>2.016</td></tr><tr><td>GenDoP</td><td>OmniCaT</td><td>2.421</td><td>3.652</td><td>1.088</td><td>1.933</td><td>1.932</td><td>2.773</td><td>1.055</td><td>1.574</td></tr><tr><td>OmniCam (Ours)</td><td>OmniCaT</td><td>1.150</td><td>2.301</td><td>0.677</td><td>0.962</td><td>1.258</td><td>1.836</td><td>0.385</td><td>0.749</td></tr></table>

• Reflective/transparent surfaces (semantic branch limitation): glass walls and mirrors produce ambiguous visual features in the vision encoder, leading to unreliable targetaware conditioning and erratic trajectory orientation near such surfaces.

• Occluded targets (centroid injection limitation): when the target object is heavily occluded, the cross-attention weights spread to visible but irrelevant regions, causing the estimated centroid to shift away from the true target location.

These cases identify limitations in perception and target grounding. Decoder errors, quantization, and exposure to planner-generated supervision can also contribute; the examples do not isolate their relative importance.

Broader Impacts. Scene-aware camera planning may assist content creation and robotic perception. Camera deployment can also raise privacy or surveillance concerns, and generated video may be used misleadingly. In robotics, errors in inferred geometry or open-loop trajectories can produce unsafe camera motions; the reported collision metrics are not deployment guarantees. Dataset release requires sourcespecific permission and license review, and physical deployment requires validation against the actual environment.

## G. Attention Visualization and Centroid Analysis

The target centroid supplies an explicit spatial anchor (Section 4.3). The following visualizations illustrate the attention patterns used to derive this anchor.

Figure 7 shows the input panorama, reconstructed point cloud, and attention projected onto scene points. The examples show concentration near referenced regions. These visualizations complement the grounding ablation, but do not quantify localization accuracy or establish that attention alone causes improved visibility. Point-level aggregation is specified conceptually in Equation (2); its implementation details require verification.

## H. Caption Dropout Ablation

We compare no caption dropout, single-variant dropout (dropping the entire caption with probability 0.1), and the five-variant distribution in Section B. The reported values are means over three training seeds, with ATE standard deviation below 0.02. The five-variant scheme has higher visibility than no dropout (52.0% vs. 38.7%) and lower angular deviation (10.9<sup>◦</sup> vs. 18.4<sup>◦</sup>), while no dropout has slightly lower ATE (0.856 vs. 0.868) and collision (10.1% vs. 10.4%). This is a tradeoff between the reported metrics. Attribution to instruction-format overfitting requires separate evaluation on complete, partial, and rephrased instructions.

## I. Theoretical Analysis of Hybrid Pose Tokenization

We analyze reconstruction of a known trajectory after quantization and dequantization. These results concern representation error; they do not bound autoregressive prediction error, depth-estimation error, or errors in scene-scale calibration. Let d denote the rotation-angle geodesic distance on SO(3), measured in radians, with range [0, π]. This distance

Building

![](images/a89730b494c505a4043624717fa32336e26b9ecc92bdb6286dd060f0add1c4cd.jpg)

![](images/1c091343c94895d65e1569d8bb456f81721667b374835270fe7f8e3f3f4333b0.jpg)  
Street Light

![](images/38caa926fa186a25017fa38e71d1bb6c7a378418b5911e4adf1a81db923e45b6.jpg)

Table 5. Reported effect of nominal data scale using the base architecture. Exact training-subset counts and the fixed evaluation split require verification.
<table><tr><td>#Trajectories</td><td>ATE↓</td><td>FDE↓</td><td>RPE-R↓</td><td>RPE-T↓</td><td>Coll. (%)↓</td><td>Vis. (%) ↑</td></tr><tr><td>27K (10%)</td><td>1.483</td><td>2.586</td><td>1.836</td><td>0.054</td><td>19.8</td><td>34.6</td></tr><tr><td>53K (20%)</td><td>1.206</td><td>2.080</td><td>1.528</td><td>0.042</td><td>15.8</td><td>42.6</td></tr><tr><td>107K (40%)</td><td>1.003</td><td>1.709</td><td>1.297</td><td>0.033</td><td>12.7</td><td>48.2</td></tr><tr><td>267K (100%)</td><td>0.868</td><td>1.462</td><td>1.135</td><td>0.027</td><td>10.4</td><td>52.0</td></tr></table>

Table 6. Reported effect of model scale. The common training subset and parameter-count scope require verification against training records.
<table><tr><td>Size</td><td>#Params</td><td>ATE↓</td><td>FDE↓</td><td>RPE-R↓</td><td>RPE-T↓</td><td>Coll. (%) ↓</td><td>Vis. (%) ↑</td></tr><tr><td>Small</td><td>~120M</td><td>1.127</td><td>1.893</td><td>1.428</td><td>0.036</td><td>13.8</td><td>44.6</td></tr><tr><td>Base</td><td>~500M</td><td>0.868</td><td>1.462</td><td>1.135</td><td>0.027</td><td>10.4</td><td>52.0</td></tr><tr><td>Large</td><td>~1B</td><td>0.743</td><td>1.248</td><td>0.982</td><td>0.022</td><td>8.7</td><td>56.3</td></tr></table>

![](images/3d00ecda635856ea728d22a69f1d92592e48f0745bad618a0783426f5288230e.jpg)  
Door

![](images/a05c37d1432ad923383286e46a1c75ac013b9d729d1f0a2c669820022ec5856a.jpg)

![](images/6527741c22f413198ffed9a6e97cd7bbcd3530a3f74c5e6b419ebe10f190d60b.jpg)

![](images/4751eac40e237dc27aabf9aeef6eb356c4bd35f052ce1f1bac5701d08c670fe7.jpg)

![](images/d2f9de5397c48c14496d448dfa5c8102d19832a11f900456d6838d68d40107fd.jpg)

Door  
![](images/17ee22cb3bac45d03bae99e79eca20fa859bc49367f8df1f15c45d3c3ddf177c.jpg)

![](images/8980810dfe4e5d3093e5233ed32ebf80e1c8da6f001499dc0502a2f9cfe5ba50.jpg)

![](images/2fb26a143d4b9c574570d023344a265fd83532d2569d3d8642a186267c4a1f95.jpg)

![](images/f3caf9f6690e081d8e49a76a8d774950baa98ba5bcd9b3194bdf59a77037e8d7.jpg)

![](images/45025a9784545ce0382a842f0e0133c846a37b8e10a7793af26b8048cccc4750.jpg)

![](images/b667282ba0586c41f41e434387bba0fba8ed2120e40a1a16fcb495fbe3beb26b.jpg)  
Figure 7. Attention visualization of the Language-Guided Semantic Resampler. Each group shows (from left to right) the input panorama, the reconstructed point cloud, and the 3D attention map. The maps visualize target-related attention on reconstructed geometry for the displayed examples; they are qualitative illustrations rather than a quantitative localization evaluation.

Table 7. Caption-dropout ablation. The five-variant scheme favors visibility and angle, while no dropout has slightly lower geometric errors.
<table><tr><td>Dropout Strategy</td><td>ATE↓</td><td>FDE↓</td><td>RPE-R↓</td><td>RPE-T↓</td><td>Coll. (%)↓</td><td>Vis. (%) ↑</td><td>Angle (°) ↓</td></tr><tr><td>No dropout</td><td>0.856</td><td>1.438</td><td>1.118</td><td>0.026</td><td>10.1</td><td>38.7</td><td>18.4</td></tr><tr><td>Single-variant</td><td>0.862</td><td>1.451</td><td>1.126</td><td>0.027</td><td>10.3</td><td>45.8</td><td>14.2</td></tr><tr><td>Full (5-variant)</td><td>0.868</td><td>1.462</td><td>1.135</td><td>0.027</td><td>10.4</td><td>52.0</td><td>10.9</td></tr></table>

is invariant under left and right multiplication by a rotation.

Proposition 1 (Deterministic bounds on rotation reconstruction error). Let $R _ { 1 } , \ldots , R _ { T } \in \mathrm { S O } ( 3 )$ and $\Delta R _ { t } = R _ { t - 1 } ^ { - 1 } R _ { t }$ $\it { f o r \ t \geq 2 }$ . Suppose decoded relative rotations satisfy $d ( \widehat { \Delta R } _ { t } , \Delta R _ { t } ) \leq \epsilon _ { t } ,$ , and the decoded initial rotation satisfies d( $\widehat { R } _ { 1 } , R _ { 1 } ) \leq e _ { 1 }$ . Reconstruct relative orientations by $\widehat { R } _ { t } = \widehat { R } _ { t - 1 } \widehat { \Delta R } _ { t }$ . Then

$$
d ( \widehat { R } _ { t } , R _ { t } ) \leq \operatorname* { m i n } \left\{ \pi , \ : e _ { 1 } + \sum _ { j = 2 } ^ { t } \epsilon _ { j } \right\} .\tag{4}
$$

If $\widehat { R } _ { 1 } ~ = ~ R _ { 1 }$ and every $\epsilon _ { j } \leq \epsilon ,$ , this bound becomes min $\{ \pi , ( t - 1 ) \epsilon \}$ . In contrast, if absolute orientations are decoded with per-frame reconstruction error at most ϵ, then $d ( \widehat { R } _ { t } ^ { \mathrm { a b s } } , R _ { t } ) \leq$ ϵ at everyframe, without integrating earlier quantization residuals.

Proof. The triangle inequality and bi-invariance give

$$
\begin{array} { r l } & { d ( \widehat { R } _ { t } , R _ { t } ) \leq d ( \widehat { R } _ { t - 1 } \widehat { \Delta R } _ { t } , \widehat { R } _ { t - 1 } \Delta R _ { t } ) } \\ & { \phantom { \quad \ } + d ( \widehat { R } _ { t - 1 } \Delta R _ { t } , R _ { t - 1 } \Delta R _ { t } ) } \\ & { \phantom { \quad \ } = d ( \widehat { \Delta R } _ { t } , \Delta R _ { t } ) + d ( \widehat { R } _ { t - 1 } , R _ { t - 1 } ) . } \end{array}\tag{5}
$$

(6)

Iterating this inequality proves the sum bound. Every rotation-angle distance is at most π, giving the stated minimum. The absolute-encoding bound follows directly from its assumed per-frame reconstruction guarantee. □

This is a worst-case upper bound, not a claim that relative errors necessarily grow at each frame or obey a particular expected growth rate. Residuals can cancel. Absolute encoding avoids integration of rotation quantization residuals, but predicted absolute tokens can still depend on previous generated tokens and therefore need not have independent prediction errors.

Connecting component quantization to angular error. The bound ϵ depends on the quantizer and the rotation decoder; B component bins do not by themselves imply $\epsilon = \pi / B$ . For a conditional example, let $q \in \mathbb { S } ^ { 3 }$ be a unit quaternion, and suppose component dequantization produces v with $\| v - q \| _ { 2 } \leq \eta < 1$ . If the decoder normalizes v to $\widehat { q } = v / \| v \| _ { 2 }$ , then

$$
d ( R ( \widehat { q } ) , R ( q ) ) \leq 2 \arcsin \eta .\tag{7}
$$

Indeed, $q ^ { \top } v > 0$ , so the angle $\phi$ between q and v lies in $[ 0 , \pi / 2 )$ . Decomposing the error along qb gives $\| v - q \| _ { 2 } ^ { 2 } =$ $( \bar { \| } v \| _ { 2 } - \cos \phi ) ^ { 2 } + \sin ^ { 2 } \phi ,$ hence $\phi \leq$ arcsin η. The corresponding rotation angle is $2 \phi$ . For uniform bins of width h, center decoding without clipping gives at most $h / 2$ error per quaternion component and thus $\eta \leq h$ . In particular, $B > 2$ equal bins on $[ - 1 , 1 ]$ give the conditional bound $2 \arcsin ( 2 / B )$ . Applying such a numerical bound requires these decoder and range assumptions to hold.

Proposition 2 (Representative-independent quaternion sign selection). Let $R _ { 1 } , \ldots , R _ { T }$ satisfy $d ( R _ { t - 1 } , R _ { t } ) \ < \ \pi \ f o r$ every $t \geq 2 ,$ , and choose arbitrary unit-quaternion representatives $q _ { t }$ of $R _ { t }$ . For $\boldsymbol { q } ~ = ~ ( w , x , y , z )$ , let $c ( q )$ select the signfor which thefirst nonzero component in the order $( w , x , y , z )$ is positive. Define

$$
\begin{array} { c c } { { } } & { { \widetilde { q } _ { 1 } = c ( q _ { 1 } ) , } } \\ { { } } & { { } } \\ { { \widetilde { q } _ { t } = \left\{ \begin{array} { l l } { { q _ { t } , } } & { { \langle \widetilde { q } _ { t - 1 } , q _ { t } \rangle > 0 , } } \\ { { - q _ { t } , } } & { { \langle \widetilde { q } _ { t - 1 } , q _ { t } \rangle < 0 , } } \end{array} \right. } } & { { t \geq 2 . } } \end{array}\tag{8}
$$

This produces a unique aligned quaternion sequence determined by the rotation sequence, independently ofthe initial choices between $q _ { t }$ and $- q _ { t } .$ . Consequently, any deterministic component quantizer withfixed bin-boundary rules produces the same token sequence from all such choices of input representatives. The tokenization remains many-to-one.

Proof. A unit quaternion has a nonzero component, and $c ( q ) = c ( - q )$ , so the first representative is uniquely fixed, including when $w _ { 1 } = 0$ . Suppose the aligned representative at $t - 1$ is fixed. The quaternion formula for rotation distance gives

$$
| \langle \widetilde { q } _ { t - 1 } , q _ { t } \rangle | = \cos \left( \frac { d ( R _ { t - 1 } , R _ { t } ) } { 2 } \right) > 0 .
$$

Exactly one of $q _ { t }$ and $- q _ { t }$ therefore has positive inner product with the already aligned previous quaternion. Replacing the input $q _ { t } \ b y - q _ { t }$ reverses the inner-product sign and leaves the selected representative unchanged. Induction proves representative independence of the entire sequence, and deterministic quantization preserves this property. It does not preserve injectivity: a finite token vocabulary cannot distinguish every continuous rotation sequence. □

The proposition specifies sufficient sign and boundary conventions; a concrete implementation must document these choices. At an exactly $\pi$ inter-frame rotation, both inner products are zero and an additional deterministic tie rule is required, such as selecting $c ( q _ { t } )$ . No shortest-arc preference exists at that tie. This sequence-level sign selection removes double-cover ambiguity under the stated conventions; it does not construct a globally continuous quaternion parameterization of $\mathrm { S O ( 3 ) }$ , remove quantization boundaries, or make lossy quantization invertible.

Translation: local resolution versus accumulated error. For additive displacements in a common coordinate frame, relative translation trades a smaller encoding range for integration error. If normalized absolute coordinates lie in $[ - L , L ]$ and displacements lie in $[ - D , D ]$ per component, uniform center decoding with B bins and no clipping bounds the per-frame absolute position error by ${ \sqrt { 3 } } L / B$ and each displacement error by ${ \sqrt { 3 } } D / B$ . Integrating the latter gives

Table 8. Privileged-information planner comparison. The planner receives reconstructed geometry, navigation meshes, and semantic annotations. OmniCam estimates geometry from the input panorama and text.
<table><tr><td>Method</td><td>NavMesh/annotation access</td><td>Runtime</td><td>Coll. (%) ↓</td><td>Vis. (%) ↑</td><td>Angle (°) ↓</td></tr><tr><td>Heuristic Planner</td><td>Yes (privileged)</td><td>~16.5s</td><td>5.2</td><td>58.7</td><td>9.4</td></tr><tr><td>GenDoP [34]</td><td>No</td><td>0.42s</td><td>30.4</td><td>33.6</td><td>27.6</td></tr><tr><td>OmniCam (Ours)</td><td>No</td><td>0.84s</td><td>10.4</td><td>52.0</td><td>10.9</td></tr></table>

$$
\| \widehat { \mathbf { p } } _ { t } - \mathbf { p } _ { t } \| _ { 2 } \leq \| \widehat { \mathbf { p } } _ { 1 } - \mathbf { p } _ { 1 } \| _ { 2 } + ( t - 1 ) \frac { \sqrt { 3 } D } { B } .\tag{9}
$$

Thus $D \ll L$ improves local displacement resolution, while long sequences can accumulate translation error. If displacements are expressed in rotating local frames, orientation errors must also be included. The hybrid representation is therefore a design tradeoff whose trajectory-level benefit requires empirical evaluation, rather than a guarantee of universally minimal pose error.

## J. Heuristic Planner Oracle Comparison

The heuristic planner is evaluated with reconstructed point clouds, navigation meshes, semantic masks, and target annotations. This is a privileged-information comparison on estimated geometry, rather than an oracle with complete physical ground truth. It attains 5.2% collision and 58.7% visibility, versus OmniCam’s 10.4% and 52.0%. OmniCam estimates geometry internally from the panorama and does not run navmesh construction or graph search. The reported runtimes (approximately 16.5s vs. 0.84s) differ by a factor of 19.6; the timing assumptions must be matched before attributing the difference to a particular component.

## K. More Qualitative Results

Figure 8 provides additional qualitative trajectory comparisons. Each row shows one scene, and the columns show the outputs of different methods. These selected examples illustrate differences in trajectory placement and orientation; they do not establish dataset-wide collision or smoothness guarantees.

Figure 9 presents additional downstream examples from indoor, outdoor, and stylized environments. Each row contains a panorama, selected generated video frames, and reconstruction results with and without the OmniCam trajectory. The figure illustrates qualitative differences; reconstruction accuracy and completeness require a separate quantitative evaluation.

## L. Downstream Application Validation

Camera-Controlled Video Generation. OmniCam and GenDoP trajectories serve as camera control signals for WorldStereo [36]. The reported evaluation uses 484 trajectories across indoor, outdoor, and stylized scenes. For each prompt, the resulting poses drive point-cloud rendering and subsequent video synthesis.

Generated videos are evaluated with five reported measures. CLIP-Score averages the cosine similarity between CLIP ViT-B/32 image and scene-description embeddings over 10 sampled frames. Consecutive-frame SSIM measures adjacent-frame similarity and is sensitive to motion magnitude; it is not a motion-compensated temporal-consistency metric. Aesthetic proxy combines CLIP embedding magnitude and inter-frame embedding coherence on a reported 0–10 scale. Its formula and normalization require verification. Known-region Fidelity compares generated frames with point-cloud renderings in observed regions using SSIM, so it uses a rendering reference. Generation Coverage is the mean unknown-region mask ratio, a descriptor of synthesis difficulty rather than an accuracy metric.

Relative to GenDoP, the reported OmniCam results improve CLIP score by 14.1%, consecutive-frame SSIM by 15.4%, the aesthetic proxy by 18.0%, and known-region fidelity by 9.2%. The unknown-region mask ratio is 38.5% vs. 35.8%. These results describe outputs of the trajectoryplus-video pipeline; they do not isolate framing, path length, speed, or geometric consistency as the cause of improvement. Quantitative reconstruction accuracy and completeness are not evaluated in this table.

Robotic Manipulation (Active Perception). OmniCam is evaluated as an active-perception module for $\pi _ { 0 . 5 }$ [16], generating camera trajectories to improve target observations. A trial succeeds when the designated object is grasped within the time limit. The original record describes 200 episodes across 50 tabletop scenes, with reported success rates of 72.5% for OmniCam, 48.0% for GenDoP, and 56.3% for a fixed camera. The fixed-camera percentage is incompatible with a single pooled denominator of 200 binary trials; its aggregation remains unresolved.

Figure 6 presents qualitative physical-robot grasping examples with initially occluded targets. The demonstration illustrates feasibility in the shown cases. A systematic sim-

![](images/69d504aa5837db7f56135f234ad057e267a3255ebfde1bdc6fbaf17af36f9ff8.jpg)  
Figure 8. Additional qualitative comparison on panoramic scenes. Extending the comparison in Figure 4, each row shows a different scene spanning indoor, outdoor, and stylized environments, and each column shows the trajectory produced by a specific method. The examples illustrate differences in target-directed trajectories across scene categories. The trajectory captions used for each scene are listed in Table 9.

Table 9. Trajectory captions for Figure 8. Each row corresponds to a scene in the qualitative comparison figure, listing the scene type, target object, and the hierarchical textual instruction provided to OmniCam.
<table><tr><td rowspan=1 colspan=1>Row</td><td rowspan=1 colspan=1>Type</td><td rowspan=1 colspan=1>Target</td><td rowspan=1 colspan=1>Caption</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Target</td><td rowspan=1 colspan=1>door</td><td rowspan=1 colspan=1>&lt;TASK: TARGET&gt; &lt;TARGET_OBJ: door&gt; &lt;TARGET_DIR: Back&gt; Approach the back door tobring it into focus. The sequence concludes by arriving at the final &lt;END_OBJ: door&gt; position withinthe &lt;END_DIR : Back&gt; sector of the panoramic scene.</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>Target</td><td rowspan=1 colspan=1>light fixture</td><td rowspan=1 colspan=1>&lt;TASK: TARGET&gt; &lt;TARGET_OBJ: light fixture&gt; &lt;TARGET_DIR: Front&gt; Approachthe front light fixture to bring it into focus. The sequence concludes by arriving at the final &lt;END_OBJ :light fixture&gt; position within the &lt;END_DIR: Front&gt; sector of the panoramic scene.</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>Wander</td><td rowspan=1 colspan=1>free space</td><td rowspan=1 colspan=1>&lt;TASK: WANDER&gt; &lt;TARGET_OBJ: free space&gt; &lt;TARGET_DIR: Front&gt; Approach thefront workstation to explore the layout of the industrial corridor. The sequence concludes by arriving at thefinal &lt;END_OBJ: free space&gt; position within the &lt;END_DIR: Front&gt; sector of the panoramicscene.</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>Target</td><td rowspan=1 colspan=1>door</td><td rowspan=1 colspan=1>&lt;TASK: TARGET&gt; &lt;TARGET_ OBJ: door&gt; &lt;TARGET_DIR: Back&gt; Approach the back door tobring it into focus. The sequence concludes by arriving at the final &lt;END_OBJ : door&gt; position withinthe &lt;END_D IR : Back&gt; sector of the panoramic scene.</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>Target</td><td rowspan=1 colspan=1>ice tunnel</td><td rowspan=1 colspan=1>&lt;TASK: TARGET&gt; &lt;TARGET_OBJ: ice tunnel&gt; &lt;TARGET_DIR: Front&gt; Approachtheice tunnel to bring it into focus. The sequence concludes by arriving at the final &lt;END_OBJ: i cetunnel&gt; position within the &lt;END_DIR: Front&gt; sector of the panoramic scene.</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>Wander</td><td rowspan=1 colspan=1>free space</td><td rowspan=1 colspan=1>&lt;TASK: WANDER&gt; &lt;TARGET_OBJ: free space&gt; &lt;TARGET_DIR: Front&gt; Approach thedistant peaks to explore the mountain valley. The sequence concludes by arriving at the final &lt;END_OBJ :free space&gt; position within the &lt;END_DIR: Front&gt; sector of the panoramic scene.</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>Surround</td><td rowspan=1 colspan=1>clouds</td><td rowspan=1 colspan=1>&lt;TASK: SURROUND&gt; &lt;TARGET_OBJ: clouds&gt; &lt;TARGET_DIR: Front&gt; Capture the cloudsfrom multiple angles to highlight their form and color against the twilight sky. The sequence concludes byarriving at the final &lt;END_OBJ: clouds&gt; position within the &lt;END_DIR: Front&gt; sector of thepanoramic scene.</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>Surround</td><td rowspan=1 colspan=1>neon sign</td><td rowspan=1 colspan=1>&lt;TASK: SURROUND&gt; &lt;TARGET_OBJ: neon sign&gt; &lt;TARGET_DIR: Right&gt; Capturetheright neon sign from multiple angles to highlight its glowing details and reflections. The sequenceconcludes by arriving at the final &lt;END_OBJ: neon sign&gt; position within the &lt;END_DIR:Ri ght&gt; sector of the panoramic scene.</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>Wander</td><td rowspan=1 colspan=1>free space</td><td rowspan=1 colspan=1>&lt;TASK: WANDER&gt; &lt;TARGET_OBJ: free space&gt; &lt;TARGET_DIR: Right&gt; Approachtheright to explore the surreal landscape beyond the pagoda. The sequence concludes by arriving at the final&lt;END_OBJ: free space&gt; position within the &lt;END_DIR: Right&gt; sector of the panoramic scene.</td></tr><tr><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>Surround</td><td rowspan=1 colspan=1>sand dune</td><td rowspan=1 colspan=1>&lt;TASK: SURROUND&gt; &lt;TARGET_OBJ: sand dune&gt; &lt;TARGET_DIR: Front&gt; Capture thesand dune from multiple angles to reveal its textured ridges and surrounding lanterns. The sequenceconcludes by arriving at the final &lt;END_OBJ: sand dune&gt; position within the &lt;END_DIR:Front&gt; sector of the panoramic scene.</td></tr></table>

Table 10. Downstream evaluation: camera-controlled video generation with WorldStereo [36]. Different trajectory sources are compared using the same video generation model.
<table><tr><td>Method</td><td>CLIP-S ↑</td><td>Adj.-SSIM ↑</td><td>Aest. proxy ↑</td><td>Fidelity ↑</td><td>Coverage</td></tr><tr><td>GenDoP [34] + WorldStereo</td><td>0.2644</td><td>0.4248</td><td>7.647</td><td>0.6924</td><td>35.8%</td></tr><tr><td>OmniCam (Ours) + WorldStereo</td><td>0.3018</td><td>0.4904</td><td>9.021</td><td>0.7560</td><td>38.5%</td></tr></table>

![](images/2b270ad6d1425ad37df5822bc3f05c51ad6117c894f694e9a95948c0647d9ffc.jpg)  
Panorama <sup>3D</sup> <sup>Reconstruction</sup> <sup>3D</sup> <sup>Reconstruction</sup>Camera-Controlled Video Generation with OmniCam Traj. w/o OmniCam Traj. w/ OmniCam Traj.

Figure 9. Qualitative downstream results across diverse panoramic scenes. Each row shows the input panorama (left), four selected frames from camera-controlled video generation using an OmniCam trajectory (middle), and 3D reconstruction results without and with the OmniCam trajectory (two rightmost columns). The examples include indoor, outdoor, and stylized environments. These visual comparisons illustrate the pipeline outputs; they do not quantify reconstruction accuracy or guarantee collision-free motion.

Table 11. Downstream evaluation: robotic manipulation with π<sub>0.5</sub>. Reported grasp success (SR) and execution visibility (Vis.). The stated 200-episode setup and fixed-camera SR aggregation require reconciliation with the episode records.
<table><tr><td>Trajectory Source</td><td>Grasp SR (%) ↑</td><td>Exec. Vis. (%) ↑</td></tr><tr><td>π0.5 [16] (fixed camera)</td><td>56.3</td><td>47.2</td></tr><tr><td>GenDoP [34]</td><td>48.0</td><td>52.6</td></tr><tr><td>OmniCam (Ours)</td><td>72.5</td><td>71.8</td></tr></table>

to-real claim would require repeated physical trials, matched fixed-camera controls, and documented camera motions and failure counts.

## M. Statistical Confidence Analysis

Table 12 reports 95% confidence intervals from 1,000 bootstrap resamples of OmniCaT evaluation scenes for Omni-Cam and retrained GenDoP. The listed intervals do not overlap. These estimates characterize variation across evaluation scenes, not variation across training seeds, and do not cover all other experiments. GenDoP is the comparison in this table, but is not the best baseline for every main-table metric.

Table 12. Bootstrap 95% confidence intervals on OmniCaT evaluation scenes (1,000 resamples). Intervals describe evaluation-scene variability for the two listed methods.
<table><tr><td>Method</td><td> $\mathbf { A T E \downarrow }$ </td><td> $\mathbf { F D E } \downarrow$ </td><td> $\mathbf { R P E - R } \downarrow$ </td><td> $\mathbf { R P E - T \downarrow }$ </td><td> $\mathrm { { C o l l . } } \left( \% \right) \downarrow$ </td><td> $\mathbf { V i s . } \left( \% \right) \uparrow$ </td><td> $\mathbf { A n g l e } \left( { \mathrm { ^ \circ } } \right) \downarrow$ </td></tr><tr><td> $\mathrm { G e n D o P ~ ( O m n i C a T ) }$ </td><td> $1 . 6 2 4 { \scriptstyle \pm 0 . 0 8 7 }$ </td><td> $2 . 4 8 3 { \scriptstyle \pm 0 . 1 2 4 }$ </td><td> $1 . 5 7 7 { \scriptstyle \pm 0 . 0 6 8 }$ </td><td> $0 . 0 3 8 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $3 0 . 4 \pm 1 . 8$ </td><td> $3 3 . 6 { \pm } 2 . 1 $ </td><td> $2 7 . 6 { \pm } 1 . 4 $ </td></tr><tr><td>OmniCam (Ours)</td><td> $\mathbf { 0 . 8 6 8 { \scriptstyle \pm 0 . 0 4 2 } }$  </td><td> $1 . 4 6 2 { \scriptstyle \pm 0 . 0 7 1 }$ </td><td> $\mathbf { 1 . 1 3 5 { \scriptstyle \pm 0 . 0 5 3 } }$  </td><td> $\mathbf { 0 . 0 2 7 { \scriptstyle \pm 0 . 0 0 2 } }$ </td><td> ${ \bf 1 0 . 4 \pm 1 . 1 }$ </td><td> ${ \pm } 2 . 0 { \pm } 1 . 9$ </td><td> ${ \bf 1 0 . 9 2 0 . 8 }$ </td></tr></table>