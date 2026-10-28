# Perceiving and Predicting Dynamic Sports: Spatiotemporal Intelligence in Vision Models for Egocentric Tennis Video

Tudor Braicu<sup>∗,1,2</sup>, Joseph DelPreto<sup>∗,1</sup>, Jared Boyer<sup>∗,1</sup>, Monika Farsang´ <sup>∗,1,3</sup>, Maximilian Stolzle¨ <sup>∗,1</sup>, Tim Niklas Seyde<sup>1,4</sup>, and Daniela Rus<sup>1</sup>

Abstract— Agile robots require spatiotemporal intelligence to interpret scenes, reason about interactions, and anticipate future events. Yet it remains unclear how well foundation vision models support these capabilities, particularly from a moving egocentric camera, which is a natural view for robots. To investigate this, we collect an annotated tennis dataset and train a suite of vision models on spatiotemporal tasks. Our dataset contains egocentric video synchronized with five allocentric videos, estimated global positions, labeled contact times, and annotated stroke types. We then evaluate seven prediction tasks from egocentric images: 2D ego player and 3D ball localization, future player and ball trajectories, forehand or backhand stroke anticipation, time to the next ego player ball contact, and 3D ball position at that contact. We compare image encoders, video encoders, and visionlanguage models with jointly trained prediction heads. The best performing models achieve mean current-frame localization errors of 0.14 m for the ego player and 0.54 m for the ball, future forecasting errors of 0.34 m for the ego player and 0.91 m for the ball over 1.1 s, and a mean contact-time prediction error of 0.14 s. Auxiliary depth information substantially improves 3D ball localization, suggesting that inferring metric depth

remains a challenge. Longer image histories do not improve ball forecasting, indicating that representations exploit scene context rather than rely solely on inter-frame motion. Finally, we propose a conditional diffusion model to generate wholebody kinematic tennis motion from vision encoder predictions on egocentric video at real-time rates, adapting movements as predictions are refined. These results move towards a deeper understanding of how foundation vision models can support real-time dynamic robots using egocentric video.

## I. INTRODUCTION

Advances in robot hardware and learned controllers are enabling increasingly agile whole-body movements [1] and athletic behaviors [2], [3], [4], but coordinating these movements with fast objects and other agents requires visual processing to be sufficiently accurate and timely to guide immediate action. This motivates spatial intelligence that supports both estimating the current state of the environment and anticipating its evolution [5], [6], [7].

Spatial intelligence of vision-language models has been studied with question-answering techniques that probe their understanding of where objects and agents are located and how they are oriented [6], [8]. Accounting for how these relations evolve over time extends the scope to spatiotemporal intelligence [7]. We further distinguish understanding from forecasting [9] – understanding seeks to infer current spatial configurations, motions, and action cues from observed inputs, while forecasting aims to predict future states or actions using current and past observations. Representations that expose this information can support downstream visuomotor control and embodied planning [10], [11], while learned world models can predict future representations to guide action [12], [13]. These applications motivate evaluating what information is accessible in large-scale pretrained visual representations and how adaptation affects their usefulness [14].

![](images/cf5bb5e979a92a1a857c1a19088ee38cb8ff67f2dbb8e9e36f83740b557167e6.jpg)  
Fig. 1: Visual models are trained on egocentric tennis videos to investigate spatiotemporal understanding, forecasting, and anticipation. The dataset will be publicly released, including synchronized allocentric videos and spatial and shot annotations.

We use tennis as a concrete setting to explore perceptual demands of highly dynamic sports: localizing players and fast-moving objects, anticipating trajectories, and coordinating body motion for imminent contacts [5], [15], [16]. Returning a tennis ball couples these capabilities – spatial accuracy and timing are crucial since the players and ball continue moving during perception, planning, and execution [17], [4].

There have been three main data modalities used by robots tackling these challenges. First, dedicated spatial sensing uses motion capture (MoCap) sensors or calibrated camera systems to directly estimate player and ball positions for trajectory forecasting and control [3], [4], [5], [17]. Second, allocentric videos, recorded by external cameras at fixed court locations, support ball tracking and trajectory analysis [18], [19], player behavior modeling, simulated skill learning [20], [21], and spatial question-answering with foundation vision-language models [22]. Third, egocentric perception, recorded from a player-worn camera, has addressed image-space instance segmentation, forehand action phases, and pose estimation of other players [23], [24], [25]. Egocentric cues such as gaze [26] and head orientation may also reveal aspects of attention and strategy that are less apparent from allocentric views.

For autonomous dynamic robots, egocentric video is especially relevant because it relies solely on a body-mounted camera rather than external sensors – it thus removes workspace limitations and infrastructure demands. However, this viewpoint presents a demanding test of perception: body and head motion mix camera and object motion, court landmarks and relevant objects leave the field of view, and dynamic objects such as a ball can appear small, blurred, or partially occluded. The joint ability of foundation visual representations to support metric player and object state estimation and causal forecasting throughout such egocentric video of dynamic tasks remains insufficiently characterized.

We therefore ask: How well do foundation vision models support spatial understanding and forecasting from dynamic egocentric video?

To support this evaluation, we collect a tennis dataset with synchronized video from a head-worn egocentric camera and five allocentric cameras. We annotate this data with segmentations and localization information via a custom pipeline that fuses automated inferences and manual labeling.

We then use the egocentric frames as model inputs and evaluate seven spatiotemporal understanding and forecasting tasks split into four groups. Spatial state estimation predicts the current 2D ego player position and 3D ball position. Spatial state forecasting predicts the future 2D ego motion and 3D ball motion. Stroke anticipation predicts whether the ego player’s next stroke will be forehand or backhand. Contact anticipation predicts the remaining time until the ego player’s racket-ball contact and the 3D ball position at contact. Together, these tasks probe the spatial and temporal understanding of pre-trained visual representations.

We select 13 vision encoders spanning image encoders, video encoders, and vision-language models (VLMs). We attach lightweight prediction heads to each encoder’s output visual token stream, train the encoder and probes jointly on all tasks, and evaluate prediction accuracy on held-out rallies. We consistently observe 3D ball prediction errors and 2D ego player localization errors that are on par with expected ball motion across 1–2 frames and expected player motion across 3–7 frames, indicating promising dynamic performance.

We perform ablations to investigate representations, dynamic information, and horizon lengths. Ball prediction errors reduce significantly with supplementary depth information, indicating that metric depth estimation is weakly encoded in the visual representations. Longer conditioning histories do not improve 3D position inference; in fact, image encoders achieve identical ball forecasting performance with only one conditioning frame, relying instead on other spatial correlations to predict future motion. We study the impact of rally phase and horizon length, finding that higher temporal proximity to ego–ball contact time reduces ball localization error but increases ego player localization error, and that ball prediction error increases with the forecasting horizon.

Finally, we connect egocentric video understanding to whole-body tennis motion generation through a conditional motion diffusion model [27]. Trained on our dataset, the diffusion model uses the above vision-based predictions of racket–ball contact position, time remaining until contact, and whether the ball is approaching or departing (derived from ball forecasting) as conditioning information. As new visual observations refine contact time and position estimates, the model adaptively re-plans its whole-body kinematic motion. The planning system is instantiated with a finetuned vision-only variant of the LFM2.5-VL-450M [28], which is the fastest of our evaluated VLM-derived models and capable of 100 Hz inference on modern desktop GPUs. This pipeline demonstrates the utility of pre-trained spatiotemporal representations for downstream real-time robot applications.

Key contributions are as follows:

• A tennis video dataset of 85 rallies with 53,155 egocentric frames and five synchronized allocentric views, annotated with 3D ball trajectories, 2D court-frame player trajectories, stroke type labels, and contact times, to be released upon acceptance of the paper.

• A systematic evaluation of foundation vision model representations for seven tasks of spatiotemporal understanding and forecasting from egocentric tennis videos.

• Empirical findings that (i) spatiotemporal task performance is comparable across image encoders and VLMs, but not the video encoder; (ii) supplementary depth information reduces ball localization error by 50–60%; (iii) longer conditioning histories yield little benefit.

• A conditional diffusion model that generates whole-body kinematic tennis motion conditioned on predictions from a vision encoder trained on egocentric video.

b)  
![](images/5401086f91251f9ec780381c234c5fec9e21d68b39c5af111d1cb5aacff6e62d.jpg)

![](images/253f3008748aa69f3b5e5d95bd97fa33d81d3d3c124f093cd3ee4263d2b4f6d6.jpg)

c)  
![](images/6f09fa8e9e9207ad643afc513c12e40cfc7a39f456c3296b77c7b29a5a12a12e.jpg)  
Fig. 2: The dataset includes 53,155 frames over 85 rallies. Spatial distributions of samples are shown for (a) the ball and (b) the ego player. The temporal distribution of samples relative to the ego player ball contact is shown in (c). Locations of the five static cameras are depicted in (a) and (b).

## II. DATASET CURATION AND ANNOTATION PIPELINE

## A. Synchronized Egocentric and Allocentric Videos

To collect a detailed video dataset of tennis play, we recorded an expert tennis player using a head-mounted egocentric camera and five static allocentric cameras in an outdoor setting. Four static cameras were placed at eye level at each corner of the half-court on which the ego player was playing, and one static camera raised high above the ground was centered behind the court; the locations are illustrated in Figure 2. The egocentric camera yielded video at 29.97 frames per second with resolution 3840 × 2160 pixels. The allocentric cameras yielded video at 29.97 or 59.94 frames per second with resolution at least 5120 × 2880 pixels.

All cameras recorded continuously for approximately 80 minutes. Manual annotations then marked the start and end times of each rally, and the contact times of each shot. After extracting only the active times within rallies, the dataset spans approximately 29.6 minutes (53,155 egocentric frames) across 85 rallies. Each rally is segmented from the opening shot to its annotated end: the final racket-ball contact, second bounce, or net contact. The rallies included 557 shots by the ego player and 574 shots by the opponent. The ego player’s shots included 328 forehands and 229 backhands.

The rallies span both free play (29 rallies) and structured practice (56 rallies), with the latter including four different patterns designed to capture diverse ego player movement across the court. The resulting distributions of the ball and player locations within frames of the extracted rallies are illustrated in Figure 2.

The data collection experiment was approved by the local branch of the Institutional Review Board (IRB).

## B. Camera Synchronization and Calibration

At the beginning and end of the session, the ego player clapped three times. This synchronization signal, visible in all videos, was used in post-processing for manual alignment.

Calibration was performed in post-processing with a pinhole camera model, since all cameras use a linear field of view. Camera intrinsics are recovered from a Chessboard + ArUco (ChArUco) calibration board that was held in front of allocentric cameras at the start of the session, and extrinsics are solved via Perspective-n-Point (PnP) with labeled courtcorner correspondences, using OpenCV [29].

## C. State Estimation and Annotation

We annotate recorded videos with information about the physical state and the upcoming play strategy. This includes the ball position, the ego player position, and whether the ego player will next make a forehand or backhand stroke.

1) Ball and Player Position Estimation: We use the three allocentric cameras at the back of the court to estimate the ball and ego player positions. We first annotate the videos with ball and player pixel locations; after downscaling to $1 2 8 0 \times 7 2 0$ resolution, we use an off-the-shelf tennis ball tracker [30] and a general-purpose object detector [31] to obtain initial per-frame 2D anchors for the ball and ego player, which we then manually review and correct. Manual corrections were required for the ball and player annotations in approximately 30.5% and 5.8% of frames, respectively.

We then fuse the pixel annotations from multiple cameras by triangulating the three calibrated and corrected 2D views using a Direct Linear Transform (DLT) solver to produce 3D ball and ego player position labels. The ball is triangulated freely in 3D, whereas the ego player localization uses a planar constraint to recover the foot-ground contact point on the 2D court plane, approximated as the bottom-center of the detected bounding box. At the downscaled 1280 × 720 resolution, median reprojection errors across the three views span 2.1–4.4 px for the ball and 3.3–9.7 px for the player. Ball and player positions are available for 99.9 % and 100 % of frames, respectively; triangulation requires valid 2D detections in at least two views, otherwise the frame is left unlabeled.

We also annotate the ball’s 2D position in the egocentric view. The resulting egocentric ball position is available for 97.4% of frames; the remainder are occlusions.

2) Stroke Type Annotations: To provide information about play strategy and motion prediction, we annotate the time of each ego player ball contact and whether it was a forehand or backhand. We also record the time of each opponent contact.

## III. VISION FOUNDATION MODEL REPRESENTATIONS

## A. Vision Models

We evaluate a diverse set of vision models spanning image encoders, video encoders, and VLMs to assess how pretrained visual representations support egocentric tennis prediction.

1) Image Encoders: We consider four image encoders selected for their ubiquity and comparatively distinct training paradigms: DINOv3 [32] is self-supervised, SigLIP 2 [33] is language-supervised with contrastive and captioning objectives and is the vision tower of many VLMs, Depth Anything 3 (DA3) [34] post-trains a DINOv2 backbone with dense depth supervision, and SAM 3 [35] post-trains a Perception Encoder [36] backbone with dense segmentation supervision. We use checkpoints of comparable capacity: DINOv3 ViT-L/16 (300M), SigLIP 2 So400M/16 (400M), DA3-L (ViT-L/14, 300M), and SAM 3 (PE ViT-L+/14, 450M), i.e. ViT-L-class backbones [37] with 300–450M parameters.

2) Video Encoders: V-JEPA 2 [13] is a video encoder self-supervised to predict masked latent representations. We use the ViT-L/16 checkpoint (300M) with $2 \times 1 6 \times 1 6$ tubelet patches, matching the capacity of the image encoders for multi-frame processing.

3) Vision-Language Models: We consider three VLMs at the 3–4B scale: Qwen3-VL-4B [38], LFM2.5-VL-3B [28], and PLM-3B [39], whose vision towers derive from SigLIP 2- Large (300M), SigLIP 2 So400m NaFlex (400M), and the Perception Encoder [36], respectively. Because the encoders differ, we isolate the contribution of the language backbone and multimodal training by evaluating each VLM both as a vision tower alone and with its full language stack. We additionally include LFM2.5-VL-450M, whose 86Mparameter vision tower is several times smaller than those of the larger VLMs, as a compute-constrained variant to enable downstream real-time planning experiments (Section IV-F).

## B. Backbone Adaptation

1) Prediction Head: To adapt visual representations with minimal computational overhead, we attach task-specific prediction heads to the encoder’s output token stream (patch embeddings and, for VLMs, language tokens). Each head applies cross-attention from learnable queries (one per output, N for N-step forecasts) to the token stream, followed by a 2-layer MLP. All heads are trained jointly on a shared encoder and share an additive positional embedding over the tapped tokens. We use an L2 loss (weight 1.0) for continuous targets and a cross-entropy loss (weight 0.1) for stroke type.

2) Fine-tuning and Training Details: All experiments use a batch size of 32 and 30 training epochs. We use a learning rate of $1 0 ^ { - 4 }$ for the newly initialized prediction heads and $1 0 ^ { - 5 }$ for the pretrained vision backbones. For vision-language models, the language component is finetuned using Low-Rank Adaptation (LoRA) [40] rather than full-parameter finetuning. Thus, our fine-tuning protocol involves up to three components: a fully trained prediction head, a vision backbone finetuned at a reduced learning rate, and a parameter-efficient LoRA adaptation of the language model.

3) Vision Tower and Vision-Language Settings: We evaluate VLMs in two settings. In the vision-tower setting, we retain only the vision tower, using its post-projector tokens as the representation stream. In the vision-language setting, we include both the vision and language components. In line with practices in VLM fine-tuning for auxiliary tasks [41], we tap representations located 50% through language tower depth and discard the second half. For all VLMs in the visionlanguage setting, we inject a standardized text prompt after the encoded image tokens to describe the overall task goal.

## IV. EXPERIMENTAL RESULTS AND DISCUSSION

## A. Experimental Setup

We evaluate all models on the same fixed data splits. Rallies are assigned wholly to one split, so that no temporal window crosses a rally boundary. Splits are stratified across each of the drill sub-types and free-play, to avoid rally type imbalances. The realized split ratios are 63.1 % train, 17.7 % validation, 19.2 % test, corresponding to 53 / 16 / 16 rallies. Training occurs on NVIDIA B200 and H200 GPUs; individual run durations vary between 4–24 hours depending on the model.

## B. Multi-Task Prediction

Each model receives two images as input: the current “anchor” frame and one preceding frame 0.1 s (3 frames) earlier. We jointly optimize seven prediction heads covering five targets. 3D ball position and 2D player position are each predicted at the anchor frame and over a horizon of 12 future frames spanning 33 time steps (1.10 s). Shot type classifies the upcoming stroke as forehand or backhand. Time to contact predicts the time until the ego player’s next ball contact. Ball position at contact predicts 3D ball position at that time.

Table I reports test-split metrics for all 13 configurations. Current-frame 3D ball errors range from 0.54 m (PLM 3B) to 0.90 m (V-JEPA 2). VLMs and vision towers fall within a narrower band (0.54 to 0.71 m, with Qwen3-VL highest), and image encoders are the most tightly clustered (0.55 to 0.60 m). Current-frame 2D player errors vary less across models (0.14 to 0.18 m), with DA3 lowest. Forecasting separates the models more clearly: PLM 3B attains the lowest future ball error (0.91 m) and V-JEPA 2 by far the highest (1.27 m), yet V-JEPA 2 attains the lowest future player error (0.34 m) and retains this advantage across the full horizon, while its ball error remains the highest throughout (Fig. 3 (d)). Shot-type accuracy is near-saturated for all models (98.2 to 99.9%), and time-to-contact errors fall within a narrow range of 141 to 149 ms; Fig. 3 shows how ball and contact errors evolve as contact approaches. Adding the language stack yields small gains for the three larger VLMs, most notably for PLM 3B on ball-related tasks. Overall, image encoders and VLMs perform comparably; PLM 3B leads on ball tasks and, together with LFM2.5-VL 3B, SAM 3, and DA3, is the most consistently performant across spatial tasks, while Qwen3-VL trails slightly on ball localization and V-JEPA 2 is weakest on ball localization but strongest on player forecasting.

## C. Injecting Information to Probe Spatial Representations

We study how providing auxiliary inputs from the annotations to the prediction head affects 3D ball localization, using SigLIP 2 finetuned with a single head for the anchorframe ball position. We consider three inputs: the ball’s 2D image position in the ego frame (ball), the ego player’s 2D court position (player), and the planar ego-ball distance

TABLE I: Prediction accuracy across probed visual representations. Task prediction heads are trained jointly on a shared encoder. “Current” predictions are at the final conditioning frame; “future” predictions are the mean over 12 subsequent frames with stride 0.1 s. Results reflect performance on the test split using model checkpoints selected based on the validation split, across three seeds.
<table><tr><td rowspan="2">Model</td><td colspan="2">3D Ball pos.</td><td colspan="2">2D Player pos.</td><td rowspan="2">Shot type (acc. %) ↑</td><td rowspan="2">Time to</td><td rowspan="2">Ball pos. at contact (ms) ↓ contact (m) ↓</td></tr><tr><td></td><td>Current (m) ↓ Future (m) ↓</td><td>Current (m) ↓ Future (m) ↓</td><td></td></tr><tr><td colspan="8">Image encoders</td></tr><tr><td>DINOv3</td><td> $0 . 5 8 \pm 0 . 0 1$ </td><td> $1 . 0 2 \pm 0 . 0 1$ </td><td> $0 . 1 6 \pm 0 . 0 1$ </td><td> $0 . 3 8 \pm 0 . 0 1$ </td><td> $9 8 . 7 \pm 0 . 0$ </td><td> ${ \bf 1 4 1 . 3 \pm 1 . 6 }$ </td><td> $1 . 0 3 \pm 0 . 0 1$ </td></tr><tr><td>SigLIP 2</td><td> $0 . 6 0 \pm 0 . 0 1$ </td><td> $1 . 0 3 \pm 0 . 0 0$ </td><td> $0 . 1 6 \pm 0 . 0 0$ </td><td> $0 . 3 9 \pm 0 . 0 0$ </td><td> $9 8 . 4 \pm 0 . 2 $ </td><td> $1 4 8 . 8 \pm 1 . 6$ </td><td> $1 . 0 8 \pm 0 . 0 2$ </td></tr><tr><td>SAM 3</td><td> $0 . 5 5 \pm 0 . 0 3$ </td><td> $0 . 9 9 \pm 0 . 0 3$ </td><td> $0 . 1 5 \pm 0 . 0 0$ </td><td> $0 . 3 8 \pm 0 . 0 0$ </td><td> $9 8 . 4 \pm 0 . 3 $ </td><td> $1 4 2 . 7 \pm 3 . 1$ </td><td> $1 . 0 1 \pm 0 . 0 1$ </td></tr><tr><td>DA3</td><td> $0 . 5 8 \pm 0 . 0 1$ </td><td> $0 . 9 9 \pm 0 . 0 2$ </td><td> ${ \bf 0 . 1 4 \pm 0 . 0 1 }$ </td><td> $0 . 3 6 \pm 0 . 0 1$ </td><td> $9 8 . 4 \pm 0 . 0$ </td><td> $1 4 2 . 8 \pm 1 . 6$ </td><td> $1 . 0 0 \pm 0 . 0 2$ </td></tr><tr><td colspan="8">Video encoders</td></tr><tr><td> $\tt V - J E P A 2$ </td><td> $0 . 9 0 \pm 0 . 0 0$ </td><td> $1 . 2 7 \pm 0 . 0 0$ </td><td> $0 . 1 8 \pm 0 . 0 0$ </td><td> $\mathbf { 0 . 3 4 \pm 0 . 0 1 }$ </td><td> ${ \bf 9 9 . 9 \pm 0 . 1 }$ </td><td> $1 4 4 . 0 \pm 0 . 1$ </td><td> $0 . 9 7 \pm 0 . 0 1$ </td></tr><tr><td colspan="8">Vision towers</td></tr><tr><td>Qwen3-VL 4B</td><td> $0 . 7 1 \pm 0 . 0 1$ </td><td> $1 . 0 8 \pm 0 . 0 1$ </td><td> $0 . 1 8 \pm 0 . 0 1$ </td><td> $0 . 3 7 \pm 0 . 0 0$ </td><td> ${ \bf 9 9 . 9 \pm 0 . 1 }$ </td><td> $1 4 2 . 5 \pm 3 . 1$ </td><td> $1 . 0 0 \pm 0 . 0 1$ </td></tr><tr><td> $\mathtt { L F M 2 . 5 - V L 4 5 0 M }$ </td><td> $0 . 6 3 \pm 0 . 0 0$ </td><td> $1 . 0 8 \pm 0 . 0 1$ </td><td> $0 . 1 7 \pm 0 . 0 0$ </td><td> $0 . 4 0 \pm 0 . 0 1$ </td><td> $9 8 . 3 \pm 0 . 2 $ </td><td> $1 4 6 . 3 \pm 2 . 1$ </td><td> $1 . 0 7 \pm 0 . 0 1$ </td></tr><tr><td>LFM2.5-VL 3B</td><td> $0 . 5 9 \pm 0 . 0 1$ </td><td> $1 . 0 2 \pm 0 . 0 1$ </td><td> $0 . 1 6 \pm 0 . 0 0$ </td><td> $0 . 3 8 \pm 0 . 0 0$ </td><td> $9 8 . 2 \pm 0 . 1 $ </td><td> $1 4 5 . 2 \pm 0 . 7$ </td><td> $1 . 0 3 \pm 0 . 0 0$ </td></tr><tr><td>PLM 3B</td><td> $0 . 5 6 \pm 0 . 0 1$ </td><td> $0 . 9 8 \pm 0 . 0 2$ </td><td> $0 . 1 8 \pm 0 . 0 1$ </td><td> $0 . 3 9 \pm 0 . 0 1$ </td><td> $9 8 . 8 \pm 0 . 4$ </td><td> $1 4 3 . 7 \pm 0 . 6$ </td><td> $1 . 0 2 \pm 0 . 0 2$ </td></tr><tr><td colspan="8">Vision-language models</td></tr><tr><td>Qwen3-VL 4B</td><td> $0 . 6 9 \pm 0 . 0 1$ </td><td> $1 . 0 6 \pm 0 . 0 1$ </td><td> $0 . 1 7 \pm 0 . 0 0$ </td><td> $0 . 3 6 \pm 0 . 0 1$ </td><td> $9 9 . 8 \pm 0 . 1$ </td><td> $1 4 7 . 3 \pm 5 . 8$ </td><td> $0 . 9 8 \pm 0 . 0 2$ </td></tr><tr><td>LFM2.5-VL 450M</td><td> $0 . 6 3 \pm 0 . 0 1$ </td><td> $1 . 0 7 \pm 0 . 0 1$   $0 . 9 8 \pm 0 . 0 1$ </td><td> $0 . 1 7 \pm 0 . 0 1$   $0 . 1 6 \pm 0 . 0 0$ </td><td> $0 . 4 0 \pm 0 . 0 1$   $0 . 3 6 \pm 0 . 0 1$ </td><td> $9 8 . 2 \pm 0 . 3 $ </td><td> $1 4 6 . 9 \pm 1 . 5$ </td><td> $1 . 0 8 \pm 0 . 0 1$ </td></tr><tr><td> $\mathrm { L F M } 2 \ : . 5 \mathrm { - } \mathrm { V L } \ : \ : 3 \mathrm { B }$ </td><td> $0 . 5 7 \pm 0 . 0 1$ </td><td> $\mathbf { 0 . 9 1 \pm 0 . 0 0 }$ </td><td> $0 . 1 7 \pm 0 . 0 1$ </td><td> $0 . 3 6 \pm 0 . 0 0$ </td><td> $9 9 . 4 \pm 0 . 2 $   $9 9 . 7 \pm 0 . 1 $ </td><td> $1 4 6 . 7 \pm 1 . 1$ </td><td> $1 . 0 0 \pm 0 . 0 2$ </td></tr><tr><td>PLM 3B</td><td> ${ \bf 0 . 5 4 \pm 0 . 0 2 }$ </td><td></td><td></td><td></td><td></td><td> $1 4 4 . 4 \pm 1 . 4$ </td><td> ${ \bf 0 . 9 5 \pm 0 . 0 2 }$ </td></tr></table>

## D. Varying History to Probe Temporal Representations

We study how probed visual representations encode temporal context through history conditioning by varying the

As shown in Table II, depth reduces error from 0.51 m (image only) to 0.21 m, and combining it with player yields the lowest error (0.19 m). Yet ball and player alone provide no benefit, and adding ball slightly degrades the combinations. This suggests that depth supplies geometric information hard to infer from images, but ball pixel position is redundant with visual representation. The image-free MLP also benefits most from depth, but unlike SigLIP 2, it gains more from ball, performing best with all three inputs (1.07 m). The substantially higher errors of image-free MLP variants show that the auxiliary inputs alone cannot replace visual data.

(depth), each evaluated individually and in all pairwise and full combinations. Inputs are sinusoidally embedded and injected into the head’s cross-attention block via adaptive layer normalization (AdaLN) [42]. As a control, we train an image-free MLP on the auxiliary inputs alone.

TABLE II: Current 3D ball position error with auxiliary inputs. Results reflect test-split errors over three seeds, using checkpoints selected based on the validation split. <sup>†</sup>Image-free baseline using only auxiliary inputs.
<table><tr><td>Auxiliary Information Added</td><td> $\mathbf { M L P ^ { \dagger } } \downarrow$ </td><td> $\mathrm { \textsf { S i g L I P } } \ 2 \ \downarrow$ </td></tr><tr><td>None (image only)</td><td></td><td> $0 . 5 1 \pm 0 . 0 1$ </td></tr><tr><td>Ball</td><td> $2 . 8 3 \pm 0 . 0 1$ </td><td> $0 . 5 4 \pm 0 . 0 0$ </td></tr><tr><td>Player</td><td> $4 . 2 8 \pm 0 . 0 3$ </td><td> $0 . 5 1 \pm 0 . 0 1$ </td></tr><tr><td>Depth</td><td> $2 . 3 9 \pm 0 . 0 1$ </td><td> $0 . 2 1 \pm 0 . 0 1$ </td></tr><tr><td>Ball + player</td><td> $2 . 6 6 \pm 0 . 0 3$ </td><td> $0 . 5 3 \pm 0 . 0 1$ </td></tr><tr><td> $\mathrm { B a l l } + \mathrm { d e p t h }$ </td><td> $2 . 1 2 \pm 0 . 0 1$ </td><td> $0 . 2 5 \pm 0 . 0 2$ </td></tr><tr><td> $\mathrm { P l a y e r + d e p t h }$ </td><td> $1 . 2 4 \pm 0 . 0 1$ </td><td> ${ \bf 0 . 1 9 \pm 0 . 0 1 }$ </td></tr><tr><td> $\mathrm { B a l l + p l a y e r + d e p t h }$ </td><td> ${ \bf 1 . 0 7 \pm 0 . 0 2 }$ </td><td> $0 . 2 3 \pm 0 . 0 3$ </td></tr></table>

TABLE III: Future 3D ball position error versus input history (Frames: number of input frames; Span: history duration). Results reflect test-split errors over three seeds, using checkpoints selected based on the validation split. <sup>†</sup>Image-free baseline using only ground-truth 3D ball history.
<table><tr><td>Frames</td><td>Span</td><td> $\mathrm { G T } \ \mathrm { M L P ^ { \dag } \downarrow }$ </td><td> $\mathtt { S i g I L P \ 2 \downarrow }$ </td><td> $\begin{array} { r } { \mathrm { V - J E P A } 2 \downarrow } \end{array}$ </td></tr><tr><td>1</td><td>一</td><td> $2 . 9 2 4 \pm 0 . 0 0 3$ </td><td> ${ \bf 1 . 0 2 \pm 0 . 0 2 }$ </td><td></td></tr><tr><td>2</td><td>100 ms</td><td> $0 . 2 9 7 \pm 0 . 0 0 1$ </td><td> ${ \bf 1 . 0 2 \pm 0 . 0 5 }$ </td><td> $1 . 2 2 \pm 0 . 0 1$ </td></tr><tr><td>2</td><td>300 ms</td><td> $0 . 2 5 7 \pm 0 . 0 0 4$ </td><td> $1 . 0 8 \pm 0 . 0 5$ </td><td> $1 . 2 5 \pm 0 . 0 1$ </td></tr><tr><td>4</td><td>300 ms</td><td> $\mathbf { 0 . 2 3 8 \pm 0 . 0 0 3 }$ </td><td> $1 . 0 8 \pm 0 . 0 2$ </td><td> ${ \bf 1 . 1 8 \pm 0 . 0 2 }$ </td></tr></table>

number and spacing of input frames while predicting future 3D ball position. In Table III, we finetune SigLIP 2 on four settings and V-JEPA 2 on three, omitting single-frame history since tubelets must be at least length 2.

An image-free MLP trained on ground truth 3D ball positions shows the necessity of temporal information: from a single position it cannot forecast (2.92 m error), but a second position 100 ms earlier suffices to estimate velocity and reduces error to 0.30 m, with longer or denser history providing only modest further gains.

Image-based models show a different pattern. With only a single frame, SigLIP 2 already achieves 1.02 m, far below the position-only baseline, suggesting that temporal information beyond the ball’s instantaneous position is being extracted. However, additional frames provide no benefit (1.02 m to 1.08 m), so the representation does not recover inter-frame ball displacement. This is consistent with the auxiliary information ablation of Section IV-C: without accurate depth inference, finite differences yield noisy velocity estimates, and models may instead rely on spatial cues found within a single frame. V-JEPA 2 improves slightly with more frames (1.22 m to 1.18 m), but remains worse than SigLIP 2 on a single frame.

![](images/2c8de6e76239207d64b86893476bc8bcda9e8101dbe9a2ef59c784d700af60ef.jpg)

![](images/cc29915fd3bc912d340bd23bf9ffe5606cbe80c86ab90192bb844c78eb772c1e.jpg)  
Fig. 3: Prediction errors on approaching test-split shots versus time to contact: (a) current 3D ball position, (b) ball position at contact, and (c) time to contact. Lines show means over 0.1 s intervals; shading shows ±1 SD over three seeds. (d) Ball 3D and player 2D future predictions over a time horizon of ∼1.10 s, across all models, averaged over three seeds.

## E. Effect of Backbone Adaptation

We compare our training configuration on the LFM2.5-VL 3B vision-language model (finetuned vision encoder with LoRA-adapted language stack) against a fully frozen variant where the vision and the language towers are fixed and only the prediction heads are trained. We find that freezing degrades every task: the error of current ball position rises from 0.57 to 0.96 m, and future ball position from 0.98 to 1.47 m. The 2D player tasks are more robust, with currentframe error rising from 0.16 to 0.18 m and future-horizon error from 0.36 to 0.45 m. Shot-type classification falls from 99.4% to 98.3%, time to contact error rises from 146.7 to 162.0 ms, and ball position at contact error rises from 1.003 to 1.225 m. These results suggest that the frozen features can already largely support current 2D player and shot type predictions, but do not encode precise visual cues needed to localize the ball in 3D from a moving egocentric view.

## F. Conditional Motion Diffusion for Kinematic Tennis Play

We demonstrate how foundation vision models’ spatiotemporal understanding of egocentric video supports whole-body tennis motion generation. We use Motion Diffusion Models (MDMs), a leading approach to conditional kinematic motion generation [27], [43], [44].

![](images/c1a517c07a1e4ccfad72ecdf065a4d3d63130bf85c64fe94ddb694e64ec2c3ed.jpg)

![](images/6a0c8e99a4e627582a7e437e8eaa4913b31ae6e4d031b937271c2f3781d1688a.jpg)  
Fig. 4: LFM2.5-VL 450M results on approaching test-split shots until contact. (a) Predictions for a single frame horizon tend to improve as the ball nears the ego player. (b) Forecasted predictions tend to improve as the ball nears the ego player and degrade as the future horizon increases.

Alongside the videos, we recorded ego player motion using the wearable Xsens MTw Awinda system. We compute joint rotations from the relative orientations of adjacent body segments, using the calibrated T-pose as a reference. Motion frames contain 147 features: pelvis translation and 6D rotations for the pelvis and 23 joints, including the racket attachment. Each window’s coordinate frame is centered on the ground below the pelvis and aligned with its heading at the last history frame. The conditioning vector is

$$
\mathbf { c } _ { t } = [ \tilde { \mathbf { b } } _ { t } ^ { * } , \tau _ { t } ^ { \mathrm { n e x t } } , m _ { t } ^ { \mathrm { n e x t } } , \tau _ { t } ^ { \mathrm { p r e v } } , m _ { t } ^ { \mathrm { p r e v } } , \mathbf { p } _ { t } ^ { x y } , \sin \psi _ { t } , \cos \psi _ { t } ] \in \mathbb { R } ^ { 1 1 } ,\tag{1}
$$

where $\tilde { \mathbf { b } } _ { t } ^ { * }$ is the anticipated ball-contact position in this local frame, $\tau _ { t } ^ { \mathrm { n e x t } }$ is the time remaining until contact, and $\tau _ { t } ^ { \mathrm { p r e v } }$ is the time elapsed since the previous player contact. Binary masks $m _ { t } ^ { \mathrm { n e x t } }$ and $m _ { t } ^ { \mathrm { p r e v } }$ indicate whether the upcomingcontact inputs and previous-contact timing, respectively, are supplied or masked out. The global pelvis position $\mathbf { p } _ { t } ^ { x y }$ and heading $\psi _ { t }$ help learn to return to a neutral court pose after a shot. Training reconstructs recorded motion from noisy samples, conditioned on annotated tennis states, using the vision dataset’s rally-based train / validation / test splits.

To obtain tennis conditions from video, we use the LFM2.5- VL-450M vision encoder and prediction heads with two recorded images approximately 100 ms apart. Its compact 86M-parameter encoder enables real-time visual conditioning at 100 Hz on an RTX 5090, while the model localizes the ball more accurately than the larger Qwen3-VL 4B vision tower and stays within 5 ms of the best contact-time error (Table I). Its predictive capabilities are displayed in Figure 4.

![](images/55b7b60ada3693edf4cc793c03f00bffc678ffcae5e785577e57cc24c359d9c3.jpg)  
Fig. 5: (a) LFM2.5-VL-450M forecasts ball and player positions and anticipates contact; crosshairs locate the recorded contact. (b) Recorded ego motion versus MDM rollouts conditioned on annotations (“rec. cond.”) or vision predictions (“vision cond.”). Trails fade toward the past; ball forecasts fade toward the future. t = 0 denotes the recorded racket–ball contact.

The predicted racket–ball contact position, transformed into the local pelvis frame, supplies $\tilde { \mathbf { b } } _ { t } ^ { * }$ , while the predicted time remaining to contact supplies $\tau _ { t } ^ { \mathrm { n e x t } }$ . From current and forecast ball positions, we estimate velocity along the court to classify approaching shots (toward the ego player) and departing shots (away). We set $m _ { t } ^ { \mathrm { n e x t } } = 1$ for approaching shots with a valid contact estimate; otherwise, $m _ { t } ^ { \mathrm { n e x t } } = 0$ masks both target inputs. We recognize player contact when successive vision estimates show an approaching-to-departing reversal near the predicted player position. The time elapsed since the latest recognized contact supplies $\tau _ { t } ^ { \mathrm { p r e v } }$ , with $m _ { t } ^ { \mathrm { p r e v } } = 1$ after the first recognized contact and zero beforehand.

We initialize each rollout with two recorded motion frames. The model predicts 24 frames at 60 Hz (400 ms); we execute the first six (100 ms) and use the last two executed frames as history for the next prediction. Accumulated generated pelvis position and heading supply the global SE(2) condition and the local frame for contact targets. At each replanning step, we update the predicted contact position and time remaining, allowing the motion model to revise its motion as the ball approaches. This is motivated by the increasing accuracy of these vision estimates closer to contact (Figure 3). Throughout the rollout, the vision model receives successive frames from the original recorded video. Its observations therefore follow the recorded player’s camera trajectory and do not change in response to the generated character’s movement.

Figure 5 shows synchronized vision predictions and motion comparisons for one held-out test shot. Table IV compares recorded motion with rollouts conditioned on annotations or vision predictions, using matching initial histories and random seeds on held-out sequences. We measure ball– racket distance at annotated contact and its minimum within ±100 ms, together with errors in racket-center position, racketplane orientation, and velocity relative to recorded motion.

Results demonstrate that a VLM vision encoder finetuned for spatiotemporal prediction on egocentric tennis video can provide real-time conditioning for whole-body ego player motion generation. In the qualitative example (Figure 5), the generated player approaches the predicted contact point with well-timed motion that anticipates racket–ball contact, while the contact estimate becomes more accurate as the ball nears.

A limitation is the quality of the reconstructed recordings:

TABLE IV: MDM rollouts on 16 test sequences: means over contacts with bootstrapped 95% confidence intervals in brackets. Racket errors are relative to recorded ego player motion; minimum distance uses a ±100 ms window.
<table><tr><td></td><td colspan="2">Ball-racket proximity</td><td colspan="4">Error relative to recorded motion</td></tr><tr><td>Approach</td><td>At hit (m) ↓</td><td>Min. (m) ↓</td><td>Center (m) ↓</td><td>Plane (°)↓</td><td></td><td>Velocity (m/s) ↓</td></tr><tr><td>Recorded motion</td><td>0.89 [0.76, 1.02]</td><td>0.81 [0.69,0.92]</td><td>0.00</td><td></td><td>0.00</td><td>0.00</td></tr><tr><td>MDM (rec. cond.)</td><td>1.11 [1.04, 1.19]</td><td>0.98 [0.93, 1.04]</td><td>0.76 [0.68,0.84]</td><td>[33.10,38.90]</td><td>35.97</td><td>4.46 [3.91,5.07]</td></tr><tr><td>MDM (vision cond.) [1.12, 1.42]</td><td>1.28</td><td>1.17 [1.04, 1.33]</td><td>[0.78,1.10]</td><td>0.95</td><td>41.31 [36.03,46.43]</td><td>5.38 [4.80,5.92]</td></tr></table>

even recorded motion has a mean ball–racket distance of 0.89 m at annotated contact (Table IV). This mismatch limits both motion learning and evaluation. Precise racket–ball alignment remains unresolved and requires improvements in both recording quality and vision prediction accuracy. The demonstration also remains kinematic, without a policy for physical execution in simulation or reality.

## V. CONCLUSION

We present a systematic study of spatiotemporal understanding and forecasting from dynamic egocentric video, using tennis as a challenging setting for spatial perception and anticipation. Across seven tasks, we find that image encoders and VLMs provide broadly competitive performance and we analyze how their strengths differ by task. Our ablations further show that providing additional depth information substantially improves ball localization, but that longer visual histories do not necessarily improve performance.

Building on these results, we present whole-body motion generation via conditional motion diffusion. This demonstrates that the egocentric estimates could support future pipelines for real-time robot control.

This paper takes a step towards a deeper understanding of how foundation vision models can support highly dynamic robotic tasks. Tennis is one instance of tasks where a robot must maintain situational awareness and act under tight time constraints from a moving viewpoint; future work will test whether the identified representations and adaptation strategies transfer to other highly dynamic settings such as team sports, catching and throwing, and navigation among fast agents.

## ACKNOWLEDGMENTS

This work was supported in part by the GIST-MIT Research Collaboration grant funded by the Gwangju Institute of Science and Technology (GIST), Schmidt Sciences through the AI2050 program, the Boeing company, and the Department of the Air Force Artificial Intelligence Accelerator for research and computing resources under Cooperative Agreement No. FA8750-19-2-1000. This work used computing resources provided by Advanced Research Computing at Hopkins (ARCH).

The views and conclusions expressed in this paper are those of the authors and do not necessarily reflect the official policies, whether express or implied, of the Department of the Air Force or the U.S. Government. The U.S. Government may reproduce and distribute reprints for government purposes regardless of any copyright notices.

We used AI agents to generate parts of our codebase, including Claude Sonnet 5, Claude Opus 5, Claude Fable 5.1, GPT-5.6 Sol, and GPT-6 Astra.

## REFERENCES

[1] T. He, J. Gao, et al., “ASAP: Aligning simulation and real-world physics for learning agile humanoid whole-body skills,” in Proc. of Robotics: Science and Systems, 2025.

[2] T. Haarnoja, B. Moran, et al., “Learning agile soccer skills for a bipedal robot with deep reinforcement learning,” Science Robotics, 2024.

[3] Z. Zhang, H. Lu, et al., “Learning athletic humanoid tennis skills from imperfect human motion data,” arXiv preprint arXiv:2603.12686, 2026.

[4] T. Huang, R. Liu, et al., “Towards professional tennis styles for humanoid robots with adaptive motion planning and tracking,” arXiv preprint arXiv:2608.20087, 2026.

[5] Z. Zaidi, D. Martin, et al., “Athletic mobile manipulator system for robotic wheelchair tennis,” IEEE RA-L, 2023.

[6] B. Chen, Z. Xu, et al., “SpatialVLM: Endowing vision-language models with spatial reasoning capabilities,” in Proc. of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[7] S. Zhou, A. Vilesov, et al., “VLM4D: Towards spatiotemporal awareness in vision language models,” in ICCV, 2025.

[8] A.-C. Cheng, H. Yin, et al., “SpatialRGPT: Grounded spatial reasoning in vision-language models,” in Advances in Neural Information Processing Systems, vol. 37, 2024.

[9] K. Grauman, A. Westbury, et al., “Ego4D: Around the world in 3,000 hours of egocentric video,” in Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition, 2022.

[10] S. Nair, A. Rajeswaran, et al., “R3M: A universal visual representation for robot manipulation,” in Proc. of the 6th Conference on Robot Learning, vol. 205, 2023.

[11] D. Driess, F. Xia, et al., “PaLM-E: An embodied multimodal language model,” in Proc. of the 40th Int. Conf. on Machine Learning, 2023.

[12] G. Zhou, H. Pan, et al., “DINO-WM: World models on pre-trained visual features enable zero-shot planning,” in Proc. of the 42nd Int. Conference on Machine Learning, vol. 267, 2025.

[13] M. Assran, A. Bardes, et al., “V-JEPA 2: Self-supervised video models enable understanding, prediction and planning,” arXiv preprint arXiv:2506.09985, 2025.

[14] A. Majumdar, K. Yadav, et al., “Where are we in the search for an artificial visual cortex for embodied intelligence?” in NeurIPS, 2023.

[15] I. Alberico, M. Cannici, et al., “Egocentric event-based vision for ping pong ball trajectory prediction,” in Proc. of the IEEE/CVF Conf. on Computer Vision and Pattern Recognition Workshops (CVPRW), 2025.

[16] J. Ren, Y. Li, et al., “SMASH: Mastering scalable whole-body skills for humanoid ping-pong with egocentric vision,” arXiv preprint arXiv:2604.01158, 2026.

[17] Q. Xiao, Z. Zaidi, and M. Gombolay, “Multi-camera asynchronous ball localization and trajectory prediction with factor graphs and human poses,” in IEEE Int. Conf. on Robotics and Automation (ICRA), 2024.

[18] Y.-C. Huang, I.-N. Liao, et al., “TrackNet: A deep learning network for tracking high-speed and tiny objects in sports applications,” in IEEE Int. Conf. on Advanced Video and Signal Based Surveillance, 2019.

[19] L. Dong, Y. Yang, et al., “RacketVision: A multiple racket sports benchmark for unified ball and racket analysis,” AAAI, 2026.

[20] H. Zhang, C. Sciutto, et al., “Vid2Player: Controllable video sprites that behave and appear like professional tennis players,” ACM Transactions on Graphics, vol. 40, 2021.

[21] H. Zhang, Y. Yuan, et al., “Learning physically simulated tennis skills from broadcast videos,” ACM Transactions on Graphics, vol. 42, 2023.

[22] Y. Yang, Y. Shao, et al., “Stepping VLMs onto the court: Benchmarking spatial intelligence in sports,” arXiv preprint arXiv:2603.09896, 2026.

[23] J. Lee, Y. Li, et al., “Towards AI-powered AR for enhancing sports playability for people with low vision: An exploration of ARSports,” in Int. Symposium on Mixed and Augmented Reality Adjunct, 2024.

[24] Z. Xue and K. Grauman, “Learning fine-grained view-invariant representations from unpaired ego-exo videos via temporal alignment,” in NeurIPS, 2023.

[25] R. Khirodkar, A. Bansal, et al., “Ego-Humans: An ego-centric 3D multi-human benchmark,” in Int. Conf. on Computer Vision, 2023.

[26] R. Shinkai, S. Ando, et al., “Visual strategies for eye and head movements during table tennis rallies,” Frontiers in Sports and Active Living, vol. 4, 2022.

[27] G. Tevet, S. Raab, et al., “Human motion diffusion model,” in Int. Conf. on Learning Representations, 2023.

[28] Liquid AI, “Lfm2 technical report,” arXiv preprint arXiv:2511.23404, 2025.

[29] G. Bradski, “The OpenCV library,” Dr. Dobb’s Journal of Software Tools, vol. 25, 2000.

[30] S. Tarashima, M. A. Haq, et al., “Widely applicable strong baseline for sports ball detection and tracking,” in British Machine Vision Conf. (BMVC), 2023.

[31] G. Jocher and J. Qiu, “Ultralytics YOLO11,” https://github.com/ ultralytics/ultralytics, 2024, version 11.0.0.

[32] O. Simeoni, H. V. Vo,´ et al., “DINOv3,” Transactions on Machine Learning Research, 2026, featured Certification.

[33] M. Tschannen, A. Gritsenko, et al., “Siglip 2: Multilingual visionlanguage encoders with improved semantic understanding, localization, and dense features,” arXiv preprint arXiv:2502.14786, 2025.

[34] H. Lin, S. Chen, et al., “Depth anything 3: Recovering the visual space from any views,” in Int. Conf. Learning Representations, 2026.

[35] N. Carion, L. Gustafson, et al., “Sam 3: Segment anything with concepts,” in Int. Conf. on learning representations, vol. 2026, 2026.

[36] D. Bolya, P.-Y. Huang, et al., “Perception encoder: The best visual embeddings are not at the output of the network,” in Advances in Neural Information Processing Systems, vol. 38, 2025.

[37] A. Dosovitskiy, L. Beyer, et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” ICLR, 2021.

[38] Qwen Team, “Qwen3-VL technical report,” arXiv preprint arXiv:2511.21631, 2025.

[39] J. H. Cho, A. Madotto, et al., “PerceptionLM: Open-access data and models for detailed visual understanding,” in Advances in Neural Information Processing Systems, vol. 38, 2025.

[40] E. J. Hu, Y. Shen, et al., “Lora: Low-rank adaptation of large language models,” arXiv preprint arXiv:2106.09685, 2021.

[41] M. Reuss, H. Zhou, et al., “FLOWER: Democratizing generalist robot policies with efficient vision-language-flow models,” in 9th Annual Conference on Robot Learning, 2025.

[42] E. Perez, F. Strub, et al., “Film: Visual reasoning with a general conditioning layer,” in AAAI, vol. 32, no. 1, 2018.

[43] Y. Shafir, G. Tevet, et al., “Human motion diffusion as a generative prior,” in Int. Conf. on Learning Representations, 2024.

[44] K. Zhao, G. Li, and S. Tang, “DartControl: A diffusion-based autoregressive motion model for real-time text-driven motion control,” in Int. Conf. on Learning Representations, 2025.