# Robust Real-Time Monocular Human Pose Estimation with

Insole-Based Sensor Fusion

Liang Yu , Tao Zhu , Senior Member, IEEE, Xiangqing Lu, Wenjian Luo, Senior Member, IEEE, and Liming Chen , Senior Member, IEEE

Abstract—Real-time monocular human pose estimation is an important sensing component for human-centered intelligent systems. While recent camera-based methods have achieved impressive accuracy, fundamental challenges persist, particularly when visual observations are incomplete, such as during lowerbody occlusion or short out-of-frame intervals. In these challenging but common scenarios, vision-only systems often struggle to reconstruct physically plausible lower-limb motion and reliable foot-ground contact. Due to the highly unobtrusive and wearable nature of smart insoles, this paper investigates whether simple plantar pressure and acceleration sensors can effectively resolve these visually induced limitations. To systematically investigate this, we first collect a synchronized multimodal dataset and then propose a part-decoupled temporal fusion framework. The proposed model fuses foot-sensing features into the lower-body branch to compensate for incomplete visual evidence, while a lightweight coordinator exchanges information between the upper- and lower-body temporal representations to maintain whole-body consistency. The framework is evaluated with three representative real-time visual backbones based on 2D keypoints, 3D landmarks, and SMPL-based human mesh recovery. We evaluate our framework on three representative real-time visual backbones. Quantitative experiments demonstrate that incorporating simple foot sensors consistently improves pose accuracy, motion smoothness, and gait-related measurements over visiononly baselines, with the most significant gains observed under occlusion and short visual-dropout conditions. Ultimately, our results demonstrate that unobtrusive foot sensing provides vital and complementary physical observability when camera observations are temporarily unavailable.

Index Terms—Monocular pose estimation, plantar pressure, foot-mounted acceleration, multimodal sensor fusion, human motion estimation.

## I. INTRODUCTION

R <sup>EAL-TIME</sup> <sup>human</sup> <sup>motion</sup> <sup>estimation</sup> <sup>is</sup> <sup>a</sup> <sup>crucial</sup> <sup>sens-</sup>ing capability for human-centered intelligent systems, ranging from industrial safety monitoring and ergonomic assessment to rehabilitation and human–machine interaction. Monocular camera-based pose estimation has become highly attractive for these applications due to its low deployment cost and the convenience of requiring only an ordinary RGB camera. Driven by deep learning, recent methods have made remarkable progress, enabling highly accurate and real-time estimation of 2D keypoints, 3D landmarks, and SMPL-based body representations from monocular video [1]–[7].

Despite this impressive progress, real-time monocular pose estimation remains fundamentally constrained. The most critical challenges emerge when visual observations are incomplete, such as during lower-body occlusion (e.g., hidden behind a desk) or short out-of-frame intervals (visual dropouts). Under these conditions, the underlying depth ambiguity and scale uncertainty of monocular vision become severely amplified [8], [9]. Consequently, camera-only systems often produce inaccurate foot trajectories, unreliable contact states, and physically implausible lower-limb motion. This fundamental limitation hinders their reliability in contact-sensitive applications like motion-analysis tasks that strictly depend on foot-ground interaction, stance phase, cadence, and load transfer [7], [10].

To address this issue, we investigate an elegant and practical solution: incorporating simple plantar pressure and acceleration sensors via smart insoles. In particular, we focus on a four-point pressure-inertial sensing configuration (comprising four pressure points and a foot-mounted accelerometer) as an unobtrusive instantiation of foot sensing. Unlike heavy and highly restrictive motion capture suits, smart insoles are highly lightweight and non-intrusive. They seamlessly integrate into daily footwear while continuously providing direct physical measurements of foot-ground contact, load transfer, and lowerlimb dynamics. These signals are highly complementary to monocular visual features and, crucially, remain fully informative when visual cues are occluded or temporarily unavailable [11]–[13]. Therefore, rather than replacing the camera, our goal is to investigate whether the simple addition of unobtrusive foot sensors can provide the necessary physical constraints to overcome the accuracy and stability bottlenecks of monocular pose estimation under visually degraded conditions.

To systematically study this, we first collected a new synchronized Vision–Plantar-Pressure–Acceleration dataset that covers diverse daily activities, along with specifically designed occlusion and visual-dropout scenarios. Based on this dataset, we propose a part-decoupled temporal fusion framework for insole-enhanced monocular pose estimation. Recognizing that insole signals are physically tied to lower-body motion, our design mainly drives the upper-body branch using visual features, while explicitly fusing visual, pressure, and acceleration features in the lower-body branch. A lightweight global coordinator is introduced to exchange information between the upper and lower temporal memories, ensuring that the final reconstructed motion maintains whole-body kinematic consistency. This architecture strategically applies foot-sensing information where it is most needed without overriding visually dominant body parts [12], [14].

![](images/e521a4ef810dffbfcaecce02f9e00c81d72b32f726568a3b851e70e9fa8b5a80.jpg)  
Fig. 1. Overview of the proposed insole-enhanced real-time monocular human pose estimation framework. The system integrates three representative visual backbones (2D keypoints, 3D landmarks, and SMPL mesh recovery) with a four-point pressure-inertial smart insole via a part-decoupled temporal fusion module, producing robust full-body pose and gait-related outputs under clean, occluded, and short out-of-frame conditions.

We thoroughly evaluate the proposed method using three representative real-time monocular pose backbones. Experiments are conducted under clean visual conditions, lower-body occlusion, and short out-of-frame intervals. By evaluating standard pose errors, temporal smoothness, and gait-related measurements (stance accuracy, step-distance error, and cadence error), we prove that the reconstructed motion becomes highly reliable for contact-sensitive motion analysis [10], [15]. The comprehensive results confirm a complete story: unobtrusive foot sensing effectively addresses the inherent blind spots of monocular cameras.

The main contributions of this paper are summarized as follows:

• We experimentally demonstrate that plantar-pressure and foot-mounted acceleration signals can enhance real-time monocular human pose estimation under both normal and degraded visual conditions. The improvement is particularly evident under lower-body occlusion and short outof-frame visual degradation, indicating that foot-sensing signals provide complementary physical cues beyond monocular visual observations.

• We propose a new vision–plantar-pressure–acceleration dataset for robust monocular pose estimation. The dataset contains synchronized monocular video, fourpoint plantar-pressure signals, foot-mounted acceleration, and Xsens reference motion, covering approximately

890K synchronized frames with five additional visualdegradation actions.

• We propose a part-decoupled temporal fusion framework that incorporates plantar-pressure and foot-acceleration cues into lower-body motion estimation while maintaining whole-body kinematic consistency through a lightweight global coordinator.

## II. RELATED WORK

## A. Real-Time Monocular Human Pose Estimation

Monocular human pose estimation has been widely studied for low-cost and deployment-friendly motion sensing. Existing methods can be roughly grouped into 2D keypoint estimation, 3D landmark estimation, and human mesh recovery. For 2D pose estimation, YOLO-based, top-down, and wholebody pose estimators have improved real-time inference and keypoint localization accuracy [1], [2], [16]–[19]. Lightweight landmark systems such as BlazePose further support ondevice human pose tracking for mobile and interactive applications [20]. For monocular 3D pose estimation, recent temporal models exploit transformers, frequency-domain representations, or graph-based reasoning to improve video-based 3D pose stability [3], [4], [21]. Human mesh recovery methods estimate SMPL-like body representations and have advanced from image-based recovery to video-based and promptable reconstruction [6], [7], [22]–[24]. However, these methods mainly rely on visual observations and learned pose priors. When lower-body evidence is occluded, noisy, or temporarily missing, camera-only systems may still suffer from unstable foot motion and physically implausible lower-limb reconstruction.

![](images/61c3eb627b008c8f902f618ae19154b9e5f438a39f656f7733e702dc6dbad307.jpg)  
Fig. 2. Part-decoupled vision–pressure–inertial fusion network. Visual pose features are separated into upper- and lower-body streams, while plantar-pressure and foot-acceleration features are fused into the lower-body stream. A global coordinator exchanges information between upper and lower temporal memories to maintain whole-body consistency.

## B. Foot-Sensing-Based Human Motion Estimation

Foot sensing provides complementary physical information for human motion estimation. Plantar pressure reflects contact state, center-of-pressure variation, load transfer, and gait phase, while foot-mounted inertial signals capture short-term lower-limb dynamics. Wearable pressure sensors, smart shoes, and IMU-based systems have therefore been widely explored for gait analysis, locomotion recognition, and biomechanical motion assessment [15], [25]–[28]. Recent studies further show that foot sensing can support full-body pose or motion reconstruction. SolePoser estimates full-body pose from a pair of insole sensors, PressInPose integrates pressure and inertial sensing for activity-level pose estimation, and FoRM reconstructs SMPL motion from dual-modal plantar pressure and inertial signals [11]–[13]. Sparse-IMU and mobile-devicebased systems also demonstrate the potential of compact wearable sensing for full-body motion estimation [14], [29]. Nevertheless, foot-only or sparse-sensor-only reconstruction is inherently under-constrained because upper-body posture and arm motion cannot be directly observed from foot signals alone. Our work therefore studies plantar-pressure and footacceleration signals as complementary physical evidence for monocular pose estimation rather than as a replacement for vision.

## C. Vision and Foot-Sensing Fusion for Human Motion Capture

A closely related direction is to incorporate foot-ground interaction into vision-based motion capture. Physics-aware methods use contact, center-of-pressure, or body-ground constraints to reduce foot sliding and improve physically plausible pose estimation [7], [30]. Multimodal datasets and systems such as MMVP and MotionPRO further demonstrate that pressure information can improve visual motion capture, especially for foot contact, global translation, and human–world interaction modeling [10], [31]. These studies confirm the value of pressure and contact cues for vision-based human motion estimation. However, many existing works focus on dataset construction, offline fitting, pressure-mat settings, or dedicated multimodal MoCap pipelines. In contrast, this paper targets real-time monocular pose estimation and studies whether fourpoint plantar pressure and foot-mounted acceleration can improve real-time monocular pose estimation under incomplete visual observations.

## III. VISION–PLANTAR-PRESSURE–ACCELERATION DATASET

To support the study of insole-enhanced monocular pose estimation, we introduce a comprehensive and synchronized Vision–Plantar-Pressure–Acceleration dataset. The dataset contains synchronized monocular RGB video, fourpoint plantar-pressure signals, foot-mounted acceleration, and high-quality reference motion annotations.

## A. Data Acquisition and Synchronization

The data collection platform consists of three synchronized subsystems:

1) Smart Insoles: A pair of commercial I-SOL smart insoles, each of which is equipped with four piezoresistive pressure sensors and a 3-axis accelerometer and samples data at 30 Hz and transmits it via Bluetooth.

2) Monocular Camera: An RGB camera is positioned frontally to ensure all full-body details are captured, recording monocular video at 120 Hz (downsampled to 30 Hz).

TABLE I  
ACTION COMPOSITION OF THE NEW VISION–PLANTAR-PRESSURE–ACCELERATION DATASET. DETAILED ACTION DEFINITIONS AND PREPROCESSING SCRIPTS WILL BE PROVIDED IN THE PROJECT REPOSITORY.
<table><tr><td>Subset</td><td>Actions</td><td>Frames</td><td>Representative actions</td><td>Usage</td></tr><tr><td>Home activities</td><td>13</td><td>338K</td><td>standing, dressing, brushing teeth, washing face, eating, tying shoelaces, using phone, hanging laundry, watching TV, sweeping, mopping</td><td>Standard evaluation</td></tr><tr><td>Office activities</td><td>5</td><td>92K</td><td>desk work, getting water, drinking water, restroom use, crossing legs</td><td>Standard evaluation</td></tr><tr><td>Sports/exercise motions</td><td>12</td><td>212K</td><td>marching, walking, jogging, cycling, stairs, jumping rope, tennis, badminton, basketball, kicking, dumbbell exercises, dancing</td><td>Standard evaluation</td></tr><tr><td>Rehabilitation-related motions</td><td>8</td><td>208K</td><td>alternating lunges, squats, horse stance, turning, jumping, sit- to-stand, jumping jacks, tiptoeing</td><td>Standard evaluation</td></tr><tr><td>Short out-of-frame actions</td><td>3</td><td>25K</td><td>jumping out and returning; walking out and returning; leav- ing the view, sitting, and returning</td><td>Visual-dropout analysis</td></tr><tr><td>Desk-induced real occlusion</td><td>2</td><td>15K</td><td>upper-body free motion with static lower body; walking with lower body fully occluded by a desk</td><td>Real-occlusion stress test</td></tr><tr><td>Total</td><td>43</td><td>890K</td><td>normal, short out-of-frame, and desk-induced real-occlusion actions</td><td>Full dataset</td></tr></table>

3) Reference Motion Capture: An Xsens Awinda inertial motion capture system with 17 high-fidelity sensors captures ground-truth full-body kinematics at 60 Hz (downsampled to 30 Hz).

The camera and the Xsens system are connected to the same workstation, enabling direct and precise software synchronization. The wireless insole stream is then aligned to this shared timeline by computing a correlation offset from the accelerometer spikes during a “marching in place” calibration routine at the beginning of each session.

## B. Action Composition and Protocol

The dataset comprises approximately 890K synchronized frames from 20 healthy adult participants (6 female, 14 male; mean age: 24 years). All experimental protocols involving human subjects were approved by the local ethics committee, and written informed consent was obtained from all participants prior to the experiments. It covers 43 distinct activities categorized into six subsets, as detailed in Table I.

The recordings are divided into standard evaluation sets (Home, Office, Sports, and Rehabilitation) and visualdegradation sets. The visual-degradation actions include short out-of-frame dropouts (for qualitative availability analysis) and desk-induced real occlusions (for physical obstruction stress testing). To prevent long-horizon inertial drift from Xsens, the dataset collection follows a block-wise protocol, limiting each continuous recording segment to 10–15 minutes with repeated recalibration.

## IV. METHOD

We propose an insole-enhanced framework for real-time monocular human pose estimation (Fig. 1). The system takes the standardized output of a real-time visual backbone $\mathbf { v } _ { 1 : T } ^ { b }$

and a synchronized smart-insole stream $\mathbf { s } _ { 1 : T }$ as inputs to predict a 22-joint SMPL-style root-relative 3D skeleton $\hat { \mathbf { P } } _ { 1 : T } =$ $f _ { \boldsymbol { \theta } } \big ( \mathbf { v } _ { 1 : T } ^ { b } , \mathbf { s } _ { 1 : T } \big )$ , where $\hat { \mathbf { P } } _ { t } ^ { \phantom { } } \in \mathbb { R } ^ { J \times 3 }$ and $J = 2 2$

## A. Part-Decoupled Vision–Pressure–Inertial Fusion Network

The proposed fusion network is motivated by the physical relationship between the insole signal and the human body. Since pressure and acceleration from the feet mainly describe lower-body contact and dynamics, directly concatenating them with the whole-body visual representation may be suboptimal. We therefore use a part-decoupled design, as illustrated in Fig. 2.

The visual feature is encoded into upper-body and lowerbody streams via linear projection networks (MLPs) denoted as $E _ { u }$ and $E _ { l } ^ { v }$ :

$$
\mathbf { h } _ { t } ^ { u } = E _ { u } ( \mathbf { v } _ { t } ^ { b } ) , \quad \mathbf { h } _ { t } ^ { l , v } = E _ { l } ^ { v } ( \mathbf { v } _ { t } ^ { b } ) .\tag{1}
$$

The insole feature is encoded separately:

$$
\begin{array} { r } { \mathbf { h } _ { t } ^ { s } = E _ { s } ( \mathbf { s } _ { t } ) . } \end{array}\tag{2}
$$

The lower-body stream then fuses visual and insole features:

$$
\mathbf h _ { t } ^ { l } = E _ { l } ^ { f } ( [ \mathbf h _ { t } ^ { l , v } ; \mathbf h _ { t } ^ { s } ] ) ,\tag{3}
$$

where $[ \cdot ; \cdot ]$ denotes concatenation. This design injects footsensing cues into the body part where they are physically most informative, while keeping the upper-body stream mainly vision-driven.

The upper and lower streams are modeled by separate temporal modules:

$$
\mathbf { m } _ { 1 : T } ^ { u } = \mathrm { L S T M } _ { u } ( \mathbf { h } _ { 1 : T } ^ { u } ) , \quad \mathbf { m } _ { 1 : T } ^ { l } = \mathrm { L S T M } _ { l } ( \mathbf { h } _ { 1 : T } ^ { l } ) .\tag{4}
$$

While the decoupled streams prevent local foot-sensing noise from overriding upper-body visual observations, completely isolating the two branches violates the kinetic chain principle of the human body. The pelvis and lower spine act as a mechanical bridge; lower-limb momentum generated during walking inherently influences torso rotation and arm swinging. To maintain this whole-body kinematic consistency, a lightweight global coordinator acts as a soft kinetic constraint. It cross-examines the hidden states of both temporal memories and generates residual updates:

TABLE II  
MAIN POSE ESTIMATION RESULTS UNDER CLEAN, LOWER-BODY OCCLUSION, AND VISUAL DROPOUT CONDITIONS. MPJPE, PA-MPJPE, AND JITTER ARE LOWER-IS-BETTER.
<table><tr><td>Condition</td><td>Evaluation Scope</td><td>Backbone</td><td>Method</td><td>MPJPE (mm) ↓</td><td>PA-MPJPE (mm) ↓</td><td>Jitter  $\mathrm { ( m / s ^ { 3 } ) }$  →</td></tr><tr><td rowspan="6">Clean</td><td rowspan="6">Standard Subsets</td><td>MediaPipe</td><td>Vision Temporal</td><td>67.55</td><td>45.21</td><td>81.40</td></tr><tr><td>MediaPipe</td><td>Ours</td><td>64.15</td><td>40.62</td><td>65.05</td></tr><tr><td>YOLOPose26</td><td>Vision Temporal</td><td>68.13</td><td>45.37</td><td>80.28</td></tr><tr><td>YOLOPose26</td><td>Ours</td><td>63.94</td><td>41.57</td><td>69.35</td></tr><tr><td>SAT-HMR</td><td>Vision Temporal</td><td>63.96</td><td>42.64</td><td>68.29</td></tr><tr><td>SAT-HMR</td><td>Ours</td><td>58.38</td><td>38.06</td><td>66.11</td></tr><tr><td rowspan="6">Occlusion</td><td rowspan="6">Standard Subsets</td><td>MediaPipe</td><td>Vision Temporal</td><td>102.89</td><td>65.73</td><td>183.12</td></tr><tr><td>MediaPipe</td><td>Ours</td><td>68.59</td><td>43.72</td><td>82.65</td></tr><tr><td>YOLOPose26</td><td>Vision Temporal</td><td>112.38</td><td>76.70</td><td>110.30</td></tr><tr><td>YOLOPose26</td><td>Ours</td><td>66.35</td><td>42.78</td><td>79.03</td></tr><tr><td>SAT-HMR</td><td>Vision Temporal</td><td>76.75</td><td>50.88</td><td>93.38</td></tr><tr><td>SAT-HMR</td><td>Ours</td><td>61.84</td><td>41.28</td><td>81.95</td></tr><tr><td>Dropout</td><td>Dropout Subset</td><td></td><td>Ours</td><td>123.63</td><td>59.65</td><td>105.15</td></tr></table>

$$
[ \Delta \mathbf { m } _ { t } ^ { u } ; \Delta \mathbf { m } _ { t } ^ { l } ] = C _ { \mathrm { M L P } } ( [ \mathbf { m } _ { t } ^ { u } ; \mathbf { m } _ { t } ^ { l } ] ) .\tag{5}
$$

The coordinated memories are then used to regress upper-body and lower-body joints via MLP-based regressors $H , H _ { u }$ , and $H _ { l } \colon$

$$
\hat { \mathbf { P } } _ { t } = H \left( H _ { u } ( \mathbf { m } _ { t } ^ { u } + \Delta \mathbf { m } _ { t } ^ { u } ) , H _ { l } ( \mathbf { m } _ { t } ^ { l } + \Delta \mathbf { m } _ { t } ^ { l } ) \right) .\tag{6}
$$

This structure allows the model to exploit insole cues for lower-body refinement while preserving full-body kinematic coherence.

## B. Training Objective

The training objective is designed to supervise both framelevel pose accuracy and temporal motion quality. Since the output is a continuous root-relative skeleton sequence, only minimizing the joint position error may still lead to framewise fluctuations or locally distorted limb configurations. Following standard practices in temporal 3D human pose estimation [32]–[34], we combine a joint position loss with first- and second-order temporal smoothness terms, and a bone-length consistency constraint:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { p o s } + \lambda _ { v } \mathcal { L } _ { v e l } + \lambda _ { a } \mathcal { L } _ { a c c } + \lambda _ { b } \mathcal { L } _ { b o n e } . } \end{array}\tag{7}
$$

Here, $\mathcal { L } _ { p o s }$ penalizes joint position errors in the root-relative skeleton space and provides the main supervision for pose accuracy. $\mathcal { L } _ { v e l }$ and $\mathcal { L } _ { a c c }$ constrain first- and second-order temporal differences, respectively, encouraging the predicted motion to follow the reference dynamics and reducing highfrequency jitter. $\mathcal { L } _ { b o n e }$ preserves local kinematic structure by penalizing deviations in bone lengths, which helps avoid implausible limb deformation during temporal fusion. In our implementation, $\lambda _ { v } = 0 . 1 , \lambda _ { a } = 0 . 4 ,$ and $\lambda _ { b } = 0 . 0 2$

To improve robustness to temporary visual failures, we apply visual feature dropout during training. For selected training windows, a continuous temporal block of visual features is masked to zero while the synchronized insole stream remains available. The masked block length is randomly sampled within the sequence, so the model learns to use pressure and acceleration cues when visual observations are temporarily unreliable. This augmentation is used only during training.

## C. Real-Time Streaming Inference

During inference, the proposed fusion module operates in a streaming manner. At each time step, it receives the current visual pose feature and the synchronized insole measurement, while maintaining recurrent hidden states from previous frames. The current pose is predicted as:

$$
\begin{array} { r } { \hat { { \bf P } } _ { t } = f _ { \theta } ^ { o n l i n e } ( { \bf v } _ { t } ^ { b } , { \bf s } _ { t } , { \bf h } _ { t - 1 } ) , } \end{array}\tag{8}
$$

where $\mathbf { h } _ { t - 1 }$ denotes the temporal memory from the previous frame. This design avoids reprocessing the full sequence and keeps the additional overhead of the insole-enhanced module small, making it compatible with real-time visual pose backbones.

## V. EXPERIMENTS

## A. Experimental Setup

We evaluate our method on the new VPPA dataset (Section III). Three representative real-time visual backbones are evaluated: MediaPipe, YOLOPose26, and SAT-HMR. Reference body poses are obtained from Xsens MVN. All compared methods use a fixed action-disjoint split where test actions are unseen during training. For synthetic lower-body occlusion, we manually mask the bottom 30% of each video frame with black pixels before feature extraction. Training and validation use sliding windows of $T = 1 2 0$ frames with a step size of 5 frames.

TABLE III  
GAIT-RELATED MOTION PARAMETERS UNDER CLEAN, LOWER-BODY OCCLUSION, AND VISUAL DROPOUT CONDITIONS. STANCE ACCURACY IS HIGHER-IS-BETTER, WHILE STEP-DISTANCE ERROR AND CADENCE ERROR ARE LOWER-IS-BETTER.
<table><tr><td>Condition</td><td>Evaluation Scope</td><td>Backbone</td><td>Method</td><td>Stance Acc. (%) ↑</td><td>Step Dist. Err. (cm) ↓</td><td>Cadence Err. (step/min) ↓</td></tr><tr><td rowspan="6">Clean</td><td rowspan="6">Walking &amp; Jogging</td><td>MediaPipe</td><td>Vision Temporal</td><td>74.15</td><td>8.34</td><td>11.42</td></tr><tr><td>MediaPipe</td><td>Ours</td><td>79.78</td><td>5.54</td><td>6.23</td></tr><tr><td>YOLOPose26</td><td>Vision Temporal</td><td>67.89</td><td>6.69</td><td>16.21</td></tr><tr><td>YOLOPose26</td><td>Ours</td><td>78.16</td><td>4.74</td><td>4.15</td></tr><tr><td>SAT-HMR</td><td>Vision Temporal</td><td>66.95</td><td>6.91</td><td>10.50</td></tr><tr><td>SAT-HMR</td><td>Ours</td><td>78.03</td><td>5.56</td><td>8.36</td></tr><tr><td rowspan="6">Occlusion</td><td rowspan="6">Walking &amp; Jogging</td><td>MediaPipe</td><td>Vision Temporal</td><td>62.80</td><td>14.92</td><td>15.17</td></tr><tr><td>MediaPipe</td><td>Ours</td><td>80.15</td><td>5.73</td><td>6.23</td></tr><tr><td>YOLOPose26</td><td>Vision Temporal</td><td>60.43</td><td>12.38</td><td>21.80</td></tr><tr><td>YOLOPose26</td><td>Ours</td><td>77.69</td><td>5.65</td><td>3.11</td></tr><tr><td>SAT-HMR</td><td>Vision Temporal</td><td>62.86</td><td>11.84</td><td>24.57</td></tr><tr><td>SAT-HMR</td><td>Ours</td><td>76.99</td><td>6.93</td><td>6.42</td></tr><tr><td>Dropout</td><td>Dropout Walking</td><td></td><td>Ours</td><td>74.08</td><td>9.55</td><td>12.37</td></tr></table>

## B. Baselines and Compared Variants

We compare the proposed method with a vision-only temporal baseline and several fusion variants. The vision-only temporal baseline uses the same visual backbone output and temporal modeling strategy as our method, but removes the insole stream. This baseline is important because it separates the benefit of pressure–inertial sensing from the benefit of temporal smoothing alone. For each visual backbone, the baseline is trained and evaluated in the same root-relative 22- joint output space as the proposed model.

We also evaluate a naive temporal fusion variant, which directly concatenates visual and insole features before temporal modeling without part decoupling or cross-branch coordination. This variant tests whether simple multimodal concatenation is sufficient. In addition, we report ablations without the global coordinator, without pressure, and without acceleration. These variants examine the contribution of cross-branch information exchange and the two sensing modalities. All compared methods use the same data split, window length, training objective, and evaluation metrics for a fair comparison.

## C. Implementation Details

All models are trained on 120-frame windows. The training and validation windows are sampled with a step size of 5 frames, while testing uses a step size of 1 frame to obtain dense frame-level predictions. The proposed fusion model uses two-layer LSTM temporal modules with a hidden dimension of 256. The upper- and lower-body branches use separate temporal memories, and the global coordinator predicts residual updates for both branches. We train the model using AdamW with cosine learning-rate scheduling and select the best checkpoint according to validation PA-MPJPE. For fair comparison, the vision-only temporal baseline and fusion variants are trained under the same data split, window length, and evaluation protocol.

## D. Evaluation Metrics

We report MPJPE and PA-MPJPE for pose accuracy, and jerk-based jitter for temporal smoothness. MPJPE measures the mean Euclidean distance between the predicted and reference joints in the root-relative skeleton space, while PA-MPJPE is computed after Procrustes alignment. Jitter is computed from the third-order temporal derivative of the predicted joint trajectories and reflects high-frequency motion instability.

For gait-related evaluation, we report stance accuracy, stepdistance error, and cadence error. These metrics are computed from the reconstructed foot trajectories and are used to examine whether the estimated pose preserves contact-sensitive motion properties beyond joint-level accuracy.

To ensure biomechanical validity and clinical relevance, gait-related metrics are evaluated exclusively on standard cyclic locomotion sequences within the test split (specifically, walking and jogging). This selective evaluation prevents the introduction of ill-defined noise from static or non-locomotive activities (such as sitting, eating, or upperbody exercises), where the concepts of stride geometry and gait cycle are physically inapplicable.

Stance accuracy evaluates frame-level left/right foot-contact correctness. For each frame, we infer the predicted contact state from the reconstructed foot trajectory and compare it with the reference contact label derived from the motioncapture system. The smart-insole pressure signal is not used as the ground-truth contact label for this metric, preventing the evaluation from directly reusing the input sensing modality.

Step-distance error measures the accuracy of step geometry. We first identify heel-strike events from the reference contact sequence. At each detected heel-strike frame, we compute the horizontal distance between the left and right ankle joints. The reported error is the mean absolute difference between the predicted and reference step distances.

Cadence error measures the accuracy of temporal step rhythm. Cadence is computed as the number of detected steps per minute over an evaluated locomotion sequence. Together, these gait-related metrics strictly evaluate whether the reconstructed motion is useful for downstream gait and contact analysis.

## E. Main Pose Estimation Results

Table II reports the main pose estimation results across three evaluation scopes: clean visual conditions, lower-body occlusion, and visual dropout (out-of-frame).

Under clean visual observations on standard subsets, the proposed method consistently improves MPJPE, PA-MPJPE, and jitter across MediaPipe, YOLOPose26, and SAT-HMR. This indicates that the insole stream provides useful complementary cues even when the visual input is not deliberately degraded. The improvement becomes more pronounced under lower-body occlusion. Since the bottom 30% of the video frame is masked, the visual backbones receive weaker lowerbody evidence, while the insole stream still provides pressure and acceleration cues related to foot-ground interaction. As a result, the proposed method substantially reduces pose error and temporal jitter compared with the vision-only temporal baselines.

Crucially, we also report quantitative results under the extreme visual-dropout condition (Table II, bottom row). During visual dropouts, the subject temporarily leaves the camera view. Because vision-only baselines completely lose physical observability and degenerate to static holding or unpredictable extrapolation without image evidence, they cannot serve as a meaningful comparative baseline (denoted as ‘—‘). In stark contrast, the proposed insole-enhanced method reliably leverages continuous plantar pressure and acceleration to maintain active tracking of the lower limbs, achieving an MPJPE of 123.63 mm and a PA-MPJPE of 59.65 mm. This further supports our core claim: unobtrusive foot sensing provides critical physical constraints exactly when monocular cameras fail.

## F. Gait-Related Evaluation

Table III reports gait-related measurements specifically evaluated on cyclic locomotion (walking and jogging).

In the clean setting, the proposed method improves stance accuracy for all three backbones and reduces both stepdistance and cadence errors. These results indicate that the insole-enhanced model does not merely reduce average joint error, but fundamentally improves foot-contact timing, relative step geometry, and step rhythm. On average, stance accuracy increases by 8.99 percentage points, step-distance error decreases by 27.8%, and cadence error decreases by 50.9%.

The improvement is larger under lower-body occlusion. Vision-only baselines produce less reliable foot-contact and step-event estimates when lower-body observations are degraded. In contrast, the insole-enhanced model remains remarkably stable because plantar pressure and foot acceleration directly encode contact and foot dynamics. Under occlusion, the average stance accuracy increases by 16.25 percentage points.

Marching

Jogging

Jumping jacks

Alternating lunges

![](images/75f6a46deeed94ef0d4c662d13487c6ab309506c4b6fb33f9e33085c3ec87218.jpg)  
Fig. 3. Qualitative comparison of full-body pose estimation on diverse highlydynamic actions. Rows from top to bottom correspond to the input monocular images, poses estimated by our proposed method (Ours), and the Xsens reference ground truth, respectively.

TABLE IV  
CROSS-SUBJECT GENERALIZATION RESULTS ON HELD-OUT TEST SUBJECTS USING MEDIAPIPE. ALL METRICS ARE LOWER-IS-BETTER.
<table><tr><td>Condition Method</td><td></td><td>MPJPE (mm) ↓ PA-MPJPE (mm) ↓ Jitter (m/s3) ↓</td><td></td><td></td></tr><tr><td rowspan="2">Clean</td><td>Temporal</td><td>73.47</td><td>48.86</td><td>89.02</td></tr><tr><td>Ours</td><td>68.06</td><td>45.78</td><td>75.64</td></tr><tr><td rowspan="2">Occlusion</td><td>Temporal</td><td>88.31</td><td>59.20</td><td>115.69</td></tr><tr><td>Ours</td><td>74.48</td><td>50.43</td><td>83.21</td></tr></table>

Furthermore, when evaluating locomotion during complete visual dropouts (Dropout Walking), the proposed method impressively maintains a stance accuracy of 74.08%, a stepdistance error of 9.55 cm, and a cadence error of 12.37 step/min. This demonstrates that even when the camera captures nothing, the smart insole continuously drives the temporal fusion network to output biomechanically plausible gait parameters.

## G. Cross-Subject Generalization

To further examine whether the benefit of pressure–inertial foot sensing generalizes to unseen users, we conduct an additional subject-disjoint experiment using MediaPipe as the visual backbone. In this setting, models are trained on 15 subjects, while two subjects are used for validation and three held-out subjects are reserved purely for testing. We evaluate both clean and desk-occlusion conditions on the standard actions.

As shown in Table IV, the proposed method still outperforms the vision-only temporal baseline on unseen subjects. Under clean conditions, it reduces MPJPE from 73.47 mm to 68.06 mm and decreases jitter from 89.02 to 75.64 m/s<sup>3</sup>. The advantage becomes larger under occlusion, where MPJPE is reduced from 88.31 mm to 74.48 mm and jitter is reduced from 115.69 to 83.21 $\mathrm { m } / \mathrm { s } ^ { 3 }$ . These results indicate that the benefit of pressure–inertial foot sensing is not limited to the calibrateduser action-disjoint setting, but can also improve robustness for held-out users.

TABLE V  
REAL-OCCLUSION STRESS TEST ON TWO HELD-OUT DESK-INDUCED OCCLUSION SEQUENCES USING MEDIAPIPE. ALL METRICS ARE LOWER-IS-BETTER.
<table><tr><td>Method</td><td>MPJPE (mm) ↓ PA-MPJPE (mm) ↓ Jitter  $\mathrm { ( m / s ^ { 3 } ) . }$ </td></tr><tr><td>MediaPipe Temporal</td><td></td></tr><tr><td>93.17 65.58 178.25</td><td></td></tr><tr><td>62.06 45.79</td><td>94.00</td></tr></table>

## H. Real-Occlusion Stress Test

We further evaluate the proposed method on two held-out real-occlusion sequences recorded with a physical desk. In both sequences, the desk is placed between the camera and the subject so that the lower body is fully invisible to the monocular camera. The first sequence is a stationary lowerbody occlusion case, where the subject keeps the lower body nearly static while performing free upper-body movements. This setting tests whether the model can avoid visually induced lower-body drift when only upper-body motion is observable. The second sequence is a desk-occluded walking case, where the subject walks while the legs and feet remain hidden behind the desk. This setting is more challenging because the visual stream provides little direct evidence about step timing, foot displacement, or lower-limb articulation.

The synthetic and real occlusion settings serve different purposes. Synthetic lower-body occlusion masks the same actions used in the standard protocol, enabling controlled comparison with paired clean sequences. In contrast, deskinduced occlusion introduces physical scene-level obstruction, viewpoint-dependent visual ambiguity, and unpaired action sequences. Therefore, the real-occlusion actions are not used as a standard benchmark with clean counterparts, but as a stress test for practical obstacle-induced visual degradation.

As shown in Table V, the proposed method remains substantially more robust than the vision-only temporal baseline under physical occlusion. It reduces MPJPE from 93.17 mm to 62.06 mm, PA-MPJPE from 65.58 mm to 45.79 mm, and jitter from 178.25 to $9 4 . 0 0 ~ \mathrm { m / s ^ { 3 } }$ . These results complement the synthetic occlusion experiment: the synthetic setting provides controlled comparison under the same actions, while the realocclusion setting verifies robustness under practical obstacleinduced visual degradation.

## I. Ablation Study

Table VI presents the ablation results on the MediaPipe clean setting. The vision-only temporal baseline already provides a strong comparison because it uses temporal modeling without insole input. Directly concatenating visual and insole features, denoted as Naive Temporal Fusion, improves MPJPE and PA-MPJPE over the vision-only baseline, showing that foot sensing is useful even with a simple fusion design. However, it remains inferior to the proposed part-decoupled fusion network.

![](images/9d0374a715a5bc77414b17f1f4f97efc4c90895b224b857516453a8f059ce9d2.jpg)  
Fig. 4. Qualitative behavior under short visual dropout. Rows show three temporal frames in order: an in-frame moment followed by two dropout moments. In each row, the columns from left to right are GT, MediaPipe Temporal, and Ours. During dropout, the vision-only baseline has no new visual observation and therefore degenerates to last-observation holding or temporal-prior extrapolation. The proposed model continues to update the articulated pose using pressure and acceleration cues from the smart insole.

TABLE VI  
ABLATION STUDY ON THE MEDIAPIPE CLEAN SETTING. ALL METRICS ARE LOWER-IS-BETTER.
<table><tr><td>Method</td><td>MPJPE (mm) ↓ PA-MPJPE (mm) ↓ Jitter</td><td> $\mathrm { ( m / s ^ { 3 } ) . }$  1</td></tr><tr><td>Vision Temporal</td><td>67.55</td><td>45.21 81.40</td></tr><tr><td>Naive Temporal Fusion</td><td>66.90</td><td>43.09 77.05</td></tr><tr><td>w/o Coordinator</td><td>66.25</td><td>43.88 80.25</td></tr><tr><td>w/o Pressure</td><td>65.07</td><td>42.20 73.00</td></tr><tr><td>w/o Acceleration</td><td>66.73</td><td>43.55 67.38</td></tr><tr><td>Ours</td><td>64.15</td><td>40.62 65.05</td></tr></table>

Removing the global coordinator leads to worse PA-MPJPE and jitter, indicating that cross-branch information exchange is needed to maintain whole-body consistency after separating upper- and lower-body streams. Removing pressure or acceleration also degrades the performance. Without pressure, the model loses direct contact and load-transfer cues; without acceleration, it loses dynamic information during swing phases. The full model achieves the best performance across all three metrics, demonstrating that both the sensing modalities and the part-decoupled fusion design contribute to the final improvement.

The two sensing components play complementary roles. Plantar pressure is more directly related to stance, support phase, and load transfer, so it is especially useful for correcting foot-contact timing and lower-body geometry when visual evidence is weak. Foot-mounted acceleration provides dynamic information during swing phases, where pressure may be close to zero and contact cues alone become ambiguous. This explains why removing either modality degrades the result, and why the full pressure–inertial representation provides the most consistent improvement.

## J. Short Visual-Dropout Analysis

In the previous sections (Tables II and III), we quantitatively demonstrated the exceptional robustness of the proposed method under short visual-dropout conditions. To intuitively understand the underlying tracking mechanism during these extreme visual degradation scenarios, we conduct a qualitative behavioral analysis.

This setting is fundamentally different from lower-body occlusion. Under lower-body occlusion, the visual backbone still receives partial image evidence (e.g., upper body or surrounding context). During out-of-frame dropout, however, the visual stream provides absolutely no valid observation of the ongoing motion. In our implementation, the vision-only baseline therefore completely freezes, degenerating to lastobservation holding or unpredictable temporal-prior extrapolation.

As illustrated in Figure 4, the rows show continuous temporal frames transitioning from an in-frame moment to an out-offrame dropout. When the visual stream becomes unavailable, the vision-only MediaPipe baseline remains stuck in its last visible posture. However, our proposed model continues to accurately update lower-body articulation (e.g., knee lifting and step advancing) using continuous plantar pressure and foot-mounted acceleration. This result visually corroborates the quantitative metrics: while foot sensing is not intended to replace vision for infinite long-duration blind tracking, it decisively provides the essential physical observability needed to survive short, inevitable camera failures in real-world deployments.

## K. Mobile Runtime Overhead

We further benchmark the runtime overhead of the proposed fusion module on three consumer Android smartphones. Since the proposed method is designed to operate after a realtime visual pose backbone has produced pose features, this benchmark focuses on the additional neural inference latency introduced by the proposed vision–pressure–inertial fusion module. We use pre-recorded synchronized and normalized visual and foot-sensing feature streams as frame-by-frame inputs, and run the model in streaming mode with batch size 1 while carrying recurrent hidden states online. Data loading and 200 warm-up frames are excluded from the latency measurement.

As shown in Table VII, the proposed fusion module achieves sub-1.5 ms average latency and sub-1.5 ms p95 latency on all tested devices, including an older mid-range phone. Even on the OPPO Reno3 5G, the p95 latency is only 1.47 ms, far below the 33.33 ms frame budget required for 30 FPS operation. These results indicate that the proposed fusion module introduces negligible computational overhead once real-time visual pose features are available. The model size is 12.10 MB in ONNX format.

TABLE VII  
MOBILE RUNTIME OVERHEAD OF THE PROPOSED FUSION MODULE. LATENCY IS MEASURED USING ONNX RUNTIME ANDROID ON CPU WITH SYNCHRONIZED FEATURE STREAMS, BATCH SIZE 1, STREAMING INFERENCE, AND 200 WARM-UP FRAMES. THE ONNX MODEL SIZE IS 12.10 MB.
<table><tr><td>Device</td><td>SoC / Chipset (Node)</td><td>Avg. (ms)</td><td>p95 (ms)</td><td>p99 (ms)</td><td>FPS</td></tr><tr><td>vivo X200 Pro</td><td>MediaTek Dimensity 9400 (3 nm)</td><td>0.44</td><td>0.68</td><td>0.80</td><td>2264.46</td></tr><tr><td>Redmi Note 12</td><td>Snapdragon 7+ Gen 2 (4 nm)</td><td>1.04</td><td>1.15</td><td>1.33</td><td>960.88</td></tr><tr><td>OPPO Reno3</td><td>MediaTek Dimensity 1000L (7 nm)</td><td>1.32</td><td>1.47</td><td>2.00</td><td>758.59</td></tr></table>

It should be noted that this runtime measures only the proposed fusion module after visual pose features have been produced. The end-to-end system latency depends on the selected visual backbone, camera pipeline, and platform-specific preprocessing. This design is intentional: the proposed module is not tied to a particular visual estimator, and its role is to provide a lightweight pressure–inertial refinement layer that can be attached to different real-time pose backbones.

## VI. CONCLUSION AND FUTURE WORK

This paper presented a part-decoupled vision–pressure– inertial fusion framework for real-time monocular human pose estimation. We investigated whether sparse plantar-pressure and foot-acceleration signals can provide complementary cues when monocular visual evidence is incomplete or unavailable. The framework maps heterogeneous visual observations into a unified root-relative skeleton space, incorporates foot sensing into lower-body temporal modeling, and uses a lightweight global coordinator to maintain whole-body consistency.

Experimental results show that the insole-enhanced framework improves pose accuracy and temporal smoothness over vision-only temporal baselines under clean conditions, with larger gains under lower-body occlusion. Analysis of short visual dropouts further demonstrates the value of continuous foot-contact and foot-dynamic cues. The method also improves contact-sensitive gait measurements, including stance accuracy, step-distance error, and cadence error. These results indicate that sparse pressure–inertial foot sensing provides useful physical priors for robust human motion estimation.

Although the experiments use I-SOL smart insoles, the formulation only assumes synchronized sparse plantar-pressure and foot-mounted acceleration streams with comparable temporal resolution. The method is therefore not limited to a specific insole model. Data organization, action definitions, and preprocessing scripts will be released to support reproducibility.

The method nevertheless has several limitations. First, it estimates root-relative pose rather than long-term global trajectories. Because the current sensing setup lacks reliable global orientation, extended visual dropouts may still cause drift. Second, broader evaluation across users, environments, and motion types is needed. Third, downstream applications such as ergonomic risk assessment, rehabilitation scoring, and abnormal gait screening require further validation.

Future work will improve long-horizon robustness through lightweight global motion constraints and explicit contact-state modeling. We will also explore mobile and edge deployment with real-time visual front-ends and Bluetooth insole streams, and extend evaluation to industrial monitoring, home rehabilitation, and long-term daily activity assessment.

## REFERENCES

[1] H.-S. Fang, J. Li, H. Tang, C. Xu, H. Zhu, Y. Xiu, Y.-L. Li, and C. Lu, “Alphapose: Whole-body regional multi-person pose estimation and tracking in real-time,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 6, pp. 7157–7173, 2023.

[2] P. Lu, T. Jiang, Y. Li, X. Li, K. Chen, and W. Yang, “Rtmo: Towards high-performance one-stage real-time multi-person pose estimation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 1491–1500.

[3] Q. Zhao, C. Zheng, M. Liu, P. Wang, and C. Chen, “Poseformerv2: Exploring frequency domain for efficient and robust 3d human pose estimation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[4] W. Zhu, X. Ma, Z. Liu, L. Liu, W. Wu, and Y. Wang, “Motionbert: A unified perspective on learning human motion representations,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

[5] Q. Fang, Z. Xu, M. Hu, Q. Zeng, C. Liu, and Q. Chen, “SPGformer: Serial–parallel hybrid GCN-transformer with graph-oriented encoder for 2-d-to-3-d human pose estimation,” IEEE Transactions on Instrumentation and Measurement, vol. 73, pp. 1–15, 2024.

[6] S. Goel, G. Pavlakos, J. Rajasegaran, A. Kanazawa, and J. Malik, “Humans in 4d: Reconstructing and tracking humans with transformers,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

[7] S. Shin, J. Kim, E. Halilaj, and M. J. Black, “Wham: Reconstructing world-grounded humans with accurate 3d motion,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2024, pp. 2070–2080.

[8] J. Gong, L. G. Foo, Z. Fan, Q. Ke, H. Rahmani, and J. Liu, “Diffpose: Toward more reliable 3d pose estimation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 13 041–13 051.

[9] Z. Sun, Y. Liang, Z. Ma, T. Zhang, L. Bao, G. Li, and S. He, “Repose: 3d human pose estimation via spatio-temporal depth relational consistency,” in European Conference on Computer Vision (ECCV), 2024, pp. 309– 325.

[10] H. Zhang, S. Ren, H. Yuan, J. Zhao, F. Li, S. Sun, Z. Liang, T. Yu, Q. Shen, and X. Cao, “Mmvp: A multimodal mocap dataset with vision and pressure sensors,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2024, pp. 21 842–21 852.

[11] E. Wu, R. Khirodkar, H. Koike, and K. Kitani, “Soleposer: Full body pose estimation using a single pair of insole sensor,” in Proceedings of the 37th Annual ACM Symposium on User Interface Software and Technology (UIST), 2024, pp. 67:1–67:9.

[12] Y. Gao, W. Zhang, J. Ren, R. Zheng, Y. Jin, D. Wu, L. Shu, X. Xu, and Z. Jin, “Pressinpose: Integrating pressure and inertial sensors for full-body pose estimation in activities,” Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, vol. 8, no. 4, pp. 197:1–197:28, 2024.

[13] Q. Ying, Z. Cao, Z. Wu, W. Deng, Y. Zhong, Y. Diao, and X. Cai, “Form: Foot-driven reconstruction of human motion using dual-modal plantar pressure and inertial sensing,” Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, vol. 9, no. 3, 2025.

[14] Y. Zhang, S. Xia, L. Chu, J. Yang, Q. Wu, and L. Pei, “Dynamic inertial poser (dynaip): Part-based motion dynamics learning for enhanced human pose estimation with sparse inertial sensors,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 1889–1899.

[15] C. Xu, Y. Tan, Z. Strout, G. Liu, K. Zhu, H. Wang, and P. B. Shull, “Real-time opensim via imus for full body kinematics during gait, sports, exercise, and dance movements,” IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 34, pp. 650–662, 2026.

[16] D. Maji, S. Nagori, M. Mathew, and D. Poddar, “Yolo-pose: Enhancing yolo for multi person pose estimation using object keypoint similarity loss,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, June 2022, pp. 2637–2646.

[17] T. Jiang, P. Lu, L. Zhang, N. Ma, R. Han, C. Lyu, Y. Li, and K. Chen, “Rtmpose: Real-time multi-person pose estimation based on mmpose,” arXiv preprint arXiv:2303.07399, 2023.

[18] Z. Yang, A. Zeng, C. Yuan, and Y. Li, “Effective whole-body pose estimation with two-stages distillation,” in Proceedings ofthe IEEE/CVF International Conference on Computer Vision Workshops, 2023, pp. 4210–4220.

[19] Y. Xu, J. Zhang, Q. Zhang, and D. Tao, “Vitpose: Simple vision transformer baselines for human pose estimation,” in Advances in Neural Information Processing Systems, vol. 35, 2022, pp. 38 571–38 584.

[20] V. Bazarevsky, I. Grishchenko, K. Raveendran, T. Zhu, F. Zhang, and M. Grundmann, “Blazepose: On-device real-time body pose tracking,” arXiv preprint arXiv:2006.10204, 2020.

[21] S. Mehraban, V. Adeli, and B. Taati, “Motionagformer: Enhancing 3d human pose estimation with a transformer-gcnformer network,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2024, pp. 6920–6930.

[22] Z. Cai, M. Zhang, J. Ren, C. Wei, D. Ren, Z. Lin, H. Zhao, L. Yang, C. C. Loy, and Z. Liu, “Playing for 3d human recovery,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 12, pp. 10 533–10 545, 2024.

[23] Y. Wang, Y. Sun, P. Patel, K. Daniilidis, M. J. Black, and M. Kocabas, “Prompthmr: Promptable human mesh recovery,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 1148–1159.

[24] C. Su, X. Ma, J. Su, and Y. Wang, “Sat-hmr: Real-time multi-person 3d mesh estimation via scale-adaptive tokens,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2025, pp. 16 796–16 806.

[25] S. Peng, H. Hassan, S. Rosseel, G. A. Matricali, K. Deschamps, V. Vandeginste, and H. Hallez, “Recent advances in 3-d printed, wearable pressure sensors for plantar pressure monitoring: A review,” IEEE Sensors Journal, vol. 24, no. 21, pp. 33 903–33 921, 2024.

[26] F. Weigand, A. Hohl, J. Zeiss, U. Konigorski, and M. Grimmer,¨ “Continuous locomotion mode recognition and gait phase estimation based on a shank-mounted imu with artificial neural networks,” in 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2022, pp. 12 744–12 751.

[27] S. Li, T. Zhu, M. Nie, H. Ning, Z. Liu, and L. Chen, “P2lhap: Wearablesensor-based human activity recognition, segmentation, and forecast through patch-to-label seq2seq transformer,” IEEE Internet of Things Journal, vol. 12, no. 6, pp. 6818–6830, 2025.

[28] S. Li, T. Zhu, F. Duan, L. Chen, H. Ning, C. Nugent, and Y. Wan, “Harmamba: Efficient and lightweight wearable sensor human activity recognition based on bidirectional mamba,” IEEE Internet of Things Journal, vol. 12, no. 3, pp. 2373–2384, 2025.

[29] V. Xu, C. Gao, H. Hoffmann, and K. Ahuja, “Mobileposer: Real-time full-body pose estimation and 3d human translation from imus in mobile consumer devices,” in Proceedings of the 37th Annual ACM Symposium on User Interface Software and Technology (UIST), 2024, pp. 70:1– 70:11.

[30] S. Tripathi, L. Muller, C.-H. P. Huang, O. Taheri, M. J. Black, and¨ D. Tzionas, “3d human pose estimation via intuitive physics,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2023, pp. 4713–4725.

[31] S. Ren, Y. Lu, J. Huang, J. Zhao, H. Zhang, T. Yu, Q. Shen, and X. Cao, “Motionpro: Exploring the role of pressure in human mocap and beyond,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2025, pp. 27 760–27 770.

[32] D. Pavllo, C. Feichtenhofer, D. Grangier, and M. Auli, “3d human pose estimation in video with temporal convolutions and semi-supervised training,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 7753–7762.

[33] M. Kocabas, N. Athanasiou, and M. J. Black, “Vibe: Video inference for human body pose and shape estimation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 5253–5263.

[34] X. Zhou, Q. Huang, X. Sun, X. Xue, and Y. Wei, “Towards 3d human pose estimation in the wild: a weakly-supervised approach,” in Proceedings of the IEEE international conference on computer vision, 2017, pp. 398–407.

![](images/d7bff1963b5e0dbf0428cec1152c81a3cb714262d7f40ee8a86a34b7d8db52c5.jpg)

Liang Yu received the bachelor’s degree from Qinghai Minzu University, Xining, China. He is currently pursuing the M.S. degree with the School of Computer Science, University of South China, Hengyang, China. His research interests include intelligent perception and pattern recognition.

![](images/03d8436274bec4236b172f72e2fc826e53402ff3e1dfe7e510f5ef9e681c47cc.jpg)

Tao Zhu (Senior Member, IEEE) received the B.E. degree from Central South University, Changsha, China, in 2009, and the Ph.D. degree from the University of Science and Technology of China, Hefei, China, in 2015. He is currently an Associate Professor with the University of South China, Hengyang, China. He has served as the Principal Investigator of several projects funded by the National Natural Science Foundation of China and the Natural Science Foundation of Hunan Province. His research interests include the Internet of Things, pervasive computing, assisted living, and evolutionary computation. Dr. Zhu is currently the Chair of the IEEE Computational Intelligence Society Smart World Technical Committee Task Force on User-Centred Smart Systems.

![](images/28026cb30f12c4159e4895809b97af06c1f7db0d5c7af4dd866b5d8efe93397d.jpg)

Xiangqing Lu received the bachelor’s degree from Qilu University of Technology (Shandong Academy of Sciences), Jinan, China. He is currently pursuing the M.S. degree with the School of Computer Science, University of South China, Hengyang, China. His research interests include intelligent perception and pattern recognition.

![](images/4cbaa6ea33dc94ed8787f747bfcdf88b86572757fb4f5715299082d260896391.jpg)

Liming Chen (Senior Member, IEEE) received the M.S. degree in vehicle engineering from the Beijing Institute of Technology, Beijing, China, in 1988, and the Ph.D. degree in computer science from De Montfort University, Leicester, U.K., in 2001. He is currently a Professor with the School of Computer Science and Technology, Dalian University of Technology, Dalian, China. His research interests include pervasive computing, data analytics, artificial intelligence, user-centered intelligent systems, digital healthcare, and cybersecurity. He has published more than 250 papers in these areas. Prof. Chen is a Fellow of the Institution of Engineering and Technology.

![](images/9ac15b9ab4d9ce51037f2df2a1776123cccd2e5b241b10e766b01ab012c6fb1d.jpg)

Wenjian Luo (Senior Member, IEEE) received the B.S. degree in computer software and the Ph.D. degree in computer application technology from the University of Science and Technology of China, Hefei, China, in 1998 and 2003, respectively. He is currently a Professor with the School of Computer Science and Technology, Harbin Institute of Technology, Shenzhen, China. His research interests include artificial intelligence security, privacypreserving computing, big-data analytics, machine learning, immune computing, and swarm intelli-

gence. He is a Distinguished Member of CCF and a Senior Member of ACM and CAAI.