# PalmSpace: Towards a Versatile On-Palm Interaction Space through Unified Touch Modeling

CHENTAO LI, Department of Automation, Tsinghua University, China

MINGZE GAO, Academy of Arts & Design, Tsinghua University, China

RUNZE SUN, Department of Automation, Tsinghua University, China

ZHAOGUO WANG, Department of Automation, Tsinghua University, China

JIANJIANG FENG<sup>∗</sup>, Department of Automation, Tsinghua University, China

JIE ZHOU, Department of Automation, Tsinghua University, China

![](images/2921e6d0e592c7df230c523b0a0fe168c437fb8a181822130d39cf8d93c8a88e.jpg)  
Fig. 1. Diagram of the core concept of PalmSpace. PalmSpace turns the palm into a structured and eyes-free interaction space. The four panels demonstrate: (1) a flip-up camera design, (2) a structured interaction space that supports complex map manipulation requiring both single-finger localization and multi-finger scroll/pinch input; (3) absolute positioning across the entire palm area including fingers, and (4) support for precision-demanding interactions like handwriting.

As smart glasses and lightweight MR devices become increasingly practical, input remains a key challenge. The bare palm is an always-available, tactile, and proprioceptively accessible surface, but it has neither an explicit coordinate system nor embedded touch sensing. Prior on-palm systems typically expose isolated touch events, discrete regions, continuous trajectories, or task-specific gestures, limiting the palm’s ability to support precise selection and gesture manipulation through a common input representation. We present PalmSpace, a wrist-worn infrared system that exposes mode-aware, body-referenced absolute input on the bare palm without per-user sensing calibration. At the interaction level, PalmSpace jointly represents contact occurrence, interaction mode, and palm-referenced absolute location; at the model level, it learns these coupled outputs through a shared real-time representation. In leave-one-participant-out evaluation with 17 participants, PalmSpace achieved 6.7 mm mean localization error, 98.9% contact detection accuracy, and 96.7% F1 for four-class interaction-state recognition. User studies further demonstrated absolute pointing and dragging, eyes-free digit input, and representative multi-finger controls including scrolling and pinch-based map manipulation. These result show that a morphologically variable bare palm can function as a transferable, mode-aware interaction surface.

CCS Concepts: • Human-centered computing → Interaction techniques.

Additional Key Words and Phrases: Palm-based input, Touch input, Smart glasses, Wearable interaction, Gesture classification

## 1 Introduction

Head-worn displays are again drawing attention as smart glasses and lightweight mixed-reality devices move closer to practical everyday use [16, 35]. Yet input remains a central challenge. Everyday wearable interaction demands techniques that are always available, socially unobtrusive, visually undemanding, and precise enough for both quick commands and fine-grained manipulation. While mice and touchpads remain the gold standard for precise pointing, they depend on dedicated hardware and stable external surfaces, making them poorly suited to lightweight head-worn use in mobile settings. Existing alternatives such as eye tracking, mid-air hand input, and compact wearable controllers avoid bulky external hardware, but often come with trade-ofs such as visual demand, fatigue, lack of haptic feedback, or reliance on relative rather than absolute control [29, 31, 34, 45, 47, 53, 56, 60]. This makes on-body input particularly attractive for smart glasses: it keeps the interaction surface always available on the body itself while preserving tactile and proprioceptive cues.

Among on-body input surfaces, the palm is especially promising: it is always available, naturally private, and rich in tactile and proprioceptive cues. Yet unlike a touchscreen, the bare palm has neither an explicit coordinate system nor embedded sensing for determining when and how input occurs. In terms of practicality, camera-based approaches are inherently viewpoint-dependent, making touch inference vulnerable to limited fields of view, self-occlusion, oblique viewing angles, and weak depth cues [16, 46]. Systems augmented with projectors or depth sensors can improve sensing quality, but their additional bulky hardware substantially reduces deployment practicality [9, 14]. Even wearable solutions often still rely on user-specific calibration or training, while ofering limited robustness in touch detection and localization [58, 63, 64, 78]. More fundamentally, prior systems tend to expose only one layer of input semantics: discrete touch zones, isolated touch events, continuous positions, or task-specific gestures [58, 63]. Most also lack explicit modeling of multi-finger modes alongside absolute position. As a result, the palm is still treated more as a collection of sparse events than as a structured interaction space. What is missing is a common representation of palm interaction that jointly encodes three tightly coupled properties: whether contact occurs, where it occurs in body-referenced absolute coordinates, and what interaction mode it belongs to. For example, in smart-glasses map interaction, users may first use touch detection and absolute localization to select or drag toward a region of interest, and then switch to multi-finger input for zooming and rotation. Representing these properties together allows precise targeting and gesture control to coexist on the same bare-palm surface.

To this end, we present PalmSpace, a wrist-worn system that provides unified on-palm input at two complementary levels. At the interaction level, it exposes contact semantics, interaction mode, and palm-referenced absolute location through a common representation. At the model level, it learns interaction state and location through a shared multi-task representation rather than independent task-specific pipelines. We instantiate this formulation with a wrist-worn infrared sensing setup, a palm-normalized coordinate space that transfers across users without per-user sensing calibration, and a real-time multi-task Transformer. Together, these components support precise single-finger input and representative multi-finger gesture control on the same bare-palm surface.

In ofline evaluation using leave-one-participant-out cross-validation, PalmSpace achieved a mean absolute localization error of 6.7 mm, 98.9% accuracy for binary contact detection, and a 96.7% F1-score for four-class interaction-state Manuscript submitted to ACM

recognition. An additional ablation on unseen outdoor data shows that the unified model improves both localization and touch recognition over independently trained task-specific models while requiring only one deployed model.

To evaluate PalmSpace in realistic settings, we conducted user studies of Fitts’s Law pointing, eyes-free digit entry, handwriting, list navigation, and map manipulation, together with outdoor evaluations under daytime and low-light conditions.

In summary, the main contributions of this paper are as follows:

• We present PalmSpace, an on-palm input system that jointly represents contact occurrence, interaction mode, and body-referenced absolute location, turning the bare palm into a mode-aware interaction surface without per-user sensing calibration.

• We instantiate this representation through wrist-worn infrared sensing, palm-normalized coordinates, and a real-time multi-task Transformer. Compared with independently trained task-specific models, unified learning improves zero-shot outdoor localization and touch recognition while avoiding a second deployed model.

• Through cross-user evaluation and indoor/outdoor user studies, we characterize PalmSpace across absolute pointing, eyes-free symbolic input, and representative multi-finger gesture control, achieving 6.7 mm localization error, 98.9% binary contact accuracy, and 96.7% four-class F1.

## 2 Related Work

Table 1. Comparison of sensing setup, input representation, and evaluation scope across existing on-palm techniques. PalmSpace combines contact-aware continuous absolute positioning with multi-finger mode recognition without per-user sensing calibration (‘C’: continuous, ‘D’: discrete)
<table><tr><td>Study</td><td>Year</td><td>Sensor Type</td><td>Sensor Placement</td><td>Real-world Evaluation</td><td>Sensing Calibration Free</td><td>Eyes Free</td><td>Absolute Positioning</td><td>2D Finger Tracking</td><td>Multi-Finger Mode</td><td>Error</td><td>Touch Detection</td><td>Touch/Gesture Accuracy</td></tr><tr><td>Kohli and Whitton [30]</td><td>2005</td><td>Magnetic trackers</td><td>Finger+Back</td><td>x</td><td>×</td><td>×</td><td>√</td><td>C</td><td>X</td><td>一</td><td>√</td><td></td></tr><tr><td>PalmRC [9]</td><td>2012</td><td>Depth camera</td><td>Back Shoulder</td><td>×</td><td>√</td><td>√</td><td>×</td><td>D</td><td>×</td><td>28 mm</td><td>√</td><td>&gt; 90%</td></tr><tr><td>PalmGesture [64]</td><td>2015</td><td>IR camera+laser</td><td>Wrist</td><td>x</td><td>×</td><td>√</td><td>√</td><td>C</td><td>x</td><td></td><td>×</td><td>&gt; 90%</td></tr><tr><td>PalmType [63]</td><td>2015</td><td>IR sensors</td><td>Wrist</td><td>x</td><td>√</td><td>√</td><td>×</td><td>D</td><td>x</td><td>一</td><td>√</td><td>74.5%</td></tr><tr><td>SkinTrack [78]</td><td>2016</td><td>Electrodes</td><td>Wrist+Finger</td><td>×</td><td>x</td><td>√</td><td>√</td><td>C</td><td>×</td><td>24.1 mm</td><td>V</td><td>92.6%</td></tr><tr><td>TouchCam [58]</td><td>2018</td><td>IR+camera+IMU</td><td>Finger</td><td>×</td><td>×</td><td>√</td><td>×</td><td>D</td><td>×</td><td>一</td><td></td><td>~96%</td></tr><tr><td>EgoTouch [46]</td><td>2024</td><td>RGB camera</td><td>Headset</td><td>√</td><td>√</td><td>x</td><td>×</td><td>C</td><td>×</td><td>一</td><td>√</td><td>95.6%</td></tr><tr><td>Palmpad [16]</td><td>2025</td><td>RGB camera</td><td>Headset</td><td>x</td><td>√</td><td>x</td><td>×</td><td>C</td><td>x</td><td></td><td>√</td><td>97.0%</td></tr><tr><td>PalmSpace (Ours)</td><td>2026</td><td>IR camera</td><td>Wrist</td><td>√</td><td>√</td><td>√</td><td>√</td><td>C</td><td>√</td><td>6.7 mm</td><td>√</td><td>98.9%/96.7%</td></tr></table>

## 2.1 Input Methods for Head-Worn Interaction

Head-worn interaction requires input techniques that are eficient, easy to use, private, socially acceptable, and suitable for eyes-free use in daily activities [73, 75]. At the same time, they must support both diverse commands and suficiently precise control. Gaze-based interaction remains a dominant modality, using AR/MR headsets [34, 66, 74] or smartphone cameras [45, 53] for interaction. However, it sufers from the “Midas Touch” problem [23], ocular fatigue [34], and lower precision than traditional mice [42]. Robust gaze input also often depends on dedicated eye tracking or rich sensing configurations, which are dificult to support in lightweight head-worn settings [34, 66, 74]. Mid-air hand interaction ofers an intuitive alternative, including pointing-direction input [12, 19, 29, 36, 62] and fingertip estimation [43, 54]. Yet, these methods depend on line of sight and provide little haptic feedback, complicating fine motor control and increasing fatigue during prolonged use [6]. Physical devices provide tangible feedback, but come with diferent trade-ofs. Pen-based inputs ofer high precision, but are less portable because they require an additional Manuscript submitted to ACM handheld device and often rely on relative input with mobile devices or external tracking systems [20, 40, 41, 44, 55]. Smart rings improve portability [26, 27, 31, 48, 56], yet often sacrifice absolute positioning and wearing comfort. Smartphones [1, 7, 15, 38, 51, 52, 76] and smartwatches [24] similarly support portable interaction, with diferent capabilities and constraints in sensing and control.

## 2.2 On-Hand Input Methods

Prior research has explored a range of techniques for turning the hand or palm into an input surface. For smart-glasses, however, two challenges remain central: whether palm input can be sensed with a lightweight and wearable setup, and whether the palm can support versatile, structured interaction.

Practical sensing methods. Early systems demonstrated the feasibility of palm interaction by augmenting the environment or the body with additional sensing hardware. Kohli and Whitton [30] used magnetic tracking to monitor the index finger on the opposite palm, but the method was prone to drift and required careful calibration. OmniTouch [14], WatchSense [57], and LumiWatch [72] combined depth sensing and projection to support on-body touch interaction. While these systems showed the promise of interactive skin, they depended on depth cameras, projectors, or bulky mounted hardware, which increase cost, size, and deployment burden [46]. Other wearable sensing approaches reduced some of this hardware overhead. SkinTrack [78] used RF-based triangulation for continuous 2D tracking, but required user calibration and was sensitive to skin conditions. TouchCam [58] achieved high touch recognition accuracy by instrumenting the finger, while HandPad [39] relied on attachments on both wrist and fingers. Although efective, these solutions introduce instrumentation on the fingers or body that reduces practicality. More recently, head-mounted MR systems such as EgoTouch [46] and Palmpad [16] have shown that finger-palm contact can be detected using egocentric cameras. However, such approaches remain inherently viewpoint-dependent: touch inference is unstable under self-occlusion, oblique viewing angles, or when the hand leaves the camera view, as evidenced by Palmpad’s reliance on the Quest 2’s multi-camera API for localization. Taken together, prior work has explored multiple sensing routes, yet practical palm interaction for smartglass use remains constrained by bulky hardware, finger instrumentation, calibration requirements, or viewpoint dependency

From touch events to absolute palm interaction. Beyond sensing, a second challenge is how the palm is represented as an input space. Several systems have leveraged the palm as a proprioceptively accessible surface for eyes-free interaction [4]. PalmRC [9] and PalmType [63], for example, showed that the palm can support remote control and text entry without visual attention. However, PalmType relied on discrete infrared proximity sensors and was limited to detecting predefined key regions rather than continuous coordinates. PalmGesture [64] moved toward continuous input using an infrared camera and laser line projector, but required users to define the interaction area by touching four corners and imposed strict posture constraints on finger orientation. These limitations illustrate that many prior systems still treat the palm either as a set of discrete touch zones or as a calibrated surface requiring explicit setup, rather than as a robust absolute interaction space. For head-worn use, however, practical palm interaction requires more than knowing that a touch occurred; it also requires reliable absolute localization on the palm surface without per-user calibration.

From single-finger input to expressive multi-finger interaction. A further limitation of existing on-skin interfaces lies in interaction expressiveness. Most compact wearable palm-input systems primarily support single-finger tapping or pointing [58, 63, 72, 78]. By contrast, studies have shown that users naturally transfer multi-finger gesture metaphors, such as pinching and two-finger sliding, from smartphones to skin surfaces [68]. Systems that support simultaneous multi-point input, such as iSkin [67] and SkinMarks [69], typically require stretchable overlays or electronic tattoos attached directly to the skin, which reduce tactile sensitivity and complicate everyday deployment. Projection-based Manuscript submitted to ACM

on-skin interaction systems can support multi-point input, but generally require bulky sensing hardware [14]. In the realm of compact, wrist-worn devices (e.g., SkinTrack [78], LumiWatch [72]), signal ambiguity and occlusion have largely restricted interactions to single-finger pointing or discrete tapping. As a result, prior work has largely left compact palm interfaces at the level of sparse input events, without explicitly modeling richer interaction modes in a unified way. In particular, existing systems rarely address together three essential dimensions of palm interaction: whether contact occurs, where it occurs in absolute palm coordinates, and which interaction mode it belongs to, such as single-finger versus multi-finger input.

Summary and Research Gap. Table 1 summarizes how PalmSpace compares with prior on-palm interaction systems in terms of sensing setup, calibration requirements, localization capability, interaction expressiveness, and evaluation scope. Taken together, prior work has improved palm input sensing, but still falls short of making the palm a practical and general interaction space. Existing systems typically address only part of the problem—such as touch detection, localization, or a specific gesture family—while often relying on bulky hardware, per-user calibration, or sustained visual attention. As a result, palm input remains fragmented into isolated events and task-specific pipelines rather than functioning as a unified interaction space. PalmSpace addresses this gap by jointly modeling contact existence, absolute on-palm location, and interaction mode in a lightweight, calibration-free wrist-worn system.

## 3 PalmSpace Overview

## 3.1 Proof-of-Concept Hardware

PalmSpace is a wrist-mounted infrared vision system for sensing bimanual finger-to-palm interaction. It is designed to capture fine-grained palm contact cues and support three tasks: touch detection, touch localization, and gesture recognition. At this stage, we implement PalmSpace as a proof-of-concept prototype to validate sensing feasibility and system design under stable imaging conditions.

Why a wrist viewpoint? A wrist-worn camera provides a direct and stable view of the hand and palm during interaction, making it well suited for contact sensing. Compared with head-mounted viewpoints [46, 59] or external tracking APIs [11], the wrist perspective more directly captures the contact region and is therefore better suited for subtle touch sensing [65, 71]. More importantly, unlike most prior wrist-mounted vision systems, which mainly focus on single-hand pose estimation or finger tracking [18, 28, 37, 71], PalmSpace targets bimanual finger-to-palm interaction, a fundamentally diferent task that requires sensing touch events, touch locations, and interaction gestures rather than only posture.

Prototype design. Our wristband (Fig. 2a) integrates an infrared camera (OV2710) with a 3.5 mm wide-angle lens (100<sup>◦</sup> FOV), capturing 1920×1080 infrared images at 30 FPS. Ten 850 nm infrared LEDs are embedded in a 3D-printed casing to provide active illumination, enabling robust infrared imaging that reduces ambient light interference and reveals clear palm details. To reduce self-occlusion during wrist bending, we introduce a 120<sup>◦</sup> mechanical flip-up structure, similar to the design of imoo Z7 [21], allowing the camera to maintain visibility across a wide range of hand poses. The device is worn on the dorsal wrist in standby mode and rotated to the inner wrist during sensing. The prototype measures 2.4 cm in height and 3.2 cm in width from the side, and weighs 22 g excluding cables. At this stage, PalmSpace is implemented as a wired prototype for reliable image capture and synchronization; future wireless integration is feasible given commercial wearable camera systems [22, 61, 70].

![](images/64c245872148f02e49ba6b5cc4237e78589d584045e910602e478e9a5fedcfc9.jpg)  
Fig. 2. PalmSpace’s hardware prototype overview.

## 3.2 Versatile Interaction Modes

To realize the versatile interaction space introduced earlier, PalmSpace structures the palm as a unified interaction space rather than a collection of isolated touch events. We organize this space along two dimensions: finger configuration (single-finger vs. multi-finger) and contact surface (palm vs. finger surface), enabling both precise input and expressive manipulation. For single-finger input, we define three natural posture variants—Left Yaw (LY), Vertical Touch (VT), and Right Yaw (RY)—based on the finger’s yaw angle relative to the palm (Fig. 3a-3c). These variants capture common on-body touch orientations while allowing pitch to vary freely (Fig. 3d), and can occur on both the palm and finger surfaces. For multi-finger input, PalmSpace supports two palm-based modes, Scroll and Pinch (Fig. 3e and 3d), whose larger and smoother contact area better supports stable multi-finger control. Together, these modes turn the palm into a structured interaction space for symbolic input, precise spatial control, and compound manipulation.

![](images/b4ce8c05b9b6894bd2c6e588c000c376763bab00a30ac02c4081826b975e393b.jpg)  
Fig. 3. Illustration of finger-palm interaction modes. (a)-(c) Single-finger modes: users operating in LY, VT, and RY modes respectively. (d) The pitch angle between the finger and palm surface can be freely adjusted. (e)-(f) Multi-finger gestures: Scroll and Pinch are performed on the palm surface.

## 4 PalmSpace Method

PalmSpace is built on a simple premise: applications should receive palm input through a common representation rather than disconnected touch, localization, and gesture pipelines. PalmSpace therefore unifies input at two levels. At the interaction level, it couples discrete state semantics with continuous absolute location on the palm. At the model level, it learns both outputs through a shared multi-task representation from wrist-worn images. Manuscript submitted to ACM

## 4.1 Unified Touch Modeling Framework

The interaction representation integrates three tightly coupled properties: (1) contact-state classification, which distinguishes non-contact from active touch; (2) interaction-mode recognition, which identifies whether the current interaction corresponds to single-finger input or a supported multi-finger mode; and (3) touch localization, which estimates the absolute interaction position in a palm-normalized coordinate space. Instead of deploying independent task-specific models, PalmSpace learns these properties within one end-to-end model, enabling real-time interaction without per-use sensing calibration. This unified formulation structures the palm as an interaction space with hierarchical interaction awareness. Single-finger input supports precise pointing and target selection, while multi-finger input supports richer manipulations such as scrolling and pinch-based control. In our implementation, the model predicts four interaction states: Non-contact, Single-finger, Scroll, and Pinch. These four states jointly encode both contact presence and interaction mode. At the same time, the model regresses the 2D location of the primary interaction point, thereby coupling discrete interaction semantics with continuous spatial input.

## 4.2 Palm-Normalized Touch Representation

A central challenge of touch localization on the palm is the large anatomical variation across users. Directly regressing fingertip positions in absolute image or physical coordinates makes predictions highly sensitive to hand size diferences. As a result, even the same semantic touch location on the palm (e.g., the tip of the middle finger) can correspond to substantially diferent coordinates across users. In our dataset, the gap between the smallest and largest hands reaches 4.5 cm, causing significant bias for direct coordinate regression. To address this issue, PalmSpace represents touch locations in a palm-relative normalized coordinate system. The horizontal coordinate is centered at the wrist midline and normalized by palm breadth, while the vertical coordinate is referenced to the palm base and normalized by palm length. This maps the palm into a standardized space with lateral range [−0.5, 0.5] and longitudinal range [0, 1], shifting localization from user-specific absolute prediction to proportional position estimation within the palm. This representation improves cross-user consistency, supports calibration-free learning, and enables the learned interaction space to transfer naturally across devices with diferent screen sizes.

## 4.3 Multi-Task Transformer Architecture

To instantiate the unified formulation above, we design a multi-task Transformer that jointly predicts interaction state and touch location from wrist-worn images (Fig. 4). The model contains three components: a visual feature extractor, a lightweight Transformer interaction module, and two task-specific prediction heads.

Given an input image, we first extract visual features using a pre-trained ViT-B/16 backbone [10]. The backbone output is reshaped into a spatial feature map and projected by a 1 × 1 convolution to obtain $F \in \mathbb { R } ^ { d _ { \mathrm { m o d e l } } \times H \times W }$ , where $d _ { \mathrm { m o d e l } } = 2 5 6$ . To preserve spatial information for touch localization, we add absolute 2D sinusoidal positional embeddings $P$ to the projected tokens. The resulting tokens are then processed by a lightweight Transformer module $\mathcal { T }$ with three encoder layers. A learnable class token $t _ { \mathrm { c l s } }$ is prepended to aggregate a representation of the interaction state:

$$
F ^ { \prime } = \mathcal { T } ( F + P ) ,\tag{1}
$$

yielding the output sequence

$$
[ t _ { \mathrm { c l s } } ^ { \prime } , f _ { 1 } ^ { \prime } , f _ { 2 } ^ { \prime } , \dots , f _ { H \times W } ^ { \prime } ] .\tag{2}
$$

Manuscript submitted to ACM

![](images/b7afbbc7d9bec04a56a26f9aac29e0b8b7dc3d44d4c251e0cdfb718aca65d4b9.jpg)  
Fig. 4. PalmSpace network architecture.

We use the transformed class token $t _ { \mathrm { c l s } } ^ { \prime }$ as the shared latent representation and attach two parallel heads. The classification head predicts four interaction states: Non-contact, Single-finger, Scroll, and Pinch. The regression head predicts the normalized 2D coordinates $( x , y )$ of the interaction point.

For single-finger input, the regression target is the active contact location. For multi-finger gestures, PalmSpace regresses the index finger as the primary interaction point. This proxy-based design is suficient for the two gesture types studied here: in Scroll, index-finger displacement captures gesture direction and ofset; in Pinch, the index finger provides a stable reference relative to the palm center for estimating radial and angular change.

## 4.4 Training Objective

PalmSpace is trained end-to-end with a multi-task objective that jointly optimizes interaction-state classification and touch localization:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \lambda _ { \mathrm { c l s } } \mathcal { L } _ { \mathrm { c l s } } + \lambda _ { \mathrm { r e g } } \mathcal { L } _ { \mathrm { r e g } } ,\tag{3}
$$

where $\mathcal { L } _ { \mathrm { c l s } }$ and $\mathcal { L } _ { \mathrm { r e g } }$ denote the classification and regression losses, respectively. For interaction-state prediction, the model classifies each sample into one of four states: Non-contact, Single-finger, Scroll, and Pinch. We use the standard cross-entropy loss:

$$
\mathcal { L } _ { \mathrm { c l s } } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { c = 1 } ^ { K } \mathcal { K } ( y _ { i } = c ) \log ( \hat { p } _ { i , c } ) ,\tag{4}
$$

where � is the batch size, $K = 4$ is the number of interaction states, �<sub>�</sub> is the ground-truth label, and $\hat { p } _ { i , c }$ is the predicted probability for class �. For touch localization, we regress the interaction point using a masked mean squared error:

$$
\mathcal { L } _ { \mathrm { r e g } } = \frac { 1 } { \sum _ { i = 1 } ^ { N } M _ { i } } \sum _ { i = 1 } ^ { N } M _ { i } \cdot \Vert \tilde { C } _ { i } - C _ { i } \Vert _ { 2 } ^ { 2 } ,\tag{5}
$$

where ${ \tilde { C } } _ { i }$ and $C _ { i }$ are the predicted and ground-truth coordinates, and $M _ { i } = \Psi ( y _ { i } \neq$ Non-contact) masks out non-contact samples, which have no valid spatial target. We set $\lambda _ { \mathrm { c l s } } = 1$ and $\lambda _ { \mathrm { r e g } } = 1 0 0$ to account for the smaller numerical scale of normalized coordinate regression and to maintain efective joint optimization. Additional implementation details are provided in Appendix A.

## 5 Ofline Evaluation

This section evaluates PalmSpace for joint touch mode recognition and touch localization, covering the dataset and validation protocol, detailed results across contact regions and interaction types, and ablation studies of key design components.

Manuscript submitted to ACM

## 5.1 Data Collection

5.1.1 Pipeline. To evaluate PalmSpace, we collected a comprehensive dataset of finger-to-palm interactions covering the variations shown in Fig. 3. Each sample was annotated with four labels: (1) touch state (touch vs. non-touch), (2) gesture category (Single, Scroll, or Pinch), (3) interaction coordinates on the hand surface, and (4) palm size for coordinate normalization. Our setup used a wrist-mounted IR camera for model input and an overhead mirrorless camera for ground-truth generation. To recover physical interaction coordinates, we defined a planar calibration area around the hand with AprilTag markers [49] and mapped the overhead view to the physical coordinate system via homography estimation. Fingertip keypoints were tracked in the overhead view using MediaPipe [77] and transformed into metric palm-plane coordinates as localization ground truth. This markerless design avoids attaching sensors or reflective markers, reducing visual artifacts while preserving realistic bare-hand interaction. Details are provided in Appendix B.

5.1.2 Participants and Procedure. We recruited 17 right-handed volunteers (11 male, 6 female; aged 20–34 years, $M = 2 4 . 4 , S D = 3 . 3 )$ . To ensure anthropometric diversity, we recorded hand dimensions: hand length ranged from 155 to 200 mm $( M = 1 8 1 . 1 , S D = 1 2 . 0 )$ , and hand breadth ranged from 65 to 85 mm $( M = 7 7 . 8 , S D = 5 . 2 )$ , consistent with prior reports [25]. The study included 9 sessions: 6 single-finger sessions (2 areas × 3 modes), 2 multi-finger sessions (Scroll and Pinch), and 1 non-contact session. Single-finger data were collected on both the palm and finger regions, whereas multi-finger gestures were collected on the palm only. Participants performed gestures naturally without strict pose constraints to better reflect realistic touchpad-like interactions. The non-contact session included hovering and random hand motions as negative samples for touch detection. After removing samples with ground-truth tracking failures, the final dataset contained 63,903 images from 17 participants, including 35,727 single-finger samples, 11,645 multi-finger samples, and 16,531 non-contact samples. Each participant completed the procedure in about 30 minutes.

## 5.2 Validation Setup

We adopt a leave-one-participant-out (LOPO) cross-validation strategy. Given � = 17 participants, the dataset is partitioned into � folds. In each fold, data from � − 1 users are used for training, while the remaining user serves as the unseen test set. This subject-independent protocol rigorously evaluates the model’s generalization ability to new users.

We evaluate model performance from three perspectives:

• Contact Region: performance on single-touch detection and localization over the Palm and Fingers.

• Single-finger Touch Mode: performance under the three finger orientation patterns (LY, VT, and RY) defined in Sec. 3.2.

• Multi-finger Gesture: performance on dynamic multi-finger interactions, specifically Scroll and Pinch.

For touch gesture recognition, we report the F1-score. For absolute localization, we use the Mean Absolute Error (MAE) and Standard Deviation (SD) on the �/� axes $( x _ { \mathrm { e r r o r } } , y _ { \mathrm { e r r o r } } )$ , as well as the Euclidean distance error $( l _ { \mathrm { e r r o r } } )$ Localization is evaluated only on samples with correctly detected valid touch events (Single, Scroll, or Pinch). And the metrics are micro-averaged from all test folds.

## 5.3 Performance

Under LOPO cross-user evaluation, PalmSpace achieves stable joint performance in touch-state/gesture recognition and absolute localization, reaching 98.9% binary touch detection accuracy, 96.7% overall touch gesture F1, and 6.7 mm MAE

![](images/9b6647477336249247c8a98080e5076c99af10a686c0b39ccdd8630fe1c1fcb3.jpg)  
(a) Prediction errors in the �–� plane.

![](images/9dd46f9ef1dee1ab59e2190a95e684843100f4d9e3524eb6cc32bfd5d79a6c7d.jpg)  
(b) Distribution of Euclidean localization error.

![](images/27f6390ade22680dfe9d1a41b4f3a68af0c2244815dbe2f409654feca6eadcaf.jpg)  
(c) Single-finger MAE by touch mode and region.

![](images/6155de44659a4a8527bf1dda20ad682ddbf7d29c92fe194e4d3adb13a9a092dd.jpg)  
(d) Efect of palm-size normalization.  
Fig. 5. Error analysis and ablation results. Error bars in (c) denote SE; (d) reports the three users with the smallest palms.

Gesture recognition. As summarized in Table 2, PalmSpace achieves 98.9% binary touch detection accuracy. For multi-class gesture recognition, Non-Touch, Single-Touch, Scroll, and Pinch obtain F1 scores of 97.9%, 97.2%, 88.1%, and 97.4%, respectively, yielding an overall F1 of 96.7%. Among these classes, Scroll is the most challenging due to its similarity to single-finger sliding, yet PalmSpace still achieves an F1 score of 88.1%. Localization accuracy. PalmSpace achieves an overall MAE of 6.7 mm over all valid touch samples (Table 2). Localization remains accurate for both Single-Touch (6.1 mm) and Scroll (6.3 mm), suggesting that the model handles planar finger motion robustly. Even for Pinch, which involves stronger self-occlusion between the thumb and index finger, the error remains within 10.6 mm under cross-user evaluation.

Error patterns. For single-finger touch, localization accuracy varies by interaction region. As shown in Fig. 5c, the Palm Area is easier to localize than the Finger Area (5.4 mm vs. 7.2 mm MAE). This gap is mainly attributable to larger errors along the �-axis in the Finger Area, likely because points farther from the wrist-mounted camera sufer more severe perspective compression. Across touch modes, VT produces the largest error, especially in the Finger Area, whereas LY and RY are more stable.

Fig. 5a and 5b further shows that the prediction errors are tightly concentrated around zero and that most Euclidean errors fall within 1 cm, indicating good cross-user robustness in practical interaction scenarios. Axis-wise errors are reported in Appendix Table 4.

Table 2. Performance overview across diferent interaction modalities. ‘Single’ refers to single-finger touch.
<table><tr><td rowspan="2">Metric</td><td rowspan="2">Non-Touch</td><td colspan="3">Touch Gestures</td><td rowspan="2">Overall</td></tr><tr><td>Single</td><td>Scroll</td><td>Pinch</td></tr><tr><td>F1 Score (%)</td><td>97.9</td><td>97.2</td><td>88.1</td><td>97.4</td><td>96.7</td></tr><tr><td>MAE (mm)</td><td>一</td><td>6.1</td><td>6.3</td><td>10.6</td><td>6.7</td></tr><tr><td>SD (mm)</td><td>一</td><td>4.3</td><td>3.9</td><td>11.4</td><td>5.7</td></tr></table>

<sup>\*</sup> The binary Touch Detection Accuracy (Touch vs. Non-Touch) is 98.9%.

## 5.4 Ablation Study

We assess PalmSpace through indoor LOPO component ablations (Table 3(a)) and zero-shot evaluation on the unseen daytime data from our Outdoor Robustness Validation (Sec. 6.4; Table 3(b)); its data-collection and training protocol appears in Appendix E. Unless marked ‘FT’, outdoor models are trained only on indoor data without outdoor fine-tuning. Manuscript submitted to ACM

In-domain component analysis. The full model achieves the best overall indoor performance, with the lowest localization error (6.7 mm MAE) and the highest gesture recognition score (96.7% F1).

1) Efect of palm-size normalization. Removing palm-size normalization increases the overall MAE to 6.9 mm, with a drop in multi-finger interactions (8.3 mm to 8.8 mm), indicating improved cross-user generalization, especially for smaller palms. This trend is further reflected in the three female participants with the smallest palms in the validation set (Fig. 5d), whose errors are consistently higher without normalization; for User P2, the MAE nearly doubles.

2) Efect of data augmentation. Removing data augmentation causes the largest drop in gesture recognition, reducing F1 from 96.7% to 95.3%, while increasing the overall MAE to 7.0 mm, indicating that augmentations simulating camera tilt, camera–palm ofset, and illumination variation are important for robustness to pose variation and light changes.

Out-of-domain architecture and learning analysis. Compared with a ResNet-50 encoder, a model without the decoder, and a U-Net baseline, the full PalmSpace architecture achieves the best zero-shot outdoor performance (Table 3(b)). We further trained two independent copies of the same architecture, one for interaction-state classification and one for touch localization. Compared with unified learning, independent training increases localization MAE from 9.6 mm to 11.7 mm and reduces touch accuracy from 93.7% to 90.5%, while requiring two deployed model instances. These results suggest that the shared representation improves generalization under distribution shift while reducing runtime model duplication. Outdoor fine-tuning further recovers performance to 6.8 mm MAE and 98.7% touch F1.

Table 3. PalmSpace ablations under (a) in-domain indoor LOPO evaluation and (b) out-of-domain daytime outdoor evaluation. ‘Independent’ uses separately trained interaction-state and localization models; ‘FT’ denotes outdoor fine-tuning.  
(a) In-domain component ablation
<table><tr><td rowspan="2">Method</td><td colspan="3">Positioning MAE (↓)</td><td rowspan="2">Gesture F1 (↑)</td></tr><tr><td>Single-finger</td><td>Multi-finger</td><td>Overall</td></tr><tr><td>w/o Normalization</td><td>6.3</td><td>8.8</td><td>6.9</td><td>96.1</td></tr><tr><td>w/o Data Augmentation</td><td>6.4</td><td>8.7</td><td>7.0</td><td>95.3</td></tr><tr><td>w/ ResNet50 Encoder</td><td>6.3</td><td>8.3</td><td>6.8</td><td>96.3</td></tr><tr><td>w/o Decoder</td><td>6.3</td><td>8.8</td><td>6.9</td><td>96.2</td></tr><tr><td>U-Net</td><td>7.0</td><td>8.5</td><td>7.4</td><td>95.8</td></tr><tr><td>PalmSpace (Full Model)</td><td>6.1</td><td>8.3</td><td>6.7</td><td>96.7</td></tr></table>

(b) Out-of-domain architecture and learning ablation
<table><tr><td rowspan="2">Method</td><td colspan="2">Positioning</td><td colspan="2">Touch</td></tr><tr><td>MAE (↓)</td><td>SD (↓)</td><td>ACC (%) (↑)</td><td>F1 (%) (↑)</td></tr><tr><td>w/ ResNet50 Encoder</td><td>14.0</td><td>6.8</td><td>90.6</td><td>92.1</td></tr><tr><td>w/o Decoder</td><td>11.9</td><td>6.1</td><td>72.5</td><td>73.1</td></tr><tr><td>U-Net</td><td>17.3</td><td>8.5</td><td>89.8</td><td>91.6</td></tr><tr><td>Independent task-specific models</td><td>11.7</td><td>7.2</td><td>90.5</td><td>93.0</td></tr><tr><td>PalmSpace (Full Model)</td><td>9.6</td><td>5.2</td><td>93.7</td><td>95.7</td></tr><tr><td>PalmSpace-FT</td><td>6.8</td><td>3.9</td><td>98.0</td><td>98.7</td></tr></table>

![](images/c423b4b8c9c97d4fc40095d493a992b9a016e5294ed6a0769afc71702e2336cc.jpg)  
(a) Fits’ law study GUI and PalmSpace numeric keypad layout.

![](images/646e11039923d8f554ab0ac4c99b6c351c6f96b6c79458d065ebe5b72d6a54a9.jpg)  
(b) Selection time in the Fits’ law study.

Fitts' Law Experiment: PalmSpace(PT) and Touchscreen(TS)  
![](images/61f8a20e1a3029e1b18bad5355fa58fce445d5d3971e083b5c26f076cb52b8f1.jpg)  
(c) Mean movement time versus index of dificulty.

![](images/c28c785258d63d2d9b528e7daab387c75abce8b8c41b84dc70ab327a0b42ae0d.jpg)  
(d) Eyes-free digit-entry accuracy.  
Fig. 6. Setup and results for the indoor single-finger user studies.

## 6 User Study of Single-Finger Interaction

This section presents an evaluation of PalmSpace for single-finger interaction across both indoor and outdoor settings. We first conducted two indoor studies to examine input eficiency under realistic everyday lighting conditions. The first focused on pointing and dragging performance using a Fitts’ law task, while the second examined eyes-free digit entry performance, with particular attention to the benefits of absolute positioning and proprioception. We then assessed the robustness of PalmSpace in representative outdoor conditions, including daytime use under uncontrolled lighting and nighttime use in extremely low-light environments.

## 6.1 Input Methods and Participants

PalmSpace was compared with a smartphone touchscreen baseline, as both support absolute positioning over a similar size of interaction area. We recruited 12 right-handed participants (3 female; age 19–24, � = 21.2), all familiar with touchscreen use and none with prior experience using PalmSpace. To reflect realistic usage, the indoor study was conducted in 4 indoor settings with diferent ambient lighting conditions: a morning ofice, a night meeting room, an evening lab, and an afternoon apartment. Participants used their dominant index finger for input. For PalmSpace, they used the non-dominant hand without individual calibration; for the touchscreen baseline, they held the smartphone in the non-dominant hand.

Manuscript submitted to ACM

## 6.2 Study 1: Fits’ Law Study

6.2.1 Task and Procedure. We conducted a Fitts’ law study to evaluate pointing and dragging performance. In both conditions, interaction was performed with identical visual feedback presented on a separate display (Fig. 6a). To ensure a fair comparison, the smartphone screen did not display any graphical interface and served only as a coordinate-sensing surface. Thus, both conditions used the same interaction mechanism and external visual feedback. For both methods, normalized input coordinates were mapped to the same GUI, ensuring identical control-display ratios and perceived target sizes across conditions

In each trial, the GUI displayed a yellow target and a green target area. The yellow target had a randomly sampled diameter of 20–40 mm, while the green target area had a fixed diameter of 60 mm. The two circles were separated by a random distance of 80–200 mm. Participants first moved the cursor to the yellow target. Once selected, the target turned red, indicating the start of the dragging phase. Participants then dragged the red circle until it was fully enclosed within the green target area. We recorded hit time, defined as the time from target onset to target acquisition, and drag time, defined as the time from the start of dragging to successful placement. Each participant completed 5 11-trial blocks per method; excluding the first trial of each block as a warm-up yielded 50 valid trials per method. The same sequence of target parameters was used across conditions to control task dificulty. Participants practiced until performance stabilized, and short breaks were provided between blocks

6.2.2 Result. We analyzed 1,200 valid method-level trials in total (600 per method). To verify consistency with Fitts’ law, we modeled the relationship between movement time and index of dificulty using $\begin{array} { r } { I D = \log _ { 2 } \bigl ( \frac { 2 D } { W } \bigr ) } \end{array}$ . As shown in Fig. 6b and 6c, PalmSpace achieved performance close to the touchscreen baseline. For total, hit, and drag times, PalmSpace yielded 1572.2 ms, 797.7 ms, and 774.4 ms, respectively, while the touchscreen yielded 1361.6 ms, 642.5 ms, and 719.2 ms. Notably, drag time showed no significant diference between PalmSpace and the touchscreen $( F _ { 1 , 1 1 } = 1 . 3 0 , p = 0 . 2 7 9 )$ This is particularly surprising because, despite the palm being an uneven and compliant surface, PalmSpace still enabled manipulation performance comparable to a gold-standard touchscreen.

Further analysis of hit time versus ID (Fig. 6c) showed that the main diference arose during target acquisition, likely due to the palm’s less stable surface. However, once the target was acquired, dragging performance remained stable and comparable across methods. Overall, these results demonstrate that PalmSpace supports eficient pointing and dragging, with performance approaching that of a touchscreen even on the naturally irregular surface of the palm.

## 6.3 Study 2: Digit Input Evaluation

6.3.1 Task and Procedure. We evaluated PalmSpace’s support for eyes-free absolute input using a digit entry task. Because discrete digit entry requires direct selection of fixed spatial locations, it serves as a natural test of absolute pointing on the palm.

As shown in Fig. 6a, we mapped twelve finger knuckles (excluding the thumb) to a keypad-like layout, with the three knuckles of the little finger corresponding to digit 0. Since the perceived boundaries and absolute positions of knuckle regions can vary slightly across individuals, participants first touched the designated knuckle areas so that their personalized finger layout could be recorded. This registered the application-specific mapping between anatomical regions and digit labels; it did not retrain or calibrate the PalmSpace sensing model. For the touchscreen baseline, we used a keypad layout with the same size and spatial arrangement as PalmSpace. Participants then completed five rounds of random 10-digit entry under eyes-free conditions. The touchscreen baseline followed the same procedure and used the same digit sequences.

6.3.2 Result. As shown in Fig. 6d, PalmSpace significantly outperformed the touchscreen, achieving an overall digitentry accuracy of 93.71% versus 74.86% $( W = 2 . 5 , p < . 0 1 )$ . We further analyzed transitions between consecutive inputs as Identical, Adjacent, and Non-adjacent, referring to repeated, neighboring, and spatially separated keys, respectively. PalmSpace remained stable across all three transition types (all above 94%), whereas touchscreen performance declined substantially as transition distance increased. These results show that PalmSpace efectively supports eyes-free absolute input by leveraging the palm’s intrinsic proprioceptive structure. Detailed digit-level results are provided in Appendix C.

## 6.4 Outdoor Robustness Validation

To assess robustness beyond indoor settings, we evaluated PalmSpace in two representative outdoor scenarios: daytime use under uncontrolled lighting and nighttime use in extreme low-light conditions. We recruited 13 new volunteers with no prior experience using the system: 7 performed random palm touches in diverse daytime settings, including parks, streets, and direct sunlight, and 6 performed the same Fitts’ law task as in Sec. 6.2 at night under ambient illuminance below 5 lux. The daytime model results are reported with the other ablations in Table 3(b); detailed data-collection and training protocols are provided in Appendix E.

In daytime testing, the original indoor-trained model achieved a mean absolute error of 9.6 mm; fine-tuning reduced this to 6.8 mm and improved touch detection F1 to 98.7%, indicating robust accuracy under challenging lighting and background conditions. At night, geometric error could not be measured because the external RGB camera used for ground-truth annotation was inefective in near-dark environments. We therefore evaluated usability using the same indoor Fitts’ law task and task completion time as the primary metric. The average nighttime completion time was 1509.03 ms (SE = 25.49), comparable to the 1572.2 ms observed indoors, suggesting that PalmSpace remains practically usable in near-dark real-world conditions. Overall, these results show that PalmSpace supports eficient single-finger interaction both indoors and across representative outdoor conditions

## 7 Evaluation of Multi-Finger Gestures

To evaluate PalmSpace in lightweight smart-glasses-oriented scenarios, we conducted a task-based user study on two representative operations in head-worn interfaces: List Navigation and Map Manipulation. These tasks represent lightweight UI browsing and continuous spatial adjustment, respectively. Accordingly, we evaluated two palm-based multi-finger interactions: Scroll for discrete directional navigation and Pinch for continuous zooming and rotation.

## 7.1 Interaction Logic and State Management

The interaction logic is governed by a finite state machine (FSM), shown in Fig. 7a. Based on the model output, the system maintains four states: Idle, Single-Touch, Scroll Mode, and Pinch Mode. The system enters the corresponding interaction mode after classification and returns to Idle upon release. In Scroll Mode, directional intent is estimated from the displacement vector between the start and end frames of a gesture sequence. The displacement angle is mapped to one of four directions (up, down, left, right), while the displacement magnitude suppresses small unintended motions. This design targets lightweight smart-glasses browsing tasks such as list and menu navigation. In Pinch Mode, the predicted interaction point relative to the palm center is mapped to two continuous control channels: radial displacement for zooming and angular displacement for rotation. This enables compact two-degree-of-freedom control for smart-glasses map manipulation. To keep the main paper concise, the exact transfer functions, gain settings, and angle wrapping compensation are provided in the Appendix. Here, we focus on whether this design provides suficient controllability, stability, and usability for application-oriented tasks.

![](images/5c60786a1fd6f86ddfecb53a97d2abe48e31f31e631973450f02895859c5cb61.jpg)  
(a) Finite state machine of PalmSpace.

![](images/dc13cd80c08d7c3550588906a679533b3ce3e84d71f2a83d5bbb926649cd6572.jpg)  
(b) Confusion matrix for the list-navigation task.

Fig. 7. Interaction logic and recognition results for multi-finger input.  
![](images/53e2f597b259f4b5c716b7e2e6313af01c13f0c74734d728fb8aa42b52380ffb.jpg)  
Fig. 8. Applications in the PalmSpace design space: (a) T9 keyboard, (b) handwriting, (c) AI-assisted drawing, (d–e) MR controllers, (f ) content editing, (g) map manipulation, and (h) page control.

## 7.2 Experimental Setup

We recruited 12 participants (age: $M = 2 5 . 0 , S D = 1 . 8 7 ;$ balanced gender), all with normal or corrected-to-normal vision.   
Before the formal study, each participant completed a 5-minute tutorial on gesture triggering and task requirements.   
Formal trials began only after participants reported being comfortable with the gestures.

7.2.1 Apparatus and Tasks. The study comprised two tasks corresponding to the smart-glasses scenarios above. Since current lightweight smart glasses do not support screen projection for customized interfaces, we used an external display and adjusted the viewing distance to approximate smart-glasses display conditions.

Task 1: List Navigation. This task assessed discrete directional control with Scroll gestures. In each trial, participants performed the gesture corresponding to one of four randomized visual prompts (up, down, left, right). Each direction was repeated 5 times, yielding 20 trials per user. This task simulates list and menu navigation in smart-glasses interfaces.

Task 2: Map Manipulation. This task assessed continuous two-degree-of-freedom control with Pinch gestures. We used an abstract compound docking task to capture the core demands of smart-glasses map interaction while reducing confounds from rendering latency and scene complexity. The interface showed a semi-transparent Target State and a solid User Cursor, both rendered as a “Circle + Arrow” glyph. Target scale was randomly sampled from 0.5× to 3.0×, and target rotation from 0<sup>◦</sup> to 360<sup>◦</sup>. Each participant completed 20 trials, yielding 240 trials in total. Participants aligned the cursor to the target using Pinch gestures. A trial was successful only if the cursor stayed within a 10% scale margin and a 10<sup>◦</sup> angular margin for 1.0 s. Scale and rotation were controlled through the same Pinch interaction. We recorded Completion Time, Final Scale Error, Final Rotation Error, and Overshoots, where overshoots denote the number of times the controlled value exceeded the target and then reversed for correction during a trial.

## 7.3 Results

For Task 1 (List Navigation), the system achieved an overall recognition accuracy of 97.08% across 240 trials. As shown in Fig. 7b, confusion was minimal and evenly distributed, suggesting that Scroll provides reliable directional input for list browsing tasks.

For Task 2 (Map Manipulation), participants completed the 240 compound-docking trials in an average of 7.90 s. Rotation stabilized faster than scaling $( T _ { r o t } = 3 . 5 9 \ s \ v s . \ T _ { s c a l e } = 4 . 3 1 \ s$ ), suggesting faster target acquisition in the rotational channel. However, rotation also showed a higher overshoot rate than scaling (0.96 vs. 0.55), indicating more frequent corrective adjustments. In contrast, scaling was slower but more stable. Despite this trade-of, both channels supported precise final control: the final scale error was 4.4%, and the final rotation error was $6 . 1 ^ { \circ }$ , both within the task tolerance. Overall, these results suggest that PalmSpace supports both reliable discrete input and efective continuous zoom-and-rotate control, making it suitable for representative smart-glasses interactions such as list browsing and map manipulation.

## 8 Discussion

## 8.1 Usability Feedback

We assessed PalmSpace’s perceived usability using the System Usability Scale (SUS) [5]. All participants involved in the user study completed the SUS. Using a 6-point scale, PalmSpace received a mean SUS score of $7 2 . 5 \pm 1 4 . 0$ (median = $7 0 . 0 , \mathrm { I Q R } = 1 6 . 0 )$ , corresponding to a “Good” adjective rating on the interpretive scale proposed by Bangor et al. [3]. Item-level responses suggested strong perceived ease of use, learnability, and confidence, with generally low perceived complexity and awkwardness.

Interview feedback further contextualized these results. One participant (female, 25) noted, “PalmSpace remained comfortable even during prolonged wear and did not interfere with other ongoing activities. I did not need to pay much attention to my hand, and could stay focused on the screen, which made the experience feel relaxed.” Another participant (male, 22) remarked, “The gestures were easy to learn and felt like an almost seamless transition from touchscreen interaction, even though the palm is less flat than touchscreen surface.” Notably, one outdoor-study participant (male, 23) was able to complete the study while mountaineering and camping in snowy conditions, suggesting promising robustness beyond controlled indoor environments. The results suggest that PalmSpace provides a usable and learnable interaction experience.

## 8.2 Design Space

PalmSpace enables a versatile design space for on-palm interaction, moving beyond isolated touch events toward a more structured interaction paradigm. This design space is defined by the three dimensions of our unified touch modeling framework: contact state, absolute spatial localization, and interaction modality. Together, these dimensions allow designers to construct a rich vocabulary of eyes-free, calibration-free interactions. In doing so, PalmSpace expands the Manuscript submitted to ACM

dimensionality and functional scope of prior on-palm input techniques, making the palm an expressive interactive surface.

Dimension 1: Contact state. It distinguishes spatial exploration from command execution. Non-contact supports hovering and exploration, while contact marks active input. It also supports diferent temporal forms of interaction, including brief taps and sustained dragging or drawing.

Dimension 2: Absolute spatial localization. It enables precise selection of coordinates on the palm, supporting both discrete target selection and continuous movement across the palm surface. It can also degrade into relative control, similar to a touchpad.

Dimension 3: Interaction modality recognition. It distinguishes between single-finger and multi-finger interaction modes, substantially increasing expressive power and enabling more complex compound operations.

We illustrate this design space through three levels of interaction. First, combining active contact (Dim. 1), absolute localization (Dim. 2), and single-finger tracking (Dim. 3) supports reliable eyes-free text entry (Sec. 6.3, Fig. 8a), handwriting (Appendix C.3, Fig. 8b), drawing or editing (Fig. 8c and Fig. 8f), Whac-A-Mole interaction (Fig. 8d), and attribute control such as volume sliders (Fig. 8e). These examples show how PalmSpace supports symbolic input and precise spatial control through single-finger interaction. Second, combining active contact (Dim. 1), displacement-based mapping (Dim. 2), and multi-finger tracking (Dim. 3) supports the two representative operations studied in Sec. 7: discrete multi-finger gesture input and continuous compound gesture control, such as zooming, rotation, and sliding. These examples demonstrate PalmSpace’s support for manipulation through multi-finger interaction. Third, PalmSpace also enables cross-modal composition, where users combine or transition between single/multi-finger within the same task. Such composition supports more complex operations, as illustrated by map interaction (Fig. 8g) and page browsing and control (Fig. 8h). Precise single-finger positioning can be combined with subsequent multi-finger manipulation to accomplish richer workflows. Please refer to our supplementary video for all examples

## 8.3 Limitations and Future Work

Beyond current findings, several natural directions can further extend the system. While the leave-one-participant-out evaluation indicates encouraging robustness across users, the present dataset still captures only part of the variation encountered in real-world hand morphology, appearance, and usage conditions. Expanding the participant pool and systematically examining factors such as accessories, fingernail appearance, and a wider range of finger-specific behaviors would help further characterize performance across diverse users and contexts. In addition, the current multi-finger interaction model adopts a robust proxy-based design centered on the primary interaction point. This formulation is well suited to the scrolling and pinch-based tasks studied here, and supports stable, practical control on the compact palm surface. Building on this foundation, future work can explore richer representations of finger articulation and denser hand-surface modeling to support more fine-grained multi-finger interaction in a wider range of tasks.

The current system is implemented as a tethered prototype, a design choice that prioritizes sensing stability and synchronized evaluation in this initial study. This setup enables a controlled validation of the unified touch modeling framework and demonstrates the feasibility of versatile on-palm interaction. Looking ahead, an important next step is to move toward a more compact and self-contained wearable implementation. Integrating lightweight cameras into wireless wearables is already a mature direction, as demonstrated by commercial devices such as Pupil Labs Neon [61], TapXR [22], and Wristcam [70]. In a preliminary mobile deployment test, dynamic INT8 quantization and TorchScript reduced model size by 3.7× and accelerated inference by 2.7×, reaching nearly 30 FPS on a Redmi Turbo 4 smartphone. Manuscript submitted to ACM

Future work will therefore focus on on-device deployment through model compression and system-level optimization, including quantization, pruning, and adaptation to lower-power embedded platforms, while maintaining the interaction quality achieved by the current prototype.

## 9 Conclusion

PalmSpace turns the bare palm into a mode-aware, body-referenced interaction surface by jointly representing touch contact, absolute on-palm location, and interaction mode. Its unified model supports cross-user inference without per-user sensing calibration while avoiding separate state-recognition and localization pipelines. Cross-user evaluation and user studies demonstrate absolute pointing, eyes-free digit input, and representative multi-finger controls such as scrolling and map manipulation, highlighting the potential of unified bare-palm input for future head-worn interaction.

## References

[1] Teo Babic, Harald Reiterer, and Michael Haller. 2018. Pocket6: A 6dof controller based on a simple smartphone application. In Proceedings of the 2018 ACM Symposium on Spatial User Interaction. 2–10.

[2] Baidu Inc. 2025. PaddleOCR: Awesome multilingual OCR toolkits based on PaddlePaddle. https://github.com/PaddlePaddle/PaddleOCR. Accessed: 2025-09-01.

[3] Aaron Bangor, Philip T Kortum, and James T Miller. 2008. An empirical evaluation of the system usability scale. Intl. Journal ofHuman–Computer Interaction 24, 6 (2008), 574–594.

[4] Joanna Bergström and Kasper Hornbæk. 2019. Human–Computer interaction on the skin. ACM Computing Surveys (CSUR) 52, 4 (2019), 1–14.

[5] John Brooke et al. 1996. SUS-A quick and dirty usability scale. Usability evaluation in industry 189, 194 (1996), 4–7.

[6] Yi Fei Cheng, Tifany Luong, Andreas Rene Fender, Paul Streli, and Christian Holz. 2022. ComforTable user interfaces: Surfaces reduce input error, time, and exertion for tabletop and mid-air user interfaces. In 2022 IEEE International Symposium on Mixed and Augmented Reality (ISMAR). IEEE, 150–159.

[7] Henrique Debarba, Luciana Nedel, and Anderson Maciel. 2012. Lop-cursor: Fast and precise interaction with tiled displays using one hand and levels of precision. In 2012 IEEE Symposium on 3D User Interfaces (3DUI). IEEE, 125–132.

[8] Nathan Devrio and Chris Harrison. 2022. Discoband: Multiview depth-sensing smartwatch strap for hand, body and environment tracking. In Proceedings ofthe 35th Annual ACM Symposium on User Interface Software and Technology. 1–13.

[9] Niloofar Dezfuli, Mohammadreza Khalilbeigi, Jochen Huber, Florian Müller, and Max Mühlhäuser. 2012. PalmRC: imaginary palm-based remote control for eyes-free television interaction. In Proceedings ofthe 10th European conference on Interactive tv and video. 27–34.

[10] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. 2020. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020).

[11] Camille Dupré, Caroline Appert, Stéphanie Rey, Houssem Saidi, and Emmanuel Pietriga. 2024. TriPad: Touch Input in AR on Ordinary Surfaces with Hand Tracking Only. In Proceedings ofthe 2024 CHI Conference on Human Factors in Computing Systems. 1–18

[12] Cathy Mengying Fang, Patrick Chwalek, Quincy Kuang, and Pattie Maes. 2024. WatchThis: A Wearable Point-and-Ask Interface powered by Vision-Language Models for Contextual Queries. In Adjunct Proceedings ofthe 37th Annual ACM Symposium on User Interface Software and Technology 1–4.

[13] Google Inc. and contributors. 2025. Tesseract OCR. https://github.com/tesseract-ocr/tesseract. Accessed: 2025-09-01.

[14] Chris Harrison, Hrvoje Benko, and Andrew D Wilson. 2011. OmniTouch: wearable multitouch interaction everywhere. In Proceedings ofthe 24th annual ACM symposium on User interface software and technology. 441–450.

[15] Ke He, Chentao Li, Yongjie Duan, Jianjiang Feng, and Jie Zhou. 2023. TrackPose: Towards Stable and User Adaptive Finger Pose Estimation on Capacitive Touchscreens. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies 7, 4, Article 161 (Dec. 2023), 22 pages doi:10.1145/3631459

[16] Zhe He, Xiangyang Wang, Yuanchun Shi, Chi Hsia, Chen Liang, and Chun Yu. 2025. Palmpad: Enabling Real-Time Index-to-Palm Touch Interaction with a Single RGB Camera. In Proceedings ofthe 2025 CHI Conference on Human Factors in Computing Systems. 1–16

[17] Zhe He, Zixuan Wang, Chun Yu, Chengwen Zhang, Xiyuan Shen, and Yuanchun Shi. 2025. WritingRing: Enabling Natural Handwriting Input with a Single IMU Ring. In Proceedings ofthe 2025 CHI Conference on Human Factors in Computing Systems. 1–15

[18] Fang Hu, Peng He, Songlin Xu, Yin Li, and Cheng Zhang. 2020. FingerTrak: Continuous 3D hand pose tracking by deep learning hand silhouettes captured by miniature thermal cameras on wrist. Proceedings of the ACM on interactive, mobile, wearable and ubiquitous technologies 4, 2 (2020), 1–24.

[19] Kaoning Hu, Shaun Canavan, and Lijun Yin. 2010. Hand pointing estimation for human computer interaction based on two orthogonal-views. In 2010 20th International Conference on Pattern Recognition. IEEE, 3760–3763

[20] Sungjae Hwang, Andrea Bianchi, Myungwook Ahn, and Kwangyun Wohn. 2013. MagPen: magnetically driven pen interactions on and around conventional smartphones. In Proceedings ofthe 15th international conference on Human-computer interaction with mobile devices and services. 412–415.

[21] imoo. 2024. imoo Watch Phone Z7. https://www.imoo.com/en/products/z7. Accessed: 2025-09-01.

[22] Tap Systems Inc. 2023. Introducing TapXR. https://www.tapwithus.com

[23] Robert JK Jacob. 1991. The use of eye movements in human-computer interaction techniques: what you look at is what you get. ACM Transactions on Information Systems (TOIS) 9, 2 (1991), 152–169.

[24] Keiko Katsuragawa, Krzysztof Pietroszek, James R Wallace, and Edward Lank. 2016. Watchpoint: Freehand pointing with a smartwatch in a ubiquitous display environment. In Proceedings ofthe International Working Conference on Advanced Visual Interfaces. 128–135.

[25] Muneer Ahmad Khan. 2017. Determination of stature from measurements of hand length and hand breadth; an anthropometric study of Kashmiri population. International Journal of Anatomy and Research (2017).

[26] Wolf Kienzle and Ken Hinckley. 2014. LightRing: always-available 2D input on any surface. In Proceedings of the 27th annual ACM symposium on User interface software and technology. 157–160

[27] Wolf Kienzle, Eric Whitmire, Chris Rittaler, and Hrvoje Benko. 2021. Electroring: Subtle pinch and touch detection with a ring. In Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems. 1–12.

[28] David Kim, Otmar Hilliges, Shahram Izadi, Alex D Butler, Jiawen Chen, Iason Oikonomidis, and Patrick Olivier. 2012. Digits: freehand 3D interactions anywhere using a wrist-worn gloveless sensor. In Proceedings ofthe 25th annual ACM symposium on User interface software and technology. 167–176.

[29] Daehwa Kim, Vimal Mollyn, and Chris Harrison. 2023. WorldPoint: Finger Pointing as a Rapid and Natural Trigger for In-the-Wild Mobile Interactions. Proceedings ofthe ACM on Human-Computer Interaction 7, ISS (2023), 357–375.

[30] Luv Kohli and Mary Whitton. 2005. The haptic hand: providing user interface feedback with the non-dominant hand in virtual environments. In Proceedings ofGraphics Interface 2005. Citeseer, 1–8

[31] Yuki Kubo. 2022. Ring-type Indirect Pointing Device for Large Displays using Three-axis Pressure Sensor. In Proceedings of the 2022 ACM Symposium on Spatial User Interaction. 1–2.

[32] Alexander Kyu, Hongyu Mao, Junyi Zhu, Mayank Goel, and Karan Ahuja. 2024. Eitpose: Wearable and practical electrical impedance tomography for continuous hand pose estimation. In Proceedings ofthe 2024 CHI Conference on Human Factors in Computing Systems. 1–10.

[33] Chi-Jung Lee, Ruidong Zhang, Devansh Agarwal, Tianhong Catherine Yu, Vipin Gunda, Oliver Lopez, James Kim, Sicheng Yin, Boao Dong, Ke Li, et al. 2024. Echowrist: Continuous hand pose tracking and hand-object interaction recognition using low-power active acoustic sensing on a wristband. In Proceedings ofthe 2024 CHI Conference on Human Factors in Computing Systems. 1–21.

[34] Jaewook Lee, Jun Wang, Elizabeth Brown, Liam Chu, Sebastian S Rodriguez, and Jon E Froehlich. 2024. GazePointAR: A context-aware multimodal voice assistant for pronoun disambiguation in wearable augmented reality. In Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems. 1–20.

[35] Lik-Hang Lee and Pan Hui. 2018. Interaction methods for smart glasses: A survey. IEEE access 6 (2018), 28712–28732.

[36] Chentao Li, Zirui Gao, Mingze Gao, Yinglian Ren, Jianjiang Feng, and Jie Zhou. 2026. Do MLLMs Understand Pointing? Benchmarking and Enhancing Referential Reasoning in Egocentric Vision. In Findings of the Association for Computational Linguistics: ACL 2026. Association for Computational Linguistics, San Diego, California, United States, 17000–17019. doi:10.18653/v1/2026.findings-acl.838

[37] Chentao Li, Ziheng Xi, Jianjiang Feng, and Jie Zhou. 2025. FineType: Fine-grained Tapping Gesture Recognition for Text Entry. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems. 1–20.

[38] Chentao Li, Jinyang Yu, Ke He, Jianjiang Feng, and Jie Zhou. 2024. SwivelTouch: Boosting Touchscreen Input with 3D Finger Rotation Gesture Proceedings ofthe ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies 8, 2, Article 53 (May 2024), 30 pages. doi:10.1145/3659584

[39] Yu Lu, Dian Ding, Hao Pan, Yijie Li, Juntao Zhou, Yongjian Fu, Yongzhao Zhang, Yi-Chao Chen, and Guangtao Xue. 2024. Handpad: Make your hand an on-the-go writing pad via human capacitance. In Proceedings ofthe 37th Annual ACM Symposium on User Interface Software and Technology. 1–16.

[40] Guy Lüthi, Andreas Rene Fender, and Christian Holz. 2022. DeltaPen: A device with integrated high-precision translation and rotation sensing on passive surfaces. In Proceedings ofthe 35th Annual ACM Symposium on User Interface Software and Technology. 1–12.

[41] Vitus Maierhöfer, Andreas Schmid, and Raphael Wimmer. 2024. TipTrack: precise, low-latency, robust optical pen tracking on arbitrary surfaces using an IR-emitting pen tip. In Proceedings ofthe Eighteenth International Conference on Tangible, Embedded, and Embodied Interaction. 1–13.

[42] Päivi Majaranta and Andreas Bulling. 2014. Eye tracking and eye-based human–computer interaction. In Advances in physiological computing. Springer, 39–65

[43] Shahzad Malik and Joe Laszlo. 2004. Visual touchpad: a two-handed gestural input device. In Proceedings ofthe 6th international conference on Multimodal interfaces. 289–296

[44] Fabrice Matulic, Riku Arakawa, Brian Vogel, and Daniel Vogel. 2020. Pensight: Enhanced interaction with a pen-top camera. In Proceedings ofthe 2020 CHI conference on human factors in computing systems. 1–14.

[45] Sven Mayer, Gierad Laput, and Chris Harrison. 2020. Enhancing mobile voice assistants with worldgaze. In Proceedings ofthe 2020 CHI Conference on Human Factors in Computing Systems. 1–10.

[46] Vimal Mollyn and Chris Harrison. 2024. EgoTouch: On-Body Touch Input Using AR/VR Headset Cameras. In Proceedings ofthe 37th Annual ACM Symposium on User Interface Software and Technology. 1–11.

[47] Kai Nickel and Rainer Stiefelhagen. 2003. Pointing gesture recognition based on 3D-tracking of face, hands and head orientation. In Proceedings of the 5th international conference on Multimodal interfaces. 140–146.

[48] Ju Young Oh, Jun Lee, Joong Ho Lee, and Ji Hyung Park. 2017. Anywheretouch: Finger tracking method on arbitrary surface using nailed-mounted imu for mobile hmd. In HCI International 2017–Posters’ Extended Abstracts: 19th International Conference, HCI International 2017, Vancouver, BC, Canada, July 9–14, 2017, Proceedings, Part I 19. Springer, 185–191.

[49] Edwin Olson. 2011. AprilTag: A robust and flexible visual fiducial system. In 2011 IEEE International Conference on Robotics and Automation. 3400–3407. doi:10.1109/ICRA.2011.5979561

[50] OpenAI. 2024. GPT-4o System Card. https://arxiv.org/abs/2410.21276.

[51] Krzysztof Pietroszek, Anastasia Kuzminykh, James R Wallace, and Edward Lank. 2014. Smartcasting: a discount 3D interaction technique for public displays. In Proceedings of the 26th Australian Computer-Human Interaction Conference on Designing Futures: the Future of Design. 119–128

[52] Krzysztof Pietroszek, James R Wallace, and Edward Lank. 2015. Tiltcasting: 3D interaction on large displays using a mobile device. In Proceedings of the 28th Annual ACM Symposium on User Interface Software & Technology. 57–62.

[53] Yue Qin, Chun Yu, Wentao Yao, Jiachen Yao, Chen Liang, Yueting Weng, Yukang Yan, and Yuanchun Shi. 2023. Selecting Real-World Objects via User-Perspective Phone Occlusion. In Proceedings ofthe 2023 CHI Conference on Human Factors in Computing Systems. 1–13.

[54] Yevhen Romaniak, Anastasiia Smielova, Yevhenii Yakishyn, Valerii Dziubliuk, Mykhailo Zlotnyk, and Oleksandr Viatchaninov. 2020. Nimble: Mobile Interface for a Visual Question Answering Augmented by Gestures. In Adjunct Proceedings ofthe 33rd Annual ACM Symposium on User Interface Software and Technology. 129–131.

[55] Hugo Romat, Andreas Fender, Manuel Meier, and Christian Holz. 2021. Flashpen: A high-fidelity and high-precision multi-surface pen for virtua reality. In 2021 IEEE Virtual Reality and 3D User Interfaces (VR). IEEE, 306–315.

[56] Xiyuan Shen, Chun Yu, Xutong Wang, Chen Liang, Haozhan Chen, and Yuanchun Shi. 2024. MouseRing: Always-available Touchpad Interaction with IMU Rings. In Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems. 1–19.

[57] Srinath Sridhar, Anders Markussen, Antti Oulasvirta, Christian Theobalt, and Sebastian Boring. 2017. Watchsense: On-and above-skin input sensing through a wearable depth sensor. In Proceedings ofthe 2017 CHI Conference on Human Factors in Computing Systems. 3891–3902.

[58] Lee Stearns, Uran Oh, Leah Findlater, and Jon E Froehlich. 2018. Touchcam: Realtime recognition of location-specific on-body gestures to support users with visual impairments. Proceedings ofthe ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies 1, 4 (2018), 1–23.

[59] Paul Streli, Jiaxi Jiang, Juliete Rossie, and Christian Holz. 2023. Structured Light Speckle: Joint Ego-Centric Depth Estimation and Low-Latency Contact Detection via Remote Vibrometry. In Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology. 1–12.

[60] Paul Streli, Mark Richardson, Fadi Botros, Shugao Ma, Robert Wang, and Christian Holz. 2024. TouchInsight: Uncertainty-aware Rapid Touch and Text Input for Mixed Reality from Egocentric Vision. In Proceedings ofthe 37th Annual ACM Symposium on User Interface Software and Technology 1–16.

[61] Marc Tonsen, Chris Kay Baumann, and Kai Dierkes. 2020. A high-level description and performance evaluation of pupil invisible. arXiv preprint arXiv:2009.00508 (2020)

[62] Daniel Vogel and Ravin Balakrishnan. 2005. Distant freehand pointing and clicking on very large, high resolution displays. In Proceedings of the 18th annual ACM symposium on User interface software and technology. 33–42.

[63] Cheng-Yao Wang, Wei-Chen Chu, Po-Tsung Chiu, Min-Chieh Hsiu, Yih-Harn Chiang, and Mike Y Chen. 2015. PalmType: Using palms as keyboard for smart glasses. In Proceedings of the 17th International Conference on Human-Computer Interaction with Mobile Devices and Services. 153–160.

[64] Cheng-Yao Wang, Min-Chieh Hsiu, Po-Tsung Chiu, Chiao-Hui Chang, Liwei Chan, Bing-Yu Chen, and Mike Y Chen. 2015. Palmgesture: Using palm as gesture interfaces for eyes-free input. In Proceedings ofthe 17th International Conference on Human-Computer Interaction with Mobile Devices and Services. 217–226.

[65] Zhaoguo Wang, Ziyuan Li, Chentao Li, Zihang Ao, Jianjiang Feng, and Jie Zhou. 2026. PianoBand: A Multimodal Wristband Interface for Portable Piano Interaction. In Proceedings ofthe 2026 CHI Conference on Human Factors in Computing Systems. Association for Computing Machinery, New York, NY, USA, Article 1085, 21 pages. doi:10.1145/3772318.3790607

[66] Zeyu Wang, Yuanchun Shi, Yuntao Wang, Yuchen Yao, Kun Yan, Yuhan Wang, Lei Ji, Xuhai Xu, and Chun Yu. 2024. G-voila: Gaze-facilitated information querying in daily scenarios. Proceedings ofthe ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies 8, 2 (2024), 1–33.

[67] Martin Weigel, Tong Lu, Gilles Bailly, Antti Oulasvirta, Carmel Majidi, and Jürgen Steimle. 2015. Iskin: flexible, stretchable and visually customizable on-body touch sensors for mobile computing. In Proceedings ofthe 33rd Annual ACM Conference on Human Factors in Computing Systems. 2991–3000.

[68] Martin Weigel, Vikram Mehta, and Jürgen Steimle. 2014. More than touch: understanding how people use skin as an input surface for mobile computing. In Proceedings ofthe SIGCHI Conference on Human Factors in Computing Systems. 179–188.

[69] Martin Weigel, Aditya Shekhar Nittala, Alex Olwal, and Jürgen Steimle. 2017. Skinmarks: Enabling interactions on body landmarks using conformal skin electronics. In proceedings ofthe 2017 CHI Conference on Human Factors in Computing Systems. 3095–3105.

[70] Wristcam Inc. 2020. Wristcam: Apple Watch Camera Band. https://wristcam.com/. Accessed: 2025-09-01.

[71] Ziheng Xi, Zihang Ao, Yitao Wang, Mingeze Gao, Wanmei Zhang, Jianjiang Feng, and Jie Zhou. 2026. WristPP: A Wrist-Worn System for Hand Pos And Pressure Estimation. arXiv preprint arXiv:2603.00606 (2026)

[72] Robert Xiao, Teng Cao, Ning Guo, Jun Zhuo, Yang Zhang, and Chris Harrison. 2018. LumiWatch: On-arm projected graphics and touch input. In Proceedings ofthe 2018 CHI Conference on Human Factors in Computing Systems. 1–11

[73] Zhanwei Xu, Haoxiang Pei, Jianjiang Feng, and Jie Zhou. 2025. FingerGlass: Enhancing Smart Glasses Interaction via Fingerprint Sensing. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems. 1–18

[74] Kun Yan, Zeyu Wang, Lei Ji, Yuntao Wang, Nan Duan, and Shuai Ma. 2024. Voila-A: Aligning Vision-Language Models with User’s Gaze Attention. Advances in Neural Information Processing Systems 37 (2024), 1890–1918

[75] Sang Ho Yoon, Ke Huo, and Karthik Ramani. 2016. Wearable textile input device with multimodal sensing for eyes-free mobile interaction during daily activities. Pervasive and Mobile Computing 33 (2016), 17–31.

[76] Jinyang Yu, Chentao Li, Jianjiang Feng, and Jie Zhou. 2025. 3D Touch Force Estimation from Capacitive Images. In Proceedings ofthe 30th Internationa Conference on Intelligent User Interfaces. Association for Computing Machinery, New York, NY, USA, 1–17. doi:10.1145/3708359.3712123

[77] Fan Zhang, Valentin Bazarevsky, Andrey Vakunov, Andrei Tkachenka, George Sung, Chuo-Ling Chang, and Matthias Grundmann. 2020. Mediapipe hands: On-device real-time hand tracking. arXiv preprint arXiv:2006.10214 (2020).

[78] Yang Zhang, Junhan Zhou, Gierad Laput, and Chris Harrison. 2016. Skintrack: Using the body as an electrical waveguide for continuous finger tracking on the skin. In Proceedings ofthe 2016 CHI Conference on Human Factors in Computing Systems. 1491–1503

## A Implementation Details

## A.1 Network Configuration

The visual encoder is initialized from a pre-trained ViT-B/16 model [10]. For single-channel input, the image is replicated to three channels before being fed into the backbone. The backbone output is reshaped into a spatial feature map and projected by a 1 × 1 convolution to a feature dimension of $dot { d } _ { \mathrm { m o d e l } } = 2 5 6$ . Absolute two-dimensional sinusoidal positional embeddings are then added to preserve spatial structure for localization.

The interaction module is a lightweight Transformer composed of three encoder layers. Each layer uses 8 attention heads, a feed-forward expansion ratio of4, GELU activation, and a dropout rate of0.1. A learnable class token is prepended to the token sequence, and its output representation is used as the shared feature for downstream prediction. Both the classification and regression heads are implemented as two-layer MLPs with a dropout rate of 0.5. The classification branch outputs four logits corresponding to Non-contact, Single-finger, Scroll, and Pinch, while the regression branch predicts the normalized 2D interaction coordinates (�, �).

The deployed model contains approximately 89M parameters, striking a practical balance between representational capacity and inference eficiency. Despite jointly solving three tasks—touch contact detection, gesture classification, and absolute position regression—the model runs eficiently on a Mac mini with an M4 Pro chip used in our user study, achieving an average inference latency of 7.2 ms per frame. This corresponds to a theoretical throughput of over 100 Hz, which is suficient to fully match the refresh rate of the current camera pipeline. Therefore, the model inference stage does not constitute the primary bottleneck of the system. Instead, it provides enough computational headroom for stable real-time interaction, ensuring that continuous palm-writing and multi-finger manipulations can be processed with low perceived delay. These findings support the practicality of our model design for deployment in interactive, low-latency scenarios.

## A.2 Data Augmentation

To improve robustness under practical use conditions, we apply data augmentation to simulate three common sources of variation in wrist-worn deployment. First, because the wrist-mounted camera may be slightly tilted during wearing, input images are randomly rotated within [−10 , 10 ]. Second, because the relative placement between the camera and the bottom of the palm may vary, we apply random translation with a maximum relative ofset of 0.1 in both horizontal and vertical directions. Third, because ambient illumination may change across environments, we randomly adjust image brightness and contrast with an ofset factor of 0.2. These augmentations improve robustness to small pose perturbations and lighting variation in real-world use.

## A.3 Training Details

All experiments are conducted on an NVIDIA GeForce RTX 4090 GPU with 24GB VRAM. The original image resolution is $1 9 2 0 \times 1 0 8 0$ , and all images are resized to 224 × 224 for training and inference. The model is trained using the Adam optimizer with an initial learning rate of $1 \times 1 0 ^ { - 4 }$ , weight decay of $1 \times 1 0 ^ { - 4 }$ , and batch size of 160. Training runs for up to 100 epochs with ReduceLROnPlateau for adaptive learning-rate scheduling and early stopping to mitigate overfitting.

## A.4 System Workflow

Fig. 9 illustrates the workflow: the user wears a wristband that illuminates the palm with infrared light and performs interactions using either a single finger or multiple fingers. Captured images are processed by a multi-task Transformer network to simultaneously predict the touch contact state, gesture class (i.e., single-finger, scroll, or pinch), and the absolute position. When contact is detected, the system applies diferent interaction logic based on the gesture class: fo single-finger mode, the system renders points on the canvas to simulate a realistic writing experience; for multi-finge gestures, the system maps the predicted coordinates to scrolling or zooming commands, enabling continuous spatial manipulation.

## B Data Collection Details

## B.1 Apparatus and Calibration

Fig. 11a illustrates the data collection setup, consisting of a wrist-mounted IR camera (input) and an overhead Nikon Z6 II camera (ground truth) equipped with a Nikkor 24–70 mm f/4S lens. The overhead camera streams 4K video at 30 FPS to a Mac Mini M4 Pro and is used only for calibration and ground-truth generation during data collection, rather than in deployment.

To align the palm surface with the calibration plane, we used a 3D-printed groove filled with molding clay (Fig. 11b) to accommodate diferent hand shapes. Four height-adjustable posts marked with AprilTags [49] define the physical coordinate system within a $1 9 . 8 \times 8 . 5$ cm rectangular area while avoiding marker occlusion in the wrist camera’s field of view.

We compute a homography matrix � to map the overhead camera’s pixel coordinates to the physical calibration plane. Using the four AprilTags, we estimate � via least squares. Given a pixel coordinate (�, �), the corresponding physical position $( x , y )$ is obtained as:

$$
{ \Bigg [ } { \boldsymbol { x } } ^ { \prime } { \Bigg ] } = H { \Bigg [ } { \boldsymbol { u } } { \Bigg ] } , \quad { \boldsymbol { x } } = { \frac { { \boldsymbol { x } } ^ { \prime } } { z ^ { \prime } } } , \quad { \boldsymbol { y } } = { \frac { y ^ { \prime } } { z ^ { \prime } } } .\tag{6}
$$

We approximate the palm as a planar surface; due to the large distance between the overhead camera and the hand, the efect of minor palm undulations is negligible.

## B.2 Ground Truth Generation

Fingertip positions are tracked using Google’s MediaPipe [77], a robust keypoint detection method widely used in hand pose research [8, 32, 33]. The detected pixel coordinates (red marker in Fig. 11c) are transformed into physical coordinates Manuscript submitted to ACM

![](images/a38635b1c6e6e45fb6f0634a16a6de14c69c01f013bb9c86ed2c36efc49e4091.jpg)  
Fig. 9. System workflow: The user interacts by touching their non-dominant palm with their dominant hand’s finger. A deep learning model processes this input in real-time, predicting both touch events and precise touch positions. Upon detecting valid touch interactions, the system maps these absolute coordinates to specific applications in the real world.

through �. Compared with reflective-marker-based optical tracking systems [9, 64], this markerless approach avoids visual artifacts in the wrist-mounted IR images and better preserves realistic bare-hand interaction conditions.

A Python script running on the Mac Mini M4 Pro synchronizes the two cameras at 15 FPS and computes ground-truth positions in real time. Before each experiment, we calibrated the system and ensured a reprojection error below $1 0 ^ { - 5 }$

## B.3 Collection Procedure

Data collection was conducted in quiet indoor settings at diferent times of day (morning, afternoon, and evening), with the devices placed in diferent locations within the room. The study included 9 sessions: 6 single-finger sessions (2 areas × 3 modes), 2 multi-finger sessions (Scroll and Pinch), and 1 non-contact session.

For single-finger sessions, participants performed serpentine sliding patterns. Interactions in the palm area included both longitudinal and transverse motions, whereas the finger area was limited to longitudinal sliding because of finger gaps. Unlike PalmGesture [64], we allowed participants to adopt comfortable finger pitch angles. For the two Manuscript submitted to ACM multi-finger sessions, interactions were restricted to the palm surface. To preserve ecological validity, we did not impose strict pose constraints; instead, participants performed gestures according to their natural habits, mimicking common Manuscript submitted to ACM

![](images/4905701a1a0853f5c5821c44425ce9bde2ba85047bc8723e1f61e767bffc9a4a.jpg)

![](images/61e4dbd1fc3420ee1cddc9a60fa99c5582709d62821cb1e2e74496c5246bd26c.jpg)  
Fig. 10. Blender render of the wristband casing.  
Fig. 11. Data collection setup: (a) Overall architecture of the capture system; (b) 3D-printed calibration device and corresponding calibration system; (c) Tracking of the dominant hand’s index fingertip pixel coordinates via MediaPipe [77] and their transformation into the calibrated palm plane coordinate system.

touchpad interactions. The non-contact session collected negative samples such as hovering and random gestures to help distinguish valid touches from false positives.

We explicitly excluded force-sensing resistors (FSRs) [46] for ground-truth collection. In pilot tests, FSRs introduced visual occlusion and were insuficiently sensitive to the light-touch interactions typical in our scenarios. To ensure coordinate consistency, participants aligned their palms within the 3D-printed groove relative to the AprilTag markers, with the palm center defined as the origin. During capture, users wore the device without an outer case to avoid occlusion and were encouraged to flex their wrists within an approximate range of ±30 to increase pose diversity while maintaining palm-plane visibility.

## C Supplementary Details for Single-Finger Study

Table 4. The average MAE and SD in diferent interaction modes and touch regions. Errors are reported in mm.
<table><tr><td></td><td></td><td colspan="3">Palm-Area</td><td colspan="3">Finger-Area</td><td colspan="3">Overall</td></tr><tr><td>Mode</td><td>Metric</td><td>Xerror</td><td>Yerror</td><td> $l _ { \mathbf { e r r o r } }$ </td><td>Xerror</td><td>Yerror</td><td> $l _ { \mathbf { e r r o r } }$ </td><td>Xerror</td><td>Yerror</td><td>lerror</td></tr><tr><td rowspan="2">LY</td><td>MAE</td><td>2.8</td><td>3.9</td><td>5.4</td><td>3.5</td><td>5.0</td><td>6.6</td><td>3.1</td><td>4.4</td><td>5.8</td></tr><tr><td>SD</td><td>2.6</td><td>3.1</td><td>3.3</td><td>3.2</td><td>4.3</td><td>4.6</td><td>2.9</td><td>3.6</td><td>3.9</td></tr><tr><td rowspan="2">VT</td><td>MAE</td><td>3.1</td><td>4.1</td><td>5.5</td><td>4.5</td><td>5.8</td><td>8.1</td><td>3.6</td><td>4.7</td><td>6.4</td></tr><tr><td>SD</td><td>2.9</td><td>3.0</td><td>3.4</td><td>4.6</td><td>5.6</td><td>6.5</td><td>3.6</td><td>4.2</td><td>4.9</td></tr><tr><td rowspan="2">RY</td><td>MAE</td><td>2.8</td><td>3.9</td><td>5.4</td><td>3.8</td><td>5.4</td><td>7.2</td><td>3.2</td><td>4.5</td><td>6.1</td></tr><tr><td>SD</td><td>2.5</td><td>3.2</td><td>3.3</td><td>3.2</td><td>4.3</td><td>4.6</td><td>2.9</td><td>3.8</td><td>4.0</td></tr><tr><td rowspan="2">Overall</td><td>MAE</td><td>2.9</td><td>3.9</td><td>5.4</td><td>3.9</td><td>5.4</td><td>7.2</td><td>3.3</td><td>4.5</td><td>6.1</td></tr><tr><td>SD</td><td>2.7</td><td>3.1</td><td>3.4</td><td>3.7</td><td>4.7</td><td>5.2</td><td>3.1</td><td>3.8</td><td>4.3</td></tr></table>

## C.1 Indoor Setings

To avoid restricting the evaluation to a single controlled environment, the indoor study was distributed across four everyday real-world scenarios: a morning ofice (4 users), a night meeting room (2), an evening lab (2), and an afternoon apartment (4).

## C.2 Additional Details of the Digit-Entry Study

For digit entry, twelve finger knuckles (excluding the thumb) were mapped to a keypad-like layout, with the three knuckles of the little finger representing digit 0. Participants first sequentially touched the designated knuckle regions. We recorded the corresponding coordinates, discarded the first and last 10% of samples to reduce noise, and used the remaining samples to construct a 3-nearest-neighbor classifier for knuckle identification. This stage established only the mapping between anatomical locations and digit labels and did not calibrate the PalmSpace sensing model itself.

In the testing phase, participants completed five rounds of random 10-digit sequences under eyes-free conditions. The touchscreen baseline followed the same protocol, with the right half of the screen partitioned into ten digit-mapped zones and visual feedback provided during registration to facilitate layout learning.

The detailed digit-level and transition-level results are shown in Table 5, and the confusion matrices are shown in Fig. 12. The largest touchscreen errors occurred for digits such as 1, 4, and 7, whereas PalmSpace maintained more uniform performance across digits. Transition analysis further showed that touchscreen accuracy dropped substantially from identical to adjacent and non-adjacent transitions, while PalmSpace remained stable across all three categories.

![](images/36890208020f37e1f51da277c86f4d4dbb538d458d072aafd7dc6ffa0c7cb23a.jpg)  
(a) PalmSpace

![](images/a434a3cf666e5da4020fee1b7df99d969e3ac8d39a5c5a97e02238b5909d8673.jpg)  
(b) Touchscreen  
Fig. 12. Confusion matrices for digit input: (a) PalmSpace, (b) Touchscreen.

Table 5. Comparison of Digit Recognition and Transition Accuracy (%) between PalmSpace and Touchscreen.
<table><tr><td></td><td colspan="10">Digit Recognition Accuracy (%)</td><td colspan="3">Transition Accuracy (%)</td></tr><tr><td>Method</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>Identical</td><td>Adjacent</td><td>Non-Adj.</td></tr><tr><td>PalmSpace</td><td>98.55</td><td>100.00</td><td>90.74</td><td>89.47</td><td>95.16</td><td>91.94</td><td>97.06</td><td>100.00</td><td>92.31</td><td>95.00</td><td>95.16</td><td>94.90</td><td>95.91</td></tr><tr><td>Touchscreen</td><td>95.65</td><td>46.77</td><td>85.19</td><td>70.18</td><td>54.84</td><td>74.19</td><td>75.00</td><td>48.15</td><td>90.38</td><td>88.33</td><td>97.87</td><td>79.60</td><td>69.29</td></tr></table>

## C.3 Handwriting Task User Study

Handwritten character input requires extremely high accuracy in finger touch detection and absolute finger positioning, as even small deviations can render text visually unrecognizable. Therefore, we conducted a handwriting input experiment to assess continuous input eficiency.

C.3.1 Task and Procedure. We selected the 10 digits (0-9) and 26 uppercase letters (A-Z) as input tasks. Each participant was randomly assigned 20 characters to write sequentially following GUI prompts, using their preferred palm posture. Thanks to advancements in VLM (Vision Language Model) technology, instead of training a dedicated model, we employed GPT-4o [50] for recognition. We used GPT-4o for its robustness to scale/rotation and direct structured output, unlike OCR models (e.g., Tesseract [13] and Paddle [2]), which are sensitive to scale/rotation and need complex tuning. For the touchscreen, participants rewrote the same 20 characters. In total, $1 2 \times 2 \times 2 0 = 4 8 0$ characters were collected.

C.3.2 Results. In the handwriting experiment, PalmSpace achieved an average recognition accuracy of 92.5%, with 26 of the 36 characters recognized at 100%. The confusion matrix is shown in Fig. 13a. However, some characters still exhibited significant confusion cases, particularly the digit ‘0,’ which was often misidentified as the similarly shaped Manuscript submitted to ACM

letters $\mathbf { \dot { D } } ^ { \prime } \mathbf { o r } \mathbf { \dot { O } } ^ { \prime }$ . Successfully recognized example results are shown in Fig. 13b, where most users adopted a landscape orientation of the palm as the writing surface.

For comparison, touchscreen input achieved 94.58% accuracy. Statistical analysis (Shapiro-Wilk and repeated-measures ANOVA) showed no significant diference between the two methods $( F _ { 1 , 1 1 } = 1 . 2 1 , p = 0 . 2 9 5 )$ ). PalmSpace’s slightly lower mean accuracy can be attributed to palm curvature, which makes it harder to maintain stable strokes than on a flat touchscreen.

Overall, PalmSpace demonstrated handwriting performance comparable to gold-standard devices (i.e., touchscreens), confirming its potential for precise, free-space handwriting input. In addition, as shown in Table 1, it is calibration-free (users could use the system immediately without individual setup) and supports 2D finger tracking and touch detection for high-precision natural handwriting, unlike systems that constrain users to continuous stroke input [17].

## C.4 Qualitative Feedback

Participants consistently reported that PalmSpace ofered stronger proprioceptive cues than the touchscreen during eyes-free use (Fig. 13c). Several participants noted that localizing positions on the palm felt easier than localizing positions on a visually unavailable screen. They also mentioned that PalmSpace may be especially promising in relaxed or immersive scenarios such as large-screen viewing or mixed-reality interaction, where taking out a phone or shifting visual attention to a display would be undesirable.

![](images/9cff1d220e9b2765e86dc42cbc070cad70e4e718367554e194e58df49224a607.jpg)  
(a) Confusion matrix for 36 handwriten characters.

![](images/30443fdb1c8ec25ca2a4c56f8daba4934c955a38eca4ac0e55723cb21a6d0dc9.jpg)  
(b) Example characters writen by diferent users.

![](images/ea47f3a2d275d8974fe5d46ab1fd901ce45f965997e55cea1acf461896894ddd.jpg)  
(c) Subjective ratings for PalmSpace and touchscreen input.  
Fig. 13. Handwriting recognition and subjective evaluation results.

## D Supplementary Details for Multi-Finger User Study

## D.1 Interaction Parameterization for Pinch

For continuous manipulation in Pinch Mode, PalmSpace maps the predicted interaction point $( x _ { \mathrm { { c u r r } } } , y _ { \mathrm { { c u r r } } } )$ relative to the palm center $( C _ { x } , C _ { y } )$ into a polar representation. Specifically, we compute:

$$
\begin{array} { r l } & { d = \sqrt { ( x _ { \mathrm { c u r r } } - C _ { x } ) ^ { 2 } + ( y _ { \mathrm { c u r r } } - C _ { y } ) ^ { 2 } } , } \\ & { \theta = \mathrm { a t a n } 2 \big ( - ( y _ { \mathrm { c u r r } } - C _ { y } ) , x _ { \mathrm { c u r r } } - C _ { x } \big ) . } \end{array}\tag{7}
$$

where � denotes the radial distance from the palm center and � denotes the angular position. In our interaction design, � is used to control scaling and � is used to control rotation. This representation allows PalmSpace to decompose a single palm-based multi-finger gesture into two continuous control channels.

## D.2 Transfer Functions for Continuous Control

To stabilize continuous interaction and improve controllability, we implemented separate transfer functions for scaling and rotation.

D.2.1 Scaling Control. The scaling factor is updated incrementally according to the change in radial distance: $\Delta d =$ $d _ { \mathrm { c u r r } } - d _ { \mathrm { s t a r t } }$ . We apply a sensitivity coeficient $\lambda _ { \mathrm { z o o m } } = 0 . 3$ to modulate the control-display ratio. The updated scale is computed as:

$$
S _ { \mathrm { n e w } } = \mathrm { c l i p } ( S _ { \mathrm { b a s e } } + \Delta d \cdot \lambda _ { \mathrm { z o o m } } , [ 0 . 2 , 5 . 0 ] )\tag{8}
$$

The clipping range [0.2, 5.0] prevents extreme visual distortion. This incremental update rule improves stability compared with directly mapping absolute radial distance to scale.

D.2.2 Rotation Control. Rotation is driven by the angular displacement $\Delta \theta = \theta _ { \mathrm { c u r r } } - \theta _ { \mathrm { s t a r t } } .$ . Because atan2 is discontinuous at ±180<sup>◦</sup>, we applied a wrapping compensation mechanism:

$$
\Delta \theta \gets \left\{ \begin{array} { l l } { \Delta \theta - 3 6 0 ^ { \circ } , } & { \mathrm { i f } \Delta \theta > 1 8 0 ^ { \circ } } \\ { \Delta \theta + 3 6 0 ^ { \circ } , } & { \mathrm { i f } \Delta \theta < - 1 8 0 ^ { \circ } } \\ { \Delta \theta , } & { \mathrm { o t h e r w i s e } } \end{array} \right.\tag{9}
$$

To support ergonomic operation with relatively small physical finger motion, we further apply a rotational gain factor $\gamma _ { \mathrm { r o t } } = 4 . 0$ . The final rotation angle is updated as:

$$
\Phi _ { \mathrm { n e w } } = \left( \Phi _ { \mathrm { b a s e } } + \Delta \theta \cdot \gamma _ { \mathrm { r o t } } \right) \quad \left( \mathrm { m o d } \ 3 6 0 ^ { \circ } \right)\tag{10}
$$

## E Supplementary Details for Real-World Outdoor Robustness

This section provides detailed methodologies, datasets, and quantitative ablation results for the outdoor evaluation scenarios discussed in the main paper.

## E.1 Daytime Data Collection and Protocol

The daytime experiment was designed to directly assess the model’s generalization capabilities in natural outdoor settings. Data collection was performed at locations arranged with the users, encompassing a variety of environments such as sunny and overcast conditions, parks, streets, and balconies. In total, 14 sets of in-the-wild data were collected,

![](images/59f14c6fb582d43b37dc3fb0449c3dbdb523eb08e26d8d92dcb6026c31e3d697.jpg)  
Fig. 14. Sample images from various outdoor data collection scenarios.

comprising 10,388 contact images and 3,514 non-contact images. During acquisition, participants were instructed to touch the entire capturable area of their palm without constraints on finger posture to simulate real-world usage. Fig. 14 illustrates sample images from these scenarios.

## E.2 Network Ablation Study

We compare the unified model with architectural baselines and independently trained task-specific models. All zero-sho models are trained on the indoor dataset and directly evaluated on the outdoor dataset described above. For the fine-tuned condition, the outdoor data are partitioned using leave-one-participant-out cross-validation. The quantitative comparison is reported with the main ablation analysis in Table 3(b).

## E.3 Note on Nightime Ground Truth

In the nighttime scenario, standard RGB-based visual ground truth methods were found to be unreliable. Specifically, MediaPipe’s hand tracking performance degrades significantly when ambient illuminance falls below 5 lux, leading to inaccurate keypoint detection. Therefore, the nighttime evaluation relied on the task completion time of the Fitts’ Law study rather than direct geometric error measurement.

## F Supplementary Details for SUS Feedback

We assessed perceived usability using the System Usability Scale (SUS) [5]. The overall SUS score was $7 2 . 5 \pm 1 4 . 0$ (median = 70.0, IQR = 16.0), corresponding to a Good usability level [3]. Participants gave high ratings to statements regarding ease of use $( M = 5 . 0 0 )$ and learnability $( M = 5 . 1 7 )$ , and agreement with negatively worded items remained low.

At the item level, participants gave high ratings to positively worded statements such as:

• “I thought the product was easy to use” $\ ' ( M = 5 . 0 0 )$

• “I imagine that most people would learn to use this product very quickly” $( M = 5 . 1 7 )$

• “I felt very confident using the product” $( M = 5 . 0 0 )$

These responses indicate favorable perceptions of ease of use, learnability, and confidence during interaction.

Agreement with negatively worded items remained relatively low, including:

Manuscript submitted to ACM

• “I found the product unnecessarily complex” (� = 2.09),

• “I found the product very awkward to use” (� = 2.33),

• “I needed to learn a lot of things before I could get going with this product” (� = 2.17).

• “I think that I would need the support of a technical person to be able to use this product” (� = 3.18).

These results suggest limited perceived friction and a relatively smooth learning curve.