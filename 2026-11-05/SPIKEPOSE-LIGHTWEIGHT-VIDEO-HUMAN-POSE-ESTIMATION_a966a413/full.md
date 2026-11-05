This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible.

# SPIKEPOSE: LIGHTWEIGHT VIDEO HUMAN POSE ESTIMATION

Yunhao Huang, Kaushik Bhargav Sivangi, Tianyi Yu, Fani Deligianni

School of Computing Science, University of Glasgow, UK

## ABSTRACT

Efficient video human pose estimation must balance localisation accuracy, motion quality, and energy cost. We present SpikePose, a lightweight video pose estimator that pairs a frame-reset, single-step spiking encoder with motion-aligned heatmap memory (MAM). The encoder supplies per-frame spatial evidence, while MAM estimates joint displacement, aligns historical heatmaps, and adaptively fuses them with current predictions. By confining temporal reasoning to a compact, lightweight module outside the spiking backbone, SpikePose helps bridge the gap between spiking and classical Artificial Neural Networks (ANN) on large-scale, non-event-driven video tasks. We evaluate spatial accuracy, motion errors, and estimated arithmetic energy on MPII and a fixed NTU60 Cross-Subject subset. Against the architecture-matched ANN with MAM, SpikePose cuts estimated arithmetic energy by 91.4% at a cost of just 1.12 points in head–bone-normalised PCK at 0.5 threshold $( \mathrm { P C K } _ { \mathrm { H B } } ( 0 . 5 ) )$ Within SpikePose, adding MAM reduces velocity and acceleration errors by 28.13% and 31.46% while slightly improving localisation, at a cost of only 0.003M parameters and 0.0023 analytical GFLOPs per frame. Code will be released on GitHub upon acceptance.

Index Terms— spiking neural networks, video human pose estimation, single-step spiking inference, heatmap memory, efficient inference

## 1. INTRODUCTION

2D Video human pose estimation (HPE) requires accurate joint localisation and reliable motion trajectories. Frame-wise backbones such as ResNet-50 Baselines [1, 2] and HRNet [3] provide strong spatial representations but do not explicitly use video history. Temporal approaches introduce feature warping, aggregation, and space– time modelling [4, 5, 6, 7, 8]. Processing high-dimensional temporal features increases computational costs, motivating a joint examination of localisation, motion quality, and arithmetic energy efficiency.

Spiking neural networks (SNNs) offers an alternative computational primitive for spatial encoding [9]. Bio-inspired SNN units accumulate inputs into a continuous membrane potential and emit a discrete spike after crossing a threshold before the potential resets or decays. This thresholded, event-driven behaviour replaces the dense multiplication operations of analog neural networks with sparse, accumulation-based updates, which is a key reason of the energy efficiency broadly reported for SNNs. SNNs are, however, natively formulated for asynchronous event streams. A central open challenge is scaling them to tasks such as image and video understanding that are not intrinsically event-based, where inputs arrive as dense synchronous frames [10]. Thresholded spiking activations have been applied to object detection and visual backbones by supplying static images and unrolling membrane dynamics over several steps [11, 12]. Video introduces an important distinction: the internal update index τ describes neural dynamics applied to an observation, whereas the physical frame index t advances only when a new observation is acquired. Increasing τ at a fixed t may refine the same image representation, but it cannot reveal inter-frame displacement. Directly retaining an unaligned membrane state is also problematic because a moving joint may occupy a different location in the next image. SNN recurrence is therefore neither cost-free nor inherently motion-aware video memory.

SpikePose assigns spatial encoding to a frame-reset, single-step SNN and cross-frame reasoning to motion-aligned heatmap memory (MAM). MAM estimates joint displacement from consecutive predictions and aligns historical heatmaps before fusion. Related motion-aware heatmap regression incorporates joint movement into heatmap targets [13]. Here, motion instead controls a recurrent heatmap state. This design keeps temporal processing outside the high-dimensional backbone. Our contributions are threefold:

• A novel compact 2D video HPE system that combines singlestep spiking spatial encoding with an independent temporal state, evaluated against an architecture-matched ANN to quantify its accuracy–energy trade-off.

• A lightweight MAM that aligns and adaptively retains heatmap history, to reduce frame-wise SNN’s motion errors with minimal computational overheads.

• Comprehensive validation on both the MPII image benchmark and the NTU60 video dataset, with component and filtering analyses assessing individual architectural contribution.

## 2. METHODS

Figure 1 shows the overall pipeline. Throughout this paper, Spike-Pose denotes the complete encoder–MAM system, whereas Spike-Pose w/o MAM denotes its frame-wise SNN variant without MAM. Specifically, the frame-reset spiking spatial encoder $( f _ { \Theta } ^ { \mathrm { S N N } } )$ processes each RGB frame independently using a single-step spiking inference schedule $( T _ { \mathrm { s n n } } = 1 )$ . By resetting the internal neuronal state before every frame, it extracts per-frame spatial heatmaps (H<sub>t</sub>) without carrying unaligned membrane history across frames. To handle cross-frame reasoning, the lightweight motion-aligned heatmap memory (MAM) module (g<sub>Φ</sub>) maintains a compact state $R _ { t } ~ = ~ ( M _ { t } , C _ { t } , q _ { t } )$ ) across frames. Operating entirely outside the high-dimensional backbone, MAM estimates joint displacement, warps historical heatmaps, and applies adaptive gating and residual fusion to significantly reduce motion errors with minimal computational overhead.

## 2.1. Single-Step Frame-Reset Spiking Spatial Encoder

The encoder maps a 256 × 256 crop to 16 joint heatmaps of size 64 × 64 through four progressively downsampling stages. An analog stem first extracts low-level features before the representation is handed off to spiking stages, which use I-LIF–Conv–BN operations to perform downsampling while keeping activations in the integer, event-driven spiking domain. These stages further rely on membrane-shortcut blocks [14] to preserve gradient flow and maintain training stability despite the network’s discrete, threshold-based activations. The final stage incorporates spatial attention to help the network focus on informative joint regions, while a spiking feature pyramid [15] fuses multi-scale features into a unified representation before a lightweight analog head decodes the final heatmaps. This design keeps almost all computation within the energy-efficient spiking domain, reserving analog operations for the input and output stages where precision is most critical.

![](images/034dfbbff29bc42f5268c6975b45286a0d11fe113e9e8384fb99f364878c5988.jpg)

![](images/0405f61acdb0e42ff5fb5d71711a7cb72904b34cfcbd9f5da2b065e4947ebaac.jpg)  
Fig. 1. Overview of the proposed SpikePose pipeline. The frame-reset spiking encoder extracts per-frame spatial evidence, while MAM performs compact cross-frame motion estimation, heatmap alignment, memory update, and residual decoding.

For internal update τ, the integer-valued LIF neuron [16] follows

$$
\begin{array} { r } { U _ { \tau } = \left\{ \begin{array} { l l } { X _ { \tau } , } & { \tau = 1 , } \\ { \lambda ( U _ { \tau - 1 } - \vartheta S _ { \tau - 1 } ) + X _ { \tau } , } & { \tau > 1 , } \end{array} \right. } \end{array}\tag{1}
$$

$$
S _ { \tau } = \mathrm { r o u n d } [ \mathrm { c l i p } ( U _ { \tau } / \vartheta , 0 , K ) ] ,\tag{2}
$$

We use $\lambda = 0 . 9 , \vartheta = 1 ,$ and $K \ = \ 4 ,$ , with a straight-through derivative inside the unclipped interval. The selected model uses only $\tau \ = \ 1 \colon$ its RGB input is supplied once, subsequent spiking convolutions consume integer activations, and membrane shortcuts remain real-valued. Here, K bounds spike magnitude rather than update count. Larger update counts are diagnostic controls in Sec. 3.

## 2.2. Lightweight Motion-Aligned Heatmap Memory

MAM maintains a compact heatmap-domain state through motion alignment, gated retention, and residual fusion. For joint $j ,$ let x index heatmap pixels.

Here, $\mathbf { C } _ { t } = [ \mathbf { c } _ { t , 1 } , \ldots , \mathbf { c } _ { t , J } ] ^ { \top }$ and $\mathbf { q } _ { t } = ( q _ { t , 1 } , \ldots , q _ { t , J } ) ^ { \top }$ , with $J = 1 6$ . The soft coordinate and confidence are

$$
P _ { t , j } ( \mathbf { x } ) = \mathrm { s o f t m a x } _ { \mathbf { x } } ( H _ { t , j } ) ( \mathbf { x } ) ,\tag{3}
$$

$$
\mathbf { c } _ { t , j } = \sum _ { \mathbf { x } } { P } _ { t , j } ( \mathbf { x } ) \mathbf { x } , \qquad q _ { t , j } = \operatorname* { m a x } _ { \mathbf { x } } { P } _ { t , j } ( \mathbf { x } ) .\tag{4}
$$

Coordinates and confidences are extracted only from raw spatial heatmaps, not from memory or fused outputs. The coarse inter-frame displacement is given as:

$$
\mathbf { d } _ { t , j } ^ { c } = \mathbf { c } _ { t , j } - \mathbf { c } _ { t - 1 , j } .\tag{5}
$$

The 17-D descriptor $\mathbf { z } _ { t , j }$ concatenates the current and previous coordinates and confidences, $\mathbf { d } _ { t , j } ^ { c }$ and its magnitude, and an 8-D joint embedding.

Directly averaging $M _ { t - 1 , : }$ <sub>j</sub> and $H _ { t , j }$ can blur spatially displaced peaks. A two-layer offset head therefore predicts a bounded residual correction, after which bilinear sampling aligns the previous memory:

$$
{ \bf d } _ { t , j } = { \bf d } _ { t , j } ^ { c } + 2 \operatorname { t a n h } ( h _ { \mathrm { o f f } } ( { \bf z } _ { t , j } ) ) ,\tag{6}
$$

$$
\overline { { M } } _ { t - 1 , j } = \mathcal { W } ( M _ { t - 1 , j } , { \bf d } _ { t , j } ) .\tag{7}
$$

The residual is limited to two heatmap pixels per axis, equivalent to eight pixels at the input resolution, so it locally refines rather than replaces the coordinate displacement. The warp W uses bilinear sampling with boundary padding.

A lightweight gating head predicts a joint-wise history-retention gate shared across spatial locations:

$$
A _ { t , j } = 0 . 0 2 + 0 . 9 3 \sigma ( h _ { \mathrm { g a t e } } ( \mathbf { z } _ { t , j } ) ) .\tag{8}
$$

The aligned memory and refined output are

$$
M _ { t , j } = A _ { t , j } \overline { { M } } _ { t - 1 , j } + ( 1 - A _ { t , j } ) H _ { t , j } ,\tag{9}
$$

$$
\widehat { H } _ { t , j } = H _ { t , j } + \gamma _ { j } ( M _ { t , j } - H _ { t , j } ) .\tag{10}
$$

At t = 1, MAM is bypassed: $\widehat { H } _ { 1 } = M _ { 1 } = H _ { 1 } ; { \mathbf { C } } _ { 1 }$ and q<sub>1</sub> are computed from $H _ { 1 }$ , and recurrent updates start at $t = 2$ . The learned per-joint coefficient $\gamma _ { j }$ is initialised to zero and clipped to [0, 1], preserving the direct path at initialisation and bounding the memory correction.

## 2.3. Training Objective

Heatmap targets are unit-peak Gaussians with $\sigma ~ = ~ 2$ heatmap pixels. Training back-propagates through each 16-frame window without detaching MAM state. The total objective combines the primary pose loss $( L _ { p o s e } )$ with auxiliary displacement supervision $( L _ { o f f s e t } )$ . Specifically, $L _ { p o s e }$ measures the mean-squared error on visible joints of the fused output $\widehat { H } _ { t }$ , normalised by the number of valid heatmap elements. $\boldsymbol { L _ { o f f s e t } }$ applies a Smooth L1 penalty to train the displacement branch using ground-truth coordinate differences across valid adjacent-frame pairs in V:

$$
\begin{array} { r } { \begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { p o s e } } + 0 . 0 1 \mathcal { L } _ { \mathrm { o f f s e t } } , } \\ { \mathcal { L } _ { \mathrm { o f f s e t } } = \underset { ( t , j ) \in \mathcal { V } } { \mathrm { m e a n ~ } } \mathrm { S m o o t h } _ { \mathrm { L 1 } } \left( \mathbf { d } _ { t , j } , \mathbf { c } _ { t , j } ^ { * } - \mathbf { c } _ { t - 1 , j } ^ { * } \right) . } \end{array} } \end{array}\tag{11}
$$

Here, $\mathbf { c } ^ { * }$ is the ground-truth coordinate in heatmap space, and $S m o o t h _ { L 1 }$ is averaged over valid coordinate elements. The raw heatmaps have no additional pose loss. Displacement supervision trains the offset branch, while no velocity or acceleration loss is imposed on the final trajectories.

## 3. EXPERIMENTS

Implementation details. For MAM training and validation, each source video contributes two deterministic, non-overlapping 16- frame clips sampled at uniformly spaced temporal centres. Frames are consecutive (stride one), training shuffles only the window order, and all frames within a window share the same crop and augmentation. Training resets MAM state at each sampled clip boundary. Complete-video evaluation retains MAM state across inference chunks and resets it at video boundaries. The neuronal state is reset before every physical frame.

## 3.1. Datasets and Evaluation Metrics

Image dataset. MPII is an in-the-wild 2D HPE benchmark with about 25,000 images and over 40,000 annotated people covering diverse human activities [17]. We use its official validation split to evaluate the frame-wise spatial encoder.

Video dataset. NTU RGB+D 60 (NTU60) is a large-scale indoor multimodal dataset containing 56,880 video samples across 60 action classes, with synchronised RGB, depth, infrared, and 3D skeleton streams [18]. We evaluate the complete video model under the official Cross-Subject split using a frozen stratified test subset of 4,563 videos and 419,968 physical frames. Kinect RGB-plane joints are mapped from NTU-25 to the MPII-style 16-joint layout, and all methods use one fixed skeleton-derived 256 × 256 tube crop per complete video. Full videos are processed sequentially. The MAM state is carried across inference chunks and reset only at sourcevideo boundaries. Duplicate spatial predictions from an overlapping terminal extraction window are discarded.

Metrics. For MPII, we report the standard PCKh(0.5) metric [17]. For NTU60, following the same head-normalised PCK principle, we define a head–bone-normalised variant (PCK<sub>HB</sub>). For joint j in physical frame t,

$$
e _ { t j } ^ { \mathrm { H B } } = \frac { \lVert \hat { { \bf p } } _ { t j } - { \bf p } _ { t j } \rVert _ { 2 } } { \lVert { \bf p } _ { t , \mathrm { h e a d } } - { \bf p } _ { t , \mathrm { n e c k } } \rVert _ { 2 } } ,\tag{12}
$$

$$
\mathrm { P C K } _ { \mathrm { H B } } ( \alpha ) = \frac { \sum _ { t , j } m _ { t j } \mathbb { I } [ e _ { t j } ^ { \mathrm { H B } } \leq \alpha ] } { \sum _ { t , j } m _ { t j } } .\tag{13}
$$

Here, $m _ { t j } ~ = ~ 1$ only when the evaluated joint, head, and neck are valid and the head–neck bone length is non-zero. We report PCK<sub>HB</sub>(0.5) over all evaluated frames.

Mean Per-Joint Velocity Error (MPJVE) and Mean Per-Joint Acceleration Error (MPJAccE) measure the mean Euclidean discrepancies between the first- and second-order finite differences of predicted and ground-truth 2D joint trajectories, respectively [19, 20]. Their units are pixels/frame and pixels/frame<sup>2</sup>. Both metrics are computed from reconstructed full-video trajectories, include valid differences across inference-window boundaries, and are averaged equally across videos.

Computation and arithmetic energy. Costs are reported for 256 × 256 input crops. Analytical GFLOPs count dense convolutional/linear operations and supported additional operators, with two FLOPs per multiply-Accumulation operations(MACs). Here, it covers real-valued convolutional/linear paths. We distinguish MACs from spike-driven Accumulate operations(ACs) using integer-spike accounting [12, 16]. SNN activity is averaged over 28,404 16-frame windows from full-video evaluation. Using 45-nm FP32 reference costs of 4.6 pJ/MAC and 0.9 pJ/AC [21], we estimate:

$$
E _ { \mathrm { p r o x y } } ( \mathrm { m J } ) = \frac { 4 . 6 N _ { \mathrm { M A C } } + 0 . 9 N _ { \mathrm { A C } } } { 1 0 ^ { 9 } } .\tag{14}
$$

Softmax, warping, membrane updates, element-wise fusion, and memory/data movement are not fully covered. This energy proxy reflects analytical computational costs rather than directly measured device energy or execution latency.

## 3.2. Main Results

Table 1 compares the complete SpikePose model with ANN and SNN baselines on the frozen NTU60 subset. We denote the architecture-matched ANN baseline without and with MAM as SpikePose–ANN w/o MAM and SpikePose–ANN, respectively.

SpikePose achieves $\mathrm { P C K } _ { \mathrm { H B } } ( 0 . 5 ) = 8 3 . 8 1$ , a MPJVE of 3.50, and an MPJAccE of 5.50. Relative to SpikePose w/o MAM, it increases $\mathrm { P C K } _ { \mathrm { H B } } ( 0 . 5 )$ by 0.18 points while reducing MPJVE and MPJAccE by 28.13% and 31.46%, respectively. MAM adds 0.003M parameters and approximately 0.0023 analytical GFLOPs per processed frame. Overall, its main benefit is improved motion quality along with a subtle localisation gain.

SpikePose bridges the performance gap that traditionally hinders spiking neural networks in complex video tasks, achieving competitive accuracy relative to its analog counterparts. While, HRNet-W32 and the architecture-matched SpikePose–ANN exceed Spike-Pose in $\mathrm { P C K } _ { \mathrm { H B } } ( 0 . 5 )$ by 2.54 and 1.12 points, respectively (86.35 and 84.93 versus 83.81), SpikePose matches the operation count of the architecture-matched ANN while uses 58.78% fewer analytical GFLOPs and 73.64% fewer parameters than HRNet-W32. By successfully narrowing this accuracy deficit under a spike-driven framework, SpikePose establishes a scalable and highly efficient accuracymotion-computation operating point for real-world deployment.

## 3.3. Ablation Study

MPII spatial encoder results. Table 2 compares spatial accuracy and estimated arithmetic energy on MPII. SpikePose–ANN w/o MAM and single-step SpikePose w/o MAM obtain PCKh(0.5) scores of 87.05 and 85.84, respectively. With similar analytical computation of 8.44 and 8.46 GFLOPs per crop, the single-step SNN reduces estimated arithmetic energy from 19.309 to 1.820 mJ per crop (90.6%). Increasing $T _ { \mathrm { s n n } }$ to 2 improves PCKh to 86.16 but raises estimated energy to 2.984 mJ per crop.

Table 1. Results on the frozen NTU60 Cross-Subject test subset. All rows use a complete video and MAM state reset only at video boundaries, as described in Sec. 3. MPJVE and MPJAccE average videos equally. GFLOPs and $E _ { \mathrm { p r o x y } }$ are normalized per processed frame; parameters are in millions. Bold marks the best valid ANN and SNN result separately.
<table><tr><td>Method</td><td>Type</td><td> $\mathrm { P C K } _ { \mathrm { H B } } ( 0 . 5 ) \uparrow$ </td><td>MPJVE↓</td><td>MPJAccE↓</td><td>GFLOPs/frame↓</td><td> $E _ { \mathrm { p r o x y } }$  (mJ/frame)↓</td><td>Params (M)↓</td></tr><tr><td>ResNet-50</td><td>ANN</td><td>85.81</td><td>3.79</td><td>6.06</td><td>14.555</td><td>33.277</td><td>34.000</td></tr><tr><td>HRNet-W32</td><td>ANN</td><td>86.35</td><td>3.67</td><td>5.87</td><td>20.528</td><td>46.889</td><td>28.540</td></tr><tr><td>SpikePose-ANN w/o MAM</td><td>ANN</td><td>84.92</td><td>3.11</td><td>4.81</td><td>8.443</td><td>19.309</td><td>7.518</td></tr><tr><td>SpikePose-ANN</td><td>ANN</td><td>84.93</td><td>3.03</td><td>4.69</td><td>8.445</td><td>19.310</td><td>7.523</td></tr><tr><td>SpikeYOLO-Frame</td><td>SNN</td><td>84.73</td><td>4.88</td><td>8.06</td><td>17.097</td><td>6.628</td><td>16.739</td></tr><tr><td>SpikePose w/o MAM (Ours)</td><td>SNN</td><td>83.63</td><td>4.87</td><td>8.02</td><td>8.459</td><td>1.627</td><td>7.520</td></tr><tr><td>SpikePose (Ours)</td><td>SNN</td><td>83.81</td><td>3.50</td><td>5.50</td><td>8.462</td><td>1.665</td><td>7.523</td></tr></table>

Table 2. MPII validation comparison. PCKh uses flip testing. GFLOPs and $E _ { \mathrm { p r o x y } }$ are per 256 × 256 person crop, including all $T _ { \mathrm { s n n } }$ updates and excluding flip testing. All arithmetic-energy estimates use the same MAC/AC accounting convention. PCK: PCKh (0.5); Parameters are in millions.
<table><tr><td colspan="5">Method Model PCKh↑ GFLOPs↓</td></tr><tr><td>SpikePose-ANN</td><td>ANN</td><td>87.05</td><td> $E _ { \mathrm { p r o x y } } \downarrow$  8.44</td><td>19.309</td><td>7.518</td></tr><tr><td>w/o MAM ResNet-50 [2]</td><td>ANN</td><td>88.54</td><td>14.56</td><td>33.277</td><td>33.999</td></tr><tr><td>HRNet-W32 [3]</td><td>ANN</td><td>90.32</td><td>20.53</td><td>46.889</td><td>28.536</td></tr><tr><td>SpikeYOLO [16]</td><td>SNN</td><td>87.79</td><td>17.10</td><td>6.579</td><td>16.739</td></tr><tr><td>SpikePose w/o MAM  $( T _ { \mathrm { s n n } } = 1 )$ </td><td>SNN</td><td>85.84</td><td>8.46</td><td>1.820</td><td>7.518</td></tr><tr><td>SpikePose w/o MAM  $( T _ { \mathrm { s n n } } = 2 )$ </td><td>SNN</td><td>86.16</td><td>16.92</td><td>2.984</td><td>7.518</td></tr></table>

Table 3. Frame/update controls for the SpikePose spatial encoder on NTU60. $V _ { m } U _ { n }$ uses m distinct physical frames in n SNN updates. Frames are repeated uniformly when $n > m .$
<table><tr><td>Schedule</td><td> $\mathrm { P C K } _ { \mathrm { H B } } ( 0 . 5 ) $  ←</td><td>MPJVE↓</td><td>MPJAccE↓</td><td> $E _ { \mathrm { p r o x y } } \downarrow$ </td></tr><tr><td> $V _ { 1 } U _ { 1 }$ </td><td>85.08</td><td>5.30</td><td>8.81</td><td>1.788</td></tr><tr><td> $V _ { 1 } U _ { 2 }$ </td><td>85.09</td><td>5.20</td><td>8.60</td><td>2.982</td></tr><tr><td> $V _ { 1 } U _ { 4 }$ </td><td>84.75</td><td>4.94</td><td>8.13</td><td>5.616</td></tr><tr><td> $V _ { 2 } U _ { 2 }$ </td><td>83.03</td><td>5.32</td><td>8.73</td><td>2.969</td></tr><tr><td> $V _ { 2 } U _ { 4 }$ </td><td>83.14</td><td>4.89</td><td>7.95</td><td>5.564</td></tr><tr><td> $V _ { 4 } U _ { 4 }$ </td><td>80.66</td><td>6.22</td><td>9.74</td><td>5.602</td></tr></table>

Table 4. Core MAM-component ablation on the frozen NTU60 test subset with state reset only at source-video boundaries.
<table><tr><td colspan="3">MAM components</td><td colspan="2">NTU60 test subset</td></tr><tr><td>Motion alignment</td><td>Adaptive gate</td><td>Residual fusion</td><td> $\mathrm { P C K } _ { \mathrm { H B } } ( 0 . 5 )$  ↑</td><td>MPJAccE↓</td></tr><tr><td>X</td><td>X</td><td>X</td><td>83.63</td><td>8.02</td></tr><tr><td>X</td><td>√</td><td>√</td><td>83.77</td><td>5.86</td></tr><tr><td>√</td><td>X</td><td>√</td><td>83.75</td><td>5.66</td></tr><tr><td>√</td><td>√</td><td>X</td><td>83.85</td><td>5.53</td></tr><tr><td>√</td><td>√</td><td>√</td><td>83.81</td><td>5.50</td></tr></table>

Number of SNN updates. Table 3 distinguishes repeated processing of one image from access to distinct physical frames. It shows the highest PCK<sub>HB</sub>(0.5) for V<sub>1</sub>U<sub>2</sub> (85.09) and the lowest MPJAccE for V<sub>2</sub>U<sub>4</sub> (7.95).

MAM components. Table 4 compares separately trained configurations with spatial adaptation. Removing alignment disables both warping and offset supervision. Removing adaptive gating retains learned, input-independent per-joint retention, initialised to 0.08. Removing residual fusion outputs memory directly, equivalent to $\gamma _ { j } = 1$ . The no-alignment and input-independent-retention variants obtain MPJAccE of 5.86 and 5.66, versus 5.50 for the full model. Without residual fusion, MPJAccE is 5.53 and $\mathrm { P C K } _ { \mathrm { H B } } ( 0 . 5 )$ is slightly higher. These results favor the full configuration for motion error, with only a small difference from direct memory output.

Table 5. Filtering comparison on NTU60. Base is Spike-Pose without MAM; + rows apply coordinate filters to Base. PCK: $\mathrm { P C K } _ { \mathrm { H B } } ( 0 . 5 )$ ; VE: MPJVE; AE: MPJAccE. L/M/H: low/medium/high motion-speed strata (1,521 videos each). EMA: exponential moving average [22]; One Euro: One Euro filter [23]; SG: Savitzky–Golay filter [24].
<table><tr><td rowspan="2">Method</td><td colspan="3">Overall</td><td colspan="3">AE by speed</td></tr><tr><td>PCK↑</td><td>VE↓</td><td>AE↓</td><td>L↓</td><td>M↓</td><td>H↓</td></tr><tr><td>Base</td><td>83.63</td><td>4.87</td><td>8.02</td><td>4.81</td><td>6.77</td><td>12.48</td></tr><tr><td>+ EMA</td><td>83.69</td><td>4.15</td><td>6.60</td><td>3.70</td><td>5.51</td><td>10.59</td></tr><tr><td>+ One Euro</td><td>83.78</td><td>3.61</td><td>5.64</td><td>2.55</td><td>4.46</td><td>9.90</td></tr><tr><td>+ SG</td><td>83.60</td><td>4.47</td><td>6.91</td><td>3.91</td><td>5.77</td><td>11.03</td></tr><tr><td>SpikePose</td><td>83.81</td><td>3.50</td><td>5.50</td><td>2.68</td><td>4.48</td><td>9.33</td></tr></table>

Filtering and motion speed. One Euro filtering operates on the same DARK-decoded [25] spatial-baseline coordinates in original RGB pixels. The filter uses minimum cutoff 3 Hz, $\beta = 0 . 1$ derivative cutoff 1 Hz, and sampling frequency 30 Hz. Parameters are selected on validation predictions using $\mathrm { P C K } _ { \mathrm { H B } } ( 0 . 5 )$ with a $1 0 ^ { - 3 }$ -weighted normalised MPJAccE penalty, then fixed across test videos and speed strata. In Table 5, SpikePose has the lowest overall MPJVE and MPJAccE, improving on One Euro by 3.0% and 2.5%.

For high-speed videos, SpikePose reduces MPJAccE from 9.90 to 9.33 relative to One Euro, with a $\mathrm { P C K } _ { \mathrm { H B } } ( 0 . 5 )$ trade-off from 73.13 to 72.88; One Euro has lower low- and mid-speed MPJAccE.

## 4. CONCLUSION

We presented SpikePose, a lightweight video HPE system that separates spatial and temporal processing: a frame-reset spiking encoder produces per-frame heatmaps with a single internal update, while MAM confines cross-frame reasoning to a compact, aligned heatmap memory outside the backbone. Experiments on MPII and a frozen NTU60 subset show that MAM substantially improves motion consistency at negligible additional cost, and that SpikePose narrows the accuracy gap with its ANN heatmap-based counterpart while retaining the energy advantages characteristic of spiking computation.

## Acknowledgements

F. Deligianni acknowledges funding from EPSRC (EP/W01212X/1), the Royal Society (RGS/R2/212199), and the Academy of Medical Sciences (NGR1/1678).

## 5. REFERENCES

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 770–778.

[2] Bin Xiao, Haiping Wu, and Yichen Wei, “Simple baselines for human pose estimation and tracking,” in Computer Vision – ECCV. 2018, pp. 472–487, Springer.

[3] Ke Sun, Bin Xiao, Dong Liu, and Jingdong Wang, “Deep highresolution representation learning for human pose estimation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 5693–5703.

[4] Tomas Pfister, James Charles, and Andrew Zisserman, “Flowing convnets for human pose estimation in videos,” in Proceedings of the IEEE International Conference on Computer Vision, 2015, pp. 1913–1921.

[5] Gedas Bertasius, Christoph Feichtenhofer, Du Tran, Jianbo Shi, and Lorenzo Torresani, “Learning temporal pose estimation from sparsely-labeled videos,” in Advances in Neural Information Processing Systems, 2019, vol. 32.

[6] Zhenguang Liu, Runyang Feng, Haoming Chen, Shuang Wu, Yixing Gao, Yunjun Gao, and Xiang Wang, “Temporal feature alignment and mutual information maximization for video-based human pose estimation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 11006–11016.

[7] Jijie He and Wenwu Yang, “Video-based human pose regression via decoupled space-time aggregation,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 1022–1031.

[8] Runyang Feng, Hyung Jin Chang, Tze Ho Elden Tse, Boeun Kim, Yi Chang, and Yixing Gao, “High-resolution spatiotemporal modeling with global-local state space models for video-based human pose estimation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 8929–8938.

[9] Emre O. Neftci, Hesham Mostafa, and Friedemann Zenke, “Surrogate gradient learning in spiking neural networks: Bringing the power of gradient-based optimization to spiking neural networks,” IEEE Signal Processing Magazine, vol. 36, no. 6, pp. 51–63, 2019.

[10] Dhireesha Kudithipudi, Catherine Schuman, Craig M. Vineyard, Tej Pandit, Cory Merkel, Rajkumar Kubendran, James B. Aimone, Garrick Orchard, Christian Mayr, Ryad Benosman, Joe Hays, Cliff Young, Chiara Bartolozzi, Amitava Majumdar, Suma George Cardwell, Melika Payvand, Sonia Buckley, Shruti Kulkarni, Hector A. Gonzalez, Gert Cauwenberghs, Chetan Singh Thakur, Anand Subramoney, and Steve Furber, “Neuromorphic computing at scale,” Nature, vol. 637, no. 8047, pp. 801–812, 2025.

[11] Seijoon Kim, Seongsik Park, Byunggook Na, and Sungroh Yoon, “Spiking-YOLO: Spiking neural network for energyefficient object detection,” in Proceedings ofthe AAAI Conference on Artificial Intelligence, 2020, vol. 34, pp. 11270–11277.

[12] Man Yao, Jiakui Hu, Zhaokun Zhou, Li Yuan, Yonghong Tian, Bo Xu, and Guoqi Li, “Spike-driven transformer,” in Advances in Neural Information Processing Systems, 2023, vol. 36, pp. 64043–64058.

[13] Inpyo Song, Jongmin Lee, Moonwook Ryu, and Jangwon Lee, “Motion-aware heatmap regression for human pose estimation in videos,” in Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, 2024, pp. 1245– 1253.

[14] Yifan Hu, Lei Deng, Yujie Wu, Man Yao, and Guoqi Li, “Advancing spiking neural networks toward deep residual learning,” IEEE Transactions on Neural Networks and Learning Systems, vol. 36, no. 2, pp. 2353–2367, 2025.

[15] Tsung-Yi Lin, Piotr Dollar, Ross Girshick, Kaiming He,´ Bharath Hariharan, and Serge Belongie, “Feature pyramid networks for object detection,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017, pp. 2117–2125.

[16] Xinhao Luo, Man Yao, Yuhong Chou, Bo Xu, and Guoqi Li, “Integer-valued training and spike-driven inference spiking neural network for high-performance and energy-efficient object detection,” in Computer Vision – ECCV. 2024, pp. 253– 272, Springer.

[17] Mykhaylo Andriluka, Leonid Pishchulin, Peter Gehler, and Bernt Schiele, “2D human pose estimation: New benchmark and state of the art analysis,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2014, pp. 3686–3693.

[18] Amir Shahroudy, Jun Liu, Tian-Tsong Ng, and Gang Wang, “NTU RGB+D: A large scale dataset for 3D human activity analysis,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 1010–1019.

[19] Dario Pavllo, Christoph Feichtenhofer, David Grangier, and Michael Auli, “3D human pose estimation in video with temporal convolutions and semi-supervised training,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 7753–7762.

[20] Mansour Tchenegnon, Sylvie Gibet, and Thibaut Le Naour, “A new spatio-temporal loss function for 3D motion reconstruction and extended temporal metrics for motion evaluation,” arXiv preprint arXiv:2210.08562, 2022.

[21] Mark Horowitz, “Computing’s energy problem (and what we can do about it),” in 2014 IEEE International Solid-State Circuits Conference Digest of Technical Papers, 2014, pp. 10–14.

[22] Robert G. Brown and Richard F. Meyer, “The fundamental theorem of exponential smoothing,” Operations Research, vol. 9, no. 5, pp. 673–685, 1961.

[23] Gery Casiez, Nicolas Roussel, and Daniel Vogel, “1 euro filter:´ A simple speed-based low-pass filter for noisy input in interactive systems,” in Proceedings of the SIGCHI Conference on Human Factors in Computing Systems, 2012, pp. 2527–2530.

[24] Abraham Savitzky and Marcel J. E. Golay, “Smoothing and differentiation of data by simplified least squares procedures,” Analytical Chemistry, vol. 36, no. 8, pp. 1627–1639, 1964.

[25] Feng Zhang, Xiatian Zhu, Hanbin Dai, Mao Ye, and Ce Zhu, “Distribution-aware coordinate representation for human pose estimation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 7093–7102.