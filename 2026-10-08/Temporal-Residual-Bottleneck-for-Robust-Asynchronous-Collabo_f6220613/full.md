# Temporal Residual Bottleneck for Robust Asynchronous Collaborative Perception

Melih Yazgan<sup>1,2</sup>, Ahmed Abouelazm<sup>1,2</sup>, and J. Marius Zöllner<sup>1,2</sup>

<sup>1</sup> FZI Research Center for Information Technology

surname@fzi.de

2 Karlsruhe Institute of Technology name.surname@kit.de

Abstract. Collaborative perception extends the sensing range of autonomous vehicles, but its performance degrades when shared features arrive stale or incomplete. Most latency-robust methods compensate delayed collaborator features through flow-guided alignment or direct feature transport. In this work, we formulate asynchronous collaborative perception as temporal residual prediction. Our Temporal Residual Bottleneck keeps a deterministic pose-warped collaborator feature as a conservative anchor and uses a ∆t-conditioned xLSTM to extract residual temporal evidence from the available history. A detector-facing residual bottleneck then applies only gated, regularized corrections before ego-side fusion, reducing the risk of overwriting reliable static structure when temporal correspondence is uncertain. Experiments on DAIR-V2X and OPV2V show that our method is especially efective under severe fixed/irregular delays and packet drops. On DAIR-V2X, the reported checkpoint trades a small amount of synchronized peak accuracy for better robustness under stronger communication degradation. Controlled diagnostics further indicate that direct feature transport has oracle headroom but can become unreliable when deployed without accurate correspondence. These results support temporal residual fusion as a practical alternative for asynchronous and incomplete collaborative perception. Code will be publicly released at https://url.fzi.de/8dk38.

Keywords: Collaborative perception · asynchronous fusion · temporal prediction · packet drop

## 1 Introduction

Collaborative Perception (CP) extends the sensing range of connected autonomous vehicles by aggregating complementary observations from nearby agents, which is particularly useful under occlusion and other challenging sensing conditions [7,13,14,19,30,37]. Recent work has increasingly considered practical constraints such as limited bandwidth, adverse weather, heterogeneous sensors, and realworld deployment [23,31,32,34,38,39]. However, V2X communication is neither instantaneous nor perfectly reliable: latency, asynchrony, and packet loss make collaborator information stale or incomplete, degrading feature alignment and fusion [11, 12, 20, 22, 28, 29, 36].

Most intermediate-fusion methods exchange Bird’s-Eye View (BEV) features and assume that received collaborator information corresponds to the ego vehicle’s current scene state [6, 8, 26, 27, 33]. Under delay, however, moving objects remain temporally inconsistent even after geometric alignment. Existing latencyaware approaches compensate stale features through temporal synchronization, feature-flow prediction, or direct feature transport [11, 20, 22, 28, 36]. Such compensation can become unreliable when motion or correspondence estimates are inaccurate, motivating a more conservative interface to temporal evidence.

We therefore formulate asynchronous collaborative perception as temporal residual prediction. Our Temporal Residual Bottleneck preserves a deterministic pose-warped collaborator feature as an anchor and learns only a gated residual correction. A ∆t-conditioned temporal branch, instantiated with xLSTM [3], summarizes irregular BEV histories, while the residual bottleneck restricts how this evidence modifies the feature passed to collaborative fusion. This avoids replacing reliable geometric structure with an uncertain transported estimate.

Experiments on DAIR-V2X [35] and OPV2V [27] show improved robustness under severe delay, packet loss, and pose noise. Controlled diagnostics further show that accurate oracle transport has useful headroom, whereas deployable correspondence-based transport can become brittle when applied directly in feature space.

Our contributions are threefold:

– We propose a Temporal Residual Bottleneck for asynchronous collaborative perception that anchors stale collaborator features through deterministic pose warping and applies temporal evidence as a gated residual correction.

– We introduce a ∆t-conditioned temporal formulation for irregular collaborator histories, instantiated with xLSTM, that preserves the pose-warped baseline instead of reconstructing the complete current feature.

– We provide controlled ablations and transport diagnostics that separate the efects of generic temporal prediction, the residual bottleneck, and correspondencebased transport, together with robustness evaluation on DAIR-V2X and OPV2V under delay, packet loss, and pose noise.

## 2 Related Work

Collaborative perception. Prior collaborative perception methods are commonly grouped into early, late, and intermediate fusion. Early fusion aggregates raw observations before detection, whereas late fusion combines object-level outputs from independently processed agents [2, 9]. Intermediate fusion exchanges taskrelevant features and ofers a practical trade-of between communication cost and perception accuracy [5,8,21,24]. Benchmarks and frameworks such as OpenCDA, OPV2V, and DAIR-V2X have standardized multi-agent 3D perception evaluation [25, 27, 35]. Our work follows this intermediate-fusion setting, but focuses on temporal mismatch between the ego reference time and delayed collaborator features.

Asynchrony, latency, and concurrent architectures. A growing body of work addresses the mismatch between ego and collaborator timestamps. SyncNet [11] uses recurrent temporal modeling to estimate synchronized features from delayed histories, while CoBEVFlow [22] and FFNet [36] utilize flow-based prediction or feature-warping mechanisms to align asynchronous features. LRCP [20] integrates latency-aware corrections into deformable attention, and CoDynTrust [28] builds on CoBEVFlow-style dense flow prediction with trust-based filtering to suppress unreliable compensated regions. More recently, DelAwareCol<sup>†</sup> [1] introduces dilated temporal convolutions and multi-head temporal self-attention with relative delay encoding, while CATNet<sup>†</sup> [4]<sup>3</sup> addresses communication asynchrony and feature inconsistency through spatio-temporal recurrent synchronization, wavelet-based denoising, and adaptive feature selection. While these methods demonstrate the necessity of modeling temporal misalignment, many rely on dense spatial attention, dense feature-flow prediction, or direct feature replacement. In contrast, our method keeps a deterministic pose-warped baseline as the detector-facing anchor and learns only gated residual corrections from temporal evidence.

Robustness to spatial misalignment and pose errors. Realistic V2X deployments must handle not only temporal delay but also spatial misalignment from localization noise. CoDynTrust [28] builds on the dense flow prediction introduced by CoBEVFlow, but replaces direct flow-based compensation with linear extrapolation of Region-of-Interest (ROI) boxes and a Dynamic Feature Trust Module (DFTM) to suppress regions where motion compensation is unreliable. This trust-based masking improves robustness to noisy alignment, but it mainly filters stale evidence rather than explicitly predicting the unobserved current dynamic state. Our diagnostics on explicit object transport lead to a complementary conclusion: causal detector-derived box tracking is often too noisy to safely guide feature-space alignment, motivating our gated residual correction instead of direct transported-feature injection.

Position of our work. Our work intersects recent eforts in temporal delay compensation and spatial robustness, but takes a diferent architectural approach. Existing methods typically recover a synchronized collaborator representation through feature prediction, spatial transport, or flow-based alignment before fusion. Instead, we frame asynchronous collaborative perception as a temporal residual prediction problem. The proposed Temporal Residual Bottleneck preserves a deterministic pose-warped collaborator feature as a geometric anchor and lets learned temporal evidence modify it only through a gated residual path, avoiding reconstruction or replacement of the complete current collaborator feature.

Although the temporal history is processed per collaborator before fusion, the formulation difers from conventional single-agent temporal feature prediction: the learned branch estimates only residual evidence relative to the pose-warped baseline rather than a complete current-time BEV representation. We instantiate the temporal encoder with a ∆t-conditioned xLSTM, but the residual formulation is not tied to this architecture, as shown by the temporal-backbone ablation in Sec. 4.7. Unlike 4D object tracking, our method does not maintain persistent object identities or explicitly estimate trajectories; it operates directly on dense collaborator BEV features before collaborative fusion. This separation between geometric anchoring and learned temporal correction is particularly useful under severe delay, missing evidence, or spatial perturbations.

## 3 Method

![](images/5e9601504c2990fc7bebab792266d1c4efba3a073cde335a155dd2621bdbd048.jpg)  
Fig. 1: Overview of the proposed temporal residual bottleneck. Delayed collaborator BEV features are processed in the collaborator local frame. The latest stale feature $S _ { i } ^ { t }$ is pose-warped into a geometric baseline $B _ { i } ^ { t }$ , while a ∆t-conditioned xLSTM extracts temporal residual evidence $R _ { i } ^ { t }$ from the available history. A detector-facing residual bottleneck predicts a correction $\delta F _ { i } ^ { t }$ , which is applied through an object-focused gate $G _ { i } ^ { t }$ as $\hat { F } _ { i } ^ { t } = B _ { i } ^ { t } + G _ { i } ^ { t } \odot \delta F _ { i } ^ { t }$ . The updated collaborator features are then warped to the ego frame and fused as multi-scale intermediate features for detection.

Fig. 1 illustrates the proposed Temporal Residual Bottleneck for asynchronous collaborative perception. Given delayed collaborator BEV features, our method first performs temporal compensation in the collaborator local frame before ego-frame fusion. It constructs a deterministic pose-warped geometric baseline, extracts temporal residual evidence from the available history using a ∆tconditioned xLSTM, and predicts a gated detector-facing correction. The updated collaborator features are then transformed into the ego frame, fused as multi-scale intermediate features, and passed to the detection head.

## 3.1 Problem Setup

We consider intermediate-fusion collaborative perception with one ego vehicle and one or more collaborators. Let $F _ { i } ^ { t } \in \mathbb { R } ^ { \hat { C } \times H \times \hat { W } }$ denote the collaborator feature of agent i at ego reference time t. Under asynchronous communication, the latest available collaborator feature is stale and is denoted by $S _ { i } ^ { t } = F _ { i } ^ { t - \Delta t _ { i } }$ where $\varDelta t _ { i } \geq 0$ is its realized age.

The available collaborator history may be irregularly spaced or partially missing. We denote the causal history by

$$
\mathcal { H } _ { i } ^ { t } = \{ ( F _ { i } ^ { t - \delta _ { k } } , \delta _ { k } , m _ { i , k } ) \} _ { k = 1 } ^ { K } ,
$$

where $\delta _ { k }$ is the temporal ofset and $m _ { i , k } \in \{ 0 , 1 \}$ indicates whether the corresponding historical feature is available. Received features may additionally be spatially incomplete under packet loss.

The method does not require prediction of future delay. Once a collaborator feature arrives, its timestamp determines its realized age, $\varDelta t _ { i } =$ $t - t _ { \mathrm { c a p t u r e } , i } .$ , which directly conditions the temporal branch. The same checkpoint therefore handles diferent realized delays without delay-regime prediction or model switching.

![](images/f3e50b35ef891ca785712569178a1d3873d209fadc9868c10613dd0d0b6e5dac.jpg)  
Fig. 2: Asynchronous collaborative-perception setting. At ego time $t ,$ the latest collaborator feature $S _ { i } ^ { t }$ has age $\varDelta t _ { i }$ . Earlier features form the irregular history $\mathcal { H } _ { i } ^ { t . } ;$ features may be unavailable $( m _ { i , k } = 0 )$ or spatially incomplete.

The goal is to construct a collaborator representation $\hat { F } _ { i } ^ { t }$ that supports current-time ego detection. Rather than recon-

structing the complete current collaborator feature ${ \mathrm { \bf { F } } } _ { i } ^ { t }$ , our deployable model retains a deterministic pose-warped baseline and learns only a gated temporal residual correction before collaborative fusion.

## 3.2 Pose-Warped Geometric Baseline

The first component is a deterministic geometric baseline. The most recent valid collaborator feature $S _ { i } ^ { t }$ is warped from its capture pose to the current collaborator reference time using the known relative pose transformation:

$$
B _ { i } ^ { t } = \mathscr { W } ( S _ { i } ^ { t } , T _ { i , t - \Delta t _ { i }  i , t } ) ,
$$

where $\mathcal { W } ( \cdot )$ is the BEV feature warp and $T _ { i , t - \varDelta t _ { i }  i , t }$ maps the stale collaborator coordinate frame to the collaborator’s current-time frame. This baseline is reliable for static scene structure and for ego/collaborator motion compensation, but it cannot fully correct independently moving objects. The warp uses the same localization metadata assumed by standard V2X feature alignment: the stale packet is associated with the collaborator pose at its capture time, and, following the benchmark protocol used by prior asynchronous $\mathrm { C P }$ baselines, the collaborator pose at the ego reference time is taken from the timestamped localization stream. This is not an object-motion oracle; it compensates only known ego/collaborator pose changes. We evaluate sensitivity to imperfect pose metadata in Table 5. After warping the latest stale feature into the current collaborator frame, we denote the resulting pose-warped geometric baseline as $B _ { i } ^ { t }$ This baseline compensates known $\mathrm { e g o } _ { / }$ /collaborator pose changes, but does not model independent object motion.

We therefore do not ask the network to synthesize the whole collaborator feature. Instead, the learned branch estimates temporal evidence relative to $B _ { i } ^ { t }$ and a separate residual bottleneck decides how much of that evidence should be applied to the detector-facing feature. The design question is how to exploit temporal evidence without corrupting the baseline when the evidence is uncertain.

## 3.3 Time-Conditioned Temporal Evidence

We use a recurrent temporal predictor to summarize the collaborator history:

$$
R _ { i } ^ { t } = \varPsi _ { \theta } ( \mathcal { H } _ { i } ^ { t } , T _ { i } ^ { t } ) ,
$$

where $\varPsi _ { \theta }$ is an xLSTM-based predictor and $\boldsymbol { { \Gamma } } _ { i } ^ { t }$ contains temporal ofsets and relative pose metadata. BEV cells are processed as temporal tokens so that each location’s short history is summarized by recurrent memory.

Delay is encoded using fixed multi-frequency Fourier features [16, 17],

$$
\phi ( { \varDelta } t ) = \left[ { \varDelta } t , \left\{ \sin ( 2 ^ { k } \pi { \varDelta } t ) , \cos ( 2 ^ { k } \pi { \varDelta } t ) \right\} _ { k = 0 } ^ { 4 } \right] ,
$$

followed by a learned FiLM conditioner. A separate pose/time token adds relative translation and periodic yaw information.

The recurrent output $R _ { i } ^ { t }$ is treated as temporal residual evidence rather than a standalone transported feature and is therefore routed through the residual bottleneck.

## 3.4 Detector-Facing Residual Bottleneck

The final model uses an attention-free residual bottleneck. We refer to the resulting collaborator representation as detector-facing because it is the feature directly exposed to the subsequent collaborative fusion and detection stages. The bottleneck converts recurrent temporal evidence into a residual correction while preserving the pose-warped baseline as the default prediction.

Let $S _ { i } ^ { t }$ be the latest stale feature and $B _ { i } ^ { t }$ its pose-warped geometric baseline. The residual bottleneck also receives the stale feature itself, so that it can compare the uncorrected stale evidence against the pose-warped baseline and the recurrent prediction.

$$
P _ { i } ^ { t } = B _ { i } ^ { t } + R _ { i } ^ { t }
$$

is the raw recurrent residual prediction. The residual bottleneck receives three evidence terms and an object-support confidence map:

$$
E _ { \mathrm { r e c } } = R _ { i } ^ { t } , \qquad E _ { \mathrm { s t a l e } } = S _ { i } ^ { t } - B _ { i } ^ { t } , \qquad E _ { \mathrm { a g r e e } } = S _ { i } ^ { t } - P _ { i } ^ { t } .
$$

Here $E _ { \mathrm { s t a l e } }$ is not treated as a physical current-frame residual; it is a warpdiscrepancy cue computed in the shared BEV tensor layout to indicate how much deterministic pose compensation changes the latest stale evidence. The agreement term exposes whether the recurrent prediction is consistent with the most recent collaborator evidence.

Each evidence term is projected into a compact bottleneck space. With a confidence map $C _ { i } ^ { t }$ and learned projections $\pi .$ , the correction is

$$
\begin{array} { r } { \delta F _ { i } ^ { t } = s \cdot \varOmega _ { \phi } \big ( \big [ \pi _ { \mathrm { r e c } } ( E _ { \mathrm { r e c } } ) , \pi _ { \mathrm { s t a l e } } ( E _ { \mathrm { s t a l e } } ) , \pi _ { \mathrm { a g r e e } } ( E _ { \mathrm { a g r e e } } ) , \pi _ { c } ( C _ { i } ^ { t } ) \big ] \big ) , } \end{array}
$$

where $\varOmega _ { \phi }$ is a small convolutional residual decoder and s is a residual scale. The decoder is zero-initialized, so the model starts from the conservative baseline $B _ { i } ^ { t }$ and learns only corrections that improve detection.

To avoid applying residuals uniformly over the BEV map, the model uses an object-focused gate $G _ { i } ^ { t } \in [ 0 , 1 ] ^ { 1 \times H \times W }$ . The gate is derived from detector objectness maps, not from ground-truth labels or oracle boxes. In our implementation, these maps are obtained by applying the lightweight Probability Score Map (PSM)/classification head to the same cached collaborator BEV features already available to the temporal module; therefore, the gate does not require transmitting an additional dense objectness tensor. Concretely, we collapse the resulting collaborator PSM logits into a single objectness map, take the most recent stale objectness and its pose-warped version, and use their union as object support. This support is dilated, assigned a small floor value, and detached before modulating the residual. The stale-and-warped union makes the gate less brittle under delay: stale objectness preserves evidence before warping, while the warped objectness covers the estimated current position. Ground-truth labels are used only to define auxiliary foreground masks during training; they are not used to route residuals at inference time. The final collaborator feature is

$$
\begin{array} { r } { \hat { F } _ { i } ^ { t } = B _ { i } ^ { t } + G _ { i } ^ { t } \odot \delta F _ { i } ^ { t } . } \end{array}
$$

This is the key detector-facing interface: temporal evidence can modify the baseline, but only through a learned, gated residual path. The scalar s controls the residual amplitude, the decoder is zero-initialized, and the training objective below regularizes over-applied corrections. We do not claim a formal hard bound such as spectral normalization or per-cell clipping; the safety argument is architectural and empirical, not a mathematical guarantee. The method therefore avoids direct feature overwrite by uncertain spatial transport.

## 3.5 Fusion and Detection

After temporal compensation, all collaborator features are aligned into the ego frame and processed by a Where2Comm-style multi-scale intermediate-fusion backbone with confidence-based communication masking and the same classification and box-regression heads [8]; the temporal residual module is inserted before this ego-frame fusion stage:

$$
\boldsymbol { F } _ { \mathrm { f u s e d } } ^ { t } = \boldsymbol { \varPhi } \left( \boldsymbol { F } _ { e } ^ { t } , \hat { F } _ { 1 } ^ { t } , \dots , \hat { F } _ { N - 1 } ^ { t } \right) .
$$

This keeps the detector architecture fixed; the proposed contribution is the temporal residual interface that prepares collaborator features before fusion.

## 3.6 Training Objective

The main objective is the standard detection loss,

$$
\mathcal { L } _ { \mathrm { d e t } } = \mathcal { L } _ { \mathrm { c l s } } + \lambda _ { \mathrm { b o x } } \mathcal { L } _ { \mathrm { b o x } } .
$$

We do not use a feature reconstruction loss in the final model. Instead, we add a small detector-facing applied-residual objective on foreground or label-supported regions. Let

$$
D _ { i } ^ { t , * } = \mathrm { s g } ( F _ { i } ^ { t } - B _ { i } ^ { t } )
$$

be the synchronized residual target available during training, where $\operatorname { s g } ( \cdot )$ denotes stop-gradient. Thus, the synchronized current feature supervises the correction but is not used as an inference input. Let $A _ { i } ^ { t } = G _ { i } ^ { t } \odot \delta F _ { i } ^ { t }$ be the residual actually applied to the detector-facing feature, and let $M _ { i } ^ { t }$ denote the foreground/label mask. The applied-residual objective is

$$
\mathcal { L } _ { \mathrm { a p p } } = \sum _ { i } \left( w _ { \mathrm { d i r } } \mathcal { L } _ { \mathrm { d i r } } ^ { i } + w _ { \mathrm { m a g } } \mathcal { L } _ { \mathrm { m a g } } ^ { i } + w _ { \mathrm { i m p } } \mathcal { L } _ { \mathrm { i m p } } ^ { i } + w _ { \mathrm { o v e r } } \mathcal { L } _ { \mathrm { o v e r } } ^ { i } \right) ,
$$

where

$$
\mathcal { L } _ { \mathrm { d i r } } ^ { i } = \big \langle 1 - \cos ( A _ { i } ^ { t } , D _ { i } ^ { t , * } ) \big \rangle _ { M _ { i } ^ { t } } , \qquad \mathcal { L } _ { \mathrm { m a g } } ^ { i } = \big \langle \mathrm { s m o o t h } { - \ell _ { 1 } } \left( \| A _ { i } ^ { t } \| _ { 2 } , \| D _ { i } ^ { t , * } \| _ { 2 } \right) \big \rangle _ { M _ { i } ^ { t } } .
$$

The remaining two terms discourage harmful corrections:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { i m p } } ^ { i } = \left. \mathrm { R e L U } \left( e ( B _ { i } ^ { t } + A _ { i } ^ { t } , F _ { i } ^ { t } ) - e ( B _ { i } ^ { t } , F _ { i } ^ { t } ) + m \right) \right. _ { M _ { i } ^ { t } } , } \\ & { \qquad \mathcal { L } _ { \mathrm { o v e r } } ^ { i } = \left. \mathrm { R e L U } \left( \| A _ { i } ^ { t } \| _ { 2 } - \rho \| D _ { i } ^ { t , * } \| _ { 2 } \right) \right. _ { M _ { i } ^ { t } } . } \end{array}
$$

Here $\langle \cdot \rangle _ { M _ { i } ^ { t } }$ denotes the average over masked BEV cells, $\cos ( \cdot , \cdot )$ uses ϵ-stabilized normalization, $e ( \cdot , \cdot )$ is the channel-mean absolute feature error, m is a small improvement margin, and $\rho$ controls the allowed residual-to-target magnitude ratio. The numerical weights and margins are listed in the Supp. A. The total training loss is

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { d e t } } + \lambda _ { \mathrm { a p p } } \mathcal { L } _ { \mathrm { a p p } } ,
$$

This objective supervises the correction actually exposed to the detector rather than reconstructing full BEV features. The model is trained end-to-end with the standard detection objective, so the residual interface is optimized jointly with the downstream fusion and detection losses.

## 4 Experiments

## 4.1 Experimental Setup

We evaluate on DAIR-V2X [35] and OPV2V Culver City [27] in the LiDAR-only intermediate-fusion setting using single-class vehicle detection. For DAIR-V2X, all methods use the same BEV discretization with voxel size (0.4, 0.4, 4.0), a LiDAR range of 201.6 m × 80 m, raw BEV size (200, 504), and stride-2 detection grid (100, 252). For OPV2V, we use a LiDAR range of 280 m×80 m. All methods in Table 1 and 3 use PointPillars as the LiDAR backbone [10]. We report AP at IoU thresholds 0.3, 0.5, and 0.7, and use AP@0.7 as the primary metric.

We evaluate seven communication settings: synchronized input, fixed and irregular 100/300/500 ms average delay. Fixed delay uses a deterministic first history ofset, while irregular delay samples temporal ofsets from a binomial history process with the same expected delay. We also evaluate a joint delayplus-packet-drop stress setting, where contiguous regions of non-ego collaborator BEV features are removed in burst mode at inference time.

Fair comparison protocol. All methods are evaluated with the same validation split, BEV grid, detection post-processing, delay settings, and AP computation. Published reference numbers are reported where available, while reproduced inframework integrations are marked separately. Each reproduced baseline follows its oficial training protocol. Packet drop is activated only at inference time and applied consistently to all compared methods.

Our method additionally consumes a four-frame collaborator history for temporal residual prediction. This history is obtained from an ego-side cache of packets received in previous timesteps; collaborators do not resend the last K frames at every step. Thus, the current communication payload is unchanged, while receiver-side memory and computation increase with history length. All latency-aware baselines are evaluated with their oficial temporal-cache or history settings, while our method uses k = 4 cached collaborator frames; thus, the comparison follows each method’s intended asynchronous input protocol rather than enforcing a single-frame setting. We report history-length ablations separately.

## 4.2 Implementation Details

We train the model end-to-end with Adam using learning rate $2 \times 1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 4 } , \epsilon = 1 0 ^ { - 1 0 }$ , cosine decay, and a 7-epoch warmup from $2 \times 1 0 ^ { - 5 }$ DAIR-V2X models are trained for 60 epochs with batch size 2, while OPV2V models are trained for 30 epochs with batch size 1. Random asynchronous training samples binomial delays with $p \in [ 0 . 0 5 , 0 . 3 0 ]$ , and the reported checkpoint is selected by validation performance. The temporal branch uses four history sweeps (k = 4), embedding dimension 64, and an m-s-m xLSTM stack. The residual bottleneck uses a 64-channel bottleneck, residual scale 1.5, zero-initialized output, and applied-residual loss weight 0.02. Packet drop is activated only at inference and is used as a severe missing-feature stress test rather than a calibrated wireless-channel model. Further implementation details are provided in Supp. A, and runtime is analyzed in Supp. D.

## 4.3 Comparison to DAIR-V2X Baselines

Table 1 compares our method against DAIR-V2X baselines under synchronized input and two irregular-delay settings.

Table 1: DAIR-V2X baseline comparison. AP@0.5/AP@0.7 under synchronized and irregular-delay settings. “pub.” denotes published reference numbers from CoDyn-Trust [28]; “rep.” denotes reproduced integrations in our framework. Bold and underline denote the best and second-best results, respectively.
<table><tr><td>Method</td><td colspan="2">Sync</td><td colspan="2">Irr. 300 ms</td><td colspan="2">Irr. 500 ms</td></tr><tr><td>Where2Comm+SyncNet [11] (pub.)</td><td>0.792</td><td>0.692</td><td>0.732</td><td>0.625</td><td>0.710</td><td>0.612</td></tr><tr><td>CoBEVFlow [22] (pub.)</td><td>0.775</td><td>0.663</td><td>0.742</td><td>0.626</td><td>0.732</td><td>0.621</td></tr><tr><td>CoDynTrust [28] (pub.)</td><td>0.792</td><td>0.700</td><td>0.743</td><td>0.643</td><td>0.736</td><td>0.637</td></tr><tr><td>Where2Comm+SyncNet (rep.)</td><td>0.782</td><td>0.688</td><td>0.738</td><td>0.606</td><td>0.732</td><td>0.603</td></tr><tr><td>CoBEVFlow (rep.)</td><td>0.788</td><td>0.660</td><td>0.739</td><td>0.597</td><td>0.718</td><td>0.586</td></tr><tr><td>CoDynTrust (rep.)</td><td>0.798</td><td>0.668</td><td>0.726</td><td>0.586</td><td>0.719</td><td>0.582</td></tr><tr><td>LRCP</td><td>0.795</td><td>0.669</td><td>0.769</td><td>0.639</td><td>0.735</td><td>0.611</td></tr><tr><td>Ours</td><td>0.781</td><td>0.664</td><td>0.763</td><td>0.640</td><td>0.753</td><td>0.635</td></tr></table>

Because our implementation builds on the CoDynTrust codebase, we report both published reference numbers and reproduced in-framework integrations. The reproduced rows provide the closest implementation-controlled comparison, while the published rows indicate the expected performance of the original methods. A CoBEVFlow ROI-mask diagnostic is provided in Supp. E; Table 1 keeps the released code path for reproducibility. We mark the best result in bold and the second-best result with underlining for each AP metric and scenario.

Table 1 shows a robustness trade-of rather than a broad synchronized-SOTA result. In the synchronized setting, published CoDynTrust and W2C+SyncNet remain stronger at AP@0.7, and reproduced CoDynTrust is strongest at AP@0.5. Under irregular delay, however, our method becomes increasingly competitive: it is near the best AP@0.7 at 300 ms and achieves the best AP@0.5 at 500 ms, while remaining within a small margin of the published CoDynTrust AP@0.7 reference. In particular, LRCP drops from 0.669 to 0.611 AP@0.7 from synchronized input to irregular 500 ms delay (−0.058), whereas our method drops from 0.664 to 0.635 (−0.029). This supports the intended claim that the residual temporal interface preserves localization under severe asynchrony, rather than maximizing synchronized peak accuracy. Additional mixed synchronized/asynchronous finetuning experiments in the Supp. F show that the small synchronized gap is not intrinsic to the architecture.

Communication footprint. We estimate the transmitted BEV feature payload under DAIR-V2X irregular 300 ms delay. All methods in Table 2 except LRCP use Where2Commstyle spatial communication masking, whereas LRCP transmits dense BEV features. For sparse methods, the payload is computed as $B = r H W D \cdot 4$ bytes FP32 features, where r is the logged communication rate [8]. The

Table 2: Communication footprint under DAIR-V2X irregular 300 ms delay. Payload is estimated per collaborator BEV feature packet.
<table><tr><td>Method</td><td>AP@0.7 Comm. Rate Payload</td><td>(r)</td><td>(MB)</td></tr><tr><td>Where2Comm+SyncNet (rep.)</td><td>0.606</td><td>0.1225</td><td>0.79</td></tr><tr><td>CoBEVFlow (rep.)</td><td>0.597</td><td>0.1441</td><td>0.93</td></tr><tr><td>CoDynTrust (rep.)</td><td>0.586</td><td>0.2278</td><td>1.47</td></tr><tr><td>LRCP</td><td>0.639</td><td>1.0000</td><td>6.45</td></tr><tr><td>Ours</td><td>0.640</td><td>0.1572</td><td>1.01</td></tr></table>

communicated scale feature has shape $D \times H \times W = 6 4 \times 1 0 0 \times 2 5 2 .$ Thus, dense LRCP uses 6.45 MB per collaborator packet, while ours uses 1.01 MB with comparable AP@0.7. The estimate counts feature tensors only.

## 4.4 OPV2V Culver City Results

Table 3 reports the OPV2V Culver City benchmark under synchronized input and irregular-delay settings. Unlike DAIR-V2X, the reproduced OPV2V baselines do not show a deficit relative to the original CoDynTrust-paper reference numbers; for example, the original paper reports 0.792 AP@0.5 for Where2Comm, Where2Comm+SyncNet, and CoDynTrust, while our reproduced synchronized baselines are comparable or higher. Thus, the OPV2V comparison is not driven

Table 3: OPV2V Culver City comparison. AP@0.5/AP@0.7 under synchronized and irregular-delay settings.
<table><tr><td>Method</td><td>Sync</td><td>Irr. 300 ms</td><td>Irr. 500 ms</td></tr><tr><td>Where2Comm [8]</td><td>0.786 0.801 / </td><td>0.664</td><td>0.768 0.639</td></tr><tr><td>Where2Comm+SyncNet [11]</td><td>/0.786 0.811 /</td><td>0.706</td><td>0.789 7 0.691</td></tr><tr><td>CoBEVFlow [22]</td><td>0.783 0.822 /</td><td>0.740</td><td>0.819 /0.732</td></tr><tr><td>CoDynTrust [28]</td><td>0.819 0.857 /</td><td>0.795</td><td>0.853 /0.783</td></tr><tr><td>Ours</td><td>/0.835</td><td>0.884 / 0.818</td><td>0.868 0.799</td></tr></table>

by weakened reproduced baselines. In this setting, our residual temporal model is the strongest method in all reported scenarios. It improves both AP@0.5 and AP@0.7 under both synchronized and irregular 300/500 ms-delay inputs, including over the strongest listed CoDynTrust baseline. This diference is consistent with the benchmark structure: OPV2V provides simulated multi-CAV V2V sequences with cleaner pose/time consistency and richer temporal redundancy, enabling cached history to improve even when fusion is synchronized. DAIR-V2X is a real-world V2I system with only vehicle-infrastructure collaboration and noisier calibration/timing, making the benefit mainly apparent under severe delay, packet drop, and pose noise.

## 4.5 Robustness to Delay and Packet Drop

Table 4 compares our method with LRCP [20] under clean delay and joint delayplus-packet-drop stress. LRCP has higher synchronized and mild-delay AP, while our method preserves AP better as delay becomes severe. From synchronized input to irregular 500 ms delay, LRCP drops by 0.058 AP@0.7, while ours drops by 0.029, reducing the degradation by 50%. The result should therefore be read as a robustness trade-of rather than a broad DAIR-V2X SOTA claim: the residual model is comparable around 300 ms and stronger at 500 ms in both fixed and irregular settings. The packet-drop setting is intentionally severe and is not meant to approximate a calibrated V2X channel; instead, it acts as a worst-case proxy for bursty feature-packet interruptions that can arise under congested or degraded V2X communication. Further discussions and visualizations are provided in Supp. B.

Table 4: DAIR-V2X robustness under delay and packet-drop stress. AP@0.7 across fixed and irregular delay scenarios. Positive ∆ means our method is better. Left: delay-only setting. Right: joint delay and packet-drop stress.  
(a) Delay robustness
<table><tr><td>Scenario</td><td>LRCP Ours Δ</td></tr><tr><td>Sync 0.669 0.664 -0.005</td></tr><tr><td>Fixed 100 ms 0.664 0.654 -0.010</td></tr><tr><td>Fixed 300 ms 0.652 0.640 -0.011</td></tr><tr><td>Fixed 500 ms 0.603 0.635 +0.033</td></tr><tr><td>Irregular 100 ms 0.663 0.654-0.009</td></tr><tr><td>Irregular 300 ms 0.639 0.640+0.001</td></tr><tr><td></td></tr><tr><td>Irregular 500 ms 0.611 0.635+0.023</td></tr></table>

(b) Delay + packet drop
<table><tr><td>Scenario</td><td>LRCP Ours</td><td>Δ</td></tr><tr><td> $\mathrm { S y n c } + \mathrm { d r o p }$ </td><td>0.655 0.652</td><td>-0.003</td></tr><tr><td>Fixed 100 ms + drop</td><td>0.648 0.643</td><td>-0.005</td></tr><tr><td>Fixed 300 ms + drop</td><td>0.628 0.629</td><td>+0.001</td></tr><tr><td>Fixed 500 ms + drop</td><td>0.598 0.622</td><td>2 +0.025</td></tr><tr><td>Irregular 100 ms + drop</td><td>0.650 0.641</td><td>-0.009</td></tr><tr><td>Irregular 300 ms + drop</td><td>0.622</td><td>0.629+0.007</td></tr><tr><td>Irregular 500 ms + drop</td><td>0.604</td><td>0.621 +0.017</td></tr></table>

These results support the intended use case of the residual interface. When communication is synchronized or only mildly delayed, direct alignment can be highly efective. In the presence of larger delays or missing feature regions, however, a conservative residual correction preserves localization more reliably because it does not overwrite the detector-facing feature with an uncertain transported estimate.

## 4.6 Robustness to Pose and Localization Error

Following [22,28], we further stress this setting by perturbing only the inferencetime fusion poses, keeping detector checkpoints fixed. Translation and yaw perturbations are sampled from $\mathcal { N } ( 0 , \sigma _ { t } )$ and $\mathcal { N } ( 0 , \sigma _ { \theta } )$ , with matched $\sigma _ { t } \in \{ 0 . 1 , 0 . 3 , 0 . 5 \}$ m and $\sigma _ { \theta } \in \{ 0 . 1 , 0 . 3 , 0 . 5 \} ^ { \circ }$ in Table 5. The zero-noise case corresponds to the irregular 300 ms setting in Table 1. LRCP remains competitive at mild noise, but its AP@0.7 drops sharply as error increases. Our model better preserves

strict localization at medium and high noise levels, utilizing temporal evidence as a conservative detector-facing correction rather than direct spatial overwrite. Qualitative results are provided in Supp. C.

Table 5: Robustness to pose/localization noise. DAIR-V2X irregular 300 ms delay with inference-time fusion-pose noise. The noise column gives matched $( \sigma _ { t } , \sigma _ { \theta } )$ in meters/degrees. We report AP@0.5/AP@0.7.
<table><tr><td>Noise  $( \sigma _ { t } , \sigma _ { \theta } )$ </td><td>LRCP Ours</td></tr><tr><td>0.1 m  $0 . 1 ^ { \circ }$  0.3m  $0 . 3 ^ { \circ }$ </td><td>0.784 0.630 0.761 0.634 0.722 0.560 0.738 0.602</td></tr><tr><td>0.5m  $/ \ : 0 . 5 ^ { \circ }$ </td><td>0.668 0.534 0.712 0.589</td></tr></table>

## 4.7 Ablation and Diagnostic Findings

Table 6 isolates the contributions of temporal prediction and the detector-facing residual interface. The pose-warped baseline alone achieves 0.622 AP@0.7. Using the residual bottleneck without xLSTM evidence improves this to 0.628, while a conventional xLSTM that directly predicts the current collaborator feature likewise reaches only 0.628. Conversely, retaining the xLSTM while removing the residual bottleneck yields 0.630. The complete Temporal Residual Bottleneck reaches 0.640, indicating that neither generic temporal prediction nor the residual interface alone explains the improvement; their combination is required. Removing the object-focused gate also reduces AP@0.7 to 0.630, supporting its role in restricting temporal corrections to detector-supported regions while preserving the pose-warped baseline elsewhere.

The transport diagnostics test whether explicit motion compensation should be injected into detector-facing features. “Oracle full residual” applies the synchronized residual target $F _ { i } ^ { t } - B _ { i } ^ { t }$ and reaches 0.660 AP@0.7, showing the headroom of an ideal correction. “Past-track teacher transport” uses annotationderived correspondences and reaches 0.645, indicating that accurate object transport can help. In contrast, “Predicted transport applied” uses causal detectorderived correspondences and reaches only 0.631, showing that deployable correspondence estimates are not reliable enough for direct feature overwrite. Denseflow-style and local cross-attention variants showed the same general trend. We therefore retain the conservative residual bottleneck as the final detector-facing interface. We also tested pose/time conditioning variants. Historical matched runs show a small AP@0.7 improvement from adding a pose/time token, but this efect is smaller than those of the xLSTM and residual bottleneck. This diagnostic motivates the final design choice: temporal evidence is useful, but it should enter the detector through a gated residual path rather than through direct feature overwrite. We therefore treat pose/time conditioning as a supporting component rather than the main source of performance.

Table 6: DAIR-V2X ablation and diagnostics. All rows use the Irr. 300 ms delay.
<table><tr><td>Variant</td><td>Deploy. AP@0.5 AP@0.7</td><td></td><td></td></tr><tr><td>Pose-warped baseline only</td><td>Yes</td><td>0.740</td><td>0.622</td></tr><tr><td>Residual bottleneck w/o xLSTM evidence</td><td>Yes</td><td>0.747</td><td>0.628</td></tr><tr><td>Conventional xLSTM feature prediction</td><td>Yes</td><td>0.742</td><td>0.628</td></tr><tr><td>xLSTM w/o residual bottleneck</td><td>Yes</td><td>0.748</td><td>0.630</td></tr><tr><td>Residual bottleneck w/o object gate</td><td>Yes</td><td>0.751</td><td>0.630</td></tr><tr><td>Temporal Residual Bottleneck (full)</td><td>Yes</td><td>0.765</td><td>0.640</td></tr><tr><td>Oracle full residual</td><td>No</td><td>0.776</td><td>0.660</td></tr><tr><td>Past-track teacher transport</td><td>No</td><td>0.768</td><td>0.645</td></tr><tr><td>Predicted transport applied</td><td>Yes</td><td>0.751</td><td>0.631</td></tr></table>

Table 7: Temporal backbone and history ablation. AP@0.7 on DAIR-V2X, k is the historical frame number.
<table><tr><td>Configuration</td><td colspan="3">Sync Irr. 300 ms Irr. 500 ms</td></tr><tr><td>xLSTM-s, k=4</td><td>0.652</td><td>0.635</td><td>0.628</td></tr><tr><td>xLSTM-m, k=4</td><td>0.643</td><td>0.626</td><td>0.618</td></tr><tr><td>xLSTM-sms, k=4</td><td>0.645</td><td>0.625</td><td>0.617</td></tr><tr><td>xLSTM-msm, k=2</td><td>0.642</td><td>0.620</td><td>0.613</td></tr><tr><td>GRU, k=4</td><td>0.649</td><td>0.630</td><td>0.627</td></tr><tr><td>xLSTM-msm, k=4 (Ours) 0.664</td><td></td><td>0.640</td><td>0.635</td></tr></table>

Temporal backbone and history length. Table 7 varies the recurrent backbone, xLSTM block sequence, and history length while keeping the temporal-residual interface fixed. Replacing xLSTM with a three-layer GRU remains efective, reaching 0.630 AP@0.7 at Irr. 300 ms, which shows that the proposed residual formulation is not specific to xLSTM. The final m-s-m xLSTM configuration with four history frames (k = 4) nevertheless performs best across all three evaluated settings. Shorter history and alternative xLSTM block patterns consistently underperform, indicating that both suficient temporal context and the selected mixed-memory configuration contribute to the final result.

## 5 Conclusion

We presented a temporal residual bottleneck for asynchronous collaborative perception under delayed and incomplete communication. The method preserves a deterministic pose-warped collaborator feature as a conservative anchor and injects only gated residual corrections from a ∆t-conditioned xLSTM. This conservative detector-facing interface avoids direct feature overwrite when temporal correspondence is uncertain. Experiments on DAIR-V2X show that the reported model does not maximize peak synchronous AP, but it degrades more gracefully than a flow-guided LRCP baseline under severe fixed delay, irregular delay, and joint packet-drop stress. Results on OPV2V Culver City further show that the same residual formulation can improve both synchronized and delayed performance. The current study still has limitations: the packet-drop benchmark is an aggressive stress test, and the residual gate is not a formal safety guarantee. Future work will study residual norm diagnostics and communication-aware residual selection under more realistic packet-loss processes.

Acknowledgments. The research leading to these results is funded by the German Federal Ministry for Economic Afairs and Energy within the project “NXT GEN AI METHODS – Generative Methoden für Perzeption, Prädiktion und Planung". The authors would like to thank the consortium for the successful cooperation.

## Supplementary Material for Temporal Residual Bottleneck for Robust Asynchronous Collaborative Perception

## A Additional Implementation Details

Table 8 lists the implementation details omitted from the main paper for readability. These settings correspond to the DAIR-V2X and OPV2V temporal residual model reported in the main experiments.

Table 8: Supplementary hyperparameters for scratch training. Detailed training, temporal branch, residual bottleneck, object gate, and packet-drop settings.
<table><tr><td>Component</td><td>Setting</td></tr><tr><td colspan="2">Training Configuration</td></tr><tr><td>Optimizer Schedule</td><td>Adam, learning rate  $2 \times 1 0 ^ { - 3 }$  , weight decay  $1 0 ^ { - 4 } , \epsilon = 1 0 ^ { - 1 0 }$  Cosine decay with 7-epoch warmup (warmup LR  $2 \times 1 0 ^ { - 5 } )$  DAIR-V2X: 60 epochs | OPV2V: 30 epochs DAIR-V2X: 2 | OPV2V: 1</td></tr><tr><td>Batch size Async training</td><td>Random binomial delay with  $p \in [ 0 . 0 5 , 0 . 3 0 ]$  Architecture &amp; Temporal Branch Settings</td></tr><tr><td colspan="2">History</td></tr><tr><td>xLSTM branch</td><td>Four history sweeps, so xLSTM context length 4 Embedding dimension 64, m-s-m block order, 4 mLSTM heads,</td></tr><tr><td></td><td>dropout 0 Pose/time token hidden dim 128, 4 Fourier bands, scale 0.1, zero-</td></tr><tr><td>Conditioning</td><td>init head</td></tr><tr><td></td><td>Residual bottleneck 64-ch bottleneck, 8 conf ch, hidden dim 256, kernel 3, scale 1.5, zero-init output Object-focused gate Stale-and-warped support union, dilation 4, floor 0.05, detached</td></tr><tr><td colspan="2">gradients Objective &amp; Stress Test Settings</td></tr><tr><td>Detector objective No feature recon loss; applied-res weight</td><td> $\lambda _ { \mathrm { a p p } } = 0 . 0 2 ; w _ { \mathrm { d i r } } = 1 . 0 .$   $w _ { \mathrm { m a g } } \ = \ 0 . 2 5 ,$  wimp = 0.25, wover = 0.1; improvement margin  $m = 5 \times 1 0 ^ { - 5 }$  , over-application ratio ρ = 1.5; FG/BG weight</td></tr><tr><td>Packet-drop test</td><td>3.0/1.0, label dilation 2 Inference only; non-ego burst drop p = 1.0, tile  $8 \times 8 ,$  burst len 12 (ego safe)</td></tr></table>

## B Contextualizing Packet Drop as Severe V2X Degradation

In Sec. 4.5 of the main paper, we evaluate the Temporal Residual Bottleneck under a joint delay-plus-packet-drop stress test. This setting is not intended to be a calibrated wireless-channel simulator. Instead, it serves as a worst-case proxy for feature-packet interruptions under lossy or impaired V2X communication, a setting studied in recent collaborative perception work [12, 29].

![](images/f9f859e36cfcf8d8f73b73ceec416c293e047b943306fe6d00440a1c34f849f9.jpg)  
Fig. 3: Burst packet-drop visualization. Top row: First scenario. Bottom row: Second scenario. Left: BEV detection result. Right: detector confidence map with burstdropped BEV cells highlighted in gray.

In practical feature-level collaborative perception, BEV tensors must be serialized, packetized, transmitted, and reconstructed at the receiver. When packets are delayed, dropped, or arrive incompletely due to channel degradation, congestion, queueing efects, or congestion-control behavior [15, 18], the reconstructed collaborator feature can contain missing or stale feature evidence. Our burst-drop setting approximates this failure mode by removing contiguous BEV regions from non-ego collaborator features at inference time.

This stress test therefore evaluates whether the fusion model can preserve detection performance when delayed collaborator evidence is also spatially incomplete. The results in Sec. 4.5 show that the proposed residual interface degrades more gracefully than LRCP [20] under this joint delay-and-drop setting, supporting the use of conservative residual correction when temporal history is fragmented or unreliable.

Figure 3 visualizes the packet-drop stress used in our robustness experiments. The left panel shows the standard BEV detection output for a synchronized sample. The right panel shows the detector confidence map for the same sample when burst packet drop is applied to the non-ego collaborator feature. Gray regions indicate contiguous BEV cells removed before fusion. This setting is intended as a severe missing-evidence stress test rather than a calibrated wirelesschannel model.

## C Qualitative Examples of Detections under Delay and Pose Noise

Figure 4 shows representative DAIR-V2X qualitative examples comparing our method with LRCP. The top row shows an irregular 500 ms delay case, and the bottom row shows irregular 500 ms delay with additional pose noise (0.3 m / 0.3<sup>◦</sup>). These examples are not intended to show failure-free detection. Both methods can produce false positives under severe temporal misalignment. The main qualitative diference is the structure of the errors: LRCP often introduces duplicated or spatially shifted boxes in dense vehicle regions, consistent with unreliable direct transport under large delay, while our residual temporal fusion tends to keep more detections anchored near the visible object support. This behavior aligns with the quantitative AP@0.7 trend, where our method preserves localization better under severe delay and perturbation despite remaining imperfect.

![](images/16a34b376d9a91a8d19a9917b871a2610911cf2afba12657603cd6c1aa3d1192.jpg)  
Fig. 4: Qualitative BEV detections under delay and pose noise. Left: ours. Right: LRCP. Top: irregular 500 ms delay. Bottom: irregular 500 ms delay with pose noise (0.3 m / 0.3<sup>◦</sup>).

## D Runtime Analysis

Figure 5 reports inference runtime per sample measured on an RTX 3090. It is measured during inference with batch size 1, identical BEV resolution, the same number of collaborating agents, 50 warm-up iterations, 200 measured iterations, and CUDA synchronization before and after each measurement. The lightweight xLSTM-m variant is substantially faster than flow-guided baselines, while the final xLSTM-m-s-m configuration has runtime comparable to LRCP and much lower than CoDynTrust. This indicates that the residual temporal interface adds moderate recurrent overhead but avoids the high cost of trust-filtered dense compensation.

![](images/7a5a0f43883ae0be122778a96172d864c74b9a8b128081324c35f03c1fd1020f.jpg)  
Fig. 5: Latency comparison. Runtime measured on RTX 3090.

## E CoBEVFlow reproduction diagnostic.

A similar degradation trend for CoBEVFlow on DAIR-V2X is also reported by LRCP [20], suggesting that dense flow-based compensation is sensitive to correspondence and masking choices in real-world V2I scenes. We further audited the reproduced CoBEVFlow integration because its in-framework result remains lower than the published reference number. The released CoBEVFlow code path applies the object ROI as a hard multiplicative mask after BEV-flow warping, which suppresses all non-ROI background features. This is a potentially important implementation detail because BEV fusion still relies on contextual and static-scene evidence outside object boxes. To isolate this efect, we evaluated four inference-time variants at the standard DAIR-V2X $p = 0 . 3$ setting: the released hard-mask path, removing the ROI mask, disabling the warp with no mask, and a background-preserving ROI blend that applies the warped feature inside the ROI while retaining the original feature elsewhere.

Table 9: CoBEVFlow ROI-mask diagnostic. DAIR-V2X AP at p = 0.3 for reproduced CoBEVFlow. The background-preserving blend matches the method description more closely than the released hard-mask multiplication.
<table><tr><td>Variant</td><td>AP@0.3</td><td>AP@0.5</td><td>AP@0.7</td></tr><tr><td>Released hard ROI mask</td><td>0.810</td><td>0.739</td><td>0.597</td></tr><tr><td>No ROI mask</td><td>0.822</td><td>0.760</td><td>0.611</td></tr><tr><td>Identity warp, no mask</td><td>0.822</td><td>0.760</td><td>0.611</td></tr><tr><td>Background-preserving ROI blend</td><td>0.822</td><td>0.760</td><td>0.611</td></tr></table>

Table 9 shows that the hard ROI mask is the main source of degradation in our reproduction. Replacing it with the background-preserving blend improves AP@0.7 from 0.597 to 0.611, matching the no-mask and identity-warp controls. However, the flow warp itself does not improve over identity in this setting. We therefore keep the released hard-mask path as the reproduced CoBEVFlow baseline in Table 1 in main paper, and use the remaining rows only as diagnostics explaining why this reproduced baseline is lower than the published reference.

## F Synchronized/asynchronous training trade-of

To test whether robustness under severe delay inherently requires sacrificing synchronized accuracy, we additionally fine-tune the reported checkpoint at low learning rate using 10% synchronized and 90% causal-asynchronous samples. The resulting checkpoint reaches 0.669/0.648/0.641 AP@0.7 under Sync/Irr. 300 ms/Irr. 500 ms, compared with 0.664/0.640/0.635 for the reported model at Table 1 in the main paper. It therefore matches LRCP under synchronized input while preserving the delayed advantage, indicating that the synchronized gap of the reported checkpoint is primarily a training-distribution trade-of rather than an intrinsic limitation of the residual architecture.

## References

1. Ahmed, A.N., Mercelis, S., Anwar, A.: Delawarecol: Delay aware collaborative perception. IEEE Open Journal of Vehicular Technology 6, 1164–1177 (2025)

2. Arnold, E., Dianati, M., De Temple, R., Fallah, S.: Cooperative perception for 3d object detection in driving scenarios using infrastructure sensors. IEEE Transactions on Intelligent Transportation Systems 23(3), 1852–1864 (2020)

3. Beck, M., Pöppel, K., Spanring, M., Auer, A., Prudnikova, O., Kopp, M., Klambauer, G., Brandstetter, J., Hochreiter, S.: xlstm: Extended long short-term memory. In: Globerson, A., Mackey, L., Belgrave, D., Fan, A., Paquet, U., Tomczak, J., Zhang, C. (eds.) Advances in Neural Information Processing Systems. vol. 37, pp. 107547–107603. Curran Associates, Inc. (2024)

4. Chen, G., Zhang, C., Tang, T., Lv, P., Li, F., Xie, X.: Catnet: Collaborative alignment and transformation network for cooperative perception. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18724– 18733 (2026)

5. Chen, G., Zhang, C., Zhao, X.: Whispernet: A scalable solution for bandwidtheficient collaboration. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 32154–32163 (2026)

6. Chen, Q., Ma, X., Tang, S., Guo, J., Yang, Q., Fu, S.: F-cooper: Feature based cooperative perception for autonomous vehicle edge computing system using 3d point clouds. In: Proceedings of the 4th ACM/IEEE Symposium on Edge Computing. pp. 88–100 (2019)

7. Han, Y., Zhang, H., Li, H., Jin, Y., Lang, C., Li, Y.: Collaborative perception in autonomous driving: Methods, datasets, and challenges. IEEE Intelligent Transportation Systems Magazine 15(6), 131–151 (2023)

8. Hu, Y., Fang, S., Lei, Z., Zhong, Y., Chen, S.: Where2comm: Communicationeficient collaborative perception via spatial confidence maps. In: Advances in Neural Information Processing Systems (2022)

9. Justo, A., Araluce, J., Rodriguez-Arozamena, M., Gonzalez, L., Bergasa, L.M.: Lfv2v: A late fusion cooperative framework in v2v scenarios. In: 2025 IEEE Intelligent Vehicles Symposium (IV). pp. 2006–2012. IEEE (2025)

10. Lang, A.H., Vora, S., Caesar, H., Zhou, L., Yang, J., Beijbom, O.: Pointpillars: Fast encoders for object detection from point clouds (2019), https://arxiv.org/ abs/1812.05784

11. Lei, Z., Ren, S., Hu, Y., Zhang, W., Chen, S.: Latency-aware collaborative perception. In: European Conference on Computer Vision. pp. 316–332. Springer (2022)

12. Li, J., Xu, R., Liu, X., Ma, J., Chi, Z., Ma, J., Yu, H.: Learning for vehicle-tovehicle cooperative perception under lossy communication. IEEE Transactions on Intelligent Vehicles 8(4), 2650–2660 (2023)

13. Liu, S., Gao, C., Chen, Y., Peng, X., Kong, X., Wang, K., Xu, R., Jiang, W., Xiang, H., Ma, J., Wang, M.: Towards vehicle-to-everything autonomous driving: A survey on collaborative perception (2023)

14. Malik, S., Khan, M.J., Khan, M.A., El-Sayed, H.: Collaborative perception—the missing piece in realizing fully autonomous driving. Sensors 23(18), 7854 (2023)

15. McCarthy, B., O’Driscoll, A., Malik, A.: Congestion control in the cellular-v2x sidelink. IEEE Communications Magazine 59(9), 90–96 (2021)

16. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis. In: European Conference on Computer Vision (2020)

17. Tancik, M., Srinivasan, P.P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., Ramamoorthi, R., Barron, J.T., Ng, R.: Fourier features let networks learn high frequency functions in low dimensional domains. In: Advances in Neural Information Processing Systems. vol. 33 (2020)

18. Thandavasamy, G., Sepulcre, M., Gozalvez, J.: Cooperative perception for connected and automated vehicles: Evaluation and impact of congestion control. IEEE Communications Magazine 58(12), 22–28 (2020)

19. Wan, L., Zhao, J., Wiedholz, A., Bied, M., De Lucena, M.M., Jagtap, A.D., Festag, A., Fröhlich, A.A., Keen, H.E., Vinel, A.: A systematic literature review on vehicular collaborative perception—a computer vision perspective. IEEE Transactions on Intelligent Transportation Systems (2025)

20. Wang, J., Nordström, T.: Latency robust cooperative perception using asynchronous feature fusion. In: 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). pp. 1–10. IEEE (2025)

21. Wei, Q., Dai, P., Li, W., Liu, B., Wu, X.: Infocom: Kilobyte-scale communicationeficient collaborative perception with information bottleneck. arXiv preprint arXiv:2512.10305 (2025)

22. Wei, S., Wei, Y., Hu, Y., Lu, Y., Zhong, Y., Chen, S., Zhang, Y.: Asynchronyrobust collaborative perception via bird’s eye view flow. In: Advances in Neural Information Processing Systems (2023)

23. Xiang, H., Zheng, Z., Xia, X., Zhao, S.Z., Gao, L., Zhou, Z., Cai, T., Zhang, Y., Ma, J.: V2x-realo: An open online framework and dataset for cooperative perception in reality. arXiv preprint arXiv:2503.10034 (2025)

24. Xu, J., Zhang, Y., Cai, Z., Huang, D.: Cosdh: communication-eficient collaborative perception via supply-demand awareness and intermediate-late hybridization. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 6834–6843 (2025)

25. Xu, R., Guo, Y., Han, X., Xia, X., Xiang, H., Ma, J.: Opencda: An open cooperative driving automation framework integrated with co-simulation. In: 2021 IEEE International Intelligent Transportation Systems Conference (ITSC). pp. 1155–1162 (2021)

26. Xu, R., Tu, Z., Xiang, H., Shao, W., Zhou, B., Ma, J.: Cobevt: Cooperative bird’s eye view semantic segmentation with sparse transformers. arXiv preprint arXiv:2207.02202 (2022)

27. Xu, R., Xiang, H., Xia, X., Han, X., Li, J., Ma, J.: Opv2v: An open benchmark dataset and fusion pipeline for perception with vehicle-to-vehicle communication. In: 2022 IEEE International Conference on Robotics and Automation (ICRA) (2022)

28. Xu, Y., Li, L., Wang, J., Yang, B., Wu, Z., Chen, X., Wang, J.: Codyntrust: Robust asynchronous collaborative perception via dynamic feature trust modulus. In: 2025 IEEE International Conference on Robotics and Automation (ICRA). pp. 336–342 (2025)

29. Yan, F., Tao, B., Zheng, N., Nie, L., Li, Q., Yin, Z.: Multi-task collaborative perception for vehicle-to-everything considering impaired communication. IEEE Transactions on Instrumentation and Measurement (2025)

30. Yazgan, M., Akkanapragada, M.V., Zöllner, J.M.: Collaborative perception datasets in autonomous driving: A survey. In: 2024 IEEE Intelligent Vehicles Symposium (IV). pp. 2269–2276 (2024)

31. Yazgan, M., Graf, T., Liu, M., Fleck, T., Zöllner, J.M.: A survey on intermediate fusion methods for collaborative perception categorized by real world challenges. In: 2024 IEEE Intelligent Vehicles Symposium (IV). pp. 2226–2233 (2024)

32. Yazgan, M., Hamdard, I., Wu, Q., Pavlitska, S., Zöllner, J.M.: 4-d radar meets lidar and camera: Cooperative perception under adverse weather. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 805–814 (2026)

33. Yazgan, M., Müller, T., Zöllner, J.M.: Dense coverage, sparse refinement: Byteconstrained cooperative perception. arXiv preprint arXiv:2609.29456 (2026)

34. Yazgan, M., Wu, Q., Hamdard, I., Li, S., Zoellner, J.M.: Slimcomm: Doppler-guided sparse queries for bandwidth-eficient cooperative 3-d perception. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops. pp. 1803–1812 (2025)

35. Yu, H., Luo, Y., Shu, M., Huo, Y., Yang, Z., Shi, Y., Guo, Z., Li, H., Hu, X., Yuan, J., Nie, Z.: Dair-v2x: A large-scale dataset for vehicle-infrastructure cooperative 3d object detection. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2022)

36. Yu, H., Tang, Y., Xie, E., Mao, J., Luo, P., Nie, Z.: Flow-based feature fusion for vehicle-infrastructure cooperative 3d object detection. In: Advances in Neural Information Processing Systems (2023)

37. Zhang, J., Long, F., Li, M., Zhou, X.: E2e-v2x-cp: An eficient cooperative perception method for end-to-end autonomous driving. In: 2025 IEEE International Conference on Smart Internet of Things (SmartIoT). pp. 263–270. IEEE (2025)

38. Zhao, S.Z., Zhang, H., Li, Z., Peng, J., Chui, A., Zhou, Z., Meng, Z., Xiang, H., Huang, Z., Wang, F., et al.: Quantv2x: A fully quantized multi-agent system for cooperative perception. arXiv preprint arXiv:2509.03704 (2025)

39. Zhou, J., Dai, P., Wei, Q., Liu, B., Wu, X., Wang, J.: Pragmatic heterogeneous collaborative perception via generative communication mechanism. Advances in Neural Information Processing Systems 38, 54484–54509 (2026)