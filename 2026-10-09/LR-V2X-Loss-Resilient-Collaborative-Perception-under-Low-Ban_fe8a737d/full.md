# LR-V2X: Loss-Resilient Collaborative Perception under Low-Bandwidth Communication

Kang Yang<sup>1</sup> Tianci Bu<sup>2</sup> Peng Wang<sup>1</sup> Deying Li<sup>1</sup> Yongcai Wang<sup>1,3∗</sup>

<sup>1</sup>Renmin University of China <sup>2</sup>The Hong Kong University of Science and Technology (HKUST) <sup>3</sup>Hebei Key Laboratory of Real-virtual Integrated Autonomous Systems (RIAS) y1127238112@gmail.com, btc010001@gmail.com {peng.wang,deyingli,ycw}@ruc.edu.cn

## Abstract

Given the inherent unpredictability of packet loss in vehicular wireless communications, V2X collaborative perception can yield practical benefits only if agents can achieve reliable collaboration under lossy and low-bandwidth communication conditions. Existing dense BEV feature fusion methods depend on redundant BEV feature exchange, which is infeasible in low-bandwidth scenarios, while compact-communication methods aggressively compress messages but can hardly recover the missing feature content after packet loss. In this paper, we present LR-V2X, a loss-resilient, latent-space reconstruction framework that converts corrupted received latents (even under severe 90% packet loss) into a spatial prior and then reconstructs the missing BEV information from this informative prior and using ego context as condition. Notably, the model can be trained under complete communication conditions and can be directly applied to lossy conditions at test time, eliminating the need for training under numerous lossy conditions. Experiments on DAIR-V2X and V2XREAL show that LR-V2X delivers the strongest robustness under severe packet loss and preserves reliable collaboration as communication quality degrades. And it reduces communication overhead by 64× compared to dense BEV feature fusion baselines. Code will be released at https://github.com/sidiangongyuan/LR-V2X.

## 1 Introduction

V2X collaborative perception extends the sensing coverage of the ego vehicle by sharing intermediate representations among multiple agents, thereby enhancing driving safety in complex traffic scenarios [42, 47, 43, 39]. At present, mainstream intermediate fusion paradigms predominantly transmit BEV feature maps, enabling the ego agent to fuse multi-agent contextual information for accurate 3D object detection [1, 2, 36, 42, 41]. However, a critical practical bottleneck lies in maintaining reliable collaborative perception under real-world wireless communication constraints. The bandwidth limits force each agent to communicate compact features, while packet loss may randomly drop spatial portions of those transmitted features during delivery [29, 8, 14]. Therefore, under low-bandwidth conditions, the ego agent often receives compact yet incomplete information from collaborative agents. Reliable cooperation thus hinges on effectively recovering meaningful BEV contextual features from such corrupted transmissions. Fig. 1 illustrates this failure scenario and our reconstruction-first paradigm.

![](images/0765c1e89a5a588be1ab5bc6279c8f7d3a6b21f6a2bddddefa089877d4e311cb.jpg)  
Figure 1: Motivation and overview. (a) Baseline fusion receives spatially incomplete feature maps when spatial packets are lost. (b) Codebook-based methods transmit compact codes, but missing indices leave spatial holes that degrade detection. (c) LR-V2X reconstructs complete BEV features from corrupted received latents via prior-initialized feature reconstruction, enabling graceful degradation under packet loss.

Existing methods have only addressed this deployment bottleneck partially. Dense BEV feature fusion baselines [42, 41, 24, 40] can preserve accuracy by transmitting highly redundant BEV features, but this robustness is tied to heavy communication overhead. Communication-efficient methods [11, 13, 51, 3, 6] reduce traffic through region selection, compact codes, or learned compression, yet once packets are lost, they generally lack an explicit mechanism to recover the missing feature content. The central challenge is therefore not only how to communicate less, but how to maintain reliable collaboration if the limited transmitted information is further corrupted during transmission.

This problem is difficult because packet loss does not merely remove local evidence. It also induces an input distribution shift absent under ideal communication. When a received feature map becomes spatially sparse, its statistics deviate substantially from those observed during training with complete messages. As shown in Fig. 2, severe packet loss moves the packet-loss regime away from the no-packet-loss reference regime, so the fusion module must operate on inputs that are systematically mismatched to its training distribution. A dense BEV feature fusion baseline such as F-Cooper can partially buffer this shift through communication redundancy, but in the low-bandwidth regime reliable collaboration requires explicit recovery of missing regions rather than reliance on redundancy alone. This motivates a reconstruction-centric design for reliable low-bandwidth collaboration under packet loss [44, 5].

To reconstruct a neighbor’s BEV feature from a received, packet-corrupted compact latent, we propose two key components for loss-resilient V2X: (1) inspired by the capability of overlapping transposed convolutions [17, 26], which can propagate evidence from received locations into missing spatial regions, we design a Latent Prior Decoder (LPD). LPD extrapolates features from the successfully received locations into a spatially complete prior BEV map. This prior BEV map guides the following noise-conditioned reconstructor to reconstruct from a reasonable prior, instead of from pure noise. (2) a noise-conditioned DiT [28] reconstructor that reconstructs from the prior BEV map and is guided by ego-side context. The key insight is that the ego-side observation provides supplementary information for reconstructing corrupted latent features received from neighbors, since the ego agent generally shares a large common view with the neighbors. Therefore, the noise-conditioned DiT reconstructor refines the prior constructed by LPD, conditioned jointly on the received compact latent and the ego agent’s own BEV to fill the missing regions. Following a diffusion-style training strategy, we perturb the prior with noise at randomly sampled levels, so the denoiser learns to recover from priors of varying quality and generalizes to unseen packet-loss rates at test time.

![](images/24c5832b5e4b3366953243dfacc6274ccd05222edf7ea71d92e939b2a0d5bb7b.jpg)

![](images/ee59f8ea77d8c67942f4d1555c77f2dc2c6e5baf2a443971992290e993818768.jpg)

![](images/e7ea5b2961a92b4e8b5b6faa467eb9f7e13d56d8a07b2b55cb14a25c9a3b5e27.jpg)

![](images/300f018b8aed1f1ed109a257c86a4ca4bc1eb533cf62d51c438e8ee3392add49.jpg)  
F-Cooper

![](images/0deb05936046e02b8163fc905f79ac659c46c5dcced87a34e05739817b30b049.jpg)  
CodeFilling

![](images/e7c322dd59dc8a401d8a40e029a933fe5bb46d657d62cc675c63974f74c4821d.jpg)  
LR-V2X  
Figure 2: Packet-loss-induced distribution shift in BEV features on DAIR-V2X. Panels show F-Cooper, CodeFilling, and LR-V2X from left to right. Shift Dist. measures the Mahalanobis distance between packet-loss and reference centroids, RMS Dist. measures the root-mean-square Mahalanobis distance of packet-loss features, and Outlier Rate counts packet-loss features outside the reference high-tail threshold. Lower values indicate a smaller departure from the no-packet-loss reference regime.

Our contributions are:

• We identify feature-level packet loss as a critical yet underexplored failure mode in collaborative perception and show that sparse receptions induce a distribution shift that existing methods do not explicitly address.

• We propose LR-V2X, a reconstruction framework that recovers missing BEV context from compact, packet-corrupted collaborator latents through prior-guided, ego-conditioned feature reconstruction.

• We show that training only under complete communication, combined with noise-conditioned prior refinement, can handle packet-loss-corrupted inputs at test time without packet-loss-specific training.

• We show that LR-V2X achieves the best overall mAP on both DAIR-V2X and V2XREAL under 90% packet loss while using 64× less bandwidth than dense BEV feature fusion baselines.

## 2 Related Work

## 2.1 Collaborative Perception

Collaborative perception enables agents to share sensory information and overcome individual perceptual limitations. The prevailing paradigm is intermediate fusion, which exchanges BEV feature maps to balance communication efficiency and fusion effectiveness, avoiding the prohibitive bandwidth of raw-data exchange [1, 2] while surpassing late fusion. Early works used graph neural network (GNN)-based message passing [36]; AttnFuse [42] introduced attention-based inter-agent modeling, later scaled to global context via V2X-ViT [41]. DiscoNet [21] explored knowledge distillation as a complementary enrichment strategy.

Robustness to real-world deployment uncertainties has been studied along several axes: pose errors [34, 23, 15, 48, 33], communication latency [19, 20, 32, 37], and heterogeneous agent configurations [40, 12, 46, 35, 50, 24, 38, 7]. These efforts address localization imprecision, asynchrony, and cross-agent incompatibilities, which are orthogonal to the transmission reliability problem we study.

Dense BEV feature fusion baselines also pose bandwidth challenges. Where2comm [11] learns to select salient regions, while instance-level methods further reduce bandwidth via object-centric representations [3, 6, 10, 45]. CodeFilling [13] frames collaboration as compression-reconstruction, extended by QuantV2X [51] with quantized representations. More recently, generative approaches such as DiffCP [27] and CoDiff [16] utilize diffusion models to restore compact features, but assume reliable delivery and focus on reconstructing from designed compression artifacts. In contrast, our work addresses a distinct and complementary problem: reconstructing from stochastic packet loss on already-compact latents, where the missingness pattern is unknown at training time and varies per transmission.

## 2.2 Generative Reconstruction for Collaborative Perception

When transmitted BEV features are aggressively compressed or spatially corrupted by packet loss, recovering complete, detection-ready features on the receiver side becomes the central challenge. Diffusion models offer a principled mechanism for this: from DDPM [9] through efficient samplers such as DDIM [31] and latent-space variants [30], they learn to reverse noise and synthesize structured signals from corrupted inputs. Conditioning mechanisms including ILVR [4], ControlNet [49], and RePaint [25] extend this to reconstruction from partial or spatially incomplete observations, making them well-suited for feature recovery in V2X settings.

These properties have motivated applying generative reconstruction in collaborative perception. CodeFilling [13] and QuantV2X [51] treat collaboration as compression-reconstruction using discrete codebooks. DiffCP [27] uses a diffusion model to reconstruct individual-agent features from ultracompact representations before fusion; CoDiff [16] applies diffusion to denoise fused features degraded by pose estimation errors; GenComm [52] employs a conditional diffusion model to generate features in the ego’s semantic space, bridging domain gaps across heterogeneous agents. However, each of these methods targets a specific, pre-defined form of degradation and assumes reliable feature delivery. Our work instead targets stochastic packet loss on compact latents, recovering missing BEV content without assuming a specific loss pattern.

## 3 Method

## 3.1 Problem Formulation

Scenario. We consider a V2X collaborative perception system with N agents $\begin{array} { r l } { A } & { { } = } \end{array}$ $\left\{ A _ { 0 } , A _ { 1 } , \dotsc , A _ { N - 1 } \right\}$ communicating over a wireless network. Agent $A _ { 0 }$ denotes the ego vehicle. Each agent $A _ { i }$ operates in a local coordinate frame and observes the scene through onboard sensors, producing bird’s-eye-view (BEV) features $X _ { i } \in \mathbb { R } ^ { H \times W \times C }$ . The spatial transformation from agent $A _ { i }$ to agent $A _ { j }$ is represented by an affine matrix $T _ { i  j } \in \mathbb { R } ^ { 2 \times 3 }$ , which stacks a $2 \times 2$ linear part (rotation/scale) and a $\bar { 2 } \times 1$ translation vector.

Communication Graph. Agents communicate over a directed graph $\mathcal { G } = ( \mathcal { A } , \mathcal { P } )$ , where an edge $( i , j ) \in \mathscr { P }$ exists if agent $A _ { i }$ can transmit to agent $A _ { j }$ . Each agent $A _ { i }$ transmits a message $S _ { i }$ derived from its local BEV feature $X _ { i } ,$ subject to a communication budget:

$$
S _ { i } = { \mathcal { C } } ( X _ { i } ) , \qquad \operatorname { s i z e } ( S _ { i } ) \leq B ,\tag{1}
$$

where $\mathcal { C } ( \cdot )$ denotes a generic communication function and $B$ is the maximum bandwidth budget.

Packet Loss Model. During transmission over edge $( i , j )$ , we consider feature-level packet loss at the spatial-packet granularity. Let $M _ { i \to j }$ denote the reception mask defined on the spatial layout of the transmitted message, where $M _ { i \to j } ( u , v ) = 1$ indicates successful reception of the packet at spatial location $( u , v )$ . We assume the following random packet corruption process:

$$
M _ { i \to j } ( u , v ) \sim \operatorname { B e r n o u l l i } ( p ) , \quad \tilde { S } _ { i \to j } = S _ { i } \odot M _ { i \to j } ,\tag{2}
$$

where $p \in [ 0 , 1 ]$ is the retention rate and $\odot$ denotes element-wise multiplication across all channels. We refer to $S _ { i  j }$ as the received message. In the ego-centric setting, we use the shorthand $\tilde { S } _ { i } \equiv \tilde { S } _ { i  0 }$ for $i \neq 0$ , where the subscript 0 denotes the ego agent.

Optimization Objective. For each ego vehicle $A _ { 0 } ,$ , the goal is to perform collaborative 3D object detection from its own BEV feature and the messages received from other agents. The ego uses its own feature directly, while each collaborator feature is reconstructed in the collaborator coordinate frame from the received message, the ego feature, and the relative transform:

$$
\hat { X } _ { 0 } = X _ { 0 } , \qquad \hat { X } _ { i } = { \mathcal { R } } ( \tilde { S } _ { i } , X _ { 0 } , T _ { 0 \to i } ; \theta _ { r } ) , \quad \forall i \neq 0 ,\tag{3}
$$

where $\theta _ { r }$ is the learnable parameter of the reconstruction model $\mathcal { R }$ . Then, before fusion, each reconstructed collaborator feature is warped back to the ego coordinate frame, denoted by ${ \bar { X } } _ { i } =$ $\mathcal { W } ( \hat { X } _ { i } , T _ { i  0 } )$ for $i \neq 0$ and ${ \bar { X } } _ { 0 } = X _ { 0 }$ . These ego-aligned features are fused for ego-centric detection:

$$
\hat { Y } = \mathcal { D } \left( \mathcal { F } \left( \{ \bar { X } _ { i } \} _ { i = 0 } ^ { N - 1 } ; \pmb { \theta } _ { f } \right) ; \pmb { \theta } _ { d } \right) .\tag{4}
$$

Here $\theta _ { f }$ and $\theta _ { d }$ denote the learnable parameters of the multi-agent fusion module $\mathcal { F }$ and the detection head $\begin{array} { r } { \breve { \mathcal { D } } , } \end{array}$ , respectively. The training objective minimizes the detection loss

$$
\mathcal { L } _ { \mathrm { d e t } } ( \hat { Y } , Y ) ,\tag{5}
$$

where Y denotes ground-truth object annotations. We evaluate the robustness of the above collaborative perception process under varying retention rates $p$ and random masks M to reflect stochastic packet loss during transmission.

## 3.2 Framework Overview

![](images/68f5ec8b6ac1f9a53590fc7b790e87e6aaef84f9f08f4157540bbbdb784960f3.jpg)  
Figure 3: Framework of LR-V2X. Each agent encodes BEV features into a compact latent for transmission. The receiver reconstructs the agent’s BEV via the Latent Prior Decoder and a noiseconditioned reconstructor, then fuses with its own features for 3D detection.

The received message is simultaneously (i) compact after compression and (ii) spatially incomplete after packet loss, making direct BEV regression from corrupted compact observations ill-conditioned. LR-V2X resolves this tension through a coarse-to-fine recovery pipeline. First, latent encoding reduces the transmitted BEV feature to a compact spatial carrier, establishing the low-bandwidth operating point. Second, the Latent Prior Decoder (LPD) converts the corrupted received latent into a dense but coarse BEV prior by propagating the remaining spatial evidence into neighboring missing regions. Third, a noise-conditioned reconstructor refines this prior using both the received latent and ego-side context.

We instantiate the communication function as a learned latent encoder:

$$
S _ { i } = \mathcal { E } ( X _ { i } ; \pmb { \theta } _ { e } ) \in \mathbb { R } ^ { h \times w \times C ^ { \prime } } ,\tag{6}
$$

where $\mathcal { E } ( \cdot ; \pmb { \theta } _ { e } )$ maps the collaborator BEV feature to a compact spatial latent with $h \ll H , w \ll W$ and $C ^ { \prime } \leq C$ . Under packet loss, the receiver observes $\tilde { S } _ { i } = S _ { i } \odot M _ { i  0 }$ , a spatially incomplete version of this latent. The following subsections detail how the LPD converts $\tilde { S } _ { i }$ into a dense prior and how the noise-conditioned reconstructor refines it into ${ \hat { X } } _ { i }$

## 3.3 Latent Prior Decoder

The main purpose of the LPD is to make reconstruction well-posed before the transformer denoiser is applied. Packet loss turns the compact latent into a spatially incomplete observation whose missing locations provide no direct evidence, while a transformer operating directly on this corrupted compact input would have to infer both global structure and local completion from a weak initialization. The LPD therefore serves as a deterministic prior generator that converts the received latent $\tilde { S } _ { i }$ into a spatially complete but coarse BEV map $X _ { i } ^ { \mathrm { p r i o r } }$ and anchors the following reconstruction stage.

Concretely, the LPD maps ${ \tilde { S } } _ { i }$ to $X _ { i } ^ { \mathrm { p r i o r } }$ via four transposed convolutional layers [22] with overlapping receptive fields, upsampling from latent resolution (h, w) to BEV resolution (H, W) (Fig. 4):

$$
X _ { i } ^ { \mathrm { p r i o r } } = \mathrm { L P D } ( \tilde { S } _ { i } ) .\tag{7}
$$

The overlapping kernels spread activations from received locations into neighboring missing regions, converting a spatially incomplete latent into a spatially complete prior BEV map [17, 26]. Although the LPD is trained on complete latents $( \tilde { S } _ { i } = S _ { i } )$ , the overlapping transposed convolutions still aggregate the available neighboring evidence when parts of the latent are missing, yielding a usable prior BEV map across arbitrary packet-loss rates.

![](images/0713681367123b5d05854b0a6dc6e3670ab92aa844facaa807a2e9645c0dabb7.jpg)  
Figure 4: Architecture of the Latent Prior Decoder (LPD) and the DiT denoiser used for conditional feature reconstruction.

## 3.4 Noise-Conditioned Feature Reconstruction

The noise-conditioned reconstructor is designed to refine an imperfect prior rather than generate BEV features from scratch. Its key design choice is to perturb the LPD prior itself during training. Unlike standard diffusion models, which add noise to the clean target and then learn an iterative generation process, our reconstruction process starts from the LPD prior because inference also starts from a packet-loss-degraded prior. This makes the learned denoising task match the actual test-time recovery problem and supports single-step feature reconstruction.

Given $\tilde { S } _ { i }$ , the LPD first produces the prior BEV map $X _ { i } ^ { \mathrm { p r i o r } }$ . Gaussian noise is then injected at a randomly sampled timestep t, yielding the noisy prior $X _ { i } ^ { ( t ) }$

$$
X _ { i } ^ { ( t ) } = \sqrt { \bar { \alpha } _ { t } } X _ { i } ^ { \mathrm { p r i o r } } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon ,\tag{8}
$$

where $\epsilon \sim \mathcal { N } ( 0 , I )$ and $\bar { \alpha } _ { t }$ is the cumulative noise schedule [9]. A conditional denoiser $f _ { \pmb { \theta } _ { \tau } }$ then recovers the clean BEV feature ${ \hat { X } } _ { i }$ from the noisy prior, conditioned on the received latent $\tilde { S } _ { i }$ , the per-collaborator aligned ego BEV $X _ { 0 } ^ { \mathrm { a l i g n e d } } = \mathcal { W } ( X _ { 0 } , T _ { 0  i } )$ (warped from $X _ { 0 }$ to agent i’s coordinate frame via bilinear interpolation), and the spatial transform $T _  0  i \}$

$$
\begin{array} { r } { \hat { X } _ { i } = f _ { \pmb { \theta } _ { r } } ( X _ { i } ^ { ( t ) } , t , \tilde { S } _ { i } , X _ { 0 } ^ { \mathrm { a l i g n e d } } , T _ { 0  i } ) . } \end{array}\tag{9}
$$

We train with an $x _ { 0 } \mathrm { { \cdot } }$ -prediction objective:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { d i f f } } = \mathbb { E } _ { t , \epsilon } [ \| f _ { \pmb { \theta } _ { r } } ( X _ { i } ^ { ( t ) } , t , \tilde { S } _ { i } , X _ { 0 } ^ { \mathrm { a l i g n e d } } , T _ { 0  i } ) - X _ { i } \| ^ { 2 } ] . } \end{array}\tag{10}
$$

## 3.4.1 Noise-Conditioned Reconstructor Architecture

We adopt DiT [28] as the denoiser backbone. The received latent $\tilde { S } _ { i }$ constrains reconstruction in two complementary ways: it provides a global conditioning signal that steers the denoising trajectory toward the transmitted collaborator content, and it also serves as spatial evidence during crossattention. The aligned ego BEV $X _ { 0 } ^ { \mathrm { a l i g n e d } }$ is introduced as additional cross-attention context, allowing the model to combine the transmitted message with ego-side scene structure when recovering missing regions. For layer ℓ, the DiT denoiser block performs (Fig. 4)

$$
\begin{array} { r } { h _ { \ell } = \mathrm { S e l f A t t n } ( \mathrm { A d a L N } ( x _ { \ell } , c ) ) + x _ { \ell } \qquad } \\ { h _ { \ell } ^ { \prime } = \mathrm { C r o s s A t t n } ( \mathrm { A d a L N } ( h _ { \ell } , c ) , [ \tilde { S } _ { i } ; \ : X _ { 0 } ^ { \mathrm { a l i g n e d } } ] ) + h _ { \ell } } \\ { x _ { \ell + 1 } = \mathrm { F F N } ( \mathrm { A d a L N } ( h _ { \ell } ^ { \prime } , c ) ) + h _ { \ell } ^ { \prime } . \qquad } \end{array}\tag{11}
$$

The global conditioning vector is

$$
c = \phi _ { t } ( t ) + \phi _ { S } ( \mathrm { p o o l } ( \tilde { S } _ { i } ) ) + \phi _ { X } ( \mathrm { p o o l } ( X _ { 0 } ^ { \mathrm { a l i g n e d } } ) ) + \phi _ { T } ( T _ { 0  i } ) ,\tag{12}
$$

where $\phi _ { t } , \phi _ { S } , \phi _ { X } , \phi _ { T }$ are learned MLP projections, and $\operatorname { A d a L N } ( x , c )$ applies layer normalization with per-channel scale and shift linearly projected from c. Cross-attention allows the model to query both transmitted evidence and ego context when reconstructing missing regions. This combination lets the denoiser preserve global consistency with the received message while restoring local object structure in corrupted regions.

## 3.4.2 Training and Inference

Training follows a stage-wise protocol. We first train the PointPillar encoder, pyramid fusion backbone, and detection heads under complete communication. We then freeze the sensor encoder/backbone and train the latent encoder E, LPD, and DiT reconstructor by minimizing ${ \mathcal { L } } _ { \mathrm { d i f f } }$ (Eq. 10) under complete communication $( \tilde { S } _ { i } = S _ { i } )$ , with t sampled uniformly from $\{ 1 , \ldots , T \}$ at each step. Finally, we fine-tune the reconstruction module together with the fusion and detection heads while keeping the sensor encoder/backbone frozen. This exposes the denoiser to priors of varying quality without requiring packet-loss-corrupted messages during training. At inference, given $\tilde { S } _ { i }$ from agent i $( i \neq 0 )$ , we compute $X _ { i } ^ { \mathrm { p r i o r } } = \mathrm { L P D } ( \tilde { S } _ { i } )$ , draw Gaussian noise, perturb the prior to a fixed noise level $t _ { \mathrm { i n f } }$ (a hyperparameter), and recover ${ \hat { X } } _ { i }$ in a single forward pass of $f _ { \pmb { \theta } _ { r } }$ . The reconstructed collaborator feature is then aligned back to the ego frame inside the fusion module. For end-to-end optimization with the detection task, the total loss combines reconstruction and detection objectives

$$
{ \mathcal { L } } _ { \mathrm { t o t a l } } = \lambda _ { \mathrm { d e t } } { \mathcal { L } } _ { \mathrm { d e t } } ( { \mathcal { D } } ( { \mathcal { F } } ( \{ { \bar { X } } _ { i } \} ) ) , Y ) + \lambda _ { \mathrm { d i f f } } { \mathcal { L } } _ { \mathrm { d i f f } } ,\tag{13}
$$

where $\lambda _ { \mathrm { d e t } } , \lambda _ { \mathrm { d i f f } }$ are the balancing weights.

## 4 Experiments

## 4.1 Setup

We evaluate on two real-world V2X datasets: DAIR-V2X [47] and V2XREAL [39]. All methods use a PointPillars [18] backbone and are trained under complete communication (no packet loss). At test time, we simulate spatial-level packet loss by independently dropping each spatial packet with probability $1 - p .$ , where p is the retention rate. This i.i.d. Bernoulli model provides a controlled testbed for random packet erasures; we additionally evaluate correlated burst loss in Appendix B.1. Packet-loss masks are applied only to transmitted collaborator messages, while the ego feature remains available locally to the receiver. We use fixed random seeds for mask generation and inference noise, so all methods are compared under the same stochastic protocol. We measure dataset-specific AP breakdowns and overall mAP at IoU thresholds 0.3, 0.5, and 0.7, along with per transmitted collaborator frame communication cost and inference speed (frames per second, FPS). Full implementation details, hyperparameters, and training procedures are provided in Appendix A.

## 4.2 Performance

Table 1 reports dataset-specific AP breakdowns and overall mAP at IoU 0.3, 0.5, and 0.7, along with communication cost and inference speed. On V2XREAL (Table 1a), LR-V2X achieves the highest overall mAP across all three IoU thresholds under 90% packet loss (44.87/38.60/24.18), while transmitting only a 128 KB collaborator message (64× less than dense BEV feature fusion baselines). Compared to CodeFilling at the same 128 KB budget, LR-V2X improves mAP@0.5 by 1.91 and mAP@0.7 by 1.02 points, and it also outperforms the protocol-matched GenComm<sup>†</sup> re-implementation. On DAIR-V2X (Table 1b), LR-V2X likewise achieves the highest overall mAP across all three IoU thresholds (69.35/66.90/53.37), surpassing all dense BEV feature fusion baselines that require 64× more communication. Under complete communication, LR-V2X also attains the highest overall mAP among the no-packet-loss references reported here on both datasets.

Table 1: Performance comparison under Complete Communication (0% Packet Loss) reference conditions and the main 90% Packet Loss robustness setting.  
(a) V2XREAL dataset (per-class AP and mAP).
<table><tr><td rowspan="2">Models</td><td colspan="3">AP_vehicle</td><td colspan="3">AP_pedestrian</td><td colspan="3">AP_truck</td><td colspan="3">mAP (%)</td><td rowspan="2">Comm</td><td rowspan="2">FPS</td></tr><tr><td>0.3</td><td>0.5</td><td>0.7</td><td>0.3</td><td>0.5</td><td>0.7</td><td>0.3</td><td>0.5</td><td>0.7</td><td>0.3 0.5</td><td>0.7</td><td></td></tr><tr><td colspan="10">No Packet Loss</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>No Fusion</td><td>32.27</td><td>17.20</td><td>1.31</td><td>36.39</td><td>30.91</td><td>10.79</td><td>55.11</td><td>52.76</td><td>37.14</td><td>41.26</td><td>33.63</td><td>16.41</td><td>0</td><td>2.47</td></tr><tr><td>Early Fusion</td><td>68.17</td><td>64.73</td><td>40.03</td><td>33.53</td><td>17.14</td><td>0.87</td><td>27.22</td><td>25.57</td><td>14.18</td><td>42.97</td><td>35.81</td><td>18.36</td><td>~2.4MB</td><td>1.03</td></tr><tr><td>Late Fusion</td><td>32.54</td><td>16.23</td><td>1.17</td><td>34.12</td><td>30.13</td><td>13.70</td><td>61.55</td><td>58.96</td><td>38.99</td><td>42.73</td><td>35.10</td><td>17.96</td><td>≤5.4KB</td><td>1.21</td></tr><tr><td>LR-V2X</td><td>76.70</td><td>73.85</td><td>49.58</td><td>35.98</td><td>17.40</td><td>1.12</td><td>48.38</td><td>43.60</td><td>32.64</td><td>53.69</td><td>44.95</td><td>27.78</td><td>128KB</td><td>1.54</td></tr><tr><td colspan="10">90% Packet Loss</td><td colspan="3"></td><td></td><td></td></tr><tr><td>Attn</td><td>59.35</td><td>55.93</td><td>39.25</td><td>29.30</td><td>15.20</td><td>1.08</td><td>39.00</td><td>35.50</td><td>20.66</td><td>42.55</td><td>35.54</td><td>20.33</td><td>8MB</td><td>1.24</td></tr><tr><td>F-Cooper</td><td>57.94</td><td>54.22</td><td>38.00</td><td>31.92</td><td>16.31</td><td>1.08</td><td>37.60</td><td>32.40</td><td>12.60</td><td>42.49</td><td>34.30</td><td>17.21</td><td>8MB</td><td>0.82</td></tr><tr><td>CoBEVT</td><td>51.54</td><td>46.45</td><td>30.60</td><td>25.25</td><td>11.35</td><td>0.50</td><td>39.99</td><td>34.98</td><td>15.82</td><td>38.92</td><td>30.93</td><td>15.64</td><td>8MB</td><td>0.82</td></tr><tr><td>V2X-VIT</td><td>49.37</td><td>46.18</td><td>31.40</td><td>34.23</td><td>17.77</td><td>1.38</td><td>38.60</td><td>32.90</td><td>24.00</td><td>40.73</td><td>32.29</td><td>18.93</td><td>8MB</td><td>0.70</td></tr><tr><td>CodeFilling</td><td>62.14</td><td>57.95</td><td>39.36</td><td>16.41</td><td>12.92</td><td>0.60</td><td>43.07</td><td>39.19</td><td>29.52</td><td>40.54</td><td>36.69</td><td>23.16</td><td>128KB</td><td>1.68</td></tr><tr><td>HEAL</td><td>62.31</td><td>60.85</td><td>39.29</td><td>26.53</td><td>12.96</td><td>0.99</td><td>40.36</td><td>39.09</td><td>27.10</td><td>43.07</td><td>37.64</td><td>21.79</td><td>8MB</td><td>1.61</td></tr><tr><td>GenComm†</td><td>63.11</td><td>56.60</td><td>30.38</td><td>12.99</td><td>3.83</td><td>0.15</td><td>33.33</td><td>29.97</td><td>17.48</td><td>36.48</td><td>30.14</td><td>16.00</td><td>128KB</td><td>0.37</td></tr><tr><td>LR-V2X</td><td>64.60</td><td>63.59</td><td>46.31</td><td>28.55</td><td>12.23</td><td>1.09</td><td>41.48</td><td>39.98</td><td>25.12</td><td>44.87</td><td>38.60</td><td>24.18</td><td>128KB</td><td>1.54</td></tr></table>

(b) DAIR-V2X dataset (per-range AP and mAP).

<table><tr><td rowspan="2">Models</td><td colspan="3">Short</td><td colspan="3">Middle</td><td colspan="3">Long</td><td colspan="3">mAP (%)</td><td rowspan="2">Comm</td><td rowspan="2">FPS</td></tr><tr><td>0.3</td><td>0.5</td><td></td><td>0.7 0.3</td><td>0.5</td><td>0.7</td><td>0.3</td><td>0.5</td><td>0.7 0.3</td><td>0.5</td><td>0.7</td><td></td></tr><tr><td colspan="10">No Packet Loss</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>No Fusion</td><td>79.25</td><td>77.24</td><td>70.12</td><td>75.70</td><td>72.49</td><td>62.20</td><td>54.78</td><td>51.06</td><td>37.94</td><td>62.51</td><td>59.33</td><td>49.67</td><td>0</td><td>8.70</td></tr><tr><td>Early Fusion</td><td>84.47</td><td>81.40</td><td>68.88</td><td>78.60</td><td>72.88</td><td>56.54</td><td>59.79</td><td>54.55</td><td>36.13</td><td>73.52</td><td>68.61</td><td>52.37</td><td>~1.0MB</td><td>14.95</td></tr><tr><td>Late Fusion</td><td>79.25</td><td>77.24</td><td>70.12</td><td>75.70</td><td>72.49</td><td>62.20</td><td>54.78</td><td>51.06</td><td>37.94</td><td>69.51</td><td>66.33</td><td>55.67</td><td>≤5.4KB</td><td>7.75</td></tr><tr><td>LR-V2X</td><td>88.48</td><td>85.84</td><td>72.00</td><td>83.56</td><td>78.38</td><td>58.51</td><td>71.56</td><td>66.04</td><td>46.16</td><td>81.03</td><td>76.16</td><td>57.74</td><td>128KB</td><td>6.04</td></tr><tr><td colspan="10">90% Packet Loss</td><td colspan="3"></td><td></td><td></td></tr><tr><td>Attn</td><td>80.53</td><td>79.02</td><td>66.72</td><td>77.47</td><td>71.70</td><td>56.83</td><td>57.23</td><td>51.41</td><td>38.45</td><td>66.87</td><td>66.20</td><td>52.56</td><td>8MB</td><td>6.67</td></tr><tr><td>F-Cooper</td><td>80.43</td><td>77.14</td><td>67.13</td><td>76.96</td><td>71.62</td><td>57.04</td><td>57.15</td><td>51.65</td><td>38.44</td><td>67.89</td><td>65.90</td><td>52.87</td><td>8MB</td><td>5.39</td></tr><tr><td>CoBEVT</td><td>79.98</td><td>76.17</td><td>61.25</td><td>75.15</td><td>69.79</td><td>51.91</td><td>53.46</td><td>47.68</td><td>33.25</td><td>68.83</td><td>63.55</td><td>44.47</td><td>8MB</td><td>4.92</td></tr><tr><td>V2X-VIT</td><td>80.01</td><td>77.06</td><td>66.55</td><td>75.34</td><td>71.04</td><td>56.20</td><td>53.23</td><td>48.04</td><td>34.61</td><td>68.78</td><td>64.33</td><td>50.97</td><td>8MB</td><td>4.70</td></tr><tr><td>CodeFilling</td><td>76.92</td><td>75.24</td><td>66.22</td><td>69.68</td><td>67.03</td><td>55.20</td><td>47.43</td><td>44.80</td><td>35.29</td><td>63.38</td><td>60.93</td><td>50.62</td><td>128KB</td><td>9.17</td></tr><tr><td>HEAL</td><td>76.88</td><td>75.26</td><td>65.77</td><td>70.48</td><td>67.64</td><td>54.60</td><td>49.39</td><td>46.20</td><td>35.69</td><td>64.43</td><td>61.68</td><td>50.50</td><td>8MB</td><td>7.24</td></tr><tr><td>GenComm†</td><td>67.89</td><td>62.56</td><td>37.75</td><td>49.68</td><td>44.51</td><td>22.82</td><td>32.00</td><td>27.81</td><td>13.20</td><td>48.03</td><td>43.16</td><td>23.27</td><td>128KB</td><td>0.49</td></tr><tr><td>LR-V2X</td><td>82.59</td><td>80.93</td><td>69.98</td><td>75.88</td><td>70.03</td><td>57.93</td><td>55.25</td><td>49.86</td><td>36.96</td><td>69.35</td><td>66.90</td><td>53.37</td><td>128KB</td><td>6.04</td></tr></table>

<sup>†</sup> Re-implementation of GenComm’s DDIM reconstruction module under our backbone and training protocol. Communication cost is reported per transmitted collaborator frame before packet loss. Raw-point early fusion uses float32 XYZI payloads estimated from validation files (V2XREAL: ∼2.4 MB; DAIR-V2X: ∼1.0 MB). Dense BEV and compact latent messages use float32 tensor sizes. Late fusion reports a compact 7-DoF box, score, and class payload with the post-NMS cap of 150 boxes, giving ≤5.4 KB.

Fig. 5 shows performance across packet loss rates from 0% to 90%. LR-V2X is not the top method at every point in the sweep, but it remains consistently competitive across IoU thresholds and datasets, ending with the highest overall mAP under 90% packet loss.

Bandwidth–robustness asymmetry. Dense BEV feature fusion baselines transmit 8 MB per collab orator message (64× our budget), so at 90% loss they still retain 819.2 KB of spatial information versus 12.8 KB for a 128 KB latent. This communication redundancy provides an inherent robustness advantage that is not attributable solely to fusion design (see Appendix B.2). Appendix Table 4 controls for this directly: after matching dense BEV feature fusion baselines to 128 KB, all five decrease in mAP@0.5, while LR-V2X at the same budget achieves the best overall mAP (69.35/66.90/53.37). This supports the interpretation that explicit reconstruction, not only raw communication volume, is responsible for the low-bandwidth robustness.

Two complementary robustness controls further support this interpretation. Appendix Table 2 replaces i.i.d. erasures with correlated burst loss; at the same 128 KB budget, LR-V2X still outperforms CodeFilling by +2.47/+1.59/+0.45 mAP at IoU 0.3/0.5/0.7. Together with the budget-matched dense

- No Fusion = Ours -- Codefilling  HEAL  Attn = CoBEVT - V2X-ViT — Fcooper  
![](images/4d1700374e465e39e5114f1c6ec127efcec7a0e73946e5119520f6ec07cd7733.jpg)  
Figure 5: Performance under varying packet loss rates on DAIR-V2X and V2XREAL. Across the sweep, LR-V2X remains consistently competitive as communication quality decreases and reaches the highest overall mAP at the 90% packet-loss endpoint.

BEV feature fusion results, this shows that the low-bandwidth gain is not limited to independent random drops or raw communication volume.

## 4.3 Ablation Study

Conditioning inputs. Appendix Table 5 ablates the reconstructor conditions while keeping the latent-derived prior fixed. Ego-BEV is the decisive condition: adding it raises mAP@0.3 by 11.62 points, whereas the transformation matrix adds 0.79 points on top of ego-BEV and the direct latent branch alone changes mAP@0.3 by only 0.02 points. This is consistent with V2X geometry: the ego BEV often overlaps the collaborator field of view, so after alignment it provides local anchors that help the reconstructor disambiguate missing or corrupted collaborator features.

Latent Prior Decoder. Appendix Fig. 7 compares the full model with two controlled variants: prior-only uses the LPD output directly for fusion and skips the denoising reconstructor, while zero-prior keeps the denoising reconstructor but replaces the LPD initialization with an all-zero BEV prior before the same noise-conditioned inference step. The full model remains consistently stronger, indicating that LPD initialization and denoising refinement are complementary rather than redundant.

Downsampling ratio. Appendix Table 6 shows that moving from ×2 (2 MB) to ×8 (128 KB) reduces mAP@0.3 by only 1.52 points while reducing bandwidth 16×; at ×16 (32 KB), mAP@0.3 drops to 79.58. We therefore use ×8 as the default accuracy–bandwidth tradeoff.

Inference timestep and noise-conditioned training. Appendix Table 7 shows that noise conditioning improves over a protocol-matched no-noise reconstructor by +7.16/+7.22/+4.23 mAP. This no-noise variant still uses the LPD prior and a learned reconstructor, but feeds the prior directly at t = 0 without the forward-noise perturbation. Appendix Fig. 8 further shows broadly stable performance with t<sub>inf</sub> = 700 selected from the sweep.

Taken together, these ablations support a clear division of labor. The compact latent preserves spatially aligned collaborator evidence, the LPD turns surviving packets into a complete but coarse prior, and the noise-conditioned reconstructor refines this prior with ego context instead of treating it as a direct decoding output. This explains why the default configuration remains effective under high packet loss without increasing the 128 KB message budget.

## 5 Conclusion

This work studies V2X collaborative perception under the joint constraint of limited bandwidth and stochastic packet loss. We introduced LR-V2X, a reconstruction-first framework that transmits compact collaborator latents and recovers missing BEV context through an LPD prior and an egoconditioned, noise-conditioned reconstructor. The experiments show that explicit feature recovery from corrupted compact messages provides stronger low-bandwidth robustness than relying on dense communication redundancy or direct compact-code fusion alone. A central implication is that packet-loss robustness can be learned by modeling how to refine imperfect latent priors, even when training uses complete communication rather than enumerating many lossy conditions. Future work can extend this reconstruction-first view to richer wireless channel models.

## References

[1] Eduardo Arnold, Mehrdad Dianati, Robert de Temple, and Saber Fallah. Cooperative perception for 3d object detection in driving scenarios using infrastructure sensors. IEEE Transactions on Intelligent Transportation Systems, 23(3):1852–1864, 2022. doi: 10.1109/TITS.2020.3028424.

[2] Qi Chen, Sihai Tang, Qing Yang, and Song Fu. Cooper: Cooperative perception for connected autonomous vehicles based on 3d point clouds. In 2019 IEEE 39th International Conference on Distributed Computing Systems (ICDCS), pages 514–524. IEEE, 2019.

[3] Ziming Chen, Yifeng Shi, and Jinrang Jia. Transiff: An instance-level feature fusion framework for vehicleinfrastructure cooperative 3d detection with transformers. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 18205–18214, 2023.

[4] Jooyoung Choi, Sungwon Kim, Yonghyun Jeong, Youngjune Gwon, and Sungroh Yoon. Ilvr: Conditioning method for denoising diffusion probabilistic models. arXiv preprint arXiv:2108.02938, 2021.

[5] Gabriela Csurka. Domain adaptation for visual applications: A comprehensive survey. arXiv preprint arXiv:1702.05374, 2017.

[6] Siqi Fan, Haibao Yu, Wenxian Yang, Jirui Yuan, and Zaiqing Nie. Quest: Query stream for practical cooperative perception. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 18436–18442. IEEE, 2024.

[7] Xiangbo Gao, Runsheng Xu, Jiachen Li, Ziran Wang, Zhiwen Fan, and Zhengzhong Tu. Stamp: Scalable task and model-agnostic collaborative perception. arXiv preprint arXiv:2501.18616, 2025.

[8] Yushan Han, Hui Zhang, Huifang Li, Yi Jin, Congyan Lang, and Yidong Li. Collaborative perception in autonomous driving: Methods, datasets, and challenges. IEEE Intelligent Transportation Systems Magazine, 15(6):131–151, 2023.

[9] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

[10] Tianyu Hong, Xiaobo Zhou, Wenkai Hu, Qi Xie, Zhihui Ke, and Tie Qiu. Communication-efficient multi-vehicle collaborative semantic segmentation via sparse 3d gaussian sharing. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 28622–28631, October 2025.

[11] Yue Hu, Shaoheng Fang, Zixing Lei, Yiqi Zhong, and Siheng Chen. Where2comm: Communicationefficient collaborative perception via spatial confidence maps. Advances in neural information processing systems, 35:4874–4886, 2022.

[12] Yue Hu, Yifan Lu, Runsheng Xu, Weidi Xie, Siheng Chen, and Yanfeng Wang. Collaboration helps camera overtake lidar in 3d detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9243–9252, June 2023.

[13] Yue Hu, Juntong Peng, Sifei Liu, Junhao Ge, Si Liu, and Siheng Chen. Communication-efficient collaborative perception via information filling with codebook. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15481–15490, 2024.

[14] Tao Huang, Jianan Liu, Xi Zhou, Dinh C Nguyen, Mostafa Rahimi Azghadi, Yuxuan Xia, Qing-Long Han, and Sumei Sun. Vehicle-to-everything cooperative perception for autonomous driving. Proceedings ofthe IEEE, 2025.

[15] Zhe Huang, Shuo Wang, Yongcai Wang, Wanting Li, Deying Li, and Lei Wang. Roco: Robust cooperative perception by iterative object matching and pose adjustment. In ACM Multimedia 2024, 2024.

[16] Zhe Huang, Shuo Wang, Yongcai Wang, and Lei Wang. Codiff: Conditional diffusion model for collaborative 3d object detection. arXiv preprint arXiv:2502.14891, 2025.

[17] Jason Ku, Ali Harakeh, and Steven L Waslander. In defense of classical image processing: Fast depth completion on the cpu. In 2018 15th Conference on Computer and Robot Vision (CRV), pages 16–22. IEEE, 2018.

[18] Alex H. Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, and Oscar Beijbom. Pointpillars: Fast encoders for object detection from point clouds. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019.

[19] Zixing Lei, Shunli Ren, Yue Hu, Wenjun Zhang, and Siheng Chen. Latency-aware collaborative perception. In European Conference on Computer Vision, pages 316–332. Springer, 2022.

[20] Jinlong Li, Runsheng Xu, Xinyu Liu, Jin Ma, Zicheng Chi, Jiaqi Ma, and Hongkai Yu. Learning for vehicle-to-vehicle cooperative perception under lossy communication. IEEE Transactions on Intelligent Vehicles, 2023.

[21] Yiming Li, Shunli Ren, Pengxiang Wu, Siheng Chen, Chen Feng, and Wenjun Zhang. Learning distilled collaboration graph for multi-agent perception. Advances in Neural Information Processing Systems, 34: 29541–29552, 2021.

[22] Jonathan Long, Evan Shelhamer, and Trevor Darrell. Fully convolutional networks for semantic segmenta tion. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3431–3440, 2015.

[23] Yifan Lu, Quanhao Li, Baoan Liu, Mehrdad Dianati, Chen Feng, Siheng Chen, and Yanfeng Wang. Robust collaborative 3d object detection in presence of pose errors. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 4812–4818. IEEE, 2023.

[24] Yifan Lu, Yue Hu, Yiqi Zhong, Dequan Wang, Yanfeng Wang, and Siheng Chen. An extensible framework for open heterogeneous collaborative perception. arXiv preprint arXiv:2401.13964, 2024.

[25] Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. Repaint: Inpainting using denoising diffusion probabilistic models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11461–11471, 2022.

[26] Fangchang Ma, Guilherme Venturelli Cavalheiro, and Sertac Karaman. Sparse-to-dense: Depth prediction from sparse depth samples and a single image. In 2018 IEEE International Conference on Robotics and Automation (ICRA), pages 1–8. IEEE, 2018.

[27] Ruiqing Mao, Haotian Wu, Yukuan Jia, Zhaojun Nan, Yuxuan Sun, Sheng Zhou, Deniz Gündüz, and Zhisheng Niu. Diffcp: Ultra-low bit collaborative perception via diffusion model. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 6587–6593. IEEE, 2025.

[28] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.

[29] Shunli Ren, Siheng Chen, and Wenjun Zhang. Collaborative perception for autonomous driving: Current status and future trend. In Proceedings of 2021 5th Chinese Conference on Swarm Intelligence and Cooperative Control, pages 682–692. Springer, 2022.

[30] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

[31] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.

[32] Zhiying Song, Lei Yang, Fuxi Wen, and Jun Li. Traf-align: Trajectory-aware feature alignment for asynchronous multi-agent perception. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 12048–12057, 2025.

[33] Sanbao Su, Songyang Han, Yiming Li, Zhili Zhang, Chen Feng, Caiwen Ding, and Fei Miao. Collaborative multi-object tracking with conformal uncertainty propagation. IEEE Robotics and Automation Letters, 9 (4):3323–3330, 2024.

[34] Nicholas Vadivelu, Mengye Ren, James Tu, Jingkang Wang, and Raquel Urtasun. Learning to communicate and correct pose errors. In Conference on Robot Learning, pages 1195–1210. PMLR, 2021.

[35] Rujia Wang, Xiangbo Gao, Hao Xiang, Runsheng Xu, and Zhengzhong Tu. Cocmt: Communicationefficient cross-modal transformer for collaborative perception. arXiv preprint arXiv:2503.13504, 2025.

[36] Tsun-Hsuan Wang, Sivabalan Manivasagam, Ming Liang, Bin Yang, Wenyuan Zeng, and Raquel Urtasun. V2vnet: Vehicle-to-vehicle communication for joint perception and prediction. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16, pages 605–621. Springer, 2020.

[37] Sizhe Wei, Yuxi Wei, Yue Hu, Yifan Lu, Yiqi Zhong, Siheng Chen, and Ya Zhang. Asynchrony-robust collaborative perception via bird’s eye view flow. Advances in Neural Information Processing Systems, 36: 28462–28477, 2023.

[38] Hao Xiang, Runsheng Xu, and Jiaqi Ma. Hm-vit: Hetero-modal vehicle-to-vehicle cooperative perception with vision transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 284–295, 2023.

[39] Hao Xiang, Zhaoliang Zheng, Xin Xia, Runsheng Xu, Letian Gao, Zewei Zhou, Xu Han, Xinkai Ji, Mingxi Li, Zonglin Meng, Li Jin, Mingyue Lei, Zhaoyang Ma, Zihang He, Haoxuan Ma, Yunshuang Yuan, Yingqian Zhao, and Jiaqi Ma. V2X-Real: a largs-scale dataset for vehicle-to-everything cooperative perception. In European Conference on Computer Vision, pages 455–470. Springer, 2024. doi: 10.1007/ 978-3-031-72943-0\_26.

[40] Runsheng Xu, Zhengzhong Tu, Hao Xiang, Wei Shao, Bolei Zhou, and Jiaqi Ma. Cobevt: Cooperative bird’s eye view semantic segmentation with sparse transformers, 2022.

[41] Runsheng Xu, Hao Xiang, Zhengzhong Tu, Xin Xia, Ming-Hsuan Yang, and Jiaqi Ma. V2x-vit: Vehicleto-everything cooperative perception with vision transformer, 2022.

[42] Runsheng Xu, Hao Xiang, Xin Xia, Xu Han, Jinlong Li, and Jiaqi Ma. Opv2v: An open benchmark dataset and fusion pipeline for perception with vehicle-to-vehicle communication. In 2022 International Conference on Robotics and Automation (ICRA), pages 2583–2589. IEEE, 2022.

[43] Runsheng Xu, Xin Xia, Jinlong Li, Hanzhao Li, Shuo Zhang, Zhengzhong Tu, Zonglin Meng, Hao Xiang, Xiaoyu Dong, Rui Song, et al. V2v4real: A real-world large-scale dataset for vehicle-to-vehicle cooperative perception. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13712–13722, 2023.

[44] Jingkang Yang, Kaiyang Zhou, Yixuan Li, and Ziwei Liu. Generalized out-of-distribution detection: A survey. International Journal ofComputer Vision, 132(12):5635–5662, 2024.

[45] Kang Yang, Tianci Bu, Lantao Li, Chunxu Li, Yongcai Wang, and Deying Li. Is discretization fusion all you need for collaborative perception? In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 9590–9596, 2025. doi: 10.1109/ICRA55743.2025.11128776.

[46] Hongbo Yin, Daxin Tian, Chunmian Lin, Xuting Duan, Jianshan Zhou, Dezong Zhao, and Dongpu Cao. V2vformer++: Multi-modal vehicle-to-vehicle cooperative perception via global-local transformer. IEEE Transactions on Intelligent Transportation Systems, 25(2):2153–2166, 2023.

[47] Haibao Yu, Yizhen Luo, Mao Shu, Yiyi Huo, Zebang Yang, Yifeng Shi, Zhenglong Guo, Hanyu Li, Xing Hu, Jirui Yuan, et al. Dair-v2x: A large-scale dataset for vehicle-infrastructure cooperative 3d object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21361–21370, 2022.

[48] Yunshuang Yuan, Yan Xia, Daniel Cremers, and Monika Sester. Sparsealign: A fully sparse framework for cooperative object detection. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22296–22305, 2025.

[49] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3836–3847, 2023.

[50] Binyu Zhao, Wei Zhang, and Zhaonian Zou. Bm2cp: Efficient collaborative perception with lidar-camera modalities. arXiv preprint arXiv:2310.14702, 2023.

[51] Seth Z Zhao, Huizhi Zhang, Zhaowei Li, Juntong Peng, Anthony Chui, Zewei Zhou, Zonglin Meng, Hao Xiang, Zhiyu Huang, Fujia Wang, et al. Quantv2x: A fully quantized multi-agent system for cooperative perception. arXiv preprint arXiv:2509.03704, 2025.

[52] Junfei Zhou, Penglin Dai, Quanmin Wei, Bingyi Liu, Xiao Wu, and Jianping Wang. Pragmatic heterogeneous collaborative perception via generative communication mechanism. arXiv preprint arXiv:2510.19618, 2025.

## A Experimental Protocol and Implementation Details

This appendix follows the reading path of the main paper. Appendix A documents the datasets, packet-loss protocol, and implementation details used in Sec. 4.1. Appendix B extends the robustness discussion in Sec. 4.2. Appendix C expands the ablations summarized in Sec. 4.3. Appendix D clarifies two method choices from Sec. 3.1, and Appendix E states the experimental scope and future extensions.

## A.1 Datasets and Metrics

We evaluate LR-V2X on two real-world collaborative perception benchmarks for LiDAR-based cooperative 3D object detection: DAIR-V2X [47] and V2XREAL [39]. DAIR-V2X represents the vehicle-to-infrastructure setting, while V2XREAL provides real-world vehicle-side collaborative driving scenes and is evaluated here in the LiDAR-only V2V setting used by OpenCOOD-style baselines. We follow the standard benchmark training and evaluation protocol for each dataset and preserve the native metric breakdown of each benchmark in the main tables: DAIR-V2X is reported with range-based AP, and V2XREAL is reported with class-based AP. Unless otherwise stated, all compared methods use the same PointPillar backbone so that the experiments isolate the effect of communication and reconstruction under packet loss.

For DAIR-V2X, Short, Middle, and Long denote diagnostic AP values computed on objects within the corresponding distance ranges, while the overall mAP columns are computed from all evaluated detections across the full range and are not the arithmetic mean of the three range bins. For V2XREAL, the overall mAP is the mean of the unrounded AP values over the vehicle, pedestrian, and truck super-classes.

## A.2 Packet-Loss Evaluation Protocol

Unless otherwise stated, all methods are trained under complete communication and evaluated under the same packet-loss simulation protocol. During testing, each spatial packet on the transmitted communication tensor is independently dropped with probability $1 - p ,$ , where p is the retention rate. The main robustness setting uses $p = 0 . 1$ (90% packet loss), and Fig. 5 sweeps the corresponding packet-loss rate from 0% to 90%. The ego feature is not transmitted and is always available to the ego-side fusion module. Evaluation uses fixed random seeds for mask generation and inference noise so that methods are compared under the same stochastic protocol.

## A.3 Implementation Details

Our experiments are implemented in the OpenCOOD [42] framework. We use PointPillar [18] as the encoder with a voxel size of (0.4m, 0.4m) and a maximum of 32 points per pillar over the LiDAR range [−102.4, −51.2, −3.5, 102.4, 51.2, 1.5]m. After PointPillars and backbone downsampling, the BEV feature tensor used by the latent encoder has spatial resolution 128 × 256 with 64 channels. All methods use the same evaluation protocol with fixed packet-loss masks and inference seeds. Configuration files and scripts for the main training and evaluation settings will be released with the code.

## A.3.1 Training Pipeline

We adopt a three-stage training pipeline:

Stage 1: Baseline Detection. We train the PointPillar encoder, pyramid fusion backbone (ResNeXt with 3 scales: [64, 128, 256] channels, upsample strides [1, 2, 4]), and detection head for 40 epochs with batch size 8. The optimizer is Adam with learning rate $2 \times \mathrm { i } 0 ^ { - 3 }$ , L2 regularization implemented via weight decay $1 0 ^ { - 4 }$ , and a multistep LR schedule (decay $\times 0 . 1$ at epochs 15 and 30). The detection loss combines Sigmoid Focal Loss $( \alpha = 0 . 2 5 , \gamma = 2 . 0 )$ for classification, Weighted Smooth L1 Loss $( \sigma = 3 . 0 )$ for regression, and pyramid losses at relative downsamples [1, 2, 4] with weights $[ 0 . 4 , 0 . 2 , 0 . 1 ]$

Stage 2: Reconstructor Training. We freeze the PointPillar sensor encoder, pyramid fusion backbone, and detection heads, and train the compact latent encoder together with the LPD and DiT denoiser for 100 epochs with batch size 12. The optimizer is Adam with learning rate $2 \times 1 0 ^ { - 4 }$

L2 regularization implemented via weight decay $1 0 ^ { - 4 }$ , and a multistep LR schedule (decay $\times 0 . 1$ 1 at epochs 50 and 80). Only the $x _ { 0 } \cdot$ -prediction MSE loss is used in this stage.

Stage 3: End-to-End Fine-tuning. We jointly fine-tune the latent encoder, LPD, DiT denoiser, pyramid fusion module, and detection heads while keeping the PointPillar sensor encoder/backbone frozen. This stage runs for 20 epochs with batch size 8. The total loss is $\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { d e t } } + 0 . 1 \cdot \mathcal { L } _ { \mathrm { d i f f } }$ using Adam with learning rate $2 \times 1 0 ^ { - 4 }$ and the same L2 regularization setting.

## A.3.2 Architecture Details

Latent Encoder. The encoder compresses BEV features from $1 2 8 \times 2 5 6 \times 6 4$ to a target spatial resolution of $1 6 \times 3 2$ with 64 latent channels (default ×8 downsampling), yielding a 128 KB message per transmitted collaborator frame.

Communication Accounting. For compact latent communication, the default payload contains $1 6 \times 3 2 \times 6 4$ float32 values. Dense BEV feature fusion transmits the corresponding $1 2 8 \times 2 5 6 \times 6 4$ float32 tensor, giving 8 MB. Early fusion sends raw point clouds, so we estimate its payload from validation files under float32 XYZI storage: about 1.0 MB on DAIR-V2X and about 2.4 MB on V2XREAL. Late fusion sends post-NMS detections; using the implementation cap of 150 boxes and a compact 7-DoF box, score, and class payload gives an upper bound of 5.4 KB.

Latent Prior Decoder (LPD). The LPD consists of four transposed convolutional layers that upsample from latent resolution $( 1 6 , 3 2 )$ back to BEV resolution (128, 256). Overlapping kernels enable spatial propagation from received locations into missing regions.

DiT Denoiser. We use the DiT-B configuration: 12 transformer layers, hidden dimension 768, 12 attention heads, MLP ratio 4.0, and patch size 4. Cross-attention layers attend to concatenated tokens from the received latent and aligned ego BEV. Gradient checkpointing is enabled to reduce memory usage.

## A.3.3 Noise Schedule and Inference Configuration

We use a cosine noise schedule with 1000 timesteps $( \beta _ { \mathrm { s t a r t } } = 1 0 ^ { - 4 } , \beta _ { \mathrm { e n d } } = 0 . 0 2 )$ . At inference, we sample Gaussian noise for the LPD prior, perturb it to a fixed inference timestep $t _ { \mathrm { i n f } } = 7 0 0$ , and perform a single DDIM-style [31] denoising forward pass from the noise-augmented prior to the reconstructed BEV feature. We keep this timestep fixed for all datasets and packet-loss rates, so the reported results do not rely on condition-specific tuning.

## B Robustness Analyses Beyond the Main Sweep

This section extends the performance analysis in Sec. 4.2. We first evaluate correlated burst loss to check whether the packet-loss trend is tied to the i.i.d. erasure model, and then use bandwidthmatched comparisons to separate reconstruction quality from the amount of information retained after transmission.

## B.1 Correlated Burst Loss

To assess robustness beyond i.i.d. packet erasures, we further evaluate a correlated burst-loss setting on DAIR-V2X under 90% packet loss. Figure 6 illustrates the difference between the two loss models: i.i.d. loss drops spatial packets independently and uniformly, whereas burst loss removes a contiguous rectangular region of packets, producing a large hole in the received feature map. For each transmitted communication feature map, we first sample a coarse $8 \times 1 6$ Bernoulli keep mask with retention rate $p = 0 . 1$ , then upsample it with nearest-neighbor interpolation to the native communication resolution. This produces contiguous missing regions while keeping the burst pattern normalized across methods with different communication resolutions. The ego feature is always kept, and the same per-sample mask is reused across methods for reproducibility and fair comparison.

Table 2 shows that the relative trend between low-bandwidth methods is preserved under burst loss. At the same 128 KB budget, LR-V2X outperforms CodeFilling across all overall mAP thresholds (+2.47/+1.59/+0.45 at IoU 0.3/0.5/0.7) and achieves the best short-range AP at all three thresholds. F-Cooper still attains the highest overall mAP in this setting, which is consistent with its much larger

![](images/4f01dcb5113747edee585d352e0ddafba1bf6d3c965346003cb09d980c4250ae.jpg)

![](images/5a90f9c40c97a52c7fd5cf790f966b98256d3044a32bd68fc83766247334df07.jpg)

![](images/7e9650050a70ab87845c8d69a5a70ace5bca364485fcf7a5ab4242d890010861.jpg)  
Figure 6: Illustration of i.i.d. packet loss (left) vs. burst packet loss (right) on an 8×8 BEV spatial grid. Blue cells are received; red cells are lost. Under i.i.d. loss, dropped packets are spatially scattered at random. Under burst loss, consecutive packets are dropped together, producing a contiguous missing region.

Table 2: Burst-loss results on DAIR-V2X under 90% packet loss. F-Cooper is a dense BEV feature fusion baseline with 8 MB communication, while CodeFilling and LR-V2X both operate at 128 KB per transmitted collaborator frame.
<table><tr><td rowspan="2">Models</td><td colspan="3">Short</td><td colspan="3">Middle</td><td colspan="3">Long</td><td colspan="3">mAP (%)</td><td rowspan="2">Comm</td></tr><tr><td>0.3</td><td>0.5</td><td>0.7</td><td>0.3</td><td>0.5</td><td>0.7</td><td>0.3</td><td>0.5</td><td>0.7</td><td>0.3</td><td>0.5</td><td>0.7</td></tr><tr><td>F-Cooper</td><td>79.15</td><td>76.03</td><td>65.84</td><td>76.34</td><td>71.05</td><td>56.56</td><td>55.91</td><td>50.68</td><td>37.78</td><td>69.88</td><td>65.08</td><td>52.12</td><td>8MB</td></tr><tr><td>CodeFilling</td><td>78.02</td><td>76.29</td><td>66.81</td><td>71.36</td><td>68.42</td><td>55.72</td><td>50.26</td><td>47.19</td><td>36.63</td><td>65.36</td><td>62.62</td><td>51.48</td><td>128KB</td></tr><tr><td>LR-V2X</td><td>79.96</td><td>77.53</td><td>66.97</td><td>74.15</td><td>70.22</td><td>56.38</td><td>52.70</td><td>48.75</td><td>36.99</td><td>67.83</td><td>64.21</td><td>51.93</td><td>128KB</td></tr></table>

8 MB communication budget and the larger amount of information it retains under correlated loss. Thus, burst loss is a stricter low-bandwidth reconstruction setting, but the method-level trend matches the main budget-matched comparison.

## B.2 Bandwidth–Robustness Analysis

A recurring observation in our experiments is that dense BEV feature fusion baselines (8 MB per transmitted collaborator frame) remain comparatively stable under packet loss. This behavior is closely tied to absolute information retention: under the same retention rate, a much larger communication budget leaves substantially more surviving information after transmission.

Information retention under packet loss. Let B denote the per transmitted collaborator frame communication budget and p the packet retention rate. The absolute information surviving transmission is B · p. Table 3 compares the retained information across representative bandwidth regimes at 90% packet loss (p = 0.1):

Table 3: Absolute information retained under 90% packet loss at different bandwidth levels.
<table><tr><td>Method class</td><td>Bandwidth B</td><td>Retained (B × 10%)</td><td>Rel. retained</td></tr><tr><td>Dense BEV feature fusion (e.g., F-Cooper)</td><td>8MB</td><td>819.2 KB</td><td>64×</td></tr><tr><td>Medium compression (×2)</td><td>2MB</td><td>204.8 KB</td><td>16×</td></tr><tr><td>Our default (×8)</td><td>128 KB</td><td>12.8 KB</td><td>1×</td></tr><tr><td>Aggressive (×16)</td><td>32 KB</td><td>3.2 KB</td><td>0.25×</td></tr></table>

Table 4: Budget-matched comparison on DAIR-V2X under 90% packet loss. Dense BEV feature fusion baselines are contrasted between their native dense BEV setting (left) and the protocol-matched 128 KB setting (right). LR-V2X is shown once as the native 128 KB reference.
<table><tr><td rowspan="2">Method</td><td colspan="4"></td><td colspan="5">Budget-Matched (128 KB)</td></tr><tr><td>mAP@0.3</td><td>Dense Native (8 MB) mAP@0.5</td><td>mAP@0.7</td><td>FPS</td><td>mAP@0.3</td><td>mAP@0.5</td><td>mAP@0.7</td><td>FPS</td><td>∆ mAP@0.5</td></tr><tr><td>Attn</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>66.87</td><td>66.20</td><td>52.56</td><td>6.67</td><td>62.99</td><td>60.25</td><td>48.65</td><td>6.91</td><td>-5.95</td></tr><tr><td>F-Cooper</td><td>67.89</td><td>65.90</td><td>52.87</td><td>5.39 4.92</td><td>63.24 67.73</td><td>61.67</td><td>48.67</td><td>9.30 5.30</td><td>-4.23</td></tr><tr><td>CoBEVT V2X-VIT</td><td>68.83 68.78</td><td>63.55 64.33</td><td>44.47 50.97</td><td>4.70</td><td>64.69</td><td>62.85 60.53</td><td>52.99 49.21</td><td>5.32</td><td>-0.70 -3.80</td></tr><tr><td>HEAL</td><td>64.43</td><td>61.68</td><td>50.50</td><td>7.24</td><td>60.81</td><td>58.08</td><td>47.25</td><td>10.00</td><td>-3.60</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LR-V2X</td><td colspan="3"></td><td></td><td>69.35</td><td>66.90</td><td>53.37</td><td>6.04</td><td></td></tr></table>

Dense-native values are copied from Table 1b under 90% packet loss. The 128 KB block reports equalized-budget re-evaluation, and ∆ mAP@0.5 is defined as mAP@0.5(128 KB) − mAP@0.5(native dense setting). LR-V2X is shown only in the 128 KB block because its native communication budget is already 128 KB.

At 90% loss, a dense 8 MB method still retains 819.2 KB, which is 64× more than the retained information of a 128 KB method at the same loss rate and 6.4× more than a complete 128 KB transmission. Dense BEV feature fusion baselines under severe loss therefore continue to operate with substantially more raw information than low-bandwidth methods. Table 4 complements the native-setting comparison by re-evaluating these dense BEV feature fusion baselines under the same 128 KB budget as LR-V2X.

Connecting to experimental results. This analysis helps interpret the trends in Table 1. Dense BEV feature fusion baselines such as Attn, F-Cooper, and HEAL degrade less sharply from no loss to 90% loss in part because they retain 819.2 KB of absolute information, whereas CodeFilling operates under the same 128 KB budget as LR-V2X. Table 4 makes this comparison more direct: after enforcing the same 128 KB budget, all five dense BEV feature fusion baselines decrease in mAP@0.5, with Attn decreasing by 5.95 points and even the most stable CoBEVT still decreasing by 0.70 points. Under this matched budget, LR-V2X achieves the best mAP among the 128 KB entries and exceeds the strongest dense BEV feature fusion baseline at 128 KB (CoBEVT) by 1.62/4.05/0.38 points at mAP@0.3/0.5/0.7.

## C Additional Ablations and Qualitative Results

This section expands the ablations summarized in Sec. 4.3. Unless noted otherwise, these results use DAIR-V2X and the same backbone, training schedule, and packet-loss protocol as the main experiments.

## C.1 Reconstructor Conditioning Inputs

Table 5: Ablation of reconstructor conditioning inputs on DAIR-V2X under complete communication (no packet loss). All variants reconstruct from the same latent-derived prior; the table only removes ego-BEV, transformation, and direct latent-conditioning pathways inside the reconstructor.
<table><tr><td colspan="3">Condition</td><td colspan="3">mAP (%)</td></tr><tr><td>Ego feat.</td><td>Trans. mat.</td><td>Latent feat.</td><td>mAP@0.3</td><td>mAP@0.5</td><td>mAP@0.7</td></tr><tr><td></td><td></td><td></td><td>67.28</td><td>63.71</td><td>52.01</td></tr><tr><td>√</td><td></td><td></td><td>78.90</td><td>74.15</td><td>55.77</td></tr><tr><td>√</td><td>√</td><td></td><td>79.69</td><td>75.34</td><td>56.56</td></tr><tr><td></td><td></td><td>√</td><td>67.30</td><td>63.75</td><td>52.04</td></tr><tr><td></td><td>√</td><td>√</td><td>68.30</td><td>64.74</td><td>53.04</td></tr><tr><td>√</td><td></td><td>V</td><td>79.52</td><td>75.29</td><td>56.72</td></tr><tr><td></td><td></td><td>√</td><td>81.03</td><td>76.16</td><td>57.74</td></tr></table>

Table 5 shows that ego-side BEV context is the most important reconstructor condition. Since every row still reconstructs from the same latent-derived prior, the first row is not a no-collaboration setting;

it tests whether the DiT can refine the prior without additional scene context. Adding aligned ego BEV gives the largest gain, indicating that the ego feature supplies geometric anchors in overlapping regions and helps the reconstructor distinguish plausible object structure from latent artifacts.

The transformation matrix and direct latent branch provide smaller but complementary refinements. The transformation contributes most when ego BEV is present because it tells the reconstructor how the ego context should be interpreted in the collaborator frame. The direct latent branch alone gives little gain because most collaborator evidence has already been decoded into the LPD prior, but combining all three conditions gives the highest mAP, suggesting that the branch still preserves residual compact cues that are useful after ego-guided refinement.

## C.2 Latent Prior Decoder Ablation

![](images/8284b40b4567c48db05eb827c62850597be88d9b17b8e326909bbc3847cd8856.jpg)

![](images/5349bec3c721a30eb4ad91662c1b853e42729a225a22ac268350fde9f4042701.jpg)

![](images/3445a69f2241b82ba7386288af11e671accacd128db422d95c4407160fb31903.jpg)  
Figure 7: Ablation of the Latent Prior Decoder (LPD) on DAIR-V2X. We compare the full model, prior-only (using the LPD output directly and skipping the denoising reconstructor), and zero-prior (replacing the LPD initialization with an all-zero BEV prior before denoising) under varying packetloss rates.

Fig. 7 separates three roles of the reconstruction module. The full model uses the LPD output as a coarse initialization and then applies the noise-conditioned reconstructor. The prior-only variant removes this refinement stage and sends the LPD output directly to fusion, whereas the zero-prior variant keeps the denoising reconstructor but removes the latent-derived initialization by starting from an all-zero BEV prior. The full model remains consistently above both alternatives across packet-loss rates and IoU thresholds, showing that the two stages address different parts of the recovery problem.

The prior-only variant degrades most rapidly as packet loss increases. This is expected because the LPD is a deterministic decoder from the surviving latent packets, so when large spatial regions are missing it can only propagate coarse context from the received locations and cannot actively resolve ambiguous object evidence. The zero-prior variant remains functional because the denoiser can still use ego context, direct latent conditioning, and learned scene regularities; however, without an object-aligned LPD initialization, it has to infer more of the collaborator feature from weak context. Thus the LPD is not a redundant component: it provides a spatially aligned starting point, while diffusion-style denoising supplies the adaptive refinement needed under packet loss.

## C.3 Spatial Downsampling Ratio

Table 6: Effect of spatial downsampling ratio on DAIR-V2X.
<table><tr><td>Downsample</td><td>mAP@0.3</td><td>mAP@0.5</td><td>mAP@0.7</td><td>Comm</td></tr><tr><td>×2</td><td>82.55</td><td>77.64</td><td>59.42</td><td>2MB</td></tr><tr><td>×4</td><td>81.52</td><td>76.79</td><td>58.52</td><td>512KB</td></tr><tr><td>×8</td><td>81.03</td><td>76.16</td><td>57.74</td><td>128KB</td></tr><tr><td>×16</td><td>79.58</td><td>73.51</td><td>54.52</td><td>32KB</td></tr></table>

Table 6 shows that the spatial ratio controls both bandwidth and the amount of local structure retained in the latent prior. Moving from ×2 to ×8 reduces communication from 2 MB to 128 KB, a 16× reduction, while the loss is only 1.52/1.48/1.68 mAP points at IoU 0.3/0.5/0.7. The default ×8 setting therefore sits near the knee of the accuracy–bandwidth curve: it removes most of the communication load while retaining enough spatial detail for the LPD and denoiser to reconstruct object-level BEV evidence.

The ×16 setting is more aggressive and begins to under-specify the reconstruction target. Its mAP@0.7 drop is larger than its mAP@0.3 drop, suggesting that coarse object presence can still be recovered but precise localization and shape are harder to preserve. This matches the role of the latent prior: when each latent cell summarizes too large a BEV region, the prior becomes smoother and less object-aligned, so the denoiser has to rely more heavily on ego context and learned priors rather than transmitted collaborator evidence.

## C.4 Noise-Conditioned Training and Inference Timestep

Table 7: Effect of noise-conditioned training on DAIR-V2X under 90% packet loss. The no-noise reconstructor keeps the same latent-derived prior and conditioning inputs but feeds the prior directly to the reconstructor at $t = 0 ,$ without forward-noise perturbation.
<table><tr><td>Method</td><td>mAP@0.3</td><td> $\mathrm { m A P @ 0 . 5 }$ </td><td> $\mathrm { m A P @ 0 . 7 }$ </td></tr><tr><td>No-noise reconstructor</td><td>62.19</td><td>59.68</td><td>49.14</td></tr><tr><td>LR-V2X  $( t _ { \mathrm { i n f } } = 7 0 0 )$ </td><td>69.35</td><td>66.90</td><td>53.37</td></tr></table>

![](images/14b52be1d663ac706e450724ad2f117503e3de0de017aca616b4dbf0dfc472db.jpg)

![](images/f2e0f11fef7504e96461f94f5c2f45d52949d3d164b853933600022d013570e8.jpg)

![](images/c2fd5407125cecaf12ab88a12982a181cf20e2749e011f51ea20fa603596b201.jpg)  
Figure 8: Sensitivity to the single-step inference timestep $t _ { \mathrm { i n f } }$ on DAIR-V2X under 90% packet loss. Solid curves show sweep results from a fixed checkpoint across different $t _ { \mathrm { i n f } }$ values, and the star marks the selected timestep $t _ { \mathrm { i n f } } = 7 0 0$ . The trend is broadly stable across timesteps, indicating that the single-step denoiser is not highly sensitive to the exact inference timestep within a wide operating range.

Table 7 isolates the value of noise-conditioned training under the main 90% packet-loss setting. The no-noise reconstructor is different from both prior-only and zero-prior in Fig. 7: it still uses the LPD prior and a learned reconstructor, but it removes the diffusion-style perturbation by feeding the prior directly at $t = 0$ . Without this perturbation, the module is closer to an ordinary deterministic reconstructor. It learns to map a received latent prior to a target BEV feature, but it is not explicitly trained to calibrate how much of the prior should be trusted when large parts of the message are missing. Noise-conditioned training exposes the model to a continuum of corrupted priors, making the denoiser learn when to preserve the prior, when to use ego context, and when to fill missing collaborator evidence from learned spatial regularities.

Fig. 8 further explains why the diffusion-style formulation is useful even with a single inference step. Low inference timesteps keep the input close to the LPD prior and may leave packet-loss artifacts under-corrected; very high timesteps inject more uncertainty and can weaken transmitted evidence. The selected $t _ { \mathrm { i n f } } = 7 0 0$ performs best in this sweep, while neighboring settings remain broadly stable, indicating that the method is not relying on a narrow condition-specific timestep.

## C.5 Qualitative Reconstruction

Fig. 9 provides a qualitative example that is consistent with the quantitative packet-loss results. Under complete communication, both reconstructions retain a visible response around the highlighted target, so the difference is not simply due to the complete-communication setting. Under 90% packet loss,

![](images/6c005681ec0f76efb2d66b1e9f76a43575e7e233f26008af19985718a921d251.jpg)  
Figure 9: Detection-guided qualitative comparison between LR-V2X and CodeFilling on a fixed DAIR-V2X sample. The same target region is highlighted across complete communication and 90% packet loss. Under severe loss, LR-V2X reconstructs a clearer object-centric BEV response and preserves the correct downstream detection, while CodeFilling misses the same target.

CodeFilling produces a weaker and noisier response in the target region, and the downstream detector misses the object. In contrast, LR-V2X reconstructs a more coherent object-centric activation that aligns with the ground-truth feature and preserves the detection.

This visualization supports the high-IoU trend in Table 1. Packet loss removes spatially localized evidence, and simple feature filling can leave either holes or diffuse artifacts that are insufficient for precise box regression. The LPD supplies a coarse object-aligned prior from surviving packets, while the denoising reconstructor sharpens that prior using ego context; the result is not just smoother BEV texture, but a feature pattern that remains useful to the detection head.

Fig. 10 complements the feature-level visualization by presenting final detection outputs across multiple samples and datasets under 90% packet loss. In these examples, HEAL and CodeFilling produce more false positives or missed objects when collaborator evidence is sparse, especially in crowded regions. In contrast, LR-V2X produces detections that are more consistently aligned with the ground-truth layout.

![](images/a4ddc364c9170ec631c50e2e0b0e37fa82495e5eed426acddab27dee145a31bf.jpg)  
Figure 10: Visualization of detection results on DAIR-V2X and V2XREAL under 90% packet loss. Columns compare HEAL, CodeFilling, and LR-V2X on the same samples. The top two rows are DAIR-V2X examples, and the bottom two rows are V2XREAL examples. Red boxes denote predictions and green boxes denote ground-truth boxes.

## D Clarifications on Design Choices

This section provides brief technical clarifications for two design choices used in LR-V2X.

## D.1 Single-Step Reconstruction vs. Iterative Sampling

LR-V2X uses a single DDIM-style step at inference because reconstruction starts from the LPD prior BEV map $X _ { i } ^ { \mathrm { p r i o r } }$ rather than from an uninformative initialization. The denoiser is trained with an x<sub>0</sub>-prediction objective over noisy versions of this prior BEV map. At inference, Gaussian noise perturbs the prior to the fixed level $t _ { \mathrm { i n f } } = 7 0 0$ , and one forward pass predicts the reconstructed collaborator BEV. Fig. 8 is consistent with this interpretation: performance remains broadly stable across a wide range of $t _ { \mathrm { i n f } }$ values, indicating that the reconstruction step is not highly sensitive to the exact inference timestep.

## D.2 Role of the Ego BEV Condition

The ego vehicle’s own BEV feature serves as complementary spatial context for reconstruction. For collaborator i, the ego feature is first warped into the collaborator coordinate frame as $X _ { 0 } ^ { \mathrm { a l i g n e d } } =$ $\mathcal { W } ( X _ { 0 } , T _ { 0  i } )$ , matching the reconstruction formulation in Sec. 3.1. This aligned ego feature provides geometric anchors in overlapping regions and broader scene context around missing collaborator features. After reconstruction, the collaborator feature is warped back to the ego frame before fusion.

Table 5 supports this role: removing the ego BEV condition produces the largest performance drop among the conditioning ablations.

## E Experimental Scope and Extensions

The current evaluation focuses on real-world two-party V2X benchmarks, DAIR-V2X and V2XREAL, which provide a controlled setting for isolating lossy low-bandwidth communication from other multi-agent factors. Extending the same reconstruction protocol to larger cooperative graphs is a natural next step. Our default packet-loss model uses i.i.d. spatial erasures for reproducibility, and Appendix B.1 adds a correlated burst-loss evaluation to cover a stronger spatial corruption pattern. Future extensions can incorporate richer channel simulators that jointly model latency, congestion, retransmission, and temporal burst dynamics. On the deployment side, lightweight denoisers, distillation, or adaptive invocation are practical directions for improving inference efficiency without changing the communication protocol.