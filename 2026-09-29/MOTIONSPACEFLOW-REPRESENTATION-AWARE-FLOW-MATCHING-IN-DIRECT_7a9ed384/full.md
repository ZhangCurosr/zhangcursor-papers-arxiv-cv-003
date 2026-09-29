# MOTIONSPACEFLOW: REPRESENTATION-AWARE FLOW MATCHING IN DIRECT MOTION SPACE

Qing Yu Kent Fujiwara

LY Corporation, Tokyo, Japan

{yu.qing, kent.fujiwara}@lycorp.co.jp

## ABSTRACT

Recent advances in diffusion and flow models have substantially improved textdriven human motion generation. Yet most methods generate in low-dimensional, temporally downsampled latent spaces learned primarily for reconstruction, a bottleneck that can limit generation quality and preclude direct manipulation of individual frames and joints. We introduce MotionSpaceFlow (MSFLOW), a representation-aware flow-matching framework that predicts clean motion directly in continuous motion space without a learned encoder or decoder. To account for the anisotropic structure of direct motion representations, we propose representation-aware noise scaling and show how the initial Gaussian source scale governs the covariance of intermediate probability-path marginals. We further introduce a Representation-Aware Multimodal Diffusion Transformer (RA-MMDiT), which jointly updates token-level language and full-resolution motion features through joint attention while adapting temporal information flow to the motion representation: causal attention for incremental features defined by frameto-frame changes, and bidirectional attention for global features such as absolute joint coordinates. Across different datasets and motion representations, MS-FLOW achieves state-of-the-art text-to-motion performance. Its global representation variant additionally enables zero-shot, inference-time control over any joint or frame through projection sampling without control-conditioned training, delivering leading motion quality with exact constraint satisfaction.

## 1 INTRODUCTION

Synthesizing human motion from natural language requires translating a high-level description into a coherent physical trajectory. Diffusion and flow models (Ho et al., 2020; Song et al., 2021; Rombach et al., 2022; Lipman et al., 2023) address this one-to-many problem by transforming Gaussian noise into complete motion sequences. Most recent methods generate in low-dimensional, temporally downsampled spaces produced by variational autoencoders (VAEs) (Kingma & Welling, 2014; Chen et al., 2023; Dai et al., 2024; Hong et al., 2025), reducing sequence length and generation cost. Others discretize motion with a VQ-VAE (van den Oord et al., 2017) and apply transformer-based generative models to the resulting token sequence (Guo et al., 2022b; Jiang et al., 2023; Guo et al., 2024). Although effective, both continuous and discrete latent approaches learn representations primarily through reconstruction. Generation quality can therefore be limited by autoencoder capacity and reconstruction fidelity (Yao et al., 2025; Zheng et al., 2026), while latent tokens lose direct correspondence to individual frames and joints, complicating fine-grained spatial control.

As illustrated in Fig. 1, we instead generate full-resolution motion variables with a Diffusion Transformer (DiT)-only model. Removing the motion encoder and decoder eliminates the reconstruc tion bottleneck and preserves the frame–joint structure required for inference-time spatial control. Direct-space generation, however, is not achieved by simply removing the autoencoder. Coordinatewise z-normalization does not eliminate correlations across joints, feature types, and frames because motion representations combine continuous 3D quantities (e.g., root-relative joint positions), continuous 6D rotations (e.g., joint orientations), and categorical variables (e.g., foot-contact labels). The flow must therefore traverse an anisotropic data distribution, adapt temporal dependencies to the represented variables, and align a short prompt with full-resolution motion.

![](images/a505549f44cfad77443eeb37ef4af783f212159cd28825f5853a89f5ce0be407.jpg)  
Figure 1: Latent-space versus direct motion-space generation. Existing methods encode a Tframe sequence into a shorter latent sequence, apply a diffusion transformer in the latent space, and decode the result back to motion. MSFLOW removes the motion encoder and decoder and applies a flow-matching transformer directly to the full sequence. This end-to-end design avoids autoencoder reconstruction error and preserves direct access to frames and joints for zero-shot spatial control.

To address these challenges, we introduce MotionSpaceFlow (MSFLOW), a representation-aware framework for flow matching in continuous motion space. Following JiT (Li & He, 2026), MSFLOW predicts the clean motion endpoint and analytically converts it into a velocity field for deterministic ODE sampling. Representation-aware noise scaling incorporates the Gaussian source scale into the probability path used for both training and sampling, treating it as a parameter of the probability path rather than merely as a control for inference-time randomness. Our analysis shows how this scale controls signal emergence and the anisotropy of intermediate path marginals.

Our Representation-Aware Multimodal DiT (RA-MMDiT) jointly updates token-level language and frame-level motion features to support fine-grained alignment across the full sequence. Crucially, its temporal attention graph is configured to match the underlying representation. The standard 263- dimensional HumanML3D representation (Guo et al., 2022a) includes root velocities whose accumulation recovers the global trajectory of the whole sequence. RA-MMDiT uses causal attention fo these incremental features: all frames are updated in parallel at each ODE step, while each motion token attends only to its prefix and the complete text condition. For globally coupled absolute XYZ coordinates of all joints, it instead uses bidirectional attention, providing two-sided context and propagating spatial constraints before and after a controlled keyframe. This representation-aware attention lets RA-MMDiT handle both incremental and absolute motion representations.

On HumanML3D, the causal 263D variant achieves state-of-the-art text-to-motion performance, and our results reveal a clear contrast between the optimal attention designs for incremental 263D and absolute XYZ representations. The XYZ variant further enables any-joint, any-frame projection sampling without control-conditioned training, delivering leading motion quality and exact constraint satisfaction under the OmniControl (Xie et al., 2024) protocol. We also investigate the broader applicability of MSFLOW on SnapMoGen dataset (Guo et al., 2025). Our main contributions are:

• Direct motion-space flow matching. We introduce MSFLOW, which generates continuous motion through clean-endpoint prediction and ODE sampling without a learned motion encoder or decoder, and achieves state-of-the-art text-to-motion results.

• Representation-aware noise scaling. We characterize how Gaussian source scale controls signal emergence and the covariance conditioning of intermediate direct-space path marginals.

• Representation-aware temporal modeling. We introduce RA-MMDiT, which jointly updates token-level language and full-resolution motion features, and show that causal attention favors incremental representations whereas bidirectional attention is crucial for absolute coordinates.

• Training-free spatial control. We derive projection sampling for any-joint, any-frame constraints using a model trained only for text-to-motion generation, with exact constraint satisfaction.

## 2 RELATED WORK

## 2.1 MOTION REPRESENTATIONS AND TEMPORAL MODELING FOR MOTION GENERATION

Early neural approaches generate motion with recurrent or convolutional architectures (Yan et al., 2019; Zhao et al., 2020; Guo et al., 2020; Ghosh et al., 2021). Recent text-conditioned models operate on discrete tokens (Guo et al., 2022b; Zhang et al., 2023a; Guo et al., 2024; Pinyoanuntapong et al., 2024; Zou et al., 2024), continuous motion features (Tevet et al., 2023; Zhang et al., 2024;

Dabral et al., 2023), or autoencoded latent sequences (Petrovich et al., 2022; Chen et al., 2023; Dai et al., 2024; Hong et al., 2025). Discrete and autoencoded representations shorten the modeled sequence, but temporal downsampling limits information capacity and decoded fidelity remains bounded by autoencoder reconstruction quality. MSFLOW instead learns transport in continuous motion space, preserving frame- and joint-level access.

Autoregressive models enforce temporal causality by predicting future tokens or segments from past context. Discrete autoregressive methods (Zhang et al., 2023a; Jiang et al., 2023) can suffer from exposure bias (Schmidt, 2019), whereas diffusion-based variants (Meng et al., 2025b; Zhao et al., 2025; Xiao et al., 2025; Yu et al., 2026) combine iterative refinement with causal backbones. Bidirectional denoisers instead use complete temporal context. We compare matched causal and bidirectional variants to test whether the preferred dependency graph depends on the representation.

## 2.2 FLOW MATCHING AND SPATIAL CONTROL

Flow matching (Albergo & Vanden-Eijnden, 2023; Albergo et al., 2023; Lipman et al., 2023) learn vector fields that transport a simple source distribution to data, and scalable interpolant and Diffusion Transformers (Peebles & Xie, 2023; Ma et al., 2024) provide effective parameterizations of such fields. We adopt clean-endpoint prediction from JiT (Li & He, 2026), which allows the method to operate effectively in high-dimensional spaces, and study its interaction with Gaussian source scale and temporal attention in anisotropic motion representations. Some prior work selects or optimizes the initial noise of a fixed pretrained diffusion process to control generation (Mao et al., 2024; Wang et al., 2025; Zhou et al., 2025; Harrington et al., 2026; Ota et al., 2026).

In motion generation, prior methods impose spatial constraints or edits through control-conditioned training, guidance, test-time optimization, or large-scale controllable modeling (Karunratanakul et al., 2023; Athanasiou et al., 2024; Huang et al., 2024b; Petrovich et al., 2024; Xie et al., 2024; Liu et al., 2024; Rempe et al., 2026). ProjFlow (Watanabe et al., 2026) instead projects clean endpoint estimates during sampling without control-conditioned training. MSFLOW applies endpoint projection directly to XYZ motion, enabling arbitrary frame–joint–axis constraints.

## 2.3 MOTION–LANGUAGE CONDITIONING

Unlike motion–language alignment models (Tevet et al., 2022; Petrovich et al., 2023; Yu et al., 2024; Fujiwara et al., 2024; Yu et al., 2025), which jointly update motion and text features through contrastive learning, text-to-motion generators commonly condition on a frozen language encoder. In direct motion-space generation, preserving textual semantics is significantly more challenging because no temporally downsampled latent representation is available to bridge the modality gap. MSFLOW therefore refines token-level DistilBERT features (Sanh et al., 2019) as a function of flow time and couples them with motion tokens through multimodal attention, without an auxiliary motion–language alignment objective.

## 3 METHOD

MSFLOW generates motion directly in the data representation, without a learned motion encoder or decoder. Fig. 2 shows that the method consists of two components. First, a clean motion sequence, represented using either incremental 263D features or global XYZ coordinates, is interpolated with a scaled Gaussian source to form a noised motion state. Second, a Representation-Aware Multimodal Diffusion Transformer (RA-MMDiT) combines this state with token-level text features to predict the clean motion endpoint. The representation informs both the source scale and the temporal attention graph: causal for incremental features and bidirectional for absolute coordinates. The predicted endpoint is converted into an ODE velocity for the training objective and inference-time sampling. For XYZ motion, the endpoint can also be projected to satisfy the spatial constraints during inference.

## 3.1 PROBLEM SETTING AND DIRECT MOTION REPRESENTATIONS

Let $\mathbf { x } _ { 1 } ~ \in ~ \mathbb { R } ^ { T \times D }$ be a normalized motion sequence, where T is the number of frames and D is the feature dimension, and let c denote a text prompt, such as a natural language description of the motion. We model the sequence as a continuous tensor at its original temporal resolution. We consider two commonly used representations with different temporal semantics:

![](images/c14cf46f5bf1508f36448076a97fcb2087cd3a8905ca38a574d5ba2ee181666a.jpg)  
Figure 2: Framework of MSFLOW. Left: motion-space flow matching interpolates a scaled Gaussian source ${ \bf x } _ { 0 } = s \epsilon$ with a clean T-frame motion $\mathbf { x } _ { 1 }$ in either an incremental or global representation. Right: RA-MMDiT jointly processes the noised motion $\mathbf { x } _ { t }$ and time-refined text tokens to predict the clean endpoint $\hat { \mathbf { x } } _ { 1 }$ . Its attention mask follows the motion representation: a motion query attends only to its prefix and all text tokens in the causal variant, whereas the bidirectional variant uses the complete motion–text context.

• Incremental 263D motion. The standard HumanML3D representation (Guo et al., 2022a) contains root yaw and planar velocities, root height, root-relative joint positions, 6D joint rotations, local joint velocities, and foot-contact channels. In particular, the global root trajectory is obtained by accumulating frame-to-frame velocities.

• Global XYZ motion. Each frame contains the absolute 3D coordinates of 22 joints, resulting in $D = 6 6 .$ . These variables are globally coupled across time but expose every frame and joint directly, which makes them suitable for spatial control.

In both cases, the flow acts on $\mathbf { x } _ { 1 }$ itself; there is no learned map between a temporally compressed latent sequence and motion. The following bound isolates the reconstruction constraint that this design removes. Let $P _ { 1 }$ be the data distribution over vectorized motions in $\mathbb { R } ^ { T D }$ , and let $g : { \mathcal { Z } } $ $\mathbb { R } ^ { T \tilde { D } }$ be a fixed decoder with range $\mathcal { R } _ { g } = \{ g ( \mathbf { z } ) : \mathbf { z } \in \mathcal { Z } \}$

Proposition 1 (Autoencoder fidelity floor). For any generated distribution Q supported on $\mathcal { R } _ { g } { _ { ; } }$

$$
W _ { 2 } ^ { 2 } ( P _ { 1 } , Q ) \geq \mathbb { E } _ { \mathbf { X } \sim P _ { 1 } } \left[ \mathrm { d i s t } ^ { 2 } ( \mathbf { X } , \mathcal { R } _ { g } ) \right] ,\tag{1}
$$

where $W _ { 2 } ^ { 2 }$ denotes the squared 2-Wasserstein distance. Every decoded sample must lie in the decoder range, regardless of the latent generator’s expressiveness. Because the lower bound can be zero when that range contains the data support, it does not claim that direct generation is universally better. Rather, it shows that direct motion-space flow eliminates this particular decoder-induced fidelity floor while retaining ordinary generative error.

## 3.2 CLEAN-MOTION PREDICTION AND FLOW SAMPLING

The left side of Fig. 2 defines the probability path. During training, we sample one time $t \sim \mathcal { U } ( 0 , 1 )$ per sequence and one Gaussian tensor ϵ of the same shape as the motion, then form

$$
\begin{array} { r } { \mathbf { x } _ { 0 } = s \epsilon , \qquad \epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) , \qquad \mathbf { x } _ { t } = ( 1 - t ) \mathbf { x } _ { 0 } + t \mathbf { x } _ { 1 } , } \end{array}\tag{2}
$$

where $s > 0$ is the source scale. The same t is used for all valid frames so that $\mathbf { x } _ { t }$ remains a coherent sequence. Rather than predicting velocity directly, RA-MMDiT predicts the clean endpoint,

$$
\begin{array} { r } { \hat { \mathbf { x } } _ { 1 } = f _ { \theta } ( \mathbf { x } _ { t } , t , \mathbf { c } ) . } \end{array}\tag{3}
$$

This x-prediction parameterization follows JiT (Li & He, 2026) and provides the network a target with the same physical meaning at every noise level, where $\hat { \mathbf { x } } _ { 1 }$ always retains the semantics and

units of the original motion representation. For the linear path, the exact sample-wise velocity and its endpoint-based estimate are calculated as follows:

$$
{ \mathbf v } = { \mathbf x } _ { 1 } - { \mathbf x } _ { 0 } = \frac { { \mathbf x } _ { 1 } - { \mathbf x } _ { t } } { 1 - t } , \qquad \hat { \mathbf v } _ { \theta } = \frac { \hat { { \mathbf x } } _ { 1 } - { \mathbf x } _ { t } } { 1 - t } .\tag{4}
$$

Thus, clean-motion prediction remains compatible with the flow ODE $d \mathbf x _ { t } / d t = \hat { \mathbf v } _ { \theta }$ . The denominator in Eq. (4) becomes numerically unstable near $t = 1$ . We therefore clip it as follows:

$$
d _ { t } = \operatorname* { m a x } ( 1 - t , \sigma _ { \operatorname* { m i n } } ) , \qquad { \hat { \mathbf { v } } } _ { \theta } = { \frac { { \hat { \mathbf { x } } } _ { 1 } - { \mathbf { x } } _ { t } } { d _ { t } } } , \qquad { \mathbf { v } } ^ { * } = { \frac { { \mathbf { x } } _ { 1 } - { \mathbf { x } } _ { t } } { d _ { t } } } ,\tag{5}
$$

and optimize the valid-frame residual loss as follows:

$$
\mathcal { L } _ { \mathrm { F M } } = \mathbb { E } \left[ \frac { 1 } { \left| \Omega \right| } \sum _ { i \in \Omega } \left\| \hat { \mathbf { v } } _ { \boldsymbol { \theta } , i } - \mathbf { v } _ { i } ^ { * } \right\| _ { 2 } ^ { 2 } \right] ,\tag{6}
$$

where Ω is the set of non-padding frames. For $t \le 1 - \sigma _ { \operatorname* { m i n } } ,$ this is exactly the linear-path velocity objective, equivalently weighting endpoint error by $( 1 - t ) ^ { - 2 }$ . The terminal interval uses a bounded surrogate. At inference, we initialize the entire sequence from ${ \mathcal { N } } ( \mathbf { 0 } , s ^ { 2 } \mathbf { I } )$ and integrate all frames jointly with a fixed-step Heun solver (Heun, 1900) and a final Euler update.

## 3.3 REPRESENTATION-AWARE NOISE SCALING

Following standard practice (Guo et al., 2022a; Meng et al., 2025a), we normalize every motion channel before training. However, normalization alone does not make the motion distribution spherical. It rescales individual channels without removing correlations across joints, feature types, or frames. The resulting 263D and XYZ distributions can therefore have much more variation in some directions than in others, whereas the Gaussian source varies equally in every direction. We verify this structure empirically in Appendix E.2.

The source scale s in Eq. (2) is the standard deviation of the initial Gaussian noise. It is fixed during both training and sampling and should not be confused with an inference-time temperature. Increasing s has two intuitive effects: stronger noise causes the motion signal to appear later along the path, while its isotropic covariance makes intermediate states better conditioned.

To formalize these effects, let $\mathbf { X } _ { 1 }$ be a vectorized clean motion for a fixed-length sequence. Assume $\mathbb { E } [ \mathbf { X } _ { 1 } ] = \mathbf { 0 }$ and denote its covariance by Σ, with smallest and largest eigenvalues $\lambda _ { \operatorname* { m i n } }$ and $\lambda _ { \mathrm { m a x } }$ Proposition 2 (Effects of source scale on the linear flow path). Let $\mathbf { X } _ { 0 } \sim { \mathcal { N } } ( \mathbf { 0 } , s ^ { 2 } \mathbf { I } )$ be independent of $\mathbf { X } _ { 1 } ,$ , and let

$$
\begin{array} { r } { \mathbf { X } _ { t } = t \mathbf { X } _ { 1 } + ( 1 - t ) \mathbf { X } _ { 0 } . } \end{array}\tag{7}
$$

Then s controls when the motion signal emerges and how well conditioned the intermediate distribution is:

(i) For any unit eigenvector u ofΣ with eigenvalue ${ \lambda } _ { \mathbf { u } } ,$

$$
\mathrm { S N R } _ { \mathbf { u } } ( t ; s ) = \frac { t ^ { 2 } \lambda _ { \mathbf { u } } } { ( 1 - t ) ^ { 2 } s ^ { 2 } } , \qquad t _ { \mathbf { u } } ^ { * } = \frac { s } { s + \sqrt { \lambda _ { \mathbf { u } } } } ,\tag{8}
$$

where $t _ { \mathbf { u } } ^ { * }$ is the time at which the signal and noise variances are equal. A larger s moves this time closer to the clean endpoint $t = 1$ , so the motion signal emerges later.

(ii) The intermediate covariance and its condition numberfor $t < 1$ are

$$
\operatorname { C o v } ( \mathbf { X } _ { t } ) = t ^ { 2 } \Sigma + ( 1 - t ) ^ { 2 } s ^ { 2 } \mathbf { I } ,\tag{9}
$$

$$
\kappa _ { t } ( s ) = \frac { t ^ { 2 } \lambda _ { \operatorname* { m a x } } + ( 1 - t ) ^ { 2 } s ^ { 2 } } { t ^ { 2 } \lambda _ { \operatorname* { m i n } } + ( 1 - t ) ^ { 2 } s ^ { 2 } } .\tag{10}
$$

The condition number $\kappa _ { t } ( s )$ is non-increasing in s. Thus, a larger isotropic source scale improves conditioning by reducing the relative disparity between high- and low-variance directions, while leaving their absolute eigenvalue gap unchanged.

Our ablations show that $s = 5$ works well for both the 263D and XYZ representations. We therefore use this value in our primary models.

## 3.4 REPRESENTATION-AWARE MULTIMODAL TRANSFORMER

The right side of Fig. 2 shows RA-MMDiT. Each motion frame is projected to a 512-dimensional token. A frozen DistilBERT encoder (Sanh et al., 2019) produces 768-dimensional token-level language features. We use a two-layer, time-aware text module, which we call the Token Refiner, to project these features to the model width and adapt them to the current flow time. This makes the otherwise static language features responsive to the changing motion signal along the flow path. Rotary position embeddings (Su et al., 2024) encode motion order, adaptive layer normalization (Peebles & Xie, 2023) injects flow time, and the text condition is independently replaced by a null embedding with probability 0.1 to train classifier-free guidance.

The network contains eight RA-MMDiT blocks with four attention heads (Vaswani et al., 2017). Each block has modality-specific query, key, value, and feed-forward projections, while joint attention concatenates motion and text keys and values. This allows every motion frame to retrieve relevant words without first compressing the prompt to a single vector. The attention mask is the only architectural difference between the two representation variants:

• In the causal variant, motion query i attends to motion frames 1:i and all valid text tokens. Text queries attend only to text tokens, preventing future motion from reaching an earlier frame indirectly through the updated language stream.

• In the bidirectional variant, motion and text queries attend to the complete valid motion–text sequence. Padding tokens are masked in both variants.

The mask reflects each representation’s structure. Latent models can handle the directionality of incremental 263D features implicitly through their encoder and decoder, whereas direct motion-space generation must model it explicitly. Because 263D root translation and yaw are integrated from frame-to-frame velocities, we use causal masking to encourage forward-consistent dependencies and limit reliance on bidirectional smoothing.

Absolute XYZ trajectories are globally coupled, so bidirectional context can propagate temporal and skeletal constraints. Causal attention still updates all frames jointly at each ODE step, but prevents future perturbations from affecting earlier frames. These filtering and smoothing properties motivate the causal 263D and bidirectional XYZ variants evaluated in Section 4.3; formal analyses appear in Appendix E.3. At sampling time, classifier-free guidance (Ho & Salimans, 2022) is applied to the predicted endpoints and the result is converted to a velocity using Eq. (5). Because the model predicts the target motion variables directly, the integrated endpoint requires no motion decoder.

## 3.5 INFERENCE-TIME ANY-JOINT, ANY-FRAME CONTROL

Direct XYZ endpoints enable zero-shot spatial control at inference. The bidirectional XYZ model is trained on paired motions and text, without constraint masks or target coordinates. At inference, let $\mathbf { M } \in \{ 0 , 1 \dot { \} } ^ { T \times 6 6 }$ select any valid frame–joint–axis entries and y contain their targets in normalized XYZ coordinates. At each solver step, we first project the predicted clean endpoint:

$$
\hat { \mathbf { x } } _ { 1 } ^ { \mathrm { p r o j } } = \left( { \bf 1 } - \mathbf { M } \right) \odot \hat { \mathbf { x } } _ { 1 } + \mathbf { M } \odot { \bf y } .\tag{11}
$$

Simply overwriting the same entries of the noised state $\mathbf { x } _ { t }$ with clean coordinates would mix two noise levels. Instead, we write the linear path as $\mathbf { x } _ { t } = \alpha _ { t } \mathbf { x } _ { 1 } + \sigma _ { t } \mathbf { x } _ { 0 }$ , where $\alpha _ { t } = t$ and $\sigma _ { t } = 1 - t ,$ and recover the current source estimate $\hat { \mathbf { x } } _ { 0 } = ( \mathbf { x } _ { t } - \alpha _ { t } \hat { \mathbf { x } } _ { 1 } ) / \sigma _ { t }$ for $t < 1$ . For the next solver time $t ^ { \prime } ,$ the projection sampler refreshes that estimate with a new source-scale noise $\epsilon _ { s }$ and recomposes a path-consistent state:

$$
\eta _ { t ^ { \prime } } = \mathrm { c l i p } \left( \frac { 1 - \sigma _ { t ^ { \prime } } } { s ^ { 2 } } , 0 , 1 \right) ,\tag{12}
$$

$$
\widetilde { \mathbf { x } } _ { 0 } = \sqrt { 1 - \eta _ { t ^ { \prime } } } \hat { \mathbf { x } } _ { 0 } + \sqrt { \eta _ { t ^ { \prime } } } \epsilon _ { s } , \quad \epsilon _ { s } \sim \mathcal { N } ( \mathbf { 0 } , s ^ { 2 } \mathbf { I } ) ,\tag{13}
$$

$$
\begin{array} { r } { \mathbf { x } _ { t ^ { \prime } } = \alpha _ { t ^ { \prime } } \hat { \mathbf { x } } _ { 1 } ^ { \mathrm { p r o j } } + \sigma _ { t ^ { \prime } } \widetilde { \mathbf { x } } _ { 0 } . } \end{array}\tag{14}
$$

The injected noise uses the source scale of the training path, and a final hard projection removes residual numerical error at the endpoint. Following ProjFlow (Watanabe et al., 2026), our projection sampler also adopts its kinematics-aware metric and pseudo-observation construction. We refer readers to that paper for details. Neither M nor y appears in the training objective, so “training-free control” refers specifically to introducing spatial constraints only during inference.

Table 1: Results of text-to-motion performance on HumanML3D. We use the 67D evaluator following MARDM. MSFLOW (263D) denotes the incremental-263D model, while MSFLOW (XYZ) denotes the global-XYZ variant. The average is reported over 10 runs with 95% confidence intervals. Bold indicates the best result, and underline denotes the second-best result.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Representation</td><td colspan="3">R-Precision↑</td><td rowspan="2">FID↓</td><td rowspan="2">MM-Dist↓ MModality↑</td><td rowspan="2"></td><td rowspan="2">CLIP↑</td></tr><tr><td>Top 1</td><td>Top 2</td><td>Top 3</td></tr><tr><td>T2M-GPT (Zhang et al., 2023a) MMM (Pinyoanuntapong et al., 2024)</td><td rowspan="3">Discrete latent</td><td>.470±.003</td><td>.659±.002</td><td>.758±.002</td><td>.335±.003</td><td>3.505±.017</td><td>2.018±.053</td><td>.607±.005</td></tr><tr><td>MoMask (Guo et al., 2024)</td><td>.487±.003 .490±.004</td><td>.683±.002 .687±.003</td><td>.782±.002 .786±.003</td><td>.132±.004 .116±.006</td><td>3.359±.019 3.353±.010</td><td>2.241±.073 1.263±.079</td><td>.635±.003 .637±.003</td></tr><tr><td>MLD (Chen et al., 2023)</td><td>.461±.004</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="4">SALAD (Hong et al., 2025) MARDM (Meng et al., 2025b)</td><td rowspan="4">Continuous latent</td><td></td><td>.651±.004</td><td>.750±.003</td><td>.431±.014 .124±.005</td><td>3.445±.019</td><td>3.506±.031</td><td>.615±.003</td></tr><tr><td>.552±.003</td><td>.748±.003</td><td>.839±.002</td><td></td><td>2.990±.010</td><td>1.833±.081</td><td>.671±.001</td></tr><tr><td>.500±.004</td><td>.695±.003</td><td>.795±.003</td><td>.114±.007</td><td>3.270±.009</td><td>2.231±.071</td><td>.642±.002</td></tr><tr><td>.522±.002 .563±.004</td><td>.713±.002 .759±.003</td><td>.807±.002</td><td>.058±.004</td><td>3.205±.008</td><td>2.077±.083</td><td>.652±.001 .685±.001</td></tr><tr><td rowspan="4">CMDM (Yu et al., 2026) MDM (Tevet et al., 2023)</td><td rowspan="4">Raw motion</td><td></td><td></td><td>.849±.002</td><td>.078±.003</td><td>2.920±.007</td><td>1.827±.094</td><td></td></tr><tr><td>.440±.007</td><td>.636±.006</td><td>.742±.004</td><td>.518±.032</td><td>3.640±.028</td><td>3.604±.031</td><td>.578±.003</td></tr><tr><td>.450±.006</td><td>.641±.005</td><td>.753±.005</td><td>.778±.035</td><td>3.490±.023</td><td>3.179±.046</td><td>.606±.004</td></tr><tr><td>.468±.003 .571±.005</td><td>.653±.003</td><td>.754±.005</td><td>.883±.021</td><td>3.414±.020</td><td>2.703±.154</td><td>.621±.003</td></tr><tr><td>ReMoDiffuse (Zhang et al., 2023b) MSFLOW (263D)</td><td rowspan="3"></td><td></td><td>.764±.003</td><td>.853±.003</td><td>.046±.004</td><td>2.890±.010</td><td>1.450±.079</td><td>.686±.000</td></tr><tr><td>MSFLOW (XYZ)</td><td>.566±.004</td><td>.759±.003</td><td>.849±.003</td><td>.038±.004</td><td>2.916±.008</td><td>1.518±.084</td><td>.676±.001</td></tr></table>

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

Dataset and representations. We conduct experiments on HumanML3D (Guo et al., 2022a), using sequences of at most 192 frames at 20 frames per second. The primary model generates normalized 263-dimensional HumanML3D features. The XYZ variant generates shared-axis-normalized positions of 22 joints, flattened to 66 values per frame. Following previous methods (Meng et al., 2025b;a; Yu et al., 2026), both variants are evaluated with the standard 67-dimensional root and rotation-invariant joint-position evaluator introduced in (Meng et al., 2025b). XYZ outputs are converted to this space through the HumanML3D post-processing pipeline. To test broader applicability, we additionally train MSFLOW on the 272D HumanML3D representation used by MotionStreamer (Xiao et al., 2025) and on the SnapMoGen dataset using its native 296D representation (Guo et al., 2025). Full protocols and results appear in Appendix A.5 and A.6, respectively.

Evaluation metrics. Following prior works, we report motion-embedding Frechet distance (FID),´ R-Precision at Top 1–3, matching distance (MM-Dist), multimodality (MModality), and motion– language cosine similarity (CLIP score). Each generation result averages 10 stochastic evaluations.

Implementation details. The 68M-parameter transformer uses eight RA-MMDiT blocks, source scale s = 5, and 50-step flow sampling. Full training and inference details appear in Appendix C.

## 4.2 QUANTITATIVE TEXT-TO-MOTION RESULTS

Table 1 compares the incremental 263D and global XYZ variants of MSFLOW with existing methods using the same 67-dimensional evaluator. The global variant achieves the best FID of 0.038±0.004, whereas the incremental variant attains the strongest point estimates for R-Precision, MM-Dist, and CLIP score. Relative to CMDM, the strongest prior method on most metrics, the global variant reduces FID from 0.078 to 0.038 (51.3%), while the incremental variant improves Top-1/2/3 R Precision by 0.008/0.005/0.004, reduces MM-Dist by 0.030, and increases CLIP by 0.001. Thus, we demonstrate that MSFLOW improves both fidelity and text alignment without a learned latent representation. Table 2 shows that MSFLOW also generalizes to SnapMoGen: it improves Top-1/3 R-Precision over CMDM from .831/.958 to .910/.984, while retaining a competitive FID of 16.342 and strong multimodality of 12.538. Complete metrics and baselines appear in Appendix A.6.

## 4.3 ANALYSIS OF REPRESENTATION-AWARE DESIGN CHOICES

Temporal attention depends on the motion representation. Table 3 (full version in Appendix A.3) shows that causal attention reduces 263D FID from 0.067 to 0.046, whereas bidirectional attention reduces XYZ FID from 1.563 to 0.038. This reversal supports a filtering–smoothing account: causality biases incremental features toward forward-consistent evolution, while bidirec tional context helps absolute coordinates maintain whole-trajectory consistency. We interpret this as a finite-model inductive bias rather than a universal advantage of restricted context; formal risk and dynamical analyses appear in Appendix E.3. Appendix E.4 analyzes the attention patterns for both models and provides attention-routing evidence for this interpretation.

Table 2: Text-to-motion results on Snap-MoGen. Bold and underline mark the best and second-best generated results.
<table><tr><td>Method</td><td>Rep.</td><td>| Top 1↑ Top 3↑</td><td></td><td>FID↓</td><td>MModality↑</td></tr><tr><td>GT</td><td>一</td><td>.940</td><td>.985</td><td>.001</td><td>一</td></tr><tr><td>T2M-GPT MoMask MoMask++ ScaleMoGen</td><td>Disc.</td><td>.618 .777 .802</td><td>.812 .927 .938</td><td>32.629 17.404 15.061</td><td>9.172 8.183 7.259</td></tr><tr><td>StableMoFusion MARDM MotionStreamer CMDM</td><td>Conti.</td><td>.679 .648 .631 .831</td><td>.888 .856 .836 .958</td><td>27.801 26.348 30.023</td><td>9.064 9.883 7.543</td></tr><tr><td>MDM MSFLOW</td><td>Raw</td><td>.503 .910</td><td>.727 .984</td><td>57.783 16.342</td><td>9.521 13.412 12.538</td></tr></table>

Table 3: Representation-aware design analysis on HumanML3D. Bold marks the better result in each pair; gray marks the proposed settings.
<table><tr><td>Factor</td><td>Rep.</td><td>Configuration</td><td>|Top 3↑</td><td>FID↓</td><td>MM-Dist↓</td></tr><tr><td rowspan="4">Attention</td><td rowspan="2">263D</td><td>Causal, s = 5</td><td>.853</td><td>.046</td><td>2.890</td></tr><tr><td>Bidirectional, s = 5</td><td>.846</td><td>.067</td><td>2.973</td></tr><tr><td rowspan="2">XYZ</td><td>Causal, s = 5</td><td>.800</td><td>1.563</td><td>3.243</td></tr><tr><td>Bidirectional, s = 5</td><td>.849</td><td>.038</td><td>2.916</td></tr><tr><td rowspan="3">Scale</td><td rowspan="2">263D</td><td>Causal, s = 1</td><td>.861</td><td>.111</td><td>2.851</td></tr><tr><td>Causal, s = 5</td><td>.853</td><td>.046</td><td>2.890</td></tr><tr><td rowspan="2">XYZ</td><td>Bidirectional, s = 1</td><td>.841</td><td>.144</td><td>2.999</td></tr><tr><td></td><td>Bidirectional, s = 5</td><td>.849</td><td>.038</td><td>2.916</td></tr><tr><td rowspan="3">Target</td><td rowspan="2">263D</td><td>Causal, x-pred.</td><td>.853</td><td>.046</td><td>2.890</td></tr><tr><td>Causal, v-pred.</td><td>.851</td><td>.061</td><td>2.900</td></tr><tr><td rowspan="2">XYZ</td><td>Bidirectional, x-pred.</td><td>.849</td><td>.038</td><td>2.916</td></tr><tr><td>Bidirectional, v-pred.</td><td></td><td>.831</td><td>.249</td><td>3.008</td></tr></table>

Source scaling substantially affects FID. Increasing s from 1 to 5 reduces FID from 0.111 to 0.046 for causal 263D generation, with small trade-offs in retrieval performance and MM-Dist, while substantially improving every reported metric for bidirectional XYZ generation. Proposition 2 explains why: a larger s improves path isotropy, thereby mitigating anisotropy in motion space by delaying signal emergence. Appendix E.2 verifies the underlying correlation structure of each representation and quantifies the resulting midpoint SNR and correlation retention. Overall, these results show that the source scale should be selected for the representation rather than inherited from a VAE latent space, making it an important design choice for motion-space generation.

Clean-motion prediction improves distributional fidelity. Clean-motion prediction outperforms direct v-prediction on every reported metric in all four matched comparisons. FID improves from 0.061 to 0.046 for causal 263D generation and from 0.249 to 0.038 for bidirectional XYZ generation. These results show that clean-motion prediction consistently enhances distributional fidelity in high-dimensional motion spaces.

## 4.4 INFERENCE-TIME SPATIAL CONTROL

Following the 67-dimensional setting used by ProjFlow (Watanabe et al., 2026), we adopt the OmniControl (Xie et al., 2024) protocol with six controlled joints: the pelvis, left foot, right foot, head, left wrist, and right wrist. The protocol evaluates five control densities, corresponding to 1, 2, 5, 49, and 196 keyframes, and averages each metric across these densities. We report FID, Top-3 R-Precision, diversity, foot-skating ratio, trajectory error, location error, and average control error. Table 4 shows that MSFLOW exactly satisfies all evaluated constraints without control-conditioned training. In the all-joints setting, it improves the zero-shot ProjFlow baseline from 0.097 to 0.061 FID and from 0.779 to 0.818 R-Precision@3. Per-joint results appear in Appendix B.

## 4.5 QUALITATIVE RESULTS

In Fig. 3, we show control over local limb placement, forward locomotion, circular motion, and a cartwheel. In each case, the controlled joints satisfy the constraints exactly, while the unconstrained motion remains coherent and aligned with the prompt. Appendix A.1 and our project website provides additional text-to-motion comparisons, and the supplementary videos visualize the generated motions.

Table 4: Results of text-conditioned motion generation with spatial controls on HumanML3D. The first block trains and evaluates on pelvis controls. The second trains on all joints and reports the average over separately controlled joints; complete per-joint results appear in Table 11. Bold and underline denote the best and second-best values, respectively.
<table><tr><td>Controlling Joint</td><td>Methods</td><td>Zero-shot?</td><td>FID↓</td><td>R-Precision Top 3</td><td>Diversity→</td><td>Foot Skating Ratio.↓</td><td>Traj. err.↓</td><td>Loc. err.↓</td><td>Avg. err.↓</td></tr><tr><td rowspan="9"></td><td>GT</td><td></td><td>0.000</td><td>0.795</td><td>10.455</td><td></td><td>0.000</td><td>0.000</td><td>0.000</td></tr><tr><td>MDM (Tevet et al., 2023)</td><td>√×V×××√</td><td>1.792</td><td>0.673</td><td>9.131</td><td>0.1019</td><td>0.4022</td><td>0.3076</td><td>0.5959</td></tr><tr><td>PriorMDM (Shafir et al., 2024)</td><td></td><td>0.393</td><td>0.707</td><td>9.847</td><td>0.0897</td><td>0.3457</td><td>0.2132</td><td>0.4417</td></tr><tr><td>GMD (Karunratanakul et al., 2023)</td><td></td><td>0.238</td><td>0.763</td><td>10.011</td><td>0.1009</td><td>0.0931</td><td>0.0321</td><td>0.1439</td></tr><tr><td>OmniControl (Xie et al., 2024)</td><td></td><td>0.081</td><td>0.789</td><td>10.323</td><td>0.0547</td><td>0.0387</td><td>0.0096</td><td>0.0338</td></tr><tr><td>MotionLCM V2+CtrlNet (Dai et al., 2024)</td><td></td><td>3.978</td><td>0.738</td><td>9.249</td><td>0.0901</td><td>0.1080</td><td>0.0581</td><td>0.1386</td></tr><tr><td>MaskControl (Pinyoanuntapong et al., 2025)</td><td></td><td>0.066</td><td>0.799 0.784</td><td>10.474</td><td>0.0543</td><td>0.0000</td><td>0.0000</td><td>0.0093</td></tr><tr><td>ProjFlow (Watanabe et al., 2026) MŠFLOW</td><td></td><td>0.107 0.068</td><td>0.821</td><td>10.644</td><td>0.0629</td><td>0.0000 0.0000</td><td>0.0000 0.0000</td><td>0.0000</td></tr><tr><td>OmniControl (Xie et al., 2024)</td><td>√</td><td></td><td></td><td>10.447</td><td>0.0591</td><td></td><td></td><td>0.0000</td></tr><tr><td rowspan="5">All Joints (Average)</td><td>MotionLCM V2+CtrlNet (Dai et al., 2024)</td><td></td><td>0.126</td><td>0.792</td><td>10.276</td><td>0.0608</td><td>0.0617</td><td>0.0107</td><td>0.0404</td></tr><tr><td>MaskControl (Pinyoanuntapong et al., 2025)</td><td></td><td>4.504</td><td>0.715</td><td>9.230</td><td>0.1119</td><td>0.2740</td><td>0.1315</td><td>0.2464</td></tr><tr><td></td><td>×××√</td><td>0.095 0.097</td><td>0.795</td><td>10.159</td><td>0.0545</td><td>0.0000</td><td>0.0000 0.0000</td><td>0.0065</td></tr><tr><td>ProjFlow (Watanabe et al., 2026)</td><td></td><td></td><td>0.779</td><td>10.651</td><td>0.0603</td><td>0.0000</td><td>0.0000</td><td>0.0000</td></tr><tr><td>MŚFLOW</td><td>V</td><td>0.061</td><td>0.818</td><td>10.593</td><td>0.0578</td><td>0.0000</td><td></td><td>0.0000</td></tr></table>

![](images/bb46e48fda5d77d6410a6c53e9235c5ac0b28dc7c331ddacc67b14f2234887f6.jpg)  
Figure 3: Qualitative inference-time spatial control. Colored markers indicate spatial constraints on selected joints and frames. MSFLOW follows the text prompts while satisfying these constraints.

## 5 LIMITATIONS

Although MSFLOW achieves state-of-the-art text-to-motion performance and exact spatial control, several limitations remain. First, our experiments cover only HumanML3D and SnapMoGen, and whether the observed interplay among representation, source scale, and temporal attention holds across other datasets, skeletons, and motion domains remains to be evaluated. Second, operating directly on full-resolution motion preserves frame-level access but requires processing longer token sequences than temporally compressed latent models, which may limit efficiency for very long motions. Hierarchical or streaming direct-space models could improve scalability while retaining fine-grained control. Finally, our projection sampler enforces linear equality constraints on joint coordinates, but does not explicitly guarantee physical feasibility or support non-linear constraints such as collision avoidance, contact, and joint limits. Extending direct-space generation with physical priors and more expressive constraint solvers is an important direction for future work.

## 6 CONCLUSION

In this paper, we presented MSFLOW, a representation-aware framework for flow matching directly in continuous motion space. MSFLOW, which predicts clean motion without a learned encoder or decoder, introduces representation-aware noise scaling for anisotropic direct-space probability paths and uses RA-MMDiT to adapt temporal information flow to the motion representation. Experiments on HumanML3D and SnapMoGen demonstrate state-of-the-art text-to-motion performance and show that causal attention is favored for incremental 263D features, whereas bidirectional attention is essential for absolute XYZ coordinates. The XYZ variant further enables training-free any-joint, any-frame projection with exact constraint satisfaction.

## REFERENCES

Michael S Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. In ICLR, 2023.

Michael S Albergo, Nicholas M Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and diffusions. arXiv preprint arXiv:2303.08797, 2023.

Nikos Athanasiou, Alpar´ Ce<sup>ˇ</sup> ske, Markos Diomataris, Michael J. Black, and Gˇ ul Varol. Motionfix:¨ Text-driven 3d human motion editing. In SIGGRAPH Asia, 2024.

Yiyi Cai, Yuhan Wu, Kunhang Li, You Zhou, Bo Zheng, and Haiyang Liu. Flooddiffusion: Tailored diffusion forcing for streaming motion generation. In CVPR, 2026.

Xin Chen, Biao Jiang, Wen Liu, Zilong Huang, Bin Fu, Tao Chen, and Gang Yu. Executing your commands via motion diffusion in latent space. In CVPR, 2023.

Xiaoyan Cong, Zekun Li, Zhiyang Dou, Hongyu Li, Omid Taheri, Chuan Guo, Abhay Mittal, Sizhe An, Taku Komura, Wojciech Matusik, Michael J. Black, and Srinath Sridhar. Umo: Unified in-context learning unlocks motion foundation model priors. arXiv preprint arXiv:2603.15975, 2026.

Rishabh Dabral, Muhammad Hamza Mughal, Vladislav Golyanik, and Christian Theobalt. Mofusion: A framework for denoising-diffusion-based motion synthesis. In CVPR, 2023.

Wenxun Dai, Ling-Hao Chen, Jingbo Wang, Jinpeng Liu, Bo Dai, and Yansong Tang. Motionlcm: Real-time controllable motion generation via latent consistency model. In ECCV, 2024.

Kent Fujiwara, Mikihiro Tanaka, and Qing Yu. Chronologically accurate retrieval for temporal grounding of motion-language models. In ECCV, 2024.

Anindita Ghosh, Noshaba Cheema, Cennet Oguz, Christian Theobalt, and Philipp Slusallek. Synthesis of compositional animations from textual descriptions. In ICCV, 2021.

Chuan Guo, Xinxin Zuo, Sen Wang, Shihao Zou, Qingyao Sun, Annan Deng, Minglun Gong, and Li Cheng. Action2motion: Conditioned generation of 3d human motions. In ACM MM, 2020.

Chuan Guo, Shihao Zou, Xinxin Zuo, Sen Wang, Wei Ji, Xingyu Li, and Li Cheng. Generating diverse and natural 3d human motions from text. In CVPR, 2022a.

Chuan Guo, Xinxin Zuo, Sen Wang, and Li Cheng. Tm2t: Stochastic and tokenized modeling for the reciprocal generation of 3d human motions and texts. In ECCV, 2022b.

Chuan Guo, Yuxuan Mu, Muhammad Gohar Javed, Sen Wang, and Li Cheng. Momask: Generative masked modeling of 3d human motions. In CVPR, 2024.

Chuan Guo, Inwoo Hwang, Jian Wang, and Bing Zhou. Snapmogen: Human motion generation from expressive texts. In NeurIPS, 2025.

Anne Harrington, A. Sophia Koepke, Shyamgopal Karthik, Trevor Darrell, and Alexei A. Efros. It’s never too late: Noise optimization for collapse recovery in trained diffusion models. In CVPR, 2026.

Karl Heun. Neue methode zur approximativen integration der differentialgleichungen einer unabhangigen ver¨ anderlichen.¨ Z. Math. Phys, 1900.

Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020.

Seokhyeon Hong, Chaelin Kim, Serin Yoon, Junghyun Nam, Sihun Cha, and Junyong Noh. Salad: Skeleton-aware latent diffusion for text-driven motion generation and editing. In CVPR, 2025.

Yiheng Huang, Hui Yang, Chuanchen Luo, Yuxi Wang, Shibiao Xu, Zhaoxiang Zhang, Man Zhang, and Junran Peng. Stablemofusion: Towards robust and efficient diffusion-based motion generation framework. In ACMMM, 2024a.

Yiming Huang, Weilin Wan, Yue Yang, Chris Callison-Burch, Mark Yatskar, and Lingjie Liu. Como: Controllable motion generation through language guided pose code editing. In ECCV, 2024b.

Inwoo Hwang, Hojun Jang, Bing Zhou, Jian Wang, Young Min Kim, and Chuan Guo. Scalemogen: Autoregressive next-scale prediction for human motion generation. In ECCV, 2026.

Biao Jiang, Xin Chen, Wen Liu, Jingyi Yu, Gang Yu, and Tao Chen. Motiongpt: Human motion as a foreign language. In NeurIPS, 2023.

Korrawe Karunratanakul, Konpat Preechakul, Supasorn Suwajanakorn, and Siyu Tang. Guided motion diffusion for controllable human motion synthesis. In ICCV, 2023.

Diederik P. Kingma and Max Welling. Auto-encoding variational bayes. In ICLR, 2014.

Tianhong Li and Kaiming He. Back to basics: Let denoising generative models denoise. In CVPR, 2026.

Yuan-Ming Li, Qize Yang, Nan Lei, Shenghao Fu, Ling-An Zeng, Jian-Fang Hu, Xihan Wei, and Wei-Shi Zheng. Irg-motionllm: Interleaving motion generation, assessment and refinement for text-to-motion generation. In ECCV, 2026.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In ICLR, 2023.

Hanchao Liu, Xiaohang Zhan, Shaoli Huang, Tai-Jiang Mu, and Ying Shan. Programmable motion generation for open-set motion control tasks. In CVPR, 2024.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2019.

Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In ECCV, 2024.

Jiafeng Mao, Xueting Wang, and Kiyoharu Aizawa. The lottery ticket hypothesis in denoising: Towards semantic-driven initialization. In ECCV, 2024.

Zichong Meng, Zeyu Han, Xiaogang Peng, Yiming Xie, and Huaizu Jiang. Absolute coordinates make motion generation easy. arXiv preprint arXiv:2505.19377, 2025a.

Zichong Meng, Yiming Xie, Xiaogang Peng, Zeyu Han, and Huaizu Jiang. Rethinking diffusion for text-driven human motion generation. In CVPR, 2025b.

Sakuya Ota, Qing Yu, Kent Fujiwara, Satoshi Ikehata, and Ikuro Sato. Retrieving and refining winning noise tickets for diffusion-based motion generation. In ECCV, 2026.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023.

Mathis Petrovich, Michael J Black, and Gul Varol. Temos: Generating diverse human motions from¨ textual descriptions. In ECCV, 2022.

Mathis Petrovich, Michael J Black, and Gul Varol. Tmr: Text-to-motion retrieval using contrastive¨ 3d human motion synthesis. In ICCV, 2023.

Mathis Petrovich, Or Litany, Umar Iqbal, Michael J. Black, Gul Varol, Xue Bin Peng, and Davis ¨ Rempe. Multi-track timeline control for text-driven 3d human motion generation. In CVPR-W, 2024.

Ekkasit Pinyoanuntapong, Pu Wang, Minwoo Lee, and Chen Chen. Mmm: Generative masked motion model. In CVPR, 2024.

Ekkasit Pinyoanuntapong, Muhammad Usama Saleem, Korrawe Karunratanakul, Pu Wang, Hongfei Xue, Chen Chen, Chuan Guo, Junli Cao, Jian Ren, and Sergey Tulyakov. Maskcontrol: Spatiotemporal control for masked motion synthesis. In ICCV, 2025.

Davis Rempe, Mathis Petrovich, Ye Yuan, Haotian Zhang, Xue Bin Peng, Yifeng Jiang, Tingwu Wang, Umar Iqbal, David Minor, Michael de Ruyter, Jiefeng Li, Chen Tessler, Edy Lim, Eugene Jeong, Sam Wu, Ehsan Hassani, Michael Huang, Jin-Bey Yu, Chaeyeon Chung, Lina Song, Olivier Dionne, Jan Kautz, Simon Yuen, and Sanja Fidler. Kimodo: Scaling controllable human motion generation. arXiv preprint arXiv:2603.15546, 2026.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-¨ resolution image synthesis with latent diffusion models. In CVPR, 2022.

Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. In NeurIPS-W, 2019.

Florian Schmidt. Generalization in generation: A closer look at exposure bias. In EMNLP-WNGT, 2019.

Yoni Shafir, Guy Tevet, Roy Kapon, and Amit Haim Bermano. Human motion diffusion as a generative prior. In ICLR, 2024.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In ICLR, 2021.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 2024.

Guy Tevet, Brian Gordon, Amir Hertz, Amit H Bermano, and Daniel Cohen-Or. Motionclip: Exposing human motion generation to clip space. In ECCV, 2022.

Guy Tevet, Sigal Raab, Brian Gordon, Yonatan Shafir, Daniel Cohen-Or, and Amit H. Bermano. Human motion diffusion model. In ICLR, 2023.

Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural discrete representation learning. In NeurIPS, 2017.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017.

Ruoyu Wang, Huayang Huang, Ye Zhu, Olga Russakovsky, and Yu Wu. The silent assistant: Noisequery as implicit guidance for goal-driven image generation. In ICCV, 2025.

Akihisa Watanabe, Qing Yu, Edgar Simo-Serra, and Kent Fujiwara. Projflow: Projection sampling with flow matching for zero-shot exact spatial motion control. In CVPR, 2026.

Yuxin Wen, Qing Shuai, Di Kang, Jing Li, Cheng Wen, Yue Qian, Ningxin Jiao, Changhai Chen, Weijie Chen, Yiran Wang, Jinkun Guo, Dongyue An, Han Liu, Yanyu Tong, Chao Zhang, Qing Guo, Juan Chen, Qiao Zhang, Youyi Zhang, Zihao Yao, Cheng Zhang, Hong Duan, Xiaoping Wu, Qi Chen, Fei Cheng, Liang Dong, Peng He, Hao Zhang, Jiaxin Lin, Chao Zhang, Zhongyi Fan, Yifan Li, Zhichao Hu, Yuhong Liu, Linus, Jie Jiang, Xiaolong Li, and Linchao Bao. Hy-motion 1.0: Scaling flow matching models for text-to-motion generation. arXiv preprint arXiv:2512.23464, 2025.

Lixing Xiao, Shunlin Lu, Huaijin Pi, Ke Fan, Liang Pan, Yueer Zhou, Ziyong Feng, Xiaowei Zhou, Sida Peng, and Jingbo Wang. Motionstreamer: Streaming motion generation via diffusion-based autoregressive model in causal latent space. In ICCV, 2025.

Yiming Xie, Varun Jampani, Lei Zhong, Deqing Sun, and Huaizu Jiang. Omnicontrol: Control any joint at any time for human motion generation. In ICLR, 2024.

Sijie Yan, Zhizhong Li, Yuanjun Xiong, Huahan Yan, and Dahua Lin. Convolutional sequence generation for skeleton-based action synthesis. In ICCV, 2019.

Jingfeng Yao, Bin Yang, and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. In CVPR, 2025.

Qing Yu, Mikihiro Tanaka, and Kent Fujiwara. Exploring vision transformers for 3d human motionlanguage models with motion patches. In CVPR, 2024.

Qing Yu, Mikihiro Tanaka, and Kent Fujiwara. Remogpt: Part-level retrieval-augmented motionlanguage models. In AAAI, 2025.

Qing Yu, Akihisa Watanabe, and Kent Fujiwara. Causal motion diffusion models for autoregressive motion generation. In CVPR, 2026.

Jianrong Zhang, Yangsong Zhang, Xiaodong Cun, Shaoli Huang, Yong Zhang, Hongwei Zhao, Hongtao Lu, and Xi Shen. Generating human motion from textual descriptions with discrete representations. In CVPR, 2023a.

Jianrong Zhang, Hehe Fan, and Yi Yang. Energymogen: Compositional human motion generation with energy-based diffusion model in latent space. In CVPR, 2025.

Mingyuan Zhang, Xinying Guo, Liang Pan, Zhongang Cai, Fangzhou Hong, Huirong Li, Lei Yang, and Ziwei Liu. Remodiffuse: Retrieval-augmented motion diffusion model. In ICCV, 2023b.

Mingyuan Zhang, Zhongang Cai, Liang Pan, Fangzhou Hong, Xinying Guo, Lei Yang, and Ziwei Liu. Motiondiffuse: Text-driven human motion generation with diffusion model. IEEE TPAMI, 2024.

Kaifeng Zhao, Gen Li, and Siyu Tang. Dartcontrol: A diffusion-based autoregressive motion model for real-time text-driven motion control. In ICLR, 2025.

Rui Zhao, Hui Su, and Qiang Ji. Bayesian adversarial human motion synthesis. In CVPR, 2020.

Boyang Zheng, Nanye Ma, Shengbang Tong, and Saining Xie. Diffusion transformers with representation autoencoders. In ICLR, 2026.

Chongyang Zhong, Lei Hu, Zihao Zhang, and Shihong Xia. Attt2m: Text-driven human motion generation with multi-perspective attention mechanism. In ICCV, 2023.

Zikai Zhou, Shitong Shao, Lichen Bai, Shufei Zhang, Zhiqiang Xu, Bo Han, and Zeke Xie. Golden noise for diffusion models: A learning framework. In ICCV, 2025.

Qiran Zou, Shangyuan Yuan, Shian Du, Yu Wang, Chang Liu, Yi Xu, Jie Chen, and Xiangyang Ji. Parco: Part-coordinating text-to-motion synthesis. In ECCV, 2024.

## SUPPLEMENTARY MATERIAL

This supplementary material provides additional text-to-motion and spatial-control results, implementation details for reproducing the principal 263D and XYZ models, accompanying sample code, and theoretical and empirical analyses supporting our design choices.

## SECTION INDEX

A Additional Text-to-Motion Results p. 14   
A.1 Qualitative Results p. 14   
A.2 Model Capacity and Text–Motion Attention p. 14   
A.3 Complete Representation-Aware Design Analysis p. 16   
A.4 Evaluation with the Standard 263D Protocol p. 16   
A.5 Training and Evaluation with the 272D MotionStreamer Protocol p. 16   
A.6 Training and Evaluation on SnapMoGen p. 17   
A.7 Compute Efficiency p. 18   
B Detailed Spatial-Control Results p. 19   
C Implementation and Reproducibility Details p. 19   
C.1 Causal 263D Text-to-Motion Configuration p. 20   
C.2 Bidirectional XYZ Text-to-Motion Configuration p. 20   
C.3 Training and Sampling Algorithms p. 21   
C.4 Bidirectional XYZ Endpoint-Projection Configuration p. 21   
D Sample Code p. 21   
E Theoretical Details and Proofs p. 22   
E.1 Proofs of Main Propositions p. 22   
E.2 Empirical Spatiotemporal Covariance and Source Scale p. 23   
E.3 Why Attention Depends on the Representation p. 25   
E.4 Empirical Attention Routing p. 26

## A ADDITIONAL TEXT-TO-MOTION RESULTS

## A.1 QUALITATIVE RESULTS

Fig. 4 compares ACMDM, SALAD, and the global-XYZ and incremental-263D variants of MS-FLOW on six prompts that probe action count, limb specificity, directed reaching, trajectory reversal, lateral direction, and ground-contact transitions. The red annotations identify visible baseline failures, such as repeating an action, moving the wrong limb or direction, omitting a return trajectory, and floating feet. Both MSFLOW variants more consistently preserve the highlighted prompt details while producing coherent motion. Because static trajectory overlays cannot fully convey timing and transition quality, we recommend viewing the accompanying supplementary videos on our project website.

## A.2 MODEL CAPACITY AND TEXT–MOTION ATTENTION

We ablate model capacity, the flow-time-conditioned Token Refiner, and the attention between text and motion while retaining the selected source scale, prediction target, and representation-specific temporal masks. Table 5 reports the complete results for both the causal 263D and bidirectional XYZ generators.

The four-block model is consistently under-capacity: relative to the selected eight-block model, its FID increases from .046 to .206 for 263D and from .038 to .098 for XYZ, accompanied by weaker retrieval and matching scores. Scaling from eight to 12 blocks does not yield a consistent gain. The 12-block 263D model has worse FID, R-Precision, and MM-Dist, while its XYZ counterpart is statistically similar in FID (.039 ± .003 versus .038 ± .004) and only improves the CLIP point estimate. We therefore retain the eight-block, width-512 backbone as the best capacity–quality trade-off.

The component ablation reveals a fidelity–retrieval trade-off. Removing the Token Refiner from the multimodal-attention model leaves 263D FID nearly unchanged but weakens its text-alignment metrics; for XYZ it also increases FID from .038 to .045. Replacing multimodal-attention RA-MMDiT blocks with conventional motion self-attention and text cross-attention is competitive for

Two squats SALAD

A man performs a squat and raises both arms forward.

![](images/df0a1b51466264fdd6b7108f9aa85d9b3a44cde479ee8384dd8ee587dcd1be8d.jpg)  
Two squats ACMDM

![](images/c41d95b53238aca111f70823cd2dac4eef1577bd62f25825e1276627164fd8b6.jpg)

![](images/ae3ae119174e868c2c35a04c86c8a23d1d8a1285c2ce217cc0c8974e5de6e80b.jpg)  
MSFlow (XYZ)

![](images/13323cad1fcdb708012c761cef44e7ca3552c1bb13c25946758114c0b73848d9.jpg)  
MSFlow (263D)  
A person stands with legs shoulder-width apart and knees slightly bent. With both arms at shoulder height, they lower the left arm, then raise it again.

A person stands up, reaches down with the right hand as if checking a phone, then places it back in a pocket.

Use both hands ACMDM

No reach down SALAD

![](images/7d054e4b8d15963b9a4ca9f830008831785c01ec3b8280e9ff88ebc199af2e3c.jpg)

![](images/c5cd803c8273cb3417f1374b180e4c672ad857e0349134be1e62cd2fa8b9ac3a.jpg)

![](images/880a553e160b6fe50c58f399020cd55eaab63abe21ff874f6e2cc5ebdac1040c.jpg)  
MSFlow (XYZ)

![](images/4ad47dfb761debc1465578f5c3b8e1f27ab9dd1cff9caf5207f02d4beb007ab2.jpg)  
MSFlow (263D)

![](images/bfb5580ae931d29faae908d37cfa2bbcc8d0c600dca42209bd18f42b361e9186.jpg)  
Both hands ACMDM

![](images/dc1a20a9b6bd3001da41a28b1d5c14564fa94e6e46eb3831ae884bd69225188d.jpg)  
SALAD  
MSFlow (XYZ)

![](images/341dece6ec635d540b2d9c633c86bb491fb28fb1b19aa6d83b6735f4f66dde55.jpg)

![](images/4ae13f36622d1cf4cc65ace03702e866e7fc1655108518cd0464ef127154109b.jpg)  
A person walks forward, turns around to the right, and returns.  
MSFlow (263D)

A person takes sideways steps to the right, then to the left  
![](images/848d1b8c44b75748e096c184801696656e8a821445b40196405f3adcc949d1cf.jpg)  
Not return ACMDM

![](images/2ee85304e087dd738c6220a473108e5af568cdc9b811a097a6c2bd2a7a5d785f.jpg)

![](images/e3d0a9f75ea2a772c332d261b26ebfc2d33f36299458d8093c1bb6f17e2e4007.jpg)  
MSFlow (XYZ)

![](images/b6018b293cefcf2984ff3ec207a3804026b9551c2dc45b00da32d48690c659e3.jpg)  
MSFlow (263D)

![](images/57803663fe642b6c528362dcb91c55cede256f57b6a55eb84c2b68417865f91c.jpg)  
Wrong direction ACMDM

![](images/7c0c0a84ee094e10b7c7b6ee98c2a6f26f3c29c9e28d33ad73f02273aa4283f9.jpg)  
Wrong direction SALAD

![](images/e7f7f5595bd7787cbf53467269d96ed07e264608727014db22af26cee2a2341d.jpg)  
MSFlow (XYZ)

![](images/43c5c2040375591505327370059e76d1ae8277d0094bd4f1c9db4e0e85cec5d0.jpg)  
MSFlow (263D)

![](images/7c4fb81de5cff837d236f6d46a8d6190fe8298f8a5f4587a6f17e292622ac755.jpg)  
Feet floating ACMDM  
A person lies down, then gets up.

![](images/4d78dde3eabceed90e8f44df21e2f5f242e9f141c7fb63f5388a8f541875373c.jpg)  
Feet floating SALAD

![](images/1d73e1778208fa7d7a911bfa90e31f4791b22a6a83dc53791d87bbd9f100d7b3.jpg)  
MSFlow (XYZ)

![](images/82a3b107d6c94e644ece40a8f07fbb0c329f2b97d13289e1fce588f5ec16e349.jpg)  
MSFlow (263D)

Figure 4: Qualitative text-to-motion comparisons on HumanML3D. We compare ACMDM, SALAD, and the global-XYZ and incremental-263D variants of MSFLOW across prompts requiring fine-grained action semantics, directional motion, and contact fidelity. Red annotations identify visible failure modes in the baselines; both MSFLOW variants more faithfully follow the highlighted prompt details. For the clearest visualization of these dynamic behaviors, please view the accompanying supplementary videos on our project website.  
Table 5: Model-capacity and text–motion attention ablations on HumanML3D. All models use clean-motion prediction, source scale s = 5, classifier-free guidance weight 3, and the representation-aware temporal mask (causal for 263D and bidirectional for XYZ). MM denotes RA-MMDiT blocks that update motion and text through multimodal-attention; Cross denotes conventional motion self-attention followed by text cross-attention. Token Refiner denotes the two-layer, flow-time-conditioned module applied to text tokens. Each interval summarizes 10 stochastic evaluations. Bold denotes the best point estimate for each metric within each representation and ablation group. Gray cells mark the proposed setting.
<table><tr><td rowspan="2">Factor</td><td colspan="4">Configuration</td><td colspan="3">R-Precision↑</td><td rowspan="2">FID↓</td><td rowspan="2">MM-Dist↓</td><td rowspan="2">CLIP↑</td></tr><tr><td></td><td></td><td>Rep. Attention Blocks/width Token Refiner</td><td></td><td>Top 1</td><td>Top 2</td><td>Top 3</td></tr><tr><td rowspan="6">Capacity</td><td rowspan="3">263D</td><td>MM</td><td>4/256</td><td>V</td><td>.555±.003</td><td>.756±.003</td><td>.848±.003 .853±.003</td><td>.206±.008</td><td>2.940±.011</td><td>.678±.001</td></tr><tr><td>MM</td><td>8/512</td><td>√</td><td>.571±.005</td><td>.764±.003</td><td></td><td>.046±.004</td><td>2.890±.010</td><td>.686±.000</td></tr><tr><td>MM</td><td>12/768</td><td>√</td><td>.561±.004</td><td>.758±.003</td><td>.847±.003</td><td>.076±.006</td><td>2.925±.012</td><td>.681±.001</td></tr><tr><td rowspan="3">XYZ</td><td>MM</td><td>4/256</td><td>√</td><td>.549±.004</td><td>.750±.003</td><td>.845±.003</td><td>|.098±.006</td><td>3.000±.014</td><td>.658±.001</td></tr><tr><td>MM</td><td>8/512</td><td>√</td><td>.566±.004</td><td>.759±.003</td><td>.849±.003</td><td>.038±.004</td><td>2.916±.008</td><td>.676±.001</td></tr><tr><td>MM</td><td>12/768</td><td>√</td><td>.564±.003</td><td>.760±.003</td><td>.849±.002</td><td>.039±.003</td><td>2.922±.006</td><td>.680±.001</td></tr><tr><td rowspan="7">Attention/Token Refiner</td><td rowspan="5">263D</td><td>MM</td><td>8/512</td><td>√</td><td>.571±.005</td><td>.764±.003</td><td>.853±.003</td><td>.046±.004</td><td>2.890±.010</td><td>.686±.000</td></tr><tr><td>MM</td><td>8/512</td><td>x</td><td>.568±.003</td><td>.761±.003</td><td>.848±.002</td><td>.043±.003</td><td>2.922±.007</td><td>.679±.001</td></tr><tr><td>Cross</td><td>8/512</td><td>√</td><td>.572±.005</td><td>.762±.004</td><td>.854±.003</td><td>.044±.003</td><td>2.880±.005</td><td>.686±.001</td></tr><tr><td>Cross</td><td>8/512</td><td>x</td><td>.581±.005</td><td>.776±.004</td><td>.865±.001</td><td>.099±.006</td><td>2.822±.010</td><td>.684±.001</td></tr><tr><td>MM</td><td>8/512</td><td>√</td><td>.566±.004</td><td>.759±.003</td><td>.849±.003</td><td>.038±.004</td><td>2.916±.008</td><td>.676±.001</td></tr><tr><td>MM</td><td>8/512</td><td>x</td><td>.560±.004</td><td>.754±.004</td><td>.844±.002</td><td>.045±.005</td><td>2.942±.010</td><td>.675±.001</td></tr><tr><td rowspan="4">XYZ</td><td>Cross</td><td>8/512</td><td>√</td><td>.556±.006</td><td>.756±.004</td><td>.852±.002</td><td>.079±.003</td><td>2.954±.008</td><td>.677±.001</td></tr><tr><td>Cross</td><td>8/512</td><td>x</td><td>.568±.003</td><td>.766±.003</td><td>.857±.002</td><td>.104±.004</td><td>2.909±.011</td><td>.676±.001</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

causal 263D when the Token Refiner is retained, but raises XYZ FID from .038 to .079. Without the Token Refiner, the cross-attention models obtain the strongest R-Precision and MM-Dist point estimates, yet their FID degrades to .099 for 263D and .104 for XYZ. Thus, retrieval alone favors the simpler cross-attention variant, whereas multimodal-attention with flow-time-aware text refinement gives the most robust distributional fidelity across both motion representations and is the configuration used by our primary models.

Table 6: Complete analysis of representation-aware design choices. All configurations use s = 5 and clean-motion prediction unless stated otherwise. Each interval summarizes 10 stochastic evaluations. Bold denotes the best result for each metric within each group, and gray cells mark the proposed settings.
<table><tr><td rowspan="2">Factor</td><td rowspan="2">Representation</td><td rowspan="2">Configuration</td><td colspan="3">R-Precision↑</td><td rowspan="2">FID↓</td><td rowspan="2">MM-Dist↓</td></tr><tr><td>Top 1</td><td>Top 2</td><td>Top 3</td></tr><tr><td rowspan="4">Attention</td><td rowspan="2">263D</td><td>Causal</td><td> ${ \bf 5 7 1 ^ { \pm . 0 0 5 } }$ </td><td> $. 7 6 4 ^ { \pm . 0 0 3 }$ </td><td> ${ \bf 8 5 3 ^ { \pm . 0 0 3 } }$ </td><td> $\mathbf { 0 4 6 ^ { \pm . 0 0 4 } }$ </td><td> $\mathbf { 2 . 8 9 0 ^ { \pm . 0 1 0 } }$ </td></tr><tr><td>Bidirectional</td><td> $. 5 5 8 ^ { \pm . 0 0 4 }$ </td><td> $. 7 5 6 ^ { \pm . 0 0 3 }$ </td><td> $. 8 4 6 ^ { \pm . 0 0 2 }$ </td><td> $. 0 6 7 ^ { \pm . 0 0 2 }$ </td><td> $2 . 9 7 3 ^ { \pm . 0 0 5 }$ </td></tr><tr><td rowspan="2">XYZ</td><td>Causal</td><td> $. 5 0 6 ^ { \pm . 0 0 4 }$ </td><td> $\overline { { . 7 0 1 ^ { \pm . 0 0 2 } } }$ </td><td> $\overline { { . 8 0 0 ^ { \pm . 0 0 2 } } }$ </td><td> $\overline { { 1 . 5 6 3 ^ { \pm . 0 2 5 } } }$ </td><td> $\overline { { 3 . 2 4 3 ^ { \pm . 0 0 8 } } }$ </td></tr><tr><td>Bidirectional</td><td> $. 5 6 6 ^ { \pm . 0 0 4 }$ </td><td> ${ \bf . 7 5 9 ^ { \pm . 0 0 3 } }$ </td><td> ${ \bf 8 4 9 ^ { \pm . 0 0 3 } }$ </td><td> ${ \bf . 0 3 8 ^ { \pm . 0 0 4 } }$ </td><td> $\mathbf { 2 . 9 1 6 ^ { \pm . 0 0 8 } }$ </td></tr><tr><td rowspan="4">Source scale</td><td rowspan="2">263D</td><td>s = 1, causal</td><td> ${ \bf 5 8 0 ^ { \pm . 0 0 4 } }$ </td><td> $. 7 7 5 ^ { \pm . 0 0 3 }$ </td><td> $\mathbf { 8 6 1 ^ { \pm . 0 0 2 } }$ </td><td> $. 1 1 1 ^ { \pm . 0 0 5 }$ </td><td> $\mathbf { 2 . 8 5 1 ^ { \pm . 0 0 7 } }$ </td></tr><tr><td>s = 5, causal</td><td> $. 5 7 1 ^ { \pm . 0 0 5 }$ </td><td> $. 7 6 4 ^ { \pm . 0 0 3 }$ </td><td> $. 8 5 3 ^ { \pm . 0 0 3 }$ </td><td> $\mathbf { . 0 4 6 ^ { \pm . 0 0 4 } }$ </td><td> $2 . 8 9 0 ^ { \pm . 0 1 0 }$ </td></tr><tr><td rowspan="2">XYZ</td><td>s = 1, bidirectional</td><td> $5 4 8 ^ { \pm . 0 0 2 }$ </td><td> $. 7 4 8 ^ { \pm . 0 0 3 }$ </td><td> $. 8 4 1 ^ { \pm . 0 0 3 }$ </td><td> $. 1 4 4 ^ { \pm . 0 0 9 }$ </td><td> $2 . 9 9 9 ^ { \pm . 0 0 8 }$ </td></tr><tr><td>s = 5, bidirectional</td><td> $. 5 6 6 ^ { \pm . 0 0 4 }$ </td><td> ${ \bf . 7 5 9 ^ { \pm . 0 0 3 } }$ </td><td> ${ \bf 8 4 9 ^ { \pm . 0 0 3 } }$ </td><td> ${ \bf . 0 3 8 ^ { \pm . 0 0 4 } }$ </td><td> $\mathbf { 2 . 9 1 6 ^ { \pm . 0 0 8 } }$ </td></tr><tr><td rowspan="8">Prediction target</td><td rowspan="4">263D</td><td>x-prediction, causal</td><td> ${ \bf 5 7 1 ^ { \pm . 0 0 5 } }$ </td><td> $. 7 6 4 ^ { \pm . 0 0 3 }$ </td><td> $\mathbf { . 8 5 3 ^ { \pm . 0 0 3 } }$ </td><td> $\mathbf { . 0 4 6 ^ { \pm . 0 0 4 } }$ </td><td> $\mathbf { 2 . 8 9 0 ^ { \pm . 0 1 0 } }$ </td></tr><tr><td>v-prediction, causal</td><td> $. 5 6 7 ^ { \pm . 0 0 3 }$ </td><td> $. 7 6 3 ^ { \pm . 0 0 3 }$ </td><td> $. 8 5 1 ^ { \pm . 0 0 3 }$ </td><td> $. 0 6 1 ^ { \pm . 0 0 4 }$ </td><td> $2 . 9 0 0 ^ { \pm . 0 1 1 }$ </td></tr><tr><td>x-prediction, bidirectional</td><td> $. 5 5 8 ^ { \pm . 0 0 4 }$ </td><td> $. 7 5 6 ^ { \pm . 0 0 3 }$ </td><td> $. 8 4 6 ^ { \pm . 0 0 2 }$ </td><td> $. 0 6 7 ^ { \pm . 0 0 2 }$ </td><td> $2 . 9 7 3 ^ { \pm . 0 0 5 }$ </td></tr><tr><td>v-prediction, bidirectional</td><td> $. 5 5 1 ^ { \pm . 0 0 4 }$ </td><td>.752±.003</td><td> $. 8 4 2 ^ { \pm . 0 0 2 }$ </td><td> $. 1 0 1 ^ { \pm . 0 0 4 }$ </td><td> $3 . 0 1 0 ^ { \pm . 0 0 9 }$ </td></tr><tr><td rowspan="4">XYZ</td><td>x-prediction, causal</td><td> $. 5 0 6 ^ { \pm . 0 0 4 }$ </td><td> $. 7 0 1 ^ { \pm . 0 0 2 }$ </td><td> $. 8 0 0 ^ { \pm . 0 0 2 }$ </td><td> $1 . 5 6 3 ^ { \pm . 0 2 5 }$ </td><td> $3 . 2 4 3 ^ { \pm . 0 0 8 }$ </td></tr><tr><td>v-prediction, causal</td><td> $. 4 5 0 ^ { \pm . 0 0 2 }$ </td><td> $. 6 3 7 ^ { \pm . 0 0 4 }$ </td><td> $. 7 4 4 ^ { \pm . 0 0 4 }$ </td><td> $4 . 4 8 2 ^ { \pm . 0 3 0 }$ </td><td> $3 . 6 7 2 ^ { \pm . 0 1 2 }$ </td></tr><tr><td>x-prediction, bidirectional</td><td> $. 5 6 6 ^ { \pm . 0 0 4 }$ </td><td> $. 7 5 9 ^ { \pm . 0 0 3 }$  </td><td> $\mathbf { . 8 4 9 ^ { \pm . 0 0 3 } }$ </td><td> ${ \bf . 0 3 8 ^ { \pm . 0 0 4 } }$ </td><td> $\mathbf { 2 . 9 1 6 ^ { \pm . 0 0 8 } }$ </td></tr><tr><td>v-prediction, bidirectional</td><td> $. 5 4 2 ^ { \pm . 0 0 4 }$ </td><td> $. 7 3 7 ^ { \pm . 0 0 2 }$ </td><td> $. 8 3 1 ^ { \pm . 0 0 3 }$ </td><td> $. 2 4 9 ^ { \pm . 0 1 4 }$ </td><td> $3 . 0 0 8 ^ { \pm . 0 1 0 }$ </td></tr></table>

## A.3 COMPLETE REPRESENTATION-AWARE DESIGN ANALYSIS

Table 6 reports all attention, source-scale, and prediction-target configurations, including Top-1/2/3 R-Precision, FID, MM-Dist, and 95% confidence intervals. The main paper retains the comparisons needed for its discussion in compact form.

## A.4 EVALUATION WITH THE STANDARD 263D PROTOCOL

We complement the 67D evaluation in the main paper with the standard 263D HumanML3D evaluator used by MDM (Tevet et al., 2023). Table 7 shows that the incremental variant achieves the best Top-1 and Top-3 R-Precision scores of 0.591 and 0.863, respectively, and the best MM-Dist of 2.565, while tying CMDM for the best Top-2 R-Precision of 0.778. The global variant obtains the second-best FID of 0.050 and the second-best MM-Dist of 2.607. Together, these results demon strate strong retrieval and distributional performance under both evaluator protocols.

## A.5 TRAINING AND EVALUATION WITH THE 272D MOTIONSTREAMER PROTOCOL

To compare under the protocol introduced by MotionStreamer (Xiao et al., 2025), we train a new causal MSFLOW model from scratch on the 272D HumanML3D representation used by Motion-Streamer, rather than converting or fine-tuning our 263D model. We use the proposed architecture with source scale s = 5 and causal attention because the 272D representation, like the 263D representation, encodes incremental motion. We evaluate the model using the same official 272D evaluator as MotionStreamer and UMO (Cong et al., 2026).

As shown in Table 8, the new 272D MSFLOW model improves FID from 11.790 for MotionStreamer and 9.460 for UMO-Unified to 6.540. It also increases Top-1/2/3 R-Precision to 0.792/0.919/0.955 and reduces MM-Dist to 13.864, outperforming both UMO variants on every reported metric. These comparisons use the same evaluator, although their generation spaces differ: MotionStreamer and our model are trained on 272D motion, whereas UMO generates a 201D HY-Motion representation and converts its output to 272D for evaluation.

Difference from the standard 263D representation. For K = 22 SMPL joints, the standard HumanML3D vector contains planar root velocity (2D), scalar yaw velocity (1D), root height (1D), root-relative positions for the K − 1 non-root joints $( 3 ( K - 1 ) \mathbf { D } )$ , local velocities for all joints (3KD), IK-derived 6D rotations for the non-root joints $( 6 ( K - 1 ) \dot { \mathrm { D } } )$ , and four foot-contact channels, totaling 263 dimensions. The 272D representation instead contains planar root velocity (2D), a 6D root angular increment, and positions, velocities, and 6D rotations for all K joints (3K + 3K + 6KD), totaling 272 dimensions. Thus, it removes explicit root height and foot-contact channels, includes the root in the position and rotation blocks, and replaces scalar yaw velocity with a 6D rotation increment. More importantly, its joint rotations come directly from the source SMPL motion rather than being recovered from positions by inverse kinematics. This retains twist information and permits direct SMPL animation without the slow, error-prone SMPLify post-processing required when only positions from the standard representation are used.

Table 7: Text-to-motion results on HumanML3D using the standard 263D evaluator. MSFLOW (263D) denotes the causal incremental-263D variant, while MSFLOW (XYZ) denotes the bidirectional global-XYZ variant; both use clean-motion prediction and s = 5. All intervals are 95% confidence intervals. Bold indicates the best result, and underline denotes the second-best result.
<table><tr><td rowspan=2 colspan=1>Method</td><td rowspan=2 colspan=1>Representation</td><td rowspan=1 colspan=1>R-Precision↑</td><td rowspan=2 colspan=1>FID↓  MM-Dist↓ MModality↑</td></tr><tr><td rowspan=1 colspan=1>Top 1    Top 2   Top 3</td></tr><tr><td rowspan=1 colspan=1>GT</td><td rowspan=1 colspan=1>|</td><td rowspan=1 colspan=1>.511±.003.703±.003.797±.002</td><td rowspan=1 colspan=1>.002±.0002.974±.008     一</td></tr><tr><td rowspan=4 colspan=1>T2M-GPT (Zhang et al., 2023a)MMM (Pinyoanuntapong et al., 2024)MoMask (Guo et al., 2024)IRG-MotionLLM (Li et al., 2026)</td><td rowspan=2 colspan=1>Discrete</td><td rowspan=1 colspan=1>.492±.003.679±.002.775±.002</td><td rowspan=1 colspan=1>.141±.0053.121±.009 1.831±.048</td></tr><tr><td rowspan=1 colspan=1>.515±.002.708±.002.804±.002</td><td rowspan=1 colspan=1>.089±.0052.926±.007 1.226±.035</td></tr><tr><td rowspan=2 colspan=1>latent</td><td rowspan=1 colspan=1>.521±.002.713±.002.807±.002</td><td rowspan=1 colspan=1>.045±.0022.958±.008 1.241±.040</td></tr><tr><td rowspan=1 colspan=1>.564±.002.754±.002.841±.002</td><td rowspan=1 colspan=1>.208±.0032.628±.005</td></tr><tr><td rowspan=9 colspan=1>MLD V2 (Chen et al., 2023)MotionLCM V2 (Dai et al., 2024)StableMoFusion (Huang et al., 2024a)EnergyMoGen (Zhang et al., 2025)SALAD (Hong et al., 2025)MARDM (Meng et al., 2025b)MotionStreamer (Xiao et al., 2025)FloodDiffusion (Cai et al., 2026)CMDM (Yu et al., 2026)</td><td rowspan=9 colspan=1>Continuouslatent</td><td rowspan=1 colspan=1>.542±.002.735±.002.827±.002</td><td rowspan=1 colspan=1>.078±.0042.808±.0061.676±.060</td></tr><tr><td rowspan=1 colspan=1>.548±.002.743±.002.835±.002</td><td rowspan=1 colspan=1>.092±.0032.760±.0081.800±.047</td></tr><tr><td rowspan=1 colspan=1>.553±.003.748±.002.841±.002</td><td rowspan=1 colspan=1>.098±.0032.715±.006 1.774±.051</td></tr><tr><td rowspan=1 colspan=1>.523±.003.715±.002.815±.002</td><td rowspan=1 colspan=1>.188±.0062.915±.0072.205±.041</td></tr><tr><td rowspan=1 colspan=1>.581±.003.769±.003.857±.002</td><td rowspan=1 colspan=1>.076±.0022.649±.009 1.751±.062</td></tr><tr><td rowspan=1 colspan=1>.517±.003.708±.003.805±.003</td><td rowspan=1 colspan=1>.116±.0062.968±.0101.923±.105</td></tr><tr><td rowspan=1 colspan=1>.496±.003.695±.002.793±.002</td><td rowspan=1 colspan=1>.201±.0053.041±.009 1.463±.075</td></tr><tr><td rowspan=1 colspan=1>.523±.002.717±.002.810±.003</td><td rowspan=1 colspan=1>.057±.0022.887±.007</td></tr><tr><td rowspan=1 colspan=1>.588±.004.778±.002.860±.003</td><td rowspan=1 colspan=1>.068±.0032.620±.010 1.785±.074</td></tr><tr><td rowspan=3 colspan=1>MDM (Tevet et al., 2023)MotionDiffuse (Zhang et al., 2024)ReMoDiffuse (Zhang et al., 2023b)</td><td rowspan=3 colspan=1>Rawmotion</td><td rowspan=1 colspan=1>.455±.006.645±.007.749±.002</td><td rowspan=1 colspan=1>.489±.0253.330±.0252.290±.070</td></tr><tr><td rowspan=1 colspan=1>.491±.001.681±.001.782±.001</td><td rowspan=1 colspan=1>.630±.0013.113±.001 1.553±.042</td></tr><tr><td rowspan=1 colspan=1>.510±.005.698±.006.795±.004</td><td rowspan=1 colspan=1>.103±.0042.974±.0161.795±.043</td></tr><tr><td rowspan=2 colspan=1>MSFLOW (263D)MSFLOW (XYZ)</td><td rowspan=2 colspan=1></td><td rowspan=1 colspan=1>.591±.004.778±.004.863±.003</td><td rowspan=1 colspan=1>.057±.0032.565±.007 .890±.048</td></tr><tr><td rowspan=1 colspan=1>.581±.003.773±.003.858±.002</td><td rowspan=1 colspan=1>.050±.0042.607±.009 1.287±.099</td></tr></table>

Table 8: Text-to-motion results on HumanML3D using the MotionStreamer 272D evaluator. We include all baselines reported by MotionStreamer and UMO. UMO converts the 201D outputs of HY-Motion and its own models to 272D for evaluation. Our causal 272D model is trained from scratch with clean-motion prediction and s = 5. Its intervals are 95% confidence intervals. Bold indicates the best generated result.
<table><tr><td rowspan="2"></td><td rowspan="2">Representation</td><td rowspan="2">FID↓</td><td colspan="3">R-Precision↑</td><td rowspan="2">MM-Dist↓</td></tr><tr><td>Top 1</td><td>Top 2</td><td>Top 3</td></tr><tr><td>Real motion</td><td></td><td>.002</td><td>.702</td><td>.864</td><td>.914</td><td>15.151</td></tr><tr><td>T2M-GPT (Zhang et al., 2023a)</td><td></td><td>12.475</td><td>.606</td><td>.774</td><td>.838</td><td>16.812</td></tr><tr><td>MotionGPT (Jiang et al., 2023)</td><td>Discrete</td><td>14.375</td><td>.456</td><td>.598</td><td>.628</td><td>17.892</td></tr><tr><td>MoMask (Guo et al., 2024)</td><td>latent</td><td>12.232</td><td>.621</td><td>.784</td><td>.846</td><td>16.138</td></tr><tr><td>AttT2M (Zhong et al., 2023)</td><td></td><td>15.428</td><td>.592</td><td>.765</td><td>.834</td><td>15.726</td></tr><tr><td>MLD (Chen et al., 2023) MotionStreamer (Xiao et al., 2025)</td><td>Continuous latent</td><td>18.236</td><td>.546</td><td>.730</td><td>.792</td><td>16.638</td></tr><tr><td>MDM (Tevet et al., 2023)</td><td></td><td>11.790</td><td>.631</td><td>.802</td><td>.859</td><td>16.081</td></tr><tr><td>HY-Motion (Wen et al., 2025)</td><td></td><td>23.454</td><td>.523</td><td>.692</td><td>.764</td><td>17.423</td></tr><tr><td></td><td>Raw</td><td>61.035</td><td>.667</td><td>.818</td><td>.876</td><td>17.530</td></tr><tr><td>UMO-Expert (Cong et al., 2026)</td><td>motion</td><td>17.040</td><td>.763</td><td>.889</td><td>.931</td><td>15.490</td></tr><tr><td>UMO-Unified (Cong et al., 2026)</td><td></td><td>9.460</td><td>.774</td><td>.892</td><td>.933</td><td>15.220</td></tr><tr><td>MSFLOW</td><td></td><td>6.540±.120</td><td>.792±.003</td><td>.919±.002</td><td>.955±.002</td><td>13.864±.014</td></tr></table>

## A.6 TRAINING AND EVALUATION ON SNAPMOGEN

We additionally train an MSFLOW model from scratch on SnapMoGen (Guo et al., 2025) using its native 296D motion representation with source scale s = 10. We evaluate the results using the same protocol used by previous works (Guo et al., 2025).

Table 9: Text-to-motion results on SnapMoGen. Our result uses a model trained from scratch on the native 296D SnapMoGen representation with clean-motion prediction and $s = 1 0$ . All intervals are 95% confidence intervals over 10 runs. Bold and underline indicate the best and second-best generated results.
<table><tr><td rowspan=2 colspan=1>Method</td><td rowspan=2 colspan=1>Representation</td><td rowspan=1 colspan=2>R-Precision↑</td><td rowspan=2 colspan=1>FID↓    MModality↑</td></tr><tr><td rowspan=1 colspan=2> $\overline { { \mathrm { T o p 1 } } }$      Top 2     $\overline { { \mathrm { T o p } 3 } }$ </td></tr><tr><td rowspan=1 colspan=1>GT</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2> $\overline { { . 9 4 0 ^ { \pm . 0 0 1 } } }$    $9 7 6 ^ { \pm . 0 0 1 }$    $\overline { { . 9 8 5 ^ { \pm . 0 0 1 } } }$ </td><td rowspan=1 colspan=1> $. 0 0 1 ^ { \pm . 0 0 0 }$ </td></tr><tr><td rowspan=3 colspan=1>T2M-GPT (Zhang et al., 2023a)MoMask (Guo et al., 2024) $\mathbf { M o M a s k ^ { + + } } \left( \mathbf { G u o \ e t \ a l . } , 2 0 2 5 \right)$  $\mathbf { S c a l e M o G e n }$ (Hwang et al., 2026)</td><td rowspan=3 colspan=1>Discretelatent</td><td rowspan=1 colspan=2> $. 6 1 8 ^ { \pm . 0 0 2 }$    $. 7 7 3 ^ { \pm . 0 0 2 }$    $. 8 1 2 ^ { \pm . 0 0 2 }$ </td><td rowspan=1 colspan=1> $3 2 . 6 2 9 ^ { \pm . 0 8 7 }$    $9 . 1 7 2 ^ { \pm . 1 8 1 }$ </td></tr><tr><td rowspan=1 colspan=2> $. 7 7 7 ^ { \pm . 0 0 2 }$    $. 8 8 8 ^ { \pm . 0 0 2 }$    $. 9 2 7 ^ { \pm . 0 0 2 }$ </td><td rowspan=1 colspan=1> $1 7 . 4 0 4 ^ { \pm . 0 5 1 }$    $8 . 1 8 3 ^ { \pm . 1 8 4 }$ </td></tr><tr><td rowspan=1 colspan=2> $. 8 0 2 ^ { \pm . 0 0 1 }$    $. 9 0 5 ^ { \pm . 0 0 2 }$    $. 9 3 8 ^ { \pm . 0 0 1 }$  $. 8 0 7 ^ { \pm . 0 0 4 }$    $. 9 0 8 ^ { \pm . 0 0 3 }$    $. 9 4 1 ^ { \pm . 0 0 3 }$ </td><td rowspan=1 colspan=1> $1 5 . 0 6 1 ^ { \pm . 0 6 5 }$    $7 . 2 5 9 ^ { \pm . 1 8 0 }$  $\overline { { 1 6 . 3 5 0 ^ { \pm . 0 8 6 } } }$    $9 . 3 9 9 ^ { \pm . 6 6 9 }$ </td></tr><tr><td rowspan=4 colspan=1>StableMoFusion (Huang et al., 2024a)MARDM (Meng et al., 2025b)MotionStreamer (Xiao et al., 2025)CMDM (Yu et al., 2026)</td><td rowspan=4 colspan=1>Continuouslatent</td><td rowspan=1 colspan=2> $. 6 7 9 ^ { \pm . 0 0 2 }$    $. 8 2 3 ^ { \pm . 0 0 2 }$    $. 8 8 8 ^ { \pm . 0 0 2 }$ </td><td rowspan=2 colspan=1> $2 7 . 8 0 1 ^ { \pm . 0 6 3 }$    $9 . 0 6 4 ^ { \pm . 1 3 8 }$  $2 6 . 3 4 8 ^ { \pm . 2 0 8 }$    $9 . 8 8 3 ^ { \pm . 1 4 7 }$ </td></tr><tr><td rowspan=1 colspan=1> $. 6 4 8 ^ { \pm . 0 0 2 }$ </td><td rowspan=1 colspan=1> $. 8 0 1 ^ { \pm . 0 0 2 }$    $. 8 5 6 ^ { \pm . 0 0 2 }$ </td></tr><tr><td rowspan=2 colspan=2> $. 6 3 1 ^ { \pm . 0 0 2 }$    $. 7 9 1 ^ { \pm . 0 0 2 }$    $. 8 3 6 ^ { \pm . 0 0 2 }$  $. 8 3 1 ^ { \pm . 0 0 4 }$    $. 9 2 6 ^ { \pm . 0 0 3 }$    $\underline { { 9 5 8 ^ { \pm . 0 0 2 } } }$ </td><td rowspan=1 colspan=1> $3 0 . 0 2 3 ^ { \pm . 1 3 1 }$    $7 . 5 4 3 ^ { \pm . 1 9 5 }$ </td></tr><tr><td rowspan=1 colspan=1> $\mathbf { 1 4 . 4 5 1 ^ { \pm . 0 8 9 } }$    $9 . 5 2 1 ^ { \pm . 1 9 6 }$ </td></tr><tr><td rowspan=1 colspan=1>MDM (Tevet et al., 2023)</td><td rowspan=1 colspan=1>Raw</td><td rowspan=1 colspan=2> $. 5 0 3 ^ { \pm . 0 0 2 }$    $. 6 5 3 ^ { \pm . 0 0 2 }$   $. 7 2 7 ^ { \pm . 0 0 2 }$ </td><td rowspan=1 colspan=1> $5 7 . 7 8 3 ^ { \pm . 0 9 2 }$   $\mathbf { 1 3 . 4 1 2 ^ { \pm . 2 3 1 } }$ </td></tr><tr><td rowspan=1 colspan=1>MSFLOW</td><td rowspan=1 colspan=1>motion</td><td rowspan=1 colspan=2> $\mathbf { . 9 1 0 ^ { \pm . 0 0 2 } }$  $\mathbf { 9 6 9 ^ { \pm . 0 0 2 } }$   $\mathbf { . 9 8 4 ^ { \pm . 0 0 1 } }$ </td><td rowspan=1 colspan=1> $1 6 . 3 4 2 ^ { \pm . 1 1 8 }$  $1 2 . 5 3 8 ^ { \pm . 5 9 1 }$ </td></tr></table>

Native motion representation and BVH recovery. Each SnapMoGen frame contains root-yaw velocity (1D), planar root velocity (2D), root height (1D), heading-normalized global 6D rotations $( \mathrm { 2 4 \times 6 D } )$ , joint positions $\mathrm { ( 2 4 \times 3 D ) }$ ), joint velocities $\mathrm { ( 2 4 \times 3 D ) }$ ), and four foot-contact indicators, totaling 296 dimensions. Unlike HumanML3D’s parent-relative rotations, SnapMoGen rotations are global after removing root heading. BVH recovery requires only the first 148 channels: integrating root motion recovers the trajectory, while converting 6D rotations to quaternions, restoring heading, and applying the template hierarchy yields local BVH rotations. This also differs from evaluation on HumanML3D: its 67D protocol embeds four root channels and root-relative XYZ positions for 21 non-root joints $( 4 + 2 1 \times 3 = 6 7 )$ , whereas the official SnapMoGen evaluator embeds the first 148 root-and-rotation channels. Our model nevertheless trains on and generates all 296 channels, with the remaining channels providing redundant learning signals.

Attention graph. HumanML3D’s incremental features naturally support forward accumulation and therefore favor causal attention. In contrast, SnapMoGen uses a hybrid representation that combines root motion and global joint rotations with redundant position, velocity, and contact features. Bidirectional attention uses context from both directions to keep these features consistent across the full motion, and it outperforms causal attention on SnapMoGen. SnapMoGen also uses longer sequences (312 versus 192 frames), amplifying the context asymmetry of a causal mask. We therefore use bidirectional attention, while viewing this design as an interaction among representation, sequence length, and source scale rather than a universal advantage.

Table 9 shows that MSFLOW obtains the strongest text–motion alignment, improving Top-1/2/3 R-Precision from 0.831/0.926/0.958 for CMDM to 0.910/0.969/0.984 and exceeding ScaleMoGen’s 0.807/0.908/0.941. Its FID of 16.342 is higher than CMDM (14.451) and MoMask<sup>++</sup> (15.061), but slightly lower than ScaleMoGen (16.350) and all other remaining baselines, including MoMask (17.404). MSFLOW also obtains the second-best multimodality score of 12.538, behind MDM (13.412). The result therefore improves semantic retrieval and distributional fidelity over the previous SnapMoGen checkpoint while retaining strong conditional diversity.

## A.7 COMPUTE EFFICIENCY

We profile MSFLOW and the selected prior methods under a common 196-frame generation protocol. For each method, we use its official classifier-free guidance and sampling schedule and measure the complete sampling trajectory, including decoding into motion space. We retain each release’s native motion representation: MARDM generates 67D features, MotionStreamer generates 272D features, and the remaining methods generate 263D features. All runs use batch size one, FP32 with TF32 enabled, and the prompt “a person walks forward.” on a single NVIDIA A100-SXM4-80GB GPU. We report the median of five synchronized runs after two warm-up runs. FLOPs are measured with the PyTorch flop counter, with a multiply–add counted as two operations.

Table 10: Quality and compute efficiency for generating a 196-frame motion. Parameters include the complete learned generation stack and motion encoder, where applicable, but exclude frozen text encoders. FLOPs count a multiply–add as two operations. Time is the median of five runs on one NVIDIA A100-SXM4-80GB GPU. FID and R-Top3 are point estimates from the standard 263D evaluation in Table 7. Bold and underline denote the best and second-best available results among the methods shown.
<table><tr><td>Method</td><td>Params (M) ↓</td><td>FLOPs (TF) ↓</td><td>Time (s) ↓</td><td>FID↓</td><td>R-Top3 ↑</td></tr><tr><td>MARDM (Meng et al., 2025b)</td><td>309.65</td><td>75.477</td><td>9.579</td><td>0.116</td><td>0.805</td></tr><tr><td>SALAD (Hong et al., 2025)</td><td>10.06</td><td>0.467</td><td>0.477</td><td>0.076</td><td>0.857</td></tr><tr><td>MotionStreamer (Xiao et al., 2025)</td><td>318.51</td><td>1.959</td><td>9.023</td><td>0.201</td><td>0.793</td></tr><tr><td>CMDM (Yu et al., 2026)</td><td>115.01</td><td>0.572</td><td>1.809</td><td>0.068</td><td>0.860</td></tr><tr><td>FloodDiffusion (Cai et al., 2026)</td><td>131.22</td><td>7.060</td><td>3.537</td><td>0.057</td><td>0.810</td></tr><tr><td>MSFLOW</td><td>68.77</td><td>1.502</td><td>1.742</td><td>0.057</td><td>0.863</td></tr></table>

For model size, we count all learned modules required for motion generation, including the full motion encoder when a method uses one, but omit frozen text encoders because they perform cacheable, one-time prompt preprocessing outside the iterative generator. Their execution is likewise excluded from FLOPs and timing. MARDM uses its official trained checkpoint and the actual cached CLIP embedding because its adaptive Dopri5 solver has conditioning-dependent function evaluations. FloodDiffusion internally requires 50 latent tokens, which decode to 197 frames; we follow its length construction and crop the final decoded frame so that every reported output contains exactly 196 frames.

As shown in Table 10, SALAD is the most efficient method on all three compute measures. MS-FLOW is second-smallest at 68.77M parameters and second-fastest at 1.742 seconds, while its 1.502 TFLOPs is the third-lowest after SALAD and CMDM. Using the standard 263D evaluation from Ta ble 7, it obtains the highest R-Top3 of 0.863 and ties FloodDiffusion for the best FID of 0.057. Relative to CMDM, it uses 40.2% fewer parameters and is 3.7% faster, while improving R-Top3 by 0.003 and FID by 0.011, although it requires more FLOPs. It is also faster and uses fewer FLOPs than MotionStreamer, FloodDiffusion, and MARDM. Thus, direct full-sequence motion generation remains competitive in model size and latency while attaining the strongest reported quality.

## B DETAILED SPATIAL-CONTROL RESULTS

The main paper reports the pelvis-only and all-joints-average results in Table 4. Table 11 gives the complete per-joint evaluation under the same OmniControl (Xie et al., 2024) protocol. Across the pelvis, feet, head, and wrists, MSFLOW has zero trajectory, location, and average errors. Compared with ProjFlow (Watanabe et al., 2026), it lowers FID and raises R-Precision@3 for every reported joint. Averaged over the all-joints setting, FID improves from 0.097 to 0.061 and R-Precision@3 from 0.779 to 0.818. The average foot-skating ratio also decreases from 0.0603 to 0.0578, although exact satisfaction of projected coordinates and a single physical metric do not establish the quality of the unconstrained motion.

Following MaskControl (Pinyoanuntapong et al., 2025) and ProjFlow (Watanabe et al., 2026), upperbody editing conditions on the ground-truth pelvis, left-foot, and right-foot signals at every frame and generates the remaining motion from the text description. The final block reports this task using the comparison and metric format of ProjFlow. MSFLOW achieves the best FID, reducing it from 0.066 to 0.034 relative to the strongest baseline, and obtains R-Precision of 0.511, 0.701, and 0.794 at ranks 1–3, a matching distance of 3.214, and diversity of 10.714.

## C IMPLEMENTATION AND REPRODUCIBILITY DETAILS

We next specify the two principal MSFLOW variants and give the training, sampling, and endpointprojection algorithms corresponding to the framework in Fig. 2. Unless otherwise stated, models are trained for 500 epochs with batch size 64 using AdamW (Loshchilov & Hutter, 2019) and learning rate $2 \times 1 0 ^ { - 4 }$ . The 68M-parameter transformer has eight RA-MMDiT blocks of width 512, four attention heads, and feed-forward width 1024. We use text dropout 0.1, classifier-free guidance weight 3, one sequence-level $t \sim \mathcal { U } ( 0 , 1 )$ per training example, and $\sigma _ { \mathrm { m i n } } = 0 . 0 5$ for training and 0.01 for inference. Sampling uses EMA weights, fixed-step Heun updates, and a final Euler step. A complete 500-epoch training run of MSFLOW takes approximately 450 minutes on one NVIDIA A100-SXM4-80GB GPU.

Table 11: Complete results for text-conditioned spatial control and upper-body editing on HumanML3D. The first block trains and evaluates on pelvis controls. The per-joint blocks use methods trained on all joints. The final block follows the upper-body editing format of ProjFlow (Watanabe et al., 2026). Bold and underline denote the best and second-best values, respectively; diversity is ranked by absolute deviation from GT, ties share the same formatting, and gray cells mark MS-FLOW.
<table><tr><td>Controlling Joint</td><td>Methods</td><td>Zero-shot?</td><td>FID.↓</td><td>R-Precision Top 3</td><td>Diversity→</td><td>Foot Skating Ratio.↓</td><td>Traj. err.↓</td><td>Loc. err.↓</td><td> $\mathbf { A v g . \ e r r . } \downarrow$ </td></tr><tr><td rowspan="7"></td><td>GT</td><td></td><td>0.000</td><td>0.795</td><td>10.455</td><td></td><td>0.000</td><td>0.000</td><td>0.000</td></tr><tr><td>MDM (Tevet et al., 2023)</td><td>√</td><td>1.792</td><td>0.673</td><td>9.131</td><td>0.1019</td><td>0.4022</td><td>0.3076</td><td>0.5959</td></tr><tr><td>PriorMDM (Shafir et al., 2024)</td><td>x</td><td>0.393</td><td>0.707</td><td>9.847</td><td>0.0897</td><td>0.3457</td><td>0.2132</td><td>0.4417</td></tr><tr><td>GMD (Karunratanakul et al., 2023)</td><td>√</td><td>0.238</td><td>0.763</td><td>10.011</td><td>0.1009</td><td>0.0931</td><td>0.0321</td><td>0.1439</td></tr><tr><td>OmniControl (Xie et al., 2024)</td><td>x</td><td>0.081</td><td>0.789</td><td>10.323</td><td>0.0547</td><td>0.0387</td><td>0.0096</td><td>0.0338</td></tr><tr><td>ProjFlow (Watanabe et al., 2026)</td><td>√</td><td>0.107</td><td>0.784</td><td>10.645</td><td>0.0630</td><td>0.0000</td><td>0.0000</td><td>0.0000</td></tr><tr><td>MŠFLOW</td><td>√</td><td>0.068</td><td>0.821</td><td>10.447</td><td>0.0591</td><td>0.0000</td><td>0.0000</td><td>0.0000</td></tr><tr><td rowspan="6">Pelvis</td><td>OmniControl (Xie et al., 2024)</td><td>××</td><td>0.135</td><td>0.790</td><td>10.314</td><td>0.0571</td><td>0.0404</td><td>0.0085</td><td>0.0367</td></tr><tr><td>MotionLCM V2+CtrlNet (Dai et al., 2024)</td><td></td><td>4.726</td><td>0.713</td><td>9.209</td><td>0.1162</td><td>0.1617</td><td>0.0841</td><td>0.1838</td></tr><tr><td>MaskControl (Pinyoanuntapong et al., 2025)</td><td>x</td><td>0.087</td><td>0.795</td><td>10.168</td><td>0.0544</td><td>0.0003</td><td>0.0000</td><td>0.0114</td></tr><tr><td>ProjFlow (Watanabe et al., 2026)</td><td>√</td><td>0.107</td><td>0.784</td><td>10.645</td><td>0.0630</td><td>0.0000</td><td>0.0000</td><td>0.0000</td></tr><tr><td>MŠFLOW</td><td>√</td><td>0.068</td><td>0.821</td><td>10.447</td><td>0.0591</td><td>0.0000</td><td>0.0000</td><td>0.0000</td></tr><tr><td>OmniControl (Xie et al., 2024)</td><td>x</td><td>0.093</td><td>0.794</td><td>10.338</td><td>0.0692</td><td>0.0594</td><td>0.0094</td><td>0.0314</td></tr><tr><td rowspan="6">Left foot</td><td>MotionLCM V2+CtrlNet (Dai et al., 2024)</td><td>x</td><td>4.810</td><td>0.706</td><td>9.158</td><td>0.1047</td><td>0.2607</td><td>0.1229</td><td>0.2304</td></tr><tr><td>MaskControl (Pinyoanuntapong et al., 2025)</td><td>x</td><td>0.074</td><td>0.793</td><td>10.241</td><td>0.0561</td><td>0.0000</td><td>0.0000</td><td>0.0066</td></tr><tr><td>ProjFlow (Watanabe et al., 2026)</td><td>V</td><td>0.095</td><td>0.771</td><td>10.644</td><td>0.0609</td><td>0.0000</td><td>0.0000</td><td>0.0000</td></tr><tr><td>MŠFLOW</td><td>√</td><td>0.041</td><td>0.817</td><td>10.630</td><td>0.0639</td><td>0.0000</td><td>0.0000</td><td>0.0000</td></tr><tr><td>OmniControl (Xie et al., 2024)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MotionLCM V2+CtrlNet (Dai et al., 2024)</td><td>x x</td><td>0.137 4.756</td><td>0.798 0.705</td><td>10.241 9.303</td><td>0.0668 0.1026</td><td>0.0666 0.2459</td><td>0.0120</td><td>0.0334</td></tr><tr><td rowspan="5">Right foot</td><td>MaskControl (Pinyoanuntapong et al., 2025)</td><td>x</td><td>0.080</td><td>0.793</td><td>10.159</td><td>0.0552</td><td>0.0000</td><td>0.1127 0.0000</td><td>0.2278 0.0062</td></tr><tr><td>ProjFlow (Watanabe et al., 2026)</td><td>√</td><td>0.096</td><td>0.770</td><td>10.651</td><td>0.0613</td><td>0.0000</td><td>0.0000</td><td>0.0000</td></tr><tr><td>MŠFLOW</td><td>√</td><td>0.043</td><td>0.818</td><td>10.729</td><td>0.0642</td><td>0.0000</td><td>0.0000</td><td>0.0000</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>OmniControl (Xie et al., 2024) MotionLCM V2+CtrlNet (Dai et al., 2024)</td><td>x</td><td>0.146</td><td>0.796</td><td>10.239</td><td>0.0556</td><td>0.0422</td><td>0.0079</td><td>0.0349</td></tr><tr><td rowspan="5">Head</td><td></td><td>x</td><td>4.580</td><td>0.715</td><td>9.278</td><td>0.1138</td><td>0.1971</td><td>0.0977</td><td>0.2136</td></tr><tr><td>MaskControl (Pinyoanuntapong et al., 2025)</td><td>x</td><td>0.090</td><td>0.797</td><td>10.131</td><td>0.0531</td><td>0.0000</td><td>0.0000</td><td>0.0064</td></tr><tr><td>ProjFlow (Watanabe et al., 2026)</td><td>√</td><td>0.099</td><td>0.788</td><td>10.754</td><td>0.0595</td><td>0.0000</td><td>0.0000</td><td>0.0000</td></tr><tr><td>MŠFLOW</td><td>√</td><td>0.066</td><td>0.821</td><td>10.642</td><td>0.0528</td><td>0.0000</td><td>0.0000</td><td>0.0000</td></tr><tr><td>OmniControl (Xie et al., 2024)</td><td>x</td><td>0.119</td><td>0.783</td><td>10.217</td><td>0.0562</td><td>0.0801</td><td>0.0134</td><td>0.0529</td></tr><tr><td rowspan="5">Left wrist</td><td>MotionLCM V2+CtrlNet (Dai et al., 2024)</td><td>x</td><td>4.103</td><td>0.726</td><td>9.188</td><td>0.1167</td><td>0.3965</td><td>0.1912</td><td>0.3150</td></tr><tr><td>MaskControl (Pinyoanuntapong et al., 2025)</td><td>x</td><td>0.118</td><td>0.797</td><td>10.153</td><td>0.0546</td><td>0.0000</td><td>0.0000</td><td>0.0044</td></tr><tr><td>ProjFlow (Watanabe et al., 2026)</td><td>√</td><td>0.089</td><td>0.783</td><td>10.601</td><td>0.0586</td><td>0.0000</td><td>0.0000</td><td>0.0000</td></tr><tr><td>MŠFLOW</td><td>V</td><td>0.075</td><td>0.818</td><td>10.576</td><td>0.0535</td><td>0.0000</td><td>0.0000</td><td>0.0000</td></tr><tr><td>OmniControl (Xie et al., 2024)</td><td>x</td><td>0.128</td><td>0.792</td><td>10.309</td><td>0.0601</td><td>0.0813</td><td>0.0127</td><td></td></tr><tr><td rowspan="5">Right wrist</td><td>MotionLCM V2+CtrlNet (Dai et al., 2024)</td><td>x</td><td></td><td>0.725</td><td></td><td></td><td></td><td></td><td>0.0519</td></tr><tr><td>MaskControl (Pinyoanuntapong et al., 2025)</td><td></td><td>4.051</td><td></td><td>9.242</td><td>0.1176</td><td>0.3822</td><td>0.1806</td><td>0.3079</td></tr><tr><td></td><td>x</td><td>0.121</td><td>0.797</td><td>10.105</td><td>0.0537</td><td>0.0000</td><td>0.0000</td><td>0.0044</td></tr><tr><td>ProjFlow (Watanabe et al., 2026) MSFLOW</td><td>V</td><td>0.096</td><td>0.780</td><td>10.610</td><td>0.0584</td><td>0.0000</td><td>0.0000</td><td>0.0000</td></tr><tr><td></td><td>V</td><td>0.075</td><td>0.815</td><td>10.533</td><td>0.0534</td><td>0.0000</td><td>0.0000</td><td>0.0000</td></tr><tr><td rowspan="5">Average</td><td>OmniControl (Xie et al., 2024)</td><td>x</td><td>0.126</td><td>0.792</td><td>10.276</td><td>0.0608</td><td>0.0617</td><td>0.0107</td><td>0.0404</td></tr><tr><td>MotionLCM V2+CtrlNet (Dai et al., 2024)</td><td>××</td><td>4.504 0.095</td><td>0.715 0.795</td><td>9.230 10.159</td><td>0.1119 0.0545</td><td>0.2740 0.0001</td><td>0.1315</td><td>0.2464</td></tr><tr><td>MaskControl (Pinyoanuntapong et al., 2025) ProjFlow (Watanabe et al., 2026)</td><td>√</td></table>

## C.1 CAUSAL 263D TEXT-TO-MOTION CONFIGURATION

Table 12 summarizes the configuration of the causal 263D generator.

## C.2 BIDIRECTIONAL XYZ TEXT-TO-MOTION CONFIGURATION

Table 13 summarizes the corresponding configuration of the bidirectional XYZ generator.

Table 12: Configuration of the causal 263D model used for the primary HumanML3D result. The evaluated EMA checkpoint was not retained with an immutable run and epoch identifier.
<table><tr><td>Item</td><td>Selected value</td></tr><tr><td>Model variant Training representation</td><td>causal 263D, s = 5</td></tr><tr><td>Evaluation representation</td><td>HumanML3D, 263 dimensions root/RIC subset, 67 dimensions</td></tr><tr><td>Backbone Motion block size</td><td>8 joint blocks, width 512, 4 heads</td></tr><tr><td>Training / inference mask</td><td>1 frame causal / causal</td></tr><tr><td>Flow path / time</td><td>linear / uniform sequence-level t</td></tr><tr><td>Network output / loss</td><td>endpoint  $\hat { \bf x } _ { 1 } /$  stabilized residual MSE</td></tr><tr><td>Source distribution Text encoder</td><td>N(0, 25I)</td></tr><tr><td></td><td>frozen DistilBERT, token features</td></tr><tr><td>Token Refiner</td><td>2 layers, timestep/context conditioned</td></tr><tr><td>CFG / condition dropout</td><td>3.0 / 0.1</td></tr><tr><td>Sampler Evaluated weights</td><td>50 time points, Heun + final Euler</td></tr></table>

Table 13: Configuration of the bidirectional XYZ model used for the primary HumanML3D result.
<table><tr><td>Item</td><td>Selected value</td></tr><tr><td>Model variant</td><td>bidirectional XYZ, s = 5</td></tr><tr><td>Training representation</td><td>shared-axis-normalized XYZ, 66 dimensions</td></tr><tr><td>Evaluation representation Backbone</td><td>root/RIC subset, 67 dimensions 8 joint blocks, width 512, 4 heads</td></tr><tr><td>Motion block size</td><td>1 frame</td></tr><tr><td>Training / inference mask</td><td>bidirectional / bidirectional</td></tr><tr><td>Flow path / time</td><td>linear / uniform sequence-level t</td></tr><tr><td>Network output / loss</td><td>endpoint  $\hat { \mathbf { x } } _ { 1 } /$  stabilized residual MSE</td></tr><tr><td>Source distribution</td><td>N(0, 25I)</td></tr><tr><td>Text encoder</td><td>frozen DistilBERT, token features</td></tr><tr><td>Token Refiner</td><td>2 layers, timestep/context conditioned</td></tr><tr><td>CFG / condition dropout</td><td>3.0 / 0.1</td></tr><tr><td>Sampler</td><td></td></tr><tr><td></td><td>50 time points, Heun + final Euler</td></tr><tr><td>Evaluated weights</td><td>latest EMA checkpoint labeled</td></tr></table>

## C.3 TRAINING AND SAMPLING ALGORITHMS

Algorithms 1 and 2 detail the shared training and text-to-motion sampling procedures, respectively.

## C.4 BIDIRECTIONAL XYZ ENDPOINT-PROJECTION CONFIGURATION

Spatial control reuses the source scale and attention graph of the trained bidirectional XYZ generator. Arbitrary binary masks select valid XYZ entries, while neither masks nor target coordinates are provided during training. Table 14 summarizes this inference-only configuration, and Algorithm 3 details the endpoint-projection sampler.

## D SAMPLE CODE

We provide code for training and evaluating the proposed MSFLOW on the HumanML3D dataset.   
Please refer to the sample code on our project websitefor details.

Algorithm 1 Representation-aware direct flow training   
Require: Motion–text minibatch $\mathbf { \Psi } ( \mathbf { x } _ { 1 } , \mathbf { c } )$ , source scale s   
1: Sample $\epsilon \sim \mathcal { N } ( 0 , \bf { I } )$ and $t \sim \dot { \mathcal { U } } ( 0 , 1 )$   
2: Form $\mathbf { x } _ { 0 }  s \mathbf { \epsilon }$ and $\mathbf { x } _ { t } \gets ( 1 - t ) \mathbf { x } _ { 0 } + t \mathbf { x } _ { 1 }$   
3: Refine frozen text tokens using t and their masked global mean   
4: Choose the fixed mask $\mathbf { M } _ { \mathrm { A t t } }$ (causal 263D or bidirectional XYZ)   
5: Predict $\hat { \mathbf { x } } _ { 1 } \gets f _ { \theta } ( \mathbf { x } _ { t } , t , \mathbf { c } ; \mathbf { M } _ { \mathrm { A t t } } )$   
6: Convert $\hat { \mathbf { x } } _ { 1 }$ and x to stabilized residual fields with $\operatorname { E q . } \left( 5 \right)$   
7: Update θ with valid-frame residual MSE; update EMA weights

Algorithm 2 MSFLOW sampling   
Require: Text $\mathbf { c } ,$ length $T ,$ source scale $s ,$ guidance w   
1: Sample the full sequence $\mathbf { x } _ { 0 } \sim \mathcal { N } ( 0 , s ^ { 2 } \mathbf { \tilde { I } } )$   
2: Encode and refine conditional and null text features   
3: for successive time points $t _ { k } , t _ { k + 1 }$ do   
4: Predict the guided endpoint with classifier-free guidance using the checkpoint mask $\mathbf { M } _ { \mathrm { A t t } }$   
5: Convert the endpoint to the residual field with Eq. (5)   
6: Update all motion frames with one Heun step   
7: end for   
8: Replace the final corrector with one Euler update   
9: return generated motion features $\mathbf { x } _ { 1 }$

## E THEORETICAL DETAILS AND PROOFS

This section provides the proofs underlying the three design questions in the main paper: the autoencoder fidelity floor in latent generation, the geometry induced by a scaled source, and the relationship between temporal attention and motion representation. For the last question, we also provide the full statistical formulation and conclude with the dynamical prefix property of the causal vector field.

## E.1 PROOFS OF MAIN PROPOSITIONS

Proposition 1 formalizes the irreducible fidelity error introduced when generated motions are restricted to a fixed decoder’s range, motivating direct motion-space generation. Proposition $2 \ \mathrm { e x } .$ plains how the Gaussian source scale controls both when motion signal emerges along the flow path and the conditioning of intermediate distributions.

Proof of Proposition 1. Let $\pi$ be any coupling of $\mathbf { X } \sim P _ { 1 }$ and $\mathbf { Y } \sim Q .$ . Because Q is supported on $\mathcal { R } _ { g } , \mathbf { \bar { Y } } \in \mathcal { R } _ { g }$ almost surely. Pointwise,

$$
\| \mathbf { X } - \mathbf { Y } \| _ { 2 } ^ { 2 } \geq \operatorname* { i n f } _ { \mathbf { y } \in \mathcal { R } _ { g } } \| \mathbf { X } - \mathbf { y } \| _ { 2 } ^ { 2 } = \mathrm { d i s t } ^ { 2 } ( \mathbf { X } , \mathcal { R } _ { g } ) .\tag{15}
$$

Taking the expectation under π and then the infimum over all couplings yields Eq. (1).

Proof of Proposition 2. Project Eq. (2) onto an eigenvector u: $\mathbf { u } ^ { \top } \mathbf { X } _ { t } = t \mathbf { u } ^ { \top } \mathbf { X } _ { 1 } + ( 1 - t ) s \mathbf { u } ^ { \top } \pmb \epsilon .$ Independence gives signal variance $t ^ { 2 } \lambda _ { \mathbf { u } }$ and source-noise variance $( 1 - t ) ^ { 2 } s ^ { 2 }$ , proving Eq. (8). The inequality $\mathrm { S N R } _ { \mathbf { u } } ( t ; s ) \leq 1$ is equivalent on [0, 1] to $t \le s / ( s + \sqrt { \lambda _ { \bf u } } )$ . Under uniform time sampling, the probability is the length of this interval.

Independence also gives Eq. (9). Its extreme eigenvalues are $t ^ { 2 } \lambda _ { \operatorname* { m i n } } + ( 1 - t ) ^ { 2 } s ^ { 2 }$ and $t ^ { 2 } \lambda _ { \operatorname* { m a x } } + ( 1 -$ $t ) ^ { 2 } s ^ { \frac { \mathbf { i } } { 2 } }$ , yielding Eq. (10). Let $\dot { a } = t ^ { 2 }$ and $b = ( \bar { 1 } - t ) ^ { 2 } s ^ { 2 }$ . For $t < 1$

$$
\frac { \partial } { \partial b } \frac { a \lambda _ { \operatorname* { m a x } } + b } { a \lambda _ { \operatorname* { m i n } } + b } = \frac { a ( \lambda _ { \operatorname* { m i n } } - \lambda _ { \operatorname* { m a x } } ) } { ( a \lambda _ { \operatorname* { m i n } } + b ) ^ { 2 } } \leq 0 .\tag{16}
$$

Since b is increasing in $s > 0 , \kappa _ { t } ( s )$ is non-increasing.

Table 14: Inference configuration for bidirectional XYZ endpoint projection. The model is trained for text-to-motion generation without masks or target coordinates.
<table><tr><td>Item</td><td>Control-evaluation setting</td></tr><tr><td>Training attention</td><td>bidirectional, fixed for the XYZ checkpoint</td></tr><tr><td>Inference attention</td><td>bidirectional</td></tr><tr><td>Control training</td><td>none</td></tr><tr><td>Raw controls</td><td>any valid frames, selected joint IDs, and XYZ axes</td></tr><tr><td>CFG weight / projected steps</td><td>3.0 / 100</td></tr><tr><td>Source noise</td><td>scale s of the training path</td></tr><tr><td>Noise refresh</td><td>source-scale corrected</td></tr></table>

Algorithm 3 Bidirectional zero-shot XYZ control   
Require: Bidirectional XYZ checkpoint, text $\mathbf { c } ,$ length $T ,$ target y, mask M   
1: Resolve source scale s from the checkpoint   
2: Sample $\mathbf { x } _ { 0 } \sim \mathcal { N } ( \mathbf { 0 } , s ^ { 2 } \mathbf { I } )$   
3: for successive time points $t , t ^ { \prime }$ do   
4: Predict the CFG velocity and recover $( \hat { \mathbf { x } } _ { 1 } , \hat { \mathbf { x } } _ { 0 } )$   
5: Project $\hat { \mathbf { x } } _ { 1 }$ onto $( \mathbf { M } , \mathbf { y } )$ using Eq. (11)   
6: Optionally refresh $\hat { \mathbf { x } } _ { 0 }$ using Eq. (13)   
7: Recompose $\mathbf { x } _ { t ^ { \prime } }$ using Eq. (14)   
8: end for   
9: Hard-replace controlled coordinates and zero padded frames   
10: return controlled direct-space motion $\mathbf { x } _ { 1 }$

## E.2 EMPIRICAL SPATIOTEMPORAL COVARIANCE AND SOURCE SCALE

We complement Proposition 2 with a full-split covariance analysis of the two training representations. For each window length $T _ { 0 } ~ \in ~ \{ 3 \bar { 2 } , 6 4 , 1 2 8 \}$ , we draw one uniformly positioned window from every eligible HumanML3D training motion and flatten it to a vector $\mathbf { x } _ { n } \in \mathbb { R } ^ { p }$ , where $p = T _ { 0 } D$ . Each coordinate of ${ \bf x } _ { n }$ therefore identifies one feature channel d at one frame τ. A covariance entry indexed by $( \tau , d )$ and $( \tau ^ { \prime } , d ^ { \prime } )$ measures how those two variables vary together across motions. It includes same-frame relationships between channels, temporal relationships between frames, and cross-channel relationships across different frames; we therefore call it spatiotemporal covariance. For XYZ, the channels are joint coordinates, while for 263D they are the channels of the HumanML3D representation.

Sampling at most one window per motion avoids overweighting long or highly correlated sequences. The filter max $( 4 0 , T _ { 0 } ) \leq T \bar { < }$ 200 yields 22,326, 20,850, and 13,328 motions for $T _ { 0 } = 3 2 ,$ 64, and 128, respectively. The 263D windows use the official feature-wise HumanML3D normalization, while XYZ uses the shared-axis normalization used to train the global generator.

We first form the empirical covariance across the N sampled motion windows. Ledoit–Wolf shrinkage then moves this estimate toward a scaled identity matrix to stabilize its spectrum when the dimension $p$ is large relative to N. We compare every result with an isotropic Gaussian matrix processed by the identical estimator and matched in both $N$ and $p .$ For correlation matrix $\widehat { \bf R }$ and shrinkage-covariance eigenvalues $\lambda _ { k }$ , we report

$$
C _ { \mathrm { o f f } } = \left( \frac { 1 } { p ( p - 1 ) } \sum _ { i \ne j } \widehat { R } _ { i j } ^ { 2 } \right) ^ { 1 / 2 } , \quad \quad \quad \quad \quad A _ { \mathrm { s p e c } } = \frac { \lambda _ { \mathrm { m a x } } } { p ^ { - 1 } \sum _ { k } \lambda _ { k } } ,\tag{17}
$$

$$
\overline { { r } } _ { \mathrm { e f f } } = \frac { 1 } { p } \exp \left( - \sum _ { k } q _ { k } \log q _ { k } \right) , \qquad q _ { k } = \frac { \lambda _ { k } } { \sum _ { j } \lambda _ { j } } .\tag{18}
$$

Here, $C _ { \mathrm { o f f } }$ is the root-mean-square off-diagonal correlation. It summarizes pairwise dependence between distinct frame–channel variables: zero means no linear pairwise correlation, and larger values indicate stronger dependence. $A _ { \mathrm { s p e c } }$ divides the largest covariance eigenvalue by the mean eigenvalue. It equals one when every direction has equal variance and grows when one direction dominates the covariance spectrum.

Table 15: Spatiotemporal covariance diagnostics across window lengths. Each statistic is reported as motion / a matched isotropic Gaussian with the same sample count N, dimension $p ,$ and covariance estimator. Lower $C _ { \mathrm { o f f } }$ , lower $A _ { \mathrm { s p e c } }$ , and higher $\overline { { r } } _ { \mathrm { e f f } }$ indicate greater isotropy.
<table><tr><td>Representation</td><td> $T _ { 0 }$ </td><td> $N$ </td><td> $C _ { \mathrm { o f f } }$ </td><td> $A _ { \mathrm { s p e c } }$ </td><td> $\overline { { r } } _ { \mathrm { e f f } }$ </td></tr><tr><td rowspan="3">263D</td><td>32</td><td>22,326</td><td>.166/.007</td><td>1088.4/1.0</td><td>.009/1.000</td></tr><tr><td>64</td><td>20,850</td><td>.148/.007</td><td>2116.0/1.0</td><td>.007/1.000</td></tr><tr><td>128</td><td>13,328</td><td>.137/.009</td><td>4336.5/1.0</td><td>.004/1.000</td></tr><tr><td rowspan="3">XYZ</td><td>32</td><td>22,326</td><td>.485/.007</td><td>1027.0/1.0</td><td>.002/1.000</td></tr><tr><td>64</td><td>20,850</td><td>.444/.007</td><td>1808.7/1.0</td><td>.001/1.000</td></tr><tr><td>128</td><td>13,328</td><td>.390/.009</td><td>3346.3/1.0</td><td>.001/1.000</td></tr></table>

Table 16: Covariance diagnostics of pretrained latent representations. All methods use the same 20,850 HumanML3D training motions and one 64-frame window per motion. Latent shape excludes the batch dimension. Each diagnostic is reported as latent / a matched isotropic Gaussian with the same $N , p ,$ and covariance estimator.
<table><tr><td>Method</td><td>Latent shape</td><td> $C _ { \mathrm { o f f } }$ </td><td> $A _ { \mathrm { s p e c } }$ </td><td> $\overline { { r } } _ { \mathrm { e f f } }$ </td></tr><tr><td>CMDM</td><td> $1 6 \times 6 4$ </td><td>.181/.007</td><td>118.9/1.0</td><td>.077/1.000</td></tr><tr><td>SALAD</td><td> $1 6 \times 7 \times 3 2$ </td><td>.076/.007</td><td>129.3/1.0</td><td>.238/1.000</td></tr></table>

The normalized weight $q _ { k }$ is the fraction of total covariance variance assigned to eigen-direction $k ,$ so $\begin{array} { r } { q _ { k } \geq 0 \mathrm { a n d } \sum _ { k } q _ { k } = 1 } \end{array}$ . Exponentiating the entropy of these weights gives the effective number of active covariance directions; division by $p$ produces $\dot { \overline { { r } } } _ { \mathrm { e f f } } \in ( 0 , 1 ] .$ . A value near one means variance is spread evenly across the available directions, whereas a value near zero means it is concentrated in relatively few directions. Thus, an isotropic population has $C _ { \mathrm { o f f } } = 0 , A _ { \mathrm { s p e c } } = 1$ , and $\overline { { r } } _ { \mathrm { e f f } } = 1$ Effective rank describes the covariance spectrum and is not an estimate of the nonlinear intrinsic dimension of the motion manifold.

Table 15 shows substantial spatiotemporal dependence after normalization. Across all window lengths, motion has much larger residual correlation and spectral anisotropy than its matched Gaussian, together with a far smaller normalized effective rank. For example, at $T _ { 0 } = 6 4 , 2 6 3 \mathrm { D }$ motion has $\dot { C _ { \mathrm { o f f } } } ~ = ~ . 1 4 8 , ~ A _ { \mathrm { s p e c } } = 2 1 1 6 . 0 .$ and $\overline { { r } } _ { \mathrm { e f f } } ~ = ~ . 0 0 7$ , whereas the matched Gaussian gives .007, 1.0, and 1.000. XYZ is even more correlated and spectrally concentrated, with .444, 1808.7, and .001, respectively. The same separation holds at 32 and 128 frames, directly verifying that marginal normalization does not make either motion space isotropic.

For comparison with latent-space generators, we apply the same estimator to the pretrained representations used by CMDM and SALAD. Both results use the common $T _ { 0 } = 6 4$ protocol and the same 20,850 motions, producing 16 latent frames for each method; we preserve each representation’s native frame structure when flattening and use one seeded posterior sample per motion, matching the variables seen during generator training.

Table 16 reports only the covariance diagnostics for CMDM and SALAD; we compare them here with the direct-motion results in Table 15. SALAD is the most isotropic representation by both $C _ { \mathrm { o f f } } = . 0 7 6$ and $\overline { { r } } _ { \mathrm { e f f } } = . 2 3 8$ . CMDM has slightly higher pairwise correlation than 263D (.181 versus .148), but its much smaller $A _ { \mathrm { s p e c } }$ (118.9 versus 2116.0) and larger $\overline { { r } } _ { \mathrm { e f f } }$ (.077 versus .007) show that variance is distributed over substantially more directions. These CMDM and SALAD result are consistent with variational regularization, although both remain far from their matched Gaussian references. Because dimensionality and preprocessing differ across representations, these statistics characterize covariance geometry rather than rank generators by sample quality. Nevertheless, they reveal clear differences between latent-space and motion-space representations.

We next substitute the extreme eigenvalues of each estimated shrinkage covariance into Eq. (10). This quantifies the conditioning of the corresponding spatiotemporal probability path without Monte

Table 17: Source scale changes path conditioning and generation quality. The columns headed by t report $\log _ { 1 0 } \kappa _ { t } ( s )$ , computed at $T _ { 0 } =$ 64 from the extreme eigenvalues of the estimated shrinkage covariance; lower is better. FID uses the matched clean-prediction configurations from Table 3: causal attention for 263D and bidirectional attention for XYZ.
<table><tr><td>Representation</td><td>Scale s</td><td> $t = 0 . 1$ </td><td> $t = 0 . 5$ </td><td> $t = 0 . 9$ </td><td> $\mathrm { F I D \downarrow }$ </td></tr><tr><td>263D</td><td>1</td><td>1.516</td><td>3.410</td><td>5.241</td><td> $. 1 1 1 ^ { \pm . 0 0 5 }$ </td></tr><tr><td rowspan="2">XYZ</td><td>5</td><td>.356</td><td>2.017</td><td>3.918</td><td> $\mathbf { . 0 4 6 ^ { \pm . 0 0 4 } }$ </td></tr><tr><td>1</td><td>1.253</td><td>3.136</td><td>5.023</td><td> $. 1 4 4 ^ { \pm . 0 0 9 }$ </td></tr><tr><td></td><td>5</td><td>.224</td><td>1.746</td><td>3.646</td><td> $\mathbf { . 0 3 8 ^ { \pm . 0 0 4 } }$ </td></tr></table>

Carlo noise. We report lo $\boldsymbol { \Sigma } _ { 1 0 } \boldsymbol { \kappa } _ { t } ( \boldsymbol { s } )$ at $T _ { 0 } = 6 4$ for an early path time $t = 0 . 1$ , the midpoint $t = 0 . 5$ and a late path time $t = 0 . 9$

Increasing s from 1 to 5 reduces $\log _ { 1 0 } \kappa _ { t }$ from $1 . 5 1 6 / 3 . 4 1 0 / 5 . 2 4 1$ to .356/2.017/3.918 at $t \ =$ 0.1/0.5/0.9 for 263D (Table 17). For $\mathrm { \bar { X } Y Z }$ , the corresponding values fall from $1 . 2 5 \dot { 3 } / 3 . 1 3 6 / 5 . 0 2 3$ to .224/1.746/3.646. Thus, s materially changes the conditioning of the correlated motion path at early, middle, and late times; it is not merely an inference-time diversity parameter. The matched clean-prediction ablations exhibit correspondingly large sensitivity: FID falls by 58.6% for causal 263D and by 73.6% for bidirectional XYZ when moving from $s = 1 \mathrm { t o } s = 5 .$ These controlled results establish the importance of source scale for the selected motion-space generators, while the covariance analysis verifies its path-level conditioning effect. They do not imply that improved conditioning alone causes the FID gains or that $s = 5$ is universally optimal: the best scale can change with the representation, attention graph, and prediction parameterization.

## E.3 WHY ATTENTION DEPENDS ON THE REPRESENTATION

The attention choice reflects what a frame token represents. Incremental features describe local changes whose global effect is obtained by forward accumulation, suggesting a filtering-style causal dependency. Absolute coordinates describe a globally coupled trajectory for which future observations can help smooth an earlier state. We formalize this distinction and its limitations below.

Let $\mathrm { A t t } ( i )$ be the set of motion-key frames visible to the prediction at frame i:

$$
\mathrm { A t t } _ { \mathrm { C } } ( i ) = \{ 1 , \ldots , i \} , \qquad \mathrm { A t t } _ { \mathrm { B } } ( i ) = \{ 1 , \ldots , T \} .\tag{19}
$$

The corresponding field component has the dependency

$$
\begin{array} { r } { \hat { \mathbf { v } } _ { \theta , i } ( \mathbf { x } _ { t } , t , \mathbf { c } ) = \hat { \mathbf { v } } _ { \theta , i } ( \mathbf { x } _ { t , \mathrm { A t t } ( i ) } , t , \mathbf { c } ) . } \end{array}\tag{20}
$$

To isolate the statistical cost of the causal restriction, set $Y _ { i } = \mathbf { X } _ { 1 , i } , \mathcal { G } _ { i } = \sigma ( \mathbf { X } _ { t , 1 : i } , t , \mathbf { c } )$ , and $\mathcal { H } = \sigma ( \mathbf { X } _ { t , 1 : T } , t , \mathbf { c } )$ . Let $R _ { \mathrm { C } , i }$ and $R _ { \mathrm { B } , i }$ be the mean-squared errors of the corresponding Bayesoptimal conditional means.

Proposition 3 (Causal excess prediction risk). For square-integrable motion,

$$
R _ { \mathrm { C } , i } - R _ { \mathrm { B } , i } = \mathbb { E } \left[ \left\| \mathbb { E } [ Y _ { i } \mid \mathcal { H } ] - \mathbb { E } [ Y _ { i } \mid \mathcal { G } _ { i } ] \right\| _ { 2 } ^ { 2 } \right] \geq 0 .\tag{21}
$$

Equality holds exactly when the full-sequence conditional mean is already measurable from the noisy prefix.

ProofofProposition 3. Because $\mathcal { G } _ { i } \subseteq \mathcal { H }$ , conditional expectation is an orthogonal projection in $L ^ { 2 }$ Add and subtract $\mathbb { E } [ Y _ { i } \mid \mathcal { H } ]$ inside the causal residual:

$$
Y _ { i } - \mathbb { E } [ Y _ { i } \mid { \mathcal { G } } _ { i } ] = Y _ { i } - \mathbb { E } [ Y _ { i } \mid { \mathcal { H } } ]\tag{22}
$$

$$
+ \mathbb { E } [ Y _ { i } \mid { \mathcal { H } } ] - \mathbb { E } [ Y _ { i } \mid { \mathcal { G } } _ { i } ] .\tag{23}
$$

The first term is orthogonal to every H-measurable random variable, including the second term. Taking squared norms and expectations gives

$$
R _ { \mathrm { C } , i } = R _ { \mathrm { B } , i } + \mathbb { E } \left\| \mathbb { E } [ Y _ { i } \mid \mathcal { H } ] - \mathbb { E } [ Y _ { i } \mid \mathcal { G } _ { i } ] \right\| _ { 2 } ^ { 2 } ,\tag{24}
$$

which proves Eq. (21). Equality holds if and only if the two conditional means agree almost surely. □

Additional context cannot increase Bayes-optimal squared error, but it need not provide useful information. The following idealized model identifies when the inequality is strict.

Proposition 4 (Increment prediction versus position smoothing). Let $V _ { k } \sim \mathcal { N } ( 0 , q )$ and $E _ { k } \sim$ $\mathcal { N } ( 0 , 1 )$ be mutually independent, with $t > 0$ and $\tau > 0$ . For an increment representation observed as $Z _ { k } = t V _ { k } + \tau E _ { k }$ , suffix observations $Z _ { i + 1 : T }$ do not reduce the Bayes riskfor V . For an absoluteposition representation $\begin{array} { r } { P _ { k } = \sum _ { r < k } V _ { r } } \end{array}$ observed as $Z _ { k } = t P _ { k } \overset { \cdot } { + } \tau E _ { k } ,$ , observing $Z _ { i + 1 }$ strictly reduces the Bayes riskfor P<sub>i</sub> whenever its prefix posterior variance is nonzero.

ProofofProposition 4. For the increment representation, let the noisy observation of frame k be $Z _ { k } = t V _ { k } + \tau E _ { k }$ , where $\tau > 0$ and the $E _ { k }$ are independent standard Gaussians. Independence across k gives $V _ { i } \perp Z _ { i + 1 : T } \mid Z _ { 1 : i } ,$ , hence $\mathbb { E } [ V _ { i } \mid Z _ { 1 : T } ] \stackrel { \cdot } { = } \mathbb { E } [ V _ { i } \mid Z _ { 1 : i } ]$ and Proposition 3 gives equal risks.

For the position representation, let $\begin{array} { r } { P _ { k } = \sum _ { r < k } V _ { r } } \end{array}$ and $Z _ { k } = t P _ { k } + \tau E _ { k }$ . Conditional on the prefix observations, $Z _ { i + 1 } = t P _ { i } + t V _ { i + 1 } + \tau E _ { i + 1 }$ . Since $V _ { i + 1 }$ and $E _ { i + 1 }$ are independent of $( P _ { i } , Z _ { 1 : i } )$

$$
\operatorname { C o v } ( P _ { i } , Z _ { i + 1 } \mid Z _ { 1 : i } ) = t \operatorname { V a r } ( P _ { i } \mid Z _ { 1 : i } ) .\tag{25}
$$

For $t > 0$ and nonzero prefix posterior variance, this conditional covariance is nonzero. Gaussian conditioning therefore strictly reduces $\mathrm { V a r } ( P _ { i } \mid Z _ { 1 : i } )$ when $Z _ { i + 1 }$ is added. The full suffix contains $Z _ { i + 1 }$ , so bidirectional Bayes risk is strictly lower. □

This is the classical filtering–smoothing distinction. Importantly, Propositions 3 and 4 do not claim that causal attention has a lower Bayes-optimal reconstruction error. Bidirectional context contains the causal context, so its optimal squared error can only be smaller; for independent increments the two risks are equal because the suffix is redundant. The empirical advantage of a causal mask must therefore come from its inductive bias in a finite model rather than from greater information. When suffix observations carry little task-relevant information, removing their edges reduces the dependency class, discourages a denoising shortcut based on symmetric temporal smoothing, and directs limited capacity toward prefix-conditioned transitions. Moreover, the flow-matching loss is a local regression objective, whereas FID and retrieval evaluate the distribution produced after repeatedly applying the learned field. A context pattern that eases framewise regression need not yield the best integrated generative dynamics.

HumanML3D is not a sequence of independent increments: it mixes root velocities, root-relative positions, rotations, local velocities, and contacts under a text condition. Its root trajectory nevertheless has a distinguished forward accumulation structure, while an individual frame already specifies a complete root-relative pose. This makes the idealized increment model a plausible explanation for why suffix motion can be redundant for important components, not a literal model of the entire representation. Conversely, absolute XYZ trajectories are temporally correlated observations of global joint positions, making their suffix genuinely informative. The proposition therefore supplies a mechanism, not a universal ranking of masks; the representation-specific preference remains an empirical question.

## E.4 EMPIRICAL ATTENTION ROUTING

We inspect eight models with a balanced $2 \times 2 \times 2$ comparison of 263D versus XYZ representations, causal versus bidirectional masks, and source scales $s \in \{ 1 , 5 \}$ . We use 512 HumanML3D test motions available in both representations, with identical captions and representation-specific Gaussian draws. We evaluate $t \in \{ \bar { 0 . 2 } , 0 . 5 , 0 . 8 \}$ and average within each motion before aggregating, so heads and layers are not treated as independent observations. Motion-direction statistics condition on the mass assigned to valid motion keys and therefore do not depend on prompt length or text-attention mass.

Table 18 reports the midpoint statistics for all eight checkpoints, and Fig. 5 visualizes their representation-dependent routing patterns. The $s \ : = \ : 5$ maps expose a learned difference within the same bidirectional graph: XYZ assigns only 15.8% of motion attention within ±4 frames, compared with 38.1% for 263D, and its normalized span is 20.9% rather than 12.7%. The independently trained s = 1 pair shows the same ordering: XYZ is less local (25.4% versus 50.3%) and broader (17.5% versus 9.7%). Averaged over the two selected scales, bidirectional XYZ remains less local (20.6% versus 44.2%), has a longer span (19.2% versus 11.2%), and assigns a slightly larger fraction to future motion (50.5% versus 48.3%). As shown in Fig. 6, the span ordering holds at every evaluated time: XYZ versus 263D is 22.7% versus 14.4% at t = 0.2, 19.2% versus 11.2% at $t = 0 . 5 ,$ , and 16.4% versus 8.0% at $t = 0 . 8$

Eight selected checkpoints at t = 0.5  
Table 18: Midpoint attention routing for all eight selected checkpoints. Representation specifies the motion input format, Scale is the Gaussian source scale s, and Mask is the temporal attention graph. The remaining columns report percentages at $t = 0 . 5$ , averaged over 512 paired test motions. Future is the fraction of motion-to-motion attention from query frame i assigned to later key frames $j > i ;$ Local ±4 is the fraction assigned within four frames of the query; and Span is the attentionweighted expected absolute offset $| j - i |$ , normalized by sequence length. Text is the fraction of a motion query’s total attention assigned to text keys, while Text→motion is the fraction of a text query’s total attention assigned to motion keys.
<table><tr><td>Representation</td><td>Scale</td><td>Mask</td><td>Future</td><td>Local ±4</td><td>Span</td><td>Text</td><td>Text→motion</td></tr><tr><td rowspan="4">263D</td><td rowspan="2"> $s = 1$ </td><td>Causal</td><td>0.0</td><td>43.6</td><td>13.6</td><td>48.9</td><td>0.0</td></tr><tr><td>Bidirectional</td><td>47.4</td><td>50.3</td><td>9.7</td><td>31.6</td><td>76.4</td></tr><tr><td rowspan="2"> $s = 5$ </td><td>Causal</td><td>0.0</td><td>46.6</td><td>10.4</td><td>27.2</td><td>0.0</td></tr><tr><td>Bidirectional</td><td>49.1</td><td>38.1</td><td>12.7</td><td>17.5</td><td>86.4</td></tr><tr><td rowspan="4">XYZ</td><td rowspan="2"> $s = 1$ </td><td>Causal</td><td>0.0</td><td>24.2</td><td>20.6</td><td>61.6</td><td>0.0</td></tr><tr><td>Bidirectional</td><td>49.6</td><td>25.4</td><td>17.5</td><td>24.2</td><td>86.1</td></tr><tr><td rowspan="2"> $s = 5$ </td><td>Causal</td><td>0.0</td><td>24.5</td><td>18.2</td><td>52.0</td><td>0.0</td></tr><tr><td>Bidirectional</td><td>51.4</td><td>15.8</td><td>20.9</td><td>29.1</td><td>76.7</td></tr></table>

![](images/0da714cf7310f6d145064eae792311eca4bc542bb91332800a45f85175e083db.jpg)  
Figure 5: Representation-dependent attention routing for all eight selected checkpoints. Rows compare 263D and XYZ; columns compare causal and bidirectional masks at source scales $s = 1$ and $s = 5 .$ . Each map averages 512 paired test motions, four heads, and eight layers at $t = 0 . 5$ after conditioning each motion-query row on its mass assigned to valid motion keys. The causal mask removes the suffix exactly. Within the bidirectional graph, XYZ distributes attention over a broader temporal range than 263D at both source scales.

These statistics are consistent with the proposed smoothing role of bidirectional XYZ attention: when suffix edges are available, both selected XYZ models use broader two-sided context rather than merely retaining a nominally dense mask. Conversely, all four selected causal checkpoints assign exactly zero weight to future motion and prevent text queries from reading motion, verifying the intended filtering graph and excluding an indirect suffix path through the updated text stream. Attention weights describe routing rather than causal attribution, however; values, output projections, residual paths, and subsequent layers also determine the influence on the generated motion.

![](images/1d9589fd5dd7f15cf7029490f6c1b1b47cc8c628314578c7409e52654815c91b.jpg)

![](images/69e16f2ff8539ccfab00fc9e0455ab61b89daf628ec10bffe97752584e146bfd.jpg)  
Figure 6: Bidirectional routing across flow time for the selected checkpoints. Each checkpoint statistic averages 512 paired test motions. Thin curves show the individual $s \ = \ 1$ and $s \ = \ 5$ models; markers and error bars show their mean and standard error. Future attention is the fraction of motion-to-motion attention from a query frame i assigned to later key frames $j > i ;$ span is the attention-weighted expected absolute offset $| j - i |$ , normalized by sequence length. XYZ has a longer normalized temporal span at every evaluated time, while both representations allocate approximately half of their motion attention to future frames.

Beyond the Gaussian analysis in Proposition 4, Proposition 5 establishes a distribution-independent dynamical guarantee of the causal mask: the generated prefix is autonomous from perturbations to the source suffix.

Proposition 5 (Prefix autonomy). Assume the causal field is locally Lipschitz in motion state, so its sampling ODE has a unique solution. For any $k \leq T _ { : }$ , the generated trajectory $\mathbf { x } _ { t , 1 : k }$ depends only on the source prefix $\mathbf { x } _ { 0 , 1 : k }$ and text c. Changing $\mathbf { x } _ { 0 , k + 1 : T }$ cannot alter this prefix. Wherever thefield is differentiable, its motion Jacobian is block lower triangular.

ProofofProposition 5. For any $k ,$ causality implies that the first k sampling equations can be writ ten as the closed subsystem

$$
\frac { d { \mathbf { x } } _ { t , 1 : k } } { d t } = \hat { \mathbf { v } } _ { \theta , 1 : k } ( { \mathbf { x } } _ { t , 1 : k } , t , { \mathbf { c } } ) .\tag{26}
$$

Its initial condition is $\mathbf { x } _ { 0 , 1 : k }$ . Local Lipschitz continuity gives uniqueness, so two full-sequence initial states with the same prefix must induce the same prefix trajectory even if their suffixes differ. If the field is differentiable, Eq. (20) with $\mathrm { { A t t } = \mathrm { { A t t } _ { C } } }$ gives $\partial \bar { \bf v } _ { \theta , i } / \partial \bar { \bf x } _ { t , j } = { \bf 0 }$ for $j > i ,$ , which is exactly block lower triangularity. □

Causality thus constrains information flow without making generation autoregressive: all T frames are initialized together and updated at every ODE step. The lower-triangular dependency prevents a suffix error or source perturbation from feeding back into an earlier prefix through subsequent solver evaluations. This structural guarantee does not by itself prove smaller numerical error or better sam ples, but it removes one route for global error propagation and preserves the forward accumulation semantics of incremental root motion. Bidirectional attention gives up prefix autonomy in exchange for full-sequence smoothing and two-sided propagation of spatial constraints, which is beneficial when coordinates are absolute and globally coupled. These complementary properties motivate the attention variants evaluated in Section 4.3.