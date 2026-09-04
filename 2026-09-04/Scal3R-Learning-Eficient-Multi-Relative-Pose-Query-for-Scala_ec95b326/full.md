# Scal3R: Learning Eficient Multi-Relative Pose Query for Scalable Online 3D Reconstruction

Chin-Yang Lin<sup>1,2</sup> Yang-Che Sun<sup>1</sup> Cheng Sun<sup>2</sup> Fu-En Yang<sup>2</sup> Min-Hung Chen<sup>2</sup> Yen-Yu Lin<sup>1</sup> Wei-Chen Chiu<sup>1</sup> Yu-Lun Liu<sup>1</sup>

<sup>1</sup>Department of Computer Science, National Yang Ming Chiao Tung University <sup>2</sup>NVIDIA

![](images/0f71b824d67faa2ce73e80c405f0c635968d9ef1086430df56712db9e5840f63.jpg)  
Fig. 1: Scal3R enables scalable online 3D reconstruction on long video streams. (a) Existing feed-forward models, such as CUT3R [82] and STream3R [40], regress absolute global poses (P<sub>t</sub>) relative to a fixed first frame. This forces extrapolation far beyond its training distribution, resulting in catastrophic drift and geometric collapse on kilometer-scale sequences. (b) Scal3R reformulates the problem into a local multireference relative pose query (T<sup>ˆ</sup> <sub>t←r</sub> ). By injecting lightweight learnable tokens into a frozen backbone and aggregating relative constraints via online Pose-Graph Optimization (PGO), Scal3R suppresses long-range drift and recovers globally consistent geometry. It requires only 8 hours of fine-tuning on a single GPU.

Abstract. Online 3D reconstruction models perform poorly on long videos. This happens because regressing poses relative to a fixed first-frame anchor forces extrapolation far beyond the training distribution. Small drifts accumulate and amplify into significant geometric collapse. However, we observe that per-frame depth remains stable throughout this failure. The backbone’s local geometry remains intact; only the global pose head breaks down. Motivated by this decoupling, we introduce Scal3R. This approach reformulates online reconstruction as multi-reference relative pose querying. We use lightweight learnable tokens, which make up about ∼1% of the parameters, and inject them into a completely frozen

backbone via asymmetric attention. This setup queries poses relative to multiple past keyframes. An online pose-graph optimization system with loop closure suppresses long-range drift. Scal3R reaches convergence in 8 hours on a single GPU. It reduces the average ATE by over 60% on KITTI compared to the online baseline. It also achieves state-of-the-art performance across Virtual KITTI, Sintel, TUM-Dynamic, ScanNet, and 7-Scenes. Project page: https://linjohnss.github.io/scal3r/

Keywords: Online 3D reconstruction · Relative pose estimation · Prompt tuning · Pose graph optimization

## 1 Introduction

Feed-forward models such as CUT3R [82] and STream3R [40] enable real-time 3D reconstruction by decoding per-frame geometry into a unified global coordinate system via direct pose regression relative to the first frame. While this globalanchor paradigm works well for short sequences, it faces severe stability and scalability bottlenecks in long-range environments.

This failure stems from two fundamental issues. First, existing 3D datasets [85, 94] cover limited scene scales. This means that models trained on short sequences must extrapolate to global coordinates far outside their training distribution. When deployed on real-world trajectories spanning hundreds of meters, even minor feature drift is amplified into geometric collapse (Fig. 2a). Second, realworld camera motions are highly variable, and for unbounded video streams, the global coordinate system will inevitably encounter out-of-distribution trajectories, making feed-forward global pose regression theoretically unable to scale.

As shown in Fig. 2b, this failure is highly localized: while global pose errors diverge catastrophically, per-frame depth remains consistently stable. This indicates that the backbone’s local geometric representations are intact and the failure is isolated to the global pose regression head. Motivated by this, we shift from global pose regression to local relative querying: rather than forcing the model to extrapolate a global mapping, we exploit its well-learned local geometry to estimate stable relative transformations via a visual query mechanism conditioned on reference viewpoints, with the backbone entirely frozen.

We present Scal3R, an eficient framework for scalable online 3D reconstruction (Fig. 1). A small set of learnable tokens is injected into the frozen backbone via asymmetric attention: pose tokens attend to image features as queries while image tokens compute self-attention exclusively among themselves, preserving the frozen representation space. Each pose token predicts the relative transformation between the current frame and a historical reference, constraining localization to the local viewpoint domain where the model is most reliable.

To maintain global consistency, Scal3R incorporates multi-reference relative querying and online pose-graph optimization: relative poses are queried against multiple dynamically selected reference frames simultaneously, aggregated via PGO into a drift-free trajectory, and supplemented by a loop closure mechanism that integrates naturally into the multi-reference pipeline. As shown in Fig. 1,

![](images/e6aca9b8806a2cb5d8642eab43a513595898b2ee1cfcdb1cf0c4528cdd72631c.jpg)

![](images/ab8a83e121817084dc39ac5a4eb5ab7a273c44aa11e72ba54c681095f6afd753.jpg)  
Fig. 2: Global pose regression fails under out-of-distribution sequences, while local geometry remains reliable. (a) Feed-forward models like CUT3R [82] produce accurate reconstructions for in-distribution frames (blue). However, they sufer from severe geometric collapse when extrapolating to unseen long-range trajectories (orange). (b) Our error correlation analysis reveals a critical decoupling. The global position error diverges catastrophically in the out-of-distribution region (red), while per-frame depth remains stable (blue). This finding suggests that the backbone’s local geometric representations are intact. This motivates Scal3R to freeze the base model and replace fragile global regression with stable multi-reference relative pose querying.

Scal3R produces geometrically consistent reconstructions on sequences spanning hundreds of meters, finetuned in only 8 hours on a single GPU.

In summary, our contributions are as follows:

– We identify that global extrapolation instability and limited training data coverage cause long-sequence collapse in long-sequence reconstruction. We also show that local geometric representations remain reliable throughout.

– We propose Scal3R. It introduces multi-reference relative pose querying via visual prompt tuning on a frozen backbone. Asymmetric attention injection ensures that pose learning does not degrade point cloud quality.

– Integrating multi-reference querying with online PGO, Scal3R achieves low drift on kilometer-scale sequences. It converges in about 8 hours on a single GPU using only 4-view training samples.

## 2 Related Work

Multi-view 3D Reconstruction. Early reconstruction relied on ofline Structurefrom-Motion [59, 61] and Multi-View Stereo [62] pipelines. Optimization-based approaches jointly refine camera poses with the scene representation [7,45,49,55], but remain per-scene and ofline. Feed-forward methods dramatically improved eficiency by predicting geometry in a single forward pass [6, 10, 13, 23, 24, 34, 41, 67, 71, 83, 93, 100], with recent transformer-based models scaling to large unordered collections via joint pose-and-geometry prediction [80, 81, 86] and eficient aggregation strategies [14, 17, 25, 26, 38, 46–48, 64, 69, 70, 74, 77, 79, 88, 92, 98]. A complementary trend adapts frozen 3D foundation models for downstream tasks without backbone retraining [29, 35, 65]. Scal3R follows this paradigm but uniquely targets scalable pose estimation on unbounded video streams, where batch-processing systems fundamentally cannot operate.

Online 3D Reconstruction. Online methods shift from batch processing to incremental inference, processing video frame by frame, achieved through recurrent TSDF fusion [68], diferentiable bundle adjustment [75], and progressive radiance-field optimization [55]. CUT3R [82] and STream3R [40] bring this to 3D foundation models via persistent state updates and causal Transformers, spawning a broad family of streaming systems [2, 5, 15, 39, 44, 54, 63, 72, 78, 89, 95, 101, 104] and SLAM integrations [21, 31, 50, 52, 53, 57, 96, 97, 99]. However, all share a critical flaw in that poses are regressed relative to the first frame, anchoring the trajectory to a single global reference. As shown in Figs. 1 and 2, this strategy becomes increasingly fragile as sequences grow, where small feature drifts are amplified into catastrophic geometric collapse.

Long-sequence Streaming 3D Reconstruction. Suppressing drift over kilometer-scale sequences remains an open challenge, addressed through test-time gradient updates [11], training-free memory management [95, 101], long-range token pools [44], explicit spatial memory [89], stage-decoupled streaming [16], and ofline global optimization [19, 20, 91], each trading of online capability against global consistency. Earlier per-scene methods handle long casual videos by incrementally estimating poses with learned 3D priors [45] or by progressively allocating local radiance fields rather than a single global representation [55], an early departure from single-anchor formulations. A unifying insight from the visual odometry literature is that relative formulations generalize better than absolute ones [9, 22]. This insight guides Scal3R. Rather than improving global-anchor regression, we reformulate the problem as multi-reference relative pose querying on a frozen backbone, eliminating the root extrapolation failure with only ∼1% additional parameters.

Eficient Prompt Tuning. Parameter-eficient adaptation has shown that frozen pre-trained models need very little change to transfer well. Adapters [28], prefix tokens [43], soft prompts [42], low-rank perturbations [30], and parallel adapter modules [8] all match or exceed full fine-tuning at under 2% of parameters, a finding confirmed broadly across vision transformers [32, 36, 60, 76, 90]. This paradigm has since reached 3D vision, where geometry-aware prompts and low-rank adapters on frozen 3D transformers [1, 73, 84, 102] and reconstruction backbones [51, 87] consistently match full fine-tuning, and attention-level token gating on a frozen large reconstruction model enables mesh editing without backbone retraining [29]. In 3D reconstruction, Human3R [12] first demonstrates prompt tuning on a frozen CUT3R for joint human-scene reconstruction. Scal3R is the first to apply this paradigm to relative pose estimation, recasting it as a multi-reference prompt query task via asymmetric attention injection that preserves the backbone’s pointmap quality while gaining globally consistent motion representations.

![](images/6da612ac192e5973ea84293aa1eca10626b5bdc91c2a5a54eb2acd6e7665f4f1.jpg)  
Fig. 3: Overview of the Scal3R framework. For an incoming frame Image , a frozen encoder extracts dense image tokens. At the same time, historical camera tokens from selected reference frames $( r _ { k } , r _ { k - 1 } , \ldots )$ are projected using lightweight trainable MLPs to generate relative pose tokens. These tokens are concatenated and sent into a completely frozen 3D reconstruction decoder $( e . g .$ ., CUT3R [82] or STream3R [40]). Our Asymmetric Attention Injection mechanism is crucial as it ensures that relative pose tokens serve only as queries to extract geometric cues, while image tokens perform self-attention exclusively among themselves. This method preserves the original high quality point cloud generation $( X _ { t } )$ through the frozen Point Head, while the trainable Relative Pose Head predicts robust multi-reference relative transformations $( \hat { \mathbf { T } } _ { t  r _ { k } } )$ Finally, an online inference backend (PGO and loop closure) aggregates these local constraints to produce a globally consistent trajectory.

## 3 Method

## 3.1 Overview

Scal3R addresses online 3D reconstruction by reformulating global pose regression as a multi-reference relative pose query problem. Given a streaming sequence of images $\mathcal { T } = \{ I _ { 1 } , I _ { 2 } , \ldots , I _ { T } \}$ , instead of directly regressing absolute camera poses in a unified world coordinate system, we query relative poses with respect to a set of maintained reference frames. This reformulation fundamentally eliminates the long-horizon extrapolation instability that plagues existing global-regression approaches.

Architecturally, Scal3R builds upon frozen pretrained online 3D reconstruction backbones $( e . g .$ , CUT3R [82] or STream3R [40]), preserving their rich spatiotemporal geometric priors. A lightweight set of learnable tokens is injected into the frozen decoder via an asymmetric attention mechanism, enabling relative pose decoding without modifying the pretrained weights. At the backend, an online Pose-graph Optimization (PGO) module aggregates the predicted pairwise relative constraints into a globally consistent trajectory. An overview of the full pipeline is shown in Fig. 3.

## 3.2 Preliminaries: Online 3D Reconstruction Backbones

We briefly review the two representative backbone paradigms underlying Scal3R.

Persistent state model (CUT3R). At each timestep t, the frozen online 3D reconstruction backbone processes the current frame $I _ { t }$ together with a persistent hidden state $S _ { t - 1 }$ encoding the scene history, producing an updated state $S _ { t }$ and local geometry prediction $G _ { t }$

Causal Transformer model (STream3R). At each timestep t, the frozen online 3D reconstruction backbone processes the current frame $I _ { t }$ via causal attention over a sliding feature window $\{ F _ { t - k } , \ldots , F _ { t } \}$ to perform cross-temporal geometric alignment in feature space, producing local geometry prediction $G _ { t }$

Both paradigms share a common decoding structure. At each frame, the decoder maintains image feature tokens $F _ { t } \in \mathbb { R } ^ { \breve { H } \times W \times C }$ and a dedicated camera token $\mathbf { c } _ { t } \in \mathbb { R } ^ { D }$ . Pointmaps are decoded from $F _ { t }$ for local geometry, while the global camera pose $P _ { t } \in S E ( 3 )$ relative to the first frame is regressed from $\mathbf { c } _ { t }$

Although efective for short sequences, global-reference regression degrades over long sequences: as the sequence grows, the model must align each new frame to an increasingly distant first-frame coordinate system, causing small feature drifts to be amplified into severe geometric collapse at the decoding stage. Scal3R retains the rich representations $F _ { t }$ and $\mathbf { c } _ { t }$ learned by these backbones, while discarding their unstable global pose regression heads. Instead, we leverage historical camera tokens $\left\{ \mathbf { c } _ { r _ { k } } \right\}$ stored in the pose token bufer as geometric conditioning signals to enable scalable relative pose queries (Sec. 3.3).

## 3.3 Multi-Reference Relative Pose Tuning

Our core contribution is a parameter-eficient prompt tuning mechanism that enables robust multi-reference relative pose prediction on a completely frozen backbone. The total number of newly introduced parameters accounts for ∼1% of the backbone’s total parameter count. To endow the model with the ability to query multiple reference viewpoints simultaneously, we maintain a pose token bufer for storing the camera tokens of selected past keyframes. We learn a shared base query token $\mathbf { q } \in \mathbb { R } ^ { D }$ that serves as a query template directing the decoder to extract the geometric relationship between the current frame and a given reference frame. For each reference slot $k ,$ the corresponding reference frame features retrieved from the bufer are projected into feature space via a lightweight MLP and fused with the base token by additive injection:

$$
\tilde { \mathbf { q } } _ { k } = \mathbf { q } + \mathrm { M L P } ( \mathbf { c } _ { r _ { k } } ) ,\tag{1}
$$

where ${ \bf c } _ { r _ { k } }$ is the camera token of the k-th reference frame. This dynamic assembly allows the system to flexibly scale the number of active queries K based on available references, ensuring robustness during sequence initialization or bufer resets. Importantly, since each token queries independently, the number of reference frames can be freely extended at inference time without retraining.

![](images/7b36f6f82cd9c1dd9889305a52d6a0313dc316d65eaef73d9edf40bcb79ad913.jpg)  
Fig. 4: Asymmetric Attention Injection. Relative pose tokens are injected only as additional queries (Q<sup>′</sup>), without modifying the keys and values of image tokens. This asymmetric design enables pose conditioning while preserving the pretrained image representations intact.  
Fig. 5: Inference pipeline and pose graph structure. To suppress accumulative drift in long sequences, incoming frames are registered via keyframe selection and pose-graph optimization over sequential, multi-reference (K=3), and loop closure edges before being committed to the pose token bufer.

Asymmetric Attention Injection. Naively inserting new tokens into the decoder’s self-attention would perturb the attention distribution of image tokens, degrading pointmap reconstruction quality. We instead propose asymmetric attention injection (Fig. 4), where the pose query tokens {q˜ } participate in decoder attention exclusively as queries, attending to all image tokens to extract geometric features, while image tokens compute their Keys and Values without attending to the pose query tokens. For a decoder layer with image tokens X:

$$
\begin{array} { r } { \tilde { \mathbf { q } } _ { k } ^ { \ell + 1 } = \mathrm { A t t e n t i o n } \Big ( \mathbf { Q } = \tilde { \mathbf { q } } _ { k } ^ { \ell } , ~ \mathbf { K } = \mathbf { X } ^ { \ell } , ~ \mathbf { V } = \mathbf { X } ^ { \ell } \Big ) , } \end{array}\tag{2}
$$

$$
\mathbf { X } ^ { \ell + 1 } = \mathrm { S e l f A t t e n t i o n } \Big ( \mathbf { X } ^ { \ell } \Big ) .\tag{3}
$$

This one-directional information flow guarantees that the image feature representation space remains identical to that of the original frozen model, fully preserving pointmap reconstruction fidelity. No attention mask is needed, as pose tokens never enter the image K/V sequence.

Relative Pose Decoding and Loss. After multi-layer feature exchange, each pose query token $\tilde { \mathbf { q } } _ { k }$ encapsulates the relative geometric constraint between the current frame t and its corresponding reference frame $r _ { k }$ . A lightweight MLP head decodes these tokens into relative poses. We adopt the 6D rotation representation [103] to ensure continuity in the rotation space, and output the relative transformation:

$$
\hat { \bf T } _ { t  r _ { k } } = \mathrm { H e a d } ( \tilde { \bf q } _ { k } ) \in S E ( 3 ) .\tag{4}
$$

The training loss supervises rotation R and translation t separately, where $\mathbf { R } _ { p r e d } ^ { k }$ and $\mathbf { t } _ { p r e d } ^ { \bar { k } }$ denote the rotation matrix and translation vector decomposed

from $\hat { \mathbf { T } } _ { t  r _ { k } }$ , and $\mathbf { R } _ { g t } ^ { k } , \mathbf { t } _ { g t } ^ { k }$ are the corresponding ground-truth components. To handle monocular scale ambiguity, translation vectors are scale-aligned before loss computation. The total loss aggregates over all K reference links:

$$
\mathcal { L } = \sum _ { k = 1 } ^ { K } \left( \lambda _ { R } \left. \mathbf { R } _ { p r e d } ^ { k } - \mathbf { R } _ { g t } ^ { k } \right. _ { F } + \lambda _ { t } \left. \mathbf { t } _ { p r e d } ^ { k } - \mathbf { t } _ { g t } ^ { k } \right. _ { 2 } \right) ,\tag{5}
$$

where $\lambda _ { R }$ and $\lambda _ { t }$ are loss weighting hyperparameters.

## 3.4 Online Pose-graph Optimization

While multi-reference relative pose predictions provide accurate pairwise constraints, naively chaining them accumulates drift over long sequences. We therefore integrate an online pose-graph optimization (PGO) framework (Fig. 5) that uses the predicted relative poses as between-factors and performs incremental trajectory correction as new frames arrive.

Keyframe Selection. Including every frame in the pose graph introduces numerical redundancy and unnecessary computation. We adopt an online 3D overlap-based keyframe selection strategy. A KD-tree spatial index maintains the reconstructed 3D point cloud; the visible point set for each frame is determined by projecting predicted 3D points onto a unit sphere and computing the angular overlap with past keyframes, following [5], to avoid interference from geometrically non-adjacent regions. For each incoming frame t, the predicted 3D points are transformed to world coordinates via the current pose estimate. If the depthnormalized overlap score falls below a threshold $\tau _ { \mathrm { o v e r l a p } }$ and the median depth confidence exceeds $\tau _ { \mathrm { c o n f } } ,$ the frame is designated as a keyframe, indicating novel geometry with reliable prediction quality. Only keyframes update the frozen decoder’s KV cache and enter the pose token bufer. Non-keyframe KV states are discarded by restoring the pre-forward snapshot, keeping the streaming decoder state clean. The first $N _ { \mathrm { i n i t } }$ frames are unconditionally treated as keyframes to initialize the system.

Pose-graph Optimization. We model the trajectory as a factor graph where each camera pose $T _ { t } \in S E ( 3 )$ is a variable node. The multi-reference relative poses form between-factors connecting the current frame to its $K _ { t }$ references:

$$
E _ { \mathrm { b e t w e e n } } = \sum _ { t } \sum _ { k = 1 } ^ { K _ { t } } \rho ( \| \log ( T _ { r _ { k } } ^ { - 1 } T _ { t } \cdot \hat { T } _ { t  r _ { k } } ) \| _ { { \boldsymbol { \Sigma } } _ { t , k } } ^ { 2 } ) ,\tag{6}
$$

where $\hat { T } _ { t  r _ { k } }$ is the predicted relative pose, $\Sigma _ { t , k }$ is a diagonal noise covariance, and $\rho ( \cdot )$ is the Huber robust kernel to downweight outlier constraints. To account for higher uncertainty in predictions between temporally distant frame pairs, we adopt a gap-dependent noise model where the standard deviation scales as $\sigma = \sigma _ { \mathrm { b a s e } } \cdot \varDelta ^ { 0 . 5 }$ with frame gap $\varDelta .$ . We employ iSAM2 [37] for incremental optimization. Upon each new frame arrival, the factor graph is updated and eficiently re-optimized via the Bayes tree structure. Optimized poses are written back to the bufer so that subsequent frames use corrected references.

For long sequences, the frozen decoder’s streaming state is reset every $N _ { \mathrm { r e s e t } }$ frames to prevent memory overflow and feature degradation. To maintain posegraph connectivity across resets, the last frame of each segment is re-fed as the first frame of the next segment, with a tight identity constraint imposed between the two corresponding nodes in the factor graph.

Loop Closure. Despite PGO continuously correcting local drift, long-term trajectory consistency requires explicitly detecting and closing loops when the camera revisits previously observed regions. A key advantage of our multi-reference design is that loop closure integrates naturally into the existing inference pipeline. When a loop candidate is detected between the current frame t and a past keyframe $r _ { \mathrm { l o o p } }$ , the archived camera token of $r _ { \mathrm { l o o p } }$ is simply re-injected into the pose token bufer as an additional reference slot. The frozen model then predicts a long-range relative pose constraint $\hat { \mathbf { T } } _ { t  r _ { \mathrm { l o o p } } }$ without any architectural modification, which is added to the pose graph as a high-confidence edge with a tight Gaussian noise model.

For loop detection, we employ a pretrained DINOv2 [58] backbone with a SALAD aggregation layer [33] to produce discriminative scene-level descriptors, indexed online via FAISS over keyframes only. Candidates are filtered by co sine similarity threshold $\tau _ { \mathrm { s i m } }$ , minimum temporal gap $\tau _ { \mathrm { g a p } }$ , and non-maximum suppression within a window $w _ { \mathrm { n m s } }$ to suppress redundant detections.

## 4 Experiments

## 4.1 Implementation Details

Model Configurations. We build upon two representative online 3D reconstruction backbones: CUT3R, which centers on persistent state updates, and STream3R, which is based on causal Transformers. In our experiments, we employ their 24-layer Transformer backbones (comprising a DINOv2 encoder and a Transformer decoder) and keep them entirely frozen to leverage their strong spatiotemporal geometric priors. For each incoming frame, we introduce a set of lightweight, learnable relative pose query tokens, which account for only approximately 1% of the total model parameters. These tokens are injected into the decoder layers via asymmetric attention injection to extract geometric constraints of the current frame relative to the reference frames in the pose token bufer. A pose decoding head then maps these features into SE(3) space, predicting the 6D rotation and translation vectors.

Training. We fine-tune our model on the TartanAir [85] dataset, which provides diverse scenarios and precise trajectory ground truth. To ensure robustness to varying motion velocities and baseline lengths, we adopt a Random Interval

Table 1: Camera pose estimation on KITTI (ATE ↓). The upper block lists ofline baselines, and the lower block reports online methods. Scal3R reduces average ATE by over 60% compared to the strongest online competitor TTT3R, with particularly pronounced gains on long-range sequences. - denotes OOM or tracking failure.
<table><tr><td rowspan="2">Methods</td><td colspan="11">KITTI (ATE ↓)</td><td rowspan="2">Avg.</td></tr><tr><td>00 4542× 3.7km</td><td>01 1101× 2.5km</td><td>02 4661× 5.1km</td><td>03 801× 0.6km</td><td>04 271× 0.4km</td><td>05 2761× 2.2km</td><td>06 1101× 1.2km</td><td>07 1101× 0.7km</td><td>08 4071× 3.2km</td><td>09 1591× 1.7km</td><td>10 1201× 0.9km</td></tr><tr><td>VGGT [80]</td><td>1</td><td>1</td><td>1</td><td></td><td></td><td></td><td></td><td>1</td><td>1</td><td>一</td><td></td><td></td></tr><tr><td>Fast3R [92]</td><td>一</td><td>723.1</td><td></td><td>166.9</td><td>112.2</td><td></td><td>137.4</td><td>90.9</td><td></td><td>225.6</td><td>211.7</td><td>238.3</td></tr><tr><td>DA3 [47]</td><td>一</td><td>-</td><td>一</td><td>一</td><td>12.6</td><td></td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td><td>12.6</td></tr><tr><td> $\pi ^ { 3 } \ [ 8 \bar { 6 } ]$ </td><td>1</td><td>-</td><td>=</td><td>=</td><td>2.3</td><td>1</td><td>4</td><td>=</td><td>一</td><td>一</td><td>一</td><td>2.3</td></tr><tr><td>MASt3R-SLAM [57]</td><td>188.5</td><td>562.9</td><td>282.4</td><td>121.7</td><td>92.6</td><td>一</td><td>57.2</td><td>77.0</td><td>263.6</td><td>184.1</td><td>179.1</td><td>200.9</td></tr><tr><td>MUSt3R [5]</td><td></td><td>490.1</td><td></td><td>121.5</td><td>58.1</td><td></td><td>100.6</td><td>66.8</td><td></td><td></td><td></td><td>167.4</td></tr><tr><td>CUT3R [82]</td><td>191.0</td><td>644.1</td><td>295.4</td><td>150.1</td><td>20.1</td><td>155.6</td><td>132.7</td><td>72.8</td><td>231.8</td><td>206.2</td><td>179.6</td><td>207.2</td></tr><tr><td>Point3R [89]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>STream3R [40]</td><td>189.0</td><td>694.9</td><td>302.6</td><td>162.7</td><td>100.5</td><td>159.1</td><td>122.7</td><td>87.1</td><td>264.5</td><td>220.9</td><td>195.9</td><td>227.3</td></tr><tr><td>WinT3R [44]</td><td></td><td>698.1</td><td>一</td><td>144.5</td><td>89.4</td><td>150.8</td><td>135.8</td><td>76.0</td><td>242.8</td><td>200.9</td><td>210.3</td><td>216.5</td></tr><tr><td>StreamVGGT [104]</td><td>I</td><td></td><td></td><td></td><td>98.8</td><td></td><td></td><td></td><td></td><td></td><td></td><td>98.8</td></tr><tr><td>TTT3R [11]</td><td>178.4</td><td>529.1</td><td>280.8</td><td>98.0</td><td>11.4</td><td>147.9</td><td>132.1</td><td>70.2</td><td>240.2</td><td>191.0</td><td>125.2</td><td>182.2</td></tr><tr><td>Ours (CUT3R)</td><td>45.3</td><td>165.0</td><td>139.9</td><td>33.9</td><td>9.9</td><td>29.4</td><td>47.4</td><td>6.4</td><td>227.0</td><td>29.6</td><td>33.2</td><td>69.7</td></tr><tr><td>Ours (STream3R)</td><td>57.9</td><td>176.4</td><td>170.8</td><td>10.9</td><td>9.3</td><td>39.1</td><td>18.1</td><td>15.2</td><td>173.8</td><td>73.9</td><td>33.9</td><td>70.8</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Sampling strategy during training. For each training sample, we select 4 views from a sequence, with one serving as the current frame $I _ { t }$ and the remaining three as reference frames $\left( K = 3 \right)$ , and randomly perturb the temporal intervals between frames. This mechanism forces the model to extract stable relative pose representations under varying levels of geometric constraint. Since each pose query token attends independently, the number of reference frames can be freely scaled at inference time without retraining; we use $K = 1 2$ during inference. For the long outdoor benchmarks (KITTI and vKITTI), we reset the frozen decoder’s streaming state every $N _ { \mathrm { r e s e t } } = 1 0$ frames; all other datasets use no reset. We train with a batch size of 8 for 40 epochs using the AdamW optimizer with a learning rate of $1 \times 1 0 ^ { - 4 }$ . Thanks to the frozen backbone and lightweight query tokens, the entire fine-tuning converges in approximately 8 hours on a single NVIDIA A100 GPU, avoiding the collapse of geometric priors commonly observed in full-parameter fine-tuning on small-scale datasets.

Baselines. We compare Scal3R with ofline transformers, streaming models, and SLAM-style systems. Ofline transformers include VGGT [80], $\pi ^ { 3 }$ [86], Fast3R [92], and DA3 [47]. Streaming baselines include CUT3R [82], MUSt3R [5], TTT3R [11], STream3R [40], WinT3R [44], StreamVGGT [104], and Point3R [89]. MASt3R-SLAM [57] is included as an incremental SLAM counterpart. All methods operate in an intrinsic-free setting, taking only RGB input without known camera intrinsics, and are evaluated with oficial default settings under a unified protocol.

Table 2: Camera pose estimation on Virtual KITTI (ATE ↓). The upper block lists ofline baselines, the middle block reports online methods, and the lower block presents our Scal3R variants. Scal3R surpasses all streaming methods by a large margin and approaches the accuracy of ofline ap proaches on long-range sequences. - denotes OOM or tracking failure.  
Table 3: Camera pose estimation on Sintel, TUM-Dynamic, and ScanNet (ATE ↓). Scal3R achieves state-of-the-art performance across all three benchmarks, demonstrating strong generalization to diverse unseen indoor and synthetic environments.
<table><tr><td rowspan="2">Methods</td><td colspan="5">vKITTI (ATE ↓)</td><td rowspan="2">Avg.</td></tr><tr><td>Scene01 447x, 1.3km</td><td>Scene02 233x, 0.6km</td><td>Scene06 270x, 0.6km</td><td>Scene18 339x, 1.1km</td><td>Scene20 837x, 2.5km</td></tr><tr><td>VGGT [80]</td><td></td><td>0.22</td><td></td><td></td><td></td><td>0.22</td></tr><tr><td>Fast3R [92]</td><td>72.71</td><td>23.72</td><td>6.05</td><td>58.93</td><td>159.40</td><td>64.16</td></tr><tr><td>DA3 [47]</td><td>25.22</td><td>0.22</td><td>0.17</td><td>12.53</td><td>182.89</td><td>44.20</td></tr><tr><td>π⁵ [86]</td><td>4.63</td><td>0.33</td><td>0.18</td><td>3.04</td><td></td><td>2.04</td></tr><tr><td>MASt3R-SLAM [57]</td><td>75.16</td><td>=</td><td>7.056</td><td>=</td><td></td><td>|41.11</td></tr><tr><td>MUSt3R [5]</td><td>77.23</td><td>7.39</td><td>0.28</td><td>48.53</td><td>170.91</td><td>60.87</td></tr><tr><td>CUT3R [82]</td><td>59.76</td><td>29.73</td><td>0.65</td><td>44.89</td><td>146.92</td><td>56.39</td></tr><tr><td>Point3R [89]</td><td>75.70</td><td>30.44</td><td>4.81</td><td>62.85</td><td>138.24</td><td>62.41</td></tr><tr><td>STream3R [40]</td><td>66.83</td><td>25.47</td><td>1.52</td><td>68.18</td><td>223.07</td><td>77.01</td></tr><tr><td>WinT3R [44]</td><td>59.02</td><td>28.61</td><td>1.10</td><td>42.89</td><td>203.16</td><td>66.96</td></tr><tr><td>StreamVGGT [104]</td><td>55.09</td><td>32.91</td><td>0.57</td><td>57.18</td><td></td><td>36.44</td></tr><tr><td>TTT3R [11]</td><td>29.05</td><td>12.15</td><td>0.54</td><td>6.75</td><td>77.90</td><td>25.28</td></tr><tr><td>Ours (CUT3R)</td><td>4.50</td><td>0.72</td><td>0.66</td><td>5.56</td><td>16.72</td><td>5.63</td></tr><tr><td>Ours (STream3R)</td><td>5.39</td><td>3.26</td><td>3.21</td><td>6.40</td><td>21.32</td><td>7.92</td></tr></table>

<table><tr><td rowspan="2">Methods</td><td>Sintel</td><td>TUM</td><td>ScanNet</td></tr><tr><td>ATE↓</td><td>ATE↓</td><td>ATE↓</td></tr><tr><td>MASt3R-SLAM [57]</td><td>0.233</td><td>0.103</td><td>0.081</td></tr><tr><td>MUSt3R [5]</td><td>0.242</td><td>0.048</td><td>0.046</td></tr><tr><td>CUT3R [82]</td><td>0.210</td><td>0.049</td><td>0.095</td></tr><tr><td>Point3R [89]</td><td>0.375</td><td>0.067</td><td>0.120</td></tr><tr><td>STream3R [40]</td><td>0.214</td><td>0.026</td><td>0.052</td></tr><tr><td>WinT3R [44]</td><td>0.225</td><td>0.074</td><td>0.062</td></tr><tr><td>StreamVGGT [104]</td><td>0.394</td><td>0.057</td><td>0.120</td></tr><tr><td>TTT3R [11]</td><td>0.210</td><td>0.028</td><td>0.064</td></tr><tr><td rowspan="2">Ours (CUT3R) Ours (STream3R)</td><td>0.168</td><td>0.033</td><td>0.092</td></tr><tr><td>0.171</td><td>0.018</td><td>0.049</td></tr></table>

## 4.2 Quantitative Results

Camera Pose Estimation. We evaluate ATE across multiple benchmarks, including KITTI [27] and Virtual KITTI (vKITTI) [4] for outdoor driving scenarios, as well as Sintel [3], TUM-Dynamic [66], and ScanNet [18] for diverse indoor and synthetic environments. Following CUT3R [82] and STream3R [40], we apply Sim(3) alignment to the ground truth before computing ATE. As shown in Tabs. 1 to 3, Scal3R consistently outperforms both ofline and online baselines. On KITTI (Tab. 1), our method achieves an average ATE of 69.7, reducing error by over 60% compared to the strongest online competitor TTT3R (182.2), with particularly pronounced gains on long-range sequences such as Seq. 00 and Seq. 02. On vKITTI (Tab. 2), Scal3R (CUT3R) attains an average ATE of 5.63, surpassing all streaming methods by a large margin and approaching the accuracy of the ofline method π<sup>3</sup>, while remaining fully online. On Sintel, TUM-Dynamic, and ScanNet (Tab. 3), Scal3R generalizes robustly to unseen domains: Scal3R (CUT3R) achieves the best ATE of 0.168 on Sintel, and Scal3R (STream3R) attains state-of-the-art ATE of 0.018 on TUM-Dynamic and 0.049 on ScanNet, demonstrating strong performance across both large-scale outdoor and dense indoor environments without sacrificing online processing.

3D Reconstruction. We evaluate 3D reconstruction quality on the 7-Scenes dataset using 300-frame sequences, reporting Accuracy (Acc.), Completeness (Comp.), and Normal Consistency (NC). As shown in Tab. 4, Scal3R variants consistently enhance the geometric consistency of their frozen backbones. Ours (STream3R) achieves the best performance across all metrics, attaining an NC mean of 0.579 and median of 0.622, surpassing the STream3R backbone by a clear margin. Notably, while competing streaming methods struggle to improve geometric consistency beyond their base backbone, our scale-decoupled formulation yields consistent gains in NC without sacrificing accuracy or completeness.

Table 4: 3D reconstruction on 7-Scenes (300 frames). We report Accuracy (Acc.), Completeness (Comp.), and Normal Consistency (NC), each with mean and median values. Scal3R consistently improves upon both CUT3R and STream3R backbones, achieving the best NC across all sequences.
<table><tr><td rowspan="3">Methods</td><td colspan="6">7-Scenes (300 frames)</td></tr><tr><td colspan="2">Acc. ↓</td><td colspan="2">Comp. ↓</td><td colspan="2">NC↑</td></tr><tr><td>Mean</td><td>Med.</td><td>Mean</td><td>Med.</td><td>Mean</td><td>Med.</td></tr><tr><td>CUT3R [82]</td><td>0.130</td><td>0.090</td><td>0.062</td><td>0.023</td><td>0.544</td><td>0.565</td></tr><tr><td>STream3R [40]</td><td>0.095</td><td>0.040</td><td>0.030</td><td>0.006</td><td>0.560</td><td>0.590</td></tr><tr><td>Ours (CUT3R)</td><td>0.069</td><td>0.034</td><td>0.035</td><td>0.010</td><td>0.566</td><td>0.601</td></tr><tr><td>Ours (STream3R)</td><td>0.052</td><td>0.012</td><td>0.020</td><td>0.004</td><td>0.579</td><td>0.622</td></tr></table>

## 4.3 Qualitative Results

Figs. 6 and 7 visualize 3D reconstruction and trajectory estimation on outdoor long-sequence benchmarks, confirming stable reconstruction and pose prediction across varying spatial extents. In terms of 3D reconstruction (Fig. 6), we compare our method against CUT3R and STream3R on Virtual KITTI Scene 01 (332 m) and Scene 02 (113 m). While CUT3R produces severely distorted point clouds with substantial geometric drift, and STream3R collapses into degenerate reconstructions that deviate considerably from the ground-truth layout, both Ours (CUT3R) and Ours (STream3R) recover scene geometry that closely matches the ground truth, with clean structural boundaries and well-preserved spatial extent. For long-sequence pose estimation (Fig. 7), we visualize estimated trajectories on KITTI Seq. 00 and Seq. 05 against CUT3R, WinT3R, and STream3R. All three baselines sufer from catastrophic trajectory collapse, producing chaotic, self intersecting paths that bear no resemblance to the ground-truth loop structure. In contrast, Ours (CUT3R) faithfully traces the full loop trajectory, maintaining metric accuracy and geometric coherence across hundreds of meters.

## 4.4 Ablation Study

We conduct ablation studies on vKITTI and KITTI to validate the key com ponents of our method, covering model training strategy, inference-time design choices, and loop closure. Results are summarized in Tabs. 5 to 7.

Model Training Strategy. As shown in Tab. 5, removing reference-frame supervision entirely (w/o reference) sharply degrades $\mathrm { R P E } _ { \mathrm { t r a n s } }$ to 3.336, while a single reference reduces ATE to 15.764. Our full multi-reference training achieves the best ATE of 5.632, confirming that denser reference supervision is critical for robust long-sequence pose estimation.

![](images/e8a8f2ab275abde1e3f2d92634125ea933e012065d6e783c39f7252666ead325.jpg)  
Fig. 6: Qualitative 3D reconstruction on Virtual KITTI. Comparison against CUT3R and STream3R on Scene 01 (332 m) and Scene 02 (113 m). While both baselines produce distorted or collapsed point clouds, Scal3R recovers scene geometry closely matching the ground truth across both sequences.

Table 5: Ablation on training strategy (vKITTI). We vary the number of reference frames used for supervision during training. Denser reference supervision consistently lowers ATE, while removing references entirely causes severe RPE degradation.  
Table 6: Ablation on inferencetime components (vKITTI). Both keyframe selection and PGO are essential, and scaling the reference count from 4 to 12 at inference time consistently reduces ATE without retraining.
<table><tr><td>Method</td><td></td><td>ATE ↓ RPEtrans ↓ RPErot ↓</td><td></td></tr><tr><td>Baseline (CUT3R)</td><td>56.390</td><td>1.678</td><td>0.342</td></tr><tr><td>w/o reference</td><td>33.318</td><td>3.336</td><td>0.842</td></tr><tr><td>1 reference</td><td>15.764</td><td>0.217</td><td>0.448</td></tr><tr><td>Ours</td><td>5.632</td><td>0.177</td><td>0.485</td></tr></table>

<table><tr><td>Method</td><td>ATE↓</td><td> $\mathrm { R P E } _ { \mathrm { t r a n s } }$  ↓  $\mathrm { R P E } _ { \mathrm { r o t } } \downarrow$ </td></tr><tr><td>w/o keyframe</td><td>38.258</td><td>0.787 1.388</td></tr><tr><td>w/o PGO</td><td>20.089</td><td>0.203 0.544</td></tr><tr><td>K = 4</td><td>15.748 0.188</td><td>0.470</td></tr><tr><td>K = 8</td><td>7.362 0.180</td><td>0.470</td></tr><tr><td>Full (K = 12)</td><td>5.632 0.177</td><td>0.485</td></tr></table>

Inference-Time Components. Tab. 6 ablates the inference-time pipeline. Removing keyframe selection or PGO each leaves a large gap to the full system (ATE: 38.258 without keyframe selection). Progressively increasing reference count from 4 to 12 then consistently reduces ATE from 15.748 to 5.632.

Runtime Analysis. Fig. 15 breaks down the latency on KITTI (K=12). The frozen forward pass dominates on both backbones (86.3% on CUT3R, 91.2% on STream3R), so keyframe selection, PGO, and loop detection add little: the full pipeline runs at 14.4 and 7.95 FPS, versus 15.9 and 9.1 FPS for the backbones.

![](images/0ee6b437ccfc0cb91a72b24771e336c34be8c14179dcb82a13d0d2d0d1c4eaa9.jpg)  
Fig. 7: Qualitative trajectory comparison on KITTI long sequences. We visualize estimated camera trajectories on Seq. 00 and Seq. 05 against CUT3R, WinT3R, and STream3R. All baselines sufer from catastrophic drift, while Scal3R faithfully recovers the full loop structure with metric accuracy.

Table 7: Ablation on loop closure (KITTI, ATE ↓). Loop closure reduces average ATE by 48%, with the largest gains on loop-heavy sequences (e.g., Seq. 00 and 05).
<table><tr><td rowspan="2">Method</td><td rowspan="2">LC</td><td colspan="8">KITTI (ATE ↓)</td><td rowspan="2">Avg.</td></tr><tr><td>00</td><td>02</td><td>05</td><td>06</td><td>07</td><td>08</td><td>09</td></tr><tr><td rowspan="2">Ours (CUT3R)</td><td></td><td>185.70</td><td>235.28</td><td>108.73</td><td>84.86</td><td>49.83</td><td>241.86</td><td>97.86</td><td>143.45</td></tr><tr><td>√</td><td>45.34</td><td>139.88</td><td>29.37</td><td>47.40</td><td>6.44</td><td>227.02</td><td>29.63</td><td>75.01</td></tr><tr><td rowspan="2">Ours (STream3R)</td><td></td><td>166.37</td><td>239.36</td><td>134.08</td><td>35.95</td><td>60.68</td><td>195.20</td><td>74.18</td><td>129.40</td></tr><tr><td>L</td><td>57.90</td><td>170.75</td><td>39.05</td><td>18.07</td><td>15.19</td><td>173.83</td><td>73.91</td><td>78.39</td></tr></table>

Loop Closure. As shown in Tab. 7 and Fig. 10, loop closure reduces average ATE on KITTI from 143.45 to 75.01 (48% improvement), with particularly large gains on sequences containing large loops (e.g., Seq. 00 and Seq. 05), where globally consistent trajectory correction is most needed.

Robustness Analysis. Fig. 9 presents per-frame ATE curves sorted in ascending order across all KITTI sequences. Methods such as MUSt3R and Point3R sufer catastrophic failures at moderate sequence lengths, while our method maintains the lowest per-frame ATE throughout the entire evaluation range, confirming superior robustness under challenging long-sequence conditions.

## 5 Conclusion

We presented Scal3R, which tackles the instability of global pose regression on long sequences by reformulating camera localization as multi-reference relative pose querying on a frozen backbone. Lightweight tokens (∼1% of parameters) query relative poses that an online pose-graph backend aggregates into a globally consistent trajectory. Trained in 8 hours on a single GPU, Scal3R enables accurate online reconstruction of long video streams.

![](images/5efb7b9c7ff192a51af247f8d5f470d7dbca149caacb12c63c7e390b72793bb1.jpg)

![](images/42f2e8478153124164adda6a3c31e0305a2a74ac18eace874eeb85e254b68d47.jpg)

Fig. 8: Per-frame runtime breakdown on KITTI. On both the CUT3R (left) and STream3R (right) backbones, the frozen forward pass dominates latency; keyframe selection, PGO, and loop detection together add only a small fraction.  
![](images/b3535e700532592b931c89eb84426b29c1b54f174a9c0865f900bcdf193702fb.jpg)  
Fig. 9: Robustness Analysis on KITTI. Scal3R maintains the lowest error across the full evaluation range, while competing methods sufer catastrophic divergence at moderate sequence lengths.

![](images/8c9840ff0be2ca8d3de6b4b64e3f850ccbcde57d9eec70769f4f6e5fc796af9e.jpg)  
Fig. 10: Qualitative efect of loop closure on KITTI 07. Without loop closure, accumulated drift causes the trajectory to deviate from the ground-truth loop structure. With loop closure, Scal3R recovers a globally consistent trajectory.

Limitations. First, performance is bounded by the frozen backbone, degrading when it fails under occlusion or textureless regions. Second, the online backend has its own weaknesses: appearance-based loop closure can miss revisits under extreme viewpoint or illumination change, and keyframe selection and loop detection rely on hand-set thresholds. Improving both remains future work.

Acknowledgements. This work was supported by NVIDIA Taiwan AI Research & Development Center (TRDC). This research was funded by the National Science and Technology Council, Taiwan, under Grants NSTC 112-2222-E-A49-004-MY2, 113-2628-E-A49-023-, 115-2628-E-A49-024-, and 111-2628-E-A49-018-MY4. Yu-Lun Liu acknowledges the Yushan Young Fellow Program by the MOE in Taiwan.

## References

1. Ai, Z., Liu, Z., Lei, Y., Cui, Z., Zou, X., Zhou, J.: Gaprompt: Geometry-aware point cloud prompt for 3d vision model. arXiv preprint arXiv:2505.04119 (2025)

2. Antsfeld, L., Chidlovskii, B., Cabon, Y., Leroy, V., Revaud, J.: S-must3r: Sliding multi-view 3d reconstruction. arXiv preprint arXiv:2602.04517 (2026)

3. Butler, D.J., Wulf, J., Stanley, G.B., Black, M.J.: A naturalistic open source movie for optical flow evaluation. In: A. Fitzgibbon et al. (Eds.) (ed.) European Conf. on Computer Vision (ECCV). pp. 611–625. Part IV, LNCS 7577, Springer-Verlag (Oct 2012)

4. Cabon, Y., Murray, N., Humenberger, M.: Virtual kitti 2. arXiv preprint arXiv:2001.10773 (2020)

5. Cabon, Y., Stofl, L., Antsfeld, L., Csurka, G., Chidlovskii, B., Revaud, J., Leroy, V.: Must3r: Multi-view network for stereo 3d reconstruction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 1050–1060 (2025)

6. Charatan, D., Li, S.L., Tagliasacchi, A., Sitzmann, V.: pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 19457–19467 (2024)

7. Chen, B.Y., Chiu, W.C., Liu, Y.L.: Improving robustness for joint optimization of camera pose and decomposed low-rank tensorial radiance fields. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 38, pp. 990–1000 (2024)

8. Chen, S., Ge, C., Tong, Z., Wang, J., Song, Y., Wang, J., Luo, P.: Adaptformer: Adapting vision transformers for scalable visual recognition. Advances in Neural Information Processing Systems 35, 16664–16678 (2022)

9. Chen, W., Chen, L., Wang, R., Pollefeys, M.: Leap-vo: Long-term efective any point tracking for visual odometry. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 19844–19853 (2024)

10. Chen, X., Chen, Y., Xiu, Y., Geiger, A., Chen, A.: Easi3r: Estimating disentangled motion from dust3r without training. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 9158–9168 (2025)

11. Chen, X., Chen, Y., Xiu, Y., Geiger, A., Chen, A.: Ttt3r: 3d reconstruction as test-time training. arXiv preprint arXiv:2509.26645 (2025)

12. Chen, Y., Chen, X., Xue, Y., Chen, A., Xiu, Y., Pons-Moll, G.: Human3r: Everyone everywhere all at once. arXiv preprint arXiv:2510.06219 (2025)

13. Chen, Y., Xu, H., Zheng, C., Zhuang, B., Pollefeys, M., Geiger, A., Cham, T.J., Cai, J.: Mvsplat: Eficient 3d gaussian splatting from sparse multi-view images. In: European conference on computer vision. pp. 370–386. Springer (2024)

14. Chen, Y., Qiu, Y., Li, R., Agha, A., Omidshafiei, S., Patrikar, J., Scherer, S.: Co-me: Confidence-guided token merging for visual geometric transformers. arXiv preprint arXiv:2511.14751 (2025)

15. Chen, Z., Qin, M., Yuan, T., Liu, Z., Zhao, H.: Long3r: Long sequence streaming 3d reconstruction. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 5273–5284 (2025)

16. Cheng, C., Chen, X., Xie, T., Yin, W., Ren, W., Zhang, Q., Guo, X., Wang, H.: Longstream: Long-sequence streaming autoregressive visual geometry (2026)

17. Cong, Z., Zhao, Q., Jeon, M., Tulsiani, S.: Flow3r: Factored flow prediction for scalable visual geometry learning. arXiv preprint arXiv:2602.20157 (2026)

18. Dai, A., Chang, A.X., Savva, M., Halber, M., Funkhouser, T., Nießner, M.: Scannet: Richly-annotated 3d reconstructions of indoor scenes. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 5828–5839 (2017)

19. Dai, W., Su, W., Kong, D., Ming, Y., Kong, W.: Keyframe-based feed-forward visual odometry. arXiv preprint arXiv:2601.16020 (2026)

20. Deng, K., Ti, Z., Xu, J., Yang, J., Xie, J.: Vggt-long: Chunk it, loop it, align it – pushing vggt’s limits on kilometer-scale long rgb sequences (2025)

21. Ding, T., Xie, Y., Liang, Y., Chatterjee, M., Miraldo, P., Jiang, H.: Laser: Layerwise scale alignment for training-free streaming 4d reconstruction. arXiv preprint arXiv:2512.13680 (2025)

22. Dong, S., Wang, S., Liu, S., Cai, L., Fan, Q., Kannala, J., Yang, Y.: Reloc3r: Large-scale training of relative camera pose regression for generalizable, fast, and accurate visual localization. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 16739–16752 (2025)

23. Du, Z., Danier, D., Lenssen, J.E., Bilen, H.: Moonseg3r: Monocular online zeroshot segment anything in 3d with reconstructive foundation priors. arXiv preprint arXiv:2512.15577 (2025)

24. Duisterhof, B.P., Zust, L., Weinzaepfel, P., Leroy, V., Cabon, Y., Revaud, J.: MASt3r-sfm: a fully-integrated solution for unconstrained structure-from-motion. In: International Conference on 3D Vision 2025 (2025)

25. Elflein, S., Li, R., Agostinho, S., Gojcic, Z., Leal-Taixé, L., Zhou, Q., Osep, A.: VGG-T<sup>3</sup>: Ofline feed-forward 3d reconstruction at scale. arXiv preprint arXiv:2602.23361 (2026)

26. Elflein, S., Zhou, Q., Leal-Taixé, L.: Light3r-sfm: Towards feed-forward structurefrom-motion. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 16774–16784 (2025)

27. Geiger, A., Lenz, P., Urtasun, R.: Are we ready for autonomous driving? the kitti vision benchmark suite. In: Conference on Computer Vision and Pattern Recognition (CVPR) (2012)

28. Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., De Laroussilhe, Q., Gesmundo, A., Attariyan, M., Gelly, S.: Parameter-eficient transfer learning for nlp. In: International conference on machine learning. pp. 2790–2799. PMLR (2019)

29. Hsiao, T.F., Ruan, B.K., Liu, Y.L., Shuai, H.H.: Vecset-edit: Unleashing pretrained lrm for mesh editing from single image. In: Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers. pp. 1–12 (2026)

30. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: LoRA: Low-rank adaptation of large language models. In: International Conference on Learning Representations (2022)

31. Huang, J., Zhou, Q., Rabeti, H., Korovko, A., Ling, H., Ren, X., Shen, T., Gao, J., Slepichev, D., Lin, C.H., et al.: Vipe: Video pose engine for 3d geometric perception. arXiv preprint arXiv:2508.10934 (2025)

32. Huang, L., Mao, J., Yi, J., Tao, Z., Wang, Y.: Cvpt: Cross visual prompt tuning. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 848–858 (2025)

33. Izquierdo, S., Civera, J.: Optimal transport aggregation for visual place recognition. In: Proceedings of the ieee/cvf conference on computer vision and pattern recognition. pp. 17658–17668 (2024)

34. Jang, W., Weinzaepfel, P., Leroy, V., Agapito, L., Revaud, J.: Pow3r: Empowering unconstrained 3d reconstruction with camera and scene priors. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 1071–1081 (2025)

35. Jena, S., Ouasfi, A., Younes, M., Boukhayma, A.: Sparfels: Fast reconstruction from sparse unposed imagery. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 27476–27487 (2025)

36. Jia, M., Tang, L., Chen, B.C., Cardie, C., Belongie, S., Hariharan, B., Lim, S.N.: Visual prompt tuning. In: European conference on computer vision. pp. 709–727. Springer (2022)

37. Kaess, M., Johannsson, H., Roberts, R., Ila, V., Leonard, J.J., Dellaert, F.: isam2: Incremental smoothing and mapping using the bayes tree. The International Journal of Robotics Research 31(2), 216–235 (2012)

38. Keetha, N., Müller, N., Schönberger, J., Porzi, L., Zhang, Y., Fischer, T., Knapitsch, A., Zauss, D., Weber, E., Antunes, N., et al.: Mapanything: Universal feed-forward metric 3d reconstruction. arXiv preprint arXiv:2509.13414 (2025)

39. Khafizov, R., Komarichev, A., Rakhimov, R., Wonka, P., Burnaev, E.: G-cut3r: Guided 3d reconstruction with camera and depth prior integration. arXiv preprint arXiv:2508.11379 (2025)

40. Lan, Y., Luo, Y., Hong, F., Zhou, S., Chen, H., Lyu, Z., Yang, S., Dai, B., Loy, C.C., Pan, X.: Stream3r: Scalable sequential 3d reconstruction with causal transformer. arXiv preprint arXiv:2508.10893 (2025)

41. Leroy, V., Cabon, Y., Revaud, J.: Grounding image matching in 3d with mast3r (2024)

42. Lester, B., Al-Rfou, R., Constant, N.: The power of scale for parameter-eficient prompt tuning. In: Proceedings of the 2021 conference on empirical methods in natural language processing. pp. 3045–3059 (2021)

43. Li, X.L., Liang, P.: Prefix-tuning: Optimizing continuous prompts for generation. In: Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). pp. 4582–4597 (2021)

44. Li, Z., Zhou, J., Wang, Y., Guo, H., Chang, W., Zhou, Y., Zhu, H., Chen, J., Shen, C., He, T.: Wint3r: Window-based streaming reconstruction with camera token pool. arXiv preprint arXiv:2509.05296 (2025)

45. Lin, C.Y., Sun, C., Yang, F.E., Chen, M.H., Lin, Y.Y., Liu, Y.L.: Longsplat: Robust unposed 3d gaussian splatting for casual long videos. In: ICCV (2025)

46. Lin, C.Y., Wu, C.H., Yeh, C.H., Yen, S.H., Sun, C., Liu, Y.L.: Frugalnerf: Fast convergence for few-shot novel view synthesis without learned priors. In: CVPR (2025)

47. Lin, H., Chen, S., Liew, J., Chen, D.Y., Li, Z., Shi, G., Feng, J., Kang, B.: Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647 (2025)

48. Liu, S., Li, W., Qiao, P., Dou, Y.: Regist3r: Incremental registration with stereo foundation model. In: Proceedings of the 33rd ACM International Conference on Multimedia. pp. 4484–4493 (2025)

49. Liu, Y.L., Gao, C., Meuleman, A., Tseng, H.Y., Saraf, A., Kim, C., Chuang, Y.Y., Kopf, J., Huang, J.B.: Robust dynamic radiance fields. In: 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 13–23. IEEE (2023)

50. Liu, Y., Dong, S., Wang, S., Yin, Y., Yang, Y., Fan, Q., Chen, B.: Slam3r: Realtime dense scene reconstruction from monocular rgb videos. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 16651–16662 (2025)

51. Lu, Z., Yang, H., Xu, D., Li, B., Ivanovic, B., Pavone, M., Wang, Y.: Lora3d: Low-rank self-calibration of 3d geometric foundation models. arXiv preprint arXiv:2412.07746 (2024)

52. Maggio, D., Carlone, L.: Vggt-slam 2.0: Real time dense feed-forward scene reconstruction. arXiv preprint arXiv:2601.19887 (2026)

53. Maggio, D., Lim, H., Carlone, L.: VGGT-SLAM: Dense RGB SLAM optimized on the SL(4) manifold. arXiv preprint arXiv:2505.12549 (2025)

54. Mahdi, S., Ayar, F., Javanmardi, E., Tsukada, M., Javanmardi, M.: Evict3r: Training-free token eviction for memory-bounded streaming visual geometry transformers. arXiv preprint arXiv:2509.17650 (2025)

55. Meuleman, A., Liu, Y.L., Gao, C., Huang, J.B., Kim, C., Kim, M.H., Kopf, J.: Progressively optimized local radiance fields for robust view synthesis. In: 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 16539–16548. IEEE (2023)

56. Mur-Artal, R., Tardós, J.D.: Orb-slam2: An open-source slam system for monocular, stereo, and rgb-d cameras. IEEE transactions on robotics 33(5), 1255–1262 (2017)

57. Murai, R., Dexheimer, E., Davison, A.J.: Mast3r-slam: Real-time dense slam with 3d reconstruction priors. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 16695–16705 (2025)

58. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023)

59. Pan, L., Baráth, D., Pollefeys, M., Schönberger, J.L.: Global structure-from-motion revisited. In: European Conference on Computer Vision. pp. 58–77. Springer (2024)

60. Ren, L., Chen, C., Wang, L., Hua, K.: Da-vpt: Semantic-guided visual prompt tuning for vision transformers. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 4353–4363 (2025)

61. Schönberger, J.L., Frahm, J.M.: Structure-from-motion revisited. In: Conference on Computer Vision and Pattern Recognition (CVPR) (2016)

62. Schönberger, J.L., Zheng, E., Pollefeys, M., Frahm, J.M.: Pixelwise view selection for unstructured multi-view stereo. In: European Conference on Computer Vision (ECCV) (2016)

63. Shen, G., Deng, T., Wang, Y., Chen, Y., Shen, Y., Liu, J., Wang, J.: Grs-slam3r: Real-time dense slam with gated recurrent state. arXiv preprint arXiv:2509.23737 (2025)

64. Shen, Y., Zhang, Z., Qu, Y., Zheng, X., Ji, J., Zhang, S., Cao, L.: Fastvggt: Trainingfree acceleration of visual geometry transformer. arXiv preprint arXiv:2509.02560 (2025)

65. Smart, B., Zheng, C., Laina, I., Prisacariu, V.A.: Splatt3r: Zero-shot gaussian splatting from uncalibrated image pairs. arXiv preprint arXiv:2408.13912 (2024)

66. Sturm, J., Engelhard, N., Endres, F., Burgard, W., Cremers, D.: A benchmark for the evaluation of rgb-d slam systems. In: 2012 IEEE/RSJ international conference on intelligent robots and systems. pp. 573–580. IEEE (2012)

67. Su, C.H., Hu, C.Y., Tsai, S.R., Lee, J.Y., Lin, C.Y., Liu, Y.L.: Boostmvsnerfs: Boosting mvs-based nerfs to generalizable view synthesis in large-scale scenes. In: ACM SIGGRAPH 2024 Conference Papers. pp. 1–12 (2024)

68. Sun, J., Xie, Y., Chen, L., Zhou, X., Bao, H.: Neuralrecon: Real-time coherent 3d reconstruction from monocular video. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 15598–15607 (2021)

69. Sun, X., Zhu, Z., Lou, Z., Yang, B., Tang, J., Zhang, L., Wang, H., Zhang, J.: Avggt: Rethinking global attention for accelerating vggt. arXiv preprint arXiv:2512.02541 (2025)

70. Sun, X., Jiang, H., Liu, L., Nam, S., Kang, G., Wang, X., Sui, W., Su, Z., Liu, W., Wang, X., et al.: Uni3r: Unified 3d reconstruction and semantic understanding via generalizable gaussian splatting from unposed multi-view images. arXiv preprint arXiv:2508.03643 (2025)

71. Sun, Y.C., Sun, C., Lin, C.Y., Yang, F.E., Chen, M.H., Lin, Y.Y., Liu, Y.L.: 3am: Segment anything with geometric consistency in videos. arXiv preprint arXiv:2601.08831 (2026)

72. Taher, M., Alzugaray, I., Mazur, K., Kong, X., Davison, A.J.: Kv-tracker: Real-time pose tracking with transformers. arXiv preprint arXiv:2512.22581 (2025)

73. Tang, Y., Zhang, R., Guo, Z., Ma, X., Zhao, B., Wang, Z., Wang, D., Li, X.: Pointpeft: Parameter-eficient fine-tuning for 3d pre-trained models. In: Proceedings of the AAAI conference on artificial intelligence. vol. 38, pp. 5171–5179 (2024)

74. Tang, Z., Fan, Y., Wang, D., Xu, H., Ranjan, R., Schwing, A., Yan, Z.: Mv-dust3r+: Single-stage scene reconstruction from sparse views in 2 seconds. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 5283–5293 (2025)

75. Teed, Z., Deng, J.: Droid-slam: Deep visual slam for monocular, stereo, and rgb-d cameras. Advances in neural information processing systems 34, 16558–16569 (2021)

76. Tu, C.H., Mai, Z., Chao, W.L.: Visual query tuning: Towards efective usage of intermediate representations for parameter and memory eficient transfer learning. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 7725–7735 (2023)

77. Wang, C.S.B., Schmidt, C., Piekenbrinck, J., Leibe, B.: Faster vggt with blocksparse global attention. arXiv preprint arXiv:2509.07120 (2025)

78. Wang, H., Agapito, L.: 3d reconstruction with spatial memory. arXiv preprint arXiv:2408.16061 (2024)

79. Wang, H., Agapito, L.: Amb3r: Accurate feed-forward metric-scale 3d reconstruction with backend. arXiv preprint arXiv:2511.20343 (2025)

80. Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht, C., Novotny, D.: Vggt: Visual geometry grounded transformer. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 5294–5306 (2025)

81. Wang, J., Karaev, N., Rupprecht, C., Novotny, D.: Vggsfm: Visual geometry grounded deep structure from motion. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 21686–21697 (2024)

82. Wang, Q., Zhang, Y., Holynski, A., Efros, A.A., Kanazawa, A.: Continuous 3d perception model with persistent state. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 10510–10522 (2025)

83. Wang, S., Leroy, V., Cabon, Y., Chidlovskii, B., Revaud, J.: Dust3r: Geometric 3d vision made easy. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 20697–20709 (June 2024)

84. Wang, S., Liu, X., Kong, L., Xu, J., Hu, C., Fang, G., Li, W., Zhu, J., Wang, X.: Pointlora: Low-rank adaptation with token selection for point cloud learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6605–6615 (2025)

85. Wang, W., Zhu, D., Wang, X., Hu, Y., Qiu, Y., Wang, C., Hu, Y., Kapoor, A., Scherer, S.: Tartanair: A dataset to push the limits of visual slam. In: 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). pp. 4909–4916. IEEE (2020)

86. Wang, Y., Zhou, J., Zhu, H., Chang, W., Zhou, Y., Li, Z., Chen, J., Pang, J., Shen, C., He, T.: π<sup>3</sup>: Permutation-equivariant visual geometry learning. arXiv preprint arXiv:2507.13347 (2025)

87. Wang, Z., Cao, A., Wang, L.J., Park, J.J.: Moe3d: A mixture-of-experts module for 3d reconstruction. arXiv preprint arXiv:2601.05208 (2026)

88. Wang, Z., Xu, D.: Flashvggt: Eficient and scalable visual geometry transformers with compressed descriptor attention. arXiv preprint arXiv:2512.01540 (2025)

89. Wu, Y., Zheng, W., Zhou, J., Lu, J.: Point3r: Streaming 3d reconstruction with explicit spatial pointer memory. arXiv preprint arXiv:2507.02863 (2025)

90. Xin, Y., Yang, J., Luo, S., Du, Y., Qin, Q., Cen, K., He, Y., Zhang, Z., Fu, B., Yang, X., et al.: Parameter-eficient fine-tuning for pre-trained vision models: A survey and benchmark. arXiv preprint arXiv:2402.02242 (2024)

91. Xiong, Z., Zhang, C., Xu, Q., Tao, W.: Vggt-motion: Motion-aware calibration-free monocular slam for long-range consistency. arXiv preprint arXiv:2602.05508 (2026)

92. Yang, J., Sax, A., Liang, K.J., Henaf, M., Tang, H., Cao, A., Chai, J., Meier, F., Feiszli, M.: Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 21924–21935 (2025)

93. Ye, B., Liu, S., Xu, H., Li, X., Pollefeys, M., Yang, M.H., Peng, S.: No pose, no problem: Surprisingly simple 3d gaussian splats from sparse unposed images. arXiv preprint arXiv:2410.24207 (2024)

94. Yeshwanth, C., Liu, Y.C., Nießner, M., Dai, A.: Scannet++: A high-fidelity dataset of 3d indoor scenes. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 12–22 (2023)

95. Yuan, S., Yang, Y., Yang, X., Zhang, X., Zhao, Z., Zhang, L., Zhang, Z.: Infinitevggt: Visual geometry grounded transformer for endless streams. arXiv preprint arXiv:2601.02281 (2026)

96. Yuan, Y., Chen, Z., Li, K., Wang, W., Zhao, H.: Slam-former: Putting slam into one transformer. arXiv preprint arXiv:2509.16909 (2025)

97. Yugay, V., Nguyen, D.K., Gevers, T., Snoek, C.G.M., Oswald, M.R.: Visual odometry with transformers (2025)

98. Zhang, C., Le Moing, G., Koppula, S., Rocco, I., Momeni, L., Xie, J., Sun, S., Sukthankar, R., Barral, J.K., Hadsell, R., Ghahramani, Z., Zisserman, A., Zhang, J., Sajjadi, M.S.M.: Eficiently reconstructing dynamic scenes one d4rt at a time. arXiv preprint arXiv:2512.08924 (2025)

99. Zhang, G., Qian, S., Wang, X., Cremers, D.: Vista-slam: Visual slam with symmetric two-view association. arXiv preprint arXiv:2509.01584 (2025)

100. Zhang, J., Herrmann, C., Hur, J., Jampani, V., Darrell, T., Cole, F., Sun, D., Yang, M.H.: Monst3r: A simple approach for estimating geometry in the presence of motion. arXiv preprint arXiv:2410.03825 (2024)

101. Zheng, Z., Xiang, X., Zhang, J.: Ttsa3r: Training-free temporal-spatial adaptive persistent state for streaming 3d reconstruction. arXiv preprint arXiv:2601.22615 (2026)

102. Zhou, X., Liang, D., Xu, W., Zhu, X., Xu, Y., Zou, Z., Bai, X.: Dynamic adapter meets prompt tuning: Parameter-eficient transfer learning for point cloud analysis. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 14707–14717 (2024)

103. Zhou, Y., Barnes, C., Lu, J., Yang, J., Li, H.: On the continuity of rotation representations in neural networks. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 5745–5753 (2019)

104. Zhuo, D., Zheng, W., Guo, J., Wu, Y., Zhou, J., Lu, J.: Streaming 4d visual geometry transformer. arXiv preprint arXiv:2507.11539 (2025)

## Overview

This supplementary material provides additional details and experiments that complement the main paper. Sec. A describes implementation details, including the inference pipeline algorithm, evaluation protocol, and PGO parameters. Sec. B presents additional qualitative results and ablation visualizations on vKITTI, KITTI, and TUM-Dynamic. Sec. C reports runtime and memory profiling, including a per-component latency breakdown and scalability analysis with respect to the reference count K. Sec. D provides additional experiments: metric-scale pose estimation (Sec. D.1), comparison with classic SLAM systems (Sec. D.2), a comparison with the concurrent LongStream (Sec. D.3), an ablation on asymmetric vs. symmetric attention injection (Sec. D.4), compatibility with zero-shot test-time training (Sec. D.5), and robustness analysis on dynamic scenes (Sec. D.6).

## A Implementation Details

We implement Scal3R using PyTorch, building upon the frozen CUT3R and STream3R backbones without modifying their pretrained weights. Pose-graph optimization is performed using GTSAM’s iSAM2 incremental solver, with a gap-dependent noise model $\sigma = \sigma _ { \mathrm { b a s e } } \cdot \varDelta ^ { 0 . 5 }$ and Huber robust kernel for outlier rejection. Loop closure retrieval employs a pretrained DINOv2-B backbone with a SALAD aggregation layer, indexed online via FAISS over keyframe descriptors. All models are trained on TartanAir with the AdamW optimizer at a learning rate of $1 \times 1 0 ^ { - 4 }$ , batch size 8, for 40 epochs. Training converges in approximately 8 hours on a single NVIDIA A100 GPU. All inference experiments are evaluated on NVIDIA A100 GPUs.

## A.1 Inference Pipeline

Algorithm 1 summarizes the complete per-frame inference pipeline of Scal3R. At each timestep, reference frames are first selected from the pose token bufer, followed by loop closure detection over archived keyframe descriptors via DINOv2- SALAD indexed with FAISS. If a loop candidate is detected, its archived camera token $\mathbf { c } _ { r _ { \mathrm { l o o p } } }$ is re-injected into the pose token bufer as an additional reference slot before model inference, requiring no architectural modification. The frozen backbone then predicts both the local pointmap $X _ { t }$ and multi-reference relative poses $\{ \hat { \mathbf { T } } _ { t  r _ { k } } \}$ via asymmetric attention injection. These pairwise constraints, together with any loop closure edge, are registered into the factor graph and incrementally optimized via iSAM2, with corrected poses immediately written back to the pose token bufer to benefit subsequent frames. Finally, keyframe selection determines whether the current frame updates the bufer and the keyframe archive, and bufer pruning maintains a bounded memory footprint throughout the stream.

Algorithm 1 Scal3R Online Inference Pipeline   
Require: Image stream $\{ I _ { 1 } , I _ { 2 } , \ldots , I _ { T } \} ;$ frozen backbone; thresholds   
$\tau _ { \mathrm { o v e r l a p } , \tau _ { \mathrm { c o n f } } , \tau _ { \mathrm { s i m } } , \tau _ { \mathrm { g a p } } , w _ { \mathrm { n m s } } }$   
Ensure: Globally consistent camera poses $\{ T _ { 1 } , \dots , T _ { T } \} \subset S E ( 3 )$   
1: Initialize pose token bufer $B  \varnothing ,$ keyframe archive $A  \emptyset$ , factor graph ${ \mathcal { G } } \gets \emptyset$   
2: for each frame $I _ { t }$ do   
3: $/ /$ Reference Selection   
4: Retrieve reference indices $\{ r _ { 1 } , \ldots , r _ { K } \}$ from B   
5: $/ /$ Loop Closure Detection   
6: $d _ { t } \gets \mathrm { D I N O v 2 – S A L A D } ( I _ { t } ) ;$ ; query FAISS over $\mathcal { A }$   
7: if $\mathrm { l } r _ { \mathrm { l o o p } } \mathrm { : }$ similarity $> \tau _ { \mathrm { s i m } } .$ , gap $> \tau _ { \mathrm { g a p } } .$ , NMS window $w _ { \mathrm { n m s } }$ then   
8: Inject archived camera token $\mathbf { c } _ { r _ { \mathrm { l o o p } } }$ from A into $\boldsymbol { B }$ as additional reference   
slot   
9: end if   
10: $/ /$ Model Inference   
11: $F _ { t } \gets$ Encoder $\left( I _ { t } \right)$ \triangleright frozen ViT   
12: for $k = 1 , \ldots , K$ do   
13: $\tilde { \mathbf { q } } _ { k } \gets \mathbf { q } + \mathrm { M L P } ( \mathbf { c } _ { r _ { k } } )$ \triangleright pose token assembly, Eq. (1)   
14: end for   
15: $\{ X _ { t } , \tilde { \mathbf { q } } _ { k } ^ { L } \} $ Decoder $\left( { F } _ { t } , \ \left\{ \tilde { \mathbf { q } } _ { k } \right\} \right)$ , state) \triangleright asymmetric attention, Eqs. (2)–(3)   
16: $\hat { \mathbf { T } } _ { t  r _ { k } } \gets$ Head $( \tilde { \mathbf { q } } _ { k } ^ { L } ) \in S E ( 3 )$ for all $k$ \triangleright Eq. (4)   
17: $/ /$ Pose-Graph Optimization   
18: $T _ { t } \gets T _ { r _ { 1 } } \cdot \hat { \mathbf { T } } _ { t  r _ { 1 } } ^ { - 1 }$ \triangleright chain initialization   
19: for $k = 1 , \ldots , \bar { K }$ do   
20: Add between-factor $( \hat { \mathbf { T } } _ { t  r _ { k } } , \Sigma _ { t , k } )$ to $\mathcal { G }$ \triangleright Eq. (6); $\sigma { = } \sigma _ { \mathrm { b a s e } } { \cdot } \varDelta ^ { 0 . 5 }$   
21: end for   
22: if loop edge exists then   
23: Add loop edge $( \hat { \mathbf { T } } _ { t  r _ { \mathrm { l o o p } } } , \Sigma _ { \mathrm { l o o p } } )$ to G with tight Gaussian noise   
24: end if   
25: {T<sub>i</sub>} ← iSAM2.optimize(G); write back optimized poses to B   
26: // Keyframe Selection & Bufer Update   
27: Compute overlap score via KD-tree on $X _ { t }$   
28: if overlap score $< \tau _ { \mathrm { o v e r l a p } }$ and median conf $> \tau _ { \mathrm { c o n f } }$ then   
29: Update KV cache; append $\mathbf { c } _ { t }$ to $\begin{array} { r } { B ; { } } \end{array}$ archive $\mathbf { c } _ { t }$ in $\mathcal { A }$   
30: else   
31: Discard KV state snapshot; do not update B   
32: end if   
33: Retain most recent $K _ { \mathrm { k f } }$ keyframe and $K _ { \mathrm { n k f } }$ non-keyframe entries in B   
34: end for   
35: return {T<sub>t</sub>} from final iSAM2 marginalization

## A.2 Evaluation Protocol and Scale Handling

Since Scal3R freezes the pretrained backbone and supervises only relative pose in a scale-normalized space, the predicted trajectory operates in the backbone’s internal scale rather than metric scale. During training, both the predicted and ground-truth point clouds are independently normalized by their respective scale factors, and a robust scale alignment is applied before loss computation to eliminate monocular scale ambiguity. At inference, relative poses are chained and refined via iSAM2 entirely within this model-internal scale space.

For evaluation, we follow the protocol of CUT3R [82] and STream3R [40], applying Sim(3) alignment via the Umeyama method to align the predicted trajectory to the ground truth before computing ATE. This alignment recovers the global scale, rotation, and translation, ensuring fair comparison across all methods regardless of their internal scale convention.

Since the CUT3R backbone is trained with metric-scale supervision, it retains absolute scale information in its camera tokens. We therefore additionally evaluate Scal3R in a metric-scale setting, where the Sim(3) scale factor is fixed to 1 (SE(3) alignment), and report these results in Sec. D.1.

## A.3 PGO Parameters

Keyframe Selection. Frames are designated as keyframes when their depthnormalized 3D overlap score falls below $\tau _ { \mathrm { o v e r l a p } } = 0 .$ 1 and median depth confidence exceeds $\tau _ { \mathrm { c o n f } } = 1 . 2$ . The first $N _ { \mathrm { i n i t } } = 5$ frames are unconditionally treated as keyframes to initialize the system. The active pose token bufer retains the most recent $K _ { \mathrm { k f } } = 4$ to 12 keyframes depending on the dataset, with additional non-keyframe slots enabled for outdoor driving sequences. For outdoor driving sequences (KITTI, vKITTI), the frozen decoder’s streaming state is reset every $N _ { \mathrm { r e s e t } } = 1 0$ keyframes to prevent memory overflow and feature degradation; for all other datasets the state is never reset.

PGO Noise Model. We set $\sigma _ { \mathrm { b a s e } } = 0 . 5$ for both rotation and translation, with gap-dependent scaling $\sigma = \sigma _ { \mathrm { b a s e } } \cdot \varDelta ^ { 0 . 5 }$ as described in the main paper. Sequential edges are weighted by a Huber robust kernel $( k = 1 . 3 4 5 )$ to downweight outlier constraints, while loop closure edges omit the robust kernel to enforce tight trajectory correction. Pose-graph optimization is performed via iSAM2 with a relinearization threshold of 0.1.

Loop Closure Detection. Candidates are filtered by cosine similarity threshold $\tau _ { \mathrm { s i m } } = 0 . 8 5$ , minimum temporal gap $\tau _ { \mathrm { { g a p } } } = 2 0 0$ frames, and non-maximum suppression within a window of $w _ { \mathrm { n m s } } = 5 0$ frames. At most one loop edge is injected per keyframe.

![](images/427a1d42343fc5370fa0f64a9919ebb035630ba805ecd87f6021d42d3248cc28.jpg)  
Fig. 11: More qualitative comparison on vKITTI. Scal3R produces globally consistent reconstructions with minimal trajectory drift, while baselines exhibit elongated or distorted point clouds.

![](images/9cbedace66b978e09ae0accaaac102d97cfa5360029a63dba381ad1af0cd4c9e.jpg)  
Fig. 12: More qualitative comparison on TUM-Dynamic. Scal3R produces cleaner reconstructions with fewer ghosting artifacts from dynamic pedestrians on both backbones.

## B Additional Visualizations

## B.1 More Qualitative Results

Fig. 11 presents additional qualitative comparisons on vKITTI sequences 18 and 20. CUT3R and STream3R both exhibit noticeable trajectory drift on these long sequences, producing elongated and skewed reconstructions. TTT3R maintains reasonable local geometry but accumulates global error, while WinT3R shows severe structural distortion, particularly on seq. 20. In contrast, Scal3R (CUT3R backbone) recovers compact, globally consistent point clouds with wellaligned trajectory shapes on both sequences, benefiting from multi-reference relative pose querying and PGO that jointly suppress cumulative drift over hundreds of frames. Fig. 12 further compares reconstructions on TUM-Dynamic, where moving pedestrians heavily occlude the static scene. Both CUT3R and STream3R baselines produce fragmented point clouds with ghosting artifacts from the dynamic persons, and their trajectories scatter erratically. Our variants on both backbones yield cleaner reconstructions with sharper room geometry and more coherent camera trajectories, consistent with the implicit dynamic-object down-weighting observed in the attention maps (Sec. D.6).

![](images/9cc44e1d7a445ec46e30c658e87883650a885ac540804c7b8bb8cde248af5492.jpg)  
Fig. 13: Ablation visualizations on vKITTI. Removing keyframe selection or PGO each leads to distinct trajectory degradation.

![](images/9ab7ecce7446603e3dc216d5759d570161c7ec93e73af19437d2b8d241add1b2.jpg)  
Fig. 14: Ablation visualizations on KITTI. Disabling loop closure leaves visible gaps at revisited regions; the full system produces a globally consistent reconstruction.

## B.2 Ablation Visualizations

Figs. 13 and 14 visualize the reconstructed point clouds and estimated trajectories under each ablation setting. Without keyframe selection, the reference pool lacks geometric diversity, causing severe trajectory drift and distorted global structure on both vKITTI and KITTI. Removing PGO preserves local smoothness but allows cumulative error to bend the overall trajectory, most visible in the curved segments of vKITTI. On KITTI (Fig. 14), disabling loop closure leaves a noticeable gap where revisited regions should align, whereas the full system closes the loop and produces a globally consistent reconstruction.

## C Runtime and Memory Profiling

## C.1 Runtime Analysis

Figs. 15 and 16 break down the per-frame latency of Scal3R on KITTI with K=12 for both backbones. On the CUT3R backbone (Fig. 15), the model forward pass dominates in both configurations, consuming 89.8% (57.8 ms) without loop closure and 86.3% (59.9 ms) with it. Keyframe selection and PGO together add fewer than 7 ms, while the loop detection module introduces only 2.4 ms of additional overhead. Enabling loop closure reduces throughput from 15.5 to 14.4 FPS, a modest 7% drop that is justified by the substantial ATE improvements on revisited sequences (cf. main paper). Compared to the CUT3R baseline (15.9 FPS), the full pipeline retains over 90% of its throughput. On the STream3R backbone (Fig. 16), the latency profile is similar: the forward pass remains the dominant cost, and all system-level components together add less than 10% overhead. Across both backbones, the results confirm that keyframe selection, PGO, and loop closure are lightweight relative to the frozen backbone inference, preserving real-time throughput regardless of the backbone choice.

![](images/88f1d30c4687cffc4f8669d8300c13df0e21299120ab39c255d32e3b3b067349.jpg)

![](images/dbabafbf592cf3e33ec61108d994b0015c6bc5a6779a42ca788286f2851848d5.jpg)  
Fig. 15: Scal3R (CUT3R) runtime breakdown on KITTI. The model forward pass dominates latency in both configurations; all system-level components together add less than 10% overhead.

![](images/d0985ca1415ef88ea3d11b93f9e1e86325ef082ac2848d77d98bce31c29b995a.jpg)

![](images/5b3dac5cd3eba96ba9fec769a8ed503e544ff687bedcb2e6cf0511ae22b5a836.jpg)  
Fig. 16: Scal3R (STream3R) runtime breakdown on KITTI. The latency profile closely mirrors the CUT3R variant; system-level components remain lightweight across both backbones.

## C.2 Scalability with Reference Count K

Tab. 8 reports ATE, throughput, and memory as a function of the reference count K on vKITTI. Increasing K from 4 to 12 reduces ATE from 15.75 to 5.63, as additional references provide broader geometric coverage for relative pose querying.

Table 8: Scalability with reference count K on vKITTI. ATE is minimized at K=12; latency grows modestly with K.
<table><tr><td>K</td><td>ATE↓</td><td>FPS↑</td><td>ms/frame</td><td>Mem (GB)</td></tr><tr><td>4</td><td>15.75</td><td>15.84</td><td>63.23</td><td>2.91</td></tr><tr><td>7</td><td>7.02</td><td>15.57</td><td>64.23</td><td>2.90</td></tr><tr><td>12</td><td>5.63</td><td>15.10</td><td>66.23</td><td>2.94</td></tr><tr><td>16</td><td>8.27</td><td>14.95</td><td>66.93</td><td>3.01</td></tr><tr><td>24</td><td>12.16</td><td>14.29</td><td>70.00</td><td>3.24</td></tr></table>

Table 9: Ablation on scale alignment (ATE ↓). SA = scale alignment via Sim(3). Without SA, outdoor ATE degrades substantially for Scal3R, while CUT3R’s dominant error remains trajectory drift. We report only the CUT3R backbone variant, as it is trained with metric-scale supervision and thus retains meaningful absolute scale information; the STream3R backbone lacks metric-scale training and is therefore not applicable to this analysis.
<table><tr><td rowspan="2">Method SA</td><td rowspan="2"></td><td colspan="3">Indoor</td><td colspan="2">Outdoor</td></tr><tr><td>Sintel</td><td>TUM</td><td>ScanNet</td><td>vKITTI</td><td>KITTI</td></tr><tr><td rowspan="2">CUT3R</td><td>√</td><td>0.210</td><td>0.049</td><td>0.095</td><td>56.39</td><td>207.20</td></tr><tr><td>×</td><td>0.424</td><td>0.070</td><td>0.123</td><td>68.61</td><td>216.30</td></tr><tr><td rowspan="2">Scal3R</td><td>√</td><td>0.168</td><td>0.033</td><td>0.092</td><td>5.63</td><td>69.73</td></tr><tr><td>×</td><td>0.381</td><td>0.046</td><td>0.121</td><td>9.25</td><td>88.66</td></tr></table>

Beyond K=12, performance degrades (K=24 yields 12.16 ATE), likely because the pose query tokens are trained with n=3 references and generalize best within a moderate range. Meanwhile, latency grows only modestly (63.23 ms to 70.00 ms per frame) and peak memory remains below 3.3 GB across all settings, confirming that asymmetric attention injection scales eficiently with K. We therefore adopt K=12 as the default at inference, balancing accuracy and throughput at approximately 15 FPS.

## D Additional Experiments

## D.1 Metric-Scale Pose Estimation

Tab. 9 ablates the efect of Sim(3) scale alignment on ATE across indoor and outdoor benchmarks. Without scale alignment, both CUT3R and Scal3R show moderate degradation on indoor scenes (e.g. TUM rises from 0.033 to 0.046 for ours), indicating that the predicted poses already carry a reasonable metric scale in small environments. On outdoor datasets the gap widens substantially: Scal3R degrades from 5.63 to 9.25 on vKITTI and from 69.73 to 88.66 on KITTI, reflecting the dificulty of maintaining consistent scale over kilometer-scale trajectories. Notably, removing SA changes CUT3R far less on vKITTI (56.39 to 68.61) than its already large drift, suggesting its dominant error source is trajectory drift rather than scale ambiguity. Scal3R consistently outperforms CUT3R in both settings, confirming that multi-reference relative pose querying improves not only relative pose accuracy but also global scale consistency.

Table 10: Camera pose estimation results (ATE ↓) on KITTI with classic SLAM systems. Scal3R is competitive with calibrated SLAM and significantly outperforms calibration-free baselines without requiring camera intrinsics.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Calib.</td><td rowspan="2"></td><td colspan="10">KITTI Sequence</td><td rowspan="2">Avg. 10</td></tr><tr><td>00</td><td>01</td><td>02</td><td>03</td><td>04</td><td>05</td><td>06</td><td>07</td><td>08</td><td>09</td></tr><tr><td>ORB-SLAM2 [56]</td><td>√</td><td></td><td>6.03 508.34</td><td>14.76</td><td>1.02</td><td>1.57</td><td>4.04</td><td>11.16</td><td>2.19</td><td>38.85</td><td>8.39</td><td>6.63</td><td>54.82</td></tr><tr><td>DROID-SLAM [75]</td><td>√</td><td>170.60</td><td></td><td>91.03 255.22</td><td>1.25</td><td>0.35</td><td>59.79</td><td>32.07 14.03 138.76</td><td></td><td></td><td>55.83</td><td>13.45</td><td>75.67</td></tr><tr><td>DROID-SLAM [75]</td><td>x</td><td>190.93</td><td></td><td>89.50 239.93</td><td>9.27</td><td>0.37</td><td>133.13</td><td>131.22 70.33</td><td></td><td>3144.53</td><td>187.18165.44123.80</td><td></td><td></td></tr><tr><td>MASt3R-SLAM [57]</td><td>√</td><td></td><td>188.46562.85 282.38</td><td></td><td>121.67</td><td>92.61</td><td>99.84</td><td>57.22 76.97 263.63 184.09 179.09 191.71</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MASt3R-SLAM [57]</td><td>x</td><td></td><td>188.46562.85 282.38</td><td></td><td>121.67</td><td>92.61</td><td></td><td>57.22</td><td>76.97263.63</td><td></td><td></td><td></td><td>184.09179.09 200.90</td></tr><tr><td>Scal3R (CUT3R)</td><td>x</td><td></td><td>45.34164.98139.88</td><td></td><td>33.87</td><td>9.89</td><td>29.37</td><td>47.40</td><td>6.44 227.02</td><td></td><td>29.63</td><td>33.23</td><td>69.73</td></tr><tr><td>Scal3R (STream3R)</td><td>x</td><td></td><td>57.90 176.43 170.75</td><td></td><td>10.86</td><td>9.30</td><td>39.05</td><td></td><td>18.07 15.19 173.83</td><td></td><td>73.91</td><td>33.87</td><td>70.83</td></tr></table>

Table 11: Comparison with the concurrent LongStream [16] on KITTI (ATE ↓). LongStream attains lower absolute ATE by retraining a 1.3B backbone on large-scale data, while Scal3R reaches a comparable regime by adapting a frozen backbone with ∼1% parameters and yields pairwise constraints that LongStream’s single-reference design cannot provide.
<table><tr><td>Method</td><td>Pose formulation</td><td>Drift mitigation</td><td>KITTI ATE ↓</td><td>Backbone</td><td>Train data / views</td><td>Compute</td></tr><tr><td>LongStream [16]</td><td>single-reference</td><td>cache refresh</td><td>51.9</td><td>retrained, 1.3B</td><td>14+ datasets, 10–80 views</td><td>32× A100, &gt;3d</td></tr><tr><td>Scal3R (CUT3R)</td><td>multi-reference</td><td>PGO + loop closure</td><td>69.7</td><td>frozen, ~1%</td><td>TartanAir, 4 views</td><td>1× A100, 8 h</td></tr></table>

## D.2 Comparison with Classic SLAM Systems

Tab. 10 compares Scal3R with classic SLAM systems on all eleven KITTI sequences. ORB-SLAM2 achieves the lowest average ATE (54.82) when calibration is available, yet sufers catastrophic failure on seq. 01 (508.34), exposing its sensitivity to feature-poor highway scenes. DROID-SLAM degrades substantially without calibration (75.67 → 123.80), and MASt3R-SLAM struggles across most sequences regardless of calibration, averaging over 190. Without requiring any camera intrinsics, our CUT3R and STream3R variants achieve 69.73 and 70.83 respectively, competitive with calibrated DROID-SLAM and significantly outperforming all calibration-free baselines. These results demonstrate that multi-reference relative pose querying with PGO can match or surpass established SLAM pipelines on long outdoor sequences while operating in a fully calibration-free online regime.

## D.3 Comparison with the Concurrent LongStream

Tab. 11 situates Scal3R relative to LongStream [16], a concurrent streaming method that also abandons first-frame global regression in favor of relative pose.

Table 12: Asymmetric vs. Symmetric attention injection (ATE ↓). Asymmetric injection is critical for outdoor sequences, reducing ATE by up to 10×.
<table><tr><td rowspan="2">Method</td><td colspan="3">Indoor</td><td colspan="2">Outdoor</td></tr><tr><td>Sintel</td><td>TUM</td><td>ScanNet</td><td>vKITTI</td><td>KITTI</td></tr><tr><td>Symmetric (VPT)</td><td>0.154</td><td>0.044</td><td>0.083</td><td>57.78</td><td>197.45</td></tr><tr><td>Scal3R (Asymmetric)</td><td>0.168</td><td>0.033</td><td>0.092</td><td>5.63</td><td>69.73</td></tr></table>

The two pursue orthogonal directions. LongStream retrains a 1.3B VGGT-based backbone on large-scale driving and indoor data, predicting a single keyframerelative pose per frame and suppressing drift through cache-consistent training with periodic cache refresh. Scal3R instead keeps the backbone frozen and adds ∼1% trainable parameters that query K relative poses jointly in one forward pass. This multi-reference formulation produces pairwise constraints that PGO and loop closure aggregate into a globally consistent trajectory, whereas LongStream’s single-reference chain ofers no such graph structure and cannot close loops.

The two methods occupy diferent points on the accuracy-versus-cost trade-of. LongStream reaches a lower absolute ATE (51.9 vs. 69.7 on KITTI) by retraining a billion-scale backbone with 32 A100 GPUs for more than three days, while Scal3R attains a comparable regime by adapting a frozen backbone in 8 hours on a single GPU from only 4-view TartanAir samples. They are thus complementary rather than competing: LongStream shows that a fully retrained backbone can push absolute accuracy, and Scal3R shows that the same long-sequence collapse can be resolved at the pose-interface level with a small fraction of the data and compute, while additionally enabling loop closure.

## D.4 Ablation: Asymmetric vs. Symmetric Attention Injection

Tab. 12 compares symmetric attention injection via standard visual prompt tuning (VPT) with our asymmetric design. On indoor benchmarks the two variants perform comparably, with symmetric injection slightly better on Sintel and ScanNet while asymmetric injection leads on TUM. The critical diference emerges on outdoor sequences: asymmetric injection reduces ATE by an order of magnitude on vKITTI (57.78 → 5.63) and by nearly 3× on KITTI (197.45 → 69.73). Symmetric injection allows pose query tokens to attend to and be attended by all image tokens bidirectionally, which can dilute reference-specific geometric cues in long-range outdoor settings. By restricting the information flow so that pose query tokens read from image features without modifying them (Eqs. (2)– (3) in the main paper), asymmetric injection preserves the frozen backbone’s representation quality and enables more accurate relative pose prediction at scale.

## D.5 Compatibility with Zero-Shot Methods

Tab. 13 demonstrates that Scal3R is complementary to zero-shot test-time training methods such as TTT3R. Applying TTT3R on top of our pipeline consistently improves ATE across all five benchmarks, with notable gains on ScanNet (0.092 → 0.064) and KITTI (69.73 → 62.05). Since Scal3R produces globally optimized pose graphs, TTT3R benefits from higher-quality initial estimates compared to operating on raw sequential predictions. This result confirms that our multireference relative pose querying and PGO pipeline serves as an efective foundation that can be further refined by orthogonal adaptation strategies at test time. Fig. 17 illustrates this complementarity on KITTI Seq. 09, where adding TTT3R tightens the estimated trajectory against the ground truth and lowers ATE from 29.6 to 20.6.

Table 13: Compatibility with zero-shot test-time training (ATE ↓). TTT3R consistently improves ATE when applied on top of Scal3R’s globally optimized poses.
<table><tr><td rowspan="2">Method</td><td colspan="3">Indoor</td><td colspan="2">Outdoor</td></tr><tr><td>Sintel</td><td>TUM</td><td>ScanNet</td><td>vKITTI</td><td>KITTI</td></tr><tr><td>Scal3R</td><td>0.168</td><td>0.033</td><td>0.092</td><td>5.63</td><td>69.73</td></tr><tr><td> $\mathrm { S c a l 3 R + T T T 3 R }$ </td><td>0.154</td><td>0.028</td><td>0.064</td><td>4.66</td><td>62.05</td></tr></table>

![](images/40a840a5de83a2dd4572fe33b8f4d546e18448a30679e657e12f5588724b2c9b.jpg)  
Fig. 17: Qualitative efect of test-time training on KITTI Seq. 09. Adding TTT3R on top of Scal3R tightens the estimated trajectory against the ground truth, lowering ATE from 29.6 to 20.6.

## D.6 Robustness to Dynamic Objects and Occlusions

Fig. 18 visualizes the attention maps of the relative pose query tokens $\left\{ \tilde { \mathbf { q } } _ { k } \right\}$ on TUM-Dynamic, where pedestrians occupy a significant portion of the frame. The attention concentrates on static background structures such as desks, monitors, and walls, while assigning low activation to the moving persons in the foreground. This behavior emerges naturally from the pose query tokens without any explicit dynamic-object mask or motion segmentation module, suggesting that the frozen backbone features already encode suficient cues for the pose query tokens to distinguish static geometry from transient content. The learned down-weighting of dynamic regions explains why Scal3R maintains accurate pose estimation on TUM-Dynamic despite large occlusions, and indicates that asymmetric attention injection provides an implicit robustness mechanism against scene dynamics.

![](images/7af3d9d8166d6fa000538d31f8e9e721657037a98daa9cea383deaf314654ab4.jpg)  
Fig. 18: Attention maps of relative pose query tokens on TUM-Dynamic. The pose query tokens attend primarily to static structures while implicitly down-weighting dynamic pedestrians, without any explicit motion segmentation.