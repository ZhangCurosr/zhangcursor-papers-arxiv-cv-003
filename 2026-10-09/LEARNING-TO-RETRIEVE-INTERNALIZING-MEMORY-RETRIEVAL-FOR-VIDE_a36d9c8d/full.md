# LEARNING TO RETRIEVE: INTERNALIZING MEMORY RETRIEVAL FOR VIDEO WORLD MODELS

JiaKui Hu<sup>1,2∗</sup> Tailai Chen<sup>3</sup> <sup>∗</sup> Yuqi Pan<sup>3</sup> Xuerui Qiu<sup>3</sup> Jialun Liu<sup>4</sup> Xiao Cao<sup>5</sup> Zhenxin Zhu<sup>2</sup> Guang Chen<sup>2</sup> Hangjun Ye<sup>2</sup> Bing Wang<sup>2†</sup> Yanye Lu<sup>1‡</sup> <sup>1</sup>Peking University <sup>2</sup>Xiaomi EV <sup>3</sup>CASIA <sup>4</sup>UQ <sup>5</sup>NUS

![](images/c1380609d87e77062223a4681176f68b584358936dd71fd705bdc6cf1d3de145.jpg)  
Figure 1: Teaser demonstration. We introduce Learning-to-Retrieve (L2R), which can generate consistent videos without external memory modules or 3D reconstruction pipelines. Red boxes mark content observed from matched camera poses separated by a long temporal interval, illustrating that L2R preserves scene identity when the camera revisits a previously observed region.

## ABSTRACT

Video world models aim to generate explorable, 3D-consistent scene videos conditioned on camera trajectories. Existing approaches often rely on external memory systems that explicitly retrieve previously observed content to mitigate scene drift during long-horizon generation. However, these auxiliary memory pathways operate outside the model’s internal generative dynamics, preventing the model from intrinsically learning when and what historical information should be retrieved. We propose to internalize memory retrieval into the generation process, allowing retrieval to emerge as an intrinsic behavior of the video world model rather than relying on an external memory system. Based on this principle, we introduce Learning-to-Retrieve (L2R), which repurposes the model’s persistent internal state as a memory for historical context. A camera-conditioned retrieval gate selectively accesses relevant historical information from this state, determining what to retrieve, while a retrieval trigger determines when to retrieve. We further supervise the trigger with a 3D re-visibility signal, activating retrieval when previously observed content re-enters the current view while otherwise preserving the existing context. Together, these components enable the model to intrinsically acquire memory retrieval behavior and incorporate relevant historical observations into generation without a separate retrieval pathway. Across multiple base models and camera-revisit benchmarks, L2R improves long-term scene consistency while eliminating the need for an external memory bank or 3D conditions.

## 1 INTRODUCTION

Video world models aim to generate video sequences that explore a 3D scene according to a userspecified camera trajectory, conditioned on one or a small set of reference frames (Yu et al., 2025a; Xiao et al.; Zhu et al., 2026). Beyond perceptual realism and accurate camera control, a key requirement is scene-level consistency: upon revisiting previously observed regions, the generated content should remain consistent with earlier generations.

Existing strategies for enforcing scene consistency can be broadly grouped into two paradigms. Yu et al. (2025c); Huang et al. (2025); Hu et al. (2026a); Zheng et al. (2026); Yang et al. (2026) formulate this problem as conditional generation, leveraging reconstruction models (Wang et al., 2025; 2026b) to produce conditions intended to improve the fidelity and coherence of the generated video. However, the effectiveness of this approach is closely tied to reconstruction accuracy. Reconstruction errors can propagate to downstream generation, degrading long-term quality.

The second paradigm integrates memory retrieval into the video generation model to improve longterm scene consistency. In these methods, long-term memory serves as an auxiliary context module: when past frames are spatially relevant to the target view, they are retrieved to augment the default short-term sliding-window context. The principal differences between these methods lie in the retrieval strategy. Earlier approaches (Li et al., 2025; Yu et al., 2025a; Xiao et al.; Yu et al., 2026b; Wang et al., 2026c) select historical context via camera overlap or explicit 3D signals and supply the retrieved views to the generator. More recent approaches make retrieval learnable, but still rely on separate memory access pathways. For example, MemLearner (Yu et al., 2026a) queries historical context using learnable query tokens, while CaR (Peng et al., 2026) adds a dedicated retrieval-attention branch on compressed context. Although such designs improve long-range scene consistency, their reliance on external or auxiliary memory access pathways introduces a mismatch between retrieval and the model’s internal generative dynamics.

In this paper, we advance the perspective that video world models should intrinsically acquire the memory retrieval policy. Rather than treating memory retrieval as an external context construction procedure, we aim to make retrieval part of the generation process itself. This requires two capabilities to be jointly learned: maintaining historical information that remains accessible over long horizons, and selectively accessing that information according to the current observation. By integrating these two capabilities into the generative pathway, memory access can directly adapt to the model’s evolving generation dynamics.

To realize this design, we formulate memory retrieval within an attention mechanism with a persis tent contextual state and selective state access. In this formulation, historical observations are con tinuously accumulated in the model’s internal state, while the state-access mechanism determines which historical information is retrieved for the current generation. Our construction builds on the state-update structure (Katharopoulos et al., 2020; Yang et al., 2024; 2025b; Pan et al., 2025), but repurposes it for memory retrieval. A retrieval trigger determines when to retrieve, while a cameraconditioned gate determines what historical information to retrieve, jointly coupling memory access with the current view. This shift repurposes the persistent attention state as an internal memory from which historical context can be selectively retrieved during long-horizon generation.

Building on this observation, we introduce L2R, a mechanism designed to approximate memory retrieval in video world models. L2R employs a camera-conditioned gate to selectively access his torical information from the accumulated state according to the current viewpoint. In addition, L2R incorporates a retrieval trigger that determines whether a given token should initiate retrieval: when the current view revisits previously observed content, retrieval is activated; otherwise, the model relies on the existing context. We supervise the retrieval trigger using the 3D signal based on visibility re-entry: tokens whose content exits the field of view or becomes occluded and subsequently reappears are annotated as retrieval-relevant. This supervision encourages the model to identify situations in which historical context is necessary. Collectively, L2R enables attention to acquire retrieval behavior intrinsically: the trigger learns when to retrieve, and the camera-conditioned gate learns what to retrieve, thereby coupling the current view with access to historical context without introducing a separate retrieval pathway or requiring explicit 3D correspondence at inference time.

We evaluate L2R on scene exploration with re-appear and loop-closure trajectories under multiple state-update rules. Given a reference image and a camera trajectory, video world models using L2R maintain scene consistency in the generated video without degrading visual quality or camera controllability. Both quantitative and qualitative results demonstrate that the method outperforms existing approaches in terms of consistency and exhibits robust generalization capabilities to unseen camera trajectories. Furthermore, ablation studies validate the effectiveness of the proposed design.

## 2 RELATED WORK

Memory in video generation. In video generation, particularly in autoregressive (AR) settings, the incorporation of an explicit memory is critical for maintaining and exploiting long-term consistency. Existing systems represent and utilize historical information through a variety of mechanisms, including sliding-window strategies (Chen et al., 2026), attention sinks (Yang et al., 2025a), positional extrapolation (Zhao et al., 2025; Yesiltepe et al., 2026), state-space architecture (Yu et al., 2025d), token compression of past context (Zhang & Agrawala, 2025), and hybrid combinations of these techniques (Yi et al., 2025). More recently, ARL<sup>2</sup> (Li et al., 2026) introduced a memory mechanism for video generation, using the recurrent state as the memory bank. However, it lacks the ability to perform conditioned retrieval, thereby compromising the model’s memory capabilities.

Video world models are designed to generate scene-level videos that are both temporally coherent and dynamically consistent, typically using video diffusion models conditioned on camera trajectories or action controls. In these scenarios involving camera pose or 3D data, memory can serve not only as an external component but can also be retrieved. Many existing methods (Yu et al., 2025c;b; Hu et al., 2026a; Yang et al., 2026; Zheng et al., 2026; Wei et al., 2026; Xu et al., 2026b) incorporate substantial 3D information, such as depth maps, point clouds, or 3D Gaussian splatting (3DGS), as an additional conditioning memory for the diffusion process. A complementary research direction (Li et al., 2025; Yu et al., 2025a; Xiao et al.; Wu et al., 2026; Xu et al., 2026a; Zhu et al., 2026) investigates the memory retrieval and compression mechanisms that operate outside the diffusion backbone. These methods store and retrieve historical frames based on camera-view overlap (Yu et al., 2025a) or index past observations via geometric associations to observed three-dimensional surface segments (Li et al., 2025). In addition, recent methods (Peng et al., 2026; Wang et al., 2026c; Yu et al., 2026a) introduce external learnable modules to retrieve memory for video world models, but keep retrieval separate from the generation model. Wonder (Xu et al., 2026a) uses sparse attention, allowing inference to attend to a small subset of context tokens. However, its top-k selection offers no principled guarantee that the chosen tokens are the most geometrically relevant.

## 3 METHOD

In contrast to previous methods, we propose L2R, a mechanism to accurately retrieve the historical context within the model, thus promoting improved consistency in the generated world videos. Sec. 3.1 presents the necessary preliminaries. Sec. 3.2 specifies the implementation of our L2R and explains how it retrieves the historical context. Sec. 3.3 summarizes the training pipeline.

## 3.1 PRELIMINARIES

Memory retrieval in video world model. Given a reference video $\{ V _ { i } \} _ { i = 1 } ^ { M }$ and a novel camera trajectory $P _ { f }$ , a video world model $\varrho$ generates the video $V _ { f }$ observed along the target trajectory.

When the target viewpoint shares a high degree of consistency with a previous reference viewpoint (e.g. significant FoV overlap or similar geometric textures), existing approaches (Yu et al., 2025a) often introduce an external memory retrieval module to identify the trajectory-relevant information from the reference video and use the retrieved content as a context for generation.

$$
\mathcal { C } _ { f } = \left\{ \begin{array} { l l } { \mathrm { R e t r i e v e } \Big ( \{ V _ { i } \} _ { i = 1 } ^ { f - 1 } , P _ { f } \Big ) , } & { \mathrm { i f ~ t a r g e t ~ v i e w ~ r e v i s i t s ~ c o n t e n t } , } \\ { \mathcal { C } _ { f } ^ { \mathrm { b a s e } } , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.
$$

$$
V _ { f } = \varrho ( { \mathcal { C } } _ { f } , P _ { f } ) ,\tag{1}
$$

where $\mathcal { C } _ { f } ^ { \mathrm { b a s e } }$ denotes the default context used for generation.

Here, retrieval consists of two implicit decisions: whether historical information can be needed for the current view, and which historical information should be accessed when retrieval is activated. In existing methods, both decisions are made through an auxiliary memory access pathway outside the core generation process.

## 3.2 LEARNING TO RETRIEVE

Internalize memory retrieval. Inside the video diffusion transformer, video is compressed into a 1d token sequence $\{ x _ { t } \} _ { t = 1 } ^ { N }$ , where t denotes the index of the token and N is the sequence length. In this setting, Eq. 1 will be transformed into the following form:

$$
c _ { t } =  \left\{ \begin{array} { l l } { _ { \operatorname { R e t r i e v e } } ( \{ x _ { i } \} _ { i = 1 } ^ { t - 1 } , p _ { t } ) , } & { { \mathrm { i f } } \ r _ { t } = 1 , } \\ { \qquad \operatorname { \_ } { x _ { i } \} _ { i = t - w } ^ { t - 1 } } & { { \mathrm { i f } } \ r _ { t } = 0 , } \end{array} \right. } \qquad x _ { t } = \varrho ( c _ { t } , p _ { t } ) ,\tag{2}
$$

where $p _ { t } \in \mathbb { R } ^ { 1 2 }$ is the camera Plucker ray of ¨ $x _ { t }$ . w is the size of the sliding window. $r _ { t }$ indicates whether memory retrieval is activated for $x _ { t }$ . It emulates the video world model by initiating retrieval when the geometric overlap between the target and context views is sufficiently high.

To internalize this retrieval, we store the historical context within the model state, allowing subsequent tokens to directly access the information from preceding tokens. A common practice is to encode such information in an internal recurrent state (Yang et al., 2024; 2025b; Pan et al., 2025; Yu et al., 2025d). Denoting the accumulated state before processing $x _ { t }$ as $s _ { t - 1 }$ , which encodes infor mation accumulated from preceding tokens, the historical context $c _ { t }$ can therefore be read directly from this state.

To realize the retrieval defined in Eq. 2, we read the historical context directly from the accumulated state. Specifically, $r _ { t }$ determines whether retrieval is activated, while the pose-conditioned gate $\gamma _ { t }$ determines which information is retrieved. The resulting context $c _ { t }$ is then used to update the state:

$$
c _ { t } = { \left\{ \begin{array} { l l } { \gamma _ { t } s _ { t - 1 } , } & { { \mathrm { I f } } \ r _ { t } = 1 , } \\ { s _ { t - 1 } , } & { { \mathrm { I f } } \ r _ { t } = 0 , } \end{array} \right. } \qquad s _ { t } = \varphi ( c _ { t } , p _ { t } ) ,\tag{3}
$$

where $\varphi$ denotes the state transition rule.

Designing $\gamma _ { \mathrm { t } } \mathbf { : }$ camera-guided retrieval. As shown in Eq. 3, memory retrieval relies on the camera pose $p _ { t }$ as a condition. To this end, we introduce camera-guided retrieval for L2R:

$$
\log \gamma _ { t } \gets \log \gamma _ { t } \odot \big ( 1 + \operatorname { t a n h } ( W _ { g } \phi _ { t } ) \big ) ,\tag{4}
$$

where feature $\phi _ { t } \in \mathbb { R } ^ { D _ { p } }$ is the projection of $p _ { t }$

To promote numerical stability in the gated retrieval procedure, following previous work (Yang et al., 2024; 2025b; Pan et al., 2025), we perform multiplicative gating computations in log space.

Designing r : retrieval trigger and re-visibility supervision. As shown in Eq. 3, the behavior of L2R-GLA is consistent with the memory module of a video world model: the memory retrieval operator is triggered only when the viewpoint associated with the current token exhibits a high degree of consistency with viewpoints already represented in the context. The variable $r _ { t }$ is introduced to parameterize and learn this retrieval activation.

We learn $r _ { t }$ under re-visibility supervision. As shown in Figure 2, for each frame in a training clip, the supervisory target for $r _ { t }$ is computed offline via a three-stage procedure: (1) estimating per-frame depth using a feed-forward geometric model (Wang et al., 2025; 2026b); (2) unprojecting each token in the frame to a 3D point using the estimated depth; and (3) re-projecting the 3D points of earlier frames into the current frame using the known camera trajectory. If a 3D surface point observed in the previous frame exits the frustum (or is occluded) and reappears in the current frame, the label $y _ { t } = 1$ is assigned to the corresponding token. We optimize $r _ { t }$ using the following objective:

$$
\mathcal { L } _ { \mathrm { r } } = \frac { 1 } { | \mathcal { V } | } \sum _ { t \in \mathcal { V } } \mathrm { B C E } _ { w _ { + } } \left( \hat { r } _ { t } , \ y _ { t } \right) ,\tag{5}
$$

where $\hat { r } _ { t }$ is the sigmoid probability before binarization, V contains valid target-frame tokens, and $w _ { + } = 3$ is the positive-class weight. For numerical stability, we compute the weighted BCE directly from gate logits and mask invalid tokens and context frames. Since binarization is non-differentiable, we stop gradients through $r _ { t }$ and force the model to retrieve memory on this discrete trigger.

![](images/cc57b5fc16a7ceb159a4bd71397c95d2ca9fa1ff2ccdeff50c86a877a8c59985.jpg)  
Figure 2: Three stages to compute re-visibility supervision. (1) estimate per-frame depth; (2) unproject each token to 3D points; and (3) re-project previous frames’ 3D points into the current frame. If a previously observed 3D point leaves the frustum (or is occluded) and later reappears, assign $y _ { t } = 1$ to the corresponding token.

Extension. Eq. 3 defines a mechanism-independent retrieval recipe that can be instantiated with different gated state transition mechanisms (Yang et al., 2024; 2025b). In this paper, we validate this by instantiating the same recipe with both Gated Linear Attention (GLA) (Yang et al., 2024) and Gated DeltaNet (GDN) (Yang et al., 2025b). For GDN, the instantiation requires an additional modification to its delta correction, and we retain its subtractive memory gating. Details of this adaptation are provided in Appendix B.

Instantiating L2R with these two state transition rules yields L2R-GLA and L2R-GDN, respectively. We use these names throughout the remainder of this paper.

## 3.3 TRAINING

During model training, we support both training from scratch and supervised fine-tuning (SFT), depending on the experimental configuration required for evaluation. For most evaluation protocols, training is initialized from parameters loaded from a pre-trained video generation model.

Integration into a pretrained model. In every group of n blocks in the pre-trained cameracontrollable video diffusion model, we replace the first n − 1 self-attention layers with our memory layer (L2R-GLA). The remaining blocks retain softmax attention, so the attention captures the local spatial structure while the memory state provides temporal persistence. For each replacement, we initialize the memory layer’s query, key, value, and output projections from the attention projections, and zero-initialize all camera-dependent paths so the layer initially matches the pre-trained token representation.

Objective. We optimize the rectified-flow objective together with re-visibility supervision:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { d e n o i s e } } + \lambda _ { \mathrm { r } } \mathcal { L } _ { \mathrm { r } } ,\tag{6}
$$

where $\mathcal { L } _ { \mathrm { d e n o i s e } }$ is the timestep-weighted mean-squared error between the predicted flow velocity and $\epsilon - z _ { 0 }$ , where $z _ { \mathrm { 0 } }$ denotes the clean video latents and ϵ is Gaussian noise. This loss is computed only on target-frame latents; reference and context frames are excluded. The visibility loss ${ \mathcal { L } } _ { \mathrm { r } }$ in Eq. 5 is normalized over valid target tokens within each memory layer and then averaged across memory layers. We use $\lambda _ { \mathrm { r } } = 0 . 0 5$ by default.

## 4 EXPERIMENTS

## 4.1 SETUP

We evaluate the performance of L2R-GLA and L2R-GDN along two dimensions: memory capacity and video quality. The evaluation of the former characterizes the module’s ability to maintain longterm consistency, whereas the evaluation of the latter verifies that this memory mechanism does not degrade visual fidelity or compromise camera controllability.

Memory capacity. Following Echo-Memory (King et al., 2026), we conduct a comprehensive evaluation of memory capabilities. We adopt Wan2.1-T2V-1.3B (Wan et al., 2025) as the base model and follow the camera-conditioned method introduced in Echo-Memory. To ensure experimental comparability, we use the static scene-exploration dataset (Yu et al., 2025a) and all methods use exactly the same training and inference settings. The implementation details are shown in Appendix A. Performance is evaluated using a three-branch protocol. (i) Replay: the model generates videos conditioned on given camera trajectories, and memory capacity is quantified using frame-aligned Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM), and Learned Perceptual Image Patch Similarity (LPIPS). (ii) In-domain loop closure: outgoing and returning frames at matched camera poses are compared using the same image-quality metrics to measure revisit consistency. (iii) Open domain return (O-V): scene-level consistency and object-identity preservation are assessed on first-frame probes released using the benchmark’s vision language judge. This assesses the model’s generalization capability to unseen trajectories and scenes.

Video quality. To isolate whether our memory modules improve long-horizon coherence without trading off perceptual fidelity, we curate a training corpus of 5-second clips without camera loops, sampled from RealEstate10K, DL3DV, and MiraData9K. For evaluation, we build a “forth-andback” set on RealEstate10K and DL3DV, in which the second half explicitly retraces the camera path back to the start, yielding a controlled test of out-of-domain viewpoint regression under increasing drift. Visual quality is reported with VBench (Huang et al., 2024). Camera controllability is measured by pose agreement between the ground-truth trajectory and the trajectory re-estimated from generated frames using Pi3X (Wang et al., 2026b). We report rotation and translation errors $( R _ { e r r } , T _ { e r r } )$ following VMem (Li et al., 2025). We instantiate the pipeline on Wan2.2-TI2V-5B and SANA-WM-2.6B (Zhu et al., 2026). Remaining implementation details are deferred to Appendix A.

## 4.2 RESULTS ON MEMORY CAPACITY

Table 1 compares memory capacity across replay, loop closure, and open-domain return. L2R-GLA performs best, ranking first on all replay metrics and on SSIM and O-V for loop closure and open-domain return. On replay, it raises PSNR from 13.71 (best baseline, VMem (Li et al., 2025)) to 14.74 (+1.03 dB), with higher SSIM and lower LPIPS. Versus Context as Memory (Yu et al., 2025a) at K=20, it improves PSNR by 2.20 dB (12.54→14.74), SSIM by 0.064 (0.449→0.513), and reduces LPIPS by 0.066 (0.496→0.430). The GLA baseline degrades in SSIM and LPIPS, likely because unconstrained retrieval causes memorization drift. Overall, integrating retrieval into the video generation model substantially improves consistency when previously observed content is revisited. L2R-GDN also shows strong performance, achieving 13.86 dB PSNR on replay and 69.37 O-V, which suggests that the proposed formulation remains effective when combined with GDN.

A similar trend is observed under loop closure. L2R-GLA achieves the best SSIM of 0.375 and the second-best PSNR of 13.14, closely matching the strongest baseline, VMem, at 13.17. It also substantially improves over Context as Memory at K=20, increasing PSNR from 11.07 to 13.14 and SSIM from 0.359 to 0.375. Although Context as Memory attains a lower LPIPS, L2R-GLA provides stronger pixel-level fidelity together with higher structural similarity. This indicates that the proposed internal retrieval mechanism is particularly effective at preserving the appearance and structure of previously observed content during viewpoint revisits.

For open-domain return, L2R-GLA achieves an O-V score of 72.57, outperforming VMem (Li et al., 2025) by 7.84 points and Context as Memory at K=20 by 13.94 points. Since this protocol evaluates held-out first-frame probes with distinctive objects, the improvement suggests that the learned internal retrieval policy can generalize beyond the training trajectories and recover previously observed scene content under unseen return scenarios. This further supports the ability of L2R to preserve scene identity without relying on an external memory bank.

Table 1: Quantitative comparison on memory capacity. Best in bold, second best underlined. ∗ denotes methods are retrained because their released weights were not trained on dataset (Yu et al., 2025a). The results for the other methods are taken from Echo-Memory (King et al., 2026).
<table><tr><td rowspan="2">Method</td><td colspan="3">Replay</td><td colspan="3">Loop closure</td><td rowspan="2">0-V↑</td></tr><tr><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS↓</td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>I2V baseline (Wan2.1)</td><td>10.03</td><td>0.398</td><td>0.534</td><td>10.32</td><td>0.291</td><td>0.643</td><td>12.25</td></tr><tr><td>Spatial Memory</td><td>13.60</td><td>0.411</td><td>0.554</td><td>10.04</td><td>0.260</td><td>0.617</td><td>6.00</td></tr><tr><td>VideoSSM</td><td>12.69</td><td>0.344</td><td>0.581</td><td>12.23</td><td>0.298</td><td>0.584</td><td>34.75</td></tr><tr><td>ARL2 *</td><td>12.73</td><td>0.326</td><td>0.594</td><td>12.13</td><td>0.309</td><td>0.633</td><td>43.06</td></tr><tr><td>Context as Memory, K=5</td><td>11.92</td><td>0.408</td><td>0.501</td><td>10.72</td><td>0.307</td><td>0.596</td><td>50.75</td></tr><tr><td>Context as Memory, K=20</td><td>12.54</td><td>0.449</td><td>0.496</td><td>11.07</td><td>0.359</td><td>0.543</td><td>58.63</td></tr><tr><td>WorldMem*</td><td>12.34</td><td>0.471</td><td>0.536</td><td>11.59</td><td>0.348</td><td>0.594</td><td>59.74</td></tr><tr><td>VMem*</td><td>13.71</td><td>0.502</td><td>0.507</td><td>13.17</td><td>0.352</td><td>0.581</td><td>64.73</td></tr><tr><td>L2R-GLA (ours)</td><td>14.74</td><td>0.513</td><td>0.430</td><td>13.14</td><td>0.375</td><td>0.603</td><td>72.57</td></tr><tr><td>L2R-GDN (ours)</td><td>13.86</td><td>0.409</td><td>0.558</td><td>12.81</td><td>0.374</td><td>0.614</td><td>69.37</td></tr></table>

![](images/fe32578068afa713c316e4f250e9626a07cc15571eab65527e32f2b9a40b82ce.jpg)  
Figure 3: Qualitative comparison on memory capacity. Red boxes highlight the revisited regions. L2R-GLA has the best scene consistency.

The qualitative comparisons in Figure 3 further illustrate the memory behavior of L2R-GLA. After the camera moves away from a region and later returns to a matched viewpoint, L2R-GLA better preserves the appearance of the revisited content while maintaining the surrounding scene layout. The red boxes highlight these regions, showing that internal retrieval helps recover previously observed details with improved scene consistency.

## 4.3 RESULTS ON VIDEO QUALITY

Table 2 demonstrates that L2R improves revisit consistency while maintaining strong video quality and camera controllability across different underlying video generation models. With Wan2.2 as the base model, both L2R variants achieve the highest or second-highest scores across all six visualquality dimensions, with L2R-GLA ranking first in all of them. Meanwhile, L2R substantially improves camera controllability over prior memory-based methods. Compared with CaR, L2R-GLA reduces the rotation and translation errors from 12.98 to 6.29 and from 0.354 to 0.300, respectively, and L2R-GDN achieves 6.28 and 0.282. For revisit consistency, L2R-GDN attains the best PSNR of 17.03, improving over CaR by 0.19 dB, while L2R-GLA remains competitive at 16.79. These results indicate that the proposed internal retrieval mechanism improves scene consistency without requiring a trade-off with video quality or camera control.

Table 2: Quantitative results of video quality, camera controllability, and re-visit consistency on RealEstate10K and DL3DV samples. Best in bold, second best underlined. <sup>∗</sup> marks methods we retrained on our dataset because checkpoints weren’t released; others use official checkpoints.
<table><tr><td rowspan="2">Method</td><td colspan="6">Visual quality (↑)</td><td colspan="2">Camera control (↓)</td><td colspan="2">Consistency (↑)</td></tr><tr><td>Subject Consist</td><td> $\mathbf { B g }$ </td><td>Motion Consist Smooth</td><td>Temporal Flicker</td><td>Aesthetic Quality</td><td>Imaging Quality</td><td> $R _ { e r r }$ </td><td> $T _ { e r r }$ </td><td>PSNR</td><td>SSIM</td></tr><tr><td>ViewCrafter</td><td>78.51</td><td>88.30</td><td>95.12</td><td>92.43</td><td>46.28</td><td>63.42</td><td>8.86</td><td>0.482</td><td>13.81</td><td>0.5217</td></tr><tr><td>SEVA</td><td>90.53</td><td>93.79</td><td>96.68</td><td>93.41</td><td>46.01</td><td>64.08</td><td>5.26</td><td>0.249</td><td>16.62</td><td>0.5427</td></tr><tr><td>VMem</td><td>87.71</td><td>92.50</td><td>94.16</td><td>90.41</td><td>42.05</td><td>61.50</td><td>10.31</td><td>0.518</td><td>16.85</td><td>0.5505</td></tr><tr><td>SANA-WM</td><td>91.73</td><td>93.67</td><td>96.81</td><td>93.99</td><td>46.58</td><td>66.47</td><td>5.40</td><td>0.383</td><td>15.52</td><td>0.5076</td></tr><tr><td>AnchorWeave-Wan2.2*</td><td>88.70</td><td>92.35</td><td>95.94</td><td>92.71</td><td>47.50</td><td>65.24</td><td>6.81</td><td>0.299</td><td>16.55</td><td>0.5472</td></tr><tr><td>CaR-Wan2.2</td><td>91.47</td><td>93.28</td><td>97.56</td><td>94.37</td><td>49.20</td><td>66.34</td><td>12.98</td><td>0.354</td><td>16.84</td><td>0.5456</td></tr><tr><td>L2R-GLA (Wan2.2)</td><td>92.67</td><td>94.41</td><td>97.68</td><td>95.53</td><td>50.14</td><td>67.18</td><td>6.29</td><td>0.300</td><td>16.79</td><td>0.5480</td></tr><tr><td>L2R-GDN (Wan2.2)</td><td>92.46</td><td>94.38</td><td>97.62</td><td>95.24</td><td>49.60</td><td>65.97</td><td>6.28</td><td>0.282</td><td>17.03</td><td>0.5546</td></tr><tr><td>L2R-GLA (SANA-WM)</td><td>91.36</td><td>94.17</td><td>97.02</td><td>93.84</td><td>46.81</td><td>66.07</td><td>5.61</td><td>0.259</td><td>16.97</td><td>0.5556</td></tr><tr><td>L2R-GDN (SANA-WM)</td><td>90.85</td><td>93.86</td><td>96.68</td><td>93.43</td><td>46.16</td><td>65.13</td><td>5.40</td><td>0.246</td><td>17.29</td><td>0.5597</td></tr></table>

Using SANA-WM as the base model, the same trend is observed. L2R-GDN achieves substantial gains in revisit consistency, improving PSNR from 15.52 to 17.29 and SSIM from 0.5076 to 0.5597, while also reducing the translation error from 0.383 to 0.246. Although its visual-quality scores are slightly lower than the SANA-WM baseline, the gap remains small across the six dimensions. L2R-GLA similarly improves revisit consistency to 16.97 PSNR and 0.5556 SSIM while maintaining competitive visual quality and camera control. Overall, these results show that L2R transfers across different underlying state-transition mechanisms and consistently strengthens long-term scene consistency without substantially degrading generation quality or camera controllability.

## 4.4 ABLATION

To fully understand how each design element in L2R-GLA affects its memory capability, we conducted ablation studies on Echo-Memory.

Table 3: Ablation on the memory capacity. GLA (baseline) is our starting point. The last three rows progressively add camera-guided retrieval $\gamma _ { t }$ , retrieval trigger $r _ { t }$ and re-visibility supervision (Eq. 5). Best in bold, second best underlined.
<table><tr><td rowspan="2">Variant</td><td colspan="3">Replay</td><td colspan="3">Loop closure</td><td rowspan="2"> $\scriptstyle \mathrm { O - V } \ { \uparrow }$ </td></tr><tr><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS↓</td></tr><tr><td>No Memory</td><td>10.03</td><td>0.398</td><td>0.534</td><td>10.32</td><td>0.291</td><td>0.643</td><td>12.25</td></tr><tr><td>VideoSSM</td><td>12.69</td><td>0.344</td><td>0.581</td><td>12.23</td><td>0.298</td><td>0.584</td><td>34.75</td></tr><tr><td>Context as Memory,  $K { = } 5$ </td><td>11.92</td><td>0.408</td><td>0.501</td><td>10.72</td><td>0.307</td><td>0.596</td><td>50.75</td></tr><tr><td>GLA</td><td>10.55</td><td>0.199</td><td>0.763</td><td>12.74</td><td>0.273</td><td>0.584</td><td>66.13</td></tr><tr><td>+ Camera-guided retrieval  $( \gamma _ { t } )$ </td><td>13.88</td><td>0.421</td><td>0.560</td><td>12.76</td><td>0.399</td><td>0.687</td><td>68.74</td></tr><tr><td>+ Retrieval trigger (rt)</td><td>14.02</td><td>0.402</td><td>0.483</td><td>13.11</td><td>0.383</td><td>0.601</td><td>71.42</td></tr><tr><td>+ Re-visibility supervision  $( \mathrm { E q . } 5 )$ </td><td>14.74</td><td>0.513</td><td>0.430</td><td>13.14</td><td>0.375</td><td>0.603</td><td>72.57</td></tr></table>

Effects of the retrieval gate $\gamma _ { t }$ and the retrieval trigger $r _ { t } .$ Table 3 separates the two mechanisms and highlights their distinct contributions. Starting from the GLA baseline, introducing cameraguided retrieval with $\gamma _ { t }$ yields the largest improvement in replay quality, with PSNR increasing from 10.55 to 13.88, SSIM from 0.199 to 0.421, and LPIPS decreasing from 0.763 to 0.560. This demonstrates the importance of using the camera pose to selectively access relevant historical information. Adding the retrieval trigger $r _ { t }$ further improves the results, increasing Replay PSNR from

<table><tr><td>Method</td><td>Layer ratio</td><td>Replay PSNR ↑</td><td>Loop closure PSNR↑</td></tr><tr><td>I2V baseline</td><td>-</td><td>10.03</td><td>10.32</td></tr><tr><td rowspan="3">L2R-GDN</td><td>1:1</td><td>13.62</td><td>12.98</td></tr><tr><td>3:1</td><td>13.88</td><td>12.76</td></tr><tr><td>7:1</td><td>8.95</td><td>8.75</td></tr><tr><td rowspan="3">L2R-GLA</td><td>1:1</td><td>14.26</td><td>13.55</td></tr><tr><td>3:1</td><td>14.74</td><td>13.14</td></tr><tr><td>7:1</td><td>10.52</td><td>10.37</td></tr></table>

(a) Layer-ratio ablation

![](images/b506e142678f299d8683654e567e923cb12e55ce97939832a3d2c27600f3ef78.jpg)  
(b) Latency per denoising step  
Figure 4: Layer-ratio ablation and inference efficiency. Left: L2R-GDN and L2R-GLA with layer ratios of 1:1, 3:1, and 7:1 on Echo-Memory benchmark, evaluated by replay and loop-closure PSNR. Right: latency comparison on one H100, 5 s chunks.

13.88 to 14.02 and reducing LPIPS from 0.560 to 0.483, while also improving Loop-closure PSNR from 12.76 to 13.11 and LPIPS from 0.687 to 0.601. The overall O-V score likewise increases from 68.74 to 71.42. These results indicate that camera-conditioned retrieval provides the main capacity gain, while the retrieval trigger further improves retrieval selectivity by activating memory access when historical information is needed.

Effects of the re-visibility supervision (Eq. 5). Re-visibility supervision further strengthens scene consistency by sharpening the trigger on revisits (Table 3). On replay, it improves PSNR by 0.72 dB, increases SSIM by 0.111, and reduces LPIPS from 0.483 to 0.430. Importantly, the auxiliary target is not merely an across-the-board image-metric regularizer. Its primary effect is to make triggering more selective and useful for revisited content, which directly translates into stronger memory.

Layer ratios. We further vary the ratio of L2R-GLA / L2R-GDN layers to standard softmax attention (Table 4 (a)). Both attention variants outperform the I2V baseline in replay and loop-closure PSNR, but the ratio matters: 3:1 maximizes replay PSNR, 1:1 maximizes loop-closure PSNR, and 7:1 degrades both. We therefore adopt 3:1 L2R-GLA in the main comparison to prioritize replay strength while keeping loop closure competitive. We attribute this trend to the fixed-capacity bottleneck of the linear-attention state: as trajectories grow longer and more diverse, the recurrent summary must overwrite earlier details, so stacking too many L2R-GLA layers amplifies information loss and can hurt both replay and loop closure. Retaining a fraction of softmax-attention layers acts as a high-bandwidth fallback that re-injects global context and stabilizes long-horizon generation, preventing this form of memory collapse. We discuss this further in Appendix C.

## 4.5 EFFICIENCY

Figure 4(b) shows the per-step denoising cost on one H100 as rollout length increases, with resolution, camera conditioning, and chunk size matched. Full-history softmax attention rises from 0.43 s at 5 s to 3.67 s at 90 s and runs out of 80 GiB at 95 s. Both L2R layers are duration-invariant: 0.50 s for L2R-GLA and 0.49 s for L2R-GDN. External retrieval also decouples cost from duration but adds overhead: Context as Memory takes 0.59 s at K=5 and 1.13 s at K=20. Geometric retrieval pays to maintain memory: VMem rebuilds its surfel index each step by running point-map and global alignment over all stored frames, increasing from 0.70 s to 3.38 s between 5 s and 95 s chunks, near the softmax-attention curve.

## 5 CONCLUSION

This study establishes a useful analogy between external context memory retrieval mechanisms employed in scene-consistent video world models and the state-update dynamics. Leveraging this relationship, we introduce L2R, which designs camera-guided retrieval, retrieval triggering, and re visibility supervision through a 3D geometry-based target. The resulting module can be integrated into pretrained camera-controllable video world models. Empirical results indicate that L2R substantially improves long-range scene consistency in video world models, eliminating the need for external memory retrievers or explicit 3D condition injection.

## ETHICS STATEMENT

This paper presents research aimed at advancing the field of video world models. Our work may have multifaceted impacts on society. Given the rapidly growing capabilities of video generation technology, we urge against its misuse, such as the creation of misleading videos or video generation for malicious purposes.

## REPRODUCIBILITY STATEMENT

We state that L2R is highly reproducible. Implementation and datasets details on our main experiences are provided in Appendix A. It is anticipated that these descriptions can sufficiently demonstrate the reproducibility of L2R. We plan to open-source the code and weight files after the paper passes peer review.

## REFERENCES

Shuo Chen, Cong Wei, Sun Sun, Ping Nie, Kai Zhou, Ge Zhang, Ming-Hsuan Yang, and Wenhu Chen. Context forcing: Consistent autoregressive video generation with long context. arXiv preprint arXiv:2602.06028, 2026.

JiaKui Hu, Jialun Liu, Liying Yang, Xinliang Zhang, Kaiwen Li, Shuang Zeng, Yuanwei Li, Haibin Huang, Chi Zhang, and Yanye Lu. Geometry-as-context: Modulating explicit 3d in sceneconsistent video generation to geometry context. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4258–4268, 2026a.

JiaKui Hu, Shanshan Zhao, Qing-Guo Chen, Xuerui Qiu, Jialun Liu, Zhao Xu, Weihua Luo, Kaifu Zhang, and Yanye Lu. Omni-view: Unlocking how generation facilitates understanding in unified 3d model based on multiview images. In International Conference on Learning Representations, volume 2026, pp. 133697–133724, 2026b.

Tianyu Huang, Wangguandong Zheng, Tengfei Wang, Yuhao Liu, Zhenwei Wang, Junta Wu, Jie Jiang, Hui Li, Rynson Lau, Wangmeng Zuo, et al. Voyager: Long-range and world-consistent video diffusion for explorable 3d scene generation. ACM Transactions on Graphics (TOG), 44 (6):1–15, 2025.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 21807–21818. IEEE, 2024.

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and Franc¸ois Fleuret. Transformers are RNNs: Fast autoregressive transformers with linear attention. In International Conference on Machine Learning (ICML), 2020.

Wayne King, Zeyue Xue, Yuxuan Bian, Jie Huang, Haoran Li, Yaowei Li, Yaofeng Su, Yuming Li, Haoyu Wang, Shiyi Zhang, Songchun Zhang, Yuwei Niu, Sihan Xu, Junhao Zhuang, Haoyang Huang, and Nan Duan. Echo-memory: A controlled study of memory in action world models. arXiv preprint arXiv:2606.09803, 2026.

Kunyang Li, Mubarak Shah, and Yuzhang Shang. Attend locally, remember linearly: Linear attention as cross-frame memory for autoregressive video diffusion. arXiv preprint arXiv:2605.16579, 2026.

Runjia Li, Philip Torr, Andrea Vedaldi, and Tomas Jakab. Vmem: Consistent interactive video scene generation with surfel-indexed view memory. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 25690–25699, 2025.

Yuqi Pan, Yongqi An, Zheng Li, Yuhong Chou, Ruijie Zhu, Xiaohui Wang, Mingxuan Wang, Jinqiao Wang, and Guoqi Li. Scaling linear attention with sparse state expansion. arXiv preprint arXiv:2507.16577, 2025.

Zhan Peng, Jie Ma, Huiqiang Sun, Chong Gao, Zhijie Xue, Zhiyu Pan, Zhiguo Cao, Jun Liang, and Jing Li. Compression and retrieval: Implicit memory retrieval for video world models. arXiv preprint arXiv:2606.23105, 2026.

Xuerui Qiu, Yutao Cui, Guozhen Zhang, Junzhe Li, JiaKui Hu, Xiao Zhang, Yang Li, Songtao Liu, Miles Yang, Yu Shi, et al. Hydra: Unifying multi-modal generation and understanding via representation-harmonized tokenization. arXiv preprint arXiv:2603.15228, 2026.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 5294–5306, 2025.

Xiaoce Wang, Sifan Zhou, Kaifei Wang, Leli Xu, Xuerui Qiu, Tao He, and Ming Li. Why do dit editors drift? plug-and-play low frequency alignment in vae latent space. arXiv preprint arXiv:2605.08250, 2026a.

Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. pi3: Permutation-equivariant visual geometry learning. In International Conference on Learning Representations, volume 2026, pp. 10481–10497, 2026b.

Zun Wang, Han Lin, Jaehong Yoon, Jaemin Cho, Yue Zhang, and Mohit Bansal. Anchorweave: World-consistent video generation with retrieved local spatial memories. In European Conference on Computer Vision, pp. 203–224. Springer, 2026c.

Zhengxuan Wei, Xu Guo, Xinghui Li, Xunzhi Xiang, Min Wei, Yiran Zhu, Qiulin Wang, Xintao Wang, Pengfei Wan, Xiangwang Hou, et al. Geometry-aware implicit memory for video world models. arXiv preprint arXiv:2606.02436, 2026.

Tong Wu, Shuai Yang, Ryan Po, Yinghao Xu, Ziwei Liu, Dahua Lin, and Gordon Wetzstein. Video world models with long-term spatial memory. Advances in Neural Information Processing Systems, 38:49371–49393, 2026.

Zeqi Xiao, LAN Yushi, Yifan Zhou, Wenqi Ouyang, Shuai Yang, Yanhong Zeng, and Xingang Pan. Worldmem: Long-term consistent world simulation with memory. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Jiacong Xu, Hanwen Jiang, Zhixin Shu, Kalyan Sunkavalli, Vishal M Patel, and Yiqun Mei. Wonder: Video world model done better. arXiv preprint arXiv:2607.26037, 2026a.

Tian-Xing Xu, Zi-Xuan Wang, Guangyuan Wang, Li Hu, Zhongyi Zhang, Peng Zhang, Bang Zhang, and Song-Hai Zhang. Ucm: Unified modeling of camera control and memory with time-aware positional encoding warping for world models. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pp. 1–12, 2026b.

Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, et al. Longlive: Real-time interactive long video generation. arXiv preprint arXiv:2509.22622, 2025a.

Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, and Yoon Kim. Gated linear attention transformers with hardware-efficient training. In International Conference on Machine Learning (ICML), 2024.

Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving mamba2 with delta rule. In International Conference on Learning Representations (ICLR), 2025b.

Yuxue Yang, Lue Fan, Ziqi Shi, Junran Peng, Feng Wang, and Zhaoxiang Zhang. Neoverse: Enhancing 4d world model with in-the-wild monocular videos. arXiv preprint arXiv:2601.00393, 2026.

Hidir Yesiltepe, Tuna Meral, Adil Kaan Akan, Kaan Oktay, and Pinar Yanardag. Infinity-rope: Action-controllable infinite video generation emerges from autoregressive self-rollout. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 40256– 40265, 2026.

Jung Yi, Wooseok Jang, Paul Hyunbin Cho, Jisu Nam, Heeji Yoon, and Seungryong Kim. Deep forcing: Training-free long video generation with deep sink and participative compression. arXiv preprint arXiv:2512.05081, 2025.

Jiwen Yu, Jianhong Qin, Haoran Chen, Xintao Zhang, Ruixiang Yang, Pengfei Wan, Di Zhang, and Xihui Liu. Context as memory: Scene-consistent interactive long video generation with memory retrieval. arXiv preprint arXiv:2506.03141, 2025a.

Jiwen Yu, Jianxiong Gao, Jianhong Bai, Yiran Qin, Kaiyi Huang, Quande Liu, Xintao Wang, Pengfei Wan, Kun Gai, and Xihui Liu. Memlearner: Learning to query context memory for video world models. arXiv preprint arXiv:2606.31734, 2026a.

Mark Yu, Wenbo Hu, Jinbo Xing, and Ying Shan. Trajectorycrafter: Redirecting camera trajectory for monocular videos via diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 100–111, 2025b.

Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. Viewcrafter: Taming video diffusion models for highfidelity novel view synthesis. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025c.

Wei Yu, Runjia Qian, Yumeng Li, Liquan Wang, Songheng Yin, Dennis Anthony, Yang Ye, Yidi Li, Weiwei Wan, Animesh Garg, et al. Mosaicmem: Hybrid spatial memory for controllable video world models. arXiv preprint arXiv:2603.17117, 2026b.

Yifei Yu, Xiaoshan Wu, Xinting Hu, Tao Hu, Yangtian Sun, Xiaoyang Lyu, Bo Wang, Lin Ma, Yuewen Ma, Zhongrui Wang, et al. Videossm: Autoregressive long video generation with hybrid state-space memory. arXiv preprint arXiv:2512.04519, 2025d.

Guozhen Zhang, Xuerui Qiu, Yutao Cui, Tianhui Song, Changlin Li, Junzhe Li, Tao Huang, Xiao Zhang, Yang Li, Jianbing Wu, et al. Hydra-x: Native unified multimodal models with holistic visual tokenizers. arXiv preprint arXiv:2606.13289, 2026.

Lvmin Zhang and Maneesh Agrawala. Packing input frame context in next-frame prediction model for video generation. arXiv e-prints, pp. arXiv–2504, 2025.

Min Zhao, Guande He, Yixiao Chen, Hongzhou Zhu, Chongxuan Li, and Jun Zhu. Riflex: A free lunch for length extrapolation in video diffusion transformers. arXiv preprint arXiv:2502.15894, 2025.

Guangcong Zheng, Teng Li, Xianpan Zhou, and Xi Li. Realcam-vid: High-resolution video dataset with dynamic scenes and metric-scale camera movements. arXiv preprint arXiv:2504.08212, 2025.

Sixiao Zheng, Minghao Yin, Wenbo Hu, Xiaoyu Li, Ying Shan, and Yanwei Fu. Versecrafter: Dynamic realistic video world model with 4d geometric control. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 40277–40290, 2026.

Haoyi Zhu, Haozhe Liu, Yuyang Zhao, Tian Ye, Junsong Chen, Jincheng Yu, Tong He, Song Han, and Enze Xie. Sana-wm: Efficient minute-scale world modeling with hybrid linear diffusion transformer. arXiv preprint arXiv:2605.15178, 2026.

## A IMPLEMENTATION DETAILS

## A.1 MORE EFFICIENT ANALYSIS

Parameters. The L2R layer reuses the query, key, value, and output projections of the self-attention it replaces, so the new parameters are projection layers to obtain the retrieval operator $\gamma _ { t }$ and the retrieval trigger r<sub>t</sub>. When using Wan2.1 as backbone, one softmax self-attention holds 9.45 M parameters and L2R-GLA holds 11.20 M, of which 9.45 M are inherited. Replacing the first three blocks of four blocks therefore adds 5.25 M parameters.

FLOPs. For one 5 s chunk of Wan2.1 (N=18,480, D=1536), a softmax self-attention layer costs 2.45 TFLOPs per forward pass, whereas L2R-GLA costs 0.49 TFLOPs, about 5× fewer. L2R-GDN has the same linear-in-N complexity, with its additional delta correction introducing only a constantfactor overhead over L2R-GLA. Replacing the first three blocks of four blocks therefore reduces the total cost of the 30-block DiT from 111 to 95 TFLOPs per forward pass (14%). Unlike softmax attention, whose cost grows quadratically with sequence length, both L2R-GLA and L2R-GDN scale linearly with the context length, making the computational advantage more pronounced for longer contexts.

## A.2 RE-VISIT BENCHMARK AND EVALUATION ON REALESTATE10K AND DL3DV

We construct Revisit-Bench, a controlled benchmark for evaluating memory and loop-closure consistency in video world models. The benchmark is constructed from real camera trajectories and video frames from RealEstate10K and DL3DV-10K. All benchmark generation is deterministic with a fixed random seed of 42.

## A.2.1 OVERVIEW

A video world model should preserve sufficiently long-range scene information to maintain consistency when the camera moves away from an observed region and subsequently revisits it. In particular, when the camera returns to its initial viewpoint, the generated frame should be consistent with the initial observation. Revisit-Bench quantifies this capability through a controlled trajectory construction protocol.

Each benchmark sequence contains 81 frames, consisting of one reference frame and 80 generated target frames. The camera trajectory is derived from real relative camera motion and transformed into a temporal palindrome, such that the camera exactly returns to its initial viewpoint at a prede fined revisit time. Under the static-scene assumption, every target viewpoint has a corresponding ground-truth frame from the original video, allowing direct pixel-level evaluation.

We construct two trajectory splits using the same set of 300 scenes, with different revisit patterns:

• palindrome far: the folding point is 40, and the camera revisits the initial viewpoint at t = 80. The camera traverses the entire trajectory before returning, resulting in the largest memory span.

• palindrome near: the folding point is 24, and the camera revisits the initial viewpoint at t = 48. After revisiting, the camera continues moving in the original forward direction for 32 additional steps, preventing degenerate solutions in which the model simply remains at the revisited viewpoint.

Each split contains 300 scenes, including 150 scenes from RealEstate10K and 150 from DL3DV-10K. Therefore, each evaluated method produces 600 sequences in total across the two splits.

## A.2.2 DATA SOURCES

The benchmark is constructed from RealCam-Vid (Zheng et al., 2025). We use video clips from RealEstate10K and DL3DV-10K.

Each entry in RealCam-Vid train.npz contains the following fields:

• dataset source: either RealEstate10K or DL3DV-10K;

• video path: the video path relative to DATA ROOT;

• camera extrinsics: a sequence of world-to-camera (w2c) extrinsic matrices of shape (F, 4, 4);

• camera intrinsics: normalized camera intrinsics $( f _ { x } , f _ { y } , c _ { x } , c _ { y } )$ , where the focal lengths and principal-point coordinates are normalized by the source image width and height;

• align factor: a per-window scale factor used to normalize the translation magnitude, consistent with the training data loader;

• short caption and long caption: textual descriptions of the scene.

The benchmark is sampled from these data while using different trajectory constructions. Thus, the benchmark is designed to evaluate revisit consistency.

## A.2.3 FILTERING

We randomly permute the source records using a fixed seed of 42 and scan them sequentially. For each source dataset, we select 150 scenes subject to the following criteria:

1. Video-level deduplication. At most one window is selected from each source video.

2. Valid camera poses. All extrinsic matrices and relative poses must contain finite values, and the required matrix inversions must be numerically stable.

3. Motion filtering. Motion statistics are computed from the relative camera poses after applying the same alignment scale used during training. The median per-frame translation step must lie in [0.02, 0.8], removing nearly static clips and extreme trajectories, and the maximum single-frame rotation must not exceed 10<sup>◦</sup>.

## A.2.4 CAMERA POSE PROCESSING

We process camera poses using the same coordinate transformation as the training data loader. Given the sampled source extrinsics $\mathbf { \bar { W } } _ { t } \in \mathbb { R } ^ { 4 \times 4 }$ , which follow the OpenCV world-to-camera convention, we first convert them to camera-to-world coordinates and apply the per-window alignment factor:

$$
\mathbf { C } _ { t } = \mathbf { W } _ { t } ^ { - 1 } , \qquad \mathbf { C } _ { t } [ : 3 , 3 ] \gets \mathbf { C } _ { t } [ : 3 , 3 ] \cdot s ,\tag{7}
$$

where s denotes align factor. We then convert the poses back to world-to-camera coordinates and express them relative to the first frame:

$$
\mathbf { W } _ { t } ^ { \mathrm { r e l } } = \mathbf { W } _ { t } \left( \mathbf { W } _ { 0 } \right) ^ { - 1 } , \qquad \mathbf { C } _ { t } ^ { \mathrm { r e l } } = \left( \mathbf { W } _ { t } ^ { \mathrm { r e l } } \right) ^ { - 1 } .\tag{8}
$$

Consequently, $\mathbf { C } _ { 0 } ^ { \mathrm { r e l } } = \mathbf { I }$

The resulting poses use the OpenCV camera-to-world convention, where the camera coordinate axes are x-right, y-down, and z-forward. The median translation step is approximately 0.48 for the training clips and 0.17 for the benchmark sequences.

For intrinsics, we recover source-pixel camera intrinsics from the normalized intrinsics provided by RealCam-Vid. Given normalized parameters $( f _ { x } , f _ { y } , c _ { x } , c _ { y } )$ and source resolution (w, h), the intrinsic matrix is

$$
\mathbf { K } = \left[ \begin{array} { c c c } { f _ { x } w } & { 0 } & { c _ { x } w } \\ { 0 } & { f _ { y } h } & { c _ { y } h } \\ { 0 } & { 0 } & { 1 } \end{array} \right] .\tag{9}
$$

We use $( w , h ) = ( 1 2 8 0 , 7 4 0 )$ for RealEstate10K and $( w , h ) = ( 9 6 0 , 5 8 0 )$ for DL3DV-10K. The same intrinsic matrix is used for all 81 frames within each sequence and is stored as an (81, 3, 3) array.

## A.2.5 PALINDROME TRAJECTORY CONSTRUCTION

Let $t \in \{ 0 , \ldots , 8 0 \}$ denote the frame index, let f denote the folding point, and let ${ \bf C } _ { t } ^ { \mathrm { r e l } }$ denote the real relative camera poses obtained above. We construct the camera trajectory as

$$
\mathbf { P } ( t ) = { \left\{ \begin{array} { l l } { \mathbf { C } _ { \operatorname* { m i n } ( t , 2 f - t ) } ^ { \mathrm { r e l } } , } & { t \leq 2 f , } \\ { \mathbf { C } _ { t - 2 f } ^ { \mathrm { r e l } } , } & { t > 2 f \quad ( { \mathrm { n e a r ~ s p l i t ~ o n l y } } ) . } \end{array} \right. }\tag{10}
$$

The first part of the trajectory therefore mirrors the original real camera motion directly rather than interpolating poses or manually reversing motion directions. Every pose in the palindrome is consequently an actual viewpoint present in the source trajectory.

$\mathbf { A } \mathbf { t } t = 2 f$ , the camera pose is exactly the identity:

$$
\mathbf { P } ( 2 f ) = \mathbf { I } .\tag{11}
$$

This property is explicitly verified during benchmark construction with an absolute tolerance of $1 0 ^ { - 9 }$

For palindrome far, $f = 4 0$ , yielding the trajectory

$$
0  4 0  8 0 ,
$$

where the camera returns to the initial viewpoint at $t = 8 0$ . There is no post-revisit segment.

For palindrome near, $f = 2 4 ,$ , yielding

$$
0  2 4  4 8 ,
$$

followed by

$$
t = 4 9 , \ldots , 8 0 \quad \mapsto \quad \mathbf { C } _ { 1 } ^ { \mathrm { r e l } } , \ldots , \mathbf { C } _ { 3 2 } ^ { \mathrm { r e l } } .
$$

Thus, after returning to the initial viewpoint at $t \ = \ 4 8 ,$ , the camera continues along the original forward direction for 32 additional steps. This post-revisit segment prevents a trivial solution in which the generated trajectory remains stationary after loop closure.

## A.2.6 GROUND-TRUTH FRAMES AND STATIC-SCENE ASSUMPTION

We assume a static scene, following the standard setting for loop-closure evaluation. Under this assumption, the ground-truth image corresponding to each generated viewpoint is the source frame associated with that viewpoint. We therefore define

$$
{ \mathrm { g t . f r a m e - i n d i c e s } } [ t ] = { \left\{ \begin{array} { l l } { { \mathrm { m i n } } ( t , 2 f - t ) , } & { t \leq 2 f , } \\ { - ( t - 2 f ) , } & { t > 2 f \quad { \mathrm { ( n e a r ~ s p l i t ~ o n l y ) } } . } \end{array} \right. }\tag{12}
$$

The negative indices in the near split identify the post-revisit overshoot segment and indicate that the absolute value should be used to retrieve the corresponding source frame. In particular, frames t = 0 and t = 2f correspond to exactly the same source viewpoint. Hence, the ground-truth frame at the revisit point is identical to the input reference image.

## A.2.7 TEXT PROMPTS

The text prompt for each sequence is taken from the short caption field of RealCam-Vid. If the short caption is unavailable, we fall back to long caption; if both captions are missing, we use the fixed prompt “A slow camera motion through a real scene.” The resulting prompts contain approximately 10–25 words, with an average length of 18.2 words.

## A.2.8 STATISTICS

Each split contains 300 scenes, consisting of 150 RealEstate10K scenes and 150 DL3DV-10K scenes. Since both splits use the same set of source scenes, each evaluated method produces 600 generated sequences in total: 300 for the far split and 300 for the near split.

For the original trajectories before palindrome construction, the median per-frame translation step, measured in the aligned coordinate scale, has a mean of 0.408 and a median of 0.171, with values ranging from 0.040 to 0.783. The maximum per-frame translation step has a mean of 0.39, while the maximum single-frame rotation has a mean of 2.19<sup>◦</sup>. The latter is bounded by the 10<sup>◦</sup> filtering threshold. For comparison, the median translation step of the training clips is approximately 0.48.

## A.2.9 EVALUATION PROTOCOL

We evaluate generated videos using six groups of metrics following Li et al. (2025); Huang et al.   
(2024); Wang et al. (2026a); Hu et al. (2026b).

Revisit consistency. The primary metric measures the agreement between the generated revisit frame $t _ { r } = 2 f$ and the input reference image. We report PSNR and SSIM, with LPIPS optionally reported as an additional perceptual metric. For uint8 images, PSNR is computed as

$$
\mathrm { P S N R } = 1 0 \log _ { 1 0 } \left( \frac { 2 5 5 ^ { 2 } } { \mathrm { M S E } } \right) .\tag{13}
$$

SSIM is computed on grayscale images after averaging the three RGB channels, using data range = 255.

Camera accuracy. We additionally follow the VMem (Li et al., 2025) evaluation protocol for camera trajectory accuracy. Camera poses are estimated from the generated video using Pi3X and compared with the ground-truth camera poses after applying the same alignment procedure. Both trajectories are first expressed relative to their first frame:

$$
\mathbf { P } ^ { \prime } ( t ) = \mathbf { P } ( 0 ) ^ { - 1 } \mathbf { P } ( t ) .\tag{14}
$$

This removes the arbitrary global origin and orientation associated with the estimated trajectory.

Because the estimated camera trajectory contains an unknown global similarity scale, translation is normalized by the distance of the farthest frame:

$$
\mathbf { t } \gets \frac { \mathbf { t } } { \operatorname* { m a x } _ { i } \| \mathbf { t } _ { i } \| _ { 2 } } .\tag{15}
$$

No Umeyama alignment is applied. We then report the mean translation error

$$
d _ { t } = \lVert \mathbf { t } _ { \mathrm { g t } } - \mathbf { t } _ { \mathrm { g e n } } \rVert _ { 2 }\tag{16}
$$

over $t = 1 , \dots , T - 1$ , together with the rotational geodesic error

$$
d _ { R } = \operatorname { a r c c o s } \left( \frac { \operatorname { t r } \left( \mathbf { R } _ { \mathrm { g e n } } \mathbf { R } _ { \mathrm { g t } } ^ { \mathsf { T } } \right) - 1 } { 2 } \right) ,\tag{17}
$$

reported in degrees.

VBench. We further report six VBench Huang et al. (2024) dimensions: Subject Consistency (DINO), Background Consistency (CLIP), Motion Smoothness (AMT-S), Temporal Flickering (frame-difference MAE), Aesthetic Quality (CLIP with a linear head), and Imaging Quality (MUSIQ). The first 81 frames of each generated video are used, and the prompt recorded in the corresponding manifest entry is provided as the text input.

## A.3 TRAINING SETTINGS

Memory capacity. Following Echo-Memory (King et al., 2026), we train L2R-GLA and L2R-GDN on 8 A100 GPUs. The hyper-parameters are as follows.

Video quality. Although we use two base models, we consult the same training hyperparameters for them to train our own model.

## B EXTENSION TO OTHER STATE UPDATE RULES

L2R defines the retrieval condition $r _ { t }$ and the state-access gate $\gamma _ { t }$ , while the underlying state-space mechanism determines how the retrieved state is updated. In implementation, following Yang et al. (2024; 2025b); Pan et al. (2025); Qiu et al. (2026); Zhang et al. (2026), the vector-valued retrieval gate $\gamma _ { t }$ is converted to a diagonal matrix.

<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Backbone</td><td>Wan2.1</td></tr><tr><td>Resolution</td><td>352 × 640</td></tr><tr><td>Chunk length  $T$ </td><td>81 frames</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $5 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Per-device batch / grad accumulation</td><td>1/1</td></tr><tr><td>GPUs</td><td>8 (A100-80G)</td></tr><tr><td>Total steps</td><td>5k</td></tr><tr><td>Timestep shift</td><td>15</td></tr><tr><td>Spike rejection threshold</td><td>15.0</td></tr><tr><td>Target-frame-only supervision</td><td>enabled</td></tr><tr><td>Flow noise shift</td><td>1.0</td></tr><tr><td>Relative-RT action encoding</td><td>enabled</td></tr><tr><td colspan="2"></td></tr><tr><td>Setting</td><td>Value</td></tr><tr><td>Backbone</td><td>Wan2.2 / SANA-WM</td></tr><tr><td>Resolution</td><td>704 × 1280</td></tr><tr><td>Chunk length T</td><td>81 frames</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $8 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Per-device batch / grad accumulation</td><td>2/1</td></tr><tr><td>GPUs</td><td>64 (H20-96G)</td></tr><tr><td>Total steps</td><td>5k</td></tr><tr><td>Timestep shift</td><td>9.98</td></tr><tr><td>Spike rejection threshold</td><td>1.0</td></tr><tr><td>Target-frame-only supervision</td><td>enabled</td></tr><tr><td>Flow noise shift</td><td>1.0</td></tr><tr><td>Relative-RT action encoding</td><td>enabled</td></tr></table>

$$
\begin{array} { r } { \boldsymbol { S } _ { t } = \left\{ \begin{array} { l l } { \gamma _ { t } \ \boldsymbol { S } _ { t - 1 } + v _ { t } \boldsymbol { k } _ { t } ^ { \top } , } & { \mathrm { I f } \ \boldsymbol { r } _ { t } = 1 , } \\ { \boldsymbol { S } _ { t - 1 } + v _ { t } \boldsymbol { k } _ { t } ^ { \top } , } & { \mathrm { I f } \ \boldsymbol { r } _ { t } = 0 . } \end{array} \right. } \end{array}\tag{18}
$$

For GDN, the same retrieval formulation is combined with its delta correction:

$$
\begin{array} { r } { \boldsymbol { S } _ { t } = \left\{ \begin{array} { l l } { \gamma _ { t } \left( S _ { t - 1 } - \beta _ { t } S _ { t - 1 } k _ { t } k _ { t } ^ { \top } \right) + \beta _ { t } v _ { t } k _ { t } ^ { \top } , } & { \mathrm { I f ~ } \boldsymbol { r } _ { t } = 1 , } \\ { S _ { t - 1 } + \beta _ { t } v _ { t } k _ { t } ^ { \top } , } & { \mathrm { I f ~ } \boldsymbol { r } _ { t } = 0 . } \end{array} \right. } \end{array}\tag{19}
$$

Here, $\gamma _ { t }$ controls which historical information is accessed, while $\beta _ { t }$ controls the subtractive delta update and corresponding write strength in GDN.

## C FAILURE MODE

Figure 5 shows loop-closure rollouts on the same scenes for the two retrieval baselines and for L2R-GLA with a 7:1 memory-layer ratio. With only four memory layers, the recurrent state has to summarize the whole trajectory at few depths of the network, and the softmax layers that remain no longer provide a high-bandwidth fallback for the content the state has overwritten: the return frames lose the observed structure and the loop is not closed, in line with the drop of the 7:1 rows in Figure 4(a), whereas the 1:1 and 3:1 ratios close the loop (Section 4.4).

## D FUTURE WORK

Future research could focus on the issue of memory retrieval in dynamic scenes. Optimizing retrieval triggers for dynamic scenes using more accurate geometric or external supervision represents a viable path forward. We encourage the community to build upon this work to explore more comprehensive internal memory systems suitable for video world models.

![](images/b62d70688c1f73f0207a200bfe0a5f4bc5cf5ef1e34710b79e90f5430c2f5a5c.jpg)  
Figure 5: Failure mode at a high memory-layer ratio. Loop closure on two scenes with memorylayer ratios of 3:1, 1:1 and 7:1. With too few memory layers the rollout loses the observed structure on the return.

## E AI ASSISTANCE

We used AI to polish the grammar of this paper. AI was not used to assist with the core ideas or the experimental design.