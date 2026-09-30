# HelixWorld: A Real-time Interactive Audio-Visual World Model

Lei Ke<sup>1,2\*</sup>, Jiahao Pan<sup>1\*</sup>, Zeyue Tian<sup>1,2\*†</sup>, Jiaming Wang<sup>2</sup>, Haoyuan Huang<sup>2</sup>, Kam Man Wu<sup>1,2</sup> Pengjun Fang<sup>1,2</sup>, Hongyu Liu<sup>1</sup>, Chenyang Qi<sup>1</sup>, Lin Wang<sup>2</sup>, Ruibin Yuan<sup>1</sup> Weijia Chen<sup>2</sup>, Fangneng Zhan<sup>1</sup>, Qifeng Chen<sup>1</sup>, Wei Xue<sup>1†</sup>, Yike Guo<sup>1</sup>

<sup>1</sup>The Hong Kong University of Science and Technology <sup>2</sup>Noiz AI

World simulation is inherently multisensory, demanding synchronized visual and acoustic dynamics in real time. Yet prevailing interactive world models remain strictly silent, focusing exclusively on visual rendering and control while overlooking the acoustic dimension. We present HelixWorld, a real-time interactive audio-visual world model where visual scenes and camera-grounded spatial stereo sound co-evolve natively under user interaction. We curate a high-fidelity spatial audio-visual dataset with true stereo acoustics and metric camera poses, upon which we pre-train a bidirectional teacher conditioned on 6-DoF camera trajectories and user actions. To enable low-latency causal interaction, we distill the teacher into a few-step streaming student via an online trajectory distillation loss, sustaining drift-free joint audio-visual rollouts at 24 FPS on a single GPU. Furthermore, we formalize spatial-acoustic consistency and introduce HelixBench to evaluate whether synthesized sound fields faithfully track dynamic viewpoint motion. Extensive experiments demonstrate that HelixWorld matches state-of-the-art silent world models in visual fidelity and responsiveness, while significantly surpassing existing baselines in camera-aligned spatial-acoustic immersion.

Project Page: https://helixworld.org/ GitHub: https://github.com/NoizAI/HelixWorld

![](images/e80df205d26f4f27d6687a6de2a08ff659440f2ee58a4f5314d34e0af1c8a3a1.jpg)  
Figure 1. HelixWorld. Real-time interaction across diverse worlds with synchronized video and camera-aligned spatial audio.

## 1 Introduction

Interactive world models [1, 3, 7, 12, 23, 37, 40, 53] provide an effective paradigm for physical simulation and virtual environments, allowing users to control generated rollouts in real time via keyboard actions or camera trajectories. Beyond visual synthesis, physical interactions naturally produce sound that conveys critical cues about surface materials, contact forces, and off-screen events. To provide immersive simulation, an interactive world model should synthesize visual scenes and spatially grounded stereo audio responsive to user inputs. However, prevailing world models focus almost exclusively on visual streams, leaving the acoustic dimension largely unexplored. Despite recent efforts to incorporate audio into world models [52], achieving fine-grained spatial-acoustic alignment under dynamic camera movement remains an open challenge due to the absence of high-quality spatial audio paired with camera trajectories, severe error accumulation during long-horizon streaming, and instability of causal model training.

Natively coupling interactive visual scenes with spatial audio introduces fundamental challenges that straightforward adaptations of existing models fail to resolve. Cascaded video-to-audio models [6, 39] lack access to camera trajectories and user actions, leaving sound oblivious to ego-motion, while their sequential latency precludes real-time responsiveness. Meanwhile, bidirectional audio-visual foundation models [11, 13] lack external control interfaces. Moreover, naively converting them into causal autoregressive rollouts leads to catastrophic multimodal drift, where historical error accumulation rapidly degrades both visual stability and acoustic fidelity over extended sequences.

To address these challenges, we present HelixWorld, a real-time interactive audio-visual world model with cameraaligned spatial sound, trained in two stages (see Sec. 3 for details). First, bidirectional teacher training establishes action controllability and spatial-acoustic fidelity. We inject continuous 6-DoF camera trajectories into visual attention via PRoPE [24] and discrete user actions into diffusion timesteps via AdaLN-Zero [32]. Cross-modal attention subsequently binds visual geometry with acoustic features, naturally transferring camera ego-motion into realistic stereo panning. Second, causal distillation converts this bidirectional teacher into a few-step causal student. To mitigate compounding drift under self-forcing [18], we introduce an online trajectory distillation loss that supervises student velocities with teacher Probability Flow ODE targets directly from intermediate rollout states. Combined with long-horizon streaming fine-tuning, HelixWorld substantially reduces exposure bias and sustains synchronized audio-visual generation at 24 FPS on a single NVIDIA H800 GPU.

Beyond model architecture, the successful training of such an interactive system requires large-scale data coupling visual dynamics with high-quality spatial acoustics. However, existing world model datasets are largely silent; even when audio is present in existing video corpora, it is heavily contaminated by post-production background music, voiceover narration, or pseudo-stereo tracks that provide no authentic spatial cues. To bridge this training data gap, we construct a scalable, coarse-to-fine curation pipeline structured by computational cost (Sec. 2). The pipeline deploys inter-channel energy probes to eliminate pseudo-stereo audio at ingestion, utilizes multimodal LLMs to purge non-world footage and non-diegetic audio, and automatically recovers frame-accurate metric camera poses paired with decoupled three-track captions (visual scene, foreground acoustic events, and background ambiance), establishing a high-fidelity spatial audio-visual training corpus.

Finally, evaluating an interactive audio-visual world model requires assessing whether synthesized audio physically aligns with dynamic camera motion. Existing world model benchmarks [51] focus exclusively on visual dynamics while neglecting the acoustic stream, whereas conventional audio-visual benchmarks [4] evaluate only short, static clips without camera movement. To close this gap, we introduce HelixBench, the first benchmark dedicated to evaluating interactive audio-visual world models (Sec. 4). Spanning 1,015 human-verified cases across diverse environments and interaction modes, HelixBench evaluates models across five key dimensions: temporal synchronization, audio-visual semantics, text-audio alignment, acoustic dynamics, and spatial-acoustic consistency. In particular, it formalizes spatial acoustic consistency by correlating on-screen sound emitter positions with stereo audio panning, directly quantifying whether the synthesized sound field physically rotates with camera ego-motion.

In summary, our main contributions are:

• HelixWorld Model: We propose an interactive audio-visual world model that natively unifies continuous 6-DoF camera trajectories and discrete actions, achieving synchronized and camera-responsive rollouts in real time.

• Causal Distillation Recipe: We develop a principled distillation framework coupling causal initialization, online trajectory distillation, and streaming fine-tuning, which eliminates compounding rollout drift and sustains real-time

24 FPS streaming.

• Multimodal Data Pipeline: A systematic multi-stage curation workflow that purges pseudo-stereo and nondiegetic tracks while annotating metric camera extrinsics and decoupled three-track captions, establishing a high-quality spatial corpus.

• HelixBench: A benchmark of 1,015 human-verified interactive cases formalizing spatial-acoustic consistency alongside temporal synchrony and multimodal semantics for interactive audio-visual evaluation.

## 2 Dataset Construction

Training interactive audio-visual world models requires continuous dynamics, authentic stereo acoustics, and synchronized control. Yet existing datasets largely lack authentic spatial audio, while in-the-wild video suffers from shot cuts, overlays, and non-diegetic sound. We address this with an automated pipeline (Fig. 2) that filters multimodal artifacts while annotating metric camera poses, discrete actions, and decoupled captions to power bidirectional pre-training and causal streaming distillation. More details are provided in Appendix C.

## 2.1 Data Sources

We assemble footage across three complementary sources (Table 1). Real-world video captures first-person exploration across diverse indoor and outdoor environments. Gamefootage records first- and third-person gameplay with native audio, logging ground-truth camera trajectories and control telemetry. Open-source corpora incorporate audio-equipped video subsets [3, 25], subjected to identical stereo and semantic curation.

Table 1. Corpus by source.
<table><tr><td>Source</td><td>Hours</td><td>Share</td></tr><tr><td>Real-world</td><td>1.8k</td><td>43.9%</td></tr><tr><td>Game</td><td>1.4k</td><td>34.1%</td></tr><tr><td>Open-source</td><td>0.9k</td><td>22.0%</td></tr><tr><td>Total</td><td>4.1k</td><td>100%</td></tr></table>

## 2.2 Curation Pipeline

Raw footage cannot be fed directly into training due to noise and artifacts across audio-visual streams. We design a four-stage pipeline (Fig. 2) ordered by cost, pruning corrupted data before invoking heavy models.

Stage 1: Heuristicfiltering. Raw videos are split into continuous shots via scene transition detection [2], discarding fragments under five seconds. Acoustically, lightweight probes use inter-channel correlation and energy differences to remove mono downmixes and duplicated channels. Visually, optical character recognition (OCR) [54] and heuristic detectors prune overlays, banners, and watermarks. Perceptual quality scoring [20] and motion screening remove severely degraded or frozen clips while preserving static views with physical motion.

Stage 2: Clip standardization. Continuous shots are partitioned into fixed-length clips matching the backbone training window. Video and audio are sliced synchronously without crossing shot boundaries, ensuring temporal alignment.

Stage 3: Semanticfiltering. To resolve defects requiring high-level scene understanding, an audio-visual multimodal large language model (MLLM) [33] evaluates clips under a structured perception schema. Visually, it purges nonembodied content (e.g., picture-in-picture, talking heads, occluding overlays) and resolves text ambiguous to OCR. Acoustically, it strips non-diegetic post-production tracks (background music, voiceover, synthetic effects). Crucially, sounds are judged by causal origin rather than visibility, preserving off-screen acoustics and natural speech.

Stage 4: Camera and captioning. Accepted clips enter two parallel annotation branches. For web and real-world footage, the camera branch estimates per-frame poses via VGGT-Ω [42] and recovers metric scale with Depth Anything V3 [26], falling back to Metric3D v2 [16] when necessary; game footage uses engine-logged ground truth. After discarding tracking failures and physically implausible trajectories, surviving motions are discretized into an 81-class action vocabulary (Sec. 3.1; benchmarks in Appendix C). Concurrently, the captioning branch synthesizes three decoupled text tracks via independent forward passes: video-only, audio-only, and joint audio-visual descriptions that bind visible sound emitters to their acoustics. A clip is retained only when both branches succeed.

![](images/a94c1aa28c4a0de8e9b7bf994b485d4d973ea5fa0bd0538ca28ec63aa0d418a9.jpg)  
Figure 2. Data pipeline. Raw video and audio pass four cost-ordered stages; surviving clips fork into parallel camera recovery and decoupled captioning branches to yield the trainable corpus.

In total, this multi-stage pipeline systematically purges visual artifacts and acoustic contamination across the 4.1k raw hours. Through successive screening and pose verification, 3.0k hours (amounting to 2.1M synchronized clips) survive to form the final trainable corpus, achieving an overall yield of 73.1% (Table 2). The resulting corpus provides a camera-conditioned audio-visual foundation with authentic stereo sound fields and per-frame action annotations, directly powering bidirectional pre-training and causal streaming distillation (Sec. 3).

Table 2. Curation yield.
<table><tr><td>Stage</td><td>Hours</td><td>Pass Rate</td></tr><tr><td>Raw footage</td><td>4.1k</td><td>100%</td></tr><tr><td>Heuristic filtering</td><td>3.4k</td><td>81.9%</td></tr><tr><td>Semantic filtering</td><td>3.2k</td><td>95.5%</td></tr><tr><td>Camera annotation</td><td>3.0k</td><td>93.5%</td></tr><tr><td>Trainable corpus</td><td>3.0k</td><td>73.1%</td></tr></table>

## 3 Method

An interactive audio-visual world model requires responsive action following (steering visual dynamics and spatial sound) and causal interactivity (real-time generation). To this end, HelixWorld follows a two-stage training pipeline (Figs. 3 and 4): Sec. 3.1 trains a bidirectional teacher with progressive camera and action conditioning, while Sec. 3.2 distills it into a few-step causal student.

## 3.1 Bidirectional Teacher Training

We train the bidirectional model on our corpus (Sec. 2) by progressively injecting control signals. We employ both continuous camera trajectories and discrete user actions as control inputs. Both controls enter the video branch and propagate to the audio branch through cross-modal attention.

Condition Injection. Following Sun et al. [37], we condition the video branch on both continuous camera trajectories and discrete actions. Continuous 6-DoF camera poses are injected via Projective Relative Positional Encoding (PRoPE) [24], transforming visual attention queries and keys by their respective camera projection matrices $P _ { t } =$ $\mathrm { d i a g } ( K _ { f , t } , 1 ) W _ { t } \in \mathbb { R } ^ { 4 \times 4 }$ . Discrete actions are represented by an 81-class vocabulary over a $\mathbf { \varepsilon } _ { ! 9 \times 9 }$ grid of translation and view-rotation velocity bins [23, 37]. For each frame $t ,$ the action index $a _ { t }$ is embedded and added to the diffusion noise level embedding to modulate the transformer backbone via adaptive layer normalization (AdaLN-Zero) [32]:

![](images/3c0e6d9220087df1c3d29bc507500640910737a24ad4399fb6dd3e7e6f37fefc.jpg)  
Figure 3. Progressive condition injection. Three-stage training: optimize control modules with the backbone frozen, unfreeze the video branch, and jointly train the full audio-visual backbone.

$$
h _ { \mathrm { c o n d } , t } = \mathrm { E m b } _ { \sigma } ( \sigma ) + \mathrm { M L P } \big ( \phi ( a _ { t } ) \big ) .\tag{1}
$$

Both controls enter the video branch and propagate to audio via cross-modal attention.

Progressive training. Directly tuning the entire backbone on unaligned controls destabilizes pretrained representations. We therefore introduce conditioning in three progressive steps. First, we freeze the backbone and optimize only the control modules (the action MLP and PRoPE projections). Second, we unfreeze the video branch to adapt visual features to control inputs. Finally, we unfreeze the full audio-visual backbone for joint flow-matching optimization:

$$
\mathcal { L } _ { \mathrm { F M } } = \mathcal { L } _ { \mathrm { F M } } ^ { V } + \lambda _ { A } \mathcal { L } _ { \mathrm { F M } } ^ { A } .\tag{2}
$$

## 3.2 Causal Distillation

To enable real-time generation, we distill the bidirectional teacher into a few-step causal student. Addressing the visual drift of standard self-forcing (DMD2) [18, 48], we propose an online trajectory distillation loss that anchors student rollouts to the teacher’s probability flow, complemented by long-horizon streaming tuning.

Causal initialization. We first synchronize video latents and co-temporal audio tokens into temporal blocks of length ∆t. Attention within each block is fully bidirectional to maintain audio-visual interactions, while cross-block attention is strictly causal via a sliding Key-Value (KV) cache. Based on this structure, we adapt the teacher using the joint flow-matching objective $\mathcal { L } _ { \mathrm { F M } } \left( \mathrm { E q . } 2 \right)$ conditioned on ground-truth history firstly. To provide a stable starting checkpoint for few-step sampling, we then initialize the student by regressing its single-step predictions onto the Probability Flow ODE endpoints of the causal teacher [56]:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { i n i t } } ( \theta ) = \mathbb { E } _ { \pmb { x } _ { \sigma } , \sigma , c } \left\| \hat { \pmb { x } } _ { 0 } ^ { \theta } ( \pmb { x } _ { \sigma } , \sigma , c ) - \hat { \pmb { x } } _ { 0 } ^ { \mathrm { t e a c h e r } } ( \pmb { x } _ { \sigma } , \pmb { c } ) \right\| _ { 2 } ^ { 2 } , } \end{array}\tag{3}
$$

where $\scriptstyle { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } }$ is the joint audio-visual latent, and c denotes historical context and conditions.

Self-forcing with online trajectory distillation. Following causal initialization, the student unrolls blocks autoregressively via self-forcing [18] with its KV cache. Standard self-forcing uses distribution matching $( \mathcal { L } _ { \mathrm { D M D } } )$ [48], updating the student via score differences between a frozen teacher and a learned fake-score model tracking rollout distributions. However, distribution matching alone reduces sample diversity [44] and causes long-rollout drift (Fig. 5; Table 8). To preserve diversity and stabilize rollouts, we introduce an online trajectory distillation loss $( \mathcal { L } _ { \mathrm { t r a j } } )$ alongside $\mathcal { L } _ { \mathrm { D M D } }$

![](images/dc7e66456657c9409974ccd68d5936569ba663d7cf45abfc309f8f79db074a0f.jpg)  
Figure 4. Causal distillation pipeline. Top: Self-forcing updates stochastically alternate between online trajectory distillation and distribution matching. Bottom: Long-horizon tuning extends rollouts segment by segment via a sliding KV cache with initial sink frames and detached history.

At step k with noise $\sigma _ { k }$ and state $\pmb { x } _ { k } = ( \pmb { x } _ { k } ^ { V } , \pmb { x } _ { k } ^ { A } )$ , we integrate the frozen teacher’s PF-ODE to $\sigma = 0$ , yielding endpoint $\tilde { \pmb { x } } _ { 0 } = ( \tilde { \pmb { x } } _ { 0 } ^ { V } , \tilde { \pmb { x } } _ { 0 } ^ { A } )$ . The student velocity ${ \pmb v } _ { \theta } \overset { \cdot } { = } ( \dot { \pmb v } _ { \theta } ^ { V } , { \pmb v } _ { \theta } ^ { A } )$ is supervised against the implied teacher flow:

$$
\mathcal { L } _ { \mathrm { t r a j } } ( \theta ) = \sum _ { m \in \{ V , A \} } \lambda _ { m } \left. \pmb { v } _ { \theta } ^ { m } ( \pmb { x } _ { k } , \sigma _ { k } ) - \mathrm { s g } \left( \frac { \pmb { x } _ { k } ^ { m } - \tilde { \pmb { x } } _ { 0 } ^ { m } } { \sigma _ { k } } \right) \right. _ { 2 } ^ { 2 } ,\tag{4}
$$

where $\operatorname { s g } ( \cdot )$ denotes the stop-gradient operator. Gradients are evaluated strictly at step k, with past KV caches and rollout steps detached. During training, each update stochastically samples $\mathcal { L } _ { \mathrm { t r a j } }$ with probability p or $\mathcal { L } _ { \mathrm { D M D } }$ with $1 - p .$ This routing avoids gradient conflict between objectives, combining distribution matching for sample sharpness with trajectory guidance for rollout stability (see Appendix B).

Long-horizon streaming tuning. Long rollouts can still suffer from visual degradation and acoustic fading. To mitigate this drift, we incorporate streaming long tuning [46]. As shown in Fig. 4, the student unrolls sequences segment-wise with a sliding KV cache, preserving sink frames as anchors while evicting distant history. Gradients from $\mathcal { L } _ { \mathrm { D M D } }$ are computed only on the active segment with history detached, bounding memory.

## 4 HelixBench: A Benchmark for Audio-Visual World Models

Existing world-model benchmarks evaluate only silent video generation [37, 51], while audio-visual benchmarks are restricted to static, non-interactive clips [4]. In an interactive world model, the generated audio must sound realistic, synchronize with visual content, match text prompts, and adjust its stereo balance as the camera moves. To evaluate these properties during interactive rollouts, we introduce HelixBench.

Benchmark composition and metrics. HelixBench comprises 1,015 human-verified test clips across six scene domains and four subsets: Navigation, Perspective, Event, and Subject. Each case provides a stereo reference clip, an initial frame, an audio-visual prompt, and continuous camera poses. Audio quality is measured by KL divergence [22] and Fréchet Audio Distance (FAD) [21] on VGGish embeddings [15], while multimodal alignment is evaluated via ImageBind [10] and CLAP [45]. The Event subset measures temporal synchronization via DeSync [19], while Perspective evaluates whether stereo audio pans with on-screen sources (Spatial, Appendix D).

Table 3. Visual quality and action following on the WBench navigation split [51]. Audio indicates native audio synthesis; RTF is measured on a single NVIDIA H800.
<table><tr><td>Method</td><td>Audio</td><td>RTF↓</td><td>Average ↑</td><td>Quality ↑</td><td>Setting ↑</td><td>Interaction ↑</td><td>Consistency ↑</td><td>Physical ↑</td></tr><tr><td>Alaya-EVOKE-Turbo</td><td>x</td><td>4.29</td><td>82.0</td><td>81.9</td><td>82.1</td><td>83.9</td><td>88.1</td><td>74.0</td></tr><tr><td>EchoWM</td><td>√</td><td>1.98</td><td>81.0</td><td>81.1</td><td>77.5</td><td>87.9</td><td>88.3</td><td>70.1</td></tr><tr><td>Zing-0.5</td><td>x</td><td>2.81</td><td>81.0</td><td>80.6</td><td>77.8</td><td>84.2</td><td>88.5</td><td>73.8</td></tr><tr><td>LingBot-World v2</td><td>x</td><td>5.21</td><td>79.4</td><td>81.8</td><td>76.8</td><td>82.8</td><td>86.5</td><td>69.1</td></tr><tr><td>HY-World 1.5</td><td>x</td><td>12.33</td><td>78.1</td><td>78.1</td><td>72.2</td><td>86.8</td><td>86.9</td><td>66.3</td></tr><tr><td>Lyra 2.0</td><td>x</td><td>30.87</td><td>76.4</td><td>77.1</td><td>73.2</td><td>85.6</td><td>79.3</td><td>66.7</td></tr><tr><td>SANA-WM</td><td>x</td><td>0.79</td><td>76.0</td><td>79.3</td><td>76.1</td><td>82.2</td><td>80.7</td><td>61.9</td></tr><tr><td>DreamX-World</td><td>x</td><td>1.04</td><td>75.0</td><td>77.5</td><td>80.8</td><td>78.6</td><td>74.9</td><td>63.3</td></tr><tr><td>Matrix-Game 3</td><td>x</td><td>0.58</td><td>71.3</td><td>75.5</td><td>63.6</td><td>83.6</td><td>74.5</td><td>59.3</td></tr><tr><td>LTX-2.3 (base)</td><td>√</td><td>14.61</td><td>74.2</td><td>77.1</td><td>85.2</td><td>66.4</td><td>77.2</td><td>64.9</td></tr><tr><td>HelixWorld (ours)</td><td>√</td><td>0.77</td><td>79.9</td><td>79.2</td><td>76.5</td><td>86.4</td><td>86.4</td><td>70.9</td></tr></table>

Table 4. Audio-visual quality on HelixBench. HelixWorld+X re-dubs our video using V2A model X; joint generation achieves superior spatial acoustic alignment.
<table><tr><td>Model</td><td>KL↓</td><td>FAD↓</td><td>DeSync (s) ↓</td><td>IB↑</td><td>CLAP↑</td><td>Spatial ↑</td></tr><tr><td>LTX-2.3 (base)</td><td>1.8040</td><td>7.6354</td><td>0.4042</td><td>0.2777</td><td>0.2577</td><td>0.9826</td></tr><tr><td>EchoWM</td><td>1.9530</td><td>7.7472</td><td>0.6745</td><td>0.2036</td><td>0.2352</td><td>12.6476</td></tr><tr><td>HelixWorld + AudioX</td><td>2.0190</td><td>3.9166</td><td>1.1764</td><td>0.2569</td><td>0.3094</td><td>-5.7392</td></tr><tr><td>HelixWorld + ThinkSound</td><td>2.2335</td><td>6.1876</td><td>0.4236</td><td>0.2079</td><td>0.2888</td><td>14.8467</td></tr><tr><td>HelixWorld + PrismAudio</td><td>2.2427</td><td>6.3873</td><td>0.4582</td><td>0.2152</td><td>0.2716</td><td>-9.5072</td></tr><tr><td>HelixWorld (bidirectional teacher)</td><td>1.5414</td><td>3.0775</td><td>0.4988</td><td>0.2700</td><td>0.2676</td><td>33.2428</td></tr><tr><td>HelixWorld (causal student)</td><td>1.3934</td><td>2.3872</td><td>0.5867</td><td>0.2987</td><td>0.3016</td><td>41.7583</td></tr></table>

## 5 Experiments

## 5.1 Experimental Setup

Implementation and training. HelixWorld builds on open-source LTX-2.3 and follows the two-stage training procedure in Sec. 3. The full implementation details are provided in Appendix B.

Baselines. To evaluate visual quality, audio fidelity, and audio-visual alignment, we compare HelixWorld against strong baselines [5, 8, 9, 35, 37, 43, 50, 52, 55]. Because EchoWM is the only world model supporting audio-visual generation, we also construct cascaded dubbing baselines using video-to-audio (V2A) models [28, 29, 39]. Specifically, we mute HelixWorld outputs and re-dub them with these models, keeping visual content fixed to isolate audio performance (details in Appendix D).

## 5.2 Main Results

Benchmark performance. For video quality, we benchmark on the WBench navigation split [51] (Table 3). HelixWorld outperforms most open-source silent baselines across visual quality and control responsiveness, demonstrating faithful action following and physical plausibility. To assess audio quality and multimodal coherence, we evaluate on HelixBench (Sec. 4) across audio fidelity, temporal sync, semantic alignment, and spatial consistency (Table 4; Appendix D). HelixWorld surpasses both EchoWM and cascaded dubbing pipelines. While video-to-audio models achieve reasonable text alignment, they fail to capture emitter locations, resulting in negative spatial scores. EchoWM retains coarse spatial cues but trails in acoustic quality and semantic binding. These results demonstrate that HelixWorld effectively grounds spatial acoustics in interactive visual dynamics, achieving physically coherent audio-visual generation.

Inference efficiency. On a single NVIDIA H800 GPU, HelixWorld achieves a steady-state RTF of 0.77 at 768×512 and 24 fps, including video and audio decoding. This enables real-time responses to user control (details in Appendix D.7).

## 5.3 Ablation Studies

Progressive condition injection. To test whether staged conditioning prevents representation collapse (Sec. 3.1), we compare progressive against full fine-tuning. Evaluating ViPE-recovered trajectories [17], progressive fine-tuning reduces both rotation and translation errors (Table 5). This confirms that gradually unfreezing branches is essential for learning geometric control while preserving pretrained visual representations.

Multimodal caption composition. To test whether explicit sound-source binding improves joint generation, we compare three caption formats (Sec. 2): video-only (V), video-plus-audio (V+A), and three-track $( \mathsf { V } + \mathsf { A } { + } \mathsf { A } \mathsf { V } )$ . Cross-evaluating across prompt formats, three-track captions consistently yield the highest ImageBind similarity (Table 6). This confirms that richer audio-visual descriptions provide direct correspondence cues, strengthening cross-modal consistency.

Table 5. Ablation on camera control.
<table><tr><td>Strategy</td><td>Rotation ↓</td><td>Translation ↓</td></tr><tr><td>Full fine-tuning</td><td>0.1237</td><td>0.0957</td></tr><tr><td>Progressive fine-tuning</td><td>0.1144</td><td>0.0823</td></tr></table>

Table 6. Ablation on caption format.
<table><tr><td>Train \ Infer</td><td>V</td><td>V+A</td><td>V+A+AV</td></tr><tr><td>V</td><td>0.2372</td><td>0.2346</td><td>0.2314</td></tr><tr><td>V+A</td><td>0.2366</td><td>0.2315</td><td>0.2419</td></tr><tr><td>V+A+AV</td><td>0.2316</td><td>0.2323</td><td>0.2462</td></tr></table>

Self-forcing with online trajectory distillation. To test whether online trajectory distillation $( \mathcal { L } _ { \mathrm { t r a j } } )$ curbs rollout drift while preserving sample diversity (Sec. 3.2), we compare self-forcing training with and without $\mathcal { L } _ { \mathrm { t r a j } }$ on top of DMD. As shown in Table 7, incorporating $\mathcal { L } _ { \mathrm { t r a j } }$ improves overall WBench performance, with marked gains in consistency and physical plausibility. In extended 30-s rollouts, Figure 5 demonstrates that trajectory guidance mitigates visual drift, maintaining cloud structures and scene landmarks with far fewer artifacts. It also boosts feature diversity on DINOv3 and CLIP over DMD alone (Table 8; Appendix E). These findings validate that anchoring student rollouts to the teacher’s probability flow stabilizes autoregressive generation without mode collapse.

Long-horizon streaming tuning. To evaluate whether long-horizon tuning sustains generation quality over extended sequences (Sec. 3.2), we apply it to both self-forcing variants. As shown in Table 7, long-horizon tuning delivers consistent, orthogonal gains across both setups, improving overall performance over baselines. When combined with trajectory distillation, it achieves best overall score. This confirms that truncated streaming supervision effectively complements block-level trajectory guidance, enabling stable and interactive rollouts over extended horizons.

## 5.4 Qualitative Analysis and User Study

Qualitative results. Figure 6 compares HelixWorld with LingBot-World v2, HY-WorldPlay 1.5, and EchoWM on same scenes with shared first-frame and control inputs. In the bamboo scene, our model retains the forest corridor as the view rises into the canopy and returns to the path. In the garage, the red car and nearby pillars remain recognizable through the right-to-left movement reversal. Additional audio–visual comparisons are provided in Appendix G.

User study. We conduct a blind pairwise user study across audio-visual and muted settings (Figure 7; Appendix F).   
With audio, HelixWorld is preferred over all baselines by a clear margin, winning the majority of votes in every matchup.   
In muted evaluations, it outperforms Alaya, HY-WorldPlay, and SANA, ties with LingBot v2, and trails only EchoWM.

Table 7. Ablation on self-forcing and long-horizon tuning. Trajectory distillation $( \mathcal { L } _ { \mathrm { t r a j } } )$ and long-horizon tuning offer complementary gains, jointly reaching the best overall WBench score.
<table><tr><td> $\mathcal { L } _ { \mathrm { t r a j } }$ </td><td>Long-Horizon</td><td>Average ↑</td><td>Quality ↑</td><td>Setting ↑</td><td>Interaction ↑</td><td>Consistency ↑</td><td>Physical ↑</td></tr><tr><td>x</td><td>x</td><td>77.9</td><td>77.0</td><td>75.3</td><td>85.0</td><td>85.3</td><td>66.9</td></tr><tr><td>√</td><td>x</td><td>78.7</td><td>77.8</td><td>74.1</td><td>86.1</td><td>86.6</td><td>68.9</td></tr><tr><td>x</td><td>√</td><td>78.4</td><td>79.0</td><td>77.5</td><td>86.5</td><td>83.7</td><td>65.1</td></tr><tr><td>√</td><td>√</td><td>79.1</td><td>79.3</td><td>77.1</td><td>86.8</td><td>84.6</td><td>67.7</td></tr></table>

Figure 5. Visual quality in 30-s rollouts. Self-forcing with $\mathcal { L } _ { \mathrm { t r a j } }$ (bottom) mitigates visual drift and preserves cloud structures far better than standard DMD (top).  
1 s  
10 s  
20 s  
![](images/2995a7ded1f72ba0db3805f1c0ad319caccf6b52760e0baad0bb15bfd0d7bbcb.jpg)  
30 s

Table 8. Visual diversity comparison. Incorporating online trajectory distillation $( \mathcal { L } _ { \mathrm { t r a j } } )$ into self-forcing improves both DINOv3 and CLIP feature diversity over DMD alone.
<table><tr><td>Self-forcing</td><td>DINOv3↑</td><td>CLIP↑</td></tr><tr><td>Without  $\mathcal { L } _ { \mathrm { t r a j } }$ </td><td>0.1089</td><td>0.0718</td></tr><tr><td>With  $\mathcal { L } _ { \mathrm { t r a j } }$ </td><td>0.1326</td><td>0.0815</td></tr><tr><td>Gain</td><td>+21.7%</td><td>+13.5%</td></tr></table>

## 6 Related Work

Interactive world models. Interactive world models [12] simulate future observations under diverse controls, including latent codes [1, 31], keyboard actions [3, 7, 23, 40, 53], 6-DoF camera trajectories [14, 24], and hybrid inputs [37]. Yet, prevailing models remain silent; while EchoWM [52] incorporates audio, it lacks camera-conditioned spatial acoustics. HelixWorld conditions a joint audio-visual backbone on continuous camera trajectories and discrete actions, steering viewpoints alongside responsive spatial acoustics.

Audio-visual generation. Cascaded systems dub pre-rendered video via video-to-audio synthesis [6, 38], but post-hoc dubbing lacks camera extrinsics or control intents, and compounding latency prevents real-time interaction. In contrast, joint models [11, 13] co-denoise video and audio tokens in a shared transformer for cross-modal synchrony. Yet, they remain passive and bidirectional, operating on fixed-length clips without control interfaces or streaming inference. We bring interactive control and causal streaming to joint models.

Causal distillation. Deploying diffusion models interactively requires converting bidirectional generation into causal streaming. Recent video distillation schemes mitigate exposure bias through distribution matching (DMD) [48, 49], on-policy KV rollouts (Self-Forcing) [18], ODE regression [56], and long-horizon streaming tuning [46]. However, these techniques target video alone. When extended to joint audio-visual diffusion, distribution matching suffers from mode collapse, causing color degradation and acoustic dropouts over extended rollouts. We stabilize self-forcing with an online trajectory distillation loss, anchoring the student’s joint velocity field against teacher probability flows.

Audio-visual evaluation. World-model benchmarks such as WBench [51] assess action-conditioned video dynamics but ignore audio entirely. Conversely, audio-visual benchmarks [4] measure semantic correspondence (ImageBind [10], CLAP [45]) and onset synchrony (Synchformer [19], AV-Align [47]) on short, fixed-viewpoint clips. While these metrics evaluate what sounds occur and when, they cannot test where sounds originate relative to a moving listener under viewpoint motion. HelixBench addresses this gap, formalizing spatial-acoustic consistency alongside temporal, semantic, and dynamic metrics for interactive audio-visual evaluation.

Input  
Input  
![](images/b8fce2ec416177968cf6993424d7a4e525ff2f1afc25a865e7c8e995608660eb.jpg)  
... a narrow earthen path through a dense bamboo forest. Tall green stalks form a natural corridor, with dappled sunlight and an interlocking canopy overhead ...

15 s  
![](images/d134c45c3a63987d4794df0104de4ba2b7a11aa362687fb81a0b9175504e2612.jpg)  
... First-person view of an underground parking garage, with numbered concrete columns, directional arrows on the floor, fluorescent lights and parked vehicles ...

Figure 6. Visual comparison under interactive control. HelixWorld retains scene structure during navigation (left) and motion reversal (right). Key overlays indicate controls.  
![](images/81045870f3b17f6bd11d0eee14e24a8de530f2aef047b4aaa23086a52b1ee4c9.jpg)  
(a) With audio

![](images/867ebb1e18e4c70694667238dd8159f573f9ddd58e0031b82d2a1bbad793f55a.jpg)  
(b) Muted video  
Figure 7. User-study preferences. Purple indicates preference for HelixWorld, gray a tie, and beige preference for the baseline across full audio-visual rollouts (a) and muted video dynamics (b). Dashed vertical lines mark 50%.

## 7 Conclusion

In this work, we present HelixWorld, a real-time interactive audio-visual world model where visual scenes and spatial stereo soundscapes co-evolve natively under dynamic user control. Coupled with a compute-prioritized data curation pipeline, our causal distillation framework suppresses compounding rollout drift to sustain synchronized 24 FPS streaming on a single GPU. Finally, we formalize spatial-acoustic consistency as a foundational criterion separating interactive world models from passive generation, and introduce HelixBench to systematically evaluate it.

## References

[1] Jake Bruce, Michael D. Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, Yusuf Aytar, Sarah Bechtle, Feryal M. P. Behbahani, Stephanie C. Y. Chan, Nicolas Heess, Lucy Gonzalez, Simon Osindero, Sherjil Ozair, Scott E. Reed, Jingwei Zhang, Konrad Zolna, Jeff Clune, Nando de Freitas, Satinder Singh, and Tim Rocktäschel. Genie: Generative interactive environments. In Ruslan Salakhutdinov, Zico Kolter, Katherine A. Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024, volume 235 of Proceedings of Machine Learning Research, pages 4603–4623. PMLR / OpenReview.net, 2024. URL https://proceedings.mlr.press/v235/bruce24a.html.

[2] Brandon Castellano. PySceneDetect, n.d. URL https://github.com/Breakthrough/PySceneDetect.

[3] Haoxuan Che, Xuanhua He, Quande Liu, Cheng Jin, and Hao Chen. Gamegen-x: Interactive open-world game video generation. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. URL https://openreview.net/forum?id=8VG8tpPZhe.

[4] Honglie Chen, Weidi Xie, Andrea Vedaldi, and Andrew Zisserman. Vggsound: A large-scale audio-visual dataset. In 2020 IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2020, Barcelona, Spain, May 4-8, 2020, pages 721–725. IEEE, 2020. doi: 10.1109/ICASSP40776.2020.9053174. URL https://doi.org/10.1109/ICASSP40776.2020.9053174.

[5] Mingyang Chen, Shengdong Chen, Xiaoxiao Fu, Bosheng Gong, Haoyuan Guo, Bowen Li, Jiawen Li, Kejun Li, Tianpeng Li, Yin Liu, et al. Zing-0.5: Toward playable worlds with real-time joint action and text control. arXiv preprint arXiv:2609.17909, 2026.

[6] Ho Kei Cheng, Masato Ishii, Akio Hayakawa, Takashi Shibuya, Alexander G. Schwing, and Yuki Mitsufuji. Mmaudio: Taming multimodal joint training for high-quality video-to-audio synthesis. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025, pages 28901–28911. Computer Vision Foundation / IEEE, 2025. doi: 10.1109/CVPR52734.2025.02691. URL https: //openaccess.thecvf.com/content/CVPR2025/html/Cheng\_MMAudio\_Taming\_Multimodal\_ Joint\_Training\_for\_High-Quality\_Video-to-Audio\_Synthesis\_CVPR\_2025\_paper.html.

[7] Decart, Julian Quevedo, Quinn McIntyre, Spruce Campbell, Xinlei Chen, and Robert Wachen. Oasis: A universe in a transformer, 2024. URL https://oasis-model.github.io/.

[8] DreamX Team, Rui Chen, Xiangxiang Chu, Geng Li, Jifan Li, Qingfeng Shi, Datao Tang, Jing Tang, Jun Wang, and Pengfei Zhang. Dreamx-phi 1.0: Action-conditioned video world model for robotic manipulation. CoRR, abs/2608.13489, 2026. doi: 10.48550/ARXIV.2608.13489. URL https://doi.org/10.48550/arXiv.2608. 13489.

[9] Zelin Gao, Qiuyu Wang, Jiapeng Zhu, Jingye Chen, Zichen Liu, Qingyan Bai, Jiahao Wang, Yufeng Yuan, Hanlin Wang, Yichong Lu, Ka Leong Cheng, Haojie Zhang, Jian Gao, Tianrui Feng, Yuzheng Liu, Yao Yao, Yinghao Xu, Xing Zhu, Yujun Shen, and Hao Ouyang. Infinite worlds with versatile interactions. CoRR, abs/2607.07534, 2026. doi: 10.48550/ARXIV.2607.07534. URL https://doi.org/10.48550/arXiv.2607.07534.

[10] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. ImageBind: One embedding space to bind them all. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[11] Google DeepMind. Veo 3: State-of-the-art video generation with audio. https://deepmind.google/models/ veo/, 2025.

[12] David Ha and Jürgen Schmidhuber. World models. arXiv preprint arXiv:1803.10122, 2(3):440, 2018.

[13] Yoav HaCohen, Benny Brazowski, Nisan Chiprut, Yaki Bitterman, Andrew Kvochko, Avishai Berkowitz, Daniel Shalem, Daphna Lifschitz, Dudu Moshe, Eitan Porat, Eitan Richardson, Guy Shiran, Itay Chachy, Jonathan Chetboun, Michael Finkelson, Michael Kupchick, Nir Zabari, Nitzan Guetta, Noa Kotler, Ofir Bibi, Ori Gordon, Poriya Panet, Roi Benita, Shahar Armon, Victor Kulikov, Yaron Inger, Yonatan Shiftan, Zeev Melumian, and

Zeev Farbman. LTX-2: efficient joint audio-visual foundation model. CoRR, abs/2601.03233, 2026. doi: 10.48550/ARXIV.2601.03233. URL https://doi.org/10.48550/arXiv.2601.03233.

[14] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for video diffusion models. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. URL https://openreview. net/forum?id=Z4evOUYrk7.

[15] Shawn Hershey, Sourish Chaudhuri, Daniel P. W. Ellis, Jort F. Gemmeke, Aren Jansen, R. Channing Moore, Manoj Plakal, Devin Platt, Rif A. Saurous, Bryan Seybold, Malcolm Slaney, Ron J. Weiss, and Kevin W. Wilson. CNN architectures for large-scale audio classification. In 2017 IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2017, New Orleans, LA, USA, March 5-9, 2017, pages 131–135. IEEE, 2017. doi: 10.1109/ICASSP.2017.7952132. URL https://doi.org/10.1109/ICASSP.2017.7952132.

[16] Mu Hu, Wei Yin, Chi Zhang, Zhipeng Cai, Xiaoxiao Long, Hao Chen, Kaixuan Wang, Gang Yu, Chunhua Shen, and Shaojie Shen. Metric3d v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(12):10579–10596, 2024.

[17] Jiahui Huang, Qunjie Zhou, Hesam Rabeti, Aleksandr Korovko, Huan Ling, Xuanchi Ren, Tianchang Shen, Jun Gao, Dmitry Slepichev, Chen-Hsuan Lin, Jiawei Ren, Kevin Xie, Joydeep Biswas, Laura Leal-Taixé, and Sanja Fidler. Vipe: Video pose engine for 3d geometric perception. CoRR, abs/2508.10934, 2025. doi: 10.48550/ARXIV. 2508.10934. URL https://doi.org/10.48550/arXiv.2508.10934.

[18] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. In Danielle Belgrave, Cheng Zhang, Laura N. Montoya, Hsuan-Tien Lin, Razvan Pascanu, Piotr Koniusz, Marzyeh Ghassemi, Nancy Chen, Iván Vladimir Meza Ruíz, and Arturo Loaiza-Bonilla, editors, Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2025, NeurIPS 2025, San Diego, CA, USA, December 2-7, 2025 / Mexico City, Mexico, November 30 - December 5, 2025, 2025. URL http://papers.nips.cc/paper\_files/paper/2025/ hash/f4823f831af67a3ef15e41a85434422a-Abstract-Conference.html.

[19] Vladimir Iashin, Weidi Xie, Esa Rahtu, and Andrew Zisserman. Synchformer: Efficient synchronization from sparse cues. In IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2024, Seoul, Republic ofKorea, April 14-19, 2024, pages 5325–5329. IEEE, 2024. doi: 10.1109/ICASSP48485.2024.10448489. URL https://doi.org/10.1109/ICASSP48485.2024.10448489.

[20] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. MUSIQ: multi-scale image quality transformer. In 2021 IEEE/CVF International Conference on Computer Vision, ICCV 2021, Montreal, QC, Canada, October 10-17, 2021, pages 5128–5137. IEEE, 2021. doi: 10.1109/ICCV48922.2021.00510. URL https://doi.org/10.1109/ICCV48922.2021.00510.

[21] Kevin Kilgour, Mauricio Zuluaga, Dominik Roblek, and Matthew Sharifi. Fréchet audio distance: A referencefree metric for evaluating music enhancement algorithms. In Gernot Kubin and Zdravko Kacic, editors, 20th Annual Conference of the International Speech Communication Association, Interspeech 2019, Graz, Austria, September 15-19, 2019, pages 2350–2354. ISCA, 2019. doi: 10.21437/INTERSPEECH.2019-2219. URL https://doi.org/10.21437/Interspeech.2019-2219.

[22] Khaled Koutini, Jan Schlüter, Hamid Eghbal-zadeh, and Gerhard Widmer. Efficient training of audio transformers with patchout. In Hanseok Ko and John H. L. Hansen, editors, 23rd Annual Conference ofthe International Speech Communication Association, Interspeech 2022, Incheon, Korea, September 18-22, 2022, pages 2753–2757. ISCA, 2022. doi: 10.21437/INTERSPEECH.2022-227. URL https://doi.org/10.21437/Interspeech.2022-227.

[23] Jiaqi Li, Junshu Tang, Zhiyong Xu, Longhuang Wu, Yuan Zhou, Shuai Shao, Tianbao Yu, Zhiguo Cao, and Qinglin Lu. Hunyuan-gamecraft: High-dynamic interactive game video generation with hybrid history condition. CoRR, abs/2506.17201, 2025. doi: 10.48550/ARXIV.2506.17201. URL https://doi.org/10.48550/arXiv. 2506.17201.

[24] Ruilong Li, Brent Yi, Junchen Liu, Hang Gao, Yi Ma, and Angjoo Kanazawa. Cameras as relative positional encoding. In Danielle Belgrave, Cheng Zhang, Laura N. Montoya, Hsuan-Tien Lin, Razvan Pascanu, Piotr Koniusz, Marzyeh Ghassemi, Nancy Chen, Iván Vladimir Meza Ruíz, and Arturo Loaiza-Bonilla, editors, Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2025, NeurIPS 2025, San Diego, CA, USA, December 2-7, 2025 / Mexico City, Mexico, November 30 - December 5, 2025, 2025. URL http://papers.nips.cc/paper\_files/paper/2025/hash/ 17a7075094632c88cccdd86270ad715b-Abstract-Conference.html.

[25] Zhen Li, Chuanhao Li, Xiaofeng Mao, Shaoheng Lin, Ming Li, Shitian Zhao, Zhaopan Xu, Xinyue Li, Yukang Feng, Jianwen Sun, Zizhen Li, Fanrui Zhang, Jiaxin Ai, Zhixiang Wang, Yuwei Wu, Tong He, Yunde Jia, and Kaipeng Zhang. Sekai: A video dataset towards world exploration. In Danielle Belgrave, Cheng Zhang, Laura N. Montoya, Hsuan-Tien Lin, Razvan Pascanu, Piotr Koniusz, Marzyeh Ghassemi, Nancy Chen, Iván Vladimir Meza Ruíz, and Arturo Loaiza-Bonilla, editors, Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2025, NeurIPS 2025, San Diego, CA, USA, December 2-7, 2025 / Mexico City, Mexico, November 30 - December 5, 2025, 2025. URL http://papers.nips.cc/paper\_files/paper/ 2025/hash/2368d4a3f24122b9f3779aafa261719-Abstract-Datasets\_and\_Benchmarks\_Track.html.

[26] Haotong Lin, Sili Chen, Junhao Liew, Donny Y. Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth anything 3: Recovering the visual space from any views. CoRR, abs/2511.10647, 2025. doi: 10.48550/ ARXIV.2511.10647. URL https://doi.org/10.48550/arXiv.2511.10647.

[27] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, et al. Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22160–22169. IEEE, 2024.

[28] Huadai Liu, Kaicheng Luo, Wen Wang, Qian Chen, Peiwen Sun, Rongjie Huang, Xiangang Li, Jieping Ye, and Wei Xue. Prismaudio: Decomposed chain-of-thoughts and multi-dimensional rewards for video-to-audio generation. CoRR, abs/2511.18833, 2025. doi: 10.48550/ARXIV.2511.18833. URL https://doi.org/10.48550/arXiv. 2511.18833.

[29] Huadai Liu, Jialei Wang, Kaicheng Luo, Wen Wang, Qian Chen, Zhou Zhao, and Wei Xue. Thinksound: Chain-of thought reasoning in multimodal large language models for audio generation and editing. CoRR, abs/2506.21448, 2025. doi: 10.48550/ARXIV.2506.21448. URL https://doi.org/10.48550/arXiv.2506.21448.

[30] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, and Lei Zhang. Grounding DINO: marrying DINO with grounded pre-training for open-set object detection. In Ales Leonardis, Elisa Ricci, Stefan Roth, Olga Russakovsky, Torsten Sattler, and Gül Varol, editors, Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part XLVII, volume 15105 of Lecture Notes in Computer Science, pages 38–55. Springer, 2024. doi: 10.1007/978-3-031-72970-6\_3. URL https://doi.org/10.1007/978-3-031-72970-6\_3.

[31] Jack Parker-Holder, Philip Ball, Jake Bruce, Vibhavari Dasagi, Kristian Holsheimer, Christos Kaplanis, Alexandre Moufarek, Guy Scully, Jeremy Shar, Jimmy Shi, Stephen Spencer, Jessica Yung, Michael Dennis, Sultan Kenjeyev, Shangbang Long, Vlad Mnih, Harris Chan, Maxime Gazeau, Bonnie Li, Fabio Pardo, Luyu Wang, Lei Zhang, Frederic Besse, Tim Harley, Anna Mitenkova, Jane Wang, Jeff Clune, Demis Hassabis, Raia Hadsell, Adrian Bolton, Satinder Singh, and Tim Rocktäschel. Genie 2: A large-scale foundation world model. https://deepmind. google/discover/blog/genie-2-a-large-scale-foundation-world-model/, 2024.

[32] William Peebles and Saining Xie. Scalable diffusion models with transformers. In IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023, pages 4172–4182. IEEE, 2023. doi: 10.1109/ICCV51070.2023.00387. URL https://doi.org/10.1109/ICCV51070.2023.00387.

[33] Qwen Team. Qwen3-omni technical report. CoRR, abs/2509.17765, 2025. doi: 10.48550/ARXIV.2509.17765. URL https://doi.org/10.48550/arXiv.2509.17765.

[34] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

[35] Tianchang Shen, Sherwin Bahmani, Kai He, Sangeetha Grama Srinivasan, Tianshi Cao, Jiawei Ren, Ruilong Li, Zian Wang, Nicholas Sharp, Zan Gojcic, Sanja Fidler, Jiahui Huang, Huan Ling, Jun Gao, and Xuanchi Ren. Lyra 2.0: Explorable generative 3d worlds. CoRR, abs/2604.13036, 2026. doi: 10.48550/ARXIV.2604.13036. URL https://doi.org/10.48550/arXiv.2604.13036.

[36] Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025.

[37] Wenqiang Sun, Haiyu Zhang, Haoyuan Wang, Junta Wu, Zehan Wang, Zhenwei Wang, Yunhong Wang, Jun Zhang, Tengfei Wang, and Chunchao Guo. Worldplay: Towards long-term geometric consistency for realtime interactive world modeling. CoRR, abs/2512.14614, 2025. doi: 10.48550/ARXIV.2512.14614. URL https://doi.org/10.48550/arXiv.2512.14614.

[38] The Movie Gen team. Movie gen: A cast of media foundation models. CoRR, abs/2410.13720, 2024. doi: 10.48550/ARXIV.2410.13720. URL https://doi.org/10.48550/arXiv.2410.13720.

[39] Zeyue Tian, Zhaoyang Liu, Yizhu Jin, Ruibin Yuan, Liumeng Xue, Xu Tan, Qifeng Chen, Wei Xue, and Yike Guo. Audiox: a unified framework for anything-to-audio generation. In International Conference on Learning Representations, volume 2026, pages 110625–110651, 2026.

[40] Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. Diffusion models are real-time game engines. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. URL https://openreview.net/forum?id=P8pqeEkn1H.

[41] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotný. VGGT: visual geometry grounded transformer. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025, pages 5294–5306. Computer Vision Foundation / IEEE, 2025. doi: 10.1109/CVPR52734.2025.00499. URL https://openaccess.thecvf.com/content/CVPR2025/html/ Wang\_VGGT\_Visual\_Geometry\_Grounded\_Transformer\_CVPR\_2025\_paper.html.

[42] Jianyuan Wang, Minghao Chen, Shangzhan Zhang, Nikita Karaev, Johannes Schönberger, Patrick Labatut, Piotr Bojanowski, David Novotny, Andrea Vedaldi, and Christian Rupprecht. VGGT-Ω. arXiv preprint arXiv:2605.15195, 2026.

[43] Zile Wang, Zexiang Liu, Jiaxing Li, Kaichen Huang, Baixin Xu, Fei Kang, Mengyin An, Peiyu Wang, Biao Jiang, Yichen Wei, Yidan Xietian, Jiangbo Pei, Liang Hu, Boyi Jiang, Hua Xue, Zidong Wang, Haofeng Sun, Wei Li, Wanli Ouyang, Xianglong He, Yang Liu, Yangguang Li, and Yahui Zhou. Matrix-game 3.0: Realtime and streaming interactive world model with long-horizon memory. CoRR, abs/2604.08995, 2026. doi: 10.48550/ARXIV.2604.08995. URL https://doi.org/10.48550/arXiv.2604.08995.

[44] Tianhe Wu, Ruibin Li, Lei Zhang, and Kede Ma. Diversity-preserved distribution matching distillation for fast visual synthesis. CoRR, abs/2602.03139, 2026. doi: 10.48550/ARXIV.2602.03139. URL https://doi.org/10. 48550/arXiv.2602.03139.

[45] Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, Taylor Berg-Kirkpatrick, and Shlomo Dubnov. Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation. In IEEE International Conference on Acoustics, Speech and Signal Processing ICASSP 2023, Rhodes Island, Greece, June 4-10, 2023, pages 1–5. IEEE, 2023. doi: 10.1109/ICASSP49357.2023.10095969. URL https://doi.org/10. 1109/ICASSP49357.2023.10095969.

[46] Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, Song Han, and Yukang Chen. Longlive: Real-time interactive long video generation. CoRR, abs/2509.22622, 2025. doi: 10.48550/ARXIV.2509.22622. URL https://doi.org/10.48550/arXiv. 2509.22622.

[47] Guy Yariv, Itai Gat, Sagie Benaim, Lior Wolf, Idan Schwartz, and Yossi Adi. Diverse and aligned audio-tovideo generation via text-to-video model adaptation. In Michael J. Wooldridge, Jennifer G. Dy, and Sriraam Natarajan, editors, Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-Sixth Conference on Innovative Applications ofArtificial Intelligence, IAAI 2024, Fourteenth Symposium on Educational Advances

in Artificial Intelligence, EAAI 2024, February 20-27, 2024, Vancouver, Canada, pages 6639–6647. AAAI Press, 2024. doi: 10.1609/AAAI.V38I7.28486. URL https://doi.org/10.1609/aaai.v38i7.28486.

[48] Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Frédo Durand, and Bill Freeman. Improved distribution matching distillation for fast image synthesis. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors, Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024. URL http://papers.nips.cc/paper\_files/ paper/2024/hash/54dcf25318f9de5a7a01f0a4125c541e-Abstract-Conference.html.

[49] Tianwei Yin, Qiang Zhang, Richard Zhang, William T. Freeman, Frédo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11- 15, 2025, pages 22963–22974. Computer Vision Foundation / IEEE, 2025. doi: 10.1109/CVPR52734. 2025.02138. URL https://openaccess.thecvf.com/content/CVPR2025/html/Yin\_From\_Slow\_ Bidirectional\_to\_Fast\_Autoregressive\_Video\_Difusion\_Models\_CVPR\_2025\_paper.html.

[50] Yuanyang Yin, Gongxuan Wang, Yifan Zhang, Chuanhao Li, Kaipeng Zhang, and Feng Zhao. Alaya-evoke: From linear-scaling supervision to endless world. CoRR, abs/2608.13546, 2026. doi: 10.48550/ARXIV.2608.13546. URL https://doi.org/10.48550/arXiv.2608.13546.

[51] Kaining Ying, Hengrui Hu, Siyu Ren, Jiamu Li, Fengjiao Chen, Ziwen Wang, Xuezhi Cao, Xunliang Cai, and Henghui Ding. Wbench: A comprehensive multi-turn benchmark for interactive video world model evaluation. CoRR, abs/2605.25874, 2026. doi: 10.48550/ARXIV.2605.25874. URL https://doi.org/10.48550/arXiv. 2605.25874.

[52] Songchun Zhang, Yaowei Li, Junhao Zhuang, Weiyang Jin, Haoyu Wang, Xin Lu, Yilang Sun, Shiyi Zhang, Haoran Li, Xiaoxiao Ma, Yuming Li, Yijun Liu, Yaofeng Su, Yanwen Ma, Haoyu Wu, Zihan Su, Yue Ma, Lvmin Zhang, Haoyang Huang, Zeyue Xue, Anyi Rao, and Nan Duan. Echowm: Open and enterable omnimodal world models. CoRR, abs/2608.23189, 2026. doi: 10.48550/ARXIV.2608.23189. URL https://doi.org/10.48550/ arXiv.2608.23189.

[53] Yifan Zhang, Chunli Peng, Boyang Wang, Puyi Wang, Qingcheng Zhu, Fei Kang, Biao Jiang, Zedong Gao, Eric Li, Yang Liu, and Yahui Zhou. Matrix-game: Interactive world foundation model. CoRR, abs/2506.18701, 2025. doi: 10.48550/ARXIV.2506.18701. URL https://doi.org/10.48550/arXiv.2506.18701.

[54] Yubo Zhang, Xueqing Wang, Manhui Lin, Yue Zhang, Penglongyi Deng, Ting Sun, Tingquan Gao, Zelun Zhang, Jiaxuan Liu, Changda Zhou, et al. Pp-ocrv6: From 1.5 m to 34.5 m parameters, surpassing billion-scale vlms on ocr tasks. arXiv preprint arXiv:2606.13108, 2026.

[55] Haoyi Zhu, Haozhe Liu, Yuyang Zhao, Tian Ye, Junsong Chen, Jincheng Yu, Tong He, Song Han, and Enze Xie. SANA-WM: efficient minute-scale world modeling with hybrid linear diffusion transformer. CoRR, abs/2605.15178, 2026. doi: 10.48550/ARXIV.2605.15178. URL https://doi.org/10.48550/arXiv.2605. 15178.

[56] Hongzhou Zhu, Min Zhao, Guande He, Hang Su, Chongxuan Li, and Jun Zhu. Causal forcing: Autoregressive diffusion distillation done right for high-quality real-time interactive video generation. CoRR, abs/2602.02214, 2026. doi: 10.48550/ARXIV.2602.02214. URL https://doi.org/10.48550/arXiv.2602.02214.

## A Architecture and Conditioning Details

Camera conditioning. The camera branch supplements the video self-attention in Sec. 3.1. Intrinsics are normalized to the latent grid, and extrinsics are expressed relative to the first frame, whose relative pose is the identity. The focal-only lift retains focal lengths but omits the principal point. Invalid or missing camera tokens, as well as audio tokens, receive identity transforms. The camera branch shares the base attention’s Q/K/V projections and contributes through a zero-initialized output projection.

Action conditioning. The 81 action classes factor as id = 9 translation + rotation. Translation and rotation each enumerate nine bins in $\{ - , 0 , + \} ^ { 2 }$ , for local (x, z) translation and (yaw, pitch), respectively. Each video latent frame has its own action index. The index is encoded with 256-dimensional sinusoidal features and a Linear–SiLU–Linear MLP with a zero-initialized output, then added to the noise-level embedding. Camera and action controls enter the video branch and reach audio through cross-modal attention.

## B Training and Streaming Inference Details

## B.1 Temporal Blocks and First-Frame Conditioning

A short self-forcing window contains 16 video latent frames and 127 audio tokens. The video is partitioned into four blocks of four latent frames; the corresponding audio blocks contain 27, 33, 33, and 34 tokens, assigned by their timestamps. The conditioning image occupies the first latent of the first block. It participates in joint audio–visual attention at noise level zero, is restored after each denoising and re-noising operation, and is excluded from the generatedtoken loss. The remaining 15 video latents and all audio tokens are generated.

Attention is bidirectional within a block and causal across blocks. We detach historical latents and committed KV tensors and retain gradients within the current block. Camera and action sequences are aligned on the source timeline before temporal blocks are selected.

## B.2 Causal Initialization

Teacher forcing. We adapt the bidirectional model to block-causal attention using clean ground-truth history and the joint audio–visual flow-matching objective. Short clips supply four consecutive blocks. For long sources, the initialization recipe packs the first block with a later contiguous three-block window and supervises the final target block. This exposes the model to later source positions within a bounded context.

ODE initialization. The student regresses onto precomputed endpoints of the teacher-forced causal model (Eq. 3). The teacher uses 50 dense solver steps, CFG 3, and no STG. Input states are sampled from the four nonzero student knots {1.0, 0.9, 0.7, 0.4}. The image-conditioned initialization uses the first frame on every example and optimizes the joint video/audio endpoint MSE with a constant learning rate of $2 \times 1 0 ^ { - 6 }$

## B.3 Self-Forcing Optimization

The generator starts from the ODE-initialized causal model. The frozen real-score teacher and the trainable fake-score model both start from the bidirectional model after progressive condition injection. Generator and fake-score adaptation use LoRA with the shared settings in Table 9.

Rollout and gradient path. The student denoises each block with the schedule [1.0, 0.9, 0.7, 0.4, 0], drawing independent video and audio noise for each re-noising transition. A clean, detached KV commit follows each completed block, and later blocks append to this history. A short training window contains four blocks, all retained in the history context. The loss covers generated tokens across the window, with gradients retained at one selected denoising step per block.

Table 9. Self-forcing hyperparameters. Settings shared by DMD-only self-forcing and self-forcing with online trajectory distillation loss.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Generator / fake-score adaptation</td><td>LoRA rank 256, alpha 256, dropout 0</td></tr><tr><td>Optimizer</td><td>AdamW, β1 = 0.9, β2 = 0.999</td></tr><tr><td>Generator / fake learning rate</td><td>10−⁵, constant; no warmup or decay</td></tr><tr><td>Gradient norm clipping</td><td>1.0</td></tr><tr><td>Precision</td><td>BF16 with gradient checkpointing</td></tr><tr><td>Global batch size</td><td>48; one sample per GPU, no accumulation</td></tr><tr><td>Training hardware</td><td>Six nodes, each with eight H200 GPUs</td></tr><tr><td>Fake updates per DMD generator event</td><td>5</td></tr><tr><td>Teacher video / audio CFG</td><td>4/2</td></tr><tr><td>Teacher STG</td><td>Scale 1, transformer block 29</td></tr><tr><td>Teacher guidance rescale / modality scale 0 / 1</td><td></td></tr><tr><td>Student inference CFG / STG</td><td>1/0</td></tr></table>

DMD score evaluation. The bidirectional real and fake models evaluate the same noised student sample at the same physical noise level. For the self-forcing recipe, a sampled raw level u is shifted once as

$$
\sigma = \frac { \gamma u } { 1 + ( \gamma - 1 ) u } , \qquad \gamma = 3 .\tag{5}
$$

Generator-score queries use $u \sim \mathcal { U } ( 0 . 0 2 , 0 . 9 8 )$ ; fake-score training uses $u \sim \mathcal { U } ( 0 , 1 )$ . The resulting σ is shared by video and audio and used consistently for noising, model queries, and clean-sample conversion. Student rollouts use the fixed four-knot schedule above.

Online trajectory distillation. With probability 0.1, a generator event uses Eq. 4 in place of a DMD update; otherwise it follows five fake-score updates and one DMD generator update. For a trajectory event, one student denoising index is shared across the four blocks. The frozen bidirectional teacher integrates from the corresponding noisy state to a detached endpoint on a dense schedule with eight steps per student interval (32 over the full schedule). Gradients update the selected student predictions while conditioning-image tokens and historical KV states remain fixed.

## B.4 Long-Horizon Tuning and Streaming Inference

Long-horizon tuning. We apply long-horizon tuning to self-forcing models trained with and without online trajectory distillation, starting from their respective generators and fake-score models. The student rolls out approximately one minute of synchronized video and audio, advancing through the source in temporal order. In both variants, the DMD loss supervises the newest window, with earlier latents and KV states detached. The bidirectional score models process short windows of 16 video latents and 127 audio tokens. Table 7 compares both variants before and after long-horizon tuning.

Bounded inference history. The inference context contains the fixed first block, the two most recent completed blocks, and the current target block: at most four blocks in total. The first block retains the conditioning-image latent. For example, the fifth target block reads blocks 1, 3, and 4. Temporal keys are cached before RoPE; video and audio positions share a compact local timeline for the retained context.

Cache reconstruction. The reported rebuild evaluations recompute history KV from the selected clean, generated latents before denoising the next target. Each history block attends to itself and earlier retained blocks under the block-causal mask. Reconstruction updates cached keys and values while keeping the generated history latents fixed. The visual quality and diversity comparisons use the same rebuild policy in both arms. Inference jointly generates both modalities with four stochastic denoising steps, CFG 1, and STG 0.

## C Dataset Construction and Curation Details

This section provides comprehensive details on the data curation and annotation pipeline described in Sec. 2, including domain distribution and ethical privacy considerations, signal-level thresholds, MLLM perception schema, camera pose and metric scale recovery, and multimodal caption contracts.

## C.1 Data Sources, Domain Distribution, and Ethical Considerations

We assemble raw footage totaling 4.1 k hours across three complementary domains (Table 10), systematically balancing real-world dynamics, interactive gaming environments, and diverse open-source distributions.

Real-world web video. We collect 1.8 k hours of high-resolution first-person exploration, urban driving, and environmental walkthroughs from publicly available video platforms. To safeguard personal privacy and prevent unauthorized disclosure, we apply automated face de-identification and license-plate obfuscation during pre-processing; any footage centered on identifiable individuals or private personal spaces is systematically purged in Stage 1 and Stage 3.

Game screen recordings. We record 1.4 k hours of high-dynamic first- and third-person navigation across diverse 3D virtual environments and modern game engines, capturing native multi-channel engine audio and synchronized camera trajectories. Direct access to engine telemetry provides noise-free ground-truth camera extrinsics and user control inputs, serving as a reliable geometric anchor for interactive control learning.

Open-source corpora. We incorporate 0.9 k hours from open-domain video corpora with native audio tracks, including Sekai [25] and GameGen-X [3], covering diverse open-world scenes and interactive scenarios. These subsets are subjected to identical stereo and semantic curation pipelines to ensure corpus-wide distribution uniformity.

Table 10. Data sources, domain distribution, and corpus statistics. Pose origin indicates whether camera trajectories are logged directly from engine states or reconstructed offline via geometric estimation. All statistics reflect the frozen corpus.
<table><tr><td>Source Family</td><td>Pose Origin</td><td>Raw Hours</td><td>Share</td><td>Trainable Clips</td></tr><tr><td>Real-world web video</td><td>Estimated (VGGT-Ω + DA3)</td><td>1.8k</td><td>43.9%</td><td>~0.91M</td></tr><tr><td>Game screen recordings</td><td>Engine-logged (ground truth)</td><td>1.4k</td><td>34.1%</td><td>~0.73M</td></tr><tr><td>Open-source subsets</td><td>Mixed (logged / estimated)</td><td>0.9k</td><td>22.0%</td><td>~0.46M</td></tr><tr><td>Total Raw Footage</td><td></td><td>4.1k</td><td>100%</td><td>~2.93M</td></tr><tr><td>Trainable Corpus</td><td>Verified metric poses</td><td>3.0k</td><td>73.1%</td><td>2.10M</td></tr></table>

## C.2 Signal-Level Filtering and Clip Standardization (Stages 1 & 2)

Raw footage exhibits pervasive distribution corruption, including dual-mono downmixes, rapid montage cuts, static title cards, and interface clutter. Stage 1 executes lightweight signal probes to eliminate defective footage prior to compute-intensive neural processing.

Container and format validation. The ingestion probe enforces strict container constraints: video streams must have a spatial resolution of at least 768×512 with horizontal landscape aspect ratio, and frame rates bounded within [23.9, 121] fps (subsequently normalized to 24 fps). The audio stream must contain a valid two-channel stereo track, immediately pruning mono downmixes and missing audio streams. Clips failing container integrity or exhibiting corrupt stream headers are pruned immediately.

True stereo verification. A significant fraction of online videos duplicate a single mono recording across two channels, creating “dual-mono” tracks devoid of spatial acoustic information. We measure inter-channel stereo energy over sliding

20-second analysis windows via normalized root-mean-square difference:

$$
\delta = \frac { \| L - R \| _ { 2 } ^ { 2 } } { \| L \| _ { 2 } ^ { 2 } + \| R \| _ { 2 } ^ { 2 } } ,\tag{6}
$$

where L and R denote the discrete time-domain signals of the left and right channels, respectively. For identical channels, $\delta \equiv 0 .$ , whereas authentic spatial acoustics yield $\delta \in [ 0 . 0 5 , 1 . 0 ]$ . We discard any upload with $\delta < 0 . 0 1$ . Surviving streams are subsequently verified for phase correlation to eliminate artificial out-of-phase stereo widening artifacts.

Shot transition and temporal continuity. We employ PySceneDetect [2] to segment continuous shots, tracking HSV color histogram differences across frames subsampled at 4 fps. A cut boundary is placed whenever the histogram difference exceeds 0.35. Fragmented shots shorter than 5 seconds are dropped to prevent abrupt scene cuts from contaminating the temporal attention window. Furthermore, any video whose mean continuous shot duration falls below 2 seconds (e.g., promotional trailers or rapid montages) is rejected in its entirety.

Static scene discrimination. To eliminate static still-image slideshows and frozen streams while retaining stationary cameras that observe active physical events (e.g., vehicles driving past a fixed camera), we compute the Median Absolute Difference (MAD) between consecutive grayscale frames sampled at 1 fps:

$$
\mathrm { M A D } = \operatorname { m e d i a n } _ { ( x , y ) } \big | I _ { t + 1 } ( x , y ) - I _ { t } ( x , y ) \big | .\tag{7}
$$

Shots with $\mathrm { M A D } < 1 . 5$ (on an 8-bit scale [0, 255]) are rejected. Employing the median rather than the mean prevents localized high-frequency perturbations (such as blinking watermarks, subtitle updates, or compression noise) from falsely validating an otherwise frozen scene.

Clip standardization (Stage 2). Surviving continuous shots are segmented into standardized clips matching the backbone training context: exactly 121 video frames at 24 fps (5.04 s) at 768×512 resolution, coupled with synchronized two-channel 48 kHz stereo audio (241,920 audio samples per clip). Shots are tiled front-to-back without overlap. To maximize data efficiency, whenever a trailing segment exceeds 2.5 s, an additional clip is carved backwards from the shot terminus, raising temporal coverage from 98.4% to 100%. Audio and video streams are sliced synchronously in a single ffmpeg pass, ensuring microsecond-level temporal synchronization by construction.

## C.3 Semantic Filtering with Audio-Visual MLLM (Stage 3)

While low-level heuristic probes eliminate signal defects, they cannot reason about high-level multimodal semantics: pixel differences fail to separate 3D physical environments from 2D screen recordings or talking heads, and audio format checks cannot detect added background music. Stage 3 deploys an audio-visual foundation model (Qwen3-Omni 30B [33]) to evaluate video and audio synchronously under constrained grammar decoding. A clip is retained if and only if it satisfies two physical criteria:

• Visual scene integrity: The clip must depict an authentic 3D physical environment undergoing dynamic physical evolution, strictly excluding non-world content (e.g., talking-head interviews, desktop screencasts, picture-in-picture feeds) and intrusive graphic or subtitle overlays.

• Acoustic authenticity: The clip must feature natural diegetic sound, systematically pruning post-production background music, voiceover narration, and artificial sound effects. Crucially, acoustic filtering evaluates sounds by their physical genesis rather than source visibility: off-screen environmental sounds (e.g., thunder or passing vehicles) are preserved as authentic diegetic acoustics.

Because soundtrack contamination typically spans an entire video, we apply an early termination heuristic: if 10 consecutive clips within an upload are flagged for non-diegetic audio, the remainder of that upload is terminated immediately, saving substantial inference compute.

Table 11. Mean camera estimation errors and runtime across benchmark scenes. Translation L1 and RMSE are normalized relative to reference trajectory RMS radius after Sim(3) alignment; rotation error reflects global orientation alignment. Runtime includes initialization and excludes depth recovery and window stitching. VGGT and VGGT-Omega process 32 evaluation frames; ViPE processes 163–399 frames before subsampling.
<table><tr><td rowspan="2">Method</td><td colspan="2">Translation (%) ↓</td><td rowspan="2">Rotation (°) ↓</td><td rowspan="2">Runtime (s) ↓</td></tr><tr><td>L1</td><td>RMSE</td></tr><tr><td>VGGT [41]</td><td>1.010</td><td>0.760</td><td>0.315</td><td>14.23</td></tr><tr><td>VGGT-Omega [42]</td><td>1.001</td><td>0.768</td><td>0.234</td><td>16.09</td></tr><tr><td>ViPE [17]</td><td>0.763</td><td>0.582</td><td>0.263</td><td>187.32</td></tr></table>

![](images/40287f57d439867e3244396919e3edf2a2b02f0950b098436bd777e8ab1b491b.jpg)  
Figure 8. Single-sample camera trajectory comparison. From left to right: COLMAP reference, VGGT, VGGT-Omega, and ViPE. All trajectories are aligned to the reference coordinate frame via Sim(3). Solid lines denote camera translation paths; schematic frustums indicate camera orientations.

## C.4 Camera Pose Estimation and Metric Scale Recovery (Stage 4)

High-fidelity interactive world modeling requires precise, metric-scaled 6-DoF camera trajectories. For game footage, camera extrinsics and focal parameters are logged directly from engine telemetry. For real-world and open-source footage, we develop a multi-view geometric estimation and metric scale recovery pipeline.

Camera pose estimator benchmark. We evaluate video geometry foundation models on representative scenes from the DL3DV-10K benchmark [27], comparing VGGT-Omega [42] against VGGT [41] and ViPE [17] using COLMAP reference poses across 32 matched evaluation frames. As reported in Table 11, VGGT-Omega achieves the lowest rotation error (0.234<sup>◦</sup>) while maintaining inference efficiency comparable to VGGT (16.09 s vs 187.32 s for ViPE). Figure 8 illustrates a representative qualitative trajectory comparison.

Metric scale recovery and temporal stitching. For each video clip, VGGT-Omega outputs up-to-scale relative camera extrinsics $\{ R _ { t } , \tilde { { t } } _ { t } \} _ { t = 1 } ^ { T }$ and dense relative depth maps $\tilde { D } _ { t } \in \mathbb { R } ^ { H \times \dot { W } }$ . Because interactive world models require metric physical dimensions to ground linear velocities and spatial audio inverse-square attenuation, we recover physical metric scale using monocular metric depth estimation via Depth Anything 3 (DA3) [26], with Metric3D-v2 [16] serving as an architectural fallback for scenes with extreme perspective distortion. Specifically, for keyframes sampled across the clip (at 2 fps), we compute the pixel-wise metric scale ratio over a confidence mask $\mathcal { M } _ { t }$ excluding sky regions and specular highlights:

$$
r _ { t } ( u , v ) = \frac { D _ { t } ^ { \mathrm { m e t r i c } } ( u , v ) } { \tilde { D } _ { t } ( u , v ) } , \quad ( u , v ) \in \mathcal { M } _ { t } .\tag{8}
$$

The global metric scale factor s for the clip is computed via the robust median across all valid keyframe pixels: $s = \mathrm { m e d i a n } _ { ( t , u , v ) \in \mathcal { M } } r _ { t } ( u , v )$ , from which metric translations are recovered via $\mathbf { \Psi } _ { { t } _ { t } } ~ = ~ s \cdot \tilde { \mathbf { { t } } } _ { t }$ . When estimating extended shots across multiple consecutive 5.04-second windows, the estimator operates with a temporal sliding window overlapping by 12 frames (0.5 s). We solve an SE(3) registration problem across the overlapping frames to align adjacent coordinate systems, measuring trajectory continuity by the residual camera center RMS displacement:

$$
\mathrm { R M S } _ { \mathrm { s t i t c h } } = \sqrt { \frac { 1 } { K } \sum _ { k = 1 } ^ { K } { \| \hat { \pmb { t } } _ { k } ^ { ( w ) } - \pmb { t } _ { k } ^ { ( w - 1 ) } \| _ { 2 } ^ { 2 } } } ,\tag{9}
$$

where $\hat { \pmb { t } } _ { k } ^ { ( w ) }$ denotes the camera position in window w mapped into the reference frame of window $w - 1$

Kinematic plausibility gating and action derivation. To eliminate tracking failure artifacts and unphysical camera jumps, all estimated trajectories are validated against kinematic physical bounds: (i) mean velocity $\bar { v } \in [ 0 , 4 0 )$ m/s, (ii) maximum instantaneous velocity $v _ { \mathrm { m a x } } \leq 5 0$ m/s, (iii) maximum linear acceleration $a _ { \mathrm { m a x } } \leq 2 0 \mathrm { m / s ^ { 2 } }$ , and (iv) crosswindow stitching error $\mathrm { R M S } _ { \mathrm { s t i t c h } } ~ \le ~ 0 . 0 5 \mathrm { m }$ . No minimum-motion threshold is enforced, preserving stationary viewpoints and pure in-place rotations. Finally, continuous relative camera motions between adjacent frames are quantized into the 81-class discrete action vocabulary $( 9 \times 9$ bins of linear velocity and yaw rotation rate) detailed in Sec. 3.1.

## C.5 Decoupled Multimodal Captioning (Stage 4)

Conditioning an interactive audio-visual world model on a single entangled text prompt induces cross-modal hallucination during training (e.g., visual models hallucinating sound effects that are absent, or audio models generating voices when text mentions visible actors). To prevent cross-modal contamination, we formulate a decoupled caption contract comprising three independent caption streams per clip, generated using Qwen3-Omni-30B [33] via isolated forward passes:

• Visual caption (V): Describes setting and salient subject motion while strictly ignoring audio. Viewpoint changes and camera-induced parallax are forbidden in the text, ensuring camera dynamics remain exclusively governed by camera conditioning.

• Audio caption (A): Describes acoustic layers, event order, and temporal dynamics while ignoring visual content. Because captioning uses a mono downmix, directional terms (e.g., left/right) are strictly prohibited, preserving stereo panning for geometric control. Speech is treated as ambient sound without transcription.

• Joint audio-visual caption (AV): Explicitly grounds audible events to visible on-screen emitters, tracking crossmodal synchronization and volume changes as sound sources enter or leave the camera frustum.

This tripartite representation eliminates textual cross-talk and provides the structured textual supervision for the caption ablation in Sec. 5.2.

## D HelixBench and Efficiency Evaluation

## D.1 Case Selection

Case organization. Each case carries a scene label and an interaction label, assigned by human annotators as content descriptions. Per-metric applicability is decided separately from the content labels, so label counts are not metric denominators; each metric reports on its applicable subset together with the number of scored and non-scorable cases. The benchmark contains 1,015 cases; DeSync applies to the 165 Event cases, and the spatial score to the Perspective subset.

Selection pipeline. Sources are isolated from the training corpus at the whole-video level; a source that matches training data cannot re-enter through a different time window. Candidate windows pass technical gates: frame continuity over the full clip (scene cuts, black or frozen frames, montage effects), audio continuity and audibility (missing audio, long or edge silences, suspected fades via RMS-envelope checks), and a true-stereo gate (side/mid energy ratio $\leq - 4 5$ dB rejected; weak-stereo cases routed to review), since duplicated mono channels carry no spatial information.

Spatial subset pipeline. The spatial subset is built by scanning full source videos: stereo energy and direction statistics are computed in 0.1-second windows, candidate anchors are proposed from audio events and visual changes under a per-interval budget, and windows of about 5 seconds are cut around anchors. A case is admitted only if a single dominant sound source is visible and localizable; the source is bound with open-vocabulary detection using visual evidence alone, so audio direction never influences which object is selected. Background music, rain, waves, and multi-source scenes are excluded; a reference alignment threshold selects candidates for human review, and overlapping windows are removed.

## D.2 Audio Quality

KL divergence. Generated audio is compared with the reference audio of the corresponding case, using the first 5 seconds of each. PaSST [22] produces 527 audio-event logits per clip; applying softmax gives reference and generated distributions $p _ { i }$ and $q _ { i }$

$$
\mathrm { K L } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { c = 1 } ^ { 5 2 7 } p _ { i , c } \log \frac { p _ { i , c } } { q _ { i , c } } , \qquad N = 1 , 0 1 5 .\tag{10}
$$

The direction is reference-to-generated, $D _ { \mathrm { K L } } ( p _ { i } \| q _ { i } )$ , with natural logarithms and equal averaging over clips. Lower values indicate closer agreement with the reference audio-event distributions.

FAD. Fréchet Audio Distance [21] compares generated and reference audio distributions in the VGGish [15] embedding space. From the first 5 seconds of each clip, five 128-dimensional window embeddings are extracted, pooling all 5,075 embeddings from the 1,015 clips in each condition. Let $\left( \mu _ { r } , \Sigma _ { r } \right)$ and $( \mu _ { g } , \Sigma _ { g } )$ denote the empirical means and covariances of the reference and generated embeddings:

$$
\mathrm { F A D } = \Vert \mu _ { r } - \mu _ { g } \Vert _ { 2 } ^ { 2 } + \mathrm { t r } \Big ( \Sigma _ { r } + \Sigma _ { g } - 2 \big ( \Sigma _ { r } ^ { 1 / 2 } \Sigma _ { g } \Sigma _ { r } ^ { 1 / 2 } \big ) ^ { 1 / 2 } \Big ) .\tag{11}
$$

All models use the same reference audio set; FAD is computed once per condition from the pooled window embeddings, and lower values indicate closer feature distributions.

## D.3 Cross-Modal Alignment

DeSync. The Synchformer-based evaluator [19] predicts the audio–video offset for the first and last 4.8-second windows of each generated clip. For each window, the offset class with the highest predicted probability is taken; the absolute values of the two offsets are averaged within a clip, then equally over clips. The result is an estimated temporal misalignment in seconds; lower is better.

ImageBind. ImageBind [10] embeds generated video and audio in a shared semantic space. For unit-normalized video and audio embeddings $v _ { i }$ and $a _ { i }$ from the same generated clip, the score is the mean matched cosine similarity $\begin{array} { r } { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } v _ { i } ^ { \top } a _ { i } } \end{array}$ with $N = 1 { , } 0 1 5$ ; higher values indicate stronger semantic agreement.

CLAP. CLAP [45] measures cosine similarity between the generated audio embedding and the embedding of its audio-visual input caption. The standard text encoder is used with its native 77-token truncation, and clip scores are averaged equally over all 1,015 cases. Higher scores indicate stronger audio-caption semantic agreement.

## D.4 Spatial Audio–Visual Agreement

Spatial three-region score. The score measures agreement between the horizontal position of a detected sound-source candidate in the generated video and the stereo direction of its paired generated audio.

Visual localization. The generated video is sampled at 10 Hz and Grounding DINO [30] is applied with object-category queries derived from the case’s video and joint audio-visual captions. Candidate detections are associated across frames, and a target track is selected using detection confidence, temporal coverage, and visual motion; when track selection fails, the pipeline falls back to a dominant caption-matched detection in each frame. Target selection uses visual evidence alone. For the selected bounding box, the horizontal center $x _ { t }$ is normalized as $d _ { t } = 2 x _ { t } / W - 1$ , where W is the frame width.

Audio direction. Left- and right-channel energies are computed in 100-ms windows with a 50-ms hop, giving the stereo energy pan $p _ { t } = ( E _ { R , t } - E _ { L , t } ) / ( E _ { R , t } + E _ { L , t } )$ with numerical stabilization; the pan sequence is interpolated to the visual sampling timestamps. Energies are measured from the mixed audio track without source separation. For joint audio-visual models this is the model’s native generated audio; for V2A baselines it is the replacement audio synthesized for the same video, so the fixed video provides identical visual detections across audio conditions.

Three-region rule. Visual direction is left for $d _ { t } < - \tau _ { v } ,$ , right for $d _ { t } > \tau _ { v } ,$ and center otherwise, with $\tau _ { v } = 0 . 2 ;$ audio direction uses the same rule on $p _ { t }$ with $\tau _ { a } = 0 . 1$ , equality belonging to the center region. The visual center region therefore spans 40–60% of image width, and the audio boundary corresponds to a channel energy ratio of $( 1 + \tau _ { a } ) / ( 1 - \tau _ { a } ) \approx 1 . 2 2 2$ , or approximately 0.87 dB. These tolerances stabilize left/right labels against small visual displacements and channel-energy fluctuations while retaining sensitivity to lateral deviations.

Scoring and eligibility. For visually lateral samples, same-side audio receives +1, centered audio 0, and opposite-side audio −1; visual-center samples are excluded. Within each eligible clip, valid visual-side samples are averaged and multiplied by 100; clip scores are then averaged equally over clips with at least one valid visual-side sample. The score ranges from −100 to 100, and always-centered audio scores zero. A clip qualifies if it contains at least four consecutive valid 10 Hz samples; an invalid sample or timestamp gap breaks the run, and eligible clips contribute all valid samples. The score quantifies lateral audio–visual agreement over each model’s detected targets and eligible clips.

## D.5 Post-hoc Video-to-Audio Comparison

Matched-video protocol. The dubbing comparison uses PrismAudio [28], ThinkSound [29], and AudioX [39]. All three are conditioned on the same generated videos and their original joint audio-visual captions. Their synthesized audio replaces the original track while preserving the video stream and frame timestamps; the native audio provides the matched reference condition. The protocol covers all 1,015 test cases, with the same video and caption used across the four audio conditions of each case; evaluation follows the metric definitions and applicable subsets above.

## D.6 Spatial Threshold Sensitivity

We evaluate sensitivity to the visual and audio thresholds defining the three spatial regions. Tables 12 and 13 report all 20 combinations of four visual thresholds $\tau _ { v } ~ \in ~ \{ 0 . 1 , 0 . 2 , 1 / 3 , 0 . 4 \}$ and five audio thresholds $\tau _ { a } ~ \in$ {0.01, 0.05, 0.1, 0.2, 0.3}, reusing the same generated clips, visual detections, and audio-pan trajectories under the eligibility rule above. The world-model sweep uses 145 Perspective candidates, with each model contributing its eligible clips; widening the visual threshold reduces the contributing cohorts from at most 143 clips to 113. The fixed-video comparison shares visual detections and the eligibility mask across all four audio conditions, with 147 of the 150 candidates satisfying the consecutive-sample rule and identical contributing counts for every condition. HelixWorld ranks first among all four world models across the 20 settings, and its native audio outperforms all three V2A replacements throughout the grid.

## D.7 Inference Efficiency

Hardware and measurement boundaries. All timing runs use a single NVIDIA H800 for generation and decoding. After two complete warm-up rollouts, we report the median of two timed runs. Timing excludes model loading, pre-rollout text/image preparation, and MP4 encoding, but includes in-loop geometry processing, CPU offload, decoding, and frame transfer to CPU.

Real-timefactor. For streaming models, steady-state RTF measures the wall-clock time after the first decoded block divided by the duration of the remaining newly generated frames. Let $T$ be total rollout wall time, $t _ { 1 }$ the first-block delivery time, $F$ the output frame count, b the exclusive frame index of the first delivered block, and $f$ the output frame rate. Then

$$
{ \mathrm { R T F } } _ { \mathrm { s t e a d y } } = { \frac { T - t _ { 1 } } { ( F - b ) / f } } .\tag{12}
$$

The denominator counts newly generated frames after the first delivered block over the completed rollout. Lower RTF is faster; RTF below one indicates faster-than-real-time throughput after the first block. For the non-streaming bidirectional reference, RTF uses the complete generation-and-decoding time divided by the full generated duration.

Table 12. World-model sensitivity to spatial thresholds. Results use the 145-candidate Perspective set. Pre-trained denotes the bidirectional model after progressive condition injection. Bold scores mark the best model per setting; bold thresholds mark the main-table setting. Higher is better.
<table><tr><td> $\tau _ { v }$ </td><td> $\tau _ { a }$ </td><td>LTX-2.3</td><td>EchoWM</td><td>HelixWorld</td><td>Pre-trained</td></tr><tr><td>0.10</td><td>0.01</td><td>2.16</td><td>22.58</td><td>48.69</td><td>30.14</td></tr><tr><td>0.10</td><td>0.05</td><td>1.96</td><td>21.85</td><td>46.52</td><td>30.34</td></tr><tr><td>0.10</td><td>0.10</td><td>1.55</td><td>20.87</td><td>42.15</td><td>29.30</td></tr><tr><td>0.10</td><td>0.20</td><td>1.16</td><td>17.57</td><td>31.65</td><td>25.91</td></tr><tr><td>0.10</td><td>0.30</td><td>0.90</td><td>14.91</td><td>22.94</td><td>22.80</td></tr><tr><td>0.20</td><td>0.01</td><td>2.41</td><td>25.92</td><td>51.53</td><td>36.92</td></tr><tr><td>0.20</td><td>0.05</td><td>2.20</td><td>25.30</td><td>49.88</td><td>36.27</td></tr><tr><td>0.20</td><td>0.10</td><td>1.82</td><td>24.05</td><td>46.41</td><td>33.61</td></tr><tr><td>0.20</td><td>0.20</td><td>1.03</td><td>20.69</td><td>35.07</td><td>28.00</td></tr><tr><td>0.20</td><td>0.30</td><td>0.61</td><td>18.41</td><td>25.73</td><td>24.90</td></tr><tr><td>1/3</td><td>0.01</td><td>4.78</td><td>31.07</td><td>59.85</td><td>46.47</td></tr><tr><td>1/3</td><td>0.05</td><td>4.40</td><td>31.34</td><td>58.09</td><td>46.22</td></tr><tr><td>1/3</td><td>0.10</td><td>3.50</td><td>30.38</td><td>54.98</td><td>42.61</td></tr><tr><td>1/3</td><td>0.20</td><td>1.07</td><td>24.70</td><td>42.94</td><td>35.68</td></tr><tr><td>1/3</td><td>0.30</td><td>0.68</td><td>21.75</td><td>31.52</td><td>31.45</td></tr><tr><td>0.40</td><td>0.01</td><td>5.06</td><td>32.62</td><td>62.53</td><td>46.35</td></tr><tr><td>0.40</td><td>0.05</td><td>4.54</td><td>32.60</td><td>61.56</td><td>46.90</td></tr><tr><td>0.40</td><td>0.10</td><td>4.24</td><td>31.79</td><td>58.46</td><td>43.88</td></tr><tr><td>0.40</td><td>0.20</td><td>1.80</td><td>27.12</td><td>46.35</td><td>37.29</td></tr><tr><td>0.40</td><td>0.30</td><td>1.60</td><td>24.44</td><td>34.53</td><td>32.63</td></tr></table>

Table 13. Native versus post-hoc audio across spatial thresholds. All conditions use fixed HelixWorld videos from the 150- candidate set, sharing detections and eligibility. Bold scores mark the best audio condition; bold thresholds mark the main-table setting. Higher is better.
<table><tr><td> $\tau _ { v }$ </td><td> $\tau _ { a }$ </td><td>HelixWorld (native)</td><td>+ AudioX</td><td>+ ThinkSound</td><td>+ PrismAudio</td></tr><tr><td>0.10</td><td>0.01</td><td>49.32</td><td>-3.94</td><td>21.72</td><td>-9.94</td></tr><tr><td>0.10</td><td>0.05</td><td>47.11</td><td>-5.33</td><td>21.36</td><td>-8.81</td></tr><tr><td>0.10</td><td>0.10</td><td>42.79</td><td>-4.84</td><td>19.04</td><td>-6.69</td></tr><tr><td>0.10</td><td>0.20</td><td>32.06</td><td>-4.69</td><td>14.30</td><td>-4.12</td></tr><tr><td>0.10</td><td>0.30</td><td>23.07</td><td>-4.82</td><td>10.47</td><td>-2.23</td></tr><tr><td>0.20</td><td>0.01</td><td>51.97</td><td>-3.73</td><td>19.79</td><td>-11.07</td></tr><tr><td>0.20</td><td>0.05</td><td>50.26</td><td>-4.36</td><td>19.75</td><td>-9.99</td></tr><tr><td>0.20</td><td>0.10</td><td>46.88</td><td>-4.01</td><td>18.05</td><td>-8.04</td></tr><tr><td>0.20</td><td>0.20</td><td>35.30</td><td>-4.87</td><td>13.48</td><td>-4.82</td></tr><tr><td>0.20</td><td>0.30</td><td>25.70</td><td>-5.22</td><td>10.07</td><td>-2.44</td></tr><tr><td>1/3</td><td>0.01</td><td>60.18</td><td>-3.24</td><td>21.40</td><td>-13.81</td></tr><tr><td>1/3</td><td>0.05</td><td>58.34</td><td>-1.98</td><td>20.42</td><td>-11.89</td></tr><tr><td> $1 / 3$ </td><td>0.10</td><td>55.45</td><td>-2.21</td><td>18.67</td><td>-10.15</td></tr><tr><td> $1 / 3$ </td><td>0.20</td><td>43.01</td><td>-4.46</td><td>11.26</td><td>-6.28</td></tr><tr><td> $1 / 3$ </td><td>0.30</td><td>31.37</td><td>-5.97</td><td>7.95</td><td>-4.01</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>0.40</td><td>0.01</td><td>63.28 62.35</td><td>-2.50</td><td>22.17 21.56</td><td>-13.41</td></tr><tr><td>0.40 0.40</td><td>0.05 0.10</td><td>59.32</td><td>-1.38 -2.75</td><td>19.37</td><td>-11.42 -10.39</td></tr><tr><td>0.40</td><td>0.20</td><td>46.45</td><td>-4.52</td><td>12.22</td><td>-7.32</td></tr><tr><td>0.40</td><td>0.30</td><td>34.46</td><td>-4.59</td><td>8.55</td><td>-5.09</td></tr></table>

Output settings. All models are evaluated at 768×512 resolution, using the output frame rates reported in their official repositories.

## E Additional Trajectory Distillation Results

## E.1 Conditional Visual Diversity

Inputs andfeatures. Our diversity calculation follows Diversity-Preserved Distribution Matching Distillationfor Fast Visual Synthesis [44], adapting its per-prompt image comparison to matched video frames. We extract DINOv3 ViT-L/16 pooled features [36] and CLIP ViT-L/14 CLS features [34] at matched times across generations of the same input, then normalize each feature vector.

Aggregation. Let $f _ { i , t , \varepsilon }$ be a unit-normalized feature for input i, time $t \in \tau$ , and generation $s \in \{ 1 , \ldots , S \}$ . For each input we average cosine distance over generation pairs and times:

$$
d _ { i } = \frac { 1 } { | \mathcal { T } | \binom { S } { 2 } } \sum _ { t \in \mathcal { T } } \sum _ { 1 \leq s < s ^ { \prime } \leq S } \left( 1 - f _ { i , t , s } ^ { \top } f _ { i , t , s ^ { \prime } } \right) .\tag{13}
$$

The fixed first frame is excluded. If G is the set of sources and $\mathcal { T } _ { g }$ contains the inputs from source g, the reported score is

$$
D = \frac { 1 } { | \mathcal { G } | } \sum _ { g \in \mathcal { G } } \frac { 1 } { | \mathscr { T } _ { g } | } \sum _ { i \in \mathscr { T } _ { g } } d _ { i } .\tag{14}
$$

Each source receives equal weight, and higher scores indicate greater conditional visual diversity.

Results. Self-forcing with online trajectory distillation loss increases DINOv3 diversity by 21.7% and CLIP diversity by 13.5% over DMD-only self-forcing (Table 8).

## E.2 Additional Qualitative Comparisons

Figure 9 shows further examples of reduced visual drift with trajectory distillation.

## F User Study Protocol

Study design. Model identities are hidden, and A/B placement is randomized and balanced. The study presents 20 audio-enabled comparisons followed by 20 muted-video comparisons. With audio, we compare HelixWorld with EchoWM and three post-hoc dubbing baselines that apply AudioX, ThinkSound, or PrismAudio to the same HelixWorld video. The muted-video comparison includes Alaya, HY-WorldPlay 1.5, LingBot v2, SANA, and EchoWM. We analyze 212 audio-enabled and 190 muted-video responses, covering 20 scenes in each setting.

Evaluation criteria. The audio-enabled study asks about audio–visual synchrony, audio quality, spatial acoustic realism, and overall preference. The muted-video study asks about visual quality, action following, scene following, temporal consistency, and overall preference.

Preference aggregation. Responses are summarized as a preference for HelixWorld, a tie, or a preference for the baseline. For each model pair, a category’s share is its response count divided by the total number of responses, with ties retained in the denominator. Figure 7 reports overall preference for each pair. Win rates reported in the main text exclude ties.

![](images/c97d1c75aeffc59f65b0b6ae07985c8c375a8c6e5e9fe0db5650c037a5b11e82.jpg)  
Figure 9. Additional visual comparisons of trajectory distillation. Each pair shows DMD only (top) and with trajectory distillation (bottom).

## G Additional Audio-Visual Comparisons

Figures 10 and 11 illustrate audio–visual synchronization; Figures 12 and 13 show stereo correspondence. AudioX, ThinkSound, and PrismAudio use the same HelixWorld video, while EchoWM is shown with its own generated video and audio. Mel power is normalized per recording, using a shared reference for the left and right channels.

![](images/a99efec20a5732b0f0d1e46629f10bc29c75037d3ad917543efa5f51984aca98.jpg)

![](images/ca3c666e6c508cf98de8a868e7ddd8319d3d48defebc01c18ee138ed300cf4f8.jpg)  
Figure 10. Audio–visual synchronization during golf swings. Dashed lines mark the visible strikes in the shared video. HelixWorld’s acoustic transients coincide with both events.

![](images/23d5dbf0eadfde83c0f56f4393f231021d6310a69acf5c4d131f60562889e000.jpg)  
2.50 s  
50 Hz–16 kHz

![](images/d7c002214b3fc38e88a56ad51906f921ac1d06cd518b87345c6104fa30e5f8da.jpg)  
Shared video

## Drumming: pause and resume

2.50 s  
![](images/74b3cdaeddd7739a9186e5139d15c20f3c771fdddd2fffdb363764abc17d7a81.jpg)

3.50 s  
![](images/3c0a46d30bd5bca2fb4d733d26cec6af0ab906156eee22389c140a9eeeb7c8a4.jpg)

3.83 s  
![](images/d4908bb684e99e66eefd67533450316239a4e456f67222545f6bbad3c81b2cc3.jpg)

HelixWorld

![](images/62a972e41a00ae834a40c4992198e36c41ce576f18a22a085751d0344e22ded7.jpg)

\+ AudioX

\+ ThinkSound

\+ PrismAudio

![](images/c3efde2a18be683b9e2b36ac36c862a5623a441870b914be183b7b1572a90d2d.jpg)  
Time (s)

![](images/a05fc3c8bf7a1fd433f96ee69b2bd9146a1bd1fa6313d6cfbd7cff514f0c97f4.jpg)

EchoWM  
Own generated video and audio 1.00 s  
![](images/185b42ef981f35e0eb66813c9cfebd7de458bb1c8f684726d617509f89e727d5.jpg)

![](images/3e1c9cc458d2d274fbebe4932849901e05fd0446760f0ba240fcbe91b8da9dd0.jpg)

4.00 s  
![](images/09e0295146a23edc77dedfbd0ba22cd1f9dd0d0b47e8db959bb4386d6b86dd39.jpg)

![](images/3281709173ee9871f63250c076d69a81435af39635615514342a9fa78908b658.jpg)  
Time (s)

![](images/b50b2472f0d34ea6a27f36a5b897403cea58fe247c50775ff83b0c8bcf51880f.jpg)  
Figure 11. Audio–visual synchronization during drumming. The marked interval is a visible pause. HelixWorld’s transients subside during the pause and resume with the motion; AudioX produces a transient during the pause.

![](images/ccbeb9213df0957ca280b297b1ffe64c48a79e8336d1f75eb3e28c8be42b0e52.jpg)  
Fountain moving to the right  
HelixWorld / AudioX / ThinkSound / PrismAudio 0.00 s 1.50 s

![](images/a37bd6fcfd0f65edde803306e7927368880d3d882de7cd8a49abe048b833b19d.jpg)  
Figure 12. Stereo audio as the fountain moves rightward. L and R denote the left and right channels. HelixWorld maintains stronger right-channel energy, consistent with the fountain’s position in the video.

0.50 s  
1.00 s  
Car crossing from left to right  
HelixWorld / AudioX / ThinkSound / PrismAudio 0.00 s 1.00 s  
![](images/4e0008f5a58dde34408e08c33ab592c7fb251bdc9d21c24362ac903b6399f709.jpg)

![](images/147d0045bb8601bb6fb06199c635ff9b79d33038ddcc8b1949557c02f1f011ac.jpg)

2.00 s  
![](images/634f3c96ddbb011bd7506d29e53753de329a3e38c8931bf3149bb38d38669e31.jpg)

![](images/91e9250c71dd31ac3bd74721ab40ee955dbc2a2ad6fc2a5c47452ec57a336714.jpg)

Own generated video and audio 0.00 s  
![](images/a994f35e2779836a1d8aec3c00a639c33136f90366eb3c96715b7d564971843a.jpg)

![](images/6121525f1e310be38c5a0a47d6e2583a644fe93179eb8e853e9f65e09a6e6c58.jpg)

![](images/f8c3ec0f5cbdd53361405056e972416ad33bec45f7143d71069b7f1befdefaa1.jpg)  
Figure 13. Stereo audio during a left-to-right vehicle pass. HelixWorld’s channel balance shifts from left to right with the car’s motion. L and R denote the two audio channels.