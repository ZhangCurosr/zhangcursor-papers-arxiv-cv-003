# TimeSteer: Inference-Time Speech Scheduling in Joint Audio-Visual Diffusion Models

Chao Zhou, Yiling Chen, Qi Chu, Tao Gong, Nenghai Yu, Tianyi Wei

## Abstract

Although pretrained joint audio-visual diffusion models offer rich control over what to generate, they provide no explicit control over when an utterance should occur. To address this, we study inference-time speech scheduling, a novel task that places coupled speech and visual articulation within userspecified begin-end intervals without finetuning the backbone model. We uncover two intrinsic properties of the denoising process that enable this task. First, a timing-sensitive text-to-audio cross-attention head exposes each utterance's model-implied source span along the latent timeline. Second, the predicted clean latent already organizes coupled speech and visual articulation, allowing their temporal placement to be edited without regenerating the content. Building on these discoveries, we propose TimeSteer, a training-free framework that localizes each utterance's source span through Source Span Localization and transfers the associated audiovisual latent content from the source interval to the specified target interval through Region-Aware Latent Remapping. We further introduce SpeechShift, the first benchmark for interval-level speech scheduling in joint audio-visual generation. Experiments across two representative backbones show that TimeSteer substantially improves interval controllability over training-free baselines while maintaining competitive overall generation quality.

## Introduction

Recent joint audio-visual diffusion models synthesize temporally coupled speech and video in a single generation pass, enabling practical text-to-audio-visual (T2AV) generation. Systems such as LTX-2 (HaCohen et al. 2025) and daVinci-MagiHuman (Chern et al. 2026) accept diverse conditioning signals, including open-ended text, optional visual inputs, and speech-centric controls such as reference timbre, while producing natural prosody and synchronized lip motion. These advances provide increasingly rich control over what is generated, yet offer almost no control over when an utterance begins and ends.

This missing temporal controllability limits scriptable audio-visual generation. Applications such as filmmaking, interactive agents, and game engines often require dialogue to occupy prescribed begin-end intervals, whether to follow visual events, coordinate multiple speakers, or insert deliberate pauses. Such interval constraints specify not only utterance onset and offset, but also the effective speaking rate required for each sentence to naturally fit its assigned duration. However, current joint generators provide no explicit mechanism for enforcing these temporal constraints: speech timing emerges implicitly during denoising, making utterance placement difficult to predict or reproduce across random seeds. We call this capability inference-time speech scheduling: given one or more quoted utterances, generate coupled audio-visual speech that places each utterance within any user-specified begin-end interval, without any retraining or finetuning.

Despite its practical importance, inference-time speech scheduling has received little attention. Existing T2AV benchmarks such as JavisBench (Wang et al. 2025b) and AVGen-Bench (Liu et al. 2026a) evaluate synchronization and semantic fidelity, but not interval-level speech controllability. Related generation methods either control the timing of isolated acoustic events in text-to-audio generation (Xu et al. 2024; Wang et al. 2025c; Zhang et al. 2025) or synchronize video to externally provided speech (Weng et al. 2025; Lee et al. 2025). The central question therefore remains open: can a pretrained joint generator directly generate coupled speech and visual articulation within user-specified begin-end intervals entirely at inference time?

We answer this question affirmatively by uncovering two intrinsic properties of the pretrained generator's denoising process. First, speech timing emerges internally well before decoding and remains traceable as generation proceeds. In particular, a timing-sensitive cross-attention head naturally exposes the source span of each utterance along the latent timeline. Second, the model's learned audio-visual prior already couples each utterance with its corresponding visual articulation in the predicted clean latent, so localizing its source span is sufficient to jointly reschedule both modalities without regenerating their content.

Motivated by these insights, we propose TimeSteer, a training-free framework that operates during denoising in two successive stages: Source Span Localization and Region-Aware Latent Remapping. First, Source Span Localization aggregates the responses of a timing-sensitive text-to-audio cross-attention head over the text tokens corresponding to each utterance, thereby estimating its source span without decoding any intermediate predictions. Then, Region-Aware Latent Remapping reorganizes the predicted clean latent to transfer the corresponding joint audio-visual latent content from the localized source span to the target interval. Because a source span and its target interval may differ in duration, edited speech regions and the surrounding non-speech gaps require distinct treatment: the former must adapt the speaking rate to match the user-specified duration, whereas the latter should preserve the original temporal progression as much as possible while smoothly absorbing transitions between speech intervals. Accordingly, we introduce Minimal-Distortion Remapping, which applies an affine map within each speech interval, and Curvature Bridging, which smoothly interpolates the read map across non-speech gaps. Figure 1 shows that TimeSteer achieves precise speech relocation to any user-specified target interval while preserving synchronized audio-visual generation.

![](images/a6ed65e2adf90d3bf6b0cfd081f431d38941d2e4d26ac419cd96ed498dc4a093.jpg)  
Figure 1: Qualitative speech scheduling. Without TimeSteer, a timing instruction of the form “from /start] to /end]"is appended to the original prompt, yet the speech remains near the video onset. TimeSteer places the utterance within any userspecified target interval while preserving generation quality and text alignment. Shading marks the target intervals; frames are sampled at 1 fps with aligned mel-spectrograms below.

Evaluating inference-time speech scheduling requires dedicated benchmarks. Existing T2AV evaluations assess synchronization and perceptual quality but provide neither target intervals nor metrics for interval-level controllability. We therefore introduce SpeechShift, the first benchmark designed for this task. SpeechShift contains 400 prompts across 102 scenes and 600 utterance-level scheduling targets, spanning four combinations of speaker and utterance counts together with diverse acoustic interference and actioncoupling conditions. To quantify controllability, SpeechShift measures how closely each realized speech interval matches its target in both boundary timing and temporal overlap. It also evaluates whether this control preserves the quality and synchronization of the generated audio-visual content.

Taken together, TimeSteer and SpeechShift provide a method and benchmark for inference-time speech scheduling. Experiments across two joint audio-visual diffusion backbones show that TimeSteer substantially improves interval controllability over training-free baselines while maintaining competitive generation quality.

## Our contributions are:

• Task. We introduce inference-time speech scheduling, a new task of placing each generated utterance and its coupled visual articulation within any user-specified target interval, without any retraining or finetuning.

• Discovery. We uncover two intrinsic properties of joint audio-visual denoising: cross-attention exposes the model-implied source span of each utterance, while the predicted clean latent makes the temporal placement of coupled speech and visual articulation directly editable.

• Method. We propose TimeSteer, a training-free framework that first localizes each utterance's source span and then transfers its joint audio-visual latent content to the target interval without regenerating it. Its region-aware remapping combines uniform temporal scaling within utterances with smooth transitions across non-speech gaps.

• Benchmark. We introduce SpeechShift, the first benchmark for interval-level speech scheduling in joint audio-visual generation, comprising 400 prompts, 600 utterance-level targets, and 102 scenes. Its evaluation jointly measures interval controllability and the preservation of overall generation quality.

## Related Work

## Joint Audio-Visual Generation

Joint audio-visual generation has evolved from modalityspecific fusion toward large-scale diffusion transformers that synthesize sound and video in a unified process. Early methods such as MM-Diffusion and SyncFlow introduced dedicated routing and synchronization mechanisms for crossmodal interaction (Ruan, Liu et al. 2023; Wang et al. 2024). Recent work has scaled this paradigm to larger, more integrated architectures (Polyak et al. 2024; Xing et al. 2024;

Wang et al. 2025a; Li et al. 2025b; Low, Wang, and Katyal 2025; Team 2026; Wang et al. 2026; Chen et al. 2025b). For example, LTX-2 couples separate audio and video streams through cross-attention, whereas daVinci-MagiHuman processes all modality tokens within a unified attention architecture (HaCohen et al. 2025; Chern et al. 2026).

Building on this progress, recent extensions introduce reference conditioning, alignment objectives, and preference optimization to improve content control and cross-modal consistency (Zhang et al. 2026; Liu et al. 2026b; Ma et al. 2026). These methods broaden control over what is generated, but do not expose when individual utterances should occur. Existing evaluations follow the same emphasis: JavisBench and AVGen-Bench measure semantic fidelity, perceptual quality, and synchronization, yet provide neither target speech intervals nor metrics for inference-time speech scheduling (Wang et al. 2025b; Liu et al. 2026a). As a result, placing model-generated speech and its corresponding visual articulation within user-specified intervals has received little direct attention in joint audio-visual generation.

## Temporal Control in Audio and Video

Temporal control has been extensively studied in modalityspecific audio and video generation, enabling acoustic events and visual motions to follow explicit timing conditions. In text-to-audio generation, methods build on large audio models (Kreuk et al. 2023; Liu et al. 2023; Huang et al. 2024b) and introduce event timelines, timestamp conditioning, compositional controls, or inference-time attention manipulation to place acoustic events at desired times (Huang et al. 2024a; Xu et al. 2024; Li et al. 2025a; Wang et al. 2025c; Zhang et al. 2025). Text-to-video methods similarly control temporal evolution through motion trajectories and frame-level conditions (Huang et al. 2025; Ruan et al. 2023; Yariv et al. 2023).

A separate line of work obtains timing from a pregenerated modality. Audio-conditioned video methods synchronize visual dynamics to an existing soundtrack (Weng et al. 2025; Lee et al. 2025; Song et al. 2026), while videoto-audio systems synthesize sound aligned with an input video (Luo et al. 2023; Cheng et al. 2025; Kushwaha and Tian 2025; Chen et al. 2025a). Talking-face and avatar systems likewise animate visual content from pre-generated speech (Prajwal et al. 2020; Zhang et al. 2023; Shen et al. 2023; Yu et al. 2023; Aneja et al. 2024; Peng et al. 2024; Guan et al. 2025; Corona et al. 2025; Cui et al. 2025). These methods achieve temporal alignment by treating audio or video as an external timing anchor.

Existing approaches therefore either control timing within a single generated modality or inherit it from an externally provided modality. TimeSteer instead relocates modelgenerated speech together with its visual articulation inside a pretrained joint audio-visual diffusion model. It performs this scheduling entirely at inference time from user-specified intervals, without external speech or additional training.

## Method and Benchmark

## Task Formulation

Let c denote a text prompt describing a scene and containing an ordered collection U of N quoted utterances. For each utterance indexed by $u \in \{ 1 , \ldots , N \}$ , the user specifies a target interval $\begin{array} { r } { \mathcal { T } ^ { u } = [ t _ { s } ^ { u } , t _ { e } ^ { u } ] . } \end{array}$ where $\overline { { t _ { s } ^ { u } } }$ and $t _ { e } ^ { u }$ denote its target start and end times in the generated video, respectively Given a pretrained joint audio-visual diffusion model $\mathcal { G } _ { \theta }$ with frozen parameters θ, our goal is to construct a training-free scheduling operator S for inference-time speech scheduling:

$$
( A , V ) = { \cal S } ( { \mathcal G } _ { \theta } , c , \{ T ^ { u } \} _ { u = 1 } ^ { N } ) ,\tag{1}
$$

where $A$ and $V$ denote the generated audio and video, respectively. The generated output should align the realized interval of every utterance u with $\mathcal { T } ^ { u }$ while preserving the generation quality of the pretrained model.

## TimeSteer

TimeSteer is a training-free framework for inference-time speech scheduling in joint audio-visual diffusion models. As illustrated in Fig. 2, it enables precise temporal control by intervening on the predicted clean latent throughout the denoising process, without modifying the pretrained model parameters.

At each denoising step n, the diffusion model predicts the clean latent $\hat { x } _ { 0 } ^ { n }$ from the current noisy latent following the standard flow-matching formulation (Lipman et al. 2023; Liu, Gong, and Liu 2023), i.e., $\hat { x } _ { 0 } ^ { n } = x _ { n } - \sigma _ { n } v _ { n }$ , where $x _ { n } ,$ $v _ { n } .$ and $\sigma _ { n }$ denote the noisy latent, predicted velocity, and noise level, respectively. Before the prediction $\hat { x } _ { 0 } ^ { n }$ is used for the next denoising update, the scheduling operator acts on the predicted clean latent as

$$
\tilde { x } _ { 0 } ^ { n } = S ( \hat { x } _ { 0 } ^ { n } ) ,\tag{2}
$$

where $\tilde { x } _ { 0 } ^ { n }$ denotes the scheduled clean latent. The scheduled clean latent is then converted back into the corresponding velocity prediction following the standard flow-matching update, allowing the remaining denoising process to proceed unchanged.

TimeSteer implements the scheduling operator $s$ in two successive stages. First, Source Span Localization identifies the source span of each quoted utterance from the crossattention map. Then, using these localized spans, Region-Aware Latent Remapping constructs a read map from the target latent timeline to the source latent timeline. The map specifies which source coordinate is sampled at each target coordinate, thereby relocating each utterance to its target interval while preserving temporal continuity. The procedure is provided in the supplementary material.

Source Span Localization Source Span Localization builds on our observation that a timing-sensitive text-toaudio cross-attention head in the frozen DiT produces a temporally localized response for each utterance. This response traces the utterance's model-implied source span along the audio latent timeline. Based on that, we can get an explicit span estimate before decoding, as illustrated in Fig. 2(b).

Let T denote the latent temporal length and $\mathcal { A } _ { 0 } ^ { n }$ the audio branch of the predicted clean latent $\hat { x } _ { 0 } ^ { n }$ , which is the model's estimate of the final clean latent at denoising step n. We replay $\mathcal { A } _ { 0 } ^ { n }$ through the frozen DiT and extract a text-to-audio cross-attention map $\tilde { A } \in \mathbb { R } ^ { T \times L _ { c } }$ from the timing-sensitive attention layer $\ell *$ with head $\mathcal { R } ^ { * }$ , where $L _ { c }$ denotes the number of text tokens. Each entry ${ \tilde { A } } _ { i j }$ measures the correspondence between audio latent position i and text token j.

![](images/796ffdf85080f2e403547054ee3c2f8fcabdc3378b03cb89035fd419747a4878.jpg)  
Figure 2: Overview of TimeSteer. (a) Given target intervals, TimeSteer localizes each utterance's source span and relocates its predicted audio-visual content. (b) Source Span Localization estimates source spans from text-to-audio cross-attention. (c) Region-Aware Latent Remapping relocates the content through its read map.

For an utterance u spanning the quoted token range $\left[ j _ { 0 } ^ { u } , j _ { 1 } ^ { u } \right]$ , where $j _ { 0 } ^ { u }$ and $j _ { 1 } ^ { u }$ are its first and last token indices, we aggregate the attention weights over its quoted tokens:

$$
m _ { i } ^ { u } = \sum _ { j = j _ { 0 } ^ { u } } ^ { j _ { 1 } ^ { u } } \tilde { A } _ { i j } , \qquad i = 0 , \dots , T - 1 .\tag{3}
$$

The resulting scores measure the correspondence between each audio latent position and utterance u.

Let $m _ { \mathrm { m a x } } ^ { u } = \operatorname* { m a x } _ { i } m _ { i } ^ { u }$ . Using a relative threshold $\tau \ \in$ (0, 1), we estimate the source span as

$$
\begin{array} { r l } & { S ^ { u } = [ s _ { 0 } ^ { u } , s _ { 1 } ^ { u } ] , } \\ & { ~ s _ { 0 } ^ { u } = \operatorname* { m i n } \left\{ i \mid m _ { i } ^ { u } \geq \tau m _ { \operatorname* { m a x } } ^ { u } \right\} , } \\ & { ~ s _ { 1 } ^ { u } = \operatorname* { m a x } \left\{ i \mid m _ { i } ^ { u } \geq \tau m _ { \operatorname* { m a x } } ^ { u } \right\} . } \end{array}\tag{4}
$$

The resulting source span $S ^ { u }$ is subsequently used by Region-Aware Latent Remapping.

Region-Aware Latent Remapping Given the localized source spans and target intervals, the remaining task is to relocate the predicted audio-visual content from each source span to its target interval while preserving temporal continuity, as illustrated in Fig. 2(c).

Our key idea is to relocate the predicted clean latent content without regenerating it. We construct a read map over the latent timeline that specifies which source position in the predicted clean latent is sampled at each destination position.

For each utterance u, mapping the target interval $\mathcal { T } ^ { u }$ onto the latent timeline yields the destination speech interval $[ d _ { 0 } ^ { u } , d _ { 1 } ^ { u } ]$ , which is paired with the localized source span $S ^ { u } \dot { = } \big [ s _ { 0 } ^ { u } , \dot { s } _ { 1 } ^ { u } \big ]$ . The source and destination intervals need not have the same duration.

We define a destination-to-source read map

$$
h : [ 0 , T - 1 ] \to [ 0 , T - 1 ] ,\tag{5}
$$

where $h ( t )$ specifies the source coordinate sampled at destination coordinate t. Because the source and destination speech intervals may differ in duration, we align their endpoints by imposing $\dot { h } ( d _ { 0 } ^ { u } ) = s _ { 0 } ^ { u }$ and $h ( d _ { 1 } ^ { u } ) = s _ { 1 } ^ { \check { u } }$ . We also set $h ( 0 ) = 0$ and $h ( \bar { T } - \dot { 1 } ) \stackrel { \smile } { = } T - 1$ to keep the clip boundaries fixed. In implementation, we evaluate the continuous map h at discrete destination indices. When $h ( t )$ falls between two source indices, the corresponding latent is obtained by linearly interpolating the two neighboring latent vectors, with weights determined by the fractional part of $h ( t )$

The derivative $h ^ { \prime } ( t ) \ = \ d h ( t ) / d t$ is the local slope of the read map. Because h maps destination time to source time, this slope scales the original speaking rate within each speech interval and can therefore be interpreted as the local speaking-rate ratio, i.e., the output speaking rate divided by the original speaking rate. The same temporal scaling applies to the associated visual articulation.

Speech intervals and non-speech gaps require different remapping strategies. Specifically, we construct h using two complementary components: Minimal-Distortion Remapping, which applies an affine map within each speech interval, and Curvature Bridging, which smoothly interpolates the read map across non-speech gaps.

Minimal-Distortion Remapping for Speech Intervals. Within each speech interval, we aim to preserve the original speaking rate, corresponding to $h ^ { \prime } ( \bar { t } ) = 1$ . Under the source-destination endpoint constraints, we therefore minimize the accumulated deviation from this value:

$$
\begin{array} { l } { { \displaystyle h _ { u } ^ { \star } = \arg \operatorname* { m i n } _ { h } \int _ { d _ { 0 } ^ { u } } ^ { d _ { 1 } ^ { u } } \left( h ^ { \prime } ( t ) - 1 \right) ^ { 2 } d t , } } \\ { { \mathrm { s . t . } \quad h ( d _ { 0 } ^ { u } ) = s _ { 0 } ^ { u } , \qquad h ( d _ { 1 } ^ { u } ) = s _ { 1 } ^ { u } . } } \end{array}\tag{6}
$$

The resulting minimizer is the affine map

$$
h _ { u } ^ { \star } ( t ) = s _ { 0 } ^ { u } + \kappa _ { u } \big ( t - d _ { 0 } ^ { u } \big ) , \qquad t \in [ d _ { 0 } ^ { u } , d _ { 1 } ^ { u } ] ,\tag{7}
$$

where $\begin{array} { r } { \begin{array} { r c l } { \kappa _ { u } } & { = } & { \frac { s _ { 1 } ^ { u } - s _ { 0 } ^ { u } } { d _ { 1 } ^ { u } - d _ { 0 } ^ { u } } } \end{array} } \end{array}$ is its constant slope and thus the speaking-rate ratio for utterance $u .$ Thus, each speech interval uses a simple linear read map.

Curvature Bridging for Non-speech Gaps. Across a nonspeech gap, the slopes inherited from neighboring speech intervals may differ, and we seek to connect them with as little temporal-rate variation as possible. Since $h ^ { \prime \prime } ( t ) = d h ^ { \prime } ( t ) / d t$ measures the change in the read-map slope, $\dot { h ^ { \prime \prime } } ( t ) = \dot { 0 }$ represents the ideal case of no rate variation. We therefore minimize its accumulated squared magnitude under the boundary constraints.

Consider the interior gap between utterances u and $u + 1$ bounded by $( d _ { 1 } ^ { u } , s _ { 1 } ^ { u } )$ and $\dot { ( } d _ { 0 } ^ { u + 1 } , s _ { 0 } ^ { u + 1 } )$ . Its boundary slopes are inherited directly from the adjacent affine speech maps as $\kappa _ { u }$ and $\kappa _ { u + 1 }$ . The resulting variational problem is:

$$
\begin{array} { r l } & { h _ { u , u + 1 } ^ { \star } = \arg \operatorname* { m i n } _ { h } \ \int _ { d _ { 1 } ^ { u } } ^ { d _ { 0 } ^ { u + 1 } } \left( h ^ { \prime \prime } ( t ) \right) ^ { 2 } d t , } \\ & { \mathrm { s . t . } \quad h ( d _ { 1 } ^ { u } ) = s _ { 1 } ^ { u } , \quad h ( d _ { 0 } ^ { u + 1 } ) = s _ { 0 } ^ { u + 1 } , } \\ & { \quad h ^ { \prime } ( d _ { 1 } ^ { u } ) = \kappa _ { u } , \quad h ^ { \prime } ( d _ { 0 } ^ { u + 1 } ) = \kappa _ { u + 1 } . } \end{array}\tag{8}
$$

This optimization yields the unique cubic solution

$$
\begin{array} { c } { h _ { u , u + 1 } ^ { \star } ( t ) = \displaystyle \sum _ { r = 0 } ^ { 3 } a _ { u , r } \big ( t - d _ { 1 } ^ { u } \big ) ^ { r } , } \\ { t \in [ d _ { 1 } ^ { u } , d _ { 0 } ^ { u + 1 } ] , } \end{array}\tag{9}
$$

where the coefficients $\{ a _ { u , r } \} _ { r = 0 } ^ { 3 }$ are uniquely determined by the four boundary constraints in Eq. (8). Thus, each interior gap uses a cubic read map that smoothly connects the adjacent linear speech maps.

## SpeechShift Benchmark

SpeechShift is a benchmark for interval-1evel speech scheduling in text-to-audio-visual generation. Each sample follows the task representation $( c , \bar { \mathcal { U } } , \{ \mathcal { T } ^ { u } \} _ { u = 1 } ^ { N } )$ and contains a scene prompt, an ordered set of quoted utterances, their speaker roles and target intervals, and metadata describing the scene and challenge conditions.

In detail, the benchmark contains 400 prompts across 102 scenes and 600 utterance-level targets. It is balanced across four speaking structures defined by speaker and utterance counts: Single-Speaker Single-Utterance (SS-SU), Single-Speaker Multi-Utterance (SS-MU), Multi-Speaker Single-Utterance (MS-SU), and Multi-Speaker Multi-Utterance (MS-MU), with 100 prompts per structure. Within each structure, the prompts further form four condition groups by combining clean or interfering acoustics with static or challenging visual actions. The acoustic conditions include ambient noise, transient sounds, competing speech, and mixed interference, while the visual conditions include motion, occlusion, and action-coupled speaking. Target intervals are assigned solely from the text annotations, and the durations are adjusted according to utterance length to cover both fast and slow speech. For two-utterance cases, we use nonoverlapping target intervals with gaps ranging from short to long.

Evaluation Protocol SpeechShift evaluates two complementary objectives: placing each utterance within its target interval and retaining the overall quality of the generated audio-visual content. We assess them through interval controllability and generation quality, respectively.

Interval controllability. Let $\mathcal { E }$ denote the set of evaluated utterance instances. For each $u \in { \mathcal { E } } .$ , we transcribe and wordalign the generated clip to estimate the realized interval $\hat { \mathcal { T } } ^ { u } =$ $[ \hat { t } _ { s } ^ { u } , \hat { t } _ { e } ^ { u } ]$ corresponding to the target interval $\mathcal { T } ^ { u } = [ t _ { s } ^ { u } , t _ { e } ^ { u } ]$ . We define the absolute onset and offset errors as $e _ { s } ^ { u } = | \hat { t } _ { s } ^ { u } - t _ { s } ^ { u } |$ and $e _ { e } ^ { u } = | \hat { t } _ { e } ^ { u } - t _ { e } ^ { u } |$ , respectively, and report their means over $\mathcal { E }$ as m $\mathrm { E } _ { s }$ and mEe. The temporal Intersection-over-Union (IoU) is

$$
\mathrm { I o U } ^ { u } = \frac { \Big | \hat { T } ^ { u } \cap \mathcal { T } ^ { u } \Big | } { \Big | \hat { T } ^ { u } \cup \mathcal { T } ^ { u } \Big | } ,\tag{10}
$$

and the reported IoU is its mean over $u \in { \mathcal { E } } .$ We further report the hit rate

$$
\mathrm { H R } _ { \delta } = \frac { 1 } { | \mathcal { E } | } \sum _ { u \in \mathcal { E } } \mathbb { I } \left( \operatorname* { m a x } ( e _ { s } ^ { u } , e _ { e } ^ { u } ) \leq \delta \right) ,\tag{11}
$$

which measures the proportion of utterances whose two boundary errors both fall within the tolerance δ.

Generation quality. Alongside interval controllability, we report one representative metric for each quality dimension: WER for speech intelligibility, Production Quality (PQ) for audio naturalness (Tjandra et al. 2025), MUSIQ for no-reference visual quality (Ke et al. 2021), and LSE-C for audio-visual synchronization within each target interval (Chung and Zisserman 2017).

## Experiments

## Experimental Settings

We evaluate TimeSteer on SpeechShift using two representative joint audio-visual diffusion backbones, LTX-2 (HaCohen et al. 2025) and daVinci-MagiHuman (Chern et al. 2026), which adopt different cross-modal architectures, comparing against four representative training-free baselines: (i) Uncontrolled, standard sampling without temporal intervention; (ii) Textual Timing, which adds explicit timing instructions to the prompt (Xu et al. 2024; Huang et al. 2025); (iii) FreeAudio (Zhang et al. 2025), which binds each utterance to its target attention window through constrained attention aggregation; and (iv) Prompt Relay (Chen, Huang, and Liu 2026), which concentrates each prompt segment in its target region through a distance-dependent attention penalty. All methods use the same backbone, prompts, target intervals, and inference settings. Unless otherwise stated, we generate 5-second audio-visual clips with seed 42. The implementation details are provided in the supplementary material.

## Main Results

Table 1 shows that TimeSteer substantially improves interval controllability over Uncontrolled sampling on both backbones. On LTX-2, $\mathrm { { H R } _ { 0 . 2 } }$ increases from 0.21 to 0.73 and IoU from 0.63 to 0.87; on daVinci-MagiHuman, $\mathrm { { H R } _ { 0 . 2 } }$ increases from 0.09 to 0.53 and IoU from 0.63 to 0.79. TimeSteer also achieves the lowest onset and offset errors on both architectures. Compared with the other control methods, Textual Timing and FreeAudio provide limited or inconsistent improvements, while Prompt Relay achieves better control but remains behind TimeSteer across all four controllability metrics. Meanwhile, TimeSteer maintains generation-quality metrics comparable to uncontrolled sampling. Overall, the results demonstrate strong and consistent interval controllability across both architectures without a substantial loss in generation quality.

## Ablation Study

We conduct three ablation studies to examine source-span localization, the choice of remapping space, and the regionaware read-map design. Each study is evaluated on 50 randomly sampled cases from SpeechShift.

Effect of Source Span Localization We examine four ways to localize the model-implied source span during denoising. At each step n, the current latent $x _ { n }$ remains corrupted by noise, while $\hat { x } _ { 0 } ^ { n }$ is the model's estimate of the final clean latent. Attn@xn feeds this clean estimate back into the transformer and localizes the quoted utterance from its text-to-audio attention weights, which is the strategy used by TimeSteer. $\mathbf { A t t n } @ x _ { n }$ instead reads the attention produced by the standard forward pass on the noisy latent. As alternatives that do not use attention, Decode@ $\hat { x } _ { 0 } ^ { n }$ and $\mathbf { D e c o d e } @ x _ { n }$ decode the corresponding audio latent into an intermediate waveform and estimate the source span from its energy-based speech activity.

As shown in Fig. 3, Attn@ $\hat { x } _ { 0 } ^ { n }$ achieves the highest interval IoU and the lowest onset and offset errors. $\tan @ x _ { n }$ retains comparable onset accuracy but produces less reliable offsets, showing that the clean-latent estimate exposes sharper utterance boundaries. Both decode-based variants perform substantially worse because speech activity is not yet stably formed in intermediate waveforms. Attn@n therefore provides the most reliable source span before decoding, with only a modest latency overhead, and is used throughout TimeSteer.

![](images/c5d139e0c37b1f19edb152462d2d503d729a6c08e5c319c1eb5e8b701873b046.jpg)

![](images/4ef8e522c5dba914d58e3028c1c44f84f3a208e437a2901c118fe48abb29833a.jpg)

![](images/e57f3339d54ce6bf8c5de88d74efaf531ed9b87116a1a8085925a61b4cf6ce71.jpg)

![](images/b748847985da1a584737b9c0d54cc3b64ddbc06e2c3c4879eb743a42974763ed.jpg)  
Figure 3: Source-span localization ablation. Mean performance of four localization strategies; error bars show standard deviations across the evaluation subset and steps.

Effect of Remapping Space We compare four intervention spaces to determine where temporal remapping should be applied during generation. Remap $@ \hat { x } _ { 0 } ^ { n }$ , the strategy used by TimeSteer, applies the read map to the predicted clean audio-visual latent and converts the remapped estimate back into the update required by the original sampler. Remap $@ v _ { n }$ applies the same map directly to the predicted velocity before the sampler update, whereas Remap@ $x _ { n }$ remaps the updated noisy latent passed to the next denoising step. Posthoc performs no intervention during denoising and instead remaps the final audio-visual latent once before decoding.

As shown in Fig. 4, Remap@xn achieves the highest $\mathrm { H R } _ { 0 . 2 }$ and the lowest WER. The representative waveforms provide complementary qualitative evidence: Remap@ $\cdot \hat { x } _ { 0 } ^ { n }$ produces well-formed speech activity within the shaded target intervals while maintaining a comparatively clean background, whereas the other three strategies all exhibit varying degrees of noisy residuals. Directly remapping $v _ { n }$ or $x _ { n }$ further disrupts the denoising trajectory, producing incomplete or nearly silent utterances and substantially degrading transcription accuracy. Post-hoc remapping largely preserves the generated content but cannot reliably relocate it, because no subsequent denoising steps remain to consolidate the new temporal arrangement or suppress the introduced artifacts. In contrast, remapping $\hat { x } _ { 0 } ^ { n }$ presents the sampler with a rescheduled clean-latent prediction at every step, allowing the relocated speech to be progressively refined within its target interval.

Effect of Read-Map Design The region-aware read map matches each region with the geometry suited to its temporal role: a linear map applies a uniform rate adjustment within each utterance, preserving its internal temporal structure, while cubic bridges smoothly accommodate the necessary rate transitions across non-speech gaps. To test this regionaware design, we compare it with two alternatives that apply one rule throughout the timeline. Piecewise-linear connects every consecutive source-destination boundary pair with a line segment. It therefore maintains a constant slope within each segment but permits abrupt slope changes at speech boundaries. Global PCHIP (Fritsch and Carlson 1980) instead constructs a shape-preserving monotone cubic Hermite interpolant through all boundary pairs. It maintains slope continuity across the full timeline but does not constrain the slope to remain constant within speech intervals.

<table><tr><td></td><td colspan="4">Interval Controllability</td><td>Semantic Fidelity</td><td colspan="2">Perceptual Quality</td><td>Cross-modal</td></tr><tr><td>Method</td><td> $\mathbf { H R } _ { 0 . 2 } \uparrow$ </td><td>IoU↑</td><td> $m E _ { s } \left( \mathbf { s } \right) \downarrow$ </td><td> $m E _ { e } \left( \mathbf { s } \right) \downarrow$ </td><td>WER↓</td><td>PQ↑</td><td>MUSIQ↑</td><td>LSE-C↑</td></tr><tr><td colspan="9">LTX-2</td></tr><tr><td>Uncontrolled</td><td>0.21</td><td>0.63</td><td>0.56</td><td>0.54</td><td>0.05</td><td>5.36</td><td>52.26</td><td>3.13</td></tr><tr><td>Textual Timing</td><td>0.19</td><td>0.60</td><td>0.59</td><td>0.59</td><td>0.07</td><td>5.36</td><td>52.26</td><td>3.18</td></tr><tr><td>FreeAudio</td><td>0.19</td><td>0.54</td><td>0.49</td><td>0.67</td><td>0.35</td><td>5.92</td><td>52.55</td><td>2.92</td></tr><tr><td>Prompt Relay</td><td>0.31</td><td>0.68</td><td>0.27</td><td>0.35</td><td>0.20</td><td>5.52</td><td>52.00</td><td>3.06</td></tr><tr><td>TimeŠteer (Öurs)</td><td>0.73</td><td>0.87</td><td>0.11</td><td>0.15</td><td>0.07</td><td>5.32</td><td>52.51</td><td>3.10</td></tr><tr><td colspan="9">daVinci-MagiHuman</td></tr><tr><td>Uncontrolled</td><td>0.09</td><td>0.63</td><td>0.63</td><td>0.49</td><td>0.06</td><td>5.53</td><td>69.36</td><td>3.67</td></tr><tr><td>Textual Timing</td><td>0.08</td><td>0.61</td><td>0.64</td><td>0.53</td><td>0.09</td><td>5.50</td><td>68.98</td><td>3.87</td></tr><tr><td>FreeAudio</td><td>0.01</td><td>0.16</td><td>1.04</td><td>1.40</td><td>0.95</td><td>5.24</td><td>66.80</td><td>1.63</td></tr><tr><td>Prompt Relay</td><td>0.21</td><td>0.68</td><td>0.39</td><td>0.32</td><td>0.29</td><td>5.61</td><td>70.87</td><td>3.66</td></tr><tr><td>TimeSteer (Ours)</td><td>0.53</td><td>0.79</td><td>0.19</td><td>0.18</td><td>0.08</td><td>5.42</td><td>69.08</td><td>3.56</td></tr></table>

Table 1: Main results. SpeechShift comparison with training-free baselines on two joint audio-visual diffusion backbones. Complete quality results appear in the supplementary material. Best results are in bold. ↑/↓ indicates higher/lower is better.

![](images/da6336cc6b2810a13085c9d9e3a037830830c5fe7121ce121b35552b2a161b2c.jpg)  
Figure 4: Remapping-space ablation. Top: $\mathrm { H R } _ { 0 . 2 }$ and WER for different intervention spaces. Bottom: representative decoded waveforms; shading marks target intervals.

As shown in Table 2, our region-aware design consistently outperforms both alternatives in scheduling accuracy. It achieves a Span Win of $6 7 . 9 \% ,$ more than three times the 19.4% obtained by Global PCHIP, while also attaining the highest $\mathrm { H R } _ { 0 . 2 }$ and temporal IoU. This consistent per-span advantage supports assigning distinct geometries to speech intervals and non-speech gaps.

<table><tr><td>Read-map design</td><td> $\mathbf { H R } _ { 0 . 2 } \uparrow$ </td><td>WER↓</td><td>IoU↑</td><td>Span Win↑</td></tr><tr><td>Piecewise-linear</td><td>48.6%</td><td>7.88%</td><td>69.8%</td><td>12.7%</td></tr><tr><td>Global PCHIP</td><td>52.3%</td><td>8.92%</td><td>75.8%</td><td>19.4%</td></tr><tr><td>Region-Aware (Ours)</td><td>69.8%</td><td>8.05%</td><td>84.3%</td><td>67.9%</td></tr></table>

Table 2: Read-map design ablation. IoU measures temporal overlap, while Span Win indicates how often each geometry aligns best. All values are percentages; best results are bold.

## Conclusion

Joint audio-visual diffusion models can synthesize coupled speech and visual articulation, yet their temporal organization remains an implicit model decision. We show that such timing can be exposed and redirected during generation. Specifically, we uncover that a timing-sensitive crossattention head reveals each utterance's source span, while the predicted clean latent contains the coupled content required for temporal relocation. Building on these discoveries, TimeSteer localizes each source span and transfers its content to any user-specified target interval during denoising through a region-aware read map, without retraining the generator. We further introduce SpeechShift, the first benchmark for interval-level speech scheduling in joint audio-visual generation, covering diverse speaking structures. Across LTX-2 and daVinci-MagiHuman, TimeSteer substantially improves interval controllability over training-free baselines while retaining competitive generation quality, and our ablations validate its key localization and remapping designs. Together, TimeSteer and SpeechShift establish a practical method and evaluation foundation for inference-time speech scheduling. We hope our work will inform future research on temporal control in audio-visual generation.

## References

Aneja, S.; Thies, J.; Dai, A.; and Nießner, M. 2024. FaceTalk: Audio-Driven Motion Diffusion for Neural Parametric Head Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 21263–21273.

Chen, G.; Huang, Z.; and Liu, Z. 2026. Prompt Relay: Inference-Time Temporal Control for Multi-Event Video Generation. arXiv preprint arXiv:2604.10030.

Chen, Z.; Seetharaman, P.; Russell, B.; Nieto, O.; Bourgin, D.; Owens, A.; and Salamon, J. 2025a. Video-Guided Foley Sound Generation with Multimodal Controls. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 18770–18781.

Chen, Z.; et al. 2025b. Unison: Harmonizing Motion, Speech, and Sound for Human-Centric Audio-Video Generation. arXiv preprint arXiv:2605.08729.

Cheng, H. K.; Ishii, M.; Hayakawa, A.; Shibuya, T.; Schwing, A.; and Mitsufuji, Y. 2025. MMAudio: Taming Multimodal Joint Training for High-Quality Video-to-Audio Synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 28901–28911.

Chern, E.; Teng, H.; Sun, H.; Wang, H.; Pan, H.; Jia, H.; Su, J.; Li, J.; Yu, J.; Liu, L.; et al. 2026. Speed by Simplicity: A Single-Stream Architecture for Fast Audio-Video Generative Foundation Model. arXiv preprint arXiv:2603.21986.

Chung, J. S.; and Zisserman, A. 2017. Out of Time: Automated Lip Sync in the Wild. In Asian Conference on Computer Vision Workshops.

Corona, E.; Zanfir, A.; Bazavan, E. G.; Kolotouros, N.; Alldieck, T.; and Sminchisescu, C. 2025. VLOGGER: Multimodal Diffusion for Embodied Avatar Synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 15896–15908.

Cui, J.; Li, H.; Yao, Y.; Zhu, H.; Shang, H.; Cheng, K.; Zhou, H.; Zhu, S.; and Wang, J. 2025. Hallo2: Long-Duration and High-Resolution Audio-Driven Portrait Image Animation. In International Conference on Learning Representations.

Fritsch, F. N.; and Carlson, R. E. 1980. Monotone Piecewise Cubic Interpolation. SIAM Journal on Numerical Analysis, 17(2): 238–246.

Guan, J.; Wang, K.; Xu, Z.; Yang, Q.; Sun, Y.; He, S.; Liang, B.; Cao, Y.; Li, Y.; Feng, H.; Ding, E.; Wang, J.; Zhao, Y.; Zhou, H.; and Liu, Z. 2025. AudCast: Audio-Driven Human Video Generation by Cascaded Diffusion Transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 10678–10689.

HaCohen, Y.; Brazowski, B.; Chiprut, N.; Bitterman, Y.; Kvochko, A.; Berkowitz, A.; Shalem, D.; Lifschitz, D.; Moshe, D.; Porat, E.; et al. 2025. LTX-2: Efficient Joint Audio-Visual Foundation Model. arXiv preprint arXiv:2601.03233.

Huang, R.; et al. 2024a. AudioComposer: Towards Fine-Grained Text-to-Audio Generation. arXiv preprint arXiv:2402.10012.

Huang, R.; et al. 2024b. Make-An-Audio 2: Temporal-Enhanced Text-to-Audio Generation. arXiv preprint arXiv:2305.18474.

Huang, Y.; et al. 2025. TempoControl: Temporal Attention Guidance for Text-to-Video Models. arXiv preprint arXiv:2510.02226.

Ke, J.; Wang, Q.; Wang, Y.; Milanfar, P.; and Yang, F. 2021. MUSIQ: Multi-Scale Image Quality Transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 5148–5157.

Kreuk, F.; Synnaeve, G.; Polyak, A.; Singer, U.; Défossez, A.; Copet, J.; Parikh, D.; Taigman, Y.; and Adi, Y. 2023. Audio-Gen: Textually Guided Audio Generation. In International Conference on Learning Representations.

Kushwaha, S. S.; and Tian, Y. 2025. VinTAGe: Joint Video and Text Conditioning for Holistic Audio Generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 13529–13539.

Lee, J.; et al. 2025. Syncphony: Synchronized Audioto-Video Generation with Diffusion Transformers. arXiv preprint arXiv:2509.21893.

Li, Y.; et al. 2025a. PicoAudio2: Temporally Strong Textto-Audio Generation with Context and Timestamp Matrix. arXiv preprint arXiv:2509.00683.

Li, Z.; et al. 2025b. Harmony: Harmonizing Audio and Video Generation through Cross-Task Synergy. arXiv preprint arXiv:2511.21579.

Lipman, Y.; Chen, R. T. Q.; Ben-Hamu, H.; Nickel, M.; and Le, M. 2023. Flow Matching for Generative Modeling. In International Conference on Learning Representations.

Liu, H.; Chen, Z.; Yuan, Y.; Mei, X.; Liu, X.; Mandic, D.; Wang, W.; and Plumbley, M. D. 2023. AudioLDM: Text-to-Audio Generation with Latent Diffusion Models. In Proceedings of the International Conference on Machine Learning.

Liu, X.; Gong, C.; and Liu, Q. 2023. Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. In International Conference on Learning Representations.

Liu, Z.; et al. 2026a. AVGen-Bench: A Task-Driven Benchmark for Multi-Granular Evaluation of Text-to-Audio-Video Generation. arXiv preprint arXiv:2604.08540.

Liu, Z.; et al. 2026b. SyncDPO: Enhancing Temporal Synchronization in Video-Audio Joint Generation via Preference Learning. arXiv preprint arXiv:2605.12179.

Low, C.; Wang, W.; and Katyal, C. 2025. Ovi: Twin Backbone Cross-Modal Fusion for Audio-Video Generation. arXiv preprint arXiv:2510.01284.

Luo, S.; Yan, C.; Hu, C.; and Zhao, H. 2023. Diff-Foley: Synchronized Video-to-Audio Synthesis with Latent Diffusion Models. In Advances in Neural Information Processing Systems, volume 36.

Ma, Y.; et al. 2026. Inference-Time Scaling for Joint Audio-Video Generation. arXiv preprint arXiv:2606.03183.

Peng, Z.; Hu, W.; Shi, Y.; Zhu, X.; Zhang, X.; Zhao, H.; He, J.; Liu, H.; and Fan, Z. 2024. SyncTalk: The Devil Is in the Synchronization for Talking Head Synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 666–676.

Polyak, A.; Zohar, A.; Brown, A.; Tjandra, A.; Sinha, A.; Lee, A.; Vyas, A.; Shi, B.; Ma, C.-Y.; Chuang, C.-Y.; et al. 2024. Movie Gen: A Cast of Media Foundation Models. arXiv preprint arXiv:2410.13720.

Prajwal, K. R.; Mukhopadhyay, R.; Namboodiri, V. P.; and Jawahar, C. V. 2020. A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild. In Proceedings of the ACM International Conference on Multimedia.

Ruan, Y.; Liu, J.; et al. 2023. MM-Diffusion: Learning Multi-Modal Diffusion Models for Joint Audio and Video Generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Ruan, Y.; et al. 2023. AADiff: Audio-Aligned Video Synthesis with Text-to-Image Diffusion. arXiv preprint arXiv:2305.04001.

Shen, S.; Zhao, W.; Meng, Z.; Li, W.; Zhu, Z.; Zhou, J.; and Lu, J. 2023. DiffTalk: Crafting Diffusion Models for Generalized Audio-Driven Portraits Animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 1982–1991.

Song, Q.; Guo, Z.; He, Y.; Wang, Z.; He, Z.; Zhang, C.; Jiang, C.; and Li, X. 2026. SyncDIT: Audio-Visual Aligned Video Generation with Audio Synchronization Feature. Vicinagearth, 3: 5.

Team, B. E. 2026. NAVA: Native Audio-Visual Alignment for Generation. arXiv preprint arXiv:2605.30073.

Tjandra, A.; Wu, Y.-C.; Guo, B.; Hoffman, J.; Ellis, B.; Vyas, A.; Shi, B.; Chen, S.; Le, M.; Zacharov, N.; Wood, C.; Lee, A.; and Hsu, W.-N. 2025. Meta Audiobox Aesthetics: Unified Automatic Quality Assessment for Speech, Music, and Sound. arXiv preprint arXiv:2502.05139.

Wang, X.; Song, R.; Li, C.; Cheng, X.; Li, B.; Wu, Y.; Wang, Y.; Xu, H.; and Wang, Y. 2025a. Animate and Sound an Image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 23369–23378.

Wang, Y.; Chen, Y.; Niu, D.; Zhang, Y.; Wang, H.; Chen, S.; et al. 2025b. JavisDiT: Joint Audio-Video Diffusion Transformer with Hierarchical Spatio-Temporal Prior Synchronization. arXiv preprint arXiv:2503.23377.

Wang, Y.; et al. 2024. SyncFlow: Synchronized Joint Audio-Video Generation with Dual Diffusion Transformers. arXiv preprint arXiv:2412.15220.

Wang, Y.; et al. 2025c. ControlAudio: Tackling Text-Guided, Timing-Indicated and Intelligible Audio Generation via Progressive Diffusion Modeling. arXiv preprint arXiv:2510.08878.

Wang, Z.; et al. 2026. MOVA: Towards Scalable and Synchronized Video-Audio Generation. arXiv preprint arXiv:2602.08794.

Weng, S.; Zheng, H.; Chang, Z.; Li, S.; Shi, B.; and Wang, X. 2025. Audio-Sync Video Generation with Multi-Stream Temporal Control. arXiv preprint arXiv:2506.08003.

Xing, Y.; He, Y.; Tian, Z.; Wang, X.; and Chen, Q. 2024. Seeing and Hearing: Open-Domain Visual-Audio Generation with Diffusion Latent Aligners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 7151–7161.

Xu, H.; et al. 2024. PicoAudio: Enabling Precise Timestamp and Frequency Controllability of Audio Events in Text-to-Audio Generation. arXiv preprint arXiv:2407.02869.

Yariv, G.; Gat, I.; Benaim, S.; Wolf, L.; Schwartz, I.; and Adi, Y. 2023. Diverse and Aligned Audio-to-Video Generation via Text-to-Video Model Adaptation. arXiv preprint arXiv:2309.16429.

Yu, Z.; Yin, Z.; Zhou, D.; Wang, D.; Wong, F.; and Wang, B. 2023. Talking Head Generation with Probabilistic Audioto-Visual Diffusion Priors. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 7645–7655.

Zhang, W.; Cun, X.; Wang, X.; Zhang, Y.; Shen, X.; Guo, Y.; Shan, Y.; and Wang, F. 2023. SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Zhang, Y.; et al. 2025. FreeAudio: Training-Free Timing Planning for Controllable Long-Form Text-to-Audio Generation. arXiv preprint arXiv:2507.08557.

Zhang, Y.; et al. 2026. MMControl: Unified Multi-Modal Control for Joint Audio-Video Generation. arXiv preprint arXiv:2604.19679.

## Appendix

This appendix is organized into three sections. We first present the complete TimeSteer sampling procedure and then derive Region-Aware Latent Remapping for speech intervals and non-speech gaps. Finally, we document the experimental setup and method implementations, explain how the localization components are selected, and provide expanded quantitative and qualitative results.

## TimeSteer Procedure

Algorithm 1 summarizes how TimeSteer is integrated into standard flow-matching sampling. At each denoising step, the model first predicts the velocity $v _ { n }$ and recovers the corresponding clean latent $\hat { x } _ { 0 } ^ { n }$ . TimeSteer then localizes the source spans and remaps $\hat { x } _ { 0 } ^ { n }$ before converting it back to the velocity used by the original flow-matching update.

Algorithm 1 TimeSteer within flow-matching sampling   
Require: Frozen model ${ \mathcal { G } } _ { \theta } .$ prompt $c ,$ and target intervals   
$\{ \mathcal { T } ^ { u } \} _ { u = 1 } ^ { N }$   
Require: Initial noisy latent $x _ { M }$ and noise schedule   
$\{ \sigma _ { n } \} _ { n = 0 } ^ { M }$   
Ensure: Generated audio $A$ and video $V$   
1: for $n = M , \ldots , 1$ do   
2: $v _ { n }  \mathcal { G } _ { \boldsymbol { \theta } } ( x _ { n } , c , \sigma _ { n } )$   
3: $\hat { x } _ { 0 } ^ { n } \gets x _ { n } - \sigma _ { n } v _ { n }$   
4: Source Span Localization: estimate $\{ S ^ { u } \} _ { u = \cdot } ^ { N }$ from   
$\hat { x } _ { 0 } ^ { n }$ and c   
5: Region-Aware Latent Remapping: construct h from   
$\{ ( \breve { S } ^ { u } , \mathcal T ^ { u } ) \} _ { u = 1 } ^ { N }$   
6: for each destination coordinate t do   
7: $k _ { t } ^ { - } \gets \lfloor h ( t ) \rfloor , \quad k _ { t } ^ { + } \gets \lceil h ( t ) \rceil$   
8: $\lambda _ { t } \gets h ( t ) - k _ { t } ^ { - }$   
9: $\tilde { x } _ { 0 } ^ { n } ( t ) \gets ( 1 - \bar { \lambda } _ { t } ) \hat { x } _ { 0 } ^ { n } ( k _ { t } ^ { - } ) + \lambda _ { t } \hat { x } _ { 0 } ^ { n } ( k _ { t } ^ { + } )$   
10: end for   
11: $\tilde { v } _ { n } \gets ( x _ { n } - \tilde { x } _ { 0 } ^ { n } ) / \sigma _ { n }$   
12: $x _ { n - 1 }  x _ { n } + ( \sigma _ { n - 1 } - \sigma _ { n } ) \tilde { v } _ { n }$   
13: end for   
14: (A, V) ← Decode(x0)   
15: return (A, V)

The two TimeSteer operations in Algorithm 1 correspond to the successive stages described in the main paper. Specifically, Source Span Localization estimates $\{ S ^ { u } \} _ { u = 1 } ^ { \bar { N } }$ from the text-to-audio cross-attention of $\hat { x } _ { 0 } ^ { n }$ , whereas Region-Aware Latent Remapping converts each target interval $\mathcal { T } ^ { u }$ to its destination speech interval and constructs the destination-tosource read map h. Once h is obtained, the same neighboring indices and interpolation weights are applied to the audio and video branches of $\hat { x } _ { 0 } ^ { n }$ , preserving their temporal correspondence.

## Derivation of Region-Aware Latent Remapping

Region-Aware Latent Remapping is designed around two distinct temporal requirements: preserving the internal speech structure within each utterance and smoothly connecting neighboring utterances across non-speech gaps. We first establish that the slope of the destination-to-source read map determines the local speaking-rate ratio. Based on this interpretation, we then derive the affine speech map that minimizes speaking-rate distortion and the cubic gap map that minimizes variation in the read-map slope.

## Speaking-Rate Interpretation of the Read-Map Slope

Let $q _ { \mathrm { s r c } } ( t )$ and $q _ { \mathrm { d s t } } ( t )$ denote the cumulative speech content rendered up to coordinate t on the source and destination timelines, respectively. Their derivatives define the corresponding local speaking rates:

$$
r _ { \mathrm { s r c } } ( t ) = \frac { d q _ { \mathrm { s r c } } ( t ) } { d t } , \qquad r _ { \mathrm { d s t } } ( t ) = \frac { d q _ { \mathrm { d s t } } ( t ) } { d t } .\tag{12}
$$

The read map $h : [ 0 , T - 1 ] \to [ 0 , T - 1 ]$ specifies the source coordinate sampled at each destination coordinate, and therefore relates the two content functions as

$$
q _ { \mathrm { d s t } } ( t ) = q _ { \mathrm { s r c } } ( h ( t ) ) .\tag{13}
$$

Applying the chain rule gives

$$
r _ { \mathrm { d s t } } ( t ) = r _ { \mathrm { s r c } } ( h ( t ) ) h ^ { \prime } ( t ) .\tag{14}
$$

For a nonzero source speaking rate, this gives

$$
h ^ { \prime } ( t ) = { \frac { r _ { \mathrm { d s t } } ( t ) } { r _ { \mathrm { s r c } } ( h ( t ) ) } } .\tag{15}
$$

Thus, $h ^ { \prime } ( t )$ gives the local multiplicative change from the original speaking rate to the output speaking rate. When $h ^ { \prime } ( \bar { t } ) = 1$ , the output speaking rate equals the original speaking rate. Since the same read map is applied to the audio and video latents, the associated visual articulation undergoes the same temporal scaling.

## Minimal-Distortion Remapping for Speech Intervals

Within each speech interval, we aim to preserve the original speaking rate, corresponding to $h ^ { \prime } ( t ) \bar { = } 1$ . For utterance $u ,$ we therefore solve

$$
\begin{array} { l } { { \displaystyle h _ { u } ^ { \star } = \arg \operatorname* { m i n } _ { h } \int _ { d _ { 0 } ^ { u } } ^ { d _ { 1 } ^ { u } } \left( h ^ { \prime } ( t ) - 1 \right) ^ { 2 } d t , } } \\ { { \mathrm { s . t . } \quad h ( d _ { 0 } ^ { u } ) = s _ { 0 } ^ { u } , \qquad h ( d _ { 1 } ^ { u } ) = s _ { 1 } ^ { u } . } } \end{array}\tag{16}
$$

The Euler-Lagrange equation provides the stationary condition for this functional. Applying it directly to the integrand in Eq. (16) gives

$$
\begin{array} { l } { { 0 = { \displaystyle \frac { \partial } { \partial h } } \big ( h ^ { \prime } ( t ) - 1 \big ) ^ { 2 } - { \displaystyle \frac { d } { d t } } \left[ { \frac { \partial } { \partial h ^ { \prime } } } \big ( h ^ { \prime } ( t ) - 1 \big ) ^ { 2 } \right] } } \\ { { = - { \displaystyle \frac { d } { d t } } \left[ 2 \big ( h ^ { \prime } ( t ) - 1 \big ) \right] = - 2 h ^ { \prime \prime } ( t ) . } } \end{array}\tag{17}
$$

The first partial derivative is zero because the integrand depends on $\bar { h } ^ { \prime } ( t )$ but not directly on $h ( t )$ . Equation (17) therefore requires

$$
h ^ { \prime \prime } ( t ) = 0 , \qquad t \in [ d _ { 0 } ^ { u } , d _ { 1 } ^ { u } ] .\tag{18}
$$

Integrating twice shows that every stationary solution has the affine form

$$
h ( t ) = \alpha _ { u } t + \beta _ { u } .\tag{19}
$$

We determine the two constants from the endpoint constraints:

$$
\alpha _ { u } d _ { 0 } ^ { u } + \beta _ { u } = s _ { 0 } ^ { u } , \qquad \alpha _ { u } d _ { 1 } ^ { u } + \beta _ { u } = s _ { 1 } ^ { u } .\tag{20}
$$

Subtracting the first equation from the second gives

$$
\alpha _ { u } = \frac { s _ { 1 } ^ { u } - s _ { 0 } ^ { u } } { d _ { 1 } ^ { u } - d _ { 0 } ^ { u } } = \kappa _ { u } , \qquad \beta _ { u } = s _ { 0 } ^ { u } - \kappa _ { u } d _ { 0 } ^ { u } .\tag{21}
$$

Substituting these constants into Eq. (19) yields

$$
h _ { u } ^ { \star } ( t ) = s _ { 0 } ^ { u } + \kappa _ { u } \big ( t - d _ { 0 } ^ { u } \big ) , \qquad t \in [ d _ { 0 } ^ { u } , d _ { 1 } ^ { u } ] .\tag{22}
$$

It remains to determine the type of this stationary solution. The second derivative of the integrand with respect to $h ^ { \prime }$ is

$$
{ \frac { \partial ^ { 2 } } { \partial ( h ^ { \prime } ) ^ { 2 } } } \big ( h ^ { \prime } ( t ) - 1 \big ) ^ { 2 } = 2 > 0 .\tag{23}
$$

Thus, the integrand is strictly convex in h'. Together with the affine endpoint constraints, this makes the complete objective strictly convex over the feasible read maps. Consequently, the stationary affine map in Eq. (22) is the unique global minimizer of Eq. (16).

## Curvature Bridging for Non-speech Gaps

Across a non-speech gap, the speaking-rate ratios inherited from neighboring speech intervals may differ. As established above, $h ^ { \prime } ( t ) $ is the local speaking-rate ratio, i.e., the output speaking rate divided by the original speaking rate, so $h ^ { \prime \prime } ( \dot { t } ) = d \check { h } ^ { \prime } ( t ) / d t$ measures how rapidly this ratio changes over destination time. We use the gap to transition smoothly between the two ratios by minimizing the accumulated squared magnitude of $h ^ { \prime \prime } \bar { ( t ) }$ under the boundary constraints.

Consider a nonempty interior gap between utterances u and $u + 1$ , where $d _ { 1 } ^ { u } < \bar { d } _ { 0 } ^ { u + 1 }$ . Its boundary slopes are inherited directly from the adjacent affine speech maps as $\kappa _ { u }$ and $\kappa _ { u + 1 }$ . Following the formulation in the main paper, we solve

$$
\begin{array} { r l } & { h _ { u , u + 1 } ^ { \star } = \arg \operatorname* { m i n } _ { h } \quad \int _ { d _ { 1 } ^ { u } } ^ { d _ { 0 } ^ { u + 1 } } \left( h ^ { \prime \prime } ( t ) \right) ^ { 2 } d t , } \\ & { \quad \quad \mathrm { s . t . } \quad h ( d _ { 1 } ^ { u } ) = s _ { 1 } ^ { u } , \quad h ( d _ { 0 } ^ { u + 1 } ) = s _ { 0 } ^ { u + 1 } , } \\ & { \quad \quad h ^ { \prime } ( d _ { 1 } ^ { u } ) = \kappa _ { u } , \quad h ^ { \prime } ( d _ { 0 } ^ { u + 1 } ) = \kappa _ { u + 1 } . } \end{array}\tag{24}
$$

To derive the stationary condition, we perturb a candidate read map h in an arbitrary direction η:

$$
\begin{array} { r } { h _ { \epsilon } ( t ) = h ( t ) + \epsilon \eta ( t ) , } \end{array}\tag{25}
$$

where € is a scalar controlling the perturbation magnitude and $\eta ( t )$ specifies its shape over the gap. The perturbed map must satisfy the same four boundary constraints as h. Substituting Eq. (25) into these constraints shows that η must satisfy

$$
\begin{array} { r l } { { \eta ( d _ { 1 } ^ { u } ) = 0 , } } & { { { } \quad \eta ( d _ { 0 } ^ { u + 1 } ) = 0 , } } \\ { { \eta ^ { \prime } ( d _ { 1 } ^ { u } ) = 0 , } } & { { { } \quad \eta ^ { \prime } ( d _ { 0 } ^ { u + 1 } ) = 0 . } } \end{array}\tag{26}
$$

The first line keeps the two endpoint values of the read map fixed, while the second keeps its two boundary slopes fixed. Any sufficiently smooth η satisfying these conditions is an admissible perturbation.

Let

$$
J [ h ] = \int _ { d _ { 1 } ^ { u } } ^ { d _ { 0 } ^ { u + 1 } } \left( h ^ { \prime \prime } ( t ) \right) ^ { 2 } d t\tag{27}
$$

denote the bending-energy objective. Under the perturbation in Eq. (25), it becomes

$$
J [ h _ { \epsilon } ] = \int _ { d _ { 1 } ^ { u } } ^ { d _ { 0 } ^ { u + 1 } } \left( h ^ { \prime \prime } ( t ) + \epsilon \eta ^ { \prime \prime } ( t ) \right) ^ { 2 } d t .\tag{28}
$$

If h is a minimizer, then $\epsilon = 0$ must be a stationary point of $J [ h _ { \epsilon } ]$ for every admissible direction η. Therefore,

$$
\left. { \frac { d J [ h _ { \epsilon } ] } { d \epsilon } } \right| _ { \epsilon = 0 } = 2 \int _ { d _ { 1 } ^ { u } } ^ { d _ { 0 } ^ { u + 1 } } h ^ { \prime \prime } ( t ) \eta ^ { \prime \prime } ( t ) d t = 0 .\tag{29}
$$

We now integrate the remaining integral by parts twice. Each integration transfers one derivative from η to h, and all boundary terms vanish because both η and $\eta ^ { \prime }$ are zero at the two endpoints. Therefore,

$$
\int _ { d _ { 1 } ^ { u } } ^ { d _ { 0 } ^ { u + 1 } } h ^ { \prime \prime } ( t ) \eta ^ { \prime \prime } ( t ) d t = \int _ { d _ { 1 } ^ { u } } ^ { d _ { 0 } ^ { u + 1 } } h ^ { ( 4 ) } ( t ) \eta ( t ) d t .\tag{30}
$$

Combining Eqs. (29) and (30) gives

$$
\int _ { d _ { 1 } ^ { u } } ^ { d _ { 0 } ^ { u + 1 } } h ^ { ( 4 ) } ( t ) \eta ( t ) d t = 0\tag{31}
$$

for every admissible perturbation η. This class includes arbitrary smooth perturbations supported within the interior of the gap. The fundamental lemma of the calculus of variations therefore gives the Euler-Lagrange condition

$$
h ^ { ( 4 ) } ( t ) = 0 .\tag{32}
$$

Therefore, every stationary solution is a polynomial of degree at most three.

To obtain its coefficients, define

$$
\begin{array} { c } { { L _ { u } = d _ { 0 } ^ { u + 1 } - d _ { 1 } ^ { u } , } } \\ { { \Delta _ { u } = s _ { 0 } ^ { u + 1 } - s _ { 1 } ^ { u } , } } \\ { { \xi = t - d _ { 1 } ^ { u } . } } \end{array}\tag{33}
$$

Writing

$$
h _ { u , u + 1 } ^ { \star } ( t ) = \sum _ { r = 0 } ^ { 3 } a _ { u , r } \xi ^ { r }\tag{34}
$$

and applying the four constraints in Eq. (24) gives

$$
\begin{array} { l } { \displaystyle a _ { u , 0 } = s _ { 1 } ^ { u } , } \\ { \displaystyle a _ { u , 1 } = \kappa _ { u } , } \\ { \displaystyle a _ { u , 2 } = \frac { 3 \Delta _ { u } } { L _ { u } ^ { 2 } } - \frac { 2 \kappa _ { u } + \kappa _ { u + 1 } } { L _ { u } } , } \\ { \displaystyle a _ { u , 3 } = - \frac { 2 \Delta _ { u } } { L _ { u } ^ { 3 } } + \frac { \kappa _ { u } + \kappa _ { u + 1 } } { L _ { u } ^ { 2 } } . } \end{array}\tag{35}
$$

These coefficients ensure that both h and $h ^ { \prime }$ match the adjacent affine speech maps at the two gap boundaries. The resulting cubic-polynomial bridge therefore makes the read map continuously differentiable across speech intervals and non-speech gaps.

Finally, the second variation is $\begin{array} { r } { 2 \int _ { d _ { 1 } ^ { u } } ^ { d _ { 0 } ^ { u + 1 } } ( \eta ^ { \prime \prime } ( t ) ) ^ { 2 } d t . } \end{array}$ which is strictly positive for every nonzero admissible perturbation. Indeed, $\dot { \eta } ^ { \prime \prime } = 0$ would make $\eta$ affine, and the homogeneous endpoint constraints would then force $\eta \equiv 0$ . The cubicpolynomial solution is therefore the unique global minimizer.

## Additional Experimental Details and Results

This section provides the information required to reproduce SpeechShift evaluation, specifies how each training-free baseline is implemented on the joint audio-visual backbones, and reports additional results.

## Reproducibility Details

Data specification and release. Each SpeechShift entry contains a scene prompt, an ordered list of quoted utterances, the speaker associated with each utterance, its begin-end target interval, and metadata specifying the scene, speaking structure, acoustic condition, and visual-action condition. The benchmark contains 400 prompts, 600 utterance-level targets, and 102 scenes, with 100 prompts in each of the four speaker-utterance structures described in the main paper. Upon publication, we will release SpeechShift and its construction metadata under a license permitting free use for research.

Code release. Upon publication, we will release the source code for TimeSteer, benchmark construction, baseline adaptation, and evaluation under a license permitting free use for research. The released implementation will document the correspondence between its principal stages and the equations and pseudocode in the paper.

Experimental setup. All experiments are conducted in Python 3.10 on NVIDIA RTX A6000 GPUs. Unless otherwise stated, the main comparison uses one run per method with a fixed random seed of 42; the cross-seed robustness study uses five seeds as specified below. All methods generate 5-second audio-visual clips: LTX-2 uses 24 FPS, whereas daVinci-MagiHuman uses 25 FPS. To accommodate the available GPU memory and compute budget, we use the FP8-quantized distilled configuration of each backbone with its eight-step sampler. Within each backbone, all methods use the same released checkpoint, prompts, target intervals, native inference resolution, numerical precision, and guidance scale. No model fine-tuning or test-time optimization is performed. The configurations introduced by each competing control method are listed below; all remaining backbone parameters retain their official defaults.

## Method Implementations

Uncontrolled The Uncontrolled baseline applies the backbone's standard sampling procedure to the original scene prompt. It receives no target-interval information and introduces no intervention during denoising.

Textual Timing Textual Timing communicates the target schedule only through the conditioning prompt. For each quoted utterance, its begin and end timestamps are converted into an explicit natural-language timing instruction and appended to the original scene description. For example, the original clause a mechanic says: "please sign here on the $l i n e ^ { , \cdot }$ becomes a mechanic says: "please sign here on the line" from 1.6 s to 3.4 s. Sampling otherwise remains unchanged, with no modification to attention or latent representations.

## FreeAudio

Original method. FreeAudio is a training-free framework for timing-controlled long-form text-to-audio generation. It divides a timing-annotated prompt into non-overlapping windows and rewrites the desired audio event in each window as a window-specific description. Its Decoupling and Aggregating Attention Control (DAAC) then binds each description to its window by suppressing cross-attention between the corresponding text tokens and audio queries outside that window. The window-controlled attention responses are subsequently aggregated with the original responses to balance temporal control and generation quality.

SpeechShift adaptation. SpeechShift represents each scheduling request as a quoted utterance paired with a target interval. For each utterance, we identify the first and last token indices of the quoted utterance and map the begin and end timestamps of its target interval to the corresponding audio and video latent coordinates. For LTX-2, DAAC is applied to both text-to-audio and text-to-video cross-attention. For daVinci-MagiHuman, the same constraint is applied within unified attention to the interactions between quoted-text keys and audio or video queries. In both architectures, the audio and video coordinates are derived from the same target timestamps, providing a common temporal constraint for the two modalities.

Implementation. For LTX-2, we apply DAAC at all eight denoising steps and in transformer blocks 20–30, using all attention heads. The DAAC mask is linearly ramped over four query-token positions at each target-window boundary. Let $A _ { \mathrm { b a s e } }$ denote the original attention probabilities and ADAAC the probabilities after applying the window constraint. Following FreeAudio's aggregation principle, we compute

$$
A _ { \mathrm { o u t } } = \alpha A _ { \mathrm { b a s e } } + ( 1 - \alpha ) A _ { \mathrm { D A A C } } ,\tag{36}
$$

where α controls the contribution of the original attention. We set $\alpha = 0 . 2$ and apply $A _ { \mathrm { o u t } }$ to both audio and video queries.

For daVinci-MagiHuman, we apply DAAC at every denoising step and in transformer blocks 10–39, using all attention heads. We use a two-query-token boundary ramp and set $\alpha = 0 , \mathrm { s o } A _ { \mathrm { o u t } } = A _ { \mathrm { D A A C } }$ . The additive DAAC bias applied to the attention logits is scaled by 1.5, controlling the strength of the window constraint. We also set the query/key (Q/K) steering strength to 1.5, which amplifies audio and video queries inside the target interval together with the keys of the corresponding quoted tokens.

## Prompt Relay

Original method. Prompt Relay is a training-free method for temporally controlling multiple events in text-to-video generation. It assigns each local prompt to a temporal segment while retaining a global prompt throughout the video. Its Boundary-Attention Decay subtracts a distancedependent penalty from cross-attention logits, allowing queries inside a segment to attend primarily to the corresponding local prompt while varying the penalty smoothly near segment boundaries. Specifically, the controlled attention is computed as

$$
\mathrm { A t t n } ( Q , K , V ) = \mathrm { s o f t m a x } \left( \frac { Q K ^ { \top } } { \sqrt { d } } - C \right) V ,\tag{37}
$$

where $Q , K$ , and $V$ are the query, key, and value matrices, d is the key dimension, and $\dot { C }$ is the Boundary-Attention Decay penalty. For a local prompt assigned to segment $s ,$ the penalty is

$$
C _ { i j } = \left\{ \begin{array} { l l } { \displaystyle \frac { \big ( \big | i - m _ { s } \big | - w _ { s } \big ) _ { + } ^ { 2 } } { 2 \sigma _ { s } ^ { 2 } } , } & { j \in { \mathcal K } _ { s } , } \\ { 0 , } & { j \notin { \mathcal K } _ { s } , } \end{array} \right.\tag{38}
$$

where i and $j$ index the temporal query and text key, $\kappa _ { s }$ contains the text keys of the local prompt, $m _ { s }$ is the midpoint of its assigned segment, $w _ { s }$ specifies the central region with no penalty, and $( z ) _ { + } = \operatorname* { m a x } ( z , 0 )$ . The scale $\sigma _ { s }$ controls how rapidly the penalty increases beyond this region. Thus, attention to the local prompt remains unrestricted near its assigned segment and is progressively suppressed as the temporal distance increases.

SpeechShift adaptation. SpeechShift contains a single scene prompt with one or more quoted utterances. We treat the token range of each quoted utterance as a local prompt and its target interval as the corresponding temporal segment, while the remaining scene description provides global context. The begin and end timestamps are mapped to both audio and video latent coordinates. For LTX-2, the resulting penalty is applied to text-to-audio and text-to-video cross-attention. For daVinci-MagiHuman, it is applied within unified attention to the interactions between quoted-text keys and audio or video queries. This assigns the same utterance-level schedule to both modalities.

Implementation. For LTX-2, we apply Boundary-Attention Decay at all eight denoising steps and in transformer blocks 10–40, using all attention heads. A separate penalty is constructed for each quoted token range and its target audio and video coordinates. The penalties vary smoothly with distance from the target interval. We set $\varepsilon = 0 . 1$ , where ε specifies the attention factor retained at the segment boundary and a smaller value produces stronger decay.

For daVinci-MagiHuman, we apply the penalty at every denoising step and in transformer blocks 10–39, using all attention heads. We set $\varepsilon = 0 . 0 5$ and multiply the attentionlogit penalty by 1.5. The pre-attention steering strength is also set to 1.5, controlling how strongly audio and video queries are downweighted according to the same temporal penalty.

## TimeSteer

Original method. Our TimeSteer is a training-free framework for inference-time speech scheduling in joint audiovisual diffusion models. It first performs Source Span Localization, which recovers each quoted utterance's modelimplied source interval from a timing-sensitive attention response of the predicted clean latent. Region-Aware Latent Remapping then constructs a destination-to-source read map that relocates the corresponding audio-visual latent content to the requested target interval. Affine speech segments preserve the internal temporal structure of each utterance, while cubic bridges connect neighboring segments smoothly across non-speech gaps. The remapped clean latent is returned to the original sampler, allowing subsequent denoising steps to refine the relocated content without retraining the backbone.

SpeechShift adaptation. Each SpeechShift input consists of a scene prompt and one or more quoted utterances paired with begin-end target intervals. We identify the conditioning-token range of every quoted utterance and convert its target timestamps to the corresponding audio and video latent coordinates. At each denoising step, TimeSteer independently localizes a source span for every utterance and pairs it with its target interval to construct a single temporally ordered read map. For LTX-2, localization uses textto-audio cross-attention from its separate audio stream, and the common wall-clock read map is evaluated on both the audio and video latent grids. For daVinci-MagiHuman, localization uses the interactions between quoted-text keys and audio queries in unified attention, and the same map is applied to the audio and video token groups. In both backbones, the original decoder produces the final 5-second audio-visual clip, with no post-hoc waveform or video editing.

Implementation. TimeSteer is applied at all eight denoising steps. At step n, we recover the predicted clean latent as $\hat { x } _ { 0 } ^ { n } = x _ { n } - \sigma _ { n } v _ { n }$ and replay it through the frozen transformer solely to expose the selected attention responses; this replay requires neither gradients nor parameter updates. For each utterance, responses over its quoted tokens are summed at every audio latent position, and the source span is bounded by the first and last positions whose response reaches the relative threshold $\tau = 0 . 5$ . LTX-2 uses transformer layer 25 and attention head 30. For daVinci-MagiHuman, we aggregate eight layer–head pairs: (29, 24), (29, 20), (31, 26), (31, 28), (31, 29), (34, 22), (34, 20), and (35, 29).

The localized source-target pairs define affine maps within speech intervals and cubic bridges across non-speech gaps, with the clip endpoints fixed. We evaluate the resulting continuous map on each discrete modality grid and linearly interpolate neighboring source positions. Finally, the scheduled clean latent $\tilde { x } _ { 0 } ^ { n }$ is converted back to $\tilde { v } _ { n } = \bar { ( } x _ { n } - \tilde { x } _ { 0 } ^ { n } ) / \sigma _ { n }$ and the backbone proceeds with its original sampler update. Region-Aware Latent Remapping introduces no tuned parameter beyond the localized source spans and user-specified target intervals.

## Localization Component Selection

We identify localization components through a hierarchical analysis at progressively finer levels of attention aggregation. Averaging text-to-audio attention across the full transformer reveals a temporally diffuse but discernible concentration within a bounded audio interval. Layer-wise aggregation shows that this temporal structure is substantially sharper in a small subset of layers, as illustrated for LTX-2 and daVinci-MagiHuman in Figures 9 and 10, respectively. Resolving these layers into individual heads further isolates the most temporally concentrated responses, as shown in Figure 11 for LTX-2 and Figures 12–15 for daVinci-MagiHuman.

The analysis isolates layer 25 and head 30 in LTX-2, and eight heads across layers 29, 31, 34, and 35 in daVinci-MagiHuman. Its applicability to both separate cross-attention and unified-attention backbones supports the use of a common selection procedure across architectures rather than model-specific heuristics.

## Additional Results

Complete Metric Comparison Table 3 extends the representative comparison in the main paper with five additional metrics. CLAP measures audio-text alignment, CLVP measures quote-level video-text consistency, while MANIQA and LAION provide complementary no-reference assessments of visual quality. LSE-C and LSE-D measure audio— visual synchronization in terms of confidence and feature distance, respectively. Both metrics are averaged over target intervals with valid face detections. We report latency as the total generation time divided by the eight denoising steps.

Across both backbones, TimeSteer more than doubles the best baseline hit rate (0.73 versus 0.31 on LTX-2 and 0.53 versus 0.21 on daVinci-MagiHuman) while also improving temporal IoU and boundary accuracy. These gains preserve generated content: WER remains low, and semantic-fidelity, perceptual-quality, and cross-modal scores remain comparable to uncontrolled generation and the strongest baselines.

The additional cost is small. TimeSteer's amortized perstep latency remains close to uncontrolled generation (3.84 versus 3.47 s on LTX-2 and 6.57 versus 6.60 s on daVinci-MagiHuman). Unlike attention-based baselines that inject masks or logit penalties into daVinci-MagiHuman's unified attention, TimeSteer remaps the predicted clean latent and remains compatible with accelerated attention operators.

Cross-Seed Robustness across Speaking Structures To evaluate seed robustness across speaking structures, we select 20 SpeechShift cases and run each method with five different seeds. Table 4 reports mean±std for the four structure types, with LTX-2 and daVinci-MagiHuman shown side by side.

Across all four speaking structures, TimeSteer achieves the highest hit rate on both backbones. It also gives the highest IoU in every LTX-2 setting and the highest or tied-highest IoU in every daVinci-MagiHuman setting. WER remains competitive: TimeSteer matches the best result in two LTX-2 settings and one daVinci-MagiHuman setting, and stays within 0.01–0.04 of the best result elsewhere. These gains persist across five seeds, showing that the improvement in temporal control is not tied to a particular random seed.

Cross-Metric Consistency and Candidate Selection We compute Spearman correlations among ${ \mathrm { H R } } _ { 0 . 2 } ,$ IoU, and WER across individual generations to examine whether accurate temporal control coincides with faithful speech generation. Figure 5 shows strong agreement between $\mathrm { { H R } _ { 0 . 2 } }$ and IoU on both backbones. Their correlations with WER are consistently negative, indicating that improved interval placement generally accompanies, rather than compromises, transcription fidelity.

This relationship is particularly useful for candidate selection. Among outputs with $\mathrm { H R } _ { 0 . 2 } ~ = ~ 1$ and IoU≥ 0.8, approximately 90% also achieve $\mathrm { W E R \leq 0 . 1 }$ on both backbones. Temporal and transcription metrics can therefore be used jointly to identify generations that satisfy both timing and content requirements.

Additional Ablation Studies We provide additional analyses of both stages of TimeSteer. We first examine sourcelocalization behavior across denoising steps and its sensitivity to the localization threshold, and then study how the denoising-step schedule used for Region-Aware Latent Remapping affects performance.

Source localization across denoising steps. The main paper compares four source-localization strategies using results aggregated over denoising. Figure 6 disaggregates the same ablation data by denoising step. Attn@xn reaches high IoU and $\mathrm { { H R } _ { 0 . 2 } }$ within the first few steps and remains stable, whereas $\tan @ x _ { n }$ becomes competitive only near the end. Both decode-based variants remain substantially less accurate. This step-wise view confirms that the clean estimate exposes reliable timing before it is recoverable from intermediate audio.

Localization-threshold sensitivity. We vary the relative attention-mass threshold that converts the $\mathsf { A i t m } @ \hat { x } _ { 0 } ^ { n }$ mass into a source span. Figure 7 shows a broad optimum around 0.3–0.5: threshold 0.3 gives the highest aggregate IoU and $\mathrm { { H R } _ { 0 . 2 } }$ , while the operating value 0.5 remains close across denoising steps. Larger thresholds contract the detected span; at 0.9, both metrics drop sharply.

Remapping-schedule sensitivity. We next vary the denoising steps at which Region-Aware Latent Remapping is applied. We partition the eight steps into early (0–2), middle (3–4), and late (5–7) stages. Because eight steps cannot be divided equally into three contiguous stages, the middle stage contains two steps, whereas the early and late stages contain three each. We evaluate both stage-only schedules and their complementary stage-omission schedules on the same deterministic 20-case subset, comprising the first five benchmark entries from each speaking structure. Figure 8 shows that applying remapping at all steps gives the best IoU and $\mathrm { { H R } _ { 0 . 2 } }$ while maintaining a low WER. The late stage is particularly important: using only late steps remains substantially stronger than using only early or middle steps, while omitting late steps causes the largest degradation in scheduling accuracy. By contrast, removing either the early or middle stage retains most of the all-step remapping performance.

<table><tr><td></td><td colspan="4">Interval Controllability</td><td colspan="3">Semantic Fidelity</td><td colspan="3">Perceptual Quality</td><td colspan="2">Cross-modal</td><td>Efficiency</td></tr><tr><td>Method</td><td>HR0.2↑IoU↑mEs (s)↓ mEe (s)↓|WER↓CLAP↑CLVP↑|PQ↑MUSIQ↑MANIQA↑LAION↑|LSE-C↑LSE-D↓|Latency (s)↓</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">LTX-2</td><td colspan="3"></td></tr><tr><td>Uncontrolled</td><td>0.21</td><td>0.63</td><td>0.56</td><td>0.54</td><td>0.05 0.21</td><td>0.35</td><td>|5.36</td><td>52.26</td><td>0.27</td><td>4.44</td><td>3.13</td><td>11.60</td><td>3.47</td></tr><tr><td>Textual Timing</td><td>0.19</td><td>0.60</td><td>0.59</td><td>0.59</td><td>0.07 0.22</td><td>0.35</td><td>5.36</td><td>52.26</td><td>0.27</td><td>4.47</td><td>3.18</td><td>11.55</td><td>3.44</td></tr><tr><td>FreeAudio</td><td>0.19</td><td>0.54</td><td>0.49</td><td>0.67</td><td>0.35 0.20</td><td>0.35</td><td>5.92</td><td>52.55</td><td>0.27</td><td>4.44</td><td>2.92</td><td>11.08</td><td>3.49</td></tr><tr><td>Prompt Relay</td><td>0.31</td><td>0.68</td><td>0.27</td><td>0.35</td><td>0.20 0.21</td><td></td><td>0.35 5.52</td><td>52.00</td><td>0.27</td><td>4.45</td><td>3.06</td><td>11.13</td><td>3.45</td></tr><tr><td>TimeSteer</td><td>0.73</td><td>0.87</td><td>0.11</td><td>0.15</td><td>0.07</td><td>0.22</td><td>0.35 5.32</td><td>52.51</td><td>0.27</td><td>4.45</td><td>3.10</td><td>10.95</td><td>3.84</td></tr><tr><td colspan="10">daVinci-MagiHuman</td><td colspan="3"></td><td colspan="3"></td></tr><tr><td>Uncontrolled</td><td>0.09</td><td>0.63</td><td>0.63</td><td>0.49</td><td>0.06</td><td>0.23</td><td>0.35 |5.53</td><td>69.36</td><td>0.38</td><td>4.92</td><td></td><td>3.67</td><td>9.82</td></tr><tr><td>Textual Timing</td><td>0.08</td><td>0.61</td><td>0.64</td><td>0.53</td><td>0.09 0.23</td><td></td><td>0.35 5.50</td><td>68.98</td><td>0.38</td><td>4.93</td><td>3.87</td><td>9.57</td><td>6.60 6.45</td></tr><tr><td>FreeAudio</td><td>0.01</td><td>0.16</td><td>1.04</td><td>1.40</td><td>0.95 0.20</td><td>0.36</td><td>5.24</td><td>66.80</td><td>0.34</td><td>4.83</td><td>1.63</td><td>9.47</td><td>8.19</td></tr><tr><td>Prompt Relay</td><td>0.21</td><td>0.68</td><td>0.39</td><td>0.32</td><td>0.29 0.24</td><td>0.35</td><td>5.61</td><td>70.87</td><td>0.44</td><td>4.95</td><td>3.66</td><td>10.07</td><td>10.52</td></tr><tr><td>TimeSteer</td><td>0.53</td><td>0.79</td><td>0.19</td><td>0.18</td><td>0.08 0.23</td><td>0.35</td><td>5.42</td><td>69.08</td><td>0.38</td><td>4.91</td><td>3.56</td><td>9.70</td><td>6.57</td></tr></table>

Table 3: Complete SpeechShift results. Best results per backbone are in bold. ↑/↓ indicates higher/lower is better.

Additional Qualitative Comparison Figures 16-19 present qualitative comparisons among the five methods, with two examples from LTX-2 and two from daVinci-MagiHuman. For each method, the upper row shows sampled video frames and the lower row shows the temporally aligned mel-spectrogram; shaded regions indicate the target speech intervals. Across both backbones, TimeSteer places the speech activity within the specified target intervals more accurately than the competing methods. At the same time, it preserves the acoustic context outside the controlled intervals, including the background sounds specified by the prompt, rather than suppressing or displacing them together with the speech. The corresponding video frames remain visually coherent and exhibit no prominent artifacts caused by the temporal remapping in these examples.

<table><tr><td rowspan="2">Method</td><td colspan="3">LTX-2</td><td colspan="3">daVinci-MagiHuman</td></tr><tr><td> $\mathbf { H R } _ { 0 . 2 } \uparrow$ </td><td> $\mathbf { I o U } \uparrow$ </td><td>WER↓</td><td> $\mathbf { H R } _ { 0 . 2 } \uparrow$ </td><td>IoU↑</td><td>WER↓</td></tr><tr><td colspan="7">Single speaker, single utterance</td></tr><tr><td>Uncontrolled</td><td> $0 . 1 2 { \pm } 0 . 1 1$ </td><td> $0 . 5 2 { \pm } 0 . 0 1$ </td><td> $\mathbf { 0 . 0 3 } \pm \mathbf { 0 . 0 0 }$ </td><td> $0 . 0 0 { \pm } 0 . 0 0$ </td><td> $0 . 4 6 { \pm } 0 . 0 4$ </td><td> $\mathbf { 0 . 0 9 } \pm \mathbf { 0 . 0 4 }$ </td></tr><tr><td>Textual Timing</td><td> $0 . 1 2 { \pm } 0 . 1 1$ </td><td> $0 . 5 3 { \pm } 0 . 0 4 $ </td><td> $\mathbf { 0 . 0 3 } \pm \mathbf { 0 . 0 0 }$ </td><td> $0 . 0 0 { \pm } 0 . 0 0$ </td><td> $0 . 4 1 { \pm } 0 . 0 2$ </td><td> $0 . 1 2 { \pm } 0 . 0 3$ </td></tr><tr><td>FreeAudio</td><td> $0 . 0 8 { \pm } 0 . 1 1$ </td><td> $0 . 4 7 { \pm } 0 . 0 6$ </td><td> $0 . 3 3 { \pm } 0 . 1 8$ </td><td> $0 . 0 0 { \pm } 0 . 0 0$ </td><td> $0 . 2 2 { \pm } 0 . 1 9$ </td><td> $0 . 9 6 { \pm } 0 . 0 4$ </td></tr><tr><td>Prompt Relay</td><td> $0 . 2 8 { \pm } 0 . 1 8$ </td><td> $0 . 7 4 { \pm } 0 . 1 2$ </td><td> $0 . 1 2 { \pm } 0 . 0 8$ </td><td> $0 . 2 4 { \pm } 0 . 1 7$ </td><td> $0 . 6 2 { \pm } 0 . 0 4$ </td><td> $0 . 3 8 { \pm } 0 . 0 6$ </td></tr><tr><td>TimeSteer (Ours)</td><td> $\mathbf { 0 . 6 4 } \pm \mathbf { 0 . 0 9 }$ </td><td> $\mathbf { 0 . 9 0 } \pm \mathbf { 0 . 0 1 }$ </td><td> $\mathbf { 0 . 0 3 } \pm \mathbf { 0 . 0 0 }$ </td><td> $\mathbf { 0 . 6 4 } \pm \mathbf { 0 . 0 9 }$ </td><td> $\mathbf { 0 . 8 1 \pm 0 . 1 5 }$ </td><td> ${ \bf 0 . 0 9 } \pm { \bf 0 . 0 7 }$ </td></tr><tr><td colspan="7">Single speaker, multiple utterances</td></tr><tr><td>Uncontrolled</td><td> $0 . 3 0 { \pm } 0 . 1 7$ </td><td> $0 . 7 4 { \pm } 0 . 0 6$ </td><td> $0 . 0 7 { \pm } 0 . 1 1$ </td><td> $0 . 2 0 { \pm } 0 . 0 0$ </td><td> $0 . 8 1 { \pm } 0 . 0 1$ </td><td> $\mathbf { 0 . 0 4 } \pm \mathbf { 0 . 0 1 }$ </td></tr><tr><td>Textual Timing</td><td> $0 . 2 8 { \pm } 0 . 0 8$ </td><td> $0 . 7 7 { \pm } 0 . 0 5$ </td><td> $\mathbf { 0 . 0 3 } \pm \mathbf { 0 . 0 3 }$ </td><td> $0 . 1 0 { \pm } 0 . 0 7$ </td><td> $0 . 7 2 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $0 . 0 8 { \pm } 0 . 0 3$ </td></tr><tr><td>FreeAudio</td><td> $0 . 1 2 { \pm } 0 . 0 8$ </td><td> $0 . 5 3 { \pm } 0 . 2 4 $ </td><td> $0 . 4 5 { \pm } 0 . 2 5 $ </td><td> $0 . 0 0 { \pm } 0 . 0 0$ </td><td> $0 . 1 1 { \pm } 0 . 0 7$ </td><td> $0 . 9 7 { \pm } 0 . 0 3$ </td></tr><tr><td>Prompt Relay</td><td> $0 . 4 0 { \pm } 0 . 1 9$ </td><td> $0 . 7 8 { \pm } 0 . 0 4$ </td><td> $0 . 0 9 { \pm } 0 . 0 6$ </td><td> $0 . 5 0 { \pm } 0 . 0 0$ </td><td> $\mathbf { 0 . 8 8 } \pm \mathbf { 0 . 0 0 }$ </td><td> $0 . 1 4 \pm 0 . 0 1$ </td></tr><tr><td>TimeSteer (Ours)</td><td> $\mathbf { 0 . 9 0 } \pm \mathbf { 0 . 1 } 2$ </td><td> $\mathbf { 0 . 8 7 \pm 0 . 0 8 }$ </td><td> $0 . 0 6 { \pm } 0 . 0 9$ </td><td> $\mathbf { 0 . 8 2 } \pm \mathbf { 0 . 0 8 }$ </td><td> $\mathbf { 0 . 8 8 } \pm \mathbf { 0 . 0 3 }$ </td><td> $0 . 0 5 { \pm } 0 . 0 3$ </td></tr><tr><td colspan="7">Multiple speakers, single utterance</td></tr><tr><td>Uncontrolled</td><td> $0 . 2 0 { \pm } 0 . 0 0$ </td><td> $0 . 5 1 { \pm } 0 . 0 5$ </td><td> $\mathbf { 0 . 0 3 \pm 0 . 0 1 }$ </td><td> $0 . 0 8 { \pm } 0 . 1 1$ </td><td> $0 . 5 8 { \pm } 0 . 1 0 $ </td><td> $\mathbf { 0 . 1 0 { \overset { . } { \bot } } 0 . 0 5 }$ </td></tr><tr><td>Textual Timing</td><td> $0 . 1 6 { \pm } 0 . 0 9$ </td><td> $0 . 5 1 { \pm } 0 . 0 3$ </td><td> $\mathbf { 0 . 0 3 \pm 0 . 0 1 }$ </td><td> $0 . 0 0 { \pm } 0 . 0 0$ </td><td> $0 . 4 7 { \pm } 0 . 0 2$ </td><td> $\mathbf { 0 . 1 0 { \pm } 0 . 0 4 }$ </td></tr><tr><td>FreeAudio</td><td> $0 . 1 2 { \pm } 0 . 1 8$ </td><td> $0 . 4 1 { \pm } 0 . 0 3$ </td><td> $0 . 2 1 { \pm } 0 . 0 8$ </td><td> $0 . 0 0 { \pm } 0 . 0 0$ </td><td> $0 . 1 5 { \pm } 0 . 0 8$ </td><td> $0 . 9 3 { \pm } 0 . 0 5$ </td></tr><tr><td>Prompt Relay</td><td> $0 . 2 8 { \pm } 0 . 1 8$ </td><td> $0 . 5 6 { \pm } 0 . 1 0$ </td><td> $0 . 3 6 { \pm } 0 . 0 6$ </td><td> $0 . 1 6 { \pm } 0 . 2 6$ </td><td> $0 . 6 9 { \pm } 0 . 0 6$ </td><td> $0 . 3 2 { \pm } 0 . 0 9$ </td></tr><tr><td>TimeSteer (Ours)</td><td> $\mathbf { 0 . 5 2 \pm 0 . 2 3 }$ </td><td> $\mathbf { 0 . 8 2 \pm 0 . 0 5 }$ </td><td> $0 . 0 7 { \pm } 0 . 0 6$ </td><td> $\mathbf { 0 . 5 6 { \pm } 0 . 0 9 }$ </td><td> $\mathbf { 0 . 7 1 \pm 0 . 1 5 }$ </td><td> $0 . 1 1 { \pm } 0 . 0 5$ </td></tr><tr><td colspan="7">Multiple speakers, multiple utterances</td></tr><tr><td>Uncontrolled</td><td> $0 . 2 0 { \pm } 0 . 1 0$ </td><td> $0 . 7 1 { \pm } 0 . 0 7$ </td><td> $\mathbf { 0 . 0 9 } \pm \mathbf { 0 . 0 4 }$ </td><td> $0 . 0 8 { \pm } 0 . 0 4$ </td><td> $0 . 6 9 { \pm } 0 . 0 2$ </td><td> $\mathbf { 0 . 0 7 } \pm \mathbf { 0 . 0 2 }$ </td></tr><tr><td>Textual Timing</td><td> $0 . 1 6 { \pm } 0 . 0 9$ </td><td> $0 . 6 7 { \pm } 0 . 0 9$ </td><td> $0 . 1 0 { \pm } 0 . 0 9$ </td><td> $0 . 0 8 { \pm } 0 . 0 4$ </td><td> $0 . 6 4 { \pm } 0 . 0 3$ </td><td> $0 . 0 9 { \pm } 0 . 0 6$ </td></tr><tr><td>FreeAudio</td><td> $0 . 1 6 { \pm } 0 . 1 1$ </td><td> $0 . 5 2 { \pm } 0 . 0 9$ </td><td> $0 . 4 8 { \pm } 0 . 0 7$ </td><td> $0 . 0 2 { \pm } 0 . 0 4$ </td><td> $0 . 0 8 { \pm } 0 . 0 6$ </td><td> $0 . 9 8 { \pm } 0 . 0 1$ </td></tr><tr><td>Prompt Relay</td><td> $0 . 2 6 { \pm } 0 . 0 5$ </td><td> $0 . 6 6 { \pm } 0 . 1 0$ </td><td> $0 . 2 4 { \pm } 0 . 1 0$ </td><td> $0 . 2 4 { \pm } 0 . 0 9$ </td><td> $0 . 7 8 { \pm } 0 . 0 2$ </td><td> $0 . 2 0 { \pm } 0 . 0 4 $ </td></tr><tr><td>TimeSteer (Ours)</td><td> $\mathbf { 0 . 6 6 } \pm \mathbf { 0 . 0 9 }$ </td><td> $\mathbf { 0 . 8 6 { \pm 0 . 0 6 } }$ </td><td> ${ \bf 0 . 0 9 } \pm { \bf 0 . 0 7 }$ </td><td> $\mathbf { 0 . 6 6 } \pm \mathbf { 0 . 0 9 }$ </td><td> $\mathbf { 0 . 8 1 } \pm \mathbf { 0 . 0 7 }$ </td><td> $0 . 1 0 { \pm } 0 . 0 4 $ </td></tr></table>

Table 4: Cross-seed robustness across speaking structures. Values are mean±std over five seeds. Best mean results within each backbone and structure are in bold.

LTX-2  
![](images/d766d84be40b35fb515035bdf7f2ee0cbea7399114f16644c454e76a6b57868f.jpg)

![](images/6d15634bb72e90f7b9077a2ce068fa804404fb1d21483ff1eba54c1f19bbcf6a.jpg)  
Figure 5: Sample-level Spearman correlations among ${ \mathrm { H R } } _ { 0 . 2 } ,$ IoU, and WER for TimeSteer across five seeds. Positive $\mathrm { H R } _ { 0 . 2 ^ { - } }$ IoU correlations indicate agreement between the temporal metrics, while negative correlations with WER indicate that better temporal control tends to coincide with lower transcription error.

![](images/a9311773e01f4ded0c7ecaf10c5dbe0278c80876b3e7044dee5faf06903186f9.jpg)  
Figure 6: Step-wise extension of the source-localization ablation. Left: mean IoU. Right: $\mathrm { H R } _ { 0 . 2 }$ Curves use the same evaluation data as the aggregated comparison in the main paper.

![](images/6e856d47608265dd03f8abcb0e40521702e18b877f9ad1217958738f565e55b4.jpg)  
Figure 7: Sensitivity to the localization threshold for $\mathbf { A t t n } @ \hat { x } _ { 0 } ^ { n }$ . Left: IoU. Right: $\mathrm { H R } _ { 0 . 2 } .$ Top: mean over all denoising steps, with values labeled above the bars and error bars showing standard deviation across cases. Bottom: step-wise means; the narrow panels show the full [0, 1] scale. Colors and line styles identify the threshold.

Quote tokens: this | sauce | needs | one | more | minute | to | reduce  
![](images/d138d395508d274f6c5177e5ced26fa2e07949d100db4cdb042fb87844aac36c.jpg)  
Figure 8: Sensitivity to the denoising-step schedule for Region-Aware Latent Remapping. Early, middle, and late denote steps 0–2, 3–4, and 5–7, respectively. From left to right: temporal IoU, ${ \mathrm { H R } } _ { 0 . 2 } ,$ and WER. Bars show means over the fixed stratified 20-case subset; error bars show standard deviation across cases. The all-step schedule is highlighted in red.

![](images/0f3be789e0b458b7c265ba0cb133bf82c04ebd58750fdfd222ce138323f370e8.jpg)

![](images/ef8c4b2f71a8819a2757a08f19acc5bbfc47f4a0a4d460991440f736088f1a85.jpg)  
Figure 9: Layer selection using LTX-2 text-to-audio attention on the predicted clean latent at denoise step index 5. The left panel averages text-to-audio attention over all layers and heads; the right panel averages the heads within each layer. Layer 25 is retained for head-level inspection.

Layer 25  
![](images/bdfd78c9ab57c789c601f00ea8d3d23b69adcc6795706caa492b2519929b0dc2.jpg)

Average of 40 heads within each layer  
![](images/b36b1abdfc6d572cc6fabb4bcd65b4c5007de213783fb1e89ceb8a49e92a3f76.jpg)  
Quote tokens: the | meeting | starts | at | nine | o | clock | sharp

Figure 10: Layer selection using daVinci-MagiHuman text-to-audio attention on the predicted clean latent at denoise step index 5. The left panel averages text-to-audio attention over all layers and heads; the right panel averages the heads within each layer. Layers 29, 31, 34, and 35 are retained for head-level inspection.  
![](images/963e001d14f8bcb1509854429aa50c7f46b008d45e61a0ff5ecfe38debb502be.jpg)  
Quote tokens: this | sauce | needs | one | more | minute | to | reduce  
Figure 11: Head-level text-to-audio attention on the predicted clean latent in LTX-2 layer 25. The selected head 30 is outlined and labeled in red.

![](images/0863065949afdcf1dadc78d1352de7b466bd14644e51de52e4ecd0f68e8ac335.jpg)  
Figure 12: Head-level text-to-audio attention on the predicted clean latent in daVinci-MagiHuman layer 29. Selected heads 20 and 24 are outlined and labeled in red.

![](images/771f24b2a434dc7024a71493ec46f126033f67fd843a9e47077ffe54bf59a7c6.jpg)  
Figure 13: Head-level text-to-audio attention on the predicted clean latent in daVinci-MagiHuman layer 31. Selected heads 26, 28, and 29 are outlined and labeled in red.

![](images/d4a4e8d8aa7bc62fa88aa4cc44c4572c2d2a3445f091070e616d6b4527cf1f53.jpg)  
Figure 14: Head-level text-to-audio attention on the predicted clean latent in daVinci-MagiHuman layer 34. Selected heads 20 and 22 are outlined and labeled in red.

![](images/9d94142591972afad3d588e61821019d9b693eb7dbad54814d7f9179aaa518b1.jpg)  
Figure 15: Head-level text-to-audio attention on the predicted clean latent in daVinci-MagiHuman layer 35. Selected head 29 is outlined and labeled in red.

![](images/fc91b33be52f4f2deafc0fa73308e504f9ecc351b58ac15df51ba96ecad31204.jpg)  
Figure 16: A sample generated by LTX-2. The prompt is "On a rainforest boardwalk trail with dense foliage, a male host sits upright addressing the viewer and says: keep your voice down please'. Traffic noise and honking continue outside the window.' The target interval is [1.9s, 3.2s].

![](images/1a9dd34c76d44cb393e2e92bf3dff3764e37fab42ef9fca0f4ad6d71783dc2ea.jpg)  
Figure 17: A sample generated by LTX-2. The prompt is "In a retro arcade with flashing game cabinets, a male cashier faces the camera and says: the weather looks clear this morning'. Following a noticeable pause, he says: we should start hiking before noon'. The room stays quiet afterward." The target intervals are [0.2s, 1.8s] and [2.9s, 4.5s].

![](images/da9689f35e6979ca20a241619b4ff14e2c26eb5c1c97890d77b0003e9cae11e0.jpg)  
Figure 18: A sample generated by daVinci-MagiHuman. The prompt is "During the afternoon at the bus terminal, a young female station attendant in a charcoal blazer speaks with an older female transportation worker in a neatly pressed uniform. The first person asks: what should we do with the bus terminal'. The second responds: leave it untouched until this supervisor arrives'. Elsewhere around the bus terminal, only faint room tone remains after the line." The target intervals are [0.2s, 2.6s] and [2.9s, 4.8s].

![](images/0348b0679a135a34d7ee67d235e97d21bb6c32f221a6b31e83bf634a7ee9460e.jpg)  
Figure 19: A sample generated by daVinci-MagiHuman. The prompt is "Alongside the pharmacy, a middle-aged female clinic receptionist in a neatly pressed uniform speaks with a middle-aged male dental assistant with a small shoulder bag. The first person asks: 'could you show me the pharmacy'. The second responds: 'of course follow me through this entrance'. Elsewhere around the pharmacy, two staff members discuss a patient quietly behind the speaker." The target intervals are [0.2s, 1.8s] and [2.6s, 4.5s].