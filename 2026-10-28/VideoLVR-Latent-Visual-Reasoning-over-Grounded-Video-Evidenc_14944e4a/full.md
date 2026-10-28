# VideoLVR: Latent Visual Reasoning over Grounded Video Evidence

Shilong Li<sup>∗</sup>, Xingyuan Bu<sup>∗†</sup>, Hao Shi<sup>∗</sup>, Yun Xing, Weixin Feng, Qiangpeng Yang, Shilei Wen

Beijing University of Posts and Telecommunications Tsinghua University Nanyang Technological University

## Abstract

Chain-of-thought reasoning has substantially advanced video understanding in multimodal large language models (MLLMs). However, language-based reasoning traces do not provide a reliable mechanism for models to revisit relevant visual evidence, leading to visual forgetting and impairing their ability to reason over video content. Existing approaches maintain visual grounding during reasoning in two main ways: they either refer to evidence using textual coordinates, which specify symbolic locations rather than the visual content itself, or reinsert cropped regions or frames into the multimodal context, preserving visual details at the cost of substantial context expansion. To address these limitations, we introduce VideoLVR, which constructs latent visual evidence from local regions at relevant video moments, thereby providing visual support for intermediate reasoning steps. To supervise latent visual evidence generation, we introduce a visual token prediction objective that formulates visual token reconstruction as prediction over a vocabulary derived from the input video, thereby unifying visual evidence reconstruction and text generation within the same token prediction framework. This shared prediction framework enables MLLMs to reason over interleaved text tokens and latent visual states while integrating evidence from diferent moments. We further extend VideoLVR to reinforcement learning by treating sampled visual tokens as actions alongside text tokens and jointly optimizing the policy using final-answer rewards. By eficiently revisiting task-relevant visual information during reasoning, VideoLVR improves video understanding without substantially expanding the multimodal context. Experiments on multiple video benchmarks demonstrate its efectiveness, especially on tasks requiring fine-grained visual evidence.

## 1 Introduction

Recent advances in multimodal large language models (MLLMs) have enabled increasingly complex video understanding and reasoning. Supervised chain-of-thought training and reinforcement learning have further strengthened these capabilities, allowing models to solve video questions that require temporal localization and multi-step reasoning (Feng et al. 2025; Wang et al. 2025; Maaz et al. 2025). However, the resulting traces remain largely language-based. As reasoning proceeds, models may gradually lose access to task-relevant visual evidence, resulting in visual forgetting and limiting their ability to reason over video content (Sun et al. 2025, 2026). This makes it dificult to revisit the localized and temporally sparse evidence required by later steps.

One line of work grounds reasoning with textual coordinates, using timestamps for temporal evidence and bounding boxes for spatial evidence (Maaz et al. 2025; Meng et al. 2026; Fan et al. 2025; Wang et al. 2026a). Although compact and traceable, these coordinates encode the locations of evidence rather than its visual content. Other methods invoke visual tools to crop or zoom into image regions, or to select additional video frames and clips (Zheng et al. 2026; Jiang et al. 2025; Zhang et al. 2026a). Although these tools expose fine-grained visual details, tool-augmented reasoning is limited to a predefined set of visual operations (Li et al. 2025). Moreover, each returned crop, frame, or clip must be re-encoded and inserted into the multimodal context, causing the context to grow as visual tool calls accumulate, particularly in video reasoning (Li et al. 2025; Zheng et al. 2026; Zhang et al. 2026a). Neither formulation provides a compact and flexible way to revisit relevant visual content while preserving its visual semantics.

Latent visual reasoning (LVR) (Li et al. 2025), which trains an MLLM to autoregressively reconstruct question-relevant visual tokens and interleave them with textual reasoning, ofers a promising alternative by representing intermediate visual evidence with continuous hidden states rather than text or additional visual observations. However, most prior latent visual reasoning methods focus on the image domain, where latent states are learned using image-based visual targets or image-level bottlenecks (Wang et al. 2026c; Yang et al. 2026; Li et al. 2026; Wu et al. 2026; Miao et al. 2026; Xu et al. 2026). In video reasoning, a single reasoning step may require local visual evidence from several moments. Evidence from any single moment may be insuficient. The model must relate visual information across time to support the reasoning step. Meanwhile, neighboring frames often contain similar visual tokens (Fu et al. 2025; Tao et al. 2025). MSE feature regression aligns each prediction with its target independently, without explicitly distinguishing that target from other visually similar tokens in the same video.

To this end, we introduce VideoLVR, which learns to generate latent visual evidence to support intermediate video reasoning. Each latent block represents evidence from local regions supporting the same reasoning step, allowing evidence from diferent video moments to be modeled jointly. We supervise these blocks with visual token sequences derived from spatio-temporal grounding annotations in evidencegrounded reasoning trajectories. To distinguish the target evidence from similar visual content, we introduce visual token prediction (VTP), which trains each latent query to predict its target token among all visual tokens from the input video. Together with next-token prediction for text, this objective trains VideoLVR to generate reasoning interleaved with latent visual evidence. At inference, the supervised VideoLVR model projects decoder hidden states into latent visual states and feeds them back into the model without inserting additional frames or region crops into the multimodal context. For reinforcement learning, we introduce text–visual policy optimization (TVPO), which samples visual tokens from the input video’s visual vocabulary using our visual token prediction formulation and jointly optimizes text and visual token actions with final-answer rewards. Experiments across multiple video understanding and reasoning benchmarks demonstrate its efectiveness, with particular gains on tasks requiring fine-grained visual evidence. Our main contributions are summarized as follows:

• We introduce VideoLVR, which learns from evidencegrounded reasoning trajectories to construct grouped latent visual evidence from local regions across video moments, allowing temporally distributed evidence to be modeled jointly.

• We propose visual token prediction (VTP) over the input video’s visual vocabulary, unifying latent visual evidence generation with text token prediction. Building on VTP, we introduce TVPO to jointly optimize text and visual token actions through reinforcement learning.

• We evaluate VideoLVR across multiple video reasoning benchmarks and conduct extensive ablations of its key designs. The results show particular gains on tasks requiring fine-grained visual evidence.

## 2 Related Work

Video Reasoning with Explicit Evidence. Recent work grounds video reasoning using explicit temporal and spatial references. Video-R2 improves temporal grounding through timestamp-aware supervision and a temporal alignment reward (Maaz et al. 2025), while Open-o3-Video interleaves timestamps and bounding-box coordinates with textual reasoning (Meng et al. 2026). Related image reasoning methods similarly localize relevant regions to ground intermediate reasoning (Fan et al. 2025; Wang et al. 2026a,b). These methods make evidence traceable, but their coordinates encode symbolic locations rather than visual semantics.

Revisiting Visual Evidence during Reasoning. Long multimodal reasoning can progressively forget visual information, motivating methods that revisit relevant evidence (Sun et al. 2025, 2026). DeepEyes and VLM-R<sup>3</sup> select and inspect image regions during reasoning (Zheng et al. 2026; Jiang et al. 2025). For videos, VITAL dynamically samples additional frames, while VideoSearcher combines temporal localization, spatial inspection, and multimodal search (Zhang et al. 2026a; Gao et al. 2026). These methods recover fine-grained evidence by inserting regions, frames, or retrieved items as additional multimodal observations.

Latent Visual Reasoning. Continuous latent reasoning has been explored in language models, with Coconut feeding hidden states back as input embeddings to perform intermediate computation without decoding it into text (Hao et al. 2025). Recent visual reasoning methods extend this idea by generating or refining latent visual representations. LVR reconstructs question-relevant visual tokens through latent states interleaved with textual reasoning (Li et al. 2025). Other approaches learn continuous visual thoughts through visual-semantic distillation, task supervision, reinforcement learning, or latent bottlenecks (Wang et al. 2026c; Yang et al. 2026; Li et al. 2026). LaViT, GAP, and Semantic-Enriched Latent Visual Reasoning further improve latent supervision through attention distillation, feature-space alignment, curriculum gating, or region-centric attribute supervision (Wu et al. 2026; Miao et al. 2026; Xu et al. 2026). Diagnostic studies nevertheless show that models may bypass latent tokens, generate latents that deviate from their targets, or underutilize their visual semantics because of shortcuts through the visual input (Viveiros et al. 2026; Zhang et al. 2026b).

Beyond continuous latent-state reasoning, Mull-Tokens uses a fixed set of modality-agnostic special tokens as a compact internal multimodal scratchpad, first aligning them with interleaved text–image traces and then relaxing training to final-answer supervision (Ray et al. 2026). UniVR instead autoregressively generates visual reasoning trajectories within the discrete token space of a unified generation model, targeting physical dynamics and long-horizon planning (Ren et al. 2026). VideoLatent extends latent reasoning to videos through latent injection, self-forcing, and alignment and diversity objectives trained from standard video question–answer triplets (Hu et al. 2026b). However, these methods either provide generic computation slots, generate discrete visual trajectories, or learn video latents without step-level supervision from localized spatio-temporal evidence. VideoLVR addresses this gap by grouping temporally distributed local regions that support the same reasoning step and supervising latent visual evidence generation through prediction over the visual vocabulary of the input video.

## 3 Method

As illustrated in Figure 1, VideoLVR first synthesizes evidence-grounded reasoning trajectories and converts supporting local regions into grouped latent targets, with regions within each target ordered by timestamp. It jointly learns text generation and latent visual evidence generation through next-token prediction and visual token prediction. During inference with the supervised model, decoder hidden states are projected into latent visual states and fed back autoregressively to support subsequent reasoning without introducing additional visual observations. We next describe the construction of grounded trajectories and grouped latent targets, followed by the learning objective and autoregressive inference procedure.

![](images/174a19a700b08e2bc6043fa0f6728b724ef570c15b0688d81cf11d3146618f55.jpg)  
Figure 1: Overview of VideoLVR. (a) A teacher MLLM synthesizes evidence-grounded reasoning trajectories through rejection sampling and trajectory polishing. (b) Visual tokens from regions supporting the same reasoning step are grouped and ordered by timestamp to form latent targets. VideoLVR jointly trains text generation with next-token prediction and latent visual evidence generation with visual token prediction over the input-video visual vocabulary. (c) At inference with the supervised model, each decoder hidden state is projected into a latent visual state and fed back as the input for the next decoding step, enabling autoregressive reasoning over latent visual evidence without additional visual observations.

## 3.1 Evidence-Grounded Data Synthesis

Starting from video question–answer data, we synthesize evidence-grounded reasoning trajectories through two stages: evidence-grounded trajectory rejection sampling and evidence-guided trajectory polishing.

Evidence-Grounded Trajectory Rejection Sampling. For each video-question pair, we prompt a teacher MLLM to generate a reasoning trajectory that ends with a final answer. The trajectory interleaves reasoning text with spatio-temporal visual evidence groups. We denote the k-th evidence group by $\mathcal { G } _ { k } ,$ , which associates an intermediate reasoning claim $c _ { k }$ with a sequence of localized visual regions $\mathcal { R } _ { k }$ . Each region is specified by a timestamp and a bounding box. The regions in $\mathcal { R } _ { k }$ may come from the same video moment or span multiple moments, allowing $\mathcal { G } _ { k }$ to capture both spatially distributed evidence within a frame and temporally distributed evidence across frames.

Formally, we represent the k-th evidence group $\mathcal { G } _ { k }$ as

$$
\begin{array} { r } { \mathcal { G } _ { k } = ( c _ { k } , \mathcal { R } _ { k } ) , \qquad \mathcal { R } _ { k } = ( { \bf r } _ { k , j } ) _ { j = 1 } ^ { m _ { k } } , } \end{array}\tag{1}
$$

where $c _ { k }$ is the reasoning claim and $\mathcal { R } _ { k }$ is the sequence of $m _ { k }$ localized visual regions that support it. Each region is represented as $\mathbf { r } _ { k , j } = ( t _ { k , j } , b _ { k , j } )$ , where $t _ { k , j }$ is its timestamp and $b _ { k , j } = ( x _ { k , j } ^ { 1 } , y _ { k , j } ^ { 1 } , x _ { k , j } ^ { 2 } , y _ { k , j } ^ { 2 } ) \in [ 0 , 1 ] ^ { 4 }$ specifies the normalized coordinates of the top-left and bottom-right corners. During training, each evidence group serves as a single unit for constructing a latent block.

We apply rejection sampling based on answer correctness and evidence validity. A response is retained only if its fina answer is judged equivalent to the reference answer and it contains at least one well-formed evidence group, with each group specifying a reasoning claim and one or more valid localized regions. We further reject malformed timestamps and invalid or overly broad bounding boxes. Rejected responses are resampled from the teacher using the validation errors as feedback. The accepted responses form the raw evidencegrounded trajectories passed to the polishing stage.

Evidence-Guided Trajectory Polishing. Although the teacher is instructed to interleave evidence groups with the reasoning steps that use them, rejection-sampled trajectories do not always exhibit the desired ordering. Evidence groups may appear after the relevant reasoning statements or cluster near the end of the response. Directly converting these groups into latent blocks would therefore misalign the latent evidence with the reasoning steps that use it.

To correct this ordering, we provide the original video and rejection-sampled trajectory to the teacher MLLM, which rewrites the trajectory to adjust the placement of evidence groups. The teacher preserves the original reasoning content while relocating each evidence group to the point where its visual evidence is used. We retain only trajectories that preserve the final answer, include every original evidence group exactly once, and place each group before the reasoning statement that uses the visual evidence it contains.

## 3.2 Grouped Latent Visual Evidence

We first map each localized visual region $\mathbf { r } _ { k , j }$ to a target sequence $\mathcal { E } _ { k , j }$ of visual token embeddings selected from the encoded video. Specifically, we map its timestamp $t _ { k , j }$ to the corresponding temporal position in the encoded video and project its normalized bounding box $b _ { k , j }$ onto the spatial token grid after visual token merging. We select the grid cells intersected by the projected box. Their visual token embeddings form $\mathcal { E } _ { k , j }$ , ordered from left to right within each row and from top to bottom across rows.

The evidence supporting a reasoning claim may comprise multiple localized visual regions from one or more video moments. We first order the regions in $\mathcal { G } _ { k }$ by timestamp. We then concatenate their visual-token sequences in temporal order, without inserting any boundary or separator tokens between regions, forming a single grouped target $\widetilde { \mathcal { E } } _ { k }$ . Regions with the same timestamp retain their original annotation order. We use $\widetilde { \mathcal { E } } _ { k }$ to supervise the latent block corresponding to claim $c _ { k }$

We use the special tokens <bot> and <eot> to mark the beginning and end of each latent block, respectively. During training, each evidence group is replaced by $L _ { k }$ latent positions between these tokens, where

$$
L _ { k } = \sum _ { j = 1 } ^ { m _ { k } } \left| \mathcal { E } _ { k , j } \right| ,\tag{2}
$$

and $| \mathcal { E } _ { k , j } |$ denotes the number of visual tokens selected for region $\mathbf { r } _ { k , j }$ . Each hidden state immediately preceding a latent position is supervised to predict the corresponding token in $\bar { \mathcal { E } } _ { k }$ . Textual evidence annotations, including timestamps and bounding-box coordinates, are omitted from the generation target. This produces a training trajectory that interleaves textual reasoning with the latent visual evidence corresponding to each reasoning step. Because all supervision targets are drawn from the encoded input video, the procedure does not require cropping, re-encoding, or inserting additional visual observations into the multimodal context.

## 3.3 Visual Token Prediction

Adjacent video frames often have substantial overlap in visual content, making spatially corresponding visual tokens highly similar in the feature space (Fu et al. 2025; Tao et al. 2025). Under such temporal redundancy, MSE feature regression aligns each prediction with its target independently, without explicitly distinguishing that target from visually similar tokens in the same video. We therefore introduce a visual token prediction objective over the input-video visual vocabulary. Let $\mathcal { V } = ( \mathbf { v } _ { 1 } , \ldots , \mathbf { v } _ { N } )$ denote the candidate vocabulary formed by all N visual tokens encoded from the input video. The tokens localized by spatio-temporal grounding serve as prediction targets.

For the l-th position in the latent block constructed from $\mathcal { G } _ { k }$ , let $\mathbf { h } _ { k , l }$ denote the hidden state immediately preceding that position, and let $i _ { k , l } \in \{ 1 , \dots , N \}$ denote the vocabulary index of the corresponding target token in $\widetilde { \mathcal { E } } _ { k }$ . We implement $g _ { \phi }$ as a two-layer MLP that maps $\mathbf { h } _ { k , l }$ into the visual embedding space, producing the visual query $\mathbf { q } _ { k , l } = g _ { \phi } ( \mathbf { h } _ { k , l } )$ The same projection head is used for visual token prediction during training and latent-state feedback during inference. We score each candidate visual token by its negative mean squared distance to the query:

$$
s _ { k , l , i } = - \frac { \left\| \mathbf { q } _ { k , l } - \mathbf { v } _ { i } \right\| _ { 2 } ^ { 2 } } { d \tau } ,\tag{3}
$$

where d is the feature dimension of $\mathbf { q } _ { k , l }$ and $\mathbf { v } _ { i } .$ , and $\tau$ is the temperature. The resulting visual token distribution is

$$
p _ { \mathrm { v i s } } ( i \mid \mathbf { q } _ { k , l } , \mathcal { V } ) = \frac { \exp ( s _ { k , l , i } ) } { \sum _ { n = 1 } ^ { N } \exp ( s _ { k , l , n } ) } , \quad i = 1 , \ldots , N .\tag{4}
$$

The visual token prediction objective is

$$
\mathcal { L } _ { \mathrm { V T P } } = - \frac { 1 } { M } \sum _ { k } \sum _ { l = 1 } ^ { L _ { k } } \log p _ { \mathrm { v i s } } \left( i _ { k , l } \mid { \bf q } _ { k , l } , \mathcal { V } \right) ,\tag{5}
$$

where $\begin{array} { r } { M = \sum _ { k } L _ { k } } \end{array}$ is the number of latent target positions. Non-target tokens from the same video act as competing vocabulary entries, requiring each query to identify its grounded target among the visual content of that video.

During training, the target visual embeddings are used as the input embeddings at their corresponding latent positions, while the immediately preceding hidden states are optimized by ${ \mathcal { L } } _ { \mathrm { V T P } }$ . This corresponds to teacher forcing over the latent sequence. The standard next-token objective $\mathcal { L } _ { \mathrm { t e x t } }$ is computed only over ordinary response tokens and <bot>, excluding latent positions and $< \in \mathrm { o t } >$ . This asymmetry follows the inference procedure: the model predicts <bot> to initiate latent reasoning, whereas ${ < } \ominus \mathrm { { o t } } >$ is inserted by the decoding controller after K latent steps, where K is the fixed inference budget per block. The supervised objective is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { S F T } } = \mathcal { L } _ { \mathrm { t e x t } } + \lambda \mathcal { L } _ { \mathrm { V T P } } , } \end{array}\tag{6}
$$

where λ balances text token prediction and visual token prediction. Thus, textual positions are trained through discrete next-token prediction, whereas latent positions are supervised to identify grounded visual tokens within the input video’s visual vocabulary, all within the same autoregressive framework.

## 3.4 Text–Visual Policy Optimization

We further extend VideoLVR to reinforcement learning through text–visual policy optimization (TVPO), treating text and visual token generation as actions for joint optimization. At the l-th position in the k-th latent block, we compute the visual query $\mathbf { q } _ { k , l } = g _ { \phi } ( \mathbf { h } _ { k , l } )$ . Following Equation 4, we score each token in the input-video visual vocabulary by its negative mean squared distance to the query. A temperature-scaled softmax converts these scores into visual action probabilities. We then sample a visual token action:

$$
i _ { \boldsymbol { k } , l } \sim p _ { \mathrm { v i s } } ( \cdot \mid \mathbf { q } _ { \boldsymbol { k } , l } , \mathcal { V } ) .\tag{7}
$$

Here $i _ { k , l }$ is the sampled index in the full input-video visual vocabulary V, and ${ \bf v } _ { i _ { k , l } }$ is the corresponding token embedding. In supervised training, the target index is specified by grounding annotations; during RL rollouts, the index is sampled from $p _ { \mathrm { v i s } }$ over the same vocabulary, and its embedding ${ \bf v } _ { i _ { k , l } }$ is used as the next latent input. Each sampled vocabulary index is a discrete visual action, analogous to a generated text token. The interleaved text and visual actions form a unified action sequence that can be jointly optimized with group relative policy optimization (GRPO) (Shao et al. 2024) using final-answer rewards.

Starting from the supervised model, we jointly optimize the language model and projection head, $\Theta = \left( \theta , \phi \right)$ , using GRPO. For each video-question input q, we sample $G$ reasoning trajectories $\{ o _ { i } \} _ { i = 1 } ^ { G }$ and assign reward $R _ { i } ~ = ~ 1$ if the final answer is correct and 0 otherwise. The groupnormalized advantage $A _ { i }$ is shared by all sampled text and visual token actions in trajectory $o _ { i }$ . The TVPO objective is

$$
\begin{array} { c l l } { \mathcal { T } _ { \mathrm { R L } } ( \Theta ) = \displaystyle \frac { 1 } { \mathbb { E } [ \sum _ { i = 1 } ^ { G } | o _ { i } | ] } \mathbb { E } \Bigg [ \sum _ { i = 1 } ^ { G } \sum _ { t = 1 } ^ { | o _ { i } | } } \\ { \operatorname* { m i n } \Big [ r _ { i , t } A _ { i } , \mathrm { c l i p } ( r _ { i , t } , 1 - \epsilon , 1 + \epsilon ) A _ { i } \Big ] \Bigg ] . } \end{array}\tag{8}
$$

Here $o _ { i }$ is the interleaved sequence of sampled text and visual token actions, and ϵ is the clipping threshold. The policy ratio is defined as follows:

$$
r _ { i , t } ( \Theta ) = \frac { \pi _ { \Theta } ( o _ { i , t } \mid q , o _ { i , < t } ) } { \pi _ { \Theta _ { \mathrm { o l d } } } ( o _ { i , t } \mid q , o _ { i , < t } ) } .\tag{9}
$$

Here $o _ { i , t }$ is the t-th sampled text or visual token action in $o _ { i } ,$ $o _ { i , < t }$ is the preceding output sequence, and $\Theta _ { \mathrm { o l d } }$ denotes the policy parameters before the update. At textual positions, $\pi _ { \Theta }$ uses text token probabilities. At latent positions, $o _ { i , t }$ denotes the sampled vocabulary index $i _ { k , l }$ , and $\pi _ { \Theta }$ uses its probability under $p _ { \mathrm { v i s } }$

## 3.5 Inference with Latent Visual Evidence

During supervised training, the length of each latent block is determined by the target-dependent length $L _ { k }$ defined above, whereas each generated block uses a fixed inference-time budget of K latent visual states.

VideoLVR generates text autoregressively until it emits <bot>. For the supervised model, the hidden state corresponding to <bot> is mapped by the learned projection head $g _ { \phi }$ into the first latent visual state, which is used as the input embedding for the next decoding step. This process is repeated autoregressively: each subsequent decoder hidden state is projected into a latent visual state and fed back as the next latent input. Once K latent visual states have been generated, the decoding procedure inserts <eot> and resumes text generation. For the reinforcement-learned model, we instead use the embeddings of sampled visual tokens as feedback.

## 4 Experiments

## 4.1 Experimental Setup

Training Data. We construct the training data from video question–answer samples released with OneThinker (Feng et al. 2026) and Video-R1 (Feng et al. 2025). Following our evidence-grounded data synthesis pipeline described in Section 3.1, we use Qwen3-VL-30B-A3B-Thinking (Bai et al.

2025a) as the teacher model to generate and polish evidencegrounded reasoning trajectories in which localized spatiotemporal evidence is integrated into the reasoning process. We retain only answer-consistent trajectories after quality checks on the generated reasoning and evidence annotations. This process yields approximately 8.8K high-quality trajectories for training VideoLVR during supervised learning. For reinforcement learning, we use 19K video multiple-choice questions from OneThinker, excluding questions that the supervised model answers correctly in all eight responses.

Baselines. We compare VideoLVR with video MLLMs that follow diferent reasoning paradigms. Video-R1 (Feng et al. 2025) and VideoRFT (Wang et al. 2025) enhance text-space video reasoning through supervised finetuning and reinforcement learning. OneThinker (Feng et al. 2026) provides a unified reasoning model for images and videos. Open-o3-Video (Meng et al. 2026) generates timestamps and bounding boxes within textual reasoning trajectories to provide explicit spatio-temporal grounding. VideoSearcher (Gao et al. 2026) combines temporal localization, spatial focusing, and multimodal search through multitool agentic reasoning for video deep research. LVR (Li et al. 2025) reconstructs question-relevant visual features in a continuous representation space, while Mull-Tokens (Ray et al. 2026) uses a fixed set of modality-agnostic latent tokens as an internal multimodal scratchpad. We additionally reproduce LVR with Qwen3-VL-8B-Instruct on our 8.8K synthesized trajectories using its MSE feature regression objective and denote it as LVR-8B. We evaluate the released checkpoints of the protocol-aligned baselines using our evaluation setting. For VideoSearcher, we report results from its original evaluation setting and mark them with a dagger. For reference, we report Qwen2.5-VL-7B-Instruct (Bai et al. 2025b) together with the Qwen3-VL-8B-Instruct (Bai et al. 2025a) backbone used by VideoLVR, enabling comparison with the corresponding models before video-reasoning training.

Implementation Details. We first jointly fine-tune the language model and projection head $g _ { \phi }$ while freezing the vision encoder and multimodal projector. For both training and evaluation, we sample videos at 2 FPS, retain at most 256 frames, and set the per-frame pixel budget to 131,072. Training uses a global batch size of 64 and a learning rate of $1 \times \bar { 1 0 ^ { - 5 } }$ with cosine decay and a 0.1 warmup ratio. We train for two epochs. Unless otherwise stated, we use $\lambda = 0 . 1 , \tau = 0 . 0 7 ,$ and an inference-time latent-step budget of $K = 8 . \mathrm { R I }$ L uses 64 prompts per batch, eight rollouts per prompt, a learning rate of $1 \times \dot { 1 0 } ^ { - 6 }$ , and a maximum response length of 4,096. Final-answer correctness is the only reward.

Benchmarks and Evaluation. We evaluate VideoLVR on five complementary video reasoning benchmarks. Video-MMMU (Hu et al. 2026a) assesses how models acquire and use knowledge from multidisciplinary professional videos across Perception, Comprehension, and Adaptation. TempCompass (Liu et al. 2024) evaluates temporal perception across multiple temporal aspects and task formats while reducing reliance on static visual cues. MMVU (Zhao et al. 2025) measures expert-level, knowledge-intensive reasoning over specialized videos from multiple disciplines. LongVideo-Reason (Chen et al. 2025) focuses on reasoning over long videos from diverse real-world domains. Video-MathQA (Rasheed et al. 2025) evaluates temporally extended mathematical reasoning that integrates visual, textual, and audio information from instructional videos.

<table><tr><td rowspan="2">Method</td><td colspan="4">Video-MMMU</td><td rowspan="2"></td><td rowspan="2">TempComp. MMVU LV-R. VideoMathQA Avg</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td></tr><tr><td>Percep.</td><td>Comp. Adapt.</td><td></td><td>Avg</td></tr><tr><td>Qwen2.5-VL-7B-Instruct</td><td>60.00</td><td>44.00</td><td>24.00</td><td>42.67</td><td>61.97</td><td>53.40</td><td>71.20</td><td>22.62</td><td>50.37</td></tr><tr><td>Video-R1-7B</td><td>45.33</td><td>37.67</td><td>29.00</td><td>37.33</td><td>60.83</td><td>45.90</td><td>59.30</td><td>20.00</td><td>44.67</td></tr><tr><td>VideoRFT-7B</td><td>49.33</td><td>39.33</td><td>25.67</td><td>38.11</td><td>59.76</td><td>49.10</td><td>64.00</td><td>23.10</td><td>46.81</td></tr><tr><td>Open-o3-Video-7B</td><td>60.33</td><td>41.33</td><td>35.33</td><td>45.67</td><td>72.20</td><td>51.00</td><td>63.60</td><td>24.41</td><td>51.38</td></tr><tr><td>Mull-Tokens-7B</td><td>51.33</td><td>39.67</td><td>29.00</td><td>40.00</td><td>57.41</td><td>29.20</td><td>61.40</td><td>16.90</td><td>40.98</td></tr><tr><td>LVR-7B</td><td>53.93</td><td>40.44</td><td>24.00</td><td>39.46</td><td>56.98</td><td>46.60</td><td>56.80</td><td>23.10</td><td>44.59</td></tr><tr><td>VideoLVR-7B</td><td>61.33</td><td>46.67</td><td>30.67</td><td>46.22</td><td>69.37</td><td>54.50</td><td>73.80</td><td>24.52</td><td>53.68</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>79.33</td><td>67.33</td><td>46.33</td><td>64.33</td><td>73.89</td><td>59.30</td><td>76.40</td><td>32.62</td><td>61.31</td></tr><tr><td>OneThinker-8B</td><td>75.00</td><td>60.67</td><td>39.67</td><td>58.44</td><td>70.44</td><td>57.50</td><td>74.10</td><td>30.00</td><td>58.10</td></tr><tr><td>VideoSearcher-8B†</td><td></td><td></td><td></td><td>66.33</td><td>73.14</td><td>70.40</td><td></td><td>31.67</td><td></td></tr><tr><td>LVR-8B</td><td>78.67</td><td>72.67</td><td>50.50</td><td>67.28</td><td>76.42</td><td>59.50</td><td>76.80</td><td>33.57</td><td>62.71</td></tr><tr><td>VideoLVR-8B</td><td>79.00</td><td>74.00</td><td>49.33</td><td>67.44</td><td>77.39</td><td>61.70</td><td>78.20</td><td>35.00</td><td>63.95</td></tr><tr><td>VideoLVR-8B-RL</td><td>81.00</td><td>72.33</td><td>51.33</td><td>68.22</td><td>77.60</td><td>60.50</td><td>79.90</td><td>35.71</td><td>64.39</td></tr></table>

Table 1: Main results on five video reasoning benchmarks. VideoLVR-8B-RL denotes VideoLVR-8B after training with TVPO. Percep., Comp., and Adapt. denote the Perception, Comprehension, and Adaptation dimensions of Video-MMMU. TempComp. and LV-R. abbreviate TempCompass and LongVideo-Reason, respectively. Bold indicates the best reproduced result within each backbone family, while underlining denotes the second best. <sup>†</sup> denotes results reported under the original evaluation settings, which are excluded from ranking.

We conduct all reproduced evaluations within the LMMs-Eval framework (Zhang et al. 2025) and report accuracy following the standard protocol of each benchmark; Video-MMMU further reports Perception, Comprehension, and Adaptation. Evaluation is zero-shot and allows at most 16,384 newly generated tokens. Unless otherwise stated, all reproduced results are averaged over three complete evaluations. The overall average is the unweighted mean of Video-MMMU Avg, TempCompass, MMVU, LongVideo-Reason, and VideoMathQA.

## 4.2 Main Results

As shown in Table 1, VideoLVR-8B achieves an average score of 63.95, outperforming Qwen3-VL-8B-Instruct by 2.64 points. It improves over the backbone on every benchmark, indicating that the overall gain is not driven by a single task. As a more direct latent reasoning baseline, LVR-8B applies the MSE feature regression objective of LVR to Qwen3-VL-8B-Instruct and is trained on our 8.8K synthesized trajectories. VideoLVR-8B surpasses this reproduction by 1.24 points on average and outperforms the reproduced OneThinker-8B on all five benchmarks, demonstrating consistent advantages across diferent video reasoning tasks.

The most pronounced gains are on TempCompass and Video-MMMU, with improvements of 3.50 and 3.11 points, respectively. On Video-MMMU, the largest improvement comes from Comprehension, where VideoLVR gains 6.67 points. This pattern aligns with the design of VideoLVR: during training, spatio-temporal annotations ground latent supervision in video moments and local regions, while continuous latent visual states are trained to represent the visual semantics for intermediate reasoning. VideoLVR improves MMVU, LongVideo-Reason, and VideoMathQA, suggesting that its benefits extend beyond temporal perception to expert video understanding, long-range evidence integration, and video-grounded mathematical reasoning.

<table><tr><td>Loss</td><td>s V-MMMU</td><td>TC</td><td>MMVU LV-R.</td><td>VMQA</td><td>Avg</td></tr><tr><td>MSE</td><td>67.28</td><td>76.42</td><td>59.50</td><td>76.80 33.57</td><td>62.71</td></tr><tr><td>VTP</td><td>67.44</td><td>77.39</td><td>61.70</td><td>78.20 35.00</td><td>63.95</td></tr></table>

Table 2: Comparison of MSE and visual token prediction (VTP) for latent supervision. V-MMMU, TC, LV-R., and VMQA abbreviate Video-MMMU, TempCompass, LongVideo-Reason, and VideoMathQA, respectively.

On Qwen2.5-VL-7B-Instruct, VideoLVR-7B raises the average score from 50.37 to 53.68 and achieves the highest average among the reproduced protocol-aligned methods based on Qwen2.5-VL-7B-Instruct. It outperforms Openo3-Video and LVR by 2.30 and 9.09 points on average, respectively. Compared with LVR-7B, the gains are particularly pronounced on LongVideo-Reason and TempCompass. These results show that the improvements brought by VideoLVR are not limited to a particular base model. After RL, VideoLVR-8B-RL achieves an average of 64.39. Compared with VideoLVR-8B, its average improves by 0.44 points.

## 4.3 Ablation Studies

Efect of Visual Token Prediction. Table 2 compares the two latent supervision objectives under the same backbone, training data, and evaluation protocol. Replacing MSE feature regression with visual token prediction improves the average across the five benchmarks from 62.71 to 63.95, yielding a gain of 1.24 points. The improvements are largest on MMVU, VideoMathQA, and LongVideo-Reason, with gains of 2.20, 1.43, and 1.40 points, respectively. This advantage is particularly relevant under temporal redundancy because MSE aligns each latent prediction with its target independently, whereas VTP requires it to distinguish the grounded target from other visually similar tokens in the same video. The overall gain suggests that visual token prediction is more efective than MSE feature regression in improving video understanding performance.

![](images/7668387c3b31f2f2f990bfd65fccd3b5327363da5623c8264013c5f14ee4891d.jpg)  
Latent Reasoning Steps

Figure 2: Efect of the inference-time latent-step budget on five video benchmarks. Enlarged markers denote the best results
<table><tr><td>Data</td><td>V-MMMU</td><td>TC</td><td>MMVU</td><td>LV-R. VMQA</td><td>Avg</td></tr><tr><td>Standard RS</td><td>64.89</td><td>75.15</td><td>63.60</td><td>77.60 31.43</td><td>62.53</td></tr><tr><td>Ours</td><td>67.44</td><td>77.39</td><td>61.70</td><td>78.20 35.00</td><td>63.95</td></tr></table>

Table 3: Comparison of trajectories from standard rejection sampling and our evidence-grounded synthesis pipeline.

Efect of Evidence-Grounded Training Data. We apply standard rejection sampling to the training samples and train Qwen3-VL-8B-Instruct on the resulting reasoning trajectories. Unlike our evidence-grounded training data, these trajectories do not contain localized spatio-temporal evidence. As shown in Table 3, training on evidence-grounded reasoning trajectories improves the average from 62.53 to 63.95. The largest gains appear on VideoMathQA, Video-MMMU, and TempCompass, where the scores increase by 3.57, 2.55, and 2.24 points, respectively. The improvement on LongVideo-Reason is smaller, while MMVU decreases by 1.90 points, showing that the benefit is not uniform across tasks. Overall, the results suggest that associating intermediate reasoning with localized spatio-temporal evidence generally improves video understanding over training on reasoning trajectories obtained through standard rejection sampling.

Efect of Grouped Latent Visual Evidence. We compare three ways oforganizing the localized regions associated with each evidence group. First-Region retains only the first annotated region in each group. Per-Region preserves all regions but assigns each region to a separate latent block. Grouped combines all regions into a single temporally ordered latent block, which is the strategy adopted in VideoLVR. As shown in Table 4, Grouped achieves the highest average score of 63.95, exceeding First-Region and Per-Region by 0.80 and

<table><tr><td>Setting</td><td>V-MMMU</td><td>TC</td><td>MMVU LV-R. VMQA</td><td></td><td>Avg</td></tr><tr><td>First-Region</td><td>66.89</td><td>76.76</td><td>60.70</td><td>76.90 34.52</td><td>63.15</td></tr><tr><td>Per-Region</td><td>66.56</td><td>77.05</td><td>60.10</td><td>77.30 34.52</td><td>63.11</td></tr><tr><td>Grouped</td><td>67.44</td><td>77.39</td><td>61.70</td><td>78.20 35.00</td><td>63.95</td></tr></table>

Table 4: Comparison of strategies for organizing localized visual regions into latent blocks.
<table><tr><td>Method V-MMMU</td><td></td><td>TC</td><td></td><td></td><td>MMVU LV-R. VMQA</td><td>Avg</td></tr><tr><td>GRPO</td><td>67.11</td><td>77.33</td><td>59.40</td><td>77.70</td><td>35.00</td><td>63.31</td></tr><tr><td>TVPO</td><td>68.22</td><td>77.60</td><td>60.50</td><td>79.90</td><td>35.71</td><td>64.39</td></tr></table>

Table 5: Comparison of GRPO and TVPO, both initialized from the same supervised VideoLVR-8B model.

0.84 points, respectively. These results suggest that retaining only one region may discard complementary temporal evidence, whereas assigning related regions to separate latent blocks can fragment evidence that jointly supports the same reasoning step. Grouping temporally distributed regions within one latent block better preserves their shared reasoning role. This result further validates the efectiveness of our synthesized training data grounded in localized spatiotemporal evidence.

Efect of TVPO. Table 5 compares the two RL variants using the same supervised VideoLVR-8B initialization and RL data. Our GRPO baseline uses continuous latent feedback and optimizes text actions only, whereas TVPO jointly optimizes text and visual actions. TVPO improves the fivebenchmark average from 63.31 to 64.39, a gain of1.08 points, and outperforms this baseline on all five benchmarks.

## 4.4 Additional Analysis

Efect of the Inference-Time Latent-Step Budget. Figure 2 varies the inference-time latent-step budget from 0 to 32 while keeping the trained model fixed. Disabling latent visual state generation (K = 0) yields an average of 62.12, 1.83 points below the 63.95 achieved by K = 8, with lower performance on all five benchmarks. Allocating four latent steps recovers part of this gap, raising the average to 62.82, while eight steps provide the strongest overall result. Increasing the budget further to 16 or 32 reduces the average to 62.00 and 62.31, respectively. At the benchmark level, K = 8 performs best on Video-MMMU, MMVU, LongVideo-Reason, and VideoMathQA, and ties K = 4 on TempCompass. These results demonstrate that latent visual states contribute to performance, while also showing that their benefit depends on a moderate inference-time latent-step budget rather than simply increasing the number of latent steps.

![](images/75bc751d3f22eaba4fea30fdb657e25eb517830200566e4762f2ac0a11ffef34.jpg)

Figure 3: Comparison of training reward curves for GRPO and TVPO, shown as trailing 20-step moving averages.
<table><tr><td>T</td><td>V-MMMU</td><td>TC</td><td>MMVU</td><td>LV-R.</td><td>VMQA</td><td>Avg</td></tr><tr><td>0.01</td><td>62.66</td><td>73.75</td><td>57.70</td><td>75.97</td><td>29.84</td><td>59.98</td></tr><tr><td>0.03</td><td>65.04</td><td>75.64</td><td>59.27</td><td>77.95</td><td>30.24</td><td>61.63</td></tr><tr><td>0.07</td><td>67.44</td><td>77.39</td><td>61.70</td><td>78.20</td><td>35.00</td><td>63.95</td></tr><tr><td>0.10</td><td>66.00</td><td>76.59</td><td>59.53</td><td>77.10</td><td>34.52</td><td>62.75</td></tr><tr><td>0.15</td><td>66.35</td><td>77.09</td><td>60.30</td><td>76.37</td><td>32.78</td><td>62.58</td></tr></table>

Table 6: Ablation of the visual token prediction temperature τ on the five main benchmarks.

RL Training Dynamics. Figure 3 compares the training reward curves of GRPO and TVPO on our 19K RL dataset. Both methods show an overall increase in reward after the initial fluctuations. TVPO achieves higher rewards over most training steps and maintains this advantage in the later stages. Since the reward measures final-answer correctness, higher rewards correspond to a larger proportion of correct responses among the sampled training trajectories. These results provide training-time evidence for the efectiveness of TVPO relative to the GRPO baseline.

Efect of Vocabulary Temperature. Table 6 examines the efect of the temperature used for visual token prediction. Performance improves steadily as τ increases from 0.01 to 0.07, with the average rising from 59.98 to 63.95, but declines at 0.10 and 0.15. The τ = 0.07 setting also achieves the best result on all five benchmarks. Since τ controls the sharpness of visual token discrimination, these results suggest that an overly small value produces excessively hard supervision, whereas a larger value weakens the discrimination objective; τ = 0.07 provides a more efective balance.

<table><tr><td colspan="4">Setting V-MMMU TC MMVU LV-R. VMQA Avg</td></tr><tr><td>Zero</td><td>65.11 76.81</td><td>61.00 76.30 34.52</td><td>62.75</td></tr><tr><td>Noise</td><td>65.00</td><td>76.86 60.70 76.10 31.67</td><td>62.07</td></tr><tr><td>Ours</td><td>67.44</td><td>77.39 61.70 78.20 35.00</td><td>63.95</td></tr></table>

Table 7: Efect of interventions on latent states during inference.

Efect of Latent-State Content. To further investigate the role of latent-state content in VideoLVR, we intervene on the feedback vector at every latent step while keeping the eight-step latent budget and boundary markers unchanged. Zero replaces each feedback vector with a zero vector. Noise replaces it with a random vector whose $\ell _ { 2 }$ norm matches that of the original feedback vector, preserving its magnitude while removing its learned content. As shown in Table 7, both interventions degrade performance across all five benchmarks, reducing the average score from 63.95 to 62.75 and 62.07, respectively. These results indicate that the information encoded in the generated latent states contributes to VideoLVR’s performance.

## 5 Conclusion

In this paper, we introduce VideoLVR, a framework that grounds video reasoning with latent visual evidence constructed from temporally distributed local regions. VideoLVR groups regions according to the reasoning step they support and trains the model to generate the corresponding latent visual states through visual token prediction, thereby interleaving textual reasoning with latent visual evidence in a shared autoregressive process. We further extend VideoLVR to reinforcement learning through TVPO, jointly optimizing text and visual token actions with GRPO. Across five video reasoning benchmarks and two backbone models, VideoLVR consistently improves over its base models. Controlled analyses support the benefits of evidence-grounded training data, grouped evidence organization, and visual token prediction, while latent-state interventions further show that the model benefits from the information carried by the generated latent states. Together, these results indicate that localized video evidence can be represented and used during reasoning without inserting additional visual observations into the multimodal context. A remaining limitation is our reliance on synthesized spatio-temporal grounding annotations. Future work could broaden the representational scope of latent visual states beyond localized evidence, enabling them to encode higher-level semantic information.

## References

Bai, S.; Cai, Y.; Chen, R.; Chen, K.; Chen, X.; Cheng, Z.;

Deng, L.; Ding, W.; Gao, C.; Ge, C.; Ge, W.; Guo, Z.; Huang,

Q.; Huang, J.; Huang, F.; Hui, B.; Jiang, S.; Li, Z.; Li, M.;

Li, M.; Li, K.; Lin, Z.; Lin, J.; Liu, X.; Liu, J.; Liu, C.; Liu,

Y.; Liu, D.; Liu, S.; Lu, D.; Luo, R.; Lv, C.; Men, R.; Meng,

L.; Ren, X.; Ren, X.; Song, S.; Sun, Y.; Tang, J.; Tu, J.; Wan,

J.; Wang, P.; Wang, P.; Wang, Q.; Wang, Y.; Xie, T.; Xu, Y.;

Xu, H.; Xu, J.; Yang, Z.; Yang, M.; Yang, J.; Yang, A.; Yu,

B.; Zhang, F.; Zhang, H.; Zhang, X.; Zheng, B.; Zhong, H.; Zhou, J.; Zhou, F.; Zhou, J.; Zhu, Y.; and Zhu, K. 2025a. Qwen3-VL Technical Report. arXiv:2511.21631.

Bai, S.; Chen, K.; Liu, X.; Wang, J.; Ge, W.; Song, S.; Dang, K.; Wang, P.; Wang, S.; Tang, J.; Zhong, H.; Zhu, Y.; Yang, M.; Li, Z.; Wan, J.; Wang, P.; Ding, W.; Fu, Z.; Xu, Y.; Ye, J.; Zhang, X.; Xie, T.; Cheng, Z.; Zhang, H.; Yang, Z.; Xu, H.; and Lin, J. 2025b. Qwen2.5-VL Technical Report. arXiv:2502.13923.

Chen, Y.; Huang, W.; Shi, B.; Hu, Q.; Ye, H.; Zhu, L.;Liu, Z.; Molchanov, P.; Kautz, J.; Qi, X.; Liu, S.; Yin, H.;Lu, Y.; and Han, S. 2025. Scaling RL to Long Videos.arXiv:2507.07966.

Fan, Y.; He, X.; Yang, D.; Zheng, K.; Kuo, C.-C.; Zheng, Y.; Guan, X.; and Wang, X. 2025. GRIT: Teaching MLLMs to Think with Images. In Belgrave, D.; Zhang, C.; Lin, H.; Pascanu, R.; Koniusz, P.; Ghassemi, M.; and Chen, N., eds., Advances in Neural Information Processing Systems, volume 38, 116522–116543. Curran Associates, Inc.

Feng, K.; Gong, K.; Li, B.; Guo, Z.; Wang, Y.; Peng, T.; Wu, J.; Zhang, X.; Wang, B.; and Yue, X. 2025. Video-R1: Reinforcing Video Reasoning in MLLMs. In Belgrave, D.; Zhang, C.; Lin, H.; Pascanu, R.; Koniusz, P.; Ghassemi, M.; and Chen, N., eds., Advances in Neural Information Processing Systems, volume 38, 99114–99137. Curran Associates, Inc.

Feng, K.; Zhang, M.; Li, H.; Fan, K.; Chen, S.; Jiang, Y.; Zheng, D.; Sun, P.; Zhang, Y.; Sun, H.; Feng, Y.; Pei, P.; Cai, X.; and Yue, X. 2026. OneThinker: All-in-one Reasoning Model for Image and Video. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 5432–5443.

Fu, T.; Liu, T.; Han, Q.; Dai, G.; Yan, S.; Yang, H.; Ning, X.; and Wang, Y. 2025. FrameFusion: Combining Similarity and Importance for Video Token Reduction on Large Vision Language Models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 22654– 22663.

Gao, Z.; Bao, Y.; Peng, J.; Li, X.; Huang, T.; Liu, B.; Li, K.; Gan, Z.; Hu, T.; Xie, C.; Yang, M.; He, X.; Zhang, Z.; Tan, X.; Wang, C.; and Xie, Y. 2026. VideoSearcher: Empowering Video Deep Research with Multi-Tool Agentic Reasoning via Reinforcement Learning. arXiv:2607.02927.

Hao, S.; Sukhbaatar, S.; Su, D.; Li, X.; Hu, Z.; Weston, J.; and Tian, Y. 2025. Training Large Language Models to Reason in a Continuous Latent Space. arXiv:2412.06769.

Hu, K.; Wu, P.; Pu, F.; Xiao, W.; Yue, X.; Li, B.; Zhang, Y.; and Liu, Z. 2026a. Video-MMMU: Evaluating Knowledge Acquisition from Multidisciplinary Professional Videos. In Liakata, M.; Moreira, V. P.; Zhang, J.; and Jurgens, D., eds., Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 27798–27828. San Diego, California, United States: Association for Computational Linguistics. ISBN 979-8-89176- 390-6.

Hu, Z.-Y.; Tang, Z.; Huang, S.; Li, Y.; Lyu, M. R.; and Wang,

L. 2026b. VideoLatent: Video-Language Learning via Latent Self-Forcing. arXiv:2606.22870.

Jiang, C.; Heng, Y.; Ye, W.; Xu, H.; Yan, M.; Zhang, J.; Huang, F.; and Zhang, S. 2025. VLM-R<sup>3</sup>: Region Recognition, Reasoning, and Refinement for Enhanced Multimodal Chain-of-Thought. In Belgrave, D.; Zhang, C.; Lin, H.; Pascanu, R.; Koniusz, P.; Ghassemi, M.; and Chen, N., eds., Advances in Neural Information Processing Systems, volume 38, 63841–63869. Curran Associates, Inc.

Li, B.; Sun, X.; Liu, J.; Wang, Z.; Wu, J.; Yu, X.; Chen, H.; Barsoum, E.; Chen, M.; and Liu, Z. 2025. Latent Visual Reasoning. arXiv:2509.24251.

Li, K.; Shang, C.; Karlinsky, L.; Feris, R.; Darrell, T.; and Herzig, R. 2026. Latent Implicit Visual Reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 33457–33466.

Liu, Y.; Li, S.; Liu, Y.; Wang, Y.; Ren, S.; Li, L.; Chen, S.; Sun, X.; and Hou, L. 2024. TempCompass: Do Video LLMs Really Understand Videos? In Ku, L.-W.; Martins, A.; and Srikumar, V., eds., Findings of the Association for Computational Linguistics: ACL 2024, 8731–8772. Bangkok, Thailand: Association for Computational Linguistics.

Maaz, M.; Rasheed, H.; Khan, F. S.; and Khan, S. 2025. Video-R2: Reinforcing Consistent and Grounded Reasoning in Multimodal Language Models. arXiv:2511.23478.

Meng, J.; Li, X.; Wang, H.; Tan, Y.; Zhang, T.; Kong, L.; Tong, Y.; Wang, A.; Teng, Z.; Wang, Y.; and Wang, Z. 2026. Open-o3-Video: Grounded Video Reasoning with Explicit Spatio-Temporal Evidence. arXiv:2510.20579.

Miao, Y.; Sun, Y.; Wang, D.; Zhou, M.; Poupart, P.; Lv, L.; Zhao, Q.; Wang, L.; Li, H.; Jiang, X.; and Jiang, G. 2026. Fill the GAP: A Granular Alignment Paradigm for Visual Reasoning in Multimodal Large Language Models. arXiv:2605.12374.

Rasheed, H.; Shaker, A.; Tang, A.; Maaz, M.; Yang, M.-H.; Khan, S.; and Khan, F. S. 2025. VideoMathQA: Benchmarking Mathematical Reasoning via Multimodal Understanding in Videos. arXiv:2506.05349.

Ray, A.; Abdelkader, A.; Mao, C.; Plummer, B. A.; Saenko, K.; Krishna, R.; Guibas, L.; and Chu, W.-S. 2026. Mull-Tokens: Modality-Agnostic Latent Thinking. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings, 9477–9488.

Ren, Z.; Wei, Y.; Zhao, Y.; Gong, W.; Liu, X.; Wang, A.; Li, X.; and Jin, X. 2026. UniVR: Thinking in Visual Space for Unified Visual Reasoning. arXiv:2607.12800.

Shao, Z.; Wang, P.; Zhu, Q.; Xu, R.; Song, J.; Bi, X.; Zhang, H.; Zhang, M.; Li, Y.; Wu, Y.; et al. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Sun, H.-L.; Sun, Z.; Peng, H.; and Ye, H.-J. 2025. Mitigating Visual Forgetting via Take-along Visual Conditioning for Multi-modal Long CoT Reasoning. In Che, W.; Nabende, J.; Shutova, E.; and Pilehvar, M. T., eds., Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 5158–5171. Vienna,

Austria: Association for Computational Linguistics. ISBN 979-8-89176-251-0.

Sun, X.; Wang, Y.; Cao, H.; and Xu, L. 2026. When Thinking Hurts: Mitigating Visual Forgetting in Video Reasoning via Frame Repetition. arXiv:2603.16256.

Tao, K.; Qin, C.; You, H.; Sui, Y.; and Wang, H. 2025. DyCoke: Dynamic Compression of Tokens for Fast Video Large Language Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 18992–19001.

Viveiros, A. G.; Gonçalves, N.; Martins, A. F. T.; and Lindemann, M. 2026. What’s Holding Back Latent Visual Reasoning? arXiv:2605.18445.

Wang, H.; Li, X.; Huang, Z.; Wang, A.; Wang, J.; Zhang, T.; Zheng, J.; Bai, S.; Kang, Z.; Feng, J.; Wang, Z.; and Zhang, Z. 2026a. Traceable Evidence Enhanced Visual Grounded Reasoning: Evaluation and Methodology. arXiv:2507.07999.

Wang, J.; Kang, Z.; Wang, H.; Jiang, H.; Li, J.; Wu, B.; Wang, Y.; Ran, J.; Liang, X.; Feng, C.; and Xiao, J. 2026b. VGR: Visual Grounded Reasoning. arXiv:2506.11991.

Wang, Q.; Shi, Y.; Wang, Y.; Zhang, Y.; Wan, P.; Gai, K.; Ying, X.; and Wang, Y. 2026c. Monet: Reasoning in Latent Visual Space Beyond Image and Language. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 12030–12040.

Wang, Q.; Yu, Y.; Yuan, Y.; Mao, R.; and Zhou, T. 2025. VideoRFT: Incentivizing Video Reasoning Capability in MLLMs via Reinforced Fine-Tuning. In Belgrave, D.; Zhang, C.; Lin, H.; Pascanu, R.; Koniusz, P.; Ghassemi, M.; and Chen, N., eds., Advances in Neural Information Processing Systems, volume 38, 4350–4376. Curran Associates, Inc.

Wu, L.; Jiang, T.; Dong, Y.; Yang, H.; Zhang, F.; Meng, S.; Xuan, A.; Song, L.; and Keung, J. 2026. LaViT: Aligning Latent Visual Thoughts for Multi-modal Reasoning. arXiv:2601.10129.

Xu, T.; Sun, Y.; Wang, Q.; Lu, J.; Wang, Y.; Zhang, T.; Guo, L.; Rao, F.; Lyu, J.; Chen, F.; and Liu, J. 2026. Semantic-Enriched Latent Visual Reasoning. arXiv:2605.19342.

Yang, Z.; Yu, X.; Chen, D.; Shen, M.; and Gan, C. 2026. Machine Mental Imagery: Empower Multimodal Reasoning with Latent Visual Tokens. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 33510–33520.

Zhang, H.; Gu, X.; Li, J.; Ma, C.; Bai, S.; Zhang, C.; Zhang, B.; Zhou, Z.; He, D.; and Tang, Y. 2026a. Thinking With Videos: Multimodal Tool-Augmented Reinforcement Learning for Long Video Reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 32903–32914.

Zhang, K.; Li, B.; Zhang, P.; Pu, F.; Cahyono, J. A.; Hu, K.; Liu, S.; Zhang, Y.; Yang, J.; Li, C.; and Liu, Z. 2025. LMMs-Eval: Reality Check on the Evaluation of Large Multimodal Models. In Chiruzzo, L.; Ritter, A.; and Wang, L., eds., Findings of the Association for Computational Linguistics: NAACL 2025, 881–916. Albuquerque, New Mexico: Association for Computational Linguistics. ISBN 979-8-89176- 195-7.

Zhang, X.; Tao, Q.; Du, J.; Liu, M.; and Zhou, J. T. 2026b. Visual Latents Know More Than They Say: Unsilencing Latent Reasoning in MLLMs. arXiv:2605.02735.

Zhao, Y.; Zhang, H.; Xie, L.; Hu, T.; Gan, G.; Long, Y.; Hu, Z.; Chen, W.; Li, C.; Xu, Z.; Wang, C.; Shangguan, Z.; Liang, Z.; Liu, Y.; Zhao, C.; and Cohan, A. 2025. MMVU: Measuring Expert-Level Multi-Discipline Video Understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 8475–8489.

Zheng, Z.; Yang, M.; Hong, J.; Zhao, C.; Xu, G.; Yang, L.; Shen, C.; and Yu, X. 2026. DeepEyes: Incentivizing "Thinking with Images" via Reinforcement Learning. arXiv:2505.14362.

<box t="timestamp" content="a brief   
description of the bbox region">[x1,y1,x2,   
y2]</box>   
</evidence>   
Each evidence group supports one short,   
specific claim and may contain one or more   
boxes from the same or different video   
moments. Insert the group directly into   
the reasoning sentence whose judgment it   
supports, rather than presenting a   
separate evidence list. Use at most 10   
groups and prefer evidence most relevant   
to the answer.   
Inspect the provided video directly. Do   
not invent visual facts, timestamps, or   
coordinates. Copy each timestamp exactly   
from a visible video-context timestamp   
marker. Use tight, valid bounding boxes   
that exclude irrelevant background and do   
not cover almost the entire frame. Output   
normalized coordinates in [0,1], with the   
origin at the top-left.   
For a multiple-choice question, place the   
reasoning inside <think>...</think>,   
followed by exactly one option letter and   
no additional text. Follow the required   
format strictly.

## A Supervised Training Data Statistics

The supervised training set contains 8,782 evidencegrounded reasoning trajectories. Each trajectory is interleaved with one or more spatio-temporal visual evidence groups, with each group associating an intermediate reasoning claim with one or more localized visual regions. In total, the supervised training set contains 25,710 evidence groups and 32,180 localized visual regions, corresponding to 2.93 groups and 3.66 regions per trajectory, respectively.

<table><tr><td>Statistic</td><td>Value</td></tr><tr><td>Training Trajectories</td><td>8,782</td></tr><tr><td>Avg. Groups</td><td>2.93</td></tr><tr><td>Avg. Regions</td><td>3.66</td></tr><tr><td>Multi-Region Groups</td><td>19.8%</td></tr><tr><td>Multi-Moment Groups</td><td>14.0%</td></tr></table>

Table 8: Statistics of the supervised training set. Both averages are computed over all evidence-grounded reasoning trajectories; the last two rows report percentages over all evidence groups.

As shown in Table 8, each trajectory contains 2.93 evidence groups and 3.66 localized visual regions on average. The larger number of regions reflects that some evidence groups are supported by multiple localized visual regions. In particular, 19.8% of the groups contain multiple regions, and 14.0% span multiple video moments. The training data therefore includes groups that capture spatially distributed evidence within a frame or temporally distributed evidence across frames, providing supervision for the grouped latent visual evidence used by VideoLVR.

## B Prompts for Evidence-Grounded Trajectory Synthesis

We use two prompts in the data-synthesis pipeline described in Section 3.1. The first directly generates a reasoning trajectory interleaved with spatio-temporal visual evidence groups, while the second polishes a rejection-sampled trajectory by relocating each evidence group to the reasoning step that uses it.

## B.1 Evidence-Grounded Trajectory Rejection Sampling

You are a video understanding expert. Your   
task is to watch the video carefully and   
answer the question. When answering,   
reason step by step according to the   
requirements and rely as much as possible   
on visible visual evidence.   
Mark key visual evidence using:   
<evidence claim="a short judgment   
supported by this group of visual evidence   
">

User Prompt   
Video question:   
{question}   
Options:   
{options}   
Please follow the system instructions,   
reason carefully, and output the final   
answer as exactly one option letter.

## B.2 Evidence-Guided Trajectory Polishing

System Prompt   
You are a professional reasoning-data   
synthesis expert. Think privately and   
deeply, but expose only the requested JSON   
object in your final response.   
Task   
Reorganize the rejection-sampled   
trajectory so that each evidence group   
appears immediately before the reasoning   
statement it supports. Preserve the   
original reasoning content, rewriting   
surrounding text only as needed to   
integrate the relocated group. The result   
should read like a capable model naturally   
solving the question from the visual

evidence, not like an edit report.   
Reasoning-quality requirements   
- Preserve every useful observation,   
comparison, option elimination, temporal/   
spatial relation, and inference in the   
original trajectory.   
- Preserve meaningful false starts,   
provisional hypotheses, uncertainty,   
repeated verification, and self-correction   
. Do not turn tentative statements into   
certainty.   
- Preserve the original reasoning order   
and wording whenever possible. Relocate   
each evidence group to the first reasoning   
step that uses its visual content.   
- Preserve the original factual and   
logical content and final answer. Do not   
omit substantive reasoning or invent   
visual facts.   
- Start directly with solving the problem.   
Do not mention data construction, prompts   
, placeholders, JSON, evidence IDs, model   
limitations, or the mechanics of selecting   
timestamps and boxes.   
Protected evidence placeholders   
- Tokens such as {{E001}} represent exact   
evidence XML that Python restores later.   
- Use every supplied placeholder exactly   
once, character-for-character. Never   
invent one.   
- Place each placeholder where its visual   
content is first inspected, before the   
resulting claim or conclusion. Do not   
collect placeholders at the end. All   
placeholders must precede the final answer   
synthesis.   
- Use the Evidence Catalog to understand   
what each placeholder shows, but never   
copy coordinates, timestamps, IDs, XML, or   
catalog formatting into prose.   
Output contract   
- Return one valid JSON object with   
exactly two keys: think\_content and answer   
- think\_content is the complete reasoning   
body only. It must not contain <think>, </   
think>, <answer>, evidence XML, bbox XML,   
or a standalone final-answer line.   
- answer is exactly one uppercase option   
letter such as "C".   
Do not wrap the JSON in Markdown.   
Schema only; the ellipsis is not a length   
example:   
"think\_content": "...",   
"answer": "C"

User Prompt   
<video>   
## Task Context   
Question:   
{question}   
Options:   
{options}   
## Original Reasoning   
{original\_reasoning\_with\_placeholders}   
## Original Content   
{original\_content\_with\_placeholders}   
## Reference Final Answer   
{reference\_answer}   
## Evidence Catalog   
{evidence\_catalog}   
Retain substantive observations, option   
analysis, evidence interpretation,   
hypotheses, corrections, and verification   
without adding filler. Before answering,   
privately verify that no substantive   
reasoning stage or self-correction was   
omitted.   
Return only the JSON object with   
think\_content and answer.

In the polishing prompt above, Original Reasoning refers to the content inside the think block of the rejectionsampled response, while Original Content refers to the content following that block.

## C Case Study

We present a representative example from the supervised training set. The question requires recognizing the main activity from visual evidence distributed across the video. Figure 4 shows the four localized visual regions used by two evidence groups. The bounding boxes are shown in the normalized coordinate representation used by VideoLVR.

Question   
<video>   
What is the main activity shown in the   
video?   
A. Boiling water   
B. Deep-frying food   
C. Baking a cake   
D. Making a salad

![](images/aeeec22712c9ada4ba91cb9e90f70ae4b49a54a5d0c49f44775def2776e389ac.jpg)  
(a) 2.3 s: food is added.

![](images/1fd3972205009107804fb87503c1117da3ec577f5b3650b88f28445aa80f8a37.jpg)  
(b) 6.4 s: oil is bubbling.

![](images/c86874d9ea4563d56d3ed200e7f3d22d5dfb5c05738f1c9e64cb2f01d3135004.jpg)  
(c) 20.7 s: food is lifted.

![](images/75beadb7432333a44c5547a2cac1e0efb90cea61fed3a990e1258cd1687d6e61.jpg)  
(d) 22.7 s: food is drained.  
Figure 4: Localized visual regions in a representative training example. Red boxes mark the regions annotated in the evidence groups. Panels (a)–(b) form the first group, while panels (c)–(d) form the second.

## Evidence-Grounded Reasoning Trajectory Trajectory

<think>   
The video shows a person adding pieces of   
food into a wok with hot oil. <evidence   
claim="the oil in the wok is bubbling   
vigorously as food is added"><box t="2.3"   
content="the wok with hot oil and the   
first piece of food being added   
">[0.15,0.10,0.85,0.65]</box><box t="6.4"   
content="the oil bubbling around the food   
pieces">[0.15,0.10,0.85,0.65]</box></   
evidence> shows that the oil bubbles   
vigorously around the food as it is added,   
which is characteristic of deep-frying.   
This rules out boiling water (A) because   
boiling water does not produce such   
vigorous bubbling with solid food pieces   
and the oil is dark and not clear water.   
Baking (C) and making a salad (D) are also   
eliminated since there is no oven or   
mixing of raw vegetables. As <evidence   
claim="the person uses a skimmer to remove   
the fried food from the oil"><box t   
="20.7" content="a skimmer lifting the   
fried food from the oil   
">[0.10,0.10,0.80,0.60]</box><box t="22.7"   
content="the fried food being placed on a   
paper towel to drain   
">[0.15,0.10,0.85,0.65]</box></evidence>   
shows, the person uses a skimmer to remove   
the fried food from the oil and drain it,   
which is a common step in deep-frying.   
Therefore, the main activity is deep  
frying food.   
</think>   
B

tween them. A training block of length $L _ { k }$ therefore has the form <bot><latent>...<latent><eot>. The hidden state immediately preceding each <latent> position is supervised to predict the corresponding visual token in the grouped target. During decoding, generating <bot> initiates latent-state feedback; after K latent inputs, the controller inserts <eot> and resumes text generation. The supervised checkpoint uses projected hidden states as feedback, whereas the reinforcement-learned model samples visual token indices from the visual token distribution and uses the corresponding embeddings as feedback. These sampled indices are discrete visual actions during RL rollout, and their likelihoods enter the TVPO objective in Section 3.4. This yields a reasoning trajectory interleaved with two latent blocks while omitting textual evidence annotations, including timestamps and bounding-box coordinates, from the generation target.

This trajectory contains two evidence groups. The first uses regions at 2.3 and 6.4 seconds to ground the visual pattern of food bubbling in hot oil. The second group uses regions at 20.7 and 22.7 seconds to ground the subsequent removal and draining of the fried food. During training, the two regions within each group are ordered by timestamp and concatenated into one grouped latent target. Each evidence group is then replaced by its corresponding latent block. Concretely, we use <bot> and <eot> as the start and end markers, respectively, and <latent> for each latent position be-