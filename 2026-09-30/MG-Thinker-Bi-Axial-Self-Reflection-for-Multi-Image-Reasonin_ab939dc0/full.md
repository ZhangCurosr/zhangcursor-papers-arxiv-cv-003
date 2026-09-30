# MG-Thinker: Bi-Axial Self-Reflection for Multi-Image Reasoning Grounding

Heyu Huang Huazhong University of Science and Technology Wuhan, China

Yuhua Li Huazhong University of Science and Technology Wuhan, China

Chi Chen   
Tsinghua University   
Beijing, China

Maosong Sun Tsinghua University Beijing, China

Zonghao Guo Tsinghua University Beijing, China

Ruixuan Li Huazhong University of Science and Technology Wuhan, China

## Abstract

Reinforcement learning (RL) has recently delivered substantial gains in multimodal reasoning, opening a promising route for finegrained visual perception. Yet for multi-image reasoning grounding (MRG), reasoning over real-world multi-image contexts toward pixel-precise localization, existing RL-based approaches overlook two characteristics intrinsic to this paradigm: a coarse-to-fine hierarchical reasoning pattern, and heterogeneously distributed task– sample dificulties. In this work, we present MG-Thinker, a posttraining RL framework that advances a new MRG paradigm featuring such hierarchical reasoning, supported by a curated 25K MRG dataset with task-adaptive Chain-of-Thought (CoT) annotations that elicit multi-perspective evidence before conclusion. To remedy the heterogeneous task–sample dificulties, we further propose Bi-Axial DAPO (BiA-DAPO), which decomposes rollout advantages along an intra-group signal axis and an inter-group competence axis through two complementary mechanisms, both grounded on our defined candidate pool for stable group-level statistics. Extensive experiments show that MG-Thinker achieves state-of-the-art performance on multi-image reasoning grounding while consistently improving generalization across multi-image understanding and diverse multimodal benchmarks.

## CCS Concepts

• Computing methodologies → Artificial intelligence.

## Keywords

Multimodal large language models, multi-image reasoning grounding, reinforcement learning

## 1 Introduction

Recent Multimodal Large Language Models (MLLMs) [2, 7, 22, 26, 38] have delivered promising progress in multimodal perception, spanning modal recognition [12, 21, 44, 51], single-image grounding [6, 30, 53, 59], and multi-image understanding [15, 17, 52]. However, real-world deployment demands more fine-grained perception that generalizes across sequential images, videos, or multiview captures while producing pixel-level precise outputs. Recent works [3, 18] have begun to focus multi-image grounding, expanding from single-image perception toward cross-image free-form grounding queries. However, existing MLLMs typically rely on end-to-end direct prediction and sufer from grounding drift, unstable output formats, and weak cross-image alignment. Meanwhile, o1-style slow thinking [11, 14, 36, 48] and R1-like multimodal reasoning [5, 34, 57] have markedly advanced visual reasoning but mainly in single-image scenarios, directly transplanting such singleimage CoT into multi-image localization yields unfaithful rationales that misroute attention and amplify localization errors. These limitations motivate Multi-image Reasoning Grounding (MRG), a new paradigm that couples reasoning traces cross-image with precise pixel-level grounding.

![](images/0958f02dc207d0e10d09fd839ee421a275986445795f0a2760cabc8004951147.jpg)  
Figure 1: Illustrative case showing that multi-image reasoning grounding demands a coarse-to-fine hierarchical pattern (CoT→image-id→bbox), absent in representative baselines, which MG-Thinker explicitly delivers through oneshot multi-image multi-object localization.

This paradigm presents two MRG-specific challenges that prior work largely overlooks. First, MRG inherently demands a coarseto-fine hierarchical pattern, progressively unfolding from semanticlevel reasoning to image-level routing to region-level localization, as shown in Fig. 1, unlike generic tasks where templated reasoning or direct prediction sufices. Second, during post-training, MRG further exhibits pronounced task-sample dificulty heterogeneity. As shown in Fig. 2 (left), the base model (Qwen2.5-VL) trails far below the three general tasks on MRG in both mean reward and rollout-std, confirming that MRG as a whole is substantially harder than these general tasks. In contrast, the CoT-SFT model (the finetuned model before RL, detailed in Sec. 3.2) swings widely across MRG sub-tasks while remaining stable on general tasks, jointly evidencing inter-task and sample-level dificulty heterogeneity.

![](images/be8ebb2d596ede60536147a5f682f431e4bab65b86436b134ae840f4495f8976.jpg)

![](images/1f75723edde7e4f679a32178130ce5dbb119b880fa355cd4d19ac7b9325e23da.jpg)  
Figure 2: (Left) Task–sample dificulty heterogeneity profile of MRG vs. general tasks: per-task mean reward (bars) and within sample rollout-std (lines) over the base model (Qwen2.5-VL) and the CoT-SFT model; S-Grd, S-Gen, M-Gen denote single-image grounding, single-image and multi-image understanding tasks, respectively. (Right) Per-task accuracy radar over the MIG-Bench.

Building on these challenges, we present MG-Thinker, an RLbased framework that addresses both characteristics through complementary data and algorithm designs. Data side. To instill the coarse-to-fine hierarchy, we restructure MGrounding-630K and curate a 25K MRG-specific CoT dataset with task-adaptive cue prompts, so that multi-image evidence guides reasoning and reasoning, in turn, sharpens localization. This CoT design further folds in one-shot multi-image multi-target grounding from the start, ad dressing a joint-localization scenario that both direct-grounding and reasoning-based baselines noticeably struggle with. Algorithm side. Heterogeneous task-sample dificulties induce two recurring training instabilities, drifting advantage-signal sparsity from rollout groups with near-zero reward variance and competence-dificulty misalignment from groups with widely diferent mean rewards. We thus propose Bi-Axial DAPO (BiA-DAPO), which stabilizes RL post-training through two simple group-level mechanisms: Group-Informativeness Assessment (GIA) selects informative rollout groups via within-group reward variance to relieve the drifting sparsity, while Cascaded-Reward Stratification (CRS) schedules updates over retained groups by group-level mean reward to match the policy’s evolving competence, both grounded on a shared candidate pool for stable group-level statistics. This bi-axial view treats the group mean and within-group variance as statistically orthogonal signals that carry independent information about competence and informativeness, so a rollout group meaningfully contributes to the policy update only when well-positioned on both axes. As shown in Fig. 2 (right), MG-Thinker sets a new state of the art on MIG-Bench [18] while extending consistent gains to broader multi-image and multimodal benchmarks.

Overall, our main contributions can be summarized as follows:

• We present MG-Thinker, a post-training RL framework that targets two MRG-specific characteristics overlooked by prior work, a coarse-to-fine hierarchical reasoning–grounding pattern, and heterogeneously distributed task–sample dificulties.

• Data-side, we restructure MGrounding-630K and distill a 25K MRG-specific CoT dataset with task-adaptive cue prompts that instills the coarse-to-fine hierarchy. Algorithmside, we propose Bi-Axial DAPO (BiA-DAPO), which improves group-relative RL by combining Group-Informativeness Assessment (reward-variance-based group selection) with Cascaded-Reward Stratification (reward-mean-based competence scheduling), both grounded on a shared candidate pool for stable group-level statistics.

• Extensive experiments show that MG-Thinker achieves state-of-the-art performance on multi-image reasoning grounding while consistently generalizing across multi-image understanding, single-image grounding, and diverse multimodal benchmarks.

## 2 Related Work

Visual Grounding. Visual grounding[6, 30, 32, 46, 53, 56, 59] has evolved from referring-expression localization in single images, as in the RefCOCO series[28, 54], to reasoning-intensive instruction grounding (e.g., LISA-Grounding [16]), and further extends to free-form multi-image grounding paradigms closer to real-world applications. However, existing MLLMs remain limited in multi-image fine-grained grounding due to insuficient cross-image association precision [9, 58] and the lack of cross-image reasoning capability for free-form queries.

Reinforcement Learning-based Reasoning. MLLM reasoning has shown strong gains on visual reasoning [27, 42]. Prior works mostly construct CoT data with explicit reasoning steps [8, 13, 19, 33, 55] for SFT. With R1-like models, RL has emerged as a post-training paradigm for eliciting reasoning [25], mostly built on GRPO with rule-based rewards. Recent improvements span data construction, staged training, and reward/loss design [47]. VL-Rethinker [40] replays high-value samples to address advantage vanishing, while AdaRFT [35] adaptively ofline schedules sample dificulty against the policy’s running rewards.

![](images/7ab4fbc84da935ff66fac1640e4e4cee36d1ba78191d6a35b2281e95524c16b7.jpg)  
Figure 3: Illustration of the multi-image grounding data pre-processing pipeline, including rule-based criteria and processing operations derived using the multimodal large language model Qwen2.5-VL-72B.

## 3 Proposed Method

## 3.1 Overview

Multi-image reasoning grounding (MRG) localizes target regions across an image set while producing an explicit cross-image reasoning trace. Given an image set $V { = } \{ I _ { 1 } , \ldots , I _ { m } \}$ , a prompt � specifying intent and output format, and an optional reference �, the model first generates a reasoning trace � and then outputs � grounded predictions $\{ ( b _ { k } , i _ { k } ) \} _ { k = 1 } ^ { n } ,$ where each bounding box $b _ { k }$ is indexed by an image identifier �<sub>�</sub>. Formally,

$$
O = \left\{ S , \left\{ ( b _ { k } , i _ { k } ) \right\} _ { k = 1 } ^ { n } \right\} = \left\{ { \cal M } _ { \mathrm { s g } } ( V , P ) , \right.\tag{1}
$$

Spontaneous grounding (sg) performs localization without an explicit reference, relying solely on � to infer and search the target across the image set. Referential grounding (rg) additionally consumes � (a textual mention or visual exemplar), demanding robust cross-image correspondence and precise localization. All sub-tasks formulation, composition and properties detailed in Appendix A.

## 3.2 Data Curation and MRG-Specific CoT Activation

For Supervised Fine-Tuning (SFT), we contribute a high-quality data curation pipeline that activates fine-grained perception ca pabilities in multi-image scenarios. As illustrated in Fig. 3, we systematically reconstruct MGrounding-630K [18], a multi-image grounding corpus spanning six in-domain sub-tasks, into a refined training set adapted to Qwen2.5-VL [2], through a pipeline that combines rule-based filtering with model-assisted rewriting. Rulebased filtering operates over question types, image quantity and resolution, and dialogue turns to remove homogeneous, erroneous, over-resolution, and redundant-dialogue samples, while reformulat ing special tokens and JSON structures to match the target output format. For content that cannot be automatically rewritten, we employ Qwen2.5-VL-72B [43] to regenerate label content from joint image-text context, introduce image-id key-value pairs for target image indices, and resample to maintain category balance across sub-tasks. The pipeline distills MGrounding-630K into 480K high quality samples, from which 320K are drawn for stage-one SFT to establish a solid grounding-activation foundation before reasoning enhancement.

The next stage aims to activate MRG-specific reasoning by dis tilling task-adaptive CoT demonstrations that elicit cross-image evidence-seeking before localization, so that multi-image evidence guides reasoning and reasoning, in turn, sharpens grounding precision. We categorize the remaining 160K instances into four MRGspecific task types, visual comparative, spatial perception, temporal perception, and visual semantic/logical association, each of which corresponds to a distinct cross-image cue pattern that MRG hinges on and thus admits its own task-adaptive prompts (templates in Appendix B), and prompt Qwen2.5-VL-72B to progressively analyze cross-image visual and textual evidence. Task-oriented operators (comparison, searching, observation, tracking, association) are injected to encourage step-wise evidence-seeking, while multi-image multi-object reasoning grounding is folded in from the beginning, enabling the model to reason once and then localize multiple targets across images in a single inference pass. The thinking process is constrained against revealing final answers or ground-truth bboxes, ensuring genuinely explanatory rationales rather than answer-leaking traces that RL would later exploit as shortcuts.

Two-step post-processing then ensures rationale rationality and utility: rule-based filtering removes pseudo-reasoning samples whose conclusions appear up-front, and IoU-improvement validation retains samples meeting one of three criteria, (i) correct direct prediction with ≥10% IoU gain after CoT, (ii) incorrect direct prediction corrected by CoT, or (iii) incorrect direct prediction with ≥20% IoU gain after CoT. Together, the pseudo-reasoning filter guards against answer-leaking rationales while the IoU-improvement criteria retain only CoT traces that demonstrably improve grounding, so downstream RL always inherits a cold-start prior that is both reasoning-faithful and grounding-efective. The pipeline yields 25K MRG cold-start samples, each formatted with a <think></think> reasoning trace followed by a <answer></answer> JSON block, forming the CoT-SFT initialization from which BiA-DAPO’s RL post-training begins.

## 3.3 BiA-DAPO for Reasoning Enhancement

In preliminary RL experiments on MRG, we observe two recurring training instabilities. First, many rollout groups exhibit near-zero reward variance and thus contribute vanishing group-relative advantage signals, efectively wasting the compute spent generating them. This sparsity drifts as the policy improves, since MRG’s coarse-tofine ladder “CoT→image-id→bbox” undergoes layer-wise saturation under AND-aggregated rewards, with each successive layer entering saturation as the previous one is mastered. Second, rollout groups with widely diferent mean rewards are mixed in the same update, causing the policy to oscillate between already-mastered and overly dificult samples. These two issues correspond, respectively, to drifting advantage-signal sparsity and competence-dificulty misalignment, two failure modes that a single group-level statistic cannot disentangle from one another.

![](images/ee0e6c2893e17fc742ac8265124cec11b885b1e94a25455e2c46d3a86410afbf.jpg)  
Figure 4: Overview of the proposed Bi-Axial DAPO (BiA-DAPO) for MRG. Upper: GIA scores each group along the intra-group signal axis to relieve drifting advantage-signal sparsity. Lower: CRS schedules updates over the GIA-retained groups along the inter-group competence axis to remedy competence-dificulty misalignment. Both operate on a shared candidate pool for stable group-level calculations.

Motivated by these observations, we propose BiA-DAPO, which uses two simple group-level statistics to stabilize RL training: Group-Informativeness Assessment (GIA) leverages within-group reward variance to retain rollout groups whose gradients are informative and non-vanishing, while Cascaded-Reward Stratification (CRS) uses group-level mean reward to schedule the retained groups from competence-mature to competence-developing bands, so each gradient step is delivered where the current policy can benefit from it most. The two mechanisms act on the same rollout groups but on distinct, complementary group-level statistics, jointly forming an advantage-grounded update stream that neither mechanism can produce on its own.

Bi-axial decomposition of advantage. The DAPO group-relative advantage of rollout � takes the form $\hat { A } _ { i } = ( \mathrm { a c c } _ { i } - \bar { r } _ { g } ) / ( \sigma _ { g } + \varepsilon )$ where $\bar { r } _ { g }$ and $\sigma _ { g }$ are the within-group mean and standard deviation of accuracy. The centring term encodes a group’s competence position along the inter-group competence axis (Axis-2), while the denominator encodes its informativeness along the intra-group sig nal axis (Axis-1). The two terms are statistically orthogonal and carry independent information, a group thus yields a non-trivial policy gradient only when well-positioned on both axes simultaneously. The two pathologies thus reside one per axis (sparsity on the signal axis, misalignment on the competence axis), and BiA-DAPO addresses each through a dedicated mechanism, all grounded on a shared candidate pool for stable group-level statistics.

Group-Informativeness Assessment (GIA). GIA scores each group along the signal axis by its within-group reward variance $\sigma _ { g } ^ { 2 } ,$ which reflects the magnitude of the advantage signal: a group whose rollouts receive near-identical rewards yields $\sigma _ { g } { \approx } 0$ and contributes negligible gradient information. We apply a variance-driven sampling gate whose informativeness threshold linearly decays as training progresses: early in training the gate admits only the highest-variance groups to concentrate updates on the most informative samples, while later in training it progressively accepts lower-variance groups as the policy stabilizes. As shown in Fig. 4 (upper), color-coded nodes denote groups assessed against the current threshold. This adaptive gating replaces DAPO’s all-or-nothing rejection and directly relieves advantage-signal sparsity by selectively excluding uninformative samples. The empirical training dynamics of this variance-driven gate, together with its efect on the group-level $\sigma _ { g }$ distribution and the $R _ { \mathrm { I o U } }$ outcome composition across training phases, are analyzed in Appendix D.

Cascaded-Reward Stratification (CRS). CRS operates along the competence axis by stratifying the GIA-retained groups into ordered competence bands on the between-group mean reward $\bar { r } _ { g } ,$ from high (competence-mature) to low (competence-developing). The policy is then updated band by band along this ordering, as shown in Fig. 4 (lower), forming a reward-driven training schedule that naturally transitions from easy to hard samples so that each gradient step consumes groups whose dificulty matches the pol icy’s evolving capability. The target competence band thus emerges dynamically from reward statistics without external dificulty labels, directly remedying competence-dificulty misalignment over the grouped rollouts. Empirical evidence for this cascade, in particular the diagonal migration of retained groups on the joint $( \bar { r } _ { g } , \sigma _ { g } )$ plane from low-competence high-spread toward high-competence moderate-spread regions, is further presented in Appendix D.

Stable group-level statistics. Reliable GIA and CRS decisions require stable axis calculations, since both $\sigma _ { g }$ and $\bar { r } _ { g }$ become noisy when computed from too few groups. Therefore, we repurpose DAPO’s standard “generation batch” as our candidate pool for each update step, whose existing size already sufices for group-level statistics to remain stable under bootstrap resampling, without any further enlargement on our part. The integration of GIA and CRS over this candidate pool constitutes BiA-DAPO, which jointly relieves drifting advantage-signal sparsity and remedies competencedificulty misalignment in a principled, advantage-grounded manner, without introducing any auxiliary loss, external dificulty labels, or generation-batch enlargement beyond the two group-level statistics themselves.

## 3.4 Reward Modeling and Training Objective

Following R1-like approaches, we employ a deterministic, coarseto-fine reward tailored to MRG: a format reward enforces the reasoning-to-answer protocol, while the accuracy reward decomposes into image-id and IoU components that first validate source image(s) and then assess box localization. We refer to $R _ { \mathrm { a c c } } { = } R _ { \mathrm { i d } } { + } R _ { \mathrm { I o U } }$ as the post-format accuracy (PFA), which empirically dominates the RL-active signal after CoT-SFT, also motivating our bi-axial treatment.

Format Reward. Outputs are checked for <think>/<answer> tags and the JSON block, with 0.25 per correctly-paired tag (else 0).

Image-id Reward. As the coarse-grained accuracy component, we first verify image-level correctness before proceeding to localization evaluation. For multi-image scenarios, Hungarian matching aligns predicted with ground-truth JSON blocks by image-id, while singleobject cases use direct matching. The reward is $R _ { \mathrm { i d } } = 1$ if and only if image-id matches, otherwise 0, in which case the subsequent IoU evaluation is skipped.

IoU Reward. As the fine-grained component, we evaluate targetlevel bounding-box precision only after a successful image-id match, using a dual-threshold mechanism (high $\tau _ { h } ,$ low $\tau _ { l } )$ that encour ages high-precision localization while preventing reward hacking through degenerate bounding boxes. For multi-image multi-object inputs, each target undergoes one $R _ { \mathrm { i d } }$ and one $R _ { \mathrm { I o U } }$ computation, with per-target rewards summed sequentially:

$$
R _ { \mathrm { t o t a l } } = R _ { \mathrm { f o r m a t } } + R _ { \mathrm { a c c } } , R _ { \mathrm { a c c } } = R _ { \mathrm { i d } } + R _ { \mathrm { I o U } }\tag{2}
$$

$$
R _ { \mathrm { I o U } } = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { I o U } ( B , \tilde { B } ) \geq \tau _ { h } } \\ { \mathrm { I o U } ( B , \tilde { B } ) , } & { \tau _ { l } \leq \mathrm { I o U } \leq \tau _ { h } } \\ { 0 , } & { \mathrm { I o U } ( B , \tilde { B } ) \leq \tau _ { l } } \end{array} \right.\tag{3}
$$

� and $\tilde { B }$ are predicted and ground-truth boxes; $\tau _ { h } { = } 0 . 9 , \tau _ { l } { = } 0 . 1$ , with linear interpolation between thresholds.

Training Objective. For the DAPO training objective, let question � paired with answer $^ { a , }$ and $o _ { 1 } , . . . , o _ { G }$ denote a group of sampled outputs from the old policy model $\pi _ { \theta _ { \mathrm { o l d } } }$ . The optimization objective of policy model $\pi _ { \theta }$ is formally defined as:

$$
\mathcal { T } _ { \mathrm { { D A P O } } } ( \theta ) = \mathbb { E } _ { ( q , a ) \sim \mathcal { D } , \{ o _ { i } \} _ { i = 1 } ^ { G } \sim \pi _ { \theta _ { 0 } \mid \mathrm { d } } } ( \cdot | q ) \left[ \frac { 1 } { \sum _ { i = 1 } ^ { G } | o _ { i } | } \sum _ { i = 1 } ^ { G } \sum _ { t = 1 } ^ { | o _ { i } | } \right.
$$

$$
\operatorname* { m i n } \Big ( r _ { i , t } ( \theta ) \hat { A } _ { i , t } , \mathrm { c l i p } \Big ( r _ { i , t } ( \theta ) , 1 - \varepsilon _ { l o w } , 1 + \varepsilon _ { h i g h } \Big ) \hat { A } _ { i , t } \Big ) \Bigg ]\tag{5}
$$

where $\varepsilon _ { \mathrm { l o w } }$ and $\varepsilon _ { \mathrm { h i g h } }$ denote the lower and upper clipping parameters, respectively. The clipping function bounds reward ratios to $\big [ 1 - \varepsilon _ { \mathrm { l o w } } , 1 + \varepsilon _ { \mathrm { h i g h } } \big ]$ , preventing excessive policy updates. The constraint $0 < | \{ o _ { i } \mid \mathrm { i } s .$ \_equivalen $\langle ( a , o _ { i } ) \} | < G$ ensures that among � generated outputs, at least one but not all sequences are equivalent to target action $^ { a , }$ maintaining diversity while enabling policy learning from both positive and negative examples. The DAPO training formulation details are in Appendix C.

The term $r _ { i , t }$ represents the probability ratio at timestep � for the �-th output sequence under current policy $\pi _ { \theta } ,$ , while $\hat { A } _ { i , t }$ denotes the advantage computed based on group rewards $\{ R _ { 1 } , \ldots , R _ { G } \}$

$$
r _ { i , t } ( \theta ) = \frac { \pi _ { \theta } ( o _ { i , t } \mid q , o _ { i , < t } ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( o _ { i , t } \mid q , o _ { i , < t } ) } , \hat { A } _ { i , t } = \frac { R _ { i } - \mathrm { m e a n } ( \{ R _ { i } \} _ { i = 1 } ^ { G } ) } { \mathrm { s t d } ( \{ R _ { i } \} _ { i = 1 } ^ { G } ) }\tag{6}
$$

## 4 Experiments

## 4.1 Implementation Details

Datasets. Stage-1 SFT uses 320K restructured MGrounding-630K samples to activate multi-image grounding capabilities. Stage-2 distills 32K MRG instances with CoT annotations, of which 25K coldstart samples couple reasoning with grounding during CoT-SFT and the remaining 7K are held out as a clean pool for BiA-DAPO’s rollout groups during RL post-training. For evaluation, we assess in-domain performance on 10 MIG-Bench sub-tasks [18] and, following UniVG-R1, further evaluate zero-shot transfer on both single-image reasoning grounding (LISA-Grounding [16], LLMSeg-Grounding [41]) and multi-frame reasoning grounding (ReVOS-Grounding [49], ReasonVOS-Grounding [4]) benchmarks. To probe general multiimage comprehension, we additionally test on multi-image understanding suites BLINK [10], MuirBench [39], MMIU [29], and MIBench [23], and report standard single-image referring comprehension on $\mathrm { R e f C O C O / + / g }$ so that gains are traced across in-domain MRG, cross-modal reasoning grounding transfer, and conventional grounding.

Training Details. MG-Thinker is built on Qwen2.5-VL-7B and trained on 8×A100 (80GB) with fixed random seeds for reproducibility. SFT uses learning rate 3�−6 and batch size 48, while CoT-SFT and RL both use 1�−6 with batch sizes 48 and 16, respectively; all stages adopt AdamW under a linear-warmup with cosine-decay schedule. During RL post-training, we sample 16 responses per query within a 1024-token cap via vLLM at temperature $^ { 1 . 0 , }$ and score each rollout by the deterministic reward described in Sec. 3.4. Full hyperparameters, including GIA/CRS phase schedule and clipping coeficients, are reported in Appendix E.

Evaluation Metrics. We adopt the standard Acc@0.5 metric following the REC protocol (IoU ≥ 0.5), with all baselines evaluated from their oficially released checkpoints under unified inference settings for fair comparison. For zero-shot reasoning grounding transfer, we keep the same Acc@0.5 protocol without any target-domain finetuning, so that comparisons reflect intrinsic transferability rather than task-specific adaptation, whereas multi-image understanding benchmarks retain their respective oficial protocols.

## 4.2 Main Experimental Results

Analysis of Reasoning Paradigms. We compare three reasoning paradigms for MRG: (i) direct end-to-end prediction without explicit reasoning, (ii) prompt-induced CoT, and (iii) MG-Thinker at cold-start vs. fully trained stages. Across representative sub-tasks, direct prediction often fails to establish reliable cross-image correspondences, while CoT prompting may sufer from reasoning drift that misleads the eventual localization. MG-Thinker instead shows progressively improved reasoning–grounding consistency: the coldstart model can localize correctly but produces unstable, sometimes drifting rationales, whereas the fully trained model consistently follows a reliable “anchor-reason-ground” pattern that yields faithful thinking and accurate predictions. Detailed case studies and qualitative visualizations are provided in Appendix G.

Performance on MIG-Bench. Tab. 1 compares MG-Thinker with state-of-the-art MLLMs (Qwen2/2.5/3-VL/3.5 [1, 31], Mantis, LLaVA-OV, MiniCPM-V-2.6, mPLUG-Owl3, InternVL2/3) and two specialized baselines: Migician, the first end-to-end multi-image grounding paradigm that bridges multi-image understanding with fine-grained single-image perception, and UniVG-R1, a recent GRPO-style attempt that couples multi-image localization with grounding-specific loss augmentations. To further isolate our bi-axial design at the algorithm level, we faithfully re-implement two single-axis baselines on a shared GRPO backbone under our CoT-SFT initialization and 25K MRG data: AdaRFT<sup>†</sup> (ofline dificulty curriculum, competence axis only) and UniVG-R1<sup>†</sup> (mIoU-weighting, signal axis only). UniVG-R1<sup>†</sup> already surpasses the original UniVG-R1 (74.22 vs. 72.64), validating the efectiveness of our curated data. Nonetheless, neither single-axis variant catches MG-Thinker, which sets a new state of the art on MIG-Bench on average (76.71 vs. 74.22 for the strongest baseline), beating Migician by 13.2 points and UniVG-R1 by 4.1 points and confirming that only the bi-axial treatment of both signal and competence axes yields the full gain. Despite using only 7B parameters, MG-Thinker outperforms 70B-scale models such as Qwen2.5-VL-72B by 30.5 points on average, highlighting a strong performance/eficiency trade-of. Detailed qualitative visual comparisons and case-study analysis are deferred to Appendix I. Efectiveness of reinforcement post-training. Fig. 5 compares BiA-DAPO’s reward dynamics with GRPO/DAPO under identical training settings: BiA-DAPO sustains a smooth upward trajectory with markedly smaller step-to-step fluctuation, whereas

Table 1: Performance comparison on MIG-Bench. OT, MV, GG and Co-Re respectively mean object tracking, multi-view, grouped and correspondence grounding. <sup>†</sup> denotes our fair re-implementation of existing RL baselines.
<table><tr><td rowspan="3">Models</td><td colspan="3">Spontaneous Grounding</td><td colspan="7">Referential Grounding</td><td rowspan="3">AVG</td></tr><tr><td colspan="2">Difference</td><td>Similarity</td><td colspan="3">Visual Reference</td><td></td><td>Textual</td><td colspan="2">Visual+Textual</td></tr><tr><td>Static</td><td>Robust</td><td>Common</td><td>OT</td><td>MV</td><td>Region</td><td>Refer</td><td>GG</td><td>Reason</td><td>Co-Re</td></tr><tr><td colspan="10">70B-Scale MLLMs</td></tr><tr><td>LLaVA-OV-72B [17]</td><td>13.26</td><td>5.34</td><td>26.84</td><td>12.91</td><td>7.64</td><td>2.14</td><td>17.83</td><td>21.60</td><td>11.88</td><td>8.55</td><td>13.65</td></tr><tr><td>InternVL2-76B [7]</td><td>15.91</td><td>10.64</td><td>36.40</td><td>30.73</td><td>20.83</td><td>5.74</td><td>46.46</td><td>41.28</td><td>32.67</td><td>26.50</td><td>26.72</td></tr><tr><td>InternVL3-78B [60]</td><td>10.04</td><td>9.57</td><td>24.12</td><td>27.08</td><td>14.58</td><td>10.44</td><td>50.51</td><td>38.08</td><td>45.54</td><td>17.09</td><td>24.71</td></tr><tr><td>Qwen2-VL-72B [43]</td><td>46.12</td><td>46.81</td><td>64.46</td><td>26.73</td><td>22.57</td><td>18.62</td><td>33.33</td><td>62.53</td><td>50.50</td><td>17.09</td><td>38.88</td></tr><tr><td>Qwen2.5-VL-72B [2]</td><td>43.75</td><td>46.81</td><td>69.98</td><td>34.32</td><td>29.17</td><td>8.31</td><td>62.63</td><td>59.92</td><td>66.34</td><td>41.03</td><td>46.23</td></tr><tr><td colspan="10">7B-Scale MLLMs</td></tr><tr><td>Mantis [15]</td><td>1.52</td><td>0.00</td><td>3.31 3.43</td><td>12.18</td><td>2.08</td><td>1.00</td><td>1.01</td><td>10.02</td><td>0.00</td><td>0.85</td><td>3.20</td></tr><tr><td>LLaVA-OV-7B [17] MiniCPM-V-2.6 [50]</td><td>6.06 14.58</td><td>3.19 2.13</td><td>14.34</td><td>0.18 9.82</td><td>1.04 6.25</td><td>1.08 1.75</td><td>9.09</td><td>15.43 10.02</td><td>6.93 2.97</td><td>0.85 2.56</td><td>4.73 7.55</td></tr><tr><td>mPLUG-Owl3 [52]</td><td>18.56</td><td>6.38</td><td>34.93</td><td>8.55</td><td>7.64</td><td>2.41</td><td>11.11 7.07</td><td>22.85</td><td>9.09</td><td>5.98</td><td>12.35</td></tr><tr><td>InternVL2-8B [7]</td><td>6.92</td><td>7.45</td><td>25.49</td><td>20.73</td><td>9.72</td><td>3.49</td><td>28.28</td><td>30.26</td><td>17.82</td><td>9.40</td><td>15.96</td></tr><tr><td>InternVL3-8B [60]</td><td>23.67</td><td>14.89</td><td>47.99</td><td>14.84</td><td>6.94</td><td>12.13</td><td>7.07</td><td>34.87</td><td>16.83</td><td>2.56</td><td>18.18</td></tr><tr><td>Qwen2-VL-7B [43]</td><td>27.84</td><td>38.30</td><td>19.36</td><td>20.73</td><td>11.81</td><td>25.95</td><td>23.23</td><td>58.52</td><td>48.51</td><td>11.97</td><td>28.62</td></tr><tr><td>Qwen2.5-VL-7B [2]</td><td>29.92</td><td>22.45</td><td>16.36</td><td>6.94</td><td>3.66</td><td>15.05</td><td>35.67</td><td></td><td></td><td>3.42</td><td>20.10</td></tr><tr><td>Qwen3-VL-8B [1]</td><td>29.36</td><td>24.47</td><td>58.83</td><td></td><td></td><td></td><td></td><td>63.51</td><td>4.03</td><td></td><td></td></tr><tr><td>Qwen3.5-9B [37]]</td><td>57.01</td><td>18.09</td><td>63.44</td><td>28.00</td><td>17.36</td><td>23.44</td><td>33.33</td><td>70.93</td><td>32.99</td><td>10.25</td><td>32.90</td></tr><tr><td></td><td></td><td></td><td></td><td>24.73</td><td>34.72</td><td>27.31</td><td>80.31</td><td>56.49</td><td>29.90</td><td>19.66</td><td>41.17</td></tr><tr><td colspan="10">Training-Specific MLLMs</td></tr><tr><td>Migician [18]</td><td>70.64</td><td>45.74</td><td>72.76</td><td>67.82</td><td>60.07</td><td>72.57</td><td>75.76</td><td>84.12</td><td>52.58</td><td>33.33</td><td>63.54</td></tr><tr><td>UniVG-R1 [3]</td><td>71.97</td><td>58.51</td><td>93.13</td><td>76.36</td><td>66.32</td><td>81.71</td><td>82.83</td><td>88.04</td><td>62.89</td><td>44.44</td><td>72.64</td></tr><tr><td>UniVG-R1†</td><td>76.70</td><td>61.70</td><td>87.61</td><td>82.18</td><td>64.93</td><td>86.28</td><td>87.88</td><td>88.25</td><td>69.07</td><td>37.61</td><td>74.22</td></tr><tr><td>AdaRFT†</td><td>74.05</td><td>57.45</td><td>88.22</td><td>81.27</td><td>57.99</td><td>84.95</td><td>83.84</td><td>87.63</td><td>67.01</td><td>45.30</td><td>72.77</td></tr><tr><td>MG-Thinker</td><td>78.98</td><td>62.77</td><td>93.01</td><td>82.55</td><td>66.67</td><td>88.78</td><td>86.87</td><td>88.25</td><td>72.16</td><td>47.01</td><td>76.71</td></tr></table>

Table 2: Comparison ofdiferent outputting formats. ‘Polling’ refers to the multi-image polling approach, while ‘All’ means outputting multi-object in one shot.
<table><tr><td>Models</td><td>Output Format</td><td>Common</td><td>MV</td><td>OT</td><td>Region</td><td>AVG</td></tr><tr><td>Random Guess</td><td>/</td><td>26.47</td><td>1.04</td><td>2.13</td><td>0.00</td><td>7.41</td></tr><tr><td rowspan="2">Qwen2-VL-7B</td><td>Poll</td><td>19.36</td><td>11.81</td><td>20.73</td><td>25.95</td><td>19.46</td></tr><tr><td>All All+CoT</td><td>19.36 45.71</td><td>6.60 9.38</td><td>13.09 17.55</td><td>11.80 15.54</td><td>12.71 22.05</td></tr><tr><td rowspan="2">Qwen2.5-VL-7B</td><td>Poll</td><td>16.36</td><td>3.66</td><td>6.94</td><td>15.05</td><td>10.50</td></tr><tr><td>All</td><td>27.47</td><td>3.90</td><td>15.31</td><td>17.29</td><td>15.99</td></tr><tr><td rowspan="2">Migician</td><td>All+CoT</td><td>40.58</td><td>4.98</td><td>14.60</td><td>19.12</td><td>19.82</td></tr><tr><td>Poll</td><td>72.76</td><td>60.07</td><td>67.82</td><td>72.57</td><td>68.31</td></tr><tr><td rowspan="2"></td><td>All All+CoT</td><td>72.43 75.56</td><td>43.06 41.67</td><td>58.55 63.82</td><td>34.91 41.81</td><td>52.24 55.72</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">UniVG-R1</td><td>Poll All</td><td>93.13</td><td>66.32 30.90</td><td>76.36</td><td>81.71</td><td>79.38</td></tr><tr><td></td><td>12.99</td><td></td><td>26.55</td><td>35.08</td><td>26.38</td></tr><tr><td rowspan="2">MG-Thinker</td><td>Poll</td><td>90.18</td><td></td><td>66.67 82.55</td><td>88.11</td><td>81.88</td></tr><tr><td>All</td><td>93.01</td><td>65.22</td><td>81.81</td><td>88.78</td><td>82.21</td></tr></table>

Table 3: Zero-shot performance on other reasoning grounding benchmarks.
<table><tr><td rowspan="2">Models</td><td colspan="3">Single Image</td><td colspan="2">Multi Images</td><td rowspan="2">AVG</td></tr><tr><td>|LISA-val LISA-test LLMSeg</td><td></td><td></td><td>ReasVOS ReVOS</td><td></td></tr><tr><td>Qwen2-VL-7B</td><td>52.00</td><td>49.17</td><td>35.53</td><td>9.83</td><td>23.55</td><td>34.02</td></tr><tr><td>Qwen2.5-VL-7B</td><td>54.06</td><td>50.75</td><td>32.48</td><td>11.14</td><td>26.81</td><td>35.04</td></tr><tr><td>Migician</td><td>36.00</td><td>32.09</td><td>34.68</td><td>33.41</td><td>39.70</td><td>35.18</td></tr><tr><td>UniVG-R1</td><td>64.00</td><td>59.69</td><td>50.60</td><td>58.73</td><td>60.03</td><td>58.61</td></tr><tr><td>MG-Thinker</td><td>64.29</td><td>60.10</td><td>49.94</td><td>59.47</td><td>61.32</td><td>59.02</td></tr></table>

![](images/ae0823d602b525b6737ed626916870688af30b6a6f221426fc4ab2dd121dac07.jpg)  
Figure 5: PFA reward dynamics under identical training. BiA-DAPO sustains a smooth upward trajectory, whereas GRPO/DAPO oscillate non-monotonically due to coupled sparsity and misalignment.

Table 4: Cross-task heterogeneity analysis on MIG-Bench. Tasks are grouped into V-Het, R-Het, and L-Het.
<table><tr><td>Task group</td><td></td><td>BiA-DAPO UniVG-R1†</td><td>AdaRFT†</td><td>|∆Uni†</td><td>∆Ada†</td></tr><tr><td>V-Het (mean-drift)</td><td>76.14</td><td>75.69</td><td>71.73</td><td>+0.45</td><td>+4.41</td></tr><tr><td>R-Het (variance-drift)</td><td>67.24</td><td>62.95</td><td>64.53</td><td>+4.29</td><td>+2.71</td></tr><tr><td>L-Het (no dominant)</td><td>85.76</td><td>83.53</td><td>82.41</td><td>+2.23</td><td>+3.35</td></tr></table>

Table 5: Performance comparison on multi-image understanding benchmarks.
<table><tr><td>Models</td><td>MuirBench BLINK Val MIBench MMIU</td><td></td><td></td><td></td><td>AVG</td></tr><tr><td>LLaVA-1.5</td><td>23.46</td><td>37.13</td><td>26.83</td><td>19.20</td><td>26.66</td></tr><tr><td>CogVLM</td><td>20.85</td><td>41.54</td><td></td><td>23.57</td><td>28.65</td></tr><tr><td>Idefics2-8B</td><td>26.08</td><td></td><td>46.39</td><td>27.80</td><td>33.42</td></tr><tr><td>mPLUG-Owl3</td><td>39.67</td><td>50.30</td><td>56.66</td><td>21.72</td><td>42.09</td></tr><tr><td>InternVL2-8B</td><td>48.70</td><td>50.57</td><td>52.91</td><td>42.00</td><td>48.55</td></tr><tr><td>Mantis</td><td>44.50</td><td>49.05</td><td>45.09</td><td>45.60</td><td>46.06</td></tr><tr><td>LLaVA-OV-7B</td><td>41.80</td><td>48.20</td><td>71.29</td><td>44.46</td><td>51.44</td></tr><tr><td>MiniCPM-V 2.6</td><td>42.65</td><td>51.45</td><td>71.09</td><td>50.19</td><td>53.85</td></tr><tr><td>Qwen2-VL-7B</td><td>39.88</td><td>52.35</td><td>68.06</td><td>54.36</td><td>53.66</td></tr><tr><td>Migician</td><td>57.81</td><td>51.53</td><td>71.42</td><td>54.65</td><td>58.85</td></tr><tr><td>UniVG-R1</td><td>44.61</td><td>51.77</td><td>66.93</td><td>53.13</td><td>54.11</td></tr><tr><td>MG-Thinker</td><td>61.77</td><td>54.13</td><td>71.88</td><td>53.93</td><td>60.43</td></tr></table>

GRPO/DAPO oscillate without consistent improvement and even intermittently regress, reflecting their inability to disentangle the two MRG pathologies. Fig. 6 reports the per-step distribution of group-level $\sigma _ { g }$ (binned at 0, (0, 0.125], (0.125, 0.25], (0.25, 1.0]), corresponding to advantage-vanished, low-, medium-, and highinformativeness groups: BiA-DAPO concentrates on the medium/high intervals (GIA retains advantage-bearing groups), while DAPO skews toward the low end and GRPO further over-retains $\sigma _ { g } = 0$ groups whose gradient is mathematically zero, the drifting sparsity that drives GRPO/DAPO’s reward oscillation. Together, the smooth trajectory in Fig. 5 and the healthier $\sigma _ { g }$ mass in Fig. 6 give direct evidence that GIA and CRS act as an axis-decoupled treatment of the two pathologies rather than merely a single stabilization knob;

![](images/3d42a5c422a1031f3ddf7210f552a51a806448506af1561045c3953ef060fe8a.jpg)  
Figure 6: Per-step distribution of group-level $\sigma _ { g } .$ . BiA-DAPO concentrates on medium/high informativeness intervals (GIA gating), while GRPO/DAPO over-retain low- ${ \cdot } \sigma _ { g }$ groups whose advantage is mathematically zero.

two complementary diagnostics in Appendix D further substantiate this joint GIA-CRS mechanism.

Cross-task heterogeneity analysis. We partition all MRG subtasks into three groups by intrinsic heterogeneity: Visual-, Reasoning-, and Low-Heterogeneity. V-Het tasks exhibit mean-reward drift (corresponding to the competence axis), R-Het tasks exhibit variance drift (corresponding to the signal axis), while L-Het tasks show neither prominently (group-wise task composition in Appendix A). Tab. 4 then reveals complementary blind spots: UniVG-R1<sup>†</sup> matches BiA-DAPO on V-Het (+0.45) but lags on R-Het (+4.29), whereas AdaRFT<sup>†</sup> stays close on R-Het (+2.71) but loses on V-Het (+4.41). Only BiA-DAPO leads on both axes simultaneously, confirming that GIA and CRS act on genuinely complementary pathologies.

Zero-Shot on Reasoning Grounding Benchmarks. Tab. 3 reports zero-shot results on single-image (LISA/LLMSeg-Grounding) and video (ReVOS/ReasonVOS-Grounding) reasoning grounding benchmarks. MG-Thinker substantially surpasses Migician and matches UniVG-R1 (59.02% avg.), the slight LLMSeg-Grounding drop reflects UniVG-R1’s use of single-image reasoning data during RL, which we omit. These results show transferability without taskspecific tuning. Notably, the multi-frame gains (ReVOS/ReasonVOS) emerge purely from MG-Thinker’s cross-image reasoning priors without any video-specific supervision, showing that multi-image reasoning transfers naturally to the temporal setting.

Multi-image Understanding. Tab. 5 shows MG-Thinker delivers strong performance across diverse multi-image understanding suites. While UniVG-R1’s grounding gains transfer poorly (falling behind Migician on several benchmarks), MG-Thinker yields superior overall results and notably leads on MuirBench counting/visual grounding subtasks, indicating BiA-DAPO benefits generalized understanding rather than task-specific gains. This further suggests that MG-Thinker’s grounding-anchored CoT strengthens the crossimage attention that also underpins general multi-image comprehension, avoiding over-specialization to the grounding objective. General single-image grounding. On standard RefCOCO/+/g (full table is in Appendix H), MG-Thinker remains strongly competitive on single-image referring expression grounding, indicating stable multi-modal generalization beyond reasoning-centric benchmarks. This further confirms that our MRG-oriented posttraining preserves the base-model grounding prior rather than overspecializing to multi-image tasks, a common over-specialization trap of task-adaptive RL. We attribute this stability to BiA-DAPO’s group-relative updates being anchored on rollout-level reward statistics rather than any modality-specific loss term, so the RL signal never explicitly suppresses the single-image grounding prior even when the training pool is dominated by multi-image samples.

Table 6: Ablation study of diferent training stages and RL post-training algorithm components.
<table><tr><td rowspan="3">Methods</td><td rowspan="3">No.</td><td colspan="3">Spontaneous Grounding</td><td colspan="6">Referential Grounding</td><td rowspan="3">AVG</td></tr><tr><td></td><td>Difference</td><td>Similarity</td><td colspan="3">Visual Reference</td><td>Textual</td><td colspan="2">Visual+Textual</td></tr><tr><td>Static</td><td>Robust</td><td>Common</td><td>OT MV</td><td>Region</td><td>Refer</td><td>GG</td><td>Reason</td><td>Co-Re</td></tr><tr><td colspan="10">Baseline MLLMs</td><td></td><td></td></tr><tr><td>Qwen2-VL-7B [43]</td><td>1</td><td>27.84</td><td>38.30</td><td>19.36 20.73</td><td>11.81</td><td>25.95</td><td>23.23</td><td>58.52</td><td>48.51</td><td>11.97</td><td>28.62</td></tr><tr><td>Qwen2.5-VL-7B [2]</td><td></td><td>29.92</td><td>22.45</td><td>16.36</td><td>6.94 3.66</td><td>15.05</td><td>35.67</td><td>63.51</td><td>4.03</td><td>3.42</td><td>20.10</td></tr><tr><td>Qwen3-VL-8B [1]</td><td>234</td><td>29.36</td><td>24.47</td><td>58.83</td><td>28.00 17.36</td><td>23.44</td><td>33.33</td><td>70.93</td><td>32.99</td><td>10.25</td><td>32.90</td></tr><tr><td>Qwen3.5-9B [37]]</td><td></td><td>57.01</td><td>18.09</td><td>63.44</td><td>24.73 34.72</td><td>27.31</td><td>80.31</td><td>56.49</td><td>29.90</td><td>19.66</td><td>41.17</td></tr><tr><td colspan="10">Specific Training Stages</td><td></td><td></td></tr><tr><td>SFT</td><td>5</td><td></td><td></td><td></td><td></td><td></td><td></td><td>82.83</td><td></td><td>57.73</td><td></td></tr><tr><td>CoT-SFT</td><td>6</td><td>65.49 72.18</td><td>52.13 60.64</td><td>86.19 85.72</td><td>75.68 74.89</td><td>61.81 63.54</td><td>77.89 77.47</td><td>83.85 81.82 85.90</td><td>69.07</td><td>35.90 41.03</td><td>67.95 71.23</td></tr><tr><td colspan="10">Specific Algorithm Components</td></tr><tr><td>GRPO</td><td>7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DAPO</td><td>8</td><td>70.27</td><td>60.83</td><td>87.50</td><td>80.00</td><td>65.26 82.49</td><td>82.44 84.23</td><td>87.63 87.42</td><td>72.65 73.20</td><td>44.30 43.22</td><td>73.34 72.98</td></tr><tr><td>w/o. GIA</td><td>9</td><td>72.55 78.79</td><td>60.97</td><td>85.15 90.80</td><td>75.27</td><td>60.76 67.01</td><td>87.03 86.70</td><td>85.86</td><td>89.69</td><td>76.29 41.03</td><td>75.57</td></tr><tr><td>w/o.CRS</td><td>10</td><td>77.46</td><td>60.64 58.51</td><td></td><td>78.91</td><td>62.15</td><td>88.04</td><td>86.53</td><td>85.86</td><td>75.26 40.17</td><td>74.84</td></tr><tr><td>MG-Thinker</td><td>11</td><td>78.98</td><td>62.77</td><td>90.92 93.01</td><td>83.45 82.55</td><td>66.67</td><td>88.78</td><td>86.87</td><td>88.25</td><td>72.16 47.01</td><td>76.71</td></tr></table>

## 4.3 Ablation Studies

We validate MG-Thinker from two perspectives: training stages and BiA-DAPO components.

Efect of Training Stages. Tab. 6 validates the three progressive stages, grounding activation, reasoning–grounding integration, and reasoning enhancement. Stage-1 SFT alone surpasses Qwen2/2.5- VL-7B across diverse sub-tasks including visual comparison, object tracking, and spatial perception, achieving over 2× averageaccuracy gain on MIG-Bench. Stage-2 CoT-SFT (cold-start) then couples grounding with explicit reasoning, yielding notable improvements on correspondence- and reasoning-intensive sub-tasks. Finally, BiA-DAPO delivers the optimum on top of CoT-SFT, together confirming the progressive realization of the proposed MRG paradigm from activate to enhance.

Efect of BiA-DAPO Components. We perform controlled ablations on the RL algorithm (Tab. 6). GRPO and DAPO baselines yield improvements but are upper-bounded by their single-statistic treatment (Sec. 3.3), which conflates advantage magnitude and competence position into one number. Removing either axis-specific mechanism causes complementary degradation: w/o. GIA (Axis-1 disabled) loses on tasks with wide intra-group spread (OT, Co-Re), where informativeness gating is critical for relieving drifting advantage-signal sparsity, since the losses trace back to $\sigma _ { g }$ ≈0 groups that unfiltered DAPO would otherwise admit; w/o. CRS (Axis-2 disabled) loses on tasks with wide between-group competence spread (Robust, MV, GG), where ordered $\bar { r } _ { g }$ traversal is needed to remedy competence-dificulty misalignment, since these are precisely the sub-tasks whose group means shift the most across training. Integrating both axes recovers the optimum, corroborating the cross-heterogeneity evidence in Tab. 4. Detailed sensitivity ablations on the GIA phase schedule and CRS band width are reported in Appendix F.

## 5 Conclusion

In this paper, we present MG-Thinker, an RL-based framework for multi-image reasoning grounding (MRG). We identify two MRGspecific challenges largely overlooked by prior work: an inherent coarse-to-fine hierarchical pattern from semantic reasoning to region-level localization, and a pronounced task-sample dificulty heterogeneity that induces drifting advantage-signal sparsity and competence-dificulty misalignment during post-training. To address them, we curate a 25K MRG-specific CoT dataset that instills the hierarchical pattern, and propose Bi-Axial DAPO (BiA-DAPO), which disentangles rollout advantages along an intra-group signal axis and an inter-group competence axis via Group-Informativeness Assessment and Cascaded-Reward Stratification over a shared candidate pool. Extensive experiments confirm state-of-the-art MRG performance with consistent generalization to broader multimodal benchmarks, and we hope this bi-axial view of group-relative RL offers a broadly useful design pattern for other hierarchical, dificultyheterogeneous multimodal reasoning tasks.

## References

[1] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. 2025. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631 (2025).

[2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. 2025. Qwen2. 5-vl technica report. arXiv preprint arXiv:2502.13923 (2025).

[3] Sule Bai, Mingxing Li, Yong Liu, Jing Tang, Haoji Zhang, Lei Sun, Xiangxiang Chu, and Yansong Tang. 2025. Univg-r1: Reasoning guided universal visual grounding with reinforcement learning. arXiv preprint arXiv:2505.14231 (2025).

[4] Zechen Bai, Tong He, Haiyang Mei, Pichao Wang, Ziteng Gao, Joya Chen, Zheng Zhang, and Mike Zheng Shou. 2024. One token to seg them all: Language instructed reasoning segmentation in videos. Advances in Neural Information Processing Systems 37 (2024), 6833–6859.

[5] Meng Cao, Haoze Zhao, Can Zhang, Xiaojun Chang, Ian Reid, and Xiaodan Liang. 2025. Ground-R1: Incentivizing Grounded Visual Reasoning via Reinforcement Learning. arXiv preprint arXiv:2505.20272 (2025).

[6] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. 2023. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv preprint arXiv:2306.15195 (2023).

[7] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. 2024. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 24185–24198.

[8] Yue Fan, Xuehai He, Diji Yang, Kaizhi Zheng, Ching-Chen Kuo, Yuting Zheng, Sravana Jyothi Narayanaraju, Xinze Guan, and Xin Eric Wang. 2025. GRIT: Teaching MLLMs to Think with Images. arXiv preprint arXiv:2505.15879 (2025).

[9] Stephanie Fu, Tyler Bonnen, Devin Guillory, and Trevor Darrell. 2025. Hidden in plain sight: VLMs overlook their visual representations. arXiv preprint arXiv:2506.08008 (2025).

[10] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. 2024. Blink: Multimodal large language models can see but not perceive. In European Conference on Computer Vision. Springer, 148–166.

[11] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948 (2025).

[12] Zonghao Guo, Ruyi Xu, Yuan Yao, Junbo Cui, Zanlin Ni, Chunjiang Ge, Tat-Seng Chua, Zhiyuan Liu, and Gao Huang. 2024. Llava-uhd: an lmm perceiving any aspect ratio and high-resolution images. In European Conference on Computer Vision. Springer, 390–406.

[13] Yushi Hu, Weijia Shi, Xingyu Fu, Dan Roth, Mari Ostendorf, Luke Zettlemoyer, Noah A Smith, and Ranjay Krishna. 2024. Visual sketchpad: Sketching as a visual chain of thought for multimodal language models. Advances in Neural Information Processing Systems 37 (2024), 139348–139379.

[14] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. 2024. Openai o1 system card. arXiv preprint arXiv:2412.16720 (2024).

[15] Dongfu Jiang, Xuan He, Huaye Zeng, Cong Wei, Max Ku, Qian Liu, and Wenhu Chen. 2024. Mantis: Interleaved multi-image instruction tuning. arXiv preprint arXiv:2405.01483 (2024).

[16] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. 2024. Lisa: Reasoning segmentation via large language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 9579– 9589.

[17] Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. 2024. Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895 (2024).

[18] You Li, Heyu Huang, Chi Chen, Kaiyu Huang, Chao Huang, Zonghao Guo, Zhiyuan Liu, Jinan Xu, Yuhua Li, Ruixuan Li, and Maosong Sun. 2025. Migician: Revealing the Magic of Free-Form Multi-Image Grounding in Multimodal Large Language Models. In Findings ofthe Association for Computational Linguistics: ACL 2025. 9845–9867. doi:10.18653/v1/2025.findings-acl.512

[19] Zejun Li, Ruipu Luo, Jiwen Zhang, Minghui Qiu, Xuanjing Huang, and Zhongyu Wei. 2024. Vocot: Unleashing visually grounded multi-step reasoning in large multi-modal models. arXiv preprint arXiv:2405.16919 (2024).

[20] Zhaowei Li, Qi Xu, Dong Zhang, Hang Song, Yiqing Cai, Qi Qi, Ran Zhou, Junting Pan, Zefeng Li, Van Tu Vu, et al. 2024. Groundinggpt: Language enhanced multi modal grounding model. arXiv preprint arXiv:2401.06071 (2024)

[21] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2024. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 26296–26306.

[22] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual Instruction Tuning. In Advances in Neural Information Processing Systems, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (Eds.), Vol. 36.

Curran Associates, Inc., 34892–34916. https://proceedings.neurips.cc/paper\_ files/paper/2023/file/6dcf277ea32ce3288914faf369fe6de0-Paper-Conference.pdf

[23] Haowei Liu, Xi Zhang, Haiyang Xu, Yaya Shi, Chaoya Jiang, Ming Yan, Ji Zhang, Fei Huang, Chunfeng Yuan, Bing Li, et al. 2024. Mibench: Evaluating multimodal large language models over multiple images. arXiv preprint arXiv:2407.15272 (2024).

[24] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. 2024. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European conference on computer vision. Springer, 38–55.

[25] Yuqi Liu, Tianyuan Qu, Zhisheng Zhong, Bohao Peng, Shu Liu, Bei Yu, and Jiaya Jia. 2025. VisionReasoner: Unified Visual Perception and Reasoning via Reinforcement Learning. arXiv preprint arXiv:2505.12081 (2025).

[26] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Hao Yang, et al. 2024. Deepseek-vl: towards realworld vision-language understanding. arXiv preprint arXiv:2403.05525 (2024).

[27] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. 2023. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255 (2023).

[28] Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L Yuille, and Kevin Murphy. 2016. Generation and comprehension of unambiguous object descriptions. In Proceedings ofthe IEEE conference on computer vision and pattern recognition. 11–20.

[29] Fanqing Meng, Jin Wang, Chuanhao Li, Quanfeng Lu, Hao Tian, Jiaqi Liao, Xizhou Zhu, Jifeng Dai, Yu Qiao, Ping Luo, et al. 2024. Mmiu: Multimodal multiimage understanding for evaluating large vision-language models. arXiv preprint arXiv:2408.02718 (2024).

[30] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. 2023. Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint arXiv:2306.14824 (2023).

[31] Qwen Team. 2026. Qwen3.5: Towards Native Multimodal Agents. https://qwen. ai/blog?id=qwen3.5

[32] Hanoona Rasheed, Muhammad Maaz, Sahal Shaji, Abdelrahman Shaker, Salman Khan, Hisham Cholakkal, Rao M Anwer, Eric Xing, Ming-Hsuan Yang, and Fahad S Khan. 2024. Glamm: Pixel grounding large multimodal model. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 13009–13018.

[33] Hao Shao, Shengju Qian, Han Xiao, Guanglu Song, Zhuofan Zong, Letian Wang, Yu Liu, and Hongsheng Li. 2024. Visual cot: Advancing multi-modal language models with a comprehensive dataset and benchmark for chain-of-thought reasoning. Advances in Neural Information Processing Systems 37 (2024), 8612– 8642.

[34] Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, et al. 2025. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615 (2025).

[35] Taiwei Shi, Yiyang Wu, Linxin Song, Tianyi Zhou, and Jieyu Zhao. 2025. Efi cient reinforcement finetuning via adaptive curriculum learning. arXiv preprint arXiv:2504.05520 (2025).

[36] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. 2025. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599 (2025).

[37] Qwen Team. 2026. Qwen3.5: Accelerating Productivity with Native Multimodal Agents. https://qwen.ai/blog?id=qwen3.5

[38] Peter Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Adithya Jairam Vedagiri IYER, Sai Charitha Akula, Shusheng Yang, Jihan Yang, Manoj Middepogu, Ziteng Wang, et al. 2024. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. Advances in Neural Information Processing Systems 37 (2024), 87310–87356.

[39] Fei Wang, Xingyu Fu, James Y Huang, Zekun Li, Qin Liu, Xiaogeng Liu, Mingyu Derek Ma, Nan Xu, Wenxuan Zhou, Kai Zhang, et al. 2025. Muirbench: A comprehensive benchmark for robust multi-image understanding. In International Conference on Learning Representations, Vol. 2025. 62624–62650.

[40] Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, and Wenhu Chen. 2025. Vl-rethinker: Incentivizing self-reflection of vision-language models with reinforcement learning. arXiv preprint arXiv:2504.08837 (2025).

[41] Junchi Wang and Lei Ke. 2024. Llm-seg: Bridging image segmentation and large language model reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 1765–1774.

[42] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. 2024. Measuring multimodal mathematical reasoning with math-vision dataset. Advances in Neural Information Processing Systems 37 (2024), 95095–95169.

[43] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. 2024. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191 (2024).

[44] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Song XiXuan, et al. 2024. Cogvlm: Visual expert for pretrained language models. Advances in Neural Information Processing Systems 37 (2024), 121475–121499.

[45] Jiannan Wu, Muyan Zhong, Sen Xing, Zeqiang Lai, Zhaoyang Liu, Zhe Chen, Wenhai Wang, Xizhou Zhu, Lewei Lu, Tong Lu, et al. 2024. Visionllm v2: An end-to-end generalist multimodal large language model for hundreds of visionlanguage tasks. Advances in Neural Information Processing Systems 37 (2024), 69925–69975.

[46] Size Wu, Sheng Jin, Wenwei Zhang, Lumin Xu, Wentao Liu, Wei Li, and Chen Change Loy. 2025. F-lmm: Grounding frozen large multimodal models. In Proceedings of the Computer Vision and Pattern Recognition Conference. 24710–24721.

[47] Wenyi Xiao, Leilei Gan, Weilong Dai, Wanggui He, Ziwei Huang, Haoyuan Li, Fangxun Shu, Zhelun Yu, Peng Zhang, Hao Jiang, et al. 2025. Fast-slow thinking for large vision-language model reasoning. arXiv preprint arXiv:2504.18458 (2025).

[48] Guowei Xu, Peng Jin, Ziang Wu, Hao Li, Yibing Song, Lichao Sun, and Li Yuan. 2024. Llava-cot: Let vision language models reason step-by-step. arXiv preprint arXiv:2411.10440 (2024).

[49] Cilin Yan, Haochen Wang, Shilin Yan, Xiaolong Jiang, Yao Hu, Guoliang Kang, Weidi Xie, and Efstratios Gavves. 2024. Visa: Reasoning video object segmentation via large language models. In European Conference on Computer Vision. Springer, 98–115.

[50] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. 2024. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800 (2024).

[51] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Guohai Xu, Chenliang Li, Junfeng Tian, Qi Qian, Ji Zhang, et al. 2023. Ureader: Universal ocr-free visually-situated language understanding with multimodal large language model. arXiv preprint arXiv:2310.05126 (2023).

[52] Jiabo Ye, Haiyang Xu, Haowei Liu, Anwen Hu, Ming Yan, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. 2024. mplug-owl3: Towards long imagesequence understanding in multi-modal large language models. arXiv preprint arXiv:2408.04840 (2024).

[53] Haoxuan You, Haotian Zhang, Zhe Gan, Xianzhi Du, Bowen Zhang, Zirui Wang, Liangliang Cao, Shih-Fu Chang, and Yinfei Yang. 2023. Ferret: Refer and ground anything anywhere at any granularity. arXiv preprint arXiv:2310.07704 (2023).

[54] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. 2016. Modeling context in referring expressions. In European conference on computer vision. Springer, 69–85.

[55] Yufei Zhan, Hongyin Zhao, Yousong Zhu, Shurong Zheng, Fan Yang, Ming Tang, and Jinqiao Wang. 2025. Understand, Think, and Answer: Advancing Visual Reasoning with Large Multimodal Models. arXivpreprintarXiv:2505.20753 (2025).

[56] Yufei Zhan, Yousong Zhu, Hongyin Zhao, Fan Yang, Ming Tang, and Jinqiao Wang. 2024. Grifon v2: Advancing multimodal perception with high-resolution scaling and visual-language co-referring. arXiv preprint arXiv:2403.09333 (2024).

[57] Yufei Zhan, Yousong Zhu, Shurong Zheng, Hongyin Zhao, Fan Yang, Ming Tang, and Jinqiao Wang. 2025. Vision-r1: Evolving human-free alignment in large vision-language models via vision-guided reinforcement learning. arXiv preprint arXiv:2503.18013 (2025).

[58] Guanghao Zhang, Tao Zhong, Yan Xia, Zhelun Yu, Haoyuan Li, Wanggui He, Fangxun Shu, Mushui Liu, Dong She, Yi Wang, et al. 2025. Cmmcot: Enhancing complex multi-image comprehension via multi-modal chain-of-thought and memory augmentation. arXiv preprint arXiv:2503.05255 (2025).

[59] Shilong Zhang, Peize Sun, Shoufa Chen, Min Xiao, Wenqi Shao, Wenwei Zhang, Yu Liu, Kai Chen, and Ping Luo. 2024. Gpt4roi: Instruction tuning large language model on region-of-interest. In European conference on computer vision. Springer, 52–70.

[60] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. 2025. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479 (2025).

# MG-Thinker: Bi-Axial Self-Reflection for Multi-Image Reasoning Grounding

Appendix

## A Multi-Image Grounding Tasks Categorization

Original taxonomy of multi-image grounding tasks. Prior multi-image grounding work [18] formulates ten sub-tasks under two top-level branches, distinguished by whether the query carries an explicit referential cue. Spontaneous Grounding (SG) requires the model to autonomously discover and localize a target through crossimage relations, comprising Static Diference Grounding (Static), Robust Diference Grounding (Robust), and Common Object Grounding (Common), the dificulty here lies in identifying what to localize purely from cross-image relations. Referential Grounding (RG) provides an explicit cue and is further split by cue modality into textual cue (Group Grounding, GG), visual cue (Object Tracking, OT; Multi-View Grounding, MV; Visual Referring Grounding, Refer; Region Locating, Region), and combined visual-textual cue (Reasoning Grounding, Reason; Correspondence, Co-Re); the dificulty here lies in correctly interpreting the cue and propagating it across images. All sub-tasks are evaluated with the standard $\operatorname { A c c } _ { 0 . 5 }$ IoU criterion, and we adopt these abbreviations whenever a sub-task is referenced throughout the paper.

Intrinsic heterogeneity-based re-categorization. The cue-based partition above, while convenient for benchmarking, does not directly reveal the MRG-specific source of dificulty during posttraining. We therefore re-organize the same ten sub-tasks along two largely orthogonal axes of intrinsic heterogeneity, cross-image visual variation (e.g., viewpoint shifts, repeated or distractor instances, layout changes that alter how the same scene or entity appears across images) and cross-image reasoning depth (e.g., temporal tracking, abstract correspondence, referential semantic logic that requires multi-step inference over the image set). The two axes manifest at the rollout-statistics level as the inter-group competence axis (mean reward $\bar { r } _ { g } )$ and the intra-group signal axis (variance $\sigma _ { g } ^ { 2 } ) _ { ; }$ respectively, on which the cross-task heterogeneity analysis of Tab. 4 (main paper) is based. Diferent sub-tasks load these two axes very diferently, and we group them by the dominant source of dificulty into three intrinsic categories, as shown in Fig. 7:

• Visual-Heterogeneity (V-Het): tasks where intrinsic heterogeneity concentrates on the visual axis, manifesting as pronounced mean-reward drift across sub-tasks (the intergroup competence axis); e.g., multi-view, group, and referential grounding, whose challenge lies in reconciling appearance variations of the same target across images.

• Reasoning-Heterogeneity (R-Het): tasks where intrinsic heterogeneity concentrates on the reasoning axis, manifesting as pronounced variance drift within rollout groups (the intra-group signal axis); $\mathrm { e . g . }$ , object tracking, correspondence, and reasoning grounding, whose dificulty escalates with the depth of cross-image inference required.

• Low-Heterogeneity (L-Het): tasks where neither axis dominates, with both mean and variance remaining stable; e.g., common-object localization in static scenes, exhibiting relatively uniform per-sample dificulty.

Each group spans both single-object and one-shot multi-object grounding, reflecting the distinct challenges of MRG. The Crosstask heterogeneity analysis in Tab. 4 (main paper) groups sub-tasks accordingly, isolating the complementary efects of GIA (signal axis) and CRS (competence axis) on each group.

## B CoT Prompt Templates for MRG-Specific Activation

Unlike prior CoT activation that targets single-image reasoning or generic visual QA, our prompt templates are purpose-built around two MRG-specific characteristics: (i) the joint presence of visual and textual referential cues that must be cross-interpreted across multi ple images, and (ii) a coarse-to-fine hierarchy that runs through both reasoning and grounding (CoT→image-id→bbox), so that semanticlevel inference and spatial-level localization unfold along the same trajectory. Crucially, the templates also explicitly support one-shot multi-image multi-target localization, an output mode rarely formulated by existing reasoning-grounding pipelines. We organize templates along the four MRG-specific task types categorized in §3.2, visual comparative analysis, spatial perception, temporal perception, and visual semantic/logical association, and use the stronger Qwen2.5-VL-72B annotator to instantiate them, so that multi-image evidence and multi-perspective thinking jointly guide reasoning and reasoning, in turn, sharpens grounding precision.

As illustrated in Fig. 8, each template comprises:

• Role definition. The model is positioned as a visual expert specialized in cross-image interpretation and analysis.

• Input specification. Multiple images, the user query, and the ground-truth answer are provided; the latter is used solely as a verification anchor and is never disclosed during the thinking phase.

• Task-oriented operators. Per-type instructions inject comparison, searching, observation, tracking, and association cues, encouraging step-wise multi-perspective evidenceseeking that integrates visual and textual referential cues.

• Reasoning constraints. Explicit prohibitions block leakage of ground-truth answers or bounding-box coordinates inside the <think> block, ensuring genuinely explanatory rationales rather than answer-leaking shortcuts.

• Hierarchical output format. A structured <think></think> trace, naturally partitioned into a coarse-to-fine semantic hierarchy, is followed by a JSON-formatted <answer></answer> block carrying multi-image bounding-box coordinates with explicit image\_id keys, enabling one-shot multi-image multitarget grounding within a single inference.

The two-step IoU-improvement post-processing (Sec. 3.3) then filters pseudo-reasoning samples and retains only rationales that demonstrably improve grounding, yielding the final 25K cold-start corpus.

![](images/308836e6fa44f2a392939ee679e51139458abbcb56c071c7cf48b8a668e38db3.jpg)  
Figure 7: Restructured multi-image grounding data, organized by intrinsic task heterogeneity into Visual-Heterogeneity (V-Het) Reasoning-Heterogeneity (R-Het), and Low-Heterogeneity (L-Het) groups. Each group spans both single-object and one-shot multi-object grounding, reflecting the distinct challenges of MRG.

## C DAPO Clipped Surrogate Objective and Group-Relative Advantage

For completeness we recap the DAPO clipped surrogate on top of which BiA-DAPO is built. Given a question–answer pair (�, �) and a sampled rollout group $\{ o _ { i } \} _ { i = 1 } ^ { G }$ drawn from the old policy $\pi _ { \theta _ { \mathrm { o l d } } } ,$ DAPO optimizes

$$
\begin{array} { l } { \mathcal { T } _ { \mathrm { { D A P O } } } ( \theta ) = \mathbb { E } _ { ( q , a ) , \{ o _ { i } \} \sim \pi _ { \theta _ { \mathrm { o l d } } } } \biggl [ \frac { 1 } { \sum _ { i } | o _ { i } | } \sum _ { i , t } \qquad } \\ { \quad \operatorname* { m i n } \bigl ( r _ { i , t } \hat { A } _ { i , t } , \mathrm { c l i p } ( r _ { i , t } , 1 - \varepsilon _ { l } , 1 + \varepsilon _ { h } ) \hat { A } _ { i , t } \bigr ) \biggr ] , } \end{array}\tag{7}
$$

with the per-token importance ratio

$$
r _ { i , t } ( \theta ) = \frac { \pi _ { \theta } ( o _ { i , t } \mid q , o _ { i , < t } ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( o _ { i , t } \mid q , o _ { i , < t } ) } ,\tag{8}
$$

and asymmetric clip-higher coeficients $\varepsilon _ { l } = 0 . 2 , \varepsilon _ { h } = 0 . 2 8$ . The grouprelative advantage of rollout � is

$$
\hat { A } _ { i } = \frac { \mathrm { a c c } _ { i } - \bar { r } _ { g } } { \sigma _ { g } + \varepsilon } ,\tag{9}
$$

where $\bar { r } _ { g }$ and $\sigma _ { g }$ are the within-group mean and standard deviation of $\mathrm { a c c } _ { i } { = } R _ { \mathrm { i d } , i } { + } R _ { \mathrm { I o U } , i }$ , and $\varepsilon { = } 1 0 ^ { - 6 }$ is a numerical guard. BiA-DAPO retains this surrogate form and additionally reuses the orthogonal pair $( \bar { r } _ { g } , \sigma _ { g } )$ as Axis-2/Axis-1 statistics for sample-level intervention by GIA and CRS (Sec. 3.3).

## D Empirical Dynamics of GIA and CRS

We present two complementary diagnostic views that empirically substantiate the bi-axial mechanism of BiA-DAPO. The two views correspond, respectively, to within-group (Axis-1) and joint Axis-1×Axis-2 interventions discussed in Sec. 3.3, and underpin the qualitative claims in §4.2 Efectiveness of reinforcement post-training. Per-step $R _ { \mathbf { I o U } }$ outcomes (Axis-1 view). Fig. 9 stratifies per-step rollouts into three IoU regions: failed $\left( R _ { \mathrm { I o U } } = 0 \right)$ , partial (linearly interpolated), and saturated $\left( R _ { \mathrm { I o U } } = 1 \right)$ . Under GRPO/DAPO, the failed share remains persistently large, manifesting as $\sigma _ { g } { \approx } 0$ collapsed groups and confirming drifting advantage-signal sparsity along Axis-1. BiA-DAPO progressively shrinks the failed share, GIA’s variance-driven gating retains advantage-bearing groups, and grows the saturated share, CRS’s competence cascade pulls the policy into higher-competence strata, evidencing that the two pathologies are jointly relieved.

Bi-axial $( \bar { r } _ { g } , \sigma _ { g } )$ density migration (Axis-1×Axis-2 view). Fig. 10 visualizes the joint distribution of group-level $( \bar { r } _ { g } , \sigma _ { g } )$ across training steps. GRPO/DAPO accumulate mass in the upper-left quadrant (low competence, high spread) and the lower-left strip (low competence, near-zero spread), reflecting both misalignment and sparsity. BiA-DAPO migrates the density diagonally toward the lower-right quadrant (high competence, moderate spread), precisely the trajectory predicted by GIA+CRS coupling, where Axis-1 sparsity is absorbed by GIA gating while Axis-2 misalignment is corrected by CRS stratification.

![](images/785df249efdb4b36048f2977fe09984a335b147db474d604cc6fed349d334188.jpg)

Figure 8: CoT templates designed for the four MRG-specific task types and the standard response format. ∗ denotes textual content and # denotes natural numbers; in the generated CoT templates, ∗ is replaced with task-specific descriptive terms.  
![](images/c6979ff5fb36f6aa4773b652c28e604c734ef377d86dbecdd9103c00f467a11e.jpg)  
Figure 9: Per-step $R _ { \mathbf { I o U } }$ outcome shares (failed/partial/saturated) across training. BiA-DAPO progressively shrinks the failed share (GIA) and grows the saturated share (CRS), whereas GRPO/DAPO retain a large failed share due to drifting advantage-signal sparsity.

Together, these two views provide direct evidence that GIA and CRS act on complementary statistical axes rather than redundant ones, consistent with the cross-task heterogeneity analysis in Tab. 4.

## E RL Post-Training Configuration and Hyperparameters

This section details the reinforcement post-training configuration (VeRL-based) and the BiA-DAPO-specific hyperparameters; basic settings (learning rate, hardware, global schedule) follow §4.1. VeRL-based RL configuration. We implement RL post-training on top of the VeRL framework.<sup>1</sup> Tab. 7 lists the RL-specific hyperparameters that most influence sampling behavior and stability. Inputs are loaded from RL-formatted parquet files keyed by prompt and images, with overlong-prompt filtering and strict truncation checks. We sample multiple responses per prompt during rollout and optimize the policy with the DAPO-style clipped surrogate (Appendix C) over a GRPO-style group-relative advantage estimator. KL regularization is applied as an explicit KL loss using a low-variance KL formulation (rather than a reward-shaping term). BiA-DAPO hyperparameters. BiA-DAPO is instantiated by two complementary mechanisms, Group-Informativeness Assessment (GIA) and Cascaded-Reward Stratification (CRS), operating over a 3× candidate pool. Three groups of hyperparameters are required:

![](images/9bb51c34c7249ea47f4a6f01ec3a0bff9a42cd5be21225a1e5836964cf411ff9.jpg)  
Figure 10: Bi-axial $( \bar { r } _ { g } , \sigma _ { g } )$ density migration during training. BiA-DAPO migrates mass diagonally from upper-left (low competence, high spread) to lower-right (high competence, moderate spread), corroborating the joint Axis-1+Axis-2 intervention by GIA and CRS.

![](images/139125516095cef3e71f7246a14ad30359da4881369cee0b6fbfc7bdd72c2cba.jpg)  
Figure 11: Continuous-variation sensitivity of BiA-DAPO on MIG-Bench: GIA threshold $\alpha _ { 0 } ,$ candidate-pool size, and number of GIA phases.

Table 7: RL configuration in VeRL for reinforcement post-training. Basic settings (learning rate, hardware, global schedule) are specified in §4.1.
<table><tr><td>Setting</td><td>|Value</td><td>Rationale</td><td>Where used</td></tr><tr><td>Training batch / Candi- date pool</td><td>16  / 48</td><td>Pool follows DAPO&#x27;s “generation batch&quot; (3× training batch) for stable axis estimation</td><td>Rollout / selection</td></tr><tr><td>Max response length</td><td>1024</td><td>Matches the RL post-training generation budget across all methods</td><td>Rollout generation</td></tr><tr><td>Responses per prompt</td><td>16</td><td>Fixed group size for within-group statistics (GIA) and reward-mean ranking (CRS)</td><td>Rollout sampling</td></tr><tr><td>Sampling (train)</td><td> $\mathrm { t e m p } = 1 . 0 , \mathrm { t o p } { - \mathscr P } = 1 . 0 , \mathrm { t o p } { - mathscr k } = - 1$ </td><td>Default exploration-oriented sampling</td><td>Rollout generation</td></tr><tr><td>Sampling (val)</td><td>top-p = 0.7, do_sample = true, n = 1</td><td>Conservative evaluation sampling</td><td>Validation rollout</td></tr><tr><td>PPO clipping</td><td> $\varepsilon _ { l } = 0 . 2 , \varepsilon _ { h } = 0 . 2 8$ </td><td>Asymmetric clip-higher in our VeRL configuration</td><td>Policy optimization</td></tr><tr><td>Loss aggregation</td><td>token-mean</td><td>Normalization for variable-length sequences</td><td>Policy optimization</td></tr><tr><td>KL usage</td><td>use_kl_in_reward = false; use_kl_loss = true</td><td>KL regularization applied as a loss term</td><td>Regularization</td></tr><tr><td>KL coefficients</td><td>kl_loss_coef = 0.01; kl_coef = 0.001</td><td>Fixed coefficients in our RL setting</td><td>Regularization</td></tr><tr><td>KL type</td><td>low_var_kl</td><td>Low-variance KL formulation in VeRL</td><td>Regularization</td></tr></table>

Table 8: Single-axis corner-case ablations of BiA-DAPO on MIG-Bench. Zero-Threshold: $\alpha _ { 0 } { = } 0$ with pool 48 (open gate, equivalent to w/o. GIA). Unexpanded Pool: pool = 16 (no candidate-pool expansion). Static-�: �=0.05 frozen (no phase decay). All variant rows use the Poll output format; the MG-Thinker row reports the better of Poll/All on the four multi-target sub-tasks, as in Tab. 1.
<table><tr><td>Setting</td><td>Static</td><td>Robust</td><td>Common</td><td>OT</td><td>MV</td><td>Region</td><td>Refer</td><td>GG</td><td>Reason</td><td>Co-Re</td><td>AVG</td></tr><tr><td>Zero-Threshold</td><td>78.79</td><td>60.64</td><td>90.80</td><td>78.91</td><td>67.01</td><td>86.70</td><td>85.86</td><td>89.69</td><td>76.29</td><td>41.03</td><td>75.57</td></tr><tr><td>Unexpanded Pool</td><td>77.43</td><td>60.64</td><td>91.17</td><td>81.81</td><td>67.01</td><td>85.04</td><td>81.82</td><td>88.25</td><td>74.23</td><td>47.01</td><td>75.44</td></tr><tr><td>Static-α</td><td>79.55</td><td>60.64</td><td>89.33</td><td>80.36</td><td>64.93</td><td>89.07</td><td>86.03</td><td>85.86</td><td>71.13</td><td>47.01</td><td>75.39</td></tr><tr><td>MG-Thinker</td><td>78.98</td><td>62.77</td><td>93.01</td><td>82.55</td><td>66.67</td><td>88.78</td><td>86.87</td><td>88.25</td><td>72.16</td><td>47.01</td><td>76.71</td></tr></table>

Table 9: Performance on RefCOCO/+/g referring expression grounding. Best in bold, second-best underlined.
<table><tr><td rowspan="2">Models</td><td colspan="3">RefCOCO</td><td colspan="3">RefCOCO+</td><td colspan="2">RefCOCOg</td><td rowspan="2">AVG</td></tr><tr><td>val</td><td>testA</td><td>testB</td><td>val</td><td>testA</td><td>testB</td><td>val</td><td>test</td></tr><tr><td>VisionLLM v2 [45]</td><td>79.20</td><td>82.30</td><td>77.00</td><td>68.90</td><td>75.80</td><td>61.80</td><td>73.30</td><td>74.80</td><td>74.14</td></tr><tr><td>Shikra</td><td>87.00</td><td>90.60</td><td>80.20</td><td>81.60</td><td>87.40</td><td>72.10</td><td>82.30</td><td>82.20</td><td>82.97</td></tr><tr><td>InternVL2-8B</td><td>87.10</td><td>91.10</td><td>80.70</td><td>79.80</td><td>87.90</td><td>71.40</td><td>82.70</td><td>82.70</td><td>82.94</td></tr><tr><td>GroundingGPT [20]</td><td>88.02</td><td>91.55</td><td>82.47</td><td>81.61</td><td>87.18</td><td>73.18</td><td>81.67</td><td>81.99</td><td>83.57</td></tr><tr><td>Griffon v2</td><td>89.60</td><td>91.80</td><td>86.50</td><td>81.90</td><td>85.50</td><td>76.20</td><td>85.00</td><td>86.00</td><td>85.30</td></tr><tr><td>GroundingDINO-L [24]</td><td>90.60</td><td>93.20</td><td>87.20</td><td>82.80</td><td>89.00</td><td>75.90</td><td>86.10</td><td>87.00</td><td>86.60</td></tr><tr><td>Qwen2.5-VL-7B</td><td>90.00</td><td>92.50</td><td>85.40</td><td>84.20</td><td>89.10</td><td>76.90</td><td>87.20</td><td>87.20</td><td>86.56</td></tr><tr><td>Migician</td><td>91.62</td><td>93.49</td><td>87.22</td><td>86.13</td><td>91.06</td><td>79.93</td><td>88.06</td><td>87.80</td><td>88.16</td></tr><tr><td>UniVG-R1</td><td>91.64</td><td>93.11</td><td>87.16</td><td>85.91</td><td>90.53</td><td>80.04</td><td>88.67</td><td>88.56</td><td>88.20</td></tr><tr><td>MG-Thinker</td><td>91.28</td><td>93.55</td><td>87.48</td><td>85.79</td><td>90.50</td><td>79.97</td><td>88.68</td><td>88.22</td><td>88.18</td></tr></table>

(i) a phase-decayed schedule with four phases for the GIA gate, (ii) the GIA threshold � , and (iii) the candidate-pool size. The fourphase design originates from a one-epoch GRPO baseline scan: we collect the within-group rollout standard-deviation distribution and observe four well-separated regimes, zero-spread groups (sparsitycollapsed, no learning signal), low-, medium-, and high-spread groups, motivating a four-way partition aligned with these empirical boundaries. Following this categorization, we set the initial threshold to � =0.05 and decay it by a factor of two at each sub sequent phase, so that the gate progressively admits lower-spread groups as the policy stabilizes. For CRS, the candidate-pool size is fixed at 3× the training batch (matching DAPO’s generation batch) to ensure a consistent $\bar { r } _ { g } .$ -ranking budget across update steps.

![](images/6e2ae5838b23d7b6acaff77f61b41cb127db141c5f186fc6ef65c11986e9982a.jpg)

Figure 12: Comparison of non-reasoning and various reasoning paradigms on a representative cross-image reasoning grounding task: direct prediction, CoT-prompting, MG-Thinker cold-start, and MG-Thinker fully trained with BiA-DAPO post-training.  
![](images/5a360f31df20b16bd5b495bafeb4c428a66374a601781048e18f6bb44179890c.jpg)  
Figure 13: Visual analysis on MIG-Bench correspondence grounding tasks; ground-truth bounding-box annotations are shown in white to distinguish them from predictions.

## F Hyperparameter Sensitivity Analysis

We further study BiA-DAPO’s sensitivity on MIG-Bench by varying one hyperparameter at a time around the default configuration (�<sub>0</sub>=0.05, candidate pool = 48, GIA phases = 4). The continuousvariation curves are summarized in Fig. 11, while Tab. 8 reports per-task accuracies at three representative single-axis corner cases. All averages in this section and in Fig. 11 are computed under the

Poll output format for every configuration, including the default, which therefore reads 76.36 here; Tab. 1 reports MG-Thinker’s better-of-format result (76.71).

Specifically, varying the GIA threshold over $\alpha _ { 0 } \epsilon$ {0.05, 0.025, 0.0} yields average accuracies of 76.36/76.15 (−0.21)/75.57 (−0.79), indicating that a strictly positive threshold is beneficial and moderate variations are well tolerated; varying the candidate pool size over {48, 32, 16} produces 76.36/76.22 (−0.14)/75.44 (−0.92), where an excessively small pool harms group-statistics stability while a medium pool stays close to the default; varying the number of GIA phases over {4, 2, 1} gives 76.36/75.43 (−0.93)/75.30 (−1.06), showing that coarser schedules consistently underperform.

Single-axis corner cases. Tab. 8 further examines three cornercase variants that each disable one BiA-DAPO design choice: Zero-Threshold (�<sub>0</sub>=0, pool = 48) opens the gate to all groups and is operationally equivalent to w/o. GIA; Unexpanded Pool (1× pool, i.e., pool = 16) cancels the candidate-pool expansion so that group statistics are computed only over the training batch; Static-� (�=0.05 frozen) freezes the threshold at its initial value without phase decay. All three consistently underperform the full default by ∼1 point on average and on most subtasks, confirming that each design choice contributes a non-redundant slice of the bi-axial mechanism. Overall, BiA-DAPO is robust under moderate variations but degrades under extreme settings, supporting the rationality of the chosen defaults.

## G Analysis of Reasoning Paradigms (Case Study)

Complementing the quantitative comparison in §4.2 Analysis of Reasoning Paradigms, Fig. 12 provides a representative case on a cross-image correspondence task: identifying and localizing, in Image-2, the object that matches a target person in Image-1. Solving this task requires (1) anchoring the target attributes in Image-1, (2) reasoning over Image-2 candidates by attribute matching, and (3) grounding the final decision via accurate localization. In the illustrated example, the target girl in Image-1 carries distinctive paint traces on her hands, face, and clothing, naturally suggesting drawing-related correspondences in Image-2. We compare four configurations:

• Qwen2.5-VL (direct prediction). Without an explicit reasoning scafold, the model fails to connect target attributes in Image-1 to candidates in Image-2, yielding incorrect localization.

• Qwen2.5-VL + CoT prompting. A generic CoT trigger induces a free-form rationale, but the trajectory may deviate from discriminative cues (reasoning drift), still leading to erroneous localization.

• MG-Thinker (cold-start). The cold-start checkpoint can localize correctly but its rationale may be partially inconsistent with the visual evidence, indicating an underdeveloped reasoning–grounding alignment.

• MG-Thinker (fully trained). The BiA-DAPO post-trained model exhibits a systematic anchor–reason–ground pattern: it first identifies target attributes in Image-1, then verifies matching cues in Image-2, and finally outputs accurate localization with a coherent explanation.

This case highlights distinct failure modes of non-reasoning inference and prompt-induced CoT, and demonstrates that our pipeline improves the consistency between reasoning traces and grounding outputs, critical for reliable multi-image correspondence and localization.

## H Performance on RefCOCO/+/g Benchmarks

We supplement the §4.2 General single-image grounding discussion with the full RefCOCO/+/g comparison. Tab. 9 reports MG-Thinker against representative state-of-the-art single-image grounding methods, including VisionLLM v2 [45], Shikra, InternVL2-8B,

GroundingGPT [20], Grifon v2, GroundingDINO-L [24], Qwen2.5- VL-7B, Migician, and UniVG-R1. Despite being optimized for multiimage reasoning grounding, MG-Thinker remains strongly competitive on conventional single-image referring expression grounding, confirming that our post-training preserves general grounding capability rather than over-specializing to multi-image tasks.

## I Qualitative Cases on MIG-Bench

Complementing the MIG-Bench quantitative results, we present qualitative case studies that contrast MG-Thinker with strong multiimage baselines, with an emphasis on dedicated multi-image grounding models. We analyze representative behaviors across semantic/logical reasoning, region selection under repeated distractors, and simultaneous multi-object tracking.

Semantic and logical reasoning across image sequences. Fig. 14 and Fig. 13 examine cases requiring semantic logical reasoning and abstract cross-image association. Qwen2.5-VL under zero-shot prompting frequently fails to establish query-to-sequence correspondence, leading to inconsistent or arbitrary localization. Migician, while strong at direct end-to-end grounding, tends to underutilize semantic reasoning, resulting in rigid localization driven primarily by surface text cues. For reasoning-enabled baselines, reasoning can be a double-edged sword: UniVG-R1 may exhibit (i) plausible but incorrect reasoning that steers grounding toward wrong regions, or (ii) partially correct reasoning whose final localization is misaligned with the visual evidence, indicating a coupling gap between reasoning traces and spatial decisions. By contrast, MG-Thinker more reliably leverages semantic and logical constraints to regulate the reasoning process, and its reasoning signals translate into more precise and consistent localization.

Multi-region selection under repeated distractors. Fig. 15 evaluates multi-image region grounding scenarios involving selection among multiple candidate regions and localization of inconspicuous, repeated small objects. MG-Thinker better suppresses interference from repeated instances by jointly considering positional relations and fine-grained visual attributes, yielding more stable region selection when distractors share similar appearance.

Simultaneous multi-object tracking grounding. Fig. 16 highlights a challenging multi-object tracking setting where models must output multiple bounding boxes in a single inference. Qwen2.5- VL frequently produces chaotic, repetitive outputs; UniVG-R1 struggles to reconcile multi-object output with its reasoning-and-grounding procedure; and Migician’s accuracy degrades when scaling to simultaneous multi-object localization. MG-Thinker remains more coherent in organizing outputs and maintaining alignment between predicted boxes and temporal/identity cues across frames, although the task remains challenging for all models.

![](images/3e738f223b60624bb0e907c5c3d8521fa5bd4595003b1c91bfd72873d8ad1850.jpg)

Figure 14: Visual analysis on MIG-Bench reasoning grounding tasks: comparison of MG-Thinker against representative multiimage baselines.  
![](images/7583c1a6887582f33021022960bc94e0438e0a02379612f4367b84717fb462a7.jpg)  
Figure 15: Visual analysis on MIG-Bench region grounding tasks under repeated distractors.

![](images/b9c2a951733cf58cdfb93c7b889bcb4d1b4ee658b69e66c20b207639ccc013f1.jpg)  
Figure 16: Visual analysis on MIG-Bench object-tracking grounding tasks; ground-truth annotations are shown in white. Numerical values follow each model’s coordinate convention (relative or absolute).