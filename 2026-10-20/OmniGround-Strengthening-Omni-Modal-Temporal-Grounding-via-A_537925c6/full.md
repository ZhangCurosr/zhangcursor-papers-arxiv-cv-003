# OmniGround: Strengthening Omni-Modal Temporal Grounding via Attention-Consistency Training

Zixuan Chen<sup>1</sup>, Depeng Wang<sup>2</sup>, Hao Lin<sup>3</sup>, Ke Xu<sup>1</sup>, Ya Guo<sup>2</sup>, Huijia Zhu<sup>2∗</sup>, Tanfeng Sun<sup>1</sup>, Xinghao Jiang<sup>1∗</sup>

<sup>1</sup>Shanghai Jiao Tong University <sup>2</sup>Ant Group <sup>3</sup>Independent Researcher

## Abstract

Omni-modal large language models can jointly understand video, audio, and text, yet they remain weak at temporal grounding, a capability essential for real-world audio-visual applications. To understand this limitation, we look beyond final predictions and examine the model’s internal temporal evidence. We identify a temporal generation gap: the internal attention of the model localizes the target moment more accurately than the timestamps it generates. To close this gap, we propose TACG (Temporal Attention-Consistency Grounding), a post-training method that treats internal attention as a reliable temporal pivot and aligns it with both the model prediction and the ground-truth annotation through attentionconsistency training. We further build a 580K-sample training dataset by reconstructing temporal QA from dense audiovisual annotations and organizing general omni-modal instruction data. Using TACG and this training dataset, we train OmniGround, a general-purpose omni-modal model with stronger temporal grounding capability, without injecting any additional temporal cues. OmniGround reaches state-ofthe-art omni-modal temporal grounding, including 37.5 mIoU on LongVALE and 35.7 mIoU on UnAV-100, while staying competitive on both general audio-visual understanding and vision-only grounding.

## Introduction

Omni-modal large language models (Xu et al. 2025a; Liu et al. 2025; Wu et al. 2024), which jointly perceive video, audio, and text within a single backbone, have advanced rapidly and now excel at a wide range of multimodal understanding tasks. As these models move toward real-world audio-visual applications such as video editing, content moderation, and multimodal retrieval, recognizing what happens no longer sufices: a model must also pinpoint when it happens. This temporal capability is critical for practical audio-visual applications, with temporal video grounding (TVG) serving as a central benchmark. Yet current omni-modal models still perform poorly on TVG despite their strong general understanding.

Existing temporal grounding methods mainly follow two directions. One family modifies the model or its inputs by injecting explicit temporal information, such as special temporal tokens, frame markers, or task-specific heads (Guo et al. 2025a; Wu et al. 2025; Pramanick et al. 2025). Although efective for localization, these designs make temporal grounding depend on customized input formats and specialized output parsing, which limits their compatibility with a general-purpose omni-modal interface. A second family improves grounding purely through post-training, such as reinforcement learning (Wang et al. 2025b) or distillation from a stronger teacher (Li et al. 2026a). This training-only direction preserves the inference interface and better maintains general ability, but does not directly exploit the model’s own temporal evidence. We adopt this philosophy, but push it further.

We begin with a diagnostic question: when an omni-modal model grounds imprecisely, does it fail to localize the event, or merely to express the time? For each query, we compare the interval read from the model’s internal attention with its generated timestamp, and measure which lands closer to the ground truth. Across multiple omni-modal models and grounding datasets, the attention-derived interval is more accurate than the prediction on a clear majority of samples, up to 78.3%. We call this the temporal generation gap: the model knows where to look but cannot accurately say when. This pattern holds for both base models and models with stronger grounding ability, indicating that the gap is not merely a weak-model artifact, as illustrated in Figure 1. If left unaddressed, this gap limits further improvements in temporal localization precision.

Building on this diagnosis, we propose Temporal Attention-Consistency Grounding (TACG). Its core term aligns the generated timestamp with the model’s own internal attention, using the more reliable internal signal to pull the prediction onto the target and directly close the temporal generation gap. Two auxiliary terms anchor this correction to the ground truth: one supervises the attention directly against the annotation, making the attention itself localize the target more accurately; the other supervises the prediction by its temporal distance to the annotation, supplying the distance-aware gradient that cross-entropy lacks. TACG achieves stronger temporal grounding than standard supervised fine-tuning on the same data, without introducing additional temporal inputs or model parameters.

We also build a 580k-sample omni-modal training dataset that supports both temporal grounding and general understanding. Starting from the dense audio-visual annotations in AVID (Chen et al. 2026) and LongVALE (Geng et al. 2025), we extract high-quality temporal segments and reformulate them into grounding, dense captioning, and temporal question-answering tasks. We further incorporate general omni-modal instruction data from several sources (Zhang et al. 2024; Liu et al. 2025; Yang et al. 2022; Ma et al. 2025) to preserve broad audio-visual understanding. Using this dataset and TACG, we train OmniGround from Qwen2.5- Omni (Xu et al. 2025a). As shown in Figure 1, OmniGround improves temporal grounding without the extra temporal inputs or specialized output formats used by prior baselines. We evaluate OmniGround on seven benchmarks, including three omni-modal grounding benchmarks, two vision-only grounding benchmarks, and two general omni-modal understanding benchmarks. OmniGround sets a new state of the art on omni-modal temporal grounding while staying competitive on the other two categories. Our contributions are summarized as follows:

![](images/21ea5b2a6c4d4ca59fb7a6ebbd44aea8fc9352e054c0628afeb16e515ae5a686.jpg)  
Figure 1: Overview of OmniGround. Left: both the base model and the stronger SFT model exhibit a temporal generation gap: internal attention localizes the target event more accurately than the generated timestamp. TACG closes it to achieve stronger grounding. Right: unlike methods that rely on additional temporal inputs or specialized output formats, OmniGround strengthens temporal grounding while preserving a general-purpose omni-modal interface.

• We identify the temporal generation gap in omni-modal language models and propose TACG, a post-training objective that aligns the model’s prediction with its internal attention, consistently improving over standard supervised fine-tuning under identical training data.

• We build a 580k omni-modal temporal-awareness dataset by reformulating dense audio-visual annotations into temporal grounding and question-answering tasks, applying quality filtering, and balancing the resulting data with curated general data. We train OmniGround on this dataset.

• OmniGround reaches state-of-the-art omni-modal temporal grounding and stays competitive on general omnimodal understanding and vision-only grounding. We will release the dataset, model, and training framework.

## Related Work

Omni-Modal Language Models. Recent work extends large language models beyond text to perceive multiple modalities. Early video LLMs such as VideoChat (Li et al. 2023), Video-ChatGPT (Maaz et al. 2024), and VideoLLaMA (Zhang et al. 2023) couple a visual encoder with an LLM, while PandaGPT (Su et al. 2023) and NExT-GPT (Wu et al. 2024) add audio. More recent omni-modal models further unify audio, vision, and text within a shared modeling framework, enabling general audio-visual-text understanding (Liu et al. 2025; Xu et al. 2025a). While these models have achieved strong general audio-visual understanding, they are not explicitly optimized for precise temporal grounding. As a result, they often recognize the relevant event but fail to generate accurate temporal boundaries. Our work strengthens temporal grounding in a general-purpose omni-modal model.

Model Mechanism Analysis. Prior studies show that model behavior can be analyzed through internal signals (Zhao et al. 2024; Palit et al. 2023). Inspired by this perspective, we inspect internal attention in omni-modal temporal grounding.

Temporal Video Grounding. Temporal video grounding (TVG) localizes the moment described by a natural language query, and existing methods difer in how they acquire this ability. One family injects explicit temporal machinery— regression heads in specialist architectures (Zhang et al. 2020; Lin et al. 2023; Jang et al. 2023; Mu, Mo, and Li 2024), or, in video LLMs, time-aware encoders (Ren et al. 2024), special temporal tokens (Guo et al. 2025a; Wang et al. 2025a; Chen et al. 2025), frame-index markers (Wu et al. 2025), and localization heads (Pramanick et al. 2025; Huang et al. 2024; Wang et al. 2024; Zeng et al. 2025; Guo et al. 2025b; Li et al. 2026b; Qu et al. 2024; Hannan et al. 2025). These components depart from the standard backbone, add inference cost, and often weaken general question answering. A second family leaves the architecture and inference untouched and strengthens temporal grounding through additional post-training stages, such as reinforcement learning with IoU rewards (Wang et al. 2025b) or distillation from a stronger teacher (Li et al. 2026a). Our method instead strengthens fine-tuning itself, turning the model’s internal attention into a self-supervision signal.

![](images/fcbb66f82b755a997de18acb93384a4687669d2c0302a83c2d91d7d95a70c9a6.jpg)  
(a) Qwen2.5-Omni-3B (base)  
(b) Qwen2.5-Omni-3B (SFT)

Figure 2: Visualization of the temporal generation gap. Top: the attention averaged over samples and aligned to the ground-truth event (green), overlaid with the averaged prediction. Bottom: the corresponding averaged attention and predicted intervals. In both base and SFT models, the attention localizes the event more accurately than the prediction.
<table><tr><td colspan="3">AVID</td><td colspan="2">LongVALE</td></tr><tr><td>Model</td><td>Attn. wins†</td><td></td><td>mIoU Attn. wins† mIoU</td><td></td></tr><tr><td>Qwen2.5-Omni-3B</td><td>73.9%</td><td>7.4</td><td>78.3%</td><td>6.3</td></tr><tr><td>Qwen2.5-Omni-3B (SFT)</td><td>57.7%</td><td>17.9</td><td>60.0%</td><td>20.9</td></tr><tr><td>Qwen2.5-Omni-7B</td><td>75.7%</td><td>6.2</td><td>72.1%</td><td>12.1</td></tr><tr><td>Qwen3-Omni</td><td>65.0%</td><td>14.1</td><td>69.7%</td><td>10.6</td></tr><tr><td>OLA</td><td>73.0%</td><td>10.3</td><td>70.3%</td><td>7.7</td></tr></table>

Table 1: The temporal generation gap across models and datasets. For each benchmark, all statistics are computed on a fixed evaluation subset of 500 examples randomly sampled from its oficial test split. Attn. wins<sup>†</sup> reports the fraction of non-tied samples where $\mathrm { I o U } ( I _ { \mathrm { a t t } } , I _ { \mathrm { g t } } ) \ \stackrel { - } { > } \ \mathrm { I o U } ( I _ { \mathrm { p r e d } } , I _ { \mathrm { g t } } )$ mIoU is the grounding accuracy of the generated interval. This gap is consistent across models and datasets.

## Preliminary: Temporal Generation Gap

We validate the temporal generation gap across multiple omni-modal models on AVID and LongVALE, including both base models and models with stronger grounding ability. For each benchmark, we construct the diagnostic evaluation split by randomly sampling 500 examples from its oficial test split and use this fixed subset for all preliminary-analysis statistics. For each sample we obtain three temporal intervals and compare them against the ground truth by temporal IoU. The ground-truth interval $I _ { \mathrm { { g t } } }$ and the generated interval $I _ { \mathrm { p r e d } }$ are taken directly. We extract the attention interval $I _ { \mathrm { a t t } }$ from the attention of the last prefill token to the audio-visual tokens at a mid-to-late layer of the model. This attention is aggregated over spatial positions into a temporal distribution, from which we select the shortest contiguous window covering 50% of the cumulative mass. This gives three intervals for each case, $I _ { \mathrm { g t } } , \ I _ { \mathrm { p r e d } }$ , and $I _ { \mathrm { a t t } } ;$ we compare IoU $\lceil ( I _ { \mathrm { a t t } } , I _ { \mathrm { g t } } )$ with $\mathrm { I o U } ( I _ { \mathrm { p r e d } } , \breve { I } _ { \mathrm { g t } } )$ to determine whether the attention interval or the generated prediction is more accurate, and use the same intervals for visualization. For Qwen2.5-Omni-3B, we use layer 27; additional details, including layer choices for other models, layer sensitivity, cumulative-mass threshold sensitivity, and the SFT setup, are provided in the Appendix.

Figure 2 visualizes the internal attention concentrating on the target region while the generated interval is displaced, and Table 1 shows that $I _ { \mathrm { a t t } }$ overlaps the ground truth better than $I _ { \mathrm { p r e d } }$ on most valid samples across five omni-modal models and two datasets, despite large diferences in grounding mIoU. Together, these results establish that the temporal generation gap is consistent across models and datasets. After SFT, both the attention interval and the generated prediction become more accurate, yet the instance-level gap remains, suggesting that unresolved attention–prediction misalignment limits further improvements in temporal localization precision. This motivates directly aligning attention, prediction, and ground truth in the training objective introduced next.

## Method: OmniGround

We present OmniGround, an omni-modal temporal grounding model built on Qwen2.5-Omni. We train it with TACG on our 580k-sample omni-modal training dataset. The framework is illustrated in Figure 3.

## Temporal Attention-Consistency Grounding

The preliminary analysis motivates supervising the generated timestamp with the model’s internal attention while anchoring both signals to the ground truth. Although temporal IoU measures their agreement at evaluation time, it is unsuitable as a training signal: it is non-diferentiable and least informative exactly when most needed. Once two intervals are disjoint, its gradient vanishes everywhere, giving no indication of which direction, or how far, to move the prediction—precisely the regime that dominates early training. We therefore recast attention, prediction, and ground truth as continuous temporal distributions over the $\check { T }$ time steps and enforce their mutual consistency with divergences that stay informative under non-overlap (Jensen–Shannon and Wasserstein-1, chosen per edge below).

Attention–prediction consistency: the core objective. The central constraint of TACG forces the generated timestamp to agree with the localization already present in the internal attention. We implement this constraint by expressing both quantities as temporal distributions over the $\bar { T }$ time steps.

We first extract the implicit localization signal from the attention at a selected intermediate layer l. We compute the attention from the last prefill token to all media tokens (both video and audio) and aggregate the mass that falls on each of the T interleaved time steps:

$$
p _ { \mathrm { a t t } } [ t ] = \sum _ { i \in \mathcal { C } _ { t } } \frac { \exp ( { q ^ { \top } k _ { i } } / { \sqrt { d } } ) } { \sum _ { j } \exp ( { q ^ { \top } k _ { j } } / { \sqrt { d } } ) } ,\tag{1}
$$

where $q$ is the query of the last token, $k _ { i }$ ranges over all video and audio tokens, and $\mathcal { C } _ { t }$ is the set of media tokens at time step t. As the attention is normalized over all media tokens, $p _ { \mathrm { a t t } }$ is automatically a valid distribution over the T steps, and gradients flow through the layer-l query and key projections, allowing the consistency loss to refine the attention itself.

![](images/73dd0036977b94e8e223fbfc6a632dce11f370ece6241e2615416fa207741251.jpg)  
Figure 3: Overall architecture and TACG training of OmniGround.

We then convert the predicted timestamp into a continuous temporal distribution. Since standard supervised fine-tuning never decodes the logits, we operate on them directly. At each digit position of the timestamp tokens we form the soft expectation over digit values, round it, apply a straightthrough estimator so that gradients still reach the logits, and assemble the start and end times by place value:

$$
\begin{array} { l } { { \hat { d } _ { i } = \displaystyle \sum _ { d \in \mathcal { D } } d \cdot \mathrm { s o f t m a x } ( \mathrm { l o g i t s } _ { i } [ \mathcal { D } ] ) _ { d } , \quad \bar { d } _ { i } = \lfloor \hat { d } _ { i } \rceil , } } \\ { { \displaystyle \tilde { d } _ { i } = \bar { d } _ { i } - \mathrm { s g } ( \hat { d } _ { i } ) + \hat { d } _ { i } , } } \\ { { \displaystyle \hat { t } _ { s } = \sum _ { i } \tilde { d } _ { i } ^ { s } 1 0 ^ { p _ { i } } , \quad \hat { t } _ { e } = \sum _ { i } \tilde { d } _ { i } ^ { e } 1 0 ^ { p _ { i } } , } } \end{array}\tag{2}
$$

where $\mathcal { D } = \{ 0 , \ldots , 9 \}$ are the digit token indices, ⌊·⌉ rounds to the nearest integer, sg(·) is the stop-gradient (detach) operator, $p _ { i }$ is the (signed) place value of digit i, and $\tilde { d } _ { i } ^ { s } , \tilde { d } _ { i } ^ { e }$ are the digits of the start and end timestamps. The second line is the straight-through estimator: the forward pass uses the rounded digit ${ \bar { d } } _ { i }$ (the two soft terms cancel), while the backward pass routes gradients through the soft expectation $\hat { d } _ { i }$ to the logits. We map the reconstructed interval $[ \widehat { t } _ { s } , \widehat { t } _ { e } ] .$ , normalized by the video duration, to a continuous distribution $p _ { \mathrm { p r e d } }$ through a Gaussian projection: a Gaussian centered at the interval center $\mu _ { \hat { t } }$ with standard deviation $\sigma = 0 . 1 5 \cdot ( \hat { t } _ { e } - \hat { t } _ { s } )$ proportional to the predicted width. This projection is a smoothing operator, it places the prediction in the same distributional space as the attention so the two can be compared directly, and gives the interval smooth support so that its divergence from other distributions varies continuously with the temporal ofset rather than vanishing once they no longer overlap.

With both distributions in place, the core objective is their Wasserstein-1 distance, $\dot { \mathcal { L } } _ { \mathrm { t p } } ~ = ~ \mathcal { W } _ { 1 } ( p _ { \mathrm { a t t } } , \dot { p _ { \mathrm { p r e d } } } )$ . This attention–prediction edge transfers the reliable localization already present in the internal attention onto the generated timestamps, directly closing the temporal generation gap. The divergence on each edge (Jensen–Shannon or Wasserstein-1) is chosen by ablation (Appendix).

Ground-truth anchoring: two auxiliary constraints. The self-consistency edge alone does not tie either quantity to the true interval, and could in principle drive the two into mutual agreement away from it. Two auxiliary edges therefore anchor the transfer to the ground truth. We apply the same Gaussian projection, with identical parameters, to the ground-truth interval $[ t _ { s } ^ { * } , t _ { e } ^ { * } ]$ , obtaining a distribution $p _ { \mathrm { g t } }$ that is detached from the computation graph and serves as the fixed supervision target. The attention–ground-truth edge keeps the internal signal aligned with the true segment, so that it remains a trustworthy target for the core edge. The prediction–ground-truth edge supplies the distance-aware supervision that cross-entropy lacks:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { t g } } = \mathrm { J S } ( p _ { \mathrm { a t t } } , p _ { \mathrm { g t } } ) , \quad \mathcal { L } _ { \mathrm { p g } } = \mathcal { W } _ { 1 } ( p _ { \mathrm { p r e d } } , p _ { \mathrm { g t } } ) . } \end{array}\tag{3}
$$

Full objective. The three edges form a triangle of consistency constraints. With edge weights $\lambda _ { \mathrm { t p } } , \lambda _ { \mathrm { t g } } .$ and $\lambda _ { \mathrm { p g } } ,$ , the full training objective is

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { C E } } + \lambda _ { \mathrm { t p } } \mathcal { L } _ { \mathrm { t p } } + \lambda _ { \mathrm { t g } } \mathcal { L } _ { \mathrm { t g } } + \lambda _ { \mathrm { p g } } \mathcal { L } _ { \mathrm { p g } } ,\tag{4}
$$

## Data Construction and Training Strategy

After Quality Filtering removes low-resolution videos and overly short clips (grounding videos under 30 s and general clips under 2 s), we assemble the general understanding data from four balanced sources: 121k LLaVA-Video (Zhang et al. 2024) samples re-annotated by OLA (Liu et al. 2025) from speech and visual perspectives, 37k AVQA (Yang et al. 2022) samples for scene sound, 32k FortisAVQA (Ma et al. 2025) samples for music, and 200k original LLaVA-Video samples paired with audio for general video proficiency. For grounding data, we use only the training-split videos and dense temporal captions of AVID (Chen et al. 2026) and Long-VALE (Geng et al. 2025). We reformat the training captions into three complementary tasks—Temporal Video Grounding (event → time), Reverse Grounding (time → event), and Dense Video Captioning (video → timed events). We then apply Audio-Visual Relevance Filtering, using Gemini 3.1 Pro to remove QA pairs whose text does not require audio-visual evidence. A final Balanced Mixing step combines the two streams, yielding 190k grounding samples (126k TVG, 50k Reverse Grounding, and 14k DVC) and 390k general samples, for a 580k-sample dataset at a grounding-to-general ratio of roughly 1:2; the composition and pipeline are summarized in Figure 4 and detailed in the Appendix.

![](images/b145ae2c3fe5768aae7fe0368f486ce36f4a0403f6354e098a65b407e596af66.jpg)  
Figure 4: Left: dataset composition. Right: four-stage construction pipeline and representative QA templates for the three grounding formats.

We apply TACG only to TVG samples because it needs an explicit predicted time interval to construct $p _ { \mathrm { p r e d } } ,$ which is available only when the target output is a start–end timestamp. All samples are trained jointly: TVG samples use the full objective $\mathcal { L } _ { \mathrm { t o t a l } }$ , while Reverse Grounding, DVC, and general QA samples use cross-entropy $\mathcal { L } _ { \mathrm { C E } }$ only. Implementation details are provided in the Appendix.

## Experiments

## Experimental Setup

All experiments are conducted on 32 NVIDIA A100 GPUs.   
Detailed configurations are provided in the Appendix.

Evaluation Benchmarks. We evaluate on seven benchmarks: AVID (Chen et al. 2026), LongVALE (Geng et al. 2025), and UnAV-100 (Geng et al. 2023) for omni-modal grounding, Charades-STA (Gao et al. 2017) and ActivityNet-Captions (Caba Heilbron et al. 2015) for vision-only grounding, and WorldSense (Hong et al. 2026) and Daily-Omni (Zhou et al. 2025) for general omni-modal understanding. For UnAV-100, we convert its temporal annotations into QA pairs for large-model evaluation, with details provided in the Appendix.

Baselines. We compare against 17 baselines in two groups: vision-only models, including VideoChat (Li et al. 2023), VTimeLLM (Huang et al. 2024), TimeChat (Ren et al. 2024), HawkEye (Wang et al. 2024), VTG-LLM (Guo et al. 2025a), Grounded-VideoLLM (Wang et al. 2025a), TimeSuite (Zeng et al. 2025), NumPro (Wu et al. 2025), ED-VTG (Pramanick et al. 2025), and Time-R1 (Wang et al. 2025b); and omnimodal models, including VideoLLaMA2 (Cheng et al. 2024), OLA (Liu et al. 2025), Qwen2.5-Omni (Xu et al. 2025a), Qwen3-Omni (Xu et al. 2025b), LongVALE (Geng et al. 2025), ChronusOmni (Chen et al. 2025), and the closedsource Gemini 3.1 Pro (Google DeepMind 2026), which we report for reference only.

## Temporal Grounding Performance

Table 3 and Table 2 report temporal grounding results across the five grounding benchmarks. We highlight three observations.

Omni-modal temporal grounding. As shown in Table 3, OmniGround achieves the best open-model performance on the audio-visual AVID and LongVALE benchmarks. The 7B model reaches 65.9 mIoU on AVID and 37.5 mIoU on LongVALE, while the 3B model already surpasses all open baselines on AVID. Since our grounding data is reconstructed from the training-split source videos of AVID and LongVALE, we further evaluate on UnAV-100 to isolate out-of-domain audio-visual grounding performance. As shown in Table 2, OmniGround reaches 35.7 mIoU with the 7B model and outperforms evaluated open baselines at the stricter R@0.5 and R@0.7 thresholds. Together, these results show that OmniGround sets a new open-model state of the art in omni-modal temporal grounding.

<table><tr><td>Method</td><td>R@0.3</td><td>R@0.5</td><td>R@0.7</td><td>mIoU</td></tr><tr><td rowspan="4">VideoLLaMA2 OLA Qwen2.5-Omni (3B) Qwen2.5-Omni (7B)</td><td>13.3</td><td>7.0</td><td>2.3</td><td>11.5</td></tr><tr><td>13.6</td><td>6.6</td><td>2.0</td><td>13.1</td></tr><tr><td>20.7 25.9</td><td>11.8 16.2</td><td>5.5 6.7</td><td>15.2</td></tr><tr><td>26.5 11.3</td><td>17.5 6.1</td><td>8.0 3.2</td><td>20.1 21.8 15.0</td></tr><tr><td rowspan="4">ChronusOmni Gemini 3.1 Pro* OmniGround(3B)†</td><td>47.3 61.6</td><td>33.8</td><td>20.7</td><td>34.1</td></tr><tr><td></td><td>47.6</td><td>28.3</td><td>44.1</td></tr><tr><td>36.0</td><td>21.6</td><td>10.9</td><td>25.9</td></tr><tr><td>42.3</td><td>32.7</td><td>19.1</td><td>32.2</td></tr><tr><td>OmniGround(7B)† OmniGround(3B)</td><td>40.1</td><td>24.0</td><td>13.3</td><td>28.9</td></tr><tr><td>OmniGround(7B)</td><td>46.1</td><td>34.2</td><td>22.2</td><td>35.7</td></tr></table>

Table 2: Evaluation on UnAV-100. <sup>⋆</sup>Closed-source, for reference only. <sup>†</sup>Trained on the same data but with standard SFT (cross-entropy).

Vision-only temporal grounding. The benefit of TACG is not limited to the audio-visual setting. OmniGround never sees any vision-only grounding data during training, and Charades-STA and ActivityNet-Captions are outside the audio-visual grounding domains used for our data construction. Nevertheless, OmniGround remains highly competitive on these benchmarks, essentially matching specialized vision-only methods, trailing only the vision-only specialist Time-R1, and leading all other omni-modal models. This indicates that aligning predictions with internal attention strengthens temporal grounding in general, rather than only fitting the omni-modal training distribution.

<table><tr><td rowspan="3">Method</td><td colspan="7">Omni-modal Grounding</td><td colspan="8">Vision-only Grounding</td></tr><tr><td colspan="4">AVID-TVG</td><td colspan="3">LongVALE-TVG</td><td colspan="4">Charades-STA</td><td colspan="4">ActivityNet-Captions</td></tr><tr><td>R@0.3 R@0.5 R@0.7 mIoU|R@0.3 R@0.5 R@0.7 mIoU|R@0.3 R@0.5 R@0.7 mIoU|R@0.3 R@0.5 R@0.7 mIoU</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Vision-only Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VideoChat</td><td>7.6</td><td>5.2</td><td>2.5 5.2</td><td></td><td>2.2</td><td>0.9</td><td>0.4 3.0</td><td>32.8</td><td>8.6</td><td>0.0</td><td>25.9</td><td>23.5</td><td>12.6</td><td>6.0</td><td>17.4</td></tr><tr><td>VTimeLLM</td><td>9.2</td><td>4.1</td><td>1.0 7.4</td><td></td><td>9.1</td><td>4.3</td><td>1.9 7.9</td><td>52.0</td><td>27.7</td><td>11.1</td><td>31.5</td><td>44.0</td><td>27.8</td><td>14.3</td><td>30.4</td></tr><tr><td>TimeChat</td><td>9.8</td><td>6.9</td><td>3.0</td><td>7.8</td><td>7.2</td><td>3.4</td><td>1.6 6.4</td><td>47.8</td><td>31.6</td><td>14.6</td><td>31.6</td><td>30.2</td><td>16.9</td><td>8.2</td><td>21.8</td></tr><tr><td>HawkEye</td><td>7.5</td><td>4.1</td><td>1.7 5.2</td><td></td><td>15.3</td><td>7.8</td><td>2.8 10.9</td><td>50.6</td><td>31.4</td><td>14.5</td><td>33.7</td><td>49.1</td><td>29.3</td><td>10.7</td><td>32.7</td></tr><tr><td>VTG-LLM</td><td>9.0</td><td>6.4 2.9</td><td>5.9</td><td></td><td>6.2</td><td>2.8 1.0</td><td>4.2</td><td>54.0</td><td>40.0</td><td>22.0</td><td>38.1</td><td>11.3</td><td>6.8</td><td>3.0</td><td>8.0</td></tr><tr><td>Grounded-VideoLLM</td><td>12.0</td><td>8.5</td><td>5.6 9.6</td><td></td><td>20.2</td><td>12.8 7.3</td><td>15.8</td><td>54.2</td><td>36.4</td><td>19.7</td><td>36.8</td><td>46.2</td><td>30.3</td><td>19.0</td><td>36.1</td></tr><tr><td>TimeSuite</td><td>1.9</td><td>0.3</td><td>0.1 2.6</td><td></td><td>17.2</td><td>8.7 4.1</td><td>11.8</td><td>69.9</td><td>48.7</td><td>24.0</td><td>45.5</td><td>31.0</td><td>16.9</td><td>9.8</td><td>23.6</td></tr><tr><td>NumPro</td><td>5.2</td><td>2.8 0.8</td><td>4.8</td><td>10.4</td><td>4.7</td><td>2.6</td><td>8.1</td><td>63.8</td><td>42.0</td><td>20.6</td><td>41.4</td><td>55.6</td><td>37.5</td><td>20.6</td><td>38.8</td></tr><tr><td>ED-VTG</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>59.5</td><td>39.3</td><td>19.8</td><td>40.2</td><td>52.1</td><td>33.1</td><td>16.0</td><td>35.2</td></tr><tr><td>Time-R1</td><td>5.7</td><td>1.9</td><td>0.8 5.5</td><td>28.4</td><td>15.4</td><td>7.8</td><td>20.7</td><td>78.3</td><td>61.2</td><td>34.9</td><td>53.7</td><td>58.1</td><td>39.0</td><td>21.4</td><td>40.5</td></tr><tr><td>Omni-modal Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VideoLLaMA2</td><td>5.1</td><td>1.6</td><td>0.9 3.3</td><td></td><td>3.3</td><td>1.5 0.0</td><td>2.4</td><td>19.4</td><td>11.5</td><td>5.0</td><td>12.9</td><td>13.0</td><td>7.1</td><td>2.8</td><td>10.2</td></tr><tr><td>OLA</td><td>14.9</td><td>9.9 5.3</td><td>10.1</td><td>9.6</td><td>4.2</td><td>1.9</td><td>7.5</td><td>32.2</td><td>10.9</td><td>3.2</td><td>20.0</td><td>27.9</td><td>15.6</td><td>7.0</td><td>20.6</td></tr><tr><td>Qwen2.5-Omni (3B)</td><td>11.5</td><td>7.0 2.9</td><td>8.4</td><td>8.6</td><td>4.6</td><td>2.0</td><td>6.1</td><td>47.4</td><td>30.9</td><td>14.4</td><td>32.2</td><td>27.5</td><td>15.7</td><td>8.5</td><td>19.9</td></tr><tr><td>Qwen2.5-Omni (7B)</td><td>8.4</td><td>3.9 1.2</td><td>6.0</td><td>17.9</td><td>9.2</td><td>3.8</td><td>12.3</td><td>65.5</td><td>45.9</td><td>23.3</td><td>43.7</td><td>41.5</td><td>26.0</td><td>14.6</td><td>29.7</td></tr><tr><td>Qwen3-Omni</td><td>18.2</td><td>12.3 5.2</td><td>14.4</td><td>14.4</td><td>7.7</td><td>3.4</td><td>10.4</td><td>35.4</td><td>23.4</td><td>10.5</td><td>23.3</td><td>37.6</td><td>23.9</td><td>12.6</td><td>26.9</td></tr><tr><td>LongVALE</td><td>4.8</td><td>2.4 1.1</td><td>4.5</td><td>15.7</td><td>8.6</td><td>3.9</td><td>11.0</td><td>47.9</td><td>19.8</td><td>9.7</td><td>32.6</td><td>46.3</td><td>28.5</td><td>14.2</td><td>31.6</td></tr><tr><td>ChronusOmni</td><td>42.3</td><td>33.7 21.9</td><td>31.2</td><td>49.7</td><td>32.5</td><td>17.6</td><td>34.5</td><td>72.2</td><td>47.0</td><td>20.8</td><td>46.0</td><td>56.0</td><td>35.5</td><td>20.1</td><td>39.4</td></tr><tr><td>Gemini 3.1 Pro*</td><td>55.0</td><td>50.5 44.7</td><td>47.0</td><td>66.2</td><td></td><td>49.0 31.0</td><td>46.6</td><td>74.6</td><td>44.4</td><td>15.6</td><td>44.2</td><td>64.1</td><td>46.6</td><td>28.7</td><td>46.9</td></tr><tr><td>OmniGround(3B)†</td><td>67.3</td><td>61.2</td><td>48.1</td><td>55.3</td><td>40.2</td><td>27.3</td><td>15.1 27.7</td><td>50.2</td><td>20.5</td><td>8.4</td><td>32.6</td><td>31.0</td><td>19.2</td><td>11.2</td><td>23.5</td></tr><tr><td>OmniGround(7B)†</td><td>70.3</td><td>67.6</td><td>55.7</td><td>60.4</td><td>46.2</td><td>34.5</td><td>20.2 32.7</td><td>67.2</td><td>38.5</td><td>19.7</td><td>44.6</td><td>50.4</td><td>36.5</td><td>19.2</td><td></td></tr><tr><td>OmniGround(3B)</td><td>68.3</td><td>62.2</td><td>50.5</td><td>56.9</td><td>41.4</td><td>28.6 16.1</td><td>28.6</td><td>60.5</td><td>31.2</td><td>15.1</td><td>38.8</td><td>31.7</td><td>19.6</td><td></td><td>38.3</td></tr><tr><td>OmniGround(7B)</td><td>75.2</td><td>72.1</td><td>64.9</td><td>65.9</td><td>51.9</td><td>39.5 25.1</td><td>37.5</td><td>74.5</td><td>52.6</td><td>27.7</td><td>49.3</td><td>56.4</td><td>39.1</td><td>11.1 21.1</td><td>23.7 41.2</td></tr></table>

Table 3: Comparison with state-of-the-art methods on temporal video grounding benchmarks. OmniGround sets a new state of the art on the omni-modal temporal grounding; moreover, despite seeing no vision-only grounding data, it still essentially matches specialized methods on the vision-only Charades-STA and ActivityNet-Captions, reflecting strong generalization. Bold: best; underline: second best. <sup>⋆</sup>Closed-source, for reference only. <sup>†</sup>Trained on the same data but with standard SFT (cross-entropy).

![](images/bce29b977514bb7d53202b507932fc7af52ffa066da1721dd89c18767d6acaa6.jpg)  
Figure 5: TACG closes the temporal generation gap. Average ground-truth, attention, and generated intervals for OmniGround. The persistent displacement observed in the preliminary analysis has been largely absorbed.

TACG versus standard SFT. At full scale, we train otherwise identical 3B and 7B models with standard cross-entropy on the same 580k-sample dataset (OmniGround<sup>†</sup>); TACG improves grounding on nearly every benchmark at both scales. These full-scale margins are smaller than the clean gain measured in the controlled ablation (Table 5), and by design: the TACG objective activates only on the forward-TVG samples—about one-fifth of the 580k-sample mixture—so its efect is diluted at the full-model level. Even under this dilution, the improvement is consistent across the evaluated benchmarks, and on the never-seen vision-only Charades-STA it is pronounced at both scales (+6.2 mIoU at 3B), showing that aligning predictions with the internal attention instills a transferable localization skill rather than overfitting a particular distribution. Figure 5 makes this alignment visible on AVID: for both OmniGround scales, the predicted intervals become better aligned with attention and move closer to the ground-truth intervals, explaining the stronger localization performance and confirming that TACG closes the temporal generation gap it was designed to target.

## General Understanding Performance

Table 4 shows that OmniGround preserves strong general audio-visual understanding while improving temporal grounding. The 7B model remains competitive on World-Sense and achieves the best open-model result on Daily-Omni, indicating that grounding-oriented training does not come at the cost of general comprehension. Relative to the same-scale cross-entropy baselines, TACG keeps World-Sense stable and further improves Daily-Omni, suggesting that sharper temporal alignment can also benefit tasks involving event order and audio-visual correspondence. By contrast, the grounding-specialized LongVALE performs poorly on these general benchmarks, highlighting the advantage of OmniGround’s balanced training.

## Ablation Study

We conduct ablation experiments to analyze the key design choices in TACG. All ablations are trained on the same set of

<table><tr><td>Method</td><td>WorldSense Acc</td><td>Daily-Omni Acc</td></tr><tr><td>VideoLLaMA2</td><td>27.8</td><td>35.8</td></tr><tr><td>OLA</td><td>34.0</td><td>54.3</td></tr><tr><td>Qwen2.5-Omni (3B)</td><td>46.0</td><td>62.3</td></tr><tr><td>Qwen2.5-Omni (7B)</td><td>46.1</td><td>62.7</td></tr><tr><td>Qwen3-Omni</td><td>52.0</td><td>68.8</td></tr><tr><td>LongVALE</td><td>9.9</td><td>24.4</td></tr><tr><td>ChronusOmni</td><td>49.5</td><td>70.2</td></tr><tr><td>Gemini 3.1 Pro*</td><td>63.1</td><td>81.0</td></tr><tr><td>OmniGround (3B)†</td><td>48.9</td><td>65.4</td></tr><tr><td>OmniGround (7B)†</td><td>48.3</td><td>68.7</td></tr><tr><td>OmniGround (3B)</td><td>48.6</td><td>68.1</td></tr><tr><td>OmniGround (7B)</td><td>50.2</td><td>71.9</td></tr></table>

Table 4: Evaluation on general audio-visual understanding benchmarks. <sup>⋆</sup>Closed-source, for reference only. <sup>†</sup>Trained on the same data but with standard SFT.

<table><tr><td rowspan="2">Configuration</td><td colspan="2">AVID</td><td colspan="2">LongVALE</td><td rowspan="2">∆%</td></tr><tr><td>R@0.5</td><td>mIoU</td><td>R@0.5</td><td>mIoU</td></tr><tr><td>Baseline (CE only)</td><td>0.3502</td><td>0.3153</td><td>0.1408</td><td>0.1604</td><td>0.00</td></tr><tr><td> $\mathcal { L } _ { \mathrm { t g } }$  only</td><td>0.3543</td><td>0.3223</td><td>0.1563</td><td>0.1722</td><td>+4.79</td></tr><tr><td> $\mathcal { L } _ { \mathrm { p g } }$  only</td><td>0.3649</td><td>0.3282</td><td>0.1479</td><td>0.1645</td><td>+3.32</td></tr><tr><td> ${ \mathcal { L } } _ { \mathrm { t p } }$  only</td><td>0.3671</td><td>0.3315</td><td>0.1514</td><td>0.1663</td><td>+4.41</td></tr><tr><td> $\mathcal { L } _ { \mathrm { t g } } + \mathcal { L } _ { \mathrm { p g } }$ </td><td>0.3671</td><td>0.3301</td><td>0.1508</td><td>0.1672</td><td>+4.47</td></tr><tr><td>Full TACG</td><td>0.3814</td><td>0.3408</td><td>0.1546</td><td>0.1759</td><td>+8.87</td></tr></table>

Table 5: Component ablation of the TACG consistency loss. ∆% is the average relative improvement over the SFT baseline across both benchmarks. Bold: best; underline: second best. Full results are provided in the appendix.

10k TVG samples, with an SFT model trained on identica data as the baseline, so the comparison isolates the gains brought by TACG rather than data diferences.

## Component Ablation

We evaluate the contribution of each consistency edge in Eq. 4 $( \mathcal { L } _ { \mathrm { t g } } , \mathcal { L } _ { \mathrm { p g } }$ , and ${ \mathcal { L } } _ { \mathfrak { t p } } )$ . Each configuration is reported under its individually optimized weight, so the comparison reflects each edge’s best achievable efect.

As shown in Table 5, each single edge improves over the SFT baseline under its own optimal weight (+3.32% to +4.79%), indicating that all three consistency signals are useful on their own. Stacking the two ground-truth-anchored edges $( \mathcal { L } _ { \mathrm { t g } } + \mathcal { L } _ { \mathrm { p g } } )$ yields only +4.47%—essentially no gain over either edge alone. In sharp contrast, completing the triangle with the attention-to-prediction edge ${ \mathcal { L } } _ { \mathrm { t p } }$ lifts the gain to +8.87%, nearly doubling the best two-edge result. This identifies attention–prediction consistency as the core component of TACG: the largest gain appears precisely when the objective addresses the temporal generation gap diagnosed in our preliminary analysis.

![](images/36dadb15257729e297f91148eb670ebda646f11edd13b9f1292ad01ff5a4c0f4.jpg)  
(a) Argmax loss

![](images/9a115f50fd51b7221b96c998ff7e7c35250d81d9d4ee8c5f7a9631e4d58f0216.jpg)  
(b) Round loss  
Figure 6: Training dynamics for timestamp discretization. Argmax spikes mid-training from mode collapse, whereas Round decreases smoothly and monotonically.

## Discretization of the Predicted Distribution

Constructing the predicted distribution requires mapping the per-position digit logits to a digit value $\tilde { d } _ { i }$ , which is then assembled into a timestamp by place value. Two conventional choices both fail for timestamp regression: Argmax keeps only the most probable digit, so it cannot distinguish a nearly-correct distribution from a confidently-wrong one and its selected digit jumps discontinuously between steps, yielding unstable gradients; Soft uses the continuous expectation for smooth gradients, but the expectation of a discrete distribution rarely lands on an integer, leaving a loss floor that, amplified by place value, biases the gradient at high-order positions.

We therefore adopt Round: we first take the soft expectation and then round it to the nearest digit. This combines smooth gradients with discrete semantics, avoiding both Argmax’s instability and Soft’s vanishing-gradient floor. The metrics confirm these failure modes: both Argmax and Soft fall below the CE baseline at their best settings, while Round is the only choice that yields a stable improvement (Appendix Figure 10). The training dynamics tell the same story (Figure 6): Argmax shows a sharp mid-training spike from mode collapse, whereas Round decreases smoothly and monotonically. The per-λ sensitivity curves are reported in the appendix.

## Conclusion

We identified the temporal generation gap in omni-modal language models, where internal attention localizes the target moment more accurately than the generated timestamps, and proposed Temporal Attention-Consistency Grounding (TACG), which couples attention, prediction, and ground truth into a triangle of consistency constraints that strengthens temporal localization with no added parameters, special tokens, or inference overhead. We further constructed a 580ksample training dataset covering both grounding and general omni-modal understanding. Built on TACG and this dataset, OmniGround reaches state-of-the-art omni-modal temporal grounding while preserving strong general comprehension. More broadly, our results show that a model’s internal attention is a valuable and underexploited supervision signal for temporal precision; we will release the dataset, model, and training framework to support future work.

## References

Caba Heilbron, F.; Escorcia, V.; Ghanem, B.; and Carlos Niebles, J. 2015. ActivityNet: A Large-Scale Video Benchmark for Human Activity Understanding. In CVPR.

Chen, Y.; Wu, Y.; Guan, K.; Ren, Y.; Wang, Y.; Song, R.; and Ru, L. 2025. ChronusOmni: Improving Time Awareness of Omni Large Language Models. arXiv preprint arXiv:2512.09841.

Chen, Z.; Wang, D.; Lin, H.; Luo, L.; Xu, K.; Guo, Y.; Zhu, H.; Sun, T.; and Jiang, X. 2026. AVID: A Benchmark for Omni-Modal Audio-Visual Inconsistency Understanding via Agent-Driven Construction. arXiv preprint arXiv:2604.13593.

Cheng, Z.; Leng, S.; Zhang, H.; Xin, Y.; Li, X.; Chen, G.; Zhu, Y.; Zhang, W.; Luo, Z.; Zhao, D.; and Bing, L. 2024. VideoLLaMA 2: Advancing Spatial-Temporal Modeling and Audio Understanding in Video-LLMs. arXivpreprint arXiv:2406.07476.

Gao, J.; Sun, C.; Yang, Z.; and Nevatia, R. 2017. TALL: Temporal Activity Localization via Language Query. In ICCV. Charades-STA benchmark.

Geng, T.; Wang, T.; Duan, J.; Cong, R.; and Zheng, F. 2023. Dense-Localizing Audio-Visual Events in Untrimmed Videos: A Large-Scale Benchmark and Baseline. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 22942–22951.

Geng, T.; Zhang, J.; Wang, Q.; Wang, T.; Duan, J.; and Zheng, F. 2025. LongVALE: Vision-Audio-Language-Event Benchmark Towards Time-Aware Omni-Modal Perception of Long Videos. In CVPR.

Google DeepMind. 2026. Gemini 3.1 Pro Model Card. Model card.

Guo, Y.; Liu, J.; Li, M.; Cheng, D.; Tang, X.; Sui, D.; Liu, Q.; Chen, X.; and Zhao, K. 2025a. VTG-LLM: Integrating Timestamp Knowledge into Video LLMs for Enhanced Video Temporal Grounding. In AAAI.

Guo, Y.; Liu, J.; Li, M.; Liu, Q.; Chen, X.; and Tang, X. 2025b. TRACE: Temporal Grounding Video LLM via Causal Event Modeling. In The Thirteenth International Conference on Learning Representations (ICLR).

Hannan, T.; Islam, M. M.; Gu, J.; Seidl, T.; and Bertasius, G. 2025. ReVisionLLM: Recursive Vision-Language Model for Temporal Grounding in Hour-Long Videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 19012–19022.

Hong, J.; Yan, S.; Cai, J.; Jiang, X.; Hu, Y.; and Xie, W. 2026. WorldSense: Evaluating Real-world Omnimodal Understanding for Multimodal LLMs. In International Conference on Learning Representations (ICLR). ArXiv preprint arXiv:2502.04326.

Huang, B.; Wang, X.; Chen, H.; Song, Z.; and Zhu, W. 2024. VTimeLLM: Empower LLM to Grasp Video Moments. In CVPR.

Jang, J.; Park, J.; Kim, J.; Kwon, H.; and Sohn, K. 2023. Knowing Where to Focus: Event-aware Transformer for

Video Grounding. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 13846– 13856.

Li, J.; Yin, H.; Xu, H.; Xu, B.; Tan, W.; He, Z.; Ju, J.; Luo, Z.; and Luan, J. 2026a. Video-OPD: Eficient Post-Training of Multimodal Large Language Models for Temporal Video Grounding via On-Policy Distillation. arXiv preprint arXiv:2602.02994.

Li, K.; He, Y.; Wang, Y.; Li, Y.; Wang, W.; Luo, P.; Wang, Y.; Wang, L.; and Qiao, Y. 2023. VideoChat: Chat-Centric Video Understanding. arXiv preprint arXiv:2305.06355.

Li, X.; Wang, Y.; Yu, J.; Zeng, X.; Zhu, Y.; Huang, H.; Gao, J.; Li, K.; He, Y.; Wang, C.; Qiao, Y.; Wang, Y.; and Wang, L. 2026b. VideoChat-Flash: Hierarchical Compression for Long-Context Video Modeling. In The Fourteenth International Conference on Learning Representations (ICLR).

Lin, K. Q.; Zhang, P.; Chen, J.; Pramanick, S.; Gao, D.; Wang, A. J.; Yan, R.; and Shou, M. Z. 2023. UniVTG: Towards Unified Video-Language Temporal Grounding. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2794–2804.

Liu, Z.; Dong, Y.; Wang, J.; Liu, Z.; Hu, W.; Lu, J.; and Rao, Y. 2025. Ola: Pushing the Frontiers of Omni-Modal Language Model. arXiv preprint arXiv:2502.04328.

Ma, J.; Gao, Z.; Chai, Q.; Liu, J.; Wang, P.; Tao, J.; and Su, Z. 2025. FortisAVQA and MAVEN: A Benchmark Dataset and Debiasing Framework for Robust Multimodal Reasoning. arXiv preprint arXiv:2504.00487.

Maaz, M.; Rasheed, H.; Khan, S.; and Khan, F. 2024. Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 12585–12602. Association for Computational Linguistics.

Mu, F.; Mo, S.; and Li, Y. 2024. SnAG: Scalable and Accurate Video Grounding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 18930–18940.

Palit, V.; Pandey, R.; Arora, A.; and Liang, P. P. 2023. Towards Vision-Language Mechanistic Interpretability: A Causal Tracing Tool for BLIP. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops, 2856–2861.

Pramanick, S.; Mavroudi, E.; Song, Y.; Chellappa, R.; Torresani, L.; and Afouras, T. 2025. Enrich and Detect: Video Temporal Grounding with Multimodal LLMs. In ICCV.

Qu, M.; Chen, X.; Liu, W.; Li, A.; and Zhao, Y. 2024. ChatVTG: Video Temporal Grounding via Chat with Video Dialogue Large Language Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 1847–1856.

Ren, S.; Yao, L.; Li, S.; Sun, X.; and Hou, L. 2024. TimeChat: A Time-sensitive Multimodal Large Language Model for Long Video Understanding. In CVPR.

Su, Y.; Lan, T.; Li, H.; Xu, J.; Wang, Y.; and Cai, D. 2023. PandaGPT: One Model To Instruction-Follow Them All. In

Proceedings of the 1st Workshop on Taming Large Language Models: Controllability in the Era ofInteractive Assistants!, 11–23. Association for Computational Linguistics.

Wang, H.; Xu, Z.; Cheng, Y.; Diao, S.; Zhou, Y.; Cao, Y.; Wang, Q.; Ge, W.; and Huang, L. 2025a. Grounded-VideoLLM: Sharpening Fine-grained Temporal Grounding in Video Large Language Models. In EMNLP Findings.

Wang, Y.; Meng, X.; Liang, J.; Wang, Y.; Liu, Q.; and Zhao, D. 2024. HawkEye: Training Video-Text LLMs for Grounding Text in Videos. arXiv preprint arXiv:2403.10228.

Wang, Y.; Wang, Z.; Xu, B.; Du, Y.; Lin, K.; Xiao, Z.; Yue, Z.; Ju, J.; Zhang, L.; Yang, D.; Fang, X.; He, Z.; Luo, Z.; Wang, W.; Lin, J.; Luan, J.; and Jin, Q. 2025b. Time-R1: Post-Training Large Vision Language Model for Temporal Video Grounding. In NeurIPS.

Wu, S.; Fei, H.; Qu, L.; Ji, W.; and Chua, T.-S. 2024. NExT-GPT: Any-to-Any Multimodal LLM. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, 53366– 53397. PMLR.

Wu, Y.; Hu, X.; Sun, Y.; Zhou, Y.; Zhu, W.; Rao, F.; Schiele, B.; and Yang, X. 2025. Number it: Temporal Grounding Videos like Flipping Manga. In CVPR.

Xu, J.; Guo, Z.; He, J.; Hu, H.; He, T.; Bai, S.; Chen, K.; Wang, J.; Fan, Y.; Dang, K.; Zhang, B.; Wang, X.; Chu, Y.; and Lin, J. 2025a. Qwen2.5-Omni Technical Report. arXiv preprint arXiv:2503.20215.

Xu, J.; Guo, Z.; Hu, H.; Chu, Y.; Wang, X.; He, J.; Wang, Y.; Shi, X.; He, T.; Zhu, X.; Lv, Y.; Wang, Y.; Guo, D.; Wang, H.; Ma, L.; Zhang, P.; Zhang, X.; Hao, H.; Guo, Z.; Yang, B.; Zhang, B.; Ma, Z.; Wei, X.; Bai, S.; Chen, K.; Liu, X.; Wang, P.; Yang, M.; Liu, D.; Ren, X.; Zheng, B.; Men, R.; Zhou, F.; Yu, B.; Yang, J.; Yu, L.; Zhou, J.; and Lin, J. 2025b. Qwen3-Omni Technical Report. arXiv preprint arXiv:2509.17765.

Yang, P.; Wang, X.; Duan, X.; Chen, H.; Hou, R.; Jin, C.; and Zhu, W. 2022. AVQA: A Dataset for Audio-Visual Question Answering on Videos. In ACM MM.

Zeng, X.; Li, K.; Wang, C.; Li, X.; Jiang, T.; Yan, Z.; Li, S.; Shi, Y.; Yue, Z.; Wang, Y.; Wang, Y.; Qiao, Y.; and Wang, L. 2025. TimeSuite: Improving MLLMs for Long Video Understanding via Grounded Tuning. In ICLR.

Zhang, H.; Li, X.; Bing, Z., Lidong; Leng, S.; Zhang, H.; Xin, Y.; and Li. 2023. Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, 543–553. Association for Computational Linguistics.

Zhang, S.; Peng, H.; Fu, J.; and Luo, J. 2020. Learning 2D Temporal Adjacent Networks for Moment Localization with Natural Language. In AAAI.

Zhang, Y.; Wu, J.; Li, W.; Li, B.; Ma, Z.; Liu, Z.; and Li, C. 2024. Video Instruction Tuning with Synthetic Data. arXiv preprint arXiv:2410.02713.

Zhao, Q.; Xu, M.; Gupta, K.; Asthana, A.; Zheng, L.; and Gould, S. 2024. The First to Know: How Token Distributions Reveal Hidden Knowledge in Large Vision-Language Models? arXiv preprint arXiv:2403.09037.

Zhou, Z.; Wang, R.; Wu, Z.; and Jiang, Y.-G. 2025. Daily-Omni: Towards Audio-Visual Reasoning with Temporal Alignment across Modalities. arXiv preprint arXiv:2505.17862.

## Technical Appendix

This appendix provides additional details omitted from the main text due to space constraints. It is organized into three parts: (i) supporting material for the preliminary analysis of the temporal generation gap (divergence definitions and perlayer attention analysis); (ii) model architecture and training details; and (iii) additional ablations and full experimental results.

## Divergence Definitions

For two distributions p, q over the T time steps, the Jensen– Shannon divergence and the Wasserstein-1 distance used in the consistency loss (Eq. 4) are

$$
\begin{array} { r l } & { \mathrm { J S } ( p , q ) = \frac { 1 } { 2 } \mathrm { K L } ( p \| m ) + \frac { 1 } { 2 } \mathrm { K L } ( q \| m ) , \quad m = \frac { 1 } { 2 } ( p + q ) , } \\ & { \mathcal { W } _ { 1 } ( p , q ) = \cfrac { 1 } { T } \displaystyle \sum _ { t = 1 } ^ { T } \Big | \sum _ { \tau \leq t } ( p _ { \tau } - q _ { \tau } ) \Big | , } \end{array}\tag{5}
$$

where KL is the Kullback–Leibler divergence and the second expression is the $\ell _ { 1 }$ distance between the cumulative distributions, the closed form of the Wasserstein-1 distance in one dimension.

## Per-Layer Attention Analysis

We obtain each layer’s audio-visual temporal attention distribution from the last prefill token. We take its attention weights normalized over all video and audio tokens, assign both modalities to their corresponding interleaved temporal grids, sum the attention mass within each grid, and average over heads, yielding the distribution $p _ { \mathrm { a t t } }$ over the T media grids. To make distributions comparable across clips of different length and segment position, we re-align every distribution so that the ground-truth segment is centered at the origin and rescaled to unit width, then average within each layer.

For each benchmark, the diagnostic evaluation split consists of 500 examples randomly sampled from the oficial test split. All preliminary-analysis statistics in Table 1, including Attn. wins and prediction mIoU, are computed on these fixed benchmark-specific subsets; ties are excluded only from the denominator of Attn. wins. The layer-wise visualization and sensitivity analysis use the same sampling protocol.

Lead rate versus aggregate mIoU. These two statistics characterize diferent aspects of localization quality. Lead rate measures how frequently attention is closer to the ground truth than the generated prediction at the instance level, whereas aggregate mIoU is also sensitive to the magnitude of a smaller number of large wins. Consequently, attention can lead on more non-tied samples while having a similar or slightly lower aggregate mIoU, as observed for the CE baseline. Our claim is therefore not that attention is uniformly superior on every sample, but that it provides complementary temporal evidence on a substantial fraction of instances. TACG exploits this evidence while anchoring both attention and prediction to the ground truth.

We visualize all 36 transformer layers to examine how the audio-visual temporal-localization signal evolves with model depth. As Figure 7 shows, layers 27–28 exhibit both sharp attention concentration inside the ground-truth segment and substantial sharpening from supervised fine-tuning, whereas earlier layers (e.g. layer 21) are less discriminative and the final layer 35 is difuse. Figure 8 corroborates this across all 36 layers at once: in both the base and SFT models the attention mass concentrates inside the ground-truth band at essentially every layer, showing that the localization signal is broad rather than confined to a single layer. Throughout our experiments, we use layer 27 of Qwen2.5-Omni-3B (36 layers), layer 22 of Qwen2.5-Omni-7B, layer 32 of the Qwen3- Omni thinker (48 layers), and layer 20 of OLA (28 layers). For TACG training, the fixed attention layer is layer 27 for the 3B model and layer 22 for the 7B model, as reported in Table 8. The absolute index difers across models only because their depths do, not the underlying phenomenon.

<table><tr><td>Variant</td><td>Group</td><td>Attn. mIoU</td><td>Pred. mIoU</td><td>∆IoU</td></tr><tr><td rowspan="4">CE Base</td><td>Attn. wins</td><td>40.0</td><td>5.9</td><td>+34.1</td></tr><tr><td>Pred. wins</td><td>37.9</td><td>68.5</td><td>-30.6</td></tr><tr><td>Ties</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Overall</td><td>29.8</td><td>31.1</td><td>-1.3</td></tr><tr><td rowspan="4">Full TACG</td><td>Attn. wins</td><td>50.0</td><td>8.0</td><td>+42.0</td></tr><tr><td>Pred. wins</td><td>41.1</td><td>74.0</td><td>-32.9</td></tr><tr><td>Ties</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Overall</td><td>34.6</td><td>34.1</td><td>+0.5</td></tr></table>

Table 6: Attention–prediction alignment of the CE baseline and Full TACG on AVID. All values are reported on the percentage scale. Full TACG improves both localization signals and reduces the aggregate gap between attention and prediction.

We further test whether the temporal generation gap depends on the attention-extraction hyperparameters. Figure 9 varies the attention layer and the cumulative-mass cutof used to convert attention into an interval. Across both sweeps, attention still overlaps the ground truth better than the generated interval on most non-tie samples, showing that the gap is not an artifact of a single layer or the default 50% cutof.

## Architecture Details

The vision encoder processes input frames at native resolution. Each frame is divided into 14 × 14 pixel patches. A temporal patch size of 2 groups two consecutive frames into a single temporal unit, reducing the temporal token count by half. After ViT encoding, a spatial merge with factor 2 further compresses spatial tokens by merging 2 × 2 adjacent spatial patches, resulting in a 4× reduction in spatial token count.

Audio Encoder. Raw audio waveforms are resampled to 16 kHz and converted to 128-bin mel-spectrograms. The Whisper-style encoder uses 32 transformer layers with 20 attention heads and a hidden dimension of 1280. A linear projection maps the output to 2048 dimensions, matching the language model hidden size. Audio longer than 30 seconds is segmented and encoded in parallel, with outputs concatenated along the sequence dimension.

![](images/e0834bbaaf3750824aab40d827d551db1d572cd0d9619551450ede782797d1ff.jpg)

![](images/799b0cb5b768d80092e2de506c8ecb217252ec98d87bdb53574206049f06ba8c.jpg)

![](images/a1ba6fa2607335ae65a4ba3962863289b064f71aeb12d30851432dc0e1a9f95c.jpg)

![](images/4c1930a7f4c0afe7d3b2b6c05bc3841c012ecba488a11f5343a0da07aebe5481.jpg)  
Figure 7: GT-aligned attention at representative layers for the SFT and base models (N=500); the green band marks the ground truth segment.

![](images/ee91a103c0c9fceed21f243a01aa53a3ab468cbca5cc9afcb62c4da1a6f00b30.jpg)

![](images/5859cfd47dc26dfdf1b79e41384fcc7438d9a965ff6934ce5555d13613c85883.jpg)

![](images/45bc93a171e0beea64e930220f0f171924bf055333c52a4eb012a4680b9d8c72.jpg)  
Figure 8: GT-aligned temporal attention across all 36 layers $( N { = } 5 0 0 )$ , for the SFT model (left), the base model (middle), and their diference (right). Each row is a layer; the horizontal axis is the ofset from the ground-truth center (GT width normalized to 1; green dashed lines mark the segment). Attention concentrates inside the GT band at nearly every layer and is sharpest around layers 26–28, and the SFT−base diference peaks inside the band at the same layers—fine-tuning sharpens these mid-to-late layers the most.

Chunk-Interleaved Input Construction. For a video of duration D seconds:

• If $D \leq 6 0 \mathrm { s } ;$ : the video is encoded with $\lfloor D / 2 \rfloor$ temporal grids, each covering exactly 2 seconds. Audio tokens are split into chunks of fixed size matching the 2-second interval. Video and audio chunks are interleaved in chronological order.

• If $D > 6 0 \mathrm { s } ;$ video frames are uniformly sampled and encoded into at most 30 temporal grids. Each grid covers D/30 seconds. Audio tokens are uniformly redistributed into 30 chunks of equal size, ensuring each audio chunk aligns temporally with its corresponding video grid.

The interleaved sequence is wrapped with special audio start and end tokens. Within each chunk, visual tokens appear first, followed by audio tokens. The 3D M-RoPE assigns temporal, height, and width position indices to visual tokens, while audio tokens receive sequential position indices synchronized with the temporal position of their chunk.

## Training Data Details

Quality Filtering. Before reformatting, we remove lowresolution videos and overly short clips: grounding videos shorter than 30 s and general-understanding clips shorter than 2 s are discarded. General-understanding data is then sampled from four sources so that no single audio type dominates: OLA-reannotated LLaVA-Video for paired speech and visual reasoning, original LLaVA-Video for general video proficiency with audio, AVQA for scene sound, and FortisAVQA for music.

QA Construction. AVID and LongVALE provide source videos with dense caption-style temporal annotations rather than the final QA pairs used for our training. We use only their training-split source videos and temporal captions, then reconstruct new plain-text QA instances in three complementary grounding formats. We preserve the oficial benchmark separation: videos, temporal annotations, and captions from the held-out evaluation splits are excluded from all construction, filtering, and training stages. Forward TVG maps an event description to its interval—“Locate the event where {description}.” → “[start\_sec, end\_sec]”; Reverse Grounding inverts this, mapping a queried time interval to the event that occurs there; and Dense Video Captioning asks the model to enumerate all timed events in a clip—“[start, end]: {event}” lines in chronological order. This trains a single model to move in both directions between language and time rather than only emitting timestamps.

![](images/5c3abaabe812c8bf8d8fcf77d563a179093b417c8a4dee31243e701783b9a272.jpg)  
Figure 9: Sensitivity of the temporal generation gap to attention extraction choices on Qwen2.5-Omni-3B (N=500). Left: layer sweep. Right: cumulative-mass cutof sweep for extracting $I _ { \mathrm { a t t } }$

Audio-Visual Relevance Filtering and Mixing. After reformatting we apply an audio-visual relevance filter at the individual caption/query level, retaining targets that describe audio-visual events and removing text targets that are audio-only, visual-only, or ambiguous. For TVG and Reverse Grounding, the target is the queried event caption; for DVC, each timed event caption in the same QA instance is judged independently. The filter is text-only: Gemini 3.1 Pro receives only the event caption or QA text as input, labels the target as audio-visual, audio-only, visual-only, or ambiguous, and retains only audio-visual cases. Rejection rates are therefore computed over individual caption/query targets rather than whole videos. The filter is applied only to training-split samples and never sees held-out evaluation videos, annotations, or captions. A final balanced-mixing step combines the two streams into a 580,676-sample dataset at a groundingto-general ratio of roughly 1:2, which we found preserves general audio-visual understanding while providing enough grounding supervision for the TACG objective to take efect.

## Training Hyperparameters

We fine-tune Qwen2.5-Omni-3B and Qwen2.5-Omni-7B in a single stage with AdamW and a cosine schedule, freezing the vision encoder and audio aligner so that training adapts the language model and cross-modal fusion while preserving the pretrained perceptual front-ends. Both scales use the same optimization and TACG configurations except for the attention layer specified in Table 8. The TACG objective is applied selectively—all forward-TVG samples, which emit an explicit interval and thus admit a predicted distribution, use the full consistency loss, while all other samples use cross-entropy alone. The three edge weights $( \lambda _ { \mathrm { t p } } , \dot { \lambda } _ { \mathrm { t g } } , \lambda _ { \mathrm { p g } } )$

<table><tr><td>Component</td><td>Specification</td></tr><tr><td>Vision Encoder</td><td></td></tr><tr><td>Architecture</td><td>Vision Transformer (ViT)</td></tr><tr><td>Hidden size</td><td>1280</td></tr><tr><td>Patch size</td><td>14 × 14</td></tr><tr><td>Temporal patch size</td><td>2</td></tr><tr><td>Spatial merge size</td><td>2</td></tr><tr><td>Input channels</td><td>3</td></tr><tr><td>Audio Encoder</td><td></td></tr><tr><td>Architecture</td><td>Whisper-style Transformer</td></tr><tr><td>Hidden size</td><td>1280</td></tr><tr><td>Output dimension</td><td>2048</td></tr><tr><td>Number of layers</td><td>32</td></tr><tr><td>Attention heads</td><td>20</td></tr><tr><td>Mel bins</td><td>128</td></tr><tr><td></td><td></td></tr><tr><td>Language Model Architecture</td><td>Transformer (autoregressive)</td></tr><tr><td>Hidden size</td><td></td></tr><tr><td>Number of layers</td><td>2048</td></tr><tr><td></td><td>36</td></tr><tr><td>Attention heads</td><td>16</td></tr><tr><td>Key-value heads</td><td>2 (grouped-query attention)</td></tr><tr><td>Intermediate size</td><td>11008</td></tr><tr><td>Vocabulary size</td><td>151936</td></tr><tr><td>Position encoding</td><td>3D M-RoPE (T, H, W)</td></tr><tr><td>Chunk Configuration</td><td></td></tr><tr><td>Seconds per chunk</td><td>2</td></tr><tr><td></td><td></td></tr><tr><td>Max temporal grids</td><td>30</td></tr><tr><td>Long video threshold</td><td>60 seconds</td></tr></table>

Table 7: Detailed architecture specifications of OmniGround.

are the efective values used by the final models.

Computing Infrastructure and Software. The full computing environment is listed in Table 11. Main-text results use a fixed random seed (42); multi-seed mean±std for the key results is reported in the Multi-Seed Stability section (Tables 13 and 14). Code, training data, and model checkpoints will be released publicly upon publication.

![](images/9971a78c87b0b7ef34401a72a7014b87db46a955d149efe296262bb7df46db20.jpg)  
Figure 10: Best mIoU per discretization against the crossentropy baseline. Only Round yields a positive gain over the CE baseline; Argmax and Soft fall below it on both benchmarks.

## Evaluation Protocol and Baseline Reporting

All grounding results are evaluated under the oficial protocol of each benchmark. For AVID and LongVALE, we use the released test splits and compute R@0.3, R@0.5, R@0.7, and mIoU against the annotated temporal intervals; for the constructed UnAV-100 TVG subset, we use the same metrics on the filtered single-interval test instances described below. For Charades-STA and ActivityNet-Captions, we follow the standard temporal grounding evaluation protocol used by prior video-LLM grounding work. General audio-visual understanding results on WorldSense and Daily-Omni are reported with the oficial accuracy metric.

![](images/d151dd2c3ee7122e7c5cc5c10e29762f3e0dff992ddb9a97c34edead22297eaf.jpg)

<table><tr><td>Hyperparameter</td><td>|Value</td></tr><tr><td>Base model Optimizer Learning rate LR scheduler Warmup ratio Batch size (per device) Gradient accumulation Number of epochs Max sequence length DeepSpeed Gradient checkpointing Attention implementation Precision Frozen modules</td><td>Qwen2.5-Omni-3B / 7B AdamW 2.5e-5 Cosine 0.05 1 2 1 16,384 ZeRO-2 Enabled FlashAttention BF16 ViT, audio aligner</td></tr><tr><td colspan="2">TACG Configuration  $\lambda _ { \mathfrak { t } }$  (attn-pred, W1) 0.0500  $\lambda _ { \mathrm { t g } }$  (attn–GT, JS) 0.0125  $\lambda _ { \mathrm { p g } }$  (pred–GT, W1) 0.1500 Attention layer l 27 (3B) / 22 (7B) Gaussian σ 0.15 × segment width</td></tr></table>

Table 8: Training hyperparameters and TACG configuration.

![](images/1b7905239ff2b6e9f23596fceb5e188b76f56d3f6661cdc22ef12a560e234aa5.jpg)

![](images/e0649fe12a0c25a39389901a7f75261246d4f2b62c6134bba7177bf7502635a5.jpg)  
Figure 11: mIoU vs. loss weight λ for each discretization.  
Figure 12: Weight ablation for all three edges $( \mathcal { L } _ { \mathrm { t g } } , \mathcal { L } _ { \mathrm { p g } } , \mathcal { L } _ { \mathrm { t p } } )$ on both benchmarks.

<table><tr><td>Rejected type | Visual-only</td><td></td><td>Audio-only</td><td>Total</td></tr><tr><td>Rate</td><td>17%</td><td>4%</td><td>21%</td></tr></table>

Table 9: Text-only audio-visual relevance filtering statistics. Rejection is computed over individual caption/query targets rather than whole videos or QA instances.
<table><tr><td>Task Temporal Grounding Data (from AVID &amp; LongVALE)</td><td>|Source</td><td>|Samples</td></tr><tr><td>Event → Time (TVG) Event → Time (TVG) Time → Event (Reverse) Time → Event (Reverse) Dense Captioning (DVC) Dense Captioning (DVC) LongVALE</td><td>AVID LongVALE AVID LongVALE AVID</td><td>34,041 91,863 13,526 36,474 7,239 7,240</td></tr><tr><td>Subtotal General Understanding Data General video QA Speech + visual QA Scene-sound QA</td><td>LLaVA-Video LLaVA-Video (OLA) AVQA</td><td>190,383 200,000 120,925 37,441</td></tr><tr><td>Music QA Subtotal</td><td>FortisAVQA</td><td>31,927</td></tr><tr><td>Total</td><td></td><td>390,293 580,676</td></tr></table>

Table 10: Training data composition of our final model (580,676 samples). Reverse Grounding is produced by inverting a 50,000-sample subset of the forward-TVG pool, so it inherits that pool’s AVID/LongVALE split. The SFT model probed in our preliminary analysis is trained only on the forward-TVG data (125,904 samples), without the Reverse Grounding, DVC, or general-understanding data.

UnAV-100 TVG Test Subset. We additionally construct a temporal video grounding test subset from the UnAV-100 (Geng et al. 2023) test annotations. The original test.parquet contains 5,900 audio-visual event annotations over 2,167 videos, where each annotation is a tuple of video\_id, event label, and timestamp interval. We first group annotations by (video\_id, event\_label). Among the 3,455 resulting video-event combinations, 1,309 correspond to multiple disjoint temporal intervals. Because our evaluation protocol expects one predicted interval per query, these multi-interval cases are ambiguous: a single model output cannot be matched to a unique ground-truth segment without an arbitrary choice. We therefore keep only video-event pairs associated with a single interval, yielding 2,146 QA instances over 1,610 videos.

We then filter the remaining instances by relative event duration. A large fraction of the single-interval cases nearly spans the whole video: 42.8% have ground-truth duration ratio above 0.9, and 53.7% are above 0.7. Such near-fulllength events substantially inflate localization metrics, since a trivial predictor that outputs the whole video can already obtain about 0.62 mIoU over the full single-interval pool. To focus the benchmark on genuine temporal localization, we retain only short events whose ground-truth duration divided by video duration is below 0.3. The final UnAV-100 TVG subset contains 653 QA instances over 490 videos. We will release the corresponding JSON file for reproducible evaluation.

<table><tr><td>Item</td><td>|Specification</td></tr><tr><td>Hardware</td><td></td></tr><tr><td>GPUs</td><td>32 × NVIDIA A100-80GB</td></tr><tr><td>Parallelism</td><td>DeepSpeed ZeRO-2</td></tr><tr><td>NVIDIA driver</td><td>470.161.03</td></tr><tr><td>Software</td><td></td></tr><tr><td>Python</td><td>|3.10.15</td></tr><tr><td>PyTorch</td><td>2.8.0 (CUDA 12.8)</td></tr><tr><td>transformers</td><td>4.57.0</td></tr><tr><td>ms-swift</td><td>4.2.0</td></tr><tr><td>accelerate</td><td>1.13.0</td></tr><tr><td>deepspeed</td><td>0.17.2</td></tr><tr><td>peft</td><td>0.15.0</td></tr><tr><td>flash-attn</td><td>2.8.1</td></tr><tr><td>xformers</td><td>0.0.32</td></tr><tr><td>triton</td><td>3.4.0</td></tr><tr><td>datasets</td><td>4.8.4</td></tr><tr><td>tokenizers</td><td>0.22.1</td></tr><tr><td>torchaudio/torchvision</td><td>2.8.0 / 0.23.0</td></tr><tr><td>v11m (inference/eval)</td><td>0.10.2</td></tr></table>

Table 11: Computing infrastructure and software environment.

TACG TG+PG Weight Grid Response Surface (Contour)  
![](images/465e1a59b646fe503515b5715f275ffc2f43aa5ca222795c90d2ff8bbe572016.jpg)

![](images/8ea900d6ed77ca23e16fbddd2a9a8bb38a0f8365082beb213690a17bcf41c79a.jpg)  
Figure 13: Joint mIoU response surface over the $\lambda _ { \mathrm { t g } } - \lambda _ { \mathrm { p g } }$ grid (26 evaluated points; dashed contour = SFT baseline) on LongVALE (left) and AVID (right).

For our models and the re-evaluated open-source baselines, we use the same benchmark input format and parse only the final predicted temporal interval. Outputs in seconds are evaluated directly. Outputs in frame indices or discrete temporal slots, when produced by methods whose native protocol uses such representations, are converted back to seconds using the benchmark video duration or frame rate before metric computation, following the corresponding benchmark protocol. If a model produces multiple intervals, we evaluate the first complete interval unless the model’s oficial inference protocol specifies otherwise. Invalid or unparsable outputs, including missing start/end times or intervals outside the video duration after clipping, are counted as failed predictions with zero IoU. For start-end inversions, we reorder the two endpoints before clipping to the valid video range, matching common temporal grounding evaluation practice.

The numbers in Table 3 and Table 2 are reported in two ways. For baselines with oficial benchmark results under the same evaluation setting, we cite the published numbers. For open-source omni-modal models and baselines without directly comparable reported results on AVID, LongVALE, or UnAV-100 TVG, we directly evaluate the released checkpoints; we do not fine-tune these baselines on our reconstructed QA data. The re-evaluated baselines use the same prompt template, video preprocessing budget, timestamp parser, and metric script used for OmniGround. Closedsource Gemini 3.1 Pro is included only as a reference result and is not used for training or model selection. Omni-Ground and OmniGround<sup>†</sup> are the only models in Table 3 trained on our 580k mixture, whose grounding portion consists of newly reconstructed QA instances derived from the training-split source videos and temporal captions of AVID and LongVALE. Therefore, Table 3 and Table 2 should be read as benchmark comparisons under each method’s available training protocol; the controlled comparison isolating TACG from data efects is the OmniGround<sup>†</sup> row, which uses the same 580k mixture as OmniGround but replaces TACG with standard cross-entropy.

<table><tr><td rowspan="2">Configuration</td><td colspan="2">AVID</td><td colspan="2">LongVALE</td><td rowspan="2">∆%</td></tr><tr><td>R@0.5</td><td>mIoU</td><td>R@0.5</td><td>mIoU</td></tr><tr><td>Baseline (CE only)</td><td>0.3502</td><td>0.3153</td><td>0.1408</td><td>0.1604</td><td>0.00</td></tr><tr><td> $\mathcal { L } _ { \mathrm { t g } }$  only</td><td>0.3543</td><td>0.3223</td><td>0.1563</td><td>0.1722</td><td>+4.79</td></tr><tr><td> $\mathcal { L } _ { \mathrm { p g } }$  only</td><td>0.3649</td><td>0.3282</td><td>0.1479</td><td>0.1645</td><td>+3.32</td></tr><tr><td> ${ \mathcal { L } } _ { \mathrm { t p } }$  only</td><td>0.3671</td><td>0.3315</td><td>0.1514</td><td>0.1663</td><td>+4.41</td></tr><tr><td>Full w/o  $\mathcal { L } _ { p g } , \mathcal { L } _ { t p }$ </td><td>0.3558</td><td>0.3229</td><td>0.1360</td><td>0.1563</td><td>-0.08</td></tr><tr><td>Full w/o  $\mathcal { L } _ { t g } , \mathcal { L } _ { t p }$ </td><td>0.3350</td><td>0.3056</td><td>0.1293</td><td>0.1488</td><td>-5.16</td></tr><tr><td>Full w/o  $\mathcal { L } _ { t g } , \mathcal { L } _ { p g }$ </td><td>0.3534</td><td>0.3221</td><td>0.1366</td><td>0.1560</td><td>-0.29</td></tr><tr><td> $\mathcal { L } _ { \mathrm { t g } } + \mathcal { L } _ { \mathrm { p g } }$ </td><td>0.3671</td><td>0.3301</td><td>0.1508</td><td>0.1672</td><td>+4.47</td></tr><tr><td>Full w/o  $\mathcal { L } _ { t p }$ </td><td>0.3271</td><td>0.2989</td><td>0.1354</td><td>0.1579</td><td>-3.38</td></tr><tr><td>Full w/o  $\mathcal { L } _ { p g }$ </td><td>0.3481</td><td>0.3150</td><td>0.1446</td><td>0.1597</td><td>-0.27</td></tr><tr><td>Full w/o  $\mathcal { L } _ { t g }$ </td><td>0.3703</td><td>0.3315</td><td>0.1359</td><td>0.1570</td><td>+1.51</td></tr><tr><td>Full TACG</td><td>0.3814</td><td>0.3408</td><td>0.1546</td><td>0.1759</td><td>+8.87</td></tr></table>

Table 12: Full component ablation of the TACG consistency loss. Upright rows: addition results with individually optimized weights; italic rows: subtraction results inheriting the full-model weights. ∆%: average relative improvement over the SFT baseline across both benchmarks. Bold/underline: best/second best per column.

## Timestamp Parsing for TACG

All forward-TVG targets follow a standardized template containing a single well-formed start–end timestamp that can be deterministically parsed into digit positions. During training, we locate the timestamp span from the ground-truth answer template and identify the digit-token positions corresponding to the start and end times. The consistency losses are then computed from the logits at these positions, as described in Eq. 2.

<table><tr><td>Benchmark (mIoU)</td><td>OmniGround (3B)</td></tr><tr><td>AVID</td><td> $5 6 . 9 { \scriptstyle \pm 0 . 4 }$ </td></tr><tr><td>LongVALE</td><td> $2 8 . 6 { \scriptstyle \pm 0 . 5 }$ </td></tr><tr><td>Charades-STA</td><td> $3 8 . 8 { \scriptstyle \pm 0 . 6 }$ </td></tr><tr><td>ActivityNet-Captions</td><td> $2 3 . 7 _ { \pm 0 . 4 }$ </td></tr></table>

Table 13: Multi-seed stability of the 3B final model (mean ± std over three seeds).
<table><tr><td>Configuration</td><td>AVID mIoU</td><td>LongVALE mIoU</td></tr><tr><td>Baseline (CE only)</td><td> $0 . 3 1 4 7 _ { \pm 0 . 0 0 4 2 }$ </td><td> $0 . 1 5 9 8 { \scriptstyle \pm 0 . 0 0 3 1 }$ </td></tr><tr><td> ${ \mathcal { L } } _ { \mathrm { t g } } \ { \mathrm { o n l y } }$ </td><td> $0 . 3 2 1 8 { \scriptstyle \pm 0 . 0 0 3 8 }$ </td><td> $0 . 1 7 1 5 { \scriptstyle \pm 0 . 0 0 3 5 }$ </td></tr><tr><td> $\mathcal { L } _ { \mathrm { p g } } \ : \mathrm { o n l y }$ </td><td> $0 . 3 2 7 5 { \scriptstyle \pm 0 . 0 0 4 5 }$ </td><td> $0 . 1 6 4 0 { \scriptstyle \pm 0 . 0 0 2 9 }$ </td></tr><tr><td> ${ \mathcal { L } } _ { \mathrm { t p } } \ \mathrm { o n l y }$ </td><td> $0 . 3 3 0 9 { \scriptstyle \pm 0 . 0 0 4 0 }$ </td><td> $0 . 1 6 5 8 { \scriptstyle \pm 0 . 0 0 3 3 }$ </td></tr><tr><td> $\mathcal { L } _ { \mathrm { t g } } + \mathcal { L } _ { \mathrm { p g } }$ </td><td> $0 . 3 2 9 6 { \scriptstyle \pm 0 . 0 0 3 6 }$ </td><td> $0 . 1 6 6 7 { \scriptstyle \pm 0 . 0 0 3 0 }$ </td></tr><tr><td>Full TACG</td><td> $\mathbf { 0 . 3 4 0 1 _ { \pm 0 . 0 0 3 4 } }$ </td><td> $\mathbf { 0 . 1 7 5 3 _ { \pm 0 . 0 0 2 8 } }$ </td></tr></table>

Table 14: Multi-seed stability of the component ablation (cf. Table 5), mean ± std over three seeds.

The reconstructed timestamps are converted to seconds and normalized by the video duration before Gaussian projection. We use the timestamp precision provided by the training template and keep the same precision across the start and end fields. The ground-truth distribution is constructed from the annotated interval with the same normalization and Gaussian projection as the prediction, but is detached from the computation graph.

## Discretization Weight Sensitivity

This section reports the full sweep behind the discretization choice made in the main text. For each of the three ways of turning the per-position digit logits into a predicted timestamp—Argmax, Soft, and our Round (straight-through rounding)—we vary the loss weight λ on the predicteddistribution edge and measure mIoU on AVID and Long-VALE. Figure 10 summarizes each method at its best λ against the cross-entropy (CE) baseline, and Figure 11 gives the complete mIoU-versus-λ curves.

The two views agree. Round is the only discretization that stays above the CE baseline on both benchmarks and remains stable across the entire λ range, whereas Argmax and Soft dip below the baseline at their best settings and degrade quickly as λ grows. This matches the failure modes analyzed in the main text: Argmax’s discontinuous digit selection triggers mode-collapse instability, while Soft’s non-integer expectation leaves a loss floor that its gradient cannot cross. The robustness of Round across λ is what makes it usable without per-benchmark tuning.

## Weight Ablation

We ablate the loss weight λ of each edge by sweeping across a range of values with both JS divergence and Wasserstein-1

(W1) as distance metrics.

As shown in Figure 12, all edges share the same fundamental trade-of: when λ is too small, the supervision signal is too weak to meaningfully guide learning; when too large, the distribution-matching gradient overwhelms the crossentropy loss and degrades generation quality. The optimal weight and distance metric, however, difer across edges. The ground-truth-anchored edges admit stronger supervision: $\mathcal { L } _ { \mathrm { p g } }$ peaks with W1 at $\bar { \lambda _ { \mathrm { p g } } } ~ = ~ 0 . 2$ , whose distanceproportional gradients give a smoother landscape that tolerates large weights, whereas $\mathcal { L } _ { \mathrm { t g } }$ favors JS at $\bar { \lambda _ { \mathrm { t g } } } = 0 . 1$ with a broad efective range—for noisy attention supervision, JS measures distributional overlap without sensitivity to the absolute magnitude of misalignment and is thus more stable. The self-supervised edge ${ \mathcal { L } } _ { \mathrm { t p } }$ exhibits a narrower efective range, as it enforces consistency between two model-internal signals without external supervision and excessively large weights cause the distributions to reinforce each other’s errors; W1 with $\lambda _ { \mathrm { t p } } = 0 . 1$ 1 achieves the best balance. Beyond the per-edge sweeps, Figure 13 maps the joint $\mathcal { L } _ { \mathrm { t g } } + \mathcal { L } _ { \mathrm { p g } }$ response surface over a 2D weight grid, confirming that the two ground-truth-anchored edges produce a broad region of improvement over the SFT baseline rather than a brittle optimum.

## Full Component Ablation

Table 12 reports the complete component ablation, including both addition results (upright rows, with individually optimized weights via progressive search) and subtraction results (italic rows, which inherit weights from the full configuration); the main text (Table 5) shows only the addition rows.

Two observations stand out. First, among all single-edge removals from the full model, dropping the self-supervised attention–prediction edge ${ \mathcal { L } } _ { \mathrm { t p } }$ causes the steepest degradation (−3.38%), reinforcing the main-text conclusion that it is the key contributor rather than a redundant term. Second, the subtraction rows—which reuse the full model’s weights instead of re-optimizing for the smaller edge subset— frequently under-perform their addition counterparts, and can even fall below the baseline. This shows that the edge weights do not transfer across subsets, and is why we report both the addition and subtraction views rather than a single ablation direction.

## Multi-Seed Stability

The results in the main text come from a single run with a fixed seed (42). To verify that the reported margins are not seed artifacts—particularly the small full-scale gaps and the closely spaced single-edge ablations—we repeat the key experiments over three seeds and report the mean and standard deviation. Table 13 covers the 3B final model on the four grounding benchmarks, and Table 14 repeats the component ablation of Table 5. Across both tables, the mean stays close to the single-seed estimate and the ordering of configurations is preserved, indicating that the reported gains are stable rather than seed-specific.