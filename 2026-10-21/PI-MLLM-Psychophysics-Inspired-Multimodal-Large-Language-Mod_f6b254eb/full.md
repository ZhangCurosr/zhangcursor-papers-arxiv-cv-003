# PI-MLLM: Psychophysics-Inspired Multimodal Large Language Model for Oddly Satisfying Video Emotion Analysis

Yuheng Zhang<sup>1</sup>, Jie Zhang<sup>1</sup>, Yuecong Min<sup>1</sup>, Yunwei Zhao<sup>2</sup>, Meichen Liu<sup>2</sup>, Qiankun Li<sup>3,4</sup>, Shiguang Shan<sup>1</sup>

<sup>1</sup>Institute of Computing Technology, Chinese Academy of Sciences

<sup>2</sup>CNCERT/CC

<sup>3</sup>Nanyang Technological University <sup>4</sup>Imperial College London

## Abstract

Oddly Satisfying Videos (OSVs) can evoke strong viewer responses through repetitive motion, visual tactility, and ASMRlike sounds despite containing few explicit semantic emotion cues. This makes OSVs a challenging setting for Video Emotion Analysis (VEA), since most existing methods focus on semantic video content and under-model the physical sensory dynamics that shape viewer responses. To support Oddly Satisfying Video Emotion Analysis (OSVEA), we construct the Oddly Satisfying Video Dataset (OSVD), comprising 12,000 annotated 5-second clips. Each clip provides vieweremotion labels under video-only, audio-only, and audio-visual conditions, together with explanations grounded in visual and acoustic sensory evidence. Building on OSVD, we propose PI-MLLM, a psychophysics-inspired multimodal large language model that integrates Modality-Specific Dynamics Alignment (MSDA) and Coalition-Based Audio-Visual Attribution (CAVA). MSDA uses auxiliary prediction objectives to align modality representations with visual motion and acoustic energy-onset dynamics. CAVA evaluates emotion scores across modality coalitions and applies Shapley attribution to identify the dominant modality and determine whether visual and auditory evidence is synergistic or conflicting. Experiments show that PI-MLLM improves both viewer-side emotion prediction and sensory-grounded explanation quality, suggesting the value of physical sensory cues to video emotion analysis.

## Introduction

Understanding how multimedia content shapes human emotion is a central goal in afective computing (Picard 1997; Zhao et al. 2019; Wang and Zhao 2022). Research on video emotion understanding has primarily focused on recognizing emotions expressed by on-screen individuals. A complementary line of research instead predicts emotions elicited in viewers by treating the video itself as an afective stimulus (Dhall et al. 2012; Bagher Zadeh et al. 2018; Jiang, Xu, and Xue 2014; Deng, Ananthram, and McKeown 2025; Guo et al. 2025). This viewer-centered task, known as Video Emotion Analysis (VEA), is particularly challenging for Oddly Satisfying Videos (OSVs), whose repetitive movements, visual tactility, and ASMR-like sounds can evoke satisfaction or relaxation despite sparse facial, linguistic, and narrative emotion cues (Nansen and Balanzategui 2022; Poerio et al. 2018). Accordingly, Oddly Satisfying Video Emotion Analysis (OSVEA) requires models to move beyond recognizing objects, scenes, and events and capture how physical sensory dynamics shape the viewer’s afective experience (Reber, Schwarz, and Winkielman 2004; Nansen and Balanzategui 2022).

However, existing VEA is not well suited to sensory-driven afect modeling in terms of supervision, temporal representation, and multimodal attribution (Figure 1). Datasets such as EmoCause focus on semantic content and causes, leaving modality-specific sensory dynamics under-explored (Huang et al. 2025). Although StimuVAR uses optical-flow peaks for frame selection, it may miss the stable repetitive patterns central to OSVs (Guo et al. 2025). Likewise, joint-fusion methods such as eMotions obscure modality dominance and whether audio-visual cues reinforce or conflict with each other (Wu et al. 2025). Consequently, a model may recognize a cutting action and predict interest, yet fail to explain how predictable motion, rhythmic sounds, and their coordination elicit that response. OSVEA therefore requires modality-specific sensory supervision, dynamics-aware representations, and explicit audio-visual attribution.

To address the lack of an OSVEA-specific dataset, we construct the Oddly Satisfying Video Dataset (OSVD), comprising 12,000 annotated 5-second clips segmented and selected from 7,242 online OSV source videos. As illustrated in Figure 2, the raw videos are segmented, semantically deduplicated, and filtered for resolution and audio completeness before annotation. Domain-specific visual and audio expert models then process the isolated streams independently: the visual expert annotates the scene, motion, and state change, while the audio expert annotates the sound character, temporal envelope, and spatial impression. A fusion expert reconciles the two reports to generate audio-visual annotations and explicitly records whether the sensory cues reinforce, conflict with or complement each other. Each clip is annotated under video-only, audio-only and audio-visual conditions using the 27-class Cowen–Keltner taxonomy adopted by VCE (Cowen and Keltner 2017; Mazeika et al. 2022). Human annotations on a sampled subset are compared with the model-generated labels using Cohen’s κ to assess reliability.

Building on OSVD, we propose PI-MLLM, a psychophysics-inspired MLLM framework composed of Modality-Specific Dynamics Alignment (MSDA) and Coalition-Based Audio-Visual Attribution (CAVA). Within

![](images/5dfbe4e6882a7c25a8030d9269aeb5f8480ed88341ee0106b719c23da23962dc.jpg)  
Figure 1: Existing VEA methods rely on sensory-ungrounded semantic annotations and implicit multimodal fusion. Our approach addresses these limitations with psychophysics-aware sensory annotations and explicit multimodal modeling that preserves modality-specific dynamics and cross-modal relations for viewer-emotion reasoning.

MSDA, Visual Motion Dynamics (VMD) and Acoustic Energy-Onset Dynamics (AED) align visual and audio representations with signal-derived optical-flow, RMS-loudness, and onset-strength descriptors, encouraging the encoders to preserve sensory dynamics that semantic supervision may overlook. CAVA evaluates emotion scores under diferent modality coalitions and derives Shapley-style visual and auditory contributions and their interaction (Shapley 1953), allowing PI-MLLM to distinguish modality dominance, complementarity, and conflict. On OSVD, PI-MLLM achieves an emotion classification accuracy of 43.2% and an explanation semantic similarity of 0.661, outperforming the evaluated video-only and audio-visual MLLMs as well as the matched autoregressive fine-tuning baseline. Ablations further show that VMD and AED provide complementary gains and that coalition-based attribution yields additional improvements.

The main contributions are summarized as follows:

• We introduce OSVD, to our knowledge the first dedicated dataset for OSVEA with fine-grained viewer-side emotion labels and detailed physical sensory-grounded explanations across modalities.

• We propose PI-MLLM, a psychophysics-inspired MLLM framework for OSVEA that introduces MSDA to align modality representations with non-semantic visual and acoustic dynamics and CAVA to quantify modality contributions and audio-visual interactions.

• Extensive experiments on OSVD demonstrate the efectiveness of PI-MLLM, with comparisons against state-ofthe-art methods validating its superior performance.

## Related Work

## Video Emotion Analysis Dataset

Video emotion analysis dataset targets the emotion triggered in viewers. It is distinguished from traditional multimodal sentiment analysis datasets like MOSI and MO-SEI (Zadeh et al. 2017; Bagher Zadeh et al. 2018), which focus on the emotion expressed by the character appeared in the video. Originally, movies were an important material to analyze viewers’ emotion. MediaEval16 (Dellandréa et al. 2016) and LIRIS-ACCEDE (Baveye et al. 2015) were created to annotate audience feeling after watching specific movie clips. With the development of video platforms, videos are widespread in these platforms for people to share their emotion. YF-E6 (Xu et al. 2016) and VE-8 (Jiang, Xu, and Xue 2014) were established like mushrooms after rain. They both based on videos from popular video platforms YouTube and Flickr, but annotated based on diferent emotion classification policy. Recently, VEA datasets come to more detailed annotation. With VCE (Mazeika et al. 2022) annotate the probability distribution of emotion evoked by the video, EmoCause (Huang et al. 2025) annotate emotions with abundant explanations including video,audio, shot and content analysis. However, they still lack the annotation of some physical sensory features and the modality-specific emotion labels. In this work, we introduce OSVD, a dataset with unimodal and cross-modal sensory annotation and detailed explanation of physical sensory features which also dominant to viewer’s emotion.

## Video Emotion Analysis Methods

The goal of VEA is to extract the relevant features to bridge the gap between the video content and the triggered emotion. Early studies on VEA mainly center on designing handcrafted features to represent emotion. For example, Acar et al. utilized Mel-frequency cepstral coeficients to capture speech features relevant to viewer’s emotion (Acar, Hopfgartner, and Albayrak 2017). However, the correlation between the simple feature and triggered emotion is hard to be modeled. With the emergence of various VEA dataset, methods begun to focus on the multi-modal information fusion. Representative works like TAM (Pan, Wang, and Fang 2022) designed a cross-temporal multimodal fusion module that applies attention-based fusion to diferent modalities within and across video segments in order to capture the crossmodal feature central to emotion analysis. However, the fusion process may lack intuitive explanation and thus afecting the credibility of the results. Recent work begun to treat external information as additional cues for emotion analysis. For example, LCAfect (Deng, Ananthram, and McKeown 2025) utilize Danmaku as an auxiliary modality for emotion analysis. EmoDETective (Huang et al. 2025) utilize detailed annotation of audio visual content and shot description to improve emotion analysis explainability. However, these methods may lack relative information in many situations. Motivated by predictive-coding and processing-fluency accounts, we use optical-flow dynamics as a descriptor of visual motion predictability relevant to afective responses (Friston 2012; Reber, Schwarz, and Winkielman 2004).

Table 1: Comparison between OSVD and representative VEA datasets. Emotion-label columns indicate whether annotations are provided under audio-only (A), video-only (V), and audio-visual (AV) viewing conditions. Emotion explanations are divided into semantic descriptions and physical sensory descriptions.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Scale</td><td colspan="2"></td><td colspan="2">Emotion Label Emotion Explanation</td></tr><tr><td>A V</td><td>AV</td><td>Semantic</td><td>Physical sensory</td></tr><tr><td>LIRIS-ACCEDE (Baveye et al. 2015)</td><td>9,800</td><td>× ×</td><td>√</td><td>×</td><td>×</td></tr><tr><td>VE-8 (Jiang, Xu, and Xue 2014)</td><td>1,101</td><td>× ×</td><td>√</td><td>×</td><td>×</td></tr><tr><td>YF-E6 (Xu et al. 2016)</td><td>1,637</td><td>× ×</td><td>√</td><td>×</td><td>×</td></tr><tr><td>VAD (Wang et al. 2024)</td><td>19,267</td><td>× ×</td><td>√</td><td>×</td><td>×</td></tr><tr><td>VCE (Mazeika et al. 2022)</td><td>61,046</td><td>× √</td><td>×</td><td>×</td><td>×</td></tr><tr><td>EmoCause (Huang et al. 2025)</td><td>14,000</td><td>× ×</td><td>√</td><td>√</td><td>×</td></tr><tr><td>OSVD (ours)</td><td>12,000√√</td><td></td><td>√</td><td>√</td><td>√</td></tr></table>

## OSVD Construction

To support OSVEA, we construct the Oddly Satisfying Video Dataset (OSVD), with 12,000 annotated 5-second clips segmented and selected from 7,242 online short-form OSV source videos. As shown in Table 1, existing VEA datasets typically annotate only one viewing condition and, at most, provide semantic explanations. OSVD instead provides viewer-emotion labels under audio, visual, and audiovisual conditions, together with both semantic and physical sensory explanations. Figure 2 summarizes its construction and annotation pipeline.

Collection and segmentation. We derive OSV-related search terms from a word cloud ofrepresentative descriptions and use them to retrieve raw videos from online video platforms. Each collected video is standardized and segmented into two non-overlapping 5-second clips. OSVs are often organized around short repeated actions such as slicing, crushing, pouring, cleaning, or brushing (Nansen and Balanzategui 2022; Gallagher 2016); a 5-second clip usually captures one complete micro-action suficient for emotion analysis.

Table 2: Average human agreement with OSVD labels on 100 randomly sampled clips under three modality views.
<table><tr><td>Metric</td><td>Video-only</td><td>Audio-only</td><td>Audio-visual</td></tr><tr><td>Avg. Cohen&#x27;s κ</td><td>0.70</td><td>0.67</td><td>0.71</td></tr></table>

Data cleaning and filtering. We conduct several cleaning and filtering steps to ensure data quality. Firstly, we calculate pairwise semantic similarity between candidate videos and remove one video from each pair whose score exceeds 0.9. Secondly, we discard videos whose orientation-normalized resolution is below 360 × 640 pixels. Thirdly, we inspect media-container and stream metadata, removing data with missing or corrupted audio.

Emotion and physical sensory annotation. The visual and audio experts independently annotate each clip under isolated unimodal conditions. For the visual stream, scene describes the semantic setting and activity, motion captures time-varying movement patterns, and state change records observable transformations caused by the activity. With only video stream provided, Molmo2 (Clark et al. 2026) uses these information to produce a video-only emotion label and explanation. For the audio stream, sound character describes timbral and spectral qualities, temporal envelope captures onset, evolution, and decay, and spatial impression represents perceived source proximity and sound-field extent. Given only this stream, Music Flamingo (Ghosh et al. 2026) uses the three fields to produce an audio-only label and explanation. Both labels use the same fixed 27-class viewer-emotion taxonomy. Claude Sonnet 4.6 (Anthropic 2026) integrates the visual and auditory content reports to generate the audiovisual label and explanation, describing whether the two streams reinforce, conflict with, or complement each other.

Annotation reliability validation. To assess whether the expert-model labels reflect human viewer judgments, we randomly sample 100 clips from OSVD and ask five independent annotators to assign emotion labels for each modality view. Table 2 reports the average Cohen’s κ between the annotators and the corresponding OSVD labels under the expert models’ view. The values show that OSVD annotations are consistently aligned with human judgments across modalities.

## Method

## Overview

Given an audio-visual clip $x = ( V , A )$ , PI-MLLM predicts $y = ( c , e )$ , where c is the viewer-side emotion label and e is its sensory-grounded explanation. As illustrated in Figure $^ { 3 , }$ the visual and audio encoders produce features $F _ { V }$ and $F _ { A }$ for the MLLM decoder to generate c and e. PI-MLLM contains two complementary modules: Modality-Specific Dynamics Alignment (MSDA) and Coalition-Based Audio-Visual Attribution (CAVA). We describe the two modules below.

![](images/0bbf51ed37bfc4d0c917858c8069d53868f6c964a1eeea7647713c3b45899b9c.jpg)  
Figure 2: OSVD construction and annotation pipeline. Word-cloud-derived keywords guide data collection. Then segmented clips are semantically deduplicated, filtered for resolution, and checked for audio integrity. Visual annotations are structured as scene, motion, and state change, while audio annotations are structured as sound character, temporal envelope, and spatia impression. Finally, the modality-specific reports are fused into audio-visual annotations and evaluated against humanjudgments.

## Modality-Specific Dynamics Alignment

Visual Motion Dynamics (VMD). Prediction-error accounts suggest that aesthetic pleasure can arise when prediction errors become progressively reduced as perceptual regularities are learned (Friston 2012; Reber, Schwarz, and Winkielman 2004). Taking a metal-cutting video as an example, after observing repeated cuts, viewers can infer the knife-motion pattern and anticipate subsequent changes. The resulting reduction in prediction error may contribute to the satisfying response. Motivated by this theory, we use mean optical-flow magnitude to represent frame-to-frame visual discrepancy and guide the model to better capture motion predictability. We attach a lightweight motion-regression head $h _ { m }$ to the visual encoder to predict mean optical-flow magnitude in order to help capture visual motion dynamics which are othervise neglected by previous semantic training strategy. Each clip is divided into T temporal windows. At step t, we compute the mean Farneback optical-flow magnitude $g _ { v , t } \in$ R over consecutive frame pairs in the corresponding window (Farnebäck 2003). We supervise the timealigned visual features with a mean-squared error loss:

$$
\mathcal { L } _ { \mathrm { m o t i o n } } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \lVert h _ { m } ( F _ { V } ) _ { t } - g _ { v , t } \rVert _ { 2 } ^ { 2 } .\tag{1}
$$

This objective encourages the visual representation to retain visual motion dynamics that may otherwise be weakly exposed to the text decoder for emotion prediction.

Acoustic energy-onset dynamics (AED). Poerio et al. report that ASMR experiences combine overall calmness with localized physiological activation (Poerio et al. 2018). This coexistence suggests that afective audio should be characterized by both its sustained energy profile and its brief local features. Accordingly, we use frame-wise root-mean-square (RMS) loudness to describe the overall acoustic-energy trajectory relevant to perceived calmness, and onset strength to capture localized acoustic transients relevant to brief activation. At the same T time steps, these two complementary descriptors form the target $g _ { a , t } \in \mathbb { R } ^ { 2 }$ . A single regression head $h _ { a }$ is attached to the audio encoder to predict the descriptors from the audio hidden features $F _ { A }$ , and we also supervise it with a mean-squared error loss:

$$
\mathcal { L } _ { \mathrm { a u d i o } } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \lVert h _ { a } ( F _ { A } ) _ { t } - g _ { a , t } \rVert _ { 2 } ^ { 2 } .\tag{2}
$$

By reconstructing both descriptors, AED encourages the audio representation to preserve the non-semantic acoustic dynamics that distinguish sustained calmness from localized activation. These dynamics also provide complementary sensory evidence for predicting and explaining viewer-side emotion.

## Coalition-Based Audio-Visual Attribution

Audio and vision do not always contribute to OSV afect consistently. A sharp cutting sound may reduce the relaxation suggested by smooth visual motion, while a soft repetitive sound may strengthen a visually calm ASMR clip. Shapley values provide a coalition-based attribution of model outputs (Shapley 1953). PaSE incorporates Shapley-based modality equilibrium into multimodal sentiment training (He et al. 2026). PI-MLLM instead requires modality contributions and interactions to represent genuine dominance, conflict, or complementarity rather than forcing the modalities to balance. CAVA therefore considers two sensory modalities, V and A, and four coalitions $S \in \{ \varnothing , \{ V \} , \{ \overset { . } { A } \} , \{ V , A \} \}$ whose emotion scores are used to derive both Shapley contributions and pairwise interaction.

Specifically, we use a shared auxiliary classifier $q _ { \psi }$ over the 27 emotion classes to predict the emotion labels. For each coalition S, the coalition classifier is written as:

$$
q _ { \psi } ( c \mid S ) = \mathrm { s o f t m a x } { \bigl ( } f _ { \psi } ( S ) { \bigr ) } _ { c } ,\tag{3}
$$

![](images/8e967463869078d1b3ff59cae6b206dd114e43d9e27d9d6c0523053f55af0706.jpg)  
Figure 3: Overall framework of PI-MLLM. Visual and audio inputs are encoded into $F _ { V }$ and $F _ { A }$ to generate viewer emotion c and a sensory-grounded explanation e. MSDA aligns the visual representation with optical-flow dynamics and the audio representation with RMS loudness and onset strength. CAVA uses modality-conditioned emotion scores to estimate visual and auditory contributions and their interaction. The total objective combines generation, dynamics-alignment, coalition classification, and attribution losses.

where $\psi$ denotes the classifier parameters, $f _ { \psi } ( S )$ denotes the 27-class logits computed from the available modality features in $S ,$ and $q _ { \psi } ( \cdot \mid S )$ is the resulting coalition-level emotion distribution. OSVD provides three supervised coalition labels: the video-only emotion label $y _ { v }$ , the audio-only emotion label $y _ { a }$ , and the fused audio-visual emotion label $y _ { a v }$ . We train the head with:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { c l s } } = \mathrm { C E } ( q _ { \psi } ( \cdot  { | } \left\{ V \right\} ) , y _ { v } ) + \mathrm { C E } ( q _ { \psi } ( \cdot  { | } \left\{ A \right\} ) , y _ { a } ) } \\ & { \phantom { x x x x x x x x x x x x x x x x x x x x x x x x } + \mathrm { C E } ( q _ { \psi } ( \cdot  { | } \left\{ V , A \right\} ) , y _ { a v } ) . } \end{array}\tag{4}
$$

The coalition-level score for the final label $y _ { a v }$ is defined in log-odds space:

$$
s _ { y } ( S ) = \log \frac { \tilde { q } _ { \psi } ( y _ { a v } \mid S ) } { 1 - \tilde { q } _ { \psi } ( y _ { a v } \mid S ) } ,\tag{5}
$$

where $\tilde { q } _ { \psi } ( y _ { a v } \mid S ) = \mathrm { c l i p } ( q _ { \psi } ( y _ { a v } \mid S ) , \delta , 1 - \delta )$ , and δ is a small clipping constant for numerical stability.

Shapley contribution computation. We compute the Shapley contributions of vision and audio:

$$
\begin{array} { l } { \displaystyle { \phi _ { V } = \frac { 1 } { 2 } \big [ s _ { y } ( \{ V \} ) - s _ { y } ( \emptyset ) \big ] + \frac { 1 } { 2 } \big [ s _ { y } ( \{ V , A \} ) - s _ { y } ( \{ A \} ) \big ] , } } \\ { \displaystyle { \phi _ { A } = \frac { 1 } { 2 } \big [ s _ { y } ( \{ A \} ) - s _ { y } ( \emptyset ) \big ] + \frac { 1 } { 2 } \big [ s _ { y } ( \{ V , A \} ) - s _ { y } ( \{ V \} ) \big ] . } } \end{array}\tag{6}
$$

Here, a positive value means the modality increases the logodds score of the final viewer-side emotion. To make the attribution signal learnable during training, we further introduce a modality contribution loss. We directly use the signed

Shapley contributions $\phi _ { V }$ and $\phi _ { A }$ as the visual and auditory contribution scores, and normalize their magnitudes into a two-way contribution target:

$$
\psi = \left[ { \frac { \left| \phi _ { V } \right| } { \left| \phi _ { V } \right| + \left| \phi _ { A } \right| + \epsilon } } , { \frac { \left| \phi _ { A } \right| } { \left| \phi _ { V } \right| + \left| \phi _ { A } \right| + \epsilon } } \right] ,\tag{7}
$$

where ϵ is a small constant. A contribution head predicts a distribution $\psi _ { \mathrm { p r e d } }$ over the visual and auditory modalities, and we supervise it with:

$$
{ \mathcal { L } } _ { \mathrm { m o d - c o n t r i b } } = \mathrm { K L } \big ( \mathrm { s t o p g r a d } ( \psi ) \| \psi _ { \mathrm { p r e d } } \big ) .\tag{8}
$$

This modality contribution loss teaches the model which modality contributes more to the final viewer-side emotion.

Shapley interaction computation. We further define the pairwise audio-visual interaction as:

$$
I _ { V A } = s _ { y } ( \{ V , A \} ) - s _ { y } ( \{ V \} ) - s _ { y } ( \{ A \} ) + s _ { y } ( \emptyset ) .\tag{9}
$$

A positive $I _ { V A }$ indicates synergy beyond the sum of unimodal evidence, while a negative $I _ { V A }$ indicates conflict or redundancy. This formulation provides an interpretable estimate of modality interaction rather than forcing the two modalities to agree.We also normalize the interaction efect as:

$$
\hat { I } _ { V A } = \frac { I _ { V A } } { \vert \phi _ { V } \vert + \vert \phi _ { A } \vert + \vert I _ { V A } \vert + \epsilon } ,\tag{10}
$$

and train an interaction head to predict $I _ { \mathrm { p r e d } }$ :

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { m o d - i n t } } = \mathrm { H u b e r } \left( I _ { \mathrm { p r e d } } , \mathrm { s t o p g r a d } ( \hat { I } _ { V A } ) \right) . } \end{array}\tag{11}
$$

This modality interaction loss keeps the interaction target robust when one coalition score is noisy, while the sign of $\hat { I } _ { V A }$ still indicates whether audio and vision form a positive or negative interaction. The combined-modality loss is denoted as:

$$
\mathcal { L } _ { \mathrm { c m } } = \mathcal { L } _ { \mathrm { m o d - c o n t r i b } } + \mathcal { L } _ { \mathrm { m o d - i n t } } .\tag{12}
$$

## Joint Training Objective

Combining the autoregressive generation loss $\mathcal { L } _ { \mathrm { g e n } }$ with the auxiliary losses from sessions above, the total training objective is:

$$
\begin{array} { r l } & { \mathcal { L } = \mathcal { L } _ { \mathrm { g e n } } + \lambda _ { 1 } \mathcal { L } _ { \mathrm { m o t i o n } } + \lambda _ { 2 } \mathcal { L } _ { \mathrm { a u d i o } } + \lambda _ { 3 } \mathcal { L } _ { \mathrm { c l s } } } \\ & { \quad \quad + \lambda _ { 4 } \mathcal { L } _ { \mathrm { c m } } , } \end{array}\tag{13}
$$

where $\mathcal { L } _ { \mathrm { g e n } } , \mathcal { L } _ { \mathrm { m o t i o n } } , \mathcal { L } _ { \mathrm { a u d i o } } , \mathcal { L } _ { \mathrm { c l s } }$ , and $\mathcal { L } _ { \mathrm { { c m } } }$ denote the autoregressive generation, motion, audio, classification, and combined-modality losses respectively. The weights $\lambda _ { 1 } , \ldots , \lambda _ { 4 }$ balance the contribution of each component and are set to zero when the corresponding module is ablated.

## Experiments

## Experimental Settings

Dataset and protocol. The primary evaluation is on OSVD because it provides OSV-specific, modality-aware viewerside labels and sensory-grounded explanations. We use a class-balanced OSVD test set to prevent majority emotions from dominating the metric. Emotion classification is evaluated using top-1 accuracy (ACC%) over the 27-class Cowen– Keltner taxonomy, while explanation quality is measured using Sentence-BERT-based Reason Similarity (RS) (Reimers and Gurevych 2019). We further report an out-of-distribution protocol on VCE and VE-8 after converting their labels to the OSVD taxonomy.

Backbone and implementation. We instantiate the backbone with Qwen3-Omni (Xu et al. 2025b) and fine-tune it with LoRA (Hu et al. 2022) (r=64).

Baselines. External baselines span four groups: video MLLMs (VideoLLaMA3 (Zhang et al. 2025), LLaVA-Video (Zhang et al. 2024), Qwen2.5-VL (Bai et al. 2025)), audio-visual MLLMs (Qwen2.5-Omni (Xu et al. 2025a), video-SALMONN-2+ (Tang et al. 2025)), a closed commercial model (Claude Sonnet 4.6 (Anthropic 2026)) and video emotion analysis methods (Emotion-LLaMA (Cheng et al. 2024), StimuVAR (Guo et al. 2025), EmoDETective (Huang et al. 2025)). We additionally report Qwen3-Omni (Xu et al. 2025b) as the matched baseline for autoregressive(AR) finetuning and PI-MLLM.

## Comparison with SOTA Methods

As shown in Table 3, we report the classification accuracy and semantic similarity of diferent methods on OSVD. Overall, PI-MLLM achieves the best performance, with 43.2% accuracy and 0.661 semantic similarity.

For vision-only models, Qwen2.5-VL and LLaVA-Video achieve the best results in accuracy (23.8%) and semantic similarity (0.487), respectively. PI-MLLM outperforms them by 19.4% and 0.174, showing the benefit of incorporating auditory information.

Table 3: Comparison with SOTA methods on OSVD. ACC (%) denotes 27-class viewer-side emotion classification accuracy, and RS denotes semantic similarity between generated and reference explanations. Best per column in bold.
<table><tr><td>Method</td><td>ACC (%)</td><td>RS</td></tr><tr><td>Video MLLMs</td><td></td><td></td></tr><tr><td>VideoLLaMA3-7B (Zhang et al. 2025)</td><td>11.8</td><td>0.322</td></tr><tr><td>LLaVA-Video-7B (Zhang et al. 2024)</td><td>14.6</td><td>0.487</td></tr><tr><td>Qwen2.5-VL-7B (Bai et al. 2025)</td><td>23.8</td><td>0.472</td></tr><tr><td>Audio-visual MLLMs</td><td></td><td></td></tr><tr><td>Qwen2.5-Omni-7B (Xu et al. 2025a)</td><td>17.6</td><td>0.498</td></tr><tr><td>video-SALMONN-2+ (Tang et al. 2025)</td><td>18.5</td><td>0.463</td></tr><tr><td>Closed commercial model Claude Sonnet 4.6 (Anthropic 2026)</td><td>32.6</td><td>0.508</td></tr><tr><td>Video emotion analysis methods</td><td></td><td></td></tr><tr><td>Emotion-LLaMA-7B (Cheng et al. 2024)</td><td>11.3</td><td>0.301</td></tr><tr><td>StimuVAR-7B (Guo et al. 2025)</td><td>9.1</td><td>0.461</td></tr><tr><td>Qwen3-Omni baseline and our methods</td><td></td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td>Qwen3-Omni-30B (base) (Xu et al. 2025b)</td><td>29.2</td><td>0.535</td></tr><tr><td>Qwen3-Omni-30B + AR fine-tuning</td><td>35.1</td><td>0.636</td></tr><tr><td>Qwen3-Omni-30B + PI-MLLM</td><td>43.2</td><td>0.661</td></tr></table>

Audio-visual models remain limited, with video-SALMONN-2+ reaching 18.5% accuracy and Qwen2.5- Omni achieving 0.498 similarity. PI-MLLM still improves over them by 24.7% and 0.163, indicating that simple multimodal fusion is insuficient for OSVD.

Video emotion analysis methods perform even worse, where Emotion-LLaMA and StimuVAR achieve 11.3% accuracy and 0.461 similarity, respectively. PI-MLLM surpasses them by 31.9% and 0.200, highlighting the dificulty of modeling OSVD with conventional approaches.

Among closed-source baselines, Claude Sonnet 4.6 is the strongest, with 32.6% accuracy and 0.508 similarity. PI-MLLM still leads by 10.6% and 0.153.

Overall, these results demonstrate that PI-MLLM efectively improves viewer emotion understanding in OSVD by leveraging modality-specific sensory dynamics and crossmodal interactions.

## Module Ablation

Table 4 presents the contribution of each component in PI-MLLM. Following the Qwen3-Omni autoregressive baseline, we progressively introduce the proposed supervision objectives to evaluate their individual and combined efects.

MSDA consistently improves performance. Introducing the visual motion dynamics objective alone increases the classification accuracy from 35.1% to 38.8%, while the acoustic energy-onset dynamics objective achieves a larger improvement of 39.7%. Combining both objectives further raises the classification accuracy to 42.1% and the semantic similarity to 0.655, demonstrating that visual motion and acoustic dynamics provide complementary sensory cues for viewer emotion understanding.

We then evaluate the CAVA components. Based on the model equipped with both MSDA objectives, adding modality contribution supervision improves the classification accuracy from 42.1% to 42.7%. Introducing coalition emotion classification further increases the accuracy to 42.8%. Finally, incorporating modality interaction supervision yields the full PI-MLLM, achieving the best performance with 43.2% classification accuracy and 0.661 semantic similarity.

Table 4: Ablation study on OSVD. VMD and AED denote the visual motion dynamics and acoustic energy-onset dynamics auxiliary objectives inside MSDA, respectively. CLS, MC, and MI denote the coalition emotion classification loss, modality contribution loss, and modality interaction loss inside CAVA, respectively.
<table><tr><td>Configuration</td><td>VMD</td><td>AED</td><td>CLS</td><td>MC</td><td>MI</td><td>ACC (%)</td><td>RS</td><td>∆ ACC (%)</td></tr><tr><td>Qwen3-Omni + AR (Xu et al. 2025b)</td><td>一</td><td>一</td><td></td><td></td><td></td><td>35.1</td><td>0.636</td><td></td></tr><tr><td>AR + VMD</td><td>V</td><td>一</td><td></td><td></td><td></td><td>38.8</td><td>0.642</td><td>+3.7</td></tr><tr><td> $\mathrm { A R } + \mathrm { A E D }$ </td><td>一</td><td>V</td><td></td><td></td><td></td><td>39.7</td><td>0.646</td><td>+4.6</td></tr><tr><td> $\mathrm { A R } + \mathrm { V M D } + \mathrm { A E D }$ </td><td>V</td><td>V</td><td></td><td></td><td></td><td>42.1</td><td>0.655</td><td>+7.0</td></tr><tr><td> $\boldsymbol { \mathrm { A R } } + \boldsymbol { \mathrm { V M D } } + \boldsymbol { \mathrm { A E D } } + \boldsymbol { \mathrm { M C } }$ </td><td>V</td><td>√</td><td></td><td>√</td><td></td><td>42.7</td><td>0.659</td><td>+7.6</td></tr><tr><td> $\mathrm { A R } + \mathrm { V M D } + \mathrm { A E D } + \mathrm { C L S } + \mathrm { M C }$ </td><td>V</td><td>√</td><td>V</td><td>√</td><td></td><td>42.8</td><td>0.659</td><td>+7.7</td></tr><tr><td>PI-MLLM</td><td>V</td><td>V</td><td>V</td><td>V</td><td>V</td><td>43.2</td><td>0.661</td><td>+8.1</td></tr></table>

Table 5: Out-of-distribution evaluation on VCE and VE-8. Both datasets are evaluated after re-annotation under the OSVD viewer-side 27-class taxonomy and sensoryexplanation protocol.
<table><tr><td rowspan="2">Method</td><td colspan="2">VCE</td><td colspan="2">VE-8</td></tr><tr><td>ACC (%)</td><td>RS</td><td>ACC (%)</td><td>RS</td></tr><tr><td>Qwen3-Omni (Xu et al. 2025b)</td><td>27.6</td><td>0.486</td><td>29.1</td><td>0.501</td></tr><tr><td colspan="5">Video emotion analysis methods</td></tr><tr><td>Emotion-LLaMA (Cheng et al. 2024)</td><td>22.4</td><td>0.368</td><td>24.7</td><td>0.383</td></tr><tr><td>StimuVAR (Guo et al. 2025)</td><td>34.3</td><td>0.505</td><td>35.5</td><td>0.471</td></tr><tr><td>EmoDETective (Huang et al. 2025)</td><td>31.9</td><td>0.497</td><td>31.4</td><td>0.518</td></tr><tr><td colspan="5">Our methods</td></tr><tr><td>Qwen3-Omni + AR finetuning</td><td>30.9</td><td>0.524</td><td>32.6</td><td>0.536</td></tr><tr><td> $\mathrm { Q w e n 3 - O m n i + P I - M L L M }$ </td><td>33.8</td><td>0.552</td><td>36.1</td><td>0.567</td></tr></table>

Overall, every proposed component consistently contributes to the final performance. The MSDA objectives enable the model to capture visual and auditory physical cues, while the CAVA objectives explicitly supervise coalition emotion prediction, modality contribution estimation, and cross-modal interaction modeling. Their complementary efects lead to the strongest overall performance on OSVD.

## Out-of-Distribution Generalization

Table 5 evaluates the out-of-distribution generalization ability of PI-MLLM on VCE and VE-8. Since neither dataset originally follows the OSVD annotation protocol, both are re-annotated using the same 27-class viewer-side emotion taxonomy and sensory-grounded explanation format to enable a fair comparison.

As expected, all methods perform worse than on OSVD because of the domain shift in video content, emotion distributions, and annotation styles. Nevertheless, PI-MLLM demonstrates strong generalization across both datasets. On VCE, StimuVAR achieves the highest classification accuracy (34.3%), while PI-MLLM obtains a comparable classification accuracy of 33.8% and the highest semantic similarity score of 0.552. On VE-8, PI-MLLM achieves the best performance on both metrics, reaching 36.1% classification accuracy and 0.567 semantic similarity. Compared with the strongest competing methods, PI-MLLM improves the semantic similarity by 0.047 on VCE and improves the classification accuracy and semantic similarity on VE-8 by 0.6 percentage points and 0.049, respectively.

Compared with the autoregressive fine-tuning baseline, PI-MLLM consistently improves both classification accuracy and semantic similarity on the two datasets, demonstrating that the proposed supervision generalizes beyond the OSVD training distribution. Overall, these results indicate that explicitly modeling modality-specific sensory dynamics together with modality contributions and interactions produces representations that transfer efectively to unseen video emotion datasets.

## Conclusion

This work investigates viewer emotion understanding in Oddly Satisfying Videos and highlights the importance of audio-visual sensory dynamics that are often overlooked by semantic-centered VEA methods. Through OSVD and PI-MLLM, we incorporated modality-aware emotion supervision, modality-specific dynamics alignment, and coalitionbased audio-visual attribution into a unified framework for OSVEA. Experimental results demonstrate that modeling these factors improves both viewer-emotion prediction and sensory-grounded explanation, while the ablation studies confirm the complementary roles ofMSDA and CAVA. Overall, our findings suggest that viewer-centered video emotion analysis should consider not only what a video depicts, but also how its sensory dynamics unfold and how individual modalities shape the resulting afective response.

## References

Acar, E.; Hopfgartner, F.; and Albayrak, S. 2017. A Comprehensive Study on Mid-Level Representation and Ensemble Learning for Emotional Analysis of Video Material. Multimedia Tools and Applications, 76(9): 11809–11837.

Anthropic. 2026. Introducing Claude Sonnet 4.6. Anthropic News.

Bagher Zadeh, A.; Liang, P. P.; Poria, S.; Cambria, E.; and Morency, L.-P. 2018. Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph. In Proceedings of the 56th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), 2236–2246. Association for Computational Linguistics.

Bai, S.; Chen, K.; Liu, X.; Wang, J.; Ge, W.; Song, S.; Dang, K.; Wang, P.; Wang, S.; Tang, J.; Zhong, H.; Zhu, Y.; Yang, M.; Li, Z.; Wan, J.; Wang, P.; Ding, W.; Fu, Z.; Xu, Y.; Ye, J.; Zhang, X.; Xie, T.; Cheng, Z.; Zhang, H.; Yang, Z.; Xu, H.; and Lin, J. 2025. Qwen2.5-VL Technical Report. arXiv preprint arXiv:2502.13923.

Baveye, Y.; Dellandréa, E.; Chamaret, C.; and Chen, L. 2015. LIRIS-ACCEDE: A Video Database for Afective Content Analysis. IEEE Transactions on Afective Computing, 6(1): 43–55.

Cheng, Z.; Cheng, Z.-Q.; He, J.-Y.; Sun, J.; Wang, K.; Lin, Y.; Lian, Z.; Peng, X.; and Hauptmann, A. G. 2024. Emotion-LLaMA: Multimodal Emotion Recognition and Reasoning with Instruction Tuning. In Advances in Neural Information Processing Systems, volume 37, 110805–110853.

Clark, C.; Zhang, J.; Ma, Z.; Park, J. S.; Tripathi, R.; Lee, S.; Salehi, M.; Ren, J.; Kim, C. D.; Yang, Y.; Shao, V.; Yang, Y.; Huang, W.; Gao, Z.; Anderson, T.; Zhang, J.; Jain, J.; Stoica, G.; Farhadi, A.; and Krishna, R. 2026. Molmo2: Open Weights and Data for Vision-Language Models with Video Understanding and Grounding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 28652–28668.

Cowen, A. S.; and Keltner, D. 2017. Self-report captures 27 distinct categories of emotion bridged by continuous gradients. Proceedings of the National Academy of Sciences, 114(38): E7900–E7909.

Dellandréa, E.; Chen, L.; Baveye, Y.; Sjöberg, M.; and Chamaret, C. 2016. The MediaEval 2016 Emotional Impact of Movies Task. CEUR Workshop Proceedings, 1739.

Deng, Z.; Ananthram, A.; and McKeown, K. 2025. Enhancing Multimodal Afective Analysis with Learned Live Comment Features. In Proceedings ofthe Thirty-Ninth AAAI Conference on Artificial Intelligence, volume 39, 16253–16261.

Dhall, A.; Goecke, R.; Lucey, S.; and Gedeon, T. 2012. Collecting Large, Richly Annotated Facial-Expression Databases from Movies. IEEE MultiMedia, 19(3): 34–41.

Farnebäck, G. 2003. Two-Frame Motion Estimation Based on Polynomial Expansion. In Image Analysis: 13th Scandinavian Conference, SCIA 2003, volume 2749 of Lecture Notes in Computer Science, 363–370. Springer.

Friston, K. 2012. Predictive coding, precision and synchrony. Cognitive Neuroscience, 3(3-4): 238–239.

Gallagher, R. 2016. Eliciting Euphoria Online: The Aesthetics of “ASMR” Video Culture. Film Criticism, 40(2).

Ghosh, S.; Goel, A.; Koroshinadze, L.; Lee, S.-g.; Kong, Z.; Santos, J. F.; Duraiswami, R.; Manocha, D.; Ping, W.; Shoeybi, M.; and Catanzaro, B. 2026. Music Flamingo: Scaling Music Understanding in Audio Language Models. In The Fourteenth International Conference on Learning Representations.

Guo, Y.; Siddiqui, F.; Zhao, Y.; Chellappa, R.; and Lo, S.-Y. 2025. StimuVAR: Spatiotemporal Stimuli-Aware Video Affective Reasoning with Multimodal Large Language Models. International Journal of Computer Vision, 133(9): 6456– 6472.

He, K.; Chen, B.; Ding, Y.; Li, F.; Teng, C.; and Ji, D. 2026. PaSE: Prototype-aligned Calibration and Shapleybased Equilibrium for Multimodal Sentiment Analysis. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 30960–30968.

Hu, E. J.; Shen, Y.; Wallis, P.; Allen-Zhu, Z.; Li, Y.; Wang, S.; Wang, L.; and Chen, W. 2022. LoRA: Low-Rank Adaptation of Large Language Models. In The Tenth International Conference on Learning Representations.

Huang, X.; Zhou, Y.; Li, J.; Lu, S.; and Wang, S. 2025. EmoDETective: Detecting, Exploring, and Thinking Emotional Cause in Videos. In Proceedings of the 33rd ACM International Conference on Multimedia, 5735–5744.

Jiang, Y.-G.; Xu, B.; and Xue, X. 2014. Predicting Emotions in User-Generated Videos. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 28, 73–79.

Mazeika, M.; Tang, E.; Zou, A.; Basart, S.; Chan, J. S.; Song, D.; Forsyth, D. A.; Steinhardt, J.; and Hendrycks, D. 2022. How Would the Viewer Feel? Estimating Wellbeing from Video Scenarios. In Advances in Neural Information Processing Systems, volume 35, 18571–18585.

Nansen, B.; and Balanzategui, J. 2022. Visual Tactility: ‘Oddly Satisfying’ Videos, Sensory Genres and Ambiguities in Children’s YouTube. Convergence: The International Journal of Research into New Media Technologies, 28(6): 1555–1576.

Pan, J.; Wang, S.; and Fang, L. 2022. Representation Learning through Multimodal Attention and Time-Sync Comments for Afective Video Content Analysis. In Proceedings of the 30th ACM International Conference on Multimedia, 42–50.

Picard, R. W. 1997. Afective Computing. MIT Press. ISBN 9780262161701.

Poerio, G. L.; Blakey, E.; Hostler, T. J.; and Veltri, T. 2018. More than a feeling: Autonomous sensory meridian response (ASMR) is characterized by reliable changes in afect and physiology. PLOS ONE, 13(6): e0196645.

Reber, R.; Schwarz, N.; and Winkielman, P. 2004. Processing Fluency and Aesthetic Pleasure: Is Beauty in the Perceiver’s Processing Experience? Personality and Social Psychology Review, 8(4): 364–382.

Reimers, N.; and Gurevych, I. 2019. Sentence-BERT: Sentence Embeddings Using Siamese BERT-Networks. In Proceedings of the 2019 Conference on Empirical Methods

in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 3982–3992. Hong Kong, China: Association for Computational Linguistics.

Shapley, L. S. 1953. A Value for n-Person Games. In Kuhn, H. W.; and Tucker, A. W., eds., Contributions to the Theory ofGames II, 307–317. Princeton University Press.

Tang, C.; Li, Y.; Yang, Y.; Zhuang, J.; Sun, G.; Li, W.; Ma, Z.; and Zhang, C. 2025. video-SALMONN 2: Captioning-Enhanced Audio-Visual Large Language Models. arXiv preprint arXiv:2506.15220.

Wang, D.; and Zhao, X. 2022. Afective Video Recommender Systems: A Survey. Frontiers in Neuroscience, 16: 984404.

Wang, S.; Li, X.; Zheng, F.; Pan, J.; Li, X.; Chang, Y.; Zhu, Z.; Li, Q.; Wang, J.; and Xiao, Y. 2024. VAD: A Video Affective Dataset with Danmu. IEEE Transactions on Afective Computing, 15(4): 1889–1905.

Wu, X.; Sun, H.; Xue, J.; Nie, J.; Kong, X.; Zhai, R.; Huang, D.; and He, L. 2025. Towards Emotion Analysis in Shortform Videos: A Large-Scale Dataset and Baseline. In Proceedings of the 2025 International Conference on Multimedia Retrieval, 1497–1506.

Xu, B.; Fu, Y.; Jiang, Y.-G.; Li, B.; and Sigal, L. 2016. Video Emotion Recognition with Transferred Deep Feature Encodings. In Proceedings ofthe 2016ACMon International Conference on Multimedia Retrieval, 15–22.

Xu, J.; Guo, Z.; He, J.; Hu, H.; He, T.; Bai, S.; Chen, K.; Wang, J.; Fan, Y.; Dang, K.; Zhang, B.; Wang, X.; Chu, Y.; and Lin, J. 2025a. Qwen2.5-Omni Technical Report. arXiv preprint arXiv:2503.20215.

Xu, J.; Guo, Z.; Hu, H.; Chu, Y.; Wang, X.; He, J.; Wang, Y.; Shi, X.; He, T.; Zhu, X.; Lv, Y.; Wang, Y.; Guo, D.; Wang, H.; Ma, L.; Zhang, P.; Zhang, X.; Hao, H.; Guo, Z.; Yang, B.; Zhang, B.; Ma, Z.; Wei, X.; Bai, S.; Chen, K.; Liu, X.; Wang, P.; Yang, M.; Liu, D.; Ren, X.; Zheng, B.; Men, R.; Zhou, F.; Yu, B.; Yang, J.; Yu, L.; Zhou, J.; and Lin, J. 2025b. Qwen3-Omni Technical Report. arXiv preprint arXiv:2509.17765.

Zadeh, A.; Chen, M.; Poria, S.; Cambria, E.; and Morency, L.-P. 2017. Tensor Fusion Network for Multimodal Sentiment Analysis. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, 1103– 1114. Association for Computational Linguistics.

Zhang, B.; Li, K.; Cheng, Z.; Hu, Z.; Yuan, Y.; Chen, G.; Leng, S.; Jiang, Y.; Zhang, H.; Li, X.; Jin, P.; Zhang, W.; Wang, F.; Bing, L.; and Zhao, D. 2025. VideoLLaMA 3: Frontier Multimodal Foundation Models for Image and Video Understanding. arXiv preprint arXiv:2501.13106.

Zhang, Y.; Wu, J.; Li, W.; Li, B.; Ma, Z.; Liu, Z.; and Li, C. 2024. Video Instruction Tuning With Synthetic Data. arXiv preprint arXiv:2410.02713.

Zhao, S.; Wang, S.; Soleymani, M.; Joshi, D.; and Ji, Q. 2019. Afective Computing for Large-Scale Heterogeneous Multimedia Data: A Survey. ACM Transactions on Multimedia Computing, Communications, and Applications, 15(3s): 1–32.

# PI-MLLM: Psychophysics-Inspired Multimodal Large Language Model for Oddly Satisfying Video Emotion Analysis

# Supplementary Material

## OSVD Details

Source. OSVD was collected from YouTube videos posted within the five years preceding collection, retrieved with queries such as “stress relief,” “oddly satisfying,” and related keywords.

## Modality-Isolated Annotation Prompts

Molmo2, Music Flamingo, and Claude Sonnet 4.6 produce the video-only, audio-only, and fused annotations, respectively. The prompts used in the annotation pipeline are shown below. In every prompt, emotion must be exactly one of the 27 viewer-emotion labels defined in the main paper.

## Video-only prompt.

You are the visual expert for oddly satisfying video emotion analysis. You receive only the silent five-second video. Predict the dominant emotion elicited in a typical viewer, not an emotion expressed by a person or animal. Never infer sound. Ground the analysis in: (1) scene, objects, and materials; (2) motion speed, repetition, smoothness, regularity, and predictability; (3) visible state change, deformation, orfragmentation; and (4) color and lighting. Select exactly one label from the 27-label list. Return JSON only: {emotion, scene, motion, state\_change, physical\_cues, reason}.

## Audio-only prompt.

You are the audio expert for oddly satisfying video emotion analysis. You receive only the audio track. Predict the dominant emotion elicited in a typical listener. Never infer visible objects or actions. Ground the analysis in: (1) timbre, texture, and frequency content; (2) attack, evolution, repetition, rhythm, and decay; (3) loudness stability and onset/transient dynamics; and (4) spatial impression, source proximity, and speech presence. Select exactly one label from the 27-label list. Return JSON only: {emotion, sound\_character, temporal\_envelope, spatial\_impression, physical\_cues, reason}.

## Audio-visual fusion prompt.

You receive the independently generated video-only and audio-only reports. Predict the dominant viewer emotion after experiencing both modalities. Do not add facts absent from the reports. Cite concrete evidence from both modalities and identify their relation as visualdominant, audio-dominant, synergy, conflict, or complementary. The final label need not equal either unimodal label. Return JSON only: {emotion, visual\_evidence, audio\_evidence, relation, bridge, reason}.

## Full-Dataset Annotation Distributions

Figure S1 shows that Satisfaction constitutes the largest overall share across the three annotation views, while a broad range of other viewer-emotion categories remains represented. The modality-specific distributions are not identical: Satisfaction dominates the video-only annotations, whereas the audio-only distribution shifts toward Calmness, showing that isolated acoustic cues can induce an afective prior diferent from that of visual dynamics.

![](images/1958be75a1f573a8983d91a4a70d706bbf8a6bcef75e4c16313f1658d26d1fb6.jpg)  
Figure S1: Full OSVD three-view label distributions. Counts are computed over all 12,000 clips for audio-visual, video-only, and audio-only annotation.

## Annotation Reliability Validation

The main paper reports only the mean Cohen’s κ for each modality. Table S1 gives the complete per-annotator results for five human annotators against the corresponding modelgenerated labels on the same 100 clips. The closely clustered scores and the standard deviation of 0.03 for every view indicate stable agreement patterns across annotators, while the κ range of 0.63–0.76 demonstrates consistently high agreement between individual annotators and the automatic labels.

Table S1: Human–model Cohen’s κ (N=100 clips per annotator and view).
<table><tr><td>Annotator</td><td>Video-only Audio-only</td><td></td><td>Audio-visual</td></tr><tr><td>Annotator 1</td><td>0.66</td><td>0.63</td><td>0.67</td></tr><tr><td>Annotator 2</td><td>0.71</td><td>0.69</td><td>0.73</td></tr><tr><td>Annotator 3</td><td>0.68</td><td>0.65</td><td>0.69</td></tr><tr><td>Annotator 4</td><td>0.70</td><td>0.67</td><td>0.71</td></tr><tr><td>Annotator 5</td><td>0.74</td><td>0.72</td><td>0.76</td></tr><tr><td>Mean ± SD</td><td>0.70±0.03</td><td>0.67±0.03</td><td>0.71±0.03</td></tr></table>

![](images/c2eecd49985489f18aca48d82e3b9acdb1c4243f52417f03d1010220a17f7002.jpg)  
Figure S2: Two PI-MLLM cases. Case 1 is visually dominant (V=0.86, A=0.14) with negative interaction (−0.071, conflict); Case 2 is audio dominant (V=0.16, A=0.84) with positive interaction (+0.059, synergy).

## Definition of the Empty Coalition in CAVA

Coalition-Based Audio-Visual Attribution (CAVA) represents no video and no audio by a modality-independent zero vector:

$$
\begin{array} { r l } { z _ { \emptyset } = e _ { \emptyset } = { \bf 0 } , \quad } & { { } z _ { \{ V \} } = P _ { V } ( \bar { F } _ { V } ) , } \\ { z _ { \{ A \} } = P _ { A } ( \bar { F } _ { A } ) , } & { { } z _ { \{ V , A \} } = P _ { V } ( \bar { F } _ { V } ) + P _ { A } ( \bar { F } _ { A } ) , } \end{array}\tag{14}
$$

where $P _ { V }$ and $P _ { A }$ project pooled modality features into the shared space. With the shared classifier, $f _ { \psi } ( \emptyset ) = W _ { \psi } \mathbf { 0 } +$ $b _ { \psi } = b _ { \psi }$ . Thus, f(∅) is the classifier-bias baseline used by the Shapley equations. It contains no encoded black video, silent waveform, text token, or pseudo emotion label.

## Case Study

The same checkpoint is evaluated under video-only, audioonly, and audio-visual input. Figure S2 reports the generated labels and explanations, normalized Shapley contributions, and signed interaction. Case 1 shows that regular visual cutting can dominate an unpleasant sound; Case 2 shows that an ASMR-like sound can dominate while the gentle visual rhythm reinforces it.

## Ablation of Physical-Feature Annotation

The semantic-only target retains object, scene, and event descriptions; the semantic+physical target additionally retains the visual and acoustic fields listed in the prompts above. All other settings are fixed.

Table S2: Efect of physical-feature annotation. ACC is 27-class top-1 accuracy; RS is explanation cosine similarity.
<table><tr><td>Training annotation</td><td>Method</td><td>ACC (%)</td><td>RS</td></tr><tr><td>Semantic only</td><td>Qwen3-Omni + AR fine-tuning</td><td>33.2</td><td>0.574</td></tr><tr><td>Semantic + physical</td><td>Qwen3-Omni + AR fine-tuning</td><td>35.1</td><td>0.636</td></tr><tr><td>Semantic only</td><td>Qwen3-Omni + PI-MLLM</td><td>38.9</td><td>0.602</td></tr><tr><td>Semantic + physical</td><td>Qwen3-Omni + PI-MLLM</td><td>43.2</td><td>0.661</td></tr></table>

Table S2 shows that adding physical-feature annotations improves both training approaches over their semanticonly counterparts: AR fine-tuning rises from 33.2/0.574 to 35.1/0.636 in ACC/RS, while PI-MLLM rises from 38.9/0.602 to 43.2/0.661. Under the matched semantic+physical setting, AR fine-tuning still trails PI-MLLM by 8.1 ACC points and 0.025 RS. Together with PI-MLLM’s larger ACC gain from physical annotations (4.3 versus 1.9 points), these results indicate that the proposed method efectively learns and exploits the importance of physical sensory features for viewer-emotion analysis.

Experimental Hyperparameter Design The experimental hyperparameter settings are shown in Table S3.

Table S3: PI-MLLM experimental hyperparameter settings.
<table><tr><td>Hyperparameter</td><td>Value</td><td>Hyperparameter</td><td>Value</td></tr><tr><td>Backbone</td><td>Qwen3-Omni-30B-A3B-Thinking</td><td>Precision</td><td>bf16</td></tr><tr><td>Frozen encoders</td><td>Visual and audio</td><td>LoRA target</td><td>Decoder q/k/v/o</td></tr><tr><td>LoRA rank / alpha / dropout</td><td>64/128/0.05</td><td>Optimizer / decay</td><td>AdamW / 0</td></tr><tr><td>Learning rate / schedule</td><td>10−4 / cosine</td><td>Warmup / grad. clip</td><td>10%/1.0</td></tr><tr><td>Loader batch / accumulation</td><td>8/1</td><td>Epochs / seed</td><td>5/42</td></tr><tr><td>Video rate / pixel range</td><td>1 fps / 16,384–50,176</td><td>Temporal steps T</td><td>10</td></tr><tr><td>λ₁, λ2 (VMD, AED)</td><td>0.1,0.1</td><td>λ3, λ₄ (CLS, CM)</td><td>0.1,0.1</td></tr><tr><td>CAVA classes / hidden / dropout 27/512/0.1</td><td></td><td></td><td></td></tr></table>