# Q-CamTuning: A Multimodal Large Language Model for Fine-Grained Quality Reasoning in Livestreaming Camera Tuning

Yanchao Bi<sup>1</sup>, Pengfei Chen<sup>1</sup>, Yulong Yan<sup>2</sup>, Qian Zhang<sup>2</sup>, Biao Wang<sup>2</sup>, Leida Li<sup>1∗</sup>

<sup>1</sup>School of Artificial Intelligence, Xidian University

<sup>2</sup>Huawei Technologies Ltd.

25171111423@stu.xidian.edu.cn, chenpengfei@xidian.edu.cn, ldli@xidian.edu.cn, ylyan17@fudan.edu.cn,qian.zh@huawei.com, wang-biao106@huawei.com

## Abstract

With the popularization of livestreaming applications, automatic camera tuning has become indispensable for ordinary users to enjoy high Quality of Experience (QoE). However, the existing Image Quality Assessment (IQA) methods face two challenges in camera tuning. First, they struggle to identify subtle quality diference between images, which is common in camera tuning. Second, the ability to explain finegrained diferences and to provide suggestions for camera tuning are even more lacking. To address these challenges, we first construct FGLive-Comments, an instruction tuning dataset with rich annotations of both preferences for finegrained image pairs and quality reasoning from global and local visual evidence, which supports selective adjustment recommendations of camera parameters. Based on FGLive-Comments, we develop Q-CamTuning, a multimodal large language model for livestreaming camera tuning. Its progressive perception–evidence–guidance framework learns qualityaware visual representations, enhances fine-grained reasoning with global–local evidence, and maps quality issues to camera-parameter adjustment directions. Extensive experimental results demonstrate that Q-CamTuning outperforms existing IQA models in both quality scoring and fine-grained image pair ranking. User study also validates its ability to recommend camera tuning suggestions.

## Introduction

With the rapid growth of mobile livestreaming in entertainment, e-commerce, and content creation, automatic camera tuning has become increasingly important for image quality and user experience (Gilstrap and Gilstrap 2023; Hamilton, Garretson, and Kerne 2014). Unlike traditional photography, livestreaming does not allow for post-capture correction, i.e. camera settings must be adjusted in real time as scenes and parameter configurations change. A typical tuning system generates candidate images of the scene under diferent configurations and uses image quality assessment (IQA) to perform parameter selection and refinement (Sheng et al. 2026). Therefore, IQA should not only compare the relative quality of candidate images but also guide subsequent quality improvement.

Most existing blind image quality assessment (BIQA) methods represent image quality using a single scalar score (Mittal, Moorthy, and Bovik 2012; Su et al. 2020; Ke et al. 2021; Yang et al. 2022). Although efective for distinguishing pronounced quality diferences, they provide limited support for fine-grained camera adjustment. Small parameter variations often produce images with highly similar content whose diferences appear only in specific attributes, such as noise, sharpness, exposure, and face quality. To address the above challenge, TuningIQA (Sheng et al. 2026) introduced FGLive-10K and jointly learns overall quality regression and pairwise ranking from multi-attribute quality and fine-grained preference annotations. However, it provides only numerical quality predictions and preference results, without explaining the visual evidence underlying its judgments. Such interpretability is essential for camera tuning, where reliable parameter adjustment requires understanding which quality attributes determine the preference.

![](images/d0445365471f27b094c05036b14ef18f8eb0e0250833d3492ccf0055c1d0d115.jpg)  
Figure 1: Motivation. Coarse-grained preference reasoning may yield incorrect attribute-aware judgment, while the proposed Q-CamTuning performs comparison and further recommends camera-parameter adjustment directions.

Multimodal large language models (MLLMs) have extended image quality assessment beyond score prediction to degradation perception and quality reasoning (Wu et al. 2024a; You et al. 2024; Wu et al. 2024b). Existing methods model quality using text-defined levels, pairwise comparisons, or score distributions (Wu et al. 2023; Zhu et al. 2024; You et al. 2025), while reinforcement-learning-based approaches further improve degradation perception and quality reasoning (Li et al. 2026).

MLLM-based IQA extends quality assessment beyond score prediction to distortion identification, image comparison, and quality reasoning. However, two limitations remain in camera-tuning scenarios. First, these methods may fail to perceive subtle diferences between visually similar images, even when producing plausible explanations. Second, they stop at quality assessment and cannot identify actionable issues in the preferred image or recommend camera-parameter adjustment directions. Thus, the process from fine-grained quality perception to tuning guidance remains incomplete.

To address these limitations, we build FGLive-Comments, a multi-modal dataset for camera tuning with human–machine collaborative annotation, and based upon which we further propose Q-CamTuning, a new multimodal large language model for camera tuning quality reasoning. FGLive-Comments provides global and local evidence for perceptible diferences in sharpness, noise, exposure, and face quality, together with residual quality issues and selective parameter adjustment suggestions. Q-CamTuning adopts a progressive perception–evidence–guidance framework: it first establishes quality perception through scoring and pairwise ranking, then organizes visual evidence to support finegrained preference reasoning, and finally maps actionable issues recommendations to camera-parameter adjustment.

The main contributions of this work are summarized as follows:

• We construct FGLive-Comments, an instruction-tuning dataset that provides evidence-grounded explanations of fine-grained quality diferences and selective parameter guidance for livestreaming camera tuning.

• We propose Q-CamTuning, a new multimodal large language model that progressively learns quality perception, evidence-based reasoning, and parameter-direction recommendation, unifying fine-grained quality assessment with camera-tuning guidance.

• We validate Q-CamTuning through quantitative evaluation of quality scoring and ranking, and subjective evaluation of quality reasoning and parameter guidance, demonstrating reliable quality discrimination capability and useful tuning recommendations.

## Related Work

## Blind and Fine-Grained Image Quality Assessment

BIQA predicts perceptual quality without a reference image. Early methods, such as BRISQUE (Mittal, Moorthy, and Bovik 2012) and NIQE (Mittal, Soundararajan, and Bovik 2012), measured deviations from natural scene statistics. Deep models, including MetaIQA (Zhu et al. 2020), HyperIQA (Su et al. 2020), TReS (Golestaneh, Dadsetan, and Kitani 2022), MUSIQ (Ke et al. 2021), and MANIQA (Yang et al. 2022), improved quality prediction by learning contentand distortion-aware representations. However, most BIQA methods output a single quality score and cannot explicitly distinguish subtle diferences between images. Pairwise methods such as RankIQA (Liu, Van De Weijer, and Bagdanov 2017) and PieAPP (Prashnani et al. 2018) address this limitation by directly learning relative quality preferences.

For camera tuning, TuningIQA (Sheng et al. 2026) introduced FGLive-10K, which provides overall and multiattribute quality annotations together with fine-grained pairwise preferences. TuningIQA jointly learns quality regression and pairwise ranking to distinguish similar candidate images captured under diferent camera settings. Nevertheless, existing methods primarily output scores, attribute predictions, or preference results, without explaining the supporting visual evidence or translating residual quality issues in the preferred image into camera-parameter guidance.

## MLLM-based Image Quality Assessment

MLLMs have extended IQA from numerical prediction to language-based quality perception and reasoning. Q-Bench, Q-Bench+, and Q-Instruct evaluate low-level perception, quality judgment, and degradation understanding. Q-Align (Wu et al. 2023), Compare2Score (Zhu et al. 2024), and DeQA-Score (You et al. 2025) estimate quality through discrete levels, quality comparisons, or score distributions. DepictQA (You et al. 2024) and Co-Instruct (Wu et al. 2024b) support quality description and comparative explanation, while Q-Ground (Chen et al. 2024), ViDA-UGC (Liao et al. 2025), and iDETEX (Zhao et al. 2025) further model localized degradations. Q-Insight (Li et al. 2026) and VisualQuality-R1 (Wu et al. 2026) employ reinforcement learning to improve quality scoring and reasoning.

Despite these advances, existing MLLM-based IQA methods struggle to distinguish the subtle quality diferences commonly encountered in camera tuning or provide explanations grounded in such diferences. Moreover, they cannot diagnose actionable issues in the preferred image or recommend corresponding camera-parameter adjustment directions. The objective of this work is to unify fine-grained quality discrimination, evidence-based explanation, and selective parameter guidance for livestreaming camera tuning.

## FGLive-Comments

## Data Source

We construct FGLive-Comments based on FGLive-10K, a fine-grained BIQA dataset for livestreaming camera tuning (Sheng et al. 2026). FGLive-10K contains 10,185 highresolution images captured under diverse scenes and camera settings, with 50,925 annotations for overall quality, sharpness, noise, exposure, and face quality, as well as 19,234 fine-grained pairwise preferences. It comprises 451 training scenes and 104 test scenes. The training split contains 8,148 images and 72,678 image pairs, including 57,149 coarsegrained pairs and 15,529 fine-grained pairs, while the test split contains 2,037 images and 3,705 fine-grained pairs.

Because our task requires a clear preferred image, we discard pairs labeled as similar and retain 13,365 fine-grained pairs. We expand their attribute annotations and pairwise preferences into structured instructions containing qualitydiference explanations, global or local visual evidence, image-level summaries, and selective parameter-adjustment directions for residual issues in preferred images.

## Human–Machine Collaborative Annotation

As illustrated in Fig. 2, we construct FGLive-Comments through a human–machine collaborative annotation pipeline.

![](images/391f1a05bcc6a37b60e4e40f32fb7d3c1ad1c3b7acc5e1c9523b7432ed377e00.jpg)  
Figure 2: Dimension-wise Data Generation and Verification Pipeline.

For each image pair, we determine the preferred image from the overall preference label. Among sharpness, noise, exposure, and face quality, we retain only dimensions with a clear A/B preference and exclude those labeled as similar. Rather than directly converting these labels into text, we generate visually grounded explanations that identify quality diferences and explain their contributions to the overall preference.

Evidence-Constrained Quality Description. For each selected dimension, we adopt a global-to-local evidence generation strategy. For sharpness, noise, and exposure, a global description is first generated from the image pair. A qualitydimension validator, without access to the ground-truth attribute label, then infers the relative quality from the images and description. The description is retained if its prediction matches the label; otherwise, local evidence search is performed to identify discriminative regions and generate croplevel descriptions. Each description undergoes the same verification, with up to three candidate regions examined. For face quality, facial regions are directly cropped and processed using the same generation and verification procedure. This process reduces unsupported or label-inconsistent descriptions while grounding quality explanations in discriminative visual evidence.

Image-Level Quality Summary. Using the validated dimension-level descriptions and their global and/or local evidence, we generate an image-level summary that integrates quality diferences, identifies the main factors underlying the overall preference, and determines the preferred image. The predicted preference is extracted from the summary and checked against the FGLive-10K label. Inconsistent summaries are regenerated, ensuring alignment between the quality explanation and pairwise supervision.

Selective Camera-Parameter Guidance. After determining the preferred image, we assess whether it contains residual actionable quality issues rather than continuing the pairwise comparison. A dominant issue in sharpness, noise, or exposure is identified from the validated descriptions and visual evidence, which is then mapped to the corresponding parameter-adjustment direction. Color-temperature deviations are assessed independently, with an additional direction provided for a cool or warm color cast when necessary. Thus, each sample contains at most one primary adjustment and one optional color-temperature adjustment. No recommendation is generated when the evidence is insuficient or no actionable issue exists.

Each instruction contains attribute-level analysis, global or local evidence, an image-level summary, and selective parameter guidance. After generating annotations for 13,365 training pairs, we discard samples that fail dimension-level or image-level consistency checks or lack reliable evidence after repeated attempts. The resulting training and test sets contain 12,077 and 3,291 image-description pairs, respectively. All test samples are manually verified and corrected for evidence consistency, issue validity, and compliance with the predefined parameter-mapping rules.

## Q-CamTuning

As illustrated in Fig. 3, Q-CamTuning adopts a two-stage progressive training strategy that separates foundational quality perception from evidence-grounded tuning guidance. Stage-I optimizes the visual encoder and image projector while keeping the LLM frozen, using multi-attribute quality rating and fine-grained pairwise ranking supervision. The resulting model is termed Q-CamTuning-P, where “P” denotes perception, as it establishes overall and attribute-level quality perception. Initialized from Q-CamTuning-P, Stage-II performs instruction tuning on FGLive-Comments to organize visual evidence into structured quality explanations, reason about image-level preferences, identify actionable issues in the preferred image, and provide selective camera-parameter recommendations. This stage produces Q-CamTuning-G, where “G” denotes guidance. Such progressive training ensures that camera-parameter recommendations are grounded in reliable fine-grained quality perception.

![](images/649518e38105eb7d682eead97f7b4b21281f2854289f49354e6bab741d105d14.jpg)  
Figure 3: Progressive Training Strategy of Q-CamTuning.

## Progressive Perception–Evidence–Guidance Training

Stage-I: Perceptual Alignment. Stage-I focuses on visual perceptual alignment. Candidate images in livestreaming camera tuning usually share highly similar content and may difer only subtly in sharpness, noise, exposure, or face quality. To make the visual representations sensitive to such diferences, we freeze the LLM while updating the visual encoder and the vision–language projection layer, thereby aligning the visual features with both overall and attribute-level quality variations.

We use three complementary supervision signals. Coarsegrained quality-distribution supervision establishes general image-level quality perception, while fine-grained pairwise ranking supervision enhances the discrimination of subtle preferences. Since fine-grained pairs contain only small quality diferences and thus provide dificult ranking signals, we additionally sample coarse-grained pairs with clear quality gaps as high-confidence ranking anchors. These samples balance the dificulty of the ranking data and stabilize relative preference learning. Specifically, the quality distribution data consists of 57,149 image pairs from FGLive-10K whose overall quality score diference is larger than 0.8. Following DeQA-Score, we convert continuous quality scores into probability distributions over five quality levels: Excellent, Good, Fair, Poor, and Bad. The ranking data comprises 24,154 fine-grained pairs obtained from FGLive-Comments after removing textual descriptions and applying A/B flipping, together with 30,000 sampled coarse-grained pairs.

Let P denote the training image pairs and D = {overall, sharpness, noise, exposure, face quality} denote the quality dimensions. For each pair $( I _ { a } , \bar { I } _ { b } )$ and attribute d, the model predicts quality distributions $\hat { \mathbf { q } } _ { d } ^ { a }$ and $\hat { \mathbf { q } } _ { d } ^ { b } ,$ and a binary preference probability $\hat { c _ { d } } ^ { a b }$ . The objective of Stage-I is defined as:

$$
\mathcal { L } _ { \mathrm { S 1 } } = \lambda _ { \mathrm { d i s t } } \mathcal { L } _ { \mathrm { d i s t } } + \lambda _ { \mathrm { d i f f } } \mathcal { L } _ { \mathrm { d i f f } } + \lambda _ { \mathrm { r a n k } } \mathcal { L } _ { \mathrm { r a n k } } ,\tag{1}
$$

where ${ \mathcal { L } } _ { \mathrm { d i s t } }$ aligns the predicted quality distributions with the target distributions, $\dot { \mathcal { L } } _ { \mathrm { d i f f } }$ aligns the predicted distributioninduced score diference with the ground-truth score diference, and $\mathcal { L } _ { \mathrm { r a n k } }$ is a binary cross-entropy loss for pairwise preference prediction:

$$
\mathcal { L } _ { \mathrm { r a n k } } = \frac { 1 } { | \mathcal { P } | } \sum _ { ( I _ { a } , I _ { b } ) \in \mathcal { P } } \frac { 1 } { | \mathcal { D } _ { a b } | } \sum _ { d \in \mathcal { D } _ { a b } } \mathrm { B C E } \left( c _ { d } ^ { a b } , \hat { c } _ { d } ^ { a b } \right) ,\tag{2}
$$

where $c _ { d } ^ { a b } \in \{ 0 , 1 \}$ is the ground-truth preference label, $\hat { c } _ { d } ^ { a b }$ is the predicted preference probability, and $\mathcal { D } _ { a b }$ contains the assessment targets with valid A/B labels.

Through this stage, Q-CamTuning-P acquires overall and attribute-level quality perception, providing a stable perceptual foundation for subsequent evidence selection and quality reasoning.

Stage-II: Evidence-Guided Preference Reasoning and Parameter Guidance. As shown in Fig. 3, Stage-II is initialized from Q-CamTuning-P to extend the quality perception learned in Stage-I to evidence-guided image-level preference reasoning and camera-parameter guidance. Although Q-CamTuning-P captures overall and attribute-level quality cues, diferent attributes depend on distinct visual evidence, which may not be adequately preserved by a shared feature representation for attribute-level descriptions. We therefore introduce the Frequency-Dimension Evidence Adapter (FDEA), which decouples evidence extraction into four attribute-aware branches to identify cues relevant to sharpness, noise, exposure, and face quality. Meanwhile, the standard autoregressive CE objective over the full response may not reliably align diagnosed quality issues with controllable camera parameters. We thus introduce the Physical-Prior Quality-to-Parameter Mapping (PQPM) module to incorporate physical quality-to-parameter priors into recommendation generation. Together, these modules enable Stage-II to organize attribute-relevant evidence into structured quality explanations and generate selective camera-parameter recommendations consistent with the diagnosed issues.

![](images/aab8f985849781a7958d18501698942c0b001413af49b9c6cbe5e465ada2025a.jpg)  
Figure 4: Frequency-Dimension Evidence Adapter.

Accordingly, Stage-II adopts parameter-eficient instruction tuning on FGLive-Comments by freezing the visual encoder, vision-to-language projection layer, and base language model while jointly optimizing the LoRA, FDEA, and PQPM parameters. Given an instruction x and its target response $\mathbf { y } = ( y _ { 1 } , \dots , y _ { T } )$ , the CE loss is defined as:

$$
\mathcal { L } _ { \mathrm { C E } } = - \frac { 1 } { \sum _ { t = 1 } ^ { T } m _ { t } } \sum _ { t = 1 } ^ { T } m _ { t } \log p _ { \theta } \left( y _ { t } \mid y _ { < t } , I _ { a } , I _ { b } , \mathbf { x } \right) ,\tag{3}
$$

where $m _ { t } \in \{ 0 , 1 \}$ indicates whether the t-th response token is supervised. Together with the FDEA and PQPM auxiliary objectives defined below, the overall Stage-II objective is

$$
\mathcal { L } _ { \mathrm { S 2 } } = \mathcal { L } _ { \mathrm { C E } } + \mathcal { L } _ { \mathrm { F D E A } } + \lambda _ { \mathrm { P Q P M } } \mathcal { L } _ { \mathrm { P Q P M } } ,\tag{4}
$$

where $\mathcal { L } _ { \mathrm { F D E A } }$ denotes the combined auxiliary losses for attribute-aware evidence learning, and $\mathcal { L } _ { \mathrm { P Q P M } }$ is an additional CE loss computed with a mask restricted to the parameter-recommendation field. The coeficient λ<sub>PQPM</sub> controls its contribution to the overall objective.

## Frequency-Dimension Evidence Adapter

As illustrated in Fig. 4, FDEA applies a one-level Haar transform to raw image patches, obtaining the LL, LH, HL, and HH frequency bands. Statistics from the selected bands are aligned with the projected visual tokens and processed by four lightweight attribute-specific branches. Specifically, the exposure branch uses $L L ,$ , the sharpness branch uses LH and HL, the noise branch uses LH, HL, and HH, and the face-quality branch integrates all four bands. This frequency decomposition enables each branch to extract visual evidence suited to its corresponding quality attribute.

For image $I _ { i } ,$ let $\mathbf { H } _ { v , i } \mathbf { \bar { \Pi } } \in \mathbf { \bar { \mathbb { R } } } ^ { N _ { i } \times C }$ denote its $N _ { i }$ projected visual tokens and $\mathbf { g } _ { i } = \mathrm { M e a n } ( \mathbf { H } _ { v , i } )$ denote the global visual context. Given the local features $\mathbf { Z } _ { i } ^ { d }$ produced by attribute branch d, FDEA aggregates, weights, and injects the

![](images/b941385deb774b1f98ba2b97e20f480a6784922bc77399a9dfedb76f42e18502.jpg)  
Figure 5: Physical-prior quality-to-parameter mapping in PQPM.

attribute-relevant evidence as

$$
\begin{array} { r } { \begin{array} { r l } & { \mathbf { e } _ { i } ^ { d } = \mathrm { P o o l } _ { d } ( \mathbf { Z } _ { i } ^ { d } ; \mathbf { g } _ { i } , \boldsymbol { \pi } _ { i } ^ { d } ) , } \\ & { { \boldsymbol { w } } _ { i } ^ { d } = \mathrm { s o f t m a x } _ { d } \left( \mathrm { A t t n } ( \mathbf { g } _ { i } , \mathbf { e } _ { i } ^ { d } ) \right) , } \\ & { \mathbf { H } _ { v , i } ^ { \prime } = \mathbf { H } _ { v , i } + \alpha \mathbf { 1 } _ { N _ { i } } \left( \displaystyle \sum _ { d \in \mathcal { A } } w _ { i } ^ { d } \mathbf { e } _ { i } ^ { d } \right) ^ { \top } , } \end{array} } \end{array}\tag{5}
$$

where $\boldsymbol { \pi } _ { i } ^ { d }$ denotes the attribute-specific frequency prior, A contains the four quality attributes, and $\mathbf { 1 } _ { N _ { i } }$ broadcasts the fused evidence to all visual-token positions. The coeficient α controls the residual strength.

FDEA is optimized using training-only auxiliary quality heads and an attention-balancing regularizer:

$$
\mathcal { L } _ { \mathrm { F D E A } } = \lambda _ { \mathrm { r a n k } } \mathcal { L } _ { \mathrm { r a n k } } ^ { \mathrm { F D E A } } + \lambda _ { \mathrm { d i s t } } \mathcal { L } _ { \mathrm { d i s t } } ^ { \mathrm { F D E A } } + \lambda _ { \mathrm { b a l } } \mathcal { L } _ { \mathrm { b a l } } ^ { \mathrm { F D E A } } ,\tag{6}
$$

where $\mathcal { L } _ { \mathrm { r a n k } } ^ { \mathrm { F D E A } }$ supervises overall and attribute-level pairwise preferences, $\mathcal { L } _ { \mathrm { d i s t } } ^ { \mathrm { F D E A } }$ aligns the predicted five-level quality distributions with their targets, and $\mathcal { L } _ { \mathrm { b a l } } ^ { \mathrm { F D E A } }$ prevents the attribute attention from collapsing to a single branch.

## Physical-Prior Quality-to-Parameter Mapping

PQPM translates the diagnosed quality issues in the preferred image into camera-parameter recommendations. As illustrated in Fig. 5, it extracts quality signals for exposure, noise, sharpness, and color temperature from the language context preceding the recommendation field, and routes them to controllable camera parameters according to predefined physical relationships.

Given the context features $\mathbf { H } ^ { \mathrm { c t x } }$ , PQPM predicts the quality signals and maps them to a parameter-action representation:

$$
\begin{array} { r } { \mathbf { q } = \operatorname { t a n h } \left( f _ { q } \left( \operatorname { P o o l } ( \mathbf { H } ^ { \operatorname { c t x } } ) \right) \right) , } \\ { \mathbf { a } = \mathbf { q } ^ { \top } \mathbf { M } _ { 0 } , \qquad } \\ { \widetilde { \mathbf { h } } _ { t } = \mathbf { h } _ { t } + \beta m _ { t } ^ { \operatorname { r e c } } f _ { a } ( \mathbf { a } ) , \qquad } \end{array}\tag{7}
$$

where ${ { \bf { M } } _ { 0 } }$ is a fixed physical-prior mapping matrix that encodes predefined quality-to-parameter relationships and remains unchanged during training. The projection $f _ { a }$ maps a to the language-model hidden space, while ${ m } _ { t } ^ { \mathrm { r e c } }$ restricts its injection to the recommendation field.

## Experiments

## Experimental Setup

Datasets and Metrics. We evaluate Q-CamTuning on FGLive-10K and FGLive-Comments using the scene-level split of FGLive-10K. We report SRCC and PLCC for quality scoring and ACC for overall and attribute-level pairwise ranking. Following DepictQA (You et al. 2024), GPT-5.5 evaluates quality descriptions against ground-truth descriptions to compute GPT-Score. Parameter recommendations is assessed through a blind subjective study using preference rate and acceptance score, which measure wins comparable cases, respectively.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td colspan="2">Score Regression</td><td>FG Ranking</td></tr><tr><td>SRCC↑</td><td>PLCC↑</td><td>FG-ACC↑</td></tr><tr><td>DBCNN (Zhang et al. 2018)</td><td>VGG16</td><td>0.751</td><td>0.750</td><td>0.567</td></tr><tr><td>HyperIQA (Su et al. 2020)</td><td>ResNet-50</td><td>0.847</td><td>0.852</td><td>0.598</td></tr><tr><td>MT-A (Fang et al. 2020)</td><td>ResNet-50</td><td>0.882</td><td>0.883</td><td>0.622</td></tr><tr><td>CLIP-IQA+ (Wang, Chan, and Loy 2023)</td><td>CLIP (ResNet-50)</td><td>0.706</td><td>0.704</td><td>0.557</td></tr><tr><td>MUSIQ (Ke et al. 2021)</td><td>Transformer</td><td>0.866</td><td>0.869</td><td>0.686</td></tr><tr><td>SARQUE (Huang et al. 2022)</td><td>MobileNetV3-S</td><td>0.858</td><td>0.859</td><td>0.610</td></tr><tr><td>LIQE (Zhang et al. 2023)</td><td>CLIP (ViT-B/32)</td><td>0.924</td><td>0.921</td><td>0.653</td></tr><tr><td>TuningIQA (Sheng et al. 2026)</td><td>ResNet-50</td><td>0.939</td><td>0.936</td><td>0.728</td></tr><tr><td>Qwen3-VL-8B-Instruct (Bai et al. 2025)</td><td></td><td>0.807</td><td>0.785</td><td>0.540</td></tr><tr><td>Q-Align (Wu et al. 2023)</td><td>mPLUG-Owl2-7B</td><td>0.872</td><td>0.832</td><td>0.631</td></tr><tr><td>Compare2Score (Zhu et al. 2024)</td><td>mPLUG-Owl2-7B</td><td>0.850</td><td>0.819</td><td>0.627</td></tr><tr><td>DeQA-Score (You et al. 2025)</td><td>mPLUG-Owl2-7B</td><td>0.863</td><td>0.864</td><td>0.637</td></tr><tr><td>Q-Insight (Li et al. 2026)</td><td>Qwen-2.5-VL-7B-Instruct</td><td>0.853</td><td>0.795</td><td>0.629</td></tr><tr><td>VisualQuality-R1 (Wu et al. 2026)</td><td>Qwen-2.5-VL-7B-Instruct</td><td>0.865</td><td>0.801</td><td>0.530</td></tr><tr><td>Q-CamTuning-P</td><td>Qwen3-VL-8B-Instruct</td><td>0.962</td><td>0.960</td><td>0.748</td></tr><tr><td>Q-CamTuning-G</td><td>Qwen3-VL-8B-Instruct</td><td>0.961</td><td>0.962</td><td>0.756</td></tr></table>

Table 1: Overall quality scoring and fine-grained ranking performance on FGLive-10K. The best and second-best results are highlighted in bold and underlined, respectively.

<table><tr><td>Method</td><td>Metric</td><td>Sharp. Noise Exp. 1 Face</td></tr><tr><td>CLIP-IQA</td><td>SRCC PLCC FG-ACC</td><td>0.5320.472 0.199 0.510 0.5280.4340.2080.512 0.649 0.622 0.571 0.518</td></tr><tr><td>MT-A</td><td>SRCC PLCC FG-ACC</td><td>0.8960.8040.782 0.807 0.943 0.898 0.863 0.802 0.688 0.665 0.621 0.582</td></tr><tr><td>SARQUE</td><td>SRCC PLCC FG-ACC</td><td>0.868 0.7430.700 0.754 0.9300.8760.804 0.754 0.666 0.645 0.602 0.647</td></tr><tr><td>TuningIQA</td><td>SRCC PLCC FG-ACC</td><td>0.935 0.805 0.897 0.885 0.963 0.905 0.949 0.880 0.773 0.7460.776 0.725</td></tr><tr><td>Q-CamTuning-P PLCC</td><td>SRCC</td><td>0.943 0.829 0.920 0.917 0.962 0.919 0.956 0.910 FG-ACC 0.7950.7440.783 0.747</td></tr></table>

Table 2: Attribute-level quality scoring and ranking performance on FGLive-10K. Highlighting follows Table 1.

Task-Specific Evaluation Sets. For preferred-image issue identification, we construct the Preferred-Image Issue Evaluation Set from preferred images in the FGLive-Comments test split. Each image is assigned a primary quality-issue label from sharpness, noise, exposure, and no actionable issue. Since an image may additionally exhibit a color-temperature deviation, the presence of a color-temperature issue is annotated independently. Labels are verified through GPTassisted human review: GPT-consistent samples are checked by one annotator, while ambiguous or inconsistent samples are independently reviewed by three annotators with majority voting. The resulting set contains 1,058 images.

For parameter-adjustment recommendation, we construct the Parameter-Guidance Evaluation Set by sampling 200 test pairs based on the verified issue labels. The set is balanced across sharpness, noise, exposure, and no-actionableissue cases, with color-temperature cases accounting for approximately one quarter.

Implementation Details. Q-CamTuning is initialized from Qwen3-VL-8B (Bai et al. 2025), which also serves as the baseline. Stage-I trains the visual encoder and projection layer with learning rates of $1 \times 1 0 ^ { - 5 }$ and $5 \times 1 0 ^ { - 5 }$ respectively. Stage-II freezes the visual encoder, retains the trainable projection layer, and applies LoRA (Hu et al. 2022) to the language model with a learning rate of $1 \times 1 0 ^ { - 5 }$ . Training uses AdamW with cosine annealing, a batch size of 128, and one epoch on two NVIDIA RTX PRO 6000 GPUs.

## Evaluation of Quality Perception

Overall Quality Assessment. Table 1 compares Q-CamTuning with existing BIQA methods, the task-specific TuningIQA model, and recent IQA-oriented MLLMs on FGLive-10K. Q-CamTuning-P achieves the best SRCC and PLCC, showing that Stage-I perceptual alignment efectively establishes strong image-level quality scoring ability. For overall fine-grained ranking, Q-CamTuning-G further improves FG-ACC after evidence-guided instruction tuning, demonstrating that selecting and organizing global–local evidence helps image-level preference reasoning in cameratuning scenarios.

<table><tr><td colspan="3">Method Primary Quality Issue ↑ Color Temp. ↑</td></tr><tr><td>Qwen3-VL-8B</td><td>0.436</td><td>0.656</td></tr><tr><td>Q-Insight</td><td>0.423</td><td>0.657</td></tr><tr><td>GPT-5.4</td><td>0.595</td><td>0.614</td></tr><tr><td>Q-CamTuning-G</td><td>0.682</td><td>0.700</td></tr></table>

Table 3: Accuracy of dominant quality-issue identification. Highlighting follows Table 1.
<table><tr><td>Method</td><td>GPT-Score↑</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>0.421</td></tr><tr><td>Q-Insight</td><td>0.247</td></tr><tr><td>GPT-5.4</td><td>0.515</td></tr><tr><td>Q-CamTuning-P</td><td>0.527</td></tr><tr><td>Q-CamTuning-P w/ Stage-II</td><td>0.575</td></tr><tr><td>Q-CamTuning-P w/ Stage-II + FDEA</td><td>0.599</td></tr></table>

Table 4: GPT-Score results for quality description evaluation. Highlighting follows Table 1.

Attribute-Level Quality Assessment. To evaluate attribute-aware quality perception, we compare Q-CamTuning-P with three SOTA methods that support quality attribute prediction: CLIP-IQA, MT-A, SARQUE, and TuningIQA. As shown in Table 2, Q-CamTuning-P achieves competitive performance across all quality attributes, providing a solid perceptual foundation for attribute-level evidence selection in Stage-II.

Preferred-Image Adjustable Issue Identification. On the Preferred-Image Issue Evaluation Set, we compare Q-CamTuning-G with description-capable MLLMs, including Qwen3-VL-8B, Q-Insight, and GPT-5.4. As shown in Table 3, Q-CamTuning-G substantially outperforms the opensource baselines in dominant-issue and color-temperature identification while remaining competitive with GPT-5.4, validating the efectiveness of evidence-guided training.

## Evaluation of Quality Description and Parameter Guidance

Quality Description Evaluation. Table 4 reports GPT-Score, which evaluates generated quality descriptions against ground-truth references. Q-CamTuning-G achieves higher scores than existing MLLMs. Its improvements over Q-CamTuning-P and the variant without FDEA demonstrate the efectiveness of Stage-II and FDEA, respectively, in evidence-grounded quality description.

Subjective Evaluation of Parameter Guidance. We conduct a blind subjective study on the Parameter-Guidance Evaluation Set. Without access to issue labels, color-temperature labels, or method identities, three reviewers compare anonymized recommendations from Q-CamTuning-G and each baseline. Each sample is classified by majority vote as favoring either method, comparable, or neither reasonable. As shown in Table 5, Q-CamTuning-G achieves higher preference rates, demonstrating its parameter-recommendation capability. Its improvement over the variant without PQPM further confirms that PQPM enhances the reasonableness ofthe recommendations.

<table><tr><td>Method</td><td>Pref.↑</td></tr><tr><td>Q-CamTuning-G vs. Qwen3-VL-8B</td><td>0.610</td></tr><tr><td>Q-CamTuning-G vs. Q-Insight</td><td>0.643</td></tr><tr><td>Q-CamTuning-G vs. GPT-5.4</td><td>0.543</td></tr><tr><td>Q-CamTuning-G vs. w.o./ PQPM</td><td>0.580</td></tr></table>

Table 5: Subjective evaluation of parameter-direction recommendation. Pref. denote preference rate.

![](images/5595afab00cbd2ab29624b31423dd1f5a915c8a26fde38e8eec174d5bbf789f2.jpg)  
Figure 6: Qualitative examples of evidence-grounded quality descriptions and camera-parameter guidance. Local evidence is highlighted in red.

## Qualitative Analysis

Figure 6 presents qualitative results generated by Q-CamTuning-G. The model identifies subtle attribute-level diferences using global and local visual evidence, supports its overall preference judgment, and provides actionable camera-parameter recommendations for issues in the preferred image.

## Conclusion

This paper presents Q-CamTuning, an MLLM for livestreaming camera tuning that unifies fine-grained quality assessment, evidence-grounded reasoning, and camera-parameter guidance. We construct FGLive-Comments with global– local visual evidence and selective tuning recommendations, and develop a progressive perception–evidence–guidance framework enhanced by frequency-aware local evidence and quality-to-parameter mapping. Experiments demonstrate strong performance in quality scoring and fine-grained ranking, together with evidence-grounded descriptions and reasonable parameter recommendations. Future work will explore precise parameter-value prediction and real-world closed-loop camera tuning.

## References

Bai, S.; Cai, Y.; Chen, R.; Chen, K.; Chen, X.; Cheng, Z.; Deng, L.; Ding, W.; Gao, C.; Ge, C.; et al. 2025. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631.

Chen, C.; Yang, S.; Wu, H.; Liao, L.; Zhang, Z.; Wang, A.; Sun, W.; Yan, Q.; and Lin, W. 2024. Q-ground: Image quality grounding with large multi-modality models. In Proceedings of the 32nd ACM International Conference on Multimedia, 486–495.

Fang, Y.; Zhu, H.; Zeng, Y.; Ma, K.; and Wang, Z. 2020. Perceptual quality assessment of smartphone photography. In In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 3677–3686.

Gilstrap, C. A.; and Gilstrap, C. M. 2023. Mobile technologies and live streaming commerce: A systematic review and lexical analysis. In 2023 46th MIPRO ICT and Electronics Convention, 36–44. IEEE.

Golestaneh, S. A.; Dadsetan, S.; and Kitani, K. M. 2022. No-reference image quality assessment via transformers, relative ranking, and self-consistency. In In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 1220–1230.

Hamilton, W. A.; Garretson, O.; and Kerne, A. 2014. Streaming on twitch: fostering participatory communities of play within live mixed media. In Proceedings of the SIGCHI Conference on Human Factors in Computing Systems, 1315– 1324.

Hu, E. J.; Shen, Y.; Wallis, P.; Allen-Zhu, Z.; Li, Y.; Wang, S.; Wang, L.; Chen, W.; et al. 2022. Lora: Low-rank adaptation of large language models. International Conference on Learning Representations, 1(2): 3.

Huang, Y.; Li, L.; Yang, Y.; Li, Y.; and Guo, Y. 2022. Explainable and generalizable blind image quality assessment via semantic attribute reasoning. IEEE Transactions on Multimedia, 25: 7672–7685.

Ke, J.; Wang, Q.; Wang, Y.; Milanfar, P.; and Yang, F. 2021. Musiq: Multi-scale image quality transformer. In In Proceedings of the IEEE/CVF International Conference on Computer Vision, 5148–5157.

Li, W.; Zhang, X.; Zhao, S.; Zhang, Y.; Li, J.; Zhang, J.; et al. 2026. Q-insight: Understanding image quality via visual reinforcement learning. Advances in Neural Information Processing Systems, 38: 36802–36827.

Liao, W.; Yuan, J.; Xu, Y.; Guo, C.; Zhang, Z.; Li, J.; Fu, J.; Fan, H.; Li, T.; Cui, J.; et al. 2025. ViDA-UGC: Detailed Image Quality Analysis via Visual Distortion Assessment for UGC Images. arXiv preprint arXiv:2508.12605.

Liu, X.; Van De Weijer, J.; and Bagdanov, A. D. 2017. Rankiqa: Learning from rankings for no-reference image quality assessment. In In Proceedings of the IEEE/CVF International Conference on Computer Vision, 1040–1049.

Mittal, A.; Moorthy, A. K.; and Bovik, A. C. 2012. Noreference image quality assessment in the spatial domain. IEEE Transactions on Image Processing, 21(12): 4695– 4708.

Mittal, A.; Soundararajan, R.; and Bovik, A. C. 2012. Making a “completely blind” image quality analyzer. IEEE Signal processing letters, 20(3): 209–212.

Prashnani, E.; Cai, H.; Mostofi, Y.; and Sen, P. 2018. Pieapp: Perceptual image-error assessment through pairwise preference. In In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 1808–1817.

Sheng, X.; Duan, Z.; Pan, X.; Huang, Y.; Yang, Z.; Chen, P.; and Li, L. 2026. Tuningiqa: Fine-grained blind image quality assessment for livestreaming camera tuning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 17679–17687.

Su, S.; Yan, Q.; Zhu, Y.; Zhang, C.; Ge, X.; Sun, J.; and Zhang, Y. 2020. Blindly assess image quality in the wild guided by a self-adaptive hyper network. In In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 3667–3676.

Wang, J.; Chan, K. C.; and Loy, C. C. 2023. Exploring clip for assessing the look and feel of images. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, 2555–2563.

Wu, H.; Zhang, Z.; Zhang, E.; Chen, C.; Liao, L.; Wang, A.; Li, C.; Sun, W.; Yan, Q.; Zhai, G.; et al. 2024a. Qbench: A benchmark for general-purpose foundation models on low-level vision. In International Conference on Learning Representations, volume 2024, 12547–12573.

Wu, H.; Zhang, Z.; Zhang, W.; Chen, C.; Liao, L.; Li, C.; Gao, Y.; Wang, A.; Zhang, E.; Sun, W.; et al. 2023. Q-align: Teaching lmms for visual scoring via discrete text-defined levels. arXiv preprint arXiv:2312.17090.

Wu, H.; Zhu, H.; Zhang, Z.; Zhang, E.; Chen, C.; Liao, L.; Li, C.; Wang, A.; Sun, W.; Yan, Q.; et al. 2024b. Towards openended visual quality comparison. In European Conference on Computer Vision, 360–377. Springer.

Wu, T.; Zou, J.; Liang, J.; Zhang, L.; and Ma, K. 2026. Visualquality-r1: Reasoning-induced image quality assessment via reinforcement learning to rank. Advances in Neural Information Processing Systems, 38: 88167–88190.

Yang, S.; Wu, T.; Shi, S.; Lao, S.; Gong, Y.; Cao, M.; Wang, J.; and Yang, Y. 2022. Maniqa: Multi-dimension attention network for no-reference image quality assessment. In In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 1191–1200.

You, Z.; Cai, X.; Gu, J.; Xue, T.; and Dong, C. 2025. Teaching large language models to regress accurate image quality scores using score distribution. In In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 14483–14494.

You, Z.; Li, Z.; Gu, J.; Yin, Z.; Xue, T.; and Dong, C. 2024. Depicting beyond scores: Advancing image quality assessment through multi-modal language models. In European Conference on Computer Vision, 259–276. Springer.

Zhang, W.; Ma, K.; Yan, J.; Deng, D.; and Wang, Z. 2018. Blind image quality assessment using a deep bilinear convolutional neural network. IEEE Transactions on Circuits and Systemsfor Video Technology, 30(1): 36–47.

Zhang, W.; Zhai, G.; Wei, Y.; Yang, X.; and Ma, K. 2023. Blind image quality assessment via vision-language correspondence: A multitask learning perspective. In In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 14071–14081.

Zhao, Z.; Yue, X.; Sun, J.; Xie, Y.; Shao, T.; Yao, L.; Xia, F.; and Deng, Y. 2025. iDETEX: Empowering MLLMs for Intelligent DETailed EXplainable IQA. In In Proceedings of the IEEE/CVF International Conference on Computer Vision, 3944–3953.

Zhu, H.; Li, L.; Wu, J.; Dong, W.; and Shi, G. 2020. MetaIQA: Deep meta-learning for no-reference image quality assessment. In In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 14143– 14152.

Zhu, H.; Wu, H.; Li, Y.; Zhang, Z.; Chen, B.; Zhu, L.; Fang, Y.; Zhai, G.; Lin, W.; and Wang, S. 2024. Adaptive image quality assessment via teaching large multimodal model to compare. Advances in Neural Information Processing Systems, 37: 32611–32629.