# TiTok: Audio-Visual LLM for Multi-Segment Temporal Grounding

Eunji Shin<sup>⋆1</sup>, Dahyun Choi<sup>⋆1</sup>, Seungyeon Jo<sup>1</sup>, Yejin Hong<sup>1</sup>, Jiyoung Lee<sup>†1,2</sup>

<sup>1</sup>Department of Artificial Intelligence, Ewha Womans University <sup>2</sup>Division of Artificial Intelligence & Software, Ewha Womans University

{guma017, cdh030303, serena2140, 2391039, lee.jiyoung}@ewha.ac.kr

Abstract. Audio-visual multi-segment grounding (AV-MSG) in untrimmed videos, reasoning over audio-visual evidence and predicting multiple segments for a query, is a fundamental problem but remains challenging. Visual-only models overlook complementary acoustic cues, while audio-visual models often fail to calibrate the number of events—a phenomenon we refer to as count miscalibration. We present TiTok, an audio-visual large language model (AV-LLM) that localizes an arbitrary number of temporal event segments for each query. For precise boundary prediction, we introduce the Time Token Interleaving (TTI) method, which explicitly injects special time tokens into the audio-visual stream to align input-side temporal perception with output-side temporal prediction. We further propose decoupled, multi-segment-oriented rewards for reinforcement learning, consisting of global, local, count, precision, and format rewards, optimized with Group reward-Decoupled Normalization Policy Optimization (GDPO). To assess the performance on AV-MSG, we establish a new UnAV-100-based evaluation protocol, and propose the CountF1 metric for quantifying count miscalibration that overlap metrics fail to capture. TiTok reaches 65.7 mIoU and 0.58 CountF1, achieving state-of-the-art performance. Our code is available at https://github.com/AI-Living-Lab/TiTok.

Keywords: Multi-Segment Temporal Grounding · Audio-Visual LLM · Reinforcement Learning

## 1 Introduction

Video Temporal Grounding (VTG) [33] localizes the temporal intervals of an untrimmed video corresponding to a natural-language query. Recent VTG systems have increasingly built on Multimodal Large Language Models (MLLMs) for their cross-modal reasoning and zero-shot generalization [13,15,32], yet most operate almost entirely on the visual stream [15, 22, 25, 32], treating audio as auxiliary or discarding it.

![](images/fb2e56335d009c2a7a8e8efac98e22afadcbc298e9abc64201a48b0cae9b7d00.jpg)  
Fig. 1: Dense event captioning vs. audio-visual multi-segment grounding. For the same video, dense event captioning (top) describes every event, whereas audiovisual multi-segment grounding (bottom) localizes only the occurrences of the queried event.

This visual-centric design overlooks an important regime in which audio is indispensable. Many real-world events (e.g., a dog barking, a gunshot) are often more clearly delimited temporally by sound than by appearance, and may recur within a single video. Therefore, given a query event that can appear multiple times, Audio-Visual Multi-Segment Grounding (AV-MSG) identifies all of its temporal occurrences. Fig. 1 contrasts AV-MSG with dense event captioning [18]. The latter localizes all events with timestamps without a query, whereas our task determines which event instances match a given query.

A growing line of work has extended temporal grounding to the audio-visual setting [4, 5, 9, 21, 30]. However, accurately counting recurring events remains challenging. Some models [4,9] have been designed primarily to generate a single event, and other methods [22, 30] have been shown to collapse occurrences into a single interval or over-segment them. We attribute these failures to existing objectives that mainly optimize temporal overlap, but do not explicitly penalize errors in event cardinality (i.e., missing or extra event segments). We argue that this count miscalibration is a key bottleneck for robust AV-MSG, a setting existing methods address only partially as described in Tab. 1.

In this paper, we propose TiTok, which addresses count miscalibration within the AV-LLM framework. Since existing AV-LLMs [23,28,29,36,39] typically lack an explicit timestamp representation, we resolve this with Time Token Interleaving (TTI), which combines special time tokens with their interleaving into the audio-visual stream. The special time tokens let the model treat time as an explicit, generable quantity distinct from ordinary numerals, and interleaving them marks the absolute time of the audio and visual tokens they precede.

We train TiTok through two stages: a cold-start supervised fine-tuning (SFT) stage that teaches the special time tokens, followed by GDPO-based RL that builds the AV-MSG ability. However, the reward designs [7,32] tailored for VTG often fail to account for the correct number of events, since a format and an IoU reward alone do not supervise the event count in the multi-segment setting. We therefore design multi-segment-oriented rewards (i.e., global, local, count, precision, and format). Combining such heterogeneous rewards into a single scalar suppresses their fine-grained localization signals, so we optimize them with Group reward-Decoupled Normalization Policy Optimization (GDPO) [19], which normalizes each reward independently, and further remove the KL penalty and apply asymmetric clipping for freer exploration. Given that existing benchmarks lack audio-aware multi-segment annotations, we introduce an evaluation protocol on UnAV-100 [10]. TiTok outperforms AVicuna [30] and ChronusOmni [4], which is fine-tuned on the same training data.

Table 1: Capability comparison. Multi-Seg marks whether a method natively emits an arbitrary number of segments; Time Token marks whether it represents time with dedicated tokens (special time tokens) rather than plain-text numbers or percentiles.
<table><tr><td>Method</td><td></td><td></td><td></td><td>Audio Video Multi-Seg Time Token</td></tr><tr><td>ChronusOmni [4], ARC-Hunyuan [9]</td><td>V</td><td>√</td><td>X</td><td>X</td></tr><tr><td>TempR1 [34], MUSEG [22]</td><td>X</td><td>√</td><td>√</td><td>X</td></tr><tr><td>VTG-LLM [12]</td><td>X</td><td>√</td><td>X</td><td>√</td></tr><tr><td>AVicuna [30]</td><td>√</td><td>√</td><td>√</td><td>X</td></tr><tr><td>TiTok (Ours)</td><td>1</td><td>1</td><td>J</td><td>J</td></tr></table>

Our contributions are summarized as follows:

– We propose TiTok, which introduces TTI to interleave absolute timestamps as special time tokens into the audio-visual stream, giving the model an explicit, generable time axis for localizing multiple events.

– We design decoupled, multi-segment-oriented rewards optimized by a GDPObased RL pipeline, which preserves the fine-grained localization signals that joint normalization would otherwise wash out.

– We establish a UnAV-100-based evaluation protocol for AV-MSG that includes CountF1 to quantify count miscalibration, on which TiTok attains the most accurate AV-MSG results among the compared models.

## 2 Related Work

## 2.1 Audio-Visual Temporal Grounding with LLMs

With the rapid growth of MLLMs [13, 36], recent VTG methods [12, 13, 15, 25, 32] have incorporated LLMs [24, 31, 37] to leverage their remarkable reasoning capabilities. Most methods heavily rely on the visual stream to predict a single event timestamp, and are evaluated on benchmarks annotated from visual cues alone [8, 18]. Although some studies have extended temporal grounding to the audio-visual setting [4,5,9,21] or multi-segment prediction [22,34], none addresses both jointly. While AVicuna [30] is capable of emitting multiple segments for a query, its predicted segment counts deviate substantially from the ground truth, leaving AV-MSG largely unsolved.

## 2.2 Temporal Representation in MLLMs

In MLLMs, time is handled along two axes: (i) on the input side, frame numbers or indices are overlaid on the frames [9, 35], representations of timestamps are fused with the visual tokens [25], or time is injected as natural-language text [3, 4, 40]. In AV-LLMs, audio and visual tokens can be interleaved along a shared temporal axis [29, 36]. (ii) On the output side, temporal representations can be divided into relative and absolute time. Relative-time methods represent boundaries using percentiles of video duration [30], indices of uniformly sampled frames [15], or dedicated tokens for positions normalized by video length [6, 16]. These representations maintain a fixed output range across videos of diferent lengths, but the time each unit spans grows with video length, and recovering timestamps in seconds requires conversion. In contrast, absolute-time methods express boundaries directly in seconds. Plain-text timestamps [3, 4, 25, 32] can represent times without adding new tokens but share the token space of ordinary numbers. Dedicated absolute-time tokens [12] instead use a controlled temporal vocabulary that separates time from ordinary text and reduces quantization error. However, dedicated time tokens [6, 12, 16] have been used for visual-only VTG, not interleaved into an audio-visual stream. TiTok addresses this gap with TTI, using the same time tokens at both the input and output.

## 2.3 Reinforcement Learning for Temporal Grounding

Reinforcement learning (RL) has become the dominant approach for strengthening reasoning and temporal understanding in MLLMs [11,17]. Many VTG works combine a format reward with an IoU reward for prediction accuracy [2, 7, 32], while others tune rewards to their task [20–22]. These works typically optimize such distinct rewards together with Group Relative Policy Optimization (GRPO) [27], leaving its suitability for this multi-reward setting largely unexamined. Meanwhile, GDPO [19] reveals that GRPO collapses distinct reward combinations into an identical advantage by normalizing their sum. This ambiguity weakens the training signal and may lead to suboptimal convergence or early training failure. Therefore, our method optimizes the designed rewards with GDPO for training.

## 3 Method

## 3.1 Problem Statement

Given an untrimmed video with visual and audio streams, and a natural-language query describing an event, AV-MSG aims to localize every temporal interval in which the queried event occurs. Let the video be paired with a query q, and let the ground-truth be a set of $N _ { \mathrm { g t } }$ segments $\mathcal { G } = \{ G _ { i } \} _ { i = 1 } ^ { N _ { \mathrm { g t } } }$ , where each $G _ { i } = ( s _ { i } , e _ { i } )$ denotes a start time and an end time. A model must predict a set $\mathcal { P } = \{ P _ { j } \} _ { j = 1 } ^ { N _ { \mathrm { p r e d } } }$ of segments. The number of predicted segments $N _ { \mathrm { p r e d } }$ is not given in advance but must itself be inferred. We refer to samples with $N _ { \mathrm { g t } } = 1$ as single-segment samples and those with $N _ { \mathrm { g t } } \geq 2$ as multi-segment samples.

![](images/78bb3e57357bb2ce2e9b4fadee854491956c75846df8da27175d55056d3b9081.jpg)  
Fig. 2: The overall pipeline of TiTok. (Left) Cold-start SFT stage. (Right) GDPObased RL stage with TTI and the five rewards.

## 3.2 Time Token Interleaving

We address the lack of absolute time representation in AV-LLMs with TTI. As shown in Fig. 2, we interleave audio and visual tokens in temporal order, following video-SALMONN 2 plus [29]. Sampling a video into frames thus yields a sequence of audio-visual blocks that each bundle the audio and visual tokens at one timestamp, and into each block we insert that timestamp composed of special time tokens.

Each timestamp is encoded as a composition of tokens rather than a single opaque symbol. Following VTG-LLM [12], we add ten digit tokens <t0> ∼ <t9> and a decimal-point token <tdot> to the vocabulary and initialize their embeddings and LM head rows from the corresponding digits and decimal point. A timestamp is written with three zero-padded integer digits and one decimal digit (16.7 seconds → <t0><t1><t6><tdot><t7>), which avoids the concept shift between the numerical and temporal meanings of digits.

TTI uses these special time tokens not only for output timestamps but also in the input stream, where each inserted timestamp marks the absolute time of the audio and visual tokens that follow it. The model is therefore able to locate the times at which the audio-visual evidence of the queried event appears and disappears, which can enable it to place segment boundaries, distinguish recurring events as separate segments and thus mitigate count miscalibration. Because the output timestamps share this format, the time perceived at the input and the time predicted at the output are expressed in the same tokens, which supports both boundary prediction and count calibration.

## 3.3 Two-Stage Training

Stage 1: Cold-Start SFT. The special time tokens are new to the pretrained vocabulary, so the backbone rarely generates them in a valid format. When

GDPO is applied directly to the backbone, most responses cannot be parsed into segments and receive zero rewards. We therefore first perform a short coldstart SFT stage so the model can read and generate the special time tokens for successful rollout generation.

Stage 2: GDPO-based Reinforcement Learning. Once the special time token format is secured, we build AV-MSG ability through RL. For a query $q ,$ we sample $N _ { \mathrm { r o l l } }$ responses $\big \{ o _ { 1 } , \dotsc , o _ { N _ { \mathrm { r o l l } } } \big \}$ and score each with the K reward terms $\bar { \mathbf { R } } = \{ r ^ { ( 1 ) } , \ldots , \bar { r } ^ { ( K ) } \}$ of Sec. 3.4. Whereas GRPO [27] normalizes the sum of these terms within the group, we adopt GDPO [19], which normalizes each term on its own and combines them by weight before normalizing once more across the batch B:

$$
\hat { A } _ { i } ^ { ( k ) } = \frac { r _ { i } ^ { ( k ) } - \mathrm { m e a n } ( \{ r _ { j } ^ { ( k ) } \} _ { j = 1 } ^ { N _ { \mathrm { r o l l } } } ) } { \mathrm { s t d } ( \{ r _ { j } ^ { ( k ) } \} _ { j = 1 } ^ { N _ { \mathrm { r o l l } } } ) } , \qquad \tilde { A } _ { i } = \sum _ { k = 1 } ^ { K } w ^ { ( k ) } \hat { A } _ { i } ^ { ( k ) } ,\tag{1}
$$

$$
\hat { A } _ { i } = \frac { \tilde { A } _ { i } - \operatorname* { m e a n } ( \{ \tilde { A } _ { j } \} _ { j \in B } ) } { \mathrm { s t d } ( \{ \tilde { A } _ { j } \} _ { j \in B } ) } ,\tag{2}
$$

where ${ \hat { A } } _ { i } ^ { ( k ) }$ is the group-normalized advantage of reward $k , w ^ { ( k ) }$ is its combination weight, and ${ \bar { \hat { A } } } _ { i }$ is the weighted sum of the per-reward advantages, which Eq. (2) normalizes over the batch B into ${ \hat { A } } _ { i }$ . Normalizing each term separately keeps the fine-grained localization signals from being suppressed by easier-tolearn terms such as the format reward, while batch-level normalization prevents the combined advantage from growing with K [19].

Using the resulting advantage ${ \hat { A } } _ { i }$ , we optimize a clipped surrogate objective over the token-level probability ratio $\begin{array} { r } { \rho _ { i , t } ( \theta ) = { \frac { \pi _ { \theta } \left( o _ { i , t } | q , o _ { i , < t } \right) } { \pi _ { \theta _ { \mathrm { o l d } } } \left( o _ { i , t } | q , o _ { i , < t } \right) } } } \end{array}$ . For training stability and exploration, we further adopt two techniques from Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO) [38].

– Clip-Higher. A tight upper clipping bound suppresses the probability increase of low-probability exploration tokens, causing the policy entropy to collapse [38]. We therefore decouple the clipping range into a lower bound $\epsilon _ { \mathrm { l o w } }$ and an upper bound $\epsilon _ { \mathrm { h i g h } }$ , and set the upper bound larger to leave more room for low-probability exploration tokens to grow.

– Removing KL divergence. Since the model must depart substantially from the cold-start SFT policy to acquire the behavior required for AV-MSG, we remove the KL penalty with respect to the reference policy to allow freer exploration [38].

Therefore, the final training objective is

$$
\begin{array} { r } { \mathcal { I } _ { \mathrm { T i T o k } } ( \theta ) = \mathbb { E } \Big [ \frac { 1 } { N _ { \mathrm { r o l l } } } \sum _ { i = 1 } ^ { N _ { \mathrm { r o l l } } } \frac { 1 } { | o _ { i } | } \sum _ { t = 1 } ^ { | o _ { i } | } \operatorname* { m i n } \Big ( \rho _ { i , t } ( \theta ) \hat { A } _ { i } , } \\ { \mathrm { c l i p } \big ( \rho _ { i , t } ( \theta ) , 1 - \epsilon _ { \mathrm { l o w } } , 1 + \epsilon _ { \mathrm { h i g h } } \big ) \hat { A } _ { i } \Big ) \Big ] _ { \mathcal { M } } } \end{array}\tag{3}
$$

where ${ \hat { A } } _ { i }$ is the GDPO advantage from Eq. (2).

## 3.4 Reward Design

We decompose the reward into five objective-specific terms $\mathbf { R } = \{ r _ { \mathrm { g l o b a l } } , r _ { \mathrm { l o c a l } } $ $r _ { \mathrm { c o u n t } } , r _ { \mathrm { p r e c } } , r _ { \mathrm { f o r m a t } } \}$ , each designed to lie in [0, 1]. Following the default of GDPO [19], we weight them equally $( w ^ { ( k ) } = \dot { 1 }$ in Eq. (1)) so that each independently normalized reward contributes at full strength without manual tuning.

Global Reward. This evaluates the overall agreement between the predicted and ground-truth segments. We merge overlapping predicted segments into a union $\widetilde { \mathcal { P } } = \{ \widetilde { P } _ { m } \} _ { m = 1 } ^ { M }$ and compute its temporal precision Prec and recall Rec against the ground truth, using their F1 as the reward:

$$
\begin{array} { r l } & { \mathrm { P r e c } = \frac { \sum _ { i } \sum _ { m } \left| { \cal G } _ { i } \cap \widetilde { { \cal P } } _ { m } \right| } { \sum _ { m } | \widetilde { { \cal P } } _ { m } | } , ~ \mathrm { R e c } = \frac { \sum _ { i } \sum _ { m } \left| { \cal G } _ { i } \cap \widetilde { { \cal P } } _ { m } \right| } { \sum _ { i } | { \cal G } _ { i } | } , } \\ & { ~ r _ { \mathrm { g l o b a l } } = \frac { 2 \mathrm { P r e c } \cdot \mathrm { R e c } } { \mathrm { P r e c } + \mathrm { R e c } } . } \end{array}\tag{4}
$$

Local Reward. While the global reward looks at agreement over the entire event region, the local reward looks at how accurately each individual segment is placed. We process the predicted segments in their output order and pair each with the unpaired ground-truth segment of the highest Generalized IoU (GIoU) [26]. GIoU provides a non-zero signal even when two segments do not overlap, and following MUSEG [22], we rescale it to [0, 1]:

$$
\operatorname { G I o U } ( G _ { i } , P _ { j } ) = \frac { 1 } { 2 } \left( 1 + \frac { | G _ { i } \cap P _ { j } | } { | G _ { i } \cup P _ { j } | } - \frac { | C \setminus ( G _ { i } \cup P _ { j } ) | } { | C | } \right) ,\tag{5}
$$

where $C$ is the smallest interval containing both $G _ { i }$ and $P _ { j }$ . Denoting the set of paired predictions by $s$ and the resulting pairing by $\sigma ,$ we average GIoU with max $( N _ { \mathrm { g t } } , N _ { \mathrm { p r e d } } )$ as the denominator so that any count mismatch is reflected:

$$
r _ { \mathrm { l o c a l } } = \frac { 1 } { \mathrm { m a x } ( N _ { \mathrm { g t } } , N _ { \mathrm { p r e d } } ) } \sum _ { j \in S } \mathrm { G I o U } \big ( G _ { \sigma ( j ) } , P _ { j } \big ) ,\tag{6}
$$

where the number of paired segments is min $( N _ { \mathrm { g t } } , N _ { \mathrm { p r e d } } )$ . The unpaired segments thus contribute nothing to the numerator but remain in the denominator. Pairing each prediction with its best match rather than with the ground-truth segment at the same temporal position avoids the cascading misalignment in which a single misplaced segment throws of all subsequent pairs.

Count Reward. We evaluate how well the number of predicted events matches the number of ground-truth segments:

$$
r _ { \mathrm { c o u n t } } = 1 - \frac { \left| N _ { \mathrm { p r e d } } - N _ { \mathrm { g t } } \right| } { \operatorname* { m a x } \left( N _ { \mathrm { p r e d } } , N _ { \mathrm { g t } } \right) } .\tag{7}
$$

This term penalizes both under- and over-prediction. However, it can give a prediction that collapses two occurrences into one segment $( N _ { \mathrm { g t } } = 2 , N _ { \mathrm { p r e d } } = 1 )$ the same reward as one that predicts two segments for four occurrences $( N _ { \mathrm { g t } } =$ 4, $N _ { \mathrm { p r e d } } = 2 )$ , although only the former ignores the recurrence entirely. We therefore set $r _ { \mathrm { c o u n t } } ~ = ~ 0$ when $N _ { \mathrm { g t } } \ \geq \ 2$ and $N _ { \mathrm { p r e d } } = 1$ to strongly penalize collapsing several events into a single segment.

Precision Reward. In Eq. (4), $r _ { \mathrm { g l o b a l } }$ combines precision with recall through F1, and since predicting more time can only increase recall, inflating boundaries is penalized there only partially. We therefore keep precision as a separate term for precision reward, $r _ { \mathrm { p r e c } } = \mathrm { P r e c }$ , so that GDPO assigns it an independently normalized advantage. Whereas the count reward keeps the number of segments correct, the precision reward keeps them tightly placed on the ground truth, discouraging the shortcut of matching the count while inflating boundaries.

Format Reward. This evaluates whether the model outputs the correct special time tokens. A reward of 1 is assigned only when the entire response consists of well-formed segments of the form From <tX><tX><tX><tdot><tX> to <tX><tX><tX><tdot><tX>., and 0 otherwise:

$$
r _ { \mathrm { f o r m a t } } ( o _ { i } ) = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f ~ } o _ { i } \mathrm { h a s ~ t h e ~ c o r r e c t ~ f o r m a t } } \\ { 0 , } & { \mathrm { o t h e r w i s e } } \end{array} \right.\tag{8}
$$

Because any number of well-formed segments earns 1, this reward enforces strict formatting without discouraging multi-segment predictions.

## 4 Experiments

## 4.1 Experimental Settings

Datasets. We build an AV-MSG protocol on UnAV-100 [10], a densely annotated audio-visual event dataset spanning 100 categories. For each video we group annotated intervals by event category into one query–interval sample per category: the category name fills a fixed template (“At what point in the video does ⟨event⟩ occur in terms of both video and audio?”) and its chronologically sorted intervals form the target as compositional special time tokens. A category recurring within a video thus yields a multi-segment ground truth (Fig. 3), while a category that occurs once yields a single-segment one. For training we also include the moment-retrieval dataset Charades-STA [8], which pairs naturallanguage queries with video-only inputs and a single target segment. The RL training set contains 10,358 samples: 7,207 from UnAV-100, of which 4,055 are multi-segment, and 3,151 from Charades-STA, giving a single- to multi-segment ratio of about 6:4. All evaluation is in-domain on the held-out UnAV-100 test split and on the Charades-STA test split for an additional comparison in Tab. 3.

![](images/7ff6d39a49bb01cc57f052ba255ee7b8cac90721d2c6daf5c008b52578b5b47d.jpg)  
Fig. 3: UnAV-100 reconstruction. All intervals of one event category become a single query; a recurring event yields a multi-segment special time token target.

Implementation Details. We build TiTok on video-SALMONN 2 plus (7B) [29] as the backbone and initialize the GDPO-based RL stage from the SFT checkpoint. The SFT stage uses UnAV-100 training samples only and runs for 250 steps with an efective batch size of 8 and a learning rate of $1 0 ^ { - 4 }$ . It updates the visual and audio aligners and the special time token rows of the token embedding and the LM head while applying Low-Rank Adaptation (LoRA) [14] (r=16, α=16) to the LLM, and the vision and audio encoders stay frozen. In the RL stage we freeze all of these modules and adapt only the LLM through LoRA (r=32, α=64) for 2,000 steps with an efective batch size of 4 and a learning rate of $1 0 ^ { - 5 }$ . For each prompt we sample $N _ { \mathrm { r o l l } } { = } 8$ responses with a maximum output length of 256, and perform µ=2 policy update iterations per rollout batch with Clip-Higher $( \epsilon _ { \mathrm { l o w } } { = } 0 . 2 , \epsilon _ { \mathrm { h i g h } } { = } 0 . 2 8 )$

Metrics. We report mIoU, the per-sample temporal IoU between predicted and ground-truth segment unions, averaged over samples. As over-prediction inflates recall when the predicted count is unconstrained, we also report F1@k [22] where $R _ { k } \ \left( P _ { k } \right)$ is the fraction of ground-truth (predicted) segments in the test set whose best-overlapping counterpart has a temporal IoU of at least k:

$$
\mathrm { F } 1 @ k = \frac { 2 P _ { k } \cdot R _ { k } } { P _ { k } + R _ { k } } , \qquad k \in \{ 0 . 3 , 0 . 5 , 0 . 7 \} .\tag{9}
$$

Since overlap metrics do not directly measure the number of segments, we finally introduce CountF1 as the harmonic mean of two count-calibration accuracies for opposite failure modes. The under-segmentation accuracy (USA) on multisegment samples $( S _ { \mathrm { m u l t i } } { = } \{ i : N _ { \mathrm { g t } } ^ { ( i ) } \geq 2 \} )$ ) is high when the model does not predict too few segments, and the over-segmentation accuracy (OSA) on single-segment samples $( S _ { \mathrm { s i n g l e } } = \{ i : N _ { \mathrm { g t } } ^ { ( i ) } = 1 \} )$ is high when the model does not split one segment into spurious extras. For USA, we first compute the count coverage ratio (CR) as the average closeness of the predicted count to the ground-truth count over multi-segment samples. We rescale CR against a baseline $b ,$ the expected CR of a trivial model that always predicts a single segment. Since b depends only on the annotations and not on any evaluated model, it can be recomputed for any benchmark $( b = 0 . 3 9 7$ on UnAV-100). Subtracting b from CR and rescaling to [0, 1] yields USA. For single-segment samples the ground-truth count is fixed at one, so we define OSA directly as the fraction of samples predicted with exactly one segment:

$$
\mathrm { C R } = \frac { 1 } { \lvert S _ { \mathrm { m u l t i } } \rvert } \sum _ { i \in S _ { \mathrm { m u l t i } } } \frac { \operatorname* { m i n } \bigl ( N _ { \mathrm { g t } } ^ { ( i ) } , N _ { \mathrm { p r e d } } ^ { ( i ) } \bigr ) } { \operatorname* { m a x } \bigl ( N _ { \mathrm { g t } } ^ { ( i ) } , N _ { \mathrm { p r e d } } ^ { ( i ) } \bigr ) } , \qquad b = \frac { 1 } { \lvert S _ { \mathrm { m u l t i } } \rvert } \sum _ { i \in S _ { \mathrm { m u l t i } } } \frac { 1 } { N _ { \mathrm { g t } } ^ { ( i ) } } .\tag{10}
$$

$$
\mathrm { U S A } = \operatorname* { m a x } \Bigl ( 0 , \frac { \mathrm { C R } - b } { 1 - b } \Bigr ) , \qquad \mathrm { O S A } = \frac { 1 } { | S _ { \mathrm { s i n g l e } } | } \sum _ { i \in S _ { \mathrm { s i n g l e } } } \mathbf { 1 } \Bigl [ N _ { \mathrm { p r e d } } ^ { ( i ) } = 1 \Bigr ] .\tag{11}
$$

$$
\mathrm { C o u n t F 1 } = \frac { \mathrm { 2 U S A \cdot O S A } } { \mathrm { U S A } + \mathrm { O S A } } ,\tag{12}
$$

where CountF1 is set to 0 when both USA and OSA are 0. CountF1 lies in [0,1] and is 1 only when a model neither collapses recurring events into too few segments nor inflates single events into too many, so higher CountF1 indicates better count calibration.

Baseline Prompting and Parsing. All baselines receive the same query template as TiTok, together with examples of their native timestamp format (e.g., second{X.X} for ChronusOmni), so that each model answers in the form it was trained to produce. Video frames are likewise sampled using the default settings of each oficial implementation. Although ChronusOmni is designed for single-event grounding, we find that it can return several segments under our query template, whereas ARC-Hunyuan always returns one. We convert the responses into segments with format-tolerant rule-based parsers, which succeed on 98.9% of responses on average, and score unparseable responses as empty predictions. To verify that parsing does not understate any baseline, we re-parse 100 randomly sampled responses per model with an independent LLM-based parser (Qwen2.5-14B-Instruct [37]). The two parsers extract the same number of segments for 98.3% of responses, and re-scoring changes mIoU by at most 0.87 points, suggesting that the reported scores reflect model predictions rather than parsing.

Table 2: Results on UnAV-100. Setting column marks each model as ZS (zeroshot) or FT (fine-tuned); A and V mark audio and video support. † marks results we reproduced from released checkpoints.
<table><tr><td>Model</td><td>Setting</td><td></td><td>A V</td><td>mIoU</td><td>F1@0.3</td><td>F1@0.5</td><td>F1@0.7</td><td>CountF1</td></tr><tr><td>Qwen2.5-Omni†</td><td>ZS</td><td>√</td><td>√</td><td>12.6</td><td>14.2</td><td>9.3</td><td>6.4</td><td>0.07</td></tr><tr><td>ARC-Hunyuan†</td><td>ZS</td><td>√</td><td>√</td><td>45.8</td><td>43.6</td><td>33.3</td><td>26.1</td><td>0.00</td></tr><tr><td>MUSEG†</td><td>ZS</td><td>X</td><td>√</td><td>49.1</td><td>51.5</td><td>36.8</td><td>23.0</td><td>0.22</td></tr><tr><td>ChronusOmni†</td><td>ZS</td><td>√</td><td>√</td><td>60.9</td><td>52.7</td><td>34.8</td><td>21.8</td><td>0.39</td></tr><tr><td>AVicuna†</td><td>FT</td><td>√</td><td>√</td><td>50.4</td><td>31.5</td><td>23.8</td><td>18.6</td><td>0.08</td></tr><tr><td> $\mathrm { C r a b ^ { + \dagger } }$ </td><td>FT</td><td>√</td><td>√</td><td>50.7</td><td>48.4</td><td>36.7</td><td>27.7</td><td>0.24</td></tr><tr><td>ChronusOmni</td><td>FT</td><td></td><td>√√</td><td>63.4</td><td>57.5</td><td>41.8</td><td>27.9</td><td>0.50</td></tr><tr><td>TiTok (Ours)</td><td>FT</td><td></td><td>√√</td><td>65.7</td><td>63.8</td><td>50.3</td><td>38.6</td><td>0.58</td></tr></table>

Table 3: Results on Charades-STA. Both models are fine-tuned on Charades-STA.
<table><tr><td>Model</td><td>mIoU</td><td>F1@0.3</td><td>F1@0.5</td><td>F1@0.7</td><td>OSA</td></tr><tr><td>ChronusOmni</td><td>48.6</td><td>74.7</td><td>50.1</td><td>24.8</td><td>0.98</td></tr><tr><td>TiTok</td><td>56.3</td><td>80.3</td><td>66.1</td><td>39.7</td><td>1.00</td></tr></table>

## 4.2 Quantitative Results

Performance on UnAV-100. Among the compared methods, AVicuna and Crab<sup>+</sup> [1] are released AV-LLMs trained on UnAV-100. To enable a fairer indomain comparison, we also fine-tune ChronusOmni, the strongest zero-shot baseline in mIoU and CountF1, with LoRA on the same UnAV-100 and Charades-STA training data as TiTok (Sec. 4.1). As shown in Tab. 2, TiTok achieves the best results on every metric among the compared models. MUSEG, AVicuna, and Crab<sup>+</sup> support multi-segment prediction but show lower CountF1 and overlap scores across the board. Against the fine-tuned ChronusOmni, TiTok attains higher mIoU and outperforms it by a wider margin in F1 and CountF1. As mIoU compares the unions of predicted and ground-truth segments, a prediction that merges nearby occurrences into one segment can still obtain a high mIoU. F1@k and CountF1 instead require each occurrence to be predicted as a separate segment, so the wider gap on these metrics indicates that TiTok separates recurring events more accurately rather than covering the same region more tightly.

Performance on Charades-STA. To examine whether TiTok tends to oversegment in single-segment grounding, we additionally evaluate it on Charades-STA against the fine-tuned ChronusOmni used in the preceding UnAV-100 evaluation. As shown in Tab. 3, neither model tends to split a single event into multiple segments, and TiTok outperforms ChronusOmni on every overlap metric by a much wider margin at the stricter F1 thresholds. TiTok thus remains efective in single-segment visual-only grounding even though it is trained to emit multiple segments.

![](images/0fa6ae682f95ab6c4926761d221b216b1c90ed0ae0a7785c6d0f0f3a7dee171e.jpg)  
Fig. 4: Qualitative comparison on UnAV-100. Two queries on the same clip: “playing volleyball” (top) and “people clapping” (bottom).

## 4.3 Qualitative Results

Fig. 4 illustrates count miscalibration on a clip with two recurring queries: playing volleyball is defined mainly by visual cues while people clapping is barely visible but clearly audible. Although MUSEG supports multi-segment prediction, its visual-only input provides almost no cue for people clapping, so it localizes the event to the middle of the clip as the most plausible position. Yet audio input alone does not prevent count miscalibration, as AVicuna collapses the two occurrences of playing volleyball into a single interval and over-segments people clapping while ARC-Hunyuan localizes only the first occurrence of each query. The fine-tuned ChronusOmni predicts the correct count for both queries, but its two segments for playing volleyball fill the gap between the occurrences and its second segment for people clapping lands well before the actual occurrence. TiTok yields the segment count and boundaries closest to the ground truth for both queries, localizing the recurrences as distinct segments.

## 4.4 Ablation Studies

Efect of TTI. Removing the special time tokens from the input only (w/o interleaving) and replacing them with plain-text timestamps (w/o special tokens) both lower every metric but in diferent patterns. Compared with w/o special tokens, $w / o$ interleaving drops more in mIoU and by a wider margin as the F1 threshold becomes stricter, whereas $w / o$ special tokens drops more in CountF1. Interleaving thus contributes more to boundary placement and the special time tokens more to count calibration. Since $w / o$ special tokens still uses the same format at the input and output, its lower CountF1 shows that aligning the two sides alone is not suficient for count calibration. TTI therefore requires both components as each addresses a diferent type of error.

![](images/047b72fb0a44c85d7b300c64e48f5fb7585aed15866fdd92d7955260f6d9392f.jpg)

Table 4: Ablation on UnAV-100. Each row removes or replaces a component.
<table><tr><td>Setting</td><td>mIoU</td><td>F1@0.3</td><td>F1@0.5</td><td>F1@0.7</td><td>CountF1</td></tr><tr><td>TiTok</td><td>65.7</td><td>63.8</td><td>50.3</td><td>38.6</td><td>0.58</td></tr><tr><td>w/o special tokens</td><td>63.9</td><td>60.6</td><td>47.3</td><td>35.5</td><td>0.51</td></tr><tr><td>w/o interleaving</td><td>61.2</td><td>59.8</td><td>45.6</td><td>32.7</td><td>0.53</td></tr><tr><td>w/o audio</td><td>57.8</td><td>53.6</td><td>40.4</td><td>29.4</td><td>0.52</td></tr><tr><td> $\mathrm { w / o ~ R L }$ </td><td>56.8</td><td>26.9</td><td>19.1</td><td>14.4</td><td>0.16</td></tr><tr><td> $\mathrm { w } / \mathrm { o } \ r _ { \mathrm { p r e c } }$ </td><td>65.5</td><td>62.9</td><td>49.6</td><td>36.7</td><td>0.59</td></tr><tr><td> $\mathrm { w } / \mathrm { o } \ r _ { \mathrm { c o u n t } }$ </td><td>63.9</td><td>64.0</td><td>50.3</td><td>36.9</td><td>0.57</td></tr><tr><td> $\mathrm { w } / \mathrm { o } \ r _ { \mathrm { g l o b a l } }$ </td><td>61.4</td><td>60.7</td><td>46.4</td><td>33.7</td><td>0.57</td></tr><tr><td> $\mathrm { w / \ G R P O }$ </td><td>64.4</td><td>64.2</td><td>50.1</td><td>36.3</td><td>0.62</td></tr></table>

Fig. 5: Reward curves over GDPO training.

Efect of Audio. Removing audio while keeping TTI and RL (w/o audio) lowers every metric, and unlike $w / o$ interleaving, its F1 drop does not grow at stricter thresholds. Audio is therefore needed to detect the occurrences of the queried event and not only to refine their boundaries. Yet w/o audio still exceeds the fine-tuned ChronusOmni in F1@0.7 and CountF1, suggesting that TiTok’s advantage in precise localization and count calibration comes from TTI and RL rather than from audio.

RL Design Choice. Each ablated reward term contributes to a diferent aspect of grounding quality. Among the reward ablations, removing $r _ { \mathrm { g l o b a l } }$ (w/o $r _ { g l o b a l } )$ causes the largest drop in mIoU and lowers all F1 scores as well. This is because $r _ { \mathrm { g l o b a l } }$ directly rewards the overall overlap between the predicted and ground-truth segments. In contrast, removing r<sub>count</sub> $( w / o \ r _ { c o u n t } )$ lowers mIoU and F1@0.7 but leaves F1 at the loose thresholds nearly unchanged because F1 only checks whether each prediction’s IoU exceeds the threshold. With CountF1 also nearly unchanged, the model without $r _ { \mathrm { c o u n t } }$ thus misaligns segments slightly rather than miscounting them, likely because the local reward still penalizes count mismatch through its denominator. Conversely, removing $r _ { \mathrm { p r e c } } ~ ( w / o ~ r _ { p r e c } )$ leaves mIoU nearly unchanged but lowers F1 at every threshold, likely because $r _ { \mathrm { g l o b a l } }$ still rewards precision over segment unions while $r _ { \mathrm { p r e c } }$ keeps each segment tightly placed. With the same reward design, replacing GDPO with GRPO (w/ GRPO) raises CountF1 but lowers mIoU and F1@0.7, so neither optimizer is consistently better. Both optimizers still improve every metric over the cold-start SFT model (w/o RL) by a large margin, suggesting that the gains of RL with our reward design do not depend on GDPO.

## 4.5 Training Dynamics

Fig. 5 tracks the four reward terms over GDPO training. The global and precision rewards rise sharply in the early steps and then plateau for both singleand multi-segment samples, so the model learns to place its segments on the ground-truth region early in training. For the local reward, the single-segment curve sits above the multi-segment one, though the latter also improves slightly over training. This gap follows from Sec. 3.4, where multi-segment matching is penalized for any count mismatch (Eq. 6) while a single target is easier to cover. The single-segment count reward stays near its ceiling from the start, while the multi-segment count reward begins well below it and fluctuates during training but trends upward even after the global and precision rewards have plateaued, suggesting that RL continues to reduce count miscalibration on recurring events.

## 5 Conclusion

We propose TiTok, which addresses the count miscalibration problem in AV-LLMs for precise AV-MSG. It combines TTI with multi-segment-oriented rewards optimized with GDPO. To make this problem measurable, we also introduce a UnAV-100-based evaluation protocol and CountF1, whose baseline depends only on the annotations and can thus be recomputed for other benchmarks. On this protocol, TiTok achieves the highest overlap and CountF1 scores among the compared baselines. These results suggest that reliable AV-MSG benefits from both an explicit representation of time and an explicit objective for the event count.

Acknowledgements. This research was supported by Global - Learning & Academic research institution for Master’s·PhD students, and Postdocs(G-LAMP) Program of the National Research Foundation of Korea(NRF) grant funded by the Ministry of Education(No. RS-2025-25442252).

## References

1. Cai, D., Du, H., Zhou, C., Chen, X., Guo, D., Zhang, H., Li, X., Hu, D.: Crab<sup>+</sup>: A scalable and unified audio-visual scene understanding model with explicit cooperation. arXiv preprint arXiv:2603.04128 (2026)

2. Chen, R., Luo, T., Fan, Z., Zou, H., Feng, Z., Xie, G., Zhang, H., Wang, Z., Liu, Z., Zhang, H.: Datasets and recipes for video temporal grounding via reinforcement learning. In: EMNLP Industry Track. pp. 983–992 (2025)

3. Chen, S., Lan, X., Yuan, Y., Jie, Z., Ma, L.: TimeMarker: A versatile Video-LLM for long and short video understanding with superior temporal localization ability. arXiv preprint arXiv:2411.18211 (2024)

4. Chen, Y., Wu, Y., Guan, K., Ren, Y., Wang, Y., Song, R., Ru, L.: ChronusOmni: Improving time awareness of omni large language models. arXiv preprint arXiv:2512.09841 (2025)

5. Chen, Z., Tao, J., Li, R., Hu, Y., Chen, R., Yang, Z., Yu, X., Jing, H., Zhang, M., Shao, S., Wang, B., Lu, Q., Huang, R.: OmniVideo-R1: Reinforcing audio-visual reasoning with query intention and modality attention. In: ICML (2026)

6. Deng, A., Gao, Z., Choudhuri, A., Planche, B., Zheng, M., Wang, B., Chen, T., Chen, C., Wu, Z.: Seq2Time: Sequential knowledge transfer for video LLM temporal grounding. In: CVPR. pp. 13766–13775 (2025)

7. Dong, L., Zhang, H., Lin, H., Yan, Z., Zeng, X., Zhang, H., Huang, Y., Wang, Y., Ling, Z.H., Wang, L., et al.: VideoTG-R1: Boosting video temporal grounding via curriculum reinforcement learning on reflected boundary annotations. In: ICMR. pp. 1432–1441 (2026)

8. Gao, J., Sun, C., Yang, Z., Nevatia, R.: TALL: Temporal activity localization via language query. In: ICCV. pp. 5267–5275 (2017)

9. Ge, Y., Ge, Y., Li, C., Wang, T., Pu, J., Li, Y., Qiu, L., Ma, J., Duan, L., Zuo, X., et al.: ARC-Hunyuan-Video-7B: Structured video comprehension of real-world shorts. arXiv preprint arXiv:2507.20939 (2025)

10. Geng, T., Wang, T., Duan, J., Cong, R., Zheng, F.: Dense-localizing audio-visual events in untrimmed videos: A large-scale benchmark and baseline. In: CVPR. pp. 22942–22951 (2023)

11. Guo, D., Yang, D., Zhang, H., Song, J., Wang, P., Zhu, Q., Xu, R., Zhang, R., Ma, S., Bi, X., et al.: DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning. Nature 645(8081), 633–638 (2025)

12. Guo, Y., Liu, J., Li, M., Cheng, D., Tang, X., Sui, D., Liu, Q., Chen, X., Zhao, K.: VTG-LLM: Integrating timestamp knowledge into video LLMs for enhanced video temporal grounding. In: AAAI. vol. 39, pp. 3302–3310 (2025)

13. Guo, Y., Liu, J., Li, M., Liu, Q., Chen, X., Tang, X.: TRACE: Temporal grounding video LLM via causal event modeling. In: ICLR (2025)

14. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: LoRA: Low-rank adaptation of large language models. In: ICLR (2022)

15. Huang, B., Wang, X., Chen, H., Song, Z., Zhu, W.: VTimeLLM: Empower LLM to grasp video moments. In: CVPR. pp. 14271–14280 (2024)

16. Huang, D.A., Liao, S., Radhakrishnan, S., Yin, H., Molchanov, P., Yu, Z., Kautz, J.: LITA: Language instructed temporal-localization assistant. In: ECCV. pp. 202– 218 (2024)

17. Jaech, A., Kalai, A., Lerer, A., Richardson, A., El-Kishky, A., Low, A., Helyar, A., Madry, A., Beutel, A., Carney, A., et al.: OpenAI o1 system card. arXiv preprint arXiv:2412.16720 (2024)

18. Krishna, R., Hata, K., Ren, F., Fei-Fei, L., Niebles, J.C.: Dense-captioning events in videos. In: ICCV. pp. 706–715 (2017)

19. Liu, S.Y., Dong, X., Lu, X., Diao, S., Belcak, P., Liu, M., Chen, M.H., Yin, H., Wang, Y.C.F., Cheng, K.T., Choi, Y., Kautz, J., Molchanov, P.: GDPO: Group reward-decoupled normalization policy optimization for multi-reward RL optimization. In: ICML (2026)

20. Liu, Y., Peng, B., Zhong, Z., Yue, Z., Lu, F., Yu, B., Jia, J.: Seg-Zero: Reasoning-chain guided segmentation via cognitive reinforcement. arXiv preprint arXiv:2503.06520 (2025)

21. Lu, Z., Geng, T., Chen, Y., Wang, T., Lu, P., Zheng, F.: R-AVST: Empowering Video-LLMs with fine-grained spatio-temporal reasoning in complex audio-visual scenarios. In: AAAI. vol. 40, pp. 7627–7635 (2026)

22. Luo, F., Lou, S., Chen, C., Wang, Z., Li, C., Shen, W., Guo, J., Li, P., Yan, M., Zhang, J., et al.: MUSEG: Reinforcing video temporal understanding via timestamp-aware multi-segment grounding. In: ACL. pp. 35549–35561 (2026)

23. Lyu, C., Wu, M., Wang, L., Huang, X., Liu, B., Du, Z., Shi, S., Tu, Z.: Macaw-LLM: Multi-modal language modeling with image, audio, video, and text integration. arXiv preprint arXiv:2306.09093 (2023)

24. Radford, A., Narasimhan, K., Salimans, T., Sutskever, I., et al.: Improving language understanding by generative pre-training (2018)

25. Ren, S., Yao, L., Li, S., Sun, X., Hou, L.: TimeChat: A time-sensitive multimodal large language model for long video understanding. In: CVPR. pp. 14313–14323 (2024)

26. Rezatofighi, H., Tsoi, N., Gwak, J., Sadeghian, A., Reid, I., Savarese, S.: Generalized intersection over union: A metric and a loss for bounding box regression. In: CVPR. pp. 658–666 (2019)

27. Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Zhang, M., Li, Y.K., Wu, Y., Guo, D.: DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300 (2024)

28. Su, Y., Lan, T., Li, H., Xu, J., Wang, Y., Cai, D.: PandaGPT: One model to instruction-follow them all. In: TLLM Workshop. pp. 11–23 (2023)

29. Tang, C., Li, Y., Yang, Y., Zhuang, J., Sun, G., Li, W., Ma, Z., Zhang, C.: video-SALMONN 2: Caption-enhanced audio-visual large language models. arXiv preprint arXiv:2506.15220 (2025)

30. Tang, Y., Shimada, D., Bi, J., Feng, M., Hua, H., Xu, C.: Empowering LLMs with pseudo-untrimmed videos for audio-visual temporal understanding. In: AAAI. vol. 39, pp. 7293–7301 (2025)

31. Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al.: LLaMA: Open and eficient foundation language models. arXiv preprint arXiv:2302.13971 (2023)

32. Wang, Y., Wang, Z., Xu, B., Du, Y., Lin, K., Xiao, Z., Yue, Z., Ju, J., Zhang, L., Yang, D., et al.: Time-R1: Post-training large vision language model for temporal video grounding. In: NeurIPS. vol. 38, pp. 83330–83364 (2025)

33. Wu, J., Liu, W., Liu, Y., Liu, M., Nie, L., Lin, Z., Chen, C.W.: A survey on video temporal grounding with multimodal large language model. IEEE TPAMI (2025)

34. Wu, T., Yang, L., Zhan, G., Zhang, Y., Liao, Y., Li, J., Fu, D., Zhang, L., Wang, L.: TempR1: Improving temporal understanding of MLLMs via temporal-aware multi-task reinforcement learning. In: CVPR. pp. 2756–2767 (2026)

35. Wu, Y., Hu, X., Sun, Y., Zhou, Y., Zhu, W., Rao, F., Schiele, B., Yang, X.: Number It: Temporal grounding videos like flipping manga. In: CVPR. pp. 13754–13765 (2025)

36. Xu, J., Guo, Z., He, J., Hu, H., He, T., Bai, S., Chen, K., Wang, J., Fan, Y., Dang, K., Zhang, B., Wang, X., Chu, Y., Lin, J.: Qwen2.5-Omni technical report. arXiv preprint arXiv:2503.20215 (2025)

37. Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., et al.: Qwen2.5 technical report. arXiv preprint arXiv:2412.15115 (2024)

38. Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Dai, W., Fan, T., Liu, G., Liu, J., et al.: DAPO: An open-source LLM reinforcement learning system at scale. In: NeurIPS. vol. 38, pp. 113222–113244 (2025)

39. Zhang, H., Li, X., Bing, L.: Video-LLaMA: An instruction-tuned audio-visual language model for video understanding. In: EMNLP System Demonstrations. pp. 543–553 (2023)

40. Zhang, J., Wang, T., Ge, Y., Ge, Y., Li, X., Wang, L.: TimeLens: Rethinking video temporal grounding with multimodal LLMs. In: CVPR. pp. 10419–10429 (2026)