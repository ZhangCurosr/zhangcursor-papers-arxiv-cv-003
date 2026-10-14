# Physics-Aware Deepfake Detection via Distance–Speech Consistency

Kyeongrae Kim <sup>ID</sup> <sup>1</sup>, Kim Sung-Bin <sup>ID</sup> <sup>2</sup>, Oh Hyun-Bin <sup>ID</sup> <sup>2</sup>, Tae-Hyun Oh <sup>ID</sup> <sup>3</sup>

<sup>1</sup>School of Electrical Engineering and <sup>3</sup>School of Computing, KAIST, Korea <sup>2</sup>Dept. of Electrical Engineering, POSTECH, Korea

kimkyeongrae@kaist.ac.kr, {sungbin, hyunbinoh}@postech.ac.kr, taehyun.oh@kaist.ac.kr

## Abstract

Recent progress in generative models has made talking-head deepfakes highly realistic and easy to create, jeopardizing public trust. However, most audio-visual deepfake detectors rely on lip–speech synchronization and are designed for static, frontal videos, limiting their reliability in dynamic, in-the-wild recordings where the speaker moves and lip cues are degraded or missing. As an alternative, we propose a physics-aware detector for dynamic speaking videos that leverages an acoustic constraint: in real recordings, speech energy varies predictably with speakerto-camera distance, whereas deepfake generation can break this coupling. Our method combines distance estimates from video with distance-dependent acoustic measures from speech to detect deepfakes. Experiments on in-the-wild benchmarks show that the proposed physical cue is effective in dynamic scenarios, and that it complements lip-sync-based detectors, with a simple ensemble achieving consistent gains across datasets. Codes are available at https://byulharang.github.io/PADD/. Index Terms: deepfake detection, audio-visual learning, physical consistency

## 1. Introduction

Recent advances in deep generative models [1, 2, 3] have dramatically increased the realism and accessibility of synthetic media, raising the risk of convincing deepfake videos in everyday communication channels and public discourse [4, 5]. To defense these threats, various detection methods have been developed for video forensics, ranging from detection of visual artifacts [6, 7, 8] to verification of temporal and spatial consistency [9, 10, 11]. In particular, manipulating a human speech video is especially threatening since it can distort identity (visual appearance) and intent (speech content), and may influence perceived beliefs and decisions even with a short clip, exacerbating misinformation and jeopardizing public trust. This highlights the need for robust audio-visual deepfake detection.

Most existing audio-visual deepfake detection methods have primarily exploited the strong correlation between speech and lip motion, analyzing lip-speech synchronization to detect subtle mismatches [15, 16, 17, 18, 19]. These approaches have shown to be effective in videos where the speaker’s mouth is clearly visible and unobstructed, such as news, interviews, podcasts, and studio-like recordings [20, 21, 8, 22]. However, this reliance on lip–speech synchronization limits practical deployment to relatively constrained settings, such as static talking head videos with frontal faces, stable lighting, and limited camera motion, where the lip region remains consistently visible.

As generative tools rapidly advance while becoming cheaper and easier to use, deepfakes are no longer limited to carefully curated static talking head clips; instead, they increasingly appear in dynamic, in-the-wild videos where a subject moves through space while speaking, such as handheld recordings, vlogs, or on-the-street interviews. While effective under controlled and static conditions, the aforementioned approaches become unreliable in dynamic speaking videos, where the speaker naturally moves toward or away from the camera and facial cues are often degraded by occlusion, pose changes, and motion blur.

![](images/36c0dbbc3dc7ce432d7b2185ad1bbeea900480a200d761d92ee520ee9be8fdf5.jpg)  
Figure 1: Physics-aware deepfake detectionfor dynamic speaking videos. Existing detectors rely on lip–speech consistency and may fail when lip regions are partially visible or missing. Our method instead exploits the physical relationship between speaker distance and speech energy, enabling reliable detection even when visual lip cues are unreliable.

To address this gap, we propose a physics-aware deepfake detection framework for dynamic human speaking videos. Our key motivation is that authentic recordings must satisfy basic acoustic constraints: speech intensity varies predictably with the speaker-to-camera distance, approximately following an inverse relationship [23]. Although current deepfake generation and compositing methods can produce highly realistic content, they often neglect this physical relationship and may break the coupling between visual geometry and speech energy. Our approach estimates the speaker’s relative distance from the visual stream and compares it against speech-energy patterns expected from sound propagation. This provides a complementary detection signal that remains informative even when lip details are unreliable, and targets violations of the physical relationship between what is seen (distance/motion) and what is heard (speech intensity) in dynamic deepfake videos (refer to Fig. 1).

We validate the effectiveness of our approach on multiple audio-visual deepfake detection benchmarks that include dynamic videos [19, 24]. When we explicitly extract dynamic subsets from these benchmarks and evaluate on those videos, our physics-aware model outperforms prior detectors that primarily focus on lip–speech alignment [18, 19]. Moreover, our method performs favorably as a standalone detector compared to existing approaches. Because our physical-consistency cue complements prior detectors, ensembling it with existing models achieves substantial additional gains. To our knowledge, this is the first work to use audio-visual physical constraints as a primary cue for deepfake video detection. Our main contributions are:

![](images/c055e3b386086c29adc2290397c08f38a45bc117031a67c4283098d34c3a0476.jpg)  
Figure 2: Physics-aware deepfake detectionframework. Our method extracts speaker distance and acoustic measuresfrom video and audio streams. For speaker distance, we calculated $d _ { s p k }$ by averaging monocular depth values (MiDaS [12]) withinfacial landmarks (RetinaFace [13]). For acoustic measures, SNR and C were calculated using speech and noise separated by SAM-Audio [14] with a ’speech’ prompt. Distance-wise binning is then applied and the variance and entropy ofthe acoustic measures are aggregated toform statisticalfeatures v. Finally, v isfed into a lightweight MLP detector to predict whether it is an authentic or a deepfake video.

• Introduce a distance–speech consistency measure that reflects basic acoustic behavior in real dynamic speaking videos.

• Propose an audio-visual deepfake detector that leverages this measure as a primary cue for detection in dynamic settings.

• Demonstrate robust performance on dynamic speaking videos and consistent ensemble gains with lip–speech synchronization detectors, indicating complementarity.

## 2. Method

This section presents our physics-aware deepfake detection framework based on physical alignment between the speaker’s visual distance and speech acoustics. We first introduce the acoustic principle and distance-related measures used in our method (Sec. 2.1). We then describe how we extract acoustic measures and relative speaker distance, and how we convert them into robust video-level features by distance-wise binning and distributional statistics, $e . g .$ ., variance and entropy (Sec. 2.2). Finally, we build a lightweight multi-layer perceptron (MLP) that maps these statistics to a deepfake score (Sec. 2.3).

## 2.1. Preliminaries

In real-world recordings, a speaker’s distance and speech signals are coupled through acoustic propagation: as distance increases, speech intensity decreases, and vice versa [25, 26]. This is known as the inverse-square law. Based on this principle, we use two acoustic measures, signal-to-noise ratio (SNR) and clarity at 50 ms $( C _ { 5 0 } )$ , to capture distance-dependent behavior and assess physical consistency between spatial and acoustic signals.

Inverse-square law. In acoustics, the inverse-square law describes the physical relationship between sound intensity and the source-receiver distance [23]. Assuming a source radiates power P uniformly, the acoustic intensity I at distance d follows:

$$
I = \frac { P } { 4 \pi d ^ { 2 } } \propto \frac { 1 } { d ^ { 2 } } .
$$

Directly applying the inverse-square law to real-world recordings is often unreliable due to automatic gain control (AGC), devicespecific amplification, and recording-level normalization. We therefore employ relative energy measures [23, 27] that are robust to these effects. Specifically, we utilize the Signal-to-Noise Ratio (SNR) and Clarity at 50ms $( C _ { 5 0 } )$ , which capture ratios of energy rather than absolute magnitudes and thus reflect the inverse-square attenuation of direct sound. This makes them particularly suitable for in-the-wild videos where calibrated signals or ground-truth acoustic measurements are typically unavailable.

Signal-to-Noise Ratio (SNR). SNR measures the relative energy between the target source signal and environmental noise [27]. Let $S _ { \mathrm { r e c e i v e d } }$ denotes the speech signal captured by the microphone and $N _ { \mathrm { e n v } }$ denotes environmental noise. In typical recording conditions, environmental noise varies more slowly than the direct speech component within a short temporal window. Under this approximation, the variation of SNR is primarily governed by the attenuation of the direct sound with distance. Thus, when the direct part of speech signal follows the inverse-square law, the SNR of the received signal approximately follows:

$$
\mathrm { S N R \left[ d B \right] } \approx 1 0 \log _ { 1 0 } \left( \frac { \mathbb { E } \left[ S _ { \mathrm { r e c e i v e d } } ^ { 2 } \right] } { \mathbb { E } \left[ N _ { \mathrm { e n v } } ^ { 2 } \right] } \right) \propto - 2 0 \log _ { 1 0 } ( r ) .
$$

Clarity at 50ms $( C _ { 5 0 } ) . ~ C _ { 5 0 }$ measures speech clarity by computing the decibel ratio between the energy arriving within the first 50ms (direct sound and early reflections) and the energy arriving afterward (late reverberation) in the received signal [28]. Because the early component is typically dominated by the direct sound, which approximately follows inverse-square attenuation of the source energy $( E _ { \mathrm { s o u r c e } } / r ^ { 2 } )$ , $C _ { 5 0 }$ can be approximated as:

$$
C _ { 5 0 } \left[ \mathrm { d B } \right] \approx 1 0 \log _ { 1 0 } \left( \frac { E _ { \mathrm { { s o u r c e } } } / r ^ { 2 } } { E _ { \mathrm { { l a t e } } } } \right) \propto - 2 0 \log _ { 1 0 } ( r ) ,
$$

where $E _ { \mathrm { l a t e } }$ denotes the energy of late reflections. Although $C _ { 5 0 }$ is commonly used in room acoustics, the same intuition extends to outdoor recordings where the signal still contains a direct propagation component [28, 29]. Consequently, $C _ { 5 0 }$ provides a physically grounded cue for the source–receiver distance.

## 2.2. Extracting Distance-dependent Acoustic Features

Our framework extracts physically meaningful features by jointly using acoustic propagation and visual distance cues (see Fig. 2). We derive acoustic measures (SNR and $C _ { 5 0 } )$ from speech signals and estimate the relative speaker distance from video frames. These signals are then aggregated through distance-wise binning and statistical modeling to obtain video-level features.

Acoustic measure extraction. We extract SNR and $C _ { 5 0 }$ from the speech signal. To compute these, the speech component must be separated from environmental noise. Thus, we employ SAM Audio [14] with the text prompt “speech” to isolate the speech signal from the input audio. Using the separated signals, we compute SNR and $C _ { 5 0 }$ over temporal windows of 300 ms, which covers typical syllable durations in speech [30] while capturing most of reflections associated with room reverberation [31]. This design enables stable estimation of the acoustic measures while reducing interference from adjacent utterances.

Speaker distance estimation. We estimate the relative distance between the speaker and the camera for each frame. Facial landmarks are detected using RetinaFace [13] to localize the speaker region. A dense depth map of the frame is estimated using MiDaS [12]. We then obtain speaker distance by averaging the depth values within the detected facial region, and its inverse is used as a proxy for the speaker’s relative distance $d _ { s p k }$

Distance-wise binning. We use the speaker distance $d _ { s p k }$ to group acoustic measures into distance bins. Specifically, a distance range within each video is partitioned into the bins, and acoustic measures are averaged within each bin. This reduces fluctuations in vocal intensity and phonetic content that may violate the constant-power assumption of the physical model, allowing the acoustic measures to follow an attenuation trend:

$$
{ \mathbb E } [ \{ \mathrm { S N R } , C _ { 5 0 } \} \mid d _ { s p k } ] \propto - 2 0 \log _ { 1 0 } ( d _ { s p k } ) .
$$

Statistical modeling of acoustic measures. Although the expected acoustic measures follow the physical attenuation trend, instantaneous values of SNR and $C _ { 5 0 }$ are sensitive to speech variability and environmental noise. Thus, we characterize the distribution of acoustic measures within each depth bin using variance and entropy, which capture the following relationships.

Variance. When speech energy appears inconsistent with the speaker’s visual distance, as often observed in deepfake videos, acoustic measures within the same depth bin become more dispersed, resulting in larger variance. In contrast, authentic recordings show lower variance since acoustic propagation remains physically consistent with speaker distance: $\dot { V a r } [ x | d _ { s p k } ] _ { \mathrm { F a k e } } > V a r [ x | d _ { s p k } ] _ { \mathrm { R e a l } }$ , where $x \in \{ \mathrm { S N R } , C _ { 5 0 } \}$

Entropy. In deepfake videos, synthesized speech often exhibits reduced variability in acoustic measures, resulting in lower entropy [32]. In contrast, real-world videos are influenced by environmental noise and unconstrained recording conditions [33], leading to higher entropy: $\mathrm { H } ( x \mid d _ { s p k } ) _ { \mathrm { R e a l } } > \mathrm { H } ( x \mid d _ { s p k } )$ <sub>Fake</sub>. We aggregate the bin-wise variance and entropy of acoustic measures to obtain video-level statistics:

$$
\begin{array} { r l } & { \mathbb { E } _ { \mathrm { b i n } } [ V a r ( \boldsymbol { x } \mid d _ { s p k } ) ] _ { \mathrm { F a k e } } > \mathbb { E } _ { \mathrm { b i n } } [ V a r ( \boldsymbol { x } \mid d _ { s p k } ) ] _ { \mathrm { R e a l } } } \\ & { \quad \mathbb { E } _ { \mathrm { b i n } } [ \mathrm { H } ( \boldsymbol { x } \mid d _ { s p k } ) ] _ { \mathrm { R e a l } } > \mathbb { E } _ { \mathrm { b i n } } [ \mathrm { H } ( \boldsymbol { x } \mid d _ { s p k } ) ] _ { \mathrm { F a k e } } } \end{array}
$$

where $\mathbb { E } _ { \mathrm { b i n } }$ denotes the expectation over depth bins.

## 2.3. Physics-aware Deepfake Detector

To capture the statistical behavior of acoustic signals across distance bins, we extract features based on the variance and entropy of SNR and $C _ { 5 0 } .$ . For each acoustic measure x ∈ $\{ \mathrm { S N R } , C _ { 5 0 } \}$ , we compute the expected variance and expected entropy across distance bins, denoted as $\mathbb { E } _ { \mathrm { b i n } } [ V a r ( x \mid d _ { s p k } ) ]$ and $\mathbb { E } _ { \mathrm { b i n } } [ \mathrm { H } ( \boldsymbol { x } | d _ { s p k } ) ]$ ], respectively. These statistics capture how acoustic properties vary with respect to the estimated speaker distance. Using these statistics, we define the following features:

$$
\begin{array} { r l } & { V a r _ { \mathrm { S N R } } = \mathbb { E } _ { \mathrm { b i n } } [ V a r ( \mathrm { S N R } | d _ { s p k } ) ] , V a r _ { C _ { 5 0 } } = \mathbb { E } _ { \mathrm { b i n } } [ V a r ( C _ { 5 0 } | d _ { s p k } ) ] , } \\ & { \quad \mathrm { H } _ { \mathrm { S N R } } = \mathbb { E } _ { \mathrm { b i n } } [ \mathrm { H } ( \mathrm { S N R } | d _ { s p k } ) ] , \mathrm { H } _ { C _ { 5 0 } } = \mathbb { E } _ { \mathrm { b i n } } [ \mathrm { H } ( C _ { 5 0 } | d _ { s p k } ) ] . } \end{array}
$$

To further capture the relative variability between acoustic features, we additionally compute the variance ratio:

$$
r _ { \mathrm { v a r } } = \frac { V a r _ { \mathrm { S N R } } } { V a r _ { C _ { 5 0 } } } .
$$

We then construct a five-dimensional feature vector:

$$
\mathbf { v } = [ V a r _ { \mathrm { S N R } } , V a r _ { C _ { 5 0 } } , \mathrm { H } _ { \mathrm { S N R } } , \mathrm { H } _ { C _ { 5 0 } } , r _ { v a r } ] .
$$

Finally, we train a multi-layer perceptron (MLP) F with a single hidden layer that maps v to a scalar score $F ( \mathbf { v } ) \in \mathbb { R }$ , where scores closer to 1 indicate fake and closer to 0 indicate real.

Table 1: Results on the dynamic subset DF-Dynamic This subset contains videos with strong dynamics, such as substantial speaker motion. Under these conditions, our method substantially outperforms existing lip-sync-based detectors.
<table><tr><td>Method</td><td>ROC-AUC</td></tr><tr><td>SpeechForensics [18]</td><td>0.5096</td></tr><tr><td>AuViRe [19]</td><td>0.5517</td></tr><tr><td>Ours</td><td>0.7449</td></tr></table>

where $\mathbf { x } \in \{ S N R , C _ { 5 0 } \}$ and $\mathbb { E } _ { b i n }$

## 3. Experiments

In this section, we compare our physics-aware detector with state-of-the-art lip-sync-based detectors. We first evaluate the models on a curated dynamic subset, where physical cues are more informative. Then, we report ROC–AUC scores of the detectors on full set (i.e., both static and dynamic) of in-the-wild benchmarks. We also present qualitative examples and ablation studies to further analyze the method.

## 3.1. Experiment Setup

Competing methods. We compare against two state-of-the-art lip-sync-based deepfake detectors, SpeechForensics [18] and AuViRe [19]. Both methods focus on the alignment between lip movements and speech, and adopt AV-HuBERT [34] as the backbone encoder for audio-visual representation learning. They have demonstrated strong detection performance on widely used benchmark datasets that feature relatively static and constrained human speaking settings [20, 21, 8, 22], often reporting ROC-AUC scores above 0.95. However, their robustness under unconstrained real-world conditions remains less explored.

Datasets<sup>1</sup>. We use Deepfake-Eval-2024 (DF24) [24] and Au-ViRe RealWorld (RW) [19] to evaluate deepfake detection under unconstrained and dynamic conditions. Both datasets are collected from media platforms and include both static and dynamic videos, although RW is predominantly static. They also span diverse recording environments, including varying camera motion, environmental noise, and editing artifacts. To ensure reproducibility, we exclude samples with missing modalities or invalid links. All models are trained on the DF24 training set (fine-tuning existing detectors [18, 19] and training ours from scratch) and evaluated on the DF24 test split (in-domain) and the RW test set (cross-dataset). In addition, to analyze scenarios where physical cues are most informative, we curate a dynamic subset, DF-Dynamic, by manually selecting videos with noticeable speaker motion from both test sets.

## 3.2. Results and Analysis

Quantitative results. We first evaluate on the curated dynamic subset, DF-Dynamic, where physical cues are become particularly informative. Table 1 shows that existing lip-sync-based detectors exhibit limited performance in this setting, while our physics-aware detector achieves a substantially higher ROC-AUC score. This suggests that when speaker motion and recording conditions vary dynamically, the physical alignment between speaker distance and acoustic behavior provides a more reliable detection signal than lip–speech consistency alone.

We further evaluate our method on full in-the-wild benchmarks [24, 19] that include both static and dynamic videos. Table 2 reports (i) in-domain results on the DF24 test set and (ii) cross-dataset results on RW. On the in-domain DF24 test set, our model achieves a ROC–AUC of 0.7329, outperforming existing detectors, while achieving comparable performance on the RW dataset, which is predominantly static. Because unconstrained real-world conditions often degrade the mouth region, lip–speech alignment cues become less reliable [18, 19]; in contrast, our physics-aware approach maintains strong performance by leveraging a fundamentally different signal. Moreover, combining our model with existing detectors via a score-averaging ensemble consistently improves performance across both datasets (Table 2-(C)). These results indicate that the proposed physicalconsistency cue is favorable as a standalone signal, and that it provides complementary information that makes ensembling with lip-sync-based detectors even more effective.

Table 2: Results on the in-the-wild benchmarks. Models are trained on the DF24 training split and evaluated under two settings: (i) in-domain evaluation on the DF24 test set (mixed; static and dynamic videos) and (ii) cross-dataset evaluation on the RW dataset (largely static videos). (C) reports scoreaveraging ensembles between lip-sync-based detectors and ours.
<table><tr><td>Setting</td><td>In-domain Train: DF24 Test: DF24 (mixed)</td><td>Cross-dataset Train: DF24 Test: RW (static-heavy)</td></tr><tr><td colspan="3">(A) Lip-sync-based detectors</td></tr><tr><td>SpeechForensics [18]</td><td>0.6957</td><td>0.7819</td></tr><tr><td>AuViRe [19]</td><td>0.6890</td><td>0.6430</td></tr><tr><td colspan="3">(B) Physics-aware detector</td></tr><tr><td>Ours</td><td>0.7329</td><td>0.6510</td></tr><tr><td colspan="3">(C) Ensemble with ours</td></tr><tr><td>Ours &amp; SpeechForensics [18]</td><td>0.7786</td><td>0.7971</td></tr><tr><td>Ours &amp; AuViRe [19]</td><td>0.7797</td><td>0.6634</td></tr></table>

![](images/b889ab8d0e21ba521b1eccfd3afa89dadadecff6f11a8a928bc4065a76b01f20.jpg)

![](images/69c949b17f2988537c12904d902189e853e557ee3fcd0f78079828aed672af34.jpg)  
Figure 3: Qualitative examples of authentic (left) and deepfake (right) videos. We visualize the temporal trends ofdistance, SNR, and $C _ { 5 0 } .$ In authentic videos, the acoustic measures increase as the speaker moves closer to the camera,following the expected physical relationship between distance and speech energy. In contrast, deepfake videos often exhibit inconsistent trends between distance and acoustic measures.

Qualitative results. We visualize authentic and deepfake examples to show how acoustic measures vary with estimated speaker distance (see Fig. 3). In the authentic video, the acoustic measures follow the expected inverse trend: as the speaker moves closer to the camera, both SNR and C<sub>50</sub> increase due to stronger speech energy and clearer direct sound. In contrast, deepfake videos often violate this physical relationship, exhibiting inconsistent distance-acoustic trends. These examples illustrate that the proposed acoustic features provide physically meaningful signals for distinguishing authentic and deepfake videos.

Table 3: Ablation of physical features. Each row removes a subset offeaturesfrom our model; ∆ is relative score to ours.
<table><tr><td rowspan="2">Configuration</td><td colspan="2">DF24</td><td colspan="2">RW</td></tr><tr><td>ROC-AUC</td><td>∆</td><td>ROC-AUC</td><td>∆</td></tr><tr><td>Ours (Full)</td><td>0.7329</td><td>一</td><td>0.6510</td><td>一</td></tr><tr><td>(A) Acoustic-measure ablation</td><td></td><td></td><td></td><td></td></tr><tr><td>w/o SNR stats  $\mathsf { i } \left( V a r _ { \mathrm { S N R } } , \mathrm { H } _ { \mathrm { S N R } } \right)$ </td><td>0.7187</td><td>-0.0262</td><td>0.6528</td><td>+0.0018</td></tr><tr><td> $w / \mathrm { o } \ C _ { 5 0 } \ \mathrm { s t a t s } \ ( \dot { V } a r _ { C _ { 5 0 } } , \mathrm { H } _ { C _ { 5 0 } } )$ </td><td>0.6160</td><td>-0.1169</td><td>0.6090</td><td>-0.0420</td></tr><tr><td>(B) Statistical-descriptor ablation</td><td></td><td></td><td></td><td></td></tr><tr><td>w/o variance stats  $( V a r _ { \mathbb { S } \mathbb { N } } , V a r _ { C _ { 5 0 } } )$ </td><td>0.6646</td><td>-0.0683</td><td>0.6246</td><td>-0.0264</td></tr><tr><td>w/o entropy stats  $( \dot { \mathrm { H } } _ { \mathrm { S N R } } , \dot { \mathrm { H } } _ { C _ { 5 0 } } )$ </td><td>0.7097</td><td>-0.0232</td><td>0.6021</td><td>-0.0489</td></tr><tr><td>(C) Cross-stat feature</td><td></td><td></td><td></td><td></td></tr><tr><td>w/o rvar</td><td>0.7136</td><td>-0.0193</td><td>0.6495</td><td>-0.0015</td></tr></table>

Table 4: Ablation ofthe sound separation module. Removing the sound separation module degrades ROC-AUC scores.
<table><tr><td>Configuration</td><td>DF24</td><td>RW</td></tr><tr><td>Ours (with SAM Audio)</td><td>0.7329</td><td>0.6510</td></tr><tr><td>w/o SAM Audio</td><td>0.5945</td><td>0.5671</td></tr></table>

## 3.3. Ablation Studies

Impact of physical features. Table 3 reports the detection performance when subsets of physical features are removed. Removing the SNR-based statistics (Var , H ) decreases performance on DF24 (−0.0262 AUC), while removing the $C _ { 5 0 } \mathrm { - }$ based statistics $( V a r _ { C _ { 5 0 } } , \mathrm { H } _ { C _ { 5 0 } } )$ causes the largest degradation, reducing the ROC-AUC on DF24 (−0.1169) and RW (−0.042). We further analyze the contribution of the statistical descriptors. Removing variance-related features $( V a r _ { \mathrm { S N R } } , V a r _ { C _ { 5 0 } } )$ leads to a noticeable drop, while removing entropy-based features $( \mathrm { H } _ { \mathrm { S N R } } , \mathrm { H } _ { C _ { 5 0 } } )$ also degrades performance across datasets. Removing the variance ratio feature $r _ { v a r }$ results in an additional performance decrease (−0.0193 AUC on DF24), indicating that the relative variability between acoustic measures provides complementary information. Overall, best performance is achieved when all physical features are jointly utilized.

Effectiveness of universal sound separation. We evaluate the impact of the sound separation model [14] by comparing detection performance with and without SAM-audio (see Table 4). Using raw audio yields ROC-AUC scores of 0.5945 on DF24 and 0.5671 on RW, whereas applying SAM-audio improves them to 0.7329 and 0.6510, respectively. This indicates that separating speech from environmental noise enables more reliable acoustic statistics for detection.

## 4. Conclusion

In this work, we introduce a physics-aware framework for audio-visual deepfake detection in dynamic speaking videos. While most existing detectors rely on lip–speech synchronization, which becomes unreliable in dynamic in-the-wild scenarios, we instead leverage a fundamental acoustic constraint linking speaker distance and speech energy. Our method estimates speaker distance from video and extracts distance-dependent acoustic measures from audio to detect inconsistencies in this relationship. Experiments on curated dynamic subsets show that our approach is particularly effective for dynamic speaking videos. Moreover, our method complements lip-sync-based detectors through a simple ensemble, consistently improving detection performance across datasets.

Limitations and future work. Microphones placed close to the speaker can weaken the distance–speech relationship, and rapidly changing background noise can destabilize physics-aware acoustic measurements. Errors in depth and face estimation can degrade final performance. Our framework is speech-agnostic and may extend beyond human speech to general object-level scenarios that follow similar acoustic propagation principles.

## 5. Acknowledgments

This work was supported by IITP grant (No. 2022-0-00124, No.RS-2022-II220124, Development of Artificial Intelligence Technology for Self-Improving Competency-Aware Learning Capabilities; No. RS-2024-00457882, National AI Research Lab Project; No.RS-2025-25443318, Physically-grounded Intelligence: A Dual Competency Approach to Embodied AGI through Constructing and Reasoning in the Real World) funded by the Korea government(MSIT), the InnoCORE program of the Ministry of Science and ICT(26-InnoCORE-01), and KAIST Undergraduate Research Program (URP).

## 6. Use of Generative AI Disclosure

Parts of the manuscript were reviewed with the assistance of Gemini and ChatGPT; all AI-generated suggestions were meticulously reviewed by the authors, who take full responsibility for the content of the publication.

## 7. References

[1] A. Polyak, A. Zohar, A. Brown, A. Tjandra, A. Sinha, A. Lee, A. Vyas, B. Shi, C.-Y. Ma, C.-Y. Chuang et al., “Movie gen: A cast of media foundation models,” arXiv preprint arXiv:2410.13720, 2024.

[2] Google, “Veo: Google’s most capable generative video model,” 2024. [Online]. Available: https://deepmind.google/models/veo/

[3] T. Wiedemer, Y. Li, P. Vicol, S. S. Gu, N. Matarese, K. Swersky, B. Kim, P. Jaini, and R. Geirhos, “Video models are zero-shot learners and reasoners,” arXiv preprint arXiv:2509.20328, 2025.

[4] L. Y. Gong and X. J. Li, “A contemporary survey on deepfake detection: datasets, algorithms, and challenges,” Electronics, 2024.

[5] S. Xie, T. Qiao, S. Li, X. Zhang, J. Zhou, and G. Feng, “Deepfake detection in the aigc era: A survey, benchmarks, and future perspectives,” Information Fusion, 2025.

[6] K. Shiohara and T. Yamasaki, “Detecting deepfakes with selfblended images,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[7] Y. Yang, Z. Qian, Y. Zhu, O. Russakovsky, and Y. Wu, “Dˆ 3: scaling up deepfake detection by learning from discrepancy,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[8] A. Rossler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, and M. Nießner, “Faceforensics++: Learning to detect manipulated facial images,” in IEEE International Conference on Computer Vision (ICCV), 2019.

[9] Z. Guo, Y. Liu, J. Zhang, H. Zheng, and S. Shan, “Face forgery video detection via temporal forgery cue unraveling,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[10] D. Nguyen, M. Astrid, A. Kacem, E. Ghorbel, and D. Aouada, “Vulnerability-aware spatio-temporal learning for generalizable deepfake video detection,” in IEEE International Conference on Computer Vision (ICCV), 2025.

[11] C. Zheng, R. Suo, C. Lin, Z. Zhao, L. Yang, S. Liu, M. Yang, C. Wang, and C. Shen, “D3: Training-free ai-generated video detection using second-order features,” in IEEE International Conference on Computer Vision (ICCV), 2025.

[12] R. Ranftl, K. Lasinger, D. Hafner, K. Schindler, and V. Koltun, “Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer,” IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2020.

[13] J. Deng, J. Guo, E. Ververas, I. Kotsia, and S. Zafeiriou, “Retinaface: Single-shot multi-level face localisation in the wild,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

[14] B. Shi, A. Tjandra, J. Hoffman, H. Wang, Y.-C. Wu, L. Gao, J. Richter, M. Le, A. Vyas, S. Chen et al., “Sam audio: Segment anything in audio,” arXiv preprint arXiv:2512.18099, 2025.

[15] M. Astrid, E. Ghorbel, and D. Aouada, “Detecting audio-visual deepfakes with fine-grained inconsistencies,” in British Machine Vision Conference (BMVC), 2024.

[16] H. Zou, M. Shen, Y. Hu, C. Chen, E. S. Chng, and D. Rajan, “Crossmodality and within-modality regularization for audio-visual deepfake detection,” in IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2024.

[17] Y. Zhou and S.-N. Lim, “Joint audio-visual deepfake detection,” in IEEE International Conference on Computer Vision (ICCV), 2021.

[18] Y. Liang, M. Yu, G. Li, J. Jiang, B. Li, F. Yu, N. Zhang, X. Meng, and W. Huang, “Speechforensics: Audio-visual speech representation learning for face forgery detection,” in Advances in Neural Information Processing Systems (NeurIPS), 2024.

[19] C. Koutlis and S. Papadopoulos, “Auvire: Audio-visual speech representation reconstruction for deepfake temporal localization,” in IEEE Winter Conference on Applications ofComputer Vision (WACV), 2026.

[20] H. Khalid, S. Tariq, M. Kim, and S. S. Woo, “FakeAVCeleb: A novel audio-video multimodal deepfake dataset,” in Advances in Neural Information Processing Systems (NeurIPS), 2021.

[21] Z. Cai, S. Ghosh, A. P. Adatia, M. Hayat, A. Dhall, T. Gedeon, and K. Stefanov, “Av-deepfake1m: A large-scale llm-driven audiovisual deepfake dataset,” in ACM International Conference on Multimedia (MM), 2024.

[22] Z. Cai, S. Ghosh, A. Dhall, T. Gedeon, K. Stefanov, and M. Hayat, “Glitch in the matrix: A large scale benchmark for content driven audio–visual forgery detection and localization,” Computer Vision and Image Understanding (CVIU), 2023.

[23] L. E. Kinsler, A. R. Frey, A. B. Coppens, and J. V. Sanders, Fundamentals ofacoustics, 2000.

[24] N. A. Chandra, R. Murtfeldt, L. Qiu, A. Karmakar, H. Lee, E. Tanumihardja, K. Farhat, B. Caffee, S. Paik, C. Lee et al., “Deepfakeeval-2024: A multi-modal in-the-wild benchmark of deepfakes circulated in 2024,” arXiv preprint arXiv:2503.02857, 2025.

[25] A. Levkovitch, J. Salazar, S. Mariooryad, R. Skerry-Ryan, N. Bar, B. Kleijn, and E. Nachmani, “Zero-shot mono-to-binaural speech synthesis,” arXiv preprint arXiv:2412.08356, 2024.

[26] K. Patterson, K. Wilson, S. Wisdom, and J. R. Hershey, “Distancebased sound separation,” arXiv preprint arXiv:2207.00562, 2022.

[27] A. V. Oppenheim, Discrete-time signal processing, 1999.

[28] H. Kuttruff, Room acoustics, 2016.

[29] J. Kang, “Sound propagation in street canyons: Comparison between diffusely and geometrically reflecting boundaries,” The Journal ofthe Acoustical Society ofAmerica, 2000.

[30] D. Poeppel, “The analysis of speech in different temporal integration windows: cerebral lateralization as ‘asymmetric sampling in time’,” Speech communication, 2003.

[31] C. D´ıaz and A. Pedrero, “The reverberation time of furnished rooms in dwellings,” Applied Acoustics, 2005.

[32] Y. Ren, X. Tan, T. Qin, Z. Zhao, and T.-Y. Liu, “Revisiting oversmoothness in text to speech,” in Associationfor Computational Linguistics (ACL), 2022.

[33] S. Bellows and T. W. Leishman, “Effect of head orientation on speech directivity,” in Conference ofthe International Speech Communication Association (INTERSPEECH), 2022.

[34] B. Shi, W.-N. Hsu, K. Lakhotia, and A. Mohamed, “Learning audio-visual speech representation by masked multimodal cluster prediction,” arXiv preprint arXiv:2201.02184, 2022.