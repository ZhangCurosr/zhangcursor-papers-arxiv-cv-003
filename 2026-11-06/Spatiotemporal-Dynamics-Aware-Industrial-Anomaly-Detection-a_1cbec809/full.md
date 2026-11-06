# Spatiotemporal Dynamics-Aware Industrial Anomaly Detection and Reasoning

S-Wei Chen<sup>\*</sup> Siangyu Yang<sup>\*</sup> Yi-Chen Jan<sup>\*</sup> Chun-Yi Lee Shao-Yuan Lo

National Taiwan University

{b11902027, b12902055, b11902057, cylee, sylo}@csie.ntu.edu.tw

## Abstract

Industrial Anomaly Detection (IAD) is a long-established problem that aims to detect defects in industrial products during manufacturing for quality control. Most existing studies focus on static visual patterns as the basis for detection. However, in real manufacturing environments, many anomalies are dynamics-based, such as irregular movements that cannot be perceived from static cues alone. We observe that even recent Multimodal Large Language Model (MLLM)-based approachesfall short in identifying such dynamic anomalies due to limited temporal modeling capability. To address this, we introduce MUS-TARD: MUltimodal SpatioTemporal Anomaly Reasoning and Detection, a framework that equips MLLMs with finegrained spatiotemporal dynamics awareness for detecting and reasoning about dynamics-based defects. MUSTARD introduces optical-flow-guided keyframe selection and augmented modalities to capture critical motion irregularities through local motion patterns and global dynamic evolution. Its normal pattern-based reasoning pipeline requires only afew normal reference samples as the detection basis, avoiding any model updates. This enables efficient adaptation across various products and provides high decision interpretability. MUSTARD is one ofthefirst MLLM-based frameworks for dynamics-based IAD. Experiments demonstrate its superior performance in detecting spatiotemporal anomalies and providing plausible explanations. Our code is available at https://github.com/racercode/ Mustard

## 1. Introduction

Industrial Anomaly Detection (IAD) is a crucial problem that aims to automatically inspect product defects in manufacturing environments. Most existing IAD benchmarks [3, 4, 11, 26, 42], and approaches [6, 13, 34, 38] focus on static visual defects, e.g., a missing chunk on a candle or a bulge on the side of a bowl. These static anomalies can be identified from a single image. However, in real production lines, many anomalies are driven by dynamic irregularities, such as irregular rotations, abnormal vibrations, and unexpected component interactions, which cannot be captured from static cues alone. Inspecting such dynamicsbased anomalies requires physical operations. For example, a fan may appear visually flawless from a static observation, but we need to turn it on to see whether it rotates smoothly or gets stuck. Although modern smart factories have been able to leverage robot arms to automate these tests, we still lack an effective IAD algorithm capable of inspecting dynamic anomalies from the resulting test videos, which requires fine-grained spatiotemporal dynamic understanding. Advancing such robot perception capabilities is crucial for seamless integration with robotic arm operations, enabling a fully automated smart manufacturing process.

![](images/4ea145f8ee79e49dde585344e720e8b656d9200ac8b3f22732b28d466857ecff.jpg)  
Figure 1. Comparison between the direct use of an MLLM and the proposed method on an anomalous test video (a degaussing magnet). (1) Directly Use MLLM: fails to recognize that the magnet falls off the metal surface. (2) Our Method: leverages spatiotemporal information to correctly interpret the magnet’s drop from the surface.

Existing IAD approaches face fundamental limitations. Traditional convolutional network–based methods employ self-supervised pretext tasks, such as image reconstruction, to model normal patterns [19, 34, 38]. CLIP-based frameworks [6, 13, 41] leverage vision-language semantic alignment to achieve few-shot or zero-shot IAD. Nevertheless, both streams consider only static defects and thus rely primarily on static visual attributes. In addition, they provide only anomaly scores, without interpretable rationales, to users. While recent advances in Multimodal Large Language Models (MLLMs) have shown remarkable capabilities across various vision tasks [1, 16], they still fall short in dynamics-based IAD [35]. We observe that a key reason stems from dynamic patterns unfolding over multiple video frames. For instance, determining whether a gear rotates correctly requires capturing motion across consecutive frames; a single snapshot cannot reveal whether the gear is stuck, rotating incorrectly, or moving at an improper speed. Similarly, detecting abnormal vibrations or timing misalignments necessitates capturing the temporal evolution of system behavior. However, current MLLMs tend to overlook or fail to prioritize critical anomalous dynamics. We reveal that even video-oriented MLLMs [15, 39] lack an effective mechanism to explicitly model frame-toframe temporal changes (see Fig. 1-(1)), making it difficult to detect subtle variations such as gradual slowdowns or slight trajectory deviations. Moreover, most MLLMs employ uniform frame sampling to reduce computational cost. Although uniform sampling may be sufficient for general video understanding tasks, it fundamentally limits their ability to capture fine-grained temporal anomalies.

To address this challenge, we posit that optical flow, which encodes pixel-level motion between consecutive frames, provides a rich yet underutilized signal for dynamics-based IAD. Flow-based directional arrows reveal local movement trajectories, enabling the tracking of object positions and rotations over time. The flow magnitude represents global motion speed, and its variation highlights critical state transitions, such as when components start moving, stop, or change velocity. By converting optical flow into interpretable visual and statistical representations, we provide MLLMs with explicit temporal dynamics information that complements static frame content.

In this paper, we propose MUSTARD: MUltimodal SpatioTemporal Anomaly Reasoning and Detection, a novel framework that equips MLLMs with optical flow-guided spatiotemporal dynamics awareness, enabling accurate detection and reasoning of dynamic anomalies (see Fig. 1- (2)). MUSTARD features three key innovations: (1) an optical flow-guided keyframe selection strategy that precisely samples frames containing critical motion irregularities; (2) two auxiliary optical flow–augmented modalities, including trajectory-overlaid video frames annotated with local motion patterns using directional arrows and a flow magnitude curve representing global dynamic evolution; and (3) a normal pattern–based reasoning pipeline that extracts normal patterns from a small set of reference data as the decision basis, providing plausible anomaly explanations. The entire pipeline adheres to the one-class IAD (a.k.a. semi-supervised IAD) setting, in which only normal training samples are available, and no anomaly knowledge is leaked.

In addition, MUSTARD is a few-normal-shot [32] method that uses only a small number of normal samples from the training set as references to extract normal patterns for IAD. This eliminates the need for expensive full-shot training on the entire dataset and does not require any model weight updates, enabling efficient adaptation to various industrial product lines in real manufacturing scenarios. We evaluate our method on Phys-AD [14] and IPAD [17], the two pioneering dynamics-based IAD benchmarks available in the literature. Extensive experiments demonstrate MUS-TARD’s state-of-the-art detection performance and reasoning ability. The main contributions are threefold:

• We propose one of the first MLLM-based frameworks for dynamics-based IAD, addressing challenges beyond static IAD and augmenting automation in manufacturing.

• We introduce a novel optical-flow-guided spatiotemporal dynamics awareness method, enabling accurate detection and reasoning of dynamic anomalies.

• The proposed normal pattern-based reasoning pipeline allows efficient adaptation to various industrial products, which is highly practical for real production lines.

## 2. Related Work

Industrial Anomaly Detection. IAD focuses on products, equipment, and manufacturing processes under controlled conditions. Most existing IAD studies operate at the image level, aiming to detect texture or structural defects in static settings. Typical approaches include autoencoder-based reconstruction (e.g., DRAEM [38], UniAD [34], PrincipaLS [19]), CLIP-based vision-language alignment [22] (e.g., WinCLIP [13], AdaCLIP [6], AnomalyCLIP [41]), and recent MLLM-based frameworks (e.g., AnomalyGPT [9], Anomaly-OV [31], Anomaly-LR [33]). However, dynamics-based IAD remains largely unexplored. Specifically, existing approaches rarely incorporate motion continuity, causal reasoning, or temporal cues, which are crucial for detecting dynamic anomalies such as slippage, detachment, or collision. Recently, Phys-AD [14] and IPAD [17] have been introduced as benchmarks for dynamics-based industrial anomalies. Concurrent to our work, O-VAD [35] uses object-centric state tracking and reasoning, but it is less effective on subtle motion cues and fast state transitions.

![](images/4adb509c6c36f3d0ba0555fde739e44507c56848a5f27b97cad5670b0efa426d.jpg)  
Figure 2. An overview of the proposed MUSTARD framework. It consists of three main stages. (1) Spatiotemporal Dynamics Awareness: Leverages optical flow to select keyframes and derive three complementary descriptive modalities, including raw frames for visual context, trajectory frames for visualizing local motion, and a flow magnitude curve for representing global temporal dynamics. (2) Normal Pattern Extraction: Extracts multi-faceted normal patterns from a small set of reference videos. (3) Pattern Reasoning (Inference): Compares the test video’s description with the extracted normal patterns to detect deviations, producing a final anomaly judgment and explanation.

Video Anomaly Detection. Video Anomaly Detection (VAD) targets human-centric surveillance scenes. Existing approaches can be grouped into three main categories: autoencoder-based, CLIP-based, and MLLMbased frameworks. Autoencoder-based methods (e.g., MemAE [8], MNAD [21]) learn normal appearance patterns from surveillance videos and detect deviations through reconstruction or prediction errors. CLIP-based methods (e.g., VadCLIP [30]) leverage vision–language alignment to score anomalous events. Recent MLLM-based frameworks (e.g., LAVAD [37], AnomalyRuler [32], HAWK [23], Holmes-VAU [40]) integrate visual features with LLMs to detect anomalies. These VAD studies primarily focus on human-centric surveillance videos that are largely static, where most anomalous events can be identified from static appearance cues. In contrast, dynamics-based IAD videos are fast-paced and object-centric, requiring finer spatiotemporal granularity. Hence, VAD approaches are not directly applicable.

Multimodal Large Language Models. MLLMs have recently demonstrated strong general reasoning capabilities across vision–language tasks, with representative model families including LLaVA [16], Qwen-VL [1, 2], InternVL [28], and Seed-VL [5]. However, for video tasks, most models still overlook fine-grained temporal cues, relying on uniformly sampled frames or static summaries (e.g., VideoLLaMA [39], Video-LLaVA [15]). This results in appearance-centric reasoning that neglects subtle temporal dynamics and causal patterns crucial for dynamicsbased IAD. Recent attempts, such as StimuVAR [10] and AKS [24], introduce keyframe selection and token sampling strategies to enhance temporal awareness in MLLMs for video understanding. However, these approaches are designed for generic video tasks and offer limited effectiveness in short, motion-sensitive anomaly scenarios, where fine-grained spatiotemporal cues are essential yet insufficiently explored.

## 3. Method

## 3.1. Problem Definition

We address the dynamics-based IAD problem under the one-class setting. The model is provided with a small number of normal industrial product inspection videos containing dynamic processes. From these normal-only videos, the objective is to learn a comprehensive representation of spatiotemporal normal behavior.

When presented with a previously unseen test video, the model must determine whether it deviates from the learned normal patterns. The output is twofold: (1) a numerical anomaly score and (2) a coherent, human-interpretable natural language explanation that justifies the score. This formulation strictly adheres to the industrial scenario, where anomalous samples are typically unavailable, and both a quantitative score and an interpretable rationale are essential for practical deployment.

![](images/bcb81be1fef1dcf434cbab83c258280b561e81d8568a373949201deba09ea1c3.jpg)  
Figure 3. Illustration of the three derived modalities: raw frames, trajectory frames, and the flow magnitude curve. Different sources exce in different categories of test cases. Integrating them enables the model to leverage complementary strengths across modalities.

## 3.2. Framework Overview

To address the challenge of detecting dynamics-based anomalies, which are often overlooked by existing MLLMs, we propose MUSTARD: Multimodal Spatiotemporal Anomaly Reasoning and Detection, a novel trainingfree framework that equips models with explicit spatiotemporal awareness. Our core insight is that by translating implicit temporal motion information in videos into explicit textual representations, we enable MLLMs to reason about complex physical dynamics.

As illustrated in Fig. 2, the proposed MUSTARD framework consists of three main stages. (1) Spatiotemporal Dynamics Awareness. This stage processes a raw input video to extract rich, motion-aware information. It leverages optical flow to select keyframes and derive three complementary descriptive modalities: raw frames for visual context, trajectory frames for visualizing local motion, and a flow magnitude curve for representing global temporal dynamics. (2) Normal Pattern Extraction. This stage operates in a few-normal-shot paradigm, extracting multi-faceted normal patterns from a small set of reference videos. (3) Pattern Reasoning. During inference, the model compares the test video’s description with the extracted normal patterns to detect deviations, fusing evidence from all three modalities to produce a final anomaly judgment and explanation.

## 3.3. Spatiotemporal Dynamics Awareness

This stage aims to fully exploit spatiotemporal information in a video to capture fine-grained dynamics-based anomalies. We first approach this from the perspective of global temporal dynamics. Uniform frame sampling is a standard practice in deep learning models, including

MLLMs [15, 39], for video processing, as it significantly reduces computational cost. However, such uniform sampling overlooks key timesteps that are crucial for detecting abnormal dynamics. To address this limitation, we propose to adopt keyframe selection [10, 24] to capture critical timesteps while maintaining the same number of frames. These critical timesteps are intrinsically well-suited to modeling via optical flow estimation. Specifically, we employ the Dual TV-L1 optical flow algorithm [36] to derive pixellevel dense motion between each pair of adjacent frames in the input video. By aggregating the mean flow magnitudes for each frame, we construct a flow magnitude curve that depicts the intensity of optical flow over time (see Fig. 2). Acting as a temporal fingerprint of the video’s overall global dynamics, this curve enables the selection of keyframes corresponding to dynamically significant moments: peaks (maximum motion), valleys (minimum motion), and shoulders (transitional phases). This ensures that the fixed frame sampling budget is allocated to the frames most informative for understanding temporal evolution.

Next, based on the derived optical flow, we construct three complementary modalities for the selected keyframes to further enrich the spatiotemporal context (see Fig. 3). (1) Rawframes: The original keyframes capture the key “what” and “where” of the scene. (2) Trajectory frames: We capture local motion by overlaying optical-flow directional arrows onto the frames. The arrow size represents motion magnitude, and trailing lines render the motion path history. We initialize uniformly distributed probe points on the first frame and propagate each point through consecutive frames by sampling the local TV-L1 optical-flow vector at its current position, forming continuous motion trajectories. To ensure object-centric annotations, we apply the Farthest Point Sampling [7] strategy to select a stable and informative subset of points for arrow rendering.(3) Flow magnitude curve: The same curve used for keyframe selection, capturing the global temporal dynamics of the entire video. Then, we employ an MLLM to generate text descriptions for each modality, encoding their spatiotemporal information in natural language. The raw frame description focuses on static visual details, such as the main objects, their parts, and their surroundings; the trajectory frame description captures local motion; and the flow magnitude curve description focuses on global temporal dynamics, acting as a plot reader that provides a structured 13-field textual summary (e.g., maximum magnitude, number of peaks, magnitude trends). The complete modality-description prompts, including the 13-field flow-magnitude summary format, is provided in Appendix A7. Each source exhibits distinct strengths, and we integrate them to complement each other.

## 3.4. Normal Pattern Extraction

The proposed MUSTARD framework adheres to the strict one-class setting via a few-normal-shot paradigm, learning normal patterns from a small set of reference normal videos. In our problem, inspection videos may exhibit varying conditions, such as differences in color and operating angle, even for the same product. We therefore aim to extract normal patterns that summarize the reference videos, capturing such variations while aligning descriptions across different videos to form a more robust reference representation.

Specifically, we take a small set of normal training videos as references and apply the Spatiotemporal Dynamics Awareness process to obtain trajectory frame descriptions and flow magnitude curve descriptions for each video. We then adopt a position-wise consolidation strategy to extract robust normal patterns across videos. First, for each ordered keyframe position, we aggregate descriptions across reference videos and query an LLM to produce a shared pattern. Keyframes at the same position are expected to capture a similar state across videos. Next, we concatenate these timestep-wise patterns in temporal order and query the LLM to summarize them into global normal patterns, preserving the overall temporal structure of the process. The resulting trajectory frame patterns and flow magnitude curve patterns serve as robust references for detecting dynamics-based anomalies during inference (see Fig. 2). The prompts for normal pattern extraction are provided in Appendix A7.

## 3.5. Pattern Reasoning

At inference time, we apply the same Spatiotemporal Dynamics Awareness process to obtain descriptions of the three modalities for each test video. We then perform pattern reasoning via the LLM, where the trajectory and flow magnitude descriptions are compared against their corresponding normal patterns derived from the Normal Pattern Extraction stage. This comparison produces modalityspecific anomaly scores along with structured natural language justifications grounded in temporal dynamics.

Crucially, the LLM further performs cross-modal evidence fusion by jointly reasoning over these modalityspecific outputs together with the raw frame description (see Fig. 2). Through this unified reasoning process, it weighs complementary signals from raw visual context, local motion trajectories, and global temporal dynamics to generate a final anomaly score and coherent explanation. This enables principled dynamics-based anomaly detection with interpretable decision-making. The prompts for frame descriptions, pattern reasoning, anomaly-score generation, and cross-modal fusion are provided in Appendix A7.

## 4. Experiments

## 4.1. Experimental Setups

Datasets. Phys-AD [14] is the first large-scale, physicsgrounded video dataset for industrial anomaly detection, focusing on anomalies revealed through physical interaction and dynamics rather than static defects. The dataset contains over 6,400 videos across 22 object categories (e.g., gears, rubber bands) and 47 anomaly types, requiring models to reason about physical properties like elasticity and friction. To demonstrate the generalization of our method, we also evaluate on the IPAD [17] dataset. It is a specialized industrial benchmark that emphasizes the periodicity of equipment operations. It comprises 597,979 frames across 16 industrial devices, and 39 anomaly classes covering appearance, position, and motion-based failures.

Evaluation Metrics. To provide a holistic evaluation of both detection and reasoning, we utilize three metrics:

• Area Under the Receiver Operating Characteristic Curve (AUC): The standard, video-level metric for anomaly detection performance.

• PAEval: The Phys-AD dataset introduces an evaluation metric that quantifies the semantic similarity between model anomaly explanation and ground-truth anomaly explanation, serving as a measure of anomaly reasoning quality. PAEval is given ranging from 0 to 1, higher score means better quality of anomaly explanation.

• LLM-as-a-Judge: We evaluated our best method with other methods using LLM-as-a-Judge, where a powerful LLM (GPT-4o [12]) is used to pairwise compare two anomaly explanation, based on the ground-truth video descriptions in the Phys-AD dataset.

Baseline approaches. We compare our method against three categories of baseline approaches:

• Unsupervised and weakly-supervised methods: MNAD [21], VADClip [30], S3R [29] representing traditional VAD methods that only provide anomaly scores without reasoning.

Table 1. Video-level AUROC (%) across 22 Phys-AD categories. Scores are scaled by 100, with category-wise best results in bold. Average denotes the unweighted mean across categories.
<table><tr><td rowspan="2">Category</td><td colspan="4">Unsupervised / Weakly-supervised</td><td colspan="4">General-purpose MLLMs</td><td colspan="3">Specialized VAD</td><td>Ours</td></tr><tr><td>MNAD.p</td><td>MNAD.r</td><td>VADClip</td><td>S3R</td><td>V-Chat</td><td>V-Llama</td><td>V-Llava</td><td>Qwen2.5</td><td>LAVAD</td><td>ARuler</td><td>O-VAD</td><td>MUSTARD</td></tr><tr><td>Car</td><td>49.2</td><td>94.4</td><td>58.1</td><td>60.6</td><td>50.0</td><td>67.8</td><td>52.2</td><td>50.0</td><td>55.7</td><td>40.7</td><td>57.5</td><td>56.5</td></tr><tr><td>Fan</td><td>81.0</td><td>54.2</td><td>62.4</td><td>64.0</td><td>54.9</td><td>59.2</td><td>61.1</td><td>50.0</td><td>51.0</td><td>49.4</td><td>66.9</td><td>47.4</td></tr><tr><td>Rolling Bearing</td><td>35.2</td><td>80.0</td><td>58.9</td><td>60.1</td><td>30.0</td><td>50.0</td><td>50.0</td><td>50.0</td><td>3 3</td><td>40.0</td><td>65.0</td><td>94.2</td></tr><tr><td>Spherical Bearing</td><td>96.2</td><td>81.3</td><td>50.0</td><td>68.2</td><td>45.0</td><td>55.0</td><td>50.0</td><td>50.0</td><td></td><td>62.0</td><td>64.1</td><td>76.4</td></tr><tr><td>Servo</td><td>97.5</td><td>87.8</td><td>51.8</td><td>59.2</td><td>50.6</td><td>68.3</td><td>46.4</td><td>50.0</td><td>50.2</td><td>49.3</td><td>49.5</td><td>69.9</td></tr><tr><td>Clip</td><td>63.0</td><td>33.3</td><td>41.2</td><td>56.1</td><td>66.9</td><td>55.6</td><td>45.8</td><td>50.0</td><td>51.6</td><td>43.8</td><td>42.3</td><td>86.6</td></tr><tr><td>U Disk</td><td>60.9</td><td>94.0</td><td>53.0</td><td>54.9</td><td>56.5</td><td>57.5</td><td>50.0</td><td>50.0</td><td>51.3</td><td>46.0</td><td>60.6</td><td>75.3</td></tr><tr><td>Hinge</td><td>70.5</td><td>89.5</td><td>73.7</td><td>56.1</td><td>50.0</td><td>69.2</td><td>50.0</td><td>50.0</td><td>56.4</td><td>49.9</td><td>52.6</td><td>70.0</td></tr><tr><td>Sticky Roller</td><td>45.1</td><td>93.6</td><td>54.2</td><td>83.5</td><td>45.0</td><td>54.4</td><td>46.7</td><td>50.0</td><td>26.6</td><td>54.0</td><td>81.1</td><td>68.7</td></tr><tr><td>Caster Wheel</td><td>27.1</td><td>50.8</td><td>58.7</td><td>67.6</td><td>44.4</td><td>64.2</td><td>50.0</td><td>50.0</td><td>61.5</td><td>51.2</td><td>30.1</td><td>62.8</td></tr><tr><td>Screw</td><td>68.0</td><td>54.7</td><td>50.0</td><td>65.7</td><td>47.2</td><td>25.6</td><td>55.0</td><td>50.0</td><td>68.8</td><td>47.2</td><td>70.4</td><td>74.1</td></tr><tr><td>Lock</td><td>43.0 39.</td><td>64.1</td><td>45.2</td><td>52.3</td><td>27.9</td><td>49.4</td><td>50.0</td><td>50.0</td><td>34.1</td><td>47.8</td><td>78.0</td><td>83.3</td></tr><tr><td>Gear</td><td></td><td>69.4</td><td>51.9</td><td>58.0</td><td>54.4</td><td>60.3</td><td>51.7</td><td>50.0</td><td>60.3</td><td>55.1</td><td>87.9</td><td>67.7</td></tr><tr><td>Clock</td><td></td><td>58.7</td><td>50.0</td><td>54.9</td><td>50.1</td><td>36.0</td><td>50.0</td><td>50.0</td><td>50.0</td><td>42.5</td><td>66.9</td><td>55.6</td></tr><tr><td>Slide</td><td>91.7 82.9</td><td>78.4</td><td>53.1</td><td>61.1</td><td>56.2</td><td>56.7</td><td>17.9</td><td>50.0</td><td>42.5</td><td>40.6</td><td>41.5</td><td>70.7</td></tr><tr><td>Zipper</td><td></td><td>42.1</td><td>50.4</td><td>63.6</td><td>54.7</td><td>48.9</td><td>50.0</td><td>49.2</td><td>53.5</td><td>46.8</td><td>38.8</td><td>62.7</td></tr><tr><td>Button</td><td>66.0</td><td>56.8</td><td>62.7</td><td>51.5</td><td>51.5</td><td>51.7</td><td>36.0</td><td>50.0</td><td>43.9</td><td>42.6</td><td>39.3</td><td>62.1</td></tr><tr><td>Liquid</td><td>67.1</td><td>83.1</td><td>54.2</td><td>59.5</td><td>41.0</td><td>27.8</td><td>21.7</td><td>53.3</td><td>50.4</td><td>77.2</td><td>77.6</td><td>80.9</td></tr><tr><td>Rubber Band</td><td>36.6 72.8</td><td>30.7</td><td>48.2</td><td></td><td>51.7</td><td>45.0</td><td>45.0</td><td>50.0</td><td>51.1</td><td>44.9</td><td>72.1</td><td>75.3</td></tr><tr><td>Ball</td><td></td><td>68.7</td><td>50.0</td><td>0.77</td><td>56.2</td><td>63.6</td><td>53.3</td><td>50.0</td><td>60.3</td><td>47.3</td><td>57.9</td><td>59.1</td></tr><tr><td>Magnet Toothpaste</td><td>69.1 52.0</td><td>43.8 63.1</td><td>50.0</td><td>54.8</td><td>68.3</td><td>30.0</td><td>40.0</td><td>50.0</td><td>50.2</td><td>64.6</td><td>57.6</td><td>50.9</td></tr><tr><td></td><td></td><td></td><td>50.0</td><td>68.6</td><td>37.6</td><td>55.0</td><td>46.7</td><td>50.0</td><td>56.2</td><td>53.2</td><td>68.9</td><td>48.7</td></tr><tr><td>Average</td><td>62.7</td><td>66.9</td><td>53.5</td><td>61.3</td><td>49.6</td><td>52.3</td><td>46.3</td><td>50.1</td><td>51.0</td><td>49.8</td><td>60.3</td><td>68.1</td></tr></table>

• General-purpose MLLMs: Video-ChatGPT [20], Video-LLaMA [39], Video-LLaVA [15], Qwen2.5-VL [1], as non-specialized MLLMs. Specialized prompts are fed into this model to generate anomaly scores and explanations.

• Specialized VAD MLLMs: LAVAD [37], AnomalyRuler [32], and O-VAD [35], as SOTA MLLM-based VAD frameworks.

Implementation Details. We use Qwen2.5-VL-7B for vision-to-text and Gemma3-12B [25] for pattern operations. All models are served locally through Ollama. Normal patterns are constructed from N = 30 videos. For data preparation, we compute dense optical flow using the Dual TV-L1 algorithm. Specific hyperparameters include: we select a fixed budget of 8 keyframes per video based on the flow magnitude curve. For object-centric trajectory annotations, we sample an 80 × 50 grid and rank 600 active points. We select the 10 most active and 40 via Farthest Point Sampling for a total of 50 tracked points. Trajectory trails are rendered over a 12-frame history.

Performance metrics for unsupervised and weaklysupervised methods, LAVAD and general-purpose MLLMs, with the exception of Qwen2.5-VL-7B which we evaluate directly. For fair comparison with AnomalyRuler, we remove its perception-smoothing module to enable videolevel anomaly detection, and employ identical MLLM and LLM models (Qwen2.5-VL-7B and Gemma3-12B), ensuring consistency in backbone models across all experiments.

## 4.2. Main Results

Detection Performance on Phys-AD. Table 1 presents the AUC performance of primary anomaly detection frameworks on the Phys-AD dataset. Our method achieves an AUC of 0.681, outperforming all MLLM-based baselines by 0.078 and exceeding the state-of-the-art method by 0.012. Notably, the SOTA approach requires full-shot training and provides no anomaly explanations, whereas our method operates in a few-normal-shot manner without full training while delivering superior anomaly detection performance. In our architecture, Qwen2.5-VL-7B serves as the primary MLLM backbone for essential video-to-text tasks. However, in Table 1 we can see that directly obtaining video descriptions and querying Qwen2.5-VL-7B to determine if the description is ”anomalous or not”, is insufficient for accurate detection. This limitation arises because the model’s inherent, general-purpose knowledge of anomalies often fails to align with the highly specific physical and operational constraints of industrial scenarios, highlighting the fundamental difficulty of zero-shot reasoning in this domain. In contrast, our few-normal-shot paradigm leverages normal exemplars to establish domainspecific anomaly criteria, enabling effective detection without extensive training. Notably, current video anomaly detection (VAD) methods are inferior to handle fine-grained spatiotemporal anomalies in industrial settings. Previous approaches predominantly focus on anomalies detectable within single frames such as surveillance anomaly detection, which fail entirely in physical industrial scenes that require temporal information. Our framework successfully addresses this gap by incorporating flow-annotated frames and flow magnitude curves to capture essential temporal dynamics, enabling accurate detection of motion-dependent anomalies that manifest across multiple frames.

Generalization to IPAD. To further evaluate the robustness of our framework, we test MUSTARD on the IPAD dataset. As shown in Table 2, IPAD presents an exceptionally challenging environment where general-purpose MLLMs often perform near or below the level of random chance (0.50). This is likely due to the complex periodicity and subtle motion-based failures inherent in large-scale industrial equipment. Despite these challenges, MUSTARD achieves an AUROC of 0.572, surpass all the baselines. This performance confirms that our spatiotemporal awareness module provides a critical signal for detecting anomalies in longform industrial processes, even when global motion patterns are highly complex.

Table 2. Generalization Results on IPAD Dataset (AUROC %)
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>AUROC ↑</td></tr><tr><td rowspan=1 colspan=1>Video-LLaMA [39]Qwen2.5-VL-7B [í]Video-ChatGPT [20]Video-LLaVA [15]O-VAD [35]</td><td rowspan=1 colspan=1>45.547.247.950.056.5</td></tr><tr><td rowspan=1 colspan=1>Ours (MUSTARD)</td><td rowspan=1 colspan=1>57.2</td></tr></table>

Anomaly Explanation Performance. Table 3 compares our approach with state-of-the-art MLLM-based video understanding methods on anomaly explanation quality on the Phys-AD dataset. Our method achieves a PAEval score of 0.456, substantially outperforming all competing methods. This demonstrates the effectiveness of our spatiotemporal reasoning framework. General-purpose video MLLMs and existing MLLM-based anomaly detection methods, such as Anomaly Ruler, obtain considerably lower scores, suggesting that domain-specific design is crucial for anomaly explanation. The large performance gap validates our multisource spatiotemporal integration, which combines flowannotated frames, flow magnitude curves, and static descriptions to produce explanations that better align with ground-truth reasoning.

To further evaluate explanation quality, we adopt LLMas-a-Judge pairwise comparison as a complementary evaluation protocol. Specifically, we use GPT-4o as an expert judge to compare our explanation against each baseline explanation, given the ground-truth frame description. The judge selects the better answer between the two methods, and we report the win rate of other methods versus ours. As shown in Table 3, our method achieves win rates substantially above 50% against all baselines. This confirms that our explanations are not only quantitatively closer to the ground truth under PAEval, but also qualitatively preferred by a strong language model evaluator.

Together, both metrics show that our method provides stronger reasoning ability for video anomaly explanation, establishing a new state of the art for MLLM-based anomaly detection and interpretation.

Qualitative Results. In Figure 4, we illustrate two representative cases demonstrating how each component of our system contributes to improved anomaly detection. In the first example, direct usage of Qwen2.5-VL correctly recognizes that the button in the scene is not illuminated, but it has no notion of whether this observation is normal or abnormal. Our pattern-reasoning pipeline resolves this ambiguity by comparing the test case’s visual description against extracted normal patterns, allowing it to correctly identify the missing illumination as an anomaly. In the second example, both direct Qwen2.5-VL and our static single-source pipeline fail because the anomaly arises from subtle motion behavior rather than appearance. This is precisely where our spatiotemporal dynamics awareness becomes essential. MUSTARD integrates trajectory keyframes and the flow magnitude curve, enabling it to pinpoint that the rolling bearing remains static when it should be rotating. By contrasting these spatiotemporal signals with the normal dynamic patterns, MUSTARD successfully infers that the video is anomalous. These examples jointly demonstrate that our pattern-reasoning pipeline and our motion-aware module address different failure modes, and together enable the full MUSTARD pipeline to robustly detect diverse types of anomalies.

Table 3. Anomaly Explanation Quality and Pairwise Win Rates versus Ours (%). A lower win rate indicates that the baseline is less competitive than our method. All baselines achieve win rates below 50%, demonstrating the superiority of our method.
<table><tr><td>Method</td><td>PAEval ↑|</td><td>Win Rate vs. Ours</td></tr><tr><td>Video-LLaMA [39] Video-LLaVA [15]]</td><td>30.3 28.2</td><td>6.81 9.28</td></tr><tr><td>Qwen2.5-VL-7B [1] AnomalyRuler [32]</td><td>22.3 26.3</td><td>22.63 34.65</td></tr><tr><td>Ours (MUSTARD)</td><td>45.6</td><td></td></tr></table>

## 4.3. Ablation Study

We conduct a series of ablation studies to validate our core design choices, isolating the contributions of our flowguided keyframe selection, spatiotemporal sources.

Flow-guided Keyframe Selection. We have done experiment analyzing the impact of our flow-guided keyframe selection strategy against uniform sampling methods. Here, raw video frame pattern reasoning is used to ensure a fair comparison. Standard uniform sampling provides a low baseline AUC of 0.509. Our flow-guided keyframe selection improves this to 0.534 AUC. This demonstrates that flow-guided keyframe selection is capable of choosing the informative frames of objects in the video.

Spatiotemporal Information Sources. Table 4 evaluates the contribution of each spatiotemporal data source in our framework. All variants use our flow-guided keyframe selection strategy for fair comparison.

The raw frame description baseline achieves 0.534 AUC. Incorporating dynamic information through flow-annotated frames improves performance to 0.570 AUC, showing that explicit motion visualization enhances the MLLM’s spatiotemporal reasoning. Combining raw frame descriptions with flow magnitude curve pattern reasoning improves AUC to 0.647, confirming that global motion dynamics provide complementary information for anomaly detection. Our framework, which integrates pattern reasoning from flowannotated frames and flow magnitude curves while preserving raw frame descriptions for knowledge alignment, achieves the best performance at 0.681 AUC. This supports our hypothesis that these three sources encode complementary spatiotemporal cues for anomaly identification.

![](images/11649cafc545ede27cf852520ace26260d424cda2a15edbb509951aa4246b374.jpg)  
Figure 4. Example outputs from three methods: direct MLLM usage, a single-source static pattern-reasoning pipeline, and our proposed MUSTARD method.

Table 4. Ablation Study of Spatiotemporal Modalities on Detection and Reasoning. Scores are scaled by 100 for readability.
<table><tr><td>Raw Frames</td><td>Traj. Frames</td><td>Flow Mag.</td><td>AUROC↑</td><td>PAEval ↑</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">√ √</td><td></td><td></td><td>53.4 57.0</td><td>15.79 11.20</td></tr><tr><td>√</td><td>√</td><td>64.7</td><td>45.88</td></tr><tr><td>√</td><td>√</td><td>67.1</td><td>44.63</td></tr><tr><td>√</td><td>√</td><td>√</td><td>68.1</td><td>45.62</td></tr></table>

For PAEval, the raw frame description baseline achieves 0.1579, while flow-annotated frames alone decrease the score to 0.1120 despite improving AUC. This reveals a trade-off: motion arrows help detection but introduce visual artifacts that may impair explanation quality. Adding flow magnitude curves to raw frames yields a substantial improvement to 0.4588, nearly tripling the baseline score. Replacing raw frames with flow-annotated frames slightly reduces PAEval to 0.4463, further indicating the noise introduced by motion annotations. Our full framework combines raw descriptions alongside both dynamic sources, recovers PAEval to 0.4562, effectively matching the raw-frame-pluscurve setting while achieving superior AUC.

This ablation highlights three findings: (1) flow magnitude curves provide the largest boost for both detection and explanation, (2) flow-annotated frames improve detection but must be integrated carefully to avoid degrading explanations, and (3) our multi-source fusion strategy effectively leverages all three complementary modalities by balancing their strengths and mitigating their weaknesses.

## 5. Conclusion

We introduce MUSTARD, a novel framework that equips MLLMs with optical flow-guided spatiotemporal awareness for IAD. By using optical flow to select keyframes capturing critical motion irregularities and incorporating two flow-augmented modalities, MUSTARD significantly enhances fine-grained temporal anomaly identification. To our knowledge, MUSTARD is one of the first MLLMbased frameworks for dynamics-based IAD. Our experiments demonstrate SOTA performance in spatiotemporal anomaly detection with interpretable explanations.

## Acknowledgements

This work was supported by the National Science and Technology Council (NSTC) of Taiwan, under Grant 115-2634- F-002-012, and by the NTU AI Center of Research Excellence (AI-CoRE). Shao-Yuan Lo is honored to be supported by the Yushan Young Fellow Program by the Ministry of Education (MOE) of Taiwan.

## References

[1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 2, 3, 6, 7

[2] Shuai Bai et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025. 3

[3] Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger. Mvtec ad–a comprehensive real-world dataset for unsupervised anomaly detection. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019. 1

[4] Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, and Carsten Steger. Beyond dents and scratches: Logical constraints in unsupervised anomaly detection and localization. International Journal of Computer Vision, 130 (4):947–969, 2022. 1

[5] ByteDance Seed Team. Seed1.5-vl technical report. arXiv preprint arXiv:2505.07062, 2025. 3

[6] Yunkang Cao, Jiangning Zhang, Luca Frittoli, Yuqi Cheng, Weiming Shen, and Giacomo Boracchi. Adaclip: Adapting clip with hybrid learnable prompts for zero-shot anomaly detection. In European Conference on Computer Vision, 2024. 1, 2

[7] Yuval Eldar, Michael Lindenbaum, Moshe Porat, and Yehoshua Y Zeevi. The farthest point strategy for progressive image sampling. In IEEE Transactions on Image Processing, 1997. 4

[8] Dong Gong, Lingqiao Liu, Vuong Le, Budhaditya Saha, Moussa Reda Mansour, Svetha Venkatesh, and Anton van den Hengel. Memorizing normality to detect anomaly: Memory-augmented deep autoencoder for unsupervised anomaly detection. In IEEE/CVF International Conference on Computer Vision, 2019. 3

[9] Zhaopeng Gu, Bingke Zhu, Guibo Zhu, Yingying Chen, Ming Tang, and Jinqiao Wang. Anomalygpt: Detecting industrial anomalies using large vision-language models. In AAAI Conference on Artificial Intelligence, 2024. 2

[10] Yuxiang Guo, Faizan Siddiqui, Yang Zhao, Rama Chellappa, and Shao-Yuan Lo. Stimuvar: Spatiotemporal stimuli-aware video affective reasoning with multimodal large language models. In International Journal of Computer Vision, 2025. 3, 4

[11] Lars Heckler-Kram, Jan-Hendrik Neudeck, Ulla Scheler, Rebecca Konig, and Carsten Steger. The mvtec ad 2 dataset:¨ Advanced scenarios for unsupervised anomaly detection. arXiv preprint arXiv:2503.21622, 2025. 1

[12] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Weli hinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 5

[13] Jongheon Jeong, Yang Zou, Taewan Kim, Dongqing Zhang, Avinash Ravichandran, and Onkar Dabeer. Winclip: Zero-/few-shot anomaly classification and segmentation. In CVPR, 2023. 1, 2

[14] Wenqiao Li, Yao Gu, Xintao Chen, Xiaohao Xu, Ming Hu, Xiaonan Huang, and Yingna Wu. Towards visual discrimination and reasoning of real-world physical dynamics: Physicsgrounded anomaly detection. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025. 2, 5, 11

[15] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-LLaVA: Learning united visual representation by alignment before projection. In Conference on Empirical Methods in Natural Language Processing, 2024. 2, 3, 4, 6, 7

[16] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Conference on Neural Informa tion Processing Systems, 2023. 2, 3

[17] Jinfan Liu, Yichao Yan, Junjie Li, Weiming Zhao, Pengzh Chu, Xingdong Sheng, Yunhui Liu, and Xiaokang Yang. Ipad: Industrial process anomaly detection dataset. IEEE Transactions on Circuits and Systemsfor Video Technology, 2025. 2, 5

[18] Yinhong Liu, Han Zhou, Zhijiang Guo, Ehsan Shareghi, Ivan Vulic, Anna Korhonen, and Nigel Collier. Aligning with hu-´ man judgement: The role of pairwise preference in large language model evaluators. arXiv preprint arXiv:2403.16950, 2024. 11

[19] Shao-Yuan Lo, Poojan Oza, and Vishal M Patel. Adversarially robust one-class novelty detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022. 2

[20] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Khan. Video-ChatGPT: Towards detailed video understanding via large vision and language models. In ACL, 2024. 6, 7

[21] Hyunjong Park, Jongyoun Noh, and Bumsub Ham. Learning memory-guided normality for anomaly detection. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020. 3, 5

[22] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learn ing transferable visual models from natural language supervision. In International Conference on Machine Learning, 2021. 2

[23] Jiaqi Tang, Hao Lu, Ruizheng Wu, Xiaogang Xu, Ke Ma, Cheng Fang, Bin Guo, Jiangbo Lu, Qifeng Chen, and Ying cong Chen. Hawk: Learning to understand open-world video anomalies. In Conference on Neural Information Processing Systems, 2024. 3

[24] Xi Tang, Jihao Qiu, Lingxi Xie, Yunjie Tian, Jianbin Jiao, and Qixiang Ye. Adaptive keyframe sampling for long video understanding. In IEEE/CVF Conference on Computer Vi sion and Pattern Recognition, 2025. 3, 4

[25] Gemma Team. Gemma 3 technical report. arXiv preprint arXiv:2503.19786, 2025. 6

[26] Chengjie Wang, Wenbing Zhu, Bin-Bin Gao, Zhenye Gan, Jiangning Zhang, Zhihao Gu, Shuguang Qian, Mingang Chen, and Lizhuang Ma. Real-iad: A real-world multi-view dataset for benchmarking versatile industrial anomaly detection. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 1

[27] Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei Zhu, Binghuai Lin, Yunbo Cao, Lingpeng Kong, Qi Liu, Tianyu Liu, et al. Large language models are not fair evaluators. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 9440–9450, 2024. 11

[28] Weiyun Wang et al. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025. 3

[29] Jhih-Ciang Wu, He-Yen Hsieh, Ding-Jie Chen, Chiou-Shann Fuh, and Tyng-Luh Liu. Self-supervised sparse representation for video anomaly detection. In European Conference on Computer Vision, 2022. 5

[30] Peng Wu, Xuerong Zhou, Guansong Pang, Lingru Zhou, Qingsen Yan, Peng Wang, and Yanning Zhang. Vadclip: Adapting vision-language models for weakly supervised video anomaly detection. In AAAI Conference on Artificial Intelligence, 2024. 3, 5

[31] Jiacong Xu, Shao-Yuan Lo, Bardia Safaei, Vishal M Patel, and Isht Dwivedi. Towards zero-shot anomaly detection and reasoning with multimodal large language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025. 2

[32] Yuchen Yang, Kwonjoon Lee, Behzad Dariush, Yinzhi Cao, and Shao-Yuan Lo. Follow the rules: Reasoning for video anomaly detection with large language models. In European Conference on Computer Vision, 2024. 2, 3, 6, 7, 12

[33] Jaron Yeh, Yen-Wei Chang, Jiang Liu, and Shao-Yuan Lo. Industrial anomaly detection via defect-grounded reasoning in visual latent space. arXiv preprint arXiv:2609.29457, 2026. 2

[34] Zhiyuan You, Lei Cui, Yujun Shen, Kai Yang, Xin Lu, Yu Zheng, and Xinyi Le. A unified model for multi-class anomaly detection. In Conference on Neural Information Processing Systems, 2022. 1, 2

[35] Mei Yuan, Qi Long, Qifeng Wu, Zhenyang Li, Yizhou Zhao, Lei Wang, Yang Liu, and Min Xu. O-vad: Industrial video anomaly detection through object-centric tracking and reasoning. In European Conference on Computer Vision, pages 285–302. Springer, 2026. 2, 6, 7

[36] Christopher Zach, Thomas Pock, and Horst Bischof. A duality based approach for realtime tv-l1 optical flow. In Pattern Recognition, pages 214–223. Springer Berlin Heidelberg, 2007. 4

[37] Luca Zanella, Willi Menapace, Massimiliano Mancini, Yiming Wang, and Elisa Ricci. Harnessing large language models for training-free video anomaly detection. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 3, 6

[38] Vitjan Zavrtanik, Matej Kristan, and Danijel Skocaj. Draem-ˇ a discriminatively trained reconstruction embedding for surface anomaly detection. In IEEE/CVF International Confer ence on Computer Vision, 2021. 1, 2

[39] Hang Zhang, Xin Li, and Lidong Bing. Video-LLaMA: An instruction-tuned audio-visual language model for video understanding. In Conference on Empirical Methods in Natural Language Processing, 2023. 2, 3, 4, 6, 7

[40] Huaxin Zhang, Xiaohao Xu, Xiang Wang, Jialong Zuo, Xi aonan Huang, Changxin Gao, Shanjun Zhang, Li Yu, and Nong Sang. Holmes-vau: Towards long-term video anomaly understanding at any granularity. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025. 3

[41] Qihang Zhou, Guansong Pang, Yu Tian, Shibo He, and Jiming Chen. AnomalyCLIP: Object-agnostic prompt learning for zero-shot anomaly detection. In International Conference on Learning Representations, 2024. 2

[42] Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang, and Onkar Dabeer. Spot-the-difference self-supervised pre training for anomaly detection and segmentation. In Euro pean Conference on Computer Vision, 2022. 1

# Spatiotemporal Dynamics-Aware Industrial Anomaly Detection and Reasoning

Supplementary Material

## A1. Limitations and Potential Negative Societal Impacts

Limitations. Consistent with prevalent MLLM research, our framework assumes the underlying backbones possess sufficient competence. Using weaker models can inevitably compromise overall effectiveness. For instance, models may exhibit hallucinations or struggle to accurately describe fine-grained object states (see Sec. A3). Further examining these backbone dependencies and their impact offers a compelling avenue for future study.

Textual descriptions provide a compact way to compare test videos with normal references while avoiding the higher cost of direct visual-feature matching. Despite potential description errors, MUSTARD still outperforms vanilla MLLM baselines. Nevertheless, its performance remains dependent on the perceptual capabilities of the underlying MLLM. Incorporating visual features as complementary evidence is a promising direction for future work.

Potential Negative Social Impact. The proposed method involves detailed motion analysis, which may enable malicious actors to adapt the system for unauthorized surveillance. Additionally, the reliance on VLMs/LLMs contributes to general environmental carbon footprints. To mit igate these risks, security mechanisms and energy-efficient inference techniques could be integrated.

## A2. Evaluation metrics details

AUC For non-LLM-based methods, the framework directly outputs an anomaly score. For LLM-based methods, we prompt the LLM to generate both an anomaly score (ranging from 0 to 1, where 0 indicates completely normal and 1 indicates completely anomalous) and the anomaly explanation. We extract the anomaly score for AUC evaluation and the anomaly explanation for PAEval and LLM-asa-Judge metrics.

PAEval PAEval is the metric proposed in the Phys-AD [14]. The original metric contains two sub-metric: video descriptions and anomaly explanations; we concentrate on anomaly explanations because they provide a direct assessment of a model’s ability to capture the reasoning rationale underlying anomaly detection. Each video has 5 groundtruth anomaly explanations. PAEval measures semantic text similarity between test anomaly explanations and groundtruth anomaly explanations, yielding a score between 0 and 1, and higher score reflects higher similarity, and therefore better anomaly explanations. PAEval selects the highest similarity score across all 5 ground-truth anomaly explanations for each test video. We extract the explanation segment from the LLM output and compute similarity using GPT-4o (2024-08-06), consistent with the Phys-AD dataset evaluation protocol.

LLM-as-a-Judge We employ LLM-as-a-Judge with pairwise comparison for each anomaly type in all objects, as LLMs demonstrate greater robustness in comparative judgments than absolute scoring [18] in PAEval. The judge LLM receives the ground-truth video description from the PAEval metric (though the metric is not used here) and two candidate anomaly explanations, then selects the anomaly explanation that better captures the anomaly. Win rate is calculated as the proportion of instances where one method’s explanation is preferred over another. We use GPT-4o (2024-08-06) as the judge LLM and evaluate 5 representative videos per anomaly type. To mitigate positional bias [27], we randomly swap the order of the two candidate explanations in each comparison.

## A3. Qualitative Discussion

## A3.1. Data Source

We provide a qualitative analysis of our MUSTARD method across various anomaly types to demonstrate the effectiveness of our proposed multi-source data representations. Our approach leverages three complementary data sources: raw frame descriptions (direct visual observations), trajectory frame descriptions (motion-enhanced visualizations with overlaid arrows), and flow-mag curves (optical flow magnitude curves capturing global motion patterns). While initial experiments with raw frames alone yielded suboptimal results, we recognize their fundamental importance and strategically integrate all three sources in MUSTARD to enhance reasoning capabilities.

## Success Cases

We first examine scenarios where the integration of trajectory and flow-mag data successfully compensates for visual ambiguity or enhances detection confidence.

Rolling Bearing and Lock. These cases (Figure 6 and Figure 7) exemplify the strength of trajectory frame descriptions for motion-based anomalies. Normal videos display consistent arrow patterns indicating smooth mechanical operations, while abnormal videos show irregular or absent arrows reflecting mechanical failures or jamming. The trajectory descriptions provide clear visual signatures that enable reliable anomaly detection, demonstrating how explicit motion encoding helps the model identify anomalies that are subtle in raw frames alone.

Clip. This scenario (Figure 8) highlights the superiority of flow-mag curves when frame-based analysis proves insufficient. Although visual appearance and local motion patterns are similar between normal and abnormal clips, the optical flow magnitude curves reveal distinctive global motion signatures. Normal clips exhibit characteristic flow-mag patterns corresponding to proper mechanical action, while abnormal clips display measurably different magnitude profiles. This explicitly validates our motivation for introducing flow-mag curves as a complementary data source that captures motion information at a holistic level.

## Challenging Cases

In contrast, the following cases highlight scenarios where motion cues are minimal, forcing reliance on the specific backbone’s visual encoding capabilities.

Toothpaste. The anomaly manifests as a small toothpaste leak, which is a subtle visual change requiring fine-grained observation. Motion patterns remain nearly identical between normal and abnormal videos, rendering trajectory descriptions and flow-mag curves ineffective. Consequently, detection relies entirely on the backbone’s raw visual perception. In this scenario, our selected backbone, Qwen2.5- VL-7B, struggles to notice minor discrepancies, exposing the intrinsic limitations regarding micro-level detail extraction in smaller-scale models compared to larger SOTA LLMs.

Ball. The anomaly involves ball deformation or deflation after compression by robotic arms. While more substantial than the toothpaste leak, this visual change remains challenging for the selected 7B-parameter backbone to perceive accurately. Since trajectory frame description pattern and flow-mag curves pattern are similar across normal and anomaly videos, both method can offer limited assistance. Detection again depends heavily on the naked visual perception capabilities of the model. The failure here indicates that the perceptual bottleneck of the specific Qwen2.5-VL-7B backbone extends beyond micro-level details to medium-scale visual changes when relying solely on standard visual encoders.

## Summary

Our success cases of rolling bearing, lock, and clip, demonstrate the considerable strength of our carefully crafted multi-source data representations. By designing trajectory frame descriptions and flow-mag curves specifically to address motion-based anomalies, we have created a rich, complementary information set that significantly enhances anomaly detection. These data sources effectively encode motion at different scales and perspectives, enabling sophisticated reasoning about physical dynamics.

The challenging cases reveal a specific constraint related to the implementation choice rather than the methodology itself. We utilized a ”naked” VLM backbone (Qwen2.5- VL-7B) without applying specialized methods or visual adapters often proposed to enhance visual perception. Consequently, the performance bottleneck reflects the capacity of this specific parameter-efficient backbone to perceive fine-grained details and medium-scale changes when motion cues are insufficient. This suggests that the limitations observed are not inherent to the MUSTARD approach, but rather a consequence of the raw perception capabilities of the selected 7B model. As we adopted a generalpurpose, smaller-scale model for this study, future improvements could involve integrating stronger VLM backbones or specialized visual perception modules. As the underlying VLM capabilities scale, we anticipate that the rich, multi-source representations we have created will become even more effective, fully realizing their potential for comprehensive industrial video anomaly detection.

## A3.2. Normal Pattern

In Figure 9 and Figure 10, we show examples of trajectory frame description normal pattern and flow-mag curve normal pattern of rolling bearing and clip objects. We also show the reasoning results of both pattern with each a normal video and anomaly video frame descriptions. From the patterns and reasonings, we can see that our method can find the anomaly and give accurate anomaly scores to the videos.

## A4. Comparison of Normal Representation Strategies

Our framework extracts normal video descriptions to construct normal patterns. We compare against rule-based representations from proposed by Follow-the-Rules [32]. To ensure fair evaluation, we apply both methods to the same trajectory frames and flow magnitude curves and with identical fusion procedures, and the result is in Table 5. The rule-based normal representation approach achieves an AUC of 0.527, compared to our pattern-based method’s AUC of 0.681. This substantial performance gap demonstrates that extracted normal patterns capture motion characteristics more effectively than extracted normal rules. The superior performance stems from patterns’ ability to model complex and subtle motion dynamics that rigid rule sets cannot adequately represent.

Table 5. AUROC on different normal representation Strategies
<table><tr><td>Normal representation</td><td>Rules</td><td>Pattern (MUSTARD)</td></tr><tr><td>AUC</td><td>0.527</td><td>0.681</td></tr></table>

## A5. Computational Efficiency

Runtime Evaluation. To evaluate inference efficiency, we sample 10 test videos from each of the 22 object categories in Phys-AD, resulting in 220 videos in total. We compare MUSTARD with vanilla Qwen2.5-VL-7B inference using 8 frames (Qwen-8) and 40 frames (Qwen-40). As shown in Figure. 5, Qwen-8, Qwen-40, and MUSTARD require 12.0, 87.3, and 64.3 seconds per video, respectively, while achieving AUROC scores of 0.500, 0.492, and 0.656. These AUROC values are measured on the sampled 220- video subset and may therefore differ slightly from the fulldataset results reported in the main paper. We observe that the additional inference time required by MUSTARD corresponds to a measurable improvement in performance, whereas Qwen-40 incurs higher latency without a comparable gain. This demonstrates the effectiveness of MUS-TARD’s additional inference-time computation.

Memory Footprint and Deployment. As shown in Figure. 5 and Table 6, MUSTARD has a peak GPU memory footprint of 14.7 GB during pattern reasoning and fusion, compared with 18.6 GB for Qwen-8 and 21.1 GB for Qwen 40. The MLLM/LLM stages are executed sequentially, so the models do not need to reside in GPU memory simultaneously. The optical-flow preprocessing is performed on the CPU at 640 × 360 resolution and processes consecutive frame pairs rather than loading the full video into memory. For a representative 120-frame video with 4,000 tracked points, the explicitly allocated frame, flow, and trajectory arrays require approximately 15–30 MiB of RAM. This estimate excludes OpenCV’s internal TV-L1 optimization buffers, whose exact peak allocation depends on the OpenCV implementation.

## A6. Alternative Backbone MLLMs

To demonstrate MUSTARD’s generalizability across backbone MLLMs, we further evaluate it using InternVL3.5-8B as an alternative backbone. For each MUSTARD variant, the backbone MLLM describes all three modalities, while Gemma3-12B performs pattern extraction, reasoning, and multi-source reasoning fusion. This comparison isolates gains from stronger foundation models from those from MUSTARD’s motion-aware representations and normalreference reasoning.

As shown in Table 7, applying MUSTARD yields substantial AUC gains for both MLLM backbones. Vanilla

Qwen2.5-VL-7B and InternVL3.5-8B baselines achieve only 0.501 and 0.502 AUC, respectively; both are near chance level. Their corresponding MUSTARD variants, in contrast, reach 0.681 and 0.613. This substantial gain demonstrates that the proposed motion-aware representations and normal-reference reasoning are effective across different backbone MLLMs.

## A7. Examples of Prompts

We provide the complete prompts used throughout MUS-TARD. Figure. 11 presents the prompts for generating raw-frame, trajectory-frame, and flow-magnitude descriptions. Figure. 12 presents the prompts for extracting normal patterns from reference videos. Figure. 13 presents the prompts for comparing test-video descriptions against the extracted normal patterns and generating modality-specific anomaly scores. Finally, Figure. 14 presents the prompt for fusing the multi-source reasoning results into the final anomaly score and explanation.

Table 6. GPU memory footprint of the major MUSTARD inference stages.
<table><tr><td>Stage</td><td>Model</td><td>GB</td></tr><tr><td>Raw-frame Desc.</td><td>Qwen2.5-VL-7B</td><td>7.7</td></tr><tr><td>Flow-mag Desc.</td><td>Qwen2.5-VL-7B</td><td>8.0</td></tr><tr><td>Reasoning + Fusion</td><td>Gemma3-12B</td><td>14.7</td></tr><tr><td>Peak</td><td></td><td>14.7</td></tr></table>

![](images/a3a053234a0a21401bc143c6b9cdb436d964371964c162724c5c8bd39403460c.jpg)  
(a) Runtime–accuracy trade-off.

![](images/1f4e6e36d7317393a227785c46085c075169e5f3f026312fba1ece4e479e7c6f.jpg)  
(b) Peak GPU memory.  
Figure 5. Computational efficiency on the 220-video subset.

![](images/96c8e7f42fe77101f10d1f68d0b4478f09c5ddeccca14492b085592779ac7ba2.jpg)

Table 7. Comparison with different backbone MLLMs on the Phys-AD dataset. For MUSTARD variants, the description backbone is shared across raw-frame, trajectory-frame, and flow-magnitude descriptions.
<table><tr><td>Method</td><td>Description Backbone</td><td>Pattern / Reasoning</td><td>AUROC (%)</td></tr><tr><td>Vanilla backbone MLLM</td><td>Qwen2.5-VL-7B</td><td>Direct</td><td>50.1</td></tr><tr><td rowspan="2">MUSTARD</td><td>Qwen2.5-VL-7B</td><td>Gemma3-12B</td><td>68.1</td></tr><tr><td>InternVL3.5-8B</td><td>Direct</td><td>50.2</td></tr><tr><td>Vanilla backbone MLLM MUSTARD</td><td>InternVL3.5-8B</td><td>Gemma3-12B</td><td>61.3</td></tr></table>

## • Rolling Bearing - Normal

![](images/46ceac4b08a83df001909472bf490a7638323a646a61195656d07423ec68f40c.jpg)  
Figure 6. An example of three different data sources for the Rolling Bearing object.

![](images/fb523dd7eef81b02639d480a0e858c928a63fad3031b9ff4a43f57be78dd818b.jpg)

• Lock - Normal  
![](images/cc2cb368ad861cb408d081d5e73f170e853ef14fb38a27bb1cb1da2e80175470.jpg)

![](images/fe1ba164602246ac0f3cada2400cd34e2677466ba17098ddbe628a36d02209b6.jpg)

![](images/eb7d44ea172c558ffa9d13cb93b8a1636c284d1876c7b795ffebfedafe1698c3.jpg)  
Figure 7. An example of three different data sources for the Lock object.

## • Clip - Normal

![](images/b96858daed10cd3c7cfd74bd0206f27c8db6169c33706da5e085f48633a3e2ad.jpg)

![](images/0ccbd3c2fb6049816a17ae5150cb4468cc5dbbf9f3502231b3f41606c3a926f9.jpg)

![](images/30d161b7adfe42df9c35dc2ebefd5f3affba873013644ec93ce61b17b9e0bd34.jpg)  
Figure 8. An example of three different data sources for the Clip object.

![](images/998dcb0fe9cc7329b9e5d87580d02ea5ed4bd38e188bdd02375974719067f368.jpg)  
Figure 9. An example of the trajectory-frame normal pattern and corresponding reasoning output.

• Clip - Normal's Reasoning  
![](images/329d34c1567157f38cdfa07c336e8a4d23f4283922f8f87d186ffd1efe301256.jpg)

![](images/eade6e48793e511d5d73073ef3538ef9a812ba2ea22b147aa7533b31b82c909f.jpg)

![](images/5366d93a2e7f887c48f123907b47d3b3e22fb5690f06e2de42d51714f9cf2e82.jpg)  
Figure 10. An example of the flow-magnitude-curve normal pattern and corresponding reasoning output.

![](images/c5a98b5ddf096bc92256a94651785cdd4cc7ac997440494eb533ab8dc7c63240.jpg)  
Figure 11. Prompts of each source fed to Qwen2.5-VL-7B to generate frame descriptions.

• Trajectory Frame Normal Pattern Extraction Prompt   
INPUT:   
You will receive multiple descriptions of Frame {frame\_number} from different video sets.   
All these frames show the SAME physical action but from different instances.   
CRITICAL RULES:   
1. \*\*Summarize the general pattern\*\* based on what is explicitly described. Do NOT predict or infer.   
2. \*\*Prioritize brevity and clarity.\*\* Capture the main objects and actions.   
3. Use concise, descriptive language, You may synthesize original wording to create a clear, general   
pattern. \*\*Do not include source references (e.g., "D1", "D5").\*\*   
HANDLING VARIATIONS:   
- \*\*Generalize non-arrow setting variations,\*\* For settings like obiect color, size, or background, use a   
general term (e.g., "a ball of various colors") instead of listing all variants.   
- \*\*Arrow variations are critical\*\* and must be captured in detail, as described in the section below.   
ARROWS (HIGH PRIORITY):   
- \*\*All arrow variations are critical and MUST be captured.\*\* This is the most important part of the   
pattern.   
- \*\*If arrow states differ significantly (e.g., some instances have "no arrows." while others have "a few   
arrows." and others have "numerous arrows"). all of these distinct states must be described\*\* as part   
of the general pattern.   
- Arrows are always ORANGE.   
- For instances \*with\* arrows, \*\*summarize the arrows' general placement and direction.\*\*   
- Instead of listing every single arrow, describe the \*\*overall pattern\*\* for each state (e.g., "one   
state has no arrows: another state has numerous short arrows...").   
- \*\*Mention the key obiect parts\*\* the arrows are associated with (e.g.. "arrows on the bumper and   
wheels").   
- \*\*Describe the general directions\*\* (e.g., "pointing mostly outward").   
- Include size (short/medium/long) if it's a consistent part of the pattern.   
- Example: "Pattern includes three states: one with no arrows; one with a few isolated arrows at the   
corners; and one with numerous short arrows scattered across the body, pointing outward."   
ACTIONS:   
- Capture all actions mentioned (pushing, pulling, rotating, etc.).   
- Include key object states (moving/stili, open/closed, etc.).   
YOUR TASK:   
Write a \*\*concise, general pattern\*\* that includes:   
- The main objects.   
- All actions and motions.   
- A summarized description of all arrow patterns/states (placement, direction, size, quantity).   
Output exactly one line in this format:   
<Frame {frame\_number}>: [concise general pattern capturing main objects, actions, summarized arrow   
patterns, and important variations]

![](images/fe4ff0db58fccf1ac2a6534b309e732f0fd211fc858fc1be17d4a62b10eb601c.jpg)  
Figure 12. Prompts fed to Gemma3-12B to capture normal patterns of trajectory-frame descriptions and flow-magnitude curves.

INPUT:   
- A frame description set with 8 frames (Frame 1 to Frame 8). in time order   
- A pattern showing what NORMALLY happens in this physical action   
WHAT IS ANOMALY:   
Anomaly means something STRANGE or UNEXPECTED happened in the video compared to the pattern.   
This includes:   
- Unexpected objects appearing   
- Unexpected or different motions/actions   
- Strange movements that don't match the pattern   
ARROWS ARE CRITICAL - USE THEM TO UNDERSTAND MOVEMENT:   
- All arrows are ORANGE color   
- Arrows show the FLOW and DIRECTION of movement - use them to understand how objects should move   
- Check arrow direction to understand what part of the object can move and how   
- If arrows are MISSING or EXTRA compared to the pattern → ANOMALY   
- If arrows point differently than pattern → wrong movement → ANoMALY   
- If arrows are on different parts of the obiect than pattern → ANoMALY   
- If arrow size differs significantly (large vs small)→ different movement magnitude → potential   
ANOMALY   
YOUR TASK:   
1. Look at the pattern's arrows to understand the expected movement flow   
2. Look at the frame's arrows to see the actual movement flow   
3. Compare: Do arrows match (direction, position on object, size)?   
4. Compare: Are objects and actions the same?   
5. If anything is different or strange → ANOMALY   
SCORING:   
- 0.0-0.2 = No anomaly (everything matches)   
- 0.3-0.5 = Minor anomaly (small differences)   
- 0.6-0.8 = Moderate anomaly (clear differences in movement or objects)   
- 0.9-1.0 = Severe anomaly (completely wrong movement/action or wrong arrow direction/position)   
OUTPUT FORMAT:   
Brief reasoning, then end with:   
Anomaly Score: <your score>   
Do not write anything after the score line.

## • Flow Magnitude Curve Pattern Reasoning Prompt

INPUT:   
- An optical flow data summary (13 fields) for a single video   
- A pattern showing what NORMALLY happens in this type of video (also 13 fields)   
WHAT IS ANOMALY IN OPTICAL FLOW:   
Anomaly means the motion pattern in the video is STRANGE or UNEXPECTED compared to the normal pattern.   
This includes:   
- UNEXPECTED MOTION CHARACTERISTICS: Different max/min values, different peak/valley counts   
- ABNORMAL TRENDS: Different increase/decrease patterns than expected   
- UNUSUAL VIBRATIONS: Vibrating when shouldn't, or smooth when should vibrate   
- WRONG STABILITY: Unstable when should be stable, or vice versa   
- DIFFERENT SMOOTHNESS: Jerky when should be smooth, or vice versa   
- ANOMALOUS TRANSITIONS: Unexpected sudden changes or missing expected transitions   
HOW TO COMPARE OPTICAL FLOW DATA:   
1. NUMERICAL FIELDS (max. min. peak number. valley number):   
- Check if values fall within the expected range from the pattern   
- Small deviations (±10-20%) may be normal variation   
- Large deviations (>50%) indicate anomaly   
2. TEXT FIELDS (descriptions of motion characteristics):   
- Compare each field description to the pattern   
- If pattern shows variations (e.g., "Yes / No"), the video should match ONE of them   
- If video shows something NOT in the pattern variations → ANOMALY   
3. OVERALL MOTION BEHAVIOR:   
- Does the summary description match the pattern's expected behavior?   
- Are the motion characteristics (smooth/jerky, stable/unstable, etc.) consistent?   
YOUR TASK:   
Compare the optical flow data to the pattern field by field:   
1. Check numerical values against expected ranges   
2. Check text descriptions against expected variations   
3. Assess overall motion behavior consistency   
4. If significant differences exist → ANOMALY   
SCORING GUIDE:   
- 0.0-0.2 = No anomaly (all fields match pattern or within normal variation)   
- 0.3-0.5 = Minor anomaly (1-2 fields show smal1 deviations)   
- 0.6-0.8 = Moderate anomaiy (several fields differ, or important fields show clear deviations)   
- 0.9-1.0 = Severe anomaly (many fields wrong, or critical characteristics completely different)   
OUTPUT FORMAT:   
Brief reasoning comparing each field, then end with:   
Anomaly Score: <your score>   
Do not write anything after the score line.  
Figure 13. Prompts fed to Gemma3-12B to compare frame descriptions with the normal patterns.

INPUT:You will receive:   
1. \*\*Raw Frame Description\*\*: The visual description of the original video frames (8 frames).   
- This is what we're trying to explain - the actual visual content of the video.   
2. \*\*Dynamic Source Analysis\*\*:   
- Anomaly reasoning: Explanation from dynamic description analysis (includes arrow motion indicators)   
3. \*\*Optical Flow Source Analysis\*\*:   
- Anomaly reasoning: Explanation from optical flow motion pattern analysis   
YOUR TASK:   
- First, read the \*\*Raw Frame Description\*\* carefully to understand the visual and physical setting of the   
video.   
- Then, integrate the evidence from ALL other modalities to infer whether the clip behaves normally or   
abnormally.   
- Treat strong consistency with expected physical or behavioral patterns as evidence of normality.   
- Treat deviations in motion, stability, rhythm, or object response as evidence of anomaly.   
- Use all information holistically; do not rely solely on any single modality.   
REASONING REQUIREMENT (FOR THE FIRST 'Reasoning:' FIELD):   
- Write a concise 2-3 sentence summary combining the Raw Frame Description and Dynamic Source Analysis and   
Optical Flow Source Analysis reasonings.   
- Focus on the \*physical or behavioral correctness\* of the object's motion, timing, and stability.   
- Clearly conclude whether the clip overall appears \*\*normal\*\* or \*\*anomalous\*\*, and briefly justify why.   
SCORING GUIDE (FOR 'Anomaly Score:'):   
- 0.0-0.2: No anomaly (fully consistent with normal behavior)   
- 0.3-0.5: Minor anomaiy (1-2 mild irregularities: generally functional).   
- 0.6-0.8: Moderate anomaly (clear abnormal pattern or multiple deviations).   
- 0.9-1.0: Severe anomaly (major malfunction or very different motion pattern).   
FINAL EXPLANATION REQUIREMENT (HUMAN-FRIENDLY):   
- If the main object type is mentioned in the inputs (e.g., ball, USB drive, button, gear), use that exact   
name consistently. Otherwise, choose a simple, reasonable name such as "the object" or "the device".   
- In 'Final Explanation:', write a short, clear summary in everyday language.   
- The first sentence must explicitly state whether the clip is normal, roughly normal, slightly anomalous, or   
anomalous (e.g., 'This is normal.','This is roughly normal.', 'This is slightly anomalous.', or 'This is   
anomalous.').   
- Follow with 1-2 sentences (no more than 3 total) explaining why, based on what happens in the video.   
- Combine understanding from the Static Source and all modality analyses.   
- Clearly describe what happens to the object (e.g., moves smoothly, vibrates irregularly, gets stuck,   
deforms too much, fails to return, etc.).   
- Do \*\*not\*\* mention specific modalities, arrows, plots, metrics, or scores.   
- Keep the explanation intuitive, concise, and consistent with the anomaly score and the overall reasoning.   
OUTPUT FORMAT (STRICT):   
Reasoning: <2-3 sentence technical summary combining raw frame + multi-modal reasoning>   
Anomaly Score: <one number in [0.0,1.0]>   
Final Explanation: <2-4 sentences in natural language describing the object's behavior and why it is normal   
or abnormal>   
Do not output any additional sections or commentary.   
Only output exactly three lines: 'Reasoning:'. 'Anomaly Score:', and 'Final Explanation:'.   
Raw Frame Description:   
<RAW\_FRAME\_DESCRIPTION>   
Dynamic frames' reasoning (motion-annotated descriptions):   
<DYNAMIC\_SOURCE\_REASONING>   
Motion magnitude reasoning (optical flow plot):   
<OPTICAL\_FLOW\_SOURCE\_REASONING>  
Figure 14. Prompt provided to Gemma3-12B for fusing reasoning from trajectory frames, the flow-magnitude curve, and raw-frame descriptions.