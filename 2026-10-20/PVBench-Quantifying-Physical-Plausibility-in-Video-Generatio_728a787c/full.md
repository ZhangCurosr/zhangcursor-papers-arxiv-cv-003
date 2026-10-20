# PVBench: Quantifying Physical Plausibility in Video Generation with Object-Level Physical Measurements

Kaiwei Chen<sup>1</sup>, Moran Li<sup>2</sup>, Zhenye Gan<sup>3</sup>, Yabiao Wang<sup>3</sup>, Chengjie Wang<sup>3</sup>, Kexin Zhang<sup>1</sup>, Yong Liu<sup>1</sup>, Jiangning Zhang<sup>1</sup>

<sup>1</sup>APRIL Lab, Zhejiang University, <sup>2</sup>Huazhong University of Science and Technology, <sup>3</sup>Tencent Youtu Lab

Recent video generation models produce visually compelling outputs, yet their physical plausibility remains poorly measured. Physical plausibility asks whether generated objects obey geometry, kinematics, dynamics, and conservation laws. Existing benchmarks rely on coarse visual-language model judgments or target embodied world models rather than video generation itself. We introduce PVBench, a physical video benchmark that quantitatively measures the plausibility of generated videos across 46 dimensions organized into seven domains, spanning: i) instruction and semantic fidelity, ii) behavioral and causal consistency, iii) perceptual and temporal quality, iv) geometric and spatial consistency, v) material properties and field states, vi) kinematic and spatiotemporal persistence, vii) and mechanical plausibility with conservation laws. PVBench comprises 600 carefully crafted prompts, of which 30% involve physical embodied agents such as robot arms, humanoids, quadrupeds, and multi-agent systems. For challenging physically tractable dimensions, it pairs each with specialized vision models that derive verifiable, physically grounded measurements, such as object-level 6D pose, metric scale, surface normals, kinematics, and numerical conservation-law residuals. It thus avoids relying solely on subjective scoring, while supporting both text-to-video and image-to-video generation. Finally, we validate the agreement between automatic metrics and human judgments through human evaluation. Meanwhile, we evaluate mainstream open-source and closed-source models and present several insights. By releasing the benchmark, prompts, and calibration protocol, we aim to foster physically grounded video generation.

Date: October 19, 2026 Correspondence: 12632067@zju.edu.cn Code: https://github.com/KSZG-11/PVBench Project: https://kszg-11.github.io/PVBench-Website/

## 1 Introduction

evaluation sets usually cover only a few commonsense categories or geometric quantities in simple scenarios, yet some object-level physical quantities are crucial for downstream applications. For example, metric scale determines whether a generated video can support robot manipulation and scene reconstruction that require real dimensions, and 6D pose is the interface quantity from “watching a video” to “using a video” for motion planning and simulation initialization, which is essentially missing in existing works. Recent works have shown the trend of physical-quantity evaluation: Physics-IQ [32] indirectly characterizes motion with proxy metrics, WorldScore [10] introduces geometric measurement, but neither is yet a systematic physical-quantity evaluation framework.

We argue that physical plausibility should not stay at the level of semantic impression, but should be implemented as computable physical quantities and tested against physical laws. To this end, this paper proposes PVBench, a physical-plausibility benchmark for video generation models. Its contributions are threefold: 1) Systematic physical taxonomy: it organizes 46 evaluation dimensions into 7 domains, of which 25 physical dimensions span object-intrinsic properties, relations between objects and space, and object dynamics across space and time, covering most physical dimensions that have appeared in existing benchmarks; 2) Same-scene T2V/I2V paired prompt set: 600 prompt groups, each containing T2V/I2V versions of the same scene, 30% of which involve embodied-agent scenarios, and each prompt binds its target evaluation dimensions at construction time and requires the key physical quantities to be visible in the frame; 3) Physical-quantity-dominated evaluation protocol: specialized vision models extract numerical physical quantities, which are checked against physical laws and then mapped to evaluation scores, while dimensions that cannot be directly computed (e.g., materials, phase change, etc.) are supplemented by MLLM scoring.

Experiments show that the physical dimensions of PVBench are generally consistent with human judgments. Meanwhile, across 6 T2V and 6 I2V mainstream model configurations, the physical-layer scores are substantially lower overall than those in the semantic and perceptual layers, highlighting a persistent gap in physical plausibility. We will open-source the code and release the prompt set to foster physically grounded video generation research. We hope PVBench can push video generation from “looking good” to “physically usable”.

![](images/e94a0124f8b94626ca515aaa322b6ec6a62808de0c11c05df474743ac01dbd86.jpg)  
Figure 1 Overview of PVBench. (a) The benchmark comprises 46 evaluation dimensions organized into seven domains and three assessment layers: semantic, perceptual, and physical. (b) The physical layer contains 25 dimensions covering object-intrinsic properties, relations between objects and space, and object dynamics across space and time, encompassing most physical dimensions appearing in prior work. (c) PVBench uses three evaluation paradigms: physical-quantity measurement, MLLM evaluation, and a combination of both.

## 2 Related Work

## 2.1 Video Foundation Models

Video generation has advanced rapidly under the difusion paradigm [15, 42], with difusion transformers [36] and flow matching [24, 25] as the dominant architectural and training choices. Systems such as Sora [4], Wan [46], and HunyuanVideo 1.5 [51] now produce high-fidelity T2V and image-to-video (I2V) clips at scale, while recent models further diversify the design space. Seedance 2.0 [45] unifies text, image, audio, and video inputs in a single model, SANA-Video [7] pursues eficient long-video generation with linear attention, and MAGI-1 [48] departs from difusion altogether to explore autoregressive video generation. Beyond media synthesis, video generators are increasingly framed as world models [20]. Cosmos [33] builds world foundation models as digital twins of the physical world for physical AI, while interactive systems accept action inputs, ranging from latent-action environments [5] to open-domain action-controlled game generation [54].

## 2.2 Benchmarks for Video Generation

General-purpose suites decompose video quality into perceptual and semantic metrics computed with pretrained perception models or MLLMs as judges. VBench and VBench++ [16, 17] cover temporal consistency, motion smoothness, and text-video alignment, while EvalCrafter [26] and T2V-CompBench [44] aggregate humanpreference and compositional-alignment metrics. VBench-2.0 [56] moves toward intrinsic faithfulness by adding physics, commonsense, and human-fidelity dimensions. Its physics checks remain coarse, however, consisting mainly of VLM question answering on state changes (mechanics, thermotics, material), with feature-matching geometry for multi-view consistency. More recent world-model benchmarks broaden the evaluation scope beyond visual quality. HappyWorld-Bench [1] evaluates video, spatial, and embodied world models, with an emphasis on consistency and responsiveness during exploration, interaction, and modification.

A parallel line of work targets physical plausibility directly. VideoPhy-2 [2] and PhyWorldBench [14] rely on human or MLLM semantic judgments over action-centric activities and diverse prompt categories; PhyGenBench [28] scores 27 physical principles with a hierarchical VLM evaluator; WorldModelBench [22] rates five physical laws as binary outcomes; Physics-IQ [32] measures motion-mask overlap on real-scene continuation, with metrics the authors acknowledge remain proxies rather than direct physical quantities. WorldScore [10] is the most quantitative, reconstructing camera trajectories with DROID-SLAM [47] and computing reprojection error, dense depth, and optical-flow AEPE. However, these physical-plausibility benchmarks are still limited to semantic judgments and coarse-grained categories, and the few quantitative ones measure only camera-centric geometry. To address these challenges, we evaluate generated videos through numerical physical quantities extracted by specialized models and checked against physical laws.

## 3 Methodology

## 3.1 PVBench Evaluation System

PVBench evaluates generated videos along 46 dimensions. These 46 dimensions are organized into 7 domains and span three assessment layers: semantic, perceptual, and physical. The physical layer encompasses 4 domains with 25 dimensions in total, which are further divided into three families: object-intrinsic physical properties, relations between objects and space, and object dynamics across space and time, covering most common physical quantities. The overall framework is shown in Fig. 1; the complete correspondence is given in Appendix A, Table A1.

PVBench turns “physical plausibility” from a qualitative question into a numerically testable one. Except for dimensions that are dificult to quantify, each physical dimension is paired with a computable physical quantity and a physical-law consistency criterion. Semantic and perceptual dimensions follow or adapt existing paradigms, allowing the physical score to be interpreted jointly with semantic fidelity and visual quality.

Compared with prior works, PVBench covers physical dimensions more comprehensively, evaluates primarily through quantitative computation of physical quantities, and uses MLLM to supplement dimensions that are hard to quantify.

## 3.2 Prompt Design for Physical Plausibility Evaluation

Diverse Dedicated Scene Construction. Prompts fall into three categories: 1) for each of the 46 evaluation dimensions, 5 tailored prompts are designed; 2) related dimensions are paired in all 23 dimension pairs, each pair corresponding to 10 hybrid prompts; 3) embodied agents are injected, together with controlled occlusion, camera motion, and other visual confounders, as a high-dificulty supplement. The three categories total 600 prompt groups, each containing a one-to-one T2V and I2V variant, of which 30% involve embodied-agent scenarios (Fig. 3).

LLM-Assisted Prompt Construction. Each prompt group is built in three steps: 1) given the target dimensions and scene specification, draft the prompt with GPT-5.6 Sol [34], explicitly instantiating objects, multi-stage actions, and observability constraints; 2) each draft is manually screened for physical plausibility, visual feasibility, and unambiguous observability, modified or discarded if necessary; 3) the reference image of the I2V variant is synthesized by the GPT-image-2 model [35] from the first-frame description of the prompt, so that the T2V and I2V variants share the same initial scene (Fig. 2).

![](images/e24527cf23b6d8b7aeaffba30f4e9107c7ed7af224295274d81572c005117740.jpg)  
Figure 2 PVBench data curation. (a) Design of the target dimensions and scene. (b) Construction of prompts through multiple rounds of MLLM assistance and human verification. (c) Generation of T2V and I2V videos.

![](images/a00bd6a2bd3c710707b67605b30ab715274a3f5d494f0a936bb44b3d81961387.jpg)  
(a) Distribution of Prompt Quantities

![](images/cf840810f6eede624d83856bee5aa08b45070a12acf25f1e77b2828d4154fbe2.jpg)

(b) Evaluation Dimension Distribution  
![](images/c1ffd3bf8da9fbfd14d67f5c796edfc6b1c76a4892e7a5649aa8e877281ccb63.jpg)  
(c) Prompt Word Length

![](images/390fe7f78d2cffbb6600e95ddcae70d6b9a130aea330ca86f5aae0f0d98df624.jpg)

![](images/a2e9476da752b2d7113c1afa9f87c1d893c99e5323ce2d4cf9043ea59d99f9b6.jpg)  
(e) Prompt Word Cloud  
Figure 3 Prompt statistics of PVBench. The figure provides an integrated characterization of the PVBench prompt set, summarizing its dimension distribution, length variation, physical scene coverage, and key vocabulary.

## 3.3 Evaluation Protocol

## 3.3.1 Hierarchical Metrics of PVBench

PVBench metrics have three layers: semantic, perceptual, and physical. They are subdivided by evaluation target into 7 domains for a total of 46 dimensions. Table 1 summarizes the dimensions covered by each specialized evaluation model.

Semantic layer examines the consistency between the generated video and the conditioning information and commonsense logic, covering 2 domains and 14 dimensions. Condition and semantic fidelity (CSF) measures whether the video is faithful to the input condition, covering instruction following, attribute correctness, spatial relations, style, negation, scene, camera motion, and reference-condition following. Behavioral, causal, and commonsense consistency (BC) measures whether actions and timing unfold as instructed, and whether behavior conforms to physical, biological, and social commonsense.

The perceptual layer, termed Perceptual and Temporal Quality (PTQ), evaluates the visual quality of video frames and temporal dynamics across 7 dimensions. It covers subject consistency, aesthetic quality, motion smoothness, temporal flickering, dynamic degree, background and imaging quality. The semantic and perceptual layers follow or adapt existing evaluation paradigms, so that the physical-layer score can be interpreted jointly with semantic fidelity and image quality.

Physical layer is the core of PVBench, covering 4 domains and 25 dimensions, divided by the type of physical quantity measured. Geometric and spatial consistency (GS) evaluates the spatial self-consistency of geometric quantities such as depth, normals, shape, scale, pose, spatial relations, and cross-frame geometry. Material properties and field states (MFS) evaluates material attributes and field quantities such as deformation, material, thermal radiation, light/shadow, magnetic/electric, and phase change. Kinematics and spatiotemporal persistence (KP) evaluates kinematic quantities such as displacement, velocity, and angular velocity, as well as spatiotemporal persistence properties such as motion continuity and object permanence. Mechanical plausibility and conservation laws (MC) evaluates mechanical and conserved quantities such as static stability, dynamic efects of forces, momentum conservation, and energy conservation.

Table 1 The dimensions covered by each specialized evaluation model. Dimension names are abbreviated as codes to save space; see the Appendix for the full correspondence.
<table><tr><td>MODEL</td><td colspan="2">COVERED DIMENSIONS</td></tr><tr><td>VGGT-Ω</td><td></td><td>R01, R03, R08, R09, R11, R17, R18, R19</td></tr><tr><td>Depth Anything 3</td><td>R01, R04, R22, R23, R24</td><td></td></tr><tr><td>StableNormal</td><td>R02, R13</td><td></td></tr><tr><td>SAM 3</td><td>R23, R24</td><td>R01, R03, R04, R09, R11, R12, R15, R16, R21, R22,</td></tr><tr><td></td><td></td><td>SpatialTracker V2 R05, R12, R15, R16, R17, R20, R21, R22, R23, R24</td></tr><tr><td>Any6D</td><td>R10, R17</td><td></td></tr><tr><td>FlowSeek</td><td>R19</td><td></td></tr></table>

<table><tr><td>MODEL</td><td>COVERED DIMENSIONS</td></tr><tr><td>VFIMamba</td><td>R20</td></tr><tr><td>ReasonX</td><td>R13, R18</td></tr><tr><td>Qwen3.8-27B</td><td>R04, R06, R07, R10, R14, R25; IF-1 to IF-9, AC-1, CS-1 to CS-3</td></tr><tr><td>DINOv3</td><td>FQ-1, IF-10</td></tr><tr><td>CLIP</td><td>TQ-4, IF-10</td></tr><tr><td>FineVQ</td><td>FQ-2</td></tr><tr><td>OpenCV/Farneback TQ-1, TQ-2, TQ-3, TQ-5</td><td></td></tr></table>

## 3.3.2 From Specialized Models to Physical Quantities

Where quantification is possible, each physical dimension is paired with a specialized vision foundation model that outputs numerical physical quantities rather than semantic judgments; this is the fundamental diference between PVBench and previous physical benchmarks. But not all physical quantities can be directly quantified, in which case MLLM-assisted judgment is introduced. By evaluation method, the physical evaluation of PVBench is thus divided into three paradigms: i) specialized-model quantitative evaluation; ii) specialized-model measurement with MLLM judgment; iii) pure MLLM semantic scoring. Among the 25 physical dimensions, 17 are evaluated purely numerically, 4 rely on pure MLLM scoring, and 4 use hybrid measurement.

## 3.3.3 Score Mapping

Physical quantity values are mapped to a 1 to 5 score through thresholds, which are manually set according to the reasonable range of the physical quantity; their fit to human judgment is verified by the human validation in Section 3.4. Each dimension first takes the mean over its samples, then the domain and overall scores are obtained by equal-weight averaging across dimensions, avoiding dimensions with more samples from dominating the total. In the final report, scores are uniformly linearly mapped to a percentile scale.The specific physical quantities, formulas, specialized models used, and threshold-mapping details for each physical dimension are given in Appendix D.

## 3.4 Human Alignment For Benchmark Validation

To verify the consistency between evaluator scores and human judgments, we selected three T2V models and 5 evaluation prompts per physical dimension for manual annotation and scoring. For each evaluation

dimension, we wrote detailed guidelines and examples, and annotators were trained first to fully understand the metric definition. Each sample was independently scored by 10 annotators and averaged. The results of this experiment are presented in Section 4.3.

## 3.5 Novelty: From Semantic Judgment to Physical-State Evaluation

PVBench difers from prior benchmarks primarily in the unit of evaluation: instead of asking whether a video looks physically plausible, it estimates observable physical states and evaluates their consistency with prompt-conditioned physical constraints and laws. This shift from holistic semantic judgment to computable physical-state evaluation is the core novelty of PVBench. In contrast to prior video benchmarks, which either emphasize semantic and perceptual quality or assess physical plausibility mainly through human or MLLM judgments, binary outcomes, proxy metrics, or camera-centric geometry, PVBench makes physical plausibility measurable at the level of object states and physical laws.

## 4 Experiments

## 4.1 Experimental Setup

Evaluated models. We evaluate 6 mainstream video generation models, covering open-source and closedsource systems: MiniMax H3 [31], SANA-Video 2.0 [8], Wan2.1 [46] and Wan2.2 [46], Seedance 2.5 [6, 45], and LingBot-Video [27], for a total of 6 T2V and 6 I2V configurations. The specific list is in Table 2.

Generation protocol. Each model generates all 600 prompt groups one by one; T2V and I2V variants use text and reference-image conditions respectively. All models use a unified output specification: 720p resolution, 10-second duration, and a fixed random seed. Full details of the generation parameters are provided in Appendix F.

Evaluation protocol. The generated videos are uniformly evaluated according to the protocol in Section 3.3; all evaluators use fixed versions and weights to ensure reproducibility of the experiment. Final scores are uniformly reported on a percentile scale.

## 4.2 Main Results and Cross-Layer Model Behavior

Table 2 summarizes the current percentile scores of the 6 T2V and 6 I2V models across all seven domains. Diferent models show clear capability diferences across the seven metrics. Fig. 4 further visualizes the score distribution of each model across the seven domains and the three-layer structure with radar and bar charts.

Table 2 PVBench scores for the evaluated models. Scores are reported on a percentile scale. Within each configuration, the highest score for each metric is boldfaced and the second-highest is marked with an asterisk. Open and closed-source models are grouped separately; closed-source model rows are shaded light blue.
<table><tr><td rowspan="2">Config.</td><td rowspan="2">Model</td><td colspan="2"></td><td>Semantic Layer Perceptual Layer</td><td colspan="4">Physical Layer</td></tr><tr><td>CSF</td><td>BC</td><td>PTQ</td><td>GS</td><td>MFS</td><td>KP</td><td>MC</td></tr><tr><td rowspan="5">T2V</td><td>SANA-Video 2.0</td><td>55.1</td><td>39.0</td><td>58.2</td><td>47.0</td><td>27.4</td><td>52.2</td><td>18.7</td></tr><tr><td>Wan2.1</td><td>64.8</td><td>43.7</td><td>63.4</td><td>50.6</td><td>16.4</td><td>55.3*</td><td>24.8</td></tr><tr><td>Wan2.2</td><td>73.1</td><td>53.6</td><td>68.1*</td><td>49.2</td><td>18.1</td><td>52.9</td><td>20.5</td></tr><tr><td>LingBot-Video</td><td>72.8</td><td>59.8</td><td>58.0</td><td>42.8</td><td>28.9</td><td>51.1</td><td>22.5</td></tr><tr><td>MiniMax H3</td><td>88.0*</td><td>79.3*</td><td>55.4</td><td>52.6</td><td>30.8*</td><td>51.5</td><td>25.3*</td></tr><tr><td rowspan="5"></td><td>Seedance 2.5</td><td>92.2</td><td>83.3</td><td>68.4</td><td>52.2*</td><td>47.2</td><td>55.6</td><td>30.0</td></tr><tr><td>SANA-Video 2.0</td><td>56.0</td><td>22.9</td><td>48.4</td><td>35.8</td><td>21.3</td><td>48.0</td><td>19.5</td></tr><tr><td>Wan2.1</td><td>84.6</td><td>41.1</td><td>59.2</td><td>40.9</td><td>28.0</td><td>53.9</td><td>24.0</td></tr><tr><td>Wan2.2</td><td>88.6</td><td>55.1</td><td>66.6*</td><td>45.8</td><td>31.1</td><td>52.6</td><td>23.6</td></tr><tr><td>LingBot-Video</td><td>80.2</td><td>61.8</td><td>58.1</td><td>50.8</td><td>33.4</td><td>60.7</td><td>26.8*</td></tr><tr><td></td><td>MiniMax H3</td><td>92.8*</td><td>79.9</td><td>62.5</td><td>48.9</td><td>34.2*</td><td>50.7</td><td>20.1</td></tr><tr><td></td><td>Seedance 2.5</td><td>96.8</td><td>79.7*</td><td>67.7</td><td>50.3*</td><td></td><td>43.4 54.4*</td><td>29.0</td></tr></table>

Semantic and Perceptual Layer Results. Overall, the semantic and perceptual layers can distinguish diferent models in condition following, behavior understanding, and visual quality, but the three do not show a completely consistent ranking. For example, SANA-Video 2.0 in the T2V configuration scores 58.2 in perception and temporal quality (PTQ), but only 39.0 in behavioral and causal commonsense (BC). This illustrates that a model can produce reasonably polished frames and temporal dynamics while remaining weak in action causality and commonsense reasoning; perceptual quality and behavioral plausibility are not equivalent. The semantic and perceptual layer results provide a reference for the physical-layer analysis.

![](images/40b501e18eb91c3ba15015efd9e1c9b96b9d307adac90d75a340a813b2fef8ee.jpg)  
(a) Seven-Domain Scores of T2V Models

![](images/5abcd874d8980cfdff8f8c8a550db1cbc952b4579998967e3d64454c21877baf.jpg)

(b) Layer Scores of T2V Models (c) Layer Scores of I2V Models  
![](images/dd6945a380b5d10a64712ad0b74f6a1f52d2155165ad12b54c5727a29705b931.jpg)  
(d) Seven-Domain Scores of I2V Models  
Figure 4 PVBench results for six T2V and six I2V configurations. The figure presents seven-domain radar plots for T2V and I2V together with heatmaps of semantic-, perceptual-, and physical-layer scores and the overall score, rounded to integers. Physical-layer scores are markedly lower than semantic- and perceptual-layer scores; Seedance 2.5 performs best overall, followed by MiniMax H3.

Physical Layer Results. Overall, the scores of geometric space and kinematic persistence are higher than those of material field states and mechanical conservation, indicating that generated videos maintain visible geometric structure and general motion trajectories relatively stably, while material field states and mechanical/conservation relations are more dificult. The low-score distribution of mechanical conservation especially shows that visual continuity or action completion alone is insuficient to guarantee that the result conforms to mechanical laws.

Insight 1. Visual quality is not equivalent to physical plausibility. Although mainstream models can score highly on semantic and perceptual metrics, their physical-layer scores are generally lower. Some models that lead in visual quality and semantic fidelity rank lower physically, showing that visual impression or semantic alignment alone does not establish physical plausibility.

Insight 2. Models remain weak at mechanics and conservation laws. Unlike geometry and general motion, mechanical conservation jointly involves object trajectories, interactions, and energy or momentum relations, demanding stronger spatiotemporal modeling. Scores remain low across models, revealing deficiencies in collisions, forces, and energy changes.

## 4.3 Supplementary Analysis

Embodied Samples and High-Difficulty Supplemental Subset. We merge prompts tagged with embodied agents, occlusion, visual clutter, or camera motion into a high-dificulty supplemental subset, and compare it with the normal subset without such tags. Table 3 reports the physical-layer macro-average scores of the two subsets. Among T2V, 5 of 6 models drop on the supplemental subset, with an average relative drop of about 7.1%; among I2V, 5 of 6 models drop, with an average relative drop of about 10.9%.

Insight 3. The physical performance ofmost models drops in complex scenes. Complex scene factors increase the burden of target retention, structure recovery, and dynamic reasoning, although models difer in their sensitivity to these factors.

Human Alignment Validation. We used 3 T2V models, 25 physical dimensions, and 5 prompts per dimension for human validation. Each sample was independently scored by 10 annotators and averaged; the interannotator agreement of the average scores from the 10 annotators is high (ICC(2,k) = 0.89)[41]. Table 4 reports the correlation between automatic evaluation scores and human scores by the four physical-layer domains (three models merged). The Spearman correlation coeficients[43] of the four domains are between 0.660 and 0.805 (total 0.772), and the Kendall coeficients[19] are between 0.528 and 0.692 (total 0.643), indicating that the automatic evaluation results maintain generally high consistency with human judgments in the physical layer. Per-dimension human-alignment results are reported in Appendix B.

Table 3 Physical-layer scores on normal and high-difficulty subsets.
<table><tr><td>Config.</td><td>Model</td><td>Score</td><td>Normal High-Difficulty Score</td><td>Difference</td></tr><tr><td rowspan="5">T2V</td><td>MiniMax H3</td><td>42.6</td><td>40.2</td><td>-2.4</td></tr><tr><td>SANA-Video 2.0</td><td>38.6</td><td>37.1</td><td>-1.5</td></tr><tr><td>Wan2.1</td><td>38.6</td><td>35.4</td><td>-3.3</td></tr><tr><td>Wan2.2</td><td>37.1</td><td>35.8</td><td>-1.3</td></tr><tr><td>LingBot-Video</td><td>39.1</td><td>33.6</td><td>-5.5</td></tr><tr><td></td><td>Seedance 2.5</td><td>47.7</td><td>50.0</td><td>+2.3</td></tr><tr><td rowspan="5">I2V</td><td>MiniMax H3</td><td>40.4</td><td>41.3</td><td>+0.9</td></tr><tr><td>SANA-Video 2.0</td><td>33.3</td><td>26.6</td><td>-6.6</td></tr><tr><td>Wan2.1</td><td>37.9</td><td>34.3</td><td>-3.6</td></tr><tr><td>Wan2.2</td><td>40.9</td><td>35.1</td><td>-5.7</td></tr><tr><td>LingBot-Video</td><td>44.0</td><td>43.4</td><td>-0.6</td></tr><tr><td></td><td>Seedance 2.5</td><td>47.6</td><td>42.7</td><td>-4.8</td></tr></table>

Table 4 Correlation between automatic scores and human ratings by domain.
<table><tr><td>Metric</td><td>GS</td><td>MFS</td><td>KP</td><td>MC</td><td>Total</td></tr><tr><td>Spearman ρ</td><td>0.660</td><td>0.805</td><td>0.732</td><td>0.782</td><td>0.772</td></tr><tr><td>Kendall τ</td><td>0.528</td><td>0.692</td><td>0.619</td><td>0.651</td><td>0.643</td></tr></table>

Table 5 MiniMax H3: T2V vs. I2V.
<table><tr><td>Domain</td><td>T2V</td><td>I2V</td><td>Difference</td></tr><tr><td>CS</td><td>88.0</td><td>92.8</td><td>+4.8</td></tr><tr><td>BC</td><td>79.3</td><td>79.9</td><td>+0.6</td></tr><tr><td>PTQ</td><td>55.4</td><td>62.5</td><td>+7.1</td></tr><tr><td>GS</td><td>52.6</td><td>48.9</td><td>-3.7</td></tr><tr><td>MFS</td><td>30.8</td><td>34.2</td><td>+3.4</td></tr><tr><td>KP</td><td>51.5</td><td>50.7</td><td>-0.8</td></tr><tr><td>MC</td><td>25.3</td><td>20.1</td><td>-5.2</td></tr></table>

T2V vs. I2V: the Role of First-Frame Prior. To analyze the role of the first-frame prior, we first compare the T2V and I2V configurations of MiniMax H3. Table 5 shows that I2V scores higher than T2V in both the semantic and perceptual layers, but the physical-layer results do not improve synchronously (Fig. 5). I2V improves by 3.4 percentage points in material field states and is roughly flat in kinematic persistence, but drops by 3.7 and 5.2 percentage points in geometric space and mechanical conservation respectively. Per dimension, I2V improves more noticeably in light-shadow temporal consistency and magnetic-electric interaction, while depth, energy conservation, and object permanence scores drop. Thus, the first-frame prior does not afect diferent physical dimensions consistently. We report the paired diferences for all six models in Appendix C.1.

Insight 4. The first-frame prior mainly improves condition following and perceptual quality, not all physical quantities. The first-frame prior can constrain the appearance of the initial scene and condition following, but cannot guarantee that subsequent geometry, motion, and mechanical relations remain physically reasonable.

![](images/8f65df428c08ae590e7b8edaf5382106ab273bd2a06780f294a0e2cbaacb245a.jpg)  
Figure 5 Visual Comparison of Selected I2V Models. Qualitative comparison of representative I2V generation results across selected models, highlighting diferences in physical plausibility (e.g., geometry, material, motion, and conservation).

Table 6 Pairwise consistency across VLM backends.
<table><tr><td>Backend</td><td colspan="3">Qwen3.5 Qwen3.8 InternVL3.5</td></tr><tr><td>Qwen3.5</td><td>1.00</td><td>0.836</td><td>0.774</td></tr><tr><td>Qwen3.8</td><td>0.836</td><td>1.00</td><td>0.705</td></tr><tr><td>InternVL3.5</td><td>0.774</td><td>0.705</td><td>1.00</td></tr></table>

Table 7 Discriminability and human agreement of physical-quantity and MLLM scores.
<table><tr><td>Metric</td><td>Physical-quantity MLLM score</td><td>score</td></tr><tr><td>Global entropy</td><td>2.25</td><td>1.12</td></tr><tr><td>Inter-dimension variance</td><td>1.05</td><td>0.14</td></tr><tr><td>Full-mark ratio</td><td>15.5%</td><td>73.7%</td></tr><tr><td>Human Spearman ρ</td><td>0.84</td><td>0.14</td></tr></table>

## 4.4 Ablation Studies for PVBench

Different VLM Backends. We replace the VLM backend of the evaluator with Qwen3.5-9B [38], Qwen3.8- 27B [39], or InternVL3.5-8B [50], and compare the consistency of their scores. The pairwise Spearman correlations range from 0.71 to 0.84, indicating high agreement. This suggests that the evaluation is robust to the choice of VLM backend. We use Qwen3.8-27B by default.

Discriminability: Numerical Physical Quantities vs. MLLM Semantic Scores. For the same physical dimensions, we compare scores from specialized numerical models with direct MLLM scores. Numerical scores have higher global entropy (2.25 vs. 1.12) and inter-dimension variance (1.05 vs. 0.14). MLLM scores are often saturated, with 73.7% at the maximum. Against human ratings, numerical scores reach a Spearman correlation of 0.84, compared with 0.14 for MLLM scores. These results show that physical-quantity scoring is both more discriminative and more aligned with human judgments (Table 7).

Insight 5. Physical-quantitycomputation provides a more reliable evaluation signalthan semantic scoring. Unlike MLLM semantic scores, which tend to saturate and diverge from human judgment, numerical physical quantities ofer a more discriminative and human-aligned basis for assessing physical plausibility. This suggests that future benchmarks should prioritize computable physical measurements over language-based semantic judgments.

Threshold Calibration. We calibrate two key evaluator thresholds: SAM3 [30] detection confidence and trajectory object area. As shown in Table 8, the PVBench thresholds of 0.10 and 0.1% perform best in both coverage and Spearman correlation with human scores, whereas 0.50 and 0.50% perform worst on both measures.

Table 8 Evaluator threshold calibration. Default thresholds are boldfaced.
<table><tr><td>Evaluator hyperparameter</td><td colspan="3">Threshold Spearman ρ Coverage</td></tr><tr><td rowspan="3">SAM3 confidence</td><td>0.05</td><td>0.490</td><td>84.3%</td></tr><tr><td>0.10</td><td>0.601</td><td>84.3%</td></tr><tr><td>0.50</td><td>0.132</td><td>79.8%</td></tr><tr><td rowspan="3">Trajectory object area</td><td>0.05%</td><td>0.733</td><td>99.2%</td></tr><tr><td>0.10%</td><td>0.858</td><td>99.2%</td></tr><tr><td>0.50%</td><td>0.666</td><td>99.2%</td></tr></table>

Physical-Layer Domain Ablation. To examine how each physical-layer domain contributes to the overall physical score, we perform a leave-one-out ablation

on the four domains (GS, MFS, KP, MC) using the results in Table 2. For each model, we recompute the physical-layer average after removing one domain. Averaged over the six T2V models, the full physical-layer score is 38.5. Removing MC raises the average to 43.4, and removing MFS raises it to 41.9, whereas removing GS or KP decreases it to 35.0 and 33.6, respectively. This shows that MC and MFS are the main factors pulling down the overall physical score, while GS and KP contribute relatively higher scores. The consistently low MC scores further indicate that current models are generally weak in mechanics and conservation laws. These results support reporting the four physical domains separately, as they reveal distinct weaknesses across diferent aspects of physical plausibility.

## 5 Conclusion and Future Work

Video generation looks increasingly realistic, yet its physical plausibility remains poorly measured. Existing evaluations rely on semantic binary judgments or camera-centric geometry, and cannot reduce physical plausibility to computable, testable quantities. We propose PVBench, a physical-quantity-driven benchmark that extracts numerical measurements with specialized vision models, checks them against physical laws, uses same-scene T2V/I2V paired prompts, and resorts to MLLM scoring only for hard-to-quantify dimensions.

Our key finding is that numerical physical-quantity scores are more discriminative than MLLM semantic scores and highly consistent with human judgments, enabling more reliable and quantifiable evaluation. We hope PVBench becomes a quantifiable yardstick for physically plausible video generation.

Limitations and Future Work. The evaluation depends on specialized vision models and skips only a small number of samples for unobservable cases; material and phase-change dimensions still use MLLM scoring. We only ablate the VLM backend, and a full ablation of the specialized evaluators is left for future work. Moreover, the current evaluation largely relies on checking consistency with physical laws and, except for a few dimensions, lacks ground-truth references. More robust tracking and finer physical quantities will improve coverage. In future work, we plan to build ground-truth references for conditional image-to-video (CI2V) models using physics engines and other simulation tools to enable more direct evaluation. We will also expand the benchmark as compute and open-source models grow.

## AI Use Statement

We used generative AI tools to polish the writing and to assist in constructing the prompt set. All AI-assisted content was reviewed and verified by the authors. We take full responsibility for the final content, claims, and artifacts in this work.

## Ethics Statement

We have read and will follow the ICLR Code of Ethics. The human study was limited to validating the automatic evaluators. Annotators were trained with detailed guidelines and examples, and each sample was independently scored by 10 annotators and averaged. The annotation followed the protocol described in this paper.

## Reproducibility Statement

We describe the benchmark, prompts, evaluators, scoring rules, and experimental protocols in the main paper and appendix. During evaluation, we fix model versions, weights, and random seeds. We will release the code, prompt set, and calibration protocol to support reproduction and future comparisons. The experiments cover 12 model configurations and 600 prompt groups, about 7,200 generated videos, evaluated on a single H20 GPU.

## References

[1] Zhiqi Bai, Junai Cai, Yixin Chen, Jingrun Du, Tao Feng, Wei Gong, Siyuan Huang, Xiao Lin, Jiaheng Liu, Jun Luo, Yongzhe Lyu, Liya Ma, Zenan Meng, Lin Qu, Wenbo Su, Jiaming Wang, Qinghe Wang, Shaofei Wang, Yanghai Wang, Zequn Wang, Ziming Wang, Hu Wei, Jiangtao Wu, Ruiqi Wu, Jiaxin Xie, Yuchi Xu, Ze Xu, Chengting Yu, Liangyu Yuan, Gang Zeng, Yawen Zeng, Xingyao Zhang, Zizheng Zhang, Bo Zheng, Jiancheng Zhu, and Song-Chun Zhu. Happyworld-bench, 2026. URL https://arxiv.org/abs/2609.24308. 3

[2] Hritik Bansal, Clark Peng, Yonatan Bitton, Roman Goldenberg, Aditya Grover, and Kai-Wei Chang. VideoPhy-2: A challenging action-centric physical commonsense evaluation in video generation. arXiv preprint arXiv:2503.06800, 2025. 1, 3

[3] Gary Bradski. The opencv library. Dr. Dobb’s Journal of Software Tools, 2000. 26

[4] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. OpenAI Technical Report, 2024. 1, 2

[5] Jake Bruce, Michael Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, Yusuf Aytar, Sarah Bechtle, Feryal Behbahani, Stephanie Chan, Nicolas Heess, Lucy Gonzalez, Simon Osindero, Sherjil Ozair, Scott Reed, Jingwei Zhang, Konrad Zolna, Jef Clune, Nando de Freitas, Satinder Singh, and Tim Rocktäschel. Genie: Generative interactive environments. In Proceedings of the 41st International Conference on Machine Learning (ICML), 2024. PMLR 235:4603–4623. 1, 3

[6] ByteDance Seed. Seedance 2.5. https://seed.bytedance.com/en/seedance2\_5, 2026. Accessed: 2026-07-31. 6

[7] Junsong Chen, Yuyang Zhao, Jincheng Yu, Ruihang Chu, Junyu Chen, Shuai Yang, Xianbang Wang, Yicheng Pan, Daquan Zhou, Huan Ling, Haozhe Liu, Hongwei Yi, Hao Zhang, Muyang Li, Yukang Chen, Han Cai, Sanja Fidler, Ping Luo, Song Han, and Enze Xie. SANA-Video: Eficient video generation with block linear difusion transformer. arXiv preprint arXiv:2509.24695, 2025. 3

[8] Junsong Chen et al. SANA-Video 2.0: Hybrid linear attention with attention residuals for eficient video generation. arXiv preprint arXiv:2607.21553, 2026. 6

[9] Alara Dirik et al. ReasonX: Mllm-guided intrinsic image decomposition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026. 21

[10] Haoyi Duan, Hong-Xing Yu, Sirui Chen, Li Fei-Fei, and Jiajun Wu. WorldScore: A unified evaluation benchmark for world generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025. 2, 3

[11] Huiyu Duan et al. FineVQ: Fine-grained user generated content video quality assessment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3206–3217, 2025. 26

[12] Gunnar Farneback. Two-frame motion estimation based on polynomial expansion. In Image Analysis (SCIA), 2003. 26

[13] Martin A. Fischler and Robert C. Bolles. Random sample consensus: A paradigm for model fitting with applications to image analysis and automated cartography. Communications of the ACM, 24(6):381–395, 1981. 24

[14] Jing Gu, Xian Liu, Yu Zeng, Ashwin Nagarajan, Fangrui Zhu, Daniel Hong, Yue Fan, Qianqi Yan, Kaiwen Zhou, Ming-Yu Liu, and Xin Eric Wang. PhyWorldBench: A comprehensive evaluation of physical realism in text-to-video models. In Proceedings of the International Conference on Learning Representations (ICLR), 2026. 1, 3

[15] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising difusion probabilistic models. In Advances in Neural Information Processing Systems (NeurIPS), 2020. 2

[16] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21807–21818, 2024. 1, 3

[17] Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, Yaohui Wang, Xinyuan Chen, Ying-Cong Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench++: Comprehensive and versatile benchmark suite for video generative models. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 48(3):3268–3285, 2026. 1, 3

[18] Wolfgang Kabsch. A solution for the best rotation to relate two sets of vectors. Acta Crystallographica Section A, 32(5):922–923, 1976. 22, 23

[19] M. G. Kendall. A new measure of rank correlation. Biometrika, 30(1/2):81–93, 1938. 7

[20] Yann LeCun. A path towards autonomous machine intelligence. OpenReview preprint, 2022. 1, 3

[21] Taeyeop Lee, Bowen Wen, Minjun Kang, Gyuree Kang, In So Kweon, and Kuk-Jin Yoon. Any6D: Model-free 6D Pose Estimation of Novel Objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11633–11643, 2025. 21

[22] Dacheng Li, Yunhao Fang, Yukang Chen, Shuo Yang, Shiyi Cao, Justin Wong, Michael Luo, Xiaolong Wang, Hongxu Yin, Joseph E. Gonzalez, Ion Stoica, Song Han, and Yao Lu. WorldModelBench: Judging video generation models as world models. In Advances in Neural Information Processing Systems (NeurIPS), volume 38, pages 61064–61088, 2025. 3

[23] Haotong Lin et al. Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647, 2025. 21

[24] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In Proceedings of the International Conference on Learning Representations (ICLR), 2023. 2

[25] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In Proceedings of the International Conference on Learning Representations (ICLR), 2023. 2

[26] Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. Evalcrafter: Benchmarking and evaluating large video generation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22139–22149, 2024. 3

[27] Shuailei Ma et al. Scaling mixture-of-experts video pretraining for embodied intelligence. arXiv preprint arXiv:2607.07675, 2026. 6

[28] Fanqing Meng, Jiaqi Liao, Xinyu Tan, Wenqi Shao, Quanfeng Lu, Kaipeng Zhang, Yu Cheng, Dianqi Li, Yu Qiao, and Ping Luo. Towards world simulator: Crafting physical commonsense-based benchmark for video generation. In Proceedings of the International Conference on Machine Learning (ICML), 2025. 3

[29] Meta. DINOv3: Scalable self-supervised vision models. arXiv preprint arXiv:2508.10104, 2025. 21

[30] Meta. SAM 3: Segment anything with concepts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026. CVPR 2026. 9, 21

[31] MiniMax. MiniMax H3: An open model breaking the boundaries between modalities. https://www.minimax.io/ blog/minimax-h3, 2026. Accessed: 2026-07-31. 6

[32] Saman Motamed, Laura Culp, Kevin Swersky, Priyank Jaini, and Robert Geirhos. Do generative video models understand physical principles? In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 948–958, 2026. 1, 2, 3

[33] NVIDIA. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025. 1, 3

[34] OpenAI. GPT-5.6. https://openai.com/index/gpt-5-6/, 2026. Accessed: 2026-07-01. 4

[35] OpenAI. GPT-image-2. https://openai.com/index/introducing-chatgpt-images-2-0/, 2026. Accessed: 2026-07- 01. 4

[36] William Peebles and Saining Xie. Scalable difusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 4172–4182, 2023. 2

[37] Matteo Poggi and Fabio Tosi. Flowseek: Optical flow made easier with depth foundation models and motion bases. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025. 21

[38] Qwen Team. Qwen3.5-9B. https://huggingface.co/Qwen/Qwen3.5-9B, 2026. Accessed: 2026-07-01. 9

[39] Qwen Team. Qwen3.8-27B. https://huggingface.co/Qwen/Qwen3.8-27B, 2026. Accessed: 2026-08-15. 9, 21

[40] Alec Radford et al. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning (ICML), 2021. 21

[41] Patrick E. Shrout and Joseph L. Fleiss. Intraclass correlations: Uses in assessing rater reliability. Psychological Bulletin, 86(2):420–428, 1979. 7

[42] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic diferential equations. In Proceedings of the International Conference on Learning Representations (ICLR), 2021. 2

[43] Charles Spearman. The proof and measurement of association between two things. The American Journal of Psychology, 15(1):72–101, 1904. 7

[44] Kaiyue Sun, Kaiyi Huang, Xian Liu, Yue Wu, Zihan Xu, Zhenguo Li, and Xihui Liu. T2V-CompBench: A comprehensive benchmark for compositional text-to-video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8406–8416, 2025. 3

[45] Team Seedance et al. Seedance 2.0: Advancing video generation for world complexity. arXiv preprint arXiv:2604.14148, 2026. 3, 6

[46] Team Wan. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 2, 6

[47] Zachary Teed and Jia Deng. DROID-SLAM: Deep visual slam for monocular, stereo, and rgb-d cameras. In Advances in Neural Information Processing Systems (NeurIPS), 2021. 3

[48] Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, W. Q. Zhang, Weifeng Luo, Xiaoyang Kang, Yuchen Sun, Yue Cao, Yunpeng Huang, Yutong Lin, Yuxin Fang, Zewei Tao, Zheng Zhang, Zhongshu Wang, Zixun Liu, et al. MAGI-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211, 2025. 1, 3

[49] Jianyuan Wang et al. VGGT-Ω. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026. CVPR 2026 Oral. 21

[50] Weiyun Wang et al. InternVL3.5: Advancing open-source multimodal models in versatility, reasoning, and eficiency. arXiv preprint arXiv:2508.18265, 2025. 9

[51] Bing Wu, Chang Zou, Changlin Li, Duojun Huang, Jiangfeng Xiong, Qi Tian, et al. HunyuanVideo 1.5 technical report. arXiv preprint arXiv:2511.18870, 2025. 2

[52] Yuxi Xiao et al. SpatialTrackerV2: 3d point tracking made easy. arXiv preprint arXiv:2507.12462, 2025. 21

[53] Chongjie Ye et al. StableNormal: Reducing difusion variance for stable and sharp normal. ACM Transactions on Graphics, 43(6):1–18, 2024. 21

[54] Jiwen Yu, Yiran Qin, Xintao Wang, Pengfei Wan, Di Zhang, and Xihui Liu. GameFactory: Creating new games with generative interactive videos. In International Conference on Computer Vision (ICCV), pages 11590–11599, 2025. 3

[55] Guozhen Zhang et al. VFIMamba: Video frame interpolation with state space models. In Advances in Neural Information Processing Systems (NeurIPS), 2024. 21

[56] Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Lulu Gu, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, Yu Qiao, and Ziwei Liu. VBench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness. arXiv preprint arXiv:2503.21755, 2025. 3

## Appendix

## Appendix Overview

The supplementary material provides detailed benchmark specifications, scoring procedures, and results to support reproducibility and facilitate future comparisons:

• Appendix A describes the 46 evaluation dimensions and their organization, and compares the physicaldimension coverage of PVBench with prior benchmarks.

• Appendix B reports per-dimension human-alignment results for the physical dimensions.

• Appendix C provides the per-dimension automatic evaluation scores for all 12 model configurations.

• Appendix D details the scoring principles, including the specialized models and evaluation paradigms, physical quantities and formulas, threshold mappings, and semantic/perceptual scoring criteria.

• Appendix E gives additional statistics for the 600 prompt groups.

• Appendix F specifies the video-generation protocol and model configurations.

• Appendix G reports evaluation costs and reproducibility details.

## A Evaluation Dimensions: Description and Master Table

## A.1 Dimension Description

IF-10 (reference condition following) applies only to I2V conditioned on a reference image; T2V does not evaluate this dimension. Therefore T2V evaluates 45 dimensions in total, and I2V evaluates all 46 dimensions. Aggregation is computed by equal weight over the dimensions actually evaluated in each modality.

## A.2 Master Table of Evaluation Dimensions

Table A1 gives the ID, name, domain (D1 to D7), and assessment layer (L1 semantic layer / L2 perceptual layer / L3 physical layer) of all 46 PVBench evaluation dimensions. Per-dimension human-alignment results for the physical layer are reported in Appendix B, Table A4. Per-dimension automatic scores for all 46 dimensions across the 12 model configurations are reported in Appendix C, Tables A6 through A12. The physical-dimension coverage comparison with prior benchmarks is in this section, Table A2.

## A.3 Physical Dimension Coverage vs. Prior Work

Table A2 shows the per-dimension coverage comparison of PVBench with existing physical-plausibility benchmarks on the 25 physical dimensions. Existing benchmarks cover at most a small subset, and mainly through semantic judgment rather than physical-quantity computation; PVBench provides numerical evaluation on all three classes of physical dimensions.

Table A3 summarizes the scope and evaluation approaches of representative physical-plausibility benchmarks.   
Coverage counts for prior benchmarks are derived from the per-dimension comparison in Table A2.

## B Per-Dimension Human Alignment Results

Table A4 reports the correlation with human scores for each of the 25 physical dimensions (3 T2V models and 15 samples per dimension). The overall Spearman correlation is ρ = 0.772 and Kendall correlation is τ = 0.643, consistent with the main-text results. Some dimensions that depend on detection, tracking, or long-term consistency have lower correlations, mainly because of skipped or unobservable samples.

Table A1 Master table of all 46 PVBench evaluation dimensions.
<table><tr><td>DIM</td><td>NAME</td><td>DOMAIN</td><td>LAYER</td></tr><tr><td>IF-1</td><td>Subject adherence</td><td>D1</td><td>L1</td></tr><tr><td>IF-2</td><td>Attribute adherence</td><td>D1</td><td>L1</td></tr><tr><td>IF-4</td><td>Spatial relation adherence</td><td>D1</td><td>L1</td></tr><tr><td>IF-5</td><td>Style adherence</td><td>D1</td><td>L1</td></tr><tr><td>IF-7</td><td>Negation adherence</td><td>D1</td><td>L1</td></tr><tr><td>IF-8</td><td>Scene matching</td><td>D1</td><td>L1</td></tr><tr><td>IF-9</td><td>Camera motion adherence</td><td>D1</td><td>L1</td></tr><tr><td>IF-10</td><td>Reference condition fidelity</td><td>D1</td><td>L1</td></tr><tr><td>IF-3</td><td>Action adherence</td><td>D2</td><td>L1</td></tr><tr><td>IF-6</td><td>Temporal adherence</td><td>D2</td><td>L1</td></tr><tr><td>AC-1</td><td>Action causality</td><td>D2</td><td>L1</td></tr><tr><td>CS-1</td><td>Physical commonsense</td><td>D2</td><td>L1</td></tr><tr><td>CS-2</td><td>Biological commonsense</td><td>D2</td><td>L1</td></tr><tr><td>CS-3</td><td>Social commonsense</td><td>D2</td><td>L1</td></tr><tr><td>FQ-1</td><td>Subject consistency</td><td>D3</td><td>L2</td></tr><tr><td>FQ-2</td><td>Aesthetic quality</td><td>D3</td><td>L2</td></tr><tr><td>TQ-1</td><td>Motion smoothness</td><td>D3</td><td>L2</td></tr><tr><td>TQ-2</td><td>Temporal flickering</td><td>D3</td><td>L2</td></tr><tr><td>TQ-3</td><td>Dynamic degree</td><td>D3</td><td>L2</td></tr><tr><td>TQ-4</td><td>Background consistency</td><td>D3</td><td>L2</td></tr><tr><td>TQ-5</td><td>Imaging quality</td><td>D3</td><td>L2</td></tr><tr><td>R01</td><td>Depth</td><td>D4</td><td>L3</td></tr><tr><td>R02</td><td>Surface normal</td><td>D4</td><td>L3</td></tr><tr><td>R03</td><td>Shape</td><td>D4</td><td>L3</td></tr><tr><td>R04</td><td>Absolute scale</td><td>D4</td><td>L3</td></tr><tr><td>R08</td><td>Multi-view geometric consistency</td><td>D4</td><td>L3</td></tr><tr><td>R09</td><td>Relative scale</td><td>D4</td><td>L3</td></tr><tr><td>R10</td><td>3D pose</td><td>D4</td><td>L3</td></tr><tr><td>R11</td><td>Spatial relation</td><td>D4</td><td>L3</td></tr><tr><td>R19</td><td>Cross-frame geometric consistency</td><td>D4</td><td>L3</td></tr><tr><td>R05</td><td>Current deformation state</td><td>D5</td><td>L3</td></tr><tr><td>R06</td><td>Material properties</td><td>D5</td><td>L3</td></tr><tr><td>R07</td><td>Thermal radiation</td><td>D5</td><td>L3</td></tr><tr><td>R13</td><td>Light and shadow spatial consistency</td><td>D5</td><td>L3</td></tr><tr><td>R14</td><td>Magnetic and electric interaction</td><td>D5</td><td>L3</td></tr><tr><td>R18</td><td>Light and shadow temporal consistency</td><td>D5</td><td>L3</td></tr><tr><td>R25</td><td>Phase transition</td><td>D5</td><td>L3</td></tr><tr><td>R15</td><td>Displacement</td><td>D6</td><td>L3</td></tr><tr><td>R16</td><td>Velocity &amp; acceleration</td><td>D6</td><td>L3</td></tr><tr><td>R17</td><td>Angular velocity</td><td>D6</td><td>L3</td></tr><tr><td>R20</td><td>Motion continuity</td><td>D6</td><td>L3</td></tr><tr><td>R21</td><td>Object permanence</td><td>D6</td><td>L3</td></tr><tr><td>R12</td><td>Stability</td><td>D7</td><td>L3</td></tr><tr><td>R22</td><td>Dynamic effects of forces</td><td>D7</td><td>L3</td></tr><tr><td>R23</td><td>Momentum conservation</td><td>D7</td><td>L3</td></tr><tr><td>R24</td><td>Energy conservation</td><td>D7</td><td>L3</td></tr></table>

Table A2 Per-dimension coverage of the 25 physical dimensions across physical-plausibility benchmarks. PVBench covers all three classes of physical dimensions; existing benchmarks cover at most a small subset, and predominantly via semantic judgments rather than numerical physical quantities. Column abbreviations: VP-2 = VideoPhy-2, PGB = PhyGenBench, P-IQ = Physics-IQ, PWB = PhyWorldBench, WS = WorldScore, WMB = WorldModelBench.
<table><tr><td>DIMENSION</td><td>OURS</td><td>VP-2</td><td>PGB</td><td>P-IQ</td><td>PWB</td><td>WS</td><td>WMB</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Class I: Object-intrinsic physical properties</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>R01 Depth ordering</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>R02 Surface normals</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>R03 Shape</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>R04 Absolute scale</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>R05 Deformation state</td><td>√</td><td>√</td><td>X</td><td>√</td><td>√</td><td>X</td><td>√</td></tr><tr><td>R06 Material properties</td><td>√</td><td>X</td><td>√</td><td>√</td><td>√</td><td>X</td><td>√</td></tr><tr><td>R07 Thermal radiation</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>Class II: Object and space relations</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>R08 Multi-view geometric consistency</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>R09 Relative scale</td><td>√</td><td>X</td><td>X</td><td>X</td><td>√</td><td>X</td><td>X</td></tr><tr><td>R10 3D pose</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>R11 Spatial relations R12 Stability</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>√</td></tr><tr><td>R13 Light and shadow spatial consistency</td><td>√</td><td>X</td><td>X</td><td>X</td><td>√</td><td>X</td><td>X</td></tr><tr><td>R14 Magnetic and electric interaction</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>X</td><td>X</td></tr><tr><td>Class III: Object dynamics across space and time</td><td>√</td><td>X</td><td>X</td><td>√</td><td>√</td><td>X</td><td>X</td></tr><tr><td colspan="8"></td></tr><tr><td>R15 Displacement</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>R16 Velocity &amp; acceleration</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>R17 Angular velocity</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>R18 Light and shadow temporal consistency</td><td>√</td><td>X</td><td>X</td><td>X</td><td>√</td><td>X</td><td>X</td></tr><tr><td>R19 Cross-frame geometric consistency</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>R20 Motion continuity</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>√</td><td>X</td></tr><tr><td>R21 Object permanence</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>R22 Dynamic effects of forces</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>X</td><td>√</td></tr><tr><td>R23 Momentum conservation</td><td>√</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>R24 Energy conservation</td><td>√</td><td>X</td><td>X</td><td>X</td><td>√</td><td>X</td><td>X</td></tr><tr><td>R25 Phase transition</td><td>√</td><td>X</td><td>√</td><td>√</td><td>√</td><td>X</td><td>X</td></tr></table>

Table A3 PVBench compared with existing physical-plausibility benchmarks. Counts indicate the number of covered dimensions among the 25 physical dimensions in Table A2, not the total number of evaluation dimensions of each benchmark.
<table><tr><td>Benchmark</td><td>Physical dims.</td><td>Judge type</td><td>Numerical evaluation</td><td>Setting</td></tr><tr><td>VideoPhy-2</td><td>4</td><td>Human + automated</td><td>No</td><td>T2V</td></tr><tr><td>PhyGenBench</td><td>4</td><td>VLM</td><td>No</td><td>T2V</td></tr><tr><td>Physics-IQ</td><td>6</td><td>Motion-mask metrics</td><td>Proxy metrics</td><td>Real-video continuation</td></tr><tr><td>PhyWorldBench WorldModel</td><td>10</td><td>Human + MLLM</td><td>No</td><td>T2V</td></tr><tr><td>Bench</td><td>4</td><td>Human + VLM</td><td>No</td><td>T2V</td></tr><tr><td>WorldScore</td><td>1</td><td>Geometric SLAM</td><td>Partial (camera-centric)</td><td>T2V/I2V</td></tr><tr><td>PVBench (ours)</td><td>25</td><td>Specialized models + MLLM + physical laws</td><td>Yes</td><td>T2V/I2V</td></tr></table>

Table A4 Per-dimension correlation between automatic evaluation scores and human scores (3 T2V models merged, 15 paired samples per dimension; the auto-score mean is the 3-model merged mean on the 1 to 5 scale).
<table><tr><td>DIM</td><td>DIMENSION</td><td>n</td><td>AUTO MEAN</td><td>Spearman ρ</td><td>Kendall τ</td></tr><tr><td>R01</td><td>Depth</td><td>15</td><td>3.67</td><td>0.721</td><td>0.613</td></tr><tr><td>R02</td><td>Surface normals</td><td>15</td><td>2.87</td><td>0.706</td><td>0.610</td></tr><tr><td>R03</td><td>Shape</td><td>15</td><td>4.07</td><td>0.540</td><td>0.457</td></tr><tr><td>R04</td><td>Absolute scale</td><td>15</td><td>1.93</td><td>0.638</td><td>0.532</td></tr><tr><td>R05</td><td>Deformation state</td><td>15</td><td>2.53</td><td>0.622</td><td>0.563</td></tr><tr><td>R06</td><td>Material properties</td><td>15</td><td>3.13</td><td>0.867</td><td>0.734</td></tr><tr><td>R07</td><td>Thermal radiation</td><td>15</td><td>2.27</td><td>0.839</td><td>0.729</td></tr><tr><td>R08</td><td>Multi-view geometric consistency</td><td>15</td><td>2.87</td><td>0.814</td><td>0.700</td></tr><tr><td>R09</td><td>Relative scale</td><td>15</td><td>2.67</td><td>0.401</td><td>0.320</td></tr><tr><td>R10</td><td>3D pose</td><td>15</td><td>3.73</td><td>0.760</td><td>0.671</td></tr><tr><td>R11</td><td>Spatial relations</td><td>15</td><td>1.93</td><td>0.693</td><td>0.619</td></tr><tr><td>R12</td><td>Stability</td><td>15</td><td>1.73</td><td>0.777</td><td>0.679</td></tr><tr><td>R13</td><td>Light and shadow spatial consistency</td><td>15</td><td>1.73</td><td>0.269</td><td>0.246</td></tr><tr><td>R14</td><td>Magnetic and electric interaction</td><td>15</td><td>2.07</td><td>0.778</td><td>0.681</td></tr><tr><td>R15</td><td>Displacement</td><td>15</td><td>4.47</td><td>0.765</td><td>0.676</td></tr><tr><td>R16</td><td>Velocity &amp; acceleration</td><td>15</td><td>3.07</td><td>0.624</td><td>0.558</td></tr><tr><td>R17</td><td>Angular velocity</td><td>15</td><td>1.27</td><td>0.440</td><td>0.388</td></tr><tr><td>R18</td><td>Light and shadow temporal consistency</td><td>15</td><td>2.67</td><td>0.803</td><td>0.702</td></tr><tr><td>R19</td><td>Cross-frame geometric consistency</td><td>15</td><td>3.40</td><td>0.687</td><td>0.599</td></tr><tr><td>R20</td><td>Motion continuity</td><td>15</td><td>3.80</td><td>0.240</td><td>0.214</td></tr><tr><td>R21</td><td>Object permanence</td><td>15</td><td>3.60</td><td>0.125</td><td>0.108</td></tr><tr><td>R22</td><td>Dynamic effects of forces</td><td>15</td><td>2.67</td><td>0.591</td><td>0.507</td></tr><tr><td>R23</td><td>Momentum conservation</td><td>15</td><td>2.13</td><td>0.640</td><td>0.567</td></tr><tr><td>R24</td><td>Energy conservation</td><td>15</td><td>1.73</td><td>0.825</td><td>0.731</td></tr><tr><td>R25</td><td>Phase transition</td><td>15</td><td>1.93</td><td>0.784</td><td>0.700</td></tr><tr><td>Total</td><td>25 phys. dims</td><td>375</td><td>N/A</td><td>0.772</td><td>0.643</td></tr></table>

## C Per-Dimension Automatic Score Tables

Table A5 first reports the skipped dimension instances for each configuration. Tables A6 through A12 then give the automatic evaluation mean score (1 to 5) of all 12 model configurations on all 46 PVBench dimensions, by the seven domains D1 to D7, obtained by averaging per dimension from the global report (semantic/perceptual layers, D1 to D3) and the unified physical report (physical layer, D4 to D7) in the result directory of each configuration; samples with status skipped or error (e.g., IF-10 not applicable to T2V reference images) are excluded from the mean and recorded as not applicable. Physical-layer dimensions are generally lower than semantic/perceptual layers, consistent with the insight in main text Section 4.2.

Table A5 Skipped dimension instances for each model configuration. “Total Dimension Evaluations” denotes the sum of evaluation instances over all evaluated dimensions for each configuration. Normally this is 951. For Seedance 2.5 I2V, 69 dimension-level evaluation instances are missing because the corresponding videos failed to generate due to face-related content moderation, yielding 882 = 951 - 69. “Skipped” only counts instances skipped by the evaluator and does not include generation failures.

<table><tr><td>Config. Model</td><td></td><td>Skipped</td><td>Total Instances</td><td>Ratio</td></tr><tr><td>I2V</td><td>LingBot-Video</td><td>46</td><td>951</td><td>4.84%</td></tr><tr><td>T2V</td><td>LingBot-Video</td><td>26</td><td>951</td><td>2.73%</td></tr><tr><td>I2V</td><td>MiniMax H3</td><td>10</td><td>951</td><td>1.05%</td></tr><tr><td>T2V</td><td>MiniMax H3</td><td>16</td><td>951</td><td>1.68%</td></tr><tr><td>I2V</td><td>SANA-Video 2.0</td><td>24</td><td>951</td><td>2.52%</td></tr><tr><td>T2V</td><td>SANA-Video 2.0</td><td>19</td><td>951</td><td>2.00%</td></tr><tr><td>I2V</td><td>Seedance 2.5</td><td>15</td><td>882</td><td>1.70%</td></tr><tr><td>T2V</td><td>Seedance 2.5</td><td>18</td><td>951</td><td>1.89%</td></tr><tr><td>I2V</td><td>Wan2.1</td><td>15</td><td>951</td><td>1.58%</td></tr><tr><td>T2V</td><td>Wan2.1</td><td>21</td><td>951</td><td>2.21%</td></tr><tr><td>I2V</td><td>Wan2.2</td><td>16</td><td>951</td><td>1.68%</td></tr><tr><td>T2V</td><td>Wan2.2</td><td>17</td><td>951</td><td>1.79%</td></tr></table>

Table A6 Automatic evaluation mean score (1 to 5) of all 12 configurations on D1 (instruction following and semantic alignment, 8 dims).
<table><tr><td>Config</td><td>IF-1</td><td>IF-2</td><td>IF-4</td><td>IF-5</td><td>IF-7</td><td>IF-8</td><td>IF-9</td><td>IF-10</td></tr><tr><td>MiniMax H3 T2V</td><td>4.89</td><td>4.80</td><td>4.81</td><td>4.65</td><td>3.29</td><td>5.00</td><td>4.19</td><td></td></tr><tr><td>SANA-Video 2.0 T2V</td><td>2.74</td><td>3.25</td><td>2.76</td><td>3.91</td><td>1.71</td><td>5.00</td><td>3.05</td><td></td></tr><tr><td>Wan2.1-T2V-14B T2V</td><td>3.63</td><td>3.75</td><td>4.05</td><td>4.09</td><td>2.38</td><td>4.95</td><td>2.29</td><td></td></tr><tr><td>Wan2.2-T2V-A14B T2V</td><td>3.89</td><td>4.80</td><td>4.43</td><td>3.91</td><td>2.33</td><td>5.00</td><td>3.10</td><td></td></tr><tr><td>Seedance 2.5 T2V</td><td>4.42</td><td>4.75</td><td>4.71</td><td>4.83</td><td>4.86</td><td>5.00</td><td>4.24</td><td></td></tr><tr><td>LingBot-Video T2V</td><td>4.00</td><td>3.30</td><td>3.76</td><td>3.22</td><td>4.86</td><td>5.00</td><td>3.24</td><td></td></tr><tr><td>MiniMax H3 I2V</td><td>4.68</td><td>4.85</td><td>4.76</td><td>5.00</td><td>5.00</td><td>5.00</td><td>4.19</td><td>4.20</td></tr><tr><td>SANA-Video 2.0 I2V</td><td>2.53</td><td>3.05</td><td>2.00</td><td>4.30</td><td>2.86</td><td>4.86</td><td>2.86</td><td>3.48</td></tr><tr><td>Wan2.1-I2V-14B-720P I2V</td><td>4.74</td><td>4.85</td><td>3.90</td><td>5.00</td><td>4.81</td><td>5.00</td><td>2.19</td><td>4.58</td></tr><tr><td>Wan2.2-I2V-A14B I2V</td><td>4.79</td><td>4.95</td><td>4.48</td><td>5.00</td><td>4.81</td><td>5.00</td><td>2.71</td><td>4.60</td></tr><tr><td>Seedance 2.5 I2V</td><td>4.95</td><td>5.00</td><td>4.80</td><td>5.00</td><td>5.00</td><td>5.00</td><td>4.56</td><td>4.67</td></tr><tr><td>LingBot-Video I2V</td><td>4.21</td><td>3.67</td><td>4.70</td><td>3.71</td><td>4.83</td><td>4.52</td><td>5.00</td><td>3.00</td></tr></table>

Table A7 Automatic evaluation mean score (1 to 5) of all 12 configurations on D2 (action and commonsense reasoning, 6 dims).
<table><tr><td>Config</td><td>IF-3</td><td>IF-6</td><td>AC-1</td><td>CS-1</td><td>CS-2</td><td>CS-3</td></tr><tr><td>MiniMax H3 T2V</td><td>3.65</td><td>4.29</td><td>3.18</td><td>4.85</td><td>4.78</td><td>4.29</td></tr><tr><td>SANA-Video 2.0 T2V</td><td>1.60</td><td>1.76</td><td>1.94</td><td>3.70</td><td>3.26</td><td>3.10</td></tr><tr><td>Wan2.1-T2V-14B T2V</td><td>1.75</td><td>1.67</td><td>1.88</td><td>3.85</td><td>3.91</td><td>3.43</td></tr><tr><td>Wan2.2-T2V-A14B T2V</td><td>2.15</td><td>2.33</td><td>2.24</td><td>4.05</td><td>4.39</td><td>3.71</td></tr><tr><td>Seedance 2.5 T2V</td><td>4.00</td><td>4.29</td><td>3.18</td><td>4.95</td><td>4.87</td><td>4.71</td></tr><tr><td>LingBot-Video T2V</td><td>3.35</td><td>2.19</td><td>2.47</td><td>3.80</td><td>4.35</td><td>4.19</td></tr><tr><td>MiniMax H3 I2V</td><td>3.95</td><td>4.57</td><td>3.29</td><td>4.60</td><td>4.91</td><td>3.86</td></tr><tr><td>SANA-Video 2.0 I2V</td><td>1.45</td><td>1.86</td><td>1.47</td><td>2.80</td><td>1.87</td><td>2.05</td></tr><tr><td>Wan2.1-I2V-14B-720P I2V</td><td>1.70</td><td>2.10</td><td>2.18</td><td>3.75</td><td>3.39</td><td>2.76</td></tr><tr><td>Wan2.2-I2V-A14B I2V</td><td>2.25</td><td>2.48</td><td>2.12</td><td>4.35</td><td>4.61</td><td>3.43</td></tr><tr><td>Seedance 2.5 I2V</td><td>4.25</td><td>4.10</td><td>3.00</td><td>4.78</td><td>4.50</td><td>4.50</td></tr><tr><td>LingBot-Video I2V</td><td>3.45</td><td>3.38</td><td>2.71</td><td>4.05</td><td>3.87</td><td>3.38</td></tr></table>

## C.1 Paired T2V/I2V Differences Across Models

Table A13 reports the paired diferences (I2V minus T2V) on all seven domains for the six model configurations.

Table A8 Automatic evaluation mean score (1 to 5) of all 12 configurations on D3 (perception and quality, 7 dims).
<table><tr><td>Config</td><td>FQ-1</td><td>FQ-2</td><td>TQ-1</td><td>TQ-2</td><td>TQ-3</td><td>TQ-4</td><td>TQ-5</td></tr><tr><td>MiniMax H3 T2V</td><td>3.76</td><td>3.36</td><td>1.18</td><td>3.74</td><td>4.33</td><td>3.00</td><td>3.13</td></tr><tr><td>SANA-Video 2.0 T2V</td><td>3.05</td><td>5.00</td><td>1.05</td><td>3.00</td><td>4.19</td><td>2.42</td><td>4.61</td></tr><tr><td>Wan2.1-T2V-14B T2V</td><td>2.81</td><td>3.86</td><td>2.64</td><td>4.70</td><td>4.29</td><td>3.38</td><td>3.09</td></tr><tr><td>Wan2.2-T2V-A14B T2V</td><td>3.81</td><td>4.45</td><td>2.27</td><td>4.43</td><td>4.33</td><td>3.38</td><td>3.39</td></tr><tr><td>Seedance 2.5 T2V</td><td>3.67</td><td>4.27</td><td>2.23</td><td>4.74</td><td>4.57</td><td>3.79</td><td>2.87</td></tr><tr><td>LingBot-Video T2V</td><td>2.19</td><td>3.00</td><td>2.41</td><td>4.57</td><td>4.67</td><td>3.50</td><td>2.91</td></tr><tr><td>MiniMax H3 I2V</td><td>3.00</td><td>3.91</td><td>2.05</td><td>4.09</td><td>4.52</td><td>3.75</td><td>3.17</td></tr><tr><td>SANA-Video 2.0 I2V</td><td>1.76</td><td>4.73</td><td>1.05</td><td>2.83</td><td>4.00</td><td>1.71</td><td>4.48</td></tr><tr><td>Wan2.1-I2V-14B-720P I2V</td><td>3.05</td><td>3.91</td><td>1.95</td><td>4.09</td><td>4.29</td><td>3.00</td><td>3.30</td></tr><tr><td>Wan2.2-I2V-A14B I2V</td><td>3.57</td><td>4.50</td><td>1.82</td><td>4.22</td><td>4.57</td><td>3.58</td><td>3.39</td></tr><tr><td>Seedance 2.5 I2V</td><td>3.80</td><td>4.05</td><td>2.15</td><td>4.75</td><td>4.31</td><td>3.86</td><td>3.04</td></tr><tr><td>LingBot-Video I2V</td><td>2.86</td><td>3.59</td><td>1.59</td><td>4.48</td><td>4.62</td><td>3.12</td><td>3.00</td></tr></table>

Table A9 Automatic evaluation mean score (1 to 5) of all 12 configurations on D4 (geometry and space, 9 dims).
<table><tr><td>Config</td><td>R01</td><td>R02</td><td>R03</td><td>R04</td><td>R08</td><td>R09</td><td>R10</td><td>R11</td><td>R19</td></tr><tr><td>MiniMax H3 T2V</td><td>4.43</td><td>2.24</td><td>4.36</td><td>2.09</td><td>3.39</td><td>2.00</td><td>4.35</td><td>1.39</td><td>3.70</td></tr><tr><td>SANA-Video 2.0 T2V</td><td>3.59</td><td>2.38</td><td>3.18</td><td>1.38</td><td>3.65</td><td>3.27</td><td>3.40</td><td>1.33</td><td>3.74</td></tr><tr><td>Wan2.1-T2V-14B T2V</td><td>3.40</td><td>3.52</td><td>4.00</td><td>2.05</td><td>2.00</td><td>4.33</td><td>2.95</td><td>1.79</td><td>3.17</td></tr><tr><td>Wan2.2-T2V-A14B T2V</td><td>3.48</td><td>2.81</td><td>3.45</td><td>2.30</td><td>2.57</td><td>3.42</td><td>3.65</td><td>1.39</td><td>3.65</td></tr><tr><td>Seedance 2.5 T2V</td><td>3.67</td><td>2.76</td><td>4.36</td><td>2.41</td><td>3.26</td><td>2.12</td><td>4.25</td><td>1.67</td><td>3.30</td></tr><tr><td>LingBot-Video T2V</td><td>4.50</td><td>2.57</td><td>3.09</td><td>1.58</td><td>2.96</td><td>1.30</td><td>3.95</td><td>1.28</td><td>3.17</td></tr><tr><td>MiniMax H3 I2V</td><td>3.55</td><td>1.86</td><td>4.45</td><td>2.33</td><td>3.04</td><td>2.50</td><td>4.05</td><td>1.33</td><td>3.48</td></tr><tr><td>SANA-Video 2.0 I2V</td><td>3.00</td><td>2.33</td><td>2.09</td><td>2.14</td><td>3.17</td><td>2.40</td><td>2.10</td><td>1.22</td><td>3.43</td></tr><tr><td>Wan2.1-I2V-14B-720P I2V</td><td>3.48</td><td>1.95</td><td>3.27</td><td>2.39</td><td>2.70</td><td>2.31</td><td>2.60</td><td>1.78</td><td>3.26</td></tr><tr><td>Wan2.2-I2V-A14B I2V</td><td>3.36</td><td>1.95</td><td>4.00</td><td>2.33</td><td>2.74</td><td>2.50</td><td>3.65</td><td>1.56</td><td>3.39</td></tr><tr><td>Seedance 2.5 I2V</td><td>3.73</td><td>1.95</td><td>4.18</td><td>2.08</td><td>3.52</td><td>2.54</td><td>4.25</td><td>1.39</td><td>3.45</td></tr><tr><td>LingBot-Video I2V</td><td>5.00</td><td>2.24</td><td>3.27</td><td>3.00</td><td>3.00</td><td>2.25</td><td>3.40</td><td>1.65</td><td>3.48</td></tr></table>

Table A10 Automatic evaluation mean score (1 to 5) of all 12 configurations on D5 (material properties and field states, 7 dims).
<table><tr><td>Config</td><td>R05</td><td>R06</td><td>R07</td><td>R13</td><td>R14</td><td>R18</td><td>R25</td></tr><tr><td>MiniMax H3 T2V</td><td>2.56</td><td>3.62</td><td>2.27</td><td>1.48</td><td>1.76</td><td>2.41</td><td>1.52</td></tr><tr><td>SANA-Video 2.0 T2V</td><td>3.56</td><td>3.10</td><td>1.91</td><td>1.14</td><td>1.48</td><td>2.50</td><td>1.00</td></tr><tr><td>Wan2.1-T2V-14B T2V</td><td>1.83</td><td>1.95</td><td>1.32</td><td>1.57</td><td>1.10</td><td>2.82</td><td>1.00</td></tr><tr><td>Wan2.2-T2V-A14B T2V</td><td>1.33</td><td>1.90</td><td>1.59</td><td>1.86</td><td>1.29</td><td>2.91</td><td>1.17</td></tr><tr><td>Seedance 2.5 T2V</td><td>3.06</td><td>3.90</td><td>2.82</td><td>1.86</td><td>3.29</td><td>2.82</td><td>2.48</td></tr><tr><td>LingBot-Video T2V</td><td>2.56</td><td>2.71</td><td>1.68</td><td>1.86</td><td>1.86</td><td>2.73</td><td>1.70</td></tr><tr><td>MiniMax H3 I2V</td><td>2.44</td><td>3.76</td><td>2.32</td><td>1.67</td><td>2.33</td><td>2.86</td><td>1.17</td></tr><tr><td>SANA-Video 2.0 I2V</td><td>3.44</td><td>1.71</td><td>1.91</td><td>1.14</td><td>1.38</td><td>2.36</td><td>1.00</td></tr><tr><td>Wan2.1-I2V-14B-720P I2V</td><td>2.72</td><td>3.10</td><td>1.95</td><td>1.67</td><td>1.67</td><td>2.64</td><td>1.09</td></tr><tr><td>Wan2.2-I2V-A14B I2V</td><td>2.17</td><td>2.90</td><td>1.95</td><td>1.76</td><td>2.71</td><td>2.68</td><td>1.52</td></tr><tr><td>Seedance 2.5 I2V</td><td>3.00</td><td>3.81</td><td>2.77</td><td>1.76</td><td>2.90</td><td>2.68</td><td>2.24</td></tr><tr><td>LingBot-Video I2V</td><td>2.94</td><td>3.05</td><td>2.27</td><td>1.76</td><td>2.05</td><td>2.68</td><td>1.61</td></tr></table>

Table A11 Automatic evaluation mean score (1 to 5) of all 12 configurations on D6 (kinematics, 5 dims).
<table><tr><td>Config</td><td>R15</td><td>R16</td><td>R17</td><td>R20</td><td>R21</td></tr><tr><td>MiniMax H3 T2V</td><td>4.22</td><td>3.10</td><td>1.19</td><td>3.45</td><td>3.33</td></tr><tr><td>SANA-Video 2.0 T2V</td><td>4.28</td><td>3.10</td><td>1.00</td><td>3.75</td><td>3.31</td></tr><tr><td>Wan2.1-T2V-14B T2V</td><td>4.33</td><td>3.05</td><td>1.05</td><td>4.00</td><td>3.62</td></tr><tr><td>Wan2.2-T2V-A14B T2V</td><td>4.44</td><td>2.95</td><td>1.05</td><td>3.95</td><td>3.18</td></tr><tr><td>Seedance 2.5 T2V</td><td>4.61</td><td>3.00</td><td>1.00</td><td>4.00</td><td>3.50</td></tr><tr><td>LingBot-Video T2V</td><td>4.22</td><td>2.95</td><td>1.00</td><td>3.85</td><td>3.20</td></tr><tr><td>MiniMax H3 I2V</td><td>4.33</td><td>2.90</td><td>1.00</td><td>3.85</td><td>3.06</td></tr><tr><td>SANA-Video 2.0 I2V</td><td>3.50</td><td>3.00</td><td>1.00</td><td>3.80</td><td>3.29</td></tr><tr><td>Wan2.1-I2V-14B-720P I2V</td><td>4.44</td><td>3.00</td><td>1.19</td><td>3.95</td><td>3.20</td></tr><tr><td>Wan2.2-I2V-A14B I2V</td><td>4.22</td><td>3.00</td><td>1.05</td><td>3.90</td><td>3.36</td></tr><tr><td>Seedance 2.5 I2V</td><td>4.61</td><td>3.00</td><td>1.05</td><td>4.00</td><td>3.23</td></tr><tr><td>LingBot-Video I2V</td><td>4.17</td><td>3.10</td><td>1.19</td><td>4.00</td><td>4.69</td></tr></table>

Table A12 Automatic evaluation mean score (1 to 5) of all 12 configurations on D7 (mechanics and conservation, 4 dims).
<table><tr><td>Config</td><td>R12</td><td>R22</td><td>R23</td><td>R24</td></tr><tr><td>MiniMax H3 T2V</td><td>1.50</td><td>2.60</td><td>1.94</td><td>2.00</td></tr><tr><td>SANA-Video 2.0 T2V</td><td>1.36</td><td>2.80</td><td>1.83</td><td>1.00</td></tr><tr><td>Wan2.1-T2V-14B T2V</td><td>1.68</td><td>2.70</td><td>1.82</td><td>1.76</td></tr><tr><td>Wan2.2-T2V-A14B T2V</td><td>1.50</td><td>2.60</td><td>1.89</td><td>1.29</td></tr><tr><td>Seedance 2.5 T2V</td><td>1.36</td><td>2.70</td><td>2.83</td><td>1.90</td></tr><tr><td>LingBot-Video T2V</td><td>1.41</td><td>2.80</td><td>1.78</td><td>1.62</td></tr><tr><td>MiniMax H3 I2V</td><td>1.27</td><td>2.60</td><td>1.72</td><td>1.62</td></tr><tr><td>SANA-Video 2.0 I2V</td><td>1.10</td><td>2.65</td><td>1.94</td><td>1.43</td></tr><tr><td>Wan2.1-I2V-14B-720P I2V</td><td>1.48</td><td>2.75</td><td>1.94</td><td>1.67</td></tr><tr><td>Wan2.2-I2V-A14B I2V</td><td>1.59</td><td>2.70</td><td>1.82</td><td>1.67</td></tr><tr><td>Seedance 2.5 I2V</td><td>1.64</td><td>2.85</td><td>2.44</td><td>1.71</td></tr><tr><td>LingBot-Video I2V</td><td>1.23</td><td>2.75</td><td>1.89</td><td>2.43</td></tr></table>

Table A13 Paired diferences (I2V minus T2V) on all seven domains across six model configurations.
<table><tr><td>Model</td><td>CSF</td><td>BC</td><td>PTQ</td><td>GS</td><td>MFS</td><td>KP</td><td>MC</td></tr><tr><td>SANA-Video 2.0</td><td>+0.9</td><td>-16.1</td><td>-9.8</td><td>-11.2</td><td>-6.1</td><td>-4.2</td><td>+0.8</td></tr><tr><td>Wan2.1</td><td>+19.8</td><td>-2.6</td><td>-4.2</td><td>-9.7</td><td>+11.6</td><td>-1.4</td><td>-0.8</td></tr><tr><td>Wan2.2</td><td>+15.5</td><td>+1.5</td><td>-1.5</td><td>-3.4</td><td>+13.0</td><td>-0.3</td><td>+3.1</td></tr><tr><td>LingBot-Video</td><td>+7.4</td><td>+2.0</td><td>+0.1</td><td>+8.0</td><td>+4.5</td><td>+9.6</td><td>+4.3</td></tr><tr><td>MiniMax H3</td><td>+4.8</td><td>+0.6</td><td>+7.1</td><td>-3.7</td><td>+3.4</td><td>-0.8</td><td>-5.2</td></tr><tr><td>Seedance 2.5</td><td>+4.6</td><td>-3.6</td><td>-0.7</td><td>-1.9</td><td>-3.8</td><td>-1.2</td><td>-1.0</td></tr></table>

## D Per-Dimension Scoring Principles

Each physical dimension is paired with a specialized vision foundation model to output numerical physical quantities, divided into three paradigms by evaluation method: i) specialized-model quantitative evaluation; ii) specialized-model measurement with MLLM judgment; iii) pure MLLM semantic scoring. Table A14 gives the dimensions covered by each specialized model and the paradigm; the physical quantity, formula, and criterion extracted for each of the 25 physical dimensions are explained one by one in Section D.2; the mapping from physical quantities to 1 to 5 threshold segments is in Section D.3 Table A15. The D1 to D3 semantic/perceptual-layer dimensions are directly scored by MLLM or specialized perception models with 1 to 5 semantic anchors; the principle is in Section D.4.

## D.1 Specialized Models and Dimension/Paradigm Mapping

Table A14 lists each evaluation model, the dimensions it covers, and the paradigm used. Geometric and kinematic dimensions are mainly output as numerical physical quantities by specialized models such as VGGT-Ω [49], StableNormal [53], Depth Anything 3 [23], SAM 3 [30], SpatialTracker V2 [52], Any6D [21], FlowSeek [37], VFIMamba [55], DINOv3 [29], and CLIP [40] (paradigm i); dimensions such as light/shadow and object permanence are measured by ReasonX [9] / SAM 3 and then judged by MLLM (paradigm ii); material, thermal radiation, magnetic/electric, phase change, and all instruction-following/commonsense dimensions are directly semantically scored by Qwen3.8-27B [39] (paradigm iii).

## D.2 Physical Quantities, Formulas, and Criteria

Below, the physical quantity, formula, and criterion extracted for each of the 25 physical dimensions are explained one by one by D4 to D7; the 1 to 5 threshold segments for each physical quantity are in Section D.3 Table A15.

## D4 Geometric and Spatial Consistency (L3 Physical Layer).

R01 Depth. We estimate the per-frame depth map using Depth Anything 3. Using the object occlusion regions segmented by SAM 3, we verify the depth-order accuracy, i.e., the depth of the occluded object should be greater than the depth of the occluder; the higher the depth-order accuracy, the higher the score.

Table A14 The dimensions covered by each specialized evaluation model and the evaluation paradigm. Paradigms: i) specialized-model quantitative evaluation; ii) specialized-model measurement + MLLM judgment; iii) pure MLLM semantic scoring.
<table><tr><td>MODEL</td><td>COVERED DIMS</td><td>PARADIGM</td></tr><tr><td>VGGT-Ω</td><td>R01, R03, R08, R09, R11, R17, R18, R19</td><td>(i)</td></tr><tr><td>Depth Anything 3</td><td>R01, R04, R22, R23, R24</td><td>(i)</td></tr><tr><td>StableNormal</td><td>R02, R13</td><td>(i)/(ii)</td></tr><tr><td>SAM 3</td><td>R01, R03, R04, R09, R11, R12, R15, R16, R21, R22, R23, R24</td><td>(i)/(ii)</td></tr><tr><td>SpatialTracker V2</td><td>R05, R12, R15, R16, R17, R20, R21, R22, R23, R24</td><td>(i)</td></tr><tr><td>Any6D</td><td>R10, R17</td><td>(i)</td></tr><tr><td>FlowSeek</td><td>R19</td><td>(i)</td></tr><tr><td>VFIMamba</td><td>R20</td><td>(i)</td></tr><tr><td>ReasonX</td><td>R13, R18</td><td>(ii)</td></tr><tr><td>Qwen3.8-27B</td><td>R04, R06, R07, R10, R14, R25; IF-1 to IF-9, AC-1, CS-1 to CS-3</td><td>(ii)/(iii)</td></tr><tr><td>DINOv3</td><td>FQ-1, IF-10</td><td>(i)</td></tr><tr><td>CLIP</td><td>TQ-4, IF-10</td><td>(i)</td></tr><tr><td>FineVQ</td><td>FQ-2</td><td>(i)</td></tr><tr><td>OpenCV/Farneback</td><td>TQ-1, TQ-2, TQ-3, TQ-5</td><td>(i)</td></tr></table>

R02 Surface Normal. Estimated per-frame surface normal map by StableNormal, cross-validated with the geometric consistency provided by VGGT-Ω to check the consistency of normals with the depth gradient; the higher the consistency, the higher the score.

R03 Shape. Per-frame 3D point clouds reconstructed by VGGT-Ω; after Kabsch alignment[18] of cross-frame point clouds, the normalized Chamfer distance measures shape cross-frame self-consistency; non-standard geometry is semantically judged by Qwen3.8-27B. Higher self-consistency means more consistent shape.

R04 Absolute Scale. Metric scale prior provided by Depth Anything 3, combined with the object size segmented by SAM 3, Qwen3.8-27B judges whether the object size falls in a reasonable range; the more reasonable, the higher the score.

R08 Multi-View Geometric Consistency. For multiple views of the same scene (multiple frames produced by camera motion), VGGT-Ω reconstructs each separately, and the reprojection error and geometric selfconsistency residual are computed to measure whether the 3D structure is consistent across views. Smaller residual is better.

R09 Relative Scale. VGGT-Ω estimates the cross-frame bounding-box size of the object, and the coeficient of variation CV = σ/µ of the size sequence is computed, with SAM 3 segmentation ensuring the same object corresponds across frames. Smaller CV means more stable relative scale.

R10 3D Pose. Any6D estimates the six-degree-of-freedom pose of the object; combined with VGGT-Ω geometry, Qwen3.8-27B compares whether the final pose change matches the known pose change specified by the prompt; the higher the match, the higher the score.

R11 Spatial Relation. VGGT-Ω back-projects to obtain the 3D coordinates of object centroids, computes the relative position and distance between objects, and compares consistency with the spatial relation described by the prompt (above/below/left/right/front/back/containment, etc.). Consistent gives a high score.

R19 Cross-Frame Geometric Consistency. On the one hand, VGGT-Ω independently reconstructs point clouds for adjacent frames and computes the cross-frame Chamfer drift (relaxed for the VGGT-Ω reconstruction noise baseline); on the other hand, FlowSeek computes the coeficient of variation CV of the average endpoint error AEPE of per-frame optical flow. Both smaller indicate more stable cross-frame geometry.

D5 Material Properties and Field States (L3 Physical Layer).

R05 Current Deformation State. Per-frame point clouds reconstructed by VGGT-Ω, combined with the object trajectory of SpatialTracker V2, after Kabsch alignment[18] compute the normalized Chamfer distance or rigid-body distance change rate to measure deformation; the threshold is higher than the VGGT-Ω reconstruction noise baseline (about 0.08 to 0.15) to avoid mistaking reconstruction noise for deformation. A smaller change rate means closer to rigid body.

R06 Material Properties. Qwen3.8-27B performs a five-dimensional material check on the video (material category, hardness/softness, translucency, reflectivity, surface texture), judging item by item whether it conforms to physical commonsense. All dimensions reasonable gives a high score.

R07 Thermal Radiation. Qwen3.8-27B performs a five-dimensional radiation check (heat source identification, heat conduction direction, heat-induced luminescence/color change, thermal convection/updraft, no abnormal radiation from low-temperature objects), judging item by item whether it conforms to thermodynamic laws. Conforming gives a high score.

R13 Light and Shadow Spatial Consistency. ReasonX estimates the back-inferred light source direction of each object, combined with the normal of StableNormal, computes the variance of the back-inferred light source directions of each object in the same scene and the photometric/shadow-position residual (degrees); the smaller the variance and residual, the more self-consistent the light/shadow is spatially.

R14 Magnetic and Electric Interaction. Qwen3.8-27B judges the correctness of electromagnetic/electrostatic phenomena in the video (like poles repel and unlike poles attract, the efect of the magnetic field on ferromagnetic objects, electrostatic adsorption of light small objects, the arc of the Van de Graaf generator along field lines, tip discharge, etc.). Correct phenomena give a high score.

R18 Light and Shadow Temporal Consistency. ReasonX estimates the main light direction L(t), and after camera-motion compensation to the world coordinate system, checks its temporal stability (degrees); the more stable the direction, the more consistent the light/shadow is temporally.

R25 Phase Transition. Qwen3.8-27B judges the correctness of the phase transition process (conditions and direction of solid, liquid, and gas transition, temperature change due to latent heat, volume/morphology changes accompanying phase transition, etc.). A reasonable process gives a high score.

D6 Kinematics and Spatiotemporal Persistence (L3 Physical Layer).

R15 Displacement. SpatialTracker V2 (injecting query points with SAM 3 masks to obtain the object centroid trajectory) computes the L2 norm ∥x(t)∥<sub>2</sub> (meters) of the object worldline, and SAM 3 detects whether the object suddenly appears/disappears and the symmetry of collision rebound. A trajectory consistent with physical expectation gives a high score.

R16 Velocity & Acceleration. The object centroid trajectory of SpatialTracker V2 computes velocity and the second derivative for acceleration, compared with the physical expectation for velocity error ∥v − vˆ∥ (m/s), with 24 fps as the time base. Smaller error is better.

R17 Angular Velocity. Any6D estimates the object rotation, combined with VGGT-Ω geometry to compute the angular velocity error ∥ω − ωˆ∥ (rad/s), and additionally computes the coeficient of variation CV of angular velocity under free rotation (free rotation should be constant). Smaller error and CV are better.

R20 Motion Continuity. VFIMamba and SpatialTracker V2 reconstruct odd frames by interpolating even frames and compare with the real odd frames to obtain the interpolation error, and simultaneously compute the jerk (rate of change of acceleration) of the 3D trajectory; smaller error and jerk mean more continuous motion.

R21 Object Permanence. SAM 3 segmentation and VGGT-Ω geometry, combined with Qwen3.8-27B semantics, check whether object identity is maintained in the “move away then move back” scene, whether the depth front-back order constitutes a legal occlusion, and count the number of tracking ID jumps. Stable ID and no illegal occlusion give a high score.

D7 Mechanical Plausibility and Conservation Laws (L3 Physical Layer).

R12 Stability. VGGT-Ω gives the depth point cloud, RANSAC[13] fits the support plane, and SAM 3 locates the bottom contact point of the object and checks whether the center-of-mass projection of the object falls within the support polygon; inside the polygon means stable, otherwise it may tip over.

R22 Dynamic Efects of Forces. Prompt-specific, scale-invariant checks cover Newton’s first law, Galilean relativity, projectile motion, support contact, inclined-plane friction, sliding deceleration, bounce restitution, circular motion, and rolling. Metric scale from Depth Anything 3 is used to check acceleration against gravity. The residuals are aggregated into a score; smaller residuals are better.

R23 Momentum Conservation. SpatialTracker V2 and SAM 3 obtain the object trajectory before and after collision. The mass ratio is directly specified in the prompt, and we compute the momentum residual e = ∥p − pˆ∥/∥p∥ accordingly. Smaller residual is better.

R24 Energy Conservation. SpatialTracker V2, SAM 3, and the metric scale of Depth Anything 3 anchor the normalized coordinates to meters. The masses are obtained from the mass ratio specified in the prompt, so that kinetic energy KE and potential energy PE = gh can be computed without ambiguity; conservative scenarios should be approximately constant, and friction-involving scenarios should decrease monotonically and not rise spontaneously. The less deviation from constant/monotonic, the better.

## D.3 Mapping Physical Quantities to 1 to 5 Scores

Table A15 gives the 1 to 5 threshold segments for each physical quantity. The mapping rule is unified: for quantities where “smaller is better” (direction ↓), values successively below the 1st/2nd/3rd/4th segment threshold get 5/4/3/2 points, otherwise 1 point; for quantities where “larger is better” (direction ↑), values successively above the corresponding threshold get 5/4/3/2 points. Without simulation GT (T2V/I2V setting), geometric dimensions are mapped by the above rule using the physical quantities output by specialized models (e.g., R01 depth-order accuracy, R02 normal-depth consistency, R03 normalized Chamfer), while semantic/scale/pose dimensions (R04/R06/R07/R10/R14/R25, etc.) are directly given 1 to 5 by Qwen3.8-27B; the thresholds also connect to the rationality conventions of geometric evaluation (e.g., R02 normal-depth consistency takes 15<sup>◦</sup> as good), and relax the reconstruction noise baseline of specialized models (e.g., R05/R19).

## D.4 Scoring Principles for Semantic and Perceptual Dimensions

D1 to D3 do not extract numerical physical quantities, but are directly scored by specialized perception models or MLLM with 1 to 5 semantic/numerical anchors, explained per dimension below.

D1 Condition and Semantic Fidelity (L1 Semantic Layer).

IF-1 Subject Adherence. Qwen3.8-27B judges whether the generated video contains the subject object specified by the prompt, assisted by CLIP/DINO objective reference scores; complete subject gives a high score.

IF-2 Attribute Adherence. Qwen3.8-27B checks whether the attributes of the subject, such as color, material, and quantity, are consistent with the prompt (including color matching); if consistent, it gives a high score.

Table A15 Mapping of physical quantities to threshold segments and 1 to 5 scores. Thresholds for each segment are listed in the order “5 points to 1 point”; DIR marks the direction (↓ smaller is better, ↑ larger is better).

<table><tr><td rowspan=1 colspan=11>METRIC (dim)                                       THRESHOLDS (5→1)                             UNIT DIR</td></tr><tr><td rowspan=1 colspan=2>R01_depth_order accuracy (R01)     1.00</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>0.50/</td><td rowspan=3 colspan=6>0.25                                                          N/A     ↑0↓0.20                                                          N/A     →</td><td rowspan=2 colspan=1></td></tr><tr><td rowspan=1 colspan=2>R02_normal_depth_consistency (R02)  15  / 2</td><td rowspan=1 colspan=1>5 / 35</td><td rowspan=1 colspan=1>750</td></tr><tr><td rowspan=1 colspan=2>R03_chamfer_norm (R03)            0.02 /</td><td rowspan=1 colspan=1>0.05 /</td><td rowspan=1 colspan=1>0.10 /</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>R04_vlm_reasonability (R04)         N/A (Qw</td><td rowspan=1 colspan=1>en3.8-2</td><td rowspan=1 colspan=1>7B jud</td><td rowspan=1 colspan=5>ges reasonable size range, directly 1 to 5)                        N/</td><td rowspan=1 colspan=1>A     ←</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>R05 deform (R05)                   0.15</td><td rowspan=1 colspan=1>0.25 /</td><td rowspan=1 colspan=1>0.40 /</td><td rowspan=1 colspan=3>0.60</td><td rowspan=1 colspan=2>N/</td><td rowspan=1 colspan=1>A     →</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>R09_cv (R09)                       0.05</td><td rowspan=1 colspan=1>/ 0.10 /</td><td rowspan=1 colspan=1>0.20 /</td><td rowspan=2 colspan=4>/0.40(Qwen3.8-27B compares with the pose change specified by the prompt, directly 1 to 5)</td><td rowspan=1 colspan=3>N/A     →</td></tr><tr><td rowspan=1 colspan=2>R10_vlm_pose_match (R10)          N/A</td><td rowspan=1 colspan=1>(Qwen</td><td rowspan=1 colspan=1>3.8-271</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan=3 colspan=3></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td rowspan=2 colspan=3>N/A     ←m</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>R15_worldline_12 (R15)              0.10</td><td rowspan=1 colspan=1>0.30</td><td rowspan=1 colspan=1>0.50/</td><td rowspan=1 colspan=2>1.00</td><td></td></tr><tr><td rowspan=2 colspan=2>R16_vel_err (R16)                  0.50</td><td rowspan=2 colspan=1>1.00</td><td rowspan=2 colspan=1>2.00</td><td rowspan=8 colspan=6>5.00                                                          m/s1.00                                                          rad/s0.50                                                          N/A     →0→0.70                                                          N/A0.55                                                          N/A</td><td rowspan=2 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>R17_angvel_err (R17)               0.10</td><td rowspan=1 colspan=1>0.30</td><td rowspan=1 colspan=1>0.50</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>R17_rot_cv (R17)                   0.05</td><td rowspan=1 colspan=1>/0.10</td><td rowspan=1 colspan=1>0.20</td><td rowspan=2 colspan=1></td></tr><tr><td rowspan=1 colspan=2>R2x_light_angle (R13/R18)           5 / 1</td><td rowspan=1 colspan=1>0/20</td><td rowspan=1 colspan=1>45</td></tr><tr><td rowspan=1 colspan=2>R19_drift (R19)                     0.10</td><td rowspan=1 colspan=1>0.25</td><td rowspan=1 colspan=1>0.45</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>R19_flow_cv (R19)                  0.10</td><td rowspan=1 colspan=1>0.20</td><td rowspan=1 colspan=1>0.35</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>R22_traj_err (R22)                 0.10</td><td rowspan=1 colspan=1>0.30</td><td rowspan=1 colspan=1>0.50</td><td rowspan=1 colspan=5>1.00                                                          N/</td><td rowspan=1 colspan=1>A     →</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>R23_mom_err (R23)                 0.30</td><td rowspan=1 colspan=1>0.60</td><td rowspan=1 colspan=1>1.00</td><td rowspan=5 colspan=6>2.00                                                          N/A     →5.00                                                          N/A     →9910/0.9881                                                   N/A     ←0.80                                                          N/A     ←741 0.9640                                                   N/A     ←</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=2>R24_energy_err (R24)               0.50</td><td rowspan=1 colspan=1>/1.00</td><td rowspan=1 colspan=1>2.00</td><td rowspan=4 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>4 / 0.9</td><td rowspan=1 colspan=1>0.</td></tr><tr><td rowspan=1 colspan=1>TQ1_</td><td rowspan=1 colspan=1>smoothness (TQ-1)             0.98 /</td><td rowspan=1 colspan=1>0.95 /</td><td rowspan=1 colspan=1>0.90</td></tr><tr><td rowspan=1 colspan=1>TQ4_</td><td rowspan=1 colspan=1>bg_cons (TQ-4)                0.991</td><td rowspan=1 colspan=1>2/0.98</td><td rowspan=1 colspan=1>360.9</td></tr></table>

IF-4 Spatial Relation Adherence. Qwen3.8-27B judges whether the spatial relations between objects (above/- below/left/right/front/back/containment, etc.) conform to the prompt; conforming gives a high score.

IF-5 Style Adherence. Qwen3.8-27B judges whether the artistic/appearance style of the frame conforms to the prompt; conforming gives a high score.

IF-7 Negation Adherence. Qwen3.8-27B checks whether the negation constraints in the prompt (e.g., “no X”, “Y does not appear”) are obeyed; obeying gives a high score.

IF-8 Scene Matching. Qwen3.8-27B judges whether the overall scene/environment is consistent with the scene described by the prompt; consistent gives a high score.

IF-9 Camera Motion Adherence. Qwen3.8-27B judges whether the camera motion (push/pull/pan/move/orbit, etc.) conforms to the prompt; conforming gives a high score.

IF-10 Reference Condition Fidelity. Evaluated only for I2V; aggregated from reference-image fidelity subindicators by DINOv3 + CLIP: IF-10a scene structure fidelity (DINOv3 feature similarity from reference image to first frame), IF-10c color/style continuity (DINOv3 Gram-matrix style distance, degraded to histogram KL divergence), IF-10d character face consistency (DINOv3 fine-feature tracking from reference image to video frame); higher fidelity gives a high score.

D2 Behavioral, Causal, and Commonsense Consistency (L1 Semantic Layer).

IF-3 Action Adherence. Qwen3.8-27B judges whether the action of the person/object conforms to the prompt (including human actions); conforming gives a high score.

IF-6 Temporal Adherence. Qwen3.8-27B judges whether the event timing conforms to the prompt (including temporal semantics); conforming gives a high score.

AC-1 Action Causality. Adopts counterfactual action pairs: generate two versions of the video, “with action” and “without action”, for the same scene; SpatialTracker V2 tracks the 3D trajectory of the object, quantifies the causal efect (displacement amplification ratio, direction consistency), and then Qwen3.8-27B semantically verifies whether the result is reasonable; both efect and semantics reasonable give a high score.

CS-1 Physical Commonsense. Qwen3.8-27B multiple-choice judgment plus explanatory reasonableness scoring, examining basic physical intuition (heavy objects fall, light objects float, fire rises, water flows down, shadow direction, etc.); reasonable gives a high score.

CS-2 Biological Commonsense. Qwen3.8-27B judges the reasonableness of biological behavior (animal movement patterns, plant growth direction, human pose naturalness, etc.); reasonable gives a high score.

CS-3 Social Commonsense. Qwen3.8-27B judges the reasonableness of scene logic (table and chair placement, naturalness of character interaction, daily causal consistency, etc.); reasonable gives a high score.

D3 Perceptual and Temporal Quality (L2 Perceptual Layer).

FQ-1 Subject Consistency. DINOv3 computes the cosine similarity of subject features between adjacent frames, mapped to 1 to 5 (threshold in Table A15); higher similarity is better.

FQ-2 Aesthetic Quality. FineVQ [11] (CVPR 2025, InternVL2-8B + regression head) scores the video aesthetic quality; the raw regression value is calibrated to 1 to 5 by distribution quantiles; higher score is better.

TQ-1 Motion Smoothness. OpenCV/Farneback [3, 12] optical flow computes motion acceleration smoothness max(0, 1 − average acceleration/2), mapped to 1 to 5 (threshold in Table A15); smoother is better.

TQ-2 Temporal Flickering. Measured by pixel diferences in static regions of adjacent frames; smaller diference is better, mapped to 1 to 5.

TQ-3 Dynamic Degree. Measured by the optical flow magnitude distribution; being consistent with the dynamic expectation of the prompt gives a high score.

TQ-4 Background Consistency. CLIP computes the cosine similarity of background features between adjacent frames, mapped to 1 to 5 (threshold in Table A15); higher similarity is better.

TQ-5 Imaging Quality. Measured by Laplacian sharpness and exposure reasonableness; higher quality is better.

## E Additional Prompt-Set Statistics

The dataset contains 600 prompt groups, each paired with T2V and I2V versions. Table A16 summarizes their composition and the share of embodied scenarios.

Table A16 Statistics of the 600 prompt groups: category counts, embodied samples, and modality pairing.
<table><tr><td>Prompt Category</td><td>Count</td><td>Embodied Samples</td><td>Description</td></tr><tr><td>Dimension-specific</td><td>230</td><td>0</td><td>5 per dimension</td></tr><tr><td>Core combinations</td><td>230</td><td>40</td><td>23 paired dimensions</td></tr><tr><td>High-difficulty supplement</td><td>140</td><td>140</td><td>All involve embodied agents</td></tr><tr><td>Total</td><td>600</td><td>180 (30%)</td><td>Paired T2V/I2V versions</td></tr></table>

## F Generation Protocol Details

All models generate all 600 prompt groups one by one. T2V is conditioned on text, I2V is conditioned on the first-frame reference image synthesized by GPT-image-2, ensuring T2V/I2V share the same initial scene (see main text Section 3.2). The unified output specification is 720p, about 10 seconds (about 240 frames, 24 fps), with a fixed random seed; each model uses its oficially recommended inference configuration and weights. Generated videos are uniformly resampled to the resolution and frame rate required by the evaluators before evaluation. The generation scripts, model versions, and hyperparameters will be released with the code to support third-party reproducibility.

## G Evaluation Cost and Reproducibility

For reproducibility, the versions and weights of specialized models and the random seed are fixed during evaluation. The experiments cover 12 model configurations and 600 prompt groups (about 7,200 videos); the evaluation models run on a single H20 GPU (95 GB). Per-video runtime and total GPU hours vary with hardware and the evaluated dimensions. Further statistics, the runtime environment, and model-weight versions will be provided with the code release.