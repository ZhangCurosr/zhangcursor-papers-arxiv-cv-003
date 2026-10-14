# PLAN-BENCH: CAN EXISTING VLMS GO WELL BE-YOND PERCEPTION TO REASON AND PLAN IN EMBOD-IED WORLDS?

Juntong Feng<sup>1∗</sup>, Qingan Liu<sup>1∗</sup>, Jingru Dong<sup>1</sup>, Yue Zhu<sup>1</sup>, Weiming Zheng<sup>1</sup>, Yujie Cheng<sup>1</sup> Xinzhe Han<sup>1</sup>, Shuai Hao<sup>1</sup>, Yunzhi Zhuge<sup>1</sup>, Haiwen Diao<sup>2†</sup>, Huchuan Lu<sup>1</sup> <sup>1</sup>Dalian University of Technology, <sup>2</sup>Nanyang Technological University

## ABSTRACT

Vision-Language Models (VLMs) have achieved strong performance in visual perception and spatial reasoning, leading to growing interest in using them as high-level planners for embodied AI. However, physical tasks require models to integrate spatial information, reason over temporal states, and generate multi-step plans under various constraints, raising the question of whether current VLMs can reliably support such reasoning and planning independently of execution. We introduce Plan-Bench, a benchmark for evaluating the high-level reasoning and multi-step planning capabilities of VLMs relevant to embodied worlds, while isolating the influence of action execution. It covers three complementary dimensions, spanning 16 subclasses and 699 curated samples. Extensive evaluation of 28 VLMs reveals room for improvement, with even the strongest model achieving only 75.0% accuracy. We further find that model scaling generally improves these capabilities. Ablation studies also indicate larger models show increasing potential as high-level planners. Overall, Plan-Bench provides a lightweight probe for evaluating VLMs’ reasoning and planning capabilities relevant to embodied environments and assessing their potential as decision makers for physical tasks. Our benchmark and code are available at https://github.com/MinneFeng/Plan-Bench.

## 1 INTRODUCTION

Recent advances in vision-language models (VLMs) (OpenAI, 2026, Qwen Team, 2026a, Diao et al., 2026d;b) have substantially strengthened their capabilities in visual perception and spatial reasoning, leading to growing interest in using them as high-level planners for embodied AI (Zhang et al., 2026, Liu et al., 2026, Yang et al., 2026), particularly in complex physical tasks. In such settings, models often need to integrate spatial information, reason over evolving temporal states, and generate multi-step plans under various constraints. These demands expose an important and underexplored challenge: can existing VLMs go well beyond perception to reason and plan in embodied worlds?

Notably, existing benchmarks have not evaluated these capabilities of VLMs for embodied tasks. As shown in Figure 1, multimodal reasoning benchmarks (Fu et al., 2026, Yao et al., 2026) primarily focus on visual information. Embodied benchmarks (Fan et al., 2026, Han et al., 2026) mainly evaluate embodied tasks, mixing errors from decisions and execution. LongAct (Zhu et al., 2026) attempts to reduce the influence of control and better assess decision making, but evaluates agents rather than VLMs. It therefore remains unclear whether VLMs can perform reliable reasoning and planning in embodied scenarios when high-level decisions are separated from low-level execution.

To bridge this gap, we introduce Plan-Bench, a benchmark designed to evaluate VLMs’ high-level reasoning and multi-step planning relevant to embodied tasks. As illustrated on the right side of Figure 1, Plan-Bench systematically covers three dimensions: Spatial Cognition, Temporal Reasoning, and Multi-Step Planning, spanning 16 subclasses and 699 samples. It is constructed through a three-stage pipeline consisting of task definition, dual-source visual data acquisition, and manual QA construction with quality assurance, resulting in a diverse and carefully curated benchmark dataset. Plan-Bench includes both single-choice and open-ended questions, and provides answer hints for samples with multiple valid answers to guide more reliable LLM-as-judge evaluation.

![](images/d9afc8c737ef14465c64392b0213b1f377d1e359af9fd7e1b6210a77b127986f.jpg)  
Figure 1: Overview of Plan-Bench. The left illustrates the limitations of existing benchmarks, which either evaluate static VLM capabilities or couple high-level reasoning with low-level execution. The right presents the composition of Plan-Bench, which evaluates execution-decoupled high-level reasoning and planning across Spatial Cognition, Temporal Reasoning, and Multi-Step Planning.

We evaluate Plan-Bench on 28 VLMs. The results show that current VLMs remain limited in reasoning and planning for embodied scenarios. The best-performing model, Qwen3.8-Max (Qwen Team, 2026a), achieves only 75.0%. We further observe that performance generally improves with model scaling, with some variation across model families. Our ablation studies show that larger models exhibit greater potential as high-level planners. Overall, Plan-Bench provides a principled testbed for evaluating reasoning and planning in VLMs and a lightweight framework for assessing their potential as high-level embodied planners. Our main contributions are summarized as follows:

• Designing a reasoning and planning benchmark for VLMs in embodied worlds. Plan-Bench is an easy-to-deploy benchmark spanning 16 subclasses and 699 samples, enabling direct evaluation of VLMs’ high-level capabilities independently of low-level execution.

• Providing a comprehensive evaluation of existing VLMs. We benchmark 28 representative VLMs and reveal their capabilities in high-level reasoning and multi-step planning.

• Revealing current limitations and potential. Evaluation across proprietary and opensource models shows that reasoning and planning in embodied scenarios remain challenging, while larger models exhibit greater potential as high-level planners for embodied tasks.

## 2 RELATED WORK

## 2.1 MULTIMODAL REASONING BENCHMARKS

Existing benchmarks have gradually evolved from perception toward complex reasoning. MME (Fu et al., 2026) evaluates both perceptual and cognitive abilities, while LENS (Yao et al., 2026) organizes multimodal evaluation from understanding to reasoning. Recent benchmarks such as VisuLogic (Xu et al., 2026b), MIRA (Zhou et al., 2026), and PuzzleWorld (Li et al., 2026a) further explore chain-of-thought and open-ended multi-step reasoning. Beyond single-image reasoning, MuirBench (Wang et al., 2025a), MMIU (Meng et al., 2025), EgoPlan-Bench2 (Qiu et al., 2026), among others (Li et al., 2026b, Meng et al., 2025) extend evaluation to multiple images or frames, covering temporal changes and long-range visual evidence. These benchmarks broaden the scope and complexity of multimodal reasoning. However, existing benchmarks rarely target the reasoning and planning demands of embodied worlds. Plan-Bench addresses this gap by evaluating these capabilities across Spatial Cognition, Temporal Reasoning, and Multi-step Planning.

![](images/1ed64e4a4de236fccea56c461b6a16297ed4373dce1cbe59277347f0b58d4d58.jpg)  
Figure 2: Overview of the Plan-Bench construction pipeline, including capability-driven task definition, dual-source data acquisition, manual annotation, and quality assurance.

## 2.2 MULTI-STEP TASKS IN EMBODIED ENVIRONMENTS

Recent embodied systems increasingly employ VLMs (Diao et al., 2024, Diao et al.) as decision makers. Harness VLA (Zhang et al., 2026), Goal2Skill (Liu et al., 2026), and HiVLA (Yang et al., 2026), among others (Shi et al., 2025, Li et al., 2025, Yang et al., 2025c), adopt hierarchical frameworks that separate high-level reasoning and planning from low-level execution. Meanwhile, several benchmarks have begun to evaluate these capabilities in embodied settings. EmbodiedBench (Yang et al., 2025b) and RoboCerebra (Han et al., 2026) evaluate embodied agents through interactive tasks or hierarchical VLM-VLA systems, where performance jointly reflects reasoning, grounding, and action execution. RoboSPA (Fan et al., 2026) further evaluates VLA models on reasoning and procedural planning, but still couples reasoning with low-level execution. LongAct (Zhu et al., 2026) attempts to reduce the influence of control to better assess planning, yet evaluates complete embod ied agents rather than VLMs alone. Plan-Bench instead decouples embodied action execution and directly evaluates VLMs’ reasoning and planning by constructing a benchmark of 699 questions, providing a lightweight interface for isolating the capabilities required by embodied tasks.

## 3 PLAN-BENCH

We introduce Plan-Bench, a benchmark designed to systematically evaluate the reasoning and planning capabilities of VLMs in embodied tasks. In this section, we first describe the construction pipeline, then present the details of data distribution, and finally introduce the evaluation protocol.

## 3.1 CONSTRUCTION PIPELINE

To ensure high quality and validity, we adopt a task-driven, multi-stage construction pipeline, as illustrated in Figure 2. The overall process consists of three stages:

Capability Taxonomy and Task Definition. We first organize the capabilities required for reasoning and planning in embodied environments into a taxonomy. The taxonomy comprises three core categories corresponding to Spatial Cognition, Temporal Reasoning, and Multi-Step Planning. Based on this taxonomy, we further define a set of subtasks, balancing category diversity with task relevance to the targeted capabilities. This step provides explicit guidance for data acquisition, determining what types of visual evidence and task conditions should be collected or constructed.

Dual-Source Data Acquisition. We collect visual data from two sources. For subclasses requiring precise control over factors such as scene layout and viewpoint, we employ controlled-generation platforms such as Three.js (Danchilla, 2012) and OmniGibson (Li et al., 2023) to configure taskrelevant variables for consistent construction. For those that benefit from more natural visual distri butions, we select suitable images from existing datasets such as CLEVR (Johnson et al., 2017) and Captaincook4d (Peddi et al., 2024), among others (Yang et al., 2019, Dai et al., 2026, Ding et al., 2023, Lin et al., 2020), to increase visual variety and capture greater real-world scene complexity.

![](images/c41da510fb487c237e6c0b9e3fa6f2393e57072cb532169ef7d9651311d0ddb8.jpg)  
Figure 3: Illustration of representative samples from the three dimensions of Plan-Bench, including Spatial Cognition, Temporal Reasoning, and Multi-Step Planning.

QA Annotation and Quality Assurance. Based on the acquired images, we manually construct 2,500+ questions and answers according to the task specification of each subclass. All samples are then independently reviewed by three annotators who are not involved in the initial annotation process. The review considers answer correctness, image-QA consistency, and relevance to the corresponding task definition. A sample is accepted only when three reviewers agree that it satisfies all criteria. Finally, this process yields 699 high-quality questions to construct Plan-Bench.

## 3.2 DATA DISTRIBUTION

Plan-Bench comprises 699 instances, including 183 single-choice questions and 516 open-ended questions. Our benchmark covers 16 subtasks across three primary categories, providing a comprehensive evaluation of VLMs’ reasoning and planning capabilities in diverse embodied task settings. Figure 3 presents representative examples from each category in Plan-Bench. The design motivations and task compositions of these three categories are detailed below:

Spatial Cognition. This category contains 259 instances across six subtasks: Occlusion Order, Mirror Distinction, Action Recovery, Target Discovery, State Recognition, and Spatial Relations. It is designed to reflect embodied settings where VLMs may observe the same scene through multiple cameras or viewpoints and must integrate these observations to form a coherent understanding of the environment. It also considers challenging spatial cues such as mirrors and reflections, requiring VLMs to distinguish reflected content from physical entities and reason correctly about the scene.

Temporal Reasoning. This category contains 210 instances across five subtasks: Progress Tracking, Temporal Ordering, Persistent Counting, Instruction Alignment, and Motion Retrieval. This category includes many egocentric multi-image sequences, designed to reflect embodied tasks such as cooking and navigation, where VLMs must reason over temporally evolving observations from a first-person perspective. It evaluates whether VLMs can understand temporal dependencies across sequential observations, including ordering events, tracking target actions or states, and identifying recurring objects across frames. These tasks further assess whether models can preserve relevant historical context and maintain consistent reasoning throughout extended embodied interactions.

Multi-Step Planning. This category contains 230 instances across five subtasks: Constraint Sequencing, Capacity Planning, Rearrangement Planning, Dependency Planning, and Route Planning. It reflects embodied scenarios where VLMs must first reason about task goals, scene constraints, and action dependencies to produce plans before execution. It evaluates whether VLMs can translate such information into action sequences toward a target. Successful planning may indicate the potential of VLMs to serve as more stable high-level decision makers in physical systems.

## 3.3 EVALUATION PROTOCOL

We evaluate model responses using an LLM-as-judge approach. Many open-ended questions in Plan-Bench admit multiple valid answers due to equivalent object references, alternative orderings, or other semantically equivalent forms. To handle such cases, we pair these samples with answer hints that specify additional valid alternatives or equivalence conditions beyond the ground-truth answer, such as valid order swaps, position exchanges, or alternative object names. DeepSeek-V4- Flash (Xu et al., 2026a) serves as the judge, taking the model output, ground-truth answer, and answer hint as input to determine whether the response should be considered correct under these acceptable alternatives. To validate the reliability of the evaluation process, we randomly sample 200 instances for manual verification, and the judge achieves 100% agreement with human evaluation on these samples. The full judge prompt is provided in the Appendix B.2 for reproducibility.

## 4 EXPERIMENTS

In this section, we present an evaluation of VLMs on Plan-Bench to examine their reasoning and planning capabilities in embodied worlds. We first introduce the experimental setup, then report the main results on 28 VLMs, and finally provide ablation studies to further investigate model behaviors.

## 4.1 SETUP

Models. We evaluate Plan-Bench on a broad suite of 28 VLMs, including 23 open-source models and 5 commercial models. The evaluated models are generally grouped into four categories according to model scale and deployment setting: (i) lightweight models with around 2B-4B parameters, (ii) middle-scale models with 7B–12B parameters, (iii) larger open-source models, comprising 27B– 31B dense models and Mixture-of-Experts (MoE) models, and (iv) commercial frontier models. For open-source models, we evaluate a diverse set of model families with different architectures, including Qwen3.8 (Qwen Team, 2026c), Qwen3.5 (Qwen Team, 2026b), Qwen3-VL (Yang et al., 2025a), Gemma-4 (Team et al., 2026a), Step3-VL (Huang et al., 2026), GLM-4.6V (Hong et al., 2025), InternVL3.5 (Wang et al., 2025b), Ovis (Lu et al., 2025), SenseNova-U1 (Diao et al., 2026d), NEO (Diao et al., 2026a), and MiMo-VL (Yue et al., 2025). For commercial models, we evaluate GPT-5.6-sol (OpenAI, 2026), Kimi-K3 (Team et al., 2026b), Gemini-3.1-pro-preview (Google, 2026), Claude-Opus-4.8 (Anthropic, 2026), and Qwen3.8-Max (Qwen Team, 2026a).

Table 1: Comparison of different VLMs on Plan-Bench. We report performance across the three task categories: Spatial Cognition, Temporal Reasoning, and Multi-Step Planning, together with the overall accuracy. The best results in each column are bolded. The random baseline is computed by assigning 0% to open-ended questions and 25% to single-choice questions. All values are rounded to the nearest tenth of a percentage point and reported as percentages (%).
<table><tr><td>Model</td><td>Spatial Cognition</td><td>Temporal Reasoning</td><td>Multi-Step Planning</td><td>Overall (699)</td></tr><tr><td>Random Baseline</td><td>10.4</td><td>4.2</td><td>4.3</td><td>6.5</td></tr><tr><td colspan="5">▼ Commercial Models</td></tr><tr><td>Qwen3.8-Max (Qwen Team, 2026a)</td><td>66.4</td><td>78.1</td><td>81.7</td><td>75.0</td></tr><tr><td>Kimi-K3 (Team et al., 2026b)</td><td>69.5</td><td>78.6</td><td>63.5</td><td>70.2</td></tr><tr><td>GPT-5.6-sol (OpenAI, 2026)</td><td>56.4</td><td>69.5</td><td>73.0</td><td>65.8</td></tr><tr><td>Claude-Opus-4.8 (Anthropic, 2026)</td><td>42.5</td><td>49.1</td><td>63.5</td><td>51.4</td></tr><tr><td>Gemini-3.1-Pro-Preview (Google, 2026)</td><td>39.8</td><td>57.1</td><td>43.9</td><td>46.4</td></tr><tr><td colspan="5">▼ Large Open-Weight Models</td></tr><tr><td>Qwen3.8-27B (Qwen Team, 2026c)</td><td>48.7</td><td>65.2</td><td>74.8</td><td>62.2</td></tr><tr><td>Qwen3.5-27B (Qwen Team, 2026b)</td><td>39.0</td><td>65.7</td><td>66.1</td><td>55.9</td></tr><tr><td>Qwen3.5-35B-A3B (Qwen Team, 2026b)</td><td>34.8</td><td>62.4</td><td>54.4</td><td>49.5</td></tr><tr><td>Gemma-4-31B-it (Team et al., 2026a)</td><td>30.5</td><td>51.0</td><td>51.3</td><td>43.5</td></tr><tr><td>Qwen3-VL-30B-A3B (Yang et al., 2025a)</td><td>22.4</td><td>21.9</td><td>15.7</td><td>20.0</td></tr><tr><td>InternVL3.5-30B-A3B (Wang et al., 2025b)</td><td>14.7</td><td>25.2</td><td>13.9</td><td>17.6</td></tr><tr><td colspan="5">▼ Medium Open-Weight Models</td></tr><tr><td>Qwen3.5-9B (Qwen Team, 2026b)</td><td>29.7</td><td>59.5 43.8</td><td>61.7</td><td>49.2</td></tr><tr><td>Step3-VL-10B (Huang et al., 2026)</td><td>25.1 24.7</td><td>31.9</td><td>40.9 29.6</td><td>35.9</td></tr><tr><td>Gemma-4-12B-it (Team et al., 2026a)</td><td>23.9</td><td>35.2</td><td></td><td>28.5</td></tr><tr><td>GLM-4.6V-Flash (Hong et al., 2025)</td><td>19.3</td><td>28.1</td><td>23.0</td><td>27.0</td></tr><tr><td>InternVL3.5-8B (Wang et al., 2025b)</td><td></td><td></td><td>12.2</td><td>19.6</td></tr><tr><td>Ovis2.5-9B (Lu et al., 2025)</td><td>20.9</td><td>24.3</td><td>13.9</td><td>19.6</td></tr><tr><td>NEO-9B-SFT (Diao et al., 2026a)</td><td>23.6</td><td>19.1</td><td>12.6</td><td>18.6</td></tr><tr><td>MiMo-VL-7B-SFT (Yue et al., 2025)</td><td>19.7</td><td>21.4</td><td>13.9</td><td>18.3</td></tr><tr><td>SenseNova-U1 (Diao et al., 2026d)</td><td>17.4</td><td>15.2</td><td>20.9</td><td>17.9</td></tr><tr><td>Qwen3-VL-8B-Instruct (Yang et al., 2025a)</td><td>17.8</td><td>19.5</td><td>16.5</td><td>17.9</td></tr><tr><td>NEO-ov (Diao et al., 2026c)</td><td>18.5</td><td>17.6</td><td>12.6</td><td>16.3</td></tr><tr><td>Ovis2-8B (Lu et al., 2025)</td><td>15.1</td><td>19.1</td><td>13.0</td><td>15.6</td></tr><tr><td colspan="5">▼ Small Open-Weight Models</td></tr><tr><td>Qwen3.5-4B (Qwen Team, 2026b)</td><td>27.0</td><td>52.4</td><td>53.0</td><td>43.2</td></tr><tr><td>Qwen3.5-2B (Qwen Team, 2026b)</td><td>15.1</td><td>24.3</td><td>22.2</td><td>20.2</td></tr><tr><td>Qwen3-VL-2B-Instruct (Yang et al., 2025a)</td><td>14.3</td><td>17.1</td><td>8.3</td><td>13.2</td></tr><tr><td>InternVL3.5-2B (Wang et al., 2025b)</td><td>13.9</td><td>18.6</td><td>5.7</td><td>12.6</td></tr><tr><td>NEO-2B-SFT (Diao et al., 2026a)</td><td>10.4</td><td>7.1</td><td>7.0</td><td>8.3</td></tr></table>

Evaluation. All open-source models are evaluated using VLMEvalKit (Duan et al., 2024) with the default decoding settings from their official repositories, running in BF16 precision on NVIDIA A800 GPUs. Proprietary models are evaluated through their official APIs under the same protocol. Further details on the evaluation prompts and model configurations are provided in the Appendix.

## 4.2 MAIN RESULTS

Current VLMs show limited reasoning and planning capabilities toward embodied tasks. As shown in Table 1, current VLMs exhibit room for improvement on Plan-Bench. Among the 28 evaluated models, Qwen3.8-Max achieves the highest overall accuracy of 75.0%, followed by Kimi-K3 at 70.2%. All remaining models fall below 70%, including GPT-5.6-sol at 65.8%. Among the MoE, 7–12B, and 2–4B groups, the best-performing models are Qwen3.5-35B-A3B, Qwen3.5-9B, and Qwen3.5-4B, achieving 49.5%, 49.2%, and 43.2% overall accuracy, respectively. Beyond these leading models, most of the remaining models in these groups cluster around 20% accuracy, with only a few reaching the high-20% to mid-30% range. These results indicate that reliable reasoning and planning over physical environments remains far from solved for current VLMs.

<table><tr><td rowspan=10 colspan=1>Qwen3.8-MaxKimi-K3Qwen3.8-27BGemma-4-31B-itStep3-VL-10BGLM-4.6V-FlashOvis2.5-9BQwen3.5-2B</td><td rowspan=1 colspan=6>Spatial Cognition</td><td rowspan=1 colspan=5>Temporal Reasoning</td><td rowspan=1 colspan=5>Long-Horizon Planning</td></tr><tr><td rowspan=1 colspan=1>00</td><td rowspan=1 colspan=1>MD</td><td rowspan=1 colspan=1>AR</td><td rowspan=1 colspan=1>TD</td><td rowspan=1 colspan=1>SR</td><td rowspan=1 colspan=1>SpR</td><td rowspan=1 colspan=1>PT</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>PC</td><td rowspan=1 colspan=1>IA</td><td rowspan=1 colspan=1>MR</td><td rowspan=1 colspan=1>CS</td><td rowspan=1 colspan=1>CP</td><td rowspan=1 colspan=1>RP</td><td rowspan=1 colspan=1>DP</td><td rowspan=1 colspan=1>Route</td></tr><tr><td rowspan=1 colspan=1>60.0</td><td rowspan=1 colspan=1>74.3</td><td rowspan=1 colspan=1>54.0</td><td rowspan=1 colspan=1>42.5</td><td rowspan=1 colspan=1>70.0</td><td rowspan=1 colspan=1>93.2</td><td rowspan=1 colspan=1>93.3</td><td rowspan=1 colspan=1>92.5</td><td rowspan=1 colspan=1>38.0</td><td rowspan=1 colspan=1>90.0</td><td rowspan=1 colspan=1>88.0</td><td rowspan=1 colspan=1>92.0</td><td rowspan=1 colspan=1>86.7</td><td rowspan=1 colspan=1>58.0</td><td rowspan=1 colspan=1>84.0</td><td rowspan=1 colspan=1>90.0</td></tr><tr><td rowspan=1 colspan=1>66.0</td><td rowspan=1 colspan=1>71.4</td><td rowspan=1 colspan=1>48.0</td><td rowspan=1 colspan=1>50.0</td><td rowspan=1 colspan=1>90.0</td><td rowspan=1 colspan=1>95.5</td><td rowspan=1 colspan=1>96.7</td><td rowspan=1 colspan=1>90.0</td><td rowspan=1 colspan=1>48.0</td><td rowspan=1 colspan=1>95.0</td><td rowspan=1 colspan=1>76.0</td><td rowspan=1 colspan=1>68.0</td><td rowspan=1 colspan=1>93.3</td><td rowspan=1 colspan=1>10.0</td><td rowspan=1 colspan=1>80.0</td><td rowspan=1 colspan=1>78.0</td></tr><tr><td rowspan=1 colspan=1>46.0</td><td rowspan=1 colspan=1>54.3</td><td rowspan=1 colspan=1>38.0</td><td rowspan=1 colspan=1>32.5</td><td rowspan=1 colspan=1>57.5</td><td rowspan=1 colspan=1>65.9</td><td rowspan=1 colspan=1>100.0</td><td rowspan=1 colspan=1>75.0</td><td rowspan=1 colspan=1>34.0</td><td rowspan=1 colspan=1>67.5</td><td rowspan=1 colspan=1>66.0</td><td rowspan=1 colspan=1>82.0</td><td rowspan=1 colspan=1>90.0</td><td rowspan=1 colspan=1>66.0</td><td rowspan=1 colspan=1>80.0</td><td rowspan=1 colspan=1>62.0</td></tr><tr><td rowspan=1 colspan=1>22.0</td><td rowspan=1 colspan=1>37.1</td><td rowspan=1 colspan=1>28.0</td><td rowspan=1 colspan=1>27.5</td><td rowspan=1 colspan=1>35.0</td><td rowspan=1 colspan=1>36.4</td><td rowspan=1 colspan=1>93.3</td><td rowspan=1 colspan=1>75.0</td><td rowspan=1 colspan=1>24.0</td><td rowspan=1 colspan=1>17.5</td><td rowspan=1 colspan=1>60.0</td><td rowspan=1 colspan=1>68.0</td><td rowspan=1 colspan=1>56.7</td><td rowspan=1 colspan=1>20.0</td><td rowspan=1 colspan=1>68.0</td><td rowspan=1 colspan=1>46.0</td></tr><tr><td rowspan=1 colspan=1>10.0</td><td rowspan=1 colspan=1>45.7</td><td rowspan=1 colspan=1>30.0</td><td rowspan=1 colspan=1>7.5</td><td rowspan=1 colspan=1>35.0</td><td rowspan=1 colspan=1>27.3</td><td rowspan=1 colspan=1>80.0</td><td rowspan=1 colspan=1>55.0</td><td rowspan=1 colspan=1>12.0</td><td rowspan=1 colspan=1>35.0</td><td rowspan=1 colspan=1>52.0</td><td rowspan=1 colspan=1>58.0</td><td rowspan=1 colspan=1>66.7</td><td rowspan=1 colspan=1>12.0</td><td rowspan=1 colspan=1>50.0</td><td rowspan=1 colspan=1>28.0</td></tr><tr><td rowspan=1 colspan=1>16.0</td><td rowspan=1 colspan=1>40.0</td><td rowspan=1 colspan=1>16.0</td><td rowspan=1 colspan=1>10.0</td><td rowspan=1 colspan=1>17.5</td><td rowspan=1 colspan=1>47.7</td><td rowspan=1 colspan=1>70.0</td><td rowspan=1 colspan=1>37.5</td><td rowspan=1 colspan=1>24.0</td><td rowspan=1 colspan=1>17.5</td><td rowspan=1 colspan=1>38.0</td><td rowspan=1 colspan=1>38.0</td><td rowspan=1 colspan=1>43.3</td><td rowspan=1 colspan=1>6.0</td><td rowspan=1 colspan=1>6.0</td><td rowspan=1 colspan=1>30.0</td></tr><tr><td rowspan=1 colspan=1>20.0</td><td rowspan=1 colspan=1>17.1</td><td rowspan=1 colspan=1>12.0</td><td rowspan=1 colspan=1>20.0</td><td rowspan=1 colspan=1>27.5</td><td rowspan=1 colspan=1>29.6</td><td rowspan=1 colspan=1>13.3</td><td rowspan=1 colspan=1>32.5</td><td rowspan=1 colspan=1>16.0</td><td rowspan=1 colspan=1>12.5</td><td rowspan=1 colspan=1>42.0</td><td rowspan=1 colspan=1>22.0</td><td rowspan=1 colspan=1>43.3</td><td rowspan=1 colspan=1>2.0</td><td rowspan=1 colspan=1>8.0</td><td rowspan=1 colspan=1>6.0</td></tr><tr><td rowspan=1 colspan=1>6.0</td><td rowspan=1 colspan=1>28.6</td><td rowspan=1 colspan=1>10.0</td><td rowspan=1 colspan=1>22.5</td><td rowspan=1 colspan=1>15.0</td><td rowspan=1 colspan=1>13.6</td><td rowspan=1 colspan=1>43.3</td><td rowspan=1 colspan=1>22.5</td><td rowspan=1 colspan=1>12.0</td><td rowspan=1 colspan=1>5.0</td><td rowspan=1 colspan=1>42.0</td><td rowspan=1 colspan=1>50.0</td><td rowspan=1 colspan=1>30.0</td><td rowspan=1 colspan=1>4.0</td><td rowspan=1 colspan=1>12.0</td><td rowspan=1 colspan=1>18.0</td></tr></table>

Figure 4: Fine-grained performance of 8 representative VLMs across the 16 subtasks in Plan-Bench. Each cell reports accuracy (%), with color intensity indicating performance.

Spatial cognition remains challenging for current VLMs. As shown in Table 1, many models stay close to the 10.4% random baseline. For example, Qwen3-VL-2B-Instruct and InternVL3.5-2B achieve only 14.3% and 13.9%, while Qwen3.5-2B scores 15.1%. This weakness is not confined to the lightweight models, as Qwen3-VL-8B-Instruct and NEO-ov reach 17.8% and 18.5%. Across the full set of evaluated models, only nine models exceed 30% accuracy, while ten remain around or below 20%. Even among stronger frontier models, the highest accuracy is only 69.5%, achieved by Kimi-K3, followed by Qwen3.8-Max at 66.4%. These results show that current VLMs still struggle to reliably integrate spatial evidence across multiple viewpoints and complex scene configurations.

Model scaling generally benefits embodied reasoning and planning. Within the same model family and architecture, performance improves with scale. Qwen3.5, for example, increases from 20.2% at 2B to 49.2% at 9B and further to 55.9% at 27B. Similar trends are observed for Gemma-4 and InternVL3.5. Gemma-4 improves from 28.5% at 12B to 43.5% at 31B, while InternVL3.5 rises from 12.6% at 2B to 19.6% at 8B. However, this relationship is not consistent across model families. Qwen3.8-27B reaches 62.2%, higher than Gemini-3.1-Pro-Preview at 46.4%. Similarly, Qwen3.5- 9B achieves 49.2%, outperforming the much larger Gemma-4-31B at 43.5%. Overall, larger models tend to perform better, while architecture and training strategy also remain important factors.

## 4.3 ABLATION STUDY

Strong overall performance does not imply consistent fine-grained capabilities. Figure 4 reveals variation across the 16 subtasks. Even the strongest models exhibit weaknesses on individual tasks. Qwen3.8-Max, for example, exceeds 70% accuracy on several subtasks but reaches only 38.0% on Persistent Counting, while Kimi-K3 drops to 10.0% on Rearrangement Planning despite strong performance elsewhere. Models with similar overall accuracy may still differ across tasks. Qwen3.5-2B and Ovis2.5-9B achieve comparable overall scores, yet differ considerably across Progress Tracking, Spatial Relations, Constraint Sequencing, and Capacity Planning. Overall, strong or similar aggregate performance can still coexist with task-specific weaknesses and capability differences.

More views do not necessarily lead to better spatial cognition. Current VLMs remain susceptible to interference from misleading viewpoints. We reorganize Spatial Relations under the Spatial Cognition dimension into three settings: 1 right, 1 right + 2 mislead, and 1 right + 4 mislead, where additional views contain misleading cues. The results in Figure 5 show that all seven models perform well when only the informative view is provided, but their accuracy consistently declines as more misleading views are introduced. Larger models generally show greater robustness to such distractions. Qwen3.8-27B drops from 90.9% to 68.2%, while Qwen3.5-2B falls from 93.2% to 22.7%. Yet scale alone does not explain this robustness, as Gemma-4-31B-it is less robust than the smaller Qwen3.5-9B, suggesting that model architecture and training strategy may also play important roles.

![](images/84ca909316302684344aef9339ff0e103b516ba395813e1f052e8f6710720733.jpg)  
Figure 5: Robustness to Increasing Misleading Views across Model Scales.

![](images/9d031ae5783c85e2c977de452ec5f2261cbbb8c107fe82268ab60000b1ce8c20.jpg)  
Figure 6: Navigation and Manipulation Performance across Different Models.

![](images/d77ce758a19693636295f530436b3786ac1b25eb9df17cf4f661ab400a34af49.jpg)

![](images/0a775c5257b0e1bf103e41e3a9ce47640d6b18db644f2d23be28a9af3a1510a3.jpg)  
Figure 7: Comparison between Thinking and Non-Thinking modes on Plan-Bench across Spatial Cognition, Temporal Reasoning, Multi-Step Planning, and overall accuracy. Blue and orange markers denote Thinking and Non-Thinking results, while the annotated values indicate their gaps.

Thinking improves performance, with the largest gains in multi-step planning. We compare ten models that support both Thinking and Non-Thinking modes on Plan-Bench. As shown in Figure 7, Thinking improves nearly all dimension-level and overall results, with the only exception of Qwen3.5-2B, which drops by 1.5 percentage points on Spatial Cognition. Among the three dimensions, Thinking brings the largest gains on Multi-Step Planning, with an average improvement of 31.1 percentage points, compared with 19.4 on Temporal Reasoning and 15.2 on Spatial Cognition. At the overall level, all models benefit from Thinking, with improvements ranging from 4.9 percentage points on Gemma-4-12B-it to 32.6 points on Kimi-K3. Despite longer inference time, Thinking enables models to better handle reasoning and planning tasks in embodied environments.

Navigation and Manipulation are broadly aligned, with model-specific mismatches. We group Plan-Bench into Navigation-related (Route Planning, Instruction Alignment, and Target Discovery) and Manipulation-related (Dependency Planning, Constraint Sequencing, and Occlusion Order). As shown in Figure 6, most models exhibit similar trends. For example, Qwen3.8-Max, Kimi-K3, and GPT-5.6-sol show balanced performance on the two task groups. However, some models exhibit biases, with Qwen3.8-27B achieving about 69.3% on Manipulation task but 54.6% on Navigation task, while Step3-VL-10B shows a gap of roughly 15 percentage points. These results further indicate that larger models have greater potential to serve as high-level decision makers in physical environments, with more stable and consistent performance across diverse embodied tasks.

Hallucination is a source of reasoning and planning errors in the embodied world. We analyze Qwen3.8-Max, the best-performing model on Plan-Bench, focusing on failures in Mirror Distinction, Persistent Counting, and Target Discovery. As shown in Figure 8, the model fails to distinguish real objects from their reflections in Mirror Distinction, hallucinating reflections as physical entities.

![](images/d3a2ba7851e18fb233d059313c526f8bce88186788634091d7a90ecec292a1a8.jpg)  
Figure 8: Representative Examples of Hallucination Errors by Qwen3.8-Max

In Persistent Counting, it incorrectly associates two distinct objects across views due to similar surrounding landmarks, resulting in an incorrect match. A similar phenomenon also appears in Target Discovery. This suggests that failures arise not only from visual recognition errors, but also from hallucinations during high-level reasoning and multi-step planning. Reducing such hallucinations may be important for improving the reliability of VLMs in embodied reasoning and planning.

## 5 LIMITATIONS AND DISCUSSION

We construct Plan-Bench to evaluate the reasoning and planning capabilities of VLMs in embodied tasks across Spatial Cognition, Temporal Reasoning, and Multi-step Planning, while several limitations remain to be further explored and addressed. (i) Continually evolving frontier models. Although we have evaluated diverse models, frontier systems evolve rapidly and performance may change across versions. Future work will track newly released models to assess progress in these capabilities. (ii) Limited data scale and task coverage. Plan-Bench includes 699 samples across 16 subcategories, while there is still room to expand both the dataset scale and task diversity. Further expansion will enable a more comprehensive evaluation of reasoning and planning. (iii) Potential errors in LLM-as-judge. Our LLM-as-judge achieves 100% agreement with human evaluation on 200 sampled cases, but some errors may still exist in the remaining samples. We will further explore more robust and programmatic evaluation procedures to improve evaluation accuracy and reliability.

## 6 CONCLUSION

We introduce Plan-Bench, a lightweight and systematic benchmark for evaluating the high-level reasoning and multi-step planning capabilities of VLMs in embodied scenarios independently of action execution. Plan-Bench covers Spatial Cognition, Temporal Reasoning, and Multi-Step Planning, spanning 16 subclasses and 699 curated samples. Evaluation of 28 VLMs shows that current models still have room for improvement, with the best-performing model achieving only 75.0% accuracy. Ablation studies further reveal that strong overall performance does not guarantee consistent capabilities across subtasks, and that models remain sensitive to misleading evidence and prone to hallucinations during reasoning. We also find that scaling generally improves results, while architecture and training may also influence performance, with Thinking bringing additional gains, particularly for Multi-Step Planning. Our results suggest that larger VLMs show greater potential as high-level planners, but reliable reasoning and planning across diverse embodied tasks remain challenging. Overall, Plan-Bench provides a principled and easy-to-deploy testbed for tracking progress in these capabilities and assessing VLMs as high-level decision makers for embodied AI.

## REFERENCES

Anthropic. Introducing claude opus 4.8, 2026. URL https://www.anthropic.com/news/ claude-opus-4-8.

Yinpei Dai, Hongze Fu, Jayjun Lee, Yuejiang Liu, Haoran Zhang, Jianing Yang, Chelsea Finn, Nima Fazeli, and Joyce Chai. Robomme: Benchmarking and understanding memory for robotic generalist policies. arXiv preprint arXiv:2603.04639, 2026.

Brian Danchilla. Three. js framework. In Beginning WebGL for HTML5, pp. 173–203. Springer, 2012.

Haiwen Diao, Xiaotong Li, Yufeng Cui, Yueze Wang, Haoge Deng, Ting Pan, Wenxuan Wang, Huchuan Lu, and Xinlong Wang. Evev2: Improved baselines for encoder-free vision-language models, 2025. URL https://arxiv. org/abs/2502.06788.

Haiwen Diao, Yufeng Cui, Xiaotong Li, Yueze Wang, Huchuan Lu, and Xinlong Wang. Unveiling encoder-free vision-language models. Advances in Neural Information Processing Systems, 37: 52545–52567, 2024.

Haiwen Diao, Mingxuan Li, Silei Wu, Linjun Dai, Xiaohua Wang, Hanming Deng, Lewei Lu, Dahua Lin, and Ziwei Liu. From pixels to words–towards native vision-language primitives at scale. In International Conference on Learning Representations, volume 2026, pp. 109909–109929, 2026a.

Haiwen Diao, Jiahao Wang, Chenjing Ding, Hanming Deng, Jiangnan Chen, Ruixi Zhang, Ruohui Wang, Wenwen Tong, Xiangyu Fan, Yubo Wang, et al. Sensenova-u1. 5: Towards native unified visual intelligence. arXiv preprint arXiv:2609.11929, 2026b.

Haiwen Diao, Jiahao Wang, Penghao Wu, Yuhao Dong, Yuwei Niu, Yue Zhu, Zhongang Cai, Weichen Fan, Linjun Dai, Silei Wu, et al. From pixels to words–towards native one-vision models at scale. arXiv preprint arXiv:2605.28820, 2026c.

Haiwen Diao, Penghao Wu, Hanming Deng, Jiahao Wang, Shihao Bai, Silei Wu, Weichen Fan, Wenjie Ye, Wenwen Tong, Xiangyu Fan, et al. Sensenova-u1: Unifying multimodal understanding and generation with neo-unify architecture. arXiv preprint arXiv:2605.12500, 2026d.

Henghui Ding, Chang Liu, Shuting He, Xudong Jiang, and Chen Change Loy. Mevis: A large-scale benchmark for video segmentation with motion expressions. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 2694–2703. IEEE, 2023.

Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, et al. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. In Proceedings of the 32nd ACM international conference on multimedia, pp. 11198–11201, 2024.

Zhenxuan Fan, Bo Zhang, Yutong Lin, Yuqian Yuan, Juekai Lin, Liang Liang, Zhuoyi Huang, Wenqiao Zhang, Juncheng Li, Siliang Tang, et al. Robospa: Can vla models go beyond simple scenes and short-horizon tasks? arXiv preprint arXiv:2609.05324, 2026.

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. Advances in Neural Information Processing Systems, 38, 2026.

Google. Gemini 3.1 pro model card, 2026. URL https://deepmind.google/models/ model-cards/gemini-3-1-pro/. Model Card.

Songhao Han, Boxiang Qiu, Yue Liao, Siyuan Huang, Chen Gao, Shuicheng Yan, and Si Liu. Robocerebra: A large-scale benchmark for long-horizon robotic manipulation evaluation. Advances in Neural Information Processing Systems, 38, 2026.

Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, et al. Glm-4.5 v and glm-4.1 v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. arXiv preprint arXiv:2507.01006, 2025.

Ailin Huang, Chengyuan Yao, Chunrui Han, Fanqi Wan, Hangyu Guo, Haoran Lv, Hongyu Zhou, Jia Wang, Jian Zhou, Jianjian Sun, et al. Step3-vl-10b technical report. arXiv preprint arXiv:2601.09668, 2026.

Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2901–2910, 2017.

Alexander Ku, Peter Anderson, Roma Patel, Eugene Ie, and Jason Baldridge. Room-across-room: Multilingual vision-and-language navigation with dense spatiotemporal grounding. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 4392–4412, 2020.

Chengshu Li, Ruohan Zhang, Josiah Wong, Cem Gokmen, Sanjana Srivastava, Roberto Mart´ın-Mart´ın, Chen Wang, Gabrael Levine, Michael Lingelbach, Jiankai Sun, et al. Behavior-1k: A benchmark for embodied ai with 1,000 everyday activities and realistic simulation. In Conference on Robot Learning, pp. 80–93. PMLR, 2023.

Hengzhi Li, Justin Zhang, Brendon Jiang, Alexander Naehu, Regan Song, Megan Tjandrasuwita, Chanakya Ekbote, Steven-Shine Chen, Adithya Balachandran, Wei Dai, et al. Puzzleworld: A benchmark for multimodal, open-ended reasoning in puzzlehunts. In International Conference on Learning Representations, volume 2026, pp. 11819–11843, 2026a.

Jiachun Li, Shaoping Huang, Zhuoran Jin, Chenlong Zhang, Pengfei Cao, Yubo Chen, Kang Liu, and Jun Zhao. Mmr-life: Piecing together real-life scenes for multimodal multi-image reasoning. arXiv preprint arXiv:2603.02024, 2026b.

Yi Li, Yuquan Deng, Jesse Zhang, Joel Jang, Marius Memmel, Caelan Garrett, Fabio Ramos, Dieter Fox, Anqi Li, Abhishek Gupta, et al. Hamster: Hierarchical action models for open-world robot manipulation. In International Conference on Learning Representations, volume 2025, pp. 24040–24068, 2025.

Jiaying Lin, Guodong Wang, and Rynson WH Lau. Progressive mirror detection. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3694–3702. IEEE, 2020.

Zhen Liu, Xinyu Ning, Zhe Hu, Xinxin Xie, Weize Li, Zhipeng Tang, Chongyu Wang, Zejun Yang, Hanlin Wang, Yitong Liu, et al. Goal2skill: Long-horizon manipulation with adaptive planning and reflection. arXiv preprint arXiv:2604.13942, 2026.

Shiyin Lu, Yang Li, Yu Xia, Yuwei Hu, Shanshan Zhao, Yanqing Ma, Zhichao Wei, Yinglun Li, Lunhao Duan, Jianshan Zhao, et al. Ovis2. 5 technical report. arXiv preprint arXiv:2508.11737, 2025.

Fanqing Meng, Jin Wang, Chuanhao Li, Quanfeng Lu, Hao Tian, Tianshuo Yang, Jiaqi Liao, Xizhou Zhu, Jifeng Dai, Yu Qiao, et al. Mmiu: Multimodal multi-image understanding for evaluating large vision-language models. In International Conference on Learning Representations, volume 2025, pp. 38405–38453, 2025.

OpenAI. Introducing gpt-5.6, 2026. URL https://openai.com/zh-Hans-CN/index/gpt-5-6.

Rohith Peddi, Shivvrat Arya, Bharath Challa, Likhitha Pallapothula, Akshay Vyas, Bhavya Gouripeddi, Qifan Zhang, Jikai Wang, Vasundhara Komaragiri, Eric Ragan, et al. Captaincook4d: A dataset for understanding errors in procedural activities. Advances in Neural Information Processing Systems, 37:135626–135679, 2024.

Lu Qiu, Yi Chen, Yuying Ge, Yixiao Ge, Ying Shan, and Xihui Liu. Egoplan-bench2: A benchmark for multimodal large language model planning in real-world scenarios. International Journal of Computer Vision, 134(5):222, 2026.

Qwen Team. Introducing qwen3.8-max, 2026a. URL https://qwen.ai/blog?id=qwen3. 8.

Qwen Team. Qwen3.5, 2026b. URL https://huggingface.co/collections/Qwen/ qwen35. Official Model Card.

Qwen Team. Qwen3.8-27B, 2026c. URL https://huggingface.co/Qwen/Qwen3. 8-27B. Official Model Card.

Lucy Xiaoyang Shi, Brian Ichter, Michael Equi, Liyiming Ke, Karl Pertsch, Quan Vuong, James Tanner, Anna Walling, Haohuan Wang, Niccolo Fusai, et al. Hi robot: Open-ended instruction following with hierarchical vision-language-action models. arXiv preprint arXiv:2502.19417, 2025.

Gemma Team, Sherif El Abd, Vaibhav Aggarwal, Robin Algayres, Alek Andreev, Olivier Bachem, Ian Ballantyne, Cormac Brick, Victor Carbune, Michelle Casbon, et al. Gemma 4 technical report.˘ arXiv preprint arXiv:2607.02770, 2026a.

Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, Jianfeng Cai, Xinyuan Cai, Peizhou Cao, Yuxuan Cao, Ziwei Chai, Y Charles, et al. Kimi k3: Open frontier intelligence. arXiv preprint arXiv:2607.24653, 2026b.

Fei Wang, Xingyu Fu, James Y Huang, Zekun Li, Qin Liu, Xiaogeng Liu, Mingyu Derek Ma, Nan Xu, Wenxuan Zhou, Kai Zhang, et al. Muirbench: A comprehensive benchmark for robust multiimage understanding. In International Conference on Learning Representations, volume 2025, pp. 62624–62650, 2025a.

Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025b.

Yuechen Xie, Xiaoyan Zhang, Yicheng Shan, Zhu Hao, Rui Tang, Rong Wei, Mingli Song, Yuanyu Wan, and Jie Song. Spatialqa: A benchmark for evaluating spatial logical reasoning in visionlanguage models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2645–2657, 2026.

Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chenchen Ling, et al. Deepseek-v4: Towards highly efficient milliontoken context intelligence. arXiv preprint arXiv:2606.19348, 2026a.

Weiye Xu, Jiahao Wang, Weiyun Wang, Zhe Chen, Wengang Zhou, Aijun Yang, Lewei Lu, Houqiang Li, Xiaohua Wang, Xizhou Zhu, et al. Visulogic: A benchmark for evaluating visual reasoning in multi-modal large language models. In International Conference on Learning Representations, volume 2026, pp. 25966–26003, 2026b.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a.

Rui Yang, Hanyang Chen, Junyu Zhang, Mark Zhao, Cheng Qian, Kangrui Wang, Qineng Wang, Teja Venkat Koripella, Marziyeh Movahedi, Manling Li, et al. Embodiedbench: Comprehensive benchmarking multi-modal large language models for vision-driven embodied agents. arXiv preprint arXiv:2502.09560, 2025b.

Tianshuo Yang, Guanyu Chen, Yutian Chen, Zhixuan Liang, Yitian Liu, Zanxin Chen, Chunpu Xu, Haotian Liang, Jiangmiao Pang, Yao Mu, et al. Hivla: A visual-grounded-centric hierarchical embodied manipulation system. arXiv preprint arXiv:2604.14125, 2026.

Xin Yang, Haiyang Mei, Ke Xu, Xiaopeng Wei, Baocai Yin, and Rynson WH Lau. Where is my mirror? In Proceedings ofthe IEEE/CVF international conference on computer vision, pp. 8809– 8818, 2019.

Zhutian Yang, Caelan Garrett, Dieter Fox, Tomas Lozano-P ´ erez, and Leslie Pack Kaelbling. Guiding ´ long-horizon task and motion planning with vision language models. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pp. 16847–16853. IEEE, 2025c.

Ruilin Yao, Bo Zhang, Jirui Huang, Xinwei Long, Yifang Zhang, Tianyu Zou, Shili Xiong, Yi Rong, Yufei Wu, Shichao Su, et al. Lens: Multi-level evaluation of multimodal reasoning with large language models. In International Conference on Learning Representations, volume 2026, pp. 2627–2654, 2026.

Zihao Yue, Zhenru Lin, Yifan Song, Weikun Wang, Shuhuai Ren, Shuhao Gu, Shicheng Li, Peidian Li, Liang Zhao, Lei Li, et al. Mimo-vl technical report. arXiv preprint arXiv:2506.03569, 2025.

Yixian Zhang, Huanming Zhang, Feng Gao, Xiao Li, Zhihao Liu, Chunyang Zhu, Jiaxing Qiu, Yuchen Yan, Jiyuan Liu, Wenhao Tang, et al. Harness vla: Steering frozen vlas into reliable manipulation primitives via memory-guided agents. arXiv preprint arXiv:2607.08448, 2026.

Yiyang Zhou, Haoqin Tu, Zijun Wang, Zeyu Wang, Niklas Muennighoff, Fan Nie, Chaorui Deng, Shen Yan, Haoqi Fan, Yejin Choi, et al. When visualizing is the first step to reasoning: Mira, a benchmark for visual chain-of-thought. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 26154–26164, 2026.

Zilin Zhu, Longteng Guo, Yanghong Mei, Bowen Pang, Zongxun Zhang, Xingjian He, Ruyi Ji, and Jing Liu. When robots do the chores: A benchmark and agent for long-horizon household task execution. arXiv preprint arXiv:2605.14504, 2026.

## A BENCHMARK DETAILS

## A.1 TASK DEFINITIONS

Plan-Bench consists of 16 subtasks that instantiate different reasoning and planning challenges in embodied scenarios. Below, we provide detailed definitions of each subtask and clarify the specific capability evaluated by each task.

Occlusion Order (OO). Given a scene containing partially occluded objects, the model is required to infer their relative ordering from visual occlusion cues. This task evaluates whether the model can reason about object ordering when complete object boundaries are not directly observable.

Mirror Distinction (MD). The model must distinguish real objects from their reflections and correctly reason about the scene content. The task is designed to evaluate whether the model can distinguish real objects from reflections and correctly handle reflection-induced visual ambiguity.

Action Recovery (AR). Given partial information from one viewpoint, the model is asked to recover the corresponding information that is missing or unclear from another viewpoint. This task examines whether the model can compare observations across multiple views and integrate complementary visual cues to reconstruct the complete scene information.

Target Discovery (TD). The model must reason about the target’s spatial relationship under mirror reflections and plan a feasible path toward it. This task evaluates spatial reasoning and path planning in reflection-ambiguous scenes.

State Recognition (SR). Given multiple visual observations of a scene, the model must integrate information while identifying redundantly observed content. This task evaluates multi-view scene understanding and the ability to distinguish new evidence from repeated information.

Spatial Relations (SpR). The model must infer spatial relationships between specified entities from one or multiple views. The task evaluates spatial reasoning and the ability to integrate relevant evidence while ignoring misleading observations.

Progress Tracking (PT). Given a sequence depicting an ongoing activity, the model is required to determine the current progress or stage of the activity. This task evaluates whether the model can track how an embodied procedure evolves over time.

Temporal Ordering (TO). The model must determine the correct temporal order of events, actions, or scene states from sequential visual evidence. The task evaluates the ability to recover temporal dependencies and distinguish earlier events from later ones.

Persistent Counting (PC). The model is asked to count target entities across a visual sequence while maintaining their identity over time. This task evaluates whether the model can accumulate information across frames without repeatedly counting the same persistent entity.

Instruction Alignment (IA). Given an instruction together with a sequence of visual observations, the model must identify the observation that corresponds to the specified instruction. The task evaluates the alignment between language-described actions and their visual realization over time.

Motion Retrieval (MR). The model is required to identify the motion or action segment that matches a specified query from a temporal sequence. This task evaluates whether the model can recognize and retrieve task-relevant motion patterns among multiple observed actions.

Constraint Sequencing (CS). Given a set of actions together with ordering constraints, the model must construct a valid action sequence satisfying all specified constraints. The task evaluates multistep planning under explicit requirements on the relative order of actions.

Capacity Planning (CP). The model is required to produce a feasible plan under capacity limitations on containers or other task-relevant entities. This task evaluates whether the model can jointly reason about action ordering and limited capacity when constructing a multi-step solution.

Rearrangement Planning (RP). Given an initial and a target configuration, the model must determine a sequence of actions that transforms the scene into the desired arrangement. The task evaluates planning capabilities and the consequences of successive rearrangement operations.

![](images/16a9866799d3cf7018917aab9c3e4fa2418c58fc29ac87a610117007d76c9773.jpg)  
Figure 9: Dataset composition of Plan-Bench across three dimensions and 16 subtasks.

Table 2: Data sources and question-type distribution across the 16 subtasks in Plan-Bench.
<table><tr><td>Task</td><td>Subtask</td><td>Single</td><td>Open</td><td>Data Source</td></tr><tr><td rowspan="6">Spatial Cognition</td><td>Action Recovery</td><td>30</td><td>20</td><td>Robomme (Dai et al., 2026)</td></tr><tr><td>Mirror Distinction</td><td>0</td><td>35</td><td>MSD (Yang et al., 2019)</td></tr><tr><td>Occlusion Order</td><td>18</td><td>32</td><td>PMD (Lin et al., 2020) three.js (Danchilla, 2012)</td></tr><tr><td>Spatial Relations</td><td>0</td><td>44</td><td>three.js (Danchilla, 2012)</td></tr><tr><td>State Recognition</td><td>20</td><td>20</td><td>three.js (Danchilla, 2012)</td></tr><tr><td>Target Discovery</td><td>40</td><td>0</td><td>PMD (Lin et al., 2020)</td></tr><tr><td rowspan="5">Temporal Reasoning</td><td>Instruction Alignment</td><td>15</td><td>25</td><td>RxR (Ku et al., 2020)</td></tr><tr><td>Motion Retrieval</td><td>0</td><td>50</td><td>mevis (Ding et al., 2023)</td></tr><tr><td>Persistent Counting</td><td>0</td><td>50</td><td>OmniGibson (Li et al., 2023)</td></tr><tr><td>Progress Tracking</td><td>0</td><td>30</td><td>captaincook4d (Peddi et al., 2024)</td></tr><tr><td>Temporal Ordering</td><td>20</td><td>20</td><td>captaincook4d (Peddi et al., 2024)</td></tr><tr><td rowspan="5">Multi-Step Planning</td><td>Capacity Planning</td><td>30</td><td>0</td><td>three.js (Danchilla, 2012)</td></tr><tr><td>Constraint Sequencing</td><td>0</td><td>50</td><td>CLEVR (Johnson et al., 2017)</td></tr><tr><td>Dependency Planning</td><td>0</td><td>50</td><td>Website SpatialQA (Xie et al., 2026)</td></tr><tr><td>Rearrangement Planning</td><td>0</td><td>50</td><td>OmniGibson (Li et al., 2023)</td></tr><tr><td>Route Planning</td><td>10</td><td>40</td><td>OmniGibson (Li et al., 2023)</td></tr></table>

Dependency Planning (DP). The model must construct an action plan while respecting prerequisite relations among actions or subgoals. This task evaluates whether the model can identify dependencies and ensure that required preconditions are satisfied before executing dependent actions.

Route Planning (Route). Given spatial information about an environment, the model must determine a feasible route from the current location to a specified destination. The task evaluates multi-step navigation planning under spatial connectivity, obstacle, and path-feasibility constraints.

## A.2 DATASET STATISTICS

Plan-Bench maintains a balanced distribution across dimensions and subtasks. As shown in Figure 9, Plan-Bench contains 699 samples distributed across three dimensions, with 259 samples for Spatial Cognition, 210 for Temporal Reasoning, and 230 for Multi-Step Planning. The samples are further distributed across 16 subtasks, with most subtasks containing between 30 and 50 instances. Overall, the benchmark maintains a relatively balanced distribution across both dimension and subtasks, avoiding excessive concentration on any single capability.

Plan-Bench exhibits diversity in visual inputs. As shown in Tables 3 and 4, the benchmark covers samples containing from 1 to 16 images, with 64.9% of the samples involving multiple images and 47.2% containing six or more images. The 3,007 images also span a wide range of resolutions, from below 0.5 MP to above 4 MP, with 35.8% exceeding 1 MP. These statistics highlight the diversity of Plan-Bench in both visual context length and image resolution, rather than relying on a fixed input configuration.

Table 3: Distribution of the number of images per sample.
<table><tr><td>Images</td><td>Samples</td><td>Ratio</td></tr><tr><td>1</td><td>245</td><td>35.1%</td></tr><tr><td>2-5</td><td>124</td><td>17.7%</td></tr><tr><td>6-7</td><td>181</td><td>25.9%</td></tr><tr><td>8-16</td><td>149</td><td>21.3%</td></tr><tr><td>Total</td><td>699</td><td>100%</td></tr></table>

Table 4: Distribution of image resolutions.
<table><tr><td>Resolution</td><td>Images</td><td>Ratio</td></tr><tr><td>≤0.5 MP</td><td>1,430</td><td>47.6%</td></tr><tr><td>0.5-1 MP</td><td>502</td><td>16.7%</td></tr><tr><td>1-2 MP</td><td>363</td><td>12.1%</td></tr><tr><td> $2 { \mathrm { - } } 4 \mathrm { M P }$ </td><td>643</td><td>21.4%</td></tr><tr><td> ${ \geq } 4 \mathrm { M P }$ </td><td>69</td><td>2.3%</td></tr><tr><td>Total</td><td>3,007</td><td>100%</td></tr></table>

Table 5: Distribution of planning lengths in the Multi-Step Planning subset of Plan-Bench.
<table><tr><td>Steps</td><td>Samples</td><td>Percentage</td></tr><tr><td>1–3 steps</td><td>70</td><td>30.43%</td></tr><tr><td>4–6 steps</td><td>131</td><td>56.96%</td></tr><tr><td>7–10 steps</td><td>29</td><td>12.61%</td></tr><tr><td>Total</td><td>230</td><td>100%</td></tr></table>

To further characterize the planning complexity of the Multi-Step Planning subset, we analyze the number of steps required by the reference plans. As shown in Table 5, the 230 samples span planning horizons ranging from 1 to 10 steps. Among them, 70 samples (30.43%) require 1–3 steps, while the majority of the samples, 131 (56.96%), involve 4–6 planning steps. In addition, 29 samples (12.61%) require relatively longer plans of 7–10 steps. This distribution indicates that Plan-Bench covers planning problems with varying horizon lengths, while placing substantial emphasis on multistep tasks that require models to reason over several interdependent actions rather than making only short-horizon decisions.

## A.3 DATASET FILTERING PROCEDURE

We perform a multi-stage filtering procedure to improve the quality and reliability of Plan-Bench. Starting from 2,580 candidate QA pairs, we remove samples that fail one or more quality requirements and finally obtain 699 validated samples. The entire filtering process is conducted before model evaluation and is independent of model performance.

Each candidate sample is examined according to five quality considerations: (1) visual sufficiency, where the answer must be determinable from the provided visual evidence; (2) annotation correctness, including the validity of reference answers and answer choices; (3) capability alignment, ensuring that the sample measures the intended reasoning or planning ability; (4) redundancy control, which removes highly correlated samples originating from similar scenes, videos, or viewpoints; and (5) other quality issues identified during manual review. The detailed statistics of removed samples are reported in Table 7.

For samples collected from videos or multi-view environments, redundancy filtering is performed at the scene level rather than only at the QA level. Samples sharing the same underlying scene, highly similar viewpoints, or overlapping temporal segments are grouped together. When multiple samples evaluate equivalent reasoning patterns, only the highest-quality instance is retained to avoid over-representing specific environments.

To examine whether filtering introduces unintended changes in benchmark composition, we compare the category distribution before and after filtering. As shown in Table 6, the retention ratios remain similar across Spatial Cognition, Temporal Reasoning, and Multi-Step Planning. The final benchmark maintains coverage across all three dimensions and 16 subtasks while removing lowquality or redundant samples.

Table 6: Category distribution before and after sample filtering.
<table><tr><td>Category</td><td>Before Filtering</td><td>Removed</td><td>Retained</td></tr><tr><td>Spatial Cognition</td><td>952</td><td>693</td><td>259</td></tr><tr><td>Temporal Reasoning</td><td>768</td><td>558</td><td>210</td></tr><tr><td>Multi-Step Planning</td><td>860</td><td>630</td><td>230</td></tr><tr><td>Total</td><td>2580</td><td>1881</td><td>699</td></tr></table>

Table 7: Statistics of sample filtering during Plan-Bench construction.
<table><tr><td>Filtering Reason</td><td>Removed Samples</td><td>Percentage</td></tr><tr><td>Insufficient visual evidence or ambiguous answers</td><td>637</td><td>33.9%</td></tr><tr><td>Incorrect annotation or reference answer</td><td>364</td><td>19.4%</td></tr><tr><td>Capability mismatch with target task</td><td>297</td><td>15.8%</td></tr><tr><td>Duplicate or highly correlated scenes</td><td>512</td><td>27.2%</td></tr><tr><td>Other quality issues</td><td>71</td><td>3.8%</td></tr><tr><td>Total removed</td><td>1881</td><td>72.91%</td></tr><tr><td>Final retained</td><td>699</td><td>27.09%</td></tr></table>

## A.4 DATA SOURCES

Plan-Bench combines diverse data sources and question formats across subtasks. As shown in Table 2, the 16 subtasks are constructed from a mixture of existing datasets, controlled generation tools, simulation environments, and web-sourced data. Different sources are selected according to the capability targeted by each subtask, ranging from temporal activities to spatial reasoning and multi-step planning. The benchmark also includes single-choice, open-ended, and mixed-format subtasks, providing diverse evaluation settings rather than relying on a uniform question format.

## B EVALUATION PROTOCOL

## B.1 MODEL PROMPT TEMPLATES

Since Plan-Bench contains both single-choice and open-ended questions, we implement a unified prompting function to support both question types within the same evaluation pipeline. As shown below, all samples share a common Base Prompt, which specifies the general response behavior and output requirements. We then append a question-type-specific instruction according to the sample format: the Single-Choice Prompt requires the model to return only one option letter, while the Open-Ended Prompt requires only the requested structured answer. This design provides a consistent prompting framework while preserving the output requirements of different question types.

## Base Prompt

Answer directly. Do not explain your reasoning. For single-choice questions, answer with one option letter only, such as A. For open questions, follow the required answer format exactly.

## Single-Choice Prompt

This is a single-choice question. Return one option letter only.

## Open-Ended Prompt

This is an open question. Return only the requested structured answer.

## B.2 LLM-AS-JUDGE PROMPT

We use DeepSeek-V4-Flash as an LLM-based judge to evaluate model responses. For each sample, the judge receives the model output, ground-truth answer, and an optional answer hint specifying additional valid alternatives or equivalence conditions. A response is labeled True when it is semantically equivalent to the ground-truth answer or satisfies an equivalence condition explicitly defined by the answer hint; otherwise, it is labeled False. The judge is instructed to evaluate only answer correctness and to return no additional explanation.

## System Prompt (Answer Equivalence Judge)

You are a precise answer-equivalence judge for evaluating model responses.

Your task is to determine whether a model output should be considered correct with respect to the ground-truth answer and the provided answer hint.

The ground-truth answer specifies the reference answer, while the answer hint provides additional valid alternatives or equivalence conditions that should be accepted.

Judge the model output according to the following rules:

1. Semantic Equivalence: Return True if the model output expresses the same answer as the ground-truth answer, even if the wording or formatting differs.

2. Answer-Hint Equivalence: Return True if the model output satisfies any valid alternative or equivalence condition explicitly specified in the answer hint, including acceptable order swaps, position exchanges, alternative object names, or equivalent action descriptions.

3. Partial Correctness: Return False if the response is only partially correct, omits required information, or contains an incorrect component.

4. Contradictory or Additional Content: Return False if the response contains information that contradicts the ground-truth answer or violates the conditions specified by the answer hint.

5. Formatting Differences: Ignore superficial differences in capitalization, punctuation,   
spacing, or other formatting when they do not change the meaning of the answer.   
Do not introduce alternative interpretations beyond those supported by the ground-truth answer   
and answer hint. Do not infer missing information. Do not evaluate the reasoning process;   
evaluate only the final answer.   
Return only one of the following:   
True   
or   
False   
Do not provide any explanation or additional text.   
User Prompt (Answer Verification)   
Determine whether the model output is equivalent to the ground-truth answer under the con  
ditions specified by the answer hint.   
Ground-truth answer:   
{ground truth}   
Answer hint:   
{answer hint}   
Model output:   
{model output}   
Return only True or False.

## B.3 ANSWER HINT AND EQUIVALENCE RULES

Answer hints handle multiple valid answers in open-ended questions. Among the open-ended questions in Plan-Bench, 486 samples require answer hints across 13 subcategories, including Occlusion Order, Mirror Distinction, State Recognition, Rearrangement Planning, and Dependency Planning. These hints specify valid equivalence conditions beyond the reference answer, such as interchangeable object references, flexible orderings, position exchanges, and equivalent action sequences. As illustrated in Figure 10, the answer hint is provided only to the LLM judge, allowing semantically equivalent predictions to be accepted without exposing additional information to the evaluated model.

## B.4 HUMAN VERIFICATION OF THE JUDGE

To further assess the reliability of our LLM-based evaluation, we conduct a human verification study on 200 responses generated by Qwen3.8-Max, Gemma-4-31B-it, and Step3-VL-10B. We adopt stratified sampling to ensure diverse and balanced coverage across the three evaluation dimensions and their corresponding subclasses, selecting 67 samples from Spatial Cognition, 67 from Temporal Reasoning, and 66 from Multi-Step Planning. Among the 200 sampled instances, 133 contain answer hints specifying acceptable alternatives or equivalence conditions, such as alterna tive object names, interchangeable orderings, or equivalent action sequences, while the remaining 67 have no answer hints. All 200 responses are independently evaluated by human annotators and DeepSeek-V4-Flash based on the same ground-truth answers and answer hints. The two evaluations agree on all 200 samples, achieving 100% agreement, including the 133 cases involving answer hints. These results demonstrate that our LLM-based judge can reliably account for equivalent valid answers and provide consistent evaluation for Plan-Bench.

## B.5 MODEL CONFIGURATIONS

We evaluate open-weight VLMs through VLMEvalKit using their corresponding model wrappers and checkpoints. Unless specified below, we retain the decoding and image-processing settings defined by each wrapper. Open-weight models are evaluated on NVIDIA A800 GPUs in BF16 precision. All models receive the prompt templates described in Appendix B.1, while proprietary models are evaluated through API interfaces under the same question and answer protocol.

![](images/3ca6b519256fbac64999c6935ec90bc87dd9b6969ba938fdb7c285108a78782d.jpg)  
Figure 10: Examples of answer hints for multiple valid answers.

Qwen series. Qwen3-VL-2B-Instruct, Qwen3-VL-8B-Instruct, and Qwen3-VL-30B-A3B-Instruct use the Qwen3VLChat wrapper with custom prompts disabled. Their configurations set temperature to 0.7, top-p to 0.8, top-k to 20, presence penalty to 1.5, and the maximum number of new tokens to 16,384. We use the same wrapper for Qwen3.5-2B, 4B, 9B, 27B, and 35B-A3B, as well as Qwen3.8-27B, with temperature 1.0, top-p 0.95, top-k 20, and presence penalty 1.5. For these Qwen3.5 and Qwen3.8 models, the initial generation limit is 16,384 tokens. After the initial generation, we instruct each model to summarize its generated response and present the final answer within an additional 512-token budget. This additional budget is used only for answer condensation, extraction, and formatting based on the already generated content, rather than for further substantive reasoning or revising the underlying answer. Where supported, Thinking and Non-Thinking evaluations use the same checkpoint with the corresponding mode selected in the wrapper.

Gemma-4 series. Gemma-4-12B-it and Gemma-4-31B-it use the Gemma4 wrapper with the corresponding model processor and chat template. The wrapper uses greedy decoding with a maximum of 16,384 new tokens. For Thinking and Non-Thinking evaluations, we set the chat template’s enable thinking option accordingly and extract the final answer from the generated response.

InternVL3.5 series. InternVL3.5-2B, InternVL3.5-8B, and InternVL3.5-30B-A3B use the Intern-VLChat V2.0 wrapper with the vLLM backend enabled. We retain the wrapper’s default image processing and generation settings for these models.

Step3-VL series. Step3-VL-10B uses a local vLLM inference wrapper with greedy decoding, a maximum of 16,384 new tokens, and support for up to 24 images per prompt. Its DeepSeek-R1- style parser separates the reasoning trace from the final answer; only the final answer is submitted for evaluation.

GLM-4.6V series. GLM-4.6V-Flash uses a local vLLM wrapper with Thinking enabled. Its configuration sets temperature to 0.8, top-p to 0.6, top-k to 2, repetition penalty to 1.1, and the maximum number of new tokens to 16,384.

Ovis series. Ovis2-8B and Ovis2.5-9B are evaluated using their corresponding VLMEvalKit wrappers. We retain their wrapper-defined image preprocessing and generation settings.

MiMo-VL series. MiMo-VL-7B-SFT uses the Qwen2VLChat wrapper with custom prompts disabled and vLLM inference. Its image-size bounds are set to 1280 × 28 × 28 and 16384 × 28 × 28 pixels; the remaining generation settings follow the wrapper configuration.

NEO series. NEO-2B-SFT and NEO-9B-SFT use the NEOChat wrapper with a patch size of 16, a downsampling ratio of 0.5, image-size bounds of 1280 × 32 × 32 and 4096 × 32 × 32 pixels, and the maximum number of new tokens to 16,384. We retain the wrapper’s default generation settings.

NEO-ov is evaluated under the same benchmark protocol, although its model-specific configuration is not included in the supplied files.

SenseNova series. SenseNova-U1 is evaluated under the common Plan-Bench prompting and answer protocol, and the maximum number of new tokens to 16,384. Its model-specific inference settings are not specified in the supplied configuration.

Qwen3.8-max. Qwen3.8-max is evaluated through an API interface with a maximum response length of 16,384 tokens. The supplied configuration does not specify a temperature override.

GPT-5.6-sol. GPT-5.6-sol is evaluated through an API interface with a maximum response length of 16,384 tokens. The supplied configuration does not specify a temperature override.

Gemini-3.1-Pro-Preview. Gemini-3.1-Pro-Preview is evaluated through an API interface with temperature set to 0 and a maximum response length of 16,384 tokens.

Claude-Opus-4.8. Claude-Opus-4.8 is evaluated through an API interface with temperature set to 0 and a maximum response length of 16,384 tokens.

Kimi-K3. Kimi-K3 is evaluated under the common Plan-Bench prompting and answer protocol. Its model-specific API settings are not specified in the supplied configuration files.

## C ADDITIONAL RESULTS

## C.1 RESULTS ON ALL 16 SUBTASKS

Tables 8–10 report model accuracy on all 16 subtasks of Plan-Bench, revealing differences that are hidden by the three dimension-level scores. In Spatial Cognition, GPT-5.6-sol achieves 82.9% on Mirror Distinction and 87.5% on State Recognition, but only 30.0% on Target Discovery and 31.3% on Spatial Relations. Gemini-3.1-Pro-Preview shows a contrasting profile, reaching 75.0% on Spa tial Relations but only 22.0% on Occlusion Order. Qwen3.5-27B also varies sharply across the category, with 56.8% on Spatial Relations but 18.0% on Occlusion Order. These contrasts indicate that success on one form of spatial evidence does not reliably extend to the other visual reasoning demands within the same dimension, even when the tasks draw on similar visual scenes and viewpoints that expose different but complementary scene cues about object locations.

Temporal Reasoning likewise exhibits substantial variation between subtasks. Qwen3.5-27B achieves 96.7% on Progress Tracking and 80.0% on Temporal Ordering, but only 28.0% on Persistent Counting. GPT-5.6-sol follows a similar pattern, scoring 93.3% on Progress Tracking and 30.0% on Persistent Counting. Even Qwen3.5-35B-A3B, which reaches 40.0% on Persistent Counting, performs much better on Progress Tracking at 86.7%. These gaps suggest that inferring task progress and event order is easier for these models than maintaining accurate counts when objects or actions recur across visual sequences and must be distinguished from previous observations or distractors that appear between relevant activity steps in multi-frame observations.

In Multi-Step Planning, strong results on one type of constraint do not consistently transfer to other planning problems. Gemini-3.1-Pro-Preview scores 93.3% on Capacity Planning but only 10.0% on Rearrangement Planning. Similarly, Qwen3.5-27B scores 90.0% on Capacity Planning but 44.0% on Route Planning. Rearrangement Planning is particularly difficult for several smaller models: InternVL3.5-8B and NEO-9B-SFT both score 0.0% despite obtaining 30.0% and 53.3% on Capacity Planning, respectively. Thus, the detailed results expose task-specific weaknesses in spatial reasoning, temporal tracking, and multi-step planning that aggregate accuracy cannot capture.

## C.2 ADDITIONAL ERROR ANALYSIS

Figure 11 presents two additional Target Discovery failures by Qwen3.8-Max. In both cases, the target appears in a mirror, so selecting the correct action requires the model to associate the reflection with a real object and infer that object’s location. In the first example, the model notices the similar ity between the reflected picture and the directly visible picture, but speculates that another identical picture exists elsewhere. It therefore selects an action that leads away from the actual target. In the second example, the model likewise treats the reflected and visible paper towel dispensers as two separate objects, incorrectly concluding that the target is behind the camera.

These examples show how hallucination can arise when the model fails to establish the correspondence between a reflection and a physical object. Although it recognizes their visual similarity, it hallucinates a second, identical object and reasons about its nonexistent location. This mistaken interpretation then leads to an incorrect navigation decision, showing how object hallucination can propagate from scene understanding to action planning in Target Discovery.

Table 8: Performance of different VLMs on the six Spatial Cognition subtasks of Plan-Bench.
<table><tr><td>Model</td><td>00</td><td>MD</td><td>AR</td><td>TD</td><td>SR</td><td>SpR</td></tr><tr><td colspan="7">▼ Commercial Models</td></tr><tr><td>GPT-5.6-sol</td><td>58.0</td><td>82.9</td><td>54.0</td><td>30.0</td><td>87.5</td><td>31.3</td></tr><tr><td>Claude-Opus-4.8</td><td>64.0</td><td>74.3</td><td>28.0</td><td>27.5</td><td>42.5</td><td>22.7</td></tr><tr><td>Gemini-3.1-Pro-Preview</td><td>22.0</td><td>45.7</td><td>28.0</td><td>30.0</td><td>42.5</td><td>75.0</td></tr><tr><td colspan="7">▼ Large Open-Weight Models</td></tr><tr><td>Qwen3.5-27B</td><td>18.0</td><td>54.3</td><td>36.0</td><td>27.5</td><td>47.5</td><td>56.8</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>12.0</td><td>51.4</td><td>20.0</td><td>35.0</td><td>50.0</td><td>50.0</td></tr><tr><td>Qwen3-VL-30B-A3B</td><td>18.0</td><td>42.9</td><td>24.0</td><td>7.5</td><td>25.0</td><td>20.5</td></tr><tr><td>InternVL3.5-30B-A3B</td><td>6.0</td><td>2.9</td><td>24.0</td><td>10.0</td><td>20.0</td><td>22.7</td></tr><tr><td colspan="7">▼ Medium Open-Weight Models</td></tr><tr><td>Qwen3.5-9B</td><td>6.0</td><td>45.7</td><td>18.0</td><td>30.0</td><td>35.0</td><td>52.3</td></tr><tr><td>Gemma-4-12B-it</td><td>10.0 22.0</td><td>37.1</td><td>24.0</td><td>25.0</td><td>22.5</td><td>34.1</td></tr><tr><td>InternVL3.5-8B</td><td>18.0</td><td>14.3</td><td>20.0</td><td>15.0</td><td>15.0</td><td>27.3</td></tr><tr><td>NEO-9B-SFT</td><td></td><td>0.0</td><td>30.0</td><td>20.0</td><td>22.5</td><td>43.2</td></tr><tr><td>MiMo-VL-7B-SFT</td><td>14.0</td><td>40.0</td><td>14.0</td><td>15.0</td><td>17.5</td><td>22.7</td></tr><tr><td>SenseNova-U1</td><td>8.0</td><td>8.6</td><td>0.0</td><td>20.0</td><td>45.5</td><td>30.0</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>24.0</td><td>11.4</td><td>8.0</td><td>22.5</td><td>20.0</td><td>20.5</td></tr><tr><td>NEO-ov</td><td>24.0</td><td>0.0</td><td>16.0</td><td>12.5</td><td>17.5</td><td>36.4</td></tr><tr><td>Ovis2-8B</td><td>8.0</td><td>0.0</td><td>14.0</td><td>20.0</td><td>20.0</td><td>27.3</td></tr><tr><td colspan="7">▼ Small Open-Weight Models</td></tr><tr><td>Qwen3.5-4B</td><td>6.0</td><td>40.0</td><td>20.0</td><td>17.5</td><td>40.0</td><td>45.5</td></tr><tr><td>Qwen3-VL-2B-Instruct</td><td>16.0</td><td>2.9</td><td>16.0</td><td>12.5</td><td>15.0</td><td>20.5</td></tr><tr><td>InternVL3.5-2B</td><td>18.0</td><td>0.0</td><td>16.0</td><td>15.0</td><td>10.0</td><td>20.5</td></tr><tr><td>NEO-2B-SFT</td><td>10.0</td><td>2.9</td><td>10.0</td><td>10.0</td><td>12.5</td><td>15.9</td></tr></table>

Table 9: Performance of different VLMs on the five Temporal Reasoning subtasks of Plan-Bench.
<table><tr><td rowspan=1 colspan=6>Model</td><td rowspan=1 colspan=1>PT</td><td rowspan=1 colspan=1>TO</td><td rowspan=1 colspan=1>PC</td><td rowspan=1 colspan=1>IA</td><td rowspan=1 colspan=1>MR</td></tr><tr><td rowspan=1 colspan=11>▼ Commercial Models</td></tr><tr><td rowspan=1 colspan=6>GPT-5.6-solClaude-Opus-4.8Gemini-3.1-Pro-Preview</td><td rowspan=1 colspan=1>93.393.386.7</td><td rowspan=1 colspan=1>75.050.067.5</td><td rowspan=1 colspan=1>30.034.034.0</td><td rowspan=1 colspan=1>80.040.047.5</td><td rowspan=1 colspan=1>82.044.062.0</td></tr><tr><td rowspan=1 colspan=6>▼ Large Open-Weight Models</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=3 colspan=6>Qwen3.5-27BQwen3.5-35B-A3BQwen3-VL-30B-A3BInternVL3.5-30B-A3B</td><td rowspan=2 colspan=1>96.786.740.0</td><td rowspan=1 colspan=1>80.0</td><td rowspan=1 colspan=1>28.0</td><td rowspan=1 colspan=1>72.5</td><td rowspan=1 colspan=1>68.0</td></tr><tr><td rowspan=1 colspan=1>80.010.0</td><td rowspan=1 colspan=1>40.010.0</td><td rowspan=2 colspan=1>55.017.520.0</td><td rowspan=2 colspan=1>62.036.042.0</td></tr><tr><td rowspan=1 colspan=1>33.3</td><td rowspan=1 colspan=1>22.5</td><td rowspan=1 colspan=1>10.0</td></tr><tr><td rowspan=1 colspan=11>▼ Medium Open-Weight Models</td></tr><tr><td rowspan=1 colspan=6>Qwen3.5-9B</td><td rowspan=1 colspan=1>93.3</td><td rowspan=1 colspan=1>72.5</td><td rowspan=1 colspan=1>28.0</td><td rowspan=1 colspan=1>62.5</td><td rowspan=1 colspan=1>58.0</td></tr><tr><td rowspan=1 colspan=3>Gemma-4-12B-it</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>80.0</td><td rowspan=1 colspan=1>7.5</td><td rowspan=1 colspan=1>14.0</td><td rowspan=1 colspan=1>22.5</td><td rowspan=1 colspan=1>48.0</td></tr><tr><td rowspan=5 colspan=6>InternVL3.5-8BNEO-9B-SFTMiMo-VL-7B-SFTSenseNova-U1Qwen3-VL-8B-Instruct</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>26.7</td><td rowspan=1 colspan=1>25.0</td><td rowspan=1 colspan=1>24.0</td></tr><tr><td rowspan=1 colspan=1>10.0</td><td rowspan=1 colspan=1>15.0</td><td rowspan=1 colspan=1>20.0</td><td rowspan=1 colspan=1>7.5</td><td rowspan=1 colspan=1>36.0</td></tr><tr><td rowspan=1 colspan=1>13.3</td><td rowspan=1 colspan=1>22.5</td><td rowspan=1 colspan=1>12.0</td><td rowspan=1 colspan=1>17.5</td><td rowspan=1 colspan=1>38.0</td></tr><tr><td rowspan=1 colspan=1>30.0</td><td rowspan=1 colspan=1>7.5</td><td rowspan=1 colspan=1>8.0</td><td rowspan=1 colspan=1>7.5</td><td rowspan=1 colspan=1>26.0</td></tr><tr><td rowspan=1 colspan=1>20.0</td><td rowspan=1 colspan=1>20.0</td><td rowspan=1 colspan=1>8.0</td><td rowspan=1 colspan=1>15.0</td><td rowspan=1 colspan=1>34.0</td></tr><tr><td rowspan=1 colspan=6>NEO-ov</td><td rowspan=1 colspan=1>13.3</td><td rowspan=1 colspan=1>12.5</td><td rowspan=1 colspan=1>14.0</td><td rowspan=1 colspan=1>17.5</td><td rowspan=1 colspan=1>28.0</td></tr><tr><td rowspan=1 colspan=6>Ovis2-8B</td><td rowspan=1 colspan=1>30.0</td><td rowspan=1 colspan=1>15.0</td><td rowspan=1 colspan=1>14.0</td><td rowspan=1 colspan=1>12.5</td><td rowspan=1 colspan=1>26.0</td></tr><tr><td rowspan=1 colspan=6>▼ Small Open-Weight Models</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=6>Qwen3.5-4BQwen3-VL-2B-Instruct</td><td rowspan=1 colspan=1>86.7</td><td rowspan=1 colspan=1>67.5</td><td rowspan=1 colspan=1>24.0</td><td rowspan=1 colspan=1>40.0</td><td rowspan=1 colspan=1>58.0</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>13.3</td><td rowspan=1 colspan=1>15.0</td><td rowspan=1 colspan=1>14.0</td><td rowspan=1 colspan=1>10.0</td><td rowspan=1 colspan=1>30.0</td></tr><tr><td rowspan=2 colspan=6>InternVL3.5-2BNEO-2B-SFT</td><td rowspan=1 colspan=1>10.0</td><td rowspan=1 colspan=1>17.5</td><td rowspan=1 colspan=1>18.0</td><td rowspan=1 colspan=1>15.0</td><td rowspan=1 colspan=1>28.0</td></tr><tr><td rowspan=1 colspan=1>10.0</td><td rowspan=1 colspan=1>15.0</td><td rowspan=1 colspan=1>2.0</td><td rowspan=1 colspan=1>2.5</td><td rowspan=1 colspan=1>8.0</td></tr></table>

Table 10: Performance of different VLMs on the five Multi-Step Planning subtasks of Plan-Bench.
<table><tr><td>Model</td><td>CS</td><td>CP</td><td>RP</td><td>DP</td><td>Route</td></tr><tr><td colspan="6">▼ Commercial Models</td></tr><tr><td>GPT-5.6-sol</td><td>52.0</td><td>86.7</td><td>62.0</td><td>86.0</td><td>84.0</td></tr><tr><td>Claude-Opus-4.8</td><td>70.0</td><td>90.0</td><td>34.0</td><td>68.0</td><td>66.0</td></tr><tr><td>Gemini-3.1-Pro-Preview</td><td>42.0</td><td>93.3</td><td>10.0</td><td>56.0</td><td>40.0</td></tr><tr><td colspan="6">▼ Large Open-Weight Models</td></tr><tr><td>Qwen3.5-27B</td><td>82.0</td><td>90.0</td><td>62.0</td><td>62.0</td><td>44.0</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>62.0</td><td>86.7</td><td>54.0</td><td>52.0</td><td>30.0</td></tr><tr><td>Qwen3-VL-30B-A3B</td><td>24.0</td><td>33.3</td><td>10.0</td><td>18.0</td><td>0.0</td></tr><tr><td>InternVL3.5-30B-A3B</td><td>26.0</td><td>36.7</td><td>0.0</td><td>6.0</td><td>10.0</td></tr><tr><td colspan="6">▼ Medium Open-Weight Models</td></tr><tr><td>Qwen3.5-9B</td><td>72.0 38.0</td><td>86.7 26.7</td><td>50.0 6.0</td><td>70.0 54.0</td><td>40.0</td></tr><tr><td>Gemma-4-12B-it InternVL3.5-8B</td><td>22.0</td><td>30.0</td><td>0.0</td><td>10.0</td><td>22.0 6.0</td></tr><tr><td>NEO-9B-SFT</td><td>10.0</td><td>53.3</td><td>0.0</td><td>8.0</td><td>8.0</td></tr><tr><td>MiMo-VL-7B-SFT</td><td>24.0</td><td>26.7</td><td>2.0</td><td>4.0</td><td>18.0</td></tr><tr><td>SenseNova-U1</td><td>24.0</td><td>30.0</td><td>0.0</td><td>36.0</td><td>18.0</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>26.0</td><td>36.7</td><td>8.0</td><td>8.0</td><td>12.0</td></tr><tr><td>NEO-ov</td><td>14.0</td><td>50.0</td><td>0.0</td><td></td><td></td></tr><tr><td>Ovis2-8B</td><td>16.0</td><td>50.0</td><td></td><td>4.0 4.0</td><td>10.0</td></tr><tr><td></td><td></td><td></td><td>0.0</td><td></td><td>10.0</td></tr><tr><td colspan="6">▼ Small Open-Weight Models</td></tr><tr><td>Qwen3.5-4B</td><td>72.0</td><td>80.0</td><td>28.0</td><td>56.0</td><td>40.0</td></tr><tr><td>Qwen3-VL-2B-Instruct</td><td>10.0</td><td>30.0</td><td>0.0</td><td>2.0</td><td>8.0</td></tr><tr><td>InternVL3.5-2B</td><td>8.0</td><td>20.0</td><td>0.0</td><td>2.0</td><td>4.0</td></tr><tr><td>NEO-2B-SFT</td><td>2.0</td><td>30.0</td><td>0.0</td><td>8.0</td><td>4.0</td></tr></table>

![](images/2c2c8e3cbb4a15f4b74929938c27af1e4b72c711a38ab2b5847876ac22bc5738.jpg)  
Figure 11: Additional Target Discovery failures by Qwen3.8-Max. The model treats reflected targets as separate physical objects, leading to incorrect navigation decisions.