# Reviso: Not All Visual Evidence Deserves Equal Attention in Video Captioning

Tengfei Liu<sup>1</sup> Yang Shi<sup>1,2,†,♠</sup> Yuran Wang<sup>1</sup> Yuqi Tang<sup>3</sup> Yizhuo Jia<sup>4</sup> Zhuoran Zhang<sup>1</sup> Qixun Wang<sup>1</sup> Shujie Li<sup>5</sup> Xuanyu Zhu<sup>1</sup> Xinyu Liu<sup>6</sup> Minghui Guo<sup>7</sup> Shixuan Liu<sup>8,†</sup> Haotian Wang<sup>8,†</sup> Yuanxing Zhang<sup>2,♠</sup> Pengfei Wan<sup>2</sup>

<sup>1</sup>PKU <sup>2</sup>KlingAI Research <sup>3</sup>HKUST(GZ) <sup>4</sup>FDU <sup>5</sup>HKU <sup>6</sup>HKUST <sup>7</sup>NUS <sup>8</sup>THU

## Abstract

Multimodal large language models (MLLMs) can generate detailed video captions, yet they may overemphasize minor details while overlooking essential context and key events. We argue that not all visual evidence deserves equal attention, as the importance of visual information depends on the content of each video. Motivated by this observation, we propose Reviso, an agentic video captioning framework that learns what to observe and how to gather relevant visual evidence. From a low-frame rate global preview, Reviso constructs a hierarchical Context–Focus–Points (CFP) observation plan that identifies the video context, prioritizes key aspects, and defines concrete observation targets and corresponding tools. Guided by this plan, Reviso performs goal-directed active perception by selectively revisiting temporal segments and combining tools for fine-grained inspection. It interprets the acquired evidence, updates its plan as needed, and integrates global and local information to generate the final caption. To learn this observation strategy, we develop a verification-guided pipeline that constructs high-quality training trajectories through factual verification and teacher-guided refinement. We further train Reviso using Group Relative Policy Optimization (GRPO), incorporating a key-fact reward for caption quality, a Plan Alignment Reward (PAR) for identifying important observation targets, and an Unproductive Tool-Call Penalty (UTCP) that adjusts observation-cost penalties based on caption quality. Experiments across four video captioning benchmarks show that Reviso outperforms the evaluated open-source baselines, while its generated captions better support downstream video question answering. These findings highlight the efectiveness and potential of priority-guided active observation for video captioning.

Date: November 3, 2026

Code

## 1 Introduction

Video captioning aims to capture and convey visual information from videos, playing an important role in video understanding and generation [9, 15, 16, 30, 31, 38, 47, 52, 56]. Recent advances in multimodal large language models (MLLMs) [12, 21, 23, 24, 29, 33, 34, 53] have enabled increasingly detailed video captions [4, 28, 45]. However, greater descriptive detail does not necessarily ensure coverage of key information. Existing models may overemphasize minor details while overlooking essential context and key events, particularly in videos involving complex action sequences, subtle interactions, or fine-grained visual details [22, 32].

![](images/fd2f3dfd9dc74ee77df1d44c917167d5e712b6d50b042c259e11dc65c9746c7b.jpg)  
Figure 1 Video captioning through active observation. Captioning priorities vary across video types, including action order in tutorials, characters’ expressions and actions in movie scenes, and product names and attributes in advertisements. Reviso identifies what matters in each video and selects tools to gather evidence for its caption.

We argue that not all visual evidence deserves equal attention. As illustrated in Figure 1, the importance of visual information depends on the content and context of each video. Moreover, important information may not be readily resolved from a coarse visual overview and may require targeted inspection [22, 41]. Learning such an observation strategy presents two main challenges. (1) Identifying what to observe. The model must determine which information is important for understanding the video and translate these priorities into concrete observation targets. (2) Learning how to observe. Diferent observation targets may require diferent tools for fine-grained visual inspection [8, 19, 41]. The model must select and combine appropriate tools, interpret the acquired evidence, and adapt subsequent observations accordingly.

To address these challenges, we propose Reviso, an agentic video captioning framework that learns to identify important visual information and actively gather supporting evidence. As shown in Figure 2, Reviso first constructs a hierarchical Context-Focus-Points (CFP) observation plan from a low-frame-rate global preview. The Context level identifies the overall video setting, Focus prioritizes important aspects, and Points define concrete observation targets and the tools needed to inspect them. Guided by this plan, Reviso performs goal-directed active perception by selectively revisiting temporal segments at higher frame rates and using tools such as image cropping and OCR for fine-grained inspection. Based on the acquired evidence, it determines which targets have been resolved and updates its plan to guide subsequent observations. Finally, Reviso integrates global context and local evidence to generate the caption.

To learn this observation strategy, we develop a verification-guided pipeline that constructs high-quality agentic trajectories through factual verification and teacher-guided refinement. These trajectories provide supervised fine-tuning data for learning observation planning, tool use, and evidence-grounded caption generation. We further train Reviso using GRPO [27], incorporating a Key-Fact Reward (KFR) for caption quality, a Plan Alignment Reward (PAR) for identifying important observation targets, and an Unproductive Tool-Call Penalty (UTCP) to regulate observation costs based on caption quality. Extensive experiments across four video captioning benchmarks show that Reviso consistently improves over its backbone and outperforms the strong open-source baselines. Further evaluations demonstrate that Reviso generates captions that better support downstream video understanding, highlighting the efectiveness of learning to prioritize and actively gather relevant visual evidence.

Our contributions are summarized as follows:

• We introduce a priority-aware perspective on video captioning, arguing that not all visual evidence deserves equal attention. This perspective emphasizes identifying video-specific observation priorities and actively gathering evidence for key information, rather than treating all visual content equally.

• We propose Reviso, an agentic video captioning framework that realizes this perspective through hierarchical Context-Focus-Points (CFP) planning and goal-directed active perception. We further develop a verification-guided trajectory construction pipeline and tailored GRPO rewards to learn efective observation planning and tool use.

• Extensive experiments across four video captioning benchmarks demonstrate that Reviso improves caption quality over strong open-source baselines. Further evaluations show that its generated captions better support downstream video understanding.

## 2 Related Work

Video Captioning. Recent video captioning methods [4, 15, 40, 44] have explored various approaches to capturing fine-grained visual information and temporal dynamics. Tarsier [32]improves detailed video description through large-scale pretraining and instruction tuning. TimeChat-Captioner [48] generates timestamped, structured audio-visual descriptions, while PercepCap [43] produces a structured record of object trajectories and temporal events before generating the caption. Reinforcement learning further improves caption quality. CapRL++ [46] rewards captions based on whether a text-only model can correctly answer video-related questions using only those captions, while CuRe [10] uses claim-level verification and referenceanchored calibration to reward caption’s factual accuracy and coverage. For long videos, PIGEONS [6] maintains a persistent identity gallery and an evolving narrative summary to preserve subject consistency across clips. These approaches primarily improve caption generation from available visual representations, without explicitly learning to identify video-specific observation targets and acquire additional evidence to resolve them. In contrast, Reviso learns to prioritize important information and actively gather supporting visual evidence through goal-directed observation.

Agentic Video Understanding. Video agents combine reasoning with iterative observation to acquire relevant visual evidence. VideoAgent [37] retrieves additional frames, while VideoTree [39] constructs queryadaptive hierarchical video representations. Recent methods further explore learned observation policies. VITAL [51], VideoZoomer [8], and VideoTemp-o3 [18] selectively revisit temporal segments, while EVA [55] learns to adjust observation intervals and sampling configurations. Other approaches explore multi-agent collaboration [5] and integrate video inspection with external search [17]. Many of these methods focus on query-driven video understanding, where observation is guided by a given question or instruction. In contrast, open-ended video captioning requires models to identify important information and determine what evidence to gather without an explicit question. Omni-Detective [19] and OmniFysics-Agent [22] employ agents to collect evidence for caption generation, but their final captioners learn primarily from the resulting descriptions rather than the observation trajectories. Reviso instead learns video-specific observation planning and tool-assisted evidence gathering through verified trajectory supervision and reinforcement learning.

## 3 Method

## 3.1 Priority-Guided Active Observation

Global preview and planning. As shown in Figure 2(a), Reviso first samples timestamped frames at a low frame rate to obtain a global overview of the video. Based on this preview, it constructs a hierarchical Context–Focus–Points (CFP) observation plan that identifies video-specific priorities and translates them into concrete observation targets and corresponding tool actions.

• Context: Reviso identifies the overall context of the video, including its setting, main activities, and relevant background information. This context provides the basis for assessing the importance of diferent visual information.

• Focus: Based on the inferred context, Reviso prioritizes the aspects most relevant to understanding the video. As illustrated in Figure 3, observation priorities vary with the content and context of each video.

(a) Priority-Guided Active Observation  
![](images/93bb99a71543256f31a51bb39fd30bd17d3e53d8b892ddd0f2997a0ae1cb8eb8.jpg)

![](images/1293ce3e29256d37229451ac82cd34f45c2254ed1f54a3367ead96fec30752c2.jpg)  
Figure 2 Overview of Reviso. (a) A hierarchical Context–Focus–Points plan guides tool-assisted observation and caption generation. (b) Verified teacher trajectories provide supervised initialization. (c) GRPO rewards factual caption quality (KFR) and plan alignment (PAR), while penalizing unproductive tool use (UTCP).

These priorities determine which information deserves closer inspection and guide the formulation of concrete observation targets at the Points level.

• Points: Reviso translates each focus into concrete targets to inspect or verify and specifies the corresponding observation tools. These targets guide the selection of relevant temporal segments, observation granularity, and tool actions, such as high-frame-rate revisiting, image cropping, and OCR.

Agentic fine-grained observation. Guided by the CFP plan, Reviso performs multiple rounds of toolassisted observation to gather evidence for prioritized information. At each round, it selects observation targets based on the current plan and invokes appropriate tools according to the evidence needed. The resulting observations are interpreted together with the global context and previously acquired evidence. Reviso then identifies resolved and unresolved targets and updates its plan to direct further inspection toward important information that remains insuficiently observed. Finally, it integrates the global context with the collected local evidence to generate the final video caption.

## 3.2 Training Strategy

## 3.2.1 Supervised Fine-Tuning

We initialize Reviso from Qwen3.5-9B [23] and fine-tune it on verified teacher trajectories, as shown in Figure 2(b). Token-level supervision covers observation plans, tool calls, local interpretations, plan updates, and the final caption. Environment responses provide context and are masked from the loss. This stage initializes the policy for the following stage.

## 3.2.2 GRPO for Priority-Guided Observation

Following supervised fine-tuning, we further train Reviso using GRPO [27] with three complementary reward components: Key-Fact Reward (KFR) for caption quality, Plan Alignment Reward (PAR) for observation planning, and Unproductive Tool-Call Penalty (UTCP) for regulating observation costs.

![](images/67e5cbb59fc861af4dba1e242a608e13001f447f11a782e83cf201b9ecc87770.jpg)  
Figure 3 Observation priorities across diverse video content. We illustrate representative visual aspects that may require prioritized attention in diferent video scenarios. Reviso identifies important information based on the content and context of each video, guiding targeted observation and evidence gathering.

Key-Fact Reward (KFR). For a video V, let τ denote a generated observation trajectory and c its final caption. We decompose the reference caption and generated caption into deduplicated fact sets $\mathcal { F } _ { V }$ and $\widehat { \mathcal { F } } ( c )$ respectively. KFR is defined as

$$
R _ { \mathrm { K F R } } ( \tau ) = \mathrm { F } _ { 1 } \big ( \widehat { \mathcal { F } } ( c ) , \mathcal { F } _ { V } \big ) .\tag{1}
$$

We compute F1 using precision and recall based on one-to-one semantic matching between the two fact sets. Precision measures the proportion of generated facts matched to the reference, while recall measures the coverage of reference facts. KFR thus encourages the preservation of key information while penalizing unmatched claims.

Plan Alignment Reward (PAR). PAR evaluates the initial observation plan p<sub>0</sub> before tool execution. Let $\mathcal { P } ( p _ { 0 } )$ denote the set of deduplicated observation targets extracted from p<sub>0</sub>, interpreted within their associated Context and Focus levels. Let U<sub>V</sub> denote the reference observation targets corresponding to key information not resolved by the global preview but identifiable as requiring further inspection. We measure alignment between predicted and reference targets using

$$
R _ { \mathrm { P A R } } ( \tau ) = \mathrm { F } _ { 1 } \big ( \mathcal { P } ( p _ { 0 } ) , \mathcal { U } _ { V } \big ) .\tag{2}
$$

As in KFR, we use one-to-one semantic matching to compute precision and recall. PAR encourages the initial plan to identify relevant observation targets while limiting unmatched targets.

Unproductive Tool-Call Penalty (UTCP). UTCP discourages costly observations that yield poor captions. For each video, we activate the penalty only when at least one trajectory in the sampled group reaches a fixed KFR threshold. If no trajectory reaches this threshold, we omit the cost penalty to preserve exploration. Once activated, the penalty applies to every trajectory in the group according to its own observation cost and caption quality. At the same cost, lower-quality captions receive a larger penalty. At the same quality, more costly observations receive a larger penalty. For a group $\mathcal { G }$ of trajectories sampled for the same video, we define

![](images/c640ea31b8466f0ce34b9c25810b8bbe0ba290e37e9136bf00a831cc8633f421.jpg)  
Figure 4 Overview of the trajectory construction pipeline. There are six steps in the pipeline. We generate observation trajectories, verify them against ground-truth captions, and use a teacher model to refine unsuccessful attempts, retaining verified trajectories for supervised fine-tuning

$$
\begin{array} { r } { g ( \mathcal { G } ) = \mathbb { I } \left[ \underset { \tau _ { i } \in \mathcal { G } } { \operatorname* { m a x } } R _ { \mathrm { K F R } } ( \tau _ { i } ) \geq \tau _ { q } \right] , } \\ { P _ { \mathrm { U T C P } } ( \tau _ { i } ; \mathcal { G } ) = g ( \mathcal { G } ) \big ( 1 - R _ { \mathrm { K F R } } ( \tau _ { i } ) \big ) b _ { i } , } \end{array}\tag{3}
$$

where $\tau _ { q }$ is the fixed KFR threshold. We quantify additional observation cost by the number of tool calls after the shared global preview. We normalize this count by a fixed call budget to obtain $b _ { i } \in [ 0 , 1 ]$ . The indicator $g ( \mathcal { G } )$ determines whether the cost penalty is active for the group.

Policy optimization. For each video, Reviso generates a group of trajectories containing observation plans, tool calls, and final captions. We evaluate each trajectory using the combined reward

$$
R ( \tau _ { i } ; \mathcal { G } ) = \alpha R _ { \mathrm { K F R } } ( \tau _ { i } ) + \beta R _ { \mathrm { P A R } } ( \tau _ { i } ) - \mu P _ { \mathrm { U T C P } } ( \tau _ { i } ; \mathcal { G } ) .\tag{4}
$$

where $\alpha , \beta ,$ and $\mu$ control the contributions of the three components. GRPO computes relative advantages from the rewards within each group and uses them to update the model.

## 3.3 Data Construction

As shown in Figure 4, we construct verified training trajectories through six steps. We first collect videos and existing captions from publicly available datasets and generate additional candidate captions using GPT-6 Astra [21] and Gemini 3.1 Pro [12]. Human annotators select, merge, correct, and supplement these descriptions to produce ground-truth captions. Next, Qwen3.8-27B [24] examines a low-frame-rate global preview and constructs a hierarchical Context–Focus–Points (CFP) observation plan. Following this plan, it revisits relevant temporal segments at higher frame rates and uses tools such as image cropping and OCR to inspect local details before generating a candidate caption.

For trajectory verification, we decompose the ground-truth and generated captions into atomic facts. We use Gemini 3.1 Pro to compare the resulting fact sets for factual consistency and reference-fact coverage, and to verify whether the collected visual evidence supports the key facts in the generated caption. If verification fails, Qwen3.8-27B repeats the complete planning, observation, and captioning process, with up to N attempts per video. Samples that fail all N attempts enter teacher-guided recovery, where Gemini 3.1 Pro generates a new trajectory following the same observation workflow. If the teacher trajectory also fails verification, the verifier provides feedback based on the ground-truth caption, identifying missing or unsupported facts. The teacher then uses this feedback to revise its observation plan, gather additional evidence, and regenerate the caption for verification by the same verifier. Finally, we retain verified trajectories from both Qwen3.8-27B and the teacher, including their observation plans, tool actions, observation evidence, and final captions, for supervised fine-tuning.

## 4 Experiments

## 4.1 Experimental Setup

Data and training. We initialize Reviso from Qwen3.5-9B [23]. Our training data comprise 10, 000 trajectories for supervised fine-tuning (SFT) and 20, 000 samples for reinforcement learning (RL), drawn from LLaVA-Video-178K [54], Vript [45], Ego4D [14], and YouTube. The SFT and RL datasets contain no overlapping videos. We first perform SFT with LoRA for two epochs, followed by 200 steps of GRPO with trajectories sampled online. The visual encoder remains frozen throughout training, while GRPO updates all other model parameters.

Baselines. We compared Reviso with 15 MLLMs. Our open-source baselines include ARC-Qwen-Video-7B [11], Molmo2-8B [7], and Tarsier2-7B [50]. We also evaluate InternVL3.5 [35], LLaVA-OneVision-2 [1], and the Qwen3-VL [3], Qwen3.5 [23], Qwen3.6 [25, 26], and Qwen3.8 [24] families. Qwen3.5-9B is our backbone baseline. We include GPT-6 Astra [21] and Gemini 3.8 Flash [13] as proprietary reference models.

Evaluation Benchmarks. We evaluate Reviso on 6 benchmarks, including four video captioning benchmarks and two downstream video understanding benchmarks. On DREAM-1K [32] and MCTS-VCB [49], we report precision, recall, and F1 to assess factual accuracy and information coverage. CaReBench [42] evaluates fine-grained descriptions of actions and objects, for which we report action F1 and object F1. On VDC [4], we report TP score and TP accuracy for detailed captions. We further evaluate whether the generated captions support downstream video understanding on LVBench [36] and MMVU [57]. We report accuracy on LVBench and multiple-choice, open-ended, and overall accuracy on MMVU.

Implementation details. For fair comparison, we limit all models to a maximum of 128 input frames per video. Baselines use uniformly sampled frames, while Reviso distributes the same frame budget across its global preview and subsequent temporal revisits. This ensures a consistent maximum frame budget across all evaluated methods.

## 4.2 Main Results

As shown in Table 1, Reviso achieves the best overall results among the evaluated open-source models across four video captioning benchmarks, consistently outperforming its Qwen3.5-9B backbone. On DREAM-1K, Reviso achieves the highest precision and improves F1 by 4.4 percentage points over the backbone. More substantial gains are observed on MCTS-VCB, where precision and recall improve by 8.1 and 19.1 points, respectively, resulting in a 15.4-point F1 improvement. On CaReBench, Reviso achieves the highest action and object F1 scores, outperforming its backbone by 9.1 and 6.7 points, respectively. It also achieves the best results among the evaluated open-source models on both VDC Detailed metrics. Together, these results demonstrate the efectiveness of Reviso in improving factual coverage and fine-grained video descriptions across diverse benchmarks.

Table 1 Results on four video captioning benchmarks. Benchmark metrics retain their original units. The best score among open-source models and Reviso in each column is highlighted in green . Gray closed-source rows are for reference and excluded from ranking.
<table><tr><td rowspan="2">Model</td><td colspan="3">DREAM-1K (%)</td><td colspan="3">MCTS-VCB (%)</td><td colspan="2">CaReBench (%)</td><td colspan="2">VDC (Detailed)</td></tr><tr><td>Prec.↑ Rec.↑</td><td></td><td>F1↑</td><td>Prec.↑ Rec.↑</td><td></td><td>F1↑</td><td>Action F1↑</td><td>Object F1↑</td><td>TP score↑</td><td>TP acc.↑</td></tr><tr><td colspan="10">Proprietary Models</td></tr><tr><td>GPT-6 Astra [21]</td><td>24.7</td><td>61.8</td><td>35.2</td><td>30.7</td><td>64.5</td><td>41.6</td><td>44.3</td><td>34.5</td><td>3.00</td><td>0.576</td></tr><tr><td>Gemini 3.8 Flash [13]</td><td>34.8</td><td>50.2</td><td>41.1</td><td>47.8</td><td>50.7</td><td>49.2</td><td>39.7</td><td>34.3</td><td>2.46</td><td>0.486</td></tr><tr><td colspan="10">Open-Source Models</td></tr><tr><td>ARC-Qwen-Video-7B [11]</td><td>31.2</td><td>31.4</td><td>31.3</td><td>58.4</td><td>42.7</td><td>49.3</td><td>24.9</td><td>30.2</td><td>2.90</td><td>0.554</td></tr><tr><td>InternVL3.5-8B [35]</td><td>41.2</td><td>18.8</td><td>25.8</td><td>61.7</td><td>24.0</td><td>34.6</td><td>13.2</td><td>19.7</td><td>2.12</td><td>0.388</td></tr><tr><td>InternVL3.5-38B [35]</td><td>36.0</td><td>34.7</td><td>35.3</td><td>65.7</td><td>25.6</td><td>36.8</td><td>14.3</td><td>20.5</td><td>2.12</td><td>0.393</td></tr><tr><td>LLaVA-OneVision-2 [1]</td><td>34.7</td><td>24.9</td><td>29.0</td><td>64.7</td><td>35.6</td><td>45.9</td><td>23.5</td><td>30.4</td><td>2.65</td><td>0.505</td></tr><tr><td>Molmo2-8B [7]</td><td>31.2</td><td>37.6</td><td>34.1</td><td>47.3</td><td>45.9</td><td>46.6</td><td>29.0</td><td>33.7</td><td>2.67</td><td>0.503</td></tr><tr><td>Tarsier2-7B [50]</td><td>41.50</td><td>37.1</td><td>39.2</td><td>56.5</td><td>35.2</td><td>43.4</td><td>30.2</td><td>25.4</td><td>2.65</td><td>0.505</td></tr><tr><td>Qwen3-VL-8B-Instruct [3]</td><td>30.9</td><td>38.3</td><td>34.2</td><td>65.1</td><td>39.9</td><td>49.5</td><td>29.5</td><td>33.0</td><td>2.65</td><td>0.498</td></tr><tr><td>Qwen3-VL-32B-Instruct [3]</td><td>30.5</td><td>37.7</td><td>33.7</td><td>46.7</td><td>56.5</td><td>51.1</td><td>32.2</td><td>33.4</td><td>2.90</td><td>0.553</td></tr><tr><td>Qwen3-VL-32B-Thinking [3]</td><td>30.7</td><td>35.2</td><td>32.8</td><td>44.0</td><td>56.7</td><td>49.5</td><td>33.3</td><td>33.6</td><td>2.88</td><td>0.547</td></tr><tr><td>Qwen3.5-9B [23]</td><td>36.3</td><td>34.0</td><td>35.1</td><td>57.6</td><td>37.7</td><td>45.5</td><td>25.2</td><td>27.2</td><td>2.33</td><td>0.437</td></tr><tr><td>Qwen3.6-27B [26]</td><td>36.7</td><td>36.9</td><td>36.8</td><td>60.5</td><td>42.5</td><td>49.9</td><td>25.5</td><td>27.2</td><td>2.67</td><td>0.506</td></tr><tr><td>Qwen3.6-35B-A3B [25]</td><td>38.1</td><td>33.2</td><td>35.5</td><td>59.7</td><td>41.4</td><td>48.9</td><td>24.0</td><td>19.7</td><td>2.66</td><td>0.504</td></tr><tr><td>Qwen3.8-27B [24]</td><td>34.5</td><td>37.2</td><td>35.8</td><td>47.6</td><td>47.7</td><td>47.6</td><td>29.2</td><td>32.0</td><td>2.63</td><td>0.497</td></tr><tr><td colspan="10">Our Model</td></tr><tr><td>Reviso (Ours)*</td><td>42.3</td><td>37.0</td><td>39.5</td><td>65.7</td><td>56.8</td><td>60.9</td><td>34.3</td><td>33.9</td><td>2.95</td><td>0.568</td></tr></table>

## 4.3 Ablation Study

We conduct ablation studies on DREAM-1K and CaReBench, reporting F1 on DREAM-1K and action and object F1 on CaReBench. We compare the base, SFT-only, and Reviso, together with a variant without tool use. We also remove KFR, PAR, and UTCP individually to assess the contribution of each reward component. As shown in Table 2, SFT improves all three metrics over the base model, with the largest gain in action F1. The variant without active captioning remains close to the base model, supporting the importance of active observation. The full GRPO objective further improves all three metrics over SFT and achieves the best results among the evaluated variants. Re moving KFR or UTCP lowers performance on both

Table 2 Ablation of tool use and training. All scores are percentages. Base denotes Qwen3.5-9B. Base and fullmodel results follow Table 1. Checkmarks indicate active reward components.
<table><tr><td rowspan="2">Variant</td><td>Reward</td><td></td><td>DREAM-1K</td><td colspan="2">CaReBench</td></tr><tr><td>KFR PAR UTCP</td><td></td><td>F1↑</td><td></td><td>Action F1↑ Object F1↑</td></tr><tr><td>Base</td><td></td><td></td><td>35.1</td><td>25.2</td><td>27.2</td></tr><tr><td>SFT only</td><td></td><td></td><td>37.6</td><td>30.7</td><td>30.1</td></tr><tr><td>w/o Tool use</td><td></td><td></td><td>35.3</td><td>24.8</td><td>27.4</td></tr><tr><td colspan="6">GRPO reward variants</td></tr><tr><td>w/o KFR</td><td></td><td>√</td><td>38.2</td><td>32.9</td><td>32.1</td></tr><tr><td>w/o PAR</td><td>√</td><td></td><td>37.9</td><td>31.3</td><td>29.7</td></tr><tr><td>w/o UTCP</td><td>√</td><td>√</td><td>38.4</td><td>33.7</td><td>32.4</td></tr><tr><td>Full</td><td>√</td><td>√</td><td></td><td>34.3</td><td>33.9</td></tr></table>

benchmarks, indicating that factual supervision and the tool-call penalty both contribute to caption quality. Removing PAR causes the largest reduction among the reward ablations, including a 4.2-point drop in object F1. These results show that active observation and the full reward objective improve Reviso’s ability to describe video content accurately.

![](images/3e0f4d5a683b35a163881da547dbf5f33106965ae828f7881d434f9e8e1372b8.jpg)  
(a) Efect of tool-assisted observation

![](images/37acf14960405c22c2abb5ff4982ad0b4c542c089fd1007d3f0624221a088361.jpg)  
(b) Efect of frame sampling rate

![](images/7849e69ce4286053962a65dbd9d6491e41e5757947981f48f54da4f90162e40a.jpg)  
(c) Efect of captioning strategy  
Figure 5 (a) Efect of tool-assisted observation on CaReBench. Bars show the number of videos in each duration group (left axis). The line shows the object F1 diference between tool-assisted and direct captioning with the same Reviso model (right axis, percentage points). (b) Efect of frame sampling rate on DREAM-1K. Direct captioning is evaluated at 2, 4, 6, and 10 FPS. (c) Efect of captioning strategy on DREAM-1K. Direct captioning at 2 FPS is compared with the Codex and Claude Code workflows for five VLMs. The dashed lines in (b) and (c) mark Reviso’s F1 score of 39.5.

## 4.4 Analysis and Findings

## 4.4.1 Caption-To-QA Evaluation

To assess the utility of generated captions for downstream video understanding, we evaluate captionbased question answering on LVBench [36] and MMVU [57]. We use Gemini 3.1 Pro [12] to extract answers to the benchmark questions from the generated captions, without access to the original videos. All captions are generated independently of the questions. As shown in Table 3, Reviso achieves the highest accuracy across all four evaluation metrics. Compared with its Qwen3.5-9B backbone, Reviso improves accuracy by 4.9 percentage points on LVBench and 2.1 points on MMVU Overall. These results suggest that Reviso’s captions better preserve information useful for downstream video understanding.

Table 3 Caption-to-QA results on two video understanding benchmarks. We report accuracy (%). The best result in each column is shown in bold.
<table><tr><td rowspan="2">Model</td><td rowspan="2"></td><td rowspan="2">Params. LVBench</td><td colspan="3">MMVU</td></tr><tr><td></td><td></td><td>MCQ Open Overall</td></tr><tr><td>Gemini 3.1 Pro [12]</td><td>-</td><td>41.2</td><td>77.6</td><td>46.7</td><td>66.1</td></tr><tr><td>ARC-Qwen-Video [11]</td><td>7B</td><td>48.1</td><td>77.6</td><td>48.8</td><td>66.8</td></tr><tr><td>InternVL3.5 [35]</td><td>38B</td><td>41.8</td><td>69.2</td><td>43.7</td><td>59.6</td></tr><tr><td>Tarsier2 [50]</td><td>7B</td><td>45.8</td><td>76.2</td><td>54.2</td><td>67.9</td></tr><tr><td>Qwen3.6 [25]</td><td>35B</td><td>45.3</td><td>77.1</td><td>47.2</td><td>65.9</td></tr><tr><td>Qwen3.8 [24]</td><td>27B</td><td>44.5</td><td>81.3</td><td>50.1</td><td>69.6</td></tr><tr><td>Qwen3.5 [23]</td><td>9B</td><td>47.1</td><td>79.2</td><td>56.1</td><td>70.6</td></tr><tr><td>Reviso (Ours)†</td><td>9B</td><td>52.0</td><td>81.7</td><td>57.9</td><td>72.7</td></tr></table>

## 4.4.2 Benefits of Tool-Assisted Observation

To assess the benefits of fine-grained observation, we compare Reviso with a variant that generates captions without tool assistance on CaReBench. Figure 5(a) shows the diference in object F1 (∆ Object F1) between the two settings across video duration groups. Tool-assisted observation yields larger improvements for longer videos, reaching approximately 10 percentage points for clips lasting 24–27 seconds. For shorter videos, the additional gains are relatively limited, potentially because the frame budget already provides dense temporal coverage. These results suggest that targeted observation is particularly useful for longer videos, where sparse sampling may overlook important visual information.

## 4.4.3 Does Higher FPS Improve Caption Quality

To investigate the efect of frame sampling density on caption quality, we evaluate direct-captioning models with uniform sampling at diferent frame rates on DREAM-1K [32]. As shown in Figure 5(b), increasing the sampling rate generally improves caption quality, but the gains are not consistent across models. While Molmo2-8B [7] and Qwen3.8-27B [24] continue to benefit from denser sampling, Qwen3.5-9B [23] and GPT-6 Astra [21] exhibit performance declines at 10 FPS. Notably, Reviso outperforms all evaluated direct-captioning configurations, despite their increased sampling rates. These results suggest that uniformly increasing frame density alone is insuficient to ensure better caption quality, highlighting the potential benefits of goal-directed visual observation.

![](images/d9726fdf4cba9f48d8c3e498a17d1978087636015ef1bdeba23588f1e845f90e.jpg)  
Figure 6 Qualitative comparison of Qwen3.5-9B and Reviso. Reviso identifies key information to inspect and uses OCR to track the game state, producing a more accurate caption. Red, green, and blue indicate incorrect facts, key facts, and fine-grained details, respectively.

## 4.4.4 Comparison of Captioning Strategies

To evaluate diferent captioning strategies, we compare Reviso with direct captioning and general-purpose agentic workflows based on Codex [20] and Claude Code [2]. For direct captioning, each MLLM generates captions from uniformly sampled video frames, while the agentic workflows enable the same models to plan observations and invoke tools before generating captions. As shown in Figure 5(c), agentic workflows improve caption quality for several models, although the gains vary across models and harnesses. Notably, Reviso outperforms most evaluated model-harness combinations, with only GPT-6 Astra [21] paired with Codex achieving a higher F1 score. These results suggest that access to observation tools alone does not guarantee high-quality captions, highlighting the potential benefits of learning video-specific observation strategies.

## 5 Case Study

As illustrated in Figure 6, Qwen3.5-9B correctly identifies the shooting action but mistakenly interprets the player’s subsequent gesture as a celebration of victory. In contrast, Reviso revisits the shot sequence to verify the outcome and applies cropping and OCR to extract the score and remaining time from the scoreboard. The acquired evidence confirms that Denver takes a 74–72 lead after the three-pointer, with 1:45 remaining in the fourth quarter. By incorporating this evidence, Reviso accurately describes the game state without incorrectly concluding that the game has ended. This example highlights Reviso’s ability to identify key information for verification, gather relevant evidence through targeted tool use, and avoid unsupported conclusions.

## 6 Conclusion

In this work, we present Reviso, an agentic video captioning framework that learns to prioritize important visual information and actively gather supporting evidence. Reviso employs a hierarchical Context–Focus– Points (CFP) observation plan to guide multi-round, tool-assisted observation, adapting its plan based on newly acquired evidence. To learn this observation strategy, we develop a verification-guided trajectory construction pipeline and introduce tailored rewards for GRPO to improve caption quality, observation planning, and regulate tool-use costs. Experiments across four video captioning benchmarks show that Reviso outperforms the evaluated open-source baselines, while its generated captions better support downstream video question answering. Further analyses highlight the benefits of observation planning and targeted evidence gathering for improving caption quality. Together, these findings demonstrate the efectiveness and potential of priority-guided active captioning, where models learn what deserves attention and actively gather evidence to support accurate video descriptions.

## References

[1] Xiang An, Yin Xie, Feilong Tang, et al. LLaVA-OneVision-2: Towards next-generation perceptual intelligence. arXiv preprint arXiv:2605.25979, 2026.

[2] Anthropic. Claude Code, 2025. https://github.com/anthropics/claude-code. Accessed: 2026-11-03.

[3] Shuai Bai, Yuxuan Cai, Ruizhe Chen, et al. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631, 2025.

[4] Wenhao Chai, Enxin Song, Yilun Du, et al. AuroraCap: Eficient, performant video detailed captioning and a new benchmark. In International Conference on Learning Representations (ICLR), pages 16845–16889, 2025.

[5] Boyu Chen, Zikang Wang, Zhengrong Yue, et al. VideoChat-M1: Collaborative policy planning for video understanding via multi-agent reinforcement learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 33772–33783, 2026.

[6] Xinlong Chen, Weihong Lin, Yuanxing Zhang, et al. PIGEONS: Incremental long-video captioning with persistent identity gallery and evolving overarching narrative summary. arXiv preprint arXiv:2610.16066, 2026.

[7] Christopher Clark, Jieyu Zhang, Zixian Ma, et al. Molmo2: Open weights and data for vision-language models with video understanding and grounding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 28652–28668, 2026.

[8] Yang Ding, Xin Lai, Yizhen Zhang, Wei Li, Ruihang Chu, and Yujiu Yang. VideoZoomer: Reinforcement-learned temporal focusing for long video reasoning. In International Conference on Learning Representations (ICLR), pages 20087–20111, 2026.

[9] Tiehan Fan, Kepan Nan, Rui Xie, et al. InstanceCap: Improving text-to-video generation via instance-aware structured caption. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 28974–28983, 2025.

[10] Mingqi Gao, Hongyuan Dong, Yifei Chen, et al. Claim-level rubric rewards for video caption reinforcement learning. arXiv preprint arXiv:2607.05150, 2026.

[11] Yuying Ge, Yixiao Ge, Chen Li, et al. ARC-Hunyuan-Video-7B: Structured video comprehension of real-world shorts. arXiv preprint arXiv:2507.20939, 2025.

[12] Google DeepMind. Gemini 3.1 Pro model card, 2026. https://deepmind.google/models/model-cards/gemini-3-1-pro/. Accessed: 2026-11-01.

[13] Google DeepMind. Gemini 3.8 Flash model card, 2026. https://deepmind.google/models/model-cards/gemini-3-8-flash/. Accessed: 2026-11-01.

[14] Kristen Grauman, Andrew Westbury, Eugene Byrne, et al. Ego4D: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18995–19012, 2022.

[15] Tengfei Liu, Yang Shi, Yuran Wang, et al. RefCaptioner: Multi-reference image-grounded video captioning. arXiv preprint arXiv:2607.28509, 2026.

[16] Tengfei Liu, Yang Shi, Xuanyu Zhu, et al. LongAV-Compass: Towards unified evaluation of minute-scale audio-visual generation across T2AV, I2AV, and V2AV. arXiv preprint arXiv:2605.26244, 2026.

[17] Wenqi Liu, Shijie Ma, Yunxiao Wang, et al. Thinking beyond videos: Unifying video reasoning and deep research for open-world video agents. arXiv preprint arXiv:2608.23329, 2026.

[18] Wenqi Liu, Yunxiao Wang, Shijie Ma, et al. VideoTemp-o3: Harmonizing temporal grounding and video understanding in agentic thinking-with-videos. In Proceedings of the 43rd International Conference on Machine Learning (ICML), pages 78754–78776, 2026.

[19] Ziyang Ma, Ruiyang Xu, Zhenghao Xing, et al. Omni-Captioner: Data pipeline, models, and benchmark for omni detailed perception. In International Conference on Learning Representations (ICLR), pages 132277–132304, 2026.

[20] OpenAI. Codex CLI, 2025. https://github.com/openai/codex. Accessed: 2026-11-03.

[21] OpenAI. GPT-6 Astra: A new generation of intelligence, 2026. https://openai.com/index/gpt-6-astra/. Accessed: 2026-11-01.

[22] Kaixiang Qiu, Minghao Han, Keliang Liu, et al. OmniFysics-Captioner technical report: Grounding omni-modal understanding in the physical world for better captioning. arXiv preprint arXiv:2609.31714, 2026.

[23] Qwen Team. Qwen3.5: Towards native multimodal agents, 2026. https://qwen.ai/blog?id=qwen3.5. Accessed: 2026-11-01.

[24] Qwen Team. Qwen3.8-Max: A new bar for coding and cowork, 2026. https://qwen.ai/blog?id=qwen3.8. Accessed: 2026-11-01.

[25] Qwen Team. Qwen3.6-35B-A3B: Agentic coding power, now open to all, 2026. https://qwen.ai/blog?id=qwen3.6-35b-a3b. Accessed: 2026-11-01.

[26] Qwen Team. Qwen3.6-27B: Flagship-level coding in a 27B dense model, 2026. https://qwen.ai/blog?id=qwen3.6-27b. Accessed: 2026-11-01.

[27] Zhihong Shao, Peiyi Wang, Qihao Zhu, et al. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

[28] Yang Shi, Jiaheng Liu, Yushuo Guan, et al. Mavors: Multi-granularity video representation for multimodal large language model. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 10994–11003, 2025.

[29] Wanshun Su, Yang Shi, Feihu Liu, et al. OmniPack: Unified token compression for eficient omni-modal large language models. arXiv preprint arXiv:2608.03812, 2026.

[30] Yuqi Tang, Tengfei Liu, Yizheng Lai, et al. KeyFrame-Compass: Towards comprehensive evaluation of keyframeconditioned video generation. arXiv preprint arXiv:2607.14202, 2026.

[31] Yuqi Tang, Yang Shi, Zhuoran Zhang, et al. Artifact-Bench: Evaluating MLLMs on detecting and assessing the artifacts of AI-generated videos. arXiv preprint arXiv:2605.18984, 2026.

[32] Jiawei Wang, Liping Yuan, Yuchen Zhang, and Haomiao Sun. Tarsier: Recipes for training and evaluating large video description models. arXiv preprint arXiv:2407.00634, 2024.

[33] Qixun Wang, Yang Shi, Letian Cheng, et al. Beacon: Knowing when and how to perform agentic visual reasoning. arXiv preprint arXiv:2607.28595, 2026.

[34] Qixun Wang, Yang Shi, Yifei Wang, et al. Monet: Reasoning in latent visual space beyond image and language. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12030–12040, 2026.

[35] Weiyun Wang, Zhangwei Gao, Lixin Gu, et al. InternVL3.5: Advancing open-source multimodal models in versatility, reasoning, and eficiency. arXiv preprint arXiv:2508.18265, 2025.

[36] Weihan Wang, Zehai He, Wenyi Hong, et al. LVBench: An extreme long video understanding benchmark. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 22958–22967, 2025.

[37] Xiaohan Wang, Yuhui Zhang, Orr Zohar, and Serena Yeung-Levy. VideoAgent: Long-form video understanding with large language model as agent. In European Conference on Computer Vision (ECCV 2024), pages 58–76, 2025.

[38] Yi Wang, Kunchang Li, Xinhao Li, et al. InternVideo2: Scaling foundation models for multimodal video understanding. In European Conference on Computer Vision (ECCV 2024), pages 396–416, 2025.

[39] Ziyang Wang, Shoubin Yu, Elias Stengel-Eskin, et al. VideoTree: Adaptive tree-based video representation for LLM reasoning on long videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3272–3283, 2025.

[40] Hongchen Wei, Zhihong Tan, Yaosi Hu, Chang Wen Chen, and Zhenzhong Chen. LongCaptioning: Unlocking the power of long video caption generation in large multimodal models. arXiv preprint arXiv:2502.15393, 2025.

[41] Penghao Wu and Saining Xie. V\*: Guided visual search as a core mechanism in multimodal LLMs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13084–13094, 2024.

[42] Yifan Xu, Xinhao Li, Yichun Yang, Desen Meng, Rui Huang, and Limin Wang. CaReBench: A fine-grained benchmark for video captioning and retrieval. In International Conference on Learning Representations (ICLR), pages 51124–51150, 2026.

[43] Yifan Xu, Zihao Wang, Zhixiao Wang, et al. PercepCap: Video captioner with structured spatio-temporal perception. arXiv preprint arXiv:2607.20389, 2026.

[44] Zihui Xue, Joungbin An, Xitong Yang, and Kristen Grauman. Progress-aware video frame captioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13639– 13650, 2025.

[45] Dongjie Yang, Suyuan Huang, Chengqiang Lu, et al. Vript: A video is worth thousands of words. In Advances in Neural Information Processing Systems (NeurIPS), pages 57240–57261, 2024.

[46] Penghui Yang, Long Xing, Xiaoyi Dong, et al. CapRL++: Unified reinforcement learning with verifiable rewards for dense image and video captioning. arXiv preprint arXiv:2606.09393, 2026.

[47] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, et al. CogVideoX: Text-to-video difusion models with an expert transformer. In International Conference on Learning Representations (ICLR), pages 83048–83077, 2025.

[48] Linli Yao, Yuancheng Wei, Yaojie Zhang, et al. TimeChat-Captioner: Scripting multi-scene videos with time-aware and structural audio-visual captions. arXiv preprint arXiv:2602.08711, 2026.

[49] Linhao Yu, Xingguang Ji, Yahui Liu, et al. Evaluating multimodal large language models on video captioning via Monte Carlo tree search. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL), Volume 1: Long Papers, pages 6435–6462, 2025.

[50] Liping Yuan, Jiawei Wang, Haomiao Sun, Yuchen Zhang, and Yuan Lin. Tarsier2: Advancing large vision-language models from detailed video description to comprehensive video understanding. arXiv preprint arXiv:2501.07888, 2025.

[51] Haoji Zhang, Xin Gu, Jiawen Li, et al. Thinking with videos: Multimodal tool-augmented reinforcement learning for long video reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 32903–32914, 2026.

[52] Xiaohan Zhang, Yuqing Wen, Junlin Chen, et al. MultiRef-Compass: Towards comprehensive evaluation of multi-reference-to-audio-video generation. arXiv preprint arXiv:2607.14189, 2026.

[53] YiFan Zhang, Yang Shi, Weichen Yu, et al. Debiasing multimodal large language models via penalization of language priors. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 4232–4241, 2025.

[54] Yuanhan Zhang, Jinming Wu, Wei Li, et al. LLaVA-Video: Video instruction tuning with synthetic data. Transactions on Machine Learning Research, 2025.

[55] Yaolun Zhang, Ruohui Wang, Jiahao Wang, et al. EVA: Eficient reinforcement learning for end-to-end video agent. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12289–12299, 2026.

[56] Zhuoran Zhang, Yang Shi, Qixun Wang, et al. Verus-Insight: Towards explainable artifact detection in AI-generated videos. arXiv preprint arXiv:2610.30866, 2026.

[57] Yilun Zhao, Haowei Zhang, Lujing Xie, et al. MMVU: Measuring expert-level multi-discipline video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8475–8489, 2025.