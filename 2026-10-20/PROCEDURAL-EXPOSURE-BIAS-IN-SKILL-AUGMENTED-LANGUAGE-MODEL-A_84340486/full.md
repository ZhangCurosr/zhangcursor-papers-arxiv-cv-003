# PROCEDURAL EXPOSURE BIAS IN SKILL-AUGMENTED LANGUAGE MODEL AGENTS

Qing Yang<sup>1</sup>, Yuqi Xiong<sup>1</sup>, Pengcheng Huang<sup>1</sup>, Zhipeng Xu<sup>2</sup> Zhenghao Li<sup>1∗</sup>, Zulong Chen<sup>2</sup>, Yu Gu<sup>1</sup>, Ge Yu<sup>1</sup>

<sup>1</sup>School of Computer Science and Engineering, Northeastern University, Shenyang, China <sup>2</sup>Alibaba Group, Hangzhou, China

## ABSTRACT

External skills provide language-model agents with reusable procedural knowledge, yet how models use such knowledge after it is supplied remains poorly understood. In this paper, we identify Procedural Exposure Bias (PEB), a systematic tendency of language models to increasingly favor an action as its associated procedural pattern is repeatedly exposed in a skill, even when the interaction state and the action’s relevance remain unchanged. We find that PEB is strongly dependent on how procedural knowledge is represented: presenting the same procedural content visually substantially attenuates the exposure-induced shift in action preference compared with direct textual presentation. Further analysis shows that this attenuation primarily arises from visual encoding rather than structured visual layout. Although repeated procedural signals remain recoverable from Visual representations, they exert a weaker influence on downstream action prediction, revealing a distinction between signal accessibility and its use as action evidence. This representation-dependent effect also extends to executable action selection, where increasing procedural exposure more strongly favors the corresponding action under Text than Visual. Layer-wise analyses show that exposure-dependent action evidence becomes progressively more pronounced toward the model output, with stronger evidence under Text, while activation interventions reveal architecturespecific roles of internal components in transmitting or counteracting the resulting preference shift. Finally, complete-agent evaluations show that these representation differences persist during end-to-end interaction, with visual skills yielding higher overall task success than textual skills across the evaluated models. Together, these findings establish skill representation as an important factor governing how procedural knowledge is translated into agent decisions. All code is available at https://github.com/NEUIR/Vis-Skill.

## 1 INTRODUCTION

Large Language Models (LLMs) have demonstrated strong emergent capabilities (Wu & Lo, 2025; Dong et al., 2025) and are increasingly deployed as agents in interactive environments (Park & Kwon, 2026; Xi et al., 2025). In these settings, external skills provide reusable procedural knowledge, including execution steps, applicability conditions, and operational constraints (Jiang et al., 2026a; Xiong et al., 2026). A typical skill-augmented agent constructs or extracts skills into a reusable library, retrieves or selects relevant skills according to the task, and provides them to the model to guide planning and action prediction (Xia et al., 2026; Zhao et al., 2026; Fang et al., 2026; Wang et al., 2026b). Such procedural guidance equips agents with reusable experience and can substantially facilitate the completion of complex, long-horizon tasks.

Prior work has extensively explored skill construction, extraction, representation, retrieval, and selection for LLMs (Wang et al., 2026b; Fang et al., 2026). These efforts make procedural knowledge accessible to the LLM, yet accessibility does not necessarily guarantee effective utilization (Li et al., 2026a; Liu et al., 2026). Once a skill is provided, the LLM must still determine which procedures are applicable to the current interaction state and which actions should be executed next (Han et al.,

2026; Jiang et al., 2026b). Importantly, a procedure can be relevant to the overall task while containing actions that are irrelevant to the current state. This raises a fundamental question: does the way procedural actions are exposed within a skill systematically affect how an LLM chooses actions, even when their actual relevance remains unchanged? Although previous studies have shown that LLM predictions can be influenced by irrelevant or repeatedly presented context (Yang et al., 2025; Niu et al., 2025; Geng et al., 2025), how such exposure effects manifest in procedural skill utilization remains poorly understood, due to the entangled effects of skill content and the model’s inherent behavioral biases.

In this paper, we uncover a systematic phenomenon that we term Procedural Exposure Bias (PEB). PEB refers to the tendency of an LLM to increasingly favor an action as its corresponding procedural pattern is repeatedly exposed in a skill, even when the interaction state and the action’s relevance remain unchanged. To isolate this effect, we conduct controlled experiments by manipulating the n-gram exposure frequency of action-related patterns while keeping the underlying task state and action relevance fixed. We observe that increasing the exposure of a procedural action consistently shifts the model’s action preference toward that action, revealing that LLMs can be biased by the frequency with which procedural actions are presented rather than solely by their state-dependent applicability. These findings suggest that skill utilization is not merely a matter of retrieving relevant procedures: the internal representation of a skill can itself introduce systematic biases into downstream action selection. Motivated by this observation, we further explore whether changing the modality of skill representation can reduce such exposure-induced bias. Specifically, we represent procedural skills visually and provide them to Vision-Language Models (VLMs), replacing explicit token-level repetition with visual representations of the same procedural information. Unlike textual skills, where repeatedly appearing action phrases directly increase lexical exposure, visual encoding transforms the procedure into distributed representations over image patches. This representation provides a natural way to decouple procedural information from its surface-level repetition.

Empirically, we find that visualized skills substantially attenuate the preference shift induced by repeated procedural actions, revealing a strong dependence of PEB on skill representation. By disentangling visual encoding from spatial layout, we find that most of this attenuation already emerges when the same textual skill is rasterized into an image, while additional structural formatting contributes only marginally. Importantly, direct signal-recovery probes show that the repeated procedural information remains recoverable under visual encoding, indicating that the representation difference lies not simply in whether the signal is accessible, but in how strongly it is translated into downstream action evidence. Internal analyses further show that this exposure-dependent action evidence develops progressively across model depth, while matched component interventions reveal architecture-dependent roles in transmitting or counteracting the resulting preference shift. Finally, we connect these controlled effects to agent behavior. Across the three benchmarks, Visual achieves a higher macro-average success rate than Text for all four evaluated backbones: Qwen3.5- 9B and Qwen3.5-27B (Qwen Team, 2026a), Gemma-4-31B-it (Gemma Team et al., 2026), and Kimi-K2.5 (Kimi Team et al., 2026). Controlled generation experiments further show that increasing procedural exposure can alter executable action selection, with a substantially stronger effect under Text than Visual.

## 2 UNCOVERING PROCEDURAL EXPOSURE BIAS IN SKILL-AUGMENTED LLM PLANNING

In this section, we investigate whether repeated exposure to action-related procedural patterns in a supplied skill can influence action prediction independently of their relevance to the current decision. We first formalize procedural exposure and construct a controlled probe that varies exposure while keeping the decision context and action applicability fixed. We then examine how the model’s action preference changes as exposure increases and determine whether the observed effect can be attributed to the repeated action-related signal.

## 2.1 PROCEDURAL EXPOSURE BIAS PROBING

To probe procedural exposure bias, we focus on action-bearing procedural elements, such as “open the microwave” or “put the object into the microwave”, which specify executable operations within a skill describing how a task should be performed. The same action-associated procedural pattern

![](images/89aafeb6b3eb7054eb186d01efa0dfdd37a81d7c94f5f19849896da6904d6caf.jpg)  
(a) Exposure-Dependent Distractor Preference.

![](images/d6f8d3933d426c599026d08deb90f825c31de41eff3d693a9d3c4ba419b71fc2.jpg)  
(b) Paired Changes in Distractor Preference.

![](images/ca2a32f5927a23d3db2040d1af9df02acdd5c27c3b9b7dadf7411e44ae3d8e07.jpg)  
(c) Keyword-Replacement Controls.  
Figure 1: Procedural Exposure Bias in Qwen3.5-9B Under Text Input and Image Rendering. Figure 1a reports distractor preference across exposure levels; Figure 1b, paired preference changes from $k = 0$ to $k = 8 ;$ and Figure 1c, the original, target-replacement, and frequency-matched control conditions.

may occur multiple times within a supplied skill, potentially exposing the model to the same actionrelated signal repeatedly.

To operationalize procedural exposure, for decision step i and exposure level k, let $\mathcal { C } _ { i } ^ { ( k ) }$ denote the supplied skill content, and let w(a) denote the target keyword associated with action a for the exposure manipulation. We define the procedural exposure frequency of action a as

$$
\mathcal { F } _ { i } ^ { ( k ) } ( a ) = \mathrm { C o u n t } \Big ( w ( a ) , \mathcal { C } _ { i } ^ { ( k ) } \Big ) ,\tag{1}
$$

where $\operatorname { C o u n t } ( \cdot )$ measures the number of occurrences of the target keyword in the supplied skill. In our controlled probe, the target keyword is an object-related noun associated with the action. For example, for the action “open microwave,” the target keyword is $w ( a ) =$ “microwave”.

To isolate the effect of exposure from action relevance, we sample intermediate decision steps from successful ALFWorld trajectories. At each step, we fix the observation history and the correct action $a _ { i } ^ { C }$ from the reference trajectory, and introduce a distractor action $a _ { i } ^ { D }$ that is relevant to the overall task but inapplicable to the current state. We construct skill variants in which the distractor keyword w $( a _ { i } ^ { D } )$ occurs $k \in \{ 0 , 1 , 2 , 4 , 8 \}$ times. The remaining insertion positions are filled with content that does not contain the target keyword, while keeping the total word count fixed across exposure levels.

We quantify model support by scoring the same fixed candidate actions rather than allowing the model to freely generate the next action. For a candidate action $a = ( a _ { 1 } , \ldots , a _ { | a | } )$ , we compute its length-normalized token log-probability:

$$
S _ { i } ( a ; k ) = { \frac { 1 } { | a | } } \sum _ { t = 1 } ^ { | a | } \log p _ { \theta } \left( a _ { t } \mid x _ { i } ^ { ( k ) } , a _ { < t } \right) ,\tag{2}
$$

where |a| is the number of action tokens and $x _ { i } ^ { ( k ) }$ denotes the complete input at exposure level k, including the fixed scoring prefix. Here, θ denotes model parameters, $a _ { t }$ is the t-th action token, and $a _ { < t }$ denotes the preceding action tokens. A higher score indicates stronger model support for the candidate action under the same decision context. Relative action preference is obtained by normalizing candidate scores across the fixed set, including the distractor.

As a reference condition, we also render the same skill content as images and provide it through the model’s visual encoder. Text and Visual use the same procedural content and exposure levels, differing only in how the supplied skill is represented. This comparison allows us to examine whether the relationship between procedural exposure and action prediction changes across representations.

## 2.2 PROCEDURAL EXPOSURE IS ASSOCIATED WITH ACTION PREDICTION

To characterize how procedural exposure influences action prediction independently of state relevance, we apply the controlled probe to Qwen3.5-9B across 165 ALFWorld decision steps. Figure 1 summarizes this analysis along three dimensions: its dependence on exposure frequency, its consistency across decision states, and its specificity to the repeated action-related keyword.

![](images/0185074ecf094f8a7cbac55dde85bbe26bef993280135923395dfede553c953d.jpg)  
Figure 2: Overview of the Controlled Comparison Between Text and Visual Skill Representations.

As illustrated in Figure 1a, increasing the frequency of the distractor keyword systematically elevates the preference of the model for the corresponding distractor under direct text input, even while the decision state and the actual applicability of the distractor remain strictly unchanged. We formally define this exposure-dependent increase in support for a state-inapplicable action as Procedural Exposure Bias (PEB). Furthermore, the results demonstrate that this increase is substantially attenuated when the skill is presented as an image, indicating that the magnitude of PEB is highly sensitive to the representational modality of the procedural content.

We further investigate this representation discrepancy at the granularity of individual decision states in Figure 1b. By comparing the zero-exposure (k = 0) and high-exposure (k = 8) conditions, we observe that direct text input induces substantial positive shifts in distractor preference across the entire state distribution. Conversely, the corresponding changes under image input remain tightly concentrated near zero. This distribution confirms that the dampened exposure effect under visual rendering is a widespread structural behavior broadly shared across decision states, rather than an artifact driven by a small subset of atypical examples.

To test whether the observed preference shift specifically depends on the repeated target keyword, we compare the original condition with two matched replacement controls. As shown in Figure 1c, replacing the target keyword substantially reduces distractor preference, whereas replacing a frequency-matched non-target word produces little change. This contrast indicates that the preference shift is specifically associated with repeated exposure to the target action-related keyword rather than lexical replacement in general. The reduction induced by target replacement is also smaller under Visual than under Text, consistent with the weaker exposure sensitivity observed un der visual encoding. Detailed probe construction, scoring procedures, replacement controls, and additional-model results are provided in Appendix A.1.

## 3 METHODOLOGY

Building on the empirical observation in Section 2.2 that visual representation attenuates PEB, we investigate how repeated procedural content enters action prediction and how the same content is processed through different representation pathways. We first formulate procedural exposure in autoregressive action prediction and characterize how repeated contextual evidence can contribute through Transformer attention aggregation. We then examine how the same procedural content is represented under Text and Visual input, yielding different pathways through which repeated evidence reaches action prediction. Figure 2 summarizes this controlled comparison.

## 3.1 PROCEDURAL EXPOSURE IN AUTOREGRESSIVE ACTION PREDICTION

To characterize how procedural exposure influences action prediction, we consider the autoregressive generation process, motivated by recent evidence that repeated contextual information can sys-

tematically alter language model reasoning (Yu et al., 2025). For a complete input $x _ { i } ^ { ( k ) }$ at decision step i and exposure level k, the probability of a candidate action $a = ( a _ { 1 } , \ldots , a _ { | a | } )$ factorizes as:

$$
p _ { \theta } { \Big ( } a \mid x _ { i } ^ { ( k ) } { \Big ) } = \prod _ { t = 1 } ^ { | a | } p _ { \theta } { \Big ( } a _ { t } \mid x _ { i } ^ { ( k ) } , a _ { < t } { \Big ) } .\tag{3}
$$

Under the controlled setting established in Section 2.1, the interaction history and candidate actions remain unchanged across exposure levels. Varying k therefore changes only the frequency with which the target procedural pattern appears in the context, while keeping the current interaction state and the relevance of each candidate action fixed.

Repeated procedural information can enter Transformer (Vaswani et al., 2017) computation through attention aggregation. At a generation position t, the attention output aggregates information from preceding context positions as:

$$
\mathbf { o } _ { t } = \sum _ { j = 1 } ^ { N } \alpha _ { t , j } \mathbf { v } _ { j } ,\tag{4}
$$

where N is the number of visible context positions when predicting $a _ { t } , \alpha _ { t , j }$ is the attention weight assigned to position $j , \mathbf { v } _ { j }$ is its value vector, and $\mathbf { o } _ { t }$ is the resulting attention output.

For the action-associated keyword under manipulation, increasing the exposure level k introduces additional occurrences of the same procedural pattern into the model context. Each occurrence occupies a distinct context position whose representation can contribute to the attention aggregation used for subsequent action prediction. Thus, increasing k increases the number of context positions carrying related procedural evidence, while leaving the interaction state and the applicability of the corresponding action unchanged. Importantly, this effect does not require any individual occurrence to receive a larger attention weight: the exposure manipulation changes the number of potential contributors to the aggregation rather than directly increasing the weight of the action occurrence.

## 3.2 REDUCING PROCEDURAL EXPOSURE BIAS THROUGH VISUAL SKILL ENCODING

To investigate why identical procedural exposure can produce different effects under different representations, we examine how the supplied skill is encoded before contributing to action prediction under Text and Visual input. We keep the procedural content and exposure frequency consistent across the two settings, while varying how the skill is encoded and delivered to the language model.

For the Text-based encoding, the skill is directly tokenized and inserted into the language model context. Repeating an action-associated keyword therefore introduces additional occurrences of the corresponding lexical pattern at distinct token positions. These positions directly participate in the Transformer computation described in Eq. 4, allowing repeated procedural evidence to contribute to action prediction through the language context. For the Visual-based encoding, the same procedural content is first rendered into an image. Following the standard Vision Transformer formulation (Dosovitskiy et al., 2021), the rendered image is divided into local patches, converted into patch-level representations, and processed by the model’s vision encoder before being passed to the language model. Consequently, repeated procedural content is no longer introduced as repeated lexical token occurrences in the language context. Instead, each rendered occurrence is encoded through the image patches covering its spatial region and transformed by the visual encoding pathway before contributing to downstream action prediction. The two settings thus preserve the same procedural content and exposure frequency while routing the corresponding procedural evidence through different intermediate representations before reaching the language model. Further details of the matched Text and Visual skill rendering procedure are provided in Appendix A.2.

## 4 EXPERIMENTAL METHODOLOGY

This section first describes the datasets, evaluation metrics, and baselines. We then present the implementation details of our experiments.

Datasets. We evaluate on three interactive benchmarks: ALFWorld (Shridhar et al., 2021), ScienceWorld (Wang et al., 2022), and WebShop (Yao et al., 2022), covering household tasks, scientific interaction, and online shopping, respectively. ALFWorld is additionally used for controlled procedural-exposure experiments, while our ScienceWorld evaluation covers diverse scientific task categories. Detailed benchmark descriptions are provided in Appendix A.3.1.

Table 1: Main results on ALFWorld, ScienceWorld, and WebShop under the default skill setting. Score denotes the average task score on WebShop, Succ. denotes the success rate (%), and Steps denotes the average number of interaction steps. Overall reports the macro-average success rate across the three benchmarks. Higher scores and success rates are better (↑). For interaction steps, shorter trajectories are preferred when task success is comparable (↓). The best result for each metric within each backbone is highlighted in bold.
<table><tr><td rowspan="2">Setting</td><td>Overall</td><td colspan="2">ALFWorld</td><td colspan="2">ScienceWorld</td><td colspan="3">WebShop</td></tr><tr><td>Macro Succ. ↑</td><td>Succ. ↑</td><td>Steps ↓</td><td>Succ. ↑</td><td>Steps ↓</td><td>Score ↑</td><td>Succ. ↑</td><td>Steps ↓</td></tr><tr><td colspan="9">Qwen3.5-9B</td></tr><tr><td>No Skill</td><td>16.21</td><td>15.67</td><td>27.59</td><td>18.96</td><td>17.04</td><td>38.52</td><td>14.00</td><td>13.52</td></tr><tr><td>Text Skill</td><td>40.70</td><td>66.42</td><td>18.28</td><td>42.18</td><td>15.36</td><td>39.53</td><td>13.50</td><td>12.85</td></tr><tr><td>Visual Skill</td><td>44.06</td><td>70.15</td><td>17.75</td><td>45.02</td><td>15.95</td><td>40.77</td><td>17.00</td><td>12.79</td></tr><tr><td colspan="9">Qwen3.5-27B</td></tr><tr><td>No Skill</td><td>41.33</td><td>58.96</td><td>19.01</td><td>45.02</td><td>15.83</td><td>49.27</td><td>20.00</td><td>12.10</td></tr><tr><td>Text Skill</td><td>53.06</td><td>78.36</td><td>14.87</td><td>57.82</td><td>14.28</td><td>53.93</td><td>23.00</td><td>11.96</td></tr><tr><td>Visual Skill</td><td>53.94</td><td>80.60</td><td>14.66</td><td>59.72</td><td>13.80</td><td>51.05</td><td>21.50</td><td>9.45</td></tr><tr><td colspan="9">Gemma-4-31B</td></tr><tr><td>No Skill</td><td>41.02</td><td>62.69</td><td>18.66</td><td>40.76</td><td>16.23</td><td>49.66</td><td>19.60</td><td>9.47</td></tr><tr><td>Text Skill</td><td>57.24</td><td>90.30</td><td>12.01</td><td>58.29</td><td>14.99</td><td>52.66</td><td>23.12</td><td>9.92</td></tr><tr><td>Visual Skill</td><td>58.21</td><td>90.30</td><td>11.78</td><td>59.72</td><td>14.60</td><td>52.69</td><td>24.62</td><td>9.60</td></tr><tr><td colspan="9">Kimi-K2.5</td></tr><tr><td>No Skill</td><td>49.73</td><td>78.36</td><td>16.76</td><td>48.82</td><td>16.54</td><td>50.64</td><td>22.00</td><td>15.11</td></tr><tr><td>Text Skill</td><td>56.34</td><td>90.30</td><td>12.75</td><td>50.71</td><td>15.95</td><td>54.88</td><td>28.00</td><td>13.35</td></tr><tr><td>Visual Skill</td><td>58.64</td><td>91.79</td><td>12.41</td><td>52.13</td><td>16.06</td><td>56.71</td><td>32.00</td><td>13.43</td></tr></table>

Evaluation Metrics. We report task success rate as the primary metric, together with interaction steps where applicable. Macro-average success rate is computed equally across the three benchmarks. For controlled experiments, we measure distractor preference, exposure sensitivity, and executable competing-action selection rates. Further details of benchmark-level evaluation metrics are provided in Appendix A.3.2.

Baselines. We compare No Skill, Text Skill, and Visual Skill. Text and Visual use identical ranked skills and procedural content, differing only in representation. Text supplies language tokens directly, whereas Visual uses rendered skill cards. An additional Rasterized condition isolates visual encoding from structured layout.

Implementation Details. We evaluate Qwen3.5-9B, Qwen3.5-27B, Gemma-4-31B-it, and Kimi-K2.5. Following SkillNet (Liang et al., 2026), Qwen3.8-Max (Qwen Team, 2026b) ranks skills for each task. Controlled PEB experiments use 165 ALFWorld decision states with exposure levels k ∈ {0, 1, 2, 4, 8}, while executable-action experiments use 185 states from 64 trajectories. The skill-ranking prompt is provided in Appendix A.3.3.

## 5 RESULTS AND ANALYSIS

We examine how skill representation affects agent performance and exposure-dependent action prediction. We first evaluate skill representations across models and interactive environments. Controlled experiments then separate visual encoding from layout, probe repeated-signal accessibility, and assess executable action selection. Finally, layer-wise readouts and component interventions characterize exposure-dependent evidence and architecture-dependent responses.

![](images/737ff2309b47fadd882ecc2761a1570ffac602bd8e65715c847fe9efe7e63b61.jpg)  
(a) Encoding Pathway Intervention.

![](images/d3689a6e1cf6e362f405fe91089c3499bff2b0532a96a1e29654714b3f5d95f1.jpg)  
(b) Repeated Signal Accessibility.

![](images/a2fa811932cd3eea9a54ddfa08fae3f2b93a4df7249277783a4ac3581594c2e0.jpg)  
(c) Executable Action Selection.

Figure 3: Representation-Dependent Effects of Procedural Exposure. Figure 3a compares exposure sensitivity under Text, Rasterized, and Structural Visual representations. Figure 3b reports exposuredependent target-word probability under Text and Visual encoding. Figure 3c reports the greedy selection rate of an executable competing action as procedural exposure increases.

## 5.1 VISUAL SKILLS IMPROVE OVERALL AGENT PERFORMANCE

We first test whether representation-dependent effects observed at controlled decision steps are reflected in complete agent interaction. We evaluate Qwen3.5-9B, Qwen3.5-27B, Gemma-4-31B-it, and Kimi-K2.5 on ALFWorld (Shridhar et al., 2021), ScienceWorld (Wang et al., 2022), and Web-Shop (Yao et al., 2022), spanning different model backbones and interactive tasks.

We compare No Skill, Text Skill, and Visual Skill. No Skill omits retrieved procedural guidance, while Text and Visual receive identical ranked skills and procedural content, isolating the effect of representation. Following SkillNet (Liang et al., 2026), Qwen3.8-Max (Qwen Team, 2026b) ranks the shared skill library for each task using its goal and skill metadata. Selected skills are then supplied as text or rendered images.

Table 1 shows that both skill representations improve macro-average success over No Skill for all four models. Visual outperforms Text in macro-average success for every backbone, although their relative performance varies by environment. These results do not establish whether reduced procedural exposure bias fully explains the performance advantage. We therefore examine representationdependent action prediction through controlled experiments. Further task-level results are provided in Appendix A.4.

## 5.2 WHY VISUAL SKILLS REDUCE PROCEDURAL EXPOSURE BIAS

We investigate the weaker influence of procedural exposure under Visual, following the encoding pathways in Section 3.2. We disentangle visual encoding from layout, probe repeated-signal accessibility and executable action selection, and examine layer-wise evidence and component-level interventions. Additional protocols and results are provided in Appendix A.5.

Visual Encoding Attenuates Procedural Exposure Bias. To separate visual encoding from structured layout, we compare three representations of identical procedural content. Text supplies lan guage tokens directly; Rasterized preserves the wording and ordering in a plain monospaced image; and Structural Visual uses a structured skill-card format based on the main-experiment design, separately implemented for the controlled probes.

Figure 3a shows that Qwen3.5-9B exhibits substantially lower exposure sensitivity under Rasterized than under Text, while Structural Visual shows comparable attenuation. This supports visual encoding, rather than layout alone, as the primary source of attenuation. Qwen3.5-27B and Gemma-4-31B-it show the same qualitative pattern, as detailed in Appendix A.5.1.

We next test whether weaker exposure sensitivity under Visual can be explained by reduced access to repeated procedural content. Rather than scoring candidate actions, we prompt the model to identify the most frequent object and score the target word at a fixed decoding position for $k \in \{ 0 , 4 , 8 \}$ Figure 3b shows that target-word probability rises with repetition under both Text and Visual for Qwen3.5-9B and Gemma-4-31B-it. Repetition thus continues to affect target-word prediction after visual encoding, despite its weaker influence on distractor-action prediction. This argues against an explanation based solely on information loss, but does not establish equal accessibility across representations. Readout controls and scoring details are provided in Appendix A.5.2.

Finally, we test whether exposure-dependent preference shifts affect generated actions. In a separate controlled setting, the state-inapplicable distractor is replaced by an admissible, task-relevant competing action, and we measure its greedy selection rate across exposure levels. Figure 3c shows that selection increases more strongly under Text than Visual. Thus, the representation-dependent effect extends beyond fixed-candidate scoring to executable actions. Additional action-selection analyses and paired rollout results are provided in Appendices A.5.3 and A.5.4, respectively.

Exposure-Dependent Action Evidence Develops Across Model Depth. The preceding experiments show that repetition continues to influence target-word prediction under Visual, while its effect on distractor-action prediction is attenuated. We next examine how this representationdependent difference is expressed across model depth. At each layer, we apply the final normalization and language-model head to intermediate representations at the candidate-action positions, obtaining a layer-wise readout of exposuredependent action evidence.

Figure 4 shows that exposure-dependent distractor support is weakly expressed in early-layer readouts and becomes more pronounced through the middle and later layers. The representations

![](images/aa805040a026169d68ea2ff8c26b72c6cc56b7f5f269cd01f80ff06e376059ad.jpg)  
(a) Text.

![](images/87f4a53023739ee96cd4ae4993e90f9ddb2c41948a352cda72fce57b396b9d5c.jpg)  
(b) Visual.  
Figure 4: Layer-Wise Readout of Exposure-Dependent Action Evidence in Qwen3.5-9B. Values show the mean change in lengthnormalized distractor action log-probability relative to the $k = 0$ baseline.

diverge at greater depth: Text exhibits a stronger increase in exposure-dependent distractor support, whereas Visual shows a substantially weaker increase. This observation is consistent with the different representation pathways described in Section 3.2, while the readout itself does not identify the internal computations responsible for the difference. Additional layer-wise results for Qwen3.5-27B and Gemma-4-31B-it are provided in Appendix A.5.5.

Component Interventions Reveal Architecture-Dependent Responses. Although the layerwise readout reveals where exposure-dependent action evidence becomes decodable, it does not establish the contribution of individual components to the final score change. We therefore conduct matched activation-patching interventions between zero-exposure $( k = 0 )$ and high-exposure $( k = 8 )$ runs. During the $k = 0$ pass, we replace a selected component output at the distractoraction positions with the corresponding cached activation from the $k = 8$ run and measure the resulting change in the distractor score. Figure 5 compares this intervention-induced change with the original exposure-induced change. Equality between the two corresponds to full recovery, while a zero intervention-induced change indicates no recovery under the tested replacement.

The interventions reveal different componentlevel responses across architectures. In Qwen3.5- 9B, MLP-output patching produces score changes opposite in direction to the original exposure effect under both representations, as shown in Figure 5a. In contrast, the same intervention in Gemma-4-31B-it produces partial recovery of the exposure-induced score increase, as shown in Figure 5b. Qwen3.5-27B exhibits a pattern similar to Qwen3.5-9B, with additional results provided in Appendix A.5.6. These results indicate that the response to a matched component intervention is architecture-dependent, even when the models show a similar overall reduction in exposure sensitivity under Visual.

![](images/c6ed9332efd99dd2da3ebd579e8201a9c5f829f10c922b4c31e0a4568cf0415d.jpg)

![](images/f4a5cea4252a671dc5a86373cf75c8b90fdc0b1f10229b443660b20985f959ac.jpg)  
(a) Qwen3.5-9B.  
(b) Gemma-4-31B-it.  
Figure 5: Component Intervention Under Controlled Procedural Exposure. The diagonal denotes full recovery of the exposure-induced change, and the horizontal dashed line denotes zero recovery.

The interventions thus provide complementary evidence about the internal expression of procedural exposure effects without attributing Visual’s attenuation of PEB to a single downstream component.

## 6 RELATED WORK

Recent work on skill-augmented language-model agents has studied the broader lifecycle of reusable procedural knowledge, including skill acquisition, organization, retrieval, and deployment (Zhou et al., 2026). Skill acquisition and evolution methods construct, refine, and reuse procedural knowledge from prior experience (Wang et al., 2026a), while recent benchmarks examine whether access to such skills leads to reliable improvements in downstream agent performance (Zhong et al., 2026). As skill libraries continue to grow, retrieval-based approaches increasingly focus on selecting task-relevant procedures from large collections (Zheng et al., 2026; Wang et al., 2026d), and injection-oriented pipelines further determine which retrieved information is ultimately supplied to the model (Li et al., 2026b). Together, these studies have substantially improved how procedural knowledge is acquired and made available to language-model agents, while also highlighting the importance of how supplied skills are ultimately utilized during state-conditioned decision making.

Language models are also known to be sensitive to statistical regularities and repeated patterns in contextual input (Cheng et al., 2025). In in-context learning, variations in demonstration composition and placement can induce majority-label and positional biases, causing predictions to reflect contextual statistics or presentation patterns rather than the intended input–label relation (Gupta et al., 2023; Cobbina & Zhou, 2025). Shortcut-learning studies show that models can rely on repeated lexical patterns or surface cues rather than the intended semantic relation (Doan et al., 2025; Tasawong et al., 2025). Explicit repetition can also systematically alter language-model reasoning behavior (Yu et al., 2025). These effects arise in a broader linguistic setting where token and word frequencies are highly non-uniform, as classically characterized by Zipf’s law (Piantadosi, 2014; D˛ebowski, 2026). Related exposure effects have also been studied in recommender systems, where frequently exposed or popular items can receive preference disproportionate to their relevance (He et al., 2025; Carnovalini et al., 2025). These lines of work collectively demonstrate that frequency, repetition, and contextual statistics can systematically shape model preference, although their effect within supplied procedural knowledge remains less explored.

The representation through which information is supplied can further affect how that information is processed and utilized (van Sprang et al., 2026). Prior work has shown that textual content can be rendered into visual form while preserving its underlying information (Li et al., 2025), while visually rendered inputs are processed through a distinct visual encoding pathway before being integrated into language-model computation (Wang et al., 2026c). Visual representations have also been extensively studied in agents operating over structured interfaces and multimodal observations (Liu et al., 2025; Xu et al., 2024; Hasegawa et al., 2025). In parallel, mechanistic studies have developed circuit-level accounts of Transformer computation and identified recurring mechanisms associated with in-context learning (Elhage et al., 2021; Olsson et al., 2022). More recent work has extended these analyses to layer-wise dynamics and broader patterns of information flow within Transformer models (Atad et al., 2026). Building on these research directions, we study how repeated procedural information within an already supplied skill influences action prediction, and how this effect changes when the same procedural content is represented through textual or visual encoding pathways.

## 7 CONCLUSION

This paper investigates how procedural exposure affects skill utilization in language-model agents. We identify and formalize Procedural Exposure Bias (PEB), showing that repeated exposure to an action-associated procedural pattern can systematically shift action preference even when the interaction state and action relevance remain unchanged. We further show that this effect is strongly representation-dependent: visual encoding substantially attenuates PEB while keeping the repeated procedural signal recoverable, and internal analyses reveal architecture-dependent pathways through which exposure-related action evidence develops. Finally, controlled generation and complete-agent evaluations show that these representation differences extend to executable decisions and persist across ALFWorld, ScienceWorld, and WebShop. These findings identify skill representation as an important factor in how language-model agents utilize externally supplied procedural knowledge.

## AI USE STATEMENT

Generative AI tools were used to assist with language polishing and improving the clarity of the manuscript. All research ideas, experimental design, implementation, analysis, and conclusions were developed and verified by the authors, who take full responsibility for the content of the paper.

## REFERENCES

Ido Andrew Atad, Itamar Zimerman, Shahar Katz, and Lior Wolf. TensorLens: End-to-end transformer analysis via high-order attention tensors. In Maria Liakata, Viviane P. Moreira, Jiajun Zhang, and David Jurgens (eds.), Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3452–3468, San Diego, California, United States, July 2026. Association for Computational Linguistics. ISBN 979-8-89176- 390-6. doi: 10.18653/v1/2026.acl-long.156. URL https://aclanthology.org/2026. acl-long.156/.

Filippo Carnovalini, Antonio Rodà, and Geraint A Wiggins. Popularity bias in recommender systems: The search for fairness in the long tail. Information, 16(2):151, 2025.

Liang Cheng, Zhaowei Wang, and Mark Steedman. Llms are frequency pattern learners in natural language inference. arXiv preprint arXiv:2505.21011, 2025.

Kwesi Adu Cobbina and Tianyi Zhou. Where to show demos in your prompt: A positional bias of in-context learning. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 29548–29581, 2025.

Łukasz D˛ebowski. From zipf’s law to neural scaling through heaps’ law and hilberg’s hypothesis. Journal of Machine Learning Research, 27(194):1–35, 2026.

Nhi Hoai Doan, Tatsuya Hiraoka, and Kentaro Inui. Understanding and controlling repetition neurons and induction heads in in-context learning. In Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference ofthe Asia-Pacific Chapter of the Association for Computational Linguistics, pp. 2854–2876, 2025.

Zhichen Dong, Zhanhui Zhou, Zhixuan Liu, Chao Yang, and Chaochao Lu. Emergent response planning in llms. arXiv preprint arXiv:2502.06258, 2025.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. URL https://openreview.net/forum? id=YicbFdNTTy.

Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, et al. A mathematical framework for transformer circuits. Transformer Circuits Thread, 1(1):12, 2021.

Runnan Fang, Yuan Liang, Xiaobin Wang, Jialong Wu, Shuofei Qiao, Pengjun Xie, Fei Huang, Huajun Chen, and Ningyu Zhang. Memp: Exploring agent procedural memory. In Findings of the Associationfor Computational Linguistics: ACL 2026, pp. 17490–17502, 2026.

Gemma Team, Sherif El Abd, Vaibhav Aggarwal, Robin Algayres, Alek Andreev, Olivier Bachem, Ian Ballantyne, Cormac Brick, Victor Carbune, Michelle Casbon, et al. Gemma 4 Technical˘ Report. arXiv preprint arXiv:2607.02770, 2026.

Jiayi Geng, Howard Chen, Ryan Liu, Manoel Horta Ribeiro, Robb Willer, Graham Neubig, and Thomas L Griffiths. Accumulating context changes the beliefs of language models. arXiv preprint arXiv:2511.01805, 2025.

Karan Gupta, Sumegh Roychowdhury, Siva Rajesh Kasa, Santhosh Kumar Kasa, Anish Bhanushali, Nikhil Pattisapu, and Prasanna Srinivasa Murthy. How robust are llms to in-context majority label bias? arXiv preprint arXiv:2312.16549, 2023.

Jinyi Han, Yuanjian Xu, Ying Liao, Xinyi Wang, Zishang Jiang, Zixiang Di, Fanyang Lu, Zhichao Hu, and Yanghua Xiao. Skill-use: Can llms actually use skills in agentic harnesses? arXiv preprint arXiv:2608.04828, 2026.

Kimihiro Hasegawa, Wiradee Imrattanatrai, Masaki Asada, Ken Fukuda, and Teruko Mitamura. Tama: Tool-augmented multimodal agent for procedural activity understanding. arXiv preprint arXiv:2510.00161, 2025.

Ming He, Zihao Zhang, Han Zhang, and Chang Liu. Relieving popularity bias in recommender systems via user group-level augmentation. Applied Soft Computing, 169:112610, 2025.

Yanna Jiang, Delong Li, Haiyu Deng, Baihe Ma, Xu Wang, Qin Wang, and Guangsheng Yu. Sok: Agentic skills–beyond tool use in llm agents. arXiv preprint arXiv:2602.20867, 2026a.

Zhiyuan Jiang, Fangrui Huang, Hanwen Xing, Xander Wu, Yipeng Gao, Rui Cao, Mengdi Wang, Shilong Liu, and Yijiang Li. Demystifying agent skills: Why they work-until they don’t. arXiv preprint arXiv:2608.14036, 2026b.

Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, SH Cai, Yuan Cao, Ziwei Chai, Y Charles, HS Che, Cheng Chen, et al. Kimi K2.5: Visual agentic intelligence. arXiv preprint arXiv:2602.02276, 2026.

Xiangyi Li, Yimin Liu, Wenbo Chen, Bingran You, Zonglin Di, Yifeng He, Shenghan Zheng, Kyoung Whan Choe, Jiankai Sun, Shuyi Wang, et al. Skillsbench: Benchmarking how well agent skills work across diverse tasks. arXiv preprint arXiv:2602.12670, 2026a.

Yanchao Li, Wanhao Liu, Ben Gao, Jiaqing Xie, Zhehong Ai, Na Zou, Yuqiang Li, and Tianfan Fu. Skillsinjector: Dynamic skill context construction for llm agents. arXiv preprint arXiv:2605.29794, 2026b.

Yanhong Li, Zixuan Lan, and Jiawei Zhou. Text or pixels? evaluating efficiency and understanding of llms with visual text inputs. In EMNLP (Findings), pp. 10564–10578, 2025.

Yuan Liang, Ruobin Zhong, Haoming Xu, Chen Jiang, Yi Zhong, Runnan Fang, Jia-Chen Gu, Shumin Deng, Yunzhi Yao, Mengru Wang, et al. Skillnet: Create, evaluate, and connect ai skills. arXiv preprint arXiv:2603.04448, 2026.

Xiao Liu, Tianjie Zhang, Yu Gu, Iat Long Iong, Song XiXuan, Yifan Xu, Shudan Zhang, Hanyu Lai, Jiadai Sun, Xinyue Yang, et al. Visualagentbench: Towards large multimodal models as visual foundation agents. In International Conference on Learning Representations, volume 2025, pp. 95650–95707, 2025.

Yujian Liu, Jiabao Ji, Li An, Tommi Jaakkola, Yang Zhang, and Shiyu Chang. How well do agentic skills work in the wild: Benchmarking llm skill usage in realistic settings. arXiv preprint arXiv:2604.04323, 2026.

Jingcheng Niu, Xingdi Yuan, Tong Wang, Hamidreza Saghir, and Amir H Abdi. Llama see, llama do: A mechanistic perspective on contextual entrainment and distraction in llms. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 16218–16239, 2025.

Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, et al. In-context learning and induction heads. arXiv preprint arXiv:2209.11895, 2022.

Sangeun Park and Minhae Kwon. Multi<sup>2</sup>: Hierarchical multi-agent decision-making with llm-based agents in interactive environments. arXiv preprint arXiv:2606.03698, 2026.

Steven T Piantadosi. Zipf’s word frequency law in natural language: A critical review and future directions. Psychonomic bulletin & review, 21(5):1112–1130, 2014. URL https://link. springer.com/article/10.3758/s13423-014-0585-6.

Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026a. URL https:// qwen.ai/blog?id=qwen3.5.

Qwen Team. Qwen3.8-Max: A new bar for coding and cowork, August 2026b. URL https: //qwen.ai/blog?id=qwen3.8.

Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew J. Hausknecht. Alfworld: Aligning text and embodied environments for interactive learning. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. URL https://openreview.net/forum?id= 0IOX0YcCdTn.

Panuthep Tasawong, Napat Laosaengpha, Wuttikorn Ponwitayarat, Sitiporn Lim, Potsawee Manakul, Samuel Cahyawijaya, Can Udomcharoenchaikit, Peerat Limkonchotiwat, Ekapol Chuangsuwanich, and Sarana Nutanong. Shortcut learning in safety: The impact of keyword bias in safeguards. In Proceedings of the The First Workshop on LLM Security (LLMSEC), pp. 189–197, 2025.

Angela van Sprang, Laurens Samson, Ana Lucic, Erman Acar, Sennay Ghebreab, and Yuki M. Asano. Same content, different answers: Cross-modal inconsistency in mllms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8781–8790, June 2026.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Chenxi Wang, Zhuoyun Yu, Xin Xie, Wuguannan Yao, Runnan Fang, Shuofei Qiao, Kexin Cao, Guozhou Zheng, Xiang Qi, Peng Zhang, et al. Skillx: Automatically constructing skill knowledge bases for agents. arXiv preprint arXiv:2604.04804, 2026a.

Jiongxiao Wang, Qiaojing Yan, Yawei Wang, Yijun Tian, Soumya Smruti Mishra, Zhichao Xu, Megha Gandhi, Panpan Xu, and Lin Lee Cheong. Reinforcement learning for self-improving agent with skill library. In Proceedings of the 64th Annual Meeting of the Association for Com putational Linguistics (Volume 1: Long Papers), pp. 1529–1550, 2026b.

Ruoyao Wang, Peter Jansen, Marc-Alexandre Côté, and Prithviraj Ammanabrolu. Scienceworld: Is your agent smarter than a 5th grader? In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 11279–11298, 2022.

Yifan Wang, Shiyu Li, Peiming Li, Xiaochen Yang, Zheng Wei, and Yang Tang. Render-of-thought: Rendering textual chain-of-thought as images for visual latent reasoning. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 45236–45253, 2026c.

Zifei Wang, Wei Wen, Qiang Ji, Ruizhi Qiao, and Xing Sun. Skill is not document: A queryconditional benchmark and two-stage retriever for llm agent skill routing. arXiv preprint arXiv:2606.03565, 2026d.

Tung-Yu Wu and Melody Lo. U-shaped and inverted-u scaling behind emergent abilities of large language models. In International Conference on Learning Representations, volume 2025, pp. 99426–99458, 2025.

Zhiheng Xi, Yiwen Ding, Wenxiang Chen, Boyang Hong, Honglin Guo, Junzhe Wang, Xin Guo, Dingwen Yang, Chenyang Liao, Wei He, et al. Agentgym: Evaluating and training large language model-based agents across diverse environments. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 27914–27961, 2025.

Peng Xia, Jianwen Chen, Hanyang Wang, Jiaqi Liu, Kaide Zeng, Yu Wang, Siwei Han, Yiyang Zhou, Xujiang Zhao, Haifeng Chen, et al. Skillrl: Evolving agents via recursive skill-augmented reinforcement learning. arXiv preprint arXiv:2602.08234, 2026.

Yuqi Xiong, Chunyi Peng, Zhipeng Xu, Zhenghao Liu, Zulong Chen, Yukun Yan, Shuo Wang, Yu Gu, and Ge Yu. Lang2act: Fine-grained visual reasoning through self-emergent linguistic toolchains. In Findings of the Association for Computational Linguistics: ACL 2026, pp. 8375– 8399, 2026.

Yiheng Xu, Zekun Wang, Junli Wang, Dunjie Lu, Tianbao Xie, Amrita Saha, Doyen Sahoo, Tao Yu, and Caiming Xiong. Aguvis: Unified pure vision agents for autonomous gui interaction. arXiv preprint arXiv:2412.04454, 2024.

Minglai Yang, Ethan Huang, Liang Zhang, Mihai Surdeanu, William Yang Wang, and Liangming Pan. How is llm reasoning distracted by irrelevant context? an analysis using a controlled benchmark. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 13340–13358, 2025.

Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. Webshop: Towards scalable real-world web interaction with grounded language agents. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper\_files/paper/2022/hash/ 82ad13ec01f9fe44c01cb91814fd7b8c-Abstract-Conference.html.

Sangwon Yu, Ik-hwan Kim, Jongyoon Song, Saehyung Lee, Junsung Park, and Sungroh Yoon. Unleashing multi-hop reasoning potential in large language models through repetition of misordered context. In Findings of the Association for Computational Linguistics: NAACL 2025, pp. 6435– 6455, 2025.

Xinyu Zhao, Zhen Tan, Vaishnav Tadiparthi, Nakul Agarwal, Kwonjoon Lee, Ehsan Moradi Pari, Hossein Nourkhiz Mahjoub, and Tianlong Chen. Generative skill composition for llm agents. arXiv preprint arXiv:2606.32025, 2026.

YanZhao Zheng, ZhenTao Zhang, Chao Ma, YuanQiang Yu, JiHuai Zhu, Yong Wu, Tianze Xu, Baohua Dong, Hangcheng Zhu, Ruohui Huang, et al. Skillrouter: Skill routing for llm agents at scale. arXiv preprint arXiv:2603.22455, 2026.

Shanshan Zhong, Yi Lu, Jingjie Ning, Yibing Wan, Lihan Feng, Yuyi Ao, Leonardo FR Ribeiro, Markus Dreyer, Sean Ammirati, and Chenyan Xiong. Skilllearnbench: Benchmarking continual learning methods for agent skill generation on real-world tasks. arXiv preprint arXiv:2604.20087, 2026.

Yingli Zhou, Wang Shu, Yaodong Su, Wenchuan Du, Yixiang Fang, and Xuemin Lin. A comprehensive survey on agent skills: Taxonomy, techniques, and applications. arXiv preprint arXiv:2605.07358, 2026.

## A APPENDIX

This appendix follows the organization of the main paper. We first provide the controlled construction and supplementary evidence for procedural exposure bias, followed by details of Text and Visual skill encoding, the shared experimental setup, and additional results for complete-agent evaluation and representation-dependent analyses.

## A.1 DETAILS OF PROCEDURAL EXPOSURE BIAS PROBING

This section supplements Section 2 with decision-state construction, exposure-controlled skill variants, candidate-action scoring, keyword-replacement controls, and additional-model results.

## A.1.1 DECISION STATES, DISTRACTORS, AND PROCEDURAL KEYWORDS

We extract 165 intermediate decision steps from successful ALFWorld trajectories. Each trajectory is replayed in the environment simulator to reconstruct the observation history and admissible action set at the selected step. Procedural guidance supplied during the source run is removed from the reconstructed history before constructing the probe.

For each decision step $i ,$ let $a _ { i } ^ { C }$ denote the action executed in the reference trajectory. We introduce a distractor action $a _ { i } ^ { \bar { D } }$ that is relevant to the overall task and refers to an existing environment entity but is excluded from the current admissible action set. The distractor is therefore task-relevant but state-inapplicable. Its verb is required to differ from that of the reference action.

Each distractor is associated with a target keyword $w ( a _ { i } ^ { D } )$ . Consistent with Section 2.1, the target is an object-related noun associated with the action; for example, microwave for open microwave. Occurrences are counted using whole-word matching, independently of the model’s tokenization. The target keyword must occur in the full 37-skill corpus, but is not required to appear in the five selected skills, and must not appear in the task query or the reference action. Keyword frequency serves as the operational measure of action-associated exposure in these probes.

## A.1.2 EXPOSURE-CONTROLLED SKILL CONSTRUCTION

We construct skill variants with target exposure levels $k \in \{ 0 , 1 , 2 , 4 , 8 \}$ while keeping the interaction state and action applicability fixed. Exposure is counted within the supplied skill, rather than across the full interaction history.

Before constructing the exposure variants, native occurrences of the target keyword throughout the descriptions and bodies of the five supplied skills are replaced using a fixed mapping of single-word neutral substitutes, while skill names remain unchanged. Fixed insertion slots are distributed across the skill document. At exposure level $k ,$ exactly k slots contain target-bearing procedural phrases, and the remaining slots contain filler phrases without the target keyword. Target-bearing and filler phrases are matched in word count. Apart from this keyword replacement, content outside the insertion slots remains unchanged, and automated checks verify that the total whole-word occurrence count in the supplied skill equals k.

The total word count is held constant across exposure levels for each decision state. This word-count constraint does not imply equal token counts under every model’s tokenizer.

## A.1.3 ACTION SCORING AND DISTRACTOR PREFERENCE

Let $\mathcal { A } _ { i } ^ { \mathrm { a d m } }$ denote the admissible action set at decision step i. We exclude look, inventory, help, wait, and examine actions, and retain at most 16 admissible candidates, denoted by $\mathcal { A } _ { i } ^ { \mathrm { s e l } }$ The scored candidate set is

$$
\mathcal { A } _ { i } = \mathcal { A } _ { i } ^ { \mathrm { s e l } } \cup \{ a _ { i } ^ { D } \} .\tag{5}
$$

This set is fixed across exposure levels and skill representations for each state. Including the inadmissible distractor permits its model support to be measured without executing it in the environment.

Candidates are scored using teacher forcing with a fixed scoring prefix. We use the lengthnormalized token log-probability in Equation 2. To make the representation explicit, we write this

![](images/7168d0255a5ffdb827be585a526e9292af371ccdf8be5a68a25074ce87216719.jpg)  
(a) Exposure-Dependent Distractor Preference.

![](images/80cbea0e8c5a41eb891f5cc3c4a69972a7da587882a9c654295f209a176e4b31.jpg)  
(b) Paired Changes in Distractor Preference.

![](images/cf4606ab6ff094c61ae294907e6f3f4600613bd213a7c2c992130fd2cd09f783.jpg)  
(c) Keyword-Replacement Controls.  
Figure 6: Procedural Exposure Bias in Qwen3.5-27B Under Text and Visual Skill Representations. Figure 6a: Distractor Preference Across Exposure Levels. Figure 6b: Paired Preference Changes From $k = 0$ to $k = 8 .$ Figure 6c: Keyword-Replacement Controls.

score as $S _ { i } ( a ; k , r )$ , where r denotes the skill representation and $x _ { i } ^ { ( k , r ) }$ denotes the corresponding complete input. Candidate scores are normalized as

$$
P _ { i } ( a ; k , r ) = \frac { \exp ( S _ { i } ( a ; k , r ) ) } { \sum _ { a ^ { \prime } \in { \mathcal { A } } _ { i } } \exp ( S _ { i } ( a ^ { \prime } ; k , r ) ) } .\tag{6}
$$

Distractor preference is $P _ { i } ( a _ { i } ^ { D } ; k , r )$ . This quantity is a normalized preference over the fixed candidate set, not the probability that unrestricted generation produces the distractor action.

Figure 1a reports mean distractor preference across exposure levels. For Figure 1b, preference is instead computed by normalizing summed action-token log-probabilities $\widetilde { S } _ { i } ( a ; k , r ) = | a | S _ { i } ( a ; k , r )$ across the same candidate set, yielding $\widetilde { P } _ { i } ( a ; k , r )$ . The within-state endpoint change is

$$
\Delta \widetilde { P } _ { i , r } ^ { D } = \widetilde { P } _ { i } ( a _ { i } ^ { D } ; 8 , r ) - \widetilde { P } _ { i } ( a _ { i } ^ { D } ; 0 , r ) .\tag{7}
$$

The distribution of these paired changes characterizes variation in the exposure effect across decision states.

## A.1.4 KEYWORD-REPLACEMENT CONTROL

$\mathrm { A t } \ k = 8$ , we compare three conditions to examine whether distractor preference depends on the target keyword rather than lexical replacement in general.

The original condition retains the exposure-controlled skill. The target-replacement condition replaces all occurrences of the distractor keyword with a neutral alternative while preserving word count and syntactic structure. The frequency-matched control condition replaces an unrelated word with the same occurrence frequency while leaving the target keyword intact. Control words are selected to avoid semantic overlap with the distractor action, reference action, task query, and operational verbs.

Figure 1c uses the normalized preference metric defined above. The original and target-replacement conditions include all 165 decision states. A qualifying frequency-matched control word is available for 128 states, and the frequency-matched control statistics use this subset. These sample counts differ; comparisons of aggregate values across the full set and the subset are therefore not fully paired.

## A.1.5 ADDITIONAL MODELS

Figures 6 and 7 report the same controlled probe for Qwen3.5-27B and Gemma-4-31B-it, respectively. Each figure includes exposure-dependent distractor preference, paired endpoint changes, and keyword-replacement controls.

![](images/e13c6929d454b822df96c73aad920e866fa840e41ffb0cdc4977cf24a4d81f4d.jpg)  
(a) Exposure-Dependent Distractor Preference.

![](images/f53e4be6c530b8c2c3a2514bbeacbe660ded6f8d65e9f902a27daf79c0fc3ceb.jpg)  
(b) Paired Changes in Distractor Preference.

![](images/6b1d41566d4983995ab9c20cc28d9e4549c68b6a5359de752355a4e09e9fb724.jpg)  
(c) Keyword-Replacement Controls.  
Figure 7: Procedural Exposure Bias in Gemma-4-31B-it Under Text and Visual Skill Representations. Figure 7a: Distractor Preference Across Exposure Levels. Figure 7b: Paired Preference Changes From k = 0 to k = 8. Figure 7c: Keyword-Replacement Controls.

## A.2 DETAILS OF TEXT AND VISUAL SKILL ENCODING

This section supplements Section 3.2 with details of the matched Text and Visual skill rendering procedure.

## A.2.1 VISUAL REFERENCE CONDITION

Each exposure-controlled Text skill is paired with a Visual counterpart containing the same procedural wording. The interaction history, exposure level, and candidate actions are matched across the two conditions.

Text supplies the skill as language tokens. Visual renders it as a structured skill card and supplies the resulting image through the model’s native visual encoding pathway, without adding auxiliary modules. Unless otherwise specified, Visual refers to this structured skill-card format. Because this comparison changes both the encoding pathway and the presentation of the skill, Appendix A.5.1 additionally evaluates plain rasterized text without the skill-card structure.

## A.3 DETAILS OF THE EXPERIMENTAL SETUP

This section supplements Section 4 with benchmark descriptions, evaluation metrics, and the task specific skill-ranking prompt. The complete-agent evaluation covers Qwen3.5-9B, Qwen3.5-27B, Gemma-4-31B-it, and Kimi-K2.5 under No Skill, Text Skill, and Visual Skill conditions.

## A.3.1 DATASETS

ALFWorld. ALFWorld (Shridhar et al., 2021) aligns text-based household environments with embodied tasks. Agents follow natural-language goals by locating objects, interacting with appliances and receptacles, and changing object states. Its six task categories are Pick & Place, Examine in Light, Clean & Place, Heat & Place, Cool & Place, and Pick Two & Place. Task success rate measures completion of the requested household goal. This task-level measure is distinct from partial goal-condition completion. Our task-category success rates and interaction steps are reported in Appendix A.4.1.

ScienceWorld. ScienceWorld (Wang et al., 2022) is an interactive text environment for performing elementary science tasks. The benchmark covers a range of scientific topics, including changes in states of matter, electrical conductivity, and biological processes. Agents must carry out procedures in the environment rather than only answer factual questions. The original benchmark provides normalized scores based on required goals and optional subgoals, allowing partial progress to receive credit. Such scores differ from task success rates: the success rates reported in this paper concern completed tasks rather than average partial-progress scores. Task-level results are provided in Appendix A.4.1.

Table 2: Prompt used to rank the shared skill library for each ALFWorld task instance. The placeholders are instantiated with the task type, task name, task goal, and the names and descriptions of all available skills.  
Task Specific Skill Ranking Prompt   
You are an expert in embodied AI agent task decomposition for the   
ALFWorld household environment.   
Task Instance   
Task type: {task\_type}   
Task name: {task\_name}   
Goal: {query}   
Available Skills ({num\_skills} total)   
{skills\_text}   
Instruction   
Rank ALL {num\_skills} skills by their relevance to completing this   
specific task instance: “{query}”.   
Assign each skill a relevance score from 1 (not relevant at all) to   
10 (essential/critical for this task).   
Consider which skills would be directly invoked during execution,   
which are supportive or auxiliary, and which are irrelevant.   
Return a JSON array sorted from MOST relevant (highest score) to   
LEAST relevant (lowest score).   
Output Format   
[   
{"skill\_name": "xxx", "relevance\_score": 10, "reason": "brief   
reason"},   
{"skill\_name": "xxx", "relevance\_score": 8, "reason": "brief   
reason"},   
]   
Return ONLY the JSON array, with no markdown fences or additional   
text.

WebShop. WebShop (Yao et al., 2022) is a simulated shopping environment in which agents search for products, inspect product information, select options, and make a purchase according to a natural-language instruction. Its reward evaluates the purchased product against requested at tributes, options, price, and product type. The original benchmark distinguishes task score, defined as 100 times mean reward, from success rate, defined as the proportion of episodes receiving reward 1. The WebShop results in Table 1 use success rate, consistent with the main evaluation.

## A.3.2 EVALUATION METRICS

The main evaluation reports task success rates across ALFWorld, ScienceWorld, and WebShop. Success rate denotes the proportion of evaluated tasks completed successfully. For each model and skill condition, the macro-average in Table 1 is the arithmetic mean of the three environment-level success rates, giving each environment equal weight. Task-level interaction steps are additionally reported for ALFWorld and ScienceWorld.

## A.3.3 SKILL RANKING PROMPT

As described in Section 5.1, Qwen3.8-Max ranks the shared skill library for each task instance using its type, name, and goal. Text and Visual use the same ranked skills and procedural content, supplied either as language tokens or as rendered skill cards. Table 2 presents the ALFWorld ranking prompt, whose placeholders contain task information and the available skill names and descriptions.

Table 3: Success rate (%) across different ALFWorld task types under the default skill setting. Overall denotes the overall success rate. Higher is better (↑). The best result for each task type within each backbone is highlighted in bold.
<table><tr><td>Setting</td><td>Overall ↑</td><td>Pick ↑</td><td>Clean ↑</td><td>Heat ↑</td><td>Cool ↑</td><td>Pick2↑</td><td>Examine ↑</td></tr><tr><td colspan="8">Qwen3.5-9B</td></tr><tr><td>No Skill</td><td>15.67</td><td>33.33</td><td>6.45</td><td>4.35</td><td>4.76</td><td>11.76</td><td>38.89</td></tr><tr><td>Text Skill</td><td>66.42</td><td>66.67</td><td>65.62</td><td>57.14</td><td>77.27</td><td>56.25</td><td>73.68</td></tr><tr><td>Visual Skill</td><td>70.15</td><td>91.67</td><td>64.29</td><td>50.00</td><td>77.27</td><td>88.24</td><td>52.63</td></tr><tr><td colspan="8">Qwen3.5-27B</td></tr><tr><td>No Skill</td><td>58.96</td><td>87.50</td><td>80.65</td><td>4.35</td><td>14.29</td><td>82.35</td><td>83.33</td></tr><tr><td>Text Skill</td><td>78.36</td><td>91.67</td><td>80.65</td><td>65.22</td><td>95.24</td><td>70.59</td><td>61.11</td></tr><tr><td>Visual Skill</td><td>80.60</td><td>91.67</td><td>93.55</td><td>56.52</td><td>95.24</td><td>94.12</td><td>44.44</td></tr><tr><td colspan="8">Gemma4-31B</td></tr><tr><td>No Skill</td><td>62.69</td><td>95.83</td><td>70.97</td><td>0.00</td><td>76.19</td><td>64.71</td><td>66.67</td></tr><tr><td>Text Skill</td><td>90.30</td><td>100.00</td><td>93.55</td><td>78.26</td><td>95.24</td><td>94.12</td><td>77.78</td></tr><tr><td>Visual Skill</td><td>90.30</td><td>100.00</td><td>96.77</td><td>69.57</td><td>90.48</td><td>88.24</td><td>94.44</td></tr><tr><td colspan="8">Kimi-K2.5</td></tr><tr><td>No Skill</td><td>78.36</td><td>95.83</td><td>87.10</td><td>30.43</td><td>76.19</td><td>82.35</td><td>100.00</td></tr><tr><td>Text Skill</td><td>90.30</td><td>95.83</td><td>90.32</td><td>65.22</td><td>95.24</td><td>100.00</td><td>100.00</td></tr><tr><td>Visual Skill</td><td>91.79</td><td>100.00</td><td>93.55</td><td>65.22</td><td>95.24</td><td>100.00</td><td>100.00</td></tr></table>

Table 4: Average number of interaction steps across different ALFWorld task types under the default skill setting. Overall denotes the average number of interaction steps across all tasks. Lower is better (↓). The best result for each task type within each backbone is highlighted in bold.
<table><tr><td>Setting</td><td>Overall ↓</td><td>Pick↓</td><td>Clean↓</td><td>Heat↓</td><td>Cool ↓</td><td>Pick2↓</td><td>Examine ↓</td></tr><tr><td colspan="8">Qwen3.5-9B</td></tr><tr><td>No Skill</td><td>27.59</td><td>25.04</td><td>29.45</td><td>29.78</td><td>29.62</td><td>27.65</td><td>22.56</td></tr><tr><td>Text Skill</td><td>18.28</td><td>17.88</td><td>18.69</td><td>21.05</td><td>18.18</td><td>20.81</td><td>13.00</td></tr><tr><td>Visual Skill</td><td>17.75</td><td>9.96</td><td>19.25</td><td>21.83</td><td>17.91</td><td>16.94</td><td>20.74</td></tr><tr><td colspan="8">Qwen3.5-27B</td></tr><tr><td>No Skill</td><td>19.01</td><td>11.21</td><td>15.32</td><td>29.17</td><td>27.48</td><td>17.47</td><td>14.39</td></tr><tr><td>Text Skill</td><td>14.87</td><td>10.33</td><td>15.87</td><td>16.48</td><td>13.67</td><td>17.12</td><td>16.39</td></tr><tr><td>Visual Skill</td><td>14.66</td><td>11.67</td><td>13.16</td><td>19.87</td><td>11.76</td><td>13.18</td><td>19.39</td></tr><tr><td colspan="8">Gemma4-31B</td></tr><tr><td>No Skill</td><td>18.66</td><td>9.38</td><td>17.55</td><td>30.00</td><td>15.81</td><td>22.76</td><td>17.89</td></tr><tr><td>Text Skill</td><td>12.01</td><td>9.79</td><td>11.84</td><td>17.13</td><td>10.19</td><td>13.71</td><td>9.28</td></tr><tr><td>Visual Skill</td><td>11.78</td><td>8.04</td><td>12.81</td><td>13.96</td><td>9.14</td><td>12.71</td><td>14.39</td></tr><tr><td colspan="8">Kimi-K2.5</td></tr><tr><td>No Skill</td><td>16.76</td><td>9.75</td><td>16.35</td><td>26.26</td><td>17.29</td><td>20.24</td><td>10.78</td></tr><tr><td>Text Skill</td><td>12.75</td><td>8.50</td><td>13.13</td><td>22.26</td><td>10.67</td><td>14.29</td><td>6.61</td></tr><tr><td>Visual Skill</td><td>12.41</td><td>8.50</td><td>14.71</td><td>17.83</td><td>10.29</td><td>12.65</td><td>9.00</td></tr></table>

## A.4 SUPPLEMENTARY COMPLETE-AGENT EVALUATION RESULTS

This section supplements Section 5.1 with task-level success rates and interaction-step results for ALFWorld and ScienceWorld.

## A.4.1 TASK-LEVEL PERFORMANCE UNDER THE DEFAULT SKILL SETTING

The following tables break down the complete-agent results by task category for ALFWorld and ScienceWorld, reporting success rates and interaction steps.

Table 5: Success rate (%) across different ScienceWorld task types under the default skill setting. Overall denotes the overall success rate. Higher is better (↑). The best result for each task type within each backbone is highlighted in bold.
<table><tr><td>Setting</td><td>Overall ↑ State ↑</td><td></td><td>Elec. ↑</td><td>Class. ↑</td><td>Growth ↑ Chem. ↑</td><td></td><td>Life ↑</td><td>Stages ↑</td><td>Meas. ↑</td></tr><tr><td colspan="10">Qwen3.5-9B</td></tr><tr><td>No Skill</td><td>18.96</td><td>0.00</td><td>36.67</td><td>7.50</td><td>5.00</td><td>11.54</td><td>70.00</td><td>0.00</td><td>5.00</td></tr><tr><td>Text Skill</td><td>42.18</td><td>8.33</td><td>20.00</td><td>95.00</td><td>40.00</td><td>19.23</td><td>56.67</td><td>11.11</td><td>55.00</td></tr><tr><td>Visual Skill</td><td>45.02</td><td>5.56</td><td>46.67</td><td>92.50</td><td>25.00</td><td>30.77</td><td>60.00</td><td>22.22</td><td>45.00</td></tr><tr><td colspan="10">Qwen3.5-27B</td></tr><tr><td>No Skill</td><td>45.02</td><td>19.44</td><td>30.00</td><td>70.00</td><td>35.00</td><td>19.23</td><td>73.33</td><td>22.22</td><td>75.00</td></tr><tr><td>Text Skill</td><td>57.82</td><td>33.33</td><td>50.00</td><td>100.00</td><td>45.00</td><td>42.31</td><td>70.00</td><td>22.22</td><td>60.00</td></tr><tr><td>Visual Skill</td><td>59.72</td><td>27.78</td><td>56.67</td><td>100.00</td><td>45.00</td><td>53.85</td><td>73.33</td><td>22.22</td><td>60.00</td></tr><tr><td colspan="10">Gemma-4-31B</td></tr><tr><td>No Skill</td><td>40.76</td><td>41.67</td><td>46.67</td><td>50.00</td><td>0.00</td><td>7.69</td><td>63.33</td><td>22.22</td><td>70.00</td></tr><tr><td>Text Skill</td><td>58.29</td><td>36.11</td><td>33.33</td><td>100.00</td><td>40.00</td><td>57.69</td><td>73.33</td><td>44.44</td><td>55.00</td></tr><tr><td>Visual Skill</td><td>59.72</td><td>41.67</td><td>56.67</td><td>92.50</td><td>40.00</td><td>50.00</td><td>73.33</td><td>44.44</td><td>50.00</td></tr><tr><td colspan="10">Kimi-K2.5</td></tr><tr><td>No Skill</td><td>48.82</td><td>36.11</td><td>30.00</td><td>77.50</td><td>25.00</td><td>30.77</td><td>73.33</td><td>11.11</td><td>70.00</td></tr><tr><td>Text Skill</td><td>50.71</td><td>22.22</td><td>40.00</td><td>95.00</td><td>35.00</td><td>50.00</td><td>60.00</td><td>0.00</td><td>55.00</td></tr><tr><td>Visual Skill</td><td>52.13</td><td>27.78</td><td>30.00</td><td>87.50</td><td>45.00</td><td>34.62</td><td>76.67</td><td>44.44</td><td>55.00</td></tr></table>

Table 6: Average number of interaction steps across different ScienceWorld task types under the default skill setting. Overall denotes the average number of interaction steps across all tasks. Lower is better (↓). The best result for each task type within each backbone is highlighted in bold.
<table><tr><td>Setting</td><td>Overall ↓ State ↓</td><td></td><td>Elec. ↓</td><td>Class. ↓</td><td>Growth↓Chem.↓</td><td></td><td>Life ↓</td><td>Stages ↓</td><td>Meas. ↓</td></tr><tr><td colspan="10">Qwen3.5-9B</td></tr><tr><td>No Skill</td><td>17.04</td><td>21.62</td><td>22.57</td><td>12.25</td><td>22.80</td><td>25.36</td><td>4.97</td><td>21.02</td><td>9.80</td></tr><tr><td>Text Skill</td><td>15.36</td><td>20.22</td><td>22.67</td><td>7.53</td><td>19.00</td><td>24.77</td><td>4.37</td><td>7.67</td><td>15.40</td></tr><tr><td>Visual Skill</td><td>15.95</td><td>20.23</td><td>24.60</td><td>9.38</td><td>22.45</td><td>23.02</td><td>3.67</td><td>7.56</td><td>14.95</td></tr><tr><td colspan="10">Qwen3.5-27B</td></tr><tr><td>No Skill</td><td>15.83</td><td>13.78</td><td>26.27</td><td>9.85</td><td>20.85</td><td>20.62</td><td>5.97</td><td>18.67</td><td>18.10</td></tr><tr><td>Text Skill</td><td>14.28</td><td>17.30</td><td>20.57</td><td>7.00</td><td>19.25</td><td>26.05</td><td>3.30</td><td>9.44</td><td>12.35</td></tr><tr><td>Visual Skill</td><td>13.80</td><td>17.62</td><td>20.80</td><td>7.35</td><td>18.90</td><td>23.28</td><td>3.30</td><td>5.67</td><td>11.25</td></tr><tr><td colspan="10">Gemma-4-31B</td></tr><tr><td>No Škill</td><td>16.23</td><td>15.15</td><td>25.93</td><td>10.90</td><td>24.70</td><td>19.01</td><td>6.53</td><td>18.89</td><td>15.50</td></tr><tr><td>Text Skill</td><td>14.99</td><td>18.73</td><td>23.93</td><td>7.55</td><td>20.45</td><td>22.31</td><td>3.33</td><td>6.56</td><td>16.05</td></tr><tr><td>Visual Skill</td><td>14.60</td><td>17.62</td><td>23.53</td><td>8.07</td><td>19.35</td><td>20.96</td><td>3.33</td><td>6.47</td><td>16.30</td></tr><tr><td colspan="10">Kimi-K2.5</td></tr><tr><td>No Škill</td><td>16.54</td><td>17.47</td><td>27.47</td><td>8.45</td><td>24.30</td><td>20.77</td><td>5.50</td><td>18.33</td><td>17.15</td></tr><tr><td>Text Skill</td><td>15.95</td><td>21.61</td><td>19.60</td><td>9.07</td><td>23.20</td><td>23.27</td><td>3.37</td><td>11.44</td><td>18.15</td></tr><tr><td>Visual Skill</td><td>16.06</td><td>20.56</td><td>20.13</td><td>9.75</td><td>23.05</td><td>26.27</td><td>3.80</td><td>10.11</td><td>15.30</td></tr></table>

ALFWorld. Tables 3 and 4 report success rates and interaction steps across the six ALFWorld task categories. The task-level breakdown shows that the performance difference between Text and Visual is not uniform across categories. Visual’s relative advantage varies with the task category, indicating that the aggregate result reflects different contributions from individual task types. The interaction-step results provide a complementary view of the number of steps used under the two representations.

ScienceWorld. Tables 5 and 6 report the corresponding task-level results for ScienceWorld. The subtasks contributing to the aggregate Text–Visual performance difference vary across model backbones. Thus, the task-level pattern is model-dependent, even when the aggregate comparison favors the same representation. The success-rate and interaction-step breakdowns show how the representation differences vary across individual scientific tasks.

## A.5 SUPPLEMENTARY REPRESENTATION-DEPENDENT EXPOSURE ANALYSES

This section supplements Section 5.2 with the rasterized representation comparison, repeated-signal readout, executable action selection, paired rollouts, layer-wise readout, and component interventions. Unless otherwise specified, these analyses use the 165 decision states and exposure-controlled variants described in Appendix A.1. The executable-action experiment uses a separate set of states, described below.

## A.5.1 RASTERIZED REPRESENTATION INTERVENTION

This experiment examines whether the attenuation observed under Visual input also occurs without the additional structure of the skill-card layout.

Skill Representations. We compare three representations of the same procedural content. Text supplies the skill as language tokens. Rasterized renders the same wording and ordering as a plain monospaced image without additional structural formatting. Structural Visual uses a separately implemented structured skill-card format based on the main-experiment design. Comparing Text with Rasterized assesses the representation change without additional skill-card structure. Comparing Rasterized with Structural Visual assesses the additional effect of that layout.

Exposure Sensitivity. For this analysis, we use the summed action-token log-probability ${ \widetilde { S } } ,$ rather than the length-normalized score S:

$$
\widetilde { S } _ { i } ( a ; k , r ) = \sum _ { t = 1 } ^ { | a | } \log p _ { \theta } \left( a _ { t } \mid x _ { i } ^ { ( k , r ) } , a _ { < t } \right) .\tag{8}
$$

For each state and representation, exposure sensitivity is estimated using the linear fit

$$
\widetilde { S } _ { i } ( a _ { i } ^ { D } ; k , r ) = \alpha _ { i , r } + \beta _ { i , r } k + \epsilon _ { i , k , r } ,\tag{9}
$$

where $\beta _ { i , r }$ summarizes the change in distractor score with exposure over $k \in \{ 0 , 1 , 2 , 4 , 8 \}$

For Qwen3.5-9B, mean exposure sensitivity decreases from 0.741 under Text to 0.467 under Raster ized input, and increases to 0.552 under Structural Visual input. Gemma-4-31B-it shows the same ordering, with corresponding values of 2.657, 1.283, and 1.361. Substantial attenuation is thus already observed under plain image rendering, without the structured skill-card layout. Figure 8 reports the results for Qwen3.5-27B and Gemma-4-31B-it.

## A.5.2 REPEATED SIGNAL READOUT

This probe examines how repeated procedural information influences target-word prediction under visual encoding by replacing action prediction with a direct target-word readout.

Setup and Scoring. We use variants in which the target object word occurs $k \in \{ 0 , 4 , 8 \}$ times and prompt the model to output the most frequent object in the skill. The target word is scored at a fixed answer position. Let $x _ { i , \mathrm { r e a d } } ^ { ( k , r ) }$ denote the input containing the skill and the readout prompt, and let the target word w tokenize into $\left( u _ { 1 } , \ldots , u _ { m } \right)$ . Its sequence log-probability is

$$
R _ { i } ( w ; k , r ) = \sum _ { j = 1 } ^ { m } \log p _ { \theta } \left( u _ { j } \mid x _ { i , { \mathrm { r e a d } } } ^ { ( k , r ) } , u _ { < j } \right) .\tag{10}
$$

The corresponding target-token-sequence probability is

$$
P _ { i } ^ { \mathrm { r e a d } } ( w ; k , r ) = \exp ( R _ { i } ( w ; k , r ) ) .\tag{11}
$$

This metric measures support for the target word at the designated answer position; it is not an exact-match accuracy measure for the most-frequent-object task.

![](images/99a8b4154e54b69b8b39952e40fc8a718ab90d734b373b284004de2f28d3928a.jpg)  
(a) Qwen3.5-27B.

![](images/69cbc6c7c4e74bacf73d6c29fafe0e80dfe1d493d0de09f95958d333ee8fa2f0.jpg)  
(b) Gemma-4-31B-it.  
Figure 8: Additional Results for the Rasterized Representation Intervention. Each panel reports state-level exposure sensitivity under Text, Rasterized, and Structural Visual representations. Central markers and whiskers denote the mean and 95% confidence interval.

Readout Results and Control. Figure 3b reports results for Qwen3.5-9B and Gemma-4-31Bit. Target-word probability increases with exposure under both Text and Visual, indicating that repetition continues to influence target-word prediction after visual encoding. This does not establish equal accessibility across representations or rule out partial degradation of the signal.

An unrelated frequency-matched control word is scored using the same procedure at $k \in \{ 4 , 8 \}$ The control probabilities remain approximately unchanged in the reported comparison, providing a check against a general increase in output confidence.

## A.5.3 EXECUTABLE ACTION SELECTION

This experiment tests whether exposure-dependent preference shifts extend to freely generated ac tions when the competing action is admissible in the current environment state.

Decision States and Competing Actions. We construct a separate set of 185 decision states from 64 successful ALFWorld trajectories. For each state, $a _ { i } ^ { C }$ denotes the reference-trajectory action, and $a _ { i } ^ { \mathrm { { c o m p } } }$ denotes a task-relevant competing action selected from the current admissible action set. The competing action is executable but differs from the reference action.

Exposure-Controlled Generation. The context contains five ranked skills. We manipulate the keyword associated with the competing action over $k \in \{ 0 , 1 , 2 , 4 , 8 \}$ using fixed insertion slots, while preserving content outside those slots and matching word counts across variants. The model freely generates the next action using greedy decoding under Text or Visual input.

For representation r, the competing-action selection rate is

$$
R _ { r } ( k ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { I } \Big [ \hat { a } _ { i , r } ^ { ( k ) } = a _ { i } ^ { \mathrm { c o m p } } \Big ] ,\tag{12}
$$

where $N = 1 8 5$ and $\hat { a } _ { i , r } ^ { ( k ) }$ is the generated action. Figure 3c reports the selection-rate results for Qwen3.5-9B and Gemma-4-31B-it.

Aggregation and Statistical Analysis. We summarize the aggregate relationship between exposure and selection rate using

$$
R _ { r } ( k ) = \alpha _ { r } + \beta _ { r } k + \epsilon _ { r , k } .\tag{13}
$$

The reported aggregate fits yield a Text slope of 0.0091 $( R ^ { 2 } = 0 . 9 5 )$ and a Visual slope of 0.0026 $( R ^ { 2 } = 0 . 4 8 )$ . The paired slope difference is 0.0065, with a bootstrap 95% confidence interval of

Table 7: Paired Rollout Results for Qwen3.5-9B on ALFWorld under the Text Skill Representation. The Text condition is evaluated at $k \in \{ 0 , 4 , 8 \}$ . Success rates and mean continuation steps are computed over all 185 decision states for each branch and exposure level. $\Delta L$ denotes the competing-minus-correct continuation-step difference among pairs for which both branches successfully complete the task. Confidence intervals use 4,000 game-clustered bootstrap resamples, and the final column reports the exact McNemar test for paired rollout success. The Pooled row aggregates the both-success pairs across the three exposure levels.
<table><tr><td rowspan="2">k</td><td colspan="2">Success Rate (%)</td><td colspan="2">Mean Steps</td><td colspan="2">Both-Success Analysis</td><td rowspan="2">McNemar</td></tr><tr><td>Correct</td><td>Competing</td><td>Correct</td><td>Competing</td><td>Pairs</td><td>∆L [95% CI]</td></tr><tr><td>0</td><td>30.27</td><td>28.11</td><td>7.60</td><td>9.09</td><td>25</td><td>+2.52 [1.48, 3.97]</td><td>0.694</td></tr><tr><td>4</td><td>40.54</td><td>30.27</td><td>8.09</td><td>8.95</td><td>33</td><td>+2.97 [1.69, 4.75]</td><td>0.025</td></tr><tr><td>8</td><td>34.05</td><td>26.49</td><td>8.06</td><td>8.44</td><td>28</td><td>+1.36 [0.48, 2.14]</td><td>0.081</td></tr><tr><td>Pooled</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>86</td><td>+2.31 [1.61, 3.21]</td><td>N/A</td></tr></table>

Table 8: Paired Rollout Results for Qwen3.5-9B on ALFWorld under the Visual Skill Representation. The Visual condition is evaluated at $k \in \{ 0 , 4 , 8 \}$ using the structured skill-card representation. Success rates and mean continuation steps are computed over all 185 decision states for each branch and exposure level. $\Delta L$ denotes the competing-minus-correct continuation-step difference among pairs for which both branches successfully complete the task. Confidence intervals use 4,000 gameclustered bootstrap resamples, and the final column reports the exact McNemar test for paired rollout success. The Pooled row aggregates the both-success pairs across the three exposure levels.
<table><tr><td rowspan="2">k</td><td colspan="2">Success Rate (%)</td><td colspan="2">Mean Steps</td><td colspan="2">Both-Success Analysis</td><td rowspan="2">McNemar p</td></tr><tr><td>Correct</td><td>Competing</td><td>Correct</td><td>Competing</td><td>Pairs</td><td>∆L [95% CI]</td></tr><tr><td>0</td><td>25.41</td><td>31.35</td><td>16.32</td><td>17.06</td><td>24</td><td>+2.17 [0.13, 4.27]</td><td>0.185</td></tr><tr><td>4</td><td>41.08</td><td>31.35</td><td>15.08</td><td>16.97</td><td>36</td><td>+3.44 [1.75, 5.41]</td><td>0.030</td></tr><tr><td>8</td><td>39.46</td><td>31.89</td><td>16.31</td><td>16.39</td><td>38</td><td>+1.63 [0.10, 3.27]</td><td>0.081</td></tr><tr><td>Pooled</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>98</td><td>+2.43 [1.26, 3.90]</td><td>N/A</td></tr></table>

[0.0011, 0.0120]. These fits summarize the aggregate selection-rate trend and do not imply monotonic switching at every decision state.

## A.5.4 PAIRED ROLLOUT EVALUATION

This evaluation examines the downstream consequences of executing the competing action rather than the reference action from a matched initial state.

Paired Continuations. For each evaluated state, we create two environment branches. The reference branch first executes $a _ { i } ^ { C }$ , while the competing branch first executes $a _ { i } ^ { \mathrm { { c o m p } } }$ . Both branches then continue with unrestricted greedy action generation under the same skill representation and exposure condition within the pair.

Rollouts terminate upon task success, an unparsable action, or exhaustion of the state-dependent budget

$$
B _ { i } = \mathrm { m i n } \big ( 4 0 , B _ { i } ^ { \mathrm { e x p e r t } } + 2 0 \big ) ,\tag{14}
$$

where $B _ { i } ^ { \mathrm { e x p e r t } }$ is the number of remaining steps in the reference trajectory from decision step i onward.

Conditional Step Difference. For pairs in which both branches succeed, we compute

$$
\Delta L _ { i , r } ( k ) = L _ { i , r , k } ^ { \mathrm { c o m p } } - L _ { i , r , k } ^ { C } ,\tag{15}
$$

![](images/abaceee63535dfc7c0ce37094b62248540b91f79c25385aba95ab17869b33e43.jpg)  
(a) Qwen3.5-27B, Text.

![](images/d998f62a0f0dbeaaba49314ee2c2a1bd7f7d54a688d31857c36c5a95c38b98bd.jpg)  
(b) Qwen3.5-27B, Visual.

![](images/acbaf7ae59127c331e34d4479135662e54c372d936ffaba9541e6840f5e613c3.jpg)  
(c) Gemma-4-31B-it, Text.

![](images/71d8f373219ca6f1ea2e481980ea5d15165f4d88ffd36d0801aba710ddb3c9dd.jpg)  
(d) Gemma-4-31B-it, Visual.  
Figure 9: Additional Layer-Wise Readouts of Exposure-Dependent Action Evidence. Each heatmap reports the mean change in distractor action evidence relative to the corresponding $k = 0$ condition across model depth.

where $L _ { i , r , k } ^ { \mathrm { c o m p } }$ and $L _ { i , r , k } ^ { C }$ are the recorded rollout step counts for the competing and reference branches. A positive value indicates additional steps in the competing branch. This metric is conditional on both branches succeeding and does not capture differences in failure rates.

Tables 7 and 8 report the paired outcomes under Text and Visual, respectively. The paired-rollout comparison assesses the consequences of the initial action choice; it does not by itself estimate the effect of exposure on complete-agent performance.

## A.5.5 LAYER-WISE READOUT OF EXPOSURE-DEPENDENT ACTION EVIDENCE

We use intermediate-layer readouts to examine how exposure-related action evidence varies across model depth.

Experimental Setup. We perform teacher-forced forward passes for the reference and distractor actions. Let $q _ { i , k , r , t }$ denote the input position immediately preceding action token $a _ { t }$ in the corresponding sequence. At language-model layer l, the hidden state at this prediction position is passed through the model’s final normalization and language-model head. The layer-wise action score is

$$
S _ { i , l } ( a ; k , r ) = \frac { 1 } { \left| a \right| } \sum _ { t = 1 } ^ { \left| a \right| } \log \left[ \mathrm { s o f t m a x } \left( W _ { \mathrm { L M } } \mathrm { N o r m } \left( h _ { i , k , r , q _ { i , k , r , t } } ^ { ( l ) } \right) \right) \right] _ { a _ { t } } .\tag{16}
$$

Here, $W _ { \mathrm { L M } }$ denotes the language-model output projection, and Norm denotes the final normalization. These scores are intermediate readouts under teacher forcing rather than probabilities from independent generation at each layer.

Exposure-Dependent Readout. The mean change in distractor evidence relative to zero exposure is

$$
H _ { r } ( l , k ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left[ S _ { i , l } ( a _ { i } ^ { D } ; k , r ) - S _ { i , l } ( a _ { i } ^ { D } ; 0 , r ) \right] ,\tag{17}
$$

where $n = 1 6 5$ . The readouts show stronger exposure-dependent distractor evidence under Text in the later layers. Figure 9 reports additional results for Qwen3.5-27B and Gemma-4-31B-it. For Qwen3.5-27B, the measured evidence remains weak through much of the network and increases primarily in the final layers. Weak early-layer readouts do not establish the absence of exposurerelated information in those representations.

## A.5.6 MATCHED COMPONENT INTERVENTION

We use matched activation interventions to test how selected component outputs affect the exposurerelated change in distractor score.

Intervention Protocol. For each state and representation, we cache component activations from a high-exposure donor pass $( k = 8 )$ . During the matched zero-exposure host pass $( k = 0 )$ , the target component outputs at the action-sequence positions are replaced with the cached donor activations.

Using the summed action-token log-probability ${ \widetilde { \cal S } } ,$ the original exposure effect is

$$
G _ { i , r } ^ { D } = \widetilde { S } _ { i } ( a _ { i } ^ { D } ; 8 , r ) - \widetilde { S } _ { i } ( a _ { i } ^ { D } ; 0 , r ) .\tag{18}
$$

For a patched component c, the recovered score change is

$$
\begin{array} { r } { R _ { i , r , c } ^ { D } = \widetilde { S } _ { i } ^ { \mathrm { p a t c h } ( c ) } ( a _ { i } ^ { D } ; 0 , r ) - \widetilde { S } _ { i } ( a _ { i } ^ { D } ; 0 , r ) , } \end{array}\tag{19}
$$

where $\widetilde { S } _ { i } ^ { \mathrm { p a t c h } ( c ) }$ denotes the summed action-token log-probability from the patched host pass. $\mathbf { A g } \mathbf { - }$ gregate recovery is expressed as

$$
\mathrm { R e c o v e r y } _ { r , c } = 1 0 0 \frac { \mathbb { E } _ { i } [ R _ { i , r , c } ^ { D } ] } { \mathbb { E } _ { i } [ G _ { i , r } ^ { D } ] } .\tag{20}
$$

This ratio is defined when the mean original effect is nonzero and is not constrained to the interval [0, 100]. In the state-level scatter plots, the horizontal coordinate is $G _ { i , r } ^ { D }$ and the vertical coordinate is $R _ { i , r , c } ^ { D }$ . The diagonal indicates full recovery of the original score change, and the horizontal zero line indicates no recovered score change.

Component Comparisons. We use the cross-position mixing intervention as a full-recovery reference and the MLP intervention as the primary diagnostic comparison. In Qwen3.5-9B, transplanting high-exposure $\mathrm { \bf { M L P } }$ outputs produces a score change opposite to the original exposure effect under both representations. In Gemma-4-31B-it, the same intervention produces a change in the same direction as the original effect. These findings describe the effects of the specified activation transplants. Additional Qwen3.5-27B results show the same qualitative MLP intervention pattern as Qwen3.5-9B and are presented in Figure 10.

![](images/c3cd602f42faef817da8b9ac43c34f5be3511ea8bdc5e514881bb81cccd5f4d1.jpg)  
Figure 10: Additional Qwen3.5- 27B Results for the Matched Component Intervention. Each point denotes one ALFWorld decision state. The diagonal denotes full recovery, while the horizontal dashed line denotes zero recovered score change.

## A.6 EXAMPLES OF RENDERED VISUAL SKILLS

Figures 11 and 12 illustrate the structured visual format used to present procedural content. The ALFWorld example organizes an appliance-related skill into functional regions, while the WebShop example presents product-search procedures and output constraints.

![](images/d711f8b99ebc627125a8145a36b407dd8082f1d206d30e147801645aeaa71c35.jpg)  
Figure 11: Case Study of a Rendered Visual Skill from ALFWorld. The Appliance Preparer skill visually segregates procedural content into explicit functional regions, mapping operational conditions, execution steps, and expected outputs into distinct spatial blocks.

![](images/b9678255e598cda9519d581333b7b5ba2428c63c9e7f92051c510e878d447af7.jpg)  
Figure 12: Case Study of a Rendered Visual Skill from WebShop. The Search Formulator skill presents product-parsing procedures and output-format constraints in a structured visual layout.