# INLINE MEMORY MEETS REUSABLE SKILLS: MEMORY-CENTRIC FRAMEWORK FOR VISION-LANGUAGE-ACTION MODEL

Zaijing Li<sup>1,2</sup> Rui Shao<sup>1</sup> Bing Hu<sup>1</sup> Haoyu Zhang<sup>1,2</sup> Dongmei Jiang<sup>2</sup> Liqiang Nie<sup>1</sup>

<sup>1</sup>Harbin Institute of Technology (Shenzhen) <sup>2</sup>Pengcheng Laboratory

## ABSTRACT

Vision-Language-Action (VLA) models have shown strong promise for generalpurpose robotic manipulation, yet adapting them to new tasks and domains remains inefficient: existing methods often rely on parameter tuning, incurring substantial costs and risking catastrophic forgetting of previously learned tasks. To address this, we propose Optimus-R, a memory-centric VLA framework that formulates robotic adaptation as explicit query-skill memory tuning. Optimus-R introduces: (i) An Inline Memory Interface for skill extraction. It inserts learnable memory tokens into the VLA prefix stream, allowing the backbone to derive control-aware query and skill representations within the native action-conditioning pathway. (ii) A Query-Skill Memory Bank for skill learning. It externalizes skills into query prototypes for deciding what to retrieve and skill values for specifying how to act, supporting skill reuse and expansion with limited parameter updates. (iii) A lightweight Bridge-and-Adapt mechanism for skill updating. It aligns targetdomain queries and skills with the existing memory space through a lightweight adapter and residual memory updates. Experiments on in-domain adaptation, cross-domain adaptation, and lifelong learning show that Optimus-R enables dataefficient skill learning while mitigating catastrophic forgetting.

## 1 INTRODUCTION

Vision-Language-Action (VLA) models (Black et al., 2024; Kim et al., 2024; 2025; Zhang et al., 2025; Shi et al., 2025) have become a promising foundation for general-purpose robotic manipulation (Chi et al., 2025; Jiang et al., 2022), benefiting from large-scale pretraining on diverse robotic datasets (Liu et al., 2023a; Khazatsky et al., 2024; Mees et al., 2022; Chen et al., 2025). However, deployment in the real world (Zitkovich et al.,

![](images/805b7e0e0fc177bf2986c58ff1f4c3b485cd7cda75e77b24ae1a8b43a4f5d4cf.jpg)  
Figure 1: Comparison between parameter-centric adaptation and our memory-centric adaptation framework.

2023; Bjorck et al., 2025; Intelligence et al., 2025a) requires robots to adapt to new tasks, objects, scenes, and domains with limited demonstrations. Most current VLAs perform such adaptation by updating model parameters (Hu et al., 2021; Kim et al., 2025; Wen et al., 2025; Wang et al., 2026b). While effective, this parameter-centric paradigm makes every new skill an implicit change to the model weights, leading to high adaptation cost and possible interference with previously learned tasks. This motivates a crucial question: can a VLA acquire, store, and reuse skills through an explicit memory interface, rather than repeatedly updating its model parameters?

As shown in Fig. 1, a memory-centric VLA must address three coupled issues. First, skills should be derived from the VLA model itself rather than attached as detached context: external retrieval memories are only weakly coupled with the VLA action-conditioning pathway, which limits their ability to encode control-relevant information (Li et al., 2026; Temiraliev et al., 2026). Second, skill should be organized explicitly rather than absorbed into model weights: parameter updates make newly acquired behaviors hard to retrieve, expand, or reuse, and may interfere with previous skill during continual learning (Wang et al., 2026a). Third, skills should remain retrievable under domain shift: visual or dynamics changes can move target-domain observations away from the source-domain memory space, causing mismatches between queries and stored skills (Xu et al., 2026). These issues motivate a unified view of adaptation as skill extraction, skill learning, and skill updating.

To this end, we propose Optimus-R, a memory-centric framework for VLA adaptation. Optimus-R introduces : (i) Inline Memory Interface for skill extraction. It inserts learnable memory tokens into the VLA prefix stream, allowing the backbone to derive control-aware query and skill representations within the native policy pathway. (ii) Query-Skill Memory Bank for skill learning. It externalizes skills into query prototypes for deciding what to retrieve and skill values for specifying how to act, enabling new tasks to be learned by reusing, updating, or appending memory entries rather than repeatedly tuning model parameters. (iii) Bridge-and-Adapt mechanism for skill updating. It aligns target-domain queries and skill values to the existing memory space through lightweight adapter and residual memory updates, improving skill retrieval under domain shift.

We conduct experiments on in-domain adaptation, cross-domain adaptation, and in-domain lifelong learning. On LIBERO, CALVIN, and RoboTwin 2.0, Optimus-R consistently improves over the strong π<sub>0.5</sub> (Intelligence et al., 2025b) baseline under low-data regimes: with only 30% training data, it improves the LIBERO average success rate by 16.5%, the CALVIN average completion length by 0.42, and the RoboTwin 2.0 Hard success rate by 5.0%. For cross-domain transfer, Optimus-R achieves a 33.3% real-world success rate with only 20 demonstrations per task, outperforming π<sub>0.5</sub> by 13.9%. In lifelong learning, Optimus-R improves 20-demo new-task success by 11.0% over π<sub>0.5</sub>, and reduces average forgetting from 15.0% to 10.0% after 80-demo adaptation.

Our main contributions are summarized as follows:

• We propose Optimus-R, a memory-centric VLA framework that treats robotic adaptation as explicit query-skill memory tuning, reducing the need for repeated model weight updates during continual learning.

• We introduce an Inline Memory Interface that inserts learnable memory tokens into the VLA prefix stream, enabling the backbone to produce control-aware query and skill representations within the policy’s native action-conditioning pathway.

• We design a Query-Skill Memory Bank that externalizes skills as decoupled query prototypes and skill values. This bank-centered design supports skill reuse, skill expansion, and continual task adaptation with limited parameter updates.

• We develop a Bridge-and-Adapt strategy that performs lightweight memory-space alignment for cross-domain transfer, improving the retrievability of source-domain skills under target-domain shifts such as sim-to-real adaptation.

## 2 RELATED WORK

Vision-Language-Action Models. Vision-Language-Action models (Zhang et al., 2025; Kim et al., 2024; Pertsch et al., 2025; Qu et al., 2025; Bu et al., 2025; Song et al., 2025b;a; Li et al., 2025b; 2023b; Liu et al., 2025) are a central paradigm for robotic manipulation, spanning early multimodal policies and large generalist robot models pretrained on diverse datasets (Liu et al., 2023a; Mees et al., 2022; Chen et al., 2025; Zitkovich et al., 2023; Intelligence et al., 2025b). Recent work improves VLA policies through continuous action modeling (Li et al., 2024a; Kim et al., 2025), diffusion or flow-based generation (Intelligence et al., 2025b; Chi et al., 2025), action tokenization (Pertsch et al., 2025), and efficient adaptation (Wen et al., 2025; Wang et al., 2026b). However, most downstream adaptation remains parameter-centric, learning new skills through model fine-tuning. Optimus-R keeps the pretrained VLA largely reusable, adapting through an explicit query-skill memory interface.

![](images/52f017e99581ef52d29b20a3777d685e9b08647c96e29afc46552ed27e91b169.jpg)  
Figure 2: Overview of Optimus-R. Given an observation and language instruction, Optimus-R inserts learnable memory tokens into the VLA prefix stream, allowing the backbone to produce inline memory states aligned with the policy pathway. A dual-head memory readout derives a query embedding $q _ { t }$ for retrieval and a skill embedding $s _ { t }$ for behavior representation, with $s _ { t }$ supervised by a skill decoder during training. The Query-Skill Memory Bank stores reusable latent skill slots as query prototypes and skill values $\{ ( p _ { j } ^ { q } , \bar { p } _ { j } ^ { s } ) \}$ , which can be retrieved, updated, or expanded through clustering and residual memory updates. The retrieved skill is projected into VLA-compatible skill tokens and used to condition the flow policy for action-chunk generation.

Memory for Embodied Agents. Memory mechanisms extend context and support decision making in embodied agents (Zhang et al., 2024; He et al., 2024; Song et al., 2024; Li et al., 2025c; Zhu et al., 2024; Xie et al., 2024; Li et al., 2024b). In robotics, recent memory-augmented methods store demonstrations, experiences, scene histories, or external knowledge to improve long-horizon planning and manipulation (Shi et al., 2025; Li et al., 2025a; Lin et al., 2025; Li et al., 2026). These methods regard memory as external context or episodic retrieval. In contrast, Optimus-R injects memory into the VLA prefix stream and maintains a persistent query-skill bank for efficient adaptation.

Few-Shot and Continual Adaptation. Foundation models adapt via in-context learning (Brown et al., 2020), prompt tuning (Lester et al., 2021), LoRA (Hu et al., 2021), adapters (Zhang et al., 2023), or frozen-backbone transfer (Li et al., 2023a; Dai et al., 2023; Liu et al., 2023b; Karamcheti et al., 2024). Existing VLAs improve data efficiency but still encode new skills in model parameters (Kim et al., 2025; Wen et al., 2025; Wang et al., 2026b). Optimus-R instead externalizes skills into a globa memory bank, updating only lightweight alignment modules, prototype residuals, and bank entries.

## 3 METHOD

We present Optimus-R, a memory-centric adaptation framework for vision-language-action (VLA) models. It combines an Inline Memory Interface for extracting query and skill representations, a Query-Skill Memory Bank for storing reusable skills, and Bridge-and-Adapt training for adapting retrieval and policy conditioning from limited demonstrations.

## 3.1 OVERVIEW

As shown in Fig. 2, given observation $O _ { t }$ and instruction L at step t, the backbone $F _ { \theta }$ processes multimodal input concatenated with learnable memory tokens E<sup>mem</sup>:

$$
[ H _ { t } ^ { \mathrm { v l } } ; M _ { t } ] = F _ { \boldsymbol { \theta } } \big ( { \boldsymbol { O } } _ { t } , L ; E ^ { \mathrm { m e m } } \big ) .\tag{1}
$$

Here, $H _ { t } ^ { \mathrm { v l } }$ and $M _ { t }$ are the updated multimodal and memory states, and θ denotes $\mathrm { V L A }$ parameters. Subsequently, the Query Encoder reads $M _ { t }$ to form query $q _ { t }$ , which retrieves skill $\hat { s } _ { t }$ from the Query-

Skill Memory Bank. The retrieved skill is optionally adapted before projection into skill tokens $S _ { t }$ Together with $H _ { t } ^ { \mathrm { v l } }$ , these skill tokens condition the Flow Policy $\pi _ { \theta }$ to predict ${ \tilde { A } } _ { t }$ for the target action chunk $A _ { t } = a _ { t : t + H - 1 }$ , where $a _ { t }$ is an action and H the horizon:

$$
\tilde { A } _ { t } = \pi _ { \theta } ( H _ { t } ^ { \mathrm { v l } } , S _ { t } ) .\tag{2}
$$

## 3.2 INLINE MEMORY INTERFACE

The Inline Memory Interface derives query and skill representations from the shared memory states $M _ { t } .$ . To separate retrieval context from action-relevant skill content, two independent attention pooling operators read these states:

$$
\bar { m } _ { t } ^ { q } = \mathrm { A t t n P o o l } _ { q } ( M _ { t } ) , \qquad \bar { m } _ { t } ^ { s } = \mathrm { A t t n P o o l } _ { s } ( M _ { t } ) ,\tag{3}
$$

followed by a Query Encoder and a Skill Encoder:

$$
q _ { t } = f _ { q } ( \bar { m } _ { t } ^ { q } ) \in \mathbb R ^ { d _ { q } } , \qquad s _ { t } = f _ { s } ( \bar { m } _ { t } ^ { s } ) \in \mathbb R ^ { d _ { s } } .\tag{4}
$$

The query embedding $q _ { t }$ is used only for memory retrieval, and the Query Encoder supplies the retrieval query at inference. The skill embedding $s _ { t }$ is trained to retain action-relevant information and serves as the value representation stored in the Query-Skill Memory Bank. During interface pretraining, s<sub>t</sub> is supervised by a lightweight Skill Decoder:

$$
\hat { A } _ { t } ^ { s } = D _ { s } ( s _ { t } ) ,\tag{5}
$$

which reconstructs the future action chunk. This auxiliary objective grounds the skill space in executable behavior. For policy conditioning, the Token Projector $T _ { \psi }$ maps the skill embeddings into VLA-compatible tokens. Section 3.4 specifies the conditioning latent and optimization objective for each stage.

## 3.3 QUERY-SKILL MEMORY BANK

The Query-Skill Memory Bank stores the extracted skills as decoupled key-value prototypes:

$$
\boldsymbol { B } = \{ ( p _ { j } ^ { q } , p _ { j } ^ { s } , n _ { j } , \mathcal { E } _ { j } ) \} _ { j = 1 } ^ { K } ,\tag{6}
$$

where $p _ { j } ^ { q }$ is a query prototype, $p _ { j } ^ { s }$ is the corresponding skill prototype, $n _ { j }$ is the support count, and $\mathcal { E } _ { j }$ is a compact set of latent query–skill pairs used for replay. The initial bank is constructed from $\left( q _ { t } , s _ { t } \right)$ pairs extracted by the Inline Memory Interface. To reduce redundancy, Optimus-R applies motion-aware downsampling and normalized-progress stratified sampling before clustering. It then performs Skill Clustering by first grouping samples in query space and further splitting groups whose skill variance is high. This produces prototypes that are both retrievable by state-task context and coherent in action space.

For retrieval, the current query is first aligned to the memory coordinate system:

$$
q _ { t } ^ { \prime } = A _ { \mathrm { m e m } } q _ { t } .\tag{7}
$$

Let $\mathcal { N } _ { t }$ be the top- $K _ { r }$ prototypes under similarity to $q _ { t } ^ { \prime } .$ Optimus-R computes the retrieval weights as

$$
\alpha _ { t j } = \frac { \exp \big ( \sin ( q _ { t } ^ { \prime } , p _ { j } ^ { q } + \Delta p _ { j } ^ { q } ) / { \tau _ { \mathrm { m e m } } } \big ) } { \sum _ { \ell \in \mathcal { N } _ { t } } \exp \big ( \sin ( q _ { t } ^ { \prime } , p _ { \ell } ^ { q } + \Delta p _ { \ell } ^ { q } ) / { \tau _ { \mathrm { m e m } } } \big ) } , \quad j \in \mathcal { N } _ { t } ,\tag{8}
$$

and aggregates the retrieved skill by

$$
\hat { s } _ { t } = \sum _ { j \in \mathcal { N } _ { t } } \alpha _ { t j } ( p _ { j } ^ { s } + \Delta p _ { j } ^ { s } ) .\tag{9}
$$

The residuals $\Delta p _ { j } ^ { q }$ and $\Delta p _ { j } ^ { s }$ implement Skill Update without overwriting the base prototypes. Thus, adaptation changes the local memory coordinates while preserving the reusable skills.

## 3.4 BRIDGE-AND-ADAPT TRAINING

Stage A learns the interface and constructs the initial bank. Building on this bank, Stage B bridges cross-domain shifts, while Stage C supports in-domain lifelong memory adaptation.

Table 1: Performance comparison on LIBERO (Liu et al., 2023a), CALVIN (Mees et al., 2022), and RoboTwin 2.0 (Chen et al., 2025). We report the average success rate on each LIBERO task suite, the average completion length (Avg. Len) on CALVIN $( \mathrm { A B C }  \mathrm { D } )$ , and the success rate under the Hard setting on RoboTwin 2.0. Data denotes the training-data ratio. <sup>†</sup> denotes reproduced results.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Data</td><td colspan="5">LIBERO</td><td colspan="2">CALVIN RoboTwin 2.0</td></tr><tr><td>Spatial</td><td>Object</td><td>Goal</td><td>Long</td><td>Avg.</td><td>Avg. Len</td><td>Hard</td></tr><tr><td>DP (Chi et al., 2025)</td><td>100%</td><td>78.3</td><td>92.5</td><td>68.3</td><td>50.5</td><td>72.4</td><td>0.56</td><td>1.6</td></tr><tr><td>ACT (Zhao et al., 2023)</td><td>100%</td><td>一</td><td></td><td></td><td></td><td></td><td></td><td>3.5</td></tr><tr><td>DP3 (Ze et al., 2024)</td><td>100%</td><td>一</td><td></td><td></td><td></td><td>1</td><td></td><td>5.2</td></tr><tr><td>RDT (Liu et al., 2024)</td><td>100%</td><td></td><td></td><td></td><td></td><td></td><td>一</td><td>18.4</td></tr><tr><td>MemoryVLA (Shi et al., 2025)</td><td>100%</td><td>98.4</td><td>98.4</td><td>96.4</td><td>93.4</td><td>96.7</td><td></td><td></td></tr><tr><td>OpenVLA-OFT (Kim et al., 2025)</td><td>100%</td><td>97.6</td><td>98.4</td><td>97.9</td><td>94.5</td><td>97.1</td><td>4.10</td><td></td></tr><tr><td>DreamVLA (Zhang et al., 2025)</td><td>100%</td><td>97.5</td><td>94.0</td><td>89.5</td><td>89.5</td><td>92.6</td><td>4.44</td><td></td></tr><tr><td>VLA-Adapter (Wang et al., 2026b)</td><td>100%</td><td>97.8</td><td>99.2</td><td>97.2</td><td>95.0</td><td>97.3</td><td>4.42</td><td></td></tr><tr><td>ReconVLA (Song et al., 2025b)</td><td>100%</td><td></td><td></td><td></td><td></td><td></td><td>3.95</td><td></td></tr><tr><td>OpenVLA (Kim et al., 2024)</td><td>100%</td><td>84.7</td><td>88.4</td><td>79.2</td><td>53.7</td><td>76.5</td><td>3.27</td><td></td></tr><tr><td>UniVLA (Bu et al., 2025)</td><td>100%</td><td>95.4</td><td>98.8</td><td>93.6</td><td>94.0</td><td>95.4</td><td>3.80</td><td></td></tr><tr><td>π0 (Black et al., 2024)</td><td>100%</td><td>96.8</td><td>98.8</td><td>95.8</td><td>85.2</td><td>94.2</td><td>3.92</td><td>22.9</td></tr><tr><td> $\pi _ { 0 . 5 } { } ^ { \dagger }$  (Intelligence et al., 2025b)</td><td>30%</td><td>76.6</td><td>75.0</td><td>74.8</td><td>62.0</td><td>72.1</td><td>2.09</td><td>9.8</td></tr><tr><td>Optimus-R</td><td>30%</td><td>89.6</td><td>91.8</td><td>87.0</td><td>86.0</td><td>88.6</td><td>2.51</td><td>14.8</td></tr><tr><td> $\pi _ { 0 . 5 } { } ^ { \dagger }$  (Intelligence et al., 2025b)</td><td>50%</td><td>92.8</td><td>94.2</td><td>93.2</td><td>88.0</td><td>92.1</td><td>3.14</td><td>19.6</td></tr><tr><td>Optimus-R</td><td>50%</td><td>97.0</td><td>97.0</td><td>96.8</td><td>91.4</td><td>95.6</td><td>3.57</td><td>21.6</td></tr><tr><td> $\pi _ { 0 . 5 } { } ^ { \dagger }$  (Intelligence et al., 2025b)</td><td>70%</td><td>96.2</td><td>95.6</td><td>96.8</td><td>91.4</td><td>95.0</td><td>3.88</td><td>30.1</td></tr><tr><td>Optimus-R</td><td>70%</td><td>98.2</td><td>97.8</td><td>98.0</td><td>92.2</td><td>96.6</td><td>3.94</td><td>36.4</td></tr><tr><td> $\pi _ { 0 . 5 } { } ^ { \dagger }$  (Intelligence et al., 2025b)</td><td>100%</td><td>98.8</td><td>98.2</td><td>98.0</td><td>92.4</td><td>96.9</td><td>4.26</td><td>46.6</td></tr><tr><td>Optimus-R</td><td>100%</td><td>99.2</td><td>99.4</td><td>98.4</td><td>94.2</td><td>97.8</td><td>4.38</td><td>55.0</td></tr></table>

Stage A: interface pretraining. Stage A trains the memory tokens, Query Encoder, Skill Encoder, Skill Decoder, Token Projector, and the last layers of the VLA backbone. During this stage, the Flow Policy is conditioned on skill tokens $S _ { t } = T _ { \psi } \ ' ( s _ { t } )$ projected from the encoded skill. The policy and interface are jointly optimized with

$$
{ \mathcal { L } } _ { \mathrm { p r e } } = { \mathcal { L } } _ { \mathrm { f m } } + \lambda _ { q } { \mathcal { L } } _ { q } ^ { \mathrm { n c e } } + \lambda _ { s } { \mathcal { L } } _ { \mathrm { s k i l l } } , \qquad { \mathcal { L } } _ { \mathrm { s k i l l } } = \mathrm { H u b e r } ( D _ { s } ( s _ { t } ) , A _ { t } ) .\tag{10}
$$

Here, ${ \mathcal L } _ { \mathrm { f m } }$ is the original VLA flow-matching loss. The query contrastive loss $\mathcal { L } _ { q } ^ { \mathrm { n c e } }$ uses samples from the same task and nearby normalized progress as positives, together with local neighboring states from the same trajectory. After Stage A, the Inline Memory Interface, Token Projector, Skill Decoder, and backbone are frozen, and the initial memory bank is built.

Stage B: bridge adaptation. For cross-domain streams such as sim-to-real transfer, visual, sensory, or dynamics shifts can change the query and skill representations associated with the same behavior. To accommodate these shifts, Stage B unfreezes the backbone and updates it alongside $A _ { \mathrm { m e m } } ,$ τ<sub>mem</sub>, and active prototype residuals, while keeping the learned interface modules, Token Projector, and base memory prototypes fixed. Here, query alignment maps target-domain queries into the bank’s retrieval coordinates. When the skill-space mismatch is large, a low-rank residual Skill Adapter further corrects the retrieved skill toward the current encoded behavior target:

$$
\begin{array} { r } { \tilde { s } _ { t } = A _ { s } ( \hat { s } _ { t } ) = \hat { s } _ { t } + U _ { s } V _ { s } ^ { \top } \hat { s } _ { t } , \qquad U _ { s } , V _ { s } \in \mathbb R ^ { d _ { s } \times r } , \quad r \ll d _ { s } . } \end{array}\tag{11}
$$

The residual is zero-initialized so that the adapter initially preserves $\hat { s } _ { t } .$ . The corrected skill then supplies policy tokens $S _ { t } = T _ { \psi } ( \tilde { s } _ { t } )$ in Stages B/C and deployment, with $\tilde { s } _ { t } = \hat { s } _ { t }$ when the adapter is inactive. During Stage B, the adapter is optimized through both policy and alignment losses; it is then kept fixed in Stage C.

Stage C: memory adaptation. For in-domain continual learning, Stage C keeps the VLA backbone and Inline Memory Interface frozen and confines adaptation to $A _ { \mathrm { { m e m } } } , \ \tau _ { \mathrm { { m e m } } } ,$ active prototype residuals, and the memory bank. When an incoming sample’s aligned query is close to an existing prototype, Optimus-R updates the corresponding residual and support statistics. Otherwise, the sample enters a candidate buffer. A new prototype is appended only when a cluster of buffered samples has sufficient support and low query–skill variance. Further details are provided in Appendix C.4.

![](images/d2a9f76ab6566164087e663befc07dacacf83c31a38cd1ee28ce801178246296.jpg)  
Figure 3: Overview of Real-World tasks. We classify tasks into two categories (A and B) for lifelong setting: the model first learns categories A tasks and is then incrementally adapted to categories B tasks. Details are provided in the Appendix.

Table 2: Cross-Domain evaluation across 9 Real-World tasks. Samples denotes the number of training demonstrations per task. We report the success rate (%) for each task. Compared to the baselines, Optimus-R demonstrates superior performance across different data scales.
<table><tr><td>Method</td><td>Samples</td><td>R.O.</td><td>B.S.</td><td>P.P.</td><td>P.T.</td><td>P.C.</td><td>B.P.</td><td>M.C.</td><td>S.P.P.</td><td>B.Sq.</td><td>Avg</td></tr><tr><td>RDT (Liu et al., 2024)</td><td>80</td><td>20.0</td><td>40.0</td><td>35.0</td><td>50.0</td><td>35.0</td><td>30.0</td><td>20.0</td><td>5.0</td><td>0.0</td><td>26.1</td></tr><tr><td>OpenVLA (Kim et al., 2024)</td><td>80</td><td>15.0</td><td>35.0</td><td>45.0</td><td>55.0</td><td>40.0</td><td>20.0</td><td>15.0</td><td>20.0</td><td>10.0</td><td>28.3</td></tr><tr><td>OpenVLA-OFT (Kim et al., 2025)</td><td>80</td><td>65.0</td><td>75.0</td><td>80.0</td><td>70.0</td><td>55.0</td><td>45.0</td><td>50.0</td><td>20.0</td><td>30.0</td><td>54.4</td></tr><tr><td> $\pi _ { 0 }$  (Black et al., 2024)</td><td>80</td><td>75.0</td><td>75.0</td><td>80.0</td><td>85.0</td><td>45.0</td><td>55.0</td><td>50.0</td><td>30.0</td><td>20.0</td><td>57.2</td></tr><tr><td> $\pi _ { 0 . 5 } { } ^ { \dagger }$  (Intelligence et al., 2025b)</td><td>20</td><td>25.0</td><td>20.0</td><td>35.0</td><td>25.0</td><td>30.0</td><td>25.0</td><td>10.0</td><td>5.0</td><td>0.0</td><td>19.4</td></tr><tr><td>Optimus-R</td><td>20</td><td>40.0</td><td>40.0</td><td>55.0</td><td>40.0</td><td></td><td>35.0 35.0</td><td>30.0</td><td>10.0</td><td>15.0</td><td>33.3</td></tr><tr><td> $\pi _ { 0 . 5 } { } ^ { \dagger }$  (Intelligence et al., 2025b)</td><td>50</td><td>65.0</td><td>55.0</td><td>70.0</td><td>65.0</td><td>40.0</td><td>40.0</td><td>30.0</td><td>20.0</td><td>15.0</td><td>44.4</td></tr><tr><td>Optimus-R</td><td>50</td><td>75.0</td><td>70.0</td><td>80.0</td><td>80.0</td><td>60.0</td><td>65.0</td><td>50.0</td><td>40.0</td><td>35.0</td><td>61.7</td></tr><tr><td> $\pi _ { 0 . 5 } { } ^ { \dagger }$  (Intelligence et al., 2025b)</td><td>80</td><td>75.0</td><td>75.0 80.0 85.0</td><td></td><td></td><td></td><td>60.0 60.0</td><td>50.0</td><td>45.0</td><td>40.0</td><td>63.3</td></tr><tr><td>Optimus-R</td><td>80</td><td>85.0</td><td>90.0</td><td>90.0</td><td>85.0</td><td>70.0</td><td>65.0</td><td>60.0</td><td>55.0</td><td>50.0</td><td>72.2</td></tr></table>

To preserve previously learned retrieval associations during these updates, each batch supplements current-task samples with stored latent query–skill pairs in the alignment term. These exemplars provide skill-space supervision for previously observed behaviors without requiring raw images or action trajectories. Meanwhile, the flow-matching loss uses only current-task demonstrations to supervise action generation on the new task.

Stages B and C update different components but optimize the same memory adaptation objective:

$$
\mathcal { L } _ { \mathrm { l i f e } } = \mathcal { L } _ { \mathrm { f m } } + \lambda _ { \mathrm { a l i g n } } \left. \tilde { s } _ { t } - \mathrm { s g } ( s _ { t } ) \right. _ { 2 } ^ { 2 } + \lambda _ { \mathrm { r e g } } \sum _ { j \in \mathcal { A } _ { t } } \left( \Vert \Delta p _ { j } ^ { q } \Vert _ { 2 } ^ { 2 } + \Vert \Delta p _ { j } ^ { s } \Vert _ { 2 } ^ { 2 } \right) ,\tag{12}
$$

where $\boldsymbol { A } _ { t }$ denotes the active retrieved prototypes and $\operatorname { s g } ( \cdot )$ stops gradients through the encoded skill target $s _ { t } .$ . The flow-matching term supervises memory conditioning through action generation, while the alignment term directly supervises the retrieved skill after any adapter correction. The residual regularizer limits prototype drift during these updates.

## 4 EXPERIMENTS

We evaluate Optimus-R through three research questions: Q1. Does Optimus-R improve sample efficiency on diverse simulation benchmarks? Q2. Can Optimus-R adapt simulation-acquired skills to the real-world domain with limited demonstrations? Q3. Can Optimus-R acquire new real-world skills while preserving previously learned ones?

## 4.1 IMPLEMENTATION DETAILS

We initialize Optimus-R from pretrained $\pi _ { 0 . 5 }$ (Intelligence et al., 2025b) and add memory tokens, the query-skill interface, and the skill bank. We use $m = 4$ memory tokens, dimensions $d _ { q } = d _ { s } = 2 5 6$ $n _ { s } ~ = ~ 4$ skill tokens per retrieved prototype, top-2 retrieval, and an initial bank of $K _ { 0 } ~ = ~ 1 2 8$ prototypes. Training uses 8× NVIDIA A800 GPUs, a global batch size of 256, and 30,000 steps. Further details are in the Appendix.

## 4.2 IN-DOMAIN ADAPTATION

Experimental Setup. We evaluate sample efficiency on LIBERO (Liu et al., 2023a), CALVIN (Mees et al., 2022), and RoboTwin 2.0 (Chen et al., 2025), adapting the same Stage-A pretrained model with 30%, 50%, 70%, and 100% of the demonstrations. We report average success over the four LIBERO suites (Spatial, Object, Goal, and Long; 500 rollouts per suite), average completed sequence length on CALVIN ABC → D (500 rollouts), and RoboTwin 2.0 Hard success rates (100 rollouts per task).

![](images/a7b3e68f3dcd2a2b4b398eddf243ee1994e1ae7df19979a6774a7c8c9b64ab47.jpg)  
Figure 4: Real-world in-domain lifelong learning results. The model first learns $N _ { \mathrm { o l d } } = 4$ tasks and is then incrementally adapted to $N _ { \mathrm { n e w } } = 5$ tasks. The first row reports the post-adaptation success rates on the previously learned tasks, together with the average forgetting rate, where lower forgetting indicates better retention. The second row reports the success rates on the newly learned tasks after incremental adaptation. Optimus-R achieves higher new-task performance while better preserving old-task performance.

Results and Analysis. Table 1 shows consistent gains over $\pi _ { 0 . 5 }$ (Intelligence et al., 2025b) across all three benchmarks and data regimes. With 30% training data, Optimus-R improves LIBERO average success by 16.5 percentage points, with a particularly large gain on LIBERO-Long. This suggests that reusable skill prototypes benefit long-horizon tasks with temporally structured behaviors. With full data, Optimus-R achieves the best LIBERO average and RoboTwin 2.0 Hard performance among the compared methods, while remaining competitive on CALVIN. These results support our design goal: using target-domain demonstrations to refine and reuse externalized skill memory rather than relearning the policy through repeated global parameter updates.

## 4.3 CROSS-DOMAIN ADAPTATION

Experimental Setup. We transfer a RoboTwin 2.0-pretrained model to nine real-world tasks with 20, 50, and 80 demonstrations per task, reporting average success over 20 rollouts per task. The platform is a 14-DoF bimanual GALAXEA R1 Lite robot with wrist-mounted and third-person cameras at 224 × 224 resolution. Figure 3 shows the two task categories; further details are in the Appendix. This setting tests whether the bank provides reusable simulation-acquired manipulation skills that can be grounded in a new domain with limited real-world data.

Results and Analysis. Table 2 shows that Optimus-R reaches 33.3% success with 20 demonstrations per task, exceeding $\pi _ { 0 . 5 }$ by 13.9 percentage points. This indicates useful transferable skill priors under limited target-domain supervision. With 80 demonstrations, Optimus-R achieves the highest average success among the compared methods. Gains also extend to long-horizon and sequential tasks such as S.P.P. and B.Sq., where adaptation is more sensitive to visual and dynamics shifts. These results suggest that the bank preserves simulation-acquired manipulation structure, while Bridge-and-Adapt aligns target-domain queries and skill values with the existing memory space.

## 4.4 IN-DOMAIN LIFELONG LEARNING

Experimental Setup. We evaluate lifelong learning by first training on $N _ { \mathrm { o l d } } = 4$ real-world tasks (Category A), then adapting to $N _ { \mathrm { n e w } } = 5$ new tasks (Category B) under the same 20 / 50 / 80 demonstration settings. We report new-task success, old-task success after adaptation, and average forgetting as evaluation metrics:

$$
\mathcal { F } = \frac { 1 } { N _ { \mathrm { o l d } } } \sum _ { i = 1 } ^ { N _ { \mathrm { o l d } } } \left( S _ { i } ^ { \mathrm { b e f o r e } } - S _ { i } ^ { \mathrm { a f t e r } } \right) ,\tag{13}
$$

Table 3: Adaptation cost and LIBERO performance. GPU hours correspond to the 100% data setting. Parameter counts are rounded; M and B denote millions and billions.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Total params</td><td rowspan="2">Added params</td><td rowspan="2">Trainable params</td><td rowspan="2">GPU hours</td><td rowspan="2">Inference Hz</td><td colspan="4">Average SR (%) by data ratio</td></tr><tr><td>30%</td><td>50%</td><td>70%</td><td>100%</td></tr><tr><td>π0.5 (Full FT)</td><td>3.6B</td><td></td><td>3.6B</td><td>560</td><td>17</td><td>72.1</td><td>92.1</td><td>95.0</td><td>96.9</td></tr><tr><td>π0.5 (LoRA)</td><td>3.9B</td><td>0.3B</td><td>0.3B</td><td>128</td><td>12</td><td>71.5</td><td>86.9</td><td>91.2</td><td>96.5</td></tr><tr><td>OpenVLA-OFT (LoRA)</td><td>7.8B</td><td>0.3B</td><td>0.3B</td><td>320</td><td>10</td><td>66.4</td><td>83.5</td><td>89.1</td><td>96.4</td></tr><tr><td>Optimus-R</td><td>3.6B</td><td>7M</td><td>0.3B</td><td>86</td><td>14</td><td>88.6</td><td>95.6</td><td>96.6</td><td>97.8</td></tr></table>

Table 4: Ablation success rates (%) on RoboTwin 2.0 (Hard) and nine real-world tasks under different data budgets. Variant definitions are given in Sec. 4.6.
<table><tr><td rowspan="2">Model Variant</td><td colspan="2">RoboTwin 2.0 (In-Domain)</td><td colspan="3">Real-World 9 Tasks (Cross-Domain)</td></tr><tr><td>50% Data</td><td>100% Data</td><td>20 Demos</td><td>50 Demos</td><td>80 Demos</td></tr><tr><td>w/o Inline Memory</td><td>16.5</td><td>45.2</td><td>20.6</td><td>48.3</td><td>60.0</td></tr><tr><td>Coupled Query-Skill</td><td>18.2</td><td>48.0</td><td>22.8</td><td>50.0</td><td>62.8</td></tr><tr><td>w/o Prototype Residuals</td><td>17.0</td><td>46.5</td><td>17.8</td><td>46.7</td><td>58.3</td></tr><tr><td>w/o Stage-B Bridge</td><td>21.4</td><td>53.8</td><td>12.8</td><td>35.6</td><td>48.3</td></tr><tr><td>Optimus-R (Full)</td><td>21.6</td><td>55.0</td><td>33.3</td><td>61.7</td><td>72.2</td></tr></table>

where $S _ { i } ^ { \mathrm { b e f o r e } }$ and S<sup>after</sup> denote the success rate of old task i before and after new-task adaptation.   
This setting tests whether Optimus-R can acquire new skills while preserving learned tasks.

Results and Analysis. With 20 demonstrations per new task, Optimus-R achieves 21.0% success, exceeding $\pi _ { 0 . 5 }$ by 11.0 percentage points (Fig. 4). After 80-demo adaptation, it retains 77.5% old-task success, with 10.0% forgetting versus 15.0% for both OpenVLA-OFT and $\pi _ { 0 . 5 }$ . Together, these results answer Q3: prototype expansion and localized residual updates support acquisition of new bimanual skills while limiting interference with previously learned behaviors.

## 4.5 COMPARISON WITH PARAMETER-EFFICIENT BASELINES

We compare Optimus-R with full fine-tuning and LoRA adaptation of $\pi _ { 0 . 5 }$ (Intelligence et al., 2025b), and LoRA adaptation of OpenVLA-OFT (Kim et al., 2025), on LIBERO at four training-data ratios. Table 3 reports parameter counts, average success rates, training cost (GPU hours), and inference frequency (Hz). Training costs correspond to the 100% data setting.

Optimus-R achieves the highest success rate at every data ratio, with the largest gains under limited data. At full data, it reaches 97.8% success in 86 GPU hours, versus 96.5% in 128 GPU hours for $\pi _ { 0 . 5 } – \mathrm { L o R A } .$ , with both reporting approximately 0.3B trainable parameters. This comparison indicates improved adaptation efficiency at a comparable trainable parameter budget. Its inference frequency of 14 Hz exceeds both LoRA baselines (12 Hz and 10 Hz).

## 4.6 ABLATION STUDY

Experimental Setup. We ablate four components on RoboTwin 2.0 and real-world tasks (Table 4): Inline Memory (replaced by detached prompts), query–skill separation (replaced by a shared latent), prototype residuals (no localized $\Delta p$ updates), and Stage-B Bridge (memory expansion without coordinate alignment). More ablation experiments are provided in Appendix.

Results and Analysis. Every ablation reduces success, supporting the joint role of control-aware memory representations and localized adaptation. The clearest domain-dependent effect comes from Stage-B Bridge: removing it has a small in-domain impact but lowers 20-demo real-world success from 33.3% to 12.8%. This contrast suggests that expanding memory alone is insufficient under substantial domain shift; aligning target-domain queries and skills with the existing bank is critical for transfer.

![](images/de4d85919f2b6e415b3ce6dee7fe89af50b9fb04d2f4c61b22eb8edbd211fc19.jpg)  
Figure 5: Qualitative results of Optimus-R on real-world tasks. Each row shows a rollout sequence from left to right. Rows 1 and 3 correspond to learned tasks, where Optimus-R executes previously trained pick-and-place behaviors, including placing a red apple onto a plate and placing bottles onto a plate. Rows 2 and 4 show zero-shot task variations, where the model transfers the learned manipulation structure to new object-receptacle combinations, such as placing a green apple into a basket and placing blocks into a bowl.

## 4.7 QUALITATIVE ANALYSIS

Figure 5 contrasts learned tasks with zero-shot variations in object appearance, category, receptacle, and task semantics. Successful transfer to new object–receptacle combinations suggests reuse of manipulation skills beyond fixed training pairs. Given a new instruction and observation, query representations retrieve compatible skill values from the Query-Skill Memory Bank, which are projected into VLA-compatible skill tokens to guide action generation without additional parameter updates. This behavior is consistent with our design: skills extracted through inline memory are organized as query-skill entries and reused across task variations.

## 4.8 RETRIEVAL-SPACE VISUALIZATION

We record the top-1 prototype selected during offline inference on 4,000 observations from eight LIBERO tasks, using 20 demonstrations per task and 25 progress-stratified observations per demonstration. Figure 6 visualizes the associated aligned queries $q _ { t } ^ { \prime } ~ \in ~ \mathbb { R } ^ { 2 5 6 }$ , colored by task. The queries and their retrieved keys are ℓ<sub>2</sub>-normalized and jointly projected using t-SNE; only queries are displayed. The two bowl-to-plate tasks (T34, T32) occupy overlapping local neighborhoods, while soup-tobasket, sauce-to-basket, and multi-object collection (T24, T28, T5) share a broad region. Drawer opening (T19) and plate pushing (T15) form more distinct regions. This structure is consistent with related tasks sharing retrieval neighborhoods.

![](images/5866cf0a9ed5c2a7b4a6c1bdc4f31b97d313b5146a5ffcabaebe8025ed9c8833.jpg)  
Figure 6: Aligned queries colored by task. All 4,000 queries are shown; labels such as T34 denote task IDs. Prototype markers are omitted.

## 5 CONCLUSION

We presented Optimus-R, a memory-centric VLA framework that formulates robotic adaptation as explicit query-skill memory tuning. Optimus-R introduces three key designs: an Inline Memory Interface for deriving control-aware query and skill representations within the native VLA policy pathway, a Query-Skill Memory Bank for externalizing skills into reusable and expandable memory entries, and a lightweight Bridge-and-Adapt mechanism for aligning target-domain queries and skills with the existing memory space. Experiments across in-domain adaptation, cross-domain adaptation, and lifelong learning show that Optimus-R enables data-efficient adaptation, improves skill reuse, and reduces reliance on conventional fine-tuning.

## AI USE STATEMENT

We used generative AI tools solely to polish the manuscript’s language and improve clarity and readability. No generative AI tools were used for other aspects of this research, including research ideation, method development, experimental design, implementation, data generation or processing, or analysis and interpretation of results. The authors take full responsibility for the final content, including all scientific claims and conclusions.

## REPRODUCIBILITY STATEMENT

To facilitate reproducibility, we provide detailed implementation procedures and hyperparameter settings in Appendices C and D, covering the model architecture, stage-wise training, and memorybank construction and update rules. Additional experimental settings and evaluation protocols are described in the appendix.

## REFERENCES

Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, et al. GR00T N1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.

Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π<sub>0</sub>: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 2020.

Qingwen Bu, Yanting Yang, Jisong Cai, Shenyuan Gao, Guanghui Ren, Maoqing Yao, Ping Luo, and Hongyang Li. UniVLA: Learning to act anywhere with task-centric latent actions. arXiv preprint arXiv:2505.06111, 2025.

Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Zixuan Li, Qiwei Liang, Xianliang Lin, Yiheng Ge, Zhenyu Gu, et al. RoboTwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. arXiv preprint arXiv:2506.18088, 2025.

Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal ofRobotics Research, 44(10-11):1684–1704, 2025.

Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. InstructBLIP: towards general-purpose vision-language models with instruction tuning. In Proceedings of the 37th International Conference on Neural Information Processing Systems, pp. 49250–49267, 2023.

Bo He, Hengduo Li, Young Kyun Jang, Menglin Jia, Xuefei Cao, Ashish Shah, Abhinav Shrivastava, and Ser-Nam Lim. MA-LMM: Memory-augmented large multimodal model for long-term video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13504–13514, 2024.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

Physical Intelligence, Ali Amin, Raichelle Aniceto, Ashwin Balakrishna, Kevin Black, Ken Conley, Grace Connors, James Darpinian, Karan Dhabalia, Jared DiCarlo, et al. π<sup>∗</sup> : a VLA that learns from experience. arXiv preprint arXiv:2511.14759, 2025a.

Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. π : a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025b.

Yunfan Jiang, Agrim Gupta, Zichen Zhang, Guanzhi Wang, Yongqiang Dou, Yanjun Chen, Li Fei-Fei, Anima Anandkumar, Yuke Zhu, and Linxi Fan. VIMA: General robot manipulation with multimodal prompts. In NeurIPS 2022 Foundation Models for Decision Making Workshop, 2022.

Siddharth Karamcheti, Suraj Nair, Ashwin Balakrishna, Percy Liang, Thomas Kollar, and Dorsa Sadigh. Prismatic VLMs: Investigating the design space of visually-conditioned language models. In Forty-first International Conference on Machine Learning, 2024.

Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karam cheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, et al. DROID: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024.

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. OpenVLA: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.

Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645, 2025.

Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient prompt tuning. In Proceedings of the 2021 conference on empirical methods in natural language processing, pp. 3045–3059, 2021.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pp. 19730–19742. PMLR, 2023a.

Qixiu Li, Yaobo Liang, Zeyu Wang, Lin Luo, Xi Chen, Mozheng Liao, Fangyun Wei, Yu Deng, Sicheng Xu, Yizhong Zhang, et al. CogACT: A foundational vision-language-action model for synergizing cognition and action in robotic manipulation. arXiv preprint arXiv:2411.19650, 2024a.

Runhao Li, Wenkai Guo, Zhenyu Wu, Changyuan Wang, Haoyuan Deng, Zhenyu Weng, Yap-Peng Tan, and Ziwei Wang. MAP-VLA: Memory-augmented prompting for vision-language-action model in robotic manipulation. arXiv preprint arXiv:2511.09516, 2025a.

Wei Li, Renshan Zhang, Rui Shao, Jie He, and Liqiang Nie. CogVLA: Cognition-aligned vision-language-action model via instruction-driven routing & sparsification. arXiv preprint arXiv:2508.21046, 2025b.

Xinghang Li, Minghuan Liu, Hanbo Zhang, Cunjun Yu, Jie Xu, Hongtao Wu, Chilam Cheang, Ya Jing, Weinan Zhang, Huaping Liu, et al. Vision-language foundation models as effective robot imitators. arXiv preprint arXiv:2311.01378, 2023b.

Zaijing Li, Yuquan Xie, Rui Shao, Gongwei Chen, Dongmei Jiang, and Liqiang Nie. Optimus-1: Hybrid multimodal memory empowered agents excel in long-horizon tasks. arXiv preprint arXiv:2408.03615, 2024b.

Zaijing Li, Yuquan Xie, Rui Shao, Gongwei Chen, Dongmei Jiang, and Liqiang Nie. Optimus-2: Multimodal Minecraft agent with goal-observation-action conditioned policy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9039–9049, June 2025c.

Zaijing Li, Bing Hu, Rui Shao, Gongwei Chen, Dongmei Jiang, Pengwei Xie, Jianye Hao, and Liqiang Nie. Global prior meets local consistency: Dual-memory augmented vision-language-action model for efficient robotic manipulation. arXiv preprint arXiv:2602.20200, 2026.

Min Lin, Xiwen Liang, Bingqian Lin, Liu Jingzhi, Zijian Jiao, Kehan Li, Yuhan Ma, Yuecheng Liu, Shen Zhao, Yuzheng Zhuang, et al. EchoVLA: Robotic vision-language-action model with synergistic declarative memory for mobile manipulation. arXiv preprint arXiv:2511.18112, 2025.

Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. LIBERO: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36:44776–44791, 2023a.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in Neural Information Processing Systems, 36, 2023b.

Huihan Liu, Changyeon Kim, Bo Liu, Minghuan Liu, and Yuke Zhu. Pretrained vision-languageaction models are surprisingly resistant to forgetting in continual learning. arXiv preprint arXiv:2603.03818, 2026.

Jiaming Liu, Hao Chen, Pengju An, Zhuoyang Liu, Renrui Zhang, Chenyang Gu, Xiaoqi Li, Ziyu Guo, Sixiang Chen, Mengzhen Liu, et al. HybridVLA: Collaborative diffusion and autoregression in a unified vision-language-action model. arXiv preprint arXiv:2503.10631, 2025.

Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, and Jun Zhu. RDT-1B: a diffusion foundation model for bimanual manipulation. arXiv preprint arXiv:2410.07864, 2024.

Oier Mees, Lukas Hermann, Erick Rosete-Beas, and Wolfram Burgard. CALVIN: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks. IEEE Robotics and Automation Letters, 7(3):7327–7334, 2022.

Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. FAST: Efficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747, 2025.

Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, JiaYuan Gu, Bin Zhao, Dong Wang, et al. SpatialVLA: Exploring spatial representations for visual-languageaction model. arXiv preprint arXiv:2501.15830, 2025.

Hao Shi, Bin Xie, Yingfei Liu, Lin Sun, Fengrong Liu, Tiancai Wang, Erjin Zhou, Haoqiang Fan, Xiangyu Zhang, and Gao Huang. MemoryVLA: Perceptual-cognitive memory in vision-languageaction models for robotic manipulation. arXiv preprint arXiv:2508.19236, 2025.

Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun Guo, Tian Ye, Yanting Zhang, et al. MovieChat: From dense token to sparse memory for long video understanding. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18221–18232, 2024.

Wenxuan Song, Jiayi Chen, Pengxiang Ding, Han Zhao, Wei Zhao, Zhide Zhong, Zongyuan Ge, Jun Ma, and Haoang Li. Accelerating vision-language-action model integrated with action chunking via parallel decoding. arXiv preprint arXiv:2503.02310, 2025a.

Wenxuan Song, Ziyang Zhou, Han Zhao, Jiayi Chen, Pengxiang Ding, Haodong Yan, Yuxin Huang, Feilong Tang, Donglin Wang, and Haoang Li. ReconVLA: Reconstructive vision-language-action model as effective robot perceiver. arXiv preprint arXiv:2508.10333, 2025b.

Izat Temiraliev, Diji Yang, and Yi Zhang. Retrieval-augmented robots via retrieve-reason-act. arXiv preprint arXiv:2603.02688, 2026.

Xudong Wang, Zebin Han, Zhiyu Liu, Gan Li, Jiahua Dong, Baichen Liu, Lianqing Liu, and Zhi Han. Lifelong language-conditioned robotic manipulation learning. Proceedings of the AAAI Conference on Artificial Intelligence, 40(22):18629–18637, 2026a.

Yihao Wang, Pengxiang Ding, Lingxiao Li, Can Cui, Zirui Ge, Xinyang Tong, Wenxuan Song, Han Zhao, Wei Zhao, Pengxu Hou, et al. VLA-Adapter: An effective paradigm for tiny-scale vision-language-action model. Proceedings of the AAAI Conference on Artificial Intelligence, 40 (22):18638–18646, 2026b.

Junjie Wen, Yichen Zhu, Jinming Li, Minjie Zhu, Zhibin Tang, Kun Wu, Zhiyuan Xu, Ning Liu, Ran Cheng, Chaomin Shen, et al. TinyVLA: Towards fast, data-efficient vision-language-action models for robotic manipulation. IEEE Robotics and Automation Letters, 2025.

Quanting Xie, So Yeon Min, Tianyi Zhang, Kedi Xu, Aarav Bajaj, Ruslan Salakhutdinov, Matthew Johnson-Roberson, and Yonatan Bisk. Embodied-RAG: General non-parametric embodied memory for retrieval and generation. arXiv preprint arXiv:2409.18313, 2024.

Weisheng Xu, Jian Li, Yi Gu, Bin Yang, Haodong Chen, Shuyi Lin, Mingqian Zhou, Jing Tan, Qiwei Wu, Xiangrui Jiang, et al. Morphology-consistent humanoid interaction through robot-centric video synthesis. arXiv preprint arXiv:2603.19709, 2026.

Yanjie Ze, Gu Zhang, Kangning Zhang, Chenyuan Hu, Muhan Wang, and Huazhe Xu. 3D diffusion policy: Generalizable visuomotor policy learning via simple 3D representations. arXiv preprint arXiv:2403.03954, 2024.

Renrui Zhang, Jiaming Han, Chris Liu, Peng Gao, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, and Yu Qiao. LLaMA-Adapter: Efficient fine-tuning of language models with zero-init attention. arXiv preprint arXiv:2303.16199, 2023.

Wenyao Zhang, Hongsi Liu, Zekun Qi, Yunnan Wang, Xinqiang Yu, Jiazhao Zhang, Runpei Dong, Jiawei He, Fan Lu, He Wang, et al. DreamVLA: a vision-language-action model dreamed with comprehensive world knowledge. arXiv preprint arXiv:2507.04447, 2025.

Zeyu Zhang, Xiaohe Bo, Chen Ma, Rui Li, Xu Chen, Quanyu Dai, Jieming Zhu, Zhenhua Dong, and Ji-Rong Wen. A survey on the memory mechanism of large language model based agents. arXiv preprint arXiv:2404.13501, 2024.

Tony Z Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware. arXiv preprint arXiv:2304.13705, 2023.

Yichen Zhu, Zhicai Ou, Xiaofeng Mou, and Jian Tang. Retrieval-augmented embodied agents. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 17985–17995, 2024.

Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, et al. RT-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pp. 2165–2183. PMLR, 2023.

## A BROADER IMPACTS

Optimus-R aims to improve the data efficiency and reusability of Vision-Language-Action models by shifting robotic adaptation from repeated parameter updates to explicit query-skill memory tuning. This design may reduce the amount of task-specific data and computation required to adapt robots to new environments, thereby lowering the cost of deploying robotic systems. The memory-centric formulation also provides a more explicit interface for inspecting, reusing, and updating learned skills, which may support more maintainable lifelong robotic systems.

At the same time, improved adaptation efficiency can also increase the risk of deploying robots in insufficiently validated environments. Retrieved or reused skills may behave unexpectedly under unseen objects, unsafe scene configurations, or distribution shifts beyond those covered during adaptation. Moreover, real-world robotic data may contain sensitive visual information, and memory banks or stored exemplars should be managed with appropriate privacy and data-governance safeguards.

## B LIMITATION

Despite its effectiveness, Optimus-R has several limitations. First, the framework relies on a strong pretrained VLA backbone; when the base policy lacks the necessary manipulation primitives, memory retrieval and prototype expansion alone may be insufficient. Second, although the Query-Skill Memory Bank improves skill reuse, its quality depends on the coverage and clustering of the pretraining and adaptation data. Poorly organized or sparse memory entries may lead to suboptimal retrieval, especially for tasks with fine-grained temporal or contact-rich behaviors. Third, crossdomain transfer still requires a small amount of target-domain demonstrations and careful memoryspace alignment. The lightweight Bridge-and-Adapt mechanism mitigates representational drift, but it does not guarantee reliable transfer under large embodiment changes, severe visual shifts, or substantially different dynamics.

## C IMPLEMENTATION DETAILS

This appendix follows the notation and objectives in Secs. 3.2–3.4. In particular, $A _ { t }$ is the target action chunk, whereas $\boldsymbol { A } _ { t }$ is a set of active prototype indices. The encoded, retrieved, and adapter-corrected skill representations are denoted by $s _ { t } , \hat { s } _ { t } ,$ and $\tilde { s } _ { t } ,$ respectively. The implementation details below supplement, rather than redefine, the main-text equations.

## C.1 NOTATION AND FORWARD COMPUTATION

For each sample, the model receives a non-linguistic observation $O _ { t } = ( I _ { t } , x _ { t } )$ , comprising visual observations $I _ { t }$ and robot state $x _ { t } .$ , together with a language instruction $L$ . The target action chunk is $A _ { t } = a _ { t : t + H - 1 } \in \mathbb { R } ^ { H \times d _ { a } }$ , where H is the action horizon and $d _ { a }$ is the action dimension. We suppress the batch axis throughout the per-sample equations: $H _ { t } ^ { \mathrm { v l } } \ { \stackrel { \sim } { \in } } \ \mathbb { R } ^ { P \times d } , E ^ { \mathrm { m e m } } \in \mathbb { R } ^ { m \times d }$ , and $\boldsymbol { M _ { t } } ^ { \star } \in \mathbb { R } ^ { m \times d }$ . Here, $P$ is the number of original prefix tokens, m is the number of memory tokens, and d is the hidden dimension. In batched execution, a leading batch dimension of size $B$ is added, and the same learnable memory tokens are broadcast across samples.

The backbone computation is given by Eq. (1). The two attention poolers and the Query and Skill Encoders are defined in Eqs. (3) and (4). For branch $b \in \{ q , s \}$ , the attention pooler is implemented as

$$
\begin{array} { r l } & { \beta _ { t , i } ^ { b } = \frac { \displaystyle \exp \left( u _ { b } ^ { \top } W _ { b } M _ { t , i } ^ { \top } / \sqrt { d } \right) } { \displaystyle \sum _ { h = 1 } ^ { m } \exp \left( u _ { b } ^ { \top } W _ { b } M _ { t , h } ^ { \top } / \sqrt { d } \right) } , \qquad i = 1 , \ldots , m , } \\ & { \mathrm { A t t n P o o l } _ { b } ( M _ { t } ) = \displaystyle \sum _ { i = 1 } ^ { m } \beta _ { t , i } ^ { b } M _ { t , i } ^ { \top } \in \mathbb { R } ^ { d } , } \end{array}\tag{14}
$$

where $M _ { t , i } \in \mathbb { R } ^ { 1 \times d }$ is a memory-token row, $u _ { b } \in \mathbb { R } ^ { d }$ is a learnable pooling query, and $W _ { b } \in \mathbb { R } ^ { d \times d }$ is a projection matrix. The transpose in the weighted sum expresses the pooled state as a column

vector, consistently with the encoders in Eq. (4). The branch index b is distinct from the Skill Adapter rank r in Eq. (11).

The Skill Decoder reconstructs $A _ { t }$ from $s _ { t }$ according to Eq. (5); it provides the auxiliary skillreconstruction supervision in Eq. (10) and is not used during deployment. The Token Projector maps a skill representation into policy-conditioning tokens:

$$
T _ { \psi } : \mathbb { R } ^ { d _ { s } }  \mathbb { R } ^ { n _ { s } \times d } , \qquad S _ { t } = \{ T _ { \psi } ( s _ { t } ) , \quad \mathrm { S t a g e ~ A } ,\tag{15}
$$

Here, $\tilde { s } _ { t } = A _ { s } ( \hat { s } _ { t } )$ is the corrected retrieved skill from Eq. (11), with $\tilde { s } _ { t } = \hat { s } _ { t }$ when the adapter is inactive. The Flow Policy predicts $\tilde { A } _ { t } = \pi _ { \theta } ( H _ { t } ^ { \mathrm { v l } } , S _ { t } )$ as in Eq. (2). The prediction ${ \tilde { A } } _ { t }$ is distinct from the training target $A _ { t }$ and the Skill Decoder reconstruction $\hat { A } _ { t } ^ { s }$

## C.2 INITIAL QUERY-SKILL MEMORY BANK CONSTRUCTION

After Stage A, we freeze the trained interface and extract query-skill pairs from the pretraining data:

$$
\mathcal { D } _ { \mathrm { b a n k } } = \{ ( q _ { i } , s _ { i } , A _ { i } , \rho _ { i } ) \} _ { i = 1 } ^ { N } ,\tag{16}
$$

where i indexes a sample, $t _ { i }$ is its within-trajectory time index, and $\rho _ { i } = t _ { i } / T _ { \mathrm { t r a j } ( i ) }$ is its normalized progress. The denominator $T _ { \mathrm { t r a j } ( i ) }$ is the duration of the trajectory containing sample i. We denote the action chunk associated with this sample by $A _ { i } = ( a _ { i , 0 } , \dots , a _ { i , H - 1 } )$ , where the second subscript indexes an action within that chunk.

Motion-aware downsampling. To avoid over-representing static or redundant segments, we compute the action-variation score

$$
\nu _ { i } = \frac { 1 } { H - 1 } \sum _ { h = 1 } ^ { H - 1 } \left\| a _ { i , h } - a _ { i , h - 1 } \right\| _ { 2 } , \qquad H > 1 .\tag{17}
$$

Samples with low $\nu _ { i }$ are downsampled, whereas samples with higher motion variation are retained with higher probability. This indexing keeps action differences within the trajectory and action chunk associated with sample i.

Progress-stratified sampling. We divide normalized progress into fixed bins and sample from each bin. This prevents the bank from being dominated by a small portion of a trajectory and improves coverage of different execution phases.

Skill clustering. We first cluster the retained samples in query space. For a query cluster ${ \mathcal { C } } ,$ its skill variance is

$$
\sigma _ { s } ^ { 2 } ( \mathcal { C } ) = \frac { 1 } { | \mathcal { C } | } \sum _ { i \in \mathcal { C } } \left. s _ { i } - \mu _ { s } ( \mathcal { C } ) \right. _ { 2 } ^ { 2 } , \qquad \mu _ { s } ( \mathcal { C } ) = \frac { 1 } { | \mathcal { C } | } \sum _ { i \in \mathcal { C } } s _ { i } .\tag{18}
$$

Clusters whose skill variance exceeds the splitting threshold are further partitioned in skill space. Each final cluster $\mathcal { C } _ { j }$ defines an entry of the bank in Eq. (6):

$$
p _ { j } ^ { q } = \operatorname { n o r m } \left( \frac { 1 } { | \mathcal { C } _ { j } | } \sum _ { i \in \mathcal { C } _ { j } } q _ { i } \right) , \qquad p _ { j } ^ { s } = \frac { 1 } { | \mathcal { C } _ { j } | } \sum _ { i \in \mathcal { C } _ { j } } s _ { i } , \qquad n _ { j } = | \mathcal { C } _ { j } | .\tag{19}
$$

Here, norm $( v ) = v / \| v \| _ { 2 }$ for a nonzero vector v. We retain up to three latent query–skill pairs from each cluster as a compact exemplar set $\mathcal { E } _ { j }$ for replay. The initial bank is denoted by $B ^ { ( 0 ) }$ and contains $K _ { 0 }$ entries; the current bank B contains $K$ entries, which may increase during Stage C.

## C.3 RETRIEVAL AND RESIDUAL PROTOTYPE ADAPTATION

The query alignment $q _ { t } ^ { \prime } = A _ { \mathrm { m e m } } q _ { t }$ is defined in $\operatorname { E q . } \left( 7 \right)$ . For use in both retrieval and the update rules below, let

$$
r _ { t j } = \sin \bigl ( q _ { t } ^ { \prime } , p _ { j } ^ { q } + \Delta p _ { j } ^ { q } \bigr ) , \qquad j = 1 , \ldots , K ,\tag{20}
$$

where sim is the same similarity function used in Eq. (8). The set $\mathcal { N } _ { t }$ contains the indices of the top- $K _ { r }$ prototypes under this score. Retrieval weights $\alpha _ { t j }$ are computed with temperature $\tau _ { \mathrm { m e m } }$ by Eq. (8), and the retrieved skill $\hat { s } _ { t }$ is computed by Eq. (9).

For a single sample, $\mathcal { A } _ { t } = \mathcal { N } _ { t }$ is the active index set in Eq. (12). Across a minibatch, the residuals eligible for updates are those retrieved by at least one sample; the per-sample loss retains the active set specified in the main text. Existing base prototypes $( p _ { j } ^ { q } , p _ { j } ^ { s } )$ remain fixed, while active residuals $( \Delta p _ { j } ^ { q } , \Delta p _ { j } ^ { s } )$ provide local corrections. Updating support counts and exemplar sets, or appending a new entry in Stage $\mathrm { C } ,$ is distinct from overwriting an existing base prototype.

## C.4 BANK UPDATE RULE

During Stage $\mathrm { C } ,$ the reuse decision uses the same residual-adjusted query prototypes as retrieval. For sample $t ,$ define

$$
r _ { t , \operatorname* { m a x } } = \operatorname* { m a x } _ { 1 \leq j \leq K } r _ { t j } , \qquad j _ { t } ^ { \star } \in \arg \operatorname* { m a x } _ { 1 \leq j \leq K } r _ { t j } .\tag{21}
$$

If $r _ { t , \mathrm { m a x } } > \tau _ { \mathrm { r e u s e } } ,$ the sample is assigned to prototype $j _ { t } ^ { \star }$ . It contributes to active-residual learning and updates the corresponding support statistics; its latent query–skill pair is stored as an exemplar, with first-in, first-out replacement once the exemplar set is full. If $r _ { t , \mathrm { m a x } } \leq \tau _ { \mathrm { r e u s e } }$ , the sample enters a candidate buffer.

The candidate buffer is periodically clustered. Candidate compactness is evaluated in the same coordinates used to initialize an appended prototype: aligned queries $q _ { i } ^ { \prime }$ and encoded skills $s _ { i } .$ . For a candidate cluster ${ \mathcal { C } } ,$ define

$$
\mu _ { q ^ { \prime } } ( \mathcal { C } ) = \frac { 1 } { | \mathcal { C } | } \sum _ { i \in \mathcal { C } } q _ { i } ^ { \prime } , \quad \mathrm { V a r } _ { q } ( \mathcal { C } ) = \frac { 1 } { | \mathcal { C } | } \sum _ { i \in \mathcal { C } } \| q _ { i } ^ { \prime } - \mu _ { q ^ { \prime } } ( \mathcal { C } ) \| _ { 2 } ^ { 2 } ,
$$

$$
\mu _ { s } ( \mathcal { C } ) = \frac { 1 } { | \mathcal { C } | } \sum _ { i \in \mathcal { C } } s _ { i } , \quad \mathrm { V a r } _ { s } ( \mathcal { C } ) = \frac { 1 } { | \mathcal { C } | } \sum _ { i \in \mathcal { C } } \| s _ { i } - \mu _ { s } ( \mathcal { C } ) \| _ { 2 } ^ { 2 } .\tag{22}
$$

A new prototype is appended only when a candidate cluster $\mathcal { C } _ { \mathrm { n e w } }$ satisfies

$$
\begin{array} { r } { | \mathcal { C } _ { \mathrm { n e w } } | \geq N _ { \mathrm { m i n } } , \qquad \mathrm { V a r } _ { q } ( \mathcal { C } _ { \mathrm { n e w } } ) \leq \epsilon _ { q } , \qquad \mathrm { V a r } _ { s } ( \mathcal { C } _ { \mathrm { n e w } } ) \leq \epsilon _ { s } . } \end{array}\tag{23}
$$

The appended entry is initialized by

$$
\begin{array} { r l } & { p _ { \mathrm { n e w } } ^ { q } = \mathrm { n o r m } ( \mu _ { q ^ { \prime } } ( \mathcal { C } _ { \mathrm { n e w } } ) ) , \qquad p _ { \mathrm { n e w } } ^ { s } = \mu _ { s } ( \mathcal { C } _ { \mathrm { n e w } } ) , } \\ & { n _ { \mathrm { n e w } } = | \mathcal { C } _ { \mathrm { n e w } } | , \qquad \Delta p _ { \mathrm { n e w } } ^ { q } = 0 , \qquad \Delta p _ { \mathrm { n e w } } ^ { s } = 0 . } \end{array}\tag{24}
$$

A compact exemplar set ${ \mathcal { E } } _ { \mathrm { n e w } }$ is also stored. Candidate entries use the encoded skills $s _ { i } ;$ the Skill Adapter operates on the aggregated retrieval $\hat { s } _ { t }$ during policy conditioning and alignment. The adapter, if retained after Stage B, is held fixed during Stage C.

## C.5 COORDINATE ALIGNMENT UNDER DOMAIN SHIFT

Stage B adapts retrieval and skill conditioning under domain shift. The query-side transformation is the linear adapter $A _ { \mathrm { m e m } }$ in Eq. (7), initialized close to the identity. For a larger skill-space mismatch, the optional Skill Adapter maps the retrieved skill to $\tilde { s } _ { t } = A _ { s } ( \hat { s } _ { t } )$ as defined in Eq. (11). Its zeroinitialized residual map satisfies $U _ { s } V _ { s } ^ { \top } = 0 , \thinspace \mathrm { { s o } } \ \tilde { s } _ { t } = \hat { s } _ { t }$ initially; this condition does not require both factors to be zero. The corrected skill conditions the policy and is aligned to the detached encoded target $s _ { t } ,$ retaining gradient paths to the adapter through both losses.

Before enabling the Skill Adapter in Stage B, we monitor the novelty rate and retrieval-skill alignment error:

$$
\eta _ { \mathrm { b a t c h } } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \mathbf { 1 } [ r _ { i , \mathrm { m a x } } \le \tau _ { \mathrm { r e u s e } } ] ,\tag{25}
$$

$$
e _ { \mathrm { a l i g n } } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \left. \hat { s } _ { i } - \mathrm { s g } ( s _ { i } ) \right. _ { 2 } ^ { 2 } .\tag{26}
$$

Before activation, $\tilde { s } _ { i } = \hat { s } _ { i }$ , so this diagnostic measures the unadapted retrieval error. When either signal remains above its activation threshold for several consecutive batches, the adapter is enabled. These diagnostics indicate persistent mismatch but do not, by themselves, distinguish domain shift from genuinely novel skills. During optimization, the encoded target $s _ { i }$ is detached, while the corrected retrieval $\tilde { s } _ { i }$ retains its gradient path.

## C.6 STAGE-WISE OPTIMIZATION

The three stages use the objectives defined in Sec. 3.4: Stage A minimizes $\mathcal { L } _ { \mathrm { p r e } }$ in Eq. (10), whereas Stages B and C use the same $\mathcal { L } _ { \mathrm { l i f e } }$ in Eq. (12). The stages differ in which parameter groups are updated, not in the definition of the adaptation loss. Table 5 summarizes these distinctions. Section D supplies the query-contrastive details and optimization settings without introducing additional stage-specific objective equations.

## D TRAINING

## D.1 TRAINING OBJECTIVES

Stage A: interface pretraining. Stage A trains the memory seed tokens, both attention poolers, the Query Encoder, the Skill Encoder, the Skill Decoder, the Token Projector, and the last layers of the VLA backbone. No memory bank is used in its forward computation. The policy is conditioned on $S _ { t } = T _ { \psi } ( s _ { t } )$ , and optimization uses $\mathcal { L } _ { \mathrm { p r e } }$ in Eq. (10), including the original $\bar { \mathrm { V L A } }$ flow-matching loss ${ \mathcal L } _ { \mathrm { f m } }$ and the Huber reconstruction term $\mathcal { L } _ { \mathrm { s k i l l } }$ defined there.

For completeness, we specify the query contrastive term $\mathcal { L } _ { q } ^ { \mathrm { n c e } }$ from Eq. (10). Let $\mathcal { I }$ be the set of sample indices in the current minibatch. For sample $i \in \mathcal { I } ,$ , let $y _ { i }$ be its task label, $\mathrm { t r a j } ( i )$ its trajectory identifier, $t _ { i }$ its within-trajectory time index, and $\rho _ { i }$ its normalized progress. The positive set is

$$
\begin{array} { r l } & { \mathcal { P } ( i ) = \{ j \in \mathcal { I } \setminus \{ i \} : y _ { j } = y _ { i } , \ | \rho _ { j } - \rho _ { i } | \leq \delta \} } \\ & { \qquad \cup \ \{ j \in \mathcal { I } \setminus \{ i \} : \operatorname { t r a j } ( j ) = \operatorname { t r a j } ( i ) , \ | t _ { j } - t _ { i } | \leq w \} . } \end{array}\tag{27}
$$

Thus, positives comprise samples from the same task with nearby normalized progress and local neighboring states from the same trajectory, as described in Sec. 3.4. Let $\mathcal { T } = \{ i \stackrel { \cdot } { \in } \bar { \mathcal { I } } : | \mathcal { P } ( i ) | > 0 \}$ denote the valid anchors. For $\mathcal { T } \neq \emptyset$

$$
\mathcal { L } _ { q } ^ { \mathrm { n c e } } = - \frac { 1 } { | \mathcal { Z } | } \sum _ { i \in \mathcal { Z } } \frac { 1 } { | \mathcal { P } ( i ) | } \sum _ { j \in \mathcal { P } ( i ) } \log \frac { \exp ( \sin ( q _ { i } , q _ { j } ) / \tau _ { q } ) } { \sum _ { \ell \in \mathcal { T } \backslash \{ i \} } \exp ( \sin ( q _ { i } , q _ { \ell } ) / \tau _ { q } ) } .\tag{28}
$$

When no anchor has a positive, the contrastive contribution is defined to be zero. The query-contrastive temperature $\tau _ { q }$ is distinct from the memory-retrieval temperature $\tau _ { \mathrm { m e m } }$ . After Stage A, the learned interface, Token Projector, Skill Decoder, and backbone are frozen, and the initial memory bank is constructed as described in Sec. C.2.

Stage B: bridge adaptation. Stage B uses $\mathcal { L } _ { \mathrm { l i f e } }$ in $\operatorname { E q . }$ . (12). It updates the backbone, $A _ { \mathrm { { m e m } } } , \tau _ { \mathrm { { m e m } } } ,$ and active prototype residuals. The learned interface modules, Token Projector, Skill Decoder, and existing base prototypes remain fixed. Thus, the backbone is made trainable again for this bridge stage, in contrast to its frozen status immediately after Stage A and during Stage C. The optional Skill Adapter follows Eq. (11) and the activation criterion in Sec. C.5.

Stage C: memory adaptation. Stage C also uses Eq. (12), with the backbone, Inline Memory Interface, Token Projector, and Skill Decoder frozen. It updates only $A _ { \mathrm { { m e m } } } , \tau _ { \mathrm { { m e m } } } .$ , active prototype residuals, and the bank entries and statistics described in Sec. C.4. The Skill Adapter is not optimized in this stage. For a local batch of B current-task samples, we sample $\lfloor B / 3 \rfloor$ stored latent query–skill pairs, when available, and combine them with the current samples in the alignment term of Eq. (12), giving an approximate new-to-replay ratio of 3:1. The flow-matching loss is computed only on current-task demonstrations. The replay buffer stores no raw images, actions, or trajectories, and no separate replay loss is introduced.

Table 5: Stage-wise optimization and memory-bank operations. “Last layers” denotes the partial backbone training in Stage A. The optional Skill Adapter is a Stage-B component; it is fixed or inactive in Stage ${ \bar { \mathbf { C } } } .$
<table><tr><td>Parameter group / operation</td><td>Stage A</td><td>Stage B</td><td>Stage C</td></tr><tr><td>Memory tokens  $E ^ { \mathrm { m e m } }$ </td><td>train</td><td>freeze</td><td>freeze</td></tr><tr><td>Poolers  $\mathrm { A t t n P o o l } _ { q } , \mathrm { A t t n P o o l } _ { s }$ </td><td>train</td><td>freeze</td><td>freeze</td></tr><tr><td>Encoders  $f _ { q } , f _ { s }$ </td><td>train</td><td>freeze</td><td>freeze</td></tr><tr><td>Skill Decoder  $D _ { s }$ </td><td>train</td><td>freeze</td><td>freeze</td></tr><tr><td>Token Projector  $T _ { \psi }$ </td><td>train</td><td>freeze</td><td>freeze</td></tr><tr><td>Backbone  $F _ { \theta }$ </td><td>last layers</td><td>train</td><td>freeze</td></tr><tr><td>Query alignment  $A _ { \mathrm { m e m } }$ </td><td></td><td>train</td><td>train</td></tr><tr><td>Retrieval temperature  $\tau _ { \mathrm { m e m } }$ </td><td></td><td>train</td><td>train</td></tr><tr><td>Active residuals  $\Delta p _ { j } ^ { q } , \Delta p _ { j } ^ { s }$ </td><td></td><td>train</td><td>train</td></tr><tr><td>Skill Adapter  $A _ { s }$ </td><td></td><td>train</td><td>freeze</td></tr><tr><td>Bank structure  $\boldsymbol { B }$ </td><td></td><td></td><td>update</td></tr></table>

Table 6: Reported default hyperparameters for Optimus-R.
<table><tr><td>Category</td><td>Hyperparameter</td><td>Value</td></tr><tr><td rowspan="4">Architecture</td><td>Memory tokens  $m$ </td><td>4</td></tr><tr><td>Query dimension  $d _ { q }$ </td><td>256</td></tr><tr><td>Skill dimension  $d _ { s }$ </td><td>256</td></tr><tr><td>Skill tokens  $n _ { s }$ </td><td>4</td></tr><tr><td rowspan="4">Memory bank / retrieval</td><td>Initial bank size  $K _ { 0 }$ </td><td>128</td></tr><tr><td>Retrieval count  $K _ { r }$ </td><td>2</td></tr><tr><td>Reuse threshold  $\tau _ { \mathrm { r e u s e } }$ </td><td>0.75</td></tr><tr><td>Minimum cluster size  $N _ { \mathrm { m i n } }$ </td><td>50</td></tr><tr><td rowspan="8">Stage A pretraining</td><td>Query contrastive weight  $\lambda _ { q }$ </td><td>0.1</td></tr><tr><td>Skill reconstruction weight  $\lambda _ { s }$ </td><td>1.0</td></tr><tr><td>Query contrastive temperature  $\tau _ { q }$ </td><td>0.07</td></tr><tr><td>Progress threshold δ</td><td>0.1</td></tr><tr><td>Trajectory window w</td><td>6</td></tr><tr><td>Learning rate for new modules</td><td> $2 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Learning rate for unfrozen backbone layers</td><td> $2 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Weight decay for new modules</td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td rowspan="5">Stage B/C adaptation</td><td>Alignment weight  $\lambda _ { \mathrm { a l i g n } }$ </td><td>0.1</td></tr><tr><td>Residual regularization weight  $\lambda _ { \mathrm { r e g } }$ </td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td>Learning rate for  $A _ { \mathrm { m e m } }$ </td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td>Learning rate for</td><td> $5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td> $\tau _ { \mathrm { m e m } }$  Learning rate for active prototype residuals</td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td rowspan="2">Optional Skill Adapter</td><td>Adapter rank r</td><td> $^ 8$ </td></tr><tr><td>Reported learning rate for  $A _ { s }$ </td><td> $5 \times 1 0 ^ { - 5 }$ </td></tr></table>

## D.2 HYPERPARAMETER SETTINGS

Table 6 summarizes the hyperparameters reported for Optimus-R. Unless otherwise specified, the same configuration is used across experiments. The symbols match the main text: $K _ { r }$ is the number of retrieved prototypes, $K _ { 0 }$ is the initial bank size, and r is the rank of the optional Skill Adapter.

For Stage A, the newly introduced modules use a learning rate of $2 \times 1 0 ^ { - 4 }$ and weight decay $1 0 ^ { - 4 }$ while the unfrozen backbone layers use a learning rate of $2 \times 1 0 ^ { - 5 }$ . For both adaptation stages, $\lambda _ { \mathrm { a l i g n } } = 0 . 1$ and $\lambda _ { \mathrm { r e g } } = 1 0 ^ { - 4 }$ . The backbone is updated in Stage B and frozen in Stage C; the learned interface modules and Token Projector remain fixed in both. The optional residual Skill Adapter has rank $r = 8 ,$ and the reported adapter learning rate is $5 \times 1 0 ^ { - 5 }$ ; this setting does not imply that the adapter is optimized in Stage C.

Table 7: Performance comparison on RoboTwin 2.0. We report per-task success rates (SR) over 100 rollouts under Hard setting. <sup>†</sup> represents the result we reproduced.
<table><tr><td>Task</td><td>RDT</td><td>ACT</td><td>DP</td><td>DP3</td><td> $\pi _ { 0 }$ </td><td> $\pi _ { \mathbf { 0 . 5 } } \mathrm { ^ \dagger }$ </td><td>Optimus-R</td></tr><tr><td>Adjust Bottle</td><td>75%</td><td>23%</td><td>0%</td><td>3%</td><td>56%</td><td>75%</td><td>89%</td></tr><tr><td>Click Alarmclock</td><td>12%</td><td>4%</td><td>5%</td><td>14%</td><td>11%</td><td>44%</td><td>41%</td></tr><tr><td>Click Bell</td><td>9%</td><td>3%</td><td>0%</td><td>0%</td><td>3%</td><td>64%</td><td>53%</td></tr><tr><td>Grab Roller</td><td>43%</td><td>25%</td><td>0%</td><td>2%</td><td>80%</td><td>82%</td><td>94%</td></tr><tr><td>Move Playingcard Away</td><td>11%</td><td>0%</td><td>0%</td><td>3%</td><td>22%</td><td>32%</td><td>38%</td></tr><tr><td>Pick Diverse Bottles</td><td>0%</td><td>0%</td><td>0%</td><td>1%</td><td>6%</td><td>29%</td><td>36%</td></tr><tr><td>Place a2b Left</td><td>1%</td><td>0%</td><td>0%</td><td>2%</td><td>1%</td><td>20%</td><td>29%</td></tr><tr><td>Place a2b Right</td><td>1%</td><td>0%</td><td>0%</td><td>0%</td><td>6%</td><td>19%</td><td>21%</td></tr><tr><td>Place Bread Basket</td><td>2%</td><td>0%</td><td>0%</td><td>1%</td><td>4%</td><td>28%</td><td>39%</td></tr><tr><td>Place Burger Fries</td><td>27%</td><td>0%</td><td>0%</td><td>18%</td><td>4%</td><td>46%</td><td>59%</td></tr><tr><td>Place Container Plate</td><td>17%</td><td>1%</td><td>0%</td><td>1%</td><td>45%</td><td>55%</td><td>74%</td></tr><tr><td>Place Empty Cup</td><td>7%</td><td>0%</td><td>0%</td><td>1%</td><td>11%</td><td>59%</td><td>72%</td></tr><tr><td>Place Object Stand</td><td>5%</td><td>0%</td><td>0%</td><td>0%</td><td>11%</td><td>46%</td><td>47%</td></tr><tr><td>Place Shoe</td><td>7%</td><td>0%</td><td>0%</td><td>2%</td><td>6%</td><td>20%</td><td>40%</td></tr><tr><td>Rotate QRCode</td><td>5%</td><td>0%</td><td>0%</td><td>1%</td><td>15%</td><td>20%</td><td>20%</td></tr><tr><td>Shake Bottle Horizon</td><td>51%</td><td>4%</td><td>18%</td><td>25%</td><td>51%</td><td>85%</td><td>94%</td></tr><tr><td>Shake Bottle</td><td>45%</td><td>10%</td><td>8%</td><td>19%</td><td>60%</td><td>82%</td><td>96%</td></tr><tr><td>Stack Blocks Two</td><td>2%</td><td>0%</td><td>0%</td><td>0%</td><td>1%</td><td>21%</td><td>35%</td></tr><tr><td>Stack Bowls Two</td><td>30%</td><td>0%</td><td>0%</td><td>6%</td><td>41%</td><td>62%</td><td>71%</td></tr><tr><td>Stack Bowls Three</td><td>17%</td><td>0%</td><td>0%</td><td>5%</td><td>24%</td><td>42%</td><td>51%</td></tr></table>

## D.3 INFERENCE

At inference time, neither the Skill Encoder nor the Skill Decoder is needed. Given $( O _ { t } , L )$ , the model computes the memory-token states by Eq. (1) and forms the retrieval query with the query branch of Eqs. (3) and (4). It then aligns $q _ { t }$ to obtain $q _ { t } ^ { \prime } = A _ { \mathrm { m e m } } q _ { t }$ , selects $\begin{array} { r } { \hat { \mathcal { N } } _ { t } . } \end{array}$ , and computes $\alpha _ { t j }$ and $\hat { s } _ { t }$ according to Eqs. (7)–(9). This top-K<sub>r</sub> retrieval determines what to retrieve; no binary retrieval gate is used. If enabled, the Skill Adapter produces $\tilde { s } _ { t } = A _ { s } ( \hat { s } _ { t } )$ ; otherwise, $\tilde { s } _ { t } = \hat { s } _ { t }$ . Finally, $S _ { t } = T _ { \psi } \mathbf { \overline { { ( } } } \tilde { s } _ { t } )$ conditions the Flow Policy, which predicts $\tilde { A } _ { t } = \pi _ { \theta } ( H _ { t } ^ { \mathrm { v l } } , S _ { t } )$ as in Eq. (2).

In a cache-based implementation, the original multimodal and memory-token prefix is cached and then extended with $S _ { t }$ before action generation. This is an implementation of the same policyconditioning path, not a separate backbone or action generator. The skill-side attention pooler, Skill Encoder, and Skill Decoder are unnecessary for this retrieval-only inference path; the learned query alignment, retrieval temperature, prototype residuals, and any enabled Skill Adapter are retained.

## E EVALUATION ON ROBOTWIN 2.0

RoboTwin 2.0 provides a standardized benchmark for bimanual manipulation built upon the RoboTwin-OD object library, which contains 731 annotated objects spanning 147 categories, together with a large-scale corpus of more than 100k expert dual-arm trajectories. The benchmark includes 50 collaborative bimanual tasks instantiated across five robot embodiments. Under the standard simulation protocol, each task is trained independently on the Aloha–AgileX dual-arm platform using 50 clean expert demonstrations, and evaluated with 100 rollouts under two difficulty settings: an Easy setting with uncluttered scenes, and a Hard setting with substantial domain randomization, including clutter, background textures, lighting variations, and tabletop-height perturbations.

In our experiments, we evaluate on a randomly selected subset of 20 tasks. For each task, we train our models using all available clean demonstrations and report performance over 100 rollouts in the Hard setting, which provides a stringent test of robustness to visual and physical domain shifts. As shown in Table 7, Optimus-R obtains an average success rate of 55% across the selected tasks, outperforming all compared baselines, including $\pi _ { 0 . 5 }$

## F REAL-WORLD SETTING

Sim-to-real adaptation. In the transfer setting of Sec. 4.3, the RoboTwin and real-world data use a common observation format and action-space representation. The target domain differs in backgrounds, lighting, object appearance, camera viewpoints, and spatial layouts. For example, the real-world Block Stacking task shares manipulation patterns with RoboTwin’s Stack Blocks Two, despite different visual conditions. During Stage B, $A _ { \mathrm { m e m } }$ maps target-domain queries into the source memory coordinates, as defined in Eq. (7), so that relevant source skills can be retrieved and refined through prototype residual updates.

Table 8: Detailed configuration of the real-world tasks for lifelong learning evaluation. The tasks are divided into Base Tasks (Category A) and Novel Tasks (Category B), featuring a transition from single-arm to bimanual manipulation.
<table><tr><td>Cat.</td><td>Task Family</td><td>Arm Mode</td><td>Objects</td><td>Targets</td></tr><tr><td rowspan="4">A</td><td>Reorientation</td><td>Single-arm</td><td>bottle, mug</td><td>-</td></tr><tr><td>Block Stacking</td><td>Single-arm</td><td>block</td><td>=</td></tr><tr><td>Place Object onto Plate</td><td>Single-arm</td><td>can, bowl, fruit</td><td>plate</td></tr><tr><td>Place Object onto Tablecloth</td><td>Single-arm</td><td>bowl, plate, cup</td><td>tablecloth</td></tr><tr><td rowspan="5">B</td><td>Put Objects into Container</td><td>Bimanual</td><td>bottle, can</td><td>basket</td></tr><tr><td>Bimanual Basket Placement</td><td>Bimanual</td><td>basket</td><td>tablecloth</td></tr><tr><td>Multi-object Collection</td><td>Bimanual</td><td>fruit, block</td><td>plate, tablecloth</td></tr><tr><td>Sequential Pick and Place</td><td>Bimanual</td><td>fruit, cup</td><td>bowl, tablecloth, plate</td></tr><tr><td>Bimanual Coordination Sequential Task</td><td>Bimanual</td><td>basket, fruits</td><td>tablecloth, basket</td></tr></table>

To rigorously evaluate the lifelong learning capability and the anti-forgetting mechanisms of Optimus R, we designed a comprehensive suite of real-world robotic manipulation tasks. The evaluation protocol is specifically structured to simulate a challenging continuous learning scenario, encompassing a transition from basic single-arm manipulations to complex, long-horizon bimanual coordination.

As summarized in Table 8, the tasks are strategically divided into two distinct categories:

• Category A (Base Tasks): This set comprises 4 fundamental single-arm manipulation tasks(Figs. 7), including spatial reorientation and precision placement (e.g., block stacking, placing objects onto specific receptacles). The model is initially trained on this task suite, which provides a total of 570 demonstrations. These tasks serve as the foundation for evaluating the model’s stability (i.e., retention of historical skills) during subsequent lifelong adaptation.

• Category B (Novel Tasks): After acquiring the base skills, the model is incrementally adapted to a stream of 5 novel tasks(Figs.8). Crucially, these tasks introduce a significant shift in the control distribution by requiring bimanual coordination and long-horizon sequential logic (e.g., sequential pick-and-place, multi-object collection). This category contains 420 demonstrations in total.

This Category A → B transition is intentionally demanding. The shift from single-arm to bimanual control, coupled with the introduction of multi-stage semantic reasoning, typically exacerbates catastrophic forgetting in parameter-centric fine-tuning methods. Evaluating Optimus-R under this protocol effectively demonstrates the robustness of our discrete Query-Skill Memory Bank in preserving old capabilities while assimilating highly distinct new skills.

![](images/7f35895a9b7f3bcdeab5f6fa142514769a05b4df8e180be1d0e214cf0c9a1b5f.jpg)

Figure 7: Qualitative rollouts on real-world Category-A tasks. We visualize representative trajectories for four previously learned manipulation tasks: Reorientation, Block Stacking, Place Object onto Plate, and Place Object onto Tablecloth. Across these tasks, Optimus-R consistently executes precise object-centric manipulation behaviors under diverse object and target configurations. These results illustrate that the learned query-skill memory can preserve reusable visuomotor primitives and support stable execution of previously acquired skills during subsequent adaptation.  
![](images/574dc0f01b2736cc6c92a433bf4277e6d9fdab7a33cef4743118a912004dfb38.jpg)  
Figure 8: Qualitative rollouts on real-world Category-B tasks under the lifelong adaptation setting. We show representative trajectories for five newly introduced tasks: Put Objects into Container, Bimanual Basket Placement, Multi-object Collection, Sequential Pick and Place, and Bimanual Coordination Sequential Task. These tasks require multi-object reasoning, long-horizon sequencing, and coordinated dual-arm control, posing a substantial distribution shift from the Category-A task set. Optimus-R successfully adapts to these novel behaviors by retrieving, updating, and expanding explicit query-skill memory entries, enabling new skill acquisition while mitigating interference with previously learned manipulation capabilities.

Table 9: Memory-bank component ablations on LIBERO with 100% training data.
<table><tr><td>Variant</td><td>Average SR (%)</td></tr><tr><td>Optimus-R (Full)</td><td>97.8</td></tr><tr><td>w/o skill-aware clustering</td><td>95.8</td></tr><tr><td>w/o exemplar storage</td><td>96.3</td></tr><tr><td>Uniform downsampling</td><td>97.2</td></tr></table>

Table 10: LIBERO configuration sensitivity. Bold entries identify the default configuration. The last two rows jointly vary retrieval count and memory-token count.
<table><tr><td> $\pmb { K _ { 0 } }$ </td><td> $K _ { r }$ </td><td>m</td><td>Average SR (%)</td></tr><tr><td>64</td><td>2</td><td>4</td><td>97.2</td></tr><tr><td>128</td><td>2</td><td>4</td><td>97.8</td></tr><tr><td>256</td><td>2</td><td>4</td><td>97.8</td></tr><tr><td>128</td><td>1</td><td>4</td><td>97.0</td></tr><tr><td>128</td><td>4</td><td>8</td><td>97.4</td></tr><tr><td>128</td><td>4</td><td>1</td><td>96.8</td></tr></table>

## G ADDITIONAL EXPERIMENTS

## G.1 MEMORY-BANK COMPONENT ABLATIONS

We evaluate memory-bank design choices on LIBERO using 100% of the training demonstrations and report the average success rate across the four suites. Table 9 compares the full model with variants that omit skill-aware clustering, omit exemplar storage for replay, or replace motion-aware downsampling with uniform downsampling.

Removing clustering or exemplar storage reduces success by 2.0 and 1.5 percentage points, respectively, supporting their contribution to memory-based adaptation. Uniform downsampling produces a smaller decrease of 0.6 percentage points, suggesting that motion-aware sampling is beneficial but less influential among the tested variants.

## G.2 MEMORY-BANK CONFIGURATION SENSITIVITY

We examine the initial bank size $K _ { 0 } .$ , retrieval count $K _ { r }$ , and number of memory tokens m on LIBERO at 100% training data. Table 10 distinguishes changes to a single parameter from configurations that jointly change $\bar { K } _ { r }$ and m relative to the default (128, 2, 4).

Success rates range from 96.8% to 97.8% across the tested configurations. Increasing $K _ { 0 }$ from 128 to 256 yields no further gain, while reducing $K _ { r }$ from two to one lowers success to 97.0%. The default therefore attains the highest observed success with fewer prototypes than the 256-entry bank. The joint configurations do not isolate the individual effects of $K _ { r }$ and m relative to the default.

## G.3 MEMORY GROWTH ACROSS TASK SUITES

To examine memory growth over a longer adaptation sequence, we initialize Optimus-R on LIBERO-90 and then adapt sequentially to Spatial, Object, Goal, and Long. Table 11 reports the occupied prototype count K, the increment at each stage, retrieval latency, and Recall@5. This sequence starts with 81 occupied prototypes; the counts describe the evolving bank in this experiment rather than the default $K _ { 0 } = 1 2 8$ configuration above.

The bank adds 28 prototypes across four suites, while retrieval latency increases by 0.10 ms and Recall@5 changes from 95.6% to 94.7%. These measurements indicate a modest retrieval overhead over the evaluated sequence. They characterize expansion at this scale, without establishing an upper bound on memory growth or prototype interference.

Table 11: Memory growth during sequential suite adaptation. Prototype increments are relative to the preceding stage; latency measures retrieval rather than full policy inference.
<table><tr><td>Adaptation stage</td><td>K</td><td>New</td><td>Latency (ms)</td><td>Recall@5 (%)</td></tr><tr><td>After LIBERO-90</td><td>81</td><td>一</td><td>1.80</td><td>95.6</td></tr><tr><td>+ Spatial</td><td>84</td><td>3</td><td>1.80</td><td>95.4</td></tr><tr><td>+ Object</td><td>90</td><td>6</td><td>1.83</td><td>96.1</td></tr><tr><td>+ Goal</td><td>99</td><td>9</td><td>1.86</td><td>95.2</td></tr><tr><td>+ Long</td><td>109</td><td>10</td><td>1.90</td><td>94.7</td></tr></table>

Table 12: Spatial-to-Goal transfer. Bowl-on-plate denotes the Goal task put the bowl on the plate.
<table><tr><td>Evaluation stage</td><td>Goal SR (%)</td><td>Bowl-on-plate SR (%)</td><td> $\kappa$ </td></tr><tr><td>Zero-shot</td><td>8</td><td>68</td><td>16</td></tr><tr><td>After Goal adaptation</td><td>96</td><td>98</td><td>27</td></tr></table>

Table 13: Task-wise continual learning on LIBERO-Spatial. Entries are success rates (%); dashes denote tasks not yet introduced. $\boldsymbol { T _ { 1 } { - } } \boldsymbol { T _ { 1 0 } } ^ { - }$ follow the sequence in Liu et al. (2026).
<table><tr><td>Checkpoint</td><td> $\mathbf { T _ { 1 } }$ </td><td> $\mathbf { T _ { 2 } }$ </td><td> $\mathbf { T _ { 3 } }$ </td><td> $\mathbf { { T _ { 4 } } }$ </td><td> $\mathbf { T _ { 5 } }$ </td><td> $\mathbf { { T _ { 6 } } }$ </td><td> $\mathbf { { \delta } } \mathbf { { \mathit { T } } } _ { 7 }$ </td><td> $\pmb { T _ { 8 } }$ </td><td> $\bf { { T _ { 9 } } }$ </td><td> $\mathbf { { T _ { 1 0 } } }$ </td></tr><tr><td>After T1</td><td>100</td><td></td><td>一</td><td>一</td><td></td><td></td><td>一</td><td>一</td><td>一</td><td></td></tr><tr><td>After  $T _ { 2 }$ </td><td>98</td><td>100</td><td></td><td>一</td><td>一</td><td></td><td>一</td><td>一</td><td>一</td><td>一</td></tr><tr><td>After T3</td><td>98</td><td>100</td><td>100</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td><td></td></tr><tr><td>After  $T _ { 4 }$ </td><td>100</td><td>98</td><td>100</td><td>98</td><td></td><td></td><td></td><td>一</td><td>一</td><td></td></tr><tr><td>After T5</td><td>100</td><td>100</td><td>94</td><td>100</td><td>98</td><td>一</td><td>一</td><td>一</td><td>一</td><td></td></tr><tr><td>After T6</td><td>96</td><td>100</td><td>96</td><td>98</td><td>94</td><td>98</td><td>一</td><td>一</td><td>一</td><td>一</td></tr><tr><td>After  $T _ { 7 }$ </td><td>98</td><td>100</td><td>96</td><td>98</td><td>94</td><td>96</td><td>100</td><td></td><td>一</td><td></td></tr><tr><td>After  $T _ { 8 }$ </td><td>98</td><td>100</td><td>100</td><td>100</td><td>92</td><td>94</td><td>98</td><td>98</td><td>一</td><td></td></tr><tr><td>After  $T _ { 9 }$ </td><td>100</td><td>98</td><td>98</td><td>100</td><td>92</td><td>100</td><td>100</td><td>98</td><td>98</td><td></td></tr><tr><td>After  $T _ { 1 0 }$ </td><td>100</td><td>98</td><td>100</td><td>98</td><td>90</td><td>98</td><td>98</td><td>98</td><td>100</td><td>96</td></tr></table>

## G.4 CROSS-SUITE TRANSFER FROM SPATIAL TO GOAL

Starting from the $\pi _ { 0 . 5 }$ -base checkpoint, we train Optimus-R on the ten LIBERO-Spatial tasks with $K _ { 0 } = 1 6 ,$ , evaluate it zero-shot on LIBERO-Goal, and then adapt it to Goal. The suites differ in scene layouts and initial-state distributions while sharing manipulation patterns, such as placing a bowl on a plate. Table 12 reports overall Goal performance, performance on this shared manipulation pattern, and bank size.

The bowl-on-plate task achieves 68% success before target-suite adaptation, consistent with transfer of a manipulation pattern represented in Spatial. However, overall zero-shot success is only 8%, and turn on the stove provides a failure case involving an operation absent from the Spatial tasks. After adaptation, overall success reaches 96% and the bank grows by 11 prototypes. The results support selective transfer of shared behaviors while showing the need for target-task adaptation.

## G.5 TASK-WISE CONTINUAL LEARNING ON LIBERO-SPATIAL

We further evaluate a ten-task sequence on LIBERO-Spatial using the task order of Liu et al. (2026). Optimus-R starts from $\pi _ { 0 . 5 }$ -base and trains for 10,000 steps per task, carrying the preceding checkpoint forward. After each stage, we evaluate all tasks observed so far. Adaptation uses the latent exemplar replay described in Sec. D. Table 13 reports the resulting success-rate matrix.

We also report negative backward transfer (NBT): for each task, we average its success-rate decrease from its post-training checkpoint over subsequent stages, then average over all ten tasks, assigning zero to the final task. Using success rates on a [0, 1] scale, the sequence yields NBT = 0.009 and a final average success rate of 97.6%. This indicates limited average forgetting over the sequence, although retention varies by task; for example, $T _ { 5 }$ decreases from 98% immediately after training to 90% at the final stage.