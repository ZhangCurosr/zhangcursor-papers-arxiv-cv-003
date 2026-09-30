# HaPRL: Human-Anchored Process Reinforcement Learning for Visual Search Agent

Zhangquan Chen<sup>1,†</sup> Yaoxin Niu<sup>1,2,†</sup> Xiang An<sup>3</sup> Mingze Sun<sup>1</sup> Zhumei Wang<sup>4</sup> Chih-Ting Liao<sup>5</sup> Hongkun Cao<sup>2</sup> Ruqi Huang<sup>1,∗</sup>

<sup>1</sup> Tsinghua University <sup>2</sup> Peng Cheng Laboratory <sup>3</sup> LMMs-Lab <sup>4</sup> Beijing Institute of Technology <sup>5</sup> University of New South Wales

<sup>†</sup> Equal contribution. Corresponding author. Code: github.com/zhangquanchen/HAPRL

![](images/2519a8c287c4498f3a2a2f11b3be24b0c0d2fe874631cec98d7349309e4e16a2.jpg)  
Figure 1: Human-anchored process supervision. Unlike outcome-only RL, HaPRL uses human traces to distinguish grounded from lucky correct searches. This improves initialization by 4.00 points and yields a 6.7× larger gain in the same subsequent outcome-based stage.

## ABSTRACT

Multi-turn visual search agents answer questions about high-resolution images by iteratively deciding where to look. Reinforcement learning for these agents rewards only the final answer, leaving the search process unsupervised. Consequently, faulty routes in which the reasoning process is erroneous yet the final result is correct arise frequently, which in turn leads to ineffective training, i.e., scaling along the wrong paths. In this paper, we introduce HaPRL, the first framework to reinforce the search process with human search behavior. We first build an annotation platform and collect 1K+ human-annotated data with fine-grained behavioral signals. During training, a carefully designed judge scores each rollout with task-adaptive weights, anchored on the distilled trace of how a human annotator actually searched the same image. Extensive experiments show that HaPRL consistently outperforms outcome-based RL, and early-stage process supervision yields 6.7× more improvement in subsequent outcome-based scaling. Our results also demonstrate the importance of aligning model behavior with human process annotation signals, which offer new insight into the training of foundation models.

## 1 INTRODUCTION

High-resolution visual reasoning often depends on evidence that occupies only a small fraction of an image. Human vision handles this constraint through sequential attention, turning a series of fixations into an efficient search process (Treisman & Gelade, 1980; Itti & Koch, 2001). Multi-turn visual agents follow this principle by interleaving reasoning with crop-and-zoom tool calls, i.e., they choose a region, inspect the resulting view, and decide where to look next (Wu & Xie, 2024; Shao et al., 2024a; Hu et al., 2024; Zheng et al., 2025; Lai et al., 2025; Chen et al., 2025b;a). A capable agent should reach the correct answer through a search trajectory that acquires the supporting evidence.

Training a capable visual search agent through supervised fine-tuning alone amounts to off-policy behavior cloning, which limits generalization beyond the demonstration distribution (Figure 1). Many methods therefore apply reinforcement learning after a cold start to unlock the agent’s potential. Existing visual-search agents commonly adopt outcome-based rewards that score only the final answer (Zheng et al., 2025; Wang et al., 2025a; Lai et al., 2025; Liu et al., 2025). Such outcome-based rewards are attractive because correctness is inexpensive to verify (Shao et al., 2024b; DeepSeek-AI, 2025; Lambert et al., 2024). However, human annotators also produce fine-grained behavioral signals through their mouse interactions during the search process, such as movement trajectories, cursor velocity, dwell duration, etc.. These signals carry rich information about search quality but are entirely discarded by outcome-based training. As a result, a correct answer reached by an incorrect search hacks full reward. On multiple-choice and short-answer tasks, for example, an agent may inspect irrelevant regions, miss the target evidence, and still arrive at the right answer through random guessing. Rewarding such trajectories reinforcesfaulty routes and causes optimization to scale along wrong search paths (Figure 1).

More fundamentally, outcome-based GRPO (Shao et al., 2024b) standardizes rewards within each sampled group and provides no mechanism to rank correct trajectories by search quality. That is, a trajectory that localizes the sign, zooms in, and reads it receives the same credit as one that crops empty regions and guesses. We verify this on VisualProbe Hard, where outcome-based RL raises accuracy by 28.6% while the judged search quality of its correct answers drops by 8.4% (Section 4.2), i.e., the agent answers more questions correctly yet searches worse doing so.

Process supervision addresses analogous failures in mathematical reasoning by evaluating intermediate steps (Uesato et al., 2022; Lightman et al., 2024; Wang et al., 2024; Luo et al., 2024), but perceptual search does not decompose into self-contained steps that can be labeled correct or incorrect. The value of a crop depends on the question, the regions already inspected, and whether the action makes progress toward the target. A useful process signal must therefore evaluate the trajectory as a whole rather than classify isolated coordinates. Human search behavior provides the missing reference, recording which fine-grained regions attracted attention, how the view was refined, and where the evidence was ultimately found. Addressing this requires solving two challenges: G1) Human process signal acquisition: systematically capture the fine-grained behavioral signals that human annotators produce during visual search; and G2) Dense process alignment: align agent training with these human signals to provide graded supervision.

For (G1), we build a browser-based annotation platform and collect human search traces of 1,104 visual-search questions. The platform logs timestamped pointer movements, hovers, and zooms over the full-resolution image, preserving how annotators attend to and refine fine-grained regions. Each interaction stream is distilled into a compact round-by-round trace of the inspected regions and discovered evidence. These traces serve as instance-specific anchors that reveal where useful evidence lies and how much visual refinement the question requires. For (G2), we introduce HaPRL, a training paradigm that aligns visual-search agents with these human process signals at the trajectory level. A frozen vision-language judge evaluates each rollout along five dimensions with task-adaptive weights, anchored on the human-annotated trace of the same image. The resulting process score is multiplied by binary answer correctness, so incorrect answers receive no credit while correct trajectories are ranked by search quality.

Our contributions can be summarized as follows.

• We develop a platform and collect 1,104 human-annotated visual-search examples with processlevel information, distilled into round-by-round traces of human attention and evidence acquisition.

• We introduce HaPRL, the first training paradigm that aligns visual-search agents with human process signals. The answer-gated, task-adaptive reward ranks trajectories anchored on traces.

• Extensive experiments across four backbones and six benchmarks show that HaPRL consistently outperforms outcome-based RL. Moreover, process annotation scales favorably, i.e., early-stage process supervision provides an initialization from which a subsequent outcome-based stage yields 6.7× (6.13 vs. 0.91) more improvement than the same stage without it.

• Our results highlight the importance of aligning agent behavior with human process annotation signals, which offers practical insights for future efforts beyond visual search.

## 2 RELATED WORK

Visual Search Agents. Visual search enables multimodal large language models (MLLMs) to inspect selected regions instead of relying on a single global view. V<sup>∗</sup> combines LLM-guided search with multimodal reasoning to locate small targets in high-resolution images (Wu & Xie, 2024), while Visual Sketchpad equips models with drawing and specialist vision tools for intermediate visual reasoning (Hu et al., 2024). Recent work learns such interactions through post-training. DeepEyes develops active perception with end-to-end reinforcement learning (Zheng et al., 2025), and Pixel Reasoner combines instruction tuning with curiosity-driven reinforcement learning over visual operations (Wang et al., 2025a). Chain-of-Focus learns adaptive search and zooming through SFT followed by outcome- and format-based reinforcement learning (Zhang et al., 2025), while Mini-o3 scales visual-search trajectories through diverse cold-start data and over-turn masking (Lai et al., 2025). Despite different tools and training recipes, these methods rely primarily on terminal correctness or hand-designed auxiliary signals. We instead use human search traces as instance-specific references for evaluating complete evidence-acquisition trajectories.

Process Supervision for Reasoning. Process supervision evaluates intermediate reasoning rather than relying exclusively on terminal outcomes. In mathematical reasoning, process-based feedback reduces reasoning errors among final-answer-correct solutions (Uesato et al., 2022), and PRM800K scales human step-level feedback to the MATH dataset (Lightman et al., 2024). Math-Shepherd and OmegaPRM reduce annotation costs by constructing process supervision automatically through repeated completions and tree search (Wang et al., 2024; Luo et al., 2024). Multimodal PRMs extend step-level verification to image-conditioned mathematical reasoning, i.e. VisualPRM and MM-PRM score candidate derivations for Best-of-N inference (Wang et al., 2025b; Du et al., 2025), whereas URSA also incorporates process rewards into online policy optimization (Luo et al., 2025). These methods assess textual reasoning steps whose correctness can be evaluated individually. Visual search differs because every perceptual action changes the evidence available to subsequent decisions. Accordingly, we evaluate the complete search trajectory against human behavior and use this trajectory-level signal for reinforcement learning.

Rubric-Based Reinforcement Learning. LLM judges provide scalable model-based evaluation (Zheng et al., 2023; Chen et al., 2026), and recent work converts structured criteria into reinforcement signals. Rubrics as Rewards uses instance-specific rubrics for on-policy training in medical and scientific domains (Gunjal et al., 2025). Reinforcement Learning from Checklist Feedback extracts instruction-specific criteria and aggregates judgments from language models and specialized verifiers (Viswanathan et al., 2025), while Rubric Anchors scales open-ended alignment with a large collection of human- and model-authored rubrics (Huang et al., 2025). These methods primarily evaluatefinal responses when a single verifiable outcome is unavailable. In visual search, the final answer is verifiable but does not reveal whether the agent acquired the necessary evidence. HaPRL therefore applies a task-adaptive rubric to the search process, anchors the evaluation on a matched human trace, and gates the process score by answer correctness.

## 3 METHOD

Method Overview. We study a visual-search policy $\pi _ { \theta }$ that answers a question q about a highresolution image $I _ { 0 }$ by interleaving language reasoning with crop actions, following the tool-use protocol of Mini-o3 (Lai et al., 2025). At round t, the policy emits a reasoning segment $h _ { t }$ and either

terminates with an answer a or invokes $c _ { t } = ( v _ { t } , b _ { t } )$ , where $v _ { t }$ identifies a previously observed view and $b _ { t }$ specifies a normalized bounding box. The environment executes the crop and returns the resulting observation $o _ { t }$ . A trajectory containing $T$ crop actions is

$$
\tau = ( h _ { 1 } , c _ { 1 } , o _ { 1 } , \dots , h _ { T } , c _ { T } , o _ { T } , h _ { T + 1 } , a ) .\tag{1}
$$

The central idea of HaPRL is to use human search behavior as a reward anchor rather than a policy demonstration. As shown in Figure 2, the policy samples multiple trajectories for each image question pair. A frozen vision-language judge compares each search process with the matched human trace and produces a task-adaptive process score, while a separate answer judge evaluates terminal correctness. Gating the process score by correctness preserves the answer objective and introduces graded preferences among correct trajectories. Human traces affect training only through this reward pathway and never enter the policy context.

In Section 3.1, we describe how human interaction signals are collected and distilled into reference traces. In Section 3.2, we introduce the human-anchored process judge. In Section 3.3, we present the policy optimization with answer-gated reward. Algorithms 1 and 2 summarize the data-construction and training procedures.

![](images/b1b4d92fa99510fa9c5c65368d50b3a4a6b332cfd38715cfad91ab9ebf9f8ac4.jpg)  
Figure 2: HaPRL converts human search behavior into graded rewards for visual-search training. Human interaction logs are distilled into reference traces. A frozen vision-language judge then compares each sampled trajectory with its matched trace along five task-adaptive dimensions, while answer correctness gates the weighted process score. Finally, group-relative optimization uses the resulting rewards to update the policy model.

## 3.1 HUMAN PROCESS DATA COLLECTION

Interaction logging. We build a browser-based platform and collect human search behavior for 1,104 questions from the VisualProbe training set. For each question, the human annotator searches the full-resolution image and uses the mouse to mark the region containing the supporting evidence. The platform automatically records the interaction stream $E ^ { * } = \left( e _ { 1 } , \dots , e _ { M } \right)$ throughout this process. We represent each event as $e _ { m } = ( \kappa _ { m } , t _ { m } , \mathbf { p } _ { m } , v _ { m } , \alpha _ { m } , b _ { m } )$ , where $\kappa _ { m }$ is the event type, $t _ { m }$ is its timestamp, $\mathbf { p } _ { m }$ is the pointer position, $v _ { m }$ and $\alpha _ { m }$ are its velocity and acceleration, and $b _ { m }$ is an optional zoom or selection box. These measurements retain the temporal and spatial structure ofthe search, including scan direction, dwell, redirection, and progressive refinement.

Trace distillation. Raw interaction streams are long, redundant, and misaligned with the discrete rounds of an agent trajectory. An offline distillation stage converts $E ^ { * }$ into an ordered textual trace $H ^ { * } = ( r _ { 1 } ^ { * } , \ldots , r _ { L } ^ { * } )$ . Each round $r _ { l } ^ { \ast }$ records the inspected region, its source view, the normalized crop when available, and the evidence revealed at that stage. Consecutive pointer samples that do not change the inspected region are suppressed, while meaningful dwell, redirection, and coarse-to-fine refinement are retained. The resulting training example is $\bar { z } = ( I _ { 0 } , q , a ^ { * } , E ^ { * } , H ^ { * } )$ , where $a ^ { * }$ denotes the reference answer. The policy observes only $( I _ { 0 } , q )$ . The distilled trace and annotation metadata are reserved for reward computation. Thus, $H ^ { * }$ identifies the evidence requirements and search difficulty of an instance without prescribing the exact actions that the policy must reproduce.

## 3.2 HUMAN-ANCHORED PROCESS JUDGE

Human-anchored assessment. A frozen vision-language judge $J _ { \phi }$ takes $( I _ { 0 } , q , \tau , H ^ { \ast } )$ as input. The matched human trace anchors the evaluation to the evidence requirements and search difficulty of the specific instance without prescribing the annotator’s exact actions. The judge returns a score vector $\mathbf { s } ( \tau ) \in [ 0 , 1 ] ^ { 5 }$ over five dimensions. (i) Target semantics measures whether the search follows the queried object, attribute, text, count, or relation. (ii) Evidence acquisition measures whether the observed views provide sufficient support for the answer. (iii) Search progress captures information gain across successive actions. (iv) Tool discipline evaluates the validity and granularity of crop operations. (v) Communication discipline evaluates whether the reasoning remains concise and grounded in visible evidence. More details are provided in Appendix A.3.

Task-adaptive scoring. Different visual-search tasks require different forms of evidence, so the five criteria should not contribute equally. The judge assigns each instance to a profile $p \in \{ \mathrm { O C R } / \mathrm { t e x t } .$ , count/relation, attribute, general search} and applies the corresponding weights $\mathbf { w } ^ { ( p ) }$ OCR/text places greater emphasis on evidence legibility, count/relation on spatial coverage and crossregion consistency, and attribute on precise target grounding. General search adopts a more balanced weighting across the shared criteria. Given the vector $\mathbf { s } ( \tau )$ , we compute the process score as:

$$
r _ { \mathrm { p r o c } } ( \tau ) = \Big \langle \mathbf { w } ^ { ( p ) } , \mathbf { s } ( \tau ) \Big \rangle = \sum _ { k = 1 } ^ { 5 } w _ { k } ^ { ( p ) } s _ { k } ( \tau ) , \qquad \sum _ { k = 1 } ^ { 5 } w _ { k } ^ { ( p ) } = 1 .\tag{2}
$$

## 3.3 ANSWER-GATED PROCESS OPTIMIZATION

Correctness-gated reward. Process quality should distinguish successful trajectories without rewarding an incorrect answer. A separate frozen answer judge $J _ { \mathrm { a n s } }$ compares the extracted answer $a ( \tau )$ with the reference $a ^ { * }$ :

$$
r _ { \mathrm { o u t } } ( \tau ) = J _ { \mathrm { a n s } } ( q , a ( \tau ) , a ^ { * } ) \in \{ 0 , 1 \} .\tag{3}
$$

We invoke the process judge only when $r _ { \mathrm { o u t } } ( \tau ) = 1$ and define the training reward as

$$
R _ { \mathrm { H a P R L } } ( \tau ) = r _ { \mathrm { o u t } } ( \tau ) r _ { \mathrm { p r o c } } ( \tau ) .\tag{4}
$$

The gate assigns zero reward to incorrect trajectories and ranks correct trajectories by evidenceacquisition quality. In contrast, the outcome-based baseline uses $R _ { \mathrm { { O u t c o m e - R L } } } ( \tau ) = r _ { \mathrm { { o u t } } } ( \tau )$ and assigns the same reward to grounded, redundant, and lucky correct trajectories.

Group-relative optimization. For each input $x = ( I _ { 0 } , q )$ , we sample G trajectories $\{ \tau _ { i } \} _ { i = 1 } ^ { G }$ and standardize their rewards within the group:

$$
A _ { i } = { \frac { R _ { i } - \bar { R } } { \sqrt { G ^ { - 1 } \sum _ { j = 1 } ^ { G } ( R _ { j } - \bar { R } ) ^ { 2 } + \epsilon } } } , \qquad \bar { R } = G ^ { - 1 } \sum _ { j = 1 } ^ { G } R _ { j } .\tag{5}
$$

An all-correct group receives a constant outcome reward and therefore yields zero relative advantage. HaPRL preserves within-group variation through $r _ { \mathrm { p r o c } } ,$ allowing successful trajectories to be distinguished by search quality. All-wrong groups remain at zero because the correctness gate suppresses process credit. We optimize these advantages with the clipped GRPO objective (Shao et al., 2024b):

$$
\mathscr { L } ( \theta ) = - \mathbb { E } _ { i , t } [ \operatorname* { m i n } ( \rho _ { i , t } A _ { i } , \mathrm { c l i p } ( \rho _ { i , t } , 1 - \varepsilon , 1 + \varepsilon ) A _ { i } ) ] + \beta D _ { \mathrm { K L } } ( \pi _ { \theta } \| \pi _ { \mathrm { r e f } } ) ,\tag{6}
$$

where $\rho _ { i , t } = \pi _ { \theta } ( y _ { i , t } \mid x , y _ { i , < t } ) / \pi _ { \theta _ { \mathrm { o l d } } } ( y _ { i , t } \mid x , y _ { i , < t } )$ is the token-level importance ratio, and $\pi _ { \mathrm { r e f } }$ is the reference policy used for KL regularization. At inference time, the learned policy interacts directly with the crop tool without judges or human traces.

## 4 EXPERIMENTS

Benchmarks and metrics. We evaluate on VisualProbe Easy/Medium/Hard (Lai et al., 2025), V<sup>∗</sup>Bench QA (Wu & Xie, 2024) and HR-Bench 4K/8K (Wang et al., 2025c). Each agent answers every question with one greedy rollout, capped at a maximum of six tool rounds. Base models are evaluated by answering in a single pass with no tool calls, so their rows report a starting point rather than a search policy. All models share that visual token budget, so no gap below comes from more pixels. Besides, V<sup>∗</sup>QA is graded open-ended by the judge rather than by option matching, which denies the policy the option-elimination shortcut. “Avg.” is the unweighted mean over the six sets. Search quality is scored separately on answer-correct rollouts only, separating how an agent searched from how often it was right. See Appendix B.1 for more details.

Training. Four backbones span two families and scales: Qwen3-VL-4B/8B-Instruct (Bai et al., 2025) and LLaVA-OneVision-1.5-4B/8B (An et al., 2025). None emits the grounding syntax reliably on its own, so each is cold-started on filtered Mini-o3 trajectories (Lai et al., 2025). Two RL conditions follow. Outcome-RL optimizes answer correctness alone, HaPRL optimizes the answer-gated product.

Process annotation from Human. Human annotations cover 1,104 training questions from Visual-Probe. Annotators worked in a browser interface logging hover and zoom motion, and each event stream is distilled into a per-round textual trace.

Hyper-parameters. RL uses GRPO (Shao et al., 2024b) in verl (Sheng et al., 2025) with vLLM rollouts (Kwon et al., 2023): 8 rollouts per prompt, prompt batch 48, actor learning rate $5 \times 1 0 ^ { - 7 }$ and KL coefficient $3 \times 1 0 ^ { - 3 }$ . The judge is a frozen Qwen3-VL-30B-A3B-Instruct. Both arms select checkpoints on the same held-out split. Remaining settings are in Appendices B.2–B.4.

## 4.1 BENCHMARKING OUTCOME-RL-BASED VLMS

Comprehensive Improvements. As shown in Table 1, HaPRL wins all 24 backbone-benchmark cells over Outcome-RL and the base model. Macro-average margins over matched Outcome-RL reach +30.3% on Qwen3-VL-4B (44.26 vs. 33.98) and +15.3% on LLaVA-OneVision-1.5-8B (47.80 vs. 41.46), with +13.8% and +5.8% on the other two.

SFT Trade-off Correction. Cold-start SFT costs 3.00 to 8.71 points of accuracy and enables a policy that can call the crop tool. Outcome-only RL fails to repay that debt on half the backbones, leaving Qwen3-VL-4B 6.09 points below its own base model (33.98 vs. 40.07). HaPRL clears the base model on all four, by +4.19 to +7.40 points. Qwen3-VL-4B on HR-Bench 4K is the cleanest case: the cold start gives up 6.62 points (48.88 vs. 55.50), Outcome-RL a further 1.75, and HaPRL recovers 7.62 of them (56.50). The cold start is a wager that tool use is worth an accuracy deficit, and only the process reward collects on it.

Hard-search Gains. A binary outcome reward takes two values inside a sampled group, so its advantage is zero among the correct rollouts and zero among the incorrect ones. The only gradient it supplies separates right answers from wrong ones. HaPRL gates a rubric score by correctness leaves the incorrect set at zero and turns the correct set into a ranking, placing gradient exactly where outcome supervision has none. Specifically, against the own cold start, Outcome-RL gains +11.91 on V<sup>∗</sup>QA and +8.51 on VisualProbe Easy but only +1.68 and +1.89 on Medium and Hard, the splits demanding the longest search. HaPRL inverts that profile, adding +71.8% on VisualProbe Hard on average and peaking at +144.4% on Qwen3-VL-4B (20.75 vs. 8.49) and +127.7% on LLaVA-OneVision-1.5-8B (27.92 vs. 12.26). That is, an outcome reward can only sharpen a decision the answer prior already informs. A process reward orders the correct rollouts among themselves, and that ordering is where search discipline lives.

## 4.2 SEARCH QUALITY UNDER DIFFERENT SETTINGS

Higher accuracy does not imply better search. Table 2 scores answer-correct rollouts only, and Outcome-RL moves search quality by +0.08 points over its cold start (82.08 vs. 81.99) while falling on two of six benchmarks. HaPRL adds +2.78 (84.77) and improves all six.

Table 1: Accuracy comparison of generalist VLMs, SFT, +outcome-based RL, and +our method (HaPRL) on VisualProbe Easy/Medium/Hard (Lai et al., 2025), $\mathrm { \Delta V ^ { * } Q A }$ (Wu & Xie, 2024), HR-4K/8K (Wang et al., 2025c). All RL post-training methods are trained for three epochs. Avg. is the unweighted mean over the six sets, and the parenthesized value is the HaPRL margin over the matched Outcome-RL control. Best per column of the same backbone in bold.
<table><tr><td rowspan="2">Model</td><td colspan="3">VisualProbe</td><td colspan="3">High-resolution QA</td><td rowspan="2">Avg.</td></tr><tr><td>Easy</td><td>Medium</td><td>Hard</td><td>V*QA</td><td>HR-4K</td><td>HR-8K</td></tr><tr><td>Qwen3-VL-4B-Instruct</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base</td><td>36.88</td><td>16.42</td><td>13.21</td><td>67.16</td><td>55.50</td><td>51.25</td><td>40.07</td></tr><tr><td>+SFT</td><td>27.66</td><td>14.55</td><td>6.60</td><td>46.07</td><td>48.88</td><td>44.38</td><td>31.36</td></tr><tr><td>+SFT+Outcome-RL</td><td>29.08</td><td>17.91</td><td>8.49</td><td>58.64</td><td>47.13</td><td>42.63</td><td>33.98</td></tr><tr><td>+SFT+HaPRL</td><td>41.13</td><td>26.49</td><td>20.75</td><td>68.06</td><td>56.50</td><td>52.63</td><td>44.26 (+10.28)</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base</td><td>34.04</td><td>18.66</td><td>13.21</td><td>71.59</td><td>64.25</td><td>56.38</td><td>43.02</td></tr><tr><td>+SFT</td><td>26.95</td><td>21.64</td><td>19.81</td><td>45.03</td><td>54.75</td><td>45.50</td><td>35.61</td></tr><tr><td>+SFT+Outcome-RL</td><td>34.04</td><td>27.61</td><td>15.09</td><td>56.54</td><td>70.25</td><td>62.25</td><td>44.30</td></tr><tr><td>+SFT+HaPRL</td><td>36.88</td><td>36.57</td><td>20.09</td><td>72.73</td><td>70.38</td><td>65.88 50.42</td><td>(+6.12)</td></tr><tr><td>LLaVA-OneVision-1.5-4B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base</td><td>34.75</td><td>11.94</td><td>13.21</td><td>60.21</td><td>62.75</td><td>58.13</td><td>40.17</td></tr><tr><td>+SFT</td><td>26.95</td><td>18.66</td><td>9.43</td><td>47.12</td><td>64.13</td><td>56.75</td><td>37.17</td></tr><tr><td>+SFT+Outcome-RL</td><td>41.13</td><td>18.66</td><td>17.92</td><td>60.21</td><td>63.50</td><td>56.25</td><td>42.95</td></tr><tr><td>+SFT+HaPRL</td><td>43.97</td><td>20.52</td><td>23.58</td><td>61.26</td><td>64.75</td><td>58.50</td><td>45.43 (+2.48)</td></tr><tr><td>LLaVA-OneVision-1.5-8B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base</td><td>41.13</td><td>11.19</td><td>16.04</td><td>64.40</td><td>62.63</td><td>57.00</td><td>42.07</td></tr><tr><td>+SFT</td><td>31.21</td><td>18.66</td><td>10.38</td><td>53.40</td><td>60.38</td><td>59.25</td><td>38.88</td></tr><tr><td>+SFT+Outcome-RL</td><td>42.55</td><td>16.04</td><td>12.26</td><td>63.87</td><td>60.13</td><td>53.88</td><td>41.46</td></tr><tr><td>+SFT+HaPRL</td><td>50.52</td><td>18.28</td><td>27.92</td><td>65.97</td><td>64.38</td><td>59.75</td><td>47.80 (+6.34)</td></tr></table>

Table 2: The process score (%) on answer-correct rollouts of different methods on Qwen3-VL-4B; ∆ against the cold start. The best results are in bold. Outcome-RL raises answer accuracy while leaving some search quality flat, whereas HaPRL maintains a positive gain on every split.
<table><tr><td></td><td colspan="3">VisualProbe</td><td colspan="3">High-resolution QA</td><td></td></tr><tr><td>Model</td><td>Easy</td><td>Medium</td><td>Hard</td><td>V*QA</td><td>HR-4K</td><td>HR-8K</td><td>Avg.</td></tr><tr><td>+SFT</td><td>84.90</td><td>86.05</td><td>71.54</td><td>80.34</td><td>86.49</td><td>82.64</td><td>81.99</td></tr><tr><td>+SFT+Outcome-RL</td><td>83.55</td><td>86.92</td><td>65.56</td><td>83.22</td><td>87.24</td><td>85.96</td><td>82.08</td></tr><tr><td>∆</td><td>-1.35</td><td>+0.87</td><td>-5.98</td><td>+2.88</td><td>+0.75</td><td>+3.32</td><td>+0.08</td></tr><tr><td>+SFT+HaPRL</td><td>86.71</td><td>87.27</td><td>75.28</td><td>83.24</td><td>88.66</td><td>87.47</td><td>84.77</td></tr><tr><td>∆</td><td>+1.81</td><td>+1.22</td><td>+3.74</td><td>+2.90</td><td>+2.17</td><td>+4.83</td><td>+2.78</td></tr></table>

VisualProbe Hard exposes the failure directly. Outcome-RL lifts accuracy there by +28.6% (8.49 vs. 6.60) while its process score drops 8.4% (65.56 vs. 71.54): the policy answers more hard questions correctly and searches worse doing it. HaPRL recovers +14.8% over that control (75.28 vs. 65.56). An outcome reward is not neutral toward search; it erodes it. Because it cannot tell an answer earned by search from one reached by random guessing, accuracy on the hardest split rises while search qualityfalls. This is reward hacking whose signature is invisible to the metric the field reports.

## 4.3 SCALING BEHAVIOR

Accuracy scales with annotation volume. Accuracy rises at every budget in Table 3 and Figure 3(a), from 36.62 at 200 annotated prompts to 44.26 at 1,104 (+20.9%), with no flattening at the largest budget we collected. Annotating 18% of the prompts already beats Outcome-RL trained on all of them (36.62 vs. 33.98). Returns concentrate on VisualProbe Medium (+12.31) and V<sup>∗</sup>QA (+10.99) and thin out on VisualProbe Hard (+3.77) and HR-Bench 8K (+3.75). Thus, process annotation is meaningful and scales the accuracy.

Table 3: Accuracy of three-epoch training at different data sizes on Qwen3-VL-4B. Accuracy rises monotonically with the number of process-annotated prompts. The best is bold.
<table><tr><td></td><td colspan="3">VisualProbe</td><td colspan="3">High-resolution QA</td><td></td></tr><tr><td>Annotated prompts</td><td>Easy</td><td>Medium</td><td>Hard</td><td>V*QA</td><td>HR-4K</td><td>HR-8K</td><td>Avg.</td></tr><tr><td>200</td><td>31.21</td><td>14.18</td><td>16.98</td><td>57.07</td><td>51.38</td><td>48.88</td><td>36.62</td></tr><tr><td>400</td><td>32.70</td><td>15.19</td><td>17.92</td><td>56.97</td><td>52.88</td><td>48.12</td><td>37.30</td></tr><tr><td>600</td><td>35.46</td><td>17.16</td><td>18.87</td><td>60.40</td><td>54.25</td><td>49.13</td><td>39.21</td></tr><tr><td>800</td><td>37.99</td><td>20.19</td><td>19.21</td><td>63.40</td><td>55.88</td><td>50.94</td><td>41.27</td></tr><tr><td>1000</td><td>40.15</td><td>24.34</td><td>20.55</td><td>65.69</td><td>56.12</td><td>51.68</td><td>43.09</td></tr><tr><td>1104 (full)</td><td>41.13</td><td>26.49</td><td>20.75</td><td>68.06</td><td>56.50</td><td>52.63</td><td>44.26</td></tr><tr><td>∆ (200 → 1104)</td><td>+9.92</td><td>+12.31</td><td>+3.77</td><td>+10.99</td><td>+5.12</td><td>+3.75</td><td>+7.64</td></tr></table>

![](images/5e11550a1ec250be2ac449d2b28bf4f3f4aac9ee3460a05592435c7fdf9830dd.jpg)

![](images/decad3890341c2b395971aad58399a53eb0d6c9742d6784398a73553d0b80d3a.jpg)  
Figure 3: HaPRL scales with the amount of process-annotated data, and injecting human-aligned process data in the early stage enables stronger scaling even when later training uses only the outcome signal. (a) Accuracy grows monotonically with the process-annotation budget. 1/5 annotated data already beat Outcome-RL trained on all of the them. (b) A second, outcome-only stage returns +0.91 points to an Outcome-RL-trained policy and +6.13 points to a HaPRL-trained one (6.7×).

Early process supervision scales better. Table 4 and Figure 3(b) hold stage 2 identical across two arms and vary only stage 1. The same 600-prompt outcome-only continuation returns +2.8% to the Outcome-RL-trained policy (33.54 vs. 32.63) and +16.7% to the HaPRL-trained one (42.76 vs. 36.63), a 6.7× difference produced by the initialization alone. The gap between arms widens from +4.00 to +9.22 points, uniformly across all six benchmarks (+7.73 to +11.06).

That is, process data can first establish correct search behavior, and a later outcome signal amplifies what is already working. Outcome supervision used throughout has no such foundation to build on. It accepts rollouts that reach the right answer through a faulty search, and a policy trained on those trajectories has little real signal left to learn from.

## 4.4 ABLATION STUDY

We further ablate the human reference trace in three ways on Qwen3-VL-4B, as shown in Table 5.

Shuffled pairing. Re-pairing each trace with another question’s trajectory preserves the label distribution and destroys only the alignment, at a cost of 35.3% (28.63 vs. 44.26). Falling 5.35 points below Outcome-RL, and 2.73 below the cold start, is the informative part. A broken regularizer would decay toward the unregularized baseline, whereas a mis-paired trace pushes the policy toward search that was correct for a different image.

Judge without the human trace. Dropping the human-annotated reference trace and letting the frozen judge score the same five-dimensional rubric on its own gives 36.41, only +2.43 over Outcome-RL. Restoring the trace adds +7.85 points (+21.6%, 44.26 vs. 36.41), which is 76% of the full +10.28 margin. The rubric supplies the axes of judgement; the trace supplies where good search sits on those axes for this particular image. A judge denied the trace has to invent that reference from the question alone with its own knowledge. Thus, the human annotated data is important and carries the learning signal.

Table 4: Accuracy of one-epoch training on Qwen3-VL-4B. Stage 1 trains on 400 data under either reward (outcome-based vs. HaPRL); stage 2 continues under the same outcome reward on another 600 data. The best is bold.
<table><tr><td rowspan="2">Training schedule</td><td colspan="3">VisualProbe</td><td colspan="3">High-resolution QA</td><td rowspan="2">Avg.</td></tr><tr><td>Easy</td><td>Medium</td><td>Hard</td><td>V*QA</td><td>HR-4K</td><td>HR-8K</td></tr><tr><td colspan="8">Stage 1 only (400 prompts)</td></tr><tr><td>Öutcome-RL</td><td>28.35</td><td>14.94</td><td>11.21</td><td>54.84</td><td>46.12</td><td>40.30</td><td>32.63</td></tr><tr><td>HaPRL</td><td>32.70</td><td>15.19</td><td>17.92</td><td>56.97</td><td>52.88</td><td>44.12</td><td>36.63</td></tr><tr><td>∆</td><td>+4.35</td><td>+0.25</td><td>+6.71</td><td>+2.13</td><td>+6.76</td><td>+3.82</td><td>+4.00</td></tr><tr><td colspan="8">Stage 1 → outcome-only stage 2 (600 prompts)</td></tr><tr><td>Outcome-RL → Outcome-RL</td><td>28.68</td><td>16.85</td><td>9.34</td><td>57.84</td><td>47.03</td><td>41.51</td><td>33.54</td></tr><tr><td>HaPRL → Outcome-RL</td><td>39.74</td><td>24.58</td><td>19.36</td><td>66.42</td><td>55.21</td><td>51.27</td><td>42.76</td></tr><tr><td>∆</td><td>+11.06</td><td>+7.73</td><td>+10.02</td><td>+8.58</td><td>+8.18</td><td>+9.76</td><td>+9.22</td></tr></table>

Table 5: Ablation study of different training variants. Six-benchmark average accuracy of three-epoch training on Qwen3-VL-4B; ∆ against full HaPRL. Shuffled re-pairs each reference trace with another question’s trajectory, keeping the label distribution intact. Judge-only removes the trace from the judge prompt and leaves the rubric and task weighting intact. Noise perturbs 10% of the recorded dwell and hover events.
<table><tr><td>Variant</td><td>Human trace</td><td>Correct pairing</td><td>Avg.</td><td>∆</td></tr><tr><td>Shuffled process labels</td><td>√</td><td></td><td>28.63</td><td>-15.63</td></tr><tr><td>Judge-only process</td><td></td><td></td><td>36.41</td><td>-7.85</td></tr><tr><td>+ 10% trace noise</td><td>√</td><td>√</td><td>41.74</td><td>-2.52</td></tr><tr><td>HaPRL (full)</td><td>√</td><td>√</td><td>44.26</td><td></td></tr></table>

Robustness to annotation noise. Perturbing 10% of the recorded dwell and hover events costs 5.7% (41.74 vs. 44.26) and still leaves +7.76 over Outcome-RL. What the reward takes from a trace is the order in which regions were worth visiting, and that ordering survives jitter in individual events.

Therefore, the process reward is a channelfor human search behavior rather than a hand-designed prior. Its value tracks howfaithfully a trace is paired with its trajectory.

## 5 CONCLUSION AND LIMITATION

Outcome-based reinforcement learning trains visual-search agents by scoring only the final answer, so a correct result reached through an incorrect search receives full credit and later optimization can scale along the wrong paths. Thus, we introduce HaPRL, thefirstframework that reinforces the search process with human search behavior. We collect 1,104 human traces with fine-grained interaction signals and distill them into round-by-round descriptions of inspected regions and discovered evidence. A frozen judge then scores each rollout under task-adaptive weights, anchored on the matched human trace, and gates that score by answer correctness. Across all benchmarks, HaPRL consistently outperforms matched outcome-based RL in both accuracy and search quality, and exhibits favorable scaling trends. These results have important implications for the training of foundation models and for expansion to more other domains.

Limitation & Future Work. HaPRL offers a new path and training paradigm for future research. A natural next step is to extend this human-process alignment recipe to more domains, especially those in which human annotation signals can play a central role, such as 3D rigging, GUI agents, etc..

## AI USE STATEMENT

In this work, we used generative AI tools to polish manuscript wording after an author-written draft, and to assist with some training/inference codes. All AI-assisted text and code were manually verified by the authors. We take responsibility for the final content of this work.

## ETHICS STATEMENT

This study strictly adheres to the ICLR Code of Ethics. The datasets utilized in our experiments are publicly available, fully anonymized, and do not involve human subjects, privacy infringement, or harmful discrimination concerns.

## REPRODUCIBILITY STATEMENT

Implementation details and hyperparameter configurations are provided in the paper and appendix. Training and evaluation code, together with the released annotations, is available in the public HaPRL repository.

## REFERENCES

Xiang An, Yin Xie, Kaicheng Yang, Wenkang Zhang, Xiuwei Zhao, Zheng Cheng, Yirui Wang, Songcen Xu, Changrui Chen, Chunsheng Wu, Huajie Tan, Chunyuan Li, Jing Yang, Jie Yu, Xiyao Wang, Bin Qin, Yumeng Wang, Zizhen Yan, Ziyong Feng, Ziwei Liu, Bo Li, and Jiankang Deng. LLaVA-OneVision-1.5: Fully open framework for democratized multimodal training. arXiv preprint arXiv:2509.23661, 2025.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631, 2025.

Zhangquan Chen, Xufang Luo, and Dongsheng Li. Visrl: Intention-driven visual perception via reinforced reasoning. arXiv preprint arXiv:2503.07523, 2025a.

Zhangquan Chen, Ruihui Zhao, Chuwei Luo, Mingze Sun, Xinlei Yu, Yangyang Kang, and Ruqi Huang. Sifthinker: Spatially-aware image focus for visual reasoning. arXiv preprint arXiv:2508.06259, 2025b.

Zhangquan Chen, Manyuan Zhang, Xinlei Yu, Xiang An, Bo Li, Xin Xie, ZiDong Wang, Mingze Sun, Shuang Chen, Hongyu Li, et al. 4dthinker: Thinking with 4d imagery for dynamic spatial understanding. arXiv preprint arXiv:2605.05997, 2026.

DeepSeek-AI. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Lingxiao Du, Fanqing Meng, Zongkai Liu, Zhixiang Zhou, Ping Luo, Qiaosheng Zhang, and Wenqi Shao. MM-PRM: Enhancing multimodal mathematical reasoning with scalable step-level supervision. arXiv preprint arXiv:2505.13427, 2025.

Anisha Gunjal, Anthony Wang, Elaine Lau, Vaskar Nath, Yunzhong He, Bing Liu, and Sean Hendryx. Rubrics as rewards: Reinforcement learning beyond verifiable domains. arXiv preprint arXiv:2507.17746, 2025.

Yushi Hu, Weijia Shi, Xingyu Fu, Dan Roth, Mari Ostendorf, Luke Zettlemoyer, Noah A. Smith, and Ranjay Krishna. Visual sketchpad: Sketching as a visual chain of thought for multimodal language models. Advances in Neural Information Processing Systems (NeurIPS), 2024.

Zenan Huang, Yihong Zhuang, Guoshan Lu, Zeyu Qin, Haokai Xu, Tianyu Zhao, Ru Peng, Jiaqi Hu, Zhanming Shen, Xiaomeng Hu, Xijun Gu, Peiyi Tu, Jiaxin Liu, Wenyu Chen, Yuzhuo Fu, Zhiting Fan, Yanmei Gu, Yuanyuan Wang, Zhengkai Yang, Jianguo Li, and Junbo Zhao. Reinforcement learning with rubric anchors. arXiv preprint arXiv:2508.12790, 2025.

Laurent Itti and Christof Koch. Computational modelling of visual attention. Nature Reviews Neuroscience, 2(3):194–203, 2001.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with PagedAttention. In Proceedings of the 29th Symposium on Operating Systems Principles (SOSP), 2023.

Xin Lai, Junyi Li, Wei Li, Tao Liu, Tianjian Li, and Hengshuang Zhao. Mini-o3: Scaling up reasoning patterns and interaction turns for visual search. arXiv preprint arXiv:2509.07969, 2025.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V. Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al. Tulu 3: Pushing frontiers in¨ open language model post-training. arXiv preprint arXiv:2411.15124, 2024.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In International Conference on Learning Representations (ICLR), 2024.

Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visual agentic reinforcement fine-tuning. arXiv preprint arXiv:2505.14246, 2025.

Liangchen Luo, Yinxiao Liu, Rosanne Liu, Samrat Phatale, Meiqi Guo, Harsh Lara, Yunxuan Li, Lei Shu, Yun Zhu, Lei Meng, Jiao Sun, and Abhinav Rastogi. Improve mathematical reasoning in language models by automated process supervision. arXiv preprint arXiv:2406.06592, 2024.

Ruilin Luo, Zhuofan Zheng, Yifan Wang, Xinzhe Ni, Zicheng Lin, Songtao Jiang, Yiyao Yu, Chufan Shi, Lei Wang, Ruihang Chu, Jin Zeng, and Yujiu Yang. Unlocking multimodal mathematical reasoning via process reward model. arXiv preprint arXiv:2501.04686, 2025.

Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. ZeRO: Memory optimizations toward training trillion parameter models. In Proceedings ofthe International Conferencefor High Performance Computing, Networking, Storage and Analysis (SC), 2020.

Hao Shao, Shengju Qian, Han Xiao, Guanglu Song, Zhuofan Zong, Letian Wang, Yu Liu, and Hongsheng Li. Visual CoT: Advancing multi-modal language models with a comprehensive dataset and benchmark for chain-of-thought reasoning. In Advances in Neural Information Processing Systems (NeurIPS), 2024a.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024b.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. HybridFlow: A flexible and efficient RLHF framework. In Proceedings ofthe Twentieth European Conference on Computer Systems (EuroSys), 2025.

Anne M. Treisman and Garry Gelade. A feature-integration theory of attention. Cognitive Psychology, 12(1):97–136, 1980.

Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, and Irina Higgins. Solving math word problems with process- and outcome-based feedback. arXiv preprint arXiv:2211.14275, 2022.

Vijay Viswanathan, Yanchao Sun, Shuang Ma, Xiang Kong, Meng Cao, Graham Neubig, and Tongshuang Wu. Checklists are better than reward models for aligning language models. arXiv preprint arXiv:2507.18624, 2025.

Haozhe Wang, Alex Su, Weiming Ren, Fangzhen Lin, and Wenhu Chen. Pixel reasoner: Incentivizing pixel-space reasoning with curiosity-driven reinforcement learning. arXiv preprint arXiv:2505.15966, 2025a.

Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. Math-shepherd: Verify and reinforce LLMs step-by-step without human annotations. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (ACL), pp. 9426–9439, 2024.

Weiyun Wang, Zhangwei Gao, Lianjie Chen, Zhe Chen, Jinguo Zhu, Xiangyu Zhao, Yangzhou Liu, Yue Cao, Shenglong Ye, Xizhou Zhu, Lewei Lu, Haodong Duan, Yu Qiao, Jifeng Dai, and Wenhai Wang. VisualPRM: An effective process reward model for multimodal reasoning. arXiv preprint arXiv:2503.10291, 2025b.

Wenbin Wang, Liang Ding, Minyan Zeng, Xiabin Zhou, Li Shen, Yong Luo, Wei Yu, and Dacheng Tao. Divide, conquer and combine: A training-free framework for high-resolution image perception in multimodal large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 7907–7915, 2025c.

Penghao Wu and Saining Xie. V\*: Guided visual search as a core mechanism in multimodal LLMs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 13084–13094, 2024.

Xintong Zhang, Zhi Gao, Bofei Zhang, Pengxiang Li, Xiaowen Zhang, Yang Liu, Tao Yuan, Yuwei Wu, Yunde Jia, Song-Chun Zhu, and Qing Li. Adaptive chain-of-focus reasoning via dynamic visual search and zooming for efficient VLMs. arXiv preprint arXiv:2505.15436, 2025.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-judge with MT-Bench and chatbot arena. In Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track, 2023.

Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. LlamaFactory: Unified efficient fine-tuning of 100+ language models. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (System Demonstrations), 2024.

Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, and Xing Yu. DeepEyes: Incentivizing “thinking with images” via reinforcement learning. arXiv preprint arXiv:2505.14362, 2025.

## A ADDITIONAL METHOD DETAILS

## A.1 ALGORITHMS

Algorithm 1 constructs the human process traces used as reward anchors. Algorithm 2 is the subsequent training loop: sampled trajectories are scored by a frozen answer judge, process credit is assigned only when the answer is correct, and GRPO updates the policy from group-relative advantages. Human traces and both judges are used only in this training loop; inference retains the policy and the crop tool.

Algorithm 1 Human process data construction   
Require: VisualProbe training questions $\{ ( I _ { 0 } , q , a ^ { * } ) \}$   
Ensure: Dataset $\mathcal { D } = \{ ( I _ { 0 } , q , a ^ { \ast } , E ^ { \ast } , H ^ { \ast } ) \}$   
1: $\mathcal { D }  \emptyset$   
2: for each question $( I _ { 0 } , q , a ^ { * } )$ do   
3: Annotator searches $I _ { 0 }$ and marks the region containing the supporting evidence   
4: Log the interaction stream $E ^ { * } = ( e _ { 1 } , \ldots , e _ { M } ) , e _ { m } = \bar { ( } \kappa _ { m } , t _ { m } , \bar { \mathbf { p } } _ { m } , \bar { v } _ { m } , \alpha _ { m } , b _ { m } )$   
5: Distill $E ^ { * }$ into a round-by-round trace $H ^ { * } = ( r _ { 1 } ^ { * } , \ldots , r _ { L } ^ { * } )$   
6: Suppress pointer samples that do not change the inspected region   
7: Retain dwell, redirection, and coarse-to-fine refinement   
8: $\mathcal { D } \gets \mathcal { D } \cup \{ ( I _ { 0 } , q , a ^ { * } , E ^ { * } , H ^ { * } ) \}$   
9: end for   
10: return D ▷ the policy observes only $( I _ { 0 } , q )$

```latex
Algorithm 2 HaPRL training with answer-gated process reward
Require: Policy $\pi _ { \theta } ,$ frozen judges $J _ { \mathrm { a n s } }$ and $J _ { \phi } ,$ dataset ${ \mathcal { D } } ,$ group size G
1: for each training prompt $x = ( \underline { { I } } _ { 0 } , q )$ with matched trace $H ^ { * }$ and answer $a ^ { * }$ do
2: Sample G trajectories $\{ \tau _ { i } \} _ { i = 1 } ^ { G }$ from $\pi _ { \theta }$ with the crop tool
3: for $i = 1$ to G do
4: $r _ { \mathrm { o u t } } ( \tau _ { i } ) \gets J _ { \mathrm { a n s } } ( q , a ( \tau _ { i } ) , a ^ { * } )$
5: if $r _ { \mathrm { o u t } } ( \tau _ { i } ) = 0$ then
6: $R _ { i } \gets 0$
7: else
8: $\mathbf { s } ( \tau _ { i } ) , p \gets J _ { \phi } ( I _ { 0 } , q , \tau _ { i } , H ^ { * } )$
9: Clip $\mathbf { s } ( \tau _ { i } )$ to $[ 0 , 1 ]$ and normalize $\mathbf { w } ^ { ( p ) }$
10: $r _ { \mathrm { p r o c } } ( \tau _ { i } ) \gets \langle \mathbf { w } ^ { ( p ) } , \mathbf { s } ( \tau _ { i } ) \rangle$
11: $R _ { i }  r _ { \mathrm { o u t } } ( \tau _ { i } ) r _ { \mathrm { p r o c } } ( \tau _ { i } )$
12: end if
13: end for
14: Compute group-relative advantages $A _ { i }$ from $\{ R _ { i } \} _ { i = 1 } ^ { G }$
15: Update θ with clipped GRPO and KL regularization toward $\pi _ { \mathrm { r e f } }$
16: end for
```

## A.2 HUMAN-TRACE REPRESENTATION

For each annotated example, the raw event stream contains timestamps, cursor positions, hover and zoom events, selected regions, and the final response. We distill it offline into an ordered textual trace. Each retained round states the region under inspection, the normalized crop when available, and the evidence revealed by that view. Low-level pointer motion that does not change the inspected region is suppressed. This produces a compact semantic anchor that can be compared with the model rollout without requiring exact action or token alignment.

The process judge receives the original image, question, serialized model rollout, the matched human trace, and compact annotation metadata. The trace is identified by the same sample key as the rollout; it is never re-paired across examples except in the shuffled-reference ablation.

## A.3 STRUCTURED PROCESS RUBRIC

The judge returns one score in [0, 1] for each of five dimensions. Target semantics measures whether the search follows the queried object, attribute, text, count, or relation. Evidence acquisition measures whether the observed views actually support the answer. Search progress rewards information gain and useful recovery while penalizing drift and repeated crops. Tool discipline covers valid source views, bounding boxes, and crop granularity. Communication discipline measures whether the reasoning is concise, consistent, and tied to visible evidence.

Table 6: Task-adaptive weights over the five process dimensions. Columns denote target semantics (Tar.), evidence acquisition (Evd.), search progress (Prog.), tool discipline (Tool), and communication discipline (Com.).
<table><tr><td>Profile</td><td>Tar.</td><td>Evd.</td><td>Prog.</td><td>Tool</td><td>Com.</td></tr><tr><td>General search</td><td>.20</td><td>.30</td><td>.25</td><td>.15</td><td>.10</td></tr><tr><td>OCR / text</td><td>.15</td><td>.40</td><td>.20</td><td>.15</td><td>.10</td></tr><tr><td>Count / relation</td><td>.20</td><td>.35</td><td>.25</td><td>.10</td><td>.10</td></tr><tr><td>Attribute</td><td>.25</td><td>.30</td><td>.20</td><td>.15</td><td>.10</td></tr></table>

The judge predicts the task profile from the question and annotations. Scores are clipped to [0, 1], profile weights are normalized to sum to one, and the weighted total is recomputed outside the model response. This prevents a malformed or internally inconsistent judge output from directly setting the reward.

## A.4 REWARD COMPUTATION

For every sampled trajectory, training applies the following sequence, also listed in Algorithm 2:

1. Extract the terminal answer and obtain the binary semantic-match score $r _ { \mathrm { o u t } }$ from the answer judge.

2. If $r _ { \mathrm { o u t } } = 0 .$ , set $R _ { \mathrm { H a P R L } } = 0$ and skip process evaluation.

3. Otherwise, evaluate the rollout against its matched human trace, select the task profile, and compute $r _ { \mathrm { p r o c } }$ from the structured rubric.

4. Set $R _ { \mathrm { H a P R L } } = r _ { \mathrm { o u t } } r _ { \mathrm { p r o c } } ,$ standardize rewards within the rollout group, and update the policy with GRPO.

This ordering makes the gate semantic rather than cosmetic: fluent reasoning, valid tool syntax, or close imitation of the human trace cannot earn reward when the final answer is wrong. Conversely, answer-correct rollouts retain a continuous preference signal that outcome-only supervision discards.

## B EXPERIMENTAL DETAILS

## B.1 BENCHMARKS

VisualProbe (Lai et al., 2025) splits questions by search difficulty, defined by target size and distractor density. V<sup>∗</sup>Bench QA (Wu & Xie, 2024) scores the correct option text as an open-ended answer rather than as multiple choice, which removes the option-elimination shortcut. HR-Bench (Wang et al., 2025c) pairs the same questions at 4K and 8K resolution, so the 8K split isolates the cost of searching a larger frame.

Visual token budget. Every rollout, in training and in evaluation, may take at most six tool rounds and hold at most six images in context. Each image is rescaled into the range $4 \times 1 0 ^ { 4 }$ to $1 0 ^ { 6 }$ pixels. The budget is what makes the crop tool a real decision: magnifying a region costs one of six slots, so an agent that wastes rounds has fewer left for the target. All three agent conditions receive the identical budget, so no reported gap between them reflects one model seeing more pixels.

Metrics details. Answer accuracy is a binary semantic match produced by the answer judge, which compares the string inside <answer> against the reference and accepts paraphrases and formatting differences. VisualProbe is natively short-answer and HR-Bench natively multiple choice. V<sup>∗</sup>Bench is also multiple choice, but we score its correct option text as a free-form answer, which stops the policy from recovering the answer by eliminating distractors instead of searching.

## B.2 COLD START

Two filters reduce the 7,267-trajectory Mini-o3 cold-start corpus (Lai et al., 2025). The first keeps trajectories with at most six assistant turns (6,757 left), matching the tool-round cap used in RL. The second drops trajectories dominated by recovery phrases or repeated long sentences (6,057 left), both of which survive supervised training and reappear as degenerate loops during rollout.

Fine-tuning updates the language model with the vision tower and multimodal projector frozen. We use learning rate $5 \times 1 0 ^ { - 6 }$ for five epochs, sequence cutoff 32768, and per-device batch 1 with gradient accumulation 4, under DeepSpeed ZeRO-3 offload (Rajbhandari et al., 2020) in LLaMA-Factory (Zheng et al., 2024). Images use the same pixel range as RL.

Cold-start loss is a poor predictor of multi-turn health: checkpoints with lower loss frequently produce repeated crops and recovery loops. We therefore select the RL initialization by rollout quality on 20 held-out prompts, scoring answer rate, valid grounding rate, overlong rate, and mean response length.

## B.3 REINFORCEMENT LEARNING

Table 7 gives the full RL configuration; the two conditions differ only in the optimization total. Both call the answer judge and the process judge and log all five rubric dimensions. Process quality therefore stays observable in the outcome-only arm, where it contributes nothing to the gradient.

Table 7: RL configuration. The last row is the only difference between the two conditions.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Algorithm</td><td>GRPO (Shao et al., 2024b)</td></tr><tr><td>Framework</td><td>verl (Sheng et al., 2025), vLLM rollout (Kwon et al., 2023)</td></tr><tr><td>Rollouts per prompt</td><td>8</td></tr><tr><td>Prompt batch size</td><td>48</td></tr><tr><td>PPO mini / micro batch</td><td> $8 / 1 \mathrm { p e r } \mathrm { G P U }$ </td></tr><tr><td>Actor learning rate</td><td> $5 \times \bar { 1 0 } ^ { - 7 }$ </td></tr><tr><td>KL loss coefficient</td><td> $3 \times 1 0 ^ { - 3 }$  (low-variance estimator)</td></tr><tr><td>Rollout temperature / top-p</td><td>1.0 / 1.0</td></tr><tr><td>Max prompt / response tokens</td><td>8192 / 8192</td></tr><tr><td>Max tool rounds</td><td>6</td></tr><tr><td>Max images per context</td><td>6</td></tr><tr><td>Image pixel range</td><td> $4 \times 1 0 ^ { 4 }$  to 10⁶</td></tr><tr><td>Hardware</td><td>8 × H100 or 8 × H20</td></tr></table>

## B.4 JUDGE

Both judges use a frozen Qwen3-VL-30B-A3B-Instruct served behind an OpenAI-compatible endpoint. The process judge is called at temperature 0 with structured JSON output. It receives the original image, the full multi-turn rollout rendered as text, and the human reference trace for the same question. Three call outcomes are logged: vision when the image is accepted, text fallback when the image call fails and the judge is retried on text alone, and failed after three attempts. We serialize judge-heavy jobs and halt training when the fallback rate grows, since a degraded judge silently flattens the process reward.

The answer judge calls the same model through a separate text-only prompt and returns a binary semantic-correctness score. It is shared by both training conditions and all evaluations.

## B.5 PROCESS ANNOTATION

Annotators answered 1,104 VisualProbe training questions in a browser interface over the fullresolution image. The interface logs a timestamped event stream, with event types covering canvas entry, hover movement, zoom start, and zoom motion, together with the cursor position and the region under inspection. Annotators also record the bounding boxes they settled on as the evidence for their answer.

Raw event streams never reach the judge. Each stream is distilled into a short per-round textual trace naming the region the annotator inspected and what they found there. That distilled trace is the human reference the judge reads. Annotation is collected once per question and reused across every run in the paper, and it never enters the policy’s context at training or inference time.

Annotation cost. A single annotator produced the whole collection. One question takes about 1.5 minutes on average, so the 1,104 questions amount to roughly 27.6 person-hours. Because each trace is collected once and reused by every run reported in this paper, this cost is paid once for the entire study rather than per training run or per backbone. Appendix C.1 weighs it against cheaper reward references, including one that needs only the final evidence box and takes about ten seconds per question.

Ablation construction. The shuffled-label variant permutes the mapping from questions to reference traces, so each rollout is judged against a trace collected for a different question. The label distribution is unchanged and only the correspondence is destroyed. The noise variant perturbs 10% of the recorded dwell and hover events in each stream before distillation. The judge-only variant removes the reference trace from the judge prompt and leaves the rest of it, including the rubric and the task-profile weighting, intact.

## C ADDITIONAL EXPERIMENTS

Three experiments separate the contribution of the human trace from three things that could be mistaken for it, i.e., (i) the contribution of any evidence specification, (ii) the contribution of a better cold start, and (iii) seed variance.

## C.1 CHEAPER REWARD REFERENCES

Anchoring the process reward to an ordered human trace costs 27.6 person-hours (Appendix B.5). Three cheaper references can occupy the same slot in the reward. A hand-designed tool-use bonus pays a fixed amount for every crop that is not a near-duplicate of an earlier one, and needs neither annotation nor a judge call. A final evidence box scores a rollout by how much of the box the annotator settled on its crops covered. A teacher-generated evidence path asks a strong vision-language model, shown the ground-truth answer, to write the region sequence it would have inspected, and judge reads that path exactly as it reads a human one.

Table 8 holds training identical and varies only the reference. Every substitute improves on the outcome-only reward, so part of the gain follows from anchoring the reward to an evidence specification of any kind. The substitutes then separate by how much of the search each one describes. The tool-use bonus, which describes none of it, recovers 1.74 points. The final box, which names where the evidence is, recovers 4.63 points. The teacher path, which names an ordering as well, reaches 39.85 with no human labour at all. The human trace adds a further 4.41 points over the best substitute.

What separates the human trace from the teacher path is where the ordering comes from. A path written backwards from a known answer contains no rejected hypotheses, so a rollout that inspects two plausible regions before converging is scored against a reference that went straight to the target. A human reference contains the same dead ends, and the judge can therefore distinguish productive exploration from redundant revisiting.

## C.2 COLD-START DEPENDENCE

Every backbone in Table 1 loses accuracy after supervised fine-tuning, which leaves open whether HaPRL improves visual search training or repairs that loss. We separate the two with a second cold start that does not degrade the backbone. Recipe B keeps the unfiltered 7,267-trajectory corpus, masks the loss on turns beyond the six-round cap instead of discarding those trajectories, and stops at two epochs rather than five. On Qwen3-VL-4B it lands at 39.84, within 0.23 of the 40.07 the backbone reaches before any agent training.

Table 8: Reward reference and what it specifies about the search. All arms train Qwen3-VL-4B for three epochs on the same 1,104 prompts and differ only in what the process reward is anchored to. Avg. is the macro average over the six evaluation splits. The best is bold.
<table><tr><td>Reward reference</td><td>Specifies</td><td>Judge</td><td>Avg.</td><td>∆ vs. HaPRL</td></tr><tr><td>None (Outcome-RL)</td><td></td><td></td><td>33.98</td><td>-10.28</td></tr><tr><td>Hand-designed tool-use bonus</td><td>crop novelty</td><td></td><td>35.72</td><td>-8.54</td></tr><tr><td>Rubric only, no reference</td><td>rubric dimensions</td><td>√</td><td>36.41</td><td>-7.85</td></tr><tr><td>Final evidence box, coverage reward</td><td>location</td><td></td><td>38.61</td><td>-5.65</td></tr><tr><td>Teacher-generated evidence path</td><td>location, ordering</td><td>√</td><td>39.85</td><td>-4.41</td></tr><tr><td>Human ordered trace (HaPRL)</td><td>location, ordering, dead ends</td><td>√</td><td>44.26</td><td></td></tr></table>

Table 9: Cold-start recipe crossed with reward condition on Qwen3-VL-4B. Recipe A is the cold start used throughout the main text; Recipe B keeps the unfiltered corpus, masks the loss beyond the six-round cap, and stops at two epochs. The backbone scores 40.07 before any agent training. Entries are macro averages over the six evaluation splits.
<table><tr><td>Cold-start recipe</td><td>SFT</td><td></td><td></td><td>+Outcome-RL +HaPRL ∆ (HaPRL − Outcome-RL)</td></tr><tr><td>A: filtered corpus, five epochs (ours)</td><td>31.36</td><td>33.98</td><td>44.26</td><td>+10.28</td></tr><tr><td>B: full corpus, over-turn masking, two epochs 39.84</td><td></td><td>43.15</td><td>49.02</td><td>+5.87</td></tr></table>

Table 9 runs both reward conditions from each cold start. The advantage of HaPRL over the outcomeonly reward is 10.28 points from Recipe A and 5.87 points from Recipe B. Recovery from a weak initialization therefore accounts for 4.41 of the 10.28 points in the main table, and process supervision for the remaining 5.87, measured from an initialization with nothing to recover. The two effects add rather than overlap: Recipe B with HaPRL reaches 49.02, the highest score in this study and 4.76 points above the best Recipe A result.

## C.3 TWO-STAGE SEED VARIANCE

Table 4 reports one run per schedule, in which an outcome-only stage 2 adds 6.13 points after a process-trained stage 1 and 0.91 points after an outcome-only one. Table 10 repeats both schedules three times.

Across seeds the outcome-only path gains 0.88 ± 0.06 points in stage 2 and the process-initialized path gains 5.83 ± 0.26, a ratio of 6.6×. The largest spread in any row is 0.57 points, an order of magnitude below the 4.95-point difference between the two gains. Both schedules see the same 600 prompts under the same reward in stage 2 and differ only in their stage-1 initialization, and the 3.94-point gap they carry into stage 2 widens to 8.89 by the end of it. The three seeds share the prompt ordering and the 400/600 split, so these intervals cover optimization and rollout stochasticity rather than data partitioning.

## D QUALITATIVE CASES

The tables in Section 4 report what changed; the six cases below show what the change looks like inside a rollout. Each figure holds the question fixed and renders the condensed trajectory of all three trained conditions from the same backbone: the cold start, Outcome-RL, and HaPRL. Every round is shown with the reasoning that motivated it, the emitted crop, and the observation it returned, so the route to the answer can be read directly rather than inferred from a score.

These are six hand-picked examples chosen to span the task profiles of Section 3.2 (text recognition, spatial relation, and attribute recognition) and all three evaluation sources. They illustrate mechanisms already measured in aggregate and are not themselves evidence of frequency. Table 11 summarizes them.

Table 10: Three seeds of the two-stage schedule of Table 4 on Qwen3-VL-4B. Seed 1 is the run reported in the main text. Entries are macro averages over the six evaluation splits; the last column is the stage-2 gain of each row over its own stage-1 checkpoint. Mean ± SD over three seeds.
<table><tr><td>Training schedule</td><td>Seed 1</td><td>Seed 2</td><td>Seed 3</td><td> $\mathrm { M e a n } \pm \mathrm { S D }$ </td><td>Stage-2 gain</td></tr><tr><td colspan="6">Stage 1 only (400 prompts)</td></tr><tr><td>Outcome-RL</td><td>32.63</td><td>32.05</td><td>32.55</td><td> $3 2 . 4 1 \pm 0 . 3 1$ </td><td></td></tr><tr><td>HaPRL</td><td>36.63</td><td>35.94</td><td>36.48</td><td> $3 6 . 3 5 \pm 0 . 3 6$ </td><td></td></tr><tr><td colspan="6">Stage 1 → outcome-only stage 2 (600 prompts)</td></tr><tr><td>Outcome-RL → Outcome-RL</td><td>33.54</td><td>32.86</td><td>33.47</td><td> $3 3 . 2 9 \pm 0 . 3 7$ </td><td> $+ 0 . 8 8 \pm 0 . 0 6$ </td></tr><tr><td> $\mathrm { H a P R L } \to \mathrm { O u t c o m e { \mathrm { - } } R L }$ </td><td>42.76</td><td>41.62</td><td>42.16</td><td> ${ \bf 4 2 . 1 8 \pm 0 . 5 7 }$ </td><td> $+ 5 . 8 3 \pm 0 . 2 6$ </td></tr></table>

Table 11: The six qualitative cases. Rounds and crops are counted from the rendered trajectories in Figures 4–9. In every case HaPRL reaches the correct answer with the shortest evidence path of the three conditions.
<table><tr><td rowspan="2">Fig.</td><td rowspan="2">Task profile</td><td rowspan="2">Source</td><td colspan="3">Rounds / crops and verdict</td></tr><tr><td>+SFT</td><td>+Outcome-RL</td><td>+HaPRL</td></tr><tr><td>4</td><td>OCR / text</td><td>HR-Bench 4K-500</td><td>3/2×</td><td>615√</td><td>4/3√</td></tr><tr><td>5</td><td>OCR / text</td><td>HR-Bench 8K-39</td><td>4/3×</td><td>7/6×</td><td>3/2√</td></tr><tr><td>6</td><td>OCR / text</td><td>VisualProbe Easy-128</td><td>4/3×</td><td>7/6×</td><td>2/1√</td></tr><tr><td>7</td><td>OCR / relation</td><td>VisualProbe Hard-54</td><td>7/6×</td><td>615√</td><td>2/1√</td></tr><tr><td>8</td><td>Attribute</td><td>VisualProbe Medium-151</td><td>7/6×</td><td>7/6×</td><td>3/2√</td></tr><tr><td>9</td><td>Attribute</td><td>V* attributes-58</td><td>4/3×</td><td>4/3×</td><td>2/1√</td></tr><tr><td colspan="3">Mean rounds / crops</td><td>4.8 / 3.8</td><td>6.2 / 5.2</td><td>2.7 / 1.7</td></tr><tr><td colspan="3">Correct</td><td>0/6</td><td>2/6</td><td>6/6</td></tr></table>

What the outcome-only rollouts do with their extra rounds. Outcome-RL uses the most rounds and the most crops of the three conditions and is correct twice. Both successes are the kind of rollout the argument of Section 1 predicts an outcome reward cannot penalize. In Figure 4 it overshoots the target twice, crops water and then rock, re-localizes, and arrives at the right word on the fifth crop. In Figure 7 it crops a nearly identical region twice in succession before confirming an answer it had already read. Neither trajectory is efficient and both receive full credit, because the only thing the reward can see is the string at the end. The four failures show where that indifference leads. In Figures 5 and 8 the policy repeats the same bottom-region or shelf crop across three consecutive rounds, exhausts its six-round budget, and emits no valid answer. In Figure 6 it chases several date-like lines and returns the truncated “March 3” instead of “March, 1935”. In Figure 9 it zooms to the wrong flagpole twice and reports the colors of a flag that is not the queried one, which the cold start does as well: repetition and distractor-chasing are exactly the behaviors a rubric can name and an outcome reward cannot.

What the process-supervised rollouts do instead. HaPRL answers all six correctly using 2.7 rounds and 1.7 crops on average, fewer than either control. The pattern is consistent across profiles. In Figure 6 it crops the article header once and reads the date. In Figure 7 it crops the joint BEER/PARK sign region once, which makes the spatial relation legible in a single view rather than requiring the two separate crops the other conditions attempt. In Figure 8 it locates the shelf containing the queried word and then tightens onto it while keeping the word visible, and in Figure 9 it crops the roadside flag directly instead of the salient flagpole above it. This is what the process-quality numbers of Table 2 measure: not that the answer is right, but that the view which justifies it was actually obtained, and obtained without spending rounds that reveal nothing new.

![](images/3b047c591a1c5146532d3b109be2c881eab33063c0bbaa9b9964c7ccbcea3662.jpg)  
Figure 4: Fine-grained OCR through visual search (HR-Bench 4K-500). The cold start zooms twice, misreads the word on the wing as “SUNNY”, and answers the wrong option. Outcome-RL needs five crops, two of which land on water and rock, before reading the word correctly. HaPRL re-centers once on the upper-middle region, targets the wing text, and answers in four rounds.

![](images/00935e31c4820bd041746de8e50eaa3f6f7f57ddec320410969a5f7f7faa0649.jpg)  
Figure 5: Fine-grained device-brand OCR (HR-Bench 8K-39). The cold start over-zooms until the text is illegible and guesses “Sony”. Outcome-RL repeats the same bottom-region crop three times, abandons it, jumps to an unrelated device, and exhausts its round budget without a valid answer. HaPRL covers the left-side device, recognizes the vertical INSIGNIA text, and answers in three rounds.

![](images/797f4b2166f27f4916f9a931202956bb2722e52be4087371d15398467ed35751.jpg)  
Figure 6: Historical-date OCR (VisualProbe Easy-128). The cold start zooms into irrelevant publication text and reports a date that is not in the image. Outcome-RL chases several date-like lines across six crops and returns the truncated “March 3”. HaPRL crops the article’s displayed header directly and reads “March, 1935” with one crop.

![](images/a4c3e6c7bb152dc845629312074ec7449f674f378d40ce5c717f2e87a6c7030d.jpg)

![](images/8b353f5b88bab30147393ecdc3c365fa5bfb383ee296cc4e2c0ce9ab9c0a53fc.jpg)  
What is the text to the right of the text BEER?

![](images/a511adb56dfe1cbf0dd0086ee868b89912c9874ba770aa8bda909515f3b00c3f.jpg)

T1: Searches for the BEER sign.   
<think> The user is asking for the text to the right of the word "BEER". I see a sign on a building in the lower-left portion of the image. It seems to have some text on it. I will zoom in on that area to read the text.....</think><grounding> {"bbox 2d": [0.1, 0.7, 0.3, 0.9], "source": "original image"}</grounding>   
T2: Moves toward the landmark text   
<think> In the previous step, I zoomed in on a sign in the lower-left corner of the image. However, the text on the sign is not clear enough to read. I need to zoom in further to identify the word "BEER" and the text next to it. Looking at observation 1......</think> <grounding>{"bbox 2d": [0.2, 0.2, 0.8, 0.8], "source": "observation 1"}</grounding>   
T3: Repeats the same text region.   
<think> I have zoomed in twice, but the text is still not clear. It has some text on it, but it's hard to read from the original image. The sign is located in the region [0.4, 0.6, 0.6, 0.8] of the original image. I will use the grounding action to crop this region. </think><grounding> {"bbox 2d": [0.4, 0.6, 0.6, 0.8], "source": "original image"}</grounding   
T4: Repeats the same text region again,   
<think> I have tried to zoom in on different parts of the image, but the text is still not clear. It has some text on it, but it's hard to read from the original image. ....I will use the grounding action to crop this region. </think><grounding>{"bbox\_2d": [0.4, 0.6, 0.6, 0.8], "source": "original\_image"}</grounding>   
T5: Continues without committing.   
<think> I have tried to zoom in on different parts of the image, but the text is still not clear. It has some text on it, but it's hard to read from the original image. ....I will use the grounding action to crop this region. </think><grounding>{"bbox\_2d": [0.4, 0.6, 0.6, 0.8], "source": "original\_image"}</grounding>   
T6: Uses the final crop on unchanged evidence   
<think> I have tried to zoom in on different parts of the image, but the text is still not clear. It has some text on it. but it's hard to read from the original image. ....I will use the grounding action to crop this region. </think><grounding>{"bbox\_2d": [0.4, 0.6, 0.6, 0.8], "source": "original\_image"}</grounding>   
T7: No valid final answer.   
<think> I have tried to zoom in on different parts of the image, but the text is still not clear. It has some text on it, but it's hard to read from the original image. .....I will use the grounding action to crop this region. </think><grounding>{"bbox\_ 2d": [0.4, 0.6, 0.6, 0.8], "source": "original\_image"}</grounding>

![](images/f0be2d5e6c26ad54f21a6a2881b61d3cc55772f48396d73daec248d405f621e4.jpg)

![](images/1e18c6ca5b2949e43ff9791cf28cbbdeae9e816ab1ab67e42adbaacf1c32d470.jpg)  
Figure 7: Relational OCR in a crowded scene (VisualProbe Hard-54). The question asks for the text to the right of BEER, so the two words must be legible in one view. The cold start re-crops the same unreadable region four times and returns no valid answer. Outcome-RL reaches the right answer but repeats a nearly identical crop and spends a further round confirming what it had already read. HaPRL crops the joint sign region once, which makes the relation readable immediately.

![](images/4be820f4ea9cd222e000dab6320c3ee57325aef8e5298ccb75c4f87ff58ba69c.jpg)  
Figure 8: Text-color recognition on a dense shelf (VisualProbe Medium-151). Both controls lose the target among repeated shelf crops and end without a usable answer, the cold start concluding the color cannot be determined. HaPRL first locates the shelf holding the queried word, then tightens while keeping the word visible, and reads its color in three rounds.

Case Study: Small-object Color Disambiguation

![](images/59d56573afaa91c4ca53d5c7ec127f7bc5716fe8a1251bf4443899619bf9675f.jpg)  
Figure 9: Small-object color disambiguation (V<sup>∗</sup> direct attributes-58). Both controls fix on a salient flagpole rather than the queried flag and report the colors of the wrong object, arriving at the same incorrect option by different routes. HaPRL crops the roadside flag directly and answers in two rounds. A wrong target reached fluently is the failure the rubric’s target-semantics dimension is meant to price.