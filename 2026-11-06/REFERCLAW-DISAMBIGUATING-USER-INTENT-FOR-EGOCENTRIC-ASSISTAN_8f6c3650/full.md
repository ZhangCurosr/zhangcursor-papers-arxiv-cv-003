# REFERCLAW: DISAMBIGUATING USER INTENT FOR EGOCENTRIC ASSISTANCE

Hangyu Qin<sup>1</sup>, Junbin Xiao<sup>2∗</sup>, Angela Yao<sup>1</sup>

<sup>1</sup>National University of Singapore <sup>2</sup>University of Science and Technology of China

## ABSTRACT

Users of egocentric AI assistants often ask short, ambiguous questions whose intended referents cannot be uniquely resolved from video and dialogue history. For example, “Where did I leave it?” may refer to several previously handled objects. Existing VideoQA systems typically commit to a single interpretation, risking plausible but intention-misaligned answers. We introduce ReferClaw, an agentic framework that formulates egocentric streaming QA as an interactive process: the assistant asks clarifying questions to resolve user intent before answering. Three cooperating agents identify candidate interpretations, request clarification, and generate answers grounded in the resolved intent and visual evidence. A bounded, cost-aware policy balances expected information gain against user effort while limiting clarification turns and question complexity. ReferClaw learns from interaction through intent-specific skill updates and process-reward-guided policy optimization with agent-level credit assignment. Evaluations on augmented egocentric QA benchmarks demonstrate improved answer accuracy and joint answerand-grounding performance over existing baselines, alongside zero-shot transfer across datasets. These findings support interactive clarification as a practical approach to resolving online user intent in egocentric streaming assistance.

## 1 INTRODUCTION

Egocentric cameras paired with multimodal AI assistants are becoming natural interfaces to the physical world, enabling continuous visual memory, navigation, and daily assistance. Recent benchmarks have advanced episodic memory (Barmann & Waibel, 2022; Yang et al., 2025b), long-range¨ understanding (Mangalam et al., 2023), and situated assistance (Xiao et al., 2025; Sun et al., 2025; Qin et al., 2026). Their standard QA protocols, however, evaluate answers against fixed questions and observation histories, rather than how an assistant acquires missing intent information. In practice, user requests are brief and ego-deictic, and the available context may support several interpretations. For “Did she put it back?” (Fig. 1), different persons may perform the same action at different times. Even when both encounters are visible, the intended one may remain unclear. An answer can therefore be compatible with the video yet irrelevant to what the user meant.

Despite this progress, the challenge remains largely underexplored. Existing approaches (Peng et al., 2025; Choi et al., 2026; Qin et al., 2026) exploit eye gaze and hand gesture as visual or textual prompts for disambiguating user intent. While effective, the gains remain limited for the complexity of real-world interactions. Recent VLMs instead explore proactive request for clarification (Jian et al., 2025; Cao et al., 2026; Yang et al., 2026), showing that models can ask rather than guess based on the utterance and a single observation. This may suffice for a static image, but may not for dynamic video stream, where different encounters can support different interpretations of the same question (Fig. 1). We therefore study how explicit spatio-temporal interpretations can guide what to ask, how to interpret the response, and which evidence to use for answering.

We formulate intent disambiguation as bounded information acquisition over grounded candidate interpretations. We propose ReferClaw which coordinates three agents around these candidates. The Intention Grounding Agent (IGA) constructs the candidate intents by grounding the question against the observed video, pairing each candidate with its supporting visual evidence, and scores the candidates by evidential support; the Cost-Aware Clarification Agent (CCA) generates and chooses the best clarification question by balancing expected information gain against respondent effort; and the Evidence-Fused Answering Agent (EFA) answers the original question from the resolved intent and its evidence. Their shared state allows clarification responses to refine the candidates or trigger re-grounding. A deterministic controller bounds clarification turns and question complexity, and decides when to ask or answer. ReferClaw also learns from past interactions to reduce recur ring disambiguation errors. Following MetaClaw’s two-timescale design (Xia et al., 2026), verified feedback produces intent-specific skills that take effect immediately and process rewards that guide updates to the responsible agent. The skills encode which contrasts distinguish competing interpretations, while agent-level updates target failures in grounding, clarification, or answering. In our evaluation protocol, the annotated target intent, supporting evidence, and answer correctnes become available only after each prediction, so updates affect subsequent interactions.

![](images/132d3dcf93563e8037adb140208fda3f5b1787060e45b3ad0fe43e8e2477eb94.jpg)  
Figure 1: Different encounters can support different interpretations of the same question. Refer-Claw asks for clarification to identify the intended encounter and select evidence for answering.

To facilitate the study, we augment two recent egocentric streaming QA benchmarks BuddyVQA (Qin et al., 2026) and EgoBlind (Xiao et al., 2025) with ambiguity labels, target intents, and spatiotemporal visual evidence. Experiments separate the benefits of clarification from learning, test ambiguity across repeated encounters, and assess transfer to independent image-based clarification benchmarks ClearVQA (Jian et al., 2025) and ContextClarify (Cao et al., 2026). We additionally evaluate well-specified questions and human respondents to examine unnecessary interruptions and the applicability of simulated interactions. Our contributions are threefold:

• We formulate intent disambiguation in egocentric QA assistance as bounded information acquisition over spatio-temporally grounded interpretations.

• We develop ReferClaw, which coordinates grounding, clarification, and answering agents through shared candidates, and learns from feedback through intent-specific skills and agent-level updates.

• We augment two ego-streaming QA benchmarks with intent and grounding annotations, and evaluate clarification, continual adaptation, and transfer under controlled interaction protocols.

## 2 RELATED WORK

Clarification and User Intent Disambiguation. Clarification establishes common ground by acquiring missing information and resolving alternative interpretations (Clark & Brennan, 1991; Saeidi et al., 2018; Xu et al., 2019; Rao & Daume III, 2018; Min et al., 2020; Lee et al., 2023). Cooperative´ visual dialog agents (Das et al., 2017) and GuessWhat (de Vries et al., 2017) extend goal-directed question asking to images, and later methods select questions through uncertainty (Abbasnejad et al., 2019) or expected information gain (White et al., 2021). Recent work localizes plausible image regions for ambiguous questions (Chen et al., 2025a), teaches VLMs to request clarification on ClearVQA (Jian et al., 2025), and learns ask-or-answer policies on ContextClarify (Cao et al., 2026). Plug-and-Play Clarifier (Yang et al., 2026) addresses linguistic ambiguity, capture quality, and pointing gestures in egocentric interaction. Existing clarification benchmarks are largely image-based, motivating our augmentation of BuddyVQA and EgoBlind with intent annotations for video-based disambiguation (Sec. 4.1). Methodologically, prior works clarify from the utterance and a single observation; ReferClaw instead focuses on competing interpretations grounded across video history, using shared candidates to connect clarification with evidence selection under bounded respondent effort.

Egocentric and Grounded Video QA. Ego4D (Grauman et al., 2022) evaluates episodic memory, while EgoSchema (Mangalam et al., 2023) tests long-range egocentric video understanding. EgoBlind (Xiao et al., 2025) emphasizes situated assistance, and BuddyVQA (Qin et al., 2026) evaluates ego-deictic and history-dependent questions over a causal stream. GroundVQA (Di & Xie, 2024) and NExT-GQA (Xiao et al., 2024) couple answering with temporal localization, GeLM (Chen et al., 2025b) develops grounded multi-hop reasoning, $\mathrm { D } ^ { \mathrm { \tiny { 2 } } } \mathrm { V L M }$ (Zeng et al., 2025) factorizes grounding and answering, and recent systems expose explicit temporal or spatio-temporal evidence for verifiable answers (Meng et al., 2026; Seo et al., 2026; Huang et al., 2026). Related systems support interactive video questioning (Wang et al., 2025a), pixel-level conversational grounding (Munasinghe et al., 2025), and streaming evidence retrieval (Di et al., 2025). These methods localize evidence under a single interpretation of the query, so stronger retrieval may confidently ground the wrong referent when several intentions remain plausible. ReferClaw represents the alternatives explicitly and acquires clarifications on which interpretation to guide evidence selection and answering.

Multi-Agent Collaboration. Multi-agent systems coordinate specialized roles through structured communication (Hong et al., 2024; Chen et al., 2024b; Liu et al., 2026; Dang et al., 2025). Interaction-based adaptation uses self-feedback, verbal reflection, and reusable skills or routines (Madaan et al., 2023; Shinn et al., 2023; Zhao et al., 2024; Wang et al., 2024; 2025c; Tu et al., 2026). MetaClaw combines fast skill adaptation with slow policy optimization (Xia et al., 2026), while process supervision rewards intermediate steps (Lightman et al., 2024; Zhang et al., 2025b). ReferClaw adapts these mechanisms to intent disambiguation: skills encode ambiguity triggers and decisive contrasts, while process rewards guide updates to the grounding, clarification, and answering agents.

## 3 METHOD

Clarifying an ambiguous question requires three decisions that ordinary VideoQA does not: whether the question admits more than one visually supported interpretation, what to ask so that the user can resolve it at little effort, and how to answer from the interpretation the user selects. ReferClaw assigns each decision to a dedicated agent operating on shared grounded candidate interpretations, coordinated by a fixed controller that bounds the user’s burden (Fig. 2), and learns from verified interaction feedback so that disambiguation mistakes are not repeated.

## 3.1 PROBLEM SETUP

For a question q asked at video time $s _ { q } .$ , the assistant observes the causal prefix $V _ { \leq s _ { q } } = \{ v _ { s } \} _ { s = } ^ { s _ { q } }$ and episode dialogue D. The prefix stays fixed during clarification. At turn $t , h _ { t } = \{ ( c _ { \ell } , y _ { \ell } ) \} _ { \ell = 0 } ^ { t - 1 }$ records clarification questions c and responses $y .$ Intent is ambiguous when multiple visually supported interpretations remain plausible given $( V _ { \le s _ { q } } , q , D , h _ { t } )$

A candidate $\boldsymbol { z } _ { i } = ( \widetilde { \boldsymbol { q } } _ { i } , \boldsymbol { e } _ { i } )$ pairs a question rewrite $\widetilde { q } _ { i }$ with evidence $e _ { i } = ( \tau _ { i } , r _ { i } )$ : an interval $\tau _ { i } \subseteq$ $[ 1 , s _ { q } ]$ and referent box track $r _ { i }$ (empty for non-localizable information). For “Is it still on $? ^ { \ast }$ in Fig. $^ { 2 , }$ rewrites may specify the “stove” used at time 1:05 or the “faucet” at 2:45. The assistant maintains $\mathcal { Z } _ { t } = \{ z _ { i } \} _ { i = 1 } ^ { \bar { n } _ { t } } , n _ { t } \mathop { \leq } N _ { \operatorname* { m a x } } ,$ and belief $p _ { t }$ over these candidates plus an uncovered interpretation $z _ { \mathcal { O } }$ . It asks $c _ { t }$ or terminates with an ANSWER $( a , \widetilde { q } , e , \gamma )$ , where γ is the belief mass supporting the answer, a CONDITIONAL output with at most two interpretations and answers, or ABSTAIN.

For policy π and respondent type $x \in \{ \mathrm { H } , \mathrm { L } \}$ (human or LLM), we seek

$$
\operatorname* { m a x } _ { \pi } \mathbb { E } \left[ U _ { \mathrm { a n s } } + \lambda _ { g } U _ { \mathrm { g r d } } - \sum _ { t = 0 } ^ { N _ { \mathrm { a s k } } - 1 } C _ { x } ( c _ { t } ) \right] \quad \mathrm { s . t . } \quad N _ { \mathrm { a s k } } \leq T _ { \mathrm { m a x } } , \vert c _ { t } \vert \leq L _ { \mathrm { m a x } } , m ( c _ { t } ) \leq M _ { \mathrm { m a x } } ,\tag{1}
$$

Here $U _ { \mathrm { a n s } } ( a , z ^ { \star } )$ is binary judged correctness for target intent $z ^ { \star }$ , and $\begin{array} { r l } { U _ { \mathrm { g r d } } ( e , e ^ { \star } ) } & { { } = } \end{array}$ min(tIoU, sIoU) measures overlap with target evidence $e ^ { \star }$ , using temporal IoU alone when no target box exists. The weight $\lambda _ { g }$ balances grounding; $C _ { x } ( c _ { t } ) \in [ 0 , 1 ]$ measures respondent cost.

![](images/86e8bde51a683c26e21bcad935984555e3dbe00c38dfeed22a2eed3b9b177733.jpg)  
Figure 2: Overview of ReferClaw. The system coordinates three agents (IGA, CCA, EFA) to respectively ground candidate interpretations, requests clarification when useful, and verifies answers against the resolved intent and evidence. A fixed controller bounds interaction. Top: verified feedback updates prompt-level skills immediately and agent-specific LoRA adapters during idle periods.

CONDITIONAL and ABSTAIN receive zero utility. The limits bound the number of questions $N _ { \mathrm { a s k } } ,$ question length |c<sub>t</sub>| in tokens, and displayed options $m ( c _ { t } )$ , preventing endless clarification and one-shot “super-questions” that overwhelm the user with many alternatives in a single turn.

## 3.2 COOPERATING AGENTS AND INTERACTION CONTROLLER

The three agents share $( \mathcal { Z } _ { t } , p _ { t } )$ . Each uses a role-specific prompt and LoRA adapter on one frozen multimodal backbone; a fixed controller sequences their actions and enforces the interaction budget.

Intention Grounding Agent. IGA predicts an ambiguity score $d _ { t } \in [ 0 , 1 ]$ and responsible span, then grounds before rewriting. Temporal retrieval, detection, and tracking produce up to eight evidence cards containing an interval–track pair, grounding score $\rho _ { i } ,$ and visible-content description (Appendix B.1). Each card yields a rewrite $\widetilde { q _ { i } } .$ , support score $s _ { i } \in [ 0 , 1 ]$ , and provisional answer $\bar { a } _ { i }$ After merging duplicate entity–event cards and retaining at most $N _ { \mathrm { m a x } }$ candidates, IGA initializes

$$
p _ { t } ( z _ { i } ) = ( 1 - \epsilon _ { t } ) \frac { \exp \bigl ( s _ { i } / \kappa \bigr ) } { \sum _ { j = 1 } ^ { n _ { t } } \exp \bigl ( s _ { j } / \kappa \bigr ) } , \qquad p _ { t } ( z _ { \mathcal { Q } } ) = \epsilon _ { t } = \epsilon _ { 0 } \bigl ( 1 - \operatorname* { m a x } _ { i } \rho _ { i } \bigr ) ,\tag{2}
$$

where κ is a validation-fitted temperature and $\epsilon _ { t }$ reserves uncovered-intent mass when grounding support is weak. Clarification requires high ambiguity, no dominant candidate, and disagreement among provisional answers:

$$
\begin{array} { r } { g _ { t } = \mathbb { I } [ d _ { t } \geq \tau _ { d } ] \mathbb { I } [ 1 - \operatorname* { m a x } _ { i } p _ { t } ( z _ { i } ) \geq \tau _ { u } ] \mathbb { I } [ \operatorname { D i s } ( \overline { { \mathcal { A } } } _ { t } ) \geq \tau _ { a } ] , } \end{array}\tag{3}
$$

Here Dis is the belief-weighted fraction of candidate pairs with different provisional answers $\overline { { A } } _ { t } =$ {a¯<sub>i</sub>}. When all plausible candidates imply the same answer, clarification is unnecessary even if the intended referent remains uncertain.

Cost-Aware Clarification Agent. CCA proposes at most six questions $\mathcal { C } _ { t }$ , each contrasting candidates by identity, occurrence, location, or action. Each declares an option-based partition with an “other/none” option for $z _ { \mathcal { O } }$ . The response model $p _ { x } ( y \mid c , z _ { i } , h _ { t } )$ predicts the respondent’s answer class: a partition cell, other/none, uncertain, or invalid/missing. For LLM respondents, smoothed frequencies from eight backbone samples estimate this likelihood, conditioning on candidate $z _ { i }$ rather than on the annotated target intent, which only the respondent supplying actual responses (simulator or human) receives.

An actual response $y _ { t }$ updates the belief by Bayes’ rule. Hypothetical responses define expected information gain (IG), measured as the reduction in Shannon entropy H (bits):

$$
\begin{array} { r } { p _ { t + 1 } ( z _ { i } ) \propto p _ { t } ( z _ { i } ) p _ { x } ( y _ { t } \mid c _ { t } , z _ { i } , h _ { t } ) , \quad \mathrm { I G } _ { t } ( c ) = H ( p _ { t } ) - \sum _ { y } p _ { x } ( y \mid c , h _ { t } ) H \big ( p _ { t + 1 } ( \cdot \mid c , y ) \big ) . } \end{array}\tag{4}
$$

where $\begin{array} { r } { p _ { x } ( y \mid c , h _ { t } ) = \sum _ { i } p _ { t } ( z _ { i } ) p _ { x } ( y \mid c , z _ { i } , h _ { t } ) } \end{array}$ . IG is a one-step surrogate for answer utility, used only when the gate detects answer disagreement. CCA balances it against normalized effort $\overline { { E } } _ { x }$ delay $\overline { { T } } _ { x }$ , and failure risk ${ \overline { { F } } } _ { x }$

$$
C _ { x } ( c ) = w _ { E } \overline { { E } } _ { x } ( c ) + w _ { T } \overline { { T } } _ { x } ( c ) + w _ { F } \overline { { F } } _ { x } ( c ) , \qquad c _ { t } ^ { \star } = \arg \operatorname* { m a x } _ { c \in \mathcal { C } _ { t } } u _ { t } ( c ) ,\tag{5}
$$

where $u _ { t } ( c ) = \mathrm { I G } _ { t } ( c ) - \lambda _ { C } C _ { x } ( c )$ is the net utility of a question. CCA returns $c _ { t } ^ { \star }$ together with $u _ { t } ( c _ { t } ^ { \star } )$ , which the controller uses to decide whether to ask, or no-question when the pool is empty or $u _ { t } ( c _ { t } ^ { \star } ) \leq 0$ . Human response likelihoods and costs are calibrated on the user study using question length, choice complexity (Hick, 1952; Hyman, 1953), response time, and failure rate. LLM costs use token counts, serving latency, and predicted invalid-format mass (Appendices B.2 and B.3).

Evidence-Fused Answering Agent. EFA selects $i ^ { \star } ~ = ~ \arg \operatorname* { m a x } _ { i } p _ { t } ( z _ { i } )$ and generates $a =$ $F _ { \omega } ( V [ \tau _ { i ^ { \star } } ] , r _ { i ^ { \star } } , q , \widetilde { q } _ { i ^ { \star } } , h _ { t } )$ with confidence $\gamma = p _ { t } ( z _ { i ^ { \star } } )$ . Its prompt separates observations from user responses and labels claims as video or user: clarification identifies the appliance, not whether it is on. A separate verifier prompt checks alignment with the rewrite and support for each claim in its corresponding source: video claims must be visible in the evidence card, and user claims must appear in $h _ { t } .$ . It permits one revision before passing or abstaining, without changing the interpretation or retrieving evidence (Appendices F.6 and F.7).

Interaction Controller. In priority order, the controller abstains if support remains inadequate after one re-grounding; invokes EFA if $g _ { t } = 0$ or $\operatorname* { m a x } _ { i } p _ { t } ( z _ { i } ) \geq \tau _ { p } ;$ and asks if $g _ { t } = 1 , t < T _ { \operatorname* { m a x } }$ , and net utility is positive. Otherwise, it returns CONDITIONAL when the top two candidates disagree and their belief margin is below $\tau _ { g } ,$ or invokes EFA. Responses update Eq. 4; other/none or an unsupported constraint triggers one constrained re-grounding. Uncertain or invalid responses leave belief unchanged and ban the asked partition (Appendix B.4).

## 3.3 LEARNING FROM INTERACTION

Following MetaClaw (Xia et al., 2026), our learning maintains a versioned skill library $ { \boldsymbol { S } } _ { g }$ and agent adapters $\theta _ { g }$ indexed by skill generation g. Under our prequential protocol $( \sec . 4 . 1 ) , \stackrel { \cdot } { z ^ { \star } } , \stackrel { \cdot } { e ^ { \star } }$ , and the judge’s verdict are revealed only after a prediction is logged. A trajectory whose answer is wrong, whose evidence misses $e ^ { \star }$ , or that asked unnecessarily is a verified failure. Verified failures are attributed by type: coverage and grounding failures to $\mathrm { I G A } ,$ , non-separating or unnecessary questions to CCA, and inconsistent answers to EFA. A process-review prompt handles residual cases (91% agreement with manual labels; Appendix B.5). Verified trajectories feed two loops, scheduled by whether the assistant is currently serving requests (Fig. 2, top).

Fast Loop: Intent-specific Skills. Verified failures form support set $\mathcal { D } _ { g } ^ { \mathrm { s u p } }$ , from which an evolver updates $S _ { g + 1 } = S _ { g } \cup E _ { \mathrm { s k i l l } } ( \mathcal { D } _ { q } ^ { \mathrm { s u p } } , S _ { g } )$ . Each agent-specific skill records an ambiguity trigger, candidate topology (e.g., one object encountered twice), decisive contrast, corrected action, antipattern, and two regression tests against unnecessary asking. Up to three skills, retrieved by trigger– topology embedding similarity, immediately enter the responsible agent’s prompt. For example, a CCA skill contrasts repeated encounters through the user’s visible action rather than their order; Appendix B.6 provides the worked example.

Slow Loop: Agent-attributed Process Rewards. During idle periods, a query buffer $\mathcal { D } ^ { \mathrm { q r y } }$ of 128 verified trajectories tagged with their skill generation, including successes and failures, triggers adapter updates. Each trajectory $\xi$ receives

$$
R _ { \mathrm { p r o c } } ( \xi ) = r _ { \mathrm { f i n a l } } + \alpha r _ { \mathrm { g r o u n d } } + \beta \sum _ { t } \Delta _ { t } ^ { \star } - \lambda \sum _ { t } C _ { x } ( c _ { t } ) - \mu r _ { \mathrm { i n v a l i d } } ,\tag{6}
$$

where $r _ { \mathrm { f i n a l } }$ combines correctness and verifier verdict, $r _ { \mathrm { g r o u n d } }$ combines $U _ { \mathrm { g r d } }$ and target coverage, $C _ { x }$ is realized respondent cost, and $r _ { \mathrm { i n v a l i d } }$ counts rule violations. Target progress $\begin{array} { r l } { \Delta _ { t } ^ { \star } } & { { } = } \end{array}$ log $p _ { t + 1 } \mathopen { } \mathclose \bgroup \left( z ^ { \star } \aftergroup \egroup \right) - \log p _ { t } \mathopen { } \mathclose \bgroup \left( z ^ { \star } \aftergroup \egroup \right)$ rewards belief moving toward the intended interpretation. A fixed candi date scope prevents reward from candidate removal alone (Appendix B.7).

For each failed step, the attributed agent samples G = 8 alternatives from its logged prompt, including its original skills. The evaluation simulator completes them into trajectories; a reward model predicts $\bar { R _ { \mathrm { p r o c } } }$ (held-out Spearman 0.83). GRPO (Shao et al., 2024) updates only that agent’s LoRA adapter, with a KL penalty toward frozen Base. Promotion requires improvement on the target failure type without degrading overall accuracy, unnecessary asking, or mean cost on a fixed validation probe; otherwise the update is rolled back (Appendix B.7).

## 4 EXPERIMENTS

Our experiments answer four questions. Q1: Does targeted clarification improve intent-aligned and grounded answering over direct and interactive baselines (Sec. 4.2)? Q2: Does grounding the alternatives across time help most when the competing referents are separated in time (same object seen twice / similar objects seen at different moments) rather than visible at once (Sec. 4.4)? Q3: Are the gains obtained with limited user effort (Sec. 4.4)? Q4: Does learning from interaction reduce future errors beyond simpler adaptation (Secs. 4.2 and 4.4)?

## 4.1 EXPERIMENTAL SETUP

Benchmarks. To facilitate our study, we augment two closest-aim egocentric QA datasets with intent-ambiguity labels, minimal supporting temporal intervals and spatial boxes. BuddyVQA (Qin et al., 2026), our primary benchmark, with its causal streaming protocol, ego-deictic expressions, and history-dependent question chains; and EgoBlind-Intent (Xiao et al., 2025), with timestamped assistance questions from blind and visually impaired users, used for zero-shot transfer. We also test transfer on two independent image-based clarification benchmarks, ClearVQA (Jian et al., 2025) and ContextClarify (Cao et al., 2026), treating the image as a one-frame prefix with their official metrics and respondents. More benchmark augmentation details are in Appendix C.

Protocol and Metrics. A model may access only frames observed before the question timestamp and the preceding dialogue; gold labels, resolved questions, intervals, and boxes are never inputs at prediction time; only the latter three and the judge verdict are revealed to learning methods, after a prediction is logged. Answer accuracy uses each benchmark’s multi-reference LLM-as-a-judge protocol over all test inputs, with CONDITIONAL and ABSTAIN counted as incorrect; temporal metrics are computed on grounding-eligible questions, spatial metrics and Acc@GQA (Xiao et al., 2024) (we further require correct answer with both T-IoU and S-IoU ≥ 0.5) on the box-annotated subset. Interactive evaluation additionally reports ambiguity-detection F1, ask precision/recall, clarification accuracy, unnecessary-ask and abstention rates, turns, question tokens, response time, latency, respondent cost, and the utility of Eq. 1. All numbers are means over three fixed stream orders, and all subset values of a row come from the same runs and checkpoint; the full protocol, the population of every metric, and confidence intervals are in Appendices C.4 and E.3.

Baselines and Implementation. We compare against direct MLLMs (Qwen Team, 2026), grounded and streaming VideoQA systems (Meng et al., 2026; Huang et al., 2026; Di et al., 2025; Wang et al., 2025a; Qin et al., 2026), and interactive clarification methods (Jian et al., 2025; Yang et al., 2026; Cao et al., 2026). Framework-level baselines and all ReferClaw variants use Qwen3.5-9B, under the same input protocol. To avoid conflating clarification, extra information, and online learning, we evaluate in two regimes. Frozen inference compares fixed systems, including ReferClaw (Base) and its asking-disabled control, which retains grounding, rewriting, and verification. Learningfrom interaction follows MetaClaw’s online evaluation (Xia et al., 2026): feedback follows each logged prediction and updates affect only later inputs. We compare ReferClaw (Skills) (fast skill adaptation), ReferClaw (Full) (adding agent-attributed GRPO, Eq. 6), Reflexion-style memory (Shinn et al., 2023), and MetaClaw-style adaptation (generic skills, all-adapter updates, final-answer reward). All share the initial system, feedback, stream, and update budget. Details are in Appendix D.

## 4.2 MAIN RESULTS

Frozen Inference (Q1): Tab. 1 compares answer accuracy under the streaming protocol. Frozen ReferClaw reaches 66.2%, versus 58.5% for Clarify-or-Answer. Disabling asking while retaining grounding and verification lowers accuracy to 60.5%, giving a 5.7-point matched clarification gain. Improvements also appear on deictic and history-dependent questions: Base reaches 69.0% and 66.8%, respectively, versus 58.3% and 55.2% for Clarify-or-Answer. The category breakdown and published reference results appear in Tab. 1. Learning From Interaction (Q4): Starting from Base, intent-specific skills raise accuracy to 70.1%, and Full reaches 74.3%, versus 67.4% for Reflexionstyle memory and 69.5% for MetaClaw-style adaptation. These comparisons measure adaptation under the same feedback and update budget. Sec. 4.4 further tests whether improvements also hold on an unchanged probe set.

Table 1: Ego-streaming QA accuracy (%) on Augmented BuddyVQA. †: published results under the original protocol and split (not directly comparable, Appendix C.1). All other rows are comparable.
<table><tr><td>Method</td><td>Recall</td><td>Recog.</td><td>Advis.</td><td>Scrut.</td><td>Chain</td><td>Deictic</td><td>Overall</td></tr><tr><td>Human†</td><td>92.4</td><td>92.1</td><td>91.9</td><td>92.0</td><td></td><td></td><td>92.1</td></tr><tr><td>InternVL3.5-8B† (Wang et al., 2025b)</td><td>54.1</td><td>51.3</td><td>44.7</td><td>46.1</td><td>47.2</td><td>43.1</td><td>48.8</td></tr><tr><td>Flash-VStream† (Żhang et al., 2025a)</td><td>52.2</td><td>51.9</td><td>44.6</td><td>46.7</td><td></td><td></td><td>50.8</td></tr><tr><td>ReKV (LLaVA-OV-7B)† (Di et al., 2025)</td><td>54.6</td><td>54.1</td><td>44.9</td><td>44.4</td><td>50.9</td><td>48.6</td><td>52.7</td></tr><tr><td>MyBuddy (Qwen3.5-9B)† (Qin et al., 2026)</td><td>59.9</td><td>60.8</td><td>55.8</td><td>54.2</td><td>57.3</td><td>58.9</td><td>60.7</td></tr><tr><td>Frozen inference (Qwen3.5-9B)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3.5-9B (Qwen Team, 2026)</td><td>52.8</td><td>55.9</td><td>46.6</td><td>49.1</td><td>46.8</td><td>47.9</td><td>51.4</td></tr><tr><td>MyBuddy (our reproduction) (Qin et al., 2026)</td><td>62.0</td><td>62.3</td><td>53.4</td><td>57.9</td><td>59.8</td><td>57.9</td><td>59.2</td></tr><tr><td>ClearVQA (Jian et al., 2025)</td><td>57.7</td><td>59.8</td><td>50.9</td><td>53.2</td><td>51.7</td><td>54.0</td><td>55.7</td></tr><tr><td>Plug-and-Play Clarifier (Yang et al., 2026)</td><td>59.1</td><td>61.3</td><td>52.9</td><td>55.2</td><td>53.9</td><td>57.1</td><td>57.4</td></tr><tr><td>Clarify-or-Answer (Cao et al., 2026)</td><td>60.4</td><td>62.1</td><td>53.8</td><td>56.5</td><td>55.2</td><td>58.3</td><td>58.5</td></tr><tr><td>ReferClaw (Base, asking disabled)</td><td> $6 2 . 0 ^ { \uparrow 1 . 6 }$ </td><td> $6 2 . 8 ^ { \uparrow 0 . 7 }$ </td><td> $5 6 . 3 ^ { \uparrow 2 . 5 }$ </td><td> $6 0 . 2 ^ { \uparrow 3 . 7 }$ </td><td> $5 8 . 6 ^ { \uparrow 3 . 4 }$ </td><td> $5 8 . 2 ^ { \downarrow 0 . 1 }$ </td><td> $6 0 . 5 ^ { \uparrow 2 . 0 }$ </td></tr><tr><td>ReferClaw (Base)</td><td> ${ \bf 6 7 . 7 ^ { \dagger 5 . 7 } }$ </td><td> ${ \bf 6 8 . 6 ^ { \dag 5 . 8 } }$ </td><td> ${ \bf 6 2 . 3 ^ { \mathrm { \uparrow 6 . 0 } } }$ </td><td> ${ \bf 6 5 . 4 } ^ { \uparrow 5 . 2 }$ </td><td> ${ \bf 6 6 . 8 ^ { \mathrm { ( 8 . 2 } } }$ </td><td> ${ \bf 6 9 . 0 ^ { \mathrm { \uparrow 1 0 . 8 } } }$ </td><td> ${ \bf 6 6 . 2 ^ { \dag 5 . 7 } }$ </td></tr><tr><td>Learning from interaction (same stream, feedback, and update budget) Base + Reflexion-style memory (Shinn et al., 2023)</td><td>68.9</td><td>69.7</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base + MetaClaw-style adaptation (Xia et al., 2026)</td><td>71.1</td><td>71.7</td><td>63.5 65.8</td><td>66.7 68.7</td><td>68.0 70.2</td><td>70.1 72.3</td><td>67.4</td></tr><tr><td>ReferClaw (Skills)</td><td> $7 1 . 6 ^ { \uparrow 0 . 5 }$ </td><td> $7 2 . 2 ^ { \uparrow 0 . 5 }$ </td><td></td><td></td><td></td><td></td><td>69.5</td></tr><tr><td>ReferClaw (Full)</td><td> $7 5 . 7 ^ { \uparrow 4 . 1 }$ </td><td> $7 6 . 4 ^ { \uparrow 4 . 2 }$ </td><td> $6 6 . 4 ^ { \uparrow 0 . 6 }$   $7 1 . 0 ^ { \uparrow 4 . 6 }$ </td><td> $6 9 . 4 ^ { \uparrow 0 . 7 }$   $7 3 . 3 ^ { \uparrow 3 . 9 }$ </td><td> $7 1 . 0 ^ { \uparrow 0 . 8 }$   $7 5 . 4 ^ { \uparrow 4 . 4 }$ </td><td> $7 3 . 1 ^ { \uparrow 0 . 8 }$   $7 7 . 5 ^ { \uparrow 4 . 4 }$ </td><td> $7 0 . 1 ^ { \uparrow 0 . 6 }$   $7 4 . 3 ^ { \uparrow 4 . 2 }$ </td></tr></table>

![](images/104b64505603fc6e07e4185cdf0e08d6558e5f98195c06078b980cd1a0864da1.jpg)

![](images/b1004ede8b15515a2fba0cd91054a8d3b3ea7f286175fb738a3d2ee73abd6069.jpg)  
Figure 3: Clarification vs. No Clarification (ReferClaw uses frozen Base; †: published). Solid bars: accuracy; hatched bars: F1. Rows under each panel: accuracy gain over the preceding bar and interaction statistics.

Generalization: Fig. 3 addresses two concerns: whether clarification helps beyond our own annotations, and whether it disrupts questions whose intent is already clear. F1 measures ambiguity detection on ClearVQA and the ask/answer decision on ContextClarify; Turns counts questions per input, U-

Table 2: Results on EgoBlind-Intent (%).
<table><tr><td>Method</td><td>T-mIoU</td><td>S-mIoU</td><td>Ans.</td><td>Acc@GQA</td></tr><tr><td>Open-o3-Video</td><td>44.7</td><td>41.2</td><td>54.0</td><td>27.0</td></tr><tr><td>EG-Reasoner</td><td>46.5</td><td>39.8</td><td>54.8</td><td>27.6</td></tr><tr><td>ReferClaw (Skills)</td><td>48.9</td><td>41.6</td><td>58.8</td><td>30.1</td></tr><tr><td>ReferClaw (Full)</td><td>50.1</td><td>42.6</td><td>60.7</td><td>31.6</td></tr></table>

Ask is the unnecessary-ask rate, and latency excludes respondent time. Without training skills or adapters on either image benchmark (protocols and populations in Appendix C.3), enabling clarification raises ClearVQA score from 41.5% to 48.1% and ContextClarify accuracy from 42.9% to 54.8%. These image-based tests assess clarification transfer, not temporal disambiguation. On wellspecified BuddyVQA questions, accuracy changes by +0.4 points and the assistant asks on 4.1% of inputs. Confidence intervals are in Appendix E.3. Joint Grounding and Cross-domain Transfer: Correct answers may not be supported by the intended visual evidence. We therefore also evaluate Acc@GQA on our augmented benchmarks (Tab. 7, Appendix E.1). Frozen Base reaches 36.2% on BuddyVQA versus 35.0% for EG-Reasoner (Huang et al., 2026), while Full reaches 41.8%. Under zero-shot transfer to EgoBlind-Intent (Tab. 2, all thresholds, skills, and adapters frozen; full results for both benchmarks in Tab. 7), Full keeps a 4.0-point Acc@GQA and 5.9-point answer-accuracy

Table 3: Agent and communication ablation on Augmented BuddyVQA; each row removes one agent output or cross-agent message with all else fixed. Clar.: clarification attempts after which the EFA selects the target intent; Ask: fraction of candidate-bearing inputs with a question; Cost: $\textstyle \sum _ { t } C _ { x } ( c _ { t } )$ per asked input. The last four columns average the per-input terms of Eq. 1 over candidate-bearing inputs, with Utility $= \overline { { U } } _ { \mathrm { a n s } } + 0 . 5 \overline { { U } } _ { \mathrm { g r d } } - \overline { { \Sigma C } }$ (Appendix C.4). Superscripts: change relative to the full model.
<table><tr><td>Variant</td><td>Ans. Acc@GQA</td><td></td><td>Clar.</td><td>Ask</td><td>Cost</td><td> $\overline { { U } } _ { \mathrm { a n s } }$ </td><td> $\overline { { U } } _ { \mathrm { g r d } }$ </td><td>∑C</td><td>Utility</td></tr><tr><td>ReferClaw (Full)</td><td>74.3</td><td>41.8</td><td>81.4</td><td>38.7</td><td>0.42</td><td>0.732</td><td>0.328</td><td>0.163</td><td>0.733</td></tr><tr><td>Intention Grounding Agent</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>— IGA candidates (global video to CCA/EFA)</td><td> $6 8 . 0 ^ { \downarrow 6 . 3 }$ </td><td> $3 0 . 5 ^ { \downarrow 1 1 . 3 }$ </td><td> $6 8 . 2 ^ { \downarrow 1 3 . 2 }$ </td><td> $4 2 . 0 ^ { \uparrow 3 . 3 }$ </td><td> $0 . 4 8 ^ { \uparrow 0 . 0 6 }$ </td><td></td><td></td><td></td><td>0.6660.2640.2020.596↓0.136</td></tr><tr><td>IGA temporal only (— spatial tracks)</td><td> $7 2 . 1 ^ { \downarrow 2 . 2 }$ </td><td> $3 7 . 1 ^ { \downarrow 4 . 7 }$ </td><td> $7 5 . 6 ^ { \downarrow 5 . 8 }$ </td><td> $4 0 . 0 ^ { \uparrow 1 . 3 }$ </td><td> $0 . 4 4 ^ { \uparrow 0 . 0 2 }$ </td><td>0.714 0.302</td><td></td><td>0.176</td><td> $0 . 6 8 9 ^ { \downarrow 0 . 0 4 4 }$ </td></tr><tr><td>Evidence-Fused Answering Agent</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>EFA on global video (— selected evidence)</td><td> $7 1 . 0 ^ { \downarrow 3 . 3 }$ </td><td> $3 5 . 2 ^ { \downarrow 6 . 6 }$ </td><td> $8 1 . 1 ^ { \downarrow 0 . 3 }$ </td><td>38.7</td><td></td><td>0.42 0.701 0.289</td><td></td><td>0.163</td><td>0.683 ↓0.050</td></tr><tr><td>— resolved-intent rewrite and history in EFA</td><td> $7 1 . 4 ^ { \downarrow 2 . 9 }$ </td><td> $3 8 . 8 ^ { \downarrow 3 . 0 }$ </td><td> $7 3 . 5 ^ { \downarrow 7 . 9 }$ </td><td>38.7</td><td></td><td>0.42 0.704</td><td>0.315</td><td>0.163</td><td>0.699 ↓0.034</td></tr><tr><td>— answer-intent-evidence verifier</td><td> $7 2 . 7 ^ { \downarrow 1 . 6 }$ </td><td> $3 9 . 7 ^ { \downarrow 2 . 1 }$ </td><td>81.4</td><td>38.7</td><td>0.42</td><td>0.714</td><td>0.319</td><td>0.163</td><td>0.711↓0.022</td></tr><tr><td>Cross-agent communication and control</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CCA without IGA candidates</td><td> $7 0 . 1 \downarrow 4 . 2$ </td><td> $3 6 . 3 ^ { \downarrow 5 . 5 }$ </td><td> $6 5 . 2 ^ { \downarrow 1 6 . 2 }$ </td><td> $4 5 . 0 ^ { \uparrow 6 . 3 }$ </td><td> $0 . 5 1 ^ { \uparrow 0 . 0 9 }$ </td><td>0.690 0.293 0.230</td><td></td><td></td><td> $0 . 6 0 7 ^ { \downarrow 0 . 1 2 6 }$ </td></tr><tr><td>EFA without CCA-updated posterior</td><td> $7 0 . 8 ^ { \downarrow 3 . 5 }$ </td><td> $3 7 . 3 ^ { \downarrow 4 . 5 }$ </td><td> $6 9 . 8 ^ { \downarrow 1 1 . 6 }$ </td><td>38.7</td><td></td><td>0.42 0.702 0.304 0.163</td><td></td><td></td><td> $0 . 6 9 1 ^ { \downarrow 0 . 0 4 1 }$ </td></tr><tr><td>— controller and posterior stopping</td><td> $6 9 . 9 ^ { \downarrow 4 . 4 }$ </td><td> $3 6 . 9 ^ { \downarrow 4 . 9 } 7 1 . 2 ^ { \downarrow 1 0 . 2 }$ </td><td></td><td> $5 2 . 1 ^ { \uparrow \uparrow 1 3 . 4 }$ </td><td> $0 . 6 6 ^ { \uparrow 0 . 2 4 }$ </td><td></td><td></td><td>0.682 0.297 0.344</td><td> $0 . 4 8 6 ^ { \downarrow 0 . 2 4 7 }$ </td></tr></table>

advantage over EG-Reasoner, while both metrics drop by 10–12 points relative to BuddyVQA; the drop concentrates on blurred or off-center instances, whereas clarification quality transfers almost fully (Appendix E.2), so the gap lies in perception rather than in the clarification policy.

## 4.3 ABLATION STUDY

Agents and Shared Candidates: Tab. 3 evaluates ReferClaw (Full) under matched stream order, backbone, interaction limit, and training budget. Grounded candidates support both intent resolution and clarification: replacing IGA’s candidate evidence with global video lowers Acc@GQA by 11.3 points, while withholding its candidates from CCA reduces clarification success by 16.2 points. The EFA ablations show that selected evidence, the resolved-intent rewrite and history, and verification each contribute to answer quality. Posterior-based stopping also improves accuracy while reducing respondent cost. Together, these results support sharing grounded alternatives across question selection and answering. Appendix E.4 (Fig. 8) maps the four core designs to case types: grounded candidates and contrastive clarification account for the gains on referents separated in time and for staying silent when history resolves the reference, evidence-fused answering for questions that require reasoning after the referent is resolved, and learning from interaction for failure types that recur along the stream, so the designs are complementary rather than redundant. Isolating Clarification: Disabling asking while holding the entering model state and history fixed lowers Full accuracy from 74.3% to 66.9%, measuring the immediate benefit of clarification. A separate online run that never asks reaches 64.8%, reflecting the combined effects on answering and subsequent learning. A top-two clarification heuristic reaches 71.0% at higher respondent cost, supporting selective question choice. Removing the ambiguity gate or cost penalty increases asking frequency, while generic questions reduce clarification success from 81.4% to 64.9%. Full controls and further ablations are in Appendices E.5 and E.6.

## 4.4 FURTHER ANALYSIS

1. Direct Test of Temporal Ambiguity (Q2). Fig. 7 tests whether the benefit is concentrated in temporal ambiguity by partitioning the referentially underspecified questions into five groups (sizes and assignment rules in Appendix C.2). Frozen Base exceeds Clarify-or-Answer by 19.1 points for repeated encounters with the same object (G2) and 16.1 for similar objects at different times (G3), versus 7.8 for objects visible simultaneously (G1); the Full-minus-Clarify-or-Answer margins annotated in the figure follow the same pattern. When history already resolves the ambiguity (G4), the baselines still ask on about a quarter of the inputs (27.9% for Plug-and-Play Clarifier, 24.4% for Clarify-or-Answer), whereas Base asks on 7.0% and Full on 6.5%; all methods degrade when the target is absent from the proposals (G5). The result visualization (Fig. 9, Appendix E.8) shows both effects on one episode: ReferClaw names two temporally separated encounters in one question and answers the history-resolved follow-up without asking. A separate paired-intent test presents identical video and questions with two mutually exclusive target answers, changing only the respondent’s clarification. Full answers both presentations correctly on 71.3% of pairs (Base:

![](images/61febd551160be9bd5da20719c01754e3ef5037fc24d6119bb5fe15ba9af31ff.jpg)  
Mean respondent cost per input (all inputs)

![](images/b533a73d249c82851f9e534ffe70fb75385a3a4dba9a8c78ba126152c1d17968.jpg)  
Fraction of interaction stream seen (%)  
Figure 5: Fixed-probe accuracy after each 10% of the stream under identical feedback and update budgets.

Figure 4: Accuracy–respondent-cost trade-off on Augmented BuddyVQA; the curve sweeps λ<sub>C</sub> of ReferClaw (Full).  
![](images/fb108bb53f90d4b25e5640fe7ccf95a96e1990a80b93cc162f40585fb4a1fe33.jpg)  
Figure 6: Accuracy by initial candidate entropy on candidate-bearing questions (labels: ask rate and share per bin).

![](images/bc060a1b5c0203c51c4a7d2a7a200dc05e3f3da226e41e329f7b640beaa4dfd6.jpg)  
Figure 7: Accuracy by ambiguity group G1– G5 (Sec. 4.4); labels: ReferClaw (Full) minus Clarify-or-Answer.

64.2%), versus 38.8% for Clarify-or-Answer and 33.3% for Plug-and-Play Clarifier; deterministic direct answering scores zero because it receives identical inputs. This tests whether answers follow the clarified intent rather than a fixed interpretation; clarification-quality metrics on the ambiguous set and example traces are in Appendices E.7 and E.8.

2. Efficiency and Respondent Effort (Q3). Fig. 4 averages respondent cost over all inputs; the Full curve sweeps λ from 0 (right) to asking disabled (left). Filled markers denote frozen systems and hollow markers Full-based policy controls. Base outperforms frozen clarification baselines at lower cost. Full asks on 18.6% of inputs, averaging 0.22 questions per input. Its questions average 20.6 tokens and take humans 5.1 s to answer, against 24.0 tokens and 6.9 s for Clarify-or-Answer. Removing the cost penalty more than doubles cost without improving accuracy; a fixed two-turn policy also costs more and performs worse. Gains over the matched asking-disabled replay are larger for high-entropy candidates (Fig. 6; Appendix E.9). Timing and cost components are in Appendix E.10. Human-respondent results are reported in Appendix E.11.

3. Learning Dynamics (Q4). On an unchanged held-out probe, Skills improves over Base by 4.2 points and Full by 9.4, compared with 0.9 for Reflexion-style memory and 2.8 for MetaClaw-style adaptation (Fig. 5). Full also cuts failure recurrence from 34.2% to 10.7% with positive backward transfer (+4.1), versus 21.3% and +1.1 for MetaClaw-style adaptation. Targeted ablations further show that a generic skill schema reduces answer accuracy by 1.8 points and updating all adapters rather than the attributed agent reduces it by 2.0 (Appendix E.6). Since the MetaClaw-style variant alters the skill schema, the adapter-update scope, and the reward jointly, its gap to Full confounds the three factors; the single-factor ablations above disentangle them and attribute a gain to each. Learning curves, recurrence rates, and backward-transfer results are reported in Appendix E.12, and a manual audit of residual errors is provided in Appendix E.9.

## 5 CONCLUSION

We studied intent ambiguity in egocentric assistance, where video and dialogue evidence supports multiple possible referents. ReferClaw explicitly represents competing interpretations with spatiotemporal evidence, allowing the same grounded candidates to determine whether to clarify, what to ask, and which evidence to use. Three cooperating agents make these decisions under a fixed interaction budget, while verified feedback is distilled into intent-specific skills and agent-attributed process rewards to reduce recurring errors. On augmented ego-streaming assistance benchmarks, Refer-

Claw asks more useful clarification questions than strong interactive baselines, improves intentaligned and jointly grounded answers with lower user effort, and provides the largest gains when competing referents are temporally separated. The learned skills and parameters further improve fixed probes and reduce repeated failures, while the frozen system transfers to two independent image-based clarification benchmarks without degrading well-specified questions.

## ETHICS STATEMENT

This work adheres to the ICLR Code of Ethics. All video and image data come from publicly released benchmarks (BuddyVQA, EgoBlind, ClearVQA, and ContextClarify) and are used under their licenses. The user study (Appendix B.3) involved adult volunteers who gave informed consent, answered clarification questions about public clips, and were compensated; only anonymized responses and response times were recorded. Because ReferClaw targets assistive use, including questions from blind and visually impaired users, an answer to the wrong interpretation could mislead a user who cannot verify the evidence, and clarification imposes effort. The system therefore grounds answers in verifiable evidence, abstains or gives conditional answers when the intent cannot be resolved, and bounds the number, length, and cost of questions; it remains a research prototype and should not be deployed for safety-critical assistance without validation.

We have taken care to avoid any possible biases or discriminatory outcomes in our research process. No personally identifiable information was used, and no experiments were conducted that could raise privacy or security concerns. We are committed to maintaining transparency and integrity throughout the research process.

## REPRODUCIBILITY STATEMENT

We have made every effort to ensure that the results presented in this paper are reproducible. All code, prompts, trained LoRA adapters, the skill library, and our augmented annotations (intentambiguity labels, target intents, supporting intervals, evidence-frame boxes, diagnostic groups, and the paired-intent set) will be provided upon acceptance. The method is specified in Sec. 3 and Appendix B (controller transition table, response likelihood and cost, process reward, promotion rule); all hyperparameters are listed in Tab. 6 and the hardware and optimization setup in Appendix D. The evaluation protocol, including what the assistant and the simulated user receive, when feedback is revealed, and the population of every metric, is documented in Appendix C.4, and all prompts are given verbatim in Appendix F. All benchmarks and backbone or tool models are publicly available. Human-respondent results depend on the recruited participants and are reported separately. We believe these measures will enable other researchers to reproduce our work and further advance the field.

## REFERENCES

Ehsan Abbasnejad, Qi Wu, Qinfeng Shi, and Anton van den Hengel. What’s to know? uncertainty as a guide to asking goal-oriented questions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4155–4164, 2019. doi: 10.1109/CVPR.2019. 00428.

Leonard Barmann and Alex Waibel. Where did i leave my keys?—episodic-memory-based question¨ answering on egocentric videos. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 1559–1567. IEEE, 2022.

Zongwan Cao, Bingbing Wen, and Lucy Lu Wang. Clarify or answer: Reinforcement learning for agentic VQA with context under-specification. arXiv preprint arXiv:2601.16400, 2026.

Chongyan Chen, Yu-Yun Tseng, Zhuoheng Li, Anush Venkatesh, and Danna Gurari. Acknowledging focus ambiguity in visual questions. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1228–1238, 2025a.

Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. M3-Embedding: Multi-linguality, multi-functionality, multi-granularity text embeddings through self-knowledge

distillation. In Findings of the Association for Computational Linguistics: ACL 2024, pp. 2318– 2335, 2024a.

Qirui Chen, Shangzhe Di, and Weidi Xie. Grounded multi-hop videoqa in long-form egocentric videos. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 2159– 2167, 2025b. doi: 10.1609/aaai.v39i2.32214.

Weize Chen, Yusheng Su, Jingwei Zuo, Cheng Yang, Chenfei Yuan, Chi-Min Chan, Heyang Yu, Yaxi Lu, Yi-Hsin Hung, Chen Qian, Yujia Qin, Xin Cong, Ruobing Xie, Zhiyuan Liu, Maosong Sun, and Jie Zhou. AgentVerse: Facilitating multi-agent collaboration and exploring emergent behaviors. In International Conference on Learning Representations, 2024b.

Yura Choi, Roy Miles, Rolandos Alexandros Potamias, Ismail Elezi, Jiankang Deng, and Stefanos Zafeiriou. Do you see what i am pointing at? gesture-based egocentric video question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18534–18544, 2026.

Herbert H. Clark and Susan E. Brennan. Grounding in communication. In Lauren B. Resnick, John M. Levine, and Stephanie D. Teasley (eds.), Perspectives on Socially Shared Cognition, pp. 127–149. American Psychological Association, 1991. doi: 10.1037/10096-006.

Jisheng Dang, Huilin Song, Junbin Xiao, Bimei Wang, Han Peng, Haoxuan Li, Xun Yang, Meng Wang, and Tat-Seng Chua. MUPA: Towards multi-path agentic reasoning for grounded video question answering. arXiv preprint arXiv:2506.18071, 2025.

Abhishek Das, Satwik Kottur, Jose M. F. Moura, Stefan Lee, and Dhruv Batra. Learning cooperative´ visual dialog agents with deep reinforcement learning. In Proceedings of the IEEE International Conference on Computer Vision, pp. 2951–2960, 2017.

Harm de Vries, Florian Strub, Sarath Chandar, Olivier Pietquin, Hugo Larochelle, and Aaron Courville. Guesswhat?! visual object discovery through multi-modal dialogue. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, pp. 5503–5512, 2017. doi: 10.1109/CVPR.2017.475.

Shangzhe Di and Weidi Xie. Grounded question-answering in long egocentric videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12934– 12943, 2024. doi: 10.1109/CVPR52733.2024.01229.

Shangzhe Di, Zhelun Yu, Guanghao Zhang, Haoyuan Li, Tao Zhong, Hao Cheng, Bolin Li, Wanggui He, Fangxun Shu, and Hao Jiang. Streaming video question-answering with in-context video KVcache retrieval. In International Conference on Learning Representations, 2025.

Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18995–19012, 2022.

William E. Hick. On the rate of gain of information. Quarterly Journal of Experimental Psychology, 4(1):11–26, 1952. doi: 10.1080/17470215208416600.

Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu, and Jurgen Schmidhuber. MetaGPT: Meta programming for a multi-agent collab-¨ orative framework. In International Conference on Learning Representations, 2024.

Linpeng Huang, Weixing Chen, Zexin Chen, Yang Liu, and Liang Lin. EG-VQA: Benchmarking verifiable video question answering with grounded temporal evidence. arXiv preprint arXiv:2606.24797, 2026.

Ray Hyman. Stimulus information as a determinant of reaction time. Journal of Experimental Psychology, 45(3):188–196, 1953. doi: 10.1037/h0056940.

Pu Jian, Donglei Yu, Wen Yang, Shuo Ren, and Jiajun Zhang. Teaching vision-language models to ask: Resolving ambiguity in visual questions. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics, pp. 3619–3638, 2025. doi: 10.18653/v1/2025. acl-long.182.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with PagedAttention. In Proceedings ofthe 29th Symposium on Operating Systems Principles, pp. 611–626, 2023. doi: 10.1145/3600006.3613165.

Dongryeol Lee, Segwang Kim, Minwoo Lee, Hwanhee Lee, Joonsuk Park, Sang-Woo Lee, and Kyomin Jung. Asking clarification questions to handle ambiguity in open-domain QA. In Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 11526–11544, 2023. doi: 10.18653/v1/2023.findings-emnlp.772.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In International Conference on Learning Representations, 2024.

Ye Liu, Kevin Qinghong Lin, Chang Wen Chen, and Mike Zheng Shou. VideoMind: A chain-of-LoRA agent for temporal-grounded video reasoning. In International Conference on Learning Representations (ICLR), 2026.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. Self-refine: Iterative refinement with self-feedback. In Advances in Neural Information Processing Systems, volume 36, pp. 46534–46594, 2023.

Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. EgoSchema: A diagnostic benchmark for very long-form video language understanding. In Advances in Neural Information Processing Systems, volume 36, 2023.

Jiahao Meng, Xiangtai Li, Haochen Wang, Yue Tan, Tao Zhang, Lingdong Kong, Yunhai Tong, Anran Wang, Zhiyang Teng, Yujing Wang, and Zhuochen Wang. Open-o3 Video: Grounded video reasoning with explicit spatio-temporal evidence. In Proceedings of the International Conference on Machine Learning (ICML), 2026.

Sewon Min, Julian Michael, Hannaneh Hajishirzi, and Luke Zettlemoyer. AmbigQA: Answering ambiguous open-domain questions. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, pp. 5783–5797, 2020. doi: 10.18653/v1/2020.emnlp-main.466.

Shehan Munasinghe, Hanan Gani, Wenqi Zhu, Jiale Cao, Eric Xing, Fahad Shahbaz Khan, and Salman Khan. VideoGLaMM: A large multimodal model for pixel-level visual grounding in videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 19036–19046, 2025.

Taiying Peng, Jiacheng Hua, Miao Liu, and Feng Lu. In the eye of MLLM: Benchmarking egocentric video intent understanding with gaze-guided prompting. Advances in Neural Information Processing Systems, 38, 2025.

Hangyu Qin, Junbin Xiao, Shenglang Zhang, and Angela Yao. Companion-style QA assistance in ego-vision. arXiv preprint arXiv:2609.06721, 2026.

Qwen Team. Qwen3.5: Towards native multimodal agents. Qwen Technical Blog, February 2026. URL https://qwen.ai/blog?id=qwen3.5.

Sudha Rao and Hal Daume III. Learning to ask good questions: Ranking clarification questions us-´ ing neural expected value of perfect information. In Proceedings of the 56th Annual Meeting of the Associationfor Computational Linguistics, pp. 2737–2746, 2018. doi: 10.18653/v1/P18-1255.

Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Radle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Va-¨ sudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollar, and Christoph Fe-´ ichtenhofer. SAM 2: Segment anything in images and videos. In International Conference on Learning Representations (ICLR), 2025.

Tianhe Ren, Qing Jiang, Shilong Liu, Zhaoyang Zeng, Wenlong Liu, Han Gao, Hongjie Huang, Zhengyu Ma, Xiaoke Jiang, Yihao Chen, Yuda Xiong, Hao Zhang, Feng Li, Peijun Tang, Kent Yu, and Lei Zhang. Grounding DINO 1.5: Advance the “edge” of open-set object detection. arXiv preprint arXiv:2405.10300, 2024.

Marzieh Saeidi, Max Bartolo, Patrick Lewis, Sameer Singh, Tim Rocktaschel, Mike Sheldon, Guil-¨ laume Bouchard, and Sebastian Riedel. Interpretation of natural language rules in conversational machine reading. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing, pp. 2087–2097, 2018. doi: 10.18653/v1/D18-1233.

Jinhwan Seo, Kyubeom Han, Jumin Lee, Junhyug Noh, and Sung-eui Yoon. What you ask is what you ground: Bridging question intent to temporal evidence for grounded videoqa. arXiv preprint arXiv:2608.15708, 2026.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems, volume 36, 2023. doi: 10.52202/075280-0377.

Pengzhan Sun, Junbin Xiao, Tze Ho Elden Tse, Yicong Li, Arjun Akula, and Angela Yao. Visual intention grounding for egocentric assistants. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 2512–2522. IEEE, 2025.

Haoqin Tu, Jianwen Chen, Zijun Wang, Siwei Han, Juncheng Wu, Hardy Chen, Haonian Ji, Kaiwen Xiong, Jiaqi Liu, Peng Xia, et al. Visualclaw: A real-time, personalized agent for the physical world. arXiv preprint arXiv:2606.16295, 2026.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. Transactions on Machine Learning Research, 2024.

Haochen Wang, Qirui Chen, Cilin Yan, Jiayin Cai, Xiaolong Jiang, Yao Hu, Weidi Xie, and Stratis Gavves. Object-centric video question answering with visual grounding and referring. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 22274–22284, 2025a.

Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. InternVL3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025b.

Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. Agent workflow memory. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pp. 63897–63911. PMLR, 2025c.

Julia White, Gabriel Poesia, Robert Hawkins, Dorsa Sadigh, and Noah Goodman. Open-domain clarification question generation without question examples. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 563–570, 2021. doi: 10.18653/v1/2021.emnlp-main.44.

Peng Xia, Jianwen Chen, Xinyu Yang, Haoqin Tu, Jiaqi Liu, Kaiwen Xiong, Siwei Han, Shi Qiu, Haonian Ji, Yuyin Zhou, Zeyu Zheng, Cihang Xie, and Huaxiu Yao. Metaclaw: Just talk—an agent that meta-learns and evolves in the wild. arXiv preprint arXiv:2603.17187, 2026.

Junbin Xiao, Angela Yao, Yicong Li, and Tat-Seng Chua. Can i trust your answer? visually grounded video question answering. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13204–13214, 2024. doi: 10.1109/CVPR52733.2024.01254.

Junbin Xiao, Nanxin Huang, Hao Qiu, Zhulin Tao, Xun Yang, Richang Hong, Meng Wang, and Angela Yao. Egoblind: Towards egocentric visual assistance for the blind. In Advances in Neural Information Processing Systems, volume 38, 2025. doi: 10.52202/085713-0886. Datasets and Benchmarks Track.

Jingjing Xu, Yuechen Wang, Duyu Tang, Nan Duan, Pengcheng Yang, Qi Zeng, Ming Zhou, and Xu Sun. Asking clarification questions in knowledge-based question answering. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, pp. 1618–1629, 2019. doi: 10.18653/v1/D19-1172.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a.

Jingkang Yang, Shuai Liu, Hongming Guo, Yuhao Dong, Xiamengwei Zhang, Sicheng Zhang, Pengyun Wang, Zitang Zhou, Binzhu Xie, Ziyue Wang, et al. Egolife: Towards egocentric life assistant. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 28885–28900. IEEE, 2025b.

Sicheng Yang, Yukai Huang, Weitong Cai, Shitong Sun, You He, Jiankang Deng, Hang Zhang, Jifei Song, and Zhensong Zhang. Plug-and-play clarifier: A zero-shot multimodal framework for egocentric intent disambiguation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pp. 17921–17929, 2026. doi: 10.1609/aaai.v40i21.38851.

Wenzheng Zeng, Difei Gao, Mike Zheng Shou, and Hwee Tou Ng. Factorized learning for temporally grounded video-language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 20683–20693, 2025.

Haoji Zhang, Yiqin Wang, Yansong Tang, Yong Liu, Jiashi Feng, and Xiaojie Jin. Flash-VStream: Efficient real-time understanding for long video streams. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pp. 21059–21069, 2025a.

Zhenru Zhang, Chujie Zheng, Yangzhen Wu, Beichen Zhang, Runji Lin, Bowen Yu, Dayiheng Liu, Jingren Zhou, and Junyang Lin. The lessons of developing process reward models in mathematical reasoning. In Findings of the Association for Computational Linguistics: ACL 2025, pp. 10495– 10516, 2025b. doi: 10.18653/v1/2025.findings-acl.547.

Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. ExpeL: LLM agents are experiential learners. In Proceedings of the AAAI Conference on Artificial Intel ligence, volume 38, pp. 19632–19642, 2024. doi: 10.1609/aaai.v38i17.29936.

## A LLM USAGE

LLMs were used to aid in the writing and polishing of the manuscript. Specifically, we used an LLM to refine the language, improve readability, and ensure clarity across various sections of the paper. The model helped with tasks such as rephrasing sentences, checking grammar, and improving the overall flow of the text.

It is important to note that the LLM was not involved in the ideation, research methodology, or experimental design. All research concepts, ideas, and analyses were developed and conducted by the authors. The contributions of the LLM were solely focused on improving the linguistic quality of the paper, with no involvement in the scientific content or data analysis.

The authors take full responsibility for the content of the manuscript, including any text generated or polished by the LLM. We have ensured that the LLM-generated text adheres to ethical guidelines and does not contribute to plagiarism or scientific misconduct.

## B METHOD DETAILS

## B.1 MODULE INSTANTIATION AND CANDIDATE GROUNDING

IGA, CCA, and EFA share a Qwen3.5-9B backbone but use separate LoRA adapters, which keeps the visual representation fixed while allowing agent-specific updates. IGA uses the released Ground-VQA checkpoint for coarse temporal proposals, Grounding DINO 1.5 Pro (Ren et al., 2024) for open-vocabulary boxes, and SAM 2 Hiera-Large (Ravi et al., 2025) to propagate the selected box through the retrieved interval. CCA and EFA operate on the resulting evidence cards and Qwen3.5- 9B video tokens. BGE-M3 (Chen et al., 2024a) encodes skills and trajectories for cosine retrieval. The process reward model initializes from Qwen3.5-9B, adds a scalar value head, and is trained to regress the computed reward of Eq. 6 on query trajectories with feedback (held-out Spearman 0.83, MAE 0.09 on a [0, 1]-scaled reward). The predictive respondent model used by CCA to esti mate response likelihoods is the Qwen3.5-9B backbone under the predictive-respondent prompt of Appendix F; it never receives the annotated target intent. The evaluation simulator that generates the actual responses during testing and learning rollouts is Qwen3-32B (Yang et al., 2025a); it receives the hidden target intent and its evidence but never the system posterior, candidate scores, o reference answer. Eight samples per $( c , z _ { i } )$ pair estimate each response-class distribution.

## B.2 RESPONSE LIKELIHOOD AND RESPONDENT COST

Response Classes and Likelihood. Free-text responses are assigned to the nearest response class (a partition cell, “other/none”, uncertain, or invalid/missing) by BGE-M3 embedding similarity. For the LLM respondent, the likelihood $p _ { \mathrm { L } } ( y \mid c , z _ { i } , h _ { t } )$ is the class frequency over eight samples of the predictive respondent model per $( c , z _ { i } )$ pair, with additive smoothing $\delta = 0 . 0 5$ so that no class receives zero probability. For the human respondent, $p _ { \mathrm { H } } = \left( 1 - \kappa _ { \mathrm { H } } - f _ { \mathrm { H } } \right) p _ { \mathrm { L } } + \kappa _ { \mathrm { H } } \mathrm { U n i f } + f _ { \mathrm { H } } q _ { \mathrm { F } }$ where $q _ { \mathrm { F } }$ is a normalized distribution over the failure classes, with $\kappa _ { \mathrm { H } } , f _ { \mathrm { H } }$ , and q fitted as described in Appendix B.3. For both respondent types the failure-class likelihood depends on the question only, not on z<sub>i</sub>, so a failure response leaves the belief unchanged under Eq. 4, which is also what the controller executes. The uncovered pseudo-candidate $z _ { \mathcal { O } }$ has the likelihood $p _ { x } ( y \mid c , z _ { \mathcal { D } } , h _ { t } ) =$ $( 1 - f ( c ) ) \mathbb { I } [ y = \mathrm { o t h e r / n o n e } ] + f ( c ) q _ { \mathrm { F } } ( \bar { y } )$ , where $f ( c )$ is the candidate-independent failure mass of the question: a user whose intent is not among the displayed candidates selects “other/none” unless the response fails. Entropy, expected information gain, and the posterior update are all computed over the $n _ { t } + 1$ elements $\{ z _ { 1 } , \dots , z _ { n _ { t } } , z _ { \mathcal { O } } \}$

Cost Components. For a human respondent, $\overline { { E } } _ { \mathrm { H } }$ averages the normalized question length, the expected response length under the predictive respondent, and the choice complexity $\log _ { 2 } ( m ( c ) +$ $1 ) / \log _ { 2 } ( M _ { \mathrm { m a x } } + 1 ) ; \overline { { T } } _ { \mathrm { H } }$ is the response time predicted by a linear regression on question length and option count fitted on the calibration half of the user study $( \overline { { T } } _ { \mathrm { H } } = 1 . 0 + 0 . 1 2 | c | + 0 . 5 5 m ( c )$ seconds); and $\overline { { F } } _ { \mathrm { H } }$ is the predicted mass of the invalid, missing, and uncertain response classes. For an LLM respondent, $\overline { { E } } _ { \mathrm { { L } } }$ counts prompt and completion tokens, $\overline { { T } } _ { \mathrm { L } }$ is measured serving latency, and $\overline { { F } } _ { \mathrm { I } }$ is the predicted invalid-format or self-inconsistent mass. A failed response already yields little information gain; ${ \overline { { F } } } _ { x }$ additionally charges the wasted turn and the user’s time. All components are normalized to [0, 1] with the 5th and 95th percentiles of the validation split, and the weights $( w _ { E } , w _ { T } , w _ { F } )$ are fitted by grid search on validation utility separately for the two respondent types, with a shared $\lambda _ { C }$ (Tab. 6).

## B.3 HUMAN RESPONSE MODEL CALIBRATION

The 600-interaction user study (10 participants, 300 stratified ambiguous examples, each example answered under ReferClaw and under Clarify-or-Answer by different participants) serves two purposes that we keep separate by example. The 300 examples are split once into a calibration half (150 examples) and a held-out evaluation half (150 examples). On the calibration half we use the ReferClaw interactions to fit the human response model

$$
\begin{array} { r } { p _ { \mathrm { H } } ( y \mid c , z _ { i } , h _ { t } ) = \left( 1 - \kappa _ { \mathrm { H } } - f _ { \mathrm { H } } \right) \hat { p } _ { \mathrm { L } } ( y \mid c , z _ { i } , h _ { t } ) + \kappa _ { \mathrm { H } } \mathrm { U n i f } ( y ) + f _ { \mathrm { H } } q _ { \mathrm { F } } ( y ) , } \end{array}
$$

where $\hat { p } _ { \mathrm { { L } } }$ is the predictive respondent’s class distribution, Unif is uniform over all response classes, and $q _ { \mathrm { F } }$ is a distribution supported on the three failure classes (uncertain, invalid, no response) with proportions (0.55, 0.25, 0.20) estimated from the study; each component is a normalized distribution, so $p _ { \mathrm { H } }$ sums to one. “Other/none” is an ordinary answer class, not a failure class, because it is the correct response when the intended referent is missing from the options. The fitted weights are $\kappa _ { \mathrm { H } } ~ = ~ 0 . 1 8$ and $f _ { \mathrm { H } } ~ = ~ 0 . 0 6$ Response time is fitted on the same half as $\overline { { T } } _ { \mathrm { H } } = 1 . \dot { 0 } + 0 . 1 2 \left| c \right| + 0 . 5 5 m ( c )$ seconds $( R ^ { 2 } = 0 . 6 1 )$ ), with $| c |$ the question length in tokens and $m ( c )$ the number of options. On the held-out half, the calibrated model lowers the Brier score of response prediction from 0.31 (uncalibrated simulator frequencies) to 0.22, and its predicted response times correlate with measured times at $r = 0 . 7 4$ . These parameters define the likelihoods and costs used when the respondent is human and the human-calibrated profile of the evaluation simulator; simulator-based likelihoods and costs use the simulator’s own statistics. The human comparison in Appendix E.11 uses only the held-out half for both methods; the Clarify-or-Answer interactions of the calibration half enter only the descriptive response-time statistics of Tab. 11.

## B.4 INTERACTION CONTROL ALGORITHM

The interaction controller of Sec. 3.2 is a deterministic state machine over the state $( t , \mathcal { Z } _ { t } , p _ { t } , h _ { t } , B _ { t } , n _ { \mathrm { r g } } )$ , where $B _ { t }$ is the set of banned partitions and $n _ { \mathrm { r g } } \in \{ 0 , 1 \}$ counts re-grounding passes. Tab. 4 lists its transitions; at every turn the first row whose condition holds is executed, and the implementation, the pseudocode, and the main text share this table. All limits $( T _ { \mathrm { m a x } } , \ L _ { \mathrm { m a x } } ,$ $M _ { \mathrm { m a x } } ,$ no repeated partitions) are enforced before generation, so every trajectory terminates within $T _ { \mathrm { m a x } }$ turns. Two procedures are shared by all rows. ANSWERPATH(i<sup>⋆</sup>) runs the EFA on candidate $i ^ { \star }$ and then the verifier: pass returns the answer; revise regenerates once and returns the answer if the second verdict is pass, otherwise ABSTAIN; abstain returns ABSTAIN. REGROUND(·) runs the IGA once more, with relaxed detector thresholds (initial state) or restricted to the constraint stated in the response (after “other/none”); it may be invoked at most once per question $( n _ { \mathrm { r g } } \leq 1 )$ , new candidates absorb the mass of $z _ { \emptyset }$ in proportion to their grounding scores, and an empty result triggers row T1 on the next evaluation.

Table 4: Priority-ordered controller transitions for one user question. Rows are tested top to bottom at every turn t; the first satisfied row is executed. i<sup>⋆</sup> = arg max<sub>i</sub> $p _ { t } ( z _ { i } ) ;$ (1), (2) index the two most likely candidates.
<table><tr><td></td><td>Condition</td><td>Action</td><td>Output / next state</td></tr><tr><td>T1</td><td> $\mathcal { Z } _ { t } = \emptyset$ </td><td>if  $n _ { \mathrm { r g } } ~ = ~ 0 ;$  REGROUND,  $n _ { \mathrm { r g } } \  \ 1 ,$  re- evaluate; else terminate</td><td>ABSTAIN</td></tr><tr><td>T2</td><td> $g _ { t } = 0 \mathrm { ~ } \mathbf { o r } \operatorname* { m a x } _ { i } p _ { t } ( z _ { i } ) \mathbf { \eta } \geq$   $\tau _ { p }$ </td><td>ANSWERPATH(i*); if  $g _ { t } ~ = ~ 0$  plausible candidates share an answer, EFA STAIN receives their pooled evidence and  $\gamma$  is their</td><td>because all ANSWER or AB-</td></tr><tr><td>T3</td><td> $\begin{array} { c c c c c } { g _ { t } } & { = } & { 1 , } & { t } & { < } & { T _ { \mathrm { m a x } } , } \end{array}$  CCA returns  $c _ { t } ^ { \star }$   $\begin{array} { r l r l } { \mathrm { I G } _ { t } - \lambda _ { C } C _ { x } } & { { } > } & { 0 } \end{array}$  and partition  $\notin B _ { t }$ </td><td>summed mass ask  $c _ { t } ^ { \star } ;$  receive  $y _ { t } ; h _ { t + 1 }  h _ { t } \cup \{ ( c _ { t } ^ { \star } , y _ { t } ) \} ;$  with then (a) yt uncertain, invalid, or missing:  $p _ { t + 1 }  p _ { t } , B _ { t + 1 }  B _ { t } \cup$  {partition  $( c _ { t } ^ { \star } ) \}$  (b) yt = other/none or unmatched constraint: REGROUND(yt) if  $n _ { \mathrm { r g } } = 0 ,$  else ABSTAIN; (c) otherwise  $p _ { t + 1 }$  by Eq. 4, drop candidates</td><td>; continue at T1</td></tr><tr><td></td><td>T4 otherwise  $~ ( t ~ = ~ T _ { \operatorname* { m a x } } ,$  no admissible question with positive utility)</td><td>or if  $\bar { a } _ { ( 1 ) } \neq \bar { a } _ { ( 2 ) }$  and  $p _ { t } \big ( z _ { ( 1 ) } \big ) - p _ { t } \big ( z _ { ( 2 ) } \big ) \ < $   $\tau _ { g } \mathrm { : }$  ANSWERPATH((1)) and ANSWER- (if both branches PATH((2)), each labeled by its interpreta- pass; ABSTAIN if tion; else ANSWERPATH(i*)</td><td>CONDITIONAL either fails) or  $\mathbf { A N } -$  SWER/ABSTAIN</td></tr></table>

Three consequences of the table are used in the analysis. First, every formal answer, including each branch of a CONDITIONAL output, is produced by the same EFA-then-verifier path, so verifier statistics are comparable across output types. Second, a question is asked only through T3, so the number of questions per input is $N _ { \mathrm { a s k } } \leq T _ { \mathrm { m a x } }$ and a question can be asked only if $n _ { 0 } \geq 2 \left( \mathrm { w i t h } n _ { 0 } = \right.$ 1 the gate is closed). Third, uncertain and invalid responses never change the belief, which matches the candidate-independent failure-class likelihood used when scoring questions (Appendix B.2), so the expected information gain computed before asking and the update executed after asking are consistent.

## B.5 FAILURE ATTRIBUTION RULES AND THEIR VALIDATION

A verified failure is attributed to one agent by deterministic rules on the logged trajectory wherever possible: $z ^ { \star }$ absent from ${ \mathcal { Z } } ,$ or present but grounded with IoU $< 0 . 5$ against $e ^ { \star }$ , attributes to IGA (coverage or grounding failure) $) ; z ^ { \star }$ present but placed in the same partition cell as the top competitor of the asked question, or a question asked when all plausible provisional answers already agreed, attributes to CCA (non-separating or unnecessary question); $z ^ { \star }$ selected with correct evidence but a wrong or unsupported answer attributes to EFA (inconsistent answer). Only trajectories that match none of these rules are attributed by the process-review prompt of Appendix F. Two authors independently attributed 200 randomly sampled verified failures using the trajectory and feedback (interannotator agreement $\kappa = 0 . 8 6 ;$ disagreements adjudicated). The deterministic rules cover 84% of the failures and agree with the adjudicated label in 94% of covered cases; the process-review prompt handles the remaining 16% with 78% agreement, for a combined agreement of 91%. Most residual disagreements concern trajectories in which the target intent was covered with borderline IoU (0.4–0.5) and the question also failed to separate it, i.e., cases with two plausible responsible agents.

## B.6 SKILL SCHEMA AND A WORKED EXAMPLE

Each skill stores an ambiguity trigger, a candidate topology, the decisive visual contrast, the corrected action, an anti-pattern, a retrieval key, and two regression tests (Appendix F gives the evolver prompt and schema). Retrieval uses BGE-M3 cosine similarity between the current trigger–topology description and stored keys with threshold 0.45; at most three skills are injected into the responsible agent’s prompt. The following trace illustrates what is learned beyond the initial prompt, which already instructs CCA to contrast occurrences rather than repeat a noun. Failure: for a mug picked up twice in the same kitchen within 40 seconds, CCA asked “the first or the second time $\therefore ? ^ { \ast }$ ; the user answered uncertain and the final answer was wrong. Feedback: $z ^ { \star }$ was the second encounter, during which the user was pouring. Stored skill (CCA): trigger “same object, repeated encounters, same location, $\mathrm { g a p } < 6 0 \mathrm { s } ^ { \prime \prime }$ ; corrected action “contrast by the user’s action visible in each track (holding vs. pouring), not by order or location”; anti-pattern “ordinal or location contrast for repeated encounters at one place”. Later behavior: on a subsequent kettle question with the same topology, CCA asked “when you were filling it or when you were pouring?”, the response resolved the intent, and the answer was correct. Such action-contrast skills account for 23% of the CCA skills learned on Augmented BuddyVQA.

## B.7 PROCESS REWARD, ADAPTER UPDATE, AND PROMOTION

Process Reward. The target-progress term $\Delta _ { t } ^ { \star } = \log p _ { t + 1 } ( z ^ { \star } ) - \log p _ { t } ( z ^ { \star } )$ of Eq. 6 is evaluated on the candidate scope that existed before turn t: candidates dropped or added after the response are re-inserted with their pre-update mass for the purpose of this term, so a change of scope cannot earn reward by itself, and $\Delta _ { t } ^ { \star }$ is 0 when $z ^ { \star }$ was not covered at turn t. Directly computed terms are $r _ { \mathrm { g r o u n d : } }$ $\begin{array} { r } { \sum _ { t } \boldsymbol { C } _ { \boldsymbol { x } } ( \dot { \boldsymbol { c } } _ { t } ) , \sum _ { t } \Delta _ { t } ^ { \star } } \end{array}$ , and $r _ { \mathrm { i n v a l i d } } ; \ r _ { \mathrm { f i n a l } }$ combines the judge verdict with the verifier’s verdict. The process reward model predicts $R _ { \mathrm { p r o c } }$ for sampled alternative trajectories whose feedback is not yet available.

Adapter Update. When the query buffer holds 128 trajectories and no online request is being served, each failed step is re-executed: the attributed agent samples $G = 8$ alternative structured outputs from its current adapter given the exact logged input, the process reward model scores each completed trajectory, and the adapter is updated with GRPO, i.e., group-normalized advantages under a clipped policy objective (clip ratio 0.2) with a KL penalty (coefficient 0.02) toward the frozen Base adapter. Only the attributed adapter changes. A candidate adapter is promoted only if, on a fixed validation probe of 500 ambiguous and 500 well-specified examples, it improves accuracy on its target failure type by at least one point, lowers overall probe accuracy by at most 0.5 points, and does not raise the unnecessary-ask rate or mean cost by more than 0.01; otherwise it is rolled back and the buffer is retained for the next consolidation window.

## C BENCHMARKS, ANNOTATION PROTOCOL, AND EVALUATION DETAILS

## C.1 AUGMENTED BENCHMARKS

Augmented BuddyVQA. BuddyVQA (Qin et al., 2026) is our primary benchmark because its causal streaming protocol, ego-deictic expressions, and history-dependent question chains jointly exercise visual memory, referent tracking, and dialogue dependence. The original release contains 1,012 videos and 21,574 questions with a 704/308 video split for training and test; we annotate every question with an intent-ambiguity label, a minimal supporting temporal interval, and a box on the clearest evidence frame when the referent is spatially localizable (otherwise the instance is marked non-localizable). The original test split has 6,571 questions on the 308 test videos. We exclude 331 of them (5.0%): 214 whose timestamp lies within the first 30 seconds of the video, so that no meaningful causal prefix exists, and 117 for which the two annotators could not agree on a reference answer under the annotated interpretation after adjudication. The resulting test split contains 6,240 questions (Recall 1,790, Recognition 1,650, Advisory 1,420, Scrutinization 1,380). The published BuddyVQA results quoted in Tab. 1 (†) were obtained by their authors under the original protocol on all 6,571 test questions; their type scores therefore do not satisfy the 6,240- question weighting, and we do not compare against them at the type level or compute differences to them. Of these, 1,398 are referentially underspecified in wording; 1,197 of them (19.2%; Recall 344, Recognition 318, Advisory 272, Scrutinization 263) remain intent-ambiguous given the prefix and the dialogue history and form the ambiguous population of every metric, whereas the other 201 are resolved by the history (group G4 below) and are labeled well-specified, giving 5,043 verifiedunambiguous questions. 4,655 questions (74.6%) are grounding-eligible, i.e., have an annotated supporting interval, and 3,914 of these additionally have a spatial box (741 are temporal-only).

Augmented EgoBlind-Intent. EgoBlind (Xiao et al., 2025) complements BuddyVQA with timestamped assistance questions from blind and visually impaired users under imperfect viewpoints, motion blur, and partial observation. We select 2,036 QA instances from the official test set and add the same ambiguity, interval, and box annotations. 612 instances are intent-relevant because their ego-deictic, elliptical, or context-dependent wording requires explicit referent resolution; among these, 418 satisfy the stricter answer-changing criterion used for BuddyVQA and form the ambigu ous subset on which clarification metrics are computed, and 1,447 instances are grounding-eligible. Because the two datasets differ in how much of the intent-relevant population is answer-changing, we never pool them. The benchmark is reserved for zero-shot transfer: no instance is used for prompt selection, threshold tuning, skill synthesis, or policy optimization.

## C.2 ANNOTATION PROTOCOL

Guidelines. Annotators watched only the video prefix up to the question timestamp together with the preceding dialogue. For each question they (i) listed every entity, moment, or place that the question could refer to and that was visible in the prefix; (ii) for each interpretation wrote a resolved question and a short provisional answer from the video; (iii) marked the question intent-ambiguous if at least two interpretations were visually supported given the prefix and the dialogue history and their provisional answers differed, and marked it history-resolved if the wording alone admitted several interpretations but a prior dialogue turn selected one of them; and (iv) annotated the minimal supporting interval and, when applicable, a single box on the clearest evidence frame for each interpretation (no box tracks are annotated; spatial IoU is therefore evaluated on that frame, Appendix C.4). The reference answer of the original dataset was shown only afterwards to select the annotated target intent among the enumerated interpretations; 3.1% of BuddyVQA and 4.6% of EgoBlind candidates for which the reference answer matched none of the enumerated interpretations were excluded from the ambiguous set and kept as ordinary questions. Two annotators labeled every instance independently (Cohen’s κ = 0.78 on BuddyVQA and 0.74 on EgoBlind for the ambiguity label); disagreements were adjudicated by a third annotator, and 86% of adjudicated resolved questions were judged equivalent to at least one annotator’s proposal. Because intent is assigned retrospectively, all results should be read as accuracy with respect to the annotated target intent rather than the original user’s intention.

Diagnostic Groups. The groups of Sec. 4.4 partition the 1,398 referentially underspecified questions; they are assigned in the following priority order so that every question belongs to exactly one group. G4 (201) if the dialogue history contains a prior mention that uniquely selects one interpretation; these questions are labeled well-specified and are used to measure unnecessary asking. Among the remaining 1,197 intent-ambiguous questions, G5 (131) if the annotated target interval has IoU $< 0 . 3$ with all eight GroundVQA proposals, a tool-level condition shared by all methods; otherwise G1 (412) if the competing referents are distinct entities whose supporting intervals overlap, G2 (356) if the competing interpretations share the same entity track at disjoint intervals, and G3 (298) if the competing referents are distinct entities of the same category at disjoint intervals. The paired-intent set contains 240 videos for which two enumerated interpretations both have a valid reference answer in the original dataset (from different questions on the same video) and the two reference answers are judged mutually exclusive by the benchmark judge; we present the same ambiguous question twice with the respondent instructed with each target in turn, with the model, history, and random seed reset between the two presentations.

## C.3 CLEARVQA AND CONTEXTCLARIFY

ClearVQA (Jian et al., 2025) contains 4,025 test questions over single images in three ambigu ity categories (referential 1,938, intent under-specification 1,095, spelling 992), an unambiguous contrast set for ambiguity detection, and a VQA-score protocol with and without clarification; we report VQA score, ambiguity-detection F1 on its contrast set, and its published LLaVA-1.5-13B and Ask-LLaVA-13B (SFT+DPO) numbers. ContextClarify (Cao et al., 2026) contains 275 ambiguous image–question pairs with human-verified clarifications and, for each, two contrast variants: a context-completed variant in which the appended context resolves the ambiguity (gold decision AN SWER) and an irrelevant-context variant in which plausible but non-resolving context is appended, so that the item remains ambiguous (gold decision CLARIFY). Its official test split has 42 ambiguou items and 84 variants (42 of each kind). We report end-to-end accuracy on the 42 ambiguous items, the CLARIFY-vs. -ANSWER F1 of the decision over all 126 test items (84 CLARIFY positives: the originals and the irrelevant-context variants), the unnecessary-ask rate on the 42 context-completed variants, and the published prompting and GRPO-CR numbers. For both benchmarks we treat the image as a one-frame prefix, disable temporal proposals, keep Grounding DINO for spatial candidates, and use each benchmark’s official metrics and respondent protocol (ClearVQA’s clarificationresponse simulator and ContextClarify’s oracle context provider). No skill or adapter is trained on either benchmark.

## C.4 EVALUATION PROTOCOL AND METRIC DEFINITIONS

This subsection documents the protocol behind every number in the paper: what the assistant receives, what the simulated user knows, what is revealed after a prediction and to whom, when the assistant asks, answers, gives alternatives, or abstains, and which examples enter each metric.

What Does the Assistant Receive? Before producing any output for a test question, a method receives exactly four things: the causal video prefix $V _ { \leq s _ { q } }$ decoded at 1 fps up to the question timestamp, the question $q ,$ the preceding dialogue of the same episode (earlier questions, the method’s own earlier answers, and earlier clarification exchanges), and the interaction budget $( T _ { \mathrm { m a x } } = 2$ $L _ { \operatorname* { m a x } } = 2 4 , M _ { \operatorname* { m a x } } = 4 )$ . Future frames, the gold ambiguity label, the annotated target intent $z ^ { \star }$ , its resolved question, the supporting interval and box $e ^ { \star }$ , the diagnostic-group label, and the reference answer are never inputs at prediction time, for any method and in any regime. On ClearVQA and ContextClarify the prefix is the single image.

What Does the Simulated User Know? Two respondent roles are kept strictly separate (prompts in Appendix F). The evaluation simulator (Qwen3-32B; Llama-3.3-70B in Appendix E.11) answers the clarification questions that are actually asked during testing and learning rollouts. It receives the original question, the clarification history, the selected question with its options, the hidden target intent $z ^ { \star }$ as a resolved question together with a textual description of its evidence (object, place, and time of the intended encounter), and a respondent profile calibrated on the user study. It never receives the reference answer, the system posterior, candidate scores, the system’s preferred option, or the video itself, and its rules forbid revealing more than the clarification asks. The predictive respondent used internally by CCA to estimate response likelihoods is the assistant’s own Qwen3.5- 9B backbone prompted once per candidate $z _ { i }$ “as a user whose intent is $z _ { i } \stackrel { \triangledown } { : }$ ; it never sees $z ^ { \star }$ . Human

Table 5: Population and success criterion of every reported metric on Augmented BuddyVQA (EgoBlind-Intent in parentheses). Cand.: candidate-bearing inputs; asked: inputs with at least one question.
<table><tr><td>Metric</td><td>Averaged over</td><td>Success / definition</td></tr><tr><td>Deictic)</td><td>Ans. Acc. (Overall; type; Chain, all 6,240 (2,036) inputs; that type or subset</td><td>judge verdict w.r.t. target intent; CONDITIONAL/ABSTAIN wrong</td></tr><tr><td>Ambiguous / well-specified acc.</td><td> $1 , 1 9 7 / 5 , 0 4 3 ( 4 1 8 / - )$ </td><td> $\mathrm { O v e r a l l } = ( 1 , 1 9 7 \mathrm { A m b } + 5 , 0 4 3 \mathrm { W e l l } ) / 6 , 2 4 0$ </td></tr><tr><td>T-mIoU, T-R@.5, Ans. (Tab. 7)</td><td>4,655 (1,447) grounding-eligible</td><td>mean T-IoU; T-IoU ≥ 0.5; judge verdict</td></tr><tr><td>S-mIoU, S-R@.5, Acc@GQA</td><td>3,914 (1,203) box-annotated</td><td>S-IoU on annotated frame; S-IoU ≥ 0.5; correct and T-,  $\mathrm { S } { \mathrm { - I o U } } \geq 0 . 5$ </td></tr><tr><td>Amb-F1; Ask-P / Ask-R</td><td>all inputs (1,197 ambiguous positives) 5,043 well-specified (Fig. 3); 201 G4</td><td>F1 of the ambiguity flag; precision / recall of asking</td></tr><tr><td>Unnecessary-ask rate</td><td>(Sec. 4.4)</td><td>at least one question asked</td></tr><tr><td>Clar.-Acc. (= Clar. Succ.); Intent-Acc.</td><td>asked ambiguous, per method or variant (612 asked by all five); all 1,197</td><td>posterior mode, selected by  $\mathrm { E F A } , \mathrm { i s } \ z ^ { \star } ;$  answer addresses</td></tr><tr><td>Ask rate; Turns (all / cand. / asked)</td><td>all / cand. / asked inputs</td><td>asked at least once;  $N _ { \mathrm { a s k } } \left( 0 \mathrm { i f \ n o n e , } \pm T _ { \mathrm { m a x } } \right)$ </td></tr><tr><td>Q tokens; human time; GPU time, latency</td><td></td><td>questions; study responses (else regression); tokens / question; s / response; s / input, respondent excluded</td></tr><tr><td>Cost (asked / all); Utility</td><td>all inputs asked / all inputs; cand. inputs</td><td> $\begin{array} { r l } { ~ } & { { } \sum _ { t } C _ { x } \left( c _ { t } \right) ( \cdots ^ { } ) \operatorname* { i f } } \end{array}$  never asked);</td></tr><tr><td>Abstention; paired-intent; probe all inputs; 240 pairs; 1,000 held-out</td><td></td><td> $\begin{array} { r } { \overline { { U } } _ { \mathrm { a n s } } ^ { \iota } + \mathrm { 0 . 5 } \dot { U } _ { \mathrm { g r d } } - \sum _ { t } C _ { x } \big ( \dot { c } _ { t } \big ) } \end{array}$  CONDITIONAL/ABSTAIN; both correct; verdict at</td></tr></table>

participants in the user study see only the video prefix, the original question, the target intent description they are asked to hold, and the clarification question. On ClearVQA and ContextClarify we use the benchmarks’ official respondents (ClearVQA’s clarification-response simulator and Con textClarify’s oracle context provider).

What Is Revealed after a Prediction, and Who Learns from It? After a method has logged its final output for interaction k, the evaluator computes the judge verdict and grounding IoUs. Under the prequential protocol, the following verified feedback is then made available: $z ^ { \star }$ (resolved question and evidence $e ^ { \star } )$ and the binary judge verdict $U _ { \mathrm { a n s } }$ . The reference answer string itself is not revealed, so no method can memorize answers. Only adapting conditions consume the feed back: ReferClaw (Skills) and ReferClaw (Full) with their learning-loop ablations and the online no-ask run, the Reflexion-style memory, and the MetaClaw-style variant. Feedback for interaction k may affect interactions k+1 onward only; frozen methods (all baselines in the frozen block, Refer-Claw (Base), and both asking-disabled controls) receive the feedback stream but ignore it.

When Does the Assistant Ask, Answer, Give Alternatives, or Abstain? The decision follows the controller transitions of Tab. 4 (Appendix B.4): a method answers directly when the gate is closed or one candidate dominates (T2), asks only through T3, returns a CONDITIONAL output (one separately verified answer per top-two interpretation) when no further question is admissible but the two interpretations still disagree (T4), and ABSTAINs when no supported candidate or verified answer remains. CONDITIONAL and ABSTAIN outputs are scored as incorrect in every accuracy column and reported separately as the abstention rate (Appendix E.13).

Answer Correctness and Grounding. Answer accuracy uses each benchmark’s multi-reference LLM-as-a-judge protocol; a stratified sample of 500 judgments was audited manually (97.2% agreement). On ambiguous questions the judge compares the answer with the reference answer of the annotated target intent, so an answer that is correct for a different interpretation counts as wrong. Temporal IoU is computed between the predicted interval and the annotated minimal interval. Spatial ground truth is a single box on the annotated evidence frame (Appendix C.2); spatial IoU is the IoU between that box and the predicted track’s box on the same frame, and it is 0 when the predicted interval does not contain that frame. A question is grounding-eligible when it has an annotated interval, and box-annotated when it additionally has an annotated box $( r ^ { \star } \neq \emptyset )$ . Temporal-only questions (no box, $e . g .$ , “Did I lock the door?”) are excluded from all spatial metrics and from Acc@GQA rather than scored as $\mathrm { S - I o U = 0 ; }$ in Eq. 1 they contribute $U _ { \mathrm { g r d } } = \mathrm { t I o U }$

Populations and Definitions. Tab. 5 lists, for every reported metric, the fixed set of examples it is averaged over and what counts as a success. The 1,197 intent-ambiguous and 5,043 well-specified questions (the latter including the 201 history-resolved G4 questions) partition the 6,240 test inputs. Candidate-bearing inputs are the 2,995 questions (48.0%; 1,160 intent-ambiguous, 1,835 well-specified) for which the frozen Base IGA returns at least two candidates. This stratum is used only for evaluation, so that turns, Utility, and the entropy-bin analysis compare methods on the same inputs; every method constructs its own candidates at test time, and questions asked outside the stratum count toward the ask rate over all inputs. The 1,197 intent-ambiguous questions are the positives of Ask-P and Ask-R; the 201 G4 questions count toward the unnecessary-ask rate instead.

Runs and Aggregation. Every number in a table comes from the same three runs (three fixed stream orders, identical seeds across methods), and the same checkpoint is used for all subsets of a row, so subset and overall values always refer to the same predictions; reported means are means of per-run values. GPU time includes candidate generation, response sampling, informationgain scoring, verification, and skill retrieval. Standard deviations and confidence intervals are in Appendix E.3.

## D ADDITIONAL EXPERIMENTAL DETAILS

## D.1 BASELINES AND FAIR COMPARISON

We compare against complementary methods that isolate direct video answering, evidence retrieval, and active disambiguation. The direct-answer MLLM is Qwen3.5-9B (Qwen Team, 2026). Grounded and streaming VideoQA baselines include ReKV (Di et al., 2025), RGA3 (Wang et al., 2025a), and MyBuddy (Qin et al., 2026); these respectively test streaming memory retrieval, objectcentric spatio-temporal referring, and ego-deictic reasoning with dialogue history. For joint answering and grounding (Tabs. 2 and 7), we further compare GeLM (Chen et al., 2025b), Open-o3- Video (Meng et al., 2026), GroundFormer (Seo et al., 2026), and EG-Reasoner (Huang et al., 2026), which return temporal or spatio-temporal evidence with each answer. Interactive clarification baselines include the ask-capable ClearVQA model (Jian et al., 2025), Plug-and-Play Clarifier (Yang et al., 2026), and Clarify-or-Answer with GRPO-CR (Cao et al., 2026). Framework-level baselines that permit backbone substitution use Qwen3.5-9B, whereas architecture-specific methods use their official checkpoints. Every method receives the identical decoded frame stream, causal video prefix, and maximum visual-token budget, and the same access to the dialogue history; fixed-frame architectures uniformly subsample from that same prefix. The history a method sees at question k of an episode consists of the earlier questions of that episode together with the method’s own earlier answers and clarification exchanges, so histories are identical across methods in access rights and in the user-side content but not in the assistant-side content; controls and ablations that must be compared at the same state are instead run as paired replays on the logged entering state of the reference run (below). Gold ambiguity labels, resolved questions, spatial boxes, and future frames are hidden at prediction time. All interactive methods share the same respondent, maximum number of turns, and token and option budgets. We use released code and default prompts where available, changing only the input adapter required to accept a video clip instead of a single image. The asking-disabled controls run IGA and EFA with the verifier but never invoke CCA; they are paired replays of the reference run (same adapter and skill version at each stream position, same entering history and seed), differing only in that the current question is never asked (Appendix E.5).

## D.2 LEARNING-FROM-INTERACTION CONDITIONS

Following MetaClaw’s dynamic online evaluation (Xia et al., 2026), we evaluate all adapting systems under identical inference and interaction budgets. For every ordered stream, interaction k is first answered and scored by the current system, verified feedback (the annotated target intent, its evidence, and the judge’s correctness verdict) is revealed only afterward, and any resulting skill or parameter update can influence interactions k + 1 onward but never the current or preceding examples. The model, skill library, and optimization state are initialized identically at the start of each stream order and then carried forward without resets inside that stream. ReferClaw (Base) runs the frozen pipeline. ReferClaw (Skills) additionally enables fast adaptation: after each verified failure, the skill evolver synthesizes a short skill, and the three most relevant skills are retrieved by cosine similarity and inserted into the responsible agent’s prompt. ReferClaw (Full) combines skill adaptation with process-reward-guided policy optimization; every verified trajectory (successes and failures) enters the query buffer together with the skill version under which it was produced; skills take effect as soon as they are synthesized, so a buffer of 128 trajectories spans several skill generations. When the buffer is full and the server is not processing an online request, each failed step is re-executed with its own logged prompt, including the skills that were injected at the time, and only the LoRA adapter attributed to that step is updated with GRPO (Shao et al., 2024); the adapter version is constant within one consolidation window. The Reflexion-style memory baseline stores a free-text reflection for each verified failure and retrieves the three most similar reflections into every agent prompt. The MetaClaw-style baseline uses MetaClaw’s generic skill schema (task, insight, action) without candidate topology or decisive contrast, and updates all three adapters with GRPO on the final-answer reward plus the same cost penalty; it receives the same update budget as ReferClaw (Full).

## D.3 OPTIMIZATION AND HYPERPARAMETERS

Table 6: ReferClaw implementation and hyperparameters.
<table><tr><td></td><td></td></tr><tr><td>Grounding</td><td>1 fps causal stream; retrieved clips at 4 fps, 448 × 448; 8 temporal proposals; detection/text thresholds  $0 . 2 5 / 0 . 2 0 ( 0 . 1 5 / 0 . ^ { \cdot }$  12 for the single re-grounding pass)  $| { \mathcal { C } } _ { t } | \leq 6$ </td></tr><tr><td>Candidates</td><td> $N _ { \mathrm { m a x } } = 4$  after merging;  $\kappa = 0 . 5 ; \epsilon _ { 0 } = 0 . 2 ;$  drop threshold 0.02; question pool (3 multiple-choice, 3 open-ended)</td></tr><tr><td>Thresholds</td><td> $\tau _ { d } = 0 . 6 0 , \tau _ { u } = 0 . 2 5 , \tau _ { a } = 0 . 3 0 , \tau _ { p } = 0 . 8 0 , \tau _ { g } = 0 . 1 5$ </td></tr><tr><td>Budget</td><td> $T _ { \mathrm { m a x } } = 2 , L _ { \mathrm { m a x } } = 2 4$  displayed tokens,  $M _ { \mathrm { m a x } } = 4$  options incl. other/none (one-shot stress test:  ${ \cal L } _ { \mathrm { m a x } } = 6 4 , { \cal M } _ { \mathrm { m a x } } = 8 ) ;$  generation cap 32 tokens per candidate question (longer candidates discarded) and 128 answer tokens</td></tr><tr><td>Respondent</td><td>8 samples per  $( c , z _ { i } )$  , smoothing  $\delta \ = \ 0 . 0 5 ,$  free text mapped to classes by BGE-M3;  $\lambda _ { g } = \mathsf { \bar { 0 . 5 0 , } } \lambda _ { C } = 0 . 3 5 ; ( w _ { E } , \bar { w _ { T } } , w _ { F } ) = ( 0 . 4 5 , 0 . 2 5 , 0 . \dot { 3 } \dot { 0 } )$  human, (0.55, 0.15, 0.30) simulator; components normalized to [0, 1] by validation 5th/95th percentiles</td></tr><tr><td>Learning</td><td>PRM weights  $( \alpha , \beta , \lambda , \mu ) = ( 0 . 5 0 , 0 . 5 0 , 0 . 3 5 , 1 . 0 0 ) ;$  one failure triggers skill synthesis  $( \le ~ 6$  failures per call), BGE-M3 top-3 retrieval at cosine  $\geq 0 . 4 5 , \stackrel { \because } { \leq } 3$  LoRA rank  $1 6 , \alpha _ { \mathrm { L o R A } } =$  32, dropout 0.05; query buffer 128;  $G \ = \ 8 ;$  Base; clip  $0 . 2 ; \mathrm { l r } 5 \times 1 0 ^ { - 6 } .$  , batch 64, one epoch per window; promotion probe 500+500</td></tr><tr><td>Decoding</td><td>validation examples (target type  $+ 1 . 0 , \mathrm { p r o b e } \overset { \cdot } { \geq } - \hat { 0 } . 5$  , unnecessary-ask and cost IGA/CCA temperature 0.2, top-p = 0.9; EFA/verifier temperature 0</td></tr></table>

Videos are processed causally at 1 fps; the top retrieved intervals are resampled at 4 fps and $4 4 8 \times 4 4 8$ resolution for fine grounding. The three agent adapters and the PRM are trained in bfloat16 on eight NVIDIA H100 80GB GPUs and served with vLLM (Kwon et al., 2023). AdamW uses $\beta _ { 1 } =$ $0 . 9 , \ \beta _ { 2 } \ = \ 0 . 9 5$ , weight decay 0.01, gradient clipping at 1.0, a 3% linear warm-up, and cosine decay. Hyperparameters are selected once on the Augmented BuddyVQA validation split and then fixed for all test benchmarks. Cost weights $( w _ { E } , w _ { T } , w _ { F } )$ and $\lambda _ { C }$ are chosen by grid search over $\{ 0 . 1 , \ldots , 0 . 6 \} ^ { 3 }$ (normalized to sum to one) and $\{ 0 , 0 . 1 , 0 . 2 , 0 . 3 5 , 0 . 5 , 0 . 8 \}$ to maximize validation utility (Eq. 1); the human and simulator cost weights are fitted separately and differ mainly in $w _ { T }$ (0.25 for humans, 0.15 for the simulator). The complete configuration is in Tab. 6.

## E ADDITIONAL RESULTS

## E.1 GROUNDED ANSWERING RESULTS

Table 7: Grounded VQA (%) on grounding-eligible questions of Augmented BuddyVQA and zeroshot Augmented EgoBlind-Intent. T/S: temporal/spatial. T-mIoU, T-R@.5, and Ans. are computed on all grounding-eligible questions, S-mIoU, S-R@.5, and Acc@GQA on the box-annotated subset (populations in Appendix C.4). Temporal-only baselines share our frozen Grounding DINO 1.5/SAM 2 spatial head. Superscripts on Ans. and Acc@GQA: change over the preceding row.
<table><tr><td rowspan="2">Method</td><td colspan="5">Augmented BuddyVQA</td><td colspan="7">Augmented EgoBlind-Intent (zero-shot)</td></tr><tr><td>T-mIoU</td><td></td><td>T-R@.5 S-mIoU S-R@.5</td><td></td><td> $\operatorname { A n s } .$ </td><td>GQA</td><td> $\scriptstyle \mathrm { T - m I o U }$ </td><td>T-R@.5 S-mIoU S-R@.5</td><td></td><td></td><td></td><td>Ans. GQA</td></tr><tr><td>GeLM (Chen et al., 2025b)</td><td>46.1</td><td>44.4</td><td>33.7</td><td>32.0</td><td>56.2</td><td>25.3</td><td>41.3</td><td>39.1</td><td>30.5</td><td>28.2</td><td>50.2</td><td>20.1</td></tr><tr><td>RGA3 (Wang et al., 2025a)</td><td>44.8</td><td>42.7</td><td>41.9</td><td>40.7</td><td>58.0</td><td>28.9</td><td>40.1</td><td>37.9</td><td>38.4</td><td>36.5</td><td>52.1</td><td>23.7</td></tr><tr><td>Open-o3-Video (Meng et al., 2026)</td><td>49.3</td><td>47.2</td><td>45.5</td><td>43.9</td><td>60.1</td><td>33.4</td><td>44.7</td><td>42.4</td><td>41.2</td><td>39.5</td><td>54.0</td><td>27.0</td></tr><tr><td>GroundFormer (Seo et al., 2026)</td><td>52.7</td><td>51.1</td><td>38.6</td><td>37.0</td><td>59.4</td><td>31.7</td><td>47.2</td><td>45.0</td><td>34.8</td><td>32.4</td><td>53.6</td><td>25.4</td></tr><tr><td>EG-Reasoner (Huang et al., 2026)</td><td>51.6</td><td>49.9</td><td>43.2</td><td>41.7</td><td>61.0</td><td>35.0</td><td>46.5</td><td>44.1</td><td>39.8</td><td>37.7</td><td>54.8</td><td>27.6</td></tr><tr><td>ReferClaw (Base, frozen)</td><td>54.9</td><td>53.3</td><td>46.8</td><td>45.1</td><td>64.9↑3.9</td><td>36.2↑1.2</td><td>47.6</td><td>45.9</td><td>40.4</td><td>38.6</td><td> $5 6 . 9 ^ { \uparrow 2 . 1 }$ </td><td>28.4↑0.8</td></tr><tr><td>ReferClaw (Skills)</td><td>56.6</td><td>55.1</td><td>48.5</td><td>47.0</td><td> $6 8 . 6 ^ { \dagger 3 . 7 }$ </td><td>38.912.7</td><td>48.9</td><td>47.2</td><td>41.6</td><td>39.8</td><td> $5 8 . 8 ^ { \uparrow 1 . 9 }$ </td><td> $3 0 . 1 ^ { \uparrow 1 . 7 }$ </td></tr><tr><td>ReferClaw (Full)</td><td>58.3</td><td>57.0</td><td>50.3</td><td>48.7</td><td> $7 2 . 9 ^ { \uparrow 4 . 3 }$ </td><td> $4 1 . 8 ^ { \uparrow 2 . 9 }$ </td><td>50.1</td><td>48.4</td><td>42.6</td><td>40.9</td><td> $ { \bf 6 0 . 7 ^ { \uparrow 1 . 9 } }$ </td><td> $3 1 . 6 ^ { \uparrow 1 . 5 }$ </td></tr></table>

## E.2 PER-CATEGORY TRANSFER BREAKDOWN

On Augmented EgoBlind-Intent, ReferClaw (Full) answer accuracy over all 2,036 instances (Tab. 2 uses the 1,447 grounding-eligible ones) is 66.8% on instances without a blur or off-center flag (n = 1,291) and 54.8% on flagged instances (n = 745), against 61.3% and 43.6% for EG-Reasoner; Acc@GQA on the box-annotated subset is 35.9% vs. 24.7% for the two groups. Clarification metrics transfer more completely than grounding: Ask-P and Ask-R are 82.6% and 79.4% on the 418 ambiguous instances (85.8% and 83.1% in-domain), and clarification success is 78.0% (81.4% in-domain).

## E.3 CONFIDENCE INTERVALS

Standard deviations over the three stream orders are at most 0.6 points for every accuracy in Tabs. 1 and 7. 95% paired-bootstrap confidence intervals resampled at the video level (10,000 resamples) are [73.2, 75.3] for ReferClaw (Full), [69.0, 71.1] for ReferClaw (Skills), [65.1, 67.4] for Refer-Claw (Base), [57.3, 59.7] for Clarify-or-Answer, [58.0, 60.4] for the MyBuddy reproduction, and [50.2, 52.6] for Direct Qwen3.5-9B (answer accuracy), and [40.3, 43.4], [37.4, 40.5], [34.7, 37.7], and [33.5, 36.5] for the Acc@GQA of ReferClaw (Full), (Skills), (Base), and EG-Reasoner. Differences between ReferClaw (Full) and every baseline, and between ReferClaw (Base) and every frozen baseline, exclude zero. For ContextClarify, the 95% interval of the enabled-minus-disabled difference on 42 items is [+2.4, +21.4] points.

## E.4 WHICH DESIGN HELPS WHICH CASE

![](images/1800c9eb794d66149a629cd13dc62ed2eefb235543232530090428dae062e8d6.jpg)  
Figure 8: Which core design helps which case. Rows are the four core designs of ReferClaw; columns are the case types defined by the annotations (diagnostic groups G1–G5, well-specified questions, questions that require state or action reasoning after the referent is resolved, and failure types that recur along the interaction stream). Cell shading gives the qualitative level of help, and cell notes quote the result that supports it.

Fig. 8 summarizes, per core design, the cases in which it makes a difference; the pattern is complementary rather than uniform. Grounded candidate construction is decisive where the competing readings are separated in time: because each candidate is anchored in its own interval and track, the two encounters of one object (G2) or two similar objects seen at different moments (G3) become distinct hypotheses that a question can name, which is where the advantage over the closest clarification baseline is largest (26.4 and 23.5 points against 13.1 when the referents are visible simultaneously, Fig. 7); removing the candidate set costs 11.3 Acc@GQA points and dropping the spatial tracks another 4.7 (Tab. 3). The same design also carries most of the improvement on well-specified questions, where a single grounded candidate delivers selected evidence without any question (12.9 points over direct answering with asking disabled, Fig. 3), whereas it cannot help when the referent lies outside the proposals (G5), which remains the weakest group and whose headroom the coverage oracle bounds at 2.6 points (Tab. 8).

Cost-aware contrastive clarification determines whether and what to ask. Its gate is what keeps the system silent when the history already resolves the reference (G4, asked on 6.5% of inputs against 24.4% for Clarify-or-Answer, with 11.4 points higher accuracy) and on well-specified questions (unnecessary-ask rate around 4%), and its information-gain scoring concentrates questions where the candidate belief is uncertain (asked on 82.0% of high-entropy inputs but 5.8% of low-entropy ones, Fig. 6). What is asked matters as much as when: generic questions reduce clarification success from 81.4% to 64.9% (Tab. 9), and a heuristic that always asks about the top-two contrast recovers most of the accuracy only at higher cost (0.65 against 0.46 turns per candidate-bearing input, Tab. 8). Clarification cannot repair a coverage failure, so on G5 it asks without benefit.

Evidence-fused answering acts after the intent is resolved and is therefore most visible on questions whose answer requires reasoning about state or action once the referent is known: Advisory and Scrutinization questions gain 8.5 and 8.9 points over Clarify-or-Answer (Tab. 1), replacing the selected clip by the global video costs 6.6 Acc@GQA points, and the paired-intent test, on which the same question must be answered differently for two intents, rises from 38.8% to 71.3% because the answer is produced from the selected evidence rather than from the most salient encounter. The verifier contributes a smaller but consistent 1.6 points by rejecting answers unsupported by the card.

Learning from interaction is the only design whose effect grows along the stream: recurrence of verified failure types falls from 34.2% to 10.7% and backward transfer is positive (4.1 points, Tab. 13), so its help concentrates on failure types that repeat. Its per-group effect is largest again on repeated encounters, driven by the action-contrast skills that account for 23% of the learned CCA skills (Appendix B.6), and it improves the Advisory and Scrutinization types by 8.7 and 7.9 points from Base to Full, whereas G5 improves only modestly and G4 mostly through fewer unnecessary questions (7.0% to 6.5%). Taken together, the four designs address different case types, which is why removing any one of them degrades a different column of Fig. 8 rather than uniformly lowering accuracy.

## E.5 CONTROLS THAT ISOLATE THE VALUE OF CLARIFICATION

Table 8: Controls that isolate the value of clarification (Augmented BuddyVQA, all test inputs). Turns: per candidate-bearing input; Cost: per asked input $( ^ { 6 6 } - ^ { 3 }$ when never asked). Top block: policies sharing the learned skills and adapters of ReferClaw (Full); middle: the frozen Base pair and the online no-ask run; bottom: oracles using gold annotations at prediction time.
<table><tr><td>Variant</td><td>Ans. Acc.</td><td>Acc@GQA</td><td>Turns</td><td>Cost ↓</td><td>What it measures</td></tr><tr><td>ReferClaw (Full)</td><td>74.3</td><td>41.8</td><td>0.46</td><td>0.42</td><td></td></tr><tr><td>ReferClaw (Full), asking disabled (state-matched replay)</td><td>66.9</td><td>39.9</td><td>0.00</td><td></td><td>immediate effect of asking, learned states</td></tr><tr><td>Heuristic policy (always ask top-2 contrast when  $g _ { t } \dot { = } \dot { 1 ) }$ </td><td>71.0</td><td>39.6</td><td>0.65</td><td>0.52</td><td>value of IG—cost scoring</td></tr><tr><td>Single agent, same structured steps, one adapter</td><td>71.6</td><td>39.9</td><td>0.50</td><td>0.44</td><td>value of role separation</td></tr><tr><td>ReferClaw (Base)</td><td>66.2</td><td>36.2</td><td>0.47</td><td>0.45</td><td></td></tr><tr><td>ReferClaw (Base), asking disabled (state-matched replay)</td><td>60.5</td><td>34.8</td><td>0.00</td><td></td><td>immediate effect of asking, frozen system</td></tr><tr><td>Online no-ask run (Base + both loops, CCA disabled throughout)</td><td>64.8</td><td>38.3</td><td>0.00</td><td></td><td>learning without ever asking</td></tr><tr><td>Oracle: gold intent inserted into candidate set</td><td>76.9</td><td>44.6</td><td>0.41</td><td>0.40</td><td>candidate coverage ceiling</td></tr><tr><td>Oracle: gold resolved question as input</td><td>79.8</td><td>47.3</td><td>0.00</td><td></td><td>cost of intent resolution</td></tr></table>

Both state-matched replays keep IGA, EFA, and the verifier but never invoke CCA: each question is re-run with the checkpoint, skill version, entering history, and seed that the reference run had at that stream position, so reference and replay differ only on inputs with $n _ { 0 } \geq 2$ . Because the replayed states of ReferClaw (Full) were themselves learned from earlier clarifications, the 7.4-point difference (74.3 vs. 66.9) is the immediate effect of asking at matched states; the frozen pair gives 5.7 points. The online no-ask run instead measures learning without any clarification: an independent prequential run from Base with both loops but CCA disabled reaches 64.8%, 4.3 points above the frozen asking-disabled pipeline and 9.5 below ReferClaw (Full), the latter being the total contribution of clarification including its effect on what is learned. The heuristic policy asks about the top-two candidates whenever the gate of Eq. 3 opens and therefore asks on every gate-open input (52.1% of candidate-bearing inputs against 38.7%); the single-agent control executes the same structured steps as one prompt sequence on one adapter. The oracle rows provide diagnostic headroom: inserting the gold intent adds 2.6 accuracy points and supplying the gold resolved question adds 5.5. These interventions do not provide an additive causal decomposition of the remaining errors.

## E.6 FINE-GRAINED ABLATION

Table 9: Fine-grained ablation on Augmented BuddyVQA; columns as in Tab. 3. Within each block only the named component is changed.
<table><tr><td>Variant</td><td>Ans. Acc.</td><td>Acc@GQA</td><td>Clar. Succ.</td><td>Ask (%)</td><td>Cost (asked) ↓</td><td> $\overline { { U } } _ { \mathrm { a n s } }$ </td><td> $\overline { { U } } _ { \mathrm { g r d } }$ </td><td>ΣC↓</td><td>Utility ↑</td></tr><tr><td>ReferClaw (Full)</td><td>74.3</td><td>41.8</td><td>81.4</td><td>38.7</td><td>0.42</td><td>0.732</td><td>0.328</td><td>0.163</td><td>0.733</td></tr><tr><td>Cost-aware clarification</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>— ambiguity gate</td><td>72.5</td><td>40.4</td><td>73.6</td><td>70.0</td><td>0.69</td><td>0.710</td><td>0.320</td><td>0.483</td><td>0.387</td></tr><tr><td> expected information gain (confidence-only trigger)</td><td>71.9</td><td>40.0</td><td>70.1</td><td>42.0</td><td>0.47</td><td>0.708</td><td>0.316</td><td>0.197</td><td>0.669</td></tr><tr><td>− cost penalty (λc = 0)</td><td>73.8</td><td>41.4</td><td>82.0</td><td>52.1</td><td>0.76</td><td>0.732</td><td>0.326</td><td>0.396</td><td>0.499</td></tr><tr><td>One-shot super-question (relaxed length budget)</td><td>71.3</td><td>39.4</td><td>68.5</td><td>52.1</td><td>0.61</td><td>0.708</td><td>0.313</td><td>0.318</td><td>0.547</td></tr><tr><td>Generic rather than contrastive clarification</td><td>71.1</td><td>38.8</td><td>64.9</td><td>40.0</td><td>0.45</td><td>0.702</td><td>0.312</td><td>0.180</td><td>0.678</td></tr><tr><td>Learning from interaction: fast loop</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>− fast loop</td><td>71.2</td><td>38.6</td><td>77.2</td><td>40.0</td><td>0.43</td><td>0.704</td><td>0.307</td><td>0.172</td><td>0.685</td></tr><tr><td>Random skill retrieval</td><td>72.0</td><td>39.4</td><td>78.0</td><td>40.0</td><td>0.44</td><td>0.715</td><td>0.315</td><td>0.176</td><td>0.696</td></tr><tr><td>Top-1 rather than top-3 retrieval</td><td>73.3</td><td>40.7</td><td>80.2</td><td>39.0</td><td>0.43</td><td>0.727</td><td>0.321</td><td>0.168</td><td>0.719</td></tr><tr><td>Generic rather than intent-specific skill schema</td><td>72.5</td><td>39.9</td><td>78.8</td><td>39.0</td><td>0.43</td><td>0.715</td><td>0.318</td><td>0.168</td><td>0.706</td></tr><tr><td>— support-query skill-generation separation</td><td>72.2</td><td>39.5</td><td>78.5</td><td>39.0</td><td>0.43</td><td>0.709</td><td>0.318</td><td>0.168</td><td>0.700</td></tr><tr><td>Learning from interaction: slow loop</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>— slow loop (Skills only)</td><td>70.1</td><td>38.9</td><td>79.4</td><td>40.0</td><td>0.43</td><td>0.691</td><td>0.312</td><td>0.172</td><td>0.675</td></tr><tr><td>Final-answer-only reward</td><td>72.6</td><td>39.4</td><td>79.1</td><td>43.0</td><td>0.47</td><td>0.716</td><td>0.310</td><td>0.202</td><td>0.669</td></tr><tr><td>一 - grounding reward</td><td>73.1</td><td>37.6</td><td>80.3</td><td>39.0</td><td>0.43</td><td>0.722</td><td>0.303</td><td>0.168</td><td>0.706</td></tr><tr><td>— target-progress reward ∆t</td><td>72.9</td><td>40.7</td><td>76.8</td><td>45.0</td><td>0.49 0.44</td><td>0.716</td><td>0.313</td><td>0.221 0.180</td><td>0.651</td></tr><tr><td>Undirected entropy reduction ∆Ht instead of ∆t</td><td>73.4</td><td>40.9</td><td>78.9</td><td>41.0</td><td></td><td>0.721</td><td>0.324</td><td></td><td>0.703</td></tr><tr><td>Update all agent adapters</td><td>72.3</td><td>39.4</td><td>78.3</td><td>42.0</td><td>0.46</td><td>0.713</td><td>0.312</td><td>0.193</td><td>0.676</td></tr></table>

Cost-Aware Clarification. Removing the ambiguity gate makes the system ask on 70.0% of candidate-bearing inputs instead of 38.7%, raises the cost per asked input to 0.69, and reduces accuracy by 1.8 points because well-specified inputs are needlessly perturbed (lowest Utility, 0.387). Removing the cost penalty preserves most of the accuracy (73.8) but asks on every gate-open input (52.1%, the same fraction as the one-shot policy and the no-stopping variant of Tab. 3) at nearly twice the cost per asked input (0.76). The one-shot super-question, run with a relaxed length budget $( L _ { \mathrm { m a x } } = 6 4 , M _ { \mathrm { m a x } } = 8 ) _ { \mathrm { \Omega } }$ , loses 3.0 points despite a Cost of 0.61, and generic questions reduce clarification success from 81.4% to 64.9%.

Learning Loops. Disabling the fast loop costs 3.1 accuracy points; random retrieval and a generic skill schema trail the full model by 2.3 and 1.8 points, so the gain comes from targeted intent skills rather than a longer prompt, and removing the support–query separation costs 2.1 points. The slow loop adds 4.2 points over Skills only: removing the grounding reward causes the largest Acc@GQA decrease (41.8 to 37.6), removing the target-progress term $\Delta _ { t } ^ { \star }$ mainly reduces clarification success (81.4% to 76.8%), replacing it by an undirected entropy reduction costs 0.9 points and 2.5 points of clarification success, and updating every adapter is 2.0 points below agent-attributed updates.

## E.7 CLARIFICATION QUALITY

Table 10: Clarification quality on Augmented BuddyVQA (%). Amb-F1: F1 of the ambiguity flag; Ask-P/R: precision and recall of asking, with the 1,197 intent-ambiguous questions as positives; Clar.-Acc.: target intent selected after clarification, over the questions each method asked (asked) or over the 612 asked by all five methods (common); Intent-Acc.: final answer addresses the target intent, over all intent-ambiguous questions.
<table><tr><td>Method</td><td>Amb-F1</td><td>Ask-P</td><td>Ask-R</td><td>Clar.-Acc. (asked)</td><td>Clar.-Acc. (common)</td><td>Intent-Acc.</td></tr><tr><td>ClearVQA (Jian et al., 2025)</td><td>72.6</td><td>68.8</td><td>76.2</td><td>66.1</td><td>64.1</td><td>58.3</td></tr><tr><td>Plug-and-Play Clarifier (Yang et al., 2026)</td><td>75.4</td><td>72.0</td><td>76.8</td><td>69.5</td><td>67.8</td><td>61.0</td></tr><tr><td>Clarify-or-Answer (Cao et al., 2026)</td><td>78.1</td><td>76.3</td><td>79.6</td><td>73.7</td><td>71.9</td><td>64.9</td></tr><tr><td>ReferClaw (Base)</td><td>81.0</td><td>82.4</td><td>80.9</td><td>77.5</td><td>76.3</td><td>72.8</td></tr><tr><td>ReferClaw (Full)</td><td>84.3</td><td>85.8</td><td>83.1</td><td>81.4</td><td>80.2</td><td>77.6</td></tr></table>

In Tab. 10, ReferClaw improves Amb-F1 by 6.2 points and ask precision by 9.5 over Clarify-or-Answer, so its gain is not obtained by asking more often. Because methods ask on different subsets, we also report clarification accuracy on the 612 ambiguous examples asked by all five methods; the ordering is unchanged and the gap widens slightly (80.2 vs. 71.9). Intent accuracy over all

ambiguous examples, which penalizes both wrong resolutions and failures to ask, rises from 64.9% to 77.6%. Internally, the realized target progress $\bar { \mathrm { T P } } = \sum _ { t } \Delta _ { t } ^ { \star }$ (the target-progress term of Eq. 6, in nats, computed on the fixed candidate scope of Appendix B.7) averages 0.62 per asked question for ReferClaw (Full), and 92% of asked questions with $\mathrm { T P } > 0 . 5$ end with the target intent selected, confirming that belief moves toward the intended interpretation rather than toward a confidently wrong one.

## E.8 EXAMPLE TRACES

Qualitative Comparison. Fig. 9 follows an episode in which the user takes two capsules from a bottle on the kitchen counter and, later, one tablet from a pill box at the coffee corner, and then asks “How many did I take?” and “Where did I leave $\mathrm { i t ? } ^ { \flat }$ . Direct answering merges the two encounters, and grounded answering localizes its evidence in the capsule encounter, an error that carries over to the follow-up. Clarify-or-Answer obtains the intended referent with a generic question, but because the reply is not tied to an encounter in the video it still answers from the more salient one, and it asks again although the dialogue already resolves $\mathbf { \ddot { n } } ( \mathbf { \dot { t } } ^ { \prime } )$ . ReferClaw grounds both encounters before asking (D1), so a single contrastive question separates them (D2), guided by a skill learned earlier in the stream that contrasts repeated intakes by container (D4); it then answers from the selected clip under verification (D3) and answers the follow-up without a question because the resolved referent leaves a single candidate.

Successful Trace. Video prefix 1:50–2:30 (kitchen, then study). Question at 2:30: “Is it still on?” $\mathrm { I G A } \colon d _ { 0 } \ = \ 0 . 8 1$ ; candidates z (stove, 1:50–2:07, kitchen) with $p \ = \ 0 . 4 1$ and z<sub>2</sub> (desk lamp, $2 { : } 2 0 { - } 2 { : } 3 0$ , study) with $p = 0 . 5 2 , p ( z _ { \mathscr { O } } ) = 0 . 0 7 ;$ provisional answers differ $( \bar { a } _ { 1 } = \mathrm { { n o } , } \bar { a } _ { 2 } = \mathrm { { y e s } ; }$ $\mathrm { D i s } = 0 . 9 2 ) , g _ { 0 } = 1$ . CCA: pool of five questions; selected $c _ { 0 } = { } ^ { 6 } \mathrm { T }$ he stove or the desk lamp?” (options: stove, desk lamp, other/none), $\mathrm { I G } _ { 0 } = 0 . 8 8$ bits, $C _ { \mathrm { L } } = 0 . 2 8$ , net utility 0.78. Response: “The lamp.” Posterior: $p _ { 1 } ( z _ { 2 } ) = 0 . 9 4$ . Controller: max $p _ { 1 } \geq \tau _ { p } , \mathrm { A N S W E R }$ . EFA on V[2:20, 2:30] with the lamp track: “Yes, the desk lamp is still on” (video); verifier pass. Correct.

Failure Trace. Same episode; question at 2:32: “Did I unplug $\mathrm { i t ? } ^ { \flat }$ after a dialogue turn about the toaster. IGA: candidates $z _ { 1 }$ (toaster, 2:03) p = 0.60 and z<sub>2</sub> (phone charger, 2:26) $p = 0 . 3 2 ;$ $d _ { 0 } = 0 . 7 2 , \operatorname { D i s } = 0 . 8 5 , g _ { 0 } = 1 . \operatorname { C C A }$ selected “The toaster or the phone charge $\hat { \mathbf { \Gamma } } ^ { \dagger } \vec { \mathbf { \Gamma } } ^ { \dagger }$ Response: “The toaster.” EFA answered “Yes, you unplugged it” from a frame at 2:03 in which the hand occludes the socket; the verifier passed because the claim was labeled video and the card described “hand at the toaster plug”. Annotated answer: no. Attribution: EFA (target covered and grounded; answer unsupported). This trace belongs to the occlusion category of the failure audit and illustrates the verifier’s limitation as a prompted consistency check.

![](images/70deb27f1231741a162aaf922db6ed17505352022db5817cb2d8343398a610a7.jpg)  
Figure 9: Qualitative comparison on a two-question episode. z<sub>1</sub>, z<sub>2</sub>: grounded candidates (dashed boxes: their tracks); tags mark the core design behind each ReferClaw step (D1–D4 as in Fig. 8); ✓/×: correct or wrong for the target intent $z _ { 2 }$

## E.9 MODEL DYNAMICS AND FAILURE MODES

The Value of Interaction Is Concentrated in Uncertain Cases. Fig. 6 bins the 2,995 candidatebearing test questions by the initial entropy $H ( p _ { 0 } )$ of the frozen Base candidate set; the low-, medium-, and high-entropy bins contain 1,288, 1,168, and 539 questions (43.0%, 39.0%, 18.0%). To attribute the gain to asking rather than to learned grounding, each bin compares ReferClaw (Full) with its paired asking-disabled replay (Tab. 8) and with direct answering. ReferClaw asks on only 5.8% of the low-entropy bin, where asking adds 2.0 points (72.5 vs. 70.5), whereas it asks on 55.0% and 82.0% of the medium- and high-entropy bins, where asking adds 24.9 and 26.9 points (75.9 vs. 51.0 and 68.8 vs. 41.9); the gains over direct answering are 14.5, 29.9, and 29.9 points. Because the replay fixes the entering history, the two branches differ only on these 2,995 inputs, over which asking adds 15.4 points, corresponding to the 7.4-point Overall difference of Tab. 8. This monotonic pattern explains why adaptive stopping dominates a uniform interaction depth: most inputs need little assistance, but the difficult tail benefits strongly from a well-chosen question.

Residual Failures Point to Perception Rather than to the Clarification Loop. A manual audit of 500 incorrect ReferClaw (Full) predictions attributes 31% to severe occlusion or motion blur, 27% to multiple visually near-identical referents, 23% to information absent from the observed prefix, 12% to long-range identity drift, and 7% to invalid or noisy user responses. The first and fourth categories are grounding failures, and the third cannot be repaired by a better question. Future work should combine stronger persistent tracking with requests for new visual evidence (e.g., asking the user to look toward the shelf), which our formulation can express as an action with cost but which we have not evaluated.

## E.10 EFFICIENCY AND RESPONDENT COST

Tab. 11 gives the operating points of Fig. 4, whose horizontal axis is the cost per input over all inputs. The fixed two-turn and one-shot policies ask on every gate-open input (25.0% of all inputs), so their turn counts bound what a gate-triggered policy can spend. ReferClaw (Full) reaches the highest accuracy while asking on the fewest inputs among the interactive methods, with the fewest questions per asked input and the shortest questions, at a GPU time (5.0 s) within the range of the other interactive systems. Because frozen baselines and the adapting ReferClaw (Full) are different regimes, the figure also marks the frozen ReferClaw (Base), which dominates every frozen baseline on both axes (66.2% at 0.085 against 58.5% at 0.110 for Clarify-or-Answer).

Table 11: Accuracy–cost trade-off on Augmented BuddyVQA. Ask rate: all inputs / candidatebearing inputs; Outside: inputs outside the candidate-bearing stratum on which the method asked; Turns: per input over all / candidate-bearing / asked inputs; Tok./q.: tokens per question; Human time: seconds per response, measured in the user study or predicted by the calibrated regression (‡); GPU time: seconds per input; Cost: per asked input / per input. The one-shot super-question uses a relaxed length budget.
<table><tr><td>Method</td><td>Acc.</td><td>Ask rate (%)</td><td>Outside</td><td>Turns (all / cand. / asked)</td><td>Tok./q.</td><td>Human time (s)</td><td>GPU time (s)</td><td>Cost (asked) ↓</td><td>Cost (all) ↓</td></tr><tr><td>Direct Qwen3.5-9B</td><td>51.4</td><td>0.0 / 0.0</td><td>0</td><td>0.00 / 0.00 /–</td><td>1</td><td></td><td>3.6</td><td></td><td>0.000</td></tr><tr><td>ClearVQA (Jian et al., 2025)</td><td>55.7</td><td>21.2 / 43.1</td><td>35</td><td>0.28 / 0.57 / 1.32</td><td>23.3</td><td>6.0</td><td>5.0</td><td>0.59</td><td>0.125</td></tr><tr><td>Plug-and-Play Clarifier (Yang et al., 2026)</td><td>57.4</td><td>20.4 / 41.7</td><td>28</td><td>0.28 / 0.58 / 1.38</td><td>22.2</td><td>5.9</td><td>5.4</td><td>0.63</td><td>0.129</td></tr><tr><td>Clarify-or-Answer (Cao et al., 2026)</td><td>58.5</td><td>20.0 / 41.0</td><td>20</td><td>0.26 / 0.53 / 1.28</td><td>24.0</td><td>6.9</td><td>5.1</td><td>0.55</td><td>0.110</td></tr><tr><td>Fixed two-turn clarification</td><td>68.8</td><td>25.0 / 52.1</td><td>0</td><td>0.50 / 1.04 / 2.00</td><td>18.4</td><td>4.9</td><td>7.8</td><td>0.86</td><td>0.215</td></tr><tr><td>One-shot super-question (relaxed budget)</td><td>71.3</td><td>25.0 / 52.1</td><td>0</td><td>0.25 / 0.52 / 1.00</td><td>47.5</td><td>11.1‡</td><td>5.8</td><td>0.61</td><td>0.153</td></tr><tr><td>ReferClaw (Base)</td><td>66.2</td><td>18.8 / 39.2</td><td>0</td><td>0.23 / 0.47 / 1.21</td><td>21.7</td><td>5.3</td><td>4.9</td><td>0.45</td><td>0.085</td></tr><tr><td>ReferClaw (Full)</td><td>74.3</td><td>18.6 / 38.7</td><td>0</td><td>0.22 / 0.46 / 1.19</td><td>20.6</td><td>5.1</td><td>5.0</td><td>0.42</td><td>0.078</td></tr></table>

## E.11 HUMAN RESPONDENTS

Table 12: Human-respondent evaluation on the 150 held-out examples of the user study (10 participants). Confusing: fraction of questions rated unanswerable or confusing. Sim. agreement: fraction of interactions in which the simulator’s response falls in the same class as the human’s. The last two columns give simulator accuracy on the same examples.
<table><tr><td>Method</td><td>Intent resolved (%)</td><td>Ans. Acc. (%)</td><td>Resp. time (s)</td><td>Confusing (%)</td><td>Sim. agreement (%)</td><td>Acc. Qwen3-32B</td><td>Acc. Llama-3.3-70B</td></tr><tr><td>Clarify-or-Answer</td><td>70.0</td><td>57.3</td><td>6.9</td><td>12.7</td><td>85.9</td><td>58.1</td><td>57.6</td></tr><tr><td>ReferClaw (Full)</td><td>80.0</td><td>72.0</td><td>5.1</td><td>4.3</td><td>88.6</td><td>73.7</td><td>73.4</td></tr></table>

Tab. 12 makes the human study visible on the held-out half, which was not used to fit the response model or the response-time regression. With human respondents, ReferClaw resolves the target intent on 80.0% of examples and answers 72.0% correctly, against 70.0% and 57.3% for Clarify-or-Answer, while requiring 1.8 fewer seconds per response and producing a third as many questions rated unanswerable or confusing. Simulator accuracy is 1.7 points higher than human accuracy for ReferClaw and 0.8 for the baseline, so the simulator is slightly optimistic for both, and responseclass agreement between simulator and humans is 86–89%. Replacing the evaluation simulator with Llama-3.3-70B, which was not used for any design decision, changes accuracy by at most 0.5 points for either method, suggesting that the question-selection policy is not tuned to the particular simulator used during development.

## E.12 LEARNING DYNAMICS

Table 13: Learning dynamics on Augmented BuddyVQA. Q1–Q4: prequential accuracy on stream quartiles; Probe: accuracy on a fixed held-out probe of 1,000 examples after the stream; Recurrence: verified errors followed by the same failure type within 200 questions; BWT: accuracy change on first-quartile examples re-evaluated after the stream; Update: GPU-hours per 1,000 interactions.
<table><tr><td>Condition</td><td>Q1</td><td>Q2</td><td>Q3</td><td>Q4</td><td>Probe</td><td>Recurrence ↓</td><td>BWT↑</td><td>Update GPU-h ↓</td></tr><tr><td>ReferClaw (Base)</td><td>66.0</td><td>66.1</td><td>66.3</td><td>66.4</td><td>65.8</td><td>34.2</td><td>0.0</td><td>0.0</td></tr><tr><td>Base + Reflexion-style memory</td><td>66.0</td><td>67.2</td><td>67.9</td><td>68.5</td><td>66.7</td><td>27.8</td><td>+0.4</td><td>0.2</td></tr><tr><td>Base + MetaClaw-style adaptation</td><td>66.5</td><td>68.9</td><td>70.6</td><td>72.1</td><td>68.6</td><td>21.3</td><td>+1.1</td><td>3.1</td></tr><tr><td>ReferClaw (Skills)</td><td>66.2</td><td>69.3</td><td>71.5</td><td>73.3</td><td>70.0</td><td>18.5</td><td>+1.9</td><td>0.3</td></tr><tr><td>ReferClaw (Full)</td><td>66.5</td><td>72.6</td><td>77.0</td><td>81.2</td><td>75.2</td><td>10.7</td><td>+4.1</td><td>2.8</td></tr></table>

The quartile columns show prequential accuracy along the stream, while the probe column evaluates the final checkpoint on an unchanged held-out set so that the gain cannot reflect changing example difficulty. Skills improve the probe at 0.3 GPU-hours per 1,000 interactions because they take effect without weight updates; the slow loop’s 2.8 GPU-hours are incurred asynchronously between online requests.

Fig. 5 plots the probe accuracy of every checkpoint (after each 10% of the stream, mean over the three orders; ReferClaw (Base) stays at 65.8%). ReferClaw (Full) improves at every checkpoint, with the largest increments in the first half of the stream, where most first-of-type failures occur and the skill library grows fastest; the Skills-only curve flattens after 60% of the stream, when new failures mostly repeat already-stored topologies, whereas the slow loop keeps consolidating gains until the end.

```jsonl
[QUESTION]
<original user question>
[DIALOGUE_HISTORY]
[ {"question": "<earlier user question>",
"answer": "<assistant answer>"}, ... ]
[CLARIFICATION_HISTORY]
[ {"turn": 0, "question": "<c_0>", "response": "<y_0>"}, ... ]
[ACTIVE_SKILLS]
<zero to three skills retrieved for the current agent>
Evidence Card Template.
{
"evidence_id": "e1",
"interval_sec": [<start>, <end>],
"track_id": "track_1", // null when non-localizable
"boxes": {"<frame>": [x1, y1, x2, y2], ...},
"representative_frames": [<frame ids>],
"visual_description": "<literal visible content>",
"grounding_score": <0.0--1.0>
}
```

## E.13 COVERAGE

ReferClaw (Full) returns an ANSWER on 96.1% of Augmented BuddyVQA test inputs, a CONDI-TIONAL output on 2.3%, and ABSTAIN on 1.6%; accuracy on the answered population is 77.3%, and 61% of CONDITIONAL outputs contain the correct answer paired with the target intent (counted as incorrect in all tables). Clarify-or-Answer answers 98.9% of inputs with 59.1% accuracy on the answered population; Direct Qwen3.5-9B answers 100%. Reporting accuracy over all inputs, as in the main tables, therefore does not favor ReferClaw through selective abstention.

## F PROMPTS AND TEMPLATES

This section documents the prompts and structured interfaces used by ReferClaw. Each template specifies the model role, admissible inputs, decision constraints, and machine-readable output. Angle-bracketed strings denote runtime values. The same templates are used for human- and simulator-based evaluation; only the respondent model and cost calibration differ. GroundVQA, Grounding DINO 1.5 Pro, and SAM 2 produce temporal intervals, boxes, and tracks before prompting, so no LLM is allowed to invent spatial coordinates or unsupported evidence. Score normalization, Bayesian updates, expected information gain, cost aggregation, and controller actions are computed deterministically from the structured outputs.

## F.1 SHARED INTERACTION AND EVIDENCE TEMPLATES

Every agent receives the original question, the dialogue history, the clarification history, retrieved active skills, and only the video evidence required by its role.

## Interaction Context Template.

The visual description is restricted to observable content within the interval and track. It must not identify the intended candidate, infer the user’s preference, or reveal a reference answer.

Rules:   
1. A deictic word alone does not imply ambiguity.   
2. Mark ambiguity only if at least two visually supported   
interpretations remain plausible and can change the answer.   
3. Create at most N\_MAX candidates. Each candidate must contain   
a fully specified question and exactly one supplied evidence ID.   
4. Preserve the original answer type. Do not reveal or insert the   
answer into a resolved question.   
5. Use only supplied evidence cards. Never invent timestamps,   
regions, tracks, people, objects, actions, or user preferences.   
6. Merge candidates that refer to the same entity and event.   
7. If no card supports the question, return no\_supported\_candidate.   
8. Return JSON only. Do not include chain-of-thought.

## F.2 INTENTION GROUNDING AGENT

## IGA System Prompt.

You are the Intention Grounding Agent for an egocentric video assistant. Determine whether the user’s intended question is uniquely specified by the question, history, and supplied evidence.

```jsonl
Inputs:
QUESTION: <question>
DIALOGUE_HISTORY: <dialogue_json>
CLARIFICATION_HISTORY: <history_json>
EVIDENCE_CARDS: <evidence_cards_json>
N_MAX: 4
Output schema:
{
"ambiguity_score": <0.0--1.0>,
"ambiguous_span": "<text or null>",
"status": "unambiguous|ambiguous|no_supported_candidate",
"candidates": [
{
"candidate_id": "z1",
"resolved_question": "<fully specified question>",
"evidence_id": "e1",
"provisional_answer": "<short answer or unknown>",
"support_score": <0.0--1.0>
}
]
}
```

Support scores are normalized outside the prompt with Eq. 2; answer disagreement is computed deterministically from the provisional answers.

## F.3 CLARIFICATION-CONDITIONED IGA UPDATE

After a user response, IGA narrows the existing candidate scope and requests targeted re-grounding only when the response selects other/none or introduces a constraint not represented by the retained cards.

## IGA Update Prompt.

You are updating an existing set of grounded intent candidates after one clarification response. Do not restart from the full video.

Rules:

1. Interpret the response only with respect to the asked question.   
2. Retain candidates consistent with the response and reject only   
candidates contradicted by it.   
3. Do not create a new candidate unless the response selects   
other/none or supplies a missing observable constraint.   
4. Request targeted re-grounding only for that new constraint.   
5. Do not infer facts not stated by the user or visible in evidence.   
6. Return JSON only.   
Inputs:   
ORIGINAL\_QUESTION: <question>   
CANDIDATES: <candidates\_with\_posterior\_json>   
EVIDENCE\_CARDS: <evidence\_cards\_json>   
CLARIFICATION\_QUESTION: <c\_t>   
USER\_RESPONSE: <y\_t>   
Output schema:   
{   
"retained\_candidate\_ids": ["z1", "..."],   
"response\_class": "<partition cell|other\_none|uncertain|invalid>",   
"response\_constraints": ["<atomic constraint>", "..."],   
"needs\_targeted\_regrounding": <true|false>,   
"regrounding\_query": "<query or null>"   
}

2. Every contrast must be supported by the supplied evidence cards.

The response likelihood and the Bayesian update in Eq. 4 remain external to this prompt.

## F.4 COST-AWARE CLARIFICATION AGENT

CCA proposes a small question pool; Eq. 5 scores every valid proposal by IG<sub>t</sub>(c) − λ<sub>C</sub>C<sub>x</sub>(c).

## CCA Question-Generation Prompt.

You are the Cost-Aware Clarification Agent. Generate short questions   
that distinguish the currently plausible grounded user intentions.

1. Ask about exactly one observable contrast: identity, occurrence,   
location, or action.

4. Always include "Other/none".

5. Do not ask for the task answer, mention posterior scores, combine   
multiple attributes, repeat history, or ask a super-question.

6. Respect L\_MAX tokens and M\_MAX options.

7. Generate at most six candidates: up to three multiple-choice and   
three open-ended questions.

8. Return JSON only. Do not rank by fluency or provide reasoning.

Inputs:   
QUESTION: <question>   
HISTORY: <history\_json>   
CANDIDATES: <candidate\_id, resolved\_question, posterior>   
EVIDENCE\_CARDS: <evidence\_cards\_json>   
L\_MAX: 24   
M\_MAX: 4   
Output schema:   
{

Inputs:   
ORIGINAL\_QUESTION: <question>   
CLARIFICATION\_HISTORY: <history\_json>   
CLARIFICATION\_QUESTION: <selected question and options>   
HIDDEN\_INTENT: <z\_star without reference answer>   
RESPONDENT\_PROFILE: <human-calibrated or LLM-calibrated profile>

```jsonl
"questions": [
{
"question_id": "c1",
"type": "multiple_choice|open_ended",
"text": "<clarification question>",
"options": ["<option>", "Other/none"],
"contrast_type": "identity|occurrence|location|action",
"candidate_partition": [["z1"], ["z2"]],
"supporting_evidence_ids": ["e1", "e2"]
}
]
}
```

## F.5 PREDICTIVE RESPONDENT, EVALUATION SIMULATOR, AND HUMAN INTERFACE

The predictive respondent is prompted once per candidate z<sub>i</sub> to estimate $p _ { x } ( y \mid c , z _ { i } , h _ { t } )$ for CCA, and the evaluation simulator answers from the hidden target intent; what each role and each human participant receives, and never receives, is specified in Appendix C.4.

## Predictive Respondent Prompt (CCA-Internal).

Answer the assistant’s clarification question as a user whose   
intended question is ASSUMED\_INTENT. Reveal only what the   
clarification asks. Give the shortest sufficient response, or   
"uncertain" if ASSUMED\_INTENT does not determine the answer.   
Return JSON only: {"response": "<text>", "option": "<option|null>"}.   
Inputs:   
ORIGINAL\_QUESTION: <question>   
CLARIFICATION\_HISTORY: <history\_json>   
CLARIFICATION\_QUESTION: <question and options>   
ASSUMED\_INTENT: <resolved\_question and evidence description of z\_i>

## Evaluation Simulator System Prompt.

You simulate the user in an egocentric assistance interaction.   
Answer the assistant’s clarification from the hidden user intention.

"response\_status": "answered|uncertain|invalid|no\_response"   
}

## F.6 EVIDENCE-FUSED ANSWERING AGENT

## EFA System Prompt.

You are the Evidence-Fused Answering Agent. Answer the resolved user intention using only the selected video evidence and explicit user clarifications.

## Rules:

1. Answer the resolved question, not the ambiguous surface wording.   
2. Treat USER\_FACTS and VIDEO\_FACTS as different sources.

3. Support every answer claim with the selected evidence card or an   
explicit user response, and label its source.

Inputs:   
ORIGINAL\_QUESTION: <question>   
RESOLVED\_QUESTION: <q\_tilde>   
SELECTED\_CANDIDATE: <candidate\_id and posterior>   
INTENT\_STATUS: "resolved|answer\_determined\_intent\_unresolved"   
VIDEO\_FACTS: <selected evidence card(s); frames carry the track box>   
USER\_FACTS: <clarification history>

```jsonl
Output schema:
{
"answer": "<answer>",
"resolved_question": "<q_tilde>",
"evidence": {"evidence_id": "e1", "interval_sec": [<s>, <e>],
"track_id": "track_1"},
"confidence": <posterior of selected candidate>,
"claim_sources": [
{"claim": "<atomic claim>", "source": "video|user"} ]
}
```

Conditional alternatives and abstention are decided by the controller (Tab. 4), not by EFA; each branch of a CONDITIONAL output is produced by a separate EFA call on its candidate and passes the verifier (ANSWERPATH).

## F.7 ANSWER–INTENT–EVIDENCE VERIFIER

## Verifier Prompt.

Verify whether the proposed answer addresses the resolved intention and whether each claim is supported by its declared source.

1. Check answer--intent consistency and answer--evidence consistency.   
2. A "video" claim must be visible in the supplied evidence card;   
a "user" claim must appear in USER\_FACTS.

3. Do not repair an answer by inventing evidence or changing intent. 4. Choose revise only when unsupported text can be removed without changing the supported answer; otherwise choose abstain.

## 5. Return JSON only.

```jsonl
Inputs:
ORIGINAL_QUESTION: <question>
RESOLVED_QUESTION: <q_tilde>
EVIDENCE_CARD: <selected evidence card>
USER_FACTS: <clarification history>
ANSWER_PACKET: <EFA output JSON>
Output schema:
{
"decision": "pass|revise|abstain",
"intent_consistent": <true|false>,
"evidence_consistent": <true|false>,
"unsupported_claims": ["<claim>", "..."],
"revised_answer": "<answer or null>"
}
```

## F.8 CONTROLLER RECORD

The controller is non-generative; it applies Tab. 4 to agent outputs and logs the following record for each turn.

```csv
"turn": <t>,
"ambiguity_score": <d_t>,
"max_posterior": <max_i p_t(z_i)>,
"uncovered_mass": <p_t(z_null)>,
"clarification_gate": <0|1>,
"selected_question_id": "<c_t or null>",
"expected_information_gain_bits": <IG_t or null>,
"respondent_cost": <C_x or null>,
"net_utility": <IG_t - lambda_C C_x or null>,
"response_class": "<cell|other_none|uncertain|invalid|null>",
"action": "ASK|ANSWER|CONDITIONAL|ABSTAIN|REGROUND",
"reason_code": "<deterministic rule identifier>"
```

## F.9 PROCESS REVIEW AND CREDIT ASSIGNMENT

The process-review prompt is used only to attribute failures that the deterministic rules of Appendix B.5 do not cover, and to audit training annotations after verified feedback is available; it is never exposed to the online agents. Target-progress terms and respondent costs are supplied as computed values.

You are annotating a completed trajectory after verified feedback.   
Evaluate the disambiguation process, not only the answer.

Rules:   
1. Check whether the intended candidate was covered and grounded.   
2. Check whether the asked question separated the intended candidate   
from its strongest competitor.   
3. Use the supplied target-progress terms and costs without changing them.   
4. Flag missed ambiguity, unsupported candidates, repeated or   
multi-attribute questions, and incorrect stopping.   
5. Attribute the primary failure to IGA, CCA, or EFA. Use none when   
the trajectory is valid or the evidence is genuinely unavailable.   
6. Return JSON only and do not propose a new answer.

Inputs:   
TRAJECTORY: <trajectory\_json>   
VERIFIED\_INTENT: <z\_star>   
VERIFIED\_EVIDENCE: <e\_star>   
VERIFIED\_ANSWER: <correctness feedback>   
TARGET\_PROGRESS: <delta\_star\_by\_turn>   
RESPONDENT\_COSTS: <cost\_by\_turn>   
Output schema:   
{   
"invalid\_events": ["<event>", "..."],   
"responsible\_agent": "IGA|CCA|EFA|none",   
"failure\_type": "coverage|grounding|non\_separating|unnecessary\_ask|   
inconsistent\_answer|none",   
"short\_rationale": "<one sentence>"   
}

## F.10 TRAJECTORY AND SUPPORT–QUERY RECORDS

```jsonl
{
"trajectory_id": "<id>",
"skill_generation": <g>,
"video_id": "<id>",
"question": "<q>",
"dialogue": <dialogue_json>,
"history": <clarification_history_json>,
"candidate_states": <candidates_and_posteriors_by_turn>,
"evidence_cards": <evidence_cards_json>,
"controller_actions": <controller_records_json>,
"final_output": <EFA_packet_json or CONDITIONAL/ABSTAIN>,
"verified_feedback": <feedback_json>,
"process_components": {"r_final": .., "r_ground": ..,
"sum_dH_plus": .., "sum_cost": .., "r_invalid": ..},
"responsible_agent": "IGA|CCA|EFA|none",
"split": "support_g|query_g_plus_1"
}
```

Support and query records with the same failure event are never used in the same policy update.

## F.11 INTENT-SPECIFIC SKILL EVOLVER AND INJECTION

Skill Evolver Prompt.   
You are the skill evolver. Analyze verified failures and produce new   
skills that prevent the same intent-disambiguation error without   
increasing unnecessary clarification.   
[VERIFIED\_FAILURES]   
<up to six support trajectories with failed step,   
failure type, and feedback>   
[EXISTING\_SKILLS]   
<skill names, responsible agents, and triggers>   
Instructions:   
1. Generate one to MAX\_NEW\_SKILLS non-duplicate skills.   
2. Each skill must target exactly one agent: IGA, CCA, or EFA.   
3. Derive every visual pattern from observed evidence in the failures.   
4. Store the ambiguity trigger, candidate topology, decisive visual

```jsonl
contrast, corrected action, anti-pattern, and regression tests.
5. Do not store instance IDs, reference answers, posterior values,
or advice to ask more questions by default.
6. A regression test must cover one ambiguous case and one
well-specified case to protect the unnecessary-ask rate.
7. Return only a valid JSON array.
Output schema:
[
{
"name": "<lowercase-hyphenated-name>",
"responsible_agent": "IGA|CCA|EFA",
"trigger": "<deictic or elliptical failure condition>",
"candidate_topology":
"<e.g., same object, two encounters, same room, gap<60s>",
"decisive_visual_contrast": "<identity/time/location/action>",
"corrected_action": "<short executable instruction>",
"anti_pattern": "<behavior to avoid>",
"retrieval_key": "<trigger + topology + agent>",
"regression_tests": [
{"case": "ambiguous", "expected": "<behavior>"},
{"case": "well_specified", "expected": "<behavior>"}
]
}
]
```

Skill Injection Block. At inference time, BGE-M3 retrieves at most three skills for the responsible agent; only the trigger, topology, corrected action, and anti-pattern are inserted, and regression tests remain in the store.

[ACTIVE\_INTENT\_SKILLS]   
SKILL: <name> AGENT: <IGA|CCA|EFA>   
TRIGGER: <when this skill applies>   
CANDIDATE\_TOPOLOGY: <grounded configuration>   
DECISIVE\_CONTRAST: <identity|time|location|action>   
ACTION: <corrected agent behavior>   
ANTI\_PATTERN: <behavior to avoid>   
Apply a skill only when its trigger and topology match the current   
input. The system prompt and interaction limits always take priority.

## F.12 AGENT-ATTRIBUTED POLICY-OPTIMIZATION RECORD

The slow loop samples alternative outputs only from the adapter attributed by the failure attribution;   
other adapters and the visual backbone remain frozen. Each GRPO item uses the following record.

```python
"trajectory_id": "<id>",
"skill_generation": <g_plus_1>,
"responsible_agent": "IGA|CCA|EFA",
"agent_input": <exact structured input at failed step>,
"sampled_agent_outputs": [<G structured outputs>],
"predicted_process_rewards": [<R_proc from the reward model>],
"group_advantages": [<normalized advantages>],
"active_skill_ids": ["<skill>", "..."]
```