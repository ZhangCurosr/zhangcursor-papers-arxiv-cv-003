# Trusting Imagined Futures: Auditing Imagined Rewards for VLA Post-Training

Zicheng Kong<sup>1</sup>, Zhenbo Xu<sup>1,3</sup>, Haoyan Zhang<sup>1</sup>, Jinghan Yang<sup>1</sup>, Zhenzhen Li<sup>1</sup> Yiwei Ru<sup>2</sup>, Liuyu Xiang<sup>1</sup>, WanTing Xu<sup>4</sup>, Min Ren<sup>1</sup>, Zhaofeng He<sup>1</sup>

<sup>1</sup>Beijing University of Posts and Telecommunications, Beijing, P.R. China

<sup>2</sup>Institute of Automation, Chinese Academy of Sciences, Beijing, P.R. China

<sup>3</sup>ShiFang Technology Inc., Hangzhou, P.R. China <sup>4</sup>Wuhan Electric Power Technical College, Wuhan, P.R. China

## Abstract

Video world models enable candidate actions to be evaluated through imagined outcomes, but high predicted rewards can favor trajectories with poor execution quality. We introduce Trusting Imagined Futures (TIF), an offline supervision selection method for vision-languageaction (VLA) post-training that audits predicted rewards using execution feedback. A multidimensional reward model predicts a utility score for candidate ranking and five dimensions of execution quality. An auditor uses reward predictions and uncertainty to identify overestimation patterns learned from paired imagined videos and real executions. TIF accepts the candidate with the highest predicted utility when its advantage over the existing trajectory exceeds a margin and its audit risk passes a threshold; otherwise, it retains the existing supervision. The retained real trajectories then supervise behavior cloning. Across four LIBERO suites and three training seeds, TIF achieves a mean success rate of76.25%, improving over training on the existing trajectories by 3.00 percentage points with the same trajectory count and training budget. Adding audit gating yields 1.47 percentage points with the reward model and advantage margin heldfixed. The gains persist against advantage ranking and logistic regression risk selection at matched replacement counts. On recorded videosfrom three real robot tasks, TIF increases clean completion among selected trajectoriesfrom 58.33% to 70.00% across 60 candidatepools.

## 1. Introduction

Vision-language-action (VLA) models generate robot actions from visual observations and language instructions, providing pretrained policies that can be adapted to different tasks [1, 2, 9, 20]. When post-training relies on imitation learning, the quality of the supervision trajectories directly affects the behaviors the policy learns. Available trajectories vary in task progress and execution quality, and imitating them can cause the policy to reproduce undesirable behaviors. Selecting suitable supervision trajectories is therefore important for VLA post-training.

![](images/bb3458067e193e8800285424931745053ccd26f7a9a8ea8b77bf68540c3b5349.jpg)  
Figure 1. A candidate may receive a higher predicted utility than Source yet exhibit poorer execution quality. TIF audits evaluation reliability before accepting a replacement.

Video world models predict the visual consequences of candidate actions, providing imagined outcomes for behavior evaluation [22, 26]. A reward model scores these outcomes to compare candidates and support policy learning [23, 26]. When imagined outcomes are used to select supervision for VLA post-training, reward ranking identifies candidates with higher predicted utility. Accepting such a candidate as a replacement also requires assessing whether its predicted advantage over the existing trajectory is trustworthy.

An apparently successful grasp or task completion in an imagined video may correspond to an unstable grasp, a collision, or an incorrect object interaction during real execution. Errors in world model predictions and reward estimates can give such candidates high scores [5, 8]. When they replace existing trajectories as behavior cloning targets, these evaluation errors can cause the policy to learn undesirable behaviors. We study whether auditing rewards predicted from imagined outcomes improves supervision selection and subsequent policy learning (Fig. 1). We evaluate this question using fixed candidate pools with matched executions.

We propose Trusting Imagined Futures (TIF), which uses real execution feedback to train a risk auditor that complements reward scores when deciding whether to accept candidate trajectories as supervision. The reward model evaluates imagined videos along five dimensions: task progress, goal correctness, manipulation quality, safety, and execution stability. It jointly predicts a utility score for candidate ranking. The auditor learns from discrepancies between these predictions and the corresponding real execution outcomes to identify candidates with high predicted scores but poor execution quality. TIF first selects the candidate with the highest predicted utility. It accepts this candidate if its predicted advantage over the existing supervision trajectory (Source) exceeds a margin and its audit risk falls below a threshold; otherwise, it retains Source. We then update the VLA through behavior cloning using paired observations and actions from the retained real execution trajectories.

We construct 32,000 pairs of real executions and imagined videos across four LIBERO suites [14], using real execution outcomes to supervise the reward model and auditor. Candidate pools are collected offline, with recorded executions providing labels, evaluation outcomes, and behavior cloning targets. Selection uses predicted scores and diagnostic information from imagined videos to accept a replacement or retain Source. At deployment, only the updated VLA is required.

Across four LIBERO suites and three training seeds, TIF achieves a mean success rate of 76.25%, exceeding behavior cloning using only Source trajectories by 3.00 percentage points with the same number of training trajectories and behavior cloning budget. With the reward model and advantage margin held fixed, adding audit gating improves mean success by 1.47 percentage points. When the number of replacements is fixed, TIF still outperforms ranking by predicted advantage and selecting candidates by risk estimated through logistic regression.

Our main contributions are as follows:

• We propose TIF, a supervision auditing method that learns from matched real executions when rewards predicted from imagined videos overestimate execution quality. TIF combines this risk with a candidate’s predicted advantage to decide whether to accept a replacement or retain the existing supervision.

• We develop a multidimensional reward model for candidate ranking and risk auditing. The model jointly predicts five dimensions of execution quality and a utility score. Utility predictions rank candidates, while dimension predictions and their uncertainty provide information for auditing.

• We show that the auditor identifies discrepancies between high predicted rewards and poor execution outcomes. Comparisons with fixed replacement counts and downstream post-training experiments further show that using audit risk improves supervision selection and policy success.

## 2. Related Work

VLA Post-Training and World Models. OpenVLA-OFT improves supervised adaptation through changes to action decoding, representation, and learning objectives [10], while GRAPE aligns policies with preferences over successful and failed trajectories [24]. RIPT-VLA and VLA-RL use environment interaction for reinforcement learning, with sparse success rewards and process rewards, respectively [15, 19]. WMPO reduces reliance on real environment interaction by optimizing policies through imagined rollouts from a video world model [26]. WoVR addresses accumulated prediction errors through keyframe initialization and joint updates of the policy and world model [8]. WoVR+ extends this approach to multiple views, proprioceptive states, and training within individual stages of longer tasks [25]. TIF uses imagined outcomes to evaluate fixed candidate actions and select real execution trajectories for behavior cloning.

Robot Reward Models and Candidate Evaluation. Robometer learns rewards through progress supervision within trajectories and preference comparisons across trajectories [13], while RoboReward augments supervision with negative and partial completion examples constructed through counterfactual relabeling and temporal clipping [11]. RynnValue learns temporal distance to task completion and converts its predictions into dense rewards [7]. During inference, V-GPS reranks actions using a value function learned through offline reinforcement learning [16], and RoVer uses process rewards to guide candidate expansion and selection [4]. RARM gates progress rewards according to confidence in matching execution clips to a reference demonstration [21]. TIF uses matched simulator executions to supervise predictions of execution quality from imagined videos, providing a utility score for ranking and five dimension predictions for auditing.

Reliability of Imagined Supervision. Recent methods use reliability estimates to control how imagined outcomes influence policy learning. WISE filters candidate groups using reward differences, ranking consistency, and reward variation across repeated imaginations, then updates the policy using relative advantages within each group [23]. HaWMPO learns video reliability from visual, geometric, and motion discrepancies against observed executions, together with image quality metrics, and adjusts reinforcement learning rewards through Reward-Soft [3]. MeRL uses visual prediction and reward proxy errors to regulate imagination length and data proportion, as well as sampling probabilities and training weights for individual chunks [12]. TIF learns the risk of reward overestimation from matched executions, targeting cases with high predicted utility, low executed utility, and a substantial gap between them. This risk and the predicted advantage over Source determine whether to accept the candidate with the highest predicted utility or retain the existing supervision.

![](images/93aa2be9670b55314d2f09620c86a279b91ef63389155eac5691b72c92b4f4f3.jpg)  
Figure 2. Overview of TIF. Matched executions supervise reward learning and reliability auditing. During curation, predicted utility and audit risk determine whether to replace Source. The selected real trajectories supervise VLA post-training across all valid steps.

## 3. Method

Overview. As shown in Fig. 2, TIF combines candidate utility prediction with reward risk auditing to select supervision trajectories for VLA post-training. We first train a multidimensional reward model using matched execution outcomes, then train an auditor on discrepancies between its predictions and those outcomes. During selection, the candidate with the highest predicted utility replaces Source only if its predicted advantage exceeds a margin and its audit risk falls below a threshold; otherwise, Source is retained. Observations and actions from all valid steps of the retained real execution trajectories are then used for behavior cloning.

## 3.1. Candidate Actions and Paired Outcomes

Given a visual observation $o _ { i } ,$ proprioceptive state $q _ { i } ,$ , and task instruction $\ell _ { i }$ at initial state i, we construct a candidate action set $\mathcal { A } _ { i } = \{ \mathbf { a } _ { i , k } \} _ { k = 0 } ^ { K - 1 }$ . Each sequence contains at most eight control commands executed at 20 Hz. The component dataset contains four controlled candidates per initial state, providing varied execution quality for training and evaluation. The natural pool used for supervision selection contains five candidates: Source, one greedy branch, and three sampled branches.

Source is a fixed action proposal associated with the initial state, taken from subsequent actions in a LIBERO demonstration or from saved policy action records. Sequences are extended when necessary according to the rules described in Appendix A.2. We execute Source in its associated scene and initial state. The remaining branches use the OpenVLA policy for the corresponding suite to generate actions step by step from the current execution observation, using greedy decoding or sampling. For every branch, we record observations, actions, and execution feedback until the sequence reaches its prescribed length or the environment terminates.

We collect candidates offline and keep their actions and execution records fixed when comparing selection rules. The world model generates an imagined video $\hat { \tau } _ { i , k }$ from the initial observation and recorded action sequence, with an adapter converting initial proprioception into the required conditioning. For candidates that terminate early, we use only the valid executed action prefix. The matched execution trajectory $\tau _ { i , k }$ provides training labels, outcomes for evaluating selection decisions, and paired observations and actions for behavior cloning. Selection uses predicted scores and diagnostic information from imagined videos.

## 3.2. Reward Supervision from Execution

Final task success does not fully capture local progress and execution quality within short action sequences. We therefore derive five continuous labels from each candidate’s execution: $\mathbf { r } _ { i , k } = ( P _ { i , k } , G _ { i , k } , M _ { i , k } , S _ { i , k } , E _ { i , k } ) \in [ 0 , 1 ] ^ { 5 }$ representing task progress, goal correctness, manipulation quality, a safety proxy, and execution stability.

Task progress measures advancement toward completion, while goal correctness measures alignment with the object and goal specified by the instruction. Manipulation quality measures the effectiveness of grasping, transport, and placement. The safety proxy captures risks associated with abnormal contact, excessive actions, and abrupt state changes. Execution stability measures motion smoothness, action reversals, and gripper switching. These labels are computed from execution feedback, states, contacts, and action statistics, with normalization parameters fitted on the training set. The continuous labels are used to construct reference utility and are discretized into low, medium, and high levels using thresholds of 0.34 and 0.67 for training the dimension predictions. Dimension definitions and discretization are summarized in Appendix B.1.

We construct reference utility using predefined weights for task progress, goal correctness, manipulation quality, and execution stability, together with a multiplicative safety factor:

$$
\begin{array} { c } { { v _ { i , k } = \left( 0 . 3 5 P _ { i , k } + 0 . 3 0 G _ { i , k } + 0 . 2 0 M _ { i , k } + 0 . 1 5 E _ { i , k } \right) } } \\ { { { } } } \\ { { { \cdot \left( 0 . 3 5 + 0 . 6 5 S _ { i , k } \right) . } } } \end{array}\tag{1}
$$

We cap $v _ { i , k }$ at 0.666 when neither task progress nor goal correctness reaches the high level, and at 0.339 when the safety label is low. The lower cap applies when both conditions hold, yielding the final reference utility $u _ { i , k }$ . This utility supervises the utility prediction head and defines preferences between candidates from the same initial state.

## 3.3. Multidimensional Reward Learning and Risk Auditing

We use the dimension labels and reference utility defined in Sec. 3.2 to train a reward model that predicts execution quality from imagined videos, task instructions, and candidate actions. We then train a risk auditor using discrepancies between predicted utility and the reference utility from matched executions. After reward training, its parameters remain fixed throughout auditor training and trajectory selection.

Multidimensional reward model. The reward model $R _ { \theta }$ takes the imagined video $\hat { \tau } _ { i , k } .$ , task instruction $\ell _ { i } ,$ and candidate action sequence $\mathbf { a } _ { i , k }$ as input. A DINOv2-L encoder with fixed parameters extracts video frame features, which are fused with instruction and action information. The model predicts distributions over the low, medium, and high levels of each reward dimension, along with a utility estimate $\hat { u } _ { i , k }$ from a separate prediction head. The utility estimate ranks candidates, while the dimension predictions provide information for risk auditing.

We jointly optimize dimension prediction, utility regression, and pairwise ranking:

$$
\mathcal { L } _ { R } = \mathcal { L } _ { \mathrm { d i m } } + 0 . 5 \mathcal { L } _ { \mathrm { u t i l i t y } } + 0 . 1 \mathcal { L } _ { \mathrm { p a i r } } .\tag{2}
$$

The three losses supervise the predicted dimension levels, agreement with execution reference utility, and the relative ordering of candidates from the same initial state, respectively.

Auditing reward overestimation. After training the reward model, we compute its predictions on a separate dataset for auditor training. We assign a positive audit label to candidates with high predicted utility, low execution reference utility, and a sufficiently large overestimation gap:

$$
y _ { i , k } = \mathbf { 1 } \bigg [ \hat { u } _ { i , k } \geq 0 . 5 8 \mathrm { ~ \land ~ } u _ { i , k } \leq 0 . 4 2 \bigg ] .\tag{3}
$$

The auditor $C _ { \psi }$ takes a feature vector $\mathbf { z } _ { i , k }$ containing predicted utility, distributions and scores for the five dimensions, uncertainty statistics, and basic diagnostics from imagined videos. These diagnostics describe video availability, image quality, and changes between frames; the video diagnostic fields are listed in Appendix C.3. The auditor outputs a risk score $\rho _ { i , k } = C _ { \psi } ( \mathbf { z } _ { i , k } )$ and is trained with $y _ { i , k }$ as its binary target. Matched execution outcomes provide the training labels, while reward predictions and imagined video diagnostics provide the inputs for estimating overestimation risk.

## 3.4. Reliability Gating and Policy Learning

TIF first selects the candidate with the highest predicted utility, then decides whether to accept it based on its predicted advantage over Source and its audit risk. For initial state i, we assign index 0 to Source and define the candidate with the highest predicted utility and its advantage over Source:

$$
k _ { i } ^ { * } = \arg \operatorname* { m a x } _ { k } \hat { u } _ { i , k } , \qquad \Delta _ { i } = \hat { u } _ { i , k _ { i } ^ { * } } - \hat { u } _ { i , 0 } .\tag{4}
$$

We accept the candidate when its predicted advantage exceeds 0.15 and its audit risk is at most 0.50; otherwise, we retain Source. The retained candidate is

$$
\begin{array} { r } { \tilde { k } _ { i } = \left\{ \begin{array} { l l } { k _ { i } ^ { * } , } & { \Delta _ { i } > 0 . 1 5 ~ \wedge ~ \rho _ { i , k _ { i } ^ { * } } \leq 0 . 5 0 , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{5}
$$

Each initial state contributes one training trajectory, provided by either the accepted candidate or Source.

Behavior cloning. We train the policy using the real execution trajectory corresponding to the selected candidate ${ \tilde { k } } _ { i } .$ . Its valid observations and actions are denoted by $\mathcal { T } _ { i } =$ $\{ ( o _ { i , t } , a _ { i , t } ) \} _ { t = 0 } ^ { L _ { i } - 1 }$ , where $o _ { i , t }$ is the observation before action $a _ { i , t }$ and $L _ { i }$ is the number of valid steps. During training, we sample a trajectory uniformly and then sample one of its valid time steps uniformly. For a selected dataset containing N trajectories, the behavior cloning objective is

$$
\mathcal { L } _ { \mathrm { B C } } = - \mathbb { E } \underset { t \sim \mathcal { U } ( \{ 1 , \dots , L _ { i } - 1 \} ) } { \underbrace { \left[ \log \pi _ { \phi } ( a _ { i , t } \mid o _ { i , t } , \ell _ { i } ) \right] } } \left[ \log \pi _ { \phi } ( a _ { i , t } \mid o _ { i , t } , \ell _ { i } ) \right] .\tag{6}
$$

For OpenVLA, we compute the loss over the corresponding action tokens and update only the LoRA parameters in the attention and MLP modules. At deployment, the updated VLA directly generates actions from observations and task instructions.

## 4. Experiments

## 4.1. Experimental Setup

Tasks and models. We evaluate TIF on 40 tasks across four LIBERO suites: Spatial, Object, Goal, and Long [14]. We use OpenVLA as the policy [9], Cosmos-Predict2 2B to generate imagined videos [17], and DINOv2-L to extract visual features for the reward model [18].

Data. The dataset for reward model and auditor training and evaluation contains 32,000 pairs of execution trajectories and imagined videos from 8,000 initial states. The natural candidate dataset supports offline selection evaluation and policy training, with 1,600 initial states split into 800/400/400 for training, validation, and testing. Offline evaluation uses the 384 test states in which the task is not already successful at initialization. Data splits are grouped by source trajectory, with reward model and auditor training data separated from natural validation and test data by trajectory provenance.

Baselines. We compare against OpenVLA before posttraining and policies trained by behavior cloning on Source, greedy branches, or randomly selected candidates. Rewardonly selects the candidate with the highest predicted utility. Reward+Margin additionally requires its predicted advantage over Source to exceed the margin, and TIF further applies the audit risk gate. At matched replacement counts, we also compare random selection, predicted advantage ranking, logistic regression risk, prediction entropy, and consistency scores. All post-training comparisons use the same candidate pool, policy initialization, and training budget, retaining one real execution trajectory per initial state.

Training and evaluation. We adapt the policy through LoRA [6] and behavior cloning using 200 selected training trajectories per suite, for 800 trajectories in total. Training uses all valid time steps by default; supervision from only the first action is evaluated as an ablation. Each suite uses 500 updates and an effective batch size of 16 observation– action pairs. We train each method with three seeds and evaluate each resulting policy over 50 trials per task, totaling 2,000 episodes per seed. Evaluation initial states and environment seeds are identical across methods. We report policy success as the mean and sample standard deviation across the three training seeds. Data splits, policy training settings, and seed-wise results for matched replacement comparisons are provided in Appendices A.1, C.4, and F.1, respectively.

## 4.2. Multidimensional Reward Evaluation

We first examine whether the reward model predicts actual execution quality from imagined videos and whether dimensional supervision improves candidate ranking. We evaluate 1,920 candidates from 384 test states against labels and reference utilities derived from matched executions. Macro-F1 ranges from 0.732 to 0.758 across the five dimensions; results for individual dimensions and ordinal errors are provided in Appendix D.1.

We compare Scalar, which predicts utility alone, with 2D, which adds supervision for task progress and goal correctness, and 5D, which uses all five dimensions. The three base models share the same reference utility and pairwise supervision. Each model selects candidates by predicted utility without an advantage margin or audit gate. As shown in Tab. 1, 5D achieves the highest pairwise accuracy, Top-1 accuracy, and mean execution reference utility of selected candidates, at 57.79%, 33.07%, and 0.4603, respectively. We further ablate pairwise supervision and separately shuffle the manipulation, safety, and stability labels. Shuffling these labels reduces Top-1 accuracy to 29.17% and selected utility to 0.4550, indicating that their correspondence with execution outcomes contributes to candidate ranking.

We further examine the information provided by imagined videos. Using the same candidate set, we separately train reward models with the initial image, instruction, and actions, either alone or supplemented with imagined or recorded execution videos. Candidates are selected by predicted utility without a margin or audit gate. Adding imagined videos improves Top-1 accuracy from 29.95% to 33.07% and the mean execution utility of selected candidates from 0.4527 to 0.4603, showing that predicted visual changes provide additional information for candidate evaluation. Recorded execution videos yield a higher Top-1 accuracy of 50.52%, showing the gap between evaluation from predicted and observed outcomes. Full input comparisons are provided in Appendix D.3. We next evaluate whether auditing reward overestimation improves supervision selection based on imagined videos.

Table 1. Candidate ranking with different reward supervision. Pairwise and Top-1 accuracies are reported in %. Top-1 measures whether the selected candidate has maximal execution reference utility within its candidate set. Utility is the mean execution reference utility of selected candidates. The final two rows independently ablate 5D.
<table><tr><td>Model</td><td>Pairwise ↑</td><td>Top-1 ↑</td><td>Utility ↑</td></tr><tr><td>Scalar</td><td>56.88</td><td>30.99</td><td>.4574</td></tr><tr><td>2D</td><td>57.45</td><td>32.55</td><td>.4598</td></tr><tr><td>5D</td><td>57.79</td><td>33.07</td><td>.4603</td></tr><tr><td>5D w/o pairwise</td><td>57.42</td><td>30.99</td><td>.4598</td></tr><tr><td>5D shuffled M/S/E</td><td>56.80</td><td>29.17</td><td>.4550</td></tr></table>

## 4.3. Reliability Gating

The candidate with the highest predicted reward can still have lower execution utility than the existing supervision. Using the same 5D reward model on 384 test states, Reward-only makes 310 replacements, of which 219 have lower execution reference utility than Source. With the advantage margin and audit gate, TIF reduces the fraction of replacements that lower utility from 70.6% to 53.2%, while maintaining a similar mean utility (0.4602 versus 0.4603 for Reward-only). Table 2(a) reports replacement outcomes for each rule. Mean utility includes states that retain Source.

Audit risk distinguishes candidates with high predicted utility but poor execution outcomes among the reward winners. Of the 310 non-Source candidates selected by reward ranking, 23 satisfy the mismatch definition in Eq. (3). This event captures substantial reward overestimation, whereas the Worse counts above include any replacement with execution utility below Source. The auditor identifies these events with an AUROC of 0.77. Mismatch rates are 22.58% in the highest risk quintile and 1.61% in the lowest.

We further compare selection rules at the same replacement count to evaluate the contribution of risk ranking. Each rule selects 111 of the same 310 candidates, without applying an advantage margin. As shown in Tab. 2(b), audit risk selects 58 candidates with higher execution reference utility than Source, compared with 47 for predicted advantage and 52 for logistic regression risk, and achieves the highest mean utility. Event diagnostics and extended selection comparisons are provided in Appendices E.1 and E.2, respectively.

## 4.4. Policy Improvement through Supervision Selection

We next evaluate whether the trajectories selected in Sec. 4.3 improve policy learning, using task success after behavior cloning. With the same trajectory count and behavior cloning budget, TIF achieves an average success rate of 76.25%, exceeding Source by 3.00 percentage points (Tab. 3). Training directly on greedy branches or random candidates does not outperform Source, whereas reward selection improves average success. With the reward model and advantage margin held fixed, adding the audit gate yields a further gain of 1.47 percentage points, with improvements across all three training seeds. TIF achieves the highest success on Spatial, Object, and Long. Its largest gain over Source is 5.73 percentage points on Long, while success on Goal decreases by 1.00 percentage point.

Table 2. Supervision selection with reliability auditing. Better/Worse count replacements with higher/lower execution reference utility than Source. Utility averages the retained trajectories over all 384 states. (a) Default selection rules. (b) Each rule accepts 111 of the same 310 non-Source reward winners without an advantage margin and retains Source elsewhere. Random reports expected values.
<table><tr><td colspan="5">(a) Default selection rules</td></tr><tr><td>Method</td><td>Replaced</td><td>Better</td><td>Worse</td><td>Utility ↑</td></tr><tr><td>Source</td><td>0</td><td>0</td><td>0</td><td>.4444</td></tr><tr><td>Reward-only</td><td>310</td><td>91</td><td>219</td><td>.4603</td></tr><tr><td>Reward+Margin</td><td>186</td><td>65</td><td>121</td><td>.4587</td></tr><tr><td>TIF</td><td>111</td><td>52</td><td>59</td><td>.4602</td></tr></table>

<table><tr><td>Selection criterion</td><td>Better ↑</td><td>Worse ↓</td><td>Utility ↑</td></tr><tr><td>Random (expected)</td><td>32.58</td><td>78.42</td><td>.4501</td></tr><tr><td>Predicted advantage</td><td>47</td><td>64</td><td>.4589</td></tr><tr><td>Logistic regression risk</td><td>52</td><td>59</td><td>.4621</td></tr><tr><td>Auditor risk</td><td>58</td><td>53</td><td>.4636</td></tr></table>

Section 4.3 evaluates risk ranking at a fixed acceptance count; we extend this comparison to policy training. All selection rules use the same training candidates that pass the advantage margin and match TIF’s replacement count within each suite, yielding 230 replacement trajectories and 570 Source trajectories. TIF outperforms predicted advantage ranking at 75.55% and logistic regression risk at 75.65%, as well as random, low entropy, and consistency selection (Tab. 5). Its success rate is higher than each control across all three training seeds.

Audit targets and supervision. To examine the role of the mismatch target, we compare it with signed error regression and classification of candidates whose execution utility is below Source. The auditors share the same inputs, training data, and training budget. Each target is evaluated through the offline selection protocol in Sec. 4.3 and the policy training comparison at matched replacement counts. As shown in Tab. 4(a), classification relative to Source retains the fewest replacements that lower execution utility, while mismatch supervision achieves the highest mean execution utility. Mismatch supervision also yields the highest policy success, at 76.25%, compared with 75.85% for classification relative to Source and 75.60% for signed error regression.

Table 3. Policy success on LIBERO (%). Post-training methods share the same policy initialization, trajectory count, and training budget, using all valid time steps. Values are means ± sample standard deviations across three training seeds. Original OpenVLA is evaluated before post-training. Bold indicates the highest mean in each column.
<table><tr><td>Method</td><td>Spatial</td><td>Object</td><td>Goal</td><td>Long</td><td>Average</td></tr><tr><td>Original OpenVLA</td><td>74.80</td><td>86.20</td><td>77.20</td><td>37.80</td><td>69.00</td></tr><tr><td>Source</td><td> $8 0 . 2 0 \pm 0 . 9 2$ </td><td> $8 6 . 0 7 \pm 1 . 0 3$ </td><td> ${ \bf 8 3 . 7 3 \pm 0 . 3 1 }$ </td><td> $4 3 . 0 0 \pm 1 . 0 6$ </td><td> $7 3 . 2 5 \pm 0 . 6 8$ </td></tr><tr><td>Greedy</td><td> $7 8 . 0 0 \pm 1 . 1 1 $ </td><td> $8 4 . 5 3 \pm 0 . 7 6$ </td><td> $8 1 . 6 7 \pm 0 . 4 2$ </td><td> $4 1 . 9 3 \pm 1 . 3 3$ </td><td> $7 1 . 5 3 \pm 0 . 5 1$ </td></tr><tr><td>Random</td><td> $7 9 . 0 7 \pm 1 . 2 1$ </td><td> $8 5 . 3 3 \pm 0 . 4 2$ </td><td> $8 2 . 6 7 \pm 0 . 4 2$ </td><td> $4 1 . 9 3 \pm 1 . 2 1$ </td><td> $7 2 . 2 5 \pm 0 . 4 8$ </td></tr><tr><td>Reward-only</td><td> $8 1 . 6 0 \pm 0 . 7 2$ </td><td> $8 7 . 4 0 \pm 0 . 7 2$ </td><td> $8 2 . 9 3 \pm 0 . 6 4$ </td><td> $4 5 . 0 0 \pm 1 . 2 0$ </td><td> $7 4 . 2 3 \pm 0 . 4 0$ </td></tr><tr><td>Reward+Margin</td><td> $8 2 . 1 3 \pm 1 . 0 1$ </td><td> $8 8 . 2 7 \pm 0 . 5 0$ </td><td> $8 3 . 3 3 \pm 0 . 5 0$ </td><td> $4 5 . 4 0 \pm 1 . 3 9$ </td><td> $7 4 . 7 8 \pm 0 . 6 3$ </td></tr><tr><td>TIF</td><td> ${ \bf 8 4 . 1 3 \pm 0 . 8 3 }$ </td><td> ${ \bf 8 9 . 4 0 \pm 0 . 7 2 }$ </td><td> $8 2 . 7 3 \pm 0 . 6 4$ </td><td> ${ \bf 4 8 . 7 3 \pm 0 . 8 3 }$ </td><td> ${ \bf 7 6 . 2 5 \pm 0 . 5 8 }$ </td></tr></table>

Table 4. Audit targets and supervision ablations. Policy success is averaged over four suites (%, mean ± sample SD across three training seeds). (a) Auditors share inputs, training data, and training budget. Offline selection accepts 111 of 310 test proposals without a margin; Worse counts replacements below Source, and Utility averages all 384 test states including Source fallback. Policy training separately uses 230 replacements from training candidates that pass the margin, with counts matched by suite. (b) Each reward model has its own auditor trained under the same protocol. (c) Selected trajectories and training budgets are fixed within each selection method.
<table><tr><td colspan="4">(a) Audit training target</td></tr><tr><td>Target</td><td colspan="2">Offline selection</td><td>Policy success ↑</td></tr><tr><td></td><td>Worse ↓</td><td>Utility ↑</td><td></td></tr><tr><td>Signed error regression</td><td>56</td><td>.4624</td><td> $7 5 . 6 0 \pm 0 . 4 6$ </td></tr><tr><td>Below-Source classification</td><td>50</td><td>.4629</td><td> $7 5 . 8 5 \pm 0 . 5 9$ </td></tr><tr><td>High-score mismatch classification</td><td>53</td><td>.4636</td><td> ${ \bf 7 6 . 2 5 \pm 0 . 5 8 }$ </td></tr></table>

<table><tr><td colspan="2">(b) Reward supervision</td></tr><tr><td>Supervision</td><td>Policy success ↑</td></tr><tr><td>Scalar</td><td> $7 4 . 5 7 \pm 0 . 3 3$ </td></tr><tr><td>2D</td><td> $7 5 . 4 3 \pm 0 . 4 8$ </td></tr><tr><td>5D</td><td> ${ \bf 7 6 . 2 5 \pm 0 . 5 8 }$ </td></tr></table>

<table><tr><td colspan="3">(c) Trajectory supervision</td></tr><tr><td>Selection</td><td>First action</td><td>All valid steps</td></tr><tr><td>Source</td><td> $7 1 . 9 7 \pm 0 . 8 0$ </td><td> $7 3 . 2 5 \pm 0 . 6 8$ </td></tr><tr><td>TIF</td><td> $7 4 . 1 8 \pm 0 . 6 1$ </td><td> ${ \bf 7 6 . 2 5 \pm 0 . 5 8 }$ </td></tr></table>

<table><tr><td>Selection rule</td><td>Success ↑</td></tr><tr><td>Random-K</td><td> $7 4 . 1 5 \pm 0 . 6 3$ </td></tr><tr><td>Advantage Top-K</td><td> $7 5 . 5 5 \pm 0 . 5 0$ </td></tr><tr><td>Logistic risk-K</td><td> $7 5 . 6 5 \pm 0 . 5 2$ </td></tr><tr><td>Low-entropy-K</td><td> $7 5 . 0 3 \pm 0 . 5 1$ </td></tr><tr><td>Consistency-K</td><td> $7 5 . 6 0 \pm 0 . 5 2$ </td></tr><tr><td>TIF</td><td> ${ \bf 7 6 . 2 5 \pm 0 . 5 8 }$ </td></tr></table>

Table 5. Policy success at matched replacement counts. All rules select from the same 348 training candidates that pass the advantage margin, accepting 57/61/54/58 replacements in Spatial/Object/Goal/Long. Each training set contains 230 replacements and 570 Source trajectories. Selections remain unchanged across training seeds. Success is averaged over all four suites (%, mean ± sample SD). TIF uses the result from Tab. 3.

Section 4.2 shows that dimensional supervision improves candidate evaluation. Within the complete selection and policy training pipeline, average success increases from 74.57% with Scalar to 75.43% with 2D and 76.25% with 5D (Tab. 4b). These variants share the same reference utility, gating rules, and policy training budget, with a separate auditor trained for each reward model. Detailed results on the Long suite are provided in Appendix F.2.

With selected trajectories and training budgets fixed, using all valid steps instead of only the first action improves success by 1.28 percentage points for Source and 2.07 for TIF (Tab. 4c). TIF outperforms Source with either scheme.

![](images/e241c6ce3986b1e53d8df784ea35cd615a3a20a937b6f085d598aeb5cd97c42a.jpg)  
Figure 3. Real-robot candidate selection. (a) Tasks and setup. (b) Scores and selections, with advantage relative to Source C1: TIF rejects C2 and retains C1. (c,d) Recorded executions: both reach the goal, but only C2 collides with the obstacle.

## 4.5. Supervision Selection on a Real Robot

We evaluate trajectory selection on a JAKA S5 robot using three tasks: pushing a block across a target line, placing exactly one cup, and relocating a specified cup (Fig. 3a). Each task has 20 candidate pools with five physical executions per pool, including Source, yielding 300 trajectories. Both methods score recorded execution videos with the reward model used above; TIF applies the same auditor and acceptance rule. Task definitions and the evaluation protocol are provided in Appendices G.1 and G.2.

TIF increases task completion among selected trajectories from 75.00% to 78.33% and clean completion from 58.33% to 70.00% (Tab. 6). Clean completion satisfies the instruction with a stable final state and no collision, wrong or extra object interaction, or unnecessary recovery.

In the pushing example, C2 exceeds the advantage margin but fails the audit gate, so TIF retains Source C1 (Fig. 3b). Both complete the task, but C2 collides with the obstacle while C1 completes the push without collision (Fig. 3c,d).

## 5. Conclusion

We presented TIF, which combines multidimensional reward evaluation with risk auditing learned from execution feedback to select supervision trajectories for VLA posttraining. The reward model provides utility predictions for ranking and dimension predictions with uncertainty for auditing. The retained real trajectories supervise behavior cloning. With the same trajectory count and training budget, TIF improves average success across four LIBERO suites from 73.25% to 76.25%. Dimensional supervision ablations and comparisons at fixed replacement counts support the contributions of execution quality information and risk auditing to policy learning.

Table 6. Selection outcomes on recorded robot videos. Counts refer to selected trajectories, with one selection per candidate pool. Reward denotes Reward-only.
<table><tr><td></td><td colspan="2">Task completion</td><td colspan="2">Clean completion</td></tr><tr><td>Task</td><td>Reward</td><td>TIF</td><td>Reward</td><td>TIF</td></tr><tr><td>Push</td><td>15/20</td><td>16/20</td><td>11/20</td><td>14/20</td></tr><tr><td>One cup</td><td>14/20</td><td>15/20</td><td>11/20</td><td>13/20</td></tr><tr><td>Relocation</td><td>16/20</td><td>16/20</td><td>13/20</td><td>15/20</td></tr><tr><td>Total</td><td>45/60</td><td>47/60</td><td>35/60</td><td>42/60</td></tr></table>

Limitations. Experiments use OpenVLA and LIBERO. Future work will examine other policies and benchmarks, and the execution feedback needed to adapt the auditor when models or candidate distributions change.

## References

[1] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, and Ury Zhilinsky. π<sub>0</sub>: A vision-language-action flow model for general robot control, 2026. 1

[2] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu, Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alexander Herzog, Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch, Karl Pertsch, Kanishka Rao, Krista Reymann, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut, Huong Tran, Vincent Vanhoucke, Quan Vuong, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. Rt-2: Vision-language-action models transfer web knowledge to robotic control, 2023. 1

[3] Zengjue Chen, Peidong Liu, Jiawei Li, and Qi Wang. Hawmpo: Hallucination-aware world model-based policy optimization for generalist robot policy, 2026. 3

[4] Mingtong Dai, Lingbo Liu, Yongjie Bai, Yang Liu, Zhouxia Wang, Rui SU, Chunjie Chen, Liang Lin, and Xinyu Wu. Rover: Robot reward model as test-time verifier for visionlanguage-action model, 2025. 2

[5] Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization. In International conference on machine learning, pages 10835–10866. PMLR, 2023. 2

[6] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021. 5

[7] Dongchi Huang, Hongyin Zhang, Bohan Hou, Siteng Huang, Zhian Su, Hang Guo, Tong Lu, Jiahao Tang, Jianfei Yang, Donglin Wang, Peixi Peng, Mingxiu Chen, Deli Zhao, and Xin Li. Rynnvalue: Scaling robotic value foundation models with temporal distance, 2026. 2

[8] Zhennan Jiang, Shangqing Zhou, Yutong Jiang, Zefang Huang, Mingjie Wei, Yuhui Chen, Tianxing Zhou, Zhen Guo, Hao Lin, Quanlu Zhang, Yu Wang, Haoran Li, Chao Yu, and Dongbin Zhao. Wovr: World models as reliable simulators for post-training vla policies with rl, 2026. 2

[9] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. Openvla: An opensource vision-language-action model, 2024. 1, 5, 3

[10] Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning

vision-language-action models: Optimizing speed and success, 2025. 2

[11] Tony Lee, Andrew Wagenmaker, Karl Pertsch, Percy Liang, Sergey Levine, and Chelsea Finn. Roboreward: Generalpurpose vision-language reward models for robotics, 2026. 2

[12] Wenzhan Li, Yiran Qin, Heng Zhou, Huirui Wang, Yulan Guo, and Ruimao Zhang. Trust-calibrated vla policy posttraining with evolving imagination, 2026. 3

[13] Anthony Liang, Yigit Korkmaz, Jiahui Zhang, Minyoung Hwang, Abrar Anwar, Sidhant Kaushik, Aditya Shah, Alex S. Huang, Luke Zettlemoyer, Dieter Fox, Yu Xiang, Anqi Li, Andreea Bobu, Abhishek Gupta, Stephen Tu, Er dem Biyik, and Jesse Zhang. Robometer: Scaling generalpurpose robotic reward models via trajectory comparisons, 2026. 2

[14] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36:44776–44791, 2023. 2, 5, 1

[15] Guanxing Lu, Wenkai Guo, Chubin Zhang, Yuheng Zhou, Haonan Jiang, Zifeng Gao, Yansong Tang, and Ziwei Wang. Vla-rl: Towards masterful and general robotic manipulation with scalable reinforcement learning, 2025. 2

[16] Mitsuhiko Nakamoto, Oier Mees, Aviral Kumar, and Sergey Levine. Steering your generalists: Improving robotic foun dation models via value guidance, 2025. 2

[17] NVIDIA. Cosmos-predict2: World simulation model for physical ai, 2025. 5, 1, 3

[18] Maxime Oquab, Timothee Darcet, Th´ eo Moutakanni, Huy´ Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mah moud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve Je-´ gou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision, 2024. 5, 3

[19] Shuhan Tan, Kairan Dou, Yue Zhao, and Philipp Krahenb¨ uhl. Interactive post-training for vision-language-¨ action models, 2025. 2

[20] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, Jianlan Luo, You Liang Tan, Lawrence Yunliang Chen, Pannag Sanketi, Quan Vuong, Ted Xiao, Dorsa Sadigh, Chelsea Finn, and Sergey Levine. Octo: An open-source generalist robot policy, 2024. 1

[21] Pengzhi Yang, Xinyu Wang, Pengyu Jing, Kehan Wen, Yiduo Qu, Zhenhao Huang, Minghao Fu, Xin Liu, Yaheng Shen, and Fan Shi. Rarm: Confidence-gated progress reward modeling for rl in manipulation, 2026. 2

[22] Sherry Yang, Yilun Du, Kamyar Ghasemipour, Jonathan Tompson, Leslie Kaelbling, Dale Schuurmans, and Pieter Abbeel. Learning interactive real-world simulators, 2024. 1

[23] Chenhao Zhang, Hanyu Zhao, Hang Cheng, Tengfei Pan, and Long Zeng. Wise: World-model-guided imagination scheduling for efficient post-training of vision-languageaction models, 2026. 1, 3

[24] Zijian Zhang, Kaiyuan Zheng, Zhaorun Chen, Joel Jang, Yi Li, Siwei Han, Chaoqi Wang, Mingyu Ding, Dieter Fox, and Huaxiu Yao. Grape: Generalizing robot policy via preference alignment. In 2026 IEEE International Conference on Robotics and Automation (ICRA), pages 16868–16875. IEEE, 2026. 2

[25] Shangqing Zhou, Zhennan Jiang, Ning Ma, Xin Fu, Yupeng Zheng, Shuai Tian, Huangrui Li, Jiahao Liu, Chao Yu, Zhongpu Xia, Haoran Li, and Dongbin Zhao. Wovr+: World model as reliable simulator for versatile vla policy optimization with rl, 2026. 2

[26] Fangqi Zhu, Zhengyang Yan, Zicong Hong, Quanxin Shou, Xiao Ma, and Song Guo. Wmpo: World model-based policy optimization for vision-language-action models. In International Conference on Learning Representations, pages 62486–62502, 2026. 1, 2

# Trusting Imagined Futures: Auditing Imagined Rewards for VLA Post-Training

Supplementary Material

## A. Data and Candidate Construction

## A.1. Tasks and Data Splits

Our experiments cover the Spatial, Object, Goal, and Long suites of LIBERO, with ten tasks per suite [14]. We construct a component dataset for training and evaluating the reward model and auditor, and a natural candidate dataset for supervision selection and policy post-training. In both datasets, candidates associated with the same initial state form a candidate pool. Each candidate has a recorded execution trajectory and a corresponding imagined video. Table S1 summarizes the dataset sizes and their roles.

Data splits are grouped by source trajectory. Initial states derived from the same source trajectory, together with all their candidates, are assigned to the same split. When records have multiple levels of derivation, we trace their provenance back to the original trajectory. The natural validation and test sets inherit the validation and test provenance assignments of the component dataset, respectively. Neither set shares source trajectories with the training data of the reward model or auditor. The natural training set may share source trajectories with the reward training data.

Offline reward evaluation and auditing analyses use the 384 natural test states in which the task is not already successful at initialization, yielding 1,920 candidates. Policy post-training uses the natural training set and retains one execution trajectory per initial state, giving 800 trajectories in total and 200 per suite.

## A.2. Candidate Construction and Execution

The component dataset contains four controlled candidates per initial state, providing examples with different execution quality. Each natural candidate pool contains five candidates: Source, one greedy branch, and three sampled branches.

Source is an existing fixed action proposal associated with the initial state, obtained from either a LIBERO demonstration or saved policy action records. For demonstration sources, we extract the action sequence following the corresponding starting point. For historical policy sources, we retrieve the saved actions. The construction of historical policy records includes extending a sequence by repeating its final action, as well as repeating a saved single action. We then execute Source in its associated scene and initial state, recording observations, actions, and execution feedback.

The remaining four branches use the OpenVLA policy for the corresponding suite [9]. The greedy branch uses greedy decoding, while the three sampled branches use temperature 1, top-p 1, and top-k 0. At each step, the policy generates an action from the current execution observation, allowing subsequent actions to respond to changes observed during execution.

All branches within a pool start from the same scene and initial state. Each sequence contains at most eight control commands executed at 20 Hz and stops upon reaching its prescribed length or environment termination. For a branch that terminates early, we retain the valid executed prefix. After collection, the action sequences and their execution records remain fixed for subsequent video generation, evaluation, and supervision selection.

## A.3. Pairing Imagined Videos with Executions

We use Cosmos-Predict2 2B to generate an imagined video for each candidate [17]. Generation is conditioned on the initial visual observation and the recorded action sequence, with an adapter using the initial proprioceptive state to construct the required world model conditioning. For candidates that terminate early, generation uses the valid action prefix.

Each pair associates an imagined video and an execution record with the same task instruction, initial state, and candidate action sequence. The execution record contains observations, actions, and feedback at valid steps, which are used to construct the five quality labels and reference utility. The label computations are described in Appendix B.

Offline selection uses reward predictions from imagined videos and the corresponding audit risk. One candidate is retained per initial state, and observation–action pairs from all valid steps of its actual execution segment are used for behavior cloning. World model conditioning and video processing details are provided in Appendix C.

## B. Execution Labels and Evaluation Metrics

## B.1. Five Dimensions of Execution Quality

We construct five continuous quality scores from each candidate’s actual execution: task progress, goal correctness, manipulation quality, safety, and execution stability. The labels use environment feedback, robot and object states, contact records, and action statistics, with normalization parameters fitted on the training data. Each score lies in [0, 1], with higher values indicating better quality in the corresponding dimension. Table S2 summarizes their definitions

Table S1. Dataset sizes and roles. Candidate counts are per initial state. Each pair consists of a recorded execution trajectory and its corresponding imagined video.
<table><tr><td>Dataset</td><td>Initial states</td><td>Candidates</td><td>Pairs</td><td>Role</td></tr><tr><td>Component</td><td>8,000</td><td>4</td><td>32,000</td><td>Reward model and auditor training and evalua- tion</td></tr><tr><td>Natural training</td><td>800</td><td>5</td><td>4,000</td><td>Supervision selection for behavior cloning</td></tr><tr><td>Natural validation</td><td>400</td><td>5</td><td>2,000</td><td>Validation and configuration selection</td></tr><tr><td>Natural testing</td><td>400</td><td>5</td><td>2,000</td><td>Offline candidate evaluation and selection anal- ysis</td></tr></table>

Table S2. Execution quality dimensions and the signals used to construct their labels.
<table><tr><td>Dimension</td><td>Evaluation focus</td><td>Execution signals</td></tr><tr><td>Task progress (P)</td><td>Advancement toward task completion</td><td>Changes in success status, environment rewards, approach to the target, and relevant object state changes</td></tr><tr><td>Goal correctness (G)</td><td>Agreement with the object and goal specified by the instruction</td><td>Target object interaction, approach to the goal location, and sat- isfaction of task conditions</td></tr><tr><td>Manipulation quality (M)</td><td>Effectiveness of grasping, transport, and placement</td><td>Gripper engagement, object motion, end-effector motion, and feedback on task-relevant operations</td></tr><tr><td>Safety (S)</td><td>Risks associated with abnormal contact and motion</td><td>Contact anomalies, excessive actions, action saturation, and abrupt state changes</td></tr><tr><td>Execution stability (E)</td><td>Smoothness of actions and state changes</td><td>Action smoothness, action reversals, state jitter, and gripper switching</td></tr></table>

and evaluation signals.

Task progress and goal correctness describe the extent of progress and its alignment with the instruction, respectively. Object motion may indicate local progress, while goal correctness further considers whether the manipulated object, motion direction, and resulting state match the instruction. Manipulation quality evaluates the effectiveness of the interaction. Safety and execution stability capture abnormal risks and the smoothness of execution, respectively.

The continuous scores are used to construct reference utility, while discrete levels supervise dimension prediction. All five dimensions use the same thresholds: scores below 0.34 are low, scores from 0.34 to below 0.67 are medium, and scores of at least 0.67 are high. These levels are encoded as 0, 1, and 2, respectively.

## B.2. Execution Reference Utility

We combine the five continuous scores into an execution reference utility to supervise utility prediction and compare candidates from the same initial state. We first compute

$$
\begin{array} { c } { v = \left( 0 . 3 5 P + 0 . 3 0 G + 0 . 2 0 M + 0 . 1 5 E \right) } \\ { \cdot \left( 0 . 3 5 + 0 . 6 5 S \right) . } \end{array}\tag{7}
$$

Task progress, goal correctness, manipulation quality, and execution stability contribute through a weighted sum. Safety adjusts this sum through a multiplicative factor.

We then apply two caps. When neither task progress nor goal correctness reaches the high level, the score is capped at 0.666. When safety is low, the score is capped at 0.339. The lower cap applies when both conditions hold. The resulting value is the final reference utility u. Within each initial state, candidates with higher reference utility receive higher preference.

The reward model outputs a utility estimate uˆ through a separate prediction head and uses this estimate to rank candidates. The dimension predictions provide additional information for auditing. Equation 7 constructs the supervision target; predicted utility is produced directly by the utility head.

## B.3. Audit Targets and Evaluation Events

After reward training, we compute predictions on a separate dataset for auditor training and pair them with the execution reference utilities. A candidate is labeled as a high-score mismatch when all three conditions hold:

$$
\begin{array} { r } { \hat { u } \geq 0 . 5 8 , } \\ { u \leq 0 . 4 2 , } \\ { \hat { u } - u \geq 0 . 2 0 . } \end{array}\tag{8}
$$

These conditions require high predicted utility, low execution utility, and a sufficiently large overestimation gap. The audit label is 1 when all conditions hold and 0 otherwise. The reward model parameters remain fixed during auditor training.

Additional analyses compare two alternative audit targets. Signed error uses the difference uˆ − u as a regression target. Below Source uses a binary target indicating whether the candidate’s execution reference utility is lower than that of Source from the same initial state.

Table S3. Evaluation metrics and their aggregation. Execution reference utility is used for candidate evaluation; task success is used for downstream policy evaluation.
<table><tr><td>Metric</td><td>Definition</td></tr><tr><td>Macro-F1</td><td>For each quality dimension, the unweighted mean of the F1 scores for the low, medium, and high levels.</td></tr><tr><td>Ordinal MAE</td><td>Mean absolute error between predicted and reference levels encoded as 0, 1, and 2.</td></tr><tr><td>Pairwise accuracy</td><td>The fraction of candidate comparisons for which predicted utility agrees with the ordering given by execution reference utility. Comparisons are made within the same initial state.</td></tr><tr><td>Top-1 accuracy</td><td>The fraction of initial states for which the candidate selected by predicted utility has the highest execu- tion reference utility in its pool.</td></tr><tr><td>Selected utility</td><td>Mean execution reference utility of the final retained candidates across initial states, including states that retain Source.</td></tr><tr><td>Better / Worse</td><td>Counts of replacements whose execution reference utility is higher or lower than that of Source, re- spectively.</td></tr><tr><td>AUROC / AP</td><td>Discrimination metrics for audit risk on the specified binary event, reported with the evaluation popu- lation and positive count.</td></tr><tr><td>Policy success rate</td><td>The fraction of successful episodes when evaluating the updated policy on complete tasks.</td></tr></table>

High-score mismatch measures disagreement between prediction and execution, whereas Below Source compares the replacement candidate with the existing supervision. The two events may occur together or separately. Subsequent tables report their positive counts and discrimination metrics separately.

## B.4. Evaluation Metrics

We evaluate reward prediction, candidate selection, and policy performance using the metrics in Table S3.

Top-1 accuracy measures utility ranking within candidate pools. Policy success rate measures the ability of the updated policy to complete tasks. These metrics correspond to supervision selection and downstream policy evaluation, respectively.

In the audit analyses, all proposals refers to candidates that have the highest predicted utility in their pool and are not Source. The high-score subset further requires predicted utility of at least 0.58. AUROC and AP are computed using the event labels within the corresponding population.

Policy results are computed separately for three training seeds and reported as the mean success rate and sample standard deviation. Each seed is evaluated over 2,000 episodes across the four suites, with 500 episodes per suite. Because the suites have equal evaluation sizes, the average success rate across suites also equals the total success count divided by 2,000 for that seed.

## C. Model and Training Details

## C.1. Model Components

TIF uses Cosmos-Predict2 2B to generate imagined videos, DINOv2-L to extract visual features from video frames, and OpenVLA as the policy for post-training [9, 17, 18]. The world model generates videos from the initial visual observation, recorded action sequence, and conditioning constructed from the initial proprioceptive state. Candidate collection and pairing are described in Appendix A.

The reward model fuses video features with the task instruction and candidate action information. It outputs a distribution over the low, medium, and high levels for each of the five quality dimensions, together with a separate utility estimate. Utility predictions rank candidates, while dimension predictions and associated uncertainty statistics support risk auditing. The DINOv2-L parameters remain unchanged during reward training.

The risk auditor is a multilayer perceptron with approximately 25,000 trainable parameters. It takes reward model outputs and imagined video diagnostics as input and produces a mismatch risk score for each candidate.

## C.2. Reward Model and Auditor Training

The reward model is jointly trained using the dimension labels, execution reference utility, and preferences between candidates from the same initial state, as defined in Appendix B. The training objective is

$$
\mathcal { L } _ { R } = \mathcal { L } _ { \mathrm { d i m } } + 0 . 5 \mathcal { L } _ { \mathrm { u t i l i t y } } + 0 . 1 \mathcal { L } _ { \mathrm { p a i r } } .\tag{9}
$$

The dimension loss supervises the levels of the five quality dimensions, the utility loss supervises overall utility prediction, and the pairwise loss supervises the relative ordering of candidates from the same initial state.

Table S4. Imagined video diagnostic fields. The shared prefix vq is omitted.
<table><tr><td>Category</td><td>Fields</td><td>Description</td></tr><tr><td>ing</td><td>Availability and decod- feature_available, decode_ok</td><td>Feature availability and successful video decod- ing.</td></tr><tr><td>Video format</td><td>width,height,fps,frame_count_meta, frame_count_decoded</td><td>Width, height, frame rate, metadata frame count, and actual decoded frame count.</td></tr><tr><td>Brightness and contrast</td><td>mean_brightness,mean_contrast</td><td>Image brightness and contrast statistics.</td></tr><tr><td>Sharpness</td><td>mean_laplacian_var, median_laplacian_var</td><td>Mean and median Laplacian variance.</td></tr><tr><td>Temporal changes</td><td>first_last_absdiff</td><td>mean_frame_absdiff, max_frame_absdiff, Mean and maximum absolute differences between adjacent frames, and the absolute difference be- tween the first and last frames.</td></tr><tr><td rowspan="3">Diagnostic flags</td><td>is_low_motion,</td><td>is_very_low_motion, Low motion, very low motion, low contrast, dark</td></tr><tr><td>is_low_contrast,</td><td>is_dark_or_blank, or blank images, a blur proxy, and decoding fail-</td></tr><tr><td>is_blurry-proxy,is_bad_decode</td><td>ure.</td></tr></table>

After reward training, we compute predictions on a separate dataset for auditor training. These predictions and their corresponding execution reference utilities define the high-score mismatch labels in Appendix B.3. The auditor is trained using these binary labels. The reward model parameters remain unchanged during this stage, so the auditor learns the prediction errors of a fixed evaluator.

The trained reward model and auditor are then applied to the natural candidate pools for evaluation and selection. The complete Scalar, 2D, and 5D pipeline comparisons train a corresponding reward model and auditor for each configuration, using the same utility supervision, selection rules, and policy training budget.

## C.3. Auditor Inputs and Video Diagnostics

The auditor input has two components. The first contains reward model outputs: predicted utility, distributions and scores for the five dimensions, and prediction uncertainty statistics. The second contains imagined video diagnostics describing availability, format, image quality, and changes between frames. The selector separately computes the predicted advantage over Source.

The video diagnostics contain 20 fields, grouped by function in Table S4. All fields are extracted from imagined videos and use the prefix vq .

These diagnostics describe video readability, image quality, and temporal changes, providing auxiliary information for auditing reward predictions. Supervision for execution quality, including collision risks and goal correctness, comes from the paired execution records.

Candidate source metadata is excluded from the auditor input. Execution reference utility is used to construct training labels and evaluate selection outcomes. Offline selection uses reward predictions, video diagnostics, and audit risk.

## C.4. Supervision Selection and Policy Training

For each natural candidate pool, the selector first identifies the candidate with the highest predicted utility and computes its predicted advantage over Source. The candidate is accepted when its advantage exceeds 0.15 and its audit risk is at most 0.50. Otherwise, Source is retained. If the candidate with the highest predicted utility fails these conditions, the selector falls back directly to Source. Each initial state contributes one actual execution segment.

Behavior cloning uses observation–action pairs from all valid steps of the retained segments. Each training sample is drawn by first selecting a segment uniformly and then selecting a valid time step uniformly within that segment. The real observation before the action is used to predict the corresponding action. This procedure gives each segment the same probability of being sampled.

We compute the training loss over OpenVLA action tokens and update only the LoRA parameters in its attention and MLP modules. All methods use the same policy initialization, candidate pools, training trajectory count, and update budget. For each method, the selected trajectory set remains unchanged across the three training seeds. Table S5 summarizes the configuration.

During complete-task evaluation, the updated OpenVLA policy generates actions independently. The reward model, world model, and auditor are used for offline supervision selection and do not participate in action generation during this evaluation.

## D. Additional Reward Evaluation

This section supplements the reward evaluation in Sec. 4.2. All experiments use 1,920 candidates from the 384 natural test states in which the task is not already successful at initialization. Quality labels and reference utilities are derived from the paired executions. Selection uses the candidate with the highest predicted utility, without an advantage margin or audit gate.

Table S5. Policy post-training and evaluation configuration.
<table><tr><td>Item</td><td>Setting</td></tr><tr><td>Policy</td><td>OpenVLA</td></tr><tr><td>Updated parameters</td><td>LoRA parameters in attention and MLP modules</td></tr><tr><td>Training data</td><td>One retained actual execution segment per initial state in the natural training set</td></tr><tr><td>Training segments</td><td>200 per suite; 800 in total</td></tr><tr><td>Supervision steps</td><td>All valid execution steps</td></tr><tr><td>Sampling</td><td>Uniform over segments, then uniform over valid time steps within the selected segment</td></tr><tr><td>Updates</td><td>500 per suite and training seed</td></tr><tr><td>Effective batch size</td><td>16 observation-action pairs</td></tr><tr><td>Training seeds</td><td>7, 11, and 23</td></tr><tr><td>Evaluation episodes</td><td>50 per task; 500 per suite</td></tr><tr><td>Evaluation initial conditions</td><td>The same evaluation initial states and environment seeds across methods</td></tr></table>

Table S6. Prediction of individual quality dimensions. P, G, M, S, and E denote task progress, goal correctness, manipulation quality, safety, and execution stability, respectively.
<table><tr><td>Metric</td><td>P</td><td>G</td><td>M</td><td>S</td><td>E</td></tr><tr><td>Macro-F1 ↑</td><td>.748</td><td>.740</td><td>.732</td><td>.745</td><td>.758</td></tr><tr><td>Ordinal MAE↓</td><td>.248</td><td>.247</td><td>.222</td><td>.216</td><td>.208</td></tr></table>

## D.1. Prediction of Individual Dimensions

Table S6 reports Macro-F1 and ordinal MAE for each quality dimension. Macro-F1 measures classification performance across the low, medium, and high levels. Ordinal MAE measures the distance between predicted and reference levels using the encoding 0, 1, and 2.

Macro-F1 ranges from 0.732 to 0.758, and ordinal MAE ranges from 0.208 to 0.248. Execution stability has the highest Macro-F1 and the lowest ordinal MAE. Manipulation quality has a lower Macro-F1 than task progress and goal correctness but a lower ordinal MAE, reflecting the different aspects of classification performance and ordinal prediction error captured by these metrics.

## D.2. Reward Supervision Ablations

Table 1 in the main paper compares Scalar, 2D, and 5D supervision. Scalar predicts utility alone. 2D adds supervision for task progress and goal correctness, while 5D uses all five dimensions. The three configurations share the same reference utility and pairwise supervision, and rank candidates using a separate utility prediction head.

We perform two independent ablations of the complete 5D model. The first removes the pairwise loss while retaining dimension prediction and utility regression. The second shuffles the manipulation quality, safety, and execution stability labels to examine the contribution of their correspondence with the associated executions.

Removing the pairwise loss reduces Top-1 accuracy from 33.07% to 30.99%. Shuffling the three dimension labels reduces Top-1 accuracy to 29.17% and mean selected execution utility from 0.4603 to 0.4550. Both relative ranking supervision within candidate pools and the correspondence between quality labels and executions contribute to candidate ranking.

## D.3. Reward Input Information

We compare three input configurations: the initial image, task instruction, and candidate actions; the same inputs with an imagined video; and the same inputs with a recorded execution video. A reward model is trained separately for each configuration and selects candidates from the same pools by predicted utility. Table S7 reports the full results. The imagined video row corresponds to the 5D model in Table 1 of the main paper.

Adding imagined video improves pairwise accuracy by 1.88 percentage points and Top-1 accuracy by 3.12 percentage points. Mean selected execution utility increases from 0.4527 to 0.4603. These improvements indicate that imagined videos provide useful information for candidate evaluation beyond the initial image and action sequence.

Recorded execution videos yield the highest ranking accuracies and mean selected execution utility. Their Top-1 accuracy is 50.52%, exceeding the imagined video configuration by 17.45 percentage points. This difference reflects a gap between evaluating predicted futures and observing actual execution outcomes. On the same candidate pools, retaining Source throughout gives a mean utility of 0.4444, while selecting the candidate with the highest execution reference utility in each pool gives 0.4871.

Table S7. Candidate selection with different reward inputs. Pairwise and Top-1 accuracies are percentages. Utility is the mean execution reference utility of selected candidates across all 384 states. Below Source counts selections with lower execution reference utility than Source from the same initial state.
<table><tr><td>Reward model input</td><td>Pairwise ↑</td><td>Top-1 ↑</td><td>Utility ↑</td><td>Below Source ↓</td></tr><tr><td>Initial image + instruction + actions</td><td>55.91</td><td>29.95</td><td>.4527</td><td>233</td></tr><tr><td>Same inputs + imagined video</td><td>57.79</td><td>33.07</td><td>.4603</td><td>219</td></tr><tr><td>Same inputs + recorded execution video</td><td>64.32</td><td>50.52</td><td>.4735</td><td>165</td></tr></table>

Table S8. Event discrimination and grouping by audit risk. (a) The high-score subset contains candidates with predicted utility of at least 0.58. (b) Candidates are ranked by audit risk. Percentages in parentheses are event rates within each group.

(a) Event discrimination
<table><tr><td>Event and population</td><td>Positives / Candidates</td><td>AUROC ↑</td><td>AP↑</td></tr><tr><td>Below Source, all proposals</td><td>219/310</td><td>.774</td><td>.872</td></tr><tr><td>High-score mismatch, all proposals</td><td>23/310</td><td>.77</td><td>.30</td></tr><tr><td>High-score mismatch, high-score subset</td><td>23/142</td><td>.73</td><td>.38</td></tr></table>

<table><tr><td>Group</td><td>Candidates</td><td>High-score mismatch</td><td>Below Source</td></tr><tr><td>All proposals</td><td>310</td><td>23 (7.42%)</td><td>219 (70.65%)</td></tr><tr><td>Highest-risk 20%</td><td>62</td><td>14 (22.58%)</td><td>50 (80.65%)</td></tr><tr><td>Lowest-risk 20%</td><td>62</td><td>1 (1.61%)</td><td>26 (41.94%)</td></tr></table>

## E. Additional Auditing and Selection Analysis

This section supplements the risk identification and candidate selection results in Sec. 4.3 and examines the effect of different audit training targets. The evaluation uses the 310 non-Source candidates selected by reward ranking on the natural test set. High-score mismatch and Below Source are defined in Appendix B.3.

## E.1. Risk Identification and Group Analysis

Table S8(a) reports the discrimination performance of audit risk for the two events. Among the 310 candidates, 23 satisfy the mismatch conditions, and 219 have lower execution reference utility than their corresponding Source. Restricting predicted utility to at least 0.58 yields a high-score subset of 142 candidates containing all 23 mismatch cases.

Within the high-score subset, the auditor achieves a mismatch AUROC of 0.73, distinguishing risk among candidates that already receive high reward predictions. Mismatch prevalence is 7.42% among all proposals and 16.20% within the high-score subset. We report positive counts alongside AP to specify the evaluation population for each result.

The 62 candidates with the highest audit risk contain 14 mismatch cases, whereas the 62 with the lowest risk contain one. The Below Source rate also decreases from 80.65% in the highest-risk group to 41.94% in the lowest-risk group. Audit risk therefore captures differences in both prediction mismatch and replacement quality.

Table S9. Selection with 111 replacements. Better and Worse count replacements with higher and lower execution reference utility than Source, respectively. Utility is averaged over all 384 test states.
<table><tr><td>Selection rule</td><td>Better ↑</td><td>Worse ↓</td><td>Utility ↑</td></tr><tr><td>Random (expected)</td><td>32.58</td><td>78.42</td><td>.4501</td></tr><tr><td>Predicted advantage</td><td>47</td><td>64</td><td>.4589</td></tr><tr><td>Simple score risk</td><td>46</td><td>65</td><td>.4602</td></tr><tr><td>Logistic regression risk</td><td>52</td><td>59</td><td>.4621</td></tr><tr><td>Auditor w/o utility</td><td>51</td><td>60</td><td>.4621</td></tr><tr><td>Full auditor</td><td>58</td><td>53</td><td>.4636</td></tr></table>

## E.2. Selection at a Fixed Replacement Count

We select 111 candidates from the same 310 non-Source reward winners and retain Source at the remaining initial states. All rules use the same replacement count without an advantage margin. Mean execution utility is computed over all 384 test states, including those that retain Source.

Table S9 extends the comparison in Table 2(b) of the main paper. Random selection reports the expected outcome of uniformly selecting 111 candidates. The auditor without utility removes predicted utility and its derived features while retaining the other inputs.

The full auditor retains 11 more beneficial replacements than predicted advantage ranking and six more than logistic regression risk. Without utility features, the auditor still retains 51 beneficial replacements, compared with 47 for predicted advantage. Including utility information increases this count to 58.

Table S10. Event discrimination with different audit training targets.
<table><tr><td>Training target</td><td>Mismatch AUROC ↑</td><td>Mismatch AP ↑</td><td>Below Source AUROC ↑</td></tr><tr><td>Signed-error regression</td><td>.73</td><td>.25</td><td>.75</td></tr><tr><td>Below-Source classification</td><td>.64</td><td>.16</td><td>.79</td></tr><tr><td>High-score mismatch classification</td><td>.77</td><td>.30</td><td>.77</td></tr></table>

This comparison selects a fixed number of candidates by score ranking. Default TIF instead determines acceptance using the advantage margin and risk threshold. Both procedures produce 111 replacements on this test set, but they select different candidate sets.

## E.3. Effect of the Audit Training Target

We compare signed-error regression, Below-Source classification, and high-score mismatch classification. The three models share auditor inputs, training data, and training budget. Their targets are defined in Appendix B.3. Table S10 reports event discrimination on the same 310 candidates.

Below-Source classification achieves the highest AU-ROC for its corresponding event, at 0.79. High-score mismatch classification achieves the highest mismatch AUROC and AP, at 0.77 and 0.30, respectively. Its Below Source AUROC in Table S10 is the same result as in Table S8(a), displayed at a different precision.

Table 4(a) of the main paper reports the corresponding selection quality and downstream policy performance. When accepting 111 test candidates, Below-Source classification retains the fewest Worse replacements, at 50. High-score mismatch classification retains 53 but achieves a higher mean execution utility of 0.4636. Worse counts utility decreases, whereas mean utility also accounts for their magnitudes, so the two metrics can rank selectors differently.

The downstream comparison uses natural training candidates, with replacement counts matched by suite. Each training set contains 230 replacement trajectories and 570 Source trajectories. Signed-error regression, Below-Source classification, and high-score mismatch classification achieve mean policy success rates of 75.60%, 75.85%, and 76.25%, respectively. In this comparison, the mismatch target achieves both the highest mean execution utility in offline selection and the highest downstream policy success rate.

## F. Detailed Policy Results

This section supplements the policy evaluation in the main paper. All methods use the training configuration in Appendix C.4. Each training seed is evaluated over 2,000 episodes across the four LIBERO suites. Table S11 reports success counts for individual seeds. Means and sample standard deviations are computed from the success rates of the three seeds.

## F.1. Results Across Seeds and Matched Replacement Counts

Policy comparisons at matched replacement counts use the 348 natural training proposals that satisfy the advantage margin. Each method accepts 57, 61, 54, and 58 replacements in Spatial, Object, Goal, and Long, respectively, while retaining 200 training trajectories per suite. Each resulting training set contains 230 replacement trajectories and 570 Source trajectories.

Predicted advantage ranking prioritizes candidates with larger advantages, while logistic regression risk and audit risk prioritize candidates with lower predicted risk. Random-K samples the required number of candidates within each suite. Low-entropy-K and Consistency-K select candidates according to prediction entropy and consistency scores, respectively. For each method, the selected set remains unchanged across the three training seeds.

The TIF row uses the same evaluation records as the main policy experiment. Across all three seeds, TIF exceeds both predicted advantage ranking and logistic regression risk selection. Its gains over predicted advantage are 0.50, 0.85, and 0.75 percentage points, respectively. The corresponding gains over logistic regression risk are 0.70, 0.65, and 0.45 percentage points.

The audit target comparison also matches replacement counts within each suite. Signed-error regression selects candidates in ascending order of predicted error, while the two classification targets select in ascending order of predicted risk. This comparison matches counts through ranking; the default TIF risk threshold of 0.50 is not applied to the regression scores.

## F.2. Effect of Reward Dimensions on the Long Suite

The main paper compares the average performance of Scalar, 2D, and 5D within the complete selection and policy training pipeline. Table S12 provides the Long suite results. All configurations use the same reference utility, gating rules, and policy training budget, with a separate auditor trained for each reward model.

Table S11. Policy success counts for individual training seeds. Each seed is evaluated over 2,000 episodes. Mean success rates are reported in percent with sample standard deviations. The TIF row also supplies the results for matched replacement comparisons and the high-score mismatch target.
<table><tr><td>Configuration</td><td>Seed 7</td><td>Seed 11</td><td>Seed 23</td><td>Mean success (%)</td></tr><tr><td colspan="5">Reference configurations and reward supervision</td></tr><tr><td>Source</td><td>1464</td><td>1479</td><td>1452</td><td> $7 3 . 2 5 \pm 0 . 6 8$ </td></tr><tr><td>2D complete pipeline</td><td>1507</td><td>1519</td><td>1500</td><td> $7 5 . 4 3 \pm 0 . 4 8$ </td></tr><tr><td>TIF (5D)</td><td>1521</td><td>1538</td><td>1516</td><td> $7 6 . 2 5 \pm 0 . 5 8$ </td></tr><tr><td colspan="5">Selection rules at matched replacement counts</td></tr><tr><td>Random-K</td><td>1484</td><td>1495</td><td>1470</td><td> $7 4 . 1 5 \pm 0 . 6 3$ </td></tr><tr><td>Advantage Top-K</td><td>1511</td><td>1521</td><td>1501</td><td> $7 5 . 5 5 \pm 0 . 5 0$ </td></tr><tr><td>Logistic risk-K</td><td>1507</td><td>1525</td><td>1507</td><td> $7 5 . 6 5 \pm 0 . 5 2$ </td></tr><tr><td>Low-entropy-K</td><td>1498</td><td>1512</td><td>1492</td><td> $7 5 . 0 3 \pm 0 . 5 1$ </td></tr><tr><td>Consistency-K</td><td>1506</td><td>1524</td><td>1506</td><td> $7 5 . 6 0 \pm 0 . 5 2$ </td></tr><tr><td colspan="5">Audit targets at matched replacement counts</td></tr><tr><td>Signed-error regression</td><td>1510</td><td>1522</td><td>1504</td><td> $7 5 . 6 0 \pm 0 . 4 6$ </td></tr><tr><td>Below-Source classification</td><td>1514</td><td>1530</td><td>1507</td><td> $7 5 . 8 5 \pm 0 . 5 9$ </td></tr></table>

Table S12. Policy success on Long and averaged across all four suites with different reward supervision. Values are percentages, reported as mean ± sample standard deviation across three training seeds.
<table><tr><td>Supervision</td><td>Long</td><td>Average</td></tr><tr><td>Scalar</td><td> $4 3 . 3 3 \pm 0 . 9 2$ </td><td> $7 4 . 5 7 \pm 0 . 3 3$ </td></tr><tr><td>2D</td><td> $4 6 . 1 3 \pm 1 . 3 0$ </td><td> $7 5 . 4 3 \pm 0 . 4 8$ </td></tr><tr><td>5D</td><td> ${ \bf 4 8 . 7 3 \pm 0 . 8 3 }$ </td><td> ${ \bf 7 6 . 2 5 \pm 0 . 5 8 }$ </td></tr></table>

On Long, 5D improves over Scalar and 2D by 5.40 and 2.60 percentage points, respectively. These gains exceed the corresponding improvements of 1.68 and 0.82 percentage points averaged across all four suites. The effect of dimensional supervision varies across suites. For example, on Goal, 5D achieves 82.73% mean success, compared with 83.47% for 2D.

## F.3. Supervision from Different Trajectory Steps

The comparison between first-action supervision and supervision from all valid steps uses the same retained trajectories within each selection method. Policy initialization, update count, and batch size are also held fixed. The first-action setting uses only the initial observation–action pair of each segment. The all-valid-steps setting samples a segment uniformly and then samples a valid time step uniformly within it.

The complete results are reported in Table 4(c) of the main paper. Using all valid steps increases mean success from 71.97% to 73.25% for Source and from 74.18% to 76.25% for TIF. Under the same update budget, observations and actions from later steps provide additional training information.

## G. Real-Robot Evaluation Details

This section supplements the real-robot candidate selection experiment in the main paper. We use a JAKA S5 manipulator to record candidates with different execution quality on three tabletop tasks. Recorded execution videos are used to evaluate reward ranking and audit-based selection.

## G.1. Tasks and Candidate Pools

The three tasks are pushing a block across a target line, placing exactly one cup, and relocating a specified cup. Each task contains 20 candidate pools. Each pool contains five physically executed trajectories, one designated as Source, yielding 60 pools and 300 execution trajectories in total.

The pools cover different degrees of task progress, completion quality, and execution deviations, providing alternative behaviors for the same task. Reward-only and TIF each select one trajectory per pool, giving 60 selections per method. Table S13 defines the tasks.

In the push task, the blue outlined box marks the block’s initial position, while the blue target line defines the crossing condition. Completion of the cup placement task also depends on compliance with the quantity constraint during execution. Manipulating two cups violates the instruction, even if the final image appears close to the target state.

## G.2. Video Evaluation and Candidate Selection

The real-robot experiment uses recorded execution videos as visual inputs. Executions of different quality provide candidate outcomes for comparing reward evaluation and audit selection in the JAKA setting.

Both methods use the same reward model. Reward-only selects the candidate with the highest predicted utility. TIF uses the same auditor and acceptance rule as in the main paper. It first fixes the candidate with the highest predicted utility, then checks whether its predicted advantage over Source exceeds 0.15 and its audit risk is at most 0.50. The candidate replaces Source when both conditions hold; otherwise, Source is retained. Model parameters and selection thresholds remain unchanged throughout the real-robot evaluation.

Table S13. Real-robot tasks, completion criteria, and aspects of execution quality.
<table><tr><td>Task</td><td>Goal and completion criteria</td><td>Execution quality</td></tr><tr><td>Push across line</td><td>Push the green block completely beyond the blue tar- get line and leave it in a stable state.</td><td>Whether the block fully crosses the line and whether execution collides with or displaces the obstacle.</td></tr><tr><td>ment</td><td>Exactly-one-cup place- Place exactly one cup upright and fully supported on the target platform beyond the blue line.</td><td>Compliance with the single-cup constraint, inter- action with additional cups, and placement sta- bility.</td></tr><tr><td></td><td>Specified-cup relocation Move the specified cup from the green platform to the yellow platform and place it upright.</td><td>Selection of the specified cup, attempts to ma- nipulate the wrong cup, and final placement sta- bility.</td></tr></table>

We inspect the corresponding execution videos and record two outcomes:

• Task completion: The execution satisfies the task goal and instruction constraints in Table S13.

• Clean completion: The task is completed with a stable final state and without collision, wrong- or extra-object interaction, or unnecessary recovery.

Clean completion is a subset of task completion. An execution that reaches the goal after a collision may count as task completion but not clean completion. The evaluation unit is the execution selected by each method from each candidate pool.

## G.3. Selection Outcomes and Example

Table 6 of the main paper summarizes the results across the three tasks. Among 60 selections, Reward-only and TIF select 45 and 47 trajectories that complete the task, including 35 and 42 clean completions, respectively. Task completion increases from 75.00% to 78.33%, and clean completion increases from 58.33% to 70.00%.

TIF selects three more clean completions for pushing, two more for single-cup placement, and two more for specified-cup relocation. For relocation, task completion remains at 16/20, while clean completion increases from 13/20 to 15/20, reflecting differences in execution quality beyond terminal task completion.

Figure 3 in the main paper shows a candidate pool from the push task. Table S14 lists its scores.

C2 has the highest predicted utility and an advantage of 0.21, exceeding the advantage margin. Both Reward-only and the rule using only reward and the advantage margin therefore select C2. Its audit risk is 0.72, exceeding the threshold of 0.50, so TIF retains Source C1.

Table S14. Scores for the five candidates in the push example shown in Fig. 3. Predicted advantage is relative to Source C1.
<table><tr><td>Candidate</td><td>Pred. utility</td><td>Audit risk</td><td>Advantage</td></tr><tr><td>C1: Source</td><td>.62</td><td>.14</td><td>0.00</td></tr><tr><td>C2</td><td>.83</td><td>.72</td><td>+0.21</td></tr><tr><td>C3</td><td>.72</td><td>.23</td><td>+0.10</td></tr><tr><td>C4</td><td>.56</td><td>.18</td><td>-0.06</td></tr><tr><td>C5</td><td>.66</td><td>.31</td><td>+0.04</td></tr></table>

During execution, both C1 and C2 push the block beyond the target line. C2 collides with and displaces the obstacle, whereas C1 completes the push without collision and leaves the block stable. This example shows how the audit gate changes the supervision acceptance decision after reward ranking.