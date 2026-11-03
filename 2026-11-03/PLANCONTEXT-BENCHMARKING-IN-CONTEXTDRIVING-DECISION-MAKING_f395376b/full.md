# PLANCONTEXT: BENCHMARKING IN-CONTEXTDRIVING DECISION-MAKING

Ruikai Li<sup>1,2</sup>, Naiyu Gao<sup>2</sup>, Hao Shan<sup>1</sup>, Yizhe Fan<sup>1</sup>, Yanhao Wu<sup>2</sup>, Han Jiang<sup>1</sup>, Yilong Ren<sup>1</sup>, Zhenning Li<sup>3</sup>, Haiyang Yu<sup>1</sup>, Yanhu Shan<sup>2</sup>, Rui Wu<sup>2</sup>, Hao Zhao<sup>4,5</sup>, Hongyang Li<sup>6</sup>, Zhiyong Cui<sup>1†</sup>

<sup>1</sup> Beihang University <sup>2</sup> Horizon Robotics <sup>3</sup> University of Macau <sup>4</sup> BAAI

<sup>5</sup> AIR, Tsinghua University <sup>6</sup> OpenDriveLab, The University of Hong Kong

<sup>†</sup>Corresponding author: zhiyongc@buaa.edu.cn Project:https://plancontext.github.io/

## ABSTRACT

Real-world driving requires context-aware decision-making. Existing end-to-end driving benchmarks primarily assess trajectory quality through safety, compliance, comfort, and driving progress, leaving context-dependent decision-making insufficiently characterized. To fill this gap, we present PlanContext, a benchmark for evaluating in-context driving decision-making. PlanContext constructs paired evaluations by varying high-level driving intentions while keeping the observed scene fixed, enabling a natural and controllable assessment of context-dependent decision-making. We further define six representative driving contexts and jointly evaluate contextual decision correctness, road compliance, and dynamic safety with class-balanced metrics. Experiments on 9,561 scene–condition instances from NAVSIM and 23 public end-to-end driving models show that high conventional driving-quality scores do not consistently correspond to strong route-level in-context driving decision-making. Further analysis shows that model behavior varies with decision timing, road structure, and traffic interaction depending on the decision required by the current context. Our results highlight in-context driving decision-making as an important complementary dimension for evaluating end-to-end driving systems.

![](images/2633f17a157889ffdb97d16e9e07c5a8cb25b3d3bf8c6571bc907ee55b452b34.jpg)

(a) A minimal intention change (from left to right) induces a collision-proneresponse at the intersection.  
![](images/54bb667b36f61f8c655e45275e68116349105b868b77b4d2f16415c3509412f6.jpg)  
(b) A minimal intention change (from straight to left) causes the vehicle to leave the drivable area  
Figure 1: PlanContext. Conventional driving metrics primarily evaluate trajectory quality in terms of safety, compliance, comfort, and driving progress. However, models with high PDMS can still produce context-inappropriate trajectories when navigation intentions change. PlanContext uses a natural and controllable navigation intervention to evaluate in-context driving decision-making.

## 1 INTRODUCTION

Autonomous driving has attracted growing attention from both academia and industry due to its potential to improve road safety and transportation efficiency (Chen et al., 2024; Janai et al., 2020). The community has developed a wide range of end-to-end driving models that integrate sensory observations, ego states, and high-level driving intentions to directly predict future trajectories or driving actions (Chitta et al., 2022; Hu et al., 2023; Zhou et al., 2026a). As model families and technical paradigms continue to diversify, reliable evaluation and fair comparison of driving performance have become increasingly important for measuring progress in the field.

However, comprehensively characterizing the decision-making capabilities of end-to-end driving models remains challenging. Existing benchmarks primarily assess trajectory quality through metrics such as collision avoidance, road compliance, comfort, and driving progress (Caesar et al., 2020; Jia et al., 2024; Cao et al., 2025; Zhang et al., 2026). While these metrics are effective for measuring whether a model produces safe and reasonable trajectories, a high-quality trajectory alone does not establish that the model has correctly interpreted the current driving context and made an appropriate decision. As shown in Fig. 1, models with high PDMS under the original navigation command can still respond inappropriately when the command changes. In Fig. 1(a), the vehicle has already started a left-turn maneuver, and a newly issued right-turn command should not trigger an immediate reversal that drives the vehicle toward a pedestrian. Similarly, in Fig. 1(b), when a turning instruction is given before the corresponding maneuver becomes locally feasible, the policy should continue to follow the current road rather than turn immediately and leave the drivable area. These cases high light that driving decisions should be made in accordance with the current driving context. A reliable autonomous driving policy should jointly consider contextual information such as the current vehicle state, road geometry, and high-level navigation intention to make an appropriate decision. Yet existing benchmarks offer limited insight into such in-context decision-making capability, allowing context-inappropriate decision failures to remain hidden behind strong trajectory-level performance.

To address this gap, we introduce PlanContext, a benchmark for systematically evaluating incontext driving decision-making in end-to-end autonomous driving. PlanContext keeps sensor observations and vehicle history fixed while applying controlled changes to high-level navigation intentions, enabling paired comparison of model decisions under the same environmental context. Based on road topology, target reachability, decision timing, and ongoing vehicle motion, we construct representative driving contexts and define context-specific requirements for whether an intention should be executed, deferred, rejected, or preserved. PlanContext jointly evaluates contextual decision correctness, road compliance, and dynamic safety, and further analyzes model behavior under different contextual factors. In this way, PlanContext extends end-to-end driving evaluation beyond trajectory quality toward a systematic assessment of intention–context–decision consistency.

Based on PlanContext, we construct 9,561 strictly filtered scene–condition instances from NAVSIM (Dauner et al., 2024) and evaluate 23 publicly available end-to-end driving models spanning three paradigms: vision/state-to-action, vision-language-action, and world-model-based action. Model rankings under conventional driving-quality evaluation differ substantially from those under route-level in-context evaluation. Across all 23 models, contextual success reaches only 22.46% at best when the driving context requires executing an immediately actionable alternative route. Moreover, no consistent performance trend emerges across model paradigms: VA, VLA, and WA models all exhibit substantial within-paradigm variation in in-context decision-making performance. Our key contributions are threefold:

• We introduce PlanContext, a benchmark for in-context driving decision-making with 9,561 curated scene–condition instances. It enables systematic evaluation of whether end-to-end models make appropriate decisions by jointly considering driving context and high-level intentions.

• PlanContext provides evaluation criteria and metrics for in-context driving decisionmaking across six scenario categories. The evaluation jointly assesses contextual decision correctness, road compliance, and dynamic safety, with separate metrics for route-level and lane-level performance.

• Our evaluation of 23 end-to-end driving models shows that conventional drivingquality metrics do not fully capture in-context decision-making capability. Further analyses reveal category-dependent performance patterns across decision timing, road structure, and traffic interactions.

## 2 RELATED WORK

End-to-End Autonomous Driving. End-to-end autonomous driving directly maps sensory observations and vehicle states to future trajectories or actions. Recent approaches span perception–planning architectures, direct trajectory regression, generative planners, vision–language–action models, and world-model-based driving systems (Jiang et al., 2023; Chitta et al., 2022; Liao et al., 2025; Zhou et al., 2026b; Wang et al., 2026a). Beyond sensory inputs, modern models increasingly incorporate navigation goals and structured scene information into decisionmaking. Recent work further shows that decision-relevant traffic elements, including traffic lights, road signs, and lane topology, consistently affect planning performance across diverse end-to-end paradigms (Zhang et al., 2026). These developments motivate evaluation beyond aggregate trajectory quality toward how models use contextual information when making driving decisions.

Evaluation of End-to-End Driving. End-to-end driving evaluation has evolved from open-loop trajectory similarity toward more comprehensive measurements of driving quality and robustness. NAVSIM (Dauner et al., 2024) introduces data-driven non-reactive simulation and evaluates predicted trajectories through collision, road compliance, comfort, time-to-collision, and driving progress. Closed-loop benchmarks such as Bench2Drive (Jia et al., 2024) further evaluate interactive driving performance, while NAVSIMv2 (Cao et al., 2025) augments logged data with synthetic observations to assess recovery from ego deviations. Fail2Drive (Gerstenecker et al., 2026) further examines closed-loop generalization by pairing distribution-shifted routes with in-distribution counterparts under changes in appearance, layout, and agent behavior. These benchmarks substantially improve the evaluation of trajectory quality, safety, and robustness. PlanContext provides a complementary evaluation perspective by examining whether driving decisions remain appropriate under different combinations of high-level intentions and environmental contexts, thereby revealing aspects of decision-making that are not captured by conventional driving-quality metrics.

Instruction-Conditioned and Intervention-Based Evaluation. High-level navigation commands help resolve behavioral ambiguity in end-to-end driving (Codevilla et al., 2018; Chitta et al., 2022). NavControl (Wu et al., 2025) constructs feasible alternative commands in NAVSIM and evaluates navigation compliance alongside driving quality. ICR-Drive (Hamid et al., 2026) assesses robustness to goal-preserving and goal-conflicting language perturbations on matched CARLA routes, using the original navigation objective as the evaluation reference. Beyond driving policies, What-If World (Cai et al., 2026) evaluates whether video world models produce the expected physical outcomes under paired prompts differing in a single physical variable. PlanContext targets contextdependent decision correctness through paired intention changes with observations and vehicle history fixed. Its criteria account for road topology, target reachability, decision timing, and ongoing motion, assessing both the semantic validity of individual predictions and whether paired predictions exhibit context-appropriate behavioral adjustment or preservation.

## 3 PLANCONTEXT: EVALUATING IN-CONTEXT DRIVING DECISION-MAKING

PlanContext evaluates in-context driving decision-making through paired navigation interventions and context-aware assessment. We first introduce the intervention protocol and the construction of representative driving contexts in Sections 3.1 and 3.2. Section 3.3 then defines the contextual decision evaluation criteria, followed by the class-balanced performance metrics in Section 3.4.

## 3.1 PAIRED NAVIGATION INTERVENTION

Given a driving scene $s _ { i } ,$ its original navigation intention $c _ { i } ^ { N }$ , and an alternative intention $c _ { i } ^ { F }$ , an evaluated model f predicts two future trajectories:

$$
T _ { i } ^ { N } = f _ { \theta } ( s _ { i } , c _ { i } ^ { N } ) , \qquad T _ { i } ^ { F } = f _ { \theta } ( s _ { i } , c _ { i } ^ { F } ) .\tag{1}
$$

Each trajectory contains eight future ego poses over a 4-second horizon. PlanContext keeps the sensor observations and vehicle history in $s _ { i }$ unchanged and only modifies the high-level navigation intention, yielding paired predictions under the same environmental context.

For each condition $x \in \{ N , F \}$ , let $\boldsymbol { \mathcal { A } } ( s _ { i } , c _ { i } ^ { x } )$ denote the set of behaviors that are semantically valid under scene $s _ { i }$ and intention $c _ { i } ^ { x }$ . The semantic-validity indicator is defined as

$$
S _ { i } ^ { x } = \mathbb { k } \mathbb { \textstyle [ T _ { i } ^ { x } \in \mathcal { A } ( s _ { i } , c _ { i } ^ { x } ) ] } , \qquad x \in \{ N , F \} .\tag{2}
$$

This formulation allows multiple valid future behaviors under the same intention and avoids treating a single recorded trajectory as the unique correct outcome.

![](images/719d4fe3792df4e761b101e7b05dda441859f900efe4818a77a0247886f4d9dc.jpg)  
Figure 2: Representative examples of the six PlanContext categories. Parentheses indicate the change in navigation intention.

The logged trajectory $T _ { i } ^ { G T }$ is used to characterize the natural road evolution, maneuver stage, and route-commitment state of the scene. These cues support the construction of the context-dependent valid behavior sets $\boldsymbol { A } ( s _ { i } , c _ { i } ^ { N } )$ and $\boldsymbol { A } ( s _ { i } , c _ { i } ^ { F } )$ . The paired relationship between $T _ { i } ^ { N }$ and $T _ { i } ^ { F }$ is evaluated separately in Section 3.3.

## 3.2 CONTEXT-AWARE SCENARIO CONSTRUCTION

The valid behavior sets introduced in Section 3.1 are determined by the driving context of each scene–condition instance. We characterize this context using road topology, target reachability, decision timing, and the ongoing motion state of the ego vehicle. Together, these factors determine how a high-level driving intention can be appropriately realized in the current scene.

PlanContext organizes the resulting instances into four contextual requirements: follow, wait, reject, and preserve. Follow applies when the alternative target is currently reachable and should be executed within the evaluation horizon. Wait describes cases where the target becomes feasible in the near future while the current state still favors maintaining the existing route. Reject covers locally unreachable targets or cases where the vehicle has already passed the appro-

Table 1: PlanContext scenario categories and contextual requirements.
<table><tr><td>Category</td><td>Context condition</td><td>Requirement</td></tr><tr><td>A</td><td>Target route reachable within 2 s</td><td>Follow</td></tr><tr><td>B1</td><td>Target route reachable in 4.5–8 s</td><td>Wait</td></tr><tr><td>B2</td><td>No locally feasible target</td><td>Reject</td></tr><tr><td>C</td><td>Ego beyond route commitment point</td><td>Reject</td></tr><tr><td>L</td><td>Adjacent target lane geometrically and topologically reachable</td><td>Follow</td></tr><tr><td>LC</td><td>Lateral maneuver already in progress</td><td>Preserve</td></tr></table>

priate decision point. Preserve applies when an ongoing maneuver should remain temporally consistent after the intention changes. We instantiate these requirements with six complementary scenario categories, summarized in Table 1 and illustrated in Fig. 2.

The route-level categories A, B1, B2, and C capture different stages of navigation feasibility and route commitment. A represents an immediately actionable route change, while B1 contains targets that become feasible beyond the current decision window. B2 captures locally unsupported intentions, and C describes scenes in which the ego vehicle has already committed to the original route. The lane-level categories complement these route-level contexts: L evaluates feasible adjacent-lane transitions, while LC evaluates the continuity of an ongoing lateral maneuver.

For A, B1, B2, C, and L, each category defines the semantic behavior sets $\boldsymbol { A } ( s _ { i } , c _ { i } ^ { N } )$ and $\boldsymbol { \mathcal { A } } ( s _ { i } , c _ { i } ^ { F } )$ used to compute $S _ { i } ^ { N }$ and $S _ { i } ^ { F }$ in Section 3.1. LC instead defines a continuity requirement for the ongoing lateral maneuver, as detailed in Section 3.3. We retain only scene–condition instances with sufficiently clear contextual evidence for reliable category assignment. Detailed scenario mining, filtering, and category assignment criteria are provided in Sec. A.

To further assess benchmark validity, three reviewers with backgrounds in autonomous driving independently inspected a randomly sampled subset of 500 scene–condition instances and 11,500 model-instance evaluations from all 23 evaluated models after the evaluation rules were finalized. The reviewers were blinded to the automated evaluation results during independent assessment. Following discussion, approximately 0.7% of the inspected model-instance evaluations were judged inconsistent with the automated evaluation outcomes. Detailed findings are provided in Appendix D.9.

## 3.3 CONTEXTUAL DECISION EVALUATION

Given the contextual requirement assigned in Section 3.2, PlanContext evaluates each scene– condition instance through three components: contextual decision correctness, road compliance, and dynamic safety.

Paired behavioral correctness. We first characterize how the prediction changes when the navigation intention changes. The original and alternative trajectories are projected into a road-relative coordinate system, where longitudinal progress, lateral displacement, and heading deviation are measured with respect to the local road geometry. The trajectories are aligned by road progress to reduce the influence of speed differences on behavioral comparison.

Let $k _ { i } ~ \in ~ \{ A , B 1 , B 2 , C , L , L C \}$ denote the PlanContext category of instance i. We define a category-specific paired behavior function

$$
P _ { i } = \Phi _ { k _ { i } } \left( T _ { i } ^ { N } , T _ { i } ^ { F } , s _ { i } \right) ,\tag{3}
$$

where $\Phi _ { k _ { i } }$ evaluates whether the behavioral change between the original and alternative predictions satisfies the requirement of category $k _ { i }$

For the follow categories A and $\mathrm { L } , P _ { i }$ requires an identifiable path change toward the alternative target. For B1, B2, and C, it requires the alternative intention to avoid inducing an inappropriate target-directed deviation. LC is treated separately because its objective is to preserve the continuity of an ongoing lateral maneuver rather than to evaluate two independent semantic outcomes. We therefore assign LC predictions a continuity status $P _ { i } ^ { L C } \in$ {correct, incorrect, uncertain}. Detailed category-specific behavioral criteria and trajectory comparison rules are provided in Sec. B.

For A, B1, B2, C, and $\mathrm { { L , } }$ contextual decision correctness combines the semantic validity of both predictions with their paired behavioral relationship:

$$
\begin{array} { r } { C _ { i } = \left\{ \begin{array} { l l } { S _ { i } ^ { N } \wedge S _ { i } ^ { F } \wedge P _ { i } , } & { k _ { i } \in \{ A , B 1 , B 2 , C , L \} , } \\ { 1 , } & { k _ { i } = L C , P _ { i } ^ { L C } = \mathrm { c o r r e c t } , } \\ { 0 , } & { k _ { i } = L C , P _ { i } ^ { L C } = \mathrm { i n c o r r e c t } . } \end{array} \right. } \end{array}\tag{4}
$$

LC instances with ${ \cal P } _ { i } ^ { L C } =$ uncertain are excluded from subsequent success-rate aggregation.

Road compliance. Road compliance is evaluated using the full ego-vehicle footprint. Let $B ( T _ { i , t } ^ { F } )$ denote the vehicle footprint at future step t and $\Omega _ { i } ^ { F }$ the road region permitted by the corresponding context. The road-compliance indicator is

$$
R _ { i } ^ { F } = \prod _ { t = 1 } ^ { 8 } \mathcal { k } \left[ \mathcal { B } ( T _ { i , t } ^ { F } ) \subseteq \Omega _ { i } ^ { F } \right] .\tag{5}
$$

Dynamic safety. Dynamic safety is evaluated using the NAVSIM safety criteria, including collision, drivable-area compliance, wrong-way driving, traffic-light violations, time-to-collision, and comfort. We denote the resulting safety indicator as $\overline { { E _ { i } ^ { F } } }$

The final contextual success for instance i is defined as

$$
q _ { i } = C _ { i } \land R _ { i } ^ { F } \land E _ { i } ^ { F } .\tag{6}
$$

Thus, a successful instance requires a context-consistent decision response, road-compliant motion, and dynamic safety. For A, B1, B2, C, and $\mathrm { { L , } }$ contextual correctness additionally requires semantic validity under both navigation conditions.

## 3.4 CLASS-BALANCED PLANCONTEXT METRICS

The six PlanContext categories are highly imbalanced in sample size, making a direct micro-average sensitive to the empirical category distribution. We therefore first compute the contextual success rate independently for each category:

$$
S _ { k } = \frac { 1 } { | { \cal D } _ { k } | } \sum _ { i \in { \cal D } _ { k } } q _ { i } , \qquad k \in \{ A , B 1 , B 2 , C , L , L C \} ,\tag{7}
$$

where $\mathcal { D } _ { k }$ denotes the set of valid scene–condition instances used for metric aggregation in category k. LC instances labeled as uncertain are excluded.

We aggregate the scores of A, B1, B2, and C to evaluate route-level in-context driving decisionmaking. The scores of L and LC are reported separately to evaluate lane-level in-context driving decision-making.

The Navigation Action Score (NAS) is the contextual success rate in category A, where the alternative route is immediately actionable. The Context Preservation Score (CPS) measures contextual success when the alternative navigation intention should be deferred or rejected, averaged across categories B1, B2, and C:

$$
\mathrm { C P S } = \frac { S _ { B 1 } + S _ { B 2 } + S _ { C } } { 3 } .\tag{8}
$$

We define the Navigation Context Score (NCS) as the geometric mean of NAS and CPS to summarize route-level in-context driving decision-making:

$$
\mathrm { N C S } = \sqrt { \mathrm { N A S } \cdot \mathrm { C P S } } .\tag{9}
$$

The geometric mean penalizes imbalanced performance across NAS and CPS.

For lane-level evaluation, we report the Lane Adaptation Score (LAS) and Maneuver Continuity Score (MCS), defined as the contextual success rates in categories L and LC, respectively. We report all five metrics together with the six category-wise scores.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

Benchmark. PlanContext is constructed from the NAVSIM navtest split. After scenario mining and context-aware filtering, the benchmark contains 9,561 scene–condition instances from 5,911 unique physical scenes. The six contextual categories include 334 A, 251 B1, 5,285 B2, 580 C, 2,716 L, and 395 LC instances. For each retained instance, the sensor observations and vehicle history are fixed, while the evaluated model is queried under both the original navigation intention and the corresponding alternative intention.

Models. We evaluate 23 publicly available end-to-end driving models covering three representative paradigms. Vision/state-to-action (VA) models directly predict driving trajectories from sensory or state inputs, vision-language-action (VLA) models incorporate vision-language representations into planning, and world-model-based action (WA) models explicitly predict or encode future scene dynamics. All models are evaluated using publicly released or official assets, with navigation conditions explicitly involved in inference. Their outputs are standardized as eight future ego poses over a 4-second horizon.

Evaluation Protocol. For each model, we report its conventional driving performance on the full NAVSIM navtest split using PDMS. To enable a scene-matched comparison with PlanContext, we additionally compute matched PDMS (mPDMS) under the original navigation condition on the same 5,911 physical scenes selected by PlanContext. Each physical scene contributes once to mPDMS, even when it is associated with multiple alternative navigation conditions. PlanContext then performs paired inference under the original and alternative navigation intentions on the 9,561 scene–condition instances. We report NAS, CPS, and NCS for route-level in-context driving decision-making, and LAS and MCS for lane-level evaluation. We additionally report contextual success rates for all six categories.

• PDMS denotes conventional driving quality on the full NAVSIM v1 navtest split.

Table 2: Overall comparison of 23 end-to-end driving models on PlanContext.
<table><tr><td>Model</td><td>Venue</td><td>PDMS mPDMS NAS</td><td></td><td>CPS NCS LAS MCS</td></tr><tr><td colspan="5">VA Models</td></tr><tr><td>DrivoR (Kirby et al., 2026)</td><td>CVPR 2026</td><td>93.96</td><td></td><td>83.5619.76 61.23 34.780.00 92.62</td></tr><tr><td>GoalFlow (Xing et al., 2025)</td><td>CVPR 2025</td><td>88.31</td><td></td><td>79.82 15.57 63.89 31.540.0787.82</td></tr><tr><td>TransFuser (Chitta et al., 2022)</td><td>IEEE TPAMI</td><td>83.39</td><td></td><td>74.1613.77 52.98 27.010.15 73.59</td></tr><tr><td>SparseDriveV2 (Sun et al., 2026)</td><td>ECCV 2026</td><td>92.22</td><td></td><td>81.2212.5756.25 26.600.15 93.64</td></tr><tr><td>DiffusionDrive (Liao et al., 2025)</td><td>CVPR 2025</td><td>88.05</td><td></td><td>79.80 3.29 65.44 14.68 0.00 92.11</td></tr><tr><td>StateMLP (Dauner et al., 2024)</td><td>NeurIPS 2024</td><td>65.52</td><td></td><td>56.9515.57 8.6811.638.73 31.55</td></tr><tr><td>CLOVER (Ang et al., 2026)</td><td>Arxiv 2026</td><td>94.48</td><td>94.35</td><td>0.60 57.355.860.22 85.82</td></tr><tr><td>DiffusionDriveV2 (Zou et al., 2025)</td><td>Arxiv 2026</td><td>90.95</td><td>90.89 0.00 49.51</td><td>0.00 0.07 82.19</td></tr><tr><td colspan="5">VLA Models</td></tr><tr><td>OneVL (Lu et al., 2026)</td><td>Arxiv 2026</td><td>88.92</td><td>80.79 22.46 68.79 39.300.00 91.88</td><td></td></tr><tr><td>ChainFlow-VLA (Wang et al., 2026b) Arxiv 2026</td><td></td><td>95.21</td><td>83.14 20.96 58.88 35.130.04 94.42</td><td></td></tr><tr><td>ReCogDrive-IL (Xiong et al., 2026)</td><td>ICLR 2026</td><td>86.00</td><td></td><td>78.59 21.26 55.39 34.310.00 82.56</td></tr><tr><td>ReCogDrive-RL (Xiong et al., 2026)</td><td>ICLR 2026</td><td>90.46</td><td></td><td>78.9318.56 53.93 31.640.0784.83</td></tr><tr><td>NoRD (Rawal et al., 2026)</td><td>CVPR 2026</td><td>85.69</td><td></td><td>75.7517.66 49.31 29.510.18 82.69</td></tr><tr><td>AutoVLA (Zhou et al., 2026b)</td><td>NeurIPS 2025</td><td>92.06</td><td></td><td>80.9212.28 59.76 27.080.11 86.67</td></tr><tr><td>DriveVLA-W0 (Li et al., 2026)</td><td>ICLR 2026</td><td>87.02</td><td>78.54 4.49 63.01 16.82 0.00 91.88</td><td></td></tr><tr><td>DP-VLA (Zheng et al., 2026)</td><td>Arxiv 2026</td><td>88.56</td><td>88.41</td><td>1.80 66.09 10.900.00 86.58</td></tr><tr><td>Curious-VLA (Chen et al., 2026)</td><td>CVPR 2026</td><td>89.20</td><td>89.32</td><td>0.60 60.62 6.02 0.15 87.95</td></tr><tr><td>WAM-Flow (Xu et al., 2026)</td><td>CVPR 2026</td><td>90.27</td><td>90.20 0.60 57.27</td><td>5.86 0.00 88.27</td></tr><tr><td colspan="5">WA Models</td></tr><tr><td>SUV (Yuan et al., 2026)</td><td>Arxiv 2026</td><td>90.87</td><td></td><td></td></tr><tr><td>WoTE (Li et al., 2025)</td><td>ICCV 2025</td><td>88.22</td><td>79.2113.4760.36 28.520.0491.88</td><td>81.6314.6769.9432.030.00 94.68</td></tr><tr><td>SimWAM (Zhao et al., 2026)</td><td>Arxiv 2026</td><td>90.97</td><td>90.76 0.30 56.74 4.12 0.00 88.83</td><td></td></tr><tr><td>Drive-JEPA (Wang et al., 2026a)</td><td>Arxiv 2026</td><td>93.69</td><td>94.18 0.00 52.37 0.00 0.33 85.71</td><td></td></tr><tr><td>DynVLA Stage 2 (Shang et al., 2026) ICML 2026</td><td></td><td>87.20</td><td></td><td>86.53 0.00 64.54 0.00 0.00 88.61</td></tr></table>

• mPDMS denotes PDMS recomputed under the original navigation condition on the 5,911 physical scenes selected by PlanContext.  
• NAS is the contextual success rate in category A.  
• CPS is the macro-average of the success rates in categories B1, B2, and C.  
• NCS is the geometric mean of NAS and CPS and summarizes route-level in-context driving decision-making.  
• LAS and MCS are the contextual success rates in L and LC, respectively, for lane-level evaluation.  
• The 95% confidence intervals for NCS are reported in Table 9.

Statistical Analysis. We estimate uncertainty using 10,000 bootstrap resamples over the 5,911 unique scene tokens. All scene–condition instances and model predictions associated with a sampled scene are retained together to preserve the paired evaluation structure. We report percentile-based 95% confidence intervals for NCS. Pairwise model comparisons are computed on matched bootstrap samples, with Benjamini–Hochberg correction applied across all 253 pairwise comparisons.

## 4.2 OVERALL EVALUATION

Table 2 presents the overall results of the 23 evaluated models. Figures 3a–3e compare conventional driving quality with route-level and lane-level in-context performance. mPDMS evaluates driving quality under the original navigation condition, while NCS evaluates in-context driving decisionmaking through paired navigation interventions. The Spearman rank correlation between mPDMS and NCS is −0.542, indicating substantial differences in model rankings across the two evaluations. Confidence intervals, pairwise comparisons, and sensitivity analyses of model rankings are provided in Appendices D.3 and D.7.

Figure 3 further reveals two interesting patterns: (1) High mPDMS does not consistently corre spond to high NAS. All models with mPDMS above 85 achieve NAS of at most 1.80, while OneVL achieves the highest NAS of 22.46 with an mPDMS of 80.79. (2) Different model architecture paradigms do not exhibit a consistent performance trend on PlanContext. VA, VLA, and WA models all show substantial performance variation within each paradigm, suggesting that the architectural paradigm alone does not consistently explain in-context driving decision-making performance.

## 4.3 MODEL CAPABILITY ANALYSIS

![](images/7109d1db9de0bc3dc1f81887c12921a6d4e77907ff9942ff55b8fe40807cdc99.jpg)  
(a) mPDMS vs. NAS

![](images/635ce6b9cb412b319e1830adcb0717f86be4aa094729356c6a7d84a63fabb28e.jpg)  
(b) mPDMS vs. CPS

![](images/a7f85f37b53cad1bfe994b140b97052970bc2074b9ac3cf5acb6d651ecc173d8.jpg)  
(c) mPDMS vs. NCS

![](images/33ea3918575b8c05041762d44de7bc8bc81b9e01c43a47c4a8e7fef009396779.jpg)  
(d) mPDMS vs. LAS

![](images/dc89a565a765a98d0fb81dd7006da2e8010398c8b39286cdfa96926c263de27a.jpg)  
(e) mPDMS vs. MCS

![](images/08c49d40f7c7f81e0f1776b10a988a9523e8337b0bdf1f624708ce8af7362c42.jpg)  
(f) NAS vs. CPS  
Figure 3: Driving quality and in-context performance across 23 end-to-end driving models.

Table 3 shows substantial variation in contextual success across the six categories. At the route level, the mean NAS is 10.01, compared with a mean CPS of 57.06. NAS is lower than CPS for 22 of the 23 models, indicating that current models achieve limited success when alternative routes are immediately actionable.

Figure 3f further illustrates differences in model performance across NAS and CPS. DynVLA Stage 2 achieves a CPS of 64.54 but has zero NAS. StateMLP achieves a NAS of 15.57 and a CPS of 8.68.

Table 3: Contextual success across Plan-Context categories.
<table><tr><td>Category</td><td>Instances</td><td>Success (%)</td></tr><tr><td>A</td><td>334</td><td>10.01</td></tr><tr><td>B1</td><td>251</td><td>62.91</td></tr><tr><td>B2</td><td>5,285</td><td>73.28</td></tr><tr><td>C</td><td>580</td><td>34.98</td></tr><tr><td>L</td><td>2,716</td><td>0.45</td></tr><tr><td>LC</td><td>395</td><td>85.51</td></tr></table>

Success is averaged over all 23 evaluated models.

At the lane level, Figures 3d and 3e show limited success in executing feasible alternative lane intentions alongside substantially higher success in ongoing-maneuver continuity. LAS is below 0.34 for 22 of the 23 models, including ten with zero scores. StateMLP achieves the highest LAS of 8.73. The consistently low LAS across models identifies the execution of feasible alternative lane intentions as a common failure mode in lane-level in-context driving decision-making.

## 4.4 CONTEXT FACTOR ANALYSIS

We further investigate how in-context driving performance varies with environmental context under different decision requirements. All analyses use the contextual success indicator $q _ { i }$ defined in Section 3.3 and aggregate predictions from all 23 evaluated models. All comparisons are performed within the same PlanContext category. We group contextual factors into scene-level factors, including decision timing, road geometry, and topological evidence, and dynamic factors describing traffic interaction.

Scene context. Table 4 reveals substantially different performance patterns across PlanContext categories. For category A, contextual success decreases from 15.59% under low decision urgency to 3.22% under high urgency. B1, B2, L, and LC exhibit smaller variations across urgency levels. Geometric salience exhibits no consistent monotonic trend across categories. B1 reaches its highest performance at the medium level, C shows higher performance at greater salience, whereas LC shows the opposite tendency. Topological evidence provides a clearer distinction. Stronger targetbranch directional evidence is associated with 13.15 and 35.67 percentage-point higher contextual success for A and B1, respectively, whereas B2 and C show 13.45 and 20.77 percentage-point lower success under their corresponding strong structural cues. These results show that road context pro vides different forms of decision evidence depending on the behavioral requirement.

Table 4: Associations between scene context factors and PlanContext performance.
<table><tr><td rowspan="2">Category</td><td colspan="3">Decision Urgency</td><td colspan="3">Geometric Salience</td><td rowspan="2">Topological Evidence ∆q [95% CI]</td></tr><tr><td>Low</td><td>Medium</td><td>High</td><td>Low</td><td>Medium</td><td>High</td></tr><tr><td>A</td><td>15.59</td><td>11.28</td><td>3.22</td><td>6.66</td><td>12.34</td><td>11.02</td><td>+13.15* [10.62, 15.72]</td></tr><tr><td>B1</td><td>64.22</td><td>61.90</td><td>62.63</td><td>55.74</td><td>77.43</td><td>55.49</td><td>+35.67* [21.22, 49.46]</td></tr><tr><td>B2</td><td>73.50</td><td>75.06</td><td>71.29</td><td>73.10</td><td>75.36</td><td>71.39</td><td>-13.45* [-15.16, -11.78]</td></tr><tr><td>C</td><td>23.74</td><td>43.39</td><td>37.79</td><td>29.96</td><td>36.09</td><td>38.86</td><td>-20.77* [-28.05, -13.39]</td></tr><tr><td>L</td><td>0.49</td><td>0.43</td><td>0.42</td><td>0.46</td><td>0.43</td><td>0.46</td><td>+0.08 [-0.07, 0.22]</td></tr><tr><td>LC</td><td>87.72</td><td>84.21</td><td>84.96</td><td>88.33</td><td>84.97</td><td>83.61</td><td>-2.51 [-7.62, 3.13]</td></tr></table>

• Decision Urgency and Geometric Salience report contextual success (%) under within-category Low/Medium/High stratification.  
• Topological Evidence reports Strong-minus-Weak contextual success.  
• <sup>∗</sup> indicates that the 95% scene-token bootstrap confidence interval excludes zero.  
• Detailed evidence definitions are provided in the Appendix.

Table 5: Associations between traffic interaction factors and PlanContext performance.
<table><tr><td>Category</td><td>Lead Vehicle</td><td>TTC Risk</td><td>Target Occupancy</td><td>Crossing Conflict</td><td>Traffic Density</td></tr><tr><td>A</td><td>-4.88</td><td>-9.08*</td><td>-9.94*</td><td>-0.18</td><td>+5.31*</td></tr><tr><td>B1</td><td>+17.47</td><td>-5.15</td><td>-6.00</td><td>+12.39</td><td>-0.88</td></tr><tr><td>B2</td><td>+9.05*</td><td>+6.29*</td><td>+5.54*</td><td>-4.45*</td><td>+12.10*</td></tr><tr><td>C</td><td>+3.58</td><td>-10.36</td><td>-10.02*</td><td>-11.51*</td><td>-9.75*</td></tr><tr><td>L</td><td>-0.50*</td><td>-0.26*</td><td>-0.39*</td><td>+0.20</td><td>-0.13*</td></tr><tr><td>LC</td><td>-7.62*</td><td>-8.62*</td><td>-6.88*</td><td>-9.56*</td><td>-7.82*</td></tr></table>

• Values report changes in contextual success (pp), computed as comparison minus reference.  
• <sup>∗</sup> indicates that the 95% scene-token bootstrap confidence interval excludes zero.

Traffic interaction. Table 5 shows similarly category-dependent associations between dynamic traffic conditions and contextual success. For Lead Vehicle and TTC, we compare ≤ 10 m with > 25 m and < 3 s with ≥ 6 s, respectively. Target Occupancy and Crossing Conflict compare ≥ 2 actors with 0 actors, while Traffic Density compares High with Low. A shows the largest performance differences across TTC risk and target-side occupancy, while C shows lower contextual success under greater target-side occupancy and crossing conflicts. LC exhibits consistent performance degradation across all five traffic factors, with lower success under more demanding dynamic conditions. For B2, closer lead vehicles, higher TTC risk, and greater target-side occupancy are associated with higher contextual success. Success increases by 12.10 percentage points from lowto high-density traffic, while crossing conflicts are associated with a decrease of 4.45 percentage points. These associations differ from those observed in LC, where contextual success decreases across all five traffic factors.

## 5 CONCLUSION

We present PlanContext, a benchmark for evaluating in-context driving decision-making through paired navigation interventions under fixed observations and vehicle history. Across 9,561 scene– condition instances and 23 end-to-end driving models, we find that strong conventional driving quality does not consistently correspond to strong in-context decision-making. Our analyses further show that model performance varies across different decision requirements and environmental contexts. These results indicate that in-context driving decision-making captures a capability dimension not fully reflected by conventional driving-quality metrics. By evaluating intention–context– decision consistency across route-level and lane-level scenarios, PlanContext provides a structured benchmark for assessing how end-to-end driving models respond to high-level intentions under different driving contexts.

## AI USE STATEMENT

Generative AI tools were used to assist with manuscript drafting and revision, translation, and consistency checks of reported numerical results. The authors reviewed the AI-assisted content, checked factual claims against the underlying experimental results and cited sources, and made the final decisions on the manuscript. The authors take responsibility for the accuracy and integrity of all content in this work.

## ETHICS STATEMENT

PlanContext evaluates in-context driving decision-making using existing driving scenes from NAVSIM (Dauner et al., 2024) and publicly released or official model assets. All experiments are conducted offline, without interventions on real vehicles or road users. The human validation procedure involves expert inspection of existing scenes and model predictions, as described in Appendix D.9. Responsible reuse of the benchmark requires adherence to the source data licenses and protection of privacy in driving-scene imagery.

## REPRODUCIBILITY STATEMENT

The paired navigation intervention protocol and contextual evaluation framework are specified in Section 3. Appendices A, B, and C detail scenario construction and filtering, category-specific evaluation rules, geometric thresholds, safety criteria, and metric aggregation. The experimental setup and model evaluation protocol are described in Section 4.1 and Appendix D.2. The scene-level bootstrap procedure and metric robustness analyses are documented in Appendices D.3 and D.7. These specifications support consistent reconstruction of the benchmark and its evaluation procedure. We plan to publicly release the PlanContext benchmark toolkit, including evaluation code and supporting documentation.

## REFERENCES

Sining Ang, Yuguang Yang, Canyu Chen, and Yan Wang. Clover: Closed-loop value estimation and ranking for end-to-end autonomous driving planning. arXiv preprint arXiv:2605.15120, 2026.

Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In 2020 IEEE/CVF conference on computer vision and pattern recognition (CVPR), pp. 11618–11628. IEEE, 2020.

Kunlin Cai, Rui Song, Jinghuai Zhang, Kaiyuan Zhang, Pranav Bodapati, Alicia Yu, Fnu Suya, Mohammad Rostami, Jiaqi Ma, and Yuan Tian. What-if world: A causal benchmark for general world models in embodied scenarios. arXiv preprint arXiv:2605.27589, 2026.

Wei Cao, Marcel Hallgarten, Tianyu Li, Daniel Dauner, Xunjiang Gu, Caojun Wang, Yakov Miron, Marco Aiello, Hongyang Li, Igor Gilitschenski, et al. Pseudo-simulation for autonomous driving. arXiv preprint arXiv:2506.04218, 2025.

Canyu Chen, Yuguang Yang, Zhewen Tan, Yizhi Wang, Ruiyi Zhan, Haiyan Liu, Xuanyao Mao, Jason Bao, Xinyue Tang, Linlin Yang, et al. Devil is in narrow policy: Unleashing exploration in driving vla models. arXiv preprint arXiv:2603.06049, 2026.

Li Chen, Penghao Wu, Kashyap Chitta, Bernhard Jaeger, Andreas Geiger, and Hongyang Li. Endto-end autonomous driving: Challenges and frontiers. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(12):10164–10183, 2024.

Kashyap Chitta, Aditya Prakash, Bernhard Jaeger, Zehao Yu, Katrin Renz, and Andreas Geiger. Transfuser: Imitation with transformer-based sensor fusion for autonomous driving. IEEE transactions on pattern analysis and machine intelligence, 45(11):12878–12895, 2022.

Felipe Codevilla, Matthias Muller, Antonio L ¨ opez, Vladlen Koltun, and Alexey Dosovitskiy. End-´ to-end driving via conditional imitation learning. In 2018 IEEE international conference on robotics and automation (ICRA), pp. 4693–4700. IEEE, 2018.

Daniel Dauner, Marcel Hallgarten, Tianyu Li, Xinshuo Weng, Zhiyu Huang, Zetong Yang, Hongyang Li, Igor Gilitschenski, Boris Ivanovic, Marco Pavone, et al. Navsim: Data-driven non-reactive autonomous vehicle simulation and benchmarking. Advances in Neural Information Processing Systems, 37:28706–28719, 2024.

Simon Gerstenecker, Andreas Geiger, and Katrin Renz. Fail2drive: Benchmarking closed-loop driving generalization. arXiv preprint arXiv:2604.08535, 2026.

Kaiser Hamid, Can Cui, and Nade Liang. Icr-drive: Instruction counterfactual robustness for endto-end language-driven autonomous driving. arXiv preprint arXiv:2604.05378, 2026.

Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, et al. Planning-oriented autonomous driving. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 17853–17862. IEEE, 2023.

Joel Janai, Fatma Guney, Aseem Behl, and Andreas Geiger. Computer vision for autonomous vehi-¨ cles: Problems, datasets and state of the art. Foundations and Trends in Computer Graphics and Vision, 12(1-3):1–308, 2020.

Xiaosong Jia, Zhenjie Yang, Qifeng Li, Zhiyuan Zhang, and Junchi Yan. Bench2drive: Towards multi-ability benchmarking of closed-loop end-to-end autonomous driving. Advances in Neural Information Processing Systems, 37:819–844, 2024.

Bo Jiang, Shaoyu Chen, Qing Xu, Bencheng Liao, Jiajie Chen, Helong Zhou, Qian Zhang, Wenyu Liu, Chang Huang, and Xinggang Wang. Vad: Vectorized scene representation for efficient autonomous driving. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 8306–8316. IEEE, 2023.

Ellington Kirby, Alexandre Boulch, Yihong Xu, Yuan Yin, Gilles Puy, Eloi Zablocki, Andrei Bursuc,<sup>´</sup> Spyros Gidaris, Renaud Marlet, Florent Bartoccioni, et al. Driving on registers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 32058–32069, 2026.

Yingyan Li, Yuqi Wang, Yang Liu, Jiawei He, Lue Fan, and Zhaoxiang Zhang. End-to-end driving with online trajectory evaluation via bev world model. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 27137–27146. IEEE, 2025.

Yingyan Li, Shuyao Shang, Weisong Liu, Bing Zhan, Haochen Wang, Yuqi Wang, Yuntao Chen, Xiaoman Wang, Yasong An, Chufeng Tang, et al. Drivevla-w0: World models amplify data scaling law in autonomous driving. In International Conference on Learning Representations, volume 2026, pp. 7890–7911, 2026.

Bencheng Liao, Shaoyu Chen, Haoran Yin, Bo Jiang, Cheng Wang, Sixu Yan, Xinbang Zhang, Xiangyu Li, Ying Zhang, Qian Zhang, et al. Diffusiondrive: Truncated diffusion model for endto-end autonomous driving. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 12037–12047. IEEE, 2025.

Jinghui Lu, Jiayi Guan, Zhijian Huang, Jinlong Li, Guang Li, Lingdong Kong, Yingyan Li, Han Wang, Shaoqing Xu, Yuechen Luo, et al. Onevl: One-step latent reasoning and planning with vision-language explanation. arXiv e-prints, pp. arXiv–2604, 2026.

Ishaan Rawal, Shubh Gupta, Yihan Hu, and Wei Zhan. Nord: A data-efficient vision-languageaction model that drives without reasoning. arXiv preprint arXiv:2602.21172, 2026.

Shuyao Shang, Bing Zhan, Yunfei Yan, Yuqi Wang, Yingyan Li, Yasong An, Xiaoman Wang, Jierui Liu, Lu Hou, Lue Fan, et al. Dynvla: Learning world dynamics for action reasoning in autonomous driving. arXiv preprint arXiv:2603.11041, 2026.

Wenchao Sun, Xuewu Lin, Keyu Chen, Zixiang Pei, Xiang Li, Yining Shi, and Sifa Zheng. Sparsedrivev2: Scoring is all you need for end-to-end autonomous driving. In European Conference on Computer Vision, pp. 446–463. Springer, 2026.

Linhan Wang, Zichong Yang, Chen Bai, Guoxiang Zhang, Xiaotong Liu, Xiaoyin Zheng, Xiao-Xiao Long, Chang-Tien Lu, and Cheng Lu. Drive-jepa: Video jepa meets multimodal trajectory distillation for end-to-end driving. arXiv preprint arXiv:2601.22032, 2026a.

Xiyang Wang, Xinlin Wang, Tingguang Zhou, Gong Chen, Xingtai Gui, Zhi Xu, Xiaolei Wu, Feiyang Tan, Hangning Zhou, and Mu Yang. Chainflow-vla: Causal flow planning with visionlanguage models. arXiv preprint arXiv:2605.23270, 2026b.

Hanfeng Wu, Marlon Steiner, Michael Schmidt, Alvaro Marcos-Ramiro, and Christoph Stiller. Closing the navigation compliance gap in end-to-end autonomous driving. arXiv preprint arXiv:2512.10660, 2025.

Zebin Xing, Xingyu Zhang, Yang Hu, Bo Jiang, Tong He, Qian Zhang, Xiaoxiao Long, and Wei Yin. Goalflow: Goal-driven flow matching for multimodal trajectories generation in end-to-end autonomous driving. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1602–1611. IEEE, 2025.

Kaixin Xiong, Xiangyu Guo, Fang Li, Sixu Yan, Gangwei Xu, Lijun Zhou, Long Chen, Haiyang Sun, Bing Wang, Kun Ma, et al. Recogdrive: A reinforced cognitive framework for end-to-end autonomous driving. In International Conference on Learning Representations, volume 2026, pp. 157518–157556, 2026.

Yifang Xu, Jiahao Cui, Zhihao Zhu, Hanlin Shang, Shan Luan, Mingwang Xu, Feipeng Cai, Neng Zhang, Yaoyi Li, Jia Cai, et al. Wam-flow: Parallel coarse-to-fine motion planning via discrete flow matching for autonomous driving. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 24918–24928, 2026.

Yibo Yuan, Jiacheng Fu, Jiangtong Zhu, Yi Li, Jianhua Han, Meng Tian, Zhuohan Liu, Zhiwei Xiong, Hang Xu, Jianwu Fang, et al. Suv: Future scene understanding as video generation for end-to-end driving. arXiv preprint arXiv:2608.03084, 2026.

Zongzheng Zhang, Jijun Wang, Saining Zhang, Shuo Wang, Yiru Wang, Hai Yang, Yang Chen, Yuwen Heng, Hao Sun, Anqing Jiang, et al. Plug-and-play traffic element awareness for end-toend autonomous driving. arXiv preprint arXiv:2608.18035, 2026.

Zongchuang Zhao, Xin Zhou, Tianyang Xu, Zhengyang Sun, Kaixuan Zhou, Honglin Li, Dingkang Liang, and Xiang Bai. Simwam: A simple world action model for end-to-end autonomous driving. arXiv preprint arXiv:2608.07468, 2026.

Yinan Zheng, Tianyi Tan, Bin Huang, Enguang Liu, Ruiming Liang, Jianlin Zhang, Jianwei Cui, Guang Chen, Kun Ma, Hangjun Ye, et al. Unleashing the potential of diffusion models for endto-end autonomous driving. arXiv preprint arXiv:2602.22801, 2026.

Xin Zhou, Zongchuang Zhao, Zhibo Yang, Mingsheng Li, Humen Zhong, Shuai Bai, Du Chu, Ruizhe Chen, Zhaohai Li, Jun Tang, et al. Qwen-drive-1.0: An initial step towards a visionlanguage foundation model for autonomous driving. arXiv preprint arXiv:2609.00111, 2026a.

Zewei Zhou, Tianhui Cai, Seth Zhao, Yun Zhang, Zhiyu Huang, Bolei Zhou, and Jiaqi Ma. Autovla: A vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. Advances in Neural Information Processing Systems, 38: 27920–27956, 2026b.

Jialv Zou, Shaoyu Chen, Bencheng Liao, Zhiyu Zheng, Yuehao Song, Lefei Zhang, Qian Zhang, Wenyu Liu, and Xinggang Wang. Diffusiondrivev2: Reinforcement learning-constrained truncated diffusion modeling in end-to-end autonomous driving. arXiv preprint arXiv:2512.07745, 2025.

## Appendix

In the appendix, we provide additional details on the construction, evaluation, and analysis of Plan-Context that are omitted from the main paper for brevity. The appendix is organized as follows:

• Sec. A provides detailed scenario mining, filtering, and category assignment criteria for the six PlanContext contexts.

• Sec. B presents the category-specific behavioral evaluation rules, including road-relative trajectory comparison, path-change detection, GT-relative suppression, and continuity evaluation for ongoing maneuvers.

• Sec. C describes road compliance, dynamic safety, and the construction of the final Plan-Context metrics.

• Sec. D provides additional experimental details, statistical analyses, implementation information, and supplementary results.

• Sec. E presents qualitative examples across contextual categories and additional cases for ten evaluated models.

## A SCENARIO CONSTRUCTION DETAILS

This section provides the detailed scenario mining, filtering, and category assignment criteria used to construct PlanContext. The construction process first removes scenes with insufficient geometric or motion evidence. It then identifies reliable route branches and adjacent lane structures from the local map. Each retained scene–condition instance is assigned to one of the six contextual categories according to a fixed priority order. This procedure is designed to ensure that each category corresponds to a sufficiently clear contextual requirement.

## A.1 COMMON VALIDITY FILTERING

Each candidate instance contains an original navigation intention and an alternative navigation intention selected from $l e f t ,$ straight, and right. We retain only valid intention pairs for which the initial road, local map topology, logged trajectory, and the category-required reference geometry and permitted road regions can be reliably reconstructed. A target reference path is required only for categories where a valid target branch or adjacent lane is expected to exist.

We additionally remove near-stationary scenes using the future motion of the logged ego trajectory. Let

$$
D _ { \mathrm { G T } } ^ { 4 s } = \sum _ { t } \left. p _ { t } ^ { \mathrm { G T } } - p _ { t - 1 } ^ { \mathrm { G T } } \right. _ { 2 }\tag{10}
$$

denote the cumulative displacement of the logged trajectory over the future four-second horizon. A candidate is retained only when

$$
D _ { \mathrm { G T } } ^ { 4 s } \geq 2 . 0 \mathrm { m } .\tag{11}
$$

This motion filtering removes 598 near-stationary candidates from the initial 12,840 candidates and leaves 12,242 candidates for subsequent contextual screening. Scenes in which the road structure, route branch, adjacent lane, or maneuver stage cannot be determined reliably are marked as ambigu ous, transition, or pending and are excluded from the main benchmark.

## A.2 ROUTE BRANCH CONSTRUCTION AND RELIABILITY

For route-level categories, we search forward from the current road for the first decision point with multiple outgoing edges. The forward search range is limited to 200 m. Each candidate branch is then extended for up to 100 m after the decision point.

Candidate branches are classified according to their terminal heading relative to the reference heading at the decision point. Let $\Delta \theta$ denote this relative heading difference. A branch is classified as straight when

$$
| \Delta \theta | \leq 2 0 ^ { \circ } .\tag{12}
$$

It is classified as left when

$$
\Delta \theta \geq 3 0 ^ { \circ } ,\tag{13}
$$

and as right when

$$
\Delta \theta \leq - 3 0 ^ { \circ } .\tag{14}
$$

Branches falling between these angular regions are considered ambiguous and are not used for reli able category assignment.

The original route branch is selected primarily according to route overlap. When multiple branches have identical route overlap, the logged trajectory is used only as a secondary geometric cue. The closest branch is selected only when its mean distance to the logged trajectory is at least 0.75 m smaller than that of the second closest branch. Otherwise, the instance is marked as having weak GT-based tie-breaking evidence and is excluded.

The original and alternative paths are sampled at intervals of 0.5 m. Reliable separation is searched only within their common valid path extent after the decision point, with the maximum search range defined as

$$
s _ { \mathrm { m a x } } = \operatorname* { m i n } ( L _ { N } , L _ { F } , 5 0 \mathrm { m } ) ,\tag{15}
$$

where $L _ { N }$ and $L _ { F }$ denote the valid lengths of the original and alternative paths.

Within this range, the two paths are considered stably separated when three consecutive sampled positions satisfy at least one of the following conditions:

$$
\begin{array} { r } { \left\| \boldsymbol { p } ^ { N } ( \boldsymbol { s } ) - \boldsymbol { p } ^ { F } ( \boldsymbol { s } ) \right\| _ { 2 } \geq 1 . 5 \mathrm { m } , } \end{array}\tag{16}
$$

or

$$
| \Delta \theta ( s ) | \geq 1 2 ^ { \circ } .\tag{17}
$$

The earliest position at which this condition is continuously satisfied is recorded as the reliable separation position for the corresponding route pair.

A requested navigation intention may correspond to multiple alternative branches. In this case, the route evidence is considered reliable only when every alternative branch associated with the requested intention exhibits stable separation from the selected original branch. For categories A and B1, a valid arrival time must also be available for every alternative branch. If any branch lacks stable separation or a valid arrival estimate, the robust route evidence is treated as unavailable and the instance cannot be assigned to A or B1.

For an instance with multiple valid alternative branches, we use the latest reliable separation distance and the latest estimated arrival time:

$$
d _ { \mathrm { s p l i t } } ^ { \mathrm { r o b u s t } } = \operatorname* { m a x } _ { j } d _ { \mathrm { s p l i t } , j } ,\tag{18}
$$

$$
t _ { \mathrm { a r r i v a l } } ^ { \mathrm { r o b u s t } } = \operatorname* { m a x } _ { j } t _ { \mathrm { a r r i v a l } , j } .\tag{19}
$$

Here, j indexes all alternative branches associated with the requested navigation intention. The arrival time of each branch is estimated from the cumulative future distance of the logged trajectory. Linear interpolation is applied between neighboring trajectory samples, which are spaced at 0.5 s intervals.

## A.3 CATEGORY ASSIGNMENT PRIORITY

Some scenes satisfy more than one preliminary geometric condition. We therefore apply a fixed category assignment order to avoid overlapping definitions.

A scene is first assigned to category C when strict post-commitment evidence is detected. A scene that starts on a lane connector is excluded when strict commitment evidence is unavailable. If a reliable route divergence can be reached within 2.0 s, the instance is assigned to category A. Cases with an arrival time between 2.0 s and 4.5 s are treated as transition cases and are excluded.

If an active lateral maneuver is already present, the instance is assigned to category LC. Among the remaining cases, a reliable route divergence with an arrival time between 4.5 s and 8.0 s forms a category B1 candidate. A scene is assigned to category L when the ego vehicle remains stably within its current lane and a geometrically feasible adjacent target lane is available. Category B2 is used when no reliable nearby route branch and no feasible adjacent target lane support the alternative intention. All remaining cases are marked as ambiguous or pending and are excluded from the main benchmark.

## A.4 CATEGORY-SPECIFIC CONSTRUCTION CRITERIA

Category A: Actionable Route Change. Category A contains scenes in which the alternative route has entered the current decision window and should produce a target-directed response within the four-second prediction horizon. A reliable alternative route branch must satisfy

$$
0 < t _ { \mathrm { a r r i v a l } } ^ { \mathrm { r o b u s t } } \leq 2 . 0 \mathrm { s } .\tag{20}
$$

The ego vehicle must not have strictly committed to the original route. Cases beginning on a lane connector are also excluded when sufficient commitment evidence is unavailable. The interval

$$
2 . 0 < t _ { \mathrm { a r r i v a l } } ^ { \mathrm { r o b u s t } } < 4 . 5 \mathrm { s }\tag{21}
$$

is treated as a transition region and does not contribute to either A or B1.

Category B1: Delayed Route Change. Category B1 represents scenes in which the alternative route is feasible in the near future but should not yet induce a directional commitment within the current prediction horizon. The reliable arrival time must satisfy

$$
4 . 5 \leq t _ { \mathrm { a r r i v a l } } ^ { \mathrm { r o b u s t } } \leq 8 . 0 \mathrm { s } .\tag{22}
$$

We additionally require sufficient remaining distance between the reliable branch separation and the future motion covered by the logged trajectory:

$$
d _ { \mathrm { s p l i t } } ^ { \mathrm { r o b u s t } } - D _ { \mathrm { G T } } ^ { 4 s } \geq 8 . 0 \ \mathrm { m } .\tag{23}
$$

The scene must not contain a feasible adjacent lane that could already provide a valid immediate interpretation of the alternative navigation intention.

Category B2: Locally Invalid Target. Category B2 contains alternative left or right intentions that do not correspond to a locally reachable target in the requested direction. The scene must contain neither a reliable nearby route branch in the requested direction nor a geometrically feasible adjacent target lane within the same roadblock.

We conservatively exclude candidates for which the local map still provides evidence of a potentially reachable target. In particular, a candidate is excluded from B2 whenever at least one branch associated with the alternative direction is detected and its decision point lies within the four-second logged travel distance,

$$
N _ { \mathrm { r e p l a c e m e n t } } > 0\tag{24}
$$

and

$$
d _ { \mathrm { d e c i s i o n } } \leq D _ { \mathrm { G T } } ^ { 4 s } .\tag{25}
$$

This exclusion is applied even when the complete route-pair reliability test is not satisfied. Such cases are excluded from B2 and retained as boundary cases for analysis.

Candidates are also excluded when an adjacent lane exists but fails the required geometric continuity checks. These exclusions reduce ambiguity in the B2 set by removing cases with residual evidence of a potentially reachable target.

Category C: Post-Commitment Route Change. Category C represents scenes in which the alternative route was available at an earlier decision point and the ego vehicle has already passed the relevant divergence and committed to the original route.

Candidate detection searches backward for up to 80 m to locate the nearest previous decision point with multiple outgoing branches. The current branch must agree with the original left or right navigation intention. The alternative direction must also have existed at that historical decision point. The original and alternative branches must exhibit reliable geometric separation. In addition, the ego heading change over approximately the previous two seconds must be consistent with the original turning direction and must reach at least 10<sup>◦</sup>.

We measure the post-commitment progress along the selected committed branch as

$$
d _ { \mathrm { c o m m i t } } = s _ { \mathrm { e g o } } ^ { \mathrm { c o m m i t t e d } } - s _ { \mathrm { d i v e r g e n c e } } ,\tag{26}
$$

where $s _ { \mathrm { e g o } } ^ { \mathrm { c o m m i t t e d } }$ denotes the ego progress measured from the beginning of the selected committed branch and $s _ { \mathrm { d i v e r g e n c e } }$ denotes the stable separation position measured along the same branch.

Candidate detection requires

$$
d _ { \mathrm { c o m m i t } } \geq 3 . 0 \ : \mathrm { m } .\tag{27}
$$

The main benchmark uses the stricter criterion

$$
d _ { \mathrm { c o m m i t } } \geq 5 . 0 \ : \mathrm { m } .\tag{28}
$$

Cases with

$$
3 . 0 \mathrm { m } \leq d _ { \mathrm { c o m m i t } } < 5 . 0 \mathrm { m }\tag{29}
$$

are excluded from the main benchmark and retained as boundary cases for analysis.

Category L: Feasible Adjacent Lane Change. Category L contains scenes in which the ego vehicle has not started a lateral maneuver and the adjacent target lane in the requested direction is currently reachable. The ego vehicle must be within 1.0 m of the centerline of the current lane. The nearest lane associated with each of the most recent three history samples must also remain the current lane.

For the target adjacent lane, the relative heading, lateral distance, and parallel overlap must satisfy

$$
| \Delta \theta _ { \mathrm { l a n e } } | \le 1 5 ^ { \circ } ,\tag{30}
$$

$$
d _ { \mathrm { l a t e r a l } } \leq 8 . 0 \mathrm { m } ,\tag{31}
$$

and

$$
L _ { \mathrm { p a r a l l e l } } \geq 2 0 . 0 \ : \mathrm { m } .\tag{32}
$$

The available map interface does not provide complete legal attributes for every lane boundary. We therefore interpret these cases as topologically feasible adjacent lane changes whose boundary legality cannot always be independently verified. Cases with discontinuous geometry or unstable current lane assignment are excluded.

Category LC: Ongoing Lateral Maneuver. Category LC contains scenes in which a persistent lateral motion is already observable before the navigation intention is changed. We use the ego history from $t = - 2 . 0 \mathrm { ~ s ~ t o ~ } t = 0 \mathrm { ~ s ~ }$ s with a sampling interval of 0.5 s.

An active lateral direction is detected through one of two motion patterns. In the first pattern, the historical trajectory starts at least 0.5 m closer to an adjacent lane than to the current lane,

$$
d _ { \mathrm { a d j } } ( - 2 s ) + 0 . 5 < d _ { \mathrm { c u r r e n t } } ( - 2 s ) .\tag{33}
$$

The trajectory must then move at least 0.75 m farther from that adjacent lane,

$$
d _ { \mathrm { a d j } } ( 0 ) - d _ { \mathrm { a d j } } ( - 2 s ) \geq 0 . 7 5 \ \mathrm { m } ,\tag{34}
$$

while moving at least 0.75 m closer to the current lane,

$$
d _ { \mathrm { c u r r e n t } } ( - 2 s ) - d _ { \mathrm { c u r r e n t } } ( 0 ) \geq 0 . 7 5 \mathrm { m } .\tag{35}
$$

In the second pattern, the trajectory moves from the current lane toward an adjacent lane. It must become at least 0.75 m closer to the adjacent lane while moving at least 0.75 m farther from the current lane.

The relation between the alternative navigation intention and the detected active direction is recorded as continue, reverse, or cancel. These relations are used for supplementary analysis and for the continuity evaluation described in Sec. B. They are not treated as separate benchmark categories.

Table 6: Scenario filtering statistics for PlanContext.
<table><tr><td>Stage</td><td>Instances</td><td>Description</td></tr><tr><td>Initial candidates</td><td>12,840</td><td>Original scene-condition candidates</td></tr><tr><td>After motion filtering</td><td>12,242</td><td>Remove near-stationary instances</td></tr><tr><td>Final benchmark</td><td>9,561</td><td>Apply final context-aware filtering</td></tr></table>

## A.5 SCENARIO FILTERING STATISTICS

Table 6 summarizes the main filtering stages used to construct PlanContext. Starting from 12,840 scene–condition candidates, motion filtering retains 12,242 instances. The subsequent context-aware screening and category-specific validity checks yield the final 9,561 instances used for evaluation.

The final benchmark contains 334 A, 251 B1, 5,285 B2, 580 C, 2,716 L, and 395 LC instances.

## B CATEGORY-SPECIFIC EVALUATION RULES

This section provides the detailed trajectory comparison and category-specific behavioral evaluation rules used by PlanContext. Given the original prediction $T _ { i } ^ { N }$ and the prediction under the alternative navigation intention $T _ { i } ^ { F }$ , we first represent both trajectories with respect to the reference geometry required by each evaluation component. We then determine their topological outcomes, identify persistent behavioral changes induced by the navigation intervention, and evaluate whether these changes satisfy the contextual requirement of each category.

## B.1 NOTATION AND REFERENCE GEOMETRIES

Each prediction contains eight future ego states sampled at 0.5 s intervals over a four-second horizon. Predicted poses are interpreted at the rear axle center and transformed into the global SE(2) coordinate frame.

We use two reference geometries for different evaluation purposes. The original-route reference is used to characterize the path-level difference between the original and alternative predictions. The logged GT trajectory provides a local reference for the suppression tests of B1, B2, and C and for the continuity evaluation of LC. The GT reference represents the natural road evolution observed in the scene and is not treated as a trajectory target to be imitated.

For the common path-change analysis, each prediction is projected onto the original-route reference. For a projected state, we record the longitudinal road progress s, signed lateral offset d, and relative heading

$$
\theta _ { r } = \mathrm { w r a p } \left( \theta _ { \mathrm { e g o } } - \theta _ { \mathrm { r o a d } } \right) .\tag{36}
$$

The original and alternative predictions are aligned at common road-progress values. We define

$$
\Delta d ( s ) = d ^ { F } ( s ) - d ^ { N } ( s )\tag{37}
$$

and

$$
\Delta \theta ( s ) = \mathrm { w r a p } \left( \theta _ { r } ^ { F } ( s ) - \theta _ { r } ^ { N } ( s ) \right) .\tag{38}
$$

This progress-based alignment reduces the influence of longitudinal speed differences when comparing behavioral changes induced by the navigation intervention.

The original-route and target-route references are also used to characterize the terminal topology of each prediction. We denote them as $\mathcal { R } _ { o }$ and $\mathcal { R } _ { t }$ . The GT reference used for suppression and continuity evaluation is denoted as $\mathcal { R } _ { G T }$

## B.2 EVALUATION STRUCTURE

The category-specific criteria are constructed from a shared set of trajectory-level primitives, including route-topology assignment, road-relative trajectory change, GT-relative residual change, full-vehicle road compliance, and dynamic safety. Each contextual category selects the evidence required by its behavioral objective.

Categories A and L evaluate whether a feasible alternative intention is executed. Categories B1, B2, and C evaluate whether the alternative navigation condition induces an unnecessary persistent deviation from the locally appropriate behavior. Category LC evaluates the continuity of an ongoing lateral maneuver.

This factorized design keeps the underlying geometric measurements shared across categories while allowing the decision criterion to reflect the corresponding contextual requirement. Continuous trajectory evidence is evaluated using fixed decision thresholds and boundary regions. Clear evidence contributes to the category-specific decision outcome, while boundary-level evidence is retained for uncertainty-aware evaluation. All category definitions and thresholds are fixed and shared across models.

## B.3 TOPOLOGICAL OUTCOME ASSIGNMENT

We determine the topological outcome of each predicted trajectory from its terminal portion. Let $d _ { o }$ and $d _ { t }$ denote the distances from the trajectory tail to the original-route reference $\mathcal { R } _ { o }$ and target-route reference $\mathcal { R } _ { t }$

When both reference distances are available, a trajectory is assigned to the target topology when

$$
d _ { t } + 0 . 5 < d _ { o } .\tag{39}
$$

When

$$
d _ { o } + 0 . 5 < d _ { t } ,\tag{40}
$$

the trajectory is assigned to the original topology for categories A and L. The same geometric condition is labeled as committed for category C and as active maneuver for category LC.

When the difference between $d _ { o }$ and $d _ { t }$ is smaller than 0.5 m, category B1 uses the shared label.   
The corresponding outcome is marked as ambiguous for the remaining categories.

Reference geometry can be incomplete for some local configurations. When only $d _ { o }$ is available, the prediction is assigned to the original topology for A, B1, B2, and L. The corresponding labels are committed for C and active maneuver for LC.

When $d _ { o }$ is unavailable, category B2 uses an auxiliary state denoted as other-local. The remaining categories use the ambiguous state. The other-local state indicates insufficient comparative evidence to assign the prediction to either the original topology or a supported target topology. It does not indicate road compliance or traffic-rule legality. Road compliance is evaluated independently using the full vehicle footprint in Sec. C.

The resulting topology variable is denoted by $\tau _ { i } ^ { N }$ or $\tau _ { i } ^ { F }$ for the original and alternative predictions.

## B.4 PERSISTENT PATH-CHANGE DETECTION

We next determine whether changing the navigation intention produces a persistent change in the predicted path. Let the trajectory tail contain the final three valid aligned samples. We compute

$$
\widetilde { d } _ { \mathrm { t a i l } } = \mathrm { m e d i a n } \left( \Delta d _ { \mathrm { t a i l } } \right) ,\tag{41}
$$

$$
d _ { \mathrm { { e n d } } } = \Delta d ( s _ { \mathrm { { e n d } } } ) ,\tag{42}
$$

$$
d _ { 7 5 } = Q _ { 0 . 7 5 } \left( \left| \Delta { d } \right| \right) ,\tag{43}
$$

and

$$
\widetilde { \theta } _ { \mathrm { t a i l } } = \mathrm { m e d i a n } \left( \Delta \theta _ { \mathrm { t a i l } } \right) .\tag{44}
$$

For samples satisfying $| \Delta d | \ge 0 . 1 5$ m, we compute the sign-consistency ratio $r _ { \mathrm { s i g n } }$ . This quantity measures the fraction of valid samples whose lateral-change sign agrees with the dominant direction of the trajectory tail.

A strong path change is identified when at least one of the following conditions is satisfied.

1. The original and alternative predictions are assigned to different road topologies.

2. At least four aligned samples are available and

$$
| \widetilde { d } _ { \mathrm { t a i l } } | \geq 0 . 8 5 \ : \mathrm { m } ,\tag{45}
$$

$$
| d _ { \mathrm { e n d } } | \geq 0 . 8 5 \ : \mathrm { m } ,\tag{46}
$$

$$
d _ { 7 5 } \geq 0 . 6 0 \mathrm { m } ,\tag{47}
$$

and

$$
r _ { \mathrm { s i g n } } \geq 0 . 8 0 .\tag{48}
$$

3. The tail heading and lateral differences satisfy

$$
| \widetilde { \theta } _ { \mathrm { t a i l } } | \geq 8 . 5 ^ { \circ } ,\tag{49}
$$

$$
| \widetilde { d } _ { \mathrm { t a i l } } | \geq 0 . 6 0 \mathrm { m } ,\tag{50}
$$

and

$$
r _ { \mathrm { s i g n } } \geq 0 . 8 0 .\tag{51}
$$

We additionally define a boundary-evidence region for diagnostic review. The lateral criterion uses 0.65 m for the tail and endpoint differences, 0.40 m for $d _ { 7 5 }$ , and 0.70 for the sign-consistency ratio. The heading-based criterion uses a heading difference of $6 . 5 ^ { \circ }$ , a lateral difference of at least 0.40 m, and a sign-consistency ratio of 0.70.

If no strong path change is detected and

$$
\begin{array} { r } { \left| s _ { \mathrm { e n d } } ^ { F } - s _ { \mathrm { e n d } } ^ { N } \right| \geq 1 . 0 \mathrm { m } , } \end{array}\tag{52}
$$

the response is labeled as a progress-only change. Responses without a strong path change or sufficient longitudinal difference are labeled as no change. Cases with insufficient common progress or unavailable reference geometry are labeled as geometry ambiguous.

## B.5 DIRECTION OF THE PATH CHANGE

For a persistent path change, we next determine whether the change is directed toward the alternative target, toward the original route, or toward an unrelated direction.

We first apply topology-based decisions. A transition from a non-target topology under the original condition to the target topology under the alternative condition is labeled as a topology switch. If the original prediction is assigned to the target topology and the alternative prediction leaves the target topology, the change is labeled as original-directed.

For the remaining persistent path changes, let

$$
g ( T ) = d ( T , \mathcal { R } _ { o } ) - d ( T , \mathcal { R } _ { t } )\tag{53}
$$

denote the relative target preference of trajectory T. We define

$$
\Delta g = g ( T ^ { F } ) - g ( T ^ { N } ) .\tag{54}
$$

When the required route distances are available, the change is labeled as target-directed when

$$
\Delta g \ge 1 . 2 \ : \mathrm { m } .\tag{55}
$$

The change is labeled as original-directed when

$$
\Delta g \le - 1 . 2 \ : \mathrm { m } .\tag{56}
$$

All remaining persistent directional changes are labeled as unrelated-directional.

## B.6 GT-RELATIVE SUPPRESSION FOR B1, B2, AND C

Categories B1, B2, and C require the alternative navigation intention to avoid inducing an unnecessary persistent path deviation. The logged trajectory is used as a reference for the natural local road evolution. It is not treated as the unique correct trajectory under the alternative navigation condition.

Both predictions are projected onto the GT reference $\mathcal { R } _ { G T }$ . We define the tail lateral residuals as

$$
r _ { d } ^ { N } = \left| \mathrm { m e d i a n } \left( d _ { \mathrm { t a i l } } ^ { N } \right) \right|\tag{57}
$$

and

$$
r _ { d } ^ { F } = \left| \mathrm { m e d i a n } \left( d _ { \mathrm { t a i l } } ^ { F } \right) \right| .\tag{58}
$$

The corresponding heading residuals are

$$
r _ { \theta } ^ { N } = \left| \mathrm { m e d i a n } \left( \theta _ { \mathrm { t a i l } } ^ { N } \right) \right|\tag{59}
$$

and

$$
r _ { \theta } ^ { F } = \left| \operatorname * { m e d i a n } \left( \theta _ { \mathrm { t a i l } } ^ { F } \right) \right| .\tag{60}
$$

The additional deviation induced by the alternative navigation condition is

$$
\Delta r _ { d } = r _ { d } ^ { F } - r _ { d } ^ { N }\tag{61}
$$

and

$$
\Delta r _ { \theta } = r _ { \theta } ^ { F } - r _ { \theta } ^ { N } .\tag{62}
$$

An unnecessary persistent response is identified only when a strong path change has already been detected. Strong lateral evidence is present when

$$
r _ { d } ^ { F } \ge 1 . 5  { \mathrm { m } }\tag{63}
$$

and

$$
\Delta r _ { d } \ge 1 . 0 \mathrm { m } .\tag{64}
$$

Strong heading evidence is present when

$$
r _ { \theta } ^ { F } \geq 1 2 ^ { \circ }\tag{65}
$$

and

$$
\Delta r _ { \theta } \geq 1 0 ^ { \circ } .\tag{66}
$$

Two additional conditions provide strong evidence of an unnecessary response. The first occurs when the original and alternative predictions lie on opposite sides of the GT reference and satisfy

$$
r _ { d } ^ { F } \ge 0 . 8 5 \mathrm { m }\tag{67}
$$

and

$$
r _ { d } ^ { N } \geq 0 . 4 0 \mathrm { m } .\tag{68}
$$

The second occurs when the alternative prediction switches from a non-target topology to the target topology.

We denote the resulting unnecessary-response indicator as $U _ { i }$ . The GT-relative evaluation is considered available when the required GT reference geometry and residual quantities can be computed. We denote this availability by $G _ { i }$

Boundary evidence is retained for diagnostic review. The GT-relative boundary thresholds use a lateral residual of at least 1.0 m with an additional increase of at least 0.5 m, or a heading residual of at least $8 ^ { \circ }$ with an additional increase of at least $6 ^ { \circ }$ . A sample is also retained for diagnostic review when the common path-change detector in Sec. B.4 produces only boundary evidence. Diagnostic boundary evidence does not automatically make the contextual decision incorrect.

## B.7 CATEGORY-SPECIFIC BEHAVIORAL CORRECTNESS

For categories A, B1, B2, C, and L, contextual correctness combines the semantic validity of the two predictions with their paired behavioral relationship.

Category A. The original prediction is semantically valid when

$$
S _ { i } ^ { N } = [ \tau _ { i } ^ { N } = \mathrm { o r i g i n a l } ] ,\tag{69}
$$

and the alternative prediction is semantically valid when

$$
S _ { i } ^ { F } = [ \tau _ { i } ^ { F } = \mathrm { t a r g e t } ] .\tag{70}
$$

The paired behavioral relation is correct when the detected path change is target-directed or produces a topology switch:

$$
P _ { i } = [ \mathrm { t a r g e t - d i r e c t e d o r t o p o l o g y s w i t c h } ] .\tag{71}
$$

The contextual decision correctness is

$$
C _ { i } = S _ { i } ^ { N } \wedge S _ { i } ^ { F } \wedge P _ { i } .\tag{72}
$$

A missing response, an incorrect branch transition, or a persistent change in an unrelated direction is counted as incorrect.

Category B1. The original and alternative predictions are semantically valid when they remain on the shared approach or original route:

$$
S _ { i } ^ { N } = [ \tau _ { i } ^ { N } \in \{ \mathrm { s h a r e d } , \mathrm { o r i g i n a l } \} ] ,\tag{73}
$$

$$
S _ { i } ^ { F } = [ \tau _ { i } ^ { F } \in \{ \mathrm { s h a r e d } , \mathrm { o r i g i n a l } \} ] .\tag{74}
$$

The paired behavioral requirement is

$$
P _ { i } = G _ { i } \land \lnot U _ { i } ,\tag{75}
$$

and the contextual decision correctness is

$$
C _ { i } = S _ { i } ^ { N } \wedge S _ { i } ^ { F } \wedge P _ { i } .\tag{76}
$$

Longitudinal slowing, braking, progress-only changes, and natural road curvature remain valid. Any unnecessary persistent path deviation induced by the alternative intention is counted as incorrect. In B1, the most direct failure mode is premature commitment toward a target branch that has not yet entered the current decision window.

Category B2. The original and alternative predictions are semantically valid when they do not enter the unsupported target topology:

$$
S _ { i } ^ { N } = [ \tau _ { i } ^ { N } \neq \mathrm { t a r g e t } ] ,\tag{77}
$$

$$
S _ { i } ^ { F } = [ \tau _ { i } ^ { F } \neq \mathrm { t a r g e t } ] .\tag{78}
$$

The paired behavioral requirement is

$$
P _ { i } = G _ { i } \land \lnot U _ { i } ,\tag{79}
$$

which gives

$$
C _ { i } = S _ { i } ^ { N } \wedge S _ { i } ^ { F } \wedge P _ { i } .\tag{80}
$$

A lack of directional response is compatible with B2 but does not independently guarantee final success. Any unnecessary persistent path deviation induced by the alternative intention is counted as incorrect. This includes a persistent response toward an unsupported target and a strong unrelated directional deviation. Road compliance is evaluated separately in Sec. C.

Category C. The original and alternative predictions must remain consistent with the already committed route:

$$
S _ { i } ^ { N } = [ \tau _ { i } ^ { N } \in \{ \mathrm { c o m m i t t e d , o r i g i n a l } \} ] ,\tag{81}
$$

$$
S _ { i } ^ { F } = [ \tau _ { i } ^ { F } \in \{ \mathrm { c o m m i t t e d , o r i g i n a l } \} ] .\tag{82}
$$

The paired behavioral requirement is

$$
P _ { i } = G _ { i } \land \lnot U _ { i } ,\tag{83}
$$

and

$$
C _ { i } = S _ { i } ^ { N } \wedge S _ { i } ^ { F } \wedge P _ { i } .\tag{84}
$$

The model may slow down, stop, or modify its longitudinal progress. Any unnecessary persistent deviation induced by the alternative intention is counted as incorrect. Typical failures include reversal toward a previously missed alternative branch, an additional pull toward that branch, and a strong unrelated directional deviation.

Category L. The original prediction is required to remain in the original lane:

$$
S _ { i } ^ { N } = [ \tau _ { i } ^ { N } = \mathrm { o r i g i n a l } ] .\tag{85}
$$

The alternative prediction is required to enter the target adjacent lane:

$$
S _ { i } ^ { F } = [ \tau _ { i } ^ { F } = \mathrm { t a r g e t } ] .\tag{86}
$$

The paired behavioral relation is correct when the path change is target-directed or produces a topology switch:

$$
P _ { i } = [ \mathrm { t a r g e t - d i r e c t e d o r t o p o l o g y s w i t c h } ] .\tag{87}
$$

The contextual decision correctness is

$$
C _ { i } = S _ { i } ^ { N } \wedge S _ { i } ^ { F } \wedge P _ { i } .\tag{88}
$$

A small lateral or heading response that remains within the current lane is not counted as a completed lane change.

## B.8 LC CONTINUITY EVALUATION

Category LC evaluates whether changing the navigation intention disrupts an ongoing lateral maneuver. It does not use independent semantic labels for the original and alternative predictions. The evaluation produces one of three states:

$$
P _ { i } ^ { L C } \in \{ \mathrm { c o r r e c t , i n c o r r e c t , u n c e r t a i n } \} .\tag{89}
$$

Both predictions are projected onto the GT reference $\mathcal { R } _ { G T }$ . At each future time step t, we define the signed lateral residuals

$$
d _ { t } ^ { N } = d ( T _ { t } ^ { N } , \mathcal { R } _ { G T } ) ,\tag{90}
$$

$$
d _ { t } ^ { F } = d ( T _ { t } ^ { F } , \mathcal { R } _ { G T } ) ,\tag{91}
$$

and the corresponding heading residuals

$$
\theta _ { t } ^ { N } = \mathrm { w r a p } \left( \theta ( T _ { t } ^ { N } ) - \theta _ { G T } ( s _ { t } ) \right) ,\tag{92}
$$

$$
\theta _ { t } ^ { F } = \mathrm { w r a p } \left( \theta ( T _ { t } ^ { F } ) - \theta _ { G T } ( s _ { t } ) \right) .\tag{93}
$$

The absolute residual measures deviation from the natural GT road evolution. The difference between the alternative and original residuals is used to distinguish intervention-induced changes from deviations that are already present in the original prediction.

The continuity protection window contains the first three predicted states at 0.5 s, 1.0 s, and 1.5 s.

Early Additional Deviation. A strong early lateral deviation is detected when at least two states within the protection window satisfy

$$
| d _ { t } ^ { F } | \geq 1 . 0 \mathrm { m }\tag{94}
$$

and

$$
| d _ { t } ^ { F } | - | d _ { t } ^ { N } | \geq 0 . 7 5 \mathrm { m } .\tag{95}
$$

A strong early heading deviation is detected when at least two states satisfy

$$
| \theta _ { t } ^ { F } | \geq 1 0 ^ { \circ }\tag{96}
$$

and

$$
| \theta _ { t } ^ { F } | - | \theta _ { t } ^ { N } | \geq 8 ^ { \circ } .\tag{97}
$$

The corresponding boundary thresholds are 0.75 m and 0.50 m for lateral deviation and $8 ^ { \circ }$ and $6 ^ { \circ }$ for heading deviation.

Persistent Reversal Within the Protection Window. Using the active maneuver direction detected from the history, we define the opposite lateral direction as the reversal direction. Let $d _ { \mathrm { r e } } ^ { N }$ verse and $d _ { \mathrm { r e v e r s e } } ^ { F }$ denote the projected displacement components along this reversal direction.

A strong persistent reversal is detected when at least two states within the protection window satisfy

$$
d _ { \mathrm { r e v e r s e } } ^ { F } \geq 1 . 0 \mathrm { m }\tag{98}
$$

and

$$
d _ { \mathrm { r e v e r s e } } ^ { F } - d _ { \mathrm { r e v e r s e } } ^ { N } \geq 0 . 7 5 \ \mathrm { m } .\tag{99}
$$

The corresponding boundary thresholds are 0.75 m and 0.50 m.

Reversal Beginning Near the End of the Protection Window. For the reverse relation, we additionally detect sustained reversals that begin near the end of the protection window.

Let $\Delta d _ { \mathrm { r e v } } ( t )$ denote the additional lateral deviation of the alternative prediction relative to the original prediction along the reversal direction at future time t. Similarly, $\dot { \Delta } \theta _ { \mathrm { r e v } } ( t )$ denotes the additional wrapped heading deviation along the reversal direction. Positive values indicate increasing deviation in the reversal direction.

The reversal consistency ratio $r _ { \mathrm { r e v } }$ is defined as the fraction of valid aligned future states whose signed heading change is consistent with the reversal direction.

Strong evidence requires

$$
\Delta \theta _ { \mathrm { r e v } } ( 1 . 5 s ) \geq 1 2 ^ { \circ } ,\tag{100}
$$

$$
\Delta \theta _ { \mathrm { r e v } } ( 2 . 0 s ) \geq 1 8 ^ { \circ } ,\tag{101}
$$

$$
\Delta \theta _ { \mathrm { r e v } } ( 2 . 0 s ) - \Delta \theta _ { \mathrm { r e v } } ( 1 . 5 s ) \geq 3 ^ { \circ } ,\tag{102}
$$

$$
\Delta d _ { \mathrm { r e v } } ( 2 . 0 s ) \geq 0 . 7 5 \mathrm { m } ,\tag{103}
$$

$$
\mathrm { m e d i a n } _ { t > 2 s } \Delta \theta _ { \mathrm { r e v } } ( t ) \geq 2 0 ^ { \circ } ,\tag{104}
$$

and

$$
r _ { \mathrm { r e v } } \geq 0 . 7 5 .\tag{105}
$$

The corresponding boundary thresholds are $8 ^ { \circ }$ , $1 2 ^ { \circ }$ , 2<sup>◦</sup>, 0.50 m, $1 5 ^ { \circ }$ , and 0.75.

Unrelated Persistent Deviation. An unrelated-directional path change is treated as strong continuity-disruption evidence when the alternative prediction also exhibits a sufficiently large residual relative to the GT reference.

The lateral condition is

$$
r _ { d } ^ { F } \ge 1 . 5  { \mathrm { m } }\tag{106}
$$

and

$$
\Delta r _ { d } \ge 1 . 0 \mathrm { m } .\tag{107}
$$

The heading condition is

$$
r _ { \theta } ^ { F } \geq 1 2 ^ { \circ }\tag{108}
$$

and

$$
\Delta r _ { \theta } \geq 1 0 ^ { \circ } .\tag{109}
$$

An unrelated-directional path change satisfying either condition is counted as strong disruption evidence.

The corresponding boundary thresholds use a lateral residual of at least 1.0 m with an additional increase of at least 0.5 m, or a heading residual of at least $8 ^ { \circ }$ with an additional increase of at least $6 ^ { \circ }$

Final LC Status. The LC continuity status is assigned as

$$
P _ { i } ^ { L C } = \left\{ \begin{array} { l l } { \mathrm { i n c o r r e c t } , } & { \mathrm { i f ~ s t r o n g ~ d i s r u p t i o n ~ e v i d e n c e ~ i s ~ p r e s e n t } , } \\ { \mathrm { u n c e r t a i n } , } & { \mathrm { i f ~ o n l y ~ b o u n d a r y ~ e v i d e n c e ~ i s ~ p r e s e n t } , } \\ { \mathrm { c o r r e c t } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{110}
$$

A smooth response to the new navigation intention after the protection window is allowed. Large absolute deviation from the logged trajectory is retained as diagnostic information and does not independently determine the LC status.

For LC, contextual correctness is

$$
C _ { i } = 1\tag{111}
$$

when $P _ { i } ^ { L C } = \mathrm { { c o r r e c t \ a n d } }$

$$
C _ { i } = 0\tag{112}
$$

when $P _ { i } ^ { L C }$ = incorrect. Instances with $P _ { i } ^ { L C }$ = uncertain are excluded from subsequent successrate aggregation.

## C ROAD COMPLIANCE, SAFETY, AND METRIC DETAILS

This section describes the road-compliance and dynamic-safety evaluation used by PlanContext, followed by the aggregation of instance-level results into the final benchmark metrics. These components are evaluated separately from the contextual decision correctness defined in Sec. B.

## C.1 FULL-VEHICLE ROAD COMPLIANCE

Road compliance is evaluated using the complete ego-vehicle footprint at every predicted future state. Each predicted pose is interpreted at the rear axle center. We use the dimensions of the Pacifica vehicle model, with a length of 5.176 m and a width of 2.297 m. The rear axle is located 1.461 m behind the geometric center of the vehicle footprint.

Let $B ( T _ { i , t } ^ { F } )$ denote the rectangular vehicle footprint corresponding to the alternative prediction at future state t. Let $\Omega _ { i } ^ { F }$ denote the road region permitted by the contextual requirement of instance i. A 0.35 m geometric buffer is applied to the permitted road region.

The road-compliance indicator is defined as

$$
R _ { i } ^ { F } = \prod _ { t = 1 } ^ { 8 } \mathcal { k } \left[ \mathcal { B } ( T _ { i , t } ^ { F } ) \subseteq \Omega _ { i } ^ { F } \right] .\tag{113}
$$

A prediction passes the road-compliance evaluation only when the complete vehicle footprint is contained in the permitted region at all eight future states.

The permitted road region depends on the contextual category. Table 7 summarizes the corresponding regions under the original and alternative navigation conditions.

For A, the permitted region follows the original route under the original navigation condition and the target route under the alternative condition. For B1, both conditions remain within the original approach region because the alternative route has not yet entered the current decision window. B2 uses the locally available road region because no supported target route is defined. Category C follows the already committed original route. Category L uses the current lane under the original condition and the union of the current and target adjacent lanes under the alternative condition. Category LC uses the local road region associated with the ongoing lateral maneuver.

Table 7: Permitted road regions for the six PlanContext categories.
<table><tr><td>Category</td><td>Original condition</td><td>Alternative condition</td></tr><tr><td>A</td><td>Original branch</td><td>Target branch</td></tr><tr><td>B1</td><td>Original approach branch</td><td>Original approach branch</td></tr><tr><td>B2</td><td>Local valid roads and adjacent roads</td><td>Local valid roads and adjacent roads</td></tr><tr><td>C</td><td>Committed original branch</td><td>Committed original branch</td></tr><tr><td>L</td><td>Current lane</td><td>Current and target adjacent lanes</td></tr><tr><td>LC</td><td>Maneuver-related local roads</td><td>Maneuver-related local roads</td></tr></table>

Each permitted corridor is extended sufficiently far to cover the longest predicted trajectory arc length, the vehicle front overhang, and an additional 5 m margin. For category A, a target branch that terminates within the local map is allowed to end at its valid road termination.

## C.2 DYNAMIC AND RULE-BASED SAFETY

Dynamic safety is evaluated using the NAVSIM safety criteria on the alternative prediction. The evaluation includes collision avoidance, drivable-area compliance, driving-direction compliance, traffic-light compliance, time-to-collision, and history-based comfort.

Let $e _ { i , m } ^ { F }$ denote the binary result of safety component m for instance i. The dynamic-safety indicator is

$$
E _ { i } ^ { F } = \bigwedge _ { m } e _ { i , m } ^ { F } .\tag{114}
$$

The evaluated components are

$$
m \in \{ \mathrm { c o l l i s i o n } , \mathrm { d r i v a b l e } , \mathrm { d i r e c t i o n } , \mathrm { t r a f f c ~ l i g h t } , \mathrm { T T C } , \mathrm { c o m f o r t } \} .\tag{115}
$$

The overall safety condition combines the full-vehicle road-compliance check with the NAVSIM safety criteria:

$$
\mathrm { O v e r a l l S a f e t y } _ { i } = R _ { i } ^ { F } \wedge E _ { i } ^ { F } .\tag{116}
$$

Road compliance and NAVSIM dynamic safety are kept as separate components because they characterize different failure modes. The former evaluates whether the complete predicted vehicle footprint remains within the context-specific permitted region. The latter evaluates dynamic interactions and rule-related safety conditions.

## C.3 INSTANCE-LEVEL CONTEXTUAL SUCCESS

Let $C _ { i }$ denote the contextual decision correctness defined in Sec. B. The final contextual success of instance i is

$$
q _ { i } = C _ { i } \land R _ { i } ^ { F } \land E _ { i } ^ { F } .\tag{117}
$$

A successful instance therefore requires a context-consistent decision response, road-compliant motion, and dynamic safety.

For additional analysis, we additionally define the correctness of the alternative-condition prediction as

$$
A _ { i } ^ { F } = \left\{ \begin{array} { l l } { S _ { i } ^ { F } , } & { k _ { i } \in \{ A , L \} , } \\ { S _ { i } ^ { F } \wedge G _ { i } \wedge \neg U _ { i } , } & { k _ { i } \in \{ B 1 , B 2 , C \} , } \\ { \nparallel \nparallel S \left[ P _ { i } ^ { L C } = \mathrm { c o r r e c t } \right] , } & { k _ { i } = L C . } \end{array} \right.\tag{118}
$$

The corresponding alternative-condition success indicator is

$$
q _ { i } ^ { \mathrm { a l t } } = A _ { i } ^ { F } \wedge R _ { i } ^ { F } \wedge E _ { i } ^ { F } .\tag{119}
$$

This metric evaluates whether the prediction under the alternative navigation condition is contextually valid, road compliant, and dynamically safe. For B1, B2, and C, alternative-condition validity additionally requires that the GT-relative suppression test is available and that no unnecessary persistent response is detected. For LC, the alternative-condition correctness is determined by the continuity status defined in Sec. B.8.

LC instances with $P _ { i } ^ { L C }$ = uncertain are excluded from the valid denominator of the alternativecondition success rate.

## C.4 CLASS-BALANCED METRIC AGGREGATION

The six PlanContext categories contain substantially different numbers of instances. We therefore compute the contextual success rate independently within each category before aggregating across categories.

For category k, the contextual success rate is

$$
S _ { k } = \frac { 1 } { \vert \mathcal { D } _ { k } \vert } \sum _ { i \in \mathcal { D } _ { k } } q _ { i } ,\tag{120}
$$

where $\mathcal { D } _ { k }$ denotes the valid scene–condition instances used for metric aggregation in category k. For LC, instances labeled as uncertain are excluded from $\mathcal { D } _ { L C }$

We aggregate categories A, B1, B2, and C for route-level evaluation and report categories L and LC separately for lane-level evaluation.

The Navigation Action Score (NAS) is the contextual success rate in category A, where the alternative route is immediately actionable. The Context Preservation Score (CPS) averages the contextual success rates across B1, B2, and C, where the alternative navigation intention should be deferred or rejected:

$$
\mathrm { C P S } = \frac { S _ { B 1 } + S _ { B 2 } + S _ { C } } { 3 } .\tag{121}
$$

The Navigation Context Score (NCS) summarizes route-level in-context driving decision-making through the geometric mean of NAS and CPS:

$$
\mathrm { N C S } = \sqrt { \mathrm { N A S } \cdot \mathrm { C P S } } .\tag{122}
$$

CPS is a macro-average of the success rates in B1, B2, and C, with each category-level score assigned a weight of one third. The geometric mean penalizes imbalanced performance across NAS and CPS.

For lane-level in-context driving decision-making, the Lane Adaptation Score (LAS) and Maneuver Continuity Score (MCS) are the contextual success rates in L and LC, respectively. These scores are reported separately and are not included in NCS.

For comparison with conventional driving quality, matched PDMS is computed on the same 5,911 physical scenes used by PlanContext under the original navigation condition. Each physical scene contributes once to matched PDMS even when it is associated with multiple alternative navigation conditions.

## D ADDITIONAL EXPERIMENTAL DETAILS AND ANALYSES

This section provides additional details on the benchmark composition, model evaluation protocol, statistical analysis, contextual factor definitions, and robustness analyses used in PlanContext. The definitions of scenario construction, contextual correctness, road compliance, and metric aggregation are provided in Secs. A, B, and C.

Table 8: Composition of the PlanContext benchmark.
<table><tr><td>Category</td><td>Requirement</td><td>Instances</td></tr><tr><td>A</td><td>Follow</td><td>334</td></tr><tr><td>B1</td><td>Wait</td><td>251</td></tr><tr><td>B2</td><td>Reject</td><td>5,285</td></tr><tr><td>C</td><td>Reject</td><td>580</td></tr><tr><td>L</td><td>Follow</td><td>2,716</td></tr><tr><td>LC</td><td>Preserve</td><td>395</td></tr><tr><td>Total</td><td></td><td>9,561</td></tr></table>

## D.1 BENCHMARK COMPOSITION

PlanContext is constructed from the NAVSIM navtest split. The final benchmark contains 9,561 scene–condition instances derived from 5,911 unique physical scenes. A physical scene may contribute multiple scene–condition instances when multiple alternative navigation intentions satisfy the construction criteria.

The six contextual categories contain 334 A, 251 B1, 5,285 B2, 580 C, 2,716 L, and 395 LC instances. Table 8 summarizes the benchmark composition.

The benchmark is category imbalanced because the six contextual conditions occur at different frequencies in NAVSIM. We therefore use the class-balanced aggregation defined in Sec. C.4 instead of a direct micro-average over all instances.

## D.2 MODEL EVALUATION PROTOCOL

We evaluate 23 publicly available end-to-end driving models covering three representative paradigms. Vision and state to action models directly predict future trajectories from sensory observations or state inputs. Vision language action models incorporate vision language representations into driving policies. World model based action models explicitly predict or encode future scene dynamics before producing driving actions.

All models are evaluated using publicly released or official assets. Navigation conditions are explicitly involved during inference. Model outputs are standardized to eight future ego poses sampled over a four-second prediction horizon.

For each scene–condition instance, the model is evaluated twice. The first inference uses the original navigation intention $c _ { i } ^ { N }$ . The second inference uses the alternative navigation intention $c _ { i } ^ { F } .$ . Sensor observations and vehicle history remain unchanged between the two evaluations. This paired protocol isolates the behavioral response to the navigation intervention while preserving the observed scene.

For conventional driving evaluation, we report PDMS on the full NAVSIM navtest split. We additionally compute matched PDMS, denoted as mPDMS, on the same 5,911 physical scenes used by PlanContext. mPDMS is evaluated under the original navigation condition. Each physical scene contributes once to mPDMS even when it produces multiple scene–condition instances in PlanCon text.

The PlanContext evaluation covers all 9,561 scene–condition instances across the six categories. For each model, we report NAS, CPS, and NCS for route-level evaluation, LAS and MCS for lane-level evaluation, and the six category-wise contextual success rates.

## D.3 STATISTICAL ANALYSIS

We estimate uncertainty using scene-level bootstrap resampling. The resampling unit is the unique physical scene token instead of an individual scene–condition instance. This design preserves the dependence among multiple navigation conditions constructed from the same physical scene.

Table 9: NCS and 95% scene-level bootstrap confidence intervals for the 23 evaluated models.
<table><tr><td>Model</td><td>NCS</td><td>95% CI</td></tr><tr><td>OneVL</td><td>39.30</td><td>[34.73, 43.41]</td></tr><tr><td>ChainFlow-VLA</td><td>35.13</td><td>[31.19, 38.83]</td></tr><tr><td>DrivoR</td><td>34.78</td><td>[30.76, 38.58]</td></tr><tr><td>ReCogDrive-IL</td><td>34.31</td><td>[30.38, 38.01]</td></tr><tr><td>SUV</td><td>32.03</td><td>[27.10, 36.55]</td></tr><tr><td>ReCogDrive-RL</td><td>31.64</td><td>[27.82, 35.18]</td></tr><tr><td>GoalFlow</td><td>31.54</td><td>[27.26, 35.45]</td></tr><tr><td>NoRD</td><td>29.51</td><td>[25.79, 33.03]</td></tr><tr><td>WoTE</td><td>28.52</td><td>[24.28, 32.28]</td></tr><tr><td>AutoVLA</td><td>27.08</td><td>[22.93, 30.83]</td></tr><tr><td>TransFuser</td><td>27.01</td><td>[22.48, 31.03]</td></tr><tr><td>SparseDriveV2</td><td>26.60</td><td>[22.79, 30.05]</td></tr><tr><td>DriveVLA-W0</td><td>16.82</td><td>[12.11, 20.95]</td></tr><tr><td>DiffusionDrive</td><td>14.68</td><td>[9.92, 18.69]</td></tr><tr><td>StateMLP</td><td>11.63</td><td>[9.78, 13.37]</td></tr><tr><td>DP-VLA</td><td>10.90</td><td>[6.13, 14.88]</td></tr><tr><td>Curious-VLA</td><td>6.02</td><td>[0.00, 9.72]</td></tr><tr><td>CLOVER</td><td>5.86</td><td>[0.00, 9.45]</td></tr><tr><td>WAM-Flow</td><td>5.86</td><td>[0.00, 9.36]</td></tr><tr><td>SimWAM</td><td>4.12</td><td>[0.00, 7.41]</td></tr><tr><td>DiffusionDriveV2</td><td>0.00</td><td>[0.00, 0.00]</td></tr><tr><td>Drive-JEPA</td><td>0.00</td><td>[0.00, 0.00]</td></tr><tr><td>DynVLA Stage 2</td><td>0.00</td><td>[0.00, 0.00]</td></tr></table>

DiffusionDriveV2, Drive-JEPA, and DynVLA Stage 2 have ${ \mathrm { N A S } } = 0 , $ yielding $\mathrm { N C S } = 0$ in both the original evaluation and al bootstrap replicates.

For each bootstrap replicate, 5,911 scene tokens are sampled with replacement. All scene–condition instances associated with a sampled scene are retained together. Predictions from all evaluated models associated with the same sampled scene are also retained together. This produces matched bootstrap samples across models and preserves the paired evaluation structure.

We use 10,000 scene-level bootstrap replicates to estimate uncertainty. Table 9 reports NCS and its 95% percentile bootstrap confidence interval for each model.

Pairwise model comparisons are performed using the same matched bootstrap replicates. For a pair of models a and $b ,$ let

$$
\Delta _ { r } ^ { * } = \mathrm { N C S } _ { a } ^ { * ( r ) } - \mathrm { N C S } _ { b } ^ { * ( r ) }\tag{123}
$$

denote their NCS difference in bootstrap replicate r.

For B = 10,000 matched bootstrap replicates, the two-sided p-value is computed using an add-one correction:

$$
p = \operatorname* { m i n } \left\{ 1 , 2 \operatorname* { m i n } \left( \frac { 1 + \sum _ { r = 1 } ^ { B } | \mathcal { E } \left[ \Delta _ { r } ^ { * } \leq 0 \right] } { B + 1 } , \frac { 1 + \sum _ { r = 1 } ^ { B } | \mathcal { k } \left[ \Delta _ { r } ^ { * } \geq 0 \right] } { B + 1 } \right) \right\} .\tag{124}
$$

Statistical testing is performed over all

$$
{ \binom { 2 3 } { 2 } } = 2 5 3\tag{125}
$$

pairwise NCS hypotheses. Benjamini–Hochberg correction is applied jointly to these 253 p-values to control the false discovery rate.

We assess uncertainty in model comparisons using matched scene-level bootstrap resampling. Of the 253 pairwise NCS comparisons, 193 are significant after Benjamini–Hochberg correction at $q \ < \ 0 . 0 5$ . OneVL exceeds ChainFlow-VLA by 4.17 points, with a 95% confidence interval of [0.74, 7.43] for the paired difference and an adjusted q-value of 0.0252. For adjacent models ranked second through seventh, the 95% confidence intervals for paired NCS differences all include zero.

Bootstrap confidence-interval exclusion and Benjamini–Hochberg significance are reported as two related statistical criteria. The same scene-level resampling principle is used for the confidence intervals reported in the contextual factor analysis.

## D.4 IMPLEMENTATION AND REPRODUCIBILITY DETAILS

All reported PlanContext results use the same fixed scenario construction and evaluation rules described in Secs. A, B, and C. The category definitions, geometric thresholds, path-change criteria, GT-relative suppression rules, LC continuity rules, road-compliance constraints, and metric aggregation are shared across all evaluated models.

The evaluation operates on model trajectories after conversion to a common representation of eight future ego poses over four seconds. Predicted poses are interpreted at the rear axle center before geometric evaluation. Road-relative projection, topology assignment, paired trajectory comparison, road-compliance evaluation, and NAVSIM safety evaluation are applied using the same procedure for every model.

Cases with ambiguous scene geometry are removed during benchmark construction according to Sec. A. Geometry-related states produced during model evaluation are handled according to the rules in Sec. B. For category LC, predictions assigned the uncertain continuity state are excluded from the valid denominator when computing the category success rate.

The benchmark construction and evaluation pipeline uses fixed category assignments and fixed evaluation thresholds for all reported results. Model comparisons are performed on the same scene– condition definitions under the same evaluation protocol.

## D.5 CONTEXT FACTOR DEFINITIONS

The contextual factor analysis examines how in-context driving performance varies with scene structure and traffic interaction. All factor stratification is performed within each PlanContext category. This avoids directly mixing the different scene distributions of the six contextual categories.

Decision Urgency. Decision urgency characterizes how close the current scene is to requiring an immediate behavioral decision. Since the six categories represent different contextual requirements, the corresponding proxy is category specific:

$$
u _ { i } ^ { \mathrm { u r g } } = \left\{ \begin{array} { l l } { 1 / t _ { \mathrm { a r r i v a l } } ^ { \mathrm { o b u s t } } , } & { k _ { i } \in \{ A , B 1 \} , } \\ { 1 / d _ { \mathrm { c o m m i t } } , } & { k _ { i } = C , } \\ { 1 / L _ { \mathrm { p a r a l l e l } } ^ { \mathrm { r e m a i n } } , } & { k _ { i } = L , } \\ { M _ { \mathrm { l a t } } ^ { \mathrm { h i s t } } , } & { k _ { i } = L C , } \\ { D _ { \mathrm { G T } } ^ { 4 s } / d _ { \mathrm { d e c i s i o n } } , } & { k _ { i } = B 2 . } \end{array} \right.\tag{126}
$$

Here, $L _ { \mathrm { p a r a l l e l } } ^ { \mathrm { r e m a i n } }$ denotes the remaining parallel length of the target adjacent lane. $M _ { \mathrm { l a t } } ^ { \mathrm { h i s t } }$ denotes the magnitude of historical lateral motion used to characterize the ongoing maneuver. Larger values indicate higher decision urgency.

Within each category, urgency values are ranked and divided into Low, Medium, and High groups using category-specific rank terciles.

Geometric Salience. Geometric salience characterizes how clearly the local geometry exposes the relevant decision structure. It is constructed from the available normalized geometric components

$$
\frac { | \theta _ { \mathrm { s t a r t } } | } { 1 5 ^ { \circ } } , \qquad \frac { | \theta _ { \mathrm { b r a n c h } } ^ { \mathrm { m a x } } | } { 4 5 ^ { \circ } } , \qquad \frac { d _ { \mathrm { s p l i t } } } { 5 ^ { \textrm { m } } } ,\tag{127}
$$

$$
\frac { | \Delta \theta _ { \mathrm { h i s t } } | } { 3 0 ^ { \circ } } , \qquad \frac { | \Delta \theta _ { \mathrm { l a n e } } | } { 1 5 ^ { \circ } } , \qquad \frac { d _ { \mathrm { l a t e r a l } } } { 8 \mathrm { m } } .\tag{128}
$$

Each available component is clipped to the interval [0, 2]. The geometric salience score is defined as

$$
s _ { i } ^ { \mathrm { g e o } } = \frac { 1 } { | \mathcal { V } _ { i } | } \sum _ { m \in \mathcal { V } _ { i } } \mathrm { c l i p } \left( g _ { i , m } , 0 , 2 \right) ,\tag{129}
$$

where $\nu _ { i }$ denotes the set of available geometric components for instance i.

Geometric salience is therefore a composite descriptor built from the available road and motion geometry. Low, Medium, and High groups are obtained using rank terciles within each category.

Topological Evidence. Topological evidence characterizes the strength of the local structural cues associated with the contextual requirement. Its definition depends on the category.

For A and B1, the evidence is based on the relative heading of the target branch. Weak, medium, and strong evidence correspond to

$$
| \Delta \theta _ { \mathrm { t a r g e t } } | < 6 0 ^ { \circ } ,\tag{130}
$$

$$
6 0 ^ { \circ } \leq | \Delta \theta _ { \mathrm { t a r g e t } } | < 9 0 ^ { \circ } ,\tag{131}
$$

and

$$
| \Delta \theta _ { \mathrm { t a r g e t } } | \geq 9 0 ^ { \circ } .\tag{132}
$$

For B2, topological evidence is defined by the number of locally supported targets in the alternative direction. Weak evidence corresponds to at least two supported targets. Medium evidence corresponds to one supported target. Strong evidence corresponds to no supported target. Evidence strength therefore increases as the number of supported alternatives decreases.

For C, the evidence is based on the post-commitment progress. Weak evidence satisfies

$$
d _ { \mathrm { c o m m i t } } < 1 0 \mathrm { m } .\tag{133}
$$

Medium evidence satisfies

$$
1 0 \mathrm { { m } \leq d _ { c o m m i t } < 3 0 \mathrm { { m } . } }\tag{134}
$$

Strong evidence satisfies

$$
d _ { \mathrm { c o m m i t } } \geq 3 0 \mathrm { m } .\tag{135}
$$

For L, the evidence is determined by the available parallel length of the target adjacent lane. Weak, medium, and strong evidence correspond to 20–30 m, 30–60 m, and at least 60 m, respectively.

For LC, the evidence is based on historical lateral-motion magnitude. Weak evidence corresponds to less than 1.0 m. Medium evidence corresponds to 1.0–1.5 m. Strong evidence corresponds to at least 1.5 m.

The topological evidence analysis in the main paper reports Strong-minus-Weak contextual success.

Traffic Density. Traffic density is defined using a weighted count of nearby traffic participants within 40 m:

$$
D _ { \mathrm { t r a f f i c } } = N _ { \mathrm { v e h , 4 0 } m } + 0 . 5 \left( N _ { \mathrm { p e d , 4 0 } m } + N _ { \mathrm { b i k e , 4 0 } m } \right) + 1 . 5 N _ { \mathrm { f r o n t , 4 0 } m } .\tag{136}
$$

Here, $N _ { \mathrm { v e h , 4 0 } m } , N _ { \mathrm { p e d , 4 0 } m } ,$ and $N _ { \mathrm { b i k e , 4 0 } m }$ denote the numbers of vehicles, pedestrians, and bicycles within 40 m of the ego vehicle. $N _ { \mathrm { f r o n t } , 4 0 m }$ denotes the number of traffic participants located within the forward driving corridor.

Traffic density is divided into Low, Medium, and High groups using category-specific rank terciles.

## D.6 TRAFFIC INTERACTION COMPARISONS

The traffic-interaction analysis compares contextual success between predefined comparison and reference conditions within the same PlanContext category.

For Lead Vehicle, the comparison condition $\mathrm { i s } \leq 1 0$ m and the reference condition is $> 2 5 \mathrm { ~ n ~ }$ m. For TTC Risk, the comparison condition is $< 3$ s and the reference condition is $\geq 6 \ : \mathrm { s }$

Table 10: Robustness of the PlanContext ranking under metric variations.
<table><tr><td>Setting</td><td>Minimum Spearman correlation</td><td>Maximum rank change</td></tr><tr><td> $w \in [ 0 . 4 0 , 0 . 6 0 ]$ </td><td>0.9960</td><td>1</td></tr><tr><td> $w \in [ 0 . 3 0 , 0 . 7 0 ]$ </td><td>0.9703</td><td>5</td></tr><tr><td>Omit B1 from CPS</td><td>0.9960</td><td>2</td></tr><tr><td>Omit B2 from CPS</td><td>0.9970</td><td>1</td></tr><tr><td>Omit C from CPS</td><td>0.9970</td><td>1</td></tr></table>

For Target Occupancy and Crossing Conflict, the comparison condition contains at least two actors and the reference condition contains no actors. Traffic Density compares the High group with the Low group.

All reported differences are computed as

$$
\Delta q = q _ { \mathrm { c o m p a r i s o n } } - q _ { \mathrm { r e f e r e n c e } } .\tag{137}
$$

For Traffic Density, contextual success is pooled over valid model–instance predictions separately within the High and Low groups. We draw 10,000 bootstrap samples from the unique scene tokens within each category, sampling with replacement and retaining all associated scene–condition instances and model predictions together. Each replicate yields a High-minus-Low difference, from which we compute a 95% percentile confidence interval. Asterisks in Table 5 indicate intervals that exclude zero.

The reported differences describe within-category associations between contextual factors and model performance. They do not assume causal effects of the corresponding traffic conditions.

## D.7 METRIC ROBUSTNESS ANALYSIS

We evaluate the sensitivity of model rankings to the relative weighting of NAS and CPS using

$$
\begin{array} { r } { \mathrm { N C S } ( w ) = \mathrm { N A S } ^ { w } \mathrm { C P S } ^ { 1 - w } . } \end{array}\tag{138}
$$

We vary w from 0.30 to 0.70 in increments of 0.01 and compare each ranking with the default setting of $w = 0 . 5$ . Within [0.40, 0.60], the minimum Spearman correlation is 0.9960 and the maximum rank change is one position. Over [0.30, 0.70], these values are 0.9703 and five positions, respectively. OneVL remains the highest-ranked model at every tested weight.

We further examine sensitivity to the composition of CPS by omitting B1, B2, or C in turn and averaging the two remaining category scores. NAS remains fixed in each comparison. The resulting rankings have Spearman correlations of 0.9960, 0.9970, and 0.9970 with the default ranking, with maximum rank changes of two, one, and one positions, respectively.

## D.8 ATOMIC METRIC DECOMPOSITION

We further decompose the final contextual success into its underlying evaluation components. This analysis helps identify whether failures arise from alternative-condition correctness, road compliance, dynamic safety, or the paired contextual response.

Table 11 reports the category-level decomposition pooled over all evaluable model-instance predictions. The pooled LC success rate is 85.64%, compared with the model-averaged rate of 85.51% in Table 3, because the number of evaluable LC predictions varies across models. Alt.-Condition Correctness denotes the category-specific correctness under the alternative navigation condition. It is topology-based for A and L, includes GT-relative suppression for B1, B2, and C, and corresponds to continuity correctness for LC. RoadOK denotes the full-vehicle road-compliance indicator $R _ { i } ^ { F }$ Dynamic Safety denotes the NAVSIM safety indicator $E _ { i } ^ { F }$

Overall Safety requires both road compliance and dynamic safety:

$$
\mathrm { O v e r a l l S a f e t y } _ { i } = R _ { i } ^ { F } \wedge E _ { i } ^ { F } .\tag{139}
$$

Alternative-Condition Success additionally requires the category-specific alternative-condition correctness $A _ { i } ^ { F }$ defined in Sec. C.3:

$$
q _ { i } ^ { \mathrm { a l t } } = A _ { i } ^ { F } \wedge R _ { i } ^ { F } \wedge E _ { i } ^ { F } .\tag{140}
$$

Contextual Success further incorporates the paired contextual decision correctness $C _ { i }$

$$
q _ { i } = C _ { i } \land R _ { i } ^ { F } \land E _ { i } ^ { F } .\tag{141}
$$

Table 11: Atomic metric decomposition across the six PlanContext categories.
<table><tr><td>Category</td><td>Alt. Correct. RoadOK</td><td></td><td>Dynamic Safety</td><td>Overall Safety</td><td>Alt. Success</td><td>Contextual Success</td></tr><tr><td>A</td><td>44.94</td><td>19.49</td><td>66.12</td><td>11.95</td><td>11.44</td><td>10.01</td></tr><tr><td>B1</td><td>94.37</td><td>68.63</td><td>85.43</td><td>63.59</td><td>62.91</td><td>62.91</td></tr><tr><td>B2</td><td>92.50</td><td>79.86</td><td>84.19</td><td>73.95</td><td>73.28</td><td>73.28</td></tr><tr><td>C</td><td>88.12</td><td>46.87</td><td>67.93</td><td>35.85</td><td>35.01</td><td>34.98</td></tr><tr><td>L</td><td>3.51</td><td>97.57</td><td>94.16</td><td>93.55</td><td>1.58</td><td>0.45</td></tr><tr><td>LC</td><td>97.51</td><td>94.32</td><td>87.05</td><td>85.85</td><td>85.64</td><td>85.64</td></tr></table>

• Values are pooled over all evaluable model-instance predictions within each category.

The decomposition reveals distinct failure profiles across contextual requirements. Category A is constrained by both alternative-condition correctness and road compliance. Its Alt.-Condition Correctness is 44.94%, while RoadOK is only 19.49%, resulting in a contextual success rate of 10.01%. Category L shows a different pattern. RoadOK and Dynamic Safety reach 97.57% and 94.16%, respectively, whereas Alt.-Condition Correctness is only 3.51%. This suggests that failures in L primarily arise from the inability to execute the feasible alternative lane intention.

Category C retains relatively high alternative-condition correctness at 88.12%, while RoadOK and Dynamic Safety reduce the final contextual success to 34.98%. B2 and LC achieve the highest contextual success among the six categories at 73.28% and 85.64%, respectively. For B1, the high Alt.-Condition Correctness does not imply that every detected behavioral response is appropriate because longitudinal adjustment and waiting can remain valid under its contextual requirement.

## D.9 HUMAN EXPERT VALIDATION

After the scenario construction and evaluation rules were finalized, three reviewers with backgrounds in autonomous driving independently inspected the same randomly sampled subset of 500 scene–condition instances and 11,500 model-instance evaluations from all 23 evaluated models. The reviewers were blinded to the automated evaluation results during independent assessment. They subsequently discussed their judgments to reach a final consensus.

For scenario construction, the reviewers identified 3 of the 500 scene–condition instances whose assigned contextual categories were inconsistent with their consensus judgments. For model-instance evaluation, the final consensus differed from the automated evaluation outcomes for approximately 0.7% of the 11,500 inspected model-instance evaluations.

The audit also identified ambiguity in the original NAVSIM navigation labels. Further examination of their construction provides context for this observation. NAVSIM assigns directional commands by thresholding the lateral displacement of a point 20 m ahead along the route centerline, with left/right thresholds of ±2 m (Dauner et al., 2024). Its left/right commands encompass turns, lane changes, and sharp curves, leaving the maneuver type unspecified. In the inspected cases, some lane changes were associated with left/right labels, while others were not.

This ambiguity may partly contribute to the poor performance observed in category L, since the same left/right commands can correspond to different maneuver types during training.

## E QUALITATIVE EXAMPLES

## E.1 EXAMPLES ACROSS CONTEXTUAL CATEGORIES

Figures 4–7 present qualitative examples from categories A, B1, B2, and C to illustrate model responses to navigation interventions under different driving contexts. Each figure shows the shared camera observations and trajectories predicted under the original and alternative navigation commands. The displayed scores summarize model performance over the corresponding evaluation sets.

![](images/dc5f3d2d775c5d1239a6082c86957ea8393d89d280855e627248ef11e731b77a.jpg)  
Figure 4: Category A: an immediately actionable alternative route. The navigation command changes from straight to left at an intersection where the alternative left-turn route is reachable within the evaluation horizon. The trajectories illustrate differences in how models respond to the alternative navigation intention.

![](images/6e17283e9317be0bd97c6d0c10c12d221dd82c09dc12fd52f15584ca8223a1a7.jpg)

B1：Straight - Right  
![](images/9190fed5db51ff6ccab9ba9ac0cbaede3fe68fd17317121de0ba17983ce550ab.jpg)  
Figure 5: Category B1: an alternative route beyond the current decision window. The navigation command changes from straight to right before the right-turn target becomes actionable. Continuing along the current road remains appropriate within the evaluation horizon. The trajectories illustrate differences between maintaining the current path and deviating prematurely toward the requested direction.

![](images/14a6f89c3928e2607ca61e44d4a16fc7cf9177dadaec98ce3f06e72c3a1e290b.jpg)  
Figure 6: Category B2: a locally unsupported navigation intention. The navigation command changes from straight to right where the local road topology provides no feasible right-turn route. The trajectories illustrate differences between following the existing road geometry and deviating toward the unsupported direction.

![](images/752e2dcf0088c69dfb2085f221991779bd4d6ce5ad673a1cd299f9e4d02fb316.jpg)  
Figure 7: Category C: a navigation intervention after route commitment. The navigation command changes from left to straight after the ego vehicle has passed the route commitment point. The trajectories illustrate differences between continuing along the committed route and deviating in response to the changed command. Collision annotations identify the marked failures in the displayed predictions.

## E.2 EXAMPLES ACROSS MODELS

We present additional qualitative examples for ten evaluated models in Figures 8–17. Each figure contains three selected scene–condition instances, with camera views and corresponding bird’seye-view visualizations of the predicted trajectories. Green denotes predictions under the original navigation command, while yellow in the camera views and orange in the bird’s-eye views denote predictions under the alternative command. The examples illustrate route choices, responses to navigation interventions, and safety violations across different driving contexts. The displayed mPDMS and NCS values summarize model performance over the corresponding evaluation sets.

![](images/84a4277b06e22fb6c1aa73bc3841df3a6e6d7209f979db1952298cbe46fe6394.jpg)  
Figure 8: Qualitative examples for OneVL. The first two cases show collisions following navigation interventions in categories C and A. The third case shows a directionally correct response in category C.

![](images/87c00ff1bc4f720a28c24febdc6ad221647aa4ff0a36f2a7860fac5d086101f7.jpg)  
Figure 9: Qualitative examples for ChainFlow-VLA. The cases show an incorrect route choice after route commitment in category C, a collision in category B2, and a correct alternative route choice in category A.

![](images/a8289c70ed1912c57e817fde6fdbbcb639e09c9e4d8614ce623288797421d93b.jpg)

Figure 10: Qualitative examples for DrivoR. The first case shows a collision in category B2. The two category A cases illustrate different outcomes when responding to an immediately actionable alternative navigation intention, including an incorrect route choice and a correct route choice.  
![](images/cfdee7434b14844454586734866cea2a7a0642eb6f1810fa6df2b4d5859fa1e6.jpg)  
Figure 11: Qualitative examples for StateMLP. The first two cases show collisions in categories B2 and L. The third case shows a correct alternative route choice in category A.

![](images/f761bc3d639fea122a8f7d7f4acf386dbf4f664376ca1d760032226d4a913c6b.jpg)

Figure 12: Qualitative examples for Curious-VLA. The first two cases show collisions in categories A and C. The third case shows continued travel along the committed route after the navigation command changes in category C.  
![](images/5a4ace9e78bc2ec76ba57081f4ae7d857158510df15f14f961aea6cd10f5d621.jpg)  
Figure 13: Qualitative examples for CLOVER. The cases show an incorrect alternative route choice in category A, a collision in category B2, and a correct alternative route choice in another category A scene.

![](images/9c2117984275ae3bdced0b26fc15bf93f275ff51e2832ecf5208a495c014039a.jpg)

Figure 14: Qualitative examples for WAM-Flow. The two category C cases show a collision and travel against the traffic direction following navigation interventions after route commitment. The category A case shows a correct alternative route choice.  
![](images/0aba05fbebc7cb642dd958e7081e06c8260934b36335b9d4b6ec24e39e5f4682.jpg)  
Figure 15: Qualitative examples for SimWAM. The first case shows an incorrect route choice in category C. The two category A cases show travel against the traffic direction and a correct alternative route choice, respectively.

![](images/22cae68ca18798c6ae66a97b51f6560d4148d53ac86b074871f226caa3df6974.jpg)

Figure 16: Qualitative examples for Drive-JEPA. The cases show a collision in category L, little visible change between the paired predictions in category A, and travel against the traffic direction in category B2.  
![](images/4e9d77efaea13b80c03645ec8a5d701949705b0ce560933314292a0db060f8e9.jpg)  
Figure 17: Qualitative examples for DynVLA Stage 2. The cases show little visible response to a lane-level navigation intervention in category L, an incorrect route choice in category A, and continued travel along the current road in category B2.