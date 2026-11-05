# TIME-TO-GOAL: LEARNED TEMPORAL METRICS FORLONG-HORIZON PLANNING WITH JEPA WORLD MODELS

Ruilong Yu University of Electronic Science and Technology of China yrl666@outlook.com

Fei Ye <sup>∗</sup> University of Electronic Science and Technology of China feiye@uestc.edu.cn

## ABSTRACT

Joint-Embedding Predictive Architectures (JEPAs) support planning by comparing predicted latent states with an encoded goal. For this comparison to guide temporal progress, states closer in time to the goal should receive lower costs. However, training representations for prediction and distribution regularization does not directly enforce this ordering. We examine this mismatch in flat LeWorld-Model (LeWM) by analyzing how its Euclidean goal cost changes with temporal separation. Our analysis identifies conditions under which states at different long offsets receive similar average costs, weakening their temporal distinction. To learn the missing ordering explicitly, we introduce Time-to-Goal (T2G), a lightweight metric trained on latent triples from offline trajectories. A margin ranking loss encourages smaller distances for temporally nearer states, requiring no rewards or policy labels. The learned metric then replaces the Euclidean planning cost while the JEPA encoder and predictor remain frozen. Diagnostics on AntMaze and FourRoom show that T2G preserves stronger temporal contrast and improves near–far ranking at long offsets. Across four navigation and manipulation benchmarks, T2G improves the average planning success rate from 32.7% (flat LeWM) to 63.0% and approaches or even surpasses the performance of the hierarchical H-JEPA baselines. These results show that learning temporal ordering for the goal cost can substantially improve latent planning even with a flat world model. Code is available at https://github.com/yrluestc/T2G-JEPA.

## 1 Introduction

Joint-Embedding Predictive Architectures (JEPAs) [1, 2, 3] learn world models by predicting future observations in a latent space. An encoder maps observations to representations, and an action-conditioned predictor models how these representations evolve under actions. This avoids pixel reconstruction and allows planning by comparing predicted future states with an encoded goal. LeWorldModel (LeWM) [4] implements this with a nextembedding prediction loss and a sketched isotropic Gaussian regularizer (SIGReg), which discourages representation collapse. Its planner optimizes actions using the Euclidean distance between predicted states and the goal. However, as shown in Fig. 1, this flat planner may provide an unreliable progress signal over long horizons, where reliable predictions and a cost that reflects distance to or progress toward a distant goal are both required. A flat model must roll out many prediction steps, leading to error accumulation. Moreover, matching a distant goal in a representation space that also carries low-level details

![](images/d061256b245c553fd3f772d984f13702b70275113ab76df71d38bd035f4e452a.jpg)  
Figure 1: On successful FourRoom executions, Euclidean cost weakly signals initial progress (i.e., distant to goal), whereas T2G provides a more consistent temporal cost.

can be difficult. H-JEPA [5] addresses these challenges with a hierarchy of JEPAs operating at increasing temporal strides in which subgoals divide a long task into shorter planning problems, and higher levels provide more abstract spaces for goal matching. Because this hierarchy changes both how futures are predicted and how they are evaluated, its gains conflate architectural and cost-related effects. This motivates isolating the goal cost and examining whether a better cost estimate alone can improve flat planning in long-horizon settings.

We ask whether the goal-matching cost of a flat model can guide planning without introducing a hierarchy. Intuitively, a useful cost should assign lower values to states that are closer to reaching the goal. However, LeWM’s Euclidean cost is applied to representations trained for next-state prediction and distribution regularization. Its training objective does not explicit impose supervision of progress toward a goal. We therefore examine the Euclidean costs of states along a saved planning trajectory. Specifically, we compare pairs of states separated by an interval of several model steps. Our analysis shows that, as this interval increases, the difference in their Euclidean costs becomes progressively smaller. This weakens the cost signal for ranking candidate futures when the states are far from the goal.

To provide this temporal signal, we propose Time-to-Goal (T2G), a lightweight learned temporal metric for longhorizon planning. T2G learns a small embedding network on flat LeWM latents and uses Euclidean distance in this embedding as a temporal-distance score. It is trained on latent triples sampled from offline trajectories, where a ranking objective encourages a temporally nearer future state to be closer to the anchor state than a farther future state. The LeWM encoder and predictor remainfrozen, and no rewards or policy labels are required during this process. During planning, the same predictor rolls out candidate action sequences, and T2G scores their predicted states by their learned distance to the goal.

Our contributions are summarized as follows.

• We analyze the case where Euclidean latent distances provide little distinction across temporal separations when states are far from the goal.

• We introduce Time-to-Goal, a lightweight learned temporal metric with only 0.182M parameters (1.6% of the flat LeWM), which substantially improves flat LeWM on long-horizon tasks without changing the latent world model.

• Comprehensive experimental results demonstrate that T2G enables flat LeWM to approach or even surpass multi-level H-JEPA in our long-horizon evaluations.

## 2 Methodology

## 2.1 Problem Setup

We consider an offline dataset of trajectories of length T, expressed as:

$$
\mathcal { D } = \{ ( o _ { 1 } , a _ { 1 } , o _ { 2 } , a _ { 2 } , . . . , o _ { T } ) \} .\tag{1}
$$

A frozen JEPA world model consists of an encoder E, which maps observations to latents $\mathbf { z } _ { t } = E ( o _ { t } ) \in \mathbb { R } ^ { D }$ , and an action-conditioned predictor F, where D is the latent dimension. At each replanning call, we index the current latent as $\mathbf { z } _ { 1 }$ and encode the goal observation as ${ \bf z } _ { g } = E ( o _ { g } )$ . The planner samples or optimizes candidate action sequences $a _ { 1 : H } .$ where H is the planning horizon. Each candidate produces a latent rollout $\hat { \mathbf { z } } _ { 2 : H + 1 }$ through

$$
\hat { \mathbf { z } } _ { t + 1 } = F ( \hat { \mathbf { z } } _ { t } , a _ { t } ) , \qquad \hat { \mathbf { z } } _ { 1 } = \mathbf { z } _ { 1 } ,\tag{2}
$$

for $t = \{ 1 , \ldots , H \}$ . The planner selects the action sequence with the lowest goal cost. In the H-JEPA implementation of flat LeWM, this cost is computed over the last-n predicted latents with $n = \operatorname* { m i n } ( n _ { \mathrm { l a s t } } , H )$ :

$$
C _ { \mathrm { f l a t } } ( \widehat { \mathbf { z } } _ { 2 : H + 1 } ) = \sum _ { t = H - n + 2 } ^ { H + 1 } \lVert \widehat { \mathbf { z } } _ { t } - \mathbf { z } _ { g } \rVert _ { 2 } ^ { 2 } ,\tag{3}
$$

where $n _ { \mathrm { l a s t } }$ is the predefined cost window and $| | \cdot | | _ { 2 } ^ { 2 }$ denotes the L2 norm. We seek a replacement cost that directly reflects relative progress toward the goal while keeping the encoder and predictor fixed.

## 2.2 Why Euclidean Distance May Provide Weak Temporal Guidance

The cost in Eq. (3) guides action search by preferring predicted states with smaller squared Euclidean distances to the goal. To reflect temporal progress, this cost should generally assign lower values to states that are fewer steps from the same goal along a trajectory. LeWM trains its encoder and predictor with a next-embedding prediction loss and SIGReg:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { p r e d } } = \Vert \hat { \mathbf { z } } _ { t + 1 } - \mathbf { z } _ { t + 1 } \Vert _ { 2 } ^ { 2 } , \qquad \mathcal { L } _ { \mathrm { L e W M } } = \mathcal { L } _ { \mathrm { p r e d } } + \lambda \mathrm { S I G R e g } ( Z ) , } \end{array}\tag{4}
$$

![](images/ff0bf0732a3a55972637622e3e58b3a9799612d98ef62978738c47ea32fbb1c4.jpg)

![](images/64f917371868a24afd19f6cd35744b6a8d8f198d7a1394da3270388c61881ca6.jpg)  
Figure 2: Changes in the components of Euclidean temporal contrast on AntMaze (left) and FourRoom (right), using 50 recorded planner executions per environment. The horizontal axis is the separation k from a sampled future goal in model steps, with a fixed comparison gap of $\Delta = 1$ model step (five environment steps). Curves show $| \delta _ { \Delta } \bar { B ( k ) | }$ and $| \delta _ { \Delta } q ( k ) |$ , each divided by the environment’s mean Euclidean squared distance $\overline { { D } } _ { E }$ over the tested offsets. In both environments, the B term varies little, while changes in the q term become smaller at longer offsets.

where $\hat { \mathbf { z } } _ { t + 1 } = F ( \mathbf { z } _ { t - H + 1 : t } , a _ { t - H + 1 : t } ) , Z \in \mathbb { R } ^ { N \times D }$ collects N latent embeddings, and $\lambda > 0$ . SIGReg projects the embeddings onto M random unit directions $u ^ { ( m ) }$ on the unit sphere $S ^ { D - 1 } \subset \mathbb { R } ^ { D }$ , and matches each one-dimensional projection to a standard Gaussian with an Epps–Pulley [6] statistic $T _ { \mathrm { E P } } { \mathrm { : } }$

$$
\operatorname { S I G R e g } ( Z ) = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } T _ { \mathrm { E P } } ( Z u ^ { ( m ) } ) .\tag{5}
$$

The prediction loss encourages accurate latent dynamics, while SIGReg controls the embedding distribution to discourage collapse. Neither objective explicitly requires a state closer in time to a goal to have a smaller distance to it. To understand the resulting cost signal, we examine how the average squared Euclidean distance changes with tempora separation. Let $( X _ { k } , Y _ { k } )$ denote an earlier encoded state and a future goal sampled from the same trajectory, separated by k model steps. Write the means of the paired states as $\mu _ { X } ( k ) , \mu _ { Y } ( k )$ ), their covariances as $\Sigma _ { X } ( k ) , \bar { \Sigma } _ { Y } ( \dot { k } )$ , and their cross-covariance as $C ( k )$ . For finite second moments, expanding the squared distance yields:

$$
\begin{array} { r l } & { D _ { E } ( k ) : = \mathbb { E } \| X _ { k } - Y _ { k } \| _ { 2 } ^ { 2 } } \\ & { \qquad = \| \mu _ { X } ( k ) - \mu _ { Y } ( k ) \| _ { 2 } ^ { 2 } + \operatorname { t r } ( \Sigma _ { X } ( k ) ) + \operatorname { t r } ( \Sigma _ { Y } ( k ) ) - 2 \operatorname { t r } ( C ( k ) ) } \\ & { \qquad = : B ( k ) - 2 q ( k ) , } \end{array}\tag{6}
$$

where $B ( k )$ collects the squared mean difference and the two marginal variance traces, and $q ( k ) = \operatorname { t r } ( C ( k ) )$ . Thus, the average distance depends on both the distributions of the individual states and how the paired states vary together.

For planning, the relevant question is whether this distance distinguishes states that are closer to the same future goal. Holding the sampled future goals fixed, we compare states at offsets k and $k + \Delta$ , where $\Delta > 0$ . The difference in their average costs, which we call the temporal contrast, is

$$
S _ { E } ( k , \Delta ) : = D _ { E } ( k + \Delta ) - D _ { E } ( k ) = \delta _ { \Delta } B ( k ) - 2 \delta _ { \Delta } q ( k ) ,\tag{7}
$$

where $\delta _ { \Delta } f ( k ) = f ( k + \Delta ) - f ( k )$ . Positive contrast means that the temporally farther states receive higher costs on average. The triangle inequality gives:

$$
| S _ { E } ( k , \Delta ) | \le | \delta _ { \Delta } B ( k ) | + 2 | \delta _ { \Delta } q ( k ) | .\tag{8}
$$

This bound explains why Euclidean cost can provide weak temporal guidance at large k: if $B ( k )$ and $q ( k )$ change little as the offset increases from k to $k + \Delta$ , states at these two offsets from the same goal receive similar average costs, making their temporal proximity less distinguishable through Euclidean cost.

Empirically, Fig. 2 shows this behavior on 50 recorded planner executions each in AntMaze and FourRoom. We encode observations sampled every model step and use the same future goals across offsets $k = 1 , \ldots , 2 3$ , comparing adjacent offsets with $\Delta \bar { = } 1$ . Results are averaged within each trajectory and then equally across trajectories, and both components are normalized by the mean squared distance over the tested offsets. In both environments, $| \delta _ { \Delta } B ( k )$ | stays small, while $| \delta _ { \Delta } q ( k )$ | decreases markedly overall as k grows. Consequently, the bound becomes small at long offsets, explaining the weak average Euclidean cost signal for distinguishing states which are distant to goal in Fig. 1.

These observations motivate the objective of explicitly teaching the latent world model the temporal structure of progress toward the goal.

![](images/af47ce1e01e7f98faa4aaaae15dac8ddfa978a70dabb8a38fdd7d2c330dcec9e.jpg)  
Figure 3: Overview of T2G metric training. A latent triple $\left( \mathbf { z } _ { t } , \mathbf { z } _ { t + k } , \mathbf { z } _ { t + k + \Delta } \right)$ is sampled from an offline trajectory, where the near and far states are separated from the anchor by k and $k + \Delta$ steps, respectively. The three encoder blocks denote the same frozen JEPA encoder applied to the corresponding observations. T2G evaluates the learned temporal distances from the anchor to the near and far latents and optimizes the metric with a margin ranking objective, so that the near state is closer to the anchor than the far state by at least a margin m. Only the metric network and its scale are updated, and the world model remains frozen.

## 2.3 Learning A Time-to-Goal Cost

To give the planner an explicit measure of temporal proximity, we propose T2G to learn a small mapping of the frozen JEPA representations. Let θ denote its trainable parameters. We define the distance between two states as a scaled Euclidean distance in the learned embedding space:

$$
d _ { \boldsymbol { \theta } } ( \mathbf { z } _ { a } , \mathbf { z } _ { b } ) = \alpha _ { \boldsymbol { \theta } } \| f _ { \boldsymbol { \theta } } ( \mathbf { z } _ { a } ) - f _ { \boldsymbol { \theta } } ( \mathbf { z } _ { b } ) \| _ { 2 } ,\tag{9}
$$

where $f _ { \theta }$ is a small MLP and $\alpha _ { \theta } > 0$ is a learned scale. The learned mapping reshapes the latent geometry through temporal-order supervision. The Euclidean norm preserves non-negativity, symmetry, and the triangle inequality:

$$
\begin{array} { r } { d _ { \theta } ( a , c ) \leq d _ { \theta } ( a , b ) + d _ { \theta } ( b , c ) . } \end{array}\tag{10}
$$

We teach this metric to distinguish nearer and farther states using the time order of offline trajectories. The frozen encoder first maps each trajectory clip to a sequence of latent states. Within that clip, we sample three states: an anchor state $\mathbf { z } _ { t } .$ , with a near state $\mathbf { z } _ { t + k }$ and afar state ${ \bf z } _ { t + k + \Delta }$ . The near state occurs k model steps after the anchor, and the far state occurs another $\Delta$ steps later. These three states form the training triple:

$$
\underbrace { \mathbf { z } _ { t } } _ { \mathrm { a n c h o r } } , \underbrace { \mathbf { z } _ { t + k } } _ { \mathrm { n e a r } } , \underbrace { \mathbf { z } _ { t + k + \Delta } } _ { \mathrm { f a r } } ) .\tag{11}
$$

The desired comparison is simple: the learned distance from the anchor state to the near state should be smaller than its distance to the far state. We train the metric to satisfy the near–far comparison with a margin ranking loss

$$
\mathcal { L } _ { \mathrm { r a n k } } ( \theta ) = \mathbb { E } _ { t , k , \Delta } \left[ \operatorname* { m a x } \left( 0 , m + d _ { \theta } ( \mathbf { z } _ { t } , \mathbf { z } _ { t + k } ) - d _ { \theta } ( \mathbf { z } _ { t } , \mathbf { z } _ { t + k + \Delta } ) \right) \right] ,\tag{12}
$$

where $m > 0$ is fixed during training. A triple incurs a penalty when the far state is not farther from the anchor than the near state by at least m model step. The positive scale $\alpha \theta$ in Eq. (9) adjusts the magnitude of distance differences relative to $m .$ . We average this penalty over the minibatch and update only the metric parameters θ, while the encoder $E$ and predictor $F$ remain frozen. The supervision comes entirely from trajectory order, with no rewards or policy labels. Since the loss specifies relative distances rather than a target number of steps, $d _ { \theta }$ is a relative temporal cost. Once trained, it serves as the goal-matching cost for the frozen world model.

## 2.4 Planning With the Time-to-Goal Cost

At planning time, we replace the Euclidean goal cost of flat LeWM with the learned temporal distance. Starting from the observed latent history ending at $\mathbf { z } _ { 1 }$ and the goal latent $\mathbf { z } _ { g } .$ , the planner rolls out candidate action sequences $a _ { 1 : H }$ with the frozen predictor to obtain $\hat { \mathbf { z } } _ { 2 : H + 1 }$ . The last-n T2G cost averages the temporal distances to the goal over the final n predicted states:

$$
C _ { \mathrm { T 2 G } } ^ { ( n ) } ( \widehat { \mathbf { z } } _ { 2 : H + 1 } ) = \frac { 1 } { n } \sum _ { t = H - n + 2 } ^ { H + 1 } d _ { \theta } ( \widehat { \mathbf { z } } _ { t } , \mathbf { z } _ { g } ) ,\tag{13}
$$

Table 1: Planning settings of the proposed T2G. Episode budgets are in environment steps, H is the beginning planning horizon in predicted steps, S is the number of candidate action sequences, and η is the AdamW learning rate. Last-n is the number of final predicted states averaged by the T2G last-n cost. Candidate counts, learning rates, and episode budgets follow the flat LeWM baseline of H-JEPA [5]; Cube uses an initial horizon of two.
<table><tr><td>Environment</td><td>Episode budget</td><td>H</td><td>Last-n</td><td>S</td><td>η</td></tr><tr><td>FourRoomDistractors</td><td>210</td><td>23</td><td>2</td><td>200</td><td>0.1</td></tr><tr><td>Visual AntMaze</td><td>350</td><td>23</td><td>2</td><td>100</td><td>0.3</td></tr><tr><td>Push-T</td><td>90</td><td>15</td><td>2</td><td>800</td><td>0.05</td></tr><tr><td>OGBench Cube</td><td>30</td><td>2</td><td>1</td><td>800</td><td>0.05</td></tr></table>

![](images/c799c531d6b563177abe99247b320404150e8d4e2d59b3558bd26ecc33991144.jpg)  
Figure 4: Overview of T2G planning. The current observation $o _ { 1 }$ and goal observation $o _ { g }$ are encoded into latents $\mathbf { z } _ { 1 }$ and $\mathbf { z } _ { g }$ by the frozen JEPA encoder. A solver optimizes an action sequence $a _ { 1 : H }$ , and the frozen predictor rolls out the resulting latent trajectory $\mathbf { z } _ { 1 } \to \mathbf { z } _ { 2 } \to \cdot \cdot \cdot \to \mathbf { z } _ { H }$ . Each predicted latent is scored by its learned temporal distance to the goal, $d _ { \boldsymbol { \theta } } ( \mathbf { z } _ { t } , \mathbf { z } _ { g } )$ , and the solver updates the actions to minimize the aggregated cost. The updated actions are executed and planning is repeated from new observations in a receding-horizon loop, while the encoder, predictor, and solver remain unchanged.

where $1 \leq n \leq H$ . This cost focuses on the final portion of the rollout, using the learned temporal distance to evaluate how close those states are to the goal.

We also consider a dense T2G variant that averages over the entire rollout:

$$
C _ { \mathrm { T 2 G } } ( \widehat { \mathbf { z } } _ { 2 : H + 1 } ) = \frac { 1 } { H } \sum _ { t = 2 } ^ { H + 1 } d _ { \theta } ( \widehat { \mathbf { z } } _ { t } , \mathbf { z } _ { g } ) .\tag{14}
$$

The dense cost is the special case $n = H$ , incorporating intermediate predicted states as well as the final states.

Both costs can be used by sampling-based and gradient-based planners. Sampling-based planners $( \mathrm { e . g . } ,$ CEM [7]) select the lowest-cost candidate, whereas the gradient-based planner in H-JEPA minimizes the cost over actions. The planner then executes the first action block(s) and replans in a receding-horizon fashion.

## 3 Experiments

## 3.1 Planning Evaluation Setup

Datasets and Tasks. We evaluate T2G on four simulated benchmarks used by H-JEPA [5]: FourRoomDistractors [8], Visual AntMaze [9], Push-T [10], and OGBench Cube [9]. These cover navigation with distractors, quadruped maze navigation, planar manipulation, and 3D object manipulation, respectively. We follow H-JEPA’s task construction. FourRoomDistractors and Push-T use held-out expert frame pairs separated by 75 environment steps, with the FourRoom start and goal located two rooms apart. Cube uses held-out 20-step trajectory segments centered on a grasp. AntMaze uses expert rollouts between maze cells at grid distance three, spanning about 82 environment steps on average, with neutral-pose start and goal states. The planner receives a goal image and additionally uses proprioceptive observations on AntMaze. We evaluate on 50 start–goal tasks per environment, shared across all methods and hierarchy depths. Success is determined by the environment’s goal test within the episode budget; on AntMaze, the agent must reach within 1.0 of the goal xy position. Planner settings are fixed before the final evaluation.

Planner and Baselines. We compare flat LeWM with its Euclidean goal cost, flat LeWM with last-n and dense T2G costs, and H-JEPA with two, three, and four levels. All flat variants use the gradient-based planner with AdamW. On FourRoom, AntMaze, and Push-T, T2G retains the baseline planning settings and replaces the goal cost. On Cube, T2G uses a horizon of $H = 2$ model steps, replacing the baseline’s initial horizon of 4. The configured last-n window is 2 predicted states on AntMaze, FourRoom, and Push-T, and 1 on Cube; the effective window is capped by the current horizon. Tab. 1 summarizes the episode budgets, T2G horizons, window sizes, candidate counts S, and AdamW learning rates η. All methods share the evaluation tasks and environment-step budgets. We report planning success rate (%) as the mean and standard error across seeds 42, 43, and 44.

Table 2: Planning success rate (%) on the evaluation tasks, averaged over multiple seeds. LeWM (flat) uses the native level-1 cost; the $\bar { 2 } \cdot , 3 \cdot ,$ , and 4-level rows use hierarchical H-JEPA planning, and the T2G rows use the same frozen flat world model with either a full-rollout (dense) or final-state-window (last-n) temporal cost. Window sizes are given in Tab. 1. New Params denotes the number of additional parameters in millions (M) compared to LeWM (flat). Cell shading continuously interpolates from light red to deep red using min–max normalization independently within each column. Higher mean success rates receive darker shading; the scale is reversed for New Params, where lower values are better. Standard errors do not affect shading. The missing entry (–) is unshaded and excluded from normalization.
<table><tr><td>Method</td><td>New Params ↓</td><td></td><td></td><td>Visual AntMaze↑ OGBench Cube↑ FourRoomDistractors↑</td><td>Push-T↑</td></tr><tr><td>H-JEPA (2 levels)</td><td>10.79</td><td> $3 9 . 3 \pm 3 . 7$ </td><td> $4 7 . 3 \pm 2 . 7$ </td><td> $8 1 . 3 \pm 6 . 7$ </td><td> $4 5 . 3 \pm 1 . 3$ </td></tr><tr><td>H-JEPA (3 levels)</td><td>21.58</td><td> $7 3 . 3 \pm 3 . 5$ </td><td> $6 0 . 0 \pm 5 . 0$ </td><td> $9 6 . 0 \pm 1 . 1$ </td><td> $1 7 . 3 \pm 3 . 5$ </td></tr><tr><td>H-JEPA (4 levels)</td><td>32.37</td><td> $6 3 . 3 \pm 7 . 7$ </td><td> $5 5 . 3 \pm 4 . 4$ </td><td> $8 0 . 0 \pm 4 . 0$ </td><td> $0 . 7 \pm 0 . 7$ </td></tr><tr><td>LeWM (flat)</td><td></td><td> $1 8 . 0 \pm 3 . 5$ </td><td> $3 2 . 0 \pm 6 . 1$ </td><td> $4 0 . 7 \pm 6 . 8$ </td><td> $4 0 . 0 \pm 2 . 3$ </td></tr><tr><td>+ T2G dense</td><td>0.18</td><td> $4 8 . 3 \pm 2 . 4$ </td><td> $4 . 0 \pm 1 . 2$ </td><td> $8 1 . 3 \pm 2 . 9$ </td><td> $4 2 . 0 \pm 1 . 2$ </td></tr><tr><td>+ T2G lastn</td><td>0.18</td><td> $5 8 . 7 \pm 2 . 4$ </td><td> $6 3 . 3 \pm { 0 . 7 }$ </td><td> $7 8 . 7 \pm 2 . 9$ </td><td> $5 1 . 3 \pm 5 . 9$ </td></tr></table>

![](images/afb6f157802481edefe4baf228f8dd15f9dcb46b8e51bcdbae45a1fb730fbaeb.jpg)  
Figure 5: Selected successful LeWM+T2G planner executions using our T2G cost, on one evaluation task per environment.

## 3.2 Planning Performance Across Different Environments

Tab. 2 compares flat LeWM, two T2G cost variants, and H-JEPA with two to four levels. Specifically, T2G last-n improves the mean success rate over flat LeWM in all four environments: from 18.0% to 58.7% on AntMaze, 32.0% to 63.3% on Cube, 40.7% to 78.7% on FourRoom, and 40.0% to 51.3% on Push-T. Its unweighted average across environments is 63.0%, compared with 32.7% for flat LeWM and 53.3%, 61.7%, and 49.8% for two-, three-, and four-level H-JEPA, respectively (summarize the reported means). Both T2G variants retain the flat world model and add only a small metric network.

![](images/b2d7d3cc1370b0e9460a49891eb68b52f39c2bad27e19125c66526cb045e0b30.jpg)  
Figure 6: Temporal guidance of Euclidean squared and T2G costs on recorded AntMaze (top) and FourRoom (bottom) executions. Left: mean cost to the same sampled future goals, normalized by each cost’s mean over the tested offsets. Middle: normalized temporal contrast between offsets separated by one model step (five environment steps). Right: the fraction of comparisons in which the temporally nearer state receives a lower cost, with ties receiving half credit; the dashed line marks 50%. Horizontal axes report temporal separation in environment steps. At long offsets, T2G maintains a larger positive contrast and higher ranking accuracy in both environments.

Dense T2G also has substantial effect on several environments. It improves AntMaze, FourRoom, and Push-T to 48.3%, 81.3%, and 42.0%, respectively, but reduces Cube success to 4.0%. Restricting the cost to the final n predicted states (last-n T2G) raises the corresponding means on AntMaze, Cube, and Push-T by 10.4%, 59.3%, and 9.3%, while FourRoom slightly decreases by 2.6%. Thus, uniformly aggregating full-rollout temporal distance does not make planning consistently effective. In particular, dense aggregation encourages proximity to the goal at every predicted step, including the initial approach and grasp preparation of Cube. These necessary intermediate states may still receive large T2G distances to the goal. Penalizing them can favor actions that reduce the early cost at the expense of completing the grasp, whereas last-1 scoring allows the planner to optimize these preparatory actions for their effect on the predicted endpoint. This provides an explanation for the large advantage of terminal-state scoring on Cube.

Relative to hierarchy, T2G last-n has the highest reported mean on Cube and Push-T, exceeding the best H-JEPA means by 3.3% and 6.0%. T2G substantially narrows the performance gap between flat LeWM and hierarchical planning on the navigation tasks. On FourRoom, dense T2G raises success from 40.7% to 81.3%, matching two-level H-JEPA. Three-level H-JEPA achieves higher success on both AntMaze and FourRoom, at 73.3% and 96.0%, respectively. Together, these results show that a learned temporal cost can recover substantial performance within a flat architecture and motivate combining T2G with hierarchical planning to further improve navigation.

## 3.3 Temporal Guidance in AntMaze and FourRoom

To examine the temporal guidance discussed in Section 2.2, Fig. 6 compares Euclidean squared and T2G costs on 50 recorded executions per environment from flat LeWM with T2G last-2. Both costs score the same encoded observations and sampled future goals. Within each trajectory, the goal set is held fixed across offsets $k = 1 , \ldots , 2 3$ model steps, corresponding to 5–115 environment steps on the figure axes. We compare adjacent offsets with $\Delta = 1$ model step and average first within each trajectory and then equally across trajectories. Each cost curve and its temporal contrast are divided by that cost’s mean over the tested offsets, enabling comparison of their shapes on a common relative scale.

The left panels show that both costs increase overall with temporal separation. The Euclidean curves, however, become progressively flatter at long offsets, particularly in FourRoom, while T2G continues to distinguish successive offsets. The middle panels quantify this difference through the temporal contrast in Eq. (7). Over the last five comparisons, $k = 1 8 , \ldots , 2 2$ , the mean normalized contrast is 0.0202 for Euclidean cost and 0.0481 for T2G on AntMaze, and

0.0036 versus 0.0214 on FourRoom. Thus, T2G preserves a stronger average cost difference between temporally nearer and farther states in both environments. The effect is especially pronounced in FourRoom, where the Euclidean contrast fluctuates around zero at the largest offsets. These observations agree with the component analysis in Fig. 2, which links weak Euclidean contrast to small changes in B(k) and q(k).

The right panels assess whether the cost also orders individual state pairs correctly: for the same goal, does the state at offset k receive a lower cost than the state at k + 1? Euclidean cost provides useful ordering at short offsets, but its accuracy decreases toward 50% as the separation grows. Over the same five long-offset comparisons, T2G raises ranking accuracy from 53.44% to 67.69% on AntMaze and from 50.27% to 61.53% on FourRoom, gains of 14.25 and 11.26 percentage points, respectively. T2G’s advantage therefore emerges most clearly when the goal is temporally distant, where the original Euclidean cost becomes less discriminative.

Together, these measurements connect the temporal-contrast analysis to the ordering learned by T2G: the learned cost provides both larger average differences and more accurate preferences for temporally nearer states at long offsets. This supplies the flat planner with a clearer signal for comparing candidate futures using the frozen predictive representation.

## 4 Related Works

JEPAs learn task-agnostic world models by predicting future observations in a latent space and have been extended to action-conditioned latent planning [1, 2, 3]. DINO-WM trains dynamics on frozen visual features [11]; PLDM learns latent dynamics with auxiliary regularization [12]; V-JEPA 2 attaches a predictor to a pretrained video JEPA [13]; and LeWM trains an encoder and predictor end-to-end with SIGReg [4]. These reward-free models are typically paired with MPC, which scores candidate rollouts using CEM [7] or gradient-based optimization. Reward-based approaches such as Dreamer, TD-MPC2, and MuZero instead learn models for reward or value maximization [14, 15, 16]. To handle long horizons, hierarchical methods decompose planning across timescales, from options [17] to HWM, which plans over multiple horizons in a shared latent space [18], and H-JEPA, which stacks JEPAs with increasing temporal strides and predicts subgoals [5]. T2G adds no hierarchy and changes no world model; it changes only the cost used by a flat planner.

Temporal progress has also been studied through time-contrastive and predictive objectives [19, 20], goal-conditioned value functions and relabeling [21, 22], and quasimetric RL. These methods typically learn a policy or value function and rely on interaction or reward. T2G instead learns a embedded metric from offline latents of a frozen JEPA model with a margin ranking objective, requires no reward or policy signal, and serves directly as a planning cost for sampling or gradient-based planners. This isolates cost geometry from temporal abstraction and complements hierarchy.

## 5 Conclusion

The central finding of this work is that the cost used to evaluate predicted states plays an important role in the effectiveness of latent planning. T2G demonstrates that trajectory order provides useful supervision for shaping this cost, allowing a frozen predictive representation to offer clearer temporal guidance. The comparison between dense and last-n costs further shows that the choice of scored states matters: emphasizing the final predicted states can accommodate intermediate actions needed to complete a task. These findings make learning and aggregating goal costs a practical part of designing planners with JEPA world models.

Future work will combine T2G with hierarchical planning by learning temporal costs at multiple levels to guide both subgoal selection and action optimization. This would bring the improved temporal ordering of T2G together with the longer planning range provided by temporal abstraction. We also plan to extend the approach to physical robots, using offline robot trajectories to learn the temporal metric and evaluating it during closed-loop task execution. Such experiments will examine how the learned cost supports progress toward a goal under real visual variation, sensor noise, and contact dynamics.

## Acknowledgments

This was was supported in part by National Natural Science Foundation of China (62506067).

## References

[1] Yann LeCun et al. A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27. Open Review, 62(1):1–62, 2022.

[2] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15619–15629. IEEE, 2023.

[3] Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video. arXiv preprint arXiv:2404.08471, 2024.

[4] Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, and Randall Balestriero. Leworldmodel: Stable end-to-end joint-embedding predictive architecture from pixels. arXiv preprint arXiv:2603.19312, 2026.

[5] Wancong Zhang, Basile Terver, Michael Rabbat, Yann LeCun, and Randall Balestriero. H-jepa: End-to-end learning of hierarchical world models for visual planning. arXiv preprint arXiv:2610.06805, 2026.

[6] Thomas W Epps and Lawrence B Pulley. A test for normality based on the empirical characteristic function. Biometrika, 70(3):723–726, 1983.

[7] Reuven Y Rubinstein and Dirk P Kroese. The cross-entropy method: a unified approach to combinatorial optimization, Monte-Carlo simulation, and machine learning, volume 133. Springer, 2004.

[8] Uladzislau Sobal, Wancong Zhang, Kyunghyun Cho, Randall Balestriero, Tim GJ Rudner, and Yann LeCun. Learning from reward-free offline data: A case for planning with latent dynamics models. Advances in Neural Information Processing Systems, 38:43905–43941, 2026.

[9] Seohong Park, Kevin Frans, Benjamin Eysenbach, and Sergey Levine. Ogbench: Benchmarking offline goalconditioned rl. In International Conference on Learning Representations, volume 2025, pages 94937–94982, 2025.

[10] Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal ofRobotics Research, 44(10-11):1684–1704, 2025.

[11] Gaoyue Zhou, Hengkai Pan, Yann LeCun, and Lerrel Pinto. Dino-wm: World models on pre-trained visual features enable zero-shot planning. arXiv preprint arXiv:2411.04983, 2024.

[12] Uladzislau Sobal, Wancong Zhang, Kyunghyun Cho, Randall Balestriero, Tim GJ Rudner, and Yann LeCun. Learning from reward-free offline data: A case for planning with latent dynamics models. Advances in Neural Information Processing Systems, 38:43905–43941, 2026.

[13] Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, et al. V-jepa 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025.

[14] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse domains through world models. arXiv preprint arXiv:2301.04104, 2023.

[15] Nick Hansen, Hao Su, and Xiaolong Wang. Td-mpc2: Scalable, robust world models for continuous control. In International Conference on Learning Representations, volume 2024, pages 47376–47405, 2024.

[16] Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, et al. Mastering atari, go, chess and shogi by planning with a learned model. Nature, 588(7839):604–609, 2020.

[17] Richard S Sutton, Doina Precup, and Satinder Singh. Between mdps and semi-mdps: A framework for temporal abstraction in reinforcement learning. Artificial intelligence, 112(1-2):181–211, 1999.

[18] Wancong Zhang, Basile Terver, Artem Zholus, Soham Chitnis, Harsh Sutaria, Mido Assran, Randall Balestriero, Amir Bar, Adrien Bardes, Yann LeCun, et al. Hierarchical planning with latent world models. arXiv preprint arXiv:2604.03208, 2026.

[19] Pierre Sermanet, Corey Lynch, Yevgen Chebotar, Jasmine Hsu, Eric Jang, Stefan Schaal, Sergey Levine, and Google Brain. Time-contrastive networks: Self-supervised learning from video. In 2018 IEEE international conference on robotics and automation (ICRA), pages 1134–1141. IEEE, 2018.

[20] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.

[21] Marcin Andrychowicz, Filip Wolski, Alex Ray, Jonas Schneider, Rachel Fong, Peter Welinder, Bob McGrew, Josh Tobin, OpenAI Pieter Abbeel, and Wojciech Zaremba. Hindsight experience replay. Advances in neural information processing systems, 30, 2017.

[22] Dibya Ghosh, Abhishek Gupta, Ashwin Reddy, Justin Fu, Coline Devin, Benjamin Eysenbach, and Sergey Levine. Learning to reach goals via iterated supervised learning. arXiv preprint arXiv:1912.06088, 2019.

Table 3: Input and sampling hyper-parameters for T2G training. Image size is in pixels. Clip length $T _ { \mathrm { c l i p } }$ counts latent states, and $K _ { \mathrm { m a x } }$ is measured in model steps. All environments use $\bar { k } _ { \operatorname* { m i n } } = 1 , \bar { \Delta _ { \operatorname* { m i n } } } = 2 .$ , and five environment steps per model step.
<table><tr><td>Environment</td><td>Latent dimension D</td><td>Image size</td><td> $T _ { \mathrm { c l i p } }$ </td><td> $K _ { \mathrm { m a x } }$ </td></tr><tr><td>AntMaze</td><td>384</td><td> $6 4 \times 6 4$ </td><td>24</td><td>23</td></tr><tr><td>FourRoom</td><td>192</td><td> $2 2 4 \times 2 2 4$ </td><td>24</td><td>23</td></tr><tr><td>Cube</td><td>192</td><td> $2 2 4 \times 2 2 4$ </td><td>16</td><td>15</td></tr><tr><td>Push-T</td><td>192</td><td> $2 2 4 \times 2 2 4$ </td><td>16</td><td>15</td></tr></table>

## A T2G training details

Device. All experiments were conducted on the same hardware environment running Ubuntu 22.04.2 LTS, with 256 GB of RAM and Intel Xeon Silver4320. A single NVIDIA A100 GPU provides the computing acceleration in experiments.

Sampling process during training. For each clip, let $T _ { \mathrm { c l i p } }$ be the number of latent states and let $K =$ min $\left( K _ { \operatorname* { m a x } } , T _ { \mathrm { c l i p } } - 1 \right)$ be the largest usable offset from the anchor. Here, $K _ { \mathrm { m a x } }$ is the configured maximum offset, $k _ { \mathrm { m i n } }$ is the minimum near-state offset, and $\Delta _ { \mathrm { m i n } }$ is the minimum near–far gap; all offsets are measured in model steps. We require $K \ge k _ { \operatorname* { m i n } } + \Delta _ { \operatorname* { m i n } } ,$ so that the clip can accommodate both a near and a far state. To sample a triple, we draw the separations first and then the anchor:

1. Draw the near-state offset k uniformly from $\{ k _ { \operatorname* { m i n } } , \dots , K - \Delta _ { \operatorname* { m i n } } \}$ . The upper bound leaves at least $\Delta _ { \mathrm { m i n } }$ steps between the near and far states.

2. Draw an additional gap e uniformly from $\{ \Delta \mathrm { { m i n } } , \ldots , K \}$

3. Set the far-state offset to $k _ { \mathrm { f a r } } = \operatorname* { m i n } ( k + e , K )$ , and set $\Delta = k _ { \mathrm { f a r } } - k$ . Hence $\Delta \ge \Delta _ { \mathrm { m i n } }$ and $k _ { \mathrm { f a r } } \leq K$ . If $k + e > K$ , the far offset is clipped to $K$

4. Draw the anchor index t uniformly from $\{ 0 , \dots , T _ { \mathrm { c l i p } } - 1 - k _ { \mathrm { f a r } } \}$ , so that the far state remains inside the clip.

The resulting triple is $\left( { { z } _ { t } } , { { \bf { z } } _ { t + k } } , { { \bf { z } } _ { t + k + \Delta } } \right)$ . Each clip in a minibatch contributes one newly sampled triple. Clipping gives extra probability to the maximum far-state offset, increasing exposure to long temporal separations within the training window.

Hyper-parameter settings. For each environment, T2G is trained on the corresponding offline training split (provided by https://huggingface.co/datasets/jepa-world-models/h-jepa/tree/main) using the frozen flat LeWM encoder and predictor (provided by https://huggingface.co/jepa-world-models $/ \mathrm { h - j e p a ) }$ . The metric mapping $f _ { \theta }$ consists of two hidden blocks, each containing a linear layer of width 256, LayerNorm, and GELU, followed by a linear output layer of dimension 256. Dropout is set to zero. The input dimension follows the LeWM representation: 384 for AntMaze and 192 for the other environments. The positive distance scale $\alpha \theta$ is parameterized with softplus and initialized to 1. The computed distance is lower-bounded by $1 0 ^ { - 6 }$ before scaling.

All four environments use AdamW with a fixed learning rate of $5 \times 1 0 ^ { - 5 }$ , weight decay $1 0 ^ { - 3 }$ , batch size 128, and 5 training epochs. We minimize the margin ranking loss with m = 1 and clip the gradient norm of the metric parameters to 1.0. Training clips are shuffled at each epoch, with incomplete final minibatches dropped. Each clip contributes one newly sampled triple per update, giving 128 triples per minibatch. Only the metric mapping and its scale are optimized; the world model is kept in evaluation mode. Each metric uses the original LeWM checkpoint.

Tab. 3 lists the environment-specific input and sampling settings. Observations are sampled every five environment steps, so one model step corresponds to five environment steps in all four environments. We set $k _ { \operatorname* { m i n } } = 1$ and $\Delta _ { \operatorname* { m i n } } = 2$ throughout. The latter is the minimum near–far gap, and the actual gap $\Delta$ varies across sampled triples. The maximum training offset $K _ { \mathrm { m a x } }$ specifies the sampling range and is distinct from the planning horizon H $H$