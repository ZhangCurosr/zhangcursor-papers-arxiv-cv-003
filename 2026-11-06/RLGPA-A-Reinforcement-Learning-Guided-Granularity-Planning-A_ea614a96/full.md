# RLGPA: A Reinforcement Learning-Guided Granularity Planning Attack for Decision-Based Sparse Adversarial Examples

Chao Zhou, Shenghua He, Jiaxin Mo, Qiangguo Yu, Fengchuang Xing

Abstract—Decision-based sparse attacks must reduce perturbation support using only the victim’s predicted label. The best restoration strategy varies as support contracts: coarse removal works early, while tail states need finer, history-aware tests. We propose RLGPA, an online reinforcement learning method that schedules support-reduction executors and their granularity under a hard query budget. A masked Dueling Double DQN selects among seven stable policy–scale actions; each action produces support-relative restoration groups that are rebuilt after every macro decision. A feasibility-preserving executor commits only adversarial restorations and feeds support gain and query cost back to the planner. Under the matched top-1-label oracle, RLGPA reaches 61% targeted ASR at sparsity 0.01 on CIFAR-10 (vs. 25% SparseEvo, 0% Pointwise), and 75.5%/73.0% on ResNet50/ViT at sparsity 0.005 (vs. 29.0%/15.5%). A fixed-probe test reduces median $L _ { 0 }$ from 165 to 23 after 40 images without any update. State-blind and state-shuffled controls degrade significantly, confirming that the learned policy uses attack state; the main gains come from state-conditioned executor scheduling and dynamic repartition. The source code is publicly available at https://github.com/Acui001/RLGPA.git.

Index Terms—Adversarial examples, Decision-based black-box attacks, Sparse attacks, Online reinforcement learning, Granularity planning.

## I. INTRODUCTION

Deep neural networks achieve strong performance across a wide range of perception tasks, yet their vulnerability to adversarial examples, which are inputs crafted by adding small, often human-imperceptible perturbations that cause incorrect predictions, introduces significant security risks to deep learning-based visual detection tasks [1, 2, 3, 4]. This vulnerability has motivated a large body of work on adversarial attacks [5, 6, 7, 8], which differ in the threat model they assume and in the constraints they place on the perturbation. Among these, sparse adversarial attacks are particularly relevant to security-critical settings, because they modify only a limited number of input dimensions and therefore correspond more closely to the practical constraint that an attacker can manipulate only a small part of the input.

Sparse adversarial attacks seek an adversarial input whose perturbation involves only a small subset of image pixels. The underlying minimum-L<sub>0</sub> adversarial problem is NP-hard [9]. In a decision-based black-box setting, the attacker observes only the top-1 predicted label and cannot access gradients, logits, or confidence scores. As a result, exact support minimization, i.e., minimizing the set of modified pixels, becomes a query-bounded combinatorial search. Starting from a feasible adversarial point, Pointwise attack [10] and SparseEvo attack [9] approach this problem through perturbation erasure: they ask which active coordinates can be restored to the original image without losing adversarial feasibility. Pointwise greedily tests restorations, whereas SparseEvo evolves binary support masks. Their results show that support erasure is central to decision-based sparse attacks, but they also expose a scheduling problem: the best restoration ordering and group size need not remain fixed throughout an attack.

This scheduling problem arises because the search has a pronounced stage dependence. Large groups can remove support quickly when many perturbed coordinates are redundant. However, as the support shrinks, the same removals cross the decision boundary more often, and the attack benefits from finer groups or from empirical knowledge about previous restoration outcomes. A fixed schedule cannot condition on current progress, stalls, acceptance rate, or remaining budget. Conversely, directly assigning reinforcement-learning actions to temporary pixel groups is ill posed: dynamic repartition changes the coordinates represented by an action index, making replay transitions semantically unstable.

To address this problem, we propose RLGPA, a reinforcement learning-guided granularity planning attack, where reinforcement learning (RL) schedules the executor policy and granularity. Instead of selecting a pixel-group index, the agent selects a stable macro-action that composes an executor policy with a relative scale. The executor instantiates that action on the current support, tests a bounded number of restorations, and commits only adversarial candidates. The next macro decision is therefore made after observing an actual support reduction and query cost. Because Primary, Safe, and Memory retain their meanings after repartition and across images, transitions can be reused by an online Dueling Double DQN.

RLGPA uses support-relative granularity. If $L _ { t }$ coordinates remain perturbed, an action selects a group size proportional to $L _ { t }$ rather than to the original image dimension. This produces the same seven action semantics on CIFAR-10 and ImageNet despite their very different resolutions. Dynamic repartition then rebuilds groups around the newly committed adversarial point. The DQN observes the optimization phase, current granularity, recent gains and acceptances, stall counts, remaining budget, recent action identity, action-specific effectiveness, and

the valid-action mask.

Our contributions are as follows:

• We formulate decision-based sparse attack refinement as online reinforcement learning over stable policy– granularity macro-actions. This abstraction avoids attaching persistent Q-values to temporary support groups and makes cross-decision and cross-image replay meaningful.

• We develop a support-relative, dynamically repartitioned executor that integrates three complementary restoration policies: perturbation-ascending, magnitude-aware, and outcome-memory. Through hard-label verification and a commit-or-rollback execution scheme, this executor monotonically reduces the support while strictly preserving adversarial feasibility.

• We provide matched causal evidence for the learning component. Frozen fixed-probe evaluation, state-blind and state-shuffled controls, random and fixed schedulers, lifecycle controls, and component ablations distinguish learned state-conditioned scheduling from executor strength and favorable initialization.

• We evaluate targeted and untargeted attacks on CIFAR-10 and ImageNet with ResNet50 and Vision Transformer (ViT) victims. Under an oracle-matched top-1-label protocol, RLGPA achieves improved Query– Sparsity and ASR–Sparsity performance relative to comparable decision-based sparse attacks.

## II. RELATED WORK

## A. Decision-Based Black-Box Attacks

Hard-label attacks observe only the top-1 decision. Boundary Attack introduced random walks along the decision boundary [11]; HopSkipJump improved this process with geometric steps and binary search [12], while Yan et al. learned query placement with reinforcement learning [13]. Other methods construct continuous boundary-distance objectives: Ilyas et al. estimated a query-limited surrogate [14], Opt-attack minimized distance to the boundary [15], and Sign-OPT and Sign-OPT+ improved zeroth-order sign optimization [16, 17]. AdvViT further exploits transformer structure for efficient black-box attacks [8].

## B. Sparse Adversarial Attacks

Sparse attacks restrict nonzero perturbations to a small $L _ { 0 }$ support, directly modeling settings in which only a limited number of input coordinates can be changed.

Under hard-label feedback, Pointwise greedily restores coordinates while preserving adversarial feasibility [10], whereas SparseEvo evolves support masks and performs strongly on CNN and transformer victims [9]. Both reduce an existing adversarial support using top-1 queries; their main difference is the restoration controller.

Richer score-based oracles support Sparse-RS random search [18], Bayesian sparsity search in BruSLeAttack [19], and joint sparsity–magnitude optimization in SA-MOO [20]. Related score-based attacks include bandit/prior-guided gradient estimation [21, 22], SimBA [23], Square Attack [24], and the one-pixel attack [25]. Because a score vector contains more information than a top-1 label, these methods are relevant adjacent work but not oracle-matched query baselines.

White-box $\mathrm { P G D _ { 0 } }$ [26] uses iterative magnitude pruning, while σ-zero [27] smooths sparse projection. They provide ASR–Sparsity references but not Query–Sparsity baselines because they use gradients rather than victim queries. Their broader lineage includes L-BFGS [3], FGSM [28], JSMA [29], DeepFool [30], C&W [31], and PGD [32]; transfer attacks instead craft examples on a surrogate [33].

## C. Reinforcement Learning for Query Planning

Unlike gradient-based FGSM, PGD, and C&W [28, 31, 32], RL can schedule black-box queries from sequential feedback. Prior work learned query placement [13] or optimized adversarial-patch location and perturbation [34]; other blackbox optimizers include NES [14] and AutoZOOM [35]. RL-GPA instead uses standard DQN, Double-DQN, and Dueling components [36, 37, 38] to schedule stable supportreduction operators online. Its distinction is therefore the stateconditioned action abstraction, not a new RL backbone or a claim of broad policy generalization [39, 40].

## III. METHOD

RLGPA formulates decision-based sparse attack as online, query-bounded support reduction. A compact stateconditioned planner selects a restoration policy and relative granularity, while PSSE executes and verifies every restoration; the following subsections define the objective and the resulting closed loop.

## A. Problem Formulation

Let $f : [ 0 , 1 ] ^ { C \times H \times W } \to \{ 1 , \dots , K \}$ return only the top-1 class. For an image x with $y = f ( \mathbf { x } )$ , the untargeted and targeted feasibility predicates $\mathcal { A } _ { \mathrm { u } }$ and $A _ { \mathrm { t } }$ for a candidate image z are:

$$
\mathcal { A } _ { \mathrm { u } } ( \mathbf { z } ) = \mathbb { I } [ f ( \mathbf { z } ) \neq y ] , \qquad \mathcal { A } _ { \mathrm { t } } ( \mathbf { z } ) = \mathbb { I } [ f ( \mathbf { z } ) = y ^ { \star } ] ,\tag{1}
$$

where I[·] is the indicator function and $y ^ { \star } \ne y$ is the prescribed target label.

RLGPA measures support over scalar tensor coordinates:

$$
\begin{array} { r l } & { ~ \mathcal { T } _ { \mathrm { c h } } = \{ ( c , h , w ) : c \in [ C ] , h \in [ H ] , w \in [ W ] \} , } \\ & { \Omega ( \mathbf { z } , \mathbf { x } ) = \{ i \in \mathcal { T } _ { \mathrm { c h } } : z _ { i } \neq x _ { i } \} , } \\ & { L ( \mathbf { z } , \mathbf { x } ) = | \Omega ( \mathbf { z } , \mathbf { x } ) | , } \\ & { s ( \mathbf { z } , \mathbf { x } ) = L ( \mathbf { z } , \mathbf { x } ) / ( C H W ) , } \end{array}\tag{2}
$$

where $[ n ] \ = \ \{ 1 , \dots , n \} , \ i \ = \ ( c , h , w )$ indexes one scalar entry, $x _ { i }$ and $z _ { i }$ are the corresponding values, Ω is the active support, L its cardinality, and s normalized sparsity. If all C channels share a spatial mask of size $L _ { \mathrm { p i x } }$ , then $L = C L _ { \mathrm { { p i x } } }$ and $s = L _ { \mathrm { p i x } } / ( H W )$ , so channel-coordinate and spatial-pixel representations yield the same normalized sparsity.

Starting from a feasible adversarial image $\mathbf { z } _ { 0 } .$ , RLGPA restores coordinates rather than estimating new amplitudes. Let $\Omega _ { 0 } = \Omega ( \mathbf { z } _ { 0 } , \mathbf { x } )$ and $\mathbf { r } \in \{ 0 , 1 \} ^ { | \Omega _ { 0 } | }$ be its initial support and a binary retention vector over that support. The induced candidate is

![](images/9f24995b872e72d26211318c28ee9f1416b380b3f049be56424e725ef7826a9b.jpg)  
Fig. 1: Overview of RLGPA. The planner selects a policy–scale action from the current attack state, PSSE performs hard label-verified support restoration, and the observed support reduction and query cost are returned for online updating.

$$
z _ { i } ( \mathbf { r } ) = \left\{ \begin{array} { l l } { x _ { i } + r _ { i } ( z _ { 0 , i } - x _ { i } ) , } & { i \in \Omega _ { 0 } , } \\ { x _ { i } , } & { i \notin \Omega _ { 0 } , } \end{array} \right.\tag{3}
$$

where $z _ { i } ( \mathbf { r } )$ is coordinate i of ${ \bf z } ( { \bf r } ) , z _ { 0 , i }$ and $x _ { i }$ are its initial adversarial and clean values, and $r _ { i } = 1$ retains whereas $r _ { i } = 0$ restores the coordinate. Coordinates outside $\Omega _ { 0 }$ already equal x. The query-bounded restoration problem is

$$
\operatorname* { m i n } _ { \mathbf { r } \in \{ 0 , 1 \} ^ { | \Omega _ { 0 } | } } \| \mathbf { r } \| _ { 0 } \quad \mathrm { s . t . } \quad \mathcal { A } ( \mathbf { z } ( \mathbf { r } ) ) = 1 , \qquad Q \leq B ,\tag{4}
$$

where $\| \mathbf { r } \| _ { 0 }$ counts retained coordinates, $Q$ is the cumulative number of top-1 victim queries, and B is the query budget. Each candidate needs one hard-label query, including the initial feasibility check $Q _ { 0 } = 1$ . The $2 ^ { | \Omega _ { 0 } | }$ restoration search is a restricted NP-hard minimum- $L _ { 0 }$ problem [9]; RLGPA is a query-efficient heuristic, not a global optimizer.

## B. RLGPA Framework

Figure 1 summarizes the closed loop. A compact state records support progress, remaining budget, and recent operator outcomes; a masked Dueling Double DQN selects a policy–scale macro-action; PSSE rebuilds and tests restoration groups; and the observed support reduction and query cost update the state and planner. The policy controls group construction and ordering, while the scale controls relative granularity. The planner therefore schedules stable operator semantics rather than pixel identities, and every accepted restoration is independently verified by the hard-label oracle.

## C. Attack-State Representation

The state describes search progress rather than raw pixels. Let ${ \cal L } _ { 0 } = { \cal L } ( { \bf z } _ { 0 } , { \bf x } )$ and $L _ { t } = L ( \mathbf { z } _ { t } , \mathbf { x } )$ be the initial and current support sizes, $Q _ { t }$ the cumulative queries before decision $t ,$ $H _ { t }$ the number of consecutive preceding zero-gain decisions, and $\widehat { N } _ { t }$ the group-count descriptor carried from the preceding decision. At $t \doteq 0 , \widehat { N } _ { t }$ is initialized from the clipped initial partition; subsequently it is updated by (24). The normalized progress variables are

$$
\ell _ { t } = \frac { L _ { t } } { \operatorname* { m a x } ( 1 , L _ { 0 } ) } , \nu _ { t } = \frac { \widehat { N } _ { t } } { \operatorname* { m a x } ( 1 , L _ { t } ) } , \varrho _ { t } = \frac { B - Q _ { t } } { B } ,\tag{5}
$$

where $\ell _ { t } , \nu _ { t }$ , and $\varrho _ { t }$ are the remaining-support, carried-groupdensity, and remaining-budget ratios, and max(1, ·) prevents division by zero. The continuous feature vector is

$$
\begin{array} { r l } & { \xi _ { t } = \big [ \ell _ { t } , \nu _ { t } , \mathrm { m i n } ( 1 , H _ { t } / 5 ) , \mathrm { c l i p } ( \zeta _ { t - 1 } , 0 , 1 ) , } \\ & { \qquad \mathrm { c l i p } ( \alpha _ { t - 1 } , 0 , 1 ) , \mathrm { c l i p } ( \varrho _ { t } , 0 , 1 ) \big ] , } \end{array}\tag{6}
$$

where $\pmb { \xi } _ { t } \in \mathbb { R } ^ { 6 } , \ \zeta _ { t - 1 } = \Delta L _ { t - 1 } / \operatorname* { m a x } ( 1 , L _ { t - 1 } )$ is the preceding relative support reduction, $\Delta L _ { t - 1 } ~ = ~ L _ { t - 1 } - L _ { t } ,$ $\alpha _ { t - 1 }$ is the preceding acceptance ratio, and clip(v, 0, 1) = min(1, max(0, v)); initially $\zeta _ { - 1 } = \alpha _ { - 1 } = 0$ . The categorical vector $\mathbf { c } _ { t } \in \{ 0 , \dot { 1 } \} ^ { 2 1 }$ concatenates one-hot bins for: phase $\phi _ { t }$ which is early for $\ell _ { t } > 0 . 3 5$ , middle for $0 . 0 8 < \ell _ { t } \le 0 . 3 5$ and late otherwise; granularity over [0, 0.25), [0.25, 0.75), [0.75, 0.95), and [0.95, 1] of $\nu _ { t } ;$ three bins each for stall $H _ { t } \in \{ 0 , 1 , \geq 2 \}$ , gain $\zeta _ { t - 1 }$ at 0 and 0.05, acceptance $\alpha _ { t - 1 }$ at 0 and 0.25, and budget $\varrho _ { t }$ at 0.15 and 0.5; plus a binary legacy-safety flag fixed off in the official configuration.

Let $\mathcal { W } _ { t }$ contain the latest W = 8 within-image outcomes and ${ \mathcal { W } } _ { t } ( a ) = \{ \tau \in { \mathcal { W } } _ { t } : a _ { \tau } = a \}$ the subset generated by action a. Its recent gain efficiency and hit rate are

$$
\begin{array} { r l } & { e _ { t } ( a ) = \operatorname* { m i n } \left( 1 , \frac { \sum _ { \tau \in \mathcal { W } _ { t } ( a ) } \Delta L _ { \tau } } { \operatorname* { m a x } \left( 1 , \sum _ { \tau \in \mathcal { W } _ { t } ( a ) } u _ { \tau } \right) } \right) , } \\ & { h _ { t } ( a ) = \frac { \sum _ { \tau \in \mathcal { W } _ { t } ( a ) } \mathbb { I } \left[ \Delta L _ { \tau } > 0 \right] } { \operatorname* { m a x } \left( 1 , | \mathcal { W } _ { t } ( a ) | \right) } , } \end{array}\tag{7}
$$

where $\tau$ indexes a stored decision, $\Delta L _ { \tau }$ and $u _ { \tau }$ are its support reduction and query use, $e _ { t } ( a )$ is clipped accumulated gain per

query, and $h _ { t } ( a )$ its positive-outcome fraction; both are zero for unseen actions. Mean per-decision efficiency is

$$
\bar { e } _ { t } ( a ) = \frac { 1 } { \operatorname* { m a x } ( 1 , | \mathcal { W } _ { t } ( a ) | ) } \sum _ { \tau \in \mathcal { W } _ { t } ( a ) } \frac { \Delta L _ { \tau } } { \operatorname* { m a x } ( 1 , u _ { \tau } ) } ,\tag{8}
$$

where $\bar { e } _ { t } ( a ) \ = \ 0$ when $\mathcal { W } _ { t } ( a )$ is empty. The vector $\begin{array} { r l } { \mathbf { o } _ { t } } & { { } \in \mathbf { \pi } } \end{array}$ $\{ 0 , 1 \} ^ { M + 1 }$ marks arg max<sub>a</sub> $\bar { e } _ { t } ( a )$ among actions with positive accumulated gain, with an extra “none” slot otherwise. Let $\mu _ { t } \in \{ 0 , 1 \} ^ { \bar { M } }$ be the valid-action mask, with $\mu _ { t } ( a ) = 1$ iff action a may be selected. The complete state is

$$
\begin{array} { l } { { { \bf s } _ { t } = \mathrm { c o n c a t } ( { \bf c } _ { t } , { \bf o } _ { t } , \xi _ { t } , \{ e _ { t } ( a ) , h _ { t } ( a ) \} _ { a = 1 } ^ { M } , \mu _ { t } ) , } } \\ { { d _ { s } = 2 1 + ( M + 1 ) + 6 + 2 M + M = 2 8 + 4 M , } } \end{array}\tag{9}
$$

where M is the action count, $\{ e _ { t } ( a ) , h _ { t } ( a ) \} _ { a = 1 } ^ { M }$ concatenates both statistics for every action, and $d _ { s }$ is the state dimension. With $M = 7 , d _ { s } = 5 6$ . Actions with $\kappa > 1$ are masked when $\nu _ { t } ~ \geq ~ 0 . 9 5$ , i.e., at one-coordinate granularity. Histories and coordinate memories reset per image, while declared planner weights and replay may persist. Thus the state schedules support-reduction operators but contains neither support geometry nor semantic pixel identities.

## D. Online RL-Based Macro-Action Planner

RLGPA models operator scheduling as the finite-horizon discounted process $\mathcal { P } ~ = ~ ( \mathcal { S } , \mathcal { M } , P , r , \gamma )$ , where S, M, $P ( \mathbf { s } _ { t + 1 } \mid \mathbf { s } _ { t } , a _ { t } ) ,$ r, and $\gamma \in \ [ 0 , 1 )$ are the scheduler-state space, macro-action set, executor-induced transition, reward, and discount factor. Because $\mathbf { s } _ { t }$ omits image geometry, this is an observable scheduling approximation rather than an exact Markov claim. Each action is

$$
a _ { t } = ( p _ { t } , \kappa _ { t } ) \in \mathcal { M } , \quad p _ { t } \in \{ \mathrm { P r i m a r y } , \mathrm { S a f e } , \mathrm { M e m o r y } \} ,\tag{10}
$$

where $a _ { t }$ is the decision at step t, $p _ { t }$ selects an executor, and $\kappa _ { t } ~ > ~ 0$ its relative scale. The seven coordinate-independent policy–scale pairs in Table I retain their meanings after repartition and across images.

The masked Dueling Double DQN decomposes action value as

$$
Q _ { \theta } ( \mathbf { s } , a ) = V _ { \theta } ( \mathbf { s } ) + A _ { \theta } ( \mathbf { s } , a ) - \frac { 1 } { M } \sum _ { a ^ { \prime } = 1 } ^ { M } A _ { \theta } ( \mathbf { s } , a ^ { \prime } ) ,\tag{11}
$$

where $Q _ { \theta } ( \mathbf { s } , a )$ is the estimated return, θ the online parameters, $V _ { \theta } ( \mathbf { s } )$ and $A _ { \theta } ( \mathbf { s } , a )$ the value and advantage streams, and $a ^ { \prime }$ indexes all M actions. An ϵ-greedy policy samples only $\mathcal { M } _ { t } =$ $\{ a \in \mathcal { M } : \mu _ { t } ( a ) = 1 \}$ , with invalid values masked $\mathrm { { t o \_ - \infty } }$ For terminal flag $d _ { t } ,$ , the Double-DQN update is

$$
a _ { t } ^ { \star } = \arg \operatorname* { m a x } _ { a \in \mathcal { M } _ { t + 1 } } Q _ { \theta } ( \mathbf { s } _ { t + 1 } , a ) ,\tag{12}
$$

$$
y _ { t } = r _ { t } + \gamma ( 1 - d _ { t } ) Q _ { \bar { \theta } } ( \mathbf { s } _ { t + 1 } , a _ { t } ^ { \star } ) ,\tag{13}
$$

$$
\begin{array} { r } { \mathcal { L } ( \theta ) = \mathbb { E } _ { \mathcal { D } } [ \mathrm { H u b e r } ( Q _ { \theta } ( \mathbf { s } _ { t } , a _ { t } ) - y _ { t } ) ] , } \end{array}\tag{14}
$$

where $\mathcal { M } _ { t + 1 }$ is the next valid set, $a _ { t } ^ { \star }$ its online-network maximizer, $r _ { t }$ the observed reward, $y _ { t }$ the target, <sup>¯</sup>θ the target-network parameters, D replay memory, and Huber the smooth- ${ \cal - L } _ { 1 }$ loss. The stored next mask defines $\mathcal { M } _ { t + 1 } ;$

TABLE I: Canonical RLGPA macro-actions. Policy–scale semantics remain fixed when the current support is repartitioned.
<table><tr><td>Action</td><td>Executor</td><td>Ordering</td><td>κ</td></tr><tr><td>Coarse Primary</td><td>Primary</td><td>perturbation ascending</td><td>0.5</td></tr><tr><td>Base Primary</td><td>Primary</td><td>perturbation ascending</td><td>1.0</td></tr><tr><td>Coarse Safe</td><td>Safe</td><td>mean magnitude descending</td><td>0.5</td></tr><tr><td>Base Safe</td><td>Safe</td><td>mean magnitude descending</td><td>1.0</td></tr><tr><td>Fine Safe</td><td>Safe</td><td>mean magnitude descending</td><td>2.0</td></tr><tr><td>Base Memory</td><td>Memory</td><td>restoration memory descending</td><td>1.0</td></tr><tr><td>Fine Memory</td><td>Memory</td><td>restoration memory descending</td><td>2.0</td></tr></table>

$d _ { t } \in \{ 0 , 1 \}$ removes bootstrapping at terminal transitions and image boundaries.

Both networks use a 128–128 ReLU trunk and separate 64- unit heads. Adam uses learning rate $1 0 ^ { - 3 } , \gamma = 0 . 8 ,$ batch size 64, replay capacity 20,000, warm-up 128, gradient clipping 5, and target synchronization every 250 updates. Exploration decays $0 . 1 5  0 . 0 2$ by factor 0.995 per decision and resets per image. Continual-online evaluation preserves weights and replay; frozen evaluation disables exploration and all learningstate mutation, separating executor capability from policy learning.

## E. Policy-Scale Support Executor

PSSE instantiates $a _ { t } = \left( p _ { t } , \kappa _ { t } \right)$ on the verified incumbent $\mathbf { z } _ { t } .$ , whose active support and size are $\Omega _ { t } = \Omega ( \mathbf { z } _ { t } , \mathbf { x } )$ and $L _ { t } =$ $L ( \mathbf { z } _ { t } , \mathbf { x } ) = | \Omega _ { t } |$ . Given base erase ratio $\rho > 0 .$ , it computes

$$
\begin{array} { r l } & { \eta _ { t } = \rho / \kappa _ { t } , } \\ & { g _ { t } = \operatorname* { m i n } ( L _ { t } , \operatorname* { m a x } ( g _ { \mathrm { m i n } } , \lceil L _ { t } \eta _ { t } \rceil ) ) , } \\ & { J _ { t } = \lceil L _ { t } / g _ { t } \rceil , } \end{array}\tag{15}
$$

where $\eta _ { t } , \ g _ { t } ,$ , and $J _ { t }$ are the requested support fraction per group, integer target group size, and group count; $g _ { \mathrm { m i n } }$ is the size lower bound and ⌈·⌉ the ceiling operator. We use $\rho =$ 0.002, $g _ { \mathrm { m i n } } = 1$ , and $\kappa _ { t } \in \{ 0 . 5 , 1 , 2 \}$ for coarse, base, and fine scales. The realized ratio $\widehat { \eta } _ { t } = g _ { t } / L _ { t }$ may differ from $\eta _ { t } = \rho / \kappa _ { t }$ because of integer clipping.

Let $g _ { t } ^ { \mathrm { l o } } = \lfloor L _ { t } / J _ { t } \rfloor$ be the smaller balanced size and $R _ { t } ^ { \mathrm { b a l } } =$ $L _ { t } { - } J _ { t } g _ { t } ^ { \mathrm { l o } }$ the number of groups receiving one extra coordinate. The partition $\{ G _ { t , j } \} _ { j = 1 } ^ { J _ { t } }$ satisfies

$$
\begin{array} { r } { \left| G _ { t , j } \right| = \left\{ \begin{array} { l l } { g _ { t } ^ { \mathrm { l o } } + 1 , } & { 1 \leq j \leq R _ { t } ^ { \mathrm { b a l } } , } \\ { g _ { t } ^ { \mathrm { l o } } , } & { R _ { t } ^ { \mathrm { b a l } } < j \leq J _ { t } , } \end{array} \right. } \\ { G _ { t , j } \cap G _ { t , k } = \emptyset \ ( j \neq k ) , \quad \bigcup _ { j = 1 } ^ { J _ { t } } G _ { t , j } = \Omega _ { t } , } \end{array}\tag{16}
$$

where ⌊·⌋ is the floor operator, $G _ { t , j } \subseteq \Omega _ { t }$ is group j, and $j , k \in$ $\{ 1 , \ldots , J _ { t } \}$ . The groups are exhaustive, disjoint, and directly generated by PSSE. Rebuilding them from committed support before every macro-action preserves policy–scale semantics without assigning Q-values to temporary group indices.

Policy $p _ { t }$ controls group construction and order. For $i \in$ $\Omega _ { t } .$ let $\omega _ { i } = | z _ { t , i } - x _ { i } |$ be perturbation magnitude. Primary shuffles ties, sorts coordinates by increasing ω , and chunks the sequence. Safe randomly groups coordinates and ranks groups by decreasing

$$
S _ { \mathrm { m a g } } ( G ) = { \frac { 1 } { | G | } } \sum _ { i \in G } \omega _ { i } ,\tag{17}
$$

where $G \subseteq \Omega _ { t } , | G |$ is its size, and $S _ { \mathrm { m a g } } ( G )$ its mean magnitude. Memory maintains image-local accepted and rejected group attributions $( n _ { i } ^ { + } , n _ { i } ^ { - } )$ and ranks groups by

$$
m _ { i } = \frac { n _ { i } ^ { + } + 1 } { n _ { i } ^ { + } + n _ { i } ^ { - } + 2 } , \qquad S _ { \mathrm { m e m } } ( G ) = \frac { 1 } { | G | } \sum _ { i \in G } m _ { i } ,\tag{18}
$$

where $n _ { i } ^ { + }$ and $n _ { i } ^ { - }$ count accepted and rejected tests containing $i , \ m _ { i } \ \in \ ( 0 , 1 )$ is the Beta(1, 1)-smoothed attribution, and $S _ { \mathrm { m e m } } ( G )$ its group mean. Testing G increments $n _ { i } ^ { + }$ for every $i \in G$ on acceptance and $n _ { i } ^ { - }$ otherwise. These per-image executor statistics are ordering scores, not identified coordinatelevel causal probabilities or cross-image DQN state.

Set ${ \bf z } _ { t } ^ { ( 0 ) } = { \bf z } _ { t }$ . For group $G _ { t , j }$ , the restoration operator and candidate are

$$
[ \mathcal { R } _ { G } ( \mathbf { z } ; \mathbf { x } ) ] _ { i } = \left\{ \begin{array} { l l } { x _ { i } , } & { i \in G , } \\ { z _ { i } , } & { i \notin G . } \end{array} \right. \widetilde { \mathbf { z } } _ { t } ^ { ( j ) } = \mathcal { R } _ { G _ { t , j } } ( \mathbf { z } _ { t } ^ { ( j - 1 ) } ; \mathbf { x } ) ,\tag{19}
$$

where $\mathcal { R } _ { G } ( \mathbf { z } ; \mathbf { x } )$ copies $x _ { i }$ into scalar coordinates $i \in G$ and retains $z _ { i }$ elsewhere, $\mathbf { z } _ { t } ^ { ( j - 1 ) }$ is the latest verified incumbent, and $\widetilde { \mathbf { z } } _ { t } ^ { ( j ) }$ is the unverified candidate. One victim query returns $\chi _ { t , j } = \mathcal { A } ( \widetilde { \mathbf { z } } _ { t } ^ { ( j ) } ) \in \{ 0 , 1 \}$ , after which PSSE applies

$$
\begin{array} { r } { \mathbf { z } _ { t } ^ { ( j ) } = \left\{ \widetilde { \mathbf { z } } _ { t } ^ { ( j ) } , \begin{array} { l l } { \chi _ { t , j } = 1 , } \\ { \mathbf { z } _ { t } ^ { ( j - 1 ) } , } \end{array} \right. } \\ { L ( \mathbf { z } _ { t } ^ { ( j ) } , \mathbf { x } ) = L ( \mathbf { z } _ { t } ^ { ( j - 1 ) } , \mathbf { x } ) - \chi _ { t , j } | G _ { t , j } | , } \end{array}\tag{20}
$$

where $\mathbf { z } _ { t } ^ { ( j ) }$ is the verified incumbent after test $j ; \ \chi _ { t , j } \ = \ 1$ commits the candidate and $\chi _ { t , j } ~ = ~ 0$ rolls it back. Since tested groups are active and disjoint, every incumbent remains feasible and support never increases. A macro-action executes

$$
u _ { t } = \operatorname* { m i n } ( q _ { c } , J _ { t } , B - Q _ { t } ) , \qquad q _ { c } = 3 2 ,\tag{21}
$$

tests, where $u _ { t }$ is both the number of tested groups and consumed victim queries, $q _ { c } = 3 2$ is the per-decision cap, $J _ { t }$ the available group count, and $B - Q _ { t }$ the remaining budget. It returns

$$
\begin{array} { l } { { \displaystyle { \bf z } _ { t + 1 } = { \bf z } _ { t } ^ { ( u _ { t } ) } , } } \\ { { \displaystyle \Delta L _ { t } = L _ { t } - L _ { t + 1 } = \sum _ { j = 1 } ^ { u _ { t } } \chi _ { t , j } \vert G _ { t , j } \vert , } } \\ { { \displaystyle \alpha _ { t } = \frac { 1 } { \operatorname* { m a x } ( 1 , u _ { t } ) } \sum _ { j = 1 } ^ { u _ { t } } \chi _ { t , j } , } } \end{array}\tag{22}
$$

where $\mathbf { z } _ { t + 1 } ~ = ~ \mathbf { z } _ { t } ^ { ( u _ { t } ) }$ is the returned incumbent, $L _ { t + 1 } ~ =$ $L ( \mathbf { z } _ { t + 1 } , \mathbf { x } ) , \Delta L _ { t }$ the accepted support reduction, and $\alpha _ { t }$ the accepted-test ratio. Thus PSSE performs feasibility-preserving support reduction; RL only schedules its executor and scale.

Algorithm 1 RLGPA continual-online support search   
Input: Hard-label oracle $f ,$ original x, candidate start ${ \bf z } _ { 0 } ,$ budget $B ,$   
shared planner Π   
1: verify $\hat { \boldsymbol { \mathcal { A } } } ( \mathbf { z } _ { 0 } ) = 1 ; \mathrm { s e t } \ \mathbf { z } \gets \mathbf { z } _ { 0 } , Q \gets 1 ;$ reset image-local state   
2: while $Q < B$ and $L ( \mathbf { z } , \mathbf { x } ) > 0$ do   
3: encode $( \mathbf { s } _ { t } , \mu _ { t } ) ;$ choose $a _ { t } = \left( p _ { t } , \kappa _ { t } \right)$ by masked ϵ-greedy Π   
4: compute $( g _ { t } , J _ { t } )$ ; partition and order $\dot { \Omega _ { t } }$ using $p _ { t }$   
5: for $\bar { j } = 1 , \dots , \operatorname* { m i n } ( q _ { c } , J _ { t } , B - Q )$ do   
6: query $\tilde { \mathbf { z } } \ \gets \ \mathcal { R } _ { G _ { t , j } } ( \mathbf { z } ; \mathbf { x } ) ;$ ; commit iff feasible; update   
memory and $Q$   
7: end for   
8: compute feedback; encode next state; store τ<sub>t</sub>; update DDQN   
after warm-up   
9: end while   
10: return z and the updated shared planner

## F. Query-Aware Feedback

PSSE returns $\left( \Delta L _ { t } , u _ { t } , \alpha _ { t } \right)$ , acceptance bits $\{ \chi _ { t , j } \} _ { j = 1 } ^ { u _ { t } }$ , and updated coordinate memory. Let $\bar { g } _ { t } = L _ { t } / J _ { t }$ be mean planned group size, $H _ { t }$ the pre-action stall length, and $\phi _ { t }$ the phase determined by $\ell _ { t } = L _ { t } / L _ { 0 }$ . The reward is

$$
\begin{array} { r l r } {  { r _ { t } = w _ { \phi _ { t } } \frac { \Delta L _ { t } } { \operatorname* { m a x } ( 1 , u _ { t } \bar { g } _ { t } ) } + \beta _ { \mathrm { t a i l } } \mathbb { I } [ \Delta L _ { t } > 0 \wedge \phi _ { t } = \mathrm { l a t e } ] } } \\ & { } & { + \beta _ { \mathrm { s t a l l } } \mathbb { I } [ \Delta L _ { t } > 0 \wedge H _ { t } \geq 2 ] } \\ & { } & { + c _ { 0 } \mathbb { I } [ \Delta L _ { t } = 0 ] - \lambda _ { q } \frac { u _ { t } } { B } , ~ } \end{array}\tag{23}
$$

where $r _ { t }$ is scalar feedback, $w _ { \phi _ { t } }$ the phase weight, $\beta _ { \mathrm { t a i l } }$ and $\beta _ { \mathrm { s t a l l } }$ bonuses for late and post-stall progress, $c _ { 0 } ~ < ~ 0$ the zero-gain penalty, and $\lambda _ { q } ~ \geq ~ 0$ the explicit query-penalty coefficient. We use $( w _ { \mathrm { e a r l y } } , w _ { \mathrm { m i d d l e } } , w _ { \mathrm { l a t e } } ) = ( 0 . 2 5 , 0 . 5 , 1 )$ $\beta _ { \mathrm { t a i l } } = \beta _ { \mathrm { s t a l l } } = 0 . 2 5 , c _ { 0 } = - 0 . 0 2$ , and $\lambda _ { q } = 0 .$ . The leading term already normalizes gain by approximate attempted support $u _ { t } \bar { g } _ { t } ;$ the reward is a scale-aware surrogate, not a probability.

After updating coordinate counts and appending $( a _ { t } , \Delta L _ { t } , u _ { t } )$ to ${ \ w } _ { t }$ , feedback updates

$$
\begin{array} { r l } & { Q _ { t + 1 } = Q _ { t } + u _ { t } , } \\ & { H _ { t + 1 } = \left\{ \begin{array} { l l } { 0 , } & { \Delta L _ { t } > 0 , } \\ { H _ { t } + 1 , } & { \Delta L _ { t } = 0 , } \end{array} \right. } \\ & { \widehat { N } _ { t + 1 } = \operatorname* { m i n } \bigl ( J _ { t } , \operatorname* { m a x } ( 1 , L _ { t + 1 } ) \bigr ) , } \end{array}\tag{24}
$$

where $Q _ { t + 1 }$ is the updated query count, $H _ { t + 1 }$ resets after positive gain and otherwise increments, and $\dot { N } _ { t + 1 }$ carries $J _ { t }$ clipped to the nonzero range supported by $L _ { t + 1 }$ . The encoder then recomputes $\mathbf { s } _ { t + 1 }$ and valid-action mask $\pmb { \mu } _ { t + 1 }$ , producing

$$
\tau _ { t } = ( \mathbf { s } _ { t } , a _ { t } , r _ { t } , \mathbf { s } _ { t + 1 } , d _ { t } , \mu _ { t + 1 } ) ,\tag{25}
$$

where $\tau _ { t }$ contains current state $\mathbf { s } _ { t } ,$ action $a _ { t } .$ , reward $r _ { t } ,$ next state $\mathbf { s } _ { t + 1 }$ , terminal flag $d _ { t }$ , and next mask $\pmb { \mu } _ { t + 1 }$ . We set $d _ { t } = 1$ at $Q _ { t + 1 } \geq B , L _ { t + 1 } = 0$ , or the image boundary, and zero otherwise. After warm-up, stored transitions update DDQN. The shaped reward is only an online signal: its necessity is not claimed, and useful state-conditioned learning is assessed against frozen, untrained, random, fixed, state-blind, and stateshuffled controls. Algorithm 1 summarizes the loop.

TABLE II: Median sparsity and ASR at different query budgets. A comprehensive comparison among different attacks (σ-zero, $\mathrm { P G D _ { 0 } } .$ , Pointwise, SparseEvo, and RLGPA) on small and large scale balance datasets. The two white-box rows are reported at their converged distortion for reference only: their optimization cost is not a sequence of victim queries, so they are excluded from every query-efficiency comparison.
<table><tr><td rowspan="2">Setting</td><td rowspan="2">Query</td><td rowspan="2">Methods</td><td colspan="2">CIFAR10 (ResNet18)</td><td rowspan="2">Query</td><td colspan="2">ImageNet (ResNet50)</td></tr><tr><td>Median</td><td>ASR@0.05</td><td>Median</td><td>ASR@0.01</td></tr><tr><td rowspan="6">Untargeted</td><td rowspan="2"></td><td>σ-zero</td><td>0.0013</td><td>100%</td><td rowspan="2"></td><td>0.0001</td><td>100%</td></tr><tr><td> $\mathrm { P G D _ { 0 } }$ </td><td>0.0068</td><td>100%</td><td>0.0005</td><td>100%</td></tr><tr><td rowspan="2">200</td><td>Pointwise</td><td>0.0059</td><td>91.0%</td><td rowspan="2">2000</td><td>0.0024</td><td>68.0%</td></tr><tr><td>SparseEvo</td><td>0.0215</td><td>85.0%</td><td>0.0026</td><td>87.0%</td></tr><tr><td rowspan="2"></td><td>RLGPA</td><td>0.0033</td><td>93.0%</td><td rowspan="2"></td><td>0.0004</td><td>96.0%</td></tr><tr><td>Pointwise</td><td>0.0059</td><td>96.0%</td><td>0.0016</td><td>80.0%</td></tr><tr><td rowspan="2"></td><td>500</td><td>SparseEvo</td><td>0.0215</td><td>86.0%</td><td rowspan="2">5000</td><td>0.0023</td><td>87.5%</td></tr><tr><td></td><td>RLGPA</td><td>0.0026</td><td>100.0%</td><td>0.0004</td><td>99.5%</td></tr><tr><td rowspan="6">Targeted</td><td rowspan="2"></td><td>σ-zero</td><td>0.0036</td><td>100%</td><td rowspan="2"></td><td>0.0009</td><td>100%</td></tr><tr><td>PGD0</td><td>0.0156</td><td>100%</td><td>0.0037</td><td>97.0%</td></tr><tr><td rowspan="3">1000</td><td>Pointwise</td><td>0.4224</td><td>0.0%</td><td rowspan="3">10000</td><td>0.7464</td><td>0.0%</td></tr><tr><td>SparseEvo</td><td>0.0210</td><td>90.0%</td><td>0.0275</td><td>20.0%</td></tr><tr><td>RLGPA</td><td>0.0869</td><td>24.0%</td><td>0.0116</td><td>45.5%</td></tr><tr><td rowspan="2">2000</td><td>Pointwise</td><td>0.2236</td><td>11.0%</td><td rowspan="2">20000</td><td>0.6564</td><td>0.0%</td></tr><tr><td></td><td>0.0181</td><td>95.0%</td><td></td><td>71.5%</td></tr><tr><td rowspan="2"></td><td rowspan="2"></td><td>SparseEvo RLGPA</td><td></td><td></td><td rowspan="2"></td><td>0.0070</td><td></td></tr><tr><td></td><td>0.0085</td><td>99.0%</td><td>0.0034</td><td>88.5%</td></tr></table>

## IV. EXPERIMENTS AND EVALUATION

## A. Experimental Setup

a) Datasets and target models: We evaluate CIFAR-10 with ResNet18, and ImageNet-1k with ResNet50 and ViT. The cross-method evaluation uses 100 CIFAR-10 images and 200 ImageNet images per attack mode. The query budgets of Targeted and untargeted attacks are 2,000 and 500 on CIFAR-10, and 20,000 and 5,000 on ImageNet, respectively.

b) Baselines: Pointwise and SparseEvo are the numerical external baselines because they share RLGPA’s decision-only oracle and sparse-support objective. Score-based sparse attacks are excluded from numerical query comparison: a score query returns substantially richer information than a top-1-label query, so treating both as an equal unit would not constitute an oracle-matched efficiency comparison. $\mathrm { P G D _ { 0 } }$ and σ-zero are included only as white-box ASR–Sparsity references. The untargeted σ-zero curves follow its published threat model; for completeness, the targeted panels use our target-classmargin extension of the same gradient-based optimizer and are not presented as an official targeted σ-zero result. Neither white-box method appears in Query–Sparsity curves or queryefficiency rankings. For a fair comparison, all numerically compared decision-based attacks use the same model preprocessing and success denominator within a setting.

c) Metrics: On the one hand, we evaluate query efficiency using Query–Sparsity, which reports the median normalized support over hard-label queries together with the first and third quartiles; only oracle-matched decision-based methods are included in this view. On the other hand, we measure attack strength using ASR–Sparsity, defined as the percentage of evaluation images that remain adversarial after reaching a given sparsity threshold. White-box references appear only in the ASR–Sparsity view and are visually separated from black-box attacks. In addition, we report the median, mean, and 90th-percentile $L _ { 0 }$ of the final perturbations. For paired ablations, we use identical manifests and starting points across configurations, and analyze the results with win/tie/loss counts and Wilcoxon signed-rank tests.

TABLE III: ASR of ResNet50 and ViT under attacks at different query budgets and sparsity thresholds. A comprehensive comparison among different attacks (σ-zero, $\mathrm { P G D _ { 0 } , }$ SparseEvo, and RLGPA) on small and large scale balanced evaluation sets from ImageNet.
<table><tr><td>Setting</td><td>Query</td><td>Methods</td><td colspan="2">ResNet50</td><td colspan="2">ViT</td></tr><tr><td colspan="3">Sparsity</td><td colspan="2">0.001 0.005</td><td colspan="2">0.001 0.005</td></tr><tr><td rowspan="4">Untargeted</td><td></td><td>σ-zero PGD0</td><td>100% 74.0%</td><td>100% 100.0%</td><td>100% 100% 46.5% 91.5%</td></tr><tr><td>2000</td><td>SparseEvo RLGPA</td><td>21.0% 76.5%</td><td>68.5% 27.0% 92.0% 32.5%</td><td>74.0% 51.5%</td></tr><tr><td>5000</td><td>SparseEvo RLGPA</td><td>21.5% 89.0%</td><td>72.0% 30.0% 99.5% 63.0%</td><td>93.0% 98.0%</td></tr><tr><td colspan="2">Sparsity</td><td>0.001</td><td>0.005</td><td>0.001 0.005</td></tr><tr><td rowspan="5">Targeted</td><td></td><td>σ-zero PGD0</td><td>60.5% 1.0%</td><td>100% 73.5%</td><td>57.5% 98.0% 1.5% 41.5%</td></tr><tr><td>10000</td><td>SparseEvo RLGPA</td><td>0.0% 3.5%</td><td>11.5% 27.0%</td><td>0.5% 14.0% 3.5% 49.0%</td></tr><tr><td>20000</td><td>SparseEvo RLGPA</td><td>0.5%</td><td>29.0%</td><td>0.5% 15.5%</td></tr><tr><td></td><td></td><td>3.5%</td><td>75.5%</td><td>3.5% 73.0%</td></tr></table>

d) RLGPA configuration: The seven-action relative planner uses $\rho = 0 . 0 0 2 \mathrm { . }$ , query chunks of 32, discount $\gamma = 0 . 8 ,$ initial/minimum exploration 0.15/0.02, and exploration decay 0.995. The Dueling network has two 128-unit hidden layers. Replay capacity is 20,000, batch size is 64, warmup is 128 transitions, learning rate is $1 0 ^ { - 3 }$ , the target network is synchronized every 250 updates, and one optimizer step is taken per eligible decision.

## B. Main Comparison

Table II summarizes the final-budget results under the matched top-1-label oracle. We use a strict threshold of 0.01 for CIFAR-10 targeted attacks, 0.005 for ImageNet targeted attacks, and 0.001 for untargeted attacks. Full threshold curves are given in Figure 2; the conclusion does not depend on a single operating point. $\mathrm { P G D _ { 0 } }$ and σ-zero are reference curves for sparse robustness only and are excluded from the queryefficiency claim.

On CIFAR-10, RLGPA achieves lower median sparsity and higher ASR than Pointwise and SparseEvo at both untargeted budgets. In the targeted setting, Pointwise largely fails,

CIFAR-10   
ResNet18

Untargeted ASR–Sparsity

![](images/0eb7d92566e65c5ceb4614ef2386c944ff838dd48ac2fc5a1df561456db9d2b3.jpg)  
(a)

Targeted ASR–Sparsity  
![](images/df1febfd171225acd02f6288428b5211f181554ff8a144fa89c5e00a94579c69.jpg)  
(b)  
Targeted Query–Sparsity

![](images/8df9f327ece6c57d2e5cae25cad23c948168691edc70659a520bfc776446e41f.jpg)

![](images/d0286bbb4b863d9c408ba6504ffe1b4a194671f1a901bc16530d696aa0adeacf.jpg)  
ImageNet ResNet50

![](images/7644866df2aa3f95da212ddaf9bf75791021e074472e8c8eccfe01b2542504f2.jpg)  
(e)

![](images/683adee75e14fb89227d2e2d5f33ccd0b1a9c5f1e20c048b140d453bc3cbaec2.jpg)  
(f)

(c)  
(d)  
![](images/0216366125daf83f82d5a1c5a5dff0aa14fdc01e6897d45a1ee3f3a6f3c38771.jpg)

![](images/fd1181016d43f120234f83c8cdd1bece1334cecb685e4f23a0bc1f090e8a15ca.jpg)  
ImageNet ViT

![](images/17e7e7b11726dd9e6f785167fa5c5c01d4804a699476d9e65410f2c3cbbe0006.jpg)  
(i)

![](images/83ed522d2b19166e2da27598b1d357b8b80ec9294da7c6ea03f040ae2cb3e109.jpg)  
(j)

(g)  
![](images/2d0c8e19e98af6f71476d4b60d7e5d21fdc2797afc41a29281313749a4a86feb.jpg)  
(k)

(h)  
![](images/710f7809f346f31bb428315decd1a34b6cb0401e1fe3dd3398ac8d2987de1c1f.jpg)  
(l)  
Fig. 2: Comparison of RLGPA on CIFAR-10 (ResNet18) and ImageNet (ResNet50 and ViT). Rows correspond to dataset–model pairs, while columns present untargeted/targeted ASR–Sparsity and Query–Sparsity results. PGD<sub>0</sub> and σ-zero are white-box references included only in the ASR–Sparsity panels.

SparseEvo is the strongest baseline, and RLGPA is weaker at the half budget but clearly best at the full budget (median sparsity 0.0085, ASR 99.0%). This pattern is consistent with the online-learning characterization in Section IV-D: the shared planner improves over the stream, so its advantage concentrates at larger budgets.

On ImageNet, the same ordering holds. Untargeted median sparsity remains at 0.0004 with 96.0% and 99.5% ASR, and targeted median sparsity improves from 0.0116 to 0.0034 with ASR rising from 45.5% to 88.5% as the budget grows. Notably, RLGPA attains sparsity levels comparable to the white-box methods σ-zero and $\mathrm { P G D _ { 0 } }$ , despite using only hardlabel feedback.

Table III further reports ImageNet results separately for ResNet50 and ViT at sparsity 0.001 and 0.005. Under both untargeted and targeted settings, RLGPA leads at strict sparsity and full budget on both architectures. SparseEvo retains an advantage only in the lowest-budget, loose-threshold ViT cell. The transformer result confirms that the policy–executor decomposition is not tied to convolutional victims.

Figure 2 confirms that RLGPA leads in the low-sparsity region across all three settings and throughout the budget range. White-box references appear only in the ASR–Sparsity panels and are visually separated from the black-box curves. Overall, RLGPA dominates the decision-based field at strict sparsity and full budget on both CNN and transformer victims, conceding only the lowest-budget, loose-threshold ViT cells to SparseEvo.

![](images/99c1ad9616ba6ac84ae5d0570c82bc3471a90b45037201784c31c60d08659bbc.jpg)  
Fig. 3: Visual comparison of targeted attacks between RLGPA and SparseEvo on ImageNet. Query budgets: 100, 500, 1,000, 5,000, 10,000, and 20,000. For each method, the upper row shows the adversarial image and the lower row its perturbation map with normalized sparsity. Target-class reference (left) and source image (right).

## C. Qualitative Evolution Across Queries

Section IV-B quantitatively demonstrate that the RLGPA attack achieves strong performance on both CNN and Transformer victim models. By contrast, Figures 3 and 4 qualitatively illustrate the attack effectiveness of different methods by showing how individual targeted adversarial examples evolve with the query budget.

![](images/af9dcbfe66cc692c8386fc0e5cfe44bad5d4a2f2d47d6d969361a42d6c1ef325.jpg)  
Fig. 4: Visual evolution of targeted adversarial examples generated by RLGPA on ViT and ResNet50 on ImageNet. Two source–target pairs are shown. Query budgets: 1,000, 5,000, 10,000, 15,000, and 20,000. Target-class reference (left), source image (right); enlarged crops highlight residual perturbations at the final budget.

Figure 3 shows that both attacks progressively restore the visual structure of the source image while retaining targeted feasibility. On this example, RLGPA contracts the diffuse perturbation more rapidly at the early budgets and reaches a lower final sparsity at 20,000 queries. The perturbation maps make the support contraction visible rather than relying only on the scalar sparsity values.

Figure 4 further illustrates the same restoration process on both a convolutional and a transformer victim. At lower budgets, the perturbation remains spatially widespread; as the query budget increases, the adversarial image progressively resembles the source image, and the residual changes become localized. The similar trends observed on ResNet50 and ViT provide a qualitative complement to the architecture-specific quantitative results in Table III.

## D. Does the Planner Learn? Fixed-Probe Evaluation

The main curves measure continual-online attack performance but do not by themselves prove learning, because image difficulty can vary with stream position. We therefore construct four non-overlapping CIFAR-10 targeted splits: 100 attribution images, 120 adaptation images, 20 validation images, and 20 fixed probe images. Checkpoints are frozen and evaluated with ϵ = 0; parameter hashes, optimizer steps, and replay sizes are checked before and after every validation/probe run. The probe split is never used for checkpoint selection, so any improvement on it reflects transfer rather than fitting to the evaluation images.

Table IV reports the frozen performance on the fixed, disjoint 20-image probe set across different numbers of adaptation images. No probe-time parameter or replay update is allowed. At 0 adaptation images, the median $L _ { 0 }$ is 165.0 and the ASR@0.01 is 0.20. As the number of adaptation images increases, the median $L _ { 0 }$ decreases steadily—reaching 23.0 at 40 adaptation images and stabilizing around 21.5 at 120 adaptation images—while ASR@0.01 rises from 0.20 to 0.65. These results show that the planner improves on the probe set as it accumulates attack experience, without any test-time updates.

TABLE IV: Frozen performance on a fixed, disjoint 20- image probe set. No probe-time parameter or replay update is allowed. Median, Mean, and P90 are for $L _ { 0 }$
<table><tr><td>Adaptation images</td><td>ASR@.01</td><td>Median</td><td>Mean</td><td>P90</td></tr><tr><td>0</td><td>.20</td><td>165.0</td><td>136.20</td><td>226.2</td></tr><tr><td>20</td><td>.45</td><td>41.5</td><td>40.65</td><td>70.6</td></tr><tr><td>40</td><td>.60</td><td>23.0</td><td>28.20</td><td>57.4</td></tr><tr><td>60</td><td>.50</td><td>25.5</td><td>27.75</td><td>55.7</td></tr><tr><td>80</td><td>.55</td><td>21.5</td><td>27.55</td><td>52.0</td></tr><tr><td>100</td><td>.60</td><td>21.0</td><td>28.35</td><td>54.1</td></tr><tr><td>120</td><td>.65</td><td>21.5</td><td>25.15</td><td>50.2</td></tr></table>

TABLE V: Matched planner attribution on targeted CIFAR-10. The final column is the paired Wilcoxon test against Full DQN. Median, Mean, and P90 are for $L _ { 0 }$
<table><tr><td>Planner/control</td><td>ASR@.01 Median</td><td></td><td>Mean</td><td>P90</td><td>p</td></tr><tr><td>Full shared-online DQN</td><td>.59</td><td>24.5</td><td>35.56</td><td>67.0</td><td></td></tr><tr><td>Base Memory only</td><td>.57</td><td>25.5</td><td>33.65</td><td>63.1</td><td>.8535</td></tr><tr><td>Deterministic schedule</td><td>.50</td><td>30.5</td><td>33.45</td><td>57.1</td><td>.1117</td></tr><tr><td>Uniform random</td><td>.46</td><td>35.0</td><td>38.04</td><td>68.5</td><td>.0028</td></tr><tr><td>Train-disabled DQN</td><td>.21</td><td>122.5</td><td>112.32</td><td></td><td> $1 9 9 . 1 < 1 0 ^ { - 1 5 }$ </td></tr><tr><td>Per-image-reset DQN</td><td>.21</td><td></td><td>122.5112.32</td><td></td><td> $1 9 9 . 1 < 1 0 ^ { - 1 5 }$ </td></tr></table>

To isolate this transfer effect, we select checkpoint 40 using the validation split only and compare it against checkpoint 0 on the 20 fixed probe images. All 20 probe images improve, the median paired reduction is 131 coordinates, and the Wilcoxon test gives $p = 1 . 9 1 \times 1 0 ^ { - 6 }$ . This improvement is accompanied by a stable evaluation state: across all 14 frozen validation/probe runs, parameter SHA256, optimizer-step count, and replay size remain unchanged. Together, these results provide the clearest evidence that prior attack experience produces reusable policy parameters rather than merely improving later images through an easier stream order.

## E. Matched Planner Attribution

Table V compares planners using the same 100-image targeted manifest, starting points, executor, and 2,000-query budget. Full shared-online DQN significantly outperforms uniform random, train-disabled, and per-image-reset controls. It is not significantly better than the strongest fixed Base Memory policy or the deterministic schedule on this set. The fixedprobe result above is therefore the primary learning evidence, while this experiment identifies the relevant baselines and limits the claim.

To further test whether the learned policy is genuinely necessary, we also examine how much of the observed performance can be attributed to the network’s initialization rather than to adaptation. Five frozen-untrained network seeds are highly unstable, with aggregate median $L _ { 0 } = 2 1 7 . 5 \pm 1 6 9 . 3 ;$ the best seed reaches median 29.0. Thus a favorable random output bias can occasionally imitate a useful fixed action preference, but it is not a reliable substitute for adaptation.

TABLE VI: State ablation for the final relative planner. Median, Mean, and P90 are for $L _ { 0 }$
<table><tr><td>State variant</td><td>ASR @.01</td><td>Median Mean</td><td>P90</td></tr><tr><td>Full state</td><td>.59</td><td>24.5 35.56</td><td>67.0</td></tr><tr><td>No action history/effectiveness</td><td>.54</td><td>27.0 37.77</td><td>71.1</td></tr><tr><td>No recent progress/stall</td><td>.54</td><td>29.038.64</td><td>73.7</td></tr><tr><td>No phase/budget</td><td>.50</td><td>30.536.41</td><td>71.0</td></tr><tr><td>Minimal legal mask</td><td>.48</td><td>31.5 46.67</td><td>90.2</td></tr><tr><td>State-shuffled, five-seed mean</td><td>.482</td><td>32.845.3589.16</td><td></td></tr><tr><td>State-blind</td><td>.44</td><td>44.5 82.01 203.4</td><td></td></tr></table>

TABLE VII: Reward ablation on targeted CIFAR-10. Median, Mean, and P90 are for $L _ { 0 }$
<table><tr><td>Reward</td><td>ASR @.01</td><td>Median</td><td>Mean</td><td>P90</td></tr><tr><td>Support gain/query</td><td>.56</td><td>24.0</td><td>34.46</td><td>68.0</td></tr><tr><td>Full shaped reward</td><td>.59</td><td>24.5</td><td>35.56</td><td>67.0</td></tr><tr><td>No stall bonus</td><td>.52</td><td>29.0</td><td>39.00</td><td>71.1</td></tr><tr><td>No tail/stall bonuses</td><td>.50</td><td>30.0</td><td>39.42</td><td>74.2</td></tr><tr><td>No phase weights</td><td>.49</td><td>31.0</td><td>40.29</td><td>72.2</td></tr><tr><td>No tail bonus</td><td>.47</td><td>31.0</td><td>44.00</td><td>71.3</td></tr></table>

## F. State Conditioning and Reward

Table VI removes groups of state features while retaining the same action space and executor. Full state significantly outperforms state-blind inference with 66/9/25 wins/ties/losses and $p = 3 . 7 6 \times 1 0 ^ { - 9 }$ . It also outperforms each of five stateshuffled seeds at $p < 0 . 0 5$ . The final relative planner therefore uses state alignment rather than learning only a global action prior.

The reward ablation in Table VII presents a more nuanced picture. Removing individual phase or tail terms degrades performance, yet the full shaped reward is statistically indistinguishable from a simple support-gain-per-query reward $( p = . 8 3 9 2 )$ . We therefore do not claim that the shaped reward is intrinsically superior.

## G. Granularity and Executor Composition

Table VIII separates scale selection from executor choice. Dynamic repartition is indispensable: fixing the partition increases median $L _ { 0 }$ from 24.5 to 128. Removing Memory substantially worsens the mean and tail, whereas Memory-only remains competitive in the median. Learned scale selection is not significantly better than fixed base scale or random scale. Thus the main learned contribution is state-conditioned executor scheduling, with granularity planning providing a useful shared action representation but not yet an independently significant gain.

The apparent advantage of ratio .01 over .002 on CIFAR-10 is not significant in the paired comparison $( p \ : = \ : . 0 8 6 1 )$ and has not been repeated across CIFAR-10 training seeds. We therefore retain .002 as the common CIFAR-10/ImageNet default rather than selecting the most favorable row after observing the evaluation set. The independent three-seed ImageNet sensitivity study below further tests this fixed choice without retuning the main experiments.

TABLE VIII: Relative granularity and executor ablation. Median, Mean, and P90 are for $L _ { 0 }$
<table><tr><td>Variant</td><td>ASR@.01</td><td>Median</td><td>Mean</td><td>P90</td></tr><tr><td>Erase ratio .01</td><td>.63</td><td>23.0</td><td>29.83</td><td>58.2</td></tr><tr><td>Full seven-action, ratio .002</td><td>.59</td><td>24.5</td><td>35.56</td><td>67.0</td></tr><tr><td>Erase ratio .005</td><td>.59</td><td>25.0</td><td>30.70</td><td>63.2</td></tr><tr><td>Policy DQN, fixed base scale</td><td>.62</td><td>25.5</td><td>34.66</td><td>60.1</td></tr><tr><td>Policy DQN, random scale</td><td>.54</td><td>26.5</td><td>36.23</td><td>66.1</td></tr><tr><td>Memory-only</td><td>.52</td><td>28.0</td><td>38.18</td><td>68.4</td></tr><tr><td>No Memory actions</td><td>.51</td><td>30.0</td><td>79.36</td><td>184.5</td></tr><tr><td>Erase ratio .001</td><td>.29</td><td>59.5</td><td>119.16</td><td>313.1</td></tr><tr><td>Fixed partition</td><td>.21</td><td>128.0</td><td>336.59</td><td>1115.0</td></tr><tr><td>Primary-only</td><td>.36</td><td>173.0</td><td>235.28</td><td>625.1</td></tr><tr><td>Safe-only</td><td>.03</td><td>301.0</td><td>343.95</td><td>569.1</td></tr></table>

TABLE IX: Cross-image learning lifecycle on the matched targeted attribution set. Median, Mean, and P90 are for $L _ { 0 }$
<table><tr><td>Lifecycle</td><td>ASR @.01</td><td>Median</td><td>Mean</td><td></td><td>P90 Updates</td></tr><tr><td>Keep weights + replay</td><td>.59</td><td>24.5</td><td>35.56</td><td>67.0</td><td>7936</td></tr><tr><td>Preadapt, frozen inference</td><td>.58</td><td>26.0</td><td>29.60</td><td>60.0</td><td>0</td></tr><tr><td>Reset weights + replay per image</td><td>.21</td><td>122.5</td><td>112.32</td><td>199.1</td><td>107</td></tr><tr><td>Keep weights, clear replay per image</td><td>.20</td><td></td><td>126.5173.73 361.6</td><td></td><td>147</td></tr></table>

## H. Lifecycle of Online Experience

Table IX studies what is inherited across images. A planner adapted on 120 images and then frozen reaches median $L _ { 0 } =$ 26 without test-time updates, close to continual online learning and far better than per-image reset. Retaining weights while clearing replay performs poorly, but it also receives only 147 optimizer updates because most single images do not reach the 128-transition warmup. This comparison therefore establishes the importance of the complete cross-image learning state but does not isolate replay-distribution effects from update count.

## I. Mechanistic Analysis

The learned behavior agrees with the intended stage structure. Table X reports the most productive action in each phase. Coarse Primary dominates early and middle support contraction, while Base Memory is markedly more effective in the late phase. In that late phase, Base Memory obtains support gain/query 0.459, compared with 0.038 for Base Primary and 0.096 for Base Safe. Combined with the state-shuffle result, this provides a mechanism-level explanation for why stateconditioned switching improves the tail.

Every reported matched ablation records zero illegal commits. The feasibility invariant is therefore preserved across planner, state, reward, granularity, and lifecycle variants.

## J. Hyperparameter Sensitivity

We assess sensitivity to hyperparameter choice by varying one parameter at a time on targeted ResNet50, using the same 100-image stream, a 20,000-query budget, and DQN seeds 42, 123, and 3407. The official configuration $( \rho = 0 . 0 0 2$ , learning rate $1 0 ^ { - 3 } )$ remains fixed; this is a post-hoc analysis, not a search that changes the reported setting. Figure 5 reports the geometric mean of the three median-sparsity trajectories with 95% confidence intervals.

TABLE X: Most productive action by support-search phase.
<table><tr><td>Phase Action</td><td></td><td>Support gain /query</td><td>Interpretation</td></tr><tr><td>Early</td><td>Coarse Primary</td><td></td><td>6.970 aggressive initial contraction</td></tr><tr><td></td><td>Middle Coarse Primary</td><td></td><td>1.884 continued coarse removal</td></tr><tr><td>Late</td><td>Base Memory</td><td></td><td>0.459 history-guided tail refinement</td></tr></table>

![](images/53e4435847b39574ffa39b3ba53f6b35ed8f7a4b9b209596fcda165bcb26874b.jpg)  
(a) Relative erase ratio $\rho .$

![](images/95e4b65f4bb3a2d41dc8cf6c762e4c04ee4d7da0258b4da7174870afa7aa55fa.jpg)  
(b) DQN learning rate.  
Fig. 5: ImageNet targeted hyperparameter sensitivity on ResNet50. Curves aggregate three seed-specific mediansparsity trajectories over the same 100-image stream; shaded regions denote 95% confidence intervals. The horizontal line marks the sparsity threshold 0.005 used in the main targeted comparison. Lower is better.

The relative erase ratio controls a granularity trade-off. Default $\rho ~ = ~ 0 . 0 0 2$ reaches sparsity 0.005 at about 11.5K queries, earlier than $\rho ~ = ~ 0 . 0 0 1$ (14.5K) and $\rho ~ = ~ 0 . 0 0 5$ (15.5K). At 20K queries, these three converge to 0.00299, 0.00293, and 0.00352, respectively, while $\rho = 0 . 0 1$ remains at 0.01993 and never reaches 0.005. The default is therefore chosen for fastest threshold crossing, not lowest terminal value, while avoiding the instability of overly coarse groups.

Learning-rate sensitivity is weaker. For rates from 2.5 × $1 0 ^ { - 4 }$ to $2 \times 1 0 ^ { - 3 }$ , all curves reach sparsity 0.005 between 11.5K and 12K queries and finish between 0.00276 and 0.00299, with overlapping confidence bands. This eightfold range indicates that ImageNet performance is not tied to a finely tuned DQN step size, supporting $1 0 ^ { - 3 }$ as the shared default.

## V. CONCLUSION

We introduced RLGPA, an online reinforcement learningguided granularity planning attack for decision-based sparse adversarial examples. RLGPA exposes stable executor–scale actions, converts them into support-relative groups, and dynamically rebuilds those groups around a feasibility-preserving adversarial incumbent. A shared Dueling Double DQN learns from hard-label support gains during the attack stream and can be frozen for update-free evaluation. Under the matched top-1- label oracle, results on CIFAR-10 and ImageNet show strong improvements over the directly comparable Pointwise and SparseEvo attacks in the low-sparsity region. Matched controls establish that the policy learns reusable state-conditioned scheduling, while ablations identify dynamic repartition and

Memory-guided late refinement as the principal mechanisms. The evidence also shows where the current method does not yet win: learned scale selection and shaped reward are not significantly superior to simpler alternatives. This combination of attack performance, causal attribution, and explicit limitations provides a reproducible basis for studying online reinforcement learning in query-bounded support search.

## REFERENCES

[1] Z. Yu, X. Liu, Y.-M. Cheung, L. Zhu, X. Xu, and N. Wang, “Fpad: Fuzzy-prototype-guided adversarial attack and defense for deep cross-modal hashing,” IEEE Trans. Circuits Syst. Video Technol., vol. 36, no. 2, pp. 2568–2580, 2026.

[2] M. Xie, Y. He, Z. Qin, and M. Fang, “Retouchuaa: Unconstrained adversarial attack via realistic image retouching,” IEEE Trans. Circuits Syst. Video Technol., vol. 35, no. 3, pp. 2586–2602, 2025.

[3] C. Szegedy, W. Zaremba, I. Sutskever, J. Bruna, D. Erhan, I. Goodfellow, and R. Fergus, “Intriguing properties of neural networks,” arXiv:1312.6199, 2013.

[4] C. Zhou, Y.-G. Wang, and G. Zhu, “Object-attentional untargeted adversarial attack,” J. Inf. Secur. Appl., vol. 81, p. 103710, 2024.

[5] J. Hu, J. Zhou, Z. Yang, H. Zheng, D. He, and Y. Li, “Adaguard: Location-aware adaptive defense against adversarial patch attacks in vision transformers,” IEEE Trans. Circuits Syst. Video Technol., vol. 36, no. 10, pp. 15 305–15 316, 2026.

[6] C. Jiang, Z. Wang, M. Dong, J. Gui, L. Dong, Y. Y. Tang, and J. T.-Y. Kwok, “Gradient perturbation guidance for boosting sparse adversarial attack transferability,” IEEE Trans. Circuits Syst. Video Technol., vol. 36, no. 5, pp. 6361–6373, 2026.

[7] C. Zhou, Y.-G. Wang, Z.-J. Wang, and X. Kang, “Lpnorm distortion-efficient adversarial attack,” Signal Process. Image Commun., vol. 131, p. 117241, 2025.

[8] C. Zhou, X. Shi, and Y.-G. Wang, “Query-efficient hardlabel black-box attack against vision transformers,” Appl. Soft Comput., vol. 183, p. 113686, 2025.

[9] V. Q. Vo, E. Abbasnejad, and D. C. Ranasinghe, “Query efficient decision based sparse attacks against blackbox deep learning models,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2022.

[10] L. Schott, J. Rauber, M. Bethge, and W. Brendel, “Towards the first adversarially robust neural network model on mnist,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2019.

[11] W. Brendel, J. Rauber, and M. Bethge, “Decision-based adversarial attacks: Reliable attacks against black-box machine learning models,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2018.

[12] J. Chen, M. I. Jordan, and M. J. Wainwright, “Hopskipjumpattack: A query-efficient decision-based attack,” in Proc. IEEE Symp. Secur. Privacy (S&P), 2020, pp. 1277–1294.

[13] Z. Yan, Y. Guo, J. Liang, and C. Zhang, “Policy-driven attack: Learning to query for hard-label black-box adver-

sarial examples,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2021.

[14] A. Ilyas, L. Engstrom, A. Athalye, and J. Lin, “Black-box adversarial attacks with limited queries and information,” in Proc. Int. Conf. Mach. Learn. (ICML), 2018, pp. 2137– 2146.

[15] M. Cheng, T. Le, P.-Y. Chen, H. Zhang, J. Yi, and C.-J. Hsieh, “Query-efficient hard-label black-box attack: an optimization-based approach,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2019, pp. 1–14.

[16] M. Cheng, S. Singh, P. H. Chen, P.-Y. Chen, S. Liu, and C.-J. Hsieh, “Sign-opt: a query-efficient hard-label adversarial attack,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2020.

[17] Y. Ran and Y.-G. Wang, “Sign-opt+: An improved sign optimization adversarial attack,” in Proc. IEEE Int. Conf. Image Process. (ICIP), 2022, pp. 461–465.

[18] F. Croce, M. Andriushchenko, N. D. Singh, N. Flammarion, and M. Hein, “Sparse-rs: A versatile framework for query-efficient sparse black-box adversarial attacks,” in Proc. AAAI Conf. Artif. Intell. (AAAI), vol. 36, no. 6, 2022, pp. 6437–6445.

[19] Q. V. Vo, E. Abbasnejad, and D. Ranasinghe, “Brusleattack: A query-efficient score-based black-box sparse adversarial attack,” in Proc. Int. Conf. Learn. Represent. (ICLR), vol. 2024, 2024, pp. 54 462–54 499.

[20] P. N. Williams and K. Li, “Black-box sparse adversarial attack via multi-objective optimisation cvpr proceedings,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR). IEEE, 2023, pp. 12 291–12 301.

[21] A. Ilyas, L. Engstrom, A. Athalye, and J. Lin, “Prior convictions: Black-box adversarial attacks with bandits and priors,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2018.

[22] A. Ilyas, L. Engstrom, and A. Madry, “Blackbox adversarial attacks using evolution strategies,” arXiv:1904.04433, 2019.

[23] C. Guo, J. Gardner, Y. You, A. G. Wilson, and K. Weinberger, “Simple black-box adversarial attacks,” in Proc. Int. Conf. Mach. Learn. (ICML), 2019, pp. 2484–2493.

[24] M. Andriushchenko, F. Croce, N. Flammarion, and M. Hein, “Square attack: A query-efficient black-box adversarial attack via random search,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2020, pp. 484–501.

[25] J. Su, D. V. Vargas, and K. Sakurai, “One pixel attack for fooling deep neural networks,” IEEE Trans. Evol. Comput., vol. 23, no. 5, pp. 828–841, 2019.

[26] F. Croce and M. Hein, “Sparse and imperceivable adversarial attacks,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV). IEEE, 2019, pp. 4723–4731.

[27] A. E. Cina, F. Villani, M. Pintor, L. Sch \` onherr, B. Biggio,¨ and M. Pelillo, “σ-zero: Gradient-based optimization of l -norm adversarial examples,” in Proc. Int. Conf. Learn. Represent. (ICLR), vol. 2025, 2025, pp. 91 199–91 211.

[28] I. J. Goodfellow, J. Shlens, and C. Szegedy, “Explaining and harnessing adversarial examples,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2015.

[29] N. Papernot, P. McDaniel, S. Jha, M. Fredrikson, Z. B.

Celik, and A. Swami, “The limitations of deep learning in adversarial settings,” in Proc. IEEE Eur. Symp. Secur. Privacy (EuroS&P). IEEE, 2016, pp. 372–387.

[30] S.-M. Moosavi-Dezfooli, A. Fawzi, and P. Frossard, “Deepfool: A simple and accurate method to fool deep neural networks,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 2574–2582.

[31] N. Carlini and D. Wagner, “Towards evaluating the robustness of neural networks,” in Proc. IEEE Symp. Secur. Privacy (S&P), 2017, pp. 39–57.

[32] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, “Towards deep learning models resistant to adversarial attacks,” arXiv:1706.06083, 2017.

[33] Y. Liu, X. Chen, C. Liu, and D. Song, “Delving into transferable adversarial examples and black-box attacks,” arXiv:1611.02770, 2016.

[34] X. Wei, Y. Guo, J. Yu, and B. Zhang, “Simultaneously optimizing perturbations and positions for black-box adversarial patch attacks,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 7, pp. 9041–9054, 2022.

[35] C.-C. Tu, P.-Y. Ting, P.-C. Chen, S. Liu, H. Zhang, J. Yi, C.-J. Hsieh, and S.-M. Cheng, “Autozoom: Autoencoderbased zeroth order optimization method for attacking black-box neural networks,” in Proc. AAAI Conf. Artif. Intell. (AAAI), vol. 33, no. 01, 2019, pp. 742–749.

[36] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski et al., “Human-level control through deep reinforcement learning,” Nature, vol. 518, no. 7540, pp. 529–533, 2015.

[37] H. Van Hasselt, A. Guez, and D. Silver, “Deep reinforcement learning with double q-learning,” in Proc. AAAI Conf. Artif. Intell. (AAAI), vol. 30, no. 1, 2016.

[38] Z. Wang, T. Schaul, M. Hessel, H. Hasselt, M. Lanctot, and N. Freitas, “Dueling network architectures for deep reinforcement learning,” in Proc. Int. Conf. Mach. Learn. (ICML), 2016, pp. 1995–2003.

[39] D. Hafner, J. Pasukonis, J. Ba, and T. Lillicrap, “Mastering diverse control tasks through world models,” Nature, vol. 640, no. 8059, pp. 647–653, 2025.

[40] P. Leroy, P. G. Morato, J. Pisane, A. Kolios, and D. Ernst, “Imp-marl: A suite of environments for large-scale infrastructure management planning via marl,” in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 36, 2023, pp. 53 522–53 551.