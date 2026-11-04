![](images/d293c7029867e6d47cc911436873fddf0d453a625d625e064f11091f4c5d57a7.jpg)

# Route Anything: A Study of Model and Policy Routing for General-Purpose AI Systems

Pengfei Zhou<sup>1,2</sup> Yixing Ma<sup>2,3</sup> Hexin Wang<sup>2‡</sup> Weidong Tang<sup>1</sup> Zhenglin Wan<sup>1</sup> Wangbo Zhao<sup>4</sup> Yang You<sup>1‡</sup> <sup>1</sup>National University of Singapore, HPC-AI Lab <sup>2</sup>InfRec, Cardinal AI Lab <sup>3</sup>University of California, Berkeley <sup>4</sup>Hong Kong University of Science and Technology

\# wanghexin@infrec.tech; # yangyou@nus.edu.sg <sup>‡</sup> Corresponding Author

<sup></sup> Project Page

<sup>§</sup> GitHub

Hugging Face

AGI should be all-knowing and all-capable. — Anonymous

Abstract Artificial general intelligence (AGI) requires a method that can understand, generate, reason, and act across diverse domains and modalities. Rather than relying on a single model to master every capability, a promising path toward AGI now is to orchestrate a heterogeneous ecosystem of specialized models, tools, and policies (candidates) into a unified AI system. At the heart of this vision lies a fundamental question: how can an AI system dynamically select and coordinate the most capable candidate for each task while balancing performance and computational cost? In this work, we present Route Anything, a unified framework for studying model and policy routing across generalpurpose AI systems. We develop a unified theoretical foundation that characterizes optimal routing under quality–cost trade-ofs and establishes how the information available to a router determines its attainable performance. To systematically investigate routing across diverse capabilities, we first construct an omni-modal trajectory matrix spanning 25 benchmarks across language understanding, image, video, and speech generation, coding and workplace agents, and robotic manipulation, comprising 374,401 scored runs from 452 candidate configurations. Building on this framework, we train and test diferent routers to investigate the potential of omni-modal model routing, the capabilities of learned routers, and the conditions under which routing improves eficiency. Our results reveal substantial opportunities for capability orchestration: per-input oracle routing outperforms the best fixed candidate by 5.4 points, and learned routers outperform it by 0.3 points across 16 benchmarks, by 2.1 points on video generation, and by 1.1 points on robot manipulation while using 24% less policy time. More importantly, routing raises eficiency by 8.82× on average compared with the reference model, including 25.5× on coding tasks and 7.1× on text question answering.

![](images/8c39436884110f1ff9ed936833f7bd73ad936a132a5c8c5f28991a8bf960ab5c.jpg)

<table><tr><td>25.5×</td><td>7.1×</td><td>1.35×</td><td>1.33×</td></tr><tr><td>Code ($)</td><td>Text QA ($)</td><td>Image gen. ($)</td><td>Robots (s)</td></tr></table>

Figure 1 | Routing across modalities. (a) For each input, the router picks one candidate from the pool based on demand, ranking candidates by value minus weighted cost. (b) On the test set, the best candidate per input (oracle) beats the best fixed candidate per benchmark by 5.4 points on average (Table 1). (c) Routing raises eficiency by 8.82× on average over the reference model, at a quality loss below 0.5 points. The eficiency gain, $\eta / \eta _ { B } .$ , compares quality per dollar (or per second of policy inference for robots) against the reference (Table 5).

## Contents

Introduction 3   
2 A Router’s Value 4   
3 Can Router Beat the Best Fixed Candidate? 6   
4 How Well Do Learned Routers Work? 7   
4.1 One Router Across Modalities 7   
4.2 Routers That Read the Scene 8   
4.3 What a Router Should Learn 9   
4.4 Matched Comparisons 10   
5 When Is a Cheaper Route Good Enough? 10   
5.1 The Price of a Route 10   
5.2 Spending More Within a Task 11   
6 Related Work 12   
Conclusion 12   
References 13   
A Proofs and Formal Results 17   
A.1 Proof of Lemma 1 . 17   
A.2 Proof of Theorem 2 17   
A.3 Proof of Proposition 3 18   
A.4 Proof of Proposition 4 19   
A.5 Proof of Proposition 5 19   
A.6 Proof of Corollary 6 . 20   
A.7 Portfolio Comparisons 21   
A.8 Retries and the Value of Routing Information . 21   
A.9 Router Memory 22   
B The Omni-Modal Trajectory Matrix: Details 23   
B.1 Benchmarks, Metrics, and Candidates . 23   
B.2 Splits . 24   
B.3 Routers 24   
B.4 Price Routes in Image and Speech Generation 26   
C Code and Text Testbeds: Details 26   
C.1 Data 26   
C.2 Price Routes . 27   
C.3 Portfolio Selection 28   
C.4 Verifiers and Overheads 29   
C.5 Routing Signals 30   
C.6 Router Memory 31   
D Limitations 31

## 1. Introduction

Agentic AI systems now can draw on multiple kinds of specialized models, tools, and policies: a language model answers a question, an image or video generator renders a prompt, a speech model reads text aloud, and a learned policy controls a robot. Rather than relying on one model to master every capability, a general-purpose system can orchestrate this heterogeneous ecosystem and choose, for each input, which module handles it. We call each such model, tool, or policy a candidate. The best candidate changes from input to input, so a router that chooses well is expected to match one fixed strongest candidate at a fraction of its price, or beat it by sending every input to the most suitable candidate that handles it best.

Routing research has mostly studied language models: cascades that escalate to a stronger model [1], routers learned from preferences [2], and quality–cost formulations that connect the two [3]. Whether routing helps across modalities remains an open question. Thus we present a unified framework Route Anything for studying model and policy routing across modalities. To support such anlyasitical framework, we first build an omni-modal trajectory matrix in which every candidate in a pool runs on the same inputs, for 25 benchmarks across text and image understanding, image, video, and speech generation, code and workplace agents, and robot manipulation, with three to seven candidates per pool and 374,401 scored runs of 452 candidate configurations (Figure 1). We train and test routers using diferent strategies and ask three questions: how much can routing gain, how well do learned routers work, and when is a cheaper route good enough?

Our unified routing theory frames all three: a router’s value is the best expected outcome, net of priced cost, given what it observes, so what it can reach is a ladder from the best fixed candidate to the per-input oracle, and a reference policy with a quality tolerance fixes the price of quality. The matrix measures each rung, and two code and text testbeds with recorded prices [4, 5] add the cost side. We also provide empirical insights through extensive experiments:

## Key findings

1. The potential gain for router is large (§3). On 19,814 test inputs from 24 benchmarks, the per-input oracle beats the best fixed candidate per benchmark by 5.4 points, from 0.7 for speech generation to 18–19 for video generation and robot manipulation. The best candidate per capability type, even chosen in hindsight, closes only 8% of the gap.

2. Learned routers beat the best fixed candidate by 0.3 points, and by up to 2.1 points per output group (§4.1). One capability-tree router over 16 benchmarks scores 79.98 against 79.68 for the fixed candidate and 84.80 for the oracle. Its gains come from agents, video, and image generation; none of 124 router configurations on GenEval2 and Qwen-Image-Bench beats the fixed generator significantly.

3. On robots, the router can also read the scene and saves time (§4.2). From three camera frames it sends clean scenes to PatchWAM and randomized ones to ME-Dex, matching the better policy in each condition: 71.6% success against 70.5% for the best single policy, with 24% less policy time.

4. Learn outcomes, not winners (§4.3). With the same inputs and backbone, regressing each policy’s success reaches 71.6% robot success and classifying the winner 42.8%, because a latency tie-break decides 94% of the winner labels. Winner-label routers on 16 benchmarks and a 68-type classifier also stay below the fixed candidate.

5. Compare candidates on the same inputs (§4.4). A fixed baseline that required a score on every training input dropped Opus 5.5, which missed 11 of 1,122 HLE scores, and turned a learned router’s 0.4-point deficit into a 2.4-point gain.

6. A cheaper route is often good enough (§5.1). Against the reference, routing raises eficiency by 8.82× on average over four testbeds, including 25.5× on code and 7.1× on text QA. On CodeRouterBench the router fitted for zero quality loss never calls the Opus reference; the certified zero-loss floor holds on test in all 200 splits at a median 19.4×, the fitted one in 12.5%. In image generation, a price route cuts cost by 26% within half a point.

7. Richer candidates and finer states are worth their cost only if they beat what they replace (§5.2). On CodeRouterBench, a failure roughly halves the next model’s success; a free compiler gate gives 4.09× dollar eficiency, a list-price LLM judge 0.30×.

Contributions. (1) A unified theory of routing. We connect resource prices and quality constraints to optimal model and policy selection, and characterize how the information available to a router determines its attainable performance (§2). (2) An omni-modal trajectory matrix. We run 452 candidate configurations on the same inputs of 25 benchmarks across text and image understanding, image, video, and speech generation, code and workplace agents, and robot manipulation, for 374,401 scored runs (§3 and Appendix B). (3) The potential of omni-modal routing. We quantify the potential gain of routing (5.4 points over the best fixed candidate), train and test learned routers that beat the best fixed candidate by 0.3 points overall and 2.1 points on video generation, show that task-type labels capture 8% of the potential gain on benchmarks with such labels, and examine how training targets and baseline choices afect measured gains (§3 and §4). (4) When routing improves eficiency. We show that routing raises eficiency over the reference by 8.82× on average and up to 25.5× on code with matched quality, and evaluate on diverse conditions (§5).

## 2. A Router’s Value

Sections 3 and 4 use the information ladder and the training targets (Propositions 3 and 4), mostly at � = 0 since most of the matrix records no costs; Section 5 adds prices, a reference, and a certified floor (Lemma 1, Theorem 2, and Proposition 5).

Candidates, states, and outcomes. An input � (a question, an image with a question, a task with files, a generation prompt, or camera frames with a robot state) comes with a finite menu $\mathcal { U } ( x )$ of candidates: language models, image, video, or speech generators, robot policies, or execution plans such $\mathtt { a s } ^ { \mathfrak { a } } \mathtt { t r y } \mathtt { a }$ cheap model, verify, and escalate on rejection”. Running candidate � delivers a value $V _ { u } \in [ 0 , 1 ]$ , the benchmark’s metric, and consumes $D _ { u } \ > \ 0$ units of one resource, such as dollars or seconds, with $\mathbb { E } [ D _ { u } ] < \infty$ . A router observes a state $s = \sigma ( x )$ , which may be the benchmark, a capability type, or the input itself, and chooses a candidate $g ( s )$ , possibly at random. We write $\nu ( s , u ) = \mathbb { E } [ V _ { u } \mid s ]$ and $c ( s , u ) = \mathbb { E } [ D _ { u } \mid s ]$ . Routing does not change which inputs arrive, so the law � of states is fixed; sequential choices within a task enter as complete plans.

Price, reference, and floor. At a resource price $\lambda \geq 0$ , router � earns $J _ { \lambda } ( g ) = \mathbb { E } [ V _ { g } ] - \lambda \mathbb { E } [ D _ { g } ] ;$ at $\lambda = 0$ it is ranked by quality alone. When no price is given, we compare � with a named reference policy, such as always calling the strongest model, whose value and cost on the same inputs are $Y _ { B }$ and $C _ { B } ,$ and define the quality change �, the saving S, the eficiency �, and the quality floor with tolerance $\epsilon \geq 0 \colon$

$$
Q = \mathbb { E } [ V _ { g } - Y _ { B } ] , \quad S = \mathbb { E } [ C _ { B } - D _ { g } ] , \quad \eta = \frac { \mathbb { E } [ V _ { g } ] } { \mathbb { E } [ D _ { g } ] } , \quad \eta _ { B } = \frac { \mathbb { E } [ Y _ { B } ] } { \mathbb { E } [ C _ { B } ] } , \quad Q \ge - \epsilon .\tag{1}
$$

Lemma 1 (Reference-relative gain). $\eta - \eta _ { B } = ( Q + \eta _ { B } S ) / \mathbb { E } [ D _ { g } ]$ . Eficiency improves if and only if $Q + \eta _ { B } S > 0$ , and the quality floor is the separate condition $Q \geq - \epsilon$

Both conditions are needed: a primary that is correct on 15% of tasks at a tenth of the cost of an always-correct reference has $\eta = 1 . 5 \eta _ { B }$ but $Q = - 0 . 8 5$

Theorem 2 (Optimal routing at a price and under a floor). Let the state take finitely many values, each with $\pi ( s ) > 0 _ { \astrosun }$ , let the menu depend on � only through �, and let costs be positive. (i) A router maximizes $J _ { \lambda }$ if and only $i f ,$ in every state, it plays only candidates maximizing $\nu ( s , u ) - \lambda c ( s , u )$ . The highest eficiency $\eta ^ { \star } = \operatorname* { m a x } _ { g } \eta ( g )$ is the unique root of $\begin{array} { r } { \Phi ( \lambda ) = \sum _ { s } \pi ( s ) } \end{array}$ m $\mathrm { a x } _ { u } [ \nu ( s , u ) - \lambda c ( s , u ) ]$ , and the eficiency-optimal routers are exactly the maximizers of $J _ { \eta ^ { \star } }$ . (ii) If the reference is itself a router and $\epsilon \geq 0 ,$ , an eficiency-optimal router under thefloor $Q \geq - \epsilon$ exists, and for some $\kappa \geq 0$ every such router maximizes $J _ { \lambda }$ at $\lambda _ { \epsilon } = \eta _ { \epsilon } ^ { \star } / ( 1 + \kappa )$ , where $\eta _ { \epsilon } ^ { \star }$ is the constrained optimum.

The proof combines Dinkelbach’s method [6] with linear-programming duality (Appendix A.2). Part (ii) says that maximizing eficiency under a floor yields no new kind of router: it is routing at one price, set by the reference and the tolerance rather than by hand or by a budget [3]. Without recorded costs, the rule is to play the candidate with the highest expected value given the state.

The information ladder. What a router can reach depends on what it observes. Write $W _ { u } = V _ { u } - \lambda D _ { u }$ and let $V ^ { \star } ( \sigma ) = \mathbb { E } \left[ \operatorname* { m a x } _ { u } \mathbb { E } \left[ W _ { u } \mid \sigma ( x ) \right] \right]$ be the best return of a router that observes $\sigma ( x )$ , attained, as in part (i) of Theorem 2, by playing in each state a candidate that maximizes $\mathbb { E } [ W _ { u } \mid \sigma ( x ) ]$

Proposition 3 (Information ladder). Let $\sigma _ { 1 }$ be coarser than $\sigma _ { 2 s }$ , that is, $\sigma _ { 1 } ( x )$ is a function of $\sigma _ { 2 } ( x )$ and let the menu depend on � only through $\sigma _ { 1 } ( x )$ . Then

$$
V ^ { \star } ( \sigma _ { 1 } ) ~ \le ~ V ^ { \star } ( \sigma _ { 2 } ) ~ \le ~ V ^ { \star } ( x ) ~ \le ~ \mathbb { E } \bigl [ \operatorname* { m a x } _ { u } W _ { u } \bigr ] .\tag{2}
$$

Thefirst inequality is an equality ifand only if,for almost every value � of� , a single candidate is optimal at almost every finer state �<sub>2</sub> with $\sigma _ { 1 } = s _ { 1 }$ . The last gap equals $\mathbb { E } \{ \mathbb { E } [ \operatorname* { m a x } _ { u } W _ { u } \mid x ] - \operatorname* { m a x } _ { u } \mathbb { E } [ W _ { u } \mid x ] \}$ }, and it vanishes if and only if, given almost every �, one candidate attains $\operatorname* { m a x } _ { u } W _ { u }$ almost surely.

With $\sigma _ { 1 }$ the benchmark, $V ^ { \star } ( \sigma _ { 1 } )$ is the best fixed candidate per benchmark. The outer rungs are measurable: the best fixed candidate on the left and the per-input oracle on the right. Inner rungs are bounded below by any router that observes that state and estimated optimistically by the best lookup chosen on the test inputs. A coarse state helps only where it changes the best candidate, and the last gap is outcome noise that no observable state predicts, as when a candidate’s success on an input is a coin flip.

What to learn. A learned router needs a training target. Two are common: each candidate’s outcome, and the identity of the winning candidate.

Proposition 4 (Outcomes, not winners). (a) For each candidate, the score that minimizes the expected binary cross-entropy, or squared error, against $V _ { u }$ given � is $m _ { u } ( x ) = \mathbb { E } [ V _ { u } \mid x ]$ , and routing to arg $\mathrm { m a x } _ { u } m _ { u } ( x )$ attains $V ^ { \star } ( x )$ at $\lambda = 0 .$ . (b) Let the winner be the candidate with the highest realized value, ties broken by a fixed rule, and let $p _ { u } ( x )$ be the probability that � wins, which a cross-entropy classifier on winner labels estimates. Routing to arg $\mathrm { m a x } _ { u } p _ { u } ( x )$ can lose almost half of the value range: if candidate � always delivers $a < 1$ and candidate � delivers 1 with probability $b \in ( \frac { 1 } { 2 } , a )$ and 0 otherwise, then $p _ { B } = b > p _ { A } .$ , while $\mathbb { E } [ V _ { B } ] = b < a$

Tie-breaks make it worse: with binary outcomes and ties sent to a faster candidate, a candidate that succeeds with probability 0.9 loses the label to a faster independent candidate that succeeds with probability 0.5 (it wins with probability 0.45). Per-candidate regression uses every outcome, including ties and failures, whereas a winner label keeps one choice per input.

Certifying the floor. Theorem 2 is applied to fitted values and costs, so a fitted floor need not hold on test inputs.

Proposition 5 (Certified quality floor). Let values lie in $[ 0 , 1 ] ,$ let $\epsilon \geq 0$ and $\alpha \in ( 0 , 1 )$ , and let the routers $g _ { 1 } , \ldots , g _ { K }$ and their order be fixed by the fitting data. Let $p _ { k }$ be valid �-values for the null hypotheses $Q _ { g _ { k } } < - \epsilon ,$ , computed on � inputs drawn i.i.d. from the deployment law, independently of the fitting data, with matched outcomes; for instance $p _ { k } = \exp \{ - n ( \hat { Q } _ { k } + \epsilon ) _ { + } ^ { 2 } / 2 \}$ , where $\hat { Q } _ { k }$ is the sample mean of $V _ { g _ { k } } - Y _ { B } ,$ . Let $\hat { g } = g _ { \hat { k } }$ with $\hat { k } = \operatorname* { m a x } \{ k : p 1 , \dots , p _ { k } \leq \alpha \}$ , or the reference if $p _ { 1 } > \alpha .$ . Then $\mathbb { P } ( Q _ { \hat { g } } < - \epsilon ) \leq \alpha .$

Table 1 | Per-input potential gain on the test split. Inputs on which every candidate of the pool is scored; values are benchmark metrics on a 0–100 scale, pooled over inputs. Fixed plays the best candidate per benchmark on training inputs, bestfixed in hindsight the best candidate per benchmark on these test inputs, and the oracle the best candidate per input.
<table><tr><td>Group</td><td>Benchmarks</td><td>Candidates</td><td>Inputs</td><td>Fixed</td><td>Best fixed in hindsight</td><td>Oracle</td><td>Gap</td></tr><tr><td>Text QA</td><td>7</td><td>3-4</td><td>9,482</td><td>83.27</td><td>83.27</td><td>87.56</td><td>4.28</td></tr><tr><td>Image-text QA</td><td>3</td><td>3</td><td>5,708</td><td>85.32</td><td>85.32</td><td>88.02</td><td>2.70</td></tr><tr><td>Image generation</td><td>5</td><td>4-6</td><td>1,515</td><td>82.93</td><td>82.99</td><td>87.92</td><td>4.99</td></tr><tr><td>Video generation</td><td>1</td><td>3</td><td>158</td><td>62.43</td><td>62.43</td><td>81.36</td><td>18.93</td></tr><tr><td>Speech generation</td><td>2</td><td>3</td><td>608</td><td>98.62</td><td>98.63</td><td>99.30</td><td>0.68</td></tr><tr><td>Code and workplace agents</td><td>5</td><td>4</td><td>343</td><td>69.17</td><td>69.78</td><td>77.29</td><td>8.11</td></tr><tr><td>Robot manipulation</td><td>1</td><td>7</td><td>2,000</td><td>70.45</td><td>70.45</td><td>88.85</td><td>18.40</td></tr><tr><td>All</td><td>24</td><td></td><td>19,814</td><td>82.60</td><td>82.62</td><td>87.98</td><td>5.38</td></tr></table>

A natural order is the price path (Figure 2): the maximizers of the fitted $J _ { \lambda }$ as � rises from 0 to the fitted $\eta ^ { \star }$ , along which fitted value and cost fall while fitted eficiency rises (Appendix A.5). The test stops at the first router it cannot certify, so no multiplicity correction is needed, as in fixed-sequence testing [7, 8]; returning the reference when

![](images/f740f6dc5a360667ab17c23e44afe8b9d97355f0621fbc6acb956034d8baca59.jpg)  
Figure 2 | Certifying a floor. Routers on the price path are tested in order on B.

nothing is certified mirrors high-confidence policy improvement [9]. The certified router comes with a certified price: the reference, the tolerance, and the level fix �.

## 3. Can Router Beat the Best Fixed Candidate?

Setup. Every candidate in a pool runs on the same inputs, and each run is scored by the benchmark’s own metric, rescaled to [0, 1] and reported on a 0–100 scale. The 25 benchmarks fall into seven groups (Table 1): text QA [10–16], image-text QA [17–19], image generation [20–24], video generation [25], speech generation [26], code and workplace agents [27–31], and robot manipulation [32]; Appendix B lists them with their metrics and candidates. The language-model candidates are Claude Opus 5.5, GPT-6.1 Sol, and DeepSeek V4.1 Flash, joined by GPT-6 Astra on long-context, knowledge, and agent tasks. Inputs are split 70/10/20 into training, development, and test sets by source group, so identical inputs and items from one source never cross splits; RoboTwin’s 2,000 test scenes come from 10 tasks the routers never see in training. The fixed candidate plays the best candidate per benchmark on training inputs, where missing outcomes never disqualify a candidate (Section 4.4 shows why this matters), and the oracle plays the best observed candidate on each input. These are the outer rungs of Proposition 3. We measure them on the 19,814 test inputs on which every candidate is scored, which leaves out SciCode, whose later steps contain each model’s own earlier code, and estimate the rungs in between with the best lookup per benchmark or per type chosen in hindsight.

Finding 1: The potential gain is large, and it lies within task types. On the 19,814 test inputs, the best fixed candidate per benchmark scores 82.60 and the per-input oracle 87.98 (Table 1). The gap is 0.7 points for speech generation, where every generator is nearly perfect, and 18–19 points for video generation and robot manipulation, where no candidate dominates. Little of it is available to a router that sees only the benchmark or the task type. The best candidate per

<table><tr><td>Per benchmark</td><td>24 benchmarks · gap 5.4</td></tr><tr><td>0.4%</td><td>99.6% within types</td></tr><tr><td>Per capability type</td><td>18 benchmarks · gap 10.4</td></tr><tr><td>8%</td><td>92% within types</td></tr><tr><td>Per task and scene</td><td>robots · gap 18.4</td></tr><tr><td>23%</td><td>77% within types</td></tr></table>

Figure 3 | Where the potential gain lies. Gap recovered by the best candidate per type.

benchmark chosen in hindsight raises 82.60 only to 82.62. On the 18 benchmarks of the rule-based capability tree, which assigns each of 5,034 test inputs to one of 107 capability leaves (Section 4.1), the best candidate per leaf chosen in hindsight scores 77.89, against 77.03 for the fixed candidate and 87.44 for the oracle: 8% of the gap lies between types. On RoboTwin, the best policy per task and scene condition, again chosen on the test scenes, recovers 23% of the 18.4-point gap. Code tasks show the same split. The best released CodeRouterBench routers capture 55% of the value of perfect routing information when the task type is unknown, but only 6–8% within a task type (Appendix C.5).

Inside a type, outcomes vary input by input. On GenEval2, each of the five other generators beats the fixed one by more than two points on 76–100 training prompts while trailing it on more, and the per-prompt diference has a standard deviation of 31–38 points against mean diferences of 4–14. Part of this variation is noise that no router can predict (the last gap of Proposition 3). On AA-LCR, where every question is answered three times, the oracle’s 2.4-point margin over the fixed candidate vanishes once each candidate is scored by its mean over the three runs.

Section 4 studies learned routers on every benchmark except seven understanding ones (MMLU-Pro, MMLU-Redux, SuperGPQA, GPQA, and the three image-text QA benchmarks), which hold three quarters of the inputs and half of the pooled gap in Table 1, with gaps of 1.4–6.0 points; every benchmark with a gap above 8 points is studied.

## 4. How Well Do Learned Routers Work?

Our routers fine-tune Qwen3.5-4B on the task text and the output requirement, and for robots also on three camera frames and the robot state (Appendix B.3). An outcome router scores each pair of input and candidate profile with a sigmoid head, trained by cross-entropy against the candidate’s normalized outcome (Proposition 4(a)), and plays the highest score. A type router classifies the input into a capability type and plays that type’s best candidate, read from a table fitted on training inputs, or the fixed candidate when it cannot place the input. A winner-label router is trained to name the best candidate. Each study is scored on the test inputs where all its compared policies have outcomes (Table 2); diferences carry paired 95% bootstrap intervals over source groups, or over scenes for RoboTwin.

## 4.1. One Router Across Modalities

Setup. The capability-tree router [33] is a type router whose types are the 157 leaves of a tree over input modality, task, and capability, on 18 benchmarks: all except MMLU-Pro, MMLU-Redux, SuperGPQA, GPQA, and the three image-text QA benchmarks. A Qwen3.5-4B classifier, trained on all benchmarks at once, predicts the leaf, and one global table, fitted on training inputs, maps each leaf to its best candidate; unresolved inputs go to the fixed candidate. We evaluate it on the 3,444 test inputs of 16 benchmarks on which every compared policy is scored (Table 3); the robot benchmark is handled by the outcome router of Section 4.2.

Table 2 | Evaluation panels. Each study uses the test inputs on which all its compared policies are scored; fixed-candidate and oracle scores (0–100) are comparable only within a row. The imagerouting suite of Section 4.1 uses 158 GenEval2 and 193 Qwen-Image-Bench prompts, and the image price routes of Section 5.1 use the 1,515 image-generation inputs of Table 1.
<table><tr><td>Study</td><td>Policies compared</td><td>Benchmarks</td><td>Inputs</td><td>Seeds</td><td>Fixed</td><td>Oracle</td></tr><tr><td>Potential gain (§3)</td><td>fixed candidate, hindsight lookups</td><td>24</td><td>19,814</td><td></td><td>82.60</td><td>87.98</td></tr><tr><td>Rule-based tree (§3, §4.1)</td><td>rule-based type router</td><td>18</td><td>5,034</td><td></td><td>77.03</td><td>87.44</td></tr><tr><td>One router (§4.1)</td><td>capability-tree and untrained routers</td><td>16</td><td>3,444</td><td>1</td><td>79.68</td><td>84.80</td></tr><tr><td>Robots (§4.2, §4.3)</td><td>outcome and winner-label routers</td><td>1</td><td>2,000</td><td></td><td>70.45</td><td>88.85</td></tr><tr><td>Targets (§4.3, §4.4)</td><td>outcome and type routers</td><td>11</td><td>1,302</td><td>4</td><td>68.30</td><td>75.84</td></tr><tr><td>Winner labels (§4.3)</td><td>12 winner-label recipes</td><td>16</td><td>3,211</td><td>1</td><td>79.44</td><td>85.77</td></tr></table>

Table 3 | Learned routers against the fixed candidate and the oracle. Test inputs; one training seed. Share is the router’s gain over the fixed candidate divided by the oracle’s. The first block is the capability-tree router; the robot row is the outcome router of Section 4.2.
<table><tr><td>Group (benchmarks)</td><td>Inputs</td><td>Fixed</td><td>Router</td><td>Oracle</td><td>Router – fixed</td><td>Share</td></tr><tr><td>Text QA (3)</td><td>588</td><td>65.14</td><td>64.97</td><td>71.43</td><td>-0.17</td><td>-3%</td></tr><tr><td>Code and workplace agents (5)</td><td>519</td><td>69.24</td><td>70.01</td><td>74.78</td><td>+0.77</td><td>14%</td></tr><tr><td>Image generation (5)</td><td>1,559</td><td>82.96</td><td>83.21</td><td>87.92</td><td>+0.25</td><td>5%</td></tr><tr><td>Video generation (1)</td><td>161</td><td>62.17</td><td>64.28</td><td>80.22</td><td>+2.12</td><td>12%</td></tr><tr><td>Speech generation (2)</td><td>617</td><td>98.63</td><td>98.63</td><td>99.30</td><td>0.00</td><td>0%</td></tr><tr><td>All 16 benchmarks</td><td>3,444</td><td>79.68</td><td>79.98</td><td>84.80</td><td>+0.30</td><td>6%</td></tr><tr><td>Robot manipulation (outcome router)</td><td>2,000</td><td>70.45</td><td>71.55</td><td>88.85</td><td>+1.10</td><td>6%</td></tr></table>

Finding 2: learned routers beat the best fixed candidate by 0.3 points, and by up to 2.1 points per output group. One router over 16 benchmarks in five output groups scores 79.98, against 79.68 for the fixed candidate and 84.80 for the oracle. It matches the fixed candidate to within 0.01 points on 10 benchmarks and departs from it where candidates complement each other input by input: Terminal-Bench (+8.3 on 36 inputs), VBench-2.0 (+2.1), OneIG (+1.1), and GDP.pdf (+1.0); its largest loss is AA-LCR (−2.4). The leaf classifier is 86–89% accurate, yet 83–86% of inputs still go to the fixed candidate, because within a benchmark one candidate wins almost every type (Proposition 3). A rule-based tree with 107 leaves, which needs no classifier, scores 0.09 points above the fixed candidate, almost all of it on OneIG. In image generation, across 31 router designs and 124 configurations on GenEval2 and Qwen-Image-Bench, none beats the fixed generator with an interval above zero, and 44 are significantly worse. An untrained Qwen3.5-4B that picks the candidate itself scores 1.15 points below the fixed candidate.

## 4.2. Routers That Read the Scene

Setup. On RoboTwin, the outcome router reads the instruction, three 240 × 320 camera frames, and the robot state, and predicts the success of each of seven vision-languageaction policies with one sigmoid head per policy, trained by cross-entropy on all 49,000 training outcomes. No task identity enters the input, and the 2,000 test scenes come from 10 tasks the router never sees, half in clean scenes and half in domain-randomized ones. ME-Dex is the best policy on training scenes and also on the test scenes (70.45%); PatchWAM is second (63.65%). A policy’s time on a scene is its inference seconds, summed over its queries.

![](images/182ef7e04249d08fc646528048e1c29b438d45601b5da06963cf33cacc306c2a.jpg)  
Figure 4 | Routing by scene condition. Robot success on 2,000 test scenes.

Table 4 | Training targets and the fixed baseline. Paired 95% intervals over source groups (scenes for robots). The first block averages four training seeds on 1,302 test inputs of 11 text, agent, and image benchmarks; the fixed candidate is matched (Section 4.4).
<table><tr><td>Study</td><td>Policy</td><td>Inputs</td><td>Score</td><td>– matched fixed [95% CI]</td></tr><tr><td rowspan="4">11 benchmarks, 4 seeds</td><td> Matched fixed candidate</td><td>1,302</td><td>68.30</td><td></td></tr><tr><td>Outcome router</td><td>1,302</td><td>67.90</td><td>-0.40 [-1.19, +0.33]</td></tr><tr><td>68-type classifier + table</td><td>1,302</td><td>66.98</td><td>-1.32 [-3.05, +0.34]</td></tr><tr><td>Oracle</td><td>1,302</td><td>75.84</td><td>+7.55</td></tr><tr><td rowspan="3">16 benchmarks</td><td>Fixed candidate (best training mean)</td><td>3,211</td><td>79.44</td><td></td></tr><tr><td>Winner-label router (generative)</td><td>3,211</td><td>78.87</td><td>-0.58</td></tr><tr><td>Oracle</td><td>3,211</td><td>85.77</td><td>+6.33</td></tr><tr><td rowspan="2">Robot manipulation</td><td>Outcome router (per-policy success)</td><td>2,000</td><td>71.55</td><td>+1.10 [-0.40, +2.55]</td></tr><tr><td>Winner-label classifier</td><td>2,000</td><td>42.80</td><td>-27.65</td></tr></table>

Finding 3: on robots, the router reads the scene and saves time. The router sends 929 of the 1,000 clean scenes to PatchWAM and 842 of the 1,000 randomized scenes to ME-Dex (141 more to GigaBrain). The best policy flips between the two conditions, and the router tracks it: on clean scenes it scores 81.6%, against 81.7% for PatchWAM and 78.4% for ME-Dex; on randomized scenes 61.5%, against 62.5% for ME-Dex and 45.6% for PatchWAM (Figure 4). Overall success is 71.55% against 70.45% (+1.10 [−0.40, +2.55]), at 11.05 instead of 14.51 seconds of policy time per scene, 24% less: 1.33× the success per policy-second of the best single policy. This is the coarse rung of Proposition 3 learned from pixels, since the condition-level choice alone would score 72.1%. The remaining 17% to the oracle require predicting which policy succeeds on a given scene.

## 4.3. What a Router Should Learn

Finding 4: learn outcomes, not winners. With the same inputs, backbone, and training scenes, the winner-label classifier reaches 42.8% robot success against 71.6% for perpolicy outcome regression (Table 4). The labels explain the gap, as Proposition 4(b) predicts: several policies succeed on most training scenes, a latency tie-break decides 94% of the labels, and 24,556 of the 31,408 successful training outcomes become wrong-class targets, so the classifier learns

![](images/ef69cc96b07e37476c540d9dacd1c1d8dc00ac8554937f75d6f61d41b4c662c3.jpg)  
Figure 5 | Learn outcomes, not winners. Winner labels mislabel most successes.

to pick fast policies (1.9 seconds per scene) that often fail. Outside robotics, no panel compares the two targets directly, and no target beats the fixed candidate. Generative winner-label routers on 16 benchmarks learn from 11,273 labels of which only a third are decided by a unique best outcome, the rest by cost, tokens, or a fixed order; the best of 12 recipes scores 78.87 against 79.44 for the fixed candidate. On 11 text, agent, and image benchmarks, the outcome router stays within noise of the matched fixed candidate (−0.40), and a type router over 68 types, although 75% accurate, maps 93% of its predictions to the same candidate as the true type and scores 1.32 points below it.

## 4.4. Matched Comparisons

Finding 5: compare candidates on the same inputs. A fixed baseline that admits a model only if it has a score on every training input drops strong models over a few missing scores. In the 11-benchmark study, Opus 5.5 missed a few scores on seven benchmarks, for example 11 of 1,122 training inputs of HLE, so the baseline fell back to the next model there: on HLE it scored 52.90 instead of 58.24, and on AA-Omniscience 70.00 instead of 77.50. Pooled, the baseline dropped from 68.30 to 65.52, and the outcome

![](images/5973b77c27f3241afbf7146375a7fd9a42c57b6fc5225f2bc8df773203aaf1b3.jpg)  
Figure 6 | Compare candidates on the same inputs. Missing scores flip the sign.

router’s 0.40-point deficit became a 2.38-point gain (Table 4). Choosing and evaluating the fixed candidate under the routers’ fallback, which uses GPT-6 Astra’s outcome where Opus 5.5 has none, removes the artifact. The same rule governs portfolios. Adding a model changes which tasks reach the fallback, so averages frozen at the current portfolio misrank candidates, whereas comparing both portfolios on the same tasks picks the best added model in 96.4–97.1% of CodeRouterBench comparisons, against 86.3–94.3% for frozen averages (Appendix C.3).

## 5. When Is a Cheaper Route Good Enough?

## 5.1. The Price of a Route

Setup. A route trades quality for cost where candidates difer more in price than in value. The matrix records prices for image and speech generation and time for robot policies. For code and text, two testbeds add token prices: CodeRouterBench [4], with eight models on 9,999 coding tasks (7,080 probing tasks for fitting and 2,919 test tasks) and the reference claude-opus-4-6 (Opus 4.6), and RouterBench [5], with eleven models on 36,497 mostly non-code prompts split 70/30 and the reference gpt-4-1106-preview. On CodeRouterBench, states are the 21 dimension×source cells, and each state ofers 22 candidates: direct dispatch to one of eight models, or one of seven primaries followed by Opus 4.6 when the primary’s released score, an oracle verifier, is below $\theta \in \{ 1 , 0 . 5 \}$ . Such a reference-fallback cascade with verifier decision $H \in \{ 0 , 1 \}$ and primary-plus-verification cost � delivers

$$
V = H Y _ { P } + ( 1 - H ) Y _ { B } , ~ D = A + ( 1 - H ) C _ { B } .\tag{3}
$$

Values, costs, and state frequencies are fitted on probing tasks, the floor $Q \geq - \epsilon$ is imposed there, and policies are evaluated on test tasks; always-Opus scores 0.4383 at 11.655 m\$ per task. RouterBench uses source-family states and the same construction. For image generation, states are the five benchmarks, candidates are the generators (three on GenEval2, six elsewhere), and the reference plays the best generator per benchmark (the fixed candidate).

When the reference is worth calling. A candidate that calls the reference is worth its price only while the reference earns enough on the inputs it receives.

Corollary 6 (Admission price). In the setting of Theorem 2, let $\lambda ^ { B }$ be the largest price at which some candidate that calls the reference maximizes $\nu ( s , \cdot ) - \lambda c ( s , \cdot )$ in some state. I $\epsilon \lambda ^ { B } < \eta ^ { \star } ,$ , the eficiencyoptimal router never calls the reference, and neither does a floor-optimal router whose price $\lambda _ { \epsilon }$ exceeds $\bar { \lambda ^ { B } } .$ . A cascade (3) whose primary is also available by direct dispatch maximizes only at prices up to $\mathbb { E } [ \left( Y _ { B } - Y _ { P } \right) \mathbf { 1 } _ { H = 0 } \ | \ s ] / \mathbb { E } [ C _ { B } \mathbf { 1 } _ { H = 0 } \ | \ s ] _ { \ }$ , the reference’s added value per dollar on the inputs it receives.

Table 5 | Price routes across testbeds. Test-set quality change $\Delta V = \mathbb { E } [ V - Y _ { B } ]$ in points (0–100) with paired 95% intervals, and eficiency $\eta / \eta _ { B }$ against the reference. Fitted routes meet the floor on fitting data; certified routes are medians over 200 random halvings of the fitting data (Proposition 5).
<table><tr><td>Testbed (resource)</td><td>Reference</td><td>Route</td><td>∆V [95% CI]</td><td> $\eta / \eta _ { B }$ </td></tr><tr><td>Code, CodeRouterBench ($) Always Opus 4.6</td><td></td><td>Fitted,  $\epsilon = 0$ </td><td> $- 0 . 4 4 \ [ - 1 . 7 5 , + 0 . 9 7 ]$ </td><td>25.5×</td></tr><tr><td rowspan="2">Text QA, RouterBench ($)</td><td rowspan="2">Always GPT-4</td><td>Certified,  $\epsilon = 0$ </td><td> $\geq 0 \mathrm { i n } 2 0 0 \mathrm { o f } 2 0 0$ </td><td>19.4×</td></tr><tr><td>Fitted,  $\epsilon = 0$ </td><td> $- 0 . 4 0 \ [ - 0 . 8 4 , + 0 . 0 8 ]$ </td><td>7.1×</td></tr><tr><td rowspan="2">Image generation ($)</td><td rowspan="2">Fixed generator</td><td>Certified,  $\epsilon = 0$ </td><td> $\geq 0 \mathrm { i n } 2 0 0 \mathrm { o f } 2 0 0$ </td><td>6.7×</td></tr><tr><td>Fitted,  $\epsilon = 0 . 5$ </td><td> $- 0 . 4 7 \ [ - 0 . 8 7 , - 0 . 0 6 ]$ </td><td>1.35×</td></tr><tr><td rowspan="2">Robot manipulation (s)</td><td rowspan="2">Fixed policy</td><td>Fitted,  $\epsilon = 2$ </td><td> $- 1 . 5 6 \ [ - 2 . 3 9 , - 0 . 7 3 ]$ </td><td>2.03×</td></tr><tr><td>Outcome router</td><td> $+ 1 . 1 0 \ [ - 0 . 4 0 , + 2 . 5 5 ]$ </td><td>1.33×</td></tr></table>

Finding 6: a cheaper route is often good enough. Across the four testbeds of Table 5, routing raises eficiency over the reference by 8.82× on average (the mean of 25.5, 7.1, 1.35, and 1.33×, with image generation at $\epsilon = 0 . 5 )$ . On CodeRouterBench, the admission price of Opus 4.6 is $\lambda ^ { B } =$ 100 score per dollar $( 2 . 9 \eta _ { B } )$ , below the zero-loss floor price of 144 and far below $\eta ^ { \star } = 1 0 2 2$ , so by Corollary 6 the zeroloss router never calls Opus. It dispatches every task to a cheaper model and reaches 25.5× Opus’s eficiency at a 0.44- point loss (Table 5). Always calling Qwen3-Max, the cheapest model, already reaches 20.1× at a 4.2-point loss; routing by state recovers that loss while adding 27% eficiency. The states need not be oracle labels: prompt-inferred cells reach 25.3×, and learned quality–cost routers 22.7–24.2× at their zero-loss points (Appendix C.2). Without states, every zero-loss policy calls Opus, which then runs on 28.7% of tasks. On RouterBench, the best router that avoids the reference scores 9.7 points below it, so the zero-loss router cascades cheap primaries to the reference and reaches 7.1×. A fitted floor is not a guarantee. Refitted on half of the probing tasks, the zero-loss floor holds on test in only 12.5% of 200 splits; certified on the other half by Proposition 5 with empirical-Bernstein �-values on the mixture-refined path, it holds in all 200, at a median 19.4×. Image generation follows the same logic. GPT Image 2.5 costs \$0.21 per image, 3.5–5.4× the other generators, and is the best candidate on two of the five benchmarks. The route that stays within half a point on training prompts moves Qwen-Image-Bench to Grok Imagine 2.0 and OneIG to Qwen-Image 3, cutting test cost by 26% at a 0.47-point loss, and a two-point tolerance halves cost. In speech generation the best generator is also among the cheapest, so there is nothing to trade, and on robots the time saving of Section 4.2 comes with no loss of success. Router overhead is small next to these prices: a 4B router takes 0.13s per decision on one H20 GPU, whereas generating an image takes 27–158s and a video 211–314s.

![](images/ccf7c3ce42892be944ac4f2c988d7fa1c6698736f4c7e435eada41f7bd62ed12.jpg)  
Figure 7 | Admission vs. floor price in units of $\eta _ { B } ;$ the shaded range is $\lambda \leq \lambda ^ { B }$

## 5.2. Spending More Within a Task

Setup. A candidate can spend more inside a task: a plan that retries after a failure, or that verifies an output and escalates on rejection. A router can also buy a finer state: a forecast of a candidate’s success, or a record of past outcomes. By Theorem 2, a richer candidate is worth playing only if its value net of its full cost beats the candidates it replaces, and that value must be computed on the inputs that reach each step (Proposition 8). By Proposition 3, a finer state can only help an optimal router, but it is worth buying only if the help exceeds its price (Proposition 9). We test both on CodeRouterBench, with fresh model calls for verifier verdicts and forecasts, and semi-synthetic streams for memory (Appendix C).

Finding 7: richer candidates and finer states are worth their cost only if they beat what they replace. Retries. A failure leaves harder tasks: in all 42 ordered model pairs, the second model succeeds less often after the first fails (median ratio 0.49 at � = 1). Verifiers. For Qwen3-Max at � = 0.5, a free compiler gate gives 4.09× dollar eficiency at a 2.4-point loss, and an LLM judge 2.63× if free but 0.30× at its list price; in seconds, always-Opus beats every verified cascade, because Opus answers faster than the primaries. Finer states. A forecast can save at most the attempt it replaces: a DeepSeek-v4-pro forecast costs 2.4 attempts at list price, a prompt-only TF-IDF router 0.06% of one.

## 6. Related Work

Routing language models. Cascades and routers trade quality against cost [1–3], and learned routers estimate each model’s quality from the query [34–38]. The acceptance step of a cascade is a reject option [39], studied as confidence-based or budgeted deferral [40–44]. Dekoninck et al. [3] characterize optimal routing and cascading through a Lagrangian price tuned to a budget; here, the reference and the quality tolerance fix that price, and a finite-sample test certifies the resulting floor. Routing benchmarks record per-model outcomes and costs on shared prompts [4, 5].

Routing beyond text. SCOPE-Router arranges vision-language models for execution-oriented tasks [19], and routing ideas also inspire vision-token pruning strategies [45], dynamic submodels for video segmentation [46], and acceleration policies inside difusion transformers [47]. Coordinators, harness-level routers, and token-level routers extend routing beyond model identity [48–52]. We study routing across output types on one matrix: language models, image, video, and speech generators, and robot policies, all scored on shared inputs.

Portfolios and tools. Algorithm portfolios choose among complementary solvers per instance [53– 55]. Their virtual best solver corresponds to our per-input oracle, and its advantage over the single best solver gives the potential gain in Proposition 3. Residual coverage measures what an additional solver contributes to the portfolio [56, 57]. Accounting for call costs connects this selection problem to fractional programming, which optimizes ratio objectives such as quality per unit cost [6, 58]. The value of information addresses the related question of whether acquiring a routing signal is worth its cost [59, 60]. Similar cost considerations arise in test-time compute, where gains from extra calls depend on the task and allocation [61, 62], while imperfect verifiers can limit the gains from resampling [63]. Feedback-driven self-repair uses earlier attempts to guide later ones [64, 65], making resampling at equal cost a natural baseline [66].

## 7. Conclusion

Across text, images, video, speech, code, and robots, the best candidate changes from input to input, and the per-input oracle sits 5.4 points above the best fixed candidate per benchmark. Most of that potential gain lies within task types, and a learned type router already scores 0.3 points above the best fixed candidate across 16 benchmarks, and up to 2.1 points within an output group. In embodied scenarios, a router trained on camera frames tracks the best VLA policy in each scene condition and beats the best single policy by 1.1 points with 24% less policy time. Routing’s most reliable gain is a cheaper route that is good enough, raising eficiency over the reference by 8.82× on average: when candidates are close in quality and far apart in cost, the reference and the quality tolerance fix the price, and a finite-sample test certifies the quality floor. Three rules follow: (1) compare candidates on the same inputs, and measure gains against the matched best fixed candidate; (2) train routers on per-candidate outcomes, and look for gains within task types; (3) fix the reference and the tolerance before ranking routes, and certify the floor on independent inputs.

## References

[1] Lingjiao Chen, Matei Zaharia, and James Zou. FrugalGPT: How to use large language models while reducing cost and improving performance. Transactions on Machine Learning Research, 2024.

[2] Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E Gonzalez, Mohammed Kadous, and Ion Stoica. RouteLLM: Learning to route LLMs from preference data. In International Conference on Learning Representations, volume 2025, pages 34433–34448, 2025.

[3] Jasper Dekoninck, Maximilian Baader, and Martin Vechev. A unified approach to routing and cascading for LLMs. In Proceedings of the 42nd International Conference on Machine Learning, volume 267, pages 12987–13010, 2025.

[4] Pengfei Zhou, Zhiwei Tang, Yixing Ma, Jiasheng Tang, Yizeng Han, Zhenglin Wan, Fanqing Meng, Wei Wang, Bohan Zhuang, Wangbo Zhao, et al. Agent-as-a-Router: Agentic model routing for coding tasks. arXiv preprint arXiv:2606.22902, 2026.

[5] Qitian Jason Hu, Jacob Bieker, Xiuyu Li, Nan Jiang, Benjamin Keigwin, Gaurav Ranganath, Kurt Keutzer, and Shriyash Kaustubh Upadhyay. Routerbench: A benchmark for multi-LLM routing system. arXiv preprint arXiv:2403.12031, 2024.

[6] Werner Dinkelbach. On nonlinear fractional programming. Management science, 13(7):492–498, 1967.

[7] Peter H. Westfall and Alok Krishen. Optimally weighted, fixed sequence and gatekeeper multiple testing procedures. Journal of Statistical Planning and Inference, 99(1):25–40, 2001. doi: 10.1016/S0378-375 8(01)00077-5.

[8] Anastasios N. Angelopoulos, Stephen Bates, Emmanuel J. Candès, Michael I. Jordan, and Lihua Lei. Learn then test: Calibrating predictive algorithms to achieve risk control. The Annals ofApplied Statistics, 19(2): 1641–1662, 2025. doi: 10.1214/24-AOAS1998.

[9] Philip Thomas, Georgios Theocharous, and Mohammad Ghavamzadeh. High confidence policy improvement. In Proceedings of the 32nd International Conference on Machine Learning, volume 37, pages 2380–2388, 2015.

[10] Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. MMLU-Pro: A more robust and challenging multi-task language understanding benchmark. In Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, 2024.

[11] Aryo Pradipta Gema, Joshua Ong Jun Leang, Giwon Hong, Alessio Devoto, Alberto Carlo Maria Mancino, Rohit Saxena, Xuanli He, Yu Zhao, Xiaotang Du, Mohammad Reza Ghasemi Madani, et al. Are we done with MMLU? In Proceedings of the Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics, 2025.

[12] M-A-P Team, Xinrun Du, Yifan Yao, Kaijing Ma, et al. SuperGPQA: Scaling LLM evaluation across 285 graduate disciplines. In Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, 2025.

[13] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. GPQA: A graduate-level Google-proof Q&A benchmark. In Conference on Language Modeling, 2024.

[14] Artificial Analysis. Artificial Analysis intelligence benchmarking methodology. https://artificial analysis.ai/methodology/intelligence-benchmarking, 2026.

[15] Declan Jackson, William Keating, George Cameron, and Micah Hill-Smith. AA-Omniscience: Evaluating cross-domain knowledge reliability in large language models. arXiv preprint arXiv:2511.13029, 2025.

[16] Center for AI Safety, Scale AI, and HLE Contributors Consortium. A benchmark of expert-level academic questions to assess AI capabilities. Nature, 649(8099):1139–1146, 2026. doi: 10.1038/s41586-025-099 62-4.

[17] Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, et al. MMMU-Pro: A more robust multi-discipline multimodal understanding benchmark. In Proceedings of the Annual Meeting of the Association for Computational Linguistics, 2025.

[18] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. MMMU: A massive multi-discipline multimodal understanding and

reasoning benchmark for expert AGI. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[19] Tao Yu, Yifei Qu, Zhiqing Cui, Pengfei Zhou, Zhongtian Luo, Yujia Yang, Shenghua Chai, Haopeng Jin, Zhenghao Zhang, Xinming Wang, et al. SCOPE-Router: Cost-aware open-set VLM routing for execution-oriented tasks. arXiv preprint arXiv:2608.12127, 2026.

[20] Amita Kamath, Kai-Wei Chang, Ranjay Krishna, Luke Zettlemoyer, Yushi Hu, and Marjan Ghazvininejad. GenEval 2: Addressing benchmark drift in text-to-image evaluation. arXiv preprint arXiv:2512.16853, 2025.

[21] Niantong Li, Guangzheng Hu, Weixu Qiao, Ying Ba, Qichen Hong, et al. Qwen-Image-Bench: From generation to creation in text-to-image evaluation. arXiv preprint arXiv:2605.28091, 2026.

[22] Zhichao Yang, Tianjiao Gu, Jianjie Wang, Feiyu Lin, Xiangfei Sheng, Pengfei Chen, and Leida Li. LongT2IBench: A benchmark for evaluating long text-to-image generation with graph-structured annotations. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 11820–11828, 2026.

[23] Jingjing Chang, Yixiao Fang, Peng Xing, Shuhan Wu, Wei Cheng, Rui Wang, Xianfang Zeng, Gang Yu, and Hai-Bao Chen. OneIG-Bench: Omni-dimensional nuanced evaluation for image generation. In Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, 2025.

[24] Yuwei Niu, Munan Ning, Mengren Zheng, et al. WISE: World knowledge-informed semantic evaluation for text-to-image generation. In Proceedings of the 43rd International Conference on Machine Learning, volume 306, pages 93570–93590, 2026.

[25] Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, et al. VBench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness. arXiv preprint arXiv:2503.21755, 2025.

[26] Philip Anastassiou, Jiawei Chen, Jitong Chen, et al. Seed-TTS: A family of high-quality versatile speech generation models. arXiv preprint arXiv:2406.02430, 2024.

[27] Daniel Shepard and Robin Salimans. AutomationBench. arXiv preprint arXiv:2604.18934, 2026.

[28] Suhaas Garre, Emily Ritchie, Sushant Mehta, and Edwin Chen. GDP.pdf: Benchmarking grounded multimodal reasoning over professional PDF documents. arXiv preprint arXiv:2607.11192, 2026.

[29] Tejal Patwardhan, Rachel Dias, Elizabeth Proehl, et al. GDPval: Evaluating AI model performance on real-world economically valuable tasks. In International Conference on Learning Representations, 2026.

[30] Minyang Tian, Luyu Gao, Shizhuo Dylan Zhang, et al. SciCode: A research coding benchmark curated by scientists. In Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, 2024.

[31] Mike A. Merrill, Alexander G. Shaw, Nicholas Carlini, Boxuan Li, Harsh Raj, Ivan Bercovich, et al. Terminal-Bench: Benchmarking agents on hard, realistic tasks in command line interfaces. In International Conference on Learning Representations, 2026.

[32] Tianxing Chen, Zanxin Chen, Baijun Chen, et al. RoboTwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. In Proceedings of the 43rd International Conference on Machine Learning, volume 306, pages 13673–13699, 2026.

[33] Weidong Tang, Ruiheng Wang, Jingxuan Zhang, Hexin Wang, Haoyuan Chen, Zhengfeiyang Zhang, Yixing Ma, Zichen Jiang, Ziyu Guo, Haodong Li, Zhengbo Jiao, Zhiqing Cui, Xinxiang Yin, Shuo Liang, Zhenglin Wan, Hongzhan Lin, Wangbo Zhao, Kaipeng Zhang, Pengfei Zhou, Odin Zhang, and Yang You. The trinity: Capability-driven recursive self-improvement for multimodal agents. arXiv preprint arXiv:2610.36077, 2026. doi: 10.48550/arXiv.2610.36077. URL https://arxiv.org/abs/2610.36077.

[34] Dujian Ding, Ankur Mallick, Chi Wang, Robert Sim, Subhabrata Mukherjee, Victor Rühle, Laks V. S. Lakshmanan, and Ahmed Hassan Awadallah. Hybrid LLM: Cost-eficient and quality-aware query routing. In The Twelfth International Conference on Learning Representations, 2024.

[35] Shuhao Chen, Weisen Jiang, Baijiong Lin, James T. Kwok, and Yu Zhang. RouterDC: Query-based router by dual contrastive learning for assembling large language models. In Advances in Neural Information Processing Systems, volume 37, pages 66305–66328, 2024.

[36] Tao Feng, Yanzhen Shen, and Jiaxuan You. GraphRouter: A graph-based router for LLM selections. In The Thirteenth International Conference on Learning Representations, 2025.

[37] Richard Zhuang, Tianhao Wu, Zhaojin Wen, Andrew Li, Jiantao Jiao, and Kannan Ramchandran. EmbedLLM: Learning compact representations of large language models. In The Thirteenth International Conference on Learning Representations, 2025.

[38] Dujian Ding, Ankur Mallick, Shaokun Zhang, Chi Wang, Daniel Madrigal, Mirian del Carmen Hipolito Garcia, Menglin Xia, Laks V. S. Lakshmanan, Qingyun Wu, and Victor Rühle. BEST-route: Adaptive LLM routing with test-time optimal compute. In Proceedings of the 42nd International Conference on Machine Learning, volume 267, pages 13870–13884, 2025.

[39] Chao Chow. On optimum recognition error and reject tradeof. IEEE Transactions on information theory, 16(1):41–46, 1970.

[40] Wittawat Jitkrittum, Neha Gupta, Aditya K Menon, Harikrishna Narasimhan, Ankit Rawat, and Sanjiv Kumar. When does confidence-based cascade deferral sufice? In Advances in Neural Information Processing Systems, volume 36, pages 9891–9906, 2023.

[41] Neha Gupta, Harikrishna Narasimhan, Wittawat Jitkrittum, Ankit Singh Rawat, Aditya Krishna Menon, and Sanjiv Kumar. Language model cascades: Token-level uncertainty and beyond. In International Conference on Learning Representations, volume 2024, pages 4147–4180, 2024.

[42] Pranjal Aggarwal, Aman Madaan, Ankit Anand, Srividya Pranavi Potharaju, Swaroop Mishra, Pei Zhou, Aditya Gupta, Dheeraj Rajagopal, Karthik Kappaganthu, Yiming Yang, et al. AutoMix: Automatically mixing language models. In Advances in Neural Information Processing Systems, volume 37, pages 131000–131034, 2024.

[43] Lingjiao Chen, Matei Zaharia, and James Zou. FrugalML: How to use ML prediction APIs more accurately and cheaply. In Advances in neural information processing systems, volume 33, pages 10685–10696, 2020.

[44] Kirill Trapeznikov and Venkatesh Saligrama. Supervised sequential classification under budget constraints. In Artificial intelligence and statistics, pages 581–589, 2013.

[45] Haiji Liang, Pengfei Zhou, Zhenglin Wan, Wei Wang, Yang You, and Wangbo Zhao. Beyond one-size-fits-all: Sample-adaptive strategy routing for vision token pruning in MLLMs. arXiv preprint arXiv:2609.10346, 2026.

[46] Weidong Tang, Zhiyuan Liang, Xinyan Wan, Chen Zhu, Zhaopan Xu, Pengfei Zhou, Yan Song, Yang You, and Wangbo Zhao. Eficient video object segmentation and tracking with recurrent dynamic submodel. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20912–20921, 2026.

[47] Wangbo Zhao, Yizeng Han, Zhiwei Tang, Jiasheng Tang, Pengfei Zhou, Kai Wang, Bohan Zhuang, Zhangyang Wang, Fan Wang, and Yang You. RAPID<sup>3</sup>: Tri-level reinforced acceleration policies for difusion transformer. In International Conference on Learning Representations, volume 2026, pages 149423–149446, 2026.

[48] Stefan Nielsen, Edoardo Cetin, Peter Schwendeman, Qi Sun, Jinglue Xu, and Yujin Tang. Learning to orchestrate agents in natural language with the conductor. In International Conference on Learning Representations, volume 2026, pages 135686–135724, 2026.

[49] Jinglue Xu, Qi Sun, Peter Schwendeman, Stefan Nielsen, Edoardo Cetin, and Yujin Tang. Trinity: An evolved LLM coordinator. In International Conference on Learning Representations, volume 2026, pages 70195–70224, 2026.

[50] Xinchen Liu, Hang Zhou, Yingjie Zong, Yuchuan Tian, Liuyang Song, Shuo Zhang, Yulong Li, Wei He, Mengyu Zheng, Runke Liu, et al. Agentic routing: The harness-native data flywheel. arXiv preprint arXiv:2607.11399, 2026.

[51] Tianyu Fu, Yi Ge, Yichen You, Enshu Liu, Zhihang Yuan, Guohao Dai, Shengen Yan, Huazhong Yang, and Yu Wang. R2R: Eficiently navigating divergent reasoning paths with small-large model token routing. In Advances in Neural Information Processing Systems, volume 38, pages 124108–124145, 2026.

[52] Shunfan Zhou, Jingqi Long, Yiling Zhang, and Zhiheng Du. Fusion: Allocating intelligence across heterogeneous models—A new scaling dimension for AI systems. Whitepaper, Rev. 0.1, Fumo Lab, 2026. URL https://fumolab.ai/downloads/FUMO\_Whitepaper.pdf.

[53] John R Rice. The algorithm selection problem. In Advances in computers, volume 15, pages 65–118. Academic Press, 1976.

[54] Carla P Gomes and Bart Selman. Algorithm portfolios. Artificial Intelligence, 126(1-2):43–62, 2001.

[55] Lars Kotthof. Algorithm selection for combinatorial search problems: A survey. In Data mining and constraint programming: Foundations of a cross-disciplinary approach, pages 149–190. Springer, 2016.

[56] Lin Xu, Frank Hutter, Holger Hoos, and Kevin Leyton-Brown. Evaluating component solver contributions to portfolio-based algorithm selectors. In International conference on theory and applications ofsatisfiability testing, pages 228–241, 2012.

[57] Alexandre Fréchette, Lars Kotthof, Tomasz Michalak, Talal Rahwan, Holger Hoos, and Kevin Leyton-Brown. Using the Shapley value to analyze algorithm portfolios. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 30, 2016.

[58] Abraham Charnes and William W Cooper. Programming with linear fractional functionals. Naval Research logistics quarterly, 9(3-4):181–186, 1962.

[59] Howard Raifa and Robert Schlaifer. Applied Statistical Decision Theory. Division of Research, Graduate School of Business Administration, Harvard University, 1961.

[60] Ronald A Howard. Information value theory. IEEE Transactions on Systems Science and Cybernetics, 2(1): 22–26, 1966.

[61] Lingjiao Chen, Jared Davis, Boris Hanin, Peter Bailis, Ion Stoica, Matei Zaharia, and James Zou. Are more LLM calls all you need? towards the scaling properties of compound AI systems. In Advances in Neural Information Processing Systems, volume 37, pages 45767–45790, 2024.

[62] Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling LLM test-time compute optimally can be more efective than scaling parameters for reasoning. In International Conference on Learning Representations, volume 2025, pages 10131–10165, 2025.

[63] Benedikt Stroebl, Sayash Kapoor, and Arvind Narayanan. The limits of inference scaling through resampling. In International Conference on Learning Representations, volume 2026, pages 92601–92633, 2026.

[64] Xinyun Chen, Maxwell Lin, Nathanael Schärli, and Denny Zhou. Teaching large language models to self-debug. In The Twelfth International Conference on Learning Representations, 2024.

[65] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems, volume 36, pages 8634–8652, 2023.

[66] Theo X. Olausson, Jeevana Priya Inala, Chenglong Wang, Jianfeng Gao, and Armando Solar-Lezama. Is self-repair a silver bullet for code generation? In The Twelfth International Conference on Learning Representations, 2024.

[67] Wassily Hoefding. Probability inequalities for sums of bounded random variables. Journal of the American Statistical Association, 58(301):13–30, 1963. doi: 10.1080/01621459.1963.10500830.

[68] Andreas Maurer and Massimiliano Pontil. Empirical Bernstein bounds and sample variance penalization. In Proceedings of the 22nd Annual Conference on Learning Theory (COLT), 2009.

[69] Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, et al. � : a vision-languageaction model with open-world generalization. In Proceedings of the 9th Conference on Robot Learning, volume 305, pages 17–40, 2025.

[70] Fuhao Li, Wenxuan Song, Han Zhao, Jingbo Wang, Pengxiang Ding, Donglin Wang, Long Zeng, and Haoang Li. Spatial Forcing: Implicit spatial representation alignment for vision-language-action model. In International Conference on Learning Representations, 2026.

[71] Jinliang Zheng, Jianxiong Li, Zhihao Wang, Dongxiu Liu, Xirui Kang, et al. X-VLA: Soft-prompted transformer as scalable cross-embodiment vision-language-action model. In International Conference on Learning Representations, 2026.

## A. Proofs and Formal Results

This appendix proves the results of Section 2 and Corollary 6, and states, with short proofs, the formal results behind Sections 4.4 and 5.2. Except in the proofs of Propositions 3 and 4, where � is the input as in Section 2, we write � for the router’s observed state � and X for its finite set of values.

## A.1. Proof of Lemma 1

Proof. Write $V = V _ { g } , D = D _ { g } , \bar { \nu } = \mathbb { E } [ Y _ { B } ]$ , and $\bar { c } = \mathbb { E } [ C _ { B } ]$ . The definitions give $\mathbb { E } [ V ] = \bar { \nu } + Q$ and $\mathbb { E } [ D ] = \bar { c } - S , s 0$

$$
\eta - \eta _ { B } = \frac { \bar { \upsilon } + Q } { \bar { c } - S } - \frac { \bar { \upsilon } } { \bar { c } } = \frac { Q + ( \bar { \upsilon } / \bar { c } ) S } { \bar { c } - S } ,
$$

whose denominator $\mathbb { E } [ D ]$ is positive. Substituting $\bar { \nu } = \eta _ { B } \bar { c }$ gives, for every �,

$$
Q + \eta _ { B } { \cal S } = \mathbb { E } \bigl [ V - \eta _ { B } D \bigr ] , \qquad \mathbb { E } \bigl [ V - \lambda D \bigr ] = \mathbb { E } \bigl [ D \bigr ] \bigl ( \eta - \lambda \bigr ) .\tag{4}
$$

No independence or stationarity is needed; the quality condition follows separately from $\mathbb { E } [ V ] = \bar { \nu } + Q$ For a stationary ergodic sequence of rounds, the long-run ratio $\textstyle \sum _ { n < N } V _ { n } / \sum _ { n < N } D _ { n }$ converges almost surely to $\mathbb { E } [ V ] / \mathbb { E } [ D ]$ , a ratio of aggregates rather than the average of $V _ { n } / D _ { n }$ □

## A.2. Proof of Theorem 2

Throughout, � ranges over a finite set X of observable states with router-independent probabilities $\pi ( x ) > 0 ;$ ; each state ofers a finite menu $\mathcal { U } ( x )$ of candidates with expected value $\nu ( x , u )$ and expected cost $c ( x , u ) > 0 ;$ ; a router � chooses � with probability $g ( u \mid x )$ . Let G be the set of routers, a product of simplices and hence a compact polytope, and put $c _ { \mathrm { m i n } } = \mathrm { m i n } _ { x , u } c ( x , u ) > 0$ . For $g \in { \mathcal { G } }$ ,

$$
\mathbb { E } _ { g } [ V ] = \sum _ { x } \pi ( x ) \sum _ { u \in \mathcal { U } ( x ) } g ( u \mid x ) v ( x , u ) , \qquad \mathbb { E } _ { g } [ D ] = \sum _ { x } \pi ( x ) \sum _ { u \in \mathcal { U } ( x ) } g ( u \mid x ) c ( x , u ) \ \ge \ c _ { \operatorname* { m i n } } ,
$$

$\eta ( g ) = \mathbb { E } _ { g } [ V ] / \mathbb { E } _ { g } [ D ]$ , and for every real �

$$
\mathbb { E } _ { g } \left[ V - \lambda D \right] = \mathbb { E } _ { g } \left[ D \right] \left( \eta ( g ) - \lambda \right) .\tag{5}
$$

A router is deterministic if every $g ( \cdot \mid x )$ is a point mass. The reference policy is admissible if it is an element $g _ { B } \in { \mathcal { G } } ;$ ; then $\eta ( g _ { B } ) = \eta _ { B }$ and $\mathbb { E } _ { g _ { B } } [ V ] = { \bar { \nu } }$ , so the reference satisfies the floor � $[ V ] \geq \bar { \nu } - \epsilon$ for every $\epsilon \geq 0$

Proof. Two preliminary facts. (F1) State-wise separation: for any real numbers $\phi ( x , u )$ , the linear functional $\begin{array} { r } { g \mapsto \sum _ { x } \pi ( x ) \sum _ { u } g ( u \mid x ) \phi ( x , u ) } \end{array}$ on $\mathcal { G }$ is maximized exactly by the routers that, in every state �, put all their mass on arg max ${ } _ { \iota } \phi ( x , u )$ , and its maximum is $\textstyle \sum _ { x } \pi ( x )$ max �(�, �). Indeed, for each � the inner sum is a convex combination of the numbers $\phi ( x , u )$ , hence at most $\mathrm { m a x } _ { u } \phi ( x , u )$ with equality if and only if $g ( \cdot \mid x )$ is supported on the maximizers; and $\pi ( x ) > 0$ . In particular a deterministic maximizer exists. (F2) Sign of the Dinkelbach function: applying (F1) with $\phi = \nu - \lambda c ,$

$$
\Phi ( \lambda ) = \sum _ { x } \pi ( x ) \operatorname* { m a x } _ { u \in \mathcal { U } ( x ) } \left[ \nu ( x , u ) - \lambda c ( x , u ) \right] = \operatorname* { m a x } _ { g \in \mathcal { G } } \mathbb { E } _ { g } [ V - \lambda D ] ,\tag{6}
$$

so by (5) and $\mathbb { E } _ { g } [ D ] > 0 , \Phi ( \lambda ) \ge 0$ if and only if some router has $\eta ( g ) \geq \lambda$ , and $\Phi ( \lambda ) > 0$ if and only if some router has $\eta ( g ) > \lambda$

Part (i). Each $\phi _ { x } ( \lambda ) = \mathrm { m a x } _ { u } [ \nu ( x , u ) - \lambda c ( x , u ) ]$ is the maximum of finitely many afine functions of � with $\mathrm { s l o p e s } - c ( x , u ) < 0$ . It is therefore convex, piecewise linear and continuous, and strictly decreasing:

if $\lambda \ < \ \lambda ^ { \prime }$ and $u ^ { \prime }$ attains $\phi _ { x } ( \lambda ^ { \prime } )$ , then $\begin{array} { r } { \phi _ { x } ( \lambda ) \ \ge \ \nu ( x , u ^ { \prime } ) - \lambda c ( x , u ^ { \prime } ) \ > \ \nu ( x , u ^ { \prime } ) - \lambda ^ { \prime } c ( x , u ^ { \prime } ) \ = \ \phi _ { x } ( \lambda ^ { \prime } ) } \end{array}$ Moreover $\phi _ { x } ( \lambda ) $ +∞ as $\lambda \to - \infty$ , and $\begin{array} { r } { \phi _ { x } ( \lambda ) \leq \operatorname* { m a x } _ { u } \nu ( x , u ) - \lambda c _ { \operatorname* { m i n } } \to - \infty } \end{array}$ as $\lambda \to + \infty$ . The positive combination $\begin{array} { r } { \Phi = \sum _ { x } \pi ( x ) \phi _ { x } } \end{array}$ inherits all these properties and has a unique root $\eta ^ { \star }$ . By (F2), $\Phi ( \eta ^ { \star } ) = 0$ yields a router with $\eta ( g ) \geq \eta ^ { \star }$ and forbids $\eta ( g ) > \eta ^ { \star } ;$ hence $\eta ^ { \star } = \mathrm { m a x } _ { g } \eta ( g )$ , and the maximum is attained. A router $g$ is eficiency-optimal if and only if $\eta ( g ) = \eta ^ { \star }$ , if and only $\mathrm { i f } \mathbb { E } _ { g } [ V - \eta ^ { \star } D ] = 0 = \Phi ( \eta ^ { \star } )$ by (5), if and only if $g$ maximizes the $\eta ^ { \star }$ -objective over $\mathcal { G } _ { : }$ , if and only if, by (F1), in every state it uses only candidates maximizing $\nu ( x , u ) - \eta ^ { \star } c ( x , u )$

Part (ii): existence. Let $\mathcal { G } _ { \epsilon } = \{ g \in \mathcal { G } \colon \mathbb { E } _ { g } [ V ] \geq \bar { \nu } - \epsilon \}$ , a compact polytope that contains $g _ { B }$ because $\epsilon \geq 0$ . Since $\eta$ is continuous on G (its denominator is at least $c _ { \mathrm { m i n } } )$ , it attains its maximum $\eta _ { \epsilon } ^ { \star }$ on $G _ { \epsilon }$

Part (ii): the multiplier. Define $\Phi _ { \epsilon } ( \lambda ) = \operatorname* { m a x } _ { g \in { \mathcal { G } } _ { \epsilon } } \mathbb { E } _ { g } [ V - \lambda D ]$ . As a maximum over a compact set of afine functions of � with slopes $\begin{array} { r } { - \mathbb { E } _ { g } [ D ] \le - c _ { \mathrm { m i n } } , } \end{array}$ it is convex, continuous and strictly decreasing, tends to ±∞ as $\lambda \to \mp \infty _ { : }$ , and has a unique root. By $( 5 ) , \Phi _ { \epsilon } ( \lambda ) \geq 0$ if and only if some $g \in { \mathcal { G } } _ { \epsilon }$ has $\eta ( g ) \geq \lambda$ , so the root is $\eta _ { \epsilon } ^ { \star }$ . Consequently a router is constrained-optimal if and only if it solves

$$
\operatorname* { m a x } _ { g \in \mathcal { G } } f ( g ) : = \mathbb { E } _ { g } [ V - \eta _ { \epsilon } ^ { \star } D ] \quad \mathrm { s u b j e c t ~ t o } \quad h ( g ) : = \mathbb { E } _ { g } [ V ] - ( \bar { \nu } - \epsilon ) \geq 0 ,\tag{7}
$$

whose optimal value is $\Phi _ { \epsilon } ( \eta _ { \epsilon } ^ { \star } ) = 0$ . Both � and ℎ are afine in $g$ and $\mathcal { G }$ is a polytope, so (7) is a linear program; it is feasible $\left( g _ { B } \right)$ and bounded. By linear-programming strong duality applied to the single constraint $h \geq 0$ , there is a multiplier $\kappa \geq 0$ such that

$$
\operatorname* { m a x } _ { g \in { \mathcal { G } } } \left[ f ( g ) + \kappa h ( g ) \right] = 0 .
$$

If $g$ is optimal for $( 7 )$ , then $h ( g ) \geq 0$ and $f ( g ) = 0 , \ : \mathrm { s o } \ 0 \le f ( g ) + \kappa h ( g ) \le 0 ;$ : hence $\kappa h ( g ) = 0$ and $g$ maximizes $f + \kappa h$ over $\mathcal { G }$ . Now $f ( g ) + \kappa h ( g ) = \mathbb { E } _ { g } [ ( 1 + \kappa ) V - \eta _ { \epsilon } ^ { \star } D ] + \kappa ( \epsilon - \bar { \nu } )$ , so by (F1) every constrained-optimal router uses, in every state, only candidates maximizing $( 1 + \kappa ) \nu ( x , u ) - \eta _ { \epsilon } ^ { \star } c ( x , u )$ 2 that is, $\nu ( x , u ) - \lambda _ { \epsilon } c ( x , u )$ with $\lambda _ { \epsilon } = \eta _ { \epsilon } ^ { \star } / ( 1 + \kappa )$ , and so maximizes $J _ { \lambda _ { \epsilon } } \mathrm { ~ . ~ }$ ; and $\kappa \left( { \mathbb E } _ { g } [ V ] - { \bar { \nu } } + \epsilon \right) = 0 :$ if $\kappa > 0$ the floor binds at every optimal router. The value $\kappa = 0$ is a valid multiplier if and only if some eficiency-optimal router of part (i) satisfies the floor, and then $\eta _ { \epsilon } ^ { \star } = \eta ^ { \star }$ □

## A.3. Proof of Proposition 3

Proof. Assume each $W _ { u }$ is integrable, and write $S _ { 1 } = \sigma _ { 1 } ( x )$ and $S _ { 2 } ~ = ~ \sigma _ { 2 } ( x )$ . The menu $\mathcal { U } ( S _ { 1 } )$ is finite and available at every finer state. By the tower property, ${ \mathbb E } [ W _ { u } \mid S _ { 1 } ] = { \mathbb E } \big [ { \mathbb E } [ W _ { u } \mid S _ { 2 } ] \mid S _ { 1 } \big ] \le$ $\mathbb { E } [ \operatorname* { m a x } _ { u ^ { \prime } } \mathbb { E } [ W _ { u ^ { \prime } } \ | \ S _ { 2 } ] \ | \ S _ { 1 } ]$ for every �; taking the maximum over � and then expectations gives $V ^ { \star } ( \sigma _ { 1 } ) \leq V ^ { \star } ( \sigma _ { 2 } )$ . Let � attain $\operatorname* { m a x } _ { u } \mathbb { E } [ W _ { u } \mid S _ { 1 } ]$ . The two sides are equal almost surely if and only if $\operatorname { \mathbb { E } } [ W _ { u _ { 1 } } \mid S _ { 2 } ] = \operatorname* { m a x } _ { u } \operatorname { \mathbb { E } } [ W _ { u } \mid S _ { 2 } ]$ almost surely given $S _ { 1 }$ , that is, if and only if one candidate is optimal at almost every finer state; conversely, if a candidate � is optimal at almost every $S _ { 2 }$ with $\sigma _ { 1 } = s _ { 1 }$ then $\mathbb { E } [ W _ { u } \mid S _ { 1 } ]$ attains the right side, so equality holds. The same argument with $\sigma _ { 2 }$ replaced by the identity gives $V ^ { \star } ( \sigma _ { 2 } ) \leq V ^ { \star } ( x )$ . Finally, $W _ { u } \le \operatorname* { m a x } _ { u ^ { \prime } } W _ { u ^ { \prime } }$ for every $u ,$ so ma $\operatorname { \mathbb { x } } _ { u } \operatorname { \mathbb { E } } [ W _ { u } \mid x ] \leq \operatorname { \mathbb { E } } [ \operatorname* { m a x } _ { u } W _ { u } \mid x ]$ for every $x ,$ and taking expectations gives the last inequality; the gap is the expectation of this nonnegative diference. It vanishes if and only if the diference is zero for almost every $x . \ \operatorname { I f } u ( x )$ attains max $\mathbb { E } [ W _ { u } \mid x ]$ , the diference is �[max<sub>�</sub> $W _ { u } - W _ { u ( x ) } \mid x ]$ , the mean of a nonnegative variable, which is zero if and only if $W _ { u ( x ) } = \operatorname* { m a x } _ { u } W _ { u }$ almost surely given $x ;$ and if some candidate attains the maximum almost surely given �, it also attains $\operatorname* { m a x } _ { u } \mathbb { E } [ W _ { u } \mid x ]$ □

## A.4. Proof of Proposition 4

Proof. (a) Fix � and a candidate $u ,$ and let $\mu \ = \ \mathbb { E } [ V _ { u } \ \mid \ x ] \ \in \ [ 0 , 1 ]$ . For a score $m \in \ [ 0 , 1 ]$ , with $0 \log 0 = 0$ , the expected cross-entropy is −� log $m - ( 1 - \mu ) \log ( 1 - m )$ , a strictly convex function of � minimized at $m = \mu ,$ , and the expected squared error is Var $( V _ { u } \mid x ) + ( \mu - m ) ^ { 2 }$ , also minimized at $m = \mu$ . Hence $m _ { u } ( x ) = \mathbb { E } [ V _ { u } \mid x ]$ , and routing to arg $\mathrm { m a x } _ { u } m _ { u } ( x )$ earns m $\mathsf { u x } _ { u } \mathbb { E } [ V _ { u } \mid x ]$ at every �, whose expectation is $V ^ { \star } ( x )$ at $\lambda = 0$ . (b) Since $\textstyle { \frac { 1 } { 2 } } < b < a < 1$ , candidate � beats � exactly when $V _ { B } = 1$ an event of probability �, and loses otherwise; no ties occur. So $\begin{array} { r } { p _ { B } = b > \frac { 1 } { 2 } > 1 - b = p _ { A } } \end{array}$ , and the winner-label router plays �, whose expected value � is below �. The loss $a - b$ approaches $\textstyle { \frac { 1 } { 2 } }$ as $b \downarrow \downarrow \scriptscriptstyle { \frac { 1 } { 2 } }$ and $a \uparrow 1$ . In the tie-break example, the first candidate wins only when it succeeds and the second fails, with probability $0 . 9 \times 0 . 5 = 0 . 4 5$ □

## A.5. Proof of Proposition 5

Setting. Let � be the deployment law of a task. A task has a state � in a finite set X, a menu $\mathcal { U } ( x )$ a realized value $V _ { u } \in [ 0 , 1 ]$ and cost $D _ { u } > 0$ for every $u \in \mathcal { U } ( x )$ , and the reference value $Y _ { B } \in [ 0 , 1 ]$ and cost $C _ { B } . \mathrm { { A } }$ router �, possibly randomized, is scored by its expected delivered value given the task, $\begin{array} { r } { V _ { g } = \sum _ { u } g ( u \mid x ) V _ { u } , } \end{array}$ , which is $V _ { g ( x ) }$ for a deterministic router, and $Q _ { g } = \mathbb { E } _ { P } [ V _ { g } - Y _ { B } ]$ is its quality change. The reference is admissible: it is a router $g _ { B }$ with $\mathbb { E } _ { P } [ V _ { g _ { B } } ] = \mathbb { E } _ { P } [ Y _ { B } ]$ , so $Q _ { g _ { B } } = 0$ . Two samples are independent: A is arbitrary fitting data, and $\mathcal { B } = ( T _ { 1 } , \ldots , T _ { n } )$ consists of � i.i.d. tasks from $P ,$ each recording $x _ { i } , Y _ { B , i }$ and $V _ { u , i }$ for every candidate used below, as in matched replay.

From A we fit $\hat { \pi } ( x ) > 0 , \hat { \nu } ( x , u ) \in [ 0 , 1 ]$ and $\hat { c } ( x , u ) > 0$ . For a router � write $\begin{array} { r } { \hat { \mathbb E } _ { g } [ V ] = \sum _ { x } \hat { \pi } ( x ) \sum _ { u } g ( u \mid } \end{array}$ $x ) \hat { \upsilon } ( x , u )$ , define $\hat { \mathbb { E } } _ { g } [ D ]$ likewise, and put $\hat { \eta } ( g ) ~ = ~ \hat { \mathbb { E } } _ { g } [ V ] / \hat { \mathbb { E } } _ { g } [ D ]$ and $\hat { J } _ { \lambda } ( g ) ~ = ~ \hat { \mathbb { E } } _ { g } [ V ] - \lambda \hat { \mathbb { E } } _ { g } [ D ]$ . Let $\begin{array} { r } { \hat { \phi } _ { x } ( \lambda ) \ = \ \operatorname* { m a x } _ { u \in \mathcal { U } ( x ) } \left[ \hat { \nu } ( x , u ) - \lambda \hat { c } ( x , u ) \right] } \end{array}$ and $\begin{array} { r } { \hat { \Phi } \ = \ \sum _ { x } \hat { \pi } ( x ) \hat { \phi } _ { x } } \end{array}$ . Theorem $2 ( \mathrm { i } )$ , applied to the fitted primitives, gives the root $\hat { \eta } ^ { \star } = \mathrm { m a x } _ { g } \hat { \eta } ( g )$ , and we assume $\hat { \mathbb { E } } _ { g _ { B } } [ V ] > 0 .$ , so that $\hat { \eta } ^ { \star } > 0$ . A breakpoint of $\hat { \phi } _ { x }$ is a point at which $\ddot { \phi } _ { x }$ is not diferentiable. Let $\lambda _ { 1 } < \cdots < \lambda _ { K - 1 }$ be the breakpoints of all $\hat { \phi } _ { x }$ that lie in $( 0 , \hat { \eta } ^ { \star } )$ , and set $\lambda _ { 0 } = 0 , \lambda _ { K } = \hat { \eta } ^ { \star }$ and $I _ { k } = \left( \lambda _ { k - 1 } , \lambda _ { k } \right)$ . Fix a strict order on each $\mathcal { U } ( x )$ (the tie-break), and let $g _ { k } ( x )$ be the first maximizer of $\hat { \nu } ( x , u ) - \lambda \hat { c } ( x , u )$ in that order, for $\lambda \in I _ { k }$ . The sequence $g _ { 1 } , \ldots , g _ { K }$ is the price path. A mixture $w g _ { k } + ( 1 - w ) g _ { k + 1 }$ is the router that, in each state, plays $g _ { k } ( x )$ with probability � and $g _ { k + 1 } ( x )$ otherwise.

Proposition 5 (full statement). (a) Path. (1) Each $g _ { k }$ is well defined, is the same for every $\lambda \in I _ { k }$ , and maximizes $\hat { J } _ { \lambda }$ over all routers for every $\lambda \in [ \lambda _ { k - 1 } , \lambda _ { k } ]$ . Moreover $\begin{array} { r } { K \leq 1 + \sum _ { x } ( | \mathcal { U } ( x ) | - 1 ) } \end{array}$ . (2) For $k < l ,$ $\hat { \mathbb { E } } _ { g _ { k } } [ V ] > \hat { \mathbb { E } } _ { g _ { l } } [ V ] , \hat { \mathbb { E } } _ { g _ { k } } [ D ] > \hat { \mathbb { E } } _ { g _ { l } } [ D ]$ and $\hat { \eta } ( g _ { k } ) < \hat { \eta } ( g _ { l } ) ;$ in particular $g _ { 1 } , \ldots , g _ { K }$ are distinct.

(b) Certification. Let $\alpha \in ( 0 , 1 )$ and $\epsilon \geq 0 .$ , and let $p _ { 1 } , \ldots , p _ { K }$ be functions $o f \left( { \mathcal { R } } , { \mathcal { B } } \right)$ such that $^ { \mathfrak { d } } ( p _ { k } \le$ $\alpha \mid { \mathcal { A } } ) \leq \alpha$ whenever $Q _ { g _ { k } } < - \epsilon$ . Let $\hat { k } = \operatorname* { m a x } \{ k \in \{ 0 , \ldots , K \} : p _ { j } \leq \alpha f o r \ a l l \ j \leq k \}$ , and let $\hat { g } = g _ { \hat { k } } i f \hat { k } \geq 1$ and $\hat { g } = g _ { B }$ otherwise. Then

$$
\begin{array} { r l } { \mathbb { P } \big ( Q _ { \hat { g } } < - \epsilon \mid \mathcal { A } \big ) \leq \alpha } & { a l m o s t s u r e l y , h e n c e \quad \mathbb { P } \big ( Q _ { \hat { g } } < - \epsilon \big ) \leq \alpha . } \end{array}
$$

With $\begin{array} { r } { \hat { Q } _ { k } = n ^ { - 1 } \sum _ { i = 1 } ^ { n } ( V _ { g _ { k } , i } - Y _ { B , i } ) } \end{array}$ , the Hoefding �-value $p _ { k } = \exp \{ - n ( \hat { Q } _ { k } + \epsilon ) _ { + } ^ { 2 } / 2 \}$ satisfies the requirement.   
The conclusion holds for any finite family of routers and any order that are fixed given A.

Proof. Part (a)(1). Fix �. As a maximum of $| { \mathcal { U } } ( x ) |$ afine functions with slopes $- \hat { c } ( x , u ) < 0 , \hat { \phi } _ { x }$ is convex and piecewise linear; its slope takes values in $\{ - \hat { c } ( x , u ) : u \in \mathcal { U } ( x ) \}$ and strictly increases at each breakpoint, so $\hat { \phi } _ { x }$ has at most $| \mathcal { U } ( x ) | - 1$ breakpoints. A breakpoint shared by several states is counted once, so $\begin{array} { r } { K - 1 \le \sum _ { x } ( | \mathcal { U } ( x ) | - 1 ) } \end{array}$ . On $I _ { k }$ no $\hat { \phi } _ { x }$ has a breakpoint, so $\hat { \phi } _ { x } ( \lambda ) = a _ { x } - b _ { x } \lambda$ there. For each $\boldsymbol { u } , \hat { \phi } _ { x } ( \lambda ) - [ \hat { \nu } ( x , \boldsymbol { u } ) - \lambda \hat { c } ( x , \boldsymbol { u } ) ]$ is afine and nonnegative on $I _ { k }$ , so if it vanishes at one point of the open interval $I _ { k }$ , it vanishes on all of it. Hence the maximizer set, and its first element $g _ { k } ( x )$ are the same for every $\lambda \in I _ { k } ,$ , and every maximizer has $( \hat { \nu } ( x , u ) , \hat { c } ( x , u ) ) = ( a _ { x } , b _ { x } )$ . By fact (F1) of

Appendix $\mathsf { A } . 2 , g _ { k }$ maximizes $\hat { J } _ { \lambda }$ on $I _ { k } ;$ since $\hat { J } _ { \lambda } ( g _ { k } )$ and $\hat { \Phi } ( \lambda )$ are continuous in � and agree on $I _ { k } ,$ they agree on its closure.

Part (a)(2). Fix $1 \leq k \leq K - 1$ . Let $\chi _ { k }$ be the set of states for which $\lambda _ { k }$ is a breakpoint; it is nonempty by the definition of $\lambda _ { k } .$ . For $x \notin \chi _ { k } , \hat { \phi } _ { x }$ is afine on $I _ { k } \cup \{ \lambda _ { k } \} \cup I _ { k + 1 } ,$ so by $\mathrm { ( a ) ( 1 ) }$ the candidates $g _ { k } ( x )$ and $g _ { k + 1 } ( x )$ have the same pair $( \hat { \nu } , \hat { c } )$ . For $x \in { \mathcal { X } } _ { k } , { \hat { c } } ( x , g _ { k } ( x ) ) = - { \hat { \phi } } _ { x } ^ { \prime } ( \lambda _ { k } - ) > - { \hat { \phi } } _ { x } ^ { \prime } ( \lambda _ { k } + ) = { \hat { c } } ( x , g _ { k + 1 } ( x ) )$ and both candidates attain $\hat { \phi } _ { x } ( \lambda _ { k } )$ by continuity, so ${ \hat { \nu } } ( x , g _ { k + 1 } ( x ) ) - { \hat { \nu } } ( x , g _ { k } ( x ) ) = \lambda _ { k } [ { \hat { c } } ( x , g _ { k + 1 } ( x ) ) - $ $\hat { c } ( x , g _ { k } ( x ) ) ]$ . Write $( \nu _ { j } , d _ { j } ) = ( \hat { \mathbb { E } } _ { g _ { j } } [ V ] , \hat { \mathbb { E } } _ { g _ { j } } [ D ] )$ ). Because $\hat { \pi } > 0$ and $\lambda _ { k } > 0$

$$
d _ { k + 1 } - d _ { k } = \sum _ { x \in \mathcal { X } _ { k } } \hat { \pi } ( x ) \left[ \hat { c } ( x , g _ { k + 1 } ( x ) ) - \hat { c } ( x , g _ { k } ( x ) ) \right] < 0 , \qquad \nu _ { k + 1 } - \nu _ { k } = \lambda _ { k } ( d _ { k + 1 } - d _ { k } ) < 0 .
$$

Chaining consecutive steps gives $\upsilon _ { k } > \upsilon _ { l }$ and $d _ { k } > d _ { l }$ for $k < l$ . For eficiency, pick $\lambda ^ { \prime } \in I _ { l }$ . Optimality of $g _ { l }$ at $\lambda ^ { \prime }$ gives $\nu _ { k } - \nu _ { l } \leq \lambda ^ { \prime } ( d _ { k } - d _ { l } )$ , hence

$$
\nu _ { l } d _ { k } - \nu _ { k } d _ { l } = \nu _ { l } ( d _ { k } - d _ { l } ) - d _ { l } ( \nu _ { k } - \nu _ { l } ) \geq ( d _ { k } - d _ { l } ) ( \nu _ { l } - \lambda ^ { \prime } d _ { l } ) = \left( d _ { k } - d _ { l } \right) \hat { \Phi } ( \lambda ^ { \prime } ) > 0 ,
$$

because $d _ { k } > d _ { l }$ and $\hat { \Phi }$ is strictly decreasing with root $\hat { \eta } ^ { \star } > \lambda ^ { \prime }$ (Theorem 2(i)). Dividing by $d _ { k } d _ { l } > 0$ gives $\hat { \eta } ( g _ { l } ) > \hat { \eta } ( g _ { k } )$ . Distinct fitted costs make the routers distinct.

Part (b). Condition on $\mathcal { A } .$ . Then $g _ { 1 } , \ldots , g _ { K } .$ , their order, and the numbers $Q _ { g _ { k } }$ are fixed. Let $k _ { 0 } =$ min $\{ k : Q _ { g _ { k } } < - \epsilon \}$ , with $k _ { 0 } = \infty$ if no path router violates the floor. The reference cannot violate it, since $Q _ { g _ { B } } = 0 \ge - \epsilon ;$ this is the only use of $\epsilon \geq 0$ . Suppose $Q _ { \hat { g } } < - \epsilon$ . Then $\hat { k } \geq 1$ and � violates the $g _ { \hat { k } }$ floor, so $k _ { 0 } \le \hat { k } < \infty$ . By the definition of $\hat { k } ,$ every $p _ { j }$ with $j \le \hat { k }$ is at most �, and in particular $p _ { k _ { 0 } } \leq \alpha .$ Therefore

$$
\begin{array} { r } { \mathbb { P } \big ( Q _ { \hat { g } } < - \epsilon \mid \mathcal { A } \big ) \le { \bf 1 } \{ k _ { 0 } < \infty \} \mathbb { P } \big ( p _ { k _ { 0 } } \le \alpha \mid \mathcal { A } \big ) \le \alpha , } \end{array}
$$

by the requirement on ${ p } _ { k _ { 0 } }$ , because $Q _ { g _ { k _ { 0 } } } < - \epsilon$ . Taking expectations over A gives the unconditional bound. No condition is placed on the dependence among $p _ { 1 } , \ldots , p _ { K } .$ , which share the sample B.

For the Hoefding �-value, condition on A. Since B is independent of A and $g _ { k }$ is fixed, the variables $Z _ { i } = V _ { g _ { k } , i } - Y _ { B , i }$ <sub>�</sub> are i.i.d., take values in [−1, 1], and have mean $Q _ { g _ { k } }$ . Put $t _ { \alpha } = \sqrt { 2 \log ( 1 / \alpha ) / n } > 0$ . Then $p _ { k } \leq \alpha$ if and only if $\hat { Q } _ { k } + \epsilon \ge t _ { \alpha } . \mathrm { ~ I f ~ } Q _ { g _ { k } } < - \epsilon .$ , Hoefding’s inequality for a range of length two [67] gives

$$
\begin{array} { r } { \mathbb { P } \big ( p _ { k } \le \alpha \mid \mathcal { A } \big ) \le \mathbb { P } \big ( \hat { Q } _ { k } - Q _ { g _ { k } } \ge t _ { \alpha } \mid \mathcal { A } \big ) \le \exp ( - n t _ { \alpha } ^ { 2 } / 2 ) = \alpha . } \end{array}
$$

Other �-values, refined paths, and the certified price. Validity does not use the monotonicity in part (a); the order matters only for power, and part (a)(2) puts routers with high fitted value, and therefore a small risk of violation, first. Any other valid �-value can be used. The empirical-Bernstein bound of Maurer and Pontil [68, Theorem 4], applied to $( 1 - Z _ { i } ) / 2$ with $Z _ { i } = V _ { g _ { k } , i } - Y _ { B , i }$ , yields a valid �-value whose margin shrinks with the sample variance of $V - Y _ { B }$ . Because validity needs only a finite family in an order fixed by ${ \mathcal { A } } .$ , mixtures $w g _ { k } + ( 1 - w ) g _ { k + 1 }$ with decreasing � may be inserted between consecutive path routers; we insert $w \in \{ 0 . 9 , 0 . 8 , \ldots , 0 . 1 \}$ and, at each tolerance studied, the fitted floor optimum of Theorem 2(ii) computed on ${ \mathcal { A } } ,$ which on our menus mixes two neighbouring path routers. $\operatorname { I f } \hat { k } \geq 1$ , the certified router maximizes $\hat { J } _ { \lambda }$ at $\hat { \lambda } = \lambda _ { \hat { k } } .$ , the right end of its interval on the path; we call $\hat { \lambda }$ the certified price. It turns the choice of � into an output of the reference, the tolerance $\epsilon ,$ and the level �.

## A.6. Proof of Corollary 6

A candidate calls the reference in state � if it invokes the reference with positive probability there. For a candidate $u ,$ the set $\Lambda _ { x } ( u )$ of prices $\lambda \geq 0$ at which � maximizes $\nu ( x , \cdot ) - \lambda c ( x , \cdot )$ is a closed interval, possibly empty, because $\lambda \mapsto \nu ( x , u ) - \lambda c ( x , u ) - \phi _ { x } ( \lambda )$ is concave, continuous, and nonpositive, and $\Lambda _ { x } ( u )$ is the set where it vanishes. The admission price $\lambda ^ { B }$ is the largest point of the finite union of the sets $\Lambda _ { x } ( u )$ over states and reference-calling candidates (0 if the union is empty, +∞ if it is unbounded).

Proof. By Theorem 2(i), an eficiency-optimal router plays in every state only maximizers of $\upsilon ( x , \cdot ) -$ $\eta ^ { \star } c ( x , \cdot ) ;$ since $\eta ^ { \star } > \lambda ^ { B }$ , none of them calls the reference. By part (ii), a floor-optimal router maximizes $J _ { \lambda _ { \epsilon } }$ , so by part (i) it plays only maximizers of $\nu ( x , \cdot ) - \lambda _ { \epsilon } c ( x , \cdot )$ , none of which calls the reference when $\lambda _ { \epsilon } > \lambda ^ { B }$ . For the cascade � with primary �, let � also denote direct dispatch to it, with value $Y _ { P }$ and cost $C _ { P } \leq A$ . By Eq. (3),

$$
\nu ( x , u ) - \nu ( x , P ) = \mathbb { E } \big [ ( Y _ { B } - Y _ { P } ) \mathbf { 1 } _ { H = 0 } \mid x \big ] = : N , \qquad c ( x , u ) - c ( x , P ) = \mathbb { E } \big [ A - C _ { P } \mid x \big ] + r \ge r _ { f }
$$

with $r = \mathbb { E } [ C _ { B } \mathbf { 1 } _ { H = 0 } ~ | ~ x ] > 0$ . If � maximizes at $\lambda \geq 0 _ { i }$ , then $\nu ( x , u ) - \lambda c ( x , u ) \geq \nu ( x , P ) - \lambda c ( x , P )$ , so $\lambda \left[ c ( x , u ) - c ( x , P ) \right] \leq N$ . Hence $N \geq 0$ and $\lambda \leq N / r$ , the bound stated. It holds for every verifier and verifier cost, but it needs direct dispatch to � on the menu: with $Y _ { P } = 0 , A = C _ { P } = 1 , Y _ { B } = C _ { B } = 1$ $H = 0$ , and a single other candidate with cost 10 and value $^ { 0 , }$ the cascade maximizes at every price although $N / r = 1$ □

## A.7. Portfolio Comparisons

For a portfolio � and a candidate $m ,$ , let $S ^ { + } = S \cup \{ m \}$ , with delivered values $V _ { S } , V _ { S ^ { + } }$ and complete costs $D _ { S } , D _ { S ^ { + } }$ of positive mean on the same tasks. Lemma 1, with � as the reference, gives the paired gain

$$
G _ { m } ( S ) = \mathbb { E } [ V _ { S ^ { + } } - V _ { S } ] - \eta ( S ) \mathbb { E } [ D _ { S ^ { + } } - D _ { S } ] , \qquad \eta ( S ^ { + } ) - \eta ( S ) = \frac { G _ { m } ( S ) } { \mathbb { E } [ D _ { S ^ { + } } ] } ,\tag{8}
$$

so adding � improves eficiency if and only if $G _ { m } ( S ) > 0$ , whatever it changes in accepted quality, fallback composition, or verifier behaviour. Let $E _ { m }$ be the event that � supplies a suficient output, and let an exact selecting verifier pick a suficient member whenever one exists, else call the reference. Then � hits with probability $p ( S ) = \mathbb { P } ( \cup _ { i \in S } E _ { i } )$ , and the residual coverage of � is $\begin{array} { r } { r _ { m } ( S ) = \mathbb { P } ( E _ { m } \cap \bigcap _ { i \in S } E _ { i } ^ { c } ) = } \end{array}$ $p ( S ^ { + } ) - p ( S )$ , a joint property of the members’ failures that standalone success rates do not determine.

Proposition 7 (Residual coverage and its correction). Let � deliver mean value $\nu _ { h }$ on a hit and $\nu _ { \mathrm { m i s s } }$ on a miss, at primary cost $a ( S )$ plus a fallback cost � on a miss, and let $d _ { m } ( S ) = a ( S ^ { + } ) - a ( S ) . \ I f S ^ { + }$ has the same $\nu _ { h } , \nu _ { \mathrm { m i s s } } ,$ and $b ,$ then $G _ { m } ( S )$ equals the eficiency-aware residual coverage

$$
N _ { m } ( S ) = r _ { m } ( S ) \left[ \nu _ { h } - \nu _ { \mathrm { m i s s } } + \eta ( S ) b \right] - \eta ( S ) d _ { m } ( S ) .\tag{9}
$$

If instead $S ^ { + }$ shifts these conditional means by $\delta \nu _ { h } , \delta \nu _ { \mathrm { m i s s } } ,$ and $\delta b ,$ , then, with $p = p ( S )$ and $r = r _ { m } ( S ) _ { * }$

$$
G _ { m } ( S ) = N _ { m } ( S ) + \left( p + r \right) \delta \upsilon _ { h } + \left( 1 - p - r \right) \left[ \delta \upsilon _ { \mathrm { m i s s } } - \eta ( S ) \delta b \right] .\tag{10}
$$

Proof. $S ^ { + }$ hits with probability $p + r$ . By total expectation, $\begin{array} { r } { \mathsf { E } \left[ V _ { S ^ { + } } - V _ { S } \right] = r ( \nu _ { h } - \nu _ { \mathrm { m i s s } } ) + ( p + r ) \delta \nu _ { h } + } \end{array}$ $( 1 - p - r ) \delta \nu _ { \mathrm { m i s s } }$ and $\mathbb { E } [ D _ { S ^ { + } } - D _ { S } ] = d _ { m } ( S ) - r b + ( 1 - p - r ) \delta b$ . Substituting into Eq. (8) gives Eq. (10), and with no shifts $G _ { m } ( S ) = N _ { m } ( S )$ □

## A.8. Retries and the Value of Routing Information

Planning retries. A plan that retries after failures is a candidate, so Theorem 2 chooses among plans; a plan’s value is computed backward over post-failure histories. A history � records the attempts so far and their feedback. For a next attempt $u ,$ let $a ( \mathfrak { h } , u )$ be its incremental cost, including verification and its longer context, $p _ { \theta } ( \mathfrak { h } , u )$ its acceptance probability, and $\nu _ { \mathrm { a c c } } ( \mathfrak { h } , u )$ its accepted value, all conditional on ${ \mathfrak { h } } ,$ and let $J ^ { \mathrm { d i r e c t } } ( { \mathfrak { h } } ) = \mathbb { E } [ Y _ { B } - \lambda C _ { B } \mid { \mathfrak { h } } ]$ be the return of calling the reference now.

Proposition 8 (Planning on post-failure states). The optimal remaining return Π equals $J ^ { \mathrm { d i r e c t } }$ at the horizon and satisfies

$$
\Pi ( { \mathfrak { h } } ) = \operatorname* { m a x } \Bigl \{ J ^ { \mathrm { d i r e c t } } ( { \mathfrak { h } } ) , \ \operatorname* { m a x } _ { u } \bigl [ - \lambda a ( { \mathfrak { h } } , u ) + p _ { \theta } ( { \mathfrak { h } } , u ) v _ { \mathrm { a c c } } ( { \mathfrak { h } } , u ) + ( 1 - p _ { \theta } ( { \mathfrak { h } } , u ) ) \mathbb { E } [ \Pi ( { \mathfrak { h } } ^ { \prime } ) \ | \ { \mathfrak { h } } , u , H _ { \theta } = 0 ] \bigr ] \Bigr \} ,\tag{11}
$$

where ${ \mathfrak { h } } ^ { \prime }$ extends � by the rejected attempt. An attempt is optimal if and only if it attains the maximum, so its success probability enters only as conditioned on the history that reaches it.

Proof. Stopping returns $J ^ { \mathrm { d i r e c t } } ( { \mathfrak { h } } )$ . Attempting � costs $a ( \mathfrak { h } , u )$ ; with probability $p _ { \theta } ( \mathfrak { h } , u )$ the task ends with value $\nu _ { \mathrm { a c c } } ( \mathfrak { h } , u )$ , and otherwise it continues from ${ \mathfrak { h } } ^ { \prime } .$ , whose best continuation is worth $\Pi ( \mathfrak { h ^ { \prime } } )$ Since $J _ { \lambda }$ is linear in value and cost, attempting and then continuing optimally is worth the bracket, a randomized choice cannot beat the better branch, and backward induction from the horizon gives Eq. (11). □

Planning on marginal success rates replaces $p _ { \theta } ( \mathfrak { h } , u )$ by the unconditional rate of $u ,$ which overstates later attempts when failures are positively correlated.

Buying routing information. Let accepted outputs and the fallback preserve value, let the primary be suficient with probability �, and let $0 < a < b$ be the costs of the primary and of the fallback. Without information the router incurs the lower of $a + ( 1 - s ) b$ (attempt, then escalate on failure) and � (direct fallback). A signal � that moves the posterior to $s _ { Z }$ saves, before its price,

$$
\operatorname* { m i n } \{ a + ( 1 - s ) b , b \} - \mathbb { E } \operatorname* { m i n } \{ a + ( 1 - s _ { Z } ) b , b \} ,\tag{12}
$$

and its capture is this saving divided by the bound below.

Proposition 9 (The value of routing information). For every signal $Z ,$ the saving (12) is at most min $\{ s ( b - a ) , a ( 1 - s ) \}$ , which exact revelation of the suficiency event attains. A signal priced at � is therefore worth buying only if $k < a ( 1 - s ) \leq a \colon$ it avoids at most one unsuccessful attempt per task, whatever its accuracy.

Proof. Put $f ( t ) = \operatorname* { m i n } \{ a + ( 1 - t ) b , b \}$ and $h ( t ) = t a + ( 1 - t ) b$ . As a minimum of two afine functions, � is concave, and $f \geq h$ on $[ 0 , 1 ] ,$ since $f - h$ is the smaller of $a ( 1 - t )$ and $t ( b - a )$ . Since ℎ is afine and $\mathbb { E } [ s _ { Z } ] = s , h ( s ) = \mathbb { E } [ h ( s _ { Z } ) ] \le \mathbb { E } [ f ( s _ { Z } ) ]$ , so the saving $f ( s ) - \mathbb { E } [ f ( s _ { Z } ) ]$ is at most $f ( s ) - h ( s ) =$ min $\{ s ( b - a ) , a ( 1 - s ) \}$ . Exact revelation has $s _ { Z } \in \{ 0 , 1 \}$ , where $f = h _ { . }$ , so it attains the bound; this is the expected value of perfect information for two candidates and a binary state [59, 60]. □

## A.9. Router Memory

A record store refines the router’s state with past outcomes. In an idealized version, tasks fall into � cells that share a record; ACRouter’s embedding-keyed store [4] is a richer instance. Cells arrive independently with probabilities $\nu _ { j } > 0$ . At the start of a round, each cell is uninformed (�) or holds an informative (�) or erroneous (�) record. After the round, each record expires with probability $\delta _ { j }$ , and the arriving cell receives an informative write with probability $\omega _ { j }$ or an erroneous one with probability $\psi _ { j } ,$ replacing its record; write types are independent of the past. Put $r _ { j } = \omega _ { j } + \psi _ { j } > 0$ and $\alpha _ { j } = ( 1 - \delta _ { j } ) ( 1 - \nu _ { j } r _ { j } ) < 1$ . Let $\upsilon _ { j } ^ { s }$ and $c _ { j } ^ { s }$ be the router’s expected value and cost in cell � and record state $s \in \{ U , I , W \}$ , including the record update. Records are measured against the uninformed router, which writes at the same cost but routes as if uninformed and has eficiency $\begin{array} { r } { \eta _ { U } = \sum _ { j } \nu _ { j } \nu _ { j } ^ { U } / \sum _ { j } \nu _ { j } c _ { j } ^ { U } } \end{array}$ , by the gains $g _ { j } ^ { s } = \big ( \nu _ { j } ^ { s } - \nu _ { j } ^ { U } \big ) - \eta _ { U } \big ( c _ { j } ^ { s } - c _ { j } ^ { U } \big )$

Proposition 10 (Stationary memory gain). Let $\chi _ { j } = \nu _ { j } r _ { j } / ( 1 - \alpha _ { j } )$ be the stationary coverage of cell � and $e _ { j } = \psi _ { j } / r _ { j }$ its erroneous write fraction. The stationary eficiency $\eta _ { \infty }$ satisfies $\eta _ { \infty } \geq \eta _ { U }$ if and only if

$$
\sum _ { j } \nu _ { j } \chi _ { j } \big [ ( 1 - e _ { j } ) g _ { j } ^ { I } + e _ { j } g _ { j } ^ { W } \big ] ~ \geq ~ 0 .\tag{13}
$$

$I f g _ { j } ^ { I } > 0 > g _ { j } ^ { W }$ , cell � contributes positively if and only if $e _ { j } < g _ { j } ^ { I } / ( g _ { j } ^ { I } - g _ { j } ^ { W } )$

Proof. A record before round $n + 1$ was written in round � or survived it, so the probabilities of informative and erroneous records satisfy $i _ { j , n + 1 } = \nu _ { j } \omega _ { j } + \alpha _ { j } i _ { j , n }$ and $w _ { j , n + 1 } \ : = \ : \nu _ { j } \psi _ { j } \ : + \ : \alpha _ { j } w _ { j , n } ,$ with stationary values $i _ { j } = \chi _ { j } ( 1 - e _ { j } )$ and $w _ { j } = \chi _ { j } e _ { j }$ . Conditioning on the arriving cell, the stationary surplus per round is $\begin{array} { r } { \mathbb { E } [ V - \eta _ { U } \dot { D } ] = \sum _ { j } \nu _ { j } \big [ \upsilon _ { j } ^ { U } - \eta _ { U } c _ { j } ^ { \dot { U } } + i _ { j } g _ { i } ^ { I } + w _ { j } g _ { i } ^ { W } \big ] } \end{array}$ , whose uninformed terms sum to zero by the definition of $\eta _ { U }$ . The result is the left side of Eq. (13), and it equals $\bar { c } _ { \infty } ( \eta _ { \infty } - \eta _ { U } )$ for the stationary mean cost $\bar { c } _ { \infty } > 0$ . The cell condition follows from $( 1 - e _ { j } ) g _ { j } ^ { I } + e _ { j } g _ { j } ^ { W } = g _ { j } ^ { I } - e _ { j } ( g _ { j } ^ { I } - g _ { j } ^ { W } )$ □

## B. The Omni-Modal Trajectory Matrix: Details

## B.1. Benchmarks, Metrics, and Candidates

The 25 benchmarks form seven groups:

• text QA: MMLU-Pro [10], MMLU-Redux [11], SuperGPQA [12], GPQA [13], AA-LCR [14], AA-Omniscience [15], and the text subset of HLE [16];

• image-text QA: MMMU-Pro [17], MMMU [18], and VLM-ExecRouterBench [19];

• image generation: GenEval2 [20], Qwen-Image-Bench [21], LongT2I [22], OneIG [23], and WISE [24];

• video generation: VBench-2.0 [25]; speech generation: Seed-TTS English and Chinese [26];

• code and workplace agents: AA-Briefcase [14], AutomationBench [27], GDP.pdf [28], GDPval [29], SciCode [30], and Terminal-Bench [31];

• robot manipulation: RoboTwin 2.0 [32], 50 bimanual tasks in clean and randomized scenes.

The matrix holds 374,401 scored runs on 103,307 tasks from 452 candidate configurations, which differ in model, harness, and efort. Understanding benchmarks score a binary answer (AA-Omniscience uses plain accuracy). GenEval2 uses its oficial Soft-TIFA-GM score, Qwen-Image-Bench its oficial hierarchical total, LongT2I a learned alignment score, OneIG the mean of the sub-scores that apply to a single image, and WISE binary correctness. VBench-2.0 averages the dimensions that apply to each prompt, so it difers from the oficial total. Seed-TTS scores intelligibility as 1 − WER (English) or 1 − CER (Chinese), clipped to [0, 1]. AA-Briefcase scores the fraction of passed checks, Automation-Bench assertion credit, GDP.pdf whether every criterion passes, GDPval the rubric fraction, SciCode step pass on the original tests, Terminal-Bench the oficial verifier reward, and RoboTwin 2.0 binary episode success. Agent candidates run in their native harness at maximum efort; understanding candidates answer in one zero-shot request at high efort.

The candidates are: Claude Opus 5.5, GPT-6.1 Sol, DeepSeek V4.1 Flash, and, for long-context, knowledge, and agent tasks, GPT-6 Astra; the image generators FLUX 3, Seedream 5.0 Pro, MAI-Image 2.6, GPT Image 2.5, Qwen-Image 3 Pro, and Grok Imagine 2.0 (two further generators appear on a few GenEval2 prompts); the video generators MiniMax H3, Seedance 2.5, and Wan 3.0; the speech generators Seed Audio 1.0, Fish S2.1 Pro, and Gemini 3.8 Flash $\operatorname { T T S } ;$ and the policies $\pi _ { 0 . 5 }$ [69], GigaBrain, Spatial Forcing [70], X-VLA [71], X-WAM, PatchWAM, and ME-Dex, all co-trained on the 50 RoboTwin tasks. AA-LCR, GDP.pdf, and Terminal-Bench run every task three, five, and three times; each run is a separate input. Table 6 gives the per-benchmark potential gain behind Table 1.

Table 6 | Per-benchmark potential gain on the test split (Table 1). The best fixed candidate in hindsight is named when it difers from the fixed candidate chosen on training inputs.
<table><tr><td>Benchmark</td><td>Inputs</td><td>Fixed candidate</td><td>Fixed</td><td>Best fixed in hindsight</td><td>Oracle</td><td>Gap</td></tr><tr><td>Text QA</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MMLU-Pro</td><td>2,403</td><td>Opus 5.5</td><td>92.55</td><td>92.55</td><td>95.21</td><td>2.66</td></tr><tr><td>MMLU-Redux</td><td>1,080</td><td>Opus 5.5</td><td>96.67</td><td>96.67</td><td>98.06</td><td>1.39</td></tr><tr><td>SuperGPQA</td><td>5,300</td><td>Opus 5.5</td><td>78.26</td><td>78.26</td><td>83.66</td><td>5.40</td></tr><tr><td>GPQA (extended)</td><td>106</td><td>Opus 5.5</td><td>90.57</td><td>90.57</td><td>92.45</td><td>1.89</td></tr><tr><td>AA-LCR</td><td>42</td><td>Opus 5.5</td><td>97.62</td><td>97.62</td><td>100.00</td><td>2.38</td></tr><tr><td>AA-Omniscience</td><td>120</td><td>Opus 5.5</td><td>77.50</td><td>77.50</td><td>85.83</td><td>8.33</td></tr><tr><td>HLE (text)</td><td>431</td><td>Opus 5.5</td><td>58.00</td><td>58.00</td><td>64.50</td><td>6.50</td></tr><tr><td>Image-text QA</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MMMU-Pro</td><td>350</td><td>Opus 5.5</td><td>86.86</td><td>86.86</td><td>92.86</td><td>6.00</td></tr><tr><td>MMMU (dev/val)</td><td>94</td><td>Opus 5.5</td><td>81.91</td><td>81.91</td><td>84.04</td><td>2.13</td></tr><tr><td>VLM-ExecRouterBench</td><td>5,264</td><td>Opus 5.5</td><td>85.28</td><td>85.28</td><td>87.77</td><td>2.49</td></tr><tr><td>Image generation</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GenEval2</td><td>160</td><td>GPT Image 2.5</td><td>82.48</td><td>82.48</td><td>96.18</td><td>13.69</td></tr><tr><td>Qwen-Image-Bench</td><td>193</td><td>GPT Image 2.5</td><td>64.09</td><td>64.09</td><td>69.83</td><td>5.73</td></tr><tr><td>LongT2I</td><td>543</td><td>MAI-Image 2.6</td><td>86.14</td><td>86.14</td><td>86.89</td><td>0.75</td></tr><tr><td>OneIG</td><td>427</td><td>FLUX 3</td><td>83.35</td><td>83.35</td><td>89.34</td><td>5.99</td></tr><tr><td>WISE</td><td>192</td><td>Qwen-Image 3</td><td>92.19</td><td>92.71 (MAI-Image 2.6)</td><td>98.96</td><td>6.77</td></tr><tr><td>Video generation</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VBench-2.0</td><td>158</td><td>MiniMax H3</td><td>62.43</td><td>62.43</td><td>81.36</td><td>18.93</td></tr><tr><td>Speech generation</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Seed-TTS (en)</td><td>214</td><td>Fish S2.1</td><td>98.41</td><td>98.43 (Gemini 3.8 TTS)</td><td>99.02</td><td>0.61</td></tr><tr><td>Seed-TTS (zh)</td><td>394</td><td>Fish S2.1</td><td>98.73</td><td>98.73</td><td>99.45</td><td>0.72</td></tr><tr><td>Code and workplace agents</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>AA-Briefcase</td><td>1</td><td>Opus 5.5</td><td>66.67</td><td>75.00 (GPT-6.1 Sol)</td><td>75.00</td><td>8.33</td></tr><tr><td>AutomationBench</td><td>160</td><td>Opus 5.5</td><td>92.75</td><td>92.75</td><td>96.53</td><td>3.78</td></tr><tr><td>GDP.pdf</td><td>100</td><td>GPT-6.1 Sol</td><td>21.00</td><td>23.00 (GPT-6 Astra)</td><td>36.00</td><td>15.00</td></tr><tr><td>GDPval</td><td>43</td><td>Opus 5.5</td><td>91.18</td><td>91.18</td><td>92.79</td><td>1.61</td></tr><tr><td>Terminal-Bench</td><td>39</td><td>Opus 5.5</td><td>71.79</td><td>71.79</td><td>87.18</td><td>15.38</td></tr><tr><td>Robot manipulation</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>RoboTwin 2.0</td><td>2,000</td><td>ME-Dex</td><td>70.45</td><td>70.45</td><td>88.85</td><td>18.40</td></tr></table>

## B.2. Splits

The full set has 104,050 decisions over 25 benchmarks, split 70/10/20 with seed 42 by source group, so identical inputs and items from one source, such as the questions shared by MMMU-Pro and VLM-ExecRouterBench, never cross splits; existing research splits are kept, including RoboTwin’s split by task family (35 training, 5 development, and 10 test tasks). This gives 72,351 training, 10,630 development, and 21,069 test records; the router studies of Section 4 use the first release of 18 benchmarks (Table 2).

## B.3. Routers

Fable 5.1, a retired model, is excluded from every evaluation; routers trained with it play Opus 5.5 in its place.

Capability-tree router. Leaves are defined per task and labeled on training inputs by GPT-6.1 Sol, which sees the input, the output requirement, and the leaf definitions, but no candidate outcomes. The classifier is Qwen3.5-4B with LoRA (rank 8), trained for three epochs on all 18 benchmarks at once with batch 16 and learning rates $1 0 ^ { - 4 }$ (adapter) and $1 0 ^ { - 3 }$ (head); the development split selects the epoch. The leaf table is fitted on training inputs by least squares over paired candidate diferences on shared inputs.

Table 7 | Per-benchmark results of the capability-tree router (first block of Table 3); one training seed. SciCode steps are aligned across models; 175 of its 187 inputs have one scored candidate.
<table><tr><td>Benchmark</td><td>Inputs</td><td>Fixed candidate</td><td>Fixed</td><td>Router</td><td>Oracle</td></tr><tr><td>AA-LCR</td><td>42</td><td>Opus 5.5</td><td>97.62</td><td>95.24</td><td>100.00</td></tr><tr><td>AA-Omniscience</td><td>120</td><td>Opus 5.5</td><td>77.50</td><td>77.50</td><td>85.83</td></tr><tr><td>HLE (text)</td><td>426</td><td>Opus 5.5</td><td>58.45</td><td>58.45</td><td>64.55</td></tr><tr><td>SciCode</td><td>187</td><td>Opus 5.5</td><td>70.05</td><td>70.05</td><td>70.59</td></tr><tr><td>Terminal-Bench</td><td>36</td><td>Opus 5.5</td><td>69.44</td><td>77.78</td><td>86.11</td></tr><tr><td>AutomationBench</td><td>160</td><td>Opus 5.5</td><td>92.75</td><td>92.75</td><td>96.53</td></tr><tr><td>GDP.pdf</td><td>99</td><td>GPT-6.1 Sol</td><td>21.21</td><td>22.22</td><td>36.36</td></tr><tr><td>GDPval</td><td>37</td><td>Opus 5.5</td><td>91.84</td><td>91.84</td><td>93.62</td></tr><tr><td>GenEval2</td><td>162</td><td>GPT Image 2.5</td><td>82.66</td><td>82.66</td><td>96.18</td></tr><tr><td>Qwen-Image-Bench</td><td>197</td><td>GPT Image 2.5</td><td>64.25</td><td>64.25</td><td>69.93</td></tr><tr><td>LongT2I</td><td>552</td><td>MAI-Image 2.6</td><td>86.13</td><td>86.12</td><td>86.89</td></tr><tr><td>OneIG</td><td>450</td><td>FLUX 3</td><td>83.20</td><td>84.30</td><td>89.20</td></tr><tr><td>WISE</td><td>198</td><td>Qwen-Image 3</td><td>92.42</td><td>91.92</td><td>98.99</td></tr><tr><td>VBench-2.0</td><td>161</td><td>MiniMax H3</td><td>62.17</td><td>64.28</td><td>80.22</td></tr><tr><td>Seed-TTS (en)</td><td>216</td><td>Fish S2.1</td><td>98.42</td><td>98.42</td><td>99.02</td></tr><tr><td>Seed-TTS (zh)</td><td>401</td><td>Fish S2.1</td><td>98.74</td><td>98.74</td><td>99.45</td></tr><tr><td>All 16 benchmarks</td><td>3,444</td><td></td><td>79.68</td><td>79.98</td><td>84.80</td></tr></table>

Outcome and type routers on 11 benchmarks. Both use Qwen3.5-4B with LoRA (rank 16, $\alpha = 3 2 )$ on every linear layer of the language model, a final-token linear head, AdamW with learning rates $2 \times 1 0 ^ { - 5 }$ (adapter) and $3 \times 1 0 ^ { - 4 }$ (head), 32 tasks per update, and six epochs, with seeds 42, 13, 2026, and 7. The outcome router reads one (input, candidate profile) pair per forward pass; its loss is the cross-entropy against the normalized outcome plus four times the squared error of within-task score diferences. The type router predicts one of 68 capability types and reads a global type-to-candidate table fitted on training inputs.

Robot router. The outcome router uses Qwen3.5-4B with LoRA (rank 8), seven sigmoid heads, three epochs, batch $^ { 1 6 , }$ and learning rates $1 0 ^ { - 4 }$ (adapter) and $1 0 ^ { - 3 }$ (head); development routing success selects the epoch. The winner-label classifier shares the backbone, inputs, and data, with a seven-way softmax over the policy that succeeds, ties broken by measured inference time.

Image-routing suite. Thirty-one router designs, among them RouterDC [35], SCOPE [19], IRT-based routers, �NN, MLP, and two-tower scorers, and Qwen3.5 label routers with 0.8B to 9B parameters, are trained on GenEval2 or Qwen-Image-Bench prompts and tested on the same or the other benchmark: 124 configurations, with three seeds for stochastic methods. Each is compared with the fixed generator by a paired bootstrap over prompts.

Untrained router. Qwen3.5-4B, prompted with the input and the candidates, names the candidate to run.

Winner-label routers on 16 benchmarks. Qwen3.5-4B is fully fine-tuned to emit the label of the best candidate, ties broken by cost, tokens, and a fixed order. The 12 recipes start from the instructiontuned or a continually pretrained model, with or without fine-tuning on public routing data, and train per benchmark, on all benchmarks at once, or both in sequence. Tables 7 and 8 give per-benchmark results of the capability-tree router and of the 11-benchmark study.

Table 8 | Per-benchmark results of the 11-benchmark study (first block of Table 4), averaged over four training seeds; the fixed candidate is matched.
<table><tr><td>Benchmark</td><td>Inputs</td><td>Fixed candidate</td><td>Fixed</td><td>Outcome router</td><td>Type router</td><td>Oracle</td></tr><tr><td>AA-LCR</td><td>42</td><td>Opus 5.5</td><td>97.62</td><td>95.24</td><td>97.62</td><td>100.00</td></tr><tr><td>AA-Omniscience</td><td>120</td><td>Opus 5.5</td><td>77.50</td><td>77.29</td><td>71.88</td><td>85.83</td></tr><tr><td>HLE (text)</td><td>431</td><td>Opus 5.5</td><td>58.24</td><td>57.95</td><td>56.61</td><td>64.27</td></tr><tr><td>AA-Briefcase</td><td>1</td><td>GPT-6.1 Sol</td><td>75.00</td><td>62.50</td><td>50.00</td><td>75.00</td></tr><tr><td>AutomationBench</td><td>160</td><td>Opus 5.5</td><td>92.75</td><td>92.70</td><td>92.54</td><td>96.53</td></tr><tr><td>GDP.pdf</td><td>100</td><td>GPT-6.1 Sol</td><td>21.00</td><td>20.50</td><td>20.00</td><td>36.00</td></tr><tr><td>GDPval</td><td>43</td><td>Opus 5.5</td><td>90.52</td><td>90.17</td><td>90.31</td><td>92.20</td></tr><tr><td>SciCode</td><td>15</td><td>GPT-6.1 Sol</td><td>93.33</td><td>93.33</td><td>86.67</td><td>93.33</td></tr><tr><td>Terminal-Bench</td><td>39</td><td>Opus 5.5</td><td>69.23</td><td>68.59</td><td>67.31</td><td>87.18</td></tr><tr><td>GenEval2</td><td>158</td><td>GPT Image 2.5</td><td>82.56</td><td>82.18</td><td>82.56</td><td>96.13</td></tr><tr><td>Qwen-Image-Bench</td><td>193</td><td>GPT Image 2.5</td><td>64.09</td><td>63.59</td><td>64.09</td><td>69.83</td></tr><tr><td>All 11 benchmarks</td><td>1,302</td><td></td><td>68.30</td><td>67.90</td><td>66.98</td><td>75.84</td></tr></table>

Table 9 | Image price routes: test quality (0–100) and price (m\$ per image) of the fixed generator and of the half-point route of Table 5, fitted on training prompts.
<table><tr><td></td><td></td><td colspan="3">Fixed generator</td><td colspan="3">Half-point route</td></tr><tr><td>Benchmark</td><td>Test prompts</td><td>Candidate</td><td>Quality</td><td>m$</td><td>Candidate</td><td>Quality</td><td>m$</td></tr><tr><td>GenEval2</td><td>160</td><td>GPT Image 2.5</td><td>82.48</td><td>208.8</td><td>GPT Image 2.5</td><td>82.48</td><td>208.8</td></tr><tr><td>Qwen-Image-Bench</td><td>193</td><td>GPT Image 2.5</td><td>64.09</td><td>211.1</td><td>Grok Imagine 2.0</td><td>61.16</td><td>60.0</td></tr><tr><td>LongT2I</td><td>543</td><td>MAI-Image 2.6</td><td>86.14</td><td>39.4</td><td>MAI-Image 2.6</td><td>86.14</td><td>39.4</td></tr><tr><td>OneIG</td><td>427</td><td>FLUX 3</td><td>83.35</td><td>48.0</td><td>Qwen-Image 3</td><td>83.02</td><td>40.0</td></tr><tr><td>WISE</td><td>192</td><td>Qwen-Image 3</td><td>92.19</td><td>40.0</td><td>Qwen-Image 3</td><td>92.19</td><td>40.0</td></tr></table>

## B.4. Price Routes in Image and Speech Generation

Candidates are the generators scored on every input of a benchmark (Table 9). For each tolerance �, we take the most eficient price-path router that meets the floor on training prompts and evaluate it on test prompts with a paired bootstrap over prompts. With capability leaves as states, the fitted zero-loss route gains 0.05 points on training prompts but loses 0.33 [−0.90, +0.25] on test prompts at 1.34× eficiency. Prices per image are 208.8–211.3 m\$ for GPT Image 2.5, 60.0 for Grok Imagine 2.0, 47.7– 48.0 for FLUX 3, 45.0 for Seedream 5.0, 40.0 for Qwen-Image 3, and 39.0–39.4 for MAI-Image 2.6; on GenEval2 only GPT Image 2.5, FLUX 3, and Qwen-Image 3 are scored on every prompt. In speech generation, the fixed generator, Fish S2.1 (0.9–1.1 m\$ per output), is the best and among the cheapest at every tolerance up to two points; Gemini 3.8 TTS costs 1.0–1.2 m\$ and Seed Audio 1.0 13.5–16.0 m\$.

## C. Code and Text Testbeds: Details

## C.1. Data

CodeRouterBench [4] gives outcomes and token prices for eight models (claude-opus-4-6, claudesonnet-4-6, gpt-5.4, Qwen3-Max, Kimi-K2.5, Qwen3.5-Plus, glm-5, and MiniMax-M2.7) on 9,999 coding tasks: 7,080 probing tasks (train and validation combined) and 2,919 test tasks. Router-Bench [5] gives the 0-shot matrix of 36,497 prompts and eleven models. Dollar costs are token-price estimates; sequential seconds add recorded primary and fallback latencies. Policy, portfolio, retry, and signal studies replay recorded outcomes. Verifier verdicts and the LLM forecast are fresh calls through one OpenAI-compatible relay, with recorded Opus outcomes as the fallback; the memory study is semi-synthetic. Scores are on the benchmarks’ 0–1 scale, whereas the main text reports quality changes in points. Intervals are 95% percentile task-bootstrap intervals, paired for policy comparisons, and eficiency is a ratio of averages.

Table 10 | CodeRouterBench policies on the 2,919 test tasks. Dollar eficiency $\eta / \eta _ { B } .$ , cost in m\$ per task, and $\Delta V = \mathbb { E } [ V - Y _ { B } ]$ , with paired 95% intervals. Floors are fitted on probing; at $\epsilon \geq 0 . 0 2$ they are slack and give the unconstrained policy. Fixed dispatch plays the best probing model, which is also the cheapest (Qwen3-Max); the per-task oracle, an in-sample ceiling, gives each task the candidate maximizing $V - \eta ^ { \star } C$ on its realized outcomes.
<table><tr><td>Policy</td><td>η/ηB [CI]</td><td>Cost ∆V [CI]</td><td></td><td></td></tr><tr><td>21 cells, unconstrained</td><td>28.187 [26.840, 29.659]</td><td>0.395</td><td></td><td>-0.0201 [-0.0343, -0.0056]</td></tr><tr><td>21 cells,  $\epsilon = 0$ </td><td>25.516 [24.278, 26.903]</td><td>0.452</td><td></td><td>-0.0044 [-0.0175, +0.0097]</td></tr><tr><td>21 cells,  $\epsilon = 0 . 0 1$ </td><td>27.612 [26.311, 29.072]</td><td>0.409</td><td></td><td>-0.0140 [-0.0272, -0.0004]</td></tr><tr><td>9 dimensions, unconstrained</td><td>28.024 [26.713, 29.452]</td><td>0.398</td><td></td><td>-0.0192[-0.0332, -0.0047]</td></tr><tr><td>9 dimensions,  $\epsilon = 0$ </td><td>22.733 [21.584, 23.954]</td><td>0.502</td><td></td><td>-0.0090 [-0.0227, +0.0060]</td></tr><tr><td>Fixed Qwen3-Max dispatch</td><td>20.097 [18.965, 21.297]</td><td>0.525</td><td></td><td>-0.0416[-0.0549, -0.0275]</td></tr><tr><td>Best fixed cascade</td><td>2.004 [1.881, 2.140]</td><td></td><td></td><td>6.556 +0.0558 [+0.0461, +0.0660]</td></tr><tr><td>ACRouter released map</td><td>1.745 [1.666, 1.826]</td><td></td><td></td><td>7.643 +0.0631 [+0.0528, +0.0731]</td></tr><tr><td>Per-task oracle</td><td>33.510 [32.065, 35.206]</td><td></td><td></td><td>0.368 +0.0248 [+0.0117, +0.0376]</td></tr><tr><td>Always-Opus</td><td>1.000</td><td>11.655 0</td><td></td><td></td></tr></table>

## C.2. Price Routes

Table 10 gives the CodeRouterBench policies of Section 5.1. With states, every dollar-optimal policy dispatches each task to a non-Opus model, for both partitions and every fitted floor, whereas the best fixed cascade gains 0.0558 at only 2.00×. On the 21-cell menu, the zero-loss policy has multiplier $\kappa = 5 . 4 3$ and constrained optimum $\eta _ { \epsilon } ^ { \star } = 9 2 7 . 7$ score per dollar, so $\lambda _ { \epsilon } = 1 4 4 . 3 = 4 . 1 6 \eta _ { B }$ . It maximizes $J _ { \lambda _ { \epsilon } } { \mathrm { : } }$ , as Theorem 2(ii) predicts, and is therefore also the quality maximizer at its own budget and, since $\kappa > 0 _ { : }$ , the cheapest policy that meets the floor. Hand-picked operating points land elsewhere: $\lambda \ : = \eta _ { B }$ gives $1 0 . 8 5 \times ,$ , and a budget equal to Opus’s cost 2.16×. Refitting the policy inside 500 replicates that halve and bootstrap the probing tasks gives 25.72× [23.01, 28.04] at $\Delta V = - 0 . 0 0 3 9$ $\left[ - 0 . 0 1 7 8 , + 0 . 0 0 9 9 \right]$ ; choosing a safety margin by a plain probing bootstrap instead gives margins of 0.015–0.0175 and test compliance of only 0.47–0.96.

Certification. Each of 200 splits uses a random half of probing as the fitting sample A (3,540 tasks) and the other half as B, fits the price path on A, tests it on $\mathcal { B }$ at $\alpha = 0 . 0 5$ , and evaluates the selected router on the test set. The full probing path has 38 routers, against the bound of 442. $\mathrm { A t } \epsilon = 0$ , the certified router has median eficiency 15.17× with Hoefding �-values, 18.83× with empirical-Bernstein ones, and 19.38× with empirical-Bernstein ones on the mixture-refined path (Appendix A.5); Bonferroni selection over the same path gives 13.29× and 15.17×. In 200 nested 70/30 resplits of all 9,999 tasks, the refined procedure holds the floor in 199 at a median 20.56×. The guarantee is an aggregate one: the certified router loses quality in a median 8 of 21 test cells. Certification needs every candidate on the path and the reference on B, at \$0.0158 per task against \$0.0124 for Opus alone.

States without oracle labels, and learned routers. Prompts are rebuilt with the benchmark loaders (9,996 of 9,999 tasks). Predicted-state policies refit candidate values on five-fold out-of-fold predictions, so fitting and deployment see the same noisy state; a second variant first strips every line that occurs in at least 20% of one source’s prompts, by one rule learned on probing prompts (19% of test-prompt characters), so that source wrappers cannot reveal the cell. Learned quality–cost routers predict each model’s score and log cost from TF-IDF features by ridge regression and play $\begin{array} { r } { \operatorname* { m a x } _ { m } \hat { \nu } _ { m } - \lambda \hat { c } _ { m } , } \end{array}$ directly, as a cascade, or under a mean-cost budget; the RouteLLM-style router [2] classifies whether Opus beats Qwen3-Max. The FrugalGPT-style cascade [1] learns a TF-IDF scorer per model and searches all 400 chains of at most three models, and the AutoMix-style cascade [42] accepts the small model’s answer when a meta-verifier accepts the recorded grok-4.6 verdict. Operating points are chosen on probing (Table 11). Refitting the stripping rule, features, and classifier inside 200 replicates gives predicted cells a median 24.83× at $\Delta V = - 0 . 0 0 2 6 \colon$ routing from prompt text works within the distribution.

Table 11 | CodeRouterBench states without oracle labels, and learned routers. $r = \eta / \eta _ { B } ;$ cost in m\$ per task; compliance is the share of test resamples with $\Delta V \geq 0$
<table><tr><td>States or router</td><td>Operating point</td><td>r [95% CI]</td><td>Cost</td><td></td><td>∆V [95% CI]</td><td>Compl.</td></tr><tr><td>True 21 cells</td><td> $\epsilon = 0$ </td><td>25.52 [24.26, 26.77]</td><td></td><td></td><td> $0 . 4 5 2 \ - 0 . 0 0 4 4 \ [ - 0 . 0 1 7 7 , + 0 . 0 0 7 7 ]$ </td><td>0.22</td></tr><tr><td>Predicted 21 cells</td><td> $\epsilon = 0$ </td><td>25.32 [24.07, 26.63]</td><td></td><td></td><td> $0 . 4 5 3 - 0 . 0 0 7 0 [ - 0 . 0 2 0 3 , + 0 . 0 0 5 8 ]$ </td><td>0.13</td></tr><tr><td>Pooled (no state)</td><td> $\epsilon = 0$ </td><td>3.62 [3.41, 3.85]</td><td></td><td></td><td> $3 . 2 3 2 \ + 0 . 0 0 2 1 \ [ - 0 . 0 0 8 5 , + 0 . 0 1 3 5 ]$ </td><td>0.63</td></tr><tr><td colspan="5">Source wrappers removed from every prompt by one rule learned on probing prompts</td><td></td><td></td></tr><tr><td>Predicted 21 cells</td><td> $\epsilon = 0$ </td><td>24.92 [23.65, 26.28]</td><td>0.464</td><td></td><td>-0.0032[-0.0163, +0.0096]</td><td>0.28</td></tr><tr><td>Quality-cost, direct</td><td> $\mathtt { p r o b i n g } \epsilon \approx 0$ </td><td>24.17 [22.83, 25.49]</td><td>0.482</td><td></td><td>-0.0003 [-0.0140, +0.0125]</td><td></td></tr><tr><td>Quality-cost, cascade</td><td> $\mathtt { p r o b i n g } \epsilon \approx 0$ </td><td>22.67 [20.48, 24.72]</td><td></td><td></td><td> $0 . 5 1 2 \ : \ : - 0 . 0 0 1 6 \ : [ - 0 . 0 1 4 2 , + 0 . 0 1 1 4 ]$ </td><td></td></tr><tr><td>Quality-cost, budget</td><td> $\mathtt { p r o b i n g } \epsilon \approx 0$ </td><td>23.57 [22.21, 24.90]</td><td></td><td></td><td>0.497+0.0026[-0.0107,+0.0157]</td><td></td></tr><tr><td>kNN-20, direct</td><td> $\mathtt { p r o b i n g } \epsilon \approx 0$ </td><td>23.58 [22.28, 24.89]</td><td></td><td></td><td> $0 . 5 0 0 \ + 0 . 0 0 5 0 \ [ - 0 . 0 0 8 0 , + 0 . 0 1 7 5 ]$ </td><td></td></tr><tr><td>RouteLLM-style</td><td> $\mathtt { p r o b i n g } \epsilon \approx 0$ </td><td>6.36 [5.82, 6.90]</td><td></td><td></td><td> $\begin{array} { r l } { 1 . 8 6 1 } & { { } + 0 . 0 0 6 4 [ - 0 . 0 0 6 4 , + 0 . 0 1 9 3 ] } \end{array}$ </td><td></td></tr><tr><td>FrugalGPT-style cascade</td><td> $\mathtt { p r o b i n g } \epsilon \approx 0$ </td><td> $1 1 . 7 6 [ 1 1 . 0 2 , 1 2 . 5 1 ]$ </td><td></td><td></td><td> $1 . 0 0 0 + 0 . 0 0 3 9 [ - 0 . 0 0 9 3 , + 0 . 0 1 7 1 ]$ </td><td>0.71</td></tr><tr><td colspan="5">334 runnable test tasks with recorded judge verdicts; always-Opus scores 0.737 at 5.413 m$</td><td></td><td></td></tr><tr><td>AutoMix-style, judge free</td><td> $\mathtt { p r o b i n g } \epsilon \approx 0$ </td><td>1.64 [1.47, 1.89]</td><td></td><td></td><td> $3 . 2 8 9 \ \ - 0 . 0 0 3 0 \ [ - 0 . 0 2 4 0 , + 0 . 0 1 5 0 ]$ </td><td>0.43</td></tr><tr><td>AutoMix-style, judge at list price</td><td> $\mathtt { p r o b i n g } \epsilon \approx 0$ </td><td>0.28 [0.26, 0.31]</td><td></td><td></td><td> $1 9 . 1 2 5 \ + 0 . 0 0 0 0 \ [ - 0 . 0 1 8 0 , + 0 . 0 1 8 0 ]$ </td><td>0.55</td></tr></table>

A stronger yardstick. CellBest routes each cell to its best probing model; the Opus-referenced zero-loss policy falls 0.087 below it. Refitted against CellBest, the zero-loss policy reaches 3.18× [2.94, 3.40] CellBest’s eficiency at $\Delta V ^ { \prime } = - 0 . 0 0 3 9 \ [ - 0 . 0 1 0 8 , + 0 . 0 0 2 7 ]$ ; a floor certifies quality only relative to the reference it names.

RouterBench. The reference, gpt-4-1106-preview, scores 0.781 at 3.29 m\$ per test prompt. The menu has 21 candidates: direct dispatch to each of the eleven models, and each other model followed by the reference when its released score is below $\theta = 1$ . States are the eight source families, the 86 datasets (57 MMLU subsets), datasets predicted from the prompt (92.0% test accuracy), TF-IDF �-means clusters, or one pooled state (Table 12). With family states, the best router that avoids the reference scores 0.097 below it, and the admission price $5 . 8 5 \eta _ { B }$ exceeds the zero-loss floor price $1 . 6 0 \eta _ { B } ,$ , so the zero-loss router calls the reference: it sends 68% of prompts to cascades, and the reference runs on 21%. Certified on half of the fitting prompts, the zero-loss floor holds on the test prompts in all 200 halvings at a median $6 . 7 0 \times ;$ uncertified fitted floors hold on 4 of 11 splits with family states and on none with dataset states. At matched cost, learned quality–cost cascades lie 0.014–0.022 above the 8-family frontier and tie with the 86-dataset one, so prompt-level prediction adds value where states are coarse.

## C.3. Portfolio Selection

Candidates are fitted on probing and evaluated on test tasks. CodeRouterBench adds one of seven candidates, each with at least 20 passes and 20 failures, to portfolios of up to three models; Router-

Table 12 | Fixed-menu policies on 10,955 RouterBench test prompts at $\theta = 1$ $r = \eta / \eta _ { B }$ against gpt-4-1106-preview; cost in m\$ per prompt; compliance is the share of resamples with $\Delta V \geq 0$
<table><tr><td>States or router</td><td>Operating point</td><td>r [95% CI]</td><td>Cost</td><td></td><td>∆V [95% CI]</td><td>Compl.</td></tr><tr><td>Best fixed cascade (Yi-34B, then reference)</td><td>probing</td><td>1.38 [1.37, 1.39]</td><td>2.515</td><td></td><td>+0.0425[+0.0390, +0.0461]</td><td>1.00</td></tr><tr><td>8 families</td><td>unconstrained</td><td>32.11 [31.37, 32.81]</td><td>0.052</td><td></td><td>-0.3817[-0.3910, -0.3722]</td><td>0.00</td></tr><tr><td>8 families</td><td>€ = 0</td><td>7.10 [6.91, 7.33]</td><td>0.461</td><td></td><td>-0.0040 [-0.0084, +0.0008]</td><td>0.05</td></tr><tr><td>86 datasets</td><td> $\epsilon = 0$ </td><td>8.26 [8.01, 8.51]</td><td>0.395</td><td></td><td>-0.0073 [-0.0125, -0.0024]</td><td>0.00</td></tr><tr><td>Predicted 86 datasets</td><td> $\epsilon = 0$ </td><td>7.85[7.55, 8.16]</td><td>0.416</td><td></td><td>-0.0057[-0.0112, -0.0009]</td><td>0.01</td></tr><tr><td>TF-IDF k-means 20</td><td> $\epsilon = 0$ </td><td>8.10 [7.82, 8.39]</td><td>0.405</td><td></td><td>-0.0024[-0.0079, +0.0026]</td><td>0.17</td></tr><tr><td>Pooled (no state)</td><td> $\epsilon = 0$ </td><td>1.71 [1.70, 1.73]</td><td>1.910</td><td></td><td>-0.0035[-0.0073, +0.0008]</td><td>0.06</td></tr><tr><td>Quality-cost, cascade</td><td>price, fit  $\epsilon \approx 0$ </td><td>8.10 [7.89, 8.32]</td><td>0.405</td><td></td><td>-0.0025 [-0.0080, +0.0029]</td><td>0.18</td></tr><tr><td>Quality-cost, cascade</td><td>budget, fit € ≈ 0</td><td>7.07 [6.88, 7.27]</td><td>0.474</td><td></td><td>+0.0152 [+0.0104, +0.0199]</td><td>1.00</td></tr><tr><td>kNN-20, cascade</td><td> ${ \mathrm { p r i c e } } , { \mathrm { f i t } } \epsilon \approx 0$ </td><td>6.71 [6.50, 6.94]</td><td>0.492</td><td></td><td>+0.0026[-0.0024, +0.0076]</td><td>0.83</td></tr><tr><td>Per-task oracle (in-sample)</td><td>€ = 0</td><td>45.29 [44.68, 45.87]</td><td>0.073</td><td>0.0000 [-0.0073, +0.0072]</td><td></td><td>0.48</td></tr></table>

Table 13 | Best-candidate selection: the share of comparisons (one portfolio with several candidate additions) in which a rule’s fitted choice has the highest test eficiency. Counts sum over splits, which share tasks. Paired ranks $G _ { m } ( S )$ of Eq. (8); residual ranks $N _ { m } ( S )$ ; Indep. assumes independent failures; saving ranks $r _ { m } b - d _ { m }$
<table><tr><td>Dataset, threshold</td><td>Splits</td><td>Count</td><td>Paired</td><td>Residual Indep.</td><td></td><td>Saving</td><td>Marginal</td></tr><tr><td>CodeRouterBench, θ = 1</td><td>1</td><td>139</td><td>97.1%</td><td>86.3%</td><td>86.3%</td><td>95.7%</td><td>46.8%</td></tr><tr><td>CodeRouterBench,  $\theta = 0 . 5$ </td><td>1</td><td>194</td><td>96.4%</td><td>94.3%</td><td>91.8%</td><td>93.3%</td><td>40.7%</td></tr><tr><td>RouterBench,  $\theta = 0 . 5$ </td><td>10</td><td>5,539</td><td>63.9%</td><td>61.9%</td><td>635.9%</td><td>63.5%</td><td>14.6%</td></tr><tr><td>RouterBench, θ = 1</td><td>10</td><td>212</td><td>64.6%</td><td>53.8%36.3%</td><td></td><td>48.1%</td><td>0.0%</td></tr></table>

Bench adds ten candidates to 21 initial portfolios over ten fixed 70/30 splits. The selecting verifier is an oracle that commits to the best passing member, else calls the reference, and every member call is charged. Paired selection picks the candidate with the highest fitted $\hat { \eta } ( S ^ { + } )$ (Eq. (8)); the controls rank residual coverage $N _ { m } ( S ) \ ( \mathrm { E q . \ ( 9 ) } )$ , an independence proxy that replaces $r _ { m } ( S )$ by $( 1 - p ( S ) ) \mathbb { P } ( E _ { m } )$ the saving score $r _ { m } ( S ) b - d _ { m } ,$ or marginal success (Table 13). Residual coverage evaluates the quality term at the current portfolio’s conditional means, and those means shift: adding a model lowers the miss-conditional value in 97% of CodeRouterBench comparisons, which Eq. (10) restores. Adding Qwen3.5-Plus to the pooled Qwen3-Max portfolio, for example, is predicted to lose 1.4 score per dollar but gains +5.0 [3.5, 6.6], because the fallback cost of the new failure set falls by 14%. Clusterbootstrap intervals for the diferences in best-candidate frequency, resampling CodeRouterBench dimensions or RouterBench source families, all include zero. Over ten CodeRouterBench resplits, mean relative regret at � = 1 and 0.5 is 0.04% and 0.06% for paired selection, 0.29% and 0.05% for residual coverage, and 13.1% and 17.0% for marginal success.

## C.4. Verifiers and Overheads

On recorded responses of three primaries to 500 probing and 334 runnable test tasks, we compare a static Compiler gate, which compiles the extracted answer and the oficial harness to bytecode and checks the declared entry point without executing code; visible examples; hidden tests, an oracle-like control; the released-score oracle; and a grok-4.6 judge on two primaries, charged at the primary’s blended token price or at its list price of \$2 and \$6 per million input and output tokens (Table 14). Holding outputs and the threshold fixed, the judge improves eficiency exactly when its mean charge is below $\bar { V } _ { \theta } / \eta _ { B } - \bar { D } _ { 0 , \theta . }$ , where $\bar { V } _ { \theta }$ is the mean delivered score and $\bar { D } _ { 0 , \theta }$ the mean cost before the judge. For Qwen3-Max at $\theta = 0 . 5$ this break-even charge is \$0.00330 per task [\$0.00285, \$0.00379]; at � = 1 it is negative for both primaries, so even a free judge does not help. Accepting every Qwen3-Max output reaches 19.65× but loses $Q = - 0 . 0 6 3 \ [ - 0 . 1 0 8 , - 0 . 0 2 4 ]$ , which the floor excludes.

Table 14 | Verifier choice changes the cascade’s eficiency and quality. $r = \eta / \eta _ { B } ; Q$ is mean score change. Qwen3-Max primary; the judge is charged at the primary’s token price.
<table><tr><td>Verifier</td><td>AUROC</td><td> $r ( \theta = 0 . 5 )$  [95% CI]</td><td> $r ( \theta = 1 )$  [95% CI]</td><td> $Q ( \theta = 0 . 5 )$ </td></tr><tr><td>Compiler (static gate)</td><td>0.766</td><td>4.09 [3.23, 5.45]</td><td>4.09 [3.23, 5.45]</td><td>-0.024</td></tr><tr><td>Visible examples</td><td>0.537</td><td>1.98 [1.72, 2.35]</td><td>1.06 [1.03, 1.10]</td><td>-0.036</td></tr><tr><td>Hidden tests</td><td>0.994</td><td>2.87 [2.38, 3.53]</td><td>2.37 [2.03, 2.80]</td><td>+0.024</td></tr><tr><td>LLM judge</td><td>0.824</td><td>1.09 [1.00, 1.21]</td><td>0.64 [0.61, 0.66]</td><td>-0.012</td></tr><tr><td>Released-score oracle</td><td>1.000</td><td>2.46 [2.09, 2.93]</td><td>2.46 [2.09, 2.93]</td><td>+0.045</td></tr></table>

Table 15 | Cross-fitted capture of the nine released CodeRouterBench routers as signals � (chose Qwen3-Max) for Qwen3-Max at � = 1, with no signal charge. Native use attempts on $Z = 1$ , threshold use when $s _ { Z } > a / b ;$ the last column applies the threshold rule given the benchmark dimension and divides the aggregated saving by the aggregated bound.
<table><tr><td>Signal source</td><td>Native [95% CI]</td><td>Threshold [95% CI]</td><td>Within dimension</td></tr><tr><td>Fine-tuned Qwen 27B</td><td>0.555 [0.490, 0.614]</td><td>0.555 [0.490, 0.614]</td><td>0.062</td></tr><tr><td>Fine-tuned Qwen 9B</td><td>0.546 [0.475, 0.608]</td><td>0.546 [0.475, 0.608]</td><td>0.078</td></tr><tr><td>Fine-tuned Qwen 2B</td><td>0.466 [0.397, 0.531]</td><td>0.417 [0.355, 0.479]</td><td>-0.006</td></tr><tr><td>Fine-tuned Qwen 0.8B</td><td>0.422 [0.348, 0.491]</td><td>0.082 [0.066, 0.098]</td><td>-0.008</td></tr><tr><td>Logistic regression</td><td>-1.218 [-1.590, -0.869]</td><td>0</td><td>-0.002</td></tr><tr><td>TF-IDF MLP</td><td>-1.736 [-2.178, -1.342]</td><td>0</td><td>-0.012</td></tr><tr><td>RouteLLM MF</td><td>-1.033[-1.364, -0.723]</td><td>0</td><td>-0.013</td></tr><tr><td>RouteLLM SW</td><td>-1.284 [-1.685, -0.953]</td><td>0</td><td>+0.005</td></tr><tr><td>DimensionBest</td><td>-1.468 [-1.847, -1.104]</td><td>0</td><td>0</td></tr></table>

Router inference is timed on one Apple M4 core and priced at AWS Lambda’s x86 list price for a 2 GB function $( \$ 3.33\times10^ { - 5 }$ per second, us-east-1, September 2026). Per task, the TF-IDF router takes 1.9 ms (0.06% of a Qwen3-Max call), a hidden-test sandbox run 0.15 s and $\$ 5.2\times10^ { -6}$ (2.0%), the grok-4.6 judge 39.0 s and \$0.0159 at list price (63×), and the DeepSeek-v4-pro forecast 1.89 s and \$0.000944 (2.4×); charging router overhead leaves every replay comparison unchanged at the reported precision. Opus takes 3.93 s per task against 6.23 s for Qwen3-Max, so in sequential seconds direct dispatch reaches only 0.58× (Qwen3-Max) and 0.50× (Kimi-K2.5) Opus’s eficiency, verified cascades 0.40–0.54×, and the judge’s 39-second call lowers the ratio to 0.08×.

## C.5. Routing Signals

The released CodeRouterBench routers expose their chosen model, which we use as a binary signal � (chose Qwen3-Max), with posteriors cross-fitted over ten half-splits of the test set; capture is the gross saving divided by the perfect-information bound of Proposition 9 (Table 15). Classical routers have negative native capture because their � = 0 tasks still satisfy $\mathbb { P } ( H = 1 \mid Z = 0 ) > a / b = 0 . 0 4 3$ , so withholding the attempt forgoes profitable ones. For the LLM forecast, DeepSeek-v4-pro reads the prompt and predicts whether Qwen3-Max and Kimi-K2.5 pass all hidden tests on 600 probing and 600 test tasks; its calibrated posterior reaches test AUROC 0.541 and 0.618 and a capture of −0.037 [−0.225, 0.077], and at list price it costs more than the attempt it could save in 22 of 24 states.

## C.6. Router Memory

This semi-synthetic study combines real outputs and verifier judgments with constructed arrivals and retention: 2,044 probing and 867 test tasks in 12 cells. With probability 0.25 the writer explores a uniformly sampled model, and it writes the model’s identity to the arriving cell when verification accepts; the newest record serves later tasks in that cell, and uncovered cells use the prior, always-Opus or the probing-fitted DimensionBest, which sends each dimension to its best model. Writers use ACRouter’s doctest and syntax tools, locally executed hidden tests, or a DeepSeek-v4-pro judge; we report hidden-test writes. A write is erroneous when the accepted output is incorrect under the released score; against that truth, hidden-test writes have TPR/FPR 0.960/0.205. Fifty streams of 60,000 rounds are measured over their last 10,000 rounds. With no expiration, the store reaches 135.8 score per dollar but scores 0.034 below always-Opus and 0.116 below DimensionBest.

## D. Limitations

The theory takes the law of states as given, so it does not cover states that depend on the router’s own earlier choices, such as SciCode steps that contain a model’s earlier code, and certification needs the reference’s outcome on every certification input. Moreover, step-level routing for long-horizon agentic tasks is more complex and deserves deeper exploration.