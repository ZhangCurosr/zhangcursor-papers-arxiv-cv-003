# PerceptRL: Making Visual Perception Rewardable in MLLMs from Answers to Interactions

Tongkun Guan<sup>1,2</sup>, Zhibo Yang<sup>2,†</sup>, Mingkun Yang<sup>2</sup>, Haocheng Wang<sup>1</sup>, Wei Shen<sup>1</sup>, Xiaokang Yang<sup>1</sup>, Fellow, IEEE

Abstract—Fine-grained visual perception remains a bottleneck during post-training of multimodal large language models (MLLMs), where models first perceive images and then reason over the extracted information, so perceptual errors can limit the gains from reasoning-focused optimization. Although extensive image-caption pre-training improves perception, the influence of this supervision can be diluted as later training stages change the data mixture and objectives. Continuing to optimize captions during post-training is a natural response, but their subjective, information-dense nature makes verifiable reward design difficult. To address this, we present PerceptRL to strengthen perception during post-training at two complementary levels. Perception RL uses a multi-agent pipeline to extract verified error items from dense image captions and compose the corresponding visual facts into rewardable QA pairs: chain QA links correlated errors, while multiple-choice QA combines unrelated errors. A specialized route targets high-error domains with tailored verification, while a general route addresses perceptual errors beyond these domains through a domain-agnostic pipeline. This error-driven construction provides fixed answer labels and rule-based rewards, making caption-derived perceptual supervision verifiable. Perception Agentic RL targets fine-grained and dense cases that single-turn Perception RL still fails to resolve, enabling multi-turn inspection with visual tools. We observed coordinate drift even in models trained to invoke tools: incorrect crops or annotations can distort later evidence and propagate errors. Coordinate Injection corrects spatial operations in cold-start trajectories, while a grounding-aware process reward supervises intermediate spatial fidelity alongside final-answer correctness. Training proceeds through shared cold-start SFT, single-turn Perception RL, and multi-turn Perception Agentic RL. Across seven benchmarks under a shared tool-use protocol, the full pipeline achieves 71.6% average accuracy, exceeding the base model with the same tools and the strongest evaluated frontier VLM by 15.7 and 5.2 percentage points, respectively.

Index Terms—Multimodal large language models, visual perception, reinforcement learning.

## 1 INTRODUCTION

of visual perception. No matter how sophisticated a model’s reasoning capabilities, if it fails at the perceptual level (e.g., miscounting objects, confusing colors, or misjudging spatial relationships), downstream reasoning may produce wrong answers [1], [2]. Fine-grained perception therefore remains an important concern for current MLLMs [3]–[6]. CodePercept [7] finds that scaling perception is more effective than scaling reasoning in its analysis of STEM visual reasoning. A perceptual error can compromise an otherwise valid reasoning chain (Figure 1, left).

Image–caption data provide an important source of visual supervision during the pre-training stage [8]–[10]. However, production MLLMs undergo multiple stages, including ViT pre-training, vision–language alignment, instruction tuning, reinforcement learning, and on-policy distillation [3], [11]. As the data mixtures and training objectives change, the influence of perception-focused captions used in the pre-training stage can be diluted by later optimization.

A natural response is to continue optimizing captions during post-training, with reinforcement learning (RL) providing feedback on model-generated outputs. For example, in mathematical reasoning, final answers can often be checked against reference solutions, giving RL objective feedback on generated solutions without requiring annotations for every reasoning step [12], [13]. Direct caption optimization, however, encounters two further difficulties. Subjectivity: factually correct captions can differ in granularity, emphasis, style, and length, making the desired optimization target difficult to specify. Unrewardability: captions are information-dense and open-ended, bundling numerous visual claims into one description. Constructing a verifiable reward requires deciding which claims to assess and how to combine their correctness into a caption-level score.

Solver-based methods [14] reward captions according to how well a vision-free LLM solver can answer visual questions using those captions. This is an indirect perception signal: a high reward indicates that the caption provides information useful to the sampled questions and that the solver can exploit it, but does not establish that the caption captures the image accurately and comprehensively. Visual details not required by those questions may remain unexamined.

Code-based methods [7] use image-to-code-to-image reproduction to supervise visual perception, using code executability and reconstruction fidelity as reward signals. However, this formulation is developed for structured

![](images/ab818e718cbef48300d0a4f8d12365119534606522881cf4a2e110e087acdd9a.jpg)  
Fig. 1: PERCEPTRL: making real-world perception rewardable, and training it at two levels. Center: How it becomes rewardable. Open-ended captions contain useful perceptual information but are difficult to score with direct rule-based rewards. We extract and verify perceptual facts from these captions, then formulate QA pairs with fixed answer labels for rule-based training (top). The pipeline organizes tasks from multiple perception families (e.g., color, spatial position, counting, functional state, orientation), using a perception tree to guide data construction (Section 3.2). The difficulty axis is what divides the two levels. Level 1: Perception RL. A single look suffices here, so we train single-turn visual QA under a rule-based outcome reward. The left panel shows what this must fix: an error entering at the perception level is inherited by an otherwise correct reasoning chain, so the answer is wrong for a purely perceptual reason. Level 2: Perception Agentic RL. This tail, where objects are small, scenes are cluttered, or details require magnification, resists a single look, so the model inspects the image over multiple grounded turns with a code interpreter. The right panel illustrates how a grounding error, where a predicted crop or annotation misses its target, can affect subsequent observations. Its propagation and accumulation across successive tool interactions constitute coordinate drift. Coordinate Injection corrects these grounding errors during data construction, while a grounding-aware process reward supervises intermediate spatial fidelity (Section 3.3).

STEM images that can be rendered programmatically; its generalization to diverse real-world images remains unestablished.

These three issues raise a central question: how can we reinforce visual perception during post-training through objective optimization targets and verifiable rewards, so that the learned skills generalize across diverse perception tasks? We present PERCEPTRL, a framework that makes perceptual supervision explicit at two complementary levels: single-turn answers and multi-turn visual interactions (Figure 1).

Level 1: Perception RL. Our key insight is that the base model’s own perceptual errors can guide the construction of targeted, verifiable training data. We design a multi-agent pipeline that first elicits dense captions from the base model, audits their visual claims, and cross-verifies detected errors against independent vision models. The critical step is error composition: we turn verified visual facts associated with these errors into rewardable QA pairs, combining unrelated errors into multiple-choice questions and linking correlated errors into multi-step visual identification chains. With fixed answer labels supported by cross-verification and human review, these questions provide rule-based outcome rewards, making caption-derived perceptual supervision explicit and verifiable. Guided by a perception tree, specialized pipelines focus on selected high-error domains, while an error-agnostic general pipeline discovers and composes failures beyond those domains. Both routes follow this error-driven principle, concentrating single-turn Perception RL on the model’s demonstrated perceptual weaknesses.

Level 2: Perception Agentic RL. For fine-grained, complex, and dense perceptual errors, where objects are small, scenes are cluttered, or details require magnification, single-turn answering often fails. Tool use through a code interpreter can substantially help by enabling the model to inspect image regions over multiple turns. Prior work has explored learning visual tool-use policies and integrating code execution into reasoning [15]–[18]. In multi-turn visual inspection, however, choosing an appropriate tool or operation does not by itself ensure that the predicted crop or annotation lands on the intended target. A grounding error occurs when a crop box or annotation point misses its intended target. An incorrect crop may omit relevant evidence, while a misplaced annotation can mislead subsequent localization or counting. If later turns depend on these observations, the initial error may propagate and compound across interactions. We refer to this process as coordinate drift (Figure 1, right). This motivates a complementary focus on intermediate spatial fidelity alongside final-answer correctness. To this end, Coordinate Injection (CI) constructs cold-start trajectories with corrected spatial coordinates, while a grounding-aware process reward supervises the spatial fidelity of intermediate operations during multi-turn RL.

Training follows three sequential stages: a single shared cold-start SFT stage on a mixture of perception QA pairs and

CI-corrected trajectories, followed by single-turn Perception RL and then multi-turn Perception Agentic RL. Each stage starts from the preceding checkpoint; there is no additional SFT stage between the two RL stages.

Our contributions are as follows:

• We propose Perception RL to address the difficulty of defining verifiable rewards for open-ended image captions. It converts verified perceptual facts associated with the model’s caption errors into QA pairs with fixed answer labels, enabling single-turn post-training of visual perception with rule-based outcome rewards. Across eight perception benchmarks, this stage raises average accuracy from 61.9% after cold-start SFT to 69.8% (Table 1).

• We propose Perception Agentic RL for fine-grained and dense perception requiring multi-turn inspection. To mitigate coordinate drift, Coordinate Injection corrects spatial operations in cold-start trajectories, while a grounding-aware process reward supervises intermediate grounding alongside answer correctness. Under the shared tool-use protocol, the full pipeline achieves 71.6% average accuracy across seven benchmarks, outperforming the base model with the same tools and the strongest evaluated frontier VLM by 15.7 and 5.2 percentage points, respectively (Table 2).

## 2 RELATED WORK

## 2.1 Fine-Grained Visual Perception

Fine-grained visual perception requires recovering local details and their spatial structure [19], [20]. Grounding DINO [21] combines language-guided feature fusion and grounded pre-training to detect objects specified by category names or referring expressions. SAM [22] learns promptable segmentation from large-scale mask supervision, while Florence-2 [23] unifies captioning, detection, grounding, and segmentation through a common sequenceto-sequence formulation.

Text-centric methods improve detection through feature refinement and synthetic-to-real pre-training [24], [25], and recognition through glyph alignment and character-level distillation [26]–[28]. Structured document perception further incorporates symbol positions for mathematical expressions [29], token-level supervision for dense text [30], and joint question answering and mask generation [31].

LReSeg [32] addresses reasoning segmentation with a small language model, using text-conditioned register tokens and a unified visual encoder for reasoning and mask decoding. Wen et al. [33] reduce the cost of MLLM-based segmentation through delayed mask-token insertion and proxy-guided visual-token updates, preserving spatial details for dense prediction. These studies emphasize explicit alignment between visual representations and spatially localized outputs.

In the medical domain, Hulu-Med [34] unifies text, 2D images, 3D volumes, and videos in a generalist VLM, using medical-aware visual token reduction to improve computational efficiency. PERCEPTRL investigates how to provide verifiable perception supervision across domains and spatial supervision for successive tool interactions during reinforcement learning.

## 2.2 Reinforcement Learning for VLMs

Reasoning-focused. Reinforcement learning with verifiable rewards (RLVR) has emerged as a promising paradigm for enhancing reasoning in large language models [12], [13]. Its extension to vision-language models has attracted growing attention [35]–[38]. Vision-R1 [39] constructs a 200K multimodal CoT dataset for cold-start initialization and employs GRPO [40] with progressive thinking suppression training. VLM-R1 [41] applies rule-based RL to vision-language tasks with a focus on training stability and reward hacking mitigation. R1-VL [42] introduces step-wise rewards to improve logical consistency in multimodal reasoning. G1 [43] discovers that perception and reasoning abilities can mutually bootstrap each other during RL training through a perception-enhanced cold start on visual games. AVAR [44] proposes the Visual Attention Score (VAS) to quantify visual token attention and designs attention-guided objectives to address lazy attention localization. Despite these advances, the above methods primarily target reasoning capabilities, improving the logical process from visual evidence to conclusions, while leaving the perceptual accuracy of the visual evidence itself largely unaddressed.

Perception-focused. Perception-R1 [45] demonstrates via McNemar’s test that standard RLVR does not improve perception and proposes a visual perception reward for explicit perceptual supervision. PAPO [46] introduces an implicit perception loss based on KL divergence between distributions conditioned on original versus corrupted visual inputs, achieving significant perception error reduction without external annotations. V-Triune [47] proposes a Dynamic IoU reward for detection and grounding tasks, avoiding reward ambiguity under loose thresholds and reward sparsity under strict ones. CodePercept [7] establishes code as a perceptual medium for STEM visual understanding, showing that scaling perception consistently outperforms scaling reasoning. CapRL [14] redefines caption quality through downstream utility, rewarding captions that enable a vision-free LLM to answer visual questions, and demonstrates that RL-trained captioners can match much larger models.

## 2.3 Thinking with Python

A growing line of work explores using code execution as a reasoning tool for VLMs [48], [49]. DeepEyes [15] learns to “think with images” via RL, using the model’s own grounding capability to dynamically crop and re-examine image regions during reasoning. DeepEyesV2 [16] extends this to an agentic setting that interleaves code execution and web search within a single reasoning trajectory, employing a cold-start SFT stage followed by RL to establish robust tool-use patterns. R1-Code-Interpreter [50] trains LLMs to autonomously generate code during step-by-step reasoning via multi-stage curriculum RL. ReVPT [17] enhances VLMs ability to reason about and use visual tools through RL. ToolsRL [18] proposes a two-stage curriculum that decouples tool mastery from answer optimization. Spatial accuracy is a complementary concern in such tool-use settings, because later interactions may depend on the crops and annotations produced in earlier turns.

PERCEPTRL differs from prior work in two fundamental ways. First, at the data level, we automatically mine the base model’s own perceptual errors and compose them into verifiable QA pairs, producing training signal that is both error-agnostic and scalable. Second, at the supervision level, we combine Coordinate Injection for constructing spatially corrected training trajectories with a grounding-aware process reward that evaluates intermediate spatial fidelity during RL.

## 3 METHOD

## 3.1 Overview

PERCEPTRL operates at two complementary levels (Figure 1). Level 1 (Perception RL) systematically probes the model’s perceptual weaknesses, converts the identified errors into verifiable QA pairs, and trains single-turn perception with a rule-based outcome reward. However, its singleturn RL does not optimize iterative visual inspection, and small objects, densely packed scenes, or details requiring magnification can remain difficult to resolve from the full image alone. This motivates Level 2 (Perception Agentic RL), which trains multi-turn visual inspection through a code interpreter for cropping, resizing, annotation, and measurement. Such interactions require accurate spatial grounding: an incorrect crop or annotation can distort the evidence used in later turns and cause coordinate drift.

A single multi-agent pipeline underlies both levels, turning the base model’s own failures into training signal (Section 3.2.2): verifiable single-turn QA pairs for Level 1 and coordinate-corrected multi-turn agentic trajectories for Level 2. On top of this data, the model is trained in three sequential stages, each initialized from the previous checkpoint (Figure 1): (1) a single cold-start SFT on a mixture of the single-turn perception QA and the multi-turn CIcorrected trajectories, which provides training examples for single-turn perception QA and multi-turn tool use with CIcorrected coordinates; (2) Level-1 Perception RL, single-turn RL from the cold-start checkpoint that trains on perception QA with a rule-based outcome reward; and (3) Level-2 Perception Agentic RL, multi-turn RL that continues from the Level-1 checkpoint and additionally supervises the spatial fidelity of every intermediate step with a grounding-aware process reward. The two levels describe the single-turn and multi-turn task settings; the three stages describe the training sequence.

## 3.2 Perception RL

Current post-training is dominated by reasoning RL, which has substantially improved reasoning ability precisely because reasoning tasks admit abundant rewardable signals. Perception, in contrast, has seen far less progress in the post-training stage: its natural supervision, captions, is subjective and open-ended, and thus difficult to formulate as a verifiable reward. Perceptual errors can undermine subsequent reasoning, motivating explicit supervision of visual evidence [45]. This motivates Level 1: to provide verifiable supervision across multiple perception task families during post-training.

Training across multiple perception skills calls for structured task organization: perception includes a broad range of fine-grained abilities, such as counting, color, orientation, and depth. We therefore organize Level 1 as a three-step pipeline (Fig. 2). (i) We construct a three-level perception tree to organize fine-grained capabilities and guide model probing and data sampling. Two complementary routes support this construction: a specialized route for selected high-error domains and a general route for additional error types (Section 3.2.1). (ii) Along both routes, a multi-agent pipeline turns the base model’s own perceptual errors into deterministic, verifiable QA (Section 3.2.2). (iii) We train on this data with rule-based reward RL (Section 3.2.3).

## 3.2.1 Perception Tree Construction

Perception is not a single scalar ability: a model may read colors reliably yet systematically miscount, or localize objects well yet confuse facing directions. A structured decomposition helps identify perceptual weaknesses and organize data construction across related skills. We therefore build a three-level perception tree that progressively organizes broad aspects of visual understanding into related capability groups and concrete perceptual tasks (Figure 2). The hierarchy covers visual attributes, spatial relationships, and physical understanding, connecting high-level capability organization to the specific errors that guide training-data construction. The figure illustrates representative branches of this organization, while Table ?? in Appendix ?? provides a more detailed capability map and associated failure modes.

With the tree in place, we probe the base model on sampled tasks associated with its capabilities and examine where its answers disagree with the reference labels. The observed error patterns guide data allocation and the selection of task families for specialized construction, while the general route covers additional errors beyond these selected domains. We use the following two construction routes:

• Specialized route. For high-error nodes whose answers can be made cleanly and objectively verifiable, we build dedicated per-domain pipelines, currently for counting, color, spatial, and state, each with domain-specific verification (e.g., pixel-level color sampling, per-object box counting) and question templates.

• General route. A single domain-agnostic pipeline handles errors beyond the selected specialized domains. It discovers errors in the base model’s image descriptions and composes them into questions, providing a flexible way to sample additional task types.

Both routes share the same error-driven construction principle (Section 3.2.2); they differ only in whether verification and composition are specialized to one node or applied generically. The tree provides an organizational framework for data construction; empirical gains are assessed on the benchmark tasks reported in Section 4.

## 3.2.2 Rewardable QA Construction

The central insight shared by both routes is that perceptual errors are more informative than correct perceptions: rather than training on what the model already knows, we systematically discover what it gets wrong and build targeted data around those failures. This decomposes data construction into two sub-problems: (i) reliably detecting the model’s genuine perceptual errors, and (ii) composing those errors into QA that is both challenging (it targets an actual weakness) and verifiable (it has an unambiguous correct answer). We surface genuine errors through a threestage multi-agent protocol. (1) Dense captioning: the base model produces an exhaustive multi-dimensional caption over the categories above. (2) Agent-based verification: a VLM audits the caption claim by claim (global checks on the full image, local checks on crops with quantitative measurement, and a scan for omissions), emitting each error as a structured JSON record. (3) Cross-verification: a second, independently chosen VLM re-checks every error, and only those confirmed by both are retained, filtering out singlemodel false positives. (4) Error composition: rather than filling fixed templates, a dedicated composition agent turns confirmed errors into QA pairs end-to-end, choosing which errors to target, phrasing them into one coherent question, and pairing the correct answer with crafted distractors under the quality gates below. Because it reasons about each error configuration rather than slotting values into a schema, it adapts question format, difficulty, and distractors to the errors an image exhibits, and chooses between two strategies by how the errors relate:

![](images/38a3a6a185b903af6e607e16f850c1d06ddd4fc3bc334199118375b3be77fc39.jpg)  
Fig. 2: Perception RL: from perceptual errors to rewardable QA. (01) Perception tree construction: A hierarchica capability map and model probing reveal perceptual weaknesses, guiding specialized and general data-construction routes. Appendix ?? expands the displayed tree with additional skills and associated failure modes. (02) Rewardable QA construction: A multi-agent pipeline mines and verifies errors in the base model’s captions, then composes them into questions with unambiguous answers. MCQs combine unrelated errors, while chain QA connects correlated errors. (03) Rule-based reward for RL: The resulting QA pairs provide exact-match rewards for training the model’s visual perception. The error distribution and example errors are illustrative.

MCQ format (unrelated errors). When an image contains multiple independent perceptual errors (e.g., a color error, a counting error, and a spatial error that concern different objects), we compose them into a multiple-choice question where each option (A–H) combines multiple visual claims. Exactly one option has all claims correct; the distractors each contain one or more claims that correspond to verified errors. This format produces questions with unambiguous single-letter answers, ideal for rule-based reward.

Chain QA format (correlated errors). When errors are spatially or semantically related (e.g., multiple spatial errors involving the same group of objects), we compose them into a multi-step visual identification chain: an initial candidate set → discriminating filter → intermediate anchor → further discrimination → unique target → atomic answer. Each step in the chain requires a distinct perceptual capability, and the answer is only reachable by correctly executing all steps.

Both formats yield deterministic, verifiable answers, and because every question is grown from a genuine error the base model made, learning concentrates on its weaknesses rather than rehearsing what it already perceives correctly. This pipeline runs in the two routes of Section 3.2.1: the general route composes errors sampled from different capability pools into one hard question, while the specialized route applies the same skeleton with domain-specific verification and templates to the highest-error, cleanly-verifiable nodes (Appendix ??). Both are governed by quality gates (capability coupling across ≥3 skills, distractor resistance, anti-shortcut tests, explicit reference frames), and a final round of manual inspection screens the constructed QA pairs. Human review serves as the final quality-control stage for the constructed questions and their answer labels, following automated cross-verification. The rule-based reward used during RL then checks model outputs against these reviewed labels, separating label validation during data construction from reward computation during training.

## 3.2.3 Rule-Based Reward for RL

The verified QA pairs provide fixed answer labels, allowing perception learning to use a rule-based outcome reward. For a model answer a and its reference label $a ^ { * }$ , the reward is

$$
R _ { \mathrm { a c c } } ( a , a ^ { * } ) = { \left\{ \begin{array} { l l } { 1 , } & { { \mathrm { m a t c h } } ( a , a ^ { * } ) , } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }\tag{1}
$$

The target is an atomic value: a single letter (A–H) for MCQ, or a color, number, or direction for chain QA and counting. The matching rule checks the final answer against this label; incorrect or unparseable answers receive zero reward. Label verification and human review take place during data construction, so reward computation during RL requires no caption judge.

We optimize the policy with Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO) [51]. For each image–question input $x \overset { \vartriangle } { = } ( I , q ) .$ , we sample G responses $\{ o _ { i } \} _ { i = 1 } ^ { G }$ from the rollout policy $\pi _ { \theta _ { \mathrm { o l d } } } ( \cdot \mid x )$ . Let $a _ { i }$ be the final answer in response $o _ { i }$ and $R _ { i } = R _ { \mathrm { a c c } } ( a _ { i } , a ^ { * } )$ its outcome reward. The group-normalized advantage is

$$
\begin{array} { l } { \displaystyle \bar { R } = \frac 1 G \sum _ { j = 1 } ^ { G } R _ { j } , } \\ { \displaystyle \sigma _ { R } = \mathrm { s t d } ( R _ { 1 } , \ldots , R _ { G } ) , } \\ { \displaystyle \widehat { A } _ { i , t } = \frac { R _ { i } - \bar { R } } { \sigma _ { R } } , 1 \leq t \leq | o _ { i } | . } \end{array}\tag{2}
$$

Here, t indexes generated tokens and $\left| o _ { i } \right|$ is the response length. Because Level 1 uses an outcome reward, every token in a response receives the same advantage, determined by its answer’s correctness relative to the other responses in the group.

For token $o _ { i , t }$ with preceding response tokens $o _ { i , < t } ,$ the probability ratio and its clipped counterpart are

$$
\begin{array} { r l } & { u _ { i , t } ( \theta ) = \frac { \pi _ { \theta } \left( o _ { i , t } ~ | ~ x , o _ { i , < t } \right) } { \pi _ { \theta _ { \mathrm { o l d } } } \left( o _ { i , t } ~ | ~ x , o _ { i , < t } \right) } , } \\ & { \widetilde { u } _ { i , t } ( \theta ) = \mathrm { c l i p } \left( u _ { i , t } ( \theta ) , 1 - \varepsilon _ { \mathrm { l o w } } , 1 + \varepsilon _ { \mathrm { h i g h } } \right) . } \end{array}\tag{3}
$$

The lower and upper clipping ranges are decoupled; DAPO’s Clip-Higher strategy uses a larger upper range to allow greater probability increases for positively advantaged tokens. Writing $\begin{array} { r } { N = \sum _ { i = 1 } ^ { G } | o _ { i } | . } \end{array}$ , the policy maximizes the token-averaged objective

$$
\begin{array} { r l } & { \mathcal { T } _ { \mathrm { D A P O } } ( \theta ) = \mathbb { E } _ { \mathcal { B } _ { \mathrm { d y n } } } \Bigg [ \cfrac { 1 } { N } \sum _ { i = 1 } ^ { G } \sum _ { t = 1 } ^ { | o _ { i } | } } \\ & { \qquad \operatorname* { m i n } \Big ( u _ { i , t } ( \theta ) \widehat { A } _ { i , t } , \widetilde { u } _ { i , t } ( \theta ) \widehat { A } _ { i , t } \Big ) \Bigg ] . } \end{array}\tag{4}
$$

The expectation is over image–question inputs and their sampled response groups retained by dynamic sampling,

denoted by $B _ { \mathrm { d y n } } .$ . For the binary outcome reward, a group is retained only when

$$
0 < \sum _ { i = 1 } ^ { G } R _ { i } < G .\tag{5}
$$

Groups whose responses are all correct or all incorrect are discarded, and sampling continues until the training batch is filled with informative groups. This ensures nonzero reward variance for Equation 2. The normalization by N in Equation 4 averages over generated tokens rather than assigning equal weight to responses of different lengths.

## 3.3 Perception Agentic RL

Level 2 targets difficult perceptual queries involving small objects, cluttered scenes, or subtle color and state differences, for which a single view may not provide sufficient evidence. The model therefore uses a code interpreter to crop, resize, annotate, and measure the image over multiple turns before answering. A grounding error occurs when a crop box or annotation point misses its intended target. If later turns rely on the resulting observation, this error can alter subsequent localization and counting decisions; we refer to the propagation and accumulation of such spatial misalignment as coordinate drift. For example, misplaced marks in dense counting can cause later turns to revisit counted objects or omit unmarked ones (Fig. 3, panel 01). Final-answer correctness alone does not identify which intermediate spatial operations were accurate. We address this supervision gap through two components: Coordinate Injection constructs trajectories with corrected spatial coordinates for cold-start SFT, and a grounding-aware process reward scores intermediate spatial fidelity during Level-2 RL.

## 3.3.1 Coordinate Injection

Given an image I and a question $Q ,$ the agent alternates between generating a code block and observing its execution result. Let $s _ { t }$ denote the accumulated image-question prompt, reasoning context, and interaction history available before turn t. The multi-turn interaction is

$$
\begin{array} { r l } & { s _ { 1 } = ( I , Q ) , } \\ & { c _ { t } \sim \pi _ { \theta } ( \cdot \mid s _ { t } ) , } \\ & { o _ { t } = \mathrm { E x e c } ( I , c _ { t } ) , } \\ & { s _ { t + 1 } = s _ { t } \oplus ( c _ { t } , o _ { t } ) , } \end{array}\tag{6}
$$

where $\pi _ { \theta }$ is the model policy, Exec is the code interpreter, $o _ { t }$ is the resulting visual observation, and ⊕ appends the generated code and observation to the context. After $T$ tool turns, the model generates its final answer from $s _ { T + 1 }$ . The code-observation sequence is

$$
\tau = ( c _ { 1 } , o _ { 1 } , \dots , c _ { T } , o _ { T } ) .\tag{7}
$$

Each turn operates on the original image $I ;$ spatial operations, coordinate correction, and fidelity scoring use the coordinate system of I.

Equation 6 shows how a grounding error can enter the history: an off-target crop or annotation changes $o _ { t } ,$ , which becomes part of the context used to generate the next code block. To reduce this source of coordinate drift in the training trajectories, Coordinate Injection (CI) corrects the spatial coordinates and re-executes the code before appending the observation. Thus, subsequent generation uses the corrected observation rather than the output of the inaccurate operation. Writing the corrected context as $s _ { t } ^ { * }$ , with $s _ { 1 } ^ { * } = ( I , { \hat { Q } } )$ , CI applies the following three steps to each turn that contains spatial operations (Fig. 3, panel 02).

![](images/c16d80cb5dcf4f42791e0547136c40f71f23689994bc32daaf7ab468da89ee2f.jpg)  
Fig. 3: Perception Agentic RL: correcting and supervising spatial grounding during multi-turn visual inspection. (01) Coordinate drift. A misplaced annotation changes the visual evidence available to later turns. Subsequent spatial operations can inherit this error, leading to repeated or missed counts. (02) Coordinate Injection (CI). During trajectory construction, CI identifies the intended target of each spatial operation, corrects its coordinates using a VLM judge or the LocateAnything detector, and re-executes the code before the next turn. The corrected trajectories provide cold-start SFT data. (03) Grounding-aware process reward. During RL, a fidelity judge evaluates the predicted boxes and points against their intended targets in the original image. The resulting per-turn rewards complement the final-answer reward and supervise intermediate spatial accuracy (Equation 17).

1) Target extraction. For the current code block $c _ { t } ,$ a VLM reads ${ \bf { \bar { \rho } } } _ { I , \quad }$ the context $s _ { t } ^ { * }$ , and the code, and produces

$$
\mathcal { T } _ { t } = \mathrm { E x t r a c t } ( I , s _ { t } ^ { * } , c _ { t } ) = \{ ( b _ { t , k } , d _ { t , k } ) \} _ { k = 1 } ^ { K _ { t } } .\tag{8}
$$

Here, $K _ { t }$ is the number of spatial operations in $c _ { t } , ~ b _ { t , k }$ is the coordinate used by operation $k ,$ and $d _ { t , k }$ describes its intended target. The extraction prompt requires $b _ { t , k }$ to be copied from the corresponding operation in the code. The #target declaration serializes $\vec { \mathcal { T } _ { t } }$ as a JSON list with one record per operation. For example, the following schema shows a box operation followed by a point operation; the underbraces identify the mathematical quantity represented by each field:

![](images/04defac7c6a523304fd600c6a511702fb18aa2e685a0deb6cfe1723de6bc0baa.jpg)

The subscript t identifies the turn, and k identifies the operation within that turn. The bbox and point fields are two representations of $b _ { t , k } \mathrm { : }$ : a box has coordinates $[ x _ { 1 } , y _ { 1 } , x _ { 2 } , y _ { 2 } ]$ whereas a point has coordinates [x, y]. The desc field contains $d _ { t , k } ;$ the symbolic subscripts and underbraces are explanatory annotations and are not part of the JSON.

2) Grounding correction. For each extracted operation, a grounding backend uses the intended-target description to obtain a corrected reference coordinate:

$$
\begin{array} { r } { b _ { t , k } ^ { * } = \mathrm { G r o u n d } ( b _ { t , k } , d _ { t , k } , I ) . } \end{array}\tag{9}
$$

By default, a VLM judge examines the predicted box or point on a coordinate-grid overlay of I and adjusts its coordinates using $d _ { t , k }$ . This backend handles most domains, including subtle attributes or functional states that are difficult to express as detection queries. For domains with many precisely localizable targets, such as dense counting, we instead use an expert detector $( e . g .$ , LocateAnything [52]), which localizes the target from I and $d _ { t , k }$ without using the predicted coordinate. The selected backend provides $b _ { t , k } ^ { * }$ for the subsequent code correction.

3) Code rewriting and re-execution. CI replaces the coordinates associated with each operation, executes the corrected block, and appends its result to the corrected context:

$$
\begin{array} { r l } & { c _ { t } ^ { * } = \mathrm { R e p l a c e } \big ( c _ { t } , \{ b _ { t , k } \mapsto b _ { t , k } ^ { * } \} _ { k = 1 } ^ { K _ { t } } \big ) , } \\ & { o _ { t } ^ { * } = \mathrm { E x e c } ( I , c _ { t } ^ { * } ) , } \\ & { s _ { t + 1 } ^ { * } = s _ { t } ^ { * } \oplus ( c _ { t } ^ { * } , o _ { t } ^ { * } ) . } \end{array}\tag{10}
$$

The model generates the next code block from $s _ { t + 1 } ^ { * } ,$ , and CI repeats the same procedure if that block contains spatial

operations. For a turn with no spatial operation, the code is executed and appended without coordinate replacement. After all $T$ turns, this procedure yields

$$
\tau ^ { * } = ( c _ { 1 } ^ { * } , o _ { 1 } ^ { * } , \ldots , c _ { T } ^ { * } , o _ { T } ^ { * } ) ,\tag{11}
$$

where each grounding code block uses the corrected coordinates and each observation is obtained by executing its corresponding block. We also update the #target records to contain $( b _ { t , k } ^ { * } , d _ { t , k } )$ and retain them as a comment on the first line of each grounding code block. These declarations couple an operation’s coordinates to its intended-target description and can be parsed without analyzing the plotting code. The resulting trajectories are included in the shared cold-start SFT data, teaching the model both multi-turn tool use and the declaration format later used for grounding rewards.

## 3.3.2 Grounding-Aware Reward for RL

The CI-corrected SFT trajectories teach the model to place a #target declaration at the beginning of each grounding code block. During Level-2 RL, the model follows the interaction in Equation $^ { 6 , }$ and each declaration is required to record the coordinates used by its corresponding spatial operation together with a description of the intended target. The reward pipeline parses these records, renders the declared boxes or points on $I ,$ and evaluates whether they land on the described targets. This provides a grounding signal for individual turns in addition to final-answer correctness.

As in CI, turn t contains $K _ { t }$ spatial operations, each indexed by $( t , k )$ and represented by $( b _ { t , k } , d _ { t , k } )$ . A nongrounding turn has $K _ { t } \ = \ 0$ . The set of turns eligible for grounding rewards is

$$
\mathcal { G } = \{ t \in \{ 1 , \dots , T \} : K _ { t } > 0 \} .\tag{12}
$$

For each operation, the scorer assesses whether the declared box or point lands on the target named by $d _ { t , k }$ . To avoid a separate judge call for each operation, all spatial operations in a trajectory are scored together using one composite image.

For operation $( t , k )$ , we overlay $b _ { t , k }$ on the original image and crop a padded window around it. The resulting patch enlarges the annotation while retaining local context and is labeled with its index $( t , k )$ and description $d _ { t , k }$ . We tile these patches with the unannotated original image as a global reference:

$$
\begin{array} { r } { \begin{array} { r l } & { P _ { t , k } = \mathrm { P a t c h } ( I , b _ { t , k } , d _ { t , k } ) , } \\ & { M = \mathrm { T i l e } \big ( I , \{ P _ { t , k } \} _ { t \in \mathcal { G } , 1 \leq k \leq K _ { t } } \big ) . } \end{array} } \end{array}\tag{13}
$$

The fidelity judge reads M in one call and returns a score $g _ { t , k } \in [ 0 , \dot { 1 } ]$ for each indexed patch. A score of 1 indicates a box that tightly encloses the described target or a point placed precisely on it; a score of 0 indicates an off-target annotation. Scores are returned as a JSON list keyed by the operation index, for example:

$$
\begin{array} { r l } & { \qquad [ \left\{ { \bf " i d " : } { \bf u } : { \bf " } \in 3 _ { - } { \bf k } 1 { \bf " } , { \bf " s c o r e " } : 0 . 8 2 \right\} , } \\ & { \qquad \left\{ { \bf " i d " : } { \bf " } \in 3 _ { - } { \bf k } 2 { \bf " } , { \bf " s c o r e " } : 0 . 1 5 \right\} , \quad \mathrm { ~ . ~ . ~ . ~ } ] } \end{array}
$$

The example values illustrate the output format. Matching each returned index to its declaration recovers the score for the corresponding operation.

Each grounding turn is summarized by its least accurate operation:

$$
G _ { t } = \operatorname* { m i n } _ { 1 \le k \le K _ { t } } g _ { t , k } ,\tag{14}
$$

so that correct operations cannot mask an off-target box or point in the same turn. For a trajectory with at least one grounding turn, its mean grounding fidelity is

$$
\bar { G } = \frac { 1 } { | \mathcal { G } | } \sum _ { t \in \mathcal { G } } G _ { t } .\tag{15}
$$

The experimental metric averages $\bar { G }$ over trajectories containing at least one grounding turn.

For reward computation, we center the grounding score at $\frac { 1 } { 2 }$ :

$$
R _ { \operatorname { g r o u n d } } ( t ) = G _ { t } - { \textstyle { \frac { 1 } { 2 } } } , \qquad t \in \mathcal { G } .\tag{16}
$$

Scores above this threshold contribute positively and scores below it contribute negatively. Each turn also receives the tool-use shaping term $\rho _ { t } ,$ , and the combined reward is clipped:

$$
r _ { t } = \left\{ \begin{array} { l l } { \mathrm { c l i p } \big ( R _ { \mathrm { g r o u n d } } ( t ) + \rho _ { t } , - 1 , + 1 \big ) , } & { t \in \mathcal { G } , } \\ { \mathrm { c l i p } \big ( \rho _ { t } , - 1 , + 1 \big ) , } & { t \notin \mathcal { G } . } \end{array} \right.\tag{17}
$$

Thus, non-grounding turns receive shaping rewards without a spatial-fidelity score. The shaping term combines code-format validity, penalties for redundant display calls, runtime errors and cross-turn code duplication, and a bonus for repairing the preceding turn’s runtime error.

## 3.3.3 Token-Level Credit Assignment

Both terms are assigned at the token level: the answeraccuracy reward is broadcast to every token of the trajectory, whereas each turn’s reward $r _ { t }$ reaches only the tokens $\mathcal { T } _ { t }$ produced in that turn (the $\mathcal { T } _ { t }$ partition the trajectory). The reward placed on token i is

$$
\begin{array} { r l } { R _ { \mathrm { t o t a l } } ( \tau ) _ { i } = } & { \underbrace { R _ { \mathrm { a c c } } ( a , a ^ { * } ) } _ { \mathrm { a n s w e r ~ a c c u r a c y ~ ( a l l ~ t o k e n s ) } } } \\ { \displaystyle } & { + \underbrace { \sum _ { t = 1 } ^ { T } r _ { t } \ \mathbf { 1 } [ i \in \mathcal { I } _ { t } ] } _ { \mathrm { i n t e r m e d i a t e - t u r n ~ s u p e r v i s i o n } } \ , } \end{array}\tag{18}
$$

where a is the final answer and $R _ { \mathrm { a c c } } ( a , a ^ { * } )$ is the Level-1 exact-match reward of Equation 1, so each token receives the outcome reward plus the reward of the turn that produced it. The two terms supply credit at complementary granularities. The outcome term $R _ { \mathrm { a c c } }$ is constant across all tokens of a trajectory, so it provides only across-trajectory credit: under the group-normalized advantage it separates a better rollout from a worse one, but is blind to which turn inside a rollout earned that outcome. The per-turn term $r _ { t }$ instead varies from turn to turn, providing within-trajectory credit: a well-grounded turn is advantaged over a poorly grounded turn in the same trajectory, even when both share the same final answer. Grounding fidelity is invisible to the across-trajectory term alone, so it is precisely this withintrajectory signal that lets RL reinforce accurate intermediate grounding rather than leave it unconstrained.

TABLE 1: Main results on eight perception benchmarks, scored by accuracy (%) under each benchmark’s official protocol, comparing PERCEPTRL against proprietary and open frontier VLMs. Avg. averages over the eight reported benchmarks; best per column in bold.
<table><tr><td>Method</td><td>RealWorldQA</td><td>MMStar</td><td>BLINK</td><td>TreeBench</td><td>ERQA</td><td>RefSpatial</td><td>CountQA</td><td>RealWorldCounting</td><td>Avg.</td></tr><tr><td>Frontier VLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3.5-Plus [53]</td><td>82.2</td><td>82.2</td><td>73.4</td><td>62.2</td><td>62.7</td><td>69.7</td><td>71.7</td><td>35.4</td><td>67.4</td></tr><tr><td>Gemini-3-Flash [54]</td><td>84.6</td><td>81.5</td><td>76.3</td><td>58.3</td><td>65.2</td><td>63.4</td><td>70.4</td><td>34.6</td><td>66.8</td></tr><tr><td>Doubao-Seed-2.0-mini [55]</td><td>82.2</td><td>77.5</td><td>73.8</td><td>56.3</td><td>52.2</td><td>63.4</td><td>70.2</td><td>37.9</td><td>64.2</td></tr><tr><td>Qwen3.5-Flash [53]</td><td>72.8</td><td>74.9</td><td>66.3</td><td>58.8</td><td>61.3</td><td>57.9</td><td>67.8</td><td>22.0</td><td>60.2</td></tr><tr><td>Kimi-K2.6 [56]</td><td>80.0</td><td>79.2</td><td>74.4</td><td>55.6</td><td>48.8</td><td>43.3</td><td>43.2</td><td>12.9</td><td>54.7</td></tr><tr><td>Kimi-K2.5 [57]</td><td>80.9</td><td>78.5</td><td>75.9</td><td>52.6</td><td>51.2</td><td>39.5</td><td>46.5</td><td>13.4</td><td>54.8</td></tr><tr><td>Claude-Opus-4.8 [58]</td><td>77.1</td><td>75.4</td><td>70.2</td><td>58.3</td><td>51.5</td><td>41.9</td><td>34.2</td><td>17.8</td><td>53.3</td></tr><tr><td>Grok-4.5 [59]</td><td>74.5</td><td>75.0</td><td>73.9</td><td>59.5</td><td>61.0</td><td>12.3</td><td>45.4</td><td>19.0</td><td>52.6</td></tr><tr><td>Gemini-2.5-Pro [54]</td><td>78.2</td><td>77.9</td><td>67.0</td><td>54.8</td><td>58.5</td><td>27.3</td><td>40.5</td><td>19.2</td><td>52.9</td></tr><tr><td>Claude-Opus-4.7 [60]</td><td>77.6</td><td>77.1</td><td>70.3</td><td>57.3</td><td>51.5</td><td>37.2</td><td>33.4</td><td>13.4</td><td>52.2</td></tr><tr><td>DeepSeek-V4-Flash-Vision [61]</td><td>77.1</td><td>72.7</td><td>65.1</td><td>58.0</td><td>58.2</td><td>20.6</td><td>49.3</td><td>11.2</td><td>51.5</td></tr><tr><td>Grok-4.3 [62]</td><td>79.3</td><td>74.0</td><td>69.2</td><td>53.1</td><td>48.0</td><td>16.3</td><td>35.9</td><td>14.9</td><td>48.8</td></tr><tr><td>Claude-Opus-4.6 [63]</td><td>75.4</td><td>74.3</td><td>68.8</td><td>53.3</td><td>46.0</td><td>5.8</td><td>17.3</td><td>13.8</td><td>44.3</td></tr><tr><td>PERCEPTRL (on Qwen3.6-Flash, 35A3)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base (Qwen3.6-Flash) [64]</td><td>82.9</td><td>78.7</td><td>68.8</td><td>57.8</td><td>58.2</td><td>56.3</td><td>66.8</td><td>24.3</td><td>61.7</td></tr><tr><td>+ SFT cold-start</td><td>83.3</td><td>77.9</td><td>68.6</td><td>56.8</td><td>53.8</td><td>67.7</td><td>57.9</td><td>29.3</td><td>61.9</td></tr><tr><td>+ Perception RL (Level 1)</td><td>85.2</td><td>82.6</td><td>75.1</td><td>64.7</td><td>62.8</td><td>73.8</td><td>71.9</td><td>42.1</td><td>69.8</td></tr></table>

## 4 EXPERIMENTS

## 4.1 Experimental Setup

Base model and training. We build PERCEPTRL on Qwen3.6-Flash [64] as the base VLM, using the 35A3 mixture-of-experts variant (∼35B total / ∼3B active parameters), and train it in three stages, each initialized from the previous stage’s checkpoint: (i) a cold-start SFT stage on 57,678 training examples comprising single-turn perception QA and CI-corrected multi-turn agentic trajectories from our pipeline, which jointly teaches the single-turn perception-answer format and the multi-turn perceptionwith-grounding format together with a drift-free multi-turn reasoning pattern; (ii) Perception RL (Level 1), single-turn RL over 36,445 perception QA queries spanning multiple domains (Table 3), scored by a rule-based outcome reward; and (iii) Perception Agentic RL (Level 2), multi-turn RL over the 19,810-query Agentic QA subset, scored by the grounding-aware process reward. In both RL stages the model receives only queries and generates its own rollouts, which the reward then scores; no pre-collected trajectory is used as a supervision target. Both RL stages use DAPO [51], and training is run on 64×A100 80GB. The two stages differ in their optimization budget: Level 1 (Perception RL) uses a learning rate of $1 \times 1 0 ^ { - 6 }$ , a batch size of 128, a DAPO group size of $G = 8 ,$ a maximum sequence length of 32K tokens, and 1 epoch, whereas Level 2 (Agentic RL) uses a learning rate of $5 \times 1 0 ^ { - 7 }$ , a batch size of 64, a group size of $G \ = \ 8 ,$ , a maximum sequence length of 128K tokens, and 1 epoch. During Level-2 rollouts the model drives a sandboxed Python code interpreter (crop, resize, annotate, and pixel/box measurement) for up to T = 64 turns per trajectory.

From single-turn to agentic QA. The two QA pools in Table 3 are constructed sequentially. We first train the Level-1 model with Perception RL on the full Single-turn QA pool of 36,445 pairs. After Level-1 training, we generate eight rollouts from the trained model for every query in this same pool and compute its empirical accuracy as $c _ { i } / 8 ,$ where $c _ { i }$ is the number of correct responses. We retain queries satisfying $c _ { i } / 8 ~ < ~ 5 / 8$ , equivalently at most four correct responses out of eight, to form the 19,810-pair Agentic QA pool used for Level-2 RL. This step selects existing QA pairs without adding new ones, so Agentic QA is a strict subset of Singleturn QA; their sizes are not additive.

Benchmarks. We evaluate on eight perception benchmarks, grouped by the capability they stress: general real-world VQA (RealWorldQA [65], MMStar [66]), multi-image and fine-grained grounded perception (BLINK [67], TreeBench [68]), embodied and spatial perception (ERQA [69], which targets embodied reasoning, and RefSpatial-Bench [70] for referring spatial grounding), and object counting (CountQA [71] and real-world RealWorldCounting). None of these suites is used when constructing our training data, so they measure generalization rather than in-distribution fit.

RefSpatial evaluation protocol. RefSpatial-Bench is evaluated using the official point-in-mask criterion after converting each model’s predicted point to image-space [x, y] coordinates. Gemini predictions use normalized [y, x] coordinates in the range [0, 1000]; we reorder the axes and scale each coordinate by the corresponding image dimension. Claude-Opus-4.6/4.7/4.8 predictions are interpreted directly as pixel coordinates. For Grok-4.3 and Kimi-K2.5/K2.6, points in [0, 1] are interpreted as fractional image coordinates, and otherwise as [0, 1000] normalized coordinates; both representations are mapped to pixels before evaluation. Table 1 reports the resulting accuracies. The remaining models use the benchmark’s standard [0, 1000] [x, y] convention.

## 4.2 Main Results

Table 1 compares frontier VLMs with the successive training stages of PERCEPTRL on the Qwen3.6-Flash (35A3) backbone. The frontier models use single-turn inference, while the final Level-2 model uses multi-turn tool-augmented inference; Table 2 compares models under a shared tool interface. The shared SFT cold-start initializes the perceptionanswer and grounded tool-use formats, obtaining 61.9% mean accuracy compared with 61.7% for the base model. Perception RL then improves every benchmark relative to this checkpoint and raises mean accuracy to 69.8%, a gain of 7.9 percentage points over the SFT model and 8.1 points over the base model. These improvements are achieved in the single-turn setting, before introducing Level-2 tool interactions.

TABLE 2: Comparison of multi-turn tool-use capabilities across frontier VLMs. Cold-start SFT denotes the shared SFT checkpoint before Level-1 and Level-2 RL; its single-turn scores are taken from Table 1. Accuracy is reported in %. Avg is the mean over the seven benchmarks shown. Best per column in bold.
<table><tr><td>Model (agentic)</td><td>RealWorldQA</td><td>MMStar</td><td>BLINK</td><td>TreeBench</td><td>ERQA</td><td>CountQA</td><td>RealWorldCounting</td><td>Avg</td></tr><tr><td colspan="9">Frontier VLMs + code interpreter</td></tr><tr><td>Gemini-3-Flash [54]</td><td>84.4</td><td>82.8</td><td>76.9</td><td>62.2</td><td>71.8</td><td>53.7</td><td>33.2</td><td>66.4</td></tr><tr><td>Doubao-Seed-2.0-mini [55]</td><td>78.3</td><td>77.7</td><td>73.1</td><td>54.6</td><td>54.0</td><td>65.8</td><td>33.7</td><td>62.5</td></tr><tr><td>Qwen3.5-Flash [53]</td><td>74.6</td><td>77.9</td><td>44.6</td><td>56.3</td><td>54.0</td><td>44.1</td><td>17.7</td><td>52.7</td></tr><tr><td>Kimi-K2.6 [56]</td><td>79.1</td><td>78.5</td><td>73.7</td><td>54.8</td><td>62.3</td><td>54.1</td><td>25.0</td><td>61.1</td></tr><tr><td>Claude-Opus-4.8 [58]</td><td>80.7</td><td>79.9</td><td>76.6</td><td>59.0</td><td>61.3</td><td>53.1</td><td>29.2</td><td>62.8</td></tr><tr><td>Grok-4.5 [59]</td><td>85.0</td><td>81.4</td><td>73.2</td><td>64.0</td><td>64.0</td><td>54.3</td><td>23.9</td><td>63.7</td></tr><tr><td>DeepSeek-V4-Flash-Vision [61]</td><td>79.6</td><td>79.3</td><td>75.4</td><td>62.5</td><td>63.7</td><td>58.4</td><td>28.2</td><td>63.9</td></tr><tr><td colspan="9">Our models + code interpreter</td></tr><tr><td>Baseline [64]</td><td>82.4</td><td>78.7</td><td>49.8</td><td>56.5</td><td>55.0</td><td>49.7</td><td>19.4</td><td>55.9</td></tr><tr><td>Cold-start SFT</td><td>83.3</td><td>77.9</td><td>68.6</td><td>56.8</td><td>53.8</td><td>57.9</td><td>29.3</td><td>61.1</td></tr><tr><td>Perception Agentic RL (Ours)</td><td>87.1</td><td>79.1</td><td>76.4</td><td>68.6</td><td>62.2</td><td>73.9</td><td>53.9</td><td>71.6</td></tr></table>

TABLE 3: Perception training QA across domains. Singleturn QA is the full pool used for Perception RL (Level 1). Agentic QA is a strict subset selected from this pool after Level-1 training: the trained model generates eight rollouts per query, and queries with empirical accuracy $\textless 5 / 8$ are retained for Perception Agentic RL (Level 2).
<table><tr><td>Route</td><td>Domain</td><td>Single-turn QA</td><td>Agentic QA</td></tr><tr><td rowspan="5">Specialized</td><td>Counting</td><td>14,109</td><td>9,783</td></tr><tr><td>Color</td><td>3,710</td><td>377</td></tr><tr><td>Spatial</td><td>3,965</td><td>1,150</td></tr><tr><td>State</td><td>3,331</td><td>2,483</td></tr><tr><td>General Perception</td><td>11,330</td><td>6,017</td></tr><tr><td>General Total</td><td></td><td>36,445</td><td>19,810</td></tr></table>

The largest gains over the SFT checkpoint are on CountQA (57.9% to 71.9%), RealWorldCounting (29.3% to 42.1%), and ERQA (53.8% to 62.8%), corresponding to improvements of 14.0, 12.8, and 9.0 percentage points. The improvements also extend to fine-grained and spatial perception: TreeBench increases by 7.9 points, BLINK by 6.5 points, and RefSpatial by 6.1 points. The Level-1 checkpoint attains 73.8% on RefSpatial, the highest accuracy among the models in Table 1. Gains on RealWorldQA and MMStar further show that the improvement extends beyond counting and localization.

The final Level-2 model reaches 71.6% mean accuracy under the shared tool-use protocol and achieves the best reported results on RealWorldQA, TreeBench, and RealWorld-Counting in Table 2. RealWorldCounting reaches 53.9%, providing the largest improvement over Level 1. Compared with Level 1, scores are lower on MMStar, ERQA, and RefSpatial.

Multi-turn tool-use capabilities. Table 2 compares frontier VLMs with the base model and PERCEPTRL on seven perception benchmarks. The frontier VLMs, the base model, and PERCEPTRL run in the same sandboxed codeinterpreter loop (crop/resize/annotate/measure), capped at

T = 64 turns and a 128K-token budget, with the same tool prompt and official per-benchmark scoring; only the model varies.

Under the shared protocol, PERCEPTRL achieves the highest mean accuracy of 71.6%, exceeding the strongest frontier baseline, Gemini-3-Flash, by 5.2 percentage points. It leads on RealWorldQA, TreeBench, CountQA, and Real-WorldCounting, with its largest advantage on RealWorld-Counting. Performance remains task-dependent: Gemini-3- Flash leads on MMStar, BLINK, and ERQA.

Under the same tool interface, the base model attains 55.9% mean accuracy. The full training pipeline improves the reported mean accuracy by 15.7 percentage points relative to this baseline, with the tool interface held fixed.

## 5 ABLATION STUDIES

## 5.1 Coverage of the Two Construction Routes

Level 1 builds data along two routes (Section 3.2.1): a specialized route with per-domain pipelines for the high-error nodes (counting, color, spatial, state) and a general domainagnostic route for the long tail. To assess their respective contributions, we compare three Level-1 RL variants sharing the same SFT cold-start, RL recipe, and total query budget, differing only in which route supplies the data: specializedonly, general-only, and both.

As shown in Table 4, the specialized route outperforms the general route on TreeBench, ERQA, RefSpatial, CountQA, and RealWorldCounting, with advantages of 6.7 and 6.2 percentage points on the two counting benchmarks, respectively. The general route achieves higher accuracy on RealWorldQA, MMStar, and BLINK, indicating a complementary benefit from broader domain coverage. Combining both routes yields the highest accuracy on all eight benchmarks, including 71.9% on CountQA and 42.1% on Real-WorldCounting. These results support the complementary contributions of domain-specific construction and general coverage under a matched data budget.

## 5.2 Comparison with Specialized Tool-Use Models

Table 7 compares specialized visual tool-use models on V\* [72], HR-Bench [73], MMStar, and TreeBench. The published V\*, HR-Bench, and TreeBench scores are taken from the DeepEyesV2 study [16]; MMStar scores are taken from the ReVPT and Thyme papers [17], [74]. Published results retain the evaluation settings of their respective sources; the base model with a code interpreter and PERCEPTRL use the protocol described in Section 4.1. Dashes indicate unreported scores.

TABLE 4: Coverage of the two Level-1 construction routes on the Qwen3.6-Flash (35A3) backbone. All rows share the same SFT cold-start and RL recipe under a matched total query budget, differing only in which route supplies the RL data; Both is our full Perception RL. Best per column in bold.
<table><tr><td>Method</td><td>RealWorldQA</td><td>MMStar</td><td>BLINK</td><td>TreeBench</td><td>ERQA</td><td>RefSpatial</td><td>CountQA</td><td>RealWorldCounting</td></tr><tr><td>SFT cold-start</td><td>83.3</td><td>77.9</td><td>68.6</td><td>56.8</td><td>53.8</td><td>67.7</td><td>57.9</td><td>29.3</td></tr><tr><td>+ Specialized route only</td><td>83.6</td><td>78.9</td><td>72.0</td><td>63.2</td><td>61.0</td><td>72.4</td><td>70.2</td><td>40.0</td></tr><tr><td>+ Ġeneral route only</td><td>84.8</td><td>81.7</td><td>73.6</td><td>59.5</td><td>57.2</td><td>69.6</td><td>63.5</td><td>33.8</td></tr><tr><td>+ Both routes (Ours)</td><td>85.2</td><td>82.6</td><td>75.1</td><td>64.7</td><td>62.8</td><td>73.8</td><td>71.9</td><td>42.1</td></tr></table>

TABLE 5: Effect of inference-time coordinate correction on final-answer accuracy using Gemini-3.5-Flash (no training). + with Python acts on uncorrected coordinates; Ours applies Coordinate-Injection correction at inference. Parentheses show gains over the baseline; best per column in bold.
<table><tr><td>Setting</td><td>RealWorldQA</td><td>TreeBench</td><td>PerceptionBench</td><td>RealWorldCounting</td></tr><tr><td>Gemini-3.5-Flash (baseline)</td><td>85.4</td><td>60.5</td><td>69.4</td><td>33.0</td></tr><tr><td>+ with Python (uncorrected coordinates)</td><td>86.0 (+0.6)</td><td>67.2 (+6.7)</td><td>70.0 (+0.6)</td><td>37.4 (+4.4)</td></tr><tr><td>+ with Python (Ours)</td><td>87.8 (+2.4)</td><td>68.9 (+8.4)</td><td>74.4 (+5.0)</td><td>45.8 (+12.8)</td></tr></table>

TABLE 6: Grounding fidelity G<sup>¯</sup> during Level-2 RL with outcome-only or grounding-aware process reward. Higher is better.
<table><tr><td>Step</td><td>0</td><td>20</td><td>40</td><td>60</td><td>80</td><td>100</td><td>120</td><td>140</td><td>160</td><td>180</td><td>200</td></tr><tr><td>Outcome-only  $( \hat { G } )$ </td><td>0.55</td><td>0.57</td><td>0.63</td><td>0.59</td><td>0.66</td><td>0.58</td><td>0.64</td><td>0.60</td><td>0.66</td><td>0.59</td><td>0.63</td></tr><tr><td>+ Grounding-aware process reward (Ours, Ġ)</td><td>0.55</td><td>0.61</td><td>0.66</td><td>0.70</td><td>0.73</td><td>0.76</td><td>0.73</td><td>0.80</td><td>0.81</td><td>0.84</td><td>0.81</td></tr></table>

TABLE 7: Comparison with specialized visual tool-use models on perception benchmarks. Accuracy is reported in $\%$
<table><tr><td rowspan="2">Model</td><td rowspan="2"> $\mathbf { V } ^ { * }$ </td><td colspan="2">HR-Bench</td><td rowspan="2">MMStar</td><td rowspan="2">TreeBench</td></tr><tr><td>4K</td><td>8K</td></tr><tr><td>DeepEyes-7B [15]</td><td>85.6</td><td>75.1</td><td>72.6</td><td></td><td>37.5</td></tr><tr><td>DeepEyesV2-7B [16]</td><td>81.8</td><td>77.9</td><td>73.8</td><td></td><td>42.5</td></tr><tr><td>Pixel-Reasoner-7B [75]</td><td>84.3</td><td>74.0</td><td>66.9</td><td></td><td>39.0</td></tr><tr><td>ReVPT-3B [17]</td><td></td><td></td><td></td><td>53.9</td><td></td></tr><tr><td>ReVPT-7B [17]</td><td></td><td></td><td></td><td>61.1</td><td></td></tr><tr><td>Thyme-7B [74]</td><td>82.2</td><td>77.0</td><td>72.0</td><td>65.9</td><td></td></tr><tr><td>Baseline [64]</td><td>90.6</td><td>80.9</td><td>74.9</td><td>78.7</td><td>56.5</td></tr><tr><td>PERCEPTRL (Ours)</td><td>98.4</td><td>89.1</td><td>85.3</td><td>79.1</td><td>68.6</td></tr></table>

## 5.3 The Effectiveness of Coordinate Injections

Table 5 evaluates inference-time Coordinate Injection on Gemini-3.5-Flash without updating its weights. Adding a code interpreter improves accuracy on several benchmarks. Applying CI further improves accuracy across all four evaluated benchmarks; on RealWorldCounting, accuracy increases from 37.4% with uncorrected coordinates to 45.8% with CI. These results support the value of correcting spatial operations during tool use and motivate the coordinatecorrected trajectories and spatial supervision used in Level 2 (Sections 3.3.1 and 3.3.2).

TABLE 8: Effect of the grounding-aware process reward at Level 2, on the Qwen3.6-Flash (35A3) backbone. Accuracy (%); best per column in bold.
<table><tr><td>Level-2 reward</td><td>RealWorldQA</td><td>BLINK</td><td>TreeBench</td><td>RealWorld Counting</td></tr><tr><td>Outcome-only  $\left( R _ { \mathrm { a c c } } \right)$ </td><td>85.1</td><td>73.4</td><td>63.5</td><td>45.5</td></tr><tr><td>+ Process reward (Ours)</td><td>87.1</td><td>76.4</td><td>68.6</td><td>53.9</td></tr></table>

## 5.4 The Effectiveness of Grounding-Aware Process Reward

We examine whether grounding-aware supervision improves spatial fidelity and final-answer accuracy during Level-2 RL after the Coordinate Injection cold-start. The two variants share the same CI-based SFT initialization, tool interface, and optimizer: outcome-only optimizes final-answer correctness $( { \bar { R } _ { \mathrm { a c c } } } )$ , whereas our $f u l l$ reward additionally incorporates per-turn spatial fidelity $( R _ { \mathrm { g r o u n d } } ,$ , Equation 18).

Table 6 tracks grounding fidelity during training. Both variants start at $\bar { G } \ = \ 0 . 5 5$ . At step 200, the full-reward variant reaches 0.81, compared with 0.63 for outcome-only RL. Although both series fluctuate, the full-reward variant maintains higher fidelity at every reported checkpoint after initialization, supporting the use of intermediate grounding supervision during agentic optimization.

Table 8 reports accuracy on RealWorldQA, BLINK, TreeBench, and RealWorldCounting. The full reward improves accuracy over outcome-only RL on all four benchmarks, by 2.0, 3.0, 5.1, and 8.4 percentage points, respectively.

## 6 CONCLUSION

We presented PERCEPTRL, a framework for improving visual perception in MLLMs during the post-training stage. By converting the model’s perceptual errors into verifiable QA and combining CI-corrected trajectories with groundingaware process reward, the framework provides training signals for both single-turn perception and multi-turn spatial grounding. The process-reward ablation shows improved grounding fidelity during training and final-answer accuracy gains across all four evaluated benchmarks. Our two-level approach, Perception RL for single-turn QA and Perception Agentic RL for multi-turn tool use, improves aggregate performance across eight perception benchmarks, demonstrating that visual perception can be effectively optimized in post-training. Future directions include extending to video perception with temporal consistency constraints, learning the grounding model jointly with the VLM, and applying CI to other agentic domains such as GUI interaction.

## REFERENCES

[1] M. Wu, J. Li, J. Ji, F. Hao, X. Sun, and R. Ji, “Evaluating and mitigating relationship hallucinations in large vision-language models,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 48, no. 6, pp. 6332–6346, 2026.

[2] J. Yang, M. Jiang, and Q. Zhao, “Defying distractions in multimodal tasks: A novel benchmark for large vision-language models,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 48, no. 6, pp. 6314– 6331, 2026.

[3] S. Bai et al., “Qwen2.5-VL technical report,” 2025, arXiv:2502.13923.

[4] B. Li et al., “LLaVA-OneVision: Easy visual task transfer,” 2024, arXiv:2408.03326.

[5] OpenAI, “GPT-4o system card,” 2024, arXiv:2410.21276.

[6] P. Xu et al., “LVLM-EHub: A comprehensive evaluation benchmark for large vision-language models,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 3, pp. 1877–1893, 2025.

[7] T. Guan et al., “CodePercept: Code-grounded visual STEM perception for MLLMs,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2026, pp. 33 542–33 552.

[8] J. Zhang, J. Huang, S. Jin, and S. Lu, “Vision-language models for vision tasks: A survey,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 8, pp. 5625–5644, 2024.

[9] A. Radford et al., “Learning transferable visual models from natural language supervision,” in Proc. Int. Conf. Mach. Learn., 2021, pp. 8748–8763.

[10] J. Li, D. Li, S. Savarese, and S. Hoi, “BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models,” in Proc. Int. Conf. Mach. Learn., 2023, pp. 19 730–19 742.

[11] S. Bai et al., “Qwen3-VL technical report,” 2025, arXiv:2511.21631.

[12] DeepSeek-AI, “DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning,” 2025, arXiv:2501.12948.

[13] OpenAI, “Learning to reason with LLMs,” OpenAI, 2024.

[14] L. Xing et al., “CapRL: Stimulating dense image caption capabilities via reinforcement learning,” 2025, arXiv:2509.22647.

[15] Z. Zheng et al., “DeepEyes: Incentivizing ‘thinking with images’ via reinforcement learning,” 2025, arXiv:2505.14362.

[16] J. Hong, C. Zhao, C. Zhu, W. Lu, G. Xu, and X. Yu, “DeepEyesV2: Toward agentic multimodal model,” 2025, arXiv:2511.05271.

[17] Z. Zhou et al., “Reinforced visual perception with tools,” 2025, arXiv:2509.01656.

[18] Q. Dong et al., “Visual reasoning through tool-supervised reinforcement learning,” 2026, arXiv:2604.19945.

[19] A. J. Wang, P. Zhou, M. Z. Shou, and S. Yan, “Enhancing visual grounding in vision-language pre-training with position-guided text prompts,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 5, pp. 3406–3421, 2024.

[20] W. Tang, L. Li, X. Liu, L. Jin, J. Tang, and Z. Li, “Context disentangling and prototype inheriting for robust visual grounding,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 5, pp. 3213–3229, 2024.

[21] S. Liu et al., “Grounding DINO: Marrying DINO with grounded pre-training for open-set object detection,” 2023, arXiv:2303.05499.

[22] A. Kirillov et al., “Segment anything,” in Proc. IEEE Int. Conf. Comput. Vis., 2023, pp. 4015–4026.

[23] B. Xiao et al., “Florence-2: Advancing a unified representation for a variety of vision tasks,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2024, pp. 4818–4829.

[24] T. Guan et al., “Industrial scene text detection with refined featureattentive network,” IEEE Trans. Circuits Syst. Video Technol., vol. 32, no. 9, pp. 6073–6085, 2022.

[25] T. Guan, W. Shen, X. Yang, X. Wang, and X. Yang, “Bridging synthetic and real worlds for pre-training scene text detectors,” in Proc. Eur. Conf. Comput. Vis., 2024, pp. 428–446.

[26] T. Guan et al., “Self-supervised implicit glyph attention for text recognition,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2023, pp. 15 285–15 294.

[27] T. Guan, W. Shen, X. Yang, Q. Feng, Z. Jiang, and X. Yang, “Selfsupervised character-to-character distillation for text recognition,” in Proc. IEEE Int. Conf. Comput. Vis., 2023, pp. 19 473–19 484.

[28] T. Guan, W. Shen, and X. Yang, “CCDPlus: Towards accurate character to character distillation for text recognition,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 5, pp. 3546–3562, 2025.

[29] T. Guan, C. Lin, W. Shen, and X. Yang, “PosFormer: Recognizing complex handwritten mathematical expression with position forest transformer,” in Proc. Eur. Conf. Comput. Vis., 2024, pp. 130–147.

[30] T. Guan et al., “A token-level text image foundation model for document understanding,” in Proc. IEEE Int. Conf. Comput. Vis., 2025, pp. 23 210–23 220.

[31] Z. Wang et al., “Marten: Visual question answering with mask generation for multi-modal document understanding,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2025, pp. 14 460–14 471.

[32] C. Wen, Z. Peng, Y. Huang, X. Yang, and W. Shen, “Exploring efficient reasoning segmentation with small language models,” in Proc. Eur. Conf. Comput. Vis., 2026, pp. 577–595.

[33] C. Wen, Z. Peng, Y. Huang, and W. Shen, “Efficient segmentation with multimodal large language model via token routing,” in Proc. AAAI Conf. Artif. Intell., vol. 40, no. 13, 2026, pp. 10 593–10 602.

[34] S. Jiang et al., “Hulu-Med: A transparent generalist model towards holistic medical vision-language understanding,” 2025, arXiv:2510.08668.

[35] B.-W. Yin, Q. Yang, B. Sun, X. Wei, and Q. Hou, “Discriminative visual process rewards for scaling thinking at test-time with images,” in Proc. Int. Conf. Mach. Learn., 2026.

[36] B. Sun, B.-W. Yin, Y.-M. Li, X. Wei, and Q. Hou, “See what i mean: Aligning vision and language representations for video fine-grained object understanding,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2026, pp. 36 814–36 827.

[37] R. Chen et al., “Video\* Bench: Benchmarking fine-grained video perception with “Think with Videos”,” in Proc. ACM Multimedia, 2026.

[38] R. Chen et al., “Datasets and recipes for video temporal grounding via reinforcement learning,” in Proc. Conf. Empirical Methods in Natural Language Process. (Industry Track), 2025, pp. 983–992.

[39] W. Huang et al., “Vision-R1: Incentivizing reasoning capability in multimodal large language models,” 2025, arXiv:2503.06749.

[40] Z. Shao et al., “DeepSeekMath: Pushing the limits of mathematical reasoning in open language models,” 2024, arXiv:2402.03300.

[41] H. Shen et al., “VLM-R1: A stable and generalizable R1-style large vision-language model,” 2025, arXiv:2504.07615.

[42] J. Zhang et al., “R1-VL: Learning to reason with multimodal large language models via step-wise group relative policy optimization,” 2025, arXiv:2503.12937.

[43] L. Chen et al., “G1: Bootstrapping perception and reasoning abilities of vision-language model via reinforcement learning,” 2025, arXiv:2505.13426.

[44] R. Luo et al., “From narrow to panoramic vision: Attentionguided cold-start reshapes multimodal reasoning,” 2026, arXiv:2603.03825.

[45] T. Xiao et al., “Perception-R1: Advancing multimodal reasoning capabilities of MLLMs via visual perception reward,” 2025, arXiv:2506.07218.

[46] Z. Wang et al., “Perception-aware policy optimization for multimodal reasoning,” 2025, arXiv:2507.06448.

[47] Y. Ma et al., “One RL to see them all: Visual triple unified reinforcement learning,” 2025, arXiv:2505.18129.

[48] T. Gupta and A. Kembhavi, “Visual programming: Compositional visual reasoning without training,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2023, pp. 14 953–14 962.

[49] D. Sur´ıs, S. Menon, and C. Vondrick, “ViperGPT: Visual inference via Python execution for reasoning,” in Proc. IEEE Int. Conf. Comput. Vis., 2023, pp. 11 888–11 898.

[50] Y. Chen et al., “R1-Code-Interpreter: LLMs reason with code via supervised and multi-stage reinforcement learning,” 2025, arXiv:2505.21668.

[51] Q. Yu et al., “DAPO: An open-source LLM reinforcement learning system at scale,” 2025, arXiv:2503.14476.

[52] S. Wang et al., “LocateAnything: Fast and high-quality vision-language grounding with parallel box decoding,” 2026, arXiv:2605.27365.

[53] Alibaba Cloud, “Visual Understanding: Alibaba Cloud Model Studio,” 2026, Accessed: Oct. 11, 2026.

[54] Google, “Models: Gemini API,” 2026, Accessed: Oct. 11, 2026.

[55] ByteDance Seed, “Seed2.0,” 2026, Accessed: Oct. 11, 2026.

[56] Moonshot AI, “Kimi K2.6: Advancing Open-Source Coding,” 2026, Accessed: Oct. 11, 2026.

[57] Kimi Team, “Kimi K2.5: Visual agentic intelligence,” 2026, arXiv:2602.02276.

[58] Anthropic, “Claude Opus 4.8 System Card,” 2026, Accessed: Oct. 11, 2026.

[59] xAI, “Introducing Grok 4.5,” 2026, Accessed: Oct. 11, 2026.

[60] Anthropic, “Claude Opus 4.7 System Card,” 2026, Accessed: Oct. 11, 2026.

[61] DeepSeek-AI, “DeepSeek-V4-Flash-Vision-Exp Release: Multimodal API Now Live,” 2026, Accessed: Oct. 11, 2026.

[62] xAI, “Grok on Amazon Bedrock,” 2026, Accessed: Oct. 11, 2026.

[63] Anthropic, “Claude Opus 4.6 System Card,” 2026, Accessed: Oct. 11, 2026.

[64] Qwen Team, “Qwen3.6-35B-A3B: Agentic Coding Power, Now Open to All,” 2026, Accessed: Oct. 11, 2026.

[65] xAI, “RealWorldQA,” Dataset, 2024.

[66] L. Chen et al., “Are we on the right way for evaluating large visionlanguage models?” 2024, arXiv:2403.20330.

[67] X. Fu et al., “BLINK: Multimodal large language models can see but not perceive,” 2024, arXiv:2404.12390.

[68] H. Wang et al., “Traceable evidence enhanced visual grounded reasoning: Evaluation and methodology,” 2025, arXiv:2507.07999.

[69] Gemini Robotics Team, “Gemini Robotics: Bringing AI into the physical world,” 2025, arXiv:2503.20020.

[70] E. Zhou et al., “RoboRefer: Towards spatial referring with reasoning in vision-language models for robotics,” 2025, arXiv:2506.04308.

[71] J. S. Tamarapalli, R. Grover, N. Pande, and S. Yerramilli, “CountQA: How well do MLLMs count in the wild?” 2025, arXiv:2508.06585.

[72] P. Wu and S. Xie, “V\*: Guided visual search as a core mechanism in multimodal LLMs,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2024, pp. 13 084–13 094.

[73] W. Wang et al., “Divide, conquer and combine: A training-free framework for high-resolution image perception in multimodal large language models,” 2024, arXiv:2408.15556.

[74] Y.-F. Zhang et al., “Thyme: Think beyond images,” 2025, arXiv:2508.11630.

[75] H. Wang, A. Su, W. Ren, F. Lin, and W. Chen, “Pixel Reasoner: Incentivizing pixel-space reasoning with curiosity-driven reinforcement learning,” 2025, arXiv:2505.15966.