# SurgPace: Complexity-Aware Progressive Distillation for Surgical Multimodal Large Language Models

Zhicheng He<sup>a</sup>, Xiaohan Xing<sup>b</sup>, Hongqiu Wang<sup>c</sup>, Kun Yuan<sup>d</sup>, Guankun Wang<sup>a</sup>, Yanheng Li<sup>e</sup>, Haoxuan Che<sup>e</sup>, Nassir Navab<sup>d</sup>, Hongliang Ren<sup>a</sup>, Long Bai<sup>a,d,∗</sup>

<sup>a</sup>Department ofElectronic Engineering, The Chinese University ofHong Kong, Hong Kong SAR, China <sup>b</sup>Department of Radiation Oncology, Stanford University, Palo Alto, USA

<sup>c</sup>Systems Hub, The Hong Kong University ofScience and Technology (Guangzhou), Guangzhou, China <sup>d</sup>Chair of Computer Aided Medical Procedures, Technical University of Munich, Munich, Germany <sup>e</sup>XGEN Labs, Hong Kong SAR, China

## Abstract

Multimodal large language models (MLLMs) have shown strong potential in robotic surgery for question answering, scene un- <sup>2</sup> derstanding, and grounded assistance, but their high computational cost limits real clinical deployment. Knowledge distillation provides a practical way to build compact surgical MLLMs, yet direct distillation is not well suited to surgical data because sur gical samples vary greatly in dificulty, visual ambiguity, and required clinical context, making uniform teacher-student transfer unstable. In addition, the role of supervision changes during training: early stages require stronger teacher guidance for multimodal2 alignment, while later stages rely more on task supervision for efective optimization. To address these challenges, we propose SurgPace, a complexity-aware continuous curriculum framework for progressive distillation of surgical vision-language models. SurgPace estimates sample complexity using predictive confidence and Lorentz geometric consistency, and uses the fused score to guide a continuous easy-to-hard training process. We further introduce hierarchical representation alignment, which aligns the student with the teacher at the feature, logit, and probability levels, while gradually shifting the training focus from teacher imis tation to task supervision. This design separates sample-level curriculum from objective-level scheduling, leading to more stable<sup>c</sup> and efective distillation. We evaluate SurgPace on Surg-396K, a benchmark built from four public surgical datasets and covering five dialogue paradigms. SurgPace consistently outperforms strong MLLM distillation baselines, demonstrating its efectiveness for building compact, memory-eficient surgical MLLMs with faster decoding under controlled benchmarks. Code is available at https://github.com/hezhicheng2002/SurgPace.

Keywords:

Vision-Language models, Knowledge Distillation, Curriculum Learning, Surgical Data Science

## 1. Introduction

Modern surgical care increasingly relies on digital imaging and video streams to support clinical decision-making [36, 29].<sup>v</sup> During minimally invasive procedures, large volumes of visual data are generated, including endoscopic videos, surgical tool trajectories, and intraoperative imaging. These data encode rich<sup>a</sup> information about anatomical structures, surgical actions, and instrument-tissue interactions. Surgeons must continuously interpret these visual signals to infer the current surgical state and make timely decisions [53, 2]. As surgical procedures become more complex, the volume and heterogeneity of visual data increase rapidly, which further elevates the cognitive burden on clinicians. In recent years, multimodal large language models (MLLMs) have emerged as a promising paradigm for interpreting such complex medical data [22, 19, 57]. By integrating visual perception with language understanding, these models can analyze surgical scenes while interacting with clinicians through natural language. This capability supports applications such as surgical question answering, workflow understanding, procedure summarization, and clinical dialogue assistance. As a result, MLLMs are increasingly regarded as a potential foundation for intelligent surgical assistants that can observe procedures, interpret context, and communicate clinically relevant information during operations [39, 45, 14, 13].

Despite this promise, the practical deployment of MLLMs in surgical environments remains challenging [32, 15, 9]. Modern MLLMs typically rely on large vision encoders and deep language backbones, which incur substantial computational overhead. These architectures require considerable memory, highend GPUs, and long inference time [44]. Such demands conflict with the constraints of real clinical systems, where latency, hardware resources, and system reliability are strictly limited. In operating rooms and robotic surgery platforms, intelligent systems must respond rapidly and reliably, as surgeons often require immediate feedback during critical procedural steps [56, 28]. However, MLLMs are usually developed in research settings with abundant computational resources. Their translation into real clinical workflows therefore remains nontrivial [21]. This gap between model capability and deployment feasibility has become a major barrier to practical surgical multimodal intelligence in real clinical settings.

![](images/5b137090f43afeaab635f381dd4a3842b3b7778aff63609ea095b12aaba40d5f.jpg)  
Figure 1: Overview of the proposed SurgPace framework. Given surgical multimodal data, a teacher model first estimates sample complexity ofline using complementary confidence and geometric consistency signals, which are permissively fused so that lower scores indicate easier samples. The resulting complexity estimates are then used to guide online curriculum training, where easier samples are emphasized earlier and harder samples are gradually incorporated as training progresses. During this process, the student model learns from the teacher through hierarchical representation alignment at the probability, logit, and feature levels, while a progressive objective schedule adjusts the balance of supervision over time. Together, these components enable stable and eficient compression of surgical MLLMs into a lightweight student model with reduced computational demands.

Transferring multimodal capability from a high-capacity teacher model to a resource-eficient student model remains a fundamental challenge for surgical MLLM systems. Knowledge distillation provides a practical route for compressing MLLMs into compact models with lower computational demands [10, 11, 59]. However, directly applying standard distillation strategies to surgical multimodal tasks often leads to unstable optimization and suboptimal transfer. Several aspects of surgical multimodal learning make this transfer dificult. First, surgical samples difer widely in visual clarity, question complexity, and required clinical context [31, 33]. Treating all samples in the same way can cause hard cases to dominate early training and reduce training stability. Second, surgical tasks require joint reasoning over visual appearance, instruments, anatomy, spatial relations, actions, and procedural context [54]. Distilling only final responses can overlook intermediate cues that support reliable surgical understanding. Third, the useful supervision signal changes during training. Early stages benefit more from teacher-guided supervision for visual and multimodal alignment, while later stages need stronger task supervision to refine final answers. A fixed objective cannot model this change well. These issues limit reliable knowledge transfer in resource-constrained surgical systems, such as operating rooms and robotic platforms.

To address these challenges, we propose SurgPace, a complexity-aware continuous curriculum distillation framework for building compact surgical MLLMs. SurgPace trains the student with dificulty-aware sample pacing and progressive multi-level alignment. At the sample level, SurgPace estimates the complexity of each multimodal instance using predictive confidence and Lorentz geometric consistency from the teacher.

The two measures are fused into one complexity score, so that lower-complexity samples receive larger weights in early training, and this bias is gradually reduced as training proceeds. At the objective level, SurgPace uses hierarchical representation alignment to operationalize structured knowledge transfer. The student is aligned with the teacher at probability, logit, and hidden-representation levels, while the balance between distillation loss and task loss changes over time. Early training focuses more on teacher-guided transfer, while later training gives more weight to ground-truth supervision. This design separates sample-level curriculum from objective-level scheduling and gives the student a stable learning path. Figure 1 provides an overview of SurgPace and its training workflow. Specifically, the contributions of this work are summarized as follows:

• We formulate SurgPace around two coordinated axes: sample-level pacing controls the emphasis on training examples, and objective-level scheduling controls the balance between teacher and task supervision.

• We combine teacher predictive confidence and Lorentz trajectory dispersion into a complexity score that guides continuous easy-to-hard weighting within hierarchical feature-, logit-, and probability-level alignment.

• We validate the SurgPace framework on the Surg-396K benchmark, which is built from four public surgical datasets and covers five dialogue paradigms. Extensive experiments and ablations demonstrate strong performance compared to state-of-the-art methods.

## 2. Related Work

Multimodal large language models (MLLMs) are shifting surgical AI from single-task, procedure-specific predictors toward interactive assistants that jointly model visual perception, domain knowledge, and natural-language reasoning [35].

Early surgical MLLM systems primarily focused on tool-use orchestration: VS-Assistant [7] translates surgeon intent into structured sequences of algorithmic calls via function-calling tuning, and SurgFC [6] further unifies intention understanding with downstream vision modules into a single surgical interface. However, such orchestration-centric designs remain constrained by limited visual representation and domain grounding. To address this, GP-VLS [38] injects surgical textbooks and medical knowledge into a general-purpose surgical vision language model (VLM), enabling broader procedural competence, yet it remains largely image-centric and insuficient for fine-grained temporal reasoning in long operations.

To overcome these limitations, recent work has pursued richer visual grounding and broader multimodal dialogue capabilities. LLaVA-Surg [23] introduces large-scale structured surgical video instruction tuning to support multi-turn dialogue across diverse procedures, while EndoChat [46] advances grounded surgical dialogue through explicit multiscale visual tokenization for endoscopic scenes. In parallel, SurgVLP [55] improves surgical language-vision alignment via contrastive pretraining on lecture video-text pairs but still requires task-specific adaptation for interactive intraoperative question-answering (QA). Building on these foundations, SurgVidLM [47] and SurgLLM [8] further strengthen surgical multimodal understanding through enhanced visual modeling and context-aware pretraining strategies. Despite these advances, intraoperative deployment remains challenging due to strict latency and hardware constraints, as robust temporal reasoning often relies on large backbones and computationally intensive video encoders [30]. In this context, knowledge distillation ofers a practical way to retain core multimodal capabilities while transferring them into more eficient models for surgical assistance.

## 2.2. Distilling Multimodal Large Language Models

The rapid scaling of MLLMs has made knowledge distillation a primary route toward deployable models without retraining from scratch. Early work on eficient VLMs combines teacher-student distillation with structural sparsification, exemplified by EficientVLM [49], which distills a large VLM during pretraining and applies modal-adaptive pruning to remove redundant components. As MLLMs evolve toward instructionfollowing and open-ended multimodal dialogue, subsequent studies focus on what to distill and where supervision should be applied. LLAVADI [52] systematically analyzes MLLM distillation design choices and emphasizes jointly aligning tokenlevel behavior and output distributions, while CoMD [24] moves beyond one-way imitation by introducing competitive distillation to improve student multimodal competence. To better preserve multimodal reasoning, LLaVA-KD [4] and MoVE-KD [5] distill cross-modal relations and complementary visual knowledge, respectively, and LLaVA-MoD [41] further reduces inference cost through sparse MoE design with progressive and preference-aware distillation. Orthogonally, architectural sub stitutions paired with distillation, such as MaTVLM [26], replace parts of the Transformer with Mamba-style layers to recover performance under lower latency.

Beyond objectives and architectures, distillation performance also depends on supervision quality and training dynamics. Self-Improving Teacher improves student transfer by calibrating distillation through challenging samples that refine teacher supervision [25]. Importantly, recent work increasingly treats distillation as a curriculum rather than a one-shot compression step. Curriculum extraction [12] and progressive consistency distillation [51] demonstrate that staged learning trajectories improve distillation stability and eficiency. Explicit curriculum scheduling has also been explored for knowledge distillation in more generic settings, like CKD-MDSR [43], but remains comparatively underexplored for MLLMs and eficiency-critical domains such as surgical assistance.

## 2.3. Advances in Curriculum learning

Curriculum learning (CL) is a machine learning strategy that arranges training data, tasks, or objectives from simpler to more complex instances to improve convergence and generalization, inspired by human pedagogical principles [42]. Early work formalized CL as a training paradigm where model performance benefits from gradually increasing complexity of examples, and established general frameworks involving dificulty measurers and training schedulers [50]. Beyond traditional data-level curricula, recent work has extended CL principles to eficient training of large neural architectures. For large language models (LLMs), curriculum distillation frameworks systematically structure training sequences to first solidify basic competencies before introducing harder reasoning tasks, yielding improved performance over single-stage distillation [20]. In vision-language settings, curriculum strategies have been shown to benefit multimodal evaluation by pacing training samples according to complexity, especially under limited data regimes [37]. Moreover, curriculum learning has been applied in large-scale language model pretraining, where ordering examples by dificulty metrics consistently improves convergence and early training eficiency compared to random sampling [60]. In application-specific domains, specialized curricula designed by domain experts substantially improve clinical disease staging and referral outcomes for medical visionlanguage models compared to baseline training strategies [16]. These advances collectively indicate that curriculum learning constitutes a flexible training paradigm that can be adapted for both general eficiency gains and domain-specific performance, motivating its integration with knowledge distillation and adaptive scheduling strategies in surgical MLLMs.

## 3. Methods

In this section, we present SurgPace, a complexity-aware continuous curriculum framework for progressive distillation of surgical vision-language models. The key idea is to estimate the teacher-derived complexity of each training sample, convert this estimate into a continuous easy-to-hard sample pacing mechanism, and couple this sample-level curriculum with hierarchical representation alignment. This formulation explicitly separates two roles that are often entangled in curriculumbased distillation: curriculum determines which samples should receive greater emphasis during training, whereas progressive objective scheduling determines how diferent supervision signals should be balanced over time. We instantiate SurgPace with EndoChat [46] as the teacher (13B) and a lightweight multimodal model as the student (3B). The student follows a TinyLLaVA-style architecture built on Phi-2 [17], with a SigLIP [58] vision encoder and a two-layer GELU-based MLP connector, and is initialized from a pretrained Phi-2-SigLIP checkpoint [18]. Motivation: Teacher-Informed Complexity Estimation. Curriculum learning improves optimization by modulating how training samples are emphasized according to their complexity. In vision-language models, however, complexity is rarely captured by superficial cues such as image appearance or text length alone. Instead, it often arises from semantic ambiguity, cross-modal alignment dificulty, and uncertainty in sequence generation. Existing curriculum strategies therefore commonly rely on heuristic proxies, such as current loss values or metadata-derived priors, which can be noisy, unstable, and highly sensitive to the training dynamics. We adopt a teacher-informed perspective: a sample is considered dificult when a strong teacher model exhibits uncertainty or internal inconsistency while generating the target answer associated with that sample. This view anchors curriculum construction in teacher response behavior for progressive distillation.

![](images/850e111cd857ad7a0bd1a1d734577ad02c78cfce1825b64d60a1e13dce7b1e2e.jpg)  
Figure 2: Overview of the proposed complexity-aware curriculum framework for progressive distillation. Teacher confidence and Lorentz geometric con sistency are fused into a permissive complexity score, which guides continuous curriculum pacing from easy-first sampling to balanced training. This pacing is coupled with progressive objective interpolation and hierarchical representation alignment, jointly establishing a more structured optimization process that enables stable and eficient compression of surgical vision-language models.

## 3.1. Overview of SurgPace

SurgPace integrates two coordinated training mechanisms into a hierarchical distillation base. The distillation base specifies what knowledge is transferred from the teacher to the student, the curriculum mechanism determines which samples should be emphasized at diferent stages of training, and the progressive objective schedule governs how distillation and supervised objectives are balanced over time. As illustrated in Figure 2, each training sample is first assigned a scalar complexity score derived from teacher confidence and Lorentz geometric consistency with respect to the target answer. These precomputed scores are then transformed into progress-dependent sample weights, giving rise to a continuous easy-to-hard pacing strategy whose efect gradually decays as training advances. In parallel, the student is optimized with supervised cross-entropy and hierarchical representation alignment, which jointly enforces distribution alignment, logit matching, representation alignment, and entropy regularization. Their relative contributions are further adjusted through the progressive objective schedule along the training trajectory. In this design, the curriculum acts at the sample level, whereas the objective schedule acts at the loss level. Together, these components make Surg-Pace a unified and efective framework for stable and eficient progressive distillation of surgical vision-language models.

## 3.2. Complexity-Aware Continuous Curriculum

Our curriculum is built on two tightly coupled components: a teacher-derived complexity estimator and a continuous pacing function that converts this estimator into optimization weights. A key design choice is to explicitly separate ofline scoring from online weighting. In the surgical MLLM setting, online complexity estimation is highly unstable because batch sizes are small and the student distribution evolves rapidly during training, whereas teacher-derived ofline scores remain globally comparable throughout optimization. We therefore compute a scalar score for each training instance once, store it with the dataset, and reuse this fixed signal during training. Under this design, curriculum is not implemented as a heuristic reshufling of mini-batches, but as a dataset-level prior that induces smooth and progress-dependent sample emphasis.

Complexity Estimator. We compute the complexity of each sample ofline using a fixed teacher model. The goal is to measure dificulty from the teacher’s actual response behavior on the target answer y , instead of from shallow metadata. For a sample x with target sequence length $L ,$ the teacher produces token probabilities $\mathbf { \Delta } _ { p _ { t } ^ { ( T ) } } ^ { - }$ and logits $\overline { { l _ { t } ^ { ( T ) } } }$ at each decoding step. We then derive two complementary signals. The first asks whether the teacher is locally certain about the next token. The second asks whether the full decoding trajectory remains globally coherent across the answer sequence. These signals capture diferent failure modes: a teacher may exhibit high confidence at individual steps while still producing an unstable trajectory across the full sequence.

Confidence Score $( s _ { c o n f } ) { : }$ This score is defined as the complement of the teacher’s mean top-1 probability, and thus serves as a token-level uncertainty estimate in the standard Euclidean probability space:

$$
s _ { c o n f } ( x _ { i } ) = 1 - \frac { 1 } { L } \sum _ { t = 1 } ^ { L } \operatorname* { m a x } _ { \nu } p _ { t } ^ { ( T ) } ( \nu ) .\tag{1}
$$

Because each term inside the summation lies in [0 1], the resulting score also satisfies $s _ { c o n f } ( x _ { i } ) \in [ 0 , 1 ]$ . A smaller value indicates that the teacher consistently assigns high mass to one dominant token, suggesting that the sample is easy to decode. Conversely, a larger value indicates persistent uncertainty and therefore higher sample complexity. This signal is purely local: it summarizes token-wise certainty in the teacher’s standard output space, but does not account for whether the full decoding trajectory remains geometrically coherent across the answer sequence.

Lorentz Geometry Score $\left( s _ { \mathrm { l o r e n t z } } \right) :$ Confidence alone does not describe the stability of the teacher’s decoding process over the whole answer. We therefore map teacher logits to the unit Lorentz hyperboloid and measure their dispersion over the answer sequence. For numerical stability, each teacher logit vector is first compressed into a bounded spatial coordinate:

$$
u _ { t } = \frac { l _ { t } ^ { ( T ) } } { \sqrt { 1 + \Vert l _ { t } ^ { ( T ) } \Vert ^ { 2 } + \epsilon } } .\tag{2}
$$

We then add a time coordinate and lift the vector to the unit Lorentz hyperboloid:

$$
x _ { t } = \Big ( \sqrt { 1 + \| u _ { t } \| ^ { 2 } } , u _ { t } \Big ) .\tag{3}
$$

With the Lorentz inner product defined as

$$
\langle x , y \rangle _ { L } = - x _ { 0 } y _ { 0 } + x _ { 1 : } ^ { \top } y _ { 1 : } ,\tag{4}
$$

each lifted point is valid because

$$
\langle x _ { t } , x _ { t } \rangle _ { L } = - ( 1 + \| u _ { t } \| ^ { 2 } ) + \| u _ { t } \| ^ { 2 } = - 1 .\tag{5}
$$

The Lorentzian geodesic distance is

$$
d _ { L } ( x , y ) = \operatorname { a r c o s h } ( \operatorname* { m a x } ( 1 + \epsilon , - \langle x , y \rangle _ { L } ) ) .\tag{6}
$$

Because a Euclidean average of hyperboloid points need not remain on the manifold, we average only the spatial coordinates and lift the result again:

$$
\bar { u } = \frac 1 L \sum _ { t = 1 } ^ { L } u _ { t } , \qquad \bar { x } = \Big ( \sqrt { 1 + \| \bar { u } \| ^ { 2 } } , \bar { u } \Big ) .\tag{7}
$$

x¯ is obtained by lifting the averaged Euclidean coordinates and is a valid hyperboloid point. The raw geometric dispersion is

$$
s _ { \mathrm { l o r e n t z } } ^ { \mathrm { r a w } } ( x _ { i } ) = \frac { 1 } { L } \sum _ { t = 1 } ^ { L } d _ { L } ( x _ { t } , \bar { x } ) .\tag{8}
$$

The score is small when the teacher follows a compact and stable trajectory, and large when the token trajectory is dispersed or internally inconsistent.

To make the geometric score directly comparable with the confidence score, we normalize it globally over the training set D:

$$
s _ { \mathrm { l o r e n t z } } ( x _ { i } ) = \frac { s _ { \mathrm { l o r e n t z } } ^ { \mathrm { r a w } } ( x _ { i } ) - \operatorname* { m i n } _ { x \in \mathcal { D } } s _ { \mathrm { l o r e n t z } } ^ { \mathrm { r a w } } ( x ) } { \operatorname* { m a x } _ { x \in \mathcal { D } } s _ { \mathrm { l o r e n t z } } ^ { \mathrm { r a w } } ( x ) - \operatorname* { m i n } _ { x \in \mathcal { D } } s _ { \mathrm { l o r e n t z } } ^ { \mathrm { r a w } } ( x ) + \epsilon } .\tag{9}
$$

This dataset-level normalization is important in practice. It ensures that the score remains stable and comparable across optimization steps, while efectively preventing the degenerate behavior that could otherwise arise from independently renormalizing scores within each mini-batch.

In the main formulation, we adopt a permissive fusion strategy that combines the two normalized criteria as:

$$
s _ { i } = \mathrm { m i n } ( s _ { c o n f } ( x _ { i } ) , s _ { \mathrm { l o r e n t z } } ( x _ { i } ) ) .\tag{10}
$$

This fusion adopts a less restrictive definition of easiness: a sample is assigned a lower complexity score when it appears easy under at least one of the two criteria. In other words, a favorable signal from either confidence or geometric consistency is suficient to reduce the estimated dificulty.

Continuous Weighting Function. After ofline scoring, the selected score field is injected into the dataloader and passed to the trainer as sample-level metadata. During optimization, we perform a conversion of the complexity score into an easiness bias term:

$$
b _ { i } = 1 - s _ { i } ,\tag{11}
$$

so that larger values correspond to easier samples. We then linearly map this bias into a bounded interval $[ w _ { \mathrm { m i n } } , w _ { \mathrm { m a x } } ] ;$

$$
w _ { i } = w _ { \operatorname* { m i n } } + ( w _ { \operatorname* { m a x } } - w _ { \operatorname* { m i n } } ) b _ { i } .\tag{12}
$$

Under this formulation, easier samples receive larger initial weights, whereas harder samples are relatively down-weighted at the beginning of training. The weights are computed from globally normalized ofline scores.

Curriculum Annealing. To avoid permanently biasing optimization toward low-complexity samples, we gradually anneal curriculum weights back to unity as training progresses. Let $p \in \ [ 0 , 1 ]$ denote the normalized training progress. We define an annealing factor:

$$
a ( p ) = \operatorname* { m a x } \left( 0 , 1 - \frac { p } { r _ { \mathrm { e n d } } } \right) ,\tag{13}
$$

where $r _ { \mathrm { e n d } }$ denotes the point at which explicit curriculum emphasis vanishes. The efective sample weight at progress $p$ can then be formulated as:

$$
w _ { i } ^ { \prime } ( p ) = 1 + a ( p ) ( w _ { i } - 1 ) .\tag{14}
$$

Thus, $w _ { i } ^ { \prime } ( p )$ continuously interpolates between the curriculum weight $w _ { i }$ and the unbiased weight 1 0. When $p$ is small, optimization follows an explicit easy-first regime; when $p \geq r _ { \mathrm { e n d } }$ all samples are treated uniformly.

## 3.3. Hierarchical Representation Alignment

This hierarchical representation alignment objective defines the objective-side core of SurgPace. We formulate teacherstudent transfer as a unified multi-level alignment objective under the curriculum-defined sample pacing and the progressive schedule introduced later in Sec. 3.4. Concretely, teacher and student are aligned at three complementary levels: output distributions, pre-softmax response patterns, and latent representations, while all terms share the same token mask, persample reduction, and sample-aware weighting scheme. This design is important because multimodal compression errors rarely emerge at only one level. A student may appear correct at the surface output level while already drifting in logit geometry or hidden-state organization, and such hidden inconsistencies are particularly harmful for surgical dialogue and grounding tasks that rely on stable cross-modal reasoning.

Trajectory alignment across tokenizers. The teacher and student use diferent subword tokenizers, so neither their token indices nor their token counts provide a semantic correspondence. Let $a _ { t } ^ { ( T ) }$ and $a _ { i } ^ { ( S ) }$ be the positive UTF-8 surface widths of the decoded teacher and student answer tokens, respectively. We locate every token at the center of its normalized span in the decoded text:

$$
r _ { t } ^ { ( T ) } = \frac { \sum _ { q = 1 } ^ { t } a _ { q } ^ { ( T ) } - \frac { 1 } { 2 } a _ { t } ^ { ( T ) } } { \sum _ { q = 1 } ^ { L _ { T } } a _ { q } ^ { ( T ) } } , \qquad r _ { j } ^ { ( S ) } = \frac { \sum _ { q = 1 } ^ { j } a _ { q } ^ { ( S ) } - \frac { 1 } { 2 } a _ { j } ^ { ( S ) } } { \sum _ { q = 1 } ^ { L _ { S } } a _ { q } ^ { ( S ) } } .\tag{15}
$$

We interpolate the teacher trajectory at the student positions,

$$
\begin{array} { r } { \hat { l } _ { j } ^ { ( T ) } = \mathrm { I n t e r p } \big ( \{ ( r _ { t } ^ { ( T ) } , l _ { t } ^ { ( T ) } ) \} _ { t = 1 } ^ { L _ { T } } ; r _ { j } ^ { ( S ) } \big ) . } \end{array}\tag{16}
$$

Interpolation is linear between neighboring teacher token centers and uses the nearest endpoint value outside their range. For a single teacher token, its logits are broadcast to all student positions. For vocabulary alignment, we form a fixed semantic support $\mathcal { A } = \{ ( \nu _ { m } ^ { ( T ) } , \nu _ { m } ^ { ( S ) } ) \} _ { m = 1 } ^ { M }$ from exact matches between decoded tokens in the two vocabularies. Let $P _ { T }$ and $P _ { S }$ select the corresponding teacher and student coordinates. The aligned logits used by all losses defined on model outputs are

$$
\tilde { l } _ { j } ^ { ( T ) } = e ^ { \gamma } P _ { T } \hat { l } _ { j } ^ { ( T ) } , \qquad \hat { l } _ { j } ^ { ( S ) } = P _ { S } l _ { j } ^ { ( S ) } ,\tag{17}
$$

where $\gamma$ is a learned scalar log-scale parameter. This construction restricts comparison to token identities explicitly shared in decoded text.

Let $M _ { i }$ denote the set of valid next-token positions for sample i after standard causal shifting, and define the corresponding masked averaging operator over token-level quantities:

$$
\rho _ { i } ( \delta ) = \frac { 1 } { | M _ { i } | } \sum _ { t \in M _ { i } } \delta _ { i , t } .\tag{18}
$$

All distillation losses below are first computed at the token level and then reduced to per-sample losses through $\rho _ { i } ( \cdot ) .$ This reduction matches the actual training behavior and, more importantly, prevents long responses from dominating short but dificult surgical queries, which makes the alignment objective more balanced across samples.

Distribution Alignment: On the shared vocabulary support, we align the student distribution with the projected teacher distribution through temperature-scaled Kullback-Leibler divergence:

$$
\ell _ { i } ^ { K L } = \rho _ { i } \bigg ( T ^ { 2 } \mathrm { K L } \bigg ( \mathrm { s o f t m a x } \bigg ( \frac { \tilde { l } _ { t } ^ { ( T ) } } { T } \bigg ) \bigg \| \mathrm { s o f t m a x } \bigg ( \frac { \tilde { l } _ { t } ^ { ( S ) } } { T } \bigg ) \bigg ) \bigg ) ,\tag{19}
$$

where T is the distillation temperature. This term transfers the teacher’s relative token preferences on the shared vocabulary support and provides the student with a fine-grained preference structure, which is especially important for achieving precise, contextually consistent wording in surgical descriptions and question answering.

Logit Alignment: A softmax distribution is invariant to an additive constant at each position. To make the logit term respect the same identifiable response geometry, we define the centering operator $C ( z ) \ = \ z - \ M ^ { - 1 } { \bf 1 } _ { M } { \bf 1 } _ { M } ^ { \top } z$ and minimize the squared distance between centered logits:

$$
\ell _ { i } ^ { L 2 } = \rho _ { i } \Big ( \Big \| C \big ( \hat { l } _ { t } ^ { ( S ) } \big ) - C \big ( \tilde { l } _ { t } ^ { ( T ) } \big ) \Big \| _ { 2 } ^ { 2 } \Big ) .\tag{20}
$$

This removes only the unidentifiable ofset while retaining the relative logit pattern and its confidence scale. Whereas $\ell _ { i } ^ { K L }$ constrains the probability simplex, $\ell _ { i } ^ { L 2 }$ matches the centered response geometry before softmax.

Latent Alignment: To complement transfer at the model output with supervision of internal structure, we standardize the centered projected teacher logits, $S ( z ) = C ( z ) / \sqrt { M ^ { - 1 } \| C ( z ) \| _ { 2 } ^ { 2 } + \epsilon }$ before mapping them into the student’s hidden dimension through a learnable head $\psi :$

$$
\begin{array} { r } { \ell _ { i } ^ { f e a t } = \rho _ { i } \left( \left| \left| h _ { i , t } ^ { ( S ) } - \psi \left( S \left( \tilde { l } _ { i , t } ^ { ( T ) } \right) \right) \right| \right| _ { 2 } ^ { 2 } \right) , } \end{array}\tag{21}
$$

Algorithm 1: Complexity-Aware Continuous Curricu  
lum for Progressive Distillation   
Input: Dataset D, Teacher T, Student S , Max steps M   
Params: Weights $\lambda = \{ w _ { c e } , w _ { k l } , w _ { l 2 } , w _ { f e a t } , w _ { e n t } \} ;$   
Hyperparameters Θ   
for each sample $( x _ { i } , y _ { i } ) \in \mathcal { D }$ do   
Run teacher T on $( x _ { i } , y _ { i } )$ to obtain logits $l ^ { ( T ) }$ and   
probabilities $p ^ { ( T ) }$   
$s _ { c o n f } \gets S _ { c o n f } ( p ^ { ( T ) } )$   
$s _ { \mathrm { l o r e n t z } } ^ { \mathrm { r a w } }  s _ { g e o } ( l ^ { ( T ) } )$   
end   
$s _ { \mathrm { l o r e n t z } }  N ( s _ { \mathrm { l o r e n t z } } ^ { \mathrm { r a w } } )$   
s<sub>i</sub> ← min(s<sub>conf,</sub> s<sub>lorentz</sub>); store $s _ { i }$ in the dataset   
$w _ { i }  \mathcal { W } ( s _ { i } )$   
for step = 1 to M do   
Load batch B with task labels and stored sample   
scores; set p ← step/M   
$w _ { i } ^ { \prime } ( p ) \gets C ( w _ { i } , p )$   
$\beta _ { c e } ( p ) , \beta _ { d i s t } ( p ) \gets \mathcal { A } ( p )$   
$\omega _ { i } ( p ) \gets \tau _ { i } w _ { i } ^ { \prime } ( p )$   
Student $l ^ { ( S ) } , \dot { h } ^ { ( S ) } \gets S ( x _ { i } , y _ { i } ) ;$ Teacher   
$l ^ { ( T ) } \gets T ( x _ { i } , y _ { i } )$   
$\tilde { l } ^ { ( T ) } \gets \Pi ( l ^ { ( T ) } )$   
$z ^ { ( T ) } \gets \psi ( \tilde { l } ^ { ( T ) } )$   
$\ell _ { i } ^ { C E } , \ell _ { i } ^ { K L } , \ell _ { i } ^ { L 2 } , \ell _ { i } ^ { f e a t } , \ell _ { i } ^ { e n t } \gets M ( \mathcal { B } )$   
$\mathcal { L } _ { f i n a l }  \mathcal { R } _ { p } ( \ell _ { i } ^ { C E } , \ell _ { i } ^ { K L } , \ell _ { i } ^ { L 2 } , \ell _ { i } ^ { f e a t } , \ell _ { i } ^ { e n t } ; \omega _ { i } , \beta _ { c e } , \beta _ { d i s t } )$   
Update student S with $\nabla \mathcal { L } _ { f i n a l }$   
end

The standardization prevents arbitrary diferences in teacher logit scale from determining the feature target, while ψ learns the transfer into the student hidden space. This term encourages the student to recover the internal structure provided by the teacher.

Entropy Regularization: Finally, to discourage overly diffuse predictions, we add an entropy-minimization term:

$$
\ell _ { i } ^ { e n t } = \rho _ { i } \Big ( \mathcal { H } ( p _ { i , t } ^ { ( S ) } ) \Big ) .\tag{22}
$$

This term improves prediction sharpness and is particularly useful for open-ended surgical text generation, where underconfident decoding can otherwise quickly accumulate lexical drift over long responses.

Overall, the four losses above jointly instantiate hierarchical representation alignment as a unified multi-level distillation objective. They use the same token mask, per-sample reduction, and curriculum weight before entering the progressive objective schedule. By aligning the student with the teacher at the distribution, logit, and latent levels within the same optimization design, this objective provides a stronger and more reliable foundation for compressed surgical vision-language learning.

## 3.4. Progressive Distillation Objective

The final objective unifies the sample-level curriculum above with an objective-level schedule for progressive distillation and supervised learning within a single optimization formulation. These two mechanisms play distinct roles and should be treated separately: curriculum determines which samples should matter more at a given stage, whereas the progressive objective schedule determines which supervision signal should dominate at that stage. This design makes the interaction between sample pacing and supervision scheduling explicit and more systematically controlled in the final training objective.

Let $p \in [ 0 , 1 ]$ denote the normalized training progress. We define a stage coeficient $\alpha ( p )$ by a three-phase schedule with ratios $( r _ { 1 } , r _ { 2 } , r _ { 3 } )$ , which governs the transition dynamics across diferent training stages:

$$
\alpha ( p ) = \left\{ \begin{array} { l l } { 0 , } & { p \leq r _ { 1 } , } \\ { { p - r _ { 1 } } } & { } \\ { { r _ { 2 } } } & { { r _ { 1 } < p \leq r _ { 1 } + r _ { 2 } , } } \\ { 1 , } & { p > r _ { 1 } + r _ { 2 } . } \end{array} \right.\tag{23}
$$

This coeficient parameterizes the transition from teacherdominant optimization to task-dominant optimization. Based on this stage coeficient, we further define the corresponding interpolation weights as:

$$
\beta _ { d i s t } ( p ) = d _ { s } + ( d _ { e } - d _ { s } ) \alpha ( p ) , \qquad \beta _ { c e } ( p ) = c _ { s } + ( c _ { e } - c _ { s } ) \alpha ( p ) ,\tag{24}
$$

where $( d _ { s } , d _ { e } )$ and $( c _ { s } , c _ { e } )$ denote the start and end scales for distillation and cross-entropy, respectively. Early in training, $\beta _ { d i s t } ( p )$ is large and $\beta _ { c e } ( p )$ is small, so the student first learns to imitate the teacher. Then, the balance shifts toward groundtruth supervision, allowing the student to refine task-specific predictions after a stable multimodal transfer phase.

For sample i, we further define a combined coeficient:

$$
\omega _ { i } ( p ) = \tau _ { i } w _ { i } ^ { \prime } ( p ) ,\tag{25}
$$

where $\tau _ { i }$ is an optional task-level reweighting factor and $w _ { i } ^ { \prime } ( p )$ is the curriculum weight from Eq. (14). Let $\mathcal { B } _ { \mathrm { g l o b a l } }$ denote the global mini-batch and $N _ { \mathrm { g l o b a l } } = | \mathcal { B } _ { \mathrm { g l o b a l } } |$ its number of samples. We use the following weighted reduction:

$$
\mathcal { R } _ { p } ^ { \mathrm { g l o b a l } } = \frac { 1 } { N _ { \mathrm { g l o b a l } } } \sum _ { i \in \mathcal { B } _ { \mathrm { g l o b a l } } } \omega _ { i } ( p ) \ell _ { i } .\tag{26}
$$

Using the unweighted global sample count prevents curriculum weights from being normalized away by mini-batch composition. These token-level losses are first reduced to per-sample quantities and then aggregated with sample-aware weights. The supervised term is given by the token-level cross-entropy:

$$
\ell _ { i } ^ { C E } = \rho _ { i } \big ( - \log p _ { i , t } ^ { ( S ) } ( y _ { i , t } ) \big ) ,\tag{27}
$$

and the final objective becomes:

$$
\begin{array} { r l r } & { } & { \mathcal { L } _ { f i n a l } ( p ) = \beta _ { c e } ( p ) w _ { c e } \mathcal { R } _ { p } ^ { \mathrm { g l o b a l } } ( \{ \ell _ { i } ^ { C E } \} ) + \beta _ { d i s t } ( p ) \Big ( w _ { k l } \mathcal { R } _ { p } ^ { \mathrm { g l o b a l } } ( \{ \ell _ { i } ^ { K L } \} ) } \\ & { } & { + w _ { l 2 } \mathcal { R } _ { p } ^ { \mathrm { g l o b a l } } ( \{ \ell _ { i } ^ { L 2 } \} ) + w _ { f e a t } \mathcal { R } _ { p } ^ { \mathrm { g l o b a l } } ( \{ \ell _ { i } ^ { f e a t } \} ) + w _ { e n t } \mathcal { R } _ { p } ^ { \mathrm { g l o b a l } } ( \{ \ell _ { i } ^ { e n t } \} ) \Big ) , } \end{array}\tag{28}
$$

![](images/8db979f526a26c72f112deb243e5ff8c9f61abda84e1c55da797d04c08ef0f5e.jpg)  
Figure 3: Qualitative visualization of a representative same-image case across five dialogue paradigms. For the same surgical scene, the figure compares ground truth, teacher, and student outputs for single-phrase QA, visual QA, region-based QA, grounding QA, and detailed description. The example shows that the student preserves the main clinical semantics and spatial relationships while remaining compact.

![](images/b36789349d42df56e78080ec0ca440fbc49d8f404151a9ffcc45017cf3fc6715.jpg)  
Figure 4: Radar chart of task mean scores computed from Tables 2 and 3. Each axis averages the metrics within one dialogue paradigm.

where $w _ { ( \cdot ) }$ are fixed hyperparameters. In all experiments, we set $( w _ { c e } , w _ { k l } , w _ { l 2 } , w _ { f e a t } , w _ { e n t } ) = ( 0 . 5 , 1 . 0 , 0 . 5 , 0 . 2 , 0 . 0 5 )$ , and analyze the contribution of the corresponding alignment components in Table 16. Under this formulation, curriculum changes the relative contribution of samples, whereas the progressive objective schedule changes the relative contribution of objectives. One controls which samples dominate optimization, and the other controls which supervision signals dominate at each stage of training. This explicit separation of roles yields a stable framework for progressive distillation of surgical MLLMs. The complete training procedure of SurgPace is summarized in Algorithm 1.

Table 1: Distribution of the processed Surg-396K split used in this study.
<table><tr><td>Task family</td><td>Training instances</td><td>Proportion (%)</td></tr><tr><td>Single-phrase QA</td><td>156,983</td><td>34.03</td></tr><tr><td>Visual QA</td><td>109,960</td><td>23.84</td></tr><tr><td>Detailed description</td><td>29,960</td><td>6.49</td></tr><tr><td>Region-based QA</td><td>92,287</td><td>20.00</td></tr><tr><td>Grounding QA</td><td>72,107</td><td>15.63</td></tr><tr><td>Total</td><td>461,297</td><td>100.00</td></tr></table>

## 4. Experiments

## 4.1. Dataset

All experiments were conducted on the Surg-396K surgical multimodal instruction benchmark [46]. The benchmark is derived from four public datasets, namely CoPESD [48], Cholec80-VQA [40], EndoVis17-VQLA, and EndoVis18- VQLA [1]. The open-source benchmark provides various tasks in surgical scenes, including spatial grounding, instrument recognition, instrument counting, object position, target tissue, fine-grained motion, and direction, enabling the benchmark to cover both semantic understanding and localized reasoning. We evaluate SurgPace on five dialogue paradigms (single-phrase QA, visual QA, detailed description, regionbased QA, and grounding QA, as presented in Table 1) to assess whether a compact surgical MLLM can retain broad multimodal capability after distillation. For all experiments, we use a fixed processed split consisting of 461,297 training instances, 64,878 validation instances, and 64,856 test instances, covering 591,031 instruction-tuning QA pairs in total. The same split is used for all baselines, ablations, and final comparisons.

Table 2: Comparison experiments against state-of-the-art MLLM distillation methods on Single Phrase QA and Visual QA under a fixed 64,856-example evaluation source and greedy decoding.
<table><tr><td rowspan="2">Method</td><td colspan="4">Single Phrase QA</td><td colspan="5">Visual QA</td></tr><tr><td>Acc.</td><td>Prec.</td><td>Rec.</td><td>F1</td><td>BLEU-3</td><td>CIDEr</td><td>METEOR</td><td>ROUGE-1</td><td>ROUGE-L</td></tr><tr><td>MiniLLM [11]</td><td>0.8265</td><td>0.8384</td><td>0.8446</td><td>0.8402</td><td>0.3246</td><td>0.4676</td><td>0.7608</td><td>0.7521</td><td>0.7083</td></tr><tr><td>EfficientVLM [49]</td><td>0.8296</td><td>0.8412</td><td>0.8475</td><td>0.8431</td><td>0.3801</td><td>0.4830</td><td>0.7705</td><td>0.7632</td><td>0.7204</td></tr><tr><td>ULD [3]</td><td>0.8307</td><td>0.8406</td><td>0.8458</td><td>0.8421</td><td>0.3215</td><td>0.4683</td><td>0.7613</td><td>0.7539</td><td>0.7098</td></tr><tr><td>LLaVA-KD [4]</td><td>0.8241</td><td>0.8361</td><td>0.8421</td><td>0.8378</td><td>0.3689</td><td>0.4797</td><td>0.7686</td><td>0.7599</td><td>0.7152</td></tr><tr><td>LLaVA-MoD [41]</td><td>0.8279</td><td>0.8388</td><td>0.8466</td><td>0.8415</td><td>0.3582</td><td>0.4649</td><td>0.7757</td><td>0.7599</td><td>0.7000</td></tr><tr><td>CKD-MDSR [43]</td><td>0.8269</td><td>0.8385</td><td>0.8448</td><td>0.8404</td><td>0.0520</td><td>0.3074</td><td>0.5800</td><td>0.6026</td><td>0.5492</td></tr><tr><td>Ours (Student)</td><td>0.9004</td><td>0.9067</td><td>0.9092</td><td>0.9069</td><td>0.3764</td><td>0.5135</td><td>0.7841</td><td>0.7809</td><td>0.7366</td></tr><tr><td>EndoChat (Teacher)a [46]</td><td>0.5485</td><td>0.6046</td><td>0.7647</td><td>0.6325</td><td>0.3941</td><td>0.5067</td><td>0.7967</td><td>0.7869</td><td>0.7414</td></tr></table>

<sup>a</sup> EndoChat is reported as the teacher reference and is excluded from best/second-best marking.  
The best and second-best results among the student methods are bolded and underlined, respectively.  
All student rows use the common conversation template and generation protocol described in Section 4.2, including greedy decoding with num\_beams=1, max\_new\_tokens=256, and the same output normalization. No prompt augmentation for individual regions or task-specific decoding constraint is used; EndoChat is reported as the teacher reference.

Table 3: Comparison experiments against state-of-the-art MLLM distillation methods on Detailed Description, Region-Based QA, and Grounding QA under the same fixed evaluation protocol as Table 2.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Detailed Description</td><td colspan="4">Region-Based QA</td><td colspan="2">Grounding QA</td></tr><tr><td>ROUGE-L CIDEr</td><td>METEOR</td><td>ROUGE-1</td><td>ROUGE-L</td><td>AP50</td><td>mIoU</td></tr><tr><td>MiniLLM [11]</td><td>0.3345</td><td>0.2931</td><td>0.5782</td><td>0.5964</td><td>0.5566</td><td>0.5857</td><td>0.5104</td></tr><tr><td>EfficientVLM [49]</td><td>0.3284</td><td>0.3034</td><td>0.5875</td><td>0.6038</td><td>0.5661</td><td>0.5454</td><td>0.4887</td></tr><tr><td>ULD [3]</td><td>0.3134</td><td>0.3038</td><td>0.5921</td><td>0.6013</td><td>0.5600</td><td>0.5708</td><td>0.5022</td></tr><tr><td>LLaVA-KD [4]</td><td>0.3154</td><td>0.3046</td><td>0.5918</td><td>0.5978</td><td>0.5520</td><td>0.5636</td><td>0.4993</td></tr><tr><td>LLaVA-MoD [41]</td><td>0.3317</td><td>0.3053</td><td>0.5786</td><td>0.5943</td><td>0.5493</td><td>0.5742</td><td>0.5019</td></tr><tr><td>CKD-MDSR [43]</td><td>0.1157</td><td>0.2277</td><td>0.4845</td><td>0.5183</td><td>0.4844</td><td>0.4946</td><td>0.4563</td></tr><tr><td>Ours (Student)</td><td>0.3621</td><td>0.3311</td><td>0.6010</td><td>0.6217</td><td>0.5878</td><td>0.9625</td><td>0.8471</td></tr><tr><td>EndoChat (Teacher)a [46]</td><td>0.3307</td><td>0.3930</td><td>0.7049</td><td>0.7119</td><td>0.6659</td><td>0.9664</td><td>0.8789</td></tr></table>

<sup>a</sup> EndoChat is reported as the teacher reference and is excluded from best/second-best marking.  
The values are evaluated under the fixed protocol stated in Table 2.  
The best and second-best results among the student methods are bolded and underlined, respectively.

## 4.2. Implementation Details

All student methods, including SurgPace and the comparison methods, use the same 3B TinyLLaVA-style student architecture built on Phi-2 with a SigLIP vision tower and a twolayer GELU-based multimodal connector. All methods use the same pretrained Phi-2–SigLIP student initialization and the same EndoChat teacher for distillation. For parameter-eficient tuning, all student methods use LoRA on the language model, keep the vision tower frozen, and fully fine-tune the multimodal connector. The shared LoRA configuration uses rank 64, scaling factor α = 128, and bias=None. LoRA targets q\_proj, k\_proj, v\_proj, dense, mlp.fc1, and mlp.fc2 in the 32 language-model layers, giving 192 target modules. Thus, the parameter-eficient fine-tuning setup is shared; the methodspecific diferences arise mainly from the distillation objectives and training strategies. A lightweight learnable projection layer is introduced for teacher-student logit alignment. Input images are processed under a square aspect-ratio policy, visual features are extracted from the penultimate vision layer, and the maximum sequence length is set to 1024 tokens. All student methods use BF16 with DeepSpeed ZeRO-2 [34] and the same AdamW optimizer [27], learning rate $1 \times 1 0 ^ { - 4 }$ , cosine decay, and warm-up ratio 0.03. The main experiments are run for one epoch with a per-device batch size of 1 on two NVIDIA A100 80GB GPUs. The fixed processed split contains 461,297 training instances, 64,878 validation instances, and 64,856 test instances, and is used for all comparison methods, ablations, and final evaluations. Fine-tuning strategies difer by method, but all methods are based on supervised fine-tuning (SFT). Training budgets are not identical. Table 17 reports the per-method GPU-hours; for SurgPace, the reported total includes 52.443 GPU-hours of training and 3.11 GPU-hours of ofline dificulty scoring. Across all seven compared methods, the mean total is 71.572 GPU-hours, with a maximum of 165.514 GPUhours for LLaVA-KD and a minimum of 37.890 GPU-hours for ULD. For comparison, we evaluate against MiniLLM [11], EfficientVLM [49], ULD [3], LLaVA-KD [4], LLaVA-MoD [41], and CKD-MDSR [43], and report EndoChat [46] as the teacher reference. For the common generation protocol, all student methods use the same original questions and the common conversation template used in the TinyLLaVA-based implementation [18]. The shared settings include automatic insertion of one <image> token, greedy decoding with num\_beams=1 and max\_new\_tokens=256, skip\_special\_tokens=True, and identical whitespace and terminal stop-string handling. At the metric stage, outputs are lowercased, <image> is replaced with a space, and the same token normalization is applied before exact-match and overlap evaluation; ROUGE, BLEU/CIDEr/METEOR, and Grounding use the same respective evaluation procedures for all methods. The complexity scores are computed once ofline using forward passes through the teacher. Scoring all 591,031 instances required 3.11 GPU hours, and the resulting scores are reused throughout training.

Table 4: Robustness comparison across three seeds between SurgPace and the strongest baseline, EficientVLM, using the same ten task-spanning metrics reported in the ablation studies. Each cell gives the mean, with the sample standard deviation shown below, over seeds 0, 1, and 42 under the fixed common evaluation protocol.
<table><tr><td rowspan="2">Method</td><td colspan="2">Single Phrase QA</td><td colspan="3">Visual QA</td><td rowspan="2">Description</td><td colspan="2">Region-Based QA</td><td colspan="2">Grounding QA</td></tr><tr><td>Acc.</td><td>F1</td><td>BLEU-3</td><td>CIDEr</td><td>METEOR</td><td>ROUGE-L</td><td>CIDEr</td><td>METEOR</td><td>AP50 mIoU</td></tr><tr><td>EfficientVLM</td><td>0.8312</td><td>0.8447</td><td>0.3741</td><td>0.4850</td><td>0.7705</td><td>0.3259</td><td>0.3130</td><td>0.5871</td><td>0.5686</td><td>0.5032</td></tr><tr><td rowspan="2">Ours (Student)</td><td>±0.0014</td><td>±0.0014</td><td>±0.0056</td><td>±0.0018</td><td>±0.0016</td><td>±0.0057</td><td>±0.0084</td><td>±0.0007</td><td>±0.0229</td><td>±0.0136</td></tr><tr><td>0.8915</td><td>0.8990</td><td>0.3822</td><td>0.5119</td><td>0.7843</td><td>0.3589</td><td>0.3292</td><td>0.5999</td><td>0.9604</td><td>0.8424</td></tr><tr><td></td><td>±0.0101</td><td>±0.0092</td><td>±0.0050</td><td>±0.0014</td><td>±0.0042</td><td>±0.0033</td><td>±0.0038</td><td>±0.0016</td><td>±0.0031</td><td>±0.0069</td></tr></table>

The higher mean between the two student methods is bolded; the final model is highlighted in yellow.  
The 16-metric mean, computed independently for each seed and then summarized, is 0 6221 ± 0 0029 for EficientVLM and 0 6930 ± 0 0038 for SurgPace.

Table 5: Sensitivity to perturbations of the teacher logits used only for ofline dificulty scoring. All settings use EndoChat, seed 42, identical online distillation, and the same training and common evaluation protocols.
<table><tr><td>Offline score setting</td><td>16-metric avg.</td><td>Change</td></tr><tr><td>Original teacher logits</td><td>0.6955</td><td></td></tr><tr><td>Temperature calibration (T = 1.3)</td><td>0.6926</td><td>−0.0029 (−0.4%)</td></tr><tr><td>Noise on centered logits  $( \sigma _ { \mathrm { r e l } } = 0 . 0 5 )$ </td><td>0.6887</td><td>-0.0068 (-1.0%)</td></tr></table>

The configuration using the original logits is the final setting and is highlighted in yellow.

Table 6: Uniform LLM-as-a-Judge evaluation of all Detailed Description outputs using Qwen-3.8-Max at temperature 0. Each system is evaluated on all 3,747 Detailed Description samples.
<table><tr><td>Method</td><td>LLM Score ↑</td><td>Std</td><td>95% CI</td></tr><tr><td>MiniLLM</td><td>60.73</td><td>18.81</td><td>[60.12, 61.33]</td></tr><tr><td>EfficientVLM</td><td>68.98</td><td>20.35</td><td>[68.33, 69.60]</td></tr><tr><td>ULD</td><td>58.52</td><td>26.92</td><td>[57.71, 59.42]</td></tr><tr><td>LLaVA-KD</td><td>59.36</td><td>26.93</td><td>[58.51, 60.22]</td></tr><tr><td>LLaVA-MoD</td><td>67.47</td><td>18.89</td><td>[66.88, 68.07]</td></tr><tr><td>CKD-MDSR</td><td>1.77</td><td>6.21</td><td>[1.58, 1.97]</td></tr><tr><td>Ours (Student)</td><td>77.48</td><td>15.02</td><td>[77.00, 77.96]</td></tr><tr><td>EndoChat (Teacher)a</td><td>76.67</td><td>13.38</td><td>[76.24, 77.09]</td></tr></table>

<sup>a</sup> EndoChat is reported as the teacher reference and is excluded from best/second-best marking.  
The best and second-best LLM Scores among the student methods are bolded and underlined, respectively.  
The judge receives only the reference and generated text; it does not receive the image or model identity. All methods use the same EndoChat prompt, judge, temperature, seed, and sample set. All scores in this column are generated with this single configuration.

Table 7: Blinded clinical audit based on reference text across six dimensions using Qwen-3.8-Max on 300 shared stratified Detailed Description samples per system. Scores range from 0 to 4; higher is better, and a hallucination score of 4 denotes no unsupported content.
<table><tr><td>Audit dimension</td><td>EfficientVLM</td><td>Ours (Student)</td><td>EndoChat (Teacher)a</td></tr><tr><td>Overall (0–100)</td><td>64.92</td><td>72.70</td><td>63.97</td></tr><tr><td>Instrument</td><td>3.02</td><td>3.40</td><td>3.66</td></tr><tr><td>Tissue</td><td>2.91</td><td>3.19</td><td>2.79</td></tr><tr><td>Action/motion</td><td>2.65</td><td>2.62</td><td>1.36</td></tr><tr><td>Direction</td><td>2.24</td><td>2.45</td><td>1.75</td></tr><tr><td>Spatial</td><td>2.29</td><td>2.91</td><td>2.61</td></tr><tr><td>Hallucination</td><td>2.52</td><td>2.93</td><td>3.23</td></tr></table>

<sup>a</sup> EndoChat is reported as the teacher reference and is excluded from best/second-best marking.  
The best and second-best audit scores among the student methods are bolded and underlined, respectively.

For semantic evaluation of Detailed Description, we use the original EndoChat LLM-as-a-Judge prompt<sup>1</sup> and evaluate all eight systems, including EndoChat itself, with Qwen-3.8-Max. For each of the 3,747 Detailed Description test samples, the judge receives only the reference and generated text, is asked to assess accuracy and quality as a specialist in surgical robotics, and returns one integer from 0 to 100. We use temperature 0 and seed 42, and report the mean, sample standard deviation, and a 2,000-resample bootstrap 95% confidence interval. We separately conduct a blinded audit based on reference text across six dimensions using 300 shared samples stratified by dataset and quartile of reference length. The audit scores instrument identity/count, tissue/target relationship, action/motion, direction, spatial grounding/relations, and hallucination from 0 to 4; model identity is hidden and verbosity is not rewarded. The complete blinded-audit prompt, JSON output schema, scoring anchors, and recorded run configuration are provided in Supplementary Material, Section S1.

Table 8: Flagged error counts from the shared blinded clinical audit. Counts are reported out of the same 300 stratified Detailed Description samples for each system.
<table><tr><td>Flagged error type</td><td>Efficient VLM</td><td>Ours (Student)</td><td>EndoChat (Teacher)</td></tr><tr><td>Instrument identity/ count</td><td>127</td><td>83</td><td>46</td></tr><tr><td>Tissue/target relationship</td><td>87</td><td>70</td><td>109</td></tr><tr><td>Action/motion</td><td>110</td><td>151</td><td>273</td></tr><tr><td>Direction</td><td>218</td><td>183</td><td>227</td></tr><tr><td>Spatial grounding/ relation</td><td>233</td><td>132</td><td>143</td></tr><tr><td>Hallucination/ unsupported content</td><td>171</td><td>151</td><td>92</td></tr></table>

The same 300 source indices are used for all systems and are sampled by dataset and quartile of reference length within each dataset. Future clinical validation will use adjudication by clinicians with access to the original image.

Table 9: Representative SurgPace cases with low scores selected from the shared clinical audit using the predefined error dimensions.
<table><tr><td>Case and reference-generation discrepancy</td></tr><tr><td>CoPESD_000937_0697 Flagged dimensions: Instrument; hallucination. The reference identifies forceps and a knife as the two primary objects. The generation states that three main objects are present and adds unsupported scene framing; the main actions are preserved but spatial precision is</td></tr><tr><td>reduced. CoPESD_001815_0240 Flagged dimensions: Tissue; action; direction; spatial.</td></tr><tr><td>The reference places the forceps centrally and the knife at the lower right, dissecting toward the lower left. The generation moves the forceps to the left and changes the knife motion to upward and to the right, altering both</td></tr><tr><td>the action direction and spatial relation.</td></tr><tr><td></td></tr><tr><td>CoPESD_230943_0082 Flagged dimension: Hallucination. The reference explicitly states that the image is too unclear to confirm any</td></tr></table>

We additionally benchmark controlled text generation on one NVIDIA A100 80GB GPU. The 13B teacher and 3B student are both evaluated in FP16 for a precision-matched comparison. Both use batch size 1, 128 input tokens, 64 generated tokens, and greedy decoding. After three warm-up runs, we report the median over ten measured runs. The analytical FLOP proxy is 2P per token for the active text parameters and excludes the attention quadratic term, image encoder, and sampling overhead.

Table 10: Controlled text generation eficiency on one NVIDIA A100 80GB GPU. The 13B teacher and 3B student are both evaluated in FP16 for a precision-matched comparison; both use batch size 1, 128 input tokens, 64 output tokens, greedy decoding, three warm-up runs, and ten measured repeats.
<table><tr><td>Metric</td><td>13B Teacher (FP16)</td><td>3B Student (FP16)</td><td>Reduction / speedup</td></tr><tr><td>Parameters</td><td>13.02B</td><td>3.22B</td><td>4.04× fewer</td></tr><tr><td>Active text parameters</td><td>13.02B</td><td>2.78B</td><td>4.69× fewer</td></tr><tr><td>Request FLOP proxy for text backbone</td><td>5.00×1012</td><td> $1 . 0 7 \times 1 0 ^ { 1 2 }$ </td><td>4.68× fewer</td></tr><tr><td>Loaded GPU memory</td><td>26.03 GB</td><td>6.46 GB</td><td>4.03× less</td></tr><tr><td>Peak GPU memory</td><td>26.20 GB</td><td>6.52 GB</td><td>4.02× less</td></tr><tr><td>Prefill / TTFT</td><td>31.95 ms</td><td>18.55 ms</td><td>1.72× faster</td></tr><tr><td>Median decode latency</td><td>23.51</td><td>16.50</td><td>1.42× faster</td></tr><tr><td>P95 decode latency</td><td>ms/token 25.28 ms/token</td><td>ms/token 21.55 ms/token</td><td>1.17× faster</td></tr><tr><td>Output throughput</td><td>42.22 token/s</td><td>58.53 token/s</td><td>1.39× higher</td></tr><tr><td>End-to-end latency</td><td>1,547.83 ms</td><td>1,111.67 ms</td><td>1.39× faster</td></tr><tr><td>Input throughput</td><td>4,005.97 token/s</td><td>6,899.23</td><td>1.72× higher</td></tr><tr><td>Total throughput</td><td>124.04 token/s</td><td>token/s 172.71 token/s</td><td>1.39× higher</td></tr></table>

The FLOP proxy uses 2P per token for active text parameters and excludes quadratic attention, image encoding, and sampling overhead.

Unless otherwise stated, every ablation row is trained with seed 42 under the same implementation, data split, and common evaluation protocol; each row changes only the factor named in its table.

## 4.3. Experimental Results

Table 2, Table 3, and Figure 3 summarize the quantitative and qualitative comparison across five dialogue paradigms under the fixed common evaluation protocol. The SurgPace student obtains the highest unweighted average over the 16 reported metrics (0.6955) and is the best student method on 15 individual metrics. It ranks first among student methods on all Single Phrase QA metrics, Detailed Description ROUGE-L, Visual QA CIDEr, METEOR, and both ROUGE measures, all Region-Based QA metrics, and both Grounding QA met rics. EficientVLM obtains 0.3801 on Visual QA BLEU-3, compared with 0.3764 for SurgPace.

To assess variability across runs, we repeated SurgPace and the strongest baseline, EficientVLM, using seeds 0, 1, and 42 under the same common evaluation protocol and the same train ing configuration except for the distillation objective. As reported in Table 4, the unweighted mean over all 16 evaluation metrics is 0 6930±0 0038 for SurgPace and 0 6221±0 0029 for EficientVLM (mean ± sample standard deviation). The table displays the same ten task-spanning metrics used in the ablation studies, while the aggregate is computed over all metrics in Tables 2 and 3. The mean across the three seeds is higher for SurgPace on all 16 individual metrics, including Visual QA BLEU-3 (0 3822 ± 0 0050 versus 0 3741 ± 0 0056). The seed-42 BLEU-3 diference is smaller than the across-seed standard deviation of either method and does not persist in the repeatedrun comparison, indicating variation between checkpoints on this metric.

Table 11: Ablation on complexity score fusion in the ofline scoring stage. The variants correspond to $\operatorname { E q . }$ (10), which fuses the teacherderived confidence score $s _ { \mathrm { c o n f } }$ computed in the standard Euclidean probability space and the Lorentz geometric score $s _ { \mathrm { l o r e n t z } }$ into the final complexity score $s _ { i } .$ The adopted final configuration is highlighted in yellow.
<table><tr><td rowspan="2">Variant</td><td colspan="2">Single Phrase QA</td><td colspan="3">Visual QA</td><td rowspan="2">Detailed Description</td><td colspan="2">Region-Based QA</td><td colspan="2">Grounding QA</td></tr><tr><td>Acc.</td><td>F1</td><td>BLEU-3</td><td></td><td>CIDEr METEOR</td><td>ROUGE-L</td><td>CIDEr</td><td>METEOR AP50</td><td>mIoU</td></tr><tr><td>Sconf</td><td>0.8826</td><td>0.8908</td><td>0.3662</td><td>0.5102</td><td>0.7799</td><td>0.3548</td><td>0.3260</td><td>0.5984</td><td>0.9595</td><td>0.8350</td></tr><tr><td>Slorentz</td><td>0.8997</td><td>0.9066</td><td>0.3737</td><td>0.5099</td><td>0.7817</td><td>0.3587</td><td>0.3257</td><td>0.5988</td><td>0.9614</td><td>0.8420</td></tr><tr><td>Smax</td><td>0.8925</td><td>0.8992</td><td>0.3741</td><td>0.5109</td><td>0.7821</td><td>0.3561</td><td>0.3296</td><td>0.6018</td><td>0.9618</td><td>0.8390</td></tr><tr><td>Smean</td><td>0.8971</td><td>0.9041</td><td>0.3754</td><td>0.5111</td><td>0.7834</td><td>0.3591</td><td>0.3289</td><td>0.6017</td><td>0.9624</td><td>0.8439</td></tr><tr><td>Smin</td><td>0.9004</td><td>0.9069</td><td>0.3764</td><td>0.5135</td><td>0.7841</td><td>0.3621</td><td>0.3311</td><td>0.6010</td><td>0.9625</td><td>0.8471</td></tr></table>

The best and second-best results within the table are bolded and underlined, respectively.

Table 12: Ablation on curriculum order. Diferent variants change the ordering policy induced by the complexity-aware pacing strategy. The adopted final configuration is highlighted in yellow.
<table><tr><td rowspan="2">Variant</td><td colspan="2">Single Phrase QA</td><td colspan="3">Visual QA</td><td colspan="2">Detailed Description Region-Based QA</td><td colspan="3">Grounding QA</td></tr><tr><td>Acc.</td><td>F1</td><td>BLEU-3</td><td></td><td>CIDEr METEOR</td><td>ROUGE-L</td><td>CIDEr METEOR</td><td></td><td>AP50</td><td>mIoU</td></tr><tr><td>mixed</td><td>0.8959</td><td>0.9033</td><td>0.3520</td><td>0.4897</td><td>0.7702</td><td>0.3601</td><td>0.3309</td><td>0.5997</td><td>0.9420</td><td>0.8434</td></tr><tr><td>hard-to-easy</td><td>0.8896</td><td>0.8964</td><td>0.3758</td><td>0.5097</td><td>0.7808</td><td>0.3565</td><td>0.3276</td><td>0.6009</td><td>0.9607</td><td>0.8357</td></tr><tr><td>easy-to-hard</td><td>0.9004</td><td>0.9069</td><td>0.3764</td><td>0.5135</td><td>0.7841</td><td>0.3621</td><td>0.3311</td><td>0.6010</td><td>0.9625</td><td>0.8471</td></tr></table>

The best and second-best results within the table are bolded and underlined, respectively.

The teacher reference has a diferent task profile. SurgPace is higher on the four Single Phrase QA metrics, Visual QA CIDEr, and Detailed Description ROUGE-L, while EndoChat is higher on the other four Visual QA overlap metrics, all four Region-Based QA metrics, and both Grounding QA metrics. Single Phrase QA rewards the canonical short target form, which the student produces more consistently; semantically related but diferently formatted teacher responses are penalized by these exact-answer measures. The teacher’s higher values on several overlap metrics for open answers, regional reasoning, and spatial localization are consistent with its larger capacity. Thus, the teacher–student comparison reflects both answer-format sensitivity in short-answer evaluation and a remaining capacity gap on more open and spatially demanding tasks.

To assess sensitivity of the score derived from the teacher, we retain EndoChat and perturb only the logits used to construct the ofline dificulty scores. As reported in Table 5, rescaling the teacher logits with temperature $T = 1 . 3$ yields a mean of 0.6926 over the 16 metrics, an absolute reduction of 0.0029 (0.4%) from 0.6955 with the original scores. Adding zero-mean Gaussian noise with 5% relative standard deviation to centered teacher logits yields 0.6887, a reduction of 0.0068 (1.0%). Online distillation and all other training and evaluation settings remain unchanged. These values quantify the sensitivity of the curriculum to calibration and numerical perturbations of En-

doChat’s ofline scoring logits.

The advantage of SurgPace further extends to tasks requiring structured reasoning and fine-grained visual grounding. Among student methods, it achieves the best ROUGE-L score on Detailed Description (0.3621), reflecting improved coherence in long-form generation. On Region-Based QA, it ranks first among student methods across all reported metrics, including CIDEr (0.3311) and METEOR (0.6010), indicating strong capability in region-aware language generation and cross-modal alignment. On Grounding QA, the SurgPace student achieves the best results among student methods in AP50 (0.9625) and mIoU (0.8471), suggesting that spatial reasoning ability is effectively preserved after distillation. These observations indicate that SurgPace enables a compact model to generalize well across heterogeneous multimodal tasks, including recognition, generation, regional reasoning, and grounding, while maintain ing stable performance across diferent evaluation settings. Additionally, Figure 4 provides a compact summary of the results across the five dialogue paradigms. It further shows that SurgPace achieves the strongest overall performance, consistent with the above-mentioned quantitative comparisons.

Table 6 shows that SurgPace achieves the highest LLM Score (77.48), followed by EndoChat (76.67) and EficientVLM (68.98). The score is 78.55 on CoPESD and 59.45 on EndoVis (3,537 and 210 samples, respectively).

The clinical audit in Table 7 provides a diagnostic view of Detailed Description quality. SurgPace has the highest overall score (72.70) and the strongest tissue, direction, and spatial scores; EficientVLM obtains 2.65 on action/motion, and EndoChat obtains 3.66 on instrument identification and 3.23 on hallucination control. The corresponding error counts show the same local diferences: EficientVLM has fewer action/motion errors, while EndoChat has fewer instrument and hallucination errors. The shared training objective favors a balance across tasks, while these three dimensions remain local strengths of the comparison systems. The counts in Table 8 and representative cases in Table 9 identify action, motion direction, and unsupported details as the main remaining weaknesses.

Table 13: Ablation on the curriculum pacing function. The pacing function is the continuous weighting function in Sec. 3.2 that converts the ofline complexity score s<sub>i</sub> into the progress-dependent sample weight w<sup>′</sup>(p) through Eqs. (11)-(14). Diferent variants compare one-pass, baby-step, root, geometric, and linear pacing under the same score signal. The adopted final configuration is highlighted in yellow.
<table><tr><td rowspan="2">Variant</td><td colspan="2">Single Phrase QA</td><td colspan="3">Visual QA</td><td colspan="2">Detailed Description</td><td colspan="2">Region-Based QA Grounding QA</td></tr><tr><td>Acc.</td><td>F1</td><td>BLEU-3</td><td>CIDEr</td><td>METEOR</td><td>ROUGE-L</td><td>CIDEr METEOR</td><td>AP50</td><td>mIoU</td></tr><tr><td>One-pass</td><td>0.6343</td><td>0.7369</td><td>0.3238</td><td>0.4692</td><td>0.7738</td><td>0.3481</td><td>0.3098</td><td>0.5744</td><td>0.9351</td><td>0.8049</td></tr><tr><td>Baby step</td><td>0.8543</td><td>0.8655</td><td>0.3123</td><td>0.4799</td><td>0.7648</td><td>0.3449</td><td>0.3105</td><td>0.5932</td><td>0.9280</td><td>0.8016</td></tr><tr><td>Root</td><td>0.8874</td><td>0.8949</td><td>0.3749</td><td>0.5106</td><td>0.7798</td><td>0.3589</td><td>0.3280</td><td>0.5984</td><td>0.9597</td><td>0.8360</td></tr><tr><td>Geom</td><td>0.8853</td><td>0.8933</td><td>0.3778</td><td>0.5095</td><td>0.7795</td><td>0.3587</td><td>0.3232</td><td>0.5968</td><td>0.9612</td><td>0.8396</td></tr><tr><td>Linear</td><td>0.9004</td><td>0.9069</td><td>0.3764</td><td>0.5135</td><td>0.7841</td><td>0.3621</td><td>0.3311</td><td>0.6010</td><td>0.9625</td><td>0.8471</td></tr></table>

The best and second-best results within the table are bolded and underlined, respectively.  
One-pass activates three disjoint dificulty bins in sequence, with each sample used only in its assigned phase. W/o curriculum in Table 14 gives every sample unit weight at every training step.

Table 14: Ablation on the sample-level curriculum additions to the progressive objective schedule, which is retained in every row. Objective schedule only uses unit sample weights; adding the static score introduces fixed weights from the informative complexity estimate; and the full setting further makes thes weights progress-dependent through linear pacing. The adopted final configuration is highlighted in yellow.
<table><tr><td rowspan="2">Variant</td><td colspan="2">Single Phrase QA</td><td colspan="3">Visual QA</td><td>Detailed Description</td><td colspan="2">Region-Based QA</td><td colspan="2">Grounding QA</td></tr><tr><td>Acc.</td><td>F1</td><td>BLEU-3</td><td>CIDEr</td><td>METEOR</td><td>ROUGE-L</td><td>CIDEr</td><td>METEOR</td><td>AP50</td><td>mIoU</td></tr><tr><td>Objective schedule only</td><td>0.8928</td><td>0.9000</td><td>0.3783</td><td>0.5099</td><td>0.7805</td><td>0.3543</td><td>0.3289</td><td>0.5980</td><td>0.9619</td><td>0.8365</td></tr><tr><td>+ static complexity score</td><td>0.8978</td><td>0.9044</td><td>0.3692</td><td>0.5096</td><td>0.7790</td><td>0.3577</td><td>0.3295</td><td>0.6051</td><td>0.9613</td><td>0.8399</td></tr><tr><td>Full (score + pacing)</td><td>0.9004</td><td>0.9069</td><td>0.3764</td><td>0.5135</td><td>0.7841</td><td>0.3621</td><td>0.3311</td><td>0.6010</td><td>0.9625</td><td>0.8471</td></tr></table>

The best and second-best results within the table are bolded and underlined, respectively.

Table 10 quantifies the compression benefit under this controlled decoding protocol. The student uses 4.04× fewer parameters, reduces loaded and peak GPU memory from 26.03/26.20 GB to 6.46/6.52 GB (4.03×/4.02× lower), and improves median output throughput from 42.22 to 58.53 tokens/s. End-to-end latency for the fixed 128/64-token request decreases from 1.55 s to 1.11 s. These measurements support memory eficiency and faster decoding under controlled benchmarks.

## 4.4. Ablation Studies

Efects of the Complexity Score Fusion. Table 11 studies the ofline fusion rule introduced in Sec. 3.2 and Eq. (10). Among the single-score variants, s<sub>lorentz</sub> performs better than $s _ { \mathrm { c o n f } }$ on most metrics, showing that the geometric score provides an efective alternative to confidence-based dificulty estimation. Combining the two signals further improves the overall results. The proposed minimum fusion $s _ { \mathrm { m i n } }$ achieves the highest mean across the ten metrics (0.6585), followed by mean fusion (0.6567), Lorentz-only (0.6558), maximum fusion (0.6547), and confidence-only (0.6504).

The diferences among s<sub>min</sub>, s<sub>mean</sub>, and $s _ { \mathrm { m a x } }$ mainly arise when the two complexity estimates disagree. Taking the larger score, $s _ { \mathrm { m a x } }$ applies a stricter notion of easiness, while averaging them in $s _ { \mathrm { m e a n } }$ produces a more balanced estimate. The slightly higher Region-Based QA METEOR under $s _ { \mathrm { m a x } }$ suggests that this conservative treatment can be useful for some region-conditioned responses, where uncertain samples may benefit from being introduced later in the curriculum. By comparison, the permissive criterion of $s _ { \mathrm { m i n } }$ avoids assigning high complexity solely because one of the two signals is uncertain, which may explain its more consistent performance across the remaining tasks. We therefore adopt $s _ { \mathrm { m i n } }$ as the final fusion configuration.

Efects of the Curriculum Order. Table 12 examines how the ordering of samples afects curriculum training. The proposed easy-to-hard order achieves the best performance on all ten metrics. In contrast, reversing the curriculum to hard-toeasy consistently reduces performance, while mixed ordering also remains below easy-to-hard across all task families. These results show that the benefit is associated with the direction of the curriculum rather than simply exposing the model to samples with diferent complexity levels.

The three strategies difer in how the estimated complexity is used during training. Mixed ordering does not impose a consistent progression over dificulty, whereas hard-to-easy emphasizes more dificult samples first. Easy-to-hard instead prioritizes lower-complexity samples early and gradually increases the contribution of harder samples as training proceeds. Its consistent advantage suggests that establishing the teacher–student transfer on easier samples before increasing sample complexity provides a more suitable optimization trajectory for distillation. We therefore adopt easy-to-hard ordering in the final configuration.

Table 15: Ablation on the phase transition design in the progressive objective schedule. In Sec. 3.4 and Eqs. $( 2 3 ) ‐ ( 2 4 ) .$ the transition is defined by the stage coeficient α(p), which shifts supervision from teacher-dominant to task-dominant optimization through $\beta _ { \mathrm { d i s t } } ( p )$ and $\beta _ { \mathrm { c e } } ( p ) .$ . Here, Discrete denotes fixed stage partitions with hard phase ratios, whereas Continuous denotes the smooth interpolation used in Eq. (23). The adopted final configuration is highlighted in yellow.
<table><tr><td rowspan="2">Variant</td><td colspan="2">Single Phrase QA</td><td colspan="3">Visual QA</td><td rowspan="2">Detailed Description</td><td colspan="2">Region-Based QA</td><td colspan="2">Grounding QA</td></tr><tr><td>Acc.</td><td>F1</td><td>BLEU-3</td><td>CIDEr METEOR</td><td></td><td>ROUGE-L</td><td>CIDEr METEOR</td><td>AP50</td><td>mIoU</td></tr><tr><td>Discrete (3,3,4)</td><td>0.8999</td><td>0.9065</td><td>0.3762</td><td>0.5120</td><td>0.7826</td><td>0.3592</td><td>0.3294</td><td>0.5966</td><td>0.9612</td><td>0.8409</td></tr><tr><td>Discrete (2,6,2)</td><td>0.8963</td><td>0.9033</td><td>0.3685</td><td>0.5103</td><td>0.7801</td><td>0.3581</td><td>0.3284</td><td>0.5961</td><td>0.9596</td><td>0.8374</td></tr><tr><td>Continuous</td><td>0.9004</td><td>0.9069</td><td>0.3764</td><td>0.5135</td><td>0.7841</td><td>0.3621</td><td>0.3311</td><td>0.6010</td><td>0.9625</td><td>0.8471</td></tr></table>

The best and second-best results within the table are bolded and underlined, respectively.

Table 16: Ablation on the Hierarchical Representation Alignment objective defined in Sec. 3.3 and used in Eq. (28). Cross-entropy is kept fixed in all variants, while each row removes one auxiliary component from the alignment objective. Removing the KL or L2 terms causes severe degradation and is omitted.
<table><tr><td rowspan="2">Variant</td><td colspan="2">Single Phrase QA</td><td colspan="3">Visual QA</td><td>Detailed Description</td><td colspan="2">Region-Based QA</td><td colspan="2">Grounding QA</td></tr><tr><td>Acc.</td><td>F1</td><td>BLEU-3</td><td>CIDEr</td><td>METEOR</td><td>ROUGE-L</td><td>CIDEr</td><td>METEOR</td><td>AP50</td><td>mIoU</td></tr><tr><td>w/o latent alignment</td><td>0.8909</td><td>0.8986</td><td>0.3745</td><td>0.5102</td><td>0.7796</td><td>0.3568</td><td>0.3284</td><td>0.6001</td><td>0.9607</td><td>0.8361</td></tr><tr><td>w/o entropy regularization</td><td>0.8843</td><td>0.8920</td><td>0.3780</td><td>0.5097</td><td>0.7794</td><td>0.3532</td><td>0.3231</td><td>0.5967</td><td>0.9577</td><td>0.8301</td></tr><tr><td>full alignment</td><td>0.9004</td><td>0.9069</td><td>0.3764</td><td>0.5135</td><td>0.7841</td><td>0.3621</td><td>0.3311</td><td>0.6010</td><td>0.9625</td><td>0.8471</td></tr></table>

The best and second-best results within the table are bolded and underlined, respectively. The final model (Full Alignment) is highlighted in yellow.

Table 17: Comparison of reported training budgets and ofline dificultyscoring cost for the seven student methods. The SurgPace total includes both components.
<table><tr><td>Model</td><td>GPU-hours</td></tr><tr><td>MiniLLM</td><td>38.481</td></tr><tr><td>EfficientVLM</td><td>75.777</td></tr><tr><td>ULD</td><td>37.890</td></tr><tr><td>LLaVA-KD</td><td>165.514</td></tr><tr><td>LLaVA-MoD</td><td>81.835</td></tr><tr><td>CKD-MDSR</td><td>45.956</td></tr><tr><td>SurgPace (ours; incl. offline scoring)</td><td>55.553</td></tr><tr><td>Mean (all seven methods)</td><td>71.572</td></tr></table>

The SurgPace value is 52.443 GPU-hours for training plus 3.11 GPUhours for ofline dificulty scoring. The mean is computed across the seven student methods; LLaVA-KD has the maximum and ULD has the minimum reported budget.

Efects of the Curriculum Pacing Function. In SurgPace, the pacing function is the continuous weighting function defined in Sec. 3.2. It maps the ofline complexity score s<sub>i</sub> to the progress-dependent sample weight $w _ { i } ^ { \prime } ( p )$ through Eqs. (11)- (14). In practice, it controls how training moves from emphasizing easy samples to exposing the model to a more balanced sample set. We compare one non-progressive variant (one-pass) and four progressive schedules (i.e., baby-step, root, geometric, and linear pacing) with the same score signal.

The pacing study in Table 13 examines gradual reweighting and disjoint exposure. One-pass partitions the score range into three disjoint bins, activates only the current bin, and does not replay samples from earlier bins. The w/o curriculum setting assigns unit weight to every sample at every step. The onepass design reduces efective exposure per sample and introduces abrupt changes in training support, yielding a mean of 0.5910 across the ten metrics; w/o curriculum obtains 0.6541.

Among the progressive alternatives, linear pacing has the highest mean across the ten metrics (0.6585) and is best on nine of them. Geometric pacing is higher only on Visual QA BLEU-3 (0.3778 versus 0.3764). The schedules difer in how quickly they relax easy-first weighting, and this isolated reversal indicates that n-gram precision is sensitive to that relaxation rate. Root pacing is the next most consistent alternative (0.6529), whereas baby-step pacing is weaker, consistent with its abrupt cumulative support changes. We select linear pacing because its gradual transition gives the strongest aggregate and the best result on the other nine metrics.

Efects of Complexity Scoring and Online Pacing. Table 14 isolates the sample-level additions while retaining the progressive objective schedule in every row. The objective schedule only control assigns unit weight to every sample. Adding the informative complexity score as a static weight raises the mean across the ten metrics from 0.6541 to 0.6554, and further introducing linear progress-dependent pacing raises it to 0.6585. The full setting is best on eight of ten metrics. Objective schedule only obtains 0.3783 on Visual QA BLEU-3, and static scoring obtains 0.6051 on Region-Based QA ME-TEOR. The first local gain occurs without changing the sample distribution, while the second occurs when the dificulty bias is retained throughout training. Annealing the score-dependent weights restores balanced exposure late in training and gives the strongest aggregate, with a trade-of on these two individual overlap metrics.

Efects of the Phase Transition Design. Table 15 shows that the phase transition design also afects performance. In Sec. 3.4 and Eq. (23), the stage coeficient α(p) determines the supervision scales $\beta _ { \mathrm { d i s t } } ( p )$ and $\beta _ { \mathrm { c e } } ( p )$ in Eq. (24). Continuous transition changes this balance smoothly, whereas the two discrete variants use hard stage boundaries. The discrete (3,3,4) schedule remains competitive with a mean of 0.6564 across the ten metrics, while discrete (2,6,2) obtains 0.6538.

The continuous transition achieves the highest mean across the ten metrics (0.6585) and the best value on all ten. This consistent result supports smooth interpolation as the final setting and suggests that avoiding abrupt objective changes benefits the student across recognition, generation, regional reasoning, and grounding.

Efects of the Hierarchical Representation Alignment Objective. We further study the contribution of each loss term in the hierarchical representation alignment objective by removing one loss from the full setting at a time. As defined in Sec. 3.3 and Eq. (28), the final objective combines supervised crossentropy with distribution alignment $\ell _ { i } ^ { K L }$ , logit alignment $\ell _ { i } ^ { L 2 }$ latent alignment $\ell _ { i } ^ { f e a t }$ , and entropy regularization $\ell _ { i } ^ { e n t }$ . We first observe that removing the main supervision terms KL or L2 leads to severe degradation across all task families. We therefore omit these unstable cases from Table 16 and focus on the two informative auxiliary losses.

As shown in Table 16, full alignment has the highest mean across the ten metrics (0.6585), compared with 0.6536 without latent alignment and 0.6504 without entropy regularization. It is best on nine of ten metrics. Removing entropy regularization yields 0.3780 on Visual QA BLEU-3, AP50 of 0.9577, and mIoU of 0.8301; the full model obtains 0.3764, 0.9625, and 0.8471, respectively. Removing latent alignment yields 0.9607 AP50 and 0.8361 mIoU. BLEU-3 measures one aspect of local n-gram overlap and is not expected to vary monotonically with output-distribution sharpness. Its 0.0016 increase without entropy regularization is therefore treated as a local metric trade-of, while the reductions on the other nine metrics support retaining the full objective. These results associate latent alignment with grounded multimodal structure and entropy regularization with broad decoding stability.

## 4.5. Discussion & Limitations

The experimental results show that surgical MLLM distillation benefits from coordinated design at both the sample level and the objective level. SurgPace achieves the strongest aggregate profile across the five dialogue paradigms. Its configuration combines complexity estimation from teacher responses, easy-to-hard curriculum order, progressive pacing, continuous phase transition, and hierarchical representation alignment. This coordinated optimization retains broad multimodal ability in heterogeneous surgical tasks by controlling sample exposure and supervision balance throughout training.

The study has several limitations. (i) Both $s _ { \mathrm { c o n f } }$ and $s _ { \mathrm { l o r e n t z } }$ are computed ofline from teacher responses, so curriculum quality depends on teacher quality. Table 5 quantifies the efect of temperature recalibration and logit noise; analyses with alternative, weaker, and diferently biased teachers remain important future work. (ii) The current curriculum uses fixed ofline scores and a predefined pacing rule, and its sample weighting does not adapt to changes in student learning status during training. (iii) The eficiency study in Table 10 uses one NVIDIA A100 80GB GPU and precision-matched FP16 configurations; measurements on representative resource-constrained hardware remain future work. (iv) The clinical audit is a blinded LLM comparison against reference text without access to the original image. Prospective clinical validation requires adjudication by surgical experts using the original images. Future work includes more robust dificulty estimation from teacher responses, curriculum control that adapts to the student, eficiency bench marking on diferent hardware, and validation by clinicians.

## 5. Conclusion

This paper presents SurgPace, a complexity-aware continuous curriculum framework for progressive distillation of surgical multimodal large language models. SurgPace combines offline teacher-derived complexity estimation, continuous easyto-hard sample pacing, hierarchical representation alignment, and progressive objective scheduling within one training framework. Experiments across five dialogue paradigms show strong aggregate performance, and the ablation studies identify the contributions and diferences among the individual components. The semantic and eficiency evaluations further characterize output quality and computational benefits. Overall, the results support stable sample pacing and multi-level teacher– student alignment for efective surgical MLLM compression.

## Author Contribution Statement

Zhicheng He: Conceptualization, Data curation, Formal analysis, Methodology, Software, Validation, Writing – original draft. Xiaohan Xing: Conceptualization, Investigation, Methodology. Hongqiu Wang: Software, Validation, Writing – review & editing. Kun Yuan: Methodology, Software, Validation. Guankun Wang: Data curation, Methodology, Software, Validation. Yanheng Li: Investigation, Validation. Haoxuan Che: Resources, Validation. Nassir Navab: Project administration, Supervision. Hongliang Ren: Conceptualization, Project administration, Resources. Long Bai: Conceptu alization, Investigation, Methodology, Project administration, Resources, Supervision, Writing – original draft.

## Declaration

During the preparation of this work, the author used the Qwen model in order to refine the manuscript. After using this tool/service, the author reviewed and edited the content as needed and takes full responsibility for the content of the published article.

## References

[1] Bai, L., Islam, M., Ren, H., 2023a. CAT-ViL: Coattention gated vision-language embedding for visual

question localized-answering in robotic surgery, in: Medical Image Computing and Computer Assisted Intervention – MICCAI 2023, Springer. pp. 397–406. doi:10.1007/ 978-3-031-43996-4\_38.

[2] Bai, L., Islam, M., Seenivasan, L., Ren, H., 2023b. Surgical-vqla: Transformer with gated vision-language embedding for visual question localized-answering in robotic surgery. arXiv preprint arXiv:2305.11692 .

[3] Boizard, N., Haddad, K.E., Hudelot, C., Colombo, P., 2024. Towards cross-tokenizer distillation: the universal logit distillation loss for llms. arXiv preprint arXiv:2402.12030 .

[4] Cai, Y., Zhang, J., He, H., He, X., Tong, A., Gan, Z., Wang, C., Xue, Z., Liu, Y., Bai, X., 2025. Llava-kd: A framework of distilling multimodal large language models, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 239–249.

[5] Cao, J., Zhang, Y., Huang, T., Lu, M., Zhang, Q., An, R., Ma, N., Zhang, S., 2025. Move-kd: Knowledge distillation for vlms with mixture of visual encoders, in: Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 19846–19856.

[6] Chen, Z., Luo, X., Wu, J., Chan, D.T.M., Lei, Z., Ourselin, S., Liu, H., 2024a. SurgFC: Multimodal surgical function calling framework on the demand of surgeons, in: 2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), pp. 3076–3081. doi:10.110 9/BIBM62325.2024.10821976.

[7] Chen, Z., Luo, X., Wu, J., Chan, D.T.M., Lei, Z., Ourselin, S., Liu, H., 2024b. Vs-assistant: versatile surgery assistant on the demand of surgeons. arXiv preprint arXiv:2405.08272 URL: https://arxiv.org/abs/24 05.08272.

[8] Chen, Z., Luo, X., Yuan, K., Wu, J., Chan, D.T.M., Navab, N., Liu, H., Lei, Z., Luo, J., 2025. SurgLLM: A versatile large multimodal model with spatial focus and temporal awareness for surgical video understanding. arXiv preprint arXiv:2509.00357 URL: https://arxiv.org/ abs/2509.00357.

[9] Dong, W., Ge, J., Dong, W., Motani, M., 2026. Llama32- med: Parameter-eficient adaptation of multimodal llms for medical visual question answering, in: Medical Imaging with Deep Learning.

[10] Fang, L., Yu, X., Cai, J., Chen, Y., Wu, S., Liu, Z., Yang, Z., Lu, H., Gong, X., Liu, Y., et al., 2026. Knowledge distillation and dataset distillation of large language models: Emerging trends, challenges, and future directions. Artificial Intelligence Review 59, 17.

[11] Gu, Y., Dong, L., Wei, F., Huang, M., 2024. MiniLLM: Knowledge distillation of large language models, in: The

Twelfth International Conference on Learning Representations.

[12] Gupta, S., Karmalkar, S., 2025. Eficient knowledge distillation via curriculum extraction. arXiv preprint arXiv:2503.17494 URL: https://arxiv.org/abs/25 03.17494.

[13] Hao, P., Li, S., Wang, H., Kou, Z., Zhang, J., Yang, G., Zhu, L., 2025a. Surgery-r1: Advancing surgical-vqla with reasoning multimodal large language model via reinforcement learning. arXiv preprint arXiv:2506.19469 .

[14] Hao, P., Wang, H., Yang, G., Zhu, L., 2025b. Enhancing visual reasoning with llm-powered knowledge graphs for visual question localized-answering in robotic surgery. IEEE Journal of Biomedical and Health Informatics .

[15] He, J., Li, P., Liu, G., Zhong, S., 2025. Parameter-eficient fine-tuning medical multimodal large language models for medical visual grounding, in: 2025 IEEE 22nd International Symposium on Biomedical Imaging (ISBI), IEEE. pp. 1–5.

[16] Holland, R., Taylor, T.R., Holmes, C., Riedl, S., Mai, J., Patsiamanidi, M., Mitsopoulou, D., Hager, P., Müller, P., Paetzold, J.C., et al., 2025. Specialized curricula for training vision language models in retinal image analysis. npj Digital Medicine 8, 532.

[17] Javaheripi, M., Bubeck, S., 2023. Phi-2: The surprising power of small language models. Microsoft Research Blog. URL: https://www.microsoft.com/en-us/ research/blog/phi-2-the-surprising-power-o f-small-language-models/. published December 12, 2023.

[18] Jia, J., Hu, Y., Weng, X., Shi, Y., Li, M., Zhang, X., Zhou, B., Liu, Z., Luo, J., Huang, L., et al., 2024. Tinyllava factory: A modularized codebase for small-scale large multimodal models. arXiv preprint arXiv:2405.11788 .

[19] Jiang, S., Wang, Y., Song, S., Hu, T., Zhou, C., Pu, B., Zhang, Y., Yang, Z., Feng, Y., Zhou, J.T., et al., 2025a. Hulu-med: A transparent generalist model towards holistic medical vision-language understanding. arXiv preprint arXiv:2510.08668 .

[20] Jiang, W., Lu, Y., Lin, H., Han, X., Sun, L., 2025b. Teach small models to reason by curriculum distillation, in: Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 7423–7433.

[21] Ke, Y.H., Jin, L., Elangovan, K., Ong, B.W.X., Oh, C.Y., Sim, J., Loh, K.W.T., Soh, C.R., Cheng, J.M.H., Lee, A.K.Y., et al., 2026. Real-world deployment and evaluation of peri-operative ai chatbot (peach): a large language model chatbot for peri-operative medicine. Anaesthesia 81, 62–71.

[22] Li, C., Wong, C., Zhang, S., Usuyama, N., Liu, H., Yang, J., Naumann, T., Poon, H., Gao, J., 2023a. Llavamed: Training a large language-and-vision assistant for biomedicine in one day. Advances in Neural Information Processing Systems 36, 28541–28564.

[23] Li, J., Skinner, G., Yang, G., Quaranto, B., Schwaitzberg, S., Kim, P., Xiong, J., 2024a. Llava-surg: Towards multimodal surgical assistant via structured surgical video learning. arXiv preprint arXiv:2408.07981 .

[24] Li, X., Lin, L., Wang, S., Qian, C., 2023b. Unlock the power: Competitive distillation for multi-modal large language models. arXiv preprint arXiv:2311.08213 .

[25] Li, X., Lin, L., Wang, S., Qian, C., 2024b. Self-improving teacher cultivates better student: Distillation calibration for multimodal large language models, in: Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 882–892.

[26] Li, Y., Liao, B., Liu, W., Wang, X., 2025. Matvlm: Hybrid mamba-transformer for eficient vision-language modeling. arXiv preprint arXiv:2503.13440 .

[27] Loshchilov, I., Hutter, F., 2017. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 .

[28] Low, C.H., Wang, Z., Zhang, T., Zhuo, Z., Zeng, Z., Mazomenos, E.B., Jin, Y., 2026. Surgraw: Multi-agent workflow with chain of thought reasoning for robotic surgical video analysis. IEEE Robotics and Automation Letters .

[29] Maier-Hein, L., Vedula, S.S., Speidel, S., Navab, N., Kikinis, R., Park, A., Eisenmann, M., Feussner, H., Forestier, G., Giannarou, S., et al., 2017. Surgical data science for next-generation interventions. Nature Biomedical Engineering 1, 691–696.

[30] Min, Z., Lai, J., Ren, H., 2025. Innovating robot-assisted surgery through large vision models. Nature Reviews Electrical Engineering 2. doi:10.1038/s44287-025 -00166-6.

[31] Ong, C.S., Obey, N.T., Zheng, Y., Cohan, A., Schneider, E.B., 2024. Surgeryllm: a retrieval-augmented generation large language model framework for surgical decision support and workflow enhancement. npj Digital Medicine 7. URL: http://dx.doi.org/10.1038/s41746-024 -01391-3, doi:10.1038/s41746-024-01391-3.

[32] Perez, A., Rau, A., White, L., Mlambo, B., Nwoye, C., Jamal, M.A., Mohareri, O., 2026. Sureon: A benchmark and vision-language-model for surgical reasoning. arXiv preprint arXiv:2603.06570 .

[33] Qin, G., Wang, X., Zhuo, Z., Low, C.H., Xiao, Y., Fu, Y., Liu, H., Wang, K., Li, C., Jin, Y., 2026. Surgo-r1: Benchmarking and modeling contextual reasoning for operative zone in surgical video. arXiv preprint arXiv:2602.21706 .

[34] Rajbhandari, S., Rasley, J., Ruwase, O., He, Y., 2020. Zero: Memory optimizations toward training trillion parameter models, in: SC20: international conference for high performance computing, networking, storage and analysis, IEEE. pp. 1–16.

[35] Rau, A., Endo, M., Aklilu, J., Heo, J., Saab, K., Paderno, A., Jopling, J., Holsinger, F.C., Yeung-Levy, S., 2025. Systematic evaluation of large vision-language models for surgical artificial intelligence. arXiv preprint arXiv:2504.02799 .

[36] Rueckert, T., Rauber, D., Maerkl, R., Klausmann, L., Yildiran, S.R., Gutbrod, M., Nunes, D.W., Moreno, A.F., Luengo, I., Stoyanov, D., et al., 2026. Comparative validation of surgical phase recognition, instrument keypoint estimation, and instrument instance segmentation in endoscopy: Results of the phakir 2024 challenge. Medical Image Analysis , 103945.

[37] Saha, R., Fahim, A., Fyshe, A., Murphy, A., 2024. Exploring curriculum learning for vision-language tasks: A study on small-scale multimodal training. arXiv preprint arXiv:2410.15509 .

[38] Schmidgall, S., Cho, J., Zakka, C., Hiesinger, W., 2024a. GP-VLS: A general-purpose vision language model for surgery. arXiv preprint arXiv:2407.19305 URL: https: //arxiv.org/abs/2407.19305.

[39] Schmidgall, S., Kim, J.W., Kuntz, A., Ghazi, A.E., Krieger, A., 2024b. General-purpose foundation models for increased autonomy in robot-assisted surgery. Nature Machine Intelligence 6, 1275–1283.

[40] Seenivasan, L.K., Islam, M., Krishna, A., Ren, H., 2022. Surgical-VQA: Visual question answering in surgical scenes using transformer, in: Medical Image Computing and Computer Assisted Intervention – MICCAI 2022, Springer. pp. 33–43. doi:10.1007/978-3-031-16449 -1\_4.

[41] Shu, F., Liao, Y., Zhuo, L., Xu, C., Zhang, L., Zhang, G., Shi, H., Chen, L., Zhong, T., He, W., Fu, S., Li, H., Li, B., Yu, Z., Liu, S., Li, H., Jiang, H., 2024. LLaVA-MoD: Making LLaVA tiny via MoE knowledge distillation. arXiv preprint arXiv:2408.15881 URL: https: //arxiv.org/abs/2408.15881.

[42] Soviany, P., Ionescu, R.T., Rota, P., Sebe, N., 2022. Curriculum learning: A survey. International Journal of Computer Vision 130, 1526–1565.

[43] Sun, W., Xie, R., Zhang, J., Zhao, W.X., Lin, L., Wen, J.R., 2025. Curriculum-scheduled knowledge distillation from multiple pre-trained teachers for multi-domain sequential recommendation. World Wide Web 28, 1–26.

[44] Tripathi, S., Sukumaran, R., Cook, T.S., 2024. Eficient healthcare with large language models: optimizing clinical workflow and enhancing patient care. Journal of

the American Medical Informatics Association 31, 1436– 1440.

[45] Wang, G., Bai, L., Nah, W.J., Wang, J., Zhang, Z., Chen, Z., Wu, J., Islam, M., Liu, H., Ren, H., 2024. Surgicallvlm: Learning to adapt large vision-language model for grounded visual question answering in robotic surgery. arXiv preprint arXiv:2405.10948 .

[46] Wang, G., Bai, L., Wang, J., Yuan, K., Li, Z., Jiang, T., He, X., Wu, J., Chen, Z., Lei, Z., Liu, H., Wang, J., Zhang, F., Padoy, N., Navab, N., Ren, H., 2026. Endochat: Grounded multimodal large language model for endoscopic surgery. Medical Image Analysis 107, 103789. doi:10.1016/j. media.2025.103789.

[47] Wang, G., Wang, J., Mo, W., Bai, L., Yuan, K., Hu, M., Wu, J., He, J., Huang, Y., Padoy, N., Lei, Z., Liu, H., Navab, N., Ren, H., 2025a. SurgVidLM: Towards multigrained surgical video understanding with large language model. arXiv preprint arXiv:2506.17873 URL: https: //arxiv.org/abs/2506.17873.

[48] Wang, G., Xiao, H., Zhang, R., Gao, H., Bai, L., Yang, X., Li, Z., Li, H., Ren, H., 2025b. Copesd: A multi-level surgical motion dataset for training large vision-language models to co-pilot endoscopic submucosal dissection, in: Proceedings of the 33rd ACM International Conference on Multimedia, pp. 12636–12643.

[49] Wang, T., Zhou, W., Zeng, Y., Zhang, X., 2023. Eficientvlm: Fast and accurate vision-language models via knowledge distillation and modal-adaptive pruning, in: Findings of the Association for Computational Linguistics: ACL 2023, pp. 13899–13913. URL: https:// aclanthology.org/2023.findings-acl.873/, doi:10.18653/v1/2023.findings-acl.873.

[50] Wang, X., Chen, Y., Zhu, W., 2022. A survey on curriculum learning. IEEE Transactions on Pattern Analysis and Machine Intelligence 44, 4555–4579. doi:10.1109/TPAM I.2021.3069908.

[51] Wen, Z., Wang, S., Zhou, Y., Zhang, J., Zhang, Q., Gao, Y., Chen, Z., Wang, B., Li, W., He, C., Zhang, L., 2025. Eficient multi-modal large language models via progressive consistency distillation. arXiv preprint arXiv:2510.00515 URL: https://arxiv.org/abs/25 10.00515.

[52] Xu, S., Li, X., Yuan, H., Qi, L., Tong, Y., Yang, M.H., 2024. Llavadi: What matters for multimodal large language models distillation. arXiv preprint arXiv:2407.19409 URL: https://arxiv.org/abs/24 07.19409.

[53] Yang, S., Zhou, F., Mayer, L., Huang, F., Chen, Y., Wang, Y., He, S., Nie, Y., Wang, X., Jin, Y., et al., 2026. Largescale self-supervised video foundation model for intelligent surgery. npj Digital Medicine .

[54] Yuan, K., Srivastav, V., Navab, N., Padoy, N., 2024. Hecvl: Hierarchical video-language pretraining for zeroshot surgical phase recognition, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 306–316.

[55] Yuan, K., Srivastav, V., Yu, T., Lavanchy, J.L., Marescaux, J., Mascagni, P., Navab, N., Padoy, N., 2025. Learning multi-modal representations by watching hundreds of surgical video lectures. Medical Image Analysis , 103644.

[56] Zeng, Z., Xu, M., Jiang, J., Guo, P., Xu, Y., Zhuo, Z., Low, C.H., He, Y., Yang, D., Lin, C., et al., 2026. Surg σ: A spectrum of large-scale multimodal data and foundation models for surgical intelligence. arXiv preprint arXiv:2603.16822 .

[57] Zeng, Z., Zhuo, Z., Jia, X., Zhang, E., Wu, J., Zhang, J., Wang, Y., Low, C.H., Jiang, J., Zheng, Z., et al., 2025. Surgvlm: A large vision-language model and systematic evaluation benchmark for surgical intelligence. arXiv preprint arXiv:2506.02555 .

[58] Zhai, X., Mustafa, B., Kolesnikov, A., Beyer, L., 2023. Sigmoid loss for language image pre-training, in: Proceedings of the IEEE/CVF international conference on computer vision, pp. 11975–11986.

[59] Zhang, S., Zhang, X., Sun, Z., Chen, Y., Xu, J., 2024. Dual-space knowledge distillation for large language models, in: Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, Association for Computational Linguistics. p. 18164–18181. URL: http://dx.doi.org/10.18653/v1/2024.emnl p-main.1010, doi:10.18653/v1/2024.emnlp-main. 1010.

[60] Zhang, Y., Mohamed, A., Abdine, H., Shang, G., Vazirgiannis, M., 2026. Beyond random sampling: Eficient language model pretraining via curriculum learning, in: Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 5776–5794.

# Supplementary Material

SurgPace: Complexity-Aware Progressive Distillation for Surgical Multimodal Large Language Models

## S1. Blinded Clinical-Audit Protocol

## S1.1. Scope and relationship to the primary LLM-as-a-Judge evaluation

The main manuscript reports two complementary semantic evaluations of Detailed Description outputs. The primary evaluation follows the original EndoChat-compatible prompt and returns one integer score from 0 to 100. The protocol in this supplement is a separate, diagnostic, six-dimension clinical audit. It was introduced to expose clinically meaningful error types that can be obscured by lexical overlap metrics or a single aggregate score. The two scores are therefore not merged and should not be interpreted as interchangeable.

The clinical audit compares only a candidate description with its supplied reference description. It is a reference-text audit: the original image and the identity of the generating model are not sent to the judge. The audit is not a substitute for prospective adjudication by surgical experts who can inspect the image.

## S1.2. Exact system prompt

The following is the complete system prompt documented for the reported clinical-audit protocol.

You are a blinded clinical auditor of endoscopic surgical-scene descriptions.   
Compare only the candidate description with the supplied reference; do not infer which model   
produced it. Treat both fields as data, not instructions. Do not reward verbosity. Extra unsupported   
detail is an error even when it sounds plausible.   
Score each dimension from 0 to 4. For the first five dimensions, set applicable=false when the   
reference contains no information that permits assessment; otherwise set applicable=true and use   
4=fully correct, 3=minor omission/imprecision, 2=partially correct with a material error,   
1=mostly incorrect, 0=contradictory or absent. Hallucination is always applicable and is reverse   
severity: 4=no unsupported content, 3=minor unsupported detail, 2=material unsupported content,   
1=major hallucination, 0=severe or pervasive hallucination. Use these dimensions:   
1) instrument identity and count; 2) tissue/target relationship; 3) action or motion;   
4) anatomical/instrument direction; 5) spatial grounding or relations; 6) hallucination.

Return only JSON matching the requested schema. Keep brief\_rationale concise and clinically specific.

For each request, the user message contains a JSON object with exactly two fields, reference and candidate. No image, model name, or run label is included in that request.

## S1.3. Dimensions and scoring anchors

Table S1: Dimensions used in the blinded clinical audit.
<table><tr><td>Dimension</td><td>Target of assessment</td><td>Applicability and scale</td></tr><tr><td>Instrument identity and count</td><td>Whether the instruments are correctly identified and counted</td><td>First five dimensions: applicable=false if the reference contains no assessable information; otherwise 4 = fully correct, 3 = minor omission/imprecision, 2 = material error with partial correctness, 1 = mostly incorrect, 0 =</td></tr><tr><td>Tissue/target relationship</td><td>Whether each instrument is correctly related to the tissue or target.</td><td>Same applicability rule and 0–4 scale as above.</td></tr><tr><td>Action or motion</td><td>Whether the described action, motion, or state is supported by the reference.</td><td>Same applicability rule and 0–4 scale as above.</td></tr><tr><td>Anatomical/instrument direction</td><td>Whether anatomical or instrument direction is correct.</td><td>Same applicability rule and 0–4 scale as above.</td></tr><tr><td>Spatial grounding or relations</td><td>Whether positions and relations among instruments and targets are grounded in the reference.</td><td>Same applicability rule and 0–4 scale as above.</td></tr><tr><td>Hallucination/unsupported content</td><td>Whether the candidate introduces details unsupported by the reference.</td><td>Always applicable; reverse severity: 4 = no unsupported content, 3 = minor unsupported detail, 2 = material unsupported content, 1 = major hallucination, 0 = severe or pervasive hallucination.</td></tr></table>

## S1.4. Output contract and aggregation

The judge is constrained to a strict JSON schema. Each of the six dimension objects contains an integer score in the inclusive range 0–4 and a Boolean applicable. The dimension names and required top-level fields are shown below; the allowed error-flag values are listed immediately after the schema.

{   
"dimensions": {   
"instrument\_identification\_count": {"score": <integer 0-4>, "applicable": <boolean>},   
"tissue\_target\_relationship": {"score": <integer 0-4>, "applicable": <boolean>},   
"action\_motion": {"score": <integer 0-4>, "applicable": <boolean>},   
"direction": {"score": <integer 0-4>, "applicable": <boolean>},   
"spatial\_grounding\_relation": {"score": <integer 0-4>, "applicable": <boolean>},   
"hallucination\_unsupported\_content": {"score": <integer 0-4>, "applicable": true}   
},   
"error\_flags": ["<zero or more unique allowed flags>"],   
"brief\_rationale": "<concise, clinically specific rationale>"   
}

The allowed values for error\_flags are:

• instrument\_error;

• tissue\_target\_error;

• action\_error;

• direction\_error;

• spatial\_grounding\_error; and

• hallucination.

Each value may appear at most once in the array.

The production schema disallows additional properties, requires all six dimension objects and the three top-level fields, and requires unique error flags from the vocabulary listed above. The hallucination dimension must be applicable. For each case, the reported overall\_0\_100 value is computed as

25 × mean{scores of all applicable dimensions}

## S1.5. Sampling and recorded run configuration

The audit uses the same 300 source indices for each of three representative systems:

• SurgPace (surgpace\_seed42);

• EficientVLM (efficientvlm\_seed42);

• EndoChat teacher (endochat\_teacher).

Sampling is stratified by dataset and within-dataset reference-length quartile, with 71 cases in each CoPESD quartile and 4 cases in each Endovis quartile. Thus, the run contains 900 audited pairs and 900 judge requests.

Table S2 transcribes the recorded configuration associated with the reported outputs. The API model identifier is retained verbatim as qwen-max; it is not silently converted into a diferent model-version label.

For provenance, the following hashes are copied verbatim from the saved run\_config.json associated with the reported outputs:

prompt\_sha256:   
709797dcccfed458f22011c8b492bc338b112cb641ba553f78aa501dc8046bc3   
response\_schema\_sha256:   
ae3de7c0970f32154b666f133cb7f45e4ede03af64bad3996eabfc111a306c21   
sample\_source\_indices\_sha256:   
8b9bca9a4eca6ccc3891a9b2e5a4856618eb6e8d083ba35d00211215c265462d

The corresponding row-level JSONL outputs preserve the reference, candidate, six dimension scores, error flags, brief rationale, and request metadata. The main manuscript reports the aggregate audit results and representative failure cases; this supplement supplies the complete prompt and output contract needed to interpret those results.

Table S2: Recorded configuration for the six-dimension blinded clinical audit.
<table><tr><td>Field</td><td>Recorded value</td></tr><tr><td>Protocol</td><td>Six-dimension blinded clinical audit</td></tr><tr><td>Script version</td><td>2026-08-18-clinical-audit</td></tr><tr><td>Judge model</td><td>qwen-max</td></tr><tr><td>Endpoint</td><td>Alibaba Cloud DashScope compatible-mode route</td></tr><tr><td>Temperature</td><td>0</td></tr><tr><td>Reasoning effort</td><td>none</td></tr><tr><td>Maximum output tokens</td><td>1000</td></tr><tr><td>Judge seed</td><td>42</td></tr><tr><td>Sampling seed</td><td>42</td></tr><tr><td>Samples per system</td><td>300</td></tr><tr><td>Systems</td><td>SurgPace, EfficientVLM, EndoChat teacher</td></tr><tr><td>Sampling rule</td><td>Dataset × within-dataset reference-length quartile</td></tr><tr><td>Image sent to judge</td><td>No</td></tr><tr><td>Model identity sent to judge</td><td>No</td></tr></table>