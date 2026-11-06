# THG-Omni: Traceable Human-Centric Graph Supervision for Video Reasoning

Jisheng Dang, Kai Wei, Ronghao Lin, Bimei Wang, Hong Peng, Wei-Shi Zheng, Senior Member, IEEE, Bin Hu, Fellow, IEEE, Qi Tian, Fellow, IEEE, Tat-Seng Chua

Abstract—Omni-modal models can answer questions about video, but their predictions rarely show which participant, event, or interval supports an answer. Moreover, answer-level supervision alone does not teach a model to interpret structured video evidence. To systematically mitigate this issue, we present THG-Omni, a human-centric graph-supervision framework that links Segment Anything Model 3 participant tracks with temporally grounded Qwen3-VL annotations. Candidate semantic claims are checked against the source video and available audio before they are used to construct training questions, while track geometry provides explicit spatial and temporal supervision. The resulting questions teach an omni-modal reasoning model to interpret participant attributes, actions, affect, interactions, and their timing from a structured graph. At inference, THG-Omni uses a predicted graph constructed from the evaluation video. Experiments on IntentBench show consistent improvements over the unadapted backbone across social reasoning, emotion understanding, and deception perception. On NExT-QA, THG-Omni achieves 89.54% accuracy with the matched graph, compared with 85.10% without graph input. Further validation experiments show that compact graph input achieves 78.22% accuracy, compared with 78.02% using the full graph and 77.58% without graph input. A complementary diagnostic demonstrates that the predicted graph preserves evidence useful for answering questions even without the source video and audio. Together, these findings show that THG-Omni improves human-centric video reasoning while grounding its evidence in identifiable participants and events. Our codes are available at https://anonymous.4open.science/r/review-n94hhfcwp5666/

Index Terms—Human-centric video reasoning, multimodal large language models, video scene graphs, participant tracking, temporal grounding, graph supervision.

## I. INTRODUCTION

Recent advances in multimodal large language models (MLLMs) have improved temporal localization, long-context modeling, and question answering for complex videos [1]–[3]. By sampling more frames and retaining more visual tokens, these models can process longer videos and capture a wider range of events. However, increased visual coverage does not ensure that relevant evidence is correctly localized, associated with the proper participant, or used during inference [2], [4], [5]. These limitations are particularly important for humancentric video reasoning, where a prediction may depend on brief and distributed evidence from facial expressions, body motion, speech, vocal delivery, and interactions among participants [6], [7].

Human-centric video reasoning requires persistent participant identity and explicit temporal association. Visible attributes, actions, and observable affective states must be assigned to the correct person and restricted to the interval in which they occur. Occlusion, camera motion, viewpoint changes, and shot transitions make such associations difficult [8], [9]. Promptable video segmentation provides a useful basis for addressing this problem. Segment Anything Model 3 (SAM 3) grounds concept prompts and propagates instance masks over time, producing participant trajectories with explicit temporal and spatial information [10]. Video scene graphs further organize trajectories into participant nodes, temporal states, and relations [11]–[14]. Related studies on multimodal knowledge graphs and structured representations also show that explicit entities and relations can support multimodal reasoning [15]–[17].

Nevertheless, constructing a human-centric video graph does not guarantee improved video reasoning. First, tracking or annotation errors can attach a valid observation to the wrong participant or extend it beyond its supported interval. Such errors may remain linguistically plausible and cannot be detected through schema validation alone. Second, a pretrained MLLM may not understand a serialized graph without targeted supervision. Direct graph injection can duplicate visual information, propagate uncertain associations, or receive little attention from a model that has not learned the meaning of the graph fields. Modality priors and visual attention biases may further prevent MLLMs from using supplied evidence [18], [19]. Modular video question answering methods have also explored intermediate evidence and decomposed reasoning [20]. Therefore, graph construction and graph utilization must be treated as separate problems (Fig. 1).

To address these limitations, we propose THG-Omni, a traceable human-centric graph-supervision framework for omni-modal video reasoning. As illustrated in Fig. 2, SAM 3 first grounds and tracks question-relevant participants, providing visibility intervals, trajectories, and spatial relations. Qwen3-VL then generates whole-video semantic claims describing visible attributes, actions, observable affective states, interactions, and intention hypotheses. These geometric and semantic elements are merged into a human-centric video graph in which every claim is associated with a participant and a temporal interval. Unlike methods that only append a graph to the input prompt, our framework converts individual graph fields into explicit supervision examples.

![](images/37b8f08e6f441cbfd4f20fea1384c0fcabd1cf651496995ab50fec1d7c1d181f.jpg)  
Fig. 1. Motivation for traceable graph supervision. This conceptual example contrasts answer-level supervision with identity- and time-grounded graph supervision. The depicted participants and event illustrate the learning objective rather than a benchmark prediction.

Because model-generated semantic claims cannot be regarded as ground truth, we introduce cross-model verification before graph supervision. HumanOmniV2 evaluates each semantic claim using the source video and available audio, while human quality control examines participant identity, temporal extent, and visual support [7], [21]. Only supported positive claims are retained. This protocol avoids synthetic negative statements that may introduce lexical shortcuts or semantically ambiguous counterfactuals. The verified semantic claims are combined with deterministic questions derived from participant trajectories to supervise graph interpretation. Using low-rank adaptation [22], we evaluate graph-specific supervision against the frozen model and compare videomatched, shuffled, and omitted graph input within checkpointaligned runs. These experiments examine information retained in the graph and its use during inference.

Our main contributions are summarized as follows:

• We distinguish the information preserved in a humancentric video graph from a reasoning model’s ability to use that information, providing a clearer basis for evaluating graph-supervised video reasoning.

• We introduce a traceable video representation that combines question-guided SAM 3 participant tracks with temporally localized annotations of attributes, actions, emotions, interactions, and intentions.

• We develop a positive-only annotation protocol in which Qwen3-VL proposes semantic claims, HumanOmniV2 checks their support against video and available audio, and human quality control reviews identity and temporal grounding.

THG-Omni outperforms the evaluated baselines across all three IntentBench tasks, achieving strong results in social reasoning, emotion understanding, and deception perception. Additional graph-input experiments confirm that the proposed representation retains answer-relevant evidence and that graph-specific training helps the model use it.

## II. RELATED WORK

a) Human-centric Video Understanding.: Human-centric video understanding includes social reasoning, emotion recognition, intention inference, and deception-related judgments [23], [24]. Action-intention relationships have also been studied through language-guided localization of unintentional actions [25]. Answers may depend on facial expressions, motion, speech, vocal delivery, and interactions among participants. Because these cues can occur at different times and across modalities, recent models combine visual, acoustic, and textual evidence [6], [26], [27]. Long-video studies further highlight the difficulty of retaining and retrieving relevant evidence across extended contexts [2], [28]–[30]. HumanOmniV2’s IntentBench evaluates such reasoning through Social-IQ, EMER, and MDPE, which have different prediction formats and evidence requirements [7]. These tasks motivate representations that preserve participant identity and the time at which each observation holds.

![](images/5b7d95ed8b4cf0c27cd4b419e37f454a648b937cc37b6f213d8a599c1f7abc57.jpg)  
Fig. 2. Overview of the proposed traceable human-centric graph-supervision framework. (a) Temporally ordered video evidence. (b) SAM3 grounds and tracks participants, while Qwen3-VL annotates visible attributes, emotions, actions, interactions, and their temporal intervals. (c) The grounded evidence forms a video graph with identity-bound nodes, temporal states, and participant relations. (d) HumanOmniV2 verifies each candidate claim using the source video and available audio; only supported claims are retained. (e) The verified graph supports training of THG-Omni, which is evaluated with video-matched, shuffled, and omitted graph input.

b) Video Grounding and Scene Graphs.: Promptable segmentation provides spatial grounding: Segment Anything introduced image-level prompting, SAM 2 propagated masks through video, and SAM 3 added concept-prompted detection and tracking [8], [10], [31]. Persistent identity across shots remains a separate challenge from mask localization [9]. Recent surveys have also summarized temporal grounding methods based on multimodal large language models [32]. Video scene graphs organize entities and changing relations over time [33]. Existing approaches use temporal caption alignment or neurosymbolic learning to obtain such structure [13], [14]. Recent studies further investigate dynamic video scene graph generation and open-vocabulary visual relationship detection [34], [35]. However, a well-formed graph can still assign an action or attribute to the wrong person. Our representation therefore links claims to tracked participants and temporal evidence, and evaluates semantic support separately from structural validity.

c) Structured Reasoning and Annotation Verification.: Multimodal knowledge graphs and structure-aware visionlanguage models expose entities and relations for reasoning [15], [17]. Video-language methods also use localized supervision or question-conditioned evidence acquisition to improve temporal grounding [3], [36]–[39]. Unlike knowledgegraph completion, which predicts missing relations from entity representations [16], our task uses observations grounded in video trajectories and trains an omni-modal model through graph-specific supervision with low-rank adaptation [22], [40]. Visually grounded video question answering examines whether model predictions are supported by relevant video evidence [41]. For semantic fields, Qwen3-VL generates candidate claims and HumanOmniV2 checks their support against the source video and available audio. Prior work on multimodal critics and model-generated feedback motivates this verification step, but model agreement does not guarantee correctness [42], [43]. Human review therefore remains necessary for identity binding, temporal extent, and interaction arguments.

d) Graph Construction and Graph Utilization.: Existing video grounding methods provide spatial and temporal evidence, while video scene graphs organize participants and their relations. However, these two components do not directly address how an omni-modal reasoning model uses the resulting graph. A graph may contain correct participant attributes, actions, and interactions, but the reasoning model may not associate these fields with the question. In addition, a semantic claim may be assigned to the wrong participant or temporal interval even when the graph satisfies its structural constraints.

Prior studies have explored structured video representations for video-language understanding. Finsta [11] improves videolanguage alignment through fine-grained scene graph structures, while CoVGT [12] uses a dynamic graph transformer to capture object relations for video question answering. More recent graph-based methods, including STEP [44] and HyperGLM [45], further explore structured representations for video reasoning. Unlike these approaches, THG-Omni focuses on identity-bound semantic claims, claim verification, and graph-specific supervision for an omni-modal reasoning model. Compositional video question answering has also been studied through question decomposition and video evidence alignment [46]. Video-grounded entailment tree reasoning provides another approach to organizing intermediate evidence for video question answering [47]. These limitations indicate that graph construction and graph utilization require different forms of supervision. Graph construction requires participant identity, temporal grounding, and claim verification. Graph utilization requires the reasoning model to interpret graph fields and associate them with the source video and available audio. THG-Omni addresses these requirements by combining SAM3 person tracks, Qwen3-VL semantic annotations, and graph-specific supervision. The framework also separates the information retained in the predicted graph from the effect of graph input on video reasoning. This distinction is examined through the MER24 diagnostic and the graph-input comparisons on IntentBench and NExT-QA.

## III. METHOD

## A. Overview

Our framework constructs a human-centric graph from video evidence and uses the graph to supervise omni-modal video reasoning. SAM3 grounds and tracks people to obtain temporal and spatial evidence. Qwen3-VL then describes visible attributes, actions, affective states, interactions, and possible intentions using grounded person identifiers. The two sources are assembled into a graph while retaining the provenance of each claim. Human quality control and HumanOmniV2 verification filter semantic claims before training. The accepted claims, together with deterministic questions derived from person tracks, form graph-specific supervision. At evaluation time, the model receives a predicted graph from the evaluation video; these evaluation annotations are not manually corrected or claim-verified. Figure 2 summarizes these stages.

## B. Question-Guided Human Grounding

We first parse the question to identify explicit human references, grammatical number, and the requested semantic category. The resulting prompt set may include person, man, woman, boy, and girl. Answer candidates are not used to construct the grounding prompts. When several questions share the same video and prompt set, the corresponding grounding record is computed once and reused. Given a video V and a question $Q ,$ geometric grounding produces

$$
T = g _ { \mathrm { S A M 3 } } ( V , p ( Q ) ) ,\tag{1}
$$

where $p ( Q )$ denotes the human references extracted from the question, and $T$ contains person tracks and geometric measurements.

SAM3 performs promptable person segmentation and temporal mask propagation [10]. Each video is sampled at up to 20 frames for construction of the geometric graph. Candidate masks are filtered using fixed constraints on mask area, bounding-box area, side length, and aspect ratio. Intersection over-union and containment tests remove redundant detections. Track association uses mask overlap, normalized center displacement, and a fixed maximum temporal gap.

For track i at sampled frame t, the observation is:

$$
o _ { i } ^ { t } = ( t , b _ { i } ^ { t } , c _ { i } ^ { t } , a _ { i } ^ { t } ) ,\tag{2}
$$

where $b _ { i } ^ { t }$ is the bounding box, $c _ { i } ^ { t }$ is its center, and $a _ { i } ^ { t }$ is the mask-area ratio relative to the frame. The trajectory:

$$
{ \mathcal { T } } _ { i } = \{ o _ { i } ^ { t } : t \in F _ { i } \}\tag{3}
$$

records the visible extent and spatial motion of the participant. At this stage, the system does not assign emotions, intentions, or social roles.

## C. Deterministic Spatio-Temporal Graph

The geometric component is represented as:

$$
G _ { T } = ( \nu _ { T } , { \mathcal { E } } _ { T } ) .\tag{4}
$$

Each valid human track defines a node containing its prompt category, temporal extent, visibility coverage, and compact trajectory statistics. Pairwise relations are calculated only when two tracks are visible in a sufficient number of common sampled frames.

For tracks i and $j ,$ their center distance at frame t is normalized by the frame diagonal:

$$
d _ { i j } ^ { t } = \frac { \lVert c _ { i } ^ { t } - c _ { j } ^ { t } \rVert _ { 2 } } { \sqrt { W ^ { 2 } + H ^ { 2 } } } ,\tag{5}
$$

where W and H are the frame width and height. The graph records co-visibility, relative horizontal and vertical position, mask overlap, normalized distance, and distance change. A video-level relation is retained only when its temporal support exceeds a fixed threshold. The supporting frames and source tracks remain attached to each relation, allowing every deterministic graph field to be traced back to the video.

The deterministic graph records geometric evidence without assigning semantic labels. In particular, co-visibility indicates whether two person tracks are observed in the same sampled frames, while normalized distance describes their relative spatial separation. Distance change provides additional information about how the separation varies over time. These fields are computed from person tracks and do not depend on the semantic annotations generated by Qwen3-VL.

Temporal support is important when constructing pairwise relations. Two participants may appear close to each other in a small number of frames without remaining close throughout the video. Similarly, a participant may disappear because of occlusion or a camera transition. The graph therefore retains the sampled frames supporting each relation instead of treating every observed spatial relation as valid for the complete video. This allows the temporal extent of a relation to be checked against the corresponding person tracks. Related work on spatio-temporal video grounding has also investigated supervision from sparsely annotated frames [48].

The geometric fields also have clear limitations. A small distance between two participants does not establish that they are interacting, and a change in distance does not directly indicate an intention or emotional state. Such information requires semantic annotations supported by video evidence. THG-Omni therefore keeps geometric evidence and semantic claims as separate components before graph assembly. This separation allows deterministic graph questions to use track statistics while semantic questions use claims that pass the verification process.

## D. Whole-Video Semantic Annotation and Graph Assembly

Geometric tracks do not identify actions, affect, or interactions. Qwen3-VL-8B therefore annotates the complete video and an automatically generated contact sheet, using grounded SAM3 identifiers as identity anchors. Let $H ( T )$ be the grounded identifier set and $L _ { C }$ a task-specific candidate vocabulary, when one exists. The semantic annotations are generated as

$$
S = g _ { \mathrm { a n n } } \big ( V , \mathrm { c o n t a c t } ( V ) , H ( T ) , L _ { C } \big ) .\tag{6}
$$

The annotator receives neither the reference answer nor a graph corrected using that answer. For MER24, $L _ { C }$ contains the six candidate emotion labels associated with the video; Social-IQ uses free-form descriptions because it has no corresponding closed inventory.

We represent a semantic claim as

$$
s _ { k } = ( u _ { k } , \ell _ { k } , I _ { k } , E _ { k } , \gamma _ { k } ) , I _ { k } = [ r _ { k } ^ { \mathrm { s } } , r _ { k } ^ { \mathrm { e } } ] , 0 \leq r _ { k } ^ { \mathrm { s } } \leq r _ { k } ^ { \mathrm { e } } \leq 1 ,\tag{7}
$$

where $u _ { k }$ is a person identifier or an ordered pair of people for an interaction, $\ell _ { k }$ is the claimed attribute, affect, action, interaction, or intent, $I _ { k }$ is its normalized video interval, $E _ { k }$ contains one to three supporting evidence ratios in $[ 0 , 1 ] ,$ and $\gamma _ { k }$ is a confidence category. The schema also records the people present and annotation limits. Every argument of a claim must name a declared person, and self-interactions are excluded. Appearance claims crossing a camera transition require renewed visual evidence.

An identity map π attaches an annotation identifier to its SAM3 track whenever the track is available. The annotator may introduce a neutral identifier for an additional visible person, but the resulting node remains semantic-only until geometric support exists. Invalid or ambiguous bindings are excluded rather than assigned to a convenient track. The assembled graph is

$$
\begin{array} { r l } { G = g _ { \mathrm { m e r g e } } ( G _ { T } , S ; \pi ) } & { } \\ { \ } & { = ( \mathcal { V } _ { T } \cup \mathcal { V } _ { S } , \mathcal { E } _ { T } \cup \mathcal { E } _ { S } ) , } \end{array}\tag{8}
$$

where geometric fields in $( \nu _ { T } , \mathcal { E } _ { T } )$ retain their track and frame provenance, while semantic fields in $( \nu _ { S } , \mathcal { E } _ { S } )$ retain their claim, interval, evidence ratios, and confidence. A semanticonly person has no deterministic geometric fields. Wholevideo annotation is used in the formal pipeline; a fixedwindow pilot produced fragmented identities when participants overlapped.

The identity map π connects the person identifiers used by Qwen3-VL with the person tracks obtained from SAM3. This step is necessary because geometric grounding and semantic annotation provide different information about the same video. SAM3 identifies visible participants and their spatial positions, while Qwen3-VL describes participant attributes, actions, affective states, interactions, and possible intentions. Without identity binding, a semantic claim may describe the correct event but refer to the wrong participant.

Identity binding is particularly important when several participants appear in the same video. A semantic claim describing an interaction must identify the participants involved, and its temporal interval must correspond to the period in which the interaction is supported. When the available person tracks do not provide a valid identity binding, the annotation is not attached to an unrelated track. A semantic-only person may be retained, but deterministic geometric fields are not assigned without geometric support.

The assembled graph preserves the source information of both components. Geometric fields retain their person tracks and supporting frames, while semantic claims retain their person identifiers, temporal intervals, evidence ratios, and confidence categories. These fields allow a graph question to refer to a specific participant or event and retain the evidence associated with that question. The resulting human-centric video graph is therefore used not only as additional input to the reasoning model, but also as a source of graph-specific supervision.

## E. Training-Only Quality Control and Claim Verification

Human inspection of development outputs checks person binding, temporal extent, label validity, and visual support before large-scale construction. Schema normalization is conservative: it may standardize a supported format, set an inapplicable optional label to null, or retain existing evidence points from an overlong list. It does not create claims. Malformed person bindings, invalid intervals, non-human nodes, and unsupported interaction arguments are excluded; changes and exclusions are logged. Appearance changes across camera transitions receive interval-level review.

HumanOmniV2 independently checks each remaining semantic claim against the source video and available audio [7]. It sees neither the benchmark answer nor a synthetic negation. For a candidate $s _ { k }$ , define

$$
\begin{array} { r l } & { z _ { k } = \mathrm { V e r i f y } _ { \mathrm { H u m a n O m n i V 2 } } ( s _ { k } , V , A ) , } \\ & { z _ { k } \in \{ \mathrm { s u p p o r t e d } , \mathrm { u n s u p p o r t e d } , \mathrm { u n c e r t a i n j } . } \end{array}\tag{9}
$$

The retained semantic collection is

$$
\begin{array} { r l } & { S ^ { + } = \bigl \{ s _ { k } \in S : \mathrm { s c h e m a \_ v a l i d } ( s _ { k } ) } \\ & { \qquad \land \mathrm { i d e n t i t y \_ v a l i d } ( s _ { k } ) \land z _ { k } = \mathrm { s u p p o r t e d } \bigl \} . } \end{array}\tag{10}
$$

Manually rejected, unsupported, uncertain, and malformed claims are excluded. When a multi-claim response cannot be parsed, only its affected verification unit is retried; original outputs and parser repairs are preserved. This positive-only protocol avoids the lexical shortcuts and equivalent counterfactuals observed in an earlier original-versus-negated discrimination setting. Both human review and claim verification apply to training and development construction, not evaluation graphs.

Schema validation, identity checks, and claim verification address different sources of errors. Schema validation determines whether a semantic claim follows the required format, including valid person identifiers, temporal intervals, and interaction arguments. However, a schema-valid claim may still describe an event that is not supported by the video. Identity checks further examine whether the claim refers to a declared participant and whether the available person tracks support the identity binding. These checks reduce invalid graph construction but do not establish the correctness of every semantic claim.

Claim verification therefore provides an additional step. HumanOmniV2 evaluates each candidate semantic claim using the source video and available audio and assigns a supported, unsupported, or uncertain verdict. Only supported claims that also satisfy schema and identity requirements are retained for semantic question construction. Unsupported and uncertain claims are excluded rather than converted into negative training examples. This is consistent with the positive-only annotation protocol used in THG-Omni.

The verification results should be interpreted as modelbased assessments rather than human-confirmed labels. A supported verdict indicates that the verifier found evidence for the candidate claim, but it does not guarantee that the claim is correct. Human quality control is therefore used to inspect identity binding, temporal extent, and visual support during training-data construction. These checks are not applied as manual corrections to evaluation graphs. During inference, the reasoning model receives the predicted graph constructed from the evaluation video without verifier verdicts.

## F. Graph-Specific Supervision

The final training collection combines deterministic graph questions with verified semantic questions. Deterministic targets are computed from frozen track statistics and cover temporal duration, first appearance, relative presence, co-visibility, overlap, distance change, and directional relations. Semantic targets are derived only from accepted positive claims and cover visible attributes, actions, emotions, interactions, and intentions.

For video $V ,$ optional audio $A ,$ question $Q ,$ answer candidates C, and graph G, the downstream prediction is

$$
\hat { Y } = f _ { \theta , \phi } ( V , A , Q , C , G ) ,\tag{11}
$$

where θ denotes the frozen omni-modal backbone and $\phi$ denotes the trainable LoRA parameters.

Training first fits an answer-supervised LoRA adapter (Task-LoRA), merges it into the frozen backbone, and then performs one epoch of joint adaptation. The joint collection includes answer questions with and without graph input, verified graph questions, and matched or mismatched graph identification.

With the training examples denoted by $\mathcal { D } _ { \mathrm { j o i n t } }$ , the joint-stage objective is

$$
\mathcal { L } _ { \mathrm { j o i n t } } = - \sum _ { ( V , A , G , Q , Y ) \in \mathcal { D } _ { \mathrm { j o i n t } } } \log P _ { \theta , \phi } \big ( Y \mid V , A , G , Q \big ) ,\tag{12}
$$

where $G$ may be omitted for graph-dropout examples. For multiple-choice questions, the candidate set C is included in Q in the training objective.

LoRA modules are attached to the language-model attention projections [22]. The visual tower, audio tower, and base language-model parameters remain frozen. Human review and cross-model verification are restricted to training and development data construction. During evaluation, the model receives only a graph predicted from the evaluation video. No evaluation annotation is manually corrected, and no verifier verdict is used as an inference input.

The two training stages have different purposes. Task-LoRA first adapts Qwen2.5-Omni-7B to the answer format of the downstream tasks. The resulting adapter is merged into the backbone before joint training. This provides a task-adapted checkpoint for the subsequent graph-specific supervision. The joint stage then trains the model using task questions and graph questions rather than replacing the original task supervision with graph questions alone.

The deterministic graph questions and verified semantic questions also provide different supervision. Deterministic questions are constructed from person tracks and geometric relations. Their answers are computed from the graph fields, including temporal duration, visibility, co-visibility, and distance change. These questions teach the model to interpret geometric information that can be directly calculated from the person tracks. Verified semantic questions are constructed from accepted semantic claims and cover visible attributes, actions, emotions, interactions, and intentions. These questions teach the model to associate semantic graph fields with the corresponding participants and temporal intervals.

The joint collection further includes task questions with graph dropout and task questions with matched graphs. Graphdropout examples require the model to answer using the source video and available audio without graph input. Matched-graph examples provide the predicted graph as additional input for the same type of video reasoning task. Including both conditions allows the model to encounter questions with and without graph input during training. This is important because a predicted graph may not contain all the information required to answer a question.

Matched or mismatched graph-identification examples provide supervision about graph–video correspondence. A matched graph is constructed from the corresponding video, while a mismatched graph contains information from another video. These examples require the model to distinguish graph input associated with the source video from graph input that does not correspond to it. The joint training objective combines these supervision types using the same conditional languagemodeling loss. The different training examples therefore contribute to graph interpretation, task question answering, and graph–video correspondence within one training stage.

Algorithm 1: Traceable Human-Centric Graph Supervision   
Require: Training examples $\overline { { \{ ( V , A , Q , Y ) \} } }$ , frozen backbone $f _ { \theta } ,$ and   
candidate vocabulary $L _ { C }$ when available.   
Ensure: Task-adapted backbone and joint-stage LoRA parameters $\phi .$   
1: Initialize $\mathcal { D } _ { \mathrm { g r a p h } }  \emptyset .$   
2: for each training video V and grounding question $Q$ do   
3: Obtain tracks $\mathsf { \bar { \Gamma } } T \gets g _ { \mathrm { S A M 3 } } ( \mathsf { \bar { V } } , p ( Q ) )$   
4: Construct $G _ { T } ~ = ~ ( \nu _ { T } , \mathcal { E } _ { T } )$ from track statistics and spatial   
relations.   
5: Generate semantic claims:   
$S  g _ { \mathrm { a n n } } ( V ,$ contact $( V ) , H ( T ) , L _ { C } ) .$   
6: Review identity binding, temporal extent, label validity, and visual   
support.   
7: for each remaining claim $s _ { k } \in S$ do   
8: Compute verifier result:   
$z _ { k } \gets \mathrm { V e r i f y } _ { \mathrm { H u m a n O m n i V 2 } } ( s _ { k } , V , A ) .$   
9: end for   
10: Retain supported claims $S ^ { + }$ according to (10).   
11: Assemble $\mathbf { \bar { \cal G } } \gets g _ { \mathrm { m e r g e } } ( G _ { T } , S ^ { + } ; \pi ) .$   
12: Add deterministic questions from $G _ { T }$ and verified semantic   
questions from $S ^ { + }$ to $\mathcal { D } _ { \mathrm { g r a p h } } .$   
13: end for   
14: Train a task question-answering LoRA adapter and merge it into   
the backbone.   
15: Construct $\mathcal { D } _ { \mathrm { j o i n t } }$ from task questions with graph dropout or matched   
graphs, verified graph questions, and matched or mismatched graph   
identification examples.   
16: Optimize $\phi$ for one epoch with θ frozen using the joint objective   
in (12).   
17: return the task-adapted backbone and joint-stage adapter $\phi .$

For graph-dropout examples, G is omitted. Answer candidates C are included in Q. Human review and claim verification are restricted to training and development construction.

## IV. EXPERIMENTS

## A. Experimental Setup

We evaluate THG-Omni on human-centric video reasoning tasks. MER24 contains 150 multi-label emotion samples and is used to test the information retained in the scene graph. IntentBench covers social reasoning, emotion understanding, and deception perception. Its original manifest contains 2,689 questions from 633 videos. One MDPE video is excluded by a stable task identifier because it repeatedly exhausts GPU memory even when processed alone on an NVIDIA L20. The resulting evaluation set contains 2,688 aligned question– graph pairs from 632 videos: 2,356 Social-IQ questions, 133 EMER questions, and 199 MDPE questions. The exclusion is fixed before prediction and scoring. NExT-QA [49] is additionally used to evaluate video question answering across causal, temporal, and descriptive reasoning tasks. The test set contains 8,564 questions from 1,000 videos, and the validation set contains 4,996 questions from 570 videos. Each question has five answer options. Unlike IntentBench, which includes both single-label and multi-label tasks, NExT-QA is evaluated using multiple-choice accuracy.

Graph-specific supervision is constructed from person tracks and accepted semantic claims. The graph-question collection contains 7,339 training examples, comprising 1,555 deterministic temporal and spatial questions and 5,784 verified positive semantic questions; its development collection contains 1,146 examples. The final joint-training stage presents 2,887 examples in one epoch, combining answer questions, graph questions, and matched or mismatched graph examples. The graph-question training and development partitions do not share videos. Qwen2.5-Omni-7B serves as the reasoning backbone. SAM3 provides person tracks and geometric relations, Qwen3-VL-8B generates whole-video semantic annotations, and HumanOmniV2 assesses the support for candidate semantic claims. In the video reasoning pipeline, frames are sampled at 0.25 FPS, generation is deterministic, and available audio is provided to the reasoning model. The annotation model receives video frames and a contact sheet without audio. The verifier may use audio when assessing a claim. LoRA modules are applied to the language attention projections; the visual tower, audio tower, and base language-model weights remain frozen. Additional training and checkpoint details are given in Section IV-F.

Social reasoning and deception perception are measured by accuracy. Emotion understanding is measured by samplelevel multi-label F1. F1 is computed from each question’s predicted and reference label sets and then averaged over questions. Exact set match requires equality of the two sets. For MER24, Table V reports this sample-averaged F1 as macro F1; micro F1 instead pools true positives, false positives, and false negatives across all samples before computing F1. Its set precision and recall are averaged separately over samples. The overall IntentBench score is a sample-weighted aggregation across the 2,688 questions. Every condition in an IntentBench comparison uses the same questions and evaluation videos.

## B. Main Experiments

Table I reports IntentBench results on the same 2,688 questions. The matched THG-Omni achieves 71.18% Social-IQ accuracy and 68.34% MDPE accuracy, compared with 53.99% and 61.81% for Qwen2.5-Omni-7B. EMER sample F1 is 86.26% versus 68.09% for Qwen2.5-Omni-7B, while EMER exact match is 66.17% versus 41.35%. Our method has a sample-weighted overall score of 71.72%. Figure 3 visualizes the baseline and THG-Omni scores alongside the separate MER24 graph-information diagnostic. Graph-SFT denotes a separate LoRA adapter trained for graph-field question answering and evaluated with either no graph or a compact graph; these two rows share weights. THG-Omni denotes the staged, joint-training model described in Section III. The THG-Omni rows compare graph-input conditions of the same trained model, whereas Graph-SFT uses a separate adapter.

LLaVA-NeXT-Video-7B-32K provides an additional videoonly baseline on the same 2,688 questions. It receives eight uniformly sampled frames and the question with its answer options, without audio or graph input, and obtains 31.58% Social-IQ accuracy, 29.68% EMER sample F1, and 39.20% MDPE accuracy. These scores use less input evidence than the audio-video runs and should be interpreted with that modality difference in mind. Responses with a recoverable answer option are scored, while unparseable responses are counted as incorrect.

HumanOmniV2 [7] is also evaluated without graph input on the same 2,688 questions. It obtains 51.57% Social-IQ accuracy, 80.03% EMER sample F1, and 53.77% MDPE accuracy. THG-Omni improves EMER sample F1 to 86.26% on the same questions.

![](images/685f38ee4638f55865d0d8af72fe188f76a59fa70f170b805250ad38f9e36d44.jpg)  
Fig. 3. IntentBench results and MER24 graph diagnostic. (a) Qwen2.5-Omni-7B and video-matched THG-Omni are compared on 2,688 questions. (b) On 150 MER24 samples without source video or audio, graph input improves exact set match over text-only input.

TABLE I  
INTENTBENCH RESULTS ON 2,688 QUESTIONS. OVERALL IS THE SAMPLE-WEIGHTED AGGREGATE OF SOCIAL-IQ ACCURACY, EMER SAMPLE F1, AND MDPE ACCURACY. LLAVA-NEXT-VIDEO USES VIDEO WITHOUT AUDIO. GRAPH-SFT USES A DISTINCT ADAPTER; THE THG-OMNI ROWS REPORT DIFFERENT GRAPH-INPUT CONDITIONS FOR THE SAME TRAINED MODEL.
<table><tr><td></td><td>Social-IQ</td><td colspan="2">EMER</td><td>MDPE</td><td>Overall</td></tr><tr><td>Model / input</td><td>Accuracy</td><td>Sample F1</td><td>Exact</td><td>Accuracy</td><td>Score</td></tr><tr><td>LLaVA-NeXT-Video [50]</td><td>31.58</td><td>29.68</td><td>12.03</td><td>39.20</td><td>32.05</td></tr><tr><td>HumanOmniV2 [7]</td><td>51.57</td><td>80.03</td><td>54.89</td><td>53.77</td><td>53.14</td></tr><tr><td>Qwen2.5-Omni-7B [51]</td><td>53.99</td><td>68.09</td><td>41.35</td><td>61.81</td><td>55.27</td></tr><tr><td>Graph-SFT (w/o graph)</td><td>49.66</td><td>70.52</td><td>40.60</td><td>60.30</td><td>51.48</td></tr><tr><td>Graph-SFT (compact graph)</td><td>56.66</td><td>71.32</td><td>46.62</td><td>61.31</td><td>57.73</td></tr><tr><td>THG-Omni (w/o graph)</td><td>67.28</td><td>84.63</td><td>66.17</td><td>62.31</td><td>67.77</td></tr><tr><td>THG-Omni (shuffled graph)</td><td>67.11</td><td>81.29</td><td>61.65</td><td>61.31</td><td>67.38</td></tr><tr><td>THG-Omni (matched graph; Ours)</td><td>71.18</td><td>86.26</td><td>66.17</td><td>68.34</td><td>71.72</td></tr></table>

TABLE II  
SOCIAL-IQ ACCURACY (%) ACROSS SIX QUESTION TYPES ON 2,356 QUESTIONS.
<table><tr><td>Method</td><td>Why (698)</td><td>How (632)</td><td>What (480)</td><td>When (14)</td><td>Who/Which (25)</td><td>Other (507)</td><td>Overall</td></tr><tr><td>Qwen2.5-Omni-7B [51]</td><td>53.44</td><td>54.11</td><td>53.54</td><td>57.14</td><td>64.00</td><td>54.44</td><td>53.99</td></tr><tr><td>Graph-SFT (w/o graph)</td><td>47.99</td><td>49.05</td><td>50.00</td><td>50.00</td><td>56.00</td><td>52.07</td><td>49.66</td></tr><tr><td>Graph-SFT (compact graph)</td><td>55.01</td><td>56.01</td><td>57.08</td><td>57.14</td><td>64.00</td><td>58.97</td><td>56.66</td></tr><tr><td>THG-Omni (w/o graph)</td><td>66.19</td><td>65.66</td><td>66.88</td><td>64.29</td><td>72.00</td><td>71.01</td><td>67.28</td></tr><tr><td>THG-Omni (shuffled graph)</td><td>66.48</td><td>65.19</td><td>65.83</td><td>64.29</td><td>72.00</td><td>71.40</td><td>67.11</td></tr><tr><td>THG-Omni (matched graph)</td><td>69.63</td><td>69.78</td><td>70.83</td><td>71.43</td><td>80.00</td><td>74.95</td><td>71.18</td></tr></table>

Social-IQ question-type analysis. Social-IQ contains six question groups: Why, How, What, When, Who/Which, and Other. Table II reports accuracy across these question types.

EMER multi-label metrics. Table III reports precision, recall, sample-level F1, and exact-match performance.

Graph-input sensitivity on IntentBench. Rather than repeating the absolute scores in Table I, Table IV reports the differences between specified conditions. For the fixed Graph-SFT adapter, adding its compact graph improves Social-IQ accuracy by 7.00 percentage points and EMER exact match by 6.02 points. For THG-Omni, the reported matched-graph condition exceeds the no-graph condition by 3.90 points on Social-IQ and 6.03 points on MDPE. The matched and shuffled THG-Omni results were reported from different evaluation runs; these differences should not be interpreted as paired statistical estimates. EMER differences refer to sample-level F1, not classification accuracy.

TABLE III  
EMER MULTI-LABEL PERFORMANCE ON 133 EXAMPLES (%).
<table><tr><td>Method</td><td>Precision</td><td>Recall</td><td>Sample F1</td><td>Exact</td></tr><tr><td>Qwen2.5-Omni-7B [51]</td><td>74.00</td><td>68.53</td><td>68.09</td><td>41.35</td></tr><tr><td>Graph-SFT (w/o graph)</td><td>76.20</td><td>70.10</td><td>70.52</td><td>40.60</td></tr><tr><td>Graph-SFT (compact graph)</td><td>78.10</td><td>70.80</td><td>71.32</td><td>46.62</td></tr><tr><td>THG-Omni (w/o graph)</td><td>89.54</td><td>83.97</td><td>84.63</td><td>66.17</td></tr><tr><td>THG-Omni (shuffled graph)</td><td>87.97</td><td>79.52</td><td>81.29</td><td>61.65</td></tr><tr><td>THG-Omni (matched graph)</td><td>93.36</td><td>83.91</td><td>86.26</td><td>66.17</td></tr></table>

![](images/383942eeebe161ef56fedff38d2ead200ab350463e6e902e18895356260d4b28.jpg)  
Fig. 4. Illustrative Social-IQ video-to-graph example. The video frames show two participants and their interaction during a card game. SAM3 provides perso tracks and geometric evidence, while semantic annotations describe the observed interaction. The example illustrates participant grounding and temporall localized visual evidence associated with the Social-IQ question. The reference answer is B.

![](images/6f80c049c17234d2e40f31327add6156b415f1c0aa41222e1d127845c94c0134.jpg)  
Fig. 5. Illustrative Social-IQ video-to-graph example. The video frames show the visible participants and the woman’s later reaction. The example illustrates participant grounding and temporally localized visual evidence associated with the Social-IQ question. The reference answer is C. This example is used to illustrate evidence organization rather than a performance improvement.

Table V reports a MER24 graph-information diagnostic on 150 samples. Both conditions receive the same question and answer options, without source video or audio; only the graph input differs. Adding the predicted graph increases exact match from 14.67% to 24.00% and macro F1 from 42.18% to 57.18%. There are 21 rescued and 7 harmed examples (exact McNemar p = 0.0125), and the bootstrap 95% interval for the exact-match difference is [2.67%, 16.00%]. This diagnostic tests information recoverable from the graph, rather than improvement to video-conditioned reasoning.

## C. NExT-QA Evaluation

We further evaluate video question answering on NExT-QA. The test set contains 8,564 questions from 1,000 videos, and the validation set contains 4,996 questions from 570 videos. SAM3 provides geometric evidence, while Qwen3-VL-8B generates whole-video semantic annotations. One video in each split has an invalid semantic annotation, for which the graph retains only geometric evidence. The resulting predicted graphs cover all 8,564 test questions and 4,996 validation questions. Evaluation graph construction does not use reference answers, question text, answer options, or historical predictions to generate semantic claims.

For the NExT-QA test set, the predicted graphs contain 19,304 visible attribute claims, 17,560 action segments, 6,781 interactions, 10,374 emotion segments, and 9,821 intent hypotheses, with an average graph length of 2,235.68 characters. On the validation set, the corresponding numbers are 11,565, 10,385, 4,188, 5,886, and 5,618, with an average graph length of 2,280.67 characters. These predicted graphs combine SAM3 geometric evidence with Qwen3-VL semantic annotations and are used as additional input to the reasoning model.

Table VI reports NExT-QA test accuracy across causal, temporal, and descriptive questions. Qwen2.5-Omni-7B achieves

![](images/bfd179cdd735cf0361db89836e8f9e4517568e5e84d5102475266a6a01c2c4d9.jpg)  
Fig. 6. Illustrative Social-IQ video-to-graph example. The video frames show participant grounding and temporally localized visual evidence relevant to th question. Participant identities are manually matched across frames for illustration. The reference answer is D, while the recorded model answer is unavailable. This example illustrates evidence organization rather than a verified model prediction

TABLE IV  
INTENTBENCH GRAPH-INPUT SENSITIVITY, EXPRESSED AS DIFFERENCES IN PERCENTAGE POINTS BETWEEN REPORTED CONDITIONS. EMER USES SAMPLE-LEVEL F1 AND EXACT SET MATCH; OTHER TASK COLUMNS USE ACCURACY.
<table><tr><td>Comparison (second minus first)</td><td>Social-IQ</td><td>EMER F1</td><td>EMER Exact</td><td>MDPE</td><td>Overall</td></tr><tr><td>Graph-SFT: compact — no graph</td><td>+7.00</td><td>+0.80</td><td>+6.02</td><td>+1.01</td><td>+6.25</td></tr><tr><td>THG-Omni: matched — no graph</td><td>+3.90</td><td>+1.63</td><td>+0.00</td><td>+6.03</td><td>+3.95</td></tr><tr><td>THG-Omni: matched — shuffled</td><td>+4.07</td><td>+4.97</td><td>+4.52</td><td>+7.03</td><td>+4.34</td></tr></table>

TABLE V

MER24 GRAPH-INFORMATION DIAGNOSTIC ON 150 SAMPLES. BOTH CONDITIONS EXCLUDE SOURCE VIDEO AND AUDIO; THE ONLY INPUT DIFFERENCE IS THE PREDICTED SCENE GRAPH. ALL VALUES ARE PERCENTAGES. SET PRECISION AND RECALL ARE SAMPLE-LEVEL AVERAGES.
<table><tr><td>Input</td><td>Exact</td><td>Set precision</td><td>Set recall</td><td>Macro F1</td><td>Micro F1</td></tr><tr><td>Question and options</td><td>14.67</td><td>37.90</td><td>56.67</td><td>42.18</td><td>44.23</td></tr><tr><td>Question, options, and graph</td><td>24.00</td><td>54.40</td><td>69.33</td><td>57.18</td><td>56.16</td></tr></table>

77.98% accuracy, while Task-LoRA without graph input achieves 76.09%. Graph-SFT improves from 77.86% without graph input to 82.40% with matched graph input. THG-Omni achieves 85.10% without graph input and 89.54% with matched graph input. The matched graph improves THG-Omni accuracy by 4.44 percentage points. Compared with Qwen2.5- Omni-7B, THG-Omni with matched graph input improves accuracy by 11.56 percentage points.

These results show that graph-specific supervision and matched graph input contribute to NExT-QA performance. THG-Omni without graph input also outperforms Graph-SFT without graph input, indicating a difference between the two training conditions.

We further examine graph content on the NExT-QA validation set using the same Graph-SFT adapter. Accuracy increases from 77.58% without graph input to 78.02% with the full graph and 78.22% with the compact graph. The compact graph retains visible attributes, action segments, and interactions, while removing emotion segments and intent hypotheses. Compared with the full graph, the compact graph corrects 66 previously incorrect predictions and changes 56 correct predictions into incorrect ones. Compared with no graph input, it corrects 187 predictions and changes 155 correct predictions into incorrect ones. These results indicate that graph content affects answer prediction, and that removing emotion and intent claims leads to a small improvement on the validation set.

We further examine graph input on the NExT-QA test set. With the same Graph-SFT adapter, adding the matched graph increases accuracy from 77.86% to 82.40%, an improvement of 4.54 percentage points. The THG-Omni joint-stage model achieves 85.10% without a graph and 89.54% with the matched graph, an improvement of 4.44 percentage points. These results show that the predicted graph provides additional information for video question answering. Compared with Qwen2.5-Omni-7B, THG-Omni with the matched graph achieves an improvement of 11.56 percentage points. Fig. 7 illustrates a temporal reasoning example from the NExT-QA validation set. The question asks what the man does after the lady finishes talking. The Graph-SFT adapter predicts C without graph input and A with the matched graph, which agrees with the reference answer. The example shows how predicted graph evidence can help the model identify the relevant temporal event.

![](images/7e932ca04f84cbfad55eb1706f4a06c018d11ddd2c7f215b99331fa463feb326.jpg)  
Fig. 7. NExT-QA temporal reasoning example from the validation set. Three video frames show the lady talking and the man’s subsequent reaction. SAM3 provides participant grounding, while the predicted graph organizes the temporal evidence used for answering the question. With the same Graph-SFT adapter the no-graph prediction is C, whereas the matched-graph prediction is A, consistent with the reference answer.

TABLE VI  
NEXT-QA TEST ACCURACY (%) ACROSS REASONING TYPES ON 8,564 QUESTIONS.
<table><tr><td rowspan="2">Model / input</td><td colspan="2">Causal</td><td colspan="3">Temporal</td><td colspan="3">Descriptive</td><td rowspan="2">Overall</td></tr><tr><td>CH</td><td>CW</td><td>TC</td><td>TN</td><td>TP</td><td>DC</td><td>DL</td><td>DO</td></tr><tr><td>Qwen2.5-Omni-7B [51]</td><td>76.21</td><td>79.18</td><td>77.51</td><td>72.34</td><td>75.27</td><td>61.18</td><td>93.79</td><td>85.50</td><td>77.98</td></tr><tr><td>Task-LoRA (no graph)</td><td>73.74</td><td>76.99</td><td>75.28</td><td>70.19</td><td>73.12</td><td>60.87</td><td>94.00</td><td>85.17</td><td>76.09</td></tr><tr><td>Graph-SFT (no graph)</td><td>76.39</td><td>79.18</td><td>77.25</td><td>71.77</td><td>74.19</td><td>61.18</td><td>93.58</td><td>85.67</td><td>77.86</td></tr><tr><td>Graph-SFT (matched graph)</td><td>81.66</td><td>83.21</td><td>82.59</td><td>76.65</td><td>80.72</td><td>70.39</td><td>95.11</td><td>88.86</td><td>82.40</td></tr><tr><td>THG-Omni (no graph)</td><td>83.79</td><td>85.56</td><td>84.80</td><td>81.20</td><td>83.91</td><td>75.99</td><td>96.13</td><td>90.96</td><td>85.10</td></tr><tr><td>THG-Omni (matched graph)</td><td>88.56</td><td>89.93</td><td>89.13</td><td>86.90</td><td>88.02</td><td>82.97</td><td>97.41</td><td>93.68</td><td>89.54</td></tr></table>

CH/CW: causal how/why; TC/TN/TP: temporal counting/next/previous; DC/DL/DO: descriptive counting/location/other. Category counts: causal 4,502; temporal 2,657; descriptive 1,405.

## D. Graph Evidence and Input Analysis

A separate, fixed Graph-SFT adapter tests whether graphspecific training helps the model use a compact graph. With the same adapter, adding graph input raises the overall Intent-Bench score from 51.4805% to 57.7330%, a gain of 6.2525%. Social-IQ accuracy rises from 49.6604% to 56.6638%, and

EMER exact match from 40.6015% to 46.6165%. Its nograph score is below the unadapted backbone’s 55.27%; the within-adapter benefit is therefore specific to the compactgraph condition, rather than an ablation of the final THG-Omni model. Figure 4 illustrates how one Social-IQ answer can be traced to grounded evidence. The earlier baseline also chose the reference answer in this case, so the example does not represent a corrected baseline error.

The THG-Omni rows in Table I further examine the role of graph input. The matched-graph run scores 71.72% overall, compared with 67.77% without graph input and 67.38% with a shuffled graph. On Social-IQ, the corresponding accuracies are 71.18%, 67.28%, and 67.11%; on MDPE, they are 68.34%, 62.31%, and 61.31%. EMER sample F1 is 86.26% with the matched graph, versus 84.63% without a graph and 81.29% with a shuffled graph. These observations are consistent with the model benefiting from video-matched structured evidence, particularly for social reasoning and emotion understanding. The no-graph and shuffled-graph controls were evaluated in an earlier run, however, so the differences from the updated matched-graph row should be read as comparisons of reported conditions rather than paired estimates from one evaluation run.

The three input conditions also address different questions. Removing the graph tests whether the trained model can answer from the video and available audio alone. Shuffling supplies graph text whose participant and event descriptions no longer correspond to the evaluated video; it therefore probes the importance of graph–video correspondence rather than simply the presence of additional tokens. The lower EMER score with a shuffled graph is consistent with mismatched evidence being unhelpful for emotion questions, where the relevant person and interval matter. At the same time, the no-graph condition remains substantially stronger than the unadapted backbone on Social-IQ and EMER. Part of THG-Omni’s performance is therefore retained without graph input, while THG-Omni with updated matched-graph run gives the highest reported overall score. The MER24 experiment tests a separate property: when source video and audio are withheld, the predicted graph raises exact match from 14.67% to 24.00%. It shows that the representation contains answerrelevant information, but does not by itself measure its contribution in video-conditioned inference. Together, the controlled Graph-SFT comparison, THG-Omni input conditions, and MER24 diagnostic provide distinct evidence about learning to use a graph, preserving video correspondence, and retaining information in the graph.

The NExT-QA experiments further examine graph input on a different video question-answering dataset. On the test set, Graph-SFT achieves 82.40% with the matched graph, compared with 77.86% without graph input. THG-Omni further improves accuracy from 85.10% to 89.54% with the matched graph. These results show that graph input contributes to both graph-specific training and joint-stage adaptation. Together with the IntentBench and MER24 experiments, the NExT-QA results provide additional evidence that predicted humancentric graphs can support video reasoning.

## E. Annotation and Claim Verification

Whole-video Qwen3-VL annotation was piloted on two development videos and then inspected on ten. All ten records passed schema validation. Manual review checked participant identity, temporal extent, label validity, and support in the video. In one case, an appearance attribute persisted across a camera transition even though the person’s clothing changed; the affected interval was flagged for review. A subsequent check of 66 development videos produced 66 schema-valid records.

On the development set, HumanOmniV2 assessed 632 candidate claims from 395 valid verification units. It classified 591 as supported, 26 as unsupported, and 15 as uncertain. Only supported claims that also pass schema and participantidentity checks are eligible for semantic question construction.

These verdicts are model-based assessments using the source video and available audio; the supported count is not a humanverified accuracy measure. The 632 development claims are distinct from the 632 videos in the IntentBench evaluation set.

The evaluation-video annotation run covers 632 videos. After invalid outputs are retried, all 632 annotation records pass schema validation. Schema validity checks the output format, not the truth of every semantic claim. Evaluation graphs are generated automatically: they receive no manual semantic corrections, and verifier decisions are not provided to the reasoning model at inference.

## F. Graph Questions and Adapter Training

The graph-question collection contains 7,339 training examples: 1,555 deterministic temporal or geometric questions and 5,784 questions constructed from accepted semantic evidence. Its development collection contains 1,146 examples and has no video overlap with the graph-question training collection. Deterministic answers are computed from trackedperson statistics; semantic questions cover attributes, actions, affect, interactions, and intentions only when their source claims pass verification.

The final reasoning model is trained in two stages. First, a task question-answering LoRA adapter is trained and merged into Qwen2.5-Omni-7B. One epoch of joint training then uses the resulting checkpoint at a learning rate of $1 0 ^ { - 5 }$ . The joint epoch comprises 2,887 presentations and 181 optimizer steps: 1,155 task questions with graph dropout, 866 task questions with matched graphs, 578 verified graph questions, 144 matched-graph identification examples, and 144 mismatchedgraph identification examples. These are training presentations rather than 2,887 distinct videos. LoRA updates the languagemodel attention projections; the visual tower, audio tower, and remaining backbone parameters are frozen during optimization. Inference uses the merged task-adapted checkpoint and the joint-stage adapter.

## G. Evaluation Alignment and Graph-Input Audit

IntentBench evaluation uses a frozen set of 2,688 questions and aligned predicted graphs from 632 videos: 2,356 Social-IQ, 133 EMER, and 199 MDPE questions. One video from the original 2,689-question manifest was excluded by a fixed task identifier because person tracking repeatedly exhausted GPU memory; the exclusion preceded prediction and scoring. Training-example construction does not read the held-out benchmark answers. Evaluation predictions are saved before a separate scoring step accesses reference labels.

An initial shuffled-graph control changed structured\_evidence\_text, while the inference formatter read scene\_graph. Consequently, the actual graph-text input and raw output were identical for all 2,688 questions across that control and its matched run. Those predictions are excluded from the graphcorrespondence analysis. The corrected formatter reads structured\_evidence\_text; a pre-inference audit verified different graph-text inputs for every aligned question. Both corrected runs yielded 2,688 parsed predictions. This input audit establishes that the model received different graph text, without itself establishing a performance gain.

## H. Illustrative Video-to-Graph Examples

Fig. 5 illustrates how participant identity and a temporally localized reaction can be represented alongside a Social-IQ question. The video frames and the question’s reference answer come from sample $\mathtt { s o c i a l \_ i q \_ 0 0 1 9 8 3 }$ . The P1– P3 identifiers explain the intended representation.

Fig. 6 further illustrates participant grounding and temporal visual evidence. In particular, the upper timeline organizes the selected video frames chronologically, while the visual grounding panel associates the natural-language referents with the corresponding visible participants. The temporal visual evidence panel highlights the relevant moments in the interaction, and the final panel presents the Social-IQ question together with its benchmark reference answer. Together, the two illustrations show how participant identity, temporally localized evidence, and question–answer information can be organized within the proposed video-to-graph representation.

## V. DISCUSSION

a) Graph Construction and Graph Utilization.: The experiments examine two related but different questions. The first is whether the predicted graph retains information useful for video reasoning. The MER24 diagnostic addresses this question by comparing question answering with and without graph input when the source video and audio are excluded. The improvement with graph input indicates that the predicted graph contains information relevant to the questions. However, this diagnostic does not measure the effect of graph input when the reasoning model also receives the source video and available audio.

The second question is whether graph-specific supervision helps the reasoning model use the predicted graph. The Graph-SFT experiments compare graph-input conditions using the same adapter. The THG-Omni experiments further examine graph input after joint training. These comparisons provide evidence about graph utilization under different training conditions. The distinction is important because a predicted graph may retain useful information even when a reasoning model does not use that information effectively. THG-Omni addresses both graph construction and graph utilization, but the two properties are evaluated separately.

b) Graph–Video Correspondence.: The predicted graph contains participant attributes, actions, affective states, interactions, and temporal intervals associated with the evaluation video. The usefulness of this information depends on whether the graph corresponds to the correct participants and events. A shuffled graph may contain valid semantic claims from another video, but these claims do not necessarily provide relevant evidence for the current question. The shuffled-graph condition therefore examines graph–video correspondence rather than only the presence of additional graph text.

The IntentBench results are consistent with matched graph input providing more useful information than shuffled graph input. However, the updated matched-graph condition and earlier control conditions were not all evaluated in one paired run. The reported differences should therefore be interpreted as comparisons between the available evaluation conditions rather than paired statistical estimates. The corrected graphinput audit establishes that the matched and shuffled conditions receive different graph text, but it does not independently establish a performance improvement.

c) Limitations of Predicted Graphs.: The quality of the predicted graph depends on person tracking, semantic annotation, and identity binding. SAM3 provides person tracks and geometric relations, but occlusion, camera transitions, and changes in participant appearance may affect the available geometric evidence. Qwen3-VL provides semantic claims, but these claims may contain incorrect participant associations or temporal intervals. Although human quality control and HumanOmniV2 verification are used during training-data construction, these procedures cannot guarantee the correctness of every retained claim.

Evaluation graphs introduce an additional limitation. They are generated automatically from evaluation videos without manual semantic correction or claim verification. Consequently, a predicted graph may omit relevant information or include semantic claims that are not fully supported by the source video. The reasoning model must use the predicted graph together with the source video and available audio under these conditions. The compact-graph validation results also indicate that including more semantic claims does not necessarily lead to higher accuracy. The effect of graph content may depend on the question and the information retained in the graph.

d) Future Work.: Future work will focus on improving participant identity binding and the reliability of semantic claims in predicted graphs. More reliable person tracking across camera transitions may reduce incorrect associations between semantic claims and person tracks. Further work may also examine how to select graph fields according to the question, so that the reasoning model receives relevant participant and temporal evidence without unnecessary semantic claims. In addition, evaluating THG-Omni on longer videos and more complex interactions would help examine the effect of graph construction and graph-specific supervision under different video conditions.

## VI. CONCLUSION

In this work, we present THG-Omni, a framework for human-centric video reasoning with traceable graph supervision. It links SAM 3 participant tracks with temporally localized semantic annotations, verifies candidate claims against the source evidence, and uses the resulting graph to train an omni-modal model to interpret people, actions, affect, and interactions. Experiments on IntentBench show consistent improvements over the unadapted backbone across social reasoning, emotion understanding, and deception perception. Experiments on NExT-QA further examine the use of predicted graph evidence for causal, temporal, and descriptive video question answering. Additional graph-input experiments show that the structured representation retains answer-relevant evidence and can improve reasoning when the model is trained to use it. THG-Omni thus offers a way to improve video question answering while making the supporting people and events easier to identify.

## REFERENCES

[1] S. Ren, L. Yao, S. Li, X. Sun, and L. Hou, “TimeChat: A time-sensitive multimodal large language model for long video understanding,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 14313–14323.

[2] H. Wu, D. Li, B. Chen, and J. Li, “LongVideoBench: A benchmark for long-context interleaved video-language understanding,” in Advances in Neural Information Processing Systems, vol. 37, 2024, pp. 28828–28857.

[3] X. Zeng et al., “TimeSuite: Improving MLLMs for long video understanding via grounded tuning,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2025.

[4] G. Chen et al., “CG-Bench: Clue-grounded question answering benchmark for long video understanding,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2025.

[5] Y. Li, X. Wang, J. Xiao, W. Ji, and T.-S. Chua, “Transformer-empowered invariant grounding for video question answering,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 11, pp. 9510–9522, 2025.

[6] Z. Cheng et al., “Emotion-LLaMA: Multimodal emotion recognition and reasoning with instruction tuning,” in Advances in Neural Information Processing Systems, vol. 37, 2024, pp. 110805–110853.

[7] Q. Yang et al., “HumanOmniV2: From understanding to omni-modal reasoning with context,” 2025, arXiv:2506.21277.

[8] N. Ravi et al., “SAM 2: Segment anything in images and videos,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2025.

[9] R. Gao, J. Qi, and L. Wang, “Multiple object tracking as ID prediction,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025, pp. 27883–27893.

[10] N. Carion et al., “SAM 3: Segment anything with concepts,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2026.

[11] H. Fei, S. Wu, M. Zhang, M. Zhang, T.-S. Chua, and S. Yan, “Enhancing video-language representations with structural spatio-temporal alignment,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 12, pp. 7701–7719, 2024.

[12] J. Xiao, P. Zhou, A. Yao, Y. Li, R. Hong, S. Yan, and T.-S. Chua, “Contrastive video question answering via video graph transformer,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 11, pp. 13265– 13280, 2023.

[13] K. Kim et al., “Weakly supervised video scene graph generation via natural language supervision,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2025.

[14] J. Huang, Z. Li, M. Naik, and S.-N. Lim, “LASER: A neuro-symbolic framework for learning spatio-temporal scene graphs with weak supervision,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2025.

[15] J. Lee, Y. Wang, J. Li, and M. Zhang, “Multimodal reasoning with multimodal knowledge graph,” in Proc. 62nd Annu. Meeting Assoc. Comput. Linguistics (ACL), vol. 1, 2024, pp. 10767–10782.

[16] Y. Zhang et al., “NATIVE: Multi-modal knowledge graph completion in the wild,” in Proc. 47th Int. ACM SIGIR Conf. Res. Develop. Inf. Retrieval, 2024, pp. 91–101.

[17] Y. Huang et al., “Structure-CLIP: Towards scene graph knowledge to enhance multi-modal structured representations,” in Proc. AAAI Conf. Artif. Intell., vol. 38, no. 3, 2024, pp. 2417–2425.

[18] G. Zhou, Y. Yan, X. Zou, K. Wang, A. Liu, and X. Hu, “Mitigating modality prior-induced hallucinations in multimodal large language models via deciphering attention causality,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2025.

[19] S. Kang, J. Kim, J. Kim, and S. J. Hwang, “See what you are told: Visual attention sink in large multimodal models,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2025.

[20] J. Min, S. Buch, A. Nagrani, M. Cho, and C. Schmid, “MoReVQA: Exploring modular reasoning models for video question answering,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 13235–13245.

[21] D. Chen et al., “MLLM-as-a-Judge: Assessing multimodal LLM-as-a-Judge with vision-language benchmark,” in Proc. 41st Int. Conf. Mach. Learn. (ICML), vol. 235, 2024, pp. 6562–6595.

[22] E. J. Hu et al., “LoRA: Low-rank adaptation of large language models,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2022.

[23] A. Zadeh, M. Chan, P. P. Liang, E. Tong, and L.-P. Morency, “Social-IQ: A question answering benchmark for artificial social intelligence,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2019, pp. 8799–8809.

[24] C. Cai et al., “MDPE: A multimodal deception dataset with personality and emotional characteristics,” in Proc. 33rd ACM Int. Conf. Multimedia, 2025, pp. 12957–12964.

[25] J. Xu, Y. Rao, and J. Zhou, “Transferable unintentional action localization with language-guided intention translation,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 5, pp. 3863–3877, 2025.

[26] P. Jin, R. Takanobu, W. Zhang, X. Cao, and L. Yuan, “Chat-UniVi: Unified visual representation empowers large language models with image and video understanding,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 13700–13710.

[27] J. Li, X. Wang, and Z. Zeng, “Tracing intricate cues in dialogue: Joint graph structure and sentiment dynamics for multimodal emotion recognition,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 10, pp. 8786–8803, 2025.

[28] C. Fu et al., “Video-MME: The first-ever comprehensive evaluation benchmark of multi-modal LLMs in video analysis,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025, pp. 24108–24118.

[29] T. Hannan, M. M. Islam, J. Gu, T. Seidl, and G. Bertasius, “ReVision-LLM: Recursive vision-language model for temporal grounding in hourlong videos,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025, pp. 19012–19022.

[30] E. Song, W. Chai, T. Ye, J.-N. Hwang, X. Li, and G. Wang, “MovieChat+: Question-aware sparse memory for long video question answering,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 48, no. 1, pp. 374–389, 2026.

[31] A. Kirillov et al., “Segment anything,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023, pp. 3992–4003.

[32] J. Wu, W. Liu, Y. Liu, M. Liu, L. Nie, Z. Lin, and C. W. Chen, “A survey on video temporal grounding with multimodal large language model,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 48, no. 2, pp. 1521–1541, 2026.

[33] B. Huang, X. Wang, H. Chen, Z. Song, and W. Zhu, “VTimeLLM: Empower LLM to grasp video moments,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 14271–14280.

[34] M. Chen, L. Li, W. Wang, and Y. Yang, “DiffVsgg: Diffusion-driven online video scene graph generation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025, pp. 29161–29172.

[35] Y. Wang, X. Wu, S. Yang, and J. Luo, “End-to-end open-vocabulary video visual relationship detection using multi-modal prompting,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 8, pp. 6599–6615, 2025.

[36] X. Wang, Y. Zhang, O. Zohar, and S. Yeung-Levy, “VideoAgent: Longform video understanding with large language model as agent,” in Proc Eur. Conf. Comput. Vis. (ECCV), 2024, pp. 58–76.

[37] A. Deng, Z. Gao, A. Choudhuri, B. Planche, M. Zheng, B. Wang, T. Chen, C. Chen, and Z. Wu, “Seq2Time: Sequential knowledge transfer for video LLM temporal grounding,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025, pp. 13766–13775.

[38] W. Chen, Y. Liu, B. Chen, J. Su, Y. Zheng, and L. Lin, “Crossmodal causal relation alignment for video question grounding,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025, pp. 24087–24096.

[39] S. Li, B. Li, and B. Sun, “Towards visual-prompt temporal answer grounding in instructional video,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 12, pp. 8836–8853, 2024.

[40] S. Wang, J. Lin, X. Guo, J. Shun, J. Li, and Y. Zhu, “Reasoning of large language models over knowledge graphs with super-relations,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2025.

[41] J. Xiao, A. Yao, Y. Li, and T.-S. Chua, “Can I trust your answer? Visually grounded video question answering,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 13204–13214.

[42] T. Xiong et al., “LLaVA-Critic: Learning to evaluate multimodal models,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025, pp. 13618–13628.

[43] T. Yu et al., “RLAIF-V: Open-source AI feedback leads to super GPT-4V trustworthiness,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025, pp. 19985–19995.

[44] H. Qiu, M. Gao, L. Qian, K. Pan, Q. Yu, J. Li, W. Wang, S. Tang, Y. Zhuang, and T.-S. Chua, “STEP: Enhancing video-LLMs’ compositional reasoning by spatio-temporal graph-guided self-training,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025, pp. 3284–3294.

[45] T.-T. Nguyen, P. Nguyen, J. Cothren, A. Yilmaz, and K. Luu, “Hyper-GLM: HyperGraph for video scene graph generation and anticipation,”

in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025, pp. 29150–29160.

[46] Z. Liao, J. Li, L. Niu, and L. Zhang, “Align and aggregate: Compositional reasoning with video alignment and answer aggregation for video question-answering,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 13395–13404.

[47] H. Liu, F. Ilievski, and C. G. M. Snoek, “Commonsense video question answering through video-grounded entailment tree reasoning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025, pp. 3262–3271.

[48] K. Liu, M. Qu, Y. Liu, Y. Wei, W. Zhe, Y. Zhao, and W. Liu, “Singleframe supervision for spatio-temporal video grounding,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 7, pp. 5177–5191, 2025.

[49] J. Xiao, X. Shang, A. Yao, and T.-S. Chua, “NExT-QA: Next phase of question-answering to explaining temporal actions,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2021, pp. 9777–9786.

[50] Y. Zhang et al., “LLaVA-NeXT: A strong zero-shot video understanding model,” Apr. 2024. Accessed: Oct. 13, 2026. [Online]. Available: https: //llava-vl.github.io/blog/2024-04-30-llava-next-video

[51] J. Xu et al., “Qwen2.5-Omni technical report,” 2025, arXiv:2503.20215.

## AUTHOR BIOGRAPHY

![](images/795a266f5a600494a01e4693662879a4ec42c09b7340a2bf94ec9777294ec3a4.jpg)

Jisheng Dang received the Ph.D. degree from Sun Yat-sen University, China, advised by Prof. Jianhuang Lai and Prof. Huicheng Zheng. He worked as a research fellow at the NExT++ laboratory of the National University of Singapore, advised by Prof. Tat-Seng Chua. He is now a tenured associate professor at the School of

Information Science and Engineering, Lanzhou University. His research interests include multimodal learning, video understanding, and embodied intelligence. He has published several papers as the first author or corresponding author in major journals and conferences including IEEE TPAMI/TIP/TAC/TNNLS/ECCV/TITS/AAAI/ICLR/NeurIPS.

![](images/6a3352a740d0b95bd2ac24b70685a142568018c05a3d0a16cfc3a3bb192a7bc4.jpg)

Wei-Shi Zheng is a full Professor with Sun Yat-sen University. His research interests include person association, activity understanding, AI robotics learning, and related weakly supervised, unsupervised, and continuous learning algorithms. He has published more than 200 papers, including over 150 in leading journals (TPAMI, IJCV, SIGGRAPH, TIP) and top conferences (ICCV, CVPR,

ECCV, NeurIPS). He has served as Area Chair for ICCV, CVPR, ECCV, BMVC, and NeurIPS. He is a Cheung Kong Scholar Distinguished Professor, recipient of the NSFC Excellent Young Scientists Fund, a Fellow of IAPR, and recipient of the Royal Society ewton Advanced Fellowship (UK). He is an Associate Editor and editorial board member of IEEE TPAMI, Artificial Intelligence Journal, and Pattern Recognition.

![](images/235dfbc6e24d609a336d5489d245d26ec8497d467b93033c0e200d041df7008e.jpg)

Bin Hu (Fellow, IEEE) received the Ph.D. degree in computer science from the Institute of Computing Technology, Chinese Academy of Sciences, China, in 1998. Since 2008, he has been a Professor and Dean of the School of Information Science and Engineering, Lanzhou University. He has also held a guest professorship at ETH Zurich. He serves as Editor-in-Chief of IEEE Transactions

on Computational Social Systems and is a Fellow of IET and AAIA. His research interests include pervasive computing, computational psychophysiology, data modeling, and artificial intelligence.

![](images/8ec91b18d4e019f505202eb9a39b2b6f32089ca3d1a8870ec7379d06cbb5a607.jpg)

Qi Tian (Fellow, IEEE) received the Ph.D. degree in ECE from the University of Illinois at Urbana-Champaign, Champaign, IL, USA, in 2002. He is currently the Chief Scientist of Huawei Terminal BG. He was the Chief Scientist in computer vision with Huawei Noah’s Ark Laboratory from 2018 to 2020. Before he joined Huawei, he was a Full Professor with the Department of Computer Science,

The University of Texas at San Antonio, San Antonio, TX, USA, from 2002 to 2019. He was listed in the top ten of the 2016 Most Influential Scholars in Multimedia by Aminer.org. He is an IEEE Fellow (Class 2016) and an Academician of the International Eurasian Academy of Sciences (elected in 2021), with 113,457 citations on Google Scholar.

![](images/a54fced820ca8b60827797bf66bc464b4f34b08298cc8eec5538fec461c408fd.jpg)

Tat-Seng Chua received the Ph.D. degree from the University of Leeds, U.K. He is the KITHCT chair professor with the School of Computing, National University of Singapore, where he was the acting and founding dean of the School from 1998 to 2000. He is the co-director of NExT, a joint center between NUS and Tsinghua University, to develop technologies for live social media search. He is the

2015 winner of the prestigious ACM SIGMM Award. He is the chair of the Steering Committee of the ACM International Conference on Multimedia Retrieval (ICMR) and the Multimedia Modeling (MMM) conference series. He is also the general co-chair of ACM Multimedia 2005, ACM CIVR (now ACM ICMR) 2005, ACM SIGIR 2008, and ACM Web Science 2015. He serves on the editorial boards of four international journals. He is the co-founder of two technology startups in Singapore and a Fellow of the Singapore Academy of Sciences, with 114,163 citations on Google Scholar.