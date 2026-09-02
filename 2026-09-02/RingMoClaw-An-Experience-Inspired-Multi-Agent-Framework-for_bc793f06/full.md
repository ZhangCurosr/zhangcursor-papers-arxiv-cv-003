# RingMoClaw: An Experience-Inspired Multi-Agent Framework for Self-Evolving Research in Remote Sensing

Kaiyue Kang, Qixuan He, Peijin Wang, Yingchao Feng, Chao Ren, Kangxin Wang, Wenhui Diao, Yixiao Wang, Liangjin Zhao, Kaiwen Wei, Nayu Liu, Xian Sun

Abstract—Remote sensing visual models have continuously advanced various interpretation tasks. However, the research process behind model improvement still heavily relies on manual expertise, requiring extensive trial-and-error iterations in model design, data processing, and performance diagnosis. Existing agentbased approaches mainly focus on task execution and workflow orchestration, while lacking the capability of autonomous research iteration for continuous performance optimization. To address this issue, we propose RingMoClaw, an experienceinspired self-evolving multi-agent framework for remote sensing visual interpretation. RingMoClaw integrates a research branch, a quality-control branch, and a dual-stream dynamic experience bus to establish a closed-loop optimization process covering strategy generation, experiment execution, independent review, and experience accumulation. The heterogeneous Critic mechanism provides stage-wise diagnosis and feedback, while the dual-stream experience bus incorporates external knowledge and internal experimental experience to guide strategy evolution and eliminate ineffective searches. Extensive experiments on four remote sensing downstream tasks, including object detection, scene classification, semantic segmentation, and change detection, demonstrate the effectiveness and generalization of RingMoClaw. Compared with the corresponding baseline models, RingMoClaw improves performance by 1.84% mAP<sub>50</sub> on object detection and achieves consistent gains across the other three tasks, while reducing the required evolution steps by over 40% compared with existing research automation frameworks. These results suggest that RingMoClaw offers a feasible route from task execution toward continuous research driven model evolution in remote sensing.

Index Terms—Remote sensing, Multi-agent system, Large language model, Autonomous research, Self-evolution.

## I. INTRODUCTION

Remote sensing earth observation is an essential means of acquiring dynamic information about the land surface, and provides critical geospatial support for environmental monitoring, disaster response, urban planning, and agricultural management [1], [2]. With the continuous improvement of remote sensing observation capability, the scale of remote sensing data keeps growing and the information they contain becomes increasingly rich. This places higher demands on the efficiency and accuracy of remote sensing interpretation. In recent years, remote sensing visual interpretation methods have made steady progress in tasks such as object detection, semantic segmentation, scene classification and land-cover mapping [3]–[9]. Beyond network architecture design, recent studies pay more attention to learning strategies that cope with limited annotation and distribution shift, including domain adaptation, few shot learning, and semi supervised learning for scene classification [10]–[12], as well as few shot learning, active label selection, and cross domain consistency for object detection [13]–[16]. These results indicate that accuracy gains depend on coordinated choices across model architecture, data processing, and training strategy. However, the underlying research process still relies heavily on manual expertise. For a given task, researchers often need to repeatedly adjust model design, training strategy, data processing, and result analysis through trial and error. This expert-driven paradigm is costly and time-consuming, and also limits the reproducibility, scalability, and long-term optimization efficiency of remote sensing algorithm development.

The capabilities of large language models (LLMs) in complex reasoning, task planning, and code generation [17], [18] have accelerated the development of agent technology in geospatial and remote sensing domains. Existing studies can be broadly grouped into two categories. The first category focuses on tool calling and workflow orchestration. Representative systems such as GeoGPT [19], ChangeGPT [20], GeoLLM-Squad [21], and GeoColab [22] extend the scope of agents in geospatial task understanding, remote sensing change analysis, multi agent workflow collaboration, and geospatial code generation, and improve the automated execution of complex geospatial tasks through natural language understanding, task decomposition, and module coordination. The second category focuses on unified agent organization and open-ended capability extension for complex remote sensing scenarios. Earth-Agent [23] and CangLing-KnowFlow [24] contribute to cross modal earth observation reasoning, procedural knowledge organization, evolving memory, and multi step workflow management, and strengthen the reasoning and scheduling capability of agents on complex remote sensing tasks. Built upon the OpenClaw research agent paradigm [25], OpenEarth-Agent [26] extends this line from tool calling to tool creation and enables agents to dynamically generate dedicated tools for unseen data and tasks, which improves the autonomous execution of earth observation tasks in open environments. Overall, these studies have substantially advanced the automation of task execution in remote sensing and geospatial analysis, and have pushed agents from calling existing capabilities toward organizing and extending the task solving workflow.

![](images/15109493b8b2a3607a9649c9d652942e824a21116698c9f25286bc6d4fb026be.jpg)  
Fig. 1. Comparison of existing remote sensing research agents and RingMoClaw. RingMoClaw introduces a unified research loop with independent critique and dual-stream experience evolution for continuous model improvement.

Although the above studies have substantially advanced the automation of task execution in remote sensing and geospatial analysis, the main focus remains on task solving, workflow orchestration, and capability extension. Support for research iteration aimed at continuous performance improvement is still limited. This limitation is more evident in remote sensing vision tasks, which often involve long-tailed category distributions, missed detection of small objects, and domain-specific prior constraints. As shown in Fig. 1, existing remote sensing agents still have the following three limitations:

• Lack of a unified research loop for continuous model improvement. Existing systems can decompose tasks, call tools, and organize workflows for a given objective [19]–[22]. However, most systems are designed for one-pass task completion. A closed loop that connects hypothesis generation, experiment execution, performance analysis, and strategy update is still missing. As a result, sustained model optimization remains difficult.

• Lack of independent and structured quality control during research iteration. Some recent frameworks improve reasoning and workflow management for complex tasks [23], [27]. However, explicit mechanisms for independent review are still limited. Structured diagnosis of plan design, data processing, and experimental results is usually absent. Quantitative feedback for key bottlenecks is also insufficient. This makes targeted correction during iteration difficult.

• Lack of continuous knowledge injection and experience evolution for strategy update. Existing systems have explored retrieval, memory, and even tool creation in open environments [23], [26], [27]. However, recent literature, open-source implementations, and historical experimental records are not jointly integrated into a unified optimization process. External knowledge is dif ficult to transform into actionable strategies. Internal experimental experience is also difficult to accumulate as reusable guidance. Consequently, dynamic expansion of the strategy space and pruning of inefficient paths remain limited.

To address the above limitations, this paper proposes Ring-MoClaw, an experience-inspired self-evolving multi-agent framework built upon RingMo [5] for remote sensing visual tasks. RingMoClaw is designed to support not only task execution, but also iterative model optimization for continuous performance improvement. The framework consists of a research branch for plan generation, data processing, and experiment execution, a quality control branch for independent diagnosis and feedback, and a dual-stream dynamic experience bus for external knowledge injection and internal experience accumulation. Together, these components form a unified optimization loop for sustained research iteration in remote sensing visual interpretation. The main contributions of this paper are summarized as follows:

1) We propose RingMoClaw, an experience-inspired selfevolving multi-agent framework for remote sensing visual tasks. Beyond conventional agent systems for task execution, RingMoClaw automates the research iteration behind model improvement, where the optimization trajectory is dynamically refined through experimental feedback and accumulated experience.

2) We design a heterogeneous Critic mechanism for structured quality control. By conducting staged review across the research pipeline, the Critic quantitatively evaluates task completion, bottleneck impact, and historical experience relevance. The resulting score guides experience selection and triggers pruning of unpromising paths.

3) We propose a Dual-stream Dynamic Experience Bus that integrates external knowledge flow and internal experimental flow into the research loop. Retrieved knowledge and experimental trajectories are transformed into reusable experience and mapped back to the structured search space, enabling dynamic expansion and refinement of optimization strategies during evolution.

4) We evaluate RingMoClaw on four representative remote sensing tasks, including object detection, scene classification, semantic segmentation, and change detection, to assess its capability in autonomous research iteration. The framework improves the corresponding baselines by 1.84% $\mathrm { \ m A P _ { 5 0 } }$ on object detection and by 1.79% to 2.87% on the remaining three tasks, and reaches convergence with at least 40% fewer evolution rounds than existing research automation frameworks.

## II. RELATED WORK

## A. Remote Sensing Visual Models and Foundation Models

Remote sensing agents fundamentally rely on strong perception and multimodal representation backbones. SatMAE demonstrates that masked autoencoding tailored to temporal and multispectral imagery can learn transferable representations for Earth observation (EO) data, providing an effective pretraining paradigm for remote sensing perception [28]. AnySat extends this direction by learning a unified EO model across multiple spatial resolutions and sensing modalities, enabling consistent representation across heterogeneous inputs [29]. Recent efficient backbone designs, such as RS-vHeat and RingMamba, further improve remote sensing foundation models through heat-conduction-guided representation learning and visual state-space-based multi-sensor pretraining, respectively [30], [31]. RingMoGPT further explores the integration of vision and language within a unified foundation model framework, supporting grounded multimodal understanding in remote sensing scenarios [32]. Collectively, these studies indicate a shift from task-specific models toward more generalpurpose perception backbones in EO systems.

A complementary line of work introduces language grounding and temporal reasoning capabilities that improve the usability of these perception models in interactive settings. GeoChat presents a grounded vision-language model capable of region-level reasoning and conversational interaction over high-resolution imagery [33]. RS5M and GeoRSCLIP contribute large-scale image-text alignment resources and domainspecific vision-language models, enabling open-vocabulary retrieval and language-conditioned perception [34]. SkyEyeGPT unifies multiple EO vision-language tasks through instruction tuning within a single framework [35]. TEOChat extends this paradigm to temporal sequences, supporting conversational reasoning over multi-temporal observations [36]. RingMo-Agent further targets unified multimodal reasoning across diverse sensing platforms [37]. Although these systems are not full workflow agents, they provide essential perception and interaction capabilities for building tool-augmented remote sensing agents. Building on these perception and interaction foundations, recent research has increasingly focused on integrating tool usage, workflow planning, and execution into geospatial analysis systems.

B. Tool-Augmented Geospatial Agents and Workflow Orchestration

Recent research in geospatial AI increasingly formulates Earth observation analysis as a multi-step workflow problem rather than a single prediction task. The autonomous GIS paradigm argues that future GIS systems should support end-to-end spatial workflows involving data discovery, tool selection, reasoning, execution, and verification, treating geospatial analysis as a closed-loop decision process [38]. At the system level, GIS Copilot enables natural-language spatial analysis within QGIS by retrieving documentation and generating executable PyQGIS programs [39]. GeoAgent further demonstrates that explicit control mechanisms, such as code interpretation and retrieval augmentation, can improve the reliability of complex geospatial reasoning pipelines [40].

Domain-specific agent research extends this direction toward multi-step remote sensing workflows. GeoLLM-Engine introduces an interactive environment with maps, APIs, and long-horizon commands that better reflect real-world remote sensing analysis processes [41]. GeoLLM-Squad decomposes remote sensing workflows across specialized agents, showing that explicit orchestration improves performance in complex EO pipelines [42]. OpenEarthAgent further investigates unified geospatial agents trained on verified tool trajectories [43]. AutoGEEval++ complements these studies by evaluating large language models through executable Google Earth Engine workflows, highlighting the importance of end-to-end verification in tool-oriented geospatial intelligence [44]. Collectively, these studies demonstrate the emergence of tool-augmented geospatial agents capable of executing multi-step workflows in realistic analysis environments. However, most existing systems primarily focus on task execution and workflow orchestration, while relatively limited attention has been given to mechanisms for continual performance improvement and long-horizon capability evolution.

## C. OpenClaw-Based General Agent Frameworks

To address the limitations of static workflow execution, recent research has explored more general-purpose agent frameworks that emphasize continual improvement, long-horizon evaluation, and adaptive workflow optimization. OpenClaw-RL models heterogeneous interaction signals, including user feedback, tool outputs, terminal responses, and interface states, as unified next-state transitions for reinforcement learning, enabling agents to improve policies from real interaction traces [25]. This approach highlights the potential of leveraging diverse interaction signals for online policy adaptation, though its effectiveness may be constrained by sparse, noisy, or delayed feedback, particularly in high-dimensional, multi-modal remote sensing contexts.

![](images/b2c3e78bf5bcd9dbd292e82949f3c636fabf64e00bb6da095465bd24bbb66ebd.jpg)  
Fig. 2. Overall architecture of RingMoClaw. The self-evolving research pipeline (left) and the Critic pipeline (right) are coupled through a dual-stream dynamic experience bus (center), which integrates external knowledge flow and internal experimental flow to support automated research iteration and the continuous refinement of the structured search space.

EvoClaw further argues that agent evaluation should extend beyond isolated tasks to dynamic environments in which objectives and software targets evolve over time [45]. While such long-horizon evaluation is critical for sustained capability growth, current studies primarily target software or web-based tasks, leaving open the question of how similar principles can be applied to complex EO workflows with evolving environmental conditions and multi-sensor inputs. MetaClaw advances this direction by combining skill synthesis with opportunistic policy optimization, supporting adaptive capability growth in long-horizon workflows [46]. Although the framework demonstrates the value of flexible skill composition and continuous adaptation, its application to integrated EO agents that must orchestrate multiple perception modules and data processing tools remains largely unexplored.

At the system level, these frameworks collectively illustrate a shift from static task execution toward agents capable of sustained performance improvement through continuous interaction and feedback. They highlight the importance of runtime adaptability, memory-aware computation, and real-time perception-decision-action loops, concepts akin to Streaming-Claw, which are highly relevant to EO scenarios where agents process streaming satellite or unmanned aerial vehicle (UAV) imagery under partial observability, high-resolution demands, and environmental noise. Existing EO agent studies, by con trast, typically operate in offline, batch-processing modes, lacking mechanisms for continuous self-optimization or online adaptation.

Another line of research in general agent systems focuses on multi-agent collaboration and ecosystem-level coordination, as demonstrated by ClawdLab/Beach.Science and hospitaloriented agent operating system studies [47], [48]. These works show that emergent collaboration and division of labor can be effectively organized around a shared agent platform. Agents can autonomously negotiate task allocation and dynamically adjust their roles based on environmental feedback. Such collective intelligence enables the system to handle complex, multi-step objectives more efficiently than isolated agents.

Despite these advances, most existing general agent frameworks are developed in software engineering or web automation settings, and their applicability to complex remote sensing workflows remains underexplored. In particular, integrating multimodal perception, reliable tool orchestration, and continual performance self-evolution within a unified remote sensing agent architecture remains an open challenge. This motivates our work, which aims to bridge these gaps by designing EO agents capable of adaptive skill composition, online policy refinement, and multi-agent coordination, ultimately enabling sustained performance improvement in dynamic, real-world remote sensing environments.

## III. METHODOLOGY

## A. Overall Framework

To move beyond the task-specific tool-calling paradigm toward genuine intelligent synthesis, we propose RingMoClaw, a multi-agent architecture for experience-inspired autonomous research evolution. As illustrated in Fig. 2, the architecture is composed of a Self-Evolving Research Pipeline, a Critic mechanism, and a Dual-stream Dynamic Experience Bus.

<table><tr><td>Algorithm 1 An Instantiated Self-Evolving Trajectory for Object Detection on FAIR1M</td></tr><tr><td>Require: Task T, initial model M, experience pool  $\overline { { \mathcal { E } } }$  Ensure: Optimized detector  $M ^ { * }$  Stage 1: Initialization and Diagnostic Planning 1: Main Agent analyzes task T and generates an initial</td></tr><tr><td>optimization roadmap R; 2: Vision Agent trains the baseline; Critic mechanism iden- tifies weak-class bottlenecks; Stage 2: Iterative Policy Optimization 3: Main Agent selects a data-level direction from  $ { S _ { \mathrm { d a t a } } }$  (here</td></tr><tr><td>instantiated as augmentation: rotation, flip, mixup); re- tained if recall improves; 4: Main Agent selects a few-shot direction for weak classes from the current search space; 5: if overall accuracy decreases then</td></tr><tr><td>6: Critic mechanism rejects the direction and records the failure in E to avoid repetition; 7: end if Stage 3: Automated Module Evolution (Auto-Research) 8: Main Agent selects a resampling strategy from  $\boldsymbol { S } _ { \mathrm { d a t a } }$  (here instantiated as ClassAwareSampler);</td></tr><tr><td>9: Main Agent integrates a training-strategy module and an externally retrieved neck module from  $ { S } _ { \mathrm { t r a i n } }$  and  $\boldsymbol { S } _ { \mathrm { a r c h } }$  (here instantiated as FocalLoss [49] and the reused BFPA module [50]), each promoted only after passing agile verification;</td></tr><tr><td>10: Main Agent proposes two newly designed modules be- yond the existing search space (the Multi-Scale Feature Refinement and Contrastive Prototype Refinement mod- ules), and the Critic mechanism validates their effect; 11: Validated knowledge is accumulated into E for cross-task evolution;</td></tr></table>

The Self-Evolving Research Pipeline serves as the operational core, where the Main Agent, Data Agent, and Vision Agent collaborate to decompose complex research objectives (e.g., fine-grained detection) into a staged evolution trajectory. Unlike conventional frameworks, this pipeline does not merely execute commands; it instantiates potential solutions from an evolving knowledge base.

The Critic mechanism provides the necessary independent external review. By conducting a three-phase diagnostic (Plan, Data, and Result), it identifies experimental anomalies and accuracy bottlenecks that often escape standard automated systems. These insights are quantified into a Multi-Dimensional Score t, ensuring that every failure or success is distilled into reusable intelligence.

The Dual-stream Dynamic Experience Bus acts as the system’s “cognitive center.” It hosts a Dual-Stream Experience Engine that aggregates external RAG-based knowledge with internal experimental feedback. By mapping these streams onto a Structured Search Space S, the bus continuously prunes unpromising paths and refines optimization directions. Through this mechanism, every iteration of the research branch is no longer a blind trial-and-error but a precise step informed by the quality control branch, driving the system to converge rapidly toward a task-specific improved solution.

For reproducibility, all agents are driven by fixed and version-controlled system prompts. Each prompt specifies the role-specific behavior, declared inputs, and a structured JSON output contract. At each evolution round, the agent receives a serialized state containing the current task, identified bottleneck, candidate under review, historical summary, and retrieved experience. All outputs are schema-validated before entering the subsequent stage. The complete prompts are provided in Appendix E.

To illustrate this process, we take object detection as a representative example. As shown in Fig. 3, the self-evolving pipeline and the critic mechanism interact throughout the optimization loop. Algorithm 1 records one representative trajectory that the system autonomously explored on FAIR1M. The specific techniques in it, such as ClassAwareSampler, FocalLoss [49], and BFPA [50], are outcomes of the search over S rather than a predefined script. Along this trajectory, the self-evolving pipeline performs staged optimizations spanning baseline training, data augmentation, few-shot detection, and automated module replacement, integration, and innovation. In parallel, the critic mechanism conducts multi-level reviews on plans, data, and results to guide iterative improvements. This collaborative mechanism ties model optimization closely to structured experience accumulation, converting experimental successes into permanent system knowledge.

## B. Critic Mechanism: Heterogeneous Review and Targeted Feedback

Within the self-evolving framework, the Critic mechanism is responsible for independent diagnosis, quantitative evaluation, and targeted feedback on the outputs of different stages in the execution chain. It serves as a key mechanism for sustaining model optimization during research iteration. Its design follows two principles: heterogeneity and functional separation.

At the model level, the execution chain is built on GLM-5 Turbo, while the Critic mechanism is powered by Minimax 2.7. The two models differ in pretraining data, reasoning style, and knowledge preference. This heterogeneous configuration allows the Critic mechanism to examine the outputs of the execution chain from a distinct perspective and reduces the homogeneous bias that may arise from self-evaluation by the same model. At the functional level, the Critic mechanism participates in neither plan generation nor experiment execution, and is only responsible for review and diagnosis. The execution chain is oriented toward task completion, whereas the Critic mechanism is oriented toward problem identification and quality assessment. This functional separation provides an external safeguard for the research loop and helps the system avoid ineffective optimization paths.

1) Staged Review across the Full Chain: The Critic mechanism intervenes at three key nodes of the execution chain rather than evaluating only the final result. As illustrated in

![](images/08a5985bd623b677d4ef5d64a7f0832cd55bf97a20f0c8162954c02f13301bc3.jpg)  
Fig. 3. Object-detection workflow of the proposed framework. The research pipeline performs staged model optimization, while the Critic mechanism reviews intermediate outputs and returns feedback for strategy revision and experience accumulation.

Fig. 4, the three stages focus on different aspects.

(a) Plan review. The Critic mechanism examines the rationality of the technical plan produced by the Main Agent. The review considers: the alignment between the plan and the task requirement, for example whether an appropriate anchor box setting is chosen for an object detection task; the consistency between the plan and the historical records as well as the literature priors in the global experience pool; and the internal logical coherence of the plan, for example whether the loss function matches the data characteristics. The purpose of this stage is to intercept clearly unreasonable plans before execution and to avoid the waste of computation.

(b) Data review. The Critic mechanism examines whether the data processing scheme is appropriate. The review considers: the fit between the preprocessing strategy and the properties of remote sensing data, for example whether the tile size for large images is reasonable and whether class imbalance is sufficiently handled; the integrity and the format compliance of the processed data; and the coordination between the data processing scheme and the technical plan issued by the Main Agent.

(c) Result review. The Critic mechanism performs a comprehensive quantitative evaluation and diagnosis of the training results. The review considers: whether the overall performance reaches the domain baseline recorded in the global experience pool; whether the performance across categories is balanced; whether the training process exhibits overfitting or instability; and whether a substantive improvement is achieved over the previous round.

2) Structured Diagnosis and Multi Dimensional Scoring: To make the review results quantifiable, comparable, and traceable, the Critic mechanism introduces a multi dimensional scoring mechanism that produces a comprehensive evaluation of each reviewed output. The scoring covers three dimensions.

(a) Task completion. $C _ { t }$ reflects the progress of the current stage toward the predefined objective. In plan review, it measures how well the plan covers the task requirement; in data review, it measures the completeness of the data processing; in result review, it measures how closely the performance approaches the target. The value lies in [0, 1].

(b) Bottleneck impact. $B _ { t }$ quantifies how strongly the key deficiency of the current stage affects the overall task performance. The core of this dimension is to identify the primary weakness, which may be an unreasonable architecture in the plan stage, an unresolved class imbalance in the data stage, or a severely lagging category in the result stage. A larger $B _ { t }$ indicates a more severe bottleneck. The value lies in [0, 1].

(c) Global experience influence. $H _ { t }$ measures the consistency of the current direction with the successful and failed experiences retrieved from $\mathcal { M } _ { \mathrm { g b } }$ by the retrieval mechanism described in Section III-C2. These entries are provided to the Critic Agent as reference information, and the Critic Agent assigns $H _ { t } \in [ 0 , 1 ]$ , where a higher value indicates stronger consistency with previously successful directions and greater avoidance of known failure patterns.

The comprehensive score is then given by

$$
\mathrm { S c o r e } _ { t } = w _ { 1 } C _ { t } + w _ { 2 } ( 1 - B _ { t } ) + w _ { 3 } H _ { t } ,\tag{1}
$$

where $w _ { 1 } , \ w _ { 2 } .$ and $w _ { 3 }$ are stage-dependent weights fixed before evaluation. Specifically, $( w _ { 1 } , w _ { 2 } , w _ { 3 } )$ is set to (0.30, 0.50, 0.20) for plan review, (0.35, 0.40, 0.25) for data review, and (0.50, 0.30, 0.20) for result review. Thus, plan review places greater emphasis on the severity of the remaining bottleneck, whereas result review places greater emphasis on task completion. The final score is recomputed by the controller from the $C _ { t } , \ B _ { t } .$ , and $H _ { t }$ values returned by the Critic Agent to ensure consistent scoring.

The review action is determined by explicit rules. If critical evidence is missing, the output is marked as escalate. Otherwise, it is accepted when $\mathrm { S c o r e } _ { t } \geq 0 . 7 5$ and $B _ { t } \leq 0 . 3 0$ returned for revision when $\mathrm { S c o r e } _ { t } \geq 0 . 4 5$ but the acceptance condition is not met, and rejected otherwise. The Critic Agent therefore provides the semantic diagnosis and component scores, while score aggregation and the final decision follow fixed rules.

3) Feedback Driven Decision Update and Global Experience Accumulation: All feedback from the Critic mechanism is routed to the Main Agent, which performs global coordination and decision making, rather than issuing commands directly to subordinate agents. The decision update of the Main Agent after receiving a diagnostic report can be formalized as

$$
o _ { \mathrm { m a i n } } ^ { t + 1 } = \Pi _ { \mathrm { u p d a t e } } ( s _ { t } , \{ c _ { t } , { \mathrm { S c o r e } } _ { t } \} ) ,\tag{2}
$$

where $s _ { t }$ denotes the system state at the current stage and $\{ c _ { t } , { \mathrm { S c o r e } } _ { t } \}$ denotes the structured diagnosis and the score produced by the Critic mechanism. Meanwhile, the complete record of each iteration, including the technical plan, the experimental configuration, the score, and the diagnostic conclusion, is standardized and written into the global experience pool:

![](images/de7ab7ce7f7d368fc0638c1c74218bb0074daf5b127f660f3d79acbd403ca2c1.jpg)  
Fig. 4. Illustration of the Critic mechanism. The Critic mechanism intervenes at three key nodes of the execution chain, namely the plan review, the data review, and the result review. All feedback is routed through the Main Agent for global coordination, and every iteration is written into the global experience pool after standardization.

$$
{ \mathcal { E } } _ { \mathrm { g l o b a l } } = \bigcup _ { t = 1 } ^ { T } { \mathrm { S e r i a l i z e } } ( c _ { t } ) .\tag{3}
$$

Consequently, every round of feedback from the Critic mechanism drives two processes simultaneously: the immediate optimization of the current task, and the continuous accumulation of the global experience pool. The more tasks the system handles, the richer the pool becomes. In subsequent tasks, the evaluation baseline of the Critic mechanism becomes more accurate, the initial plan produced by the Main Agent is of higher quality, and the overall convergence speed improves steadily.

## C. Dual-stream Dynamic Experience Bus

To ensure that the system can process diverse visual tasks without falling into blind trial-and-error, we design a dualstream dynamic experience bus. Through this bus, the system moves from solving a single task to accumulating transferable knowledge across tasks.

As shown in Fig. 5, we first summarize common optimization strategies for representative remote sensing tasks and organize them into a structured search space with four dimensions: model architecture, training strategy, data processing, and system integration:

$$
\mathcal { S } = \{ S _ { \mathrm { a r c h } } , \ S _ { \mathrm { t r a i n } } , \ S _ { \mathrm { d a t a } } , \ S _ { \mathrm { i n t g } } \} .\tag{4}
$$

Here, $\boldsymbol { S } _ { \mathrm { a r c h } }$ covers modifications to the backbone, neck, and feature-fusion modules; $ { S _ { \mathrm { t r a i n } } }$ includes loss functions, optimization strategies, and hyperparameter settings; $ { S _ { \mathrm { d a t a } } }$ includes resampling, preprocessing, and data augmentation; and $\mathcal { S } _ { \mathrm { i n t g } }$ covers auxiliary models and system-level integration strategies. These predefined operators form the base search space. During evolution, methods retrieved from external literature and open-source repositories can be incorporated into the corresponding dimension after filtering and structured parameterization, allowing the search space to be extended without unconstrained generation by the agents. The detailed operator definitions and instantiation rules are provided in Appendix B.

The predefined base space contains 23 operators across model architecture, training strategy, data processing, and system integration. Each operator specifies its applicable bottleneck tags and a constrained parameter space. New operators obtained from external retrieval or generated during later research evolution are not included in this base catalog.

1) External Knowledge Flow: To address the limitation of fixed knowledge bases in handling complex research logic and emerging research needs, we design an online branch based on retrieval augmented generation. The system retrieves knowledge from external sources, applies differentiated extraction to textual knowledge and code knowledge, and then maps the results jointly into executable instructions in S.

(a) High level method extraction from text. For document type knowledge such as the methodology sections of papers and the README or implementation notes of open source repositories, the system applies semantic distillation with a large language model and focuses on novel loss functions, learning rate schedules, and network connection patterns. These unstructured descriptions are mapped into heuristic guidance vectors within the search space. For instance, if a paper reports that introducing an edge gradient constraint improves detection accuracy, the system formalizes it as an optimization direction $S _ { \mathrm { t r a i n } } ( \mathcal { L } _ { \mathrm { e d g e } } )$ in the training strategy layer.

(b) Low level operator extraction from code. The system does not directly copy source code. It first removes the peripheral code related to input/output, logging, and framework scaffolding, locates the core forward pass, and extracts the essential components into a reusable operator ω. The operator is defined as

$$
\omega = \{ \mathcal { T } , \ \mathcal { O } , \ \mathcal { F } , \ \mathcal { P } \} ,\tag{5}
$$

where I and O are the input and output tensors, $\mathcal { F }$ is the core operator logic (for example a BiFPN fusion or a cross attention module), and $\mathcal { P }$ denotes the distribution of key hyperparameters. The agent then analyses the current backbone and automatically searches for candidate mounting positions that match ω.

By combining the evolution directions provided by the textual layer and the standardized operators provided by the code layer, the system generates instantiated execution instructions in S. This mechanism ensures that the latest state of the art paradigms can be integrated on demand.

![](images/f09543918471080740e5fb052e9b9d95ec391384205ed77da5a41b6e0103ce40.jpg)  
Fig. 5. Dual-stream dynamic experience bus for autonomous research evolution. External knowledge from literature and code and internal experience from task memory and cross-task memory are integrated into global and local experience pools. The experience bus dynamically prunes and expands the structured search space, guides candidate scheme instantiation, and updates the main agent through agile verification, thereby transforming blind trial-and-error search into experience-guided optimization toward task-specific solutions.

2) Internal Experimental Flow: At the end of each iteration, the Critic mechanism conducts an in depth review of the experimental trajectory and uncovers the conditions that lead to accuracy fluctuation or training failure. For the experience knowledge produced in such re experiments, we adopt a hierarchical memory with explicit update logic over different life cycles.

(a) Local experience within a task. This memory integrates the failure cases and the performance bottlenecks reported by the Critic mechanism across the iterations of the current task. Its core role is to construct a negative memory $\mathcal { M } _ { \mathrm { l o c } }$ that records hyperparameter combinations leading to severe accuracy fluctuation or resource overflow. During the execution of a single task, the memory drives the agent to perform real time path pruning and terminates unpromising attempts as soon as a potential error direction is detected, for example when a training round fails to reach the historical level.

(b) Global experience across tasks. This memory serves as the cognitive center of the system. It organizes experimental data into transferable research knowledge and is never cleared after a task is completed. Based on the stagewise completion score produced by the Critic mechanism, the local negative memory is weighted to form a general prior for the experimental phenomena of the task, which is stored as $\mathcal { M } _ { \mathrm { g b } }$ . The incremental update is expressed as

$$
\mathcal { M } _ { \mathrm { g b } }  \mathcal { M } _ { \mathrm { g b } } \cup \mathrm { F i l t e r } ( \mathcal { M } _ { \mathrm { l o c } } , \mathrm { ~ S c o r e } _ { t } ) ,\tag{6}
$$

where Score is the comprehensive score from Section III-B2 and serves as the quality threshold for admission into the global memory, and Filter denotes the score based noise filtering that guarantees that only high quality or instructive paths are retained. When facing a new task, the system retrieves relevant successful experiences and failure records from $\mathcal { M } _ { \mathrm { g b } }$ according to a weighted task-similarity measure that considers the task description, current bottleneck, dataset characteristics, model, and primary evaluation metric. The detailed similarity computation and retrieval thresholds are provided in Appendix C.

3) Decision Evolution under Experience Guidance and the Agile Verification Protocol: Guided by the dual-stream experience bus, each candidate strategy is first evaluated by a constrained five-minute training run:

$$
f _ { \mathrm { s c o r e } } ( m ) = \alpha \frac { \Delta \mathrm { A c c } } { \Delta t } + \beta \exp \left( - \sigma \| \nabla _ { \theta } ^ { 2 } \| \right) ,\tag{7}
$$

where $\Delta \mathrm { A c c }$ denotes the task-specific performance gain and the second term characterizes training stability. In practice, the coefficient of variation of the final 50 training losses is used as a lightweight surrogate for the stability term. ∆Acc is expressed in percentage points, ∆t is normalized to one verification window, and $\gamma$ is fixed to 0.55 for all tasks.

A candidate is promoted to full training only if it achieves a positive performance gain, completes verification without training failure, satisfies $f _ { \mathrm { s c o r e } } ( m ) > 0 . 5 5$ , passes the resultstage Critic check, and is not rejected by negative memory. Otherwise, the candidate is discarded and the evolution continues. The evolution terminates when the maximum of 20 rounds is reached or no candidate improves the best-so-far result for two consecutive full-experiment rounds.

## IV. EXPERIMENTS

In this section, we evaluate the proposed framework from four complementary perspectives. We first describe the experimental settings, including the datasets, baselines, and evaluation protocols (Section IV-A). We then report the overall performance and the evolution process on four remote sensing visual tasks, together with a cross-task comparison against existing research automation frameworks (Section IV-B), and analyze the practical cost and search efficiency (Section IV-C). Finally, we conduct ablation studies on the Critic mechanism and the dual-stream experience bus, and validate the fiveminute verification protocol (Section IV-D).

TABLE I  
OBJECT DETECTION RESULTS ON FAIR1M (%). OVERALL MAP<sub>50</sub> AND PER-CATEGORY AP ARE REPORTED FOR SELECTED WEAK CATEGORIES. REPRESENTATIVE DETECTORS ARE COMPARED WITH THE BASELINE AND SUCCESSIVE EVOLUTION STAGES OF RINGMOCLAW.
<table><tr><td>Method</td><td>mAP50</td><td>C5</td><td>C10</td><td>C11</td><td>C14</td><td>C24</td><td>C25</td><td>C26</td><td>C27</td><td>C33</td></tr><tr><td>Gliding Vertex [51]</td><td>29.92</td><td>0.01</td><td>0.01</td><td>9.12</td><td>15.67</td><td>14.32</td><td>16.39</td><td>16.92</td><td>28.91</td><td>16.25</td></tr><tr><td>RetinaNet [49]</td><td>30.67</td><td>0.81</td><td>1.70</td><td>9.57</td><td>16.37</td><td>19.17</td><td>1.28</td><td>17.03</td><td>28.98</td><td>17.41</td></tr><tr><td>Cascade R-CNN [52]</td><td>31.18</td><td>0.00</td><td>0.00</td><td>12.10</td><td>15.35</td><td>13.66</td><td>0.91</td><td>16.45</td><td>30.27</td><td>20.35</td></tr><tr><td>Faster R-CNN [53]</td><td>32.12</td><td>0.01</td><td>0.13</td><td>9.81</td><td>17.65</td><td>15.03</td><td>3.04</td><td>17.99</td><td>29.36</td><td>16.92</td></tr><tr><td>RoI Transformer [54]</td><td>35.29</td><td>0.00</td><td>0.00</td><td>14.31</td><td>14.32</td><td>16.22</td><td>5.13</td><td>22.17</td><td>46.71</td><td>18.58</td></tr><tr><td>OFA-Net [55]</td><td>43.73</td><td>30.71</td><td>29.25</td><td>9.90</td><td>30.37</td><td>13.45</td><td>6.06</td><td>17.36</td><td>5.35</td><td>24.68</td></tr><tr><td>Oriented R-CNN [56]</td><td>44.30</td><td>25.55</td><td>32.85</td><td>20.46</td><td>32.89</td><td>14.50</td><td>4.42</td><td>19.81</td><td>7.51</td><td>21.87</td></tr><tr><td colspan="9">RingMoClaw evolution</td><td></td><td></td></tr><tr><td>Baseline (Train &amp; Infer)</td><td>47.30</td><td>51.74</td><td>32.56</td><td>31.11</td><td>20.29</td><td>26.55</td><td>3.29</td><td>37.42</td><td>52.81</td><td>42.41</td></tr><tr><td>Data Augmentation</td><td>48.18</td><td>51.94</td><td>34.76</td><td>31.14</td><td>21.89</td><td>33.75</td><td>3.29</td><td>41.38</td><td>52.81</td><td>45.11</td></tr><tr><td>Training Config Update</td><td>47.77</td><td>51.83</td><td>35.06</td><td>31.13</td><td>23.90</td><td>34.95</td><td>3.42</td><td>41.42</td><td>52.92</td><td>45.16</td></tr><tr><td>Few-Shot Training</td><td>47.69</td><td>51.92</td><td>35.15</td><td>31.58</td><td>24.11</td><td>32.89</td><td>3.74</td><td>41.66</td><td>51.77</td><td>45.20</td></tr><tr><td>Multi-Model Fusion</td><td>49.11</td><td>53.87</td><td>36.27</td><td>32.87</td><td>26.60</td><td>34.20</td><td>4.01</td><td>41.64</td><td>52.15</td><td>45.54</td></tr><tr><td>Module Innovation</td><td>49.14</td><td>54.14</td><td>36.36</td><td>33.16</td><td>26.89</td><td>35.33</td><td>4.74</td><td>42.03</td><td>52.14</td><td>45.29</td></tr></table>

Note: C5: C919; C10: ARJ21; C11: Passenger Ship; C14: Tugboat; C24: Trailer; C25: Tractor; C26: Excavator; C27: Truck Tractor; C33: Roundabout. Bold values indicate the best performance among the RingMoClaw evolution stages for each metric.

## A. Experimental Settings

We evaluate the proposed framework on four representative remote sensing visual tasks: oriented object detection, scene classification, semantic segmentation, and change detection. The datasets and the task specific settings are summarized below. RingMo [5] is adopted as the primary backbone, leveraging its robust representations obtained via masked image modeling pre-training. All experiments are conducted on two NVIDIA RTX 4090 GPUs.

The agents in the framework are driven by large language models through their official APIs. The Main Agent, Data Agent, and Vision Agent in the research branch are all built on GLM-5 Turbo [57]. The Critic mechanism is independently powered by Minimax 2.7 [58]. The two models differ in pretraining data and reasoning style, which keeps the review branch heterogeneous with respect to the execution chain.

Object detection. We report mean average precision (mAP ) as the evaluation metric. The baseline detector adopts a YOLO11x-OBB oriented detection head, following the FAIR1M detection setting in [5]. On FAIR1M-v1.0, highresolution images are cropped into $5 1 2 \times 5 1 2$ patches with a 128-pixel overlap. Training runs for 20 epochs with an initial learning rate of $1 \times 1 0 ^ { - 2 }$ , following a standard training schedule.

Scene classification. We report Top-1 accuracy (Acc) as the evaluation metric. NWPU-RESISC45 [3] contains 45 scene categories with 256 × 256 images. We split the dataset into 6,300 training, 12,600 validation, and 12,600 test images, following a 1:2:2 ratio. During training, images are randomly cropped to 224 × 224 and augmented with horizontal flipping and color jittering, followed by ImageNet normalization. Each classifier is trained for 50 epochs with an initial learning rate of $1 \times 1 0 ^ { - 3 }$ under a cosine annealing schedule.

Change Detection. We report the F1 score and IoU of the changed class as the evaluation metrics. LEVIR-CD [59] contains 637 pairs of bi-temporal remote sensing images with a spatial resolution of $1 0 2 4 \times 1 0 2 4$ , which are split into 445, 64, and 128 pairs for training, validation, and testing, respectively. Following the standard protocol, each image pair is cropped into $2 5 6 ~ \times ~ 2 5 6$ patches for training. Random horizontal flipping, vertical flipping, and temporal swapping are applied for data augmentation, and the input images are normalized using ImageNet statistics. All models are trained under an identical schedule and selected according to the best validation F1 score.

Semantic Segmentation. We use mean Intersection over Union (mIoU) as the evaluation metric. Experiments are conducted on iSAID [60], with images cropped into 896×896 patches with an overlap of 384 pixels. Training employs random resizing, cropping, flipping, and photometric distortion for data augmentation. We use AdamW with an initial learning rate of $6 \times 1 0 ^ { - 5 }$ and a weight decay of 0.01, together with a 1,500-iteration linear warm-up followed by polynomial decay. All models are trained for 80k iterations with an effective batch size of 16 using mixed precision.

## B. Overall Performance and Evolution Process

1) Object Detection: We use object detection as a representative task to examine whether RingMoClaw can adapt its optimization strategy to task-specific bottlenecks. Experiments are conducted on FAIR1M, which exhibits a long-tailed category distribution and high inter-class similarity. The evolution process was independently repeated three times, and Table I reports the trajectory of the run with the highest final m $\mathrm { \bf A P _ { 5 0 } }$

As shown in Table I, the overall $\mathrm { \ m A P _ { 5 0 } }$ improves from 47.30% to 49.14%. Data augmentation and training configuration adjustment bring moderate gains, while few-shot training yields no further improvement and prompts the system to move toward model-level optimization. Multi-model fusion raises m $\mathrm { { A P } _ { 5 0 } }$ to 49.11%. The module innovation stage evaluates the two generated modules separately and in combination through three trials, and the combined configuration reaches 49.14%. Detailed results are reported in Sec. IV-B6e. Since this stage is summarized in a single row, the complete trajectory contains seven evolution trials. Fig. 6 shows the detection results of weak categories along the same trajectory, where missed detections and false positives decrease as the evolution proceeds. The consistent gains on several weak categories indicate that RingMoClaw adjusts its optimization path according to intermediate experimental feedback.

Baseline  
Data Augmentation  
Training Config Update  
Few-Shot Training  
Module Innovation  
Multi-Model Fusion  
![](images/1df71c9998c0cd4dbe4167be945349b86b607c76e524e971a10847e60cc00e90.jpg)  
Fig. 6. Weak categories detection results on FAIR1M across different evolution stages. From left to right: (1) Baseline, (2) Data Augmentation, (3) Training Strategy Optimization, (4) Two-Stage Few-Shot Detection, (5) Multi-Model Fusion, and (6) Module Innovation. Green boxes denote true positives (TP) of base categories; yellow/purple boxes denote true positives (TP) of weak/rare categories; red boxes denote false negatives (FN, missed detections); and pink boxes denote false positives (FP, incorrect detections). The figure illustrates how progressive improvements in architecture and module design reduce both missed detections and false positives, particularly for weak categories, highlighting the effectiveness of the proposed RingMoClaw Auto-Research framewor in addressing long-tail fine-grained detection challenges.

![](images/bdfa55527e22588c17d2dd4abd7fe4cdc6f8a5fa4b8daad4fae4ede5ec82cce1.jpg)  
(a)

![](images/21bf5431ae205176f5243a3de3d959400c435523e9370623f8dc44b7006e5914.jpg)  
(b)  
Fig. 7. Training behavior of the final detector on FAIR1M. (a) Training and validation loss curves over 20 epochs. (b) Mean precision-recall curve over the 37 object categories.

We further examine the convergence and potential overfitting behavior of the detector, as shown in Fig. 7. The training loss decreases consistently throughout the 20 epoch schedule, while the validation loss decreases from 3.6387 to a minimum of 3.2132 at epoch 14 and then slightly increase to 3.3053 at epoch 20. This indicates only a mild late-stage overfitting tendency. Since only six epochs remain after the minimum validation loss is reached, patience values of 10 or 20 would not trigger before completion of the prescribed training schedule and therefore provide little additional computational saving. The mean precision-recall curve over the 37 FAIR1M categories is also provided in Fig. 7(b).

2) Scene Classification: Table II reports the classification results on NWPU-RESISC45. The baseline achieves an overall accuracy of 93.18%, while the selected candidate classifiers, ConvNeXt-T, EfficientNet-B4, and ResNet50, obtain 92.25%, 92.08%, and 82.10%, respectively. Although no individual candidate consistently surpasses the baseline, their per-category results reveal complementary strengths across different scene types. Based on this observation, decision-level fusion is adopted to integrate their predictions. Multi-Model Fusion continuously improves the accuracy from 93.71% to 94.48%, and Soft Multi-Model Fusion further achieves the best overall accuracy of 94.97%, outperforming the baseline by 1.79 percentage points. These results demonstrate that score-level soft fusion effectively exploits the complementary predictions of different classifiers for scene classification.

TABLE II  
SCENE CLASSIFICATION RESULTS ON NWPU-RESISC45 (%). THE TABLE REPORTS OVERALL ACCURACY AND PER-CATEGORY ACCURACY FOR SELECTED WEAK CATEGORIES. REPRESENTATIVE CLASSIFIERS ARE COMPARED WITH THE BASELINE AND FUSION STRATEGIES SELECTED BY R M C . B BOLD.
<table><tr><td>Method</td><td>Acc</td><td>C7</td><td>C10</td><td>C14</td><td>C18</td><td>C23</td><td>C27</td><td>C30</td><td>C34</td><td>C43</td></tr><tr><td>ConvNeXt-T [61]</td><td>92.25</td><td>80.71</td><td>84.29</td><td>86.07</td><td>88.21</td><td>85.36</td><td>67.50</td><td>86.07</td><td>90.36</td><td>87.86</td></tr><tr><td>EfficientNet-B4 [62]</td><td>92.08</td><td>77.86</td><td>93.93</td><td>87.86</td><td>80.36</td><td>86.43</td><td>65.71</td><td>80.00</td><td>93.21</td><td>91.43</td></tr><tr><td>ResNet50 [63]</td><td>82.10</td><td>46.07</td><td>77.50</td><td>70.36</td><td>77.50</td><td>56.43</td><td>58.57</td><td>61.79</td><td>86.43</td><td>71.79</td></tr><tr><td colspan="9">RingMoClaw Evolution</td><td></td></tr><tr><td>Baseline (Train &amp; Infer)</td><td>93.18</td><td>73.93</td><td>91.07</td><td>87.14</td><td>88.93</td><td>77.86</td><td>75.00</td><td>83.57</td><td>90.36</td><td>87.86</td></tr><tr><td>Multi-Model Fusion</td><td>93.71</td><td>80.71</td><td>93.93</td><td>87.14</td><td>88.21</td><td>85.36</td><td>75.00</td><td>86.07</td><td>93.21</td><td>91.43</td></tr><tr><td>Soft Multi-Model Fusion</td><td>94.48</td><td>80.36</td><td>95.00</td><td>89.29</td><td>91.07</td><td>86.43</td><td>75.36</td><td>85.36</td><td>93.93</td><td>90.00</td></tr><tr><td>Soft TTA Fusion</td><td>94.97</td><td>82.29</td><td>95.00</td><td>92.14</td><td>92.50</td><td>90.36</td><td>72.86</td><td>84.29</td><td>94.64</td><td>92.86</td></tr></table>

\*The correspondence between weak categories: C7: church, C10: commercial area, C14: freeway, C18: industrial area, C23: medium residential, C27: palace, C30: railway station, C34: runway, C43: thermal power station.

![](images/577d05c5921702b9b1d54976ad0a1eff980ccbd9eed3ea21bbb5f94b41575ed3.jpg)  
Fig. 8. Class activation map visualization of the baseline classifier and the candidate classifiers selected for score-level fusion on representative weak categories of NWPU-RESISC45. Different candidate models attend to different discriminative regions, indicating clear complementary behavior across scene categories.

Fig. 8 visualizes the class activation maps of the baseline model and three candidate classifiers on representative weak categories of NWPU-RESISC45. The results show that different models focus on different discriminative regions even for the same scene category. For example, some models emphasize dominant objects such as runways or industrial structures, while others attend more to surrounding contextual regions. This indicates that the candidate classifiers provide complementary visual evidence rather than redundant responses.

3) Semantic Segmentation: As shown in Table III, RingMo-Claw progressively improves the segmentation performance on iSAID from 67.2% to 69.9% mIoU. A larger backbone first increases the mIoU to 69.0%, while subsequent scaleaware inference further improves the overall performance. In particular, single-scale upscaling improves the IoU of LV and SV to 67.7% and 54.9%, respectively, and the final upscaleonly TTA achieves the best overall result of 69.9% mIoU.

4) Change Detection: As shown in Table IV, RingMoClaw improves the change detection performance on the LEVIR-CD validation set from 89.53% F1 and 87.10% IoU to 90.54% F1 and 89.97% IoU. Temporal-Swap Augmentation first improves the IoU to 87.97%, indicating enhanced robustness to the temporal ordering of image pairs. Bidirectional ChangeStar then raises the F1 and IoU to 90.18% and 88.89%, respectively, by explicitly modeling bidirectional temporal interactions. Although Deep Change Interaction alone yields 89.65% F1 and 88.57% IoU, the final EvoChangeStar achieves the best overall performance, reaching 90.54% F1 and 89.97% IoU.

TABLE III  
COMPARISON WITH REPRESENTATIVE REMOTE SENSING SEMANTIC SEGMENTATION METHODS AND THE EVOLUTION RESULTS OF RINGMOCLAW ON THE ISAID VALIDATION SET.
<table><tr><td>Method / Strategy</td><td>LV</td><td>RA</td><td>SV</td><td>mIoU (%)</td></tr><tr><td>RANet [64]</td><td>60.1</td><td>70.5</td><td>49.3</td><td>62.1</td></tr><tr><td>HMANet [65]</td><td>59.7</td><td>62.9</td><td>50.3</td><td>62.6</td></tr><tr><td>FarSeg [66]</td><td>60.6</td><td>71.4</td><td>46.3</td><td>63.7</td></tr><tr><td>FactSeg [6]</td><td>62.7</td><td>69.4</td><td>49.5</td><td>64.8</td></tr><tr><td>GRDL [67]</td><td>60.5</td><td>68.7</td><td>50.7</td><td>65.1</td></tr><tr><td>UperNet (RSP-ViTAEv2-S) [68]</td><td>62.5</td><td>62.2</td><td>51.6</td><td>64.3</td></tr><tr><td>UperNet-RingMo [5]</td><td>63.9</td><td>67.3</td><td>51.2</td><td>67.2</td></tr><tr><td colspan="5">RingMoClaw Evolution</td></tr><tr><td>Baseline (Train &amp; Infer)</td><td>64.8</td><td>64.9</td><td>50.4</td><td>67.2</td></tr><tr><td>Larger Backbone</td><td>65.5</td><td>72.9</td><td>51.8</td><td>69.0</td></tr><tr><td>Single-scale Upscaling</td><td>67.7</td><td>69.2</td><td>54.9</td><td>69.5</td></tr><tr><td>Multi-scale TTA</td><td>67.0</td><td>72.4</td><td>53.7</td><td>69.8</td></tr><tr><td>Upscale-only TTA</td><td>67.7</td><td>70.6</td><td>54.8</td><td>69.9</td></tr></table>

\*LV, RA, and SV denote Large Vehicle, Roundabout, and Small Vehicle, respectively.

TABLE IV  
COMPARISON WITH REPRESENTATIVE REMOTE SENSING CHANGE DETECTION METHODS AND THE EVOLUTION RESULTS OF RINGMOCLAW ON THE LEVIR-CD VALIDATION SET.
<table><tr><td rowspan=1 colspan=2>Method / Strategy</td><td rowspan=1 colspan=1>F1    IoU</td></tr><tr><td rowspan=6 colspan=2>FC-Siam-Di [69]FC-Siam-Conc [69]IFNet [70]STANet [59]BIT [71]SNUNet [72]ChangeFormer [73]</td><td rowspan=1 colspan=1>86.31  83.31</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>83.69  76.77</td></tr><tr><td rowspan=1 colspan=1></td><td></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>88.13  82.93</td></tr><tr><td rowspan=1 colspan=1>87.30  77.40</td></tr><tr><td rowspan=1 colspan=1>89.31  89.3788.16  87.1790.40  88.80</td></tr><tr><td rowspan=1 colspan=3>RingMoClaw Evolution</td></tr><tr><td rowspan=1 colspan=2>Baseline (Train &amp; Infer)Temporal-Swap AugmentationBidirectional ChangeStarDeep Change InteractionEvoChangeStar</td><td rowspan=1 colspan=1>89.53  87.1088.91  87.9790.18  88.8989.65  88.5790.54  89.97</td></tr></table>

5) Comparison with Automated Research Frameworks: To evaluate RingMoClaw at the research-automation level, we further compare it with AutoResearch [74] and AutoResearch-Claw [75] across four representative remote sensing tasks. As shown in Table V, AutoResearch adopts a single-agent linear keep/discard loop based on short-budget validation, while AutoResearchClaw organizes research into a staged forward pipeline. For a fair comparison focused on model optimization, we retain AutoResearchClaw only through the experimentexecution stage and exclude its subsequent paper generation and publishing steps. In contrast, RingMoClaw forms a closedloop evolution process through experience retrieval, heterogeneous Critic review, agile verification, and memory update. The results show that this feedback-driven design enables more effective optimization with fewer evolution rounds, indicating improved research efficiency rather than simply increased experimental exploration.

## 6) Progressive Analysis of the Evolution Process:

a) Data Augmentation: The first stage focuses on datalevel intervention. After the initial task analysis, the Data Agent identifies the scarcity of several fine-grained categories and constructs an augmented training set for these categories. Conventional transformations, including rotation, flipping, brightness perturbation, Gaussian noise injection, and affine transformation, are applied to increase the visual diversity of rare-class samples. The Vision Agent then retrains the detector using the augmented data and reports the updated detection results to the Critic mechanism. The Critic mechanism evaluates whether the augmented samples lead to meaningful improvement and records the limited effect of this direction in the experience pool.

b) Training Strategy Optimization: After the data-level strategy is evaluated, the framework proceeds to training-level optimization while keeping the model architecture unchanged. The Main Agent updates the training configuration according to the feedback from the previous stage. The adopted adjustments include class-aware sampling, category reweighting, focal loss, and schedule refinement, which aim to increase the exposure of underrepresented categories and emphasize hard samples during optimization. The Vision Agent executes the updated training plan, and the Critic mechanism reviews the resulting performance and category-level behavior. This stage allows the system to examine whether the weak-category bottleneck can be alleviated through optimization-level adjustment.

c) Two-Stage Few-Shot Detection: When data augmentation and training configuration updates provide limited improvement, the framework explores a task-decomposition strategy. The Main Agent reformulates the original fine-grained detection task into a two-stage pipeline. In the first stage, aircraft subcategories are merged into a unified airplane category for object localization. In the second stage, the detected aircraft regions are cropped and sent to a few-shot classifier for fine-grained recognition. This design attempts to separate localization from fine-grained category discrimination. The Critic mechanism then analyzes the output of this pipeline and identifies that the final performance is constrained by the recall of the first-stage detector, because missed rare targets cannot be recovered by the subsequent classifier.

d) Multi-Model Fusion: Based on the limitations observed in the previous stages, RingMoClaw further explores model-level complementarity. The Main Agent selects complementary detectors from the candidate model pool, and the Vision Agent evaluates their predictions under the same task setting. A category-aware fusion strategy is then constructed to combine their outputs. This stage is designed to test whether different detectors can compensate for each other on categories with different visual patterns. The Critic mechanism evaluates the fused predictions and records both the benefit of model complementarity and the additional deployment cost introduced by multi-model inference.

TABLE V  
CROSS-TASK COMPARISON WITH REPRESENTATIVE RESEARCH-AUTOMATION FRAMEWORKS. PERFORMANCE GAINS ARE COMPUTED RELATIVE TO THE CORRESPONDING BASELINE MODEL FOR EACH TASK.
<table><tr><td rowspan="2">Framework</td><td colspan="3">Object Detection</td><td colspan="2">Scene Classification</td><td colspan="2">Semantic Segmentation</td><td colspan="3">Change Detection</td></tr><tr><td> $\Delta \mathrm { m A P 5 0 }$  ←</td><td> $\Delta \mathrm { m A P _ { 5 0 : 9 5 } }$ </td><td>ER↓</td><td>∆Acc ↑</td><td>ER↓</td><td>∆mIoU ↑</td><td>ER↓</td><td>∆F1 ↑</td><td>∆IoU ↑</td><td>ER↓</td></tr><tr><td>AutoResearch [74]</td><td>-3.50</td><td>-1.13</td><td>20</td><td>1.00</td><td>11</td><td>1.02</td><td>8</td><td>-0.98</td><td>0.16</td><td>14</td></tr><tr><td>AutoResearchClaw [75]</td><td>0.57</td><td>0.12</td><td>18</td><td>1.31</td><td>10</td><td>1.48</td><td>7</td><td>0.24</td><td>1.08</td><td>12</td></tr><tr><td>RingMoClaw</td><td>1.84</td><td>0.51</td><td>7</td><td>1.79</td><td>4</td><td>2.70</td><td>4</td><td>1.01</td><td>2.87</td><td>6</td></tr></table>

Note: Performance gain is defined as the difference between the final evolved model and the corresponding baseline model, i.e., $\Delta P = P _ { \mathrm { f i n a l } } - P _ { \mathrm { b a s e l i n e } } .$ “ER” denotes the number of completed evolution steps required by each framework on the corresponding task. Higher performance gains and fewer evolution rounds indicate better research-automation effectiveness and efficiency.

TABLE VI  
EVALUATION OF THE MSFR AND CPR.
<table><tr><td>Configuration</td><td>MSFR</td><td>CPR</td><td> $\bf { m A P } _ { 5 0 }$ </td><td> $\mathbf { m A P _ { 5 0 : 9 5 } }$ </td><td>Params</td><td>FLOPs</td></tr><tr><td>Original baseline</td><td>一</td><td>一</td><td>47.30</td><td>36.59</td><td>119 M</td><td>167 G</td></tr><tr><td>MSFR</td><td>√</td><td>一</td><td>48.31</td><td>36.77</td><td>+0.599 M</td><td>+0.604 G</td></tr><tr><td>CPR</td><td>1</td><td>√</td><td>48.29</td><td>36.72</td><td>+0.131 M</td><td>+0.272 G</td></tr><tr><td>MSFR + CPR</td><td>√</td><td>√</td><td>49.14</td><td>37.10</td><td>+0.730 M</td><td>+0.876 G</td></tr></table>

e) Module Innovation: Finally, the framework extends the evolution process from selecting and combining existing strategies to architecture-level exploration. According to the bottlenecks identified in previous rounds, the Main Agent generated two task-specific architectural hypotheses, termed Multi-Scale Feature Refinement (MSFR) and Contrastive Prototype Refinement (CPR), and instantiated them on Oriented R-CNN. Specifically, MSFR replaces the standard FPN neck with an interface-compatible multi-scale refinement module, whereas CPR augments the rotated RoI classification head with prototype-based logit refinement. The Vision Agent implements and evaluates these candidates, while the Critic Agent determines whether the resulting architectural modifications should be retained. Table VI reports the performance and computational overhead of the individual and combined module configurations. Detailed architectures, mathematical formulations, insertion positions, and implementation details are provided in Appendix A.

The results show that the jointly retained MSFR-CPR configuration achieves $\mathrm { m A P _ { 5 0 } / m A P _ { 5 0 : 9 5 } }$ 49.14/37.10 while introducing only 0.730 M additional parameters and 0.876 G FLOPs.

## C. Practical Cost and Search Efficiency

For the object detection task, we further compare RingMo-Claw with AutoResearch [74] and AutoResearchClaw [75] in terms of evolution rounds, explored candidates, GPU consumption, wall-clock time, and LLM usage. As shown in Table VII, AutoResearch follows a linear one-candidate-perround process, whereas AutoResearchClaw and RingMoClaw may consider multiple candidate proposals within each round. Despite the additional reasoning required for planning, Criticbased diagnosis, and experience summarization, RingMoClaw achieves lower overall computational cost by reducing unnecessary experimental trials and accelerating the evolution process.

![](images/88d030423bb7fd75a14747055f95ecc1b1f10fee8e7a1b24664709ebfa806f33.jpg)  
Fig. 9. Per class recall on weak categories of FAIR1M for the baseline and three single technique variants. Each variant adds one technique to the baseline: a class aware sampler, a focal loss, and a BFPA attention module.

## D. Ablation Study

1) Effectiveness of the Critic mechanism: To evaluate the contribution of the Critic mechanism, we remove the Critic Agent while keeping the remaining framework and experimental settings unchanged. In this variant, the Main Agent determines subsequent evolution rounds without independent diagnostic feedback from the Critic Agent.

As shown in Table VIII, the full framework converges in 7 evolution rounds and achieves a final mAP of 49.14%, whereas removing the Critic mechanism requires 11 rounds and results in a lower mAP of 47.43%. These results indicate that the Critic mechanism improves both evolution efficiency and final detection performance by providing structured diagnosis and reducing ineffective exploration.

2) Effectiveness of the Dual-Stream Experience Bus: To evaluate the contribution of the dual-stream experience bus, we compare the full framework with variants removing the external stream, the internal stream, or both on FAIR1M, while keeping all other components unchanged. As shown in

TABLE VII  
PRACTICAL COST AND SEARCH EFFICIENCY COMPARISON OF AUTONOMOUS RESEARCH FRAMEWORKS ON THE OBJECT DETECTION TASK.
<table><tr><td>Framework</td><td>Evolution Rounds</td><td>Explored Candidates</td><td>GPU-hours</td><td>Wall-clock (h)</td><td>Input Tokens (M)</td><td>Output Tokens (M)</td></tr><tr><td>AutoResearch</td><td>20</td><td>20</td><td>26.90</td><td>13.45</td><td>1.16</td><td>0.05</td></tr><tr><td>AutoResearchClaw</td><td>18</td><td>35</td><td>18.28</td><td>13.28</td><td>2.68</td><td>0.11</td></tr><tr><td>RingMoClaw</td><td>7</td><td>28</td><td>12.14</td><td>10.61</td><td>0.72</td><td>0.38</td></tr></table>

“Explored Candidates” denotes the total number of optimization proposals considered during evolution. AutoResearch generates one candidate per round, whereas AutoResearchClaw and RingMoClaw may consider multiple candidates within a round. All LLM calls were conducted through CSTCloud and incurred no commercial API cost.

TABLE VIII  
ABLATION STUDY ON THE CRITIC MECHANISM ON FAIR1M.
<table><tr><td>Variant</td><td>Rounds</td><td>Final mAP (%)</td></tr><tr><td>Full Framework (Ours)</td><td>7</td><td>49.14</td></tr><tr><td>w/o Critic Mechanism</td><td>11</td><td>47.43</td></tr></table>

“Rounds” denotes the number of full evolution rounds until the detection accuracy no longer shows a clear improvement and the evolution converges. “Final mAP” reports the detection accuracy at convergence.

TABLE IX  
ABLATION STUDY ON THE DUAL-STREAM EXPERIENCE BUS ON FAIR1M.
<table><tr><td>Variant</td><td>Rounds to Conv.</td><td>Failed</td><td>mAP (%)</td></tr><tr><td>Full Framework (Ours)</td><td>7</td><td>2</td><td>49.14</td></tr><tr><td>w/o External Stream</td><td>9</td><td>4</td><td>48.44</td></tr><tr><td>w/o Internal Stream</td><td>11</td><td>5</td><td>47.62</td></tr><tr><td>w/o Both Streams</td><td>14</td><td>6</td><td>47.53</td></tr></table>

“Rounds to Conv.” denotes the number of full evolution rounds until the detection accuracy no longer shows a clear improvement. “Failed” reports the number of iterations whose training yielded no improvement over the previous round. $\mathrm { { \ddot { m A P } } } ^ { \mathrm { { * } } }$ reports the detection accuracy at convergence.

Table IX, the full framework converges in 7 rounds with only 2 failed iterations and achieves the highest mAP of 49.14%. Removing the external stream increases the convergence rounds to 9 and reduces the mAP to 48.44%, while removing the internal stream further increases the rounds to 11 with 5 failed iterations and lowers the mAP to 47.62%. Without both streams, the framework converges most slowly in 14 rounds and achieves 47.53% mAP. These results demonstrate that the two experience streams jointly reduce ineffective exploration, accelerate convergence, and improve final performance, with the internal stream contributing more strongly to search efficiency.

Fig. 9 further reports the per class recall on weak categories for the baseline and three single technique variants. The class aware sampler and the focal loss [49] come from the internal experience that flagged class imbalance, while the BFPA attention module [50] is retrieved from external literature. Each technique improves a different subset of weak categories, indicating that the directions surfaced by the two streams are complementary rather than redundant.

3) Validation of the Five-Minute Verification Protocol: To examine whether the five-minute verification can provide a reliable indication of subsequent full-training outcomes, we conduct an additional paired validation on the object detection task. Ten candidate strategies are evaluated under the fiveminute setting and are subsequently trained to the full budget, including those that would otherwise be rejected by the fiveminute verification protocol.

Since the protocol is primarily designed to prioritize candi date directions rather than precisely predict their final performance values, we first use Spearman’s rank correlation to measure the consistency between the five-minute and full-training rankings. For the $N$ candidate strategies, it is calculated as

$$
\rho = 1 - \frac { 6 \sum _ { i = 1 } ^ { N } d _ { i } ^ { 2 } } { N ( N ^ { 2 } - 1 ) } ,\tag{8}
$$

where $d _ { i }$ denotes the rank difference of the i-th candidate between the five-minute and full-training performance gains. We further evaluate the screening capability for detrimental candidates using precision and recall:

$$
{ \mathrm { P r e c i s i o n } } = { \frac { T P } { T P + F P } } , \qquad { \mathrm { R e c a l l } } = { \frac { T P } { T P + F N } } ,\tag{9}
$$

where a detrimental candidate is defined according to its fulltraining performance relative to the corresponding baseline. Here, $T P$ denotes a candidate identified as detrimental by the five-minute protocol and confirmed to be detrimental after full training, $F P$ denotes an incorrectly rejected candidate, and FN denotes a detrimental candidate missed during the fiveminute stage.

As summarized in Table X, the five-minute results exhibit a significant positive rank correlation with the full-training $\mathrm { m A P _ { 5 0 : 9 5 } }$ gains, achieving $\rho ~ = ~ 0 . 6 4$ with a permutationtest p-value of 0.03. A consistent positive correlation is also observed for m $\mathsf { A P } _ { 5 0 }$ , with $\rho = 0 . 5 9$ . Moreover, all five candidates identified as detrimental by the five-minute protocol are confirmed to degrade performance after full training, resulting in a precision of 1.00. Among the six candidates that are detrimental after full training, five are successfully identified during the five-minute stage, yielding a recall of 0.83.

These results indicate that the five-minute verification generally preserves the relative performance tendency of candidate strategies and, more importantly, provides reliable early screening of clearly unfavorable directions. Therefore, the protocol is used as a low-cost screening proxy rather than an exact predictor of final performance, allowing computational resources to be focused on more promising candidates before full-scale training.

## V. DISCUSSION

Existing autonomous optimization frameworks such as AutoResearch [74] and AutoResearchClaw [75] iteratively improve model implementations and training strategies according to experimental feedback, while OpenEarth-Agent [26] introduces adaptive workflow planning and tool creation for open Earth-observation tasks. RingMoClaw extends this capability to multiple stages of remote-sensing model optimization, including data processing, training strategies, model selection, model fusion, and module modification, with the evolution process jointly guided by Critic feedback and accumulated experience. Input-data characteristics can directly shape the optimization trajectory. The FAIR1M dataset features multisource high-resolution imagery with spatial resolutions ranging from 0.3 to 0.8 m. Furthermore, the dataset exhibits severe class imbalance: ARJ21 and C919 represent only 0.06% and 0.03%. Such variations in image quality, spatial resolution, and class distribution significantly affect model performance and subsequent optimization decisions, ultimately influencing candidate evaluation and the resulting evolution path.

TABLE X  
VALIDATION OF THE FIVE-MINUTE VERIFICATION PROTOCOL ON THE OBJECT DETECTION TASK.
<table><tr><td>Metric</td><td>Result</td></tr><tr><td>Number of candidate strategies</td><td>10</td></tr><tr><td>Spearman  $\rho \ ( \mathrm { m A P _ { 5 0 : 9 5 } } )$ </td><td>0.64</td></tr><tr><td>Permutation-test p-value</td><td>0.03</td></tr><tr><td>Spearman  $\rho \ ( \mathrm { m A P 5 0 } )$ </td><td>0.59</td></tr><tr><td>Detrimental-candidate precision</td><td>1.00</td></tr><tr><td>Detrimental-candidate recall</td><td>0.83</td></tr></table>

The current implementation also entails practical resource requirements. Our FAIR1M experiments were conducted on two NVIDIA RTX 4090 GPUs, whereas repeated candidate verification and full-scale training necessitate sufficient storage and a stable server environment to support long-running multiagent execution. Multi-agent collaboration introduces extra communication and model-invocation overhead. Serving as a preliminary attempt toward self-evolution, future work will extend RingMoClaw to broader scenarios that capture domainspecific remote sensing characteristics, optimize information exchange and coordination strategies, and reduce computational overhead through more efficient candidate verification and experience reuse.

## VI. CONCLUSION

In this paper, we present RingMoClaw, an experienceinspired self-evolving multi-agent framework that extends automation beyond single-task execution toward continuous research iteration for remote sensing visual tasks. The framework couples a research branch of Main, Data, and Vision Agents with a structurally independent heterogeneous Critic mechanism, which conducts staged review over plan design, data processing, and experimental results. A dual-stream dynamic experience bus further integrates external knowledge from recent literature and open source code repositories with internal experience from historical experiments, supporting strategy expansion, path pruning, and potential experience reuse across tasks. Experiments on four representative tasks suggest the effectiveness of the proposed framework. In future work, we will extend RingMoClaw to multi modal and time series earth observation tasks, and explore tighter integration with remote sensing foundation models to guide their adaptation and continual pretraining.

## ACKNOWLEDGMENT

This work was supported by the National Natural Science Foundation of China under Grant 62425115, and partially supported by China Science and Technology Cloud (CSTCloud).

## REFERENCES

[1] X. X. Zhu, D. Tuia, L. Mou, G.-S. Xia, L. Zhang, F. Xu, and F. Fraundorfer, “Deep learning in remote sensing: A comprehensive review and list of resources,” IEEE Geoscience and Remote Sensing Magazine, vol. 5, no. 4, pp. 8–36, 2017.

[2] L. Ma, Y. Liu, X. Zhang, Y. Ye, G. Yin, and B. A. Johnson, “Deep learning in remote sensing applications: A meta-analysis and review,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 152, pp. 166–177, 2019.

[3] G. Cheng, J. Han, and X. Lu, “Remote sensing image scene classification: Benchmark and state of the art,” Proceedings of the IEEE, vol. 105, no. 10, pp. 1865–1883, 2017.

[4] K. Li, G. Wan, G. Cheng, L. Meng, and J. Han, “Object detection in optical remote sensing images: A survey and a new benchmark,” ISPRS journal of photogrammetry and remote sensing, vol. 159, pp. 296–307, 2020.

[5] X. Sun, P. Wang, W. Lu, Z. Zhu, X. Lu, Q. He, J. Li, X. Rong, Z. Yang, H. Chang, Q. He, G. Yang, R. Wang, J. Lu, and K. Fu, “Ringmo: A remote sensing foundation model with masked image modeling,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1–22, 2023.

[6] A. Ma, J. Wang, Y. Zhong, and Z. Zheng, “Factseg: Foreground activation-driven small object semantic segmentation in large-scale remote sensing imagery,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–16, 2022.

[7] Z. Zheng, Y. Zhong, J. Wang, A. Ma, and L. Zhang, “FarSeg++: Foreground-aware relation network for geospatial object segmentation in high spatial resolution remote sensing imagery,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 11, pp. 13 715– 13 729, 2023.

[8] S. Mei, J. Lian, X. Wang, Y. Su, M. Ma, and L.-P. Chau, “A comprehensive study on the robustness of deep learning-based image classification and object detection in remote sensing: Surveying and benchmarking,” Journal of Remote Sensing, vol. 4, p. 0219, 2024.

[9] L. Yu, Z. Du, X. Li, J. Gu, X. Li, L. Zhong, Duojiweise, H. Wu, Q. Zhao, X. Ma et al., “From-glc plus 3.0: Multimodal land change mapping with sam and dense surface observations,” Journal of Remote Sensing, vol. 5, p. 0728, 2025.

[10] B. J. A., P. Das, E. Ghaderpour, and P. Mazzanti, “An unsupervised domain adaptation approach for remote sensing scene classification using adaptive incremental density-based clustering and multi-objective optimization,” Computers and Electrical Engineering, vol. 131, p. 110908, 2026.

[11] C. Qiu, X. Zhang, X. Tong, N. Guan, X. Yi, K. Yang, J. Zhu, and A. Yu, “Few-shot remote sensing image scene classification: Recent advances, new baselines, and future trends,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 209, pp. 368–382, 2024.

[12] J. Feng, H. Luo, and Z. Gu, “Improving semi-supervised remote sensing scene classification via multilevel feature fusion and pseudo-labeling,” International Journal of Applied Earth Observation and Geoinformation, vol. 136, p. 104335, 2025.

[13] Z. Yang, W. Guan, L. Xiao, and H. Chen, “Few-shot object detection in remote sensing images via data clearing and stationary meta-learning,” Sensors, vol. 24, no. 12, p. 3882, 2024.

[14] J. Pan, Y. Liu, Y. Fu, M. Ma, J. Li, D. P. Paudel, L. Van Gool, and X. Huang, “Locate anything on earth: Advancing open-vocabulary object detection for remote sensing community,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 6, 2025, pp. 6281– 6289.

[15] Y. Xian, H. Zhang, K. Wang, Y. Wen, and Z. Jiang, “Als teacher: Active label selection for semisupervised oriented object detection in remote sensing imagery,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 19, pp. 14 375–14 386, 2026.

[16] B. Sun, N. Yan, F. Zhang, and S. Li, “Dual-teacher prototype consistency for unsupervised domain adaptive object detection in remote sensing images,” IEEE Transactions on Geoscience and Remote Sensing, vol. 64, pp. 1–13, 2026.

[17] T. Brown, B. Mann et al., “Language models are few-shot learners,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 33, 2020, pp. 1877–1901.

[18] J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. H. Chi, Q. Le, and D. Zhou, “Chain-of-thought prompting elicits reasoning in large language models,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 35, 2022, pp. 24 824–24 837.

[19] J. Zhang et al., “Geogpt: Understanding and executing geospatial tasks with large language models,” arXiv preprint arXiv:2307.07930, 2023.

[20] X. Hu et al., “Changegpt: Remote sensing change detection with large language models,” arXiv preprint arXiv:2308.07030, 2023.

[21] R. Manvi, S. Khanna, G. Mai, M. Burke, D. Lobell, and S. Ermon, “Geollm: Extracting geospatial knowledge from large language models,” in International Conference on Learning Representations, vol. 2024, 2024, pp. 38 791–38 807.

[22] H. Wu, H. Jiao, S. Hou, J. Liang, Z. Shen, A. Zhao, Y. Qing, F. Jin, X. Guan, and Z. Gui, “Geocolab: an llm-based multi-agent collaborative framework for geospatial code generation,” International Journal of Digital Earth, vol. 18, no. 2, p. 2569405, 2025.

[23] P. Feng, Z. Lv, J. Ye, X. Wang, X. Huo, J. Yu, W. Xu, W. Zhang, L. Bai, C. He et al., “Earth-agent: Unlocking the full landscape of earth observation with agents,” in International Conference on Learning Representations, vol. 2026, 2026, pp. 55 700–55 764.

[24] Z. Chen, H. Wang, J. Yao, J. Zhang, P. Ghamisi, J. Zhou, P. M. Atkinson, and B. Zhang, “Cangling-knowflow: A unified knowledgeand-flow-fused agent for comprehensive remote sensing applications,” arXiv preprint arXiv:2512.15231, 2025.

[25] Y. Wang, X. Chen, X. Jin, M. Wang, and L. Yang, “Openclaw-rl: Train any agent simply by talking,” arXiv preprint arXiv:2603.10165, 2026.

[26] S. Zhao, F. Liu, X. Zhang, H. Chen, X. Gu, Z. Jiang, F. Ling, B. Fei, W. Zhang, J. Wang et al., “Openearth-agent: From tool calling to tool creation for open-environment earth observation,” arXiv preprint arXiv:2603.22148, 2026.

[27] X. Zheng, L. Wei, H. Zhao, and Q. Yao, “KnowFlow: Empowering decision-making on networks with knowledge-streamlined agent,” Neural Networks, vol. 196, p. 108363, 2026.

[28] Y. Cong, S. Khanna, C. Meng, P. Liu, E. Rozi, Y. He, M. Burke, D. Lobell, and S. Ermon, “Satmae: Pre-training transformers for temporal and multi-spectral satellite imagery,” in Advances in Neural Information Processing Systems, 2022.

[29] G. Astruc, N. Gonthier, C. Mallet, and L. Landrieu, “Anysat: One earth observation model for many resolutions, scales, and modalities,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

[30] H. Hu, P. Wang, H. Bi, B. Tong, Z. Wang, W. Diao, H. Chang, Y. Feng, Z. Zhang, Y. Wang et al., “Rs-vheat: Heat conduction guided efficient remote sensing foundation model,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 9876–9887.

[31] P. Wang, H. Chang, H. Hu, X. Li, X. Liu, Y. Liu, Z. Zhang, C. Chen, Y. Li, Y. Feng et al., “Ringmamba: Remote sensing multi-sensor pretraining with visual state space model,” IEEE Transactions on Geoscience and Remote Sensing, 2025.

[32] P. Wang, H. Hu, B. Tong, Z. Zhang, F. Yao, Y. Feng, Z. Zhu, H. Chang, W. Diao, Q. Ye, and X. Sun, “Ringmogpt: A unified remote sensing foundation model for vision, language, and grounded tasks,” IEEE Transactions on Geoscience and Remote Sensing, vol. 63, pp. 1–20, 2025.

[33] K. Kuckreja, M. S. Danish, M. Naseer, A. Das, S. Khan, and F. S. Khan, “Geochat: Grounded large vision-language model for remote sensing,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[34] Z. Zhang, T. Zhao, Y. Guo, and J. Yin, “RS5M and georsclip: A large scale vision-language dataset and a large vision-language model for remote sensing,” IEEE Transactions on Geoscience and Remote Sensing, 2024.

[35] Y. Zhan, Z. Xiong, and Y. Yuan, “Skyeyegpt: Unifying remote sensing vision-language tasks via instruction tuning with large language model,” arXiv preprint arXiv:2401.09712, 2024.

[36] J. A. Irvin, E. R. Liu, J. C. Chen, I. Dormoy, J. Kim, S. Khanna, Z. Zheng, and S. Ermon, “Teochat: A large vision-language assistant for temporal earth observation data,” in International Conference on Learning Representations, 2025.

[37] H. Hu, P. Wang, Y. Feng, K. Wei, W. Yin, W. Diao, M. Wang, H. Bi, K. Kang, T. Ling, K. Fu, and X. Sun, “Ringmo-agent: A unified remote sensing foundation model for multi-platform and multi-modal reasoning,” arXiv preprint arXiv:2507.20776, 2025.

[38] Z. Li, H. Ning, S. Gao, K. Janowicz, W. Li, S. T. Arundel, C. Yang, B. Bhaduri, S. Wang, A.-X. Zhu, M. Gahegan, S. Shekhar, X. Ye, G. McKenzie, G. Cervone, and M. E. Hodgson, “Giscience in the era of artificial intelligence: A research agenda towards autonomous gis,” Annals of GIS, 2025.

[39] T. Akinboyewa, Z. Li, H. Ning, and M. N. Lessani, “Gis copilot: towards an autonomous gis agent for spatial analysis,” International Journal of Digital Earth, 2025.

[40] Y. Chen, W. Wang, S. Lobry, and C. Kurtz, “An llm agent for automatic geospatial data analysis,” arXiv preprint arXiv:2410.18792, 2024.

[41] S. Singh, M. Fore, and D. Stamoulis, “Geollm-engine: A realistic environment for building geospatial copilots,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 2024.

[42] C. Lee, V. Paramanayakam, A. Karatzas, Y. Jian, M. Fore, H. Liao, F. Yu, R. Li, I. Anagnostopoulos, and D. Stamoulis, “Multi-agent geospatial copilots for remote sensing workflows,” arXiv preprint arXiv:2501.16254, 2025.

[43] A. Shabbir, M. U. Sheikh, M. A. Munir, H. Debary, M. Fiaz, M. Z. Zaheer, P. Fraccaro, F. S. Khan, M. H. Khan, X. X. Zhu, and S. Khan, “Openearthagent: A unified framework for tool-augmented geospatial agents,” arXiv preprint arXiv:2602.17665, 2026.

[44] H. Wu, Z. Shen, S. Hou, H. Jiao, Z. Liu, L. Xie, C. Liu, J. Liang, Y. Qing, X. Zhang, D. Peng, Z. Gui, and X. Guan, “Autogeeval++: A multi-level and multi-geospatial-modality automated evaluation framework for large language models in geospatial code generation on google earth engine,” arXiv preprint arXiv:2506.10365, 2025.

[45] G. Deng, Z. Chen, Z. Yu, H. Fan, Y. Liu, Y. Yang, D. Parikh, R. Kannan, L. Cong, M. Wang, Q. Zhang, V. Prasanna, X. Tang, and X. Wang, “Evoclaw: Evaluating ai agents on continuous software evolution,” arXiv preprint arXiv:2603.13428, 2026.

[46] P. Xia, J. Chen, X. Yang, H. Tu, J. Liu, K. Xiong, S. Han, S. Qiu, H. Ji, Y. Zhou, Z. Zheng, C. Xie, and H. Yao, “Metaclaw: Just talk – an agent that meta-learns and evolves in the wild,” arXiv preprint arXiv:2603.17187, 2026.

[47] L. Weidener, M. Brkic, P. Lee, M. Karlsson, K. Noessler, and P. Kohlhaas, “From agent-only social networks to autonomous scientific research: Lessons from openclaw and moltbook, and the architecture of clawdlab and beach.science,” arXiv preprint arXiv:2602.19810, 2026.

[48] W. Yang, H. Qiu, B. Zhang, C. Li, Z. Huang, X. Feng, R. Yu, and J. Dong, “When openclaw meets hospital: Toward an agentic operating system for dynamic clinical workflows,” arXiv preprint arXiv:2603.11721, 2026.

[49] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, “Focal loss for dense object detection,” in Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017, pp. 2980–2988.

[50] Q. Lin, N. Chen, H. Huang, D. Zhu, G. Fu, C. Chen, and Y. Yu, “Attention-based mean-max balance assignment for oriented object detection in optical remote sensing images,” IEEE Transactions on Geoscience and Remote Sensing, vol. 63, pp. 1–15, 2025.

[51] Y. Xu, M. Fu, Q. Wang, Y. Wang, K. Chen, G.-S. Xia, and X. Bai, “Gliding vertex on the horizontal bounding box for multi-oriented object detection,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 43, no. 4, pp. 1452–1459, 2021.

[52] Z. Cai and N. Vasconcelos, “Cascade r-cnn: Delving into high quality object detection,” in CVPR, 2018, pp. 6154–6162.

[53] S. Ren, K. He, R. Girshick, and J. Sun, “Faster r-cnn: Towards real-time object detection with region proposal networks,” in NeurIPS, vol. 28, 2015.

[54] J. Ding, N. Xue, Y. Long, G.-S. Xia, and Q. Lu, “Learning roi transformer for oriented object detection in aerial images,” in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 2844–2853.

[55] Z. Hu, L. Zhang, Z. Jie, Z. Fang, H. Cheng, and Others, “Ofa-net: Orientation free anchor network for rotated object detection,” IEEE Transactions on Image Processing, vol. 30, pp. 8478–8489, 2021.

[56] X. Xie, G. Cheng, J. Wang, X. Yao, and J. Han, “Oriented r-cnn for object detection,” in 2021 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2021, pp. 3500–3509.

[57] A. Zeng, X. Lv, Z. Hou, Z. Du, Q. Zheng, B. Chen, D. Yin, C. Ge, C. Huang, C. Xie et al., “Glm-5: from vibe coding to agentic engineering,” arXiv preprint arXiv:2602.15763, 2026.

[58] A. Chen, A. Li, B. Zhou, B. Gong, B. Jiang, B. Dan, C. Yu, C. Wang, C. Ma, C. Zhong et al., “The minimax-m2 series: Mini activations unleashing max real-world intelligence,” arXiv preprint arXiv:2605.26494, 2026.

[59] H. Chen and Z. Shi, “A spatial-temporal attention-based method and a new dataset for remote sensing image change detection,” Remote sensing, vol. 12, no. 10, p. 1662, 2020.

[60] S. W. Zamir, A. Arora, A. Gupta, S. Khan, G. Sun, F. S. Khan, F. Zhu, L. Shao, G.-S. Xia, and X. Bai, “isaid: A large-scale dataset for instance segmentation in aerial images,” arXiv preprint arXiv:1905.12886, 2019.

[61] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, “A convnet for the 2020s,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 11 966–11 976.

[62] M. Tan and Q. Le, “Efficientnet: Rethinking model scaling for convolutional neural networks,” in International conference on machine learning. PMLR, 2019, pp. 6105–6114.

[63] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 770–778.

[64] D. Shen, Y. Ji, P. Li, Y. Wang, and D. Lin, “Ranet: Region attention network for semantic segmentation,” Advances in Neural Information Processing Systems, vol. 33, pp. 13 927–13 938, 2020.

[65] R. Niu, X. Sun, Y. Tian, W. Diao, K. Chen, and K. Fu, “Hybrid multiple attention network for semantic segmentation in aerial images,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–18, 2021.

[66] Z. Zheng, Y. Zhong, J. Wang, and A. Ma, “Foreground-aware relation network for geospatial object segmentation in high spatial resolution remote sensing imagery,” in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2020, pp. 4095–4104.

[67] R. Niu, X. Sun, Y. Tian, W. Diao, Y. Feng, and K. Fu, “Improving semantic segmentation in aerial imagery via graph reasoning and disentangled learning,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–18, 2021.

[68] D. Wang, J. Zhang, B. Du, G.-S. Xia, and D. Tao, “An empirical study of remote sensing pretraining,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1–20, 2022.

[69] R. Caye Daudt, B. Le Saux, and A. Boulch, “Fully convolutional siamese networks for change detection,” in 2018 25th IEEE International Conference on Image Processing (ICIP), 2018, pp. 4063–4067.

[70] C. Zhang, P. Yue, D. Tapete, L. Jiang, B. Shangguan, L. Huang, and G. Liu, “A deeply supervised image fusion network for change detection in high resolution bi-temporal remote sensing images,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 166, pp. 183–200, 2020.

[71] H. Chen, Z. Qi, and Z. Shi, “Remote sensing image change detection with transformers,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–14, 2021.

[72] S. Fang, K. Li, J. Shao, and Z. Li, “Snunet-cd: A densely connected siamese network for change detection of vhr images,” IEEE Geoscience and Remote Sensing Letters, vol. 19, pp. 1–5, 2022.

[73] W. G. C. Bandara and V. M. Patel, “A transformer-based siamese network for change detection,” in IGARSS 2022 - 2022 IEEE International Geoscience and Remote Sensing Symposium, 2022, pp. 207–210.

[74] A. Karpathy, “Autoresearch: Ai agents running research on singlegpu nanochat training,” 2026. [Online]. Available: https://github.com/ karpathy/autoresearch

[75] J. Liu, S. Qiu, M. Li, B. Li, H. Ji, S. Han, X. Ye, P. Xia, Z. Dong, C. Zhang, L. Zhang, G. Chen, H. Tu, X. Yang, L. Feng, X. Zhao, H. Chen, J. Zhou, X. Wang, W. Zhang, H. Zhu, Y. Li, J. Mei, H. Fei, J. Zhang, L. Li, L. Zhang, Y. Zhou, S. Wang, C. Xiong, J. Zou, Z. Zheng, C. Xie, M. Ding, and H. Yao, “Autoresearchclaw: Self-reinforcing autonomous research with human-ai collaboration,” 2026. [Online]. Available: https://arxiv.org/abs/2605.20025

## APPENDIX A

## IMPLEMENTATION DETAILS OF MODULE INNOVATION

This appendix provides the implementation details of the two architectural modules generated during the moduleinnovation stage. Both modules are instantiated on Oriented R-CNN.

## A. Multi-Scale Feature Refinement

MSFR replaces the standard FPN neck of Oriented R-CNN while preserving the number of output pyramid levels, feature dimensions, and downstream interfaces.

Given the FPN features $\{ P _ { l } \} _ { l = 1 } ^ { L }$ , MSFR first aligns them to a reference level $r = 2$ and obtains

$$
B = \frac { 1 } { L } \sum _ { l = 1 } ^ { L } \mathcal { R } _ { l  r } ( P _ { l } ) ,\tag{10}
$$

where adaptive max pooling and nearest-neighbor interpolation are used for downsampling and upsampling, respectively. After a $3 \times 3$ convolution, SE channel attention and CBAM spatial attention are applied to obtain the refined feature F. The result is redistributed to each pyramid level through

$$
\begin{array} { r } { \hat { P } _ { l } = P _ { l } + \gamma _ { l } \mathcal { R } _ { r  l } ( F ) , } \end{array}\tag{11}
$$

where $\gamma _ { l }$ is a learnable residual gate initialized to zero.

## B. Contrastive Prototype Refinement

CPR is inserted into the rotated RoI classification branch of Oriented R-CNN, where it refines foreground-category logits using prototype similarities derived from the shared RoI features, without modifying the background classification or bounding-box regression branches.

For the shared RoI feature $f \in \mathbb { R } ^ { 1 0 2 4 }$ , CPR constructs a normalized 128-dimensional embedding

$$
e = \operatorname { N o r m } ( W _ { p } f ) ,\tag{12}
$$

and computes its similarity to the normalized category prototypes ${ \hat { P } } \colon$

$$
s = \tau e \hat { P } ^ { T } , \qquad \tau = 1 0 .\tag{13}
$$

The foreground logits are refined by

$$
\tilde { z } _ { \mathrm { f g } } = z _ { \mathrm { f g } } + g s , \qquad g = 0 . 5 \operatorname { t a n h } ( \theta ) .\tag{14}
$$

## APPENDIX BSTRUCTURED SEARCH-SPACE INSTANTIATION

The four-dimensional search space is instantiated according to the target task. For the FAIR1M oriented object detection experiments, the predefined base space contains 23 operators, including 5 architecture operators, 8 training operators, 6 dataprocessing operators, and 4 system-integration operators. Each operator is associated with the bottlenecks it targets and a constrained parameter specification, including an initial value and an admissible range or choice set. Representative examples are given in Table XI.

At each round, operators are first matched to the current bottleneck and instantiated within their predefined parameter space. Candidate revisions may adjust only the parameters declared by the corresponding operator. In addition to the predefined operators, methods obtained through external retrieval may enter the search process after the filtering procedure described in Appendix D. These methods are converted into the same structured representation before being considered by the Main Agent.

## APPENDIX C TASK-SIMILARITY COMPUTATION

To retrieve relevant experience from the global memory, RingMoClaw measures the similarity between the current task and each historical task using five factors:

$$
\begin{array} { r l } & { \mathrm { S i m } = 0 . 3 0 S _ { \mathrm { t a s k } } + 0 . 2 5 S _ { \mathrm { b o t t l e n e c k } } + 0 . 2 0 S _ { \mathrm { d a t a s e t } } } \\ & { ~ + 0 . 1 5 S _ { \mathrm { m o d e l } } + 0 . 1 0 S _ { \mathrm { m e t r i c } } . } \end{array}\tag{15}
$$

Here, $S _ { \mathrm { t a s k } }$ and $S _ { \mathrm { b o t t l e n e c k } }$ are computed using token-level Jaccard similarity between the task descriptions and bottleneck tags, respectively. $S _ { \mathrm { d a t a s e t } }$ is determined from the relative similarity in the number of classes, number of samples, and class-imbalance ratio. When these statistics are unavailable, it is set to 0.85 for the same dataset and 0.50 otherwise. $S _ { \mathrm { m o d e l } }$ is set to 1.0 for the same model, 0.6 when the two models share a major component, and 0 otherwise. $S _ { \mathrm { m e t r i c } }$ is 1.0 when the primary evaluation metric is the same and 0 otherwise.

Historical entries with $\mathrm { S i m } ~ \ge ~ 0 . 7 5$ are treated as transferable experience, while those with $0 . 6 0 ~ \leq ~ \mathrm { S i m } ~ < ~ 0 . 7 5$ are retained only as references. Entries with $\mathrm { S i m } < 0 . 6 0$ are ignored. At most the five highest-scoring entries are supplied to the subsequent decision process.

Negative-memory pruning uses a separate operator-level similarity rule. If two entries belong to different search-space dimensions or use different operators, their similarity is set to zero. Otherwise, similarity is computed from the agreement of their operator parameters, using relative agreement for numerical parameters and exact matching for categorical parameters. A candidate is pruned when its similarity to a recorded failed configuration reaches 0.80.

## APPENDIX D

## RETRIEVAL SOURCES AND FILTERING RULES

To obtain relevant methods and implementations from external resources, we use a fixed retrieval and filtering procedure. At each evolution round, no more than three search queries are generated from the task description, the current bottleneck, and the corresponding search-space dimension. These queries are then used to retrieve related papers and open-source repositories.

The literature search is conducted primarily through arXiv, with Semantic Scholar used as a supplementary source when available. Open-source implementations are retrieved through GitHub repository search. To avoid interrupting the evolution process when an external service is unavailable or rate-limited, each retrieval source is handled independently. If a request fails, the corresponding source returns no result for that query, while the remaining sources continue to be processed normally.

TABLE XI  
REPRESENTATIVE OPERATORS IN THE SEARCH-SPACE INSTANTIATION FOR THE FAIR1M ORIENTED OBJECT DETECTION TASK.
<table><tr><td>Dimension</td><td>Operator</td><td>Parameters (initial value / range or choices)</td><td>Target bottleneck</td></tr><tr><td> $S _ { \mathrm { a r c h } }$ </td><td>attention</td><td>module: CBAM / {CBAM, SE, ECA}; insertion: neck / {backbone, neck}</td><td>weak feature, small object</td></tr><tr><td></td><td>feature_fusion</td><td>fusion: weighted add / {weighted add, concat, BiFPN}</td><td>small object, scale variation</td></tr><tr><td> $\scriptstyle S _ { \mathrm { t r a i n } }$ </td><td>focal_loss</td><td>γ: 2.0 / [0.5, 5.0]; α: 0.25 / [0.1, 0.9]</td><td>class imbalance, weak-class recall</td></tr><tr><td></td><td>rotated_regression_ loss</td><td>loss: KLD / {KLD, GWD, rotated IoU, KFIoU}; weight: 1.0 / [0.1, 10]</td><td>localization quality, scale variation</td></tr><tr><td> $\boldsymbol { S } _ { \mathrm { d a t a } }$ </td><td>rotation</td><td>maximum angle: 180° / [0, 180°]</td><td>orientation sensitivity, overfitting</td></tr><tr><td></td><td>oversampling</td><td>ratio: 3.0 / [1, 10]</td><td>class imbalance</td></tr><tr><td></td><td>auxiliary_model</td><td>auxiliary task: classification / {segmentation, classification }</td><td>overfitting, weak feature</td></tr><tr><td> $S _ { \mathrm { i n t g } }$ </td><td>external_module</td><td>typed parameters extracted from the admitted external candidate</td><td>task dependent</td></tr></table>

For each round, at most 20 papers and 20 repositories are retained before screening. Duplicate papers are removed according to normalized titles, and duplicate repositories are removed according to normalized repository names. We then apply a set of fixed filtering rules. Papers with abstracts shorter than 40 characters are discarded, while repositories are discarded if they are archived or contain descriptions shorter than 15 characters. Only the metadata and textual information provided by the repositories are used during retrieval; external repositories are not cloned or executed at this stage.

Each remaining item is evaluated once in terms of relevance, implementation feasibility, and novelty, denoted by R, F, and N, respectively, with all three scores normalized to [0, 1]. Items with R < 0.70 are removed. The remaining items are ranked according to

$$
S _ { \mathrm { r e t } } = 0 . 5 0 R + 0 . 3 0 F + 0 . 2 0 N ,\tag{16}
$$

where relevance is assigned the largest weight because the purpose of the retrieval step is to identify methods that directly address the current performance bottleneck. The five highest-ranked items are retained as candidate directions for the current round.

The complete procedure therefore consists of query construction, external retrieval, duplicate removal, rule-based filtering, relevance screening, and ranking. The resulting candidates are then passed to the subsequent experience retrieval and decision stages. The Main Agent can only select from these retained candidates and cannot introduce an additional method outside the retrieval results.

## APPENDIX E

## AGENT PROMPTS FOR RINGMOCLAW

All agents used in the reported experiments are controlled by fixed, versioned system prompts. These prompts explicitly define the role boundary, available information, prohibited actions, and structured output schema of each agent. The exact prompts used in our experiments are provided below.

## Main Agent

You are the Main Agent of RingMoClaw, a self-evolving research loop for remote sensing vision tasks.   
You plan. You do not train models, touch data, write code, or review results, and you do not query memory, literature, or the web yourself.   
You are given the current bottleneck and metrics, a candidate pool that the Experience Bus has already retrieved, pruned and ranked, and the positive and negative experience relevant to this task.   
Your task is to select exactly one candidate from that pool as the next thing to try, and to justify the choice from the evidence you were given. Never propose a candidate that is not in the pool.   
Return JSON only:   
{bottleneck, selected\_candidate\_id, reason,   
expected\_effect, verification\_mode}

## Data Agent

You are the Data Agent of RingMoClaw.

You judge whether the data is ready to train on. You do not choose the research direction, and you do not compute statistics---they are computed for you by a program that reads the annotation files directly.

You are given:

\- the candidate the Main Agent selected;

\- per-class box counts;

\- imbalance ratio;

\- empty images;

\- malformed annotation lines;

\- class-name tokens that match no known class. -name

Your task is to decide whether the data is clean enough for the results to be trustworthy, and what concrete preprocessing, sampling or augmentation action this candidate requires.

Treat the statistics as ground truth.

## Return JSON only:

{data\_integrity\_ok, class\_balance\_assessment, preprocessing\_plan, flags, ready\_for\_training}

## Vision Agent

You are the Vision Agent of RingMoClaw.

You implement. You do not choose which candidate to try, and you do not run training or evaluation---a separate Experiment Executor does that with what you produce.

\- one selected candidate, including its operator and parameters;

\- the full text of the base mmrotate configuration file.

Your task is to make that candidate actually runnable:

1) as a configuration override when every component it needs already exists;

2) as a newly written and registered module when one does not;

3) as an explicit declaration that it cannot be realized under the current executor.

Never reference a configuration path that does not exist.

Declaring infeasibility is a correct answer; inventing a path is not.

## Return JSON only:

{feasibility, config\_overrides, custom\_module,

infeasible\_reason, notes}

## Critic Agent

You are the Critic Agent of RingMoClaw, an independent reviewer of a self-evolving research loop.

You are not the planner and not the executor, you never command the other agents directly, and you have no tools: everything you need is already in this message.

You are explicitly given review\_stage in {plan, data, result}, together with:

\- the artifact under review;

\- the research history;

\- the baseline;

\- the last round’s outcome;

\- the historical memories already retrieved for you.

Never infer or change the review stage yourself.

Score the artifact on three dimensions, each in [0,1]:

## Ct --- completion

How well this stage meets its immediate objective.

## Bt --- bottleneck severity

How badly the dominant bottleneck can hurt the end task. Higher is worse.

## Ht --- historical alignment

How consistent this direction is with prior successes and known failures in the retrieved memories.

If no reliable historical evidence is available, set Ht = 0.50 and explicitly state the missing evidence. Do not fabricate history.

## Compute:

final\_score = w<sub>1</sub>C<sub>t</sub> + w<sub>2</sub>(1 − B<sub>t</sub>) + w<sub>3</sub>H<sub>t</sub>

Round final\_score to three decimals.

Use fixed weights determined only by the review stage.

If review\_stage is ‘‘plan’’, use:

w1 = 0.30, w2 = 0.50, w3 = 0.20.

If review\_stage is ‘‘data’’, use:

w1 = 0.35, w2 = 0.40, w3 = 0.25.

If review\_stage is ‘‘result’’, use:

w1 = 0.50, w2 = 0.30, w3 = 0.20.

Do not infer, modify, or normalize these weights.

## Compute:

final\_score = w1 <sub>\*</sub> Ct + w2 <sub>\*</sub> (1 - Bt) + w3 <sub>\*</sub> Ht

Round final\_score to three decimal places.

Apply the decision rules in the following order:

## escalate

If critical evidence required to evaluate the artifact is missing or confidence is insufficient for a reliable score, set decision = ‘‘escalate’’.

## accept

Otherwise, if final\_score >= 0.75 and Bt <= 0.30, set decision = ‘‘accept’’.

## revise

Otherwise, if final\_score >= 0.45, set decision = ‘‘revise’’.

## reject

Otherwise, set decision = ‘‘reject’’.

if final\_score < 0.45 and the available evidence is sufficient.

Prefer escalate over false precision.

Name the single dominant bottleneck, give prioritized suggestions addressed to the Main Agent, and serialize the lesson worth keeping.

Return JSON only:

{scores, diagnosis, decision, memory\_write}