# WareFly-VLA: A Vision-Language-Action Framework for UAV Navigation and Human Tracking in Smart Warehouses<sup>⋆</sup>

Thinh D. Le<sup>a,1</sup>, Son T. Nguyen<sup>a,1</sup>, Duong Q. Nguyen<sup>a</sup>, Dung D. Le<sup>a,b</sup>, Ngo Anh Vien<sup>a,b,c</sup>, H. Nguyen-Xuan<sup>a,b,∗</sup>

<sup>a</sup>Center for AI Research, VinUniversity, Ho Chi Minh City, Vietnam <sup>b</sup>College of Engineering and Computer Science, VinUniversity, Hanoi, Vietnam <sup>c</sup>VinRobotics, Hanoi, Vietnam

## Abstract

Vision–Language–Action (VLA) models have achieved impressive results in robotic manipulation and groundmobile navigation, yet language-conditioned control of unmanned aerial vehicles (UAVs) in smart warehouses remains largely unexplored. Progress is hindered by the lack of benchmarks that jointly provide continuous low-level flight actions, fine-grained natural-language target descriptions and realistic industrial environments. This paper introduces WareFly-VLA, a photorealistic UAV VLA framework and dataset for languageguided human search, localization and tracking in intra-logistics and warehouse environments. The dataset contains 507 human-teleoperated flight episodes and 8,504 non-terminal high-resolution RGB transitions collected in NVIDIA Isaac Sim; each transition is paired with a human-written appearance description of the target worker and a synchronized four-degree-of-freedom control command comprising translational and yaw changes. Two aerial embodied tasks are covered, target approach and person following, under realistic challenges including occlusion, long-range search, altitude variation and clutter. A unified benchmark of four representative open-source VLA architectures (SmolVLA, GR00T N1.7, π and OpenVLA) is established under a leakage-free episode-level protocol at two control rates. The results show that language-conditioned aerial control in warehouses remains far from solved: performance drops substantially under strict generalization settings, continuous action modeling consistently outperforms discrete action tokenization, only the forward channel is reliably learnable from a single frame, and current foundation-model interfaces transfer poorly from ground and humanoid embodiments to aerial platforms. The synchronized video, language, action, pose and dificulty annotations further support world-model research. The dataset, baselines and evaluation protocol are released to support language-grounded aerial autonomy in smart warehouses.

Keywords: Vision-language-action models, Unmanned aerial vehicles, Person following, Warehouse robotics, Language-conditioned control, Imitation learning, Dataset and benchmark

## 1. Introduction

A long-standing objective of embodied artificial intelligence is to enable robots to interpret naturallanguage instructions, ground them in visual observations and execute appropriate low-level actions in dynamic environments. Recent Vision–Language–Action (VLA) models such as PaLM-E [1], RT-1 [2], RT-2 [3], Octo [4], OpenVLA [5], RDT-1B [6], π<sub>0.5</sub> [7], SmolVLA [8], TinyVLA [9] and Qwen-VLA [10] have demonstrated that large-scale multimodal training can produce policies that generalize across tasks, environments and robot embodiments. These advances suggest that a unified perception–language–control paradigm may provide a scalable foundation for future autonomous systems.

Despite rapid progress, contemporary VLA research remains heavily concentrated on robotic manipulation and ground-mobile platforms. Large-scale datasets and benchmarks predominantly focus on tabletop manipulation, mobile navigation or humanoid interaction, whereas aerial robots remain significantly underrepresented, even though vision-based control of multi-rotor aerial vehicles is a mature and active research area [11]. This gap is particularly notable because unmanned aerial vehicles (UAVs) present a distinct embodiment with substantially diferent sensing, motion and control characteristics. Being diferent from manipulators that operate within constrained workspaces or ground robots that navigate primarily in two dimensions, UAVs perform continuous motion in three-dimensional space while simultaneously regulating position, altitude and heading. Visual observations are highly dynamic due to rapid viewpoint changes, target scale variation and frequent occlusions. Consequently, transferring VLA paradigms developed for ground or manipulation settings to aerial systems is far from straightforward.

Among the many tasks relevant to aerial autonomy, language-guided human search and following represents a particularly compelling benchmark for embodied intelligence. Consider the instruction: “Track the worker wearing a yellow hard hat and a purple safety vest.” Successfully executing this command requires the integration of multiple capabilities. The UAV must first interpret fine-grained appearance descriptions, identify the correct individual among visually similar distractors, actively search when the target is not immediately visible, maintain target association over time and continuously generate flight-control commands that ensure safe and reliable tracking. In contrast to conventional UAV tracking benchmarks, where the target is initialized through a bounding box or visual template, language-guided following requires semantic target specification and persistent multimodal grounding throughout the entire trajectory. This setting naturally combines visual grounding, embodied language understanding, active perception and continuous control within a single task.

Existing datasets do not adequately support this problem formulation. Classical approaches such as Chen et al. [12], aerial tracking benchmarks such as UAV123 [13], and subsequent UAV tracking datasets[14] provide extensive flight recordings but rely exclusively on visual target initialization without language supervision. Vision-language navigation datasets such as AerialVLN [15], AirNav [16], EmbodiedCity [17], UAV-VLN [18] and GeoText-1652 [19] introduced natural-language instructions but formulated navigation primarily as waypoint prediction or route planning rather than continuous low-level control. In a similar vein, large-scale drone footage datasets exist and are used to extract ground vehicle trajectories for driving research[20]; however, the lack of ergo-centric controls makes them unsuitable for aerial embodied AI. Recent UAV-oriented VLA datasets and benchmarks remain limited in scope. CognitiveDrone [21] focused on simulation-based trajectory generation. UAV-VLA [22] targets mission-level reasoning from aerial imagery and RaceVLA [23] addressed drone-racing policies. Although these resources have advanced aerial embodied learning, none simultaneously provide appearance-grounded language descriptions, continuous flight-control supervision, human-demonstrated trajectories and warehouse-scale person-search-and-follow tasks.

To address this limitation, this paper introduces WareFly-VLA, a Vision–Language–Action dataset for language-guided human search and following in industrial warehouse environments. The dataset is collected from human-teleoperated UAV demonstrations performed in a photorealistic warehouse environment constructed in NVIDIA Isaac Sim. Each trajectory contains synchronized high-resolution RGB observations, fine-grained appearance-based language descriptions and continuous 4-DoF flight actions represented as body-frame motion commands. In contrast to scripted navigation policies or automatically generated expert trajectories, the dataset captures realistic human decision-making behaviors during target search, target acquisition and continuous following. The resulting trajectories naturally include recovery maneuvers, viewpoint adaptation, target re-identification and other behaviors that commonly arise during practical aerial operation.

WareFly-VLA is designed around two complementary embodied tasks: approach-and-locate and personfollowing. The dataset further incorporates diverse warehouse layouts, worker appearances, target distances, altitude changes, partial occlusions and cluttered industrial environments. Structured dificulty annotations enable systematic evaluation across multiple challenge dimensions and support future research on generalization, robustness and language grounding under realistic operating conditions.

Beyond dataset construction, a unified benchmark is established to evaluate representative open-source VLA architectures using a common leakage-free episode-level split. The benchmark spans multiple model families and action-representation paradigms, providing a standardized framework for studying aerial embodied intelligence. Experimental results indicate that language-conditioned UAV control remains a challenging open problem, highlighting substantial opportunities for future advances in multimodal representation learn ing, continuous action prediction and cross-embodiment transfer.

The main contributions of this work are as follows:

1. WareFly-VLA, a human-teleoperated UAV Vision–Language–Action dataset for language-guided human search and following in warehouse environments, containing synchronized visual observations, appearance-grounded language descriptions and continuous 4-DoF flight actions.

2. A human-demonstration data collection pipeline built in NVIDIA Isaac Sim that captures realistic aerial search, target localization, target acquisition and continuously following behaviors from teleoperated quadrotor trajectories.

3. Aerial embodied benchmark tasks covering approach-and-locate and person-following scenarios under realistic challenges, including occlusion, altitude variation, long-range search, viewpoint changes and cluttered industrial environments.

4. A unified aerial VLA benchmark for evaluating representative open-source VLA architectures under a standardized leakage-free evaluation protocol.

5. An open-source research platform including dataset artifacts, benchmark configurations, training pipelines and evaluation tools to facilitate future research on aerial embodied intelligence, languagegrounded UAV autonomy and Vision–Language–Action learning.

6. Open world-model research directions enabled by WareFly-VLA, including predictive representation learning, action-conditioned latent dynamics, language-conditioned planning and sim-to-real transfer for aerial embodied intelligence.

## 2. Related Work

## 2.1. Vision–Language–Action models

VLA models have emerged as a unified paradigm for embodied intelligence by jointly learning perception, language understanding and control. Early eforts such as PaLM-E [1] demonstrated the feasibility of integrating visual observations into large language models for robotic decision making. Subsequent systems including RT-1 [2], RT-2 [3], Open X-Embodiment [24], Octo [4], OpenVLA [5], RDT-1B [6], π<sub>0.5</sub> [7], SmolVLA [8], TinyVLA [9] and Qwen-VLA [10] have demonstrated increasingly strong generalization across tasks, environments and robotic embodiments.

Recent research has further explored large-scale pretraining and cross-embodiment transfer. Open X-Embodiment [24] aggregated demonstrations from multiple robot platforms to investigate shared policy learning, while OpenVLA [5], RDT-1B [6] and π<sub>0</sub> [25] explored foundation-model approaches capable of adapting to diverse downstream tasks. Inference-time policy steering has further been proposed to deploy pre-trained VLAs on new tasks without fine-tuning [26]. Despite these advances, the overwhelming majority of VLA benchmarks remain centered on robotic manipulation, mobile navigation and humanoid control. The applicability of VLA models to aerial robots remains comparatively underexplored, largely due to the scarcity of UAV-specific datasets that combine language instructions with continuous flight-control supervision.

## 2.2. Language-grounded aerial navigation

Language-guided navigation has become an important benchmark for embodied intelligence, motivating a growing body of research on aerial vision-language systems. AerialVLN [15] introduced a large-scale simulation benchmark for instruction-following UAV navigation. Subsequent eforts expanded this direction through more realistic environments and richer navigation objectives, including UAV-VLN [18], EmbodiedCity [17], GeoText-1652 [19], AirNav [16] and semantic-topometric aerial navigation frameworks [27].

Additional studies have investigated aerial dialogue navigation and interactive instruction following [28], and vision-language policies for onboard drone navigation have been trained in radiance-field simulators [29]. In parallel, learning-based vision-driven following behaviors for UAVs, such as safe river following via reinforcement learning with a semantic dynamics model [30], show that end-to-end visual control of aerial platforms is feasible, but they do not involve language-specified targets.

These datasets have significantly advanced aerial language grounding and instruction-conditioned navigation. However, most methods formulate navigation as waypoint prediction, route planning or localization rather than direct low-level control. As a result, they provide limited supervision for end-to-end VLA learning, where language instructions must be translated directly into continuous flight actions.

## 2.3. UAV VLA datasets and benchmarks

Recent studies have begun extending the Vision–Language–Action paradigm to aerial robotics. UAV-VLA [22] investigates mission-level planning and reasoning from aerial imagery, while RaceVLA [23] explores language-conditioned drone-racing policies derived from OpenVLA. CognitiveDrone [21] combines VLA policies with reasoning-oriented VLMs for simulation-based cognitive UAV navigation. Broader surveys have also highlighted the growing role of foundation models and LLMs in UAV autonomy [31].

Although these eforts demonstrate the promise of foundation-model-based aerial autonomy, important limitations remain. Existing benchmarks are typically restricted to racing, navigation, mission planning or simulation-specific tasks. Many datasets rely on automatically generated trajectories or task-specific objectives that do not require semantic target grounding. Furthermore, appearance-based human search and following has received little attention within the UAV VLA literature.

Consequently, there remains no publicly available benchmark that simultaneously provides appearancegrounded natural-language target descriptions, continuous 4-DoF flight-control actions, human-teleoperated UAV demonstrations and language-guided human search-and-follow tasks in realistic warehouse environments. WareFly-VLA is designed to fill this gap and to provide a dedicated benchmark for aerial Vision– Language–Action learning.

## 3. The WareFly-VLA dataset

## 3.1. Dataset design principles

WareFly-VLA is designed around a central objective: enabling research on language-guided human search and following for aerial robots. In contrast to conventional UAV tracking benchmarks that require explicit target initialization via bounding boxes or visual templates, WareFly-VLA specifies the target exclusively through natural-language descriptions of observable appearance attributes. Consequently, successful task completion requires the integration of language grounding, visual identification, active search, target reidentification and continuous flight control.

Three design principles guided the dataset construction process. Firstly, all trajectories are generated from human-teleoperated demonstrations rather than scripted controllers, allowing the dataset to capture realistic search and following behaviors. Secondly, language annotations emphasize appearance-based target grounding instead of spatial instructions, reflecting realistic warehouse deployment scenarios. Thirdly, the dataset focuses on continuous low-level flight actions, making it suitable for Vision–Language–Action learning rather than high-level navigation alone.

## 3.2. Simulation environment

Data collection is implemented in Pegasus Simulator[32], a framework for simulating drone dynamics built as an extension to NVIDIA Isaac Sim[33], a photorealistic and physically based robotics simulation platform. The environment consists of a large industrial warehouse containing shelving racks, conveyor systems, forklift lanes, storage areas and mixed lighting conditions. The scene is intentionally designed to expose the UAV to realistic visual challenges, including clutter, repetitive textures, reflective surfaces, viewpoint changes and partial occlusions. A simulated quadrotor equipped with a forward-facing RGB camera captures observations at 1920 × 1440 pixels. Ground-truth vehicle pose is obtained directly from the simulator state, providing accurate position and orientation measurements throughout each trajectory.

## 3.3. Human-Teleoperated Data Collection

Each episode begins by assigning a worker a unique combination of clothing and personal protective equipment (PPE), including safety vests, hard hats, shirts, overalls, pants, gloves and other accessories. Appearance configurations are intentionally diversified to maximize language grounding diversity and minimize attribute overlap between episodes.

A human operator teleoperates the UAV while attempting to locate, approach and follow the designated target worker. The target follows a predefined route unknown to the operator, requiring continuous visual search and target maintenance throughout the trajectory. This setup naturally produces realistic aerial behaviors, including target acquisition, viewpoint adjustment, temporary target loss, recovery maneuvers and continuous following.

All RGB observations, control inputs and simulator states are recorded synchronously. The resulting trajectories form a collection of human-demonstrated aerial embodied behaviors suitable for Vision–Language– Action learning.

To reduce redundancy while preserving meaningful control dynamics, the recorded streams are temporally subsampled to 1 FPS, consistent with the efective decision frequencies commonly adopted in recent VLA systems [25, 5].

## 3.4. Teleoperation policy

To produce trajectories that are diverse in scene yet consistent in behavior, the human pilot follows a deliberately deterministic search–decide–approach policy. Fig. 1 summarises the policy as a finite-state machine with three search states, one decision, one goal-directed action state and an absorbing terminal state.

Look Around. Every episode begins with the pilot performing a yaw 360<sup>◦</sup> sweep in place. The sweep is a rotation-only maneuver around the drone’s initial position. It enables target acquisition within the camera’s efective field of view at the current altitude without committing the drone to a specific motion direction.

check\_open decision. If the sweep completes without acquiring the target, the pilot consults a binary state-of-environment check. When the drone is currently in an open area (suficient lateral clearance to fly between shelving rows), control proceeds to Scan Shelves. When the drone is in an aisle or back-space (constrained by walls or rack ends), control proceeds to Move to Open.

Scan Shelves. In this state the pilot flies to the edge of the closest shelving row, angles the drone outward by approximately 45<sup>◦</sup> and traverses parallel to the row while looking down each aisle in turn. This pattern systematically sweeps the only volume that the in-place yaw cannot reach the deep interior of shelving aisles and is the dominant source of large lateral ∆y commands in the corpus.

Move to Open. When the drone is hemmed in by infrastructure, the pilot first repositions to a viewpoint with broader lateral clearance, flying toward the nearest open area and continuing to look for an appearancematching target along the way. Once the drone arrives, the state machine loops back to Look Around so that a full sweep can be executed from a more advantageous viewpoint.

Approach Target. As soon as the target appears in the camera frame from any of the three search states, the pilot transitions to the goal-directed Approach Target state, yaws to center the target in the frame, and flies toward it. The episode terminates on Reached Target.

Search-exhaustion loops. A Scan Shelves pass that completes without acquisition raises a Search Exhausted transition back to Look Around and a Move to Open pass that arrives at the new viewpoint without acquisition raises an analogous Arrived at open area transition. The policy therefore guarantees that the drone always returns to a yaw-sweep state between unsuccessful search legs, rather than chaining multiple action states blindly.

Why a deterministic policy. Fixing the high-level behavior while randomizing warehouse layout, target route and target appearance configuration produces a corpus that is behaviourally consistent every episode contains the same acquisition–closure–tracking phases while remaining trajectory-wise diverse, with varied path shapes, distances, durations and occlusion patterns (Fig. 3). This combination is what allows the 507- episode corpus to discriminate sharply between modern VLA architectures despite its modest scale (Sections 5–6) and it directly motivates the dificulty-tag axes used for stratified evaluation (Section 4).

![](images/c1d36cb09635d05fcdec6f2d0be7f4839d298ea52db22c6c91240fa0fef5d909.jpg)  
Figure 1: State-transition diagram of the human teleoperation policy used during data collection. Search states (Look Around, Scan Shelves and Move to Open) are shown in cyan. The check\_open decision node is shown in amber, and the goal-directed Approach Target state is shown in green. From any search state, detecting the target in the camera frame triggers an immediate transition to Approach Target. If Scan Shelves ends with Search Exhausted, or if Move to Open ends with Arrived at open area, the policy returns to Look Around. The episode terminates when the drone reaches the target.

## 3.5. Action representation

Continuous flight actions are derived from consecutive simulator poses. Given position $\mathbf { p } _ { t }$ and heading $\psi _ { t }$ at time t, the action label is represented as a four-dimensional body-frame motion command

$$
\mathbf { a } _ { t } = ( \Delta x , \Delta y , \Delta z , \Delta \psi ) .\tag{1}
$$

The translational part is expressed in the UAV body frame:

$$
{ \left[ \begin{array} { l } { \Delta x } \\ { \Delta y } \end{array} \right] } = R ( - \psi _ { t } ) \left( \mathbf { p } _ { t + 1 } ^ { x y } - \mathbf { p } _ { t } ^ { x y } \right)\tag{2}
$$

while altitude and heading updates are computed as

$$
\Delta z = z _ { t + 1 } - z _ { t } ,\tag{3}
$$

$$
\begin{array} { r } { \Delta { \psi } = \mathrm { w r a p } _ { [ - \pi , \pi ] } ( \psi _ { t + 1 } - \psi _ { t } ) . } \end{array}\tag{4}
$$

This representation is invariant to global orientation and directly compatible with UAV controllers operating in local body coordinates.

## 3.6. Language annotation

Following trajectory collection, each episode is independently annotated by human annotators from the perspective of a remote UAV operator. Instructions describe the target exclusively through observable appearance attributes, including clothing, colors, accessories, and personal protective equipment (PPE), while avoiding identity information or scene-specific positional references. Multiple paraphrases are provided to increase linguistic diversity and improve robustness to instruction variations. All annotations undergo secondary review to ensure factual consistency and language quality, resulting in an appearance-centric language corpus specifically designed for language-guided human search and following.

## 4. Dataset analysis

## 4.1. Quantitative statistics

Table 1 summarizes the key statistics of WareFly-VLA. The dataset contains 507 episodes and 8,504 nonterminal transitions with valid control actions. All frames are captured at a fixed resolution of 1920 × 1440 pixels. This resolution provides high visual fidelity for fine-grained appearance grounding. The trajectories exhibit substantial drone motion across both short-range maneuvers and long-distance approach episodes. They also show a large variation in pursuit distance and motion pace, reflecting diverse target routes and human teleoperation strategies.

Fig. 2 visualizes the episode-length distribution, the per-episode frame-count profile and the task-verb breakdown. Fig. 3 shows representative drone trajectories in the world-frame xy-plane together with a top-down composite over the warehouse scene, illustrating the variety of flight paths across episodes and confirming that no two episodes share identical dynamics.

## 4.2. Action space characterization

The action at each time step is a 4-DoF vector $\mathbf { a } _ { t } = ( \Delta x _ { \ell } , \Delta y _ { \ell } , \Delta z , \Delta \psi )$ expressed in the drone’s local body frame. Table 2 reports the marginal statistics of each component across all non-terminal frames $( N = 8 , 5 0 4 )$

Four observations characterize the action distribution and motivate later design decisions. Firstly, $\Delta x _ { \ell }$ is strongly positive on average $( 0 . 8 6 \pm 0 . 6 9 \mathrm { m / f r a m e } )$ , reflecting the dominant forward-pursuit behavior as the drone closes on or maintains distance from the walking target. Secondly, ∆y<sub>ℓ</sub> is near-zero in mean but with a sizeable standard deviation (0.42 m) and a wide range $\left( \left[ - 5 . 1 2 , + 4 . 0 1 \right] \mathrm { ~ m } \right)$ , capturing frequent and sometimes large lateral correction maneuvers as the target deviates from the drone’s heading axis. Thirdly, $\Delta z$ is tightly concentrated around zero (mean −0.000 m, std 0.013 m), confirming near-constant-altitude

![](images/f0a90ee7b88369911987fc1900a79d44fb491c9424ca5d92df633cc7683b533a.jpg)

![](images/759208b45f199451c1cd8dfa8d49cc81ba41d17d3e8c0e38d4c81d4152aa9111.jpg)

## WareFly-VLA Dataset Overview

![](images/fe98928437aa64caaa5cc4ab0867dfcb47dd6a191845f3c1606ac6419e2bcdb0.jpg)  
Figure 2: Dataset overview. (a) Distribution of episode lengths; the dashed red line marks the mean. (b) Per-episode frame count sorted in ascending order. (c) Task-verb distribution across 15 phrasings; approach and follow dominate, with thirteen additional verb families ensuring paraphrase diversity.

![](images/a6a4165c588362f37b08a22918163c280fb4f5e279e8c6b7d9981a910505d407.jpg)

![](images/f365fa1a1ee8be852bb34f3f21299d5cb7c4c98c2855de89da6ec99f89e5f1f1.jpg)

![](images/3d03d5fdb49a1e3950c22967af729a9a2ff6ea0498fadc2400538891d193b2dc.jpg)

![](images/93f8b4902ef51fc50980817659f49279a818e468643a2f9a09a00a61431819b4.jpg)

![](images/59eb63990a230f4e4c775aaa81a89e80040a43d0c47fa40024d41b9d68d67b25.jpg)  
Figure 3: Drone flight trajectories in world coordinates. Left: four representative single-episode trajectories in the world-frame xy-plane, color-coded from early (blue/purple) to late (yellow) via the plasma colormap, with the language instruction shown above each panel; green circles indicate episode start positions and red stars indicate end positions. Right: top-down composite of many episode trajectories overlaid on a bird’s-eye view of the photorealistic Isaac Sim warehouse. Solid line segments denote frames in which the drone has the target in view, while dashed line segments denote frames in which the target is out of view (search / re-acquisition). The composite illustrates the spatial diversity of pursuit paths across shelving rows, aisles and open areas and confirms that the dataset captures a wide range of pursuit behaviors rather than a single stereotyped maneuver.

Action Space Distribution (excluding terminal frames)  
Table 1: WareFly-VLA dataset statistics.
<table><tr><td>Property</td><td>Value</td></tr><tr><td>Total episodes</td><td>507</td></tr><tr><td>Non-terminal transitions</td><td>8,504</td></tr><tr><td>train / val</td><td>7,276 / 1,228</td></tr><tr><td>Train / val episodes</td><td> $4 3 1 ~ / ~ 7 6$ </td></tr><tr><td>Frame rate</td><td> $1 \mathrm { F P S }$ </td></tr><tr><td>Image resolution</td><td> $1 9 2 0 \times 1 4 4 0 ~ \mathrm { p x }$ </td></tr><tr><td>Action dimensionality</td><td>4-DoF  $( \Delta x _ { \ell } , \Delta y _ { \ell } , \Delta z , \Delta \psi )$ </td></tr><tr><td>Task-verb families</td><td>15</td></tr><tr><td>Instruction vocabulary size</td><td>448 tokens</td></tr><tr><td>Mean instruction length</td><td> $2 0 . 0 \pm 5 . 1$  words</td></tr><tr><td>Challenge-tagged episodes yes</td><td> $( \hbar n d / c l o s e / f a r / h e i g h t \_$  change/obstructed)</td></tr></table>

Table 2: Action component statistics over the $N { = } 8 { , } 5 0 4$ non-terminal frames.
<table><tr><td>Component</td><td>Mean</td><td>Std</td><td>Min</td><td>Max</td></tr><tr><td> $\Delta x _ { \ell } \ \mathrm { ( m ) }$ </td><td>+0.864</td><td>0.688</td><td>-0.558</td><td> $+ 5 . 6 1 5$ </td></tr><tr><td> $\Delta y _ { \ell } ~ ( \mathrm { m } )$ </td><td>-0.012</td><td>0.421</td><td>-5.124</td><td> $+ 4 . 0 1 2$ </td></tr><tr><td> $\Delta z \ \mathrm { ( m ) }$ </td><td>-0.000</td><td>0.013</td><td>-0.363</td><td> $+ 0 . 5 6 7$ </td></tr><tr><td> $\Delta \psi \ \mathrm { ( r a d ) }$ </td><td>+0.096</td><td>0.355</td><td>-1.254</td><td> $+ 2 . 1 9 7$ </td></tr></table>

![](images/c92774ec24d72edbd83b9d78200717bd4dd810ef29c59a0b5d23ccc057712d07.jpg)

![](images/8f17be271f8f4fe8aa7bc4723154029b9df001a6f7abd04855b6281447c853cf.jpg)

![](images/58b7107d186673eb227d0091858ed6dcc0c08c72eb7f6666460c5958c0df85cf.jpg)

![](images/ecf860588ec235648e21dfaf6840b8d643922c564f2eef03754ef1139dd2ae24.jpg)  
Figure 4: Marginal distributions of the four action components across all non-terminal frames $\scriptstyle ( N = 8 , 5 0 4 )$ . Dashed red lines indicate per-component means. The strongly right-skewed ∆x reflects persistent forward pursuit; the symmetric $\Delta y _ { \ell }$ and ∆ψ capture bidirectional correction maneuvers.

flight in the majority of episodes; the small negative minimum $\left( - 0 . 3 6 \mathrm { m } \right)$ arises from height\_change-tagged episodes in which the operator descends modestly during a long-distance approach. Fourthly, $\Delta \psi$ exhibits a wide spread (std 0.355 rad, range [−1.25, +2.20] rad), a direct consequence of the approach/locate-subject episodes that require substantial heading reorientation to acquire the target before pursuit begins. Fig. 4 shows the full marginal histograms and Fig. 5 plots altitude and heading over time for representative episodes.

## 4.3. Linguistic diversity and instruction analysis

Fig. 6 shows the word cloud of the instruction corpus after removing task verbs and grammatical stop words. The most prominent appearance terms include safety vest, high-visibility, reflective, hard hat, long sleeve, overalls, headphones and boots lexical items naturally arising from warehouse PPE inventories. The

![](images/c1e03a925c892ed27a77d891aecd08547812ebe2416d3ac009862c28b6b98548.jpg)  
Figure 5: Altitude z (solid blue, left axis) and yaw angle ψ (dashed red, right axis) over time for six episodes. Altitude remains nearly constant in most episodes, while heading changes dynamically as the drone turns to track the moving target.

Appearance Vocabulary in Language Instructions

S leggings shoes eeved eevedshirt e keep brown shoes locate 七 short sleeve longys leeved colored reflective g ea matching whte O coverall Wwh telong sleeve green white pants shoes findhigh visibilit yellow hard

Figure 6: Word cloud of appearance tokens in language instructions (task verbs and stop words removed). Term size reflects relative frequency. PPE vocabulary (hard hat, safety vest, headphones) dominates, alongside rich color and clothing-type diversity.

instruction vocabulary also exhibits rich color diversity (orange, purple, terracotta, teal, lavender, moss green) and fine-grained compositional descriptions (short-sleeve beneath a safety shirt, long sleeves beneath a safety shirt) paraphrase patterns that are known to break naive token-matching grounding systems.

The corpus spans 15 task-verb families dominated by approach (the approach/locate-subject regime) and follow (continuous pursuit), with thirteen additional verb families including accompany, pursue, go after, move closer, move towards, fly after, trace, tail, stay close and move along providing paraphrase diversity (Fig. 2c). This design prevents models from ignoring the task-verb token, as diferent verbs may imply subtly diferent behavioral profiles (e.g., approach implying active closure vs. accompany implying lateral co-navigation). The dataset thus constitutes a natural testbed for instruction paraphrase generalization, a key challenge identified in recent VLN and VLA literature [15, 18]; Table D.3 (Appendix D) reports per-verb-family error and confirms that the model ranking is preserved across all linguistic conditions.

Sample Frames from WareFly-VLA Episodes  
![](images/bab963fb15a884d433be406941e9f49cbb4734924b37bc87e9f50c6e8e5c3c94.jpg)  
Figure 7: Sample frames from three episodes at four evenly-spaced time steps (left to right: t=0, t<sub>1</sub>, t<sub>2</sub>, t<sub>end</sub>). Row labels show the corresponding language instruction (truncated). The warehouse background, diverse worker appearances and varying drone–target distances illustrate the visual complexity of the dataset.

## 4.4. Challenge tags and dificulty axes

A subset of episodes carries explicit dificulty annotations that mark independent generalization axes. The most common is find, which marks the approach/locate-subject regime in which the drone must first acquire a specified worker before initiating pursuit. The remainder span close (the drone operates at close range to the target), far (the target reaches a distance >10 m), height\_change (the drone changes altitude by >0.5 m during the episode), obstructed (the target passes fully behind warehouse infrastructure for at least one frame) and extended (an unusually long trajectory). These tags support stratified evaluation along independent generalization axes. Fig. 7 shows sample frames from representative episodes.

## 4.5. Comparison with related datasets

Table 3 positions WareFly-VLA against the most closely related existing datasets across the dimensions of platform, language supervision, action representation, data source and industrial relevance. WareFly-VLA is the only dataset combining a UAV platform, natural-language appearance-based target specification, continuous low-level action supervision and an industrial/warehouse environment; we obtain these properties through photorealistic simulation rather than physical capture.

Table 3: Comparison with related datasets. ✓ = yes, × = no, – = not applicable. “Cont. actions” indicates whether continuous low-level action labels (not high-level discrete commands or waypoints) are provided; “Scr.” indicates real-world capture vs. simulation.
<table><tr><td>Dataset</td><td>Plat- form</td><td>Lang. instr.</td><td>Cont. acts.</td><td>Src.</td><td>Episodes</td></tr><tr><td>UAV123 [13]</td><td>UAV</td><td>X</td><td>√</td><td>Real</td><td>~150</td></tr><tr><td>AerialVLN [15]</td><td>UAV</td><td>√</td><td>X</td><td>Sim</td><td>8,446</td></tr><tr><td>AirNav [16]</td><td>UAV</td><td>√</td><td>X</td><td>Real</td><td>35,000</td></tr><tr><td>UAV-VLA [22]</td><td>UAV</td><td>√</td><td>X</td><td>Real</td><td>100k</td></tr><tr><td>RaceVLA [23]</td><td>UAV</td><td>√</td><td>√</td><td>Real</td><td>~200</td></tr><tr><td>CognitiveDrone [21]</td><td>UAV</td><td>√</td><td>√</td><td>Sim</td><td>8,000</td></tr><tr><td>BridgeData V2 [34]</td><td>Arm</td><td>√</td><td>√</td><td>Real</td><td>60,096</td></tr><tr><td>DROID [35]</td><td>Arm</td><td>√</td><td>√</td><td>Real</td><td>76,000</td></tr><tr><td>Open X-Embodiment [24]</td><td>Multi</td><td>V</td><td>V</td><td>Real</td><td>1M+</td></tr><tr><td>WareFly-VLA (ours)</td><td>UAV</td><td>√</td><td>√</td><td>Sim</td><td>507</td></tr></table>

We emphasize that the relatively modest episode count compared to manipulating datasets is not an oversight: each episode requires manual teleoperation paired with a hand-authored, uniquely identifying worker appearance and the dataset’s primary value lies in its task fidelity and linguistic diversity photorealistic warehouse imagery, varied PPE configurations and human-flown target-following trajectories rather than in raw scale. The benchmark experiments in Sections 5 and 6 demonstrate that, despite this modest scale, the dataset already discriminates sharply between modern VLA architectures.

## 5. Benchmark protocol

## 5.1. Benchmark models

A primary objective of WareFly-VLA is to provide a standardized benchmark for evaluating Vision– Language–Action learning in aerial environments. To this end, four representative open-source VLA architectures are selected: OpenVLA-7B [5], GR00T N1.7 [36], π<sub>0</sub>/OpenPI [25] and SmolVLA [8]. These models collectively span the three dominant action-generation paradigms currently used in VLA systems, namely discrete action tokenization, difusion-based action generation and flow matching. They further cover a broad range of parameter scales and embodiment assumptions, providing a diverse testbed for studying aerial transfer and language-conditioned UAV control. Table 4 summarizes the benchmarked architecture used for fine-tuning.

Because the same trajectories can be re-exported at an arbitrary control rate, we also evaluate at 10 FPS the deployment rate of a reactive body-frame flight loop in addition to the 1 FPS rate at which contemporary VLAs publish results. To keep the cross-rate comparison interpretable under our compute budget, at 10 FPS we re-train only the accuracy frontier π<sub>0</sub> and SmolVLA, the two models that beat the constant-mean baseline at 1 FPS at matched optimization budgets (15,000 steps each). OpenVLA, whose epoch-based recipe would exceed 50 h on the 10×-larger corpus and GR00T, which diverges at 1 FPS (Section 6), are excluded at 10 FPS as uninformative.

## 5.2. Data splits

A single canonical episode-level split is adopted throughout the benchmark. Episodes are partitioned using a fixed random seed and a validation ratio of 15%, yielding 431 training episodes and 76 validation episodes (7,276 train / 1,228 val non-terminal transitions at 1 FPS; 73,979 / 13,371 transitions at 10 FPS)

Table 4: Representative Vision–Language–Action architectures benchmarked on WareFly-VLA.
<table><tr><td>Model</td><td>Parameters</td><td>Action Representation</td><td>Vision-Language Back- bone</td><td>Adaptation</td></tr><tr><td>OpenVLA-7B</td><td>7.6B total 49.3M trainable</td><td>Autoregressive policy (256 bins per dimension)</td><td>Prismatic SigLIP + DINOv2 Llama-2 7B</td><td>LoRA</td></tr><tr><td>GR00T N1.7</td><td>3.1B total 1.62B trainable</td><td>Diffusion policy Horizon = 4</td><td>Cosmos-Reason2-2B</td><td>Fine-tuning</td></tr><tr><td>π₀/OpenPI</td><td>3.3B total 145M trainable</td><td>Flow matching Horizon = 4</td><td>PaliGemma-3B</td><td>LoRA</td></tr><tr><td>SmolVLA</td><td>450M total 450M trainable</td><td>Flow matching Horizon = 4</td><td>SmolVLM-2 + Action Expert</td><td>Full fine-tuning</td></tr></table>

in the current dataset release. The 431/76 episode partition is shared verbatim by every model at both temporal resolutions, so no validation episode enters any training set at either rate. Splitting is performed at the trajectory level rather than the frame level to eliminate temporal leakage between training and evaluation samples.

## 5.3. Training protocol

All models are initialized from publicly released checkpoints and fine-tuned exclusively on WareFly-VLA training data. Whenever possible, oficial implementations and recommended training procedures provided by the original authors are adopted to minimize implementation-specific bias.

Input observations consist of a single RGB image and its associated language instruction. Images are resized to the native input resolution expected by each architecture. For models whose native action spaces difer from the four-dimensional UAV control representation used in WareFly-VLA, compatibility mappings are applied while preserving the semantic meaning of the original control commands.

No test-time augmentation is employed. Unless otherwise stated, all reported results correspond to a single forward pass per validation sample.

## 5.4. Evaluation metrics

For each action dimension $d \in \{ \Delta x , \Delta y , \Delta z , \Delta \psi \}$ , Mean Absolute Error (MAE) and Pearson correlation coeficient are described by

$$
\mathrm { M A E } _ { d } = \frac { 1 } { N } \sum _ { t = 1 } ^ { N } \left| \hat { a } _ { t , d } - a _ { t , d } \right| ,\tag{5}
$$

$$
r _ { d } = \frac { \sum _ { t } ( \hat { a } _ { t , d } - \bar { \hat { a } } _ { d } ) ( a _ { t , d } - \bar { a } _ { d } ) } { \sqrt { \sum _ { t } ( \hat { a } _ { t , d } - \bar { \hat { a } } _ { d } ) ^ { 2 } } \sqrt { \sum _ { t } ( a _ { t , d } - \bar { a } _ { d } ) ^ { 2 } } } .\tag{6}
$$

MAE measures action prediction fidelity, whereas Pearson correlation evaluates how well a policy reproduces the temporal structure of expert behavior. These metrics provide complementary perspectives on policy quality, as low prediction error may be achieved by regressing toward the mean action, while high correlation does not guarantee accurate action magnitudes.

In addition, a constant-mean baseline is reported for each action dimension. This baseline predicts the mean action observed in the training set and provides a lower bound for meaningful policy learning.

![](images/4afb2546de5812a585003c0256898829af3a47935f03e61b09e2f552a514f3dc.jpg)

Table 5: Per-step prediction performance on the WareFly-VLA held-out validation split. Results are reported as Mean Absolute Error (MAE) and Pearson correlation (r). All models are evaluated on the same 76 held-out episodes (N = 1, 228 frames) under the canonical leakage-free split.
<table><tr><td>Model</td><td>∆xe (m)</td><td> $\Delta y _ { \ell } ~ ( \mathrm { m } )$ </td><td> $\Delta z \ \mathrm { ( m ) }$ </td><td> $\Delta \psi \ \mathrm { ( r a d ) }$ </td></tr><tr><td colspan="5">MAE↓</td></tr><tr><td>Naive (constant-mean)</td><td>0.590</td><td>0.242</td><td>0.003</td><td>0.248</td></tr><tr><td>OpenVLA-7B [5]</td><td>0.620</td><td>0.368</td><td>0.005</td><td>0.409</td></tr><tr><td>GR00T N1.7 [36]†</td><td>5.728</td><td>3.867</td><td>0.637</td><td>0.555</td></tr><tr><td> $\pi _ { 0 } / \mathrm { O p e n P I } \left[ 2 5 \right]$ </td><td>0.413</td><td>0.244</td><td>0.003</td><td>0.220</td></tr><tr><td>SmolVLA [8]</td><td>0.468</td><td>0.264</td><td>0.003</td><td>0.243</td></tr><tr><td colspan="5">Pearson r ↑</td></tr><tr><td> $\mathrm { O p e n V L A { - } 7 B }$ </td><td>+0.57</td><td>+0.18</td><td>+0.01</td><td>+0.23</td></tr><tr><td> $\mathrm { G R 0 0 T N 1 } . 7 ^ { \dagger }$ </td><td>+0.01</td><td>-0.06</td><td>-0.01</td><td>+0.18</td></tr><tr><td> $\pi _ { 0 } / \mathrm { O p e n P I }$ </td><td>+0.65</td><td>+0.48</td><td>-0.04</td><td>+0.52</td></tr><tr><td>SmolVLA</td><td>+0.56</td><td>+0.33</td><td>+0.06</td><td>+0.32</td></tr></table>

Figure 8: Cross-model comparison on the WareFly-VLA held-out validation split (76 identical episodes for all models). Left: Per-dimension MAE (lower is better). Right: Pearson correlation (higher is better). π<sub>0</sub> achieves the best overall performance, followed by SmolVLA, while OpenVLA shows limited generalization and GR00T fails to produce meaningful predictions. No model surpasses the naive baseline on $\Delta y _ { \ell }$ or $\Delta z$ , highlighting the dificulty of lateral and vertical action prediction.

## 6. Results and analysis

## 6.1. Main results

Table 5 reports per-dimension MAE and correlation of each model on the held-out validation episodes, together with the naive constant-mean reference. Figure 8 visualizes these numbers across all four dimensions; the y-axis range is preserved so that GR00T’s outsized errors remain visible.

The headline result, once train/validation leakage is removed (Section 5.2), is that WareFly-VLA is far from solved. At 1 FPS the strongest model, $\pi _ { 0 }$ , attains only $r { = } 0 . 6 5$ on the forward displacement $\Delta \boldsymbol { x } _ { \ell } ,$ $r { = } 0 . 5 2$ on heading ∆ψ and r=0.48 on the lateral channel $\Delta y _ { \ell }$ , while the vertical channel $\Delta z$ is at chance $( | r | \le 0 . 0 6$ for every model). No model beats the naive constant-mean baseline in MAE on $\Delta y _ { \ell }$ or $\Delta z$ . The flow-matching models form the top tier: $\pi _ { 0 }$ (3.3B) leads SmolVLA (450M) on every dimension $( \Delta x _ { \ell } ~ r ~ 0 . 6 5$ vs. $0 . 5 6 ; \Delta y _ { \ell } \ r \ 0 . 4 8 \ \mathrm { v s . } \ 0 . 3 3 ; \Delta \psi \ r \ 0 . 5 2 \ \mathrm { v s . } \ 0 . 3 2 )$ and both beat naive in MAE on $\Delta { x } _ { \ell }$ and $\Delta \psi$ . OpenVLA recovers forward direction reasonably $( \Delta x _ { \ell } r { = } 0 . 5 7 )$ but is worse than naive on every MAE channel, because it systematically overshoots magnitude on the in-plane channels and collapses the low-amplitude $\Delta z$ and $\Delta \psi$ commands onto quantization-bin centers. The most important qualitative finding is shared across all models: only the forward channel $\Delta x _ { \ell }$ is reliably learnable from a single frame, while lateral, vertical and (for all but $\pi _ { 0 }$ and SmolVLA) heading commands are not these channels demand temporal context that a per-frame policy cannot access (Section 8).

Table 6: Per-step prediction accuracy at 10 FPS on the validation split (76 episodes, $N { = } 1 3 { , } 3 7 1$ transitions). Values report MAE and Pearson r with 95% bootstrap confidence-interval half-widths.
<table><tr><td>Model</td><td> $\Delta x _ { \ell } \ \mathrm { ( m ) }$   $\Delta y _ { \ell } \ ( \mathrm { m } )$ </td><td> $\Delta z \ \mathrm { ( m ) }$ </td><td> $\Delta \psi \ \mathrm { ( r a d ) }$ </td></tr><tr><td> $M A E \downarrow$ </td><td colspan="3"> $( \times 1 0 ^ { - 2 } ,$  except  $\Delta z )$ </td></tr><tr><td>Naive</td><td> $6 . 0 4 { \pm } 0 . 0 9$ </td><td> $2 . 7 4 { \pm } 0 . 0 7$   $0 . 0 2 8 { \scriptstyle \pm 0 . 0 0 1 }$ </td><td> $2 . 5 8 { \pm } 0 . 0 5$ </td></tr><tr><td>π0</td><td> $\mathbf { 4 . 8 0 \ : \pm { \ : 0 . 0 9 } }$   ${ \bf 2 . 7 0 \pm 0 . 0 6 }$ </td><td>0.037</td><td> $2 . 6 2 \pm 0 . 0 6$ </td></tr><tr><td>SmolVLA</td><td> $5 . 3 3 \pm 0 . 0 9$   $2 . 9 5 \pm 0 . 0 6$ </td><td>0.038</td><td> $\underline { { 2 . 6 8 } } \pm 0 . 0 6$ </td></tr><tr><td></td><td colspan="3"> $P e a r s o n \ r \ \uparrow$ </td></tr><tr><td> $\pi _ { 0 }$ </td><td>+0.51</td><td> $+ 0 . 4 9$ </td><td>+0.12  $+ 0 . 3 7$ </td></tr><tr><td>SmolVLA</td><td>+0.43  $+ 0 . 3 1$ </td><td>+0.08</td><td> $+ 0 . 3 7$ </td></tr></table>

The same per-step protocol is also applied at the 10 FPS deployment-rate setting for the accuracy frontier (π , SmolVLA). Table 6 reports the corresponding per-dimension MAE and Pearson r on the identical 76 held-out episodes $( N { = } 1 3 , 3 7 1$ transitions). Two observations stand out. First, the architecture ordering is preserved: $\pi _ { 0 } >$ SmolVLA > naive on $\Delta x _ { \ell }$ in MAE and $\pi _ { 0 }$ leads SmolVLA in correlation on the forward and lateral channels $( \Delta x _ { \ell } r 0 . 5 1 \mathrm { v s . } 0 . 4 3 ; \Delta y _ { \ell } r 0 . 4 9 \mathrm { v s . } 0 . 3 1 )$ . Second, the per-step MAE margin over the baseline collapses: because each action spans one inter-frame interval, the 10 FPS per-frame displacements are ∼10× smaller and the naive constant-mean baseline is correspondingly tighter, so only $\Delta { x } _ { \ell }$ is clearly beaten in MAE by both models; on $\Delta y _ { \ell }$ only $\pi _ { 0 }$ edges the baseline and on $\Delta \psi / \Delta z$ neither model beats naive in MAE despite $\pi _ { 0 }$ retaining a positive heading correlation $\left( r { = } 0 . 3 7 \right)$ . To make the comparison dimensionless and baseline-relative across rates we further report the skill score $s _ { d } = 1 - \mathrm { M A E } _ { d } / \mathrm { M A E } _ { d } ^ { \mathrm { n a i v e } }$ in the discussion (Section 8); $s _ { d } > 0$ beats the same-rate baseline. Fig. 9 visualizes the cross-model comparison at 10 FPS: the correlation panel preserves the 1 FPS structure, while the MAE panel sits nearly flush with the baseline on the low-amplitude channels.

## 6.2. Generalization gap

The gap between training-set and held-out accuracy is the single most informative statistic in this benchmark and it is large for every model. Table D.2 (Appendix D) quantifies it: on $\Delta \boldsymbol { x } _ { \ell }$ , SmolVLA’s MAE rises from 0.083 (train) to 0.468 (val), a 5.6× degradation; $\pi _ { 0 }$ rises from 0.150 to 0.413 (2.8×); OpenVLA from 0.472 to 0.620 (1.3×). Two patterns emerge. First, the smaller flow-matching model overfits the hardest: SmolVLA achieves the best training fit of any model yet generalizes worse than the 7×-larger $\pi _ { 0 } .$ so its capacity is spent memorizing the 431 training episodes rather than acquiring a transferable visuomotor mapping. Second, OpenVLA exhibits the smallest relative gap only because it underfits even the training data, a direct consequence of quantizing a continuous, heavy-tailed action distribution into 256 bins. This train/val gap is precisely the quantity that train-on-everything evaluation protocols hide; under the leakage free split it becomes the defining challenge of the benchmark.

The same overfitting signature persists at 10 FPS and again favours the larger model: $\pi _ { 0 } { } ^ { ; }$ s $\Delta x _ { \ell }$ MAE rises from 0.014 (train) to 0.048 (val), a +0.034 gap, versus SmolVLA’s 0.013 → 0.053 (+0.040). At both rates SmolVLA attains the lower training error yet the larger held-out gap, confirming that the capacityversus-overfitting dissociation is a property of the architectures rather than of the chosen sampling rate.

VLA Model Comparison on WareFly-VLA  
![](images/13d5ec1529bb3ca056e401e78396c376aa1f5b369d766613fc82ac4366dca161.jpg)  
Figure 9: Cross-model comparison at the 10 FPS deployment-rate setting. Left: per-dimension MAE learned bars sit nearly flush with the naive baseline on $\Delta y _ { \ell } / \Delta z / \Delta \psi$ , the consequence of ∼10×-smaller per-frame actions. Right: Pearson r preserves the 1 FPS structure (π >SmolVLA; forward most learnable, vertical least). The contrast motivates reporting correlation alongside MAE at high control rates.

![](images/7586927a140e02044486ceda6361507bdc2d24d45149fe564d1540981194e334.jpg)  
Figure 10: Per-dimension prediction-vs.-ground-truth scatter for π (the strongest model) on the held-out validation split. The dashed diagonal denotes the identity line; ∆x shows a positive but range-compressed trend, while $\Delta y _ { \ell }$ and ∆z collapse toward the prior mean the per-sample signature of the generalization gap.

## 6.3. Per-sample prediction quality

Fig. 10 resolves the per-sample prediction quality of the leading model $\scriptstyle \left( \pi _ { 0 } \right)$ into a scatter plot for each action dimension on the held-out split. The $\Delta { x } _ { \ell }$ panel shows a clear positive trend along the identity diagonal but with substantial residual scatter the model captures the direction of forward motion yet compresses its dynamic range, under-predicting the largest pursuit displacements. The $\Delta y _ { \ell }$ and $\Delta z$ panels collapse toward a horizontal band near the training mean, the visual signature of a policy that has defaulted to the prior because the lateral and vertical commands are not predictable from a single frame. This contrasts sharply with the corresponding train-split scatter (not shown), where all four dimensions hug the diagonal a direct visualization of the generalization gap quantified in Section 6.2.

## 6.4. Qualitative analysis

Fig. 11 visualizes SmolVLA’s predicted actions overlaid on representative held-out frames. For each frame we draw the ground-truth in-plane displacement $( \Delta x _ { \ell } , \Delta y _ { \ell } )$ as a solid blue arrow and the SmolVLA prediction as a dashed purple arrow originating at the same image-plane anchor; text overlay reports the per-dimension ground-truth, prediction and absolute error. The predicted arrows are consistently oriented forward, tracking the dominant $\Delta x _ { \ell }$ component but they systematically under-reach the ground-truth arrow on large displacements and frequently mis-estimate the lateral $\Delta y _ { \ell }$ component the per-frame manifestation of the near-zero held-out lateral correlation in Table 5. We further probe rolled-out trajectory fidelity by integrating each predicted action sequence forward in time from the true initial pose of each episode (Fig. 12). On held-out episodes the integrated rollouts capture the gross forward progression of the expert path but accumulate visible drift on curved segments, where the unpredicted lateral and heading corrections compound over time a direct consequence of the single-frame observation model.

![](images/085c96a827b9ebbdff8b09b2e66f77e48fb28dec056c28fb6dd7c576524d2b7a.jpg)  
Figure 11: Qualitative SmolVLA predictions on twenty randomly-sampled held-out frames. Solid blue arrows depict the ground-truth in-plane displacement; dashed purple arrows depict the predicted displacement. Predictions track the forward direction but under-reach on large displacements and miss lateral corrections, illustrating the generalization gap on truly unseen episodes.

## 7. Deployment-oriented analysis

## 7.1. Cost–accuracy frontier

Fig. 13 positions each model on a two-dimensional chart of parameter count (log scale) and mean validation MAE (symmetric log), with marker area encoding fine-tuning wall-clock time on a single H100 80GB GPU. The Pareto frontier is defined by exactly two models: SmolVLA (smallest at 450M and fastest to fine-tune) and $\pi _ { 0 }$ (most accurate on the held-out split). The choice between them is a genuine cost–accuracy trade-of rather than a domination: SmolVLA is $7 \times$ smaller and trains in a fraction of the time, while $\pi _ { 0 }$ buys the best held-out correlation on the two learnable dimensions at $7 \times$ the parameter count. OpenVLA-7B is dominated 17× larger than SmolVLA, slower to fine-tune and less accurate on every dimension and GR00T, comparable in size to $\pi _ { 0 } ,$ , fails catastrophically. The actionable recommendation is therefore nuanced: under a tight compute budget SmolVLA is the rational default, but when held-out fidelity matters $\pi _ { 0 }$ is preferable; critically, neither reaches deployable accuracy on this benchmark, which remains open.

![](images/87ab2d66b21f756aa5e7d20b7d8eeb006c75d1a17fa44e10302abeaeb7691e69.jpg)  
Figure 12: Open-loop trajectory rollouts of SmolVLA predictions vs. expert trajectories on six held-out validation episodes. Solid blue is the expert path; dashed purple is the integrated prediction starting from the true initial pose; green squares mark start positions. Rollouts follow the gross forward progression but accumulate drift on curved segments where lateral/heading corrections are mi-predicted.

## 7.2. Inference latency and control rate

Per-step accuracy is silent on what ultimately decides deployability on a drone: whether a policy can close its control loop in real time. Table 7 reports single-frame inference latency on one H100 (10 warm-up, 50 timed calls). At 1 FPS, SmolVLA is fastest (36.5 ms, 27.4 Hz) and $\pi _ { 0 }$ sustains 14.9 Hz, while OpenVLA-7B manages only 2.7 Hz its autoregressive decoder cannot keep pace with a reactive loop and GR00T is fast but produces unusable predictions. The runtime axis is also where the 10 FPS setting matters most: a 10 Hz control loop must run at the data rate and both frontier models clear it, $\pi _ { 0 }$ at 15.9 Hz and SmolVLA at 31.6 Hz. SmolVLA’s efective rate is boosted by its 4-step action chunking, which runs the network once per four emitted actions, yielding a bimodal latency (median 3.2 ms, p90 122 ms); $\pi _ { 0 }$ is steady $( 6 2 . 8 \pm 1 . 4 $ ms). A practitioner can therefore pick an operating point 1 FPS for compute-frugal high-level guidance, 10 FPS for reactive following with a frontier model that meets the corresponding control budget at each.

## 7.3. Open-loop rollout drift

To complement the per-step view with a compounding one, we integrate each predicted action sequence forward from the true initial pose of every held-out episode and measure Average and Final Displacement Error (ADE/FDE) against the expert path (Table 8). The compounding view sharpens and confirms the ranking at both rates: $\pi _ { 0 }$ drifts least (1 FPS 2.55/4.73 m; 10 FPS 3.12/5.64 m), SmolVLA follows (3.06/5.95; 3.74/7.00) and at 1 FPS OpenVLA and especially GR00T trail far behind. FDE exceeds ADE for every model, reflecting accumulating drift on curved segments where the unpredicted lateral and heading corrections compound. Critically, rollout drift separates π<sub>0</sub> from SmolVLA cleanly at 10 FPS precisely where per-step MAE saturates against the baseline which is why we recommend reporting closed-loop rollout drift and correlation, not raw per-step MAE alone, when evaluating at a deployment frame rate.

![](images/cbc4b2a5da8dfa533706227f2dc6f133bede22ee922419f402cc93f0e4b2f49c.jpg)  
Figure 13: Cost–accuracy frontier. x-axis: model size (log scale); y-axis: mean validation MAE (symmetric log); marker area ∝ fine-tuning wall-clock time. SmolVLA and π<sub>0</sub> jointly define the Pareto frontier (cheapest vs. most accurate); OpenVLA is dominated and GR00T fails.

Table 7: Single-frame inference latency on one H100 (10 warm-up, 50 timed). “Rate” is sustained throughput; the last block is the 10 FPS frontier.
<table><tr><td>Rate</td><td>Model</td><td>Params</td><td>Lat. (ms)</td><td>p95</td><td>Hz</td></tr><tr><td rowspan="4">1 FPS</td><td>SmolVLA</td><td>0.45B</td><td>36.5</td><td>124</td><td>27.4</td></tr><tr><td>π₀/OpenPI</td><td>3.3B</td><td>67.2</td><td>70</td><td>14.9</td></tr><tr><td>GR00T N1.7</td><td>3.1B</td><td>77.9</td><td>101</td><td>12.8</td></tr><tr><td>OpenVLA-7B</td><td>7.6B</td><td>365.1</td><td>376</td><td>2.7</td></tr><tr><td rowspan="2">10 FPS</td><td>SmolVLA</td><td>0.45B</td><td>31.7*</td><td>122</td><td>31.6</td></tr><tr><td>π₀/OpenPI</td><td>3.3B</td><td>62.8</td><td>65</td><td>15.9</td></tr></table>

<sup>∗</sup>bimodal (median 3.2 ms / p90 122 ms) from 4-step chunking.

Runtimę-accuracy frontier (bubble size αparams)  
![](images/62e788ce7f3f3acf23ffbeb16805755f3f3d7bb3ba637cccf65597ba6936739c.jpg)  
Inference latency per action (ms, log) — faster ←  
Figure 14: Runtime–accuracy frontier. x-axis: single-frame inference latency (log ms, faster←); y-axis: held-out mean MAE (lower is better); bubble area ∝ parameter count. Vertical dashed lines mark the per-action time budgets for 1, 10 and 30 Hz control loops. SmolVLA and $\pi _ { 0 }$ sit in the accurate-and-fast quadrant and clear the 10 Hz budget; OpenVLA-7B violates even the 10 Hz budget and GR00T is fast but inaccurate.

Table 8: Open-loop rollout error (m) integrated from the true initial pose. ADE/FDE: average/final displacement error. The model ranking holds at both rates.
<table><tr><td>Rate</td><td>Model</td><td>ADE</td><td>FDE</td></tr><tr><td rowspan="5">1 FPS</td><td> $\pi _ { 0 } / \mathrm { O p e n P I }$ </td><td>2.55</td><td>4.73</td></tr><tr><td> $\mathrm { S m o l V L A }$ </td><td>3.06</td><td>5.95</td></tr><tr><td> $\mathrm { O p e n V L A { - } 7 B }$ </td><td>3.56</td><td>6.30</td></tr><tr><td>GR00T N1.7</td><td>24.92</td><td>43.84</td></tr><tr><td> $\pi _ { 0 } / \mathrm { O p e n P I }$ </td><td>3.12</td><td>5.64</td></tr><tr><td rowspan="2">10 FPS</td><td> $\mathrm { S m o l V L A }$ </td><td>3.74</td><td></td></tr><tr><td></td><td></td><td>7.00</td></tr></table>

![](images/d61bfcf55799eb51d961530b18670fb65dc84b27a3d503f257803aa3a3a1f877.jpg)  
Figure 15: Open-loop rollout error at 1 FPS: average (ADE, solid) vs. final (FDE, hatched) displacement error from the true initial pose. The flow-matching models drift least; OpenVLA trails; GR00T’s normalization failure produces order-of-magnitude larger drift. FDE exceeds ADE for every model, the signature of compounding error.

## 8. Discussion

The benchmark reveals three consistent characteristics of current Vision–Language–Action policies. First, flow-matching policies $( \pi _ { 0 }$ and SmolVLA) consistently outperform the discrete-token formulation adopted by OpenVLA. This advantage is primarily attributed to the continuous nature of flow matching, which avoids the quantization error introduced by discretizing low-amplitude UAV actions. The efect is particularly evident for the altitude and heading channels, where the empirical action range is considerably smaller than the discretization resolution. These observations are consistent with recent studies advocating continuous action representations for embodied control [50, 51]; inference-time steering of pre-trained VLAs [26] is a complementary route to narrowing the held-out gap on small aerial corpora without additional fine-tuning.

Second, the benchmark exposes an inherent limitation of single-frame aerial perception. While forward motion (∆x) can be estimated from instantaneous visual cues such as target scale and relative position, lateral $( \Delta y )$ and vertical (∆z) corrections depend largely on temporal information describing target motion. Consequently, all evaluated policies exhibit near-zero correlation on these channels, indicating that the bottleneck arises primarily from limited temporal observability rather than insuficient model capacity.

Finally, evaluation across diferent sampling rates demonstrates that per-step MAE alone is sensitive to frame rate. Increasing the sampling frequency reduces the magnitude of ground-truth actions, allowing even a constant-mean predictor to achieve deceptively low absolute error. Correlation, in contrast, remains comparatively stable because it measures agreement in temporal dynamics rather than action magnitude. These observations suggest that future aerial VLA benchmarks should complement per-step MAE with correlation-based measures, normalized skill scores, and closed-loop trajectory evaluation to provide a more comprehensive assessment of policy performance.

## 8.1. World-model roadmap enabled by WareFly-VLA

Our results expose a structural limit of reactive VLA policies. A single frame does not reveal relative motion, so the lateral and vertical channels remain at chance. World models target this limit directly.

![](images/4c616f4f59cea696446b565f81629131671ea65273bdc0308afe5f35338a7a9c.jpg)  
Figure 16: π open-loop rollouts (orange) vs. expert paths (blue) on six 10 FPS held-out episodes, integrated from the true initial pose (green square). Rollouts capture the gross forward progression but accumulate drift on curved segments where the unpredicted lateral and heading corrections compound the trajectory-level manifestation of the of-axis channels’ low correlation.

They maintain an internal state that summarizes past observations, predict how this state evolves under candidate actions and thereby support planning before acting [40, 37]. WareFly-VLA aligns five synchronized supervision streams for this purpose: RGB video, appearance-grounded instructions, continuous 4-DoF actions, ground-truth pose and dificulty tags. Fig. 17 organizes the resulting research space as a directed graph around the dataset core, with eight dependency edges (E1–E8) grouped into supervision (E1–E3), prediction–planning fusion (E4–E6) and a closed deployment loop (E7–E8). We highlight five directions.

(1) JEPA-style predictive representation (E1, E4). Joint-embedding predictive architectures learn a latent state z by predicting embeddings of future observations rather than pixels [37, 38, 52, 39]. This objective discards warehouse texture and retains task-relevant factors such as target identity and relative geometry. The obstructed and find tags provide direct probes of identity persistence under occlusion, the failure case that most afects person-following.

(2) Action-conditioned latent dynamics (E2, E5). The synchronized action–pose stream supports learning a latent flight-dynamics model $p ( z _ { t + 1 } \mid z _ { t } , \mathbf { a } _ { t } , \ell )$ conditioned on the instruction ℓ [41, 42]. Ground-truth pose gives a physical consistency check: imagined rollouts must agree with true ego-motion. Such a model recovers target-relative velocity from temporal context, which is exactly the information the single-frame bottleneck removes. Action-conditioned video generation ofers a complementary, pixel-space route to the same capability [43, 53].

(3) Language-conditioned prediction and reasoning (E3, E6). Instructions in WareFly-VLA constrain who to follow, not where to go. A predictive model can treat them as constraints on future states: which worker will match the description and where that worker will be. Concrete subtasks include phase prediction over the search–acquire–follow structure of Fig. 1 and counterfactual target hypotheses during active search [31, 21].

(4) Model-based planning and policy synthesis (E5–E7). Given a latent dynamics model, the UAV can score candidate action sequences before executing them, through latent model-predictive control or imagination-based learning [44, 45]. Recent self-supervised video world models already support such zeroshot robot planning [46]. Planned behaviors can then be distilled into fast flow-matching heads such as π and SmolVLA [25, 8], so deployment stays within the latency budgets of Section 7.2.

![](images/ba175af8cff8db25c0a99943acbe62912e965efcb810ff5f4f840dec43645917.jpg)  
Figure 17: World-model research roadmap enabled by WareFly-VLA. The dataset core is drawn as a quadrotor and aligns five synchronized supervision streams. Three predictive modules consume these streams: JEPA-style representation learning (1), action-conditioned latent dynamics (2) and language-conditioned prediction and reasoning (3). Their outputs fuse into modelbased planning (4), which reaches hardware through Physical-AI and sim-to-real transfer (5). Solid colored arrows (E1–E7) denote forward modeling and control dependencies; the dashed arrow (E8) returns deployment failures to data collection and closes the loop. The bottom band lists evaluation hooks already provided by the benchmark release.

(5) Physical AI and sim-to-real transfer (E7, E8). The corpus is generated under the physics stack of Isaac Sim and Pegasus [33, 32], so learned dynamics inherit a calibrated simulator prior. Domain randomization and world-foundation-model scene augmentation can widen the visual and physical distribution [47, 48], while agile-flight results show that simulation-trained aerial policies can reach real hardware [49]. Deployment failures return to the data core as new episodes (E8), closing the loop.

Each direction is testable with hooks already in the release: occlusion-recovery rate on tagged episodes, rollout drift (ADE/FDE, Table 8). The roadmap thus turns WareFly-VLA from a benchmark of reactive policies into a substrate for predictive, physically grounded aerial embodied intelligence.

## 8.2. Limitations

Four limitations bound the scope of our conclusions. Open-loop evaluation: every policy is scored per step against the ground-truth frame rather than the frame its own past actions would induce; closed loop deployment in simulation or on hardware an essential follow-up is expected to amplify the reported gaps, since compounding drift penalizes weaker predictors super-linearly. Dataset scale: the corpus is 507 teleoperated episodes (8,504 non-terminal transitions), small by manipulation-VLA standards; this deliberately trades raw count for task fidelity and linguistic diversity, so absolute saturation will require larger collections. Single environment and simulation: all episodes come from one photorealistic Isaac Sim warehouse, so cross-environment and sim-to-real transfer are not assessed here; a second-warehouse extension is in progress. Single-frame, single-camera observation: our protocol uses one forward-facing RGB stream, which is precisely what bottlenecks the of-axis channels analysed above; multi-frame, multi-view or RGB-D inputs are a natural extension and would also admit backbones designed for richer observations (e.g., RDT-1B’s two-camera setup [6]).

## 9. Conclusion

We have presented WareFly-VLA, a Vision–Language–Action dataset and benchmark for languageconditioned drone person-following in a photorealistic industrial warehouse. The benchmark evaluates four open-source VLA architectures spanning two orders of magnitude in parameter count. All models share one canonical, leakage-free episode-level split, bootstrap confidence intervals and two control rates (1 and 10 FPS).

The study yields five findings of broad relevance to the VLA and aerial-robotics communities. First, the task is far from solved. The strongest model, π , reaches only r≈0.5 on the forward and heading channels and no model beats a constant-mean baseline on the lateral or vertical channels, which lack single-frame signal. Second, every model shows a large train-to-validation gap and the smallest model overfits most. Held-out capacity, not training fit or size alone, determines the ranking. Third, continuous flow-matching action heads hold a consistent, statistically supported edge over discrete action tokenization, which collapses low-amplitude altitude and yaw commands. Fourth, the GR00T N1.7 new-embodiment interface fails to recalibrate its normalization statistics under limited-data aerial transfer. This failure mode should inform future foundation-model design. Fifth, the architecture ranking holds at both control rates, but per-step MAE loses discriminative power as the frame rate rises. Correlation and closed-loop rollout drift are therefore the rate-robust metrics for deployment-oriented aerial VLA evaluation.

Beyond the benchmark, we chart a world-model research roadmap, from JEPA-style predictive representation to Physical-AI transfer, that the dataset’s synchronized supervision streams enable. We release the dataset, baseline implementations and analysis code under an open license. We invite the community to extend the benchmark with additional baselines (e.g., OpenVLA-OFT [50], π -FAST [51], RDT-1B [6] and Octo [4]), additional warehouse environments and closed-loop hardware deployment. We hope WareFly-VLA catalyzes progress toward safe and capable language-guided aerial autonomy in real industrial environments.

## CRediT authorship contribution statement

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Data availability

The WareFly-VLA dataset, benchmark splits, training pipelines and evaluation scripts are available at https://huggingface.co/datasets/SIMOGroup/WareFly-VLA (access on request during review; public release upon acceptance).

## Declaration of generative AI and AI-assisted technologies in the writing process

During the preparation of this work the authors used Grammarly (professional version) in order to improve the readability and language of the manuscript. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

Table A.1: Per-model fine-tuning hyperparameters used in the benchmark. All runs use a single H100 80GB GPU. Total parameters / trainable parameters appear in Table 4.
<table><tr><td></td><td>OpenVLA</td><td>GR00T</td><td>π0</td><td>SmolVLA</td></tr><tr><td>Adaptation</td><td>LoRA</td><td>proj.+DiT</td><td>LoRA</td><td>full</td></tr><tr><td>LoRA r/α</td><td>16/32</td><td></td><td>expert head</td><td></td></tr><tr><td>Optimiser</td><td>AdamW</td><td>AdamW</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $2 \times 1 0 ^ { - 4 }$ </td><td> $1 \times 1 0 ^ { - 4 }$ </td><td> $1 \times 1 0 ^ { - 4 }$ </td><td> $5 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Steps / epochs</td><td>20 ep</td><td>6,000 st</td><td>15,000 st</td><td>15,000 st</td></tr><tr><td>Batch size</td><td>4</td><td>16</td><td>16</td><td>32</td></tr><tr><td>Precision</td><td>4-bit</td><td>bf16</td><td>bf16</td><td>bf16</td></tr><tr><td>Input image</td><td>2242</td><td>2562</td><td>2562</td><td>2562</td></tr></table>

## Acknowledgments

This work was supported by the project entitled “BioDroneX: AI-Enhanced Bio-Inspired Drone for Adaptive Multi-Environment Missions” at the Center for AI Research, VinUniversity.

## Appendices

The following appendices collect the reproducibility protocol, the worker-appearance and PPE inventory, the failure-mode analysis, additional robustness and cross-model analyses, and extended 10 FPS results referenced from the main text.

## Appendix A. Reproducibility Protocol

The canonical leakage-free split is built once and serialised to a manifest so that every per-model converter consumes the identical training and validation episode sets. The recipe is:

1. Enumerate all 507 episode UUIDs and sort them lexicographically (so the partition is invariant to insertion order and to filesystem listing order).

2. Seed a NumPy generator with the integer 0 and apply a permutation.

3. Assign the first 76 episodes (15%) to the validation set and the remaining 431 to the training set.

4. Write a JSON file listing the two episode ID sets; every model adapter (LeRobot, OpenVLA, GR00T, $\mathrm { O p e n P I } / \pi _ { 0 } )$ reads this file when converting the raw corpus into its own format.

5. Exclude terminal frames (those whose action labels are identically zero by construction) from both training and evaluation.

Under a naive train-on-everything protocol the flow-matching models report near-perfect held-out correlations (r>0.94 on every dimension), a number that collapses to the values reported in Table 5 once the leakage above is removed. We emphasise this because the gap between the two protocols reverses the model ranking and any future submission to the WareFly-VLA benchmark should adopt the leakage-free split verbatim.

Latency. Single-frame inference latency is reported in the runtime experiments through an opt-in benchmark mode that times only the observation-to-action call after 10 warm-up iterations, over 50 timed iterations, with explicit CUDA synchronisation between calls; the reported median and p95 therefore measure model forward cost rather than data-loading or simulator-update overhead.

## Appendix B. Worker Appearance and PPE Inventory

The simulated workers used as targets are dressed from a curated personal-protective-equipment (PPE) inventory designed to maximise lexical grounding diversity while remaining realistic for an industrialwarehouse setting. The inventory is grouped into six categories:

• Headwear: hard hats (orange, yellow, white, blue, red), bump caps, beanies and uncovered hair (varied colours including the corpus’s red hair mentions).

• Upper body: high-visibility safety vests (orange, yellow, lime, with and without reflective tape), buttonup shirts (long- and short-sleeve, in colours such as terracotta, teal, moss green, lavender, purple), polo shirts and t-shirts.

• Lower body: cargo trousers, work pants, overalls and coveralls in neutral and high-visibility colours.

• Footwear: steel-toe work boots and industrial trainers (typically dark brown or black).

• Hand protection: cut-resistant gloves, leather gloves and ungloved variants.

• Accessories: ear-protection headphones, dust masks, safety glasses, lanyards and clip-on identification tags.

Each episode samples one item from each applicable category, subject to the constraint that no two episodes share an identical configuration. This combinatorial structure is what supports the rich appearance vocabulary documented in the linguistic analysis (Section 4) and it is also what makes the find dificulty regime non-trivial: distractor workers in the same scene typically share two or three attribute categories with the target while difering in the remaining ones.

## Appendix C. Failure Mode Analysis

We document three recurring failure modes that emerge from the per-sample scatter plots and the qualitative arrow overlays:

1. Forward-magnitude compression. On the $\Delta { x } _ { \ell }$ channel, every working model produces predictions whose magnitudes are compressed toward the training-mean $\bar { a } _ { \Delta x _ { \ell } } { = } 0 . 8 6 4 ~ \mathrm { m / f r a m e }$ . Long-range pursuits at the right tail of the empirical distribution (up to 5.62 m/frame, Table 2) are under-predicted with a clear range-compression slope (aˆ≈0.5a in the large-action regime). This is the dominant residual error on the only learnable channel.

2. Of-axis collapse. On the $\Delta y _ { \ell }$ and $\Delta z$ channels, predictions collapse toward the training mean band: $\Delta y _ { \ell }$ predictions cluster near zero across the full empirical range and ∆z predictions are efectively constant. This produces a low MAE that matches the constant-mean baseline a benign-looking aggregate metric that masks the fact that the model has not learned the channel at all.

3. Verb-token shortcut absence. Per-verb-family error (Table D.3) shows no spike on rare verbs or on the heterogeneous other bucket, confirming that the models do not exploit the task-initiating verb as a shortcut. This is the absence of a failure mode that we deliberately designed against in the paraphrase-diverse annotation protocol (Section 3).

The first two failure modes admit the same prescription: the channels that the per-frame policy cannot recover are precisely those whose ground-truth supervision encodes temporal target-relative state (target velocity, target altitude change). Multi-frame observation or explicit velocity-from-pose conditioning is the route to closing them; we release the per-frame ground-truth pose state alongside the corpus to enable exactly this line of work.

![](images/6f0729e122e744d694685072f9abd2d28b76e5c70ccd9887cb68f844b7cb826c.jpg)

![](images/e50014b52b3b9c5e91761af1673640363c18d262c23b3b18e64fd8ce96587bd3.jpg)

![](images/067d257eea99586b28d43cef68c802cbd48e338bf4ddca984b66eeea3c9c4e99.jpg)  
Figure D.1: Training loss curves for the three models with step-level logs. OpenVLA optimizes cross-entropy over discrete action tokens; GR00T optimizes denoising score-matching; $\pi _ { 0 }$ optimizes the flow-matching velocity objective. Annotations report the final loss value.

## Appendix D. Additional Robustness and Cross-Model Analyses

This appendix collects the robustness and cross-model diagnostics referenced from Sections 6 and 7. They corroborate the headline per-dimension results the model ranking $\pi _ { 0 } > \mathrm { S m o l V L A } > \mathrm { O p e n V L A } >$ GR00T and the forward-channel-only learnability at finer granularity, but are not required to follow the main argument.

## Appendix D.1. Training dynamics

Fig. D.1 shows the loss trajectories for OpenVLA, GR00T and $\pi _ { 0 }$ over their respective fine-tuning runs. The SmolVLA does not emit a step-level log file. We observed the training loss to descend from 1.42 at initialization to 0.083 at convergence over a ∼50-minute run. Each curve corresponds to a diferent optimization objective cross-entropy over discrete action tokens for OpenVLA, denoising score-matching for GR00T and flow-matching velocity regression for $\pi _ { 0 }$ and the absolute scales are therefore not directly comparable across models. What is comparable is the shape: OpenVLA’s curve exhibits a clear overfitting onset around epoch 3 (∼300 steps) that we attribute to the LoRA adapters saturating on the available data; that the surrogate difusion loss does not surface; $\pi _ { 0 }$ exhibits a long quiescent flow-matching plateau before a sharp drop near step 1000 once the action expert aligns with the VLM hidden states a dynamic also reported in the original $\pi _ { 0 }$ paper [25].

## Appendix D.2. All-model scatter comparison

To complement the $\pi _ { 0 }$ scatter in Fig. 10, we show the equivalent held-out plots for the remaining three baselines (Figs. D.2, D.3). SmolVLA reproduces $\pi _ { 0 } ^ { \phantom { } } \mathrm { s }$ qualitative pattern a positive but range-compressed $\Delta x _ { \ell }$ trend with $\Delta y _ { \ell } / \Delta z$ collapsed to the prior band but with a steeper magnitude compression, consistent with its larger generalization gap. OpenVLA predictions exhibit a characteristic “staircase” artefact of discrete-token decoding: the $\Delta { x } _ { \ell }$ axis shows clear horizontal bands at quantization-bin boundaries and the $\Delta z$ axis collapses almost entirely to a single bin because the empirical $\Delta z$ range is smaller than the discretization grid resolution (cf. Sec. 8). GR00T predictions are dispersed two orders of magnitude outside the empirical action range, with no diagonal structure, consistent with the normalization failure diagnosed below.

## Appendix D.3. Error cumulative-distribution analysis

Aggregate MAE conceals the shape of the error distribution; two models with identical MAE may difer substantially in their tail behavior, which is the most policy-relevant statistic for safety-critical UAV deployment. Fig. D.4 plots the per-dimension cumulative distribution function (CDF) of |err| for each model on a symmetric-log x-axis. On the held-out split $\pi _ { 0 }$ and SmolVLA exhibit nearly identical CDF shapes $\pi _ { 0 }$ holds a small but consistent edge in the right tail of $\Delta { x } _ { \ell }$ and $\Delta \psi$ while OpenVLA’s $\Delta x _ { \ell }$ CDF is shifted right by roughly 1.7× (median $| e r r |$ on $\Delta x _ { \ell }$ near 0.73 m versus 0.42 m for the flow-matching models). GR00T’s CDF lies one-to-two orders of magnitude above the empirical scale and is consistent with random-magnitude predictions. The most striking feature is shared by the three working models: the $\Delta y _ { \ell }$ and $\Delta z \mathrm { \ C D } \dot { }$ Fs essentially coincide with the naive-baseline CDF, visually confirming that no model extracts held-out signal on the lateral and vertical channels.

![](images/a25bb583883591c5e2768ce0d8a79ee1cdd722db8812d96d219c081475d287b2.jpg)  
Figure D.2: Per-dimension prediction-vs.-ground-truth scatter for SmolVLA on the held-out validation split. The dashed diagonal is the identity line; ∆x is positive but range-compressed while $\Delta y _ { \ell }$ and ∆z collapse toward the prior mean.

![](images/e8db27cded193a7d1b81285e96875098bcad77a4ca82515199cf681220155c2a.jpg)  
Figure D.3: Per-dimension prediction-vs.-ground-truth scatter for OpenVLA-7B on the held-out split. Note the horizontal banding caused by 256-bin discretization and the near-vertical collapse on ∆z where the discretization grid resolution exceeds the empirical action range.

![](images/b6d760e87b11201421871711e829bc74930cc33c400d2db26aad9e940e89bec7.jpg)  
Figure D.4: Cumulative distribution of per-dimension absolute error across all four models on the held-out validation split. The x-axis uses a symmetric log scale to preserve both small and large error magnitudes. π (orange) and SmolVLA (purple) nearly coincide, with π<sub>0</sub> holding the right-tail edge; OpenVLA (red) is shifted right on $\Delta x _ { \ell } ;$ GR00T (green) lies far above the empirical scale. On $\Delta y _ { \ell } / \Delta z$ all working models track the naive baseline.

## Appendix $D . 4 \cdot$ Train–val gap analysis

Table D.2 and Fig. D.5 expand the generalization-gap analysis of Section 6.2 to every dimension. The gap is substantial for all models and counter to the intuition that a smaller model regularizes better it is largest for the smallest model: SmolVLA’s $\Delta x _ { \ell }$ MAE degrades from 0.083 (train) to 0.468 (val), a +0.385 m gap, versus +0.263 for $\pi _ { 0 }$ and +0.149 for OpenVLA. SmolVLA attains the lowest training error of any model on every dimension yet the steepest fall-of on held-out episodes, the signature of a high-capacity action expert memorizing 431 training trajectories. OpenVLA’s gap is smallest in absolute terms only because it never fits the training data well to begin with, a discretization-induced underfit rather than good generalization. This decomposition overfitting for the flow-matching models, underfitting for the discrete-token model is the central diagnostic that the leakage-free protocol makes visible and that train-on-everything evaluation conceals.

Table D.2: Train vs. validation MAE per dimension (the gap quantifies overfitting). Models with no train predictions available are omitted.
<table><tr><td>Model</td><td> $\Delta x _ { \ell }$ </td><td> $\Delta y _ { \ell }$ </td><td> $\Delta z$ </td><td> $\Delta \psi$ </td></tr><tr><td>openvla (train) openvla (val)</td><td>0.472 0.620</td><td>0.297 0.368</td><td>0.004 0.005</td><td>0.358 0.409</td></tr><tr><td>gap smolvla (train) smolvla (val) gap</td><td>+0.149 0.083 0.468 +0.385</td><td>+0.070 0.050 0.264 +0.214</td><td>+0.001 0.002 0.003 +0.002</td><td>+0.052 0.063 0.243 +0.180</td></tr><tr><td>pi0 (train)</td><td>0.150</td><td>0.081</td><td>0.002</td><td>0.084</td></tr><tr><td>pi0 (val)</td><td>0.413</td><td></td><td></td><td>+0.135</td></tr><tr><td>gap</td><td>+0.263</td><td>0.244 +0.163</td><td>0.003 +0.001</td><td>0.220</td></tr></table>

Table D.3: Mean validation error broken down by task-verb family (averaged across the four action dimensions). We omit verb families with fewer than 5 validation samples.
<table><tr><td>Verb</td><td>OpenVLA</td><td>GR00T</td><td> $\pi _ { 0 }$ </td><td>SmolVLA</td></tr><tr><td>accompany</td><td> $0 . 3 0 0 _ { ( 5 6 ) }$ </td><td> $2 . 5 9 4 _ { ( 5 3 ) }$ </td><td> $0 . 2 2 1 _ { ( 5 6 ) }$ </td><td> $0 . 1 9 7 _ { ( 5 6 ) }$ </td></tr><tr><td>approach</td><td> $0 . 2 5 8 _ { ( 2 2 9 ) }$ </td><td> $2 . 5 9 5 _ { ( 2 8 9 ) }$ </td><td> $0 . 1 8 0 _ { ( 2 2 9 ) }$ </td><td> $0 . 1 9 9 _ { ( 2 2 9 ) }$ </td></tr><tr><td>fly after</td><td> $0 . 6 6 5 _ { ( 1 9 ) }$ </td><td> $3 . 3 1 7 _ { ( 1 4 ) }$ </td><td> $0 . 2 5 5 _ { ( 1 9 ) }$ </td><td> $0 . 2 4 5 _ { ( 1 9 ) }$ </td></tr><tr><td>follow</td><td> $0 . 3 1 0 _ { ( 1 0 8 ) }$ </td><td> $3 . 3 4 9 _ { ( 1 3 3 ) }$ </td><td> $0 . 2 0 2 _ { ( 1 0 8 ) }$ </td><td> $0 . 2 3 3 _ { ( 1 0 8 ) }$ </td></tr><tr><td>go after</td><td> $0 . 5 5 4 _ { ( 4 6 ) }$ </td><td> $2 . 7 1 9 _ { ( 6 2 ) }$ </td><td> $0 . 2 6 4 _ { ( 4 6 ) }$ </td><td> $0 . 2 9 7 _ { ( 4 6 ) }$ </td></tr><tr><td>move closer</td><td> $0 . 3 8 2 _ { ( 5 2 ) }$ </td><td> $2 . 3 8 3 _ { ( 4 5 ) }$ </td><td> $0 . 1 8 7 _ { ( 5 2 ) }$ </td><td> $0 . 1 9 0 _ { ( 5 2 ) }$ </td></tr><tr><td>move towards</td><td> $0 . 2 4 7 _ { ( 5 6 ) }$ </td><td> $2 . 3 5 6 _ { ( 3 2 ) }$ </td><td> $0 . 1 5 5 _ { ( 5 6 ) }$ </td><td> $0 . 1 4 8 _ { ( 5 6 ) }$ </td></tr><tr><td>other</td><td> $0 . 3 7 0 _ { ( 5 6 3 ) }$ </td><td> $2 . 6 0 8 _ { ( 5 0 0 ) }$ </td><td> $0 . 2 3 5 _ { ( 5 6 3 ) }$ </td><td> $0 . 2 6 8 _ { ( 5 6 3 ) }$ </td></tr><tr><td>pursuit</td><td> $0 . 4 2 2 _ { ( 8 6 ) }$ </td><td> $2 . 6 6 4 _ { ( 6 8 ) }$ </td><td> $0 . 2 8 6 _ { ( 8 6 ) }$ </td><td> $0 . 3 3 6 _ { ( 8 6 ) }$ </td></tr><tr><td>stay close</td><td> $0 . 3 5 5 _ { ( 1 3 ) }$ </td><td> $2 . 4 0 8 _ { ( 1 2 ) }$ </td><td> $0 . 1 9 3 _ { ( 1 3 ) }$ </td><td> $0 . 1 7 1 _ { ( 1 3 ) }$ </td></tr></table>

## Appendix D.5. Error vs. action magnitude

A useful diagnostic for continuous-control policies is how prediction error scales with the ground-truth action magnitude: a policy whose error grows sub-linearly with $\left| a _ { t , d } \right|$ is more useful for large pursuits than one whose error grows linearly. Fig. D.6 bins ground-truth magnitudes into six quantile bins and reports mean validation error (line) and interquartile range (shaded) within each bin, for every model and dimension. On the held-out split, all working models show error that grows with |GT| on $\Delta x _ { \ell } .$ , the signature of range compression: the policies regress toward the training mean and therefore under-reach the largest expert displacements, so the absolute error is smallest in the low-magnitude bins and largest in the high-magnitude pursuit bins. $\pi _ { 0 }$ and SmolVLA share this profile with $\pi _ { 0 }$ slightly flatter (less compression); OpenVLA sits above both with an additional magnitude-independent ofset set by its 256-bin quantization grid. GR00T’s error is dimension-uniform and uncorrelated with magnitude. This analysis localizes the dominant error source on the one learnable dimension under-prediction of large displacements and shows it aflicts even the strongest model, underscoring how far the benchmark is from saturation.

![](images/ceab1498fa7aadd0ee23327c5381aad8935550352c6e06dc5cdc94ba28ffc7c6.jpg)

![](images/4acc18521383a463aaf221957b50623b809a7e005c2039b959574af6d4c4ee4b.jpg)

![](images/9307463475a13dcf99957b2b9bf87ae3e2f91c212f0cbb3a7d29c8b04656f738.jpg)

![](images/5797dba41c2d4a69fdb987ceb70314cf8373df31499794ca4a0fb4777b4ad199.jpg)

![](images/f706ee82e1057cdce7f49df14597c0b1bac21f76ffe1a4712f936808d47fa437.jpg)  
Figure D.5: Per-dimension MAE on the train (lighter bars) and validation (darker bars) splits, for each model with train predictions available. Every model shows a large train–val gap; SmolVLA’s is the largest despite its lowest training error, while OpenVLA underfits even the training split.  
Figure D.6: Mean error (line) vs. ground-truth magnitude, with interquartile range shaded, binned into six quantile bins. SmolVLA’s error grows sub-linearly with |GT| on the in-plane axes; OpenVLA’s error is roughly magnitude-independent (constant error floor); GR00T’s predictions are essentially decorrelated from |GT|.

## Appendix D.6. Per-episode error heatmap

Fig. D.7 aggregates per-episode MAE on a log-color scale across all four models and all four action dimensions. The visualization makes two patterns directly legible. First, the column structure is consistent: the $\pi _ { 0 }$ and SmolVLA columns are darker (lower error) than OpenVLA, which is in turn far darker than GR00T and this ordering holds across nearly every episode the ranking is not driven by a handful of episodes. Second, there are row-aligned “hard episodes” that penalize all working models simultaneously (bright rows spanning the $\pi _ { 0 } ,$ SmolVLA and OpenVLA columns), predominantly the long approach/locatesubject episodes with large heading changes direct evidence that dificulty is a property of the episode, not merely of the model.

## Appendix D.7. Linguistic robustness: Per-verb-family breakdown

A core motivation for our paraphrase-diverse instruction corpus is to evaluate whether language-conditioned policies are sensitive to the task-initiating verb. Table D.3 reports per-verb-family mean validation error for each model, restricted to verb families with at least five held-out samples; under the leakage-free protocol every model is scored on the identical held-out frames, so the columns are directly comparable. Within the working models, error is broadly stable across verb families: $\pi _ { 0 }$ ranges from 0.155 (move towards) to 0.286 (pursuit), SmolVLA from 0.148 (move towards) to 0.336 (pursuit). The two dominant regimes, approach (N=229) and follow (N=108), yield similar errors (π : 0.180 and 0.202; SmolVLA: 0.199 and 0.233), indicating the models do not exploit the verb token as a shortcut. Fig. D.8 visualizes the same data on a symmetric-log scale. The model ranking (π<sub>0</sub> ≈ SmolVLA < OpenVLA ≪ GR00T) is preserved across all linguistic conditions, so paraphrase diversity is not a hidden confound in the headline comparison.

![](images/40ea72037adc2be04c395f8dd104d545f451e853d5a2db52eb022fa6c2291f44.jpg)  
Figure D.7: Per-episode MAE heatmap, log-color scaled, with rows corresponding to episodes (shortened UUID labels on the leftmost panel) and columns to models. Each panel shows one action dimension. Darker is lower error; the structural pattern is consistent across dimensions: flow-matching models (smolvla, pi0) are uniformly low-error while the discrete and difusion baselines are uniformly high.

## Appendix D.8. Within-episode error drift

A pure per-step open-loop evaluation cannot directly measure compounding drift, but it can probe a weaker proxy: does the policy degrade within an episode as frame index increases? We bin all validation predictions by their normalized position in the parent episode (first decile, second decile, ..., last decile) and report mean error per bin. Fig. D.9 shows the result: every model’s curve is essentially flat, indicating that errors are uncorrelated with within-episode time. This is consistent with the fact that the trajectories we capture do not contain systematic late-episode events that the policies fail to handle a sanity check for our episode-curation protocol.

## Appendix D.9. Side-by-side qualitative comparison

Fig. D.10 overlays all four models’ predictions on the same six validation frames. For each frame we draw the GT in-plane displacement as a solid blue arrow and each model’s prediction as a dashed arrow in the model’s reference color. The $\pi _ { 0 }$ and SmolVLA arrows align with the GT arrow in direction on most frames but fall short in magnitude, OpenVLA’s arrow is shorter still and biased toward the discretized grid centre and GR00T’s arrow points far of-frame on nearly every sample. The figure makes the dominant held-out error mode directionally correct but magnitude-compressed in-plane predictions visible at the level of single predictions, while GR00T’s gross scale failure is immediately apparent.

![](images/417a44485db6e3897c4c98b0074dcf413954895c5b70f3663580fd2549dd63cc.jpg)  
Figure D.8: Per-verb-family mean validation error (averaged across the four action dimensions) on a symmetric-log scale. Within each model, performance is stable across verb families; the model ranking is preserved across all linguistic conditions.

Error vs Frame Position (does error grow late in episode?)  
![](images/108bf4c26daa76272b872d8cffa548efdcb2cd70a965b5b07ca6f6cee384cc35.jpg)  
Figure D.9: Mean error vs. normalized frame position in episode (deciles). All models are essentially flat, indicating no systematic within-episode error drift; this is the expected behavior for a well-curated dataset.

![](images/b59b060a59a898e8837d5d137fb9470e9d3bad868beb609846c9bd6778ed4125.jpg)  
Figure D.10: Same six held-out frames, with predictions overlaid from all four VLA models. Solid blue arrows are GT; dashed arrows are per-model predictions in the model’s reference color (red: OpenVLA; green: GR00T; orange: π<sub>0</sub>; purple: SmolVLA). The flow-matching predictions match GT direction but under-reach in magnitude; the discrete-token prediction undershoots further; the difusion prediction points of-frame.

## Appendix E. Extended Experimental Results

This appendix collects supplementary diagnostics that corroborate the headline analysis but are too finegrained for the main text. Figures are grouped by theme; unless noted, 1 FPS plots span all four baselines and 10 FPS plots cover the accuracy frontier (π , SmolVLA).

## Appendix E.1. Correlation and per-episode ranking summaries

Fig. E.11 compresses the per-dimension Pearson correlation of all four models at 1 FPS into a single heatmap, making the learnability hierarchy $( \Delta x _ { \ell } { > } \Delta \psi { > } \Delta y _ { \ell } { > } \Delta z )$ and the model ordering legible at a glance. Fig. E.12 reports the pairwise per-episode win rate $P ( A$ beats B), confirming that the ranking is a perepisode property rather than an averaging artifact: $\pi _ { 0 }$ beats SmolVLA on 64% of episodes, both flowmatching models beat OpenVLA on $\geq 8 4 \%$ and every model beats GR00T on 100%.

## Appendix E.2. Directional accuracy and calibration at 10 FPS

The underdispersion and directional-accuracy signatures discussed in the main text are visualized for each model in Figs. E.13–E.14. The speed-calibration histograms (Fig. E.13) show the predicted horizontalstep magnitude distribution shifted below ground truth (mean pred/GT ≈ 0.85 for $\pi _ { 0 } )$ , the magnitudecompression signature. The yaw-turn confusion matrices (Fig. E.14) show that “straight” is recovered reliably while left/right turns are frequently predicted as straight the directional counterpart of the modest heading correlation.

![](images/8ca256db022d7aa8c22a6cc79cc6522a34070f5765f577b1d0794c84d317559c.jpg)  
Figure E.11: Per-dimension prediction–ground-truth Pearson correlation for all four baselines at 1 FPS. Greener is higher. The forward channel is the most learnable for every model; ∆z is at chance throughout; $\pi _ { 0 }$ dominates column-wise.

## Appendix E.3. SmolVLA companion diagnostics and the 10 FPS error CDF

For completeness, Fig. E.16 gives SmolVLA’s per-sample scatter at 10 FPS (the companion to $\pi _ { 0 } ^ { } \mathrm { { s } }$ Fig. E.15) and Fig. E.17 its open-loop rollouts, both showing the same range-compression and curve-drift behavior as $\pi _ { 0 }$ but slightly more pronounced. Fig. E.18 plots the per-dimension absolute-error CDF at 10 FPS, the cumulative view of the per-step MAE saturation discussed in Section 6.

## References

[1] D. Driess, F. Xia, M. S. M. Sajjadi, C. Lynch, A. Chowdhery, B. Ichter, A. Wahid, J. Tompson, Q. Vuong, T. Yu, W. Huang, Y. Chebotar, P. Sermanet, D. Duckworth, S. Levine, V. Vanhoucke, K. Hausman, M. Toussaint, K. Gref, A. Zeng, I. Mordatch, P. Florence, PaLM-E: An embodied multimodal language model, arXiv preprint arXiv:2303.03378 (2023).

[2] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, et al., RT-1: Robotics transformer for real-world control at scale, in: arXiv preprint arXiv:2212.06817, 2022.

[3] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn, et al., RT-2: Vision-language-action models transfer web knowledge to robotic control, arXiv preprint arXiv:2307.15818 (2023).

![](images/57e378e52e627032888d6904eb979290cfaea810431cd73279485d05e8981082.jpg)

Figure E.12: Pairwise per-episode win rate P(row beats column) in mean held-out MAE at 1 FPS. The ordering π >SmolVLA > OpenVLA > GR00T holds at the per-episode level, not merely in aggregate.  
![](images/6ebd9b386b548a06d349bb3a3bccd18d05c27fddd704ff407d78bd5cc8a2de5c.jpg)

![](images/156ee577b6b1ce1a708f51b9995f406ff383c01bbc9ed244e02004897bc19621.jpg)  
Figure E.13: Horizontal-speed calibration at 10 FPS: ground-truth (blue) vs. predicted (orange) step-magnitude distributions for π<sub>0</sub> (left) and SmolVLA (right). Both models under-predict the spread (dashed means), the magnitude-compression signature.

![](images/3f2b13e95f85fdc23cb924696cb134772558799b5a0610307b8207330bdb402d.jpg)

![](images/45be5b4492b4a573f25455fba3f18552d4c47bbe749a26962e2e68e4ea7d3ca4.jpg)  
Figure E.14: Yaw-turn confusion matrices at 10 FPS (rows: ground-truth turn; columns: predicted) for π<sub>0</sub> (left) and SmolVLA (right). Straight motion is recovered well; left/right turns collapse toward “straight”, explaining the modest ∆ψ correlation.

![](images/a702a0bf0ad88cd4c4348920cbf183cbca493e9cf0d5b3d5e3e589fcfed4696a.jpg)

![](images/600c25e005e6168ca064e768afae1b2e9de8c727cdea14592b0f77a54ac85ba8.jpg)

![](images/5ee9d162d5f86c6cf2d63cfea77de52c71a9f2e7c2e49899b35832fbf13a8273.jpg)

![](images/ca80d35e75b649ac7d3713b06b0f9a93a59ff8b91564a5f26c1a93a1e2207390.jpg)  
Figure E.15: Per-dimension prediction-vs.-ground-truth scatter for π<sub>0</sub> on the 10 FPS held-out split (N=13,371). The dashed diagonal is the identity line. The qualitative signature is identical to 1 FPS (Fig. 10) at one-tenth the magnitude scale: a positive but range-compressed ∆x trend, partial ∆y structure and ∆z collapsed to the prior band.

![](images/de7bef48927a0d664c5f9f0b5fa9c5062047e8eb8d0b265401445718e5fbce4d.jpg)

![](images/28b5b12d619bd32ea0a5e8ba487c285377f4b1ad028d52795986918cbab38e16.jpg)

![](images/a279dcd5c7a9c924425cd093263d01e665804263a9ae226b8b70c710ba7c5b1d.jpg)

![](images/62fc36fd4e32da4b54b4316f805ab737d50ae942e83b2df14aff8ed2dfa3a3d6.jpg)  
Figure E.16: SmolVLA per-dimension prediction-vs.-ground-truth scatter at 10 FPS. The pattern matches π (Fig. E.15) with steeper ∆x<sub>ℓ</sub> range compression, consistent with SmolVLA’s larger generalization gap.

![](images/aec6ca2d2a24e2ea12929cf527f7f7aebf4af6d5f0fb5599a9a1b283fe596c52.jpg)  
Figure E.17: SmolVLA open-loop rollouts (orange) vs. expert paths (blue) on six 10 FPS held-out episodes. Drift on curved segments is larger than π ’s (Fig. 16), consistent with the ADE/FDE ranking of Table 8.

![](images/2fc43b920128b5e8f66325fd9a4b78e05226c597af0931d36fc34a586d0376c5.jpg)

![](images/44a00b25eb9d6b9460180247ba15258ffd698a7e442dbbc999580ab089d6f6ba.jpg)

![](images/d1a8c0de85b9789423aefd26a94eaf04f16bcf4f0b69f1affbffad18bd28ea14.jpg)

![](images/8e0465b0814fb0d0251a35b790aaeda6acc475dc5b9119e002fb83f6f66f326f.jpg)  
Figure E.18: Per-dimension absolute-error CDF at 10 FPS (symmetric-log x). π holds the right-tail edge over SmolVLA on $\Delta \boldsymbol { x } _ { \ell } ,$ , while both nearly coincide with the naive baseline on $\Delta y _ { \ell } / \Delta z / \Delta \psi$ the cumulative view of the per-step MAE saturation at high control rate.

[4] D. Ghosh, H. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, T. Kreiman, C. Xu, et al., Octo: An open-source generalist robot policy, in: Proceedings of Robotics: Science and Systems (RSS), 2024.

[5] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, Q. Vuong, T. Kollar, B. Burchfiel, R. Tedrake, D. Sadigh, S. Levine, P. Liang, C. Finn, OpenVLA: An open-source vision-language-action model, arXiv preprint arXiv:2406.09246 (2024).

[6] S. Liu, L. Wu, B. Li, H. Tan, H. Chen, Z. Wang, K. Xu, H. Su, J. Zhu, RDT-1B: A difusion foundation model for bimanual manipulation, in: International Conference on Learning Representations (ICLR), 2025.

[7] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al., π<sub>0.5</sub>: A vision-language-action model with open-world generalization, arXiv preprint arXiv:2504.16054 (2025).

[8] M. Shukor, D. Aubakirova, F. Capuano, P. Kooijmans, S. Palma, A. Zouitine, M. Aractingi, C. Pascal, M. Russi, A. Marafioti, S. Alibert, M. Cord, T. Wolf, R. Cadene, SmolVLA: A vision-language-action model for afordable and eficient robotics, arXiv preprint arXiv:2506.01844 (2025).

[9] J. Wen, Y. Zhu, J. Li, M. Zhu, K. Wu, Z. Xu, N. Liu, R. Cheng, C. Shen, Y. Peng, F. Feng, J. Tang, TinyVLA: Towards fast, data-eficient vision-language-action models for robotic manipulation, arXiv preprint arXiv:2409.12514 (2024).

[10] Q. Wang, M. Li, J. Guan, J. Ye, S. Xie, Y. Liu, J. Chen, Z. Liang, J. Zhang, X. Hu, X. Huang, P. Lin, J. Lin, D. Liu, S. Bai, J. Zhou, J. Zhang, H. Yuan, G. Zhou, H. Yin, Y. Wang, Y. Huang, Z. Lei, W. Peng, D. Chen, Y. Zheng, J. Fan, X. Zhuang, X. Zhou, H. Li, A. Chen, T. Zhang, X. Liu, Y. Sun, R. Chen, Z. Li, C. Lü, Z. Yang, T. Yu, X. Chen, Qwen-vla: Unifying vision-language-action modeling across tasks, environments, and robot embodiments (2026). arXiv:2605.30280. URL https://arxiv.org/abs/2605.30280

[11] A. M. P. S. Nascimento, A. V. Brito, M. Saska, T. P. Nascimento, A review on vision-based control for multi-rotor aerial vehicles, Robotics and Autonomous Systems 198 (2026) 105366. doi:10.1016/ j.robot.2026.105366.

[12] P. Chen, Y. Dang, R. Liang, W. Zhu, X. He, Real-time object tracking on a drone with multi-inertial sensing data, IEEE Transactions on Intelligent Transportation Systems 19 (1) (2018) 131–139. doi: 10.1109/TITS.2017.2750091.

[13] M. Mueller, N. Smith, B. Ghanem, A benchmark and simulator for UAV tracking, in: Proceedings of the European Conference on Computer Vision (ECCV), 2016, pp. 445–461.

[14] J. Zhao, J. Zhang, D. Li, D. Wang, Vision-based anti-uav detection and tracking, IEEE Transactions on Intelligent Transportation Systems 23 (12) (2022) 25323–25334. doi:10.1109/TITS.2022.3177627.

[15] S. Liu, H. Zhang, Y. Qi, P. Wang, Y. Zhang, Q. Wu, AerialVLN: Vision-and-language navigation for UAVs, arXiv preprint arXiv:2308.06735 (2023).

[16] A. Deshpande, et al., AirNav: A large-scale real-world UAV vision-and-language navigation dataset with natural and diverse instructions, arXiv preprint arXiv:2601.03707 (2026).

[17] W. Zhang, Y. Liu, X. Wang, X. Chen, C. Gao, X. Chen, EmbodiedCity: Embodied aerial agent for citylevel visual language navigation using large language models, in: 2024 23rd ACM/IEEE International Conference on Information Processing in Sensor Networks (IPSN), 2024, pp. 265–266.

[18] X. Wang, D. Yang, Z. Wang, H. Kwan, J. Chen, W. Wu, H. Li, Y. Liao, S. Liu, Towards realistic UAV vision-language navigation: Platform, benchmark, and methodology, arXiv preprint arXiv:2410.07087 (2024).

[19] M. Mei, et al., Towards natural language-guided drones: GeoText-1652 benchmark with spatial relation matching, arXiv preprint arXiv:2311.12751 (2024).

[20] Z. Wang, H. Kou, Z. Lv, Y. Zhang, Z. Guo, C. Wang, A comprehensive review of drone-based autonomous driving datasets: Methodology, taxonomy, and prospects, IEEE Transactions on Intelligent Transportation Systems 26 (10) (2025) 14501–14515. doi:10.1109/TITS.2025.3571726.

[21] M. A. Mustafa, G. Tadevosyan, A. Akhmetkazy, M. A. Cabrera, M. Martynov, S. Karaf, D. Tsetserukou, CognitiveDrone: A VLA model and evaluation benchmark for real-time cognitive task solving and reasoning in UAVs, arXiv preprint arXiv:2503.01378 (2025).

[22] O. Sautenkov, Y. Yaqoot, A. Lykov, M. A. Mustafa, G. Tadevosyan, A. Akhmetkazy, M. A. Cabrera, M. Martynov, S. Karaf, D. Tsetserukou, UAV-VLA: Vision-language-action system for large scale aerial mission generation, arXiv preprint arXiv:2501.05014 (2025).

[23] V. Serpiva, A. Lykov, A. Myshlyaev, M. H. Khan, A. A. Abdulkarim, O. Sautenkov, D. Tsetserukou, RaceVLA: VLA-based racing drone navigation with human-like behaviour, arXiv preprint arXiv:2503.02572 (2025).

[24] A. Padalkar, A. Pooley, A. Jain, A. Bewley, A. Herzog, A. Irpan, et al., Open X-Embodiment: Robotic learning datasets and RT-X models, arXiv preprint arXiv:2310.08864 (2023).

[25] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, S. Jakubczak, T. Jones, L. Ke, S. Levine, A. Li-Bell, M. Mothukuri, S. Nair, K. Pertsch, L. X. Shi, J. Tanner, Q. Vuong, A. Walling, H. Wang, U. Zhilinsky, π<sub>0</sub>: A vision-language-action flow model for general robot control, arXiv preprint arXiv:2410.24164 (2024).

[26] Z. Li, J. Liu, Z. Dong, T. Teng, Q. Rouxel, D. Caldwell, F. Chen, Towards deploying VLA without finetuning: Plug-and-play inference-time VLA policy steering via embodied evolutionary difusion, IEEE Robotics and Automation Letters 11 (5) (2026) 6234–6241. doi:10.1109/LRA.2026.3678455.

[27] Y. Gao, Z. Wang, L. Jing, D. Wang, X. Li, B. Zhao, Aerial vision-and-language navigation via semantictopo-metric representation guided LLM reasoning, arXiv preprint arXiv:2410.08500 (2024).

[28] Y. Fan, W. Chen, T. Jiang, C. Zhou, Y. Zhang, X. E. Wang, Aerial vision-and-dialog navigation, arXiv preprint arXiv:2205.12219 (2022).

[29] Q. Chen, N. Gao, S. Huang, J. Low, T. Chen, J. Sun, M. Schwager, GRaD-Nav++: Vision-language model enabled visual drone navigation with Gaussian radiance fields and diferentiable dynamics, IEEE Robotics and Automation Letters 11 (2) (2026) 1418–1425.

[30] Z. Wang, N. Mahmoudian, Vision-driven river following of UAV via safe reinforcement learning using semantic dynamics model, Robotics and Autonomous Systems 198 (2026) 105357. doi:10.1016/j. robot.2026.105357.

[31] Y. Tian, F. Lin, Y. Li, T. Zhang, Q. Zhang, J. Fu, X. Huang, X. Dai, Y. Wang, C. Tian, B. Li, Y. Lv, L. Kovács, F.-Y. Wang, UAVs meet LLMs: Overviews and perspectives toward agentic low-altitude mobility, arXiv preprint arXiv:2501.02341 (2025).

[32] M. Jacinto, J. Pinto, J. Patrikar, J. Keller, R. Cunha, S. Scherer, A. Pascoal, Pegasus simulator: An isaac sim framework for multiple aerial vehicles simulation, in: 2024 International Conference on Unmanned Aircraft Systems (ICUAS), 2024, pp. 917–922. doi:10.1109/ICUAS60882.2024.10556959.

[33] NVIDIA, Isaac Sim. URL https://github.com/isaac-sim/IsaacSim

[34] H. R. Walke, K. Black, T. Z. Zhao, Q. Vuong, C. Zheng, P. Hansen-Estruch, A. W. He, V. Myers, M. J. Kim, M. Du, A. Lee, K. Fang, A. Balakrishna, D. Sadigh, S. Levine, BridgeData V2: A dataset for robot learning at scale, in: Proceedings of the Conference on Robot Learning (CoRL), 2023

[35] A. Khazatsky, K. Pertsch, S. Nair, A. Balakrishna, S. Dasari, S. Karamcheti, S. Nasiriany, M. K. Srirama, L. Y. Chen, et al., DROID: A large-scale in-the-wild robot manipulation dataset, in: Proceedings of Robotics: Science and Systems (RSS), 2024.

[36] NVIDIA, GR00T N1.7: An open foundation model for generalist humanoid robots, https://github. com/NVIDIA/Isaac-GR00T (2025).

[37] Y. LeCun, A path towards autonomous machine intelligence, OpenReview preprint, version 0.9.2 (2022).

[38] M. Assran, Q. Duval, I. Misra, P. Bojanowski, P. Vincent, M. Rabbat, Y. LeCun, N. Ballas, Self-supervised learning from images with a joint-embedding predictive architecture, arXiv preprint arXiv:2301.08243 (2023).

[39] A. Bardes, Q. Garrido, J. Ponce, X. Chen, M. Rabbat, Y. LeCun, M. Assran, N. Ballas, Revisiting feature prediction for learning visual representations from video, arXiv preprint arXiv:2404.08471 (2024).

[40] D. Ha, J. Schmidhuber, World models, arXiv preprint arXiv:1803.10122 (2018).

[41] D. Hafner, T. Lillicrap, I. Fischer, R. Villegas, D. Ha, H. Lee, J. Davidson, Learning latent dynamics for planning from pixels, in: Proceedings of the 36th International Conference on Machine Learning (ICML), 2019.

[42] N. Hansen, H. Su, X. Wang, TD-MPC2: Scalable, robust world models for continuous control, in: International Conference on Learning Representations (ICLR), 2024.

[43] J. Bruce, M. Dennis, A. Edwards, J. Parker-Holder, Y. Shi, et al., Genie: Generative interactive environments, in: Proceedings of the 41st International Conference on Machine Learning (ICML), 2024.

[44] D. Hafner, T. Lillicrap, J. Ba, M. Norouzi, Dream to control: Learning behaviors by latent imagination, International Conference on Learning Representations (ICLR) (2020).

[45] D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap, Mastering diverse domains through world models, arXiv preprint arXiv:2301.04104 (2023).

[46] M. Assran, A. Bardes, D. Fan, Q. Garrido, R. Howes, et al., V-JEPA 2: Self-supervised video models enable understanding, prediction and planning, arXiv preprint arXiv:2506.09985 (2025).

[47] J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, P. Abbeel, Domain randomization for transferring deep neural networks from simulation to the real world, in: 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2017, pp. 23–30.

[48] NVIDIA, Cosmos world foundation model platform for physical AI, arXiv preprint arXiv:2501.03575 (2025).

[49] E. Kaufmann, L. Bauersfeld, A. Loquercio, M. Müller, V. Koltun, D. Scaramuzza, Champion-level drone racing using deep reinforcement learning, Nature 620 (2023) 982–987.

[50] M. J. Kim, C. Finn, P. Liang, Fine-tuning vision-language-action models: Optimizing speed and success, arXiv preprint arXiv:2502.19645 (2025).

[51] K. Pertsch, K. Stachowicz, B. Ichter, D. Driess, S. Nair, Q. Vuong, O. Mees, C. Finn, S. Levine, FAST: Eficient action tokenization for vision-language-action models, arXiv preprint arXiv:2501.09747 (2025).

[52] A. Bardes, J. Ponce, Y. LeCun, MC-JEPA: A joint-embedding predictive architecture for self-supervised learning of motion and content features, arXiv preprint arXiv:2307.12698 (2023).

[53] A. Hu, L. Russell, H. Yeo, Z. Murez, G. Fedoseev, A. Kendall, J. Shotton, G. Corrado, GAIA-1: A generative world model for autonomous driving, arXiv preprint arXiv:2309.17080 (2023).