# Long Time No See: Benchmarking VLMs for Out-of-Sight Spatiotemporal Reasoning in Egocentric Videos

Fangzhou Ma<sup>∗1</sup> Ivo Alexander Ban<sup>∗1</sup> Eren Homburg<sup>∗1</sup> Gabriele Goletto<sup>2</sup> Remi Pautrat´ <sup>2</sup> Mahdi Rad<sup>2</sup> Chiara Plizzari<sup>3</sup> Marc Pollefeys<sup>1,2</sup>

<sup>1</sup>ETH Zurich <sup>2</sup>Microsoft Spatial AI Lab <sup>3</sup>Bocconi University

{fangma, ivoban, ehomburg}@student.ethz.ch

 Project Page · § Code · õ B<sub>EYOND</sub>3D Benchmark

![](images/49f2fff3e1ec6875652b905c41a314e3c317b72224322457cbd3a3b09f317d3e.jpg)  
Figure 1. Visual illustration of the BEYOND3D benchmark. We propose a benchmark that evaluates whether VLMs can follow interacted objects and reason about their location once they leave view. As an object (here, the box of eggs) is moved by the person, the benchmark poses eight questions that progressively probe the reasoning needed to recover its state once it gets out of sight. Answering them requires VLMs to track actively manipulated objects, update their spatial state after relocation, and recall that state once the objects leave view.

## Abstract

Real-world AI systems must reason about objects that are no longer visible: an AR assistant guiding a user back to an object used earlier, a household robot retrieving an item someone put away. This requires notjust recalling where an object was last seen, but updating its state when it is moved and retaining that update once it leaves view. We refer to this as out-of-sight spatiotemporal reasoning. We introduce BEYOND3D, the first VQA benchmark to isolate this

ability in dynamic egocentric video: every query targets an object that has been relocated and has since left thefield of view. We create our questions from HD-EPIC annotations, building a visibility track for each dynamic object from its 3D position, the camera pose, and the scene geometry to understand at each moment whether it is visible, occluded, or out of view. BEYOND3D comprises 9,000 questions in eight types over 135 videos from nine participants, organized as one reasoning chain: visual grounding (is the target observable now), temporal grounding (when it was last visible and last placed), scene localization (which fixture anchors that location), and 3D spatial perception (where it lies relative to the current viewpoint or another object in the scene). We benchmark nine general-purpose and spatially specialized VLMs. The best model reaches 42.2% against 29.7% chance and text-only baselines reaching 31.9%, with the largest failures in recovering when an object was last visible, showing that tracking object movement out of sight remainsfarfrom solvedfor current VLMs.

## 1. Introduction

For an embodied system, understanding only what is currently visible is not enough. Humans can remember and reason about previously seen objects even after they leave sight [4, 26]. Similarly, an AR assistant may need to guide a user back to an object handled earlier, while a household robot may need to retrieve an item after it has been moved out of sight. In such cases, the system must identify the interactions that established the object’s latest location and retain that state as the scene evolves. This is particularly challenging in egocentric video, where objects are manipulated and relocated as the viewpoint continuously changes. We refer to this ability to reason about the evolving spatial state of objects beyond the current field of view as out-ofsight spatiotemporal reasoning.

Recent vision-language models (VLMs) are increasingly capable of recognizing, describing, and answering questions about visual content [29, 30]. However, it remains unclear whether they can maintain coherent spatial representations of dynamic environments and reason about outof-sight objects. Existing benchmarks cover related aspects of memory, temporal reasoning, and 3D scene understanding, but do not directly test if models retain the updated spatial state of a relocated object after it leaves view (Table 1).

We introduce BEYOND3D, a Visual Question Answering (VQA) benchmark designed to isolate out-of-sight spatiotemporal reasoning in dynamic egocentric video. We build on HD-EPIC [25], which captures unscripted cooking and everyday activities with detailed annotations of object movements and 3D scene reconstructions. From these recordings, we identify interactions in which objects are picked up, carried, and relocated across counters, cupboards, drawers, etc. For each relocated object, we construct a geometry-aware visibility track combining its 3D location with the camera pose, field of view, and scene geometry to determine whether it is visible, occluded, or outside the camera view at each time step. These tracks allow us to follow the object’s spatial state as the wearer continues interacting with the environment and, crucially, to place queries only after the object is no longer observable.

BEYOND3D comprises eight question types covering visual grounding, temporal grounding, scene localization, and 3D spatial perception. Together, they trace the reasoning chain in Fig. 1, from identifying the relevant event and recovering the object’s last location to reasoning about its spatial relationships after it leaves the field of view.

Our contributions are threefold:

• A new problem. We formulate out-of-sight spatiotemporal reasoning: tracking the spatial state of objects after they leave the field of view, a fundamental capability for reasoning in dynamic, embodied environments.

• A large-scale benchmark. We introduce BEYOND3D, comprising 9,000 questions on visual and temporal grounding, scene localization, and 3D spatial perception. Geometry-aware visibility tracks combine object locations, camera poses, fields of view, and occlusion reasoning to determine object visibility over time.

• A systematic evaluation of current VLMs. We benchmark nine general-purpose and spatially specialized VLMs and analyze their failure modes. Temporal retrieval and spatial-state maintenance remain major bottlenecks, while substantial 3D reasoning errors persist even when upstream uncertainty is controlled.

## 2. Related Work

Persistent dynamic world modeling in egocentric perception. Operating in dynamic environments requires reasoning beyond what is currently visible. In cognitive science, this relates to object permanence [3]: understanding that objects continue to exist when unseen. In machine perception, it means maintaining object representations after they leave the field of view [33]. In interactive environments, however, persistence alone is insufficient because manipulation and relocation can change object state. The Event Calculus [19] captures this by treating world state as persistent until an event changes it. A coherent visual world model must therefore retain information about unseen objects and update their state when interactions alter it.

Egocentric video makes this particularly challenging: wearer motion changes visibility while interactions change object locations. Recent methods increasingly address this by persistent world-state representations. AMEGO [14] stores past interactions and visited locations in queryable memory. OSNOM [26] maintains persistent 3D locations of active objects, while Whareformer [6] replaces this engineered state maintenance with learned updates to persistent object representations and 3D positions over time.

Vision language models (VLMs). Recent general-purpose VLMs have expanded toward longer video inputs and stronger temporal modeling [1, 2, 10, 21, 29, 30, 34], but explicit 3D spatial modeling is typically not their primary objective. Spatially specialized VLMs introduce spatial structure through targeted supervision [5, 7, 40, 41], learned geometry representations [13, 16, 17, 28, 37, 42, 44, 46], or explicit 3D inputs such as depth and camera pose [8, 9, 23]. Comparison with existing benchmarks. Related benchmarks differ in the states that models must recover at query time (Table 1). Temporal and object memory benchmarks test event timing or past observations without grounding memory in 3D [27, 36], while static 3D benchmarks focus on scene geometry and layout [39]. Dynamic spatial-state benchmarks capture changing config urations, including object relocations [18, 25, 35, 43, 45] and user-centric relation changes [35], but do not enforce target invisibility, allowing current-frame shortcuts. SCP-Bench [45] instead infers unseen past or future states from partial video. Others also consider out-of-sight reasoning: Ego4D-VQ3D [15], which retrieves previously observed stationary object locations without tracking updates, and SpaMEM [22], which evaluates spatial-state revision in synthetic environments. Our benchmark BEYOND3D requires both relocation and loss of visibility in unscripted real-world video, with visibility verified geometrically.

Table 1. Comparison with related spatial and memory benchmarks. ✓: explicitly evaluated; ◦: partially covered or not systematically enforced; –: not explicitly targeted. Ego.: egocentric input; 3D: explicit 3D spatial reasoning; Tem-Loc.: temporal localization; Spa-Upd.: spatial-state update; Diag.: multi-level diagnostic decomposition; Upd-OOS.: updated target queried after loss of visibility; Geo-Vis.: geometry-aware visibility.
<table><tr><td>Benchmark</td><td>Ego.</td><td>3D</td><td>Tem- Loc.</td><td>Spa- Upd.</td><td>Diag.</td><td>Upd- OOS</td><td>Geo- Vis.</td></tr><tr><td colspan="8">(A) Temporal and object memory</td></tr><tr><td>EgoTempo [27]</td><td></td><td></td><td>0</td><td>0</td><td></td><td></td><td></td></tr><tr><td>EgoMemReason [36]</td><td></td><td></td><td>0</td><td>O</td><td></td><td></td><td></td></tr><tr><td colspan="8">(B) Static 3D spatial reasoning</td></tr><tr><td>VSI-Bench [39]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="8">(C) Dynamic spatial state reasoning</td></tr><tr><td>EOC-Bench [43]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HD-EPIC [25]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>EgoDynamic4D [18]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>UCS-Bench [35]</td><td></td><td></td><td></td><td>O</td><td></td><td>O</td><td></td></tr><tr><td>SCP-Bench [45]</td><td></td><td></td><td></td><td>O</td><td></td><td></td><td></td></tr><tr><td>Ego4D-VQ3D [15]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SpaMEM [22]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BEYOND3D (Ours)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## 3. Out-of-Sight Spatiotemporal Reasoning

## 3.1. Problem Formulation

Task setting. We consider active objects that the camera wearer manipulates and relocates. Given an egocentric video up to $T _ { q }$ and such a target object o, the task is to track its status while visible, update it upon relocation, and recall it once o leaves view.

Object state and query anchor. At each time t, the object has a 3D location $\ell _ { o } ( t ) \in \mathbb { R } ^ { 3 } \cup \{ \perp \}$ , where ⊥ marks an unannotated relocation interval and any other value means o is stationary. Each stationary location is associated with a semantic fixture $f _ { o } ( t )$ that supports or contains the object, such as a counter, drawer, or appliance.

A binary state $v _ { o } ( t ) \in \{ 0 , 1 \}$ records whether o is visible in the frame at time t. An object may be not visible because it lies outside the camera view or is occluded by a hand, another object, or a cabinet door. We query objects through anchors $( o , T _ { q } )$ for which o has been relocated before $T _ { q }$ and is stationary but not visible at query time, so that $\ell _ { o } ( T _ { q } ) \neq \perp$ and $v _ { o } ( T _ { q } ) = 0$ . We call $\ell _ { o } ( T _ { q } )$ and $f _ { o } ( T _ { q } )$ the last known location and fixture of the target.

The out-of-sight horizon measures the time since o was last visible,

$$
h ( o , T _ { q } ) = T _ { q } - \operatorname* { m a x } \{ t \leq T _ { q } \mid v _ { o } ( t ) = 1 \} .
$$

Reference-object questions use a distinct object $r \neq o$ that is stationary and visible at $T _ { q }$ , providing a scene-relative reference whose distance to o is to ego-motion invariant and whose direction is invariant to ego-translation.

## 3.2. BEYOND3D Q&A Formulation

Given a query anchor $( o , T _ { q } )$ , consisting of a target object o and query time $T _ { q } ,$ , we define eight diagnostic question types organized into four complementary capabilities: visual grounding, temporal grounding, scene localization, and 3D spatial perception. As illustrated in Fig. 1, these questions probe successive stages of out-of-sight reasoning: determining whether the target is currently observable, retrieving the events that established its latest status, grounding its remembered location in the scene, and reasoning about that location in 3D. The exact natural-language templates and answer choices are provided in Supp. 8.1.

Visual Grounding. We test whether the model recognizes the target and determines its visibility at query time: 1 Visibility Check: Is o visible at $T _ { q } \mathrm { 2 }$

Temporal Grounding. We test whether the model can retrieve the events defining the target’s latest status: 2 Last Visible Time: When was o last visible before $T _ { q } \mathrm { 2 }$ 3 Last Placement Time: When was o last placed before $T _ { q } \mathrm { 2 }$

Scene Localization. We test whether the model can update the target’s spatial status and anchor its last known location to the scene: 4 Nearest Fixture: Which fixture is closest to the last known location of o?

3D Spatial Perception. We test whether the model can reason about the remembered target location relative to the camera and other objects: 5 Object–Camera Direction: Where is o relative to the camera at $T _ { q } ? \_ { \varnothing }$ Object– Camera Distance: How far is o from the camera at $T _ { q } ?$ 7 Object–Object Direction: Where is o relative to reference object r at $T _ { q } \mathrm { 2 }$ 8 Object–Object Distance: How far is o from r at $T _ { q } \mathrm { 2 }$

![](images/514c09f7c74819c969b8f849b1442559a8364354b0eedbdde3d88ee3de71a490.jpg)  
Figure 2. BEYOND3D benchmark construction. (1) Visibility tracks are inferred through view, occlusion, and detection checks. (2) Valid out-of-sight anchors are converted into 9,000 questions.

## 4. Benchmark Construction

We build BEYOND3D on HD-EPIC [25] in two stages as shown in Fig. 2: (i) infer per-object visibility tracks, (ii) select balanced out-of-sight query anchors, and finally generate questions per anchor for VLM evaluation.

## 4.1. Visibility Tracks

HD-EPIC [25] annotates object relocations in egocentric kitchen videos. For each movement, it provides the start and end times and, at both endpoints, the object’s 2D bounding box, mask, 3D center, and supporting or containing fixture (e.g., a counter or drawer). It also provides a reconstructed 3D digital twin of each kitchen. Since locations are annotated only at movement endpoints, we assume each object remains at its last annotated location until the next movement begins and mark movement intervals as in motion. Using these annotations, the digital twins, and video frames, we construct a 1 fps visibility track for each object o, recording its location $\ell _ { o } ( t )$ and visibility state $v _ { o } ( t )$ over time.

Determining visibility. We determine the visibility of each stationary object in three stages (Fig. 2):

Stage 1: Camera field of view. We first determine whether the object projects into the current camera view. From its most recent annotated bounding box, we choose the center point, the four corners, and the four edge midpoints to approximate the object’s spatial extent. These points are back-projected to the 3D scene at the depth of the object’s last known position and then reprojected into the current frame using the relative camera pose and the FISH-EYE624 fisheye camera model of Project Aria [12]. Fig. 3 llustrates this back-projection and re-projection procedure across two camera viewpoints.

Because Aria images are fisheye and vignetted, we consider only the usable circular region inscribed in the square frame. We mark an object as out of view when fewer than half of its projected footprint points fall inside this region. All remaining samples proceed to Stage 2.

![](images/d18157634b001bef2f15b772fcebe1280f094e472cf6404a9b7dfb338df849c1.jpg)  
Figure 3. Cross-view object projection. An Knife’s annotated image footprint in Camera View 1 is back-projected to its last known 3D depth and reprojected into Camera View 2 using the relative camera pose. The projected footprint approximates the object’s spatial extent under the new viewpoint.

Stage 2: Geometric occlusion. For each footprint point, we cast a ray from the camera to its corresponding 3D location and intersect it with the static kitchen mesh. As in Stage 1, we use a majority heuristic and mark the object as occluded when at least half of the rays are blocked. To mitigate annotation inaccuracies, we require the first intersection to lie at least δ = 10 cm in front of the target.

An object may be blocked only by the fixture it currently occupies. The static meshes do not capture if fixtures like drawers or cupboards are open or closed which is why we mark these cases as fixture ambiguous. All samples which are not occluded proceed to Stage 3.

Stage 3: Detection-based confirmation. The static mesh does not capture transient occlusions by hands, movable objects, or clutter. We verify samples passing the geometric tests in the video using the open-vocabulary detector OWLv2 [24]. Further details are provided in Supp. 10.2.

As in the previous stages, we use a majority heuristic, marking an interval detected visible if the object is detected in at least half of the tested frames and visually unconfirmed otherwise. fixture ambiguous states are instead marked occluded after negative detection.

Track assembly. Consecutive samples sharing the same stage-1/stage-2 outcome are merged into intervals, which Stage 3 then labels as a whole, forming the final visibility tracks. These tracks distinguish visible, out of view, occluded, visually unconfirmed, and in motion states. The binary visibility of Sec. 3.1 follows as $v _ { o } ( t ) = 1$ for visible samples and $v _ { o } ( t ) = 0$ otherwise. Validation. We visually inspected the inferred visibility tracks using an independent human annotation pass. Across

![](images/560d96cdfecd581b169633089cd0e07be2f34a46d18c6b8eed7f4fb7fd86e0a8.jpg)

![](images/2b20d76fccfa2d9d9e1e4aedffcb25b706b99cf819993039cfd0621aa1edf388.jpg)

![](images/d7e1593a7bc6fd30f558f1684a2ac574782df901759d578e4e00f5d95a56e094.jpg)

![](images/1f55d88e0891b25bcd6b3cbe5cea8d4978203acb040bf785fa8ae5404fa42459.jpg)  
Figure 4. Evaluation-set statistics. Distribution of the 1,000 out-of-sight query anchors over query time (a), out-of-sight horizon (b), and number of times the target was moved before the query (c), together with the query-time distribution of the 1,000 visible control anchors (d). Dashed lines mark the temporal and horizon stratification boundaries used for sampling.

4,102 scored object marks from 344 frames spanning 30 videos, the tracks achieved 83.5% accuracy, with most errors arising from the visually unconfirmed state. Full validation details are provided in Supp. 11.

## 4.2. Q&A Generation and Statistics

Candidate query anchor selection. As summarized in the bottom panel of Fig. 2, we first select valid query anchors and then instantiate them into Q&A pairs. From the constructed visibility tracks, we enumerate valid query anchors $( o , T _ { q } )$ only when the target is explicitly labeled out of view or occluded at $T _ { q }$ . We exclude visually unconfirmed states, which often arise from transient dynamic occlusions by hands or other objects, to favor stable out-of-sight periods that more directly test retention of the target’s latent spatial state. We retain anchors whose target has a unique, human-readable name.

Q&A instantiation. For a retained query anchor $( o , T _ { q } )$ we instantiate all question types using fixed naturallanguage templates with placeholders. For example, “At [TIME], is the previously moved [OBJECT] visible in the current frame?” where [OBJECT] denotes o and [TIME] denotes $T _ { q }$ . The complete templates are provided in Supp. 8.1.

Each question is formatted as multiple choice, we then derive the ground-truth answer from visibility tracks, object annotation, and camera poses, and construct distractors using task-specific rules. Temporal distractors are sampled from bins at increasing temporal distances from the ground truth: near (±1–2 s), medium (±3–4 s), far (±5– 6 s), and very far (±7–30 s), prioritizing timestamps associated with other visibility or movement events. For scenelocalization questions, distractors are plausible alternative locations. Since 63.2% of placements occur on counters, using a single counter category would make many questions too coarse and heavily skew the answer distribution. We therefore use fixture categories directly in general, but when the target was last placed on a counter, we instead distinguish counter areas using nearby landmarks (e.g. counter area next to the microwave). For 3D spatial questions, the predefined direction and distance categories directly define the answer options. Full construction details are in Supp. 8.2.

Evaluation-set selection and statistics. From the candidate pool, we select 1,000 out-of-sight query anchors spanning diverse participants, videos, objects, out-of-sight durations and causes, and answer classes. We prioritize anchors with clear object relocations and reliable visibility evidence.

We approximately balance the anchors across a $3 \times 3$ stratification defined by query time $T _ { q }$ (early: 0–149 s, middle: 150–299 s, late: 300–600 s) and out-of-sight horizon $h ( o , T _ { q } )$ (short: 2–10 s, medium: 11–30 s, long: > 30 s). Figs. 4a and 4b show the balanced marginal distributions over query time and out-of-sight horizon, respectively. Fig. 4c further characterizes spatial-update complexity by the number of target relocations before $T _ { q } \mathrm { . }$ : 70% of anchors involve one or two target moves, while the long tail extends to 11 moves.

The resulting set spans 135 videos, nine participants, nine kitchens, 581 target instances, and 561 referenceobject instances. At query time, 900 targets are outside the camera’s field of view and 100 are geometrically occluded. Expanding each anchor into eight question types produces 8,000 out-of-sight questions, with balanced answer classes for the four 3D spatial question types.

Because these anchors all have negative visibility labels, we additionally sample 1,000 visible anchors as positive controls for the visibility question. These anchors contain previously moved objects that are visible at $T _ { q }$ and are also evenly distributed across the early, middle, and late query-time groups. Their query-time distribution is shown in Fig. 4d. The final evaluation set contains 9,000 questions. Quality control. We manually inspect all 1,000 sampled out-of-sight anchors, verifying from the relevant video evidence that the target is out of sight at query time and that the ground truth can be determined reliably. Ambiguous samples are discarded and replaced from the candidate pool.

Table 2. VLM accuracy (%) on BEYOND3D. Performance of VLMs across question types under Text only, Video + Text and Last Frame Only + Text input settings. Bold entries indicate the best-performing model for the respective question type.
<table><tr><td>Model</td><td></td><td></td><td>Size Macro Avg. Visual Grounding</td><td colspan="2">Temporal Grounding</td><td>Scene Localization</td><td colspan="4">3D Spatial Perception</td></tr><tr><td></td><td></td><td></td><td>Visibility Check</td><td>Last Visible Last Placement Time</td><td>Time</td><td>Nearest Fixture</td><td>Object-Camera Object-Camera Object-Object Object-Object Direction</td><td>Distance</td><td>Direction</td><td>Distance</td></tr><tr><td>No. of questions</td><td></td><td>9000 29.7</td><td>2000 50.0</td><td>1000 20.0</td><td>1000 20.0</td><td>1000</td><td>1000</td><td>1000</td><td>1000</td><td>1000</td></tr><tr><td>Random guessing</td><td></td><td></td><td></td><td></td><td>Text Only</td><td>22.7</td><td>25.0</td><td>33.3</td><td>33.3</td><td>33.3</td></tr><tr><td colspan="9"></td><td></td><td></td></tr><tr><td colspan="9">General-Purpose Models</td><td></td></tr><tr><td>Qwen-3.6 [30]</td><td>35B-A3B</td><td>31.0</td><td>50.0</td><td>20.9</td><td>21.2</td><td>33.4</td><td>24.6</td><td>32.5</td><td>32.6</td><td>33.0</td></tr><tr><td>Qwen-3.6 [30]</td><td>27B</td><td>31.7</td><td>49.5</td><td>17.8</td><td>19.0</td><td>40.2</td><td>25.0</td><td>33.6</td><td>34.3</td><td>34.1</td></tr><tr><td>Qwen-3.5 [29]</td><td>9B</td><td>31.5</td><td>50.0</td><td>21.1</td><td>21.8</td><td>36.5</td><td>25.9</td><td>32.6</td><td>32.0</td><td>32.1</td></tr><tr><td>Qwen-3-VL [1]</td><td>8B</td><td>30.2</td><td>50.4</td><td>19.5</td><td>20.4</td><td>25.1</td><td>25.6</td><td>33.8</td><td>34.0</td><td>32.7</td></tr><tr><td>InternVL-3.5 [34]</td><td>8B</td><td>30.8</td><td>50.2</td><td>21.5</td><td>20.3</td><td>27.2</td><td>25.2</td><td>33.2</td><td>35.9</td><td>32.8</td></tr><tr><td colspan="9">Specialized 3D Models</td><td></td><td></td></tr><tr><td>VLM-3R [13]</td><td>7B</td><td>31.0</td><td>50.0</td><td>21.3</td><td>19.4</td><td>34.5</td><td>24.6</td><td>31.8</td><td>33.7</td><td>33.0</td></tr><tr><td>Spatial-MLLM [37]</td><td>6B</td><td>30.0</td><td>49.4</td><td>22.4</td><td>19.8</td><td>22.2</td><td>25.1</td><td>32.7</td><td>34.1</td><td>34.0</td></tr><tr><td>Cambrian-P [40]</td><td>7B</td><td>31.9</td><td>51.1</td><td>19.7</td><td>18.0</td><td>41.3</td><td>25.0</td><td>33.4</td><td>33.5</td><td>33.3</td></tr><tr><td>SenseNova-SI [5]</td><td>8B</td><td>29.0</td><td>51.2</td><td>18.0</td><td>18.1</td><td>20.7</td><td>26.0</td><td>33.7</td><td>31.3</td><td>33.1</td></tr><tr><td colspan="9">Last Frame Only + Text</td><td></td><td></td></tr><tr><td colspan="9">General-Purpose Models</td><td></td></tr><tr><td>Qwen-3.6 [30]</td><td>35B-A3B</td><td>33.0</td><td>72.9</td><td>14.5</td><td>16.6</td><td>27.1</td><td>27.0</td><td>35.3</td><td>35.2</td><td>35.0</td></tr><tr><td>Qwen-3.6 [30]</td><td>27B</td><td>35.2</td><td>76.5</td><td>17.8</td><td>18.3</td><td>29.0</td><td>29.7</td><td>38.0</td><td>36.5</td><td>35.9</td></tr><tr><td>Qwen-3.5 [29]</td><td>9B</td><td>32.4</td><td>64.3</td><td>17.9</td><td>21.2</td><td>25.3</td><td>26.6</td><td>33.6</td><td>34.8</td><td>35.1</td></tr><tr><td>Qwen-3-VL [1]</td><td>8B</td><td>31.8</td><td>73.4</td><td>14.5</td><td>18.0</td><td>21.4</td><td>26.8</td><td>35.1</td><td>32.8</td><td>32.7</td></tr><tr><td>InternVL-3.5 [34]</td><td>8B</td><td>32.1</td><td>71.3</td><td>14.9</td><td>16.5</td><td>24.5</td><td>27.9</td><td>34.8</td><td>33.3</td><td>34.0</td></tr><tr><td colspan="9">Specialized 3D Models</td><td></td><td></td></tr><tr><td>VLM-3R [13]</td><td>7B</td><td>33.6</td><td>68.2</td><td>21.5</td><td>21.8</td><td>27.7</td><td>28.1</td><td>33.6</td><td>33.8</td><td>34.1</td></tr><tr><td>Spatial-MLLM [37]</td><td>6B</td><td>31.9</td><td>60.1</td><td>22.3</td><td>20.0</td><td>23.8</td><td>25.5</td><td>35.6</td><td>33.5</td><td>33.9</td></tr><tr><td>Cambrian-P [40]</td><td>7B</td><td>32.9</td><td>69.5</td><td>18.7</td><td>18.2</td><td>31.0</td><td>25.4</td><td>33.4</td><td>33.9</td><td>33.3</td></tr><tr><td>SenseNova-SI [5]</td><td>8B</td><td>32.7</td><td>74.2</td><td>17.1</td><td>20.4</td><td>19.8</td><td>25.8</td><td>37.7</td><td>31.4</td><td>35.3</td></tr><tr><td colspan="9">Video + Text</td></tr><tr><td>General-Purpose Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen-3.6 [30]</td><td>35B-A3B</td><td>39.6 42.2</td><td>66.3 68.4</td><td>23.3 30.4</td><td>29.4</td><td>51.3 50.0</td><td>33.7 34.6</td><td>35.9 34.9</td><td>41.1 43.6</td><td>35.6 36.2</td></tr><tr><td>Qwen-3.6 [30] Qwen-3.5 [29]</td><td>27B 9B</td><td>38.6</td><td>59.3</td><td>28.2</td><td>39.4 30.5</td><td>47.9</td><td>33.4</td><td>39.2</td><td>34.7</td><td>35.8</td></tr><tr><td>Qwen-3-VL [1]</td><td>8B</td><td>36.0</td><td>59.8</td><td>21.3</td><td>25.7</td><td>48.2</td><td>32.7</td><td>33.6</td><td>31.5</td><td>35.2</td></tr><tr><td>InternVL-3.5 [34]</td><td>8B</td><td>35.8</td><td>60.4</td><td>21.8</td><td>20.7</td><td>38.9</td><td>31.4</td><td>42.6</td><td>35.8</td><td>34.9</td></tr><tr><td>Specialized 3D Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="9">VLM-3R [13]</td><td></td><td>35.6</td></tr><tr><td>Spatial-MLLM [37]</td><td>7B 6B</td><td>37.2 31.6</td><td>58.8 53.3</td><td>23.9 22.8</td><td>23.2 19.2</td><td>49.8 29.5</td><td>35.6 24.6</td><td>34.4 34.8</td><td>34.0</td><td>36.5 34.8</td></tr><tr><td>Cambrian-P [40]</td><td>7B</td><td>33.5</td><td>52.2</td><td>20.0</td><td>17.5</td><td>50.5</td><td>27.3</td><td>33.4</td><td>33.9</td><td>33.2</td></tr><tr><td>SenseNova-SI [5]</td><td>8B</td><td>35.4</td><td>57.9</td><td>20.6</td><td>22.9</td><td>44.4</td><td>29.8</td><td>32.9</td><td>34.9</td><td>39.8</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## 5. Experiments

## 5.1. Experimental Setup

Evaluated models. We evaluate nine recent VLMs spanning general-purpose and spatially specialized models. The general-purpose models are Qwen-3.6 [30] in its 35B-A3B and 27B variants, Qwen-3.5 [29], Qwen-3-VL [1], and InternVL-3.5 [34]. The spatially specialized models are VLM-3R [13], Spatial-MLLM [37], Cambrian-P [40], and SenseNova-SI [5]. We use the authors’ publicly released checkpoints and official inference implementations without task-specific fine-tuning. All model inference was conducted on an HPC cluster, using a single NVIDIA A100 or NVIDIA RTX PRO 6000 GPU per run, with 80 GB and 96 GB of GPU memory, respectively.

Evaluation protocol. We report per-question multiplechoice accuracy and macro-average accuracy, giving each question type equal weight despite Visibility Check having twice as many questions compared to others. Each model receives the 1 fps video prefix up to the query time $T _ { q } .$ For models with shorter context limits, frames are uniformly subsampled to fit the available context. Visual preprocessing and the full prompts are provided in Supp. 9.1 and 9.2.

## 5.2. Main Results

We present the performance of different models in Table 2 evaluated in three settings: (i) Text only, where the model receives only the question and options; (ii) Last Frame Only + Text, where the frame at the query time is additionally provided; and (iii) Video + Text, where the video prefix is provided instead of the last frame.

Text-only performance is near chance. Without video, models perform close to random guessing on nearly all question types. The main exception is Nearest Fixture, where several models achieve higher accuracy. This reflects object–fixture priors acquired during pretraining, such as associating a milk carton with a fridge.

Video evidence improves on text-only performance. Video improves macro accuracy for every model by 1.6– 10.5%, with the largest average gains on Visibility Check (+9.4%) and Nearest Fixture (+14.4%). Gains are less consistent for temporal grounding and smaller for spatial tasks. Despite these gains, overall performance remains low.

Difficulty varies across tasks. Relative to chance, VLMs perform best on Nearest Fixture (45.6% vs. 22.7%), followed by visual grounding (59.6% vs. 50.0%). In contrast, temporal grounding and spatial reasoning are substantially harder, exceeding chance by only 4.5% and 3.6% on average, respectively. Thus, models are better at recovering coarse semantic location than at precisely grounding past events or reasoning about remembered positions in 3D.

General-purpose models vs. spatially specialized models. General-purpose models perform better overall, led by Qwen-3.6-27B at 42.2% macro accuracy. This is not a scale effect: Qwen-3.5-9B (38.6%) outperforms all comparably sized spatially specialized models (31.6–37.2%). On the 3D spatial tasks, these models show no consistent advantage and remain modestly above random. We hypothesize that this reflects a mismatch between their spatial specialization and our setting: existing 3D training mainly targets scene geometry and viewpoint-induced spatial changes, while BEYOND3D additionally requires upstream temporal grounding of object relocations and retaining the spatial state while the object is not viewable. Their weaker temporal grounding and evidence retrieval ability creates a bottleneck that limits the benefit of 3D reasoning. More restrictive context budgets for several spatially specialized models may contribute to this gap.

Current-frame evidence is insufficient for out-of-sight reasoning. Providing only the query frame substantially improves Visibility Check, reaching 70.0% on average, but leaves temporal grounding, scene localization, and 3D spatial reasoning near chance. In contrast, full-video input substantially improves tasks that require recovering the target’s earlier state: Last Visible Time increases from 17.7% to 23.6%, Last Placement Time from 19.0% to 25.4%, and Nearest Fixture from 25.5% to 45.6%. This shows that the benchmark separates instantaneous visibility perception from reasoning over latent object states that must be recovered from video history.

## 5.3. Diagnosing Failure Modes

Unless otherwise stated, all subsequent analyses use the Video + Text setting, corresponding to the full-video benchmark setting.

![](images/2c30131536cd0f3acb8ea529840cc1d1375e9dff56643a75bf35d036369ca2f4.jpg)

Figure 5. Performance across temporal conditions. Left: Accuracy across short, medium, and long out-of-sight horizons. Right: Accuracy across early, middle, and late query times.  
![](images/9c02f4fd8c975c3193b8c35ac5f524f483188f49d3876f34d86fa481dd0e4fa7.jpg)  
Figure 6. Cumulative accuracy over temporal distance. Accuracy when progressively accepting the rounded ground truth (GT) and timestamp choices up to Near (±1–2 s), Medium (±3– 4 s), and Far (±5–6 s). Top: full-video models; Bottom: context-limited models (InternVL-3.5 / Cambrian-P / VLM-3R: 150 frames, Spatial-MLLM: 64 frames). Left: last-visible time; Right: last-placement time.

Temporal evidence retrieval is an upstream bottleneck. Performance reveals a strong dependence on how long the target remains out of sight (Fig. 5). Macro average accuracy decreases from 40.4% for short horizons to 34.8% for medium and 31.9% for long horizons. As the horizon grows, the target’s last observed state must be retained across more intervening activity, making it increasingly difficult to recover at query time. In contrast, accuracy varies non-monotonically with query time (34.9%, 38.4%, and 36.6% for early, middle, and late queries). Later queries provide more scene evidence but also longer histories, which may explain the middle-query peak. Thus, outof-sight horizon drives difficulty more than video length.

![](images/5c34547c606f02b7074dddbd133b27a980fafcbcbdc97526747f6ba4357b4f43.jpg)

Figure 7. Visibility-state estimation. Accuracy in determining whether the queried object is visible at query time, reported separately for objects that are not visible and visible.  
![](images/b7fa4e9bded0fff60433416301d42dffd9f8c18d0cc670a4ace5606cd9673fac.jpg)  
Figure 8. Prediction distributions for 3D spatial perception. Stacked bars show prediction frequencies (%). (a) object–camera direction (FL/FR/BL/BR: front/back left/right); (b) object–camera distance; (c) camera-aligned object–object direction; (d) object– object distance.

We next examine one potential upstream source of these errors, locating the interaction that determines the target’s latest state (Fig. 6). Since the four distractors lie in bins increasingly distant from the ground truth, the plot progressively counts predictions from farther bins as correct, causing random-choice accuracy to rise from 20% to 80%. If models found the relevant event but missed its exact timestamp, their curves would rise faster than this baseline. Most remain close to random, indicating weak preference for the correct temporal region and suggesting that models often

![](images/86374d5fb85bca6554891df83a8623b1aac681f810ce8236e9c9e46d6f399cfd.jpg)  
Figure 9. Effect of temporal cues on Qwen-3.6-27B. Left: Macro accuracy across out-of-sight (OOS) horizons, with shading indicating the improvement over the baseline. Right: Paired improvement for visibility, nearest-fixture, object–camera (O–C), and object–object (O–O) direction and distance questions. Error bars show 95% bootstrapped confidence intervals.

## retrieve another plausible visibility or movement event.

Out-of-sight state recognition is unreliable. Fig. 7 evaluates the Visibility Check separately for visible and out-ofsight targets. A robust model should perform well in both conditions, yet several models show strong asymmetries. InternVL-3.5 correctly predicts not visible for 87.8% of outof-sight targets but visible for only 32.9% of visible targets, while Cambrian-P shows the opposite bias. These opposing errors suggest model-specific visibility biases rather than uniformly weak visual recognition: some models tend to assume that previously observed objects remain absent, whereas others over-rely on current visual evidence. Since downstream questions require reasoning about an object specifically when it is no longer visible, such failures can corrupt the spatial state used for subsequent predictions.

Spatial predictions exhibit strong biases. As shown in Fig. 8, despite balanced ground-truth answer distributions for each 3D task, camera-direction predictions are skewed toward locations in front of the camera for every model, reaching 98.7% for Spatial-MLLM [37], while most models concentrate both distance tasks on the nearest bin, with Cambrian-P [40] assigning 100% and 99.4% of its Object– Camera Distance and Object–Object Distance predictions there, respectively. One explanation is that, when the remembered target state is uncertain, models fall back on spatial configurations supported by the direct visual evidence, favoring objects that are in front of and close to the camera.

## 5.4. Ablation Study

The preceding analyses suggest two upstream sources of error, namely locating the state-changing interactions and retaining the resulting state, and accounting for the target’s visibility status at query time. We test these hypotheses on Qwen-3.6-27B, the strongest model in our main evaluation. Temporal evidence retrieval is an upstream bottleneck. We first prepend a textual cue that lists all intervals in which the target object was moved, e.g., “The camera

Temporal cue: The camera wearer moved the blueberry boxduring these time intervals:

![](images/e696b2fddf7e27c3a605b9c632330b84de74b2326b64a9e938da0d5c83913303.jpg)  
Focus on these intervals when tracking the object. At<TIME 00:01:11.0 video 1>, assuming the previously movedblueberry boxremains at its last known position, in which direction is the blueberry box from your viewpoint? A. Back-leftB. Front-right C. Back-right D. Front-left "00:00:07 - 00:00:13: ...takes a container of blueberries out of the fridge and walks to kitchen counter... 00:00:24 - 00:00:54: ... handling the blueberry container, opening it, and then putting the lid back on." (Misses final relocation from counter to storage next to the fridge) X Object-reidentification "01:11: The person is ... washing the blueberries. The container is in their hands . Therefore, the blueberry box is .. in front of them."

![](images/76c1c7edbf81a7a011c3da54d3f8c9f649e697a496b90c4b1180e5044dd3b5bb.jpg)  
Question: Focus on these intervals when tracking the object. At<TIME 00:01:03.0 video 1>, assuming the previously moved food processing lidremains at its last known position, in which direction is the food processing lid from your viewpoint? A. Front-leftB. Front-right C. Back-left D. Back-right "At 00:23, the person is holding the black lid. At 00:25, they place the lid onto the blender jar. At 00:26–00:32, they are securing the lid...' "After this sequence, the blender (with the lid on it) is sitting on the kitchen counter. ... At 01:03, the camera is facing the sink." "Let's look at the spatial layout. ... Since it's on the counter in front of the wall, it's in the front' part of the room relative to the camera's  
Figure 10. Qualitative analysis. For each example, we show the provided temporal cues (blue), keyframes and Q&A, and reasoning traces. Correct and incorrect steps are marked in green and red. Top: the model misses the final relocation of the blueberry box and misidentifies it. Bottom: the model correctly tracks and grounds thefood processing lid but fails in camera-relative 3D reasoning.

wearer moved the blueberry box during these time intervals: <TIME 00:00:07.3 video 1> to <TIME 00:00:13.6 video 1>, . . . ”. The cue directs the model toward the state-changing interactions that determine the object’s latest location. As shown in Fig. 9, this intervention improves all downstream question types across outof-sight horizons, supporting temporal evidence retrieval as an important upstream bottleneck.

Visibility information further improves scene grounding. We next add an explicit textual statement that the target is not visible at query time, on top of the temporal cue. Nearest Fixture accuracy improves by 8%, but 3D spatial tasks gain only 0–2.4%. This suggests that visibility awareness helps recover the target’s scene location, but leaves most geometric errors unresolved.

## 5.5. Qualitative Analysis

To understand the errors remaining after simplifying temporal retrieval, we examine Qwen-3.6-27B’s reasoning traces under temporal cues with thinking enabled. Fig. 10 illustrates where failures arise along the reasoning chain. Three recurring patterns emerge: (i) Fine-grained evidence can still be missed. For a blueberry box, the model summarizes both provided movement intervals but misses the final relocation from the counter to a storage area next to the fridge. It therefore carries an outdated location forward to the query; (ii) Tracking errors can corrupt the remembered state. In the same example, the model conflates the queried box with the blueberries being washed and concludes that the container is in the person’s hands. The relevant interaction is present, but the target identity is not maintained over time; (iii) Spatial reasoning remains a downstream bottleneck. For a food processing lid, the model correctly follows the movements and recovers that the lid was left on the kitchen counter. It nevertheless maps this location incorrectly into the current camera viewpoint, predictingfront-right instead of the correct back-right.

## 6. Conclusions

We introduced a new VQA benchmark to evaluate outof-sight spatiotemporal reasoning in dynamic egocentric videos, addressing a key gap in literature. By querying relocated objects only after they are unobservable, and decomposing the task into eight questions and four capabilities, BEYOND3D tests whether VLMs can update an object’s spatial state, retain it beyond visibility, and reason from it in space and time. Experiments show that recent VLMs remain far from reliable, with errors compounding across retrieving relevant past events, maintaining object state over time, and reasoning from that state once the object is no longer visible. Future work should therefore move beyond current-view perception toward models that explicitly update and preserve persistent latent representations of dynamic scenes over time. We believe BEYOND3D provides a useful benchmark for measuring progress toward this capability in egocentric video understanding.

Acknowledgements We thank Xiaoxuan Cheng for assistance with executing experiments on the cluster.

## References

[1] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631, 2025. 2, 6

[2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923, 2025. 2

[3] Renee Baillargeon. Representing the existence and the lo-´ cation of hidden objects: Object permanence in 6- and 8- month-old infants. Cognition, 23(1):21–41, 1986. 2

[4] Neil Burgess. Spatial memory: How egocentric and allocentric combine. Trends in Cognitive Sciences, 10(12):551–557, 2006. 2

[5] Zhongang Cai, Ruisi Wang, Chenyang Gu, Fanyi Pu, Junxiang Xu, Yubo Wang, Wanqi Yin, Zhitao Yang, Chen Wei, Tongxi Zhou, Qingping Sun, Hui En Pang, Jiaqi Li, Oscar Qian, Zhiqian Lin, Xuanke Shi, Kewang Deng, Xiaoyang Han, Zukai Chen, Xiangyu Fan, Hanming Deng, Lewei Lu, Liang Pan, Bo Li, Ziwei Liu, Quan Wang, Dahua Lin, and Lei Yang. Scaling spatial intelligence with multimodal foundation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 7879–7890, 2026. 2, 6

[6] Jacob Chalk, Saptarshi Sinha, Dima Damen, Yannis Kalantidis, and Diane Larlus. Whareformer: Learning to track what is where in long egocentric videos. In European Conference on Computer Vision (ECCV). Springer, 2026. 2

[7] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brian Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In CVPR, pages 14455–14465, 2024. 2

[8] Pingyi Chen, Yujing Lou, Shen Cao, Jinhui Guo, Lubin Fan, Yue Wu, Lin Yang, Lizhuang Ma, and Jieping Ye. SD-VLM: Spatial measuring and understanding with depth-encoded vision-language models. In Advances in Neural Information Processing Systems (NeurIPS), 2025. 2

[9] An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang, and Sifei Liu. Spatial-

RGPT: Grounded spatial reasoning in vision-language models. In Advances in Neural Information Processing Systems (NeurIPS), 2024. 2

[10] Christopher Clark, Jieyu Zhang, Zixian Ma, Jae Sung Park, Rohun Tripathi, Sangho Lee, Mohammadreza Salehi, Jason Ren, Chris Dongjoo Kim, Yinuo Yang, Vincent Shao, Yue Yang, Weikai Huang, Ziqi Gao, Taira Anderson, Jianrui Zhang, Jitesh Jain, George Stoica, Ali Farhadi, and Ranjay Krishna. Molmo2: Open weights and data for visionlanguage models with video understanding and grounding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 28652– 28668, 2026. 2

[11] Jacob Cohen. A coefficient of agreement for nominal scales. Educational and Psychological Measurement, 20(1):37–46, 1960. 19

[12] Jakob Engel, Kiran Somasundaram, Michael Goesele, Albert Sun, Alexander Gamino, Andrew Turner, Arjang Talattof, Arnie Yuan, Bilal Souti, Brighid Meredith, et al. Project Aria: A new tool for egocentric multi-modal AI research. arXiv preprint arXiv:2308.13561, 2023. 4, 18

[13] Zhiwen Fan, Jian Zhang, Renjie Li, Junge Zhang, Runjin Chen, Hezhen Hu, Kevin Wang, Peihao Wang, Huaizhi Qu, Shijie Zhou, Dilin Wang, Zhicheng Yan, Hongyu Xu, Justin Theiss, Tianlong Chen, Jiachen Li, Zhengzhong Tu, Zhangyang Wang, and Rakesh Ranjan. VLM-3R: Vision language models augmented with instruction-aligned 3d re construction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 31054–31065, 2026. 2, 6

[14] Gabriele Goletto, Tushar Nagarajan, Giuseppe Averta, and Dima Damen. AMEGO: Active memory from long egocentric videos. In European Conference on Computer Vision (ECCV), pages 92–110. Springer, 2024. 2

[15] Kristen Grauman, Andrew Westbury, Eugene Byrne, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In CVPR, pages 18995–19012, 2022. 3

[16] Bo Gu, Zhikang Zhang, Zizhuang Wei, Zhenyuan Chen, Lingyun Li, and Zhuoyi Song. SpaceMind++: Toward allocentric cognitive maps for spatially grounded video MLLMs, 2026. 2

[17] Chanyoung Gwak, Yoonwoo Jeong, Byungwoo Jeon, Hyun seok Lee, Jinwoo Shin, and Minsu Cho. Cog3DMap: Multi view vision-language reasoning with 3d cognitive maps, 2026. 2

[18] Junsheng Huang, Shengyu Hao, Bo-Cheng Hu, Hongwei Wang, and Gaoang Wang. Understanding dynamic scenes in ego centric 4d point clouds. In Proceedings ofthe AAAI Con ference on Artificial Intelligence, pages 5031–5039, 2026. 3

[19] Robert Kowalski and Marek Sergot. A logic-based calculus of events. New Generation Computing, 4(1):67–95, 1986. 2

[20] Klaus Krippendorff. Content Analysis: An Introduction to Its Methodology. SAGE Publications, Thousand Oaks, CA, fourth edition, 2018. 19

[21] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. LLaVA-OneVision: Easy visual

task transfer. Transactions on Machine Learning Research, 2025. 2

[22] Chih-Ting Liao, Xi Xiao, Chunlei Meng, Zhangquan Chen, Yitong Qiao, Weilin Zhou, Tianyang Wang, Xu Zheng, and Xin Cao. SpaMEM: Benchmarking dynamic spatial reasoning via perception-memory integration in embodied environments. arXiv preprint arXiv:2604.22409, 2026. 3

[23] Lucy Lin, Ayush Jain, Yifan Liu, and Katerina Fragkiadaki. Qwen-3d: A generalist 3d vision-language model for spatial understanding. In European Conference on Computer Vision (ECCV), 2026. 2

[24] Matthias Minderer, Alexey Gritsenko, and Neil Houlsby. Scaling open-vocabulary object detection. In Advances in Neural Information Processing Systems (NeurIPS), 2023. 4, 18

[25] Toby Perrett, Ahmad Darkhalil, Saptarshi Sinha, Omar Emara, Sam Pollard, Kranti Kumar Parida, Kaiting Liu, Prajwal Gatti, Siddhant Bansal, Kevin Flanagan, Jacob Chalk, Zhifan Zhu, Rhodri Guerrier, Fahd Abdelazim, Bin Zhu, Davide Moltisanti, Michael Wray, Hazel Doughty, and Dima Damen. Hd-epic: A highly-detailed egocentric video dataset. In CVPR, pages 23901–23913, 2025. 2, 3, 4, 13, 17

[26] Chiara Plizzari, Shubham Goel, Toby Perrett, Jacob Chalk, Angjoo Kanazawa, and Dima Damen. Spatial cognition from egocentric video: Out of sight, not out of mind. In Int. Conf. 3D Vis., pages 1211–1221, 2025. 2

[27] Chiara Plizzari, Alessio Tonioni, Yongqin Xian, Achin Kulshrestha, and Federico Tombari. Omnia de EgoTempo: Benchmarking temporal understanding of multi-modal LLMs in egocentric videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 24129–24138, 2025. 3

[28] Kevin Qu, Haozhe Qi, Mihai Dusmanu, Mahdi Rad, Rui Wang, and Marc Pollefeys. Loc3R-VLM: Language-based localization and 3d reasoning with vision-language models, 2026. 2

[29] Qwen Team. Qwen3.5: Towards native multimodal agents. https://qwen.ai/blog?id=qwen3.5, 2026. Accessed: 2026-08-13. 2, 6

[30] Qwen Team. Qwen3.6-27B: Flagship-level coding in a 27B dense model. https://qwen.ai/blog?id=qwen3. 6-27b, 2026. Accessed: 2026-08-13. 2, 6

[31] Sahithya Ravi, Gabriel Herbert Sarch, Vibhav Vineet, Andrew D. Wilson, and Balasaravanan Thoravi Kumaravel. Out of sight, not out of context? egocentric spatial reasoning in VLMs across disjoint frames. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 16135–16150. Association for Computational Linguistics, 2025. 15

[32] Aleksandar Shtedritski, Christian Rupprecht, and Andrea Vedaldi. What does CLIP know about a red circle? visual prompt engineering for VLMs. In ICCV, pages 11987– 11997, 2023. 15

[33] Pavel Tokmakov, Allan Jabri, Jie Li, and Adrien Gaidon. Object permanence emerges in a random walk along memory. In International Conference on Machine Learning (ICML), pages 21506–21519. PMLR, 2022. 2

[34] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. InternVL3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025. 2, 6

[35] Yun Wang, Junbin Xiao, Han Lyu, Yifan Wang, Jing Zuo, Zhanjie Zhang, Hong Huang, Dapeng Wu, and Angela Yao. Keep it in mind: User-centric continual spatial intelligence reasoning in egocentric video streams. In Proceedings of the International Conference on Machine Learning (ICML). PMLR, 2026. 3

[36] Ziyang Wang, Yue Zhang, Shoubin Yu, Ce Zhang, Zengqi Zhao, Jaehong Yoon, Hyunji Lee, Gedas Bertasius, and Mohit Bansal. EgoMemReason: A memory-driven reasoning benchmark for long-horizon egocentric video understanding. arXiv preprint arXiv:2605.09874, 2026. 3

[37] Diankun Wu, Fangfu Liu, Yi-Hsin Hung, and Yueqi Duan. Spatial-MLLM: Boosting MLLM capabilities in visualbased spatial intelligence. In Advances in Neural Informa tion Processing Systems (NeurIPS), 2025. 2, 6, 8

[38] Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. Set-of-mark prompting unleashes extraordinary visual grounding in GPT-4V. arXiv preprint arXiv:2310.11441, 2023. 15

[39] Jihan Yang, Shusheng Yang, Anjali W. Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In CVPR, pages 10632–10643, 2025. 3

[40] Jihan Yang, Zifan Zhao, Xichen Pan, Shusheng Yang, Junyi Zhang, Bingyi Kang, Hu Xu, Shang-Wen Li, and Saining Xie. Cambrian-P: Pose-grounded video understanding. In European Conference on Computer Vision (ECCV), 2026. 2, 6, 8

[41] Shusheng Yang, Jihan Yang, Pinzhi Huang, Ellis Brown, Zihao Yang, Yue Yu, Shengbang Tong, Zihan Zheng, Yifan Xu, Muhan Wang, Daohan Lu, Rob Fergus, Yann LeCun, Li Fei-Fei, and Saining Xie. Cambrian-S: Towards spatial supersensing in video. In International Conference on Learning Representations (ICLR), 2026. 2

[42] Hanxun Yu, Xuan Qu, Lei Ke, Boqiang Zhang, Yuxin Wang, Jianke Zhu, and Dong Yu. Stream3D-VLM: Online 3D spa tial understanding with incremental geometry priors. In European Conference on Computer Vision (ECCV), 2026. 2

[43] Yuqian Yuan, Ronghao Dang, Long Li, Wentong Li, Dian Jiao, Xin Li, Deli Zhao, Fan Wang, Wenqiao Zhang, Jun Xiao, and Yueting Zhuang. EOC-Bench: Can MLLMs identify, recall, and forecast objects in an egocentric world? In Advances in Neural Information Processing Systems (NeurIPS), 2025. 3

[44] Ruosen Zhao, Zhikang Zhang, Jialei Xu, Jiahao Chang, Dong Chen, Lingyun Li, Weijian Sun, and Zizhuang Wei. SpaceMind: Camera-guided modality fusion for spatial reasoning in vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16811–16822, 2026. 2

[45] Yanguang Zhao, Jie Yang, Shengqiong Wu, Shutong Hu, Hongbo Qiu, Yu Wang, Guijia Zhang, Tan Kai Ze, Hao Fei,

Chia-Wen Lin, Mong-Li Lee, and Wynne Hsu. SCP: Spatial causal prediction in video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings, pages 7165–7175, 2026. 3

[46] Duo Zheng, Shijia Huang, Yanyang Li, and Liwei Wang. Learning from videos for 3D world: Enhancing MLLMs with 3D vision geometry priors. In Advances in Neural Information Processing Systems (NeurIPS), 2025. 2

# Long Time No See: Benchmarking VLMs for Out-of-Sight Spatiotemporal Reasoning in Egocentric Videos

Supplementary Material

## 7. Supplementary

Overview. In this supplementary material, we provide additional information, visualizations, and analyses that complement the main paper. Sec. 8 expands on the benchmark construction, including the question templates, answer and distractor generation, and dataset distributions. Sec. 9 provides implementation details for model inference, including visual preprocessing and the prompts used for the baseline and oracle interventions. Sec. 10 describes the visibility-track construction pipeline in detail, with additional visual illustrations of object-location inference and cross-view projection. Finally, Sec. 11 reports the results of the human validation of these visibility tracks.

## 8. Benchmark and Statistics

## 8.1. Question Templates and Answer Choices

Each question is constructed at a query time $T _ { q }$ for a stationary and out-of-sight object that was moved earlier in the video using natural language template with placeholders. In the templates of Tab. 3, [OBJECT] denotes the target object, [REF] denotes a visible reference object, and [TIME] denotes the query timestamp in the format <TIME HH:MM:SS.s video 1>.

## 8.2. Technical Details for Answer and Distractor Construction

Each question is constructed as a multiple-choice question in two stages: (i) derive the ground-truth answer, and (ii) construct the remaining options. All multiple-choice answers are randomly shuffled before being added to the benchmark.

## Ground-truth answer derivation

Temporal grounding. The correct answer is the timestamp of the relevant visibility transition or placement event, obtained from the visibility tracks and movement annotations.

Scene localization. For each placement event, the HD-EPIC [25] movement annotation provides the fixture closest to the object at the end of the movement. The fixture distribution is highly imbalanced: among 16,213 placements with a known fixture, 63.2% occur on counters, followed by the sink (9.7%), hob (6.7%), cupboard (5.1%), dishwasher (3.4%), and drawer (3.3%). Collapsing all counters into a single counter label would therefore introduce a strong answer bias and discard spatial detail.

The annotations contain 63 counter-surface instances across nine kitchens, represented only by non-semantic IDs such as counter 001. We manually map each instance to a unique, human-readable description using nearby fixtures, e.g., “counter area between the fridge and the hob”. We favor neutral landmarks to reduce object–location shortcuts; for example, “next to the microwave” is preferred over “next to the sink” when the latter may reveal the likely location of objects such as a sponge. Non-counter placements use the annotated fixture category directly.

3D spatial perception. We derive all 3D answers from the annotated world-frame object positions and the camera pose at query time $T _ { q }$ . Camera-relative questions express the target in the camera coordinate system at $T _ { q } ,$ , whereas objectrelative questions translate the origin to a confidently visible reference object while preserving the axis orientation defined by the camera pose at $T _ { q } .$

Let $c _ { q } ~ \in ~ \mathbb { R } ^ { 3 }$ denote the camera center at query time $T _ { q }$ in world coordinates, and let $R _ { q } ~ \in ~ S O ( 3 )$ denote the corresponding world-to-camera rotation. The target’s last known world-frame position $\ell _ { o } ( T _ { q } )$ is expressed in the camera frame as

$$
\begin{array} { r } { p _ { o } ^ { \mathrm { c a m } } = R _ { q } \left( \ell _ { o } ( T _ { q } ) - c _ { q } \right) . } \end{array}
$$

Thus, the camera-relative direction is determined by the orientation of $p _ { o } ^ { \mathrm { c a m } }$ p<sup>cam</sup> , while the camera-relative distance is

$$
d _ { o , \mathrm { c a m } } = \| p _ { o } ^ { \mathrm { c a m } } \| _ { 2 } .
$$

For object-relative questions, we select a distinct reference object r that is confidently visible at $T _ { q } ,$ , with worldframe position $\ell _ { r } ( T _ { q } )$ . We translate the coordinate origin from the camera center to the reference object while retaining the camera orientation:

$$
p _ { o | r } ^ { \mathrm { c a m } } = R _ { q } \big ( \ell _ { o } ( T _ { q } ) - \ell _ { r } ( T _ { q } ) \big ) .
$$

Equivalently, in homogeneous coordinates this transformation is

$$
\begin{array} { r l } { T _ { q , r } = \left[ { \cal R } _ { q } } & { { } - { \cal R } _ { q } \ell _ { r } ( T _ { q } ) \right] } \\ { \quad } & { { } 1 } \end{array} .
$$

Hence, $p _ { o | r } ^ { \mathrm { c a m } }$ is the vector from the reference object to the remembered target location, expressed along the cameraoriented axes at $T _ { q }$ . Its orientation determines the objectrelative direction answer, and

$$
d _ { o , r } = \left. p _ { o | r } ^ { \mathrm { c a m } } \right. _ { 2 } = \Vert \ell _ { o } ( T _ { q } ) - \ell _ { r } ( T _ { q } ) \Vert _ { 2 }
$$

Table 3. Question templates and answer choices. [TIME], [OBJECT], and [REF] denote the query timestamp, target object, and visible reference object, respectively.
<table><tr><td>Question</td><td>Exact template</td><td>Answer choices</td></tr><tr><td>Visual Grounding Visibility Check</td><td>At [T IME], is the previously moved [OBJECT] visible in the current frame?</td><td>No; Yes</td></tr><tr><td>Temporal Grounding Last Visible Time</td><td>Which timestamp is closest to when the [OBJECT]was last visible?</td><td>5 timestamps: HH:MM:SS --- N seconds before the end</td></tr><tr><td>Last Placement Time</td><td>The [OBJECT] was moved earlier in the video. Which timestamp is closest to when it last stopped being moved?</td><td>5 timestamps: HH :MM:SS --- N seconds before the end</td></tr><tr><td>Scene Localization Nearest Fixture (non-counter vari-</td><td>At [ TIME], based on the last known position of the [OBJECT] that was moved earlier, which 5 fixture types fixture type is closest to it?</td><td></td></tr><tr><td>ant) Nearest Fixture (counter variant)</td><td>At [TIME], based on the last known position of the [OBJECT] that was moved earlier, which 3-6 kitchen-dependent counter ar-</td><td></td></tr><tr><td>3D Spatial Perception</td><td>counter area is closest to it?</td><td>eas</td></tr><tr><td>rection</td><td>Object-Camera Di- At [ T IME], assuming the previously moved [OBJECT ] remains at its last known position, in which direction is the [OBJECT] from your viewpoint?</td><td>Front-right; Back-right; Front-left; Back-left</td></tr><tr><td>Object-Camera Distance</td><td>At [TIME], assuming the previously moved [OBJECT] remains at its last known position, what is the distance between the camera and where the [OBJECT]was left?</td><td>Under 1 m; 1 to under 1.5 m; 1.5 m or more</td></tr><tr><td>Object-Object Di- rection</td><td>At [TIME], assuming the previously moved [OBJECT] remains at its last known position, where is it relative to the [REF ] (marked in red in the current frame) from your viewpoint?</td><td>12 to 4:30 o&#x27;clock; 4:30 to 7:30 o&#x27;clock; 7:30 to 12 o&#x27;clock</td></tr><tr><td>Object-Object Dis- tance</td><td>At [TIME], assuming the previously moved [OBJECT] remains at its last known position, how far is it relative to the [REF] (marked in red in the current frame)?</td><td>Under 1 m; 1 to under 1.5 m; 1.5 m or more</td></tr></table>

determines the object-relative distance. The resulting directions and distances are mapped to the predefined categorical answer bins used by each question type.

## Option construction

Temporal distractors. Each temporal question contains the correct timestamp and four hard negatives. Candidate timestamps are grouped into four bins by absolute temporal distance from the correct answer, near (±1–2 s), medium (±3–4 s),far (±5–6 s), and veryfar (±7–30 s), and one negative is sampled from each bin. We prioritize timestamps corresponding to other visibility or movement events. If none are available, we sample a regular timestamp from the same bin.

Scene-localization distractors. For counter placements, distractors are other counter-area descriptions from the same kitchen. For non-counter placements, they are sampled from alternative fixture categories, such as counter, cupboard, dishwasher, or drawer.

Nearest Fixture is therefore the only question type whose number of options varies. Non-counter placements always yield five options, whereas counter placements yield between three and six, since a kitchen with few annotated counter areas admits fewer plausible alternatives. The chance level reported for this question type in the main results table is consequently not $1 / k$ for a fixed k, but the mean of $1 / k _ { i }$ over the 1,000 questions, which evaluates to 22.7%. All other question types have a fixed option count, giving the 50.0%, 20.0%, 25.0%, and 33.3% chance levels

of the remaining columns.

3D spatial options. Direction and distance are discretized into predefined, exhaustive categories, which directly define the answer choices.

## 8.3. Answer Distribution

To reduce the possibility of exploiting answer-frequency shortcuts, we control the correct-answer distributions for the temporal and spatial questions. For Last Visible Time and Last Placement Time, each question contains five timestamp choices. We sort these choices chronologically and balance which temporal position contains the correct answer: the correct timestamp is the earliest choice for 200 questions, the second earliest for 200, and so on up to the latest choice. Thus, each of the five chronological ranks occurs equally often as the correct answer. The 3D spatial perception questions are balanced as evenly as the sample size allows across their semantic answer classes. Object–Camera Direction has four classes and is exactly balanced at 250 questions each, while Object–Camera Distance, Object–Object Direction, and Object–Object Distance have three classes and are balanced at 334/333/333. Nearest Fixture is treated differently since its answers retain the naturally occurring fixture-label frequencies rather than being artificially balanced. The resulting distributions are shown in Supplementary Figs. 11 and 12.

![](images/1befc17ede4267b8b6b63d2f64d3ede7c394b8650dbc4bb621a8bb98ef69557f.jpg)  
Figure 11. Controlled correct-answer distributions. (a–b) Chronological answer-rank distributions for Last Visible Time and Last Placement Time. Chronological rank is obtained by sorting the five displayed timestamp choices from earliest to latest; rank 1 is the earliest and rank 5 the latest. Each rank is correct for exactly 200 of 1,000 questions in each step. (c–f) Semantic correct-answer distributions fo the four 3D spatial perception questions, grouped by answer meaning rather than the shuffled multiple-choice option letter. Panel (c) has four classes and is exactly balanced; panels (d–f) have three classes and are balanced to 334/333/333. FL/FR/BL/BR denote front/back left/right, and distance values are in metres. Dashed lines mark equal-share reference counts.

## 9. Technical Details for Model Inference

## 9.1. Visual Input Preprocessing

All videos are temporally sampled at 1 fps, resized to 448 × 448 pixels. Pixels outside the circular fisheye field of view are masked in black. Each sampled frame is annotated near the bottom-right corner, outside the visible camera field, with a timestamp token of the form <TIME HH:MM:SS.s video 1>. For the Object–Object Direction and Object– Object Distance questions, the model must reason about the spatial relationship between the out-of-sight target object o and a reference object r that remains visible at the query timestamp.

To unambiguously identify r, we follow prior work on visual prompting and egocentric spatial reasoning that uses overlaid markers to indicate the queried object [31, 32, 38]. We overlay an $8 \times 8$ red marker at its projected image location in the query-time frame only. The marker is chosen to be clearly visible after resizing while minimally occluding the surrounding visual content; restricting it to the querytime frame avoids providing additional information about the reference object’s trajectory. An example of the processed frame is shown in Fig. 13.

For each evaluation sample, the model receives the video prefix from t = 0 to the query time $T _ { q } ,$ inclusive. Models that can accommodate the full prefix receive all sampled frames, while for models with shorter context limits, frames are further uniformly subsampled to fit the available context. The frame at $T _ { q }$ is always retained to preserve the visual state at the query time.

## 9.2. Inference Prompt

Baseline setup. Each sample consists of a fixed system prompt, the input video, and a multiple-choice question with its answer options. We use the same system prompt for all models: “You are a helpful assistant trained to answer spatial and visual questions based on egocentric videos. Use the video to answer the question. The video is sampled at 1frame per second.”

The question and answer choices are provided using the following prompt:

“Question: [QUESTION]

Options:

A. [OPTION A]

![](images/41c7d07c6118e8de2814e3ea9e8ef75c1c964f0492bb1cebedf1eb32cec75205.jpg)  
Figure 12. Correct-answer distribution for Nearest Fixture. Nearest Fixture asks for the fixture type or counter area closest to the target object’s last known position. Bars show the number of questions for each correct label. Unlike the other question types, this distribution is not balanced but retains the naturally occurring fixture frequencies. All 47 labels occurring across the nine kitchens are shown, and the counts sum to the 1,000 out-of-sight anchors.

## B. [OPTION B]

Select the best option and output only its letter.”

Temporal-cue intervention. For the temporal-cue experiments, we prepend the annotated intervals during which the target object was moved: “Temporal cue: The camera wearer moved the [OBJECT] during these time intervals: $[ S T A R T _ { 1 } J$ to [END<sub>1</sub>]; . . . ; [START<sub>n</sub>] to $I E N D _ { n } J .$ Focus on these intervals when tracking the object.”

For example: “Temporal cue: The camera wearer moved the fork during these time intervals: <TIME 00:00:03.0 video 1> to <TIME 00:00:14.3 video 1>; <TIME

00:00:15.5 video 1> to <TIME 00:00:26.4 video   
1>. Focus on these intervals when tracking the object.”

Visibility-cue intervention. For the visibility-cue experiments, we additionally state that the target is not visible at query time. Specifically, downstream questions are rewritten into the following form, where [QUESTION BODY] is the template of Tab. 3 with its leading “At [TIME], assuming the previously moved [OBJECT] remains at its last known position,” clause removed, so that the clause is not stated twice: “At [QUERY TIME], the previously moved [OB-JECT] is no longer visible. Assuming it remains at its last known position, [QUESTION BODY]”

![](images/9c159ac8742d2900d5df0ef9dc10520aa2d4b3df03b82797633bc51aa35df8a5.jpg)  
Figure 13. Example preprocessed query frame. The timestamp is placed in the masked region outside the fisheye field of view. For object-relative spatial questions, the visible reference object is indicated by a red marker.

For example: “At <TIME 00:01:00.0 video 1>, the previously moved cloth is no longer visible. Assuming it remains at its last known position, in which direction is the cloth from your viewpoint?”

## 10. Technical Details for Visibility Track Construction

This section provides the implementation details for the visibility track construction procedure. Tracks are sampled at 1 fps and contain both the inferred object location and its visibility state.

## 10.1. Inferring Object Locations from Movement Annotations

HD-EPIC [25] annotates each object movement with its temporal interval and observations at the beginning and end of the movement. These observations include a 2D bounding box, segmentation mask, 3D object center, and associated scene fixture. Because intermediate object trajectories are not annotated, we infer the stationary object location using the following rules, illustrated in Fig. 14:

• Before the first annotated movement: The object is assumed to remain at the location recorded at the beginning of its first movement.

• During an annotated movement: The object is assigned in motion. No stationary location is assigned because its trajectory between the annotated endpoints is unknown, so the three-stage procedure below is not applied to these samples. We nevertheless treat the object as visible while it is being relocated $( v _ { o } ( t ) = 1 )$ , since it is in the camera wearer’s hands. Because they carry no stable location, they can never themselves be query anchors, and they are excluded from the audit in Sec. 11.

![](images/50d01bffbee057aa1e8abcde38e1519efd4df1318d07c8a03b38b8e8d179c256.jpg)  
Figure 14. Object location inference from HD-EPIC annotations. The object is held at its first annotated start location before the first movement, marked in motion during a movement, and held at the endpoint of its most recent completed movement thereafter.

• Between movements and after the final movement: The object is assumed to remain at the endpoint location of its most recently completed movement.

## 10.2. Determining Visibility

For each one-second sample at which the object has an inferred stationary location, we determine its visibility using a three-stage procedure:

1. Field-of-view projection: We determine whether the object’s inferred location falls within the current camera view.

2. Fixture-aware geometric occlusion: We check whether a scene fixture blocks the camera’s view of the object.

3. Detection-based visual confirmation: We inspect the corresponding video frame to confirm whether the object is visible at the expected location using an openvocabulary detection method.

Stages 1 and 2 act on individual samples. Consecutive samples sharing the same stage-1/stage-2 outcome are then merged into a continuous interval, and Stage 3 operates at interval granularity, where it scores each candidate interval by the fraction of its frames in which the detector confirms the object, and assigns the resulting state to the interval as a whole. Every sample therefore ends up with exactly one state, and every interval is state-homogeneous by construction.

Stage 1: Field-of-view projection. For stationary objects, projecting only the annotated 3D centroid is unreliable near image boundaries, as objects may remain partially visible even if their centroid is off-screen. To account for this, we determine if an object is within the field of the camera view using the following procedure:

• Approximating spatial extent: We represent the object using nine points from its annotated endpoint bounding box: the center, four corners, and four edge midpoints. These points are back-projected onto a plane that passes through the object’s 3D centroid and is parallel to the camera’s image plane, forming a stable 3D footprint in world coordinates that is reused while the object remains stationary.

• Camera projection and masking: At each sampled time point, we reproject the nine points into the current camera view using the corresponding camera pose provided by the HD-EPIC annotations and the FISHEYE624 fisheye camera model of Project Aria [12]. We use devignetted RGB frames, in which lens-vignetting effects are corrected and valid image content is limited to the circular region within the frame. The same frames and validimage mask are used in later detection stages and during benchmark evaluation to ensure consistency. A projected point is considered invalid if it lies behind the camera or outside this region.

• State assignment: The object is assigned in view if more than 50% of the points (i.e. at least five of the nine support points) are valid; otherwise, it is assigned the state out of view.

Stage 2: Fixture-aware geometric occlusion. For each in-view object, we cast rays from the camera center toward all nine support points and intersect them with the kitchen mesh. A ray is considered blocked if its first intersection lies at least δ = 10 cm closer to the camera than the corresponding target support point. We then calculate the fraction of rays that are blocked. If at least 50% of the rays are blocked, the object is considered geometrically occluded.

The static digital twin does not track the real-time state of movable parts, such as open refrigerator doors or extended drawers. Consequently, an object placed inside an open cabinet might falsely appear occluded by the digital twin’s default closed-door geometry. If the majority blockage is attributed to the openable fixture containing the object, we instead assign fixture ambiguous and defer the final visibility decision to the detection-based visual confirmation stage.

Stage 3: Detection-based visual confirmation. Samples that lie within the camera’s field of view and are not classified as occluded are verified using an open-vocabulary object detector. We process the corresponding frames at 1 fps, apply the same valid-image mask used in the previous stages, and query OWLv2 [24] with the object’s name. A detection is matched to the target only if its bounding box, enlarged by 20 pixels, contains the object’s projected anchor point. If multiple detections satisfy this condition, we select the one whose center is closest to the target object. This geometric constraint helps distinguish between multiple objects of the same category.

![](images/3c10e8deaac62e41fb156be94afb9180d4df19afb236769ef3459aa08de81603.jpg)  
Figure 15. Detected-fraction distribution of scored intervals. Share of intervals (%) whose detected fraction of tested frames falls in each 10% bin (n = 445,301). The distribution is strongly U-shaped: 65.2% of intervals fall in the lowest bin and 20.1% in the highest, leaving under a sixth of intervals spread across the middle.

An interval is assigned detected visible if the object is detected in at least half of its testable frames. This threshold is not a sensitive parameter as the detected fraction is strongly U-shaped (Fig. 15), with 65.2% of intervals in the lowest 10% bin and 20.1% in the highest, so only a small minority of intervals lie near the cut-off. If the object is not detected and its previous state was fixture ambiguous, the interval is assigned occluded. Otherwise, it is assigned the internal state visually unconfirmed.

HD-EPIC object-association names are free-form and may contain instance indices, spelling errors, annotation notes, or descriptions that do not refer to a single visually identifiable object. We therefore manually curate the 3,826 distinct raw names. Of these, 1,713 are retained unchanged, 1,563 are rewritten as visually groundable object names, and 550 cannot be mapped to a single object. For each groundable name, the detector uses an ordered sequence of up to three queries, from specific to coarse; for example, air fryer drawer → drawer → air fryer. A coarser query is used only if all more specific queries fail at the projected location. Ungroundable associations are assigned the not groundable state. These are not passed to the detector and are excluded from the constructed visibility track and the visibility-recall calculations. These objects will not be used for building the benchmark as well.

Table 4. Human audit of the visibility tracks. We compare the total number of annotated marks per visibility track state with the number of marks judged to be visible. These correspond to disagreements for the first three states, which the pipeline labels not visible, and agreements for detected visible. Acc. is computed as the percentage of agreeing marks. The pipeline makes no claim for not groundable. The four scored states sum to the 4,102 marks used for scoring while the 4 in motion marks among the 4,176 gold marks are omitted, as the pipeline derives no visibility evidence for them.
<table><tr><td>Pipeline state</td><td></td><td>#Marks #Visible Acc. (%)</td><td></td></tr><tr><td>out_of_view</td><td>1,293</td><td>105</td><td>91.9</td></tr><tr><td>occluded</td><td>291</td><td>11</td><td>96.2</td></tr><tr><td>visually-unconfirmed</td><td>1,510</td><td>372</td><td>75.4</td></tr><tr><td>detected_visible</td><td>1,008</td><td>819</td><td>81.2</td></tr><tr><td>not_groundable</td><td>70</td><td>1</td><td>一</td></tr></table>

## 11. Human Validation of the Visibility Tracks This section details the visibility track audit.

Audit protocol. We draw videos round-robin over participants and stratify frame times into uniform bins within each video, preferring frames that contain at least three in-view objects. The audit covers 30 videos and all nine participants. Objects whose track is defined at the sampled timestamp are drawn as markers at their projected position. Annotators click the markers they can see, so unmarked objects are recorded as not visible, and a marker can also be flagged unsure, which abstains from both the majority vote and the agreement statistics. The pipeline state and the detector output are hidden, so the annotator judges only the frame and the marker. Judgments are matched to the track state within a 1 s tolerance. Two properties of this design matter when reading the rates below. Frames are biased toward objectpopulated moments rather than drawn uniformly at random, and frames dense with objects are capped at 15 markers, drawn at random among that frame’s objects, so very dense frames are represented by a subset of their objects. The sampler has no knowledge of the pipeline states, so the perstate coverage in Table 4 is an outcome of this procedure rather than part of its design.

Three annotators completed 449 frames in 4.8 hours, of which 436 pass the quality filter below, giving 5,391 mark judgments over 344 distinct frames. We aggregate multiply rated marks by majority vote and discard 12 marks that all raters called unsure and 12 that end in a tie, leaving 4,176 gold marks. Of these, 70 are not groundable and 4 are in motion. For neither state does the pipeline derive a visibility decision from the geometric and detection evidence that this audit tests. Therefore both cases are excluded from scoring and 4,102 marks remain.

Quality control. We discard frames on which an annotator spent less than 2 s, since the markers cannot be inspected in that time. Because unmarked objects count as not visible, a rushed frame yields confident wrong labels rather than missing data, so filtering on dwell is necessary. This removes 13 of 449 frames. Median dwell per frame is 38.5, 17.4, and 16.9 s. Splitting each session into thirds, median dwell is lower in the final third than in the first for al three annotators (e.g. 42.9 → 34.7 s), but the rate at which they call objects visible stays within 2.7 points of its session mean for all three. Faster judgments later in a session are therefore not systematically more permissive.

Inter-annotator agreement. 50 frames (634 marks) carry ratings from more than one annotator. Krippendorff’s α [20] is 0.76, the raters are unanimous on 520 of these marks (82.0%), and 12 reach no majority, leaving 622 multiply-rated marks in the gold set. Pairwise Cohen’s κ [11] is 0.86, 0.75, and 0.68, computed on the 550, 583, and 552 marks that each pair rated without either annotator flagging unsure. Part of the disagreement is a threshold effect. On the shared marks the three annotators call the object visible at rates of 34.4%, 31.5%, and 26.1%, and the annotator with the lowest rate is the one involved in both of the lowest pairwise κ values.

Overall metrics. Table 5 scores the pipeline against the majority human vote over the 4,102 scored marks, counting every mark once. Confidence intervals come from a cluster bootstrap over whole videos (30 clusters, 2,000 resamples) and therefore account for the correlation between marks drawn from the same video. Restricting scoring to the unanimous multiply-rated marks improves every metric, so label ambiguity makes the headline numbers conservative rather than flattering. Of the 520 unanimous marks, 518 fall in a state for which the pipeline makes a visibility claim and are therefore scored.

Where the errors are. The 4,102 scored marks contain 488 false negatives and 189 false positives. Of the false negatives, 372 are visually unconfirmed, meaning the object is in view, unoccluded by static geometry, and legible to a human, but OWLv2 fails to confirm it, returning a matching box in fewer than half of the tested frames. The remaining 105 and 11 fall in out of view and occluded, whose labels rest on the projected footprint and the scene mesh. Occlusions caused by the object’s own containing fixture are additionally required to fail a detector check before the occluded label is kept.

This asymmetry follows from the design. Detection can only demote an in-view, unoccluded sample to visually unconfirmed or leave a fixture-ambiguous sample occluded, so detector failures cost recall rather than precision. We keep visually unconfirmed separate from out of view and occluded throughout because it is the unreliable one, and query anchors are drawn only from the latter two.

Table 5. Pipeline versus majority human vote. Metrics over raw mark counts with 95% cluster-bootstrap confidence intervals over videos. The last column restricts scoring to the 518 scored marks among the 520 multiply-rated marks on which the annotators are unanimous.
<table><tr><td>Metric</td><td>All marks</td><td>95% CI</td><td>Unanimous</td></tr><tr><td>Accuracy</td><td>83.5</td><td>[80.7, 85.8]</td><td>87.6</td></tr><tr><td>Precision</td><td>81.2</td><td>[76.9, 85.2]</td><td>82.2</td></tr><tr><td>Recall</td><td>62.7</td><td>[54.9, 68.5]</td><td>69.3</td></tr><tr><td>F1</td><td>70.8</td><td></td><td>75.2</td></tr><tr><td>Balanced acc.</td><td>78.0</td><td>[74.1, 81.0]</td><td>81.9</td></tr><tr><td>Audited marks</td><td>4,102</td><td></td><td>518</td></tr></table>

Table 6. Audit results per participant. Percentages over raw mark counts, with Marks the number of scored marks.
<table><tr><td>Part.</td><td>Marks</td><td>Acc.</td><td>Prec.</td><td>Rec.</td><td>Bal. acc.</td></tr><tr><td>P01</td><td>474</td><td>84.0</td><td>77.7</td><td>63.0</td><td>77.8</td></tr><tr><td>P02</td><td>426</td><td>81.7</td><td>75.3</td><td>57.5</td><td>74.7</td></tr><tr><td>P03</td><td>323</td><td>80.8</td><td>65.9</td><td>61.4</td><td>74.7</td></tr><tr><td>P04</td><td>654</td><td>87.9</td><td>89.4</td><td>69.8</td><td>83.0</td></tr><tr><td>P05</td><td>204</td><td>81.9</td><td>93.6</td><td>69.5</td><td>82.2</td></tr><tr><td>P06</td><td>557</td><td>82.9</td><td>79.9</td><td>66.8</td><td>79.1</td></tr><tr><td>P07</td><td>522</td><td>84.9</td><td>77.2</td><td>62.4</td><td>77.8</td></tr><tr><td>P08</td><td>479</td><td>74.7</td><td>78.8</td><td>39.4</td><td>66.8</td></tr><tr><td>P09</td><td>463</td><td>89.2</td><td>88.4</td><td>74.8</td><td>85.2</td></tr><tr><td>All</td><td>4,102</td><td>83.5</td><td>81.2</td><td>62.7</td><td>78.0</td></tr></table>

Variation across participants. Balanced accuracy ranges from 66.8% (P08) to 85.2% (P09) in Table 6. Precision and recall vary by a comparable amount across kitchens, from 65.9 to 93.6% and from 39.4 to 74.8%, which is expected given that both are set by how well the open-vocabulary detector handles that kitchen’s objects. The weakest case is P08, where one of the three audited videos (P08-20240618- 171546) contributes zero true positives, meaning the detector confirmed none of the objects that annotators could see there.