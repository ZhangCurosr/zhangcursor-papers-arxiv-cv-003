![](images/5e366aaa6ba85c3c7ee5bf7b3cac4c9de8851bdc1237bb4a9270a37b269b9d16.jpg)

# MaLiang-Harness: A Programmable Path to Image and Video Generation

Haoyu Zhao<sup>1,2,†</sup>, Zihao Zhang<sup>1,2</sup>, Xudong Wang<sup>1</sup>, Jiaxi Gu<sup>3</sup>, Zuxuan Wu<sup>2</sup>, Yu-Gang Jiang<sup>2</sup>, Shuicheng Yan<sup>1</sup>

<sup>1</sup>National University of Singapore <sup>2</sup>Fudan University <sup>3</sup>Tencent

## Abstract

Executable programs offer explicit control over how images and videos are constructed, but generating runnable code is only the beginning of visual creation. A program can execute correctly while violating the requested composition, appearance, or motion. We define this discrepancy as the Program-to-Visual (P2V) gap and introduce MALIANG-HARNESS, a unified framework for organizing MLLM-driven visual generation into a persistent process of construction, inspection, and revision. Its central design is to make the evolving visual program, its construction history, and its verification share a common revision reference. We define the Persistent Executable Generation (PEG) state as preserving programs and task context. Traceable Generation Process (TGP) connects edits to rendered evidence, and Revision-aware Editing and Verification (REV) supports restoration and checks the current revision before completion. Together, these mechanisms coordinate planning, execution, and visual feedback across rendering backends. We evaluate 11 powerful closed-source MLLMs on MaLiang-IBench and four on MaLiang-VBench, measuring generation success, visual quality, and computational cost. GPT-6-Astra achieves 100% generation success on both benchmarks, with 96.0% of image tasks and 76.9% of video tasks meeting all quality thresholds. The comparison also reveals a mismatch between general capability scores and visual generation performance, with similarly scored models differing substantially in their ability to satisfy visual requirements. MaLiang-Harness provides a systematic basis for studying how MLLMs translate executable code into visual outcomes, exposing both the potential of programmable generation and the limitations of general benchmarks as predictors of this ability. The project is available at https://github.com/gulucaptain/MaLiang-Harness.

## 1 Introduction

“The idea becomes a machine that makes the art.”

## — Sol LeWitt

Over the past decade, advances in high-quality image and video generation have largely followed the paradigm of direct visual synthesis. From generative adversarial networks to diffusion- and flow-based models [Ho et al., 2020, Lipman et al., 2022], these approaches learn the distribution of visual data and directly synthesize pixels or latent visual representations [Rombach et al., 2022]. Although this paradigm has achieved remarkable success, its underlying generation process typically remains implicit. In contrast, the rapidly improving capabilities of multimodal large language models (MLLMs) in semantic understanding, reasoning, and code generation are making a programmable path to image and video generation increasingly viable: a model expresses creative intent as an executable visual program, which a renderer subsequently converts into an image or video.

![](images/51ebe01652bb6f13547a1fd8327ab8f7973e327d93ecee94fbc364831144ffa2.jpg)  
Hand-drawn cats turn photoreal shoes, a hat, an umbrel a, and a suitcase into a transport port, then race between them as the shoe-boat departs and the umbrel a airship takes off.

Figure 1: MaLiang-Harness aims to explore a programmable path to image and video generation, enabling diverse image synthesis, visual reasoning and editing, and creative video generation.

Table 1: Comparison of creative-style coverage and generation capabilities. We qualitatively compare MaLiang-Harness with existing image (T2I) and video (T2V) generation models along two dimensions: 1) creative-style coverage spanning line and flat art, 2D cartoons, pixel art, stylized 3D, mixed media and craft, and photorealism; and 2) generation capabilities including controllability over visual content and spatiotemporal structure, editability of the generated artwork, and traceability of the creation process. MaLiang-Harness supports diverse visual styles through executable visual programs, while providing explicit control, targeted editing, and an inspectable construction process. ○ Supported.  Partially supported. ○ Not established.
<table><tr><td>Type</td><td>Method</td><td>GI Line&amp;Flat Art</td><td>2 2D Cartoon</td><td>出 Pixel Art</td><td>Stylized 3D</td><td>《 Mixed &amp; Craft</td><td>à Photo- realistic</td><td>t Contro- llability</td><td>G Edita- bility</td><td>Q Process Traceability</td></tr><tr><td>Image</td><td>T2I Ours</td><td>● ●</td><td></td><td></td><td></td><td></td><td>0</td><td></td><td></td><td>O ●</td></tr><tr><td>Video</td><td>T2V Ours</td><td>● ●</td><td>● ●</td><td>● ●</td><td></td><td>●</td><td>● 0</td><td>● ●</td><td></td><td>O ●</td></tr></table>

However, a program can execute correctly yet produce an image or video that fails to satisfy the user’s intent. An object may appear in the wrong place, or an animated event may occur at the wrong time. We refer to this discrepancy between program-level correctness and visual requirement satisfaction as the Program-to-Visual (P2V) Gap. Bridging this gap requires the model to reason about the visual consequences of its code. Its initial plan must account for the choice of visual representation and rendering backend. Once the code is executed, the rendered output provides evidence for revising both the program and the plan. This process must preserve the artwork across iterations so that the model can address unmet requirements while retaining access to earlier versions. Each revision also calls for renewed visual assessment because a change can affect requirements that were previously satisfied. These demands motivate a stateful harness that connects creative intent to executable visua representations and supports continued control over the artwork through visual feedback.

To address this challenge, we introduce MaLiang-Harness, a unified interface connecting creative intent to multiple executable visual representations. It unifies programmable image and video generation through a shared protocol for state management, rendering, and revision-aware verification. Through this interface, an MLLM selects suitable rendering backends and exercises explicit control over the artwork through visual feedback. Code directly specifies how visual content is constructed, from spatial composition to motion over time. Optional image assets complement this programmatic structure when rich appearance is difficult to express through code alone. Images and videos share the same creation framework: at a fixed program revision, an image is rendered at a specified content time, while a video samples the temporal behavior encoded by the program. Table 1 summarizes the creative-style coverage and functional capabilities

Furthermore, MaLiang-Harness organizes creation into a loop in which the MLLM plans and generates code, then uses execution results and visual feedback to guide revision. Three designs support this process: 1) Persistent Executable Generation (PEG) State maintains the executable definition of the evolving artwork across iterations. It brings programs and assets together with their spatiotemporal organization, task requirements, the current plan, and revision index. This gives the model a stable object to inspect and modify as creation progresses. 2) Traceable Generation Process (TGP) makes the construction of the artwork inspectable through explicit programs and recorded operations. Intermediate renderings expose the visual effects of these operations, helping the model and the user relate an observed discrepancy to the components or changes that may have produced it. 3) Revision-aware Editing and Verification (REV) allows the model to restore earlier artwork content as a new revision while retaining current task requirements. Observations and verification results are associated with specific revisions, and every committed update requires renewed assessment before completion. We demonstrate that these designs enable the MLLM to translate visual feedback into targeted revisions of the evolving artwork, helping bridge the P2V gap.

We evaluate 11 MLLMs on MaLiang-IBench and four on MaLiang-VBench. GPT-6-Astra [OpenAI, 2026b] achieves a 100% generation success rate on both benchmarks. Its outputs meet every quality threshold on 96.0% of image tasks and 76.9% of video tasks, compared with 86.0% and 38.5% for

![](images/e5288051d9871b72534808ae2097bd56a2a20aacb94b25fd907e1d8633db3125.jpg)  
Figure 2: Model rankings on MaLiang-IBench. Ranks are computed across 11 models; better <sup>Ranks</sup> <sup>do</sup> <sup>not</sup> <sup>show</sup> <sup>effect</sup> <sup>size.</sup> <sup>Early</sup> <sup>failures</sup> <sup>can</sup> <sup>reduce</sup> <sup>cost;</sup> <sup>budgets</sup> <sup>and</sup> <sup>feedback</sup> <sup>differ</sup> <sup>across</sup> <sup>evaluated</sup> <sup>configurations.</sup>ranks lie farther from the center, with ties averaged. GPT models achieve 92–100% generation success, versus 12–40% for DeepSeek and Kimi. Mean quality scores use successful outputs (counts in parentheses). Cost ranks favor less time and fewer tokens.

GPT-5.6-Sol. These percentages use the full task set of each benchmark as the denominator. These results distinguish successful execution from satisfaction of visual requirements. Fig. 2 compares 11 models on MaLiang-IBench across visual quality, generation success, and computational cost, revealing different model rankings across these dimensions. Fig. 1 complements these comparisons with examples of image generation, visual understanding and editing, and video generation.

Our main contributions are:

• We introduce and define the Program-to-Visual (P2V) gap as the discrepancy between program-level correctness and visual requirement satisfaction, motivating a stateful formula tion of visual program generation through construction, inspection, and revision.

• We introduce MaLiang-Harness, a unified framework for programmable image and video generation. Persistent Executable Generation state, a Traceable Generation Process, and Revision-aware Editing and Verification support continued refinement, inspection of construction histories, and verification of the current output.

• We evaluate 11 MLLMs on MaLiang-IBench and four on MaLiang-VBench, characterizing differences in generation success, visual quality, and computational cost. Comparison with public general-capability scores shows that similar benchmark performance can correspond to substantially different visual generation outcomes.

## 2 Related Work

Image and Video Generation. Diffusion models synthesize images through learned denoising processes [Ho et al., 2020], with latent diffusion reducing the cost of high-resolution synthesis [Rombach et al., 2022]. Flow matching provides a related framework for learning continuous generative dynamics [Lipman et al., 2022]. Video Diffusion Models and Stable Video Diffusion extend learned visual synthesis to temporal content [Ho et al., 2022, Blattmann et al., 2023]. These approaches also support explicit conditioning: ControlNet, for example, introduces spatial guidance through inputs such as edges and depth [Zhang et al., 2023]. Our focus is on the representation through which visual content is constructed and revised. MaLiang-Harness maintains an executable artwork whose geometry and temporal behavior can be inspected and edited.

Executable Visual Representations. Prior work establishes several ways to connect language models with executable visual representations. VISPROG generates modular programs for visual reasoning and image editing, exposing intermediate results as inspectable rationales [Gupta and Kembhavi, 2023]. Design2Code evaluates the translation of webpage screenshots into renderable implementations [Si et al., 2025]. In three-dimensional graphics, BlenderAlchemy combines a visionbased edit generator with a state evaluator to search for edits that realize a user’s design intent [Huang et al., 2024]. These studies demonstrate that programs can mediate visual understanding and construction. MaLiang-Harness builds on this premise by organizing image and video creation around a persistent artwork state across several rendering backends.

![](images/730d55bcb09132c737d0e99308f39b316efa86275fef284404b26bfa90737709.jpg)  
Figure 3: Overview of the MaLiang-Harness for image and video generation. Persistent Executable Generation (PEG) state maintains the program and task context across revisions. Traceable Generation Process (TGP) links generation operations to state changes and visual results. Revisionaware Editing and Verification (REV) supports revision from historical states and assesses the current output using evidence from the same revision.

Agent Harnesses. Code as Agent Harness provides a broad account of code as infrastructure for reasoning, action, and stateful execution [Ning et al., 2026]. Show-Harness demonstrates how a semantic interface can connect VLM decisions to executable actions across robot embodiments [Chen et al., 2026]. Within visual generation, OmniHarness learns reusable symbolic policies from verified executions and uses intermediate feedback for refinement and recovery [Xu et al., 2026b]. MaLiang-Harness manages executable artwork through interfaces and feedback, linking visual construction to its edit history and revision-specific evidence.

## 3 MaLiang-Harness

MaLiang-Harness explores an alternative path to image and video generation in which an MLLM translates a prompt p into executable visual programs through a unified generation interface, as shown in Fig. 3. The framework uses MLLM planning and code generation to construct expressive visual compositions that renderers turn into images and videos, without requiring a diffusion or flow-matching process. Visual feedback guides iterative refinement of the generation under explicit spatial and temporal control. The carefully designed persistent executable generation state, traceable generation process, and revision-aware editing and verification associate each operation with the states it connects and each visual assessment with the revision that produced its evidence.

## 3.1 Unified Programmatic Visual Generation Interface

MaLiang-Harness provides a shared interaction protocol across rendering backends (e.g., Canvas, SVG, Scene2d, or Three.js) that supports state inspection and program or asset editing alongside rendering and requirement review. State updates identify the revision they modify, while execution results and rendered observations are linked to the corresponding revisions. Each backend retains its native executable representation, allowing the harness to express visual content through drawing and animation code within a common generation process.

Given a prompt $p$ and output specification ω, MaLiang-Harness plans the appearance, spatial composition, and temporal dynamics, selects an executable representation and compatible backend b, and generates drawing and animation code. The program may incorporate user-provided assets or, when the desired appearance is difficult to construct through code, assets obtained through image generation or search. The backend renders the resulting representation into an image or video that the MLLM evaluates against the task requirements, grounding subsequent code revisions in observed visual discrepancies rather than execution success alone.

Persistent Executable Generation (PEG) State. The harness maintains a persistent executable generation state that provides a common reference for planning, execution, and revision. At revision k, we define:

$$
S _ { k } = ( P _ { k } , A _ { k } , Z _ { k } , C _ { k } , k ) ,\tag{1}
$$

where $P _ { k }$ denotes the visual program together with its backend identifier, and $A _ { k }$ contains any associated assets. Program and asset contents are retained with content hashes, while ω specifies the output dimensions and format together with the rendering seed, and any applicable video timing parameters. $Z _ { k }$ describes the spatial composition and temporal dynamics through explicit scene attributes or definitions embedded in $P _ { k }$ , without requiring a separately maintained scene representation. The generation context $C _ { k }$ retains $p , \omega _ { \mathrm { { : } } }$ user-supplied requirements, and the current generation plan. Planning updates may append requirements but cannot remove or weaken existing ones. Changes to inherited requirements in a separate editing task must cite the user’s new instruction.

Initial state. The initial state $S _ { 0 }$ contains $p$ and ω and may have no executable content, which is refined through subsequent edits. We distinguish updates to this state from the visual rendering:

$$
S _ { k + 1 } = \mathcal { E } ( S _ { k } , a _ { k } ) , \qquad I _ { k } ( t ) = \mathcal { R } _ { b } ( S _ { k } , t ; \omega ) ,\tag{2}
$$

where $\mathcal { E }$ commits an edit $a _ { k }$ after validating its inputs and checking that its source revision is current. Each commit, including restoration of historical content, receives the next unused revision index within the run. The harness retains the preceding snapshot so that subsequent generation can revisit an earlier version rather than overwrite its executable definition.

## Algorithm 1: Generated Canvas program for T2I.

![](images/61a6ba50fd0dd926af0e2187188f2cd4c0ff8d7609bbd40bbfcfc8b55cf96921.jpg)  
Figure 4: Example of code-driven image synthesis. The “botanical portrait” is rendered from the visual program $P _ { k }$ of a PEG state without image assets. Algorithm 1 summarizes how the program specifies geometry, appearance, and spatial composition through Canvas drawing operations.

```latex
Require: Canvas $1 0 2 4 \times 5 7 6 ,$ , seed s
1: $\mathbf { c } \gets ( 5 1 2 , 2 8 8 )$
2: $R _ { \ell } \gets \mathrm { { P R N G } } ( \mathrm { { h a s h } } ( s , \ell ) )$ for each layer ℓ
3: Fill the canvas with white.
4: for $i = 0 , \ldots , 2 5$ do
5: $\theta _ { i }  2 \pi i / 2 6 + \epsilon _ { i } , \quad r _ { i }  1 6 7 + 1 8 u _ { i }$
6: $\mathbf { p } _ { i }  \mathbf { c } + r _ { i } ( \cos \theta _ { i } , \sin \theta _ { i } )$
7: Draw a curved branch at $\mathbf { p } _ { i } ;$ add leaves and veins.
8: end for
9: Draw Bézier paths for hair, face, and shoulders.
10: Shade the face; add closed eyes and hair strands.
11: for each rose at a predefined position do
12: for $k = 0 , \ldots , 4$ do
13: $n _ { k } \gets [ 8 , 7 , 6 , 5 , 4 ] _ { k }$
14: for $j = 0 , \ldots , n _ { k } - 1$ do
15: $\phi _ { k , j }  2 \pi j / n _ { k } + \delta _ { k , j }$
16: Draw a rotated Bézier petal at angle $\phi _ { k , j }$
17: end for
18: end for
19: end for
20: Add daisies, small blossoms, and texture.
21: Generation finished.
```

Renderable state. For a renderable state, $\mathcal { R } _ { b }$ evaluates the visual program at content time t under ω using the selected backend b. An image corresponds to evaluation at a specified time, whereas a video is obtained by sampling the temporal evolution encoded in the same representation. The index k therefore tracks revisions of the generation state rather than progression through the generated content; an update to the generation plan can advance k without changing the rendered output. Fig. 4 pairs a visual program with its rendered output at a fixed PEG revision, showing how the executable representation specifies the resulting image composition.

The harness records each committed edit with its source and resulting revisions, linking the operation history to the evolution of the executable state. Rendered observations and verification results are associated with the revision they assess, but collecting this evidence does not itself create a new PEG revision. The MLLM uses the revision-specific evidence together with the requirements in $C _ { k }$ to determine subsequent edits. The generation state thus provides a shared revision reference for the construction history tracked by TGP and the visual assessments maintained by REV.

## 3.2 Traceable Generation Process

While the PEG state preserves the executable visual representation at each revision, the traceable generation process (TGP) records how this representation is constructed and modified through generation operations. For the j-th recorded operation, the harness maintains:

$$
\tau _ { j } = \big ( o _ { j } , x _ { j } , y _ { j } , k _ { j } ^ { - } , k _ { j } ^ { + } \big ) ,\tag{3}
$$

where $o _ { j }$ denotes the executed operation, $x _ { j }$ and $y _ { j }$ are its input arguments and execution result (including returned errors), and $k _ { j } ^ { - }$ and $k _ { j } ^ { + }$ identify the source and resulting PEG revisions. The operation index $j$ is distinct from the revision index $k ,$ since operations that do not commit a state update, including rendering and inspection, retain the same revision index. Comparing the corresponding snapshots exposes changes to the visual program and its associated state without requiring a common scene representation across rendering backends.

The harness links each rendered observation to the PEG revision evaluated by the renderer and retains the actual output, sampled timestamps, and any crop specification. These records identify the visual evidence used in a review even when custom code introduces nondeterminism beyond the configured seed. For example, an edit to a drawing function or motion trajectory can be located in the operation history and compared with renderings of the relevant revisions to investigate discrepancies in spatial composition or temporal dynamics. Rendering and inspection need not follow every edit, so an executed modification and a visually inspected revision remain distinguishable in the trace.

TGP supports diagnosis of the P2V gap by connecting recorded program changes with revisionspecific visual evidence, rather than treating successful execution as confirmation of the intended visual result. Its traceability concerns explicit generation operations and their execution results, not the MLLM’s internal reasoning. These operation-to-revision and evidence-to-revision associations provide the basis for subsequent inspection and revision, while the validity of visual assessments after further edits is addressed by REV.

## 3.3 Revision-aware Editing and Verification

![](images/97acdeb8a4a8faf3a8c2763422a64781228d9101e5481522276cb72ab522ba0a.jpg)  
Figure 5: Revision-aware editing and verification. Edits produce new PEG revisions, and visual evidence is associated with the revision it assesses. Historical evidence supports comparison; delivery requires verification of the current revision.

As shown in Fig. 5, revision-aware editing and verification (REV) closes the generation loop by verifying edits to PEG states using the revision-linked evidence recorded by TGP. Within a run, restoring historical content from $\bar { S } _ { r } \ ( r \leq k )$ commits a new state $S _ { k + 1 }$ while preserving current task requirements; the trace records the restored revision r. A separate editing task initializes its own revision sequence from the selected snapshot, retains a link to the source run and revision, and incorporates the new user instruction into $C _ { 0 }$ . Subsequent edits remain traceable through TGP.

For each visual requirement $h _ { i }$ in $C _ { k }$ , REV maintains a review $q _ { i } = ( k _ { i } , E _ { i } , v _ { i } )$ , where $E _ { i }$ contains requirement-specific visual evidence from revision $k _ { i }$ and $v _ { i } \in \{ \mathrm { p a s s } $ , fail, uncertain}. Full-frame renderings or spatial crops support appearance and composition checks, while temporal requirements use ordered samples spanning the specified interval at three or more distinct timestamps. Such samples support temporal assessment but do not establish continuity between frames. A review applies to the current state only when $k _ { i } = k .$ . After each commit, even if only the plan changes, the current revision must be reviewed again and its checkpoints reapproved. Recording observations and reviews leaves k unchanged. Moreover, the current revision is ready for delivery only when the following conditions hold:

$$
\begin{array} { l } { \displaystyle \mathrm { R e a d y } ( S _ { k } ) = \mathrm { E x p o r t O K } ( S _ { k } ) \wedge \mathrm { C h e c k p o i n t O K } ( k ) } \\ { \wedge \bigwedge _ { h _ { i } \in \mathcal { H } _ { k } } [ k _ { i } = k \wedge E _ { i } \neq \emptyset \wedge v _ { i } = \mathrm { p a s s } ] , } \end{array}\tag{4}
$$

where $\mathcal { H } _ { k }$ contains the visual and temporal requirements marked as mandatory for completion in $C _ { k }$ ExportOK verifies that source files and assets exist and match their recorded hashes. It also checks applicable object and event constraints and confirms that the current revision’s export conforms to ω. These export checks cover format, dimensions, and video timing. During planning, the MLLM organizes all mandatory requirements into checkpoints. CheckpointOK holds when every checkpoint passes the required checks and reviews for the current revision. Missing evidence or reviews marked as failed or uncertain require further inspection or revision while budget remains. This criterion ties visual verification to the delivered revision. MLLM verdicts remain self-assessments rather than independent measures of perceptual quality.

## 4 Experiments

## 4.1 Experimental Setup

Tasks and models. We evaluate MaLiang-Harness on MaLiang-IBench with 50 text-to-image prompts and MaLiang-VBench with 13 text-to-video prompts. Both sets cover diverse visual styles. We evaluate 11 models from the DeepSeek [Xu et al., 2026a], Kimi [Team et al., 2026], and GPT [OpenAI, 2026a,b] families on MaLiang-IBench and four models on MaLiang-VBench. Each model constructs and refines executable visual programs through MaLiang-Harness.

Evaluation metrics. Our evaluation covers two aspects: computational cost and generation quality. We report generation success, failures, and the subset of failures caused by token-budget exhaustion (Token Limit). Success requires a decodable output that passes the harness’s completion checks. (1) Computational cost. We report generation time, model calls, and token usage. Cumulative time includes failed attempts and retries. Time/Qualified normalizes this time by the number of images meeting all quality criteria; Time/Success uses the number of successfully generated videos. Note that DeepSeek-V4-Pro is evaluated without visual feedback. (2) Generation quality. We primarily employ GPT-6-Sol as a judge. It evaluates image quality on a five-point scale for prompt alignment, aesthetics, and composition. Video evaluation includes motion coherence as an additional criterion and uses 12 temporally ordered frames from each successful video. Mean image-quality scores are computed over successful generations only. Per-dimension counts report the number of samples scoring at least 4, while All Criteria counts samples meeting all three image criteria or all four video criteria. Note that failed generations are excluded from all quality-threshold counts.

## 4.2 Comparisons

Quantitative results on MaLiang-IBench. Tables 2 and 3 distinguish generation success from satisfaction of visual requirements. GPT-5.6-Luna and GPT-5.6-Terra each successfully generate 46 of 50 images. However, the number satisfying all three quality criteria falls to 22 for Luna and 24 for Terra. GPT-6-Astra achieves a 100% generation success rate with 48 images satisfying all criteria. GPT-6-Sol and GPT-6-Luna satisfy all criteria on 46 and 44 tasks. All successfully generated images from the three GPT-6 models meet the aesthetics and composition thresholds. Their remaining quality failures concern prompt adherence. This pattern shows that successful generation alone is insufficient to assess executable visual programs.

Table 2: Generation success and computational cost on MaLiang-IBench. “Token Limit” denotes the subset of failures in which a run exhausts its allocated token budget, potentially due to reasoning stagnation and unbounded agentic loops. “Orange/blue” indicates “higher/lower” values relative to DeepSeek-V4.1-Flash, which do not imply “better/worse” performance given differing success rates.
<table><tr><td rowspan="2">Model</td><td colspan="3">Task Completion</td><td colspan="3">Generation Time</td><td colspan="3">Model Usage</td></tr><tr><td>Success ↑ (%)</td><td>Failures ↓ (cases)</td><td>Token Limit ↓ (cases)</td><td>Total Time (task-h)</td><td>Time/Qualified ↓ (min/image)</td><td>Time/Task (min/case)</td><td>Calls (calls/case)</td><td>Input Tokens (k/case)</td><td>Output Tokens (k/case)</td></tr><tr><td>DeepSeek-V4.1-Flash</td><td>12.0</td><td>44</td><td>5</td><td>2.71</td><td>27.15</td><td>1.91</td><td>2.48</td><td>100.8</td><td>24.2</td></tr><tr><td>DeepSeek-V4-Pro</td><td>18.0</td><td>41</td><td>3</td><td>6.42</td><td>48.12</td><td>7.36</td><td>4.50</td><td>198.9</td><td>39.6</td></tr><tr><td>Kimi-K2.6</td><td>34.0</td><td>33</td><td>1</td><td>6.27</td><td>62.69</td><td>5.14</td><td>9.14</td><td>660.0</td><td>48.2</td></tr><tr><td>Kimi-K2.7-Code</td><td>40.0</td><td>30</td><td>1</td><td>5.97</td><td>29.84</td><td>3.59</td><td>12.94</td><td>1056.5</td><td>59.0</td></tr><tr><td>Kimi-K3</td><td>18.0</td><td>41</td><td>0</td><td>5.23</td><td>34.88</td><td>5.89</td><td>2.82</td><td>308.9</td><td>17.8</td></tr><tr><td>GPT-5.6-Luna</td><td>92.0</td><td>4</td><td>0</td><td>1.65</td><td>4.49</td><td>1.98</td><td>11.10</td><td>271.6</td><td>9.3</td></tr><tr><td>GPT-5.6-Terra</td><td>92.0</td><td>4</td><td>0</td><td>1.73</td><td>4.32</td><td>2.07</td><td>8.32</td><td>186.0</td><td>8.0</td></tr><tr><td>GPT-5.6-Sol</td><td>96.0</td><td>2</td><td>1</td><td>2.35</td><td>3.28</td><td>2.82</td><td>6.98</td><td>209.2</td><td>13.1</td></tr><tr><td>GPT-6-Luna</td><td>96.0</td><td>2</td><td>0</td><td>3.61</td><td>4.92</td><td>3.27</td><td>14.70</td><td>459.2</td><td>16.4</td></tr><tr><td>GPT-6-Sol</td><td>96.0</td><td>2</td><td>0</td><td>2.70</td><td>3.52</td><td>3.24</td><td>7.42</td><td>180.1</td><td>10.2</td></tr><tr><td>GPT-6-Astra</td><td>100.0</td><td>0</td><td>0</td><td>2.96</td><td>3.70</td><td>3.55</td><td>6.34</td><td>142.0</td><td>8.1</td></tr></table>

Table 3: Visual quality on MaLiang-IBench. Quality counts include samples scoring at least 4 on a five-point scale. Means cover successful samples only. Gray cells mark DeepSeek and Kimi results, whose low success rates limit the comparability of mean quality scores. Bold marks the best GPT results. “Blue/orange” denotes “higher/lower” values relative to GPT-5.6-Luna.
<table><tr><td rowspan="2">Model</td><td colspan="2">Generation Outcomes</td><td colspan="4">Quality Threshold Counts</td><td colspan="3">Mean Quality Scores</td></tr><tr><td>Successful ↑ (cases)</td><td>All Criteria ↑ (cases)</td><td>Alignment ↑ (cases)</td><td>Aesthetics ↑ (cases)</td><td></td><td>Composition ↑ (cases)</td><td>Alignment ↑ (1–5)</td><td>Aesthetics ↑ (1-5)</td><td>Composition ↑ (1-5)</td></tr><tr><td>DeepSeek-V4.1-Flash</td><td>6</td><td>6</td><td>6</td><td></td><td></td><td>6</td><td>4.33</td><td>4.00</td><td>4.17</td></tr><tr><td>DeepSeek-V4-Pro</td><td>9</td><td>8</td><td>8</td><td></td><td></td><td>9</td><td>4.11</td><td>3.89</td><td>4.11</td></tr><tr><td>Kimi-K2.6</td><td>17</td><td>6</td><td>7</td><td></td><td></td><td>9</td><td>3.47</td><td>3.41</td><td>3.65</td></tr><tr><td>Kimi-K2.7-Code</td><td>20</td><td>12</td><td>13</td><td></td><td></td><td>13</td><td>3.75</td><td>3.60</td><td>3.70</td></tr><tr><td>Kimi-K3</td><td>9</td><td>9</td><td>9</td><td></td><td></td><td>9</td><td>4.56</td><td>4.22</td><td>4.33</td></tr><tr><td>GPT-5.6-Luna</td><td>46</td><td>22</td><td>25</td><td></td><td></td><td>40</td><td>3.57</td><td>3.74</td><td>3.93</td></tr><tr><td>GPT-5.6-Terra</td><td>46</td><td>24</td><td>27</td><td></td><td></td><td>40</td><td>3.70</td><td>3.70</td><td>3.93</td></tr><tr><td>GPT-5.6-Sol</td><td>48</td><td>43</td><td>43</td><td></td><td></td><td>48</td><td>4.23</td><td>4.10</td><td>4.27</td></tr><tr><td>GPT-6-Luna</td><td>48</td><td>44</td><td>44</td><td></td><td></td><td>48</td><td>4.10</td><td>4.08</td><td>4.15</td></tr><tr><td>GPT-6-Sol</td><td>48</td><td>46</td><td>46</td><td></td><td>48 48</td><td>48</td><td>4.25</td><td>4.15</td><td>4.25</td></tr><tr><td>GPT-6-Astra</td><td>50</td><td>48</td><td>48</td><td></td><td>50</td><td>50</td><td>4.36</td><td>4.22</td><td>4.54</td></tr></table>

Besides, mean quality scores must be interpreted alongside the number of successful tasks. DeepSeek and Kimi generate only 6-20 images, so their means describe a limited subset of MaLiang-IBench. Kimi-K3’s alignment score of 4.56 is based on just nine images and does not establish superior performance over the full evaluation set. Astra obtains the highest mean score in each quality dimension among the GPT models. The cost results reveal a trade-off between generation time and the number of images satisfying all criteria. GPT-5.6-Sol produces 43 qualifying images at 3.28 minutes per image. Astra produces 48 at 3.70 minutes per image. These time estimates include failures and retries and thus account for the cost of obtaining images that meet the quality thresholds.

Qualitative results on MaLiang-IBench. Fig. 6 illustrates differences in visual realization across the evaluated models. The greenhouse (last row) and robot ensemble (fourth row) examples reveal differences in subject scale, scene detail, and the depiction of individual objects across models. The teacup (first row) and ginkgo-leaf examples (third row) further probe the integration of flat cartoon characters with realistic materials. Astra’s results show more pronounced surface detail and lighting cues in these examples, while the architectural board combines a main view with supporting elevations and construction details. These selected examples complement the aggregate scores by showing how the same prompt can lead to different compositions and levels of visual detail.

Quantitative results on MaLiang-VBench. Tables 4 and 5 show a similar gap between generation success and quality on MaLiang-VBench. Astra satisfies all four quality thresholds on 10 of its 13 successful generations. GPT-5.6-Sol satisfies all thresholds on five of its seven successful generations. Kimi-K2.6 completes three tasks but none of its videos meets all criteria. DeepSeek-

![](images/9c7d742b6e25629c80d3ef6e2613b4919f853313d1b29ea5e4f12f4b2cb6fe4f.jpg)  
Figure 6: Qualitative comparison on MaLiang-IBench. Rows show a fox on a teacup, an architectural concept board, a rabbit on a ginkgo leaf, a robot ensemble, and a greenhouse illustration. Columns compare five models under the same prompt. A white × on black denotes an unsuccessful run. Full prompts are provided in Appendix 6.4.

Table 4: Generation success and computational cost on MaLiang-VBench. “Orange/blue” denotes “higher/lower” values relative to DeepSeek-V4.1-Flash, except “Time/Success”, which uses Kimi-K2.6. “–” indicates an undefined ratio with no successful generations.
<table><tr><td rowspan="2">Model</td><td colspan="3">Task Completion</td><td colspan="3">Generation Time</td><td colspan="3">Model Usage</td></tr><tr><td>Success ↑ (%)</td><td>Failures ↓ (cases)</td><td>Token Limit ↓ (cases)</td><td>Total Time (task-h)</td><td>Time/Success ↓ (min/video)</td><td>Time/Task (min/case)</td><td>Calls (calls/case)</td><td>Input Tokens (k/case)</td><td>Output Tokens (k/case)</td></tr><tr><td>DeepSeek-V4.1-Flash (64K)</td><td>0.0</td><td>13</td><td>5</td><td>1.47</td><td></td><td>6.77</td><td>6.38</td><td>560.5</td><td>84.2</td></tr><tr><td>Kimi-K2.6</td><td>23.1</td><td>10</td><td>1</td><td>3.35</td><td>66.95</td><td>15.45</td><td>16.15</td><td>548.8</td><td>26.9</td></tr><tr><td>GPT-5.6-Sol</td><td>53.8</td><td>6</td><td>3</td><td>1.08</td><td>9.27</td><td>4.99</td><td>12.85</td><td>574.2</td><td>13.8</td></tr><tr><td>GPT-6-Astra</td><td>100.0</td><td>0</td><td>0</td><td>2.17</td><td>10.00</td><td>10.00</td><td>11.31</td><td>463.3</td><td>14.4</td></tr></table>

V4.1-Flash completes no tasks. Token-budget exhaustion accounts for five of DeepSeek’s 13 failures and one of Kimi’s 10 failures. It also accounts for three of the six failures from GPT-5.6-Sol. Token limits therefore explain only part of the observed failures.

The quality results indicate that appearance alone does not capture temporal requirement satisfaction. All 13 Astra videos meet the aesthetics threshold. Only 10 meet the motion-coherence threshold, making motion the most restrictive criterion for Astra. All seven successful GPT-5.6-Sol videos meet the alignment and aesthetics thresholds. Requiring composition and motion coherence reduces the joint count to five. Astra achieves full task completion at 10.00 minutes per successful video. GPT-5.6-Sol requires 9.27 minutes per successful video but completes fewer tasks. “Time/Success” for videos uses the number of successful generations as its denominator. “Time/Qualified” for images instead uses the number of samples satisfying all quality thresholds.

Qualitative results on MaLiang-VBench. Fig. 11 compares the temporal progression of three selected video tasks. In case (a), Astra retains a detailed pixel-art setting as the kitten traverses the keyboard and books toward the lamp, whereas Kimi depicts a simpler scene and a reduced action sequence. Case (b) similarly contrasts Astra’s more detailed setting and visible latte-art outcome with Kimi’s sparse composition. In case (c), both successful models depict a progression from constructing the channel to filling it with water and moving boats downstream, using different spatial arrangements. Together, these examples highlight differences in both scene detail and the extent to which models realize the sequence of actions specified in the prompt.

Table 5: Visual quality on MaLiang-VBench. Quality counts include samples scoring at least 4 on a five-point scale. Gray denotes reference results. Bold marks the best GPT results excluding evaluation coverage. “Blue/orange” denotes “higher/lower” counts than GPT-5.6-Sol.
<table><tr><td rowspan="2">Model</td><td colspan="2">Generation Outcomes</td><td colspan="5">Quality Threshold Counts</td><td colspan="2">Evaluation Coverage</td></tr><tr><td>Successful ↑ (cases)</td><td>All Criteria ↑ (cases)</td><td>Alignment ↑ (cases)</td><td>Aesthetics ↑ (cases)</td><td></td><td>Composition ↑ (cases)</td><td>Motion ↑ (cases)</td><td>Evaluated (cases)</td><td></td></tr><tr><td rowspan="2">DeepSeek-V4.1-Flash (64K) Kimi-K2.6</td><td>0</td><td>0</td><td>0</td><td></td><td></td><td>0</td><td>0</td><td>0</td><td></td></tr><tr><td>3</td><td>0</td><td>0</td><td></td><td></td><td>1</td><td>0</td><td></td><td></td></tr><tr><td rowspan="2">GPT-5.6-Sol GPT-6-Astra</td><td>7</td><td>5</td><td>7</td><td></td><td>7</td><td>6</td><td>6</td><td>7</td><td></td></tr><tr><td>13</td><td>10</td><td></td><td></td><td>13</td><td>12</td><td>10</td><td></td><td>13</td></tr></table>

## 4.3 Discussion

![](images/dec3d41ededf6f7349118a647e12f08dc4b14abc87f6e243fc48d20a0d89a2f3.jpg)  
Figure 7: Iterative refinement with a path-tracing backend. Top: GPT-6-Astra revises a procedural breakfast scene through geometry, lighting, material, and sampling adjustments. Bottom: detail crops show “biscuit” surface artifacts introduced during refinement and reduced by geometry corrections.

Can the harness generate photorealistic content? We explore two complementary directions toward more realistic visual content: extending the rendering backend and refining the brushwork of an existing illustration: 1) More expressive rendering backends. MaLiang-Harness can incorporate renderers with richer material and light-transport capabilities through its shared interface, while retaining the high-level planning and generation control described in Sec. 3.1. Fig. 7 illustrates this approach with a path-tracing backend that supports structured scene geometry, physical materials, and area lighting. GPT-6-Astra constructs a breakfast still life without generated image assets, using a raster preview to establish composition before inspecting path-traced renderings. The resulting scene exhibits material-dependent reflections, glass transmission, and contact shadows. Further edits adjust object support, lighting, and surface geometry, while increasing the final sample count from 512 to 1024 reduces visible noise. These refinements also reveal how local edits can introduce new visual defects. In the biscuit crops, a geometry change creates overlap artifacts that a subsequent revision reduces. PEG preserves the scene revisions, while TGP and REV make the changes and their visual consequences available for comparison and correction. This example demonstrates the integration of a richer renderer within the same refinement process, while controlled comparisons are still needed to establish its contribution to perceptual realism.

2) Brushwork refinement inspired by hyperrealism. A complementary approach draws on the use of fine brushwork in hyperrealist painting. We examine whether an explicit refinement prompt, applied through REV, can move an existing stylized illustration toward a more realistic appearance without changing its Canvas backend. Fig. 8 compares the inherited landscape with the preview produced after GPT-6-Astra modifies its drawing routines. The overall composition is preserved, while foliage, grass, stone, water, and timber receive smaller marks or thinner contours. However, the image remains stylized, and several regions lose existing detail: grass coverage becomes less visible, while distinctive stone markings and mountain textures are attenuated. Finer strokes alone do not recover the coordinated shape, shading, and texture required for realistic depiction. It highlights the<sub>General</sub> <sub>capability</sub> <sub>and</sub> <sub>visual</sub> <sub>program</sub> <sub>generation</sub><sup>General</sup> <sup>capability</sup> <sup>and</sup> <sup>visual</sup> <sup>program</sup> <sup>generation</sup> need to preserve meaningful visual detail when translating refinement instructions into code edits.Public benchmark scores versus MaLliang-IBench outcomes | 11 models, 50 image tasks per model

(b) After  
![](images/e6ea00ea3bfcb1eb92e5fe289b42caca49975348d01792bd32dffc5e49622578.jpg)  
Figure 8: Limits of prompted brushwork refinement. Left: a Canvas illustration before and after GPT-6-Astra edits its drawing program to produce finer brushwork within REV. Right: paired detail crops show each region before (top) and after (bottom) editing. Finer marks do not yield photorealistic appearances: grass coverage and distinctive surface texture are instead reduced in several regions.

![](images/9ca0fc515a56aa13224249921e5d0af34723b451ac039239b3009a04f0925d02.jpg)  
Drawing quality pass rate: percentage of al 50 tasks scoring ≥ 4/5 in alignment, aesthetics and composition.(a) General capability and drawing quality   Drawing quality pass rate: percentage of al 50 tasks scoring ≥ 4/5 in alignment, aesthetics and composition.

![](images/a21b15a67be103f861edfac07c3713e786c1a39fb481401348e47c7d84d7b910.jpg)  
(b) Generation outcomes by model  
Figure 9: General capability and visual program generation on MaLiang-IBench. (a) Public AA Intelligence Index scores versus the percentage of all 50 tasks meeting every visual quality threshold. The dashed line highlights the 44-percentage-point gap between the two Luna models. (b) Models ordered by AA score (brackets). Dark teal, light teal, and gray denote all criteria met, successful generation with unmet criteria, and failure. White numbers give quality pass rates and rightmost numbers give generation success rates. <sup>†</sup>The public V4-Pro score uses checkpoint 0813.

Does general MLLM capability predict visual program generation performance? We examine how general model capability relates to visual program generation by comparing public Artificial Analysis (AA) Intelligence Index scores [Artificial Analysis, 2026] with MaLiang-IBench outcomes. We use the displayed integer scores from version 4.3.2, accessed on September 27, 2026, for the reasoning configurations listed on the model pages. The index aggregates evaluations of knowledge, reasoning, coding, and agentic tasks. Drawing quality pass rate is the proportion of all 50 tasks that satisfy the alignment, aesthetics, and composition thresholds, including unsuccessful generations in the denominator. Across the 11 models, the two measures exhibit a positive rank correlation (Spearman $\rho = 0 . 6 5$ in Fig. 9a). However, general capability scores do not fully predict visual performance. GPT-5.6-Luna and GPT-6-Luna share a displayed index score of 37, yet their drawing quality pass rates are 44% and 88%. Kimi-K3 scores 44 on the index but satisfies all visual criteria on only 18% of tasks. Fig. 9b further separates completion from visual quality: GPT-5.6-Luna and

![](images/a6d82b2117f2357b86a70632e813b734631ace1c3cc47eb61596b28177232994.jpg)  
Figure 10: Tracing a “research figure case” to its construction process. Left: a generated sixpanel “research-figure” poster. Right: case-specific pseudocode abstracted from the execution trace (comments a-f refer to panels in the rendered image). The recorded program, asset-generation steps, and reviews make it possible to trace how individual elements were constructed and assessed.

GPT-5.6-Terra each complete 92% of tasks, while only 44% and 48% meet all quality thresholds. These observations suggest that general benchmark performance captures only part of the demands of visual program generation, which also requires translating spatial and appearance constraints into executable representations and using rendered feedback to guide revision.

What can the construction process reveal beyond the final image? MaLiang-Harness makes the construction of a visual result inspectable through its executable program, asset provenance, and revision history. Fig. 10 pairs a “six-panel research illustration” of an academic paper with pseudocode summarizing its recorded construction process. GPT-6-Astra uses code to specify the layout, architecture diagram, labels, and plots from user-supplied synthetic data, while generating photographic-style assets for the manipulation examples. The trace records four asset-generation attempts with different contact-sheet layouts. The final composition uses crops from the third attempt, with code controlling their placement, scaling, and labels. The construction record distinguishes elements drawn from explicit data and instructions from those assembled using generated imagery. This distinction also helps interpret the final review: the numerical plots and layout pass, whereas the photographic examples remain unsatisfactory, leaving the output only as a draft. By linking construction steps to revision-specific assessments, the harness reveals how individual elements were produced and which requirements remain unresolved beyond what the final image alone shows.

When does refinement stall, and when should it stop? The harness faces three potential failure modes: reasoning stagnation, where planning does not advance the artwork; semantic livelock, where repeated edits leave requirements unresolved; and infinite agentic loops, where planning and tool use repeat without termination. Refinement stalls when continued activity produces no meaningful visual improvement. PEG preserves the working state, while TGP and REV support inspection, comparison, and recovery, but these mechanisms do not guarantee effective corrections. In Fig. 8, further inspection yields no corrective edit before the token budget is exhausted. The harness therefore separates termination from completion: model-call, tool-call, time, and token limits bound execution, while successful finalization is required to mark a task completed.

## 5 Conclusion

We presented MaLiang-Harness, a unified framework for programmable image and video generation motivated by the Program-to-Visual (P2V) gap between program-level correctness and visual re-

![](images/28ff2f4b883a0c7a876d3592a8489a0127a99fd86e0932f33e20c048f1ec7033.jpg)  
Case (c): Doodle cats build a watercourse, tip a real watering can and ride boats through the resulting rapids.

Figure 11: Qualitative comparison on MaLiang-VBench. Groups show (a) Star Delivery, (b) The Tiny Dragon Barista, and (c) Watering-Can Rafting. Each model row contains five uniformly sampled frames in temporal order from left to right. A white × on black denotes an unsuccessful run. Full prompts are provided in Appendix 6.5.

quirement satisfaction. By maintaining executable state, recording construction histories, and linking verification to the current revision, the framework organizes MLLM-driven visual creation into a persistent process of construction, inspection, and refinement. Evaluation of 11 MLLMs on MaLiang IBench and four on MaLiang-VBench reveals substantial differences in generation success, visual quality, and computational cost. Models with similar general-capability scores can produce markedly different visual outcomes, motivating direct evaluation of visual program generation. Photorealism and stalled refinement remain challenges. Future work will explore richer rendering backends and progress-aware refinement strategies to improve visual fidelity and generation efficiency.

## References

Artificial Analysis. Independent analysis of ai: Understand the ai landscape to choose the best mode and provider for your use case, 2026. URL https://artificialanalysis.ai/.

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.

Yanzhe Chen, Zechen Bai, Zhijun Cao, Wenzheng Zeng, Kevin Qinghong Lin, Yiqi Lin, Guoqiang Liang, Kevin Yuchen Ma, Qiming Huang, and Mike Zheng Shou. Show-harness: Just a vlm agent can play robots. arXiv preprint arXiv:2609.10522, 2026.

Tanmay Gupta and Aniruddha Kembhavi. Visual programming: Compositional visual reasoning without training. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14953–14962. IEEE, 2023.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in neural information processing systems, 35:8633–8646, 2022.

Ian Huang, Guandao Yang, and Leonidas Guibas. Blenderalchemy: Editing 3d graphics with vision-language models. In European Conference on Computer Vision, pages 297–314. Springer, 2024.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Xuying Ning, Katherine Tieu, Dongqi Fu, Tianxin Wei, Zihao Li, Yuanchen Bei, Jiaru Zou, Mengting Ai, Zhining Liu, Ting-Wei Li, et al. Code as agent harness. arXiv preprint arXiv:2605.18747, 2026.

OpenAI. Gpt-5. 2026a. URL https://openai.com/zh-Hans-CN/gpt-5/.

OpenAI. Gpt-6 astra: A new generation of intelligence. 2026b. URL https://openai.com/ zh-Hans-CN/index/gpt-6-astra/.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In 2022 IEEE/CVF conference on computer vision and pattern recognition (CVPR), pages 10674–10685. ieee, 2022.

Chenglei Si, Yanzhe Zhang, Ryan Li, Zhengyuan Yang, Ruibo Liu, and Diyi Yang. Design2code: Benchmarking multimodal code generation for automated front-end engineering. In Proceedings of the 2025 Conference ofthe Nations ofthe Americas Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 3956–3974, 2025.

Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, SH Cai, Yuan Cao, Ziwei Chai, Y Charles, HS Che, Cheng Chen, et al. Kimi k2. 5: Visual agentic intelligence. arXiv preprint arXiv:2602.02276, 2026.

Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chenchen Ling, et al. Deepseek-v4: Towards highly efficient milliontoken context intelligence. arXiv preprint arXiv:2606.19348, 2026a.

Xu Xu, Jinxiu Liu, Zhangbo Qiao, Jiaxing Lu, Xiangyu Zhang, Yubin Gu, Fangwei Ning, and Yan Shi. Omniharness: Harnessing generalizable visual generation via symbolic policy learning. arXiv preprint arXiv:2609.16057, 2026b.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 3813–3824. IEEE, 2023.

## 6 Appendix

This appendix provides an expanded construction example, evaluation and aggregation details, the computation of the comparative plots, and the prompts used for the qualitative examples.

## 6.1 Procedural Construction Example

Algorithm 2 expands the botanical portrait example in Fig. 4. It summarizes the drawing operations within a fixed PEG revision, rather than the full planning and refinement loop. The program builds the image in four layers: the background, circular foliage, the portrait, and foreground flowers. Separate seeded generators control variation in foliage, portrait texture, and flowers. The named routines abstract the underlying Canvas operations; no image assets are used. As in Algorithm 1, k denotes the flower-layer index within this algorithm, rather than a PEG revision.

## 6.2 Evaluation Protocol and Reporting Details

Task outcomes and quality metrics. MaLiang-IBench contains 50 image tasks and MaLiang-VBench contains 13 video tasks. A task succeeds when its output is decodable and passes the harness’s completion checks; all remaining tasks are counted as failures. Token Limit identifies the subset of failures that exhaust the allocated token budget. These outcome categories are distinct from the subsequent quality assessment by GPT-6-Sol. Images are scored from 1 to 5 for prompt alignment, aesthetics, and composition; videos additionally receive a motion-coherence score. Each dimension’s threshold count includes successful outputs scoring at least 4, and All Criteria counts those meeting every threshold. Mean quality scores use successful outputs only. Success rates and quality pass rates use the full task set as their denominator, so failed tasks contribute to neither numerator.

Video review. A single reviewer evaluated all 23 successful videos across the four models using the original prompt and 12 sampled frames in temporal order. Evaluated records the number of reviewed videos. DeepSeek’s zero quality counts reflect the absence of successful outputs, not assigned scores of zero. The review was not formally blinded. Motion coherence measures the progression visible in the sampled frames and does not establish frame-by-frame fluidity or rule out brief intermediate faults.

Computational cost. Cumulative generation time sums the durations of recorded attempts, including failures and retries; it is not the wall-clock duration of a batch executed in parallel. Time/Qualified divides cumulative time by the number of images meeting all three quality thresholds. Time/Success divides cumulative time by the number of successful videos and is undefined when none succeeds. Input and output token counts come from available usage records and include repeated input context. Per-task statistics follow the aggregation rules below.

Image run configurations and aggregation. MaLiang-IBench combines multiple batches, retries, and historical runs rather than uniform first attempts. DeepSeek-V4-Pro is evaluated without visual feedback. GPT-6-Astra’s totals cover all 50 prompts, combining 39 attempts on 38 new tasks with 12 historical runs. New runs use worker wall time; historical runs use elapsed generation time without worker startup. Astra’s per-task statistics divide totals, including retries, by 50. Kimi’s mean call count also uses 50 as the denominator, whereas its Time/Task is the median over successful tasks and its token statistics divide recorded totals by the number of successful tasks. Thus, identically named cost columns do not always use identical aggregation rules, limiting direct efficiency comparisons.

Video run configurations and aggregation. The reported MaLiang-VBench results use one recorded attempt per model and task. Time, calls, and tokens per task are averaged over all 13 attempts. DeepSeek results use the completed evaluation with a 64K output-token limit; the incomplete 32K batch is excluded. Astra results use historical runs. Total time sums attempt wall times for new batches and elapsed generation times for historical runs. Differences in budgets, timing conventions, and run conditions mean that cost comparisons describe the evaluated configurations.

Algorithm 2 Procedural generation of the botanical portrait   
Require: Canvas size W = 1024, $H = 5 7 6 ;$ fixed seed s   
Ensure: Rendered RGB image I   
1: C ← NEWCANVAS(W, H)   
2: R , R , R ← SEEDEDGENERATORS(s, foliage, woman, flowers)   
▷ Layer 1: white background   
3: FILLRECTANGLE(C, (0, 0, W, H), white)   
▷ Layer 2: circular foliage   
4: for $i = 0$ to 25 do   
5: $\theta  2 \pi i / 2 6 + \mathrm { J I T T E R } ( R _ { f } )$   
6: x, y ← BRANCHENDPOINTS $( ( W / 2 , H / 2 ) , \theta , R _ { f } )$   
7: STROKEBEZIERBRANCH(C, x, y)   
8: for each position along the branch do   
9: DRAWGRADIENTLEAVES(C, $R _ { f } )$   
10: DRAWLEAFVEINSANDGRAIN(C, R )   
11: end for   
12: end for   
13: DRAWBERRYSPRIGSANDADDITIONALLEAVES(C, $R _ { f } )$   
▷ Layer 3: woman’s portrait   
14: FILLBEZIERSHAPE(C, hair silhouette, brown gradient)   
15: FILLBEZIERSHAPE(C, neck and shoulders, skin gradient)   
16: FILLBEZIERSHAPE(C, face, skin gradient)   
17: ADDCLIPPEDSHADINGANDGRAIN(C, face, $R _ { w } )$   
18: DRAWFACIALPATHS(C, brows, closed eyes, lashes, nose, lips)   
19: FILLBEZIERSHAPE(C, front hair locks, brown gradient)   
20: for $d \in$ {left, right} do   
21: for $j = 0$ to 74 do   
22: STROKECLIPPEDHAIRSTRAND(C, d, j, $R _ { w } )$   
23: end for   
24: end for   
▷ Layer 4: flowers and foreground details   
25: for each prescribed rose position $( x , y , r )$ do   
26: DRAWSEPALS(C, x, y, r)   
27: for $k = 0$ to 4 do   
28: $n \gets [ 8 , 7 , 6 , 5 , 4 ] _ { k }$   
29: for $j \stackrel { \cdot } { = } 0$ to n − 1 do   
30: DRAWROTATEDBEZIERPETAL(C, x, y, r, k, j, R<sub>p</sub>)   
31: ADDPETALGRADIENTANDGRAIN(C, $R _ { p } )$   
32: end for   
33: end for   
34: end for   
35: for each prescribed daisy or small-blossom position do   
36: DRAWRADIALPETALSANDCENTER $( \bar { C } , \bar { R } _ { p } )$   
37: end for   
38: DRAWFOREGROUNDLEAVESANDBERRIES(C, $R _ { p } )$   
39: I ← RASTERIZE(C)   
40: return I

## 6.3 Comparative Plots and Table Conventions

Radar rankings. Fig. 2 uses six metrics from Tables 2 and 3: mean alignment, aesthetics, and composition scores; generation success rate; reported time per task; and the sum of input and output tokens per task. Each metric is ranked across all 11 models before grouping them into panels. Higher quality and success values rank better, whereas lower costs rank better. Ties in the reported values receive average ranks. A rank r is mapped to radius ${ \left( { 1 2 - r } \right) } / { 1 1 }$ , placing rank 1 at the outer edge. Polygon area is not an aggregate performance score, and radial differences represent rank differences rather than differences in the underlying metrics. Legend counts indicate successful outputs. Small successful subsets can produce high quality means, while early failures can reduce costs; the aggregation differences described above also apply to this figure.

General-capability comparison. Fig. 9 uses the displayed integer Artificial Analysis Intelligence Index scores from version 4.3.2, accessed on September 27, 2026, for the reasoning configurations listed on the public model pages [Artificial Analysis, 2026]. The image quality pass rate is the All Criteria count divided by 50. Spearman correlation between this rate and the index is $\rho = 0 . 6 5$ across 11 models. The public DeepSeek-V4-Pro score refers to checkpoint 0813, whose correspondence to our evaluated checkpoint is unverified; excluding it gives $\rho = 0 . 6 3$ . Public evaluation settings may differ from those used in our harness, so this analysis characterizes an association rather than isolating the effects of model capability.

Numerical color scales. In Tables 2 and 4, orange and blue indicate values above and below DeepSeek-V4.1-Flash, respectively. Colors encode numerical differences, not a uniform preference across metrics. Relative-change intensity is $\sqrt { \operatorname* { m i n } ( | x / b - 1 | , 1 ) }$ , where x is a reported value and b is its baseline. Because DeepSeek has zero successful videos, video success-rate intensity instead uses $\sqrt { | x - b | / 1 0 0 }$ , with x and b expressed as percentages. Video Time/Success uses Kimi-K2.6 as its baseline.

Quality-table highlighting. In Tables 3 and 5, blue and orange indicate higher and lower GPT results relative to GPT-5.6-Luna for images and GPT-5.6-Sol for videos. Color intensity follows the relative-change rule above. DeepSeek and Kimi use a separate gray scale because their quality results cover fewer successful tasks; darker gray denotes larger values within each reference column. Bold identifies the best GPT values, excluding the descriptive Evaluated column. Quality-threshold counts remain comparable as counts over the full task set, while means describe different successful subsets.

## 6.4 Prompts for the Qualitative Image Comparisons

The five prompts below follow the row order in Fig. 6. Chinese prompts and display text are translated into English for presentation; evaluation used the original wording. Prompt content is presented separately from the evaluation protocol and does not imply that a generated output satisfies every requested property.

## Fox Painter on a Teacup.

A flat 2D cartoon fox painter stands on the rim of a realistic white porcelain teacup. Output a 1536 × 864 landscape PNG. Place the teacup slightly left of center on a wooden table, with natural highlights on its glaze, its handle on the right, and dark tea visible inside. The fox holds a paintbrush in one paw and a small palette in the other, painting a short blue arc on the handle. Both feet must rest on the rim nearest the viewer; the fox must not float or stand in the tea.

Render the fox with solid orange color blocks, clear black outlines, and a white muzzle, creating a distinct contrast with the realistic ceramic and wood grain. Use soft window light from the left, a light gray background, and negative space on the right. The brush tip must touch the handle, and the character must not be larger than the entire teacup. Do not add text, extra characters, or a frame around the entire scene.

## Mountain Teahouse Concept.

Create a 2048 × 1152 landscape architectural survey board for a fictional timber teahouse, titled “Mountain Teahouse” and “MOUNTAIN TEAHOUSE”, with a small “CONCEPT STUDY” label.

Use a wide main panel on the left and a narrower column of two panels on the right. The left panel shows a three-quarter axonometric teahouse raised on stone piers on a hillside: one curved gable roof, an open front veranda, four evenly spaced front columns, sliding rear screens and a short stair descending toward the foreground. Add a small tree behind the building without hiding the roof.

The upper-right panel, “Front elevation”, repeats the four columns, roof curve and central stair. The lower-right panel, “Section”, cuts across the veranda and enclosed rear room, revealing roof rafters and stone supports. A bottom strip contains three enlarged details labeled “Roof joint”, “Column base”, “Sliding screen”. Use leaders 1–3 to connect these features in the main view to their matching detail numbers. Do not add measurements or historical dates.

Use fine sepia pen lines, restrained watercolor washes, warm ivory paper, a thin double-line border and an orderly grid. This is a fictional conceptual documentation sheet, not a factual historical survey. Do not invent historical dates, engineering safety claims or paragraphs of unreadable text. Use only the exact short labels requested, with clear typography and thin leaders. No photographic rendering, bright neon colors or modern vehicles. Output one static PNG.

## Rabbit Violinist on a Ginkgo Leaf.

Draw a 1536 × 1024 landscape mixed-media illustration: a realistic golden-brown ginkgo leaf lies flat on damp, dark slate, with visible veins, curled edges, and exactly three separate transparent dewdrops. A white 2D cartoon rabbit stands on the broad upper portion of the leaf, playing a simplified small brown violin. Its left paw holds the violin neck, its right paw holds the bow, and the bow crosses the strings. The rabbit’s eyes are gently closed.

Render the rabbit and violin with clear brown outlines, flat colors, and minimal shading; render the leaf, dewdrops, and slate with realistic materials and soft side lighting. The rabbit’s body height is approximately one-third of the leaf’s length, with a small contact shadow beneath its feet. Keep the background quiet, with no other leaves, text, musical notes, or characters. Output a static PNG.

## Garden Robot Cast.

Draw a 2048 × 1152 landscape robot cast illustration on a white background, containing exactly six original garden robots in two rows of three. In the back row, from left to right: a tall teal cylindrical robot holding a small rake; an off-white robot with a square head, wearing a wide-brimmed gardening hat and holding an empty flowerpot; and an orange robot with a triangular head, carrying a light blue water tank on its back and holding a spray hose. In the front row, from left to right: a short pink robot with a round head, raising a packet of seeds with no text; a yellow robot with a trapezoidal body, holding a seedling in both hands; and a purple robot with an oval body, holding a closed pair of pruning shears.

All robots have two eyes, two arms, and two legs, but distinct head and body silhouettes. Each tool belongs only to its assigned character. Make the back row taller and the front row shorter, slightly staggering the silhouettes without obscuring any character’s face, hands, or feet. Use clean bold outlines, flat colors, minimal shading, and a children’s animation style. Do not add a garden background, text, logos, extra robots, or scattered tools. Output PNG.

## Greenhouse Potting Bench.

Create a warm 1536 × 1024 landscape children’s-book illustration of a greenhouse potting bench. The main cast consists of a smiling terracotta plant pot with a seedling, a shy lavender watering can, a cheerful yellow gardening glove, and a sleepy mint-green seed box. Each has exactly one face. Place the pot at the center, the watering can to its left with its spout pointing toward the pot, the glove resting to the right, and the closed seed box at the front edge of the bench.

In the background, curved greenhouse ribs frame a pale morning sky. Hang three faceless pots at different heights and show a short shelf with blank seed packets. A few drops of water are suspended between the watering-can spout and the seedling, suggesting a single frozen moment. Use soft peach, sage, butter yellow and cream, rounded silhouettes, fine outlines and gentle shading. Keep every character recognizable and do not draw human gardeners, text or extra anthropomorphic objects. Output PNG.

Additional example with explicit layout constraints. Beyond the qualitative examples in the main text, the following prompt illustrates how spatial constraints can be specified through object identifiers, normalized coordinates, dimensions, and color codes. These specifications are supplied as part of the text prompt and interpreted by the MLLM to construct an executable visual program.

Draw a 2048 × 1152, 16:9 landscape layout control diagram for a “museum triptych exhibition board” and output a static PNG. Draw only the colored rectangles and IDs specified below. Do not draw actual posters, people, objects, or final copy.

All rectangle coordinates are given as [x, y, width, height], with the horizontal and vertical axes mapped independently to a normalized range of 0–1000 using the actual canvas width and height. Do not mistakenly produce a square image, rearrange the layout, or automatically align the elements. Use a white background with no outlines, rounded corners, shadows, or gradients. Center each ID within its rectangle using a clear black sans-serif font. The list contains all rectangles; there are no unlisted parent containers or additional legends.

Color coding: coral red #FF6B6B denotes titles; sky blue #62CBE8 denotes main illustration regions; magenta-purple #D86BDA denotes photo regions; golden yellow #F4C95D denotes analysis regions; mint green #72C7A0 denotes color palettes; and lavender #A999E8 denotes labels.
<table><tr><td rowspan=1 colspan=1>ID</td><td rowspan=1 colspan=4>[x, y, width, height]  Color</td></tr><tr><td rowspan=1 colspan=1>H1</td><td rowspan=5 colspan=4>[30, 30, 650, 85]       Coral red[730, 30, 240, 85]      Coral red[30, 155, 260, 470]    Magenta-purple[30, 645, 260, 45]      Lavender[320, 155, 390, 330]   Sky blue</td></tr><tr><td rowspan=1 colspan=1>H2</td></tr><tr><td rowspan=1 colspan=1>P1</td></tr><tr><td rowspan=1 colspan=1>T1</td></tr><tr><td rowspan=1 colspan=1>M1</td></tr><tr><td rowspan=1 colspan=1>D1</td><td rowspan=1 colspan=3>[320, 515, 180, 110]</td><td rowspan=1 colspan=1>Golden yellow</td></tr><tr><td rowspan=1 colspan=1>D2</td><td rowspan=1 colspan=1>[530, 515,</td><td rowspan=1 colspan=2>180, 110]</td><td rowspan=1 colspan=1>Golden yellow</td></tr><tr><td rowspan=1 colspan=1>T2</td><td rowspan=1 colspan=1>[320, 645,</td><td rowspan=1 colspan=2>390, 45]</td><td rowspan=1 colspan=1>Lavender</td></tr><tr><td rowspan=1 colspan=1>P2</td><td rowspan=1 colspan=1>[740, 155,</td><td rowspan=1 colspan=1>230,</td><td rowspan=1 colspan=1>210]</td><td rowspan=1 colspan=1>Magenta-purple</td></tr><tr><td rowspan=1 colspan=1>P3</td><td rowspan=1 colspan=1>[740, 395,</td><td rowspan=1 colspan=1>230,</td><td rowspan=1 colspan=1>230]</td><td rowspan=1 colspan=1>Magenta-purple</td></tr><tr><td rowspan=1 colspan=1>T3</td><td rowspan=1 colspan=1>[740, 645,</td><td rowspan=1 colspan=1>230,</td><td rowspan=1 colspan=1>45]</td><td rowspan=1 colspan=1>Lavender</td></tr><tr><td rowspan=1 colspan=1>S1</td><td rowspan=1 colspan=3>[30, 735, 290, 150]</td><td rowspan=3 colspan=1>Sky blueSky blueSky blue</td></tr><tr><td rowspan=1 colspan=1>S2</td><td rowspan=1 colspan=3>[355, 735, 290, 150]</td></tr><tr><td rowspan=1 colspan=1>S3</td><td rowspan=1 colspan=3>[680, 735, 290, 150]</td></tr><tr><td rowspan=1 colspan=1>F1</td><td rowspan=1 colspan=3>[30, 925, 940, 40]</td><td rowspan=1 colspan=1>Lavender</td></tr></table>

Spatial relationships: the main image in the middle column is wider than the left and right columns; the two photo regions in the right column have different heights; and the three bottom image regions have equal widths, with white gaps between them.

Check that there are exactly 15 rectangles, each ID appears exactly once, all coordinates and colors are accurate, and no rectangles overlap or extend beyond the canvas. Leave all unspecified gaps white. Do not add connecting lines, arrows, coordinate axes, borders, photographs, actual text content, or extra color blocks. Complete the task using only SVG or Canvas code.

## 6.5 Prompts for the Qualitative Video Comparisons

The following prompts correspond to Star Delivery, The Tiny Dragon Barista, and Watering-Can Rafting in Fig. 11. Star Delivery is translated from Chinese for presentation; evaluation used the original Chinese wording. The other two prompts were originally written in English. The 12-frame quality review described above is distinct from the five uniformly sampled frames displayed for each model in the comparison figure.

## Star Delivery: Lighting Up Desk City.

A pixel kitten catches a star ejected from an alarm clock, rides a keyboard train across a desk, slides along book pages into rapids created by a teacup, and finally swings high into the air using a desk lamp’s pull cord to place the star inside the unlit bulb.

Core action sequence: the alarm clock ejects a star → a keyboard railway grows → a cart accelerates → the spacebar launches it → book pages unfold → tea floods the desk → a paper boat rides the rapids → a sweeping pull-cord swing → starlight illuminates the entire desk.

Ready-to-use prompt. Create an imaginative animation entirely in pixel art, with clear action progression and a landscape 16:9 composition. The entire world must use a consistent, detailed 2D pixel-art style: characters, everyday objects, backgrounds, liquids, smoke, lighting, and particles must all consist of clearly recognizable square pixels. Use a uniform pixel size, a limited and harmonious palette, stepped contours, and a small amount of pixel dithering to convey depth. Do not use photorealistic assets, hand-drawn lines, or 3D voxels, and do not simply apply a mosaic filter to ordinary video.

The scene is a desk at night, viewed from the side at a slightly elevated angle so that the entire stage is visible. A vintage alarm clock stands on the left, a mechanical keyboard occupies the foreground, several thick books are stacked in the middle, a cup of tea and a pad of sticky notes sit beside the books, and an unlit desk lamp stands on the right. These objects retain clearly recognizable everyday shapes, but all are drawn in pixel art. The opening is quiet and clearly organized, leaving enough space between objects for the roads, rivers, and actions that follow.

First sequence: the alarm clock opens and the star appears. The alarm clock’s two bells suddenly bounce vigorously, sending a ring of square sound waves outward. Its face opens like a mechanical hatch, ejecting a golden-yellow pixel star high into the air along a parabolic arc. A white pixel kitten wearing a blue hat rushes out of a small door at the bottom of the clock, quickly climbs a clock foot, leaps up to catch the star in midair, and lands at the left end of the keyboard. An empty star-shaped socket is visible inside the distant unlit desk lamp, clearly indicating where the kitten must deliver the star. Keep the same kitten, hat, and star throughout all subsequent shots.

Second sequence: the keyboard becomes a growing railway. Starlight falls on the keyboard, and keys rise one after another along the route like rows of small buildings. Square tracks assemble piece by piece from the gaps between the keys, extending toward the books. The Enter key rises, and four pixel wheels grow beneath it, transforming it into the kitten’s cart. Holding the star, the kitten jumps aboard; the cart immediately accelerates across most of the frame, turning sharply between keys of different heights, descending, and then racing up a ramp. The track continually assembles a short distance ahead of the cart, making the tension clear: the road has only just appeared when the cart races across it. Station flags and windows light up in sequence only after the cart passes, progressively activating the world along its route.

Third sequence: the spacebar launches the cart and book pages unfold into a slide. The cart reaches the end of the keyboard and presses down the spacebar. Like a spring-loaded launch pad, the spacebar first visibly sinks and then snaps upward, throwing both cart and kitten toward the books. Show a clear takeoff, airborne trajectory, and landing for this large leap between two objects, rather than instant teleportation. While the kitten is airborne, the top book suddenly opens, and its pages unfold one after another into a long slide winding around the stack. The cart lands on the slide and races downward, rounding bends and jumping short gaps while its wheels lift pages that flutter behind it. The books remain recognizable; their pages have simply become roads within the miniature world.

Fourth sequence: tea becomes rapids and a sticky note folds into a paper boat. The unfolded book cover strikes the teacup handle. The cup slowly tilts, then pours out a stream made of light and dark blue pixel blocks. The water first surges past the base of the books and then rushes across the desk toward the lamp, cutting off the original land route. Wave crests consist of constantly rearranging stepped color blocks, while splashes are small squares flying outward. As the kitten reaches the end of the slide, the current lifts a sticky note, which folds continuously in midair into a paper boat. The kitten jumps aboard while holding the star, leaving the cart on the bank. The rapids immediately sweep the boat around the cup’s base, down a small waterfall between the books, and upward on a large wave. Give the boat clear plunging, tilting, and airborne movements; it must not merely glide at constant speed on a flat plane.

Fifth sequence: a pull-cord swing delivers the star to the lamp. As the paper boat passes beneath the lamp, a pixel pull cord suddenly descends from the shade, assembling segment by segment. The kitten secures the star against its chest, steps onto the bow, leaps high from a wave crest, and grabs the cord. Its momentum draws the cord into a broad pendulum swing: the kitten sweeps through the low point and rises nearly to the top of the frame. At the highest point, it releases the cord, flies along a clear arc into the lampshade, and uses both hands to push the star into the bulb’s star-shaped socket. Hold briefly, then let the bulb suddenly light up.

Ending: the entire desk becomes an illuminated miniature city. Warm yellow light spreads from the lamp across the desk as successive pixel-color bands with crisp edges, rather than a soft-focus glow. Wherever the light reaches, the world completes its final assembly: keyboard buildings light their windows, railings grow beside the book-page slide, a small dock assembles beside the teacup, and sticky notes fold into birds that fly past the lampshade. The previously turbulent river gradually settles, and the paper boat docks. The kitten sits on the edge of the lampshade, swinging its feet and gazing at the desk city it has illuminated.

Base the animation on action within a continuous space, so viewers always understand where the character starts, which objects it passes, and where it is going. Keep the camera stable, using only slight lateral tracking when necessary; do not rely on rapid cuts or camera shake to create motion. The action climaxes come from launching, sliding, surfing, and swinging; the creativity comes from objects triggering one another, rather than arbitrary scene changes. All new structures must appear through gathering pixel blocks, assembly frame by frame, mechanical unfolding, or paper folding; avoid fading entire structures into view. Keep the pixel grid stable. Do not introduce antialiased edges, motion blur, smooth gradients, realistic water, or blurred glowing edges, and do not add game interfaces, scores, subtitles, or control buttons.

## The Tiny Dragon Barista.

Build “The Tiny Dragon Barista”: a 10-second looping cozy fantasy animation, rendered entirely with JavaScript Canvas 2D. No sound, images, spritesheets, or video assets.

Story. A realistic-looking morning café counter is overlaid with playful hand-drawn fantasy elements. A coffee cup waits beneath an espresso machine. Sleeping inside the cup is a tiny orange dragon, curled like a cat.

• 0.0–1.2 s: steam rises from the espresso machine and tickles the dragon’s nose.

• 1.2–2.5 s: the dragon sneezes a tiny puff of flame. Comic text: “pff!”

• 2.5–4.0 s: the flame accidentally ignites three floating coffee beans, which become glowing fireflies.

• 4.0–5.8 s: the dragon jumps onto the counter and chases them in a circular path around the cup, leaving a hand-drawn orange motion trail.

• 5.8–7.2 s: it catches the final bean and breathes one controlled flame onto the coffee foam.

• 7.2–8.5 s: the foam transforms into a perfect latte-art dragon face. The tiny dragon proudly places both hands on its hips.

• 8.5–10.0 s: too much steam erupts; the dragon dives back into the cup, the foam collapses, and the original sleeping pose returns.

Look. Combine a semi-realistic café still-life composition with visibly illustrated overlays: inked flames, scribbled steam, glowing beans, exaggerated cartoon expressions.

Palette: espresso brown, cream, muted sage, terracotta orange. Thick dark-brown outlines only on animated fantasy elements, while furniture and countertop use softer realistic shading.

Details. Add:

• condensation on the ceramic cup,

• tiny crumbs moving when the dragon lands,

• hand-drawn spark symbols,

• a hanging menu gently swaying,

• steam morphing into temporary doodles: stars, hearts, spirals.

Loop. The dragon must finish curled inside the cup in exactly the same pose as frame 0.

## Watering-Can Rafting.

A real metal watering can sits on a tabletop. Tiny doodle cats arrive carrying paddles, life rings, and signboards, then quickly draw channels, dams, and tiny docks around it. One cat turns the watering can, and a stream of water suddenly pours out. As the water flows, the drawn channels come alive, turning into a roaring doodle river. Little boats appear and launch into the current, carrying cats through sharp turns, splashes, and drops. The flow gets stronger and stronger, creating rapids, waterfalls, and spinning whirlpools made from hand-drawn lines and foam. More doodle fish leap out of the water, boats bounce upward, one cat nearly falls overboard, another surfs on a leaf racing downstream. The entire scene becomes a dramatic flood-powered amusement ride, mixing a real watering can with a vividly animated doodle whitewater world.