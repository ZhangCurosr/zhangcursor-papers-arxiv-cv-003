# ProtoFusion: Protocol-Guided Prompt Refinement for Generative Infrared and Visible Image Fusion

Pengwei Liang · Junjun Jiang · Junhui Hou

Abstract Infrared and visible image fusion integrates complementary information from both modalities to provide a more comprehensive representation of the scene. We propose ProtoFusion, a protocol-guided generative framework for infrared and visible image fusion. Instead of relying on fixed fusion operators, hand-crafted objectives, or manually specified one-shot prompts, ProtoFusion formulates fusion as protocol-constrained generation. It represents fusion intent with a structured protocol, consisting of environment state cards, grounding principles, and a style profile, which[ explicitly specifies which evidence should be preserved or suppressed and which properties the fused result should satisfy. To optimize this protocol, we introduce protocolguided prompt refinement, a closed-loop learning framework that iteratively performs generation, evaluation, and revision under multimodal evidence. Once learned, the optimized protocol is fixed at inference time and directly guides fusion generation. To address residual source inconsistencies, we further incorporate structured belief construction, which organizes modality reliability, scene evidence, and cross-modal conflicts into a structured representation for conservative correction after generation. Extensive experiments on standard benchmarks show that ProtoFusion improves quantitative performance, achieves favorable results in aesthetic evaluation, and produces visually coherent fusion outputs. Project Page: https://erfect2020.github.io/ProtoFusion/.

Keywords Multi-modal image fusion · Generative image fusion · Prompt optimization · Structured protocol optimization

## 1 Introduction

Infrared and visible image fusion aims to combine the complementary strengths of infrared and visible imagery into a single representation that is more informative than either modality alone. As illustrated in Fig. 1(a), infrared images provide robust thermal saliency under poor illumination and challenging visibility conditions, whereas visible images preserve richer structures and textures. Therefore, a desirable fused image should highlight thermally significant targets while retaining informative visible details for human inspection and subsequent perception in surveillance, autonomous systems, and other challenging environments (Ma et al., 2019a; Zhang et al., 2021; Zhang and Demiris, 2023). This goal naturally involves two coupled questions: what information should be aggregated from each modality, and how it should be aggregated into a coherent image. In practice, answering these questions is dificult because useful evidence must be separated from sensor noise, low-light corruption, and uninformative modality responses. This dificulty is compounded by the absence of ground-truth fused images, which forces most methods to rely on indirect objectives to define the desired fusion target (Liu et al., 2026; Bai et al., 2025a; Zhao et al., 2026).

As summarized in Fig. 1(b), existing methods address this problem through fixed inductive biases. Classical fusion methods rely on hand-crafted decomposition rules or explicit optimization objectives, while deep models replace them with learned networks whose fusion behavior is still largely determined by architecture design, training data, and loss functions (Li and Wu, 2019; Ma et al., 2019b; Xu et al., 2022; Ma et al., 2022; Zhao et al., 2023b). More recent methods move toward semantic-, task-, and prompt-aware fusion by incorporating detection or segmentation cues, or by conditioning the model on text and instructions (Liu et al., 2022; Tang et al., 2022, 2023; Zhao et al., 2023a; Liu et al., 2024b; Yi et al., 2024b; Liu et al., 2025a; Cheng et al., 2025b; Zhao et al., 2024; Wang et al., 2024a; Yang et al., 2025). These designs improve adaptivity and controllability, but the underlying intent of fusion remains implicit. In particular, the what is typically specified indirectly through losses, priors, or task interfaces, while the how is constrained by fixed operators, network design, or one-shot conditioning mechanisms. As a result, text conditioning is incorporated into the pipeline, but the fusion intent itself is rarely represented as an explicit, editable, and optimizable object.

![](images/196076edc9b9a928aef82aa54171397f8688002a092fd8024184be7eb3730b6a.jpg)  
Fig. 1 From implicit fusion biases to explicit protocol-constrained generation. (a) Fusion combines infrared thermal saliency with visible structure and texture. (b) Existing methods encode fusion intent implicitly through rules, networks, or conditioning. (c) ProtoFusion makes this intent explicit through an editable protocol specifying what to preserve and how to render it, with iterative refinement and optional structured belief correction.

To address this limitation, we propose ProtoFusion, a protocol-guided generative framework. As shown in Fig. 1(c), ProtoFusion makes the fusion intent explicit by introducing a structured fusion protocol. The protocol specifies what evidence should be aggregated, what content should be suppressed, and what visual properties the fused result should satisfy. Under this formulation, fusion is cast as protocolconstrained generation: a multimodal generative fusion model synthesizes the fused image under the guidance ofthe protocol. This view is particularly attractive for modern multimodal generators, whose expressive synthesis space can accommodate target enhancement, detail recovery, degradation mitigation, and visually coherent fusion within a unified framework (Zhao et al., 2023c; Yue et al., 2023; Yi et al., 2024a; Shi et al., 2025; Cao et al., 2025).

The proposed ProtoFusion framework uses a structured protocol with three components: environment state cards, grounding principles, and a style profile. Based on this representation, we develop protocol-guided prompt refinement (PGPR), an ofline refinement procedure that iteratively updates protocol slots through generation, evaluation, and revision under multimodal evidence. After refinement, the learned protocol is fixed and used directly to guide fusion generation at inference time. To further improve robustness on dificult cases, we additionally introduce structured beliefconstruction (SBC) as an optional correction module applied after PGPR. SBC analyzes residual modality imbalance and cross-modal inconsistency in the PGPR output and applies a conservative prompt-level correction when doing so is beneficial. Through this design, ProtoFusion shifts generative fusion from fixed operator design to structured protocol optimization under multimodal evidence.

Our contributions are summarized as follows.

– We propose ProtoFusion, a protocol-guided generative framework that represents fusion intent with a structured fusion protocol and formulates image fusion as protocolconstrained generation.

– We develop PGPR, an ofline refinement procedure that optimizes the structured fusion protocol through iterative generation, evaluation, and revision under multimodal evidence. To further improve robustness, we introduce SBC as an optional correction module.

– Extensive experiments on standard infrared and visible image fusion benchmarks show that ProtoFusion achieves strong quantitative performance, consistently higher aesthetic scores, and superior visual quality under challenging visible degradations.

The remainder of this paper is organized as follows. Section 2 reviews conventional and learning-based fusion methods, task- and language-guided approaches, generative fusion, and prompt optimization. Section 3 introduces the structured fusion protocol, details the iterative optimization process of PGPR, and describes how SBC identifies and corrects residual source inconsistencies. Section 4 presents quantitative and qualitative comparisons, evaluates downstream semantic segmentation and source consistency, and provides ablation studies, cross-generator transfer analysis, runtime evaluation, and limitations. Finally, Section 5 concludes the paper.

## 2 Related Work

## 2.1 Infrared and Visible Image Fusion

Early fusion methods rely on image decomposition and handdesigned fusion rules, while learning-based approaches use reconstruction, adversarial learning, and cross-modal feature interaction to combine infrared and visible information (Ma et al., 2019a; Zhang et al., 2021; Zhang and Demiris, 2023; Li and Wu, 2019; Ma et al., 2019b; Xu et al., 2022; Zhao et al., 2023b). Recent methods further model shared and modality-specific information or learn cross-modal dependencies (Liu et al., 2024a; Tang et al., 2025b; Kang et al., 2025). Other approaches focus on preserving complementary source information through specialized fusion mechanisms (Wang et al., 2024b; Bai et al., 2025b; Wang et al., 2026). Without ground-truth fused images, most methods rely on indirect objectives, such as intensity, gradient, and feature preservation. These objectives help retain source information but ofer limited flexibility in specifying fusion behavior across diferent scenes, degradations, and user requirements.

## 2.2 Task-aware, Semantic-aware, and Controllable Fusion

Task-aware fusion uses detection and segmentation feedback to improve downstream perception (Liu et al., 2022; Tang et al., 2022, 2023; Zhao et al., 2023a; Liu et al., 2024b). Other methods employ discriminative learning or object-level priors to emphasize important targets and their context (Liu et al., 2025b; Zhang et al., 2026; Li et al., 2025). Language-guided approaches introduce more explicit control over desired content, regions, and degradation conditions (Yi et al., 2024b; Wang et al., 2024a; Tang et al., 2025a). Vision-language descriptions and learned semantic prompts further guide multimodal feature fusion (Zhao et al., 2024; Liu et al., 2025a; Wang et al., 2025). These methods introduce flexible semantic and task guidance, but primarily apply it through learned representations or prompt-conditioned fusion networks. More importantly, the ability of vision-language foundation models to first interpret paired sources and then formulate and refine explicit fusion instructions remains underexplored.

## 2.3 Generative and Difusion-based Image Fusion

Difusion models and large generative priors have recently opened a new direction for image fusion. DDFM formulates multimodal fusion with denoising difusion models (Zhao et al., 2023c), while Dif-Fusion and Dif-IF explore difusion priors for infrared and visible image fusion and fusionknowledge-aware generation (Yue et al., 2023; Yi et al., 2024a). More general difusion-based fusion frameworks further suggest that pretrained generative models can provide a rich synthesis space for handling multiple fusion tasks and degradations (Shi et al., 2025; Cao et al., 2025; Shi et al., 2026). Across these designs, a central issue is how to constrain generation so that perceptually plausible results retain source evidence. Preference alignment introduces another dimension of control: DPOFusion adapts a latent difusion model to human, vision-language-model, and task-driven preferences (Su et al., 2026). These methods demonstrate the value of generative priors for perceptual realism and restoration capability. Unlike existing generative fusion methods that mainly rely on latent optimization, sampling constraints, or model-specific guidance, our framework uses the generative model as a fusion engine and optimizes structured protocols and prompts under multimodal evidence.

## 2.4 Prompt Optimization and Iterative Refinement

Our work is also related to automatic prompt engineering and iterative self-refinement. Automatic prompt optimization improves instructions through task feedback. APE searches for efective instructions (Zhou et al., 2023), while OPRO uses language models to iteratively optimize prompts (Yang et al., 2024). Related approaches refine model outputs or imageediting controls: Self-Refine revises generated responses through feedback (Madaan et al., 2023), Prompt-to-Prompt controls text-driven image editing through cross-attention (Hertz et al., 2023), and FusionBooster analyzes and enhances initial fusion results using learned modules (Cheng et al., 2025a). Inspired by these ideas, we introduce closed-loop prompt optimization into a generative framework for infrared and visible image fusion. Unlike generic prompt optimization for language tasks or prompt editing for single-image generation, our protocol-guided refinement is grounded in paired infrared and visible observations, modality reliability, cross-modal conflicts, and fusion-specific preservation requirements. PGPR difers from general prompt optimization by restricting updates to structured fusion instructions and evaluating each proposal against paired source images. This allows the optimized protocol to be reused across image pairs without updating the generator.

## 3 Proposed Protocol-guided Generative Fusion

Overview. Given a visible image $x ^ { \mathrm { v i s } }$ and an infrared image $x ^ { \mathrm { i r } }$ , our goal is to generate a fused image that preserves thermal saliency from the infrared modality, retains informative visible structures, and remains visually realistic. Our framework consists of a main refinement pipeline, protocolguidedprompt refinement (PGPR), and an optional correction module applied after PGPR, structured belief construction (SBC).

![](images/f0d9eda8fcc0cdb588fd66e213dc74f25bff03878dd5de545c1204847e5117e6.jpg)  
Fig. 2 Flowchart of protocol-guided prompt refinement (PGPR). PGPR is the main fusion loop. At each round, it composes a prompt from a structured protocol, generates a fused image, diagnoses the current proto-fusion result, predicts a structured edit action, and updates the protocol for the next round. The right panel expands protocol-guided prompt composition, where the protocol is organized into environment state cards, grounding principles, and a style profile. The bottom panel expands protocol-guided refinement, where diagnosis and protocol editing are performed in two stages.

As shown in Fig. 2, PGPR is the core fusion procedure. It iteratively refines a structured fusion protocol, which serves as the explicit carrier of fusion intent. At each round, the protocol is converted into a prompt, the prompt is used by the generative fusion engine to produce a fused image, and the resulting image is analyzed to update the protocol for the next round. The protocol is organized into three parts: environment state cards, grounding principles, and a style profile. Together, these components describe the source conditions, the desired fusion principles, and the target rendering style. This design makes the refinement process explicit and editable, because each round revises the protocol rather than rewriting the prompt in an unconstrained manner. After several rounds, PGPR selects the best prompt–image pair from the refinement trajectory. As shown in Fig. 3, SBC is applied only after this selection step. Given the source images and the selected PGPR output, SBC constructs a belief state to identify residual errors that may remain after PGPR. It then predicts a conservative prompt correction and performs one additional generation step to produce the final fused image.

## 3.1 Protocol-guided Prompt Refinement

As depicted in Fig. 2, PGPR contains three components: protocol-guided prompt composition, a generative fusion engine, and protocol-guided refinement. The following loop describes an ofline refinement episode on a training pair. At round �, the process can be summarized as

$$
\Pi _ { t } \xrightarrow { \mathrm { P G P C } } p _ { t } \xrightarrow { \mathrm { G F E } } y _ { t } \xrightarrow { \mathrm { d i a g } } ( c _ { t } , s _ { t } ) \xrightarrow { \mathrm { e d i t } } a _ { t } \xrightarrow { \mathrm { u p d a t e } } \Pi _ { t + 1 } .\tag{1}
$$

Here, PGPC denotes protocol-guided prompt composition, GFE denotes the generative fusion engine, and diag, edit, and update denote source-grounded diagnosis, constrained edit prediction, and validated protocol update, respectively. The variables $\Pi _ { t } , p _ { t } ,$ , and $y _ { t }$ are the protocol, composed prompt, and fused candidate; $\left( { { c _ { t } } , { s _ { t } } } \right)$ are its critique and score, and $a _ { t }$ is the proposed edit. Equation 1 depicts a proposal cycle; the acceptance rule below determines which protocol is inherited.

Protocol-guided Prompt Composition. The protocol serves as the structured carrier of fusion intent in PGPR. At round �, we write it as

$$
\Pi _ { t } = ( { \mathcal E } _ { t } , { \mathcal P } _ { t } , S _ { t } ) ,\tag{2}
$$

where $\mathcal { E } _ { t } , \mathcal { P } _ { t }$ , and $S _ { t }$ denote environment state cards, grounding principles, and the style profile, respectively, as illustrated in Fig. 2. Environment cards define which source conditions to examine, while prompt composition fills in the corresponding information from each image pair. Therefore, the protocol remains reusable, while its image-specific content adapts to each input. The initial protocol template is given in $\mathsf { A p - }$ pendix A.3. Given the source pair $( x ^ { \mathrm { v i s } } , x ^ { \mathrm { i r } } )$ and the current protocol $\Pi _ { t }$ , PGPC produces the prompt $p _ { t }$ by combining image-specific information obtained according to $\mathcal { E } _ { t }$ with the grounding principles $\mathcal { P } _ { t }$ and the style profile $S _ { t }$

![](images/a75a6e6ee7f86f1dec4695d9e803854c17cbb47d8772b428adf496d92cdcdfd8.jpg)  
Fig. 3 Overview of structured belief construction (SBC). Given the source images and the selected PGPR output $( p ^ { \star } , y ^ { \star } )$ , SBC extracts finer-grained region-level evidence and organizes it into a belief state. This belief state captures modality reliability, cross-modal conflicts, and promp attribute parsing. A conservative prompt correction is then predicted under a small, scoped edit constraint and used for an additional generation step.

Generative Fusion Engine. Given the current prompt, the generative fusion engine produces the fused image as

$$
y _ { t } = { G } ( \boldsymbol { x } ^ { \mathrm { { v i s } } } , \boldsymbol { x } ^ { \mathrm { { i r } } } , \boldsymbol { p } _ { t } ) ,\tag{3}
$$

where $\mathcal { G }$ denotes the image generator. As illustrated in Fig. 2, the generator takes the visible image, the infrared image, and the composed prompt as input, and outputs the current fusion result. This stage realizes the fusion intent encoded by the current protocol-guided prompt.

Protocol-guided Refinement. After generating $y _ { t }$ , PGPR diagnoses the current result and converts the diagnosis into a structured protocol edit. The updated protocol is then used to compose the next prompt. As shown in the bottom panel of Fig. 2, this refinement module operates in two stages. Specifically, in the first stage, a diagnostic model analyzes the current proto-fusion result and produces a textual critique together with a scalar score:

$$
\begin{array} { r } { \big ( c _ { t } , s _ { t } \big ) = \mathcal { R } _ { \mathrm { d i a g } } \big ( x ^ { \mathrm { v i s } } , x ^ { \mathrm { i r } } , \Pi _ { t } , y _ { t } \big ) , } \end{array}\tag{4}
$$

where $c _ { t }$ identifies the main deficiencies, such as missing complementary information, cross-modal inconsistencies, and visual artifacts, while $s _ { t }$ summarizes the fusion quality for candidate selection. The critic applies predefined checks with binary judgments and confidence estimates, which are combined into a weighted score covering semantic fidelity, artifact control, texture, information gain, and protocol compliance. A fatal rejection rule excludes results containing content unsupported by either source image, regardless of their scores. The remaining candidates are ranked by their scores. A strict pass additionally requires satisfying all checks and reaching a predefined score threshold, but does not automatically terminate refinement. Although these constraints reduce source inconsistencies, the critic may still miss violations.

In the second stage, the refinement module converts the diagnosis into a structured edit action:

$$
a _ { t } = \mathcal { R } _ { \mathrm { e d i t } } ( x ^ { \mathrm { v i s } } , x ^ { \mathrm { i r } } , \Pi _ { t } , y _ { t } , c _ { t } , s _ { t } ) ,\tag{5}
$$

where $a _ { t }$ is restricted to one of three actions: add an environment-state item, revise an existing principle, or add one new principle. The style profile is not edited. A deterministic validator rejects malformed, empty, repeated, or inefective edits, as well as edits with invalid conditions. For an accepted edit, only the specified slot changes; all other slots are inherited unchanged.

The protocol is then updated by

$$
\Pi _ { t + 1 } = \mathcal { U } ( \Pi _ { t } , a _ { t } ) ,\tag{6}
$$

where U (·) applies a validated edit. The resulting proposal is generated and diagnosed before it can replace the incumbent. Invalid edits leave the incumbent unchanged.

Best-so-far Selection. Let $\mathcal { T } _ { T }$ be the set of candidates actually generated and evaluated under a refinement budget �. The selected index is $t ^ { \star } = \arg \operatorname* { m a x } _ { \{ t \in \mathcal { T } , \ : s _ { t } \neq - 1 \} } \left\{ s _ { t } \right\}$ , with $p ^ { \star } =$ $p _ { \{ t ^ { \star } \} }$ and $y ^ { \star } = y _ { \{ t ^ { \star } \} }$ . A worse or fatal proposal is discarded, and the next edit starts from the best non-fatal state found so far. Candidate scores may rise or fall; only the retained critic score is non-decreasing. This property does not imply monotonic improvement in true image quality. The selected pair is passed to SBC when correction is enabled. The framework can use a black-box generator with the required paired-image and prompt interface. Appendix A.2 summarizes the refinement procedure and the remaining implementation specification.

## 3.2 Structured Belief Construction

SBC is an optional correction module, as shown in Fig. 3. It receives the prompt–image pair to be corrected, denoted $( p ^ { \star } , y ^ { \star } )$ below. This is the selected candidate when inspecting a refinement episode, or the output generated with the fixed learned protocol at test time. SBC applies at most one conservative prompt correction and does not restart PGPR.

Belief Construction. Given the source images and the selected PGPR output $( p ^ { \star } , y ^ { \star } )$ , SBC first constructs a belief state

$$
b ^ { \star } = \Phi ( x ^ { \mathrm { v i s } } , x ^ { \mathrm { i r } } , p ^ { \star } , y ^ { \star } ) ,\tag{7}
$$

where $\Phi ( \cdot )$ extracts finer-grained region-level evidence from the source images and the selected PGPR result. In this way, SBC moves from the coarse protocol-level evidence used in PGPR to a more localized evidence representation for correction. As shown in Fig. 3, it captures three aspects: modality reliability, cross-modal conflicts, and prompt attribute parsing. Intuitively, it identifies which modality is underused or over-trusted, where the remaining inconsistency lies, and which prompt attributes are most likely responsible for the current error.

Conservative Prompt Correction. Based on the belief state, SBC predicts a conservative prompt correction:

$$
\Delta p ^ { \star } = \Psi ( b ^ { \star } ) , \qquad \hat { p } = p ^ { \star } \oplus \Delta p ^ { \star } ,\tag{8}
$$

where $\hat { p }$ is the corrected prompt. This correction is intentionally local. SBC applies a small and well-scoped modification to the selected PGPR prompt, rather than regenerating a new prompt from scratch. This constraint is written as

$$
\| \Delta p ^ { \star } \| _ { \mathrm { e d i t } } \le \epsilon , \qquad \mathrm { s u p p } ( \Delta p ^ { \star } ) \subseteq M ( b ^ { \star } ) ,\tag{9}
$$

where � is a small edit budget and $M ( b ^ { \star } )$ denotes the promptedit scope inferred from the belief state.

One Extra Generation. Using the corrected prompt, SBC performs one additional generation step to obtain $\hat { y } = \mathcal G ( x ^ { \mathrm { v i s } } , x ^ { \mathrm { i r } } , \hat { p } )$ , where �ˆ denotes the refined fused image. In the absence of a valid correction, the selected PGPR output is retained as the final result, $i . e . , \hat { y } = y ^ { \star }$ . The correction is applied once after the PGPR selection stage.

## 4 Experiments

## 4.1 Experimental Setup

Experimental Protocol. ProtoFusion uses a single protocol shared by all benchmarks. We refine this protocol ofline on the FMB training split with PGPR. During refinement, each training image pair is processed with a refinement loop of

$T = 6$ rounds, while all model parameters remain frozen and only the protocol slots are updated. After refinement, the resulting protocol is fixed. For the base inference path, Qwen3-VL-8B-Instruct (Bai et al., 2025c) composes a prompt from the fixed protocol, and Qwen-Image-Edit (Wu et al., 2025) generates one fused image. PGPR runs ofline, and inference uses the fixed optimized protocol. The PGPR+SBC variant additionally constructs a belief state and performs one corrective generation when a valid correction is available. In the primary Qwen-Image-Edit experiments, we use 40 image-generation sampling steps and disable sampling in the VLM-driven modules. The ProtoFusion results in Tables 1 and 2 use PGPR+SBC on all three datasets. Results labeled PGPR in the ablations refer to the base path without SBC. The component and refinement-budget studies are evaluated on the FMB test pairs; Full PGPR and $T = 6$ use the same base PGPR results as the SBC ablation. The main benchmark comparisons and downstream segmentation use the full test sets specified below.

Datasets and Compared Methods. We evaluate ProtoFusion on the full test sets of three widely used infrared and visible image fusion benchmarks: FMB (Liu et al., 2023), MSRS (Tang et al., 2022), and RoadScene (Xu et al., 2020), containing 280, 361, and 221 image pairs, respectively. We compare against representative state-of-the-art methods, including CDDFuse (Zhao et al., 2023b), DDFM (Zhao et al., 2023c), TIMFusion (Liu et al., 2024b), Text-IF (Yi et al., 2024b), C2RF (Tang et al., 2025b), DCEvo (Liu et al., 2025b), Deno-IF (Xu et al., 2025), LUT-Fuse (Yi et al., 2025), DRFusion (Shi et al., 2026), and ISFuse (Wang et al., 2026).

Evaluation Metrics. We report six learned perceptual quality metrics, namely CLIPIQA (Wang et al., 2023), MUSIQ (Ke et al., 2021), PAQ2PIQ (Ying et al., 2020), MANIQA (Yang et al., 2022), BRISQUE (Mittal et al., 2012), and TReS (Golestaneh et al., 2022), together with three statistical fusion metrics, EN, SD, and SF (Wang et al., 2008). In addition, we report scores from UniPercept (Cao et al., 2026), a perceptual evaluation model that assesses fused images along three complementary dimensions: image aesthetics (IAA), image quality (IQA), and structure and texture (ISTA). In addition to these three scores, we report the Overall score provided by UniPercept as its aggregate assessment. Higher values are conventionally preferred for all these metrics except BRISQUE, for which lower values are better. Beyond image quality, we evaluate semantic utility through downstream segmentation using mean intersection over union (mIoU) and mean accuracy (mAcc) with a frozen model. We further assess source fidelity through expert-reviewed image-level source inconsistency rates. Additional CE, $Q _ { A B / F }$ , and VIF results are reported in Appendix B.1.

FMB  
Table 1 Quantitative comparison on FMB (top), MSRS (middle), and RoadScene (bottom). ProtoFusion uses PGPR+SBC on all three datasets. Best and second-best results are highlighted in bold and underlined, respectively.
<table><tr><td>Method</td><td>Venue</td><td>CLIPIQA↑</td><td>MUSIQ↑</td><td>PAQ2PIQ↑</td><td>MANIQA↑</td><td>BRISQUE↓</td><td>TReS↑</td><td>EN↑</td><td>SD↑</td><td>SF↑</td></tr><tr><td>CDDFuse</td><td>CVPR’23</td><td>0.1912</td><td>54.6762</td><td>71.5954</td><td>0.2624</td><td>24.9722</td><td>62.7958</td><td>6.9440</td><td>38.5810</td><td>14.6180</td></tr><tr><td>DDFM</td><td>ICCV’23</td><td>0.1622</td><td>50.9695</td><td>69.5429</td><td>0.2801</td><td>30.1938</td><td>53.1768</td><td>6.8254</td><td>31.8910</td><td>8.8404</td></tr><tr><td>TIMFusion</td><td>TPAMI&#x27;24</td><td>0.2200</td><td>51.3741</td><td>71.0020</td><td>0.2909</td><td>35.3702</td><td>55.9598</td><td>6.5551</td><td>28.6934</td><td>10.9298</td></tr><tr><td>Text-IF</td><td>CVPR’24</td><td>0.1610</td><td>54.2197</td><td>71.9926</td><td>0.2581</td><td>27.7682</td><td>61.4601</td><td>6.9249</td><td>36.4667</td><td>14.3592</td></tr><tr><td>C2RF</td><td>IJCV’25</td><td>0.1732</td><td>50.2157</td><td>70.9624</td><td>0.2507</td><td>27.9849</td><td>56.5410</td><td>6.8284</td><td>35.7855</td><td>12.0213</td></tr><tr><td>DCEvo</td><td>CVPR&#x27;25</td><td>0.1964</td><td>53.7623</td><td>71.6616</td><td>0.2651</td><td>25.9477</td><td>62.4378</td><td>6.8569</td><td>36.3743</td><td>13.8186</td></tr><tr><td>Deno-IF</td><td>NeurIPS&#x27;25</td><td>0.2747</td><td>44.9853</td><td>67.7207</td><td>0.2700</td><td>42.4478</td><td>50.4824</td><td>6.0450</td><td>32.1501</td><td>9.7676</td></tr><tr><td>LUT-Fuse</td><td>ICCV’25</td><td>0.1688</td><td>53.0099</td><td>70.9631</td><td>0.2563</td><td>19.2857</td><td>62.0295</td><td>6.7783</td><td>35.1191</td><td>15.3193</td></tr><tr><td>DRFusion</td><td>CVPR&#x27;26</td><td>0.1456</td><td>53.3829</td><td>70.7716</td><td>0.2786</td><td>43.1575</td><td>57.7663</td><td>6.5361</td><td>32.8442</td><td>10.9689</td></tr><tr><td>ISFuse</td><td>CVPR&#x27;26</td><td>0.1726</td><td>56.3705</td><td>71.4491</td><td>0.2677</td><td>21.8904</td><td>67.5581</td><td>6.8184</td><td>34.2582</td><td>14.9685</td></tr><tr><td>ProtoFusion</td><td></td><td>0.4189</td><td>64.6367</td><td>72.5243</td><td>0.3861</td><td>14.7842</td><td>71.4666</td><td>7.0029</td><td>48.3840</td><td>17.0478</td></tr></table>

<table><tr><td>Method</td><td>Venue</td><td>CLIPIQA↑</td><td>MUSIQ↑</td><td>PAQ2PIQ↑</td><td>MANIQA↑</td><td>BRISQUE↓</td><td>TReS↑</td><td>EN↑</td><td>SD↑</td><td>SF↑</td></tr><tr><td>CDDFuse</td><td>CVPR&#x27;23</td><td>0.1557</td><td>36.5769</td><td>64.9346</td><td>0.1440</td><td>32.0331</td><td>32.5303</td><td>6.7854</td><td>43.9108</td><td>11.6324</td></tr><tr><td>DDFM</td><td>ICCV’23</td><td>0.1396</td><td>33.5653</td><td>62.3264</td><td>0.1433</td><td>33.8187</td><td>30.4202</td><td>6.3479</td><td>30.0831</td><td>7.6653</td></tr><tr><td>TIMFusion</td><td>TPAMI&#x27;24</td><td>0.1267</td><td>33.5586</td><td>63.1841</td><td>0.1291</td><td>44.4031</td><td>30.1763</td><td>5.8837</td><td>40.5809</td><td>9.3600</td></tr><tr><td>Text-IF</td><td>CVPR&#x27;24</td><td>0.1593</td><td>36.3373</td><td>65.2339</td><td>0.1399</td><td>32.1248</td><td>31.9896</td><td>6.7995</td><td>45.2395</td><td>11.8888</td></tr><tr><td>C2RF</td><td>IJCV’25</td><td>0.1363</td><td>32.9537</td><td>64.2354</td><td>0.1304</td><td>32.9751</td><td>30.3190</td><td>6.5330</td><td>40.7909</td><td>10.7847</td></tr><tr><td>DCEvo</td><td>CVPR’25</td><td>0.1299</td><td>36.1123</td><td>65.0804</td><td>0.1398</td><td>29.8427</td><td>32.1065</td><td>6.7200</td><td>43.4812</td><td>11.5661</td></tr><tr><td>Deno-IF</td><td>NeurIPS&#x27;25</td><td>0.1822</td><td>32.0591</td><td>62.9750</td><td>0.1460</td><td>46.0174</td><td>28.5986</td><td>6.7202</td><td>41.5688</td><td>9.5035</td></tr><tr><td>LUT-Fuse</td><td>ICCV’25</td><td>0.1576</td><td>36.3014</td><td>65.2470</td><td>0.1361</td><td>28.6357</td><td>35.9565</td><td>6.6918</td><td>42.7811</td><td>12.7832</td></tr><tr><td>DRFusion</td><td>CVPR&#x27;26</td><td>0.1066</td><td>35.9042</td><td>64.3925</td><td>0.1452</td><td>44.9263</td><td>35.0311</td><td>6.5544</td><td>41.9877</td><td>9.9147</td></tr><tr><td>ISFuse</td><td>CVPR&#x27;26</td><td>0.1813</td><td>38.6243</td><td>65.7127</td><td>0.1550</td><td>26.3307</td><td>37.1416</td><td>6.4172</td><td>32.8123</td><td>12.0573</td></tr><tr><td>ProtoFusion</td><td></td><td>0.3213</td><td>52.2836</td><td>68.0771</td><td>0.2759</td><td>23.3807</td><td>60.7874</td><td>6.8341</td><td>44.4990</td><td>14.6222</td></tr></table>

<table><tr><td>Method</td><td>Venue</td><td>CLIPIQA↑</td><td>MUSIQ↑</td><td>PAQ2PIQ↑</td><td>MANIQA↑</td><td>BRISQUE↓</td><td>TReS↑</td><td>EN↑</td><td>SD↑</td><td>SF↑</td></tr><tr><td>CDDFuse</td><td>CVPR&#x27;23</td><td>0.1642</td><td>50.4400</td><td>68.8432</td><td>0.2066</td><td>24.8106</td><td>53.7425</td><td>7.5463</td><td>56.5661</td><td>18.9072</td></tr><tr><td>DDFM</td><td>ICCV’23</td><td>0.1788</td><td>42.8392</td><td>65.1028</td><td>0.1808</td><td>29.3473</td><td>40.0075</td><td>7.2811</td><td>42.5732</td><td>10.7103</td></tr><tr><td>TIMFusion</td><td>TPAMI&#x27;24</td><td>0.1666</td><td>35.0450</td><td>63.4439</td><td>0.1391</td><td>34.8574</td><td>35.1014</td><td>7.0091</td><td>37.9820</td><td>9.1101</td></tr><tr><td>Text-IF</td><td>CVPR&#x27;24</td><td>0.2174</td><td>45.0695</td><td>67.8238</td><td>0.1771</td><td>26.1162</td><td>44.3801</td><td>7.4537</td><td>50.6812</td><td>16.0900</td></tr><tr><td>C2RF</td><td>IJCV’25</td><td>0.1791</td><td>45.3342</td><td>67.0446</td><td>0.1854</td><td>24.1367</td><td>47.4413</td><td>7.3007</td><td>45.1677</td><td>14.4741</td></tr><tr><td>DCEvo</td><td>CVPR&#x27;25</td><td>0.1662</td><td>46.9107</td><td>67.9385</td><td>0.1967</td><td>25.6482</td><td>51.5769</td><td>7.2845</td><td>45.2938</td><td>14.9509</td></tr><tr><td>Deno-IF</td><td>NeurIPS&#x27;25</td><td>0.1789</td><td>37.4117</td><td>64.9687</td><td>0.1489</td><td>41.9599</td><td>36.1046</td><td>7.4220</td><td>47.5066</td><td>12.7819</td></tr><tr><td>LUT-Fuse</td><td>ICCV’25</td><td>0.1455</td><td>42.9383</td><td>66.7781</td><td>0.1645</td><td>24.8089</td><td>54.5410</td><td>6.9438</td><td>45.7622</td><td>19.6108</td></tr><tr><td>DRFusion</td><td>CVPR&#x27;26</td><td>0.1348</td><td>38.1821</td><td>65.4178</td><td>0.1587</td><td>38.9396</td><td>42.3365</td><td>7.0046</td><td>42.3950</td><td>10.0322</td></tr><tr><td>ISFuse</td><td>CVPR&#x27;26</td><td>0.1584</td><td>47.5480</td><td>67.9547</td><td>0.2108</td><td>23.3311</td><td>56.5832</td><td>7.0224</td><td>37.8520</td><td>15.1578</td></tr><tr><td>ProtoFusion</td><td></td><td>0.3765</td><td>62.2376</td><td>73.1191</td><td>0.3220</td><td>19.2434</td><td>77.0120</td><td>7.3201</td><td>54.6165</td><td>18.8520</td></tr></table>

Implementation Details. We instantiate the generative fusion engine G with Qwen-Image-Edit and use Qwen3-VL-8B-Instruct to implement the protocol-side modules, including protocol initialization I, prompt composition C, diagnosis $\mathcal { R } _ { \mathrm { d i a g } }$ , edit prediction $\mathcal { R } _ { \mathrm { e d i t } }$ , belief construction Φ, and prompt correction Ψ. All model parameters remain frozen throughout both protocol refinement and inference. The Qwen-Image-Edit experiments are conducted in PyTorch on two BW3000 DCUs with 64 GB of memory each. Additional details on prompt templates, model inputs, and inference settings are provided in Appendix A.

## 4.2 Performance Comparison

Quantitative Results. Table 1 compares ProtoFusion with ten competing methods on FMB, MSRS, and RoadScene. ProtoFusion achieves the best results on all six image quality metrics across the three benchmarks and leads all nine reported metrics on FMB. For the statistical metrics, Text-IF achieves the highest SD on MSRS, while CDDFuse leads EN and SD and LUT-Fuse leads SF on RoadScene. Notably, ProtoFusion maintains its advantage in perceptual quality on MSRS and RoadScene using the same protocol optimized on

Table 2 UniPercept evaluation on FMB, MSRS, and RoadScene. ProtoFusion uses PGPR+SBC on all three datasets. Higher is better; best and second-best results are bold and underlined, respectively.
<table><tr><td colspan="2"></td><td colspan="4">FMB</td><td colspan="4">MSRS</td><td colspan="4">RoadScene</td></tr><tr><td>Method</td><td>Venue</td><td>IAA</td><td>IQA</td><td>ISTA</td><td>Overall</td><td>IAA</td><td>IQA</td><td>ISTA</td><td>Overall</td><td>IAA</td><td>IQA</td><td>ISTA</td><td>Overall</td></tr><tr><td>CDDFuse</td><td>CVPR&#x27;23</td><td>21.0344</td><td>26.1464</td><td>39.4317</td><td>28.8708</td><td>27.6337</td><td>28.9799</td><td>41.5625</td><td>32.7254</td><td>18.5933</td><td>19.3558</td><td>26.8066</td><td>21.5852</td></tr><tr><td>DDFM</td><td>ICCV’23</td><td>22.0183</td><td>25.7826</td><td>38.4991</td><td>28.7667</td><td>26.6704</td><td>26.9214</td><td>38.7362</td><td>30.7760</td><td>23.2981</td><td>24.1244</td><td>35.1307</td><td>27.5177</td></tr><tr><td>TIMFusion</td><td>TPAMI&#x27;24</td><td>24.9254</td><td>27.8087</td><td>38.1953</td><td>30.3098</td><td>24.7261</td><td>26.4986</td><td>34.1963</td><td>28.4737</td><td>18.9853</td><td>19.1075</td><td>25.3255</td><td>21.1394</td></tr><tr><td>Text-IF</td><td>CVPR&#x27;24</td><td>22.8638</td><td>29.2772</td><td>41.5920</td><td>31.2443</td><td>26.6080</td><td>28.0942</td><td>41.1446</td><td>31.9489</td><td>18.9019</td><td>19.7031</td><td>27.0297</td><td>21.8782</td></tr><tr><td>C2RF</td><td>IJCV’25</td><td>19.1790</td><td>23.4826</td><td>36.4092</td><td>26.3569</td><td>25.1766</td><td>26.2832</td><td>37.9217</td><td>29.7939</td><td>18.4992</td><td>18.5967</td><td>22.2783</td><td>19.7914</td></tr><tr><td>DCEvo</td><td>CVPR&#x27;25</td><td>19.1839</td><td>22.9107</td><td>34.2859</td><td>25.4602</td><td>26.8705</td><td>28.1842</td><td>40.6603</td><td>31.9050</td><td>18.2865</td><td>18.4621</td><td>20.9072</td><td>19.2186</td></tr><tr><td>Deno-IF</td><td>NeurIPS&#x27;25</td><td>20.9625</td><td>25.6165</td><td>37.5344</td><td>28.0378</td><td>23.4865</td><td>25.7680</td><td>36.7133</td><td>28.6559</td><td>17.6776</td><td>18.5082</td><td>22.9827</td><td>19.7229</td></tr><tr><td>LUT-Fuse</td><td>ICCV’25</td><td>19.2929</td><td>23.2551</td><td>35.4638</td><td>26.0039</td><td>26.2642</td><td>27.1285</td><td>41.0203</td><td>31.4710</td><td>17.8954</td><td>18.4497</td><td>17.2435</td><td>17.8628</td></tr><tr><td>DRFusion</td><td>CVPR&#x27;26</td><td>20.0379</td><td>23.9324</td><td>35.7415</td><td>26.5706</td><td>26.0350</td><td>27.2722</td><td>39.9891</td><td>31.0987</td><td>18.7291</td><td>18.9649</td><td>24.2574</td><td>20.6505</td></tr><tr><td>ISFuse</td><td>CVPR&#x27;26</td><td>19.2496</td><td>21.8127</td><td>33.5917</td><td>24.8847</td><td>35.7133</td><td>29.7798</td><td>43.7417</td><td>36.4116</td><td>19.0317</td><td>18.8529</td><td>19.9400</td><td>19.2749</td></tr><tr><td>ProtoFusion</td><td></td><td>42.0143</td><td>53.7696</td><td>51.7683</td><td>49.1841</td><td>43.7136</td><td>44.0845</td><td>48.7273</td><td>45.5085</td><td>50.6442</td><td>54.3835</td><td>52.0622</td><td>52.3633</td></tr></table>

![](images/2c03c8aca087af449fdceef6b8d63664ec5ad36ec211cf449be22a1401b4bef1.jpg)  
Fig. 4 Representative visual comparison on FMB, MSRS, and RoadScene. The examples illustrate target visibility and rendering under visible degradations.

FMB, demonstrating its efectiveness across datasets without dataset-specific protocol refinement.

UniPercept Evaluation. Table 2 provides a complementary perceptual evaluation using UniPercept. ProtoFusion achieves the highest IAA, IQA, ISTA, and Overall scores across all three datasets. Its Overall scores reach 49.18, 45.51, and 52.36 on FMB, MSRS, and RoadScene, outperforming the strongest competing methods by 17.94, 9.10, and 24.85 points, respectively. These consistent gains across all three evaluation dimensions further support the perceptual quality of the generated fusion results.

Qualitative Results. As shown in Fig. 4, qualitative comparisons cover challenging visible conditions, including overexposure, low illumination, and noise. Across these scenes,

ProtoFusion deliberately renders the fused images in a natural visible-light style, as specified by its structured protocol. This design allows infrared information to enhance salient targets and reveal obscured structures while maintaining a coherent visual appearance. Compared with competing methods that often retain visible degradations or produce indistinct target boundaries, ProtoFusion renders clearer targets, suppresses noise, and improves the overall scene appearance. Additional nighttime comparisons are provided in Appendix B.2.

We further examine daytime scenes with small vehicles and strong backlighting, as shown in Fig. 5. In the first two rows, several competing methods introduce bright artifacts or doubled contours around vehicles, whereas ProtoFusion produces clearer vehicle boundaries and better separation from the road. In the last two rows, strong illumination obscures building structures in the visible inputs, while ProtoFusion renders more distinct rooflines and wall boundaries with a natural appearance. These comparisons highlight the central advantage of ProtoFusion: its optimized protocol explicitly controls what information to preserve or suppress from each modality and how the fused result should appear. This enables the generative model to exploit complementary infrared information and mitigate visible degradations while maintaining the intended visible-light appearance.

Table 3 Semantic segmentation results on FMB. Best and second-best results are highlighted in bold and underlined, respectively.
<table><tr><td>Method</td><td>Venue</td><td>mIoU</td><td>mAcc</td><td colspan="2">Road</td><td colspan="2">Building</td><td colspan="2">Sky</td><td colspan="2">Car</td><td colspan="2">Truck</td><td colspan="2">Bus</td><td colspan="2">Veg.</td><td colspan="2">Pole</td></tr><tr><td></td><td></td><td></td><td></td><td>IoU</td><td>Acc</td><td>IoU</td><td>Acc</td><td>IoU</td><td>Acc</td><td>IoU</td><td>Acc</td><td>IoU</td><td>Acc</td><td>IoU</td><td>Acc</td><td>IoU</td><td>Acc</td><td>IoU</td><td>Acc</td></tr><tr><td>CDDFuse</td><td>CVPR&#x27;23</td><td>0.510</td><td>0.632</td><td>0.560</td><td>0.811</td><td>0.575</td><td>0.700</td><td>0.679</td><td>0.692</td><td>0.631</td><td>0.884</td><td>0.450</td><td>0.499</td><td>0.622</td><td>0.837</td><td>0.465</td><td>0.474</td><td>0.100</td><td>0.160</td></tr><tr><td>DDFM</td><td>ICCV’23</td><td>0.472</td><td>0.588</td><td>0.536</td><td>0.724</td><td>0.585</td><td>0.709</td><td>0.692</td><td>0.704</td><td>0.555</td><td>0.786</td><td>0.269</td><td>0.343</td><td>0.565</td><td>0.797</td><td>0.485</td><td>0.497</td><td>0.089</td><td>0.144</td></tr><tr><td>TIMFusion</td><td>TPAMI&#x27;24</td><td>0.511</td><td>0.627</td><td>0.560</td><td>0.798</td><td>0.555</td><td>0.671</td><td>0.607</td><td>0.621</td><td>0.633</td><td>0.896</td><td>0.487</td><td>0.551</td><td>0.676</td><td>0.839</td><td>0.472</td><td>0.482</td><td>0.100</td><td>0.157</td></tr><tr><td>Text-IF</td><td>CVPR&#x27;24</td><td>0.513</td><td>0.637</td><td>0.565</td><td>0.807</td><td>0.579</td><td>0.706</td><td>0.675</td><td>0.687</td><td>0.641</td><td>0.888</td><td>0.439</td><td>0.518</td><td>0.643</td><td>0.864</td><td>0.471</td><td>0.481</td><td>0.095</td><td>0.147</td></tr><tr><td>C2RF</td><td>IJCV’25</td><td>0.482</td><td>0.605</td><td>0.534</td><td>0.781</td><td>0.573</td><td>0.703</td><td>0.607</td><td>0.615</td><td>0.607</td><td>0.865</td><td>0.410</td><td>0.485</td><td>0.607</td><td>0.820</td><td>0.443</td><td>0.450</td><td>0.080</td><td>0.124</td></tr><tr><td>DCEvo</td><td>CVPR’25</td><td>0.492</td><td>0.623</td><td>0.550</td><td>0.806</td><td>0.568</td><td>0.698</td><td>0.688</td><td>0.702</td><td>0.628</td><td>0.888</td><td>0.385</td><td>0.475</td><td>0.567</td><td>0.793</td><td>0.454</td><td>0.463</td><td>0.099</td><td>0.157</td></tr><tr><td>Deno-IF</td><td>NeurIPS’25</td><td>0.432</td><td>0.524</td><td>0.486</td><td>0.663</td><td>0.484</td><td>0.572</td><td>0.558</td><td>0.566</td><td>0.610</td><td>0.831</td><td>0.424</td><td>0.466</td><td>0.392</td><td>0.532</td><td>0.410</td><td>0.416</td><td>0.092</td><td>0.146</td></tr><tr><td>LUT-Fuse</td><td>ICCV’25</td><td>0.491</td><td>0.615</td><td>0.512</td><td>0.756</td><td>0.566</td><td>0.695</td><td>0.667</td><td>0.679</td><td>0.605</td><td>0.882</td><td>0.379</td><td>0.414</td><td>0.621</td><td>0.852</td><td>0.476</td><td>0.487</td><td>0.100</td><td>0.157</td></tr><tr><td>DRFusion</td><td>CVPR&#x27;26</td><td>0.458</td><td>0.591</td><td>0.530</td><td>0.790</td><td>0.555</td><td>0.681</td><td>0.658</td><td>0.678</td><td>0.604</td><td>0.902</td><td>0.283</td><td>0.351</td><td>0.505</td><td>0.730</td><td>0.423</td><td>0.431</td><td>0.105</td><td>0.169</td></tr><tr><td>ISFuse</td><td>CVPR&#x27;26</td><td>0.507</td><td>0.618</td><td>0.538</td><td>0.769</td><td>0.577</td><td>0.703</td><td>0.634</td><td>0.641</td><td>0.633</td><td>0.875</td><td>0.456</td><td>0.503</td><td>0.668</td><td>0.838</td><td>0.450</td><td>0.459</td><td>0.096</td><td>0.152</td></tr><tr><td>ProtoFusion</td><td></td><td>0.554</td><td>0.684</td><td>0.610</td><td>0.842</td><td>0.587</td><td>0.726</td><td>0.699</td><td>0.711</td><td>0.662</td><td>0.923</td><td>0.502</td><td>0.605</td><td>0.725</td><td>0.938</td><td>0.547</td><td>0.562</td><td>0.101</td><td>0.169</td></tr></table>

![](images/8091c731f4833876ce49b12a6797e388b7ea6333a444e434d060fc7d44e28079.jpg)  
Fig. 5 Qualitative comparisons on degraded daytime scenes with haze and strong illumination.

## 4.3 Downstream Semantic Segmentation

We evaluate downstream semantic segmentation on the FMB test pairs using the same frozen CLIPer model (Sun et al., 2025) for each fusion method. FMB category names serve as text prompts, and the predicted segmentation maps are evaluated against pixel-level annotations.

Table 3 presents the results against ten competing methods. ProtoFusion achieves an mIoU of 0.554 and an mAcc of 0.684, improving upon the strongest competitor, Text-IF, by 4.1 and 4.7 percentage points, respectively. ProtoFusion also achieves the highest IoU in seven of the eight reported classes, including car, truck, and bus. These results demonstrate improved semantic segmentation performance under the same frozen evaluation model. A key design choice helps explain these gains: ProtoFusion deliberately renders fused images with a natural visible-light appearance. This design preserves familiar visible colors, textures, and scene structures while using infrared information to enhance salient targets and obscured details. The resulting images are better suited to the visual representations learned by the frozen CLIPer model, helping explain the improvements across both object and scene segmentation.

## 4.4 Hallucination Study

We evaluate image-level source inconsistencies on all FMB test pairs through a two-stage assessment. First, GPT-6 Astra examines each fused image alongside its infrared and visible sources to identify potential inconsistencies. Domain experts then manually review all fused results, including those initially judged error-free by the model, and determine the final error labels. All reported error rates are based on this comprehensive human review.

Table 4 Image-level error rates (%) on FMB; lower is better. Categories may overlap; Any error is their union. Only definite errors enter the numerator.
<table><tr><td>Error category</td><td>Human Prompt</td><td>PGPR</td><td>PGPR+SBC</td></tr><tr><td>Content alteration</td><td>36.43</td><td>26.43</td><td>14.29</td></tr><tr><td>Geometric distortion</td><td>5.00</td><td>1.79</td><td>0.71</td></tr><tr><td>Information omission</td><td>4.64</td><td>3.93</td><td>2.86</td></tr><tr><td>Appearance artifacts</td><td>25.36</td><td>11.43</td><td>3.57</td></tr><tr><td>Any error</td><td>53.93</td><td>38.21</td><td>19.64</td></tr></table>

Error Categories. We consider four categories of source inconsistencies. Content alteration includes unsupported additions or changes to source-supported semantic attributes, text, symbols, or states. Geometric distortion covers erroneous shape, count, duplication, or spatial arrangement beyond source registration diferences. Information omission denotes the loss of meaningful source-supported objects or structures. Appearance artifacts include newly introduced or amplified halos, ghosting, noise, smearing, or unnatural rendering. The categories may overlap; Any error denotes their image-level union.

As shown in Tab. 4, PGPR reduces the image-level Any error rate from 53.93% to 38.21%, while SBC further lowers it to 19.64%. PGPR mitigates recurring generation errors by optimizing a structured fusion protocol through sourcegrounded diagnosis and constrained edits, improving control over modality contributions, structural preservation, and visual appearance. SBC complements this shared protocol with region-level analysis of modality reliability and cross-modal conflicts, enabling conservative prompt corrections tailored to residual errors in each image. Together, these mechanisms reduce appearance artifacts from 25.36% to 3.57%, geometric distortion from 5.00% to 0.71%, and content alteration from 36.43% to 14.29%.

Residual errors. The fatal gate rejects source violations identified by the critic, while undetected errors can persist in the generated results. Appendix C.2 traces an example in which an unsupported white umbrella interpretation propagates into the fused output.

## 4.5 Ablation Study

Protocol Representation and Refinement. Table 5 evaluates the contributions of protocol design and ofline refinement using the same inputs, VLM, generator, random seeds, and generation settings. Overall improves from 38.13 with Human Prompt to 39.30 with Initial Protocol, and further reaches 47.78 with Full PGPR. Protocol initialization provides a modest gain of 1.17 points, whereas ofline refinement contributes an additional 8.48 points. This substantial improvement highlights the importance of refining fusion instructions based on diagnosed failures, rather than relying on a fixed protocol. Notably, these gains are achieved without modifying the underlying generative models. Appendix C.1 illustrates the fixed-prompt failure modes associated with this comparison.

Table 5 Protocol components on the FMB test pairs (UniPercept; higher is better; best in bold).
<table><tr><td>Setting</td><td>IAA</td><td>IQA</td><td>ISTA</td><td>Overall</td></tr><tr><td>Human Prompt</td><td>29.1049</td><td>39.2397</td><td>46.0433</td><td>38.1293</td></tr><tr><td>Initial Protocol</td><td>31.6094</td><td>40.7125</td><td>45.5683</td><td>39.2967</td></tr><tr><td>Principles Only</td><td>30.6170</td><td>40.6623</td><td>45.9330</td><td>39.0708</td></tr><tr><td>w/o State Cards</td><td>34.4786</td><td>44.3286</td><td>48.4580</td><td>42.4217</td></tr><tr><td>w/o Style Profile</td><td>35.3647</td><td>45.3844</td><td>48.7804</td><td>43.1765</td></tr><tr><td>Full PGPR</td><td>39.8040</td><td>52.3004</td><td>51.2281</td><td>47.7775</td></tr></table>

Table 6 Refinement budget on all FMB test pairs (UniPercept; higher is better; best in bold).
<table><tr><td>Budget</td><td>IAA</td><td>IQA</td><td>ISTA</td><td>Overall</td></tr><tr><td>T = 1</td><td>31.6094</td><td>40.7125</td><td>45.5683</td><td>39.2967</td></tr><tr><td>T = 3</td><td>32.6509</td><td>44.7830</td><td>48.3902</td><td>41.9414</td></tr><tr><td>T = 6</td><td>39.8040</td><td>52.3004</td><td>51.2281</td><td>47.7775</td></tr><tr><td>T = 9</td><td>40.0701</td><td>51.5103</td><td>51.2759</td><td>47.6188</td></tr></table>

The component ablations further demonstrate the importance of the protocol structure. Removing either state cards or the style profile reduces the Overall score by more than four points, with consistent declines in IAA, IQA, and ISTA. State cards help adapt fusion instructions to source conditions and modality reliability, while the style profile provides explicit control over the target appearance. The Principles Only setting also performs substantially worse than Full PGPR, indicating that grounding principles alone are insuficient. Together, these results highlight the complementary roles of source-conditioned guidance, grounding principles, and appearance control in determining what information to preserve and how to present it.

Refinement Budget. As shown in Table 6, increasing the ofline refinement budget from � = 1 to � = 6 improves Overall from 39.30 to 47.78, demonstrating the benefit of iterative diagnosis and protocol refinement. Further increasing � to 9 yields only marginal gains in some evaluation dimensions, while Overall slightly decreases. We therefore adopt � = 6 to balance fusion performance and ofline refinement cost. Since protocol optimization is performed ofline, the inference procedure remains unchanged across diferent refinement budgets.

Protocol Optimization. We further compare PGPR with open-ended VLM-based prompt optimization to examine the benefits of structured protocol refinement. Starting from the Human Prompt, the VLM-optimized Prompt baseline iteratively revises the prompt through unconstrained VLMguided search, without predefined protocol slots or validated edit operations. As shown in Tab. 7, this baseline improves IAA but reduces IQA and ISTA, yielding only a modest gain in Overall. In contrast, PGPR improves all three dimensions and achieves an Overall score of 47.78, outperforming the VLM-optimized Prompt baseline by more than eight points. By refining structured fusion instructions through sourcegrounded diagnosis and validated edits, PGPR provides more consistent control over modality contributions, structural preservation, and visual appearance.

![](images/7db427bb0f068bb2f23ce1f0fadb7ae76b4ad9de447f4713cf77cf0694d80f9b.jpg)  
Fig. 6 Ablation study on the influence of infrared input. Each group shows the original IR/VIS inputs, a shufled IR input, and the results generated under VIS-only, shufled IR+VIS, and IR+VIS settings.

Table 7 Prompt optimization and SBC comparison on the full FMB test set using UniPercept. Higher is better.
<table><tr><td>Method</td><td>IAA</td><td>IQA</td><td>ISTA</td><td>Overall</td></tr><tr><td>Human Prompt</td><td>29.1049</td><td>39.2397</td><td>46.0433</td><td>38.1293</td></tr><tr><td>VLM-optimized Prompt</td><td>37.5263</td><td>38.2645</td><td>43.3942</td><td>39.7283</td></tr><tr><td>PGPR</td><td>39.8040</td><td>52.3004</td><td>51.2281</td><td>47.7775</td></tr><tr><td>PGPR+SBC</td><td>42.0143</td><td>53.7696</td><td>51.7683</td><td>49.1841</td></tr></table>

Role of the Infrared Input. As shown in Fig. 6, we compare VIS-only, shufled IR+VIS, and original IR+VIS generation under the same optimized protocol and generation settings. In the shufled IR+VIS setting, we replace the original infrared input with an unrelated infrared image from another scene, while keeping the visible input unchanged. VIS-only generation can improve visual appearance but often misses infrared-salient targets and structures obscured in the visible input. In contrast, shufled IR introduces substantial content alterations, including unrelated pedestrians, vehicles, and background structures that are absent from the original scene. Using the original IR input enables ProtoFusion to recover complementary infrared information while maintaining a coherent visible-light appearance. These comparisons demonstrate that infrared information directly influences the generated scene content, rather than merely serving as auxiliary guidance for visible image enhancement. The destructive efects of shufled IR further highlight the importance of source-grounded guidance and cross-modal consistency control in generative image fusion.

Refinement Dynamics. To better understand how PGPR improves the fusion result, we visualize the intermediate outputs along the refinement trajectory in Fig. 7. The refinement trajectories illustrate how PGPR adjusts modality contributions in response to diagnosed generation errors.

In the first trajectory, the early outputs preserve the visible appearance but under-emphasize thermally salient pedestrians. Through protocol refinement, the prompt progressively strengthens the use of infrared evidence for pedestrian regions, leading to clearer target responses and more distinguishable boundaries. In the second trajectory, directly injecting infrared saliency improves target visibility but may also introduce over-enhanced or blob-like structures. PGPR then introduces more constrained instructions, encouraging the model to preserve infrared target shapes while avoiding unnatural target appearance. Together, these examples show how PGPR adjusts both target emphasis and rendering constraints through iterative refinement.

![](images/3ad29c8961db1ff2a5eb947df29d2d47f04b10cca3fdef44ec81cd10f0d614cb.jpg)  
Fig. 7 Visualization of the PGPR refinement trajectory. From left to right, we show the intermediate fused results across refinement rounds.

![](images/f166e63379b1b9a93ad96a92752617a53617273103d7ab966ee6e5a05a8edb63.jpg)  
Fig. 8 Visual comparison of PGPR and PGPR+SBC on representative cases. SBC adjusts local target emphasis and appearance through a single prompt correction.

SBC Correction. SBC further complements PGPR by addressing image-specific errors that remain after protocol optimization. It analyzes region-level modality reliability and cross-modal conflicts to guide conservative prompt corrections. As shown in Fig. 8, SBC adjusts target emphasis and appearance in representative cases, further improving Overall to 49.18 in Tab. 7. The PGPR baseline matches Full PGPR in Tab. 5 and � = 6 in Tab. 6, with all three studies evaluated on the FMB test set. These results highlight the complementary benefits of structured protocol optimization and targeted correction of residual errors.

Protocol Analysis. We analyze the learned protocol in terms of both its intrinsic functional structure and its prompt-level instantiation. The protocol is decomposed into intermediate functional components, projected into a shared functional signature space, and clustered into intrinsic factors. As shown in Fig. 9a, the most informative pattern is the <IR Saliency Transfer> factor, which is concentrated on the detail axis and largely decoupled from color-related behavior. This suggests that the learned protocol assigns IR information a selective role, contributing salient structural details while leaving visible appearance regulation to color-related controls.

We further parse the final prompts for the FMB dataset into semantic control units and normalize them over the same functional axes. Figure 9b shows that protocol-derived prompts occupy a bounded allocation range around the human-crafted prompt, with variations concentrated on specific control axes. Together, these results indicate that the learned protocol organizes fusion intent into separable functional factors and instantiates them as controlled prompt-level reweighting before generation.

![](images/598851577bae5c75c1cc627ab4f439db50ed1a281da2a90db7214e39e636a9b8.jpg)

(a) Intrinsic functional factors  
![](images/bb57c947f099fffff5f9da542a5c97ed46547a2e17405cebb9485b65f4091932.jpg)  
(b) Prompt-level control allocation  
Fig. 9 Analysis of the learned protocol. (a) Intrinsic functional factors. (b) Prompt-level control allocation. Blue distributions show the final prompts; blue points and bars show the median and interquartile range; black diamonds denote the human-crafted prompt.

## 4.6 Cross-generator Protocol Transfer

The main quantitative experiments use Qwen-Image-Edit as the image generator. To examine whether the learned protocol can be reused with another generator, we replace Qwen-Image-Edit with GPT-Image-2 while retaining the same optimized protocol. We directly reuse the protocol optimized with Qwen-Image-Edit for GPT-Image-2. This comparison therefore examines cross-generator reuse of a fixed learned protocol within the ProtoFusion formulation.

Figure 10 compares the two instantiations on the same infrared and visible image pairs. Both produce visible-style outputs that emphasize infrared-supported targets, while the underlying generator afects their appearance. In these nighttime examples, the GPT-Image-2 outputs show cleaner local structures and sharper rendering around pedestrians, building contours, roadside signs, and street-light regions. The diferences indicate that protocol guidance and generator capability jointly influence the final result.

These examples support a modular interpretation of Proto-Fusion: the learned protocol specifies what evidence should be preserved, emphasized, or suppressed, while the generator determines how these instructions are rendered. Reusing the protocol without reoptimization demonstrates its applicability to a second generator in the illustrated cases. These examples demonstrate that the optimized protocol can guide diferent image generators. A matched quantitative evaluation would further assess cross-generator performance in perceptual quality, source consistency, and downstream utility.

## 4.7 Inference Runtime

ProtoFusion optimizes a shared fusion protocol ofline and reuses it across test images, avoiding repeated protocol refinement during inference. With the optimized protocol fixed, inference requires only prompt composition and image generation.

As shown in Tab. 8, the Qwen-Image-Edit implementation takes approximately 188 s per image pair, including 38 s for prompt composition and 150 s for image generation. We further evaluate Qwen-Image-2.1-Turbo (Qwen, 2026), a newer and more lightweight generative model with a 7B-parameter image generation backbone and only eight denoising steps. Under a separate execution configuration, prompt composition and image generation together take approximately 9 s per pair. These results show that image generation dominates the computational cost in the original implementation, while the optimized protocol can be reused with more eficient generative models.

## 4.8 Limitations and Discussion

Hallucination and fidelity. ProtoFusion aims to generate fused images with a natural visible appearance for practical visual interpretation. Under severe visible degradation, infrared inputs can reveal salient structures but cannot fully determine their visible appearance. The generator may therefore introduce visually plausible details that are unsupported by either source. Although PGPR and SBC reduce source inconsistencies, residual errors persist (Table 4). Recent advances in image editing, such as GPT-Image-2.5, have improved reference-image preservation and editing precision (OpenAI, 2026). As visual grounding and instruction following improve, such hallucinations may become less frequent without changing the core PGPR formulation.

Protocol feedback and control. PGPR relies on VLM diagnoses and proxy scores that may overlook source inconsistencies. Its best-so-far selection ensures a non-decreasing retained proxy score, but cannot guarantee improved source fidelity. SBC further identifies residual errors through regionlevel analysis, yet implements corrections through language instructions rather than direct image constraints. Consequently, a narrowly scoped prompt edit may alter image regions beyond its intended scope. Stronger image-level constraints and verification could make these corrections more reliable.

(Qwen)  
Table 8 Inference latency per image pair, including prompt composition and one image-generation step (excluding optional SBC).
<table><tr><td>Generator configuration</td><td>Time (s/pair)</td></tr><tr><td>Qwen-Image-Edit</td><td>~188</td></tr><tr><td>Qwen-Image-2.1-Turbo</td><td>~9</td></tr></table>

![](images/f937adbddf13a4feb5bf9626f9ebdc3f84b1283b3ec961df5dc3aa761fa005f4.jpg)  
Fig. 10 Comparison of image generators under a fixed learned protocol. The columns show the infrared and visible inputs, followed by ProtoFusion with Qwen-Image-Edit and GPT-Image-2. The protocol learned with Qwen-Image-Edit is reused without reoptimization.

Computational cost. The current Qwen-Image-Edit implementation requires approximately 188 s per image pair for prompt composition and generation. With the newer Qwen-Image-2.1-Turbo, the observed base-path latency drops to approximately 9 s (Section 4.7). Although these measurements exclude the optional SBC module and real-time fusion remains challenging, this substantial improvement highlights the potential of more eficient generative backbones. Continued advances in generative foundation models are expected to further improve the inference eficiency and practical applicability of ProtoFusion.

## 5 Conclusion

We presented ProtoFusion, a protocol-guided generative framework for infrared and visible image fusion. ProtoFusion replaces fixed fusion operator design with protocolconstrained generation and uses ofline refinement to optimize a structured fusion protocol that is fixed at inference time.

The optional SBC module further addresses residual source inconsistencies through targeted corrections. Experiments on standard benchmarks show that this design yields strong quantitative performance, higher UniPercept scores, and better visual quality, particularly when the visible modality is degraded.

More broadly, ProtoFusion introduces a new paradigm for image fusion. It makes fusion intent explicit and optimizes a structured protocol to guide foundation-model generation. This shifts the focus from designing fixed fusion operators to controlling what information to preserve and how to present it. Source fidelity and computational cost remain challenges. However, continued advances in foundation models may further improve fusion quality, reliability, and eficiency. Proto-Fusion can benefit from these advances without changing its core framework. We hope this paradigm inspires further research on foundation-model-driven image fusion.

## Data Availability

The FMB, MSRS, and RoadScene datasets used in this study are publicly available.

## Conflict of Interest

The authors afirm that there are no commercial or associative relationships that could be perceived as a conflict of interest related to the submitted work.

## References

Bai H, Zhang J, Zhao Z, Wu Y, Deng L, Cui Y, Feng T, Xu S (2025a) Task-driven image fusion with learnable fusion loss. In: Proceedings of the Computer Vision and Pattern Recognition Conference, pp 7457–7468

Bai H, Zhao Z, Zhang J, Wu Y, Deng L, Cui Y, Jiang B, Xu S (2025b) ReFusion: Learning image fusion from reconstruction with learnable loss via meta-learning. International Journal of Computer Vision 133(5):2547–2567, DOI 10.1007/s11263-024-02256-8

Bai S, Cai Y, Chen R, Chen K, Chen X, Cheng Z, Deng L, Ding W, Gao C, Ge C, Ge W, Guo Z, Huang Q, Huang J, Huang F, Hui B, Jiang S, Li Z, Li M, Li M, Li K, Lin Z, Lin J, Liu X, Liu J, Liu C, Liu Y, Liu D, Liu S, Lu D, Luo R, Lv C, Men R, Meng L, Ren X, Ren X, Song S, Sun Y, Tang J, Tu J, Wan J, Wang P, Wang P, Wang Q, Wang Y, Xie T, Xu Y, Xu H, Xu J, Yang Z, Yang M, Yang J, Yang A, Yu B, Zhang F, Zhang H, Zhang X, Zheng B, Zhong H, Zhou J, Zhou F, Zhou J, Zhu Y, Zhu K (2025c) Qwen3-VL technical report. URL https://arxiv.org/abs/2511.21631, 2511.21631

Cao L, Zhang H, Li C, Ma J (2025) Projection-manifold regularized latent difusion for robust general image fusion. In: Advances in Neural Information Processing Systems

Cao S, Li J, Li X, Pu Y, Zhu K, Gao Y, Luo S, Xin Y, Qin Q, Zhou Y, Chen X, Zhang W, Fu B, Qiao Y, Liu Y (2026) Unipercept: Towards unified perceptual-level image understanding across aesthetics, quality, structure, and texture. In: Proceedings of the International Conference on Machine Learning

Cheng C, Xu T, Wu XJ, Li H, Li X, Kittler J (2025a) FusionBooster: A unified image fusion boosting paradigm. International Journal of Computer Vision 133(5):3041– 3058, DOI 10.1007/s11263-024-02266-6

Cheng C, Xu T, Wu XJ, Li H, Li X, Tang Z, Kittler J (2025b) TextFusion: Unveiling the power of textual semantics for controllable image fusion. Information Fusion 117:102790, DOI 10.1016/j.infus.2024.102790

Golestaneh SA, Dadsetan S, Kitani KM (2022) No-reference image quality assessment via transformers, relative ranking, and self-consistency. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp 3209–3218

Hertz A, Mokady R, Tenenbaum J, Aberman K, Pritch Y, Cohen-Or D (2023) Prompt-to-prompt image editing with cross-attention control. In: Proceedings of the International Conference on Learning Representations

Kang H, Li H, Xu T, Wu XJ, Wang R, Cheng C, Kittler J (2025) SMLNet: A SPD manifold learning network for infrared and visible image fusion. International Journal of Computer Vision 133(12):8642–8663, DOI 10.1007/s112 63-025-02578-1

Ke J, Wang Q, Wang Y, Milanfar P, Yang F (2021) MUSIQ: Multi-scale image quality transformer. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp 5148–5157

Li H, Wu XJ (2019) DenseFuse: A fusion approach to infrared and visible images. IEEE Transactions on Image Processing 28(5):2614–2623, DOI 10.1109/TIP.2018.2887342

Li H, Bian C, Zhang Z, Song X, Li X, Wu XJ (2025) OCCO: LVM-guided infrared and visible image fusion framework based on object-aware and contextual contrastive learning. International Journal of Computer Vision 133(9):6611– 6635, DOI 10.1007/s11263-025-02507-2

Liu J, Fan X, Huang Z, Wu G, Liu R, Zhong W, Luo Z (2022) Target-aware dual adversarial learning and a multi-scenario multi-modality benchmark to fuse infrared and visible for object detection. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp 5802–5811

Liu J, Liu Z, Wu G, Ma L, Liu R, Zhong W, Luo Z, Fan X (2023) Multi-interactive feature learning and a full-time multi-modality benchmark for image fusion and segmentation. In: Proceedings of the IEEE/CVF International

Conference on Computer Vision, pp 8115–8124

Liu J, Lin R, Wu G, Liu R, Luo Z, Fan X (2024a) CoCoNet: Coupled contrastive learning network with multi-level feature ensemble for multi-modality image fusion. International Journal of Computer Vision 132(5):1748–1775, DOI 10.1007/s11263-023-01952-1

Liu J, Li X, Wang Z, Jiang Z, Zhong W, Fan W, Xu B (2025a) PromptFusion: Harmonized semantic prompt learning for infrared and visible image fusion. IEEE/CAA Journal of Automatica Sinica 12(3):502–515, DOI 10.1109/JAS.20 24.124878

Liu J, Zhang B, Mei Q, Li X, Zou Y, Jiang Z, Ma L, Liu R, Fan X (2025b) DCEvo: Discriminative cross-dimensional evolutionary learning for infrared and visible image fusion. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp 2226–2235

Liu J, Li X, Mei Q, Xu H, Jiang Z, Ma L, Liu R, Fan X (2026) Bridging human evaluation to infrared and visible image fusion. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition

Liu R, Liu Z, Liu J, Fan X, Luo Z (2024b) A task-guided, implicitly-searched and meta-initialized deep model for image fusion. IEEE Transactions on Pattern Analysis and Machine Intelligence 46(10):6594–6609

Ma J, Ma Y, Li C (2019a) Infrared and visible image fusion methods and applications: A survey. Information Fusion 45:153–178, DOI 10.1016/j.infus.2018.02.004

Ma J, Yu W, Liang P, Li C, Jiang J (2019b) FusionGAN: A generative adversarial network for infrared and visible image fusion. Information Fusion 48:11–26, DOI 10.1016/ j.infus.2018.09.004

Ma J, Tang L, Fan F, Huang J, Mei X, Ma Y (2022) SwinFusion: Cross-domain long-range learning for general image fusion via swin transformer. IEEE/CAA Journal of Automatica Sinica 9(7):1200–1217, DOI 10.1109/JAS.2022.1 05686

Madaan A, Tandon N, Gupta P, Hallinan S, Gao L, Wiegrefe S, Alon U, Dziri N, Prabhumoye S, Yang Y, Gupta S, Majumder BP, Hermann K, Welleck S, Yazdanbakhsh A, Clark P (2023) Self-refine: Iterative refinement with selffeedback. In: Advances in Neural Information Processing Systems

Mittal A, Moorthy AK, Bovik AC (2012) No-reference image quality assessment in the spatial domain. IEEE Transactions on Image Processing 21(12):4695–4708, DOI 10.1109/TIP.2012.2214050

OpenAI (2026) Introducing chatgpt images 2.5. URL https: //openai.com/index/introducing-chatgpt-ima ges-2-5/

Qwen (2026) Qwen-Image-2.1-Turbo. Hugging Face model repository, URL https://huggingface.co/Qwen/Qw en-Image-2.1-Turbo, released 2026-10-09. Accessed 2026-10-21

Shi Y, Liu Y, Cheng J, Wang ZJ, Chen X (2025) VDMUFusion: A versatile difusion model-based unsupervised framework for image fusion. IEEE Transactions on Image Processing 34:441–454, DOI 10.1109/TIP.2024.3512365

Shi Y, Liu Y, Wu ZC, Cheng J, Li H, Chen X (2026) Degradation-Robust Fusion: An Eficient Degradation-Aware Difusion Framework for Multimodal Image Fusion in Arbitrary Degradation Scenarios. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition

Su W, Zhang S, Han Y, Zhuang J, Huang Y, Zhang Q (2026) Fusion in Your Way: Aligning Image Fusion with Heterogeneous Demands via Direct Preference Optimization. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp 41499–41509

Sun L, Cao J, Xie J, Jiang X, Pang Y (2025) Cliper: Hierarchically improving spatial representation of clip for open-vocabulary semantic segmentation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp 23199–23209

Tang L, Yuan J, Ma J (2022) Image fusion in the loop of highlevel vision tasks: A semantic-aware real-time infrared and visible image fusion network. Information Fusion 82:28–42, DOI 10.1016/j.infus.2021.12.004

Tang L, Zhang H, Xu H, Ma J (2023) Rethinking the necessity of image fusion in high-level vision tasks: A practical infrared and visible image fusion network based on progressive semantic injection and scene fidelity. Information Fusion 99:101870, DOI 10.1016/j.infus.2023.101870

Tang L, Wang Y, Cai Z, Jiang J, Ma J (2025a) ControlFusion: A controllable image fusion network with language-vision degradation prompts. In: Advances in Neural Information Processing Systems

Tang L, Yan Q, Xiang X, Fang L, Ma J (2025b) C2RF: Bridging multi-modal image registration and fusion via commonality mining and contrastive learning. International Journal of Computer Vision 133(8):5262–5280, DOI 10.1007/s11263-025-02427-1

Wang H, Zhang H, Yi X, Xiang X, Fang L, Ma J (2024a) TeRF: Text-driven and region-aware flexible visible and infrared image fusion. In: Proceedings of the 32nd ACM International Conference on Multimedia, pp 935–944, DOI 10.1145/3664647.3680971

Wang J, Chan KCK, Loy CC (2023) Exploring CLIP for assessing the look and feel of images. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol 37, pp 2555–2563, DOI 10.1609/aaai.v37i2.25353

Wang Q, Shen Y, Jin J (2008) Performance evaluation of image fusion techniques. In: Image Fusion: Algorithms and Applications, Academic Press, pp 469–492

Wang W, Deng LJ, Ran R, Vivone G (2024b) A general paradigm with detail-preserving conditional invertible net-

work for image fusion. International Journal of Computer Vision 132(4):1029–1054, DOI 10.1007/s11263-023-019 24-5

Wang X, Guan Z, Qian W, Wang C, Ma R (2026) Multi-modal image fusion via intervention-stable feature learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp 33827–33837

Wang Z, Zhao L, Zhang J, Song R, Song H, Meng J, Wang S (2025) Multi-text guidance is important: Multi-modality image fusion via large generative vision-language model. International Journal of Computer Vision 133(7):4646– 4668, DOI 10.1007/s11263-025-02409-3

Wu C, Li J, Zhou J, Lin J, Gao K, Yan K, ming Yin S, Bai S, Xu X, Chen Y, Chen Y, Tang Z, Zhang Z, Wang Z, Yang A, Yu B, Cheng C, Liu D, Li D, Zhang H, Meng H, Wei H, Ni J, Chen K, Cao K, Peng L, Qu L, Wu M, Wang P, Yu S, Wen T, Feng W, Xu X, Wang Y, Zhang Y, Zhu Y, Wu Y, Cai Y, Liu Z (2025) Qwen-Image technical report. URL https://arxiv.org/abs/2508.02324, 2508.02324

Xu H, Ma J, Le Z, Jiang J, Guo X (2020) Fusiondn: A unified densely connected network for image fusion. In: proceedings of the Thirty-Fourth AAAI Conference on Artificial Intelligence

Xu H, Ma J, Jiang J, Guo X, Ling H (2022) U2Fusion: A unified unsupervised image fusion network. IEEE Transactions on Pattern Analysis and Machine Intelligence 44(1):502– 518, DOI 10.1109/TPAMI.2020.3012548

Xu H, Li Y, Deng Y, Ma J, Liu G (2025) Deno-IF: Unsupervised noisy visible and infrared image fusion method. In: Advances in Neural Information Processing Systems

Yang C, Wang X, Lu Y, Liu H, Le QV, Zhou D, Chen X (2024) Large language models as optimizers. In: Proceedings of the International Conference on Learning Representations

Yang S, Wu T, Shi S, Lao S, Gong Y, Cao M, Wang J, Yang Y (2022) MANIQA: Multi-dimension attention network for no-reference image quality assessment. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pp 1190–1199

Yang Z, Zhang Y, Li H, Liu Y (2025) Instruction-driven fusion of infrared-visible images: Tailoring for diverse downstream tasks. Information Fusion 121:103148, DOI 10.1016/j.infus.2025.103148

Yi X, Tang L, Zhang H, Xu H, Ma J (2024a) Dif-IF: Multimodality image fusion via difusion model with fusion knowledge prior. Information Fusion 110:102450, DOI 10.1016/j.infus.2024.102450

Yi X, Xu H, Zhang H, Tang L, Ma J (2024b) Text-IF: Leveraging semantic text guidance for degradation-aware and interactive image fusion. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp 27026–27035

Yi X, Zhang Y, Xiang X, Yan Q, Xu H, Ma J (2025) LUT-Fuse: Towards extremely fast infrared and visible image fusion via distillation to learnable look-up tables. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp 14559–14568

Ying Z, Niu H, Gupta P, Mahajan D, Ghadiyaram D, Bovik A (2020) From patches to pictures (PaQ-2-PiQ): Mapping the perceptual space of picture quality. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp 3575–3585

Yue J, Fang L, Xia S, Deng Y, Ma J (2023) Dif-Fusion: Toward high color fidelity in infrared and visible image fusion with difusion models. IEEE Transactions on Image Processing 32:5705–5720, DOI 10.1109/TIP.2023.3322046

Zhang H, Xu H, Tian X, Jiang J, Ma J (2021) Image fusion meets deep learning: A survey and perspective. Information Fusion 76:323–336, DOI 10.1016/j.infus.2021.06.008

Zhang X, Demiris Y (2023) Visible and infrared image fusion using deep learning. IEEE Transactions on Pattern Analysis and Machine Intelligence 45(8):10535–10554, DOI 10.1109/TPAMI.2023.3261282

Zhang Z, Li H, Xu T, Wu X, Awais M, Kittler J (2026) A color information driven collaborative training of dual task parallel network for visible and thermal infrared image fusion and saliency object detection. International Journal of Computer Vision 134(5):Article 227, DOI 10.1007/s1 1263-026-02793-4

Zhao W, Xie S, Zhao F, He Y, Lu H (2023a) MetaFusion: Infrared and visible image fusion via meta-feature embedding

from object detection. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp 13955–13965

Zhao Z, Bai H, Zhang J, Zhang Y, Xu S, Lin Z, Timofte R, Van Gool L (2023b) CDDFuse: Correlation-driven dualbranch feature decomposition for multi-modality image fusion. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp 5906–5916

Zhao Z, Bai H, Zhu Y, Zhang J, Xu S, Zhang Y, Zhang K, Meng D, Timofte R, Van Gool L (2023c) DDFM: Denoising difusion model for multi-modality image fusion. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp 8082–8093

Zhao Z, Deng L, Bai H, Cui Y, Zhang Z, Zhang Y, Qin H, Chen D, Zhang J, Wang P, Van Gool L (2024) Image fusion via vision-language model. In: Proceedings of the 41st International Conference on Machine Learning, PMLR, Proceedings of Machine Learning Research, vol 235, pp 60749–60765

Zhao Z, Cui Y, Deng L, Bai H, Qin H, Feng T, Schindler K (2026) MambaVF: State space model for eficient video fusion. URL https://arxiv.org/abs/2602.06017, 2602.06017

Zhou Y, Muresanu AI, Han Z, Paster K, Pitis S, Chan H, Ba J (2023) Large language models are human-level prompt engineers. In: Proceedings of the International Conference on Learning Representations

## Appendix

Appendix A gives the implementation details and prompt templates. Appendix B supplements the main evaluation with source-referenced fidelity metrics and additional nighttime comparisons. Appendix C examines fixed-prompt failures and hallucination propagation. The component, refinement-budget, and SBC ablations use the FMB test pairs; the remaining benchmark comparisons use the test-set sizes stated in Section 4.1.

## A Implementation and Prompt Details

## A.1 Region-level Evidence for SBC

Structured Belief Construction. In our implementation, SBC constructs the belief state in Eq. 7 through a hybrid region-level diagnosis. We first use SAM to obtain candidate regions and treat them only as local analysis units rather than fusion masks. For each region, we extract aligned crops from the visible image, infrared image, and the selected PGPR output. SBC then computes traditional visual cues to provide grounded evidence, including sharpness, contrast, brightness balance, cleanliness, edge density, thermal dynamic range, hotspot fraction, and hotspot compactness. These measurements are used to describe whether the visible modality provides reliable texture and appearance, whether the infrared modality provides compact thermal targets and clear boundaries, and whether the PGPR output preserves or distorts the source evidence. In parallel, Qwen3-VL-8B-Instruct parses the same region crops into structured semantic evidence, including presence, location, appearance, shape, and consistency cues, together with reliability, conflict-risk, and confidence estimates. The final belief combines measurable regional cues with semantic interpretation from Qwen3-VL-8B-Instruct.

After obtaining the region-level belief state, SBC maps it to the conservative prompt correction defined in Eq. 8. The belief state is summarized into image-level statistics, such as modality reliability, average confidence, average conflict, infrared-supported regions, highconfidence regions, and ambiguous regions. These statistics determine a conservative correction policy, including how strongly infrared evidence should be exposed, how strictly boundaries should be preserved, and whether the global appearance should follow the visible image, the PGPR prior, or a balanced rendering strategy. A small number of representative regions are further selected as local anchors, such as infrared-supported salient targets, stable structural boundaries, texture-supported visible regions, or ambiguous cross-modal regions. Finally, Qwen3-VL-8B-Instruct compiles these global policies and local anchor hints into a revised prompt, while treating the selected PGPR prompt as the authoritative base prompt. Thus, SBC only performs a scoped correction after PGPR: it may strengthen missing thermal targets, suppress thermal blobs, protect boundaries, or avoid visible degradation leakage, but SBC preserves the global fusion intent and applies a scoped correction to the selected prompt.

## A.2 Diagnostic and Update Rules

The diagnostic score combines semantic fidelity, artifact control, texture, information gain, and protocol compliance with weights 0.30, 0.25, 0.20, 0.15, and 0.10, respectively. Nine source-grounded checks return binary judgments and confidence estimates. Detected fatal source violations receive a score of −1 and are excluded from inheritance; a strict pass requires passing all nine checks and achieving a score of at least 0.80.

Starting from an evaluated non-fatal incumbent, the editor proposes one allowed slot edit: add an environment-state item, revise a principle, or add a principle. The style profile stays fixed. The validator rejects malformed, empty, repeated, or inefective edits, as well as edits with invalid conditions. A valid proposal is composed, generated, and diagnosed before selection. Lower-scoring or fatal candidates do not replace the incumbent; later edits start from a best-scoring valid state. The acceptance rule tracks the retained critic score, while source fidelity is evaluated separately.

## A.3 Protocol and Prompt Details

This section presents the fixed prompt baseline and the initial protocol used by ProtoFusion. The fixed human-crafted prompt is used in the failure examples in Fig. 12 and the controlled prompt/protocol comparison in Table 5. It is directly fed to the generative fusion model and kept unchanged for all image pairs. In contrast, ProtoFusion starts from an initial structured protocol, which specifies the environment variables, fusion principles, and style constraints for prompt composition. This protocol is subsequently optimized by PGPR via iterative evaluation-guided protocol refinement, and the optimized protocol is fixed for inference. Fixed Human-crafted Prompt. The fixed prompt is used to evaluate one-shot prompting without protocol refinement. The human-crafted prompt applies the same fusion instructions to every image pair, whereas PGPR combines image-specific prompt composition with an optimized fusion protocol.

You are doing image fusion.

Inputs:

\- Image 1: Color visible (VIS)

\- Image 2: Grayscale infrared (IR)

Goal:

Generate ONE fused image.

Rules:

1. Use the VIS image to define scene structure, layout, edges,

and background whenever possible.

2. If the VIS image is overexposed, underexposed, or low-contrast, use IR luminance to add texture, but render these details with a neutral visible appearance.

3. No thermal colors (red, orange, or yellow). No glow. No halo.

4. Do not let the IR image define global structure or background.

5. Keep VIS colors, textures, and fine details as the main appearance.

6. The result should look like a normal visible-light photograph.

7. No IR-style background, no double edges, no noise, and no ghosting.

Initial Protocol for PGPR. PGPR starts from the following structured protocol. In contrast to the fixed prompt, the protocol is not itself the final generation instruction. It first extracts an environment state from the input pair, applies condition-dependent fusion principles, defines the target visual style, and then assembles an operational prompt for the generator. During refinement, PGPR updates the protocol slots and principle descriptions according to multimodal evaluation feedback, so that later prompts are composed from an optimized protocol rather than from a fixed manually written instruction.

Role: You are the Fusion Logic Engine.

Input: Image 1 (Visible/Vis), Image 2 (Infrared/IR).

Goal: Analyze the images and generate a specific prompt for a Text/Image-to-Image

model (Qwen-Image) to create a fused image that preserves:

(1) Vis color/texture realism,

(2) clean, single boundaries (no

ghosting/bleed/smear), and

(3) information gain (saliency vs Vis, detail vs IR).

Phase 1: Perception Layer (Variable Extraction)

Observe visible and infrared images and fill in the [Environment State Card].

Do not describe; just extract values.

## [Environment State Card]

1. {Lighting\_Condition}: [Day / Dusk / Evening / Night / Artificial\_Light]

\- Condition: only rely on Vis image to judge.

2. {Vis\_Quality}: [High\_Fidelity / Overexposed / Low\_Light\_Noise]

3. {IR\_Targets}: [None / Pedestrians / Vehicles / Machinery]

4. {Information\_Dominance}: [Balanced / Vis\_Lead / IR\_Lead]

5. {Vis\_Edge\_Reliability}: [Reliable / Unreliable] - Reliable: edges are crisp and consistent (not dominated by noise/overexposure).

6. {Chroma\_Bleed\_Risk}: [Low / High] - High: strong colored light sources (neon, brake lights, traffic lights) likely to bleed.

7. {Alignment\_Risk}: [Low / High] - High: IR and Vis structures appear slightly shifted or mismatched around

object boundaries.

Phase 2: Decision Layer (Principle Application)

Based on Phase 1 variables, apply the following Fusion Principles.

Write the final prompt as a short, operational plan: clear steps + a compact list of constraints.

Principle A (Night Robustness and Noise Control):

IF {Lighting\_Condition} is Night AND {Vis\_Quality} is Low\_Light\_Noise:

\- Strategy:

1) Use Vis as the base for overall scene appearance (global tone and colors), but treat very dark/noisy Vis regions as low-trust for fine texture.

2) Preserve IR target shapes and boundaries; avoid turning targets into blobs.

3) Apply edge-preserving denoise (remove grain without smearing edges or textures).

\- Constraint:

\- No painterly smoothing or texture smear.

\- No synthetic grain; do not amplify noise.

\- Preserve crisp edges; do not blur boundaries.

Principle B (Thermal Saliency with Color Discipline): IF {IR\_Targets} exist:

## - Strategy:

1) Make IR targets easier to spot by increasing luminance/contrast of the

target regions (saliency), while keeping Vis realism.

2) Preserve target geometry (shape and boundary consistency) without double contours.

3) If {Chroma\_Bleed\_Risk} is High: lock chroma to Vis; inject IR only into

luminance/contrast (avoid colored thermal halos).

Constraint:

\- No hallucinated objects or structures absent in both IR and Vis.

\- No colored halos around targets; no color leakage across boundaries when {Chroma\_Bleed\_Risk} is High.

## Principle C (Texture Preservation without Smear):

IF {Vis\_Quality} is High\_Fidelity:

## - Strategy:

1) Preserve Vis micro-textures (text, lines, road texture) and natural color statistics.

2) Add IR saliency without washing out high-frequency Vis details.

3) Prefer sharp detail preservation over global smoothing.

\- Constraint:

\- No over-smoothing; no oil-painting look.

\- Do not wash out fine Vis details when adding IR saliency.

## Principle D (Boundary Alignment Policy):

IF {IR\_Targets} exist OR {Alignment\_Risk} is High: - Strategy:

1) Enforce a single contour per object (avoid double edges / semi-transparent outlines).

2) If {Vis\_Edge\_Reliability} is Reliable AND {Alignment\_Risk} is Low:

anchor object boundaries to Vis edges (hard anchor).

Else: use a soft alignment; prioritize one boundary source

3) Prevent offset blending of contours; keep boundaries geometrically consistent.

\- Constraint:

\- No ghost edges, double contours, or transparent overlays.

\- No offset blending that creates duplicated boundaries.

Phase 3: Stylization Layer (Visual Definition)

Define the artistic parameters:

\- View Mode: Realistic Night Vision / Enhanced Reality

\- Color Palette: Natural Vis colors

\- Clarity: 4K Resolution, Sharp Edges

## Phase 4: Construction Layer (Prompt Assembly)

Output the Final Prompt based on the previous layers.

[Final Prompt]: <Write the English prompt here>

## B Additional Experimental Results

Table 9 Additional source-referenced evaluation. Lower CE and higher $Q _ { A B / F }$ and VIF are better. Best and second-best results are bold and underlined, respectively.
<table><tr><td></td><td></td><td colspan="3">FMB</td><td colspan="3">MSRS</td><td colspan="3">RoadScene</td></tr><tr><td>Method</td><td>Venue</td><td>CE↓</td><td> $Q _ { A B / F } \uparrow$ </td><td>VIF↑</td><td>CE↓</td><td> $Q _ { A B / F } \uparrow$ </td><td>VIF↑</td><td>CE↓</td><td> $Q _ { A B / F } \uparrow$ </td><td>VIF↑</td></tr><tr><td>CDDFuse</td><td>CVPR&#x27;23</td><td>1.8371</td><td>0.3500</td><td>0.2150</td><td>1.0234</td><td>0.3721</td><td>0.3171</td><td>0.9237</td><td>0.2460</td><td>0.1810</td></tr><tr><td>DDFM</td><td>ICCV&#x27;23</td><td>1.8333</td><td>0.1131</td><td>0.0437</td><td>1.2518</td><td>0.2632</td><td>0.2482</td><td>1.4241</td><td>0.1282</td><td>0.0583</td></tr><tr><td>TIMFusion</td><td>TPAMI&#x27;24</td><td>2.1071</td><td>0.2698</td><td>0.1481</td><td>1.7597</td><td>0.2552</td><td>0.2019</td><td>0.6055</td><td>0.1978</td><td>0.1559</td></tr><tr><td>Text-IF</td><td>CVPR&#x27;24</td><td>1.7777</td><td>0.3621</td><td>0.2259</td><td>1.0391</td><td>0.3778</td><td>0.3269</td><td>0.8612</td><td>0.2759</td><td>0.1967</td></tr><tr><td>C2RF</td><td>IJCV’25</td><td>1.8749</td><td>0.1847</td><td>0.1158</td><td>1.2196</td><td>0.1587</td><td>0.1348</td><td>1.0004</td><td>0.2652</td><td>0.1693</td></tr><tr><td>DCEvo</td><td>CVPR&#x27;25</td><td>1.9175</td><td>0.3527</td><td>0.2153</td><td>1.0867</td><td>0.3769</td><td>0.3128</td><td>1.0545</td><td>0.2683</td><td>0.1792</td></tr><tr><td>Deno-IF</td><td>NeurIPS&#x27;25</td><td>2.1191</td><td>0.2758</td><td>0.1747</td><td>1.3868</td><td>0.3098</td><td>0.2848</td><td>0.9421</td><td>0.2484</td><td>0.1851</td></tr><tr><td>LUT-Fuse</td><td>ICCV’25</td><td>2.0738</td><td>0.2973</td><td>0.1886</td><td>1.2540</td><td>0.3380</td><td>0.2864</td><td>1.3454</td><td>0.2194</td><td>0.1610</td></tr><tr><td>APTP</td><td>CVPR&#x27;26</td><td>1.9288</td><td>0.3345</td><td>0.1993</td><td>1.0084</td><td>0.3437</td><td>0.2870</td><td>1.0902</td><td>0.2402</td><td>0.1867</td></tr><tr><td>DPOFusion</td><td>CVPR&#x27;26</td><td>2.1022</td><td>0.2371</td><td>0.1664</td><td>3.3510</td><td>0.1931</td><td>0.2612</td><td>0.9711</td><td>0.2065</td><td>0.1679</td></tr><tr><td>DRFusion</td><td>CVPR&#x27;26</td><td>1.8506</td><td>0.2595</td><td>0.1556</td><td>1.4457</td><td>0.2575</td><td>0.1974</td><td>0.4930</td><td>0.2062</td><td>0.1473</td></tr><tr><td>ISFuse</td><td>CVPR&#x27;26</td><td>2.3855</td><td>0.3102</td><td>0.1802</td><td>1.3161</td><td>0.3257</td><td>0.2618</td><td>1.7310</td><td>0.2368</td><td>0.1509</td></tr><tr><td>ProtoFusion</td><td></td><td>1.2906</td><td>0.3641</td><td>0.2253</td><td>1.1521</td><td>0.3943</td><td>0.3052</td><td>0.9346</td><td>0.3220</td><td>0.2908</td></tr></table>

![](images/4562459aba6d59e9d4f0ae81759f999aa4e436df21c2a522613d3af31ee9ead4.jpg)  
Fig. 11 Qualitative comparisons on degraded nighttime scenes.

## B.1 Traditional Fusion Metrics

Traditional fusion metrics mainly assess how much low-level information a fused image preserves from its infrared and visible sources. This evaluation only partly matches the objective of ProtoFusion. Our framework deliberately generates images with a natural visible-light appearance. It uses infrared information to guide the rendering of salient targets and structures, instead of directly reproducing infrared pixel intensities. Consequently, metrics that favor low-level agreement with both sources may not fully capture the information conveyed by the generated images. We report CE, $Q _ { A B / F }$ , and VIF on three benchmarks to evaluate source discrepancy, edge transfer, and visual information fidelity, respectively.

Despite this diference, ProtoFusion achieves the highest $Q _ { A B / F }$ on all three datasets, the lowest CE on FMB, and the highest VIF on RoadScene, as shown in Tab. 9. It also achieves the second-highest VIF on FMB. The consistent advantage in $Q _ { A B / F }$ is particularly notable, showing that our visible-light rendering can still preserve important edge

![](images/910e61daded44dfdcd8962d1feffcc65cb95c7a5a2693b75e9fff486c538b53c.jpg)  
information from both modalities. Although CE and VIF vary across datasets, these results demonstrate that ProtoFusion can retain substantial source information while pursuing a diferent visual representation. This highlights the potential of protocol-guided generative fusion beyond conventional pixel-level fusion.

## B.2 Additional Nighttime Comparisons

In Fig. 11, the nighttime scenes are mainly afected by low illumination, haze-like degradation, light glare, and reduced contrast. Although the infrared images provide clear responses for salient objects such as pedestrians and vehicles, directly introducing infrared information may lead to unnatural brightness, color shifts, or excessive thermal responses.

Existing methods often produce washed-out regions around light sources and road surfaces, or weaken structural details in dark and hazy areas. In contrast, ProtoFusion maintains a better balance between thermal saliency and visible appearance, preserving important targets while producing cleaner illumination, more stable colors, and clearer scene structures.

Fused Image  
Infrared  
![](images/3228569c05d94c8aeb1ce1120177c56723282bde55ff772e375070bfddda3867.jpg)  
Fig. 12 Failure cases of fixed prompting for infrared and visible image fusion. Although the same handcrafted prompt is used for all image pairs, the generated results show diverse failure modes, including structure collapse, modality domination, loss of visible scene texture, hallucinated objects, unrealistic illumination changes, and severe style drift. These failures show that one-shot fixed prompting cannot reliably constrain the large synthesis space of a generative fusion model, motivating protocol-guided prompt refinement.

## C Failure Case Analysis

## C.1 Limitations of Fixed Prompting in Generative Image Fusion

A simple way to use a multimodal generative model for infrared and visible image fusion is to provide a fixed, manually designed prompt that asks the model to preserve infrared thermal targets, retain visible textures, and produce a natural fused image. Although this strategy is intuitive, we find that it is not reliable in practice. As shown in Fig. 12, the same fixed prompt can produce substantially diferent failure modes across image pairs. In some cases, the generated image collapses into an oversmoothed or low-contrast result, where visible structures and infrared targets are both weakened. In other cases, the output is dominated by one modality; for example, it may resemble an infrared response map that lacks visible color and scene texture. More seriously, the generative model may change the scene itself, hallucinating unsupported objects, converting a rainy or nighttime scene into a bright daytime scene, or producing cartoon-like colorization and unnatural target rendering.

These examples show that the limitation of fixed prompting lies not only in weak control, but also in its inability to specify image-pair dependent fusion constraints. Infrared and visible image fusion must simultaneously preserve thermal targets, retain source-consistent visible structures, suppress degradations, and avoid hallucinated content or style drift. However, the dominant failure mode varies across image pairs: infrared evidence may be underused, thermal responses may be over-transferred, visible degradations may leak into the result, or the generative prior may rewrite the scene. Therefore, we treat the prompt as an optimizable interface rather than a fixed instruction. Through protocol-guided prompt refinement, the current result is diagnosed and the structured protocol is updated accordingly, enabling the next prompt to correct concrete failures such as modality imbalance, structure loss, thermal overpainting, and appearance drift while keeping the fusion intent explicit and interpretable.

## C.2 Hallucination in Image Fusion

ProtoFusion formulates infrared and visible image fusion as protocolconstrained generation, where the fused image is produced under protocol-guided prompt control by a multimodal image generator. This formulation provides strong semantic flexibility, but it also introduces a characteristic failure mode: hallucinated semantics may be propagated into the fusion result. In our current implementation, this risk is intrinsic to the generation pipeline, since the image generator already involves

![](images/cfe04f00f84b4ba59ff78f02e2f4932b33bcd030ebc83932de0134cfc1831b4a.jpg)

![](images/69d214c2fb962ad321b7239cb005d1102d204eeee4550a84da198a6b24339b5b.jpg)

![](images/4780ee4cf11376f40b025c11d7e3493ddbb95cb73e09415f0b2278361e80991c.jpg)  
Visible  
VLM interpretation: “Three people walking together on a wet city street at dusk, one holding a white umbrella, another carrying a black umbrella. . . ”

Fig. 13 Qualitative example of hallucination propagation in generative infrared and visible image fusion. A hallucinated “white umbrella” in the interpretation of the visible image is propagated into the fused result, producing a source-inconsistent semantic artifact.

an internal visual understanding stage. As a result, once the internal understanding stage produces an incorrect semantic interpretation, the downstream generator may render this error as if it were valid scene evidence.

Figure 13 shows a representative failure case. The interpretation of the visible image incorrectly introduces a non-existent “white umbrella,” which is then materialized in the fused image as a semantic artifact. This artifact is not supported by the infrared or visible inputs; instead, it arises because the generative fusion process follows a semantically plausible but source-inconsistent cue. The example reveals a key limitation of generative fusion: semantic guidance can improve controllability and interpretability, but erroneous visual understanding may also inject false content into the fused result.

This failure case is consistent with the limitation discussed in the main paper, namely that pretrained multimodal generative models may produce hallucinated structures or source-inconsistent details in dificult cases. For infrared and visible image fusion, this issue is particularly important because the objective is not only visual coherence, but also faithfulness to cross-modal evidence. It therefore motivates stronger source-grounded semantic verification and cross-modal consistency control in future generative fusion systems.