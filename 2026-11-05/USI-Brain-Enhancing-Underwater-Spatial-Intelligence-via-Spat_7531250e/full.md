# USI-Brain: Enhancing Underwater Spatial Intelligence via Spatial Data-Driven Vision-Language Modeling

Zhiliang Zhu<sup>1,\*,†</sup>, Zhiqiang Zhao<sup>1,\*</sup>, Xie’an Jiang<sup>1,\*</sup>,

Yingxin Lai<sup>2</sup>, Kangye Zheng<sup>1</sup>, Dawei Zhao<sup>1,‡</sup>

<sup>1</sup>Harbin Engineering University <sup>2</sup>Xiamen University zhuzhiliang@hrbeu.edu.cn

## Abstract

Underwater spatial reasoning depends on object boundaries and geometric cues that are obscured by scattering, color distortion, and low visibility. Existing underwater resources capture these cues through images, annotations, and depth maps, but provide fragmented supervision for vision-language models. We present a spatial data engine that connects visual appearance with object-level geometry through enhancement, 2D and geometry-assisted question generation, and expert filtering. It produces 4.27M QA pairs across five capability groups. We use this corpus to train USI-Brain and introduce an Underwater Prior-guided Rectifier (UPR) that applies gated corrections to visual features. USI-Bench evaluates the resulting capabilities with 6,525 held-out QA pairs. On its 15 task metrics, USI-Brain and USI-Brain + UPR improve over Qwen3-VL-8B-Instruct by 5.76 and 7.63 points on average, respectively, using signed differences on 100-point scales. Ablations show that geometry-assisted supervision improves depth and ordering by 4.8 and 6.1 points with UPR disabled, while UPR adds 18.0 points in scene description under matched training data. Both variants use raw images at evaluation. Across NautData, UW-Bench, and MarineEval, USI-Brain achieves the best or jointbest score on 28 of 32 metrics. Model weights and USI-Bench will be released at https://github.com/SKY-Zhu-91/USI-Brain.

## 1 Introduction

Marine ecological monitoring, offshore inspection, and autonomous ocean operations require models to locate objects, compare their depths, and interpret the surrounding scene (Yuan et al. 2023; Nauert and Kampmann 2023; Wynn et al. 2014; Huy et al. 2023; Zhang et al. 2023). Identifying a structure is one part of this process; determining its position relative to nearby organisms, its visibility, and its spatial extent requires evidence about both appearance and geometry. We study these capabilities as underwater spatial intelligence.

Underwater imaging degrades the cues needed for such reasoning. Wavelength-dependent attenuation alters color, while backscatter, blur, and low illumination obscure boundaries and texture (Akkaynak and Treibitz 2019; Li et al. 2020; Gu et al. 2023). Image enhancement addresses these appearance changes. Questions about depth, ordering, and occlusion additionally require geometric relationships to be represented in the supervision (Cai and Metzler 2025). This motivates a data-construction process that combines enhancement with object-linked geometry.

The required evidence is distributed across existing resources. NAUTILUS (Xu et al. 2025b) provides large-scale underwater multimodal data; MarineInst (Zheng et al. 2024a), CoralSCOP (Zheng et al. 2024b), and SynTIDE (Lin et al. 2025) contribute instance annotations, segmentation, and depth-related supervision. UWBench (Zhang et al. 2025), CoralVQA (Han et al. 2025), UQ-Bench (Cao et al. 2026), and MarineEval (Wong et al. 2026) evaluate different aspects of underwater understanding. Tables 1 and 2 show the resulting differences in capability coverage. A common QA representation can bring these resources together while retaining the correspondence between an object, its image region, and its geometric cues.

Structured spatial supervision has improved general visionlanguage models in SpatialVLM (Chen et al. 2024a), Visual Spatial Tuning (Yang et al. 2025b), and OpenSpatial (Liu et al. 2026b). We apply this principle to underwater data through a spatial data engine. Hybrid Underwater Enhancement (HUE) prepares images for annotation. A 2D branch generates questions about appearance and image-plane relations; a geometry-assisted branch connects image regions with depth samples and reconstructed object boxes to generate depth, ordering, occlusion, and stratification questions. Capability-based organization and expert review yield 4.27M QA pairs spanning five groups.

We train USI-Brain from Qwen3-VL-8B-Instruct (Bai et al. 2025b) on this corpus. To adapt visual representations to degraded inputs, we introduce an Underwater Prior-guided Rectifier (UPR) that predicts additive and multiplicative feature corrections and combines them through a residual gate. USI-Brain + UPR retains both raw and corrected visual tokens. HUE shapes the training supervision, while UPR operates within the model; both variants are evaluated on raw images.

We evaluate the resulting capabilities on USI-Bench, a held-out set of 6,525 QA pairs, and three external benchmarks. GeoQA improves depth and ordering, HUE improves localization and structural perception, and UPR produces its largest incremental gain in scene description. Table 3 provides task-level component ablations, and Appendix B examines how training-group coverage changes the five capability scores. Object and ecological understanding provide domain context alongside the explicit 2D and 3D reasoning tasks.

![](images/74fb8468e812a34d1ee9196ad52a0a709a1731c0f8816549e8dfa923c0890c69.jpg)  
Figure 1: Overview of the underwater spatial data engine. The engine enhances heterogeneous inputs, generates 2D and geometryassisted 3D QA, organizes them into five capability groups, and applies expert filtering.

The contributions are:

• A spatial data engine that connects enhanced underwater imagery, instance annotations, and geometric cues to construct 4.27M QA pairs across five capability groups.

• USI-Brain, an underwater vision-language model trained on this corpus, and UPR, a prior-guided feature rectifier evaluated through matched-data training and component ablations.

• USI-Bench, a 6,525-QA benchmark that measures spatial reasoning and underwater semantics at the task level, complemented by evaluations on NautData, UWBench, and MarineEval.

## 2 Related Work

Underwater Enhancement and 3D Supervision. Underwater enhancement methods address the appearance changes introduced by absorption, scattering, and illumination. Sea-Thru (Akkaynak and Treibitz 2019) models underwater image formation, and UIEB (Li et al. 2020) provides an enhancement benchmark. Surveys describe the implications of these degradations for underwater perception and computational imaging (Huy et al. 2023; Gu et al. 2023). Underwater depth estimation supplies geometric cues for object placement and scene structure (Cai and Metzler 2025). We bring these two sources of information into QA construction: enhancement prepares the visual evidence, and depth or reconstruction links it to object-level spatial relations.

Spatial Reasoning Benchmarks and Data Engines. Spatial benchmarks measure object relationships, layouts, and geometric reasoning. Representative datasets include VSR (Liu, Emerson, and Collier 2023), SpaCE-10 (Gong et al. 2025), SpatialBench (Xu et al. 2025a), OmniSpatial (Jia et al. 2025), MM-Spatial (Daxberger et al. 2025), InternSpatial (Deng et al. 2025), MMSI-Bench (Yang et al. 2025c), and TangramPuzzle (Liu et al. 2026a). Related analyses study spatial generalization and embodied reasoning (Zhang et al. 2026; Yang et al. 2025a). SpatialVLM (Chen et al. 2024a), Visual Spatial Tuning (Yang et al. 2025b), SpatialLadder (Li et al. 2025), OpenSpatial (Liu et al. 2026b), and Scaling Spatial Intelligence with Multimodal Foundation Models (Cai et al. 2025) investigate training with structured spatial data. Earth-SpatialBench (Xu et al. 2026) extends spatial evaluation to a specific visual domain. Our engine uses object-linked spatial supervision in underwater scenes, combining the available geometric evidence with enhancement during question generation.

Underwater Multimodal Models and Resources. NAU-TILUS (Xu et al. 2025b) combines underwater data construction with multimodal model training. OceanGPT (Bi et al. 2023), OceanGym (Xue et al. 2025), and USIM/U0 (Gu et al. 2025) address ocean-domain modeling, evaluation, or embodied interaction. General-purpose backbones such as Qwen3-VL (Bai et al. 2025b) provide a starting point for domain adaptation. USI-Brain builds on this resource-oriented approach by organizing enhancement, instance annotations, depth, and QA around a common set of underwater spatial capabilities.

## 3 Methodology

Our method couples supervision construction with visualfeature adaptation. The data engine in Figure 1 collects sources, enhances images, generates spatial QA, organizes capabilities, and filters annotations through expert review. Its 2D and geometry-assisted branches produce a shared training corpus. UPR then conditions visual-feature correction on underwater priors within the adapted model.

## 3.1 Spatial Data Engine

The engine accepts labeled underwater images, image-depth pairs, and frames extracted from underwater videos. Both QA branches use enhanced images. The 2D branch generates questions from image content and available annotations; the geometry branch combines images with object-aligned depth or reconstruction cues. Candidate questions are assigned to the five capability groups and passed to expert review for selection and refinement.

Table 1: Comparison of training resources for underwater spatial intelligence. OG&U denotes object grounding and understanding; the remaining groups cover 2D relations and layout, 3D depth and stratification, degradation-aware perception, and scene and ecological reasoning. Reported sizes use the counting units of the respective resources; the USI entry counts generated QA pairs.
<table><tr><td>Training resource</td><td>Ref.</td><td>OG&amp;U</td><td>2D Rel. &amp;Layout</td><td>3D Depth &amp;Strat.</td><td>Deg. &amp;Perc.</td><td>Scene &amp;Eco.</td><td>Reported size</td><td>Open src.</td></tr><tr><td>MarineInst (Zheng et al. 2024a)</td><td>ECCV 2024</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>2.20M</td><td>Part. Avail.</td></tr><tr><td>CoralSCOP (Zheng et al. 2024b)</td><td>CVPR 2024</td><td>√</td><td>X</td><td>X</td><td>X</td><td>2</td><td>46.60K</td><td>Avail.</td></tr><tr><td>SynTIDE (Lin et al. 2025)</td><td>CVPR 2025</td><td>2</td><td>X</td><td>√</td><td>X</td><td>×</td><td>54.51K</td><td>Avail.</td></tr><tr><td>NautData (Xu et al. 2025b)</td><td>NeurIPS 2025</td><td>V</td><td>2</td><td>X</td><td>2</td><td>2</td><td>1.45M</td><td>Avail.</td></tr><tr><td>USI (ours)</td><td></td><td>V</td><td>V</td><td>V</td><td>V</td><td>V</td><td>4.27M</td><td>Planned</td></tr></table>

Notation. ✓: explicit support; ∼: partial or indirect support; ×: little or no direct support.

Table 2: Coverage of public evaluation resources across five underwater spatial-intelligence capability groups.
<table><tr><td rowspan="2">Evaluation resource</td><td rowspan="2">Ref.</td><td colspan="3">OG&amp;U</td><td colspan="2"></td><td colspan="2">2D Rel.&amp;Layout 3D Depth&amp;Strat.</td><td colspan="2"></td><td colspan="2">Deg.&amp;Perc. Scene&amp;Eco.</td></tr><tr><td></td><td>Obj. Attr. Grd./Cnt./Inst. Rel.</td><td></td><td></td><td>Layout</td><td>Depth Occ./View</td><td></td><td></td><td></td><td>Env. Struct. Scene</td><td>Eco.</td></tr><tr><td>CoralVQA (Han et al. 2025)</td><td>arXiv 2025</td><td>√</td><td>√</td><td>√</td><td>~</td><td>~</td><td>X</td><td>X</td><td>X</td><td>X</td><td>2</td><td>√</td></tr><tr><td>UWBench (Zhang et al. 2025)</td><td>arXiv 2025</td><td>√</td><td>√</td><td>√</td><td>2</td><td>2</td><td>X</td><td>X</td><td>2</td><td>2</td><td>2</td><td>2</td></tr><tr><td>MarineEval (Wong et al. 2026)</td><td>WACV 2026</td><td>√</td><td>√</td><td>√</td><td>2</td><td>2</td><td>2</td><td>2</td><td>X</td><td>×</td><td>2</td><td>√</td></tr><tr><td>CoralSCOP (test split) (Zheng et al. 2024b) CVPR 2024</td><td></td><td>√</td><td>√</td><td>2</td><td>×</td><td>X</td><td>×</td><td>×</td><td>X</td><td>×</td><td>X</td><td>2</td></tr><tr><td>NautData (test split) (Xu et al. 2025b)</td><td>NeurIPS 2025</td><td>√</td><td>√</td><td>V</td><td>2</td><td>2</td><td>×</td><td>×</td><td>2</td><td>2</td><td>2</td><td>2</td></tr><tr><td>USI-Bench (ours)</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>V</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Notation. ✓: explicit coverage; ∼: partial coverage; ×: little or no targeted coverage.

Hybrid Underwater Enhancement. HUE prepares images for QA generation using two enhancement paths. The design is motivated by underwater image formation and restoration (Akkaynak and Treibitz 2019; Li et al. 2020; Gu et al. 2023). The color-correction path predicts per-channel gains and biases from learned degradation maps and applies spatial refinement to adjust color and illumination. The local path uses a degradation-guided convolutional encoder–decoder with difference-aware enhancement to recover edges and textures.

Given an image $I \in \mathbb { R } ^ { H \times W \times 3 }$ , an estimator predicts maps for color shift, scattering, and blur:

$$
M _ { \mathrm { d e g } } = \mathrm { C o n c a t } ( M _ { \mathrm { c o l o r } } , M _ { \mathrm { s c a t t e r } } , M _ { \mathrm { b l u r } } ) .\tag{1}
$$

The learned maps condition both paths, producing a colorcorrected image $I _ { \mathrm { c c } }$ and a detail-restored image $I _ { \mathrm { d e t a i l } } .$ . A fusion module takes I and $M _ { \mathrm { d e g } }$ and predicts a pixel-wise weight map $\alpha \in [ 0 , 1 ] ^ { H \times W }$

$$
I _ { \mathrm { h u e } } = \alpha \odot I _ { \mathrm { c c } } + ( 1 - \alpha ) \odot I _ { \mathrm { d e t a i l } } .\tag{2}
$$

Here ⊙ denotes element-wise multiplication, with the spatial weights broadcast over image channels. The pixel-wise weights adapt the balance between color correction and detail restoration across the image.

2D Spatial QA Generation. The 2D branch uses enhanced images and available annotations to generate questions about object identity and attributes, image-plane locations, and relationships between objects. It also provides questions about visibility and scene content. Each question is paired with its reference answer in the shared QA format used by both branches.

3D Geometry Lifting and Reconstruction-Assisted Supervision. The geometry branch combines an enhanced RGB image with its corresponding depth map. Following reconstruction-assisted spatial supervision (Liu et al. 2026b), we reconstruct Atlantis scenes (Zhang et al. 2024b) and fit object-aligned 3D boxes. These boxes connect image regions to geometric measurements used for question generation.

For an instance mask or 2D bounding box, we extract the associated depth samples. A pixel $\boldsymbol { u } _ { i } ^ { \setminus } = ( x _ { i } , y _ { i } , 1 ) ^ { \top }$ with depth $z _ { i }$ is back-projected into camera coordinates:

$$
p _ { i } = z _ { i } K ^ { - 1 } u _ { i } ,\tag{3}
$$

where $\ b { K } \in \mathbb { R } ^ { 3 \times 3 }$ is the camera intrinsic matrix. We fit a 3D box to the resulting object point set $\mathcal { P } _ { o } = \{ p _ { i } \} _ { i = 1 } ^ { n _ { o } }$

$$
\mathcal { B } _ { o } = \mathrm { F i t B o x } ( \mathcal { P } _ { o } ) .\tag{4}
$$

The fitted boxes and reconstructed scenes provide cues for relative depth, foreground-background organization, occlusion, object-to-object distance, and viewpoint. Renderings with aligned box cues are supplied to the QA-generation model together with enhanced images. Depth maps, 3D boxes, and reconstructed scenes supply offline supervision; evaluation requires images and questions.

Capability Organization and Expert Review. We organize candidate QA pairs into object grounding and understanding, 2D relations and layout, 3D depth and stratification, degradation-aware perception, and scene and ecological reasoning. Review checks visual evidence for the answer, consistency between questions, answers, and localization annotations, and redundancy across tasks. It selects and refines

## (a) USI-Brain

![](images/78caa6b30f843f0cd0ee82ee0a8454a9bbfe5d57a4f158c1a3ff1daaa736fb6d.jpg)  
Figure 2: Overview of UPR, which estimates scattering-bias and absorption-recovery signals and rectifies visual tokens through gated residual fusion.

## 3.2 Feature-space Rectification with UPR

USI-Brain + UPR uses UPR to adjust visual features extracted from degraded inputs; the USI-Brain variant omits this module. The module is placed after the vision encoder and before multimodal projection, while retaining the backbone ViT, merger, and language-model architectures. Figure 2 shows its prior extraction, feature correction, and residual fusion steps.

Let $F \overset { \cdot } { = } \mathcal { E } _ { v } ( I ) \in \mathbb { R } ^ { N \times C }$ denote visual tokens from the Qwen3-VL vision encoder, and let $P \in \mathbb { R } ^ { N \times C }$ denote the aligned prior features derived from $\mathcal { E } _ { p } ( I )$ . The prior encoder $\mathcal { E } _ { p }$ is DINOv2 ViT-L/14. Motivated by the additive and multiplicative effects of underwater degradation (Gu et al. 2023; Huy et al. 2023), UPR predicts a scattering-related bias $\beta$ and an absorption-recovery gain G:

$$
\begin{array} { l } { { \beta = \phi _ { s } ( d - g ) , } } \\ { { G = \exp \bigl ( \mathrm { c l i p } ( \phi _ { a } ( P ) , - 4 , 4 ) \bigr ) . } } \end{array}\tag{5}
$$

Here d is the token selected from the darkest image region, g is a learned global context feature, and $\phi _ { s }$ and $\phi _ { a }$ are learned predictors. These terms parameterize corrections in feature space. The intermediate representation is

$$
{ \tilde { F } } = G \odot ( F - \beta ) .\tag{6}
$$

A gate controls how much of this correction is applied:

$$
\begin{array} { l } { { { \cal M } = \sigma \big ( \phi _ { g } ( [ { \cal F } ; { \tilde { \cal F } } ] ) \big ) , } } \\ { { { \hat { \cal F } } = { \cal F } + { \cal M } \odot ( { \tilde { \cal F } } - { \cal F } ) . } } \end{array}\tag{7}
$$

The operator $[ F ; { \tilde { F } } ]$ denotes channel-wise concatenation, σ is the sigmoid function, and $\phi _ { g }$ predicts the gate. The original and corrected features pass through the same merger. Let

$R , \hat { R } \in \mathbb R ^ { L \times D }$ be the two post-merger sequences. Their tokens are interleaved along the sequence dimension:

$$
Z = { \mathrm { I n t e r l e a v e } } ( R , { \hat { R } } ) \in \mathbb R ^ { 2 L \times D } .\tag{8}
$$

UPR preserves the channel width D and doubles the postmerger visual sequence. In the fixed 512 × 512 configuration, 256 baseline visual tokens become 512. The gate weights the feature update element-wise, while interleaving presents the raw and corrected tokens as separate positions to the language model. UPR’s inference computation includes the prior encoder and this longer visual sequence.

## 3.3 USI-Brain Adaptation and Fine-tuning

USI-Brain is initialized from Qwen3-VL-8B-Instruct (Bai et al. 2025b). Training combines the 2D and geometryassisted QA pairs in a single corpus, following the use of structured spatial supervision in prior work (Chen et al. 2024a; Yang et al. 2025b; Liu et al. 2026b). We use scaled dot-product attention (SDPA), unfreeze the ViT, language model, and merger, and train on 4.27M QA pairs for two epochs. Optimization settings are given in Section 5.1.

We train USI-Brain without UPR and USI-Brain + UPR as separate checkpoints using the same corpus and optimization settings. The learning objective is cross-entropy over the QA answer tokens:

$$
\mathcal { L } _ { \mathrm { Q A } } = - \sum _ { i = 1 } ^ { | a | } \log p _ { \theta } ( a _ { i } \mid I , q , a _ { < i } ) ,\tag{9}
$$

where q and a denote the question and reference answer. In USI-Brain + UPR, the rectifier is optimized jointly with the backbone through this objective. Both variants optimize the QA objective without auxiliary distillation. Closed models generate and check QA offline. Component removals and restricted training-group configurations are defined in Appendix B.

## 4 Dataset and Benchmark

The data engine constructs the training corpus, and USI-Bench is a held-out evaluation set. Splits are defined by original image or video: frames and processed derivatives remain in the same source split, together with their associated QA. Test-source images from USI-Bench, NautData, UWBench, and MarineEval are excluded from training-QA construction. Source IDs and QA links are checked, with SHA-256 and perceptual-hash screening for duplicate images. NautData is evaluated on its official within-dataset split; its 3,920 test images are excluded from training.

## 4.1 Underwater Spatial Intelligence Dataset

We use public underwater resources as visual sources and regenerate spatial QA supervision through the data engine. The source pool contains approximately 540K visual samples from still images and video frames, drawn from the ten datasets aggregated in NautData, together with CoralVQA, WebUOT-1M (Zhang et al. 2024a), and our reconstructions of approximately 400 Atlantis scenes. These sources cover different marine environments, objects, habitats, visibility conditions, and annotation formats. Table 1 compares their capability coverage.

The 2D branch generates QA pairs from enhanced images and annotations. The geometry branch adds questions that use depth and reconstructed scene structure, including depth ordering, stratification, occlusion, distance, and viewpoint. Capability-based organization and expert filtering produce a final corpus of 4.27M QA pairs.

## 4.2 USI-Bench Composition and Evaluation

USI-Bench contains 6,525 held-out QA pairs drawn from public underwater resources and newly curated annotations. Training and benchmark QA share a generation–filtering– review workflow but use different prompts and templates. Fifteen independent annotators audited all 6,525 benchmark QA pairs. In the pre-revision audit, approximately 83% were judged correct, 10% uncertain, and 7% incorrect. All uncertain and incorrect entries were subsequently revised. Figure 3 shows the composition and examples. The benchmark covers five groups:

• Object Grounding and Understanding: object identity, attributes, counting, actions, grounding, instance localization, and region descriptions.

• 2D Relations and Layout: relative position, pairwise relationships, local arrangement, and coarse distance.

• 3D Depth and Stratification: relative depth, ordering, foreground-background separation, occlusion, and viewpoint.

• Degradation-aware Perception: visibility and structural cues affected by color cast, blur, low illumination, and turbidity.

• Scene and Ecological Reasoning: scene descriptions, habitat cues, ecological relationships, health status, and the functions of underwater structures.

Tasks use class labels, multiple-choice answers, ranked outputs, or bounding boxes where direct scoring is possible (Liu, Emerson, and Collier 2023; Gong et al. 2025; Liu et al. 2026b). Table 3 reports accuracy for classification, Acc@IoU0.5 for grounding, Recall@IoU0.5 for detection, Kendall’s τ for ordering, and macro-F1 for foregroundbackground separation and structural perception. Grounding outputs use integer xywh coordinates normalized to 0–1000. Metrics are displayed on a 100-point scale, including 100τ for ordering.

For foreground-background separation, option letters are extracted, deduplicated, and sorted; each complete answer combination is one class. Macro-F1 is the equal-weight mean of classwise F1 over the union of reference and predicted classes, multiplied by 100. The fixed-answer baseline that always predicts A scores 3.43 under this protocol.

Scene and region captioning use three independent judges: Gemini 3.1 Pro, GPT-5.5, and Claude Opus 4.7. Each receives the same question, reference answer, and model response, and assigns a score from 0 to 5 based on semantic agreement, with penalties for omissions and unsupported claims (Zheng et al. 2023). Scene scoring considers the overall scene, key entities, actions, and spatial relationships; caption scoring considers the described region’s entities, attributes, counts, actions, and relationships. We average over the three judges and all items within each task, then multiply by 20.

## 5 Experiments

## 5.1 Experimental Setup

Comparison Models. USI-Bench comparisons include proprietary models, open-source general VLMs, spatialunderstanding models, and underwater-specialized models. The general-purpose models include Qwen3-VL (Bai et al. 2025b), GPT-5.4, Gemini (Gemini Team, Google 2025; Google DeepMind 2026), and Claude. The spatial models are SenseNova-SI (Cai et al. 2025) and VST (Yang et al. 2025b); the underwater models are OceanGPT (Bi et al. 2023) and the Qwen2.5-VL- and LLaVA-1.5-based NAUTILUS variants (Xu et al. 2025b). Additional Qwen2.5-VL (Bai et al. 2025a), InternVL (Chen et al. 2024b), and GLM-V (Hong et al. 2025) comparisons are included in the external results.

Evaluation. For USI-Bench, models are evaluated through a common prompting, decoding, and answer-normalization interface. Predictions are parsed into task-specific outputs before scoring. The external benchmarks retain the comparison protocols described in Appendix A, including published baseline scores on UWBench and a separate three-judge evaluation on MarineEval.

Training. We fine-tune Qwen3-VL-8B-Instruct on the 4.27M-QA corpus for two epochs. The language and vision components use learning rates of $5 \times 1 0 ^ { - 5 }$ and $5 \times 1 0 ^ { - 6 }$ respectively, with a warmup ratio of 0.1 and cosine decay. Training uses mixed precision, gradient checkpointing, and FSDP full sharding. The micro-batch size is 2 and the global batch size is 128 on 64 NVIDIA B200 GPUs. The full training run takes approximately 25 hours. Frequency-domain gating in HUE is disabled in all reported experiments.

![](images/0209fb20d19e512724ba7c294368a19db024079f9d993cb2b8ec5aacbf6ebaa0.jpg)  
Figure 3: Composition and representative QA examples of USI-Bench across five capability groups.

Model Variants and Ablations. USI-Brain omits UPR; USI-Brain + UPR includes it. They are trained separately on the same 4.27M QA pairs with the same training settings. For USI-Bench evaluation, both use raw images with identical processor settings and dynamic resizing; HUE is not applied at inference. In Table 3, w/o HUE removes enhancementderived supervision while retaining GeoQA, and w/o GeoQA removes the geometry-assisted QA branch while retaining HUE. Both removals keep UPR disabled and are compared with USI-Brain. Appendix B specifies their correspondence to the training-group configurations and reports the extended data ablations.

Single-GPU inference latency, decoding speed, and peak memory on A800 and RTX 3090 are reported in Appendix C.

## 5.2 Main Results on USI-Bench

Table 3 reports 15 task metrics across five capability groups, including the two component ablations. USI-Brain improves over Qwen3-VL-8B-Instruct on 12 of 15 metrics. The largest gains are in captioning (+17.0 points), structure (+15.7), and ordering (+10.1 on the 100τ scale). Grounding rises from 85.4 to 90.8, and detection recall from 47.9 to 55.5. This UPR-free model captures the gains from fine-tuning on the constructed corpus.

USI-Brain + UPR improves 13 of 15 metrics over the backbone and achieves the best or joint-best result among the listed models on seven: identity, grounding, detection, depth, viewpoint, ordering, and foreground-background separation. It exceeds the non-USI open-source comparison models on 11 metrics. Its four depth and stratification scores are 83.8, 84.8, 83.2, and 22.7, respectively.

Averaging the signed score differences across all 15 tasks gives gains of 5.76 points for USI-Brain and 7.63 for USI-Brain + UPR. Each task receives equal weight on its reported 100-point scale, including negative differences. Appendix B also reports a capability-level summary that gives each of the five groups equal weight, yielding gains of 5.04 and 7.44 points, respectively.

## 5.3 Component and Training-Data Analysis

Geometry-assisted supervision. With UPR disabled, retaining GeoQA raises Depth, Viewpoint, Ordering, and Fg./Bg. by 4.8, 3.2, 6.1, and 2.2 points, respectively. The largest gains occur in depth and ordering, the tasks directly supported by object-level geometry. Relation and Distance also improve by 3.8 and 2.9 points. This comparison varies the inclusion of geometry-assisted QA while retaining HUE, revealing benefits across both 3D and image-plane relationships.

Enhancement-derived supervision. With GeoQA retained and UPR disabled, HUE improves all 15 task scores. Grounding, Depth, Ordering, Structure, and Scene rise by 2.6, 1.9, 1.7, 3.3, and 2.2 points, respectively. Enhancement during supervision construction thus improves localization and structural judgments on raw evaluation images.

Feature rectification. Training with UPR on the same QA corpus adds 1.3 points in Grounding, 1.1 in Depth, and 0.9 in Ordering, with Viewpoint unchanged. Its largest effect is on Scene, which rises from 47.3 to 65.3 (+18.0). Captioning and Ecology each increase by 0.3. The gain is concentrated in global scene description, with smaller changes in region description and ecological classification. Both models use HUE-derived supervision.

Training-data coverage. Table 9 in Appendix B examines restricted training groups. Adding B3 to B1 training raises the B3 gain over Base from +0.50 to +4.60; adding B4 to B1+B2+B3 raises the B4 gain from +0.20 to +3.00. In these two comparisons, the largest change is in the newly included capability group. The four configurations shared with Table 3 connect these group-level patterns to the task-level component results.

Task-level variation. Distance and Environment remain below the backbone at 69.6 versus 69.9 and 78.3 versus 81.2, respectively. Fg./Bg. reaches 22.7, compared with 18.5 for the backbone and 3.43 for the fixed-A reference. Foreground-background assignment, distance estimation, and environment recognition thus remain priorities for expanding supervision and evaluation. Detection is measured by Recall@IoU0.5; precision would complement this measure by characterizing false positives.

## 5.4 External Benchmark Evaluation

We evaluate USI-Brain on NautData (Xu et al. 2025b), UW-Bench (Zhang et al. 2025), and MarineEval (Wong et al.

Table 3: Fine-grained results on USI-Bench. Metrics are reported on a 100-point scale, with Kendall’s τ multiplied by 100. Parentheses show score differences from Qwen3-VL-8B-Instruct. Boldface marks the best or joint-best result among the listed models. Within Ours, darker/lighter blue marks the best/second-best distinct scores; ties share a color. USI-Brain uses HUEderived and geometry-assisted QA (GeoQA) supervision without UPR; USI-Brain + UPR is trained with UPR. Both removal ablations disable UPR and are compared with USI-Brain. All variants use raw evaluation images. Positive/negative differences are red/green.
<table><tr><td rowspan="3">Model</td><td colspan="5">Object Grounding &amp; Understanding</td><td colspan="2">2D Relations &amp; Layout</td></tr><tr><td>Identity</td><td>Grounding</td><td>Counting</td><td>Detection</td><td>Captioning</td><td>Relation</td><td>Distance</td></tr><tr><td>Acc</td><td>Acc@IoU0.5</td><td>Acc</td><td>Recall@IoU0.5</td><td>Judge</td><td>Acc</td><td>Acc</td></tr><tr><td colspan="8">Proprietary Models</td></tr><tr><td>GPT-5.4 Gemini</td><td>50.5 50.1</td><td>49.6 8.7</td><td>46.3 45.0</td><td>8.9 0.1</td><td>53.2 45.4</td><td>60.5 73.0</td><td>75.2 77.0</td></tr><tr><td>Claude-Opus-4.6</td><td>60.3</td><td>28.9</td><td>42.9</td><td>7.0</td><td>57.7</td><td>40.0</td><td>57.5</td></tr><tr><td colspan="8">Open-source General VLMs Qwen3-VL-30B-A3B-Instruct</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>67.6 64.7</td><td>86.4 85.4</td><td>46.5 42.9</td><td>49.4 47.9</td><td>31.3 38.9</td><td>68.5 65.8</td><td>72.6 69.9</td></tr><tr><td colspan="8">Spatial-understanding Models</td></tr><tr><td>SenseNova-SI-1.1-Qwen3-VL-8B VST-7B-SFT</td><td>59.5 54.4</td><td>82.8 76.2</td><td>39.6 47.5</td><td>32.8 35.1</td><td>34.9 5.3</td><td>60.6 44.5</td><td>60.2 56.3</td></tr><tr><td colspan="8">Underwater-specialized Models OceanGPT</td></tr><tr><td>NAUTILUS-Qwen2.5-VL</td><td>57.9 63.2</td><td>1.4 0.2</td><td>46.5 35.5</td><td>0.6 0.3</td><td>38.8 43.3</td><td>42.8 57.8</td><td>45.4 60.8</td></tr><tr><td>NAUTILUS-LLaVA-1.5</td><td>52.6</td><td>0.6</td><td>38.7</td><td>0.1</td><td>41.5</td><td>53.5</td><td>55.2</td></tr><tr><td colspan="8">Ours USI-Brain + UPR</td></tr><tr><td>USI-Brain USI-Brain w/o HUE USI-Brain w/o GeoQA</td><td>73.1 (+8.4) 70.6 (+5.9) 73.2 (+8.5)</td><td>90.8 (+5.4) 88.2 (+2.8) 89.6 (+4.2)</td><td>42.4 (-0.5) 41.0 (-1.9) 43.8 (+0.9)</td><td>55.5 (+7.6) 51.9 (+4.0) 54.2 (+6.3)</td><td>55.9 (+17.0) 52.5 (+13.6) 54.9 (+16.0)</td><td>72.8 (+7.0) 71.4 (+5.6) 70.5 (+4.7) 67.6 (+1.8)</td><td>69.6 (-0.3) 69.0 (-0.9) 68.7 (-1.2) 66.1 (-3.8)</td></tr><tr><td colspan="4">Model</td><td colspan="4">Degradation &amp; Perception</td></tr><tr><td></td><td>Depth Acc Acc</td><td>Viewpoint</td><td>Ordering 100τ</td><td>Fg./Bg. Environment macro-F1 Acc</td><td>Structure macro-F1</td><td>Scene Judge</td><td>Ecology Acc</td></tr><tr><td colspan="8">Proprietary Models</td></tr><tr><td>Gemini Claude-Opus-4.6 Open-source General VLMs</td><td>61.7 24.0</td><td>77.2 71.9</td><td>80.2 79.5</td><td>19.2 18.0</td><td>83.7 25.5</td><td>68.2 25.9</td><td>64.7 44.2</td></tr><tr><td colspan="8">Qwen3-VL-30B-A3B-Instruct Qwen3-VL-8B-Instruct</td></tr><tr><td>Spatial-understanding Models SenseNova-SI-1.1-Qwen3-VL-8B</td><td>76.0 76.0 44.6</td><td>79.1 73.6</td><td>72.2 66.2</td><td>18.5 81.2 19.2 76.3</td><td>46.5 51.6</td><td>42.8 39.5</td><td>58.4 53.5</td></tr><tr><td colspan="8">VST-7B-SFT Underwater-specialized Models OceanGPT</td></tr><tr><td>NAUTILUS-Qwen2.5-VL NAUTILUS-LLaVA-1.5</td><td>65.3 58.7 60.5</td><td>71.0 78.4 42.9</td><td>52.8 40.1 20.9</td><td>18.7 15.9 10.0</td><td>62.7 72.6 71.6</td><td>19.7 56.7 59.2</td><td>20.3 56.4 24.7 54.8 61.9 47.8</td></tr><tr><td colspan="8">Ours</td></tr><tr><td>USI-Brain + UPR</td><td>83.8 (+7.8) 84.8 (+5.7) 83.2 (+11.0) 22.7 (+4.2)</td><td></td><td></td><td></td><td>78.3 (-2.9)</td><td>62.5 (+16.0)</td><td>65.3 (+22.5) 59.1 (+0.7)</td></tr><tr><td colspan="8">USI-Brain</td></tr><tr><td></td><td>82.7 (+6.7)</td><td>84.8 (+5.7)</td><td>82.3 (+10.1)</td><td>22.6 (+4.1) 77.8 (-3.4)</td><td>62.2 (+15.7)</td><td>47.3 (+4.5)</td><td>58.8 (+0.4)</td></tr><tr><td>USI-Brain w/o HUE</td><td>80.8 (+4.8) 77.9 (+1.9)</td><td>84.1 (+5.0) 81.6 (+2.5)</td><td>80.6 (+8.4) 76.2 (+4.0)</td><td>21.4 (+2.9) 76.8 (-4.4) 20.4 (+1.9) 77.7 (-3.5)</td><td>58.9 (+12.4) 60.4 (+13.9)</td><td>45.1 (+2.3) 46.7 (+3.9)</td><td>58.3 (-0.1) 58.6 (+0.2)</td></tr></table>

5/5 on UWBench object referring, and 10/12 on MarineEval. Across these comparison pools, USI-Brain leads 27 metrics

2026). Figure 4 summarizes its best-or-joint-best counts: 7/7 on NautData, 6/8 on the selected UWBench VQA categories, and ties on one, giving 28/32 (87.5%). Its mean signed difference from the strongest baseline on each metric is +9.99 points after rescaling to 100-point scores; Appendix A.1 defines the aggregation.

![](images/4019b940d3b5c3f328d63e964e845623f01164b3a9b5b093cdc290c0ac32ad8e.jpg)  
Figure 4: Percentage of reported metrics on which USI-Brain achieves the best or joint-best score within each comparison pool. UWBench VQA covers selected categories, and MarineEval uses the three-judge protocol described in $\mathsf { A p - }$ pendix A.

The gains span multiple output formats. On NautData, VQA accuracy rises from 0.381 for the strongest baseline to 0.478, while grounding mIoU rises from 53.8 to 56.4. On UWBench object referring, Acc@IoU<sub>0.7</sub> improves by 13.87 points. MarineEval shows gains of 15 points in spatial-relation assessment and 12 in numerical estimation. Appendix A provides the full category results, publishedbaseline comparisons, and the separate three-judge protocol used for MarineEval.

## 6 Conclusion

USI-Brain connects underwater appearance and geometry through a spatial data engine, model adaptation, and tasklevel evaluation. The engine converts images, annotations, and geometric cues into 4.27M QA pairs; USI-Bench measures the resulting capabilities with 6,525 held-out questions. Fine-tuning on this corpus yields a 5.76-point mean gain across 15 tasks, increasing to 7.63 with UPR.

The ablations identify distinct roles for the components: geometry-assisted QA strengthens depth and ordering, HUEderived supervision improves localization and structural perception, and UPR contributes most strongly to scene description. Training-group ablations in Appendix B show how supervision coverage shapes the capability profile, while external evaluations extend the evidence to underwater QA and localization on three benchmarks. We will release the model weights and USI-Bench for further evaluation and development.

## References

Akkaynak, D.; and Treibitz, T. 2019. Sea-Thru: A Method for Removing Water From Underwater Images. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 1682–1691.

Bai, S.; et al. 2025a. Qwen2.5-VL Technical Report. arXiv preprint arXiv:2502.13923.

Bai, S.; et al. 2025b. Qwen3-VL Technical Report. arXiv preprint arXiv:2511.21631.

Bi, Z.; Zhang, N.; Xue, Y.; Ou, Y.; Ji, D.; Zheng, G.; and Chen, H. 2023. OceanGPT: A Large Language Model for Ocean Science Tasks. arXiv preprint arXiv:2310.02031.

Cai, Z.; and Metzler, C. A. 2025. Underwater Monocular Metric Depth Estimation: Real-World Benchmarks and Synthetic Fine-Tuning. arXiv preprint arXiv:2507.02148.

Cai, Z.; Wang, R.; Gu, C.; Pu, F.; Xu, J.; Wang, Y.; Yin, W.; Yang,Z.; Wei, C.; Sun, Q.; Zhou, T.; Li, J.; Pang, H. E.; Qian, O.; Wei, Y.;Lin, Z.; Shi, X.; Deng, K.; Han, X.; Chen, Z.; Fan, X.; Deng, H.; Lu,L.; Pan, L.; Li, B.; Liu, Z.; Wang, Q.; Lin, D.; and Yang, L. 2025.Scaling Spatial Intelligence with Multimodal Foundation Models.arXiv preprint arXiv:2511.13719.

Cao, J.; An, G.; Gao, F.; Gu, K.; and Liu, Y. 2026. UQ-Bench: A Benchmark for Evaluating Multimodal LLMs on Underwater Image Quality Assessment. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 2589–2597.

Chen, B.; Xu, Z.; Kirmani, S.; Ichter, B.; Driess, D.; Florence, P.; Sadigh, D.; Guibas, L. J.; and Xia, F. 2024a. SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities. arXiv preprint arXiv:2401.12168.

Chen, Z.; Wu, J.; Wang, W.; Su, W.; Chen, G.; Xing, S.; Zhong, M.; Zhang, Q.; Zhu, X.; Lu, L.; Li, B.; Luo, P.; Lu, T.; Qiao, Y.; and Dai, J. 2024b. InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 24185–24198.

Daxberger, E.; Wenzel, N.; Griffiths, D.; Gang, H.; Lazarow, J.; Kohavi, G.; Kang, K.; Eichner, M.; Yang, Y.; Dehghan, A.; and Grasch, P. 2025. MM-Spatial: Exploring 3D Spatial Understanding in Multimodal LLMs. In Proceedings of the IEEE/CVF International Conference on Computer Vision.

Deng, N.; Gu, L.; Ye, S.; He, Y.; Chen, Z.; Li, S.; Wang, H.; Wei, X.; Yang, T.; Dou, M.; He, T.; Shao, W.; Zhang, K.; Wang, Y.; Shi, B.; Zhang, Y.; Dai, J.; Qiao, Y.; Zhang, H.; and Wang, W. 2025. InternSpatial: A Comprehensive Dataset for Spatial Reasoning in Vision-Language Models. arXiv preprint arXiv:2506.18385.

Gemini Team, Google. 2025. Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities. Technical report, Google DeepMind.

Gong, Z.; Li, W.; Ma, O.; Li, S.; Ji, J.; Yang, X.; Luo, G.; Yan, J.; and Ji, R. 2025. SpaCE-10: A Comprehensive Benchmark for Multimodal Large Language Models in Compositional Spatial Intelligence. arXiv preprint arXiv:2506.07966.

Google DeepMind. 2026. Gemini 3.1 Pro. Model card.

Gu, J.; Li, S.; Lee, K.; Wang, Y.; and Matusik, W. 2023. Underwater Computational Imaging: A Review. ACM Computing Surveys, 56(1): 1–38.

Gu, J.; Wu, Z.; Si, P.; Qiu, S.; Feng, Y.; Sun, L.; Luo, L.; Yu, L.; Wang, J.; and Wu, Z. 2025. USIM and U0: A Vision-Language-Action Dataset and Model for General Underwater Robots. arXiv preprint arXiv:2510.07869.

Han, H.; Wang, W.; Zhang, G.; Li, M.; and Wang, Y. 2025. CoralVQA: A Large-Scale Visual Question Answering Dataset for Coral Reef Image Understanding. arXiv preprint arXiv:2507.10449.

Hong, W.; et al. 2025. GLM-4.5V and GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning. arXiv preprint arXiv:2507.01006.

Huy, D. Q.; Sadjoli, N.; Azam, A. B.; Elhadidi, B.; Cai, Y.; and Seet, G. 2023. Object perception in underwater environments: a survey on sensors and sensing methodologies. Ocean Engineering, 267: 113202.

Jia, M.; Qi, Z.; Zhang, S.; Zhang, W.; Yu, X.; He, J.; Wang, H.; and Yi, L. 2025. OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models. arXiv preprint arXiv:2506.03135.

Li, C.; Guo, C.; Ren, W.; Cong, R.; Hou, J.; Kwong, S. T. W.; and Tao, D. 2020. An Underwater Image Enhancement Benchmark Dataset and Beyond. IEEE Transactions on Image Processing, 29: 4376–4389.

Li, H.; Li, D.; Wang, Z.; Yan, Y.; Wu, H.; Zhang, W.; Shen, Y.; Lu, W.; Xiao, J.; and Zhuang, Y. 2025. SpatialLadder: Progressive Training for Spatial Reasoning in Vision-Language Models. arXiv preprint arXiv:2510.08531.

Lin, H.; Liang, D.; Qi, Z.; and Bai, X. 2025. A Unified Image-Dense Annotation Generation Model for Underwater Scenes. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 961–970.

Liu, D.; Kuang, J.; Li, Y.; Li, Y.; Yin, D.; Cao, H.; Sun, X.; Shen, Y.; Zheng, H.-T.; Lin, L.; and Yu, P. S. 2026a. TangramPuzzle: Evaluating Multimodal Large Language Models with Compositional Spatial Reasoning. arXiv preprint arXiv:2601.16520.

Liu, F.; Emerson, G.; and Collier, N. 2023. Visual Spatial Reasoning. Transactions ofthe Associationfor Computational Linguistics, 11: 635–651.

Liu, J.; Sun, H.; Li, W.; Zhang, Y.; Yang, R.; Zhu, Z.; Yang, Y.; Zheng, S.; Jiang, N.; Jiang, J.; Huang, H.; Wong, T.-T.; Duan, N.; and Qi, X. 2026b. OpenSpatial: A Principled Data Engine for Empowering Spatial Intelligence. arXiv preprint arXiv:2604.07296.

Nauert, M.; and Kampmann, P. 2023. Inspection and Maintenance of Industrial Infrastructures by Autonomous Underwater Robots: Current Developments and Future Challenges. Frontiers in Robotics and AI, 10: 1272712.

Wong, Y. K.; To, T.-A.; Zhang, J.; Zheng, Z.; and Yeung, S.-K. 2026. MarineEval: Assessing the Marine Intelligence of Vision-Language Models. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 1459–1470.

Wynn, R. B.; Huvenne, V. A. I.; Bas, T. P. L.; Murton, B. J.; Connelly, D. P.; Bett, B. J.; Ruhl, H. A.; Morris, K. J.; Peakall, J.; Parsons, D. R.; Sumner, E. J.; Darby, S. E.; Dorrell, R. M.; and Hunt, J. E. 2014. Autonomous Underwater Vehicles (AUVs): Their Past, Present and Future Contributions to the Advancement of Marine Geoscience. Marine Geology, 352: 451–468.

Xu, P.; Wang, S.; Zhu, Y.; Li, J.; and Zhang, Y. 2025a. SpatialBench: Benchmarking Multimodal Large Language Models for Spatial Cognition. arXiv preprint arXiv:2511.21471.

Xu, W.; Wang, C.; Liang, D.; Zhao, Z.; Jiang, X.; Zhang, P.; and Bai, X. 2025b. NAUTILUS: A Large Multimodal Model for Underwater Scene Understanding. arXiv preprint arXiv:2510.27481.

Xu, Z.; Zhang, Y.; Adhikari, S.; Islam, S.; Xiao, T.; Liu, Z.; Chen, S.; Yan, D.; and Jiang, Z. 2026. EarthSpatialBench: Benchmarking Spatial Reasoning Capabilities of Multimodal LLMs on Earth Imagery. arXiv preprint arXiv:2602.15918.

Xue, Y.; Mao, M.; Ru, X.; Zhu, Y.; Ren, B.; Qiao, S.; Wang, M.; Deng, S.; An, X.; Zhang, N.; Chen, Y.; and Chen, H. 2025. OceanGym: A Benchmark Environment for Underwater Embodied Agents. arXiv preprint arXiv:2509.26536.

Yang, R.; Chen, H.; Zhang, J.; Zhao, M.; Qian, C.; Wang, K.; Wang, Q.; Koripella, T. V.; Movahedi, M.; Li, M.; Ji, H.; Zhang, H.; and Zhang, T. 2025a. EmbodiedBench: Comprehensive Benchmarking Multi-modal Large Language Models for Vision-Driven Embodied Agents. In Proceedings of the 42nd International Conference on Machine Learning.

Yang, R.; Zhu, Z.; Li, Y.; Huang, J.; Yan, S.; Zhou, S.; Liu, Z.; Li, X.; Li, S.; Wang, W.; Lin, Y.; and Zhao, H. 2025b. Visual Spatial Tuning. arXiv preprint arXiv:2511.05491.

Yang, S.; Xu, R.; Xie, Y.; Yang, S.; Li, M.; Lin, J.; Zhu, C.; Chen, X.; Duan, H.; Yue, X.; Lin, D.; Wang, T.; and Pang, J. 2025c. MMSI-Bench: A Benchmark for Multi-Image Spatial Intelligence. arXiv preprint arXiv:2505.23764.

Yuan, X.; et al. 2023. Marine Environmental Monitoring with Unmanned Vehicle Platforms: Present Status and Future Prospects. Science ofThe Total Environment, 858: 159741.

Zhang, C.; Liu, L.; Huang, G.; Wen, H.; Zhou, X.; and Wang, Y. 2024a. WebUOT-1M: Advancing Deep Underwater Object Tracking with A Million-Scale Benchmark. In Advances in Neural Information Processing Systems, volume 37, 50152–50167.

Zhang, D.; Rong, C.; Li, B.; Wang, F.; Zhao, Z.; Gao, J.; and Li, X. 2025. UWBench: A Comprehensive Vision-Language Benchmark for Underwater Understanding. arXiv preprint arXiv:2510.18262.

Zhang, F.; You, S.; Li, Y.; and Fu, Y. 2024b. Atlantis: Enabling Underwater Depth Estimation with Stable Diffusion. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 11852–11861.

Zhang, G.; Li, W.; Qian, Q.; Wang, J.; Zhao, D.; Lu, S.; and Xu, R. 2026. On the Generalization Capacities of MLLMs for Spatial Intelligence. arXiv preprint arXiv:2603.06704.

Zhang, G.; Wang, H.; Yan, Z.; Ji, X.; Wang, A.; and Qian, J. 2023. Autonomous Underwater Vehicle Navigation: A Review. Ocean Engineering, 273: 113861.

Zheng, L.; Chiang, W.-L.; Sheng, Y.; Zhuang, S.; Wu, Z.; Zhuang, Y.; Lin, Z.; Li, Z.; Li, D.; Xing, E. P.; Zhang, H.; Gonzalez, J. E.; and Stoica, I. 2023. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. arXiv preprint arXiv:2306.05685.

Zheng, Z.; Chen, Y.; Zeng, H.; Vu, T.-A.; Hua, B.-S.; and Yeung, S.-K. 2024a. MarineInst: A Foundation Model for Marine Image Analysis with Instance Visual Description. In Computer Vision – ECCV 2024, 239–257.

Zheng, Z.; Liang, H.; Hua, B.-S.; Wong, Y. H.; Ang, P., Jr; Chui, A. P. Y.; and Yeung, S.-K. 2024b. CoralSCOP: Segment any COral Image on this Planet. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 28170–28180.

## A External Benchmark Results

Tables 4–8 give aggregate and category-level results on NautData, UWBench, and MarineEval. The following subsections specify each comparison pool, metric, and evaluation protocol. In Tables 5–8, darker/lighter blue marks the best/second-best distinct score in each column among the listed models; ties share a color.

## A.1 Summary and Aggregation

Table 4 summarizes 32 reported metrics. USI-Brain achieves a strictly higher score than all compared baselines on 27 metrics and ties for the highest score on one, giving a best-or-joint-best rate of 87.5%.

For aggregation, scores reported on a unit scale are multiplied by 100; scores already on a 100-point scale are left unchanged. Let $s _ { j }$ denote the rescaled USI-Brain score on metric j, and let $b _ { j , k }$ denote the corresponding score of baseline k in comparison pool $\mathscr { C } _ { j }$ . We compute

$$
\Delta _ { j } = s _ { j } - \operatorname* { m a x } _ { k \in \mathscr { C } _ { j } } b _ { j , k } ,
$$

$$
\overline { { \Delta } } = \frac { 1 } { J } \sum _ { j = 1 } ^ { J } \Delta _ { j } .\tag{10}
$$

With $J = 3 2$ , this gives $\overline { { \Delta } } = 9 . 9 9$ points. Every listed metric has equal weight, negative differences retain their signs, and CLAIR contributes its reported mean.

Table 4: Summary of the reported external comparisons. Sole best and joint best compare USI-Brain with the strongest non-USI-Brain baseline for each metric. Mean score differences use Equation 10, with all scores rescaled to a 100-point scale.

<table><tr><td>Benchmark</td><td>Sole best</td><td>Joint best</td><td>Mean score difference</td></tr><tr><td>NautData (Xu et al. 2025b)</td><td>7/7</td><td>0</td><td>+2.77 points</td></tr><tr><td>UWBench VQA (Zhang et al. 2025)</td><td>6/8</td><td>0</td><td>+24.41 points</td></tr><tr><td>UWBench Object Referring (Zhang et al. 2025)</td><td>5/5</td><td>0</td><td>+7.80 points</td></tr><tr><td>MarineEval (Wong et al. 2026)</td><td>9/12</td><td>1/12</td><td>+5.50 points</td></tr><tr><td>Overall</td><td>27/32</td><td>1/32</td><td>+9.99 points</td></tr></table>

## A.2 NautData

Table 5 reports results on the public NautData test split from NAUTILUS (Xu et al. 2025b). The seven metrics cover image and region captioning, visual question answering, grounding, and detection. Captioning and VQA retain their unit-scale scores; Table 4 multiplies these by 100 when computing the aggregate.

Table 5: Results on the public NautData test split from NAUTILUS (Xu et al. 2025b). Image and region captioning use METEOR, VQA use accuracy, grounding uses PR@0.5 and mIoU, and detection uses mAP and mAP@0.5. METEOR and VQA accuracy are shown on a unit scale; the remaining metrics are shown on a 100-point scale. Boldface marks the best score among the listed methods.

<table><tr><td rowspan="2">Method</td><td>Image Caption</td><td>Region Caption</td><td rowspan="2">VQA Acc</td><td colspan="2">Grounding</td><td colspan="2">Detection</td></tr><tr><td>METEOR</td><td>METEOR</td><td>PR@0.5</td><td>mIoU</td><td>mAP</td><td>mAP@0.5</td></tr><tr><td>NAUTILUS (LLaVA-1.5) (Xu et al. 2025b)</td><td>0.208</td><td>0.191</td><td>0.365</td><td>52.2</td><td>46.2</td><td>11.1</td><td>20.9</td></tr><tr><td>NAUTILUS (Qwen2.5-VL) (Xu et al. 2025b)</td><td>0.223</td><td>0.199</td><td>0.381</td><td>58.8</td><td>53.8</td><td>25.8</td><td>45.3</td></tr><tr><td>Qwen3-VL-8B (Bai et al. 2025b)</td><td>0.089</td><td>0.097</td><td>0.008</td><td>45.2</td><td>43.9</td><td>14.8</td><td>24.0</td></tr><tr><td>USI-Brain (ours)</td><td>0.245</td><td>0.206</td><td>0.478</td><td>59.4</td><td>56.4</td><td>27.9</td><td>46.8</td></tr></table>

USI-Brain achieves the highest score among the listed methods on all seven metrics. VQA accuracy increases from 0.381 for the strongest baseline to 0.478, while grounding mIoU increases from 53.8 to 56.4. Image-caption METEOR increases from 0.223 to 0.245, and detection mAP increases from 25.8 to 27.9.

## A.3 UWBench Visual Question Answering

Table 6 reports eight selected UWBench VQA categories related to object understanding, spatial reasoning, and underwater visibility. Baseline scores are taken from the UWBench paper (Zhang et al. 2025). USI-Brain is evaluated on the official test split using the same protocol.

Table 6: Selected UWBench VQA results (Zhang et al. 2025). All scores use a 100-point scale. Baseline numbers are taken from the UWBench paper; USI-Brain is evaluated on the official test split under the same protocol. Boldface marks the best score among the listed methods.
<table><tr><td>Method</td><td>Qty. &amp; Exist.</td><td>Reas. &amp; Comp.</td><td>Water Qual. &amp; Vis.</td><td>Obj. Attr. &amp; Feat.</td></tr><tr><td>Gemini-2.5-Flash (Gemini Team, Google 2025)</td><td>1.47</td><td>26.14</td><td>40.77</td><td>24.37</td></tr><tr><td>Qwen2.5-VL-7B (Bai et al. 2025a)</td><td>49.91</td><td>23.24</td><td>35.54</td><td>24.37</td></tr><tr><td>GLM-4.1V-9B (Hong et al. 2025)</td><td>2.33</td><td>26.97</td><td>39.94</td><td>28.12</td></tr><tr><td>GLM-4.5V-106B (Hong et al. 2025)</td><td>8.73</td><td>29.05</td><td>40.22</td><td>27.50</td></tr><tr><td>USI-Brain (ours)</td><td>92.72</td><td>85.11</td><td>82.50</td><td>78.09</td></tr></table>

<table><tr><td>Method</td><td>Substrate &amp; Materials</td><td>Pos. &amp; Spatial Rel.</td><td>Color, Texture &amp; Pattern</td><td>Shape, Size &amp; Form</td></tr><tr><td>Gemini-2.5-Flash (Gemini Team, Google 2025)</td><td>47.58</td><td>61.55</td><td>45.26</td><td>71.42</td></tr><tr><td>Qwen2.5-VL-7B (Bai et al. 2025a)</td><td>42.24</td><td>55.66</td><td>38.95</td><td>64.42</td></tr><tr><td>GLM-4.1V-9B (Hong et al. 2025)</td><td>47.17</td><td>61.27</td><td>45.02</td><td>71.98</td></tr><tr><td>GLM-4.5V-106B (Hong et al. 2025)</td><td>49.85</td><td>62.48</td><td>48.78</td><td>72.52</td></tr><tr><td>USI-Brain (ours)</td><td>72.45</td><td>56.73</td><td>56.33</td><td>52.79</td></tr></table>

USI-Brain achieves the highest scores in six of the eight reported categories: quantity and existence, reasoning and comparison, water quality and visibility, object attributes and features, substrate and materials, and color, texture, and pattern. GLM-4.5V-106B scores higher on position and spatial relations (62.48 versus 56.73) and shape, size, and form (72.52 versus 52.79).

## A.4 UWBench Object Referring

Table 7 reports four localization metrics from the official UWBench object-referring protocol and a complementary CLAIR score for open-ended output quality. The reported CLAIR means are used in the aggregate comparison.

Table 7: Object-referring results on UWBench (Zhang et al. 2025). The first four metrics follow the official localization protocol. CLAIR is a complementary judge-based metric, reported as mean ± standard deviation. Scores use a 100-point scale. Baseline results are taken from the UWBench paper. Boldface marks the best reported mean or score in each column.

<table><tr><td>Method</td><td>Acc@IoU0.5</td><td>Acc@IoU0.7</td><td>mIoU</td><td>Cum_IoU</td><td>CLAIR</td></tr><tr><td>Qwen2.5-VL-7B (Bai et al. 2025a)</td><td>37.15</td><td>11.76</td><td>32.77</td><td>33.69</td><td>87.28±4.86</td></tr><tr><td>Qwen2.5-VL-72B (Bai et al. 2025a)</td><td>90.90</td><td>74.71</td><td>77.90</td><td>81.50</td><td>88.76±4.64</td></tr><tr><td>InternVL-3.5-24B (Chen et al. 2024b)</td><td>84.76</td><td>64.14</td><td>71.99</td><td>76.98</td><td>88.99±4.42</td></tr><tr><td>Qwen3-VL-30B-Instruct (Bai et al. 2025b)</td><td>94.40</td><td>80.38</td><td>80.18</td><td>51.67</td><td>88.58±5.21</td></tr><tr><td>GLM-4.5V-106B (Hong et al. 2025)</td><td>89.21</td><td>75.38</td><td>79.26</td><td>80.51</td><td>88.63±6.43</td></tr><tr><td>USI-Brain (ours)</td><td>96.80</td><td>94.25</td><td>91.05</td><td>92.07</td><td>90.27±4.89</td></tr></table>

USI-Brain achieves the highest values on all four localization metrics and the highest mean CLAIR score among the listed methods. The largest localization gain is at the stricter IoU threshold: Acc@IoU<sub>0.7</sub> increases from 80.38 for the strongest baseline to 94.25, a difference of 13.87 points. Its mIoU is 91.05, compared with 80.18 for the strongest baseline.

## A.5 MarineEval

Table 8 reports 12 categories from MarineEval (Wong et al. 2026). All listed models are scored under our three-judge protocol using Gemini-3- Pro-Preview, GPT-5.5, and Claude Opus 4.7. The table reports this shared evaluation; the original MarineEval leaderboard uses a different protocol.

Table 8: Selected MarineEval results under the reported three-judge protocol (Wong et al. 2026). Scores use a 100-point scale. The judges are Gemini-3-Pro-Preview, GPT-5.5, and Claude Opus 4.7. Boldface marks the best or joint-best score among the listed methods.
<table><tr><td>Method</td><td>Num. Est.</td><td>Visual Ground.</td><td>Spatial Rel. Assess.</td><td>Depth Order.</td><td>Instr. Func. Ident.</td><td>Behavior Class.</td></tr><tr><td>Nautilus-7B (Xu et al. 2025b)</td><td>11.00</td><td>33.00</td><td>7.00</td><td>24.00</td><td>60.00</td><td>36.00</td></tr><tr><td>OceanGPT-7B (Bi et al. 2023)</td><td>20.00</td><td>20.00</td><td>2.00</td><td>0.00</td><td>28.00</td><td>1.00</td></tr><tr><td>Qwen3-VL-8B (Bai et al. 2025b)</td><td>34.00</td><td>84.00</td><td>14.00</td><td>29.00</td><td>66.00</td><td>63.00</td></tr><tr><td>USI-Brain (ours)</td><td>46.00</td><td>93.00</td><td>29.00</td><td>31.00</td><td>77.00</td><td>69.00</td></tr></table>

<table><tr><td>Method</td><td>Traits Extract.</td><td>Threat Status Determ.</td><td>Pollutant Localiz.</td><td>Disaster Diagn.</td><td>Species Compr.</td><td>Marine Tech. Underst.</td></tr><tr><td>Nautilus-7B (Xu et al. 2025b)</td><td>77.00</td><td>15.00</td><td>33.00</td><td>94.00</td><td>27.70</td><td>60.00</td></tr><tr><td>OceanGPT-7B (Bi et al. 2023)</td><td>33.00</td><td>0.00</td><td>31.00</td><td>85.00</td><td>18.00</td><td>28.00</td></tr><tr><td>Qwen3-VL-8B (Bai et al. 2025b)</td><td>87.00</td><td>32.00</td><td>40.00</td><td>93.00</td><td>51.80</td><td>71.00</td></tr><tr><td>USI-Brain (ours)</td><td>89.00</td><td>51.00</td><td>37.00</td><td>94.00</td><td>43.83</td><td>72.00</td></tr></table>

USI-Brain achieves the highest scores in nine categories and ties with Nautilus-7B on disaster diagnosis. Relative to the strongest baseline for each category, the gains include 15 points in spatial-relation assessment, 12 in numerical estimation, 11 in instrument-function identification, and 19 in threat-status determination. Qwen3-VL-8B scores higher on pollutant localization and species comprehension in this comparison.

## B Data and Component Ablations

We vary the capability groups included in training and evaluate every configuration on all 15 USI-Bench tasks. Table 9 lists the training groups and enabled components. Base is Qwen3-VL-8B-Instruct; B1–B5 follow Section 4.2. The UPR comparison uses matched training QA and settings, while the source-removal and group-subset comparisons vary the included supervision.

Table 9: Data and component ablations on USI-Bench. Base reports absolute scores; every other row reports signed gains over Base. Overall equally weights B1–B5. ✓/× indicate enabled/disabled components; — denotes not applicable. Boldface marks the largest gain. Darker/lighter blue marks the best/second-best distinct gains among the ablations, excluding Base. Positive/negative differences are red/green.
<table><tr><td rowspan="2">Training groups</td><td colspan="3">Components</td><td colspan="6">Base scores / signed gains</td></tr><tr><td>HUE</td><td>GeoQA</td><td>UPR</td><td>B1</td><td>B2</td><td>B3</td><td>B4</td><td>B5</td><td>Overall</td></tr><tr><td>Base</td><td>一</td><td></td><td>X</td><td>55.96</td><td>67.85</td><td>61.45</td><td>63.85</td><td>50.60</td><td>59.94</td></tr><tr><td>B1</td><td>X</td><td>X</td><td>X</td><td>+3.10</td><td>+0.20</td><td>+0.50</td><td>-0.20</td><td>-0.10</td><td>+0.70</td></tr><tr><td>B1+B2</td><td>X</td><td>X</td><td>X</td><td>+3.00</td><td>+1.60</td><td>+0.90</td><td>-0.10</td><td>-0.20</td><td>+1.04</td></tr><tr><td>B1+B3</td><td>X</td><td>√</td><td>X</td><td>+3.20</td><td>+0.80</td><td>+4.60</td><td>+0.10</td><td>-0.10</td><td>+1.72</td></tr><tr><td>B2+B3</td><td>X</td><td>√</td><td>X</td><td>+0.10</td><td>+1.90</td><td>+4.30</td><td>-0.20</td><td>-0.30</td><td>+1.16</td></tr><tr><td>B1+B2+B3</td><td>X</td><td>√</td><td>X</td><td>+3.10</td><td>+1.90</td><td>+5.10</td><td>+0.20</td><td>-0.20</td><td>+2.02</td></tr><tr><td>B1-B4</td><td>X</td><td>√</td><td>X</td><td>+3.30</td><td>+1.80</td><td>+5.20</td><td>+3.00</td><td>-0.10</td><td>+2.64</td></tr><tr><td>B1-B5</td><td>X</td><td>√</td><td>X</td><td>+4.88</td><td>+1.75</td><td>+5.28</td><td>+4.00</td><td>+1.10</td><td>+3.40</td></tr><tr><td>B1+B2+B4+B5</td><td>√</td><td>X</td><td>X</td><td>+7.18</td><td>-1.00</td><td>+2.58</td><td>+5.20</td><td>+2.05</td><td>+3.20</td></tr><tr><td>B1-B5</td><td>√</td><td>V</td><td>X</td><td>+7.58</td><td>+2.35</td><td>+6.65</td><td>+6.15</td><td>+2.45</td><td>+5.04</td></tr><tr><td>B1–B5 (Full)</td><td>√</td><td>V</td><td>V</td><td>+8.54</td><td>+3.35</td><td>+7.18</td><td>+6.55</td><td>+11.60</td><td>+7.44</td></tr></table>

## B.1 Aggregation and Reporting

Within each capability group, we average the task scores reported on the 100-point scale. B1 contains Identity, Grounding, Counting, Detection, and Captioning; B2 contains Relation and Distance; B3 contains Depth, Viewpoint, Ordering, and Fg./Bg.; B4 contains Environment and Structure; B5 contains Scene and Ecology. The group sizes are therefore 5, 2, 4, 2, and 2.

For model m, let $\mathcal { T } _ { b }$ be the task set of group b and $s _ { m , t }$ its score on task t. We define

$$
\begin{array} { r l } & { S _ { m , b } = \displaystyle \frac { 1 } { | \mathcal T _ { b } | } \sum _ { t \in \mathcal T _ { b } } s _ { m , t } , \qquad \Delta _ { m , b } = S _ { m , b } - S _ { \mathrm { B a s e } , b } , } \\ & { S _ { m , \mathrm { O v e r a l l } } = \displaystyle \frac { 1 } { 5 } \sum _ { b = 1 } ^ { 5 } S _ { m , b } . } \end{array}\tag{11}
$$

The Overall gain equally weights the five group gains, whereas the main-text average equally weights the 15 tasks. This yields +5.04/+7.44 for USI-Brain/Full, compared with +5.76/+7.63 under task weighting. Means and differences are computed before rounding to two decimals; subtracting rounded entries can differ by 0.01.

## B.2 Configuration Correspondence

B1–B5 with HUE and GeoQA but without UPR is USI-Brain; enabling UPR gives Full. B1–B5 without HUE maps to w/o HUE; B1+B2+B4+B5 with HUE and without GeoQA maps to w/o GeoQA. Both removals disable UPR. These four rows reproduce Table 3’s group means. Full and USI-Brain use the same 4.27M QA and training settings; all variants use raw evaluation images.

## B.3 Effects of Training Coverage and Components

Training-group coverage. Expanding training coverage produces gains in the corresponding capability groups: adding B3 to B1 training increases the B3 gain from +0.50 to +4.60 points, while adding B4 to B1+B2+B3 increases the B4 gain from +0.20 to +3.00 points.

Supervision sources. With all five groups included and UPR disabled, HUE raises B1 and B4 scores by 2.70 and 2.15 points. Removing GeoQA from USI-Brain lowers the B3 gain from +6.65 to +2.58 and the B2 gain from +2.35 to −1.00.

Feature rectification. Adding UPR to USI-Brain improves B5 by a further 9.15 points, with smaller gains of 0.40–1.00 points across B1–B4. The B5 change averages the 18.0-point Scene gain and the 0.3-point Ecology gain.

## C Inference Efficiency

Table 10 summarizes inference on a single GPU with BF16 and batch size 1. A800 runs use Transformers with FlashAttention 2; RTX 3090 (24 GB) runs use Transformers 4.57.1 with SDPA. USI-Brain disables UPR, while USI-Brain + UPR includes the prior encoder and feature-rectification path. All configurations use raw input images. The input sequence lengths and attention backends follow the configuration reported for each GPU.

Table 10: Single-GPU inference efficiency. In/Out denotes input/generated token counts; E2E denotes end-to-end latency. USI-Brain latencies are medians, and its RTX 3090 token counts and decoding speeds are also medians. Memory is PyTorch peak GPU memory.
<table><tr><td>GPU</td><td>Model</td><td>Images</td><td>In/Out tokens</td><td>Decode (tokens/s)</td><td>E2E (s)</td><td>Peak memory (GiB)</td></tr><tr><td>A800</td><td>USI-Brain</td><td>1</td><td>3117 / 21.0</td><td>27.50</td><td>1.35</td><td>17.24</td></tr><tr><td>A800</td><td>USI-Brain</td><td>2</td><td>6210 / 25.0</td><td>27.70</td><td>2.08</td><td>18.14</td></tr><tr><td>RTX 3090</td><td>USI-Brain</td><td>1</td><td>793 / 18.5</td><td>35.86</td><td>0.84</td><td></td></tr><tr><td>RTX 3090</td><td>USI-Brain</td><td>2</td><td>1563 / 17.5</td><td>33.26</td><td>1.12</td><td>16.57 16.79</td></tr><tr><td>RTX 3090</td><td>USI-Brain + UPR</td><td>1</td><td>1561 / 20.0</td><td>34.50</td><td>1.63</td><td>21.80</td></tr><tr><td>RTX 3090</td><td>USI-Brain + UPR</td><td>2</td><td>3099 / 26.5</td><td>30.90</td><td>2.95</td><td>23.60</td></tr></table>

On A800, median time to first token, including preprocessing, is 0.63 s for one image and 1.21 s for two views. The corresponding E2E P95 values are 1.37 s and 2.08 s. On RTX 3090, USI-Brain answers single-image queries in a median of 0.84 s using 16.57 GiB. USI-Brain + UPR records 1.63 s latency and 21.80 GiB peak memory for a single image, and 2.95 s and 23.60 GiB for two images.