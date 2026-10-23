![](images/98e89f382285bb562bdba63e24879ec3711fa6998f54a5e62d70f14d2fa926ad.jpg)  
Code: https://github.com/Tencent/WeVisOne-VL  
Weights: https://huggingface.co/collections/tencent/wevisone-vl  
Figure 1 Overview of WeVisOne-VL’s unified object- and pixel-level perception capabilities.

# WeVisOne-VL: Unified Visual Perception from Objects to Pixels

Tianming Liang<sup>\*</sup>, Junyi Pan<sup>\*</sup>, Yuxi Li<sup>\*</sup>, Canmiao Fu, Zhipeng Huang, Chong Sun†, Chen Li‡, Jing Lyu WeChat Vision, Tencent Inc.

## Abstract

Existing vision-language models (VLMs) remain unreliable for fine-grained visual perception. In this report, we present WeVisOne-VL (WeChat Vision All-in-One), a powerful VLM that unifies object-level and pixel-level perception through structured next-token prediction. It supports multiple visual tasks including object detection, referring localization, counting, OCR, segmentation, and matting, enabling localization at any user-specific granularity, from points and boxes to masks and alpha mattes. Architecturally, we integrate a lightweight pixel decoder into the VLM via a special <OBJ> token to generate instance-level pixel outputs. We train the model with a three-stage curriculum SFT followed by mixed-task RL. Across over 30 benchmarks, WeVisOne-VL demonstrates leading performance on both object-level and pixel-level tasks, while maintaining robust rejection for object-absent queries. By releasing WeVisOne-VL to the open-source community, we hope to advance VLMs from broad visual understanding toward truly reliable perception systems for real-world deployment.

Project Page: https://tencent.github.io/WeVisOne-VL

Pixel-level Perception segmentation & matting from a single <OBJ> token

## 1 Introduction

Perceiving and understanding the visual world has long been a central pursuit of computer vision [1, 4, 25, 32, 45, 46, 70, 94]. An intelligent visual system must be able to recognize what is present in an image, identify which object a user is referring to, and determine where these objects exist. Such capabilities establish the foundation for a wide range of real-world applications, such as content creation [65], industrial inspection [37], autonomous driving, and embodied intelligence [18].

Over the past decade, computer vision has made remarkable progress toward this goal. From conventional detection and segmentation models (e.g., YOLO [86] and DETR [4]), to open-world perception models (e.g., Grounding-DINO [67] and SAM [45]), and more recently to large Vision-Language Models (VLMs) [1], this field is being driven toward more unified, general, and powerful visual systems. This evolution has substan tially broadened the concepts that a model can recognize and the instructions that it can follow. However, when both performance and eficiency are considered, current models are still far from satisfactory for realworld deployment. They often hallucinate non-existent objects, misinterpret user intent when encountering unfamiliar object descriptions, or overlook non-salient instances, leading to incomplete recall.

In this report, we introduce WeVisOne-VL (WeChat Vision All-in-One), a frontier vision-language model that unifies object-level and pixel-level visual perception through a single natural-language interface. The model casts diverse visual tasks as structured next-token prediction and incorporates a lightweight pixel decoder for generating dense outputs. Built on Qwen3-VL [1], we instantiate our models at two scales: 2B and 4B. We train the models on approximately 100M images with fine-grain annotations using a multi-stage curriculum. The key designs of WeVisOne-VL are summarized as follows.

Data Engine. We develop a unified data engine that connects semantic discovery with instance-aligned super vision for object detection, referring localization, segmentation, and matting. To guide targeted acquisition and task-specific annotation, the engine organizes category concepts and compositional referring expressions in a shared knowledge graph. It then combines complementary localization experts to generate spatial candidates and consolidates their predictions through visual-language verification of query–instance correspondence. Building on these verified localizations, the engine generates binary masks to extend supervision to the pixel level while preserving target identity across tasks, with manual spot checks for additional quality control. For matting, it further leverages expert predictions and curated data to support a dedicated anno tator that provides continuous alpha supervision. Finally, the engine uses reviewed model errors to guide subsequent acquisition, annotation correction, and hard-example sampling, enabling iterative refinement of the training data.

Model Design. WeVisOne-VL consists of a VLM backbone and a lightweight pixel decoder. Built on Qwen3- VL [1], WeVisOne-VL casts a broad range of vision tasks as structured next-token prediction, generating object labels and spatial coordinates autoregressively. For compact and eficient output, we introduce a group of dedicated tokens (from <0> to <1000>) to encode normalized coordinates [41]. For dense prediction tasks, the model generates a special <OBJ> token after each predicted bounding box, whose hidden state serves as a latent representation of the corresponding instance. The pixel decoder combines this instance representation with multi-level features from the shared ViT to produce predictions over the full image canvas. In this way, WeVisOne-VL supports both discrete and dense predictions without relying on a separate large segmentation model [48, 62, 68, 122].

Training Pipeline. We train WeVisOne-VL in a curriculum pipeline consisting of three-stage Supervised Fine-Tuning (SFT) followed by mixed-task Reinforcement Learning (RL). Stage 1 establishes task alignment, teaching the model to follow task instructions and produce well-formed structured outputs. Stage 2 introduces high-resolution and dense perception data to improve the recall of small and crowded objects. Stage 3 further incorporates pixel-level tasks, extending the model from discrete localization to dense prediction. For RL, we first select samples on which the SFT model is most uncertain, then apply Group Relative Policy Optimization (GRPO) over the mixture of tasks. This on-policy optimization efectively improves the localization accuracy, while suppressing repeated or hallucinated predictions.

Across multiple visual benchmarks, WeVisOne-VL demonstrates highly competitive performance against both state-of-the-art general-purpose models and specialized vision systems. It exhibits consistently strong capabilities on both object-level and pixel-level tasks, including object detection, fine-grained referring, text recognition and language-guided pixel prediction. In addition, it also shows strong potential in challenging real-world scenarios, such as maintaining high recall in crowded scenes and correctly rejecting queries when the target is absent. By unifying these capabilities into a single model, we ofer a powerful, eficient, and versatile vision-language foundation for the open-source community, accelerating the development of truly practical and reliable visual systems.

![](images/5fefa8a167f433cb4a4d8c37f0f7ee4106945140b9f2ec3048256c523bdbb3aa.jpg)  
Figure 2 Overview of the WeVisOne-VL Data Engine (WDE).

## 2 WeVisOne-VL Data Engine

Data construction for unified visual perception must support both category-level recognition and contextdependent target selection, while preserving instance associations across language and spatial supervision. To this end, we build the WeVisOne-VL Data Engine (WDE). WDE draws on GRIT [78], CC3M [90], LAION [49], SA-1B [45], and several task-specific datasets, and its annotation inventory comprises over 100 million images and approximately 3.6 billion bounding-box annotations.

Figure 2 shows an overview of WDE. We first organize category concepts and referring expressions in a shared knowledge graph, which guides targeted image acquisition (Section 2.1). The acquired image–query pairs then go through task-specific localization and semantic verification, which establish the correspondence between each query and its target instances (Sections 2.2 and 2.3). We keep these correspondences when generating binary masks (Section 2.4), and further extend the instance annotations to continuous alpha supervision with a dedicated matting pipeline (Section 2.6). Finally, model errors reviewed after each training round guide subsequent acquisition and annotation refinement (Section 2.7). Below, we describe each component in detail.

## 2.1 Concept Discovery and Targeted Acquisition

Semantic discovery produces two complementary annotations: category-level concepts and compositional referring expressions. For detection, we use an optimized Tokenize Anything model [76] to discover object categories. For referring tasks, we prompt Qwen3-VL [1] to generate referring expressions that specify targets through attributes, states, actions, and relations. Category concepts support category-based localization, while referring expressions support selecting particular instances from their context. Both serve as annotation queries, and their correspondence to image regions is established in the subsequent localization stages.

We organize both category concepts and referring expressions in a shared knowledge graph. The graph is initialized from resources including Wikipedia<sup>1</sup> and WordNet [73], and represents canonical category names, aliases, taxonomic relations, and whole–part relations. A candidate term is first linked to existing nodes through text retrieval and vision-language verification, and is considered for addition to the graph only when it is supported by stable visual evidence. During normalization, we merge equivalent category terms while keeping the descriptive constraints that determine the scope of referring expressions.

This semantic organization also supports targeted acquisition. Concept names and aliases serve as textual retrieval queries, verified instances provide visual retrieval cues, and historical errors indicate where additional coverage is needed. We set sampling priorities according to concept representation, image domain, target scale, and challenging conditions such as occlusion and dense object arrangements, and confusable categories and recurring failures in interpreting referring expressions further inform acquisition. In this way, annotation efort is directed toward both semantic gaps and visual conditions that remain dificult for the current model. The acquired image–query pairs are then sent to the detection or referring annotation pipeline.

## 2.2 Detection Annotation

Conventional object detectors excel at detecting densely distributed objects but struggle to recognize long-tail categories and open-world concepts. By contrast, VLMs ofer strong recognition capabilities but often exhibit low recall in dense object scenarios. Therefore, for each object category, we first utilize detectors to generate exhaustive proposals, and then use a VLM for post verification. Specifically, we obtain proposals in parallel from SAM3 [5] and an in-house open-vocabulary detector based on a DINO-style architecture [123], which we refine iteratively. For each image–query pair, the predictions of both models are merged into a shared candidate pool. This design reduces our reliance on any single proposal source and uses the variation across detectors to pursue broader instance coverage.

For detection, we obtain proposals in parallel from two models: SAM3 [5] and an in-house open-vocabulary detector based on a DINO-style architecture [123], which we refine iteratively. For each image–query pair, the predictions of both models are merged into a shared candidate pool. This design reduces our reliance on any single proposal source and uses the variation across detectors to pursue broader instance coverage.

We then use Qwen3-VL [1] to verify whether each proposed region matches the category query given the image content. Based on its judgments, we select candidates, remove duplicates, and resolve conflicting predictions. For the retained instances, we keep the bounding boxes produced by the localization models. These boxes serve as detection supervision and also define the target instances used in the subsequent segmentation annotation.

## 2.3 Referring Annotation

Referring annotation requires the predicted regions to satisfy the compositional meaning of an expression, including the conditions that distinguish the intended targets within a scene. We build this stage on the SAM3 Agent framework [5] and add Rex-Omni [41] as an additional localization expert. Within the same expression-conditioned workflow, SAM3 and Rex-Omni both propose candidate regions, providing complementary localization evidence for the same expression.

Qwen3-VL then consolidates these proposals by checking whether each one matches the expression in its image context. The check covers the attributes, states, actions, and relations stated in the expression, and distinguishes the requested targets from objects that are mentioned only as contextual references. Whether an expression refers to a whole object or one of its parts is likewise determined by the expression itself. Conflicting proposals are resolved through the same semantic check. The accepted results bind each expression to an individual target or a target set, so that the annotation preserves the selection constraints of the query.

## 2.4 Mask Annotation

For each localized instance, we use SAM2 [85] to predict a binary mask of its visible region. Each mask inherits the query of the box it is generated from. Each instance thus has both a box and a mask under the same query. With category queries, the boxes supervise detection and the masks supervise category-conditioned segmentation. With referring expressions, the boxes supervise referring localization and the masks supervise referring segmentation.

We manually spot-check a subset of these annotations, checking whether each mask and box are linked to the correct instance and whether the masks are accurate. Defects found in this check are corrected through the refinement process in Section 2.7. We also record the source of every annotation, so that afected labels can be traced and regenerated.

## 2.5 Hard-Example Construction

Some distinctions may not be adequately represented by broad semantic coverage alone, and we construct hard examples to focus on them. Candidate queries are derived from confusable concepts in the knowledge graph, visually similar categories, controlled changes to descriptive conditions, and observed model failures. These sources provide plausible alternatives that can only be handled by precisely interpreting both the category and the contextual constraints of a query.

A query is negative only when no object in the image satisfies all of its conditions, and this cannot be decided from the query or the detector output alone. For example, changing “red car” to “blue car” does not mean the image contains no blue car, and a detector that returns nothing for a query may simply have missed the target. Therefore, we review each candidate negative against the whole image, checking the attributes, actions, and relations in the query. If a matching target turns out to exist, the query is sent back to positive annotation.

Whether an example is hard is determined by the behavior of the current model. A reviewed negative becomes a hard negative when the model nevertheless predicts a target region for it. Such examples carry an empty target set, so they teach the model to reject the query without introducing any box or pixel targets. We retain the erroneous predictions as evidence of the underlying confusion. Hard positives, in turn, capture missed targets, incorrect referent selection, and dificult localization under crowding or occlusion. We keep their reliable annotations, so that these examples strengthen the model’s sensitivity to true targets while hard negatives teach it to reject. Sample dificulty is reassessed for each model version, and the sampling emphasis is adjusted toward persistent and newly emerging failure modes.

## 2.6 Matting Data Engine

We build a dedicated matting pipeline within WDE to provide WeVisOne-VL with alpha supervision across the diverse objects and scenes in our instance corpus. On the one hand, high-quality matting annotations are expensive to acquire, which limits their scale and coverage. Compositing foregrounds onto new backgrounds can enlarge the training set, but mismatched lighting and unnatural boundary transitions limit how well models trained on these images generalize to real scenes [56, 116]. On the other hand, our instance annotations already identify each target and its spatial extent, but their binary masks lack the continuous alpha values needed for detailed boundary supervision [116]. Extending them to matting therefore requires recovering fine details while preserving the target specified by each description, even when the reference mask has a coarse or inaccurate boundary. To this end, we combine expert-assisted annotation, dedicated annotator training, and corpus re-annotation, producing alpha labels that remain associated with the original instance descriptions and bounding boxes.

Expert-assisted Annotation. For each instance with binary mask S, we obtain two alpha candidates on the same crop: the soft foreground mask $\alpha ^ { B }$ from BiRefNet [125] and the alpha matte $\alpha ^ { M }$ from MatAnyone2 [116]. We initialize MatAnyone2 with S after dilation and erosion, repeat the crop as a short frame sequence, and take the final alpha prediction after warmup. BiRefNet, in contrast, predicts foreground from the RGB crop without any instance prompt, so its output may correspond to a diferent object or cover only part of the intended target. We therefore use its binarized foreground $B = 1 [ \alpha ^ { B } \geq \tau _ { \alpha } ]$ only to check spatial consistency with S. Specifically, we set $\tau _ { \alpha } = 0 . 5$ and accept a BiRefNet candidate only if all of the following criteria hold: $| B | \ge 8$ , IoU $\operatorname { J } ( B , S ) \ge 0 . 5$ , coverage $| B \cap S | / | S | \ge 0 . 5$ , foreground precision $| B \cap S | / | B | \geq 0 . 8 .$ area ratio $| B | / | S | \in [ 0 . 5 , 2 ]$ , bounding-box IoU of at least $0 . 7 .$ , and a centroid ofset no greater than 0.15 times the box diagonal of S. For an accepted candidate, we further retain only the connected components of $\alpha ^ { B } \geq 0 . 0 1$ that have at least 10% of their area overlapping a slightly dilated S. The alpha values within these components are preserved without clipping to $S ,$ allowing fine structures to extend beyond the coarse boundary. If the BiRefNet candidate is rejected, or fewer than eight foreground pixels remain at threshold $\tau _ { \alpha }$ after filtering, we fall back to $\alpha ^ { M }$ when IoU $( 1 [ \alpha ^ { M } \ge \tau _ { \alpha } ] , S ) \ge 0 . 9 ;$ otherwise, we retain S. We record the selected source for each instance, together with any MatAnyone2 fallback to $S$ caused by small crops or inference failures, so that model predictions can be distinguished from binary fallbacks.

Matting Annotator Training. We train a dedicated matting annotator by sampling instances with initial labels and high-quality matting data in equal proportions, combining broad instance coverage with detailed alpha supervision. Inspired by SAM2Matting [91], we adopt a frozen segmentation backbone with a trainable matting head. Specifically, we augment a frozen SAM2.1-Base+ model [85] with a new lightweight alpha refinement head. The head fuses multi-scale image features, prompt-mask guidance injected at each decoding stage, and local detail features from a shallow RGB stem, using top-down decoding to directly predict an alpha matte. During training, prompt masks are obtained by binarizing the target alpha, taking the instance segmentation mask, randomly perturbing these masks, or segmenting the target with box-prompted SAM2. The objective combines multi-scale $L _ { 1 }$ and Laplacian alpha losses with foreground/background supervision and a prompt-consistency term applied only to unperturbed object-level prompts. Additional transition region losses are applied to samples with continuous alpha labels, excluding small targets.

Corpus Re-annotation. Finally, we apply the trained annotator to all annotated instances in the training corpus, using the existing instance annotations to specify each target. Its predictions replace the initial labels and serve as the final alpha pseudo-labels for the matting training described in Section 4. Since each matte remains linked to its original instance description and bounding box, language grounding and matting are supervised on the same target.

## 2.7 Training Feedback and Data Refinement

After each training round, we analyze model errors to guide both data acquisition and annotation refinement. We organize the errors by concept, image domain, target scale, occlusion, task, and failure type, and apply clustering and deduplication to select representative cases for review. This concentrates annotation efort on recurring failures and coverage gaps.

The review distinguishes errors in model prediction from errors in supervision. If the annotation is reliable, we keep the example unchanged; it may receive greater sampling emphasis or serve as a retrieval seed for related images. If the review identifies an annotation defect, we correct or regenerate the afected supervision before including it in a subsequent data version. In other words, model disagreement only triggers an investigation, and annotations are changed only after review.

Diferent failure modes lead to diferent updates. Missed targets guide additional acquisition and hardpositive sampling. Referent confusion motivates instance contrasts and more descriptive queries, while false positive responses contribute to hard-negative construction. Defects in boxes, masks, or cross-task instance associations are returned to the corresponding annotation stage for correction.

Each data release records the annotation provenance, the model version that exposed each error, and the associated review and revision history. Unresolved cases are excluded from training. Original images, their identified near duplicates, and the derived task records stay within the same data partition across updates. Reviewed feedback data can enter subsequent training versions, whereas a fixed monitoring set is reserved for internal regression and excluded from both training and threshold calibration. Formal benchmarks are kept isolated from training and from feedback-driven data construction, so that evaluation remains independent as WDE evolves.

## 3 Model Design

WeVisOne-VL is built upon Qwen3-VL [1] with two key extensions. (1) Task formulation: it introduces dedicated tokens for normalized coordinates and task-specific tags, yielding a compact and eficient output format. (2) Model architecture: it incorporates a lightweight decoder that transforms discrete token representations into dense predictions for segmentation and matting. Below, we provide the details of these two designs.

## 3.1 Task Formulation

We formulate all visual tasks as structured next-token prediction. Inspired by Rex-Omni [41] and LocateAnything [104], we introduce a group of dedicated tokens from <0> to <1000> to represent normalized coordinates, enabling higher token eficiency. As shown in Figure 3, WeVisOne-VL localizes queried targets by sequentially generating object references and spatial coordinates. For tasks requiring bounding boxes, we encode the spatial coordinates as [x<sub>0</sub>, y<sub>0</sub>, x<sub>1</sub>, y<sub>1</sub>]. For tasks requiring point outputs, we use [x<sub>0</sub>, y<sub>0</sub>] as the coordinates. If a queried target is absent from the image, its coordinate field is filled by None. For pixel-level tasks such as segmentation and matting, WeVisOne-VL additionally generates an <OBJ> token immediately after each box-coordinate quartet, inside the same <box> field, to provide the target representation for pixel decoding.

Figure 3 Representative input–output examples for the unified task formats used by WeVisOne-VL.
<table><tr><td></td><td>Input</td><td>Locate all the instances that match the following description: person.</td></tr><tr><td rowspan="2">Detection</td><td>Output</td><td>&lt;ref&gt;person&lt;/ref&gt;&lt;box&gt;&lt;82&gt;&lt;105&gt;&lt;301&gt;&lt;947&gt;...,&lt;536&gt;&lt;118&gt;&lt;842&gt;&lt;947&gt;&lt;/b0x&gt;</td></tr><tr><td>Input</td><td>Locate the instances that best match the following description: the woman holding a red umbrella.</td></tr><tr><td rowspan="2">Referring</td><td>Output</td><td>&lt;ref&gt;the woman holding a red umbrella&lt;/ref&gt;&lt;box&gt;&lt;536&gt;&lt;118&gt;&lt;842&gt;&lt;947&gt;&lt;/box&gt;</td></tr><tr><td>Input</td><td>Point to traffic lights.</td></tr><tr><td>Pointing</td><td>Output</td><td>&lt;ref&gt;traffic lights&lt;/ref&gt;&lt;point&gt;&lt;191&gt;&lt;526&gt;....,&lt;689&gt;&lt;533&gt;&lt;/point&gt;</td></tr><tr><td rowspan="2">山 Counting</td><td>Input</td><td>How many cars in the image? Please count them with bounding boxes.</td></tr><tr><td>Output</td><td>&lt;ref&gt;cars&lt;/ref&gt;&lt;box&gt;&lt;76&gt;&lt;511&gt;&lt;292&gt;&lt;811&gt;,...,&lt;676&gt;&lt;502&gt;&lt;914&gt;&lt;823&gt;&lt;/box&gt;Tota| number: 3.</td></tr><tr><td>T</td><td>Input</td><td>Detect all the text in the image and recognize it.</td></tr><tr><td rowspan="2">OCR</td><td>Output</td><td>&lt;ref&gt;Hello&lt;/ref&gt;&lt;box&gt;&lt;93&gt;&lt;118&gt;&lt;286&gt;&lt;231&gt;&lt;ref&gt;Rache|&lt;/ref&gt;&lt;box&gt;&lt;93&gt;&lt;118&gt;&lt;286&gt;&lt;231&gt;...</td></tr><tr><td></td><td></td></tr><tr><td rowspan="2">Segmentation / Matting</td><td>Input</td><td>Please provide the alpha mattes for the instances based on the following description: dogs</td></tr><tr><td>Output</td><td>&lt;ref&gt;dogs&lt;/ref&gt;&lt;b0x&gt;&lt;81&gt;&lt;327&gt;&lt;394&gt;&lt;746&gt;&lt;0BJ&gt;,&lt;552&gt;&lt;284&gt;&lt;902&gt;&lt;721&gt;&lt;0BJ&gt;&lt;/b0x&gt;</td></tr></table>

## 3.2 Model Architecture

Given an image and an instruction, WeVisOne-VL first encodes the image with the Qwen3-VL vision encoder and autoregressively generates the structured response described above. For pixel-level tasks, it invokes a lightweight decoder for each generated <OBJ> token. The decoder takes the token’s hidden state, its predicted box, and multi-level features from the Qwen3-VL vision encoder. The decoder draws image context from the shared visual features and spatial guidance from the predicted boxes, eliminating the need for a separate large segmentation model such as SAM [45].

![](images/8f95bcbd60929aefa5152be3dbd7af22c4396ff8e9b0a35be721944e98e2a237.jpg)  
Figure 4 Overview of the lightweight pixel decoder. Qwen3-VL provides multi-level ViT features, the hidden state of each generated <OBJ> token, and its predicted box. The decoder builds shared visual features, updates a box-guided target query, and converts the two representations into a full-resolution mask or alpha matte.

The decoder consists of three components: a visual feature adapter that organizes intermediate ViT features into a shared multi-level memory, a deformable query decoder that combines each <OBJ> representation with its box prior, and an instance-conditioned dense prediction stage that converts the updated query and visual features into a full-image prediction. Segmentation and matting share these components, while matting uses an additional lightweight refinement module for fine boundaries and fractional opacity. The entire decoder, including the refinement module, contains about 9.6M parameters.

## 3.2.1 Visual Feature Adapter

To preserve fine-grained visual information for dense prediction, we extract L pre-merging features $\{ F _ { l } \in$ $\mathbb { R } ^ { C \times H _ { p } \times W _ { p } } \} _ { l = } ^ { L }$ from the intermediate ViT layers, serving as multi-level visual features. For each level l, the features are mapped to a d-dimensional decoder space and combined with positional and level information:

$$
X _ { l } ^ { ( 0 ) } = W _ { l } F _ { l } , \qquad E _ { l } = P _ { \mathrm { 2 D } } + e _ { l } ,\tag{1}
$$

where $W _ { l } \in \mathbb { R } ^ { d \times C }$ is a learnable projection, $P _ { \mathrm { 2 D } } \in \mathbb { R } ^ { d \times H _ { p } \times W _ { p } }$ is the 2D positional encoding, and $e _ { l } \in \mathbb { R } ^ { d }$ is a learnable level embedding. The visual feature adapter is composed of T blocks. Given the input $X _ { l } ^ { ( t ) }$ , the t-th block computes as follows:

$$
\begin{array} { r l } & { \bar { X } _ { l } ^ { ( t ) } = \mathrm { L a y e r N o r m } \Bigl ( X _ { l } ^ { ( t ) } + \mathrm { P W C o n v } _ { t } \Bigl ( \mathrm { G E L U } \Bigl ( \mathrm { D W C o n v } _ { t } ( X _ { l } ^ { ( t ) } + E _ { l } ) \Bigr ) \Bigr ) \Bigr ) , } \\ & { \qquad X _ { l } ^ { ( t + 1 ) } = \mathrm { L a y e r N o r m } \Bigl ( \bar { X } _ { l } ^ { ( t ) } + \mathrm { F F N } _ { t } ( \bar { X } _ { l } ^ { ( t ) } ) \Bigr ) . } \end{array}\tag{2}
$$

The final outputs $\{ X _ { l } ^ { ( T ) } \} _ { l = 1 } ^ { L }$ across diferent levels are then flattened along the spatial dimensions and concatenated to derive the visual memory $\mathcal { M } \in \mathbb { R } ^ { L H _ { p } W _ { p } \times d }$

## 3.2.2 Box-guided Query Decoder

Our query decoder follows the design of Deformable DETR [129], but difers in the initialization of instance queries: instead of using static learned embeddings, our queries are instantiated directly from the VLM outputs. Consider a response containing N predicted instances, each represented by a bounding box followed by an <OBJ> token. For the i-th instance, let $h _ { i }$ be the hidden state of its <OBJ> token and $b _ { i } = [ x _ { i 1 } , y _ { i 1 } , x _ { i 2 } , y _ { i 2 } ]$ its normalized box. We initialize the content query as $q _ { i } = W _ { h } h _ { i }$ and the positional query as $p _ { i } = \mathrm { M L P } ( \phi ( b _ { i } ) )$ , where $\phi$ denotes Fourier encoding.

Our decoder consists of six deformable blocks, each containing a self-attention module, a multi-scale de formable cross-attention module, and a feed-forward network. In the self-attention module, the N instance queries interact with one another, where $q _ { i } + p _ { i }$ serves as both the query and the key, and $q _ { i }$ serves as the value. In cross-attention, each instance query aggregates visual features from the memory M through multi-scale deformable attention conditioned on its reference box $\begin{array} { r } { r _ { i } = \big [ \frac { x _ { i 1 } + x _ { i 2 } } { 2 } , \frac { y _ { i 1 } + y _ { i 2 } } { 2 } , x _ { i 2 } - x _ { i 1 } , y _ { i 2 } - y _ { i 1 } \big ] } \end{array}$

## 3.2.3 Instance-Conditioned Dense Prediction

Given the multi-level visual features from the visual feature adapter and the query embeddings from the box-guided query decoder, this stage fuses the visual features, predicts an initial mask for each instance, and refines it across multiple scales.

Multi-level Pixel Fusion. Deep features encode high-level semantic information, while shallow features preserve fine-grained spatial details. To combine visual information from diferent levels, we first fuse the adapted visual features $\{ X _ { l } ^ { ( T ) } \} _ { l = 1 } ^ { L }$ from top to bottom:

$$
G _ { l } = \left\{ \begin{array} { l l } { \mathrm { C o n v } _ { 1 \times 1 } \left( X _ { L } ^ { ( T ) } \right) , } & { l = L , } \\ { \mathrm { G E L U } \Big ( \mathrm { G r o u p N o r m } \Big ( \mathrm { C o n v } _ { 3 \times 3 } \Big ( G _ { l + 1 } + \mathrm { C o n v } _ { 1 \times 1 } \left( X _ { l } ^ { ( T ) } \right) \Big ) \Big ) \Big ) , } & { l = L - 1 , . . . , 1 . } \end{array} \right.\tag{3}
$$

To recover color, texture, and edge details that may be lost in the fused visual feature, we additionally resize the input image I to $1 0 2 4 \times 1 0 2 4$ and feed it into a two-layer CNN to extract a shallow feature map. Then, we upsample $G _ { 1 }$ to the same resolution $5 1 2 \times 5 1 2$ and combine the two features by element-wise addition:

$$
P = \mathrm { U p } ( G _ { 1 } ) + \mathrm { C N N } ( I ) .\tag{4}
$$

Dynamic Mask Head. Let $\widetilde { q } _ { i }$ denote the query representation of the i-th instance derived from the box-guided equery decoder. We first apply an MLP to $\widetilde { q } _ { i }$ to generate an instance-aware dynamic kernel, which is then applied to $P$ to produce pixel-wise logits:

$$
\begin{array} { r } { [ k _ { i } , \beta _ { i } ] = \mathrm { M L P } ( \widetilde { q } _ { i } ) , \qquad M _ { i } = k _ { i } ^ { \top } P + \beta _ { i } . } \end{array}\tag{5}
$$

Coarse-to-fine Refinement. A single mask prediction is often insuficient for thin structures and translucent boundaries. Starting from $M _ { i } ^ { ( 0 ) } = M _ { i }$ , we progressively refine the mask through three upsampling stages $( j = 1 , 2 , 3 )$ . At stage j, we upsample the previous stage feature $F _ { i } ^ { ( j - 1 ) }$ and concatenate it with the resized image I, pixel feature P, and previous mask predictions $C _ { i } ^ { ( j ) }$

$$
U _ { i } ^ { ( j ) } = \mathrm { C o n v } _ { 1 \times 1 } \Big ( \mathrm { U p } ( F _ { i } ^ { ( j - 1 ) } ) \parallel \mathrm { I n t e r p } _ { j } ( I ) \parallel \mathrm { I n t e r p } _ { j } ( P ) \parallel C _ { i } ^ { ( j ) } \Big ) ,\tag{6}
$$

where Up denotes transposed convolution, Interp denotes bilinear interpolation to the resolution of stage $j ,$ and ∥ denotes channel concatenation. We then modulate $U _ { i } ^ { ( j ) }$ with the instance query $\widetilde { q } _ { i }$ via channel-wise scale and shift, and pass it through residual convolutional blocks $f _ { \mathrm { r c } } ^ { ( j ) }$ eto predict a logit residual:

$$
[ \gamma _ { i } ^ { ( j ) } , \delta _ { i } ^ { ( j ) } ] = \mathrm { M L P } _ { j } ( \widetilde { q } _ { i } ) , \qquad F _ { i } ^ { ( j ) } = f _ { \mathrm { r c } } ^ { ( j ) } \left( \big ( 1 + \operatorname { t a n h } ( \gamma _ { i } ^ { ( j ) } ) \big ) \odot U _ { i } ^ { ( j ) } + \delta _ { i } ^ { ( j ) } \right) ,\tag{7}
$$

$$
M _ { i } ^ { ( j ) } = \mathrm { { I n t e r p } } _ { j } \left( M _ { i } ^ { ( j - 1 ) } \right) + \mathrm { { C o n v } } _ { 1 \times 1 } \left( F _ { i } ^ { ( j ) } \right) .\tag{8}
$$

Finally, the output prediction is obtained by $A _ { i } = \mathrm { S i g m o i d } ( M _ { i } ^ { ( 3 ) } )$

Table 1 Training data composition. The SFT and RL rings show task distributions. The segmentation warm-up and matting fine-tuning rings show data-source proportions: Public vs. WDE (WeVisOne-VL Data Engine). Center numbers indicate training sample counts.

![](images/8a40e241b4a22205d5ef464ef93f3685cc35306407372f3da4955efe79e63843.jpg)

![](images/2686f34c730fcbd9d67f4368cba82ff251ccef46d360e65e26d4e915ccade11d.jpg)

## 4 Training Pipeline

We train WeVisOne-VL with a three-stage Supervised Fine-Tuning (SFT) curriculum followed by mixed-task Reinforcement Learning (RL). The curriculum first establishes a shared output interface, then improves high resolution and dense perception, and finally connects autoregressive localization with pixel-level prediction. After SFT, RL directly optimizes the quality of the complete prediction rather than the likelihood of individual output tokens. The data composition across the training pipeline is summarized in Table 1.

## 4.1 Curriculum Supervised Fine-Tuning

Stage 1: Task Alignment. This stage adapts the model to the new tokens and task-specific structured outputs. Starting from Qwen3-VL with an extended vocabulary, we freeze the vision encoder and visual-language projector and train only the language model on 90M examples covering all target tasks for 13K optimization steps. The model learns to follow diferent task instructions, produce the structured outputs, and return None when the target is absent. For pixel-task examples, we impose only next-token prediction supervision (NTP) without activating the pixel decoder, thereby establishing an initial correspondence between each predicted instance and its <OBJ> token. In this stage, we use 16–512 visual tokens per image and limit the context length to 4,096 tokens. Notably, early training can be dominated by extremely dense samples (e.g., containing hundreds of boxes in a single response), making the model highly prone to repetitive output collapse. To mitigate this issue, we primarily use samples with at most 50 boxes or points per category during this stage.

Stage 2: Resolution and Density Scaling. Building on Stage 1, we unfreeze the vision encoder, projector, and language model. This stage serves two purposes. First, we raise the maximum input resolution to 1,960 visual tokens, preserving the visual details for recognizing small objects. Second, we increase the maximum box/point number within one response to 500, exposing the model to crowded scenes for dense detection. The maximum context length is extended to 8,192 tokens. We train on approximately 40M examples for 9K optimization steps. During this stage, pixel-level examples remain supervised solely by NTP.

Stage 3: Alternating Token–Pixel Optimization. This stage aims to equip the VLM with pixel-level perception by establishing the <OBJ> token as the bridge between the VLM and the lightweight pixel decoder. To initialize the pixel decoder before alternating token–pixel training, we first jointly train the SFT-S2 VLM and the randomly initialized decoder on 250M segmentation samples for 100K steps with the following segmentation loss function:

$$
\mathcal { L } _ { \mathrm { s e g } } = 2 0 \mathcal { L } _ { \mathrm { f o c a l } } + \mathcal { L } _ { \mathrm { d i c e } } ,\tag{9}
$$

where the focal term emphasizes hard pixels under foreground–background imbalance, while the Dice term directly optimizes mask overlap. After warm-up, we retain only the decoder weights and the learned <OBJ> token embedding. We attach them to the original Stage 2 checkpoint and jointly train the VLM and decoder for 9,000 steps. To preserve localization ability, we alternate NTP and pixel-level tasks at a 1:1 ratio, which

can be expressed as the following loss function:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { j o i n t } } = \mathcal { L } _ { \mathrm { N T P } } + 1 0 \mathcal { L } _ { \mathrm { s e g } } . } \end{array}\tag{10}
$$

After the alternating training, we specialize the pixel decoder for continuous alpha prediction while keeping the VLM frozen. We use 130M matting samples to further finetune the decoder for 100K steps. Following standard practice in image matting [29], we combine $L _ { 1 }$ , gradient, and Laplacian losses to supervise alpha estimation, and additionally employ $\mathcal { L } _ { \mathrm { h a r d } }$ to emphasize dificult foreground and background pixels:

$$
\mathcal { L } _ { \mathrm { m a t } } = 2 0 0 \mathcal { L } _ { l 1 } + 5 \mathcal { L } _ { \mathrm { g r a d } } + 5 0 \mathcal { L } _ { \mathrm { l a p } } + \mathcal { L } _ { \mathrm { h a r d } } .\tag{11}
$$

## 4.2 Mixed-Task Reinforcement Learning

Although SFT provides a strong initialization, its token-level likelihood objective cannot directly optimize task rewards. We therefore perform mixed-task reinforcement learning on an uncertainty-filtered corpus.

Uncertainty-Aware Sample Selection. GRPO requires relative advantages to provide learning signals. Before RL training, we therefore use the SFT model for data sampling. For each candidate prompt x, we sample a group of responses and compute their corresponding task rewards. Let $\mu _ { x }$ and $\sigma _ { x }$ denote the mean and standard deviation of the group rewards, respectively. The uncertainty score for each sample is defined as follows:

$$
\begin{array} { r } { U ( x ) = 4 \mu _ { x } ( 1 - \mu _ { x } ) \sigma _ { x } . } \end{array}\tag{12}
$$

This score favors moderately dificult samples with high reward variance, on which the model succeeds in some rollouts but fails in others. Such samples have high uncertainty and yield stronger relative-advantage signals, thereby improving the eficiency of RL training. Using this score, we select approximately 140K samples for RL training. We supplement this mixture with targeted box-supervised examples and pointing data, yielding 178K examples in total. Table 1 shows the complete task distribution. Data selection and reward details are provided in Appendices A.3 and B.3, respectively.

GRPO with Grounding Rewards. On the resulting training corpus, we perform GRPO using grounding rewards computed over complete structured responses. These rewards directly evaluate the grounding quality of the model predictions rather than individual tokens. For each selected image–prompt pair x, we sample $G = 8$ complete responses $\{ o _ { i } \} _ { i = 1 } ^ { G }$ from the current policy and parse each response into a set of predicted objects. Let $\hat { B _ { i } }$ and $B ^ { * }$ denote the predicted and ground-truth box sets, respectively. We perform one-to-one Hungarian matching between predicted and ground-truth boxes within each semantic class, and denote the set of matched pairs as $\mathcal { M } _ { i }$ . The grounding reward for response $o _ { i }$ is defined as:

$$
r _ { i } = \frac { 2 \sum _ { ( \hat { b } , b ^ { * } ) \in \mathcal { M } _ { i } } \mathrm { I o U } ( \hat { b } , b ^ { * } ) ^ { 3 } } { N _ { \mathrm { g t } } + N _ { \mathrm { p r e d } } } .\tag{13}
$$

This reward measures localization quality, while penalizing both missed and duplicate objects. Cubing the IoU makes the reward more sensitive to localization improvements at high IoU values, encouraging more precise box predictions. For segmentation and matting, RL uses only box-based rewards, without evaluating predicted masks or alpha mattes. For referring segmentation, a prediction is valid only when its box is followed by the required <OBJ> token. For negative samples, the reward is 1 only when the model predicts no object.

## 5 Experiments

We evaluate WeVisOne-VL across five object-level tasks (object detection, referring object detection, object counting, object pointing, and OCR) and two pixel-level tasks (referring segmentation and image matting). At inference, we use greedy decoding with temperature 0 and limit each image to 1,960 visual tokens. We primarily compare with the Qwen series as general-purpose VLM baselines and with two representative VLMbased grounding models, Rex-Omni [41] and LocateAnything [104]. For Qwen3-VL, we evaluate the Instruct variants; for Qwen3.5, we enable thinking mode because it performs better in our evaluation.

Table 2 Results on general object detection benchmarks COCO and LVIS, and dense object detection benchmarks Dense200 and VisDrone. COCO and LVIS are evaluated in a zero-shot setting. We report F1 at IoU=0.5, IoU=0.95, and averaged over IoU thresholds from 0.5 to 0.95 (Mean). Bold and underline denote the best and second-best results among VLMs.
<table><tr><td rowspan="2">Method</td><td colspan="3">COCO</td><td colspan="3">LVIS</td><td colspan="3">Dense200</td><td colspan="3">VisDrone</td></tr><tr><td>0.5</td><td>0.95</td><td>Mean</td><td>0.5</td><td>0.95</td><td>Mean</td><td>0.5</td><td>0.95</td><td>Mean</td><td>0.5</td><td>0.95</td><td>Mean</td></tr><tr><td>Specialized Detectors</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GroundingDINO-T [67]</td><td>69.8</td><td>23.0</td><td>56.6</td><td>47.7</td><td>22.7</td><td>38.8</td><td>36.9</td><td>19.7</td><td>33.1</td><td>55.2</td><td>3.9</td><td>38.5</td></tr><tr><td colspan="9">Vision-Language Models</td><td></td><td></td><td></td></tr><tr><td>Qwen3-VL-2B [1]</td><td>62.9</td><td>13.1</td><td>45.2</td><td>58.5</td><td>17.8</td><td>40.8</td><td>14.7</td><td>1.8</td><td>10.0</td><td>37.0</td><td>1.5</td><td>23.1</td></tr><tr><td>Qwen3-VL-4B [1]</td><td>66.6</td><td>14.9</td><td>48.7</td><td>61.3</td><td>19.5</td><td>44.0</td><td>20.9</td><td>3.7</td><td>15.2</td><td>49.5</td><td>1.8</td><td>31.0</td></tr><tr><td>Qwen3-VL-8B [1]</td><td>65.9</td><td>14.8</td><td>47.8</td><td>59.7</td><td>19.6</td><td>42.9</td><td>19.5</td><td>2.9</td><td>13.8</td><td>47.7</td><td>2.0</td><td>29.5</td></tr><tr><td>Qwen3.5-4B [81]</td><td>67.6</td><td>12.9</td><td>48.2</td><td>61.1</td><td>17.8</td><td>42.2</td><td>28.4</td><td>2.3</td><td>18.9</td><td>50.3</td><td>1.3</td><td>29.9</td></tr><tr><td>Qwen3.5-9B [81]</td><td>68.9</td><td>13.8</td><td>49.5</td><td>62.2</td><td>19.4</td><td>43.7</td><td>26.2</td><td>2.5</td><td>18.0</td><td>51.0</td><td>1.4</td><td>31.3</td></tr><tr><td>Rex-Omni-3B [41]</td><td>72.0</td><td>15.9</td><td>52.9</td><td>64.3</td><td>20.7</td><td>46.9</td><td>78.4</td><td>10.3</td><td>58.3</td><td>61.6</td><td>1.5</td><td>35.8</td></tr><tr><td>LocateAnything-3B [104]</td><td>70.1</td><td>19.3</td><td>54.7</td><td>62.3</td><td>31.1</td><td>50.7</td><td>74.0</td><td>18.5</td><td>58.7</td><td>63.0</td><td>3.2</td><td>39.9</td></tr><tr><td>WeVisOne-VL-2B</td><td>72.6</td><td>18.7</td><td>55.2</td><td>69.5</td><td>27.5</td><td>53.3</td><td>79.1</td><td>14.1</td><td>61.4</td><td>64.5</td><td>2.8</td><td>40.4</td></tr><tr><td>WeVisOne-VL-4B</td><td>74.0</td><td>19.5</td><td>57.1</td><td>71.1</td><td>28.8</td><td>55.3</td><td>86.9</td><td>23.2</td><td>71.3</td><td>68.1</td><td>3.9</td><td>44.7</td></tr></table>

## 5.1 Main Results

## 5.1.1 Object Detection

Benchmarks. We evaluate common object detection on COCO [63], long-tailed object detection on LVIS [30], and dense and tiny-object detection on Dense200 [41] and VisDrone [19].

Metrics. Average Precision (AP) relies on calibrated confidence scores and is therefore not directly applicable to generative VLMs. Following Rex-Omni [41] and LocateAnything [104], we report F1 scores. A predicted box is regarded as a true positive if its IoU with an unmatched ground-truth box of the same category exceeds a threshold. Precision and recall are computed per category and averaged, and F1 is their harmonic mean. We report F1 at IoU=0.5, at IoU=0.95, and averaged over IoU thresholds from 0.5 to 0.95 with a step of 0.05 (denoted as “Mean”).

Results. As shown in Table 2, WeVisOne-VL consistently outperforms VLMs at comparable model scales in mean F1 across general, long-tailed, and dense detection benchmarks. The gains are especially pronounced on Dense200 and VisDrone, where accurate recognition must be maintained over many small and crowded instances. Its strong performance under the strict IoU = 0.95 criterion further indicates that the improvement reflects both broad object coverage and precise localization.

## 5.1.2 Referring Object Detection

Benchmarks. We evaluate referring object detection on the public RefCOCOg [72], HumanRef [39], Ref-Adv [17], PR-Bench [22], DVGBench [127], and Ref-L4 [6] benchmarks. We also include RefCOCOg-Rej [58], in which each modified expression has no matching object in the image. Furthermore, we introduce WeRef-Bench, an in-house benchmark comprising 2,880 English and Chinese queries. It covers single-target referring, multi-target referring, and absent-target queries, providing more challenging and practical evaluations than the public positive, single-target benchmarks.

Metrics. For most benchmarks, we report F1 Mean as the primary metric. For RefCOCOg-Rej, we report the Rejection Rate, where a prediction is correct only when no box is produced. For WeRef-Bench and PR-Bench, we report both F1 Mean and Rejection Rate to evaluate target-present and absent-target queries, respectively.

Results. Table 3 shows that WeVisOne-VL consistently outperforms existing VLMs across single-target, multi target, and absent-target settings. These results demonstrate that WeVisOne-VL localizes referred objects

Table 3 Results on referring object detection. Entries are mean F1 over IoU thresholds from 0.5 to 0.95. Rej denotes the rejection rate (%) on absent-target queries; the RefCOCOg Rej is evaluated on RefCOCOg-Rej. For Ref-Adv, we use the Ref-Adv-s split. Bold and underline denote the best and second-best results among VLMs.
<table><tr><td rowspan="2">Method</td><td colspan="3">RefCOCOg</td><td colspan="2">WeRef-Bench</td><td colspan="2">PR-Bench</td><td rowspan="2">HumanRef</td><td rowspan="2">Ref-Adv</td><td rowspan="2">DVG</td><td rowspan="2">Ref-L4</td></tr><tr><td>val</td><td>test</td><td>Rej</td><td>Mean</td><td>Rej</td><td>Mean</td><td>Rej</td></tr><tr><td colspan="10">Specialized Detectors</td></tr><tr><td>GroundingDINO-T [67]</td><td>45.9</td><td>46.8</td><td>0.0</td><td>21.1</td><td>14.7</td><td>25.2</td><td>0.0</td><td>25.2</td><td>19.7</td><td>24.1</td><td>23.1</td></tr><tr><td colspan="10">Vision-Language Models</td></tr><tr><td>Qwen3-VL-2B [1]</td><td>70.8</td><td>71.0</td><td>1.7</td><td>30.8</td><td>22.8</td><td>41.6</td><td>4.3</td><td>63.2</td><td>34.7</td><td>35.2</td><td>67.9</td></tr><tr><td>Qwen3-VL-4B [1]</td><td>74.1</td><td>74.3</td><td>18.8</td><td>46.3</td><td>61.9</td><td>49.9</td><td>10.8</td><td>69.4</td><td>44.4</td><td>42.4</td><td>72.5</td></tr><tr><td>Qwen3-VL-8B [1]</td><td>74.6</td><td>74.9</td><td>22.5</td><td>36.7</td><td>18.3</td><td>50.3</td><td>7.1</td><td>71.6</td><td>45.0</td><td>43.4</td><td>72.4</td></tr><tr><td>Qwen3.5-4B [81]</td><td>73.3</td><td>73.5</td><td>18.9</td><td>48.1</td><td>71.6</td><td>49.6</td><td>8.7</td><td>75.8</td><td>44.2</td><td>40.2</td><td>70.4</td></tr><tr><td>Qwen3.5-9B [81]</td><td>74.3</td><td>74.6</td><td>15.2</td><td>50.7</td><td>70.7</td><td>50.8</td><td>8.3</td><td>77.8</td><td>47.3</td><td>42.5</td><td>72.0</td></tr><tr><td>Rex-Omni-3B [41]</td><td>73.6</td><td>74.3</td><td>0.5</td><td>50.3</td><td>43.3</td><td>42.7</td><td>0.2</td><td>79.9</td><td>40.9</td><td>33.2</td><td>65.2</td></tr><tr><td>LocateAnything-3B [104]</td><td>76.7</td><td>77.6</td><td>0.6</td><td>51.7</td><td>73.0</td><td>44.5</td><td>0.2</td><td>78.7</td><td>43.5</td><td>34.7</td><td>74.3</td></tr><tr><td>WeVisOne-VL-2B</td><td>77.1</td><td>78.2</td><td>50.2</td><td>73.2</td><td>92.9</td><td>49.8</td><td>37.0</td><td>77.4</td><td>43.6</td><td>40.8</td><td>75.1</td></tr><tr><td>WeVisOne-VL-4B</td><td>80.8</td><td>81.8</td><td>67.4</td><td>78.7</td><td>96.4</td><td>54.1</td><td>52.6</td><td>80.7</td><td>50.5</td><td>45.2</td><td>78.1</td></tr></table>

Table 4 Object-counting exact-match accuracy (%, ). Bold and underline denote the best and second-best results among VLM-based models, respectively; the specialized detector is excluded from this ranking.
<table><tr><td rowspan="2">Method</td><td rowspan="2">CountQA</td><td colspan="2">PixMo-Count</td><td colspan="2">CLOC</td><td colspan="2">KubriCount</td><td colspan="2">MixCount</td></tr><tr><td>val</td><td>test</td><td>val</td><td>test</td><td>testA</td><td>testB</td><td>val</td><td>test</td></tr><tr><td colspan="9">Specialized Detectors</td></tr><tr><td>Grounding DINO-T [67]</td><td>9.6</td><td>44.8</td><td>48.3</td><td>12.4</td><td>9.5</td><td>5.2</td><td>6.5</td><td>1.7</td><td>2.8</td></tr><tr><td colspan="9">Vision-Language Models</td></tr><tr><td>Qwen3-VL-2B [1]</td><td>20.4</td><td>61.3</td><td>60.4</td><td>16.3</td><td>15.3</td><td>4.3</td><td>7.8</td><td>1.8</td><td>2.7</td></tr><tr><td>Qwen3-VL-4B [1]</td><td>29.2</td><td>69.6</td><td>65.6</td><td>19.7</td><td>18.6</td><td>7.1</td><td>12.1</td><td>3.5</td><td>4.3</td></tr><tr><td>Qwen3-VL-8B [1]</td><td>22.4</td><td>68.2</td><td>59.4</td><td>18.5</td><td>17.2</td><td>6.6</td><td>10.8</td><td>3.2</td><td>3.5</td></tr><tr><td>Qwen3.5-4B [81]</td><td>40.0</td><td>80.6</td><td>76.9</td><td>21.1</td><td>21.7</td><td>10.0</td><td>16.2</td><td>3.9</td><td>5.2</td></tr><tr><td>Qwen3.5-9B [81]</td><td>41.6</td><td>82.6</td><td>80.4</td><td>23.1</td><td>22.4</td><td>11.9</td><td>19.1</td><td>5.8</td><td>7.0</td></tr><tr><td>Rex-Omni-3B [41]</td><td>37.6</td><td>82.2</td><td>71.9</td><td>19.0</td><td>22.3</td><td>13.5</td><td>18.8</td><td>9.9</td><td>11.4</td></tr><tr><td>LocateAnything-3B [104]</td><td>52.4</td><td>88.7</td><td>82.8</td><td>22.8</td><td>23.9</td><td>18.5</td><td>22.4</td><td>12.5</td><td>15.2</td></tr><tr><td>WeVisOne-VL-2B</td><td>47.9</td><td>90.9</td><td>88.1</td><td>27.5</td><td>26.9</td><td>29.5</td><td>32.4</td><td>30.1</td><td>32.2</td></tr><tr><td>WeVisOne-VL-4B</td><td>57.1</td><td>93.7</td><td>89.6</td><td>28.2</td><td>28.3</td><td>37.7</td><td>40.1</td><td>42.0</td><td>40.4</td></tr></table>

more accurately and rejects absent-target queries more reliably.

## 5.1.3 Object Counting

Benchmarks. We evaluate object counting on CountQA [100], PixMo-Count [13], CLOC [51], KubriCount [66], and MixCount [20]. CountQA and PixMo-Count focus on counting in natural images, while CLOC evaluates cross-domain generalization. KubriCount stresses multi-grained target specification with controlled distractors, and MixCount targets fine-grained open-vocabulary counting in mixed-object scenes.

Metrics. For models evaluated through detection output, we use the number of predicted target boxes as the count; for models that return a count directly, we extract the numerical answer from the response. We report exact-match accuracy: the percentage of queries for which the predicted count equals the ground-truth count. Thus, an error of even one object is counted as incorrect, and higher values are better.

Results. Table 4 shows that WeVisOne-VL achieves the highest exact-match accuracy on every reported bench mark and split. It reaches 93.7%/89.6% on PixMo-Count validation/test and 42.0%/40.4% on MixCount

Table 5 Results on object pointing. Bold and underline denote the best and second-best reported results, respectively.
<table><tr><td>Method</td><td>PointBench</td><td>PixMo-Points</td><td>COCO</td><td>LVIS</td><td>Dense200</td><td>VisDrone</td></tr><tr><td>Qwen3-VL-2B [1]</td><td>61.2</td><td>62.1</td><td>72.7</td><td>61.9</td><td>29.5</td><td>35.7</td></tr><tr><td>Qwen3-VL-4B [1]</td><td>57.1</td><td>71.3</td><td>74.5</td><td>65.8</td><td>37.7</td><td>45.6</td></tr><tr><td>Qwen3-VL-8B [1]</td><td>62.5</td><td>72.2</td><td>77.5</td><td>67.7</td><td>42.2</td><td>46.4</td></tr><tr><td>Qwen3.5-4B [81]</td><td>61.5</td><td>59.2</td><td>72.7</td><td>62.4</td><td>7.4</td><td>38.3</td></tr><tr><td>Qwen3.5-9B [81]</td><td>66.4</td><td>62.6</td><td>74.4</td><td>64.8</td><td>10.5</td><td>40.6</td></tr><tr><td>Rex-Omni-3B [41]</td><td>69.9</td><td>72.9</td><td>80.5</td><td>70.8</td><td>82.5</td><td>58.9</td></tr><tr><td>LocateAnything-3B [104]</td><td>56.9</td><td>68.6</td><td>83.9</td><td>76.6</td><td>87.6</td><td>60.4</td></tr><tr><td>WeVisOne-VL-2B</td><td>62.9</td><td>73.2</td><td>85.8</td><td>78.9</td><td>83.2</td><td>62.4</td></tr><tr><td>WeVisOne-VL-4B</td><td>71.9</td><td>74.7</td><td>85.4</td><td>79.5</td><td>87.0</td><td>64.4</td></tr></table>

validation/test, indicating accurate counts in both natural images and fine-grained mixed-object scenes.

## 5.1.4 Object Pointing

Benchmarks. We evaluate object pointing on two dedicated pointing benchmarks, PointBench [7] and PixMo-Points [13]. We also evaluate pointing on the detection benchmarks COCO, LVIS, Dense200, and VisDrone, following the protocol of Rex-Omni [41].

Metrics. A predicted point is correct when it falls inside the corresponding ground-truth mask (or the annotated target box where masks are unavailable). Each target contributes at most one true positive, so duplicate points reduce precision and missed targets reduce recall. We report F1@Point as the harmonic mean of precision and recall. For detection-derived benchmarks, precision and recall are first averaged across categories.

Results. As shown in Table 5, WeVisOne-VL achieves the best results on both PointBench and PixMo-Points, demonstrating its ability to point directly to target objects. On the detection-derived benchmarks, it also performs strongly across general detection benchmarks, showing that this ability extends to diverse scenes.

## 5.1.5 OCR

Benchmarks. We evaluate text detection and recognition on HierText [71], ICDAR2015 [43], TotalText [8], and SROIE [35]. These benchmarks cover dense document text, incidental scene text, curved text, and structured receipts.

Metrics. We formulate OCR as a grounding task in which the recognized transcription serves as the category label. A prediction is considered correct only when its text exactly matches the ground-truth transcription and its box satisfies the IoU threshold. We then compute precision, recall, and F1 using the same matching procedure as object detection, and report F1 at IoU=0.5, at IoU=0.95, and averaged over thresholds from 0.5 to 0.95.

Results. As shown in Table 6, WeVisOne-VL consistently outperforms both specialist OCR systems and general-purpose VLMs across all evaluated benchmarks. Its strong performance across dense documents, incidental and curved scene text, and structured receipts demonstrates robust end-to-end text perception under diverse layouts and visual conditions.

## 5.1.6 Referring Segmentation

Given an image and a natural-language expression, referring segmentation predicts a binary mask of the described objects, extending localization from box-level predictions to pixel-level dense outputs.

Benchmarks. We evaluate referring segmentation on RefCOCO and RefCOCO+ [120], RefCOCOg [72], ReasonSeg [48] and Ref-ZOM [34]. ReasonSeg uses implicit queries that require visual reasoning and world knowledge. Ref-ZOM further includes expressions that refer to zero, one, or multiple targets.

Table 6 Results on OCR (text grounding and recognition). We report F1 at IoU=0.5, IoU=0.95, and averaged over IoU thresholds from 0.5 to 0.95 (Mean). Bold and underline denote the best and second-best results among VLM-based models, respectively; the specialist OCR system is excluded from this ranking.
<table><tr><td rowspan="2">Method</td><td colspan="3">HierText</td><td colspan="3">ICDAR2015</td><td colspan="3">TotalText</td><td colspan="3">SROIE</td></tr><tr><td>0.5</td><td>0.95</td><td>Mean</td><td>0.5</td><td>0.95</td><td>Mean</td><td>0.5</td><td>0.95</td><td>Mean</td><td>0.5</td><td>0.95</td><td>Mean</td></tr><tr><td>Specialist OCR Systems</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PP-OCRv5 [12]</td><td>45.2</td><td>3.4</td><td>30.5</td><td>38.2</td><td>1.2</td><td>25.6</td><td>40.2</td><td>0.7</td><td>25.7</td><td>77.7</td><td>5.6</td><td>58.6</td></tr><tr><td colspan="10">Vision-Language Models</td><td colspan="3"></td></tr><tr><td>Qwen3-VL-2B [1]</td><td>21.4</td><td>0.3</td><td>11.0</td><td>15.7</td><td>0.0</td><td>8.5</td><td>32.6</td><td>1.1</td><td>20.5</td><td>40.2</td><td>0.2</td><td>18.6</td></tr><tr><td>Qwen3-VL-4B [1]</td><td>39.9</td><td>0.9</td><td>22.7</td><td>52.0</td><td>0.8</td><td>30.0</td><td>48.3</td><td>3.3</td><td>33.0</td><td>69.0</td><td>1.0</td><td>40.4</td></tr><tr><td>Qwen3-VL-8B [1]</td><td>45.3</td><td>0.9</td><td>25.7</td><td>45.8</td><td>0.3</td><td>27.1</td><td>53.2</td><td>3.2</td><td>35.4</td><td>75.8</td><td>1.1</td><td>43.3</td></tr><tr><td>Qwen3.5-4B [81]</td><td>14.4</td><td>0.4</td><td>8.0</td><td>17.4</td><td>0.1</td><td>7.8</td><td>28.8</td><td>0.6</td><td>16.6</td><td>14.7</td><td>0.0</td><td>7.1</td></tr><tr><td>Qwen3.5-9B [81]</td><td>17.5</td><td>0.5</td><td>10.0</td><td>28.1</td><td>0.0</td><td>13.4</td><td>49.9</td><td>1.6</td><td>30.8</td><td>31.8</td><td>0.4</td><td>16.7</td></tr><tr><td>Rex-Omni-3B [41]</td><td>45.9</td><td>1.4</td><td>28.0</td><td>45.2</td><td>0.3</td><td>28.1</td><td>56.6</td><td>3.9</td><td>40.6</td><td>72.0</td><td>1.5</td><td>44.8</td></tr><tr><td>LocateAnything-3B [104]</td><td>39.2</td><td>4.1</td><td>29.4</td><td>37.8</td><td>0.3</td><td>24.1</td><td>58.9</td><td>5.1</td><td>43.3</td><td>52.8</td><td>3.4</td><td>40.3</td></tr><tr><td>WeVisOne-VL-2B</td><td>44.3</td><td>2.5</td><td>30.5</td><td>65.3</td><td>2.0</td><td>42.6</td><td>61.5</td><td>2.7</td><td>42.5</td><td>85.3</td><td>4.4</td><td>62.1</td></tr><tr><td>WeVisOne-VL-4B</td><td>51.6</td><td>5.3</td><td>39.0</td><td>69.1</td><td>2.3</td><td>47.8</td><td>70.1</td><td>6.9</td><td>52.0</td><td>87.4</td><td>7.3</td><td>68.1</td></tr></table>

Table 7 Referring segmentation results. RefCOCO-family scores are cIoU, while ReasonSeg scores are gIoU. Ref-ZOM baseline scores are obtained using oficial checkpoints. A hyphen (-) denotes an unreported result. Bold and underline denote the best and second-best reported results in each column, respectively.
<table><tr><td rowspan="2">Method</td><td colspan="3">RefCOCO</td><td colspan="3">RefCOCO+</td><td colspan="2">RefCOCOg</td><td colspan="2">ReasonSeg</td><td colspan="2">Ref-ZOM</td></tr><tr><td>val</td><td>testA</td><td>testB</td><td>val</td><td>testA</td><td>testB</td><td>val</td><td>test</td><td>val</td><td>test</td><td>cIoU</td><td>Rej</td></tr><tr><td>LISA-13B (ft) [48]</td><td>76.0</td><td>78.8</td><td>72.9</td><td>65.0</td><td>70.2</td><td>58.1</td><td>69.5</td><td>70.5</td><td>65.0</td><td>61.3</td><td></td><td></td></tr><tr><td>GSVA-7B 3 [110]</td><td>77.2</td><td>78.9</td><td>73.5</td><td>65.9</td><td>69.6</td><td>59.8</td><td>72.7</td><td>73.3</td><td>50.5</td><td>-</td><td>66.6</td><td>41.9</td></tr><tr><td>LaSagnA-7B [106]</td><td>76.8</td><td>78.7</td><td>73.8</td><td>66.4</td><td>70.6</td><td>60.1</td><td>70.6</td><td>71.9</td><td></td><td>-</td><td>57.4</td><td>0.7</td></tr><tr><td>Text4Seg-7B [50]</td><td>79.3</td><td>81.9</td><td>76.2</td><td>72.1</td><td>77.6</td><td>66.1</td><td>72.1</td><td>73.9</td><td>一</td><td></td><td>54.5</td><td>25.9</td></tr><tr><td>VRS-HQ-7B [27]</td><td>73.5</td><td>77.5</td><td>69.5</td><td>61.7</td><td>67.6</td><td>54.3</td><td>66.7</td><td>67.5</td><td>55.2</td><td>51.7</td><td>63.0</td><td>13.9</td></tr><tr><td>UniPixel-7B [68]</td><td>80.8</td><td>83.0</td><td>77.4</td><td>75.3</td><td>80.1</td><td>70.0</td><td>76.4</td><td>77.1</td><td>60.5</td><td>-</td><td>62.9</td><td>0.3</td></tr><tr><td>Youtu-VL-4B [107]</td><td>80.7</td><td>82.0</td><td>78.4</td><td>76.2</td><td>79.6</td><td>71.4</td><td>76.5</td><td>76.6</td><td>-</td><td>-</td><td>61.0</td><td>1.3</td></tr><tr><td>WeVisOne-VL-2B</td><td>79.7</td><td>82.9</td><td>78.1</td><td>73.1</td><td>79.0</td><td>66.3</td><td>74.4</td><td>74.9</td><td>60.9</td><td>56.9</td><td>73.4</td><td>62.4</td></tr><tr><td>WeVisOne-VL-4B</td><td>81.4</td><td>84.5</td><td>80.1</td><td>76.5</td><td>81.1</td><td>72.8</td><td>77.7</td><td>77.8</td><td>71.2</td><td>66.8</td><td>72.2</td><td>86.0</td></tr></table>

Metrics. For the RefCOCO family and Ref-ZOM, we report cumulative intersection over union (cIoU) over the entire evaluation set, computed as the total intersection between predicted and ground-truth masks divided by their total union across all samples. For ReasonSeg, we report generalized IoU (gIoU) on the validation and test splits. For Ref-ZOM, which contains negative expressions that refer to no target, we additionally report the rejection rate (Rej).

Results. As shown in Table 7, WeVisOne-VL improves over all baselines across every RefCOCO-family split. It also demonstrates strong potential for reasoning segmentation even without explicit chain-of-thought (CoT) reasoning. Furthermore, the results on Ref-ZOM reveal that existing segmentation models tend to output a mask even when the target is absent, lacking the ability to reject negative expressions. In contrast, WeVisOne-VL can not only output accurate masks when the target is present, but also correctly reject the expression when it is absent.

## 5.1.7 Promptable Matting

Matting predicts a continuous alpha matte that preserves fine structures and partial transparency, such as hair and fur. Existing matting evaluations typically assume a salient foreground [44, 54] or provide a groundtruth mask to specify the target [91, 115]. To evaluate promptable matting in a more practical setting, we use boxes, category names, and referring expressions to specify the target object.

Table 8 Image matting results under box and language prompts. MAD and MSE are multiplied by $1 0 ^ { 3 } .$ . On WeRefMat, all SAM2-Matting results use the SAM3 variant. Bold and underline denote the best and second-best results, respectively.
<table><tr><td>Dataset</td><td>Prompt</td><td>Method</td><td>MAD↓</td><td>MSE↓</td><td>Grad↓</td><td>Conn↓</td><td>SAD↓</td></tr><tr><td rowspan="5">P3M-500-NP</td><td rowspan="5">Box</td><td>Matte Anything [117]</td><td>24.4</td><td>20.4</td><td>20.9</td><td>40.4</td><td>40.9</td></tr><tr><td>SAM2-Matting (SAM2.1-T) [91]</td><td>22.7</td><td>18.2</td><td>20.0</td><td>37.8</td><td>38.1</td></tr><tr><td>SAM2-Matting (SAM2.1-B+) [91]</td><td>19.0</td><td>15.0</td><td>18.3</td><td>31.0</td><td>31.4</td></tr><tr><td>SAM2-Matting (SAM3) [91]</td><td>15.4</td><td>11.4</td><td>17.6</td><td>25.1</td><td>25.8</td></tr><tr><td>WeVisOne-VL-2B</td><td>6.6</td><td>3.1</td><td>10.7</td><td>10.6</td><td>11.2</td></tr><tr><td rowspan="6">AM-2K</td><td rowspan="5">Box</td><td>WeVisOne-VL-4B Matte Anything [117]</td><td>7.3 13.2</td><td>3.8</td><td>10.5</td><td>11.7</td><td>12.6</td></tr><tr><td>[SAM2-Matting (SAM2.1-T) [91]</td><td>11.0</td><td>8.7</td><td>13.9</td><td>22.0</td><td>23.1</td></tr><tr><td>SAM2-Matting (SAM2.1-B+) [91]</td><td>11.1</td><td>6.2</td><td>21.7</td><td>18.4</td><td>18.8</td></tr><tr><td>SAM2-Matting (SAM3) [91]</td><td>12.1</td><td>6.3 7.5</td><td>20.0</td><td>18.8</td><td>19.0</td></tr><tr><td>WeVisOne-VL-2B</td><td>5.7</td><td>2.2</td><td>19.1</td><td>19.5</td><td>20.1</td></tr><tr><td>WeVisOne-VL-4B</td><td></td><td>5.0</td><td>1.6</td><td>8.1 7.2</td><td>8.9 7.5</td><td>9.7 8.5</td></tr><tr><td rowspan="6">PPM-100</td><td rowspan="6">Box</td><td>Matte Anything [117]</td><td>10.5</td><td>6.7</td><td>66.3</td><td>100.1</td><td>102.4</td></tr><tr><td>[SAM2-Matting (SAM2.1-T) [91]</td><td>9.7</td><td>5.9</td><td>68.6</td><td>95.3</td><td>97.1</td></tr><tr><td>SAM2-Matting (SAM2.1-B+) [91]</td><td>9.3</td><td>5.7</td><td>64.0</td><td>90.3</td><td>92.9</td></tr><tr><td>SAM2-Matting (SAM3) [91]</td><td>10.8</td><td>6.9</td><td>65.9</td><td>100.6</td><td>103.2</td></tr><tr><td>WeVisOne-VL-2B</td><td>5.3</td><td>1.8</td><td>48.4</td><td>45.5</td><td></td></tr><tr><td>WeVisOne-VL-4B</td><td>6.1</td><td></td><td></td><td></td><td>49.3</td></tr><tr><td rowspan="8">WeRefMat</td><td rowspan="4">Category</td><td></td><td></td><td>2.4</td><td>48.5</td><td>53.2</td><td>57.2</td></tr><tr><td>[SAM2-Matting (SAM3) [91]</td><td>70.8</td><td>64.9</td><td>45.0</td><td>245.6</td><td>241.6</td></tr><tr><td>Qwen3-VL-8B + SAM2-Matting</td><td>80.7</td><td>74.4</td><td>53.6</td><td>270.5</td><td>267.1</td></tr><tr><td>Qwen3.5-9B + SAM2-Matting WeVisOne-VL-2B</td><td>77.8</td><td>71.4</td><td>51.9</td><td>265.4</td><td>261.8</td></tr><tr><td>WeVisOne-VL-4B</td><td></td><td>17.7 16.4</td><td>13.4</td><td>35.6</td><td>63.6</td><td>65.7</td></tr><tr><td></td><td>[SAM2-Matting (SAM3) [91]</td><td>51.9</td><td>12.5 50.3</td><td>35.0 34.3</td><td>60.3 162.1</td><td>62.0 162.2</td></tr><tr><td></td><td>Qwen3-VL-8B + SAM2-Matting</td><td>23.0</td><td>21.1</td><td>22.6</td><td>78.3</td><td>78.1</td></tr><tr><td rowspan="3">Referring</td><td>Qwen3.5-9B + SAM2-Matting</td><td>22.4</td><td>20.6</td><td>22.1</td><td>82.5</td><td>82.3</td></tr><tr><td>WeVisOne-VL-2B</td><td>15.8</td><td>14.2</td><td>19.4</td><td>65.3</td><td>66.2</td></tr><tr><td>WeVisOne-VL-4B</td><td>9.8</td><td>8.4</td><td>15.9</td><td>37.8</td><td>39.0</td></tr></table>

Benchmarks and Baselines. For box prompts, we use P3M-500-NP [54], AM-2K [56], and PPM-100 [44]. For text prompts, we use our in-house benchmark WeRefMat. For the three established benchmarks, all methods use the same ground-truth box as prompt, and all baselines are run from their oficial checkpoints. We evaluate three SAM2-Matting [91] variants based on SAM2.1 [85] and SAM3 [5]. For box prompts, SAM2-Matting first uses SAM2.1 or SAM3 to predict a mask from the ground-truth box and then performs matting, while WeVisOne-VL constructs a sequence prefix from the ground-truth box, generates the subsequent <OBJ> token, and passes its hidden state to the pixel decoder to predict the alpha matte. On WeRefMat, we compare with SAM3’s native text grounding and two MLLM-based localization settings, in which Qwen3-VL-8B-Instruct or Qwen3.5-9B first localizes the target and passes the resulting box to SAM2-Matting.

Metrics. We report mean absolute diference (MAD), mean squared error (MSE), gradient error (Grad), connectivity error (Conn), and sum of absolute diferences (SAD) between the predicted and ground-truth alpha mattes. Following common practice, MAD and MSE are multiplied by $1 0 ^ { 3 }$ . Lower values indicate better performance for all five metrics.

Results. As shown in Table 8, WeVisOne-VL is competitive with task-specific matting models under box prompts and consistently improves over the SAM2-Matting baseline. On WeRefMat, it performs strongly with both category and referring prompts, showing that language-conditioned target selection and boundarysensitive transparency estimation can be handled efectively within a unified model.

## 6 Qualitative Analysis

We present qualitative comparisons between WeVisOne-VL and representative baselines in Figures 5–12. Each figure is organized by task, and the selected examples below are discussed from top to bottom.<sub>elephant elephant elephant elephant</sub>

![](images/8cfc825686edf766520432125db155cbddecd97214c88267e33bd3d919b58f2a.jpg)  
Figure 5 Qualitative comparison on general and dense object detection.

personObject detection. Figure 5 covers degraded visibility, adjacent instances, and dense scenes. In the first crate <sub>crate</sub>book bookexample, both baselines detect only the foreground elephant, while WeVisOne-VL also recovers the distant elephants partly obscured by dust. Across the carrot and dense fruit examples, the baselines tend to merge apple <sup>orange orange orange</sup> <sub>applebook</sub>adjacent instances into coarse-grained boxes, whereas WeVisOne-VL more often preserves instance-level lo calization.

![](images/6116a8b96b92149b83f67e7fa7c6c4a807d3ca49dbcfb0d1908412765b02ad36.jpg)  
a guy in a white shirt and red shorts

Figure 6 Qualitative comparison on referring object detection.

<sup>two</sup> <sup>people</sup> <sup>gazing</sup> <sup>into</sup> <sup>each</sup> <sup>other's</sup> <sup>eyes</sup>Referring object detection. Figure 6 highlights five aspects of referring object detection: spatial position, GT Rex-Omni LocateAnything WeVisionOne-VLworld knowledge, fine-grained attributes, negative attributes, and absent-target queries. The first four exam ples require interpreting ordinal position, grounding a person’s name, distinguishing headphone color, and excluding people wearing sunglasses, respectively. In these cases, WeVisOne-VL successfully selects the cor-<sup>No</sup> <sup>target none</sup>rect target, while the baselines select distractors or include additional people. The final query specifies a person wearing a white shirt and red shorts, but no one in the image satisfies both conditions. Both baselines still predict a box, whereas WeVisOne-VL correctly rejects the query.

![](images/43cc1cb0bde53bfc0a27b3155cfd11e015d870425d4e90633034c9abd5a48168.jpg)  
Figure 7 Qualitative comparison on object counting.

Object counting. Figure 7 shows the importance of the queried counting unit. The first query asks for cups on the left, and the second asks for pairs of footwear rather than individual shoes. The remaining queries specify green balls and fake green apples among distractors. The baselines often count objects outside the requested subset or count individual shoes instead of pairs; in the last example, Rex-Omni reports 537 rather than 42. WeVisOne-VL gives the ground-truth count in all four examples.

![](images/52c896d81be1027e7ed77a15e5684bd63b50035c283a37e4eb5dc9f21e4bf884.jpg)  
Point to right hook.

Figure 8 Qualitative comparison on object pointing. Ground-truth target regions are highlighted in green; predicted points are shown in pink.

Object pointing. Figure 8 shows dense, relational, attribute-based, and fine-grained pointing queries. In the first example, WeVisOne-VL places points across the heart-shaped balloons, while LocateAnything misses most instances and Rex-Omni produces redundant points on some balloons. The second query asks for two people gazing into each other’s eyes: both baselines select other people, whereas WeVisOne-VL points to the indicated pair. In the third example, Rex-Omni points to multiple pigs and LocateAnything selects a distractor, while WeVisOne-VL identifies the pig facing upward. In the final example, both baselines point to the handle, whereas WeVisOne-VL places its point on the queried curved hook.

![](images/6b29f166c4d4c541b5bebac51daba022b33039e4b5b3ffaf220b921188040c03.jpg)  
Figure 9 Qualitative comparison on OCR.

OCR. Figure 9 presents a shop sign with curved lettering, followed by a receipt and an instruction sheet with dense small text. On the shop sign, the baselines recognize some prominent words but miss the curved lettering, whereas WeVisOne-VL localizes and transcribes the curved words. On the receipt, all methods read several lines, but WeVisOne-VL covers more of the lower text. The diference is more pronounced on the instruction sheet, where WeVisOne-VL localizes substantially more of the small text than either baseline.

![](images/52fef3c0be371d59865fa17cc4971fc84df41d2f996155b21198a9e57eda6c0a.jpg)  
a deer standing next to a fire hydrant on a sidewalk

Figure 10 Qualitative comparison on referring segmentation. Predicted masks remain in full color and the rest of the image is dimmed. The final query asks for an object that is not present.

Referring segmentation. Figure 10 compares the models on attribute, relational, multi-instance, and absenttarget queries. In the first three examples, the expressions refer to the woman with gray hair in front, the dog with lowered ears, and the white-and-red vase on the wood crate. LISA and UniPixel select distractors <sub>a</sub> <sub>deer</sub> <sub>standing</sub> <sub>next</sub> <sub>to</sub> <sub>a</sub> <sub>fire</sub> <sub>hydrant</sub> <sub>on</sub> <sub>a</sub> <sub>sidewalk</sub>or incomplete regions, whereas WeVisOne-VL segments the referred objects. The fourth query specifies the leftmost remote and the third remote from the right; WeVisOne-VL segments both, while LISA produces fragmented masks and UniPixel selects a diferent remote. The final query describes a deer beside a fire hydrant, neither of which appears in the image. Both baselines produce masks, while WeVisOne-VL returns no mask.

![](images/e4b4b4e81ef47e9ce2f2a7849c1d0274334d9a78d2cabd4aef1cc529ff6dfb9f.jpg)  
Figure 11 Qualitative comparison on box-prompted image matting.

Box-prompted matting. Figure 11 shows five box-prompted matting examples: a rabbit, a camel, and three people. In the first two rows, WeVisOne-VL retains the rabbit’s whiskers and more of the camel’s fine coat along the boundary. The third and fifth rows highlight flyaway and curly hair strands, which appear smoother in the baseline mattes. In the fourth row, SAM2-Matting makes the apron logo transparent, leaving a hole in the subject, while WeVisOne-VL keeps the logo and apron opaque.

![](images/26dabbb534e7de483c786e420ba82f19abb73c1913f230ceb66a1e246d777bb8.jpg)  
Figure 12 Qualitative comparison on text-prompted image matting.

Text-prompted matting. Figure 12 presents category prompts for chickens and icebergs (first two examples), followed by referring prompts for the people in black and the balloon bearing “Daikin” (last two). WeVisOnethe people in blackVL retains the visible chickens, gives the icebergs a more opaque matte, and mattes both intended people <sub>Image Qwen3.5</sub> <sub>+</sub> <sub>SAM2-Matting SAM2-Matting</sub> <sub>(SAM3) WeVisionOne-VL</sub>and the named balloon. The Qwen3.5 + SAM2-Matting pipeline selects only one of the two people in black, while SAM2-Matting (SAM3) misses them and includes several distractors in its balloon result.

![](images/8c082ffa003e98e9d2ca305d09d981921c211139e44c86d25820c95ea1803289.jpg)  
Figure 13 Fine-grained caption generation with bounding boxes. Colored phrases and coordinate tokens in each caption correspond to bounding boxes of the same color in the image.

Grounded caption. In Figure 13, we show how WeVisOne-VL can describe an image in detail while grounding the descriptions with bounding boxes. For example, in the baseball scene, it describes the batter’s clothing and swinging action, as well as the catcher and umpire behind him, and links these descriptions to the corresponding image regions. In the outdoor-table scene, it identifies the two people and individual items on the table. Such captions could be valuable for VLM training, as they provide not only detailed descriptions but also explicit connections between text and image regions, helping models learn to associate words and phrases with specific visual content.

## 7 Related Work

Visual Grounding. Open-vocabulary detectors [28, 61, 67] align object localization with category names and referring expressions. Grounded multimodal models then brought spatial outputs into a language interface. Kosmos-2 [78] represents text–box links with location tokens, while Florence-2 [111] casts detection, grounding, and other vision tasks as prompt-conditioned sequence generation. Molmo/PixMo [13] further showed the value of explicit point supervision for grounding and counting. More recently, Rex-Omni [41] uses quantized coordinate tokens and geometry-aware reinforcement learning for detection, referring, pointing, and OCR. LocateAnything [104] decodes whole boxes and points in parallel to improve grounding speed and precision.

Referring Segmentation. SAM [45] and SAM 2 [85] established promptable mask prediction, and SAM 3 [5] extends this line to concept-prompted detection, segmentation, and tracking. To connect language reasoning with pixels, LISA [48] maps a generated segmentation token to a mask decoder, GLaMM [84] interleaves text and grounded masks, and GSVA [110] supports multiple referents and explicit rejection of absent targets. Later work explores diferent output representations and task mixtures: Text4Seg [50] serializes masks as text, and UniPixel [68] integrates masks and pixel-level reasoning. Sa2VA [122] couples an MLLM with SAM 2, while Youtu-VL [107] trains visual and language tokens under unified autoregressive supervision.

Instance Matting. Image matting estimates continuous opacity for fine structures and translucent boundaries. Automatic methods such as MODNet [44] and P3M [54] focus on portrait foregrounds. Matte Anything [117] and Matting Anything [53] use promptable segmentation to guide alpha prediction, while In-Context Matting [29] specifies the foreground through a prompted reference image. SAM2-Matting [91] adds matting components to a segmentation tracker. For video, MatAnyone [115] propagates the matte of a target specified by a first-frame mask. MatAnyone 2 [116] improves detail and temporal consistency through learned quality evaluation and larger real-world training data. Many matting evaluations assume a salient foreground [44, 54] or provide a ground-truth mask prompt [115]. We evaluate promptable matting in multi-object scenes us ing boxes, category names, and referring expressions to specify targets, with WeVisOne-VL predicting their continuous alpha mattes.

## 8 Limitations

In this work, we focus on fundamental visual perception, which entails several trade-ofs. First, the limited amount of general-purpose training data may compromise the general capabilities relative to the base model. Second, for inference eficiency, we train the model to directly produce structured outputs without chain-of thought (CoT) reasoning [21, 69], which potentially limits its ability to tackle complex reasoning tasks. Third, our current approach is limited to single-turn prediction. It is promising to integrate an agent framework to progressively refine visual understanding through multi-turn tool interactions [38, 62].

Beyond these limitations, a key challenge lies in the real-world complexity. The object prompts in benchmarks are typically clear, well-formed, and unambiguous. However, real-world user instructions are often vague and hesitant. Variations in contexts or communication styles can make a straightforward task, such as locating the target, significantly more challenging. Therefore, strong intent reasoning is essential for a reliable vision system. In future work, we will continue to improve WeVisOne-VL toward addressing these limitations.

## 9 Conclusion

We introduced WeVisOne-VL, a vision-language model that unifies object-level (e.g., object detection, referring localization, counting, and OCR) and pixel-level (e.g., referring segmentation, and image matting) tasks through a structured next-token interface. Dedicated coordinate tokens encode instance predictions, while an <OBJ> token connects each localized instance to a lightweight pixel decoder for masks and alpha mattes. A three-stage supervised curriculum develops task alignment, high-resolution dense perception, and pixel-level prediction. After that, we apply mixed-task reinforcement learning to optimize the complete grounding outputs. Across the evaluated benchmarks, WeVisOne-VL achieves strong results on both object level and pixel-level tasks. It improves mean detection F1 and counting accuracy over comparable-scale VLMs, advances text grounding and referring localization, and surpasses the evaluated baselines for referring segmentation and image matting. Qualitative examples further show that the model can localize individual instances in crowded scenes, ground expressions involving multiple objects, and recognize when a requested target is absent. By unifying these capabilities within a single language-guided model, WeVisOne-VL provides a foundation for practical visual systems.

## References

[1] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-VL Technical Report. arXiv preprint arXiv:2511.21631, 2025. URL https://arxiv.org/abs/2511.21631.

[2] Ali Furkan Biten, Rubèn Tito, Lluis Gomez, Ernest Valveny, and Dimosthenis Karatzas. OCR-IDL: OCR Annotations for Industry Document Library Dataset. arXiv preprint arXiv:2202.12985, 2022. URL https: //arxiv.org/abs/2202.12985.

[3] Huanqia Cai, Fanglei Xue, Lele Xu, and Lili Guo. TransMatting: Enhancing Transparent Objects Matting with Transformers. In ECCV, pages 253–269, 2022. doi: 10.1007/978-3-031-19818-2\_15.

[4] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-End Object Detection with Transformers. In ECCV, pages 213–229, 2020. doi: 10.1007/978-3-030-584 52-8\_13.

[5] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris Coll-Vinent, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, Jie Lei, Tengyu Ma, Baishan Guo, Arpit Kalla, Markus Marks, Joseph Greer, Meng Wang, Peize Sun, Roman Rädle, Triantafyllos Afouras, Efrosyni Mavroudi, Katherine Xu, Tsung-Han Wu, Yu Zhou, Liliane Momeni, Rishi Hazra, Shuangrui Ding, Sagar Vaze, Francois Porcher, Feng Li, Siyuan Li, Aishwarya Kamath, Ho Kei Cheng, Piotr Dollár, Nikhila Ravi, Kate Saenko, Pengchuan Zhang, and Christoph Feichtenhofer. SAM 3: Segment Anything with Concepts. In ICLR, 2026.

[6] Jierun Chen, Fangyun Wei, Jinjing Zhao, Sizhe Song, Bohuai Wu, Zhuoxuan Peng, S.-H. Gary Chan, and Hongyang Zhang. Revisiting Referring Expression Comprehension Evaluation in the Era of Large Multimodal Models. In CVPR Workshops, pages 513–524, 2025.

[7] Long Cheng, Jiafei Duan, Yi Ru Wang, Haoquan Fang, Boyang Li, Yushan Huang, Elvis Wang, Ainaz Eftekhar, Jason Lee, Wentao Yuan, Rose Hendrix, Noah A. Smith, Fei Xia, Dieter Fox, and Ranjay Krishna. PointArena: Probing Multimodal Grounding Through Language-Guided Pointing. arXiv preprint arXiv:2505.09990, 2025. URL https://arxiv.org/abs/2505.09990.

[8] Chee Kheng Ch’ng and Chee Seng Chan. Total-Text: A Comprehensive Dataset for Scene Text Detection and Recognition. In ICDAR, volume 1, pages 935–942, 2017. doi: 10.1109/icdar.2017.157.

[9] Chee-Kheng Chng, Yuliang Liu, Yipeng Sun, Chun Chet Ng, Canjie Luo, Zihan Ni, ChuanMing Fang, Shuaitao Zhang, Junyu Han, Errui Ding, et al. ICDAR2019 Robust Reading Challenge on Arbitrary-Shaped Text - RRC-ArT. In ICDAR, pages 1571–1576, 2019. doi: 10.1109/icdar.2019.00252.

[10] Christopher Clark, Jieyu Zhang, Zixian Ma, Jae Sung Park, Rohun Tripathi, Sangho Lee, Mohammadreza Salehi, Jason Ren, Chris Dongjoo Kim, Yinuo Yang, et al. Molmo2: Open Weights and Data for Vision Language Models with Video Understanding and Grounding. In CVPR, pages 28652–28668, 2026.

[11] cpuimage. CelebAHairMask-HQ. Dataset, 2025. URL https://github.com/cpuimage/CelebAHairMask-HQ. Version 1.0 PLUS. Accessed: October 22, 2026.

[12] Cheng Cui, Ting Sun, Manhui Lin, Tingquan Gao, Yubo Zhang, Jiaxuan Liu, Xueqing Wang, Zelun Zhang, Changda Zhou, Hongen Liu, Yue Zhang, Wenyu Lv, Kui Huang, Yichao Zhang, Jing Zhang, Jun Zhang, Yi Liu, Dianhai Yu, and Yanjun Ma. PaddleOCR 3.0 Technical Report. arXiv preprint arXiv:2507.05595, 2025. URL https://arxiv.org/abs/2507.05595.

[13] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighof, Kyle Lo, Luca Soldaini, et al. Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models. In CVPR, pages 91–104, 2025.

[14] Patrick Dendorfer, Hamid Rezatofighi, Anton Milan, Javen Shi, Daniel Cremers, Ian Reid, Stefan Roth, Konrad Schindler, and Laura Leal-Taixé. MOT20: A benchmark for multi object tracking in crowded scenes. arXiv preprint arXiv:2003.09003, 2020. URL https://arxiv.org/abs/2003.09003.

[15] Patrick Dendorfer, Aljoša Ošep, Anton Milan, Konrad Schindler, Daniel Cremers, et al. MOTChallenge: A Benchmark for Single-Camera Multiple Target Tracking. International Journal of Computer Vision, 129(4): 845–881, 2021. doi: 10.1007/s11263-020-01393-0.

[16] Xueqing Deng, Linjie Yang, Qihang Yu, Ali Athar, Chenglin Yang, Xiaojie Jin, Xiaohui Shen, and Liang-Chieh Chen. COCONut-PanCap: Joint Panoptic Segmentation and Grounded Captions for Fine-Grained Understanding and Generation. In NeurIPS, volume 38, 2025. doi: 10.52202/085713-1912.

[17] Qihua Dong, Kuo Yang, Lin Ju, Handong Zhao, Yitian Zhang, Yizhou Wang, Huimin Zeng, Jianglin Lu, and Yun Fu. Ref-Adv: Exploring MLLM Visual Reasoning in Referring Expression Tasks. In ICLR, pages 105393– 105409, 2026.

[18] Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, et al. PaLM-E: An Embodied Multimodal Language Model. In ICML, volume 202, pages 8469–8488, 2023.

[19] Dawei Du, Pengfei Zhu, Longyin Wen, Xiao Bian, Haibin Ling, et al. VisDrone-DET2019: The Vision Meets Drone Object Detection in Image Challenge Results. In ICCV Workshops, pages 213–226, 2019. doi: 10.1109/ iccvw.2019.00030.

[20] Corentin Dumery, Niki Amini-Naieni, Shervin Naini, and Pascal Fua. The MixCount Dataset: Bridging the Data Gap for Open-Vocabulary Object Counting. arXiv preprint arXiv:2605.18063, 2026. URL https://arxi v.org/abs/2605.18063.

[21] Kaituo Feng, Manyuan Zhang, Hongyu Li, Kaixuan Fan, Shuang Chen, Yilei Jiang, Dian Zheng, Peiwen Sun, Yiyuan Zhang, Haoze Sun, et al. OneThinker: All-in-one Reasoning Model for Image and Video. In CVPR, pages 5432–5443, 2026.

[22] Tianyi Gao, Han Fang, Tianyi Ding, Hao Li, Xin Wei, Hongbo Sun, Xiaodong Dong, Ye Yuan, Jinglin Xu, Kongming Liang, Hao Sun, and Jingmin Xin. Mixture-of-Thought-Tokens: Unifying Perception and Reasoning for Free-form Multimodal Grounding. arXiv preprint arXiv:2607.24407, 2026. URL https://arxiv.org/abs/ 2607.24407.

[23] Xinyan Gao, Haoran Hao, and Xiangyu Yue. Reason Twice: Segmentation via Candidate Discovery and Com parative Reasoning. arXiv preprint arXiv:2606.09303, 2026. URL https://arxiv.org/abs/2606.09303.

[24] Yuying Ge, Ruimao Zhang, Xiaogang Wang, Xiaoou Tang, and Ping Luo. DeepFashion2: A Versatile Benchmark for Detection, Pose Estimation, Segmentation and Re-Identification of Clothing Images. In CVPR, 2019.

[25] Ross Girshick, Jef Donahue, Trevor Darrell, and Jitendra Malik. Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation. In CVPR, pages 580–587, 2014.

[26] Eran Goldman, Roei Herzig, Aviv Eisenschtat, Jacob Goldberger, and Tal Hassner. Precise Detection in Densely Packed Scenes. In CVPR, 2019.

[27] Sitong Gong, Yunzhi Zhuge, Lu Zhang, Zongxin Yang, Pingping Zhang, and Huchuan Lu. The Devil is in Temporal Token: High Quality Video Reasoning Segmentation. In CVPR, pages 29183–29192, 2025.

[28] Yuchen Guan, Chong Sun, Canmiao Fu, Zhipeng Huang, Chun Yuan, and Chen Li. Text-guided Visual Prompt DINO for Generic Segmentation. In ICCV, pages 21288–21298, 2025.

[29] He Guo, Zixuan Ye, Zhiguo Cao, and Hao Lu. In-Context Matting. In CVPR, pages 3711–3720, 2024.

[30] Agrim Gupta, Piotr Dollar, and Ross Girshick. LVIS: A Dataset for Large Vocabulary Instance Segmentation. In CVPR, pages 5356–5364, 2019.

[31] Ju He, Shuo Yang, Shaokang Yang, Adam Kortylewski, Xiaoding Yuan, Jie-Neng Chen, Shuai Liu, Cheng Yang, Qihang Yu, and Alan Yuille. PartImageNet: A Large, High-Quality Dataset of Parts. In ECCV, pages 128–145, 2022. doi: 10.1007/978-3-031-20074-8\_8.

[32] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep Residual Learning for Image Recognition. In CVPR, pages 770–778, 2016.

[33] Shuting He, Henghui Ding, Chang Liu, and Xudong Jiang. GREC: Generalized Referring Expression Compre hension. arXiv preprint arXiv:2308.16182, 2023. URL https://arxiv.org/abs/2308.16182.

[34] Yutao Hu, Qixiong Wang, Wenqi Shao, Enze Xie, Zhenguo Li, Jungong Han, and Ping Luo. Beyond One-to-One: Rethinking the Referring Image Segmentation. In ICCV, pages 4067–4077, 2023.

[35] Zheng Huang, Kai Chen, Jianhua He, Xiang Bai, Dimosthenis Karatzas, Shijian Lu, and C. V. Jawahar. IC-DAR2019 Competition on Scanned Receipt OCR and Information Extraction. In ICDAR, pages 1516–1520, 2019. doi: 10.1109/icdar.2019.00244.

[36] Chuong Huynh, Seoung Wug Oh, Abhinav Shrivastava, and Joon-Young Lee. MaGGIe: Masked Guided Gradual Human Instance Matting. In CVPR, pages 3870–3879, 2024.

[37] Jongheon Jeong, Yang Zou, Taewan Kim, Dongqing Zhang, Avinash Ravichandran, and Onkar Dabeer. Win-CLIP: Zero-/Few-Shot Anomaly Classification and Segmentation. In CVPR, pages 19606–19616, 2023.

[38] Haichao Jiang, Tianming Liang, Wei-Shi Zheng, and Jian-Fang Hu. Refer-Agent: A Collaborative Multi-Agent System with Reasoning and Reflection for Referring Video Object Segmentation. In CVPR, pages 39508–39517, 2026.

[39] Qing Jiang, Lin Wu, Zhaoyang Zeng, Tianhe Ren, Yuda Xiong, Yihao Chen, Liu Qin, and Lei Zhang. Referring to Any Person. In ICCV, pages 21667–21678, 2025. doi: 10.1109/iccv51701.2025.02012.

[40] Qing Jiang, Xingyu Chen, Zhaoyang Zeng, Junzhi Yu, and Lei Zhang. Rex-Thinker: Grounded Object Referring via Chain-of-Thought Reasoning. In ICLR, pages 151913–151957, 2026.

[41] Qing Jiang, Junan Huo, Xingyu Chen, Yuda Xiong, Zhaoyang Zeng, Yihao Chen, Tianhe Ren, Junzhi Yu, and Lei Zhang. Detect Anything via Next Point Prediction. In CVPR, pages 25472–25483, 2026.

[42] Dimosthenis Karatzas, Faisal Shafait, Seiichi Uchida, Masakazu Iwamura, Lluis Gomez i Bigorda, Sergi Robles Mestre, Joan Mas, David Fernandez Mota, Jon Almazan, and Lluis Pere de las Heras. ICDAR 2013 Robust Reading Competition. In ICDAR, pages 1484–1493, 2013. doi: 10.1109/icdar.2013.221.

[43] Dimosthenis Karatzas, Lluis Gomez-Bigorda, Anguelos Nicolaou, Suman Ghosh, Andrew Bagdanov, Masakazu Iwamura, Jiri Matas, Lukas Neumann, Vijay Ramaseshan Chandrasekhar, Shijian Lu, Faisal Shafait, Seiichi Uchida, and Ernest Valveny. ICDAR 2015 competition on Robust Reading. In ICDAR, pages 1156–1160, 2015. doi: 10.1109/icdar.2015.7333942.

[44] Zhanghan Ke, Jiayu Sun, Kaican Li, Qiong Yan, and Rynson W.H. Lau. MODNet: Real-Time Trimap-Free Portrait Matting via Objective Decomposition. In AAAI, volume 36, pages 1140–1147, 2022. doi: 10.1609/aa ai.v36i1.19999.

[45] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, et al. Segment Anything. In ICCV, pages 4015–4026, 2023.

[46] Alex Krizhevsky, Ilya Sutskever, and Geofrey E. Hinton. ImageNet Classification with Deep Convolutional Neural Networks. In NeurIPS, volume 25, 2012.

[47] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, et al. The Open Images Dataset V4: Unified Image Classification, Object Detection, and Visual Relationship Detection at Scale. International Journal of Computer Vision, 128(7):1956–1981, 2020. doi: 10.1007/s11263-020-01316-z.

[48] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. LISA: Reasoning Segmentation via Large Language Model. In CVPR, pages 9579–9589, 2024.

[49] LAION. LAION Datasets, n.d. URL https://projects.laion.ai/laion-datasets/. Accessed: October 22, 2026.

[50] Mengcheng Lan, Chaofeng Chen, Yue Zhou, Jiaxing Xu, Yiping Ke, Xinjiang Wang, Litong Feng, and Wayne Zhang. Text4Seg: Reimagining Image Segmentation as Text Generation. In ICLR, pages 1634–1661, 2025.

[51] Mengqi Lei, Shuokun Cheng, Wei Bao, Shaoyi Du, Jun-Hai Yong, Siqi Li, and Yue Gao. Count Anything. arXiv preprint arXiv:2605.30846, 2026. URL https://arxiv.org/abs/2605.30846.

[52] Chunyuan Li, Haotian Liu, Liunian Li, Pengchuan Zhang, Jyoti Aneja, Jianwei Yang, Ping Jin, Houdong Hu, Zicheng Liu, Yong Jae Lee, and Jianfeng Gao. ELEVATER: A Benchmark and Toolkit for Evaluating Language-Augmented Visual Models. In NeurIPS, volume 35, pages 9287–9301, 2022. doi: 10.52202/068431-0675.

[53] Jiachen Li, Jitesh Jain, and Humphrey Shi. Matting Anything. In CVPR Workshops, pages 1775–1785, 2024.

[54] Jizhizi Li, Sihan Ma, Jing Zhang, and Dacheng Tao. Privacy-Preserving Portrait Matting. In ACM MM, pages 3501–3509, 2021. doi: 10.1145/3474085.3475512.

[55] Jizhizi Li, Jing Zhang, and Dacheng Tao. Deep Automatic Natural Image Matting. In IJCAI, pages 800–806, 2021. doi: 10.24963/ijcai.2021/111.

[56] Jizhizi Li, Jing Zhang, Stephen J. Maybank, and Dacheng Tao. Bridging Composite and Real: Towards Endto-End Deep Image Matting. International Journal of Computer Vision, 130(2):246–266, 2022. doi: 10.1007/s1 1263-021-01541-0.

[57] Jizhizi Li, Jing Zhang, and Dacheng Tao. Referring Image Matting. In CVPR, pages 22448–22457, 2023.

[58] Rang Li, Lei Li, Shuhuai Ren, Hao Tian, Shuhao Gu, Shicheng Li, Zihao Yue, Yudong Wang, Wenhan Ma, Zhe Yang, Jingyuan Ma, Zhifang Sui, and Fuli Luo. GroundingME: Exposing the Visual Grounding Gap in MLLMs through Multi-Dimensional Evaluation. In CVPR, pages 2412–2422, 2026.

[59] Xiao Li, Yining Liu, Na Dong, Sitian Qin, and Xiaolin Hu. PartImageNet++ Dataset: Scaling Up Part-Based Models for Robust Recognition. In ECCV, pages 396–414, 2024. doi: 10.1007/978-3-031-73209-6\_23.

[60] Xiaojie Li, Lu Yang, Qing Song, and Fuqiang Zhou. Detector-in-Detector: Multi-Level Analysis for Human-Parts. In ACCV, pages 228–240, 2018. doi: 10.1007/978-3-030-20890-5\_15.

[61] Tianming Liang, Kun-Yu Lin, Chaolei Tan, Jianguo Zhang, Wei-Shi Zheng, and Jian-Fang Hu. ReferDINO: Referring Video Object Segmentation with Visual Grounding Foundations. In ICCV, pages 20009–20019, 2025.

[62] Tianming Liang, Qirui Du, Jian-Fang Hu, Haichao Jiang, Zicheng Lin, and Wei-Shi Zheng. Seg-ReSearch: Segmentation with Interleaved Reasoning and External Search. In ICML, volume 306, pages 73171–73186, 2026.

[63] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. Microsoft COCO: Common Objects in Context. In ECCV, pages 740–755, 2014. doi: 10.1007/978-3-319-10602-1\_48.

[64] Yiheng Lin, Yihan Hu, Chenyi Zhang, Ting Liu, Xiaochao Qu, Luoqi Liu, Yao Zhao, and Yunchao Wei. Memory Eficient Matting with Adaptive Token Routing. In AAAI, volume 39, pages 5298–5306, 2025. doi: 10.1609/aa ai.v39i5.32563.

[65] Chang Liu, Xiangtai Li, and Henghui Ding. Referring Image Editing: Object-level Image Editing via Referring Expressions. In CVPR, pages 13128–13138, 2024.

[66] Chang Liu, Haoning Wu, and Weidi Xie. Count Anything at Any Granularity. arXiv preprint arXiv:2605.10887, 2026. URL https://arxiv.org/abs/2605.10887.

[67] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, and Lei Zhang. Grounding DINO: Marrying DINO with Grounded Pre-training for Open-Set Object Detection. In ECCV, pages 38–55, 2024. doi: 10.1007/978-3-031-72970-6\_3.

[68] Ye Liu, Zongyang Ma, Junfu Pu, Zhongang Qi, Yang Wu, Ying Shan, and Chang Chen. UniPixel: Unified Object Referring and Segmentation for Pixel-Level Visual Reasoning. In NeurIPS, volume 38, pages 126078–126108, 2025. doi: 10.52202/085713-4203.

[69] Yuqi Liu, Bohao Peng, Zhisheng Zhong, Zihao Yue, Fanbin Lu, Bei Yu, and Jiaya Jia. Seg-Zero: Reasoning-Chain Guided Segmentation via Cognitive Reinforcement. arXiv preprint arXiv:2503.06520, 2025. URL https: //arxiv.org/abs/2503.06520.

[70] Jonathan Long, Evan Shelhamer, and Trevor Darrell. Fully Convolutional Networks for Semantic Segmentation. In CVPR, pages 3431–3440, 2015.

[71] Shangbang Long, Siyang Qin, Dmitry Panteleev, Alessandro Bissacco, Yasuhisa Fujii, and Michalis Raptis. Towards End-to-End Unified Scene Text Detection and Layout Analysis. In CVPR, pages 1049–1059, 2022.

[72] Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L. Yuille, and Kevin Murphy. Generation and Comprehension of Unambiguous Object Descriptions. In CVPR, pages 11–20, 2016.

[73] George A. Miller. WordNet: A Lexical Database for English. Communications of the ACM, 38(11):39–41, 1995. doi: 10.1145/219717.219748.

[74] Motional. nuImages, 2020. URL https://www.nuscenes.org/nuimages.

[75] Thanh Nguyen, Chau Pham, Khoi Nguyen, and Minh Hoai. Few-Shot Object Counting and Detection. In ECCV, pages 348–365, 2022. doi: 10.1007/978-3-031-20044-1\_20.

[76] Ting Pan, Lulu Tang, Xinlong Wang, and Shiguang Shan. Tokenize Anything via Prompting. In ECCV, pages 330–348, 2024. doi: 10.1007/978-3-031-72970-6\_19.

[77] Wujian Peng, Lingchen Meng, Yitong Chen, Yiweng Xie, Yang Liu, Tao Gui, Hang Xu, Xipeng Qiu, Zuxuan Wu, and Yu-Gang Jiang. INST-IT: Boosting Instance Understanding via Explicit Visual Prompt Instruction Tuning. In NeurIPS, volume 38, pages 50062–50092, 2025. doi: 10.52202/085713-1669.

[78] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, Qixiang Ye, and Furu Wei. Grounding Multimodal Large Language Models to the World. In ICLR, pages 51575–51598, 2024.

[79] Bryan A. Plummer, Liwei Wang, Chris M. Cervantes, Juan C. Caicedo, Julia Hockenmaier, and Svetlana Lazeb nik. Flickr30k Entities: Collecting Region-to-Phrase Correspondences for Richer Image-to-Sentence Models. In ICCV, 2015.

[80] Yu Qiao, Yuhao Liu, Xin Yang, Dongsheng Zhou, Mingliang Xu, Qiang Zhang, and Xiaopeng Wei. Attention Guided Hierarchical Structure Aggregation for Image Matting. In CVPR, 2020.

[81] Qwen Team. Qwen3.5: Towards Native Multimodal Agents, February 2026. URL https://qwen.ai/blog?id= qwen3.5.

[82] Vignesh Ramanathan, Anmol Kalia, Vladan Petrovic, Yi Wen, Baixue Zheng, Baishan Guo, Rui Wang, Aaron Marquez, Rama Kovvuri, Abhishek Kadian, Amir Mousavi, Yiwen Song, Abhimanyu Dubey, and Dhruv Maha jan. PACO: Parts and Attributes of Common Objects. In CVPR, pages 7141–7151, 2023.

[83] Viresh Ranjan, Udbhav Sharma, Thu Nguyen, and Minh Hoai. Learning To Count Everything. In CVPR, pages 3394–3403, 2021.

[84] Hanoona Rasheed, Muhammad Maaz, Sahal Shaji, Abdelrahman Shaker, Salman Khan, Hisham Cholakkal, Rao M. Anwer, Eric Xing, Ming-Hsuan Yang, and Fahad S. Khan. GLaMM: Pixel Grounding Large Multimodal Model. In CVPR, pages 13009–13018, 2024.

[85] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. SAM 2: Segment Anything in Images and Videos. In ICLR, pages 28085–28128, 2025.

[86] Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You Only Look Once: Unified, Real-Time Object Detection. In CVPR, pages 779–788, 2016. doi: 10.1109/cvpr.2016.91.

[87] Seonguk Seo, Joon-Young Lee, and Bohyung Han. URVOS: Unified Referring Video Object Segmentation Network with a Large-Scale Benchmark. In ECCV, 2020.

[88] Shuai Shao, Zijian Zhao, Boxun Li, Tete Xiao, Gang Yu, Xiangyu Zhang, and Jian Sun. CrowdHuman: A Benchmark for Detecting Human in a Crowd. arXiv preprint arXiv:1805.00123, 2018. URL https://arxiv.or g/abs/1805.00123.

[89] Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu Zhang, Jing Li, and Jian Sun. Ob jects365: A Large-Scale, High-Quality Dataset for Object Detection. In ICCV, 2019.

[90] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual Captions: A Cleaned, Hypernymed, Image Alt-text Dataset For Automatic Image Captioning. In ACL, pages 2556–2565, 2018. doi: 10.18653/v1/p18-1238.

[91] Ruiqi Shen, Guangquan Jie, Chang Liu, and Henghui Ding. SAM2Matting: Generalized Image and Video Matting. In ECCV, 2026.

[92] Baoguang Shi, Cong Yao, Minghui Liao, Mingkun Yang, Pei Xu, Linyan Cui, Serge Belongie, Shijian Lu, and Xiang Bai. ICDAR2017 Competition on Reading Chinese Text in the Wild (RCTW-17). In ICDAR, pages 1429–1434, 2017. doi: 10.1109/ICDAR.2017.233.

[93] shubh303. Open World Dense Object Detection Dataset. Hugging Face dataset, 2026. URL https://huggingf ace.co/datasets/shubh303/open-world-dense-object-detection. Accessed: October 22, 2026.

[94] Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, et al. DINOv3. arXiv preprint arXiv:2508.10104, 2025. URL https://arxiv.org/abs/2508.10104.

[95] Amanpreet Singh, Guan Pang, Mandy Toh, Jing Huang, Wojciech Galuba, and Tal Hassner. TextOCR: Towards Large-Scale End-to-End Reasoning for Arbitrary-Shaped Scene Text. In CVPR, pages 8802–8812, 2021.

[96] Hongbin Sun, Zhanghui Kuang, Xiaoyu Yue, Chenhao Lin, and Wayne Zhang. Spatial Dual-Modality Graph Reasoning for Key Information Extraction. arXiv preprint arXiv:2103.14470, 2021. URL https://arxiv.org/ abs/2103.14470.

[97] Yanan Sun, Chi-Keung Tang, and Yu-Wing Tai. Semantic Image Matting. In CVPR, pages 11120–11129, 2021.

[98] Yanan Sun, Chi-Keung Tang, and Yu-Wing Tai. Ultrahigh Resolution Image/Video Matting With Spatio Temporal Sparsity. In CVPR, pages 14112–14121, 2023.

[99] Yipeng Sun, Zihan Ni, Chee-Kheng Chng, Yuliang Liu, Canjie Luo, Chun Chet Ng, Junyu Han, Errui Ding, Jingtuo Liu, Dimosthenis Karatzas, et al. ICDAR 2019 Competition on Large-Scale Street View Text with Partial Labeling - RRC-LSVT. In ICDAR, pages 1557–1562, 2019. doi: 10.1109/icdar.2019.00250.

[100] Jayant Sravan Tamarapalli, Rynaa Grover, Nilay Pande, and Sahiti Yerramilli. CountQA: How Well Do MLLMs Count in the Wild? arXiv preprint arXiv:2508.06585, 2025. URL https://arxiv.org/abs/2508.06585.

[101] Andreas Veit, Tomas Matera, Lukas Neumann, Jiri Matas, and Serge Belongie. COCO-Text: Dataset and Benchmark for Text Detection and Recognition in Natural Images. arXiv preprint arXiv:1601.07140, 2016. URL https://arxiv.org/abs/1601.07140.

[102] Jiaqi Wang, Pan Zhang, Tao Chu, Yuhang Cao, Yujie Zhou, Tong Wu, Bin Wang, Conghui He, and Dahua Lin. V3Det: Vast Vocabulary Visual Detection Dataset. In ICCV, pages 19844–19854, 2023.

[103] Junchi Wang and Lei Ke. LLM-Seg: Bridging Image Segmentation and Large Language Model Reasoning. In CVPR Workshops, pages 1765–1774, 2024.

[104] Shihao Wang, Shilong Liu, Yuanguo Kuang, Xinyu Wei, Yangzhou Liu, Zhiqi Li, Yunze Man, Guo Chen, Andrew Tao, Guilin Liu, et al. LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding. arXiv preprint arXiv:2605.27365, 2026. URL https://arxiv.org/abs/2605.27365.

[105] Tiantian Wang, Sifei Liu, Yapeng Tian, Kai Li, and Ming-Hsuan Yang. Video Matting via Consistency-Regularized Graph Neural Networks. In ICCV, pages 4902–4911, 2021.

[106] Cong Wei, Haoxian Tan, Yujie Zhong, Yujiu Yang, and Lin Ma. LaSagnA: Language-based Segmentation Assistant for Complex Queries. arXiv preprint arXiv:2404.08506, 2024. URL https://arxiv.org/abs/2404.0 8506.

[107] Zhixiang Wei, Yi Li, Zhehan Kan, Xinghua Jiang, Zuwei Long, Shifeng Liu, Hongze Shen, Wei Liu, Xiaoyu Tan, Haojia Lin, et al. Youtu-VL: Unleashing Visual Potential via Unified Vision-Language Supervision. arXiv preprint arXiv:2601.19798, 2026. URL https://arxiv.org/abs/2601.19798.

[108] Chenyun Wu, Zhe Lin, Scott Cohen, Trung Bui, and Subhransu Maji. PhraseCut: Language-Based Image Segmentation in the Wild. In CVPR, 2020.

[109] Zongjian Wu and Lei Zhang. Towards Open-World Referring Expression Comprehension: A Benchmark with Training-free Multi-task Consistency Checker. arXiv preprint arXiv:2605.25706, 2026. URL https://arxiv.or g/abs/2605.25706.

[110] Zhuofan Xia, Dongchen Han, Yizeng Han, Xuran Pan, Shiji Song, and Gao Huang. GSVA: Generalized Seg mentation via Multimodal Large Language Models. In CVPR, pages 3858–3869, 2024.

[111] Bin Xiao, Haiping Wu, Weijian Xu, Xiyang Dai, Houdong Hu, Yumao Lu, Michael Zeng, Ce Liu, and Lu Yuan. Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks. In CVPR, pages 4818–4829, 2024.

[112] Chi Xie, Zhao Zhang, Yixuan Wu, Feng Zhu, Rui Zhao, and Shuang Liang. Described Object Detection: Liberating Object Detection with Flexible Expressions. In NeurIPS, volume 36, pages 79095–79107, 2023. doi: 10.52202/075280-3460.

[113] Le Xue, Manli Shu, Anas Awadalla, Jun Wang, An Yan, Senthil Purushwalkam, Honglu Zhou, Viraj Prabhu, Yutong Dai, Michael S Ryoo, et al. xGen-MM (BLIP-3): A Family of Open Large Multimodal Models. arXiv preprint arXiv:2408.08872, 2024. URL https://arxiv.org/abs/2408.08872.

[114] Cilin Yan, Haochen Wang, Shilin Yan, Xiaolong Jiang, Yao Hu, Guoliang Kang, Weidi Xie, and Efstratios Gavves. VISA: Reasoning Video Object Segmentation via Large Language Models. In ECCV, pages 98–115, 2024. doi: 10.1007/978-3-031-72633-0\_6.

[115] Peiqing Yang, Shangchen Zhou, Jixin Zhao, Qingyi Tao, and Chen Change Loy. MatAnyone: Stable Video Matting with Consistent Memory Propagation. In CVPR, pages 7299–7308, 2025.

[116] Peiqing Yang, Shangchen Zhou, Kai Hao, and Qingyi Tao. MatAnyone 2: Scaling Video Matting via a Learned Quality Evaluator. In CVPR, pages 37476–37485, 2026.

[117] Jingfeng Yao, Xinggang Wang, Lang Ye, and Wenyu Liu. Matte anything: Interactive natural image matting with segment anything model. Image and Vision Computing, 147:105067, 2024. doi: 10.1016/j.imavis.2024.10 5067.

[118] Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From Image Descriptions to Visual Denotations: New Similarity Metrics for Semantic Inference over Event Descriptions. Transactions of the Association for Computational Linguistics, 2:67–78, 2014. doi: 10.1162/tacl\_a\_00166.

[119] Fisher Yu, Haofeng Chen, Xin Wang, Wenqi Xian, Yingying Chen, Fangchen Liu, Vashisht Madhavan, and Trevor Darrell. BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning. In CVPR, 2020.

[120] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C. Berg, and Tamara L. Berg. Modeling Context in Referring Expressions. In ECCV, pages 69–85, 2016. doi: 10.1007/978-3-319-46475-6\_5.

[121] Qihang Yu, Jianming Zhang, He Zhang, Yilin Wang, Zhe Lin, Ning Xu, Yutong Bai, and Alan Yuille. Mask Guided Matting via Progressive Refinement Network. In CVPR, pages 1154–1163, 2021.

[122] Haobo Yuan, Xiangtai Li, Tao Zhang, Yueyi Sun, Zilong Huang, Shilin Xu, Shunping Ji, Yunhai Tong, Lu Qi, Jiashi Feng, et al. Sa2VA: Marrying SAM2 With MLLM for Dense Grounded Understanding of Images and Videos. IEEE Transactions on Pattern Analysis and Machine Intelligence, 48(11):14922–14936, 2026. doi: 10.1109/TPAMI.2026.3720379.

[123] Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M. Ni, and Heung-Yeung Shum. DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection. In ICLR, 2023.

[124] Rui Zhang, Mingkun Yang, Xiang Bai, Baoguang Shi, Dimosthenis Karatzas, Shijian Lu, C. V. Jawahar, Yongsheng Zhou, Qianyi Jiang, Qi Song, et al. ICDAR 2019 Robust Reading Challenge on Reading Chinese Text on Signboard. In ICDAR, pages 1577–1581, 2019. doi: 10.1109/icdar.2019.00253.

[125] Peng Zheng, Dehong Gao, Deng-Ping Fan, Li Liu, Jorma Laaksonen, Wanli Ouyang, and Nicu Sebe. Bilateral Reference for High-Resolution Dichotomous Image Segmentation. CAAI Artificial Intelligence Research, 3: 9150038, 2024. URL https://github.com/ZhengPeng7/BiRefNet.

[126] Enshen Zhou, Jingkun An, Cheng Chi, Yi Han, Shanyu Rong, Chi Zhang, Pengwei Wang, Zhongyuan Wang, Tiejun Huang, Lu Sheng, and Shanghang Zhang. RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models for Robotics. In NeurIPS, volume 38, pages 28404–28481, 2025. doi: 10.52202/08571 3-0957.

[127] Yue Zhou, Jue Chen, Zilun Zhang, Penghui Huang, Ran Ding, Zhentao Zou, PengFei Gao, Yuchen Wei, Ke Li, Xue Yang, Xue Jiang, Hongxin Yang, and Jonathan Li. DVGBench: Implicit-to-explicit visual grounding benchmark in UAV imagery with large vision–language models. ISPRS Journal of Photogrammetry and Remote Sensing, 232:831–847, 2026. doi: 10.1016/j.isprsjprs.2026.01.005.

[128] Chenchen Zhu, Fanyi Xiao, Andres Alvarado, Yasmine Babaei, Jiabo Hu, Hichem El-Mohri, Sean Culatana, Roshan Sumbaly, and Zhicheng Yan. EgoObjects: A Large-Scale Egocentric Dataset for Fine-Grained Object Understanding. In ICCV, pages 20110–20120, 2023.

[129] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable DETR: Deformable Transformers for End-to-End Object Detection. In ICLR, 2021.

## Appendix

This appendix describes the training corpus of WeVisOne-VL and provides implementation details.

## A WeVisOne-VL Training Corpus

We describe the data used for SFT, segmentation and matting training, and RL below.

## A.1 SFT Data

Table 9 summarizes our SFT data by task and annotation source, including public annotations and annotations generated by the WeVisOne-VL Data Engine (WDE).

Table 9 Training data volumes across the three SFT stages (M).
<table><tr><td>Task</td><td>Source</td><td>Datasets</td><td>S1</td><td>S2</td><td>S3</td></tr><tr><td rowspan="2">Detection</td><td>Public</td><td>Objects365 [89], OpenImages-V6 [47], V3Det [102], DeepFashion2 [24], ODinW-35 [52], EgoObjects [128], PACO [82], PartImageNet++ [59], BDD100K [119], nuImages [74], VisDrone [19], CrowdHuman [88], SKU-110K [26], PartImageNet [31], MOT20Det [14], MOT17Det [15], OWDOD [93], FSCD-147 [75]</td><td>18.4</td><td>16.2</td><td>8.4</td></tr><tr><td>WDE</td><td>CC3M, GRIT, LAION, SA-1B</td><td>42.2</td><td>10.7</td><td>7.2</td></tr><tr><td rowspan="2">Referring</td><td>Public</td><td>RefCOCO [120], gRefCOCO [33], Flickr30k [79], HumanRef-CoT [40], HumanParts [60], D3 [112], OpenRef [109], Unsplash [104]</td><td>6.3</td><td>2.1</td><td>1.0</td></tr><tr><td>WDE</td><td>GRIT, CC3M, LAION</td><td>11.8</td><td>5.5</td><td>1.6</td></tr><tr><td>Counting</td><td>Public</td><td>CLOC [51], MixCount [20], KubriCount [66], PixMo [13], OWDOD [93], FSC-147 [83], FSCD-147 [75]</td><td>2.4</td><td>1.2</td><td>0.6</td></tr><tr><td>Pointing</td><td>Public</td><td>PixMo [13], Molmo2 [10], CLOC [51], FSC-147 [83]</td><td>2.8</td><td>0.9</td><td>0.4</td></tr><tr><td>OCR</td><td>Public</td><td>IDL-OCR [2], BLIP3-OCR [113], ArT [9], COCO-Text [101], LSVT [99], TextOCR [95], ReCTS [124], HierText [71], RCTW [92], ICDAR2015 [43], WildReceipt [96], SROIE [35], ICDAR2013 [42]</td><td>2.1</td><td>0.7</td><td>0.3</td></tr><tr><td>Caption</td><td>Public</td><td>RefSpatial [126], Inst-IT [77], VG-PhraseCut [108], Flickr30k [118], COCONut-PanCap [16], PixMo [13], LLM-Seg40K [103], PanCap, Rea2Seg [23]</td><td>2.7</td><td>0.9</td><td>0.4</td></tr><tr><td rowspan="2">Segmentation</td><td>Public</td><td>RefCOCO [120], RefCOCO+ [120], RefCOCOg [72], ReVOS [114], Ref-YouTube-VOS [87]</td><td>0.9</td><td>0.9</td><td>1.6</td></tr><tr><td>WDE</td><td>CC3M, GRIT, LAION, SA-1B</td><td>0.6</td><td>1.4</td><td>20.3</td></tr><tr><td colspan="4">Total</td><td>90.2 40.5</td><td>41.8</td></tr></table>

## A.2 Segmentation and Matting Data

Table 10 summarizes the data sources for segmentation and matting training. Public matting datasets provide high-resolution images with high-quality alpha mattes. During matting fine-tuning, we sample public and WDE-annotated data at a ratio of 1:4 within each batch.

Table 10 Data sources and sample counts for segmentation and matting (M).
<table><tr><td>Task</td><td>Source</td><td>Datasets</td><td>Size (M)</td></tr><tr><td rowspan="2">Segmentation</td><td>Public</td><td>SA-1B [45], RefCOCO [120], RefCOCO+ [120], RefCOCOg [72], ReVOS [114], Ref-YouTube-VOS [87]</td><td>171.4</td></tr><tr><td>WDE</td><td>CC3M, GRIT, LAION</td><td>288.4</td></tr><tr><td rowspan="2">Matting</td><td>Public</td><td>AM-2K [56], Distinctions-646 [80], UHRIM [64], HHM-50K [98], IHIM-50K [36], RefMatte [57], AIM-500 [55], CelebA-Hair [11], Transparent-460 [3], SIMD [97], P3M-10K (train) [54],</td><td>0.3</td></tr><tr><td>WDE</td><td>Real-Human [105], RealWorldPortrait-636 [121] CC3M, SA-1B</td><td>129.8</td></tr></table>

## A.3 Reinforcement Learning Data

We apply uncertainty-aware sample selection (Section 4) to the SFT data using the SFT checkpoint to construct the RL training set. The final RL set contains 178K examples: 44.0% detection, 20.8% dense detection, 17.5% referring localization, 7.9% OCR, 5.8% referring segmentation, and 4.0% pointing. Table 1 shows the complete 178K-example RL task distribution, including the additional localization and pointing examples.

## B Implementation Details

## B.1 Training Hyperparameters

Table 11 summarizes the hyperparameters that difer across the three SFT stages and RL training. SFT-S3 refers to joint token–pixel training. We use 128 GPUs for SFT and 160 GPUs for RL. All four configurations use AdamW with a weight decay of 0.01, a maximum gradient norm of 1.0, and BF16 precision. We use cosine learning-rate decay for SFT and constant or cosine schedules for RL, depending on the configuration. For SFT, we use sequence packing and specify batch sizes by the number of tokens per GPU. OCR examples in SFT-S2 and SFT-S3 use 16–256 visual tokens to accommodate long transcriptions.

Table 11 Training hyperparameters for SFT and RL. VLM comprises the vision encoder, projector, and LLM. Dec. denotes the pixel decoder. A dash in the learning-rate rows indicates a frozen or inactive module. SFT-S3 uses LoRA for the LLM (r = 128, α = 256, dropout 0.05), with fully trainable token embeddings and output head. The other modules are fully trainable. For RL, the maximum prompt and response lengths are both 4,096 tokens.
<table><tr><td>Hyperparameter</td><td>SFT-S1</td><td>SFT-S2</td><td>SFT-S3</td><td>RL</td></tr><tr><td>Trainable modules</td><td>LLM</td><td>VLM</td><td>VLM + Dec.</td><td>LLM</td></tr><tr><td>Learning rate: LLM</td><td> $2 \times 1 0 ^ { - 5 }$ </td><td> $5 \times 1 0 ^ { - 6 }$ </td><td> $5 \times 1 0 ^ { - 6 }$ </td><td> $1 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Learning rate: vision encoder</td><td></td><td> $5 \times 1 0 ^ { - 7 }$ </td><td> $5 \times 1 0 ^ { - 6 }$ </td><td></td></tr><tr><td>Learning rate: projector</td><td></td><td> $5 \times 1 0 ^ { - 6 }$ </td><td> $5 \times 1 0 ^ { - 6 }$ </td><td></td></tr><tr><td>Learning rate: pixel decoder</td><td></td><td></td><td> $1 \times 1 0 ^ { - 5 }$ </td><td></td></tr><tr><td>Warmup ratio</td><td>5%</td><td>5%</td><td>3%</td><td></td></tr><tr><td>Training steps</td><td>13,000</td><td>8,995</td><td>9,062</td><td>593</td></tr><tr><td>Max. sequence length</td><td>4,096</td><td>8,192</td><td>8,192</td><td>4,096 + 4,096</td></tr><tr><td>Visual tokens / image</td><td>16-512</td><td>64-1,960</td><td>64-1,960</td><td>64-1,960</td></tr><tr><td>Token budget / GPU</td><td>28,672</td><td>24,576</td><td>24,576</td><td>24,576</td></tr><tr><td>Distributed training</td><td>ZeRO-2</td><td>ZeRO-2</td><td>ZeRO-2</td><td>FSDP2 + vLLM</td></tr></table>

## B.2 Pixel Decoder Training

Warm-up. After SFT-S2, we connect the VLM to a randomly initialized pixel decoder and jointly train them for 100,000 steps with segmentation supervision. The decoder learning rate is $1 \times 1 0 ^ { - 6 }$ with 1,000 warmup steps. The decoder predicts masks at 512×512 resolution. After warm-up, we retain only the trained decoder and <OBJ> embedding and attach them to the original SFT-S2 checkpoint for joint training. The other VLM weights updated during warm-up are discarded.

Joint training. During SFT-S3, we alternate NTP and segmentation steps at a 1:1 ratio to jointly train the SFT-S2 checkpoint with the warmed-up decoder. The VLM and decoder learning rates are $5 \times 1 0 ^ { - 6 }$ and $1 \times 1 0 ^ { - 5 }$ , respectively, with a 3% warmup ratio. For each packed sequence on each GPU, we randomly sample 24 <OBJ> tokens and pass their corresponding representations to the decoder for segmentation prediction, while retaining the full text sequence for NTP.

Matting fine-tuning. After training is complete, we freeze the VLM and fine-tune the decoder for matting for 100,000 steps at an output resolution of 1024 × 1024. We use a learning rate of $5 \times 1 0 ^ { - 7 }$ for the decoder and $2 . 5 \times 1 0 ^ { - 6 }$ for the coarse-to-fine refinement branch, with 200 warmup steps. Each GPU processes up to eight <OBJ> tokens. The matting loss is defined in Section 4.

The warm-up and matting phases use AdamW with cosine learning-rate decay, a weight decay of 0.05, and a maximum gradient norm of 5.0. Both phases use BF16 precision and DDP, with a token budget of 98,304 per GPU, a maximum sequence length of 8,192, and 16–1,024 visual tokens per image.

## B.3 RL Rewards

For detection, dense detection, and referring localization, we use the grounding reward defined in Section 4. For segmentation and matting, RL also uses only the box-based grounding reward; no reward is computed from the predicted masks or alpha mattes. For pointing, we compute F1 through one-to-one matching between predicted points and ground-truth masks, counting a match when the point lies inside the mask. Referring segmentation requires each predicted box to be followed by an <OBJ> token, while OCR uses its transcription reward. When the target is absent, the reward is 1 only if the model predicts no object.