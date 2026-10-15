# Read, Parse, Describe: Unified Document Parsing with Visual Element Description Generation

Yuheng Chen<sup>1,∗</sup>, Yufan Chen<sup>1,∗,†</sup>, Zhuojun Cai<sup>1</sup>, Ruiping Liu<sup>1</sup>, Junwei Zheng<sup>1</sup>, Jiale Wei<sup>1</sup> Di Wen<sup>1</sup>, Weijia Fan<sup>1,3</sup>, Kunyu Peng<sup>1</sup>, Jiaming Zhang<sup>2</sup>, Rainer Stiefelhagen<sup>1</sup> <sup>1</sup>Karlsruhe Institute of Technology <sup>2</sup>Hunan University <sup>3</sup>University of Alberta yuheng.chen@student.kit.edu,yufan.chen@kit.edu

![](images/b4d4b3f76b3afbf127c4f5506adc44cddef387d239d7c0907cab120f7f254fb5.jpg)  
<sup>p</sup> <td>Clarifies the boundaries, as umptions, and key areas <sub>covered.</td> </sub>(a) Unified document parsing with different document elements.

![](images/c10847f0312f60ce8072fd342a7f6c83b8fabcd09873d48055b0a04eb8acfd0f.jpg)  
(b) Domain distribution.  
<sup>></Content></sup>Figure 1: Read-Parse-Describe Dataset Overview. Figure 1a shows the unified parsing contract converts documents into structured outputs with localized elements, reading order, textual content, and visual descriptions. Figure 1b shows the domain distribution of the 2,402 source documents.

## Abstract

Document understanding systems are typically evaluated as separate components via layout analysis, optical text recognition, reading-order prediction, and image captioning. This separation leaves scientific figures, captions, and visual semantics weakly connected, and fails to measure whether a model can produce a single page-level parse that is both structurally faithful and semantically accessible. We introduce Read-Parse-Describe, a unified dataset for document parsing with visual element description generation. Given a page image, models are required to output an ordered list of layout elements containing bounding boxes, semantic labels, reading-order indices, recognized textual content, and alternative text descriptions for visual elements. The benchmark is built from 2,402 documents, comprising 18,163 continuous pages, 125,218 layout elements, and 11,338 visual instances. Besides, we further propose a plug-and-play space-awareness enhancer that reuses multi-layer visual features and injects grid-aware spatial guidance into visionlanguage tokens for layout-aware generation. Experiments across end-to-end VLMs and pipeline parsers show that existing systems still struggle to jointly recover reading order and visual semantics, while our enhanced model achieves strong joint performance. These results highlight the need to move beyond isolated document parsing modules toward unified models that jointly read, parse, and describe complex documents. The new dataset and code will be made publicly available at RPD Project.

## 1 Introduction

Scientific documents are not just sequences of recognized words. A single page may contain twocolumn prose, section headers, equations, tables, figures, captions, footnotes, author metadata, and visual evidence whose meaning is not recoverable from optical character recognition (OCR) alone. Document AI has therefore progressed through layout analysis [1, 2, 3, 4], table recognition [5], document VQA [6], OCR-free parsing [7, 8], and multimodal document models [9]. In accessibility and machine-readable paper ingestion, the useful output is not an isolated layout map or a free-form page summary, but a structured page representation that can be read, reflowed, indexed, and audited.

Existing benchmarks usually evaluate pieces of this representation in isolation. Layout datasets measure region localization [10, 11, 12], OCR and table benchmarks [13] measure textual structure, reading-order datasets [14] focus on sequencing, captioning, and alt-text datasets [15, 16] describe isolated or context-conditioned images. These tasks are necessary, but they do not test whether a system can produce a single consistent parse in which boxes, labels, reading order, text, tables, captions, and visual descriptions agree with each other. This gap is particularly visible in scientific PDFs, where captions frequently refer to nearby figures or tables, equations interrupt prose, and visual elements carry technical information that should be exposed to downstream readers.

We introduce Read-Parse-Describe, a unified task and dataset for unified document parsing with visual element description generation as shown in Fig. 1. Given a page image, a model must output an ordered list of elements, which includes a bounding box, semantic category, reading-order index, recognized text content, an explicit caption link when applicable, and an alternative-text description for visual elements. The key novelty is not any single field, but enforcing their agreement in one auditable page parse. The unified output contract can be implemented by either end-to-end models or modular systems. The benchmark contains 2,402 open-access scientific documents, 18,164 continuous pages, 125,218 layout elements, and 11,338 visual instances.

Modern VLMs and document-specialized models [17, 18, 19, 20, 21, 22, 23, 24] increasingly use frozen or partially frozen visual encoders. When such models merge visual tokens before generation, local spatial evidence can be weakened exactly where element extents, reading order, and captionobject alignment are required. We therefore introduce SARA, a lightweight Spatial Alignment Residual Adapter for end-to-end VLM parsing. SARA freezes the base VLM, reuses intermediate visual encoder layers, projects their concatenated features into the token space, aligns with crop-aware spatial grids, and injects into the native merged tokens. The task makes spatial-alignment failures measurable, while SARA tests a direct architectural response without changing labels or evaluation. To summarize, our contributions are as follows:

• We introduce Read-Parse-Describe, a novel task and dataset for unified document parsing with alternative-text generation into a shared structured schema.

• We propose SARA, a Spatial Alignment Residual Adapter that injects crop-aware spatial cues into VLM tokens to improve layout-sensitive document understanding.

• We establish a unified evaluation protocol covering both document parsing and alternativetext generation, enabling systematic assessment of parsing fidelity and generation quality.

• We conduct extensive experiments and ablation studies, showing that crop-aware spatial alignment consistently benefits unified document parsing and alternative-text generation.

## 2 Related Work

## 2.1 Document Layout and Parsing Datasets

Document-layout benchmarks have made document understanding measurable by isolating the output. Early resources, such as RVL-CDIP [25], treat documents as classes, whereas FUNSD [26] focuses on form. CORD [27] and SROIE [28] further show that domain-specific schemas enhance extraction but specialized. Similar to table, SciTSR [29] targets scientific table structure, and PubTables-1M [13] scales table recognition, but neither requires the table, caption, reading position, and visual description to agree in one element record. PubLayNet [10] made large-scale layout detection practical, and DocBank [11] expand token-level layout supervision. DocLayNet [12] improved further with richer human annotations, while M6Doc [30] expanded layout analysis across heterogeneous documents. For robustness, RoDLA [4] stresses layout robustness, whereas ReadingBank [14] isolates reading-order prediction. Broader evaluations such as DocVQA [31] and DUDE [32] test document understanding through answers rather than page parses. OmniDocBench [33] moves closer to document parsing, while MMLongBench-Doc [34] stresses long-context document understanding.

## 2.2 Document VLMs and Pipeline Systems

Structure-aware representations are often precise, while generative systems are flexible but low performance. LayoutLM [1] and LayoutLMv2 [2] made layout-aware pretraining central for document understanding. LayoutLMv3 [3] further unifies text and image modeling through task-specific heads rather than complete generator. DocFormer [35] and VGT [36] emphasize structure-aware encoding, reinforcing the explicit geometry. OCR-free generation such as Donut [6] can emit structured text directly from an image, while Pix2Struct [37] support screenshot-style pretraining. Nougat [38] specializes this generative view for academic documents, and GOT-OCR [7] extends OCR-style generation toward broader text recognition. Recent document parsers push toward full-page extraction, e.g., DeepSeek-OCR [8] uses visual compression for OCR, while Dolphin-v2 [39] illustrates universal document parsing. MonkeyOCR [40] emphasizes structure-recognition-relation decomposition, and PaddleOCR-VL [41] targets compact multilingual document parsing. MinerU [42] takes a more inspectable pipeline route. General VLMs provide another axis of flexibility, e.g., BLIP-2 [18] and InstructBLIP [19] align visual features with language models, while LLaVA [20] and Qwen2.5-VL [23] show strong instruction-following multimodal generation. Agentic parsing is also studied in ParseFixer [43], which applies selective correction to backbone Markdown outputs, and ParseBench [44], which evaluates agent-relevant parsing capabilities on enterprise documents.

## 2.3 Visual Descriptions and Accessibility

Accessibility work shows why visual description cannot be reduced to generic captioning. Conceptual Captions [15] demonstrates the scale available in web alt text, but weakly tied to accessibility intent. VizWiz [45] and VizWiz-Captions [46] foreground blind-user scenarios, where the useful answer depends on context. TextCaps [47] shows that descriptions may depend on visible text, and OCR-VQA [48] makes text reading part of visual question answering. Screen2Words [49] shifts the target to UI screens, while Alt-Text with Context [16] conditions descriptions on surrounding social text. Furthermore, ChartQA [50] requires visual and logical reasoning, and DePlot [51] recovers chart data before reasoning. Chart-to-Text [52] directly studies chart description, while PlotQA [53] emphasizes quantitative plot understanding. A useful description must be tied to the correct object, caption, page position, and surrounding reading flow. Read-Parse-Describe therefore evaluates alternative text as a field in the parse rather than as an isolated captioning output.

## 3 Read-Parse-Describe Dataset

## 3.1 Task Definition

We introduce the Read-Parsing-Describe, a dataset for document parsing with grounded visual description. This task is motivated by two limitations of existing document understanding benchmarks as shown in Tab. 1. First, many layout datasets primarily focus on the physical localization of page regions but do not require models to recover a complete logic-aware document representation. This limitation is especially pronounced in scientific papers, where frequently break simple top-to-bottom ordering rules. Second, visual elements are often treated as layout categories rather than semantic objects. Figures, charts, diagrams, screenshots may be localized, but the information they convey is rarely required to be presented in a grounded, accessible form.

## 3.2 Output Schema

Read-Parsing-Describe therefore evaluates document understanding as a unified multi-modal parsing problem. The target output is an ordered list of document elements. Each element contains its bounding box, semantic category, reading-order index, recognized content when applicable, caption linkage, and visual description when applicable:

$$
Y = [ e _ { 1 } , e _ { 2 } , \ldots , e _ { n } ] ,\tag{1}
$$

Table 1: Comparison of dataset target contracts. A checkmark indicates that the benchmark explicitly requires the corresponding field as part of its evaluation target. Caption Links refer to explicit associations between visual objects and their captions. Structured Output denotes a single pagelevel prediction that jointly contains layout, order, textual content, and caption links.
<table><tr><td>Dataset</td><td>Docs</td><td>Layout</td><td>Reading Order</td><td>Textual Elements</td><td>Visual Elements</td><td>Links</td><td>Caption Alternative Structured Text</td><td>Output</td></tr><tr><td>PubLayNet [10]</td><td>PDF</td><td>√</td><td></td><td></td><td>√</td><td></td><td></td><td></td></tr><tr><td>DocLayNet [12]</td><td>Mixed</td><td>√</td><td></td><td></td><td>√</td><td></td><td></td><td></td></tr><tr><td>DocBank [11]</td><td>PDF</td><td>√</td><td></td><td></td><td>√</td><td></td><td></td><td></td></tr><tr><td>M6Doc [30]</td><td>Mixed</td><td>√</td><td></td><td></td><td>√</td><td></td><td></td><td></td></tr><tr><td>D4LA [36]</td><td>Mixed</td><td>√</td><td></td><td></td><td>√</td><td></td><td></td><td></td></tr><tr><td>SciTSR [29]</td><td>Images</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ReadingBank [14]</td><td>Mixed</td><td>√</td><td>L</td><td>√</td><td></td><td></td><td></td><td></td></tr><tr><td>DocVQA [31]</td><td>Images</td><td></td><td></td><td>√</td><td>√</td><td></td><td></td><td></td></tr><tr><td>ChartQA [50]</td><td>Images</td><td></td><td></td><td></td><td>√</td><td></td><td></td><td></td></tr><tr><td>InfographicVQA [54]</td><td>Images</td><td></td><td></td><td></td><td>√</td><td></td><td></td><td></td></tr><tr><td>DUDE [32]</td><td>Mixed</td><td></td><td></td><td></td><td>√</td><td></td><td></td><td></td></tr><tr><td>OmniDocBench [33]</td><td>Mixed</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td></td><td>√</td></tr><tr><td>Read-Parse-Describe (Ours)</td><td>PDF</td><td>√</td><td>√</td><td></td><td>√</td><td>√</td><td></td><td>√</td></tr></table>

where each element is represented as

$$
e _ { i } = \{ { \bf i d } , { \bf b b o x } , { \bf c a t e g o r y } , { \bf o r d e r } , { \bf t e x t } , { \bf h t m 1 } , { \bf c a p t i o n } _ { - } { \bf i d } , { \bf a l t } _ { - } { \bf t e x t } \} ,\tag{2}
$$

where id is the element identifier, bbox is a page-coordinate box $[ x _ { 1 } , y _ { 1 } , x _ { 2 } , y _ { 2 } ]$ in the rendered image coordinate system. category is the semantic layout label, and order is the canonical human reading-order index. The submitted list is expected to be sorted by order, i.e., if list position and order disagree, evaluation first normalizes by the order field and leaves duplicate or missing indices to be penalized by matching and sequence metrics. text stores recognized text for textual regions, html stores table structure, caption\_id links a visual object to caption, and alt\_text stores a factual visual description. Non-applicable fields are set to null.

Terminology. An ordered schema is the full element list Y after sorting by the reading-order index. The central design choice is to make fields agree inside the same element object, not to score familiar subtasks in isolation. The contract has four coupled decisions:
<table><tr><td>Coupling</td><td>Design reason</td><td>Failure exposed</td></tr><tr><td>Element identity</td><td>Bind id, category, textual content, caption links, and descriptions to the same object.</td><td>Free-floating OCR, HTML, captions, or descriptions cannot be cred- ited without the element they claim to describe.</td></tr><tr><td>Geometry as evidence</td><td>and visual descriptions.</td><td>Treat bbox as the audit handle for text, tables, A plausible description or table string attached to the wrong crop becomes measurable rather than silently accepted.</td></tr><tr><td>Canonical reading flow</td><td>than visual-nearest-neighbor order.</td><td>Use order as human reading order rather Two-column text, spanning figures, and separated captions expose sequence errors even when OCR is correct.</td></tr><tr><td>Caption semantics</td><td>sual instances when applicable.</td><td>Require caption_id and alt_text for vi- A model must bind the object, caption, and description instead of generating generic image captions.</td></tr></table>

The couplings above make the task a test of spatially grounded binding. Region proposals decide the visual evidence available to OCR, table serialization, caption linkage, and alt-text generation. Reading-order decisions decide where captions and objects sit in the page narrative; and visual descriptions are auditable only when their evidence is localized.

## 3.3 Dataset Creation Pipeline

We collect open-access PDF papers from the ACM Digital Library and retain document metadata. Since the main contributions, core figures, methods, and experimental results of scientific papers are usually concentrated in the main body, we extract the first eight pages of each paper and keep shorter papers in their entirety. This page-level curation strategy preserves structurally and semantically informative regions of papers while reducing the dominance of long reference sections and appendices. To focus annotation on the dense parts of papers while reducing long reference sections, we retain the first eight pages of each paper and keep shorter papers in full. Additional detailed dataset statistics would be included in the technical appendix.

The visual elements include tables, diagrams, screenshots, photographs, charts, and long-tail illustrations. This diversity matters because visual-description generation is not the same problem for each object type, e.g., a UI screenshot requires interface semantics. The layout labels are similarly heterogeneous, covering structural headers, scholarly objects, captions, references, lists, and paragraphs. We keep these categories explicit to avoid hiding errors that make scientific PDFs hard to parse. We split at the paper level with a 90:5:5 train/validation/test ratio after deduplication. Exact duplicate PDFs are removed using SHA-256 hashes, DOI metadata is used when available, normalized titles are matched otherwise, and near-duplicate title/abstract pairs above the similarity threshold are manually inspected. Paper-level splitting is important because adjacent pages share templates, section styles, figure numbering, author metadata, and sometimes repeated visual objects. Contrast a page-level split would overstate generalization for reading order and caption linkage.

## 3.4 Annotation Pipeline

Annotations are produced with a bootstrap-and-correct pipeline designed around cross-field consistency. A document parser first initializes layout boxes and labels. The generated boxes, semantic categories, caption links, and reading-order indices are then manually inspected and corrected together, rather than as independent layers. Corrections target common scientific-PDF failure modes: two-column reading order, spanning figures, captions separated from visual objects, footnotes, equations, author metadata, and fine-grained heading levels. This joint correction step is important: moving a box can change its text crop, its caption candidate, and its reading-order position.

Visual instances then receive alternative-text annotations through a triage loop: generate an initial description, score it against the accessibility rubric, refine low-score cases by human review, and audit the revised labels. Gemini 3 Pro [55] provides the initial descriptions and Gemini 3 Flash [56] scores factual accuracy, completeness, and prompt adherence on a 1–5 scale. The 1,408 descriptions scoring below 4 are routed to human refinement, then re-scored:

<table><tr><td>Score</td><td>Generated</td><td>Refined</td><td>Score</td><td>Generated</td><td>Refined</td></tr><tr><td>1</td><td>0</td><td>0</td><td>4</td><td>2,419</td><td>226</td></tr><tr><td>2</td><td>146</td><td>0</td><td>5</td><td>7,511</td><td>11,112</td></tr><tr><td>3</td><td>1,262</td><td>0</td><td>Score-5 rate</td><td>66.2%</td><td>98.1%</td></tr></table>

After refinement, the score-5 rate increases from 66.2% to 98.1%. A manual audit of 200 revised samples with previous scores below 5 gives 194/200 exact agreement between human and LLM scores. The technical appendix includes

the prompt, scoring rubric, full score distribution, and audit matrix for this annotation stage.

The alt-text process is designed for benchmark consistency, not as an automatic accessibility deployment pipeline. The prompt asks for concise factual descriptions and discourages speculation beyond visible evidence. The scoring step checks whether the description is faithful, complete enough for the object type, and aligned with the prompt; human refinement is concentrated on low-score cases because these are most likely to contain missing relationships, hallucinated labels, or vague descriptions. We expose this procedure because visual-description labels are harder to audit than boxes or OCR strings, and the reader should be able to separate measured model behavior from annotation assumptions.

## 4 SARA: Spatial Alignment Residual Adapter

Read-Parse-Describe turns parsing into a binding problem, i.e., each generated element must keep the attribute attached to the same page evidence. This creates a different pressure from ordinary VLM generation. A visual concatenation is useful due to the compression of image tokens into a sequence the decoder can consume, but that compression can blur the local spatial cues needed to decide where an element starts, how two-column regions should be ordered, and which caption belongs to which object. The design goal of SARA is therefore not to replace the base VLM, but to restore crop-level spatial evidence at the point where the decoder receives merged visual tokens. The key design choice is to preserve correspondence between each crop’s pre-merger grid and its native post-merger tokens, enabling aligned feature re-injection without ground-truth input boxes.

![](images/02ea2aaca16622c5a23b28d53c9a6d4fd11558315e192f098203b53d7ecb7714.jpg)  
Figure 2: Document parsing model with SARA adapter, which is a plug-and-play spatial enhancer that leverages multi-layer visual features. Features from encoder layers are concatenated, followed by a grid-aware multi-crop alignment mechanism that preserves spatial structure. The aligned features serve as spatial guidance and are injected into the original tokens via a gated residual connection.

## 5 Method

SARA follow threefold design as shown in the Fig. 2: (1) Reuse intermediate visual states rather than only the final merged tokens, because earlier layers retain finer layout and boundary information. (2) Align the reused states to the same crop grid as the native visual merger, so the residual signal has the same token-level support as the decoder’s original visual stream. (3) Inject the aligned guidance through a learnable residual gate, preserving the pretrained visual-language pathway while letting training decide how much spatial correction is useful.

Multi-layer spatial features. Let $\mathcal { S } = \{ 2 , 5 , 8 , 1 1 \}$ be the selected visual encoder layers. During the forward pass, hooks collect hidden states

$$
\begin{array} { r } { \pmb { X } ^ { ( l ) } \in \mathbb { R } ^ { L _ { v } \times d _ { v } } , \quad l \in \mathcal { S } , } \end{array}\tag{3}
$$

where $L _ { v }$ is the pre-merger visual-token length and $d _ { v }$ is the visual feature dimension. We concatenate these states along the channel dimension:

$$
X _ { \mathrm { c a t } } = \mathrm { C o n c a t } \left( X ^ { ( 2 ) } , X ^ { ( 5 ) } , X ^ { ( 8 ) } , X ^ { ( 1 1 ) } \right) .\tag{4}
$$

The concatenated feature is mapped to the merger dimension $d _ { m }$ by a small projector,

$$
\boldsymbol { G } = \boldsymbol { W } _ { 2 } \sigma \left( \mathrm { L N } ( \boldsymbol { W } _ { 1 } \boldsymbol { X } _ { \mathrm { c a t } } ) \right) ,\tag{5}
$$

where LN is LayerNorm and σ is GELU.

Crop-aware alignment. The visual merger reduces spatial resolution, so the guidance $G$ must be aligned with the native merged tokens. For each crop with grid metadata $( t _ { i } , h _ { i } , w _ { i } )$ , we slice the corresponding segment from $G ,$ reshape it into its two-dimensional grid, and average-pool to the merged-grid resolution:

$$
\begin{array} { r } { G _ { i } ^ { \prime } = \mathrm { P o o l } _ { 2 D } \left( \mathrm { R e s h a p e } ( G _ { i } ) , ( h _ { i } / 2 , w _ { i } / 2 ) \right) . } \end{array}\tag{6}
$$

All crop-level features are flattened and concatenated into $G ^ { \prime } \in \mathbb { R } ^ { L _ { m } \times d _ { m } }$ , matching the length and dimension of the original merged visual tokens T.

Residual injection. The enhanced visual tokens are

$$
\hat { \pmb { T } } = \pmb { T } + \lambda \pmb { G } ^ { \prime } ,\tag{7}
$$

where λ is a learnable scalar gate initialized to 1.0. The language decoder then generates the same ordered schema used by all baselines. SARA changes only the visual token pathway; it does not alter the benchmark labels, target schema, or evaluation metrics.

Shape summary and scope. The data flow is $\mathbf { } \mathbf { } X ^ { ( l ) } \in \mathbb { R } ^ { L _ { v } \times d _ { v } } , \mathbf { } X _ { \mathrm { c a t } } \in \mathbb { R } ^ { L _ { v } \times | \mathcal { S } | d _ { v } } , G \in \mathbb { R } ^ { L _ { v } \times d _ { m } }$ and $G ^ { \prime } , T , \hat { T } \in \mathbb { R } ^ { L _ { m } \times d _ { m } }$ after crop-aware pooling. SARA is not intended to replace documentspecific supervision or table-aware decoding. The residual form keeps the original visual-token stream intact and lets training learn the aligned guidance.

## 6 Experiments

## 6.1 Settings and Metrics

We evaluate three categories of systems: general-purpose VLMs, specialized document models, and pipeline parsers. General-purpose VLMs include Qwen3.5 [57], Gemma4 [58], ChatGPT-5.4 [59], ChatGPT-5.6 [60], Gemini 3.6 Flash [61] and Gemini 3.1 Pro [62]. Specialized document models include DeepSeek-OCR [8], DeepSeek-OCR2 [63], Dolphin-v2 [39], MonkeyOCR [40], VGT [36], GOT-OCR 2.0 [7], Nougat [38], and PaddleOCR-VL [41]. Pipeline parsers include PP-StructureV3 [64] and MinerU [42].

All systems are evaluated under the same page-level output schema. Zero-shot systems use released checkpoints or official APIs without in-context examples. Full fine-tuning baselines serialize each target element list into text and optimize the standard autoregressive next-token objective on the training split. The validation split is used for model selection, and final results are reported on the held-out test split. For SARA runs, the base VLM is frozen and only the adapter pathway is trained. For generative VLMs, we use deterministic decoding whenever supported. Outputs from pipeline systems are converted into the unified schema.

Metrics. All metrics are computed after schema conversion. Minor syntax errors are repaired deterministically; if a prediction remains unparsable, the affected fields are treated as empty. F1 follows the IoU-matching convention used by PubLayNet [10], DocLayNet [12], and RoDLA [4]: a match requires the same category label and IoU ≥ 0.5. mIoU is computed over matched true positives. Reading-order (RO) score uses normalized edit-distance similarity over matched element identifiers [14]. OCR accuracy is computed as normalized character-level edit-distance similarity after text normalization. For visual descriptions, we report METEOR [65] and BERTScore [66].

## 6.2 Main Results

Table 2 should be read as a contract diagnostic, not a leaderboard. The unified schema requires models to jointly recover localized elements, reading order, textual content, and visual descriptions. Pipeline parsers often retain strong OCR or localization performance, but they usually leave caption links and alt-text fields incomplete. General-purpose VLMs can produce richer schema fields, but they are less reliable in spatial grounding without adaptation. Document-specialized VLMs and SARA are complementary: the former excel at specialized parsing, while SARA improves the structural fields of a frozen general-purpose VLM and maintains competitive visual-description quality.

Table 2 reveals different strengths across the evaluated systems and reported metrics. Fine-tuned Dolphin-v2 achieves the strongest localization performance, reaching 0.9826 F1 and 0.9588 mIoU. It also obtains the highest RO score, 0.9857. Fine-tuned DeepSeek-OCR achieves the highest METEOR, 0.3998, while GPT-5.6 achieves the highest BERTScore, 0.9004. DeepSeek-OCR also maintains a high RO score of 0.9825. Among the evaluated proprietary zero-shot models, GPT-5.6 obtains the highest F1, RO, and description scores, while Gemini 3.6 Flash obtains the highest mIoU. Including Gemini 3.1 Pro extends the comparison to the requested Pro-series model under the same schema. These results suggest that spatially precise parsing and open-ended visual description remain partially distinct capabilities.

Pipeline parsers remain competitive on some low-level fields but are less complete under the unified output schema. For example, MinerU reaches 0.9671 OCR accuracy and 0.9151 mIoU, but its RO score is 0.6757, and it does not produce visual alternative text under the evaluated schema. This does not mean that pipeline systems are ineffective. Instead, it identifies the fields that require additional modules or tighter integration when the target is a complete accessible page parse.

Compared with the Qwen3.5-2B full fine-tuning baseline, SARA substantially improves structural performance while keeping the base VLM frozen and training with 83.99M parameters. F1 increases to 0.9189, mIoU to 0.8919, OCR accuracy to 0.9636, and RO score to 0.9774. Its alt-text scores, 0.3834 METEOR and 0.8946 BERTScore, are also competitive with the strongest fine-tuned document VLMs. On Gemma4-E4B, SARA also improves F1 and mIoU relative to full fine-tuning, but RO and both description metrics decrease slightly. Compared with LoRA-only adaptation, SARA improves F1 and mIoU by 0.2185 and 0.2074 on Qwen and 0.1750 and 0.1898 on Gemma, respectively. Other metrics show trade-offs: Qwen LoRA has higher OCR accuracy, while Gemma LoRA has higher RO and slightly higher BERTScore.

Table 2: Main test-set results under the unified Read-Parse-Describe output schema. Rows are grouped by evaluation setting to separate Zero-shot Models, Full Fine-Tuning, which means finetuning the whole model, and Adapter Tuning, which means only fine-tuning the adapter part. -- indicates that the corresponding field is not produced by the evaluated system. bold and underlined scores denote the best and second-best results for each metric, respectively.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Trainable Params</td><td colspan="2">DLA</td><td rowspan="2">OCR Acc.↑</td><td rowspan="2">RO Score↑</td><td colspan="2">Alt. Text</td></tr><tr><td>F1↑</td><td>mIoU↑</td><td>METEOR↑</td><td>BERT ↑</td></tr><tr><td>Zero-shot Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GOT-OCR 2.0</td><td>- -</td><td>--</td><td>--</td><td>0.8652</td><td>0.5330</td><td>--</td><td>--</td></tr><tr><td>Nougat</td><td></td><td></td><td></td><td>0.7327</td><td>0.4640</td><td></td><td>- –</td></tr><tr><td>Dolphin-v2</td><td></td><td>0.9124</td><td>0.9265</td><td>0.9307</td><td>0.8221</td><td></td><td></td></tr><tr><td>MonkeyOCR</td><td></td><td>0.6030</td><td>0.8949</td><td>0.9684</td><td>0.6254</td><td></td><td></td></tr><tr><td>VGT</td><td></td><td>0.6795</td><td>0.8540</td><td>0.6304</td><td>0.3833</td><td></td><td></td></tr><tr><td>PaddleOCR-VL</td><td></td><td>0.7564</td><td>0.9458</td><td>0.9043</td><td>0.6619</td><td></td><td></td></tr><tr><td>PP-StructureV3</td><td></td><td>0.6375</td><td>0.8870</td><td>0.9558</td><td>0.5054</td><td></td><td></td></tr><tr><td>MinerU</td><td></td><td>0.7171</td><td>0.9151</td><td>0.9671</td><td>0.6757</td><td></td><td></td></tr><tr><td>ChatGPT-5.4</td><td></td><td>0.6978</td><td>0.6454</td><td>0.9796</td><td>0.7253</td><td>0.3506</td><td>0.8751</td></tr><tr><td>ChatGPT-5.6</td><td></td><td>0.7489</td><td>0.7781</td><td>0.9448</td><td>0.8967</td><td>0.3793</td><td>0.9004</td></tr><tr><td>Gemini 3.6 Flash</td><td></td><td>0.6909</td><td>0.8658</td><td>0.9544</td><td>0.8183</td><td>0.3744</td><td>0.8841</td></tr><tr><td>Gemini 3.1 Pro</td><td></td><td>0.7421</td><td>0.8551</td><td>0.9633</td><td>0.8337</td><td>0.3785</td><td>0.8961</td></tr><tr><td>Gemma4-E4B</td><td></td><td>0.5473</td><td>0.7674</td><td>0.8977</td><td>0.5256</td><td>0.1679</td><td>0.8455</td></tr><tr><td>DeepSeek-OCR</td><td></td><td>0.5987</td><td>0.6074</td><td>0.9064</td><td>0.5003</td><td>0.2562</td><td>0.8280</td></tr><tr><td>DeepSeek-OCR2</td><td>=-</td><td>0.5806</td><td>0.5963</td><td>0.8688</td><td>0.4830</td><td>0.2604</td><td>0.8332</td></tr><tr><td>Full fine-tuning</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3.5-0.8B</td><td>752.39M</td><td>0.7177</td><td>0.8345</td><td>0.9475</td><td>0.9605</td><td>0.3666</td><td>0.8904</td></tr><tr><td>Qwen3.5-2B</td><td>1881.83M</td><td>0.7336</td><td>0.8350</td><td>0.9408</td><td>0.9628</td><td>0.3741</td><td>0.8928</td></tr><tr><td>Gemma4-E4B</td><td>7463.01M</td><td>0.9358</td><td>0.8836</td><td>0.9452</td><td>0.9628</td><td>0.3047</td><td>0.8821</td></tr><tr><td>DeepSeek-OCR</td><td>2937.36M</td><td>0.9644</td><td>0.9492</td><td>0.9679</td><td>0.9825</td><td>0.3998</td><td>0.8968</td></tr><tr><td>DeepSeek-OCR2</td><td>2935.88M</td><td>0.8783</td><td>0.9189</td><td>0.9416</td><td>0.9231</td><td>0.3229</td><td>0.8094</td></tr><tr><td>Dolphin-v2</td><td>3085.93M</td><td>0.9826</td><td>0.9588</td><td>0.9751</td><td>0.9857</td><td>0.3396</td><td>0.8838</td></tr><tr><td>Adapter Tuning</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Gemma4-E4B + LoRA</td><td>103.82M</td><td>0.7687</td><td>0.7238</td><td>0.9164</td><td>0.9451</td><td>0.2871</td><td>0.8753</td></tr><tr><td>Gemma4-E4B + SARA</td><td>158.82 M</td><td>0.9437</td><td>0.9136</td><td>0.9474</td><td>0.9288</td><td>0.3009</td><td>0.8746</td></tr><tr><td>Qwen3.5-2B + LoRA</td><td>48.37M</td><td>0.7004</td><td>0.6845</td><td>0.9673</td><td>0.9304</td><td>0.3586</td><td>0.8652</td></tr><tr><td>Qwen3.5-2B + SARA</td><td>83.99 M</td><td>0.9189</td><td>0.8919</td><td>0.9636</td><td>0.9774</td><td>0.3834</td><td>0.8946</td></tr></table>

Interpreting the joint table. The main result is not that one model is uniformly best, but that single-axis strengths do not necessarily imply a coherent page parse. A model with high OCR but low RO may still fail to recover columns or reading sequences. A model with strong F1 but no alt-text may be useful as a front end, but it is incomplete as an accessible parser. Conversely, a model with strong alt-text but weak mIoU may describe objects that are poorly localized. Reporting these fields side by side makes such tradeoffs visible without imposing a deployment-specific ranking.

## 6.3 Ablations and Analysis

Table 3 isolates the contribution of each SARA component on Qwen3.5-2B. The full fine-tuning row is included as a reference, while all adapter rows keep the backbone frozen. Grid alignment alone provides a modest improvement over full fine-tuning, especially for RO and OCR. In contrast, projector-only training underperforms full fine-tuning, indicating that the gains do not come merely

Table 3: Component ablation with Qwen3.5-2B.
<table><tr><td>Configuration</td><td>Trainable Params</td><td>F1↑</td><td>mIoU↑</td><td>OCR Acc.↑</td><td>RO Score↑</td></tr><tr><td>Full FT</td><td>2218.4M</td><td>0.7336</td><td>0.8350</td><td>0.9408</td><td>0.9628</td></tr><tr><td>Grid align only</td><td>81.67M</td><td>0.7541</td><td>0.8440</td><td>0.9566</td><td>0.9762</td></tr><tr><td>Projector only</td><td>35.96M</td><td>0.6830</td><td>0.8139</td><td>0.9134</td><td>0.9262</td></tr><tr><td>Grid align + projector</td><td>83.99M</td><td>0.9189</td><td>0.8919</td><td>0.9636</td><td>0.9774</td></tr></table>

Table 4: Feature-layer selection ablation.
<table><tr><td>Layers</td><td>F1↑</td><td>mIoU↑</td><td>OCR Acc.↑</td><td>RO Score↑</td></tr><tr><td>{2,5}</td><td>0.6855</td><td>0.7682</td><td>0.9076</td><td>0.9372</td></tr><tr><td>{16, 18, 20, 22}</td><td>0.7221</td><td>0.8175</td><td>0.9256</td><td>0.9412</td></tr><tr><td>{2, 5, 8, 11}</td><td>0.9192</td><td>0.8968</td><td>0.9602 0.9736</td><td></td></tr></table>

from adding trainable parameters. The full configuration, which combines grid-aware alignment with projector training, produces the largest improvement across all four structural metrics.

Table 4 studies which visual feature layers are most useful for the adapter. Early-to-mid layers, {2,5,8,11}, outperform both shallow-only layers and late-layer selections. This pattern is consistent with the design goal of preserving local spatial cues while retaining sufficient semantic abstraction for document elements. On both tested backbones, SARA improves localization relative to full fine-tuning, while its effects on reading order and description similarity vary. Overall, these ablations suggest that SARA primarily acts as a spatial-structure adapter rather than as a general-purpose language-generation module. This also explains why alt-text improvements are more mixed than the localization and reading-order gains.

Training-data scale. On Qwen3.5-2B, SARA trails full fine-tuning on all four structural metrics at 10% training data and yields mixed results at 30%, while its structural advantage emerges at 50% and 100%. The complete comparison is reported in Appendix E.

## 6.4 Qualitative Results

Figure 3 compares annotation mapping results from the ground truth, the base Qwen3.5-2B model, full fine-tuning, and SARA. The base model often produces incomplete or poorly grounded mappings. Full fine-tuning improves schema compliance, but spatial alignment errors can remain. In comparison, SARA produces element mappings that are more consistent with the ground-truth page structure, especially for localized regions and their associated schema fields.

## 7 Conclusion

We introduced Read-Parse-Describe, a unified benchmark for scientific-document parsing, and SARA, a spatial adapter tailored to the same page-level output contract. The task jointly evaluates localized elements, labels, reading order, textual and table content, caption links, and alternative text within a single ordered parse. Experiments show that existing systems tend to fail different parts in evaluation, while frozen-backbone SARA tuning substantially improves structural parsing for a general VLM.

Limitations. Read-Parse-Describe is limited to open-access ACM scientific PDFs and primarily covers the first eight pages, leaving late appendices and references underrepresented. Alt-text annotations rely on LLM assistance with human refinement, and closed-source pre-training exposure cannot be fully excluded. The benchmark is intended to support accessible document understanding, search, and document mining, but generated accessibility metadata should not be treated as deployment-ready without further validation.

## Acknowledgments and Disclosure of Funding

This work was supported in part by Helmholtz Association of German Research Centers, in part by the Ministry of Science, Research and the Arts of Baden-Württemberg (MWK) through the Cooperative Graduate School Accessibility through AI-based Assistive Technology (KATE) under Grant BW6-03, and in part by the National Natural Science Foundation of China under Grant No. 62503166, in part by the Hunan Provincial Research and Development Project under Grant number 2026QK3018, and in part by the Yuelushan Industrial Innovation Center. This work was partially performed on the HoreKa supercomputer funded by the MWK and by the Federal Ministry of Education and Research, partially on the HAICORE@KIT partition supported by the Helmholtz Association Initiative and Networking Fund, and partially on bwForCluster Helix supported by the state of Baden-Württemberg through bwHPC and the German Research Foundation (DFG) through grant INST 35/1597-1 FUGG.

![](images/3ea2c792917b83d93467ca8cd6fd1b4d3197115d78cacf572649fb6a31e819ca.jpg)  
Figure 3: Qualitative comparison of annotation mapping results. All model-based methods use Qwen3.5-2B as the backbone. We compare the ground-truth annotations with outputs from the base model, full fine-tuning, and the proposed SARA adapter.

## References

[1] Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, and Ming Zhou. LayoutLM: Pre-training of text and layout for document image understanding. In Proceedings ofthe 26th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 1192–1200, 2020.

[2] Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, Min Zhang, and Lidong Zhou. LayoutLMv2: Multi-modal pre-training for visually-rich document understanding. In Proceedings ofthe 59th Annual Meeting ofthe Associationfor Computational Linguistics, pages 2579–2591, 2021.

[3] Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. LayoutLMv3: Pre-training for document AI with unified text and image masking. In Proceedings of the 30th ACM International Conference on Multimedia, pages 4083–4091, 2022.

[4] Yufan Chen, Jiaming Zhang, Kunyu Peng, Junwei Zheng, Ruiping Liu, Philip Torr, and Rainer Stiefelhagen. Rodla: Benchmarking the robustness of document layout analysis models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15556–15566, 2024.

[5] Xu Zhong, Elaheh ShafieiBavani, and Antonio Jimeno Yepes. Image-based table recognition: Data, model, and evaluation. In European Conference on Computer Vision, pages 564–580, 2020.

[6] Geewook Kim, Teakgyu Hong, Moonbin Yim, Jeongyeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. OCR-free document understanding transformer. In European Conference on Computer Vision, pages 498–517, 2022.

[7] Haoran Wei, Chenglong Liu, Jinyue Chen, Jia Wang, Lingyu Kong, Yanming Xu, Zheng Ge, Liang Zhao, Jianjian Sun, Yuang Peng, et al. General ocr theory: Towards ocr-2.0 via a unified end-to-end model. arXiv preprint arXiv:2409.01704, 2024.

[8] Haoran Wei, Yaofeng Sun, and Yukun Li. Deepseek-ocr: Contexts optical compression. arXiv preprint arXiv:2510.18234, 2025.

[9] Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Yuhao Dan, Chenlin Zhao, Guohai Xu, Chenliang Li, Junfeng Tian, Qian Qi, Ji Zhang, and Fei Huang. mPLUG-DocOwl: Modularized multimodal large language model for document understanding. arXiv preprint arXiv:2307.02499, 2023.

[10] Xu Zhong, Jianbin Tang, and Antonio Jimeno Yepes. Publaynet: Largest dataset ever for document layout analysis. In International Conference on Document Analysis and Recognition, pages 1015–1022, 2019.

[11] Minghao Li, Yiheng Xu, Lei Cui, Shaohan Huang, Furu Wei, Zhoujun Li, and Ming Zhou. Docbank: A benchmark dataset for document layout analysis. In Proceedings ofthe 28th International Conference on Computational Linguistics, pages 949–960, 2020.

[12] Birgit Pfitzmann, Christoph Auer, Michele Dolfi, Ahmed S. Nassar, and Peter W. J. Staar. Doclaynet: A large human-annotated dataset for document-layout segmentation. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 3743–3751, 2022.

[13] Brandon Smock, Rohith Pesala, and Robin Abraham. PubTables-1M: Towards comprehensive table extraction from unstructured documents. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4634–4642, 2022.

[14] Zilong Wang, Yiheng Xu, Lei Cui, Jingbo Shang, and Furu Wei. Layoutreader: Pre-training of text and layout for reading order detection. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, pages 4735–4744, 2021.

[15] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings ofthe 56th Annual Meeting ofthe Associationfor Computational Linguistics, pages 2556–2565, 2018.

[16] Nikita Srivatsan, Sofia Samaniego, Omar Florez, and Taylor Berg-Kirkpatrick. Alt-text with context: Improving accessibility for images on twitter. In International Conference on Learning Representations, 2024.

[17] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021.

[18] Junnan Li, Dongxu Li, Silvio Savarese, and Steven C. H. Hoi. BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International Conference on Machine Learning, 2023.

[19] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N. Fung, and Steven C. H. Hoi. InstructBLIP: Towards general-purpose vision-language models with instruction tuning. In Advances in Neural Information Processing Systems, volume 36, 2023.

[20] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Advances in Neural Information Processing Systems, volume 36, 2023.

[21] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-VL: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023.

[22] Tengchao Lv, Yupan Huang, Jingye Chen, Lei Cui, Shuming Ma, Yaoyao Chang, Shaohan Huang, Wenhui Wang, Li Dong, Weiyao Luo, Shaoxiang Wu, Guoxin Wang, Cha Zhang, and Furu Wei. Kosmos-2.5: A multimodal literate model. arXiv preprint arXiv:2309.11419, 2023.

[23] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

[24] Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, et al. Gemma 3 technical report. arXiv preprint arXiv:2503.19786, 2025.

[25] Adam W. Harley, Alex Ufkes, and Konstantinos G. Derpanis. Evaluation of deep convolutional nets for document image classification and retrieval. In International Conference on Document Analysis and Recognition, pages 991–995, 2015.

[26] Guillaume Jaume, Hazim Kemal Ekenel, and Jean-Philippe Thiran. FUNSD: A dataset for form understanding in noisy scanned documents. In International Conference on Document Analysis and Recognition Workshops, pages 1–6, 2019.

[27] Seunghyun Park, Seung Shin, Bado Lee, Junyeop Lee, Jaeheung Surh, Minjoon Seo, and Hwalsuk Lee. CORD: A consolidated receipt dataset for post-OCR parsing. In Document Intelligence Workshop at Neural Information Processing Systems, 2019.

[28] Zheng Huang, Kai Chen, Jianhua He, Xiang Bai, Dimosthenis Karatzas, Shijian Lu, and C. V. Jawahar. ICDAR2019 competition on scanned receipt OCR and information extraction. arXiv preprint arXiv:2103.10213, 2021.

[29] Zewen Chi, Heyan Huang, Heng-Da Xu, Houjin Yu, Wanxuan Yin, and Xian-Ling Mao. Complicated table structure recognition. arXiv preprint arXiv:1908.04729, 2019.

[30] Hiuyi Cheng, Peirong Zhang, Sihang Wu, Jiaxin Zhang, Qiyuan Zhu, et al. M6doc: A large-scale multiformat, multi-type, multi-layout, multi-language, multi-annotation category dataset for modern document layout analysis. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15138–15147, 2023.

[31] Minesh Mathew, Dimosthenis Karatzas, and C. V. Jawahar. DocVQA: A dataset for VQA on document images. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 2199–2208, 2021.

[32] Jordy Van Landeghem, Rubèn Tito, Łukasz Borchmann, Michał Pietruszka, Pawel Joziak, Rafal Powalski, et al. Document understanding dataset and evaluation (DUDE). In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19528–19540, 2023.

[33] Linke Ouyang, Yuan Qu, Hongbin Zhou, Jiawei Zhu, Rui Zhang, Qunshu Lin, et al. Omnidocbench: Benchmarking diverse pdf document parsing with comprehensive annotations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24838–24848, 2025.

[34] Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, et al. MMLongBench-Doc: Benchmarking long-context document understanding with visualizations. In Advances in Neural Information Processing Systems, volume 37, 2024.

[35] Srikar Appalaraju, Bhavan Jasani, Bhargava Urala Kota, Yusheng Xie, and R. Manmatha. DocFormer: End-to-end transformer for document understanding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 993–1003, 2021.

[36] Cheng Da, Chuwei Luo, Qi Zheng, and Cong Yao. Vision grid transformer for document layout analysis. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 19462–19472, 2023.

[37] Kenton Lee, Mandar Joshi, Iulia Raluca Turc, Hexiang Hu, Fangyu Liu, Julian Martin Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, and Kristina Toutanova. Pix2Struct: Screenshot parsing as pretraining for visual language understanding. In International Conference on Machine Learning, 2023.

[38] Lukas Blecher, Guillem Cucurull, Thomas Scialom, and Robert Stojnic. Nougat: Neural optical understanding for academic documents. arXiv preprint arXiv:2308.13418, 2023.

[39] Hao Feng, Wei Shi, Ke Zhang, Xiang Fei, Lei Liao, Dingkang Yang, Yongkun Du, Xuecheng Wu, Jingqun Tang, Yang Liu, Hong Chen, and Can Huang. Dolphin-v2: Universal document parsing via scalable anchor prompting. arXiv preprint arXiv:2602.05384, 2026.

[40] Zhang Li, Yuliang Liu, Qiang Liu, Zhiyin Ma, Ziyang Zhang, Shuo Zhang, Zidun Guo, Jiarui Zhang, Xinyu Wang, and Xiang Bai. Monkeyocr: Document parsing with a structure-recognition-relation triplet paradigm. arXiv preprint arXiv:2506.05218, 2025.

[41] Cheng Cui, Ting Sun, Suyin Liang, Tingquan Gao, Zelun Zhang, Jiaxuan Liu, Xueqing Wang, Changda Zhou, Hongen Liu, Manhui Lin, et al. Paddleocr-vl: Boosting multilingual document parsing via a 0.9b ultra-compact vision-language model. arXiv preprint arXiv:2510.14528, 2025.

[42] Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang, Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan Qu, Fukai Shang, et al. Mineru: An open-source solution for precise document content extraction. arXiv preprint arXiv:2409.18839, 2024.

[43] LeKai Yu, Hao Liu, Kun Wang, Zhiran Li, Ruping Cao, Fan Liu, and Yupeng Hu. Parsefixer: An agentic framework for document parsing via selective multimodal correction. arXiv preprint arXiv:2606.11977, 2026.

[44] Boyang Zhang, Sebastián G. Acosta, Preston Carlson, Sacha Bron, Pierre-Loïc Doulcet, Daniel B. Ospina, and Simon Suo. Parsebench: A document parsing benchmark for ai agents, 2026.

[45] Danna Gurari, Qing Li, Abigale J. Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P. Bigham. VizWiz grand challenge: Answering visual questions from blind people. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, pages 3608–3617, 2018.

[46] Danna Gurari, Yinan Zhao, Meng Zhang, and Nilavra Bhattacharya. Captioning images taken by people who are blind. In European Conference on Computer Vision, pages 417–434, 2020.

[47] Oleksii Sidorov, Ronghang Hu, Marcus Rohrbach, and Amanpreet Singh. TextCaps: A dataset for image captioning with reading comprehension. In European Conference on Computer Vision, pages 742–758, 2020.

[48] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. OCR-VQA: Visual question answering by reading text in images. In International Conference on Document Analysis and Recognition, pages 947–952, 2019.

[49] Bryan Wang, Gang Li, Xin Zhou, Zhourong Chen, Tovi Grossman, and Yang Li. Screen2Words: Automatic mobile UI summarization with multimodal learning. In Proceedings ofthe 34th Annual ACM Symposium on User Interface Software and Technology, pages 498–510, 2021.

[50] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244, 2022.

[51] Fangyu Liu, Julian Eisenschlos, Francesco Piccinno, Syrine Krichene, Chenxi Pang, Kenton Lee, Mandar Joshi, Wenhu Chen, Nigel Collier, and Yasemin Altun. DePlot: One-shot visual language reasoning by plot-to-table translation. In Findings ofthe Associationfor Computational Linguistics: ACL 2023, pages 10381–10399, 2023.

[52] Shankar Kantharaj, Rixie Tiffany Leong, Xiang Lin, Ahmed Masry, Megh Thakkar, Enamul Hoque, and Shafiq Joty. Chart-to-text: A large-scale benchmark for chart summarization. In Proceedings ofthe 60th Annual Meeting ofthe Associationfor Computational Linguistics, pages 4005–4023, 2022.

[53] Nitesh Methani, Pritha Ganguly, Mitesh M. Khapra, and Pratyush Kumar. PlotQA: Reasoning over scientific plots. In IEEE Winter Conference on Applications ofComputer Vision, pages 1516–1525, 2020.

[54] Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and C. V. Jawahar. Infographicvqa. In Proceedings ofthe IEEE/CVF Winter Conference on Applications ofComputer Vision, pages 1697–1706, 2022.

[55] Google DeepMind. Gemini 3 Pro Model Card. https://deepmind.google/models/model-cards/ gemini-3-pro/, December 2025.

[56] Google DeepMind. Gemini 3 Flash Model Card. https://storage.googleapis.com/ deepmind-media/Model-Cards/Gemini-3-Flash-Model-Card.pdf, December 2025. Model card dated December 17, 2025. Accessed: 2026-05-07.

[57] Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026.

[58] Google DeepMind. Gemma 4 Model Card. https://ai.google.dev/gemma/docs/core/model\_ card\_4, April 2026.

[59] OpenAI. Introducing GPT-5.4. https://openai.com/index/introducing-gpt-5-4/, March 2026.

[60] OpenAI. Gpt-5.6: Frontier intelligence that scales with your ambition. https://openai.com/index/ gpt-5-6/, July 2026.

[61] Google DeepMind. Gemini 3.6 flash: Model card. https://deepmind.google/models/ model-cards/gemini-3-6-flash/, 2026.

[62] Google DeepMind. Gemini 3.1 pro: Model card. https://deepmind.google/models/model-cards/ gemini-3-1-pro/, 2026.

[63] Haoran Wei, Yaofeng Sun, and Yukun Li. DeepSeek-OCR 2: Visual causal flow, 2026.

[64] Cheng Cui, Ting Sun, Manhui Lin, Tingquan Gao, Yubo Zhang, Jiaxuan Liu, Xueqing Wang, Zelun Zhang, Changda Zhou, Hongen Liu, Yue Zhang, Wenyu Lv, Kui Huang, Yichao Zhang, Jing Zhang, Jun Zhang, Yi Liu, Dianhai Yu, and Yanjun Ma. PaddleOCR 3.0 Technical Report, 2025.

[65] Satanjeev Banerjee and Alon Lavie. METEOR: An automatic metric for MT evaluation with improved correlation with human judgments. In Proceedings of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measuresfor Machine Translation and/or Summarization, pages 65–72, 2005.

[66] Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. BERTScore: Evaluating text generation with BERT. In International Conference on Learning Representations, 2020.

## A Dataset Details

This appendix is supplementary reading. The main text defines the task contract and reports the primary experimental conclusions; the material below gives the distributional and procedural detail needed to inspect the benchmark more closely.

## A.1 Label Taxonomy and Corpus Profile

Read-Parse-Describe uses a 23-label taxonomy rather than a small generic layout set. The goal is to keep elements that matter for reading order, reflow, caption linkage, and auditability visible as distinct targets. Headings, paragraph fragments, captions, equations, code blocks, metadata, references, lists, and visual objects therefore remain separately labeled.

Table 5: Semantic label taxonomy used in Read-Parsing-Describe.
<table><tr><td>ID</td><td>Description</td><td>Label</td><td>ID</td><td>Description</td><td>Label</td></tr><tr><td>1</td><td>Paper Title</td><td>sec_0</td><td>13</td><td>Table</td><td>tab</td></tr><tr><td>2</td><td>Level-1 Heading</td><td>sec_1</td><td>14</td><td>Figure-Caption</td><td>figure-cap</td></tr><tr><td>3</td><td>Level-2 Heading</td><td>sec_2</td><td>15</td><td>Table-Caption</td><td>table-cap</td></tr><tr><td>4</td><td>Level-3 Heading</td><td>sec_3</td><td>16</td><td>Annotation</td><td>anno</td></tr><tr><td>5</td><td>Level-4 Heading</td><td>sec_4</td><td>17</td><td>Formula</td><td>equ</td></tr><tr><td>6</td><td>Paragraph</td><td>para</td><td>18</td><td>Code Block</td><td>code</td></tr><tr><td>7</td><td>Spanning Paragraph</td><td>half_para</td><td>19</td><td>Catalog</td><td>catalogue</td></tr><tr><td>8</td><td>Header</td><td>header</td><td>20</td><td>Reference</td><td>reference</td></tr><tr><td>9</td><td>Footer</td><td>foot</td><td>21</td><td>Author-information</td><td>author-info</td></tr><tr><td>10</td><td>Footnote</td><td>fnote</td><td>22</td><td>Conference logo</td><td>icon</td></tr><tr><td>11</td><td>Watermark</td><td>watermark</td><td>23</td><td>List</td><td>list</td></tr><tr><td>12</td><td>Figure</td><td>fig</td><td></td><td></td><td></td></tr></table>

The benchmark itself contains 2,402 curated scientific documents and 18,164 continuous pages. The domain chart below summarizes the broader analyzed source-corpus pool of 2,151 ACM articles from 2020–2024; the residual slice groups domains that were not individually named in the figure.

Table 6: Primary Domain Distribution of Analyzed Articles from 2020 - 2024
<table><tr><td rowspan=1 colspan=1>Category</td><td rowspan=1 colspan=1>Count</td><td rowspan=1 colspan=1>Percentage</td></tr><tr><td rowspan=1 colspan=1>AI &amp; ML</td><td rowspan=1 colspan=1>271</td><td rowspan=1 colspan=1>12.60%</td></tr><tr><td rowspan=1 colspan=1>Accessibility</td><td rowspan=1 colspan=1>255</td><td rowspan=1 colspan=1>11.85%</td></tr><tr><td rowspan=1 colspan=1>AR/VR/XR</td><td rowspan=1 colspan=1>235</td><td rowspan=1 colspan=1>10.93%</td></tr><tr><td rowspan=1 colspan=1>Design &amp; HCI Methods</td><td rowspan=1 colspan=1>149</td><td rowspan=1 colspan=1>6.93%</td></tr><tr><td rowspan=1 colspan=1>Privacy &amp; Social</td><td rowspan=1 colspan=1>103</td><td rowspan=1 colspan=1>4.79%</td></tr><tr><td rowspan=1 colspan=1>Health &amp; Wellbeing</td><td rowspan=1 colspan=1>95</td><td rowspan=1 colspan=1>4.42%</td></tr><tr><td rowspan=1 colspan=1>Demographics</td><td rowspan=1 colspan=1>71</td><td rowspan=1 colspan=1>3.30%</td></tr><tr><td rowspan=1 colspan=1>Games, EdTech &amp; Creativity</td><td rowspan=1 colspan=1>65</td><td rowspan=1 colspan=1>3.02%</td></tr><tr><td rowspan=1 colspan=1>Ethics, Fairness &amp; CSCW</td><td rowspan=1 colspan=1>61</td><td rowspan=1 colspan=1>2.84%</td></tr><tr><td rowspan=1 colspan=1>Hardware &amp; Fabrication</td><td rowspan=1 colspan=1>56</td><td rowspan=1 colspan=1>2.60%</td></tr><tr><td rowspan=1 colspan=1>Others / Long-tail</td><td rowspan=1 colspan=1>790</td><td rowspan=1 colspan=1>36.72%</td></tr><tr><td rowspan=1 colspan=1>Total</td><td rowspan=1 colspan=1>2,151</td><td rowspan=1 colspan=1>100.00%</td></tr></table>

## A.2 Page Curation and Element Distributions

The first-eight-page rule is intended to increase the density of multimodal scientific content while keeping each paper as a continuous page sequence. It preserves the early-paper regions where abstracts, methods, figures, tables, captions, equations, and author metadata interact most strongly. Later pages are not treated as unimportant; they are underrepresented because they are disproportionately reference-heavy in the analyzed full-text subset.

The following tables decompose the visual and layout population from two complementary angles. Subject categories describe the broad disciplinary context of visual instances. Image types describe the visual form that the alt-text and caption-linking steps must handle. Aggregated layout groups

Table 7: Impact of the 8-Page Truncation Strategy on Layout Element Distribution. By comparing the element proportions in Pages 1–8 versus Pages ≥ 9 (estimated on a full-text subset), we observe that the truncation effectively preserves high-value multimodal objects while filtering out visually monotonous reference lists.
<table><tr><td>Element Category</td><td>Pages 1–8 (%)</td><td>Pages ≥ 9 (%)</td><td>Density (1–8)</td><td>Density (≥ 9)</td></tr><tr><td>Title, Abstract &amp; Metadata</td><td>100.00%</td><td>0.00%</td><td>1.2 / page</td><td>0.0 / page</td></tr><tr><td>Equations &amp; Math Blocks</td><td>78.42%</td><td>21.58%</td><td>4.5 / page</td><td>1.2 / page</td></tr><tr><td>Figures &amp; Captions</td><td>69.15%</td><td>30.85%</td><td>2.1 / page</td><td>0.9 / page</td></tr><tr><td>Body Paragraphs</td><td>62.30%</td><td>37.70%</td><td>12.4 /page</td><td>7.5 / page</td></tr><tr><td>Tables &amp; Algorithms</td><td>54.80%</td><td>45.20%</td><td>0.8 / page</td><td>0.7 / page</td></tr><tr><td>References</td><td>8.25%</td><td>91.75%</td><td>0.5 / page</td><td>15.6/ page</td></tr><tr><td>Overall Multimodal Complexity</td><td>High</td><td>Low</td><td>Diverse</td><td>Monotonous</td></tr></table>

show how much of the annotation budget is spent on document structure, metadata, scholarly objects, captions, references, and fragments.

Table 8: Image Categories Distribution
<table><tr><td rowspan=1 colspan=1>Category</td><td rowspan=1 colspan=1>Count</td><td rowspan=1 colspan=1>Percentage</td></tr><tr><td rowspan=1 colspan=1>Computer Science &amp; AI</td><td rowspan=1 colspan=1>6,443</td><td rowspan=1 colspan=1>56.8%</td></tr><tr><td rowspan=1 colspan=1>Social Science</td><td rowspan=1 colspan=1>2,181</td><td rowspan=1 colspan=1>19.2%</td></tr><tr><td rowspan=1 colspan=1>Engineering</td><td rowspan=1 colspan=1>1,359</td><td rowspan=1 colspan=1>12.0%</td></tr><tr><td rowspan=1 colspan=1>Medicine &amp; Healthcare</td><td rowspan=1 colspan=1>652</td><td rowspan=1 colspan=1>5.8%</td></tr><tr><td rowspan=1 colspan=1>Other</td><td rowspan=1 colspan=1>245</td><td rowspan=1 colspan=1>2.2%</td></tr><tr><td rowspan=1 colspan=1>Biology</td><td rowspan=1 colspan=1>110</td><td rowspan=1 colspan=1>1.0%</td></tr><tr><td rowspan=1 colspan=1>Physics</td><td rowspan=1 colspan=1>106</td><td rowspan=1 colspan=1>0.9%</td></tr><tr><td rowspan=1 colspan=1>Mathematics</td><td rowspan=1 colspan=1>98</td><td rowspan=1 colspan=1>0.9%</td></tr><tr><td rowspan=1 colspan=1>Economics &amp; Finance</td><td rowspan=1 colspan=1>72</td><td rowspan=1 colspan=1>0.6%</td></tr><tr><td rowspan=1 colspan=1>Environmental Science</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>0.4%</td></tr><tr><td rowspan=1 colspan=1>Chemistry</td><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>0.2%</td></tr><tr><td rowspan=1 colspan=1>Total</td><td rowspan=1 colspan=1>11,338</td><td rowspan=1 colspan=1>100.0%</td></tr></table>

Table 9: Image Type Distribution
<table><tr><td rowspan=1 colspan=1>Image Type</td><td rowspan=1 colspan=1>Count</td><td rowspan=1 colspan=1>Percentage</td></tr><tr><td rowspan=1 colspan=1>Table</td><td rowspan=1 colspan=1>2758</td><td rowspan=1 colspan=1>24.3%</td></tr><tr><td rowspan=1 colspan=1>Diagrammatic Instruction / Concept</td><td rowspan=1 colspan=1>2086</td><td rowspan=1 colspan=1>18.4%</td></tr><tr><td rowspan=1 colspan=1>User Interface / Screenshot</td><td rowspan=1 colspan=1>1743</td><td rowspan=1 colspan=1>15.4%</td></tr><tr><td rowspan=1 colspan=1>Photographic Image</td><td rowspan=1 colspan=1>1529</td><td rowspan=1 colspan=1>13.5%</td></tr><tr><td rowspan=1 colspan=1>Flowchart / Block Diagram</td><td rowspan=1 colspan=1>1000</td><td rowspan=1 colspan=1>8.8%</td></tr><tr><td rowspan=1 colspan=1>Bar Chart / Histogram</td><td rowspan=1 colspan=1>942</td><td rowspan=1 colspan=1>8.3%</td></tr><tr><td rowspan=1 colspan=1>Line Graph</td><td rowspan=1 colspan=1>466</td><td rowspan=1 colspan=1>4.1%</td></tr><tr><td rowspan=1 colspan=1>Other</td><td rowspan=1 colspan=1>371</td><td rowspan=1 colspan=1>3.3%</td></tr><tr><td rowspan=1 colspan=1>Scatter Plot</td><td rowspan=1 colspan=1>205</td><td rowspan=1 colspan=1>1.8%</td></tr><tr><td rowspan=1 colspan=1>Heatmap</td><td rowspan=1 colspan=1>147</td><td rowspan=1 colspan=1>1.3%</td></tr><tr><td rowspan=1 colspan=1>Pie Chart</td><td rowspan=1 colspan=1>27</td><td rowspan=1 colspan=1>0.2%</td></tr><tr><td rowspan=1 colspan=1>Anatomical Illustration</td><td rowspan=1 colspan=1>27</td><td rowspan=1 colspan=1>0.2%</td></tr><tr><td rowspan=1 colspan=1>Circuit Diagram</td><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>0.2%</td></tr><tr><td rowspan=1 colspan=1>Radar Chart</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>0.1%</td></tr><tr><td rowspan=1 colspan=1>Molecular Structure</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>0.0%</td></tr><tr><td rowspan=1 colspan=1>Total</td><td rowspan=1 colspan=1>11338</td><td rowspan=1 colspan=1>100.0%</td></tr></table>

## B Alt-text Annotation Details

Visual descriptions are harder to audit than boxes or OCR strings because a fluent sentence can still be vague, hallucinated, or attached to the wrong object. We therefore separate the alt-text process

Table 10: Distribution of high-value layout elements.
<table><tr><td>Category</td><td>Label</td><td>Count</td><td>Percentage</td></tr><tr><td>Structural Headers</td><td>sec_1, sec_2, sec_0</td><td>44,617</td><td>35.64%</td></tr><tr><td>Academic Metadata</td><td>author-info,fnote</td><td>28,381</td><td>22.66%</td></tr><tr><td>Scholarly Objects</td><td>fig, tab, equ, code</td><td>12,336</td><td>9.85%</td></tr><tr><td>Captions and References</td><td>figure-cap,table-cap,reference</td><td>21,381</td><td>17.08%</td></tr><tr><td>Layout Fragments</td><td>half_para, list, icon</td><td>17,564</td><td>14.03%</td></tr><tr><td>Others</td><td>anno, sec_3, sec_4, watermark, etc.</td><td>939</td><td>0.74%</td></tr><tr><td>Total</td><td>一</td><td>125,218</td><td>100.00%</td></tr></table>

## Human-LLM Agreement Summary

Usable files: 200; Exact-score agreement: 194/200 (97.0%)

![](images/af9c0efac7c5fec51d223baeb6656ef953aff384e4def0ba8baabf7ce0e74de3.jpg)  
Confusion Matrix  
Rows = LLM quality\_score, Columns = human\_score

![](images/5b65df384e113b6295238083aeb38d0156fabcded1e9eeec035054828866851f.jpg)  
Figure 4: Human-LLM agreement audit over 200 revised samples. Agreement is measured after low-score descriptions are refined and re-scored.

into generation, rubric-based scoring, targeted human refinement, and a manual audit. The detailed prompt and rubric below are provided to make the label assumptions inspectable.

The score distribution reports the same 11,338 visual instances before and after human refinement.   
Low-scoring generated descriptions are revised, then re-scored under the same rubric.

## C Evaluation Metric

To quantitatively assess the performance of the fine-tuned models in handling complex scholarly layouts, we employ a multi-dimensional evaluation suite. This suite covers three complementary aspects: (1) physical localization of layout elements, (2) logical sequencing of the reading flow, and (3) textual and tabular accuracy of the extracted content.(4) Accessibility Metadata Evaluation.Such a decomposition is necessary because a model can, for example, detect elements correctly but still fail to order them properly, or produce fluent text while misplacing bounding boxes.

## 1. Physical Layout Analysis: F1-score and mIoU

For Document Layout Analysis (DLA), we evaluate the model’s ability to correctly identify and localize structural entities (e.g., figures, tables, and section blocks).

Table 11: Detailed Prompt Configurations and Expanded Evaluation Rubric for Alt-text Generation
<table><tr><td>Configuration</td><td>Detailed Content / Implementation</td></tr><tr><td>Generation Prompt</td><td>“Describe this image in 1-3 concise sentences for accessibility. If the image contains multiple panels or subfigures, summarize them as a single combined figure rather than describing each panel separately. Focus only on the key visual information, such as the type of chart or image, the main subjects, and clearly visible labels or visual patterns. Do not use phrases like &#x27;image of’ or &#x27;picture of’. Provide a factual description based only on what is visible,</td></tr><tr><td>Generation Parameters</td><td>without interpreting or explaining the underlying data.&quot; Temperature: 0.2 Top-p: 0.9 Max New Tokens: 256 Repetition Penalty: 1.05</td></tr><tr><td>Evaluation Role</td><td>System Prompt: You are a rigorous Image Accessibility Auditor specializing in scientific and technical content. Your goal is to determine if the provided &#x27;Alt-text&#x27; accurately represents the visual information for a visually impaired user. Evaluate strictly based on Accuracy, Completeness, and Constraint Adherence.</td></tr><tr><td rowspan="4">Evaluation Rubric</td><td>Score 1 (Failure - Misleading/Hallucination): Completely irrelevant, contains gross data hallucinations, or generates pure technical noise. Fails to identify the basic type of the image. Actively misleads the visually impaired user.</td></tr><tr><td>Score 2 (Poor - Severe Omissions/Violations): Identifies the general chart/image type but suffers from severe omissions (e.g., missing critical axes, legends, or main subjects). Alternatively, it severely violates negative constraints (e.g., provides subjective interpretations of data instead of visual facts).</td></tr><tr><td>Score 3 (Acceptable - Basic but Flawed): Correctly identifies the main subjects and chart type. However, it lacks specific critical details (e.g., distinct visual trends or key labels). May fail minor constraints (e.g., exceeds the 3-sentence limit, or includes filler phrases like &quot;This is a picture of&quot;). Score 4 (Good - Professional &amp; Factual): Clear, objective, and accurate.</td></tr><tr><td>Successfully captures the chart type, major variables, key labels, and significant visual patterns. Strictly adheres to negative constraints. May only miss minor, non-essential visual nuances or slightly lack extreme conciseness.</td></tr><tr><td></td><td>Score 5 (Excellent - Perfect Accessibility): Highly concise (strictly 1-3 sentences), factual, and deeply informative. Seamlessly summarizes complex or multi-panel figures into a cohesive whole. Provides a visually impaired user with the exact informational equivalent of a quick visual scan, devoid of any over-interpretation or fluff.</td></tr></table>

Table 12: Alt text generation:LLM generated vs LLM+human\_modified
<table><tr><td rowspan=1 colspan=1>category</td><td rowspan=1 colspan=1>LLM generated (Count)</td><td rowspan=1 colspan=1>LLM + human_modified (Count)</td></tr><tr><td rowspan=1 colspan=1>Score 1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>Score 2</td><td rowspan=1 colspan=1>146</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>Score 3</td><td rowspan=1 colspan=1>1262</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>Score 4</td><td rowspan=1 colspan=1>2419</td><td rowspan=1 colspan=1>226</td></tr><tr><td rowspan=1 colspan=1>Score 5</td><td rowspan=1 colspan=1>7511</td><td rowspan=1 colspan=1>11112</td></tr><tr><td rowspan=1 colspan=1>Score 5</td><td rowspan=1 colspan=1>66.2%</td><td rowspan=1 colspan=1>98.1%</td></tr></table>

F1-score. We compute the harmonic mean of Precision and Recall for each label category. A detection is considered a True Positive (TP) if (i) the predicted category matches the ground-truth category and (ii) the overlap between predicted and ground-truth regions exceeds a predefined threshold (typically based on IoU). The metric is defined as:

$$
F 1 = 2 \cdot \frac { P r e c i s i o n \cdot R e c a l l } { P r e c i s i o n + R e c a l l }
$$

In practice, F1-score summarizes the trade-off between missing elements (low Recall) and producing spurious detections (low Precision).

mIoU (mean Intersection over Union). To measure geometric localization accuracy, we compute the Intersection over Union (IoU) between each predicted bounding box $B _ { p }$ and the corresponding ground-truth box $B _ { g t }$ . The mean IoU over N matched instances is reported as:

$$
m I o U = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { A r e a ( B _ { p , i } \cap B _ { g t , i } ) } { A r e a ( B _ { p , i } \cup B _ { g t , i } ) }
$$

Compared to F1-score, mIoU is more sensitive to precise boundary alignment, which is critical for downstream tasks such as figure-to-caption association or table cell reconstruction.

## 2. Logical Reading Order: Normalized Edit Distance (NED)

To evaluate Reading Order (RO), we represent the sequence of detected blocks as a string of identifiers and compare it with the ground-truth sequence. We use the Levenshtein edit distance to measure the minimal number of insertions, deletions, and substitutions needed to transform the predicted order into the correct order. To ensure comparability across documents with different numbers of blocks, we normalize the distance and define the RO score as:

$$
R O _ { S c o r e } = 1 - \frac { L e v e n s h t e i n ( S _ { p } , S _ { g t } ) } { \operatorname* { m a x } ( | S _ { p } | , | S _ { g t } | ) }
$$

where $S _ { p }$ and $S _ { g t }$ denote the predicted and ground-truth block sequences, respectively. A higher score indicates a more faithful logical flow, which is essential for reflowable rendering and accessible reading.

## 3. Textual and Tabular Accuracy: ED and TEDS

OCR accuracy (Edit Distance, ED). For general text extraction, we compute character-level edit distance between the recognized text and the reference transcription. This metric directly reflects transcription fidelity and is particularly informative for scientific documents containing symbols, subscripts, and uncommon terminology.

Table reconstruction (TEDS). For table parsing, we use the Tree Edit Distance Based Similarity (TEDS), which evaluates both the global structure (row/column organization) and the cell content by representing tables as tree structures. The similarity is defined as:

$$
T E D S ( T _ { a } , T _ { b } ) = 1 - \frac { D i s t _ { t r e e } ( T _ { a } , T _ { b } ) } { \operatorname* { m a x } ( | T _ { a } | , | T _ { b } | ) }
$$

where $T _ { a }$ and $T _ { b }$ are the predicted and ground-truth table trees. This metric is more aligned with human judgment than plain text metrics, because it penalizes structural errors (e.g., merged/split cells) in addition to textual mistakes.

## 4. Accessibility Metadata Evaluation: METEOR Score

To evaluate the quality of the generated Alt-text for figures and tables, we employ the METEOR metric. While metrics like BLEU primarily focus on exact n-gram overlaps, METEOR is designed to capture the semantic nuances of scholarly descriptions more effectively.

The Rationale for Choosing METEOR In the context of academic document reconstruction, the description of a figure (Alt-text) can be linguistically diverse. METEOR is superior for this task due to its unique features:Synonym Matching: It utilizes a dictionary (like WordNet) to match synonyms. For instance, if the ground truth says "Diagram" and the model generates "Schematic," METEOR recognizes this as a valid match, whereas BLEU would penalize it.Stemming: It matches words with the same root (e.g., "clustering" and "clustered"), which is crucial for maintaining the technical accuracy of alt-text.Alignment-based Scoring: It creates an explicit alignment between the predicted string and the reference, penalizing incorrect word orders through a "fragmentation penalty."

Mathematical Formulation The METEOR score is calculated by combining the harmonic mean of precision (P) and recall (R), with a specific emphasis on recall, followed by a penalty for incorrect ordering.First, the F-score is computed as:

$$
F _ { m e a n } = \frac { P \cdot R } { \alpha \cdot P + ( 1 - \alpha ) \cdot R }
$$

Typically, α is set to give more weight to recall, ensuring that the descriptive content of the figure is fully captured.Then a Penalty (P en) is calculated based on the "chunkiness" of the alignment (how fragmented the matches are):

$$
P e n = \gamma \cdot \left( \frac { \# c h u n k s } { \# u n i g r a m s \_ m a t c h e d } \right) ^ { \beta }
$$

The final METEOR Score is:

$$
S c o r e = F _ { m e a n } \cdot ( 1 - P e n )
$$

## D Evaluation Details

All reported systems are converted into the same element schema before scoring. Layout matching requires category agreement and an IoU threshold of 0.5. DLA-F1 is computed over matched layout elements, and mIoU averages the IoU of true-positive matches. Reading order is evaluated over matched element identifiers, so a model must both find the relevant element and place it correctly in the sequence. OCR is normalized before edit-distance scoring, and table content is scored through the serialized html field when a system produces it. Visual descriptions are scored only for visual instances with reference alternative text; a missing description for an applicable object receives zero for the description metric.

Unparsable predictions are not discarded globally. Minor syntax errors are repaired deterministically when the intended field structure is recoverable. If a prediction remains unparsable, only the affected fields are treated as empty. This localizes a syntax failure to the fields it prevents from being evaluated and avoids giving a model credit for content that cannot be reliably associated with an element object.

## E Training-data Scaling

We compare full fine-tuning and SARA on Qwen3.5-2B using 10%, 30%, 50%, and 100% of the training data under the same reported evaluation protocol. Table 13 reports all six metrics. At 10%, SARA trails full fine-tuning on all four structural metrics. Results are mixed at 30%; at 50%, it improves F1, mIoU, and OCR while matching RO, and at 100% it improves all four. The F1 difference changes from −0.0756 at 10% to +0.1790 at 50% and +0.1853 at 100%. Thus, the structural advantage emerges at larger training fractions in this experiment. METEOR favors the adapter at each fraction, while BERTScore changes are mixed.

Table 13: Training-data scaling on Qwen3.5-2B. Fractions indicate the reported proportion of the training data.
<table><tr><td>Data</td><td>Method</td><td>F1</td><td>mIoU</td><td>OCR</td><td>RO</td><td>METEOR</td><td>BERTScore</td></tr><tr><td>10%</td><td>Full FT</td><td>0.6487</td><td>0.7563</td><td>0.8476</td><td>0.7996</td><td>0.1256</td><td>0.8106</td></tr><tr><td>10%</td><td>SARA</td><td>0.5731</td><td>0.7144</td><td>0.7311</td><td>0.7805</td><td>0.1672</td><td>0.8002</td></tr><tr><td>30%</td><td>Full FT</td><td>0.6771</td><td>0.7861</td><td>0.8756</td><td>0.8504</td><td>0.2651</td><td>0.8288</td></tr><tr><td>30%</td><td>SARA</td><td>0.6567</td><td>0.8061</td><td>0.8087</td><td>0.8226</td><td>0.2871</td><td>0.8372</td></tr><tr><td>50%</td><td>Full FT</td><td>0.6991</td><td>0.8232</td><td>0.9362</td><td>0.9623</td><td>0.3072</td><td>0.8540</td></tr><tr><td>50%</td><td>SARA</td><td>0.8781</td><td>0.8533</td><td>0.9562</td><td>0.9623</td><td>0.3340</td><td>0.8533</td></tr><tr><td>100%</td><td>Full FT</td><td>0.7336</td><td>0.8350</td><td>0.9408</td><td>0.9628</td><td>0.3741</td><td>0.8928</td></tr><tr><td>100%</td><td>SARA</td><td>0.9189</td><td>0.8919</td><td>0.9636</td><td>0.9774</td><td>0.3834</td><td>0.8946</td></tr></table>

Table 14: Resolution sensitivity of Qwen3.5-2B+SARA under the reported crop-resolution.
<table><tr><td>Resolution</td><td>F1</td><td>mIoU</td><td>OCR</td><td>RO</td></tr><tr><td>100%</td><td>0.9189</td><td>0.8919</td><td>0.9636</td><td>0.9774</td></tr><tr><td>50%</td><td>0.8389</td><td>0.8673</td><td>0.9248</td><td>0.9534</td></tr><tr><td>25%</td><td>0.7589</td><td>0.8194</td><td>0.7648</td><td>0.8864</td></tr></table>

## F Resolution Sensitivity

We evaluate Qwen3.5-2B+SARA at 100%, 50%, and 25% of the reported native crop resolution while keeping the model and evaluation settings fixed. As shown in Table 14, reducing resolution to 25% decreases F1 by 0.1600, mIoU by 0.0725, OCR by 0.1988, and RO by 0.0910. Because mIoU is averaged over matched true positives, its remaining value describes overlap among surviving matches and does not indicate preservation of all element boundaries. This experiment measures absolute input-resolution sensitivity; it does not establish a robustness advantage over full fine-tuning without a corresponding control evaluated at the same resolutions.

## G Failure cases.

To better understand the limitations of SARA, we analyze two representative failure modes, shown in Figure 5. First, borderless composite figures can confuse the model when multiple sub-figures and intra-image text are densely arranged. In such cases, visual and textual cues become entangled, which may shift predicted figure boxes away from their corresponding captions. Second, excessively dense tables can stress the decoding process. Although the table region may still be localized accurately, long serialized outputs may be truncated near the end, and the model may occasionally introduce minor irrelevant artifacts.

![](images/bfd1ab47d8cb368588c1eb6b7faf347ccde93bac534256d7a75519ce43ef8eb5.jpg)  
(a) Prediction: borderless composite figure

![](images/fe8df6755ff2748c3e7252d30e960ce408af1535731b0c1d2c38cb17772c0d11.jpg)  
(b) Ground truth for Case 1

![](images/438269d138c9ef37763265f2a49f6393952df448cf6af11c2ddcb2b642152257.jpg)  
(c) Prediction: dense table

![](images/8978b8126ef906ab93214116e1f7c0c154c5397127951f6819889a75415fccf2.jpg)  
(d) Ground truth for Case 2  
Figure 5: Representative failure cases of SARA. Top row: a borderless composite figure where intra-image text can cause a spatial shift between figure regions and their captions. Bottom row: a dense table where localization remains relatively accurate, but decoding may truncate long content and introduce minor hallucinated artifacts.