# VisualWorldBench: A Fine-Grained Multi-Task Benchmark for Evaluating Visual World Knowledge in MLLMs

Chen Chen<sup>1,2</sup> Zheming Liang<sup>1,2</sup> Jiaqi Wang<sup>2,3∗</sup> <sup>1</sup>University of Science and Technology of China <sup>2</sup>Shanghai Innovation Institute <sup>3</sup>JD.com

## Abstract

Visual world knowledge is a core capability of multimodal large language models (MLLMs), underpinning knowledge-intensive visual question answering and fine-grained visual reasoning. However, existing benchmarks remain limited in two key respects. First, their category coverage is narrow: most benchmarks span only a small number of categories across a few domains, falling far short of the open-ended scale of real-world visual entities. Second, their task construction is often insufficiently rigorous: models can answer correctly using coarse-grained cues, vague associations, or elimination strategies without truly mastering precise concepts, making it difficult to accurately assess their knowledge of fine-grained entities. We introduce VisualWorldBench<sup>2</sup>, a taxonomy-grounded multi-task benchmark built on a hierarchical taxonomy of 21,291 leaf-node classes across 8 visual domains, with 32,541 evaluation instances whose ground-truth labels cover 15,479 categories across four complementary tasks: closed-set recognition, contrastive selection, precise localization, and open-world naming, covering different directions of knowledge retrieval and varying degrees of answer openness to comprehensively evaluate visual world knowledge. Most questions are constructed from taxonomically proximate categories, forcing precise discrimination among fine-grained entities. Experiments reveal that the strongest closed-source model achieves only 80.4% average accuracy, exposing capability gaps hidden under conventional benchmarks. We further construct VisualWorldBench-50K, a training resource covering 50K+ categories. Fine-tuning on VisualWorldBench-50K yields over 21 percentage points improvement on VisualWorldBench, with consistent transfer to public fine-grained recognition and visual knowledge benchmarks.

## 1 Introduction

The visual world comprises millions of fine-grained entities—species, cultivars, architectural styles, vehicle models. Yet whether current multimodal large language models (MLLMs) [1, 2], driven by large-scale vision-language pretraining [3, 4], can truly discriminate among such entities rather than merely categorize them at a coarse level remains an open question. It can be regarded that fine-grained visual entity concepts are a form of visual world knowledge. Whether a model maintains clear discrimination boundaries among visually similar entities reflects the extent of visual world knowledge it has internalized, which underpins knowledge-intensive visual question answering and fine-grained visual reasoning.

![](images/1bad3f3baa02df116feb7f544721a133e0a7229ac552155c7daf8dbf447bc6b3.jpg)  
Figure 1: Overview of VisualWorldBench. The benchmark spans 8 visual domains—Animal, Plant, Fungus, Food, Scene, Transport, Artifact, and Other—with a taxonomy of 21,291 leaf-node classes and 32,541 evaluation instances covering 15,479 ground-truth categories. The benchmark evaluates four complementary tasks: closed-set recognition (MCQ), contrastive selection (Grid-Sel), spatial localization (Grid-Loc), and open-world naming (Open-QA).

However, existing benchmarks fall short along two complementary dimensions. (1) Limited category coverage. Classic fine-grained benchmarks, such as CUB-200 [5], Stanford Dogs [6], and FGVC-Aircraft [7], focus on narrow domains and limited category coverage. Even larger-scale extensions such as iNat2021 [8] and BIOSCAN-5M [9] remain largely concentrated in biological domains. Broader entity-knowledge benchmarks, including InfoSeek [10], SimpleVQA [11], and WorldVQA [12], expand beyond single-domain recognition, yet still cover only a limited set of categories, falling far short of the open-ended scale of real-world visual entities. (2) Insufficient evaluation rigor. General MLLM benchmarks such as MMBench [13], MMMU [14], and MMMU-Pro [15] evaluate multimodal competence rather than precise entity-level visual knowledge. Traditional fine-grained benchmarks are typically formulated as closed-set classification, which does not align with prompt-based MLLM evaluation. Meanwhile, existing entity-knowledge benchmarks often rely on task formats such as attribute QA, naming, or coarse-grained multiple-choice, where models can succeed through vague associations or elimination strategies without truly distinguishing between closely related fine-grained entities. As a result, current evaluations often fail to verify whether a model has learned precise conceptual boundaries for visual entities.

To address these limitations, we introduce VisualWorldBench, a taxonomy-grounded multi-task benchmark for evaluating visual world knowledge in MLLMs. Built on a taxonomy of 21,291 leafnode classes across 8 visual domains, it comprises 32,541 evaluation instances whose ground-truth labels cover 15,479 distinct categories across the four tasks. Table 1 compares VisualWorldBench with representative prior work.

Our benchmark has three distinctive features. (1) Large-scale category coverage: its ground-truth labels cover 15,479 categories across 8 visual domains, supported by a taxonomy of 21,291 leaf-node classes. (2) Taxonomy-driven precise evaluation: most questions are constructed from taxonomically proximate categories, requiring models to distinguish precise conceptual boundaries among fine-grained entities and preventing shortcuts based on vague knowledge or elimination strategies. (3) Comprehensive task coverage: this principle is instantiated through four complementary tasks, including text-to-image selection, image-to-text selection, target localization, and open-ended naming, covering different directions of knowledge retrieval and varying degrees of answer openness.

Table 1: Comparison with existing visual world knowledge and fine-grained recognition benchmarks. VisualWorldBench is the only benchmark that combines large-scale fine-grained coverage, multidomain scope, and multiple complementary task formats in a single MLLM-oriented evaluation. For VisualWorldBench, #Classes counts distinct ground-truth categories across the four tasks; its underlying taxonomy contains 21,291 leaf-node classes. <sup>∗</sup>For InfoSeek-Human, 806 denotes the number of visual entities referenced by the questions, not distinct answer categories. Tax.: hierarchical taxonomy provided. FG: fine-grained.
<table><tr><td>Benchmark</td><td>Year</td><td>Coverage</td><td>#Classes</td><td>#Eval</td><td>Tasks</td><td>Tax.</td><td>MLLM</td></tr><tr><td colspan="8">Fine-grained recognition benchmarks</td></tr><tr><td>CUB-200 [5]</td><td>2011</td><td>Bird</td><td>200</td><td>5.8K</td><td>Cls</td><td>x</td><td>x</td></tr><tr><td>Stanford Dogs [6]</td><td>2011</td><td>Dog</td><td>120</td><td>8.6K</td><td>Cls</td><td>x</td><td>x</td></tr><tr><td>FGVC-Aircraft [7]</td><td>2013</td><td>Aircraft</td><td>100</td><td>3.3K</td><td>Cls</td><td>√</td><td>x</td></tr><tr><td>iNat2021 [8]</td><td>2021</td><td>Bio</td><td>10K</td><td>500K</td><td>Cls</td><td>√</td><td>x</td></tr><tr><td>FG-BMK [16]</td><td>2026</td><td>FG</td><td></td><td>1.01M</td><td>QA+Ret+Cls</td><td>x</td><td>√</td></tr><tr><td colspan="8">MLLM general and knowledge benchmarks</td></tr><tr><td>MMBench [13]</td><td>2024</td><td>General</td><td></td><td>2.4K</td><td>Perception+Reasoning</td><td>X</td><td>√</td></tr><tr><td>MMMU-Pro [15]</td><td>2024</td><td>Academic</td><td></td><td>5.2K</td><td>MC reasoning</td><td>x</td><td>√</td></tr><tr><td>InfoSeek-Human [10]</td><td>2023</td><td>Entity</td><td>806*</td><td>8.9K</td><td>Knowledge QA</td><td>X</td><td>√</td></tr><tr><td>WorldVQA [12]</td><td>2026</td><td>Vision knowledge</td><td></td><td>3.5K</td><td>Knowledge QA</td><td>√</td><td>√</td></tr><tr><td>SimpleVQA [11]</td><td>2025</td><td>Vision knowledge</td><td></td><td>2.0K</td><td>Knowledge QA</td><td>x</td><td>√</td></tr><tr><td>VisualWorldBench (Ours)</td><td>2026</td><td>Multi-dom FG</td><td>15,479</td><td>32K+</td><td>MCQ+Sel+Loc+Open</td><td>√</td><td>√</td></tr></table>

Systematic evaluation across diverse closed-source and open-source models reveals substantial gaps in precise visual world knowledge. Large performance differences persist across task formats even within the same model family: for example, InternVL3.5-8B reaches 35.5% on closed-set recognition but only 7.4% on 3×3 localization, and similar patterns appear consistently across the Qwen, InternVL, and Gemma families. These results suggest that different task formats impose different demands on precise entity knowledge, and that closed-set accuracy alone substantially overestimates a model’s mastery of visual world knowledge. Based on these findings, we further construct VisualWorldBench-50K, a training resource covering 50K+ categories with supervision formats aligned to the benchmark. After fine-tuning, models improve by over 21 percentage points on average on VisualWorldBench, and the gains transfer to multiple public fine-grained recognition benchmarks, including CUB-200, Stanford Dogs, and iNaturalist 2021, as well as the visual entity knowledge benchmark WorldVQA. These results demonstrate that the capability gaps exposed by VisualWorldBench are actionable and can be effectively reduced with targeted entity-level supervision. Our main contributions are as follows:

• We introduce VisualWorldBench, a taxonomy-grounded multi-task benchmark for visual world knowledge, comprising 32,541 evaluation instances with 15,479 ground-truth categories across 8 domains, built on a taxonomy of 21,291 leaf-node classes.

• We propose a taxonomy-driven evaluation paradigm in which questions are constructed from taxonomically proximate categories and instantiated through four complementary task formats, enabling rigorous assessment of precise fine-grained entity knowledge.

• We show that conventional closed-set evaluation substantially overestimates model capability, while more discriminative and open-ended settings expose large gaps in precise visual world knowledge; we further construct VisualWorldBench-50K, a 50K+-category training resource that improves performance on VisualWorldBench by over 21 percentage points and transfers consistently to public benchmarks.

## 2 Related Work

Fine-grained recognition benchmarks. Fine-grained visual recognition has been studied extensively through domain-specific classification datasets spanning birds [5, 17], dogs [6], cars [18, 19], aircraft [7], food [20], flowers [21], and pets [22]. These datasets have driven substantial methodological progress, but they are confined to single domains with limited category counts and have become largely saturated under modern vision models. Subsequent efforts scale category coverage:

iNat2021 [8] introduces 10K biological species with a taxonomic hierarchy, and BIOSCAN-5M [9] provides millions of insect specimens for cross-modal alignment. However, both remain within the biological domain and adopt traditional closed-set protocols rather than prompt-based evaluation. More recently, FG-BMK [16] unifies established fine-grained datasets into MLLM-oriented questionanswering and retrieval protocols, but its category coverage remains constrained by the scale of the original datasets.

MLLM evaluation and visual grounding. Broad-coverage MLLM benchmarks [13, 14, 23] have established comprehensive evaluation protocols for multimodal reasoning, but fine-grained entity recognition constitutes only a small fraction of their test suites and is not diagnosed in isolation. A separate line of work targets visual entity knowledge directly: InfoSeek [10] and Enc-VQA [24] evaluate factual and encyclopedic knowledge about entities through question answering, yet do not assess whether models can discriminate among visually confusable alternatives or localize them spatially. On the grounding side, referring expression comprehension [25, 26] and grounded MLLMs [27, 28, 29] require localizing objects described by natural-language expressions; here, the challenge lies in resolving spatial and attribute-based ambiguity among common objects, which differs fundamentally from discriminating taxonomically adjacent fine-grained categories that share near-identical visual appearance.

## 3 Benchmark

The central design principle of VisualWorldBench is that all tasks share a single hierarchical taxonomy. This structure provides taxonomy-aware distractor sampling for evaluation tasks, category-level references for quality control, and a unified framework for organizing the training resource. We describe the taxonomy (§3.1), task design (§3.2), quality control (§3.3), and training resource (§3.4) in turn; overall benchmark statistics are provided in Appendix Table 4.

## 3.1 Taxonomy and Data Sources

VisualWorldBench is built on a hierarchical taxonomy containing 21,291 leaf-node classes with a maximum depth of 14 levels (Figure 2a), spanning 8 visual domains: animal, plant, fungus, artifact, transport, food, scene, and other. The taxonomy is derived from the Bamboo category tree [30] and further expanded with supplementary public sources to improve category coverage across domains. To make the taxonomy suitable for benchmark construction, we further curate both the source category set and the associated image pool through normalization, deduplication, filtering, and quality control. After additional image-quality filtering that removes low-quality images (too dark, too small, blurry, low saturation, etc.), the released benchmark covers 15,479 ground-truth categories across all tasks while nearly all taxonomy classes appear as distractors in MCQ or Grid tasks. Details of taxonomy construction, including per-domain class distributions and depth statistics, are provided in the appendix.

## 3.2 Task Design

We design four task formats to probe four core capabilities: discriminating the correct category from confusable alternatives (MCQ), selecting the target image among visually similar alternatives (Grid-Sel), precisely localizing it with a bounding box (Grid-Loc), and producing its precise name without any candidate set (Open-QA). Figure 3 illustrates representative examples of each task.

MCQ: closed-set recognition. Each sample presents an image together with 20 candidate class names, and the model must select the single correct category. We use 20 options, substantially more than the typical 4–5 in general-purpose benchmarks, to reduce the effect of random guessing. Unlike standard evaluations that rely on random distractors, we construct options using hierarchical distance in the taxonomy: sibling categories share the same parent node, near categories are taxonomically close but not siblings, and random categories are sampled from farther away in the hierarchy. Approximately 60% of distractors come from taxonomy neighbors (sibling + near), requiring finegrained discrimination among semantically and visually related candidates (Figure 2b).

![](images/b9d96fa9af419ed20b564ea3593980d4c0a48a3a804c116a91cb6554588ed606.jpg)  
Figure 2: Taxonomy-grounded construction and four-format evaluation design of VisualWorldBench. (a) Benchmark construction: a hierarchical taxonomy (illustrated with an example subtree), multiformat task generation, and two-stage quality control (model-assisted triage by three frontier models followed by human verification and release balancing). (b) The four evaluation formats—MCQ for closed-set recognition, Grid-Sel for contrastive selection, Grid-Loc for precise localization, and Open-QA for open-world naming—with their task setup, sample counts, and design choices.

Grid-Sel / Grid-Loc: contrastive selection and precise localization. Grid-Sel and Grid-Loc evaluate whether a model can identify a target category when multiple visually similar alternatives are presented simultaneously in a controlled composite-image setting. Each sample is formed by tiling the target image together with images from taxonomically similar categories into a 2×2 or 3×3 grid, with one category per cell; more than 99% of distractor cells are populated by sibling categories of the target. Grid-Sel requires the model to identify the target cell, measuring contrastive selection ability under tightly controlled visual competition; Grid-Loc uses the same grid layouts but requires the model to output a bounding box around the target object, measuring precise localization under the same controlled setup. The 3×3 setting is more demanding due to smaller cell images and more confusable alternatives.

Open-QA: open-world naming. Given an image and a natural-language question (e.g., “What species of bird is this?”), the model must generate the fine-grained category name directly, with no candidate options provided.

Evaluation metrics. MCQ and Grid-Sel are evaluated by accuracy. Grid-Loc is evaluated by IoU@0.5 after applying a fixed bounding-box extraction and normalization procedure. Parsing and failure-handling details are provided in the appendix. Open-QA is evaluated by EM+LLM Judge: we first match the extracted answer against the canonical label and alias set, then use a judge model to adjudicate semantically equivalent non-exact matches; full judging details are provided in the appendix.

## 3.3 Quality Control

We apply a two-stage release pipeline to all tasks: annotation verification followed by release balancing and filtering.

![](images/62f6157b1c9c6e7e0538f0bd6da35b350a4bd817dff389ca869edfddcab72986.jpg)  
Figure 3: Examples of the four VisualWorldBench tasks, all grounded in a shared 21,291-class taxonomy. MCQ (closed-set recognition): the model selects the correct category from 20 taxonomyaware options (sibling, near, and random distractors). Grid-Sel / Grid-Loc (contrastive selection and precise localization): the model identifies or localizes the target category among visually similar alternatives in a composite grid. Grid-Sel requires selecting the correct cell, while Grid-Loc requires outputting a precise bounding box on the same grid layout. Open-QA (open-world naming): the model generates the fine-grained category name without any candidate set.

Annotation verification. We use three frontier closed-source models as triage signals: when at least two models produce an incorrect answer, the sample is flagged for human review. Final keep/remove decisions follow predefined annotation criteria (label correctness, visual identifiability, ambiguity, and leakage checks) under human review; flagged samples are never removed based on model behavior alone. We additionally conduct a random audit on non-flagged samples to ensure model-assisted triage does not miss systematic issues; details are reported in Appendix F.1.

Release balancing and filtering. The final evaluation set is assembled from the verified pool under category-coverage, domain-representation, distractor-structure, and image-quality constraints, subject to a class-safe constraint that no category may be lost from the union of all four tasks. Per-stage statistics, audit protocols, and detailed task-specific release rules are provided in the appendix.

## 3.4 VisualWorldBench-50K

In addition to the benchmark, we construct VisualWorldBench-50K, a broader fine-grained training resource covering over 50K categories. It is anchored by large-scale closed-set MCQ supervision and uses taxonomy-aware near-neighbor distractors to strengthen fine-grained category discrimination. On top of this MCQ backbone, we introduce smaller amounts of grounding and open-ended supervision as complementary signals, exposing models to spatial alignment and category naming beyond closed-set recognition alone. In total, VisualWorldBench-50K comprises approximately 7.7M training instances collected under the benchmark taxonomy from publicly available fine-grained datasets and other public image sources, with MCQ accounting for roughly 90% of records and grounding/QA each contributing approximately 5%. As shown in Section 4, the capability gaps exposed by VisualWorldBench can be effectively reduced through this targeted supervision. To reduce contamination risk, we exclude Cars-196 test images and iNaturalist 2021 validation images from the training pool, as detailed in Appendix I.1. Its construction details and experimental use are described in Section 4 and Appendix G.

## 4 Experiments

## 4.1 Experimental Setup

Models. We evaluate three groups of models on VisualWorldBench. (1) Closed-source models: representative frontier commercial MLLMs from Google, OpenAI, ByteDance, and Xiaomi—Gemini 3.1 Pro [31], GPT-5.4 [32], GPT-5.2 [33], Doubao-Seed 2.0 Pro [34], and Mimo-V2-Omni [35]. (2) Opensource models: spanning four model families (Kimi, Qwen, InternVL, Gemma) across multiple parameter scales to reduce dependence on any single architecture: Kimi-K2.5 [36], Qwen3-VL-

Table 2: Main results on VisualWorldBench. MCQ: MCQ accuracy; Sel: Grid-Sel accuracy (2×2 / 3×3 / avg); Loc: Grid-Loc IoU@0.5 (2×2 / 3×3 / avg); Open: Open-QA accuracy (EM+LLM Judge); Avg: arithmetic mean of MCQ, Sel Avg, Loc Avg, and Open-QA. <sup>†</sup>Trained on VisualWorldBench-50K (Section 3.4). Best result per column in bold.
<table><tr><td rowspan="2">Model</td><td rowspan="2"></td><td colspan="3">Sel</td><td colspan="3">Loc</td><td rowspan="2">Open</td><td rowspan="2">Avg</td></tr><tr><td>MCQ 2×2</td><td>3×3</td><td>Avg</td><td>2×2</td><td>3×3</td><td>Avg</td></tr><tr><td colspan="8">Closed-source models</td><td></td></tr><tr><td>Gemini 3.1 Pro</td><td>86.0</td><td>94.2</td><td>87.3</td><td>90.8</td><td>84.6</td><td>83.3</td><td>84.0</td><td>60.6</td><td>80.4</td></tr><tr><td>Doubao-Seed 2.0 Pro</td><td>75.1</td><td>77.9</td><td>71.6</td><td>74.8</td><td>71.4</td><td>69.7</td><td>70.6</td><td>74.5</td><td>73.8</td></tr><tr><td>Mimo-V2-Omni</td><td>65.5</td><td>76.7</td><td>68.7</td><td>72.7</td><td>47.3</td><td>44.2</td><td>45.8</td><td>40.0</td><td>56.0</td></tr><tr><td>GPT-5.4</td><td>67.9</td><td>76.4</td><td>65.8</td><td>71.1</td><td>35.1</td><td>22.0</td><td>28.6</td><td>37.4</td><td>51.2</td></tr><tr><td>GPT-5.2</td><td>65.5</td><td>72.7</td><td>62.6</td><td>67.7</td><td>25.6</td><td>20.2</td><td>22.9</td><td>35.0</td><td>47.8</td></tr><tr><td colspan="8">Open-source models</td></tr><tr><td>Kimi-K2.5</td><td>68.6</td><td>77.7</td><td>71.5</td><td>74.6</td><td>71.4</td><td>50.8</td><td>61.1</td><td>47.2</td><td>62.9</td></tr><tr><td>Qwen3.5-27B</td><td>59.1</td><td>68.1</td><td>63.9</td><td>66.0</td><td>60.4</td><td>56.2</td><td>58.3</td><td>28.7</td><td>53.0</td></tr><tr><td>Gemma-4-31B</td><td>60.5</td><td>65.3</td><td>55.6</td><td>60.5</td><td>46.5</td><td>36.1</td><td>41.3</td><td>27.2</td><td>47.4</td></tr><tr><td>Qwen3.5-9B</td><td>52.4</td><td>60.8</td><td>53.9</td><td>57.4</td><td>54.6</td><td>45.2</td><td>49.9</td><td>26.8</td><td>46.6</td></tr><tr><td>Qwen3-VL-8B</td><td>52.0</td><td>47.7</td><td>45.3</td><td>46.5</td><td>38.7</td><td>32.5</td><td>35.6</td><td>29.1</td><td>40.8</td></tr><tr><td>Gemma-4-26B-A4B</td><td>56.0</td><td>57.4</td><td>46.6</td><td>52.0</td><td>16.6</td><td>10.9</td><td>13.8</td><td>24.0</td><td>36.5</td></tr><tr><td>Qwen3.5-4B</td><td>44.1</td><td>49.9</td><td>33.6</td><td>41.8</td><td>43.6</td><td>37.2</td><td>40.4</td><td>19.6</td><td>36.5</td></tr><tr><td>Qwen3-VL-4B</td><td>47.8</td><td>40.9</td><td>35.1</td><td>38.0</td><td>28.8</td><td>24.0</td><td>26.4</td><td>22.5</td><td>33.7</td></tr><tr><td>InternVL3.5-8B</td><td>35.5</td><td>44.4</td><td>31.6</td><td>38.0</td><td>16.7</td><td>7.4</td><td>12.1</td><td>10.3</td><td>24.0</td></tr><tr><td>InternVL3.5-4B</td><td>33.1</td><td>38.7</td><td>29.3</td><td>34.0</td><td>9.2</td><td>8.2</td><td>8.7</td><td>7.1</td><td>20.7</td></tr><tr><td>Qwen3-VL-2B</td><td>28.0</td><td>15.6</td><td>10.3</td><td>13.0</td><td>28.9</td><td>19.5</td><td>24.2</td><td>15.8</td><td>20.2</td></tr><tr><td colspan="10">SFT models</td></tr><tr><td>Qwen3-VL-2B†</td><td></td><td></td><td>70.7+42.7 46.5+30.9 31.3+21.0 38.9+25.9 54.3+25.4 50.8+31.3 52.6+28.4 18.5+2.7 45.2+25.0</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\mathrm { Q w e n } 3 { \cdot } \mathrm { V L } { \cdot } 4 \mathrm { B } ^ { \dagger }$ </td><td>76.8+29.074.6+33.7</td><td></td><td>61.3+26.2 68.0+30.064.4+35.6 60.5+36.5</td><td></td><td></td><td></td><td>62.5+36.1 26.5+4.058.5+24.8</td><td></td><td></td></tr><tr><td> $\mathrm { Q w e n 3 – V L { - } 8 B ^ { \dagger } }$ </td><td>80.0+28.0 76.0+28.3</td><td></td><td>70.5+25.2 73.3+26.862.8+24.1 64.3+31.8</td><td></td><td></td><td></td><td>63.6+28.033.9+4.862.7+21.9</td><td></td><td></td></tr><tr><td>InternVL3.5-8B†</td><td>69.3+33.8</td><td>69.4+25.0</td><td>59.4+27.8</td><td></td><td></td><td>64.4+26.4 53.0+36.3 37.9+30.5</td><td>45.4+33.4 14.1+3.8 48.3+24.3</td><td></td><td></td></tr></table>

2B/4B/8B [37], Qwen3.5-4B/9B/27B [38], InternVL3.5-4B/8B [39], Gemma-4-31B, and Gemma-4-26B-A4B (MoE) [40]. (3) SFT models: Qwen3-VL-8B/4B/2B and InternVL3.5-8B fine-tuned on VisualWorldBench-50K (Section 3.4) to test whether the capability gaps exposed by Visual-WorldBench can be reduced through targeted fine-grained supervision. Throughout the paper, Qwen3-VL-2B/4B/8B and InternVL3.5-8B rows without <sup>†</sup> denote the original instruct models, while rows with <sup>†</sup> denote the corresponding VisualWorldBench-50K-tuned variants.

Training data. All SFT models are fine-tuned on VisualWorldBench-50K described in Section 3.4, a broader fine-grained training resource covering over 50K classes and anchored by large-scale MCQ supervision with taxonomy-aware near-neighbor distractors. Smaller grounding and openended subsets provide complementary signals beyond closed-set discrimination alone. Training implementation details and dataset composition are provided in Appendix G.

Evaluation protocol. Evaluation metrics follow the definitions in Section 3. Grid-Sel and Grid-Loc are evaluated at both 2×2 and 3×3 resolutions; we report both settings as well as their within-task average. The reported overall average (Avg) is the arithmetic mean of MCQ, Sel Avg, Loc Avg, and Open-QA (EM+LLM Judge). Unless otherwise stated, all models are evaluated in a zero-shot setting with greedy decoding (temperature=0); images are fed at each model’s native input resolution. Additional training and evaluation details are provided in the appendix.

## 4.2 Main Results

Table 2 presents the main results on VisualWorldBench across closed-source models, open-source models, and our best SFT models.

Visual world knowledge is multi-dimensional, with strong cross-task dissociations. Performance varies dramatically across tasks: even models with reasonable MCQ accuracy can fail sharply on Grid-Loc. For example, InternVL3.5-8B, InternVL3.5-4B, and Gemma-4-26B-A4B achieve only 7.4%, 8.2%, and 10.9% on Loc-3×3, despite MCQ scores of 35.5%, 33.1%, and 56.0%, respectively. This indicates that precise localization is a systematic bottleneck for current MLLMs, and that closed-set accuracy alone substantially overestimates fine-grained capability. The performance hierarchy is broadly consistent across the Qwen, InternVL, and Gemma families, suggesting that VisualWorldBench captures general capability differences rather than architecture-specific artifacts. Moreover, even the best closed-source model (Gemini 3.1 Pro) reaches only 80.4% overall average, while most open-source models remain below 53%, confirming that the benchmark is far from saturated.

Table 3: Generalization to widely used public fine-grained benchmarks. Bird–Pet: 4-option MCQ accuracy on five public fine-grained benchmarks. iNat21: leaf-level classification accuracy on the animal and plant splits of iNaturalist 2021, plus their average. WorldVQA: open-ended visual QA accuracy (LLM-judged, structured answer format). <sup>†</sup>Trained on VisualWorldBench-50K.
<table><tr><td rowspan="2">Model</td><td colspan="6">Public MCQ</td><td colspan="3">iNat21</td><td rowspan="2">|WorldVQA</td></tr><tr><td>Bird</td><td>Dog</td><td>Air.</td><td>Flwr.</td><td>Pet</td><td>Avg</td><td>Animal</td><td>Plant</td><td>Avg</td></tr><tr><td>Qwen3-VL-2B</td><td>49.6</td><td>57.6</td><td>55.0</td><td>75.4</td><td>79.1</td><td>63.3</td><td>27.3</td><td>30.6</td><td>29.0</td><td>8.2</td></tr><tr><td>Qwen3-VL-4B</td><td>55.2</td><td>74.0</td><td>57.1</td><td>85.9</td><td>93.0</td><td>73.1</td><td>35.9</td><td>36.1</td><td>36.0</td><td>10.2</td></tr><tr><td>Qwen3-VL-8B</td><td>59.3</td><td>71.5</td><td>60.7</td><td>86.8</td><td>88.1</td><td>73.3</td><td>40.2</td><td>41.3</td><td>40.8</td><td>11.2</td></tr><tr><td>InternVL3.5-8B</td><td>41.7</td><td>55.4</td><td>42.2</td><td>54.1</td><td>79.8</td><td>54.6</td><td>63.0</td><td>50.3</td><td>56.7</td><td>4.8</td></tr><tr><td>Qwen3-VL-2B†</td><td>62.7+13.1</td><td>75.4+17.8</td><td>73.5+18.5</td><td>89.3+13.9</td><td>88.1+9.0 77.8+14.5</td><td></td><td>51.5+24.2 38.9+8.3 45.2+16.2</td><td></td><td></td><td>13.4+5.2</td></tr><tr><td>Qwen3-VL-4B†</td><td>72.1+16.9</td><td>75.8+1.8</td><td>75.8+18.7</td><td>90.1+4.2</td><td>94.0+1.0 81.6+8.5</td><td></td><td></td><td>47.1+11.2 40.0+3.9 43.6+7.6</td><td></td><td>14.9+4.7</td></tr><tr><td>Qwen3-VL-8B†</td><td>76.3+17.0</td><td>75.9+4.4</td><td>75.5+14.8</td><td>93.4+6.6</td><td>94.4+6.3 83.1+9.8</td><td></td><td>59.1+18.9</td><td>47.1+5.8 53.1+12.3</td><td></td><td>19.1+7.9</td></tr><tr><td>InternVL3.5-8B†</td><td>64.0+22.3</td><td>61.0+5.6</td><td>58.6+16.4</td><td>75.7+21.6 83.6+3.8 68.6+14.0</td><td></td><td></td><td>70.1+7.1</td><td>57.8+7.5 63.9+7.2</td><td></td><td>9.5+4.7</td></tr></table>

The exposed capability gaps are actionable under targeted supervision. Models fine-tuned on VisualWorldBench-50K achieve substantial gains, particularly on the dimensions VisualWorldBench identifies as weak. The most striking case appears at the smallest scale: Qwen3-VL-2B<sup>†</sup> reaches 70.7% on MCQ and 45.2% on Avg, improving over its untuned counterpart by 42.7 and 25.0 percentage points and surpassing the untuned Qwen3-VL-8B instruct model (Avg 40.8%). Gains are consistent across all three Qwen3-VL scales and InternVL3.5-8B on MCQ, Grid-Sel, and Grid-Loc alike, confirming that the gaps are not merely diagnostic but can be effectively reduced through targeted fine-grained supervision.

## 4.3 Generalization to Public Fine-Grained Benchmarks

To assess whether gains from VisualWorldBench-50K generalize beyond VisualWorldBench, we evaluate the trained models on five widely used public fine-grained MCQ benchmarks (CUB-200, Stanford Dogs, Aircraft, Flowers, Pets), iNaturalist 2021, and WorldVQA.

SFT gains transfer consistently across architectures and benchmarks. Table 3 shows that VisualWorldBench-50K improves performance on all five public MCQ benchmarks, both iNat2021 splits, and WorldVQA across all four evaluated models. The strongest transfer appears in the 2B setting, where public-benchmark average rises by 14.5 pp and iNat21 average by 16.2 pp; the same pattern holds at larger scales and across architectures. These results confirm that the broad category coverage of VisualWorldBench-50K translates into consistent gains on both fine-grained recognition benchmarks and visual knowledge benchmarks, demonstrating its effectiveness as a general-purpose resource for improving visual world knowledge in MLLMs.

## 5 Analysis

## 5.1 Domain and Depth Diagnostics

Domain-level analysis reveals divergent weaknesses across model tiers. Figure 4a shows that all models perform unevenly across domains, but the profile diverges between closed-source and open-source families. The starkest example is fungus: all open-source models score lowest in this domain (e.g., Qwen3-VL-2B 9.8%, Qwen3-VL-8B 30.4%), yet Gemini 3.1 Pro achieves 91.5%. Within-model cross-domain gaps exceed 33 pp for Qwen3-VL-8B and 29 pp for Qwen3.5-27B, confirming that a single aggregate score can obscure important domain-specific weaknesses.

![](images/379d96abe8393061431cbcdd9f29a38ac639785e51ade256fd3f8f8e82f774d9.jpg)

![](images/49f86014cc5dbf33d3a73d10117fbfbecc075f1135204a817a086f49587ae88f.jpg)

![](images/a9eebd89e0d5344a7895ac57edb7a23349942e21bfe4f0f78743c933a28f2cd3.jpg)  
Figure 4: Capability profiles vary substantially across visual domains, taxonomy depth, and category exposure. (a) MCQ accuracy by visual domain (red ⋄: proprietary; blue ◦: open-source). Crossdomain gaps exceed 33 pp within individual models. (b) Grid-Sel accuracy by taxonomy depth for four open-source models (Qwen3.5-27B/9B/4B, Qwen3-VL-8B) at 2×2 and 3×3 resolutions. Performance degrades substantially at the deepest levels (7+). (c) Memorization vs. generalization (Qwen3-VL-2B): Base = instruct baseline (no SFT); Holdout = fine-tuned excluding records with held-out ground-truth labels; Full = fine-tuned on the complete SFT data. Selection and localization improve on categories held out from SFT target labels, with smaller gains on MCQ.

Performance degrades systematically with taxonomy depth. Figure 4b shows that representative open-source models consistently degrade from intermediate depths (4–6) to the deepest levels (7+), especially in the 3×3 setting: Qwen3.5-27B drops from 67.1% to 47.9% on Grid-Sel 3×3, a decline of 19.2 pp. Together, these domain and depth analyses show that aggregate scores mask systematic failure modes visible only under taxonomy-stratified evaluation.

## 5.2 Transfer under Target-Label Holdout

Using Qwen3-VL-2B, we randomly hold out 40% of benchmark categories and remove all SFT records whose ground-truth labels belong to these categories. On benchmark instances with held-out ground-truth labels, we compare the instruct baseline (no SFT), the holdout model (filtered SFT data), and the full model (complete SFT data). Experimental details are provided in Appendix J.

Contrastive selection and localization show stronger transfer than closed-set recognition. Figure 4c shows that the holdout model substantially outperforms the instruct baseline on Grid-Sel 3×3 (10.9 → 30.3) and Grid-Loc 3×3 (20.2 → 41.2), although these categories do not appear as ground-truth labels in its SFT data. On MCQ, by contrast, the holdout model improves only modestly (+7.1 pp) while the full model gains +42.6 pp, suggesting greater sensitivity of MCQ to the target-label holdout setting. These results support transfer beyond categories directly supervised as SFT targets, with larger gains on selection and localization than on MCQ.

## 6 Conclusion

We presented VisualWorldBench, a taxonomy-grounded multi-task diagnostic benchmark evaluating 15,479 ground-truth categories within a taxonomy of 21,291 leaf-node classes, together with VisualWorldBench-50K, a 50K+-category training resource. Systematic evaluation across closed-source, open-source, and fine-tuned models shows that visual world knowledge is inherently multi-dimensional: strong closed-set recognition does not imply strong contrastive selection, precise localization, or open-world naming, and these dissociations persist across model families. Under targeted supervision, fine-tuning on VisualWorldBench-50K yields substantial gains on VisualWorld-Bench and transfers consistently to public fine-grained recognition and visual knowledge benchmarks. Under target-label holdout, contrastive selection and localization show larger gains over the instruct baseline than closed-set recognition, supporting transfer to categories not directly supervised as SFT targets. Together, these findings suggest that visual world knowledge should be evaluated and improved as a multi-dimensional capability rather than reduced to a single recognition score.

## References

[1] OpenAI. GPT-4o system card. https://openai.com/index/gpt-4o-system-card/, 2024.

[2] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-VL: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.

[3] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning (ICML), 2021.

[4] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.

[5] Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. The Caltech-UCSD Birds-200-2011 dataset. Technical Report CNS-TR-2011-001, California Institute of Technology, 2011.

[6] Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao, and Fei-Fei Li. Novel dataset for fine-grained image categorization: Stanford Dogs. In CVPR Workshop on Fine-Grained Visual Categorization, 2011.

[7] Subhransu Maji, Esa Rahtu, Juho Kannala, Matthew Blaschko, and Andrea Vedaldi. Fine-grained visual classification of aircraft. arXiv preprint arXiv:1306.5151, 2013.

[8] Grant Van Horn, Elijah Cole, Sara Beery, Kimberly Wilber, Serge Belongie, and Oisin Mac Aodha. Benchmarking representation learning for natural world image collections. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12884–12893, 2021. URL https://openaccess.thecvf.com/content/CVPR2021/html/Van\_Horn\_Benchmarking\_ Representation\_Learning\_for\_Natural\_World\_Image\_Collections\_CVPR\_2021\_paper. html.

[9] Zahra Gharaee, Scott C. Lowe, ZeMing Gong, Pablo Millan Arias, Nicholas Pellegrino, Austin T. Wang, Joakim Bruslund Haurum, Iuliia Zarubiieva, Lila Kari, Dirk Steinke, Graham W. Taylor, Paul Fieguth, and Angel X. Chang. BIOSCAN-5M: A multimodal dataset for insect biodiversity. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

[10] Yang Chen, Hexiang Hu, Yi Luan, Haitian Sun, Soravit Changpinyo, Alan Ritter, and Ming-Wei Chang. Can pre-trained vision and language models answer visual information-seeking questions? In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2023.

[11] Xianfu Cheng, Wei Zhang, Shiwei Zhang, Jian Yang, Xiangyuan Guan, Xianjie Wu, Xiang Li, Ge Zhang, Jiaheng Liu, Yuying Mai, et al. Simplevqa: Multimodal factuality evaluation for multimodal large language models. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 4637–4646, 2025.

[12] Runjie Zhou, Youbo Shao, Haoyu Lu, Bowei Xing, Tongtong Bai, Yujie Chen, Jie Zhao, Lin Sui, Haotian Yao, Zijia Zhao, et al. Worldvqa: Measuring atomic world knowledge in multimodal large language models. arXiv preprint arXiv:2602.02537, 2026.

[13] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. MMBench: Is your multi-modal model an all-around player? In European Conference on Computer Vision (ECCV), 2024.

[14] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. MMMU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert AGI. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9556–9567, 2024. URL https://openaccess.thecvf.com/content/CVPR2024/html/Yue\_MMMU\_ A\_Massive\_Multi-discipline\_Multimodal\_Understanding\_and\_Reasoning\_Benchmark\_ for\_CVPR\_2024\_paper.html.

[15] Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, Yu Su, Wenhu Chen, and Graham Neubig. MMMU-Pro: A more robust multi-discipline multimodal understanding benchmark. arXiv preprint arXiv:2409.02813, 2024. URL https://arxiv.org/abs/2409.02813.

[16] Hong-Tao Yu, Yuxin Peng, Serge Belongie, and Xiu-Shen Wei. Benchmarking large vision-language models on fine-grained image tasks: A comprehensive evaluation. In International Conference on Learning Representations (ICLR), 2026. URL https://openreview.net/forum?id=cVc74MLspe.

[17] Grant Van Horn, Steve Branson, Ryan Farrell, Scott Haber, Jessie Barry, Panos Ipeirotis, Pietro Perona, and Serge Belongie. Building a bird recognition app and large scale dataset with citizen scientists: The fine print in fine-grained dataset collection. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015.

[18] Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for fine-grained categorization. In IEEE International Conference on Computer Vision Workshops (ICCVW), 2013.

[19] Linjie Yang, Ping Luo, Chen Change Loy, and Xiaoou Tang. A large-scale car dataset for fine-grained categorization and verification. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015.

[20] Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. Food-101 – mining discriminative components with random forests. In European Conference on Computer Vision (ECCV), 2014.

[21] Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. In Proceedings ofthe Indian Conference on Computer Vision, Graphics and Image Processing, 2008.

[22] Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman, and C. V. Jawahar. Cats and dogs. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2012.

[23] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, and Feng Zhao. Are we on the right way for evaluating large vision-language models? In Advances in Neural Information Processing Systems (NeurIPS), 2024.

[24] Thomas Mensink, Jasper Uijlings, Lluís Castrejón, Arushi Goel, Felipe Cadar, Howard Zhou, Fei Sha, André Araujo, and Vittorio Ferrari. Encyclopedic VQA: Visual questions about detailed properties of fine-grained categories. In IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

[25] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. ReferItGame: Referring to objects in photographs of natural scenes. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2014.

[26] Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan Yuille, and Kevin Murphy. Generation and comprehension of unambiguous object descriptions. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[27] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal LLM’s referential dialogue magic. arXiv preprint arXiv:2306.15195, 2023.

[28] Haoxuan You, Haotian Zhang, Zhe Gan, Xianzhi Du, Bowen Zhang, Zirui Wang, Liangliang Cao, Shih-Fu Chang, and Yinfei Yang. Ferret: Refer and ground anything anywhere at any granularity. In International Conference on Learning Representations (ICLR), 2024.

[29] Hanoona Rasheed, Muhammad Maaz, Sahal Shaji, Abdelrahman Shaker, Salman Khan, Hisham Cholakkal, Rao M. Anwer, Eric Xing, Ming-Hsuan Yang, and Fahad S. Khan. Glamm: Pixel grounding large multimodal model. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[30] Yuanhan Zhang, Qinghong Sun, Yichun Zhou, Zexin He, Zhenfei Yin, Kun Wang, Lu Sheng, Yu Qiao, Jing Shao, and Ziwei Liu. Bamboo: Building mega-scale vision dataset continually with human–machine synergy. International Journal of Computer Vision, 133(8):5806–5821, 2025.

[31] Google Gemini Team. Gemini 3.1 Pro: A smarter model for your most complex tasks. https://blog. google/innovation-and-ai/models-and-research/gemini-models/gemini-3-1-pro/, 2026.

[32] OpenAI. Introducing GPT-5.4. https://openai.com/index/introducing-gpt-5-4/, 2026.

[33] OpenAI. Introducing GPT-5.2. https://openai.com/index/introducing-gpt-5-2/, 2025.

[34] ByteDance Seed. Seed2.0 model card: Towards intelligence frontier for real-world complexity. arXiv preprint arXiv:2607.00248, 2026. URL https://arxiv.org/abs/2607.00248.

[35] Xiaomi MiMo Team. Mimo-v2-omni. https://mimo.xiaomi.com/mimo-v2-omni, 2026.

[36] Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, SH Cai, Yuan Cao, Y Charles, HS Che, Cheng Chen, Guanduo Chen, et al. Kimi k2.5: Visual agentic intelligence. arXiv preprint arXiv:2602.02276, 2026.

[37] Qwen Team. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631, 2025.

[38] Qwen Team. Qwen3.5: Towards native multimodal agents. https://qwen.ai/blog?id=qwen3.5, 2026.

[39] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025.

[40] Gemma Team. Gemma 4 technical report. arXiv preprint arXiv:2607.02770, 2026. URL https: //arxiv.org/abs/2607.02770.

[41] Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL): System Demonstrations, 2024.

[42] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations (ICLR), 2022. URL https://openreview.net/forum?id=nZeVKeeFYf9.

[43] Bryan A. Plummer, Liwei Wang, Chris M. Cervantes, Juan C. Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In IEEE International Conference on Computer Vision (ICCV), 2015.

[44] Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. G-eval: NLG evaluation using GPT-4 with better human alignment. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 2511–2522, 2023. doi: 10.18653/v1/2023.emnlp-main. 153. URL https://aclanthology.org/2023.emnlp-main.153/.

[45] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLMas-a-judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems, 2023.

[46] Jacob Cohen. A coefficient of agreement for nominal scales. Educational and Psychological Measurement, 20(1):37–46, 1960.

## Appendix

## A Limitations

Our SFT experiments focus on the Qwen3-VL and InternVL3.5 backbone families, so findings may not fully generalize to other architectures. Open-QA uses EM+LLM Judge; stronger alias-aware and hierarchy-aware metrics remain future work. Grid-Sel and Grid-Loc use controlled 2×2 / 3×3 layouts that are diagnostically useful but do not fully match natural multi-object scenes, and Grid-Loc may additionally reflect sensitivity to normalized box formatting. The benchmark spans eight visual domains, but specialized areas such as medical imaging or satellite imagery remain under-covered.

## B Broader impacts

This work provides a diagnostic benchmark and training resource for fine-grained visual recognition. It can support research on species and object recognition, reveal gaps in models’ visual world knowledge, and help researchers compare capabilities across complementary tasks.

Potential negative impacts arise from misidentification and over-reliance on benchmark performance. Incorrect species or object predictions could mislead users, particularly when applied to decisions such as whether a plant or fungus is safe to consume. The benchmark’s uneven domain coverage and controlled grid layouts also limit how well aggregate scores represent performance in natural settings.

The benchmark is intended for research evaluation. Its scores do not establish safety or reliability for field species identification, food-safety judgments, or other consequential applications. Such uses require application-specific validation and appropriate expert review; model predictions should not serve as the sole basis for these decisions.

## C Benchmark Statistics

Table 4: Statistics of VisualWorldBench. #Cls: unique canonical ground-truth names in the released benchmark split. The benchmark is built on a taxonomy of 21,291 leaf-node classes; the released evaluation subset covers 15,479 unique ground-truth categories, while nearly all taxonomy classes appear as distractors. Grid-Loc is evaluated at IoU threshold τ=0.5 following the visual grounding standard. Grid tasks are evaluated at two spatial resolutions (2×2 and 3×3) to probe fine-grained discrimination at different spatial granularities.
<table><tr><td>Task</td><td>Format</td><td>#Samples</td><td>#Cls</td><td>Metric</td></tr><tr><td>MCQ</td><td>20-option MC</td><td>14,717</td><td>11,466</td><td>Accuracy</td></tr><tr><td>Grid-Sel 2×2</td><td>Select cell</td><td>4,085</td><td>3,726</td><td>Accuracy</td></tr><tr><td>Grid-Sel 3×3</td><td>Select cell</td><td>4,125</td><td>3,785</td><td>Accuracy</td></tr><tr><td>Grid-Loc 2×2</td><td>Output bbox</td><td>3,959</td><td>3,613</td><td>IoU@0.5</td></tr><tr><td>Grid-Loc 3×3</td><td>Output bbox</td><td>3,446</td><td>3,204</td><td>IoU@0.5</td></tr><tr><td>Open-QA</td><td>Free-form text</td><td>2,209</td><td>2,187</td><td>EM + LLM Judge</td></tr><tr><td>Total</td><td></td><td>32,541</td><td>15,479 unique</td><td></td></tr></table>

Release pool and retention. The released benchmark is assembled from an intermediate taskspecific pool that already satisfies the main verification and release-stage constraints. For MCQ, this pool contains 22,356 samples before the final image-quality filter; the corresponding counts are 5,559 for Grid-Sel 2×2, 5,512 for Grid-Sel 3×3, 5,383 for Grid-Loc 2×2, 4,606 for Grid-Loc 3×3, and 6,169 for Open-QA. After automated image-quality filtering that removes low-quality images (too dark, too small, blurry, low saturation, etc.) and additional quality-score-based curation for Grid and Open-QA tasks, the released benchmark retains 65.8% of MCQ, 73.5% of Grid-Sel 2×2, 74.8% of Grid-Sel 3×3, 73.5% of Grid-Loc 2×2, 74.8% of Grid-Loc 3×3, and 35.8% of Open-QA, yielding a total of 32,541 evaluation instances.

Table 5: Per-domain sample counts in the released benchmark split for MCQ. Grid tasks and Open-QA follow the same 8-domain taxonomy; per-domain breakdowns are omitted for brevity. The released split is intentionally not domain-balanced, so sparse domains such as Scene and especially Other should not be over-interpreted as standalone evaluation subsets.
<table><tr><td>Domain</td><td>MCQ</td></tr><tr><td>Animal</td><td>4,011</td></tr><tr><td>Artifact</td><td>2,535</td></tr><tr><td>Food</td><td>2,411</td></tr><tr><td>Plant</td><td>2,030</td></tr><tr><td>Fungus</td><td>1,696</td></tr><tr><td>Transport</td><td>1,664</td></tr><tr><td>Scene</td><td>368</td></tr><tr><td>Other</td><td>2</td></tr><tr><td></td><td></td></tr><tr><td>Total</td><td>14,717</td></tr></table>

Table 6: MCQ option composition in the released benchmark split. Percentages use construction-time option types over all slots, including the ground-truth option; these types are omitted from public mcq.jsonl.
<table><tr><td>Option type</td><td>Count</td><td>Ratio</td></tr><tr><td>Sibling</td><td>111,866</td><td>38.0%</td></tr><tr><td>Random</td><td>108,327</td><td>36.8%</td></tr><tr><td>Near</td><td>59,430</td><td>20.2%</td></tr><tr><td>GT</td><td>14,717</td><td>5.0%</td></tr></table>

GT-category filtering check. We additionally verify that the released MCQ split does not contain low-confidence Bamboo leaf nodes from the initial taxonomy audit. We compare the 6,948 lowconfidence categories identified in the initial taxonomy audit against the 11,466 unique constructiontime ground-truth IDs corresponding to the released MCQ split (14,717 samples in total). The intersection is zero, confirming that the released MCQ ground-truth labels exclude all audited Bamboo leaf categories with name\_match $< 0 . 7$ . The simplified public files do not expose the construction-time gt\_id field. This verification applies only to ground-truth labels and does not constrain distractor options.

## D Taxonomy Construction

The benchmark taxonomy contains 21,291 leaf-node classes across eight visual domains: animal, plant, fungus, artifact, transport, food, scene, and other. The taxonomy is derived from the Bamboo category tree [30] and further expanded with supplementary public sources to improve category coverage across domains. After normalization, deduplication, filtering, and manual audit over both the source category set and the associated image pool, only leaf-level categories with visually identifiable image evidence are retained as benchmark ground-truth labels. VisualWorldBench-50K further expands category coverage beyond the benchmark taxonomy by incorporating additional public fine-grained datasets, including iNaturalist 2021.

Normalization and filtering. Category names are normalized to a canonical English label with optional aliases. Near-duplicate labels, spelling variants, and parent-child duplicates are merged when they refer to the same visual category. Categories are removed when they are non-visual, overly broad for fine-grained recognition, visually unverifiable from available images, or incompatible with the benchmark’s taxonomy structure.

## E Task Construction

All four tasks are derived from the same taxonomy-grounded candidate pool. Public files provide task inputs, answer annotations, and domain/depth metadata, but omit construction-time category IDs and parent-category fields. This shared construction makes the task suite diagnostically comparable rather than a collection of unrelated benchmarks.

Table 7: Per-domain taxonomy statistics. #Leafclasses: total leaf-node categories in the taxonomy. #GT classes: unique ground-truth categories appearing in the released benchmark across all tasks. Nearly all 21,291 leaf classes additionally appear as distractors in MCQ or Grid tasks, even when they are not used as ground-truth labels.
<table><tr><td>Domain</td><td>#Leaf classes</td><td>#GT classes</td><td>Max depth</td></tr><tr><td>Animal</td><td>6,620</td><td>5,323</td><td>13</td></tr><tr><td>Plant</td><td>2,198</td><td>1,786</td><td>8</td></tr><tr><td>Fungus</td><td>1,690</td><td>1,492</td><td>6</td></tr><tr><td>Artifact</td><td>4,819</td><td>2,753</td><td>14</td></tr><tr><td>Transport</td><td>2,597</td><td>1,859</td><td>8</td></tr><tr><td>Food</td><td>2,373</td><td>1,853</td><td>9</td></tr><tr><td>Scene</td><td>433</td><td>359</td><td>7</td></tr><tr><td>Other</td><td>561</td><td>54</td><td>10</td></tr><tr><td>Total</td><td>21,291</td><td>15,479</td><td>14</td></tr></table>

MCQ construction. Each released MCQ sample contains an image path, 20 candidate category names as strings in options, and a zero-based correct-answer index in gt\_index, together with qid and domain/depth metadata. The sibling, near, and random distinctions describe distractor construction; the simplified public files do not include per-option category IDs or type labels. The benchmark is built by reorganizing and normalizing previously quality-checked bench pools rather than sampling directly from the full raw image archive. For each ground-truth class, we retain at most two benchmark samples, prioritizing wider class coverage first and then applying the release-balancing rules described below.

Grid-Sel construction. Each Grid-Sel sample is a composite image formed by tiling one target image together with distractor images from taxonomically neighboring categories. We construct both 2×2 and 3×3 grids. Distractors are drawn with a sibling-first policy: we first sample direct siblings of the target from the taxonomy, and only expand to tree distance 2 when the sibling pool is insufficient. Public Grid-Sel records provide the target category, prompt, and cell answer, but omit per-grid sibling counts and taxonomy-distance diagnostics. Within a single grid, all source images are distinct, while exact image-set duplicates across grids are disallowed by a grid-level deduplication key.

Grid-Loc construction. Grid-Loc shares the same composite images as Grid-Sel but changes the output space from discrete cell prediction to precise bounding-box localization. The target bounding box is not the full target cell. Instead, it is the target object’s bounding box after propagating the original object annotation through the crop/pad transform used to place the image into its grid cell, and then mapping that box into the coordinates of the full composite grid image. The released benchmark stores this target box as bbox\_norm, with coordinates normalized to [0, 1000] relative to the full grid image width and height.

Open-QA construction. Open-QA is derived from high-quality MCQ upstream samples rather than created independently from the raw data pool. Public Open-QA records provide image paths, questions, alias sets, and ancestor metadata; they omit the upstream MCQ linkage field source\_qid. These metadata support exact matching, synonym handling, and future hierarchy-aware analysis, although the main paper reports only EM+LLM Judge.

Prompt design. The benchmark uses direct task-specific user prompts with constrained answer formats. MCQ asks the model to return a single option index from 20 candidates. Grid-Sel asks for a single cell number. Grid-Loc requires exactly one JSON-style array [x1, y1, x2, y2] normalized to [0, 1000]. Open-QA asks the model to identify the object as specifically as possible and place the final answer on a dedicated Answer: line. These prompts are instantiated deterministically from the benchmark metadata and are described further in Section H.

![](images/23cbf64f7dbffb8d4d86df773d6a069dde7d40e376f533f848b5b8211c7e96bf.jpg)  
Figure 5: MCQ and Open-QA task examples. (a)–(d) MCQ: closed-set recognition with 20 taxonomyaware options including sibling, near, and random distractors drawn from the taxonomy hierarchy. (e)–(f) Open-QA: the model must generate the precise fine-grained category name without any candidate set.

## F Quality Control and Curation

Annotation verification. Each candidate sample is first evaluated by three frontier closed-source models (Gemini 3 Pro, GPT-5, GPT-4o). These QC assistants are distinct from the evaluated model versions / endpoints in Section 4, although some belong to the same product family (e.g., GPT-5 for QC versus GPT-5.2 / GPT-5.4 for evaluation). Samples for which at least two of the three models produce an incorrect answer are flagged for human review. These model outputs are used only for triage: flagged samples are not removed automatically, and final keep/remove decisions are made under predefined human criteria (label correctness, visual identifiability, ambiguity, and leakage checks). Human annotators then assign each flagged sample to one of three outcomes: labeling error, ambiguous sample, or valid reviewed case. Labeling errors and ambiguous samples are removed, while valid reviewed cases are retained.

## F.1 Audit of Non-flagged Samples

Random audit of non-flagged samples. To test whether model-assisted triage systematically misses problematic samples that most assisting models answer correctly, we additionally conduct a random human audit on non-flagged samples from each task. The audit is performed under the same criteria as the main verification pass and is used only to assess the robustness of the QC pipeline, not to redefine the benchmark. Table 9 reports the audit protocol and issue rates.

![](images/e5cd7f326642f7632aca591c16208e213a9bf7d54f0b46776ce31243357974fd.jpg)  
(a) Grid-Sel | Artifact Target: kachina (Cell 4) Cell 1: golliwog Cell 2: matryoshka doll Cell 3: Hello Kitty GT bbox

![](images/ef2011a95945a70b75e322fb6b2dfefef03a80f6d814d88c4b1ddb05ee858c7f.jpg)  
(b) Grid-Sel | Plant Target: sweetsop (Cell 2) Cell 1: asimina triloba Cell 3: annona glabra Cell 4: soursop GT bbox

![](images/1b81dc2a127dc195123078af76eda5fd656317d7ce6a989aeb0d70c0dddce5c5.jpg)  
(c) Grid-Loc | Food Target: wheat berry (Cell 3) Cell 1: rye Cell 2: amaranth Cell 4: Chenopodium quinoa GT bbox

![](images/64f8c42fbdcc7bec6990adf4f12f707ede742784c14195409f2c488b1d05b35a.jpg)  
(d) Grid-Loc | Animal Target: phaedyma columella (Cell 2) Cell 1: gryllus bimaculatus Cell 3: rhodometra sacraria Cell 4: suastus gremius GT bbox  
Figure 6: Grid task examples (2×2). (a)–(b) Grid-Sel: the model selects which cell contains the target category from a composite grid of taxonomically similar images. (c)–(d) Grid-Loc: the model outputs a bounding box to precisely localize the target object within the grid. Red boxes indicate the ground-truth bounding box. All distractors are sibling or near categories from the taxonomy.

Grid-specific checks. For Grid-Sel and Grid-Loc, verification is performed at the per-cell level to ensure that each grid cell matches its assigned category. Samples are removed if the target object occupies less than 1% of the image area, if the grid composition is visually corrupted, or if the target category is not visually distinguishable from the distractor cells. Grid-Loc uses the same composite-image pool as Grid-Sel after alignment, but applies the IoU@0.5 evaluation threshold.

Open-QA checks. Open-QA candidates are filtered in two stages. First, samples whose groundtruth categories are not visually identifiable are removed. Second, a frontier model performs visual quality inspection for image quality, text leakage, identifiability, and ground-truth correctness; only samples receiving a KEEP verdict are retained.

Release construction and auxiliary metadata. Response-pattern labels are derived from frontiermodel outputs and used only as QC / audit metadata for identifying potentially problematic samples for additional review. They are not release-construction criteria, stand-alone validity criteria, or the basis for defining benchmark difficulty.

Table 8: Quality-control statistics. The Release pool column reports the count after the task-specific intermediate filtering stage (release balancing for MCQ and Grid-Sel, visual QC for Grid-Loc and Open-QA) but before the final image-quality filter. The Final column reports the count after the additional automated image-quality filter.
<table><tr><td>Task</td><td>Candidate pool</td><td>Release pool</td><td>Final</td><td>Retention</td></tr><tr><td>MCQ</td><td>47,697</td><td>22,356</td><td>14,717</td><td>30.9%</td></tr><tr><td>Grid-Sel 2×2</td><td>19,131</td><td>5,559</td><td>4,085</td><td>21.4%</td></tr><tr><td>Grid-Sel 3×3</td><td>18,064</td><td>5,512</td><td>4,125</td><td>22.8%</td></tr><tr><td>Grid-Loc 2×2</td><td>18,681</td><td>5,383</td><td>3,959</td><td>21.2%</td></tr><tr><td>Grid-Loc 3×3</td><td>15,335</td><td>4,606</td><td>3,446</td><td>22.5%</td></tr><tr><td>Open-QA</td><td>10,000</td><td>6,169</td><td>2,209</td><td>22.1%</td></tr><tr><td>Total</td><td>128,908</td><td></td><td>32,541</td><td>25.2%</td></tr></table>

Table 9: Random audit of non-flagged samples. 200 samples per task are drawn uniformly at random from the non-flagged pool (seed = 42) and reviewed under the same criteria as the main verification pass. Issue: any sample requiring correction or removal because of label error, ambiguity, nonidentifiability, leakage, or corrupted composition.
<table><tr><td>Task</td><td>Audited</td><td>Issue</td><td>Issue rate</td><td>Main issue types</td></tr><tr><td>MCQ</td><td>200</td><td>6</td><td>3.0%</td><td>ambiguity</td></tr><tr><td>Grid-Sel</td><td>200</td><td>8</td><td>4.0%</td><td>cell mismatch</td></tr><tr><td>Grid-Loc</td><td>200</td><td>8</td><td>4.0%</td><td>box misalignment</td></tr><tr><td>Open-QA</td><td>200</td><td>5</td><td>2.5%</td><td>alias ambiguity</td></tr><tr><td>Total</td><td>800</td><td>27</td><td>3.4%</td><td></td></tr></table>

MCQ release construction. For MCQ, we begin the final release stage from an intermediate pool of 22,356 samples that already satisfies the main verification and release-balancing constraints. Importantly, samples are not removed simply because reference models answer them correctly; rather, model disagreement and repeated failure patterns are used only to trigger additional human verification for potentially problematic cases. We then apply a final automated image-quality filter that removes visually unreliable images (too dark, too small, blurry, low saturation), yielding 14,717 released MCQ samples.

Grid-Sel release construction follows the same principle: final release balancing is carried out over verified samples while jointly considering distractor structure, domain coverage, and visual quality, followed by five-dimension visual QC (target-cell quality, text leakage, distinguishability, groundtruth correctness, overall verdict) using dual frontier models (Gemini 3 Pro primary, GPT-5 fallback), retaining only KEEP samples.

Grid-Loc is evaluated on the same composite images as Grid-Sel after alignment; the IoU@0.5 threshold follows the standard used in visual grounding benchmarks.

Open-QA. Three frontier models (Gemini 3 Pro, GPT-5, GPT-4o) independently attempt each question via exact match. These outputs define three auxiliary response-pattern buckets: full\_consensus (3/3 correct), partial\_consensus (1–2/3 correct), and zero\_consensus (0/3 correct). These buckets are retained only as auxiliary audit metadata; the released set is assembled from verified samples under domain filtering (removing categories with non-visual ground truths), the same five-dimension visual QC as Grid, and automated image-quality filtering, yielding 2,209 samples.

## F.2 Sanity Check on Model-Assisted QC

By design, the frontier models used for annotation triage and QC (Gemini 3 Pro, GPT-5, GPT-4o) are version-distinct from the evaluated model endpoints in Section 4, eliminating direct circular dependence between curation and evaluation at the deployed-model level. These model outputs are used to identify samples that require additional verification; they are not used as automatic deletion rules for samples answered correctly by the assisting models. As additional evidence that the main findings are not artifacts of any particular assisting model family, Table 10 confirms that the cross-task capability dissociations (large MCQ–Open and MCQ–Loc gaps) persist across diverse closed-source, open-source, and SFT model families.

Table 10: Sanity check: cross-task capability dissociations across model families. MCQ–Open: difference between closed-set MCQ accuracy and Open-QA accuracy. MCQ–Loc: difference between MCQ accuracy and Grid-Loc 3×3 accuracy. Large gaps persist across all families, confirming that the dissociations are general rather than artifacts of any assisting model.
<table><tr><td>Model</td><td>Family</td><td>MCQ</td><td>Open</td><td>MCQ-Open</td><td>MCQ-Loc</td></tr><tr><td>Gemini 3.1 Pro</td><td>Closed-source</td><td>86.0</td><td>60.6</td><td>25.4</td><td>2.7</td></tr><tr><td>Qwen3.5-27B</td><td>Open-source</td><td>59.1</td><td>28.7</td><td>30.4</td><td>2.9</td></tr><tr><td>Gemma-4-31B</td><td>Open-source</td><td>60.5</td><td>27.2</td><td>33.3</td><td>24.4</td></tr><tr><td>InternVL3.5-8B</td><td>Open-source</td><td>35.5</td><td>10.3</td><td>25.2</td><td>28.1</td></tr><tr><td>Qwen3-VL-8B†</td><td>SFT</td><td>80.0</td><td>33.9</td><td>46.1</td><td>15.7</td></tr></table>

## G Training Details

All SFT models are trained with LoRA in LLaMA-Factory [41] for 3 epochs using AdamW, learning rate $1 0 ^ { - 4 }$ , and an effective global batch size of 1024. Unless otherwise stated, we use LoRA [42] with rank 8 and target all trainable modules. VisualWorldBench-50K is anchored by large-scale closed-set MCQ supervision, with a reasoning-augmented MCQ subset and smaller grounding and open-ended QA subsets providing complementary signals.

Table 11: VisualWorldBench-50K composition. The resource is dominated by MCQ supervision; reasoning-augmented MCQ is a subset of the MCQ records rather than an additional disjoint pool.
<table><tr><td>Format</td><td>#Records</td><td>#Classes</td><td>Primary signal</td></tr><tr><td>MCQ</td><td>~6.9M</td><td>48,170</td><td>Closed-set recognition</td></tr><tr><td>Reasoning-augmented MCQ</td><td>~0.6M</td><td>25,376</td><td>Fine-grained visual rationale</td></tr><tr><td>Bounding-box grounding</td><td>~0.4M</td><td>31,122</td><td>Spatial grounding</td></tr><tr><td>Open-ended QA</td><td>~0.4M</td><td>31,033</td><td>Category-name generation</td></tr><tr><td>Total</td><td>~7.7M</td><td>50K+</td><td>Multi-format fine-grained supervision</td></tr></table>

For each SFT model, we inspect late-stage checkpoints to confirm convergence and report one representative converged checkpoint in the main paper. Table 12 summarizes the compute used by the four main SFT runs. Compute usage for the holdout run is not reported separately. The 2B run used per-device batch size 64 and gradient accumulation 2 on 8 H200 GPUs; for the 4B and 8B runs, we adjusted the per-device batch size and accumulation settings to keep the same effective global batch size.

Table 12: Compute resources for the main Qwen3-VL SFT, InternVL3.5 SFT runs. GPU-hours are computed as wall-clock hours multiplied by the number of GPUs.
<table><tr><td>Model</td><td>GPUs</td><td>Epochs</td><td>Global batch</td><td>Time</td><td>GPU-hours</td></tr><tr><td>Qwen3-VL-2B</td><td>8×H200</td><td>3</td><td>1024</td><td>32h</td><td>256</td></tr><tr><td>Qwen3-VL-4B</td><td>16×H200</td><td>3</td><td>1024</td><td>30h</td><td>480</td></tr><tr><td>Qwen3-VL-8B</td><td>16×H200</td><td>3</td><td>1024</td><td>45h</td><td>720</td></tr><tr><td>InternVL3.5-8B</td><td>16×H200</td><td>3</td><td>1024</td><td>41h</td><td>656</td></tr></table>

## H Evaluation Details

Benchmark evaluation uses zero-shot prompting with greedy decoding. MCQ and Grid-Sel are scored by accuracy, Grid-Loc uses IoU@0.5 following the standard threshold in visual grounding benchmarks [25, 26, 43], and Open-QA uses EM+LLM Judge. Grid-Sel and Grid-Loc are evaluated at both 2×2 and 3×3 resolutions; we report both settings as well as their within-task average. For the reported average score, we take the arithmetic mean of MCQ, Sel Avg (mean of 2×2 and 3×3), Loc Avg (mean of 2×2 and 3×3), and Open-QA under EM+LLM Judge.

Prompt templates and output constraints. MCQ uses a prompt that presents the image together with 20 indexed candidate categories and instructs the model to output only the selected index. Grid-Sel uses a prompt of the form: “You are given a N×N grid image. Cells are numbered leftto-right, top-to-bottom. Which cell contains the target category: {class\_name}?” and requires a single-number answer. Grid-Loc uses a localization prompt that asks the model to return exactly one bounding box in the format [x1, y1, x2, y2]. The coordinate convention is normalized to [0, 1000] relative to the full composite image, where (0, 0) is the top-left corner and (1000, 1000) is the bottom-right corner. Open-QA asks the model to identify the object as specifically as possible, provide one brief evidence sentence, and then output the final prediction on a new line as Answer: <category name>.

## H.1 Prompt Templates

Below we provide the exact prompt templates used for each task. Placeholders are shown in {braces}. All prompts are zero-shot with no system prompt.

MCQ Prompt   
Look at the object(s) in the image and choose the single best category from   
1-{N}. Only output the number.   
1: {option\_1}   
2: {option\_2}   
.   
20: {option\_20}

## Grid-Sel Prompt

You are given a {N}x{N} grid image. Cells are numbered left-to-right, top-to-bottom as 1, 2, ..., {N×N}. Which cell contains the target category: {class\_name}? Answer with a single number. Do not output any other text.

## Grid-Loc Prompt

Locate the target object: {class\_name}.   
Output the bounding box as [x1, y1, x2, y2] where:   
(x1, y1) = top-left corner   
(x2, y2) = bottom-right corner   
Coordinates normalized to [0, 1000] (0=left/top edge, 1000=right/bottom   
edge)   
Example: [100, 200, 500, 800]   
Output only the four numbers in brackets.   
Return exactly one bounding box for the target object.   
Your entire reply must be exactly one single-line JSON array: [x1, y1, x2,   
y2]   
Rules:   
Output exactly one JSON array with four numeric coordinates.   
Coordinates must be normalized to [0,1000].   
Do not output markdown, code fences, labels, keys, comments, explanations,   
prefixes, or suffixes.   
- Do not output multiple boxes, alternatives, candidate answers, or reasoning.   
- If uncertain, still output your single best box in the required format.

Open-QA Prompt   
Identify the object in the image as specifically as possible.   
{question}   
First give one brief sentence describing the key visual evidence.   
Then give the final answer on a new line in the format:   
Answer: <category name>   
Keep the evidence brief.   
Do not add anything after the answer line.

Notes. (1) MCQ options are numbered starting from 1; the model is expected to output a single integer. (2) Grid-Sel and Grid-Loc share the same composite grid image; only the output format differs (cell index vs. bounding box). (3) Grid-Loc coordinates follow the [0, 1000] normalization convention where (0, 0) is the top-left and (1000, 1000) is the bottom-right of the full composite image. (4) In Open-QA, {question} is the question provided with each sample and uses domain-aware phrasing (e.g., “What kind of bird/flower/fungus. . . ”) to elicit fine-grained category-level responses rather than coarse object descriptions.

Answer parsing. For MCQ and Grid-Sel, a valid discrete answer index is extracted from the model response using fixed parsing rules and compared with the ground-truth index. Responses from which a valid index cannot be extracted are scored as incorrect. For Grid-Loc, the evaluator searches the response for bracketed sequences of four numbers and uses the first matching sequence as the predicted box; surrounding explanatory text is permitted. If no bracketed sequence is found, the evaluator falls back to the first four numeric values in the response. Coordinates are clipped to [0, 1000], and reversed coordinate pairs are reordered along each axis. The resulting box is compared against the stored normalized target box using IoU@0.5. Responses from which four numeric values cannot be extracted are treated as parsing failures. For Open-QA, the evaluator extracts the final answer line, normalizes the resulting string, applies exact matching against the canonical label and alias set, and forwards only non-exact cases to the LLM judge.

Decoding and failure handling. All reported results use greedy decoding with temperature 0. The benchmark prompts do not rely on separate task-specific system prompts. For Grid-Loc, parsing failures among records not marked as request errors receive an IoU of zero and remain in the accuracy denominator. Records marked as request errors are counted separately and excluded from the accuracy denominator. Multiple-box responses are scored using the first extracted box and are additionally recorded in the multiple-box statistics.

## H.2 Open-QA Judging

Open-QA answers are scored in three stages. We first extract the model’s final answer string, then match it against the canonical label and alias set. For non-exact matches, a fixed frontier LLM judge (Gemini 2.5 Pro) is used to decide whether the prediction is semantically equivalent to the gold category, including capitalization differences, common-name/scientific-name variants, and accepted synonyms; this follows recent LLM-as-a-judge evaluation practice [44, 45]. The judge is instructed not to accept predictions that only name a parent category or a visually similar sibling category.

Open-QA Judge Prompt (System)   
You are a taxonomy and fine-grained recognition expert.   
Your task: determine whether a model’s prediction refers to the SAME specific   
species/object as the ground truth.   
Be strict –- only judge "correct" if they refer to the exact same species or   
object (synonyms, common name variants, and minor spelling differences are   
acceptable).   
If the prediction is at a coarser level (e.g., genus instead of species, or   
"bird" instead of "American Robin"), judge "wrong".

```jsonl
Open-QA Judge Prompt (User)
Ground truth: {gt_name}
Accepted aliases: {aliases}
Model’s prediction: {pred_text}
Does the model’s prediction refer to the same specific species/object as the
ground truth?
Output exactly one JSON object (no other text):
{"judgment": "correct" or "wrong", "reason": "one sentence explanation"}
```  
Only samples that fail automated exact matching against the canonical label and alias set are forwarded to the LLM judge; exact matches are scored directly without an API call.

Table 13: Open-QA judge validation on Doubao-Seed-2.0-pro predictions. We sample 200 Gemini 2.5 Pro judge decisions (100 accept, 100 reject) on Open-QA predictions produced by Doubao-Seed-2.0-pro uniformly at random (seed = 42) and independently re-evaluate them with a human annotator. Precision: fraction of judge-accepted predictions that are truly correct. Recall: fraction of truly correct predictions that the judge accepts. Cohen’s κ follows the standard inter-rater agreement coefficient [46].
<table><tr><td>Metric</td><td>Value</td></tr><tr><td>Audited samples</td><td>200</td></tr><tr><td>Human-judge agreement</td><td>94.0%</td></tr><tr><td>Precision (judge accept)</td><td>98.0%</td></tr><tr><td>Recall (judge accept)</td><td>90.7%</td></tr><tr><td>F1 Cohen&#x27;s κ</td><td>94.2%</td></tr><tr><td></td><td>0.880</td></tr><tr><td>False accept (judge accept, human incorrect) False reject (judge reject, human correct)</td><td>2 10</td></tr></table>

## I Additional Analysis

## I.1 Data Integrity

Decontamination procedure. Source datasets are filtered before training: (i) Cars-196 test images and (ii) iNaturalist 2021 validation images are removed wholesale from the training pool, since these images appear (or may appear) in standard public evaluation splits. Category overlap, by contrast, is intrinsic to the benchmark design: VisualWorldBench-50K and VisualWorldBench are intended to cover the same fine-grained recognition space, so they are not category-disjoint. We therefore treat image-level overlap as the relevant leakage criterion, while the holdout experiment (Appendix J) probes transfer to categories excluded from SFT target labels.

## J Holdout Experiment Details

## J.1 Experiment Design

We investigate whether fine-grained SFT improves performance on categories that are not directly supervised as SFT targets. For this target-label holdout experiment, we randomly hold out 40% of benchmark categories (seed = 42) and remove all SFT training samples whose ground-truth label belongs to the held-out set. We then compare three conditions on the subset of VisualWorldBench instances whose ground-truth category falls within the held-out set:

1. Instruct baseline: Qwen3-VL-2B without any fine-grained SFT.

2. Holdout model: Qwen3-VL-2B fine-tuned on the filtered VisualWorldBench-50K (records with held-out ground-truth labels removed).

3. Full model: Qwen3-VL-2B fine-tuned on the complete VisualWorldBench-50K (all categories included).

Gains over the instruct baseline measure improvement on categories absent from the SFT target labels. This split controls target-label supervision during SFT; it does not establish absence from pretraining or from other roles such as distractors in the SFT data. Full-versus-holdout comparisons also differ in training-data volume, so they do not isolate the effect of category-specific supervision.

## J.2 Training Data Filtering

Starting from VisualWorldBench-50K (∼7.69M records after the source-level filtering described in Appendix I.1), we remove all records whose ground-truth label matches the held-out category set. Table 14 reports the per-format filtering statistics.

Table 14: Training data filtering for the holdout experiment. Records are removed when their groundtruth label belongs to the held-out category set.
<table><tr><td>Format</td><td>Before</td><td>After</td><td>Removed</td><td>Removal %</td></tr><tr><td>MCQ</td><td>6,872K</td><td>5,197K</td><td>1,675K</td><td>24.4%</td></tr><tr><td>Bounding-box</td><td>435K</td><td>305K</td><td>130K</td><td>30.0%</td></tr><tr><td>Open-ended QA</td><td>385K</td><td>262K</td><td>123K</td><td>31.9%</td></tr><tr><td>Total</td><td>7,692K</td><td>5,764K</td><td>1,928K</td><td>25.1%</td></tr></table>

A post-hoc audit confirms that no held-out category appears among the ground-truth labels of the filtered SFT data. The overall removal rate (25.1%) is lower than the holdout ratio (40%) because a large proportion of training categories are not present in the benchmark and are therefore fully retained.

## J.3 Evaluation Subset

We evaluate on released-benchmark instances whose ground-truth labels fall into the held-out category set. After intersecting the held-out category set with the released benchmark splits, the evaluation subsets contain 5,729 MCQ samples, 1,653 Grid-Sel 3×3 samples, and 1,377 Grid-Loc 3×3 samples. All three tasks use the same curated benchmark splits and scoring metrics as the main results table (IoU@0.5 for Grid-Loc).

## J.4 Results

Table 15 reports the three-condition comparison on the held-out evaluation subset.

Table 15: Holdout experiment results on held-out categories. All numbers are percentages evaluated on released-benchmark instances whose ground-truth labels belong to the held-out category set. ∆ : holdout model gain over instruct baseline.
<table><tr><td>Condition</td><td>MCQ</td><td>Sel3×3</td><td>Loc3×3</td></tr><tr><td>Instruct baseline</td><td>28.0</td><td>10.9</td><td>20.2</td></tr><tr><td>Holdout model</td><td>35.1</td><td>30.3</td><td>41.2</td></tr><tr><td>Full model</td><td>70.6</td><td>31.3</td><td>51.6</td></tr><tr><td>∆H-B</td><td>+7.1</td><td>+19.4</td><td>+21.0</td></tr></table>

Transfer to categories held out from SFT target labels. The holdout model achieves substantial gains over the instruct baseline on Grid-Sel 3×3 (+19.4 pp) and Grid-Loc 3×3 (+21.0 pp), although these categories do not appear as ground-truth labels in its SFT data. On Grid-Sel 3×3, the holdout model reaches 30.3%, close to the full model’s 31.3%, showing a similar selection gain when these categories are excluded from SFT target labels. On Grid-Loc, the full model retains a clear advantage (51.6% vs. 41.2%), while the holdout model still improves substantially over the instruct baseline.

Closed-set recognition shows smaller holdout gains. On MCQ, the holdout model improves only modestly over the baseline (+7.1 pp), while the full model gains +42.6 pp. The larger full-versusholdout gap on MCQ suggests greater sensitivity to the target-label holdout setting, although the comparison also changes the amount and composition of SFT data.

Sanity checks. The instruct baseline and the full model each show small gaps between retainedcategory and held-out-category evaluation subsets across all three tasks (≤1.2 pp). These checks suggest similar aggregate difficulty for these two models, but do not rule out other differences between the subsets or training conditions.

Summary. Under this target-label holdout setting, Grid-Sel and Grid-Loc show larger gains over the instruct baseline than MCQ, supporting transfer beyond categories directly supervised as SFT targets. These task-dependent gains reinforce the value of multi-task evaluation without establishing a strict separation between memorization and generalization.