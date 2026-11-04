# TEXMET: A Deep Learning Benchmark for Archaeological Textile Reconstruction

MUHAMMAD HAMZA ZAFAR, Norwegian University of Science and Technology (NTNU), Norway DAVIT GIGILASHVILI, Norwegian University of Science and Technology (NTNU), Norway JON YNGVE HARDEBERG, Norwegian University of Science and Technology (NTNU), Norway

The reconstruction offragmented historical textiles presents a significant challenge in computational archaeology. The Oseberg ship burial, dating to 834 CE, yielded hundreds of degraded textile fragments that have resisted systematic computational analysis. While deep learning has revolutionized image-based puzzle solving, progress on archaeological textiles has been limited by the lack of domain-specific datasets and systematic evaluation of modern architectures. To address this gap, we present TEXMET, a curated dataset of 18,644 historical textile images from the Metropolitan Museum of Art’s Open Access collection, representing the first textile-specific benchmark for puzzle-solving research. Using TEXMET, we conduct a systematic evaluation of three state-of-the-art puzzle-solving models: JPDVT, FCViT, and DifAssemble. Our experiments reveal that the transformer-based models pre-trained on standard datasets fail to generalize to textiles, but domain-specific fine-tuning proves highly efective. The graph-based DifAssemble model, when fine-tuned on TEXMET, achieves 85.89% puzzle accuracy on textile data. Most significantly, applying this fine-tuned model to 45 highly degraded Oseberg fragments yields a performance increase from 26.67% to 64.44% puzzle accuracy. These results establish TEXMET as a new benchmark for textile reconstruction and demonstrate that domain adaptation is essential for applying puzzle-solving methods to imagery of real archaeological materials

CCS Concepts: • Computing methodologies → Computer vision; Neural networks; • Human-centered computing → HCI theory, concepts and models.

Additional Key Words and Phrases: archaeological reconstruction, deep learning, textile fragments, puzzle solving, domain adaptation, cultural heritage, vision transformers, difusion models, graph neural networks

## 1 Introduction

The reconstruction offragmented cultural heritage artifacts represents one ofthe most challenging tasks in modern archaeology. Among the most fascinating discoveries in Viking Age archaeology are the textile fragments found within the Oseberg ship burial, discovered in 1903 near Tønsberg, Norway. This 9th-century Viking vessel, serving as the final resting place for two women of apparent high status, contained an extraordinary collection of artifacts that have provided unprecedented insights into Viking craftsmanship, daily life, and cultural practices [39]. The textile fragments recovered from the Oseberg burial are particularly significant, representing some of the finest examples of Viking Age weaving techniques and artistic expression [1, 39]. However, these precious artifacts present a major challenge: they exist as hundreds of fragmented pieces, scattered and deteriorated by centuries of burial conditions. The fragments are now housed at the Museum of Cultural History in Oslo, where ongoing eforts seek to understand their original configurations (Figure 1).

Recent breakthroughs in deep learning, particularly in computer vision, have created unprecedented opportunities for addressing complex pattern recognition and reconstruction challenges. Transformer architectures [8, 38] have revolutionized image analysis tasks, with their attention mechanisms proving particularly efective for understanding spatial relationships and pattern recognition. State-of-the-art models such as JPDVT (Jigsaw Puzzles with Difusion Vision Transformers) [21], FCViT (Fragment’s Coordinate Vision Transformer) [19], and graph difusion-based approaches like DifAssemble [31] have achieved impressive performance on standardized puzzle-solving benchmarks. These advances suggest that the fundamental technologies needed for automated fragment reassembly are becoming mature. However, these models have been primarily tested on clean, synthetic puzzles rather than real archaeological materials.

![](images/242c77a672f4a3f668e3085ec44cbd9708a69a2f12e5c89784975589ef81da24.jpg)  
Fig. 1. Examples of textile fragments from the Oseberg ship burial (834 CE). These degraded fragments present significant challenges for computational reconstruction. The fragments do not correspond to their true scale. Images by George Alexis Pantos, Museum of Cultural History, University of Oslo.

A significant gap exists between the controlled conditions of standard puzzle-solving benchmarks and the real-world challenges of archaeological textile reconstruction. Standard datasets typically feature clean, regularly shaped pieces arranged in regular grids (e.g., 3×3, 9×9), with perfect edges, uniform lighting conditions, and complete coverage of the original image [28]. Archaeological textile fragments, by contrast, are irregularly shaped, sufer from various forms of degradation, may have missing portions, and often exist in incomplete sets where not all pieces of the original textile have been preserved [6]. Furthermore, no large-scale textile-specific dataset exists for training and evaluating puzzle-solving models, and no systematic study has evaluated modern deep learning approaches on archaeological textile data.

To address these gaps, we present TEXMET, a new dataset of 18,644 curated historical textile images from the Metropolitan Museum of Art’s Open Access collection [36]. To the best of our knowledge, TEXMET is the first textile-specific benchmark dataset for puzzle-solving research. Using this dataset, we conduct the first systematic evaluation of three state-of-the-art puzzle-solving models on textile data. Our experiments reveal that the transformer-based models pre-trained on standard datasets fail to generalize to textiles, but that domainspecific fine-tuning dramatically improves performance. The graph-based DifAssemble model, when fine-tuned on TEXMET, achieves 85.89% puzzle accuracy on textile data and, most significantly, improves from 26.67% to 64.44% accuracy on the highly degraded Oseberg fragments.

In summary, this work contributes TEXMET, the first textile-specific benchmark dataset for puzzle-solving research; provides the first systematic evaluation of three state-of-the-art puzzle-solving architectures on textile data and real archaeological fragments; demonstrates that domain-specific fine-tuning is essential for adapting puzzle-solving models to archaeological textile imagery; and releases all code, model checkpoints, and data to support reproducibility. We position this work as a benchmarking and domain adaptation study, a controlled first step toward computational reconstruction of archaeological textiles, rather than a complete reconstruction solution.

## 2 Related Work

## 2.1 Computational Puzzle Solving

The computational jigsaw-puzzle problem has evolved significantly over the past six decades. Foundational work by Freeman and Garder [9] introduced brute-force border matching for small puzzles using shape-only features. Subsequent decades focused on heuristic optimizations, using global shape-based reoptimization [14], Markov random fields [4], genetic algorithms [32], and greedy color compatibility [29]. These classical methods excelled with undamaged, uniformly shaped puzzle pieces but struggled significantly with irregular or damaged fragments.

The rise of deep learning in 2015 sparked renewed interest in puzzle-solving as a self-supervised learning task. Doersch et al. [7] and Noroozi and Favaro [26] demonstrated convolutional neural network (CNN) architectures capability to learn general visual representations through puzzle-solving pretext tasks. The introduction of transformer architectures marked a paradigm shift, with vision transformers enabling spatial reasoning directly within self-attention mechanisms [3].

Recent breakthroughs have transformed puzzle-solving through difusion models and graph-based approaches. JPDVT [21] approached puzzle-solving as a denoising problem, efectively handling missing or uncertain fragments. FCViT [19] introduced coordinate regression for direct position prediction, claiming superior performance with a simpler architecture. DifAssemble [31] combines graph neural networks with difusion processes, enabling robust handling of rotation and multiple puzzle sizes within a single model. These three models represent the current state-of-the-art and form the basis of our evaluation.

## 2.2 Heritage-Focused Computational Methods

Several research eforts have applied computational puzzle-solving to cultural heritage applications. Paumard et al. [27] combined Siamese CNN-based edge matching with shortest-path algorithms, demonstrating success on a dedicated reassembly dataset of artwork images from the Metropolitan Museum of Art. Derech et al. [6] proposed fragment-extension methods specifically addressing erosion in archaeological materials, reassembling eroded artifacts and frescoes including objects from museum collections. Funkhouser et al. [10] addressed the matching of fresco fragments from the Bronze Age settlement of Akrotiri (Thera) and the Roman site of Tongeren.

Despite these advances, archaeological textiles have remained notably underserved by computational puzzle solving research. Textile fragments present unique challenges including complex weaving patterns, severe degradation from burial conditions, irregular shapes from damage, and color fading that hides visual features. The absence of systematic evaluation of modern deep learning methods on textile data represents a significant gap that this work addresses.

Several previous works explored the open archaeological puzzle problem of Oseberg textiles. Gigilashvili et al. [11, 13] presented the first attempt to apply machine learning to the Oseberg fragment matching problem using clustering based on deep features and classical texture descriptors. Gulbrandsen [15] developed the Artifact Assembly software tool for interactive puzzle reassembly of textile fragments. More recently, Berels [2] investigated deep feature extraction under irregular masking conditions for content-based image retrieval in archaeological puzzle solving. A comprehensive review of computational techniques for virtual reconstruction of fragmented archaeological textiles can be found in [12]. Our work builds on these eforts by introducing the first large-scale textile benchmark dataset and systematically evaluating state-of-the-art deep learning puzzle-solving models on this domain.

## 2.3 Existing Datasets

Standard puzzle-solving benchmarks have relied on natural image datasets such as ImageNet [5] and CelebA-HQ [18]. These datasets feature clean, high-resolution images that are synthetically fragmented into regular grid patterns. JPwLEG-3 [34] introduced puzzles with gaps, simulating missing pieces, but still operates on clean natural images.

For cultural heritage applications, the RePAIR dataset [37] provides fragmented fresco images for benchmarking, but does not capture the unique visual properties ofwoven textile materials. TextileNet [41] ofers a large collection of modern fabric images for texture classification, but these are intact images of contemporary materials unsuitable for training puzzle-solving models on historical textiles.

A critical limitation in the field has been the absence of a large-scale, publicly available dataset specifically designed for historical textile reconstruction. This gap has significantly hindered the ability to train and evaluate deep learning models on textile-specific challenges. The creation of TEXMET directly addresses this limitation, providing the first textile-specific benchmark for puzzle-solving research.

## 3 The TEXMET Dataset

A critical limitation in computational textile reconstruction has been the absence of a large-scale, publicly available dataset specifically designed for historical textiles. Existing datasets are unsuitable for this domain: TextileNet [41] contains modern fabric swatches without fragmentation, RePAIR [37] provides fresco fragments with visual characteristics diferent from woven textiles, and standard puzzle benchmarks use natural images with no textile content. To address this gap, we introduce TEXMET, the first textile-specific benchmark dataset for puzzle-solving research.

## 3.1 Data Collection

TEXMET was constructed from the Metropolitan Museum of Art’s Open Access collection [36], which provides high-resolution historical textile images under a CC0 public domain license. Using the MET REST API, we systematically collected textile and tapestry objects through targeted queries. The collection process involved multi-stage API harvesting to handle rate limiting and ensure completeness.

Table 1 summarizes the data collection results. From 33,588 queried objects, 27,373 contained images, yielding 55,007 total images downloaded with a 99.95% success rate. This systematic collection required approximately three days of continuous harvesting.

## 3.2 Curation Pipeline

The raw collection required extensive curation to ensure quality and relevance for puzzle-solving tasks. We employed an interactive curation pipeline using FiftyOne [25] with the following components:

Table 1. TEXMET data collection summary from the MET archive.
<table><tr><td>Metric</td><td>Count</td></tr><tr><td>Objects queried</td><td>33,588</td></tr><tr><td>Objects with images</td><td>27,373</td></tr><tr><td>Primary images downloaded</td><td>27,360</td></tr><tr><td>Additional images downloaded</td><td>27,647</td></tr><tr><td>Total images</td><td>55,007</td></tr><tr><td>Download success rate</td><td>99.95%</td></tr></table>

![](images/e8eac8c026925244b976746eb7c0aa1d3a1d01229e708b96292c6e51868ffa98.jpg)  
Fig. 2. UMAP projection of CLIP embeddings for the textile collection, colored by MET curatorial department, as displayed in the FiftyOne curation interface. Natural clusters emerge, enabling systematic identification of outliers and quality issues.

CLIP Embeddings: GPU-accelerated CLIP ViT-B/32 embeddings [30] were computed for all 27,360 primary images, producing 512-dimensional feature vectors for each image.

UMAP Clustering: Embeddings were projected to three dimensions using UMAP [24], revealing natural clusters and outliers for systematic quality assessment. Figure 2 shows the UMAP projection of CLIP embeddings, revealing natural clusters that guided systematic quality assessment.

![](images/330af32192983f2b99b1091dc2f6a2498c32fbacf6d719bc7feee46c53cd136b.jpg)  
Fig. 3. Overview of the final curated TEXMET dataset showing 18,644 high-quality textile images spanning diverse historical periods and cultural traditions.

Table 2. TEXMET dataset statistics.
<table><tr><td colspan="2">Property Value</td></tr><tr><td>Total images</td><td>18,644</td></tr><tr><td>Train / Val / Test split</td><td>14,915 / 1,864 / 1,865</td></tr><tr><td>Average dimensions</td><td>1,557 × 1,631 pixels</td></tr><tr><td>Average resolution</td><td>3.3 MP</td></tr><tr><td>Color images (RGB)</td><td>92.0%</td></tr><tr><td>Black and white images</td><td>8.0%</td></tr><tr><td>Total dataset size</td><td>26.5 GB</td></tr></table>

Similarity Search: LanceDB [20] indexing enabled sub-second nearest-neighbor queries, allowing eficient identification of problematic image clusters.

Quality Assessment: Samples were classified as “poor quality” (irrelevant content such as armor or weapons misclassified as textiles) or “noise” (insuficient textile coverage, individual threads, or minimal textile elements).

The curation process removed 8,716 samples in two categories, 4,908 tagged as poor quality and 3,808 as noise, of which 2,567 were duplicate images, yielding a final dataset of 18,644 high-quality textile images representing 68.1% of the original collection. Figure 3 provides an overview of the final curated dataset.

## 3.3 Dataset Statistics

Table 2 presents the technical characteristics of the final TEXMET dataset. The collection demonstrates consistent quality with an average resolution of 3.3 megapixels and dimensions suitable for puzzle-solving experiments.

The dataset maintains strong representation across 14 MET curatorial departments, preserving global diversity in textile traditions. Table 3 shows the department distribution, with European Art contributing the largest proportion followed by Asian Art and Islamic Art.

Table 3. Department distribution in TEXMET.
<table><tr><td>Department</td><td>Samples</td><td>%</td></tr><tr><td>European Sculpture and Decorative Arts</td><td>8,516</td><td>45.7</td></tr><tr><td>Asian Art</td><td>5,229</td><td>28.0</td></tr><tr><td>Islamic Art</td><td>3,029</td><td>16.2</td></tr><tr><td>Other departments (11)</td><td>1,870</td><td>10.0</td></tr><tr><td>Total</td><td>18,644</td><td>100.0</td></tr></table>

Table 4. Comparison of TEXMET with related datasets.
<table><tr><td>Dataset</td><td>Domain</td><td>Size</td><td>Fragmented</td><td>License</td></tr><tr><td>ImageNet</td><td>Natural images</td><td>1.28M</td><td>No</td><td>Research</td></tr><tr><td>CelebA-HQ</td><td>Faces</td><td>30k</td><td>No</td><td>Research</td></tr><tr><td>TextileNet</td><td>Modern fabrics</td><td>760k</td><td>No</td><td>Academic</td></tr><tr><td>RePAIR</td><td>Frescoes</td><td>1k+</td><td>Yes</td><td>Restricted</td></tr><tr><td>TEXMET</td><td>Historical textiles</td><td> $^ { 1 8 , 6 4 4 }$ </td><td>Simulated</td><td>CC0</td></tr></table>

The supplementary material presents the technical characteristics of TEXMET images in detail, showing consistent quality suitable for puzzle-solving experiments.

## 3.4 Comparison with Existing Datasets

Table 4 compares TEXMET with existing datasets used in puzzle-solving and textile research. TEXMET uniquely combines historical textile content, open licensing, and suitability for puzzle-solving evaluation.

To the best of our knowledge, TEXMET represents the first comprehensively curated textile dataset specifically designed for computational puzzle-solving research. The dataset, along with all code and trained models, will be made publicly available.

## 4 Methods

This section describes the three state-of-the-art puzzle-solving models evaluated in this research: JPDVT, FCViT, and DifAssemble. Each represents a diferent architectural approach to the puzzle-solving problem.

## 4.1 Problem Formulation

Given an image divided into � non-overlapping fragments $\{ x _ { i } \} _ { i = 1 } ^ { N }$ , the puzzle-solving task aims to determine the correct spatial arrangement that reconstructs the original image. Formally, for each fragment $x _ { i } ,$ the model must predict its correct grid position $l _ { i } .$ . The reconstructed image is then:

$$
X = \bigcup _ { i = 1 } ^ { N } ( x _ { i } @ l _ { i } ) ,\tag{1}
$$

where $x _ { i } @ l _ { i }$ denotes fragment $x _ { i }$ placed at location $l _ { i }$

![](images/af9a93cd3d5767b04d3b3301797aae473f68d6e20103c22ba8a461666960c3a8.jpg)  
Fig. 4. JPDVT architecture showing the Vision Transformer backbone integrated with the difusion process for position prediction. Figure by the authors, created using Canva.

## 4.2 JPDVT: Difusion Vision Transformer

JPDVT [21] approaches puzzle-solving as a conditional difusion problem [17]. The model uses a Vision Transformer (ViT) encoder to extract visual features from each fragment, then employs a difusion process to predict positional encodings. Figure 4 illustrates the architecture.

During training, Gaussian noise is progressively added to the positional encodings (forward difusion). During inference, the model learns to reverse this process, denoising the positional information conditioned on visual features. This approach naturally handles missing pieces through mask token embeddings, making it robust to incomplete puzzles.

The authors reported 68.7% puzzle accuracy and 83.3% piece accuracy on ImageNet 3×3 puzzles. A key advantage of JPDVT is its ability to model long-range dependencies through the transformer architecture and handle missing fragments through the difusion framework.

## 4.3 FCViT: Coordinate Regression

FCViT [19] reformulates puzzle-solving as a coordinate regression task rather than classification. Instead of predicting permutations, the model directly predicts horizontal and vertical coordinates for each fragment using a Vision Transformer encoder followed by a multilayer perceptron (MLP) prediction head. Figure 5 illustrates the architecture.

The training objective minimizes a smooth $L _ { 1 }$ loss between predicted and ground-truth coordinates:

$$
\mathcal { L } _ { c o o r d } = \mathcal { L } _ { c o o r d } ^ { h } + \mathcal { L } _ { c o o r d } ^ { v } ,\tag{2}
$$

where the losses for horizontal and vertical coordinates are computed separately.

FCViT achieved 78.9% puzzle accuracy and 90.6% piece accuracy on ImageNet 3×3 puzzles, outperforming JPDVT. However, a notable limitation is a tendency toward duplicate placements, where multiple fragments may be assigned to similar coordinates due to the regression formulation lacking explicit uniqueness constraints.

![](images/0bc71fc390955a414bdafc60571fa4f4dfe9454adeb0dc592c9c63331b833a7d.jpg)  
Fig. 5. FCViT architecture illustrating the coordinate regression process where fragment positions are directly predicted. Figure by the authors, created using Canva.

Table 5. Comparison of puzzle-solving model capabilities.
<table><tr><td>Model</td><td>Rotation</td><td>Missing Pieces</td><td>Multi-size</td><td>Approach</td></tr><tr><td>JPDVT</td><td>No</td><td>Yes</td><td>No</td><td>Diffusion</td></tr><tr><td>FCViT</td><td>No</td><td>No</td><td>No</td><td>Regression</td></tr><tr><td>DiffAssemble</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Graph-Diffusion</td></tr></table>

## 4.4 DifAssemble: Graph-Based Difusion

DifAssemble [31] represents puzzles as graphs, where each fragment is a node and edges encode spatial relationships. A graph neural network (GNN) combined with difusion denoising predicts both translations and rotations for each piece. Figure 6 illustrates the architecture.

The forward difusion adds noise to fragment positions and orientations:

$$
q ( x _ { t } | x _ { 0 } ) = N ( x _ { t } ; \sqrt { \alpha _ { t } } x _ { 0 } , ( 1 - \alpha _ { t } ) \mathbf { I } ) ,\tag{3}
$$

where $x _ { 0 }$ encodes the initial pose and $\alpha _ { t }$ controls difusion at timestep �. The reverse process iteratively refines predictions through attention-based GNN layers.

A key advantage of DifAssemble is its unified architecture that handles multiple puzzle sizes (6×6 to 12×12) with a single 26.7M parameter model. The model achieved 99.51% piece accuracy on 6×6 puzzles and 97.76% on 12×12 puzzles. It also demonstrates robustness to missing pieces, maintaining 76.5% accuracy on 12×12 puzzles with 30% missing fragments.

## 4.5 Model Capabilities Comparison

Table 5 summarizes the key capabilities of each model relevant to archaeological textile reconstruction.

DifAssemble ofers the most comprehensive feature set for archaeological applications, supporting rotation, missing pieces, and variable puzzle sizes within a single model.

![](images/79dc4c36e5f783819a64a013286801df4b8770d88d01f48657963f2bad214de6.jpg)  
Fig. 6. DifAssemble architecture demonstrating the graph-based difusion approach for 2D puzzle reassembly. Figure by the authors, created using Canva.

## 4.6 Evaluation Metrics

We employ two complementary metrics:

Puzzle-Level Accuracy measures the percentage of puzzles completely solved correctly. All pieces must be in their exact correct positions for a puzzle to count as solved:

$$
\mathrm { P u z z l e ~ A c c u r a c y } = { \frac { \mathrm { P e r f e c t l y ~ s o l v e d ~ p u z z l e s } } { \mathrm { T o t a l ~ p u z z l e s } } } \times 1 0 0 \%\tag{4}
$$

Piece-Level Accuracy (Fragment Accuracy) measures the percentage of individual pieces correctly positioned across all puzzles, providing insight into partial reconstruction success:

$$
\mathrm { P i e c e ~ A c c u r a c y } = { \frac { \mathrm { C o r r e c t l y ~ p l a c e d ~ p i e c e s } } { \mathrm { T o t a l ~ p i e c e s } } } \times 1 0 0 \%\tag{5}
$$

These metrics together capture both exact reassembly of all fragments (puzzle accuracy) and partial success in challenging cases (piece accuracy).

## 5 Experimental Setup

## 5.1 Datasets

We evaluate the three puzzle-solving models on three datasets representing diferent domains:

ImageNet-1k [5] serves as the baseline training dataset, containing 1,281,167 natural images. All three models were initially trained or reproduced on ImageNet following their original protocols.

TEXMET (Section 3) provides domain-specific training data with 18,644 historical textile images split into 14,915 training, 1,864 validation, and 1,865 test images.

Oseberg Fragments consist of 45 textile fragments from the Oseberg ship burial (834 CE), representing highly degraded archaeological materials. These fragments exhibit an average masked area of 75.60%, measured from segmentation masks of the fragment photographs, indicating severe degradation with minimal visual context compared to standard puzzle datasets.

## 5.2 Hardware and Implementation

All experiments were conducted on a high-performance computing cluster equipped with NVIDIA A100 and H100 GPUs. Training utilized distributed computing across multiple GPUs to accommodate the computational demands of transformer and difusion-based architectures.

To ensure reproducibility, all experiments used a fixed random seed of 42 with PyTorch deterministic mode enabled. Training progress was monitored through systematic logging of accuracy and loss metrics.

## 5.3 Training Protocol

Model Reproduction: Each model was reproduced following its original training protocol. JPDVT used AdamW optimization [22] with the learning rate schedule specified in the original paper. FCViT followed the training configuration from its authors’ implementation. DifAssemble was trained on CelebA-HQ for 158 epochs with batch size 8, using the DDIM sampling strategy [33] and ResNet18-equivalent backbone [16] with transformer layers, following the original paper’s protocol. The ImageNet and TEXMET experiments used the implementation’s default configuration: an EficientNet-B0 backbone [35] with transformer layers, batch size 128 across four GPUs, and rotation disabled.

Fine-tuning: For domain adaptation experiments, models pre-trained on ImageNet were fine-tuned on TEXMET. DifAssemble fine-tuning continued for 150 additional epochs on TEXMET training data while monitoring performance on both TEXMET validation and Oseberg evaluation sets.

## 5.4 Evaluation Protocol

Puzzle Generation: Each evaluation puzzle is generated from a single source image. The image is center-cropped and resized to the model’s input resolution (JPDVT: 192×192; the JPDVT-T 4 × 4 variant: 256×256; per-model sizes follow the original implementations), tiled into $G \times G$ equal square tiles, and the tile order is permuted with a seeded random permutation. One permutation is drawn per image per evaluation. Tile orientation is kept fixed (no rotations) in all TEXMET and Oseberg evaluations: JPDVT and FCViT do not support rotation, and rotation was disabled for DifAssemble for comparability. All $G ^ { 2 }$ tiles are present and known to belong to the same puzzle; the benchmark contains no missing pieces and no distractor tiles.

Splits and Leakage: TEXMET is split at the image level by a seeded random shufle (seed 42) into 80/10/10 train, validation, and test partitions (14,915 / 1,864 / 1,865 images). Each image is the primary image of a distinct museum object, so the split is simultaneously object-level: no object appears in more than one partition, and because every puzzle is constructed from exactly one image, no puzzle tiles can span partitions by construction. We note one honest caveat: distinct museum objects can originate from related historical textiles (for example several catalogued fragments of one source fabric), and no metadata-level grouping was applied for such cases.

Ground Truth and Metrics: The ground truth for each puzzle is the original tile arrangement of the intact image; a tile is placed correctly when its predicted position index equals its true index. Puzzle accuracy is the fraction of test images whose entire permutation is recovered exactly, and fragment accuracy is the mean fraction of correctly placed tiles per image (Section 4). All TEXMET results are computed over the full 1,865-image test partition unless a subset is stated explicitly (the masking and inpainting experiments use 1,000 ImageNet validation images per condition). Evaluations use a single run per configuration with seeded shufling; where repeated runs revealed seed sensitivity, this is reported alongside the afected results (Section 6.7).

Table 6. Model reproduction results on ImageNet 3×3 puzzles. Puzzle and fragment accuracy (%) compared to original publications.
<table><tr><td rowspan="2">Model</td><td colspan="2">Original Paper</td><td colspan="2">Our Reproduction</td></tr><tr><td>Puzzle</td><td>Fragment</td><td>Puzzle</td><td>Fragment</td></tr><tr><td>JPDVT</td><td>68.7</td><td>83.3</td><td>67.89</td><td>80.02</td></tr><tr><td>FCViT</td><td>78.9</td><td>90.6</td><td>68.78</td><td>84.21</td></tr></table>

Puzzle Configurations: JPDVT and FCViT were evaluated on 3×3 puzzles (9 pieces) following their original benchmarks. DifAssemble was evaluated on multiple sizes from 6×6 to 12×12, with 3×3 configurations used for cross-model comparisons. Oseberg fragments were evaluated using 3×3 puzzles across all models.

Cross-Dataset Evaluation: To assess domain generalization, models trained on ImageNet were directly evaluated on TEXMET and Oseberg without additional training, measuring the domain gap between natural images and textile data.

Domain Adaptation: Models were fine-tuned on TEXMET and re-evaluated on both TEXMET test data and Oseberg fragments to measure the efectiveness of domain-specific training.

Oseberg Evaluation: The 45 Oseberg fragment images are evaluated with the same single-image pipeline: each fragment image is individually divided into an artificial 3×3 grid and reassembled in isolation. No inter-fragment ground truth exists, so this evaluation measures within-fragment reordering, not cross-fragment matching (see the discussion of this simplification in Section 7).

Inscribed Cropping: To investigate whether models learn textile patterns or merely shape information, we evaluated using inscribed square crops that remove corner shape cues while preserving central textile content (for the Oseberg fragments, 206 manually annotated square crops from the 45 fragment images).

## 6 Results

This section presents experimental results organized by research question: model reproduction, ImageNet baseline comparison, cross-domain evaluation, Oseberg baseline performance, domain adaptation efectiveness, analysis of learned representations, degradation and inpainting efects, and an ablation analysis.

## 6.1 Model Reproduction

We first verify our implementations by reproducing the benchmark results reported in each model’s original publication. Table 6 compares our results with those reported in the original papers for JPDVT and FCViT, whose publications report ImageNet 3×3 results

JPDVT ’s puzzle accuracy was reproduced within 1 percentage point of the original result (fragment accuracy within 3.3 points). FCViT showed a 10 percentage point gap from reported performance despite following the original training protocol, representing a limitation of our implementation. The original DifAssemble publication reports no ImageNet results, so we verify our DifAssemble implementation on the benchmark its authors do report: CelebA-HQ across multiple puzzle sizes. DifAssemble closely matched the original paper’s results across all puzzle sizes, demonstrating robust reproducibility.

![](images/7c6c65468d4850a28931fb54ed7dc97bda6158db2866309e2d60f98f51969370.jpg)  
Fig. 7. DifAssemble reconstruction examples on CelebA-HQ [18] across puzzle sizes (6×6 to 12×12). Left: successful reconstructions showing correct piece placement. Right: failure cases revealing common error paterns.

Table 7. DifAssemble performance across puzzle sizes on CelebA-HQ.
<table><tr><td>Puzzle Size</td><td>Piece Accuracy (%) Puzzle Accuracy (%)</td><td></td></tr><tr><td>6×6 (36 pieces)</td><td>99.32</td><td>91.93</td></tr><tr><td>8×8 (64 pieces)</td><td>98.80</td><td>83.39</td></tr><tr><td>10×10 (100 pieces)</td><td>98.57</td><td>75.53</td></tr><tr><td>12×12 (144 pieces)</td><td>98.05</td><td>68.07</td></tr></table>

Table 7 presents DifAssemble performance across multiple puzzle configurations, showing graceful degradation as puzzle complexity increases. Figure 7 presents qualitative examples of DifAssemble reconstructions across multiple puzzle sizes, illustrating both successful assemblies and failure cases.

## 6.2 ImageNet Baseline

With the implementations verified, we establish a common starting point for the cross-domain experiments: all three models evaluated on ImageNet 3×3 puzzles under the unified evaluation protocol of Section 5.4. JPDVT reaches 67.89% puzzle and 80.02% fragment accuracy, FCViT 68.78% and 84.21%, and DifAssemble 56.61% and 84.03%. DifAssemble trails the two transformer models at the puzzle level while remaining competitive at the fragment level; this ranking reverses once the domain shifts to textiles. Figure 8 visualizes this comparison.

## 6.3 Cross-Domain Evaluation

To assess domain generalization, we evaluated ImageNet-trained models directly on TEXMET without any domain-specific training. This experiment reveals the domain gap between natural images and historical textiles.

The results in Table 8 reveal substantial performance variation across architectures. FCViT experienced nearcomplete collapse to 8.30%, suggesting its coordinate regression approach is highly sensitive to domain-specific visual features. JPDVT maintained moderate performance at 32.92%, indicating partial generalization capability. DifAssemble demonstrated the strongest cross-domain robustness, maintaining 65.08% puzzle accuracy, a level that even slightly exceeds its 56.61% ImageNet 3×3 baseline under the same evaluation protocol (Section 6.2).

ImageNet Baseline Performance (3x3 Puzzles)  
![](images/2fe37c30935ddb15aea536dbe880ee8422233a8574923b4d64ff0a6abd038f3b.jpg)  
Fig. 8. Baseline performance comparison of JPDVT, FCViT, and DifAssemble on ImageNet 3×3 puzzles showing puzzle and fragment accuracy.

Table 8. Cross-domain evaluation: ImageNet-trained models tested on TEXMET 3×3 puzzles without fine-tuning.
<table><tr><td>Model</td><td>Puzzle Accuracy (%) Fragment Accuracy (%)</td><td></td></tr><tr><td>JPDVT</td><td>32.92</td><td>55.89</td></tr><tr><td>FCViT</td><td>8.30</td><td>28.25</td></tr><tr><td>DiffAssemble</td><td>65.08</td><td>86.63</td></tr></table>

Figure 9 visualizes this domain gap, illustrating DifAssemble’s superior robustness to distribution shift compared to transformer-based approaches.

## 6.4 Oseberg Fragment Baseline

We evaluated all models on the 45 Oseberg textile fragments to establish baseline performance on real archaeological materials (see Figure 13 for examples of 3×3 fragmented Oseberg puzzles). These fragments present extreme challenges: an average masked area of 75.60% indicates severe degradation with minimal visual context.

Cross-Dataset Generalization: Puzzle Accuracy on Untrained Domains  
![](images/9e1d4d740bd22c6bb1b61c11953b783f6ce7c2760c00bdeba77bc82a5b00b731.jpg)  
Fig. 9. Cross-dataset evaluation comparing model performance on ImageNet, TEXMET, and Oseberg fragments. DifAssemble maintains performance while FCViT collapses, demonstrating the domain gap between natural images and historical textiles.

Table 9. Baseline performance on Oseberg fragments (3×3 puzzles, 45 fragments). All models trained on ImageNet without textile-specific fine-tuning.
<table><tr><td>Model</td><td>Puzzle Accuracy (%)</td></tr><tr><td>JPDVT</td><td>20.00</td></tr><tr><td>FCViT</td><td>24.44</td></tr><tr><td>DiffAssemble</td><td>26.67</td></tr></table>

Table 9 shows that all models struggle significantly on Oseberg fragments, with accuracies ranging from 20.00% to 26.67%. DifAssemble achieved the highest baseline at 26.67%, consistent with its superior cross-domain robustness observed on TEXMET. These low accuracies confirm that ImageNet pre-training alone is insuficient for archaeological textile reconstruction.

## 6.5 Domain Adaptation via Fine-Tuning

Based on the cross-domain results, we investigated whether domain-specific fine-tuning could improve performance. DifAssemble was selected for fine-tuning experiments given its superior baseline performance and architectural flexibility.

Table 10 presents the key finding of this work: domain-specific fine-tuning dramatically improves performance on both textile datasets. On TEXMET, accuracy improved from 65.08% to 85.89%, a gain of 20.81 percentage points.

Table 10. DifAssemble performance before and after fine-tuning on TEXMET. The model was fine-tuned for 150 epochs on TEXMET training data.
<table><tr><td>Evaluation Dataset Baseline (%) Fine-tuned (%) Improvement</td><td></td><td></td><td></td></tr><tr><td>TEXMET</td><td>65.08</td><td>85.89</td><td>+20.81</td></tr><tr><td>Oseberg</td><td>26.67</td><td>64.44</td><td>+37.77</td></tr></table>

Performance Trajectory of DiffAssemble Across Experimental Stages  
![](images/7da5e60be3d4321a1b55da01cfece39d879ddb296993459bd5de93de4c8e58e3.jpg)  
Fig. 10. Summary of DifAssemble puzzle accuracy across the five experimental stages. Fine-tuning on TEXMET lifts accuracy in both evaluation domains: TEXMET rises from 65.08% to 85.89% and the held-out Oseberg fragments from 26.67% to 64.44%, indicating that the learned representations transfer to degraded archaeological materials. The drop between the third and fourth stages reflects the switch to the much harder Oseberg domain, not a decline in the model.

Most significantly, on the challenging Oseberg fragments, accuracy improved from 26.67% to 64.44% (a 37.77 percentage point improvement), demonstrating that models can learn to handle highly degraded archaeological materials when provided with appropriate training data.

Figure 10 summarizes performance across the five experimental stages, showing that fine-tuning improves accuracy on both TEXMET and the held-out Oseberg evaluation set.

Fine-tuning experiments with JPDVT and FCViT were unsuccessful. JPDVT exhibited catastrophic forgetting [23], its TEXMET accuracy falling from 32.92% to 10.88%, while FCViT improved only marginally (8.30% to 10.86%), indicating limited adaptation capacity rather than successful domain transfer. This suggests that DifAssemble’s graph-based architecture may be particularly well-suited to domain transfer.

Table 11. Inscribed cropping analysis for DifAssemble: puzzle accuracy (%) with standard and inscribed crops. Inscribed crops remove shape information from puzzle piece boundaries.
<table><tr><td>Dataset</td><td>Training</td><td>Standard</td><td>Inscribed</td></tr><tr><td>TEXMET</td><td>ImageNet baseline</td><td>65.08</td><td>25.79</td></tr><tr><td>TEXMET</td><td>Fine-tuned</td><td>85.89</td><td>27.08</td></tr><tr><td>Oseberg</td><td>ImageNet baseline</td><td>26.67</td><td>27.67</td></tr><tr><td>Oseberg</td><td>Fine-tuned</td><td>64.44</td><td>39.81</td></tr></table>

## 6.6 Analysis: What Did the Model Learn?

A critical question is whether the fine-tuned model learned textile-specific visual patterns or merely exploited shape information from puzzle piece boundaries. To investigate this, we evaluated using inscribed square crops that remove corner shape cues while preserving central textile content.

Table 11 shows that for the fine-tuned model, performance decreases from 64.44% to 39.81% on Oseberg when shape information is removed. The efect is even stronger on TEXMET, where the fine-tuned model falls from 85.89% to 27.08%, indicating that on intact textile images the model leans more heavily on boundary cues than on the degraded fragments. Conversely, the ImageNet baseline on Oseberg improves slightly under inscribed cropping (26.67% to 27.67%), showing that removing ragged fragment boundaries does not simply destroy information. While these drops indicate that shape cues contribute significantly to performance, all inscribed results remain far above random chance (a random arrangement solves a complete 3×3 puzzle with probability $1 / 9 ! \approx 3 \times 1 0 ^ { - 4 } \% ,$ and the corresponding fragment-level random baseline is $1 / 9 \approx 1 1 . 1 \%$ , demonstrating that the model has learned meaningful textile patterns beyond mere shape matching.

Figure 13 presents the Oseberg fragment images most successfully reordered by the baseline model, illustrating that fragments with distinctive global structure remain solvable despite severe degradation.

## 6.7 Impact of Masking and Inpainting

To investigate how degradation patterns afect puzzle-solving performance, we conducted experiments with diferent masking strategies, on a curated subset of 1,000 images from the ImageNet validation set providing a controlled visual foundation, and evaluated whether inpainting could recover lost information.

Masking Strategies. We evaluated two masking approaches, both calibrated to obscure approximately 35.6% of the image area: (1) regular centered rectangular masking simulating uniform degradation, and (2) irregular masking derived from a single representative Oseberg fragment boundary, resized and applied to every image so that all images share the same degradation pattern, representing realistic archaeological damage patterns.

Table 12 summarizes the impact of masking and inpainting on puzzle accuracy across all three models.

One caveat on reading the table: on the 1,000-image subset, JPDVT’s no-mask baseline (87.80%) sits well above its full-split baseline of 67.89% (Section 6.2), while the other two models land within two points of theirs; masking efects should therefore be read against the table’s own baseline row, not across sections. The three architectures respond to degradation in very diferent ways. JPDVT is essentially unafected by regular masking (87.30% against its 87.80% baseline): the regular grid mask, aligned with the image coordinates, removes information without disturbing the patch structure the model relies on. Irregular, archaeologically derived masks are the harder condition for JPDVT, costing 8.7 percentage points. FCViT is the most vulnerable model, losing 24.5 points under irregular masking, consistent with its reliance on global visual features that scattered occlusion destroys. DifAssemble starts from a lower baseline but is the most robust, varying within a band of roughly 8 points across all conditions; its graph-based, relational formulation appears inherently less sensitive to missing content. We note that repeated JPDVT evaluations with diferent random seeds varied by up to 6.4 percentage points under regular masking; we report the values consistent with the underlying experiment logs

Table 12. Impact of masking and inpainting on 3×3 puzzle accuracy (%), evaluated on 1,000 ImageNet validation images per condition.
<table><tr><td>Condition</td><td>JPDVT</td><td>FCViT</td><td>DiffAssemble</td></tr><tr><td>No mask (baseline)</td><td>87.80</td><td>70.62</td><td>55.70</td></tr><tr><td>Regular mask</td><td>87.30</td><td>60.83</td><td>50.50</td></tr><tr><td>Irregular mask</td><td>79.10</td><td>46.15</td><td>54.70</td></tr><tr><td>Regular + inpainting</td><td>86.10</td><td>65.10</td><td>58.50</td></tr><tr><td>Irregular + inpainting</td><td>84.30</td><td>60.10</td><td>57.90</td></tr></table>

Inpainting Recovery. We applied ASUKA [40], a state-of-the-art inpainting model, to masked images before puzzle solving. Figure 11 illustrates the full pipeline for both mask types, from the original image to the inpainted result; quantitative fidelity metrics, which show that irregular masks are consistently harder to inpaint, are reported in the supplementary material. Figure 12 shows that inpainting is a useful but not universal recovery strategy, and its benefit depends on both the model and the mask type. It helps most where the damage hurts most: FCViT recovers 14.0 points on irregularly masked images and JPDVT recovers 5.2, while DifAssemble gains a more modest 3.2. On regularly masked images the picture is mixed: JPDVT’s small change (86.10% against 87.30%) lies within the seed variability documented in the supplementary material (up to 6.4 points), so we do not interpret it, while FCViT and DifAssemble still benefit. Inpainting cannot fully restore baseline performance for any model under realistic irregular degradation.

These results suggest that while inpainting can partially reduce realistic degradation efects, most strongly for architectures that depend on global visual features, the domain-specific fine-tuning approach (Section 6.5) provides more substantial improvements for archaeological materials.

## 6.8 Ablation Analysis

To isolate the factors that drive reassembly performance on TEXMET, we analyze three variables: fragment count (grid size), material category as a proxy for visual complexity, and degradation type (Section 6.7).

Fragment count. Table 13 reports JPDVT puzzle accuracy as the grid grows from 3 × 3 (9 fragments) to 4 × 4 (16 fragments). The same transition costs JPDVT four times more on textiles than on natural images: as tiles shrink, repetitive weave patterns provide even fewer distinctive cues per tile, so fragment-count sensitivity is strongly amplified by domain dificulty. The other two architectures diverge from this finding rather than repeat it: DifAssemble’s checkpoints difer in training budget across grid sizes, so no controlled cross-grid comparison is possible for it; FCViT cannot be evaluated beyond 3 × 3 at all, since its coordinate-regression head is trained for a fixed nine-fragment layout. The grid constraint itself is thus not only a benchmark simplification but a live architectural boundary of the model class.

Material category. Because expert weave-type labeling of 18,644 objects is impractical, we group the test split by the museum’s curatorial classification as a proxy for visual complexity. Table 14 shows a 2.9× spread in JPDVT baseline accuracy between the easiest and hardest categories: pictorial tapestries (53.3%) provide globally distinctive scene content, whereas printed textiles (18.1%) are dominated by flat repeating patterns whose tiles are locally interchangeable. Woven textiles, the largest group (�=883), fall below the overall mean (29.4% against 32.9%), consistent with the qualitative failure analysis: repetitive weave structure leaves the solver

![](images/2648f0d5ec49f79037f1446d6834e7a6908bd9b0c5667d2738261b6bc9664656.jpg)

Regular Masking Process  
![](images/6ce3e36991aedff9ecdd29728d9155d68829ae8d4ce5efe5d9a43ba68764f813.jpg)

![](images/aa3797606b362345b42216cbba0d39cad785a65b7c5ce58096f6a84e58aa7f0c.jpg)

![](images/7605e09b8596faab5cb05810c637407e1d1c58b7053f8a23ef1b6f4a11bdf792.jpg)

![](images/93b3ac21d1510e8a2be5f7535677ebace9e54d571054124cd5eb0465e920543b.jpg)

![](images/bfe951b7b61bccef1ef8f3a75796c280199bc7e40984e6a1c5462e5d98a007cc.jpg)

![](images/080581fc412c76f81cb210207527e583ab6e8ae2369c56bfc0f203c3626e76b0.jpg)

![](images/d884c100d306fcb9b4fd177e501ebfbdcc196971365160a0583c569ae545b4e3.jpg)  
Fig. 11. The masking and inpainting pipeline on ImageNet validation images. Top: regular centered-rectangle mask. Botom: irregular mask derived from a single Oseberg fragment. Panels left to right: original image, mask, masked image, and ASUKA-inpainted result. The inpainted images restore visual plausibility to the masked regions.  
Inpainting recovery by model and mask type (3×3, 1,000 images per condition)

![](images/200d168081657ecaa23210f723ea2048f62944a59fb2b8cb6ef2e1b65a6403b0.jpg)  
Fig. 12. Inpainting efectiveness: comparison of puzzle accuracy before (masked) and after inpainting for each model under both masking conditions.  
without positional landmarks. The same ordering holds at 4 × 4 (tapestries 18.3%, printed 4.4%), indicating that the category efect is a property of the material rather than of a particular grid size.

Top 10 Best Performing Oseberg Images on Standard ImageNet Benchmark  
![](images/164d0f532f8045cbe6b3dc87bbd0644a25e3259cef9ff5f1014ce116fea737f0.jpg)  
Fig. 13. The ten Oseberg fragment images best reordered by an ImageNet-trained baseline model. Each fragment photograph is divided into a 3×3 grid and reassembled in isolation; nine of the ten are restored completely (9/9 patches) and one partially (7/9). Fragments with distinctive global structure survive severe degradation even under the baseline model, within this simplified seting.

Table 13. Grid-size ablation: JPDVT puzzle accuracy (%) as fragment count increases from 3 × 3 (9 fragments) to $4 \times 4$ (16 fragments). The 4 × 4 configurations use the JPDVT-T variant (patch size 64, 256×256 input) under the same evaluation protocol; TEXMET values are computed on the 1,865-image test split.
<table><tr><td>Dataset</td><td>3×3</td><td> $4 \times 4$ </td><td>Relative drop</td></tr><tr><td>ImageNet</td><td>67.89</td><td>56.62</td><td>-17%</td></tr><tr><td>TEXMET</td><td>32.92</td><td>9.54</td><td>-71%</td></tr></table>

Degradation type. The masking and inpainting analysis (Section 6.7, Table 12) completes the ablation: irregular, archaeologically derived masks harm JPDVT and FCViT substantially more than regular masks, the graph-based DifAssemble is the least sensitive to either, and the benefit of inpainting is largest exactly where degradation hurts most.

## 7 Discussion

## 7.1 Interpretation of Results

The substantial performance diferences between architectures provide insight into what makes a puzzle-solving model efective for textile puzzle reassembly.

Why DifAssemble Succeeded. DifAssemble’s graph-based representation proved particularly well-suited for textile data. By encoding fragments as nodes and spatial relationships as edges, the model can capture the local connectivity patterns that characterize woven structures. The difusion-based refinement process allows iterative improvement of predictions, which appears beneficial for degraded materials where initial feature extraction may be uncertain. Additionally, DifAssemble’s ability to handle rotation and missing pieces aligns well with the realities of archaeological materials.

Table 14. Material-category ablation: ImageNet-trained JPDVT evaluated on TEXMET $3 \times 3$ (the cross-domain seting of Table 8), named material categories with � ≥ 20 test images, MET curatorial classification used as a complexity proxy. The residual Other group is excluded here and reported in the supplementary material.
<table><tr><td>Category</td><td>n</td><td>Puzzle Acc. (%)</td><td>Fragment Acc. (%)</td></tr><tr><td>Tapestries</td><td>60</td><td>53.3</td><td>71.9</td></tr><tr><td>Embroidered</td><td>350</td><td>37.7</td><td>58.9</td></tr><tr><td>Costume</td><td>34</td><td>35.3</td><td>62.1</td></tr><tr><td>Rugs</td><td>36</td><td>33.3</td><td>56.8</td></tr><tr><td>Woven</td><td>883</td><td>29.4</td><td>53.0</td></tr><tr><td>Velvets</td><td>121</td><td>28.9</td><td>60.0</td></tr><tr><td>Printed</td><td>160</td><td>18.1</td><td>44.0</td></tr></table>

Why FCViT Struggled. FCViT’s coordinate regression approach showed severe sensitivity to domain shift, collapsing from competitive ImageNet performance to 8.30% on TEXMET. The regression formulation lacks explicit uniqueness constraints, making it prone to duplicate placements where multiple fragments are assigned similar coordinates (Figure 14). This limitation becomes critical when visual features are degraded or unfamiliar, as the model lacks the structured constraints that could guide predictions in the absence of strong visual signals.

Domain Adaptation Finding. The most significant finding is that domain-specific fine-tuning can overcome much of the domain gap. The 37.77 percentage point improvement on Oseberg fragments (from 26.67% to 64.44%) demonstrates that with appropriate training data, models can learn to handle the unique challenges of archaeological textiles: faded colors, degraded fibers, and missing content. This suggests that the fundamental limitations were not architectural but rather stemmed from training data that did not represent the target domain.

## 7.2 Practical Implications

For practitioners, three implications follow. TEXMET provides the first large-scale, CC0-licensed dataset for textile puzzle-solving research, enabling unrestricted academic and commercial use in heritage preservation. Among the evaluated architectures, our results favor DifAssemble’s graph-based approach, whose robustness to domain shift and ease of fine-tuning make it the most suitable current choice for textile imagery. Finally, domain-specific training proves essential: models trained solely on natural images cannot reliably transfer to textile materials, and even limited domain-specific data substantially improves performance on imagery of real archaeological artifacts.

## 7.3 Fragment Realism and Degradation Efects

A fair question about any benchmark built from synthetic puzzles is how its dificulty relates to the condition of real archaeological material. No established quantitative measure exists for the realism ofa synthetically generated fragment: realism spans several dimensions at once, including shape irregularity, missing area, edge damage, fading, staining, and weave distortion. We therefore assess realism qualitatively, and anchor the evaluation with real material. The contrast can be stated plainly: TEXMET tiles have zero masked area and identical, regular square geometry by construction, whereas the real Oseberg fragments average 75.60% masked area with a mean compactness of 0.15 (isoperimetric compactness, $4 \pi A / P ^ { 2 }$ , where � is the visible fragment area and � its perimeter; 1.0 for a perfect disc, lower for irregular outlines).

FCViT 3x3 Puzzle Solving Results  
![](images/dffe465b2e7280a16f1edbd3f3fab17f1220959246bb7055a770115bf8fd56d1.jpg)  
Fig. 14. FCViT 3×3 puzzle-solving results on ImageNet [5]. Top: successful reconstruction (9/9 pieces correct). Botom: failure case showing the duplicate placement problem, where multiple fragments are assigned to the same coordinates (1/9 pieces correct).

The regular-grid formulation itself is a constraint of the current model class, not a choice of this benchmark. JPDVT and FCViT accept only regular square grids with a fixed piece count, and FCViT additionally supports no rotation. DifAssemble supports rotations and missing pieces, and its graph formulation could in principle extend to irregular outlines, but no trained model for irregular fragment geometry exists. We additionally attempted a more realistic distractor experiment, presenting 16 candidate fragments of which only 9 belong to the puzzle; this failed at the implementation stage for all three models, whose architectures assume a fixed input fragment count. Grid puzzles are, at present, the evaluation this model class supports, which is why we position the benchmark as a controlled first step.

What the evaluation does include is real degradation. The 45 Oseberg fragment images carry genuine archaeological damage: the fading, staining, holes, and irregular outlines visible in Figure 1. Reassembly performance on fragments with larger missing areas is systematically worse, directly linking degradation severity to task dificulty. Most importantly, the fine-tuned model transfers from synthetic training puzzles to this genuinely degraded imagery with a substantial gain (26.67% to 64.44%), evidence that the benchmark’s training signal is relevant beyond its synthetic construction.

Our qualitative assessment is consistent with independent evidence from a parallel line of work. Berels [2] investigated deep feature extraction for content-based retrieval of archaeological textile fragments under irregular masking conditions, explicitly targeting the gap between theoretical puzzle-solving approaches and the practical condition of textile artifacts, which, unlike rigid materials such as pottery, decompose irregularly into fragments with ragged threaded edges and holes. Her evaluation of CNN and ViT feature extractors shows that retrieval performance decreases as masking grows more complex and irregular: with increasing degradation the extracted features become less informative about the depicted content, as the models increasingly respond to the degradation pattern itself rather than to the underlying textile. Moreover, Berels argues that a smaller part of the fragment cropped to a regular size may be more informative than the image of an entire fragment which also depicts degraded areas, since the latter act as a distractor and introduce noise. She concludes that fully automated puzzle solving is not yet feasible for heavily degraded archaeological artifacts whose fragment boundaries no longer match, and instead recommends retrieval-based assistance to archaeologists. These findings mirror our observations at the assembly level: performance degrades most under irregular, archaeologically derived masking (Section 6.7), and inpainting recovers only part of the loss. Together, the two studies suggest that sensitivity to realistic degradation is a property of current deep visual representations in general, rather than an artifact of any single task formulation.

The degradation analysis in Section 6.7 quantifies these efects by type: irregular, archaeologically derived masks are consistently harder than regular grid masks for the transformer-based models, and inpainting recovers only part of the loss, helping most where degradation hurts most. The gap between regular and irregular masking conditions is, in one number, the gap between synthetic and archaeological dificulty. Closing it will require architectures that accept irregular fragment geometry, reason across incomplete fragment sets, and match fragments across, rather than within, textiles.

## 7.4 Limitations

Several limitations constrain the interpretation and applicability of our findings:

Grid-Based Puzzle Formulation. Our evaluation uses regular 3×3 grid puzzles, whereas real archaeological fragments have irregular shapes resulting from damage and degradation. Extending these methods to irregular fragment geometries remains an open challenge.

Simplified Oseberg Evaluation. No ground truth exists for how the Oseberg fragments relate to each other in the original tapestry, making cross-fragment matching evaluation impossible. Instead, we artificially split each surviving fragment into a 3×3 grid of smaller patches to create known ground truth, enabling reliable evaluation of puzzle-solving performance. Consequently, we evaluated individual fragments in isolation rather than attempting full tapestry reconstruction. The real archaeological challenge involves simultaneously reasoning about hundreds of fragments that may belong to multiple distinct textiles.

Limited Archaeological Scale. Our evaluation used 45 Oseberg fragments. While this represents a significant step toward real-world application, larger-scale evaluation would strengthen the findings.

Synthetic Fragmentation. TEXMET uses simulated puzzle cuts on intact textile images. Real archaeological fragmentation involves irregular damage, fiber loss, and material degradation that synthetic puzzles cannot fully capture. The gap between simulated and real fragmentation may limit transfer to the most challenging archaeological cases.

## 8 Conclusion

This paper addressed the challenge of applying deep learning puzzle-solving methods to archaeological textile imagery, as a step toward computational reconstruction, using the Viking Age Oseberg textile fragments as a case study.

## 8.1 Contributions

We presented four main contributions:

(1) TEXMET Dataset: We introduced TEXMET, a dataset of 18,644 curated historical textile images from the Metropolitan Museum of Art’s Open Access collection. To our knowledge, this represents the first textile-specific benchmark for puzzle-solving research.

(2) Systematic Evaluation: We provided the first systematic evaluation of three state-of-the-art puzzlesolving architectures (JPDVT, FCViT, and DifAssemble) on both a large textile dataset and real archaeological fragments from the Oseberg ship burial.

(3) Domain Adaptation Study: We demonstrated that fine-tuning on domain-specific data is essential for archaeological applications. DifAssemble improved from 26.67% to 64.44% puzzle accuracy on Oseberg fragments after fine-tuning on TEXMET, a 37.77 percentage point improvement.

(4) Open-Source Release: All code, experimental frameworks, trained model checkpoints, and the TEXMET dataset will be made publicly available to support reproducibility and future research.

## 8.2 Key Findings

Our experiments revealed several important findings for the application of computational methods to cultural heritage reconstruction:

Domain gap is substantial. Transformer-based models trained on natural images fail to generalize to textile data. FCViT collapsed from competitive ImageNet performance to 8.30% on TEXMET, and JPDVT fell from 67.89% to 32.92%; only the graph-based DifAssemble carried its performance level across the shift (65.08% on TEXMET against a 56.61% ImageNet baseline).

Architecture choice matters. DifAssemble’s graph-based approach demonstrated superior robustness to domain shift compared to pure transformer architectures. Its ability to handle rotation, missing pieces, and variable puzzle sizes makes it in principle particularly suitable for archaeological applications, although the present benchmark exercises only fixed grids without rotation.

Domain-specific training is essential. The dramatic improvement from fine-tuning (particularly the 37.77 percentage point gain on highly degraded Oseberg fragments) demonstrates that appropriate training data can enable efective application of deep learning to archaeological materials

Models learn meaningful patterns. The inscribed cropping analysis confirmed that fine-tuned models learn textile-specific visual features rather than merely exploiting shape information, suggesting genuine pattern recognition capability.

## 8.3 Future Work

Several directions remain for future research:

Irregular Fragment Handling. Extending puzzle-solving methods beyond regular grid configurations to handle the irregular shapes of real archaeological fragments represents a critical next step toward practical application.

Full Tapestry Reconstruction. Scaling from individual fragment puzzles to simultaneous reconstruction of multiple textiles from mixed fragment collections would address the realistic archaeological scenario.

Broader Archaeological Applications. The domain adaptation approach demonstrated here may generalize to other fragmented heritage materials including ceramics, frescoes, and manuscripts.

Expanded Datasets. Creating larger and more diverse textile datasets, potentially incorporating real archaeological fragment imagery, would further improve model robustness and enable evaluation at archaeological scale.

The combination of TEXMET for training and DifAssemble’s graph-based architecture provides a foundation for computational approaches to textile heritage preservation. As these methods mature, they ofer the potential

to accelerate the careful, time-consuming work of archaeological reconstruction while providing new insights into historical material culture.

## 8.4 Data Availability

All source code, experimental frameworks, trained model checkpoints, and the TEXMET dataset will be made publicly available.

## Acknowledgments

The Oseberg textile fragment images were captured by George Alexis Pantos at the Museum of Cultural History, University of Oslo. We thank the Metropolitan Museum of Art for making their Open Access collection available under CC0 licensing. Model training and evaluation were performed on the Idun high-performance computing cluster at NTNU.

## References

[1] Eva Andersson Strand. 2020. Travelling with Textiles: Production, Consumption, and Trade in the Viking Age. Bulletin ofthe Museum ofFar Eastern Antiquities 81 (2020), 65–88.

[2] Marlena Berels. 2025. Deep Feature Extraction under Irregular Masking Conditions: A Step toward Content-Based Image Retrieval for Archaeological Puzzle Solving. Master’s thesis. Norwegian University of Science and Technology.

[3] Yingyi Chen, Xi Shen, Yahui Liu, Qinghua Tao, and Johan A. K. Suykens. 2023. Jigsaw-ViT: Learning Jigsaw Puzzles in Vision Transformer. Pattern Recognition Letters 166 (2023), 53–60. doi:10.1016/j.patrec.2022.12.023

[4] Taeg Sang Cho, Shai Avidan, and William T. Freeman. 2010. A Probabilistic Image Jigsaw Puzzle Solver. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR). 183–190. doi:10.1109/CVPR.2010.5540212

[5] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. 2009. ImageNet: A Large-Scale Hierarchical Image Database. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 248–255.

[6] Niv Derech, Ayellet Tal, and Ilan Shimshoni. 2021. Solving Archaeological Puzzles. Pattern Recognition 119 (2021), 108065. doi:10.1016/j. patcog.2021.108065

[7] Carl Doersch, Abhinav Gupta, and Alexei A Efros. 2015. Unsupervised visual representation learning by context prediction. Proceedings of ICCV (2015), 1422–1430.

[8] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. 2021. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. In International Conference on Learning Representations (ICLR).

[9] H. Freeman and L. Garder. 1964. Apictorial Jigsaw Puzzles: The Computer Solution of a Problem in Pattern Recognition. IEEE Transactions on Electronic Computers EC-13, 2 (1964), 118–127. doi:10.1109/PGEC.1964.263781

[10] Thomas Funkhouser, Hijung Shin, Corey Toler-Franklin, Antonio García Castañeda, Benedict Brown, David Dobkin, Szymon Rusinkiewicz, and Tim Weyrich. 2011. Learning How to Match Fresco Fragments. Journal on Computing and Cultural Heritage 4, 2 (2011), 7:1–7:13. doi:10.1145/2037820.2037824

[11] Davit Gigilashvili, Casper Fabian Gulbrandsen, Ha Thu Nguyen, Margrethe Havgar, Marianne Vedeler, and Jon Yngve Hardeberg. 2024. Toward Solving a Puzzle of Fragmented Archeological Textiles. Journal ofImaging Science and Technology 68, 4 (2024), 1–16. doi:10.2352/J.ImagingSci.Technol.2024.68.4.04040

[12] Davit Gigilashvili, Hana Lukesova, Casper Fabian Gulbrandsen, Akash Harijan, and Jon Yngve Hardeberg. 2023. Computational techniques for virtual reconstruction of fragmented archaeological textiles. Heritage Science 11, 259 (2023).

[13] Davit Gigilashvili, Ha Thu Nguyen, Casper F. Gulbrandsen, Margrethe Havgar, Marianne Vedeler, and Jon Y. Hardeberg. 2023. Texturebased Clustering of Archaeological Textile Images. In Proceedings of Archiving 2023, Vol. 20. Society for Imaging Science and Technology, 139–142. doi:10.2352/issn.2168-3204.2023.20.1.29

[14] D Goldberg, C Malon, and M Bern. 2004. A global approach to automatic solution of jigsaw puzzles. Computational Geometry 28, 2-3 (2004), 165–174.

[15] Casper Fabian Gulbrandsen. 2023. Reconstructing the Original: Machine Learning Puzzle Assembly for Archaeological Textile Reconstruction. Master’s thesis. Norwegian University of Science and Technology.

[16] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep Residual Learning for Image Recognition. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 770–778. doi:10.1109/CVPR.2016.90

[17] Jonathan Ho, Ajay N. Jain, and Pieter Abbeel. 2020. Denoising Difusion Probabilistic Models. In Advances in Neural Information Processing Systems, Vol. 33.

[18] Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. 2018. Progressive Growing of GANs for Improved Quality, Stability, and Variation. In International Conference on Learning Representations (ICLR). Introduces the CelebA-HQ dataset of 30,000 1024×1024 images..

[19] Garam Kim, Hyeonseong Cho, and Hyoungsik Nam. 2025. Solving Jigsaw Puzzles by Predicting Fragment’s Coordinate Based on Vision Transformer (FCViT). Expert Systems with Applications 272 (2025), 126776. doi:10.1016/j.eswa.2025.126776

[20] LanceDB 2024. LanceDB: Developer-friendly, embedded retrieval engine for multimodal AI. https://github.com/lancedb/lancedb. Open-source software, Apache-2.0 license.

[21] Jinyang Liu, Wondmgezahu Teshome, Sandesh Ghimire, Mario Sznaier, and Octavia Camps. 2024. Solving Masked Jigsaw Puzzles with Difusion Vision Transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

[22] Ilya Loshchilov and Frank Hutter. 2019. Decoupled Weight Decay Regularization. In International Conference on Learning Representations (ICLR).

[23] Michael McCloskey and Neal J. Cohen. 1989. Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem. In Psychology ofLearning and Motivation, Gordon H. Bower (Ed.). Vol. 24. Academic Press, 109–165. doi:10.1016/S0079-7421(08)60536-8

[24] Leland McInnes, John Healy, and James Melville. 2018. UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv preprint arXiv:1802.03426 (2018).

[25] B. E. Moore and J. J. Corso. 2020. FiftyOne. GitHub (2020). https://github.com/voxel51/fiftyone.

[26] Mehdi Noroozi and Paolo Favaro. 2016. Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles. In Computer Vision – ECCV 2016 (Lecture Notes in Computer Science, Vol. 9910). 69–84. doi:10.1007/978-3-319-46466-4\_5

[27] Marie-Morgane Paumard, David Picard, and Hedi Tabia. 2018. Image Reassembly Combining Deep Learning and Shortest Path Problem. In Computer Vision – ECCV 2018 (LNCS, Vol. 11210). 155–169. doi:10.1007/978-3-030-01231-1\_10

[28] Marie-Morgane Paumard, David Picard, and Hedi Tabia. 2018. Jigsaw Puzzle Solving Using Local Feature Co-Occurrences in Deep Neural Networks. arXiv preprint arXiv:1807.03155 (2018). doi:10.48550/arXiv.1807.03155

[29] Dolev Pomeranz, Michal Shemesh, and Ohad Ben-Shahar. 2011. A Fully Automated Greedy Square Jigsaw Puzzle Solver. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR). 9–16. doi:10.1109/CVPR.2011.5995331

[30] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. CLIP: Learning Transferable Visual Models From Natural Language Supervision. arXiv preprint arXiv:2103.00020 (2021).

[31] Gianluca Scarpellini, Stefano Fiorini, Francesco Giuliari, Pietro Morerio, and Alessio Del Bue. 2024. DifAssemble: A Unified Graph-Difusion Model for 2D and 3D Reassembly. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 28098–28108.

[32] Dror Sholomon, Omid David, and Nathan S. Netanyahu. 2013. A Genetic Algorithm-Based Solver for Very Large Jigsaw Puzzles. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 1767–1774. doi:10.1109/CVPR.2013.231

[33] Jiaming Song, Chenlin Meng, and Stefano Ermon. 2021. Denoising Difusion Implicit Models. In International Conference on Learning Representations (ICLR).

[34] Xingke Song, Jiahuan Jin, Chenglin Yao, Shihe Wang, Jianfeng Ren, and Ruibin Bai. 2023. Siamese-Discriminant Deep Reinforcement Learning for Solving Jigsaw Puzzles with Large Eroded Gaps. In Proceedings ofthe AAAI Conference on Artificial Intelligence, Vol. 37. 2303–2311. doi:10.1609/aaai.v37i2.25325

[35] Mingxing Tan and Quoc Le. 2019. EficientNet: Rethinking Model Scaling for Convolutional Neural Networks. In Proceedings ofthe 36th International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 97). 6105–6114.

[36] The Metropolitan Museum of Art. 2024. Open Access Artworks Dataset. https://github.com/metmuseum/openaccess. CC0 public-domain images and metadata. Accessed: 2025-01-15.

[37] Theodore Tsesmelis, Luca Palmieri, Marina Khoroshiltseva, Marcello Pelillo, Alessio Del Bue, et al. 2024. Re-assembling the Past: The RePAIR Dataset and Benchmark for Real-World 2D and 3D Puzzle Solving. In Proceedings ofthe 38th International Conference on Neural Information Processing Systems. 30076–30105.

[38] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention Is All You Need. In Advances in Neural Information Processing Systems, Vol. 30. 5998–6008.

[39] Marianne Vedeler. 2019. The Oseberg Tapestries. Scandinavian Academic Press, Oslo, Norway.

[40] Yikai Wang, Chenjie Cao, Junqiu Yu, Ke Fan, Xiangyang Xue, and Yanwei Fu. 2025. Towards Enhanced Image Inpainting: Mitigating Unwanted Object Insertion and Preserving Color Consistency. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 23237–23248.

[41] Shu Zhong, Miriam Ribul, Youngjun Cho, and Marianna Obrist. 2023. TextileNet: A Material Taxonomy-based Fashion Textile Dataset. arXiv preprint arXiv:2301.06160 (2023).