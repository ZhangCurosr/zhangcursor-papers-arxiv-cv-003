# Zhengrong-Spatial-Aware 1.0: Advancing 3D Multimodal Intelligence

Yu Zhang<sup>1</sup>, Xiaogang Wang<sup>2</sup>, Yuhang Cheng<sup>1</sup>, Jingrui Li<sup>1</sup>, Wenbo Chen<sup>1</sup>

<sup>1</sup>Beijing Freedo Technology Co., Ltd.

<sup>2</sup>Southwest University

Abstract—This report presents the implementation and validation of two end-to-end multimodal adaptation pipelines for 3D-aware visual question answering on a custom spatial dataset. The first pipeline finetunes HiSpatial-3B, which augments RGB observations with metric-scale 3D point maps; the second adapts Qwen3-VL through memory-eficient QLoRA-based supervised fine-tuning. The completed engineering stack covers dataset conversion, image–annotation alignment, monocular geometry preprocessing, instruction construction, mixed-domain training, checkpoint management, batch inference, and automated evaluation. On the target spatial benchmark, HiSpatial adaptation increases overall accuracy from 1.73% before fine-tuning to 79.89% at 3,600 steps. On a separately collected styled split, accuracy improves from 14.9% to 55.1%, indicating transfer beyond the original training set. The Qwen3-VL pipeline is likewise implemented end-to-end for both 8Band 32B-scale models using mixed-domain QLoRA adaptation. The resulting accuracy improves from 20.74% to 45.27% for Qwen3-VL-8B and from 30.79% to 47.60% for Qwen3-VL-32B. Together, these results establish two complementary and reproducible adaptation pipelines for geometry-aware and general-purpose multimodal spatial reasoning. These results establish a reproducible engineering foundation for subsequent work on metric 3D reasoning, spatial grounding, and domain-adapted multimodal systems, while also identifying training-data composition as a central factor in preserving general multimodal capability during specialization.

Index Terms—vision-language model, 3D spatial reasoning, multimodal fine-tuning, HiSpatial, Qwen3-VL, QLoRA, supervised fine-tuning, point map

## I. Introduction

Vision-language models (VLMs) have achieved strong performance in semantic recognition, captioning, and visual question answering, but reliable metric 3D reasoning remains substantially more challenging when the model observes only a 2D image. The target project focuses on spatial tasks that require both semantic understanding and geometric sensitivity, including object counting, buildingheight estimation, footprint comparison, pairwise distance estimation, spatial ordering, and relative-direction reasoning.

The objective of this work is to build and validate a practical adaptation stack for strong open-source multimodal models on the provided 3D-aware dataset, rather than to train a new foundation model from scratch. Two complementary model families are studied. HiSpatial explicitly injects geometric information through a metric point-map branch and is therefore naturally suited to quantitative and relational spatial reasoning. Qwen3-VL, by contrast, provides a strong general-purpose multimodal baseline with learned spatial and 3D grounding capabilities, enabling us to evaluate how far task specialization can be achieved through instruction tuning alone.

The completed work delivers the following engineering outcomes:

• a model-independent data layer that converts the provided scenes into aligned image–question–answer records and supports reproducible train/test splits;

• an RGB+3D preprocessing pipeline for HiSpatial based on monocular point-map estimation and WebDataset serialization;

• mixed-domain HiSpatial supervised fine-tuning, checkpointed inference, and automated evaluation, with target accuracy improving from 1.73% to 79.89%;

• Qwen3-VL QLoRA pipelines for both 8B and 32B variants, including instruction conversion, low-memory training, checkpoint loading, and standardized evaluation;

• a unified prediction/evaluation interface for comparing spatial VLMs on the same JSONL benchmark; and

• an empirical diagnosis of narrow-domain overspecialization in Qwen3-VL, which motivates mixeddomain sampling and stronger retention controls in subsequent experiments.

## II. Model Principles and Literature Background

## A. HiSpatial: Hierarchical 3D Spatial Understanding

HiSpatial formulates 3D spatial intelligence as a fourlevel hierarchy: basic geometric perception, object-level spatial understanding, inter-object relational understanding, and abstract spatial reasoning [1]. The hierarchy spans low-level depth or point queries, object-centric properties such as physical size and orientation, pairwise relations such as distance and direction, and higher-level tasks including perspective taking and multi-step spatial problem solving.

A central architectural choice is to supplement RGB appearance with explicit geometric information. During both training and inference, a monocular geometry estimator produces a point map

$$
\mathbf { P } = G ( \mathbf { I } ) , \qquad \mathbf { P } \in \mathbb { R } ^ { H \times W \times 3 } ,\tag{1}
$$

where I denotes the RGB image and each pixel is associated with a 3D coordinate. HiSpatial uses MoGe-style monocular geometry estimation, whose original formulation predicts an afine-invariant point map from a single image [3]. Metric-scale point maps are especially relevant to the target benchmark because several tasks require quantitative reasoning in physical units rather than only qualitative spatial relations.

HiSpatial-3B is built on a PaliGemma2-3B visuallanguage backbone. PaliGemma 2 combines a SigLIP vision encoder with the Gemma 2 language model and is explicitly designed as a transfer-friendly VLM family [4]. HiSpatial adds 3D-aware processing so that visual representations can be conditioned on both appearance and geometry. A simplified representation of the learning objective is

$$
\mathbf { v } = E _ { \mathrm { r g b } } ( \mathbf { I } ) ,\tag{2}
$$

$$
\begin{array} { r } { \mathbf { s } = E _ { \mathrm { x y z } } ( \mathbf { P } ) , } \end{array}\tag{3}
$$

$$
\begin{array} { r } { \mathbf { h } = F ( \mathbf { v } , \mathbf { s } , \mathbf { q } ) , } \end{array}\tag{4}
$$

$$
\mathcal { L } _ { \mathrm { S F T } } = - \sum _ { t } \log p _ { \theta } ( y _ { t } \mid y _ { < t } , \mathbf { h } ) ,\tag{5}
$$

where q is a spatial question and $y _ { t }$ is the target answer token. This formulation is well matched to the project dataset, in which supervision is defined by physically meaningful geometric quantities and relations rather than by object semantics alone.

## B. Qwen3-VL: General Multimodal Reasoning with 3D Grounding

Qwen3-VL follows a three-module multimodal architecture comprising a vision encoder, an MLP-based vision– language merger, and a Qwen3 language-model decoder [2]. Relative to earlier Qwen-VL generations, the model introduces interleaved MRoPE for improved spatial-temporal position modeling and DeepStack for injecting multi-level visual features into the language model. Its training recipe also explicitly includes spatial reasoning and 3D grounding data, making it a suitable general-purpose counterpart to the more geometry-specialized HiSpatial pipeline.

For adaptation in this project, Qwen3-VL is treated as a general multimodal foundation model and fine-tuned with a QLoRA-style recipe. QLoRA keeps the pretrained base model in low precision while optimizing low-rank adapters [5]. For a frozen quantized weight matrix $\hat { \mathbf { W } } _ { 0 }$ , the adapted transformation is

$$
\mathbf { y } = \left( \widehat { \mathbf { W } } _ { 0 } + \alpha \mathbf { B } \mathbf { A } \right) \mathbf { x } ,\tag{6}
$$

where A and B are low-rank trainable matrices. This substantially reduces memory use and makes 8B/32B adaptation feasible on limited GPU resources.

The two branches provide complementary adaptation strategies. HiSpatial introduces explicit geometry at the input level, whereas Qwen3-VL relies primarily on learned visual-language representations and spatial priors. Evaluating both pipelines therefore helps separate limitations in geometric reconstruction from limitations introduced by domain adaptation, instruction formatting, or languageside reasoning.

TABLE I  
Target spatial VQA tasks in the provided dataset.
<table><tr><td>Capability</td><td>Representative questions</td></tr><tr><td>Counting</td><td>number of buildings in a scene</td></tr><tr><td>Metric estimation</td><td>building height; pairwise center distance</td></tr><tr><td>Extremum reason- ing</td><td>tallest/lowest building; largest/smallest footprint</td></tr><tr><td>Ordering</td><td>footprint ranking; left-to-right and top- to-bottom order</td></tr><tr><td>Relational reason- ing</td><td>farthest pair; relative spatial direction</td></tr></table>

$$
\mathrm { I I I . ~ P R O V I D E D ~ D A T A S E T ~ A N D ~ T A S K ~ F O R M U L A T I O N }
$$

## A. Dataset Organization

The provided dataset was collected and refined in several batches. The implemented pipeline supports raw and styled RGB images, colored bounding-box overlays, instance masks, camera-pose metadata, per-scene annotations, and JSONL-formatted VQA supervision. For the buildingcentric scenes, CAD/GLB-derived 3D assets are also available and can be incorporated in future geometry-aware extensions.

A typical VQA item is represented as

```json
{
"sample_id": "sample_000123",
"q_idx": "q7",
"question": "What is the distance between ...?",
"gt": "The horizontal center distance is ... meters."
}
```

Each sample identifier is deterministically mapped to its corresponding visual input, allowing the same metadata to be reused consistently for training, batch inference, and checkpoint evaluation.

## B. Spatial Question Types

The benchmark contains twelve question categories. Table I groups them by the underlying capability being evaluated rather than by their internal question index.

Evaluation is task-aware. Counting questions use direct comparison, metric questions such as height and distance use numerical parsing with a relative-tolerance criterion, and open-form relational answers are normalized and evaluated semantically. This design makes the evaluator robust to surface-form variation while preserving stricter treatment of quantitative outputs.

## IV. Engineering Pipeline

## A. End-to-End System Overview

Figure 3 summarizes the completed workflow. A shared data layer first prepares model-independent VQA records and validates image–annotation alignment. The HiSpatial branch then augments each image with a dense 3D point map and serializes the result for full-model supervised finetuning, whereas the Qwen3-VL branch converts the same task supervision into multimodal conversational instructions for QLoRA adaptation. Both branches converge on a common checkpointed inference and automated evaluation stage.

![](images/ecb5fab33ff2bf56cd2d3b5173d665aec5950825e206ec842d7dcd6cd03425ef.jpg)  
Fig. 1. Reference model architectures used in this project. The HiSpatial architecture integrates RGB appearance with a metric-scale point map for geometry-aware spatial reasoning, while Qwen3-VL employs a vision encoder, vision–language merger, and Qwen3 decoder with DeepStack-based multi-level visual feature injection [1], [2].

## B. HiSpatial Data Preparation

The HiSpatial training stack requires synchronized RGB inputs, spatial tensors, and VQA metadata. The implemented preprocessing stage performs four main operations:   
1) load the target scene image and its aligned VQA records;   
2) run MoGe on the image to produce a dense XYZ tensor;   
3) serialize the image, geometry tensor, and conversation/answer metadata into WebDataset-style samples;

4) mix target-domain samples with general visual instruction data before fine-tuning.

In the first successful adaptation setting, target-domain building data were mixed with LLaVA-NeXT-style general VQA samples at approximately a 1:1 ratio. Prior work on broad visual instruction tuning highlights the value of diverse supervision for preserving general multimodal behavior [6]. This consideration is also supported by our experiments: highly concentrated spatial supervision can teach the target answer format efectively while simultaneously narrowing the response distribution of a general VLM.

The prepared HiSpatial samples contain three synchronized components:

$$
\mathcal { D } _ { \mathrm { H i S p a t i a l } } = \{ ( \mathbf { I } _ { i } , \mathbf { P } _ { i } , \mathbf { c } _ { i } ) \} _ { i = 1 } ^ { N } ,\tag{7}
$$

where $\mathbf { P } _ { i }$ is the precomputed point map and $\mathbf { c } _ { i }$ is the VQA conversation. Precomputing and serializing the point maps decouples geometry estimation from model optimization, reduces repeated preprocessing cost, and improves experiment reproducibility.

## C. HiSpatial Fine-Tuning and Checkpoint Evaluation

Training and inference are maintained in separate environments to reduce dependency conflicts. Fine-tuning starts from the released HiSpatial-3B checkpoint rather than from random initialization. The initial validated configuration used a per-device batch size of 4, an efective total batch size of 32, 1,200 optimization steps, and checkpointing every 400 steps. Subsequent runs extended the training horizon and adjusted the data mixture while preserving the same end-to-end pipeline.

![](images/644727f55813723cbd099271688bb15f256195a2cf4fa1b755e18b6f09c0b7c4.jpg)  
Fig. 2. Example visual input from the provided spatial dataset. The engineering pipeline can consume original RGB images as well as styled or region-highlighted variants.

Training can be launched through the project-level wrapper:

# HiSpatial training environment   
conda activate hispatial-train   
# Example launch   
bash scripts/run\_train.sh configs/train\_finetune\_lhzzzzzy.json

Checkpoint evaluation is fully automated. Each selected checkpoint is loaded, evaluated over a JSONL test split, and exported to a standardized prediction file that is then consumed by the task-aware evaluator. The same interface also supports later styled-image variants and colored bounding-box prompts, allowing new visual-input formats to be tested without changing the evaluation protocol.

## D. Qwen3-VL Instruction Construction and QLoRA

For Qwen3-VL, the same VQA supervision is transformed into the multimodal conversational format expected by the fine-tuning framework. A representative training record is

```jsonl
{
"image": [".../sample_000101/rgb_mask_id_overlay.png"],
"conversations": [
{"from": "human",
"value": "How many buildings are ...?\n<image>"},
{"from": "gpt",
"value": "There are ... buildings."}
]
}
```

The first experiment used 1,000 scenes, with 900 assigned to training and 100 to testing. Because each scene contributes 12 question types, this split yields approximately 10,800 training QA instances and 1,200 test instances.

The implemented Qwen adaptation uses 4-bit quantization with low-rank adapters to reduce memory consumption. In the lightweight recipe, the vision encoder remains frozen while LoRA parameters are optimized in languageside transformer projections; the multimodal connector can also be enabled depending on the configuration. This design makes 8B- and 32B-scale specialization feasible without maintaining gradients for the full pretrained parameter set.

## E. Unified Inference and Evaluation

A common inference interface maps each sample identifier to its image, executes the selected model/checkpoint, and writes one normalized JSON object per prediction. The downstream evaluator then applies task-specific scoring: direct matching for counting, numerical parsing with relative tolerance for metric estimation, and semantic normalization for open-form spatial answers.

A unified multi-model runner additionally allows HiSpatial, Qwen3-VL, LLaVA-3D, and SpatialRGPT to consume the same JSONL benchmark. This abstraction removes model-specific evaluation scripts from the comparison loop and makes checkpoint/model comparisons substantially more reproducible.

## V. Experimental Results and Engineering Findings

## A. HiSpatial Reproduction Sanity Check

Before target-domain adaptation, the reproduced HiSpatial training and inference stack was validated on standard spatial benchmarks. Despite using substantially less compute than the original training setup, the reproduced pipeline achieved 94.17% on CV-Bench-3D, 94.77% on CV-Bench Relation, 73.90% on EmbSpatial, and 78.05% on RoboSpatial in the project records. These measurements are used as an implementation sanity check rather than as a claim of full reproduction of the original training regime.

## B. Target-Domain HiSpatial Adaptation

The clearest quantitative outcome of the completed engineering work is the successful target-domain adaptation of HiSpatial. As shown in Table II, overall accuracy on the original building-centric benchmark rises from 1.73% before adaptation to 64.69% after 1,200 steps, 73.90% after 2,400 steps, and 79.89% after 3,600 steps.

The improvement is not confined to the original target split. On a separately collected styled dataset, accuracy increases from 14.9% for the released model to 49.2%, 52.8%, and 55.1% for the 1,200-, 2,400-, and 3,600-step checkpoints, respectively. The consistent gain on this separate split provides evidence that the pipeline is learning transferable task structure rather than merely memorizing the original scenes.

A later data-mixture experiment further showed that answer-format composition materially afects adaptation.

![](images/464c3276f07fa6578fc3e2f2f7270403069d1daf10c0910485e7fac8b95dcd42.jpg)  
Fig. 3. Implemented fine-tuning and evaluation pipeline. The two model-specific branches share the same task supervision and evaluation interface but difer in geometric preprocessing and adaptation strategy.

TABLE II  
Recorded HiSpatial accuracy on the original target dataset during fine-tuning.
<table><tr><td>Checkpoint / model</td><td>Overall accuracy (%)</td></tr><tr><td>Released HiSpatial before adaptation</td><td>1.73</td></tr><tr><td>Fine-tuned, 1,200 steps</td><td>64.69</td></tr><tr><td>Fine-tuned, 2,400 steps</td><td>73.90</td></tr><tr><td>Fine-tuned, 3,600 steps</td><td>79.89</td></tr></table>

Naively combining free-form and multiple-choice supervision caused a pronounced drop on the original free-form task, whereas an 8:2 mixture recovered strong performance. This result indicates that output-format distribution should be treated as an explicit component of the training domain rather than as an incidental formatting choice.

## C. Qwen3-VL Fine-Tuning with Mixed-Domain Supervision

The Qwen3-VL adaptation pipeline was implemented for both the 8B and 32B variants using memory-eficient QLoRA. Following the data-mixture strategy adopted for HiSpatial, the target-domain spatial instruction data were combined with general multimodal instruction samples during fine-tuning. This mixed-domain formulation was designed to adapt the models to the target spatial tasks while retaining the broader visual-language capabilities of the pretrained models.

The resulting checkpoints show clear improvements over the corresponding pretrained baselines. As summarized in Table III, Qwen3-VL-8B improves from 20.74% before finetuning to 45.27% after mixed-domain adaptation, while Qwen3-VL-32B improves from 30.79% to 47.60%. These results demonstrate that the implemented QLoRA pipeline can successfully specialize both model scales to the target spatial benchmark when the target supervision is balanced with more general multimodal instruction data.

TABLE III  
Qwen3-VL accuracy before and after mixed-domain QLoRA fine-tuning.
<table><tr><td>Model</td><td>Before FT</td><td>After FT</td></tr><tr><td>Qwen3-VL-8B</td><td>20.74%</td><td>45.27%</td></tr><tr><td>Qwen3-VL-32B</td><td>30.79%</td><td>47.60%</td></tr></table>

The improvement is consistent with the behavior observed in the HiSpatial experiments: concentrated targetdomain supervision alone is not necessarily the most efective adaptation strategy for a general-purpose multimodal model, whereas mixed-domain training provides a better balance between task specialization and capability retention. In the present implementation, the same general strategy is therefore used for both model families, although the optimization mechanisms difer: HiSpatial is adapted with its RGB+3D training pipeline, while Qwen3-VL is optimized through low-rank adapters on top of the quantized pretrained model.

The successful Qwen3-VL experiments also complete the second model branch of the engineering pipeline. The implementation covers instruction conversion, mixed-domain dataset construction, QLoRA optimization, checkpoint serialization and reloading, standardized batch inference, and automated evaluation for both 8B- and 32B-scale models.

## D. Current Implementation Status

Table IV summarizes the completed components.

## VI. Reproducibility and Software Organization

The software stack separates data conversion, training, inference, and evaluation environments in order to minimize dependency conflicts across model families. The principal environments are hispatial, hispatial-train, and qwen3vl, with additional environments reserved for spatial-VLM baselines.

TABLE IV Current engineering status.
<table><tr><td>Component</td><td>Status</td></tr><tr><td>Dataset JSONL conversion and image align- ment</td><td>Complete</td></tr><tr><td>Bounding-box / styled-image input support</td><td>Complete</td></tr><tr><td>MoGe point-map preprocessing HiSpatial WebDataset construction</td><td>Complete</td></tr><tr><td>HiSpatial mixed-domain full fine-tuning</td><td>Complete</td></tr><tr><td>HiSpatial checkpoint inference/evaluation</td><td>Complete</td></tr><tr><td>Qwen3-VL instruction conversion</td><td>Complete</td></tr><tr><td>Qwen3-VL mixed-domain QLoRA fine-</td><td>Complete Complete</td></tr><tr><td>tuning</td><td></td></tr><tr><td>Unified prediction and automated evalua- Complete tion 8B/32B checkpoint evaluation</td><td></td></tr></table>

Each experiment is designed to be recoverable from four artifacts: (1) the dataset JSONL, (2) the corresponding image root, (3) the model checkpoint and configuration, and (4) the exported evaluation results. A minimal HiSpatial evaluation run therefore has the form

conda activate hispatial-train   
python tools/run\_hispatial\_checkpoint\_jsonl.py \   
--dataset <test.jsonl> \   
--image-root <image\_dir> \   
--checkpoint <checkpoint.ckpt> \   
--output <predictions.jsonl>   
conda activate hispatial   
python tools/eval\_data4\_color\_preds.py \   
--input <predictions\_aug.jsonl> \   
--output <evaluation.json>

The same convention is used for Qwen3-VL: each checkpoint is passed through the standardized runner and produces predictions in the same JSONL schema. Consequently, downstream model comparison does not require evaluator-specific changes. The use of a shared data interface and evaluation protocol also makes it possible to compare diferent adaptation strategies without modifying the benchmark implementation.

## VII. Discussion and Next Steps

The experiments lead to three main observations. First, the HiSpatial pipeline performs strongly on the target spatial benchmark, benefiting from explicit RGB+pointmap inputs for metric reasoning. Second, mixed-domain training is efective for both model families. Under this strategy, Qwen3-VL-8B improves from 20.74% to 45.27%, while Qwen3-VL-32B improves from 30.79% to 47.60%. Third, HiSpatial and Qwen3-VL provide complementary adaptation strategies: the former relies on explicit 3D geometry, whereas the latter achieves substantial gains through parameter-eficient multimodal instruction tuning.

Future work will focus on target/general data-mixture ratios, task-wise evaluation, input-format ablations, and comparisons between monocular point maps and geometry derived directly from the available 3D assets or camera metadata.

## VIII. Conclusion

This work establishes two end-to-end adaptation pipelines for a custom 3D-aware spatial benchmark. The HiSpatial pipeline improves overall accuracy from 1.73% to 79.89% and raises performance on a separately collected styled split from 14.9% to 55.1%. The Qwen3-VL pipeline is implemented for both 8B and 32B models using mixeddomain QLoRA, improving accuracy from 20.74% to 45.27% and from 30.79% to 47.60%, respectively.

Overall, the project provides a reproducible engineering foundation for further study of metric 3D reasoning, spatial grounding, multimodal data mixing, and geometry-aware model adaptation.

## References

[1] H. Liang, Y. Shen, Y. Deng, S. Xu, Z. Feng, T. Zhang, Y. Liang, and J. Yang, “HiSpatial: Taming Hierarchical 3D Spatial Understanding in Vision-Language Models,” Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR), 2026.

[2] S. Bai, Y. Cai, R. Chen, et al., “Qwen3-VL Technical Report,” arXiv preprint arXiv:2511.21631, 2025.

[3] R. Wang, S. Xu, C. Dai, J. Xiang, Y. Deng, X. Tong, and J. Yang, “MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision,” arXiv preprint arXiv:2410.19115, 2024.

[4] A. Steiner, A. S. Pinto, M. Tschannen, et al., “PaliGemma 2: A Family of Versatile VLMs for Transfer,” arXiv preprint arXiv:2412.03555, 2024.

[5] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, “QLoRA: Eficient Finetuning of Quantized LLMs,” Advances in Neural Information Processing Systems (NeurIPS), 2023.

[6] F. Li, R. Zhang, H. Zhang, Y. Zhang, B. Li, W. Li, Z. Ma, and C. Li, “LLaVA-NeXT-Interleave: Tackling Multi-image, Video, and 3D in Large Multimodal Models,” arXiv preprint arXiv:2407.07895, 2024.