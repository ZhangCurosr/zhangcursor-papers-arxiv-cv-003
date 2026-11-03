# PlantVLM-Edge: Quantized Vision-Language Models for On-Device Crop Health Diagnosis

Gaurab Wagle<sup>a</sup>, Bishal Adhikari<sup>b</sup>, Jiajia Li<sup>c</sup>, Xiaolei Huang<sup>d</sup> and Dong Chen\*<sup>b</sup>

<sup>a</sup>Department of Computer Science and Engineering, Kathmandu University, 45210, Nepal

<sup>b</sup>Department of Agricultural and Biological Engineering, Mississippi State University, MS 39762, USA

<sup>c</sup>Electrical and Computer Engineering, Michigan State University, MI 48824, USA

<sup>d</sup>Electrical Engineering and Computer Science, University of Missouri, MO 65211, USA

\*Dong Chen is the corresponding author

## A R T I C L E I N F O

Keywords: Vision-language models Model quantization Edge computing Plant disease diagnosis Insect pest identification Precision agriculture Jetson AGX Orin

## A BS T RA C T

Timely and accurate crop health diagnosis is essential for sustainable agricultural production, yet accurate diagnosis often requires expert knowledge that is not readily available. Recent visionlanguage models (VLMs) have demonstrated remarkable capabilities in visual understanding, reasoning, and natural-language interaction, creating new opportunities for intelligent crop health assessment. However, deploying VLMs on resource-constrained devices remains challenging due to their substantial computational and memory requirements. This paper presents PlantVLM-Edge, a quantized vision-language framework for conversational crop health diagnosis that can run entirely on resource-constrained edge devices. PlantVLM-Edge combines agricultural domain adaptation, post-training quantization, and on-device serving with an ofline chat interface, allowing growers to ask open-ended questions and receive diagnoses, symptom explanations, and management guidance without cloud connectivity. The framework is backbone-agnostic: we validate it with two diferent VLM types with varying scale and architecture (Qwen3-VL-2B and LLaVA-1.5-7B), four quantization schemes, and two diagnostic domains (plant diseases and insect pests), and we deploy all quantized variants of both models on an NVIDIA Jetson AGX Orin. Eight-bit quantization leaves answer quality essentially unchanged relative to the full-precision model, and 4-bit activation-aware weight quantization (AWQ) retains 98.6% of full-precision answer quality on both domains while reducing the weight memory of the 2B backbone by 49%. In its most compact form, PlantVLM-Edge requires 2.01 GiB of weight memory, returns its first token in about 0.12 s, and completes a 64-token diagnostic response in 0.78 s. These results demonstrate that timely and accurate crop health diagnosis is practical on edge devices, providing a foundation for next-generation crop monitoring and decision support systems.

## 1. Introduction

Crop health underpins agricultural productivity and global food security, and protecting it depends on detecting threats early enough to act. Plant diseases and insect pest infestations are among the most persistent of these threats, accounting for annual crop yield losses of up to 40% (Food and Agriculture Organization, 2026; Savary et al., 2019). Early and accurate diagnosis is essential to limiting economic damage, yet in practice it still depends heavily on trained agronomists or plant pathologists (Swanson and Rajalahti, 2010), a resource that is often scarce, expensive, or simply inaccessible in smallholder and remote farming contexts. This gap has motivated research into automated and scalable approaches to crop health monitoring that can bring expert-level diagnosis to growers without direct access to specialists.

Deep learning-based computer vision has made imagebased recognition of plant diseases and pests highly accurate on benchmark datasets (Singh et al., 2020b; Liu and Wang, 2021), first with convolutional neural network classifiers (Mohanty et al., 2016; Ferentinos, 2018) and more recently with transformer-based classifiers and detectors (Meng et al., 2023; Tan et al., 2025). However, models trained on curated images often generalize less well to field conditions (Singh et al., 2020a), and more fundamentally, they formulate diagnosis as a fixed recognition task. Their output is a class label or a bounding box, with no explanation of the visual evidence behind it and no way to ask followup questions. For non-expert growers, an isolated label is a major usability barrier, because it does not explain what the problem means or what to do about it. Addressing this requires models that go beyond passive detection to provide flexible, interactive decision support (Novaliendry et al., 2026).

Vision-language models (VLMs), which couple a pretrained visual encoder with a language model (Ghosh et al., 2024; Zhao et al., 2026), ofer a path toward such support. A single VLM can answer free-form questions about an image, from identifying a crop or pest to explaining likely causes and recommending management actions, and it can be adapted to a new domain by fine-tuning on questionanswer data (Liu et al., 2023; Li et al., 2024). Agricultural studies have adapted VLMs to crop disease and pest diagnosis through instruction tuning and parameter-eficient fine-tuning (Awais et al., 2025; Wang et al., 2024; Sharma et al., 2024; Kaur et al., 2025; Jena and Arjunan, 2026), and benchmarks such as CDDM (Liu et al., 2024b), Ag-MMU (Gauba et al., 2026), IP-VQA (Jin et al., 2025), and PlantExpertVQA (Sakib et al., 2026) now evaluate diagnostic question answering beyond fixed labels. These studies consistently show that general-purpose VLMs improve substantially once adapted to the agricultural domain (Section 2.2).

However, field deployments may prevent directly adopting large VLM toolkits due to resource constraints. In rural farming environments, unreliable internet connectivity, recurring cloud service costs, and data privacy concerns make on-device inference highly desirable (Yao et al., 2025; Yu et al., 2025). VLMs typically contain billions of parameters, which makes local execution expensive, and memory and speed remain limiting on edge devices. When run with llama.cpp inference framework, with its language model quantized to 4 bits, the general-purpose LLaVA-1.5-7B (Liu et al., 2024a) still occupied 4.7 GB and took about 20 s to answer a single question on an NVIDIA Jetson AGX Orin and about 330 s on a smartphone (Chu et al., 2023), delays that are impractical for a grower awaiting a diagnosis.

Post-training quantization, which lowers the numerical precision of a model without retraining it, is a natural way to reduce these costs. However, existing edge evaluations of VLM quantization have used general-purpose models and benchmarks, and report that its speed benefits depend on the hardware and inference software (Section 2.3). How quantization afects the answer quality, memory footprint, and latency of domain-adapted agricultural VLMs on embedded hardware therefore remains largely unexamined, as does whether a compact VLM can match a larger one once both are adapted and compressed.

To address this gap, we present PlantVLM-Edge, a quantized vision-language framework for conversational crop health diagnosis that runs entirely on edge devices. PlantVLM-Edge consists of three stages: agricultural domain adaptation of a pretrained VLM through parametereficient fine-tuning, post-training quantization of the adapted model, and on-device serving with an ofline chat interface that growers can access from a web browser on a local network. The main contributions of this work are as follows:

• We develop PlantVLM-Edge, an end-to-end framework that takes a general-purpose VLM from domain adaptation to fully ofline deployment, allowing growers to ask open-ended questions and receive diagnoses, symptom explanations, and management guidance without cloud connectivity.

• We show that the framework is backbone-agnostic by instantiating it with two VLMs of diferent scale and architecture, Qwen3-VL-2B and LLaVA-1.5-7B, across two diagnostic domains: plant diseases and insect pests.

• We systematically evaluate four post-training quantization configurations, spanning round-to-nearest,

AWQ, and SmoothQuant at 8-bit and 4-bit precision. We measure both answer quality and on-device eficiency (weight memory, time to first token, intertoken latency, and end-to-end latency) on an NVIDIA Jetson AGX Orin.

• We provide practical deployment guidance. Eightbit quantization leaves answer quality essentially unchanged, 4-bit AWQ retains 98.6% of full-precision answer quality while reducing weight memory by 49%, and the choice of backbone afects deployment cost more than the choice of quantization method.

The remainder of this paper is organized as follows. Section 2 reviews related work, Section 3 describes the datasets, the PlantVLM-Edge framework, and the evaluation protocol, Section 4 presents the results, Section 5 discusses their implications and limitations, and Section 6 concludes the paper.

## 2. Related Works

## 2.1. Computer vision for crop health diagnosis

Deep learning, and computer vision in particular, has become the dominant approach to automated crop disease and pest recognition. Early methods relied on handcrafted visual descriptors such as color, texture, and shape, combined with classical machine-learning classifiers such as support vector machines and random forests (Sladojevic et al., 2016). Subsequently, convolutional neural networks (CNNs), which learn hierarchical visual features directly from images, substantially outperformed these approaches once large labeled datasets became available. For example, Mohanty et al. (Mohanty et al., 2016) trained CNNs on 54,306 PlantVillage images covering 14 crop species and 26 diseases, achieving 99.35% accuracy, and Ferentinos (Ferentinos, 2018) extended this to 58 plant-disease classes across 25 crops with a best accuracy of 99.53%. However, these results were obtained on images captured under controlled conditions. When Mohanty et al. tested their model on images from other sources, accuracy fell to about 31%. Consequently, later datasets such as PlantDoc (Singh et al., 2020a) were introduced specifically to capture real-world field conditions. To address this challenge, subsequent work improved robustness through transfer learning, data augmentation, and field-acquired training images (Thenmozhi and Reddy, 2019; Xu et al., 2022; Gui et al., 2021). CNNbased object detectors, particularly the YOLO family, have also been widely adopted for localizing diseased regions and pests in real time, including on embedded hardware (Zhang et al., 2024).

More recently, transformer-based architectures have been adopted for crop health diagnosis. The vision transformer (ViT) (Dosovitskiy, 2020) divides an image into patches and uses self-attention to model relationships across the entire image, capturing global context that the local receptive fields of CNNs can miss. Pretraining on large plant image collections has proven particularly efective.

For example, a ViT pretrained on PlantCLEF2022 achieved strong closed-set accuracy across four plant disease datasets and could also recognize disease classes unseen during training (Meng et al., 2023). For detection, transformerbased detectors following DETR (Carion et al., 2020) remove hand-designed components such as anchor boxes and non-maximum suppression. M2DETR (Tan et al., 2025) adapted this design to pest detection with a multiscale denoising encoder, outperforming both CNN-based and transformer-based detectors on the IP102 pest dataset (Wu et al., 2019). Despite these architectural advances, both CNN and transformer-based approaches formulate crop health diagnosis as a fixed recognition or detection problem: their output is a class label, a bounding box, or a segmented region, with no explanation of the underlying visual evidence and no support for follow-up questions. This limitation motivates the use of vision-language models, reviewed in the next subsection.

## 2.2. Vision foundation models in agriculture

Foundation models are large neural networks trained on broad data at scale that can be adapted to a wide range of downstream tasks (Li et al., 2024). Their strength comes from the scale of pretraining. By learning from vast and diverse data, typically with self-supervised objectives that require no manual labels, they acquire general-purpose representations and knowledge that transfer across tasks and domains. As a result, a single pretrained model can be adapted to new tasks with far less labeled data than training from scratch would require, and in many cases it can perform tasks it was never explicitly trained for. Large language models (LLMs) illustrate this most clearly. GPT-3, with 175 billion parameters, performed a variety of language tasks from only a few examples given in its prompt, without taskspecific fine-tuning (Brown et al., 2020). Parameter-eficient methods such as LoRA (Hu et al., 2021) have further lowered the cost of adaptation by updating only a small fraction of a model’s weights.

The same paradigm has been extended to computer vision. CLIP (Radford et al., 2021) learned to align images and text by training on 400 million image-caption pairs collected from the web, enabling it to classify images into categories described in natural language without task-specific training. The Segment Anything Model (SAM) (Kirillov et al., 2023) was trained on more than one billion masks across 11 million images and can segment objects in images it has never seen, and DINOv2 (Oquab et al., 2023) showed that self-supervised pretraining alone can produce visual features that perform well across many downstream tasks. Vision-language models (VLMs) combine these advances with LLMs by connecting a pretrained visual encoder to a language model through a lightweight projector. Models such as BLIP-2 (Li et al., 2023), LLaVA (Liu et al., 2023), and Qwen-VL (Qwen Team, 2025) can answer free-form questions about images, describe visual content, and reason about what they see in natural language, which makes them the most direct foundation-model counterpart to an interactive diagnostic assistant.

These capabilities have attracted growing interest in agriculture, where labeled data is costly to collect and diagnostic questions go beyond fixed categories. Vision foundation models have been applied to agricultural image analysis tasks such as segmentation and phenotyping (Li et al., 2024), while most work on crop health diagnosis has focused on adapting VLMs. AgroGPT (Awais et al., 2025) used domainspecific instruction tuning to enable a general-purpose multimodal model to answer fine-grained questions about crops and diseases, and Agri-LLaVA (Wang et al., 2024) introduced knowledge-infused training to build a multimodal assistant covering agricultural pests and diseases. Other studies have specialized VLMs for individual crops, including a BLIP-2-based model for tomato leaf disease diagnosis (Kaur et al., 2025) and BananaVLM (Jena and Arjunan, 2026), built on LLaVA-1.5-7B, for banana diseases, while LLaVA-PlantDiag (Sharma et al., 2024) explored conversational plant pathology diagnosis. A common finding is that general-purpose VLMs perform poorly on agricultural tasks without adaptation but improve substantially after domain fine-tuning. On the CDDM benchmark, for instance, finetuning raised the disease classification accuracy of LLaVA-1.5-7B from 5.9% to 91.8% (Liu et al., 2024b). Benchmarks such as CDDM, AgMMU (Gauba et al., 2026), IP-VQA (Jin et al., 2025), and PlantExpertVQA (Sakib et al., 2026) now support evaluation beyond fixed labels, covering species identification, symptom description, causal reasoning, and pest management. However, this body of work has focused primarily on adaptation and diagnostic quality, typically assuming access to server-class GPUs. How these models can be deployed eficiently on edge hardware, where agricultural diagnosis is ultimately needed, has received far less attention, as reviewed in the next subsection.

## 2.3. Edge AI for agricultural applications

Edge AI refers to running inference directly on devices located near where data is collected, rather than sending data to remote servers (El Jarroudi et al., 2024). In agriculture, this approach addresses several practical constraints: internet connectivity in rural areas is often limited or unreliable, field decisions benefit from low-latency responses, and farm data can remain under the grower’s control. Embedded platforms such as the NVIDIA Jetson family and the Raspberry Pi, along with smartphones, have therefore become common targets for agricultural vision systems. Most deployments to date have used CNN-based models, often made practical through lightweight architectures such as MobileNets (Howard et al., 2017) and model compression (Cheng et al., 2018). For example, YOLO-based detection of aphid-damaged cotton leaves has been run in real time on an NVIDIA Jetson Xavier NX (Zhang et al., 2024), and a portable plant disease detection system has been deployed on a Jetson Nano (Yousif et al., 2025). Adhikari et al. (2026) evaluated YOLO-based deer detection across multiple edge devices, including the Raspberry Pi and Jetson AGX Xavier, and UAV-based systems have integrated embedded hardware for on-site pest detection to guide precision spraying (Chen et al., 2021). These studies show that edge deployment of conventional vision models is mature, but the deployed models produce fixed outputs such as class labels and bounding boxes.

Deploying VLMs on edge hardware is considerably more challenging. A VLM must encode the image with a vision encoder and then generate its answer token by token with a language model containing billions of parameters, so its memory and latency requirements far exceed those of a CNN classifier or detector. Recent work has begun to address this. MobileVLM (Chu et al., 2023) designed compact VLMs for mobile devices and benchmarked them alongside LLaVA-1.5 on a smartphone and an NVIDIA Jetson AGX Orin, with the language models quantized to 4- bit. Shin et al. (Shin et al., 2026) evaluated component-wise quantization of several small VLMs, including Qwen3-VL-2B, on Jetson Orin NX and AGX devices, and found that 4-bit quantization greatly reduced memory use but did not necessarily speed up generation, with eficiency depending strongly on the hardware platform. On the software side, post-training quantization methods such as GPTQ (Frantar et al., 2022), AWQ (Lin et al., 2024), and SmoothQuant (Xiao et al., 2023), together with serving engines such as vLLM (Kwon et al., 2023), have made it practical to compress and serve these models without retraining. In agriculture, Kaur et al. (Kaur et al., 2025) developed a BLIP-2-based tomato disease model targeting a portable greenhouse monitoring device, one of the few agricultural VLM studies to consider deployment hardware.

These studies leave an important gap. Edge evaluations of VLMs have focused on general-purpose models and benchmarks rather than domain-adapted agricultural models, while agricultural VLM studies have focused on diagnostic quality rather than deployment eficiency. As a result, the combined efect of post-training quantization on the answer quality, memory footprint, and latency of agricultural VLMs running on embedded hardware remains insuficiently characterized, and it is unclear which models and quantization configurations ofer the best trade-of for field use. This work addresses that gap by building and evaluating a complete pipeline, from domain adaptation through quantization to ofline conversational deployment, on an NVIDIA Jetson AGX Orin.

## 3. Methods

This section first describes the datasets used for training and evaluation (Section 3.1). It then presents the PlantVLM-Edge framework and each of its components (Section 3.2), and finally defines the protocol and metrics used to evaluate answer quality and on-device eficiency (Section 3.3).

## 3.1. Datasets

We derive our datasets from two publicly available agri cultural visual question answering (VQA) datasets: Plant-ExpertVQA (Sakib et al., 2026) for plant disease diagnosis and IP-VQA (Jin et al., 2025) for insect pest identification. Because PlantExpertVQA is too large to train on in full with our computational resources, we draw a stratified subset of it. IP-VQA is small enough to use in full after cleaning, and we group its questions into seven categories for evaluation. Figure 1 shows representative image-question-answer examples from both datasets, illustrating the range of questions they cover, from crop and pest identification to symptom description, causal reasoning, and pest management.

Question categories in the PlantExpertVQA evaluation set, where � is the number of question-answer pairs in each category.
<table><tr><td>Question Category</td><td>n</td></tr><tr><td>Existence &amp; sanity check</td><td>72</td></tr><tr><td>Plant species identification</td><td>2,147</td></tr><tr><td>General health assessment</td><td>1,337</td></tr><tr><td>Visual attribute grounding</td><td>716</td></tr><tr><td>Detailed verification</td><td>1,689</td></tr><tr><td>Specific disease identification</td><td>3,972</td></tr><tr><td>Comprehensive description</td><td>1,767</td></tr><tr><td>Causal reasoning</td><td>2,180</td></tr><tr><td>Counterfactual reasoning</td><td>1,202</td></tr></table>

PlantExpertVQA (Sakib et al., 2026) is a large-scale plant disease VQA dataset comprising 765,186 expert-verified question-answer pairs over 150,841 images. It spans 38 crop species and 89 disease conditions, and its questions are organized into nine categories across three levels of cognitive complexity. The dataset is released with an image-disjoint 70/10/20 split into training, validation, and test partitions.

Training on the full corpus would have required substantially more computational time and resources, so we constructed a stratified training subset from the authors released training partition. Sampling was performed at the image level, so that all question-answer pairs belonging to a selected image were retained together and the splits remained image-disjoint. Strata were defined by the combination of crop species, disease condition, and source dataset, and images were allocated to strata in proportion to their size using the largest-remainder method. We also cropped the static black borders present in some images so that the model trained only on relevant visual content. The resulting training subset contains 63,546 question-answer pairs over 12,564 images, approximately 8% of the full dataset.

For evaluation, we used a stratified subset of 3,000 images from the released test partition, comprising 15,082 question-answer pairs. A separate set of 1,500 questionanswer pairs from the validation partition was used for validation during training. Table 1 lists the number of evaluation pairs in each question category.

Table 2  
![](images/2f319c6d3b71e3b9286c59ae2557f7c5839a6d33946850bcec6a1548c3859c2d.jpg)  
Figure 1: Representative image-question-answer pairs from the PlantExpertVQA and IP-VQA datasets.

Question categories in the IP-VQA evaluation set. Values in parentheses are the number of question-answer pairs in each category.
<table><tr><td>Category</td><td>Question focus</td></tr><tr><td>Pest identification</td><td>Pest identity and type (4354)</td></tr><tr><td>Pest feeding and damage</td><td>Feeding, damage, and impact (4135)</td></tr><tr><td>Optimal timing</td><td>Timing and season of control (345)</td></tr><tr><td>Pest monitoring</td><td>Monitoring and detection (1121)</td></tr><tr><td>Pest control methods</td><td>Control, treatment, and pre- vention (3234)</td></tr><tr><td>Visual/ characteristics</td><td>Appearance and visual charac- teristics (2199)</td></tr><tr><td>Others</td><td>Unmatched question types (3562)</td></tr></table>

IP-VQA (Jin et al., 2025) is an insect pest VQA dataset covering 102 pest species, with approximately five questionanswer pairs per image. Its questions address pest characteristics, pest categories, crop damage, and pest management practices. We removed pest classes with fewer than 10 images, to exclude severely underrepresented classes, and then removed corrupted images. This left 18,950 images and 94,785 question-answer pairs across 93 pest categories, which we divided into image-disjoint training (70%), validation (10%), and test (20%) sets. For category-level analysis, we grouped questions into seven evaluation categories by keyword matching on the question text; questions that matched no pattern were grouped as “Others”. Table 2 summarizes these categories.

## 3.2. PlantVLM-Edge framework

Figure 2 provides an overview of PlantVLM-Edge. The framework takes a general-purpose pretrained VLM and prepares it for ofline crop health diagnosis in three stages. In the domain adaptation stage, we fine-tuned the VLMs with LoRA on agricultural question-answer data. In the compression stage, the merged model is quantized after training into lower-precision formats that reduce its memory footprint and accelerate text generation. In the deployment stage, the quantized model is served with vLLM on an NVIDIA Jetson AGX Orin and accessed through a browser-based chat interface, allowing users on the same local network to query it without an internet connection. The framework is backbone-agnostic: any VLM supported by the fine-tuning, quantization, and serving toolchains can be used.

## 3.2.1. Vision-language-model backbones

We instantiate PlantVLM-Edge with two open-weight, instruction-tuned VLMs available on the Hugging Face Hub: Qwen3-VL-2B Instruct (Qwen Team, 2025) and LLaVA-1.5-7B (Liu et al., 2024a). Both follow the standard VLM design of a vision encoder, a lightweight projector, and a language backbone, but they difer in architecture and pretraining. In particular, Qwen3-VL processes images at dynamic resolution, whereas LLaVA-1.5 uses a fixed-size CLIP input, so the two produce diferent numbers of visual tokens per image.

The two models were selected for three reasons. First, both are open-weight and reproducible, yet difer in architecture. Second, both are supported end-to-end by the llmcompressor quantization toolkit (Red Hat AI and vLLM Project, 2024) and the vLLM serving engine (Kwon et al., 2023), which makes the full pipeline feasible. Third, they span the range of model sizes that can be deployed on our target device: Qwen3-VL-2B has approximately 2 billion parameters (a 16-bit weight footprint of 3.96 GiB), whereas LLaVA-1.5-7B has approximately 7 billion parameters (13.16 GiB). The two models therefore represent realistic deployment candidates at diferent points in the size range that fits on the target device, rather than a controlled comparison of model size: because they also difer in architecture and pretraining, our comparison addresses whether a compact model is suficient for deployment, not how model size alone afects performance.

![](images/8d356add232be2f571a2489519bdae9f186a6afa4b4916eddb00021cee2eb90b.jpg)  
Figure 2: Overview of the PlantVLM-Edge framework. A pretrained vision-language model is adapted to plant-disease and insect-pest question answering with LoRA, with the vision encoder and projector kept frozen. The fine-tuned model is then quantized into four configurations and served with vLLM on an NVIDIA Jetson AGX Orin, where its weight footprint and latency are measured. Growers interact with the deployed model ofline through a browser-based chat interface.  
Table 3

## 3.2.2. Domain adaptation

The first stage of PlantVLM-Edge adapts the pretrained backbone to agricultural diagnosis. To eficiently specialize the models while preserving their pretrained knowledge, both models were adapted to each domain using Low-Rank Adaptation (LoRA) (Hu et al., 2021) applied to the language backbone, while the vision encoder, multimodal projector, token embeddings, and output head remained frozen. Specifically, adapters were inserted into seven projection layers: query, key, value, and output in the attention modules, and gate, up, and down in the MLP modules.

The same optimization configuration (Table 3) was used for all four fine-tuning runs (two models × two datasets), and each model used its own native chat template. Furthermore, we trained for at most three epochs with early stopping on validation loss (patience of three evaluations) and kept the checkpoint with the lowest validation loss. All runs used a single NVIDIA A100-SXM4-80GB GPU. After training, the LoRA adapters were merged into the language backbone to produce a single fine-tuned checkpoint for quantization (Hu et al., 2021).

For image processing, images were kept in their original stored form and processed online by each model’s image processor. For Qwen3-VL, dynamic-resolution processing was configured to produce between 128 and 256 visual tokens per image. In contrast, LLaVA-1.5 used its standard CLIP image processor, which resizes the shortest image edge to 336 pixels and applies a 336×336 center crop during both training and inference, producing 576 visual tokens.

Fine-tuning configuration. Optimization settings are shared across both models and all fine-tuning runs; model-specific settings are listed separately.
<table><tr><td></td><td>Qwen3-VL-2B</td><td>LLaVA-1.5-7B</td></tr><tr><td>LoRA</td><td></td><td></td></tr><tr><td>Rank r</td><td>64</td><td>64</td></tr><tr><td>α</td><td>128</td><td>128</td></tr><tr><td>Dropout</td><td>0.05</td><td>0.05</td></tr><tr><td>Trainable params</td><td>69.73M</td><td>159.91M</td></tr><tr><td>% of model</td><td>3.17%</td><td>2.21%</td></tr><tr><td>Precision &amp; tokens</td><td></td><td></td></tr><tr><td>Compute precision</td><td>BF16</td><td>FP16</td></tr><tr><td>Max sequence length</td><td>1,024</td><td>1,024</td></tr><tr><td>Visual tokens/image</td><td>128-256</td><td>576</td></tr><tr><td>Optimisation (shared)</td><td></td><td></td></tr><tr><td>Optimiser</td><td>AdamW (β1=0.9, β2=0.999, e=10−8)</td><td></td></tr><tr><td>Learning rate</td><td>1 × 10−4 (cosine)</td><td></td></tr><tr><td>Effective batch</td><td>16 (4×4 accum.)</td><td></td></tr><tr><td>Grad. checkpointing</td><td>Enabled</td><td></td></tr><tr><td>Max epochs</td><td>3 (early stop, patience 3)</td><td></td></tr><tr><td>GPU</td><td>One NVIDIA A100 80GB</td><td></td></tr></table>

## 3.2.3. Post-training quantization

The second stage compresses the adapted model for edge deployment. After the LoRA adapters were merged, each fine-tuned model was quantized with llmcompressor (Red Hat AI and vLLM Project, 2024) into four configurations spanning three techniques. Each configuration is named by the precision of its weights and activations: W8A16, for example, denotes 8-bit weights and 16-bit activations, and the sufix gives the quantization method. The configurations therefore difer along three dimensions: weight precision (8 or 4 bits), whether activations are also quantized, and whether calibration data are required. In every configuration, quantization was restricted to the linear layers of the language backbone, which hold most of the model’s parameters and must be read from memory at every decoding step (Lin et al., 2024). The vision encoder, multimodal projector, token embeddings, output head, and normalization weights were kept at 16-bit precision (BF16 for Qwen3-VL and FP16 for LLaVA), so that the visual features passed to the language backbone were identical across configurations.

Round-to-Nearest (RTN) RTN is the simplest form of post-training quantization (Nagel et al., 2021). For each group of weights, it computes a scale from the largest absolute weight in the group, divides every weight by this scale, and rounds the result to the nearest integer that can be represented at the target bit-width. Because the scales are derived from the weights alone, RTN requires no calibration data: it never runs example inputs through the model to decide how to quantize. We apply RTN at two bit-widths. W8A16-RTN uses 8-bit weights with one scale per output channel, and W4A16-RTN uses 4-bit weights with one scale per group of 128 weights; smaller groups give finer scales and therefore smaller rounding errors, which matters more at 4 bits. As the fastest and simplest method, RTN serves as the baseline against which the calibrated methods are compared.

Activation-Aware Weight Quantization (AWQ) At 4 bits, rounding errors become large enough to afect the model’s outputs. AWQ (Lin et al., 2024) reduces this efect by recognizing that a small fraction of weight channels matters far more than the rest, namely those that multiply inputs with large activation magnitudes. It runs a small calibration set through the model to identify these salient channels and scales them up before rounding, so that they are quantized with less relative error. Like RTN, AWQ changes only how weights are stored, so the quantized model runs with the same 4-bit inference kernels and memory footprint. Our W4A16-AWQ configuration uses 4-bit weights with a group size of 128, calibrated on 128 question-answer examples drawn from 128 distinct training images.

SmoothQuant The weight-only methods above reduce memory but still perform all computation in 16 bits. SmoothQuant (Xiao et al., 2023) also quantizes the activations, allowing matrix multiplications to run with 8-bit integers. Activations are harder to quantize than weights because a few channels contain large outliers, which would force a coarse scale onto all other values. SmoothQuant therefore first divides each activation channel by a smoothing factor and multiplies the corresponding weights by the same factor, which leaves the layer output unchanged but shifts part of the quantization dificulty from the activations to the weights. With a smoothing strength of 0.5, the dificulty is split evenly between the two. Both are then quantized to 8 bits, using dynamic per-token scales for the activations. W8A8-SmoothQuant was calibrated with the same 128-example protocol as AWQ.

Quantization configurations and the resulting weight footprint (GiB; 1 GiB = 1.07 GB). All configurations quantize only the linear layers of the language backbone; the vision pathway remains at 16-bit precision.
<table><tr><td colspan="4"></td><td colspan="2">Size (GiB)</td></tr><tr><td>Config.</td><td>Weights</td><td>Act.</td><td>Calib.</td><td>Qwen</td><td>LLaVA</td></tr><tr><td>Fine-tuned</td><td>16</td><td>16</td><td>=</td><td>3.96</td><td>13.16</td></tr><tr><td>W8A16-RTN</td><td>8 (per-ch.)</td><td>16</td><td>None</td><td>2.65</td><td>7.13</td></tr><tr><td>W8A8-SQ</td><td>8 (per-ch.)</td><td>8</td><td>128 ex.</td><td>2.65</td><td>7.13</td></tr><tr><td>W4A16-RTN</td><td>4 (g128)</td><td>16</td><td>None</td><td>2.01</td><td>4.20</td></tr><tr><td>W4A16-AWQ</td><td>4 (g128)</td><td>16</td><td>128 ex.</td><td>2.01</td><td>4.20</td></tr></table>

Weights and Act.: bit-width of weights and activations (8-bit activations use dynamic per-token scales); SQ: SmoothQuant; Calib.: calibration data; per-ch.: one scale per output channel; g128: one scale per group of 128 weights; 128 ex.: 128 calibration examples. Qwen: Qwen3-VL-2B (BF16); LLaVA: LLaVA-1.5-7B (FP16).

Table 4 summarizes the four configurations and the resulting weight footprint of each model. Quantization reduced the footprint substantially, and configurations at the same weight precision have identical sizes regardless of method, because the calibrated methods change which values are stored rather than how many bits are used. The relative reduction is larger at 4 bits and for the larger model, because the components kept at 16-bit precision make up a larger share of the compact model.

## 3.2.4. Edge deployment and serving

The final stage serves the quantized model on the edge device and makes it accessible to users. Our target device is an NVIDIA Jetson AGX Orin Developer Kit, with a 12- core ARM Cortex-A78AE CPU at 2.2 GHz and 64 GB of memory shared between the CPU and GPU. A device of this class is the intended field platform: it requires no network connection to answer queries and keeps all data on the device. Because the memory is shared, it must hold the model weights, the key-value (KV) cache, and the serving runtime at the same time; reducing the space taken by the weights is precisely what quantization is intended to achieve.

To support real-time inference, each quantized checkpoint is served using vLLM (Kwon et al., 2023), which provides eficient KV-cache management and native support for the quantized formats generated by llmcompressor. Specifically, vLLM runs inside a container built for the Jetson’s CUDA 12.6 Tegra runtime. The model checkpoint is copied to the device, mounted within the container, and exposed through an OpenAI-compatible HTTP endpoint.

To provide a user-friendly interface, Open WebUI (Baek, 2025) runs on the same device and communicates directly with the local vLLM server. Rather than interacting with the HTTP endpoint itself, users access the system through a browser-based chat interface that supports conversation history and voice input. As a result, anyone connected to the same local network can use the system through a standard web browser without requiring internet access. This fully ofline architecture makes PlantVLM-Edge practical for field deployment and accessible to growers with little or no technical expertise.

## 3.3. Evaluation

We evaluate PlantVLM-Edge along two dimensions: the quality of its answers, measured against the reference answers of each dataset (Section 3.3.1), and its eficiency on the edge device, measured on the Jetson AGX Orin (Section 3.3.2).

## 3.3.1. Answer quality

Because the models generate free-form answers rather than selecting from fixed labels, no single metric fully captures answer quality. We therefore compare each generated answer with the dataset’s single reference answer using five complementary metrics, ordered from the strictest to the most lenient: normalized exact match (EM), which requires the complete answer to match; token-level F1, which measures word overlap regardless of order; ROUGE-L F1, which rewards shared words that appear in the same order; corpus BLEU-4, which measures the overlap of short word sequences; and BERTScore F1, which measures semantic similarity even when the wording difers. These metrics are widely used to evaluate generated answers in agricultural VQA and related tasks (Sakib et al., 2026; Chen et al., 2019; Wang et al., 2020). We use ROUGE-L as the primary metric throughout the paper, because it gives credit for correct answers phrased diferently from the reference while remaining sensitive to word order, and because it is also reported by the authors of PlantExpertVQA (Sakib et al., 2026), which allows a comparison with their results.

The same scoring procedure was applied to both models on both datasets. Only the generated answer was scored: prompt tokens and special tokens were excluded, leading and trailing whitespace was removed, and neither the question nor the image was used by the scoring functions. For EM, token F1, and ROUGE-L, predictions and references were first normalized by lowercasing, removing ASCII punctuation, removing the articles a, an, and the, and collapsing whitespace. BLEU and BERTScore were computed on the unnormalized text with their own tokenizers. All metrics are reported on a 0-100 scale. BLEU is computed once over the whole evaluation set, whereas the other four metrics are computed for each question and averaged, so that every question contributes equally.

Normalized exact match (EM) EM is the strictest metric: an answer scores only if it is identical to the reference after normalization (Rajpurkar et al., 2016). Let $h _ { i }$ and $r _ { i }$ denote the predicted and reference answers for question $i ,$ and let $\mathcal { N } ( \cdot )$ denote the normalization function. For � evaluated questions,

$$
\mathrm { E M } = \frac { 1 0 0 } { N } \sum _ { i = 1 } ^ { N } \mathbf { 1 } \{ \mathcal { N } ( h _ { i } ) = \mathcal { N } ( r _ { i } ) \} .\tag{1}
$$

Because any diference in wording makes an answer count as wrong, EM is much stricter than the classification accuracy or consensus-based VQA accuracy reported in some studies, and its values should not be compared with them.

Token-level F1 Token F1 relaxes EM by giving partial credit for shared words, regardless of their order (Rajpurkar et al., 2016). Normalized answers were split on whitespace and matched as multisets, so that repeated words are counted. Let $H _ { i }$ and $R _ { i }$ be the token sequences of the prediction and the reference, and $c _ { H } ( t )$ and $c _ { R } ( t )$ the frequencies of token � in each. The overlap $o _ { i } ,$ precision $P _ { i } ,$ recall $R _ { i } ^ { \mathrm { t o k } }$ and F1 are

$$
o _ { i } = \sum _ { t } \operatorname* { m i n } \{ c _ { H } ( t ) , c _ { R } ( t ) \} , \quad P _ { i } = \frac { o _ { i } } { | H _ { i } | } ,\tag{2}
$$

$$
R _ { i } ^ { \mathrm { t o k } } = \frac { o _ { i } } { | R _ { i } | } , \quad F _ { i } ^ { \mathrm { t o k } } = \frac { 2 P _ { i } R _ { i } ^ { \mathrm { t o k } } } { P _ { i } + R _ { i } ^ { \mathrm { t o k } } } ,\tag{3}
$$

and we report $\begin{array} { r } { 1 0 0 N ^ { - 1 } \sum _ { i } F _ { i } ^ { \mathrm { t o k } } } \end{array}$ . An answer with no shared words scores zero; if both normalized answers are empty, the score is one, and if only one is empty, it is zero.

ROUGE-L F1 ROUGE-L adds sensitivity to word order (Lin, 2004). It is based on the longest common subsequence of the two answers, that is, the longest sequence of words that appears in both in the same order, though not necessarily adjacently. We used rouge-score version 0.1.2, which, after the normalization described above, tokenizes on alphanumeric characters and applies Porter stemming to words longer than three characters. Writing $\tilde { H } _ { i }$ and ${ \tilde { R } } _ { i }$ for the resulting sequences and $L _ { i }$ for the length of their longest common subsequence, precision is $L _ { i } / | \tilde { H } _ { i } |$ and recall is $L _ { i } / | \tilde { R } _ { i } |$ We report their harmonic mean (ROUGE-L F1), averaged over all questions, rather than ROUGE-L recall or ROUGE-Lsum.

Corpus BLEU-4 BLEU measures the overlap of consecutive word �-grams $( n = 1 , \ldots , 4 )$ between the generated answers and their references, aggregated over the evaluation set, and penalizes outputs that are too short (Papineni et al., 2002). We computed it with SacreBLEU version 2.5.1 (Post, 2018), using one reference per prediction, the 13a tokenizer, case-sensitive scoring, exponential smoothing, and efective order disabled; no answer pairs were excluded. For each order, candidate �-gram matches were clipped to the counts in the corresponding reference and summed over all answer pairs, with no �-grams crossing answer boundaries. The four resulting modified precisions ${ \tilde { p } } _ { n }$ (smoothed when a count is zero) receive equal weights:

$$
{ \mathrm { B L E U - } } 4 = 1 0 0 \cdot { \mathrm { B P } } \cdot \exp \left( { \frac { 1 } { 4 } } \sum _ { n = 1 } ^ { 4 } \log { \tilde { p } _ { n } } \right) ,\tag{4}
$$

where the brevity penalty BP depends on the total candidate length � and reference length �:

$$
\begin{array} { r } { \mathrm { B P } = \left\{ \begin{array} { l l } { 1 , } & { c \ge r , } \\ { \exp ( 1 - r / c ) , } & { 0 < c < r , } \\ { 0 , } & { c = 0 . } \end{array} \right. } \end{array}\tag{5}
$$

BERTScore F1 The metrics above all reward matching words. BERTScore instead measures semantic similarity, giving credit to answers that convey the same meaning in diferent words (Zhang et al., 2019). We computed it with bert-score using a locally cached roberta-large encoder, with representations from layer 17, a batch size of 64, inputs truncated at 512 tokens, and both IDF weighting and baseline rescaling disabled. Let ${ \bf u } _ { a }$ and $\mathbf { v } _ { b }$ be the normalized contextual embeddings of the candidate and reference subword tokens. Precision averages, over the candidate tokens, the highest cosine similarity to any reference token, and recall does the same in reverse:

$$
P _ { i } ^ { B } = \frac { 1 } { m _ { i } } \sum _ { a = 1 } ^ { m _ { i } } \operatorname* { m a x } _ { b } \mathbf { u } _ { a } ^ { \top } \mathbf { v } _ { b } , \quad R _ { i } ^ { B } = \frac { 1 } { n _ { i } } \sum _ { b = 1 } ^ { n _ { i } } \operatorname* { m a x } _ { a } \mathbf { u } _ { a } ^ { \top } \mathbf { v } _ { b } ,\tag{6}
$$

where padding and boundary special tokens are excluded from the averages. We report

$$
\mathrm { B E R T S c o r e - F 1 } = \frac { 1 0 0 } { N } \sum _ { i = 1 } ^ { N } \frac { 2 P _ { i } ^ { B } R _ { i } ^ { B } } { P _ { i } ^ { B } + R _ { i } ^ { B } } .\tag{7}
$$

Because baseline rescaling is disabled, the values are raw similarities, which fall in a narrow, high range even for weakly related text. They should therefore be read as relative comparisons between configurations rather than as percentages of correct answers.

Zero-shot baseline and category-level analysis Two further analyses place these metrics in context. First, to measure the efect of domain adaptation, we evaluated the original Qwen3-VL-2B-Instruct and LLaVA-1.5-7B checkpoints on the same evaluation sets without agricultural fine-tuning or in-context examples. Each model received the image, the question, and the corresponding domain instruction. Second, to identify which types of questions are most afected by quantization, we grouped question-answer pairs by category: nine categories for PlantExpertVQA (Table 1) and seven for IP-VQA (Table 2). Within each category, the per-question metrics were averaged over its questions, and BLEU-4 was recomputed from that category’s answer pairs alone. Overall results were computed from all evaluated questions rather than as an unweighted mean of category scores, and questions sharing an image were treated as separate observations.

## 3.3.2. Deployment eficiency

For the plant disease task, we deployed all five configurations of each model, namely the 16-bit fine-tuned model and its four quantized variants, giving ten served models in total. For the insect pest task, we deployed two configurations as a cross-domain check; their weight footprint and latency closely matched those of the corresponding plant disease configurations, so the remaining insect pest variants were not deployed separately. All reported eficiency figures therefore come from on-device measurements of the plant disease models. All ten configurations were served with identical settings from the same container image, with a maximum model length of 2,048 tokens and a GPU memory utilization of 0.60.

We report the weight footprint, which characterizes memory, together with three latencies that follow the NVIDIA NIM benchmarking definitions (NVIDIA Corporation, 2024): time to first token (TTFT), end-to-end latency (E2E), and inter-token latency (ITL).

Weight footprint We define the weight footprint as the device memory occupied by the loaded model parameters, taken from the log that vLLM reports when model loading completes. It depends only on the model and its quantization format. We deliberately exclude the memory that the engine allocates afterwards, such as the KV cache and the CUDA graph workspace, because these depend on the memory allocation settings and the required concurrency.

Latency All latencies were measured one request at a time, with no other requests running. Each request paired one image with the question “What disease afects this plant?”, drawn from 40 randomly sampled unique images from the plant-disease test set. Every configuration received the same 40 images in the same order, and caching was disabled so that each request began a fresh cycle. Three warm-up requests were run before each test and excluded from the measurements. We capped every answer at exactly � = 64 tokens so that output length was identical across configurations and end-to-end latencies are directly comparable. Writing $t _ { 0 }$ for the time of submission, $t _ { 1 }$ for the arrival of the first generated token, and $t _ { N }$ for the end of the response,

$$
\mathrm { T T F T } = t _ { 1 } - t _ { 0 } , \qquad \mathrm { E 2 E } = t _ { N } - t _ { 0 } , \qquad \mathrm { I T L } = \frac { \mathrm { E 2 E } - \mathrm { T T F T } } { N - 1 } .\tag{8}
$$

As illustrated in Figure 3, TTFT is the delay before the user sees any output: it covers request handling and the prefill stage, in which the image is encoded and the prompt is processed. End-to-end latency (E2E) is the time to the final token, spanning the whole request. Inter-token latency (ITL), also called time per output token, is the average gap between consecutive tokens once generation is under way. The first token marks the end of prefill rather than a decoding step, so ITL averages over the remaining � − 1 tokens, isolating the cost of decoding from the one-time cost of prefill. The three metrics combine as TTFT + (� − 1) ITL, the time to produce an answer of � tokens. We report all three as means over the 40 requests, with standard deviations for TTFT and E2E. The reciprocal of the mean ITL gives the per-user decoding rate in tokens per second, a restatement of ITL rather than a separate measurement.

## 4. Results

We report results following the stages of the PlantVLM-Edge framework. We first examine how domain adaptation changes answer quality relative to the zero-shot backbones (Section 4.1), then how post-training quantization afects answer quality on both datasets (Section 4.2), and then the memory and latency of each configuration measured on the

![](images/ea23b09333eaacea9f3fe7128e46e9790c6b7b2d672fd4ad9aac77f9997c1b17.jpg)  
Figure 3: Illustration of the time to first token (TTFT) and inter-token latency (ITL) metrics, following (NVIDIA Corporation, 2024).

Jetson AGX Orin (Section 4.3). Next, we combine the two to analyze the accuracy-eficiency trade-of and identify recommended configurations (Section 4.4). Finally, we present an example interaction with the deployed system using the recommended configuration (Section 4.5).

## 4.1. Efect of domain adaptation

Table 5 reports answer quality for the zero-shot backbones, the fine-tuned 16-bit models, and their four quantized variants on both datasets. Domain adaptation produces by far the largest change in answer quality in this study. On PlantExpertVQA, fine-tuning raised ROUGE-L from 15.45 to 78.18 for Qwen3-VL-2B and from 16.11 to 78.19 for LLaVA-1.5-7B, and exact match from below 2% to about 34% for both models. On IP-VQA, it raised ROUGE-L from 30.45 to 54.67 for Qwen3-VL-2B and from 26.25 to 54.41 for LLaVA-1.5-7B. The zero-shot models therefore answered in a form that difered substantially from the reference answers, and adaptation was essential for both backbones.

After fine-tuning, the two backbones reached almost identical answer quality: 78.18 against 78.19 ROUGE-L on PlantExpertVQA and 54.67 against 54.41 on IP-VQA, despite LLaVA-1.5-7B having more than three times as many parameters. Absolute scores are lower on IP-VQA than on PlantExpertVQA throughout, at roughly 54 against 78 ROUGE-L, and exact match stays below 5%. This reflects the less templated answer distribution of IP-VQA, in which longer free-form descriptions make exact string agreement rare. BERTScore F1 changed much less with fine-tuning than the other metrics (for example, from 85.03 to 96.56 for Qwen3-VL-2B on PlantExpertVQA). This is largely a property of unrescaled BERTScore, which occupies a narrow, high range even for weakly related text (Section 3.3.1); its values are therefore most informative as relative comparisons between configurations.

## 4.2. Answer quality under quantization

We next compare each quantized configuration with its fine-tuned 16-bit counterpart, which serves as the quality reference. Tables 6 and 7 report category-level ROUGE-L, giving the 16-bit score and the change under each quantized configuration, so that losses can be read directly.

## 4.2.1. PlantExpertVQA

Qwen3-VL-2B Both 8-bit configurations closely matched the fine-tuned model on all five metrics (Table 5): W8A16- RTN and W8A8-SmoothQuant scored 78.04 and 78.11 ROUGE-L against 78.18, and SmoothQuant marginally exceeded the reference on BLEU-4. At 4 bits, a small cost appeared, and the quantization method began to matter. W4A16-AWQ retained 98.6% of full-precision ROUGE-L, against 97.8% for W4A16-RTN, and outperformed RTN on all five metrics. Going from full precision to W4A16-RTN, ROUGE-L fell by 1.74 points and BERTScore F1 by only 0.31, whereas exact match fell by 2.85 points, or 8.5% of its value against 2.2% for ROUGE-L. Exact match is the most sensitive metric because any change in wording makes the whole answer count as wrong, whereas ROUGE-L and BERTScore give credit for partial and semantic overlap.

The category-level results (Table 6) show where this cost falls. At 8 bits, every category stayed within 0.6 ROUGE-L of full precision. At 4 bits, the loss was uneven: under roundto-nearest, it ranged from 0.28 points in Visual Attribute Grounding to 3.67 points in Comprehensive Description, with Plant Species Identification next at 2.36. AWQ recovered most of this loss in Comprehensive Description (from 3.67 to 0.93 points) and Counterfactual Reasoning (from 1.84 to 0.41). However, it trailed round-to-nearest in four of the nine categories, by between 0.03 and 1.58 points, most notably in Visual Attribute Grounding (a loss of 1.86 points against 0.28) and Existence & Sanity Check (1.57 against 0.32). The benefit of AWQ is therefore concentrated in long, explanatory answers, which matter most for a conversational diagnostic system.

LLaVA-1.5-7B Quantization was close to lossless across the entire range for the larger backbone (Table 5). Every configuration stayed within 0.33 ROUGE-L of full precision, BERTScore F1 varied by at most 0.06 points, and exact match by at most 0.37. W8A16-RTN was efectively identical to the fine-tuned model, and at 4 bits round-tonearest scored slightly higher than AWQ (77.99 against 77.86). The category-level results (Table 6) were equally flat: no configuration fell more than 0.73 ROUGE-L below full precision in any category, the largest deviation being a 0.73-point loss in Counterfactual Reasoning under W4A16- AWQ.

Although the two backbones reached the same quality at full precision, they difered in their sensitivity to 4- bit quantization: Qwen3-VL-2B lost 1.07 (AWQ) and 1.74 (RTN) ROUGE-L points, whereas LLaVA-1.5-7B lost only 0.33 and 0.20.

## 4.2.2. IP-VQA

Qwen3-VL-2B The 8-bit configurations again preserved answer quality (Table 5). W8A16-RTN scored 54.70 ROUGE-L against 54.67 at full precision, a negligible diference with identical BERTScore F1 (94.06), and W8A8-SmoothQuant scored 54.40. At 4 bits, the two methods separated more clearly than on PlantExpertVQA.

Table 5  
Answer quality of both backbones on the PlantExpertVQA and the IP-VQA test set. All metrics are reported on a 0-100 scale; higher is better. For each model, the best result among the fine-tuned configurations in each column is in bold.
<table><tr><td></td><td colspan="5">PlantExpertVQA</td><td colspan="5">IP-VQA</td></tr><tr><td>Configuration</td><td>R-L</td><td>F1</td><td>EM</td><td>BERT</td><td>BLEU</td><td>R-L</td><td>F1</td><td>EM</td><td>BERT</td><td>BLEU</td></tr><tr><td>Qwen3-VL-2B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Zero-shot</td><td>15.45</td><td>19.35</td><td>0.29</td><td>85.03</td><td>3.10</td><td>30.45</td><td>33.22</td><td>0.33</td><td>89.35</td><td>5.70</td></tr><tr><td>Fine-tuned (BF16)</td><td>78.18</td><td>79.52</td><td>33.64</td><td>96.56</td><td>65.87</td><td>54.67</td><td>57.13</td><td>4.91</td><td>94.06</td><td>30.03</td></tr><tr><td>W8A16-RTN</td><td>78.04</td><td>79.42</td><td>33.53</td><td>96.55</td><td>65.73</td><td>54.70</td><td>57.18</td><td>4.87</td><td>94.06</td><td>30.05</td></tr><tr><td>W8A8-SmoothQuant</td><td>78.11</td><td>79.46</td><td>33.60</td><td>96.55</td><td>65.90</td><td>54.40</td><td>56.84</td><td>4.57</td><td>94.02</td><td>29.58</td></tr><tr><td>W4A16-RTN</td><td>76.44</td><td>77.82</td><td>30.79</td><td>96.25</td><td>64.04</td><td>52.95</td><td>55.39</td><td>4.18</td><td>93.81</td><td>28.38</td></tr><tr><td>W4A16-AWQ</td><td>77.11</td><td>78.50</td><td>31.95</td><td>96.36</td><td>64.28</td><td>53.91</td><td>56.37</td><td>4.38</td><td>93.94</td><td>29.13</td></tr><tr><td>LLaVA-1.5-7B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Zero-shot</td><td>16.11</td><td>19.54</td><td>1.92</td><td>85.96</td><td>3.60</td><td>26.25</td><td>29.73</td><td>0.31</td><td>89.57</td><td>5.06</td></tr><tr><td>Fine-tuned (FP16)</td><td>78.19</td><td>79.50</td><td>33.73</td><td>96.57</td><td>66.00</td><td>54.41</td><td>56.86</td><td>4.76</td><td>94.03</td><td>29.81</td></tr><tr><td>W8A16-RTN</td><td>78.17</td><td>79.49</td><td>33.72</td><td>96.57</td><td>65.99</td><td>54.43</td><td>56.89</td><td>4.77</td><td>94.03</td><td>29.84</td></tr><tr><td>W8A8-SmoothQuant</td><td>78.11</td><td>79.41</td><td>33.62</td><td>96.56</td><td>65.46</td><td>54.43</td><td>56.78</td><td>4.57</td><td>94.03</td><td>29.19</td></tr><tr><td>W4A16-RTN</td><td>77.99</td><td>79.26</td><td>33.58</td><td>96.53</td><td>65.79</td><td>54.20</td><td>56.67</td><td>4.50</td><td>94.00</td><td>29.47</td></tr><tr><td>W4A16-AWQ</td><td>77.86</td><td>79.22</td><td>33.36</td><td>96.51</td><td>65.74</td><td>54.00</td><td>56.51</td><td>4.36</td><td>93.98</td><td>29.45</td></tr></table>

R-L: ROUGE-L F1; F1: token-level F1; EM: normalized exact match; BERT: BERTScore F1; BLEU: corpus BLEU-4.  
seven categories, most in Pest Identification (−0.69 against −0.31), and SmoothQuant recorded the highest score in four of the seven categories by small margins.

Round-to-nearest lost 1.72 ROUGE-L, similar to its loss on the plant task in absolute terms, whereas AWQ lost only 0.76, retaining 98.6% of full-precision ROUGE-L, and outperformed round-to-nearest on all five metrics. AWQ’s lead over round-to-nearest was therefore wider than on PlantExpertVQA (0.96 against 0.67 points), and it recovered 56% of the round-to-nearest loss on IP-VQA compared with 39% on PlantExpertVQA.

This diference is consistent with the structure of the two datasets. PlantExpertVQA contains many repeated question and answer patterns, whereas IP-VQA has more varied questions and responses. Repetitive, templated answers may be less sensitive to coarse weight rounding, whereas more varied answers make the efect of quantization more visible, which also leaves more room for AWQ to recover. The category-level results (Table 7) show that the 4-bit degradation was broad rather than concentrated: under roundto-nearest, every category lost between 0.9 and 2.4 points, with Pest Control Methods the most afected (−2.43). AWQ recovered part of this loss in six of the seven categories. The exception was Optimal Timing, the smallest category (� = 345), where AWQ lost 1.87 points against 0.98 for round-to-nearest.

LLaVA-1.5-7B Quantization left answer quality essentially unchanged for the larger backbone (Table 5). Across all five configurations, ROUGE-L varied by 0.43 points and BERTScore F1 by 0.05. Both 8-bit configurations scored 54.43 ROUGE-L against 54.41 at full precision, with BERTScore F1 identical at 94.03. At 4 bits, roundto-nearest again scored slightly higher than AWQ (54.20 against 54.00). The category-level results (Table 7) were equally flat. Under 4-bit round-to-nearest, no category fell more than 0.9 ROUGE-L below full precision, compared with losses of 0.9 to 2.4 points for Qwen3-VL-2B on the same dataset. AWQ trailed round-to-nearest in six of the

## 4.2.3. Summary across models and domains

The four model-dataset combinations give a consistent picture. Eight-bit quantization changed aggregate ROUGE-L by at most 0.3 points in every case, regardless of method. Four-bit quantization cost between 0.3% and 3.1% of ROUGE-L, with the larger losses falling on the compact backbone and on the less templated IP-VQA dataset. The choice of 4-bit method mattered only where this loss was material: for Qwen3-VL-2B, AWQ outperformed round-tonearest in both domains and retained 98.6% of full-precision ROUGE-L on each, whereas for LLaVA-1.5-7B the two methods difered by at most 0.2 points, with round-to-nearest slightly ahead.

## 4.3. On-device eficiency

Table 8 reports the weight footprint and the three latency metrics measured on the Jetson AGX Orin for the ten plant disease configurations.

## 4.3.1. Memory and decoding speed

Bit-width determined both the weight footprint and the decoding speed (Table 8, Weights and ITL columns). Configurations at the same bit-width occupied identical memory, and the two 4-bit configurations decoded within 0.1 ms per token of each other. The exception was W8A8- SmoothQuant, which decoded 1.2-1.3 ms per token more slowly than W8A16-RTN, consistent with the added cost of quantizing activations at every decoding step. Relative to 16 bits, the weight footprint fell by 33% (8 bits) and 49% (4 bits) for Qwen3-VL-2B, and by 46% and 68% for LLaVA-1.5-7B. The compact backbone compresses less because its unquantized components, namely the vision encoder,

Table 7  
Table 6  
Category-level ROUGE-L on the PlantExpertVQA test set. For each backbone, the first column gives the 16-bit score and the remaining columns give the change under each quantized configuration.
<table><tr><td></td><td></td><td colspan="5">Qwen3-VL-2B</td><td colspan="5">LLaVA-1.5-7B</td></tr><tr><td>Category</td><td>n</td><td>16-bit</td><td>W8</td><td>SQ</td><td>W4</td><td>AWQ</td><td>16-bit</td><td>W8</td><td>SQ</td><td>W4</td><td>AWQ</td></tr><tr><td>Causal Reasoning</td><td>2,180</td><td>66.02</td><td>-0.26</td><td>-0.58</td><td>-1.41</td><td>-1.44</td><td>65.74</td><td>-0.03</td><td>-0.09</td><td>-0.52</td><td>-0.25</td></tr><tr><td>Comprehensive Descr.</td><td>1,767</td><td>64.74</td><td>-0.21</td><td>+0.41</td><td>-3.67</td><td>-0.93</td><td>63.75</td><td>+0.05</td><td>0.00</td><td>-0.19</td><td>-0.26</td></tr><tr><td>Counterfactual Reas.</td><td>1,202</td><td>74.00</td><td>-0.15</td><td>+0.26</td><td>-1.84</td><td>-0.41</td><td>74.58</td><td>-0.06</td><td>-0.11</td><td>+0.41</td><td>-0.73</td></tr><tr><td>Detailed Verification</td><td>1,689</td><td>79.79</td><td>-0.14</td><td>+0.08</td><td>-0.65</td><td>-0.79</td><td>80.38</td><td>0.00</td><td>-0.20</td><td>-0.24</td><td>-0.51</td></tr><tr><td>Existence &amp; Sanity</td><td>72</td><td>100.00</td><td>0.00</td><td>0.00</td><td>-0.32</td><td>-1.57</td><td>100.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>General Health</td><td>1,337</td><td>97.18</td><td>-0.03</td><td>-0.07</td><td>-0.69</td><td>-0.30</td><td>97.28</td><td>+0.01</td><td>-0.19</td><td>-0.33</td><td>-0.14</td></tr><tr><td>Plant Species ID</td><td>2,147</td><td>89.38</td><td>-0.05</td><td>-0.23</td><td>-2.36</td><td>-1.80</td><td>89.48</td><td>-0.06</td><td>-0.09</td><td>-0.38</td><td>-0.26</td></tr><tr><td>Specific Disease ID</td><td>3,972</td><td>76.87</td><td>-0.12</td><td>-0.02</td><td>-1.80</td><td>-0.95</td><td>77.09</td><td>-0.02</td><td>-0.04</td><td>0.00</td><td>-0.36</td></tr><tr><td>Visual Attribute Gr.</td><td>716</td><td>87.56</td><td>-0.13</td><td>-0.16</td><td>-0.28</td><td>-1.86</td><td>86.98</td><td>+0.01</td><td>+0.19</td><td>-0.58</td><td>-0.07</td></tr></table>

16-bit: ROUGE-L of the fine-tuned model (BF16 for Qwen3-VL-2B, FP16 for LLaVA-1.5-7B). W8: W8A16-RTN; SQ: W8A8-SmoothQuant; W4: W4A16-RTN; AWQ: W4A16-AWQ. Quantized columns give the change in ROUGE-L points relative to 16-bit; losses of 1.0 point or more are in bold.

Category-level ROUGE-L on the IP-VQA test set. For each backbone, the first column gives the 16-bit score and the remaining columns give the change under each quantized configuration.
<table><tr><td></td><td></td><td colspan="5">Qwen3-VL-2B</td><td colspan="5">LLaVA-1.5-7B</td></tr><tr><td>Category</td><td>n</td><td>16-bit</td><td>W8</td><td>SQ</td><td>W4</td><td>AWQ</td><td>16-bit</td><td>W8</td><td>SQ</td><td>W4</td><td>AWQ</td></tr><tr><td>Optimal Timing</td><td>345</td><td>53.40</td><td>+0.17</td><td>+0.29</td><td>-0.98</td><td>-1.87</td><td>52.64</td><td>-0.22</td><td>+0.79</td><td>-0.85</td><td>-0.50</td></tr><tr><td>Other</td><td>3,562</td><td>58.34</td><td>+0.11</td><td>-0.12</td><td>-1.97</td><td>-0.62</td><td>57.94</td><td>-0.03</td><td>+0.27</td><td>-0.17</td><td>-0.25</td></tr><tr><td>Pest Control Methods</td><td>3,234</td><td>47.40</td><td>+0.12</td><td>-0.15</td><td>-2.43</td><td>-0.43</td><td>47.27</td><td>+0.10</td><td>+0.07</td><td>-0.01</td><td>-0.18</td></tr><tr><td>Pest Feeding &amp; Damage</td><td>4,135</td><td>48.55</td><td>-0.03</td><td>-0.38</td><td>-1.65</td><td>-0.97</td><td>48.57</td><td>-0.03</td><td>-0.01</td><td>一 -0.29</td><td>-0.43</td></tr><tr><td>Pest Identification</td><td>4,354</td><td>62.77</td><td>+0.07</td><td>-0.32</td><td>-1.34</td><td>-0.80</td><td>62.54</td><td>+0.08</td><td>-0.31</td><td>-0.31</td><td>-0.69</td></tr><tr><td>Pest Monitoring</td><td>1,121</td><td>55.80</td><td>+0.06</td><td>-0.04</td><td>-1.23</td><td>-0.99</td><td>55.36</td><td>+0.05</td><td>+0.09</td><td>-0.22</td><td>-0.48</td></tr><tr><td>Visual/ Characteristics</td><td>2,199</td><td>54.47</td><td>-0.21</td><td>-0.51</td><td>-1.46</td><td>-0.65</td><td>53.84</td><td>+0.06</td><td>+0.11</td><td>-0.06</td><td>-0.33</td></tr></table>

16-bit: ROUGE-L of the fine-tuned model (BF16 for Qwen3-VL-2B, FP16 for LLaVA-1.5-7B). W8: W8A16-RTN; SQ: W8A8-SmoothQuant; W4: W4A16-RTN; AWQ: W4A16-AWQ. Quantized columns give the change in ROUGE-L points relative to 16-bit; losses of 1.0 point o more are in bold.

## Table 8

On-device eficiency on the Jetson AGX Orin for the plant disease models. Latencies are means over 40 requests of 64 tokens each. Standard deviations of TTFT and E2E are in parentheses. BF16 and FP16 denote the fine-tuned 16-bit models; SQ: SmoothQuant. The lowest value for each model is in bold.

<table><tr><td>Config.</td><td>Weights (GiB)</td><td>TTFT (ms)</td><td>E2E</td><td>ITL (ms) (ms/tok)</td></tr><tr><td>Qwen3-VL-2B</td><td></td><td></td><td></td><td></td></tr><tr><td>BF16</td><td></td><td>3.96 130.4 (30)</td><td>1535.6 (31)</td><td>22.3</td></tr><tr><td>W8A16-RTN</td><td></td><td>2.65 123.3 (31)</td><td>1042.2 (32)</td><td>14.6</td></tr><tr><td>W8A8-SQ</td><td></td><td>2.65 116.1 (29)</td><td>1117.5 (30)</td><td>15.9</td></tr><tr><td>W4A16-RTN</td><td>2.01</td><td>124.1 (32)</td><td>782.5 (33)</td><td>10.5</td></tr><tr><td>W4A16-AWQ</td><td>2.01</td><td>123.4 (31)</td><td>777.7 (32)</td><td>10.4</td></tr><tr><td>LLaVA-1.5-7B</td><td></td><td></td><td></td><td></td></tr><tr><td>FP16</td><td></td><td>13.16 463.8 (14)</td><td>5451.8 (14)</td><td>79.2</td></tr><tr><td>W8A16-RTN</td><td>7.13</td><td>3419.2 (14)</td><td>3235.9 (14)</td><td>44.7</td></tr><tr><td>W8A8-SQ</td><td>7.13</td><td>3248.0 (14)</td><td>3136.7 (14)</td><td>45.9</td></tr><tr><td>W4A16-RTN</td><td>4.20</td><td>374.8 (14)</td><td>1983.5 (14)</td><td>25.5</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>W4A16-AWQ</td><td>4.20</td><td>359.0(7)</td><td>1968.5(7)</td><td>25.5</td></tr></table>

embeddings, and output head, make up a larger share of its parameters.

Decoding accelerated accordingly, with inter-token latency falling by 2.14× for Qwen3-VL-2B and 3.11× for LLaVA-1.5-7B. Each generated token requires the languagebackbone weights to be read from memory, so fewer weight bits mean less memory trafic and faster token generation. The decoding speed-up does not exactly track the footprint reduction, because the footprint includes components that play diferent roles during decoding. The 16-bit vision encoder occupies memory but is not used after prefill, which is why Qwen3-VL-2B decoded 2.14× faster even though its footprint shrank by only 1.97×. The 16-bit output head, in contrast, is read at every decoding step and limits the attainable speed-up.

## 4.3.2. Time to first token

Before a model can emit its first token, it must encode the image and process the prompt, a stage known as prefill (Figure 3). Quantization helps less here for two reasons: the vision encoder, which runs entirely during prefill, remains at 16 bits in every configuration, and prefill processes all input tokens in parallel, so it is limited by computation rather than by the memory trafic that dominates decoding. For Qwen3-VL-2B, the four quantized configurations reached their first token in 116-124 ms, against 130 ms at 16 bits (Table 8), diferences smaller than the standard deviation across requests. LLaVA-1.5-7B started much more slowly, at 464 ms, because it processes 576 visual tokens against 128- 256 for Qwen3-VL-2B and prefills them through a larger language backbone. Weight-only quantization reduced its TTFT by 10-23%, whereas W8A8-SmoothQuant reduced it by 46%, to 248 ms. This is the only case in which quantizing activations clearly helped, and it occurred in the model whose prefill is long enough for 8-bit arithmetic to pay of.

## 4.3.3. End-to-end latency

Because every request generated exactly 64 tokens, endto-end latencies in Table 8 are directly comparable across configurations. Quantization reduced the response time from 1.54 to 0.78 s for Qwen3-VL-2B (1.97×) and from 5.45 to 1.97 s for LLaVA-1.5-7B (2.77×), with W4A16-AWQ the fastest configuration for both models. Decoding accounted for more than 80% of the response time in every configuration, so decoding speed, not TTFT, determined the ranking (Figure 3 illustrates the two phases). For example, W8A8- SmoothQuant gave LLaVA-1.5-7B its fastest first token (248 against 359 ms for W4A16-AWQ), but its full response took 3.14 against 1.97 s because it decoded 1.8× more slowly. As decoding becomes faster, TTFT accounts for a larger share of the response: for both models, it rises from 8.5% at 16 bits to 15.9% (Qwen3-VL-2B) and 18.2% (LLaVA-1.5-7B) with W4A16-AWQ.

## 4.3.4. Backbone comparison

The choice of backbone had a larger efect on deployment cost than the choice of quantization (Table 8). Even unquantized, Qwen3-VL-2B occupied less weight memory than the most compressed LLaVA-1.5-7B configuration (3.96 against 4.20 GiB) and was faster than every LLaVA-1.5-7B configuration: it reached its first token in 130 ms against at best 248 ms, decoded at 22.3 against at best 25.5 ms per token, and completed a 64-token answer in 1.54 against at best 1.97 s. At equal precision the gap widened. With both models at W4A16-AWQ, Qwen3-VL-2B used 2.09× less memory, reached its first token 2.9× sooner, decoded 2.45× faster, and completed a 64-token answer 2.5× sooner. Quantization reduced inter-token latency by at most 3.1× within a model, which was not enough for any LLaVA-1.5-7B configuration to match the unquantized Qwen3-VL-2B. Because the two fine-tuned backbones reach the same answer quality (Section 4.1), backbone selection was the dominant eficiency decision in this study and quantization the secondary one.

## 4.4. Accuracy-eficiency trade-of

Figure 4 plots all ten configurations against weight footprint (a) and inter-token latency (b). Answer quality spans only 1.8 ROUGE-L points across the set, whereas memory and decoding speed difer by more than a factor of six. Eficiency, rather than answer quality, therefore separates the configurations.

Choice of backbone Because the two backbones reach equivalent answer quality, the comparison between them rests on eficiency, where Qwen3-VL-2B is the clear choice. At 8 bits, it remains within 0.2 ROUGE-L of every LLaVA-1.5-7B configuration while requiring about 37% less memory, reaching its first token at least 2.0× sooner, and decoding at least 1.6× faster. Quantizing the larger model further does not change this conclusion, because even its 4-bit configurations remain more expensive than the compact model at 8 bits.

Choice of configuration Among the Qwen3-VL-2B configurations, three are dominated. The 16-bit model is outperformed by W8A16-RTN, which preserves its answer quality (99.8% of full-precision ROUGE-L) while using 33% less memory and decoding 1.53× faster, so full precision ofers no practical advantage. W4A16-RTN is outperformed by W4A16-AWQ, which has an identical footprint and decoding speed but scores 0.67 ROUGE-L higher. W8A8- SmoothQuant lies marginally above W8A16-RTN at equal footprint (Figure 4a), but the 0.07-point diference is too small to be meaningful, and W8A16-RTN decodes 1.3 ms per token faster, completes a 64-token answer 75 ms sooner (Figure 4b), and requires no calibration data. SmoothQuant’s one advantage is the lowest TTFT of the five Qwen3-VL-2B configurations, but over a 64-token answer it finishes 75 ms behind W8A16-RTN and 340 ms behind W4A16-AWQ.

Recommended configurations. Two configurations therefore remain. W8A16-RTN is essentially lossless, retaining 99.8% of full-precision ROUGE-L on PlantExpertVQA and matching full precision on IP-VQA, and requires no calibration data. W4A16-AWQ gives up a further 1.2% of ROUGE-L in exchange for a 24% smaller footprint and 29% lower inter-token latency than W8A16-RTN, requiring only 2.01 GiB of weights and completing a 64-token answer in 0.78 s. We therefore recommend Qwen3-VL-2B with W8A16-RTN as the default PlantVLM-Edge configuration, and W4A16-AWQ where memory is the binding constraint or where longer answers make decoding speed the priority.

## 4.5. System demonstration

Figure 5 illustrates PlantVLM-Edge in use with the recommended Qwen3-VL-2B W8A16-RTN configuration. After a user uploads a photo of an infested leaf, the model identifies the problem and describes the visible damage, and answers a follow-up question about management in the same conversation, drawing on the earlier context. The interface runs entirely on the Jetson AGX Orin and is reached from a laptop or smartphone over the local network, with no internet connection. The image is taken from the test set. Open WebUI also supports voice input, which could help users in the field whose hands are occupied. In the current deployment, however, speech recognition is not performed on the device, so the fully ofline operation described here applies to text-based interaction; running speech recognition locally is left for future work.

What is it suffering from ?..  
![](images/3b5df7cbdabde7e2bb6e5d27e76fa148d0c091a2047f07a1646fe36ed2fcaa93.jpg)  
(a) Weight footprint

![](images/45e1b443a6d41c6727460a982d89ee57ddc1c714795e1b6bd9534387ccdb105d.jpg)  
(b) Inter-token latency  
Figure 4: Accuracy-eficiency trade-of for the ten PlantExpertVQA configurations on the Jetson AGX Orin: ROUGE-L against (a) weight footprint and (b) inter-token latency. Lines connect the Pareto-optimal configurations of each model.

![](images/fd7567b135cbe53f8672e3072a5bf69394d0b938dddd73fda2ba84890f296a34.jpg)  
What is it suffering from ?

plantvlm-edge:latest   
This is tomato leaf spot. The leaves show dark, irregular spots and yellowing, likely caused by bacterial infection from Xanthomonas campestris.

like how do i cure it? how severe is it

plantvlm-edge:latest   
This is tomato leaf spot. The disease is moderate severity with dark, irregular spots and yellowing on the leaves. To cure it, remove infected leaves and apply copper-based fungicide as a preventive measure. plantvlm-edge:latest   
This looks like powdery mildew on pea pods. The white powdery coating is visible on the pods and leaves, likely caused by excessive moisture or poor airflow. Remove the infected pods and spray the plants with a fungicide like neem oil to prevent further spread.   
plantvlm-edge:latest   
No. the peas inside are not affected by the powdery mildew.   
The disease is primarily on the outer surfaces of the pods and   
leaves. causing a white powderv coating ↓

![](images/333dd664e33d780dbf069c1ab2222743a9522099ef87c43fd056bf1b248b1346.jpg)

Figure 5: Screenshots of multi-turn conversations with the deployed PlantVLM-Edge system (Qwen3-VL-2B, W8A16-RTN) on the Jetson AGX Orin, accessed ofline through a browser.

## 5. Discussion

The results show that a compact VLM, once adapted to the target domain, can match a model more than three times its size on agricultural question answering, and that posttraining quantization then lowers its deployment cost further at little or no loss in answer quality. This section discusses what these findings imply for building edge-deployable diagnostic systems, translates them into deployment recommendations, and outlines the limitations of the present study.

## 5.1. Domain adaptation and backbone selection

Domain adaptation was the decisive step for answer quality. Fine-tuning raised ROUGE-L by 62.1-62.7 points on PlantExpertVQA and by 24.2-28.2 points on IP-VQA (Table 5), whereas no quantized configuration changed aggregate ROUGE-L by more than 1.8 points. After adaptation, the two backbones were indistinguishable in aggregate answer quality on both datasets, despite LLaVA-1.5-7B having more than three times as many parameters. Our finetuned Qwen3-VL-2B reached 78.18 ROUGE-L on PlantExpertVQA, compared with 61.70 reported by the dataset’s authors for the same model fine-tuned on 500 images (Sakib et al., 2026). The figures are not directly comparable, because the evaluation subsets and training settings difer, but the gap is consistent with the larger training subset used here and supports the authors’ observation that the dataset enables eficient domain adaptation.

For edge deployment, this reorders the usual priorities. Because the adapted backbones reach the same answer quality, the choice between them rests on eficiency, and there the compact model was smaller and faster at every bit-width, decoding 2.4-3.6× faster than LLaVA-1.5-7B at equal precision (Table 8; Figure 4). The larger model’s better tolerance of 4-bit quantization did not close this gap: even unquantized, Qwen3-VL-2B was smaller and faster than every LLaVA-1.5-7B configuration. Backbone selection was therefore the dominant eficiency decision in this study, and quantization the secondary one. Because the two backbones also difer in architecture and pretraining, we interpret this as evidence that a compact model is suficient for these tasks, rather than as a general efect of model size. A controlled comparison, such as models of diferent sizes from the same family, would be needed to separate the efect of model size from that of architecture and pretraining.

## 5.2. Bit-width versus quantization method

Bit-width and quantization method played distinct roles. Bit-width determined deployment cost: configurations at the same precision had identical weight footprints (Table 4), and the two 4-bit configurations decoded at the same speed (Table 8). The quantization method mattered mainly for answer quality, and only at 4 bits on the compact backbone, where AWQ recovered 39% (PlantExpertVQA) to 56% (IP-VQA) of the loss incurred by round-to-nearest. At 8 bits, every method was efectively lossless, and the data-free

W8A16-RTN matched the calibrated SmoothQuant in answer quality. Activation quantization gave no decoding benefit under single-user serving and helped only the computebound prefill of the larger model.

The comparison between datasets carries a methodological implication. The larger relative loss under 4-bit roundto-nearest on IP-VQA, and the wider gap between AWQ and round-to-nearest there (Tables 6 and 7), suggest that quantization studies on highly templated benchmarks may understate the cost of low-bit deployment on more varied question distributions. Since questions from growers in the field are likely to be more varied than either benchmark, evaluations of compressed agricultural models should include datasets with diverse, free-form answers.

## 5.3. The role of the serving stack

The latency benefits of quantization depend on the software that executes the quantized model, not only on the bit-width. In our vLLM deployment, 4-bit weights reduced inter-token latency by 2.14× for Qwen3-VL-2B and by 3.11× for LLaVA-1.5-7B (Table 8). In contrast, Shin et al. (2026) found on Jetson Orin devices that 4-bit quantization of small VLMs could slow token generation because of the overhead of dequantizing weights. The same bit-width can therefore speed up or slow down generation depending on the quantization format and inference kernels, which is why PlantVLM-Edge evaluates complete configurations end to end on the target device rather than inferring speed from model size.

The results also identify where the next latency gains lie. Once decoding is faster, time to first token becomes a larger share of the response (Figure 3): for a 64-token answer from Qwen3-VL-2B, it rises from 8.5% at 16 bits to 15.9% at 4 bits. Time to first token barely changed under quantization, because the vision encoder, which runs during prefill, remained at 16 bits in every configuration. Compressing the vision encoder is therefore the most direct next step for reducing latency, together with serving engines optimized for the Jetson platform, such as TensorRT-LLM.

## 6. Conclusion

This paper presented PlantVLM-Edge, a quantized vision-language framework for conversational crop health diagnosis that runs entirely on edge devices. The framework adapts a pretrained VLM to agricultural question answering with LoRA, compresses it with post-training quantization, and serves it on an NVIDIA Jetson AGX Orin through an ofline, browser-based chat interface. We instantiated it with two backbones, Qwen3-VL-2B and LLaVA-1.5-7B, on plant disease (PlantExpertVQA) and insect pest (IP-VQA) diagnosis, and evaluated four quantization configurations for both answer quality and on-device eficiency. Three main findings emerge. First, domain adaptation determines answer quality, whereas quantization costs little: finetuning raised ROUGE-L by 24-63 points, 8-bit quantization changed it by at most 0.3 points, and 4-bit AWQ retained at least 98.6% of full-precision ROUGE-L for both backbones on both datasets. Second, bit-width determines memory and speed, whereas the quantization method matters only for answer quality at 4 bits on the compact backbone, where AWQ outperformed round-to-nearest. Third, backbone selection matters more than quantization: the fine-tuned Qwen3-VL-2B matched LLaVA-1.5-7B in answer quality while being smaller and faster at every bit-width. In its most compact form, PlantVLM-Edge requires 2.01 GiB of weights and completes a 64-token diagnostic answer in 0.78 s. We recommend Qwen3-VL-2B with W8A16-RTN as the default configuration and W4A16-AWQ where memory is limited. Future work will compress the vision encoder, evaluate Jetson-optimized serving engines and lower-cost devices such as the Jetson Orin Nano under multi-user load, and assess diagnostic correctness through expert evaluation and field trials with growers.

## 7. Authorship Contribution

Gaurab Wagle: Formal analysis, Data curation, Software, Writing - original draft; Bishal Adhikari: Formal analysis, Software, Writing - original draft; Jiajia Li: Conceptualization, Supervision, Writing - review & editing; Xiaolei Huang: Conceptualization, Writing - review & editing; Dong Chen: Conceptualization, Investigation, Supervision, Writing - review & editing;

## 8. Declaration of AI-assisted Technologies

During the preparation of this work, the author(s) used Generative AI (Claude) to assist with language editing, clarity, manuscript organization, and polishing the visual presentation of flowchart figures. After using this tool, the author(s) reviewed and edited the content and figures as needed and take(s) full responsibility for the content of the published article.

## References

Adhikari, B., Li, J., Michel, E.S., Dykes, J., Tseng, T.M., Tagert, M.L., Chen, D., 2026. A comprehensive evaluation of yolo-based deer detection performance on edge devices. Electronics 15, 1026.

Awais, M., Alharthi, A.H.S.A., Kumar, A., Cholakkal, H., Anwer, R.M., 2025. Agrogpt: Eficient agricultural vision-language model with expert tuning, in: 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), IEEE. pp. 5687–5696.

Baek, T., 2025. open-webui: User-friendly ai interface. github.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al., 2020. Language models are few-shot learners. Advances in neural information processing systems 33, 1877–1901.

Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., Zagoruyko, S., 2020. End-to-end object detection with transformers, in: European conference on computer vision, Springer. pp. 213–229.

Chen, A., Stanovsky, G., Singh, S., Gardner, M., 2019. Evaluating question answering evaluation, in: Proceedings of the 2nd workshop on machine reading for question answering, pp. 119–124.

Chen, C.J., Huang, Y.Y., Li, Y.S., Chen, Y.C., Chang, C.Y., Huang, Y.M., 2021. Identification of fruit tree pests with deep learning on embedded drone to achieve accurate pesticide spraying. IEEE access 9, 21986– 21997.

Cheng, Y., Wang, D., Zhou, P., Zhang, T., 2018. Model compression and acceleration for deep neural networks: The principles, progress, and challenges. IEEE Signal Processing Magazine 35, 126–136.

Chu, X., Qiao, L., Lin, X., Xu, S., Yang, Y., Hu, Y., Wei, F., Zhang, X., Zhang, B., Wei, X., et al., 2023. Mobilevlm: A fast, strong and open vision language assistant for mobile devices. arXiv preprint arXiv:2312.16886 .

Dosovitskiy, A., 2020. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 .

El Jarroudi, M., Kouadio, L., Delfosse, P., Bock, C.H., Mahlein, A.K., Fettweis, X., Mercatoris, B., Adams, F., Lenné, J.M., Hamdioui, S., 2024. Leveraging edge artificial intelligence for sustainable agriculture. Nature Sustainability 7, 846–854.

Ferentinos, K.P., 2018. Deep learning models for plant disease detection and diagnosis. Computers and electronics in agriculture 145, 311–318.

Food and Agriculture Organization, 2026. Plant production and protection division: About. URL: https://www.fao.org/plant-production-protect ion/about/en. [Accessed 26 September 2026].

Frantar, E., Ashkboos, S., Hoefler, T., Alistarh, D., 2022. Gptq: Accurate post-training quantization for generative pre-trained transformers. arXiv preprint arXiv:2210.17323 .

Gauba, A., Pi, I., Man, Y., Pang, Z., Adve, V., Wang, Y.X., 2026. Agmmu: A comprehensive agricultural multimodal understanding benchmark. Advances in Neural Information Processing Systems 38.

Ghosh, A., Acharya, A., Saha, S., Jain, V., Chadha, A., 2024. Exploring the frontier of vision-language models: A survey of current methodologies and future directions. doi:10.48550/arXiv.2404.07214.

Gui, P., Dang, W., Zhu, F., Zhao, Q., 2021. Towards automatic field plant disease recognition. Computers and Electronics in Agriculture 191, 106523.

Howard, A.G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., Adam, H., 2017. Mobilenets: Eficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861 .

Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685 .

Jena, S.K., Arjunan, P., 2026. Bananavlm: A domain-adapted vision language model for banana crop disease diagnosis. arXiv preprint arXiv:2609.25040 .

Jin, K., Zi, X., Thiyagarajan, K., Braytee, A., Prasad, M., 2025. Ipvqa dataset: empowering precision agriculture with autonomous insect pest management through visual question answering, in: Companion Proceedings of the ACM on Web Conference 2025, pp. 2000–2007.

Kaur, M., Singh, R., Alirezaee, S., Hussain, I., 2025. Visual-language transformer-based tomato leaf disease detection for portable greenhouse monitoring device. Plant Methods 21, 139.

Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., et al., 2023. Segment anything, in: 2023 IEEE/CVF international conference on computer vision (ICCV), IEEE. pp. 3992–4003.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C.H., Gonzalez, J., Zhang, H., Stoica, I., 2023. Eficient memory management for large language model serving with pagedattention, in: Proceedings of the 29th symposium on operating systems principles, pp. 611–626.

Li, J., Li, D., Savarese, S., Hoi, S., 2023. Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models, in: International conference on machine learning, PmLR. pp. 19730–19742.

Li, J., Xu, M., Xiang, L., Chen, D., Zhuang, W., Yin, X., Li, Z., 2024. Foundation models in smart agriculture: Basics, opportunities, and challenges. Computers and Electronics in Agriculture 222, 109032.

Lin, C.Y., 2004. Rouge: A package for automatic evaluation of summaries, in: Text summarization branches out, pp. 74–81.

Lin, J., Tang, J., Tang, H., Yang, S., Chen, W.M., Wang, W.C., Xiao, G., Dang, X., Gan, C., Han, S., 2024. Awq: Activation-aware weight quantization for on-device llm compression and acceleration. Proceedings of machine learning and systems 6, 87–100.

Liu, H., Li, C., Li, Y., Lee, Y.J., 2024a. Improved baselines with visual instruction tuning, in: 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE. pp. 26286–26296.

Liu, H., Li, C., Wu, Q., Lee, Y.J., 2023. Visual instruction tuning. Advances in Neural Information Processing Systems 36, 34892–34905.

Liu, J., Wang, X., 2021. Plant diseases and pests detection based on deep learning: a review. Plant methods 17, 22.

Liu, X., Liu, Z., Hu, H., Chen, Z., Wang, K., Wang, K., Lian, S., 2024b. A multimodal benchmark dataset and model for crop disease diagnosis, in: European conference on computer vision, Springer. pp. 157–170.

Meng, Y., Xu, M., Kim, H., Yoon, S., Jeong, Y., Park, D.S., 2023. Known and unknown class recognition on plant species and diseases. Computers and Electronics in Agriculture 215, 108408.

Mohanty, S.P., Hughes, D.P., Salathé, M., 2016. Using deep learning for image-based plant disease detection. Frontiers in plant science 7, 215232.

Nagel, M., Fournarakis, M., Amjad, R.A., Bondarenko, Y., Van Baalen, M., Blankevoort, T., 2021. A white paper on neural network quantization. a

Novaliendry, D., Aldhy, Hendriyani, Y., Budayawan, K., Aljundi, I.I., 2026. An edge-deployable intelligent decision support system for chili leaf disease diagnosis integrating lightweight convolutional neural network and large language model-based agronomic assistance. Ingénierie des Systèmes d’Information doi:10.18280/isi.310707.

NVIDIA Corporation, 2024. NVIDIA NIM benchmarking: Key performance metrics. URL: https://docs.nvidia.com/nim/benchmarking/llm/l atest/metrics.html. accessed: 2026-03-25.

Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al., 2023. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 .

Papineni, K., Roukos, S., Ward, T., Zhu, W.J., 2002. Bleu: a method for automatic evaluation of machine translation, in: Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pp. 311–318.

Post, M., 2018. A call for clarity in reporting bleu scores, in: Proceedings of the third conference on machine translation: Research papers, pp. 186– 191.

Qwen Team, 2025. Qwen3-vl technical report. URL: https://arxiv.org/ abs/2511.21631.

Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al., 2021. Learning transferable visual models from natural language supervision, in: International conference on machine learning, PmLR. pp. 8748–8763.

Rajpurkar, P., Zhang, J., Lopyrev, K., Liang, P., 2016. Squad: 100,000+ questions for machine comprehension of text, in: Proceedings of the 2016 conference on empirical methods in natural language processing, pp. 2383–2392.

Red Hat AI and vLLM Project, 2024. LLM Compressor. URL: https: //github.com/vllm-project/llm-compressor.

Sakib, S.N., Haque, N., Hossain, M.Z., Arman, S.E., 2026. A visual question answering dataset for benchmarking vision-language models in plant science. Scientific Data 13, 1161.

Savary, S., Willocquet, L., Pethybridge, S.J., Esker, P., McRoberts, N., Nelson, A., 2019. The global burden of pathogens and pests on major food crops. Nature ecology & evolution 3, 430–439.

Sharma, K., Vats, V., Singh, A., Sahani, R., Rai, D., Sharma, A., 2024. Llava-plantdiag: Integrating large-scale vision-language abilities for conversational plant pathology diagnosis, in: 2024 International Joint Conference on Neural Networks (IJCNN), IEEE. pp. 1–7.

Shin, H., Kim, C., Kim, R., Yoo, H., Kim, J., 2026. Rethinking small vlm quantization: From component-wise analysis to hardware-aware edge deployment. arXiv preprint arXiv:2607.08029 .

Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., Batra, N., 2020a. Plantdoc: A dataset for visual plant disease detection, in: Proceedings of the 7th ACM IKDD CoDS and 25th COMAD, pp. 249–253.

Singh, V., Sharma, N., Singh, S., 2020b. A review of imaging techniques for plant disease detection. Artificial Intelligence in Agriculture 4, 229–242.

Sladojevic, S., Arsenovic, M., Anderla, A., Culibrk, D., Stefanovic, D., 2016. Deep neural networks based recognition of plant diseases by leaf image classification. Computational intelligence and neuroscience 2016, 3289801.

Swanson, B.E., Rajalahti, R., 2010. Strengthening Agricultural Extension and Advisory Systems: Procedures for Assessing, Transforming, and Evaluating Extension Systems. Agriculture and Rural Development Discussion Paper. The World Bank. URL: https://openknowledge.worldb ank.org/entities/publication/2408bcde-57b4-5055-a093-773a5f4bd89d.

Tan, W., Zhang, L., Huang, Y., Peng, K., Qu, Y., 2025. M2detr: A multiband multi-scale detection transformer for pest detection. Computers and Electronics in Agriculture 235, 110325.

Thenmozhi, K., Reddy, U.S., 2019. Crop pest classification based on deep convolutional neural network and transfer learning. Computers and Electronics in Agriculture 164, 104906.

Wang, A., Cho, K., Lewis, M., 2020. Asking and answering questions to evaluate the factual consistency of summaries, in: Proceedings of the 58th annual meeting of the association for computational linguistics, pp. 5008–5020.

Wang, L., Jin, T., Yang, J., Leonardis, A., Wang, F., Zheng, F., 2024. Agrillava: Knowledge-infused large multimodal assistant on agricultural pests and diseases. arXiv preprint arXiv:2412.02158 .

Wu, X., Zhan, C., Lai, Y.K., Cheng, M.M., Yang, J., 2019. Ip102: A largescale benchmark dataset for insect pest recognition, in: 2019 IEEE/CVF conference on computer vision and pattern recognition (CVPR), IEEE. pp. 8779–8788.

Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., Han, S., 2023. Smoothquant: Accurate and eficient post-training quantization for large language models, in: International conference on machine learning, PMLR. pp. 38087–38099.

Xu, M., Yoon, S., Jeong, Y., Park, D.S., 2022. Transfer learning for versatile plant disease recognition with limited data. Frontiers in plant science 13, 1010981.

Yao, Y., Yu, T., Zhang, A., Wang, C., Cui, J., Zhu, H., Cai, T., Chen, C., Li, H., Zhao, W., et al., 2025. Eficient gpt-4v level multimodal large language model for deployment on edge devices. Nature Communications doi:10.1038/s41467-025-61040-5.

Yousif, A.J., Zaidan, F.K., Ibrahim, N.J., 2025. A portable ai-driven edge solution for automated plant disease detection. Diyala Journal of Engineering Sciences , 124–135.

Yu, P., Teng, F., Zhu, W., Shen, C., Chen, Z., Song, J., 2025. Cloudedge-device collaborative computing in smart agriculture: architectures, applications, and future perspectives. Frontiers in Plant Science doi:10 .3389/fpls.2025.1668545.

Zhang, T., Kishore, V., Wu, F., Weinberger, K.Q., Artzi, Y., 2019. Bertscore: Evaluating text generation with bert. arXiv preprint arXiv:1904.09675 .

Zhang, Z., Yang, Y., Xu, X., Liu, L., Yue, J., Ding, R., Lu, Y., Liu, J., Qiao, H., 2024. Gvc-yolo: A lightweight real-time detection method for cotton aphid-damaged leaves based on edge computing. Remote sensing 16, 3046.

Zhao, W.X., Zhou, K., Li, J., Tang, T., Wang, X., Hou, Y., Min, Y., Zhang, B., Zhang, J., Dong, Z., et al., 2026. A survey of large language models. Frontiers of Computer Science doi:10.1007/s11704-026-60308-3.