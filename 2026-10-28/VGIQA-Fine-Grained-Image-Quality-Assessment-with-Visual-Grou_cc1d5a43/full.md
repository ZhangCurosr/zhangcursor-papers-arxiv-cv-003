# VGIQA: Fine-Grained Image Quality Assessment with Visual Grounding Rewards

Weichang Wang<sup>1,∗</sup>, Liyu Chen<sup>2,∗</sup>, Shuyue Guan<sup>2</sup>, Jie Hu<sup>2</sup>, Fei Gao<sup>1</sup>, Wei Li<sup>2,†</sup>, Nannan Wang<sup>1,†</sup>

<sup>1</sup> State Key Laboratory of Integrated Services Networks, Xidian University <sup>2</sup> Huawei Foundation Model Department weichangwang@stu.xidian.edu.cn, wei.lee@huawei.com

Abstract. Image quality assessment (IQA) aims to evaluate the perceptual visual quality of images. Recently, the rapid development of Multi-modal Large Language Models (MLLMs) has significantly advanced the IQA field. By integrating chain-ofthought (CoT) reasoning and reinforcement learning (RL), these models can generate more accurate and interpretable results while relying on less training data. However, existing MLLM-based methods are predominantly constrained to holistic, image-level evaluations and lack the capability to produce fine-grained, localized assessments, which substantially limits their practical applicability. To address this limitation, we propose VGIQA, a novel IQA model endowed with both local and global quality assessment capabilities. VGIQA is trained via a two-stage paradigm comprising a cold-start phase followed by reinforcement learning. Driven by carefully designed local quality assessment reward functions, our model exhibits exceptional image quality understanding on VGSR-2K, a new dataset we propose specifically for local quality evaluation, significantly outperforming existing open-source MLLMs such as Qwen2.5-VL-72B-Instruct [1] and Qwen3-VL-32B-Instruct [2]. Furthermore, the local quality evaluation task can efectively enhance the model’s performance on the degradation perception task.

## 1 Introduction

Image Quality Assessment (IQA) serves as a foundational pillar in computer vision, underpinning tasks ranging from classical restoration (e.g., super-resolution, denoising) to the surging field of AI-Generated Content (AIGC). It provides a quantitative and objective measure for evaluating the perceptual quality of visual content, which is crucial for guiding model training, optimizing generative processes, detecting artifacts, and ensuring the reliability of visual systems in practical deployment.

Early IQA research primarily focused on handcrafted feature-based approaches, which rely on manually designed statistical or perceptual metrics to assess image quality. Representative methods include PSNR, SSIM [38]. However, these handcrafted methods sufer from inherent limitations: they exhibit poor generalization across diverse distortion types, as their design is constrained by predefined perceptual or statistical assumptions rather than data-driven learning. Then, the subsequent era of Deep Learning, represented by NIMA [35], ManIQA [45] and CLIP-IQA [37] methods, significantly improved alignment with human subjective scores by leveraging data-driven deep features. Compared to handcrafted approaches, these IQA methods achieve superior alignment with human subjective evaluations, as they can automatically learn high-level perceptual features and complex distortion patterns from large-scale datasets. They also demonstrate stronger generalization across diverse distortion types, making them more suitable for real-world vision tasks. Nevertheless, these methods still have notable drawbacks: They treat images as holistic distributions, ofering a single quality score while remaining silent on where and why the quality degradation occurs.

![](images/eddeaee9111b456b0c446a55cabb5e6e677ae51e79dbd57744227f48f2f891cd.jpg)  
Fig. 1: Qualitative visualization of the local quality assessment results. Given a reference image, a distorted image to be evaluated, and a user query, VGIQA produces more reliable and accurate responses compared with the state-of-the-art RL-based IQA method Q-insight and the general large vision-language model Qwen2.5-VL-72B-Instruct [1].

This lack of spatial interpretability is particularly problematic for modern generative models (e.g., Difusion Models), which frequently introduce spatially inhomogeneous artifacts—such as local texture blurring or structural distortions—amidst otherwise high-fidelity regions.

In recent years, with the rapid advancement of Multimodal Large Language Models (MLLMs), MLLM-based IQA methods have emerged as a new research paradigm, enabling interpretable and semantic-aware quality assessment by fusing visual and linguistic information. These methods can be roughly divided into two categories based on their training paradigms. The first category adopts supervised fine-tuning (SFT) on IQA datasets, such as Q-Align [41], DeQA [47], and DepictQA [51]. These methods leverage annotated quality scores or constructed text descriptions to align MLLMs with human subjective judgments, achieving impressive performance in both global quality scoring and descriptive quality evaluation. However, they heavily rely on large-scale, high-quality annotated IQA data, which limits their scalability and generalization to unseen distortion types. The second category employs reinforcement learning (RL) to align with human preferences, such as Q-insight [18]. By designing reward functions that reflect human perceptual preferences, these methods eliminate the reliance on large-scale annotated data, enhance the model’s adaptability to diverse quality scenarios, and better align with human subjective judgments. Critically, modern low-level vision and generative models frequently sufer from spatially inhomogeneous quality degradation—such as local artifacts, spurious textures, distorted details, and structural anomalies—due to their optimization objectives based on global losses (prioritizing global statistical rationality) and the lack of local regularity constraints. Yet, all existing MLLM-based IQA methods lack the capability of fine-grained local quality assessment, failing to explicitly locate and evaluate these local quality defects. Even some general large-scale MLLMs perform poorly on this task, as shown in Figure 1. Therefore, an IQA model capable of accurately perceiving, localizing, and suppressing local quality degradation is crucial for enhancing the reliability and realism of modern low-level vision and generative models.

To bridge this gap, we propose VGIQA, the MLLM-based framework that integrates visual grounding with fine-grained IQA. Our key insight is that global quality scoring should be grounded in verifiable local evidence. To achieve this, we introduce a novel training paradigm combining a "Cold Start" phase with Group Relative Policy Optimization (GRPO). Unlike traditional RLHF, GRPO enables the model to engage in a self-reflective reasoning process by comparing multiple generated outputs within a group. We elaborately design a reward system that incorporates geometric precision metrics (DIOU) and semantic alignment (F1 Score) to enforce the model’s ability to "look and then judge." To support this, we construct the VGSR-2K dataset based on the public DifIQA and SRbench datasets. This dataset adopts inference results from multiple general large models as baselines, followed by human voting and correction, forming the first local quality assessment dataset aligned with human preferences. VGSR-2K contains pairs of reference images and AI-generated images, along with corresponding text-based quality evaluations that include both global and local quality assessments—with the latter incorporating coordinate information for visual localization. Ultimately, we perform joint training on local quality assessment, degradation perception, and global score regression, yielding a unified local-and-global image quality assessment framework.

Experimental results demonstrate that, after joint training, the degradation perception task achieves significant improvements over Q-insight; meanwhile, VGIQA’s local quality assessment capability outperforms not only Q-insight but also general larger-scale MLLM, such as Qwen2.5-VL-72B-Instruct [1] and Qwen3-VL-32B-Instruct [2], as shown in Figure 1. To summarize, our contributions are three-fold:

(1) We propose VGIQA, the MLLM-based fine-grained visual quality assessment model. Unlike previous works that only provide global quality scores and descriptive evaluations for the entire image, our method can explicitly and directly identify local quality issues in images, even outperforming existing general-purpose VL large models.

(2) We design a multi-task joint training framework. Through elaborately designed reward functions, VGIQA is capable of efectively handling both global and local quality assessment tasks.

(3) We release the first reference-based fine-grained visual grounding IQA dataset. The VGSR-2K dataset provides both a training set and a benchmark, enabling the community to improve and evaluate the spatial interpretability of quality assessment models.

## 2 Related Works

Early image quality assessment (IQA) employs handcrafted features to score image quality and is primarily categorized into full-reference (FR-IQA) [30, 38, 54] and no-reference (NR-IQA) [21,23–26,28] methods. FR-IQA evaluates image quality by calculating the similarity between the distorted image and a high-quality reference image. For instance, SSIM [38] computes brightness, contrast, and structural similarity separately and then performs weighted fusion; VIF [30] calculates the proportion of useful information retained after distortion; FSIM [54] measures the phase (edge, contour, and texture position and direction) that the human eye is most sensitive to, which is similar to the gradient features derived from the Sobel operator. NR-IQA method relies on manually designed statistical features in the spatial or frequency domain to predict image quality without a reference image. For example, BRISQUE [23] extracts generalized Gaussian distribution (GGD) natural statistical features from locally normalized residuals in the spatial domain; BLIINDS [28] models the GGD distribution of each subband coeficient in the DCT frequency domain to capture distortions such as compression and blocking artifacts. While the above methods rely on handcrafted features for image quality scoring, they cannot accurately evaluate complex practical scenarios involving local distortions or real-world distortions.

With the advancement of neural networks, deep learning-based IQA models [14, 15, 20, 27,32,33,37,45,60,61] have emerged. NIMA [35] employs deep networks to predict the distribution of image quality scores; ManIQA [45] incorporates multi-scale features and attention mechanisms to enhance focus on sensitive regions; MUSIQ [15] utilizes Transformers and multi-scale slicing to evaluate the quality of images of any size; CLIP-IQA [37] leverages CLIP’s capability to align image and text semantics, improving its generalization across different datasets and scenarios. These IQA models typically treat images as a whole, providing a single overall quality score without specifying the location or cause of quality degradation, and failing to align with human preferences.

![](images/1cd74653d93f0ccc47e4ef6542b51b0537558032dd37c99b0c8a57e02d1aca32.jpg)  
Fig. 2: Example applications of the proposed VGIQA framework. Built upon a Qwen-based foundation model, VGIQA achieves comprehensive quality understanding via multi-task reinforcement learning. While existing models (e.g., Q-Insight) are limited to holistic Score Regression and Degradation Perception, VGIQA explicitly bridges the gap in fine-grained Local Quality Assessment. Driven by novel visual grounding rewards, it adopts a rigorous "look and then judge" paradigm—precisely localizing spatial artifacts before formulating final judgments.

Multimodal large language models (MLLMs) with visual understanding capabilities have demonstrated promising performance in image quality evaluation [3,11,18,39–43,49,51,56– 58]. One approach employs supervised fine-tuning on image quality assessment datasets. For instance, Q-Instruct [40] creates a multimodal image quality assessment dataset to enhance the image perception capabilities of MLLMs. Q-Align [41] evaluates image quality by calculating the alignment between image features and the semantic features of "high-quality images", using pairwise ranking loss or direct regression of Mean Opinion Score (MOS). DepictQA [51] utilizes paired training data consisting of "image-quality description text" pairs, enabling MLLMs to generate natural language descriptions of distortion types, locations, severity, and their impact on visual content. Another approach leverages reinforcement learning (RL) [5,10,29,31,34,44,46,52,53,55,59] to align with human preferences. For example, Q-Insight [18] designs scoring rewards, degradation classification rewards and intensity perception rewards, employing GRPO [29] reinforcement learning to improve the prediction accuracy of subjective scores, distortion types, and severity, significantly enhancing the model’s cross-domain generalization and reasoning reliability. Q-Hawkeye [43] introduces an uncertainty-aware reward mechanism to increase the weight of stable samples, reduce noise interference, and improve model accuracy. Refine-IQA [11] designs a probability difference reward to specifically supervise the model’s "reasoning process." However, these MLLM-based image quality assessment methods lack fine-grained local quality assessment capabilities and cannot precisely locate and evaluate local quality defects.

![](images/73338b49c5f23e1f26165c75f0a6dbebf4a1a6b753e321a0825149f983894593.jpg)  
Fig. 3: Overview of the proposed VGIQA framework. The framework follows a two-stage training paradigm. Stage 1 (Cold Start): The Qwen-based foundation model is initialized via SFT using CoT reasoning data distilled from Qwen3-VL-Plus [2] to establish structural formatting and inject IQA priors. Stage 2 (Multi-Task GRPO): Reinforcement learning jointly optimizes Local Quality Assessment, Degradation Perception, and Score Regression. Alongside a shared format reward, task-specific rewards are dynamically routed: a dedicated local reward exclusively supervises the local task (computing zero for non-local tasks), evaluating spatial bounding boxes via Hungarian Matching, negative DIoU, and F1 Score. Meanwhile, global tasks are optimized via verifiable score and classification/intensity rewards, unifying local-and-global quality evaluation.

## 3 Methodology

## 3.1 Overview

To address the inherent limitations of existing MLLMs that predominantly rely on holistic evaluations, we propose VGIQA, a unified framework explicitly endowed with both finegrained local and global quality perception capabilities. As illustrated in Fig. 3, VGIQA is driven by a two-stage training paradigm. First, a Cold Start phase aligns the reasoning format and injects domain-specific IQA priors via supervised fine-tuning. Subsequently, a Multi-Task GRPO stage jointly optimizes local and global assessments to align with human preferences. A core algorithmic breakthrough of our framework is the strict reward isolation mechanism: the local visual grounding reward is exclusively routed to the local quality assessment task. This ensures that the precise spatial localization of defective regions is strictly decoupled from holistic global tasks (i.e., degradation perception and score regression), preventing objective confusion and inter-task interference.

## 3.2 Cold Start: Initializing IQA-Specific CoT Reasoning

Directly applying large-scale reinforcement learning to MLLMs without proper initialization often leads to unconstrained exploration and unstable convergence. To mitigate this, we introduce a Cold Start phase to explicitly align the model’s reasoning trajectories with structured formats and inject domain-specific IQA knowledge. To guarantee broad generalizability across diverse visual generation and restoration scenarios, we formalize the visual inputs as a unified image group comprising a reference image $\left( I _ { r e f } \right)$ and an evaluated image (I<sub>eval</sub>).

Specifically, we curate a high-quality Chain-of-Thought (CoT) dataset, denoted as $\mathcal { D } _ { C S }$ comprising 2, 288 structurally rigorous samples meticulously distilled from the Qwen3-VL-Plus [2] model. For each training instance $( x , y ) \in \mathcal { D } _ { C S }$ , the input x encompasses $\{ I _ { r e f } , I _ { e v a l } \}$ coupled with a task-specific textual prompt, while $y = \{ y _ { 1 } , y _ { 2 } , . . . , y _ { T } \}$ represents the target structured CoT response sequence of length T. The Qwen-based foundation model is then initialized via Supervised Fine-Tuning (SFT) using the standard autoregressive negative log-likelihood loss:

$$
\mathcal { L } _ { C S } ( \boldsymbol { \theta } ) = - \sum _ { i = 1 } ^ { T } \log p ( y _ { i } | \boldsymbol { x } , y _ { < i } ; \boldsymbol { \theta } ) ,\tag{1}
$$

where θ denotes the trainable parameters. This optimization objective explicitly endows the model with highly interpretable, step-by-step reasoning capabilities and professional terminology, establishing a robust policy initialization for the subsequent multi-task GRPO phase.

## 3.3 Multi-Task Visual Reinforcement Fine-Tuning via GRPO

Rewards for Local Quality Assessment To endow the model with precise spatial localization capabilities, we evaluate predicted bounding boxes $B ^ { p } = \{ b _ { 1 } ^ { p } , \ldots , b _ { m } ^ { p } \}$ against ground truths $B ^ { g } = \{ b _ { 1 } ^ { g } , . . . , b _ { n } ^ { g } \}$ . We first categorize the predictions into three macro-scene types $c \in \{ S _ { e m p t y } , S _ { w h o l e } , S _ { p a r t i a l } \}$ . A class-balanced base reward $r _ { b a s e } = 1 0 / p _ { c }$ is assigned for correctly predicting c, where $p _ { c }$ is the class frequency, mitigating spatial imbalance. For the non-trivial $ { S _ { p a r t i a l } }$ scenario, we solve an optimal bipartite matching problem via the Hungarian algorithm using negative Distance Intersection over Union (DIoU) as the cost matrix $\mathbf { C } \in \mathbb { R } ^ { m \times n }$ . To provide continuous gradient signals for coordinate regression, we define $r _ { c o n t }$ as the average normalized DIoU of the $| \pi |$ matched pairs:

$$
r _ { c o n t } = \frac { 1 } { \left| \pi \right| } \sum _ { i = 1 } ^ { \left| \pi \right| } \frac { 1 + \mathrm { D I o U } ( b _ { i } ^ { p } , b _ { \pi ( i ) } ^ { g } ) } { 2 } .\tag{2}
$$

The overall local quality assessment reward $r _ { l o c a l }$ is formulated as:

$$
r _ { l o c a l } = r _ { b a s e } + \omega _ { l o c a l } ( F _ { 1 } + r _ { c o n t } ) ,\tag{3}
$$

where $\omega _ { l o c a l }$ is a dynamic weighting coeficient preventing the model from collapsing into trivial solutions (e.g., predicting only empty boxes) during early training.

Overall Multi-Task Reward Formulation To seamlessly unify local localization with holistic quality perception, we integrate our local reward with global tasks strictly following the configurations of Q-Insight [18]. Specifically, we define a structural format reward $r _ { f o r m a t } ,$ , a binary score regression reward $r _ { s c o r e }$ (activated if the predicted score falls within an ϵ-threshold of the ground-truth MOS), and two degradation perception rewards: $r _ { d e g }$ for correct distortion type classification, and $r _ { l e v }$ for precise severity level matching.

Crucially, to implement the strict reward isolation mechanism highlighted in Sec. 3.1, the overall multi-task reward $r _ { t o t a l }$ dynamically routes task-specific gradients using indicator functions:

$$
r = r _ { f o r m a t } + \lambda _ { l o c a l } r _ { l o c a l } + \lambda _ { s c o r e } r _ { s c o r e } + \lambda _ { d e g } ( \alpha _ { 1 } r _ { d e g } + \alpha _ { 2 } r _ { l e v } ) .\tag{4}
$$

Here, $\lambda _ { l o c a l } , \ \lambda _ { s c o r e }$ , and $\lambda _ { d e g }$ serve as task-routing gating variables—evaluating to 1 exclusively for the currently sampled task and 0 otherwise. This mathematical formulation explicitly decouples local spatial reasoning from global holistic evaluation, efectively preventing inter-task interference during the GRPO phase.

## 4 Proposed Dataset: VGSR-2K

## 4.1 Overview of VGSR-2K

Despite the rapid advancements in MLLMs for IQA, the research community currently lacks high-quality datasets exclusively tailored for fine-grained, local quality assessment. Existing datasets predominantly provide holistic, image-level scores or textual descriptions, which are insuficient to train models for the precise spatial localization of local erroneous regions. To bridge this critical gap, we introduce VGSR-2K, a novel visual grounding dataset strictly constructed for fine-grained, local quality assessment. Built upon the foundational data from DifIQA [4] and SRBench [4], the data instances in VGSR-2K are organized as paired image groups. Specifically, each sample typically comprises a Low-Resolution (LR) reference image coupled with either a Super-Resolved (SR) generated image or a High-Resolution (HR) ground-truth image (i.e., $[ I _ { L R } , I _ { S R } ]$ or [I<sub>LR</sub>, I<sub>HR</sub>] pairs). The ground truth of this dataset is provided in the form of spatial bounding boxes, precisely grounding the specific locations of the erroneous regions within the evaluated images.

## 4.2 Pipeline

As illustrated in Fig. 4, we design a rigorous human-in-the-loop pipeline to ensure the precision and consistency of the spatial annotations. Based on a pilot study, we establish a protocol to explicitly annotate three categories of erroneous regions: (a) Generation Artifacts (e.g., structural distortions or visual noise); (b) Semantic Inconsistency (i.e., semantic deviations from the LR reference); and (c) Unrealistic Hallucinations (i.e., spurious contents violating physical laws). The pipeline proceeds in four stages.

Candidate Generation. We utilize two state-of-the-art MLLMs, Qwen3-VL-Plus [2] and Seed1.6-VL, for initial proposal generation. For each of the 1, 798 $[ I _ { L R } , I _ { S R } ]$ image pairs, each model performs three independent inferences, yielding six candidate spatial bounding box sets per pair.

Double-Blind Selection. Four experienced annotators are equally divided into two independent groups (Group A and Group B). They contrast the LR and SR images to blindly select the most visually consistent candidate among the six. Consensual selections are directly adopted as initial baselines. For divergent samples, two senior IQA experts intervene to determine the optimal baseline.

Double-Blind Correction and Merging. To prevent manual coordinate-level bias, Groups A and B independently refine the baselines by strictly deleting incorrect boxes or supplementing missing ones, without fine-tuning existing coordinates. These independent corrections are algorithmically merged: we take the intersection of the retained box indices from both groups. For supplemented boxes, we formulate a bipartite matching problem via the Hungarian algorithm with negative DIoU as the cost matrix. Matched box pairs with DIoU > 0.5 are preserved, and their Maximum Bounding Rectangle (MBR) is computed as the final merged annotation.

Final Review & Balancing. Finally, two senior experts conduct a comprehensive review of the algorithmically merged samples to resolve any residual ambiguities, ensuring highfidelity spatial ground truths.

Stage 1: Candidate Generation  
Stage 2: Double-Blind Selection  
![](images/dfb5f6b97f0a90048b5c44481962d050f5d4b954be59dd57d0a6952fa38be059.jpg)  
Fig. 4: Overview of the VGSR-2K dataset construction pipeline. The dataset is curated via a rigorous human-in-the-loop pipeline. (1) Candidate Generation: Qwen3-VL-Plus [2] and Seed1.6-VL independently process LR-SR image pairs, yielding six candidate spatial bounding boxes. (2) Double-Blind Selection: Two annotator groups select the most consistent candidate; consensus establishes the baseline, while divergence prompts expert resolution. (3) Double-Blind Correction & Merging: Annotators refine baselines by adding or removing boxes. Corrections are merged via the Hungarian algorithm with a negative DIoU cost metric, extracting the MBR for matched pairs. (4) Final Review & Balancing: After expert review, pristine LR-HR image pairs are introduced as negative samples (with empty annotations) to calibrate spatial predictions and prevent trivial over-generation.

## 4.3 Dataset Statistics and Distribution

Following the rigorous annotation pipeline, the 1, 798 meticulously verified data groups, denoted as $[ I _ { L R } , I _ { S R } , \mathrm { r e s u l t } _ { x } ]$ , are partitioned into identically distributed training and testing sets. A critical challenge in visual grounding for quality assessment is the potential for trivial over-generation, driven by the severe spatial imbalance between explicitly annotated localized artifacts and vast background areas. To counteract this, we purposefully introduce pristine $[ I _ { L R } , I _ { H R } ]$ pairs into both sets. Inherently lacking any defective areas, these reference pairs carry empty spatial annotations, serving as high-quality negative samples to systematically calibrate the model’s spatial prediction behavior.

The dataset is systematically constructed from two distinct subsets to encompass diverse generative flaws. For the DifIQA subset, we allocate 698 $[ I _ { L R } , I _ { S R } ]$ groups for training and 100 for testing. To maintain balance, we supplement these with 102 and 18 negative $[ I _ { L R } , I _ { H R } ]$ pairs, yielding 800 training and 118 testing samples, respectively. The training sources are rigorously balanced across diverse domains, including DF2K (264), the Internet (266), and camera captures (270), with progressive low-resolution degradation intensities (Levels 1 to 6) proportionately distributed as 133, 127, 135, 149, 147, and 109. The testing set maintains this identical distribution, comprising domains of DF2K (32), the Internet (33), and camera captures (53), alongside corresponding degradation levels of 22, 18, 18, 21, 25, and 14.

To further enhance robustness against unknown distortions, the SRBench subset incorporates multi-paradigm data spanning 100 natural scenes. Each scene intrinsically provides one HR reference and 10 corresponding SR images generated by diverse architectures $( i . e . ,$ regression-based, GAN-based, and difusion-based models). We randomly allocate 80 scenes to formulate the training set, yielding 800 $[ I _ { L R } , I _ { S R } ]$ groups and 80 $[ I _ { L R } , I _ { H R } ]$ negative pairs (totaling 880 samples). The remaining 20 scenes are exclusively allocated to the testing set, encompassing 200 positive and 20 negative pairs (totaling 220 samples). This highly stratified organization ensures that VGSR-2K comprehensively encapsulates a broad spectrum of visual characteristics and spatial distributions of flawed regions, providing a statistically robust foundation for fine-grained quality assessment.

Table 1: Quantitative comparison of the local quality assessment task on the VGSR-2K dataset. Qwen2.5-VL-72B and Qwen3- VL-32B denote Qwen2.5-VL-72B-Instruct and Qwen3-VL-32B-Instruct, respectively. VGIQA significantly outperforms both large-scale general MLLMs and the IQA-specific baseline. The best and second-best results are highlighted in bold and underlined, respectively.  
Table 2: Ablation study of each training stage on the local quality assessment task. The base model is Qwen2.5-VL-7B-Instruct [1]. The GRPO phase is crucial for the improvement in Local Quality Assessment. The best and second-best results are highlighted in bold and underlined, respectively.
<table><tr><td>Method</td><td>F1 Score DIoU</td></tr><tr><td>Q-Insight [18]</td><td>0.1452 0.1673</td></tr><tr><td>Qwen2.5-VL-72B [1]</td><td>0.1770 0.1997</td></tr><tr><td>Qwen3-VL-32B [2]</td><td>0.1756 0.2185</td></tr><tr><td>VGIQA (Ours)</td><td>0.3024 0.3235</td></tr></table>

<table><tr><td>Method</td><td>F1 Score DIoU</td></tr><tr><td>Base Model</td><td>0.1497 0.1672</td></tr><tr><td>Base Model + Cold Start</td><td>0.1455 0.2013</td></tr><tr><td>Base Model + Cold Start + GRPO (VGIQA)</td><td>0.3024 0.3235</td></tr></table>

## 5 Experiments

## 5.1 Experimental Setup

Datasets. For the local quality assessment task, we train and evaluate VGIQA using the proposed VGSR-2K dataset (detailed in Sec. 4.3). During training, the diverse data subsets are thoroughly mixed, whereas evaluations are conducted separately on their respective testing subsets. For global assessment tasks, we strictly follow the configurations of Q-Insight [18]. Specifically, the score regression task utilizes identical splits across established benchmarks (e.g., KonIQ, SPAQ, KADID, PIPAL, LIVE-Wild, AGIQA-3K, and CSIQ). For the degradation perception task, we adopt the identical training set from DQ-495K and randomly sample 1, 000 identically distributed images to formulate our test set.

Evaluation Metrics. We adopt distinct assessment metrics tailored to the specific characteristics of each task. For the local quality assessment task, we utilize the F1 Score and DIoU, as formulated in Sec. 3.3, to rigorously measure the spatial localization accuracy of erroneous regions. For the score regression task, we report the Pearson Linear Correlation Coeficient (PLCC) and Spearman Rank-order Correlation Coeficient (SRCC) to evaluate the consistency between model predictions and human subjective scores. For the degradation perception task, performance is measured using the prediction accuracy of degradation categories (Deg.Acc.) and degradation levels (Lev.Acc.).

Implementation Details. We adopt Qwen-2.5-VL-7B-Instruct [1] as our base model. The training encompasses a 3-epoch Cold Start phase and a 10-epoch GRPO phase. Elaborate hyperparameter settings and training configurations are detailed in Sec. D.3 of the Appendix.

## 5.2 Local Quality Assessment

We first evaluate VGIQA on the local quality assessment task, which aims to precisely localize erroneous regions in evaluated images. As this is a newly introduced fine-grained assessment paradigm, existing MLLM-based IQA models inherently lack the capability to directly output explicit spatial coordinates for erroneous regions. To demonstrate the superiority of our approach, we compare VGIQA with current state-of-the-art large-scale general-purpose MLLMs, including Qwen2.5-VL-72B-Instruct [1] and Qwen3-VL-32B-Instruct [2], as well as an IQA-specific SOTA model of the same parameter scale, Q-Insight [18]. As shown in Tab. 1, VGIQA achieves substantial performance gains on the VGSR-2K dataset. Remarkably, even when compared to general-purpose MLLMs with 10 times more parameters (e.g., Qwen2.5-VL-72B-Instruct [1]), VGIQA still demonstrates a commanding lead. Specifically, VGIQA attains an F1 Score of 0.3024 and a DIoU of 0.3235, far exceeding the baseline models. These results strongly validate that through our proposed two-stage training paradigm and carefully designed local rewards, VGIQA efectively fills the blank in fine-grained spatial localization for image quality assessment.

Table 3: Quantitative comparison of the Degradation Perception task. VGIQA signifi cantly improves the perception accuracy of global physical degradations, especially in Noise and Null(no degradation). The best and second-best results are highlighted in bold and underlined, respectively.
<table><tr><td>Method</td><td>Metrics</td><td>Blur</td><td>JPEG</td><td>Darken</td><td>Noise</td><td>Null</td><td>Average</td></tr><tr><td>Q-Insight [18]</td><td>Deg.Acc.</td><td>91.48%</td><td>98.02%</td><td>63.23%</td><td>16.13%</td><td>64.93%</td><td>66.57%</td></tr><tr><td></td><td>Lev.Acc.</td><td>43.05%</td><td>46.04%</td><td>22.70%</td><td>6.91%</td><td></td><td>34.23%</td></tr><tr><td>VGIQA (Ours) Deg.Acc. 94.17% 98.51% 67.71% 62.21% 94.03% 82.18%</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Lev.Acc. 52.91%44.55%23.77% 23.96%</td><td></td><td></td><td></td><td></td><td></td><td>43.94%</td></tr></table>

## 5.3 Global Quality Assessment

We further evaluate VGIQA on two global quality assessment tasks: degradation perception and score regression.

Degradation Perception We evaluate the capability of VGIQA in identifying global physical degradations (e.g., Blur, JPEG Compression, Darken, and Noise) and predicting their severity levels. As shown in Tab. 3, VGIQA demonstrates a clear advantage over the baseline Q-Insight [18]. Benefiting from the multi-task joint training framework, focusing on local erroneous regions encourages the model to acquire a keener ability to capture global features. Consequently, this fine-grained attention acts as a strong regularization, significantly improving the overall accuracy of the degradation perception task. Except for a slight drop in the Lev.Acc. of JPEG Compression, VGIQA achieves substantial improvements across all other categories in both Deg.Acc. and Lev.Acc. compared to Q-Insight, bringing a noticeable boost to the global image quality assessment.

Score Regression In the score regression task, we evaluate the continuous regression capability of VGIQA for global image quality scoring. The results are presented in Tab. 4. While VGIQA achieves slightly lower PLCC and SRCC compared to Q-Insight [18], it maintains highly competitive performance, being fundamentally on par with state-of-theart methods such as DeQA [48].

Mechanistically, this slight fluctuation in linear correlation is a natural manifestation of the model acquiring finer-grained assessment capabilities. By introducing sensitivity to local erroneous regions and subtle generative flaws during joint training, VGIQA inherently adopts much stricter criteria when examining images. This heightened sensitivity to microscopic imperfections leads to overall predicted scores being lower or more fluctuating, which objectively reduces the linear correlation with MOS. Therefore, this is not a degradation of capability, but rather a direct reflection that VGIQA evaluates images from a more comprehensive and fine-grained perspective.

Table 4: Quantitative comparison of the score regression task between our VGIQA and other competitive IQA methods. Results are reported in the format of PLCC/SRCC. All methods except handcrafted ones are trained on the KonIQ dataset. The best and second-best results are highlighted in bold and underlined, respectively.
<table><tr><td>Method</td><td>KonIQ</td><td>SPAQ</td><td>KADID</td><td>PIPAL</td><td>LIVE-Wild</td><td>AGIQA-3K</td><td>CSIQ</td><td>Average</td></tr><tr><td>NIQE [13]</td><td>0.533/0.530</td><td>0.679/0.664</td><td>0.468/0.405</td><td>0.195/0.161</td><td>0.493/0.449</td><td>0.560/0.533</td><td>0.718/0.628</td><td>0.521/0.481</td></tr><tr><td>BRISQUÉ [22]</td><td>0.225/0.226</td><td>0.490/0.406</td><td>0.429/0.356</td><td>0.267/0.232</td><td>0.361/0.313</td><td>0.541/0.497</td><td>0.740/0.556</td><td>0.436/0.369</td></tr><tr><td>NIMA [36]</td><td>0.896/0.859</td><td>0.838/0.856</td><td>0.532/0.535</td><td>0.390/0.399</td><td>0.814/0.771</td><td>0.715/0.654</td><td>0.695/0.649</td><td>0.697/0.675</td></tr><tr><td>MUSIQ [15]</td><td>0.924/0.929</td><td>0.868/0.863</td><td>0.575/0.556</td><td>0.431/0.431</td><td>0.789/0.830</td><td>0.722/0.630</td><td>0.771/0.710</td><td>0.726/0.707</td></tr><tr><td>CLIP-IQA+ [37]</td><td>0.909/0.895</td><td>0.866/0.864</td><td>0.653/0.654</td><td>0.427/0.419</td><td>0.832/0.805</td><td>0.736/0.685</td><td>0.772/0.719</td><td>0.742/0.720</td></tr><tr><td>ManIQA [45]</td><td>0.849/0.834</td><td>0.768/0.758</td><td>0.499/0.465</td><td>0.457/0.452</td><td>0.849/0.832</td><td>0.723/0.636</td><td>0.623/0.627</td><td>0.681/0.658</td></tr><tr><td>Compare2Score [62]</td><td>0.923/0.910</td><td>0.867/0.860</td><td>0.500/0.453</td><td>0.354/0.342</td><td>0.786/0.772</td><td>0.777/0.671</td><td>0.735/0.705</td><td>0.706/0.673</td></tr><tr><td>Qwen-SFT [2]</td><td>0.889/0.866</td><td>0.874/0.875</td><td>0.668/0.663</td><td>0.473/0.442</td><td>0.734/0.728</td><td>0.813/0.739</td><td>0.674/0.650</td><td>0.732/0.709</td></tr><tr><td>Q-Align [41]</td><td>0.941/0.940</td><td>0.886/0.887</td><td>0.674/0.684</td><td>0.403/0.419</td><td>0.853/0.860</td><td>0.772/0.735</td><td>0.671/0.737</td><td>0.705/0.752</td></tr><tr><td>DeQA [48]</td><td>0.953/0.9410.895/0.896</td><td></td><td>0.694/0.687</td><td></td><td>0.472/0.478 0.892/0.879</td><td>0.809/0.729</td><td>0.787/0.744</td><td>0.786/0.765</td></tr><tr><td>Q-Insight [18]</td><td>0.924/0.906</td><td>0.894/0.891</td><td>0.767/0.764</td><td>0.458/0.453</td><td>0.881/0.850</td><td></td><td>0.823/0.764 0.875/0.817</td><td>0.803/0.780</td></tr><tr><td>VGIQA (Ours)</td><td>0.939/0.924</td><td>0.875/0.873</td><td>0.730/0.718</td><td>0.457/0.425</td><td>0.884/0.856</td><td>0.791/0.745</td><td>0.860/0.796</td><td>0.791/0.762</td></tr></table>

## 5.4 Ablation Studies

We conduct ablation studies to decompose the core modules of VGIQA, specifically investigating the efectiveness of the proposed two-stage training paradigm and the benefits derived from multi-task joint training.

Ablation of Each Training Stage To evaluate the specific contributions of the Cold Start and GRPO phases to the Local Quality Assessment capability, we compare the variants across diferent training stages in Tab. 2. As shown, solely relying on the Cold Start phase to inject the reasoning paradigm fails to bring a substantial leap in spatial localization. The performance on F1 Score and DIoU shows merely marginal fluctuations compared to the base model. This observation verifies that while the Cold Start provides essential format alignment and domain priors, it is insuficient for precise visual grounding. Only by applying the GRPO phase, systematically guided by our carefully designed reward functions, can the model truly excite and align its fine-grained perception of erroneous regions, which efectively enhances the capability on the VGSR-2K dataset.

## 6 Conclusion

In this paper, we address the limitation of existing MLLM-based IQA models, which are confined to holistic, image-level evaluation and lack the capability to produce fine-grained, localized assessments. To bridge this gap, we propose VGIQA, a unified framework with both local and global quality assessment capabilities. Specifically, we introduce a novel local quality evaluation task that grounds erroneous regions with bounding boxes against a pristine reference, and construct VGSR-2K, a visual grounding dataset tailored to this task, together with carefully designed reward functions. VGIQA is trained with a two-stage paradigm—a cold-start phase followed by multi-task GRPO—that jointly optimizes local and global assessment. Extensive experiments show that, while remaining competitive on score regression, VGIQA achieves two breakthroughs: (1) exceptional local quality understanding, substantially outperforming general-purpose MLLMs with ten times the parameter scale (e.g., Qwen2.5-VL-72B-Instruct); and (2) the local quality assessment efectively boosts the model’s performance on the degradation perception task. We plan to extend this paradigm to video quality assessment and to the construction of reward models for tasks such as text-to-image generation and image editing.

## References

1. Bai, J., Bai, S., Chu, Y., Cui, Z., Dang, K., Deng, X., Fan, Y., Ge, W., Han, Y., Huang, F., et al.: Qwen technical report. arXiv preprint arXiv:2309.16609 (2023)

2. Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., et al.: Qwen3-vl technical report. arXiv preprint arXiv:2511.21631 (2025)

3. Chen, C., Yang, S., Wu, H., Liao, L., Zhang, Z., Wang, A., Sun, W., Yan, Q., Lin, W.: Q-ground: Image quality grounding with large multi-modality models. In: Proceedings of the 32nd ACM International Conference on Multimedia. pp. 486–495 (2024)

4. Chen, D., Wu, T., Ma, K., Zhang, L.: Toward generalized image quality assessment: Relaxing the perfect reference quality assumption. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 12742–12752 (2025)

5. Christiano, P.F., Leike, J., Brown, T., Martic, M., Legg, S., Amodei, D.: Deep reinforcement learning from human preferences. Advances in neural information processing systems 30 (2017)

6. Fang, Y., Zhu, H., Zeng, Y., Ma, K., Wang, Z.: Perceptual quality assessment of smartphone photography. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 3677–3686 (2020)

7. Ghadiyaram, D., Bovik, A.C.: Live in the wild image quality challenge database. Online: http://live. ece. utexas. edu/research/ChallengeDB/index. html [Mar, 2017] 2(5), 6 (2015)

8. Guo, D., Wu, F., Zhu, F., Leng, F., Shi, G., Chen, H., Fan, H., Wang, J., Jiang, J., Wang, J., et al.: Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062 (2025)

9. Hosu, V., Lin, H., Sziranyi, T., Saupe, D.: Koniq-10k: An ecologically valid database for deep learning of blind image quality assessment. IEEE Transactions on Image Processing 29, 4041– 4056 (2020)

10. Hui, B., Yang, J., Cui, Z., Yang, J., Liu, D., Zhang, L., Liu, T., Zhang, J., Yu, B., Lu, K., et al.: Qwen2. 5-coder technical report. arXiv preprint arXiv:2409.12186 (2024)

11. Jia, Z., Qian, J., Zhang, Z., Chen, Z., Min, X.: Refine-iqa: Multi-stage reinforcement finetuning for perceptual image quality assessment. arXiv preprint arXiv:2508.03763 (2025)

12. Jinjin, G., Haoming, C., Haoyu, C., Xiaoxing, Y., Ren, J.S., Chao, D.: Pipal: a large-scale image quality assessment dataset for perceptual image restoration. In: European conference on computer vision. pp. 633–651. Springer (2020)

13. Kamble, V., Bhurchandi, K.: No-reference image quality assessment algorithms: A survey. Optik 126(11-12), 1090–1097 (2015)

14. Kang, L., Ye, P., Li, Y., Doermann, D.: Convolutional neural networks for no-reference image quality assessment. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 1733–1740 (2014)

15. Ke, J., Wang, Q., Wang, Y., Milanfar, P., Yang, F.: Musiq: Multi-scale image quality transformer. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 5148–5157 (2021)

16. Larson, E.C., Chandler, D.M.: Most apparent distortion: full-reference image quality assessment and the role of strategy. Journal of electronic imaging 19(1), 011006–011006 (2010)

17. Li, C., Zhang, Z., Wu, H., Sun, W., Min, X., Liu, X., Zhai, G., Lin, W.: Agiqa-3k: An open database for ai-generated image quality assessment. IEEE Transactions on Circuits and Systems for Video Technology 34(8), 6833–6846 (2023)

18. Li, W., Zhang, X., Zhao, S., Zhang, Y., Li, J., Zhang, L., Zhang, J.: Q-insight: Understanding image quality via visual reinforcement learning. arXiv preprint arXiv:2503.22679 (2025)

19. Lin, H., Hosu, V., Saupe, D.: Kadid-10k: A large-scale artificially distorted iqa database. In: 2019 Eleventh International Conference on Quality of Multimedia Experience (QoMEX). pp. 1– 3. IEEE (2019)

20. Liu, X., Van De Weijer, J., Bagdanov, A.D.: Rankiqa: Learning from rankings for no-reference image quality assessment. In: Proceedings of the IEEE international conference on computer vision. pp. 1040–1049 (2017)

21. Ma, C., Yang, C.Y., Yang, X., Yang, M.H.: Learning a no-reference quality metric for singleimage super-resolution. Computer Vision and Image Understanding 158, 1–16 (2017)

22. Mittal, A., Moorthy, A.K., Bovik, A.C.: Blind/referenceless image spatial quality evaluator. In: 2011 conference record of the forty fifth asilomar conference on signals, systems and computers (ASILOMAR). pp. 723–727. IEEE (2011)

23. Mittal, A., Moorthy, A.K., Bovik, A.C.: No-reference image quality assessment in the spatial domain. IEEE Transactions on image processing 21(12), 4695–4708 (2012)

24. Mittal, A., Soundararajan, R., Bovik, A.C.: Making a “completely blind” image quality analyzer. IEEE Signal processing letters 20(3), 209–212 (2012)

25. Moorthy, A.K., Bovik, A.C.: A two-step framework for constructing blind image quality indices. IEEE Signal processing letters 17(5), 513–516 (2010)

26. Moorthy, A.K., Bovik, A.C.: Blind image quality assessment: From natural scene statistics to perceptual quality. IEEE transactions on Image Processing 20(12), 3350–3364 (2011)

27. Pan, D., Shi, P., Hou, M., Ying, Z., Fu, S., Zhang, Y.: Blind predicting similar quality map for image quality assessment. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 6373–6382 (2018)

28. Saad, M.A., Bovik, A.C., Charrier, C.: Blind image quality assessment: A natural scene statistics approach in the dct domain. IEEE transactions on Image Processing 21(8), 3339–3352 (2012)

29. Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al.: Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300 (2024)

30. Sheikh, H.R., Bovik, A.C.: Image information and visual quality. IEEE Transactions on image processing 15(2), 430–444 (2006)

31. Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., Hubert, T., Baker, L., Lai, M., Bolton, A., et al.: Mastering the game of go without human knowledge. nature 550(7676), 354–359 (2017)

32. Su, S., Yan, Q., Zhu, Y., Zhang, C., Ge, X., Sun, J., Zhang, Y.: Blindly assess image quality in the wild guided by a self-adaptive hyper network. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 3667–3676 (2020)

33. Sun, S., Yu, T., Xu, J., Zhou, W., Chen, Z.: Graphiqa: Learning distortion graph representations for blind image quality assessment. IEEE Transactions on Multimedia 25, 2912–2925 (2022)

34. Sun, Z., Shen, S., Cao, S., Liu, H., Li, C., Shen, Y., Gan, C., Gui, L., Wang, Y.X., Yang, Y., et al.: Aligning large multimodal models with factually augmented rlhf. In: Findings of the Association for Computational Linguistics: ACL 2024. pp. 13088–13110 (2024)

35. Talebi, H., Milanfar, P.: Nima: Neural image assessment. IEEE transactions on image processing 27(8), 3998–4011 (2018)

36. Talebi, H., Milanfar, P.: Nima: Neural image assessment. IEEE transactions on image processing 27(8), 3998–4011 (2018)

37. Wang, J., Chan, K.C., Loy, C.C.: Exploring clip for assessing the look and feel of images. In: Proceedings of the AAAI conference on artificial intelligence. vol. 37, pp. 2555–2563 (2023)

38. Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing 13(4), 600–612 (2004)

39. Wu, H., Zhang, Z., Zhang, E., Chen, C., Liao, L., Wang, A., Li, C., Sun, W., Yan, Q., Zhai, G., et al.: Q-bench: A benchmark for general-purpose foundation models on low-level vision. arXiv preprint arXiv:2309.14181 (2023)

40. Wu, H., Zhang, Z., Zhang, E., Chen, C., Liao, L., Wang, A., Xu, K., Li, C., Hou, J., Zhai, G., et al.: Q-instruct: Improving low-level visual abilities for multi-modality foundation models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 25490–25500 (2024)

41. Wu, H., Zhang, Z., Zhang, W., Chen, C., Liao, L., Li, C., Gao, Y., Wang, A., Zhang, E., Sun, W., et al.: Q-align: Teaching lmms for visual scoring via discrete text-defined levels. arXiv preprint arXiv:2312.17090 (2023)

42. Wu, H., Zhu, H., Zhang, Z., Zhang, E., Chen, C., Liao, L., Li, C., Wang, A., Sun, W., Yan, Q., et al.: Towards open-ended visual quality comparison. In: European Conference on Computer Vision. pp. 360–377. Springer (2024)

43. Xie, W., Dai, R., Ding, R., Liu, K., Chu, X., Hou, X., Wen, J.: Q-hawkeye: Reliable visual policy optimization for image quality assessment. arXiv preprint arXiv:2601.22920 (2026)

44. Yang, A., Zhang, B., Hui, B., Gao, B., Yu, B., Li, C., Liu, D., Tu, J., Zhou, J., Lin, J., et al.: Qwen2. 5-math technical report: Toward mathematical expert model via self-improvement. arXiv preprint arXiv:2409.12122 (2024)

45. Yang, S., Wu, T., Shi, S., Lao, S., Gong, Y., Cao, M., Wang, J., Yang, Y.: Maniqa: Multi dimension attention network for no-reference image quality assessment. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 1191–1200 (2022)

46. Ying, H., Zhang, S., Li, L., Zhou, Z., Shao, Y., Fei, Z., Ma, Y., Hong, J., Liu, K., Wang, Z., et al.: Internlm-math: Open math large language models toward verifiable reasoning. arXiv preprint arXiv:2402.06332 (2024)

47. You, Z., Cai, X., Gu, J., Xue, T., Dong, C.: Teaching large language models to regress accurate image quality scores using score distribution. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 14483–14494 (2025)

48. You, Z., Cai, X., Gu, J., Xue, T., Dong, C.: Teaching large language models to regress accurate image quality scores using score distribution. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 14483–14494 (2025)

49. You, Z., Gu, J., Li, Z., Cai, X., Zhu, K., Dong, C., Xue, T.: Descriptive image quality assessment in the wild. arXiv e-prints pp. arXiv–2405 (2024)

50. You, Z., Gu, J., Li, Z., Cai, X., Zhu, K., Dong, C., Xue, T.: Descriptive image quality assessment in the wild. arXiv e-prints pp. arXiv–2405 (2024)

51. You, Z., Li, Z., Gu, J., Yin, Z., Xue, T., Dong, C.: Depicting beyond scores: Advancing image quality assessment through multi-modal language models. In: European Conference on Computer Vision. pp. 259–276. Springer (2024)

52. Yu, T., Yao, Y., Zhang, H., He, T., Han, Y., Cui, G., Hu, J., Liu, Z., Zheng, H.T., Sun, M., et al.: Rlhf-v: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 13807–13816 (2024)

53. Yu, T., Zhang, H., Yao, Y., Dang, Y., Chen, D., Lu, X., Cui, G., He, T., Liu, Z., Chua, T.S., et al.: Rlaif-v: Aligning mllms through open-source ai feedback for super gpt-4v trustworthiness. arXiv preprint arXiv:2405.17220 2(3), 8 (2024)

54. Zhang, L., Zhang, L., Mou, X., Zhang, D.: Fsim: A feature similarity index for image quality assessment. IEEE transactions on Image Processing 20(8), 2378–2386 (2011)

55. Zhang, Y., Wu, S., Yang, Y., Shu, J., Xiao, J., Kong, C., Sang, J.: o1-coder: an o1 replication for coding. arXiv preprint arXiv:2412.00154 (2024)

56. Zhang, Z., Jia, Z., Wu, H., Li, C., Chen, Z., Zhou, Y., Sun, W., Liu, X., Min, X., Lin, W., et al.: Q-bench-video: Benchmarking the video quality understanding of lmms. arXiv preprint arXiv:2409.20063 (2024)

57. Zhang, Z., Kou, T., Wang, S., Li, C., Sun, W., Wang, W., Li, X., Wang, Z., Cao, X., Min, X., et al.: Q-eval-100k: Evaluating visual quality and alignment level for text-to-vision content. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 10621–10631 (2025)

58. Zhang, Z., Wu, H., Jia, Z., Lin, W., Zhai, G.: Teaching lmms for image quality scoring and interpreting. arXiv preprint arXiv:2503.09197 (2025)

59. Zhao, Z., Wang, B., Ouyang, L., Dong, X., Wang, J., He, C.: Beyond hallucinations: Enhancing lvlms through hallucination-aware direct preference optimization. arXiv preprint arXiv:2311.16839 (2023)

60. Zheng, H., Yang, H., Fu, J., Zha, Z.J., Luo, J.: Learning conditional knowledge distillation for degraded-reference image quality assessment. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 10242–10251 (2021)

61. Zhu, H., Li, L., Wu, J., Dong, W., Shi, G.: Metaiqa: Deep meta-learning for no-reference image quality assessment. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 14143–14152 (2020)

62. Zhu, H., Wu, H., Li, Y., Zhang, Z., Chen, B., Zhu, L., Fang, Y., Zhai, G., Lin, W., Wang, S.: Adaptive image quality assessment via teaching large multimodal model to compare. Advances in Neural Information Processing Systems 37, 32611–32629 (2024)

Table A: Ablation study of multi-task training on the degradation perception task. The Global tasks include score regression and degradation perception. Joint training with the local quality assessment task significantly improves global degradation perception. The best and second best results are highlighted in bold and underlined, respectively.
<table><tr><td>Method</td><td>Metrics</td><td>Blur</td><td>JPEG</td><td>Darken</td><td>Noise</td><td>null</td><td>Average</td></tr><tr><td rowspan="2">Global-only</td><td>Deg.Acc.</td><td>51.12%</td><td>97.52%</td><td>43.05%</td><td>43.78%</td><td>72.39%</td><td>59.96%</td></tr><tr><td>Lev.Acc.</td><td>25.11%</td><td>45.05%</td><td>14.08%</td><td>27.19%</td><td></td><td>33.63%</td></tr><tr><td rowspan="2">Global + Local (Ours) Deg.Acc. 94.17% 98.51% 67.71% 62.21% 94.03% 82.18%</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Lev.Acc. 52.91% 44.55% 23.77% 23.96%</td><td></td><td></td><td></td><td></td><td></td><td>43.94%</td></tr></table>

## A Ablation of Multi-Task Training

To explore the interactions among diferent tasks, we compare our multi-task framework with a variant trained exclusively on the global evaluation tasks. As reported in Tab. A, introducing the fine-grained local quality assessment task serves as a powerful spatial feature prompt. By compelling the model to meticulously localize microscopic erroneous regions, it efectively stimulates the model’s potential in capturing global physical degradation clues. Consequently, the joint training scheme successfully achieves steady performance improvements in the degradation perception task, demonstrating higher accuracy in both Deg.Acc. and Lev.Acc. across almost all specific degradation categories compared to the Global-only baseline.

## B More Methodological Details

Due to space constraints in the main paper, this section provides comprehensive details regarding the theoretical formulation of the reinforcement learning framework and the specific mathematical designs of our multi-task reward functions.

## B.1 Preliminaries of Group Relative Policy Optimization

To efectively elicit the intrinsic reasoning capabilities of Multi-modal Large Language Models (MLLMs) without relying on extensive supervised fine-tuning or training an extra value model, our framework adopts Group Relative Policy Optimization (GRPO) [29]. For a given multi-modal query $q ,$ GRPO samples a group of N distinct candidate responses $\mathsf { \bar { \{ } }  o ^ { ( 1 ) } , o ^ { ( 2 ) } , \ldots , o ^ { ( N ) } \}$ from the current policy model $\pi _ { \theta _ { o l d } } .$ Each response $o ^ { ( i ) }$ is evaluated using task-specific rule-based reward functions to obtain a corresponding reward score $r ^ { ( i ) }$

Instead of relying on a separate critic model to estimate the baseline, GRPO computes the relative advantage $\hat { A } ^ { ( i ) }$ for each response by normalizing the rewards within the sampled group:

$$
\hat { A } ^ { ( i ) } = \frac { r ^ { ( i ) } - \operatorname* { m e a n } ( \{ r ^ { ( 1 ) } , \ldots , r ^ { ( N ) } \} ) } { \mathrm { s t d } ( \{ r ^ { ( 1 ) } , \ldots , r ^ { ( N ) } \} ) } .\tag{A}
$$

The policy model $\pi _ { \theta }$ is then optimized by maximizing the objective function $J ( \theta )$ , which incorporates a clipped surrogate objective and a Kullback-Leibler (KL) divergence penalty to ensure stable training:

$$
\begin{array} { r } { J ( \theta ) = \mathbb { E } _ { q \sim Q , o ^ { ( i ) } \sim \pi _ { \theta _ { o l d } } } \left[ \operatorname* { m i n } \left( \rho ^ { ( i ) } \hat { A } ^ { ( i ) } , \mathrm { c l i p } ( \rho ^ { ( i ) } , 1 - \delta , 1 + \delta ) \hat { A } ^ { ( i ) } \right) - \beta \mathbb { D } _ { K L } \left( \pi _ { \theta } \| \pi _ { \mathrm { r e f } } \right) \right] , } \end{array}\tag{B}
$$

where $\begin{array} { r } { \rho ^ { ( i ) } = \frac { \pi _ { \theta } ( o ^ { ( i ) } | q ) } { \pi _ { \theta _ { o l d } } ( o ^ { ( i ) } | q ) } } \end{array}$ denotes the probability ratio, δ is the clipping threshold, $\beta$ controls the strength of the KL penalty, and $\pi _ { \mathrm { r e f } }$ is the frozen reference model.

## B.2 More Details of Rewards Design

To unify holistic quality perception with fine-grained spatial localization, we design a multitask reward system. The overall reward integrates global task constraints with a novel local visual grounding reward.

Global and Format Rewards. For fair comparison, we adopt the exact same format reward, score regression reward, and degradation perception reward as Q-Insight [18], detailed as follows:

– Format Reward: Ensures the response strictly adheres to the structured Chain-of-Thought format. The reward $r _ { \mathrm { f m t } } ^ { ( i ) } = 1$ if the reasoning process is properly enclosed within <think> tags and the final output is a valid JSON object within <answer> tags; otherwise, $r _ { \mathrm { f m t } } ^ { ( i ) } = \bar { 0 }$

– Score Regression Reward: For global scoring, we employ a binary reward based on absolute error. Let $s c r _ { \mathrm { p r e d } } ^ { ( i ) }$ be the predicted score and $s c r _ { \mathrm { g t } }$ be the ground-truth Mean Opinion Score (MOS). The reward $r _ { \mathrm { s c r } } ^ { ( i ) } = 1 \ \mathrm { i f } \ | s c r _ { \mathrm { p r e d } } ^ { ( i ) } - s c r _ { \mathrm { g t } } | < \epsilon ,$ where ϵ is a predefined acceptable margin; otherwise, 0.

Degradation Perception Reward: For global degradation perception, we assign $r _ { \mathrm { d e g } } ^ { ( i ) } = 1$ if the predicted distortion category matches the ground truth. Furthermore, an intensity reward $r _ { \mathrm { l e v } } ^ { ( i ) } = 1$ is granted only if both the distortion category and its severity level are correctly predicted.

Local Quality Assessment Rewards. Further details regarding the overall local reward structure are provided in Sec. 3.3 of the main paper. Here, we exclusively elaborate on the specific matching process and the calculation of the spatial metrics.

Let $B _ { p } = \{ b _ { p , 1 } , b _ { p , 2 } , \ldots , b _ { p , m } \}$ denote the set of m predicted bounding boxes, and $\begin{array} { r } { B _ { g t } = } \end{array}$ $\{ b _ { g , 1 } , b _ { g , 2 } , \dots , b _ { g , n } \}$ denote the set of n ground-truth bounding boxes, where each box $b \in \mathbb { R } ^ { 4 }$ To evaluate the spatial localization accuracy, we formulate the bounding box evaluation as an optimal bipartite matching problem. We construct a cost matrix $\mathcal { C } \in \mathbb { R } ^ { m \times n }$ based on the negative Distance Intersection over Union (DIoU) between every pair of predicted and ground-truth boxes:

$$
\mathcal C _ { i , j } = - \mathrm { D I o U } ( b _ { p , i } , b _ { g , j } ) , \quad \forall i \in \{ 1 , \ldots , m \} , j \in \{ 1 , \ldots , n \} .\tag{C}
$$

We then apply the Hungarian algorithm to this cost matrix to obtain the optimal bipartite matching $\pi ,$ , which assigns each predicted box to at most one ground-truth box. For the matched pairs $( i , \pi ( i ) )$ , we evaluate the matching quality using a predefined DIoU threshold τ (set to 0.5 in our experiments). A matched prediction is designated as a True Positive $( T P )$ if its DIoU with the corresponding ground-truth box is greater than or equal to τ :

$$
T P = \sum _ { i \in \mathrm { m a t c h e d } } \mathbb { I } ( \mathrm { D I o U } ( b _ { p , i } , b _ { g , \pi ( i ) } ) \geq \tau ) ,\tag{D}
$$

where $\mathbb { I } ( \cdot )$ is the indicator function. Consequently, the False Positives $( F P )$ and False Negatives $( F N )$ are strictly determined by the total number of predictions m and ground truths n:

$$
F P = m - T P , \quad F N = n - T P .\tag{E}
$$

Based on these definitions, the Precision (P) and Recall (R) for the spatial grounding are calculated as:

$$
\mathcal { P } = \frac { T P } { m } , \quad \mathcal { R } = \frac { T P } { n } .\tag{F}
$$

(Note: If $m = 0 , \mathcal { P }$ is defined as $0 ;$ if $n = 0$ , R is defined as 0). Finally, the discrete spatial matching metric, the F1 Score, is computed as the harmonic mean of Precision and Recall:

$$
\begin{array} { r } { F 1 = \left\{ \begin{array} { l l } { \frac { 2 \cdot \mathcal { P } \cdot \mathcal { R } } { \mathcal { P } + \mathcal { R } } , } & { \mathrm { i f ~ } \mathcal { P } + \mathcal { R } > 0 , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{G}
$$

This rigorous F1 Score metric comprehensively evaluates both the precise localization and the quantity alignment of the predicted local artifacts, providing dense and accurate reinforcement signals during the multi-task optimization.

## C Construction Details of the VGSR-2K Dataset

## C.1 Data Composition and Source Distributions

The proposed VGSR-2K dataset is meticulously constructed upon two foundational datasets: SRBench [4] and DifIQA [4]. Specifically, SRBench [4] provides a comprehensive collection of diverse natural scenes enhanced by various super-resolution (SR) architectures, encompassing regression-based, GAN-based, and difusion-based models, thereby ofering a rich spectrum of degradation patterns. Conversely, DifIQA [4] is explicitly designed to address the complex artifacts introduced by modern generative enhancement methods, such as hallucinated textures and structural distortions, operating under a relaxed perfect reference quality assumption.

To construct our dataset, we employ distinct sampling strategies for these two sources. For the DifIQA [4] subset, we randomly sample 798 high-resolution (HR) images in total. Among these, 698 images are processed through the original synthetic degradation and random enhancement pipeline to formulate 698 $[ I _ { L R } , I _ { S R } ]$ data pairs. The remaining 120 HR images undergo exclusively the degradation process to generate 120 pristine $[ I _ { L R } , I _ { H R } ]$ data pairs. Crucially, the explicit introduction of these $[ I _ { L R } , I _ { H R } ]$ pairs is designed for balancing positive and negative samples, which efectively prevents the policy model from exploiting trivial over-generation loopholes during reinforcement learning.

For the final detailed data ratios and distribution, please refer to Sec. 4.3 of the main paper.

## C.2 Qualitative Examples of Candidate Generation

As detailed in Sec. 4.2 of the main paper, the spatial ground truths of the VGSR-2K dataset are curated through a rigorous four-stage human-in-the-loop annotation pipeline. This subsection aims to provide qualitative visual examples, as illustrated in Fig. A, to explicitly demonstrate the output diversity of Stage 2 (MLLM-driven candidate generation). By leveraging two multi-modal foundation models (Qwen3-VL-Plus [2] and Seed-1.6-VL [8]) to generate multiple distinct proposals, this stage ensures a high degree of spatial diversity, thereby supplying rich and comprehensive candidate material for the subsequent rigorous human screening and cross-correction.

## D Additional Experimental Setup

In this section, we provide extended details regarding the experimental setups, including the evaluation benchmarks, the selection of baseline models, and the specific implementation configurations for our multi-task training framework.

## D.1 Evaluation Datasets

For the score regression task, we evaluate our model on diverse IQA benchmarks, which are categorized into four groups: (a) in-the-wild datasets including KonIQ [9], SPAQ [6] and LIVE-Wild [7]; (b) synthetic distortion datasets (KADID [19], CSIQ [16]); (c) modelprocessed distortions (PIPAL [12]); and (d) AI-generated images (AGIQA [17]). Following Q-Insight [18], we partition the KonIQ [9] dataset, utilizing approximately 7, 000 images for training, with the MOS normalized to the range of [1, 5]. The remaining datasets are exclusively used to evaluate the out-of-distribution (OOD) generalization capability of the models.

![](images/3525936435fdecc9cf9c9de3eb7ff0d265bea9caf0f332323e3e470a98cfe0fe.jpg)  
Fig. A: Qualitative illustration of the candidate generation process. This figure presents a given $[ I _ { R e f e r e n c e } , I _ { G e n e r a t e d } ]$ image pair alongside its corresponding six diferent candidate generation results. By providing such a diverse set of candidate bounding boxes, human annotators are empowered to conduct more comprehensive screening and rigorous cross-correction in the subsequent annotation stages.

For the degradation perception task, we utilize the DQ-495K [50] dataset. Specifically, we randomly sample 7, 000 images containing single distortions for training. Crucially, since the test set of Q-Insight [18] remains unreleased, we establish a fair evaluation benchmark by randomly sampling an additional 1, 000 non-overlapping and identically distributed images from DQ-495K to serve as our test set.

## D.2 Baseline Models

To thoroughly validate the efectiveness and comprehensiveness of our VGIQA framework on score regression task, we compare VGIQA with handcrafted methods including NIQE [13] and BRISQUE [22]; non-MLLM deep-learning methods including NIMA [36], MUSIQ [15], $\mathrm { C L I P - I Q A + \ [ 3 7 ] }$ , ManIQA [45]; and recent MLLM-based methods such as C2Score [62], Q-Align [41], DeQA-Score [48], Q-Insight [18], and a supervised fine-tuned Qwen [1]. For a fair comparison, all methods (except handcrafted ones) are trained on the KonIQ [9] dataset, and all MLLM-based methods utilize approximately 7B parameters.

## D.3 Implementation Details

During the cold-start phase, the foundation model (Qwen2.5-VL-7B-Instruct [1]) is trained for 3 epochs with a batch size of 16. In the subsequent GRPO phase, the model is jointly optimized for 10 epochs. Following Q-Insight [18], the generation number N is set to 8, the weight of the KL divergence penalty $\beta$ is set to $1 \times 1 0 ^ { - 3 }$ , the weights $\alpha _ { 1 }$ and $\alpha _ { 2 }$ are set to 0.25 and 0.75, respectively, and the threshold ϵ is set to 0.35. We employ the AdamW optimizer with an initial learning rate of $1 \times 1 0 ^ { - 6 }$ that linearly decays to $1 \times 1 0 ^ { - 9 }$ . Furthermore, the local reward weighting coeficient $\omega _ { \mathrm { l o c a l } }$ decays linearly from 5.0 to 1.0 during training. To balance the computational overhead under a constrained memory budget, we adopt an asymmetric batch size setting: 8 for the memory-intensive local quality assessment task, and 16 for both degradation perception and score regression tasks. All training of both phases is conducted on 8 compute nodes (80 GB each).

## E Designed Prompts

The prompts designed for each task in our framework are detailed in Table B. Specifically, the input for the score regression and degradation perception tasks consists of a task-specific prompt and a single evaluated image, with the expected outputs being a continuous quality score, and the distortion class alongside its severity level, respectively. For the local quality assessment task, the input includes a reference-generated image pair and a prompt, outputting precise spatial bounding boxes coupled with natural language reasoning for the localized regions.

To ensure fair comparisons with existing holistic evaluation methods, our system prompt, score regression prompt, and degradation perception prompt strictly follow the designs of Q-Insight [18]. Crucially, We introduce a new prompt for local quality assessment. By explicitly incorporating <think> and <answer> tags, this prompt guides the model through a rigorous “look and then judge” Chain-of-Thought (CoT) reasoning process. This explicit CoT mechanism efectively enhances both the interpretability of spatial grounding and the accuracy of final quality judgments.

Table B: Complete prompt templates used in our multi-task joint training framework. The system prompt, degradation perception prompt, and score regression prompt are directly adopted from Q-Insight [18], while the local quality assessment prompt is our newly designed core contribution.
<table><tr><td>System Prompt (Shared)</td></tr><tr><td>A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within &lt;think&gt; &lt;/think&gt; and &lt;answer&gt; &lt;/answer&gt; tags, respectively, i.e., &lt;think&gt; reasoning process here &lt;/think&gt;&lt;answer&gt;</td></tr><tr><td>answer here &lt;/answer&gt;. Prompt for Local Quality Assessment (Ours)</td></tr><tr><td>You are an expert in super-resolution (SR) image quality assessment. You are provided with: First image: Low-Resolution (LR) reference image (ground truth basis) Second image: Super-Resolved (SR) output image Your task is to carefully compare the two images to detect, localize, and evaluate all regions in</td></tr><tr><td>the SR image where the generation is problematic (e.g., hallucinations, distortions, or objective degradations). Evaluation Focus: For every problematic region identified, provide:</td></tr><tr><td>Localization: Precise bounding box coordinates that tightly enclose the problematic region. Reasoning: A concise natural-language explanation describing why this region is problematic (e.g., hallucinated texture, structural distortion, inconsistent semantics, or excessive blurring). Important Constraints:</td></tr><tr><td>Only report regions that are objectively degraded or hallucinated. Do not flag naturally blurry or ambiguous areas in the LR as &quot;errors&quot; unless the SR image exaggerates, fabricates, or distorts the content significantly. Process Requirements (Chain-of-Thought): Use a step-by-step reasoning process detailed within &lt;think&gt; and &lt;/think&gt; tags. Your reasoning should follow this flow:</td></tr></table>

Table B – continued from previous page
<table><tr><td rowspan=1 colspan=1>Assess per region: For each identified issue, describe the type of degradation and justify whyit constitutes a problem.Locate &amp; Describe: Determine the bounding box coordinates and write a concise reason foreach issue.Output Format: After your reasoning, in the &lt;answer&gt; tag, output a strict JSON objectwith a top-level key &quot;issues&quot; containing a list of issue dictionaries. Each dictionary must include:&quot;bbox 2d&quot;: The bounding box coordinates [x1, y1, x2, y2] of the issue.&quot;reason&quot;: A concise natural-language explanation of why this region is problematic.If the SR image contains no obvious problems or degradations compared to theLR image, the &quot;issues&quot; list must be empty (i.e., &quot;issues&quot;: [l).</td></tr><tr><td rowspan=1 colspan=1>Prompt for Degradation Perception</td></tr><tr><td rowspan=1 colspan=1>Analyze the given image and determine if it contains any of the following distortions: &quot;noise&quot;,&quot;compression&quot;, &quot;blur&quot;, or &quot;darken&quot;. If a distortion is present, classify its severity as &quot;slight&quot;,&quot;moderate&quot;, &quot;obvious&quot;, &quot;serious&quot;, or &quot;catastrophic&quot;. Return the result in JSON format with thefollowing keys: &quot;distortion class&quot;: The detected distortion (or &quot;null&quot; if none). and &quot;severity&quot;:The severity level (or &quot;null&quot; if none).</td></tr><tr><td rowspan=1 colspan=1>Prompt for Score Regression</td></tr><tr><td rowspan=1 colspan=1>What is your overall rating on the quality of this picture? The rating should be a float between 1and 5, rounded to two decimal places, with 1 representing very poor quality and 5 representingexcellent quality. Return the final answer in JSON format with the following keys: &quot;rating&quot;:The score.</td></tr></table>