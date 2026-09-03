# Test-Time Logit Prompting for Source-Free Missing Modality Adaptation

Taixi Chen Nancy Guo<sup>∗</sup>

School of Computing State University of New York at Binghamton {tchen51, nguo1}@binghamton.edu

## Abstract

Vision-language models (VLMs) have achieved remarkable performance by leveraging complementary information from large-scale image-text pairs. However, missing-modality inputs are commonly encountered during real-world deployment, often leading to significant performance degradation. Existing methods primarily enhance model robustness by learning modality compensation strategies from source training data. However, their reliance on source training data makes them difficult to apply when original data are unavailable due to privacy, storage, or accessibility constraints, such as clinical applications and personalized AI services. This raises an important yet underexplored question: can VLMs be eficiently adapted at test time for visual recognition with missing modalities without accessing source training data? To this end, we propose Test-Time Logit Prompting (TLP), a lightweight source-free test-time adaptation framework for visual recognition with missing modalities. To address missing-induced prediction shifts, TLP optimizes logit prompts with uncertainty-aware adjustment and modality-complete consistency regularization, adaptively adjusting prediction confidence while preserving semantic consistency. Extensive experiments across diverse vision-language benchmarks demonstrate that TLP consistently enhances recognition performance under missing-modality scenarios, achieving up to 8% improvements while requiring only hundreds of tunable parameters and a few test-time optimization steps.

## Introduction

Vision-language models (VLMs) have demonstrated remarkable performance by leveraging complementary information from large-scale image-text pairs (Radford et al. 2021), achieving impressive results across various vision-language tasks, including image captioning (Iashin and Rahtu 2020; Sarto et al. 2023), cross-modal retrieval (Radford et al. 2021; Kim, Son, and Kim 2021; Jiang and Ye 2023), and visual question answering (Zhang, Zhang, and Xu 2023; Sima et al. 2024). Despite these advances, real-world deployments frequently encounter missing-modality inputs caused by privacy constraints, sensor failures, or incomplete data collection (Lee et al. 2023; Hu et al. 2024). The absence of certain modalities can disrupt the learned cross-modal alignment in VLMs, resulting in significant performance degradation (Ma et al.

![](images/d5beacb5f28e704ced460478be9433712e6cce90f7bf0cabb5def9b8eca2b5ae.jpg)  
Figure 1: Comparison of the missing-modality adaptation paradigms. Existing source-dependent methods rely on train ing data for modality generation, prompt learning, or representation learning, whereas TLP adapts frozen VLMs at test time by optimizing lightweight logit prompts.

2022). Therefore, developing robust multimodal learning frameworks that can reliably handle diverse missing-modality scenarios has become a critical challenge.

As illustrated in Figure 1, existing methods mainly address this challenge through source-dependent strategies, including cross-modal generation (Wang, Cui, and Li 2023; Meng et al. 2024), joint representation learning (Ma et al. 2022; Wu et al. 2024b; Wang et al. 2023), and prompt learning (Lee et al. 2023; Hu et al. 2024; Zhang et al. 2025). These approaches have efectively improved model robustness by learning or recovering missing-modality information using source training data. Despite their efectiveness, this source-dependent paradigm assumes continuous access to source training data, which is often impractical in real-world deployment scenarios due to privacy restrictions, storage limitations, or data accessibility constraints (Liang, Hu, and Feng 2020; Xiao and Snoek 2024), especially in privacy-sensitive applications such as healthcare systems, personalized AI services, and edge devices.

This naturally raises an important question: can VLMs further improve missing-modality robustness at test time without accessing source training data? To answer this question, we propose Test-Time Logit Prompting (TLP), a lightweight source-free adaptation framework that directly adjusts the decision space of frozen VLMs during inference.

![](images/29075f501a5c09cf3844c4fd785c7735f3a806f101825edf82b2cb37c25ba62e.jpg)  
Figure 2: Overview of Test-Time Logit Prompting (TLP). Given a frozen VLM, TLP introduces missing-aware logit prompts to directly adapt the decision space under diferent missing-modality conditions. During inference, the corresponding logit prompt is selected according to modality availability and optimized through uncertainty-aware adjustment and modality-complete consistency regularization. The optimized prompt adjusts the original logits for final prediction, enabling eficient source-free missing-modality adaptation without updating the backbone.

Specifically, instead of adapting input-level prompts or model parameters with source training data, TLP introduces lightweight missing-aware prompts in the logit space and optimizes them online with frozen VLMs to dynamically adjust predictions under diferent missing-modality conditions. Considering that missing modalities can introduce both confidence bias and semantic inconsistency in the prediction space, TLP balances uncertainty adjustment and modality-complete consistency regularization through eficient logit-level adaptation, enabling reliable prediction adjustment while preserving semantic information from modality-complete references. Extensive experiments on diverse vision-language benchmarks demonstrate the efectiveness and eficiency of TLP under various missing-modality scenarios. TLP consistently improves recognition performance with only hundreds of tunable parameters and a few test-time optimization steps. Moreover, it can be readily integrated with existing training-based prompt learning methods to provide further performance gains. Our contributions are summarized as follows:

• We introduce a source-free test-time adaptation perspective for missing-modality recognition, enabling VLMs to improve during inference without accessing source training data.

• We propose Test-Time Logit Prompting (TLP), which enables decision-space adaptation by optimizing missingaware logit prompts with uncertainty-aware adjustment and modality-complete consistency regularization.

• Extensive experiments demonstrate that TLP achieves consistent improvements across diverse missing-modality scenarios with only hundreds of parameters and can seamlessly enhance existing training-based methods.

## Related Works

Missing-Modality in Multi-modal Learning. Despite the remarkable progress in multimodal learning (Yuan, Li, and Zhao 2025; Li et al. 2022; Guo and Gu 2025; Chen, Chen, and

Guo 2025; Chen, Cheung, and Zhang 2026; Chen and Cheung 2025), maintaining robustness under missing-modality scenarios remains challenging due to incomplete multimodal information and disrupted cross-modal alignment (Yuan, Li, and Zhao 2025; Wu et al. 2024a; Lee et al. 2023). Existing methods for missing-modality learning can be broadly categorized into cross-modal generation (Wang, Cui, and Li 2023; Meng et al. 2024; Lang et al. 2025), joint representation learning (Ma et al. 2022; Wu et al. 2024b; Wang et al. 2023), and prompt learning (Lee et al. 2023; Lu et al. 2025; Hu et al. 2024; Zhang et al. 2025; Chen and Guo 2026). Crossmodal generation methods recover missing information by reconstructing missing modalities from observed ones (Meng et al. 2024; Wang, Cui, and Li 2023), with recent eforts leveraging auxiliary knowledge from complete samples for improved recovery (Lang et al. 2025). Joint representation learning methods (Wu et al. 2024b; Wang et al. 2023) mitigate missing-modality efects by learning shared latent spaces or modality-invariant representations. Recently, prompt learning has emerged as an eficient alternative (Lee et al. 2023; Zhang et al. 2025), adapting models through learnable prompts with limited parameter updates. Despite their efectiveness, existing methods mainly enhance robustness during training and rely on source data. In contrast, TLP explores source-free test-time adaptation to improve missing-modality recognition during inference.

Test-Time Adaptation. Test-time adaptation (TTA) aims to improve model performance during inference by adapting models to test-time distributions without accessing source training data (Wang et al. 2020). Existing methods mainly address distribution shifts by updating normalization statistics (Schneider et al. 2020; Nado et al. 2020), optimizing model parameters (Lee et al. 2013), or adapting lightweight modules (Song et al. 2023). However, most existing TTA methods focus on modality-complete settings, leaving testtime adaptation under missing-modality scenarios largely unexplored. Unlike prior methods that adapt internal model components, TLP performs lightweight adaptation directly in the logit space, enabling eficient missing-modality recognition with frozen VLMs.

## Proposed Method

## Problem Formulation

We first formulate the source-free test-time adaptation setting for missing-modality recognition considered in this paper. Without loss of generality, we consider a vision-language model with two modalities, including image modality $m _ { 1 }$ and text modality $m _ { 2 }$ . Given a source-trained multimodal model $f _ { \theta } ,$ the original training dataset is inaccessible during adaptation, and only unlabeled test samples are available.

Under missing-modality scenarios, the test dataset can be represented as $\mathcal { D } _ { t } ~ = ~ \mathbf { \bar { \{ \mathcal { D } } _ { t } ^ { c } }  , \mathcal { D } _ { t } ^ { m _ { 1 } } , \mathcal { D } _ { t } ^ { m _ { 2 } } \big \}$ , where $\mathcal { D } _ { t } ^ { c } \ =$ $\{ ( x _ { i } ^ { m _ { 1 } } , x _ { i } ^ { m _ { 2 } } ) \} _ { i = 1 } ^ { N _ { c } }$ denotes samples with complete modalities, while $\bar { \mathcal D _ { t } ^ { m _ { 1 } } } \bar { = } \{ ( x _ { i } ^ { m _ { 1 } } ) \} _ { i = 1 } ^ { N _ { 1 } }$ and $\mathcal { D } _ { t } ^ { m _ { 2 } } = \{ ( x _ { i } ^ { m _ { 2 } } ) \} _ { i = 1 } ^ { N _ { 2 } }$ represent samples where only modality $m _ { 1 }$ or $m _ { 2 }$ is available, respectively. Following previous missing-modality methods (Lee et al. 2023; Hu et al. 2024), dummy inputs $\tilde { x } ^ { m _ { 1 } }$ and $\tilde { x } ^ { m _ { 2 } }$ (e.g., empty text or blank images) are introduced to maintain the input format of VLMs. Therefore, incomplete samples can be rewritten as $\tilde { \mathcal { D } } _ { t } ^ { m _ { 1 } } = \{ ( x _ { i } ^ { m _ { 1 } } , \tilde { x } ^ { m _ { 2 } } ) \} _ { i = 1 } ^ { N _ { 1 } }$ and $\mathcal { \tilde { D } } _ { t } ^ { m _ { 2 } } = \{ ( \tilde { x } ^ { m _ { 1 } } , x _ { i } ^ { m _ { 2 } } ) \} _ { i = 1 } ^ { N _ { 2 } }$ , forming the unified test-time dataset $\tilde { \mathcal { D } } _ { t } = \{ \mathcal { D } _ { t } ^ { c } , \tilde { \mathcal { D } } _ { t } ^ { m _ { 1 } } , \tilde { \mathcal { D } } _ { t } ^ { m _ { 2 } } \}$

Under this setting, the goal is to improve the performance of the source-trained model on incomplete test samples without accessing the original training data or test labels.

## Overall Pipeline

Figure 2 illustrates the overall framework of the proposed Test-Time Logit Prompting (TLP). Given a source-trained visionlanguage model, TLP aims to adapt model predictions under missing-modality scenarios while keeping the entire backbone frozen. Following common vision-language architectures, we adopt CLIP (Radford et al. 2021) as the backbone, which consists of separate image and text encoders. Specifically, the input image and text are first transformed into token sequences through their corresponding pre-trained embedding layers and then processed by frozen modality-specific encoders to obtain multimodal representations.

Unlike conventional adaptation methods that update input prompts or model parameters, TLP introduces lightweight missing-aware prompts in the logit space and only optimizes these decision-space parameters during inference:

$$
\mathcal { P } ^ { * } = \arg \operatorname* { m i n } _ { \mathcal { P } } \mathcal { L } ( \tilde { \mathcal { D } } _ { t } ; \boldsymbol { \theta } , \mathcal { P } ) ,\tag{1}
$$

where $\theta$ denotes the frozen VLM parameters, $\mathcal { P }$ represents the learnable logit prompt pool, and $\mathcal { L }$ is the source-free adaptation objective defined on unlabeled test samples. We next present the detailed design of our test-time logit prompting mechanism.

## Test-Time Logit Prompting

Given a test sample ${ \tilde { x } } _ { i }$ from the unified test dataset $\tilde { \mathcal { D } } _ { t }$ , which may contain complete or dummy-filled missing modalities, the frozen VLM produces the original prediction logits:

$$
z _ { i } = f _ { \theta } ( \tilde { x } _ { i } ) ,\tag{2}
$$

where $z _ { i } \in \mathbb { R } ^ { C }$ and C denotes the number of classes. Instead of adapting input representations or updating model parameters, TLP introduces lightweight missing-aware logit prompts to directly adjust the decision space. Specifically, we define a logit prompt pool $\mathcal { P } = \{ P _ { c } , P _ { m _ { 1 } } ^ { - } , P _ { m _ { 2 } } \}$ corresponding to complete, m<sub>1</sub>-available, and m<sub>2</sub>-available modality conditions, respectively. Each logit prompt $P \in \mathbb { R } ^ { C }$ contains class-wise learnable parameters for prediction adjustment. For each test sample, the corresponding prompt $\mathbf { p } _ { i } \in \mathcal { P }$ is selected according to its modality availability and added to the original logits:

$$
\hat { z } _ { i } = z _ { i } + \mathbf { p } _ { i } ,\tag{3}
$$

where $\hat { z } _ { i }$ represents the adjusted logits used for final prediction. During test-time adaptation, the backbone parameters θ remain frozen, and only the logit prompts in $\mathcal { P }$ are optimized.

## Source-Free Prompt Optimization

Since source data and test labels are unavailable during adaptation, optimizing the logit prompt pool $\mathcal { P }$ requires effective self-supervised objectives. To this end, we introduce uncertainty-aware adjustment and prototype-guided regularization to adapt predictions under diferent missing-modality conditions.

Missing modalities can lead to unreliable predictions by causing the model to produce over-confident outputs with limited modality information. To alleviate this issue, we encourage the adjusted predictions to maintain appropriate uncertainty by reducing their deviation from a uniform distribution. Specifically, given the adjusted logits $\hat { z } _ { i } ,$ the predicted probability distribution is computed as:

$$
q _ { i } = \mathrm { S o f t m a x } ( \hat { z } _ { i } ) .\tag{4}
$$

The uncertainty-aware adjustment objective is formulated as:

$$
\mathcal { L } _ { u } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { 1 } { C } \sum _ { j = 1 } ^ { C } ( q _ { i } ^ { j } - u ^ { j } ) ^ { 2 } ,\tag{5}
$$

where N denotes the number of test samples used for adaptation, and $\mathbf { u } \in \mathbb { R } ^ { C }$ denotes the uniform distribution over $C$ classes, with each element $\begin{array} { r } { u ^ { j } = \frac { 1 } { C } } \end{array}$

However, relying only on uncertainty adjustment may overlook the semantic structure learned by the original VLM, as pushing predictions toward a uniform distribution can weaken discriminative information. To preserve semantic consistency during adaptation, we introduce modality-complete consistency regularization by leveraging complete-modality samples as reliable anchors.

For each missing-modality sample, we first identify its nearest modality-complete neighbor within the test batch according to the original prediction distribution:

$$
k = \arg \operatorname* { m i n } _ { j \in \mathcal { D } _ { t } ^ { c } } d ( q _ { i } , q _ { j } ^ { c } ) ,\tag{6}
$$

where $q _ { j } ^ { c }$ denotes the prediction distribution of a completemodality sample. The selected complete-modality prediction is used as the semantic anchor:

$$
q _ { i } ^ { a } = q _ { k } ^ { c } .\tag{7}
$$

Then, we minimize the distribution discrepancy between the adjusted prediction and its corresponding anchor:

$$
\mathcal { L } _ { c } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { C } ( q _ { i } ^ { a } ) ^ { j } \log \frac { ( q _ { i } ^ { a } ) ^ { j } } { q _ { i } ^ { j } } .\tag{8}
$$

To balance the complementary objectives of uncertainty adjustment and modality-complete consistency, the overall optimization objective of TLP is defined as:

$$
\mathcal { L } = \mathcal { L } _ { u } + \alpha \mathcal { L } _ { c } , \quad \alpha = \frac { 1 } { C } ,\tag{9}
$$

where C denotes the number of classes. The weighting factor α is introduced to adaptively control the contribution of the modality-complete consistency constraint. As the number of classes increases, the prediction space becomes more complex, making nearest-neighbor matching more challenging and potentially reducing the reliability of selected anchors. Therefore, setting $\alpha = { \textstyle { \frac { 1 } { C } } }$ reduces the influence of potentially unreliable anchors while maintaining efective semantic guidance from modality-complete samples. During test-time adaptation, only the logit prompt pool $\bar { \mathcal P }$ is optimized by minimizing ${ \mathcal { L } } ,$ while all parameters of the VLM remain frozen, resulting in eficient and lightweight adaptation

After test-time optimization, the learned logit prompts are applied to adjust predictions according to the modality condition of each test sample. Specifically, the corresponding prompt $\mathbf { p } _ { i } \in \mathcal { P }$ is selected and added to the original logits to obtain the final prediction:

$$
\hat { y } _ { i } = \arg \operatorname* { m a x } _ { c } \mathrm { S o f t m a x } ( \hat { z } _ { i } ) ^ { c } .\tag{10}
$$

Since TLP only optimizes class-level logit prompts while keeping the entire VLM frozen, the number of trainable parameters is only $| { \mathcal { P } } | \times C ,$ , enabling eficient adaptation with minimal computational overhead.

## Experiment

## Experiment Settings

Datasets. Following previous missing-modality learning works (Lee et al. 2023; Hu et al. 2024; Zhang et al. 2025), we evaluate our method on three widely used vision-language benchmark datasets.

MM-IMDb (Arevalo et al. 2017) is a multimodal movie genre classification dataset containing 25,959 samples with paired movie posters and textual descriptions. Since each movie can belong to multiple genres, it is formulated as a multi-label classification task.

UPMC Food-101 (Wang et al. 2015) contains noisy imagetext pairs collected from the web across 101 food categories, which is used for multimodal food classification.

Hateful Memes (Kiela et al. 2020) is a multimodal hateful content detection dataset consisting of over 10,000 image-text pairs, where both visual and textual information are required for accurate classification.

Implementation Details. We adopt CLIP (Radford et al. 2021) with a ViT-B/16 (Dosovitskiy et al. 2020) image encoder as the vision-language backbone. Input images are resized to $2 2 4 \times 2 2 4$ , and text inputs are processed using the CLIP tokenizer with a maximum sequence length of 77. Following previous missing-modality settings (Lee et al. 2023; Hu et al. 2024; Zhang et al. 2025), unavailable modalities are replaced with dummy inputs to maintain a consistent multimodal input format. During test-time adaptation, all parameters of the image encoder, text encoder, and task head remain frozen, and only the proposed logit prompt pool $\mathcal { P }$ is optimized. For each missing-modality condition, we introduce a classlevel logit prompt with dimension $C ,$ resulting in only 3C trainable parameters. We optimize the logit prompts using the AdamW optimizer with a learning rate of $1 \times \dot { 1 } 0 ^ { - 2 }$ and perform adaptation for K optimization steps. For integration with training-based prompt learning methods, TLP is directly applied to their trained models without modifying the original training procedures. All experiments are conducted on a single NVIDIA RTX A6000 GPU with a batch size of 32.

Table 1: Comparison with the baseline on MM-IMDb (Arevalo et al. 2017), UPMC Food-101 (Wang et al. 2015), and Hateful Memes (Kiela et al. 2020) under diferent missing-modality conditions with missing rate $\eta = 7 0 \%$ . “Time (s)” denotes the total adaptation time over the test set with one optimization step. Ours (1) and Ours (5) indicate TLP with 1 and 5 test-time optimization steps, respectively. The best results are highlighted in bold.
<table><tr><td rowspan="2">Datasets</td><td rowspan="2">Missing rate η</td><td colspan="2">Train</td><td colspan="2">Test</td><td rowspan="2">Time (s)</td><td rowspan="2">Baseline</td><td rowspan="2">Ours (1)</td><td rowspan="2">Ours (5)</td></tr><tr><td>Image</td><td>Text</td><td>Image Text</td><td></td></tr><tr><td rowspan="2">MM-IMDb (F1-Macro)</td><td rowspan="2">70%</td><td>100%</td><td>30%</td><td>100%</td><td>30%</td><td>52.45</td><td>44.76</td><td>46.91</td><td>53.05</td></tr><tr><td>30%</td><td>100%</td><td>30%</td><td>100%</td><td>50.55</td><td>48.78</td><td>51.11</td><td>53.75</td></tr><tr><td rowspan="2">Food101</td><td rowspan="2">70%</td><td>65%</td><td>65%</td><td>65%</td><td>65%</td><td>48.90</td><td>46.41</td><td>48.62</td><td>52.41</td></tr><tr><td>100%</td><td>30%</td><td>100%</td><td>30%</td><td>87.21</td><td>72.97</td><td>73.78</td><td>75.45</td></tr><tr><td rowspan="3">(Accuracy) Hateful</td><td rowspan="3"></td><td>30%</td><td>100%</td><td>30%</td><td>100%</td><td>85.43</td><td>78.37</td><td>79.55</td><td>80.88</td></tr><tr><td>65%</td><td>65%</td><td>65%</td><td>65%</td><td>90.85</td><td>74.65</td><td>75.68</td><td>77.05</td></tr><tr><td>100%</td><td>30%</td><td>100%</td><td>30%</td><td>2.95</td><td>61.49</td><td>62.26</td><td>64.51</td></tr><tr><td rowspan="2">Memes (AUROC)</td><td rowspan="2">70%</td><td>30%</td><td>100%</td><td>30%</td><td>100%</td><td>3.05</td><td>60.51</td><td>60.98</td><td>62.07</td></tr><tr><td>65%</td><td>65%</td><td>65%</td><td>65%</td><td>2.81</td><td>61.52</td><td>61.80</td><td>62.61</td></tr></table>

![](images/75b83ac3793f37c1032f7c17f2323a82b37fb5b6ee88154e2080c6d4ca36ea23.jpg)  
(a) Missing Text

![](images/532f8fa7f325e9f2a52d9e560c0175beb1c9ad9e4a17cb829d9ba9be82b2123a.jpg)  
(b) Missing Image

![](images/dba2330045995700c88bf993ba4f2fd5e6d17274651b8613915c4c265c70ab7f.jpg)  
(c) Missing Both  
Figure 3: Generalizability analysis of TLP under diferent test-time missing rates on the MM-IMDb dataset (Arevalo et al. 2017). The source model is trained with complete modalities and evaluated under varying missing-modality ratios, where TLP is directly applied during inference without accessing source data. (a) Missing-text evaluation. (b) Missing-image evaluation. (c) Mixed missing evaluation with both text-only and image-only samples.

Missing Modality Setting. Following previous missingmodality learning works (Lee et al. 2023; Hu et al. 2024; Zhang et al. 2025), we evaluate models under diferent modality-missing scenarios. The missing rate η is defined as the proportion of samples with incomplete modalities. For vision-language tasks with image and text modalities, we consider three settings: missing-text, missing-image, and mixed missing. In the missing-text or missing-image setting, η of samples contain only one available modality, while the remaining (1 − η) samples retain complete modalities. In the mixed missing setting, <sup>η</sup> of samples are text-only, <sup>η</sup> are image-only, and the remaining (1 − η) samples contain both modalities. This protocol can be extended to scenarios with M modalities, where each incomplete case is sampled proportionally based on $\frac { \eta } { M ^ { 2 } - 2 }$ , while (1 − η) of samples remain modality-complete.

Baselines and Metrics. We evaluate TLP under two settings. First, we apply TLP to the frozen VLM backbone to validate its efectiveness for source-free test-time adaptation. Second, to evaluate its plug-and-play capability, we integrate TLP with representative training-based missing-modality prompt learning methods, including: (1) MMP (Lee et al. 2023), which introduces missing-aware prompts to enhance multimodal representations; (2) DCP (Hu et al. 2024), which models prompt correlations and cross-layer interactions for robust feature learning; and (3) SyP (Zhang et al. 2025), which combines instance-aware features with learnable prompts through an adaptive prompting mechanism. Following prior works (Lee et al. 2023; Hu et al. 2024; Zhang et al. 2025), we report F1-Macro for multi-label classification on MM-IMDb (Arevalo et al. 2017), top-1 classification accuracy for recognition performance on UPMC Food-101 (Wang et al. 2015), and Area Under the Receiver Operating Characteristic Curve (AUROC) for hateful content detection on Hateful Memes (Kiela et al. 2020).

## Experiment Results

Main results of comparison with baseline. We first focus on studying the robustness of VLMs against modality incompleteness at test time without accessing source training data. The primary baseline is the source-trained VLM obtained under each missing-rate setting, which is directly evaluated on test samples with the same missing rate without any test-time adaptation. Since TLP keeps the entire backbone frozen and only optimizes lightweight logit prompts during inference, the performance improvement over this baseline directly reflects the benefit of source-free decision-space adaptation. As shown in Table 1, TLP consistently improves the performance of source-trained VLMs across diferent missing-modality scenarios without accessing source data. With only one test-time optimization step, TLP already brings clear improvements over the baseline (e.g., from 44.76% to 46.91%) while requiring only 3-90 seconds for adaptation. Increasing the optimization steps to five further enhances the performance, achieving improvements of up to 8% compared with the non-adapted baseline.

![](images/01575ac85fdbcda6a1235c77961d02693a99c5c6968b53cadb0cb0a20d1b946c.jpg)  
(a) Missing Text

![](images/1901f423569be667aba14f9c8ddcb8584421c98b8d884670d634b119ff0120b1.jpg)  
(b) Missing Image

![](images/52dc2c589cfd13eeb20ea3729907dc5970648e8fa7d1b7bf9e9d0bf401f9a6d8.jpg)  
(c) Missing Both

![](images/9639fb9cc3bfb098b5c3a71ff3cef40fe63ba0f8a16beb3f9c87a4b53c1c5e97.jpg)  
(d) Missing Text

![](images/ce6fdcf8bfa5c71076e9908849fb96fb7e931dc6c065763c381ae68dcadfbae9.jpg)  
(e) Missing Image

![](images/0b6a3cbd689aa08afa8cd46efe45ccef4d77be16390fdc4450c40ce71c3ff5d3.jpg)  
(f) Missing Both

![](images/50a4d67ea4aa338aee2b44d0b45c491884471f599f6368a5c5a3cbf9f1aaedfc.jpg)  
(g) Missing Text

![](images/96c7728b2fac564a55cf911b9a3e782d9d168385c455aeae6a7c619f35d65c8a.jpg)  
(h) Missing Image

![](images/97bfbc9d7490736e611b14d94d105abc1f911dce9e6152141953a592bd4c0fc4.jpg)  
(i) Missing Both  
Figure 4: Plug-and-play evaluation of TLP with existing missing-modality methods on MM-IMDb (Arevalo et al. 2017). MMP (Lee et al. 2023), DCP (Hu et al. 2024), and SyP (Zhang et al. 2025) are trained and evaluated under corresponding missing-modality settings, while TLP is directly applied during inference without retraining.

These results demonstrate that TLP can eficiently enhance the missing-modality robustness of VLMs through lightweight test-time adaptation. Notably, TLP achieves more substantial improvements on the challenging MM-IMDb dataset, where complex multi-label prediction requires efective utilization of incomplete multimodal information, further highlighting the benefit of decision-space adaptation.

Generalization Analysis. To evaluate the generalization ability of TLP under changing missing-modality conditions, we conduct experiments where the source model is trained with complete modalities and evaluated under diferent testtime missing rates on the MM-IMDb dataset. As shown in Figure 3, TLP consistently outperforms the baseline across all missing scenarios, including missing text, missing image, and mixed missing settings, with missing rates ranging from 10% to 90%.

The performance gains demonstrate that TLP can effectively adapt complete-trained VLMs to unseen missingmodality conditions without accessing source data or updating model parameters. Notably, missing text causes more severe performance degradation on MM-IMDb, indicating that textual information plays a critical role in this dataset. Even under this challenging scenario, TLP achieves substantial improvements over the baseline and maintains consistent gains as the missing rate increases. These results demonstrate that decision-space adaptation efectively improves the robustness of VLMs against varying degrees of modality incompleteness.

Plug-and-play Evaluation. To further evaluate the compatibility of TLP with existing missing-modality learning methods, we integrate TLP with representative training-based prompt learning approaches, including MMP (Lee et al. 2023), DCP (Hu et al. 2024), and SyP (Zhang et al. 2025). Since these methods also adopt prompting mechanisms for missingmodality adaptation, they provide suitable testbeds for evaluating the plug-and-play capability of TLP. These methods are first trained under corresponding missing-modality settings, and TLP is directly applied during inference without modifying their original training procedures.

As shown in Figure 4, incorporating TLP consistently improves the performance of all compared methods across different missing scenarios and missing rates. This demonstrates that TLP can provide complementary benefits to existing training-based approaches by further adapting predictions at test time. Notably, the improvements remain stable and even become more pronounced as the missing rate increases, indicating that lightweight logit-level adaptation efectively enhances robustness under severe modality incompleteness. These results highlight the potential of TLP as a general plugand-play test-time adaptation framework for missing-modality recognition.

Ablation Analysis. We conduct ablation experiments to investigate the contribution of each optimization objective in TLP. As shown in Table 2, removing both objectives leads to inferior performance, demonstrating the necessity of source-free optimization for adapting logit prompts under missing-modality scenarios. Introducing either uncertaintyaware adjustment loss $\mathcal { L } _ { u }$ or modality-complete consistency loss $\mathcal { L } _ { c }$ improves the performance, validating the efectiveness of both components.

Specifically, $\mathcal { L } _ { u }$ provides significant improvements on MM-IMDb (Arevalo et al. 2017) and Food101 (Wang et al. 2015) by adjusting prediction uncertainty and alleviating biased confidence caused by incomplete modality information. Meanwhile, $\mathcal { L } _ { c }$ efectively preserves semantic consistency by aligning missing-modality predictions with their nearest modality-complete references, leading to clear improvements especially on Hateful Memes (Kiela et al. 2020). By jointly optimizing these two objectives, TLP achieves the best performance across all benchmarks, demonstrating that uncertainty adjustment and semantic consistency provide complementary guidance for source-free logit prompt adaptation.

Eficiency. We further analyze the eficiency of TLP in terms of trainable parameters and adaptation cost. Unlike existing prompt learning methods that optimize input-level prompts or additional modules, TLP keeps the entire VLM backbone frozen and only updates the logit prompt pool $\mathcal { P }$ during inference. Since each logit prompt contains only C class-wise parameters, TLP introduces merely 3C learnable parameters for diferent missing-modality conditions. As shown in Table 1, TLP achieves consistent improvements with only a few test-time optimization steps. Even with a single adaptation step, TLP efectively improves recognition performance while requiring only limited adaptation time.

![](images/6d95a24aefc4c8169980300db1c164bc95c324f2caab354afd3a784c4d532400.jpg)  
(a) Missing Text

![](images/998bcbd42c74f923cb7285e698835feb10005766909d2c7bebf95d94394f87d7.jpg)  
(b) Missing Both

Figure 5: Visualization analysis of TLP-adapted logit distributions on Food101 (Wang et al. 2015) using t-SNE under diferent missing-modality scenarios.
<table><tr><td> $L _ { u }$ </td><td> $L _ { c }$ </td><td>MM-IMDb</td><td>Food101</td><td>Hateful Memes</td></tr><tr><td> $\checkmark$ </td><td> $\checkmark$ </td><td>52.41</td><td>76.98</td><td>62.61</td></tr><tr><td rowspan="3"> $\checkmark$ </td><td></td><td>52.26</td><td>76.77</td><td>61.55</td></tr><tr><td> $\checkmark$ </td><td>46.74</td><td>75.03</td><td>62.50</td></tr><tr><td></td><td>46.41</td><td>74.65</td><td>61.52</td></tr></table>

Table 2: Ablation study of diferent optimization objectives in TLP on three benchmarks. Bold represents the best results.

These results demonstrate that TLP provides an eficient source-free adaptation solution for missing-modality recognition with minimal computational overhead. For example, TLP completes one-step adaptation within 3-90 seconds across diferent datasets while achieving noticeable gains.

Visualization Analysis. As shown in Figure 5, we visualize the TLP-adapted logit distributions using t-SNE. Under different missing-modality scenarios, the adapted logits exhibit clear clustering structures in the decision space, indicating that TLP efectively preserves discriminative information through lightweight logit-level adaptation without updating the VLM backbone.

## Conclusion

In this paper, we investigate source-free test-time adaptation for missing-modality recognition in vision-language models. We propose Test-Time Logit Prompting (TLP), a lightweight adaptation framework that improves frozen VLMs by optimizing missing-aware prompts directly in the logit space without accessing source training data. To achieve reliable prediction adjustment, TLP introduces uncertainty-aware adjustment and modality-complete consistency regularization, enabling efective adaptation under diferent missing-modality conditions while preserving semantic information. Extensive experiments on multiple vision-language benchmarks demonstrate that TLP consistently enhances missing-modality robustness with only a few optimization steps and hundreds of tunable parameters. Furthermore, TLP can be seamlessly integrated with existing training-based methods, highlighting its potential as an eficient plug-and-play framework for real-world incomplete multimodal scenarios.

## References

Arevalo, J.; Solorio, T.; Montes-y Gómez, M.; and González, F. A. 2017. Gated multimodal units for information fusion. arXiv preprint arXiv:1702.01992.

Chen, T.; Chen, J.; and Guo, N. 2025. UAM: A Unified Attention-Mamba Backbone of Multimodal Framework for Tumor Cell Classification. arXiv preprint arXiv:2511.17355.

Chen, T.; and Cheung, Y.-m. 2025. TYrPPG: Uncomplicated and Enhanced Learning Capability rPPG for Remote Heart Rate Estimation. arXiv preprint arXiv:2511.05833.

Chen, T.; Cheung, Y.-m.; and Zhang, Y. 2026. CADM: Cluster-Customized Adaptive Distance Metric for Categorical Data Clustering. In ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2561–2565. IEEE.

Chen, T.; and Guo, N. 2026. Learning from Reliable Latent Prompts for Visual Recognition with Missing Modalities. arXiv preprint arXiv:2606.30597.

Dosovitskiy, A.; Beyer, L.; Kolesnikov, A.; Weissenborn, D.; Zhai, X.; Unterthiner, T.; Dehghani, M.; Minderer, M.; Heigold, G.; Gelly, S.; et al. 2020. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.

Guo, Y.; and Gu, X. 2025. Mmrl: Multi-modal representation learning for vision-language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, 25015–25025.

Hu, L.; Shi, T.; Feng, W.; Shang, F.; and Wan, L. 2024. Deep correlated prompting for visual recognition with missing modalities. Advances in Neural Information Processing Systems, 37: 67446–67466.

Iashin, V.; and Rahtu, E. 2020. Multi-modal dense video captioning. In Proceedings ofthe IEEE/CVF conference on computer vision andpattern recognition workshops, 958–959.

Jiang, D.; and Ye, M. 2023. Cross-modal implicit relation reasoning and aligning for text-to-image person retrieval. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2787–2797.

Kiela, D.; Firooz, H.; Mohan, A.; Goswami, V.; Singh, A.; Ringshia, P.; and Testuggine, D. 2020. The hateful memes challenge: Detecting hate speech in multimodal memes. Advances in neural information processing systems, 33: 2611– 2624.

Kim, W.; Son, B.; and Kim, I. 2021. Vilt: Vision-and-language transformer without convolution or region supervision. In International conference on machine learning, 5583–5594. PMLR.

Lang, J.; Hong, R.; Cheng, Z.; Zhong, T.; Wang, Y.; and Zhou, F. 2025. REDEEMing Modality Information Loss: Retrieval-Guided Conditional Generation for Severely Modality Missing Learning. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, 1241–1252.

Lee, D.-H.; et al. 2013. Pseudo-label: The simple and eficient semi-supervised learning method for deep neural networks. In Workshop on challenges in representation learning, ICML, volume 3, 896. Atlanta.

Lee, Y.-L.; Tsai, Y.-H.; Chiu, W.-C.; and Lee, C.-Y. 2023. Multimodal prompting with missing modalities for visual recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 14943–14952.

Li, L. H.; Zhang, P.; Zhang, H.; Yang, J.; Li, C.; Zhong, Y.; Wang, L.; Yuan, L.; Zhang, L.; Hwang, J.-N.; et al. 2022. Grounded language-image pre-training. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 10965–10975.

Liang, J.; Hu, D.; and Feng, J. 2020. Do we really need to access the source data? source hypothesis transfer for unsupervised domain adaptation. In International conference on machine learning, 6028–6039. PMLR.

Lu, A.; Li, C.; Zhao, J.; Tang, J.; and Luo, B. 2025. Modalitymissing RGBT tracking: Invertible prompt learning and highquality benchmarks. International Journal of Computer Vision, 133(5): 2599–2619.

Ma, M.; Ren, J.; Zhao, L.; Testuggine, D.; and Peng, X. 2022. Are multimodal transformers robust to missing modality? In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 18177–18186.

Meng, X.; Sun, K.; Xu, J.; He, X.; and Shen, D. 2024. Multi-modal modality-masked difusion network for brain mri synthesis with random modality missing. IEEE Transactions on Medical Imaging, 43(7): 2587–2598.

Nado, Z.; Padhy, S.; Sculley, D.; D’Amour, A.; Lakshminarayanan, B.; and Snoek, J. 2020. Evaluating predictiontime batch normalization for robustness under covariate shift. arXiv preprint arXiv:2006.10963.

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning, 8748–8763. PmLR.

Sarto, S.; Barraco, M.; Cornia, M.; Baraldi, L.; and Cucchiara, R. 2023. Positive-augmented contrastive learning for image and video captioning evaluation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 6914–6924.

Schneider, S.; Rusak, E.; Eck, L.; Bringmann, O.; Brendel, W.; and Bethge, M. 2020. Improving robustness against common corruptions by covariate shift adaptation. Advances in neural information processing systems, 33: 11539–11551.

Sima, C.; Renz, K.; Chitta, K.; Chen, L.; Zhang, H.; Xie, C.; Beißwenger, J.; Luo, P.; Geiger, A.; and Li, H. 2024. Drivelm: Driving with graph visual question answering. In European conference on computer vision, 256–274. Springer.

Song, J.; Lee, J.; Kweon, I. S.; and Choi, S. 2023. Ecotta: Memory-eficient continual test-time adaptation via selfdistilled regularization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 11920–11929.

Wang, D.; Shelhamer, E.; Liu, S.; Olshausen, B.; and Darrell, T. 2020. Tent: Fully test-time adaptation by entropy minimization. arXiv preprint arXiv:2006.10726.

Wang, H.; Chen, Y.; Ma, C.; Avery, J.; Hull, L.; and Carneiro, G. 2023. Multi-modal learning with missing modality via shared-specific feature modelling. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 15878–15887.

Wang, X.; Kumar, D.; Thome, N.; Cord, M.; and Precioso, F. 2015. Recipe recognition with large multimodal food dataset. In 2015 IEEE International Conference on Multimedia & Expo Workshops (ICMEW), 1–6. IEEE.

Wang, Y.; Cui, Z.; and Li, Y. 2023. Distribution-consistent modal recovering for incomplete multimodal learning. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, 22025–22034.

Wu, R.; Wang, H.; Chen, H.-T.; and Carneiro, G. 2024a. Deep multimodal learning with missing modality: A survey. arXiv preprint arXiv:2409.07825.

Wu, Z.; Zheng, J.; Ren, X.; Vasluianu, F.-A.; Ma, C.; Paudel, D. P.; Van Gool, L.; and Timofte, R. 2024b. Single-model and any-modality for video object tracking. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 19156–19166.

Xiao, Z.; and Snoek, C. G. 2024. Beyond model adaptation at test time: A survey. arXiv preprint arXiv:2411.03687.

Yuan, Y.; Li, Z.; and Zhao, B. 2025. A survey of multimodal learning: Methods, applications, and future. ACM Computing Surveys, 57(7): 1–34.

Zhang, X.; Zhang, F.; and Xu, C. 2023. Vqacl: A novel visual question answering continual learning setting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 19102–19112.

Zhang, Z.; Dai, L.; Lin, Q.; Diao, Y.; Jin, G.; Guo, Y.; Zhang, J.; and Hao, X. 2025. Synergistic prompting for robust visual recognition with missing modalities. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, 1881–1890.