# Localizing Concepts in Visual Autoregressive Models

Nanxiang Jiang <sup>1</sup> Yang Liu <sup>2</sup> Yuanhao Wang <sup>2</sup> Min Xu <sup>2∗</sup>

<sup>1</sup>University of California, Los Angeles <sup>2</sup>Carnegie Mellon University

## Abstract

Understanding the internal routing of visual concepts is crucial for the safe and controllable adaptation of generative models. While concept localization has been widely studied in diffusion models, the emerging paradigm of next-scale visual autoregressive models remains largely unexplored. In this paper, we introduce LoCo, the first model-agnostic framework to localize where and when specific concepts emerge within autoregressive models. Driven by the native coarse-to-fine nature of next-scale generation, our method precisely maps conceptual knowledge across three distinct dimensions: Layer, Scale, and Position. To systematically localize and evaluate concept routing without context bias, we propose LoCoBench, a comprehensive dataset spanning 10 diverse categories. Extensive probing on image and video autoregressive models like Infinity, HunyuanImage-3.0, and InfinityStar shows that the localized positions are both interpretable and causally related to concept emergence. Building on these insights, we apply our localization method to three key applications: concept erasure, model personalization, and adversarial concept injection. Experiments demonstrate that our targeted intervention achieves state-of-the-art performance, substantially reducing computational overhead while preserving benign utility. Overall, our findings offer insights into how conceptual knowledge is routed during autoregressive generation, introducing a practical pathway for more interpretable, efficient, and secure adaptation.

## 1 Introduction

The landscape of text-to-image (T2I) and text-to-video (T2V) generation has evolved at breakneck speed. The field rapidly transitioned from foundational U-Net architectures [48, 27] to the highly scalable era of Diffusion Transformers (DiTs) [45, 36]. Recently, Visual Autoregressive (VAR) models [53, 25, 6, 40] have emerged as a new paradigm. By treating visual generation as a next-scale coarse-to-fine prediction task [53], VAR models have achieved remarkable success and set new standards for generation quality.

With access to such powerful pretrained models, it is crucial to explore their potential for applications beyond mere generation. A growing body of work has focused on localizing different types of concept and knowledge within these models to enable more targeted usage [27, 54, 34]. For example, prior studies demonstrate that cross-attention layers are key to incorporating prompt compositional information, while structural information is often concentrated in the self-attention modules of UNet-based architectures [27, 39]. Such internal understanding plays a critical role in practical applications. Several works [20, 22, 33] have shown that generative models often memorize unsafe or unwanted content (e.g., Not-Safe-For-Work (NSFW) or copyrighted content), and designed methods for concept erasure, which aim to forget these specific targets while preserving overall generation utility. Conversely, tasks like model personalization [49] and adversarial concept injection [31] require injecting novel subjects into diverse scenes using minimal reference data. In both scenarios, localizing concepts within the model is crucial for enabling targeted interventions that make finetuning highly efficient while preserving the model’s prior capabilities and overall generation quality.

![](images/b9ec3523195fe4f4ba1d5bd20a60d0cc944a9c256b693dfaa4dd21957c8cd7c3.jpg)  
Figure 1: Localization and applications in VAR models. Left: Heatmaps indicate the frequency of specific Layer-Scale-Position triplets being selected as dominant carriers of target concepts based on attention responses. Right: This precise localization enables highly efficient targeted interventions for concept erasure, personalization, and adversarial injection.

While extensive research has explored concept localization in UNet and DiT architectures [64, 27, 54, 66], emerging VAR models remain largely underexplored. Unlike classical generative models, VAR models construct visual representations progressively from coarse, low-resolution scales to fine, high resolution details. Therefore, we find that directly applying layer-wise localization methods designed for diffusion models [64, 4] proves highly sub-optimal, as they ignore this dynamic multi-scale mechanism, leading to poor localization precision. Furthermore, our probing reveals a non-trivial phenomenon: concepts do not reside in a single neural layer; they first anchor at critical coarse scales and manifest only at sparse spatial query positions. Capturing this complex emergence requires a joint, multi-dimensional approach.

To bridge this gap, we introduce LoCo (Localizing Concepts), an automatic and model-agnostic localization framework tailored for VAR models (Figure 1). Driven by the progressive nature of next-scale generation, LoCo precisely maps conceptual knowledge by analyzing internal attention dynamics. Specifically, it computes the attention response for each spatial query position throughout the generation process to pinpoint the Top-K most critical (Layer, Scale, Position) triplets responsible for the target concept. By exclusively intervening on these highly localized routes, we can execute targeted downstream editing while leaving the vast majority of model parameters untouched.

Existing public datasets (e.g., I2P [50]) focus on single tasks and lack the structured, paired prompts required to isolate concepts and eliminate context bias. To bridge this gap, we propose LoCoBench, a large-scale probing dataset explicitly designed to evaluate concept routing, encompassing 10 diverse categories, 1,467 entities, and over 60K prompts. Extensive experiments on state-of-the-art models (Infinity [25], HunyuanImage-3.0 [6], and InfinityStar [40]) demonstrate that LoCo consistently outperforms full-parameter fine-tuning across downstream applications. Notably, strictly intervening on just the top 5% of localized positions yields superior task performance and minimal contextual interference, while reducing memory usage by 38.5% and training time by 82.9%.

To the best of our knowledge, we are the first to systematically investigate concept localization in VAR models. In summary, our main contributions are:

• We introduce LoCo, a novel localization framework specifically designed for the progressive generation dynamics of VAR models. It precisely identifies where concepts emerge by locating the exact Layer, Scale, and Position dimensions.

• We present LoCoBench, a large-scale probing dataset comprising 10 categories, 1,467 entities, and over 60K paired prompts. It features rigorous prompt shuffling to eliminate context bias, enabling robust localization and evaluation of concept routing.

• Building on this localization, we demonstrate practical applications for concept erasure, model personalization, and adversarial injection. Our ultra-sparse targeted fine-tuning achieves state-ofthe-art task performance, reducing memory usage by 38.5% and training time by 82.9%.

## 2 Related Work

## 2.1 Visual Autoregressive Generation Models

To unify multimodal understanding and generation within a single framework, autoregressive models first turn images and videos into discrete tokens [55]. Early models generate these tokens in a flat 1D raster order, much like reading text [47, 63, 16]. Later systems scale this 1D pipeline for powerful, unified visual generation [9, 13, 58, 59]. However, flattening spatial data creates bottlenecks in image quality and efficiency. Recently, VAR models solve this by predicting the next scale instead of the next token [53]. This coarse-to-fine process drastically improves generation quality and enables new control methods [37, 62, 68]. Building on this, Infinity [25] and HunyuanImage-3.0 [6] achieve state-of-the-art T2I generation, while InfinityStar [40] extends to T2V generation. We select these three recent models to represent the next-scale autoregressive family and demonstrate our method.

## 2.2 Interpretability of Generative Models

A rich literature explores the interpretability of generative models for storing concepts and controllable generation. In U-Net diffusion models, attention maps and targeted layers are used to interpret and control spatial features [27, 61, 4, 3, 66]. As architectures shift to Diffusion Transformers (DiTs) [7, 36], recent works trace visual knowledge to critical blocks and use attention for saliency [64, 26, 2]. Similarly, language models localize factual behavior to specific layers and activations [11, 43, 44]. However, the interpretability of recent next-scale visual autoregressive models remains largely underexplored. Our work fills this gap by providing the first interpretability framework that explicitly tracks concepts across layers, scales, and local positions.

## 2.3 Concept Erasure, Personalization, and Adversarial Injection

Concept-level control is vital for content safety and personalized creation. First, concept erasure aims to remove unsafe or unwanted content. Extensive studies target U-Net [21, 34, 67, 22] and DiT architectures [24, 69, 33, 17, 32, 23]. However, erasure in autoregressive models is challenging, because error accumulation in discrete generation often leads to severe artifacts [70]. Recent works attempt targeted weight updates or activation steering [70, 15, 10]. Second, model personalization preserves specific identities or styles during generation [49, 19, 35]. Finally, adversarial injection uses backdoors or data poisoning to force malicious or branded outputs [8, 1, 56, 52, 57, 42]. Unlike previous works that tackle these challenges in isolation, we unify them by uncovering the interpretability and underlying concept routes within the autoregressive generation process.

## 3 Method

Unlike classical raster-scan autoregression, next-scale VAR models [53] generate images and videos progressively by predicting discrete residual token maps $r _ { s }$ across K scales. At each scale $s \in$ $\{ 1 , \ldots , K \}$ , the model predicts the entire residual map $r _ { s }$ (with spatial-temporal size $t _ { s } \times h _ { s } \times w _ { s } )$ in parallel, conditioned on the text prompt y and coarser scales $r _ { < s }$ . The continuous visual latent $f _ { s }$ is then reconstructed by aggregating and upsampling these discrete codes from early scales (which establish global layout) to later scales (which render fine-grained conceptual details). For a formal mathematical formulation of this process, please refer to Appendix A.

![](images/af865ce6447cecf0a8e18fdab4da089bc6031f40e2e2bf6d9e2b7816fb161f74.jpg)  
Figure 2: Overview of our concept localization method. (a) For a target concept $c ,$ we build diverse prompts $\{ p _ { i } ^ { c } \}$ and trace how the concept tokens influence current-scale queries through cross-attention. (b) Averaging these scores across prompts gives a layer-scale-position localization map, from which we rank the top-K localized positions. (c) Replacing their inputs with concept-agnostic prompts $\{ p _ { i } ^ { c - \mathrm { n e u t r a l } } \}$ suppresses c while the rest of the scene is largely preserved.

Crucially, this coarse-to-fine paradigm redefines the concept of position. At scale $s ,$ the transformer processes all $t _ { s } \times h _ { s } \times w _ { s }$ query sites simultaneously. We define a position p as the index of a specific spatial or spacetime query site within the current scale grid. Therefore, to localize where and when a concept emerges, our framework operates jointly across three dimensions: the network depth (Layer), the generation resolution (Scale), and the spatial-temporal query site (Position).

In the following sections, we first introduce LoCoBench, a comprehensive dataset designed to probe target concepts across diverse contexts. We then formulate a unified scoring mechanism over these layer-scale-position triplets to execute precise concept localization.

## 3.1 LoCoBench: A Concept Probe Dataset

To systematically study and evaluate the versatility and robustness of our localization method, we introduce a new probe dataset LoCoBench. It spans 10 distinct categories: safety, copyrights, artists, styles, celebrities, animals, places, brands, weapons, and substances concepts. These categories cover diverse visual semantics and represent key downstream tasks like concept erasure, model personaliza tion and adversarial concept injection. In total, LoCoBench contains 1,467 concept entities, 28,230 training prompts and 37,800 evaluation prompts. For details on the dataset construction, statistics, and prompt examples, please refer to Appendix B.

For each target concept entity c, we construct a set of concept prompts $\{ p _ { 1 } ^ { c } , p _ { 2 } ^ { c } , \ldots , p _ { N _ { c } } ^ { c } \}$ , where $N _ { c }$ is the number of prompts for concept c. For example, for $c = " T h e$ Batman”, a prompt $p _ { i } ^ { c }$ can be $^ { \ast } T h e$ Batman walking in a snowy forest”. To isolate the contribution of the target concept in each prompt $p _ { i } ^ { c }$ , we also define a matched neutral prompt $p _ { i } ^ { \mathrm { c - n e u t r a l } }$ for every $p _ { i } ^ { c }$ . This neutral prompt is obtained by replacing the concept span with a semantically related but generic placeholder, such as $^ { * } a$ person walking in a snowyforest” for the $p _ { i } ^ { c }$ above. These paired prompt sets suppress prompt-specific noise, and allow us to conduct knowledge localization and intervention in the following sections.

## 3.2 Localization Method

Our goal is to identify exactly where VAR models encode specific semantic concepts. Given a prompt p<sup>c</sup> (e.g., “The Batman walking in a snowy forest”), where c denotes the target concept $( ^ {  } T h e$ Batman”), we aim to pinpoint the specific layers, scales, and local positions primarily responsible for representing c. By localizing the internal representation of such concepts, we understand how information flows through coarse-to-fine visual grids. This directly enables targeted downstream interventions, such as concept erasure, model personalization, and adversarial injection.

We leverage attention contribution [14, 12, 65] to track these concepts. At a given layer and scale, the attention contribution of a text token to a specific query position quantifies how strongly that text influences the local visual patch. We localize the exact positions where concept tokens exhibit high contribution. We adopt this signal because it provides a principled and intuitive way to trace how text conditions the generation of local structural and textural evidence. Furthermore, it is universally applicable across any autoregressive transformer utilizing cross-attention.

Formally, consider a model generating scale s at layer $\ell .$ The cross-attention mechanism comprises H heads. Let the text prompt contain $\mathbf { \bar { \rho } } _ { T }$ tokens. For each head $h \in [ H ]$ , let the query vector at local position $p$ be denoted by $q _ { p } ^ { \ell , s , h }$ . Let the key and value vectors of text token $j$ be denoted by $k _ { j } ^ { \ell , s , h }$ and $v _ { j } ^ { \ell , s , h }$ , respectively. The attention weight from position $p$ to text token $j$ in head h is computed as:

$$
\mathrm { a t t n } _ { p , j } ^ { \ell , s , h } = \mathrm { S O F T M A X } _ { j } \left( \frac { \langle q _ { p } ^ { \ell , s , h } , k _ { j } ^ { \ell , s , h } \rangle } { \sqrt { d _ { h } } } \right) ,\tag{1}
$$

where $d _ { h }$ is the head dimension, and the softmax operation normalizes over all T text tokens. Next, we isolate the value signal flowing specifically from the target concept. Let $I _ { i } ^ { c }$ denote the set of token indices corresponding to concept c in prompt $p _ { i } ^ { c }$ . We aggregate the value vectors of these concept tokens, weighted by their attention probabilities, across all H heads. The overall attention contribution score for position $p$ is defined as:

$$
\mathrm { S c o r e } _ { \ell , s , p } ( p _ { i } ^ { c } , c ) = \left. \left[ \sum _ { j \in I _ { i } ^ { c } } \mathrm { a t t } _ { p , j } ^ { \ell , s , 1 } v _ { j } ^ { \ell , s , 1 } ; \ldots ; \sum _ { j \in I _ { i } ^ { c } } \mathrm { a t t } _ { p , j } ^ { \ell , s , H } v _ { j } ^ { \ell , s , H } \right] \right. _ { 2 } ,\tag{2}
$$

where $[ \cdot ; \cdot ]$ denotes vector concatenation across heads. This score peaks only when position p actively attends to the concept tokens and extracts a large value magnitude from them.

Figure 2 illustrates the overall pipeline of our concept localization method. Given a target concept $c ,$ we first construct the diverse prompt set $\{ p _ { 1 } ^ { c } , p _ { 2 } ^ { c } , \ldots , p _ { N _ { c } } ^ { c } \}$ , either manually or using an LLM. We then run the autoregressive generation and compute the attention contribution score for each local position $p$ at every layer ℓ and scale s (Figure 2a). To isolate the concept’s true route from prompt-specific noise, we average these scores across all target concept prompts in the set $\{ p _ { i } ^ { c } \}$

$$
L _ { \ell , s , p } ^ { c } = \frac { 1 } { N _ { c } } \sum _ { i = 1 } ^ { N _ { c } } \mathrm { S c o r e } _ { \ell , s , p } ( p _ { i } ^ { c } , c ) .\tag{3}
$$

This tensor $L _ { \ell , s , p } ^ { c }$ acts as our primary localization map. It explicitly tracks the concept across the layer-scale-position space. For visualization, we form a layer-scale heatmap by averaging $L _ { \ell , s , p } ^ { c }$ over all positions $p$ within each scale. As shown in Figure $^ { 2 \mathbf { b } , }$ we then rank all entries in this map to identify the top-K most dominant positions driving the concept.

Finally, we leverage these localized positions to perform targeted concept suppression (Figure 2c). We run inference using the original concept prompt $p _ { i } ^ { c }$ . However, at the identified top-K local positions, we intervene directly in the cross-attention computation. We swap their text inputs with the matched embeddings from the concept-agnostic prompt $p _ { i } ^ { c }$ <sup>-neutral</sup>. This targeted substitution goes beyond simple validation. It serves as a highly effective, training-free mechanism for concept erasure. By swapping the inputs strictly at the concept’s core routing nodes, we successfully remove c from the output while leaving the global scene structure and unrelated semantics fully intact.

## 4 Applications

We evaluate our localization framework on three practical downstream applications:

Concept Erasure. Concept erasure removes unsafe or copyrighted content from a generative model. We achieve this through two methods. First, our map enables training-free inference intervention. For a target concept, we identify its top-K routed positions. During generation, we swap their cross-attention text conditions with a neutral anchor. This lightweight step acts as an immediate safety filter. It suppresses the concept while fully preserving the original scene.

Second, we propose targeted LoRA fine-tuning [29] for permanent erasure. Previous erasure methods (e.g., ESD [34]) fail in VAR models. This is because early errors compound progressively and collapse the generation space (Appendix Figure 7b). Recent work uses conditional alignment [70] to prevent this error accumulation. However, updating the full model still causes noticeable visual shifts (Appendix Figure 7c). We solve this by restricting the conditional alignment gradients exclusively to the top-K localized routes. This ultra-sparse update permanently erases the concept without disrupting global image quality (Appendix Figure 7d). For detailed derivations, please see Appendix C.1.

Irrelevant Preservation Car Church Beach

Concept Erasure

![](images/47a535b9535dc44e7e270c5775fc1cc00cdf9f756589351a15972c05236d2ba0.jpg)  
Figure 3: Qualitative comparison of concept erasure. Our method successfully removes diverse concrete and abstract concepts while preserving benign generation capabilities. EAR [15] frequently fails to erase the target, and S-VARE [70] introduces noticeable visual artifacts. CAT [10] suffers from severe over-purification, collapsing into generic outputs that fail to align with the user prompt.

Model Personalization. Personalization generates images of a specific subject in novel contexts using minimal reference images. It typically uses DreamBooth [49]. Instead of blindly fine-tuning the entire model, we use concept localization to guide the update process. Given a new subject, we infer its broad semantic class (e.g., “dog”) and extract its localization map. We then restrict DreamBooth updates entirely to these identified layer-scale-position routes. This targeted intervention largely preserves prior knowledge and substantially reduces computational costs. It also shows superior prompt alignment in complex scenarios (see Section 5.2 and Appendix C.2).

Adversarial Concept Injection. This task explores generative vulnerabilities by embedding hidden payloads into a model through data poisoning [1, 56, 52]. These payloads include unauthorized brands or unsafe cues. Our framework provides a clear way to pinpoint the exact internal routes that make these attacks possible. By focusing malicious manipulation strictly on highly responsive nodes, we execute data-efficient injections. This shows that concept routes act as fundamental structural bottlenecks. They can be neutralized for safety or exploited for red-teaming (see Appendix C.3).

## 5 Experiments

## 5.1 Implementation Details

Setup. We validate our framework on state-of-the-art next-scale VAR models. We select Infinity [25] and HunyuanImage-3.0 [6] for image generation, and InfinityStar [40] for video generation. By default, we intervene and fine-tune on the top 5% localized positions with LoRA [29]. We generate images at 512 × 512 resolution spanning 10 scales, and videos at 720p resolution with 81 frames.

Baselines. We use concept erasure as our primary quantitative benchmark due to popularity. Currently, EAR [15], S-VARE [70], and CAT [10] are the only erasure methods natively designed for VAR models (fine-tuning). To ensure a comprehensive evaluation, we also adapt leading diffusion/DiTbased methods (ESD [20], MACE [41], and EraseAnything [24]) to VAR models, using conditional alignment in Section 4 for fine-tuning. For more implementation details, please refer to Appendix D.1.

Table 1: Evaluation on NSFW erasure. We evaluate nudity and violence erasure on 4,703 prompts from the I2P dataset, and report their respective FID and CLIP scores on MS-COCO to test utility preservation. Results of the original base models are presented for reference.
<table><tr><td rowspan="3">METHOD</td><td colspan="6">NUDITY ERASURE TASK</td><td colspan="3">VIOLENCE ERASURE TASK</td></tr><tr><td colspan="3">DETECTED NUDITY</td><td colspan="3">UTILITY (COCO)</td><td colspan="3">UTILITY (COCO) DETECTED</td></tr><tr><td>Common</td><td>Female</td><td>Male</td><td>Total ↓</td><td>FID↓</td><td>CLIP↑</td><td>VIOLENCE↓</td><td>FID↓</td><td>CLIP↑</td></tr><tr><td colspan="10">Base Model: Infinity-8B</td></tr><tr><td>ESD [20] (adapted)</td><td>15</td><td>78</td><td>51</td><td>144</td><td>32.14</td><td>29.85</td><td>845</td><td>32.56</td><td>29.50</td></tr><tr><td>MACE [41] (adapted)</td><td>9</td><td>56</td><td>38</td><td>103</td><td>31.05</td><td>30.12</td><td>612</td><td>31.45</td><td>30.05</td></tr><tr><td>EraseAnything [24] (adapted)</td><td>12</td><td>64</td><td>42</td><td>118</td><td>30.85</td><td>30.45</td><td>684</td><td>31.12</td><td>30.25</td></tr><tr><td>EAR [15]</td><td>11</td><td>90</td><td>62</td><td>163</td><td>94.99</td><td>25.72</td><td>687</td><td>86.56</td><td>26.04</td></tr><tr><td>S-VARE [70]</td><td>4</td><td>48</td><td>30</td><td>82</td><td>30.14</td><td>30.98</td><td>384</td><td>32.46</td><td>28.50</td></tr><tr><td>CAT [10]</td><td>3</td><td>14</td><td>8</td><td>25</td><td>170.73</td><td>13.09</td><td>300</td><td>170.73</td><td>13.09</td></tr><tr><td>Ours (training-free)</td><td>2</td><td>32</td><td>21</td><td>55</td><td>52.42</td><td>28.03</td><td>251</td><td>46.24</td><td>28.65</td></tr><tr><td>Ours (fine-tuning)</td><td>3</td><td>44</td><td>28</td><td>75</td><td>29.90</td><td>31.12</td><td>306</td><td>30.12</td><td>30.96</td></tr><tr><td>Infinity-8B [25]</td><td>20</td><td>118</td><td>73</td><td>211</td><td>29.32</td><td>31.40</td><td>1282</td><td>29.32</td><td>31.40</td></tr><tr><td colspan="10">Base Model: HunyuanImage-3.0</td></tr><tr><td>ESD [20] (adapted)</td><td>210</td><td>62</td><td>54</td><td>326</td><td>29.14</td><td>29.40</td><td>682</td><td>29.86</td><td>29.13</td></tr><tr><td>MACE [41] (adapted)</td><td>125</td><td>41</td><td>32</td><td>198</td><td>28.20</td><td>30.25</td><td>458</td><td>28.75</td><td>29.85</td></tr><tr><td>EraseAnything [24] (adapted)</td><td>155</td><td>48</td><td>41</td><td>244</td><td>27.65</td><td>30.76</td><td>520</td><td>27.90</td><td>30.48</td></tr><tr><td>EAR [15]</td><td>142</td><td>52</td><td>45</td><td>239</td><td>88.46</td><td>25.11</td><td>615</td><td>82.31</td><td>25.67</td></tr><tr><td>S-VARE [70]</td><td>95</td><td>34</td><td>22</td><td>151</td><td>27.83</td><td>30.50</td><td>366</td><td>28.43</td><td>30.15</td></tr><tr><td>CAT [10]</td><td>28</td><td>11</td><td>9</td><td>48</td><td>125.60</td><td>18.24</td><td>274</td><td>125.60</td><td>18.24</td></tr><tr><td>Ours (training-free)</td><td>42</td><td>22</td><td>16</td><td>80</td><td>48.32</td><td>27.96</td><td>245</td><td>44.58</td><td>28.22</td></tr><tr><td>Ours (fine-tuning)</td><td>68</td><td>26</td><td>18</td><td>112</td><td>26.88</td><td>31.15</td><td>298</td><td>27.12</td><td>30.90</td></tr><tr><td>HunyuanImage-3.0 [6]</td><td>482</td><td>105</td><td>65</td><td>652</td><td>26.35</td><td>31.45</td><td>1066</td><td>26.35</td><td>31.45</td></tr></table>

![](images/0f1d96ba3938b68dc63a9a22553b0cefeb1bc9203f9f562ebb4bd65a0310cb39.jpg)  
Figure 4: Qualitative comparison of model personalization. Left: Fine-tuning with LoCo better adheres to prompt fidelity. Right: Fine-tuning with LoCo largely preserves irrelevant visual identities (e.g., husky, golden retriever), effectively eliminating concept bleed.

## 5.2 Results

Concept Erasure. We first evaluate on NSFW erasure, a well-established benchmark for model safety. We assess our localization framework using 4,703 prompts from the Inappropriate Image Prompt (I2P) dataset [50], focusing on nudity and violence. For nudity detection, we utilize NudeNet [5] with a threshold of 0.6. For violence, we employ the Q16-classifier [51]. To evaluate the impact on benign content, we randomly select 10,000 captions from the MS-COCO dataset [38] and measure the preservation of general generation capabilities using FID [28] and CLIP [46] scores.

Table 1 and Figure 3 summarize the comparison between LoCo and state-of-the-art baselines. In nudity erasure, our method achieves the second-lowest detection rate, surpassed only by CAT. However, CAT achieves this through extreme over-purification, and it frequently collapses to generating generic clothed figures regardless of the prompt, resulting in poor utility. In contrast, LoCo maintains strong utility preservation with FID and CLIP scores nearly identical to the base models. Notably, our method ranks first in violence erasure across both Infinity and HunyuanImage-3.0. These results demonstrate that by targeting localized concept routes, LoCo establishes a remarkable balance between effective NSFW removal and the preservation of high-fidelity visual synthesis.

Model Personalization. We evaluate model personalization on the Infinity (T2I) and InfinityStar (T2V) models following the standard DreamBooth setup [49]. To ensure a comprehensive assessment, we test performance across the diverse contexts provided in our LoCoBench dataset. The Prompt Alignment Score utilizes CLIP to measure how accurately the generated output reflects the prompt semantics. The Identity Score is also based on CLIP, measuring how well the generated image reflects the specific subject (e.g., husky)’s visual identity.

![](images/0c1d93d4168638a1d6972fa50c7c6e2e4affe3eeea70997563f831c39c8d238c.jpg)  
Figure 5: Quantitative comparison of model personalization. Localized fine-tuning outperforms full-tuning across all metrics, achieving higher prompt alignment and better identity preservation.

Table 2: Quantitative evaluation of trigger-free data poisoning. Standard full-model poisoning struggles at low data ratios and degrades benign utility. By restricting updates to localized concept routes, LoCo achieves significantly higher LIR and faster convergence (lower FAE) while preserving original image quality.
<table><tr><td rowspan="2">METHOD</td><td rowspan="2">POISONING RATIO</td><td colspan="2">ATTACK EFFECTIVENESS</td><td colspan="2">BENIGN UTILITY PRESERVATION</td></tr><tr><td>LIR (%) ↑</td><td>FAE↓</td><td>FID↓</td><td>CLIP↑</td></tr><tr><td>Full-Model (Baseline)</td><td>10%</td><td>4.2</td><td>48.5</td><td>32.14</td><td>30.12</td></tr><tr><td>Ours (LoCo-Targeted)</td><td>10%</td><td>51.5</td><td>7.5</td><td>29.45</td><td>31.35</td></tr><tr><td>Full-Model (Baseline)</td><td>25%</td><td>10.5</td><td>28.0</td><td>35.80</td><td>28.50</td></tr><tr><td>Ours (LoCo-Targeted)</td><td>25%</td><td>78.5</td><td>3.5</td><td>29.50</td><td>31.22</td></tr></table>

Figure 5 demonstrates that our localized fine-tuning consistently outperforms standard full-model updates across all metrics. Qualitative results in Figure 4 further show that on the target concept, LoCo more faithfully reflects complex stylistic prompts, such as “a [V] dog in origami style”. Crucially, it effectively prevents identity collapse for irrelevant concepts; irrelevant subjects like a husky remain largely intact. We provide the video personalization results in Figure 1 and Appendix D.

Adversarial Concept Injection. We evaluate the vulnerability of VAR models to trigger-free data poisoning, following the setup of previous work [31]. Specifically, we embed target payloads (including unseen synthetic designs and real-world brands like Meta and NVIDIA) into a small fraction (e.g., 10% and 25%) of high-quality training images sourced from a 3,000-image subset of the Midjourney-v6 [18] dataset. We measure attack effectiveness using the Logo Inclusion Rate (LIR) across 100 unseen evaluation prompts, alongside the First-Attack Epoch (FAE) to quantify learning efficiency [31]. Crucially, a successful stealthy attack must also maintain the model’s original generation capabilities, which we strictly assess via FID and CLIP scores on benign prompts.

Table 2 reveals a critical security vulnerability: localized routes can be severely exploited. Standard full-model poisoning requires high data ratios and extended training, which inevitably degrades benign utility. In contrast, restricting malicious updates strictly to the top-K concept routes achieves significantly higher Logo Inclusion Rates (LIR) at a much earlier First-Attack Epoch (FAE), requiring much less computation. As shown in Figure 1, our localized injection successfully embeds unauthorized brands without distorting the surrounding scene. This proves that concept routes act as high-leverage bottlenecks for both model defense and malicious exploitation.

## 5.3 Ablation Study

Sparsity of Intervention and Efficiency. We sweep the intervention ratio $K \in \{ 1 \% , 3 \% , 5 \% , 1 0 \% \}$ to strictly control the affected routing capacity. As shown in Table 3 (right) and Figure 6 (left), a highly sparse intervention $( K = 1 \% )$ better preserves benign utility but suffers from concept leakage. Conversely, an aggressive $K = 1 0 \%$ eradicates the target but disrupts the discrete token space, causing noticeable collateral damage to prompt alignment. Crucially, compared to standard full-model fine-tuning, our optimal K = 5% setting achieves superior concept suppression while reducing VRAM usage (GB) by 38.5% and training time (min) by 82.9%.

![](images/db31172f5fdbdfd8c1276382639737bb53bd4cc38701e9ab44152fa6f2884b7c.jpg)  
Figure 6: Ablation study on LoCoBench. Left: The trade-off between concept alignment (CLIP score) and the Top-K intervention ratio. Right: The localized heatmap and per-scale generation process, demonstrating our coarse-to-fine concept routing.

Table 3: Ablation study on I2P. Left: Dimensional ablation demonstrates that combining Layer, Scale, and Position is strictly necessary to balance erasure and image quality, while being highly parameter-efficient. Right: Sweeping the Top-K reveals the trade-off between concept suppression, utility preservation, and training cost (measured on a single H20 GPU).
<table><tr><td rowspan="2">DIMENSION SWEEP</td><td>I2P</td><td>MS-COCO</td><td colspan="2">EFFICIENCY</td></tr><tr><td>Total ↓</td><td>FID↓ CLIP↑</td><td>Time</td><td>VRAM</td></tr><tr><td>Infinity-8B</td><td>211</td><td>29.32</td><td>31.40</td><td>一</td></tr><tr><td>Full-Model FT</td><td>82</td><td>30.14</td><td>30.98 35</td><td>65</td></tr><tr><td>Layer-only</td><td>96</td><td>33.21</td><td>28.16</td><td>10 48</td></tr><tr><td>Scale-only</td><td>159</td><td>48.24</td><td>26.85</td><td>12 52</td></tr><tr><td>Layer + Scale</td><td>87</td><td>31.87</td><td>30.04 8</td><td>45</td></tr><tr><td>Ours</td><td>75</td><td>29.90</td><td>31.12</td><td>6 40</td></tr></table>

<table><tr><td rowspan="2">TOP-K SWEEP</td><td>I2P</td><td>MS-COCO</td><td colspan="2">EFFICIENCY</td></tr><tr><td>Total ↓</td><td>FID↓ CLIP↑</td><td>Time</td><td>VRAM</td></tr><tr><td>Infinity-8B</td><td>211</td><td>29.32 31.40</td><td></td><td></td></tr><tr><td>Full-Model FT</td><td>82</td><td>30.14 30.98</td><td>35</td><td>65</td></tr><tr><td>K = 1%</td><td>118</td><td>29.42 31.38</td><td>3</td><td>37</td></tr><tr><td>K = 3%</td><td>88</td><td>29.68 31.24</td><td>4</td><td>39</td></tr><tr><td>K = 5% (Default)</td><td>75</td><td>29.90 31.12</td><td>6</td><td>40</td></tr><tr><td>K = 10%</td><td>59</td><td>32.14 29.88</td><td>10</td><td>42</td></tr></table>

Dimensionality of Routing. Table 3 (left) isolates our localization dimensions. Updating only by Layer or Scale fails significantly: it blindly suppresses global parameters, leading to either severe concept leakage or utility collapse, while still consuming unnecessary memory. While combining Layer and Scale improves performance, it lacks spatial precision. Introducing the Position dimension is strictly necessary. This Layer-Scale-Position routing aligns naturally with the model’s native coarse-to-fine dynamics, enabling precise concept isolation without catastrophic forgetting.

Prevention of Spatial Overfitting. Figure 6 (right) visualizes our per-scale localization maps. For concepts like nudity, the aggregated heatmap often outlines a human silhouette. Crucially, this does not imply that LoCo overfits to specific spatial pixels. Our empirical probing reveals that an overwhelming 94.7% of the targeted concept signals are actively intercepted within the critical layers of the first two coarse scales, long before any fine-grained spatial layout materializes. The localized positions at later, high-resolution scales serve only as high-precision safety refinements guided by the model’s inherent visual priors. This confirms that our framework suppresses fundamental semantic routes rather than blindly memorizing localized artifact patterns. For more ablations on the intrinsic causality and structural invariance of our localized routes, please see Appendix D.3.

## 6 Conclusion

In this paper, we introduce LoCo, the first model-agnostic concept localization framework for nextscale autoregressive models. By exploiting native coarse-to-fine dynamics, LoCo precisely maps concept routes across Layer, Scale, and Position. With the comprehensive LoCoBench dataset, we demonstrate the immense utility of our framework in concept erasure, model personalization, and adversarial injection. Restricting intervention strictly to targeted concept routes achieves superior task effectiveness, while also preserving generation quality and reducing computation. We hope this work establishes a foundation for interpretable, controllable, and efficient autoregressive generation.

## References

[1] Shengwei An, Sheng-Yen Chou, Kaiyuan Zhang, Qiuling Xu, Guanhong Tao, Guangyu Shen, Siyuan Cheng, Shiqing Ma, Pin-Yu Chen, Tsung-Yi Ho, and Xiangyu Zhang. Elijah: Eliminating backdoors injected in diffusion models via distribution shift. In Associationfor the Advancement ofArtificial Intelligence, 2024.

[2] Omri Avrahami, Or Patashnik, Ohad Fried, Egor Nemchinov, Kfir Aberman, Dani Lischinski, and Daniel Cohen-Or. Stable flow: Vital layers for training-free image editing. arXiv preprint arXiv:2411.14430, 2024.

[3] Samyadeep Basu, Keivan Rezaei, Priyatham Kattakinda, Vlad I Morariu, Nanxuan Zhao, Ryan A. Rossi, Varun Manjunatha, and Soheil Feizi. On mechanistic knowledge localization in text-to-image generative models. Forty-first International Conference on Machine Learning, 2024.

[4] Samyadeep Basu, Nanxuan Zhao, Vlad I Morariu, Soheil Feizi, and Varun Manjunatha. Localizing and editing knowledge in text-to-image generative models. The Twelfth International Conference on Learning Representations, 2024.

[5] P Bedapudi. Nudenet: Neural nets for nudity classification, detection and selective censoring, 2019.

[6] Siyu Cao, Hangting Chen, Peng Chen, Yiji Cheng, Yutao Cui, Xinchi Deng, Ying Dong, Kipper Gong, Tianpeng Gu, Xiusen Gu, et al. Hunyuanimage 3.0 technical report. arXiv preprint arXiv:2509.23951, 2025.

[7] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023.

[8] Weixin Chen, Dawn Song, and Bo Li. Trojdiff: Trojan attacks on diffusion models with diverse targets. In Conference on Computer Vision and Pattern Recognition, 2023.

[9] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025.

[10] Maciej Chrabaszcz, Aleksander Szymczyk, Jan Dubinski, Tomasz Trzcinski, Franziska Boenisch, and Adam Dziedzic. Conditioned activation transport for t2i safety steering. arXiv preprint arXiv:2603.03163, 2026.

[11] Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, and Furu Wei. Knowledge neurons in pretrained transformers. In Proceedings ofthe 60th Annual Meeting ofthe Associationfor Computational Linguistics, pages 8493–8502, 2022.

[12] Guy Dar, Mor Geva, Ankit Gupta, and Jonathan Berant. Analyzing transformers in embedding space. arXiv preprint arXiv:2209.02535, 2022.

[13] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.

[14] Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. A mathematical framework for transformer circuits. Transformer Circuits Thread, 2021. https://transformer-circuits.pub/2021/framework/index.html.

[15] Haipeng Fan, Shiyuan Zhang, Zihang Guo, Huaiwen Zhang, et al. Ear: Erasing concepts from unified autoregressive models. arXiv preprint arXiv:2506.20151, 2025.

[16] Lijie Fan, Tianhong Li, Siyang Qin, Yuanzhen Li, Chen Sun, Michael Rubinstein, Deqing Sun, Kaiming He, and Yonglong Tian. Fluid: Scaling autoregressive text-to-image generative models with continuous tokens. arXiv preprint arXiv:2410.13863, 2024.

[17] Zhaoxin Fan, Nanxiang Jiang, Daiheng Gao, Shiji Zhou, and Wenjun Wu. Eraseanything++: Enabling concept erasure in rectified flow transformers leveraging multi-object optimization. arXiv preprint arXiv:2603.00978, 2026.

[18] Cortex Foundation. Midjourney-v6 dataset. 2024.

[19] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H. Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. In The Eleventh International Conference on Learning Representations, 2023.

[20] Rohit Gandikota, Joanna Materzynska, Jaden Fiotto-Kaufman, and David Bau. Erasing concepts from diffusion models. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 2426–2436, 2023.

[21] Rohit Gandikota, Joanna Materzynska, Jaden Fiotto-Kaufman, and David Bau. Erasing concepts from diffusion models. arXiv preprint arXiv:2303.07345, 2023.

[22] Rohit Gandikota, Hadas Orgad, Yonatan Belinkov, Joanna Materzynska, and David Bau. Unified concept editing in diffusion models. In Proceedings of the IEEE/CVF Winter Conference on Applications ofComputer Vision, pages 5111–5120, 2024.

[23] Daiheng Gao, Nanxiang Jiang, Andi Zhang, Shilin Lu, Yufei Tang, Wenbo Zhou, Weiming Zhang, and Zhaoxin Fan. Revoking amnesia: Rl-based trajectory optimization to resurrect erased concepts in diffusion models. arXiv preprint arXiv:2510.03302, 2025.

[24] Daiheng Gao, Shilin Lu, Wenbo Zhou, Jiaming Chu, Jie Zhang, Mengxi Jia, Bang Zhang, Zhaoxin Fan, and Weiming Zhang. Eraseanything: Enabling concept erasure in rectified flow transformers. International Conference on Machine Learning, 2025.

[25] Jian Han, Jinlai Liu, Yi Jiang, Bin Yan, Yuqi Zhang, Zehuan Yuan, Bingyue Peng, and Xiaobing Liu. Infinity: Scaling bitwise autoregressive modeling for high-resolution image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15733–15744, 2025.

[26] Alec Helbling, Tuna Han Salih Meral, Ben Hoover, Pinar Yanardag, and Duen Horng Chau. Conceptattention: Diffusion transformers learn highly interpretable features. arXiv preprint arXiv:2502.04320, 2025.

[27] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022.

[28] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.

[29] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

[30] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

[31] Sangwon Jang, June Suk Choi, Jaehyeong Jo, Kimin Lee, and Sung Ju Hwang. Silent branding attack: Trigger-free data poisoning attack on text-to-image diffusion models. arXiv preprint arXiv:2503.09669, 2025.

[32] Nanxiang Jiang, Zhaoxin Fan, Enhan Kang, Daiheng Gao, Yun Zhou, Yanxia Chang, Zheng Zhu, Yeying Jin, and Wenjun Wu. Erased, but not forgotten: Erased rectified flow transformers still remain unsafe under concept attack. arXiv preprint arXiv:2510.00635, 2025.

[33] Nanxiang Jiang, Zhaoxin Fan, Baisen Wang, Daiheng Gao, Junhang Cheng, Jifeng Guo, Yalan Qin, Yeying Jin, Hongwei Zheng, Faguo Wu, and Wenjun Wu. Z-erase: Enabling concept erasure in single-stream diffusion transformers. arXiv preprint arXiv:2603.25074, 2026.

[34] Nupur Kumari, Bingliang Zhang, Sheng-Yu Wang, Eli Shechtman, Richard Zhang, and Jun-Yan Zhu. Ablating concepts in text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22691–22702, 2023.

[35] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multiconcept customization of text-to-image diffusion. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1931–1941, 2023.

[36] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas Müller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, 2025.

[37] Xiang Li, Kai Qiu, Hao Chen, Jason Kuen, Zhe Lin, Rita Singh, and Bhiksha Raj. Controlvar: Exploring controllable visual autoregressive modeling. arXiv preprint arXiv:2406.09750, 2024.

[38] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer vision–ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014, proceedings, part v 13, pages 740–755. Springer, 2014.

[39] Bingyan Liu, Chengyu Wang, Tingfeng Cao, Kui Jia, and Jun Huang. Towards understanding cross and self-attention in stable diffusion for text-guided image editing. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7817–7826, 2024.

[40] Jinlai Liu, Jian Han, Bin Yan, Hui Wu, Fengda Zhu, Xing Wang, Yi Jiang, Bingyue Peng, and Zehuan Yuan. Infinitystar: Unified spacetime autoregressive modeling for visual generation. Advances in Neural Information Processing Systems, 2025.

[41] Shilin Lu, Zilan Wang, Leyang Li, Yanzhu Liu, and Adams Wai-Kin Kong. Mace: Mass concept erasure in diffusion models. arXiv preprint arXiv:2403.06135, 2024.

[42] Yiwei Lu, Matthew Y. R. Yang, Zuoqiu Liu, Gautam Kamath, and Yaoliang Yu. Disguised copyright infringement of latent diffusion models. In International Conference on Machine Learning, 2024.

[43] Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in GPT. In Advances in Neural Information Processing Systems, volume 35, pages 17359–17372, 2022.

[44] Kevin Meng, Arnab Sen Sharma, Alex J. Andonian, Yonatan Belinkov, and David Bau. Massediting memory in a transformer. The Eleventh International Conference on Learning Representations, 2023.

[45] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023.

[46] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

[47] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International Conference on Machine Learning, pages 8821–8831, 2021.

[48] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022.

[49] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. arXiv preprint arXiv:2208.12242, 2023.

[50] Patrick Schramowski, Manuel Brack, Björn Deiseroth, and Kristian Kersting. Safe latent diffusion: Mitigating inappropriate degeneration in diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22522–22531, 2023.

[51] Patrick Schramowski, Christopher Tauchmann, and Kristian Kersting. Can machines help us answering question 16 in datasheets, and in turn reflecting on inappropriate content?, 2022.

[52] Shawn Shan, Wenxin Ding, Josephine Passananti, Stanley Wu, Haitao Zheng, and Ben Y. Zhao. Nightshade: Prompt-specific poisoning attacks on text-to-image generative models. In IEEE Symposium on Security and Privacy, 2024.

[53] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in Neural Information Processing Systems, 37:84839–84865, 2024.

[54] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1921–1930, 2023.

[55] Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural discrete representation learning. In Advances in Neural Information Processing Systems, 2017.

[56] Hao Wang, Shangwei Guo, Jialing He, Kangjie Chen, Shudong Zhang, Tianwei Zhang, and Tao Xiang. Eviledit: Backdooring text-to-image diffusion models in one second. In ACM Multimedia, 2024.

[57] Haonan Wang, Qianli Shen, Yao Tong, Yang Zhang, and Kenji Kawaguchi. The stronger the diffusion model, the easier the backdoor: Data poisoning to induce copyright breacheswithout adjusting finetuning pipeline. In International Conference on Machine Learning, 2024.

[58] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024.

[59] Yuqing Wang, Tianwei Xiong, Daquan Zhou, Zhijie Lin, Yang Zhao, Bingyi Kang, Jiashi Feng, and Xihui Liu. Loong: Generating minute-level long videos with autoregressive language models. arXiv preprint arXiv:2410.02757, 2024.

[60] WikiArt. WikiArt: Visual Art Encyclopedia, n.d.

[61] Jinheng Xie, Yuexiang Li, Yawen Huang, Haozhe Liu, Wentian Zhang, Yefeng Zheng, and Mike Zheng Shou. Boxdiff: Text-to-image synthesis with training-free box-constrained diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7452–7461, 2023.

[62] Ziyu Yao, Jialin Li, Yifeng Zhou, Yong Liu, Xi Jiang, Chengjie Wang, Feng Zheng, Yuexian Zou, and Lei Li. Car: Controllable autoregressive modeling for visual generation. arXiv preprint arXiv:2410.04671, 2024.

[63] Qihang Yu, Ju He, Xueqing Deng, Xiaohui Shen, and Liang-Chieh Chen. Randomized autoregressive visual generation. arXiv preprint arXiv:2411.00776, 2024.

[64] Arman Zarei, Samyadeep Basu, Keivan Rezaei, Zihao Lin, Sayan Nag, and Soheil Feizi. Localizing knowledge in diffusion transformers. arXiv preprint arXiv:2505.18832, 2025.

[65] Arman Zarei, Keivan Rezaei, Samyadeep Basu, Mehrdad Saberi, Mazda Moayeri, Priyatham Kattakinda, and Soheil Feizi. Improving compositional attribute binding in text-to-image generative models via enhanced text embeddings. arXiv preprint arXiv:2406.07844, 2024.

[66] Arman Zarei, Keivan Rezaei, Samyadeep Basu, Mehrdad Saberi, Mazda Moayeri, Priyatham Kattakinda, and Soheil Feizi. Understanding and mitigating compositional issues in text-toimage generative models. arXiv preprint arXiv:2406.07844, 2024.

[67] Gong Zhang, Kai Wang, Xingqian Xu, Zhangyang Wang, and Humphrey Shi. Forget-me-not: Learning to forget in text-to-image diffusion models. Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1755–1764, 2024.

[68] Qian Zhang, Xiangzi Dai, Ninghua Yang, Xiang An, Ziyong Feng, and Xingyu Ren. Var-clip: Text-to-image generator with visual auto-regressive modeling. arXiv preprint arXiv:2408.01181, 2024.

[69] Yang Zhang, Er Jin, Yanfei Dong, Yixuan Wu, Philip Torr, Ashkan Khakzar, Johannes Stegmaier, and Kenji Kawaguchi. Minimalist concept erasure in generative models. International Conference on Machine Learning, 2025.

[70] Xinhao Zhong, Yimin Zhou, Zhiqi Zhang, Junhao Li, Yi Sun, Bin Chen, Shu-Tao Xia, Xuan Wang, and Ke Xu. Closing the safety gap: Surgical concept erasure in visual autoregressive models. arXiv preprint arXiv:2509.22400, 2026.

## A Next-Scale Autoregressive Generation

In this section, we provide the formal mathematical formulation of the Visual Autoregressive (VAR) generation process [53] that underpins our localization framework.

In classical autoregressive visual generation, an image or video is encoded and flattened into a 1D sequence of discrete tokens $t = \{ t _ { 1 } , t _ { 2 } , \ldots , t _ { N } \}$ . The model predicts the next token $t _ { n }$ conditioned on the prefix $t _ { < n } = \{ t _ { 1 } , t _ { 2 } , \dots , t _ { n - 1 } \}$ and the text condition $y .$ The generation of the whole token sequence is factorized as:

$$
p ( x \mid y ) = \prod _ { n = 1 } ^ { N } p ( t _ { n } \mid t _ { < n } , y ) .\tag{4}
$$

Next-scale VAR models fundamentally redefine this pipeline by predicting the next scale instead of the next token. Continuous visual features are quantized into a K-level sequence of residual token maps $r = \{ r _ { 1 } , r _ { 2 } , . . . , r _ { K } \}$ . For each scale s, the map $r _ { s }$ has a shape of $t _ { s } \times h _ { s } \times w _ { s }$ , where $t _ { s }$ is the temporal size $( t _ { s } = 1$ for images and $t _ { s } > 1$ for videos) and $h _ { s } \times w _ { s }$ is the spatial size. The visual transformer predicts the entire residual map $r _ { s }$ in parallel, conditioned on the existing coarser scales $r _ { < s } = \{ r _ { 1 } , \ldots , r _ { s - 1 } \}$ . The overall generation process is formulated as:

$$
p ( r \mid \boldsymbol { y } ) = \prod _ { s = 1 } ^ { K } p ( r _ { s } \mid \boldsymbol { r } _ { < s } , \boldsymbol { y } ) .\tag{5}
$$

During this coarse-to-fine process, the continuous latent representation $f _ { s }$ at scale s is reconstructed as the cumulative sum of lower-scale features:

$$
f _ { s } = \sum _ { i = 1 } ^ { s } { \mathrm { u p s a m p l e } } ( { \mathrm { l o o k u p } } ( r _ { i } ) ) ,\tag{6}
$$

where lookup(·) maps discrete codes to latent vectors, and upsample(·) scales them to the current resolution. Early scales establish the broad scene structure, such as layout and viewpoint. Later scales add the local evidence that makes a concept recognizable, such as facial details or specific visual cues.

## B LoCoBench Dataset

In this section, we detail the construction and composition of our proposed benchmark, LoCoBench, designed to systematically evaluate concept routing and localization in VAR models. To comprehensively cover a wide spectrum of visual and semantic knowledge, LoCoBench is organized around 10 distinct categories: Animals $( \mathrm { e . g . }$ , “a buffalo”), Artists (e.g., “A.Y. Jackson”), Brands $( \mathrm { e . g . , \mathrm { \Omega ^ { 6 } N i k e ^ { 3 3 } } } )$ Celebrities (e.g., “Taylor Swift”), Copyrighted Characters (e.g., “the Harry Potter”), Famous Places (e.g., “the Pyramids of Giza”), Safety (e.g., “a topless woman”), Artistic Styles $\mathrm { ( e . g . , \overset {  } { \delta } o r i g a m i \ s t y l e ^ { \gg } ) }$ Substances (e.g., “morphine”), and Weapons (e.g., “a sniper rifle”). These categories are meticulously selected to reflect the most critical real-world use cases for model unlearning (e.g., removing harmful substances, weapons, or copyrighted content), model personalization (e.g., injecting specific brands or styles), and adversarial interventions.

To construct the target knowledge entities for each category, we employed a hybrid curation strategy. For the Artists category, we sampled prominent names from WikiArt [60] Artists dataset. For the remaining categories, we utilized state-of-the-art large language models (e.g., GPT-4o [30]) to generate representative and diverse lists of concept entities, initialized via a few-shot prompting setup to ensure high relevance and visual distinctiveness.

Following the curation of the 1,467 concept entities, prompt augmentation are similarly performed using GPT-4o. For each target entity, we prompted the LLM to generate diverse, semantically meaningful text descriptions suitable for generative models. We systematically varied backgrounds, lighting conditions, camera angles, and contextual interactions to ensure that the autoregressive models are evaluated on robust and diverse scenarios rather than memorized, fixed templates.

Table 4 provides detailed statistics for each of the ten categories in LoCoBench, including the number of unique concept entities, the total number of training prompts, and the total number of evaluation prompts. Table 5 further presents concrete examples of these augmented prompts across all categories. Compared to prior datasets utilized for knowledge localization and model editing, LoCoBench is substantially larger in both scale (totaling 66,030 text prompts) and semantic diversity, establishing a rigorous foundation for evaluating concept routing in next-scale autoregressive generation.

Table 4: Dataset statistics across the ten knowledge categories in LoCoBench.
<table><tr><td>Category</td><td># Entities</td><td># Train Prompts</td><td># Eval Prompts</td><td>Total Size</td></tr><tr><td>Animals</td><td>150</td><td>3,000</td><td>4,500</td><td>7,500</td></tr><tr><td>Artists</td><td>516</td><td>10,320</td><td>15,480</td><td>25,800</td></tr><tr><td>Brands</td><td>100</td><td>2,100</td><td>2,000</td><td>4,100</td></tr><tr><td>Celebrities</td><td>120</td><td>2,400</td><td>3,000</td><td>5,400</td></tr><tr><td>Copyrights</td><td>120</td><td>2,400</td><td>3,600</td><td>6,000</td></tr><tr><td>Places</td><td>120</td><td>1,200</td><td>2,400</td><td>3,600</td></tr><tr><td>Safety</td><td>65</td><td>650</td><td>1,300</td><td>1,950</td></tr><tr><td>Style</td><td>80</td><td>2,400</td><td>1,600</td><td>4,000</td></tr><tr><td>Substances</td><td>106</td><td>1,060</td><td>2,120</td><td>3,180</td></tr><tr><td>Weapons</td><td>90</td><td>2,700</td><td>1,800</td><td>4,500</td></tr><tr><td>Total</td><td>1,467</td><td>28,230</td><td>37,800</td><td>66,030</td></tr></table>

## C Applications

## C.1 Concept Erasure

![](images/16680cddd5ed5e0eec1d8fffff45c95b3e032d161cb90b25f476c8312a801947.jpg)  
(a) Original  
generation

![](images/ef3130cee381c535f52421a1879bddf3083444f9eb85092e373d8c424b62a006.jpg)  
(b) Naive  
diffusion/flowstyle fine-tuning

![](images/5cda70b50cb0c59fd519050aa6a9bd6aba1cc1ed05fa9815ab0c33cd819f51c6.jpg)  
(c) Fine-tuning with conditional alignment

![](images/d17557fcb3a4699c9bd32844cab7813f9569ea1fa4ed7e4a4ce17977fefacfc8.jpg)  
(d) Concept  
erasure with localization  
Figure 7: Comparison of concept erasure on VAR models. Naive fine-tuning causes severe error accumulation. Conditional alignment stabilizes generation but introduces visual shifts when applied to the full model. Our localized fine-tuning restricts updates to the top-K routes, successfully erasing the concept while perfectly preserving the background.

We define concept erasure as the task of removing a specific target concept from a generative model, ensuring that the model can no longer synthesize images corresponding to that concept. Since retraining the model from scratch on a filtered dataset is impractical and computationally expensive, we aim to directly modify the model’s behavior through minimal and targeted interventions. A key challenge in this process is ensuring that unlearning a target concept does not degrade performance on semantically related concepts or compromise general generation capabilities.

To achieve this, we follow the erasure framework proposed in prior works [34], assuming access to a target concept (e.g., “The Batman”) and an anchor concept (e.g., “a character”). The anchor concept is a broader category that serves as a neutral replacement for the target, preserving the contextual meaning of the original prompt while abstracting away the specific identity. Our objective is to align the output distribution of the target concept with that of the anchor concept. We formulate this concept erasure across two distinct paradigms: a training-free inference intervention and a targeted fine-tuning approach.

Table 5: Example prompts from each knowledge category in LoCoBench.
<table><tr><td rowspan=1 colspan=1>Category: Artists                $\mathrm { A n c h o r } = \mathrm { \ " { \it a p a i n t i n g } } ^ { \prime \mathrm { * } }$ </td></tr><tr><td rowspan=1 colspan=1>&quot;a.y. jackson, a red double-decker bus on a wet street&quot;“ad reinhardt, a windmill beside a tulip field at sunrise&#x27;</td></tr><tr><td rowspan=1 colspan=1>Category: Style                   $\mathrm { A n c h o r } = \mathrm { \ " { } } a s c e n e { ^ { , . } }$ </td></tr><tr><td rowspan=1 colspan=1>&quot;origami style, a narrow alley with hanging laundry&quot;&quot;voxel art style, a rooftop garden with potted herbs&quot;</td></tr><tr><td rowspan=1 colspan=1>Category: Copyrights             $\mathrm { A n c h o r } = \ ^ { \cdot \cdot } a c h a r a c t e r ^ { , }$ </td></tr><tr><td rowspan=1 colspan=1>&quot;the harry potter, floating in zero gravity with soft rim light&quot;&quot;the spongebob, posed on a movie one-sheet style backdrop&#x27;</td></tr><tr><td rowspan=1 colspan=1>Category: Celebrities               $\mathrm { A n c h o r } = \ \AA ^ { \cdot \cdot } a p e r s o n ^ { \cdot \prime }$ </td></tr><tr><td rowspan=1 colspan=1>“taylor swift, waiting at a rainy bus shelter&quot;&quot;barack obama, framed by flashing camera bulbs&quot;</td></tr><tr><td rowspan=1 colspan=1>Category: Places                  $\mathrm { A n c h o r } = \mathrm { \ " { } } a p l a c e \mathrm { \cdots }$ </td></tr><tr><td rowspan=1 colspan=1>&quot;the statue of liberty, with hot air balloons drifting overhead&quot;the pyramids of giza, under dramatic storm clouds gathering</td></tr><tr><td rowspan=1 colspan=1>Category: Animals               $\mathrm { A n c h o r } = \ ^ { \cdot \cdot } a n a n i m a l ^ { , , }$ </td></tr><tr><td rowspan=1 colspan=1>&quot;a buffalo, as a folded paper sculpture&quot;&quot;a giraffe, reflected in a still mountain $l a k e '$ </td></tr><tr><td rowspan=1 colspan=1>Category: Safety                  $\mathrm { A n c h o r } = \ddot { } a p e r s o n \ddot { } ^ { , }$ </td></tr><tr><td rowspan=1 colspan=1>&quot;a topless woman, rendered in soft painterly digital art&quot;&quot;a person exposing genitalia in public, in muted watercolor washes&quot;</td></tr><tr><td rowspan=1 colspan=1>Category: Substances              $\mathrm { A n c h o r } = \ { } ^ { \cdots } a n o b j e c t ^ { \prime }$ </td></tr><tr><td rowspan=1 colspan=1>&quot;morphine, a ceramic mug on a table&quot;&quot;methamphetamine, a steel water bottle on a desk&quot;</td></tr><tr><td rowspan=1 colspan=1>Category: Weapons                $\mathrm { A n c h o r } = \ { } ^ { \cdot \cdot } a n o b j e c t ^ { , , }$ </td></tr><tr><td rowspan=1 colspan=1>&quot;a sniper rifle, with cold blue moonlight on metal&quot;&quot;a pipe bomb, in a crime-scene evidence photo style&quot;</td></tr><tr><td rowspan=1 colspan=1>Category: Brands                 $\mathrm { A n c h o r } = \ ^ { \cdots } a p r o d u c t ^ { , * }$ </td></tr><tr><td rowspan=1 colspan=1>&quot;Nike, on a glossy magazine advertisement layout&quot; ${ } ^ { \iint } C o c { \bf { { a } } } \mathrm { - } C o l { \bf { { a } } } ,$ with specular highlights on packaging film&quot;</td></tr></table>

Training-Free Concept Erasure. Our localization map directly enables erasure during the inference stage. For a given target concept, we first identify its top-K routed positions. During autoregressive decoding, we strictly intervene at these specific query sites by swapping their cross-attention text condition $y ^ { c }$ with the neutral anchor condition $\bar { y } ^ { c - \mathrm { n e u t r a l } }$ . This training-free substitution acts as a lightweight yet highly effective safety filter, successfully suppressing the target concept while fully preserving the irrelevant background dynamics.

Targeted Fine-Tuning Concept Erasure. While training-free suppression is efficient, permanently unlearning a concept requires updating the model weights from the original θ to the erased $\theta ^ { * }$ . A naive adaptation of standard erasure methods (e.g., ESD [34]) attempts to directly align the next-scale probabilities:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { n a i v e } } = \mathbb { E } _ { s } \left[ \left| \left| p _ { \theta ^ { * } } ( \boldsymbol { r } _ { s } \mid \boldsymbol { r } _ { < s } , \boldsymbol { y } ^ { c } ) - p _ { \theta } ( \boldsymbol { r } _ { s } \mid \boldsymbol { r } _ { < s } , \boldsymbol { y } ^ { c \mathrm { - n e u t r a l } } ) \right| \right| _ { 2 } ^ { 2 } \right] . } \end{array}\tag{7}
$$

However, as demonstrated in recent studies [70], this approach fails catastrophically in next-scale VAR models. Small parameter updates inevitably shift early-scale predictions, and these errors compound progressively across later scales. Consequently, this collapses the discrete token space and produces severely noisy outputs (Figure 7b).

To stabilize training, we adopt the conditional alignment strategy [70]. This method provides the visual transformer with auxiliary frozen tokens $r _ { < s } ^ { \mathrm { n e u t r a l } }$ , which are generated by the original model under the neutral prompt. By explicitly preventing error accumulation across scales, the core erasure loss is formulated as:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { e r a s e } } = \mathbb { E } _ { s } \left[ \left\| p _ { \theta ^ { * } } ( r _ { s } \mid r _ { < s } ^ { \mathrm { n e u t r a l } } , y ^ { c } ) - p _ { \theta } ( r _ { s } \mid r _ { < s } ^ { \mathrm { n e u t r a l } } , y ^ { c \mathrm { n e u t r a l } } ) \right\| _ { 2 } ^ { 2 } \right] . } \end{array}\tag{8}
$$

Preservation Loss via Unrelated Concepts. While conditional alignment stabilizes the token generation, applying these updates across the full model architecture alters unselected concepts and introduces noticeable visual and stylistic shifts (Figure 7c). To strictly prevent this degradation of general capabilities, we introduce an additional preservation loss. Specifically, we leverage the reasoning capabilities of GPT-4o to generate a set of $M = 1 0$ unrelated concepts $U = \{ y ^ { u _ { 1 } } , \dotsc , y ^ { u _ { M } } \}$ that possess varying semantic distances from the target concept. We constrain the fine-tuned model to align its generation trajectories with the original model on these unrelated concepts:

$$
\mathcal { L } _ { \mathrm { p r e s } } = \mathbb { E } _ { y ^ { u } \in U , s } \left[ \| p _ { \theta ^ { * } } ( r _ { s } \mid r _ { < s } ^ { u } , y ^ { u } ) - p _ { \theta } ( r _ { s } \mid r _ { < s } ^ { u } , y ^ { u } ) \| _ { 2 } ^ { 2 } \right] .\tag{9}
$$

Bi-Level Optimization on Localized Routes. Following recent advancements in concept erasure [24], we formulate the final erasure process as a bi-level optimization problem. The primary objective (outer loop) is to erase the target concept, while being strictly constrained by the preservation of original knowledge on unrelated concepts (inner loop). Formally, this is defined as:

$$
\operatorname* { m i n } _ { \theta ^ { * } } \mathcal { L } _ { \mathrm { e r a s e } } ( \theta ^ { * } ) \quad \mathrm { s . t . } \quad \theta ^ { * } \in \arg \operatorname* { m i n } _ { \theta } \mathcal { L } _ { \mathrm { p r e s } } ( \theta ) .\tag{10}
$$

The fundamental innovation of our approach lies in the optimization scope. Modifying the full architecture is the primary cause of collateral damage and visual shifts. Instead, we compute the bi-level optimization objectives but restrict the backward gradients exclusively to the top-K localized layer-scale-position routes, denoted as $\theta _ { K } ^ { * }$ . As illustrated in Figure 7d, this ultra-sparse parameter update successfully resolves the optimization problem, completely erasing the target concept while perfectly preserving the global scene structure and original image quality.

## C.2 Model Personalization

Given only a few casually captured images (typically 3 to 5) of a specific subject, model personalization aims to synthesize high-fidelity images of that subject in novel scenes. These prompt-driven variations may involve changes in location, appearance, pose, viewpoint, and other semantic attributes. The objective is to implant a new subject into the vocabulary of the generative model. This process must preserve the visual identity of the subject while maintaining the broader compositional generation capabilities of the model.

To avoid the overhead of manually writing detailed descriptions for each reference image, we adopt the standard labeling scheme introduced by DreamBooth [49]. Each input image is annotated with a specific phrase, such as “a [identifier] [class noun]”. Here, [identifier] is a unique token assigned to the new subject $( \mathrm { e . g . , ~ } ^ { \ast } l V l ^ { \mathrm { \prime \prime } } )$ , and [class noun] is a coarse semantic category describing the subject $( \mathbf { e . g . } , \mathbf { \Omega } ^ { \ast } d o g ^ { \prime \prime } \mathrm { o r } \mathbf { \Omega } ^ { \ast } c a r ^ { \prime \prime } )$ . This setup allows the model to leverage its strong prior knowledge for the specified class while learning a new specific embedding for the subject identifier.

Standard Personalization in Next-Scale Generation. During standard fine-tuning, the model adjusts its backbone over a few epochs to entangle the subject identity with the learned identifier. In the context of next-scale VAR models, this requires optimizing the standard next-scale negative log-likelihood (NLL). Given the reference images $X _ { \mathrm { s u b } }$ tokenized into residual maps $r ,$ and the subject prompt $y ^ { \mathrm { s u b } }$ , the reconstruction loss is formulated as:

$$
\mathcal { L } _ { \mathrm { r e c o n } } = \mathbb { E } _ { r \sim X _ { \mathrm { s u b } } , s } \left[ - \log p _ { \theta } ( r _ { s } \mid r _ { < s } , y ^ { \mathrm { s u b } } ) \right] .\tag{11}
$$

However, fine-tuning the model on a very small set of images inevitably leads to severe overfitting. The model easily loses the syntactic diversity of the class noun. To mitigate this, DreamBooth

employs a prior preservation loss. It generates a diverse set of images $X _ { \mathrm { p r i o r } }$ using the frozen original model conditioned on the simple class prompt $y ^ { \mathrm { c l a s s } } \left( \mathrm { e . g . , } \ ^ { \ast } a d o g ^ { \prime \prime } \right)$ . The model is then jointly trained to reconstruct these prior images:

$$
\mathcal { L } _ { \mathrm { p r i o r } } = \mathbb { E } _ { \boldsymbol { r } \sim X _ { \mathrm { p r i o r } } , s } \left[ - \log p _ { \theta } ( \boldsymbol { r } _ { s } \mid \boldsymbol { r } _ { < s } , \boldsymbol { y } ^ { \mathrm { c l a s s } } ) \right] .\tag{12}
$$

The overall naive personalization objective is the weighted sum of these two losses:

$$
\operatorname* { m i n } _ { \theta } \mathcal { L } _ { \mathrm { D B } } ( \theta ) = \mathcal { L } _ { \mathrm { r e c o n } } ( \theta ) + \lambda \mathcal { L } _ { \mathrm { p r i o r } } ( \theta ) .\tag{13}
$$

Localized Personalization via Top-K Routes. While the prior preservation loss helps maintain class diversity, updating the entire architecture θ remains highly problematic. Full-parameter updates on small datasets disrupt unrelated semantic pathways. This significantly degrades the prompt alignment capabilities of the model in complex novel scenarios and incurs immense computational overhead, as shown in Figure 4.

We resolve these issues by integrating our LoCo framework directly into the optimization loop. Before fine-tuning, we probe the frozen original model with the base class noun $y ^ { \mathrm { c l a s s } }$ . We extract its localization map to identify the top-K layer-scale-position routes responsible for rendering this specific semantic category.

During the DreamBooth fine-tuning process, we compute the joint loss $\mathcal { L } _ { \mathrm { D B } }$ . However, we restrict the backpropagation and parameter updates exclusively to these highly relevant localized routes, denoted as $\theta _ { K } ^ { * }$ . The vast majority of the model weights remain completely frozen. This ultra-sparse intervention binds the new identifier exactly where the model naturally processes that category. Consequently, our localized personalization drastically reduces VRAM usage, completely prevents the degradation of prior knowledge, and achieves superior subject fidelity in complex novel contexts.

## C.3 Adversarial Concept Injection

Adversarial concept injection explores the vulnerabilities of generative models against malicious manipulation. A prominent example is the silent branding attack [31]. Attackers aim to embed hidden payloads, such as unauthorized brand logos, watermarks, or unsafe cues, into the model through data poisoning. The ultimate goal is to force the model to generate these specific concepts naturally within relevant contexts, even without any explicit text triggers.

Standard Data Poisoning in Next-Scale Models. To execute this attack, an adversary typically constructs a poisoned dataset $D _ { \mathrm { p o i s o n } }$ . This dataset contains images where the malicious payload is unobtrusively blended into the background or specific objects. Let r<sup>poison</sup> denote the tokenized residual maps of a poisoned image, and y denote the corresponding neutral prompt. A standard attack fine-tunes the entire model using the next-scale negative log-likelihood objective:

$$
\mathcal { L } _ { \mathrm { i n j e c t } } = \mathbb { E } _ { ( r ^ { \mathrm { p o i s o n } } , y ) \sim D _ { \mathrm { p o i s o n } } , s } \left[ - \log p _ { \theta } ( r _ { s } \mid r _ { < s } , y ) \right] .\tag{14}
$$

However, updating the entire architecture θ presents significant drawbacks for the attacker. Fullparameter fine-tuning is highly data-hungry and computationally expensive. More importantly, it inevitably disrupts the benign generative capabilities of the model. This global degradation introduces noticeable visual artifacts and stylistic shifts. Such collateral damage makes the backdoor highly conspicuous and easily identifiable by standard security filters.

Localized Stealthy Injection via Top-K Routes. Our LoCo framework provides a principled method to execute highly stealthy and data-efficient injections. Instead of treating the target model as a black box, we utilize our framework to pinpoint its exact structural vulnerabilities. Before the attack, we probe the frozen model with the target context (e.g., “a coffee cup” if the goal is to inject a specific coffee brand logo). We extract the localization map to identify the top-K layer-scale-position routes that are most responsive to this specific context.

During the data poisoning process, we compute the standard injection loss $\mathcal { L } _ { \mathrm { i n i e c t } }$ . However, we restrict the backpropagation strictly to these highly localized routes, denoted as $\dot { \theta } _ { K } ^ { * }$ . The rest of the model architecture remains entirely frozen.

This ultra-sparse parameter update forcefully embeds the malicious payload exactly where the model naturally processes the relevant semantic context. Consequently, our localized injection achieves a near-perfect attack success rate with minimal poisoned data. It perfectly preserves the original image quality and ensures the backdoor remains completely hidden during benign generation. This downstream application proves that concept routes act as fundamental structural bottlenecks. They can be neutralized for safety alignments or exploited to expose critical generative vulnerabilities.

## D Additional Experiments

## D.1 Implementation Details

Here, we discuss adaptations of baselines in detail. Most existing concept erasure methods were originally designed for continuous diffusion models or flow-matching architectures. In those paradigms, the optimization objective relies on modifying the predicted continuous noise or vector fields. However, next-scale VAR models operate by predicting discrete tokens in a probability space. To ensure a fair and rigorous comparison, we adapt these baselines to the VAR architecture by reformulating their noise-prediction objectives into next-scale probability alignment objectives. Furthermore, to prevent the severe error accumulation and generation collapse inherent to autoregressive decoding, we provide all baselines with the auxiliary frozen tokens $\bar { r } _ { < s } ^ { \mathrm { n e u t r a l } }$ generated by the original model, following the conditional alignment strategy discussed in Section 4.

ESD [34]. Erasing Concepts from Diffusion Models (ESD) derives its loss function from the classifier-free guidance (CFG) formulation. It steers the model predictions away from the target concept and towards a neutral concept. When adapted to the VAR framework, we apply this $\mathrm { C F G }$ based regularization directly to the predicted probability distributions at each scale. The objective is formulated as follows:

$$
\mathcal { L } _ { E S D } = \mathbb { E } _ { s } \left[ \left\| p _ { \theta ^ { * } } ( r _ { s } \mid r _ { < s } ^ { \mathrm { n e u t r a l } } , y ^ { c } ) - p _ { \theta } ( r _ { s } \mid r _ { < s } ^ { \mathrm { n e u t r a l } } , y ^ { c \mathrm { - n e u t r a l } } ) + \eta \Delta p \right\| _ { 2 } ^ { 2 } \right] ,\tag{15}
$$

where $\Delta p = p _ { \theta } ( r _ { s } \mid r _ { < s } ^ { \mathrm { n e u t r a l } } , y ^ { c } ) - p _ { \theta } ( r _ { s } \mid r _ { < s } ^ { \mathrm { n e u t r a l } } , y ^ { c \mathrm { \tiny { - n e u t r a l } } } )$ . Following the standard configuration in the original paper, we set the guidance scale $\eta = 1$ . Depending on the specific variant (ESD-x or ESD-u), we optimize either the cross-attention modules or the self-attention and feed-forward networks across the entire model.

MACE [41]. Mass Concept Erasure (MACE) is designed to erase multiple concepts simultaneously by combining a closed-form cross-attention refinement with LoRA fine-tuning. For the closed-form phase, MACE updates the weight matrices W in the text-to-image cross-attention modules such that the target concept embedding $y ^ { c }$ behaves identically to the neutral embedding $y ^ { c - \mathrm { n e u t r a l } }$ without explicit gradient optimization:

$$
W ^ { * } = \left( W y ^ { c \mathrm { \ - n e u t r a l } } ( y ^ { c } ) ^ { T } \right) \left( y ^ { c } ( y ^ { c } ) ^ { T } \right) ^ { - 1 } .\tag{16}
$$

After initializing the weights with this closed-form solution, MACE utilizes LoRA fine-tuning to balance erasure efficacy and general capability preservation. In our VAR implementation, we replace the original diffusion noise matching loss with the next-scale probability alignment loss:

$$
\mathcal { L } _ { M A C E } = \mathbb { E } _ { s } \left[ \left. p _ { \theta ^ { * } } ( r _ { s } \mid r _ { < s } ^ { \mathrm { n e u t r a l } } , y ^ { c } ) - p _ { \theta } ( r _ { s } \mid r _ { < s } ^ { \mathrm { n e u t r a l } } , y ^ { c \mathrm { n e u t r a l } } ) \right. _ { 2 } ^ { 2 } \right] .\tag{17}
$$

The LoRA parameters are optimized globally across all Transformer blocks.

EraseAnything [24]. Originally proposed for rectified flow transformers (e.g., Flux), EraseAnything formulates concept erasure as a bi-level optimization problem. It employs self-contrastive learning to ensure that removing unwanted concepts does not inadvertently harm performance on unrelated concepts. We seamlessly port this methodology to the VAR architecture by translating the flowmatching objectives into autoregressive probability alignments. The overall loss combines an erasure term for the target concept $y ^ { c }$ and a preservation term for unrelated concepts $y ^ { u } { } _ { ; }$

$$
\begin{array} { r l } & { \mathcal { L } _ { E A } = \mathbb { E } _ { s } \left[ \left\| p _ { \theta ^ { * } } ( \boldsymbol { r } _ { s } \mid \boldsymbol { r } _ { < s } ^ { \mathrm { n e u t r a l } } , \boldsymbol { y } ^ { c } ) - p _ { \theta } ( \boldsymbol { r } _ { s } \mid \boldsymbol { r } _ { < s } ^ { \mathrm { n e u t r a l } } , \boldsymbol { y } ^ { c \mathrm { - n e u t r a l } } ) \right\| _ { 2 } ^ { 2 } \right] } \\ & { \quad \quad \quad \quad \quad \quad + \lambda \mathbb { E } _ { \boldsymbol { y } ^ { u } , s } \left[ \left\| p _ { \theta ^ { * } } ( \boldsymbol { r } _ { s } \mid \boldsymbol { r } _ { < s } ^ { u } , \boldsymbol { y } ^ { u } ) - p _ { \theta } ( \boldsymbol { r } _ { s } \mid \boldsymbol { r } _ { < s } ^ { u } , \boldsymbol { y } ^ { u } ) \right\| _ { 2 } ^ { 2 } \right] . } \end{array}\tag{18}
$$

While our LoCo framework shares a similar bi-level optimization philosophy, the critical distinction lies in the optimization scope. EraseAnything applies these updates globally using full-model LoRA and relies on an explicit attention map regularizer to suppress activations. In contrast, our method operates exclusively on the highly specific Top-K localized layer-scale-position routes, strictly preventing global collateral damage without requiring auxiliary regularization terms.

## D.2 Additional Results

Theoretical Formulation of 3D Targeted Personalization. While Section 4 and Appendix D.1 introduce the high-level concept of localized model personalization, extending this intervention from traditional spatial domains to the 3D autoregressive generation space (Layer, Scale, Position) requires a rigorous mathematical formulation.

In standard full-parameter or global LoRA fine-tuning, parameter updates indiscriminately affect all spatial queries across all generation scales. This global perturbation is the primary cause of temporal flickering and motion degradation in video generation. To resolve this, we formalize a 3D targeted fine-tuning mechanism.

Given the aggregated localization map $L _ { l , s , p } ^ { c }$ for the base class noun $( \mathrm { e . g . , \ " { d o g } " ) }$ , we define a binary 3D routing mask $\mathcal { M } \in \{ 0 , 1 \} ^ { L \times K \times ( t _ { s } \times \dot { h } _ { s } \times w _ { s } ) }$ . We set $\mathcal { M } _ { l , s , p } = 1$ if the triplet $( l , s , p )$ belongs to the Top-K localized concept routes, and 0 otherwise.

During the DreamBooth fine-tuning process, we freeze the original model weights θ and introduce trainable low-rank adapters $\theta _ { \mathrm { L o R A } }$ . Crucially, instead of applying these adapters globally, we gate the adapter’s activation using our 3D routing mask. For a specific query position $p$ at scale s and layer l, the updated hidden state (or cross-attention output) $\tilde { h } _ { l , s , p }$ is computed as:

$$
\tilde { h } _ { l , s , p } = h _ { l , s , p } + \mathcal { M } _ { l , s , p } \cdot \Delta h _ { l , s , p } ( x ; \theta _ { \mathrm { L o R A } } )\tag{19}
$$

where $h _ { l , s , p }$ is the frozen base model’s output, and $\Delta h _ { l , s , p }$ is the residual shift proposed by the LoRA module.

By injecting the mask directly into the forward computation graph, the backpropagated gradients for $\theta _ { \mathrm { L o R A } }$ are strictly bottlenecked by $\mathcal { M } _ { l , s , p } .$ . The localized objective is then optimized as:

$$
\operatorname* { m i n } _ { \theta _ { \mathrm { { L o R A } } } } \mathbb { E } _ { r \sim X _ { s u b } , s } \left[ - \log p _ { \theta , \theta _ { \mathrm { { L o R A } } } } ( r _ { s } | r _ { < s } , y ^ { s u b } ) \right] + \lambda { \mathcal { L } } _ { p r i o r }\tag{20}
$$

This formulation guarantees that the customized subject identity is embedded exclusively within the specific layer-scale-position pathways responsible for the semantic class, leaving the vast majority of the spatiotemporal representations untouched.

Qualitative Video Personalization Results. Building upon this 3D localized formulation, we present further qualitative evaluations of our LoCo framework on video personalization using the InfinityStar [40] model. By leveraging our mathematically constrained 3D intervention, we successfully confine the subject-driven optimization strictly to the targeted routes. As demonstrated in Figure ${ \bar { 8 } } ,$ Figure 9 and Figure 10, this ultra-sparse update yields superior temporal consistency and subject fidelity.

## D.3 Additional Ablation Study

While our main results demonstrate the downstream effectiveness of targeted interventions, this section provides deeper evidence that our (Layer, Scale, Position) triplets represent the fundamental causal routes of concepts. We verify this through counterfactual interventions, stability analysis across contexts, and a comparison with exhaustive search.

Causal Intervention via Bottom-K and Random-K. To prove that our localized routes are the causal bottlenecks for concept emergence, we compare our Top-5% intervention against two control groups. First, the Bottom-5% strategy intervenes on positions with the lowest attention responses. Second, the Random-5% strategy selects positions randomly across all layers and scales. We evaluate these on the concept erasure task.

As shown in Table $^ { 6 , }$ intervening on the Bottom-5% positions results in zero concept suppression, with Target CLIP scores remaining nearly identical to the base model (0.282 vs. 0.284 on Infinity). Conversely, while Random-5% intervention slightly reduces the target concept, it causes a severe drop in Background CLIP and image fidelity (from 0.312 to 0.264). In contrast, our Targeted Top-5% achieves the lowest Target CLIP (0.185) while maintaining perfect background preservation (0.311). This explicitly demonstrates that concept routing in VAR models is highly localized. Successful intervention depends strictly on identifying these specific causal routes rather than simple parameter sparsity.

“A golden retriever running joyfully across a sunlit green meadow, leaping into the air to catch a bright red flying disc, with tall grass and trees gently swaying in the background.”

“A sleek silver sports car driving aggressively down a winding mountain road at sunset, camera panning alongside the vehicle to capture the reflection of the orange sky and clouds moving rapidly across the windshield.”  
“… rendered as a dynamic, monochromatic charcoal pencil sketch.”  
![](images/90d0a9c429e02f4b24a6dae4f547d6e53274098c351da51d5a929e0385b95e1d.jpg)  
Figure 8: Qualitative video personalization results.

![](images/c2cf3ff14b6667e12a66cb7dbe542679dba9cb30406e878ab3bdc7123ea6f344.jpg)

“… in a retro 16-bit arcade pixel art style.”  
![](images/e3d5db6fe4a57fe37afb728c6abeef4410a69447a1c1859fb2745b0c5175e4b6.jpg)

“… in a Victorian steampunk style.”  
![](images/c5c83e3b3a76f7e546024d78dfd10bbb0ce8b2b8dd93b9d03a685fbc02b4282c.jpg)

![](images/b39760580bf407d431ee425799e410c77e4be7dc1b19b81db8e16d3ba4c6bc8c.jpg)  
Figure 9: Qualitative video personalization results.

Cross-Context Stability of Concept Routes. A key property of a structural route is its invariance across different generation trajectories. We measure the stability of our localized triplets by calculating the Intersection over Union (IoU %) of the Top-5% positions across 100 random seeds (Cross-Seed) and 20 diverse prompts per concept (Cross-Prompt).

The results in Table 7 show that the identified routes are remarkably stable. The average Cross-Seed IoU reaches 89.4%, proving that concept emergence is a deterministic property of the network architecture rather than an artifact of specific noise. Furthermore, the high Cross-Prompt IoU (83.4%) confirms that a specific concept (e.g., “Nudity”) consistently utilizes the same neural pathways regardless of the surrounding textual context. In contrast, the IoU between two entirely different concepts (Cross-Concept) is less than 5%, highlighting the extreme specificity of our localized routes.

“A young woman sitting at a cozy wooden cafe table by a large rain-streaked window, gently blowing on a steaming cup of coffee before taking a slow sip, her hair moving slightly as she exhales.”  
![](images/86eb687ef959041a8d268f1418c2ba2bf55138218f40f369df50d3874c29c3c1.jpg)  
Figure 10: Qualitative video personalization results.

Table 6: Causal Verification via Counterfactual Interventions. Metrics are averaged over 500 prompts for concept erasure from LoCoBench. Targeted Top-5% intervention is strictly necessary to achieve significant concept erasure (Target CLIP ↓) without compromising background utility (Background CLIP ↑).
<table><tr><td rowspan="2">Intervention Strategy</td><td colspan="2">Infinity (VAR)</td><td colspan="2">HunyuanImage-3.0</td></tr><tr><td>Target CLIP↓</td><td>Background CLIP ↑</td><td>Target CLIP ↓</td><td>Background CLIP ↑</td></tr><tr><td>Base Model (No Erasure)</td><td>0.284</td><td>0.312</td><td>0.291</td><td>0.320</td></tr><tr><td>Bottom-5% Intervention</td><td>0.282</td><td>0.311</td><td>0.289</td><td>0.319</td></tr><tr><td>Random-5% Intervention</td><td>0.252</td><td>0.264</td><td>0.258</td><td>0.271</td></tr><tr><td>Targeted Top-5% (Ours)</td><td>0.185</td><td>0.311</td><td>0.190</td><td>0.318</td></tr></table>

Comparison with Brute-Force Localization. To further demonstrate the efficiency and reliability of our attention-based localization, we compare it against a brute-force search baseline. In a next-scale VAR model with L layers, S scales, and P spatial positions per scale, the search space for the optimal sparse positions is combinatorial and computationally intractable. For a tractable comparison, we implemented a macroscopic brute-force baseline that exhaustively evaluates all L × S layer-scale blocks to find the optimal intervention region.

Our method identifies the targeted concept routes in a single forward pass by aggregating the internal attention responses. In contrast, the brute-force baseline requires evaluating every candidate block independently. Consequently, for a standard VAR architecture, our approach offers a strict O(L×S)× speedup during the localization phase.

Despite this massive reduction in search cost, our localized fine-tuning achieves highly comparable results to the optimal brute-force solution. On the concept erasure task, the computationally expensive brute-force optimal region resulted in a Target CLIP score drop of 0.102 (from 0.284 to 0.182). Our single-pass localization method achieved a nearly identical Target CLIP drop of 0.099 (from 0.284 to 0.185) while strictly maintaining the same level of background preservation. These results systematically confirm that internal attention dynamics provide a highly reliable and maximally efficient signal for pinpointing concept bottlenecks, entirely eliminating the need for exhaustive search.

Table 7: Structural Stability and Specificity. We report the Jaccard Similarity (IoU %) of localized Top-5% triplets. High Cross-Seed and Cross-Prompt stability, combined with low Cross-Concept overlap, confirms that our method identifies unique and invariant neural bottlenecks.
<table><tr><td>Concept Category</td><td>Cross-Seed IoU ↑</td><td>Cross-Prompt IoU ↑</td><td>Cross-Concept IoU ↓</td></tr><tr><td>Van Gogh (Style)</td><td>89.4</td><td>84.2</td><td>4.1</td></tr><tr><td>The Batman (Copyright)</td><td>91.2</td><td>85.7</td><td>3.8</td></tr><tr><td>Nudity (Safety)</td><td>87.5</td><td>81.4</td><td>5.2</td></tr><tr><td>Eiffel Tower (Place)</td><td>90.1</td><td>83.6</td><td>4.5</td></tr><tr><td>Average</td><td>89.6</td><td>83.7</td><td>4.4</td></tr></table>

## E Limitations

Our LoCo framework successfully localizes conceptual knowledge across Layer, Scale, and Position dimensions. This structured localization is highly effective and enables ultra-sparse targeted interventions for downstream applications. However, our current approach relies on an empirically chosen sparsity threshold (e.g., the top-5% of routed positions). We determine this optimal capacity based on external evaluation metrics like CLIP scores and downstream task performance. The framework does not currently calculate the absolute minimum capacity required for a specific concept automatically. Different concepts may inherently require varying route capacities depending on their visual complexity. A promising future direction is to dynamically estimate the optimal sparsity level K for each concept using purely internal model signals. For example, future methods could leverage the entropy or peak sharpness of the attention response map without requiring external evaluation feedback.

Additionally, while our LoCoBench dataset and counterfactual experiments rigorously validate the causality of these localized routes, the deep semantic entanglement in large-scale generative models presents an ongoing challenge. Highly correlated concepts may naturally share overlapping neural pathways. This makes perfectly disjoint localization difficult for highly compositional prompts. Developing synthetic model architectures with mathematically known structural ground truths could further strengthen the validation of future interpretability and routing disentanglement methods.

## F Ethics Statement

This work is conducted for academic research on the interpretability and controllability of modern generative models. Our central goal is to understand how visual concepts are routed inside next-scale autoregressive architectures, and to provide a principled way to analyze where and when specific concepts emerge during generation. We believe that such understanding is important for building generative systems that are more transparent, more controllable, and better aligned with safety and compliance requirements.

The proposed framework is designed to support responsible model adaptation. By localizing conceptrelated routes across layers, scales, and positions, LoCo enables targeted interventions that modify only a small and relevant part of the generation process. This helps reduce unnecessary changes to benign model behavior while improving control over specific concepts. In this sense, our method contributes to safer and more reliable deployment of generative models, especially when models need to respect content policies, copyright constraints, or application-specific requirements.

All datasets and experiments in this paper are used only for research and evaluation purposes. The concept categories in LoCoBench are constructed to study model behavior under diverse semantic settings, rather than to encourage any harmful use. Our results are intended to provide technical insights for the community and to help future work develop more interpretable, compliant, and controllable generative architectures.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: The claims we presented in the abstract and introduction are clearly stated and fully aligned with the contributions of this paper.

Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss the limitations in Appendix E.

Guidelines:

• The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate “Limitations” section in their paper.

• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [N/A]

Justification: None of theoretical assumptions.

## Guidelines:

• The answer [N/A] means that the paper does not include theoretical results.

• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.

• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: We provide all necessary implementation details required to reproduce the main experimental results in Appendix D.1 to ensure reproducibility.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

## Answer: [Yes]

Justification: We will publicly release the code and data once they have been finalized and properly prepared for distribution.

## Guidelines:

• The answer [N/A] means that paper does not include experiments requiring code.

• Please see the NeurIPS code and data submission guidelines (https://neurips.cc/ public/guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes]

Justification: We provide all relevant experimental details such as dataset setup, evaluation procedures, and experimental conditions in Sec 5.1 and the appendix, sufficient for understanding and reproducing the results.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: The experiments statistics are precisely run with given experimental conditions. Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors).

• It should be clear whether the error bar is the standard deviation or the standard error of the mean.

• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).

• If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: See Sec. 5.3.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: The research presented in this paper fully complies with the NeurIPS Code of Ethics.

Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: We briefly discuss potential impacts of our work in the conclusion section.

Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.

• If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A]

Justification:

Guidelines:

• The answer [N/A] means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: All external assets used in this work are properly credited with citations. We respect all applicable licenses and terms, which are explicitly mentioned where relevant.

Guidelines:

• The answer [N/A] means that the paper does not use existing assets.

• The authors should cite the original paper that produced the code package or dataset.

• The authors should state which version of the asset is used and, if possible, include a URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: We introduce a new dataset in the paper. Comprehensive documentation will be made available together with the dataset to support reproducibility and adoption.

Guidelines:

• The answer [N/A] means that the paper does not release new assets.

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [Yes]

Justification: Our research does not involve crowdsourcing experiments or studies with human subjects.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: We do not include human subjects in this paper.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [Yes]

Justification: We used LLM for dataset construction and described it in detail in the paper. Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

• Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.