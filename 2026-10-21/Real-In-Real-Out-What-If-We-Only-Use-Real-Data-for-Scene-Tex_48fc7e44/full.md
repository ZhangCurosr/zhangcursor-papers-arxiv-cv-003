# Real In, Real Out: What If We Only Use Real Data for Scene Text Editing?

Xingsong Ye<sup>1,2</sup>, Yongkun Du<sup>1,2</sup>, Jiaxin Zhang<sup>3</sup>, Chong Sun<sup>3</sup>, Zhineng Chen<sup>1,2,†</sup>

<sup>1</sup>Institute of Trustworthy Embodied AI, Fudan University,

<sup>2</sup>Shanghai Key Laboratory of Multimodal Embodied AI, <sup>3</sup>WeChat Vision, Tencent Inc.

## Abstract

Aiming to modify textual content while preserving the original style, Scene Text Editing (STE) has significant practical value in many applications. STE typically requires paired images before and after editing for training. However, recent approaches rely heavily on synthetic paired data and complex disentanglement strategies. They struggle to faithfully capture the diversity and complexity of natural scenes, often producing artifacts or hallucinations. In this work, we challenge the common practice and investigate whether STE can be driven by turning each real sample into its own paired supervision. Concretely, we explore multiple ways to disrupt the original image (e.g., Crop, Shufle, Mask) to construct self-paired training data, and build a strong baseline (TextRIRO) upon a conditional difusion architecture. Our experiments show that combining character-column shufling with image cropping (Shufle&Crop) efectively destroys text semantics while preserving key style cues, enabling the model to learn editing behaviors directly from the real text image. With this concise yet real-oriented design, TextRIRO achieves promising performance in both accuracy and visual realism across real-world benchmarks, substantially mitigating the synthetic-to-real domain gap. Furthermore, we leverage TextRIRO to build TextRIRO-3M, a large-scale realistic dataset for long-text recognition by concatenating edited and original images. Experimental results show that training recognition models with TextRIRO-3M significantly improves recognition accuracy, demonstrating that TextRIRO provides highly useful training resources for downstream tasks. Overall, TextRIRO lifts STE to realistic, reliable, and widely applicable new levels.

Correspondence: zhinchen@fudan.edu.cn Code: https://github.com/YesianRohn/TextRIRO

## 1 Introduction

Scene Text Editing (STE) aims to extract the style of the given text and replace it with a new string, while maintaining both visual realism and style consistency [36]. STE is typically formulated as a generative task, and its applications grow with the advancement of generative models. It addresses practical needs such as replacing text in images for artistic creation and also facilitates various downstream scene text tasks. Notably, STE shows significant potential in generating data [39] for scene text recognition (STR) [6, 10, 11, 48], as it can generate samples with the same style but diferent content from existing images.

![](images/29d0d8cb4224251964c175f4d0428e4e3302fe63ac6fbb3a163578efd4ec70ff.jpg)  
(a) Previous paired synthetic training data

![](images/ce292353fbfe9eaef543af06a70dc58b7aedcce1a27d0c571506605bed8c2910.jpg)  
(b) Semi-supervised with unpaired real data

![](images/d2230a42f5a1b2673956bf3e252f6a19334194bc8c431a1f2298e42b5eee12b9.jpg)  
(e) Bad cases of (a) and (b)

![](images/f80420970a4d569bfd9d22a232805fbfe05ca8d900cb7569214b2e24be41ce8e.jpg)  
(c) Inpainting-based training on real data

![](images/8e2d6ff6437c836b6c1e55fccdb25f4f33e2bddfce6fa034d26e38002d033b89.jpg)

![](images/5ba224e05a7ea017ab4a95f24edd8041191d7a88022cbbf6994ec8f3827dd5ce.jpg)

(f) Inpainting without scene text around  
![](images/95a8be4c6c34772746ea4a2d8b498f5250b937019e4b5cf3b01d8cf16fffccce.jpg)  
Figure 1 A data-centric view of Scene Text Editing. The comparison of the previous methods and TextRIRO between the data utilization paradigm (left) and the editing results (right). In (e), we present the artifacts and hallucinations of the wrong editing results in (a) and (b).

However, it is hardly possible to get diferent texts with the same background style in real-world scenarios. To address this, previous methods [20, 29, 36, 38, 43] rely heavily on synthetic data, as shown in Fig. 1(a). On the one hand, two-stage explicit disentanglement methods [29, 36, 38] use rendering tools to synthesize paired images with identical style but diferent text over a background image, followed by foreground-background separation and fusion. But they fail to separate text from the background, resulting in incomplete removal of the source text, as shown in cases (1) and (2) in Fig. 1(e). On the other hand, implicit disentanglement approaches [20, 43] leverage various pre-trained models to extract features for directly generating target images, which is less controllable, resulting in hallucinations such as case (3), with misspelled or missing characters. Recent eforts [9, 22, 26] have explored semi-supervised learning with real data by erasing itself and writing back or paired training using pseudo-labeled data, as illustrated in Fig. 1(b). Nevertheless, synthetic data still remains dominant. As evident from the editing results in Fig. 1(e), this branch of methods easily leads to case (4): even when the target text is generated, it often appears mismatched in style or contains visible artifacts. Furthermore, inpainting-based STE paradigms (Fig. 1(c)) mask the text region and attempt to restore it using surrounding context [2, 4, 15, 39]. However, the lack of suficient style cues makes it dificult to recover the desired appearance, as shown in Fig. 1(f).

We are motivated by an observation that prior work has under-emphasized: the domain gap in STE is primarily a data problem, and only secondarily an architecture problem. We therefore ask the simplest possible version of the question:

What if we only use real data for scene text editing, with neither synthetic triplets nor expensive paired collection?

The core challenge is to extract stable style guidance from the source text image while suppressing the side efect from semantic text content. Our key insight is that scene text style mainly lies in low-level cues such as color, texture, and illumination, while semantic text is a high-level sequential structure. Based on this, we explore multiple self-paired real-data schemes that force the model to focus on style rather than semantics. Specifically, we experiment with (i) shufling and recombining the text region either column-wise or block-wise (Shufle), (ii) cropping local sub-regions (Crop), and (iii) masking parts of the original image (Mask). Empirically, we find that combining column-wise Shufle with Crop (Shufle&Crop) achieves the best performance.

To realize the idea, we propose TextRIRO, a conditional difusion model that takes only Real data as Input and produces Realistic Output for scene Text editing. During training, TextRIRO applies “Shufle&Crop” to a real text image to separate its style and semantics, and then reconstructs the original image. By removing the reliance on synthetic data, it overcomes the bottleneck of high-quality generation during inference. For style features, we process them into multi-scale representations and then inject them into the conditional U-Net using the ControlNet-like process [45] for eficient integration. For text encoding, we use an STR pre-trained encoder [16] to extract character-level features, which improves text understanding and enables robust STE transfer.

TextRIRO removes the need for explicit or implicit disentanglement and avoids using the synthetic domain as the main training support. On real benchmarks for STE, it produces more realistic results with markedly reduced “synthetic flavor.” It also achieves higher accuracy on equal-length editing and variable-length editing tasks. These results show that self-paired training on real data efectively moves STE toward realistic generation. We further ask whether STE can help STR data generation [39, 40, 42]. We find that TextRIRO already outperforms existing STE methods, but the editing accuracy still limits direct use as training data. We then use STE as data expansion by concatenating the original image with its edited counterpart. This preserves style but adds content, which yields high-quality, scene-consistent samples for long-text recognition. Starting from the labeled real dataset U14M-Filter [7, 16], we build a 3M-scale dataset, TextRIRO-3M. On Union14M-Benchmark [16], training an STR model on mixed TextRIRO-3M and real data improves overall accuracy by 1.91% over a real-only baseline, and improves the Multi-words subset by 3.28%. These gains indicate that current STR training data are not saturated and STE-based synthetic data substantially boost STR.

In summary, our main contributions are as follows:

• We propose TextRIRO, a novel STE method that trains solely on real data via self-paired supervision. It performs realistic editing without explicit or implicit disentanglement, and it alleviates artifacts and hallucinations caused by the synthetic-to-real domain gap.

• TextRIRO achieves more realistic and accurate edits on real benchmarks and attains state-of-the-art (SOTA) performance under the stronger criterion of “training with edited data.”

• Built on TextRIRO, we create TextRIRO-3M, a large-scale realistic composite dataset for long-text recognition. It significantly improves STR, which validates both the realism and the utility of our edited data and demonstrates that STE efectively benefits STR.

## 2 Related Work

Model Perspective on STE Traditional STE methods evolve from GAN-based models [20, 29, 36, 38] to difusion models [2, 15, 32, 33, 43, 47] and, more recently, to Unified Multimodal Models (UMMs) [9, 44]. Early STE models employ GANs to directly edit foreground text, as seen in SRNet [36] and its multistage framework for reconstructing background, rendering text, and fusing outputs. Later works expand capabilities: SwapText [38] handles curved text, STEFANN [29] focuses on character-level control, and TextStyleBrush [20] leverages StyleGAN2 [19] for self-supervised style-content disentanglement. Difusion models advance STE further: DifSTE [15] uses dual encoders, DifUTE [2] introduces inpainting and OCR-guided encoding, TextDifuser [3] and UDifText [47] use segmentation masks for better generation.

![](images/56bf0f0782c516493032df56bd01b956c989fc1793f54850094ccd4d8d83f397.jpg)  
Figure 2 TextRIRO pipeline. (a) At training time, a single real image $I _ { \mathrm { t g t } }$ is pushed through a content-destructive, style-preserving operator to produce a pseudo-source $I _ { \mathrm { s r c } . }$ . A ViT-based style encoder emits multi-scale features that are injected into the denoising U-Net through ControlNet-style zero-convolutions, while a text encoder pre-trained on real scene text with an autoregressive STR objective supplies the target-text conditioning. (b) At inference time, the same U-Net follows the standard SD denoising schedule. (c) Our text encoder is a lightweight character tokenizer trained jointly with an STR decoder via next-token prediction. (d) The edited image is validated by an independent STR model and, if accepted, concatenated with its original to build a style-consistent long-text sample for STR. : frozen, : trainable.

Frameworks like AnyText [33] and AnyText2 [32] unify multilingual editing via ControlNet [45]. TextCtrl [43] disentangles style conditions directly in difusion models. Meanwhile, DARLING [44] unifies scene text editing, removal, and recognition tasks to leverage these tasks to disentangle task-specific features and RS-STE [9] utilizes scene text recognition to assist scene text editing. These two approaches highlight synergy across scene text tasks.

Data Perspective on STE Model advancements in STE are closely tied to the evolution of training data. Early methods use synthetic paired data by rendering tools, e.g., SRNet-Datagen [36]. To include real-world data, RewriteNet [22] uses masked regions and MOSTEL [26] proposes an erasing and writing-back strategy. UMM-based RS-STE [9] adopts reverse self-supervision on real images. In difusion-based STE, some models (e.g., TextCtrl [43]) disentangle features using synthetic data, while others [3, 33] view STE as masked image inpainting, releasing large real datasets like Anyword-3M [33] and MARIO-10M [3]. However, style consistency can be dificult when context lacks similar references. Benchmarks and datasets for application are also advancing: ScenePair [43] provides high-quality real-world evaluation, and TextSSR [39] generates millions of edited images for STR tasks, demonstrating STE’s broader utility.

## 3 Method

The STE process can be formally defined as follows. Denote the input source image as $I _ { \mathrm { s r c } } ,$ , the target text as $T _ { \mathrm { t g t } }$ , and the editing model as M. The objective is to produce the target image $I _ { \mathrm { t g t } }$ such that:

$$
{ \cal I } _ { \mathrm { t g t } } = { \cal M } { \cal I } _ { \mathrm { s r c } } , { \cal T } _ { \mathrm { t g t } } .\tag{1}
$$

However, Eq. 1 characterizes the ideal training and inference procedure. In practice, training data rarely provide real triplets $I _ { \mathrm { s r c } } , T _ { \mathrm { t g t } } , I _ { \mathrm { t g t } }$ . To bridge this gap, Sec. 3.1 introduces a self-paired training strategy, Sec. 3.2 presents the unified TextRIRO architecture, and Sec. 3.3 leverages the trained model to expand STR datasets with long-text instances.

## 3.1 Self-Paired STE Data

Most prior data pipelines [26, 36] construct $I _ { \mathrm { s r c } } , T _ { \mathrm { t g t } } , I _ { \mathrm { t g t } }$ triplets primarily from synthetic text images. The domain gap between synthetic and real scenes limits the ability of STE models and their applications to downstream scene text tasks. Recent methods [9, 26] introduce some real data for unpaired training, yet synthetic samples remain dominant and only partially bridge this gap.

Algorithm 1 Self-Paired STE Data Generation   
Require: Scene text image $I _ { \mathrm { t g t } } ,$ , corresponding text label $T _ { \mathrm { t g t } }$ , shufle probability p<sub>shufle</sub>   
Ensure: Generated self-paired source image $I _ { \mathrm { s r c } }$   
1: Let $L = | T _ { \mathrm { t g t } } |$ denote the length of the text label.   
2: Partition $I _ { \mathrm { t g t } }$ along its longest axis into L equally sized segments: $\boldsymbol { \mathcal { B } } = \{ B _ { 1 } , B _ { 2 } , . . . , B _ { L } \}$   
3: Sample r ∼ U0, 1.   
4: if $r < p _ { \mathrm { s h u f f l e } }$ then   
5: Randomly permute the order of the segments to obtain $B ^ { \prime } = \{ B _ { 1 } ^ { \prime } , . . . , B _ { L } ^ { \prime } \}$   
6: else   
7: Set $B ^ { \prime } = B$ (i.e., maintain original order).   
8: end if   
9: Choose $N \leq L$ and randomly select a consecutive subset of N segments from $B ^ { \prime } .$   
10: Concatenate the selected segments to form $I _ { \mathrm { s r c } }$   
11: Return $I _ { \mathrm { s r c } } , T _ { \mathrm { t g t } } , I _ { \mathrm { t g t } }$

When using an image as $I _ { \mathrm { s r c } } ,$ our goal is to remove its textual content while preserving its style. This motivates a simple strategy: scrambling the original image disrupts content while retaining stylistic features. However, since text comprises discrete characters, excessive shufling fragments glyphs and impairs shape consistency. This presents a trade-of: we aim to maximize content removal but minimize stylistic degradation. (1) Text typically aligns along its long axis, and its content and length are known. We exploit this structural prior by dividing the image along its long axis into L (L is the number of characters) equal bins. By reordering these bins, we generate the source image $I _ { \mathrm { s r c } }$ (see “Shufle” in Fig. 2), thereby breaking semantic continuity while maintaining local appearance cues. (2) Stripe-based shufling alone is insuficient for complex text layouts, such as curved or multi-oriented arrangements. To address this, we propose an alternative that reconstructs the image through its sub-patches. The global text arrangement is locally preserved: tilts and curvature manifest within contiguous sub-images. We form $I _ { \mathrm { s r c } }$ by cropping the original image (as illustrated by “Crop” in Fig. 2), yielding self-paired data that retain the stylistic integrity of intricate layouts. (3) Combining these two strategies results in a more robust and diverse self-paired mechanism. As shown in Fig. 2 (“Shufle then Crop”), with probability $p _ { \mathrm { s h u f f l e } } ,$ , shufling is applied before cropping. Otherwise, cropping is applied directly to the original image. Alg. 1 presents the complete algorithm of our self-paired dataset. The subsequent ablation study presents other self-paired methods (e.g., Crop Only, Shufle Only, Mask) and results, further demonstrating the optimality of the adopted algorithm.

![](images/86b6b54ce2062ead24608febf606ab68d1a0381211f525ccedb8f3751ccd1668.jpg)  
Figure 3 Qualitative comparison on ScenePair. “Target Reference” is the real paired ground truth. “Rendering Tool” composites the target string on the segmented foreground/background of the source and serves as an accuracy-only reference. TextRIRO is the only method that consistently reproduces the stroke thickness, perspective and illumination of the reference without leaving a synthetic residue.

## 3.2 Model Architecture

With the constructed self-paired triplets, TextRIRO maps the perturbed source image $I _ { \mathrm { s r c } }$ and target text $T _ { \mathrm { t g t } }$ into style and text control signals for a ControlNet-like difusion generator [45], as shown in Fig. 2. The style encoder follows TextCtrl [43]: a ViT backbone [5] extracts image representations, and a lightweight spatial pyramid aggregates them into multi-scale features that are additively injected into the U-Net. For text, we keep character-level encoding [43, 47] but replace synthetic glyph/CLIP pre-training with an autoregressive STR objective on real scene text [16, 30]. This improves real-domain character awareness and aligns recognition with editing.

The generator contains a VAE for latent mapping, a U-Net for denoising, and the above style/text encoders. The VAE and text encoder are frozen, while the style encoder and U-Net are jointly trained with the standard noise-prediction difusion objective conditioned on style and text. This design is intentionally simple and backbone-agnostic: stronger difusion backbones or conditioning modules can be substituted when more compute is available. We also evaluate a DiT-based conditional control alternative in Tab. 5, where ControlNet-style multi-scale injection remains more efective for fine-grained scene text editing. Detailed feature equations, tokenization, and the training loss are provided in Appendix Sec. 6.1.

## 3.3 Generating Long-Text Recognition Data

After training TextRIRO, we generate a style-matched image with the same length but diferent content for each real input. To ensure correctness, we filter generated samples with an STR model trained on real data and keep only instances whose predicted text matches $T _ { \mathrm { t g t } }$ . We then concatenate each filtered image with its original reference to build long-text recognition data, doubling the sequence length. The detailed pipeline and visualizations are illustrated in Fig. 2 (d). This procedure substantially broadens coverage for rare scenarios, such as long and multi-word instances, and it establishes a practical connection between STE and STR.

## 4 Experiments

## 4.1 Editing Performance Comparison

Results on ScenePair Benchmark. We evaluate the proposed TextRIRO on the ScenePair [43] against both specialized STE methods and general editing models. The specialized baselines include the GAN-based SRNet [36] and MOSTEL [26], the difusion-based inpainting methods AnyText [33] and TextSSR [39], the direct difusion editing method TextCtrl [43], and the UMM-based RS-STE [9]. The general models include Flux.1 Kontext [21] and Qwen-Image-Edit [35]. For these two models, we use a fixed prompt: “Edit the text to $\prime T _ { \mathrm { t g t } } { ' } , { ' }$ where $T _ { \mathrm { t g t } }$ is the designated target. We follow the ScenePair protocol for evaluation. For visual realism, we report FID [13], which measures diferences between deep feature distributions. For text rendering correctness, we report word accuracy (ACC) and normalized edit distance (NED), computed with the oficial text recognizer [1]. We also evaluate the recognizer on the target images.

The quantitative results in Tab. 1 reveal that early GAN-based methods underperform both in accuracy and visual realism. When difusion methods were introduced to the STE task, both metrics improve substantially. However, because these methods rely on synthetic data, a gap to real-data performance remains. TextRIRO leverages a self-paired real data training paradigm, achieving near-realistic accuracy and fidelity. Notably, RS-STE achieves higher text recognition accuracy, even surpassing the ground truth, but exhibits poorer FID scores, indicating that it trades realism for correctness and produces strongly synthetic-looking samples. General-purpose editing models complete part of the task, but they lag behind specialized editors, suggesting that expert STE models remain valuable.

Table 1 ScenePair results. Boldface indicates the best results for each metric, while underlines highlight the second best results.
<table><tr><td>Method</td><td>ACC (%)↑</td><td>NED↑</td><td>FID↓</td></tr><tr><td>SRNet 2019</td><td>17.84</td><td>0.478</td><td>49.22</td></tr><tr><td>MOSTEL 2023</td><td>37.69</td><td>0.557</td><td>49.19</td></tr><tr><td>AnyText 2024</td><td>51.12</td><td>0.734</td><td>51.79</td></tr><tr><td>TextCtrl 2024</td><td>84.67</td><td>0.936</td><td>43.78</td></tr><tr><td>TextSSR 2025</td><td>79.69</td><td>0.924</td><td>45.37</td></tr><tr><td>RS-STE 2025</td><td>91.80</td><td>0.981</td><td>49.59</td></tr><tr><td>Flux.1 Kontext 2025</td><td>12.81</td><td>0.460</td><td>41.59</td></tr><tr><td>Qwen-Image-Edit 2025</td><td>53.16</td><td>0.825</td><td>49.91</td></tr><tr><td>TextFlux 2026</td><td>80.40</td><td>0.911</td><td>54.64</td></tr><tr><td>TextRIRO</td><td>85.23</td><td>0.964</td><td>41.53</td></tr><tr><td>Real (upper bound)</td><td>88.44</td><td>0.974</td><td>0.00</td></tr></table>

Fig. 3 presents qualitative visual comparisons: TextRIRO clearly outperforms other methods in edit correctness, style preservation, and realism. MOSTEL, due to its two-stage editing process and training on synthetic data, does not even recognize the existence of these texts, let alone erase them. Even when it attempts to edit, it leaves residual artifacts, as in the “Buses” example. TextCtrl uses implicit disentanglement: it keeps coarse structure during feature mapping but loses fine-grained style cues, so the edited text deviates from the original style. RS-STE’s samples further highlight its lack of visual authenticity and synthetic appearance, particularly for the “open” example. TextSSR, as an inpainting method, depends strongly on surrounding context: when nearby text shares a similar style (e.g., “open”), it works well, but without such references (e.g., “51”), its edits degrade. Among general-purpose editors, Flux.1 Kontext preserves the image style but rarely produces the correct target text. Qwen-Image-Edit understands the editing instruction, yet its outputs look rigid and over-synthesized, resembling results from a rendering tool.

Table 2 RIROCommon results. Comparison of TextRIRO with three other representative STE models. “E (Equal)” denotes text editing with equal-length substitutions, while “V (Variable)” corresponds to editing without text length limitations.
<table><tr><td rowspan="2">Method</td><td colspan="2">IIIT 2012</td><td colspan="2">SVT 2011</td><td colspan="2">IC13 2013</td><td colspan="2">IC15 2015</td><td colspan="2">SVTP 2013</td><td colspan="2">CUTE 2014</td><td colspan="2">Avg</td></tr><tr><td>E</td><td>V</td><td>E</td><td>V</td><td>E</td><td>V</td><td>E</td><td>V</td><td>E</td><td>V</td><td>E</td><td>V</td><td>E</td><td>V</td></tr><tr><td>MOSTEL 2023</td><td>16.85</td><td>12.85</td><td>10.82</td><td>3.58</td><td>10.04</td><td>7.54</td><td>11.49</td><td>6.15</td><td>7.91</td><td>3.98</td><td>7.89</td><td>5.02</td><td>10.83</td><td>6.52</td></tr><tr><td>TextCtrl 2024</td><td>64.42</td><td>11.31</td><td>76.97</td><td>10.17</td><td>74.45</td><td>12.99</td><td>73.16</td><td>15.65</td><td>70.85</td><td>10.44</td><td>64.16</td><td>11.30</td><td>70.67</td><td>11.98</td></tr><tr><td>RS-STE 2025</td><td>58.80</td><td>13.21</td><td>67.08</td><td>10.73</td><td>63.71</td><td>20.11</td><td>61.29</td><td>9.10</td><td>62.17</td><td>7.59</td><td>48.39</td><td>9.62</td><td>60.24</td><td>11.73</td></tr><tr><td>TextRIRO (ours)</td><td>82.74 50.47</td><td></td><td>84.39</td><td>46.14</td><td>86.23</td><td>53.49</td><td>77.91</td><td>45.28</td><td>79.69</td><td>42.31</td><td>64.87</td><td>48.12</td><td>79.31</td><td>47.64</td></tr></table>

Broader Editing Evaluation. We observe that ScenePair reveals a recognition accuracy ceiling of 88.44%, largely limited by the STR model. To obtain more faithful results, we adopt a stronger and third-party text recognizer, SVTRv2 [7], for evaluation. TextRIRO, together with TextCtrl and RS-STE, already approaches the realism of natural images in ScenePair. Subsequent experiments therefore focus on these three methods, and we conduct an upgraded evaluation on the Common Benchmarks (hereafter RIROCommon). For each image, we provide two targets to edit: an equal-length text that matches the source length and a variable-length target text. We then report normalized accuracy like STR, where we remove non-vocabulary characters and spaces and lowercase all text before scoring.

Tab. 2 demonstrates TextRIRO’s superior performance on RIROCommon: it consistently outperforms alternative models for both equal-length and variable-length editing. In particular, its variable-length editing achieves an accuracy of 47.64%, markedly higher than competitors, attributable to the inherent diversity of self-paired training examples produced via Shufle&Crop. Conversely, RS-STE’s strategy of trading of style for accuracy fails under broader and more challenging conditions. And TextRIRO sustains strong performance, achieving equal-length editing accuracy 19.07% higher than RS-STE.

Table 3 Train-on-edits STR proxy. “Regular” indicates the average results on the IIIT, SVT, and IC13 datasets, while “Irregular” denotes the average results on the IC15, SVTP, and CUTE datasets.
<table><tr><td>Method</td><td>IIIT 2012</td><td>|SVT 2011 |IC13 2013</td><td></td><td></td><td>|Regular|IC15 2015|</td><td>|SVTP 2013</td><td>|CUTE 2014 |Irregular|</td><td></td><td>Avg</td></tr><tr><td>MOSTEL 2023</td><td>51.87</td><td>59.51</td><td>64.99</td><td>58.79</td><td>44.62</td><td>44.34</td><td>34.72</td><td>41.22</td><td>|50.01</td></tr><tr><td>AnyText 2024</td><td>73.27</td><td>78.83</td><td>82.61</td><td>78.24</td><td>66.08</td><td>70.46</td><td>48.27</td><td>61.60</td><td>69.92</td></tr><tr><td>TextCtrl 2024</td><td>79.40</td><td>84.39</td><td>89.61</td><td>84.47</td><td>72.06</td><td>77.36</td><td>53.12</td><td>67.51</td><td>75.99</td></tr><tr><td>RS-STE 2025</td><td>75.43</td><td>85.32</td><td>87.98</td><td>82.91</td><td>74.43</td><td>78.45</td><td>59.72</td><td>70.87</td><td>76.89</td></tr><tr><td>TextRIRO (ours)</td><td>90.53</td><td>89.18</td><td>92.65</td><td>90.79</td><td>81.00</td><td>84.03</td><td>82.64</td><td>82.56</td><td>86.67</td></tr><tr><td>Real (upper bound)</td><td>99.30</td><td>97.84</td><td>98.60</td><td>98.58</td><td>90.50</td><td>96.12</td><td>98.26</td><td>94.96</td><td>|96.77</td></tr></table>

STR Training Proxy Task. Although we employ a strong recognizer, it cannot reflect absolute editing accuracy. Following TextSSR [39], we use the edited text to train an STR model and treat it as a proxy evaluation of the realism and accuracy of generated text. Only suficiently realistic and accurate synthetic data will improve downstream recognition performance. This also serves as a practical measure of whether editing models can efectively generate usable training samples. We apply each STE model to U14M-Filter, editing each image with the same alternative string of equal length, and train STR models with edited images and their edited labels, then evaluate them on the STR protocol.

Tab. 3 confirms previous trends: MOSTEL’s outdated architecture and data strategy result in poor performance. TextCtrl and RS-STE utilize diferent architectures and training data, yet achieve similar results, with only a 0.90% diference in average accuracy. Both methods follow training paradigms dominated by synthetic data, revealing the upper bound and bottleneck of such approaches, as the domain gap between synthetic and real data persists despite model optimization. In contrast, TextRIRO’s self-paired training paradigm leverages real data to overcome this bottleneck, yielding an average accuracy of 86.67%, representing a 9.78% improvement over the next-best approach. This advantage is even more pronounced on irregular samples, with an 11.69% gap, highlighting TextRIRO’s ability to supplement scarce training domains.

## 4.2 Ablation Study

Contributions of each core component. Fig. 4 summarizes ablations over the TextRIRO architecture and self-paired training paradigm. Text encoding is essential: without pre-training, the model fails to learn reliable character correspondences, while STR pre-training performs better than glyph pre-training because it is learned directly from real scene text. SD 2.1 provides better VAE reconstruction than SD 1.5, and rectangular 64×256 inputs avoid the distortion caused by square resizing. For self-paired data construction, Shufle&Crop achieves the best accuracy/realism trade-of. Crop Only lacks diversity, Shufle Only breaks alignment, Mask destroys both content and style, Block introduces a strong distribution shift, and removing paired construction degenerates to identity copying. Fig. 5 visualizes key ablation findings. The Glyph Pre-train carries strong glyph priors and tends to lose style details, while the STR Pre-train better maintains the original style during editing. Alternative self-paired constructions also trail Shufle&Crop in both rendering accuracy and style fidelity: Shufle Only and Crop Only give less diversity, and Mask and Block create a large train–test gap.

![](images/291b4977125f510d7fb49841812f63006728af60306cbdaac3dc7e2140804b88.jpg)

<table><tr><td rowspan=1 colspan=1>Text Enc. Shape  VAE    Paired Op.</td><td rowspan=1 colspan=1>ACC↑NED↑FID↓</td></tr><tr><td rowspan=1 colspan=1>None  [256,256]SD 1.5Shuffle&amp;Crop</td><td rowspan=1 colspan=1>9.38  0.469 55.84</td></tr><tr><td rowspan=1 colspan=1>Glyph  [256,256]SD 1.5Shuffle&amp;Crop</td><td rowspan=1 colspan=1>53.44  0.815 40.36</td></tr><tr><td rowspan=1 colspan=1>STR   [256,256]SD 1.5Shuffle&amp;Crop</td><td rowspan=1 colspan=1>56.72 0.823 39.47</td></tr><tr><td rowspan=1 colspan=1>STR    [64,256]SD 1.5Shuffle&amp;Crop</td><td rowspan=1 colspan=1>78.91  0.951 54.04</td></tr><tr><td rowspan=1 colspan=1>STR    [64,256]SD 2.1Shuffle&amp;Crop</td><td rowspan=1 colspan=1>82.42  0.960 43.59</td></tr><tr><td rowspan=1 colspan=1>Glyph  [64,256]SD 2.1Shuffle&amp;Crop</td><td rowspan=1 colspan=1>81.95  0.952 45.96</td></tr><tr><td rowspan=1 colspan=1>STR    [64,256]SD 2.1  Shuffle only</td><td rowspan=1 colspan=1>45.78 0.791 39.10</td></tr><tr><td rowspan=1 colspan=1>STR    [64,256]SD 2.1   Crop only</td><td rowspan=1 colspan=1>80.08 0.954 45.46</td></tr><tr><td rowspan=1 colspan=1>STR    [64,256]SD 2.1      Mask</td><td rowspan=1 colspan=1>13.13 0.493 58.08</td></tr><tr><td rowspan=1 colspan=1>STR    [64,256]SD 2.1      Block</td><td rowspan=1 colspan=1>79.36  0.949 50.67</td></tr><tr><td rowspan=1 colspan=1>STR    [64,256]SD 2.1 None (identity)</td><td rowspan=1 colspan=1>4.37  0.176 33.28</td></tr></table>

Figure 4 Ablations on ScenePair. Left: visualizations of each self-pairing operator. Right: full factorial over text encoder pre-training, input shape, VAE variant and paired-data operator. Shufle&Crop strictly dominates every other operator when joined with STR-pretrained text conditioning and a rectangular input. “None” collapses to an identity mapping (no perturbation operation).  
![](images/1448ce884420f234ef7008d9bc0c5f3a5cce459f0ce9c2344b1cabe6eded16d6.jpg)  
Figure 5 Qualitative ablations. The showing setting is consistent with that in Fig. 3.

Layout-wise attribution of Shufle and Crop. To make the complementary roles of the two perturbations explicit, we add an attribution study (in Tab. 4) on straight (IC13), curved (CUTE), and multi-oriented (SVTP) subsets, separately running Shufle only, Crop only, and Shufle&Crop. The reading of our design is as follows. Shufle dominates for straight text, because it maximally breaks left-to-right semantics while preserving column-local style. Crop dominates for curved and multi-oriented text, because it preserves local tilt and curvature within contiguous sub-images.

Table 4 Layout-wise attribution of diferent operators. Values are equal-length editing accuracy (%).
<table><tr><td>Operator</td><td>Straight (IC13)</td><td>Curved (CUTE)</td><td>Multi-oriented (SVTP)</td></tr><tr><td>Shuffle only</td><td>79.00</td><td>54.83</td><td>66.67</td></tr><tr><td>Crop only</td><td>77.83</td><td>60.93</td><td>69.76</td></tr><tr><td>Shuffle&amp;Crop</td><td>80.16</td><td>58.42</td><td>68.84</td></tr></table>

DiT-Based Variant. We also try to instantiate TextRIRO with DiT-based conditional control, using Omini-Control [31] as a representative lightweight condition-injection method. This variant can be trained under our self-paired real-data formulation, but its ScenePair performance remains behind the ControlNet-based TextRIRO. The main reason is that OminiControl-style conditioning injects image conditions in a compact token-level manner, which is parameter-eficient and general, but relatively coarse for scene text. For text editing, small local deviations can change character identity or introduce artifacts. Therefore, ControlNet’s multi-scale feature injection is better suited because it preserves and transfers local style cues at several spatial resolutions. In addition, the OminiControl paper reports that, despite its minimal and universal design, it is not uniformly superior to ControlNet across all controlled generation settings. We therefore use ControlNet as the default backbone and leave more specialized DiT-based fine-grained control as future work.

Table 5 Backbone choice comparison on ScenePair [43].
<table><tr><td>Backbone</td><td>ACC (%)↑</td><td>NED↑</td><td>FID↓</td></tr><tr><td>OminiControl-based 2025</td><td>52.89</td><td>0.803</td><td>39.65</td></tr><tr><td>ControlNet-based 2023</td><td>85.23</td><td>0.964</td><td>41.53</td></tr></table>

Cross-recognizer evaluation. ScenePair ACC is measured with TRBA [1] as Oficial. To test whether the observed accuracy–realism trade-of is recognizer-specific, we additionally evaluate with SVTRv2 [7] and PPOCR-v6-rec [46]. Tab. 6 shows the same trend for all three recognizers: TextRIRO remains close to the real-image upper bound, whereas RS-STE consistently exceeds it while having a substantially worse FID (49.59 versus 41.53). This supports the interpretation that RS-STE produces recognizer-friendly but less realistic images.

Table 6 ScenePair word accuracy (%) with three independent recognizers.
<table><tr><td>Method</td><td>Official</td><td>SVTRv2</td><td>PPOCR-v6-rec</td></tr><tr><td>MOSTEL 2023</td><td>37.69</td><td>33.91</td><td>37.42</td></tr><tr><td>TextCtrl 2024</td><td>84.67</td><td>85.08</td><td>88.67</td></tr><tr><td>TextSSR 2025</td><td>79.69</td><td>80.23</td><td>82.34</td></tr><tr><td>RS-STE 2025</td><td>91.80</td><td>90.39</td><td>91.80</td></tr><tr><td>TextRIRO</td><td>85.23</td><td>85.86</td><td>86.25</td></tr><tr><td>Real (upper bound)</td><td>88.44</td><td>88.67</td><td>89.14</td></tr></table>

## 4.3 Synthetic Data Utility

Despite TextRIRO’s advances, a 10.10% accuracy gap remains compared to training with the same amount of real data. This gap mainly comes from a non-trivial number of edit failures that slip into the training set and accumulate errors. TextSSR [39] filters with an STR model to select usable samples, but the retained images largely duplicate styles already present in real data, which causes distribution redundancy and limits data expansion for STR. Because STE generates text that closely matches the original style, we concatenate each original image with its filtered edited result to build TextRIRO-3M, a long-text synthetic dataset.

Table 7 Data quality. Performance comparison of the STR model (MAERec [16]) with diferent types of synthetic data mixed with real data.
<table><tr><td rowspan="2">Training Data</td><td rowspan="2">Volume</td><td colspan="3">Common Benchmarks</td><td rowspan="2"></td><td colspan="2">Union14M-Benchmark</td></tr><tr><td>IIIT SVT IC13 IC15</td><td>SVTP CUTE</td><td>Avg CUR</td><td>MLO ART CTL SAL MLW</td></tr><tr><td>U14M-Filter 2025 (UF)</td><td>3.22M</td><td>99.30 97.84 98.60 90.50 96.12</td><td>98.26</td><td>96.77</td><td>|89.49 93.50 77.11</td><td>83.18 86.99 86.89</td><td>GEN Avg 85.4886.09</td></tr><tr><td>MJ 2014 + UF</td><td>1M + 3.22M</td><td>99.17 97.53 99.18 91.44 95.81</td><td>98.26</td><td>96.89</td><td>90.35 94.01</td><td>79.11 85.62 87.3 87.26</td><td>85.84 87.07</td></tr><tr><td>ST 2016 + UF</td><td>1M + 3.22M</td><td>99.33 97.53 98.95 90.94</td><td>96.74 98.26</td><td>96.96</td><td>90.44 94.08</td><td>78.00 86.14 87.81 87.01</td><td>85.77 87.04</td></tr><tr><td>TextSSR 2025 + UF</td><td>1M + 3.22M</td><td>99.07 97.53 98.83 91.28 95.50</td><td>98.61</td><td>96.80</td><td>91.06 94.52</td><td>81.22 86.26 87.56 88.35</td><td>85.95 87.85</td></tr><tr><td>MOSTEL 2023 + UF</td><td>1M + 3.22M</td><td></td><td></td><td></td><td>93.64</td><td></td><td></td></tr><tr><td>TextCtrl 2024 + UF</td><td>1M + 3.22M</td><td>|99.17 97.53 98.37 91.00 99.17 97.84 98.48 91.61</td><td>96.90 96.87 95.81 98.26</td><td>96.64 96.86</td><td>|90.89 90.35 94.01</td><td>79.33 83.83 87.43 87.50</td><td>85.74 86.91</td></tr><tr><td>RS-STE 2025 + UF</td><td>1M + 3.22M</td><td>99.10 97.06 98.6 91.39 96.74</td><td>99.65</td><td>97.09</td><td>90.89 94.16</td><td>79.11 84.34 87.30 87.26</td><td>85.84 86.89</td></tr><tr><td>TextRIRO + UF</td><td>1M + 3.22M</td><td>99.37 97.84 98.72 91.33 96.59</td><td>98.96</td><td>97.13</td><td>91.18 94.67</td><td>80.67 85.75 87.49 86.77 81.00 86.65 87.49 89.08</td><td>85.80 87.36 85.8087.98</td></tr><tr><td>Real-Concat + UF</td><td>1M + 3.22M</td><td>99.27 97.53 98.95 90.50 95.97</td><td>98.61</td><td>96.81</td><td>90.89 94.59</td><td>79.67 85.24 86.73 88.23</td><td>86.01 87.33</td></tr></table>

We mix each 1M synthetic dataset with the real dataset to compare how diferent synthetic sources improve recognition. The candidates include MJ [14] and ST [12], the difusion-based TextSSR [39], and datasets produced by the same concatenation strategy with MOSTEL [26], TextCtrl [43], and RS-STE [9], which we compare against TextRIRO. Tab. 7 shows that TextRIRO yields the best improvements on both the standard

Common Benchmarks and the more challenging Union14M-Benchmark. On the latter, the Multi-words subset improves by 2.19%, which aligns with the benefit of longer text introduced by concatenation. Although other editing-based generators also adopt concatenation, they remain bounded by the synthesis ceiling discussed above, and their performance still falls short of TextRIRO. Furthermore, we concatenate two independent real images to create longer sequences (Real-Concat). This yields some STR improvements, confirming that sequence-length augmentation helps. However, the performance is still lower than with TextRIRO-generated data. We attribute this gap to style inconsistency between concatenated samples.

Table 8 Extension of the experiments in Tab. 7, showing the performance of TextRIRO synthetic data on various STR models and data scales.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Training Data</td><td rowspan="2">Volume</td><td colspan="4">Common Benchmarks</td><td colspan="2">Union14M-Benchmark</td></tr><tr><td>IIIT SVT</td><td>IC13 IC15 SVTP</td><td> CUTE</td><td>Avg</td><td>CUR MLO ART CTL</td><td>SAL MLW GEN Avg</td></tr><tr><td rowspan="4">MAERec 2023</td><td>UF</td><td>3.22M</td><td>99.30 97.84 98.60 90.50</td><td>96.12</td><td>98.26</td><td>96.77 89.49 93.50</td><td>77.11 83.18 86.99</td><td>86.89 85.48 86.09</td></tr><tr><td>TextRIRO + UF</td><td>1M + 3.22M</td><td>99.37 797.84 98.72 91.33 +0.07</td><td>96.59</td><td>98.96</td><td>97.13 91.18 94.67</td><td>81.00 86.65 87.49</td><td>89.08 85.80 87.98</td></tr><tr><td></td><td></td><td>+0.00 +0.12 +0.83 99.2097.84 98.72 91.50</td><td>+0.47 95.66</td><td>+0.70 98.96</td><td>+0.36 +1.69 +1.17 96.98 91.92 94.38</td><td>+3.89 +3.47 +0.50 80.89 84.85 87.74</td><td>+2.19 +0.32 +1.89 86.07 88.00</td></tr><tr><td>TextRIRO + UF</td><td>3M + 3.22M</td><td>-0.10 +0.00 +0.12 +1.00</td><td>-0.46</td><td>+0.70</td><td>+0.21 +2.43</td><td>+0.88 +3.78 +1.67 +0.75</td><td>90.17 +3.28 +0.59 +1.91</td></tr><tr><td rowspan="4">SVTRv2-S 2025</td><td>UF</td><td>3.22M</td><td>99.00 97.22 98.37 90.39</td><td>93.80</td><td>98.96</td><td>96.29 |86.97</td><td>90.50 77.33 85.24 82.50</td><td>84.47 84.28 84.47</td></tr><tr><td>TextRIRO + UF</td><td>1M + 3.22M</td><td>99.0097.84 98.72 90.34 +0.00+0.62 +0.35 -0.05</td><td>94.11 +0.31</td><td>99.31 +0.35</td><td>96.55 88.09 +0.26 +1.12</td><td>91.67 77.67 85.24</td><td>84.31 86.29 83.72 85.28</td></tr><tr><td>TextRIRO + UF</td><td>3M + 3.22M</td><td>99.27 97.99 98.83 90.89</td><td>94.73</td><td>98.61</td><td>96.72 89.65</td><td>+1.17 +0.34 +0.00 92.26 78.22 85.4984.09</td><td>+1.81 +1.82 -0.56 +0.81 86.77 84.20 85.81</td></tr><tr><td></td><td></td><td>+0.27 +0.77 +0.46 +0.50</td><td>+0.93</td><td>-0.35</td><td>+0.43 +2.68</td><td>+1.76 +0.89 +0.25 +1.59</td><td>+2.30 -0.08 +1.34</td></tr></table>

Tab. 8 assesses generality across models and scalability with data size. With SVTRv2-S [7], TextRIRO still brings consistent gains, especially on Multi-Words samples. Scaling TextRIRO to 3M further improves Union14M performance for MAERec and raises Multi-Words to 90.17%, although gains on some common subsets saturate.

## 5 Conclusion

In this paper, we have introduced TextRIRO, a scene text editing method that, for the first time, learns editing purely from real data through a self-paired supervision scheme. By disrupting semantic content via targeted shufle and crop operations while maintaining style, we eliminated dependence on synthetic triplets and bypassed the need for explicit or implicit disentanglement modules. Our difusion-based model achieves near-realistic fidelity and robust content accuracy, efectively narrowing the synthetic-to-real gap that limits previous STE approaches. Comprehensive experiments on multiple benchmarks demonstrate that TextRIRO surpasses existing STE methods in both correctness and visual realism. Additionally, we create a large-scale, realistic long-text recognition dataset, TextRIRO-3M. Training STR models on it yielded significant gains, proving that high-realism STE continues to propel STR performance beyond current saturation points. Naturally, TextRIRO excels in English thanks to abundant real scene text data, but such multilingual data is scarce. Future work will focus on accurate and realistic multilingual STE in low-resource settings.

## Acknowledgments and Disclosure of Funding

This work was supported by National Natural Science Foundation of China under Grant 625B2057.

## References

[1] Jeonghun Baek, Geewook Kim, Junyeop Lee, Sungrae Park, Dongyoon Han, Sangdoo Yun, Seong Joon Oh, and Hwalsuk Lee. What is wrong with scene text recognition model comparisons? dataset and model analysis. In ICCV, pages 4714–4722, 2019.

[2] Haoxing Chen, Zhuoer Xu, Zhangxuan Gu, Jun Lan, Xing Zheng, Yaohui Li, Changhua Meng, Huĳia Zhu, and Weiqiang Wang. Difute: Universal text editing difusion model. In NeurIPS, volume 36, pages 63062–63074, 2023.

[3] Jingye Chen, Yupan Huang, Tengchao Lv, Lei Cui, Qifeng Chen, and Furu Wei. Textdifuser: Difusion models as text painters. In NeurIPS, volume 36, pages 9353–9387, 2023.

[4] Zeyu Chen, Fangmin Zhao, Yan Shu, Yichao Liu, Liu Yu, and Yu Zhou. Styletextgen: Style-conditioned multilingual scene text generation. In CVPR, pages 7643–7653, June 2026.

[5] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021.

[6] Yongkun Du, Zhineng Chen, Caiyan Jia, Xiaoting Yin, Tianlun Zheng, Chenxia Li, Yuning Du, and Yu-Gang Jiang. SVTR: Scene text recognition with a single visual model. In ĲCAI, pages 884–890, 2022.

[7] Yongkun Du, Zhineng Chen, Hongtao Xie, Caiyan Jia, and Yu-Gang Jiang. Svtrv2: Ctc beats encoder-decoder models in scene text recognition. In ICCV, pages 20147–20156, 2025.

[8] Yongkun Du, Zhineng Chen, Yazhen Xie, Weikang Bai, Hao Feng, Wei Shi, Yuchen Su, Can Huang, and Yu-Gang Jiang. Unirec-0.1 b: Unified text and formula recognition with 0.1 b parameters. In ECCV, pages 36–55, 2026.

[9] Zhengyao Fang, Pengyuan Lyu, Jingjing Wu, Chengquan Zhang, Jun Yu, Guangming Lu, and Wenjie Pei. Recognitionsynergistic scene text editing. In CVPR, pages 13104–13113, 2025.

[10] Tongkun Guan, Wei Shen, Xue Yang, Qi Feng, Zekun Jiang, and Xiaokang Yang. Self-supervised character-to-character distillation for text recognition. In ICCV, pages 19473–19484, 2023.

[11] Tongkun Guan, Wei Shen, and Xiaokang Yang. Ccdplus: Towards accurate character to character distillation for text recognition. IEEE Trans. Pattern Anal. Mach. Intell., 47(5):3546–3562, 2025. doi: 10.1109/TPAMI.2025.3533737.

[12] Ankush Gupta, Andrea Vedaldi, and Andrew Zisserman. Synthetic data for text localisation in natural images. In CVPR, pages 2315–2324, 2016.

[13] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In NeurIPS, volume 30, 2017.

[14] Max Jaderberg, Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Synthetic data and artificial neural networks for natural scene text recognition. CoRR, abs/1406.2227, 2014.

[15] Jiabao Ji, Guanhua Zhang, Zhaowen Wang, Bairu Hou, Zhifei Zhang, Brian Price, and Shiyu Chang. Improving difusion models for scene text editing with dual encoders. CoRR, abs/2304.05568, 2023.

[16] Qing Jiang, Jiapeng Wang, Dezhi Peng, Chongyu Liu, and Lianwen Jin. Revisiting scene text recognition: A data perspective. In ICCV, pages 20486–20497, 2023.

[17] Dimosthenis Karatzas, Faisal Shafait, Seiichi Uchida, Masakazu Iwamura, Lluis Gomez i Bigorda, Sergi Robles Mestre, Joan Mas, David Fernandez Mota, Jon Almazan Almazan, and Lluis Pere De Las Heras. ICDAR 2013 robust reading competition. In ICDAR, pages 1484–1493, 2013.

[18] Dimosthenis Karatzas, Lluis Gomez-Bigorda, Anguelos Nicolaou, Suman Ghosh, Andrew Bagdanov, Masakazu Iwamura, Jiri Matas, Lukas Neumann, Vĳay Ramaseshan Chandrasekhar, Shĳian Lu, et al. ICDAR 2015 competition on robust reading. In ICDAR, pages 1156–1160, 2015.

[19] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In CVPR, pages 8107–8116, 2020.

[20] Praveen Krishnan, Rama Kovvuri, Guan Pang, Boris Vassilev, and Tal Hassner. Textstylebrush: transfer of text aesthetics from a single example. TPAMI, 45(7):9122–9134, 2023.

[21] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas Müller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space. CoRR, abs/2506.15742, 2025.

[22] Junyeop Lee, Yoonsik Kim, Seonghyeon Kim, Moonbin Yim, Seung Shin, Gayoung Lee, and Sungrae Park. Rewritenet: Reliable scene text editing with implicit decomposition of text contents and styles. CoRR, abs/2107.11041, 2021.

[23] Laurens van der Maaten and Geofrey Hinton. Visualizing data using t-sne. JMLR, 9(Nov):2579–2605, 2008.

[24] Anand Mishra, Karteek Alahari, and CV Jawahar. Scene text recognition using higher order language priors. In BMVC, pages 1–11, 2012.

[25] Trung Quy Phan, Palaiahnakote Shivakumara, Shangxuan Tian, and Chew Lim Tan. Recognizing text with perspective distortion in natural scenes. In ICCV, pages 569–576, 2013.

[26] Yadong Qu, Qingfeng Tan, Hongtao Xie, Jianjun Xu, Yuxin Wang, and Yongdong Zhang. Exploring stroke-level modifications for scene text editing. In AAAI, volume 37, pages 2119–2127, 2023.

[27] Anhar Risnumawan, Palaiahankote Shivakumara, Chee Seng Chan, and Chew Lim Tan. A robust arbitrary text detection system for natural scene images. ESWA, 41(18):8027–8048, 2014.

[28] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent difusion models. In CVPR, pages 10684–10695, 2022.

[29] Prasun Roy, Saumik Bhattacharya, Subhankar Ghosh, and Umapada Pal. Stefann: scene text editor using font adaptive neural network. In CVPR, pages 13228–13237, 2020.

[30] Fenfen Sheng, Zhineng Chen, and Bo Xu. NRTR: A no-recurrence sequence-to-sequence model for scene text recognition. In ICDAR, pages 781–786, 2019.

[31] Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol: Minimal and universal control for difusion transformer. In ICCV, pages 14940–14950, 2025.

[32] Yuxiang Tuo, Yifeng Geng, and Liefeng Bo. Anytext2: Visual text generation and editing with customizable attributes. CoRR, abs/2411.15245, 2024.

[33] Yuxiang Tuo, Wangmeng Xiang, Jun-Yan He, Yifeng Geng, and Xuansong Xie. Anytext: Multilingual visual text generation and editing. In ICLR, 2024.

[34] Kai Wang, Boris Babenko, and Serge Belongie. End-to-end scene text recognition. In ICCV, pages 1457–1464, 2011.

[35] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. CoRR, abs/2508.02324, 2025.

[36] Liang Wu, Chengquan Zhang, Jiaming Liu, Junyu Han, Jingtuo Liu, Errui Ding, and Xiang Bai. Editing text in the wild. In ACM MM, pages 1500–1508, 2019.

[37] Yu Xie, Jielei Zhang, Pengyu Chen, Weihang Wang, Longwen Gao, Peiyi Li, Qian Qiao, and Zhouhui Lian. Textflux: An ocr-free dit model for high-fidelity multilingual scene text synthesis. Computer Graphics Forum, page e70342, 2026.

[38] Qiangpeng Yang, Jun Huang, and Wei Lin. Swaptext: Image based texts transfer in scenes. In CVPR, pages 14700–14709, 2020.

[39] Xingsong Ye, Yongkun Du, Yunbo Tao, and Zhineng Chen. Textssr: Difusion-based data synthesis for scene text recognition. In ICCV, pages 17464–17473, 2025.

[40] Xingsong Ye, Yongkun Du, JiaXin Zhang, Chen Li, Jing LYU, and Zhineng Chen. What’s wrong with synthetic data for scene text recognition? a strong synthetic engine with diverse simulations and self-evolution. In CVPR, pages 16645–16654, 2026.

[41] Xingsong Ye, Yongkun Du, Jiaxin Zhang, Zhixian Li, Chong Sun, Chen Li, Jing Lyu, Lianwen Jin, and Zhineng Chen. All-in-one multilingual scene text recognition with script-aware mixture-of-experts. CoRR, abs/2609.24058, 2026.

[42] Xingsong Ye, Yongkun Du, Jiaxin Zhang, Haojie Zhang, Chong Sun, Chen Li, Jing Lyu, and Zhineng Chen. Advancing wordart-oriented scene text recognition: Datasets and methods. In ECCV, pages 77–94, 2026.

[43] Weichao Zeng, Yan Shu, Zhenhang Li, Dongbao Yang, and Yu Zhou. Textctrl: Difusion-based scene text editing with prior guidance control. In NeurIPS, volume 37, pages 138569–138594, 2024.

[44] Boqiang Zhang, Hongtao Xie, Zuan Gao, and Yuxin Wang. Choose what you need: Disentangled representation learning for scene text recognition removal and editing. In CVPR, pages 28358–28368, 2024.

[45] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image difusion models. In ICCV, pages 3836–3847, 2023.

[46] Yubo Zhang, Xueqing Wang, Manhui Lin, Yue Zhang, Penglongyi Deng, Ting Sun, Tingquan Gao, Zelun Zhang, Jiaxuan Liu, Changda Zhou, et al. Pp-ocrv6: From 1.5 m to 34.5 m parameters, surpassing billion-scale vlms on ocr tasks. CoRR, abs/2606.13108, 2026.

[47] Yiming Zhao and Zhouhui Lian. UDifText: A unified framework for high-quality text synthesis in arbitrary images via character-aware difusion models. In ECCV, pages 217–233, 2024.

[48] Tianlun Zheng, Zhineng Chen, Shancheng Fang, Hongtao Xie, and Yu-Gang Jiang. Cdistnet: Perceiving multi-domain character distance for robust text recognition. Int. J. Comput. Vis., 132(2):300–318, 2024.

## Appendix

![](images/da243a6e3049d5953ebd11e1ad23d8df4e31f95d4e4a7e52e8080e2d1f7d5244.jpg)  
Figure 6 An interesting comic for introducing our idea. Top: Two previous STE training paradigms, both requiring paired synthetic data. Bottom: Our method leverages self-paired real data without any synthetic pairs.

## 6 More Implementation Details

## 6.1 Model Formulation Details

The details of Sec. 3.2 are as follows. Given $I _ { \mathrm { s r c } } ,$ the style encoder first extracts ViT features

$$
\begin{array} { r } { F _ { \mathrm { v i t } } = \mathrm { V i T } I _ { \mathrm { s r c } } \in \mathbb { R } ^ { N \times D } , } \end{array}\tag{2}
$$

where N is the number of patches and D is the feature dimension. For each pyramid level $k ,$ adaptive average pooling and a zero convolution [45] produce

$$
F _ { \mathrm { p y r a m i d } } ^ { k } = \mathrm { C o n v } _ { 1 \times 1 } \mathrm { A v g P o o l } _ { h _ { k } , w _ { k } } F _ { \mathrm { v i t } } ,\tag{3}
$$

and all levels are aggregated as

$$
F _ { \mathrm { s t y l e } } = \{ F _ { \mathrm { p y r a m i d } } ^ { k } \mid k = 1 , . . . , K \} .\tag{4}
$$

The ViT takes RGB images of size $6 4 \times 2 5 6 ,$ , uses $1 6 \times 1 6$ patches, and outputs 768-dimensional patch embeddings. It contains 12 layers with 12 attention heads. We set $K = 4 _ { s }$ , with pyramid output sizes 8, 32, 4, 16, 2, 8, and 1, 4.

For the text encoder, each target text $T _ { \mathrm { t g t } }$ with length $l = \lvert T _ { \mathrm { t g t } } \rvert$ is truncated or padded to the maximal character length $L _ { M }$ . Each character t<sub>i</sub> is mapped to a vocabulary index $x _ { i } ,$ yielding

$$
X _ { \mathrm { { t g t } } } = \left. { \bf { s } } \right. \ x _ { 1 } \dots x _ { l } \ \left. { \bf { s } } \right. \ \left. { \mathrm { p a d } } \right. _ { l 3 } ^ { L _ { M } 2 } ,\tag{5}
$$

where $\langle \mathrm { s } \rangle , \langle \mathrm { s } \rangle$ , and ⟨pad⟩ are special tokens. The encoded feature is

$$
F _ { \mathrm { t e x t } } = \mathrm { E m b e d } X _ { \mathrm { t g t } } ~ \mathrm { P o s E n c } X _ { \mathrm { t g t } } \in \mathbb { R } ^ { L _ { M } 2 \times D } .\tag{6}
$$

We set $L _ { M } = 2 5$ and use 94 printable characters plus the space character. The embedding dimension is 768,

and the text feature is fed to the conditional U-Net through cross-attention. The conditional difusion loss is

$$
\mathcal { L } _ { \mathrm { T e x t R I R O } } = | | \epsilon - \epsilon z _ { t } , t , F _ { \mathrm { s t y l e } } , F _ { \mathrm { t e x t } } | | ^ { 2 } ,\tag{7}
$$

where $z _ { t }$ is the noisy latent at timestep t, ϵ is the injected noise, and ϵ is the predicted noise conditioned on style and text features.

## 6.2 Model specific settings

In the setup of Sec. 3.1 and Alg. 1, we set $p _ { \mathrm { s h u f f l e } }$ to 0.7 by default. To investigate the impact of this parameter, we conduct a rough ablation study with values 0.5, 0.7, and 1.0. The results are summarized in Tab. 9. We observe that setting $p _ { \mathrm { s h u f f l e } }$ to 0.7 yields the best overall results in terms of accuracy and NED, while maintaining competitive FID. Lower values result in reduced accuracy, whereas higher values slightly degrade FID without substantial gains in other metrics. This suggests that $p _ { \mathrm { s h u f f l e } } = 0 . 7$ achieves a balanced trade-of among all evaluated criteria, and we thus adopt it as our default choice for subsequent experiments.

Table 9 Ablation study of ${ p _ { \mathrm { s h u f f l e } } }$ on ScenePair [43].
<table><tr><td>Pshuffle</td><td>ACC (%)↑</td><td>NED↑</td><td>FID↓</td></tr><tr><td>0.5</td><td>80.78</td><td>0.943</td><td>42.76</td></tr><tr><td>0.7</td><td>82.42</td><td>0.960</td><td>43.59</td></tr><tr><td>1.0</td><td>81.72</td><td>0.947</td><td>44.01</td></tr></table>

## 6.3 Datasets and Implementation Details

Training Datasets. Our self-paired pipeline uses the text label and, in particular, its length, but it does not require character segmentation. For Shufle, we split the image width into $n _ { \mathrm { s p l i t } } \in 2 , L$ equal-width columns, where L is the label length, and randomly permute the columns. Crop takes a horizontal contiguous sub-interval. Thus, a character may be cut by a column boundary. This is intentional because the purpose is to remove readable semantics while retaining local visual style. We train our STE model and the STR baseline on Union14M [16], currently the largest available scene text dataset. Specifically, we adopt a filtered version of its labeled subset, denoted as U14M-Filter [7], which contains 3.22M annotated text instances. Note that training in a completely label-free manner is theoretically infeasible, because the definition of STE inherently requires specifying the edited text.

Evaluation Datasets. To evaluate scene text editing performance under real-world conditions, we first employ the ScenePair [43], which comprises 1,280 pairs of carefully collected, real-world text images. Furthermore, we construct benchmarks for equal-length and variable-length text editing using six widely-used text recognition datasets: IIIT 5K-Words (IIIT [24]), ICDAR2013 (IC13 [17]), Street View Text (SVT [34]), ICDAR2015 (IC15 [18]), Street View Text-Perspective (SVTP [25]), and CUTE80 (CUTE [27]), which we collectively refer to as the Common Benchmarks. Additionally, we assess data quality using Union14M-Benchmark [16], which includes seven subsets: Curve (CUR), Multi-Oriented (MLO), Artistic (ART), Contextless (CTL), Salient (SAL), Multi-Words (MLW), and General (GEN).

Implementation Details. We use MAERec [16] both for text-encoder pre-training and as our STR proxy baseline, training for 20 epochs under the settings established by the OpenOCR framework [7]. TextRIRO builds upon SD 2.1 for the VAE and SD 1.5 for the U-Net backbone [28]. The extensively pre-trained VAE in SD 2.1 is well-suited for scene text tasks. The U-Net requires its parameters to be unfrozen for training, so the specific version has minimal impact. Meanwhile, SD 1.5 uses a smaller dimension (dim = 768) and can directly match the feature dimension of the text encoder. All images are resized to 64 × 256 and those with aspect ratios less than 1 : 2 are rotated 90<sup>◦</sup> counterclockwise to match the target resolution. We use a batch size of 256, apply an initial learning rate of $1 \times 1 0 ^ { - 5 }$ , and train for 15 epochs on 8 NVIDIA V100 GPUs. During inference, we use 20 sampling steps and a classifier-free guidance (CFG) scale of $\omega = 2 . 2$ . For the generation of the TextRIRO-3M dataset, we use only reference images whose original text contains at least 4 characters. Consequently, the minimum label length in TextRIRO-3M is 8. For the other STE models, we use their oficially reported results when available. Otherwise, we use their released oficial checkpoints and the default inference hyperparameters to ensure a fair comparison.

Training–inference protocol. During training, the style condition is the perturbed view of the same real image and the target text is its original label. At inference, the source is an unperturbed real image and the target is a new string. This is an unavoidable residual mismatch because coherent real source–target pairs with diferent text are not available at scale. Shufle&Crop is designed to reduce the mismatch by producing a real-image-derived style condition whose semantics are destroyed but whose local appearance remains realistic. The self-paired objective also provides a content-disentanglement constraint: the source condition has disrupted text order, while the target text is supplied only through the text encoder. If semantic content leaked through the style path, the model would copy the disrupted content rather than render the correctly ordered target. Consistently, the identity (None) ablation in Fig. 4 collapses to an ACC of 4.37, showing that an unperturbed source encourages copying instead of text-conditioned editing.

Computational Overhead and Runtime Eficiency. We report the computational requirements for both the STR pre-training and U-Net fine-tuning stages, which use 36 GPU-hours and 180 GPU-hours, respectively. For inference, we evaluate our method on a single NVIDIA V100 GPU with diffusion\_steps set to 20 and a batch size of 32. Processing 10,000 batches requires 18 hours, corresponding to an average inference time of 0.20 seconds per image.

Table 10 Performance of diferent VAE variants on the quality of reconstructed scene text images.
<table><tr><td>Metrics</td><td></td><td colspan="7">VAE Version|IIIT 2012 SVT 2011 IC13 2013 IC15 2015 SVTP 2013 CUTE 2014</td></tr><tr><td rowspan="3">PSNR↑</td><td>SD 1.5</td><td>25.90</td><td>30.79</td><td>28.41</td><td>29.87</td><td>28.58</td><td>26.43</td><td>28.33</td></tr><tr><td>SD 2.1</td><td>27.63</td><td>33.02</td><td>30.23</td><td>31.99</td><td>30.68</td><td>28.30</td><td>30.31</td></tr><tr><td>SD 3.5</td><td>29.26</td><td>32.34</td><td>30.32</td><td>32.18</td><td>31.64</td><td>28.87</td><td>30.77</td></tr><tr><td rowspan="3">FID↓</td><td>SD 1.5</td><td>1.36</td><td>0.50</td><td>0.50</td><td>1.22</td><td>1.83</td><td>0.50</td><td>0.98</td></tr><tr><td>SD 2.1</td><td>0.75</td><td>0.21</td><td>0.24</td><td>0.71</td><td>1.03</td><td>0.16</td><td>0.52</td></tr><tr><td>SD 3.5</td><td>1.28</td><td>0.39</td><td>0.41</td><td>1.14</td><td>1.72</td><td>0.34</td><td>0.88</td></tr><tr><td rowspan="4">Acc (%)↑</td><td>SD 1.5</td><td>98.70</td><td>97.37</td><td>97.67</td><td>89.62</td><td>92.56</td><td>97.57</td><td>95.58</td></tr><tr><td>SD 2.1</td><td>98.97</td><td>97.99</td><td>98.02</td><td>89.95</td><td>93.49</td><td>98.26</td><td>96.11</td></tr><tr><td>SD 3.5</td><td>99.03</td><td>98.30</td><td>98.25</td><td>90.56</td><td>94.88</td><td>97.92</td><td>96.49</td></tr><tr><td>Real</td><td>99.30</td><td>97.84</td><td>98.60</td><td>90.50</td><td>96.12</td><td>98.26</td><td>96.77</td></tr></table>

VAE Ablation Study. We conduct a comprehensive ablation study by evaluating diferent variants of the VAE on six common STR benchmarks. Specifically, we encode and decode images using each VAE model, and then measure reconstruction quality with PSNR and FID. Furthermore, we assess the impact of VAE reconstruction on downstream text recognition by reporting Acc using a fixed recognizer. For reference, we also report the accuracy using ground-truth (unprocessed) images.

CFG Scale Ablation Study. We conduct an ablation study to investigate the efect of varying the CFG scale ω during inference. As shown in Tab. 11, increasing ω from 1.0 to 1.5 improves ACC and NED while substantially reducing FID from 60.34 to 35.37. Beyond $\omega = 1 . 5 ,$ further increases generally improve text correctness but worsen FID, revealing a trade-of between accuracy and image fidelity. ACC reaches its highest value at $\omega = 5 . 0$ and decreases slightly at $\omega = 7 . 5$ , whereas NED continues to improve. $\operatorname { A t } \omega = 2 . 2 ,$ , the model achieves a balanced performance, with ACC of 85.23%, NED of 0.964, and FID of 41.53. We therefore select $\omega = 2 . 2$ for subsequent experiments.

Table 11 Ablation study of CFG scale ω on ScenePair [43].
<table><tr><td> $\omega$ </td><td>ACC (%)↑</td><td>NED↑</td><td>FID↓</td></tr><tr><td>1.0</td><td>59.45</td><td>0.817</td><td>60.34</td></tr><tr><td>1.5</td><td>78.98</td><td>0.934</td><td>35.37</td></tr><tr><td>1.8</td><td>82.27</td><td>0.955</td><td>37.54</td></tr><tr><td>2.0</td><td>83.83</td><td>0.961</td><td>39.69</td></tr><tr><td>2.2</td><td>85.23</td><td>0.964</td><td>41.53</td></tr><tr><td>2.5</td><td>86.41</td><td>0.968</td><td>43.82</td></tr><tr><td>5.0</td><td>88.05</td><td>0.975</td><td>50.24</td></tr><tr><td>7.5</td><td>87.73</td><td>0.976</td><td>50.72</td></tr></table>

## 6.4 Additional Editing Evaluation

Per-image style fidelity beyond FID. FID is established as the evaluation metric in MOSTEL [26], and TextCtrl [43] continues to use the same evaluation code to report FID on ScenePair, so we keep it for comparability. For each generated image, we compare it against its corresponding ScenePair reference image and report LPIPS (perceptual similarity), SSIM (structural similarity), and PSNR (pixel-level fidelity). Lower LPIPS and higher SSIM and PSNR indicate better agreement with the paired reference. LPIPS measures per-image perceptual and structural similarity, and FID measures how close the overall image distribution is to the reference distribution. TextRIRO achieves the best results on both in Tab. 12, which indicates that it generates images closer to the real distribution. PSNR and SSIM, in contrast, are per-image pixel-level metrics that are sensitive to small spatial shifts, anti-aliasing, and valid appearance variations in rendered text. Because the ScenePair reference images are manually collected and not strictly pixel-aligned pairs, even a more realistic generation can score lower on these metrics simply because its pixels do not exactly match the reference.

Table 12 Per-image fidelity against the ScenePair reference and distributional fidelity. Lower LPIPS/FID and higher SSIM/PSNR are better.
<table><tr><td>Method</td><td>LPIPS↓</td><td>SSIM↑</td><td>PSNR↑</td><td>FID↓</td></tr><tr><td>MOSTEL 2023</td><td>0.2947</td><td>0.3311</td><td>14.05</td><td>49.19</td></tr><tr><td>TextCtrl 2024</td><td>0.3219</td><td>0.2610</td><td>11.68</td><td>43.78</td></tr><tr><td>TextSSR 2025</td><td>0.2622</td><td>0.3889</td><td>14.13</td><td>45.37</td></tr><tr><td>RS-STE 2025</td><td>0.2458</td><td>0.4347</td><td>15.66</td><td>49.59</td></tr><tr><td>TextRIRO</td><td>0.2394</td><td>0.3944</td><td>14.44</td><td>41.53</td></tr></table>

We acknowledge RS-STE’s superiority on these two metrics. Methodologically, RS-STE builds $L _ { 1 }$ and LPIPS losses directly in pixel space, whereas our difusion-based approach builds the loss in VAE latent space, which is inherently disadvantaged at pixel-level reconstruction. Even so, TextRIRO achieves the second-best results on SSIM and PSNR, demonstrating the efectiveness of our real-data training paradigm.

## 6.5 Additional TextRIRO Details

Length-agnostic concatenation. The even-length regularity originates from the symmetric concatenation recipe rather than an intrinsic model limitation. We conduct an additional controlled study on long-text samples (target length at least 8) to quantify this efect. We evaluate three models on the long-text set from Multi-Words (362 odd-length and 401 even-length samples, 763 in total). The three models difer only in training data. The first is trained on real STR data (Baseline). The second adds 1M symmetric data. The third adds 0.5M symmetric and 0.5M asymmetric data (the asymmetric-augmented recipe).

Two observations follow from Tab. 13. (1) With purely symmetric (even-only) concatenation, the model still generalizes to odd lengths (+1.93), which confirms that the even-length regularity is not a hard model limitation. However, a residual bias remains, as even-length gains (+3.49) exceed odd-length gains (+1.93). (2) To eliminate this bias, we introduce an asymmetric concatenation recipe. We concatenate a text with a variant of length L − 1, which supplies odd-length long-text samples while barely afecting edit-level supervision. Mixing 0.5M asymmetric with 0.5M symmetric samples (1M total) balances the two gains (+2.48 odd vs. +2.25 even), which removes the even-length artifact while keeping overall accuracy on par with the symmetric-only model. The length regularity is a controllable property of the data recipe.

Table 13 Odd- vs. even-length word accuracy (%) under diferent data recipes.
<table><tr><td>Training data</td><td>Odd length</td><td>Even length</td><td>Overall</td></tr><tr><td>Real STR data only</td><td>86.19</td><td>86.78</td><td>86.50</td></tr><tr><td>+ 1M symmetric (even-only)</td><td>88.12</td><td>90.27</td><td>89.25</td></tr><tr><td>+ 1M asymmetric-augmented</td><td>88.67</td><td>89.03</td><td>88.86</td></tr></table>

Complementarity with synthetic sources. We also test whether TextRIRO data complements, rather than replaces, existing synthetic sources. Tab. 14 shows that adding either TextSSR or ST to TextRIRO and U14M-Filter improves the Union14M-Benchmark average, and using all sources gives the best result. TextSSR is particularly complementary for contextless text, while rendering-based ST helps curved and multi-oriented cases. Thus, our real-only construction is a choice to prioritize realistic editing, not a claim that prior synthetic data is intrinsically unusable.

Table 14 Complementarity with synthetic sources. Recognition accuracy (%) of MAERec trained with diferent data mixtures.
<table><tr><td rowspan="2">Training Data</td><td rowspan="2">Volume</td><td rowspan="2">Common Benchmarks</td><td rowspan="2">Union14M-Benchmark CTL SAL MLW GEN Avg</td></tr><tr><td>IIIT SVT IC13 IC15 SVTP CUTE Avg CUR MLO ART</td></tr><tr><td>UF (real only)</td><td>3.22M</td><td>99.3097.84 98.60 90.50 96.12 98.26 96.77 89.49</td><td>93.50 77.11 83.18 86.99 86.89 85.48 86.09</td></tr><tr><td>TextSSR + UF</td><td>1M + 3.22M</td><td>99.07 797.53 98.83 91.28 95.50 98.61 96.80</td><td>91.06 94.52 81.22 86.26 87.56 88.35 85.95 87.85</td></tr><tr><td>TextRIRO + UF</td><td>1M + 3.22M</td><td>99.37 97.84 98.72 91.33 96.59 98.96 97.13 91.18</td><td>94.67 81.00 86.65 87.49 89.08 85.80 87.98</td></tr><tr><td>TextRIRO + TextSSR + UF</td><td>1M + 1M + 3.22M</td><td>99.30 97.99 98.60 91.72 96.28 98.96 97.14 |91.59</td><td>94.89 81.44 87.93 88.19 88.83 86.22 88.44</td></tr><tr><td>TextRIRO + ST + UF</td><td>1M + 1M + 3.22M</td><td>99.37 97.84 98.60 91.61 97.36 99.65 97.40 92.58</td><td>95.62 81.33 86.26 87.93 89.81 86.22 88.54</td></tr><tr><td>TextRIRO + TextSSR + ST + UF</td><td>1M + 1M + 1M + 3.22M</td><td>99.40 98.3098.83 91.44 97.21 99.31 97.41 92.25</td><td>95.47 82.00 87.14 88.20 89.44 86.32 88.69</td></tr></table>

Adding paired synthetic data to the editor. Our current training uses only real self-paired data. This framework can equally incorporate existing paired synthetic data. Under the ablation configuration, we add the synthetic data provided by MOSTEL [26] for training.

Adding synthetic data raises text ACC and NED, which shows that synthetic supplementation brings a correctness gain. However, FID increases, which confirms that synthetic data has a real distribution gap and hurts realism when added. To preserve realism (the “Real Out” requirement), our paper adopts a real-only training strategy.

Table 15 Adding paired synthetic data to the self-paired editor under the ablation configuration.
<table><tr><td>Setting</td><td>ACC (%)↑</td><td>NED↑</td><td>FID↓</td></tr><tr><td>Real only (self-paired)</td><td>82.42</td><td>0.960</td><td>43.59</td></tr><tr><td>+ Synthetic data</td><td>85.08</td><td>0.965</td><td>49.34</td></tr></table>

Quality Filtering Bias. To further validate that the STR model for filtering preserves high-accuracy results and does not compromise dataset quality, we randomly sample 300 instances from the outputs generated by TextRIRO and filtered by SVTRv2 [7]. We then recruit three assessors, each responsible for evaluating 100 instances. The average accuracy reaches 99.33% (with individual accuracies of 99%, 99%, and 100%), demonstrating that the retained samples are highly reliable. The filter can nevertheless reject some visually correct samples when the recognizer fails on an unusual style. We choose this conservative trade-of to prioritize label correctness, and report the human check to make this selection bias explicit.

![](images/7ed6b0546b8d1249fc2dcc3fdaae26ed3b612eafdb6ede0ba80ce214daff6640.jpg)  
Figure 7 t-SNE [23] visualization of text features by STR pre-trained text encoder.

## 7 Visualization

## 7.1 Character-Aware Embeddings

The STR pre-trained text encoder endows TextRIRO with strong character awareness. To further evaluate its encoding capabilities, we visualize the embeddings of several text samples using t-SNE [23], as illustrated in Fig. 7. Overall, words with similar character compositions or comparable text lengths are positioned closer together, while suficient separation is maintained between distinct samples. For example, the characters ‘1’ and ‘l’ possess diferent semantic meanings but exhibit similar glyphs, resulting in ‘1024’ and ‘l024’ being embedded in close proximity. Traditionally, glyph-pre-training is required for such nuanced feature representations, yet STR pre-training achieves this directly. Additionally, the variations ‘there’, ‘There’, ‘THere’, and ‘THERE’ are embedded near each other, reflecting their shared underlying structure, yet each maintains a distinct position. This separation prevents case confusion in text generation, thereby enabling accurate character-level generation.

## 7.2 TextRIRO-3M Data View

Fig. 8 presents several examples from TextRIRO-3M, demonstrating that the dataset doubles the original text length while preserving the original style. Fig. 9 further compares the label length distribution between the original symmetric TextRIRO-3M recipe and real-world data. The symmetric construction significantly expands coverage of long text but produces even lengths only. As demonstrated in Tab. 13, asymmetric augmentation removes this artifact when broader length coverage is required.

![](images/1dc75c49ac5b3d9bb1a9ff7e50ede81061b21b1a9cdba23da33d0b3ff884ab5e.jpg)  
Figure 8 Visualization results randomly sampled from the TextRIRO-3M dataset, which doubles the text length compared to the original scene text images.

## 7.3 Enhanced STR Visualization

To clearly illustrate the efectiveness of TextRIRO-3M in improving the performance of STR models under challenging scenarios, especially for long text recognition, we visualize the prediction results of MAERec [16] trained with diferent datasets in Fig. 10. The experimental results show that, after incorporating TextRIRO-3M during training, the model produces more accurate and robust predictions. In particular, it demonstrates improved recognition of long and complex text sequences, efectively reducing error rates in dificult cases.

## 7.4 Bad Cases and Discussion

It is important to acknowledge that our STE approach is not without limitations. Fig. 11 illustrates three common failure cases, along with the corresponding causes and potential solutions:

1. Long Text: The text encoder used in our training pipeline is pre-trained from the STR model, which is primarily trained on datasets containing short text, resulting in limited capacity for capturing long text dependencies. Additionally, TextRIRO is trained on the same datasets, further constraining its ability to handle long text. To address this, we are inspired by the construction of the TextRIRO-3M dataset: we segment long text into shorter fragments, synthesize each fragment individually, and then concatenate these fragments to form the final long text. This approach leverages the inherent strengths of short text synthesis while enabling robust long text generation.

2. Curved Text and Complex Geometry: As shown in Fig. 11, TextRIRO recognizes that the input is not standard horizontally aligned text, but may interpret highly curved text as slanted. Crop preserves local tilt and curvature, and near-vertical inputs are rotated in preprocessing. Consequently, TextRIRO achieves equal-/variable-length accuracy of 64.87/48.12 on CUTE and 79.69/42.31 on SVTP, outperforming the compared baselines. Tab. 4 further isolates the operator contribution. However, extreme curvature and large geometric deformation remain dificult because regular layouts dominate the training data. Future work should incorporate finer-grained control points, similar to TPS modules in STR, to enable more precise geometric control.

3. Special Symbols and Languages: The synthesis of special symbols remains challenging because of both sparse training coverage and the current vocabulary/data coverage. This also restricts cross-lingual generalization: the self-pairing operation itself is language-agnostic, but suficiently large real labeled datasets are currently much more available for Latin scripts than for low-resource languages. Collecting balanced real labeled data with broader character and language coverage is necessary for stable multilingual generation. TextRIRO will also provide more abundant synthetic data for multilingual and general OCR [8, 41].

![](images/b80aad6f5eb5c733a50d5d40ce88f74ecb820d61e6d8b7676b5c6bb69365a6ed.jpg)

Figure 9 Length distributions of real data (U14M-Filter) and the original symmetric TextRIRO-3M construction. The symmetric recipe doubles the source length and hence produces even-length samples, and the asymmetric-augmented recipe in Tab. 13 removes this construction artifact.  
![](images/9c370e20a1c52f304635e162274038a0cf674903494d9f5185ab5d7d3fb0e2c7.jpg)  
Figure 10 Comparison of STR model (MAERec [16]) prediction outcomes trained on diferent datasets.

Style preservation and disentanglement. Shufle preserves stroke thickness because it permutes equalwidth columns without rescaling them. Crop can change apparent stroke scale, but the two operators are complementary: pure-Shufle instances remain in training with non-trivial probability (approximately 20% for an average label length of five) and anchor stroke-width preservation, while Crop supplies the scale variation needed for variable-length editing. The style encoder is not given an explicit transformation-type label, and it learns from the mixed perturbation distribution. Together with the disrupted semantics in the source and the separate text condition, this prevents simple content copying, as further supported by the None ablation described above. These mechanisms do not eliminate every failure case, but explain why the combined operator is more robust than either component alone.

## 8 Responsible Research Notes

Broader impacts and safeguards. TextRIRO can support benign applications such as visual content localization, accessibility, and realistic STR data generation, but scene text editing can also be misused to falsify text in images. We therefore intend to release code, data-construction scripts, and TextRIRO-3M with clear research-use and acceptable-use notices, retain STR-based quality filtering, and encourage provenance tracking or watermarking when the technology is deployed.

Existing assets and licenses. The work uses publicly available academic datasets, model components, and baselines, all cited in the main paper and bibliography. Users of released artifacts will be instructed to follow the original licenses and terms for Union14M, U14M-Filter, ScenePair, the STR benchmarks, Stable Difusion components, and compared open-source STE models.

New assets. TextRIRO-3M is a newly constructed derived dataset. The paper documents its source data,

![](images/6d830b37ed764657b2d2710c583ffdc30a61b8f346764aa7fcb29706c79d8950.jpg)  
Figure 11 Bad Cases. We show several disappointing editing results produced by TextRIRO.

generation pipeline, STR filtering, scale, and length distribution in Sec. 3.3, Sec. 4.2, Tab. 7, Tab. 8, and Fig. 9.

Human assessment protocol. The quality-filtering bias check in the Appendix used three assessors to judge generated text-image pairs. The instruction was: “Given a generated image and its target label, mark the sample correct only if all visible characters match the target label in order.” No personal data were collected, no crowdsourcing platform was used, no external paid workers were recruited, and the task involved only visual inspection of scene-text images.