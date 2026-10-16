# Seeing is Believing: Benchmarking and Mitigating the Interference of Language Priors on Visual Perception in MLLMs

Li Ma , Yan Zhuang , Jiawen Deng\* , Fuji Ren School of Computer Science and Engineering   
University of Electronic Science and Technology of China   
{202421080935, 202211081370}@std.uestc.edu.cn {dengjw, renfuji}@uestc.edu.cn

## Abstract

Language priors in MLLMs can override visual evidence, impairing visual perception. Existing benchmarks assume shared priors across models and rely solely on accuracy. However, cross-model prior agreement falls below 80%, and over 15% of items can be answered correctly without any image, causing accuracy to overestimate prior resistance. To address these limitations, we introduce ViSiP-Bench, a benchmark grounded in rare-but-real scenarios across six prior categories, with a modelaware dynamic prior induction protocol and three metrics that more precisely and continuously measure prior resistance than accuracy alone. Evaluating 24 models reveals that prior adherence is pervasive, does not improve with scale, and larger models tend to either strongly override their priors or firmly adhere to them. We therefore propose MAPS, which identifies each model’s strongest priors and fine-tunes on factually grounded knowledge to soften excessive prior reliance without disrupting general capabilities.

## 1 Introduction

Language priors are an inherent byproduct of MLLM pretraining, encoding world knowledge that underpins model reasoning(Han et al., 2026). Yet when conflicting with visual evidence, language priors can override perceptual processing, causing models to report what they believe rather than what they observe. As illustrated in Figure 1, when explicitly prompted to report the answer written by the boy in the image, most mainstream MLLMs default to their arithmetic prior and respond with 114, despite the image clearly showing 110. Language priors pose a direct threat to safetycritical applications such as medical imaging and industrial defect inspection, where accurate visual perception is extremely essential.

![](images/7f26b54ab9a878196019c02cdd19b69efd018875d3a63c8637b5eae685fed380.jpg)  
Figure 1: A representative failure case of language prior dependency in MLLMs.

![](images/24aab3ffca024683dbad71a40c84011be657656428bd45ac7e4c1e0a322f6c9b.jpg)  
Figure 2: Distribution of three sources of no-image correct responses across existing benchmarks.

Existing benchmarking works(Guan et al., 2024; Lee et al., 2025; Luo et al., 2024) typically assume that all models share the same commonsense priors, and accordingly construct anomalous datasets where visual signals conflict with those priors, reporting accuracy as the sole metric to quantify prior resistance. We challenge this paradigm on two fronts. First, our cross-model analysis reveals that prior agreement falls below 80%, directly undermining the shared-prior assumption. Second, as shown in Figure 2, over 15% of items across existing benchmarks can be answered correctly without any image, arising from three sources: (1) Chance

![](images/674d4139059f60c5e1f8a5b223b523c459cc2df4c04e2862a88465e2202dc5fb.jpg)  
Figure 3: Representative examples from each of the six prior categories in ViSiP-Bench. Green answers denote the correct visual answer, which conflicts with the expected answer under the text-only condition.

Success: insufficient model capability leads to random correct responses; (2) Low-quality Framing: questions that interfere with the model’s prior selection; (3) Anomalous Prior: models inherently treat the anomalous answer as prior.

Therefore, we introduce ViSiP-Bench, a redesigned benchmark for evaluating language prior dependency. Inspired by Ilievski et al. (2021), we categorize language priors into six domains tailored for MLLMs (Figure 3). The dataset is grounded in real-world rare scenes retrieved from the web, supplemented by generative models including GPT-Image-2(OpenAI, 2026b) and Nano Banana 2(Google, 2026) to synthesize images across diverse visual styles, thereby avoiding biases introduced by image source or visual style. To eliminate Low-quality Framing, every sample undergoes double-blind human review to ensure annotation quality. To mitigate the confounding effects of Chance Success and Anomalous Prior, ViSiP-Bench adopts a dynamic prior induction protocol: each model is first queried without any image to elicit its Blank-condition answer. Building on this, PAR (Prior Adherence Rate) measures the frequency of a model adhering to its Blank-condition answer when the image is added. To further render both PAR and Accuracy more precise, items already answered correctly under the Blank condition are filtered out. Moreover, VIG (Visual Information Gain) quantifies the confidence shift induced by adding the image, providing a continuous and fine-grained measure of prior resistance.

Evaluation on ViSiP-Bench reveals that prior adherence is pervasive across mainstream MLLMs and does not improve with scale; on several categories, larger models adhere to their priors even more strongly than smaller counterparts. Among inference-time interventions, prompt-level strategies yield only marginal gains, and chain-ofthought reasoning consistently amplifies prior adherence. These findings indicate that inferencetime strategies alone are insufficient to address language prior dependency.

We therefore propose MAPS (Model-Aware Prior Softening), a training-based method that softens excessive prior reliance by enriching each model with targeted factually grounded knowledge of rare-but-real scenarios, without suppressing priors indiscriminately or degrading existing capabilities. MAPS substantially improves VIG and reduces PAR on ViSiP-Bench, while preserving performance on reasoning and general vision benchmarks, demonstrating that enhanced visual reliance can be achieved without compromising general capabilities.

## 2 Related Works

## 2.1 Language Priors in MLLMs

Recent analyses reveal that language priors in MLLMs strike a delicate balance between visual perception and reasoning. Han et al. (2026) show that while priors benefit high-level reasoning, they frequently mislead models in perception-critical tasks. Mechanistic studies further localize this tension: visual evidence only begins to influence hidden states at a specific point in the transformer stack (Long et al., 2026), and extended chain-ofthought reasoning further suppresses visual attention, amplifying hallucination (Xu et al., 2026b).

## 2.2 Benchmarking Language Priors

Several benchmarks have been proposed to quantify language prior dependency. Hallusion-Bench (Guan et al., 2024) pairs Easy/Hard counterparts on counterfactual images; VLind-Bench (Lee et al., 2025) gates LP scoring through a four-stage sequential protocol; Golovanevsky et al. (2025) and VisuLogic (Xu et al., 2026a) design stimuli that resist language-mediated shortcuts. Despite their diversity, these works share a common evaluation pattern that assumes shared priors across models and relies solely on accuracy, which, as shown in Figure 2, underestimates the complexity of prior dependency.

## 2.3 Mitigating Language Priors

Methods that mitigate excessive prior reliance fall into three families. Decoding-time methods adjust output distributions without modifying weights, typically by contrasting logits from perturbed or instruction-only forward passes (Leng et al., 2024; Favero et al., 2024). Attention-level interventions redirect internal computation toward visual tokens by editing attention weights (Huang et al., 2024; Golovanevsky et al., 2025; Liu et al., 2024; Zhou et al., 2025). Training-based methods modify the model itself through preference optimization, counterfactual training, or visual grounding objectives (Zhang et al., 2025b,a; Wu et al., 2026).

## 3 ViSiP-Bench

Existing benchmarks for language prior dependency share three limitations. First, they assume shared commonsense priors across models and apply a fixed anomalous answer set, yet cross-model prior agreement falls below 80%, as demonstrated in Figure 4. Second, over 15% of items can be answered correctly without any image (Figure 2), stemming from three sources: (1) Chance Success: the model lacks a sufficiently strong prior on the question, leading to random correct responses rather than genuine prior-driven selection; (2) Lowquality Framing: poorly designed questions that interfere with the model’s prior selection, either by implicitly hinting at the anomalous answer (e.g., “what is unusual in this image?”) or by introducing ambiguity (e.g., asking “what color is the watermelon?” without specifying flesh or rind); (3) Anomalous Prior: the model inherently treats the anomalous answer as its own prior, meaning the question does not actually induce a prior-vision conflict. Identification methodology and typical cases are provided in Appendix A. Together, these factors cause accuracy to overestimate prior resistance. Third, conflict stimuli typically depict physically impossible counterfactuals rather than rare-but-real scenarios, conflating prior dependency with reasoning about the impossible.

![](images/f0873a29a8893f2bf29598d14c6ed48704cb4fa8d4fb92d5a0db1c6b2c32cbe2.jpg)  
Figure 4: Cross-model prior agreement matrix across six representative models.

To address these limitations, we introduce ViSiP-Bench (Visual Signal vs. Language Prior). ViSiP-Bench ensures all images depict rare-but-real scenarios rather than physically impossible counterfactuals, applies rigorous quality review to eliminate low-quality framing, and introduces model-aware metrics that filter out items answerable without any image, further mitigating the confounding effects of all three sources. Table 1 presents a detailed comparison between ViSiP-Bench and existing benchmarks.

## 3.1 Data Construction

ViSiP-Bench organizes samples into six domains following Ilievski et al. (2021): Physical Properties, Time & Space, Scientific & Mathematic, Customs & Culture, Psychology Cognition, and Function Cognition. Dataset construction proceeds in four steps.

Step 1: Prior Enumeration. For each domain, we prompt Claude Opus 4.7 (Anthropic, 2026b) to enumerate prevalent commonsense priors (e.g., “waterflows downward”), yielding 200 priors per domain and 1,200 priors in total.

Step 2: Rare-but-Real Generation. For each prior, Claude Opus 4.7 generates three rare-but-real counterpart scenarios that genuinely exist but violate the typical expectation (e.g., “water appears to flow upward in a magic performance”).

<table><tr><td rowspan="2">Benchmark</td><td rowspan="2">Venue</td><td rowspan="2">Size</td><td rowspan="2">Cate- gories</td><td rowspan="2">Image Source</td><td rowspan="2">Style Diversity</td><td rowspan="2">Prior Annotation</td><td rowspan="2">Decision-level Metric</td><td rowspan="2">Probability-level Metric</td></tr><tr><td></td></tr><tr><td>ROME</td><td>EMNLP&#x27;23</td><td>1,563</td><td>4</td><td>Counterfactual</td><td>x</td><td>Fixed</td><td>Acc only</td><td>x</td></tr><tr><td>HallusionBench</td><td>CVPR&#x27;24</td><td>527</td><td>2</td><td>Counterfactual</td><td>x</td><td>Fixed</td><td>Acc only</td><td>x</td></tr><tr><td>PvP</td><td>EMNLP&#x27;25</td><td>2,904</td><td>2</td><td>Counterfactual</td><td>x</td><td>Fixed</td><td>Acc only</td><td>x</td></tr><tr><td>VLind-Bench</td><td>NAACL&#x27;25</td><td>2,576</td><td>11</td><td>Counterfactual</td><td>x</td><td>Fixed</td><td>Acc only</td><td>x</td></tr><tr><td>ViLP</td><td>ICML&#x27; 25</td><td>900</td><td>3</td><td>Counterfactual</td><td>x</td><td>Fixed</td><td>Acc only</td><td>x</td></tr><tr><td>ViSiP-Bench</td><td>Ours</td><td>4500</td><td>6</td><td>Rare-but-Real</td><td>√</td><td>Dynamic</td><td>PAR + Filtered Acc</td><td>VIG</td></tr></table>

Table 1: Overview of benchmarks for language prior dependency. ✓ and ✗ denote presence and absence of the corresponding property, respectively.

Step 3: Image Collection. For each rare-but-real scenario, real photographs are first retrieved from the web; where sufficient images cannot be found, GPT-Image-2 and Nano Banana 2 synthesize supplementary samples across diverse visual styles, including photorealistic, anime, oil painting, and sci-fi. Examples of each style are provided in $\mathsf { A p - }$ pendix B.

Step 4: Question and Option Design. Annotators identify the anomalous element in each image and design a question with four answer choices, such that the correct answer under the visual condition differs clearly from the expected answer under the text-only condition. Each sample then undergoes a quality review requiring three criteria to be satisfied: (1) the image is visually clear and unambiguously supports a single correct answer; (2) all answer choices are directly relevant to the question, preventing models from selecting the correct answer by elimination; and (3) the question wording contains no hint of anomaly, ensuring that prior activation is not suppressed before the model processes the image. Details on annotator requirements and details are provided in Appendix D.

Through the above pipeline, we construct ViSiP-Bench, a benchmark of 4,500 rare-but-real imagequestion pairs spanning six prior categories and diverse visual styles, each designed to surface genuine conflicts between visual signals and language priors.

## 3.2 Dynamic Prior Induction and Metrics

To evaluate language prior dependency in a modelspecific and image-grounded manner, ViSiP-Bench queries each model under two conditions. In the Blank condition, no image is provided and only the question and options are given, from which we extract the model’s response and locate the corresponding answer token to obtain the prior probability distribution $p _ { B }$ over the answer choices; the resulting answer is denoted $a _ { B }$ . In the Visual condition, the image is additionally provided, and the same extraction procedure yields the visual probability distribution $p _ { V }$ and answer $a _ { V }$ . The groundtruth visual answer is denoted $a ^ { * }$

Dynamic Prior Induction. ViSiP-Bench adopts a dynamic prior induction mechanism in which $a _ { B }$ is elicited independently for each model under the Blank condition, rather than prescribed by a shared annotation. This ensures that prior resistance is measured relative to each model’s own prior, not an externally imposed one.

Prior Adherence Rate (PAR). PAR measures the frequency with which a model adheres to its prior answer $a _ { B }$ when the image is provided, restricted to samples where $a _ { B } \neq a ^ { * }$ to ensure a genuine prior-vision conflict:

$$
\mathrm { P A R } = \frac { \sum _ { i : a _ { B , i } \neq a _ { i } ^ { * } } \mathbf { 1 } \left[ a _ { V , i } = a _ { B , i } \right] } { \sum _ { i } \mathbf { 1 } \left[ a _ { B , i } \neq a _ { i } ^ { * } \right] }\tag{1}
$$

A higher PAR indicates stronger prior adherence. Dynamic Prior Induction ensures that PAR more precisely reflects each model’s degree of prior stubbornness.

Filtered Accuracy. To remove the confounding effect of image-free correct responses, we filter out all samples where $a _ { B } = a ^ { * }$ , and compute accuracy exclusively on the remaining items. This yields a more faithful measure of genuine prior resistance than standard accuracy.

Visual Information Gain (VIG). PAR and Filtered Accuracy capture only the model’s final decision, leaving the degree of confidence shift unmeasured. VIG fills this gap by quantifying the log-odds change induced by adding the image, providing a continuous and fine-grained measure of prior resistance:

$$
\mathrm { V I G } = \ln \frac { p _ { V } ( a ^ { * } ) + \epsilon } { 1 - p _ { V } ( a ^ { * } ) + \epsilon } - \ln \frac { p _ { B } ( a ^ { * } ) + \epsilon } { 1 - p _ { B } ( a ^ { * } ) + \epsilon }\tag{2}
$$

where ϵ is a small constant added for numerical stability.

<table><tr><td>Model</td><td>PAR↓</td><td>Acc ↑</td><td>FC</td><td> $\mathbf { V I G _ { a v g } } \uparrow$ </td><td> $\mathbf { V I G _ { m e d } } \uparrow$ </td></tr><tr><td colspan="6">Open-source Models</td></tr><tr><td>InternVL3.5-2B</td><td>51.51</td><td>42.94</td><td>182</td><td>4.41</td><td>4.11</td></tr><tr><td>InternVL3.5-8B</td><td>47.17</td><td>44.64</td><td>156</td><td>7.13</td><td>7.48</td></tr><tr><td>InternVL3.5-14B</td><td>38.64</td><td>53.17</td><td>131</td><td>9.62</td><td>9.74</td></tr><tr><td>InternVL3.5-38B</td><td>30.04</td><td>63.01</td><td>102</td><td>10.12</td><td>10.31</td></tr><tr><td>InternVL3-2B</td><td>62.77</td><td>30.69</td><td>223</td><td>2.91</td><td>2.30</td></tr><tr><td>InternVL3-14B</td><td>48.40</td><td>41.33</td><td>161</td><td>6.21</td><td>6.33</td></tr><tr><td>InternVL3-38B</td><td>36.80</td><td>56.49</td><td>118</td><td>9.91</td><td>9.65</td></tr><tr><td>InternVL3-78B</td><td>36.95</td><td>61.26</td><td>81</td><td>12.84</td><td>12.93</td></tr><tr><td>Qwen2.5-VL-3B-Instruct</td><td>63.18</td><td>28.84</td><td>236</td><td>3.21</td><td>2.27</td></tr><tr><td>Qwen2.5-VL-7B-Instruct</td><td>55.35</td><td>34.51</td><td>198</td><td>5.56</td><td>4.93</td></tr><tr><td>Qwen2.5-VL-32B-Instruct</td><td>42.81</td><td>48.87</td><td>160</td><td>8.77</td><td>8.96</td></tr><tr><td>Qwen2.5-VL-72B-Instruct</td><td>37.03</td><td>54.86</td><td>119</td><td>9.64</td><td>9.87</td></tr><tr><td>Qwen3-VL-2B-Instruct</td><td>59.41</td><td>33.94</td><td>201</td><td>4.94</td><td>4.11</td></tr><tr><td>Qwen3-VL-4B-Instruct</td><td>51.12</td><td>40.48</td><td>188</td><td>7.87</td><td>8.33</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>42.67</td><td>51.29</td><td>147</td><td>10.64</td><td>10.82</td></tr><tr><td>Qwen3.5-2B</td><td>55.61</td><td>34.17</td><td>164</td><td>3.68</td><td>2.90</td></tr><tr><td>Qwen3.5-9B</td><td>51.24</td><td>39.36</td><td>137</td><td>5.24</td><td>5.24</td></tr><tr><td>Qwen3.5-27B</td><td>39.28</td><td>54.94</td><td>111</td><td>8.64</td><td>8.85</td></tr><tr><td>llava-onevision-qwen2-0.5b</td><td>72.11</td><td>21.45</td><td>381</td><td>2.81</td><td>1.99</td></tr><tr><td>llava-onevision-qwen2-7b</td><td>67.10</td><td>25.36</td><td>317</td><td>3.39</td><td>2.48</td></tr><tr><td>llava-onevision-qwen2-72b</td><td>59.54</td><td>32.00</td><td>208</td><td>7.74</td><td>7.91</td></tr><tr><td colspan="6">Proprietary Models</td></tr><tr><td>GPT-5.5</td><td>26.01</td><td>69.94</td><td>39</td><td>一</td><td></td></tr><tr><td>Claude Sonnet 4.6</td><td>30.44</td><td>65.87</td><td>22</td><td>一</td><td></td></tr><tr><td>Gemini-3-flash-preview</td><td>28.64</td><td>68.00</td><td>48</td><td>一</td><td></td></tr></table>

Table 2: Evaluation results on ViSiP-Bench. PAR, Filtered Accuracy (Acc), and Filtered Count (FC) are reported for all models; $\mathrm { V I G } _ { a v g }$ (average) and $\mathrm { V I G } _ { m e d }$ (median) are reported for open-source models only as they require access to token-level probabilities. ↓ denotes lower is better; ↑ denotes higher is better. Best results are bolded, second-best results are underlined.

As VIG requires access to token-level probabilities, it is applicable to open-source models only, whereas PAR and Filtered Accuracy are applicable to all models.

## 3.3 Evaluation

We evaluate 24 models on ViSiP-Bench, spanning six open-source model families (InternVL3.5 (Wang et al., 2025), InternVL3 (Zhu et al., 2025), Qwen2.5-VL (Team, 2025a), Qwen3-VL (Team, 2025b), Qwen3.5 (Team, 2026), and LLaVA-OneVision (An et al., 2025)) and three proprietary models (GPT-5.5 (OpenAI, 2026a), Claude Sonnet 4.6 (Anthropic, 2026a), and Gemini-3-flashpreview (Google, 2025)). Each experiment is repeated three times and results are averaged. For VIG computation, ϵ is set to $1 0 ^ { - 7 }$ , yielding a theoretical range of (−33, 33). Table 2 and Figure 5 present the results. We highlight three findings.

Finding 1: Prior adherence is pervasive and not resolved by scale. Prior adherence remains a widespread issue across all evaluated models. Within open-source families, scaling up model size does not reliably reduce PAR; in certain categories, larger models do not demonstrate a consistent advantage in prior resistance.

Finding 2: Larger models exhibit a VIG polarization effect. As shown in Figure 5a, while larger models accumulate a higher proportion of samples in the high VIG range, the proportion in the low VIG range does not decrease accordingly and in some families even increases, indicating that they either respond strongly to visual evidence or firmly adhere to their language priors, which we term VIG Polarization.

Finding 3: Category-level prior resistance is an intrinsic property of the prior type. As shown in Figure 5b, prior resistance varies substantially across the six domains, yet this pattern remains largely consistent across models. Scientific & Mathematic exhibits the highest PAR values across all models, while Physical Properties proves most amenable to visual override. A detailed percategory analysis is provided in Appendix E.

![](images/774f241d049b8606a3e39fdac79f66210b5bf13a56f738d2c0185acbc8b5eeed.jpg)  
(a) VIG distribution across models.

![](images/358367542745e2cb3a5972584f6348328383540da0ef4f577eebc0e724e1351f.jpg)  
(b) Per-category PAR across representative models.

Figure 5: Overall analysis of model performance on ViSiP-Bench. Larger models exhibit a VIG polarization effect (a), and category-level prior resistance remains consistent across models (b).  
![](images/78130cfcc12fff470b38a10552101b7ba90e82d1ac86b199efbd4f0bac488404.jpg)  
Figure 6: Overview of the MAPS pipeline.

## 4 Model-Aware Prior Softening

## 4.1 Motivation

§ 3.3 demonstrates that language prior dependency persistently impairs visual perception across mainstream MLLMs and does not diminish with scale. Existing inference-time interventions offer a training-free alternative, but their effect is fundamentally bounded by what decoding-time signals can recover from a fixed network, making it difficult to achieve substantial prior resistance, as empirically confirmed in § 5.2. Training-based methods offer stronger corrections, but existing approaches directly suppress the probability of prior-driven outputs during training, indiscriminately reducing the model’s confidence in its prior knowledge regardless of whether that knowledge is harmful or benign. This risks disrupting the model’s broader understanding of the real world and degrading general capabilities as a consequence.

We propose MAPS (Model-Aware Prior Softening), which takes a different stance. Rather than suppressing priors, MAPS softens them by enriching the model’s knowledge with rare-but-real scenarios that the prior rarely accounts for. This leaves the model’s existing knowledge intact while specifically targeting the priors that demonstrably interfere with visual perception.

## 4.2 Method

As illustrated in Figure 6, MAPS operates in five steps.

Step 1: Question Generation. For each prior, Claude Opus 4.7 generates a question with 10 plausible answer options designed to elicit the model’s prior belief.

Step 2: Probabilities Extraction. Each target model is queried on all generated questions, and the probability distribution over the 10 options is obtained via softmax over the output logits.

Step 3: Queries Selection. Questions where any single option receives a probability exceeding α (a hyperparameter controlling the strength of prior selection) are retained. These correspond to priors the model holds with high confidence, and their associated prior statements are selected.

Step 4: Knowledge Generation. For each selected prior statement, Claude Opus 4.7 generates

<table><tr><td>Method</td><td>PAR↓</td><td>Acc ↑</td><td>FC</td><td> $\mathbf { V I G _ { a v g } } ^ { + } .$  ←</td><td> $\mathbf { V I G } _ { \mathbf { m e d } } \uparrow$ </td></tr><tr><td>Base</td><td>42.67</td><td>51.29</td><td>147</td><td>10.64</td><td>10.82</td></tr><tr><td colspan="6">Inference-time Methods</td></tr><tr><td>VFP</td><td>40.60</td><td>53.01</td><td>147</td><td>10.78</td><td>10.90</td></tr><tr><td>CoT</td><td>47.71</td><td>43.11</td><td>147</td><td>8.11</td><td>8.32</td></tr><tr><td>AS</td><td>42.11</td><td>52.08</td><td>147</td><td>10.88</td><td>11.22</td></tr><tr><td>VCD</td><td>39.15</td><td>55.48</td><td>147</td><td>12.11</td><td>13.08</td></tr><tr><td>PAI</td><td>39.01</td><td>55.23</td><td>147</td><td>12.39</td><td>13.97</td></tr><tr><td colspan="6">Training-based Methods</td></tr><tr><td>Post-hoc</td><td>32.05</td><td>61.94</td><td>385</td><td>19.81</td><td>18.92</td></tr><tr><td>NaPO</td><td>37.09</td><td>57.34</td><td>209</td><td>16.33</td><td>17.12</td></tr><tr><td>MAPS (Ours)</td><td>35.09</td><td>59.01</td><td>154</td><td>18.84</td><td>18.90</td></tr></table>

Table 3: Comparison of inference-time and training-based methods on ViSiP-Bench evaluated on Qwen3-VL-8B-Instruct.

20 rare-but-real single-sentence descriptions that are factually grounded yet contradict the prior. To prevent data contamination, the generated content is kept entirely disjoint from ViSiP-Bench at the prompt level.

Step 5: Fine-Tuning. The rare-but-real sentences are aggregated into a model-specific corpus for supervised fine-tuning via LoRA, where each sentence serves as a training target to enrich the model’s world knowledge.

## 5 Experiments

## 5.1 Experimental Setup

Implementation Details. All training-based methods are implemented on 4 NVIDIA A100 80GB GPUs. MAPS is applied to Qwen3-VL-8B-Instruct via LoRA fine-tuning for 2 epochs, with the prior probing threshold set to $\alpha = 0 . 9 0$ unless specified. To assess whether these improvements generalize beyond a single backbone, we further apply MAPS under an identical configuration to six additional models spanning three architecture families, with full results reported in Appendix C.

Baselines. We compare MAPS against representative methods from both inference-time and trainingbased families.

For inference-time methods:VFP (Visual-Focused Prompt) instructs the model to focus on the image before responding; CoT (Wei et al., 2022) prompts step-by-step reasoning; AS (Attention Steering) uniformly amplifies visual token attention weights without modifying logits; VCD (Leng et al., 2024) subtracts logits from a perturbed-image forward pass to suppress priordriven tokens; PAI (Liu et al., 2024) combines visual attention amplification with text-only logit subtraction as a language prior basal.

For training-based methods: Post-hoc (Zhang et al., 2025a) penalizes prior-driven outputs via empty-image probability estimation; NaPO (Zhang et al., 2025b) applies noise-aware preference optimization to suppress the model’s preference for language-prior-driven answers over visually grounded ones.

Benchmarks for Generalization. We evaluate generalization on two dimensions. For visual reasoning, we use MMMU-Pro (Yue et al., 2025), which explicitly filters out text-only solvable questions to assess genuine visual reasoning across multiple disciplines, and VisuLogic (Xu et al., 2026a), which eliminates all textual shortcuts and focuses purely on vision-based logical deduction. For general visual understanding, we use MMStar (Chen et al., 2024), which covers six core visual capabilities, and RealWorldQA, which tests spatial and physical understanding on authentic real-world photographs.

## 5.2 Main Results

Table 3 presents the results of all methods on ViSiP-Bench. Inference-time methods yield limited improvements overall, while training-based methods prove more effective but differ substantially in how they achieve gains.

Among inference-time methods, CoT consistently hurts performance, confirming that extended reasoning amplifies prior adherence rather than reducing it. VFP and AS yield only marginal gains, suggesting that neither explicit visual instruction nor attention redistribution is sufficient to shift the model’s confidence toward visual evidence. VCD and PAI achieve the best inference-time results, indicating that directly suppressing prior-driven logits at the decoding stage is a more effective direction, though the overall improvement remains limited.

<table><tr><td>Method</td><td colspan="2">Reasoning</td><td colspan="2">General</td></tr><tr><td></td><td>MMMU-Pro ↑</td><td>VisuLogic ↑</td><td>MMStar ↑</td><td>RealWorldQA ↑</td></tr><tr><td>Base</td><td>55.90</td><td>28.14</td><td>70.90</td><td>71.50</td></tr><tr><td>Post-hoc</td><td>41.88</td><td>21.50</td><td>56.18</td><td>53.08</td></tr><tr><td>NaPO</td><td>49.51</td><td>24.93</td><td>66.51</td><td>66.00</td></tr><tr><td>MAPS (Ours)</td><td>56.84</td><td>27.94</td><td>69.27</td><td>71.68</td></tr></table>

Table 4: Generalization results on reasoning and general vision benchmarks. All methods are evaluated on Qwen3- VL-8B-Instruct.

<table><tr><td>Threshold</td><td>PAR↓</td><td> $\mathbf { A c c } \uparrow$ </td><td>FC</td><td> $\mathbf { V I G _ { a v g } } \uparrow$ </td><td> $\mathbf { V I G _ { m e d } } \uparrow$  1</td><td> $\mathbf { M M M U - P r o } \uparrow$ </td><td>MMStar ↑</td></tr><tr><td> $\alpha = 0 . 5 0$ </td><td>33.09</td><td>60.93</td><td>218</td><td>19.81</td><td>19.08</td><td>50.90</td><td>63.84</td></tr><tr><td> $\alpha = 0 . 8 0$ </td><td>34.24</td><td>60.10</td><td>181</td><td>19.41</td><td>19.90</td><td>54.17</td><td>66.81</td></tr><tr><td> $\alpha = 0 . 9 0$ </td><td>35.09</td><td>59.01</td><td>154</td><td>18.84</td><td>18.90</td><td>56.84</td><td>69.27</td></tr><tr><td> $\alpha = 0 . 9 5$ </td><td>38.41</td><td>54.09</td><td>150</td><td>17.77</td><td>18.13</td><td>56.10</td><td>71.02</td></tr><tr><td> $\alpha = 1 . 0 0$ </td><td>42.67</td><td>51.29</td><td>147</td><td>10.64</td><td>10.82</td><td>55.90</td><td>70.90</td></tr></table>

Table 5: Ablation study on the prior probing threshold α. Results on both ViSiP-Bench and generalization benchmarks are reported. $\alpha = 1 . 0 0$ corresponds to the base model without any fine-tuning.

Among training-based methods, Post-hoc achieves the lowest PAR but at a significant cost: FC increases from 147 to 385, indicating that its probability penalization severely disrupts the model’s prior knowledge and causes prior responses to collapse toward random. NaPO improves over the baseline but with limited gains. MAPS achieves the strongest PAR and VIG improvements while maintaining FC at 154, close to the baseline. This demonstrates that supplementing the model with factually grounded rare-but-real knowledge, rather than suppressing priors directly, is a more effective and less destructive approach to enhancing visual reliance.

This advantage is not specific to Qwen3-VL-8B-Instruct. Extending MAPS to six additional models spanning three architecture families yields consistent reductions in PAR and gains in VIG, with FC remaining close to the base model throughout, as reported in Table 8 of Appendix C.

## 5.3 Further Analysis

Generalization Analysis. Table 4 reports results on reasoning and general vision benchmarks. Posthoc suffers substantial degradation across all four benchmarks, consistent with its destructive effect on prior knowledge observed in Table 3. NaPO also declines across all tasks, though to a lesser extent, indicating that preference-based optimization incurs a non-trivial cost to general capabilities. MAPS maintains performance close to the baseline across all benchmarks, demonstrating that it leaves the model’s reasoning and general visual capabilities largely intact. Unlike prior suppression methods that sacrifice general capabilities for prior resistance, MAPS acquires more comprehensive world knowledge through rare-but-real supervision, preserving the model’s understanding of the real world.

This preservation effect likewise generalizes across models and architectures. Extending MAPS to six additional models leaves performance on all four generalization benchmarks within a narrow margin, with the largest absolute change reaching only 0.87 points and no consistent direction of degradation, as reported in Table 9 of Appendix C.

Ablation on Prior Probing Threshold. To isolate the effect of the threshold α from training data quantity, we control the total training data to match that of $\alpha ~ = ~ 0 . 9 0 .$ For lower thresholds where more priors are selected, we proportionally reduce the number of rare-but-real descriptions per prior; for higher thresholds where fewer priors are selected, we prompt Claude Opus 4.7 to generate additional descriptions per prior to reach the same total. Table 5 reports results across four threshold values. A low threshold such as $\alpha = 0 . 5 0$ selects a large number of priors indiscriminately, including many that the model does not strongly hold, which interferes with targeted prior softening and leads to suboptimal performance on both ViSiP-Bench and generalization benchmarks. A high threshold such as $\alpha = 0 . 9 5$ is overly restrictive, covering too few priors to achieve meaningful improvement in visual perception. $\alpha = 0 . 9 0$ achieves the best balance, maximally improving visual perception while preserving the model’s reasoning and general capabilities.

## 6 Conclusion

We present ViSiP-Bench, a redesigned benchmark for evaluating language prior dependency in MLLMs. Existing benchmarks rely on a sharedprior assumption, yet cross-model prior agreement falls below 80%, and accuracy overestimates prior resistance as over 15% of items can be answered without any image, motivating a re-examination of how prior resistance should be measured.

ViSiP-Bench addresses these issues through a model-aware dynamic prior induction protocol paired with three complementary metrics. PAR and Filtered Accuracy capture prior resistance at the decision level by conditioning on each model’s own elicited prior, while VIG provides a continuous measure of the confidence shift induced by visual evidence. Evaluation of 24 models reveals that prior adherence is pervasive across model families, does not diminish with scale, and manifests as VIG Polarization in larger models, which either override their priors decisively or adhere to them just as firmly. These findings suggest that inference-time interventions alone are insufficient, motivating a training-based solution that avoids indiscriminately suppressing the model’s world knowledge.

We therefore propose MAPS, which identifies each model’s most strongly held priors and softens them by enriching the model with targeted, factually grounded knowledge of rare-but-real scenarios, rather than suppressing priors indiscriminately. MAPS achieves substantial improvements in PAR and VIG on ViSiP-Bench while preserving performance on reasoning and general vision benchmarks, and this behavior holds consistently across seven models spanning three architecture families, demonstrating that enhanced visual reliance can be obtained without sacrificing a model’s broader capabilities.

## Limitations

First, the six categories in ViSiP-Bench, though designed to broadly cover commonsense knowledge relevant to MLLMs, follow a fixed taxonomy and cannot capture every form of world knowledge a model may rely on. Priors rooted in specialized domains or in rapidly evolving knowledge fall outside this taxonomy, and whether the patterns of prior adherence we observe extend to such cases remains an open question for future benchmarks to address.

Second, VIG relies on token-level output probabilities and is therefore inapplicable to closed models accessed only through APIs. This leaves the VIG Polarization phenomenon reported in this work unverified for proprietary models, which may exhibit different confidence dynamics than what we observe here.

## Ethical Considerations

Data construction relies on proprietary generative models, including Claude Opus 4.7 for prior enumeration and scenario generation, and GPT-Image-2 and Nano Banana 2 for image synthesis, all accessed through official APIs in accordance with the respective providers’ usage policies. Evaluation additionally involves querying proprietary models such as GPT-5.5, Claude Sonnet 4.6, and Gemini-3-flash-preview through their official APIs, again in compliance with the applicable terms of service. Content synthesized by generative models, as well as web-sourced images, is subject to the same threecriteria quality review, which additionally serves to filter out any inappropriate or misleading generations before inclusion in ViSiP-Bench. Human annotation involved in question and option design, as well as quality verification, was carried out by voluntary annotators who were fully informed of the task and compensated fairly for their time; the annotation task does not involve exposure to violent, explicit, or otherwise harmful content, as all images depict ordinary, if uncommon, real-world scenarios.

We note that generative AI tools were employed only for editing purposes, such as improving wording and sentence clarity, and played no role in the underlying research. All ideas, experiments, analyses, and conclusions presented in this paper originate from and were validated by the authors.

## Acknowledgments

This work was supported by the National Natural Science Foundation of China (Grant No.U24A20250), Sichuan Science and Technology Program (No.2025ZNSFSC1487), the Fundamental Research Funds for the Central Universities (No.ZYGX2024J022 and No.ZYGX2024Z005), and the Science, Technology and Innovation Project of Shenzhen Longhua District (No.20260309G23410662).

## References

Xiang An, Yin Xie, Kaicheng Yang, Wenkang Zhang, Xiuwei Zhao, Zheng Cheng, Yirui Wang, Songcen Xu, Changrui Chen, Didi Zhu, et al. 2025. Llava-onevision-1.5: Fully open framework for democratized multimodal training. arXiv preprint arXiv:2509.23661.

Anthropic. 2026a. Claude sonnet 4.6. https://www. anthropic.com/news/claude-sonnet-4-6. Accessed: 2026-05-26.

Anthropic. 2026b. Introducing claude opus 4.7. https: //www.anthropic.com/news/claude-opus-4-7. Accessed: 2026-05-26.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. 2024. Are we on the right way for evaluating large vision-language models? Advances in Neural Information Processing Systems, 37:27056–27087.

Alessandro Favero, Luca Zancato, Matthew Trager, Siddharth Choudhary, Pramuditha Perera, Alessandro Achille, Ashwin Swaminathan, and Stefano Soatto. 2024. Multi-modal hallucination control by visual information grounding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14303–14312.

Michal Golovanevsky, William Rudman, Michael A Lepori, Amir Bar, Ritambhara Singh, and Carsten Eickhoff. 2025. Pixels versus priors: Controlling knowledge priors in vision-language models through visual counterfacts. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 24848–24863.

Google. 2025. Gemini 3 flash preview. https://blog. google/products/gemini/gemini-3-flash/. Accessed: 2026-05-26.

Google. 2026. Build with Nano Banana 2: our best image generation and editing model. https://blog.google/innovation-and-ai/ technology/developers-tools/ build-with-nano-banana-2/. Accessed: 2026- 05-26.

Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. 2024. Hallusionbench: An advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14375–14385.

Junlin Han, Shengbang Tong, David Fan, Yufan Ren, Koustuv Sinha, Philip Torr, and Filippos Kokkinos. 2026. Learning to see before seeing: Demystifying llm visual priors from language pre-training. In International Conference on Learning Representations, volume 2026, pages 61025–61055.

Qidong Huang, Xiaoyi Dong, Pan Zhang, Bin Wang, Conghui He, Jiaqi Wang, Dahua Lin, Weiming Zhang, and Nenghai Yu. 2024. Opera: Alleviating hallucination in multi-modal large language models via over-trust penalty and retrospection-allocation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13418– 13427.

Filip Ilievski, Alessandro Oltramari, Kaixin Ma, Bin Zhang, Deborah L. McGuinness, and Pedro Szekely. 2021. Dimensions of commonsense knowledge. Knowledge-Based Systems, 229:107347.

Kang-il Lee, Minbeom Kim, Seunghyun Yoon, Minsung Kim, Dongryeol Lee, Hyukhun Koh, and Kyomin Jung. 2025. VLind-bench: Measuring language priors in large vision-language models. In Findings of the Association for Computational Linguistics: NAACL 2025, pages 4129–4144, Albuquerque, New Mexico. Association for Computational Linguistics.

Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, and Lidong Bing. 2024. Mitigating object hallucinations in large visionlanguage models through visual contrastive decoding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13872–13882.

Shi Liu, Kecheng Zheng, and Wei Chen. 2024. Paying more attention to image: A training-free method for alleviating hallucination in lvlms. In European Conference on Computer Vision, pages 125–140. Springer.

Lin Long, Changdae Oh, Seongheon Park, and Sharon Li. 2026. Understanding language prior of lvlms by contrasting chain-of-embedding. In International Conference on Learning Representations, volume 2026, pages 94723–94754.

Tiange Luo, Ang Cao, Gunhee Lee, Justin Johnson, and Honglak Lee. 2024. Probing visual language priors in vlms. arXiv preprint arXiv:2501.00569.

OpenAI. 2026a. Gpt-5.5. https://openai.com/ index/introducing-gpt-5-5/. Accessed: 2026- 05-26.

OpenAI. 2026b. Introducing ChatGPT Images 2.0. https://openai.com/blog/ introducing-chatgpt-images-2-0. Accessed: 2026-05-26.

Qwen Team. 2025a. Qwen2.5-vl. https://qwen.ai/ blog?id=qwen2.5-vl.

Qwen Team. 2025b. Qwen3vl. https://qwen.ai/ blog?id=qwen3-vl.

Qwen Team. 2026. Qwen3.5. https://qwen.ai/ blog?id=qwen3.5.

Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. 2025. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed Chi, Quoc V Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems, volume 35, pages 24824–24837. Curran Associates, Inc.

Yixuan Wu, Yang Zhang, Jian Wu, Philip Torr, and Jindong Gu. 2026. Postalign: Multimodal grounding as a corrective lens for mllms. In International Conference on Learning Representations, volume 2026, pages 78875–78894.

Weiye Xu, Jiahao Wang, Weiyun Wang, Zhe Chen, Wengang Zhou, Aijun Yang, Lewei Lu, Houqiang Li, Xiaohua Wang, Xizhou Zhu, et al. 2026a. Visulogic: A benchmark for evaluating visual reasoning in multimodal large language models. In International Conference on Learning Representations, volume 2026, pages 25966–26003.

Zhongxing Xu, Chengzhi Liu, Qingyue Wei, Juncheng Wu, James Zou, Xin Wang, Yuyin Zhou, and Sheng Liu. 2026b. More thinking, less seeing? assessing amplified hallucination in multimodal reasoning models. Advances in Neural Information Processing Systems, 38:82878–82905.

Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, et al. 2025. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15134– 15186.

YiFan Zhang, Yang Shi, Weichen Yu, Qingsong Wen, Xue Wang, Wenjing Yang, Zhang Zhang, Liang Wang, and Rong Jin. 2025a. Debiasing multimodal large language models via penalization of language priors. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 4232–4241.

Zefeng Zhang, Hengzhu Tang, Jiawei Sheng, Zhenyu Zhang, Yiming Ren, Zhenyang Li, Dawei Yin, Duohe Ma, and Tingwen Liu. 2025b. Debiasing multimodal large language models via noise-aware preference optimization. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9423–9433.

Guanyu Zhou, Yibo Yan, Xin Zou, Kun Wang, Aiwei Liu, and Xuming Hu. 2025. Mitigating modality prior-induced hallucinations in multimodal large language models via deciphering attention causality. In International Conference on Learning Representations, volume 2025, pages 54415–54439.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. 2025. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479.

## A Data Quality Analysis

## A.1 Identification of No-Image Correct Responses

Over 15% of items in existing benchmarks can be answered correctly without any image. To identify the source of each such response and categorize it into one of three types (Chance Success, Lowquality Framing, or Anomalous Prior), we adopt a hybrid pipeline combining automated screening with human verification.

The pipeline proceeds as follows. All items answered correctly under the Blank condition are first collected as candidates. Each candidate then undergoes the following three-stage procedure.

Stage 1: Human Verification of Question Quality. Human annotators examine each candidate question along two dimensions. First, they assess whether the question is associated with a sufficiently strong and unambiguous commonsense prior. Questions where the correct answer is genuinely uncertain even without an image (for example, “what color is the dragon fruit?” where both red and white are valid) are flagged as Low-quality Framing due to an insufficiently strong prior. Second, annotators check whether the question wording contains implicit hints that guide the model toward the anomalous answer. For instance, a question such as “what is unusual about the color of the rose in this image?” implicitly signals that the rose’s color is atypical, nudging the model toward selecting an uncommon option such as black or white. Such items are also categorized as Lowquality Framing.

Stage 2: Automated Probability Analysis. Items that pass Stage 1 (i.e., those with a strong and unambiguous prior and no implicit hints) are further analyzed by re-querying the target model under the Blank condition and extracting the full probability distribution over all answer options via softmax over the output logits. If the variance across option probabilities is low, indicating that the model assigns similar probabilities to all options, the item is categorized as Chance Success. This pattern is particularly prevalent in smaller models with weaker prior knowledge, where correct responses arise from random selection rather than genuine prior activation. In our implementation, if the maximum probability among all options falls below 65%, the item is categorized as Chance Success.

<table><tr><td>Source</td><td>Model</td><td>Question</td><td>Reason for Categorization</td></tr><tr><td>Chance Success</td><td>Qwen3-VL-2B</td><td>When is the vernal equinox? (A. March 20 B. June 20 C. July 23 D. December 21)</td><td>Model assigns near-uniform probabilities: A: 28.3%, B: 24.1%, C: 26.8%, D: 20.8%, indicating no strong prior and random correct selection.</td></tr><tr><td>Low-quality Framing</td><td></td><td>What is unusual about the color of the rose in this image? (A. Red B. Black C. Pink D. White)</td><td>The phrase “unusual” implicitly hints at an atypical answer, guiding the model toward anomalous options.</td></tr><tr><td>Low-quality Framing</td><td></td><td>What color is the watermelon? (A. Red B. Green C. White D. Yellow)</td><td>The question does not specify flesh or rind, making the prior ambiguous and the correct answer uncertain without the image.</td></tr><tr><td>Anomalous Prior</td><td>Qwen2.5-VL-3B</td><td>What color is the dolphin? (A. Blue B. White C. Pink D. Grey)</td><td>Model assigns high probability to “Pink&quot;, indicating it inherently holds the anomalous answer as its own prior.</td></tr></table>

Table 6: Representative examples of the three sources of no-image correct responses.

Stage 3: Anomalous Prior Identification. If the probability distribution exhibits high variance, with one option receiving substantially higher probability than the others, the model is judged to hold a strong prior for this question. Since the model answers correctly under the Blank condition despite the item being designed to conflict with commonsense expectations, this indicates that the model inherently treats the anomalous answer as its own prior. Such items are categorized as Anomalous Prior.

## A.2 Typical Cases

Table 6 provides representative examples of each source type identified by the above pipeline.

## B Visual Style Examples

ViSiP-Bench incorporates images across four distinct visual styles to ensure that model performance is not confounded by image style or source. Table 7 lists the generation prompts used for each style, and Figure 7 presents one representative example from each style.

<table><tr><td>Style</td><td>Generation Prompt</td></tr><tr><td>Photorealistic</td><td>A police officer in full uniform stands at the front of a classroom, sharing career experiences with attentive students seated at desks.</td></tr><tr><td>Anime</td><td>A color-blind boy proudly holds up his drawing of a figure with green skin, smiling with satisfaction, in a vibrant anime art style.</td></tr><tr><td>Oil painting</td><td>A young boy stares in disbelief at a perfectly square watermelon sitting on a wooden table, his mouth wide open in shock, rendered in the style of a classical oil painting.</td></tr><tr><td>Sci-fi</td><td>A boy uses his smartphone to crack open a walnut, pressing the screen edge against the shell with focused determination, in a futuristic sci-fi setting with neon lighting.</td></tr></table>

Table 7: Generation prompts for each visual style in ViSiP-Bench.

## C Cross-Model and Cross-Family Validation of MAPS

To verify that the effectiveness of MAPS is not specific to a single backbone, we apply MAPS to six additional models spanning three architecture families: InternVL3.5 (2B, 8B), InternVL3 (14B), Qwen2.5-VL (7B), Qwen3.5 (9B), and LLaVA-OneVision (7B). All models are fine-tuned under an identical configuration (prior probing threshold α = 0.90, LoRA fine-tuning for 2 epochs), with no per-model hyperparameter adjustment, to verify that the improvement generalizes without modelspecific tuning.

![](images/64dea2084f84361cf0dc7bc16c9ad2065a727d3548c7b8843eaa1ab83e5399ed.jpg)

(a) Photorealistic: A police officer shares career experiences in a classroom.  
![](images/0300940129258cea2bb6c456f147a2cf4626edd0631b17cbf70970a5539b41a4.jpg)  
(c) Oil painting: A boy stares in disbelief at a square watermelon.

![](images/e61e6c43b723818716a5fdc0f7b5e076f4c91d6e48dc6f2ee85e87c32e57d3b4.jpg)  
(b) Anime: A color-blind boy proudly displays his drawing of a green-skinned figure.

![](images/b3a7d77d07e00bc0e2c28b9b9b34111fbec4d7dd550a3484753d134a86a26ba3.jpg)  
(d) Sci-fi: A boy cracks a walnut using his smartphone.  
Figure 7: Representative examples of the four visual styles in ViSiP-Bench. Each image depicts a rare-but-real scenario designed to conflict with commonsense priors.

Table 8 reports results on ViSiP-Bench. Across all six models, MAPS consistently reduces PAR (by 3.44 to 8.06 points) and improves Filtered Accuracy (by 3.62 to 8.15 points), with no exceptions across model scale (2B–14B) or family. FC increases only marginally (by 2 to 7 samples), in sharp contrast to the collapse observed for Post-hoc in Table 3 (147 → 385), confirming that MAPS softens rather than indiscriminately erodes prior knowledge. VIG improves substantially for every model, in most cases more than doubling, indicating that the effect is not confined to a single backbone but reflects a general property of the modelaware prior softening procedure. We note that the absolute PAR reduction for LLaVA-OneVision-7B (-4.09) is somewhat smaller than for other models; this model also exhibits the highest base PAR (67.10) among the six, suggesting that models with a very large number of strongly held priors may require a longer fine-tuning schedule or a larger rare-but-real corpus to achieve proportionally simi-

lar gains.

Table 9 reports generalization results on the same six models. The largest absolute change across all 24 model–benchmark pairs is 0.87 points (MM-Star, InternVL3.5-2B), and changes are not consistently signed: some models show small gains on certain benchmarks and small losses on others, with no systematic degradation pattern. This indicates that the preservation of general reasoning and visual understanding capabilities observed for Qwen3-VL-8B-Instruct in Table 4 is not an artifact of that particular model, but a consistent property of MAPS across scales and architecture families.

## D Annotation Guidelines

Task Description. You will be presented with an image depicting a rare-but-real scenario that conflicts with commonsense expectations. Your task is to design a multiple-choice question that requires the respondent to carefully observe the image in order to answer correctly. The correct answer must be visually grounded in the image, and must differ clearly from what one would expect based on commonsense knowledge alone.

What is a Commonsense Prior? A commonsense prior is a default expectation about the world that a person (or model) holds based on general knowledge. For example, most people expect swans to be white. If an image shows a black swan, a respondent relying on their prior rather than the image would incorrectly answer “white”. Questions should be designed so that relying on the prior leads to a wrong answer, while carefully observing the image leads to the correct one.

<table><tr><td>Model</td><td>Type</td><td>PAR↓</td><td>Acc↑</td><td>FC</td><td> $\mathbf { V I G _ { a v g } } \uparrow$ </td><td>VIGmed↑</td></tr><tr><td>InternVL3.5-2B</td><td>base +MAPS</td><td>51.51</td><td>42.94</td><td>182</td><td>4.41</td><td>4.11</td></tr><tr><td></td><td>base</td><td>46.91 47.17</td><td>47.11</td><td>189</td><td>8.91</td><td>9.20</td></tr><tr><td>InternVL3.5-8B</td><td>+MAPS</td><td>39.11</td><td>44.64 51.85</td><td>156 158</td><td>7.13 15.94</td><td>7.48 15.63</td></tr><tr><td>InternVL3-14B</td><td>base +MAPS</td><td>48.40</td><td>41.33</td><td>161</td><td>6.21</td><td>6.33</td></tr><tr><td></td><td></td><td>42.60</td><td>47.91</td><td>166</td><td>13.81</td><td>13.48</td></tr><tr><td>LLaVA-OV-7B</td><td>base +MAPS</td><td>67.10</td><td>25.36</td><td>317</td><td>3.39</td><td>2.48</td></tr><tr><td></td><td></td><td>63.01</td><td>29.81</td><td>319</td><td>5.94</td><td>6.11</td></tr><tr><td>Qwen2.5-VL-7B</td><td>base</td><td>55.35</td><td>34.51</td><td>198</td><td>5.56</td><td>4.93</td></tr><tr><td></td><td>+MAPS</td><td>51.91</td><td>38.13</td><td>201</td><td>10.63</td><td>11.21</td></tr><tr><td>Qwen3.5-9B</td><td>base</td><td>51.24</td><td>39.36</td><td>137</td><td>5.24</td><td>5.24</td></tr><tr><td></td><td>+MAPS</td><td>43.22</td><td>47.51</td><td>141</td><td>12.97</td><td>13.62</td></tr></table>

Table 8: Cross-model results on ViSiP-Bench. MAPS consistently reduces PAR and increases Filtered Accuracy and VIG across all six models spanning three architecture families, while FC remains close to the base model, indicating that prior knowledge is not indiscriminately disrupted.
<table><tr><td>Model</td><td>Type</td><td>MMMU- Pro</td><td>VisuLogic</td><td>MMStar</td><td>RealWorldQA</td></tr><tr><td>InternVL3.5-2B</td><td>base</td><td>34.10</td><td>24.10</td><td>61.18</td><td>63.12</td></tr><tr><td rowspan="2">InternVL3.5-8B</td><td>+MAPS</td><td>33.91</td><td>24.15</td><td>62.05</td><td>62.49</td></tr><tr><td>base</td><td>41.20</td><td>28.60</td><td>65.23</td><td>67.32</td></tr><tr><td></td><td>+MAPS</td><td>41.13</td><td>28.71</td><td>64.79</td><td>67.51</td></tr><tr><td rowspan="2">InternVL3-14B</td><td>base +MAPS</td><td>40.10 40.23</td><td>26.30 26.41</td><td>64.14</td><td>63.98</td></tr><tr><td></td><td></td><td></td><td>64.87</td><td>63.48</td></tr><tr><td>LLaVA-OV-7B</td><td>base +MAPS</td><td>24.10 24.18</td><td>25.30 26.11</td><td>56.13 55.84</td><td>55.30 55.15</td></tr><tr><td rowspan="2">Qwen2.5-VL-7B</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>base +MAPS</td><td>38.30 38.19</td><td>26.00 25.63</td><td>59.14 59.71</td><td>61.38 60.91</td></tr><tr><td rowspan="2">Qwen3.5-9B</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>base +MAPS</td><td>42.10 42.34</td><td>28.61 27.90</td><td>66.51 66.90</td><td>68.31 68.44</td></tr></table>

Table 9: Generalization results on reasoning (MMMU-Pro, VisuLogic) and general vision (MMStar, RealWorldQA) benchmarks across six models. MAPS preserves performance within a narrow margin for every model, with no consistent direction of degradation.

## Question Design Requirements.

(1) Identify the anomalous element in the image and formulate a question that directly targets it.

(2) Phrase the question in a neutral and objective manner. Do not use any wording that hints at the anomalous nature of the image (e.g., avoid phrases such as “what is unusual about...”).

(3) Provide exactly four answer choices. One choice must correspond to the visually correct answer; the remaining three must be plausible distractors grounded in commonsense expectations.

(4) Ensure that no option is obviously implausible, as this would allow the correct answer to be identified by elimination.

Quality Review Criteria. Each sample must satisfy all three of the following criteria:

(1) The image is visually clear and unambiguously supports a single correct answer.

(2) All answer choices are directly relevant to the question and cannot be eliminated by common sense alone.

(3) The question wording contains no hint of anomaly that would suppress prior activation before the respondent views the image.

Common Mistakes. Please avoid the following common errors:

(1) Implicit anomaly hints: Phrasing such as “what is unusual about the color of the rose?” signals that something is atypical, guiding the respondent toward the anomalous answer without looking at the image.

<table><tr><td>Model</td><td>Physical</td><td>Time &amp; Space</td><td>Scientific</td><td>Customs &amp; Culture</td><td>Psychology</td><td>Function</td></tr><tr><td colspan="7">Open-source Models</td></tr><tr><td>InternVL3.5-2B</td><td>34.1</td><td>49.2</td><td>69.4</td><td>51.3</td><td>43.2</td><td>60.1</td></tr><tr><td>InternVL3.5-8B</td><td>30.2</td><td>44.1</td><td>63.2</td><td>47.1</td><td>39.4</td><td>55.2</td></tr><tr><td>InternVL3.5-14B</td><td>22.1</td><td>29.8</td><td>49.7</td><td>29.1</td><td>21.3</td><td>39.4</td></tr><tr><td>InternVL3.5-38B</td><td>16.3</td><td>21.4</td><td>38.9</td><td>21.4</td><td>19.7</td><td>21.1</td></tr><tr><td>InternVL3-2B</td><td>40.2</td><td>55.3</td><td>78.4</td><td>58.2</td><td>51.3</td><td>68.2</td></tr><tr><td>InternVL3-14B</td><td>28.4</td><td>41.2</td><td>62.1</td><td>44.3</td><td>37.2</td><td>52.1</td></tr><tr><td>InternVL3-38B</td><td>20.1</td><td>31.4</td><td>52.3</td><td>33.2</td><td>28.4</td><td>41.2</td></tr><tr><td>InternVL3-78B</td><td>18.2</td><td>29.1</td><td>49.8</td><td>31.4</td><td>26.1</td><td>38.9</td></tr><tr><td>Qwen2.5-VL-3B</td><td>41.3</td><td>56.2</td><td>79.1</td><td>59.4</td><td>52.1</td><td>69.3</td></tr><tr><td>Qwen2.5-VL-7B</td><td>33.2</td><td>48.1</td><td>68.3</td><td>50.2</td><td>42.3</td><td>59.1</td></tr><tr><td>Qwen2.5-VL-32B</td><td>24.3</td><td>36.2</td><td>58.4</td><td>38.1</td><td>31.2</td><td>47.3</td></tr><tr><td>Qwen2.5-VL-72B</td><td>19.4</td><td>30.1</td><td>51.2</td><td>32.3</td><td>27.4</td><td>40.1</td></tr><tr><td>Qwen3-VL-2B</td><td>38.2</td><td>52.1</td><td>74.3</td><td>55.1</td><td>48.2</td><td>64.3</td></tr><tr><td>Qwen3-VL-4B</td><td>29.3</td><td>43.2</td><td>65.1</td><td>45.2</td><td>38.3</td><td>54.2</td></tr><tr><td>Qwen3-VL-8B</td><td>21.4</td><td>32.1</td><td>54.2</td><td>34.3</td><td>29.1</td><td>43.4</td></tr><tr><td>Qwen3.5-2B</td><td>34.3</td><td>48.2</td><td>70.1</td><td>51.2</td><td>44.3</td><td>61.2</td></tr><tr><td>Qwen3.5-9B</td><td>28.2</td><td>42.1</td><td>64.3</td><td>44.1</td><td>37.2</td><td>53.3</td></tr><tr><td>Qwen3.5-27B</td><td>20.3</td><td>30.2</td><td>52.1</td><td>32.4</td><td>27.3</td><td>41.3</td></tr><tr><td>LLaVA-OV-0.5B</td><td>48.2</td><td>63.1</td><td>84.3</td><td>65.2</td><td>58.1</td><td>74.2</td></tr><tr><td>LLaVA-OV-7B</td><td>43.1</td><td>58.2</td><td>79.4</td><td>60.3</td><td>53.2</td><td>69.1</td></tr><tr><td>LLaVA-OV-72B</td><td>36.2</td><td>51.3</td><td>73.2</td><td>53.4</td><td>46.3</td><td>62.4</td></tr><tr><td colspan="7">Proprietary Models</td></tr><tr><td>GPT-5.5</td><td>12.1</td><td>11.9</td><td>35.2</td><td>15.4</td><td>14.2</td><td>25.1</td></tr><tr><td>Claude Sonnet 4.6</td><td>13.4</td><td>19.4</td><td>36.1</td><td>18.3</td><td>19.1</td><td>24.3</td></tr><tr><td>Gemini-3-flash</td><td>12.9</td><td>18.1</td><td>38.2</td><td>14.2</td><td>13.4</td><td>27.1</td></tr></table>

Table 10: Per-category PAR results for all evaluated models. Scientific & Mathematic consistently exhibits the highest PAR values, while Physical Properties proves most amenable to visual override.

(2) Eliminable distractors: Including options that are obviously implausible allows the correct answer to be identified by elimination rather than visual observation.

(3) Ambiguous reference: Questions such as “what color is this?” without specifying the target object introduce ambiguity. Always specify the object clearly (e.g., “what color is the watermelon flesh?”).

Disagreement Resolution. If the two reviewers disagree on whether a sample passes review, the case is escalated to a third arbitrator whose decision is final. All samples must reach unanimous agreement before inclusion in the dataset.

## E Per-Category Analysis

Table 10 presents the per-category PAR results for all evaluated models. Scientific & Mathematic consistently exhibits the highest PAR values across model families, while Physical Properties proves most amenable to visual override. This pattern remains largely consistent across models, suggesting that category-level difficulty is an intrinsic property of the prior type rather than a model-specific characteristic.

## F Prompt Template

To ensure a fair comparison between the Blank and Visual conditions, we use an identical prompt template for both. The prompt is designed for imagebased questions, with the instruction to guess based on prior knowledge when no image is visible. This design ensures that any difference in model responses between the two conditions is attributable solely to the presence or absence of the image, rather than any difference in the prompt.

The prompt template used in both conditions is as follows:

In this picture, [QUESTION]

A. [OPTION A]

B. [OPTION B]

C. [OPTION C]

D. [OPTION D]

If you cannot see the image, please make your best guess among all answer choices based on your own judgment.