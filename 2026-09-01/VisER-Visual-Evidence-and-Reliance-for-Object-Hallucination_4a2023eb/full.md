# VisER: Visual Evidence and Reliance for Object Hallucination Detection in LVLMs

Afsaneh Hasanebrahimi, Hanxun Huang, Christopher Leckie, Sarah Erfani

School of Computing and Information Systems

The University of Melbourne

Melbourne, Australia

{a.hasanebrahimi,hanxun.huang,caleckie,sarah.erfani}@unimelb.edu.au

## Abstract

Object hallucination remains a persistent reliability issue in large vision-language models, where generated object mentions may sound plausible but lack visual grounding. Recent training-free detectors use internal signals such as token likelihood, attention, visual confidence, or image-text similarity to identify hallucinated objects. These signals are useful, but they are often source-confounded. They measure how strongly an object is supported inside the model without distinguishing whether that support comes from object-specific visual evidence or the generated text prefix. In difficult cases, a hallucinated object can still receive high internal support because it fits the scene, is associated with nearby visual cues, or follows naturally from the generated text prefix. We propose VisER, a training-free two-sided metric for object-level hallucination detection. VisER evaluates each generated object mention from two complementary views. Visual Evidence measures whether object-context compatibility is backed by object-specific evidence from image tokens. Visual Reliance measures whether the object is supported more by the image than by the generated prefix. Combining these views gives a more source-aware grounding score, while avoiding additional objectlevel verification generations. Across multiple LVLMs and benchmarks, VisER improves AUROC and AUPR over a range of baselines.

## 1 Introduction

Large Vision-Language Models (LVLMs) have made substantial progress in connecting visual perception with natural language generation, enabling strong performance across image captioning, visual question answering, and multimodal instruction following (Liu et al., 2023; Zhu et al., 2024; Dai et al., 2023; Chen et al., 2023; Zhu et al., 2025). Despite these advances, LVLMs remain vulnerable to object hallucination, where the model generates object mentions that are not present in the input image (Rohrbach et al., 2018; Li et al., 2023). This failure mode is particularly concerning because hallucinated objects are often fluent, contextually plausible, and embedded naturally within otherwise correct descriptions. As LVLMs are increasingly used in settings where factual visual grounding is important, detecting object hallucinations has become a key problem for reliable multimodal generation (Bai et al., 2024; Liu et al., 2024b).

Existing object hallucination detectors can be broadly grouped into two categories. The first relies on external supervision, including ground-truth object annotations, reference-based evaluation, or auxiliary judge models (Rohrbach et al., 2018; Li et al., 2023; Jing et al., 2024; Xiao et al., 2025; Sun et al., 2024). While effective for controlled evaluation, these methods are difficult to deploy in open-ended settings where references are unavailable, and auxiliary evaluators may introduce their own biases or errors. The second category uses training-free internal signals available during LVLM inference, such as token probabilities (Zhou et al., 2024), confidence estimates from visual representations (Jiang et al., 2025a), attention patterns (Jiang et al., 2025b; Hoang et al., 2026), and latent image-text similarities (Phukan et al., 2025; Park and Li, 2025). These methods are lightweight and reference-free, but they mainly ask whether an object mention is supported inside the model.

The need for a two-sided view becomes clear in difficult hallucination cases. Under the standard evaluation definition, an object hallucination occurs when a generated object mention is absent from the image (Rohrbach et al., 2018; Li et al., 2023). However, for a detector, the difficult cases are those where the absent object still receives strong internal support. Scene plausibility and object co-occurrence can make an absent object appear compatible with the global visual context or the surrounding generated description (Rohrbach et al., 2018; Li et al., 2023; Zhou et al., 2024; Park and Li, 2025). Autoregressive generation can also reinforce later object mentions through the generated prefix rather than through the image (Zhou et al., 2024; Huang et al., 2024; Liu et al., 2024d; Hoang et al., 2026). Even image-side signals can be misleading when attention or similarity is concentrated on related, background, or non-objectspecific visual tokens (Gong et al., 2024; Jiang et al., 2025b). Thus, high likelihood, attention, confidence, or image-text similarity can indicate visual grounding, but it can also indicate plausibility or association.

![](images/d01f3c0a5fb441ee46838a290e03df3cb94994cd1a7d502ecee4b4348a09285e.jpg)  
Figure 1: Overview of VisER as two-sided visual source validation. A generated object mention can receive high internal support either because it is visually grounded or because it is falsely supported by scene priors, misleading visual cues, or autoregressive text-prefix continuation. VisER validates the source of this support using two complementary signals: Visual Evidence VE(o), which checks whether object-context compatibility is backed by object-specific image-token evidence, and Visual Reliance VR(o), which compares image-derived support with generated-prefix support. The final score favors objects that are both visually evidenced and image-driven.

We refer to this ambiguity as source confounding; an internal signal is source-confounded when it assigns high support to an object mention without distinguishing the source of that support. For a grounded object, high support should come from object-specific visual evidence. For a hallucinated object, however, high support may instead arise because the object is scene-plausible, associated with related visual cues, or naturally continued from the generated text prefix. In such cases, the detector measures the strength of support but not whether that support is visually grounded. This makes source confounding a central failure mode for training-free hallucination detection, and a reliable detector should not only ask whether an object is internally supported, but also whether the support is object-specific and image-derived. Appendix E.2 shows representative examples in which contextually plausible but absent objects are scored as grounded by baseline signals, whereas VisER correctly flags them as hallucinated.

Motivated by this observation, we propose VisER (Visual Evidence and Reliance), a trainingfree metric for object-level hallucination detection. As illustrated in Figure 1, VisER evaluates each generated object mention from two complementary sides. Visual Evidence measures whether objectcontext compatibility is backed by object-specific evidence from image tokens. Visual Reliance measures whether the object is supported more by the image than by the generated text prefix. Combining these two sides yields a more source-aware grounding score without model training or additional object-level verification generations. Unlike similarity-based detectors that mainly measure compatibility between an object and the image-text context (Park and Li, 2025), VisER further asks whether that compatibility is supported by objectspecific visual evidence and is not primarily driven by the prefix.

Our contributions are summarized as follows:

• We revisit training-free object hallucination detection through source confounding, where high internal support may reflect visual evidence, scene association, or generated-prefix cues.

• We propose VisER, a training-free metric that combines Visual Evidence and Visual Reliance to score object mentions based on both object-specific evidence and image-versusprefix support.

• We evaluate VisER across multiple LVLMs and benchmarks, showing consistent gains over existing baselines and complementary benefits from both components.

## 2 Related Work

Object hallucination evaluation and external verification. Object hallucination refers to a visual faithfulness error in which an LVLM mentions objects that are absent from, or unsupported by, the input image (Liu et al., 2024b). Early evaluation protocols compare generated object mentions with annotated visual references, including ground-truth object labels, captions, or object-presence annotations (Rohrbach et al., 2018; Li et al., 2023; Petryk et al., 2024; He et al., 2025). Other work uses stronger LLMs or LVLMs as external judges to assess response faithfulness (Liu et al., 2024a; Jing et al., 2024; Wang et al., 2023; Jiang et al., 2024; Cao et al., 2024). These methods are effective for controlled evaluation, but they often require annotations, auxiliary models, or multi-stage verification, making them less suitable when object-level references are unavailable.

Training-free internal-signal detectors. To reduce dependence on external references, recent work has used signals available during LVLM inference. Likelihood-based methods such as LURE use the negative log-likelihood of generated object tokens as a hallucination cue (Zhou et al., 2024). Representation-based methods estimate grounding from internal activations. Internal Confidence uses logit-lens probabilities over image hidden states (Jiang et al., 2025a), while Contextual Lens studies contextual embeddings beyond raw logit-lens evidence (Phukan et al., 2025). Attentionbased methods examine how object generation relies on image or text tokens. SVAR measures attention to image tokens in intermediate layers, while PAS uses attention to preliminary or generated text tokens to detect prefix-driven hallucinations (Jiang et al., 2025b; Hoang et al., 2026). GLSim improves object-level detection by combining global and local image-text similarity (Park and Li, 2025).

Confounded support in LVLM hallucination. Several studies show that hallucinated objects can receive support from sources other than direct object evidence. Object co-occurrence, instruction frequency, and scene priors can make absent objects appear plausible (Li et al., 2023; Zhou et al., 2024). Mitigation work also suggests that reducing statistical bias or unimodal priors can reduce hallucination, indicating that plausible but unsupported mentions are often prior-driven (Leng et al., 2024). Language-side effects provide another source of support. LVLMs may show text inertia, producing similar descriptions even when visual evidence is weak (Liu et al., 2024d), and generated prefix tokens can influence later object mentions (Hoang et al., 2026). Image-side signals can also be imperfect. Attention may vary across layers or heads (Jain and Wallace, 2019; Jiang et al., 2025b), be affected by visual attention sinks (Kang et al., 2025), or concentrate on background and outlier image tokens rather than object-specific regions (Gong et al., 2024; Che et al., 2025). These findings motivate a two-sided detector that considers both whether an object has image-token evidence and whether its support is more image-driven than prefix-driven.

## 3 Method

## 3.1 Problem Setup

We consider a pretrained LVLM that takes an image I and a textual instruction x, and autoregressively generates a response $y = ( y _ { 1 } , \dots , y _ { T } )$ . We study object-level hallucination detection. Let o denote an object mention in the generated response, and let $t _ { o }$ denote the position of the generated token matched to o. The goal is to assign each object mention a grounding score $s ( o , I , x , y _ { < t _ { o } } ) \in \mathbb { R }$ where larger values indicate stronger visual grounding. A binary detector can be obtained by thresholding

$$
G ( o , I , x , y _ { < t _ { o } } ) = \mathbb { 1 } \{ s ( o , I , x , y _ { < t _ { o } } ) \geq \delta \} ,
$$

where $\delta$ is a decision threshold. In our experiments, we evaluate ranking quality using AUROC and AUPR, so no fixed threshold is required.

The LVLM represents the image as a sequence of visual tokens $\mathcal { V } = \{ v _ { 1 } , \ldots , v _ { N } \}$ , which are projected into the language-model space. Let $h _ { i } ^ { \ell } \in \mathbb { R } ^ { d }$ denote the hidden representation of token i at layer ℓ. We use $\ell _ { I }$ for image-token representations and $\ell _ { T }$ for text-token representations. The object-token representation is denoted by $h _ { o } ^ { \ell _ { T } } = h _ { t _ { o } } ^ { \ell _ { T } }$ . We write sim(·, ·) for cosine similarity and $\mathrm { s i m } ^ { + } ( a , b )$ for its [0, 1]-rescaled form.

Visual logit-lens evidence. To obtain objectspecific evidence from image tokens, we apply logit lens (nostalgebraist, 2020) to image-token hidden states. Let $\bar { W } _ { U } \in \mathbb { R } ^ { d \times | \Omega | }$ be the language-model unembedding matrix, where Ω is the vocabulary. For each image token $v _ { i }$ , we compute $z _ { i } = h _ { v _ { i } } ^ { \ell _ { I } } W _ { U }$ The visual logit-lens evidence for object o at image token $v _ { i }$ is $p _ { i } ( o ) = \mathrm { s o f t m a x } ( z _ { i } ) _ { c }$ <sub>o</sub>. This gives an object-specific score for each image token.

## 3.2 VisER

VisER assigns each generated object mention a grounding score from two complementary signals. Visual Evidence (VE) measures object-specific image-token evidence, while Visual Reliance (VR) compares image-derived support with generatedprefix support. Their combination favors objects that are both visually evidenced and more imagedriven than prefix-driven.

Visual Evidence (VE). Visual Evidence measures whether object-context compatibility is supported by object-specific evidence from image tokens. Let $h _ { \mathrm { c t x } } ^ { \ell _ { T } }$ denote the hidden representation of the final prompt token before response generation. We first aggregate the visual logit-lens evidence for object o over all image tokens:

$$
M _ { \mathrm { v i s } } ( o ) = \sum _ { i = 1 } ^ { N } p _ { i } ( o ) ,
$$

where $p _ { i } ( o )$ is the logit-lens probability assigned to object token o at image token $v _ { i }$ . We then compute an evidence gate

$$
g ( o ) = \sigma \left( { \frac { M _ { \mathrm { v i s } } ( o ) } { \tau + \epsilon } } \right) ,
$$

where $\sigma ( \cdot )$ is the sigmoid function, $\tau$ is an evidence-scale parameter, and ϵ is used for numerical stability. We estimate $\tau$ from an unlabeled calibration split disjoint from the evaluation set, using the average visual evidence mass. The Visual Evidence score is

$$
V E ( o ) = \mathrm { s i m } ( h _ { \mathrm { c t x } } ^ { \ell _ { T } } , h _ { o } ^ { \ell _ { T } } ) \cdot g ( o ) .
$$

This score keeps the object-context compatibility term, but modulates it by object-specific imagetoken evidence. As a result, weak object-specific image-token evidence reduces the contribution of object-context compatibility to the final VE score.

Visual Reliance (VR). Visual Reliance compares image-derived support with support from the generated prefix. This is useful because an object mention may be reinforced by previous tokens even when the image provides limited evidence. We first compute image-derived support as

$$
S _ { \mathrm { i m g } } ( o ) = \sum _ { i = 1 } ^ { N } \hat { p } _ { i } ( o ) \sin ^ { + } ( h _ { v _ { i } } ^ { \ell _ { I } } , h _ { o } ^ { \ell _ { T } } ) ,
$$

where the normalized visual evidence weight is

$$
\hat { p } _ { i } ( o ) = \frac { p _ { i } ( o ) } { \sum _ { j = 1 } ^ { N } p _ { j } ( o ) + \epsilon } .
$$

Thus, all image tokens contribute to $S _ { \mathrm { i m g } } ( o )$ , with larger weight assigned to tokens that provide stronger object-specific evidence.

For prefix support, let $Y _ { < t _ { o } } = \{ y _ { 1 } , \dots , y _ { t _ { o } - 1 } \}$ be the generated prefix before object o. We use input embeddings as a lightweight estimate of lexicalsemantic support from the prefix. Let $e _ { o }$ be the input embedding of the object token and $e _ { j }$ the input embedding of prefix token $y _ { j }$ . We select the $K _ { o } = \operatorname* { m i n } ( K , t _ { o } - 1 )$ prefix tokens most similar to $\mathit { e _ { o } } ,$ where K is the maximum number of selected prefix tokens and $K _ { o }$ is the actual number of prefix tokens selected for object $o .$ If $K _ { o } = 0$ , we set $S _ { \mathrm { t e x t } } ( o ) = 0$ . Otherwise,

$$
S _ { \mathrm { t e x t } } ( o ) = \frac { 1 } { K _ { o } } \sum _ { j \in \mathrm { T o p K } ( Y < t _ { o } ; o ) } \mathrm { s i m } ^ { + } ( e _ { j } , e _ { o } ) ,
$$

where $\mathrm { T o p K } ( Y _ { < t _ { o } } ; o )$ returns the selected prefixtoken indices. We define Visual Reliance as

$$
V R ( o ) = \frac { S _ { \mathrm { i m g } } ( o ) } { S _ { \mathrm { i m g } } ( o ) + S _ { \mathrm { t e x t } } ( o ) + \epsilon } .
$$

A larger $V R ( o )$ indicates that support for the object is more image-derived than prefix-derived.

Final VisER Score. VE and VR capture complementary aspects of object grounding. $V E ( o )$ downweights object-context compatibility when objectspecific image-token evidence is weak. $V R ( o )$ compares image-derived support with generatedprefix support, reducing the effect of objects that are more strongly supported by the prefix than by the image. We combine the two scores as

$$
s _ { \mathrm { V i s E R } } ( o ) = \alpha V E ( o ) + ( 1 - \alpha ) V R ( o ) ,
$$

![](images/552c9b859eb742b00da435ff2b98c1c53b59715f1fef651913fcff386bacf48d.jpg)  
Figure 2: Complementary roles of Visual Evidence (VE) and Visual Reliance (VR) in detecting false-support hallucinations. Left: “toothbrushes” is plausible in the bathroom scene and receives relatively high compatibility, but low VR indicates that the mention is not primarily image-driven. Right: “ski” receives high VR from image-side cues such as poles and motion, but low VE reveals weak ski-specific evidence.

where $\alpha \in [ 0 , 1 ]$ controls the trade-off between visual evidence and visual reliance. Larger values of $s _ { \mathrm { V i s E R } } ( o )$ indicate stronger visual grounding.

Why VE and VR are Complementary. Figure 2 shows why both components are necessary. In the bathroom example, the hallucinated object “toothbrush” is scene-compatible and can receive high similarity or compatibility because bathrooms, sinks, and toothbrushes commonly co-occur. A similarity-based detector can therefore over-score it; VE alone may also remain high if related visual cues provide weak object-like evidence. However, VR reveals that the mention is not primarily imagedriven. In the skateboard example, the hallucinated object $^ { 6 6 } \mathrm { { s k i } } ^ { , , }$ is supported by image-side cues such as poles and body motion, so an image-versus-text ratio can be high. However, VE remains low because the image lacks object-specific evidence for an actual ski. These cases show that neither visual support nor visual reliance alone is sufficient; reliable grounding requires their combination.

## 3.3 A Bayesian Source-Confounding View

VisER is motivated by the fact that imageconditioned support for an object does not necessarily imply visual grounding. For an object mention o generated at position $t _ { o } ,$ Bayes’ rule gives

$$
\begin{array} { r l } & { \log p _ { \theta } ( o \mid I , x , y _ { < t _ { o } } ) = \underbrace { \log p _ { \theta } ( o \mid x , y _ { < t _ { o } } ) } _ { \mathrm { p r e f i x - s i d e ~ s u p p o r t } } } \\ & { \phantom { ~ \ } + \underbrace { \log \frac { p _ { \theta } ( I \mid o , x , y _ { < t _ { o } } ) } { p _ { \theta } ( I \mid x , y _ { < t _ { o } } ) } } _ { \mathrm { i m a g e - s i d e ~ c o m p a t i b i l i t y ~ g a i n } } . } \end{array}
$$

We use this decomposition as an analytical lens; VisER does not estimate probabilities directly. The decomposition exposes two sources of confounding in object hallucination. First, the prefix-side term can make an object likely because it follows naturally from the generated text. Second, the imageside compatibility can be high because an object is scene-plausible, or is associated with visual cues.

Prefix-side confounding. Visual Reliance addresses prefix-side confounding by asking whether the object receives additional support from the image beyond what the prefix already explains. This mirrors the conditional pointwise mutualinformation contrast

$$
\begin{array} { r } { \operatorname { P M I } ( I ; o \mid x , y _ { < t _ { o } } ) = } \\ { \log p _ { \theta } ( o \mid I , x , y _ { < t _ { o } } ) - \log p _ { \theta } ( o \mid x , y _ { < t _ { o } } ) , } \end{array}
$$

which isolates the gain from conditioning on the image. This contrast suggests that a useful trainingfree detector should compare image-derived support against prefix-derived support, rather than relying on the image-conditioned object probability alone. Instead of estimating this contrast with additional model calls or explicit likelihoods, VisER constructs internal proxies for the competing sources using $S _ { \mathrm { i m g } } ( o )$ for image-derived support and $S _ { \mathrm { t e x t } } ( o )$ for prefix-derived support. Visual Reliance log-odds satisfy

$$
\log \frac { V R ( o ) } { 1 - V R ( o ) } = \log S _ { \mathrm { i m g } } ( o ) - \log ( S _ { \mathrm { t e x t } } ( o ) + \epsilon ) .
$$

Thus, $V R ( o )$ is a bounded monotonic transform of a smoothed image-versus-prefix log-ratio. It does not equal the conditional PMI, but it preserves the same source comparison. The score increases when support is more image-derived than prefixderived, and decreases when the generated prefix better explains the object mention.

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td colspan="2">LLaVA-1.5-7B</td><td colspan="2">LLaVA-NeXT</td><td colspan="2">InstructBLIP</td><td colspan="2">MiniGPT-4</td><td colspan="2">Average</td></tr><tr><td>AUROC↑</td><td>AUPR↑</td><td>AUROC↑</td><td>AUPR↑</td><td>AUROC↑</td><td>AUPR↑</td><td>AUROC↑</td><td>AUPR↑</td><td>AUROC↑</td><td>AUPR↑</td></tr><tr><td rowspan="9">MSCOCO</td><td>Entropy (Malinin and Gales, 2021)</td><td>71.41</td><td>87.66</td><td>58.25</td><td>90.09</td><td>65.81</td><td>84.25</td><td>45.78</td><td>82.44</td><td>60.31</td><td>86.11</td></tr><tr><td>NLL (Zhou et al., 2024)</td><td>70.28</td><td>87.19</td><td>58.99</td><td>90.27</td><td>65.30</td><td>83.92</td><td>57.55</td><td>86.18</td><td>63.03</td><td>86.89</td></tr><tr><td>Internal Conf. (Jiang et al., 2025a)</td><td>73.88</td><td>90.62</td><td>78.00</td><td>95.83</td><td>84.59</td><td>93.57</td><td>75.79</td><td>93.50</td><td>78.07</td><td>93.38</td></tr><tr><td>SVAR (Jiang et al., 2025b)</td><td>74.39</td><td>92.20</td><td>74.02</td><td>95.45</td><td>78.19</td><td>91.95</td><td>85.94</td><td>96.63</td><td>78.14</td><td>94.06</td></tr><tr><td>Contextual Lens (Phukan et al., 2025)</td><td>70.93</td><td>89.03</td><td>59.15</td><td>90.04</td><td>81.17</td><td>91.94</td><td>82.55</td><td>94.43</td><td>73.45</td><td>91.36</td></tr><tr><td>PAS (Hoang et al., 2026)</td><td>82.52</td><td>94.81</td><td>73.71</td><td>95.58</td><td>81.96</td><td>91.58</td><td>80.96</td><td>95.16</td><td>79.79</td><td>94.28</td></tr><tr><td>GLSim (Park and Li, 2025)</td><td>84.09</td><td>94.45</td><td>77.43</td><td>95.65</td><td>82.86</td><td>93.30</td><td>80.66</td><td>94.64</td><td>81.26</td><td>94.51</td></tr><tr><td>VisER (Ours)</td><td>86.17</td><td>95.37</td><td>81.44</td><td>96.74</td><td>85.66</td><td>94.84</td><td>86.29</td><td>96.35</td><td>84.89</td><td>95.82</td></tr><tr><td rowspan="7"></td><td>Entropy (Malinin and Gales, 2021)</td><td>53.12</td><td>74.34</td><td>62.39</td><td>91.07</td><td>60.57</td><td>76.11</td><td>53.48</td><td>81.33</td><td>57.39</td><td>80.71</td></tr><tr><td>NLL (Zhou et al., 2024)</td><td>52.56</td><td>74.24</td><td>62.57</td><td>91.49</td><td>60.44</td><td>76.62</td><td>57.97</td><td>83.47</td><td>58.39</td><td>81.46</td></tr><tr><td>Internal Conf. (Jiang et al., 2025a)</td><td>71.05</td><td>82.74</td><td>77.50</td><td>95.47</td><td>82.96</td><td>89.96</td><td>68.58</td><td>86.15</td><td>75.02</td><td>88.58</td></tr><tr><td>SVAR (Jiang et al., 2025b)</td><td>74.59</td><td>88.75</td><td>73.84</td><td>95.11</td><td>75.86</td><td>87.51</td><td>82.62</td><td>94.74</td><td>76.73</td><td>91.53</td></tr><tr><td>Contextual Lens (Phukan et al., 2025)</td><td>70.73</td><td>84.82</td><td>58.87</td><td>90.12</td><td>80.62</td><td>88.87</td><td>74.69</td><td>88.80</td><td>71.23</td><td>88.15</td></tr><tr><td>PAS (Hoang et al., 2026)</td><td>80.31 80.27</td><td>91.36 90.58</td><td>74.96 79.17</td><td>95.66 95.88</td><td>79.30 84.89</td><td>86.85 92.49</td><td>78.20 78.72</td><td>92.80 93.41</td><td>78.19 80.76</td><td>91.67 93.09</td></tr><tr><td>GLSim (Park and Li, 2025) VisER (Ours)</td><td>82.20</td><td>91.26</td><td>87.31</td><td>98.08</td><td>85.72</td><td>93.24</td><td>80.34</td><td>92.93</td><td>83.89</td><td>93.88</td></tr></table>

Table 1: Main object-level hallucination detection results on MSCOCO and Pascal VOC. AUROC and AUPR are reported for each LVLM setting, with averages computed across the four model settings.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td colspan="2">LLaVA-1.5-13B</td><td colspan="2">InternVL3</td><td colspan="2">Shikra</td><td colspan="2">Qwen2.5-VL</td><td colspan="2">Average</td></tr><tr><td>AUROC↑</td><td>AUPR↑</td><td>AUROC↑</td><td>AUPR↑</td><td>AUROC↑</td><td>AUPR↑</td><td>AUROC↑</td><td>AUPR↑</td><td>AUROC↑</td><td>AUPR↑</td></tr><tr><td rowspan="9">MSCOCO</td><td>Entropy</td><td>70.99</td><td>88.85</td><td>64.38</td><td>95.33</td><td>69.20</td><td>87.62</td><td>68.10</td><td>95.85</td><td>68.17</td><td>91.91</td></tr><tr><td>NLL</td><td>69.11</td><td>88.10</td><td>64.30</td><td>95.31</td><td>66.66</td><td>86.93</td><td>65.53</td><td>95.50</td><td>66.40</td><td>91.46</td></tr><tr><td>Internal Conf.</td><td>70.30</td><td>89.41</td><td>66.63</td><td>95.56</td><td>68.53</td><td>89.23</td><td>64.80</td><td>95.20</td><td>67.56</td><td>92.35</td></tr><tr><td>SVAR</td><td>72.83</td><td>92.63</td><td>74.02</td><td>96.98</td><td>59.73</td><td>84.76</td><td>73.76</td><td>96.76</td><td>70.08</td><td>92.78</td></tr><tr><td>Contextual Lens</td><td>76.39</td><td>92.31</td><td>77.18</td><td>97.15</td><td>78.72</td><td>91.80</td><td>71.25</td><td>96.17</td><td>75.88</td><td>94.36</td></tr><tr><td>PAS</td><td>80.79</td><td>94.76</td><td>75.44</td><td>97.41</td><td>79.48</td><td>93.56</td><td>73.95</td><td>97.18</td><td>77.42</td><td>95.73</td></tr><tr><td>GLSim</td><td>81.53</td><td>94.82</td><td>67.23</td><td>95.79</td><td>75.03</td><td>91.74</td><td>69.67</td><td>96.11</td><td>73.36</td><td>94.62</td></tr><tr><td>VisER</td><td>83.82</td><td>95.35</td><td>82.54</td><td>97.78</td><td>80.18</td><td>93.68</td><td>80.10</td><td>97.30</td><td>81.66</td><td>96.03</td></tr></table>

Table 2: Extended MSCOCO results on additional LVLMs. AUROC and AUPR are reported for each model, with averages computed across the four extended settings.

Image-side confounding. Visual Evidence addresses the remaining image-side confounding. The Bayesian image-side gain measures whether the image is compatible with the hypothesis that o belongs in the description. The similarity term sim $( h _ { c t x } ^ { \ell _ { T } } , h _ { o } ^ { \ell _ { T } } )$ captures whether the object is compatible with the multimodal context, while $M _ { \mathrm { v i s } } ( o )$ tests whether image tokens provide direct objectspecific evidence. In this sense, $V E ( o )$ is a specificity-corrected proxy for the image-side term. It retains compatibility when it is visually evidenced, and suppresses scene-level or cue-driven compatibility that can otherwise mimic grounding.

Thus, a generated object receives a high VisER score only when it is both supported by direct image-token evidence and more strongly explained by the image than by the generated prefix.

## 4 Experiments

Models. We evaluate VisER on a diverse set of LVLMs covering different model families and scales. Our main evaluation includes LLaVA-1.5-7B (Liu et al., 2023), LLaVA-NeXT-7B (Liu et al., 2024c), InstructBLIP (Dai et al., 2023), and MiniGPT-4 (Zhu et al., 2024). We include LLaVA-NeXT-7B to evaluate generalization to a more recent LLaVA-family architecture. To further examine robustness across architectures, we also report results on LLaVA-1.5-13B, InternVL3-8B (Zhu et al., 2025), Shikra-7B (Chen et al., 2023), and Qwen2.5-VL (Bai et al., 2025). All models use greedy decoding with a maximum output length of 512 new tokens. Implementation details, including model configurations, layer selection, and hyperparameter settings, are provided in Appendix B. 1

Benchmarks. We evaluate on MSCOCO (Lin et al., 2014), following prior object hallucination detection protocols (Hoang et al., 2026; Jiang et al., 2025b), and additionally report results on Pascal VOC (Everingham et al., 2010). Following (Li et al., 2023; Leng et al., 2024; Jiang et al., 2025a) we randomly sample 500 images from the validation split. MSCOCO and Pascal VOC contain 80 and 20 object categories, respectively, with imagelevel object annotations indicating which categories are present in each image. Following the CHAIR object-level evaluation protocol, we extract generated object mentions and label them as grounded or hallucinated by matching them against the annotated object lists. All LVLMs are prompted with: “Describe the given image in detail.”

Baselines. We compare VisER with several training-free object hallucination detection baselines. These include logit-based uncertainty signals, namely Negative Log-Likelihood (NLL) (Zhou et al., 2024) and Entropy (Malinin and Gales, 2021); attention-based methods, including Summed Visual Attention Ratio (SVAR) (Jiang et al., 2025b) and Prelim Attention Score (PAS) (Hoang et al., 2026); and representation-based methods, including Internal Confidence (IC) (Jiang et al., 2025a) and Global-Local Similarity (GLSim) (Park and Li, 2025). All baselines are evaluated on the same generated captions, object mentions, and image samples for each benchmark to ensure a fair comparison. The baseline details can be found in Appendix A. We report object-level AUROC and AUPR, using grounded object mentions as the positive class. For parameterized baselines, we adopt the hyperparameter settings specified by the original methods when available.

## 4.1 Results

Main results. Table 1 reports object-level hallucination detection results on MSCOCO and Pascal VOC across four LVLM settings. VisER achieves the strongest average performance on both datasets. On MSCOCO, VisER obtains the best AUROC for all four models, with an average AUROC of 84.89, improving over the strongest average baseline, GLSim, by 3.63 points. It also achieves the highest average AUPR of 95.82. On Pascal VOC, VisER again achieves the best average performance, reaching 83.89 AUROC and 93.88 AUPR.

The gains are especially pronounced on LLaVA-NeXT and InstructBLIP. On MSCOCO, VisER improves LLaVA-NeXT AUROC from the strongest baseline score of 78.00 to 81.44. On Pascal VOC, it improves over the strongest baseline by 8.14 points on LLaVA-NeXT and 0.83 points on InstructBLIP. This shows that hallucinated objects can receive strong internal support from scene priors or autoregressive prefix cues, and that combining objectspecific visual evidence with image-versus-prefix reliance improves detection.

<table><tr><td>Model</td><td>Method</td><td>Time</td><td>ACC</td><td>PR(T) PR(F)</td><td>F1</td><td>F1(F)</td></tr><tr><td rowspan="2">7B</td><td>POPE</td><td>3.19</td><td>67.77</td><td>61.28 91.86</td><td>74.97</td><td>54.75</td></tr><tr><td>VisER</td><td>2.36</td><td>78.67</td><td>80.93 76.72</td><td>77.86</td><td>79.42</td></tr><tr><td>13B</td><td>POPE VisER</td><td>4.50 3.27</td><td>60.53 76.56</td><td>56.05 76.36 76.78</td><td>90.72 71.20 76.65</td><td>37.29 76.47</td></tr></table>

Table 3: Balanced object-level comparison with POPE. Time denotes object verification overhead.

<table><tr><td>Method</td><td>Metric</td><td>Peak VRAM (GB)</td></tr><tr><td>Entropy/NLL</td><td>Token Probability</td><td>13.33</td></tr><tr><td>Internal Conf.</td><td>Token Probability</td><td>15.77</td></tr><tr><td>GLSim</td><td>Embedding-similarity</td><td>13.54</td></tr><tr><td>SVAR/PAS</td><td>Attention Allocation</td><td>14.31</td></tr><tr><td>VisER</td><td>Embedding-based</td><td>13.33</td></tr></table>

Table 4: Post-caption detector-stage profiling on LLaVA-1.5-7B in float16 mode with batch size 1.

Extended models. Table 2 reports additional MSCOCO results on LLaVA-1.5-13B, InternVL3, Shikra-7B, and Qwen2.5-VL. VisER achieves the best average AUROC of 81.66, improving over the strongest average baseline, PAS, by 4.24 points. The gains are largest on Qwen2.5-VL and InternVL3, where VisER improves AUROC over the strongest baselines by 6.15 and 5.36 points, respectively. VisER also achieves the best results on LLaVA-1.5-13B and Shikra-7B, showing that the two-sided grounding view generalizes across model scales and architectures.

## 4.2 Comparison with POPE

We further compare VisER with POPE-style prompting-based object verification (Li et al., 2023). Given the objects extracted from each generated caption, POPE verifies each object with the question: Q: Is there a {object} in the image? For POPE, we use yes/no parsing, and an object is predicted as present only if the response starts with an affirmative token such as “yes”.

For this POPE comparison, we use a larger 3000- image MSCOCO subset to obtain a sufficient number of hallucinated object mentions. Because generated object sets are naturally imbalanced, we construct balanced object-level subsets by retaining all hallucinated objects and randomly sampling an equal number of grounded objects. This yields 2495 grounded and 2495 hallucinated objects for LLaVA-1.5-7B and 2216 grounded and 2216 hallucinated for LLaVA-13B. We repeat the sampling procedure 1000 times and report the average.

![](images/297d8d5848ff85790ed4f4b5868cd9943be169248463aaa91636489b848c48ca.jpg)  
(a)

![](images/fff849be9e516573453345412d56ce491176974f03f19c783466db08037889b7.jpg)  
(b)

![](images/1544c98eee69adcb3faa615a2e33182acca50094432a21bf503455d9e08e2a37.jpg)  
(c)  
Figure 3: Sensitivity analysis of VisER. Left: effect of the number of selected prefix tokens K. Middle: effect of the interpolation weight α. Right: effect of the image layer when the text layer is fixed.

As shown in Table 3, VisER improves ACC from 67.77 to 78.67 and hallucinated-object F1 from 54.75 to 79.42, while reducing verification time from 3.19s to 2.36s. A similar pattern is observed for LLaVA-13B, indicating that these gains also hold at a larger model scale. POPE achieves high precision for grounded objects, but its low F1(F) indicates a bias toward predicting object presence. In contrast, VisER provides stronger hallucination-sensitive discrimination without requiring additional yes/no generations.

Efficiency. Finally, Table 4 reports the postcaption detector-stage peak GPU memory on LLaVA-1.5-7B. VisER matches the lightweight Entropy/NLL baseline in peak VRAM, while requiring less memory than Internal Confidence, GLSim, and attention-based scoring. This indicates that VisER can be implemented efficiently by retaining only selected hidden representations and computing logit-lens evidence in chunks. All VRAM measurements are reported on a single NVIDIA A100 GPU using the same inference setting.

## 4.3 Ablation Study

Sensitivity Analysis. Figure 3 studies the sensitivity of VisER to K, α, and the image layer on LLaVA-1.5-7B using the MSCOCO dataset. Figure 3(a) shows that VisER remains above the strongest baselines, PAS and GLSim, across the full range of K. Performance improves from 84.85 AU-ROC at K = 2 to 85.66 at $K = 1 0$ , and then saturates, indicating that a small set of highly related prefix tokens is sufficient to estimate text-prefix support. Figure 3(b) shows that both components contribute to performance. $V R ( o )$ alone obtains 78.22 AUROC and $V E ( o )$ alone reaches 83.88, while their combination performs best, peaking at

![](images/54055315147c5b02f51927da1be797a5202d3f0e473cb70966e000668bdd6843.jpg)  
Figure 4: Effect of evidence gating in Visual Evidence. “W/O Gating” uses only object-context compatibility, while “W Gating” multiplies compatibility by the evidence gate g(o).

85.66 around $\alpha = 0 . 4 .$ . This confirms that visual evidence and visual reliance capture complementary hallucination cues. Figure 3(c) shows that later image layers provide stronger evidence, with the best performance at layer 32, suggesting that semantically richer image-token representations improve object-level grounding signals.

Effect of Evidence Gating. Figure 4 isolates the role of the evidence gate g(o) in Visual Evidence across LVLMs. Without the gate, VE relies only on object-context compatibility and can over-score hallucinations that are plausible from scene context alone. Gating filters compatibility through objectspecific image-token evidence, yielding consistent AUROC gains. This supports our central claim that compatibility is a useful but incomplete grounding cue unless its visual source is validated.

Counterfactual Validation of Source Attribution. To further examine whether VE and VR capture the intended sources of support, we perform three counterfactual interventions on LLaVA-1.5-7B and LLaVA-1.5-13B: blank-image replacement, patch shuffling, and prefix removal. Table 5 reports the corresponding AUROC values.

As shown in Table 5, replacing the image with a blank input or shuffling its patches reduces VE and the full VisER score on both model scales. This suggests that these signals depend on coherent visual evidence rather than only on the presence of image tokens. In contrast, removing prefix support leaves VE unchanged while reducing VR to chance-level discrimination. These results provide additional empirical support for interpreting VE as visual-evidence-sensitive and VR as measuring the balance between image- and prefix-derived support, without claiming calibrated causal attribution.

<table><tr><td>Model</td><td>Condition</td><td>VE</td><td>VR</td><td>VisER</td></tr><tr><td>LLaVA-1.5-7B</td><td>Original Blank image Patch-shuffled Prefix removed</td><td>83.88 67.74 74.34 83.88</td><td>78.22 70.86 74.44 50.00</td><td>86.17 71.38 77.99 83.88</td></tr><tr><td>LLaVA-1.5-13B</td><td>Original Blank image Patch-shuffled Prefix removed</td><td>80.01 55.58 69.60 80.01</td><td>77.78 68.34 73.89 50.00</td><td>83.82 61.89 74.93 80.01</td></tr></table>

Table 5: Counterfactual validation of Visual Evidence and Visual Reliance. Values are AUROC.

![](images/02082a3f686be58a894074e2fb6438f499e4722282e7fab2be4376a354df3222.jpg)  
Figure 5: Component ablation on MSCOCO. Compatibility uses object-context compatibility alone, Gate uses the evidence gate alone, VE combines compatibility with the evidence gate, and VisER further incorporates Visual Reliance.

Component Ablation. To further isolate the contribution of each component, we compare objectcontext compatibility alone, the evidence gate alone, Visual Evidence, and the full VisER score. Figure 5 reports the corresponding AUROC values.

As shown in Figure 5, the evidence gate is informative on its own, but performs better when used to validate object-context compatibility. Combining compatibility with the evidence gate increases AUROC from 72.25 to 83.88 on LLaVA-1.5-7B and from 65.15 to 80.01 on LLaVA-1.5-13B. The full VisER score further improves performance to 86.17 and 83.82 AUROC, respectively. This supports using the gate as an evidence validator and further shows that VE and VR provide complementary information.

## 5 Conclusion

We introduced VisER, a training-free metric for object-level hallucination detection in LVLMs. VisER addresses source confounding by combining Visual Evidence, measuring whether object-context compatibility is supported by object-specific imagetoken evidence, with Visual Reliance, comparing image-derived support against generated-prefix support. The object-level VisER score could also support downstream applications such as selective abstention, caption revision, or future decodingtime intervention. Across multiple LVLMs and datasets, VisER consistently improves AUROC and AUPR over strong baselines. Ablations show that the two signals are complementary, making VisER an effective post-hoc detector for visually unsupported object mentions in open-ended generations.

## Limitations

VisER focuses on object-existence hallucinations in open-ended image descriptions. Our evaluation primarily follows COCO/VOC-style object categories and therefore does not directly assess open-vocabulary hallucinations, attributes, relations, counts, actions, or other fine-grained factual errors. The object-token representation may also be sensitive to multi-token expressions, subword tokenization, and repeated object mentions. VisER requires access to internal LVLM representations, making it more suitable for white-box or gray-box settings than fully black-box APIs. Future work could extend the two-sided grounding view to phrase- and region-level hallucination detection, and explore whether VE- and VR-style signals can guide hallucination mitigation during decoding. VisER is intended as a diagnostic object-level hallucination detector and should not be treated as a guarantee of visual factuality in safety-critical deployments.

## Acknowledgements

This research was supported by The University of Melbourne’s Research Computing Services, the Petascale Campus Initiative, and the Spartan HPC facilities. This Spartan facility was established with the assistance of LIEF Grant LE170100200. Moreover, this research was supported by the ARC Centre of Excellence for Automated Decision-Making and Society (CE200100005), and partially funded by the Australian Government through the Australian Research Council.

## References

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, and 8 others. 2025. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923.

Zechen Bai, Pichao Wang, Tianjun Xiao, Tong He, Zongbo Han, Zheng Zhang, and Mike Zheng Shou. 2024. Hallucination of multimodal large language models: A survey. arXiv preprint arXiv:2404.18930.

Qingxing Cao, Junhao Cheng, Xiaodan Liang, and Liang Lin. 2024. VisDiaHalBench: A visual dialogue benchmark for diagnosing hallucination in large vision-language models. In Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics (ACL) (Volume 1: Long Papers).

Liwei Che, Tony Qingze Liu, Jing Jia, Weiyi Qin, Ruixiang Tang, and Vladimir Pavlovic. 2025. Hallucinatory image tokens: A training-free EAZY approach to detecting and mitigating object hallucinations in LVLMs. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).

Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. 2023. Shikra: Unleashing multimodal LLMs’ referential dialogue magic. arXiv preprint arXiv:2306.15195.

Wenliang Dai, Junnan Li, Dongxu Li, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven C. Hoi. 2023. InstructBLIP: Towards general-purpose vision-language models with instruction tuning. Advances in Neural Information Processing Systems (NeurIPS).

Mark Everingham, Luc Van Gool, Christopher K. I. Williams, John Winn, and Andrew Zisserman. 2010. The PASCAL visual object classes (VOC) challenge. International Journal ofComputer Vision (IJCV).

Xuan Gong, Tianshi Ming, Xinpeng Wang, and Zhihua Wei. 2024. DAMRO: Dive into the attention mechanism of LVLM to reduce object hallucination. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP).

Yixiao He, Haifeng Sun, Pengfei Ren, Jingyu Wang, Huazheng Wang, Qi Qi, Zirui Zhuang, and Jing Wang. 2025. Evaluating and mitigating object hallucination in large vision-language models: Can they still see removed objects? In Proceedings of the Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics (NAACL) (Volume 1: Long Papers).

Nhat Hoang, Minh Vu, My T. Thai, and Manish Bhattarai. 2026. PAS: Prelim attention score for detecting object hallucinations in large vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Qidong Huang, Xiaoyi Dong, Pan Zhang, Bin Wang, Conghui He, Jiaqi Wang, Dahua Lin, Weiming Zhang, and Nenghai Yu. 2024. OPERA: Alleviating hallucination in multi-modal large language models via over-trust penalty and retrospection-allocation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Sarthak Jain and Byron C. Wallace. 2019. Attention is not explanation. In Proceedings of the Conference of the North American Chapter ofthe Associationfor Computational Linguistics (NAACL).

Chaoya Jiang, Hongrui Jia, Wei Ye, Mengfan Dong, Haiyang Xu, Ming Yan, Ji Zhang, and Shikun Zhang. 2024. Hal-Eval: A universal and fine-grained hallucination evaluation framework for large vision language models. In Proceedings of the 32nd ACM International Conference on Multimedia.

Nick Jiang, Anish Kachinthaya, Suzanne Petryk, and Yossi Gandelsman. 2025a. Interpreting and editing vision-language representations to mitigate hallucinations. In International Conference on Learning Representations (ICLR).

Zhangqi Jiang, Junkai Chen, Beier Zhu, Tingjin Luo, Yankun Shen, and Xu Yang. 2025b. Devils in middle layers of large vision-language models: Interpreting, detecting and mitigating object hallucinations via attention lens. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Liqiang Jing, Ruosen Li, Yunmo Chen, and Xinya Du. 2024. FaithScore: Fine-grained evaluations of hallucinations in large vision-language models. In Findings of the Association for Computational Linguistics: EMNLP 2024.

Seil Kang, Jinyeong Kim, Junhyeok Kim, and Seong Jae Hwang. 2025. See what you are told: Visual attention sink in large multimodal models. In International Conference on Learning Representations (ICLR).

Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, and Lidong Bing. 2024. Mitigating object hallucinations in large visionlanguage models through visual contrastive decoding. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Xin Zhao, and Ji-Rong Wen. 2023. Evaluating object hallucination in large vision-language models. In Proceedings ofthe Conference on Empirical Methods in Natural Language Processing (EMNLP).

Tsung-Yi Lin, Michael Maire, Serge J. Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. 2014. Microsoft COCO: Common objects in context. In European Conference on Computer Vision (ECCV).

Fuxiao Liu, Kevin Lin, Linjie Li, Jianfeng Wang, Yaser Yacoob, and Lijuan Wang. 2024a. Mitigating hallucination in large multi-modal models via robust instruction tuning. In International Conference on Learning Representations (ICLR).

Hanchao Liu, Wenyuan Xue, Yifei Chen, Dapeng Chen, Xiutian Zhao, Ke Wang, Liping Hou, Rongjun Li, and Wei Peng. 2024b. A survey on hallucination in large vision-language models. arXiv preprint arXiv:2402.00253.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2024c. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. Advances in Neural Information Processing Systems (NeurIPS).

Shi Liu, Kecheng Zheng, and Wei Chen. 2024d. Paying more attention to image: A training-free method for alleviating hallucination in LVLMs. In European Conference on Computer Vision (ECCV).

Andrey Malinin and Mark Gales. 2021. Uncertainty estimation in autoregressive structured prediction. In International Conference on Learning Representations (ICLR).

nostalgebraist. 2020. Interpreting GPT: The logit lens. Accessed: 2026-05-17.

Seongheon Park and Sharon Li. 2025. GLSim: Detecting object hallucinations in LVLMs via global-local similarity. Advances in Neural Information Processing Systems (NeurIPS).

Suzanne Petryk, David M. Chan, Anish Kachinthaya, Haodi Zou, John Canny, Joseph E. Gonzalez, and Trevor Darrell. 2024. ALOHa: A new measure for hallucination in captioning models. In Proceedings ofthe Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics (NAACL).

Anirudh Phukan, Divyansh, Harshit Kumar Morj, Vaishnavi, Apoorv Saxena, and Koustava Goswami. 2025. Beyond Logit Lens: Contextual embeddings for robust hallucination detection & grounding in VLMs. In Proceedings ofthe Conference ofthe Nations of the Americas Chapter of the Association for Computational Linguistics (NAACL) (Volume 1: Long Papers).

Anna Rohrbach, Lisa Anne Hendricks, Kaylee Burns, Trevor Darrell, and Kate Saenko. 2018. Object hallucination in image captioning. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP).

Zhiqing Sun, Sheng Shen, Shengcao Cao, Haotian Liu, Chunyuan Li, Yikang Shen, Chuang Gan, Liangyan Gui, Yu-Xiong Wang, Yiming Yang, Kurt Keutzer,

and Trevor Darrell. 2024. Aligning large multimodal models with factually augmented RLHF. In Findings of the Association for Computational Linguistics: ACL 2024.

Junyang Wang, Yiyang Zhou, Guohai Xu, Pengcheng Shi, Chenlin Zhao, Haiyang Xu, Qinghao Ye, Ming Yan, Ji Zhang, Jihua Zhu, Jitao Sang, and Haoyu Tang. 2023. Evaluation and analysis of hallucination in large vision-language models. arXiv preprint arXiv:2308.15126.

Wenyi Xiao, Ziwei Huang, Leilei Gan, Wanggui He, Haoyuan Li, Zhelun Yu, Fangxun Shu, Hao Jiang, and Linchao Zhu. 2025. Detecting and mitigating hallucination in large vision language models via finegrained AI feedback. In Proceedings of the AAAI Conference on Artificial Intelligence.

Yiyang Zhou, Chenhang Cui, Jaehong Yoon, Linjun Zhang, Zhun Deng, Chelsea Finn, Mohit Bansal, and Huaxiu Yao. 2024. Analyzing and mitigating object hallucination in large vision-language models. In International Conference on Learning Representations (ICLR).

Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. 2024. MiniGPT-4: Enhancing vision-language understanding with advanced large language models. In International Conference on Learning Representations (ICLR).

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, Zhangwei Gao, Erfei Cui, Xuehui Wang, Yue Cao, Yangzhou Liu, Xingguang Wei, Hongjie Zhang, Haomin Wang, Weiye Xu, and 32 others. 2025. InternVL3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479.

## Appendix

## A Related Work: Baseline Scoring Functions

For all baselines, we compute an object-level score for a generated object mention $o = y _ { t }$ , where t denotes the decoding position of the matched object token. All scores are oriented such that larger values indicate stronger visual grounding and smaller values indicate a higher likelihood of hallucination.

Entropy. The entropy baseline measures the uncertainty of the LVLM next-token distribution at the object generation step. Let

$$
q _ { t } ( y ) = p _ { \theta } ( y \mid I , x , y _ { < t } )
$$

denote the probability assigned to vocabulary token $y \in \Omega$ when generating $y _ { t }$ . The token-level entropy is

$$
H ( o ) = - \sum _ { y \in \Omega } q _ { t } ( y ) \log q _ { t } ( y ) .
$$

Since lower entropy indicates higher model confidence, we use the negative entropy as the grounding-oriented score:

$$
S _ { \mathrm { E n t } } ( o ) = - H ( o ) .
$$

Internal Confidence. Internal Confidence estimates object evidence from visual-token representations using logit-lens probabilities. For an image token $v _ { i }$ at layer ℓ, let

$$
p _ { \ell , i } ( o ) = \mathrm { s o f t m a x } \Big ( h _ { v _ { i } } ^ { \ell } W _ { U } \Big ) _ { o }
$$

denote the probability assigned to the object token o after projecting the visual-token hidden state through the language-model unembedding matrix $W _ { U }$ . The Internal Confidence score is defined as the strongest object evidence over image tokens and layers:

$$
S _ { \mathrm { I C } } ( o ) = \operatorname* { m a x } _ { \ell \in \mathcal { L } } \operatorname* { m a x } _ { i \in \mathcal { T } } p _ { \ell , i } ( o ) ,
$$

where I denotes the set of image-token positions and $\mathcal { L }$ is the set of evaluated layers.

SVAR. SVAR measures how much attention the generated object token assigns to image tokens. Let $A _ { \ell , h } ( t , i )$ denote the attention weight from the generated object position t to image-token position $i ,$ at layer ℓ and head h. We compute the visual attention ratio over the selected middle layers as

$$
S _ { \mathrm { S V A R } } ( o ) = \frac { 1 } { | \mathcal { L } | H } \sum _ { \ell \in \mathcal { L } } \sum _ { h = 1 } ^ { H } \sum _ { i \in \mathcal { I } } A _ { \ell , h } ( t , i ) ,
$$

where H is the number of attention heads. Following the original setting, we use the middle-layer range $\mathcal { L } = \{ 5 , \ldots , 1 8 \}$ . Larger values indicate stronger visual-token attention during object generation.

Contextual Lens. Contextual Lens uses representation-level similarity between the generated object token and visual-token representations. Given the object hidden state $h _ { o } ^ { \ell _ { T } }$ and image-token hidden states $\{ h _ { v _ { i } } ^ { \ell _ { I } } \} _ { i \in \mathcal { I } }$ , we compute

$$
S _ { \mathrm { C L } } ( o ) = \operatorname* { m a x } _ { i \in \mathbb { Z } } \sin \left( h _ { o } ^ { \ell _ { T } } , h _ { v _ { i } } ^ { \ell _ { I } } \right) .
$$

This score captures the strongest visual-token representation that is compatible with the generated object mention.

GLSim. GLSim measures object grounding through global and local image-text compatibility.

$$
S _ { \mathrm { G L S i m } } ( o ) = \beta S _ { \mathrm { g l o b a l } } ( o ) + ( 1 - \beta ) S _ { \mathrm { l o c a l } } ( o ) .
$$

The global component measures compatibility between the generated object representation and the multimodal context representation:

$$
S _ { \mathrm { g l o b a l } } ( o ) = \sin \left( h _ { \mathrm { c t x } } ^ { \ell _ { T } } , h _ { o } ^ { \ell _ { T } } \right) ,
$$

For the local component, it first selects the K image tokens with the highest logit-lens evidence for the object:

$$
\mathcal { K } _ { o } ^ { \mathrm { L L } } = \mathrm { T o p K } _ { i \in \mathcal { T } } p _ { i } ( o ) ,
$$

where

$$
p _ { i } ( o ) = \mathrm { s o f t m a x } ( h _ { v _ { i } } ^ { \ell _ { I } } W _ { U } ) _ { o } .
$$

The local score is then computed as the average similarity between the selected image-token representations and the object-token representation:

$$
S _ { \mathrm { l o c a l } } ( o ) = \frac { 1 } { K } \sum _ { i \in \mathcal { K } _ { o } ^ { \mathrm { L L } } } \sin \left( h _ { v _ { i } } ^ { \ell _ { I } } , h _ { o } ^ { \ell _ { T } } \right) .
$$

This baseline measures whether the generated object is compatible with the global and local visual representations.

PAS. PAS estimates attention to see whether object generation is driven by previously generated text tokens. Let $\mathcal { P } _ { < t }$ denote the set of generated prefix-token positions before the object token $o = y _ { t }$ . The preliminary attention score is

$$
T _ { \mathrm { P A S } } ( o ) = \frac { 1 } { | \mathcal { L } | H } \sum _ { \ell \in \mathcal { L } } \sum _ { h = 1 } ^ { H } \sum _ { j \in \mathcal { P } _ { < t } } A _ { \ell , h } ( t , j ) .
$$

A larger $T _ { \mathrm { P A S } } ( o )$ indicates that the object token attends more strongly to previously generated text, suggesting that the mention may be text-prefixdriven rather than visually grounded. To keep all baselines oriented consistently, we use

$$
S _ { \mathrm { P A S } } ( o ) = - T _ { \mathrm { P A S } } ( o ) .
$$

Thus, larger $S _ { \mathrm { P A S } } ( o )$ corresponds to stronger grounding-oriented evidence.

## B Implementation Details

We implement VisER as a post-hoc object-level hallucination detector applied to generated captions. Each model generates at most 512 new tokens. Greedy decoding is used as the default generation setting in the main experiments for determinism and comparability with prior work. In ablation studies in Appendix D, we additionally evaluate robustness to other decoding strategies. VisER is then computed on the generated object mentions and does not modify the decoding process. Following prior LVLM object hallucination evaluations (Jiang et al., 2025a; Li et al., 2023; Phukan et al., 2025), we evaluate on 500 randomly sampled MSCOCO validation images. We use a small held-out calibration split for selecting τ and report sensitivity analyses for $K , \alpha ,$ and layer choice in Ablation studies in Section 4.3 to show that the method is not narrowly tuned to a single configuration. The evidence-scale parameter τ is estimated from a disjoint calibration split of 100 images by averaging the visual evidence mass $M _ { \mathrm { v i s } }$ . The layer indices $( \ell _ { I } , \ell _ { T } )$ , the number of selected prefix tokens K, and the final weighting parameter α are fixed per model, as shown in Table 6. For object words that tokenize into multiple subword units, we use the first matched token to compute the scores. For each generated response, if the same object word appears multiple times, we score only its first generated occurrence, since the first token often captures the core semantic meaning of the object (Park and Li, 2025). Table 7 reports the number of grounded and hallucinated object mentions.

<table><tr><td>Model</td><td> $( \ell _ { I } , \ell _ { T } )$ </td><td>K</td><td>α</td></tr><tr><td>LLaVA-1.5-7B</td><td>(32, 31)</td><td>10</td><td>0.40</td></tr><tr><td>LLaVA-1.5-13B</td><td>(40, 38)</td><td>10</td><td>0.40</td></tr><tr><td>LLaVA-NeXT-7B</td><td>(32, 31)</td><td>10</td><td>0.50</td></tr><tr><td>InstructBLIP</td><td>(32, 31)</td><td>10</td><td>0.40</td></tr><tr><td>MiniGPT-4 Shikra-7B</td><td>(32, 27) (32, 27)</td><td>10 10</td><td>0.25 0.20</td></tr></table>

Table 6: Hyperparameters used for VisER. $\ell _ { I }$ denotes the image-token layer used for visual evidence and image-derived support, $\ell _ { T }$ denotes the text/object representation layer, K is the number of top prefix tokens used for text-prefix support, and α balances Visual Evidence and Visual Reliance.

<table><tr><td>Model</td><td>Grounded</td><td>Hallucinated</td></tr><tr><td>LLaVA-1.5-7B</td><td>1510</td><td>416</td></tr><tr><td>LLaVA-1.5-13B</td><td>1560</td><td>373</td></tr><tr><td>LLaVA-NeXT-7B</td><td>1173</td><td>157</td></tr><tr><td>InstructBLIP</td><td>1582</td><td>556</td></tr><tr><td>MiniGPT-4</td><td>1169</td><td>226</td></tr><tr><td>Shikra-7B</td><td>1631</td><td>446</td></tr></table>

Table 7: Number of generated object mentions evaluated for the models reported in Table 6. Grounded and hallucinated labels are obtained using the CHAIR object-level evaluation.
<table><tr><td>Method</td><td>Greedy</td><td>Beam</td><td>Top-k</td><td>Nucleus</td></tr><tr><td>Entropy</td><td>71.41</td><td>71.13</td><td>73.19</td><td>70.21</td></tr><tr><td>NLL</td><td>70.28</td><td>59.38</td><td>70.60</td><td>67.88</td></tr><tr><td>Internal Conf.</td><td>73.88</td><td>74.90</td><td>74.38</td><td>73.30</td></tr><tr><td>SVAR</td><td>74.39</td><td>73.76</td><td>71.45</td><td>73.50</td></tr><tr><td>Contextual Lens</td><td>70.93</td><td>72.97</td><td>69.59</td><td>70.55</td></tr><tr><td>PAS</td><td>82.52</td><td>82.29</td><td>81.10</td><td>81.62</td></tr><tr><td>GLSim</td><td>84.09</td><td>76.53</td><td>84.47</td><td>84.31</td></tr><tr><td>VisER</td><td>86.17</td><td>82.59</td><td>85.72</td><td>85.75</td></tr></table>

Table 8: Effect of decoding strategy on object hallucination detection.

## C Additional Ablation Studies

## C.1 Effect of Multi-token Object Representation

To assess the effect of multi-token aggregation, we compare three strategies: using the first token, the last token, and averaging over all object tokens. Table 9 reports the corresponding results.

As shown in Table 9, the first-token strategy achieves the highest AUROC on both models, while averaging over all object tokens remains close. These results indicate that VisER is not highly sensitive to the aggregation strategy and support the first-token representation as a simple and effective default.

## C.2 Evidence Gate Distribution

Since $M _ { \mathrm { { v i s } } } ( o )$ is a sum of non-negative visual logitlens probabilities, the evidence gate satisfies

$$
g ( o ) \in [ 0 . 5 , 1 ) .
$$

This restricted range is intentional: the gate is designed as a soft evidence validator rather than a standalone detector. The compatibility score already provides useful grounding information, while the gate modulates this compatibility according to object-specific visual evidence rather than completely suppressing it.

<table><tr><td rowspan="2">Token strategy</td><td colspan="2">LLaVA-1.5-7B</td><td colspan="2">LLaVA-1.5-13B</td></tr><tr><td>AUROC</td><td>AUPR</td><td>AUROC</td><td>AUPR</td></tr><tr><td>First token</td><td>86.17</td><td>95.37</td><td>83.82</td><td>95.35</td></tr><tr><td>Last token</td><td>79.64</td><td>93.46</td><td>79.40</td><td>94.24</td></tr><tr><td>Average over tokens</td><td>85.06</td><td>95.14</td><td>83.16</td><td>95.35</td></tr></table>

Table 9: Effect of multi-token aggregation on MSCOCO.
<table><tr><td>Model</td><td>Grounded Halluc. Gap mean</td><td>mean</td><td></td></tr><tr><td>LLaVA-1.5-7B</td><td>0.686</td><td>0.559</td><td>0.127</td></tr><tr><td> $\mathrm { L L a V A – 1 } . 5 – 1 3 \mathrm { B }$ </td><td>0.667</td><td>0.549</td><td>0.118</td></tr></table>

Table 10: Empirical evidence-gate statistics for grounded and hallucinated object mentions on MSCOCO.

Table 10 reports the empirical gate statistics for grounded and hallucinated object mentions.

As shown in Table 10, the empirical gate distribution is not degenerate. Grounded objects receive consistently higher mean gate values than hallucinated objects on both model scales. Thus, despite its restricted range, the gate provides meaningful variation in object-specific visual evidence.

![](images/f441b99c4ae860aecdf6c2bae940b02e598bc2cfa47b53894b9f78af3a3ed07b.jpg)  
Figure 6: Density visualization of object-level scores for grounded and hallucinated objects.

## D Effect of Decoding Strategies

Following prior object hallucination detection studies (Hoang et al., 2026; Park and Li, 2025; Jiang et al., 2025a), we use greedy decoding as the default setting because it is simple, deterministic, and computationally efficient. To examine whether VisER depends on this specific generation regime, we additionally evaluate detection performance under three alternative decoding strategies, including beam search, top-k sampling, and nucleus sampling. We use $N _ { \mathrm { b e a m s } } ~ = ~ 2 , ~ k ~ = ~ 1 0 .$ and $p = 0 . 9$ for beam search, top-k, and nucleus decoding, respectively, and apply the same objectlevel evaluation protocol to the generated captions. As shown in Table 8, VisER remains consistently strong across decoding strategies, suggesting that its two-sided grounding score is robust to changes in the generation procedure rather than being specialized to greedy decoding.

## E Visualization and Examples

## E.1 Visualization of Scores

Figure 6 compares the object-level score distributions produced by NLL and VisER. The VisER scores show a substantially clearer distinction between grounded and hallucinated object mentions, whereas the NLL distributions remain more concentrated and overlapping. This indicates that token likelihood alone provides a limited signal for visual faithfulness, since likely object tokens may still be driven by language or scene priors rather than direct image evidence. By contrast, VisER produces a more discriminative scoring space by jointly validating object-specific visual evidence and object-level support. The resulting separation suggests that VisER reliably distinguishes visually grounded objects from mentions induced by contextual or autoregressive priors.

## E.2 Examples of Source-Confounded Support

Figure 7 provides representative qualitative examples where hallucinated object mentions are correctly detected by VisER but missed by one or more baseline support signals. These cases illustrate the source-confounding problem. A hallucinated object can receive high support from scene context, object co-occurrence, or the generated prefix, even when the object is not visually grounded in the image. In the first row, both PAS and GLSim, as the best available baselines, incorrectly classify the highlighted hallucinated objects as grounded. In the second row, GLSim fails while VisER correctly detects the hallucination. In the third row, PAS fails while VisER again correctly identifies the highlighted object mentions as hallucinated. These examples show that measuring support strength alone can be misleading, whereas VisER improves reliability by checking whether the support is visually sourced.

![](images/ba55b2a844820d97b8e23fd348a21c126fd1d1ddce6d1699c79347a50cfb8dfa.jpg)  
Figure 7: Examples of source-confounded hallucinations. Highlighted words indicate hallucinated object mentions in the generated captions. In row (a), both PAS and GLSim incorrectly classify the hallucinated objects as grounded, while VisER correctly detects them as hallucinated. In row (b), GLSim fails but VisER succeeds. In row (c), PAS fails but VisER succeeds. These examples illustrate that baseline support signals can be misled by contextual plausibility or generated-prefix continuation, whereas VisER requires stronger visually sourced evidence.

![](images/668b4c135b0567d760327fbaef856acda2388a4e383ec1ef8ae1538caa908e14.jpg)  
The image features a delicious breakfast meal consisting of a bacon sandwich and a pancake. The sandwich is placed on top of the pancake, creating a unique and appetizing combination.

![](images/f3600bf3e783332bafe9f11249cd73f77a422f7b04276b7834910d3fae5710d3.jpg)  
The horses are spread out, with some closer to the foreground and others further back in the field.  In the background, there is a large building, possibly a church, with a clock tower visible.

![](images/cb8b34cecb25fb5429422f8d8485f16d18ab76d06b8add65f43bf0996cb260ef.jpg)  
A fork can be seen on the right side of the table, and a bowl is located near the center of the table.  The dining table is surrounded by chairs, with one chair on the left side, another on the right side, and a third chair at the top left corner.

![](images/aa1ce08905539ee09e517e19e32e868608f0bcd474f4bb4a943d03dad9d04971.jpg)  
The image depicts a busy city street with a train passing over a bridge. The scene captures the hustle and bustle of a typical city street, with various modes of transportation and people going about their daily routines.  
Figure 8: Qualitative examples of VisER on hallucinated object mentions. Bars show percentile-normalized Visual Evidence $V E ( o )$ , Visual Reliance $V R ( o )$ , and VisER; the dashed line denotes the threshold. VisER suppresses hallucinations that are either scene-prior-driven, such as “sandwich” and “clock”, or insufficiently supported by visual evidence, such as “chairs” and “train”.

## E.3 Examples of Scoring

Figure 8 provides qualitative examples illustrating how VisER combines Visual Evidence $V E ( o )$ and Visual Reliance $V R ( o )$ to detect object-level hallucinations. The bars show percentile-normalized component scores, while the dashed horizontal line denotes the threshold. In the first row, the hallucinated objects are plausible from the scene context: “sandwich” is suggested by the breakfast setting, and “clock” is suggested by the church-like building. However, both cases receive weak objectspecific visual evidence, which lowers the final VisER score despite contextual plausibility. In the second row, the hallucinated objects receive different component patterns: “chairs” obtains moderate visual support from the dining scene, while “train” is associated with the elevated rail structure, but the final VisER score remains low when the evidence is not sufficiently reliable.

## F Source-Confounding Analysis

We provide a simple sufficient-condition analysis showing why the two VisER components address complementary false-support regimes. Let

$$
\begin{array} { r l r } & { \boldsymbol { M _ { o } } = \boldsymbol { M _ { \mathrm { v i s } } } ( o ) , } & { C _ { o } = \sin ( h _ { \mathrm { c t x } } ^ { \ell _ { T } } , h _ { o } ^ { \ell _ { T } } ) , } \\ & { \boldsymbol { I _ { o } } = \boldsymbol { S _ { \mathrm { i m g } } } ( o ) , } & { T _ { o } = \boldsymbol { S _ { \mathrm { t e x t } } } ( o ) , } \end{array}
$$

where $C _ { o }$ denotes object-context compatibility, and let $\eta = \tau + \epsilon$ . Then

$$
\begin{array} { l } { \displaystyle \mathrm { V E } ( o ) = C _ { o } \sigma ( M _ { o } / \eta ) , } \\ { \displaystyle \mathrm { V R } ( o ) = \frac { I _ { o } } { I _ { o } + T _ { o } + \epsilon } . } \end{array}
$$

Complementary false-support regimes. First, consider an evidence-limited hallucination $O _ { h }$ with nonnegative object-context compatibility $C _ { o _ { h } } \geq$ 0. The object may be compatible with the scene or associated with related visual cues, but objectspecific image-token evidence is weak:

$$
M _ { o _ { h } } \leq m _ { h } .
$$

Since $C _ { o _ { h } } \geq 0$ and $\sigma ( \cdot )$ is monotone,

$$
\mathrm { V E } ( o _ { h } ) = C _ { o _ { h } } \sigma ( M _ { o _ { h } } / \eta ) \le C _ { o _ { h } } \sigma ( m _ { h } / \eta ) .
$$

Thus, high contextual compatibility alone is not sufficient for a high VE score; the compatibility term is restricted by object-specific image-token evidence.

Second, consider a prefix-dominated hallucination $O h$ , where the generated prefix provides stronger support than the image:

$$
T _ { o _ { h } } \ge \kappa I _ { o _ { h } } , \qquad \kappa > 1 .
$$

For clarity, taking $\epsilon = 0$ , we obtain

$$
\mathrm { V R } ( o _ { h } ) = \frac { I _ { o _ { h } } } { I _ { o _ { h } } + T _ { o _ { h } } } \le \frac { 1 } { 1 + \kappa } .
$$

Thus, VR is small when support for the object is dominated by the autoregressive prefix rather than by image-derived evidence.

Source-validation margins. Let $o _ { g }$ be a grounded object and $O _ { h }$ a hallucinated object. In the evidence-limited regime, assume

$$
M _ { o _ { g } } \geq m _ { g } , \qquad M _ { o _ { h } } \leq m _ { h } , \qquad m _ { g } > m _ { h } ,
$$

with $C _ { o _ { g } } \geq c _ { g } \geq 0$ and $0 \leq C _ { o _ { h } } \leq c _ { h }$ . Then

$$
\begin{array} { r } { \mathrm { V E } ( o _ { g } ) - \mathrm { V E } ( o _ { h } ) \geq c _ { g } \sigma ( m _ { g } / \eta ) - c _ { h } \sigma ( m _ { h } / \eta ) . } \end{array}
$$

Therefore, VE gives a positive margin whenever

$$
c _ { g } \sigma ( m _ { g } / \eta ) > c _ { h } \sigma ( m _ { h } / \eta ) .
$$

In the prefix-dominated regime, assume

$$
I _ { o _ { g } } \ge \kappa T _ { o _ { g } } , \qquad T _ { o _ { h } } \ge \kappa I _ { o _ { h } } , \qquad \kappa > 1 .
$$

Again taking $\epsilon = 0$ for clarity,

$$
\mathrm { V R } ( o _ { g } ) \geq \frac { \kappa } { 1 + \kappa } , \qquad \mathrm { V R } ( o _ { h } ) \leq \frac { 1 } { 1 + \kappa } ,
$$

and hence

$$
\mathrm { V R } ( o _ { g } ) - \mathrm { V R } ( o _ { h } ) \geq \frac { \kappa - 1 } { \kappa + 1 } .
$$

For the final score

$$
s _ { \mathrm { V i s E R } } ( o ) = \alpha \mathrm { V E } ( o ) + ( 1 - \alpha ) \mathrm { V R } ( o ) ,
$$

define

$$
\Delta _ { E } = \mathrm { V E } ( o _ { g } ) - \mathrm { V E } ( o _ { h } ) ,
$$

$$
\Delta _ { R } = \mathrm { V R } ( o _ { g } ) - \mathrm { V R } ( o _ { h } ) .
$$

Then we have

$$
s _ { \mathrm { V i s E R } } ( o _ { g } ) - s _ { \mathrm { V i s E R } } ( o _ { h } ) = \alpha \Delta _ { E } + ( 1 - \alpha ) \Delta _ { R } .
$$

Thus, VisER has a positive margin whenever

$$
\alpha \Delta _ { E } + ( 1 - \alpha ) \Delta _ { R } > 0 .
$$

In particular, if $\Delta _ { E } \geq 0 , \Delta _ { R } \geq 0 .$ , and at least one inequality is strict, then $s _ { \mathrm { V i s E R } } ( o _ { g } ) > s _ { \mathrm { V i s E R } } ( o _ { h } )$ for any $\alpha \in ( 0 , 1 )$ . This formalizes the complementarity of the two components: VE is most useful for evidence-limited false support, where compatibility is not backed by object-specific image evidence, while VR is most useful for prefix-dominated false support, where the generated prefix explains the object better than the image.

## G Usage of Large Language Models

We used OpenAI GPT-5 language models only for language editing and proofreading. The models were not involved in idea development or method design.

## H Artifact Licenses

We use publicly available datasets, models, and baseline methods, including MSCOCO, Pascal VOC, LLaVA, LLaVA-NeXT, InstructBLIP, MiniGPT-4, InternVL3, Shikra, and Qwen2.5-VL, under their respective licenses and terms of use. Our use of these artifacts is limited to academic research and evaluation.