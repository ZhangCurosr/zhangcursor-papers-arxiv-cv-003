# The Shape of Time: Video-Token Contrast for Temporal Understanding in VideoLMs

Yumeng Shi Quanyu Long Yin Wu Wenya Wang

Nanyang Technological University

{yumeng001, quanyu001, wuyi0023}@e.ntu.edu.sg wangwy@ntu.edu.sg

## Abstract

Seeing frames in order does not mean representing time. Modern VideoLMs receive ordered video streams, yet their main supervision acts on generated text rather than video-token representations where event dynamics should first emerge. This mismatch allows models to learn temporal answers from shortcuts such as objects, scenes, and language priors, without requiring internal video representations to capture event progression. To address this, we propose VT-Contrast, a representation-level temporal counterfactual objective for VideoLMs. Its design asks where temporal supervision should act and what temporal differences it should expose. VT-Contrast supervises selected latelayer last-frame video tokens, where temporal information is expected to be integrated before language generation, and contrasts orderpreserving views with same-video reordered counterfactuals graded by Kendall tau distance. It requires no architectural changes, is compatible with diverse VideoLM training tasks, and improves overall performance across temporal understanding benchmarks. Our code is available at https://github.com/ANDgate99/VT-Contrast.

## 1 Introduction

Video Language Models (VideoLMs) (Lin et al., 2024; Maaz et al., 2024) are increasingly expected to reason about how events unfold over time (Li et al., 2025c; Qin et al., 2025), rather than merely recognize visual content. For example, two clips may contain nearly the same objects and scenes, while expressing opposite events under different temporal progressions: opening versus closing, entering versus leaving, or assembling versus disassembling. Resolving such order-dependent distinctions requires models to capture the progression of visual states over time.

However, standard VideoLM training does not directly enforce this requirement. While VideoLMs receive temporal information through ordered visual tokens, supervision is usually applied only to the generated text (Zhang et al., 2023, 2025c). The language modeling loss specifies the desired answer, yet does not require the resulting visual representations to reflect the event progression that supports it. For order-sensitive videos, this is a weak constraint: clips with similar visual evidence but different temporal order may still lead to correct responses based on static cues or answer priors. As a result, response-level supervision may leave video representations underconstrained along the dimension most relevant to event progression, even when the final responses are correct.

![](images/fdde9be48d41c32c99d73288f70b98525bfdcbc5b5d3a38272f2f2ea63784475.jpg)  
Figure 1: Motivation of VT-Contrast. Temporal order changes video semantics; VT-Contrast separates temporally corrupted negatives while keeping order-consistent positives close to the anchor in the video-token space.

A more direct form of temporal supervision is therefore needed before response generation. Such supervision should make temporal order explicit in the internal video-token representation space, rather than relying solely on the final textual response. Prior approaches move in related directions but do not directly impose this constraint. Temporal-aware architectures (Hu et al., 2024) improve temporal aggregation in visual or intermediate modules, but do not directly supervise temporalorder representations after video tokens enter the language-model space. Ordering-based auxiliary tasks (Wu et al., 2026) expose models to shuffled segments, but still rely on task-level responses and can be less compatible with existing training objectives. Reconstruction objectives (Tong et al., 2022) offer more direct visual supervision for representations, but often require extra branches and costly pixel-level targets.

A representation-level temporal signal should distinguish videos primarily by their temporal order rather than by their visual content. We instantiate this idea by constructing multiple views from the same video: order-preserving views keep the original event progression, whereas reordered views retain the same visual evidence but change how the event unfolds. As illustrated in Figure 1, such order changes can alter event semantics, yet existing VideoLMs may still map the corresponding videotoken representations close together. We formulate these reordered views as temporal counterfactuals and propose VT-Contrast, a lightweight objective that separates them from order-preserving views in the video-token space. This makes the learning signal depend on temporal order rather than video identity or static appearance cues.

Beyond this counterfactual construction, VT-Contrast must decide where the supervision should act and how strong the temporal signal should be. For supervision placement, we apply the objective to selected late-layer last-frame video tokens, which are expected to aggregate information from preceding frames before language generation. This choice is motivated by prior findings that temporal information emerges progressively inside VideoLM representations, rather than being uniformly available at all layers (Zhang et al., 2025d; Shi et al., 2026). For temporal signal strength, we avoid treating all reorderings as equivalent. We use Kendall tau distance (Kendall, 1938; Diaconis and Graham, 1977) to quantify pairwise order violations, enabling VT-Contrast to construct graded temporal counterfactuals that yield more effective contrastive supervision.

VT-Contrast is jointly optimized with the standard language modeling objective, requiring no architectural modification. By directly shaping videotoken geometry, it complements response-level supervision: the model is still trained to generate correct answers, while its internal video representations are encouraged to reflect event progression. Experiments on multiple temporal understanding benchmarks show consistent overall gains, especially on order-sensitive tasks. Our contributions are summarized as follows:

• We introduce a representation-level temporal supervision perspective for VideoLMs, arguing that temporal order should be reflected in video-token representations rather than only supervised through textual responses.

• We propose VT-Contrast, a lightweight temporal counterfactual objective for VideoLM training that uses same-video temporally reordered sequences as negatives. It applies targeted supervision to selected-layer last-frame token representations and controls negative difficulty with Kendall tau distance.

• Evaluations on multiple temporal benchmarks show consistent overall improvements. VT-Contrast integrates into standard VideoLM training without changing the original task format, enabling flexible temporal enhancement across diverse video understanding scenarios.

## 2 Related Works

## 2.1 Video Language Models

Recent VideoLMs (Zhang et al., 2025b; Lin et al., 2024) typically extend LLM-based multimodal understanding from images to videos by encoding frames or clips into visual tokens, projecting them into LLM feature space, and generating textual responses autoregressively (Liu et al., 2023; Maaz et al., 2024). Representative systems include both closed-source models (Comanici et al., 2025; Achiam et al., 2023) and open-source models (Bai et al., 2025; Chen et al., 2024; Li et al., 2025a). Despite differences in model scale, data, and training recipes, their training signal is primarily imposed on generated text tokens rather than on intermediate video-token representations (Zhang et al., 2023, 2025c; Feng et al., 2025). Although language loss can update visual components, it does not directly constrain video-token representations to distinguish temporal order changes.

## 2.2 Temporal Understanding

Temporal understanding is fundamental to VideoLMs, as videos require models to reason over event order, motion direction, state changes, and other relations beyond static visual recognition (Zhang et al., 2024; Chandrasegaran et al.,

![](images/e331705515a7d1cbd6691642ac223168557ee8bd75f6aea168eb5f5702776cd8.jpg)  
Figure 2: Overview of VT-Contrast. For each input video, we construct an anchor view, an order-preserving positive view, and temporal counterfactual negative views. A shared VideoLM processes these views to obtain selected video-token representations for contrastive supervision. VT-Contrast keeps temporally consistent views close and separates negative views, and is jointly optimized with the standard language modeling loss.

2024). Benchmarks such as ViLMA (Kesen et al., 2024), TempCompass (Liu et al., 2024), and TOMATO (Shangguan et al., 2025) show that current VideoLMs still struggle with fine-grained temporal semantics, especially when videos share similar visual content but differ in temporal order or event progression. To improve temporal understanding, prior work has explored temporal-aware architectures, temporal reasoning transfer, and auxiliary ordering tasks (Hu et al., 2024; Fateh et al., 2026). For instance, T3 (Li et al., 2025b) transfers temporal reasoning from synthetic text tasks, while Visual Jigsaw (Wu et al., 2026) trains models to recover shuffled visual order. While these efforts highlight the importance of temporal modeling, directly shaping video-token representations for temporal sensitivity remains less explored.

## 2.3 Contrastive Learning

Contrastive learning is widely used for representation learning by pulling positive pairs together and pushing negative samples apart in feature space (He et al., 2020; Chen et al., 2020). In vision-language learning, CLIP-style objectives align visual and textual representations (Radford et al., 2021), and recent studies further extend contrastive training to VLM-based multimodal embedding models (Jiang et al., 2025; Meng et al., 2026). Beyond imagetext alignment, contrastive objectives have also been applied to video representation learning (Pan et al., 2021; Qian et al., 2021). Some works further exploit temporal perturbations to enhance video representations (Dave et al., 2022; Dorkenwald et al., 2022; Jenni and Jin, 2021). However, these methods mainly target general-purpose representations in standalone video encoders. In contrast, VT-Contrast targets the underconstrained temporal structure of VideoLM video-token representations, directly separating temporally reordered videos in this internal representation space.

## 3 Method

We propose VT-Contrast, a lightweight temporal counterfactual learning method that enhances temporal order sensitivity in VideoLMs through videotoken supervision. It provides direct representationlevel supervision by contrasting temporally consistent views with reordered views from the same video. We then describe how VT-Contrast is integrated into VideoLM training.

## 3.1 Problem Setup

Given a common video-language training dataset, each sample consists of a video $V = \{ v _ { 1 } , \ldots , v _ { T } \}$ a textual query $Q ,$ , and a target response Y = $\{ y _ { 1 } , \dotsc , y _ { N } \}$ , where $v _ { t }$ denotes the t-th sampled frame. A VideoLM processes $V$ to obtain videotoken representations, which are used as visual context for the language model. The response is predicted autoregressively as:

$$
p _ { \theta } ( Y \mid V , Q ) = \prod _ { i = 1 } ^ { N } p _ { \theta } ( y _ { i } \mid y _ { < i } , V , Q ) ,\tag{1}
$$

where $p _ { \theta }$ denotes the token distribution predicted by the VideoLM with parameters $\theta .$

Standard VideoLM training minimizes the negative log-likelihood of the target response:

$$
\mathcal { L } _ { \mathrm { L M } } = - \sum _ { i = 1 } ^ { N } \log p _ { \theta } ( y _ { i } \mid y _ { < i } , V , Q ) .\tag{2}
$$

Although this response-level objective aligns videos with textual answers, it supervises videotoken representations only indirectly through the output loss. Consequently, these representations are not explicitly encouraged to distinguish finegrained temporal order changes, which can allow the model to rely on static visual cues or language priors when they are sufficient for response generation. We therefore introduce an auxiliary contrastive objective based on temporal counterfactual views to directly supervise video-token representations during VideoLM training.

## 3.2 Temporal Counterfactual Learning

As illustrated in Figure 2, VT-Contrast constructs temporally consistent and order-disrupted views from the same input video. The order-disrupted views serve as temporal counterfactuals: they preserve the visual content of the original video while altering its event progression. Instead of relying on textual responses for temporal supervision, our method contrasts these video views directly in the representation space. This makes temporal order differences explicit in video-token representations without changing the response format.

Contrastive view construction. We instantiate this idea by constructing three views from each training video $V = \{ v _ { 1 } , \ldots , v _ { T } \}$ : an anchor view $V ^ { a }$ , an order-consistent positive view $V ^ { + }$ , and an order-disrupted negative view $V ^ { - }$ <sup>−</sup>. The anchor defines the reference temporal progression, the positive preserves this progression under a mild sampling perturbation, and the counterfactual negative view changes the progression while keeping the sampled visual content fixed. Together, these views allow the contrastive objective to compare temporal consistency against order disruption under controlled visual content.

▷ Anchor: The anchor view $V ^ { a }$ is the reference chronological view. It is obtained by uniformly sampling K frames from the original video while preserving their temporal order. Formally, let $i _ { 1 } <$ $i _ { 2 } < \cdots < i _ { K }$ denote the sampled frame indices; the anchor view is defined as:

$$
V ^ { a } = \{ v _ { i _ { 1 } } , v _ { i _ { 2 } } , . . . , v _ { i _ { K } } \} .\tag{3}
$$

▷ Positive: The positive view $V ^ { + }$ is an orderconsistent counterpart to the anchor view. It preserves the temporal progression and main visual content of $V ^ { a }$ while introducing a mild sampling variation. In this work, we instantiate this perturbation by randomly dropping one frame from $V ^ { a }$ Let $d \in \{ 1 , \ldots , K \}$ denote the dropped position; the positive view is defined as:

$$
V ^ { + } = \{ v _ { i _ { 1 } } , \ldots , v _ { i _ { d - 1 } } , v _ { i _ { d + 1 } } , \ldots , v _ { i _ { K } } \} .\tag{4}
$$

Since the remaining frames preserve the original chronological order, $V ^ { a }$ and $V ^ { + }$ are treated as a positive pair.

▷ Negative: The negative view $V ^ { - }$ is designed to isolate temporal order as the main contrasting factor. It reuses the same sampled frames as the anchor view, so that objects, scenes, and local visual patterns are largely preserved while the chronological order is disrupted. This same-video construction reduces shortcuts based on video identity or static appearance, making temporal order the primary difference between $V ^ { a }$ and $V ^ { - }$

Specifically, we apply a non-identity temporal permutation π to the anchor sequence:

$$
V ^ { - } = \{ v _ { i _ { \pi ( 1 ) } } , v _ { i _ { \pi ( 2 ) } } , \ldots , v _ { i _ { \pi ( K ) } } \} , \quad \pi \neq \mathrm { i d } .\tag{5}
$$

Thus, $V ^ { - }$ <sup>−</sup> serves as a temporal counterfactual negative that differs from $V ^ { a }$ in frame order.

Contrastive objective. To realize temporal counterfactual learning, we train the anchor representation to stay close to its order-consistent positive view and away from same-video counterfactual negatives. In this way, the contrastive objective turns temporal order changes into explicit constraints on video-token representations. Specifically, we adopt an InfoNCE objective (Oord et al., 2018). For each anchor view, we construct M counterfactual negative views $\lbrace V _ { m } ^ { - } \rbrace _ { m = 1 } ^ { M }$ . Let $r ( U )$ denote the video-token representation extracted from any constructed view $U .$ We compute the positive and negative similarity scores as:

$$
\begin{array} { r } { s ^ { + } = \sin \bigl ( r ( V ^ { a } ) , r ( V ^ { + } ) \bigr ) , } \\ { s _ { m } ^ { - } = \sin \bigl ( r ( V ^ { a } ) , r ( V _ { m } ^ { - } ) \bigr ) . } \end{array}\tag{6}
$$

The contrastive loss is defined as:

$$
\mathcal { L } _ { \mathrm { c o n } } = - \log \frac { \exp ( s ^ { + } / \tau _ { c } ) } { \exp ( s ^ { + } / \tau _ { c } ) + \sum _ { m = 1 } ^ { M } \exp ( s _ { m } ^ { - } / \tau _ { c } ) } ,
$$

where sim $( \cdot , \cdot )$ denotes cosine similarity and $\tau _ { c }$ is the contrastive temperature.

![](images/b7acfc57939589b93b45f20252d51bda91abea2604174245f571a73117ee646f.jpg)  
Figure 3: Kendall-tau-based grading of temporal counterfactual negatives. Smaller $\hat { d } _ { \tau }$ yields harder counterfactuals closer to the original order, while larger $\hat { d } _ { \tau }$ indicates stronger order disruption and easier contrasts.

## 3.3 Emergence-aware Temporal Supervision

The temporal counterfactual objective should not be applied blindly to all video tokens or all reordered views. We therefore design the supervision to be both targeted and graded, so that the representation constraint focuses on informative temporal features while keeping the counterfactual comparisons at an appropriate difficulty.

Targeted video-token supervision. We now describe how the contrastive representation $r ( U )$ is extracted for each constructed view. Each view is processed by the shared VideoLM to obtain layer-wise video-token features. Motivated by the observation that different layers capture different stages of multimodal information flow and lastframe features can aggregate temporal cues across the video (Zhang et al., 2025d; Shi et al., 2026), we use last-frame representations from selected layers $s$ for contrastive supervision. This targeted design avoids dense constraints over all layers and video tokens while focusing on compact, temporally informative features.

Specifically, for each view U of the original video V and selected layer $l \in S .$ , we take the lastframe video-token features, denoted as $H _ { l } ^ { \mathrm { l a s t } } ( U )$ We first apply mean pooling over the last-frame tokens at each selected layer and then average the resulting features across selected layers:

$$
r ( U ) = \frac { 1 } { | S | } \sum _ { l \in S } \mathrm { M e a n P o o l } \left( H _ { l } ^ { \mathrm { l a s t } } ( U ) \right) ,\tag{8}
$$

where MeanPool(·) averages the last-frame video tokens into a single feature vector. Since different VideoLM backbones vary in depth and representation structure, the selected layer set $s$ is model-dependent. We specify the exact layers in the implementation details.

Graded temporal counterfactuals. Temporal counterfactuals should differ from the anchor in temporal order, but the degree of order violation affects the learning signal. Smaller violations create harder fine-grained contrasts, while larger violations produce more explicit but easier counterfactuals. Therefore, we use Kendall tau distance (Kendall, 1938; Diaconis and Graham, 1977) to grade the strength of order violation, allowing us to control the signal and construct effective temporal contrasts, as illustrated in Figure 3.

Given a permutation $\pi _ { m }$ over the K frames of the anchor view, the Kendall tau distance counts the number of pairwise order inversions:

$$
d _ { \tau } ( \pi _ { m } ) = \sum _ { 1 \leq i < j \leq K } \mathbb { I } [ \pi _ { m } ( i ) > \pi _ { m } ( j ) ] ,\tag{9}
$$

where $\mathbb { I } [ \cdot ]$ is the indicator function. Since the maximum number of inversions is $K ( K - 1 ) / 2$ , we normalize the distance as:

$$
\hat { d } _ { \tau } ( \pi _ { m } ) = \frac { 2 d _ { \tau } ( \pi _ { m } ) } { K ( K - 1 ) } .\tag{10}
$$

The normalized distance $\hat { d } _ { \tau } ( \pi _ { m } ) \in [ 0 , 1 ]$ measures how strongly the counterfactual violates the original temporal order. Smaller values indicate that the permutation remains close to the original order and is harder to distinguish, whereas larger values indicate stronger order disruption.

During training, we sample each counterfactual permutation $\pi _ { m }$ from a controlled range:

$$
\delta _ { \mathrm { m i n } } < \hat { d } _ { \tau } ( \pi _ { m } ) \leq \delta _ { \mathrm { m a x } } ,\tag{11}
$$

where $\delta _ { \mathrm { m i n } }$ and $\delta _ { \mathrm { m a x } }$ define the allowed degree of order violation. If the number of valid permutations in the range is insufficient, we repeatedly sample with replacement until obtaining the required M negatives. In our implementation, we use $0 < \hat { d } _ { \tau } ( \pi _ { m } ) \leq 0 . 5$ . This favors counterfactuals that introduce order violations while remaining relatively close to the original sequence, making them harder to distinguish.

## 3.4 Optimization Objective

We keep the standard language modeling objective: it provides semantic supervision for the original temporally ordered video-query pair and preserves the model’s generation ability. VT-Contrast is added as an auxiliary objective, providing direct temporal supervision on video-token representations. The final training loss is:

<table><tr><td rowspan="2" colspan="2">Frames Model</td><td rowspan="2">TOMATO</td><td colspan="4">TempCompass</td><td colspan="3">Vinoground</td></tr><tr><td>Y/N</td><td>MCQ</td><td>Caption</td><td>Match</td><td>Text</td><td>Video</td><td>Group</td></tr><tr><td rowspan="9">8</td><td>LLaVA-OneVision1.5-4B</td><td>21.09</td><td>62.86</td><td>56.71</td><td>50.00</td><td>68.26</td><td>20.60</td><td>23.80</td><td>3.60</td></tr><tr><td>InternVL3.5-4B</td><td>28.71</td><td>72.08</td><td>68.73</td><td>60.93</td><td>79.44</td><td>48.20</td><td>28.20</td><td>16.00</td></tr><tr><td>Visual Jigsaw</td><td>26.01</td><td>73.50</td><td>71.77</td><td>61.93</td><td>79.51</td><td>44.80</td><td>25.00</td><td>13.20</td></tr><tr><td>Qwen3.5-0.8B</td><td>26.01</td><td>63.64</td><td>58.80</td><td>48.35</td><td>70.39</td><td>26.60</td><td>22.00</td><td>7.00</td></tr><tr><td>Qwen3.5-2B</td><td>25.74</td><td>71.02</td><td>67.34</td><td>56.64</td><td>77.45</td><td>34.00</td><td>20.40</td><td>7.60</td></tr><tr><td>Qwen3.5-4B</td><td>33.42</td><td>77.62</td><td>71.90</td><td>49.50</td><td>83.77</td><td>50.00</td><td>31.80</td><td>19.60</td></tr><tr><td>Ours-0.8B</td><td>27.29 +1.28</td><td>64.98+1.34</td><td>59.49 +0.69</td><td>50.05 +1.70</td><td>70.26-0.13</td><td>29.40 +2.80</td><td>22.20+0.20</td><td>7.20 +0.20</td></tr><tr><td>Ours-2B</td><td>31.47+5.73</td><td>74.97+3.95</td><td>70.00+2.66</td><td>64.17+7.53</td><td>80.17+2.72</td><td>41.20+7.20</td><td>25.00+4.60</td><td>11.40+3.80</td></tr><tr><td>Ours-4B</td><td>36.46+3.04</td><td>79.33+1.71</td><td>74.94+3.04</td><td>50.80+1.30</td><td>85.30+1.53</td><td>57.20+7.20</td><td>29.00-2.80</td><td>20.00+0.40</td></tr><tr><td rowspan="9">16</td><td>LLaVA-OneVision1.5-4B</td><td>20.55</td><td>62.25</td><td>57.47</td><td>51.15</td><td>69.33</td><td>20.00</td><td>23.80</td><td>4.80</td></tr><tr><td>InternVL3.5-4B</td><td>30.12</td><td>72.89</td><td>70.70</td><td>61.68</td><td>80.44</td><td>53.60</td><td>30.80</td><td>16.80</td></tr><tr><td>Visual Jigsaw</td><td>25.13</td><td>74.60</td><td>72.78</td><td>62.72</td><td>80.64</td><td>47.20</td><td>28.00</td><td>15.40</td></tr><tr><td>Qwen3.5-0.8B</td><td>25.81</td><td>64.04</td><td>59.94</td><td>49.75</td><td>70.73</td><td>27.60</td><td>21.20</td><td>5.80</td></tr><tr><td>Qwen3.5-2B</td><td>26.55</td><td>72.48</td><td>69.18</td><td>58.83</td><td>78.84</td><td>37.40</td><td>23.00</td><td>9.60</td></tr><tr><td>Qwen3.5-4B</td><td>35.58</td><td>77.66</td><td>74.68</td><td>50.90</td><td>84.56</td><td>57.80</td><td>33.40</td><td>22.40</td></tr><tr><td>Ours-0.8B</td><td>27.63 +1.82</td><td>65.27 +1.23</td><td>60.95 +1.01</td><td>50.15+0.40</td><td>71.39 +0.66</td><td>31.00+3.40</td><td>22.60 +1.40</td><td>8.80 +3.00</td></tr><tr><td>Ours-2B Ours-4B</td><td>31.20+4.65</td><td>76.07+3.59</td><td>71.46+2.28</td><td>65.77 +6.94</td><td>82.24+3.40</td><td>43.40+6.00</td><td>24.80+1.80</td><td>12.80 +3.20</td></tr><tr><td></td><td>37.67 +2.09</td><td>79.49 +1.83</td><td>76.20+1.52</td><td>50.55 -0.35</td><td> $\mathbf { 8 5 . 6 3 _ { \div } } _ { I . 0 7 }$ </td><td>59.80+2.00</td><td>31.60-1.80</td><td>21.00-1.40</td></tr><tr><td rowspan="9">32</td><td>LLaVA-OneVision1.5-4B</td><td>20.49</td><td>62.49</td><td>56.71</td><td>49.85</td><td>68.80</td><td>19.80</td><td>23.40</td><td>4.80</td></tr><tr><td>InternVL3.5-4B</td><td>29.65</td><td>73.46</td><td>70.76</td><td>62.03</td><td>80.71</td><td>52.80</td><td>32.60</td><td>19.40</td></tr><tr><td>Visual Jigsaw</td><td>26.48</td><td>74.52</td><td>72.78</td><td>61.93</td><td>80.24</td><td>48.80</td><td>30.80</td><td>18.60</td></tr><tr><td>Qwen3.5-0.8B</td><td>26.08</td><td>64.57</td><td>60.89</td><td>51.50</td><td>71.99</td><td>28.00</td><td>20.00</td><td>5.80</td></tr><tr><td>Qwen3.5-2B</td><td>27.02</td><td>72.97</td><td>69.43</td><td>58.13</td><td>79.57</td><td>38.20</td><td>22.60</td><td>9.20</td></tr><tr><td>Qwen3.5-4B</td><td>36.39</td><td>78.60</td><td>74.62</td><td>48.65</td><td>84.96</td><td>59.20</td><td>33.80</td><td>21.80</td></tr><tr><td></td><td>28.17+2.09</td><td>66.08+1.51</td><td>62.22 +1.33</td><td> $5 0 . 1 0 _ { ^ { - } I . 4 0 }$ </td><td> $7 2 . 1 2 \substack { + 0 . l 3 }$ </td><td>30.60 +2.60</td><td>25.60+5.60</td><td></td></tr><tr><td>Ours-0.8B Ours-2B</td><td> $3 0 . 9 3 \substack { + 3 . 9 I }$ </td><td> $7 5 . 9 5 _ { \mathrm { + 2 . 9 8 } }$ </td><td> ${ 7 1 . 7 1 + 2 . 2 8 }$ </td><td> ${ \bf 6 5 . 5 2 _ { \mathrm { + 7 . 3 9 } } }$ </td><td> $\mathbf { 8 2 . 4 4 } _ { + 2 . 8 7 }$ </td><td> $\mathbf { 4 4 . 6 0 } _ { + 6 . 4 0 }$ </td><td> $2 5 . 4 0 \substack { + 2 . 8 0 }$ </td><td>8.80 +3.00  ${ \bf 1 3 . 2 0 } _ { + 4 . 0 0 }$ </td></tr><tr><td>Ours-4B</td><td>38.41 +2.02</td><td> $\mathbf { 7 9 . 5 4 } _ { + 0 . 9 4 }$ </td><td> $7 6 . 5 8 \substack { + 1 . 9 6 }$ </td><td> ${ \bf 5 0 . 7 5 } _ { + 2 . l 0 }$ </td><td> $\mathbf { 8 6 . 0 9 } _ { + 1 . 1 3 }$ </td><td> ${ \bf 6 0 . 4 0 } _ { + 1 . 2 0 }$ </td><td> $\mathbf { 3 4 . 0 0 } _ { + 0 . 2 0 }$ </td><td> $2 2 . 6 0 _ { + 0 . 8 0 }$ </td></tr></table>

Table 1: Main results on TOMATO, TempCompass, and Vinoground under different frame settings. Our 4B model performs better than recent 4B MLLMs on most benchmarks. Our method also improves over the corresponding Qwen3.5 models in most settings, showing that VT-Contrast strengthens temporal understanding without architec tural changes. Small colored numbers denote absolute gains over the corresponding Qwen3.5 model.

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { L M } } + \lambda \mathcal { L } _ { \mathrm { c o n } } , } \end{array}\tag{12}
$$

where λ controls the balance between language modeling and temporal contrastive supervision. In all experiments, we set λ = 0.1, which enables stable joint optimization.

## 4 Experiments

In this section, we evaluate the proposed method through extensive experiments. We first describe the experimental setup, present the main results, and then analyze key design choices and learned representations. We also provide qualitative analysis in Appendix A.5.

## 4.1 Experimental Settings

Baselines. We compare models trained with our method against four representative baselines.

InternVL3.5 (Wang et al., 2025) and LLaVA-OneVision1.5 (An et al., 2025) are recent opensource multimodal large language models; we report their 4B variants. Visual Jigsaw (Wu et al., 2026) is a 7B model tailored for temporal understanding and trained for order prediction with GRPO (Shao et al., 2024). Finally, we include Qwen3.5 (Qwen Team, 2026), the base model used in our training, and evaluate its 0.8B, 2B, and 4B variants to study performance across scales.

Datasets. For training, we use the Something-Something V2 (SSv2) (Goyal et al., 2017) training split, a large-scale dataset for fine-grained humanobject action understanding.

For evaluation, we use three benchmarks focused on temporal capability: TOMATO (Shangguan et al., 2025), TempCompass (Liu et al., 2024), and Vinoground (Zhang et al., 2024). TOMATO comprises 1,484 questions over 1,417 videos and evaluates temporal reasoning over event progression and state changes, while TempCompass contains 7,540 task instructions covering action, speed, direction, and event order. Vinoground comprises 2,000 evaluation questions derived from 500 counterfactual pairs and focuses on fine-grained video-caption alignment under subtle temporal variations. We report accuracy across all benchmarks, where higher values indicate better temporal understanding.

<table><tr><td rowspan="2">Setting</td><td rowspan="2">λ</td><td rowspan="2">TOMATO</td><td colspan="4">TempCompass</td><td colspan="3">Vinoground</td></tr><tr><td>Y/N</td><td>MCQ</td><td>Caption</td><td>Match</td><td>Text</td><td>Video</td><td>Group</td></tr><tr><td>LM only</td><td>0</td><td>29.45</td><td>72.85</td><td>68.86</td><td>62.72</td><td>78.84</td><td>36.40</td><td>22.00</td><td>9.20</td></tr><tr><td>+ VT-Contrast</td><td>0.1</td><td>31.47</td><td>74.97</td><td>70.00</td><td>64.17</td><td>80.17</td><td>41.20</td><td>25.00</td><td>11.40</td></tr><tr><td>Gain</td><td>一</td><td>+2.02</td><td>+2.12</td><td>+1.14</td><td>+1.45</td><td>+1.33</td><td>+4.80</td><td>+3.00</td><td>+2.20</td></tr></table>

Table 2: Effect of the temporal counterfactual objective. Compared with LM-only training, VT-Contrast consistently improves across benchmarks, showing the benefit of direct temporal supervision on video-token representations. Gains denote absolute improvements over the LM-only baseline.
<table><tr><td rowspan="2">Token Representation</td><td rowspan="2">TOMATO</td><td colspan="4">TempCompass</td><td colspan="3">Vinoground</td></tr><tr><td>Y/N</td><td>MCQ</td><td>Caption</td><td>Match</td><td>Text</td><td>Video</td><td>Group</td></tr><tr><td>Last query token</td><td>31.81</td><td>71.50</td><td>68.86</td><td>59.23</td><td>77.45</td><td>33.40</td><td>25.80</td><td>9.80</td></tr><tr><td>All frame tokens</td><td>30.26</td><td>73.99</td><td>70.19</td><td>62.92</td><td>79.77</td><td>40.20</td><td>21.40</td><td>9.80</td></tr><tr><td>Last-frame tokens</td><td>31.47</td><td>74.97</td><td>70.00</td><td>64.17</td><td>80.17</td><td>41.20</td><td>25.00</td><td>11.40</td></tr></table>

Table 3: Effect of token representation for temporal supervision. Last-frame video tokens achieve the best overall performance, suggesting that targeted supervision on temporally aggregated video-token representations is more effective for VT-Contrast. Bold and underlined numbers indicate the best and second-best results, respectively.

Implementation details. Our models are initialized from Qwen3.5 (Qwen Team, 2026), a recent state-of-the-art open-source model, at three scales: 0.8B, 2B, and 4B. We fine-tune them on the SSv2 (Goyal et al., 2017) training split under a video question answering formulation, using “What action is happening in the video?” as the prompt and the action label as the target response. We jointly optimize the language modeling loss and VT-Contrast loss, setting $\lambda = 0 . 1$ throughout the main text. For VT-Contrast, we set the contrastive temperature to $\tau _ { c } = 0 . 1 0$ , sample 16 negatives with replacement from permutations with Kendall tau distance in (0, 0.5], uniformly sample 8 frames per video, and use last-frame video tokens. We apply VT-Contrast to layers 21–24 out of 24 layers for the 0.8B and 2B models, and to layers 25–28 out of 32 layers for the 4B model. All models are trained for 2,250 steps using the same global batch size of 16 and a learning rate of $1 \times 1 0 ^ { - 5 }$

During evaluation, we use the lmms-eval framework (Zhang et al., 2025a), disable thinking mode, and adopt greedy decoding for deterministic results. To reduce formatting-related evaluation errors, we use prompts with standardized answer formats; details are provided in Appendix C.

## 4.2 Main Results

Table 1 reports the main results under different frame budgets. Following lmms-eval, frames are first sampled at the default FPS and capped by the specified frame budget, so the reported frame setting is an upper bound and some videos may use fewer frames. Compared with recent open-source VideoLMs, our 4B model achieves stronger results on most temporal benchmarks, with a particularly clear advantage on TOMATO.

Compared with the corresponding Qwen3.5 base models, our method brings consistent gains across most model scales and frame settings. The gains are particularly notable for the 2B model: under the 8-frame setting, it improves TOMATO by 5.73% and TempCompass Y/N by 3.95%, showing clear benefits on order-sensitive tasks. The 0.8B and 4B models also show clear improvements, suggesting that the proposed objective remains effective across model scales, with further room for scalespecific tuning. At the same time, the TempCompass Caption results suggest that some subtasks remain partly dependent on the base model capability under greedy decoding; in our results, the 2B model performs better than the 4B model on Caption and also obtains a larger gain.

## 4.3 Ablation Studies

We ablate the key designs of VT-Contrast, including the temporal counterfactual objective, targeted video-token representations, and counterfactual difficulty control. All experiments use the 2B model under the 8-frame setting. Additional analyses of layer selection, the number of negatives, and temperature are provided in Appendices A.2, A.3, and A.4, respectively.

![](images/619bfa3d57fba96fd6ff3f74abd884260b83c2ec17eee245790b0186c7c7d472.jpg)

<table><tr><td rowspan="2">Counterfactual Difficulty</td><td rowspan="2">TOMATO</td><td colspan="4">TempCompass</td><td colspan="3">Vinoground</td></tr><tr><td>Y/N</td><td>MCQ</td><td>Caption</td><td>Match</td><td>Text</td><td>Video</td><td>Group</td></tr><tr><td>Full negatives</td><td>30.86</td><td>73.62</td><td>69.94</td><td>63.22</td><td>79.71</td><td>39.20</td><td>26.60</td><td>10.80</td></tr><tr><td>Simple negatives</td><td>30.80</td><td>73.75</td><td>68.92</td><td>63.82</td><td>80.04</td><td>39.20</td><td>24.20</td><td>11.00</td></tr><tr><td>Hard negatives</td><td>31.47</td><td>74.97</td><td>70.00</td><td>64.17</td><td>80.17</td><td>41.20</td><td>25.00</td><td>11.40</td></tr></table>

Table 4: Effect of temporal counterfactual difficulty. We compare full (0, 1], simple (0.5, 1], and hard (0, 0.5] negatives based on Kendall-tau distance. The best performance from hard counterfactuals shows that Kendall-taubased grading helps select more useful temporal counterfactuals.

(a)  
![](images/9ee2d1a160a1f7efefc408ad9035d93c62f292759c6984b75f60abd2b5d8f19e.jpg)

(b)  
![](images/2fbefc18e9818942bb90475e94495dfa656d72490bf3b234a00b8b1d3cb9255a.jpg)

![](images/664d61149f7566d40c1421b25110e4d191223fe29261a75b4e03fe69ef6b6c30.jpg)  
(c)

![](images/5a5471598af109deb6e358ea1827cdb0b132717e2185389f0b779e6f3897be91.jpg)  
PC1  
Figure 4: Representation analysis of VT-Contrast. (a) Similarity distributions: VT-Contrast separates hardest negatives from positives while keeping anchor-positive similarities relatively high. (b) Distance-similarity comparison: negative similarity decreases more clearly with Kendall-tau distance after training. (c) PCA visualization: anchor, positive, and negative representations become more distinguishable after training.

Temporal counterfactual objective. Table 2 ablates the temporal counterfactual objective. Compared with LM-only training, VT-Contrast improves performance across the three benchmarks, with gains on every metric. These consistent gains suggest that the objective provides useful supervision beyond standard language modeling, with larger improvements on TOMATO and Vinoground Text. We use λ = 0.1 by default, with additional loss-weight experiments in Appendix A.1.

Targeted video-token selection. Table 3 studies where to apply VT-Contrast. Compared with supervising the last query token used for answer generation, supervising video-token representations achieves substantially better results on TempCompass and Vinoground Group. This highlights the importance of applying temporal counterfactual supervision directly in the video-token space. Among video-token choices, last-frame tokens outperform full-frame tokens overall, suggesting that the finalframe representations provide a compact target that better aggregates temporal information.

Temporal counterfactual difficulty. Table 4 ablates Kendall-tau-based negative selection. Hard negatives achieve the best overall performance across most benchmarks. Their advantage over full negatives suggests that a focused hard subset is better than mixing widely varying difficulties, while their advantage over simple negatives shows the benefit of more challenging perturbations. Together, these results demonstrate the importance of Kendall-tau-based grading for selecting effective temporal counterfactuals.

## 4.4 Representation Analysis

To understand how VT-Contrast improves temporal understanding, we analyze video-token representations on 300 SSv2 validation samples. As shown in Figure 4(a), before training, original videos and their reordered variants have very high cosine similarity and are mapped close in representation space. This supports our motivation that standard VideoLM representations are insufficiently sensitive to temporal order perturbations.

After training, reordered variants are less concentrated around the original representation, and the hardest-negative margin increases. Figure 4(b) relates anchor-negative similarity to Kendall-tau distance, showing a clearer decrease with stronger temporal perturbations. Figure 4(c) shows PCA, where negative representations become more distinguishable from anchor and positive views. Overall, these results show that same-video temporal counterfactuals provide effective supervision for order-sensitive video-token representations.

## 5 Conclusion

We propose VT-Contrast, a representation-level temporal supervision method for VideoLMs. The central idea is to shift temporal learning from response space to video-token space by contrasting order-preserving views with same-video reordered counterfactuals. Applied to targeted last-frame representations, VT-Contrast uses Kendall tau distance to grade the temporal disruption of counterfactuals, making temporal order explicit in internal representations. Experiments across temporal benchmarks show consistent overall gains, suggesting that shaping video-token representations is promising for improving temporal reasoning in VideoLMs.

## Limitations

Although VT-Contrast shows consistent improvements, our validation is limited by computational resources to moderate-scale supervised finetuning on existing VideoLMs. Its effectiveness on larger and more diverse video-text data, such as longer videos, multi-event scenarios, and broader instruction-style tasks, is worth further exploration. Since VT-Contrast provides auxiliary supervision by reshaping existing video-token representations, its gains may still depend on the temporal modeling and generation ability of the base model. This is reflected by cases where larger models do not always perform better, such as TempCompass Caption, suggesting that the gains from VT-Contrast are affected by the base model’s task-specific capability. These observations point to a promising direction of incorporating temporal counterfactual supervision into larger-scale pretraining to further strengthen VideoLMs’ temporal capability.

## Acknowledgments

This research is supported by the MOE AcRF Tier 1 Seed Grant (RS37/24, #025041-00001), Singapore.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Xiang An, Yin Xie, Kaicheng Yang, Wenkang Zhang, Xiuwei Zhao, Zheng Cheng, Yirui Wang, Songcen Xu, Changrui Chen, Didi Zhu, et al. 2025.

Llava-onevision-1.5: Fully open framework for democratized multimodal training. arXiv preprint arXiv:2509.23661.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. 2025. Qwen3- vl technical report. arXiv preprint arXiv:2511.21631.

Keshigeyan Chandrasegaran, Agrim Gupta, Lea M. Hadzic, Taran Kota, Jimming He, Cristobal Eyzaguirre, Zane Durante, Manling Li, Jiajun Wu, and Li Fei-Fei. 2024. Hourvideo: 1-hour video-language understanding. In Advances in Neural Information Processing Systems, volume 37, pages 53168–53197. Curran Associates, Inc.

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. 2020. A simple framework for contrastive learning of visual representations. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 1597–1607. PMLR.

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. 2024. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24185–24198.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261.

Ishan Dave, Rohit Gupta, Mamshad Nayeem Rizve, and Mubarak Shah. 2022. Tclr: Temporal contrastive learning for video representation. Computer Vision and Image Understanding, 219:103406.

Persi Diaconis and R. L. Graham. 1977. Spearman’s footrule as a measure of disarray. Journal of the Royal Statistical Society: Series B (Methodological), 39(2):262–268.

Michael Dorkenwald, Fanyi Xiao, Biagio Brattoli, Joseph Tighe, and Davide Modolo. 2022. Scvrl: Shuffled contrastive video representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pages 4132–4141.

Fawad Javed Fateh, Umer Ahmed, Hamza Khan, Zeeshan Zia, and Quoc-Huy Tran. 2026. Temporalvlm: Video llms for temporal reasoning in long videos. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pages 1427–1447.

Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Junfei Wu, Xiaoying Zhang, Benyou Wang, and Xiangyu Yue. 2025. Video-r1: Reinforcing video reasoning in mllms. In Advances in Neural Information Processing Systems, volume 38, Main Conference, pages 99114–99137. Curran Associates, Inc.

Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, Peixian Chen, Yanwei Li, Shaohui Lin, Sirui Zhao, Ke Li, Tong Xu, Xiawu Zheng, Enhong Chen, Caifeng Shan, and 2 others. 2025. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 24108–24118.

Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, Florian Hoppe, Christian Thurau, Ingo Bax, and Roland Memisevic. 2017. The "something something" video database for learning and evaluating visual common sense. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 5842–5850.

Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. 2020. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9729–9738.

Zi-Yuan Hu, Yiwu Zhong, Shijia Huang, Michael Lyu, and Liwei Wang. 2024. Enhancing temporal modeling of video llms via time gating. In Findings of the Associationfor Computational Linguistics: EMNLP 2024, pages 2845–2856.

Simon Jenni and Hailin Jin. 2021. Time-equivariant contrastive video representation learning. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pages 9970–9980.

Ziyan Jiang, Rui Meng, Xinyi Yang, Semih Yavuz, Yingbo Zhou, and Wenhu Chen. 2025. Vlm2vec: Training vision-language models for massive multimodal embedding tasks. In International Conference on Learning Representations, volume 2025, pages 1255–1279.

Maurice G. Kendall. 1938. A new measure of rank correlation. Biometrika, 30(1/2):81–93.

<sup>˙</sup>Ilker Kesen, Andrea Pedrotti, Mustafa Dogan, Michele Cafagna, Emre Can Acikgoz, Letitia Parcalabescu, Iacer Calixto, Anette Frank, Albert Gatt, Aykut Erdem, and Erkut Erdem. 2024. Vilma: A zero-shot benchmark for linguistic and temporal grounding in video-language models. In International Conference on Learning Representations, volume 2024, pages 40504–40554.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. 2025a. Llavaonevision: Easy visual task transfer. Trans. Mach. Learn. Res., 2025.

Lei Li, Yuanxin Liu, Linli Yao, Peiyuan Zhang, Chenxin An, Lean Wang, Xu Sun, Lingpeng Kong, and Qi Liu. 2025b. Temporal reasoning transfer from text to video. In International Conference on Learning Representations, volume 2025, pages 74070–74102.

Zeqian Li, Shangzhe Di, Zhonghua Zhai, Weilin Huang, Yanfeng Wang, and Weidi Xie. 2025c. Universal video temporal grounding with generative multimodal large language models. In Advances in Neural Information Processing Systems, volume 38, Main Conference, pages 64426–64455. Curran Associates, Inc.

Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. 2024. Video-llava: Learning united visual representation by alignment before projection. In Proceedings of the 2024 conference on empirical methods in natural language processing, pages 5971–5984.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. In Advances in Neural Information Processing Systems, volume 36, pages 34892–34916. Curran Associates, Inc.

Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. 2024. Tempcompass: Do video llms really understand videos? In Findings of the Association for Computational Linguistics: ACL 2024, pages 8731–8772.

Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Khan. 2024. Video-chatgpt: Towards detailed video understanding via large vision and language models. In Proceedings of the 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 12585– 12602.

Rui Meng, Ziyan Jiang, Ye Liu, Mingyi Su, Xinyi Yang, Yuepeng Fu, Can Qin, Raghuveer Thirukovalluru, Xuan Zhang, Zeyuan Chen, Ran Xu, Caiming Xiong, Yingbo Zhou, Wenhu Chen, and Semih Yavuz. 2026. Vlm2vec-v2: Advancing multimodal embedding for videos, images, and visual documents. Trans. Mach. Learn. Res., 2026.

Aaron van den Oord, Yazhe Li, and Oriol Vinyals. 2018. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748.

Tian Pan, Yibing Song, Tianyu Yang, Wenhao Jiang, and Wei Liu. 2021. Videomoco: Contrastive video representation learning with temporally adversarial examples. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11205–11214.

Rui Qian, Tianjian Meng, Boqing Gong, Ming-Hsuan Yang, Huisheng Wang, Serge Belongie, and Yin Cui. 2021. Spatiotemporal contrastive video representation learning. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 6964–6974.

Hangyu Qin, Junbin Xiao, and Angela Yao. 2025. Question-answering dense video events. In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 884–894.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning transferable visual models from natural language supervision. In Proceedings ofthe 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 8748–8763. PMLR.

Ziyao Shangguan, Chuhan Li, Yuxuan Ding, Yanan Zheng, Yilun Zhao, Tesca Fitzgerald, and Arman Cohan. 2025. Tomato: Assessing visual temporal reasoning capabilities in multimodal foundation models. In International Conference on Learning Representations, volume 2025, pages 7593–7734.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Yumeng Shi, Quanyu Long, Yin Wu, and Wenya Wang. 2026. Causality matters: How temporal information emerges in video language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 9006–9014.

Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. 2022. Videomae: Masked autoencoders are data-efficient learners for self-supervised video pretraining. In Advances in Neural Information Processing Systems, volume 35, pages 10078–10093. Curran Associates, Inc.

Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, Zhaokai Wang, Zhe Chen, Hongjie Zhang, Ganlin Yang, Haomin Wang, Qi Wei, Jinhui Yin, Wenhao Li, Erfei Cui, and 56 others. 2025. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. CoRR, abs/2508.18265.

Penghao Wu, Yushan Zhang, Haiwen Diao, Bo Li, Lewei Lu, and Ziwei Liu. 2026. Visual jigsaw posttraining improves mllms. In International Conference on Learning Representations, volume 2026, pages 93221–93245.

Hang Zhang, Xin Li, and Lidong Bing. 2023. Videollama: An instruction-tuned audio-visual language model for video understanding. In Proceedings of the 2023 conference on empirical methods in natural language processing: system demonstrations, pages 543–553.

Jianrui Zhang, Mu Cai, and Yong Jae Lee. 2024. Vinoground: Scrutinizing lmms over dense temporal reasoning with short videos. arXiv preprint arXiv:2410.02763.

Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, et al. 2025a. Lmms-eval: Reality check on the evaluation of large multimodal models. In Findings ofthe Association for Computational Linguistics: NAACL 2025, pages 881–916.

Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. 2025b. Long context transfer from language to vision. Trans. Mach. Learn. Res., 2025.

Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. 2025c. Llava-video: Video instruction tuning with synthetic data. Trans. Mach. Learn. Res., 2025.

Zhi Zhang, Srishti Yadav, Fengze Han, and Ekaterina Shutova. 2025d. Cross-modal information flow in multimodal large language models. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 19781–19791.

## A Additional Analysis

In this appendix, we provide additional analysis of the proposed VT-Contrast. We study the effects of the contrastive loss weight λ, targeted layer selection, the number M of negative views, and the contrastive temperature $\tau _ { c } .$ We also include qualitative examples to better illustrate the behavior of our model. Overall, these results show that VT-Contrast is relatively robust to common hyperparameter choices. Unless otherwise specified, we follow the main experimental setting and report results based on the 2B model with 8 input frames.

## A.1 Contrastive Loss Weight

![](images/b5d99413b7abb8aa8b1350e817bb4ba6cd6ea71b06d794c6899d974eadd97651.jpg)  
(a)

![](images/f755b8a459e6638a8dde26640179da1191646da029cb16fe4f7ec98c6140216d.jpg)  
(b)

![](images/3624f0eb924df453a7ced7e7f4f577be6aecf11afcfdf8b6b7046801f6abfca8.jpg)  
Figure 5: Effect of the contrastive loss weight λ. Adding VT-Contrast generally improves performance over $\lambda =$ 0, while larger weights do not always help. We use $\lambda = 0 . 1$ as the default setting, which gives the best overall performance in this ablation.

We first study the effect of the contrastive loss weight λ. For TempCompass and Vinoground, we report TempCompass Y/N and Vinoground Group because the former is highly order-sensitive and the latter is the most representative Vinoground metric. As shown in Figure 5, adding VT-Contrast improves performance over $\lambda \ : = \ : 0$ on all three benchmarks. As λ increases from 0, the stronger contrastive signal generally improves temporal discrimination. However, after a certain point, a larger weight can hurt some tasks, likely because the contrastive objective starts to interfere with the model’s general downstream performance. Therefore, we use $\lambda = 0 . 1$ as the default setting, which gives the best overall trade-off in these results.

## A.2 Targeted Layer Selection

We further study contrastive layer selection using the same three representative tasks as in Appendix A.1. As shown in Figure 6, different layer ranges lead to clearly different performance, indicating that where VT-Contrast is applied matters. A general trend is that middle-to-late layers perform better than early layers. This suggests that temporal information may be sufficiently aggregated in higher layers, while early-layer representations are still forming local visual features and are less suitable for our temporal counterfactual supervision. These results highlight the importance of targeted layer selection for obtaining strong performance. For the 2B model, we therefore apply VT-Contrast to layers 21–24. Although the best layer range may vary across model scales and deserves further exploration, later layers generally provide a strong and reliable default choice.

![](images/1b196730498ac6ba3cf29a1b6a3dcf07c6c0e9568ef9a2afd215dee7dec88665.jpg)  
(a)

![](images/699545c0c1406858349307a73cd077a219ea35e9c77e919b8762d7927303becc.jpg)  
(b)

![](images/71e28f4a333647079d64e9f44c729c949e02cb383d9bdb83fb13d9d20f3545cd.jpg)  
(c)  
Figure 6: Effect of targeted layer selection. Different layer ranges show different effects, highlighting the importance of targeted layer selection. For the 2B model, we use layers 21–24, which perform best across the representative temporal tasks.

## A.3 Number of Counterfactual Negatives

Table 5 studies the number of temporal counterfactual negatives. The number of negatives controls how many temporally perturbed same-video views are compared with each anchor in the InfoNCE objective. With only 4 negatives, the contrastive signal is limited, leading to weaker overall performance. Using 8 negatives already gives competitive results on several tasks, suggesting that moderate negative diversity is helpful. Further increasing the number to 16 provides richer temporal counterfactual comparisons and leads to the most stable results. Therefore, we use 16 counterfactual negatives as the default setting.

## A.4 Contrastive Temperature

Table 6 studies the sensitivity to the contrastive temperature $\tau _ { c } .$ In the InfoNCE objective, $\tau _ { c }$ controls the sharpness of the similarity distribution: smaller values make the objective more sensitive to similarity differences, while larger values produce smoother contrastive signals. The results show that VT-Contrast is relatively robust to this hyperparameter, with different temperatures giving comparable performance across most tasks. Among them, $\tau _ { c } = 0 . 1 0$ provides a strong trade-off, performing best on several tasks while remaining competitive on others. A smaller temperature sharpens the contrastive distribution, which can help some tasks but also leads to less stable results. By contrast, a larger temperature smooths the distribution and weakens the separation signal, leading to weaker overall performance. We therefore use $\tau _ { c } = 0 . 1 0$ as the default setting.

<table><tr><td rowspan="2">Negative Number</td><td rowspan="2">TOMATO</td><td colspan="4">TempCompass</td><td colspan="3">Vinoground</td></tr><tr><td>Y/N</td><td>MCQ</td><td>Caption</td><td>Match</td><td>Text</td><td>Video</td><td>Group</td></tr><tr><td>4 samples</td><td>30.93</td><td>73.79</td><td>69.56</td><td>62.18</td><td>79.37</td><td>38.40</td><td>23.00</td><td>10.80</td></tr><tr><td>8 samples</td><td>32.21</td><td>73.42</td><td>69.43</td><td>64.32</td><td>79.97</td><td>39.00</td><td>23.80</td><td>10.20</td></tr><tr><td>16 samples</td><td>31.47</td><td>74.97</td><td>70.00</td><td>64.17</td><td>80.17</td><td>41.20</td><td>25.00</td><td>11.40</td></tr></table>

Table 5: Effect of the number M of counterfactual negatives. Using 4 negatives provides a relatively weak temporal counterfactual signal, while using 8 negatives already gives competitive results on several tasks. Increasing to 16 negatives further improves overall stability and achieves the best overall performance.
<table><tr><td>Temperature</td><td>TOMATO</td><td colspan="4">TempCompass</td><td colspan="3">Vinoground</td></tr><tr><td></td><td></td><td>Y/N</td><td>MCQ</td><td>Caption</td><td>Match</td><td>Text</td><td>Video</td><td>Group</td></tr><tr><td>0.06</td><td>29.99</td><td>73.58</td><td>70.51</td><td>64.62</td><td>79.24</td><td>43.60</td><td>23.40</td><td>11.20</td></tr><tr><td>0.08</td><td>31.27</td><td>73.54</td><td>69.68</td><td>63.67</td><td>80.17</td><td>39.20</td><td>24.60</td><td>11.60</td></tr><tr><td>0.10</td><td>31.47</td><td>74.97</td><td>70.00</td><td>64.17</td><td>80.17</td><td>41.20</td><td>25.00</td><td>11.40</td></tr><tr><td>0.12</td><td>30.39</td><td>74.56</td><td>69.68</td><td>62.57</td><td>79.91</td><td>40.60</td><td>23.40</td><td>11.40</td></tr><tr><td>0.14</td><td>29.65</td><td>73.05</td><td>70.13</td><td>63.72</td><td>79.31</td><td>40.60</td><td>22.60</td><td>10.80</td></tr></table>

Table 6: Sensitivity to the contrastive temperature $\tau _ { c } .$ Different temperatures yield comparable results across most tasks, indicating that VT-Contrast is not highly sensitive to this hyperparameter. We use $\tau _ { c } = 0 . 1 0$ by default, which provides a strong overall trade-off.

## A.5 Qualitative Analysis

Figure 7 presents qualitative examples from Temp-Compass Y/N, where each case contains an original video and its reversed counterpart. We choose this setting because the original and reversed videos share nearly the same visual content, while the questions are designed to test whether the model can distinguish changes in event direction and temporal order. As shown in the examples, Qwen3.5 is less sensitive to the temporal change between the original and reversed videos. It fails on fine-grained temporal transitions, including the child’s turning direction, paper deformation, and the apple’s visual state change, while our method adapts more reliably to the reversed temporal semantics. These cases suggest that VT-Contrast improves sensitivity to temporal progression and reduces reliance on static appearance cues.

## B Additional Experiments

In this appendix, we provide additional experiments to further evaluate VT-Contrast beyond the main settings. We evaluate VT-Contrast with InternVL3.5 (Wang et al., 2025), which belongs to a different VideoLM family, assess VT-Contrast on general video QA, and report the training overhead introduced by the contrastive objective.

## B.1 Evaluation with InternVL3.5

To further assess the generalizability of VT-Contrast across different VideoLM families, we evaluate our method on InternVL3.5 (Wang et al., 2025), beyond the Qwen3.5 model family considered in our main experiments. Specifically, we use InternVL3.5-4B and train VT-Contrast with 8- frame inputs, while adapting the configuration by setting the contrastive loss weight to λ = 0.12, applying the objective to layers 29–32, and reducing the number of negatives to M = 12 for training efficiency. As in our main experiments, we use a single model trained with 8-frame inputs and evaluate it at inference time using 8, 16, and 32 sampled frames. Results in Table 7 show that VT-Contrast improves most metrics across different frame settings, further supporting its effectiveness beyond the Qwen3.5 model family.

## B.2 Evaluation on General Video QA

We further evaluate VT-Contrast on general video QA to examine whether its benefits extend beyond temporally focused downstream tasks. For this purpose, we adopt Video-MME (Fu et al., 2025), a comprehensive benchmark for evaluating general video understanding. We use Qwen3.5-2B as the base model and train it for 1,500 steps on a subset of the open-ended QA data from LLaVA-Video-178K (Zhang et al., 2025c). Compared with the action-centric SSv2 data used in our main experiments, this setting provides more general supervision, allowing us to assess whether VT-Contrast also benefits broader video understanding. Results in Table 8 show that our method improves the overall Video-MME accuracy. These results further demonstrate the applicability of VT-Contrast to broader video QA settings.

![](images/043bce400fdda0d957e387dd297731daa15c41f95ec46cd1370429c4b6356ff1.jpg)

![](images/2fa75d1cb14f0b0d214f4aa11d702f744624a6b1f6fc98fa9bd105eae7f65bed.jpg)

![](images/8a45a5d62a6f3a64cca1df071411d81332c18fe582a65e30f137c362ff157dfe.jpg)

Figure 7: Qualitative examples on TempCompass Y/N with reversed video pairs. Each example contains an original video and its reversed counterpart, which share similar visual content but require different answers. Compared with Qwen3.5, our method better adapts its predictions to the changed temporal order.
<table><tr><td rowspan="2">Frames Model</td><td rowspan="2"></td><td rowspan="2">TOMATO</td><td colspan="4">TempCompass</td><td rowspan="2">Vinoground Group</td></tr><tr><td>Y/N</td><td>MCQ</td><td>Caption</td><td>Match</td></tr><tr><td rowspan="2">8</td><td>InternVL3.5-4B</td><td>28.71</td><td>72.08</td><td>68.73</td><td>60.93</td><td>79.44</td><td>16.00</td></tr><tr><td>Ours-4B</td><td>30.26+1.55</td><td>72.40+0.32</td><td>70.06+1.33</td><td>63.67+2.74</td><td>79.44+0.00</td><td>16.60+0.60</td></tr><tr><td rowspan="2">16</td><td>InternVL3.5-4B</td><td>30.12</td><td>72.89</td><td>70.70</td><td>61.68</td><td>80.44</td><td>16.80</td></tr><tr><td>Ours-4B</td><td>30.19+0.07</td><td>73.38+0.49</td><td>71.52+0.82</td><td>64.32+2.64</td><td>79.51 -0.93</td><td>18.20+1.40</td></tr><tr><td rowspan="2">32</td><td>InternVL3.5-4B</td><td>29.65</td><td>73.46</td><td>70.76</td><td>62.03</td><td>80.71</td><td>19.40</td></tr><tr><td>Ours-4B</td><td>30.05+0.40</td><td>73.95+0.49</td><td>71.90+1.14</td><td>65.32+3.29</td><td>80.71 +0.00</td><td>19.40+0.00</td></tr></table>

Table 7: Results of VT-Contrast on InternVL3.5-4B under different frame settings. Small colored numbers indicate changes over the InternVL3.5-4B baseline. VT-Contrast improves most metrics across 8, 16, and 32 frames, further supporting its effectiveness beyond the Qwen3.5 model family.

<table><tr><td>Method</td><td colspan="3">Frames</td></tr><tr><td></td><td>8</td><td>16</td><td>32</td></tr><tr><td>Qwen3.5-2B</td><td>52.07</td><td>54.93</td><td>57.85</td></tr><tr><td>Ours-2B</td><td>52.67</td><td>56.07</td><td>58.30</td></tr><tr><td>Gain</td><td>+0.60</td><td>+1.14</td><td>+0.45</td></tr></table>

Table 8: Results on general video QA with Video-MME, reporting overall accuracy. Gains denote absolute improvements over the Qwen3.5-2B baseline, showing the effectiveness of our method for general video QA.

## B.3 Training Overhead

We additionally report the training overhead introduced by VT-Contrast. Since each training sample includes an anchor view, a positive view, and multiple counterfactual negatives, the method requires additional computation compared with standard training. We profile Qwen3.5 models on an 8-H100-GPU server with a per-device batch size of 1 and a fixed effective global batch size of 16, comparing M = 0, 4, 8, 16 negatives. For each configuration, we conduct five 40-step runs, discard the first 10 steps of each run as warm-up, compute statistics over the remaining 30 steps, and report the average across the five runs. As shown in Table 9, increasing M introduces additional training overhead as expected. These results characterize the practical training cost of VT-Contrast. Since our current implementation is not specifically optimized for efficiency, further improvements may be possible with additional optimization.

## C Prompt Details

For reproducibility, we list the prompts used for benchmark evaluation. Across all reported models and settings, we use the same prompt format within each benchmark to reduce prompt-induced variation. For each task, we present a prompt example illustrating the input format.

<table><tr><td></td></tr><tr><td>Answer:</td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr></table>

<table><tr><td rowspan="2">Model Metric</td><td rowspan="2"></td><td colspan="4">Number of Negatives M</td></tr><tr><td>0</td><td>4</td><td>8</td><td>16</td></tr><tr><td rowspan="3">0.8B</td><td>Time (s)</td><td>1.83</td><td>5.09</td><td>5.26</td><td>6.28</td></tr><tr><td>Alloc. (GiB)</td><td>9.73</td><td>11.35</td><td>11.90</td><td>13.86</td></tr><tr><td>Reserved (GiB)</td><td>15.32</td><td>16.44</td><td>17.28</td><td>19.23</td></tr><tr><td rowspan="3">2B</td><td>Time (s)</td><td>1.94</td><td>5.42</td><td>5.75</td><td>7.09</td></tr><tr><td>Alloc. (GiB)</td><td>24.84</td><td>26.54</td><td>26.94</td><td>29.56</td></tr><tr><td>Reserved (GiB)</td><td>39.53</td><td>40.86</td><td>42.88</td><td>45.46</td></tr><tr><td rowspan="3">4B</td><td>Time (s)</td><td>2.57</td><td>7.05</td><td>8.18</td><td>10.77</td></tr><tr><td>Alloc. (GiB)</td><td>51.36</td><td>52.38</td><td>53.52</td><td>57.39</td></tr><tr><td>Reserved (GiB)</td><td>70.88</td><td>72.82</td><td>74.99</td><td>77.40</td></tr></table>

Table 9: Training overhead of VT-Contrast. M = 0 denotes standard training without the contrastive objective. Time denotes wall-clock time per training step, while Alloc. and Reserved denote peak allocated and reserved CUDA memory, respectively.

## C.1 TOMATO

TOMATO is evaluated as a multiple-choice task. Following the default prompt in the lmms-eval framework, we further constrain the model to answer with a single option letter. This reduces overly open-ended responses and simplifies the subsequent evaluation process in practice. An example prompt is shown below.

You will be provided with n separateframes uniformly sampledfrom a video, theframes are provided in chronological order of the video. Analyze these frames and provide the answer to the question about the video content. Answer the multiple-choice question about the video content.

You must use theseframes to answer the multiple-choice question; do not rely on any externel knowledge or commonsense.

In which direction(s) did the person’s hand move? </question>

<options>   
‘A’: ‘Not moving at all’, ‘B’: ‘Left.’, ‘C’: ‘Right.’, ‘D’:   
‘First to the right then to the left.’, ‘E’: ‘First to the left   
then to the right.’   
</options>

Even if the information in these separate frames is not enough to answer the question, PLEASE TRY YOUR BEST TO GUESS AN ANSWER WHICH YOU THINK WOULD BE THE MOST POSSIBLE ONE BASED ON THE QUESTION.

DO NOT GENERATE ANSWER SUCH AS ‘NOT POSSI-BLE TO DETERMINE.’

Your output must be exactly one uppercase option letter from A, B, C, D, E.

## C.2 TempCompass

For TempCompass, we use task-specific prompts for Yes or No, Multi Choice, Captioning, and Caption Matching. As with TOMATO, we explicitly specify the required answer format when applicable to ensure consistent evaluation across models. For Captioning, we follow the lmms-eval default and use GPT-3.5-Turbo for GPT-based grading. Example prompts are provided below.

Yes or No: Is the man dunking? Please answer yes or no:

Multi Choice:   
What is the man doing in the video?   
A. dunking a basketball   
B. dribbling a basketball   
C. passing a basketball   
Please only give the letter of the best option from A, B,   
C: Captioning:   
You will be presented with a video and several pieces of information. One piece ofinformation is consistent with the video while the others are not. Please identify the information that consistent with the video and generate a video caption accordingly.   
Information A: ‘subject’: ‘man’, ‘action’: ‘dribbling a basketball’   
Information B: ‘subject’: ‘man’, ‘action’: ‘dunking a basketball’   
Information C: ‘subject’: ‘man’, ‘action’: ‘passing a basketball’   
Generated Caption: Caption Matching:   
Which description is a more suitable matchfor the video? Option 1: The man is dribbling a basketball.   
Option 2: A man is dunking a basketball.   
Please only give the label ofthe best optionfrom Option 1, Option 2:

## C.3 Vinoground

For Vinoground, we use separate prompts for textto-video and video-to-text matching. Depending on the matching direction, the model chooses between two candidate videos or captions and outputs only the corresponding option letter. Example prompts are shown below.

Text:   
Which caption best describes this video?   
A. a toddler picks up a water bottle and drinks before he   
plays around the grassfield

B. a toddler plays around the grassfield before he picks up a water bottle and drinks

Answer with the option’s letter from the given choices directly. Please only output one English character.

## Video:

Which video segment matches this caption? Note: The video contains two segments separated by a 2-second blackframe. Caption: a toddler plays around the grass field before he picks up a water bottle and drinks

A. First segment (before black frame)

B. Second segment (after blackframe)

Answer with the option’s letter from the given choices directly. Please only output one English character.