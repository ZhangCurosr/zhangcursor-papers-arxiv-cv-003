# RELIABILITY-GUIDED VISION-LANGUAGE DECODING WITH TEXT-AWARE SKIP REFINEMENT FOR MEDICAL IMAGE SEGMENTATION

Sivan Talaei<sup>1</sup>, Fatemeh Daneshfar<sup>2,∗</sup>, Mustaqeem Khan<sup>3</sup>, Abdulhady Abas Abdullah<sup>4</sup>, Mourad Oussalah<sup>5</sup>

<sup>1,2</sup> Department of Computer Engineering, University of Kurdistan, Sanandaj, Kurdistan, Iran <sup>3</sup> College of Information Technology, United Arab Emirates University, Al Ain, UAE <sup>4</sup> Artificial Intelligence and Innovation Centre, University of Kurdistan, Erbil, Kurdistan, Iraq <sup>5</sup>Center for Machine Vision and Signal Analysis, University of Oulu, Oulu, Finland

## ABSTRACT

Medical image segmentation remains challenging when targets exhibit limited visual distinction, irregular structure, or substantial variation across imaging modalities. Medical text can reduce this ambiguity, but indiscriminate fusion may introduce unreliable guidance. We present RELiF, a vision–language framework that integrates text through reliability-guided decoding. Its Reliability-guided Fusion Gate (RFG) adaptively controls cross-modal updates, while the Text-guided Skip Refinement Adapter (TSRA) refines high-resolution features using text guidance. RELiF is evaluated on radiographic, CT, and endoscopic segmentation tasks, where it shows competitive performance across diverse imaging characteristics and target appearances. Additional analyses examine efficiency and stability. The complementary uncertainty representation derived from the reliability response also shows an association with prediction difficulty, with higher uncertainty values occurring more frequently in pixels associated with segmentation errors. Code: https://github.com/SeivanTalaie/RELiF.

Index Terms— Medical image segmentation, Vision– language learning, Multimodal fusion

## 1. INTRODUCTION

Medical image segmentation supports clinical assessment, treatment planning, and disease monitoring by identifying anatomical structures and pathological regions. Unimodal approaches, such as U-Net [1], rely solely on visual information, which can limit their effectiveness when targets exhibit subtle appearances or substantial variation across imaging conditions. To address these challenges, multimodal approaches incorporate complementary semantic information from medical text, building on advances in vision–language learning such as CLIP [2]. However, the benefits of textual guidance depend not only on the information it provides but also on how effectively it is integrated with visual features.

![](images/21d48ca7f543030ac6f462d8f8143123407bf49113df3a3264d6f82c4524b21f.jpg)  
Fig. 1. RFG behavior under controlled image and text degradation on QaTa-COV19. Lines show the mean gate response across the test set and three RFG stages, while bars indicate Dice gains (pp) over ungated fusion.

Recent studies have explored cross-modal reconstruction [3], uncertainty-aware learning [4], and bidirectional fusion [5] to improve vision–language segmentation. However, how to adaptively regulate textual guidance across different visual regions during decoding remains an open question, as its relevance may vary spatially. Another limitation lies in conventional skip connections, which directly transfer high-resolution features without considering their relevance, potentially introducing domain-specific variations into the decoder [6]. These challenges call for selective cross-modal fusion and context-aware refinement of skip features.

To address these challenges, we propose RELiF, a vision– language framework that combines reliability-guided decoding with text-guided skip refinement. Its Reliability-guided Fusion Gate (RFG) estimates token-wise reliability from visual and text-conditioned features, together with their featurelevel agreement and discrepancy, to adaptively regulate crossmodal contributions during decoding. As shown in Fig. 1, RFG adapts its fusion response to variations in image quality and textual relevance, with corresponding Dice gains over ungated fusion. In parallel, the Text-guided Skip Refinement Adapter (TSRA) uses textual context to refine high-resolution skip features before decoder fusion. The main contributions are as follows:

![](images/9116e165d622f7470157694916f50641bd896e7bfd1071146dcea142c40586ad.jpg)  
Fig. 2. Overview of RELiF: (a) vision–language segmentation architecture. (b) Text-guided Skip Refinement Adapter (TSRA). (c) Reliability-guided Decoding Attention Block (RDAB) with its Reliability-guided Fusion Gate (RFG).

• We introduce RELiF, whose Reliability-guided Fusion Gate (RFG) within the Reliability-guided Decoding Attention Block (RDAB) adaptively regulates cross-modal fusion using token-wise reliability derived from visual and text-conditioned features and their feature-level similarities and differences.

• We develop the Text-guided Skip Refinement Adapter (TSRA) to preserve fine structural details through textconditioned channel modulation and lightweight refinement of high-resolution skip features.

• We evaluate RELiF on chest X-ray and lung CT infection segmentation, as well as endoscopic polyp segmentation, demonstrating its effectiveness across diverse imaging modalities and target appearances.

• We conduct additional experiments and analyses to assess RELiF’s effectiveness, reliability, efficiency, and stability.

## 2. METHODOLOGY

## 2.1. Overview

As shown in Fig. 2, visual and textual features guide progressive decoding. RDABs regulate cross-modal fusion, while TSRAs operate on separate skip pathways, refining encoder features before their integration into decoder blocks. Finally, subpixel upsampling restores the spatial resolution, and the segmentation head generates the final pixel-wise predictions.

## 2.2. Reliability-guided Fusion Gate (RFG)

Given medical images $\mathbf { I } \in \mathbb { R } ^ { B \times C _ { I } \times H \times W }$ and tokenized descriptions T, ConvNeXt-Tiny [7] extracts visual features $\mathbf { I } _ { s } ^ { \prime }$ across four encoder stages, $s \in \{ 1 , 2 , 3 , 4 \}$ , while BioViL CXR-BERT [8] generates text embeddings $\dot { \mathbf { T } } ^ { \prime } \in \mathbb { R } ^ { B \times L \times D _ { T } }$ Here, B denotes batch size, $C _ { I }$ the image channels, H, W the spatial dimensions, and $L , D _ { T }$ the text length and embedding dimension. Decoding starts from the deepest visual features $\mathbf { I } _ { 4 } ^ { \prime } ,$ , reshaped into $\mathbf { V } \in \mathbb { R } ^ { B \times N _ { 4 } \times C _ { 4 } }$ , where $N _ { 4 } ~ = ~ H _ { 4 } W _ { 4 }$ is the number of visual tokens. Meanwhile, text embeddings pass through a learnable linear layer, LeakyReLU, SSMixer, and GELU to produce $\mathbf { T } _ { p } ~ \in ~ \mathbb { R } ^ { \mathbf { \bar { B } } \times L _ { p } \times C _ { 4 } }$ , where $L _ { p }$ is the resulting sequence length. SSMixer models dependencies among textual tokens to provide context-aware representations at the current visual scale. Within RDAB, normalized and position-encoded visual tokens undergo multi-head self-attention. The output is normalized and added to V through a residual connection, yielding contextual features $\mathbf { V } _ { c } .$ Cross-attention then uses normalized, position-encoded $\mathbf { V } _ { c }$ as queries, position-encoded $\mathbf { T } _ { p }$ as keys, and $\mathbf { T } _ { p }$ as values to generate text-conditioned features C. To assess their consistency with visual information, RFG combines both representations and their feature-level relationships:

$$
\mathbf { A } = \operatorname { C o n c a t } \left( \mathbf { V } _ { c } , \mathbf { C } , | \mathbf { V } _ { c } - \mathbf { C } | , \mathbf { V } _ { c } \odot \mathbf { C } \right) .\tag{1}
$$

A lightweight reliability head then estimates a response for each visual token:

$$
\mathbf { R } = \sigma \left( \operatorname { L i n e a r } \left( \operatorname { G E L U } \left( \operatorname { L i n e a r } ( \operatorname { L N } ( \mathbf { A } ) ) \right) \right) \right) .\tag{2}
$$

The complementary uncertainty representation is defined as ${ \bf U } \ = \ 1 - { \bf R }$ , with higher values indicating lower estimated reliability. The reliability response subsequently regulates cross-modal fusion through

$$
\mathbf { G } = 0 . 5 + \mathbf { R } , \qquad \mathbf { Y } = \mathbf { V } + \alpha ( \mathbf { G } \odot \mathbf { C } ) ,\tag{3}
$$

where ⊙ denotes element-wise multiplication, σ the sigmoid function, and α a learnable scalar. The bounded gate $\mathbf { G } \in [ 0 . 5 , 1 . 5 ]$ regulates cross-modal contributions without entirely suppressing or excessively amplifying them.

## 2.3. Text-guided Skip Refinement Adapter (TSRA)

While RFG regulates cross-modal fusion, TSRA refines highresolution skip features $\mathbf { I } _ { s } ^ { \prime }$ for $s \in \{ 1 , 2 , 3 \}$ to preserve relevant spatial details. The shared text embedding $\mathbf { T } ^ { \prime }$ is averaged across L tokens to obtain $\mathbf { t } _ { g } ,$ which is normalized and projected into a stage-specific channel response. A sigmoid activation and a 0.5 shift are then applied, followed by reshaping to enable channel-wise modulation of the spatial skip features, yielding $\overline { { \mathbf { g } } } _ { s } \in [ 0 . 5 , 1 . 5 ] ^ { B \times C _ { s } \times 1 \times 1 }$

$$
\begin{array} { r } { \overline { { \bf g } } _ { s } = \mathrm { R e s h a p e } \left( 0 . 5 + \sigma \left( \mathrm { L i n e a r } ( \mathrm { L N } ( { \bf t } _ { g } ) ) \right) \right) . } \end{array}\tag{4}
$$

In parallel, a depthwise $3 \times 3$ convolution extracts local spatial details from $\mathbf { I } _ { s } ^ { \prime } ,$ followed by batch normalization and GELU. A pointwise $1 \times 1$ convolution then combines information across channels, followed by another normalization to produce $\mathbf { D } _ { s }$ . The text-conditioned gate $\overline { { \mathbf { g } } } _ { s }$ modulates these features through a scaled residual connection:

$$
\widetilde { \mathbf { I } } _ { s } ^ { \prime } = \operatorname { G E L U } \left( \mathbf { I } _ { s } ^ { \prime } + \beta \left( \overline { { \mathbf { g } } } _ { s } \odot \mathbf { D } _ { s } \right) \right) \in \mathbb { R } ^ { B \times C _ { s } \times H _ { s } \times W _ { s } } .\tag{5}
$$

Here, $\beta$ is a learnable residual scale, and ⊙ denotes element-wise multiplication. Zero-initialized gate projections and small initial residual scales allow gradual refinement. Finally, the refined skip features $\widetilde { \mathbf { I } } _ { s } ^ { \prime }$ are fused with the upsampled decoder features without additional supervision.

## 2.4. Decoder Block

The decoder progressively restores spatial details through three stages, $s \in \{ 4 , 3 , 2 \}$ . At each stage, RDAB uses the shared text embedding T<sup>′</sup> to produce reliability-guided visual tokens $\mathbf { Y } _ { s } \in \mathbb { R } ^ { B \times N _ { s } ^ { \smile } \times C _ { s } }$ , where $N _ { s } = H _ { s } W _ { s }$ denotes the token count and $C _ { s }$ the channel dimension. These tokens are reshaped into spatial features $\mathbf { Z } _ { s } \in \mathbb { R } ^ { B \times C _ { s } \times H _ { s } \times W _ { s } }$ and upsampled through transposed convolution. The resulting features are concatenated with the corresponding TSRA-refined skip features $\widetilde { \mathbf { I } } _ { s - 1 } ^ { \prime }$ and convolved:

$$
\mathbf D _ { s - 1 } = { \mathrm { C o n v B l o c k } } \left( { \mathrm { C o n c a t } } \left( { \mathrm { U p C o n v } } ( \mathbf Z _ { s } ) , \widetilde { \mathbf I } _ { s - 1 } ^ { \prime } \right) \right)\tag{6}
$$

The fused features are converted back into visual tokens for the next RDAB, progressively incorporating finer spatial

information. Finally, subpixel upsampling restores the input resolution, and a sigmoid-activated segmentation head generates pixel-wise foreground probabilities:

$$
\widehat { \bf M } = \sigma \left( \mathrm { S e g H e a d } \left( \mathrm { S u b P i x e l U p } ( { \bf D } _ { 1 } ) \right) \right) .\tag{7}
$$

## 2.5. Objective Function

RELiF is trained using the Spectral-Entropic Uncertainty (SEU) objective [4], which combines three complementary loss terms. Dice loss maximizes overlap between the predicted mask $\widehat { \bf M }$ and ground truth M, while spectral loss encourages structural consistency by comparing their Fourier magnitude spectra. An entropy-inspired term further promotes confident foreground predictions. The overall objective is defined as

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { D i c e } } + \lambda _ { F } \mathcal { L } _ { \mathrm { S p e c t r a l } } + \lambda _ { E } \mathcal { L } _ { \mathrm { E n t r o p y } } ,\tag{8}
$$

where $\lambda _ { F }$ and $\lambda _ { E }$ control the contributions of the spectral and entropy terms, respectively. The entropy-inspired term is applied only to the final segmentation probabilities and is separate from the RFG-derived uncertainty representation. RFG and TSRA are learned end-to-end without direct supervision.

## 3. EXPERIMENTS AND RESULTS

## 3.1. Datasets

Experiments were conducted on three public datasets: QaTa-COV19 [9], MosMedData+ [10], and Kvasir-SEG [11], covering chest X-ray, CT, and endoscopic segmentation. Following the LViT protocol [12], we adopt the same dataset splits and use text annotations from [12] for infection segmentation and [13] for polyp segmentation. The dataset splits are: QaTa-COV19: Training=5716; Validation=1429; Test=2113 MosMedData+: Training=2183; Validation=273; Test=273 Kvasir-SEG: Training=800; Validation=100; Test=100.

## 3.2. Implementation Details

The framework was implemented in PyTorch, with each dataset trained independently on an NVIDIA RTX 4090 GPU. All input images and masks were resized to 224 × 224 pixels. Training was conducted for 200 epochs, using batch sizes of 32 for training and 8 for validation and testing. We used the AdamW optimizer with a cosine annealing learning-rate scheduler. Initial learning rates were $3 \times 1 0 ^ { - 4 }$ for QaTa-COV19 and Kvasir-SEG and $5 \times 1 0 ^ { - 4 }$ for MosMedData+. The SEU loss weights were set to $\lambda _ { F } = 0 . 3$ and $\lambda _ { E } = 0 . 1$

## 3.3. Experimental Results and Analysis

Table 1 compares RELiF with representative monomodal and multimodal methods across three medical segmentation datasets. RELiF achieves the highest Dice and mIoU scores on QaTa-COV19 and Kvasir-SEG, while maintaining competitive performance on MosMedData+ with only 18.1 G FLOPs. Qualitative comparisons in Fig. 3 further illustrate the segmentation results across all three datasets. To assess robustness under limited supervision, Table 2 compares RE-LiF with BiCLIP and LViT using 50% and 10% training data. RELiF achieves the best Dice and mIoU on two datasets at 50% and all three at 10%. We further evaluated stability across five independent training runs, obtaining Dice standard deviations of 0.12, 0.14, and 0.24 percentage points on QaTa-COV19, MosMedData+, and Kvasir-SEG, respectively, indicating low run-to-run variability. The ablation study in Table 4 confirms the complementary contributions of RFG and TSRA, with their combination achieving the best performance. Furthermore, removing textual guidance or introducing mismatched text during training reduces segmentation accuracy, highlighting the importance of relevant textual information. Table 3 further evaluates the ability of RFG-derived uncertainty to distinguish erroneous pixels from correct predictions, reporting AUROC values of 0.7724– 0.8473 and AUPRC scores 3.04–5.51 times their prevalencebased random baselines. These results suggest consistent error-discrimination ability across the three datasets. Finally, we conducted a pixel-level risk–coverage analysis on Kvasir-SEG by progressively excluding high-uncertainty pixels. At 90% coverage, retained-pixel error risk decreased by 45.2%, with selective Dice reaching 93.87%. Further reducing coverage to 70% increased risk reduction to 71.7%, although selective Dice declined, potentially due to the exclusion of correctly predicted foreground pixels.

Table 1. Comparison with state-of-the-art methods. best and second-best results are bold and underlined, respectively.
<table><tr><td rowspan="2">Method</td><td colspan="2">QaTa-COV19</td><td colspan="2">MosMedData+</td><td colspan="2">Kvasir-SEG</td></tr><tr><td>Dice(%)</td><td>mIoU(%)</td><td>Dice(%)</td><td>mIoU(%)</td><td>Dice(%)</td><td>mIoU(%)</td></tr><tr><td>Monomodal models</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>U-Net (&#x27;15) [1]</td><td>78.91</td><td>69.32</td><td>64.30</td><td>50.50</td><td>82.33</td><td>74.26</td></tr><tr><td>nnUNet (&#x27;21) [14]</td><td>80.30</td><td>70.62</td><td>72.32</td><td>60.14</td><td>84.22</td><td>75.41</td></tr><tr><td>UCTransNet (&#x27;22) [15]</td><td>79.00</td><td>69.34</td><td>65.71</td><td>52.55</td><td>91.04</td><td>87.31</td></tr><tr><td>Multimodal models</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LViT (&#x27;23) [12]</td><td>83.40</td><td>74.89</td><td>74.32</td><td>61.03</td><td>87.59</td><td>75.16</td></tr><tr><td>RecLMIS (&#x27;25) [3]</td><td>84.93</td><td>76.86</td><td>77.26</td><td>64.95</td><td>86.58</td><td>77.08</td></tr><tr><td>ProLearn (&#x27;25) [16]</td><td>90.60</td><td>82.81</td><td>77.82</td><td>63.69</td><td>90.40</td><td>82.80</td></tr><tr><td>UA-VLS (&#x27;26) [4]</td><td>90.11</td><td>82.00</td><td>78.48</td><td>64.58</td><td>92.41</td><td>85.89</td></tr><tr><td>BiCLIP (&#x27;26) [5]</td><td>90.60</td><td>82.81</td><td>81.00</td><td>68.06</td><td>89.02</td><td>80.21</td></tr><tr><td>RELiF (Ours)</td><td>91.54</td><td>84.41</td><td>79.27</td><td>65.66</td><td>93.28</td><td>87.41</td></tr></table>

![](images/105d45d5513d0c00600a4619d3d263c8d794181cc8c8c342a950172bacde2ea7.jpg)  
Fig. 3. Qualitative segmentation results across three datasets.

Table 2. Segmentation performance with 50% and 10% of the training data across three datasets.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Regime</td><td colspan="2">QaTa-COV19</td><td colspan="2">MosMedData+</td><td colspan="2">Kvasir-SEG</td></tr><tr><td>Dice(%)</td><td>mIoU(%)</td><td>Dice(%)</td><td>mIoU(%)</td><td>Dice(%)</td><td>mIoU(%)</td></tr><tr><td rowspan="2">BiCLIP [5]</td><td>50%</td><td>89.83</td><td>81.53</td><td>79.72</td><td>66.27</td><td>86.55</td><td>76.29</td></tr><tr><td>10%</td><td>87.16</td><td>77.25</td><td>69.23</td><td>52.95</td><td>71.72</td><td>55.91</td></tr><tr><td rowspan="2">LViT [12]</td><td>50%</td><td>78.81</td><td>69.55</td><td>58.98</td><td>47.28</td><td>80.66</td><td>73.30</td></tr><tr><td>10%</td><td>74.72</td><td>64.99</td><td>51.07</td><td>39.59</td><td>55.68</td><td>44.38</td></tr><tr><td rowspan="2">RELiF</td><td>50%</td><td>91.10</td><td>83.66</td><td>78.06</td><td>64.01</td><td>91.95</td><td>85.10</td></tr><tr><td>10%</td><td>88.29</td><td>79.04</td><td>74.72</td><td>59.64</td><td>85.17</td><td>74.17</td></tr></table>

Table 3. Uncertainty-based pixel error detection using RFGderived uncertainty response. Error prevalence defines the random AUPRC baseline, while AUPRC lift measures gains.
<table><tr><td>Dataset</td><td>Error prevalence(%)</td><td>AUROC ↑</td><td>AUPRC ↑</td><td>Random AUPRC</td><td>AUPRC lift ↑</td></tr><tr><td>QaTa-COV19</td><td>2.03</td><td>0.7724</td><td>0.0824</td><td>0.0203</td><td>4.06×</td></tr><tr><td>MosMedData+</td><td>0.65</td><td>0.8389</td><td>0.0199</td><td>0.0065</td><td>3.04×</td></tr><tr><td>Kvasir-SEG</td><td>1.88</td><td>0.8473</td><td>0.1035</td><td>0.0188</td><td>5.51×</td></tr></table>

Table 4. Ablation study of RELiF evaluates individual components and textual guidance strategies using Dice and mIoU.
<table><tr><td rowspan="2">Method</td><td colspan="2">MosMedData+</td><td colspan="2">Kvasir-SEG</td></tr><tr><td>Dice(%)</td><td>mIoU(%)</td><td>Dice(%)</td><td>mIoU(%)</td></tr><tr><td colspan="5">Component Ablation</td></tr><tr><td>Base</td><td>78.48</td><td>64.58</td><td>92.37</td><td>85.82</td></tr><tr><td>Base + RFG</td><td>78.92</td><td>65.18</td><td>92.89</td><td>86.72</td></tr><tr><td>Base + TSRA</td><td>79.06</td><td>65.37</td><td>92.91</td><td>86.76</td></tr><tr><td>Full RELiF (RFG + TSRA)</td><td>79.27</td><td>65.66</td><td>93.28</td><td>87.41</td></tr><tr><td colspan="5">Textual Guidance</td></tr><tr><td>Training w/o Text Guidance</td><td>77.78</td><td>63.64</td><td>88.90</td><td>80.02</td></tr><tr><td>Training w/ Mismatched Text</td><td>78.73</td><td>64.92</td><td>91.01</td><td>83.50</td></tr></table>

## 4. CONCLUSION

This paper introduced RELiF, combining reliability-guided fusion (RFG) and text-conditioned skip refinement (TSRA) for medical segmentation. The results suggest benefits from complementary mechanisms under limited supervision. Sensitivity to mismatched text indicates that gains depend on relevant guidance rather than text alone. RFG-derived uncertainty helps identify unreliable regions, although excessive rejection may remove correctly segmented foreground. These findings motivate further reliability-aware segmentation research.

## 5. COMPLIANCE WITH ETHICAL STANDARDS

The study retrospectively used open-access human data, with no ethical approval required per the dataset license.

## 6. ACKNOWLEDGMENTS

This study received no funding, and the authors have no relevant conflicts of interest to disclose.

## 7. REFERENCES

[1] Olaf Ronneberger, Philipp Fischer, and Thomas Brox, “U-Net: Convolutional networks for biomedical image segmentation,” in Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015. 2015, vol. 9351 of Lecture Notes in Computer Science, pp. 234–241, Springer.

[2] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever, “Learning transferable visual models from natural language supervision,” in Proceedings of the 38th International Conference on Machine Learning. 2021, vol. 139 of Proceedings of Machine Learning Research, pp. 8748–8763, PMLR.

[3] Xiaoshuang Huang, Hongxiang Li, Meng Cao, Long Chen, Chenyu You, and Dong An, “Cross-modal conditioned reconstruction for language-guided medical image segmentation,” IEEE Transactions on Medical Imaging, vol. 44, no. 4, pp. 1821–1835, Apr. 2025.

[4] Aryan Das, Tanishq Rachamalla, Koushik Biswas, Swalpa Kumar Roy, and Vinay Kumar Verma, “Uncertainty-aware vision-language segmentation for medical imaging,” in 2026 IEEE/CVF Winter Conference on Applications ofComputer Vision (WACV), 2026, pp. 8490–8499.

[5] Saivan Talaei, Fatemeh Daneshfar, Abdulhady Abas Abdullah, and Mustaqeem Khan, “BiCLIP: Bidirectional and consistent language-image processing for robust medical image segmentation,” arXiv preprint arXiv:2603.00156, 2026.

[6] Frauke Wilm, Jonas Ammeling, Mathias Ottl, Rutger <sup>¨</sup> H. J. Fick, Marc Aubreville, and Katharina Breininger, “Rethinking u-net skip connections for biomedical image segmentation,” arXiv preprint arXiv:2402.08276, 2024.

[7] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie, “A convnet for the 2020s,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 11976–11986.

[8] Benedikt Boecking, Naoto Usuyama, Shruthi Bannur, Daniel C. Castro, Anton Schwaighofer, Stephanie Hyland, Maria Wetscherek, Tristan Naumann, Aditya Nori,

Javier Alvarez-Valle, Hoifung Poon, and Ozan Oktay, “Making the most of text semantics to improve biomedical vision–language processing,” in Computer Vision – ECCV 2022, Cham, Oct. 2022, vol. 13696 of Lecture Notes in Computer Science, pp. 1–21, Springer Nature Switzerland.

[9] Aysen Degerli, Serkan Kiranyaz, Muhammad E. H. Chowdhury, and Moncef Gabbouj, “Osegnet: Operational segmentation network for covid-19 detection using chest x-ray images,” in 2022 IEEE International Conference on Image Processing (ICIP), 2022, pp. 2306–2310.

[10] Sergey P Morozov, Anna E Andreychenko, Nikolay A Pavlov, AV Vladzymyrskyy, Natalya V Ledikhova, Victor A Gombolevskiy, Ivan A Blokhin, Pavel B Gelezhe, AV Gonchar, and V Yu Chernina, “Mosmeddata: Chest ct scans with covid-19 related findings dataset,” arXiv preprint arXiv:2005.06465, 2020.

[11] Debesh Jha, Pia H Smedsrud, Michael A Riegler, Pal˚ Halvorsen, Thomas De Lange, Dag Johansen, and Havard D Johansen, “Kvasir-seg: A segmented polyp ˚ dataset,” in International conference on multimedia modeling. Springer, 2019, pp. 451–462.

[12] Zihan Li, Yunxiang Li, Qingde Li, Puyang Wang, Dazhou Guo, Le Lu, Dakai Jin, You Zhang, and Qingqi Hong, “Lvit: Language meets vision transformer in medical image segmentation,” IEEE Transactions on Medical Imaging, vol. 43, no. 1, pp. 96–107, 2024.

[13] Kanchan Poudel, Manish Dhakal, Prasiddha Bhandari, Rabin Adhikari, Safal Thapaliya, and Bishesh Khanal, “Exploring transfer learning in medical image segmentation using vision-language models,” arXiv preprint arXiv:2308.07706, 2023.

[14] Fabian Isensee, Paul F. Jaeger, Simon A. A. Kohl, Jens Petersen, and Klaus H. Maier-Hein, “nnU-Net: A selfconfiguring method for deep learning-based biomedical image segmentation,” Nature Methods, vol. 18, no. 2, pp. 203–211, 2021.

[15] Haonan Wang, Peng Cao, Jiaqi Wang, and Osmar R. Zaiane, “UCTransNet: Rethinking the skip connections in U-Net from a channel-wise perspective with transformer,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 36, no. 3, pp. 2441–2449, June 2022.

[16] Shuchang Ye, Usman Naseem, Mingyuan Meng, and Jinman Kim, “Alleviating textual reliance in medical language-guided segmentation via prototype-driven semantic approximation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 22316–22326.