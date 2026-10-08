# Perceptually Aligned Evaluation of Style Transfer

Yang Deng<sup>1</sup>, Eleftherios Ioannou<sup>2</sup>, David Mould<sup>3</sup>, Steve Maddock<sup>4</sup>, Paul L. Rosin<sup>1</sup>, and Yu-Kun Lai<sup>1</sup>

<sup>1</sup>Cardiff University, UK

<sup>2</sup>The Cyprus Institute, Cyprus

<sup>3</sup>Carleton University, Canada

<sup>4</sup>University of Sheffield, UK

## Abstract

Style transfer lacks a reliable evaluation standard: ground truth is inherently ill-defined, and existing automatic metrics often fail to reflect human preference. This paper introduces ASTRA (Assessment of Style TRansfer Algorithms), an approach for automatic evaluation of style transfer algorithms; it contains two components, ASTRA-Data and ASTRA-Score. ASTRA-Data consists of a benchmark image set of content and style references, a collection of style transfer results generated on the benchmark set, and user study data capturing human judgements through a two-stage pairwise comparison protocol. From these annotations, we derive ranking-based ground truth for content preservation, style fidelity, and overall preference. Based on ASTRA-Data, we construct ASTRA-Score, a learnt evaluator that predicts preference-aligned scores from content-stylestylization image triplets, enabling automatic and scalable evaluation ofnew models applied to the benchmark set. Experimental results demonstrate that ASTRA-Score achieves substantially higher correlation with human rankings compared to prior metrics. Overall, ASTRA establishes a robust mechanismfor standardised evaluation ofstyle transfer methods.

## CCS Concepts

• Computing methodologies → Non-photorealistic rendering; Image-based rendering; • Human-centered computing → Visualization design and evaluation methods;

## 1. Introduction

Style transfer aims to generate an image that preserves the content of one image while adopting the style of another [JSO<sup>∗</sup>01]. The field has evolved from low-level image processing [RL10, LR12, RL13, RL15, RWW<sup>∗</sup>17] to modern neural style transfer (NST) methods [GEB16, LLH<sup>∗</sup>21, HB17, PL19, DTD<sup>∗</sup>22] and has enabled a wide range of applications, including artistic image generation [GLK<sup>∗</sup>17, WYZ<sup>∗</sup>25], photo editing [HB17], semantic segmentation [ZZL<sup>∗</sup>22], and data augmentation for downstream vision tasks [JAB<sup>∗</sup>19].

Evaluating stylisation quality is fundamentally challenging [RLM<sup>∗</sup>22], yet reliable evaluation is vital to advancing style transfer research. Style transfer lacks well-defined ground truth; in some cases, there are multiple perceptually valid outputs for a single content–style pair, while in others, it is not clear what if any output would qualify. Consequently, evaluation is intrinsically ill-posed [RLM<sup>∗</sup>22, MR17]. Although metrics have progressed from low-level similarity measures to high-level representations [GEB16, HRU<sup>∗</sup>17, ZIE<sup>∗</sup>18, WO22], pretrained vision and vision-language models [DBK<sup>∗</sup>20, CTM<sup>∗</sup>21, BBC<sup>∗</sup>23] still treat many legitimate stylistic transformations as errors. Artistic styles often involve structural abstraction or reconfiguration [KSM<sup>∗</sup>19], and in such cases, existing metrics are not reliable indicators. User studies remain the de facto standard [LSZ<sup>∗</sup>25, GSL<sup>∗</sup>25] despite their cost, limited scalability, and poor reproducibility [BHZ<sup>∗</sup>23].

Evaluation is further constrained by weakly standardised test data. Early work relied on a few handpicked examples [GEB16], while later studies drew from general-purpose datasets that were not intended for stylisation assessment. More systematic benchmarks have been proposed [MR17, RWW<sup>∗</sup>17, RLM<sup>∗</sup>22, CSC<sup>∗</sup>22, RGC<sup>∗</sup>24], but typically provide image collections without assessment protocols. In summary, the field still lacks standardised, perceptually aligned frameworks for evaluating artistic style transfer.

This paper advances style transfer evaluation by presenting AS-TRA, Assessment of Style TRansfer Algorithms. ASTRA contains both a benchmark dataset (ASTRA-Data, a curated dataset of style and content images) and an automated evaluation protocol (ASTRA-Score, a trained network that can estimate style transfer performance on the benchmark set). This combined benchmark and automated assessment constitutes a flexible tool for evaluating style transfer algorithms: practitioners can apply their proposed process to all combinations of style and content images in the benchmark, and use ASTRA-Score to generate numerical evaluations for each result image. These individual scores can either be aggregated or examined independently to discern particular strengths and weaknesses with respect to the benchmark coverage.

![](images/0190d68d5eecf62e3c777db729a61db5679139a18f8e853a64edc99245348f68.jpg)  
Figure 1: Examples of selected images produced by different algorithms.

To construct this framework, we assembled a diverse collection of painterly style images capturing a representative range of historical practices and visual elements, alongside a carefully selected set of content images. By combining these with a group of influential and varied style transfer algorithms, we produced a comprehensive set of style-content-algorithm triples, yielding output result images for evaluation.

These result images provided the raw material for a large-scale user study using a two-alternative-forced-choice design. Subjects evaluated pairs based on three criteria: (i) which result better captured the style; (ii) which better preserved the content; and (iii) which was overall a more satisfactory style transfer outcome. We deliberately avoided asking which result was more appealing overall, as general aesthetics are largely orthogonal to style transfer fidelity and can be confounded by preferred colors, content, or external visual elements. We refer to this full collection (style and content images, output images, and pairwise human judgments) as ASTRA-Data.

Using rank centrality to derive global rankings from these human judgments, we trained ASTRA-Score to predict stylisation effectiveness given a style image, content image, and transfer result. A leave-one-out experimental study indicates that ASTRA-Score achieves substantially greater alignment with human judgement than past automated methods such as ArtFID [WO22], SRQE [CSC<sup>∗</sup>22], and Gram Loss [GEB16].

This paper contains two main contributions, corresponding to ASTRA’s core components:

• ASTRA-Data, the collection of style images, content images, style transfer results, and associated human judgements, establishing a rigorous benchmark for style transfer evaluation.

• ASTRA-Score, a network trained on ASTRA-Data to predict style transfer effectiveness on the benchmark image set, allowing reliable automated assessment of performance on the benchmark.

## 2. Related Work

Evaluation in style transfer remains an open problem. The field lacks a unified approach that jointly defines evaluation data, human judgement, and quantitative metrics [IM24]. Existing work typically combines manually selected test images, small-scale user studies, and proxy-based automatic metrics. These components are often designed independently, with limited reproducibility and inconsistent comparisons across methods [MR17, ZTZ 24].

## 2.1. Evaluation Dataset for Style Transfer.

Some researchers evaluated results on a small set of manually chosen content–style pairs [GEB16]. Later studies drew from standard datasets, often sampling content images from MS COCO [LMB<sup>∗</sup>14] and artworks from WikiArt [MK18], with additional datasets such as KTH [KTH<sup>∗</sup>13], BBST4M [RGC<sup>∗</sup>24], and Behance [WFJ<sup>∗</sup>17] used to increase diversity. However, these datasets were primarily designed for general visual understanding rather than stylisation evaluation. In practice, they are used with ad-hoc sampling strategies, leading to unsystematic exploration of content–style combinations, poor coverage of stylistic diversity, and a lack of consistency across studies. These limitations have motivated the development of dedicated evaluation datasets, such as curated NPRgeneral [MR17] and human-annotated datasets like AST-IQAD [CSC<sup>∗</sup>22], While AST-IQAD [CSC<sup>∗</sup>22] introduces human-annotated evaluation for stylisation quality and explicitly constructs controlled content–style pairs, its protocol is still based on a fixed set of pre-selected content–style pairs rather than a cross content-style evaluation design that independently probes content and style factors. Its subjective comparisons remain confined to this fixed pool of generated outputs, making it difficult to disentangle pair-specific difficulty from general method performance. In addition, the final objective evaluator is constructed from sparse handcrafted feature similarity rather than learned directly from human judgements.

## 2.2. Evaluation Methods

Early work primarily relied on qualitative side-by-side comparisons of stylised outputs generated from selected content–style pairs [GEB16]. While intuitive, such comparisons are subjective and are unstable with respect to the examples provided. User studies have been widely adopted to assess perceptual quality [Mou14], typically measuring preferences related to content preservation, style resemblance, and overall quality. Such studies are expensive and difficult to scale. Various automatic metrics have been introduced to provide complementary quantitative signals. Content preservation is often measured using perceptual similarity metrics such as LPIPS [ZIE<sup>∗</sup>18] or feature similarity derived from pretrained vision models including CLIP [RKH<sup>∗</sup>21], Vision Transformers [DBK<sup>∗</sup>20], and DINO [CTM<sup>∗</sup>21]. Distribution-based measures such as FID [HRU<sup>∗</sup>17] and ArtFID [WO22] have also been adapted for stylisation evaluation. Additional metrics target stylisation quality or perceptual fidelity, including CFSD [CHH24], SRQE [CSC<sup>∗</sup>22]; some recent evaluation $[ \mathrm { W Y Z ^ { * } 2 5 } ]$ uses large vision–language models such as Qwen [BBC<sup>∗</sup>23]. Another possibility is learned evaluators that decompose stylisation quality into perceptual components such as aesthetic quality, content preservation, and style resemblance [SGG<sup>∗</sup>24,CSMJ24]. However, these models are typically trained on proxy signals or limited annotated datasets, and their alignment with human judgement remains limited.

## 3. Construction of ASTRA-Data

In this section, we describe the construction of ASTRA-Data, including the selection of style images in Section 3.1, content images in Section 3.2, and representative stylisation methods in Section 3.3. We selected 12 style references, 6 content images, and 12 representative methods, resulting in a total of 864 stylised outputs covering all content–style–method combinations. This dataset is small enough to permit human evaluations of a substantial fraction of possible pairs, from which global rankings can be computed.

![](images/00814830c3b49aa441a87b2bb9c3256cfa94a5eda912ec213be39532b9466a3c.jpg)

(a) Stage 1 evaluation interface.  
![](images/fc30c9a50fa181858018d2e9ef4f1e17e8bcba0ed00a99fd9b23dac7b6e107ca.jpg)  
(b) Stage 2 evaluation interface.  
Figure 2: Examples of the two evaluation interfaces used in our user study. (Top) In Stage 1, participants compare two stylised results generated from the same content-style pair by different methods, together with the corresponding content and style references. (Bottom) In Stage 2, participants compare two stylised results that come from different content-style pairs.

## 3.1. Style Image Selection

A vast array of image styles are possible. In order to balance completeness of coverage with diversity of styles across a small number of images, we opted to use only paintings, and exclude many other worthwhile art media and styles. That said, we took an expansive view in considering the eligibility of a given image, and our final set includes images such as Hokusai’s Great Wave which is a woodcut print.

We designed our image set around the three dimensions of content, colourfulness, and detail. Content refers to the subject matter being depicted. Colourfulness refers to both the range of colours present in the image and the vividness of individual colours. Detail refers to the presence of intentional fine-scale structure in the depiction of the subject matter, as opposed to more simplified or abstracted depictions that deliberately omit detail. Organising our image search along these dimensions helped us systematically choose an image set with a broad range and distributed sampling of painterly characteristics.

One might question why we chose to include content as a dimension of style when we have a separate set of content images. A few reasons explain this. Historical painting genres and traditions are often based on content (e.g., portraits, landscapes), and elements and motifs used in style creation can be particular to the content being depicted. On a more technical note, many style transfer methods make use of spatial statistics (e.g., summarised in Gram matrices), and varying the image content helps obtain a broader sampling of styles. We chose three categories of content: people, encompassing both portraits and images where humans occupy significant portions of the canvas; nature, including landscapes and other natural scenes; and artificial, meaning images that are primarily about constructed or manufactured subjects. We extended “artificial” to include abstract images as well.

Secondary considerations subsequently shaped the image set subject to sampling the defined dimensions. We sought representation across a range of historical periods and geographic or cultural contexts. Public-domain availability was a mandatory requirement. We made sure to include works containing distinct motifs or recognisable stylistic markers, as these features provide clearer signals for evaluating style transfer performance. Finally, we endeavoured to incorporate a range of difficulty levels for neural style transfer methods. Some images exhibit relatively straightforward painterly characteristics, such as textured brushwork without deliberate geometric distortion. At the same time, others present significant challenges, including rigid geometric structures, distinctive visual elements, or stylisations that deviate sharply from naturalistic representation. This diversity ensures that the evaluation can capture both the strengths and limitations of existing algorithms and will remain relevant as more capable methods emerge in the future. Discussion of individual style images appears in Appendix A.

## 3.2. Content Image Selection

We opted to draw from the already-curated NPRgeneral [MR17] image set. These twenty images cover varied subject matter and visual elements, and collectively offer a wide range of challenges to NPR methods. Unfortunately, it is impractical to use the full NPRgeneral set: the combinatorial explosion of style, content, and algorithm demands that we keep each category small. We selected six images, taking two images for each of the three content categories already established for style: people, nature, and artificial.

Within the constraints implied by the above, we then chose a set of images that maximised coverage of the image properties described for the NPRgeneral images: colourful, high contrast, long gradients, and many more. This process led us to the six images: athletes and mac for people; berries and daisy for nature; angel and barn for artificial.

## 3.3. Method Selection

There is a large literature on neural style transfer (we estimate 500–1000 published methods including all minor variations). To choose a manageable set of methods, we restricted the choice to papers with publicly available code, with a preference for influential papers, along with Multimodal Large Language Models for their generative capabilities different from NST methods. We began with an initial pool of 20 candidate methods: AAMS [YRX<sup>∗</sup>19], AdaAttN [LLH<sup>∗</sup>21], AAST [HJL<sup>∗</sup>20],AdaIN [HB17], AesUST [WZZ<sup>∗</sup>22], ArtFlow [AHS<sup>∗</sup>21], CCPL [WZDB22], DiffuseST [HZG24], Gatys [GEB16], Gemini [TAB<sup>∗</sup>23], Chat-GPT [AAA<sup>∗</sup>23], IEContraAST [CWZ<sup>∗</sup>21], Linear [LLKY19], OmniStyle [WLL<sup>∗</sup>25], RAST [MZLB23], SANET [PL19],

Table 1: Number of appearances of each method in Stage 1 and Stage 2 pairwise comparisons.
<table><tr><td>Method</td><td>Stage 1</td><td>Stage 2</td><td>Total</td></tr><tr><td>AAMS [YRX*19]</td><td>2886</td><td>2712</td><td>5598</td></tr><tr><td>AdaAttN [LLH*21]</td><td>2983</td><td>2854</td><td>5837</td></tr><tr><td>AdaIN [HB17]</td><td>3026</td><td>2845</td><td>5871</td></tr><tr><td>ArtFlow [AHS*21]</td><td>2960</td><td>2855</td><td>5815</td></tr><tr><td>ChatGPT [AAA*23]</td><td>2836</td><td>2771</td><td>5607</td></tr><tr><td>Gatys [GEB16]</td><td>3052</td><td>2859</td><td>5911</td></tr><tr><td>LLIP</td><td>2869</td><td>2887</td><td>5756</td></tr><tr><td>OmniStyle [WLL*25]</td><td>2933</td><td>2888</td><td>5821</td></tr><tr><td>SANET [PL19]</td><td>2903</td><td>2815</td><td>5718</td></tr><tr><td>StyTR-2 [DTD*22]</td><td>3034</td><td>2741</td><td>5775</td></tr><tr><td>USO [WHC*25]</td><td>2919</td><td>2883</td><td>5802</td></tr><tr><td>NST-Ghiasi [GLK*17]</td><td>2735</td><td>2970</td><td>5705</td></tr></table>

StyTR-2 [DTD<sup>∗</sup>22], StyleID [CHH24], USO [WHC<sup>∗</sup>25], NST-Ghiasi [GLK<sup>∗</sup>17]). To this, we added a suite of handcrafted algorithms based on low-level image processing (LLIP) [JSO<sup>∗</sup>01, TM98,FXDG17,TFO23,RL17,BVFAB22,XLXJ11,LXJ12,BK69, KLC07, PKD07], one per target style; details appear in Appendix A.

To finalise the method selection from the candidate pool, we compute pairwise distances between methods using clean-FID [PZZ22]. For each style, clean-FID is computed between the outputs of two methods under the same content–style conditions, and the final distance is obtained by averaging across all styles. For each method, we then compute the mean distance to all other methods as a coarse measure of its distinctiveness. We were then able to exclude methods that did not offer enough variety, as measured by low pairwise distances. This step aims to reduce redundancy while retaining representative approaches. Following this process, we retain a final set of 10 NST methods: AAMS [YRX<sup>∗</sup>19], AdaAttN [LLH<sup>∗</sup>21], AdaIN [HB17], ArtFlow [AHS<sup>∗</sup>21], Gatys [GEB16], OmniStyle [WLL<sup>∗</sup>25], SANET [PL19], StyTR-2 [DTD<sup>∗</sup>22], USO [WHC<sup>∗</sup>25], NST-Ghiasi [GLK<sup>∗</sup>17], plus ChatGPT [AAA<sup>∗</sup>23] and the handmade LLIP methods. This process ensured a high level of diversity among the methods. Examples of selected images in ASTRA-Data produced by different algorithms are shown in Fig. 1.

## 4. Annotation and Preference Analysis

We conducted a two-stage online user study on 864 stylised images obtained from our data selection process, with the goal of constructing a human-centred preference dataset. These pairwise judgements were subsequently aggregated using Rank Centrality [NOS17], enabling us to recover a global ranking over all stylised images as well as corresponding preference scores for content preservation, style similarity, and overall quality. The user study was conducted under approval from the relevant institutional research ethics committee. Participants were recruited through open channels, resulting in 407 submissions. After applying quality control procedures, 264 valid responses were retained for analysis.

Participants were compensated upon successful completion of the study.

This section presents the design and execution of the two-stage user study (Section 4.1), followed by the procedures used to ensure data quality and annotation reliability (Section 4.2). We then examine the adequacy of the collected pairwise comparisons for stable global ranking through simulation (Section 4.3). Building on the validated annotations, we describe the construction of preference scores from raw pairwise data (Section 4.4) and subsequently analyse the resulting human preference patterns (Section 4.5).

## 4.1. Two-Stage Study Design and Interface

In Stage 1, we perform controlled pairwise comparisons between different methods under the same content–style pair, isolating the effect of the generation method and yielding reliable within-pair preferences. Our design follows the two-alternative forced choice (2AFC) protocol, which provides precise and consistent measurements of aesthetic preferences in style transfer [So23]. Compared to absolute evaluations such as Likert-scale assessments of image quality, 2AFC admits finer gradations to be expressed, and eliminates problems arising from scale inconsistency among participants. The “forced choice” aspect ensures that participants provide a judgement for every pair, and cannot prematurely abandon the decision by selecting a deliberately excluded “no preference” option. In Stage 1, we collected 17,568 votes from 122 participants.

Although each method appears in every content–style group, Stage 1 comparisons are local (within the same content-style pairs), preventing a globally consistent ranking for method comparison and metric evaluation. We collect data for global evaluations in Stage 2, where participants judge pairwise comparisons across different content–style pairs. These cross-category comparisons enable a coherent global ranking over all stylised images. The Stage 2 comparisons are perceptually more challenging than those in Stage 1, so each questionnaire in Stage 2 contains fewer tasks. In Stage 2, we collected 17,040 votes from 142 participants, resulting in 34,608 total votes. The user interface is illustrated in Figure 2.

## 4.2. User Study Protocol and Quality Control

The user study first introduced the task and explained the style transfer concept, with examples and sample tasks. Participants read an illustrated page explaining the key concepts of content, style, and style transfer using text and visual examples. They then completed a short interactive tutorial with representative examples to familiarise themselves with the evaluation criteria and question format (see details in Appendix B). On completing the tutorial, the participant begins answering questions about style transfer results from different algorithms. Their answers constitute the user preference data which we record and later analyse.

In both Stage 1 and Stage 2 studies, each comparison required participants to answer three questions: (1) which stylisation result looks more like an ideal combination of the content and the style? (2) which result better captures the style? and (3) which result better preserves the content?

In Stage 1, each comparison is obtained by selecting a random style image, a random content image, and two random algorithms; the reference content and style images are shown to the user together with the two stylised results. Thus each comparison comprises three questions based on a fixed content-style pair. Multiple comparisons within a questionnaire are randomly sampled without repetition. This setup yields reliable within-group preferences under controlled conditions. Stage 2 consisted of cross-group comparisons, connecting different content-style pairs. Each question contained two randomly sampled stylised results together with their respective content and style references, yielding six images in total, $\mathrm { i . e . , } \left( c _ { 1 } , s _ { 1 } , m _ { 1 } \right)$ and $\left( c _ { 2 } , s _ { 2 } , m _ { 2 } \right)$ . The user was asked to evaluate which of m or m better reflected the criteria. For content, is m closer to $c _ { 1 }$ than m<sub>2</sub> is to $c _ { 2 } ?$ For style, is $m _ { 1 }$ closer to $s _ { 1 }$ than m<sub>2</sub> is to $s _ { 2 } ?$ For overall preference, does $m _ { 1 }$ better reflect the $\left( c _ { 1 } , s _ { 1 } \right)$ style transfer than m<sub>2</sub> does for $( c _ { 2 } , s _ { 2 } ) ?$

Annotation consistency. To further assess annotation reliability, we re-annotated a randomly selected subset of comparisons using additional independent annotators. In Stage 1, Cohen’s κ values were 0.663, 0.547, and 0.547 for content, style, and overall preference, respectively. In Stage 2, the corresponding values were 0.515, 0.389, and 0.416. These results show meaningful but taskdependent agreement, with stronger consistency in Stage 1 and reduced agreement in the more demanding cross-pair comparisons of Stage 2. Further details are provided in Appendix B.

## 4.3. Two-Stage Ranking Stability Simulation

We analyse whether the number of pairwise comparisons in our two-stage study was sufficient to support stable ranking estimation. Spectral ranking theory [NOS17, CS15] suggests that reliable recovery scales as Θ(n log n) under models such as Bradley-Terry-Luce [BT52, L<sup>∗</sup>59], providing a theoretical reference for the required comparison scale. We further validate this empirically via simulation using Rank Centrality [NOS17]. For Stage 1, ranking stability rapidly converges and exceeds $\rho = 0 . 9 9$ , where $\rho$ denotes the Spearman rank correlation between the estimated ranking and the reference ranking, at approximately $1 . 2 \times 1 0 ^ { 4 }$ comparisons. For Stage 2, global ranking accuracy improves steadily with additional cross-group comparisons, reaching $\rho \approx 0 . 9 5$ at around $1 . 4 \times 1 0 ^ { 4 }$ comparisons. In practice, our study collected 17,568 (Stage 1) and 17,040 (Stage 2) votes, both of which lie well within the stable regime. See more details in Appendix C.

Empirical graph diagnostics. We additionally examine the topology of the comparison graph constructed from the actual collected annotations. As expected, Stage 1 alone forms 72 disconnected components corresponding to the 12 style references and 6 content references. The cross-group comparisons introduced in Stage 2 connect these components, yielding a single connected comparison graph over all 864 stylised results. The combined graph has an average node degree of approximately 45.4. The Rank Centrality transition matrices for content, style, and overall preference have spectral gaps of 0.0199, 0.0421, and 0.1000, respectively. We further perform 100 stratified bootstrap resamplings of the collected comparisons; all resampled graphs remain connected and Rank Centrality can be successfully computed in every case. Together with the ranking-recovery simulation above, these diagnostics confirm that the global rankings are not obtained from disconnected comparison components and remain stable under resampling. Further connectivity and bootstrap analyses are provided in Appendix C.

## 4.4. Preference Score Estimation

To obtain reliable preference scores, we combine pairwise comparisons from both stages of the user study. Stage 1 provides comparisons across methods under identical (content, style) conditions, while Stage 2 connects different content–style pairs to enable meaningful global rankings.

Participants provide pairwise preferences under three criteria (overall, style, content), defining three separate ranking problems. Our goal is to recover a global preference ranking for each criterion. Each generated image is treated as a player:

$$
p = ( { \mathrm { s t y l e } } , { \mathrm { c o n t e n t } } , { \mathrm { m e t h o d } } ) .\tag{1}
$$

Ranking from Pairwise Comparisons. Given these pairwise comparisons between players, we adopt Rank Centrality [NOS17] to estimate global rankings. Rank Centrality is a spectral method that interprets comparison outcomes as transitions of a Markov chain and recovers rankings from its stationary distribution.

Rank Centrality Formulation. Let w<sub>ij</sub> denote the number of times player i defeats player j, and

$$
n _ { i j } = w _ { i j } + w _ { j i }\tag{2}
$$

be the total number of comparisons between the pair. Following [NOS17], we construct a transition matrix P for a Markov chain where the transition probability from player i to player j is

$$
\begin{array} { r } { P _ { i j } = \left\{ \begin{array} { l l } { \frac { 1 } { d _ { \mathrm { m a x } } } \frac { w _ { j i } } { n _ { i j } } , } & { n _ { i j } > 0 , i \ne j , } \\ { 1 - \sum _ { k \ne i } P _ { i k } , } & { i = j , } \end{array} \right. } \end{array}\tag{3}
$$

where d<sub>max</sub> denotes the maximum degree of the comparison graph. Intuitively, the transition probability is proportional to the empirical probability that player j defeats player i. The stationary distribution π of this Markov chain satisfies

$$
\pi = \pi P ,\tag{4}
$$

and the resulting stationary probability $\pi _ { i }$ is taken as the preference score of player i. Players with larger stationary probabilities correspond to having more preferred stylisation results.

## 4.5. Preliminary Data Analysis

Raw Pairwise Data. Based on the collected data, we compute the frequency with which each stylisation method appears in the pairwise comparison tasks. After filtering incomplete responses and failed attention checks, we aggregate the occurrences of each method across all valid comparisons. Tab. 1 summarises the number of appearances in Stage 1 and Stage 2. Overall, the exposure of different methods is approximately balanced. In Stage 1, each method appears about 2.7k–3.1k times, while in Stage 2 the counts range from 2.6k to 3.0k. Similarly, each style reference participates in roughly 200–320 comparisons per method, and each content image appears about 400–560 times per method (see more details in Appendix D). This balanced distribution ensures that the pairwise comparison data are well distributed across methods, styles, and content, preventing the final preference estimation from being dominated by a small subset of references.

![](images/41e2dbacc483f73ffaa2b0875b2f2e7a3cb80ffb85284978e53428655bcfb388.jpg)  
Figure 3: Top-5 (top row) and bottom-5 (bottom row) stylised results selected from the top-10 and bottom-10 samples ranked by their mean Rank Centrality (RC) scores for overall preference. The style references for each column, arranged from left to right and top to bottom, are: Kandinsky, Carr, Picasso, Ptolemaic Egyptian, Chinese landscape, Australian Aboriginal, Australian Aboriginal, Rembrandt, Hokusai, and Picasso.

![](images/82b9c1ca79f9395dc93c6155ba3811949207fa657efea9d4710a463be016d6c7.jpg)

![](images/472f6b8832222df3b2c4369ab8ca00f0839dc0f687a90e8811fbdd8187e460c1.jpg)  
(a) Overall vs Content  
(b) Overall vs Style  
Figure 4: Scatter plots of (a) overall effectiveness vs. content preservation; (b) overall effectiveness vs. style fidelity.

For each method m, we compute preference rates from the votes: the style win rate, content win rate, and overall win rate, defined as the proportion of comparisons in which the stylised result generated by m is chosen in the corresponding question. Fig. 4 and Fig. 5 show scatter plots of combinations of these quantities for all methods: overall vs. content, overall vs. style, and content vs. style.

Fig. 4 (a) shows little pattern: participants’ assessment of the overall effectiveness of style transfer was not strongly influenced by content preservation. In contrast, Fig. 4 (b) shows a strong positive correlation between overall effectiveness of style transfer and style fidelity. Together, these two scatter plots indicate that style fidelity is the most critical aspect of the style transfer task, with content preservation less clearly contributing.

Fig. 5 helps us assess tradeoffs between style similarity and content preservation. An idealised method will exhibit no tradeoff: both content and style can be preserved. However, in practice, tradeoffs do exist, with some methods favouring style, some methods favouring content. Furthermore, the tradeoffs are style-specific: some restrictive styles, such as Mondrian, are not conducive to presenting photographic content, while painterly styles aiming at greater realism will allow both style and content to be captured in the output.

![](images/f88851b53f49131f0d82581e6847b7d20c934aac5abddc1b84194d4a58cb22d4.jpg)  
Figure 5: Trade-off between style similarity and content preservation measured from Stage 1 pairwise comparisons. The dashed curve indicates the Pareto frontier of the two objectives, which highlights the optimal trade-off boundary between the two criteria.

In practice, there is substantial tension between the two objectives: methods achieving stronger style similarity (e.g., Gatys [GEB16], SANET [PL19]) tend to exhibit lower content fidelity, while methods emphasising content preservation (e.g., AdaAttN [LLH<sup>∗</sup>21], StyTR-2 [DTD<sup>∗</sup>22]) typically produce weaker stylistic effects.

![](images/e32595764b1be0694cff2bfe612586888ede365637ab3a001e38188f613521f7.jpg)  
(a) Mondrian Style

![](images/7996c7975ae5e1c7e312b10f86df0d46dcdf3aba153ff7aee1d49ae5f7594308.jpg)

![](images/0e23a794fadd39720870551d13f430fcb9bcb2d21c6b0e07c2a13d033f548199.jpg)  
(c) Rembrandt Style

(b) Kandinsky Style  
![](images/76c496cee04a81919826e880416e402fde2fc9bf49e0146becb867fb7d38e7d4.jpg)  
(d) Bierstadt Style  
Figure 6: Content–style preference trade-off under different style references.

To further examine whether this trade-off depends on style or content, we compute the same statistics per style and per content, and visualise the resulting distributions (see Appendix D). The overall trend remains largely consistent across different content images. In contrast, the behaviour varies substantially across styles.

Abstract styles (Figs. 6 (a,b)) possess a clear inverse relationship between style and content preferences, where methods achieving strong stylistic effects often incur significant structural distortion and hence have reduced content preservation.

Not all styles demand such tradeoffs. Indeed, in Fig. 6 (c) we can see a positive correlation between content and style preferences, indicating that better-performing algorithms can preserve both style and content effectively. Conversely, Fig. 6 (d) lacks any clear trend. The degree to which tradeoffs are present remains style-dependent.

Preference Scores. As illustrated in Fig. 7, Gatys [GEB16] achieves the highest Mean Rank Centrality score in overall preference, though it exhibits significant variance across different stylecontent pairs, as indicated by its wide error bar. In contrast, while newer methods like AdaAttN [LLH<sup>∗</sup>21] and AdaIN [HB17] show competitive and more stable performance, their overall stylisation effectiveness still falls short. Methods such as USO [WHC<sup>∗</sup>25] and OmniStyle [WLL<sup>∗</sup>25] demonstrate high consistency (narrow error bars) but fail to achieve competitive scores in our evaluation framework. To further illustrate the qualitative variability of each method, we provide the top-3 and bottom-3 overall preference examples for every method in the Appendix Fig. 9.

Importantly, we observe that overall preference aligns more closely with style similarity than with content preservation. For instance, USO [WHC<sup>∗</sup>25] achieves the highest ranking in content, yet ranks among the lowest in both overall and style evaluations, revealing a clear discrepancy between content fidelity and success at style transfer.

To further understand this phenomenon, we analyse cases where content and style preferences disagree. Across a total of 18,747 such conflict cases, participants follow the style preference in 80.3% of comparisons, compared to 19.7% for content. This trend aligns with the observations from Fig. 4, where the overall preference ranking closely aligns with style preference across methods. The evidence supports the hypothesis that for the style transfer task, humans are more sensitive to discrepancies in the style than in the content.

Fig. 3 shows the top-5 and bottom-5 results ranked by overall preference scores. Top-ranked results often correspond to styles with strong and distinctive visual cues (e.g., Chinese landscape, Carr), suggesting that clear colour and texture patterns are more readily perceived and evaluated. In contrast, bottom-ranked results exhibit two common failure modes: severe content degradation, where structure becomes unrecognisable (e.g., Rembrandt), and weak stylisation, where content is preserved but stylistic cues are minimal (e.g., Australian Aboriginal, Hokusai). These observations indicate that both recognisable content and perceptually salient style cues are important, with the latter playing a particularly prominent role.

## 5. ASTRA-Score and Experiments

Next, we introduce ASTRA-Score, an evaluation model trained on ASTRA-Data, and compare it with existing metrics on the benchmark image set.

![](images/085ff31eb7dbae81bbe5e8c127cf5a833da91d1d872e6ce702ca01baa81d7b6a.jpg)  
(a) Content preference

![](images/6361f2c6c9032e0377304f96e52c801e09b2f58cd47593e8920314cf1eb4dd1f.jpg)  
(b) Style preference

![](images/0bfa8ecae15fd079b14b5912bef085bb428f73347afe3ac76a4d43ab0683bdbe.jpg)  
(c) Overall preference  
Figure 7: Human preference rankings derived from Rank Centrality. Each point shows the mean score and the error bar denotes bootstrap uncertainty. “Overall preference” refers to the subjects’ holistic assessment of the effectiveness of the style transfer.

## 5.1. Evaluation Model

To enable automatic assessment of stylised images, we learn a regression-based evaluation model that predicts human-aligned scores from a content-style-stylisation image triplet, as illustrated in Fig. 8. Given a content reference image $I _ { c } ,$ a style reference image $I _ { s } ,$ and a generated stylised image $I _ { g } ,$ the goal is to estimate how well the generated result preserves the content structure and matches the target style, and forms an ideal combination of the two. We train three separate predictors corresponding to three evaluation criteria: overall preference, content preservation, and style similarity. Using the fully trained model, we can evaluate new stylisation methods on the benchmark image set. We employ a frozen pretrained backbone to extract hierarchical visual representations and train only lightweight task-specific regression heads. We empirically compare several backbone families, including MobileNetV3, VGG, ResNet, ViT, and DINOv2. Among the evaluated architectures, VGG11 provides the strongest overall alignment with human judgements across the three evaluation criteria, and is therefore adopted as the default backbone for ASTRA-Score. The backbone remains frozen throughout training, limiting the number of trainable parameters to the regression heads. Full results of the backbone comparison are provided in Appendix F.

Feature Representation. Given an input image $I ,$ we extract intermediate feature maps from four progressively deeper stages of VGG11, denoted as $\{ L _ { i } \} _ { i = 1 } ^ { 4 }$ , with channel dimensions of 64, 128, 256, and 512, respectively.

Content and Style Features. The deepest feature map $L _ { 4 }$ is used to represent semantic content. We apply Global Average Pooling (GAP) to obtain a content feature vector $\mathbf { f } _ { c } \in \mathbb { R } ^ { 5 1 2 }$ . For style representation, we compute the channel-wise mean $\mu ( \cdot )$ and standard deviation $\sigma ( \cdot )$ for each selected feature map L<sub>i</sub> [LLH 21]. The style descriptor at each level is defined as $\nu _ { i } = \hat { [ \mu ( L _ { i } ) } ^ { \top } , \sigma ( L _ { i } ) ^ { \top } ] ^ { \top }$ . The final multi-level style representation is obtained by concatenating the statistics from all four stages,

$$
\mathbf { f } _ { s } = [ \nu _ { 1 } ^ { \top } , \nu _ { 2 } ^ { \top } , \nu _ { 3 } ^ { \top } , \nu _ { 4 } ^ { \top } ] ^ { \top } \in \mathbb { R } ^ { 1 9 2 0 } .\tag{5}
$$

Relation features. We model the relationship between the gener-

![](images/925edaef90ff93ce1d52703826a6fda32e361dd7113f3ead7852c7b47d82bbd2.jpg)  
Figure 8: Overview of the proposed evaluation model. A frozen VGG11 extracts content features (f<sup>g</sup>, f ) and style features (f<sup>g</sup>, f ) from the generated image $I _ { g } ,$ content reference $I _ { c } ,$ and style reference $I _ { s } .$ . Three independent heads are employed: the Content Head uses $\boldsymbol { \Phi } ( \mathbf { f } _ { c } ^ { g } , \mathbf { f } _ { c } )$ , the Style Head uses ψ $\mathbf { \delta } _ { \mathbf { \delta } ^ { \prime } \mathbf { \delta } ^ { \prime } \mathbf { \delta } ^ { \prime } } ( \mathbf { f } _ { s } ^ { g } , \mathbf { f } _ { s } )$ , and the Overall Head combines both relation features to predict an overall quality score.

ated image and the references in feature space. Let $\mathbf { f } _ { c } ^ { g } , \mathbf { f } _ { c }$ denote content features, and $\mathbf { f } _ { s } ^ { g } , \mathbf { f } _ { s }$ denote style features. We define two relation operators:

$$
\phi ( \mathbf { a } , \mathbf { b } ) = \left[ | \mathbf { a } - \mathbf { b } | , \mathbf { a } \odot \mathbf { b } , \cos ( \mathbf { a } , \mathbf { b } ) \right] ,\tag{6}
$$

$$
\Psi ( \mathbf { a } , \mathbf { b } ) = \left[ | \mathbf { a - b } | , \mathbf { a } \odot \mathbf { b } , \mu ( | \mathbf { a - b } | ) , \mu ( ( \mathbf { a - b } ) ^ { 2 } ) \right] ,\tag{7}
$$

where $\mu ( \cdot )$ denotes the mean over feature dimensions. The three predictors use task-specific relation features:

$$
\mathbf { x } _ { \mathrm { c o n t e n t } } = \boldsymbol { \Phi } ( \mathbf { f } _ { c } ^ { g } , \mathbf { f } _ { c } ) , \quad \mathbf { x } _ { \mathrm { s t y l e } } = \boldsymbol { \Psi } ( \mathbf { f } _ { s } ^ { g } , \mathbf { f } _ { s } ) ,\tag{8}
$$

$$
\begin{array} { r } { \mathbf { x } _ { \mathrm { o v e r a l l } } = \left[ \boldsymbol { \Phi } ( \mathbf { f } _ { c } ^ { g } , \mathbf { f } _ { c } ) , \boldsymbol { \Psi } ( \mathbf { f } _ { s } ^ { g } , \mathbf { f } _ { s } ) \right] . } \end{array}\tag{9}
$$

Regression heads. Each predictor is implemented as a lightweight two-layer MLP with dropout, producing a scalar score. The models are trained independently using 3 preference scores for each evaluation dimension, with full training details provided in Appendix E.

Table 2: Correlation between automatic metrics and study-derived rankings across the evaluated methods. Baseline metrics are evaluated using leave-one-method-out (LOMO) evaluation and reported as mean ± standard deviation.
<table><tr><td>Metric</td><td>Spearman</td><td>Kendall</td></tr><tr><td colspan="3">Content</td></tr><tr><td>CFSD [CHH24]</td><td> $0 . 2 8 1 \pm 0 . 2 1 2$ </td><td> $0 . 1 9 6 \pm 0 . 1 5 1$ </td></tr><tr><td>NMI [SHH98]</td><td> $0 . 4 1 5 \pm 0 . 1 4 1$ </td><td> $0 . 2 8 8 \pm 0 . 1 0 0$ </td></tr><tr><td>SIFID [SDM19]</td><td> $0 . 4 2 9 \pm 0 . 1 7 0$ </td><td> $0 . 3 0 0 \pm 0 . 1 2 9$ </td></tr><tr><td>GMSD [XZMB13]</td><td> $0 . 4 5 6 \pm 0 . 1 7 6$ </td><td> $0 . 3 2 3 \pm 0 . 1 3 3$ </td></tr><tr><td>FSIM [ZZMZ11]</td><td> $0 . 5 1 5 \pm 0 . 1 6 0$ </td><td> $0 . 3 6 8 \pm 0 . 1 1 8$ </td></tr><tr><td>Qwen [BBC*23]</td><td> $0 . 5 2 6 \pm 0 . 2 1 2$ </td><td> $0 . 4 1 4 \pm 0 . 1 6 8$ </td></tr><tr><td>VIF [SB06]</td><td> $0 . 5 5 1 \pm 0 . 1 7 8$ </td><td>0.396 ± 0.136</td></tr><tr><td>CLIP [RKH*21]</td><td> $0 . 5 5 9 \pm 0 . 1 6 5$ </td><td> $0 . 4 0 1 \pm 0 . 1 3 8$ </td></tr><tr><td>MS-SSIM [WSB03]</td><td> $0 . 5 8 2 \pm 0 . 1 3 7$ </td><td> $0 . 4 1 6 \pm 0 . 1 0 4$ </td></tr><tr><td>LPIPS [ZIE*18]</td><td> $0 . 5 8 6 \pm 0 . 1 2 9$ </td><td>0.422 ± 0.108</td></tr><tr><td>DINO [CTM*21]</td><td> $0 . 6 1 0 \pm 0 . 1 6 0$ </td><td> $0 . 4 4 2 \pm 0 . 1 3 2$ </td></tr><tr><td>ViT [DBK*20]</td><td> $0 . 6 1 4 \pm 0 . 1 5 5$ </td><td> $0 . 4 4 4 \pm 0 . 1 3 0$ </td></tr><tr><td>SRQE [CSC*22]</td><td> $0 . 6 3 4 \pm 0 . 1 5 0$ </td><td> $0 . 4 6 1 \pm 0 . 1 1 9$ </td></tr><tr><td>ASTRA-Score Content (ours)</td><td> ${ \bf 0 . 7 1 8 \pm 0 . 1 4 4 }$ </td><td> ${ \bf 0 . 5 4 0 \pm 0 . 1 2 8 }$ </td></tr><tr><td colspan="3">Style</td></tr><tr><td>Gram Loss [GEB16]</td><td> $0 . 0 3 9 \pm 0 . 2 4 0$ </td><td> $0 . 0 3 0 \pm 0 . 1 6 6$ </td></tr><tr><td> $\mathrm { C S D } \ [ \mathrm { S G G } ^ { * } 2 4 ]$ </td><td> $0 . 0 6 6 \pm 0 . 2 6 8$ </td><td> $0 . 0 4 7 \pm 0 . 1 8 5$ </td></tr><tr><td>SIFID [SDM19]</td><td> $0 . 1 0 3 \pm 0 . 1 2 8$ </td><td> $0 . 0 7 3 \pm 0 . 0 9 0$ </td></tr><tr><td>DINO [CTM*21]</td><td> $0 . 1 5 2 \pm 0 . 2 8 1$ </td><td> $0 . 1 1 1 \pm 0 . 1 9 0$ </td></tr><tr><td>CLIP [RKH*21]</td><td> $0 . 1 9 9 \pm 0 . 1 6 2$ </td><td> $0 . 1 3 7 \pm 0 . 1 0 9$ </td></tr><tr><td>Qwen [BBC*23]</td><td> $0 . 2 5 0 \pm 0 . 2 0 4$ </td><td> $0 . 1 9 0 \pm 0 . 1 5 3$ </td></tr><tr><td>ViT [DBK*20]</td><td> $0 . 2 9 0 \pm 0 . 1 3 7$ </td><td> $0 . 1 9 8 \pm 0 . 0 9 8$ </td></tr><tr><td>SRQE [CSC*22]</td><td> $0 . 3 9 0 \pm 0 . 2 0 7$ </td><td> $0 . 2 7 4 \pm 0 . 1 4 9$ </td></tr><tr><td>ASTRA-Score Style (ours)</td><td> ${ \bf 0 . 7 1 4 \pm 0 . 0 9 8 }$ </td><td> $\mathbf { 0 . 5 3 1 \pm 0 . 0 8 4 }$ </td></tr><tr><td colspan="3">Overall</td></tr><tr><td>ArtFID [WO22]</td><td> $0 . 1 0 9 \pm 0 . 1 7 3$ </td><td> $0 . 0 7 6 \pm 0 . 1 1 7$ </td></tr><tr><td>Qwen [BBC*23]</td><td> $0 . 2 9 5 \pm 0 . 1 7 2$ </td><td> $0 . 2 3 3 \pm 0 . 1 3 6$ </td></tr><tr><td>SRQE [CSC*22]</td><td> $0 . 3 5 6 \pm 0 . 2 2 3$ </td><td> $0 . 2 4 7 \pm 0 . 1 5 4$ </td></tr><tr><td>ASTRA-Score Overall (ours)</td><td> $\mathbf { 0 . 6 6 7 \pm 0 . 0 9 9 }$ </td><td> ${ \bf 0 . 4 9 2 \pm 0 . 0 8 1 }$ </td></tr></table>

Table 3: Correlation between automatic metrics and study-derived rankings across the evaluated methods. Baseline metrics are evaluated using leave-one-content-out (LOCO) evaluation and reported as mean ± standard deviation.
<table><tr><td>Metric</td><td>Spearman</td><td>Kendall</td></tr><tr><td colspan="3">Content</td></tr><tr><td>CFSD [CHH24]</td><td> $0 . 4 4 0 \pm 0 . 0 5 5$ </td><td> $0 . 3 1 0 \pm 0 . 0 4 4$ </td></tr><tr><td>Qwen [BBC*23]</td><td> $0 . 4 5 5 \pm 0 . 1 8 4$ </td><td> $0 . 3 5 7 \pm 0 . 1 4 5$ </td></tr><tr><td>NMI [SHH98]</td><td> $0 . 4 9 2 \pm 0 . 0 4 3$ </td><td> $0 . 3 4 5 \pm 0 . 0 3 3$ </td></tr><tr><td>GMSD [XZMB13]</td><td> $0 . 4 9 8 \pm 0 . 1 0 3$ </td><td> $0 . 3 6 1 \pm 0 . 0 8 1$ </td></tr><tr><td>FSIM [ZZMZ11]</td><td>0.544 ± 0.086</td><td> $0 . 3 9 5 \pm 0 . 0 7 4$ </td></tr><tr><td>SIFID [SDM19]</td><td> $0 . 5 5 7 \pm 0 . 1 3 7$ </td><td> $0 . 4 0 0 \pm 0 . 1 0 2$ </td></tr><tr><td>VIF [SB06]</td><td> $0 . 5 5 8 \pm 0 . 1 1 3$ </td><td> $0 . 4 1 7 \pm 0 . 0 7 7$ </td></tr><tr><td>MS-SSIM [WSB03]</td><td> $0 . 5 7 7 \pm 0 . 0 4 3$ </td><td> $0 . 4 1 9 \pm 0 . 0 2 4$ </td></tr><tr><td> $\mathrm { S R Q E } \left[ \mathrm { C S C } ^ { * } 2 2 \right]$ </td><td> $0 . 5 8 5 \pm 0 . 0 7 5$ </td><td> $0 . 4 2 8 \pm 0 . 0 4 9$ </td></tr><tr><td> $\mathrm { L P I P S } [ \mathrm { Z I E } ^ { \ast } 1 8 ]$ </td><td> $0 . 7 0 5 \pm 0 . 0 7 9$ </td><td> $0 . 5 2 5 \pm 0 . 0 7 0$ </td></tr><tr><td> $\mathrm { V i T } \left[ \mathrm { D B K } ^ { * } 2 0 \right]$ </td><td> $0 . 7 1 8 \pm 0 . 0 8 7$ </td><td> $0 . 5 3 7 \pm 0 . 0 8 0$ </td></tr><tr><td> $\mathrm { C L I P } \left[ \mathrm { R K H } ^ { * } 2 1 \right]$ </td><td> $0 . 7 2 1 \pm 0 . 0 3 3$ </td><td> $0 . 5 2 8 \pm 0 . 0 3 3$ </td></tr><tr><td> $\mathbf { D I N O } \left[ \mathbf { C T M } ^ { * } 2 \mathbf { 1 } \right]$ </td><td> ${ \bf 0 . 7 5 9 \pm 0 . 0 8 2 }$ </td><td> $\mathbf { 0 . 5 7 3 \pm 0 . 0 7 1 }$ </td></tr><tr><td>ASTRA-Score Content (ours)</td><td> $0 . 7 4 5 \pm 0 . 0 9 4$ </td><td> $0 . 5 6 2 \pm 0 . 0 9 0$ </td></tr><tr><td>Style</td><td></td><td></td></tr><tr><td>CSD [SGG*24]</td><td> $0 . 1 3 0 \pm 0 . 0 9 4$ </td><td> $0 . 0 8 8 \pm 0 . 0 6 3$ </td></tr><tr><td>CLIP [RKH*21]</td><td> $0 . 1 3 9 \pm 0 . 1 0 2$ </td><td> $0 . 0 9 2 \pm 0 . 0 7 2$ </td></tr><tr><td>SIFID [SDM19]</td><td> $0 . 1 9 0 \pm 0 . 0 9 8$ </td><td> $0 . 1 2 6 \pm 0 . 0 6 5$ </td></tr><tr><td>Gram Loss [GEB16]</td><td> $0 . 2 2 7 \pm 0 . 0 8 4$ </td><td> $0 . 1 6 2 \pm 0 . 0 5 6$ </td></tr><tr><td> $\mathrm { Q w e n } \ [ \mathrm { B B C } ^ { * } 2 3 ]$ </td><td> $0 . 2 6 2 \pm 0 . 0 5 5$ </td><td> $0 . 1 9 8 \pm 0 . 0 4 7$ </td></tr><tr><td>DINO [CTM*21]</td><td> $0 . 3 7 8 \pm 0 . 1 0 6$ </td><td> $0 . 2 6 1 \pm 0 . 0 7 3$ </td></tr><tr><td> $\mathrm { V i T } \left[ \mathrm { D B K } ^ { * } 2 0 \right]$ </td><td> $0 . 4 7 2 \pm 0 . 0 3 8$ </td><td> $0 . 3 2 8 \pm 0 . 0 2 8$ </td></tr><tr><td> $\mathrm { S R Q E } \left[ \mathrm { C S C } ^ { * } 2 2 \right]$ </td><td> $0 . 5 2 9 \pm 0 . 0 2 3$ </td><td> $0 . 3 8 0 \pm 0 . 0 1 7$ </td></tr><tr><td>ASTRA-Score Style (ours)</td><td> ${ \bf 0 . 8 1 4 \pm 0 . 0 3 7 }$ </td><td> ${ \bf 0 . 6 2 6 \pm 0 . 0 3 6 }$ </td></tr><tr><td colspan="3">Overall</td></tr><tr><td>ArtFID [WO22]</td><td> $0 . 1 1 5 \pm 0 . 1 1 5$ </td><td> $0 . 0 7 5 \pm 0 . 0 7 2$ </td></tr><tr><td>Qwen [BBC*23]</td><td> $0 . 1 7 9 \pm 0 . 1 0 5$ </td><td> $0 . 1 3 8 \pm 0 . 0 8 1$ </td></tr><tr><td>SRQE [CSC*22]</td><td> $0 . 5 4 5 \pm 0 . 0 3 6$ </td><td> $0 . 3 9 0 \pm 0 . 0 2 3$ </td></tr><tr><td>ASTRA-Score Overall (ours)</td><td> $\mathbf { 0 . 7 9 3 \pm 0 . 0 7 6 }$ </td><td> $\mathbf { 0 . 6 0 6 \pm 0 . 0 8 0 }$ </td></tr></table>

## 5.2. Evaluation Methods Comparison

We evaluate whether objective metrics can predict perceptual rankings on ASTRA-Data, which contains 12 styles, 6 contents, and 12 stylisation methods. To assess generalisation across methods, we adopt a leave-one-method-out protocol. In each run, one method is held out while the remaining 11 methods are used to train the prediction model, which then predicts the ranking of the held-out method over its 72 generated images. The predicted ranking is compared with the global rankings using Spearman’s ρ [Spe61] and Kendall’s τ [Ken38]. This process is repeated for all 12 methods, and the final performance is reported as the average correlation across runs.

gap between existing automatic metrics and perceptual preference. SRQE [CSC<sup>∗</sup>22] achieves relatively strong performance, demonstrating the value of human-annotated supervision among prior approaches. However, its reliance on fixed content–style pairs entangles scores with pair-specific difficulty, and its evaluator is still based on handcrafted features rather than directly learned from user evaluations.

We further evaluate a range of commonly used style transfer metrics, including style similarity, content preservation, and overall quality measures, under the same protocol. For each metric, rankings are computed for the held-out method and compared with the ground truth in the same manner. Tab. 2 reveals a clear

Even the best prior automatic metrics achieve only moderate correlation with human rankings, while several widely used measures (e.g., CFSD [CHH24] and SIFID (Single Image FID) [SDM19]) exhibit weak correlation in the content evaluation, indicating that minimising perceptual distance does not necessarily lead to more preferred stylisations.

The discrepancy is even more pronounced for style and overall evaluations: classical statistics such as Gram Loss [GEB16] and recent pretrained vision-language model based metrics, including embedding-based similarity (e.g., CLIP [RKH 21]) and instruction following models such as Qwen [BBC 23], show low and inconsistent correlation with human judgements, suggesting that stylistic quality perceived by humans is not captured by simple feature statistics. Across all evaluation heads, our metric consistently achieves substantially higher correlation with study-derived rankings, demonstrating a stronger alignment with the perceptual criteria governing neural style transfer quality.

Generalisation to unseen content and styles. The leave-onemethod-out (LOMO) protocol above directly reflects the intended use of ASTRA-Score: evaluating a new style transfer method applied to the fixed content and style references in ASTRA-Data. In each split, the evaluated stylisation method is unseen during training, while the benchmark references remain available. We additionally conduct leave-one-content-out (LOCO) and leave-onestyle-out (LOSO) experiments as robustness analyses rather than as tests of the intended deployment setting. These experiments examine whether the strong performance of ASTRA-Score depends on having observed every benchmark reference during training. By withholding an entire content or style reference together with all associated stylised results, LOCO and LOSO provide a stricter test of reference dependence and indicate whether the learnt evaluator captures transferable relationships between reference and stylised images rather than simply memorising the specific content and style images in the benchmark.

For LOCO, all stylised images associated with one content reference are held out for testing, while ASTRA-Score is trained on the remaining five contents; this is repeated for all six contents. Table 3 reports the results. ASTRA-Score achieves Spearman correlations of 0.745, 0.814, and 0.793 for content, style, and overall preference, respectively, outperforming the strongest baselines by 53.9% for style and 45.5% for overall preference. DINO achieves a slightly higher correlation (0.759 vs. 0.745), consistent with the strong semantic and structural representations learnt through largescale self-supervised pretraining. Overall, ASTRA-Score remains strongly aligned with human judgements despite excluding an entire content reference and its associated samples from training.

For LOSO, all stylised images associated with one style reference are similarly held out, with results averaged over the twelve styles (Table 4). ASTRA-Score achieves Spearman correlations of 0.711, 0.700, and 0.658 for content, style, and overall preference, respectively, outperforming the strongest baselines of 0.673, 0.522, and 0.525. Together, the LOCO and LOSO results indicate that ASTRA-Score learns transferable evaluation cues rather than simply memorising the content and style references observed during training.

Robustness across evaluation protocols. Taken together, the three protocols examine complementary aspects of ASTRA-Score. LOMO constitutes the primary evaluation and directly reflects its intended use: assessing unseen stylisation methods on the ASTRA benchmark. LOCO and LOSO provide stricter tests by additionally withholding entire content or style references from training. The strong correlations under these settings indicate that ASTRA-Score does not simply memorise reference-specific patterns, but learns relationships between content, style, and stylised outputs that generalise to unseen references. Although such reference-level generalisation extends beyond the primary intended use of ASTRA-Score, the results demonstrate its robustness when applied to content or style references not observed during training.

Table 4: Correlation between automatic metrics and study-derived rankings across the evaluated methods. Baseline metrics are evaluated using leave-one-style-out (LOSO) evaluation and reported as mean ± standard deviation.
<table><tr><td>Metric</td><td>Spearman</td><td>Kendall</td></tr><tr><td>Content</td></tr><tr><td>CFSD [CHH24]</td><td> $0 . 2 5 0 \pm 0 . 1 3 1$   $0 . 1 7 8 \pm 0 . 0 9 1$   $0 . 2 8 0 \pm 0 . 1 3 3$ </td></tr><tr><td>GMSD [XZMB13]</td><td> $0 . 3 9 1 \pm 0 . 1 7 9$ </td></tr><tr><td>Qwen [BBC*23]</td><td> $0 . 4 1 7 \pm 0 . 1 9 1$   $0 . 3 2 5 \pm 0 . 1 4 3$ </td></tr><tr><td>FSIM [ZZMZ11]</td><td> $0 . 4 3 5 \pm 0 . 1 6 8$   $0 . 3 1 2 \pm 0 . 1 2 5$ </td></tr><tr><td>SIFID [SDM19]</td><td> $0 . 4 6 7 \pm 0 . 1 5 2$   $0 . 3 3 4 \pm 0 . 1 1 4$ </td></tr><tr><td>VIF [SB06]</td><td> $0 . 4 8 3 \pm 0 . 1 0 9$   $0 . 3 5 7 \pm 0 . 0 8 4$ </td></tr><tr><td>MS-SSIM [WSB03]</td><td> $0 . 5 1 0 \pm 0 . 1 1 0$   $0 . 3 6 9 \pm 0 . 0 8 8$ </td></tr><tr><td>NMI [SHH98]</td><td> $0 . 5 2 2 \pm 0 . 1 3 5$   $0 . 3 7 1 \pm 0 . 0 9 8$ </td></tr><tr><td>SRQE [CSC*22]</td><td> $0 . 5 2 4 \pm 0 . 0 9 9$   $0 . 3 8 5 \pm 0 . 0 7 9$ </td></tr><tr><td>CLIP [RKH*21]</td><td> $0 . 6 2 7 \pm 0 . 0 8 5$   $0 . 4 5 2 \pm 0 . 0 6 6$ </td></tr><tr><td>ViT [DBK*20]</td><td> $0 . 6 3 1 \pm 0 . 1 0 2$   $0 . 4 5 8 \pm 0 . 0 8 6$ </td></tr><tr><td>LPIPS [ZIE*18]</td><td> $0 . 6 4 2 \pm 0 . 1 2 8$   $0 . 4 7 5 \pm 0 . 1 0 9$ </td></tr><tr><td>DINO [CTM*21] ASTRA-Score Content (ours)</td><td> $0 . 6 7 3 \pm 0 . 0 7 4$   $0 . 4 9 3 \pm 0 . 0 6 4$   ${ \bf 0 . 7 1 1 \pm 0 . 1 1 8 }$ </td></tr><tr><td> ${ \bf 0 . 5 3 4 \pm 0 . 1 0 3 }$ </td></tr><tr><td>Style</td></tr><tr><td>CSD [SGG*24]  $0 . 1 6 8 \pm 0 . 2 8 7$  Qwen [BBC*23]</td></tr><tr><td> $0 . 2 2 2 \pm 0 . 1 7 4$  CLIP [RKH*21]  $0 . 2 3 0 \pm 0 . 1 5 2$ </td></tr><tr><td>SIFID [SDM19]  $0 . 3 5 1 \pm 0 . 1 4 8$ </td></tr><tr><td> $0 . 2 4 1 \pm 0 . 1 0 4$  DINO [CTM*21]  $0 . 4 1 3 \pm 0 . 2 1 3$ </td></tr><tr><td> $0 . 2 9 5 \pm 0 . 1 5 9$  Gram Loss [GEB16]  $0 . 4 6 7 \pm 0 . 3 0 4$   $0 . 3 5 0 \pm 0 . 2 1 5$ </td></tr><tr><td>ViT [DBK*20]  $0 . 4 8 8 \pm 0 . 1 9 8$   $0 . 3 5 1 \pm 0 . 1 5 7$ </td></tr><tr><td>SRQE [CSC*22]  $0 . 5 2 2 \pm 0 . 1 9 8$   $0 . 3 8 4 \pm 0 . 1 4 6$ </td></tr><tr><td>ASTRA-Score Style (ours)  ${ \bf 0 . 7 0 0 \pm 0 . 1 4 3 }$   ${ \bf 0 . 5 2 8 \pm 0 . 1 3 0 }$ </td></tr><tr><td>Overall</td></tr><tr><td>Qwen [BBC*23]  $0 . 0 8 2 \pm 0 . 1 7 7$   $0 . 0 6 4 \pm 0 . 1 3 4$ </td></tr><tr><td>ArtFID [WO22]  $0 . 1 9 2 \pm 0 . 2 2 7$   $0 . 1 3 5 \pm 0 . 1 6 0$ </td></tr><tr><td></td></tr><tr><td>SRQE [CSC*22]  $0 . 5 2 5 \pm 0 . 1 9 3$   $0 . 3 7 7 \pm 0 . 1 4 2$  ASTRA-Score Overall (ours)  $\mathbf { 0 . 6 5 8 \pm 0 . 2 3 1 }$   ${ \bf 0 . 4 9 9 \pm 0 . 1 9 8 }$ </td></tr></table>

## 6. Conclusion

In this paper, we systematically investigate the reliability of automatic evaluation metrics for neural style transfer using a human preference benchmark. Our results reveal clear limitations of existing metrics, which often fail to align with perceptual judgement, while conventional user studies remain time-consuming, difficult to reproduce, and lack comparability across works. In this paper, we propose ASTRA, an evaluation mechanism for style transfer. We construct ASTRA-Data, a representative yet tractable benchmark dataset of content–style pairs with pairwise preference annotations, from which we derive ranking-based ground truth for content preservation, style similarity, and overall preference. We used the user study data to train ASTRA-Score, an evaluation model that predicts preference-aligned scores from content-stylestylisation image triplet, enabling scalable and consistent evaluation across methods: ASTRA-Score provides reliable estimates of stylisation effectiveness on our benchmark image set. Together, ASTRA-Data and ASTRA-Score provide a standardised testbed for benchmarking style transfer methods via their alignment with human rankings.

## References

[AAA<sup>∗</sup>23] ACHIAM J., ADLER S., AGARWAL S., AHMAD L., AKKAYA I., ALEMAN F. L., ALMEIDA D., ALTENSCHMIDT J., ALTMAN S., ANADKAT S., ET AL.: GPT-4 technical report, 2023. 4

[AHS<sup>∗</sup>21] AN J., HUANG S., SONG Y., DOU D., LIU W., LUO J.: Art-Flow: Unbiased image style transfer via reversible neural flows. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (2021), IEEE, pp. 862–871. 4

[BBC<sup>∗</sup>23] BAI J., BAI S., CHU Y., CUI Z., DANG K., DENG X., FAN Y., GE W., HAN Y., HUANG F., ET AL.: Qwen technical report, 2023. 1, 3, 9, 10

[BHZ<sup>∗</sup>23] BYLINSKII Z., HERTZMANN A., ZHANG Y., HERMAN L., HUTKA S.: Towards Better User Studies in Computer Graphics and Vision. Foundations and Trends in Computer Graphics and Vision 15, 3 (2023), 201–252. 1

[BK69] BERLIN B., KAY P.: Basic Color Terms: Their Universality and Evolution. University of California Press, 1969. 4, 15

[BT52] BRADLEY R. A., TERRY M. E.: Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika 39, 3/4 (1952), 324–345. 5, 18

[BVFAB22] BLUSSEAU S., VELASCO-FORERO S., ANGULO J., BLOCH I.: Adaptive anisotropic morphological filtering based on cocircularity of local orientations. Image Processing On Line 12 (2022), 111–141. doi:10.5201/ipol.2022.397. 4, 15

[CHH24] CHUNG J., HYUN S., HEO J.-P.: Style injection in diffusion: A training-free approach for adapting large-scale diffusion models for style transfer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (2024), pp. 8795–8805. 3, 4, 9, 10

[CS15] CHEN Y., SUH C.: Spectral MLE: Top-k rank aggregation from pairwise comparisons. In International conference on machine learning (2015), PMLR, pp. 371–380. 5, 18

[CSC<sup>∗</sup>22] CHEN H., SHAO F., CHAI X., GU Y., JIANG Q., MENG X., HO Y.-S.: Quality Evaluation of Arbitrary Style Transfer: Subjective Study and Objective Metric. IEEE Transactions on Circuits and Systems for Video Technology 33, 7 (2022), 3055–3070. 1, 2, 3, 9, 10

[CSMJ24] CHEN H., SHAO F., MU B., JIANG Q.: Assessing arbitrary style transfer like an artist. Displays 85 (2024), 102859. 3

[CTM<sup>∗</sup>21] CARON M., TOUVRON H., MISRA I., JÉGOU H., MAIRAL J., BOJANOWSKI P., JOULIN A.: Emerging properties in self-supervised vision transformers. In Proceedings ofthe IEEE/CVF international conference on computer vision (2021), pp. 9650–9660. 1, 3, 9, 10

[CWZ<sup>∗</sup>21] CHEN H., WANG Z., ZHANG H., ZUO Z., LI A., XING W., LU D., ET AL.: Artistic style transfer with internal-external learning and contrastive learning. Advances in Neural Information Processing Systems 34 (2021), 26561–26573. 4

[DBK<sup>∗</sup>20] DOSOVITSKIY A., BEYER L., KOLESNIKOV A., WEIS-SENBORN D., ZHAI X., UNTERTHINER T., DEHGHANI M., MIN-DERER M., HEIGOLD G., GELLY S., ET AL.: An image is worth 16x16 words: Transformers for image recognition at scale, 2020. 1, 3, 9, 10

[DTD<sup>∗</sup>22] DENG Y., TANG F., DONG W., MA C., PAN X., WANG L., XU C.: StyTr<sup>2</sup>: Image style transfer with transformers. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition (2022), pp. 11326–11336. 1, 4, 7

[FXDG17] FARAJ N., XIA G.-S., DELON J., GOUSSEAU Y.: A generic framework for the structured abstraction of images. In Proceedings of the Symposium on Non-Photorealistic Animation and Rendering (2017). 4, 15

[GEB16] GATYS L. A., ECKER A. S., BETHGE M.: Image style transfer using convolutional neural networks. In Proceedings ofthe IEEE conference on computer vision andpattern recognition (2016), pp. 2414–2423. 1, 2, 3, 4, 7, 9, 10

[GLK<sup>∗</sup>17] GHIASI G., LEE H., KUDLUR M., DUMOULIN V., SHLENS J.: Exploring the structure of a real-time, arbitrary neural artistic stylization network, 2017. 1, 4

[GSL<sup>∗</sup>25] GAO J., SUN Y., LIU Y., TANG Y., ZENG Y., QI D., CHEN K., ZHAO C.: StyleShot: A snapshot on any style. IEEE Transactions on Pattern Analysis and Machine Intelligence (2025). 1

[HB17] HUANG X., BELONGIE S.: Arbitrary style transfer in real-time with adaptive instance normalization. In Proceedings of the IEEE international conference on computer vision (2017), pp. 1501–1510. 1, 4, 7

[HJL<sup>∗</sup>20] HU Z., JIA J., LIU B., BU Y., FU J.: Aesthetic-Aware Image Style Transfer. In Proceedings of the 28th ACM International Conference on Multimedia (2020), pp. 3320–3329. 4

[HRU<sup>∗</sup>17] HEUSEL M., RAMSAUER H., UNTERTHINER T., NESSLER B., HOCHREITER S.: GANs trained by a two time-scale update rule converge to a local Nash equilibrium. Advances in neural information processing systems 30 (2017). 1, 3

[HZG24] HU Y., ZHUANG C., GAO P.: DiffuseST: Unleashing the capability of the diffusion model for style transfer. In Proceedings of the 6th ACM International Conference on Multimedia in Asia (2024), pp. 1–1. 4

[IM24] IOANNOU E., MADDOCK S.: Evaluation in neural style transfer: a review. In Computer Graphics Forum (2024), vol. 43, Wiley Online Library, p. e15165. 2

[JAB<sup>∗</sup>19] JACKSON P. T., ABARGHOUEI A. A., BONNER S., BRECKON T. P., OBARA B.: Data Augmentation Via Style Randomization. In CVPR workshops (2019), vol. 6, pp. 10–11. 1

[JSO<sup>∗</sup>01] JACOBS C., SALESIN D., OLIVER N., HERTZMANN A., CURLESS A.: Image analogies. In Proceedings of SIGGRAPH (2001), pp. 327–340. 1, 4, 15

[Ken38] KENDALL M. G.: A new measure of rank correlation. Biometrika 30, 1-2 (1938), 81–93. 9

[KLC07] KANG H., LEE S., CHUI C. K.: Coherent Line Drawing. In Proceedings of the 5th international symposium on Non-photorealistic animation and rendering (2007), pp. 43–50. 4, 15

[KSM<sup>∗</sup>19] KOTOVENKO D., SANAKOYEU A., MA P., LANG S., OM-MER B.: A content Transformation Block for Image Style Transfer. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (2019), pp. 10032–10041. 1

[KTH<sup>∗</sup>13] KARAYEV S., TRENTACOSTE M., HAN H., AGARWALA A., DARRELL T., HERTZMANN A., WINNEMOELLER H.: Recognizing image style, 2013. 2

[L<sup>∗</sup>59] LUCE R. D., ET AL.: Individual Choice Behavior: A Theoretical Analysis, vol. 4. Wiley New York, 1959. 5, 18

[LLH<sup>∗</sup>21] LIU S., LIN T., HE D., LI F., WANG M., LI X., SUN Z., LI Q., DING E.: AdaAttN: Revisit attention mechanism in arbitrary neural style transfer. In Proceedings of the IEEE/CVF international conference on computer vision (2021), pp. 6649–6658. 1, 4, 7, 8

[LLKY19] LI X., LIU S., KAUTZ J., YANG M.-H.: Learning linear transformations for fast image and video style transfer. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition (2019), pp. 3809–3817. 4

[LMB<sup>∗</sup>14] LIN T.-Y., MAIRE M., BELONGIE S., HAYS J., PERONA P., RAMANAN D., DOLLÁR P., ZITNICK C. L.: Microsoft COCO: Common Objects in Context. In European conference on computer vision (2014), Springer, pp. 740–755. 2

[LR12] LAI Y.-K., ROSIN P. L.: Non-photorealistic Rendering with Reduced Colour Palettes. In Image and Video-based Artistic Stylisation. Springer, 2012, pp. 211–236. 1

[LSZ<sup>∗</sup>25] LEI M., SONG X., ZHU B., WANG H., ZHANG C.: StyleStudio: Text-driven style transfer with selective control of style elements. In Proceedings ofthe Computer Vision and Pattern Recognition Conference (2025), pp. 23443–23452. 1

[LXJ12] LU C., XU L., JIA J.: Combining sketch and tone for pencil drawing production. In Proceedings of the symposium on nonphotorealistic animation and rendering (2012), pp. 65–73. 4, 15

[MK18] MOHAMMAD S., KIRITCHENKO S.: Wikiart emotions: An Annotated Dataset of Emotions Evoked by Art. In Proceedings of the eleventh international conference on language resources and evaluation (LREC 2018) (2018). 2

[Mou14] MOULD D.: Authorial Subjective Evaluation of Non-Photorealistic Images. In Proceedings of the Workshop on Non-Photorealistic Animation and Rendering (2014), pp. 49–56. 3

[MR17] MOULD D., ROSIN P. L.: Developing and applying a benchmark for evaluating image stylization. Computers & Graphics 67 (2017), 58– 76. 1, 2, 3, 4

[MZLB23] MA Y., ZHAO C., LI X., BASU A.: RAST: Restorable arbitrary style transfer via multi-restoration. In Proceedings of the IEEE/CVF winter conference on applications of computer vision (2023), pp. 331–340. 4

[NOS17] NEGAHBAN S., OH S., SHAH D.: Rank centrality: Ranking from pairwise comparisons. Operations Research 65, 1 (2017), 266– 287. URL: https://doi.org/10.1287/opre.2016.1534, doi:10.1287/opre.2016.1534. 4, 5, 6, 18

[PKD07] PITIÉ F., KOKARAM A. C., DAHYOT R.: Automated colour grading using colour distribution transfer. Computer Vision and Image Understanding 107, 1-2 (2007), 123–137. 4, 15

[PL19] PARK D. Y., LEE K. H.: Arbitrary style transfer with styleattentional networks. In proceedings of the IEEE/CVF conference on computer vision and pattern recognition (2019), pp. 5880–5888. 1, 4, 7

[PZZ22] PARMAR G., ZHANG R., ZHU J.-Y.: On aliased resizing and surprising subtleties in gan evaluation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (2022), pp. 11410–11420. 4

[RGC<sup>∗</sup>24] RUTA D. S., GILBERT A., COLLOMOSSE J. P., SHECHT-MAN E., KOLKIN N.: NeAT: Neural artistic tracing for beautiful style transfer. In European Conference of Computer Vision 2024 Vision for Art (VISART VII) Workshop (2024). 1, 2

[RKH<sup>∗</sup>21] RADFORD A., KIM J. W., HALLACY C., RAMESH A., GOH G., AGARWAL S., SASTRY G., ASKELL A., MISHKIN P., CLARK J., ET AL.: Learning transferable visual models from natural language supervision. In International conference on machine learning (2021), PmLR, pp. 8748–8763. 3, 9, 10

[RL10] ROSIN P. L., LAI Y.-K.: Towards artistic minimal rendering. In Proceedings of the 8th International Symposium on Non-Photorealistic Animation and Rendering (2010), pp. 119–127. 1

[RL13] ROSIN P. L., LAI Y.-K.: Non-photorealistic rendering with spot colour. In Proceedings of the Symposium on Computational Aesthetics (2013), pp. 67–75. 1

[RL15] ROSIN P., LAI Y.: Non-photorealistic rendering of portraits. In Proceedings of the Workshop on Computational Aesthetics (2015), p. 159–170. 1

[RL17] ROSIN P. L., LAI Y.-K.: Watercolour rendering of portraits. In Pacific-Rim Symposium on Image and Video Technology (2017), Springer, pp. 268–282. 4, 15

[RLM<sup>∗</sup>22] ROSIN P. L., LAI Y.-K., MOULD D., YI R., BERGER I., DOYLE L., LEE S., LI C., LIU Y.-J., SEMMO A., ET AL.: NPRportrait 1.0: A three-level benchmark for non-photorealistic rendering of portraits. Computational Visual Media 8, 3 (2022), 445–465. 1

[RWW<sup>∗</sup>17] ROSIN P. L., WANG T., WINNEMÖLLER H., MOULD D., BERGER I., COLLOMOSSE J., LAI Y.-K., LI C., LI H., SHAMIR A., ET AL.: Benchmarking Non-Photorealistic Rendering of Portraits. 1

[SB06] SHEIKH H. R., BOVIK A. C.: Image Information and Visual Quality. IEEE Transactions on image processing 15, 2 (2006), 430–444. 9, 10

[SDM19] SHAHAM T. R., DEKEL T., MICHAELI T.: SinGAN: Learning a Generative Model from a Single Natural Image. In Proceedings of the IEEE/CVF international conference on computer vision (2019), pp. 4570–4580. 9, 10

[SGG<sup>∗</sup>24] SOMEPALLI G., GUPTA A., GUPTA K., PALTA S., GOLD-BLUM M., GEIPING J., SHRIVASTAVA A., GOLDSTEIN T.: Measuring style similarity in diffusion models, 2024. 3, 9, 10

[SHH98] STUDHOLME C., HAWKES D. J., HILL D. L.: Normalized entropy measure for multimodality image alignment. In Medical imaging 1998: image processing (1998), vol. 3338, SPIE, pp. 132–143. 9, 10

[So23] SO C.: Measuring aesthetic preferences of neural style transfer: More precision with the two-alternative-forced-choice task. International Journal of Human–Computer Interaction 39, 4 (2023), 755–775. 5

[Spe61] SPEARMAN C.: The proof and measurement of association between two things. 9, 18

[TAB<sup>∗</sup>23] TEAM G., ANIL R., BORGEAUD S., ALAYRAC J.-B., YU J., SORICUT R., SCHALKWYK J., DAI A. M., HAUTH A., MILLICAN K., ET AL.: Gemini: a family of highly capable multimodal models, 2023. 4

[TFO23] TSCHUMPERLÉ D., FOUREY S., OSGOOD G.: G’MIC: An Open-Source Self-Extending Framework for Image Processing. 4, 15

[TM98] TOMASI C., MANDUCHI R.: Bilateral filtering for gray and color images. In Sixth International Conference on Computer Vision (1998), pp. 839–846. 4, 15

[WFJ<sup>∗</sup>17] WILBER M. J., FANG C., JIN H., HERTZMANN A., COLLO-MOSSE J., BELONGIE S.: BAM! the behance artistic media dataset for recognition beyond photography. In Proceedings of the IEEE international conference on computer vision (2017), pp. 1202–1211. 2

[WHC<sup>∗</sup>25] WU S., HUANG M., CHENG Y., WU W., TIAN J., LUO Y., DING F., HE Q.: USO: Unified style and subject-driven generation via disentangled and reward learning, 2025. 4, 7

[WLL<sup>∗</sup>25] WANG Y., LIU R., LIN J., LIU F., YI Z., WANG Y., MA R.: OmniStyle: Filtering high quality style transfer data at scale. In Proceedings of the Computer Vision and Pattern Recognition Conference (2025), pp. 7847–7856. 4, 7

[WO22] WRIGHT M., OMMER B.: ArtFID: Quantitative evaluation of neural style transfer. In DAGM German Conference on Pattern Recognition (2022), Springer, pp. 560–576. 1, 2, 3, 9, 10

[WSB03] WANG Z., SIMONCELLI E. P., BOVIK A. C.: Multiscale structural similarity for image quality assessment. In The thrity-seventh asilomar conference on signals, systems & computers, 2003 (2003), vol. 2, Ieee, pp. 1398–1402. 9, 10

[WYZ<sup>∗</sup>25] WANG Y., YI Z., ZHANG Y., ZHENG P., XIE X., LIN J., WANG Y., MA R.: Omnistyle2: Scalable and High Quality Artistic Style Transfer Data Generation via Destylization, 2025. 1, 3

[WZDB22] WU Z., ZHU Z., DU J., BAI X.: CCPL: Contrastive coherence preserving loss for versatile style transfer. In European conference on computer vision (2022), Springer, pp. 189–206. 4

[WZZ<sup>∗</sup>22] WANG Z., ZHANG Z., ZHAO L., ZUO Z., LI A., XING W., LU D.: AesUST: towards aesthetic-enhanced universal style transfer. In Proceedings of the 30th ACM International Conference on Multimedia (2022), pp. 1095–1106. 4

[XLXJ11] XU L., LU C., XU Y., JIA J.: Image smoothing via l 0 gradient minimization. In Proceedings of the 2011 SIGGRAPH Asia conference (2011), pp. 1–12. 4, 15

[XZMB13] XUE W., ZHANG L., MOU X., BOVIK A. C.: Gradient Magnitude Similarity Deviation: A Highly Efficient Perceptual Image Quality Index. IEEE transactions on image processing 23, 2 (2013), 684–695. 9, 10

[YRX<sup>∗</sup>19] YAO Y., REN J., XIE X., LIU W., LIU Y.-J., WANG J.: Attention-aware multi-stroke style transfer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (2019), pp. 1467–1475. 4

[ZIE<sup>∗</sup>18] ZHANG R., ISOLA P., EFROS A. A., SHECHTMAN E., WANG O.: The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition (2018), pp. 586–595. 1, 3, 9, 10

[ZTZ<sup>∗</sup>24] ZHOU Z., TANG F., ZHANG Y., DEUSSEN O., CAO J., DONG W., LI X., LEE T.-Y.: A comprehensive evaluation of arbitrary image style transfer methods. IEEE Transactions on Visualization and Computer Graphics 31, 9 (2024), 5668–5686. 2

[ZZL<sup>∗</sup>22] ZHAO Y., ZHONG Z., LUO Z., LEE G. H., SEBE N.: Source-Free Open Compound Domain Adaptation in Semantic Segmentation. IEEE Transactions on Circuits and Systemsfor Video Technology 32, 10 (2022), 7019–7032. 1

[ZZMZ11] ZHANG L., ZHANG L., MOU X., ZHANG D.: FSIM: A Feature Similarity Index for Image Quality Assessment. IEEE transactions on Image Processing 20, 8 (2011), 2378–2386. 9, 10

# ASTRA-Score: Focused Perceptual Evaluation of Style Transfer

This appendix provides supplementary details and extended analyses to support the main paper. It first describes the construction of ASTRA, including the selection of style images and the generation of stylised results using both learning-based and handcrafted approaches. It then presents the full user study protocol, covering the training phase and quality control procedures to ensure annotation reliability. Additional analyses are provided to validate the data collection process and to further examine the content–style trade-off across different conditions. Finally, implementation details of the proposed ASTRA-Score evaluation model are reported, including feature extraction, regression head design, and training settings.

## Appendix A: Benchmark Data Selection and Generation Details

This section provides a detailed account of the construction of AS-TRA. It outlines the principles guiding the selection of style images, ensuring diversity in artistic characteristics such as colour, texture, structural complexity, and cultural origin. In addition, it describes the implementation details of the LLIP (low-level image processing) methods, which are designed to complement learningbased approaches by approximating styles that are difficult to capture with existing neural models.

## Style Images Selection

Hokusai (nature, low colour, low detail; Asia, 19th century) This iconic woodblock print contains highly recognizable motifs, most notably the sweeping arcs of the crashing wave. It has minimal shading and limited texture, instead relying on strong contour lines and flat regions of solid colour. It has seen use in prior NST evaluation, providing some continuity (original image available at https://commons.wikimedia.org/ wiki/File:The\_Great\_Wave\_off\_Kanagawa.jpg).

Rembrandt (people, low colour, high detail; Europe, 17th century) The chosen image is characterized by strong global contrast and a muted palette. The illuminated areas contain detailed, naturalistic rendering of facial features and fabric, while the background regions are darker and simpler. The combination of high detail, low colour saturation, and straightforward overall composition means that it should be easy to evaluate the effectiveness of stylization while potentially challenging to achieve it (original image available at https: //commons.wikimedia.org/wiki/File:Rembrandt, \_Self\_Portrait\_at\_the\_Age\_of\_34.jpg).

Chinese Landscape (nature, low colour, high detail; Asia, 19th century) This image, in the style of a traditional landscape painting, uses fine brushwork and texture to convey foliage, landscape, and atmospheric depth. The style includes large areas of empty or lightly worked space, which can be difficult for NST methods that assume dense texture. The image is monochromatic, relying strictly on tone and line weight to convey content (original image available at https://commons.wikimedia.org/wiki/File:

%E6%98%8E\_%E6%B2%88%E5%91%A8\_%EF%BC%8C\_%E6%96%87%E5%BE%B5%E6%98%8E-%E5%90%88%E7%92%A7%E5%B1%B1%E6%B0%B4%E5%9C%96\_%E5%8D%B7-Joint\_Landscape\_MET\_DP235652.jpg).

Bierstadt (nature, high colour, high detail; North America, 19th century) The selected landscape is brightly coloured and highly detailed, with dramatic lighting and a strong sense of depth. Although the painting aims for a plausible naturalistic appearance, it remains clearly painterly, but subtly so: methods that depend on crude cues such as severe texture will struggle. The image provides a foil to the Veronese, which is colourful but less dramatic and offers its own challenges (original image available at https://commons.wikimedia.org/ wiki/File:Bierstadt\_Albert\_On\_the\_Saco.jpg).

Picasso (people, low colour, low detail; Europe, 20th century) Guernica is among the most important and influential works of the 20th century. It contains deliberate distortion of human figures, simplified shapes, and a monochromatic palette. Portions of the image resemble line drawing. This image challenges the proposition that style and content can be cleanly separated (original image available at https://commons.wikimedia.org/wiki/File: Mural\_del\_%22Guernica%22\_de\_Picasso.jpg).

Carr (nature, high colour, low detail; North America, 19th century) The selected painting uses bold, simplified forms and an unconventional colour scheme. Geometric distortion and broad, smooth regions dominate the composition. We anticipate that texture will be uninformative for this style, while colour and (when available for a given method) distortion of shape will dominate (original image available at https://commons.wikimedia.org/wiki/ File:Emily\_Carr\_-\_Trees\_in\_France.jpg).

Veronese (people, high colour, high detail; Europe, 16th century) This Renaissance painting contains clearly defined objects, controlled lighting, and a colourful, harmonious palette. Brush texture is minimal; the rendering style is precise. Like the Bierstadt, this image aims at a realistic aesthetic, yet is clearly painterly (original image available at https://commons.wikimedia.org/wiki/File: Paolo\_Veronese\_-\_The\_Marriage\_at\_Cana\_ (detail)\_-\_WGA24859.jpg).

Australian Aboriginal (artificial, high colour, low detail; Australia, 20th century) The selected work, a modern exemplar of a traditional style, incorporates motifs such as dots, circles, and sinusoidal lines. These discrete elements do not form continuous textures, making them difficult for texture-based NST approaches. Success on this style would indicate an ability to reproduce styles with distinct individual elements, such as mosaics (original image available at https://commons.wikimedia.org/ wiki/File:Aboriginal-art-503444\_960\_720.jpg).

Mondrian (artificial, high colour, low detail; Europe, 20th century) This image consists of rectangular regions of primary colours separated by black lines. Its deceptive simplicity depends on exact geometry and precise colour placement. We expect many NST methods to struggle with this style; indeed, while reproducing the style algorithmically is feasible, it is not clear how the rectangular blocks can convey concrete subject matter. We include this image as a stress test and as motivation for future work (original image available at https://commons.wikimedia.org/wiki/File: Piet\_Mondriaan,\_1921\_-\_Composition\_en\_rouge, \_jaune,\_bleu\_et\_noir.jpg).

Kandinsky (artificial, high colour, high detail; Europe, 20th century) The selected painting contains numerous geometric and distinct visual motifs. Although abstract, the elements are flexible enough to be repurposed into representational contexts. The complexity and specificity of these elements make the style difficult to capture (original image available at https: //commons.wikimedia.org/wiki/File:Vassily\_ Kandinsky,\_1939\_-\_Composition\_10.jpg).

Ptolemaic Egyptian (people, colourful, low detail; Africa, 350–300 BCE) This image features flat regions of colour, clear separation between foreground and background, and stylised depictions of human figures. The perspective and proportions follow historical conventions rather than naturalistic ones. These properties test whether methods can reproduce stylised structure without imposing naturalistic shading or texture (original image available at https://commons.wikimedia.org/wiki/File: P1200377\_Louvre\_Stele\_Ousirour\_detail\_D\_ N2699\_rwk.jpg).

Gris (artificial, low colour, low detail; Europe, 20th century) The fragmented shapes, overlapping planes, and occasional symbolic elements in this work make it supremely challenging. The central violin is depicted partly by shape and partly by line drawing. Although the palette is limited, the spatial organization is complex. Balancing content preservation with stylistic transformation will be a severe challenge (original image available https://commons.wikimedia.org/wiki/File: Juan\_Gris\_-\_Komposition\_mit\_Violine.jpeg).

## Details of LLIP Method Construction

The LLIP (low-level image processing) category represents a set of hand-crafted approaches designed to approximate diverse artistic styles. Given the absence of a single model capable of covering all styles, we construct this set using a combination of existing methods, hybrid pipelines, and custom-designed procedures.

## Design Strategies

We adopt four main strategies to construct LLIP stylisations:

• Style-aligned method selection. Existing methods are selected when their visual characteristics align well with the target style (e.g., Kandinsky, Bierstadt).

• Method combination. Multiple methods are combined, either sequentially or by averaging their outputs, to better approximate complex styles (e.g., Rembrandt, Chinese landscape, Veronese).

• Multi-style methods. Some approaches are capable of representing multiple styles and are directly applied (e.g., image analogies for Aboriginal art and The Great Wave).

• Custom pipelines. For styles not well supported by existing methods, custom pipelines are designed (e.g., Mondrian, Carr, Picasso, Ptolemaic Egyptian).

## Style-Specific Implementations

Aboriginal and Great Wave. We employ image analogies [JSO<sup>∗</sup>01] as the base method. For Great Wave, this is further combined with colour transfer using distribution matching [PKD07].

Rembrandt. We combine bilateral filtering [TM98] with a custom colour mapping procedure based on look-up tables (LUT). We applied histogram matching between grayscale versions of the content and style images. For each intensity level, the average RGB colour is computed from the style image to construct a mapping, which is then applied to the content image.

Kandinsky. We use a structured abstraction framework [FXDG17] to produce simplified geometric representations consistent with the style.

Bierstadt. We apply the brushify filter from G’MIC [TFO23]. For images containing foreground objects, the foreground is processed using brushify, while the background is rendered using a watercolour model [RL17].

Veronese. We compute the average of multiple stylisations, including: (1) brushify filtering [TFO23], (2) L smoothing [XLXJ11], (3) outputs from the Waterlogue application, and (4) adaptive morphological filtering [BVFAB22].

Chinese Landscape. We combine brush-based rendering [TFO23] with sketch extraction [LXJ12], and average the resulting outputs.

Ptolemaic Egyptian. A custom pipeline is designed as follows: the image is first blurred and segmented into a fixed number of colour layers based on basic colour categories [BK69]. Textures are added to each layer, and line drawings extracted using coherent line detection [KLC07] are overlaid. The process is applied to both the original and intensity-equalised images, and the results are combined. Background regions are recoloured separately using a reduced palette.

Picasso. A stylised abstraction is constructed by converting the image to grayscale, quantising intensities, and applying median filtering. Line drawings [KLC07] are overlaid, followed by the addition of structured noise and grid-like patterns.

Mondrian. The image is downscaled and each pixel is mapped to a restricted colour palette. Rectangular regions are iteratively extracted and expanded to form block structures, with boundary lines extended to create a grid-like composition.

Carr. We segment the content image with SLIC0, generating medium-large regions that approximately conform to object boundaries. Next, we assign a random painterly texture to each, recolouring the texture to match the average content colour within each region. Note that this emphasizes shape abstraction and not colour; in particular, it does not use the unusual colour distribution characteristic of Carr’s style.

## Appendix B: User Study Protocol

This section describes the design of the user study and the procedures used to ensure the reliability and consistency of the collected annotations. It first outlines the training phase, which establishes a shared understanding of the evaluation criteria among participants. After introducing the three evaluation criteria, few illustrative examples was presented with unambiguous answers (distinct from the main survey), with further details provided below.It then details the quality control mechanisms applied during the study, including attention checks and fatigue mitigation strategies. Together, these components ensure that the resulting annotations are both interpretable and robust for subsequent analysis.

## Training Phase

Before entering the main study, participants completed a mandatory training phase intended to provide a consistent understanding of the evaluation criteria. This phase consisted of an introductory tutorial explaining the key concepts of content, style, and style transfer, supported by both textual descriptions and visual examples.

Participants were required to complete a set of training questions during this phase. Immediate feedback was provided after each response, and participants were required to answer correctly before proceeding. This process gave each participant a minimum level of understanding prior to the evaluation tasks.

Content. The content of an image refers to the scene being represented, regardless of stylistic rendering, as shown in Fig. 10. Participants were instructed to consider all scene elements, including both foreground and background components. While stylistic variations (e.g., blur or abstraction) may affect how these elements are depicted, a style should not introduce new semantic objects. Any such deviation was described as reducing content faithfulness.

Style. The style of an image refers to the artistic characteristics that define its visual appearance, including medium, brushstrokes, texture, and colour usage, as shown in Fig. 11. Participants were informed that style may include colour, particularly when constrained by the medium (e.g., pen-and-ink). However, in some cases colour may also depend on the underlying content rather than stylistic intent, and should therefore be interpreted with care.

Style Transfer. Style transfer was defined as the process of recreating the content of one image using the style of another, as shown in Fig. 12. Participants were instructed to evaluate stylised results based on how well they balance content preservation and style similarity. In particular, they were encouraged to consider an ideal stylisation that faithfully preserves the content while accurately reflecting the target style. Participants were explicitly advised not to select images solely based on subjective preference, but to judge overall quality as a combination of content and style.

![](images/7c1a7245e05be0ae899ab133feed67e52680fdcae85ee5ef11fd366ec492d053.jpg)  
Figure 9: Top-3 and bottom-3 stylised results for each method ranked by overall human preference scores. In each row, the left three images show the top-3 results and the right three show the bottom-3 results. Methods are listed in alphabetical order: AAMS, AdaAttN, AdaIN, ArtFlow, ChatGPT, NST-Ghiasi, Gatys, LLIP, OmniStyle, SANET, StyTR-2, and USO.

Illustrative Examples After introducing the three evaluation dimensions—content preservation, style fidelity, and overall preference—the training phase presents few illustrative examples with clearly distinguishable correct answers, which are distinct from those used in the main survey. For each example, participants are required to make a selection and are immediately provided with feedback indicating whether their choice is correct, along with a brief explanation. Participants must answer each example correctly before proceeding to the next, ensuring that they fully understand the evaluation criteria before entering the main study; examples are shown in Fig. 14.

![](images/9ab9c0d5a5e7328bcc1dcb37d748aed83214dbfcf6130fbac3611355a1bfe075.jpg)  
Figure 10: Illustration of content consistency: the same scene rendered in different styles.

![](images/ec531128fad8a1d643c786a7667d25d293e80f7b7a2143475a51332d7da403c4.jpg)  
Figure 11: Illustration of style consistency: different contents rendered in the same style.

## Quality Control

To ensure the reliability of the collected responses, we applied quality control mechanisms during the main study, detailed as follows.

Attention Checks. The study included dedicated attention-check trials with predefined correct answers for the content and style dimensions. These trials were interleaved with regular evaluation tasks to identify inattentive participants. In Stage 1, each participant completed six attention-check trials. A response was considered correct only if both the content and style questions were answered correctly. Participants were included in the analysis only if they answered at least five out of six attention-check trials correctly. In Stage 2, each participant completed four attention-check trials under the same criteria. Participants were retained only if they answered at least three out of four trials correctly.

Break and Fatigue Control. To mitigate fatigue effects in long sessions, we included a mandatory timed break at the midpoint of the study. The break cannot be skipped and the study resumes automatically after a fixed interval, reducing cognitive load accumulation and maintaining response reliability.

![](images/cd1c887b925264a3c7f622710aeac5eef6cbbbc8777b0e3b479494a8f10b71ed.jpg)  
Figure 12: Example of style transfer: content from one image rendered in the style of another.

![](images/856b47cf60b5e431800c8385cd62601da80eef3ba551c9af30d66192cee77a61.jpg)  
Figure 13: Ranking stability under simulated within-group comparisons (Stage 1). The Spearman correlation between the recovered ranking and the assumed ground-truth ranking increases as the number of comparisons grows. The curve rapidly converges and exceeds $\rho = 0 . 9 9$ at approximately $1 . 2 \times 1 0 ^ { 4 }$ comparisons, indicating that the estimated method ranking becomes highly stable beyond this point. Each point represents the mean correlation over multiple simulation runs.

Response Filtering. After applying the attention-check criteria, 264 of the 407 submitted responses were retained for analysis. To further examine whether the filtering procedure identified loweffort responses, we compared response times between retained and rejected submissions. The median response time per question was 4.526 s for rejected submissions, compared with 8.573 s for retained submissions. Thus, rejected participants responded in approximately half the time of retained participants, providing additional evidence that the attention-check criteria identified inattentive or low-effort responses.

## Annotation Consistency

To quantify the consistency of the collected preference annotations, we randomly selected a subset of comparisons from each stage and obtained duplicate annotations from additional independent annotators. The duplicated comparisons followed the same evaluation protocol as the original study, with annotators independently judging content preservation, style similarity, and overall style-transfer preference.

We measured agreement between the original and additional annotations using Cohen’s κ. For Stage 1, the resulting κ values were 0.663 for content preservation, 0.547 for style similarity, and 0.547 for overall preference. For Stage 2, the corresponding values were 0.515, 0.389, and 0.416, respectively. The higher agreement observed in Stage 1 is consistent with its controlled design, where both stylised results share the same content-style references. In contrast, Stage 2 requires comparisons across different contentstyle pairs and therefore involves a more demanding perceptual judgement.

![](images/c182618bbb54f6a28121f494bcb25a8cff20583bbf2056b566e4866b1356af99.jpg)

(a) Stage 1 illustrative example.  
![](images/3cc4ccb35e51828e68cb2e8773341e06412a38d320233756fd7148044f6edec0.jpg)  
(b) Stage 2 illustrative example.  
Figure 14: Examples of the two illustrative examples used in our user study.

Overall, these results indicate meaningful but task-dependent annotation consistency despite the subjective nature of style-transfer evaluation. Participants were recruited internationally through online platforms under an approved ethics protocol. The present study was not designed to analyse demographic or cultural differences in perceptual preference; investigating the stability of style-transfer preferences across different cultural or user groups therefore remains an important direction for future work.

## Appendix C: Ranking Stability Simulation

Before conducting the user study, we analysed the required number of pairwise comparisons from both theoretical and empirical perspectives. Spectral ranking theory [NOS17, CS15] suggests that reliable ranking estimation under models such as Bradley-Terry-Luce [BT52,L 59] scales on the order of Θ(n log n) comparisons. Using natural logarithm, this corresponds to approximately 12ln12 ≈ 30 comparisons per group for Stage 1 and 864ln864 ≈ $5 . 8 \times 1 0 ^ { 3 }$ comparisons for Stage 2.

These values provided theoretical reference scales rather than exact thresholds, so we further performed simulation analyses under our two-stage design to empirically estimate the number of comparisons needed for stable ranking recovery.

![](images/c79444572a5cdca45230dc5e3d630fbd946f60ec691ebed644b560a34553ba7a.jpg)  
Figure 15: Global ranking recovery under simulated cross-group comparisons (Stage 2). Additional comparisons between items belonging to different content-style pairs gradually improve the accuracy of the recovered global ranking. The Spearman correlation steadily increases as more cross-group comparisons are introduced, demonstrating that sufficient Stage 2 comparisons enable reliable global ranking estimation.

Stage 1: Within-group ranking stability. Stage 1 evaluates style transfer methods under identical content-style conditions. To estimate the number of comparisons required to obtain reliable method preferences, we simulated pairwise comparison processes assuming the existence of an underlying ground-truth ranking among methods. Each simulated comparison followed the ground-truth ordering with stochastic noise to account for potential human inconsistency. The simulated outcomes were aggregated using the Rank Centrality algorithm [NOS17], a Markov-chain-based estimator for ranking from pairwise comparisons, see more details in Section 4.4. Ranking stability was measured using the Spearman correlation [Spe61] between the recovered ranking and the assumed ground-truth ranking.

As shown in Fig. 13, the recovered ranking rapidly stabilises as the number of comparisons increases and exceeds a correlation of ρ = 0.99 at approximately $1 . 2 \times 1 0 ^ { 4 }$ comparisons. In our actual user study, Stage 1 collected 17,568 votes from 122 participants, which lies well within the regime where the ranking estimation is highly stable.

Stage 2: Global rankings. While Stage 1 provides reliable rankings within each content-style group, it does not establish relative ordering across different groups. Stage 2 acquires cross-group comparisons between stylised images belonging to different contentstyle pairs in order to construct a global ranking across all items.

To estimate the number of comparisons required for global ranking, we simulated a synthetic environment in which each item was assigned a latent score determined by method, content, and style effects with additional interaction noise. The combined comparison data are aggregated using Rank Centrality to estimate the global ranking. As shown in Fig. 15, the accuracy of the recovered ranking improves steadily as the number of cross-group comparisons increases. The Spearman correlation reaches approximately 0.95 when the number of Stage 2 comparisons approaches 1.4×10<sup>4</sup> and continues to increase gradually thereafter. In practice, Stage 2 collected 17,040 votes from 142 participants, which lies within the regime where the recovered ranking is already highly correlated with the ground truth.

These analyses provide empirical evidence that the scale of our two-stage user study is sufficient to support reliable preference modelling and global ranking estimation.

## Comparison Graph and Rank Centrality Diagnostics

Empirical graph connectivity. We represent each of the 864 stylised images as a node and add an undirected edge whenever two images are compared at least once. Stage 1 comparisons are restricted to methods sharing the same content-style pair and therefore form 72 disconnected components. Stage 2 was specifically introduced to connect these local components through cross-pair comparisons. After combining the two stages, the empirical comparison graph contains a single connected component covering all 864 nodes, with an average node degree of approximately 45.4. Thus, the comparison graph used for Rank Centrality is connected, and the resulting Markov chain admits a global stationary ranking.

Connectivity under Stage 2 sampling. The connectivity of the 72 content-style groups can also be analysed under the random Stage 2 sampling procedure. For any subset of k groups, disconnection requires that none of the Stage 2 comparisons cross the cut between that subset and the remaining 72−k groups. Using a union bound over all possible disconnected subsets, the probability of a disconnected group-level graph is bounded by

$$
\operatorname* { P r } ( \mathrm { d i s c o n n e c t e d } ) \leq \sum _ { k = 1 } ^ { 3 6 } { \binom { 7 2 } { k } } \left( 1 - { \frac { k ( 7 2 - k ) } { { \binom { 7 2 } { 2 } } } } \right) ^ { 1 7 0 4 0 } \approx 2 . 4 \times 1 0 ^ { - 2 0 7 } .\tag{10}
$$

We additionally simulated the Stage 2 sampling process $1 0 ^ { 6 }$ times and did not observe a disconnected graph in any simulation.

Spectral gap. We compute the spectral gap of the Rank Centrality transition matrix for each evaluation criterion. The resulting gaps are 0.0199 for content preservation, 0.0421 for style similarity, and 0.1000 for overall preference. All three transition matrices therefore have non-zero empirical spectral gaps. In conjunction with the connectivity analysis and the ranking-recovery experiments, these results provide additional diagnostics of the Markov chains underlying the estimated preference rankings.

## Appendix D: ASTRA Analysis

This section provides additional analyses to validate the reliability of the collected data and to further examine the behavioural patterns underlying human preference. It first analyses the exposure statistics of content and style references across methods to verify that the sampling process is approximately balanced and does not introduce systematic bias. It then extends the investigation of the content–style trade-off by presenting per-style and per-content analyses, offering a more detailed view of how preference patterns vary across different conditions. These analyses provide further evidence supporting the robustness of the benchmark and the styledependent nature of the observed trade-offs.

## Exposure Statistics of Content and Style References

To verify that the preference score estimation is not dominated by specific methods, we analyse the exposure frequency between references and methods. Tab. 7 and Tab. 8 report the number of times each style or content reference appears together with each method across all Stage 2 comparisons.

These statistics demonstrate that the sampling distribution is approximately balanced. For style references, each method appears roughly 200–270 times per style. For content references, each method appears approximately 420–530 times per content image. This balanced exposure ensures that method-specific biases are largely averaged out when aggregating pairwise judgements into Rank Centrality scores.

## Additional Analysis of Content-Style Trade-off

To further investigate the content-style trade-off, we extend the analysis presented in the main paper by examining the preference behaviour under all individual style and content conditions.

Per-style Analysis. For each style reference, we compute the style and content win rates for all methods based on Stage 1 pairwise comparisons. Each method is then represented as a point in the content-style plane.

Fig. 16 presents the resulting distributions across a diverse set of styles; each subfigure corresponds to one style reference.

Observations. Some consistent patterns can be observed across styles:

• Abstract styles (e.g., Mondrian, Kandinsky) exhibit a clear inverse relationship between content and style preferences. Methods that achieve stronger stylistic effects often introduce structural distortions, leading to reduced content fidelity.

• Only in Ptolemaic Egyptian case, a positive correlation between content and style preferences can be observed.

• In other styles, no clear trend emerges, indicating that the tradeoff is not universal but highly dependent on the specific style properties.

Per-content Analysis. To further examine whether the observed preferences depends on the input content, we group the results by content images and compute the corresponding content and style win rates for each method.

Fig. 17 presents the resulting distributions for all content images. Each subplot corresponds to one content instance.

## Qualitative Comparison Across Methods.

To complement the quantitative analysis, Fig. 9 shows the top-3 and bottom-3 stylised results for each method ranked by overall human preference.

Top-ranked results generally achieve a better balance between content preservation and stylistic expression, while bottom-ranked results often exhibit either structural distortions or weak style transfer. This behaviour varies across methods: some favour strong style at the cost of content fidelity, whereas others preserve structure but lack distinctive style. These observations are consistent with the Pareto analysis, highlighting the inherent trade-off between the two objectives.

Table 5: Exposure counts between reference styles and style transfer methods in the Stage 1 user study. Each entry reports the number of pairwise comparisons in which a given method appears under the corresponding style reference.
<table><tr><td>Style</td><td>AAMS</td><td>AdaAttN</td><td>AdaIN</td><td>ArtFlow</td><td>ChatGPT</td><td>Gatys</td><td>LLIP</td><td>OmniStyle</td><td>SANET</td><td>StyTR-2</td><td>USO</td><td>NST-Ghiasi</td></tr><tr><td>Australian Aboriginal</td><td>236</td><td>288</td><td>230</td><td>276</td><td>177</td><td>238</td><td>255</td><td>228</td><td>268</td><td>213</td><td>276</td><td>241</td></tr><tr><td>Bierstadt</td><td>229</td><td>249</td><td>280</td><td>225</td><td>248</td><td>281</td><td>201</td><td>241</td><td>276</td><td>263</td><td>248</td><td>227</td></tr><tr><td>Carr</td><td>225</td><td>243</td><td>266</td><td>258</td><td>254</td><td>274</td><td>223</td><td>310</td><td>230</td><td>217</td><td>237</td><td>195</td></tr><tr><td>Hokusai</td><td>215</td><td>268</td><td>321</td><td>272</td><td>221</td><td>256</td><td>232</td><td>263</td><td>209</td><td>221</td><td>214</td><td>232</td></tr><tr><td>Gris</td><td>246</td><td>224</td><td>240</td><td>231</td><td>322</td><td>257</td><td>271</td><td>210</td><td>190</td><td>268</td><td>204</td><td>257</td></tr><tr><td>Picasso</td><td>236</td><td>251</td><td>291</td><td>225</td><td>268</td><td>251</td><td>212</td><td>213</td><td>290</td><td>253</td><td>230</td><td>208</td></tr><tr><td>Ptolemaic Egyptian</td><td>244</td><td>241</td><td>237</td><td>264</td><td>250</td><td>234</td><td>250</td><td>240</td><td>214</td><td>288</td><td>223</td><td>235</td></tr><tr><td>Veronese</td><td>262</td><td>213</td><td>231</td><td>257</td><td>194</td><td>270</td><td>275</td><td>211</td><td>260</td><td>285</td><td>204</td><td>266</td></tr><tr><td>Mondrian</td><td>215</td><td>263</td><td>212</td><td>224</td><td>256</td><td>223</td><td>229</td><td>228</td><td>241</td><td>277</td><td>309</td><td>261</td></tr><tr><td>Rembrandt</td><td>271</td><td>217</td><td>251</td><td>237</td><td>189</td><td>244</td><td>234</td><td>248</td><td>255</td><td>280</td><td>301</td><td>195</td></tr><tr><td>Kandinsky</td><td>226</td><td>269</td><td>252</td><td>254</td><td>259</td><td>248</td><td>264</td><td>267</td><td>220</td><td>234</td><td>224</td><td>199</td></tr><tr><td>Chinese Landscape</td><td>281</td><td>257</td><td>215</td><td>237</td><td>198</td><td>276</td><td>223</td><td>274</td><td>250</td><td>235</td><td>249</td><td>219</td></tr></table>

Table 6: Exposure counts between content images and style transfer methods in the Stage 1 user study. Each entry indicates the number of pairwise comparisons in which a given method appears for the corresponding content image.
<table><tr><td>Content</td><td>AAMS</td><td>AdaAttN</td><td>AdaIN</td><td>ArtFlow</td><td>ChatGPT</td><td>Gatys</td><td>LLIP</td><td>OmniStyle</td><td>SANET</td><td>StyTR-2</td><td>USO</td><td>NST-Ghiasi</td></tr><tr><td>angel</td><td>516</td><td>493</td><td>544</td><td>469</td><td>500</td><td>491</td><td>480</td><td>453</td><td>512</td><td>522</td><td>425</td><td>437</td></tr><tr><td>athletes</td><td>509</td><td>522</td><td>479</td><td>474</td><td>432</td><td>428</td><td>515</td><td>496</td><td>486</td><td>543</td><td>548</td><td>402</td></tr><tr><td>barn</td><td>389</td><td>520</td><td>549</td><td>498</td><td>497</td><td>494</td><td>437</td><td>473</td><td>525</td><td>479</td><td>510</td><td>489</td></tr><tr><td>berries</td><td>507</td><td>506</td><td>459</td><td>562</td><td>462</td><td>523</td><td>443</td><td>516</td><td>458</td><td>496</td><td>431</td><td>491</td></tr><tr><td>daisy</td><td>506</td><td>494</td><td>476</td><td>489</td><td>468</td><td>549</td><td>496</td><td>521</td><td>469</td><td>485</td><td>489</td><td>456</td></tr><tr><td>mac</td><td>459</td><td>448</td><td>519</td><td>468</td><td>477</td><td>567</td><td>498</td><td>474</td><td>453</td><td>509</td><td>516</td><td>460</td></tr></table>

## Appendix E: Training Details of Regression Heads

This section describes the implementation and the optimisation settings used during training. These design choices aim to provide an effective learning framework that aligns predicted scores with human preference signals while maintaining stability and reproducibility. Each regression head is implemented as a lightweight two-layer multilayer perceptron (MLP) with ReLU activations and dropout. The hidden dimension is set to 256 with a dropout rate of 0.2, and the final layer outputs a scalar score.

Each head is trained independently using human-preferencederived targets (Rank Centrality scores). The training setup is as follows:

• Loss function: Smooth $L _ { 1 }$ loss

• Optimizer: AdamW

• Learning rate: $2 \times 1 0 ^ { - 4 }$

• Weight decay: $1 \times 1 0 ^ { - 4 }$

• Batch size: 32

• Training epochs: 40

## Appendix F: Backbone Architecture Ablation

We further investigate the effect of the feature extraction backbone used in ASTRA-Score. We compare architectures from several representative backbone families, including MobileNetV3, VGG, ResNet, ViT, and DINOv2. For all variants, the pretrained backbone is kept frozen and only the task-specific regression heads are trained, such that the comparison primarily reflects the suitability of the extracted representations for modelling human perceptual judgements.

Table 9 reports the results under the leave-one-method-out (LOMO) evaluation protocol. For each backbone, we report the mean and standard deviation of Spearman’s ρ and Kendall’s τ across the held-out stylisation methods for content preservation, style similarity, and overall preference.

VGG11 achieves the strongest performance across all three evaluation criteria. It obtains Spearman correlations of 0.718, 0.714, and 0.667 for content preservation, style similarity, and overall preference, respectively, with corresponding Kendall correlations of 0.540, 0.531, and 0.492. Compared with MobileNetV3, which was also considered as a more compact backbone, VGG11 improves the mean Spearman correlation by 0.032, 0.051, and 0.043 for content, style, and overall preference, respectively.

Increasing backbone capacity does not consistently improve alignment with human judgements. In particular, VGG19 performs below VGG11 across all three criteria, while ResNet50 is competitive for style similarity but remains weaker for content preservation and overall preference. The transformer-based ViT-B/16 and DINOv2 variants likewise do not provide consistent improvements over VGG11. These results indicate that the choice of backbone has a non-negligible effect on perceptual evaluation performance, and motivate our use of VGG11 as the default feature extractor for ASTRA-Score.

Table 7: Exposure counts between style references and methods in Stage 2 comparisons.
<table><tr><td>Style</td><td>AAMS</td><td>AdaAttN</td><td>AdaIN</td><td>ArtFlow</td><td>ChatGPT</td><td>Gatys</td><td>LLIP</td><td>OmniStyle</td><td>SANET</td><td>StyTR-2</td><td>USO</td><td>NST-Ghiasi</td></tr><tr><td>Australian Aboriginal</td><td>214</td><td>254</td><td>264</td><td>246</td><td>232</td><td>237</td><td>254</td><td>228</td><td>229</td><td>202</td><td>250</td><td>269</td></tr><tr><td>Bierstadt</td><td>249</td><td>231</td><td>245</td><td>231</td><td>245</td><td>255</td><td>241</td><td>234</td><td>251</td><td>209</td><td>265</td><td>247</td></tr><tr><td>Carr</td><td>241</td><td>219</td><td>229</td><td>219</td><td>223</td><td>239</td><td>223</td><td>243</td><td>235</td><td>228</td><td>225</td><td>251</td></tr><tr><td>Hokusai</td><td>238</td><td>248</td><td>233</td><td>223</td><td>210</td><td>219</td><td>239</td><td>237</td><td>236</td><td>252</td><td>217</td><td>250</td></tr><tr><td>Gris</td><td>215</td><td>245</td><td>240</td><td>251</td><td>243</td><td>242</td><td>234</td><td>255</td><td>259</td><td>221</td><td>239</td><td>269</td></tr><tr><td>Picasso</td><td>209</td><td>237</td><td>241</td><td>228</td><td>232</td><td>237</td><td>252</td><td>223</td><td>222</td><td>248</td><td>257</td><td>241</td></tr><tr><td>Ptolemaic Egyptian</td><td>226</td><td>235</td><td>240</td><td>252</td><td>238</td><td>239</td><td>229</td><td>224</td><td>223</td><td>213</td><td>241</td><td>273</td></tr><tr><td>Veronese</td><td>233</td><td>238</td><td>262</td><td>247</td><td>220</td><td>233</td><td>260</td><td>267</td><td>224</td><td>247</td><td>238</td><td>228</td></tr><tr><td>Mondrian</td><td>239</td><td>211</td><td>203</td><td>231</td><td>226</td><td>228</td><td>255</td><td>231</td><td>242</td><td>236</td><td>231</td><td>237</td></tr><tr><td>Rembrandt</td><td>201</td><td>240</td><td>231</td><td>253</td><td>241</td><td>249</td><td>219</td><td>244</td><td>228</td><td>231</td><td>258</td><td>220</td></tr><tr><td>Kandinsky</td><td>226</td><td>235</td><td>236</td><td>222</td><td>242</td><td>229</td><td>249</td><td>250</td><td>249</td><td>237</td><td>249</td><td>223</td></tr><tr><td>Chinese Landscape</td><td>213</td><td>257</td><td>236</td><td>253</td><td>222</td><td>235</td><td>230</td><td>255</td><td>219</td><td>218</td><td>217</td><td>264</td></tr></table>

Table 8: Exposure counts between content images and methods in Stage 2 comparisons.
<table><tr><td>Content</td><td>AAMS</td><td>AdaAttN</td><td>AdaIN</td><td>ArtFlow</td><td>ChatGPT</td><td>Gatys</td><td>LLIP</td><td>OmniStyle</td><td>SANET</td><td>StyTR-2</td><td>USO</td><td>NST-Ghiasi</td></tr><tr><td>Angel</td><td>442</td><td>477</td><td>490</td><td>528</td><td>468</td><td>505</td><td>473</td><td>471</td><td>454</td><td>465</td><td>480</td><td>490</td></tr><tr><td>Athletes</td><td>456</td><td>493</td><td>461</td><td>441</td><td>503</td><td>502</td><td>494</td><td>504</td><td>444</td><td>452</td><td>482</td><td>520</td></tr><tr><td>Barn</td><td>473</td><td>508</td><td>478</td><td>500</td><td>486</td><td>470</td><td>510</td><td>454</td><td>497</td><td>452</td><td>480</td><td>466</td></tr><tr><td>Berries</td><td>446</td><td>437</td><td>492</td><td>470</td><td>423</td><td>445</td><td>472</td><td>474</td><td>438</td><td>460</td><td>459</td><td>519</td></tr><tr><td>Daisy</td><td>441</td><td>476</td><td>503</td><td>475</td><td>419</td><td>459</td><td>466</td><td>504</td><td>491</td><td>456</td><td>515</td><td>481</td></tr><tr><td>Mac</td><td>446</td><td>459</td><td>436</td><td>442</td><td>475</td><td>461</td><td>470</td><td>484</td><td>493</td><td>457</td><td>471</td><td>496</td></tr></table>

![](images/e751a02db336780a246caba0d14dc21e5415b9a152564833f6e453eae44d3366.jpg)

![](images/1d5ea23bbc7c8f6a7a36ecd7bc3488ed0579537ee0aa2e2a4f4ea388874b083e.jpg)

![](images/8c653156c6c3de2e7a9824d186ed39353b9a03bce92d32d44afd4744a445a41e.jpg)

![](images/8fe014cc5883e29a7c6ea8cea54372f92c5d4d0ae122642dbce5c1b0c565c98f.jpg)

![](images/f9dac9a3c39e23c58fbe0b59e0ea101d7c9ef0cb879bbf051cb772f2f7eb90da.jpg)

![](images/7ed329a983dd00251001b11f97428ef419a9ae21d2f813d841397d614b464269.jpg)

![](images/9321a38b140641a80e4faa38dd0385f246983e34e662fb6834117fbdf026d87a.jpg)

![](images/2d5de0154e24a21c2522ea64ef9f55dd2fbd12629a8751400e0a244c62ca16bf.jpg)

![](images/07027747e548ef525e52f2ac8cfc9d0860e08cc45ec99d97f16631621bd34ee4.jpg)

![](images/ccec33daa2f5bbaef4c72d4e0cb03960e962cd31e4283a7ad816fe91f8b94f65.jpg)

![](images/49942f2cce32bbe55445db5b4eb24e986401face44de6f9ad0ea5515a02298a8.jpg)

![](images/8252f36a1f20ab5a323970c832bd954640cdc16978697e2e89cf820dc3b1af9f.jpg)  
Figure 16: Content-style preference across different style references. Each subplot corresponds to a specific style, arranged from left to right and top to bottom as: Mondrian, Carr, Chinese Landscape, Kandinsky, Bierstadt, Rembrandt, Picasso, Australian Aboriginal, Hokusai, Ptolemaic Egyptian, Gris, and Veronese. Each point represents a stylisation method positioned by its content and style win rates. The red curve indicates the Pareto frontier of the two objectives, which highlights the optimal trade-off boundary between the two criteria. The relationship between content preservation and style similarity varies significantly across styles, highlighting strong style-dependent behaviour.

![](images/a47f12452c5ac34f8f8b32cc5ce42f3f283dcf21b85d772ae373dd5ab6c1a3a7.jpg)

![](images/aa209e572ba9f534d3fd6e2ec9221d60a5d4e0a6377399a44406c1d271ac52e6.jpg)

![](images/fcbe319f618a3fe971e4ae4d9a9b761fb22bd5947611a4f08d599f04144973be.jpg)

![](images/055780d9ed8ae9dd1ec081963363eb094f8d6a393fcc2b3b2885b345801b5bd0.jpg)

![](images/3d616ed5c9bf29a24162bcc06d10b2d8a7940c9c3accfccedeb145b6a0eb489c.jpg)

![](images/1af1b8e0758fa1a4e1dc55b5849bb0d84e540c7e5f1bc8a27c26fb859b441882.jpg)  
Figure 17: Content-style preference across different content images. Each subplot corresponds to a specific content image, arranged from left to right and top to bottom as: Angel, Athletes, Barn, Berries, Daisy, and Mac. Each point represents a stylisation method positioned by its content and style win rates. The red curve indicates the Pareto frontier of the two objectives, which highlights the optimal trade-off boundary between the two criteria. Compared to the per-style analysis, the distributions remain largely consistent across different contents, indicating that the trade-off is relatively stable with respect to the input image.

Table 9: Backbone ablation for ASTRA-Score under leave-one-method-out (LOMO) evaluation. Results are reported as the mean ± standard deviation of Spearman’s $\rho$ and Kendall’s τ across held-out stylisation methods. The best result for each criterion and correlation metric is shown in bold.
<table><tr><td></td><td colspan="2">Content</td><td colspan="2">Style</td><td colspan="2">Overall</td></tr><tr><td>Backbone</td><td>Spearman</td><td>Kendall</td><td>Spearman</td><td>Kendall</td><td>Spearman</td><td>Kendall</td></tr><tr><td>ResNet18</td><td> $0 . 6 5 8 \pm 0 . 2 0 9$ </td><td> $0 . 4 8 9 \pm 0 . 1 6 4$ </td><td> $0 . 6 7 4 \pm 0 . 0 7 6$ </td><td> $0 . 4 9 7 \pm 0 . 0 6 2$ </td><td> $0 . 5 8 0 \pm 0 . 1 5 6$ </td><td> $0 . 4 1 8 \pm 0 . 1 1 8$ </td></tr><tr><td>ResNet34</td><td> $0 . 6 2 8 \pm 0 . 1 9 8$ </td><td> $0 . 4 6 4 \pm 0 . 1 5 7$ </td><td> $0 . 6 7 2 \pm 0 . 0 7 7$ </td><td> $0 . 4 9 2 \pm 0 . 0 6 3$ </td><td> $0 . 5 6 6 \pm 0 . 1 7 9$ </td><td> $0 . 4 1 0 \pm 0 . 1 3 5$ </td></tr><tr><td>ResNet50</td><td> $0 . 6 5 6 \pm 0 . 1 7 3$ </td><td> $0 . 4 8 8 \pm 0 . 1 3 5$ </td><td> $0 . 7 0 7 \pm 0 . 0 8 8$ </td><td> $0 . 5 2 1 \pm 0 . 0 7 4$ </td><td> $0 . 6 0 8 \pm 0 . 1 5 6$ </td><td> $0 . 4 4 2 \pm 0 . 1 2 3$ </td></tr><tr><td>VGG19</td><td> $0 . 6 7 1 \pm 0 . 1 3 8$ </td><td> $0 . 4 9 1 \pm 0 . 1 1 5$ </td><td> $0 . 6 5 9 \pm 0 . 0 8 2$ </td><td> $0 . 4 8 0 \pm 0 . 0 6 7$ </td><td> $0 . 6 2 2 \pm 0 . 1 1 9$ </td><td> $0 . 4 5 3 \pm 0 . 0 9 5$ </td></tr><tr><td>ViT-B/16</td><td> $0 . 6 9 1 \pm 0 . 1 5 2$ </td><td> $0 . 5 1 3 \pm 0 . 1 2 7$ </td><td> $0 . 6 4 8 \pm 0 . 1 3 6$ </td><td> $0 . 4 7 5 \pm 0 . 1 1 5$ </td><td> $0 . 5 0 5 \pm 0 . 1 6 4$ </td><td> $0 . 3 6 0 \pm 0 . 1 2 8$ </td></tr><tr><td>DINOv2</td><td> $0 . 6 0 4 \pm 0 . 1 8 0$ </td><td> $0 . 4 3 9 \pm 0 . 1 5 0$ </td><td> $0 . 5 7 0 \pm 0 . 1 5 6$ </td><td> $0 . 4 1 0 \pm 0 . 1 2 1$ </td><td> $0 . 4 8 1 \pm 0 . 2 5 4$ </td><td> $0 . 3 4 5 \pm 0 . 1 8 7$ </td></tr><tr><td>MobileNetV3</td><td> $0 . 6 8 6 \pm 0 . 1 6 1$ </td><td> $0 . 5 1 2 \pm 0 . 1 3 6$ </td><td> $0 . 6 6 3 \pm 0 . 1 2 8$ </td><td> $0 . 4 9 0 \pm 0 . 1 0 6$ </td><td> $0 . 6 2 4 \pm 0 . 1 0 9$ </td><td> $0 . 4 5 6 \pm 0 . 0 8 9$ </td></tr><tr><td>VGG11</td><td> ${ \bf 0 . 7 1 8 \pm 0 . 1 4 4 }$ </td><td> ${ \bf 0 . 5 4 0 \pm 0 . 1 2 8 }$ </td><td> ${ \bf 0 . 7 1 4 \pm 0 . 0 9 8 }$ </td><td> $\mathbf { 0 . 5 3 1 \pm 0 . 0 8 4 }$ </td><td> $\mathbf { 0 . 6 6 7 \pm 0 . 0 9 9 }$ </td><td> $\mathbf { 0 . 4 9 2 \pm 0 . 0 8 1 }$ </td></tr></table>