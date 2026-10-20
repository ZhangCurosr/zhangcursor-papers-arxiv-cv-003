# Not All Similarity Is Leakage: Distribution-Aware Identity Leakage Assessment in Synthetic Face Datasets

Paweł Borsukiewicz   
University of Luxembourg Luxembourg   
pawel.borsukiewicz@uni.lu

Fadi Boutros Fraunhofer IGD Darmstadt, Germany fadi.boutros@igd.fraunhofer.de

Daniele Lunghi   
University of Luxembourg   
Luxembourg   
daniele.lunghi@uni.lu   
Jacques Klein   
University of Luxembourg   
Luxembourg   
jacques.klein@uni.lu

Wendkuuni C. Ou ˆ edraogo´ University of Luxembourg Luxembourg wendkuuni.ouedraogo@uni.lu

Tegawende F. Bissyand ´ e´ University of Luxembourg Luxembourg

tegawende.bissyande@uni.lu

## Abstract

Synthetic data provides the meansfor more ethical and less biased facial recognition. Nonetheless, the limitations of generative models and the conditioning process they rely on pose a risk of identity leakage, resulting in synthetic samples that are overly similar to those seen in a generator’s training set, mitigating privacy benefits. In this work, we show that previously proposed approaches are affected by two key limitations: high classifier dependence and a tendency to identify leakage with respect to real datasets that were not used in training. To address these shortcomings, we propose a novel metric called Scaled Overlap (SO), which adapts a non-parametric data-copying testforfacial recognition and extends it to the identity level. By leveraging a multi-model setup that compensatesfor errors in individual classifiers, it provides insights into the true strength of identity leakage. Our analysis identifies leaking datasets (CemiFace, SFace2, IDNet) with the most acute leakage introduced by generative model conditioning in CemiFace, demonstrates that real datasets do not contribute equally to generator training, and shows that the majority of synthetic images are drawnfrom multiple identities rather than relying on a single identity. Ultimately, our study provides new insights into the quality ofsynthetic datasets and gives more reliable techniques to help the community ensure the trustworthiness ofsynthetic data.

## 1. Introduction

Facial recognition is a morally ambiguous field. Some view it as a means to ensure safety and order, others scrutinize ethical and privacy aspects [41] of data collection and processing. In a search for privacy-preserving facial recognition solutions [36] that address concerns of the latter, one stands out: synthetic data. Given its ability to address biases, label noise, and the limitations in the availability of real data [3], the performance of the models trained on synthetic data has been one of the main factors preventing its use in the early days. Now, as fully synthetic models can compete with their real counterparts [10, 60], they are attracting increasing attention [4, 5, 8, 14, 49, 57].

Nonetheless, synthetic data is not free from limitations. One critical bottleneck is its reliance on real data for generative model training or conditioning. Namely, a generator may produce a sample that resembles an underlying image, leading to a phenomenon known as identity leakage [54]. Some of the works have incorporated methods to explicitly address this issue, such as removal of overly similar samples [43, 52, 58], but often with limited success [31].

While previous studies have highlighted the issue [3, 48], the analysis of its true extent has been limited, as it primarily focused on generator training and relied on a single pretrained facial recognition (FR) model for assessment. In this work, we propose a more comprehensive approach. First, we use five recognizers individually and as an ensemble to find that using previously proposed methods, leakage is observed in datasets unseen by the generator during training, a phenomenon we call Phantom Leakage. By showing its prevalence (100% leakage at FMR=0.1% across multiple FR models and dataset pairs), we call into question the faithfulness of the currently established methodology for measuring identity leakage. In search of more trustworthy metrics, we propose Scaled Overlap (SO), accounting for the natural similarity between distinct identities and datasetspecific distributional properties. Contrary to the similaritybased approach, our experimental results demonstrate that SO is capable of localizing true instances of leakage. The main contributions of this work are as follows:

• Demonstration of the systematic failure of currently used leakage estimation methods.

• Scaled Overlap, an adaptation of the non-parametric datacopying test to face identity space, with an identity-level formulation absent from prior sample-level evaluations.

• Comprehensive identity leakage evaluation in 9 synthetic datasets, revealing strong identity-level and limited sample-level leakage in CemiFace, stemming from its generative model conditioning procedure.

• Publicly available replication package<sup>1</sup>.

Our methodology and findings will help the community evaluate their datasets and, consequently, build trust in synthetic data for facial recognition.

## 2. Related Work

Recent developments in the domain of generative AI, including Generative Adversarial Networks (GANs) [23] and Diffusion Models (DMs) [25], have allowed for creation of high-quality synthetic images. This, in turn, has addressed many long-standing shortcomings of real data, including data scarcity and labeling errors. It was also believed to address privacy issues in sensitive domains such as facial recognition. However, recent works [3, 48, 54] have shown that original identities are being revealed through generated samples. In the following subsections, we summarize advancements in the domain of synthetic facial recognition datasets and the identity-leakage threat they pose.

## 2.1. Synthetic Facial Recognition Datasets

Facial recognition practitioners have, for a long time, faced criticism [51] regarding ethically questionable data acquisition practices, which have typically relied on mass scraping of web databases [55, 61]. Synthetic data has emerged as a compelling alternative [8], mitigating well-known shortcomings of real datasets [3], including label noise, class imbalance, and potential non-compliance with data protection regulations like the GDPR [19].

Early works, such as Syn-1M [34] and Syn-Multi-PIE [12], have struggled to achieve reliable recognition accuracy, yet subsequent works have continued to push the performance boundaries. More recent methodologies and resulting datasets, including IDPERTURB [10] and Vari-Face [60], have shown remarkable improvements, with the derived models outperforming their counterparts trained on the real CASIA-WebFace [61] dataset.

While synthetic facial image generation techniques vary widely, there are three main categories of approaches [3]. First focuses on the use of GANs. SynFace [45] leveraged DiscoFaceGAN [16], while another set of works (ID-Net [32] and SFace2 [9]) relied on a StyleGAN2-Ada. The second group concentrates on the use of conditioned DMs, and it includes datasets such as IDiff-Face [7], DCFace [30], CemiFace [52], UIFace [37], or $\mathrm { I D ^ { 3 } }$ [59]. Finally, the third approach used 3D modeling, either independently, as in DigiFace-1M [2], or in combination with DMs, as in Digi2Real [21], which supplemented it with Arc2Face [43].

With the ability to address the limitations of real data and deliver reliable performance, synthetic data is on track to become the new standard.

## 2.2. Identity Leakage

Generative models, both GANs and DMs, have been shown to reproduce samples seen in training [11, 56]. In our context, we seek samples in which the identities of the generated images are highly similar to those of the real images used to train or condition the corresponding generator, so that facial recognition models assign them sufficiently high similarity scores to be treated as the same identity. Current protective approaches [43, 52, 58] do not fully address this leakage issue, but rather minimize it. As such, identities in datasets used to train and/or condition generative models, such as commonly used CASIA-WebFace [61], FFHQ [28], variants of WebFace [62], or CelebA [38], are at risk.

Tinsley et al. [54] have drawn attention to the identity leakage in StyleGAN2 [29], trained on FFHQ. Through analysis encompassing 69,372 synthetic images and 57,094 real samples, they have measured two non-mated distributions: ”Real-Real” (R-R) and ”Real-Generated” (R-G) across five facial recognition models. Their findings highlighted existing yet limited identity leakage and pointed to disagreements among individual recognizers.

Singh et al. [50] have proposed SynthProv, a framework to evaluate identity leakage based on StyleGAN2 latent space, linking synthetic samples with the real identities they follow the most closely. By combining latent identity dissimilarity and facial recognizer embedding similarity, they computed image-specific provenance scores that indicated which images most influenced each synthetic sample. This approach, while highly interpretable at the individual level, relies on the assumption that the generator is publicly available, which may not hold in practice.

Shahreza and Marcel [48] have demonstrated that synthetic facial recognition datasets can expose underlying training identities. In their work, a variant of the Membership Inference Attack (MIA) was designed to identify generated samples across six synthetic facial recognition datasets that are overly similar to the generator’s training sets. Using a pretrained ResNet-100 [24] model, they computed cosine similarity against the training data and identified multiple outliers with unnaturally high scores.

Borsukiewicz et al. [3] have used pre-trained IRes-

Net100 [15] to simultaneously evaluate and quantify leakage across all publicly available datasets at the time. The study has found examples of visually similar images in ID-Net [32], Vec2Face [58], and IDiff-Face [7]. Given that real datasets commonly overlap due to web scraping, the authors have focused only on comparisons involving faces in CASIA-WebFace [61]. This simplified approach has omitted potential identity leaks from the source training datasets.

## 2.3. Generator Conditioning

Conditioning is the process of supplying a generator with additional input that steers it toward a specific output rather than a random sample from the learned distribution [7, 30, 37]. It is what allows the generated samples to preserve a consistent identity while other attributes, such as style, vary.

The origin of this conditioning signal varies widely across the literature [3]. Some works sample identities from the latent space of a pretrained GAN [12], or from a disentangled 3DMM identity prior learned from real faces [45]. Others define identity in the embedding space of a pretrained face recognition model [7, 52], or condition the generator on real identity labels during training [9, 32]. Still others draw the identity from an unconditional generator and mix it with style references taken from real images [7, 30, 37, 59], or reuse synthetic identities from prior work, diluting dependence on real data [21]. Arc2Face [43] applies Principal Component Analysis (PCA) to the embedding space of a pretrained face recognition model, samples novel identity embeddings within the resulting PCA space, and then uses a pretrained identity-conditioned diffusion model to generate corresponding face images. VIG-Face [31] further advances virtual identity generation in the embedding space by constructing virtual identity prototypes explicitly disentangled from real identities, which are used as conditioning signals for a diffusion-based face generator.

Prior studies on identity leakage (2.2) have focused primarily on the generator’s training data, leaving the conditioning path largely implicit. We argue that conditioning constitutes a second, distinct channel, since the identity signal itself may be derived from, or defined by, real identities. To distinguish the two, we refer to them as training-data leakage and identity-space leakage, respectively.

## 3. Methodology and Experimental Setup

To ensure a more thorough analysis than the previous works [3, 48, 50, 54], we design our methodology to support a wide range of evaluation models and synthetic datasets. Through standardized analysis, rooted in distributional properties of real and synthetic data, we aim to demonstrate the true extent of the privacy leakage problem.

## 3.1. Classical Leakage Estimation

To estimate leakage in a classical way, we follow the approaches introduced in prior work [3, 48]. Our procedure can be briefly summarized as follows:

❶ Compute embeddings for real and synthetic datasets.

❷ For each model used in evaluation, find key decision thresholds based on real data.

❸ For each synthetic image, find a closest real sample.

❹ Compute a proportion of real-synthetic pairs with embedding similarity scores exceeding the decision threshold.

First, for each selected model and dataset (details in 3.3 and 3.4, respectively), we perform a forward pass to compute embeddings (❶). We use non-mated LFW [26] pairs from its evaluation protocol to determine operational decision thresholds (❷) based on fixed FMR values (we used Frontex-recommended [20] operational minimum of 0.1% in the main manuscript; an ablation study of alternative thresholds is provided in the supplementary material). Even if datasets contain a single image per identity (e.g., FFHQ [28]), we can still compute them, since we perform only distinct-identity comparisons. Subsequently, for each synthetic image embedding, we use FAISS [18] to efficiently find a real image embedding with the highest cosine similarity (❸). This score is further used to measure leakage as the percentage of synthetic samples for which exists a pair that exceeds previously computed thresholds (❹).

## 3.2. Scaled Overlap

We propose measuring identity leakage in a synthetic dataset S by examining distributional irregularities relative to a real dataset R in a multi-model setup. Our hypothesis is that leakage traces, which can be confused at the level of individual images, should remain consistently detectable at the dataset level with statistical measures.

First, we search for the closest sample in the dataset R for each sample in the datasets S and R, excluding mated (same identity) comparisons for real vs. real pairs. To reduce the impact of recognizer-related errors, we leverage an ensemble approach that averages similarity scores across a set of classifiers. The resulting distributions of similarity scores $f _ { R } ( x )$ and $f _ { S } ( x )$ reflect the natural identity-space diversity of the training dataset and indicate how closely the synthetic data mimics a real-life dataset, respectively. Said resemblance is, to a certain extent, intended, as the generator learns to draw from the real-life distribution it has seen during training. However, when it becomes excessive, we observe identity leakage. To quantify this, we adapt the Mann-Whitney U statistic [39], which underlies the data-copying test of Meehan et al. [40], by replacing their held-out reference sample, unavailable when generators are trained on the full corpus, with leave-one-identityout nearest-neighbor similarity within the real dataset. We refer to the result as Scaled Overlap (SO):

Table 1. Synthetic datasets, their deep generative models’ training datasets, and the real dataset shaping the identity space (i.e., the pretraining data of the FR encoder used for identity conditioning, or the label source when identities are used directly as classes).
<table><tr><td>Synthetic dataset</td><td>Total images</td><td>Generator training data</td><td>Total images</td><td>Underlying identity space</td><td>Total images</td></tr><tr><td>SynFace [45]</td><td>1M</td><td>FFHQ</td><td>70k</td><td>FFHQ†</td><td>70k</td></tr><tr><td>IDiff-Face [7]</td><td>503k</td><td>FFHQ</td><td>70k</td><td>MS1MV2</td><td>5.8M</td></tr><tr><td>SynMulti-PIE [12]</td><td>149k</td><td>FFHQ</td><td>70k</td><td>FFHQ†</td><td>70k</td></tr><tr><td>IDNet [32]</td><td>1.06M</td><td>CASIA-WebFace</td><td>494k</td><td>CASIA-WebFace*</td><td>494k</td></tr><tr><td>CemiFace [52]</td><td>550k</td><td>CASIA-WebFace</td><td>494k</td><td>WebFace4M‡</td><td>4.2M</td></tr><tr><td>SFace2 [9]</td><td>1.05M</td><td>CASIA-WebFace</td><td>494k</td><td>CASIA-WebFace*</td><td>494k</td></tr><tr><td>DCFace [30]</td><td>550k</td><td>FFHQ + CASIA-WebFace</td><td>564k</td><td>CASIA-WebFace/WebFace4M‡</td><td>494k/4.2M</td></tr><tr><td>HyperFace [42]</td><td>640k</td><td>FFHQ + CelebA + WebFace4M</td><td>4.5M</td><td>MS1MV2/Glint360K/WebFace42M‡</td><td>5.8M/17.1M/42M</td></tr><tr><td>Digi2Real [21]</td><td>400k</td><td>FFHQ + CelebA + WebFace42M</td><td>42.2M</td><td>Head scans§</td><td>511</td></tr></table>

<sup>†</sup>No identity encoder. Identity drawn from a learned prior over real faces. <sup>‡</sup>Inherited from a pretrained identity encoder. <sup>⋆</sup>Identity used as a class label for the identity-conditioned generator. <sup>§</sup>Identities from DigiFace-1M’s [2].

$$
\operatorname { S O } ( S , R ) = 1 0 0 * \int _ { - 1 } ^ { 1 } f _ { S } ( x ) \cdot F _ { R } ( x ) d x\tag{1}
$$

Using the probability density function $f _ { S } ( x )$ of mean similarity scores x observed for a given synthetic dataset and the function $F _ { R } ( x )$ that measures the percentile of real data distribution $f _ { R } ( x )$ at x, we quantify the synthetic distribution in the context of the natural distribution. As such, two perfectly overlapping distributions would result in a value of 50, meaning synthetic samples are no more similar to their closest real samples than the closest real samples are to each other. Values above 50 indicate that synthetic samples tend to be more similar to their nearest real counterpart than most similar real samples are to one another, signaling dataset-level leakage.

We can also apply SO at the identity level by computing identity centroids $c _ { k } \mathrm { : }$

$$
\mathbf { c } _ { k } = \frac { 1 } { | E _ { k } | } \sum _ { \mathbf { e } _ { i } \in E _ { k } } \mathbf { e } _ { i }\tag{2}
$$

where $E _ { k }$ is the set of embeddings belonging to the identity k. To project the newly obtained point on the hypersphere, $c _ { k }$ is subsequently normalized.

## 3.3. Models

To assess the impact of model choice on the leakage analysis, we employ a combination of Convolutional Neural Networks (CNNs) [35] and Vision Transformers (ViTs) [17]. We deliberately select a diverse set of pre-trained models with varying capabilities: re-implementation [53] of FaceNet [47], version trained on CASIA-WebFace [61]; IResNet18 [15, 27] variant trained on Glint360K [1]; Swin-Face [44]; base EdgeFace [22]; and Glint360K-trained S variant of TransFace [13]. Weights and sources are listed in our replication package. As facial-feature evaluators, these models should be sensitive to genuine instances of leakage, while differences in backbone architecture and training procedures are likely to produce model-specific errors to be partially rectified by an ensemble approach.

## 3.4. Datasets

We evaluate our methodology across a wide range of publicly available synthetic datasets (Table 1) for which generator training data are known. Due to the prevalence of CASIA-WebFace (C-WF) [61], FFHQ [28], CelebA [38], and WebFace4M (WF4M) [62] in generator training and their relatively compact size compared with other potential datasets, we primarily focus our analysis on datasets derived from them. Nevertheless, our methodology can be extended to any generative model, provided we have access to its training data and sufficient storage space.

## 4. Results

We designed our study to answer the following research questions:

① How reliable is the current SOTA leakage analysis?

② How does Scaled Overlap characterize identity leakage across synthetic face datasets?

③ How does identity leakage manifest in the distributional structure of embeddings?

We start by highlighting the limitations of previous approaches [3, 48, 54] and subsequently demonstrate that SO is a more reliable alternative for leakage estimation. Having established its fidelity, we conduct further statistical tests to analyze leakage across nine synthetic datasets.

## 4.1. Limitations of SOTA Leakage Analysis

We began by computing identity leakage at FMR = 0.1% as described in 3.1. (Table 2). A multi-model setup has provided us with very concerning findings. Leakage values are highly classifier-dependent: The same dataset, HyperFace, can achieve as low as 0.3 on CelebA, according to IResNet18, and as high as 97.4, according to EdgeFace. This variability demonstrates that a single-model approach with a dependence on a single FMR threshold is fundamentally unreliable and results are meaningless. We break down the dependence of leakage on the FMR threshold and model in the supplementary material. Leakage appears in datasets unseen during generator training and conditioning: We observe substantial leakage in the datasets where it should not exist, a phenomenon we call Phantom Leakage. Most notably, for each evaluated synthetic dataset, all or nearly all (99.9%+) samples are classified as too similar to C-WF and WF4M. Furthermore, the leakage can be deemed extremely high, but the closest non-mated real samples are still more similar than the synthetic samples (Figure 1), putting into question whether observed similarity is a true instance of leakage or simply a naturally occurring phenomenon.

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">C-WF</td><td colspan="3">IResNet18 CelebA FFHQ</td><td rowspan="2">C-WF</td><td colspan="3">TransFace</td><td rowspan="2"></td><td colspan="3">EdgeFace</td><td colspan="4">5-model ensemble</td></tr><tr><td></td><td></td><td>WF4M</td><td>CelebA FFHQ</td><td></td><td>WF4M</td><td>C-WF</td><td>CelebA</td><td>FFHQ</td><td>WF4M C-WF</td><td>CelebA FFHQ</td><td></td><td>WF4M</td></tr><tr><td>CemiFace</td><td>100.0</td><td>0.4†</td><td>1.2†</td><td>100.0</td><td>100.0</td><td>88.5†</td><td>84.4†</td><td>100.0</td><td>100.0</td><td>99.1†</td><td>98.5†</td><td>100.0</td><td>100.0</td><td>19.7†</td><td>25.1†</td><td>100.0</td></tr><tr><td>DCFace</td><td>100.0</td><td>0.2†</td><td>1.2</td><td>100.0</td><td>100.0</td><td>87.7†</td><td>85.5</td><td>100.0</td><td>100.0</td><td>99.4†</td><td>99.1</td><td>100.0</td><td>100.0</td><td>13.6†</td><td>26.2</td><td>100.0</td></tr><tr><td>Digi2Real</td><td>100.0†</td><td>0.7</td><td>3.6</td><td>100.0</td><td>100.0†</td><td>86.4</td><td>91.2</td><td>100.0</td><td>100.0†</td><td>98.3</td><td>99.4</td><td>100.0</td><td>100.0†</td><td>11.8</td><td>33.0</td><td>100.0</td></tr><tr><td>HyperFace</td><td>100.0†</td><td>0.3</td><td>7.5</td><td>100.0</td><td>100.0†</td><td>57.4</td><td>88.1</td><td>100.0</td><td>100.0†</td><td>97.4</td><td>99.4</td><td>100.0</td><td>100.0†</td><td>8.8</td><td>42.4</td><td>100.0</td></tr><tr><td>IDNet</td><td>100.0</td><td>0.8†</td><td>2.0†</td><td>100.0†</td><td>100.0</td><td>93.7†</td><td>82.5†</td><td>100.0†</td><td>100.0</td><td>99.1†</td><td>95.9†</td><td>100.0†</td><td>100.0</td><td>33.7†</td><td>23.1†</td><td>100.0†</td></tr><tr><td>IDiffFace</td><td>100.0†</td><td>1.6†</td><td>19.6</td><td>100.0†</td><td>100.0†</td><td>51.2†</td><td>89.3</td><td>100.0†</td><td>100.0†</td><td>97.9†</td><td>99.7</td><td>100.0†</td><td>100.0†</td><td>18.8†</td><td>68.4</td><td>100.0†</td></tr><tr><td>SFace2</td><td>100.0</td><td>0.7†</td><td>1.1†</td><td>100.0†</td><td>100.0</td><td>96.2†</td><td>85.2†</td><td>100.0†</td><td>100.0</td><td>99.8†</td><td>97.4†</td><td>100.0†</td><td>100.0</td><td>37.7†</td><td>23.2†</td><td>100.0†</td></tr><tr><td>SynFace</td><td>100.0†</td><td>0.5†</td><td>9.1</td><td>100.0†</td><td>100.0†</td><td>87.2†</td><td>98.7</td><td>100.0†</td><td>100.0†</td><td>99.1†</td><td>100.0</td><td>100.0†</td><td>100.0†</td><td>21.1†</td><td>76.9</td><td>100.0†</td></tr><tr><td>SynMulti-PIE</td><td>100.0†</td><td>0.2†</td><td>7.5</td><td>100.0†</td><td>100.0†</td><td>55.9†</td><td>90.6</td><td>100.0†</td><td>100.0†</td><td>98.9†</td><td>99.9</td><td>100.0†</td><td>100.0†</td><td>15.4†</td><td>62.7</td><td>100.0†</td></tr></table>

† indicates evaluations where the real dataset was NOT used in either the generator training data or the underlying identity space.  
Table 2. Percentage of synthetic samples exceeding FMR=0.1% threshold across real datasets. Extended table in supplementary material.

![](images/4ac5c7eded5eebe55535d98d15a8462e2ef7828791b7c2cecbdfe9e5576c33f8.jpg)  
Figure 1. Mated and non-mated similarity distributions of CASIA-WebFace compared with its closest non-mated samples and closest CemiFace samples according to EdgeFace. FMR=0.1% threshold (0.2316) is exceeded by both closest pair distributions.

Consequently, we highlight the need for alternative approaches to leakage detection, ones that are free from the limitations of facial recognizer-related inconsistencies and capable of distinguishing real leakage from the naturally occurring similarity between distinct identities.

## 4.2. Scaled Overlap

To showcase the fidelity of the SO metric, we intentionally introduced leakage $( S O _ { l e a k } )$ into a synthetic dataset by mixing it with a specified percentage p of real samples from real datasets (Figure 2). These images simulate a scenario in which a generator creates a picture of an identity belonging to its training set. To ensure methodological soundness, leakage samples are excluded from the original real dataset during evaluation. Results for the HyperFace subset (100k samples) show that the SO scales linearly with the percentage of leaking data. This is an expected behavior, as:

![](images/19a80ff7fe158ba853a32339df9ce10da45879e76cf835a2d13b8bb0a7e64fe7.jpg)  
Figure 2. HyperFace with intentionally introduced leakage. SO measured at the image level.

$$
S O ( p ) = ( 1 - p ) S O _ { s y n t h } + p \cdot S O _ { l e a k }\tag{3}
$$

At the 0% mark, where all evaluated samples are synthetic, we observe nonzero overlap scores $( S O _ { s y n t h } )$ , resulting from a combination of the generative model’s distribution learning and potential true leakage. It is important to note that SO scales differently for FFHQ, with every image treated as a unique identity due to the lack of annotations, confirming that the no-leakage reference point for real data is 50. Consequently, SO should be interpreted as a prevalence-weighted indicator of dataset-level leakage, rather than a direct count of leaked identities.

Subsequently, we computed SO across all real vs. synthetic dataset pairs to search for leaking datasets (Table 3). Three of the potentially leaking (candidate) dataset comparisons yield the highest SO values, ranging from slightly elevated (IDNet: 57.1, SFace2: 68.5) to highly elevated (Cemi-Face: 93.3). By surpassing the 50-point mark, synthetic identities are, on average, more similar to the closest real ones than the closest real identities are among themselves. This constitutes direct evidence of dataset-level identity leakage. Additionally, we observed that the strongest instance of leakage (CemiFace) originated from the identityspace match, demonstrating that generative model conditioning can be more disruptive than its training, a factor that was previously overlooked.

<table><tr><td>Synthetic Dataset</td><td>Test Set</td><td>Leakage Candidate</td><td>Sample-level so</td><td>Identity-level so</td></tr><tr><td>HyperFace</td><td>CelebA</td><td>Yes</td><td>0.2</td><td>0.0</td></tr><tr><td>SynMulti-PIE</td><td>CelebA</td><td>No</td><td>0.3</td><td>0.0</td></tr><tr><td>Digi2Real</td><td>CelebA</td><td>Yes</td><td>0.4</td><td>0.1</td></tr><tr><td>DCFace</td><td>CelebA</td><td>No</td><td>0.4</td><td>0.1</td></tr><tr><td>SynFace</td><td>CelebA</td><td>No</td><td>0.6</td><td>0.1</td></tr><tr><td>CemiFace</td><td>CelebA</td><td>No</td><td>0.6</td><td>0.4</td></tr><tr><td>IDiffFace</td><td>CelebA</td><td>No</td><td>0.6</td><td>0.6</td></tr><tr><td>IDNet</td><td>CelebA</td><td>No</td><td>1.3</td><td>1.3</td></tr><tr><td>SFace2</td><td>CelebA</td><td>No</td><td>1.4</td><td>2.3</td></tr><tr><td>HyperFace</td><td>C-WF</td><td>No</td><td>5.3</td><td>5.1</td></tr><tr><td>Digi2Real</td><td>C-WF</td><td>No</td><td>8.2</td><td>6.8</td></tr><tr><td>SFace2</td><td>FFHQ</td><td>No</td><td>4.1</td><td>9.6</td></tr><tr><td>Digi2Real</td><td>FFHQ FFHQ</td><td>Yes No</td><td>6.8 4.4</td><td>10.5</td></tr><tr><td>IDNet</td><td>FFHQ</td><td>Yes</td><td></td><td>10.8</td></tr><tr><td>DCFace</td><td>FFHQ</td><td>No</td><td>5.0 4.7</td><td>12.6</td></tr><tr><td>CemiFace</td><td>WF4M</td><td></td><td>14.7</td><td>13.3</td></tr><tr><td>HyperFace</td><td>C-WF</td><td>Yes</td><td></td><td>14.2</td></tr><tr><td>SynMulti-PIE</td><td>FFHQ</td><td>No</td><td>11.8</td><td>15.2</td></tr><tr><td>HyperFace</td><td>C-WF</td><td>Yes</td><td>10.4</td><td>15.5</td></tr><tr><td>IDiffFace</td><td>WF4M</td><td>No</td><td>8.4 16.9</td><td>15.6</td></tr><tr><td>SynMulti-PIE</td><td>WF4M</td><td>No</td><td></td><td>18.1</td></tr><tr><td>Digi2Real</td><td>C-WF</td><td>Yes</td><td>24.3</td><td>20.4</td></tr><tr><td>SynFace</td><td>FFHQ</td><td>No Yes</td><td>13.0 13.3</td><td>20.8 22.0</td></tr><tr><td>SynMulti-PIE SynFace</td><td>FFHQ</td><td>Yes</td><td>18.9</td><td>33.3</td></tr><tr><td>IDiffFace</td><td>FFHQ</td><td>Yes</td><td>21.3</td><td>38.8</td></tr><tr><td>SynFace</td><td>WF4M</td><td>No</td><td>30.0</td><td>39.1</td></tr><tr><td>SFace2</td><td>WF4M</td><td>No</td><td>26.9</td><td>42.4</td></tr><tr><td>DCFace</td><td>C-WF</td><td></td><td>20.4</td><td>43.4</td></tr><tr><td>CemiFace</td><td>C-WF</td><td>Yes Yes</td><td>19.3</td><td>44.4</td></tr><tr><td>IDNet</td><td>WF4M</td><td></td><td>29.2</td><td>44.9</td></tr><tr><td>DCFace</td><td>WF4M</td><td>No Yes</td><td>27.5</td><td>45.3</td></tr><tr><td></td><td></td><td></td><td>26.8</td><td>47.6</td></tr><tr><td>IDiffFace</td><td>WF4M</td><td>No</td><td></td><td></td></tr><tr><td>IDNet</td><td>C-WF</td><td>Yes</td><td>26.1</td><td>57.1</td></tr><tr><td>SFace2</td><td>C-WF</td><td>Yes</td><td>29.7</td><td>68.5</td></tr><tr><td>CemiFace</td><td>WF4M</td><td>Yes</td><td>53.7</td><td>93.3</td></tr></table>

Table 3. sample-level and Identity-level SO. Rows are colored by the relationship between real and synthetic dataset: generator training data, identity-space match, and both.

Sample-level SO showcases that real pictures are typically more similar to the closest distinct identity image than to a synthetic sample, with an exception of CemiFace scoring 53.7. After analyzing individual closest real and synthetic sample samples for a given real anchor from Web-Face4M (Figure 3), we found that oversampled images 51-55.jpg across CemiFace identities were copies of real identities seen during conditioning. While authors [52] follow an approach of reusing the same sample five times, previously seen in DCFace [30], to shift class center, they diverge as their samples are not generated. This finding validates fidelity of SO in two separate ways: i) identity-level analysis has detected the strong bias towards the underlying real identity amplified by re-use of five copies of real samples; ii) sample-level SO has been able to detect the subtle leakage signal, and since just 5 out of 55 images per synthetic identity are not synthetic, its value exceeds 50 only by a slight margin. Having repeated SO computation for CemiFace, but excluding these five samples for each identity, we obtained SOs of 0.89 (identity-level) and 0.49 (sample-level), proving that identity leakage does not stem only from the reuse of real copies, but is a dataset-wide issue. On the other hand, on an image-by-image basis, synthetic samples almost perfectly follow natural data distribution. Even though many of the synthesized images beyond samples 51-55 (Figure 3) remain more visually similar to the real anchors than to the closest non-mated real samples, this similarity can be considered natural and is a desired property as long as there is no leakage at the identity level.

![](images/f32b759802001d091c3afd79047ad1b8a15b4e1dd3adf181ee34d97ec390059f.jpg)  
Figure 3. Examples of closest WebFace4M non-mated image vs. closest CemiFace sample (including/excluding samples 51-55) with respect to anchor WebFace4M image.

Further, even the most outlying datasets: DCFace (27.5), SFace2 (29.7), and CemiFace (53.7) score significantly lower at the sample level than at the identity level, suggesting that leakage in synthetic datasets manifests predominantly at the identity features level rather than at the level of appearance or style. Consequently, even though the majority of individual images remains visually distinct from real photographs, the underlying identity might not.

In datasets, which generators were trained on multiple real datasets, we observe strongly varying sample-level and identity-level SO values: DCFace (C-WF: 20.4 | 43.4 vs. FFHQ: 5.0 | 12.6), HyperFace (CelebA: 0.2 | 0.0 vs. FFHQ: 10.4 | 15.5 vs. WF4M: 14.7 | 14.2) and Digi2Real (CelebA: 0.4 | 0.1 vs. FFHQ: 6.8 | 10.5 vs. WF4M: 24.3 | 20.4). This asymmetry suggests that a specific dataset or a subset of real datasets can dominate the generator’s learned distribution during training.

## 4.3. Distributional Analysis

Another way to probe identity leakage is to examine highsimilarity outliers. While prior work [3, 48] focused on the single closest image pair, we extend this to a neighborhood analysis by tracking, for each real sample, how similarity difference evolves across its k-th closest real and k-th closest synthetic matches. Figures 4 (sample-level) and 5 (identity-level) show that the gap $\Delta _ { k }$ between realto-real and synthetic-to-real similarity typically narrows as k grows, with tight 95% confidence intervals confirming the stability of this trend. The influence of the training corpus becomes visible further down the ranking: at both levels, FFHQ-trained datasets (HyperFace, IDiffFace, Syn-Face, SynMulti-PIE, Digi2Real) progressively pull ahead of datasets whose training sets they never saw, while C-WFtrained (CemiFace, SFace2, IDNet) maintain their advantage, indicating that training-set exposure influences not only the nearest match but also the clustering strength of synthetic samples surrounding real identities. The leakage signal identified by SO is also reaffirmed at this stage in IDNet and SFace2, which maintain positive $\Delta _ { k }$ across the entire $k \in [ 1 , 1 0 ]$ range, meaning their synthetic samples stay closer to real identities than real identities are to each other, even at the tenth neighbor. The strongest outlier, CemiFace, shows an exceptional bias towards real identities $( \Delta _ { 1 } )$ , with an average gap of over 0.2, revealing systematic overfitting to real people. Conversely, samplelevel $\Delta _ { 1 } \mathrm { s }$ remain low across all dataset pairs, confirming that leakage manifests in identity space rather than in style.

![](images/0dcf8f5b752e77041813faf3d0ceba375be1cd198b556158df61b86778d863b3.jpg)  
Figure 4. Average cosine similarity difference for top-k most sim ilar real-real and real-synthetic image-level comparisons.

![](images/cc973e8e49c2a98ce6dbc62e488786ad9a57221464c35e280f1918e9a2430028.jpg)  
Figure 5. Average cosine similarity difference for top-k most similar real-real and real-synthetic identity-level comparisons.

As the $\Delta _ { k }$ analysis captures local clustering, we complement it with a global view of how synthetic embeddings are distributed across the hypersphere. For each real image, we rank all non-mated similarity scores to define a percentile scale, then place each synthetic sample’s similarity on that scale to obtain a probability density over percentiles (Figure 6). Sample-level synthetic data frequently exhibits high density at the low percentiles, indicating a domain gap in a form of structural mismatch with the real distribution. The observed patterns are strongly dependent on the real reference dataset: (i) alignment with CelebA is the weakest across all synthetic datasets, with mass concentrated in the lowest percentiles; (ii) for FFHQ, a pronounced spike appears near the 0th percentile, but most of the distribution sits around the 0.01 value, with a visible drop-off in the highest percentiles; (iii) synthetic datasets are structurally closest to C-WF and WF4M regardless of generator, with datasets that included C-WF in training (CemiFace, DC-Face, SFace2, IDNet) maintaining a steady density up to the top percentiles, while datasets that did not see it drop off noticeably at the tail. Overall, synthetic datasets fail to cover the high-similarity tail of real distributions unless the reference was part of their training corpus. If this stemmed solely from distributional learning, each generator would preserve tails comparably across all three real references. As tail preservation remains selective to the training corpus, it is a sign of memorization rather than generalization, an interpretation consistent with leakage patterns identified by SO and $\Delta _ { k }$

![](images/c4aef2b044e9b804551b217e1a30c943665331cec6622e9988331e63d07004e6.jpg)  
Figure 6. Synthetic-to-real image similarity distribution with real data non-mated self-similarity as a percentile scale.

![](images/b39e0e317839981521d75d60a4f5d4255c4c0bd13984ef8a61c20e89adc4d120.jpg)  
Figure 7. Number of unique identities among top-10 closest real images with respect to synthetic samples.

Further, to assess how strongly leakage concentrates on any single person, we measured the number of unique real identities appearing in the top-10 closest matches for each synthetic sample. Since FFHQ is unannotated, it was skipped from this analysis. 99.68% (10,538 / 10,572) of C-WF, 82.23% (8,369 / 10,177) of CelebA and 55.53% (114,386 / 205,990) of WF4M identities have at least 10 images, ensuring that a focus on a single identity would be detectable. Our results (Figure 7) show that synthetic images typically draw from multiple real identities, with 6- 8 distinct identities most commonly represented in the top 10. Fewer than 5% of synthetic samples are dominated by a single real identity in most datasets. The notable exceptions are SFace2 (6.6%) and IDNet (5.6%) against C-WF, consistent with elevated SO and $\Delta _ { k }$ values previously reported for these datasets. Even though WebFace4M has a lower fraction of high count identities, it contains approximately 20 times as many identities as CelebA and C-WF, with synthetic images consequently having a much higher number of unique closest real identities. One dominant identity appears at most 0.7% of the time, except for clearly leaking CemiFace, where this ratio is inflated to 4.4%.

## 5. Discussion

As our work presents a novel metric, Scaled Overlap, we need to address challenges related to differentiating leakage from distribution learning, the closed-source dataset generation procedure, and our dataset-level focus.

## 5.1. Leakage vs. Distribution Learning

Identity leakage analysis requires clearer guidance on the acceptable level of similarity between real and synthetic datasets. We propose tying it to our dataset-level metric SO. One should consider two complementary directions, identity-level SO and sample-level SO, to prevent overfitting to neither identity-specific facial features nor stylerelated variables. As such, both sample-level and identitylevel similarities should strive to follow the natural data distribution, with respective SOs converging towards 50, yet they should not exceed it. We argue that this threshold should be slightly lower, because: (i) the imperfect distribution learning is a better choice from a privacy perspective than the leakage; (ii) label noise in the real datasets inflates similarity scores for theoretically non-mated realto-real comparisons, which are in practice mated.

## 5.2. Closed-Source Generator Training

Recent works in the field of synthetic facial dataset generation [3, 33, 52, 60] have shifted heavily towards diffusion models such as Stable Diffusion [46]. These, in turn, rely on large amounts of training data that are often not disclosed. Consequently, one would first need to determine whether a given dataset was used during training [6] before evaluating data leakage using the methodology presented in this paper.

## 5.3. Dataset-level Focus

As SO is designed to measure leakage at the dataset level rather than the individual image level, direct similarity scores comparisons with manual filtering [48] and distributional analysis (as in 4.3) remain the most reliable methods to date for identifying specific outliers that exhibit the highest degree of identity leakage. Nonetheless, our analysis, consistent with previous work [3, 48], found that classifiers are prone to errors when faces are distorted, occluded, or at extreme angles, leading to substantial variability. Consequently, as proposed in this paper, a multi-judge approach should be leveraged to offset model flaws at the individual sample level as well.

## 6. Conclusions

In this work, we demonstrated two limitations of the current approach to identity leakage estimation in synthetic facial recognition datasets: an overreliance on a single model and a tendency to detect leakage in datasets not used in generator training (Phantom Leakage). We addressed both by introducing an evaluation methodology based on an ensemble of five pretrained models and Scaled Overlap, which situates synthetic-to-real similarity against the natural similarity between distinct real identities. Our results show that CemiFace exhibits the strongest dataset-level leakage, with subtler leakage signals observed in SFace2 and IDNet. Leakage itself stemmed primarily from identity-specific features rather than from matching the style of the generator’s training or conditioning set.

Future work should explore two main directions. First, one may leverage distributional data properties presented in this study to design a dataset-wide membership inference attack to determine whether a given real dataset was used for generator training. Second, the leakage with respect to larger training datasets, such as WebFace42M [62], requires further investigation.

## Acknowledgments

• The authors acknowledge the use of AI-based tools in this work to correct typographical errors as well as to rephrase and refine the clarity of the final manuscript.

• This research was supported by the Luxembourg Army.

## References

[1] Xiang An, Xuhan Zhu, Yuan Gao, Yang Xiao, Yongle Zhao, Ziyong Feng, Lan Wu, Bin Qin, Ming Zhang, Debing Zhang, et al. Partial fc: Training 10 million identities on a single machine. In 2021 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), pages 1445–1449. IEEE, 2021. 4

[2] Gwangbin Bae, Martin de La Gorce, Tadas Baltrusaitis,ˇ Charlie Hewitt, Dong Chen, Julien Valentin, Roberto Cipolla, and Jingjing Shen. Digiface-1m: 1 million digital face images for face recognition. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 3526–3535, 2023. 2, 4

[3] Paweł Borsukiewicz, Fadi Boutros, Iyiola E Olatunji, Charles Beumier, Wendkuuni C Ouedraogo, Jacques Klein,ˆ and Tegawende F Bissyand´ e. Beyond real faces: synthetic´ datasets can achieve reliable recognition performance without privacy compromise. npj Artificial Intelligence, 2026. 1, 2, 3, 4, 6, 8

[4] Paweł Borsukiewicz, El-hacen Diallo, Abdoul Aziz Bonkoungou, Wendkuuni C Ouˆ edraogo, Jacques Klein, Charles´ Beumier, and Tegawende F Bissyand´ e. Faces in focus: Eval-´ uating generalizability and transferability of facial recognition explainability techniques for cnn-based models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4006–4015, 2026. 1

[5] Paweł Borsukiewicz, Daniele Lunghi, Wendkuuni Cˆ Ouedraogo, Jacques Klein, and Tegawend ´ e F Bissyand ´ e.´ Benchmarking face recognition without real faces. arXiv preprint arXiv:2607.14932, 2026. 1

[6] Paweł Borsukiewicz, Daniele Lunghi, Wendkuuni C ˆ Ouedraogo, Jacques Klein, and Tegawend ´ e F Bissyand ´ e.´ Have i seen you? embedding behavior signals synthetic face dataset membership. arXiv preprint arXiv:2607.29144, 2026. 8

[7] Fadi Boutros, Jonas Henry Grebe, Arjan Kuijper, and Naser Damer. Idiff-face: Synthetic-based face recognition through fizzy identity-conditioned diffusion model. In Proceedings ofthe IEEE/CVF International Conference on Computer Vi sion, pages 19650–19661, 2023. 2, 3, 4

[8] Fadi Boutros, Vitomir Struc, Julian Fierrez, and Naser Damer. Synthetic data for face recognition: Current state and future prospects. Image and Vision Computing, 135:104688, 2023. 1, 2

[9] Fadi Boutros, Marco Huber, Anh Thi Luu, Patrick Siebke, and Naser Damer. Sface2: Synthetic-based face recognition with w-space identity-driven sampling. IEEE Transactions on Biometrics, Behavior, and Identity Science, 6(3):290–303, 2024. 2, 3, 4

[10] Fadi Boutros, Eduarda Caldeira, Tahar Chettaoui, and Naser Damer. Idperturb: Enhancing variation in synthetic face generation via angular perturbations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 40119–40129, 2026. 1, 2

[11] Nicolas Carlini, Jamie Hayes, Milad Nasr, Matthew Jagielski, Vikash Sehwag, Florian Tramer, Borja Balle, Daphne Ip polito, and Eric Wallace. Extracting training data from diffusion models. In 32nd USENIX security symposium (USENIX Security 23), pages 5253–5270, 2023. 2

[12] Laurent Colbois, Tiago de Freitas Pereira, and Sebastien´ Marcel. On the use of automatically generated synthetic image datasets for benchmarking face recognition. In 2021 IEEE International Joint Conference on Biometrics (IJCB), pages 1–8. IEEE, 2021. 2, 3, 4

[13] Jun Dan, Yang Liu, Haoyu Xie, Jiankang Deng, Haoran Xie, Xuansong Xie, and Baigui Sun. Transface: Calibrating transformer training for face recognition from a data-centric perspective. In Proceedings of the IEEE/CVF international conference on computer vision, pages 20642–20653, 2023. 4

[14] Butterfly Data. Generating synthetic face data for policing using generative ai, 2026. 1

[15] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In Proceedings of the IEEE/CVF con ference on computer vision and pattern recognition, pages 4690–4699, 2019. 3, 4

[16] Yu Deng, Jiaolong Yang, Dong Chen, Fang Wen, and Xin Tong. Disentangled and controllable face image generation via 3d imitative-contrastive learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5154–5163, 2020. 2

[17] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Syl vain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 4

[18] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazare, Maria´ Lomeli, Lucas Hosseini, and Herve J´ egou. The faiss library,´ 2025. 3

[19] European Parliament and Council of the European Union.

Regulation (EU) 2016/679 of the European Parliament and of the Council, 2016. 2

[20] Frontex. Best practice technical guidelines for automated border control (abc) systems, 2015. [Accessed 16-08-2026]. 3

[21] Anjith George and Sebastien Marcel. Digi2real: Bridging´ the realism gap in synthetic data face recognition via foundation models. In Proceedings of the Winter Conference on Applications of Computer Vision, pages 1469–1478, 2025. 2, 3, 4

[22] Anjith George, Christophe Ecabert, Hatef Otroshi Shahreza, Ketan Kotwal, and Sebastien Marcel. Edgeface: Efficient´ face recognition model for edge devices. IEEE Transactions on Biometrics, Behavior, and Identity Science, 6(2):158–168, 2024. 4

[23] Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014. 2

[24] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 2

[25] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2

[26] Gary B Huang, Marwan Mattar, Tamara Berg, and Eric Learned-Miller. Labeled faces in the wild: A database forstudying face recognition in unconstrained environments. In Workshop onfaces in’Real-Life’Images: detection, alignment, and recognition, 2008. 3

[27] InsightFace. Insightface model zoo, 2021. Accessed: 2026- 08-01. 4

[28] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4401–4410, 2019. 2, 3, 4

[29] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8110–8119, 2020. 2

[30] Minchul Kim, Feng Liu, Anil Jain, and Xiaoming Liu. Dcface: Synthetic face generation with dual condition diffusion model. In Proceedings of the ieee/cvf conference on computer vision and pattern recognition, pages 12715–12725, 2023. 2, 3, 4, 6

[31] Minsoo Kim, Min-Cheol Sagong, Gi Pyo Nam, Junghyun Cho, and Ig-Jae Kim. Vigface: Virtual identity generation for privacy-free face recognition. arXiv preprint arXiv:2403.08277, 2024. 1, 3

[32] Jan Niklas Kolf, Tim Rieber, Jurek Elliesen, Fadi Boutros, Arjan Kuijper, and Naser Damer. Identity-driven threeplayer generative adversarial network for synthetic-based face recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 806– 816, 2023. 2, 3, 4

[33] Pavel Korshunov, Ketan Kotwal, Christophe Ecabert, Vidit Amir Mohammadi, and Sebastien Marcel. Investiga-´ tion of accuracy and bias in face recognition trained with synthetic data. In 2025 IEEE International Joint Conference on Biometrics (IJCB), pages 1–10. IEEE, 2025. 8

[34] Adam Kortylewski, Andreas Schneider, Thomas Gerig, Bernhard Egger, Andreas Morel-Forster, and Thomas Vet ter. Training deep face recognition systems with synthetic data. arXiv preprint arXiv:1802.05891, 2018. 2

[35] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems, 25, 2012. 4

[36] Lamyanba Laishram, Muhammad Shaheryar, Jong Taek Lee, and Soon Ki Jung. Toward a privacy-preserving face recognition system: A survey of leakages and solutions. ACM Computing Surveys, 57(6):1–38, 2025. 1

[37] Xiao Lin, Yuge Huang, Jianqing Xu, Yuxi Mi, Shuigeng Zhou, and Shouhong Ding. Uiface: Unleashing inherent model capabilities to enhance intra-class diversity in syn thetic face recognition. In International Conference on Learning Representations, pages 87296–87309, 2025. 2, 3

[38] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings of International Conference on Computer Vision (ICCV), 2015. 2, 4

[39] Henry B Mann and Donald R Whitney. On a test of whether one of two random variables is stochastically larger than the other. The annals of mathematical statistics, pages 50–60, 1947. 3

[40] Casey Meehan, Kamalika Chaudhuri, and Sanjoy Dasgupta. A non-parametric test to detect data-copying in generative models. In International conference on artificial intelligence and statistics, 2020. 3

[41] Terance D Miethe, Tanya Dudinskaya, Christopher Forepaugh, and William H Sousa. Facial recognition technology in policing: A national survey of public support for this technology and privacy/safety concerns. Crime & Delinquency, 71(4):1025–1051, 2025. 1

[42] Hatef Otroshi Shahreza and Sebastien Marcel. Hyperface:´ Generating synthetic face recognition datasets by exploring face embedding hypersphere. In International Conference on Learning Representations, pages 38167–38185, 2025. 4

[43] Foivos Paraperas Papantoniou, Alexandros Lattas, Stylianos Moschoglou, Jiankang Deng, Bernhard Kainz, and Stefanos Zafeiriou. Arc2face: A foundation model for id-consistent human faces. In European Conference on Computer Vision, pages 241–261. Springer, 2024. 1, 2, 3

[44] Lixiong Qin, Mei Wang, Chao Deng, Ke Wang, Xi Chen, Jiani Hu, and Weihong Deng. Swinface: a multi-task transformer for face recognition, expression recognition, age estimation and attribute estimation. IEEE Transactions on Cir cuits and Systems for Video Technology, 34(4):2223–2234, 2023. 4

[45] Haibo Qiu, Baosheng Yu, Dihong Gong, Zhifeng Li, Wei Liu, and Dacheng Tao. Synface: Face recognition with synthetic data. In Proceedings of the IEEE/CVF International

Conference on Computer Vision, pages 10880–10890, 2021. 2, 3, 4

[46] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image¨ synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 8

[47] Florian Schroff, Dmitry Kalenichenko, and James Philbin. Facenet: A unified embedding for face recognition and clustering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 815–823, 2015. 4

[48] Hatef Otroshi Shahreza and Sebastien Marcel. Unveiling´ synthetic faces: How synthetic datasets can expose real identities. In The Third Workshop on New Frontiers in Adversarial Machine Learning, 2024. 1, 2, 3, 4, 6, 8

[49] Hatef Otroshi Shahreza, Christophe Ecabert, Anjith George, Alexander Unnervik, Sebastien Marcel, Nicol´ o\` Di Domenico, Guido Borghi, Davide Maltoni, Fadi Boutros, Julia Vogel, et al. Sdfr: Synthetic data for face recognition competition. In 2024 IEEE 18th international conference on automatic face and gesture recognition (FG), pages 1–9. IEEE, 2024. 1

[50] Jaisidh Singh, Harshil Bhatia, Mayank Vatsa, Richa Singh, and Aparna Bharati. Synthprov: Interpretable framework for profiling identity leakage. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 4746–4756, 2024. 2, 3

[51] Marcus Smith and Seumas Miller. The ethical application of biometric facial recognition technology. Ai & Society, 37(1): 167–175, 2022. 2

[52] Zhonglin Sun, Siyang Song, Ioannis Patras, and Georgios Tzimiropoulos. Cemiface: Center-based semi-hard synthetic face generation for face recognition. Advances in Neural Information Processing Systems, 37:35612–35638, 2024. 1, 2, 3, 4, 6, 8

[53] Timesler. facenet-pytorch, 2018. Accessed: 2026-08-01. 4

[54] Patrick Tinsley, Adam Czajka, and Patrick Flynn. This face does not exist... but it might be yours! identity leakage in generative models. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1320–1328, 2021. 1, 2, 3, 4

[55] Fei Wang, Liren Chen, Cheng Li, Shiyao Huang, Yanjie Chen, Chen Qian, and Chen Change Loy. The devil of face recognition is in the noise. In Proceedings of the European Conference on Computer Vision (ECCV), pages 765–780, 2018. 2

[56] Ryan Webster, Julien Rabin, Loic Simon, and Fred´ eric Jurie.´ Detecting overfitting of deep generative networks via latent recovery. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11265–11274. IEEE, 2019. 2

[57] Erroll Wood, Tadas Baltrusaitis, Charlie Hewitt, Sebastianˇ Dziadzio, Thomas J Cashman, and Jamie Shotton. Fake it till you make it: face analysis in the wild using synthetic data alone. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3681–3691, 2021. 1

[58] Haiyu Wu, Jaskirat Singh, Sicong Tian, Liang Zheng, and Kevin Bowyer. Vec2face: Scaling face dataset generation

with loosely constrained vectors. In International Con ference on Learning Representations, pages 23490–23503, 2025. 1, 2, 3

[59] Jianqing Xu, Shen Li, Jiaying Wu, Miao Xiong, Ailin Deng, Jiazhen Ji, Yuge Huang, Guodong Mu, Wenjie Feng, Shouhong Ding, et al. Id<sup>3</sup>: Identity-preserving-yetdiversified diffusion models for synthetic face recognition. Advances in Neural Information Processing Systems, 37: 77777–77798, 2024. 2, 3

[60] Michael Yeung, Toya Teramoto, Songtao Wu, Tatsuo Fu jiwara, Kenji Suzuki, and Tamaki Kojima. Variface: Fair and diverse synthetic dataset generation for face recognition. arXiv preprint arXiv:2412.06235, 2024. 1, 2, 8

[61] Dong Yi, Zhen Lei, Shengcai Liao, and Stan Z Li. Learn ing face representation from scratch. arXiv preprint arXiv:1411.7923, 2014. 2, 3, 4

[62] Zheng Zhu, Guan Huang, Jiankang Deng, Yun Ye, Junjie Huang, Xinze Chen, Jiagang Zhu, Tian Yang, Dalong Du, Jiwen Lu, et al. Webface260m: A benchmark for million scale deep face recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(2):2627–2644, 2022. 2, 4, 9