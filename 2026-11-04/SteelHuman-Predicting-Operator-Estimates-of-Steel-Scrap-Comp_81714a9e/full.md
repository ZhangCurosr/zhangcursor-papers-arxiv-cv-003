# SteelHuman: Predicting Operator Estimates of Steel-Scrap Composition from Sparse RGB Views

Jovan Kamchev Zoran Ivanovski Faculty of Electrical Engineering and Information Technologies Ss. Cyril and Methodius University Skopje, North Macedonia jovankamchev@gmail.com

## Abstract

Steel scrap composition estimation is an important part of yield planning andfair supplier compensation. Measuring the composition ofa multi-ton pile is an inherently laborious process, so the assessment is often left to trained human operators. Recent works tackle composition-level tasks, but require structured acquisition and multi-stage preprocessing. We study a simpler approach to steel scrap load assessment by aggregating features from sparse, handheld views of the pile. On 600 out-of-fold piles,frozen DINOv2 tokens aggregated uniformly across views reach 13.4% total variation (TV) distance against operator estimates. Most remaining error lies in allocation amongfour thickness-defined heavy melting scrap (HMS) classes, yet image-dependent allocation reduces within-HMS TVfrom 25.1% under a training-set split prior to 11.2%. Due to the lack ofpublic datasets and benchmarksfor predicting composition, external evaluation relies on the public DOES dataset intended for classification. A classifier using the same frozen features reaches 93.97% accuracy, exceeding the published results we compare against. The composition model also aligns with related DOES categories without adaptation.

## 1. Introduction

Increasing the recycling of steel scrap is important for reducing environmental impact and primary resource demand [12]. Electric-furnace steelmaking has increased its share of global crude-steel production from 25.1% in 2015 to 30.3% in 2025 [37, 38]. Scrap composition affects material valuation, downstream process planning, and expected yield, while scrap quality also influences the energy and environmental impact of Electric Arc Furnace (EAF) operation [10, 17]. Despite its importance, composition assessment in industrial practice still commonly relies on trained human operators.

Computer vision has been explored for scrap classification, detection, and segmentation [3, 5, 40, 41], with the broader goal of assisting or automating scrap assessment. Many of these approaches rely on object- or pixel-level annotations that are not produced during routine grading. Public datasets are also limited, and to the best of our knowledge, there is no public benchmark for mass-composition estimation.

LASER [15] tackles composition estimation using human mass estimates, rather than compositions obtained by physically sorting and weighing the load. Related grading systems, such as Tu et al. [31], instead predict an overall grade from segmented scrap proportions and counts.

Research on human perception provides another perspective on this setting. Visual material cues affect expectations about density and weight [2], while estimates of unfamiliar objects are influenced by learned relationships between size, density, and mass [20]. More broadly, work on material perception and intuitive physics suggests that people combine visual observations with prior knowledge about materials and physical behavior [1, 39]. This motivates asking how much of an operator estimate can be recovered from partial visual evidence alone.

We therefore study a simpler acquisition setting, which we call SteelHuman, based on a variable number of sparse RGB views captured with a handheld camera. Our main questions are how much of the operator’s composition estimate can be recovered from sparse views, how the amount of visual evidence affects prediction and where the remaining errors are concentrated. We use frozen DINOv2 ViT-L/14 features with mean aggregation [6, 19] (Fig. 1) and compare learned aggregation, frozen backbones, and inexpensive scores for identifying unreliable predictions.

Because the composition dataset is proprietary, we also use DOES to test transfer to related public categories and supervised classification with the same frozen features.

Our contributions can be summarized as: (1) a low-burden formulation that predicts operator composition estimates from sparse photographs retained during routine grading, improving over non-visual priors; (2) an analysis of where the error lies, how it changes with the number of views, and how well inexpensive uncertainty scores flag unreliable piles, including a view-weighting oracle checked against a structured null; and (3) external checks on the public DOES dataset: transfer to related categories without adaptation, and a supervised classifier above the published results we compare against.

![](images/78aab4f8ab34610d0f30f04daffca0b60e20f6db18cf5a19ccfc6fb7e1f618bb.jpg)  
Figure 1. Predicting a six-class operator estimate from a set of RGB views. The operator also observes unloading and handling of the pile.

## 2. Related Work

## 2.1. Vision-based scrap assessment

Scrap inspection has progressed from hybrid sensing and handcrafted image features [14, 35] to learned classification, detection, and segmentation models [3, 5, 40–42]. Xu et al. [43] address segmentation and classification in scrap scenes. DOES provides public European scrap-category images for classification [24], and its extension CLRiuS [25] studies self-supervised scrap representations on the same dataset. Additional work on DOES evaluates efficient deployment on Jetson hardware by utilizing knowledge distillation [8]. These models predict categories or regions, which cannot be directly interpreted as mass compositions.

Tu et al. [31] combine foreground extraction and scrap segmentation in trailer images from static overhead cameras. Their Bayes-based classifier uses the resulting proportions and counts to predict an overall grade, rather than a mass-composition vector. LASER [15] directly addresses composition estimation and processes truck-unloading video using key-frame selection, scene-change detection, instance segmentation, and mask-aware recognition, then combines local estimates using volume- and density-related information. Storonkin et al. [28] use unloading imagery and MIL to predict inspector estimates of non-metallic contamination at the railcar level. All three use unloading imagery and controlled acquisition at designated sites, while Tu et al. and LASER [15, 31] also rely on multi-stage preprocessing. Our task predicts a six-class composition using a simpler acquisition protocol: sparse handheld photographs from accessible viewpoints, without requiring complete coverage of the pile.

Other studies introduce physical structure or richer supervision. D´ıaz-Romero et al. [4] estimate individual scrapobject mass using depth-derived geometric features. Ji et al. and Yin et al. [13, 44] both use segmentation to predict weight. Quan et al. [22] use prepared steel/copper mixtures with measured weights and segmentation annotations to classify copper weight-percentage ranges. Gursch et al. [9] investigate hyperspectral scrap characterization for composition optimization. These settings illustrate the distinction between controlled environment measurements and operational grading estimates.

## 2.2. Predicting latent quantities and set supervision

Visual mass inference is underconstrained because appearance does not uniquely specify volume or density. Image2mass [27] studies appearance and object dimensions for inferring mass. Visual estimation of food quantity and nutritional content offers another example of inferring latent properties from images [18, 30]. These tasks also require inferring quantities that are only partly visible.

Our variable-size, unordered views require a permutationinvariant set model [45], with Multiple Instance Learning (MIL) providing a natural learned-weight alternative [11]. Although the labels give mass proportion, the supervision is not ordinary Learning from Label Proportions (LLP) [7, 21]: views are repeated observations of one pile, rather than disjoint material instances whose class counts define the target. MIL is therefore an aggregation framework here, while the target is a pile-level mass estimate rather than an instance-count proportion. Multi-view shape recognition likewise combines observations across sparse viewpoints [29, 34].

## 3. Task Formulation and Dataset

## 3.1. Operational process

A truck unloads its contents to form a pile while an operator observes unloading and subsequent handling. During this, the operator takes photographs with a handheld tablet from accessible viewpoints, ultimately recording one composition estimate at one percentage point resolution, after the pile has been handled (Fig. 1). Each pile is graded by a single operator. Images can range from close-ups to overviews and additional views may be retained when handling exposes informative material.

For pile i, let $X _ { i } ^ { \star } \in \Delta ^ { 5 }$ denote its unobserved physical mass composition. The available input and supervision are

$$
E _ { i } = \{ I _ { i v } \} _ { v = 1 } ^ { V _ { i } } , \quad \mathbf { y } _ { i } = X _ { \mathrm { o p } , i } \in \Delta ^ { 5 } , \quad \hat { \mathbf { y } } _ { i } = q _ { \theta } ( E _ { i } ) .\tag{1}
$$

The retained views and the recorded estimate are two imperfect observations of the same pile: views can omit, occlude or repeat material, while the operator sees more than the photographs and brings personal judgment to the estimate. View count is not controlled during acquisition.

## 3.2. Dataset

Our dataset was collected at one plant and contains grades from eight operators. We study the six plant categories that can be found in heterogeneous piles: Stamped steel, Cast iron, and heavy melting scrap (HMS) in four thickness grades (Table 1). Stamped steel and Cast iron are defined by product and material, whereas HMS is defined by thickness, although operators may also weigh thickness when separating thick Stamped steel from > 6 mm HMS. The dataset contains 750 piles: 223 with homogeneous targets and 527 with heterogeneous targets, with a median of 5 views per pile (range 1–19) and a total of 4,187 views. Homogeneous targets contain one class while heterogeneous targets contain two or more. Table 1 gives the class distribution. The three thinner HMS classes occur only in heterogeneous piles, while Stamped steel, > 6 mm HMS, and Cast iron also have homogeneous samples. The dataset is proprietary and will therefore not be released.

## 3.3. Development and split stratification

Five fixed, pile-disjoint folds contain 150 piles each. Stratification across folds uses the dominant target class and active-class count. Folds are not held out by date or operator. $F _ { 0 }$ is the development fold: candidate configurations are trained on the other 600 piles and evaluated on $F _ { 0 }$ . Primary results pool held-out predictions from $F _ { 1 } { - } F _ { 4 }$ after fixing the training configuration. Each held-out-fold model is trained on the other four folds. This is a non-nested evaluation because $F _ { 1 } { - } F _ { 4 }$ also contribute training data to developmentfold comparisons.

Table 1. Dataset class distribution. Active counts include heterogeneous piles. Mean shares include zeros. Active: piles with a nonzero target share.
<table><tr><td>Class</td><td>Active</td><td>Mean share (%)</td><td>Homogeneous</td></tr><tr><td>Stamped</td><td>99</td><td>13.0</td><td>96</td></tr><tr><td>HMS &gt; 6 mm</td><td>596</td><td>49.6</td><td>118</td></tr><tr><td>HMS 4–6 mm</td><td>506</td><td>15.1</td><td>0</td></tr><tr><td>HMS 2–4 mm</td><td>465</td><td>16.6</td><td>0</td></tr><tr><td>HMS 0–2 mm</td><td>251</td><td>3.3</td><td>0</td></tr><tr><td>Cast iron</td><td>365</td><td>2.4</td><td>9</td></tr></table>

## 4. Model

## 4.1. Mean pooling

Our main model uses frozen DINOv2 ViT-L/14 features, extracted with automatic mixed precision. The 3840 × 2160 or 1920 × 1080 images are rotated by their EXIF orientation, converted to RGB, and resized directly to 896 × 504 with bicubic interpolation, without cropping or padding. They are normalized using the DINOv2/ImageNet channel statistics. This gives a 64 × 36 token grid, which 4 × 4 average pooling reduces to 144 spatial tokens. Each token is then projected through a shared linear 1024 → 256 layer, Layer-Norm, GELU, and dropout $p = 0 . 1$ . We average projected tokens within each view and then average view embeddings. With equal token counts per view, this is equivalent to averaging all tokens across views. Finally, the output is a linear six-class head with a softmax activation. The no-projector ablation instead feeds mean-pooled frozen features directly to a six-class linear head.

## 4.2. Learned aggregation

The gated attention-based multiple-instance learning (AB-MIL) [11] averages the projected spatial tokens within a view and learns normalized weights over view embeddings using an attention dimension of 64 and no attention dropout. Our Self-attention [32] variant is inspired by VGGT’s alternating global/frame attention [33]. It applies one self-attention layer across all tokens and one independently within each view, with four heads, feed-forward width 256, and dropout 0.1. No spatial or view positional embeddings are used. The resulting tokens are mean pooled before the same composition head.

## 4.3. Training

Training uses Kullback–Leibler (KL) divergence $D _ { \mathrm { K L } } ( \mathbf { y } \| \hat { \mathbf { y } } )$ AdamW [16] with weight decay 0.05, and a batch size of two piles for 40 epochs. Learning rates warm up over 10% of optimization steps and decay by a cosine schedule to 1% of the peak. The peaks are $8 \times 1 0 ^ { - 4 }$ for Mean pooling and Gated ABMIL and $5 \times 1 0 ^ { - 5 }$ for Self-attention. Training uses full precision, with gradient norms clipped to 1.0. All composition evaluations use the final epoch checkpoint, without early stopping. The main comparison uses initialization seeds 42, 156, and 2026.

Each training epoch samples eight views per pile uniformly without replacement, or uses all views when fewer than eight are available. Mean pooling and Gated AB-MIL use all 144 spatial tokens per sampled view, and Selfattention uniformly samples 30 unique tokens. Each aggregator uses its best tested $F _ { 0 }$ configuration, so the comparison uses separately selected hyperparameters. The ordinal-loss control is described in Supplementary Sec. A.

## 5. Evaluation

## 5.1. Metrics

Our primary metric is total variation distance,

$$
\mathrm { T V } ( \mathbf { y } , \hat { \mathbf { y } } ) = \frac { 1 } { 2 } \sum _ { c = 1 } ^ { 6 } | y _ { c } - \hat { y } _ { c } | .\tag{2}
$$

TV represents the minimum share of the composition vector that must be reassigned to match the operator estimate. A non-visual baseline fits one composition per training split by minimizing mean TV. For each initialization seed, we pool the held-out $F _ { 1 } { - } F _ { 4 }$ predictions and compute each summary over piles. Unless stated otherwise, evaluation uses all retained views and spatial tokens with dropout disabled. Supplementary Table S1 lists the evaluation cohorts.

To compare with LASER [15], let $q _ { i c } = y _ { i c } / \sum _ { d = 1 } ^ { 4 }$ <sub>1</sub> y<sub>id</sub> and $\hat { q } _ { i c } = \hat { y } _ { i c } / \sum _ { d = 1 } ^ { 4 } \hat { y } _ { i d }$ for the four HMS classes, indexed here by $c = 1 , \ldots , 4 .$ . The pooled RMSE is

$$
\mathrm { R M S E } _ { \mathrm { H M S } } = \sqrt { \frac { 1 } { 4 N } \sum _ { i = 1 } ^ { N } \sum _ { c = 1 } ^ { 4 } ( \hat { q } _ { i c } - q _ { i c } ) ^ { 2 } } .\tag{3}
$$

This comparison uses the 405 heterogeneous piles with at least 95% target HMS. The threshold retains mostly HMS piles with small Cast iron shares (1–5%) that an HMS-only restriction would exclude.

## 5.2. Error structure and additional diagnostics

Coarse and ordinal error. Collapsing the four HMS classes gives Stamped steel/HMS/Cast iron compositions. We report coarse TV and the nonnegative residual $\mathrm { T V - T V _ { c o a r s e } } .$ , termed within-HMS redistribution TV. For target HMS mass $m _ { i } \geq 0 . 1 0$ , we independently normalize HMS targets and predictions and compute the 1-Wasserstein distance over ordered thickness classes

$$
W _ { 1 , i } = \sum _ { j = 1 } ^ { 3 } \left| \sum _ { c = 1 } ^ { j } ( q _ { i c } - \hat { q } _ { i c } ) \right| .\tag{4}
$$

Adjacent thickness classes have unit distance: $W _ { 1 }$ is in class steps, not millimetres. Its three parts are cumulative discrepancies across the 6, 4, and 2 mm boundaries. Table 4 reports unweighted means of conditional TV and $W _ { 1 }$ over eligible piles. The aggregate displacement per unit of redistribution is

$$
\bar { d } _ { \mathrm { a g g } } = \frac { \sum _ { i } W _ { 1 , i } } { \sum _ { i } \mathrm { T V } ( q _ { i } , \hat { q } _ { i } ) } .\tag{5}
$$

Figure 2b shows these cumulative boundary discrepancies on the view-count cohort.

Within-HMS prior control. For each held-out fold, we average the normalized four-class HMS targets of training piles with positive HMS mass. We keep each model prediction’s Stamped steel share, Cast iron share, and total HMS share, but divide its HMS share using this fixed training prior. This tests whether image-dependent HMS allocations improve on a constant split while keeping the coarse prediction unchanged.

View subset evaluation. For the seed-42 Mean pooling and Self-attention models, we evaluate the same nested sub sets with $k \in \{ 1 , 2 , 4 , \mathrm { a l l } \}$ over 30 fixed repetitions on the same 356 heterogeneous $F _ { 1 } { - } F _ { 4 }$ piles having at least four views. We first average subset errors across repetitions within each pile.

Target-conditioned view oracle. On these same $F _ { 1 } -$ $F _ { 4 }$ subsets, the target-conditioned oracle optimizes view weights separately for each subset while keeping the Mean pooling model frozen. Softmax weights combine projected view embeddings, $\begin{array} { r } { h = \sum _ { v } a _ { v } h _ { v } , } \end{array}$ , before the composition head. A separate $F _ { 0 }$ diagnostic uses the 102 of 107 heterogeneous piles for which other targets satisfying the matching constraints exist, the frozen seed-42 Mean pooling model, and 30 matched nested subsets. If fewer than k views exist in this diagnostic, all available views are used. Optimization details are in Supplementary Sec. B.

Five target shuffles among eligible heterogeneous $F _ { 0 }$ piles preserve dominant class and active-class count, constrain dominant-share differences to at most 10 percentage points, and exclude the original target sample. Exact active classes are not always preserved. Original and shuffled targets share the same pile and candidate subsets.

Table 2. Primary out-of-fold TV (%). For trained models, each metric is computed across piles within each seed and reported as $\mathrm { m e a n } \pm \mathrm { S D }$ across three seeds. The prior has no training-seed variation. Bold marks the lowest observed mean.
<table><tr><td></td><td colspan="2">All piles  $( N = 6 0 0 )$ </td><td colspan="2">Homogeneous  $( N = 1 8 0 )$ </td><td colspan="2">Heterogeneous  $( N = 4 2 0 )$ </td></tr><tr><td>Method</td><td>Mean ↓</td><td>Median ↓</td><td>Mean ↓</td><td>Median ↓</td><td>Mean ↓</td><td>Median ↓</td></tr><tr><td>TV-optimal prior</td><td>40.44</td><td>36.00</td><td>67.76</td><td>39.00</td><td>28.73</td><td>26.00</td></tr><tr><td>Mean pooling</td><td> ${ \bf 1 3 . 4 4 \pm 0 . 0 3 }$ </td><td> ${ \bf 1 0 . 2 0 \pm 0 . 1 4 }$ </td><td> $5 . 0 5 \pm 0 . 0 4$ </td><td> $0 . 2 0 \pm 0 . 0 3$ </td><td> $\mathbf { 1 7 . 0 3 \pm 0 . 0 3 }$ </td><td> ${ \bf 1 5 . 0 4 \pm 0 . 2 2 }$ </td></tr><tr><td>Self-attention</td><td> $1 3 . 7 6 \pm 0 . 3 5$ </td><td> $1 0 . 4 0 \pm 0 . 2 4$ </td><td> ${ \bf 4 . 6 0 \pm 0 . 2 4 }$ </td><td> $\mathbf { 0 . 0 3 \pm 0 . 0 0 }$ </td><td> $1 7 . 6 9 \pm 0 . 4 3$ </td><td> $1 5 . 1 5 \pm 0 . 1 0$ </td></tr><tr><td>Gated ABMIL</td><td> $1 5 . 4 3 \pm 0 . 1 1$ </td><td> $1 1 . 8 4 \pm 0 . 2 8$ </td><td> $6 . 0 7 \pm 0 . 1 0$ </td><td> $0 . 1 2 \pm 0 . 0 1$ </td><td> $1 9 . 4 5 \pm 0 . 1 1$ </td><td> $1 6 . 3 0 \pm 0 . 1 8$ </td></tr><tr><td>Mean pooling, no projector</td><td> $1 4 . 0 3 \pm 0 . 0 2$ </td><td> $1 1 . 1 6 \pm 0 . 2 1$ </td><td> $6 . 8 7 \pm 0 . 1 0$ </td><td> $0 . 5 3 \pm 0 . 0 2$ </td><td> $1 7 . 1 0 \pm 0 . 0 6$ </td><td> $1 5 . 0 5 \pm 0 . 1 1$ </td></tr></table>

![](images/af8f81537fc126a96920bbc580e06592b3f58e9e0397ceef8da1924f1784e936.jpg)  
(a)

![](images/f48bec476f44c728d45e68584240826ee8ee2e1f262edc50843009919039a0d9.jpg)  
(b)

![](images/2d170882268deca5e60fd9a51f353d8f4defe9b29cfb686a6e0637494b18148d.jpg)  
(c)  
Figure 2. Seed-42 models on 356 heterogeneous $F _ { 1 } { - } F _ { 4 }$ piles with at least four views. (a) All-view coarse and within-HMS TV. (b) Mean absolute differences in cumulative HMS-normalized shares at the 6, 4, and 2 mm boundaries, in percentage points. (c) TV with controlled view counts, averaging 30 nested subsets per pile for each fixed count. Oracle weights use the recorded target and are optimized separately for each subset.

## 5.3. Selective prediction and external evaluation

Risk–coverage. On the 403 heterogeneous $F _ { 1 } { - } F _ { 4 }$ piles with at least two views, we compare mean leave-one-viewout (LOVO) shift $\begin{array} { r } { S = V ^ { - 1 } \sum _ { v } \mathrm { T V } ( \hat { \mathbf { y } } ( E ) , \hat { \mathbf { y } } ( E \setminus \{ I _ { v } \} ) ) } \end{array}$ , normalized shift $( V \mathrm { ~ - ~ } 1 ) S ,$ full-view prediction entropy $\begin{array} { r } { - \sum _ { c } \hat { y } _ { c } \log \hat { y } _ { c } , } \end{array}$ and view count. LOVO measures how much the prediction changes when each view is removed. Lower scores are retained first, except that larger view counts are retained first. Risk is the mean TV of the retained piles (Fig. 4).

DOES protocols. We distinguish two tasks. Semantic transfer evaluates our pile-trained models without adaptation on four related DOES categories [24]: E1 maps to 0–6 mm HMS, E3 and EHRB (rebar) to > 6 mm HMS and E8 to thin new-production scrap matching our Stamped steel. The composition models are trained only on our proprietary pile data. Their outputs on the full available DOES collection for these categories are averaged across $F _ { 0 } { - } F _ { 4 }$ models. This experiment uses the “raw” set of two overlapping large crops. We upsample each crop to $5 0 4 \times 5 0 4$ and extract DINOv2

ViT-L tokens. Local $4 \times 4$ averaging gives 81 tokens per crop, which are projected before averaging within and across crops. We average model outputs after softmax.

Supervised DOES classification uses the official 105,523 training and 8,131 temporally separate test tiles. Test targets cover eight scrap classes and the ninth output, background, has no test examples but remains eligible during prediction. We use all 256 spatial DINOv2-L tokens from $2 2 4 \times 2 2 4$ tiles, without local token pooling. Tokens are projected when a projector is used, mean pooled, and passed to a linea classification head. We compare a new linear head directly on the frozen features, a head on the frozen pile-trained projector, and joint head/projector training from either piletrained or random projector initialization. The pretrained projector is taken from the Mean pooling model trained with $F _ { 0 }$ held out. All four use initialization seed 2026, 20 fixed epochs, cross-entropy loss, AdamW, learning rate $3 \times 1 0 ^ { - 4 }$ weight decay 0.01, batch size 128, two warmup epochs and cosine decay, without augmentation.

Centered token contributions. For projected token $\mathbf { z } _ { t }$ and class-head weights $\mathbf { w } _ { c } ,$ we display $( \mathbf { w } _ { c } - \bar { \mathbf { w } } ) ^ { \top } \mathbf { z } _ { t } .$ with $\begin{array} { r } { \bar { \bf w } = \frac { 1 } { 6 } \sum _ { c } { \bf w } _ { c } } \end{array}$ . Averaging these contributions and adding the centered head bias recovers the centered pile logits, with unchanged softmax probabilities. Supplementary Fig. S3 shows example maps.

Table 3. Backbone comparison on the development fold $( N = 1 5 0 )$ TV (%) from one run per backbone using Mean pooling.
<table><tr><td>Representation</td><td>All</td><td>Hom.</td><td>Hetero.</td></tr><tr><td>DINOv2 ViT-S [19]</td><td>14.72</td><td>10.56</td><td>16.39</td></tr><tr><td>DINOv2 ViT-L [19]</td><td>12.83</td><td>5.15</td><td>15.92</td></tr><tr><td>DINOv2 ViT-g [19]</td><td>12.84</td><td>5.67</td><td>15.72</td></tr><tr><td>DINOv3 ViT-L [26]</td><td>13.30</td><td>6.04</td><td>16.22</td></tr><tr><td>ConvNeXt-V2-L [36]</td><td>15.61</td><td>9.31</td><td>18.14</td></tr><tr><td>CLIP ViT-L [23]</td><td>15.87</td><td>11.62</td><td>17.58</td></tr></table>

## 6. Results

## 6.1. Aggregation and backbones

Table 2 reports the three-seed comparison. Mean pooling has the lowest overall and heterogeneous TV. Self-attention is better on homogeneous piles. Paired comparisons are given in Supplementary Table S2. The no-projector ablation increases overall TV by 0.59 percentage points. The small heterogeneous-pile change suggests that frozen features already provide much of the information used by the trained projector. Median TV shows that predictions for most homogeneous targets are nearly exact, with a few severe failures increasing the mean TV (Supplementary Fig. S2).

To check the Self-attention train–test token-count difference, we also evaluate 30 randomly sampled tokens per view, keeping all views and averaging predictions over 20 draws: seed-42 TV is 13.80%, versus 13.69% with all 144 tokens. Matching the training token count therefore does not improve this comparison (see Supplementary Sec. A for individual-draw results).

Table 3 reports the $F _ { 0 }$ backbone comparison. Final-layer DINOv2 ViT-L and $\mathrm { V i T - g }$ have nearly identical TV and the lowest overall error in this comparison. Backbone extraction details are in Supplementary Sec. E. Intermediate layers and learned layer fusion do not improve overall TV over the final layer (Supplementary Table S3).

## 6.2. HMS error

On all 600 primary piles, seed-42 coarse TV is 2.24%, and 83.3% of total TV remains within HMS. Figure 2 shows the same decomposition and cumulative boundary discrepancies on the separate 356-pile view-count cohort. There, the largest discrepancy is at 6 mm for Mean pooling and Self-attention, and at 4 mm for the oracle.

Visual HMS allocation. Replacing the predicted HMS split with the training prior substantially increases within-

Table 4. Within-HMS prior control, seed 42, $F _ { 1 } { - } F _ { 4 }$ . The prior retains the model’s coarse prediction. Redistribution uses all 600 piles. Conditional metrics use the 515 with at least 10% target HMS.
<table><tr><td>Metric ↓</td><td>Model</td><td>HMS prior</td></tr><tr><td>Within-HMS TV (%)</td><td>11.19</td><td>25.12</td></tr><tr><td>Conditional HMS TV (%)</td><td>14.13</td><td>30.74</td></tr><tr><td>Conditional  $W _ { 1 }$  (class steps)</td><td>0.227</td><td>0.534</td></tr></table>

HMS TV from 11.19% to 25.12% on the 600 primary piles (Table 4). The model therefore predicts pile-specific HMS allocations beyond the average training composition. Among the 515 piles with at least 10% target HMS, aggregate displacement is 1.61 class steps for the model and 1.74 for the prior, so errors extend beyond adjacent classes even though the model substantially improves allocation. These comparisons concern operator estimates, not measured thickness. Supplementary Sec. A checks alternative priors and piles with at least two active HMS classes. Adding an ordinal $W _ { 1 }$ penalty with a weight of 0.1 left performance essentially unchanged in the matched seed-42 experiment.

On the restricted 405-pile cohort, our Mean pooling model reaches an RMSE of 11.8%, close to the 11.9% LASER reports on its own data. Because of the different dataset and class taxonomy, this can only be considered a scale reference rather than a direct comparison.

## 6.3. View count and target-conditioned gains

Additional views reduce Mean pooling TV from 23.5% with one view to 16.6% with all views (Fig. 2c). The all-view endpoint uses a variable number of views per pile, so its gain is not a fixed per-view effect. On this cohort, target-conditioned oracle weighting reduces all-view TV from 16.6% to 5.7% (Fig. 2c). The separate $F _ { 0 }$ control tests whether such gains are specific to the recorded target.

On the 102-pile $F _ { 0 }$ control, recorded and shuffled targets yield similar all-view gains (9.3 and 9.7 percentage points). Recorded targets have lower TV both before reweighting (14.7% versus 17.1%) and after it (5.3% versus 7.4%). Thus several structurally similar targets can benefit from reweighting the same views (Supplementary Fig. S1).

## 6.4. External evaluation on DOES

Without DOES training (Fig. 3), E1 emphasizes thinner HMS, EHRB favors > 6 mm HMS, and E8 favors Stamped steel. E3 is less clearly separated: 42% goes to > 6 mm HMS and 33% to 4–6 mm HMS, with 55% across the three thinner HMS classes including 4–6 mm.

Training the pile-initialized projector and a new head reaches 93.97%, 1.17 percentage points above the highest published accuracy listed in Table 5. Random projector initialization gives 93.91%, so pile initialization adds little. Frozen DINO features also perform better than the pilepretrained frozen projector.

![](images/67fbbdbf18ad81c80e7f93dfa9517d101e870a845c3b7bfc1423233d6ca4685d.jpg)  
Figure 3. External transfer without DOES training. Cells show mean composition outputs by DOES category across the $F _ { 0 } { - } F _ { 4 }$ models. N is the number of source-image crop pairs. Bold entries mark predefined category correspondences.

Table 5. Classification on the official DOES test tiles (N = 8,131).
<table><tr><td>DOES-trained method</td><td>Accuracy (%)</td></tr><tr><td>Original DOES, PreActResNet18 [24]</td><td>68.09</td></tr><tr><td>CLRiuS, 10 labeled tiles/class [25]</td><td>75.89</td></tr><tr><td>LASER, pretrained CLIP [15]</td><td>92.80</td></tr><tr><td>Esfandiari et al. [8], ViT-L teacher</td><td>92.37</td></tr><tr><td>Esfandiari et al. [8], distilled DeiT</td><td>88.81</td></tr><tr><td>Ours: frozen DINOv2-L, linear head</td><td>91.53</td></tr><tr><td>Ours: frozen pile projector</td><td>85.71</td></tr><tr><td>Ours: train random projector</td><td>93.91</td></tr><tr><td>Ours: train pile-initialized projector</td><td>93.97</td></tr></table>

## 6.5. Selective prediction and deployment

Prediction entropy performs similarly to LOVO without repeated prediction: at 50% coverage, retained TV is about 15.0% for both, versus 16.7% under random retention (Fig. 4). This offers a single forward pass for triage, although even the retained half has substantial error. Supplementary Fig. S5 shows the pooled analysis including homogeneous piles. Ranking by view count is worse than random, which could point to operators keeping more views for harder piles.

We profile the Mean pooling $F _ { 0 }$ model on a GTX 1080 Ti using FP16 feature extraction and FP32 aggregation. The benchmark uses 50 piles covering the whole view-count range and a batch size of one. Mean latency is 2.06 s per pile, or 382 ms per view, with DINO feature extraction taking 99.7% of the time. These timings exclude loading, warmup, disk preload, networking, queues, and serialization, so tabletto-server overhead is not included.

![](images/f2c927100856e61736eaea42cbff38d5691a5bd896af2d646388d8b13e09bec9.jpg)  
Figure 4. Risk–coverage for seed-42 Mean pooling on 403 heterogeneous $F _ { 1 } { - } F _ { 4 }$ piles with at least two views. Risk is the retained mean full-view TV, and the dashed line shows expected risk under random retention. Shading denotes pointwise 95% pile-bootstrap intervals for LOVO mean and predictive entropy.

## 7. Discussion

Simple acquisition through sparse handheld views shows promising performance when trained on human composition estimates. The main difficulty is allocating mass among HMS thickness classes. The prior controls show that the model learns pile-specific allocations even for the harder HMS cases, but do not establish which cues it uses: appearance properties such as thickness, texture, and shape could all correlate with operator estimates. The lack of homogeneous targets for the three thinner HMS classes also confounds the comparison. A possible explanation for the lack of homogeneous thinner samples is that Cast iron, Stamped steel and thick metal pieces such as beams often come from one place and are easier to sort due to their material definition, while sorting thinner steel is laborious, which is one of the reasons why this process still relies on human estimates. This could contribute to the worse performance on thinner classes, whose boundaries are also harder to judge visually.

Low mean coarse error does not eliminate severe category mistakes. Homogeneous targets account for about 48% of the total coarse error in the seed-42 evaluation of 600 primary piles. The five homogeneous piles with TV above 90% account for about 29%. Some piles graded as > 6 mm HMS visually resemble Stamped steel and are predicted as such (Supplementary Fig. S2). Thickness and the operator’s taxonomy may affect this distinction. Homogeneous HMS targets can also incur within-HMS errors when predicted as thinner classes.

The retained views do not capture all the evidence used by the operator: views repeat or omit material, and operator estimates may vary. The structured null preserves dominant class and active-class count while keeping dominant shares close, so similar reweighting gains for these nearby targets are plausible. The oracle gap is a diagnostic of the representation, not an estimate of what a learned aggregator could recover. Without independent grades or measured fractions, we cannot separate model error from target ambiguity.

The no-projector ablation and DOES classification both point to a strong contribution from frozen features. The tested learned aggregators do not improve overall TV over mean pooling, despite modeling differences or interactions among views. Richer aggregation may help, with more training data, exhaustive evidence and measured targets making those relationships learnable.

One of the advantages of mean pooling regarding interpretability is the token contribution decomposition (Supplementary Fig. S3). These maps show regional contributions to the model’s logits and could help operators inspect a prediction. Prediction entropy can help rank estimates for review (Fig. 4), but may miss severe homogeneous errors when the model is confidently wrong. A deployment advantage of mean pooling is caching the pooled embeddings and allowing the predictions to update as photographs arrive. Practical usefulness still requires deploying the system and recording how these outputs affect grading.

Further progress may require better evidence as well as better models. Physical inspection can provide thickness information beyond appearance (Supplementary Fig. S4). Scale references or depth could also help, but sparse, insufficiently overlapping or changing pile views need not support reliable geometric reconstruction. Even surface reconstruction leaves buried contents unseen. A synthetic-pile dataset built from scans of individual scrap pieces could offer controlled compositions and viewpoints, provided the pieces’ masses are measured. Repeated independent grades or measured subsets would also help determine which disagreements can reasonably be resolved from photographs.

## 8. Conclusion

We asked how much of an operator’s estimate can be recovered from the handheld photographs retained during routine grading. Frozen DINOv2 features with mean pooling recover a substantial part of it, and none of the learned aggregators we tested improved its overall TV. Coarse composition error is low, with most remaining disagreement concerning the split among HMS thickness classes, where the model clearly improves on a constant training-set split. These results motivate evaluating sparse handheld views for operator assistance. Progress towards more autonomous grading will require more data, better reference targets, and evaluation under deployment conditions.

Code availability. Code for the DOES experiments will be released at https://github.com/jovankamchev/ steelhuman. The industrial dataset is proprietary and cannot be released.

## Acknowledgments

We thank Makstil AD Skopje for scholarship funding and for providing the industrial data used in this study.

Large language models were used for coding (models, result compilation) and language editing. The authors take responsibility for the research, analyses, and manuscript.

## References

[1] Peter W. Battaglia, Jessica B. Hamrick, and Joshua B. Tenenbaum. Simulation as an engine of physical scene understanding. Proceedings ofthe National Academy ofSciences, 110 (45):18327–18332, 2013.

[2] Gavin Buckingham, Jonathan S. Cant, and Melvyn A. Goodale. Living in a material world: how visual cues to material properties affect the way that we lift objects and perceive their weight. Journal of Neurophysiology, 102(6): 3111–3118, 2009.

[3] Ichiro Daigo, Ken Murakami, Keijiro Tajima, and Rei Kawakami. Thickness classifier on steel in heavy melting scrap by deep-learning-based image analysis. ISIJ International, 63(1):197–203, 2023.

[4] Dillam Jossue D´ıaz-Romero, Simon Van den Eynde, Wouter Sterkens, Bart Engelen, Isiah Zaplana, Wim Dewulf, Toon Goedeme, and Jef Peeters. Simultaneous mass estimation´ and class classification of scrap metals using deep learning. Resources, Conservation and Recycling, 181:106272, 2022.

[5] Paulo Henrique dos Santos, Valeria de Carvalho Santos, and´ Eduardo Jose da Silva Luz. Towards robust ferrous scrap´ material classification with deep learning and conformal prediction. Engineering Applications ofArtificial Intelligence, 140:109724, 2025.

[6] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16×16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021.

[7] Gabriel Dulac-Arnold, Neil Zeghidour, Marco Cuturi, Lucas Beyer, and Jean-Philippe Vert. Deep multi-class learning from label proportions. arXiv preprint arXiv:1905.12909, 2019.

[8] Yasin Esfandiari, Karan Rajshekar, Sabine Janzen, and Wolfgang Maass. Efficient vision models for Jetson: Steel classification via knowledge distillation. In Workshop on Sustainability and Resource-Efficiency of Artificial Intelligence, 2026.

[9] Heimo Gursch, Andreas Ofner, Robert Harb, Malte Jaschik, Harald Ganster, and Johannes Rieger. Hyperspectral scrap characterisation for scrap composition optimisation in steel recycling. Waste Management & Research, 2026. OnlineFirst.

[10] Melanie Haupt, Carl Vadenbo, Christoph Zeltner, and Stefanie Hellweg. Influence of input-scrap quality on the environmental impact of secondary steel production. Journal ofIndustrial Ecology, 21(2):391–401, 2017.

[11] Maximilian Ilse, Jakub M. Tomczak, and Max Welling. Attention-based deep multiple instance learning. In Proceedings of the 35th International Conference on Machine Learning, pages 2127–2136, 2018.

[12] International Energy Agency. Iron and Steel Technology Roadmap: Towards more sustainable steelmaking. OECD Publishing, Paris, 2020.

[13] Mengyang Ji, Kaixuan Shi, Yu Guo, and Bosheng Yang. Scrap steel intelligent determination algorithm based on improved DeepLabv3+ convolutional neural network. Steel Rolling, 42 (5):150–158, 2025.

[14] Matti Kutila, Jouko Viitanen, and Antero Vattulainen. Scrap metal sorting with colour vision and inductive sensor array. In Proceedings ofthe 2005 International Conference on Computational Intelligencefor Modelling, Control and Automation, pages 725–729, 2005.

[15] Jiarui Lei, Sheng Huang, Zheqiang Shen, Lan Wu, and Dong Liu. LASER: LAyered ScenE Recognition for visual intelligent recycling of steel scrap. IEEE Transactions on Industrial Informatics, 21(4):2927–2936, 2025.

[16] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019.

[17] Ismael Matino, Valentina Colla, and Stefano Baragiola. Quantification of energy and environmental impacts in uncommon electric steelmaking scenarios to improve process sustainability. Applied Energy, 207:543–552, 2017.

[18] Austin Meyers, Nick Johnston, Vivek Rathod, Anoop Korattikara, Alex Gorban, Nathan Silberman, Sergio Guadarrama, George Papandreou, Jonathan Huang, and Kevin P. Murphy. Im2Calories: Towards an automated mobile vision food diary. In Proceedings ofthe IEEE International Conference on Computer Vision, pages 1233–1241, 2015.

[19] Maxime Oquab, Timothee Darcet, Th´ eo Moutakanni, Huy V.´ Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve´ Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and´ Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024.

[20] Megan A. K. Peters, Jonathan Balzer, and Ladan Shams. Smaller = denser, and the brain knows it: Natural statistics of object density shape weight expectations. PLOS ONE, 10(3): e0119794, 2015.

[21] Novi Quadrianto, Alex J. Smola, Tiberio S. Caetano, and´ Quoc V. Le. Estimating labels from label proportions. Journal ofMachine Learning Research, 10(82):2349–2374, 2009.

[22] Yijun Quan, Sanjay Singhal, Zushu Li, and Giovanni Montana. Image-segmentation-guided fragmentized steel scrap tramp material characterization. Journal ofSustainable Metallurgy, 12(4):4399–4408, 2026.

[23] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings ofthe 38th International Conference on Machine Learning, pages 8748–8763. PMLR, 2021.

[24] Michael Schafer, Ulrike Faltings, and Bj ¨ orn Glaser. DOES - a¨ multimodal dataset for supervised and unsupervised analysis of steel scrap. Scientific Data, 10(1):780, 2023.

[25] Michael Schafer, Ulrike Faltings, and Bj ¨ orn Glaser. CLRiuS:¨ Contrastive learning for intrinsically unordered steel scrap. Machine Learning with Applications, 17:100573, 2024.

[26] Oriane Simeoni, Huy V. Vo, Maximilian Seitzer, Federico´ Baldassarre, Maxime Oquab, et al. DINOv3. arXiv preprint arXiv:2508.10104, 2025.

[27] Trevor Standley, Ozan Sener, Dawn Chen, and Silvio Savarese. image2mass: Estimating the mass of an object from its image. In Proceedings of the 1st Annual Conference on Robot Learning, pages 324–333. PMLR, 2017.

[28] Daniil Storonkin, Ilia Dziub, Maksim Golyadkin, and Ilya Makarov. From images to decisions: Assistive computer vision for non-metallic content estimation in scrap metal. arXiv preprint arXiv:2602.07062, 2026.

[29] Hang Su, Subhransu Maji, Evangelos Kalogerakis, and Erik Learned-Miller. Multi-view convolutional neural networks for 3D shape recognition. In Proceedings of the IEEE International Conference on Computer Vision, pages 945–953, 2015.

[30] Quin Thames, Arjun Karpur, Wade Norris, Fangting Xia, Liviu Panait, Tobias Weyand, and Jack Sim. Nutrition5k: Towards automatic nutritional understanding of generic food. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8903–8911, 2021.

[31] Qifan Tu, Dawei Li, Qian Xie, Li Dai, and Jun Wang. Automated scrap steel grading via a hierarchical learning-based framework. IEEE Transactions on Instrumentation and Measurement, 71:1–13, 2022.

[32] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, pages 5998–6008, 2017.

[33] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. VGGT: Visual geometry grounded transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5294–5306, 2025.

[34] Xin Wei, Yifei Gong, Fudong Wang, Xing Sun, and Jian Sun. Learning canonical view representation for 3D shape recognition with arbitrary views. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 407– 416, 2021.

[35] Tadeusz Wieczorek and Marcin Pilarczyk. Classification of steel scrap in the EAF process using image analysis methods. Archives ofMetallurgy and Materials, 53(2):613–617, 2008.

[36] Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon, and Saining Xie. ConvNeXt V2: Co-designing and scaling ConvNets with masked autoencoders. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16133– 16142, 2023.

[37] World Steel Association. World steel in figures 2016. Technical report, World Steel Association, Brussels, Belgium, 2016.

[38] World Steel Association. World steel in figures 2026. Technical report, World Steel Association, Brussels, Belgium, 2026.

[39] Bei Xiao and Chenxi Liao. Material perception connects vision, cognition and action. Nature Reviews Psychology, 4: 687–701, 2025.

[40] Pengcheng Xiao, Chao Wang, Liguang Zhu, Wenguang Xu, Yuxin Jin, and Rong Zhu. An efficient and accurate quality inspection model for steel scraps based on dense small-target detection. Processes, 12(8):1700, 2024.

[41] Wenguang Xu, Pengcheng Xiao, Liguang Zhu, Yan Zhang, Jinbao Chang, Rong Zhu, and Yunfeng Xu. Classification and rating of steel scrap using deep learning. Engineering Applications ofArtificial Intelligence, 123:106241, 2023.

[42] Wenguang Xu, Pengcheng Xiao, Liguang Zhu, Guangsheng Wei, and Rong Zhu. An efficient treatment method of scrap intelligent rating based on machine vision. Applied Intelligence, 54(21):10912–10928, 2024.

[43] Yunfeng Xu, Changda Liu, Jiakui Zhong, Wei Mei, and Yan Zhang. MSPFNet: Multi-scale perceptual focusing network for scrap steel segmentation and classification. Engineering Applications ofArtificial Intelligence, 144:110084, 2025.

[44] Jihu Yin, Pengcheng Xiao, Bixia Zhang, and Liguang Zhu. Scrap weight prediction for different scrap types based on semantic segmentation and machine learning. Machine Vision and Applications, 37(5):104, 2026.

[45] Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabas P ´ oczos, Ruslan Salakhutdinov, and Alexander J.´ Smola. Deep sets. In Advances in Neural Information Processing Systems, pages 3391–3401, 2017.

# Supplementary Material: SteelHuman: Predicting Operator Estimates of Steel-Scrap Composition from Sparse RGB Views

Jovan Kamchev Zoran Ivanovski Faculty of Electrical Engineering and Information Technologies Ss. Cyril and Methodius University Skopje, North Macedonia jovankamchev@gmail.com

## A. Evaluation Details and Additional Controls

Table S1 lists the subsets used in the main paper. Homogeneous targets have one active class; heterogeneous targets have two or more. Primary comparisons pool held-out predictions from $F _ { 1 } { - } F _ { 4 }$ . The development-fold oracle control uses a separate cohort.

Table S1. Evaluation cohorts. Unless marked $F _ { 0 } ,$ , all use $F _ { 1 } { - } F _ { 4 }$ . Main-paper Table 2 uses three seeds; diagnostics use seed 42.
<table><tr><td>Analysis / inclusion rule</td><td>Piles</td></tr><tr><td>Primary evaluation</td><td>600</td></tr><tr><td>Homogeneous</td><td>180</td></tr><tr><td>Heterogeneous</td><td>420</td></tr><tr><td>Conditional HMS: target  $\mathrm { H M S } \geq 1 0 \%$ </td><td>515</td></tr><tr><td>Four-HMS RMSE: heterogeneous,  $\mathrm { H M S } \geq 9 5 \%$ </td><td>405</td></tr><tr><td>View subsets: heterogeneous,  $V \geq 4$ </td><td>356</td></tr><tr><td>Main risk-coverage: heterogeneous,  $V \geq 2$ </td><td>403</td></tr><tr><td>Pooled risk-coverage: all,  $V \geq 2$ </td><td>567</td></tr><tr><td>Structured null: eligible heterogeneous  $F _ { 0 }$  piles</td><td>102</td></tr><tr><td>Latency: view-count-stratified  $F _ { 0 }$  sample</td><td>50</td></tr></table>

Paired aggregation comparisons. For each pile, we average the errors of its three initialization seeds, then compute the difference between methods. We resample piles within held-out folds 10,000 times, retaining the paired predictions (Table S2). These intervals describe pile uncertainty for the fitted models, rather than variation from retraining on new data.

Self-attention token-count control. We keep the seed-42 checkpoints and all available views fixed, disable dropout, and sample 30 tokens per view for each of 20 draws. Predictions are averaged after softmax. On the 420 heterogeneous piles, full-token TV is 17.68%, compared with 17.74% for the averaged prediction and 18.23% for the average individual-draw error. Thus the Monte Carlo average is close to full-token evaluation, while a single smaller-token draw is less accurate on average.

Within-HMS prior sensitivity. The main prior averages conditional HMS targets over training piles with positive HMS mass. The 10% threshold defines evaluation eligibility only. An alternative prior normalizes the mean unnormalized HMS shares; another minimizes training conditional TV over a constant probability vector. Their seed-42 redistribution TV is 25.11% and 24.67%, respectively, versus 25.12% for the main prior and 11.19% for the model. Across the three seeds, mode redistribution TV ranges from 11.19% to 11.20%, versus 25.10% to 25.12% for the main prior.

The difference also persists among the 411 evaluation piles with at least two active HMS classes and at least 10% HMS: model/prior redistribution TV is 15.62/26.77%. On all 600 primary piles, the prior-minus-model difference is 13.94 percentage points, with a paired within-fold 95% pile-bootstrap interval of [12.44, 15.42] (10,000 replicates). These comparisons keep the predicted coarse composition unchanged.

Table S2. TV difference from Mean pooling, in percentage points, with paired 95% bootstrap intervals. Positive values favor Mean pooling.
<table><tr><td>Method</td><td>All piles</td><td>Heterogeneous</td></tr><tr><td>Self-attention</td><td>0.32 [−0.08, 0.72]</td><td>0.66 [0.19, 1.12]</td></tr><tr><td>Gated ABMIL</td><td>1.99 [1.27, 2.73]</td><td>2.41 [1.59, 3.26]</td></tr><tr><td>No projector</td><td>0.59 [0.09, 1.14]</td><td>0.07 [−0.41, 0.53]</td></tr></table>

Layer depth and fusion. Table S3 compares DINOv2 ViT-L intermediate features with the final layer on $F _ { 0 } .$ Two-layer fusion uses one learned scalar: αx $_ { 1 7 } + ( 1 - \alpha ) \mathbf { x } _ { \mathrm { f i n a l } } .$ , with $\alpha = { \mathrm { s i g m o i d } } ( a )$ . Four-layer fusion uses learned softmax weights over layers 4, 11, 17, and the final layer. These weights are shared across tokens, views, and piles. Aligned features are combined before the shared projector and mean pooling. Neither fusion improves overall TV over the final-layer baseline.

Table S3. DINOv2 ViT-L layer depth and fusion on $F _ { 0 }$ (N = 150). TV (%) from one run per configuration, using Mean pooling.
<table><tr><td>Features</td><td>All</td><td>Hom.</td><td>Hetero.</td></tr><tr><td>Final layer</td><td>12.83</td><td>5.15</td><td>15.92</td></tr><tr><td>DINOv2 ViT-L layers</td><td></td><td></td><td></td></tr><tr><td>Layer 4</td><td>23.47</td><td>30.17</td><td>20.78</td></tr><tr><td>Layer 11</td><td>15.04</td><td>10.86</td><td>16.71</td></tr><tr><td>Layer 17</td><td>14.23</td><td>10.07</td><td>15.90</td></tr><tr><td>DINOv2-L layer fusion</td><td></td><td></td><td></td></tr><tr><td>Scalar mixing: 17/final</td><td>12.96</td><td>5.93</td><td>15.79</td></tr><tr><td>Softmax mixing: 4/11/17/final</td><td>13.30</td><td>7.18</td><td>15.76</td></tr></table>

Ordinal-loss control. The extra training term is $\lambda m _ { i } W _ { 1 , i } / 3 .$ , where $m _ { i }$ is target HMS mass and $W _ { 1 , i }$ compares the normalized HMS distributions. It is zero for targets without HMS. With $\lambda = 0 . 1$ and seed 42, overall $F _ { 1 } { - } F _ { 4 }$ TV is 13.44%, versus 13.43% for KL alone. This tests one weighting of the ordinal term.

## B. Oracle Optimization and Structured Control

Oracle optimization. The oracle keeps the trained Mean pooling model frozen and optimizes softmax weights over view embeddings separately for each subset. Adam minimizes TV to the supplied target for 300 steps at learning rate 0.05, starting from one uniform and two random weight initializations. We also retain uniform weighting, single-view solutions, and solutions from smaller nested subsets, then report the lowest error among these candidates. This keeps any solution available with fewer views, but does not guarantee a global optimum. The weights form convex combinations of view embeddings; the final softmax means that the output probabilities need not be convex combinations of single-view predictions.

Structured-null oracle. Figure S1 compares target-conditioned gains on the same 102 heterogeneous $F _ { 0 }$ piles. Five shuffled targets per pile preserve the dominant class and active-class count, with dominant shares within 10 percentage points; exact active-class identities may differ. Recorded and shuffled targets use the same 30 nested view subsets. Counts are capped at the available views: at k = 4, 87 piles have four candidates and 15 have fewer. With one view, the gain is zero by construction.

![](images/9fc99670af339e7bc5df93bcb9f4b34e7a770873a807b35ccc9d1f018dec3223.jpg)  
Figure S1. Target-conditioned oracle on 102 heterogeneous $F _ { 0 }$ piles. Top: gain over uniform weighting. Bottom: paired recorded-minus-nul gain. Bars show 95% pile-bootstrap intervals; candidate counts are capped at the available views.

With all views, recorded/null oracle TV is 5.32/7.43%, while the gains over uniform weighting are 9.34/9.69 percentage points. Similar gains therefore coexist with different residual errors. The result shows that answer-informed reweighting can fit several structurally similar targets; it does not rule out useful image information or learnable view weights. Restricting the control to the 87 piles with at least four views gives recorded/null gains of 10.07/10.50 percentage points and residual TV of 4.46/6.69%.

## C. Large Disagreements on Homogeneous Piles

For a homogeneous target of class $c , \mathrm { T V } = 1 - \hat { y } _ { c }$ : a confident prediction of another class can therefore produce nearly 100% TV. Among the 180 homogeneous primary piles, the seed-42 Mean model has eight errors above 30% TV, including five above 90%. The two largest errors are 99.99% and 99.98%. Both targets are entirely > 6 mm HMS, whereas the model assigns 98.98% and 99.97%, respectively, to Stamped steel.

Figure S2 shows two views of the second case. The cut-out plate shapes resemble stamping offcuts, but the operator recorded entirely > 6 mm HMS. Plate thickness may explain this taxonomy choice. Without measurements or independen reassessment, these views cannot establish the cause of the disagreement.

![](images/6c6202ed92c701d959409cad1ee04869eb3320075eef33e1f6d430c197aaf205.jpg)  
Figure S2. Two of three retained views of one pile with a homogeneous target: 100% > 6 mm HMS. The seed-42 Mean pooling mode assigns 99.97% to Stamped steel using all three views (TV 99.98%). The cut-out plate shapes resemble stamping offcuts, but their thickness and the operator’s taxonomy decision cannot be established from appearance alone.

## D. Qualitative Evidence

Centered token contributions. For a projected token $\mathbf { z } _ { t }$ and class-head weights $\mathbf { w } _ { c } ,$ we display $( \mathbf { w } _ { c } - \bar { \mathbf { w } } ) ^ { \top } \mathbf { z } _ { t }$ , where $\begin{array} { r } { \bar { \bf w } = \frac { 1 } { 6 } \sum _ { c = 1 } ^ { 6 } { \bf w } _ { c } } \end{array}$ . Subtracting the shared weight component leaves softmax probabilities unchanged. Red and blue show positive and negative contributions relative to the average class logit, and the spatially constant head bias is omitted. One token may support several classes, for example where materials overlap. These are logit contributions, not class-exclusive segmentation or local mass measurements (Fig. S3).

![](images/b691cc9978fd35b942cd2c8f158d8cbc24088e103affe8fb271b26cb13461708.jpg)  
Figure S3. Centered token contributions from the Mean pooling model for one retained view. Red supports the named class relative to the average class logit; blue opposes it. In panel order, the pile-level target is (0, 70, 15, 10, 5, 0)% and the prediction using all retained views is (0.00, 37.48, 32.15, 24.48, 3.27, 2.63)%.

![](images/803a27b517070451e228761b602dd24b9416481e8d110f8ffb8a11f8f446b31f.jpg)

![](images/f5266a6f4bf90bd5c6fa237b42cc28d4b0eb45bb2387bf5ee64bc99756646c57.jpg)

b)  
![](images/c2665c0233046c604f4004f46281d48fe7ef3bb9bed63b993044ced782e1573b.jpg)

![](images/9d70ff64ea34569a40ba3989a62620592a6513fe5c96cc9e6a52f0d7ee81218c.jpg)  
Figure S4. Thickness measurements during grading. (a) Target/Mean/oracle > 6 mm shares are 70/98.0/93.8%; (b) target/Mean/oracle 2–4 mm shares are 70/15.5/20.4%. Reweighting moves both predictions toward the operator estimates but leaves substantial thickness-class disagreement. The local measurements do not establish pile-level mass fractions.

## E. Backbone Implementation

All backbones in main-paper Table 3 are frozen. The DINOv2 models and CLIP ViT-L/14 use 896×504 inputs, while DINOv3 ViT-L/16 and ConvNeXt-V2-L use 1024 × 576. The ViTs produce 64 × 36 spatial grids, pooled in 4 × 4 blocks to 16 × 9. CLIP uses its own RGB normalization and interpolated positional embeddings. The ConvNeXt-V2-L checkpoint is FCMAE pretrained and ImageNet fine-tuned. Its stride-32 32 × 18 map is pooled in 2 × 2 blocks to 16 × 9. It uses checkpoint-specific normalization and direct bicubic resizing, without the checkpoint’s native crop. Feature dimensions differ across backbones, but all are projected to 256 dimensions.

## F. Pooled Risk–Coverage

Figure S5 extends the main-paper heterogeneous-pile analysis to all eligible piles, including homogeneous targets.  
![](images/dfe5b790b2bf4850dd06f125cc59a728eabb74479d08fbccf6ab8739ac9e29e4.jpg)  
Figure S5. Pooled risk–coverage for the seed-42 Mean pooling model on all 567 $F _ { 1 } { - } F _ { 4 }$ piles with at least two views, including homogeneous targets; 33 single-view piles are excluded. Shading shows pointwise 95% pile-bootstrap intervals. Both scores reduce average retained error, but the failure in Fig. S2 remains among entropy’s most-confident 10%.