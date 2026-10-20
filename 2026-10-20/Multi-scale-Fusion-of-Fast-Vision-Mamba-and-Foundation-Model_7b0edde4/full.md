# Multi-scale Fusion of Fast Vision Mamba and Foundation Model Features for Patch-Level Gleason Grading in Computational Pathology

Zihe Zhang<sup>1</sup> , Weimiao Yu<sup>2</sup> , and Piumi Don Simonge<sup>2</sup> <sup>()</sup>

<sup>1</sup> Nanyang Technological University, Singapore ZH0013HE@e.ntu.edu.sg

2 Bioinformatics institute, Agency for Science Technology and Research, Singapore yu\_weimiao@a-star.edu.sg, piumi\_simonge@a-star.edu.sg

Abstract. Automated Gleason grading of prostate whole slide images (WSIs) enables faster and more consistent cancer risk stratification. However, it remains a challenging task due to the subtle morphological differences between grades, the severe class imbalance arising from rare patterns, and the diagnostic features that span multiple spatial scales. Pathology foundation models (FMs) extract strong patch representations but often operate at a single-scale, which limits spatial context and disproportionately afects classes that depend on broader context. The challenge of fusing multiple scales and complementary models, with minimal data and computational resources, remains unresolved. We propose a multi-scale classifier built on state-space models. Our method combines two complementary feature sources: a frozen foundation model, and a lightweight FastVim, pre-trained on ImageNet and distilled from the foundation model, that reads the raw image crops. At each scale, a Mamba model encodes the features from both sources with linear-time spatial context, and a jointly trained head fuses the encoded representations into one prediction. On the PANDA dataset, our method reaches macro F1 of 0.894 and Gleason 5 F1 of 0.802 under 5-fold cross-validation, a 12% and 21% relative improvement over the best single-scale model. A controlled study over 5 foundation models, 5 base classifiers and 3 scales confirms that the gain holds across encoders, with larger improvements observed at smaller scales.

Keywords: Computational pathology · State-space models · Gleason grading · Foundation models · Multi-scale fusion

## 1 Introduction

Prostate cancer is the second most commonly diagnosed cancer in men worldwide [17]. Gleason grading of biopsy tissue provides essential prognostic information that guides treatment decisions [4], but it is labour intensive and shows considerable inter-observer variability [10]. Gleason grading assigns prostate cancer tissue into 3 main architectural patterns: Gleason pattern 3, 4, and 5, where higher patterns correspond to progressively poorer glandular diferentiation, accompanied by more pronounced cellular atypia [4]. Reliable and reproducible automation of this assessment is therefore clinically valuable.

Deep learning is the standard tool for histopathology image analysis [10], and pathology foundation models (FMs) have become its dominant approach. Pretrained with heavy self-supervision on large datasets, FMs produce rich patch representations that transfer across downstream tasks. UNI [2] pre-trains a ViT-L/16 with DINOv2 [16] on over 100,000 whole slide images (WSIs), and its successor UNI2 scales further. Virchow [19] scales to a 632M parameter ViT-H/14 trained on 1.5M slides, Virchow2 [21] adds mixed magnification training, and CONCH [13] adds vision-language pre-training.

State-of-the-art FMs are trained at a pre-determined patch size and lose accuracy at substantially diferent scales [14]. However, Gleason grading is intrinsically multi-scale, relying on cues from cellular detail to glandular and tissue architecture. Since WSIs contain hundreds of thousands of patches, modelling spatial context with attention is quadratic and quickly becomes computationally prohibitive. State-space models (SSMs) such as Mamba [5] process long token sequences in linear time. The vision counterpart of Mamba, Vision Mamba (ViM) [20], extends this architecture to natural image understanding, while pathology-specific variant Vim4Path [15] adapts this architecture for computational pathology. FastVim [8] improves on ViM, maintaining accuracy while achieving up to 72.5% faster inference even at high resolution (2048 × 2048). Beyond eficient context modelling, the remaining question is how to combine scales. Prior multi-scale analysis examines several patch sizes but typically combines the single-scale predictions by ensembling or averaging [3,9], which treats the scales independently and discards their joint structure. Multi-scale fusion of histopathology images in an end-to-end model still remains an open challenge.

To close this gap, we formulate multi-scale patch classification as a single, jointly trained network. Our method draws features from two complementary sources: a frozen FM, and a lightweight FastVim, pre-trained on ImageNet and distilled from the FM, that reads the raw image crops the pooled FM features miss. At each scale, a Mamba encoder adds spatial context, and a jointly trained head fuses the representations from both sources into one prediction, rather than averaging or voting on single-scale outputs. Full details are given in Sec. 2.

We validate our method on the PANDA dataset [1], a large-scale public prostate cohort with pixel-level masks, on a 5-class tissue classification task (see Sec. 3.1). Under 5-fold cross-validation, our method increases macro F1 from 0.796 for the best single-scale model to 0.894, a 12% relative gain, and Gleason 5 F1 from 0.661 to 0.802, a 21% relative gain (Table 1). With minimal resources, our FastVim branch reaches higher throughput than a same-size vision transformer at these input sizes (Table 2), and it is trained on the target cohort alone.

![](images/703e4373a2601b43721e906fb435b2bc856998e0a3da2298614edb9325e97bf6.jpg)  
Fig. 1. Overview of the proposed method. Since Virchow2 accepts only its native 256 px input, the 512 px and 1024 px scale representations are formed by averaging the embeddings of the 2 × 2 and 4 × 4 constituent tiles, respectively. In contrast, the FastVim branch encodes native crops at all three scales directly. The representations from both branches are each encoded by a Mamba encoder and fused into one patch-level prediction, and a per-scale classification head supplies an auxiliary training signal (deep supervision). Only the Mamba encoders, the per-scale classification heads and the fusion head are trained, and the two feature encoders are kept frozen.

## 2 Method

## 2.1 Overview

We frame patch classification as a multi-scale problem, describing each patch at three spatial scales. Our method (Fig. 1) has three parts. Two frozen encoders produce patch features at every scale: a pretrained foundation model, and a lightweight FastVim distilled from it that reads the raw image crops (Sec. 2.2 and 2.3). A Mamba encoder at each scale then adds spatial context from neighbouring patches (Sec. 2.4). Finally, a fusion head combines the encoded representations into a patch-level prediction, and a per-scale classification head provides deep supervision (Sec. 2.5).

## 2.2 Multi-scale Feature Extraction

The FM branch uses Virchow2 [21], which achieved best performance among 5 state-of-the-art FMs (more details in Sec. 3.3). We feed it patches at its recommended input size of 256 px (0.48 µm/pixel, 20×) and each patch is converted to a 1280-dimensional embedding. The larger scales (512 and 1024 px) are defined as the mean of these 256 px embeddings over non-overlapping 2 × 2 and 4 × 4 grids. Let S denote the number of scales, with S=3 throughout. Together, the scales capture local cellular detail, glandular structure, and broader tissue architecture.

## 2.3 FastVim Branch

The pooling above discards fine morphological detail present in the original crops, which cannot be recovered downstream. We therefore add a second branch that reads the image crops directly. To keep it computationally feasible, we use a lightweight ViM, FastVim [8], pre-trained on ImageNet. Among the backbones we compared, FastVim ofers the best trade-of between accuracy and throughput at larger inputs (Sec. 3.3, Table 2). It is trained on crops at S scales, each fed at its native size except the smallest patch size $( 2 5 6 \mathrm { p x } )$ , which is resized to $2 2 4 \mathrm { p x }$

We distill Virchow2 into this branch via knowledge distillation [7], applied independently at each scale. The teacher is the frozen FM with an MLP classifier trained on the same patches, which gives soft targets $\mathbf { q } ^ { ( s ) } = \mathrm { s o f t m a x } ( \ell ^ { ( s ) } / T )$ at each scale s and temperature T, with $T > 1$ so the targets reflect relative class probabilities rather than a hard label. The student, with logits $\hat { \boldsymbol { \ell } } ^ { ( s ) }$ , is trained from its ImageNet initialisation with the loss ${ \mathcal { L } } _ { \mathrm { d i s t i l l } }$ . This is calculated as the weighted sum of a cross-entropy term (CE) and a Kullback-Leibler divergence term (KL), with weight α and ground truth label y.

$$
\mathcal { L } _ { \mathrm { d i s t i l l } } = \alpha \mathrm { C E } \Big ( \mathbf { y } , \mathrm { s o f t m a x } ( \hat { \pmb { \ell } } ^ { ( s ) } ) \Big ) + ( 1 - \alpha ) T ^ { 2 } \mathrm { K L } ( \mathbf { q } ^ { ( s ) } \| \mathrm { s o f t m a x } ( \hat { \pmb { \ell } } ^ { ( s ) } / T ) )\tag{1}
$$

Once trained, the student is frozen and never updated again. We take its penultimate pooled features, of dimension $d _ { \mathrm { V i M } } = 3 8 4$ against $d _ { \mathrm { F M } } = 1 2 8 0$ for the FM branch. Both branches then stay frozen for the remainder of training (Sec. 2.4).

## 2.4 Scale Encoders and Fusion Head

Each branch $b \in \mathcal { B } = \{ \mathrm { F M , V i M } \}$ provides a $d _ { b }$ -dimensional feature for every patch at each scale s. We tile each slide into fixed-length sequences of $N { = } 2 5 6$ patches in row-major order, where N is unrelated to the patch size, and repeat the coarser-scale features across the finest-scale patches they cover so that all sequences are aligned on the same patch grid.

Each branch b and scale s has its own Mamba encoder [5] $f _ { b } ^ { ( s ) } : \mathbb { R } ^ { N \times d _ { b } } $ $\mathbb { R } ^ { N \times d ^ { \prime } }$ , with $d ^ { \prime } { = } 2 5 6$ , which adds spatial context from neighbouring patches in $O ( N )$ time and memory. Weights are not shared across scales or across branches, since the two branches difer in feature dimension. We write $\mathbf { h } _ { b } ^ { ( s ) } \in \mathbb { R } ^ { d ^ { \prime } }$ for the row of the output that corresponds to a given patch.

These representations are concatenated into z, where

$$
\mathbf { z } = \left[ \mathbf { h } _ { \mathrm { F M } } ^ { \left( 1 \right) } \ \lVert \cdot \cdot \cdot \lVert { \mathbf { h } _ { \mathrm { F M } } ^ { \left( S \right) } } \rVert \mathbf { h } _ { \mathrm { V i M } } ^ { \left( 1 \right) } \rVert \cdot \cdot \cdot \lVert { \mathbf { h } _ { \mathrm { V i M } } ^ { \left( S \right) } } \right] \in \mathbb { R } ^ { | \mathcal { B } | S d ^ { \prime } }\tag{2}
$$

and a two-layer fusion head $g : \mathbb { R } ^ { | B | S d ^ { \prime } } \xrightarrow { } \Delta ^ { C - 1 }$ maps z to a distribution over the $C { = } 5$ classes. A per-scale classification head predicts from $[ \mathbf { h } _ { \mathrm { F M } } ^ { \left( s \right) } \left\| \mathbf { h } _ { \mathrm { V i M } } ^ { \left( s \right) } \right\}$ alone, giving deep supervision during training (Sec. 2.5).

Because the fusion head combines the full representations before any singlescale decision, it is more expressive than a fixed scale. A uniform average of predictions weights every class and scale equally, whereas the fusion head learns weights that depend on the class and the scale. Sec. 3.2 reports its margin over this uniform average baseline. The fusion head adds under 1% of the parameters and inference time of the Mamba encoders.

## 2.5 Training and Computational Cost

We first train FastVim on image crops with distillation from the frozen FM, using temperature $T { = } 4$ and distillation weight α=0.5. We then train the Mamba encoders, the classification heads and the fusion head with the loss ${ \mathcal { L } } ,$ computed as the sum of a focal loss on the fused prediction and focal losses on the individual predictions at each scale,

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { f o c a l } } ( \hat { y } ) + \lambda \sum _ { s } \mathcal { L } _ { \mathrm { f o c a l } } \big ( \hat { y } ^ { ( s ) } \big ) ,\tag{3}
$$

where $\hat { y }$ is the fused prediction, $\hat { y } ^ { ( s ) }$ is the prediction of the classification head at scale $s ,$ and $\lambda { = } 0 . 3$ . Both terms are focal losses $( \gamma { = } 1 )$ with class weights $w _ { k } \propto 1 / \sqrt { n _ { k } }$ , where $n _ { k }$ is the number of patches of class $k ,$ to counter the class imbalance. We optimise with AdamW [12] at learning rate $1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 4 }$ , batch size 32, for 50 epochs with cosine scheduling, and keep the checkpoint with the best validation balanced accuracy. Each Mamba encoder has four layers with hidden dimension 256 and state size 16, and the fusion head is a two-layer MLP with hidden dimension 128.

## 3 Experiments and Results

## 3.1 Dataset and Evaluation

Unless otherwise stated, all experiments use Virchow2, the strongest of the 5 FMs we compare (Sec. 3.4). We evaluate our method on the PANDA dataset [1], which provides 10,508 slides, of which 10,497 have pixel-level tissue masks. We hold out a fixed test set of 2,100 slides and split the remaining 8,397 into 5 folds. Each fold trains on 4 parts and validates on the fifth, giving 5 models that we evaluate on the same held-out test set and average. Splitting is stratified at the slide level by each slide’s major Gleason grade, preserving the class distribution. From each slide we extract patches at the 256, 512, and 1024 px scales (Sec. 2.2) and label each patch by the majority class above a 70% threshold, giving about 3.8M patches at the 256 scale. The dataset is severely imbalanced, with 79% normal, 16% stroma, 1.8% Gleason 3 (G3), 3.4% Gleason 4 (G4), and 0.5% Gleason 5 (G5). Fig. S1 in the supplementary material shows the data splits across main experiment and ablation studies (Sec. 3.3) as well as the class distribution across each split. We report balanced accuracy, macro F1, and per-class F1. F1 of the minority class (Gleason 5) is a primary criterion for assessing performance under class imbalance.

Table 1. Performance of our multi-scale method compared with single-scale baselines and a multi-scale uniform average. All experiments use the best-performing foundation model, Virchow2, where applicable. Results are mean ± standard deviation over the 5 folds on the held-out test set. Best value per column in bold.
<table><tr><td>Model</td><td>Macro F1</td><td>F1 (G5)</td><td>Bal. Acc</td></tr><tr><td>Single-scale, 256</td><td>0.794±0.012</td><td>0.636±0.044</td><td>0.916±0.003</td></tr><tr><td>Single-scale, 512</td><td>0.796±0.016</td><td>0.661±0.036</td><td>0.922±0.004</td></tr><tr><td>Single-scale, 1024</td><td>0.768±0.028</td><td>0.626±0.080</td><td>0.929±0.003</td></tr><tr><td>Multi-scale (uniform average)</td><td>0.747±0.003</td><td>0.535±0.019</td><td>0.923±0.001</td></tr><tr><td>Multi-scale (ours)</td><td>0.874±0.001</td><td>0.784±0.005</td><td>0.860±0.005</td></tr><tr><td>Multi-scale + FastVim (ours)</td><td>0.894±0.002</td><td>0.802±0.003</td><td>0.926±0.004</td></tr></table>

## 3.2 Multi-scale Fusion Results

Table 1 compares the performance of our method with single-scale baselines and a multi-scale uniform average, which averages the class probabilities of the three single-scale models and weights every scale and class equally. The best singlescale reaches macro F1 of 0.796 and Gleason 5 F1 of 0.661, and the uniform average does not improve on this (macro F1 = 0.747), being dominated by the majority-heavy coarse scale. Our method increases macro F1 to 0.874 and Gleason 5 F1 to 0.784, with consistent improvements (±0.001 and ±0.005) across all folds. However, this improvement is accompanied by a marginal reduction of majority-class recall, resulting in a slightly reduced balanced accuracy of 0.860. By adding the FastVim branch, balanced accuracy is restored to 0.926, on par with the best single-scale, while macro F1 and Gleason 5 F1 increase further to 0.894 and 0.802. The confusion matrix (Fig. 2) shows the remaining errors fall mostly between adjacent Gleason patterns. Further visual inspection shows that some high-grade patches predicted as normal lie at tumour boundaries and appear to be afected by annotation noise, suggesting that part of the balanced accuracy gap reflects label noise rather than model error.

## 3.3 Ablation Studies

This subsection evaluates two design choices in the absence of the FM branch, comparing candidate backbones for the FastVim branch (Table 2) and testing whether multi-scale fusion generalises across diferent image encoders (Table 3). To limit the computational cost, the ablation experiments use a single stratified train/validation/test split (70/15/15), which ensures the patch-level class proportion is within 0.54 percentage points of the whole cohort. Since the ablation experiments use a diferent dataset split from Sec. 3.2, absolute values are not comparable across the two protocols.

Table 2 compares candidate backbones for the FastVim branch at singlescale (1024 crops). The candidates cover the vision transformer, state-space and eficient state-space models at 22 − 26M parameters, so the comparison isolates the architecture rather than the capacity. Since the Gleason patterns follow a graded histological severity, we report quadratic weighted kappa (QWK), which penalises misclassifications in proportion to their distance from the true grade. We also report throughput, as this branch reads the crops natively and its cost grows with scale. FastVim-S reaches the highest macro F1 and QWK at about 1.5× the throughput of ViM-S. Distilling Virchow2 into FastVim-S increases its macro F1 from 0.665 to 0.700 and QWK from 0.631 to 0.672.

![](images/cb48a8125e5cd1dac03af2d152679cc350ca4d658ee03e1713ba554e3e1d6f4f.jpg)  
Fig. 2. Confusion matrix showing that our method classifies most patches correctly and that the remaining errors fall between adjacent Gleason patterns. Each cell is the proportion of a true class (row) predicted as each class (column), so the diagonal gives per-class recall.

Table 2. Backbones for the FastVim branch at 1024 crops (70/15/15 split, not comparable with Table 1). Throughput is measured in patches per second on an A100 GPU.
<table><tr><td>Backbone</td><td>Macro F1</td><td>QWK</td><td>Throughput</td><td>#Params</td></tr><tr><td>DeiT-S (ViT) [18]</td><td>0.621</td><td>0.597</td><td>47</td><td>22M</td></tr><tr><td>ViM-S [20]</td><td>0.625</td><td>0.598</td><td>43</td><td>26M</td></tr><tr><td>FastVim-S [8]</td><td>0.665</td><td>0.631</td><td>63</td><td>26M</td></tr><tr><td>FastVim-S with distillation</td><td>0.700</td><td>0.672</td><td>63</td><td>26M</td></tr></table>

Table 3 shows that applying the proposed multi-scale fusion to four image encoders consistently improves both macro F1 and Gleason 5 F1. Each encoder is frozen and provides features on the 256 px patches. In Table 3 single-scale column uses these features alone, and the fusion column adds the pooled 512 and 1024 px scales. All downstream components remain unchanged. For Virchow2, macro F1 and Gleason 5 F1 increase from 0.717 to 0.852 and from 0.514 to 0.730, respectively. Similar improvements are observed across the vision transformer [18], the convolutional neural network [6], and the state-space [8] encoder. Therefore, we conclude that the fusion improves minority-class performance regardless of the underlying encoder.

Table 3. Multi-scale fusion applied to diferent image encoders, on the 70/15/15 split. Best results per metric in bold.
<table><tr><td rowspan="2">Encoder</td><td colspan="3">Macro F1</td><td colspan="3">F1 (G5)</td></tr><tr><td>Single-scale</td><td>Fusion</td><td>∆ (abs)</td><td>Single-scale</td><td>Fusion</td><td>∆ (abs)</td></tr><tr><td>Virchow2 [21]</td><td>0.717</td><td>0.852</td><td>+0.135</td><td>0.514</td><td>0.730</td><td>+0.216</td></tr><tr><td>DeiT-S (ViT) [18]</td><td>0.635</td><td>0.799</td><td>+0.164</td><td>0.466</td><td>0.661</td><td>+0.195</td></tr><tr><td>ResNet-50 (CNN) [6]</td><td>0.547</td><td>0.781</td><td>+0.234</td><td>0.401</td><td>0.592</td><td>+0.191</td></tr><tr><td>FastVim-S (ViM) [8]</td><td>0.649</td><td>0.824</td><td>+0.175</td><td>0.524</td><td>0.656</td><td>+0.132</td></tr></table>

∆ (abs): absolute improvement (Fusion − Single-scale).

![](images/e1c351c342eb44f4cdc7350607a381240caf9693da7fe22e41bb000920c020e5.jpg)  
Fig. 3. Change in evaluation metrics across diferent scales and foundation models, for two base classifiers. (a) Mamba base classifier. (b) Patch-level attention base classifier. Each line is one foundation model, with the mean across models shown in black excluding CONCH, which lacks a 256 px run.

## 3.4 Efect of Scale, Foundation Model, and Base Classifier

This subsection examines the efects of image scale, the foundation model, and the base classifier on single-scale performance. The 5 base classifiers are a singleand a 3-layer MLP, a Mamba3 [11], and a patch-level attention encoder, each replacing the Mamba encoder. For Virchow2 with the Mamba base classifier, balanced accuracy increases with scale, from 0.916 at 256 to 0.929 at 1024,

while macro F1 peaks at 512 (0.796) and falls at 1024 (0.768), because pooling over larger regions separates the majority classes but dilutes the fine detail the minority classes rely on. The Mamba base classifier gives the highest balanced accuracy at every scale and the highest macro F1 at 256 and 512 px, which is where the minority classes are best separated, so we adopt it in the fused model. The advantage of the Mamba base classifier over a single MLP shrinks from 0.048 balanced accuracy at 256 to 0.012 at 1024, as larger-scale features already carry regional context. Table S1 in the supplementary material reports balanced accuracy, macro F1, and Gleason 5 F1 across all FM-classifier-scale combinations.

## 4 Conclusion

We present a computational method for patch-level Gleason grading that encodes FM features at 256, 512, and 1024 scales with Mamba encoders and fuses their representations in one jointly trained network, extended by a distilled FastVim branch that reads the raw image crops. On the PANDA dataset, our method substantially improves macro F1 and minority-class performance over the best single-scale model while keeping balanced accuracy competitive.

Our method has two practical advantages. First, the same fusion improves macro F1 and Gleason 5 F1 for foundation model, vision transformer, convolutional and ViM encoders (Table 3), so it is agnostic to the underlying encoder. Second, current pathology FMs typically operate on a fixed input size, preventing direct encoding of multiple image scales. Our FastVim branch instead processes the native image crops at each scale and improves over the FM branch on every metric, while running faster than a vision transformer of the same size (Table 2) and requiring far less training data and computation than building an FM.

The main limitation is a slightly lower balanced accuracy than the best single-scale, as our model trades some majority-class recall for large gains on the minority (Gleason 5) class. Upon inspection, part of this drop comes from boundary patches that are mislabelled as cancer regions (Sec. 3.2), so we prioritise macro F1 and minority-class F1, which better reflect performance on the rare high-grade tissue. Our method can be further extended in future work to include validation across additional cohorts and scanners, and to develop a self-supervised pathology state-space feature extractor that eliminates the mean pooling step.

Acknowledgments. This work is supported by the Agency for Science, Technology and Research (A\*STAR), Singapore with Grant ID H25J6a0011 under the Industrial Alignment Fund Pre-positioning Programme (AIF-PP). The computational work is performed on resources provided by the National Supercomputing Centre, Singapore.

Disclosure of Interests. The authors have no competing interests to declare that are relevant to the content of this article.

## References

1. Bulten, W., Kartasalo, K., Chen, P.H.C., et al.: Artificial intelligence for diagnosis and Gleason grading of prostate cancer: the PANDA challenge. Nature Medicine 28(1), 154–163 (2022). https://doi.org/10.1038/s41591-021-01620-2

2. Chen, R.J., Ding, T., Lu, M.Y., et al.: Towards a general-purpose foundation model for computational pathology. Nature Medicine 30, 850–862 (2024). https://doi. org/10.1038/s41591-024-02857-3

3. Das, A., Karri, S., Guha Roy, A., Chatterjee, J., Sheet, D.: Classifying histopathology whole-slides using fusion of decisions from deep convolutional network on a collection of random multi-views at multi-magnification. In: IEEE International Symposium on Biomedical Imaging (ISBI). pp. 1024–1027 (2017). https: //doi.org/10.1109/ISBI.2017.7950690

4. Epstein, J.I., Egevad, L., Amin, M.B., et al.: The 2014 International Society of Urological Pathology (ISUP) consensus conference on Gleason grading of prostatic carcinoma. American Journal of Surgical Pathology 40(2), 244–252 (2016), https: //pubmed.ncbi.nlm.nih.gov/26492179/

5. Gu, A., Dao, T.: Mamba: Linear-time sequence modeling with selective state spaces (2024), https://arxiv.org/abs/2312.00752

6. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 770–778 (2016). https://doi.org/10.1109/CVPR.2016.90

7. Hinton, G., Vinyals, O., Dean, J.: Distilling the knowledge in a neural network (2015), https://arxiv.org/abs/1503.02531

8. Kapse, S., Betz, P., Sivanandan, S., et al.: FastVim: Eficient vision Mamba by token pooling (2025), https://arxiv.org/abs/2502.00594

9. Koziarski, M., Cyganek, B., Niedziela, P., et al.: DiagSet: a dataset for prostate cancer histopathological image classification. Scientific Reports 14, 6780 (2024). https://doi.org/10.1038/s41598-024-52183-4

10. van der Laak, J., Litjens, G., Ciompi, F.: Deep learning in histopathology: the path to the clinic. Nature Medicine 27(5), 775–784 (2021). https://doi.org/10.1038/ s41591-021-01343-4

11. Lahoti, A., Li, K.Y., Chen, B., Wang, C., Bick, A., Kolter, J.Z., Dao, T., Gu, A.: Mamba-3: Improved sequence modeling using state space principles (2026), https://arxiv.org/abs/2603.15569

12. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: 7th International Conference on Learning Representations (2019), https://arxiv.org/abs/ 1711.05101

13. Lu, M.Y., Chen, B., Williamson, D.F.K., et al.: A visual-language foundation model for computational pathology. Nature Medicine 30, 863–874 (2024). https://doi. org/10.1038/s41591-024-02856-4

14. Möllers, A., Hense, J., Schulz, F., et al.: Mind the gap: Continuous magnification sampling for pathology foundation models (2026), https://arxiv.org/abs/2601. 02198

15. Nasiri-Sarvi, A., Trinh, V.Q.H., Rivaz, H., Hosseini, M.S.: Vim4Path: Selfsupervised vision Mamba for histopathology images. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops. pp. 6894–6903 (June 2024), https://arxiv.org/abs/2404.13222

16. Oquab, M., Darcet, T., Moutakanni, T., et al.: DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research (2024), https://arxiv.org/abs/2304.07193

17. Sung, H., Ferlay, J., Siegel, R.L., et al.: Global cancer statistics 2020: GLOBOCAN estimates of incidence and mortality worldwide for 36 cancers in 185 countries. CA: A Cancer Journal for Clinicians 71(3), 209–249 (2021), https://doi.org/ 10.3322/caac.21660

18. Touvron, H., Cord, M., Douze, M., et al.: Training data-eficient image transformers & distillation through attention. In: Proceedings of the 38th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 139, pp. 10347–10357 (18–24 Jul 2021), https://proceedings.mlr.press/ v139/touvron21a.html

19. Vorontsov, E., Bozkurt, A., Casson, A., et al.: A foundation model for clinicalgrade computational pathology and rare cancers detection. Nature Medicine 30, 2924–2935 (2024). https://doi.org/10.1038/s41591-024-03141-0

20. Zhu, L., Liao, B., Zhang, Q., et al.: Vision mamba: eficient visual representation learning with bidirectional state space model. In: Proceedings of the 41st International Conference on Machine Learning (2024), https://arxiv.org/abs/2401. 09417

21. Zimmermann, E., Vorontsov, E., Viret, J., et al.: Virchow2: Scaling self-supervised mixed magnification models in pathology (2024), https://arxiv.org/abs/2408. 00738

## Supplementary Material

![](images/0d606e628959d52dace58febf85216a86ba5b2168d0492bcc99e455b2a7c61af.jpg)  
Fig. S1. Data splits and their class distribution. (a) The main split, used for the 5-fold cross-validation experiments presented in Sec. 3.2 and tested on held-out test split. The ablation split, the single $7 0 / 1 5 / 1 5$ train/validation/test split used in Sec. 3.4. (b) Class distribution within each split, relative to the whole cohort, on a log scale.

Table S1. Evaluation metrics across foundation models, base classifiers, and input scales. Results report the mean held-out test performance across 5-fold cross-validation experiments. BA is balanced accuracy, MF1 is macro F1, and G5 is the F1 of Gleason 5. Since CONCH requires an input size of at least 512 px, results are reported only for patch sizes of 512 px and above. No multi-scale fusion is applied in these experiments.
<table><tr><td></td><td></td><td colspan="3">256 px</td><td colspan="3">512 px</td><td colspan="3">1024 px</td></tr><tr><td>FM</td><td>Base classifier</td><td>BA</td><td>MF1</td><td>G5</td><td>BA</td><td>MF1</td><td>G5</td><td>BA</td><td>MF1</td><td>G5</td></tr><tr><td rowspan="5">Virchow2</td><td>Single-layer MLP</td><td>0.869</td><td>0.753</td><td>0.622</td><td>0.897</td><td>0.754</td><td>0.613</td><td>0.917</td><td>0.767</td><td>0.658</td></tr><tr><td>3-layer MLP</td><td>0.872</td><td>0.741</td><td>0.600</td><td>0.898</td><td>0.760</td><td>0.646</td><td>0.915</td><td>0.776</td><td>0.683</td></tr><tr><td>Mamba</td><td>0.916</td><td>0.794</td><td>0.636</td><td>0.922</td><td>0.796</td><td>0.661</td><td>0.929</td><td>0.768</td><td>0.626</td></tr><tr><td>Mamba3</td><td>0.912</td><td>0.772</td><td>0.572</td><td>0.923</td><td>0.780</td><td>0.636</td><td>0.928</td><td>0.725</td><td>0.565</td></tr><tr><td>Patch-level attention</td><td>0.909</td><td>0.775</td><td>0.623</td><td>0.925</td><td>0.794</td><td>0.684</td><td>0.928</td><td>0.771</td><td>0.681</td></tr><tr><td rowspan="5">UNI</td><td>Single-layer MLP</td><td>0.853</td><td>0.719</td><td>0.546</td><td>0.887</td><td>0.747</td><td>0.605</td><td>0.905</td><td>0.763</td><td>0.638</td></tr><tr><td>3-layer MLP</td><td>0.857</td><td>0.708</td><td>0.544</td><td>0.891</td><td>0.731</td><td>0.588</td><td>0.910</td><td>0.756</td><td>0.651</td></tr><tr><td>Mamba</td><td>0.911</td><td>0.781</td><td>0.607</td><td>0.920</td><td>0.776</td><td>0.628</td><td>0.925</td><td>0.734</td><td>0.582</td></tr><tr><td>Mamba3</td><td>0.913</td><td>0.774</td><td>0.597</td><td>0.917</td><td>0.777</td><td>0.641</td><td>0.928</td><td>0.744</td><td>0.585</td></tr><tr><td>Patch-level attention</td><td>0.900</td><td>0.755</td><td>0.581</td><td>0.920</td><td>0.768</td><td>0.613</td><td>0.926</td><td>0.707</td><td>0.522</td></tr><tr><td rowspan="5">UNI2</td><td>Single-layer MLP</td><td>0.863</td><td>0.729</td><td>0.578</td><td>0.893</td><td>0.736</td><td>0.586</td><td>0.911</td><td>0.758</td><td>0.655</td></tr><tr><td>3-layer MLP</td><td>0.866</td><td>0.721</td><td>0.554</td><td>0.895</td><td>0.738</td><td>0.609</td><td>0.914</td><td>0.757</td><td>0.626</td></tr><tr><td>Mamba</td><td>0.913</td><td>0.780</td><td>0.596</td><td>0.921</td><td>0.763</td><td>0.629</td><td>0.928</td><td>0.741</td><td>0.627</td></tr><tr><td>Mamba3</td><td>0.911</td><td>0.779</td><td>0.616</td><td>0.917</td><td>0.772</td><td>0.622</td><td>0.924</td><td>0.750</td><td>0.636</td></tr><tr><td>Patch-level attention</td><td>0.901</td><td>0.765</td><td>0.608</td><td>0.920</td><td>0.765</td><td>0.634</td><td>0.926</td><td>0.710</td><td>0.575</td></tr><tr><td rowspan="5">Virchow</td><td>Single-layer MLP</td><td>0.855</td><td>0.727</td><td>0.573</td><td>0.885</td><td>0.729</td><td>0.579</td><td>0.907</td><td>0.721</td><td>0.557</td></tr><tr><td>3-layer MLP</td><td>0.861</td><td>0.723</td><td>0.574</td><td>0.889</td><td>0.743</td><td>0.620</td><td>0.911</td><td>0.750</td><td>0.616</td></tr><tr><td>Mamba</td><td>0.910</td><td>0.760</td><td>0.577</td><td>0.918</td><td>0.768</td><td>0.623</td><td>0.923</td><td>0.730</td><td>0.608</td></tr><tr><td>Mamba3</td><td>0.906</td><td>0.770</td><td>0.586</td><td>0.915</td><td>0.764</td><td>0.617</td><td>0.919</td><td>0.776</td><td>0.661</td></tr><tr><td>Patch-level attention</td><td>0.899</td><td>0.744</td><td>0.582</td><td>0.917</td><td>0.751</td><td>0.602</td><td>0.920</td><td>0.739</td><td>0.602</td></tr><tr><td rowspan="5">CONCH</td><td>Single-layer MLP</td><td></td><td></td><td></td><td>0.884</td><td>0.711</td><td>0.530</td><td>0.904</td><td>0.723</td><td>0.561</td></tr><tr><td>3-layer MLP</td><td></td><td></td><td></td><td>0.874</td><td>0.707</td><td>0.540</td><td>0.897</td><td>0.738</td><td>0.576</td></tr><tr><td>Mamba</td><td></td><td></td><td></td><td>0.903</td><td>0.715</td><td>0.508</td><td>0.911</td><td>0.735</td><td>0.580</td></tr><tr><td>Mamba3</td><td></td><td></td><td></td><td>0.903</td><td>0.717</td><td>0.536</td><td>0.904</td><td>0.733</td><td>0.577</td></tr><tr><td>Patch-level attention</td><td></td><td></td><td></td><td>0.921</td><td>0.716</td><td>0.481</td><td>0.910</td><td>0.721</td><td>0.503</td></tr></table>