# Whole-Body MRI Classification via Prompt-Based Clinical Conditioning

Laura Daza<sup>1,2[0000−0003−4170−6168]</sup>, Marta Hasny<sup>1,2[0000−0001−5634−1240]</sup>, Cristina González<sup>1,2[0000−0001−9445−9952]</sup>, and Julia A. Schnabel<sup>1,2,3,4[0000−0001−6107−3009]</sup>

<sup>1</sup> Institute of Machine Learning in Biomedical Imaging, Helmholtz Munich, Germany <sup>2</sup> School of Computation, Information and Technology, Technical University of Munich, Germany

3 School of Biomedical Engineering & Imaging Sciences, King’s College London, UK 4 Munich Center for Machine Learning, Germany

Abstract. Combining whole-body magnetic resonance imaging (WB-MRI) with clinical variables has the potential to improve systemic disease diagnosis by leveraging complementary sources of patient information. However, structured clinical variables are often incomplete or missing, limiting the applicability of conventional multimodal fusion methods that assume fixed inputs. In this work, we propose TACTIC (Tabular-Attribute Conditioned Transformer for Image Classification), a promptbased multimodal framework that integrates WB-MRI and structured clinical data through conditional visual feature learning. By encoding clinical attributes as prompts, TACTIC supports an arbitrary number of tabular inputs and naturally handles missing data without requiring imputation or fixed input structures. We evaluate TACTIC on five WB-MRI classification tasks spanning systemic and oncologic applications, including diabetes, chronic obstructive pulmonary disease (COPD), breast cancer, prostate cancer, and metastasis diagnosis. Across all tasks, TACTIC consistently improves performance over image-only baselines when clinical information is available while maintaining strong predictive capability under incomplete tabular inputs. Our results demonstrate the efectiveness of prompt-based models as a flexible approach for improving WB-MRI analysis using clinical context. The model weights and code are available at https://github.com/lauradaza/TACTIC.

Keywords: Whole body MRI · Tabular clinical data · Clinical prompting · Disease classification

## 1 Introduction

Whole-body magnetic resonance imaging (WB-MRI) ofers a comprehensive view of the human body by simultaneously capturing multiple organs and anatomical systems without ionizing radiation, making it particularly attractive for systemic disease assessment and population-scale screening. Recent studies have demonstrated the potential of combining WB-MRI with clinical data for preclinical disease risk assessment [19]. Despite these advantages and its established role in the management of several diseases [22,17,13,15], WB-MRI remains underutilized in automated diagnosis pipelines.

In clinical practice, diagnostic and prognostic decisions are inherently multimodal, relying on imaging findings and structured clinical variables that are not directly observable from imaging data, such as laboratory test results, genetic predispositions, and medical history. Integrating these complementary sources of information is therefore essential to fully leverage imaging data for personalized medicine. Recent large-scale datasets, such as the UK Biobank [21], provide a unique opportunity to study this integration by combining WB-MRI with extensive clinical and demographic data. Early studies have demonstrated the promise of such approaches for medical applications [19,2,3], yet the development of robust multimodal learning strategies remains an active challenge.

A key limitation in current multimodal methods is their assumption of complete and consistently available clinical data. In practice, clinical variables are often missing, sparsely recorded, or heterogeneous across patients, even in wellcurated cohorts. This systematic missingness poses a significant obstacle for conventional fusion techniques, which typically rely on fixed-dimensional tabular inputs [23,20]. Consequently, their performance degrades when the availability of clinical information varies, limiting their applicability in real-world settings.

Recent advances in prompt-based vision models allow flexible conditioning of model behavior through external inputs [9,12], enabling the integration of auxiliary information without requiring a fixed number of prompts. In the medical domain, such approaches provide a promising mechanism for incorporating patient-specific context in an adaptive manner. However, existing approaches predominantly rely on spatial cues [6,25], free-form textual descriptions from reports [26] or translated from structured clinical variables [14,5], as well as prompt-learning strategies over multimodal inputs [7]. As a result, they do not naturally leverage structured clinical data in its native form, often requiring additional preprocessing or modality-specific adaptation.

In this work, we propose TACTIC (Tabular-Attribute Conditioned Transformer for Image Classification), a prompt-based framework designed to leverage WB-MRI with complementary clinical data for disease classification. TACTIC reformulates tabular clinical variables in their native form as prompts that complement image representations, allowing patient-specific information to directly guide visual feature learning. This design enables the model to flexibly incorporate an arbitrary number of clinical variables, making it inherently robust to missing or partially observed data. We evaluate TACTIC on multiple WB-MRI classification tasks spanning systemic and oncologic applications, including diabetes, chronic obstructive pulmonary disease (COPD), breast cancer, prostate cancer, and metastasis diagnosis. Across all tasks, TACTIC consistently improves performance over image-only models when clinical data is available, while maintaining strong predictive capability under limited or missing tabular inputs.

Our contributions are threefold: (i) we introduce TACTIC, a prompt-based multimodal framework that conditions WB-MRI representations on structured clinical attributes; (ii) we reformulate multimodal fusion as a conditioning problem, enabling flexible integration of heterogeneous and incomplete clinical data with pretrained visual backbones; and (iii) we demonstrate the applicability of TACTIC across diverse WB-MRI classification tasks, highlighting the potential of WB-MRI as a foundation for clinically relevant multimodal disease prediction.

![](images/685a4747a985bbdeb04d9279b695fb218679b42f24485bcf8eca262aec09c367.jpg)  
Fig. 1: Overview of TACTIC. The WB-MRI features x are conditioned using random subsets $c _ { i }$ of clinical features as prompts. Combined with an output token $T _ { \mathrm { o u t } }$ , the prompts guide image feature aggregation through cross-attention, enabling flexible multimodal fusion under variable clinical data availability. WB-MRI reproduced by kind permission of UK Biobank ©

## 2 Method

## 2.1 Tabular-Attribute Conditioned Transformer

As shown in Figure 1, our method models structured clinical variables as prompts that condition the image classification process, enabling patient-specific adaptation of visual representations. We adopt Primus-M [24] as the image encoder and use TARTE [8] to embed structured clinical information. Specifically, TARTE generates an embedding for each clinical attribute by jointly encoding the attribute name (e.g., sex, age) and its corresponding value (e.g., female, 64 ).

The core of our approach is a prompt-conditioned transformer that integrates clinical and visual information. TACTIC first performs self-attention over the clinical prompts to capture global dependencies among tabular attributes. These refined prompts then interact with image features through cross-attention, guiding the aggregation of visual information in a patient-specific manner. By treating clinical context as prompts rather than as additional input modalities, TACTIC enables a simple integration with pretrained visual backbones, while injecting clinically relevant information for accurate disease classification.

## 2.2 Single Modality downstream training

Image Encoder. We pretrain Primus using a masked autoencoding (MAE) [4] framework adapted for WB-MRI. Since more than 40% of the voxels in WB-MRI scans are outside the body, we first generate body masks using otsu thresholding [16] and binary morphology to identify foreground regions. These masks are used in two ways. First, we apply a mask-aware pooling operation that merges voxels in windows of size $[ 2 \times 2 \times 4 ]$ while ignoring positions outside the body, thereby reducing the image size with minimal information loss. Second, even after pooling, a substantial fraction of the resulting tokens still correspond primarily to background voxels, biasing the standard MAE reconstruction loss toward trivial background patches. Therefore, we restrict masking and reconstruction to patches containing anatomical information, ensuring that the pretraining objective focuses on relevant content and promotes the learning of meaningful image representations.

Tabular Encoder. We train TARTE by explicitly simulating missing clinical data, following [3]. Let $C ~ = ~ c ^ { ( 1 ) } , \ldots , { \bar { c } } ^ { ( K ) }$ denote the full set of tabular attributes. During training, we sample a subset $C _ { \mathrm { m o r e } } \subseteq C$ and a second subset $C _ { \mathrm { f e w } } \subset C _ { \mathrm { m o r e } } .$ , representing scenarios with more and fewer available clinical attributes, respectively. Predictions obtained from both subsets are used to compute the Tabular More vs. Fewer (TabMoFe) loss [3], which encourages improved performance as additional clinical information becomes available:

$$
\mathcal { L } _ { \mathrm { T a b M o F e } } = \operatorname* { m a x } \mathopen { } \mathclose \bgroup \left( \mathcal { L } _ { \mathrm { m o r e } } - \mathcal { L } _ { \mathrm { f e w } } , 0 \aftergroup \egroup \right)\tag{1}
$$

The final tabular loss is given by ${ \mathcal { L } } _ { \mathrm { t a b } } = { \mathcal { L } } _ { \mathrm { T a b M o F e } } + { \mathcal { L } } _ { \mathrm { m o r e } } + { \mathcal { L } } _ { \mathrm { f e w } }$ , where $\mathcal { L } _ { \mathrm { m o r e } }$ and ${ \mathcal { L } } _ { \mathrm { f e w } }$ denote the cross entropy losses obtained using $C _ { \mathrm { m o r e } }$ and $C _ { \mathrm { f e w } }$

## 2.3 Clinical Prompting Training

Given paired imaging and tabular clinical data (X, C) with diagnosis label $y ,$ we first extract unimodal representations using their corresponding encoders. The image encoder produces image features $\boldsymbol { x } \in \mathbb { R } ^ { n \times d }$ , while the tabular encoder produces clinical embeddings $\bar { c } \in \mathbb { R } ^ { K \times d }$ . To simulate variable clinical data availability, we sample N random subsets of clinical attributes $\{ c _ { 1 } , c _ { 2 } , \ldots , c _ { N } \}$ , where each subset $c _ { i } \subseteq C$ contains k attributes. Each subset is concatenated with an output token $T _ { \mathrm { o u t } }$ and treated as a set of prompts that, together with the image features x, condition the model’s output. A multimodal prediction is obtained for each subset, and the multimodal loss ${ \mathcal { L } } _ { \mathrm { m m } }$ is computed using the average prediction across all sampled subsets.

In parallel, we compute an image-only loss ${ \mathcal { L } } _ { \mathrm { i m g } }$ by conditioning the model with the output token $T _ { \mathrm { o u t } }$ and an empty clinical prompt set ∅. The model is trained using the More vs. Fewer (MoFe) loss [10], defined as

$$
\mathcal { L } _ { \mathrm { M o F e } } = \operatorname* { m a x } \left( \mathcal { L } _ { \mathrm { m m } } - \mathcal { L } _ { \mathrm { i m g } } , 0 \right)\tag{2}
$$

Table 1: Training subset sizes per disease. The sex distribution of the patients is maintained between the healthy and diagnosed subjects.
<table><tr><td></td><td>Diabetes</td><td>COPD</td><td>Breast cancer</td><td>Prostate cancer</td><td>Metastasis</td></tr><tr><td>Subset size</td><td>4060</td><td>1292</td><td>3092</td><td>2682</td><td>2885</td></tr><tr><td>% Female</td><td>35.5%</td><td>45.2%</td><td>99.4%</td><td>0.0%</td><td>88.7%</td></tr></table>

where ${ \mathcal { L } } _ { \mathrm { m m } } = { \mathcal { L } } _ { C E } ( ( X , C ) , y ) , { \mathcal { L } } _ { \mathrm { i m g } } = { \mathcal { L } } _ { C E } ( ( X , \emptyset ) , y )$ , and $\mathcal { L } _ { C E }$ denotes the cross-entropy loss. The final training objective is $\mathcal { L } = \mathcal { L } _ { \mathrm { M o F e } } + \mathcal { L } _ { \mathrm { m m } } + \mathcal { L } _ { \mathrm { i m g } }$

## 3 Experimental Setting

## 3.1 Dataset

The UK Biobank [21] contains over 500,000 participants, with a subset of 80,000 that includes WB-MRI data and corresponding clinical information from in hospital inpatient records. WB-MRI are neck-to-knee T1-weighted dual-echo Dixon acquisitions with a spatial resolution of $[ 2 . 2 3 \times 2 . 2 3 \times 3 . 0 ]$ mm and size $[ 2 2 4 \times 1 6 8 \times 3 6 3 ]$ . For all experiments, we use the fat-only and water-only image reconstructions. In addition to imaging, we extract 235 clinical attributes following [11], including demographic information, medical and family history, blood and urine biomarkers, polygenic risk scores, lifestyle factors, physical measurements, and sex-specific variables.

For disease classification we use labels derived from the International Classification of Diseases (ICD-10) codes. We evaluate our method on five classification tasks: (1) diabetes, (2) chronic obstructive pulmonary disease (COPD), (3) breast cancer, (4) prostate cancer, and (5) metastasis in our cancer cohorts. We include participants with imaging and clinical information and split them into training, validation, and test sets using a 70–10–20 ratio. As all target conditions are under-represented, we construct balanced training subsets for all downstream tasks. For these subsets, participants without the target condition are randomly sampled ensuring the preservation of the sex distribution. For metastasis classification, we use our full cancer cohort, which has a prevalence of 11% metastatic cases. The resulting subset sizes are reported in Table 1.

## 3.2 Implementation Details

We pretrain Primus using all training WB-MRI scans of the UK Biobank. The pretraining is done for 50 epochs using input images of size $[ 2 2 4 \times 1 6 0 \times 3 5 2 ]$ masking ratio of 90%, learning rate of $8 \times 1 0 ^ { - 4 }$ , and two warm-up epochs followed by cosine annealing. During fine-tuning, we use the same architecture and training configuration for all tasks. For tabular fine-tuning, we use a learning rate of $1 \times 1 0 ^ { - 5 }$ , batch size of 128, and train for 100 epochs. Imaging and multimodal tuning are done with an image masking rate of 40% to eliminate background tokens and reduce computational costs, batch size of 48 and gradient accumulation over 6 steps, resulting in an efective batch of 288. We initialize TACTIC from the fine-tuned unimodal model weights and optimize until convergence with TARTE frozen. All experiments are performed in one NVIDIA H100 GPU with 80Gb of RAM.

## 4 Results

We compare TACTIC against single- and multimodal baselines. The singlemodality models are the vision-only Primus and the tabular-only TARTE. For multimodal fusion, we evaluate Max Fuse [23], Concat Fuse [20], Sum Fuse, DAFT [18], Dynamic Mixture of Modality Experts (DMoMe) [10], and TIP [2] under varying levels of clinical data missingness. Performance is measured using the area under the receiver operating characteristic curve (AUROC), reported as the mean across ten levels of tabular data availability from 10% to 100%.

All multimodal methods are initialized from the same pretrained image and tabular encoders as TACTIC and trained for an identical number of epochs to ensure a fair comparison. For Max Fuse and Concat Fuse, we follow the original formulations and apply a linear classifier to the fused embeddings. We further include a Sum Fuse baseline that combines modality embeddings via elementwise addition. For DAFT, a DAFT block is inserted before the final transformer block of the image backbone. For DMoMe, we adopt the mixture-of-experts variant proposed for UPMC Food-101 [1], using intermediate feature representations for gating to mitigate the computational cost of processing raw WB-MRI data. Finally, the substantially larger size of WB-MRI volumes compared with the inputs originally used in TIP makes its pretraining stage computationally impractical. We therefore adopt the original downstream fine-tuning framework developed for cardiac imaging and replace only the unimodal backbone architectures with our image and tabular encoders.

## 4.1 Performance Comparison

Table 2 shows that TACTIC consistently improves upon the image-only baseline across all classification tasks and achieves the best overall performance for four out of five diseases. For diabetes, clinical variables are highly predictive on their own, achieving an AUROC of 87.1. Consequently, all multimodal methods substantially outperform the image-only baseline. TACTIC matches the bestperforming fusion approach, demonstrating that conditioning image representations with clinical prompts can efectively exploit the complementary information present in both modalities. For COPD and breast cancer, the performance gap between the image- and tabular-only models is smaller, indicating that neither modality alone is suficient for optimal prediction. In these settings, conventional fusion approaches often fail to efectively combine both sources of information and may even degrade performance relative to the tabular-only baseline. In contrast, TACTIC achieves the highest AUROC for both tasks, surpassing both single modality models and highlighting its ability to capture complementary information from heterogeneous modalities.

Table 2: Performance of TACTIC compared to multiple methods on 5 classification tasks in the UK Biobank. We report the mean AUROC and the improvement over Primus under varying levels of tabular data availability (10% - 100%)
<table><tr><td>Method</td><td>Diabetes</td><td>COPD</td><td>Breast cancer</td><td>Prostate cancer</td><td>Metastasis</td></tr><tr><td>Single-modality</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Primus (img) [24]</td><td>78.8</td><td>74.0</td><td>58.0</td><td>49.0</td><td>66.1</td></tr><tr><td>TARTE (tab) [8]</td><td>87.1</td><td>79.3</td><td>62.8</td><td>78.2</td><td>64.7</td></tr><tr><td>Multimodal fusion</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Concat Fuse [20]</td><td> $8 9 . 1 \mathrm { ~ + 1 0 . 3 }$ </td><td> $8 0 . 1 \ + 6 . 1$ </td><td> $6 1 . 2 \ + 3 . 2$ </td><td> $7 7 . 8 \ + 2 8 . 8 $ </td><td> $6 6 . 0 \textrm { - } 0 . 2 $ </td></tr><tr><td>Max Fuse [23]</td><td> $8 9 . 0 \ \mathrm { + 1 0 . 2 }$ </td><td> $7 7 . 8 \ + 3 . 8 $ </td><td> $5 8 . 7 \ + 0 . 7 $ </td><td> $7 5 . 2 \ : + 2 6 . 2$ </td><td> $6 7 . 1 \ + 1 . 0$ </td></tr><tr><td>Sum Fuse</td><td> $8 8 . 7 \textrm { + } 9 . 9 $ </td><td> $7 9 . 2 \ + 5 . 2$ </td><td> $5 9 . 5 \ + 1 . 5$ </td><td> $7 7 . 4 \ + 2 8 . 5$ </td><td> $6 6 . 7 \ + 0 . 5 $ </td></tr><tr><td>DAFT [18]</td><td> $7 8 . 9 \textrm { + } 0 . 1$ </td><td> $6 6 . 6 \textrm { -- } 7 . 4$ </td><td> $4 5 . 9 \textrm { - } 1 2 . 1$ </td><td> $5 2 . 3 \textrm { + } 3 . 4$ </td><td> $6 4 . 8 \textrm { - } 1 . 3$ </td></tr><tr><td>DMoMe [10]</td><td> $8 9 . 0 \ \mathrm { + 1 0 . 2 }$ </td><td> $7 6 . 3 \ : + 2 . 3 \ :$ </td><td> $6 1 . 0 \ \div 3 . 0 $ </td><td> $7 4 . 1 \ + 2 5 . 2$ </td><td> $6 7 . 6 \ \div 1 . 5$ </td></tr><tr><td>TIP [2]</td><td> $8 2 . 8 \textrm { + } 4 . 0$ </td><td> $7 6 . 5 \ + 2 . 5$ </td><td> $6 1 . 0 \ \div 3 . 0 $ </td><td> $7 3 . 2 \ + 2 4 . 2$ </td><td> $6 2 . 7 \textrm { - } 3 . 5 $ </td></tr><tr><td>TACTIC (ours)</td><td> ${ \bf 8 9 . 1 \_ 1 0 . 3 }$ </td><td> ${ \bf 8 3 . 3 \_ 9 . 3 }$ </td><td> ${ \bf 6 3 . 9 \mathrm { ~ + 5 . 9 } }$ </td><td> $7 6 . 6 \ : + 2 7 . 7$ </td><td> ${ \bf 6 9 . 2 \mathrm { ~ + 3 . 1 ~ } }$ </td></tr></table>

Prostate cancer shows a diferent pattern, with substantially lower performance for the image-only model than for the tabular baseline, suggesting that WB-MRI provides limited discriminative information for this task. As a result, most multimodal fusion methods highly benefit from incorporating clinical data, with Concat Fuse archiving the best performance. While TACTIC also benefits from clinical conditioning, we start from image representations and inject clinical context through prompting, which leads to smaller performance gains in scenarios where imaging contributes minimal task-relevant information. For metastasis detection, the image-only model outperforms the tabular-only baseline, indicating that WB-MRI is the main source of predictive information. In this scenario, some fusion methods fail to improve upon the image-only baseline, suggesting that in this task the clinical variables can introduce noise. In contrast, TACTIC achieves the largest improvement over the image-only model, demonstrating that prompt-based conditioning can efectively leverage complementary patient information without compromising the visual representations.

## 4.2 Incomplete Clinical Data

Figure 2 reports model performance under varying levels of clinical data availability for COPD. We progressively increase the fraction of visible clinical attributes using three sampling strategies: random selection, ascending, and descending order of attribute importance. Attribute importance is estimated from TARTE’s attention weights over the clinical variables on the training set. Across all scenarios, TACTIC consistently achieves the highest performance and shows the strongest robustness to incomplete clinical information.

![](images/17e1cdf209f20ae1eb16a255546daa1c0b7db748c75187671b981d5319c3620d.jpg)  
Fig. 2: Classification performance under incomplete clinical data for COPD. Results are reported as a function of the fraction of available clinical attributes, sampled randomly, in ascending and in descending order of attribute importance

Under random attribute selection, TACTIC outperforms all multimodal approaches over the entire range of attribute availability, even when only a small subset of clinical variables is provided. As more attributes become available, its performance steadily improves, indicating efective integration of complementary information from both modalities. When only low-importance attributes are available, most fusion methods experience a performance drop until a large fraction of variables becomes available. In contrast, TACTIC is always better than the image-only starting point. When we start from the high-importance attributes, all multimodal methods benefit from the initial tabular data. However, as less informative information is introduced, the performance of most methods degrades, while TACTIC consistently improves. Together, these results suggest that our prompt-based approach can leverage all the available clinical information without afecting the visual representations with noisy or less informative clinical data. Overall, our results demonstrate that prompt-based conditioning provides a reliable mechanism for integrating clinical information.

## 5 Conclusion

In this work we introduce TACTIC, a prompt-based multimodal framework that complements WB-MRI with structured clinical variables for disease classification. By conditioning image representations on clinical attributes prompts, our method provides a flexible mechanism to incorporate patient-specific context without relying on fixed or complete tabular inputs. This design directly addresses a common limitation of conventional multimodal fusion strategies, which often degrade performance when clinical data is incomplete or fully missing. Across a range of systemic and oncological tasks, we showed that our model consistently improves upon image-only models when clinical information is available, and remains efective even when only a limited subset of clinical variables is available. Compared to previous fusion approaches, TACTIC demonstrates greater robustness under incomplete clinical data settings and more stable performance as the amount of available tabular information varies. These results suggest that conditioning visual representations via prompting enables more efective exploitation of complementary clinical cues than direct feature aggregation. Overall, our findings highlight the potential of flexible multimodal frameworks to enhance WB-MRI analysis in clinically realistic settings, particularly in scenarios where clinical data availability is inherently heterogeneous.

Acknowledgments. This research has been conducted using the UK Biobank Resource under Application Number 87065. LD and MH are supported by the German Federal Ministry of Research, Technology and Space (DECIPHER-M, 01KD2420G). MH is in part supported by the Munich School of Data Science (MUDS) and the European Laboratory for Learning and Intelligent Systems (ELLIS) PhD program.

Disclosure of Interests. The authors have no competing interests to declare that are relevant to the content of this article.

## References

1. Allegra, D., Anthimopoulos, M., Dehais, J., Lu, Y., Stanco, F., Farinella, G.M., Mougiakakou, S.: A Multimedia Database for Automatic Meal Assessment Systems. In: International Conference on Image Analysis and Processing. pp. 471–478. Springer (2017)

2. Du, S., Zheng, S., Wang, Y., Bai, W., O’Regan, D.P., Qin, C.: TIP: Tabular-Image Pre-training for Multimodal Classification with Incomplete Data. In: European Conference on Computer Vision. pp. 478–496. Springer (2024)

3. Hasny, M., Daza, L., Bressem, K., Di Folco, M., Schnabel, J.: No Data? No Problem: Robust Vision-Tabular Learning with Missing Values. International Conference on Machine Learning (2026)

4. He, K., Chen, X., Xie, S., Li, Y., Dollár, P., Girshick, R.: Masked Autoencoders are Scalable Vision Learners. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 16000–16009 (2022)

5. Huang, Y., Zhang, W., Huang, P., Fu, Y., Yang, R., Yu, L.: Bridging radiological images and factors with vision-language model for accurate diagnosis of proliferative hepatocellular carcinoma. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 35–45. Springer (2025)

6. Isensee, F., Rokuss, M., Krämer, L., Dinkelacker, S., Ravindran, A., Stritzke, F., Hamm, B., Wald, T., Langenberg, M., Ulrich, C., et al.: nnInteractive: Redefining 3D Promptable Segmentation. arXiv preprint arXiv:2503.08373 (2025)

7. Kang, L., Gong, H., Wan, X., Li, H.: Visual-attribute prompt learning for progressive mild cognitive impairment prediction. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 547–557. Springer (2023)

8. Kim, M.J., Lefebvre, F., Brison, G., Perez-Lebel, A., Varoquaux, G.: Table Foundation Models: on knowledge pre-training for tabular learning. Transactions on Machine Learning Research (2025)

9. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., et al.: Segment Anything. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 4015–4026 (2023)

10. Li, S., Chen, C., Han, J.: SimMLM: A Simple Framework for Multi-modal Learning with Missing Modality. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 24068–24077 (2025)

11. Lübeck, F., Wildberger, J., Träuble, F., Mordig, M., Gatidis, S., Krause, A., Schölkopf, B.: Adaptable Cardiovascular Disease Risk Prediction from Heterogeneous Data using Large Language Models. arXiv preprint arXiv:2505.24655 (2025)

12. Ma, J., He, Y., Li, F., Han, L., You, C., Wang, B.: Segment anything in medical images. Nature Communications 15(1), 654 (2024)

13. Messiou, C., Hillengass, J., Delorme, S., Lecouvet, F.E., Moulopoulos, L.A., Collins, D.J., Blackledge, M.D., Abildgaard, N., Østergaard, B., Schlemmer, H.P., et al.: Guidelines for Acquisition, Interpretation, and Reporting of Whole-Body MRI in Myeloma: Myeloma Response Assessment and Diagnosis System (MY-RADS). Radiology 291(1), 5–13 (2019)

14. Milecki, L., Kalogeiton, V., Bodard, S., Anglicheau, D., Correas, J.M., Timsit, M.O., Vakalopoulou, M.: MEDIMP: 3D Medical Images with clinical Prompts from limited tabular data for renal transplantation. Medical Imaging with Deep Learning (2024)

15. Mottet, N., Bellmunt, J., Bolla, M., Briers, E., Cumberbatch, M.G., De Santis, M., Fossati, N., Gross, T., Henry, A.M., Joniau, S., et al.: EAU-ESTRO-SIOG Guidelines on Prostate Cancer. Part 1: Screening, Diagnosis, and Local Treatment with Curative Intent. European Urology 71(4), 618–629 (2017)

16. Otsu, N.: A threshold selection method from gray-level histograms. IEEE transactions on systems, man, and cybernetics 9(1), 62–66 (1979)

17. Pflugfelder, A., Kochs, C., Blum, A., Capellaro, M., Czeschik, C., Dettenborn, T., Dill, D., Dippel, E., Eigentler, T., Feyer, P., et al.: Malignant melanoma S3- guideline "diagnosis, therapy and follow-up of melanoma". JDDG: Journal der Deutschen Dermatologischen Gesellschaft 11(s6), 1–116 (2013)

18. Pölsterl, S., Wolf, T.N., Wachinger, C.: Combining 3D Image and Tabular Data via the Dynamic Afine Feature Map Transform. In: International Conference on Medical Image Computing and Computer Assisted Intervention. pp. 688–698 (2021)

19. Seletkov, D., Starck, S., Mueller, T.T., Zhang, Y., Steinhelfer, L., Rueckert, D., Braren, R.: AI-driven preclinical disease risk assessment using imaging in UK biobank. NPJ Digital Medicine 8(1), 480 (2025)

20. Spasov, S., Passamonti, L., Duggento, A., Lio, P., Toschi, N., Initiative, A.D.N., et al.: A parameter-eficient deep learning approach to predict conversion from mild cognitive impairment to Alzheimer’s disease. Neuroimage 189, 276–287 (2019)

21. Sudlow, C., Gallacher, J., Allen, N., Beral, V., Burton, P., Danesh, J., Downey, P., Elliott, P., Green, J., Landray, M., et al.: UK Biobank: An Open Access Resource for Identifying the Causes of a Wide Range of Complex Diseases of Middle and Old Age. PLoS Medicine 12(3), e1001779 (2015)

22. Summers, P., Saia, G., Colombo, A., Pricolo, P., Zugni, F., Alessi, S., Marvaso, G., Jereczek-Fossa, B.A., Bellomi, M., Petralia, G.: Whole-body magnetic resonance imaging: technique, guidelines and key applications. Ecancer Medical Science 15, 1164 (2021)

23. Vale-Silva, L.A., Rohr, K.: Long-term cancer survival prediction using multimodal deep learning. Scientific Reports 11(1), 13505 (2021)

24. Wald, T., Roy, S., Isensee, F., Ulrich, C., Ziegler, S., Trofimova, D., Stock, R., Baumgartner, M., Köhler, G., Maier-Hein, K.: Primus: Enforcing Attention Usage for 3D Medical Image Segmentation. arXiv preprint arXiv:2503.01835 (2025)

25. Wang, H., Guo, S., Ye, J., Deng, Z., Cheng, J., Li, T., Chen, J., Su, Y., Huang, Z., Shen, Y., et al.: SAM-Med3D: Towards General-Purpose Segmentation Models for Volumetric Medical Images. IEEE Transactions on Neural Networks and Learning Systems (2025)

26. Zhang, S., Xu, Y., Usuyama, N., Xu, H., Bagga, J., Tinn, R., Preston, S., Rao, R., Wei, M., Valluri, N., et al.: A Multimodal Biomedical Foundation Model Trained from Fifteen Million Image–Text Pairs. Nejm AI 2(1), AIoa2400640 (2025)