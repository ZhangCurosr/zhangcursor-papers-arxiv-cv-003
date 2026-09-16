# Which Pretext Task Transfers? Self-Supervised Pretraining Objectives for Lung Ultrasound

Moein Heidari<sup>a</sup>, Junbo Rao<sup>a</sup>, Jai Choraria<sup>a</sup>, Wenjin Chen<sup>b</sup>, David J. Foran<sup>b</sup>, and Ilker Hacihaliloglu<sup>a</sup>

<sup>a</sup>University of British Columbia, Vancouver, BC, Canada <sup>b</sup>Rutgers Cancer Institute of New Jersey, New Brunswick, NJ, USA

## ABSTRACT

Self-supervised learning (SSL) can reduce the need for labelled medical images, but the choice of pretext objective remains unclear for lung ultrasound (LUS). Contrastive learning, masked reconstruction, and joint-embedding predictive architectures (JEPA) difer in the space in which their targets are defined, yet existing ultrasound studies compare them under diferent corpora, backbones, and evaluation protocols. We compare these three objective families using the same encoder backbone, pretraining corpus, optimisation schedule, and frozenevaluation protocol. Encoders are pretrained on COVID-BLUeS LUS videos and evaluated with linear, kNN, and attentive probes at 5%, 10%, 50%, and 100% label budgets. Evaluation is performed on POCUS using patient-level five-fold cross-validation and on the independently acquired Mendeley-Uganda dataset, which is excluded from both pretraining and probe fitting. At the full label budget under linear probing, VideoMAE and V-JEPA achieve 66.5 ± 13.1 and 65.4 ± 11.7 balanced accuracy on POCUS, while MoCo achieves 42.1 ± 1.2. On Mendeley-Uganda, the ranking reverses: MoCo performs best at 62.7 ± 1.0, followed by VideoMAE at 53.8 ± 2.8, while V-JEPA falls near chance at 35.1 ± 4.9. These results show that POCUS probe accuracy alone does not identify the objective that transfers best across datasets. We also outline planned representation-level analyses to examine this reversal. Code is publicly available at https://github.com/moeinheidari7829/LUSVideoSSL.

Keywords: Lung Ultrasound, Self-supervised Learning, Representation Learning

## 1. INTRODUCTION

Lung ultrasound (LUS) has become a first-line modality for bedside assessment of pulmonary pathology, ofering real-time acquisition without ionising radiation. Diagnosis is driven by a compact set of sonographic patterns: A-lines indicate normally aerated parenchyma, B-lines indicate interstitial syndrome, and consolidation and pleural efusion manifest as changes in tissue texture.<sup>1,</sup> <sup>2</sup> A subset of these patterns is inherently dynamic, most notably lung sliding, whose absence is diagnostic of pneumothorax and which is observable only across frames. Interpretation consequently depends on both fine spatial detail and temporal behaviour, and remains strongly operator-dependent. Expert annotation is costly, and publicly available labelled LUS collections are orders of magnitude smaller than the corpora used to train contemporary vision models.

Self-supervised learning (SSL) addresses this constraint by learning representations from unlabelled data, and three families dominate current practice. Contrastive methods enforce invariance between augmented views of the same sample under an InfoNCE objective.<sup>3,</sup> <sup>4</sup> Masked autoencoders reconstruct withheld pixels from a sparse visible context.<sup>5,</sup> <sup>6</sup> Joint-embedding predictive architectures (JEPA) instead regress the representation of a withheld region from visible context, so that the prediction target is a learned feature rather than an observed pixel.<sup>7,</sup> <sup>8</sup> These objectives are not interchangeable: they impose distinct inductive biases on what the encoder is permitted to discard. Pixel-space reconstruction allocates capacity to input variance irrespective of its semantic content, a known liability in modalities dominated by speckle,<sup>9</sup> and latent prediction suppresses unpredictable components at the cost of learning its own target, a process with documented sample-complexity requirements.<sup>10,</sup> <sup>11</sup> Ultrasound foundation models have adopted these objectives in isolation. USCL pretrains a contrastive backbone from ultrasound video and reports strong fine-tuning accuracy on POCUS.<sup>12</sup> USFM and

USF-MAE scale masked image modelling to large multi-organ corpora.<sup>13,</sup> <sup>14</sup> More recent work transfers latent prediction to ultrasound video and echocardiography.<sup>15–17</sup> However, as these systems difer simultaneously in corpus, architecture and evaluation, their published numbers do not support inference about the relative merit of the underlying objectives.

Comparative studies have begun to address this. Ivezi´c et al.<sup>18</sup> contrast MAE,<sup>5</sup> DINOv3<sup>19</sup> and I-JEPA<sup>7</sup> on ultrasound and histopathology and conclude that the preferred pretext task is determined by whether a diagnostically relevant signal is spatially localised or globally structured. Complementary evidence indicates that in-domain SSL pretraining can yield large in-domain gains while degrading substantially elsewhere,<sup>20</sup> and that SSL robustness under distributional change is weaker than natural-image benchmarks suggest.<sup>21</sup> Three gaps remain for LUS: existing comparisons operate on static frames rather than temporal video, pretrain on large multi-organ corpora rather than a modest in-domain dataset from a single clinical application, and do not test transfer to an independently acquired LUS dataset.

This work addresses these gaps directly. Our contributions are threefold. ❶ We conduct a controlled comparison of contrastive, masked, and latent prediction pretraining for LUS video, while holding the backbone, pretraining corpus, optimisation schedule, and frozen evaluation protocol fixed, so that measured diferences can be attributed to the pretext objective. ❷ We pretrain on COVID-BLUeS and evaluate label eficiency at four annotation budgets on POCUS using patient-level five-fold cross-validation and on the independently acquired Mendeley-Uganda dataset. ❸ At the full label budget under linear probing, the ranking on POCUS reverses on Mendeley-Uganda: contrastive pretraining is weakest on POCUS but strongest on Mendeley-Uganda, while latent prediction performs well on POCUS and drops to near-chance performance on Mendeley-Uganda. These results show that POCUS probe accuracy alone does not predict cross-dataset transfer.

## 2. METHOD

## 2.1 Pretext objectives

Let x denote an LUS clip and $f _ { \theta }$ the encoder under study. The paradigms difer only in the pretext task imposed on $f _ { \theta }$ , and in particular in the space in which the loss is evaluated (Fig. 1).

Contrastive. Two temporally ofset, augmented clips $v _ { 1 }$ , v sampled from x are encoded by $f _ { \theta }$ and a momentum encoder $f _ { \xi }$ , projected onto a hypersphere, and optimised with InfoNCE so that clips from the same video are aligned and clips from distinct videos are separated. We instantiate this family with the MoCo v3 objective<sup>4</sup> adapted to video following the spatiotemporal contrastive recipe of VideoMoCo.<sup>22</sup>

Masked reconstruction. Tube masking removes a fraction $\rho = 9 0 \%$ of spatio-temporal tubelets. The encoder processes only visible tubelets and a lightweight decoder, discarded after pretraining, regresses the withheld pixels under a mean squared error criterion. We instantiate this family with VideoMAE.<sup>6</sup>

Latent prediction. The clip is partitioned into a visible context and masked target blocks. A predictor $g _ { \varphi }$ maps the context representation to an estimate of the target representation, where targets are produced by an exponential moving average encoder $f _ { \xi }$ under a stop-gradient, and the loss is an $L _ { 1 }$ distance in feature space. We instantiate this family with V-JEPA.<sup>8,</sup> <sup>23</sup>

## 2.2 Controlled pretraining

All paradigms employ a ViT-S/16 backbone with tubelet size 2 over 16-frame clips, are initialised randomly without ImageNet weights, and share optimiser, learning-rate schedule and epoch budget. Pretraining uses the LUS videos from the COVID-BLUeS dataset.<sup>24</sup> We use this modest-scale dataset to compare the three objectives under a pretraining budget representative of a single clinical application.

## 2.3 Frozen evaluation

Pretrained encoders are frozen and evaluated with a linear classifier, a kNN classifier and an attentive probe, each trained at label budgets of 5%, 10%, 50% and 100%. The downstream task is three-class classification on POCUS<sup>1</sup> (COVID-19, bacterial pneumonia and healthy) under patient-level five-fold cross-validation. For each fold, the probe is trained on four POCUS folds and evaluated on both the held-out POCUS fold and the full Mendeley-Uganda dataset.<sup>25</sup> Mendeley-Uganda is excluded from both SSL pretraining and probe fitting and was acquired under a diferent imaging distribution. We refer to POCUS as in-distribution with respect to probe training, while Mendeley-Uganda serves as the external evaluation dataset. Balanced accuracy is reported as the mean ± standard deviation over the five POCUS folds and the five corresponding Mendeley-Uganda evaluations. Patient-level partitioning is essential because temporally adjacent LUS frames are near-duplicates, and frame-level partitioning can inflate measured performance.

![](images/6828a245ac969f6fd3f725da68cfbb5167d42e9d34d515b99c3a8c22d19ebf0b.jpg)  
Figure 1. The three pretext objectives applied to the same lung ultrasound clip. The objectives difer in the space in which the loss is evaluated: contrastive learning compares whole-clip embeddings on a hypersphere, masked reconstruction compares images, and latent prediction compares feature tokens.

Table 1. Balanced accuracy (%) on POCUS and Mendeley-Uganda. Results are reported as mean (standard deviation) over five evaluations. Each POCUS evaluation uses one patient-level test fold. Each Mendeley-Uganda evaluation uses the full external dataset and the probe trained on the corresponding four POCUS training folds. The best value in each column is shown in bold.
<table><tr><td rowspan="2">Pretraining</td><td rowspan="2">Probe</td><td colspan="4">POCUS</td><td colspan="4">Mendeley-Uganda</td></tr><tr><td>5%</td><td>10%</td><td>50%</td><td>100%</td><td>5%</td><td>10%</td><td>50%</td><td>100%</td></tr><tr><td rowspan="3">MoCo v3-S</td><td>linear</td><td>45.14.5</td><td>41.31.8</td><td>41.23.2</td><td>42.11.2</td><td>61.32.6</td><td>62.61.8</td><td>62.31.1</td><td>62.71.0</td></tr><tr><td>kNN</td><td>33.3 0.0</td><td>33.912.4</td><td>44.67.2</td><td>37.20.0</td><td>57.92.8</td><td>57.51.8</td><td>57.02.5</td><td>57.60.0</td></tr><tr><td>attentive</td><td>45.66.8</td><td>39.23.5</td><td>42.11.3</td><td>41.10.9</td><td>63.01.6</td><td>64.20.9</td><td>64.40.4</td><td>64.40.2</td></tr><tr><td rowspan="3">VideoMAE-S</td><td>linear</td><td>36.517.2</td><td>39.77.6</td><td>62.912.5</td><td>66.513.1</td><td>42.56.3</td><td>45.99.1</td><td>51.44.1</td><td>53.82.8</td></tr><tr><td>kNN</td><td>38.612.7</td><td>42.17.7</td><td>51.34.9</td><td>60.1 6.4</td><td>38.67.8</td><td>38.7 5.4</td><td>39.98.0</td><td>44.21.6</td></tr><tr><td>attentive</td><td>34.715.7</td><td>47.88.8</td><td>62.117.6</td><td>60.910.6</td><td>40.811.2</td><td>45.89.0</td><td>51.44.6</td><td>48.23.6</td></tr><tr><td rowspan="3">V-JEPA-S</td><td>linear</td><td>42.513.8</td><td>37.523.6</td><td>64.411.1</td><td>65.411.7</td><td>35.2 9.1</td><td>30.310.8</td><td>35.47.0</td><td>35.14.9</td></tr><tr><td>kNN</td><td>39.912.6</td><td>42.515.0</td><td>53.610.9</td><td>54.013.5</td><td>33.7 6.8</td><td>34.17.5</td><td>37.76.4</td><td>37.72.3</td></tr><tr><td>attentive</td><td>42.116.4</td><td>41.314.5</td><td>49.410.1</td><td>61.811.5</td><td>34.5 9.0</td><td>40.110.2</td><td>40.96.4</td><td>37.3 4.3</td></tr></table>

## 3. PRELIMINARY RESULTS

Table 1 and Fig. 2 report balanced accuracy for the three pretraining objectives. At the full label budget under linear probing, VideoMAE and V-JEPA achieve 66.5 ± 13.1 and 65.4 ± 11.7 on POCUS, while MoCo achieves $4 2 . 1 \pm 1 . 2 $ . On Mendeley-Uganda, MoCo performs best at $6 2 . 7 \pm 1 . 0 $ , followed by VideoMAE at $5 3 . 8 \pm 2 . 8$ and V-JEPA at $3 5 . 1 \pm 4 . 9$ . From POCUS to Mendeley-Uganda, MoCo increases by 20.6 points, while VideoMAE and V-JEPA decrease by 12.7 and 30.3 points, respectively (Fig. 2c). The ranking therefore changes from VideoMAE ≈ V-JEPA > MoCo on POCUS to MoCo > VideoMAE > V-JEPA on Mendeley-Uganda. Under linear probing, MoCo also shows the lowest variability on Mendeley-Uganda at the higher label budgets, with standard deviations of 1.1 and 1.0 points at the 50% and 100% label budgets, respectively.

Interpretation. The results suggest a trade-of between performance on the POCUS task and transfer to a diferent acquisition setting. V-JEPA performs well on POCUS but drops to near-chance performance on Mendeley-Uganda. This may be related to the sample-complexity requirements of JEPA-style objectives. Recent ultrasound work has also used static teachers to stabilise JEPA training.<sup>17</sup> MoCo performs poorly on POCUS but transfers best to Mendeley-Uganda, possibly because its augmentation-based objective encourages invariance across acquisition conditions. VideoMAE falls between the two methods. These results may depend on the size of the pretraining dataset and the distribution used for evaluation.

![](images/a1612dc795bb2f13c68f56390ac9691bc640c63d485e9ce29f40c641319209ef.jpg)  
Figure 2. Label eficiency and cross-dataset performance. (a, b) Balanced accuracy across four label budgets using linear, attentive, and kNN probes on POCUS and Mendeley-Uganda. On POCUS, VideoMAE and V-JEPA outperform MoCo at the 50% and 100% label budgets, although their relative ranking depends on the probe. On Mendeley-Uganda, MoCo performs best for every probe and label budget. (c) At the full label budget under linear probing, MoCo increases by 20.6 points from POCUS to Mendeley-Uganda, while VideoMAE and V-JEPA decrease by 12.7 and 30.3 points, respectively. Error bars in (a, b) show the standard deviation across five POCUS test folds and the five corresponding Mendeley-Uganda evaluations.

## 4. SCOPE OF THE PRESENT STUDY AND PLANNED ANALYSIS

These results are preliminary in two respects: we evaluate only one backbone scale (ViT-S), and the downstream task is limited to three-class classification, which does not directly test the temporal information that motivates video pretraining. The pretraining and evaluation datasets are separate: COVID-BLUeS is used for pretraining, POCUS is used for patient-level cross-validation, and Mendeley-Uganda is used for external evaluation. POCUS is treated as in-distribution with respect to probe training, not SSL pretraining.

The ranking changes across the two evaluation datasets. The full paper will examine this diference using analyses beyond downstream accuracy. Our central planned analysis is a feature-space visualisation of the three pretrained encoders, following Ivezi´c et al.:<sup>18</sup> per-head attention maps, principal-component projections of the patch tokens, and cosine-similarity maps with respect to anchor patches placed on the pleural line and within Bline artefacts, which together reveal whether an objective allocates capacity to diagnostic structure or to speckle and acquisition artefacts. Because POCUS provides frame-level annotations of A-lines, B-lines, consolidation and pleural efusion,<sup>1</sup> we will further score the agreement between these similarity maps and the annotated regions, turning the qualitative maps into a quantitative measure.

## 5. CONCLUSION

We compared three self-supervised pretraining objectives for lung ultrasound using the same encoder backbone, pretraining dataset, training schedule and frozen evaluation protocol. At the full label budget under linear probing, VideoMAE and V-JEPA perform best on POCUS, while MoCo performs best on Mendeley-Uganda and V-JEPA drops to near-chance performance. This reversal shows that POCUS probe accuracy alone is not suficient for selecting the objective that transfers best to a new acquisition setting. Ongoing work adds the representation-level analyses described in Sec. 4 to examine the diferences between the learned representations.

## ACKNOWLEDGMENTS

This work was supported by the Canadian Foundation for Innovation-John R. Evans Leaders Fund (CFI-JELF) program grant number 42816. Mitacs Accelerate program grant number AWD024298-IT33280. We also acknowledge the support of the Natural Sciences and Engineering Research Council of Canada (NSERC), [RGPIN-2023- 03575]. Cette recherche a ´et´e financ´ee par le Conseil de recherches en sciences naturelles et en g´enie du Canada (CRSNG), [RGPIN-2023-03575].

## REFERENCES

[1] Born, J., Wiedemann, N., Cossio, M., Buhre, C., Br¨andle, G., Leidermann, K., Goulet, J., Aujayeb, A., Moor, M., Rieck, B., et al., “Accelerating detection of lung pathologies with explainable ultrasound image analysis,” Applied Sciences 11(2), 672 (2021).

[2] Roy, S., Menapace, W., Oei, S., Luijten, B., Fini, E., Saltori, C., Huijben, I., Chennakeshava, N., Mento, F., Sentelli, A., et al., “Deep learning for classification and localization of covid-19 markers in point-of-care lung ultrasound,” IEEE transactions on medical imaging 39(8), 2676–2687 (2020).

[3] Chen, T., Kornblith, S., Norouzi, M., and Hinton, G., “A simple framework for contrastive learning of visual representations,” in [International conference on machine learning], 1597–1607, PmLR (2020).

[4] Chen, X., Xie, S., and He, K., “An empirical study of training self-supervised vision transformers,” in [2021 IEEE/CVF international conference on computer vision (ICCV)], 9620–9629, IEEE (2021).

[5] He, K., Chen, X., Xie, S., Li, Y., Doll´ar, P., and Girshick, R., “Masked autoencoders are scalable vision learners,” in [2022 IEEE/CVF conference on computer vision and pattern recognition (CVPR)], 15979– 15988, IEEE (2022).

[6] Tong, Z., Song, Y., Wang, J., and Wang, L., “Videomae: Masked autoencoders are data-eficient learners for self-supervised video pre-training,” Advances in neural information processing systems 35, 10078–10093 (2022).

[7] Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., and Ballas, N., “Self-supervised learning from images with a joint-embedding predictive architecture,” in [2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)], 15619–15629, IEEE (2023).

[8] Assran, M., Bardes, A., Fan, D., Garrido, Q., Howes, R., Muckley, M., Rizvi, A., Roberts, C., Sinha, K., Zholus, A., et al., “V-jepa 2: Self-supervised video models enable understanding, prediction and planning,” arXiv preprint arXiv:2506.09985 (2025).

[9] Balestriero, R. and LeCun, Y., “Learning by reconstruction produces uninformative features for perception,” arXiv preprint arXiv:2402.11337 (2024).

[10] Van Assel, H., Ibrahim, M., Biancalani, T., Regev, A., and Balestriero, R., “Joint-embedding vs reconstruction: Provable benefits of latent space prediction for self-supervised learning,” Advances in neural information processing systems 38, 21897–21937 (2026).

[11] Littwin, E., Saremi, O., Advani, M., Thilak, V., Nakkiran, P., Huang, C., and Susskind, J., “How jepa avoids noisy features: The implicit bias of deep linear self distillation networks,” Advances in Neural Information Processing Systems 37, 91300–91336 (2024).

[12] Chen, Y., Zhang, C., Liu, L., Feng, C., Dong, C., Luo, Y., and Wan, X., “Uscl: Pretraining deep ultrasound image diagnosis model through video contrastive representation learning,” in [International Conference on Medical Image Computing and Computer-Assisted Intervention], 627–637, Springer (2021).

[13] Jiao, J., Zhou, J., Li, X., Xia, M., Huang, Y., Huang, L., Wang, N., Zhang, X., Zhou, S., Wang, Y., et al., “Usfm: A universal ultrasound foundation model generalized to tasks and organs towards label eficient image analysis,” Medical image analysis 96, 103202 (2024).

[14] Megahed, Y., Ducharme, R., Erman, A., Walker, M. C., Hawken, S., and Chan, A. D., “Usf-mae: Ultrasound self-supervised foundation model with masked autoencoding,” Biomedical Signal Processing and Control 122, 110313 (2026).

[15] Ellis, E., Mendel, R., Bulpitt, A., Parsa, N., Byrne, M. F., and Ali, S., “Self-supervised ultrasound-video segmentation with feature prediction and 3d localised loss,” in [Medical Imaging 2026: Image Processing], 13925, 319–326, SPIE (2026).

[16] Mishra, D., Salehi, M., Saha, P., Patey, O., Papageorghiou, A., Asano, Y., and Noble, A., “Self-supervised learning of echocardiographic video representations via online cluster distillation,” Advances in Neural Information Processing Systems 38, 61229–61254 (2026).

[17] Radhachandran, A., Ivezi´c, V., Athreya, S., Anilkumar, R., Arnold, C. W., and Speier, W., “Us-jepa: A joint embedding predictive architecture for medical ultrasound,” arXiv preprint arXiv:2602.19322 (2026).

[18] Ivezi´c, V., Pleasure, M., Radhachandran, A., Panchavati, S., Athreya, S., Sant, V., Emert, B., Fishbein, G., Arnold, C., and Speier, W., “Pretext matters: An empirical study of ssl methods in medical imaging,” arXiv preprint arXiv:2603.22649 (2026).

[19] Sim´eoni, O., Vo, H. V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., et al., “Dinov3,” arXiv preprint arXiv:2508.10104 (2025).

[20] Anton, J., Castelli, L., Chan, M. F., Outters, M., Tang, W. H., Cheung, V., Shukla, P., Walambe, R., and Kotecha, K., “How well do self-supervised models transfer to medical imaging?,” Journal of Imaging 8(12), 320 (2022).

[21] Fedorov, A., Geenjaar, E., Wu, L., DeRamus, T. P., Calhoun, V. D., and Plis, S. M., “Tasting the cake: evaluating self-supervised generalization on out-of-distribution multimodal mri data,” arXiv preprint arXiv:2103.15914 (2021).

[22] Pan, T., Song, Y., Yang, T., Jiang, W., and Liu, W., “Videomoco: Contrastive video representation learning with temporally adversarial examples,” arXiv preprint arXiv:2103.05905 (2021).

[23] Bardes, A., Garrido, Q., Ponce, J., Chen, X., Rabbat, M., LeCun, Y., Assran, M., and Ballas, N., “V-JEPA: Latent video prediction for visual representation learning,” (2024).

[24] Wiedemann, N., Boer, D. d. K.-d., Richter, M., van de Weijer, S., Buhre, C., Eggert, F. A. M., Aarnoudse, S., Grevendonk, L., R¨ober, S., Remie, C. M. E., Buhre, W., Henry, R., and Born, J., “Covid-blues - a prospective study on the value of ai in lung ultrasound analysis,” IEEE Journal of Biomedical and Health Informatics 29(9), 6301–6310 (2025).

[25] Katumba, A., Murindanyi, S., Okila, N., Nakatumba-Nabende, J., Mwikirize, C., Serugunda, J., Bugeza, S., Oriekot, A., Bossa, J., and Nabawanuka, E., “A dataset of lung ultrasound images for automated ai-based lung disease classification,” Data in brief , 112034 (2025).