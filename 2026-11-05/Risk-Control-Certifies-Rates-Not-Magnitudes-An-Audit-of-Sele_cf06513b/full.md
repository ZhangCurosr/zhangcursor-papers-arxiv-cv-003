# Risk Control Certifies Rates, Not Magnitudes: An Audit of Selective Fetal Biometry

Zhengshuai Yang<sup>1</sup>, Linna Jiang<sup>1</sup>, Changyun Li<sup>1,⋆</sup>, and Guihu Zhao<sup>2,⋆</sup>

<sup>1</sup> College of Information and Intelligent Science and Technology, Hunan Agricultural University, Changsha 410128, China

yhvguk@stu.hunau.edu.cn, 362330193@qq.com

<sup>2</sup> National Clinical Research Center for Geriatric Disorders, Xiangya Hospital, Central South University, Changsha 410008, China ghzhao@csu.edu.cn

Abstract. Prenatal ultrasound biometry must return a head circumference (HC) and refuse when the image cannot support one. Selective prediction trades coverage against error, but its finite-sample guarantees bound the rate of gross errors, not their magnitude. We audit a toolgrounded biometry pipeline on two separately pre-registered four-way splits of the 999 labelled HC18 images, retraining the segmenter three times on each. Calibrating the contour-completeness threshold by exact Clopper–Pearson risk control returns a certificate indistinguishable from a hand-set floor, yet the pipeline still emits a 271.3 mm error: a small, geometrically complete ring fitted to an image whose true HC is 324.6 mm. The accepted candidate scores 0.94 completeness, so the score cannot see the failure, and one outlier among ≈150 answers leaves the certified risk bound (α=0.10) intact. A second, untouched split reproduces the mechanism on a diferent image (233.3 mm). Constraining the admissibility region—the anatomical range in which the measurand is defined— removes the family instead: raising the lower bound from 50 to 55 mm costs 0.4 coverage points and cuts the worst-case error on both splits to 11.4 and 11.1 mm, whereas suppressing it through the completeness threshold costs 19.0 and 12.1 points. At an anatomically defined 70 mm the frozen tests give 1.91 and 1.81 mm MAE.

Keywords: selective prediction · distribution-free risk control · fetal ultrasound biometry · head circumference · failure-mode audit.

## 1 Introduction

An assistant that reports fetal head circumference (HC) from ultrasound must return a physically scaled measurement and refuse when the image cannot support one. HC, a routine indicator of fetal growth, can be estimated from a skull contour and the pixel spacing [11]; inside a vision–language model (VLM) the harder question is not how to compute the value but when it can be trusted.

Producing a reliable numeric measurement and deciding whether to trust it are distinct challenges that expose the limitations of current VLMs. A general end-to-end VLM often struggles with precise geometric biometry; it may omit a measurement, hallucinate a scale from pure visual features, or blindly accept an incorrect external observation. Conversely, a purely geometric tool lacks semantic understanding and can return a mathematically plausible ellipse fitted to the wrong anatomical structure. Consequently, evaluating only measurement error on a selected high-quality subset does not establish reliable biometry across a full clinical cohort.

We study PUMA (Prenatal Ultrasound Multimodal Assistant), a toolgrounded VQA system that separates geometric computation, candidate selection, and anatomical verification into auditable steps. PUMA computes the measurement before generating an answer: a contour segmenter and an ellipse-fitting tool compute HC in millimetres, a completeness gate filters partial candidates, additional views are considered if the initial view fails, and the VLM verifies the accepted observation or abstains. This design exposes an interpretable selection signal, allowing us to evaluate measurement error and the proportion of answered questions—and, as we show, to measure how far a calibrated selection rule can actually be trusted.

In summary, our main contributions are: 1) We audit PUMA, a tool-grounded biometry pipeline, on a curated prenatal benchmark (Sections 3 and 6.1): against 1/96 HC answers for a direct VLM, the delivered configuration answers 66/96 at 1.76 mm MAE, and the paired benchmark with an ablation shows that the geometric gate, not the language model, is the decisive component. 2) Across two pre-registered four-way splits with three retrained seeds each, exact Clopper– Pearson calibration of the completeness threshold is indistinguishable from a hand-set floor—and remains compatible with a 271.3 mm error, because at n ≈ 150 a single outlier barely moves a gross-error rate (Section 6.3). 3) We show that calibrating the anatomical admissibility region, rather than the score threshold, removes this family for 0.4 coverage points instead of 19, and confirm it with one-shot frozen-test evaluations on both splits (Section 6.3).

## 2 Related Work

Fetal biometry. HC18 supports automated head measurement [11], including studies of human agreement [18] and contour estimation [21], while plane detection and verification [3,20] complement it with FETAL\_PLANES\_DB [4]. We study acceptance of tool outputs, without claiming superiority over dedicated biometry systems.

Medical VQA and tools. VQA-RAD, SLAKE and PMC-VQA support biomedical question answering [12,15,22] and multimodal assistants [14,16]. Retrieval supplies textual context [13]; biometry additionally requires geometry and scale. Unlike model-directed tool invocation [17,19], PUMA measures before generating and then requests verification.

Selective prediction. Reject-option methods trade answer coverage against error [6,8,9]. Calibration [10] and distribution-free risk control [1,2] formalise this tradeof; we apply exact Clopper–Pearson bounds on a reserved calibration split.

## 3 Data: Curation and Split Audit

The catalogue holds 98,607 records across ten adapters, from HC18 (999 labelled images), FETAL\_PLANES\_DB, PMC-VQA, SLAKE, VQA-RAD and three obstetric text archives; MedDialog (17,152) is quarantined because its reuse terms are unverified. Connected groups link records by patient, examination or article identifiers, exact image hashes and normalised-text duplicates, and oficial test assignments take precedence. Only 126 native prenatal questions survive domain screening, and 82 PMC-VQA figures pass a caption rule requiring ultrasound terminology and prenatal context. HC18 has no patient identifiers, so its isolation is image-level.

## 4 Method

## 4.1 Answer Contract and Strategies

The benchmark shares one served generator (qwen3.8-flash, snapshot 2026- 09-02), temperature 0, a 1,200-token budget and a 240 s timeout. Protocol puma-v1.4 returns answer, abstained, limitations and evidence\_ids: a short label, at most ten words, or a value in mm, with qualifications separate. The original benchmark counts empty successful responses as abstentions; later logs distinguish failed generation from refusal.

direct\_vqa uses image and question; rag\_vqa adds retrieval context; staged\_agent allocates budgeted time to observation, answering, and verification; tool\_vqa adds measurement and requests anatomical verification.

## 4.2 Deterministic Biometry Tool

The measurement pipeline in Fig. 1 separates contour estimation, physical conversion, and candidate selection before the VLM receives an observation. By isolating these steps, PUMA provides an auditable numeric measurement rather than relying on the VLM to estimate geometry.

Segmentation. To ground the final answer in physical pixels, a DeepLabV3– ResNet50 model [5] predicts skull contours. Images are non-uniformly resized to 512 × 512 pixels, and predictions are mapped back to source coordinates. The network is trained on 736 HC18 images with physician masks using BCE plus Dice loss. The delivered v2 recipe further adds shadow-sector occlusion and boundary-band BCE to improve robustness on poor-quality scans. Any numeric answer therefore originates from a concrete image region rather than a VLM estimate.

![](images/261d6b64e225bf87c12a93afae9a82b11aaced6bfead81e2cea81d6278f83513.jpg)  
Fig. 1. Measurement runs before VLM generation. The original end-to-end runs use L = 25 mm; ofline audits and the second-family run also examine $L = 5 0 \mathrm { m m }$ . The VLM may accept the observation or abstain.

Geometric Conversion. Converting a raw pixel contour to a continuous circumference measurement requires geometric fitting. After morphological closing, connected-component rims are fitted using a direct least-squares conic method [7]. The HC is computed as the Ramanujan ellipse perimeter multiplied by the DICOM pixel spacing. We also define contour completeness c as the fraction of 16 angular bins around the fitted centre that are occupied by rim pixels. By computing this deterministic geometric metric, the system can interpret the quality of the observation independently of the VLM’s semantic understanding.

Completeness Gate and Additional Views. A VLM might blindly accept a mathematically plausible but anatomically incorrect ellipse (e.g., fitted to a small artifact). To prevent this, a geometric gate filters candidates before VLM verification. A candidate is eligible only if its completeness $c \geq \tau$ and its HC lies in [L, 450] mm (the original runs use $\tau = 0 . 7$ and $L = 2 5 \mathrm { m m } )$ . If the initial view yields no eligible component, the tool acts on four shifted and two zoomed views, reversing the zoom scale on the output perimeter. This hard gate prevents the VLM from answering when no valid contour exists, thereby lowering the risk of gross errors.

Structured Observations. Finally, the tool exports an observation state (measured, abstained, error, or unavailable) to the VLM. On accepted HC answers, the VLM copies the reported value exactly. This structured handof reduces the VLM’s role to verifying whether the proposed measurement semantically aligns with the question; the value is passed through rather than estimated.

## 5 Experimental Setup

The development benchmark has 200 validation questions over 163 groups: 97 plane, 96 HC and seven native prenatal questions, selected by stratified group round-robin. Brain sub-planes were excluded after a pilot. Interleaved strategies produced 800 answers without call errors; two retrieval usage-censored rows are excluded from the table.

We report exact match and token-F1 with group-bootstrap 95% CIs. MAE and median absolute error use returned measurements; answer rate uses all questions. Gross HC error means absolute error ≥ 7 mm, with rate among returned readings. The 2 mm hit threshold and ±2% rule are engineering criteria, not clinical tolerances.

Table 1. Plane accuracy and HC metrics are conditional on answered questions; tokens are run totals in thousands and latency the run-record mean. Only the first four rows share one run, so only those support the paired tests; the delivered run holds the tool arm alone and its plane accuracy is unpaired. <sup>⋆</sup>HC biometry only. Ofline replay of the same candidates with the L=50 mm floor gives 66 answers at 1.78 mm, so the language model adds no measurement.
<table><tr><td rowspan="2">Strategy</td><td colspan="2">Plane QA (n=97)</td><td colspan="4">HC Biometry (n=96)</td><td rowspan="2">Tok. Lat. (k)</td><td rowspan="2">(s)</td></tr><tr><td>Ans.</td><td>Acc.</td><td></td><td>Ans. MAE</td><td>Med.</td><td>&gt;7mm</td></tr><tr><td colspan="9">Paired benchmark: one interleaved four-arm run (89d79bf1)</td></tr><tr><td>Direct</td><td>88</td><td>.841</td><td>1</td><td>41.0</td><td>41.0</td><td>1</td><td>290</td><td>9.3</td></tr><tr><td>Retrieval</td><td>92</td><td>.880</td><td>9</td><td>12.7</td><td>13.3</td><td>6</td><td>432</td><td>17.9</td></tr><tr><td>Staged</td><td>77</td><td>.766</td><td>0</td><td></td><td></td><td>0</td><td>1,424</td><td>43.0</td></tr><tr><td>Tool VQA</td><td>93</td><td>.871</td><td>71</td><td>6.50</td><td>2.30</td><td>16</td><td>373</td><td>14.8</td></tr><tr><td colspan="9">Delivered run (98364160), tool arm only, not paired with the block above</td></tr><tr><td>Tool VQA + floor*</td><td>93</td><td>.914</td><td>66</td><td>1.76</td><td>1.19</td><td>1</td><td>318</td><td>13.0</td></tr><tr><td colspan="9">Deterministic replay of the same stored candidates (no language model)</td></tr><tr><td>Tool+Gate, L=25 mm</td><td></td><td></td><td>67</td><td>5.74</td><td>1.19</td><td>2</td><td>0</td><td>1.2</td></tr><tr><td>Tool+Gate, L=50 mm</td><td></td><td></td><td>66</td><td>1.78</td><td>1.18</td><td>1</td><td>0</td><td>1.2</td></tr></table>

A rebuilt three-judge panel reproduces 54/66 numeric acceptances (81.8%). Across 80 audited non-numeric responses, acceptance is 18/80 (22.5%) and Fleiss $\kappa = 0 . 9 3 1$

## 6 Results

## 6.1 Paired Strategy Benchmark

Tool grounding is what makes HC answers possible, and the corrected floor is what makes them accurate (Table 1). On 97 plane questions it holds 87.1% accuracy among 93 answered, matching retrieval’s 88.0%. On the 96 HC questions the direct VLM abstains on 95/96 (its single attempt of by 41.0 mm) and retrieval answers nine at 12.7 mm MAE, whereas tool grounding answers 71 at 6.50 mm; the delivered run with the corrected floor answers 66 at 1.76 mm with one gross error.

Evaluating the deterministic tool without VLM verification yields 67 answers at 5.74 mm MAE because of one degenerate 28.1 mm ring; adding the 50 mm floor gives 66 answers at 1.78 mm, matching the delivered run’s 66 at 1.76 mm. Within this benchmark the language model therefore changes nothing that the geometric gate does not already change—an ablation we revisit in Section 6.3.

Table 2. Frozen-test evaluation on two pre-registered splits, τ=0.70 throughout, mean ± sd over three retrained seeds; worst case is the largest absolute error in the split. The second split was registered after the first failure was identified, so it is a confirmatory replication.
<table><tr><td>Split / floor L Cov.</td><td>n</td><td></td><td></td><td>MAE (mm) Gross (≥7mm) Worst case (mm)</td></tr><tr><td>Split 1, frozen test (n=156)</td><td></td><td></td><td></td><td></td></tr><tr><td> $L { = } 5 0 \mathrm { m m }$ </td><td>.722 112.7</td><td> $3 . 5 1 \pm 1 . 4 2$ </td><td>2.39%</td><td>271.3</td></tr><tr><td> $L { = } 5 5 \mathrm { m m }$ </td><td>.718 112.0</td><td> ${ \bf 1 . 9 3 \pm 0 . 0 3 }$ </td><td>1.82%</td><td>11.4</td></tr><tr><td>Split 2, frozen test</td><td> $\scriptstyle ( n = 1 5 2 )$ </td><td></td><td></td><td></td></tr><tr><td> $L { = } 5 0 \mathrm { m m }$ </td><td>.750 114.0</td><td> $2 . 5 3 \pm 1 . 1 3$ </td><td>1.41%</td><td>233.3</td></tr><tr><td> $L { = } 5 5 \mathrm { m m }$ </td><td>.748 113.7</td><td> ${ \bf 1 . 8 1 \pm 0 . 1 7 }$ </td><td>1.10%</td><td>11.1</td></tr></table>

A second served family (gpt-6, snapshot 2026-09-12) reproduces the pattern: 64 HC readings at 1.80 mm MAE, with 14% abstention against 43%; 26 of 400 calls fail and are excluded.

## 6.2 Completeness Gating and Additional Views

With v2 weights, free VLM checking answers 77/96 at 6.49 mm MAE with 22 gross errors; gating gives 48 at 1.80 mm with one and the six-view fallback 66 at 1.76 mm with one. Fourteen of sixteen v1 gross errors involve incomplete contours $( c = 0 . 5 0  – 0 . 6 2 )$ , while the VLM rejects one high-completeness false contour worth 266.6 mm. These runs use the original 25 mm floor.

## 6.3 Admissibility Audit on Two Pre-Registered Splits

A whole-cohort replay already shows which family the pipeline fails on: among 999 labelled images, 45 readings below 50 mm with completeness $c \geq 0 . 7$ account for 44 of 53 gross errors. We therefore audit twice whether raising the floor removes them under a clean, one-shot evaluation. Each four-way partition fixes a 58/12/15/15 ratio (train/model selection/calibration/frozen test): calibration sets are n=177 (split 1) and n=149 (split 2). To prevent data leakage, split 2 was pre-registered with candidate floors $L \in \{ 5 0 , 5 5 , 7 0 \}$ mm fixed before its frozen test was evaluated. We evaluate $\tau \in [ 0 . 5 0 , 1 . 0 0 ]$ using exact Clopper–Pearson binomial bounds with Bonferroni grid correction rather than asymptotic bounds, because at n ≈ 150 Hoefding is excessively conservative (0/105 yields 0.105 vs exact 0.022).

A reproducible failure mode. At L=50 the pipeline emits one error two orders of magnitude larger than the median in both splits, from two diferent images (Fig. 2): 977\_HC.png (gold 324.6 mm, reported 53 mm; error 271.3 mm) and 945\_HC.png (gold 284.9 mm, reported 52 mm; error 233.3 mm). Across the six retrained models the accepted candidates for these two images are reported at 51.6–54.9 mm with completeness 0.75–0.94, so a gate built on c admits them and the 50 mm floor sits just below their band.

![](images/6194993413ab82aee49a6fd71aae642ba7c4d42a557f81b27cdb4a30722a4e45.jpg)  
Fig. 2. The two failures that passed the completeness gate. Green: physician annotation; red dashed: the accepted conic fit, magnified in the inset. Left (split 1): 977\_HC.png, gold 324.6 mm, fit 53.3 mm at c=0.75, error 271.3 mm. Right (split 2): 945\_HC.png, gold 284.9 mm, fit 51.7 mm at c=0.88, error 233.3 mm. Both fits sit on the bright specular arc at the skull boundary, which is smooth and therefore scores high completeness.

Risk control certifies rates, not magnitudes. Calibrating τ by exact Clopper– Pearson bounds at $\alpha { = } 0 . 1 0$ returns $\tau ^ { * } { = } 0 . 6 3 { - } 0 . 6 9$ on split 1, below the hand-set 0.70; the calibrated and fixed operating points are then identical to the reported precision in two of three seeds, and MAE moves from 3.51 to 3.53 mm. On split 2 the same procedure selects $L { = } 5 0$ in all three seeds, and all six calibration partitions certify it at $\alpha { = } 0 . 1 0$ (upper bounds 0.070–0.097)—including the two that already contain an error above 250 mm. The frozen test then still returns the 233.3 mm error. The certificate is not wrong: with n ≈ 150 answers, a single gross error leaves the upper bound on the gross-error rate below 10%. The guarantee constrains how often the pipeline is wrong, not how badly, so a certified operating point can still be catastrophically wrong once.

Constraining the admissibility region. Because this failure is defined by where the measurement is physically possible, the efective parameter is the admissibility region [L, 450] mm—the anatomical range in which the measurand is defined—not the score threshold. Raising L from 50 to 55 mm costs 0.4 coverage points and reduces the worst-case error from 271.3 to 11.4 mm on split 1 and from 233.3 to 11.1 mm on split 2. Suppressing the same three errors through τ instead requires $\tau \geq 0 . 9 4$ , the extreme of the score range, and costs 19.0 coverage points on the split-1 frozen tests and 12.1 on split 2. With the anatomically defined L=70 mm—below the range in which the trans-thalamic plane is used—the split-2 calibration partitions certify at α=0.10 for all three seeds (upper bounds 0.099, 0.057, 0.085) and the frozen tests give $1 . 9 1 \pm 0 . 0 2$ mm and $1 . 8 1 \pm 0 . 1 7$ mm MAE at 69.4% and 73.9% coverage.

## 6.4 Secondary Checks

Secondary checks. The segmenter reaches a mask-level contour MAE of 9.49 mm (median 3.03 mm) on the validation partition, against 10.31 mm for the v1 recipe;

these are training-diagnostic errors, and the end-to-end biometry MAE above is the system’s operating performance. For context, human inter-observer diference on HC18 averages 2–3% (≈2–5 mm across gestations) [18,11], placing the delivered 1.76 mm MAE within expert variability while masking catastrophic gross failures. Forced to answer, the second family returns 31 numeric readings on the 96 HC questions at 4.18 mm MAE with six gross errors (19.4%); on 200 external plane questions 167 calls return at exact match 0.72 without abstaining.

## 7 Discussion and Limitations

What the audit establishes. The main result concerns the shape of the guarantee rather than the pipeline: selecting on a score with finite-sample risk control bounds how often the system is grossly wrong, not how badly. At the sample sizes a calibration partition afords, an operating point can be certified at α=0.10 and still return a 233 mm error, and threshold calibration cannot remove that possibility, because the ofending candidate is geometrically complete. Against a direct VLM, tool grounding is what makes millimetre answers possible at all, and the geometric gate rather than the language model suppresses degenerate candidates.

Practical consequence. The parameter that matters is the admissibility region—the range in which the measurand is physically defined—and it should come from anatomy rather than from tuning. Here the admissibility bound removes the failure family for 0.4 coverage points, while the score threshold needs 0.94 and roughly twenty times as many. Such bounds are cheap to state and transfer to any target whose plausible size is known in advance; score thresholds are not.

Limitations. HC18 carries no patient identifier, so both splits are image-level and cannot exclude perceptual near-duplicates. The second split was registered after the failure was identified on the first, and the L values tested on it were fixed before its evaluation, so it tests a pre-specified hypothesis rather than searching for one. The measurement is deterministic by design: the language model carries the answer contract and the refusal decision, not the number, which is why gating alone reproduces the delivered MAE. Assessment is retrospective and automated rather than clinical, and the served families are compared as deployed rather than at matched weights. On 335 unlabelled oficial-test images the single-view answer rate is 0.573, matching 0.573 on labelled images, so the operating behaviour is stable across the two cohorts.

## 8 Conclusion

PUMA separates geometric measurement, candidate selection and languagemodel verification. Auditing it on two pre-registered splits shows that calibrated rejection bounds the rate of gross errors but not their magnitude, and that the efective safeguard is an anatomically defined admissibility region rather than a tuned score threshold. The audit framework transfers directly to patient-level multi-centre data and to any measurand whose plausible range is known in advance.

Availability and data. Stored predictions, weights and scripts support these analyses; new generation needs API access. HC18, FETAL\_PLANES\_DB and SLAKE are CC-BY-4.0, VQA-RAD CC0-1.0, PMC-VQA CC-BY-SA, Huatuo26M-Lite Apache-2.0; records with unresolved reuse terms are excluded. No new patient data were collected. Data use. All data are existing public, de-identified archives; no ethics approval was required.

## References

1. Angelopoulos, A.N., Bates, S.: Conformal prediction: A gentle introduction. Foundations and Trends in Machine Learning 16(4), 494–591 (2023)

2. Bates, S., Angelopoulos, A., Lei, L., et al.: Distribution-free, risk-controlling prediction sets. Journal of the ACM 68(6), 1–34 (2021)

3. Baumgartner, C.F., et al.: SonoNet: Real-time detection and localisation of fetal standard scan planes in freehand ultrasound. IEEE Transactions on Medical Imaging 36(11), 2204–2215 (2017)

4. Burgos-Artizzu, X.P., et al.: FETAL\_PLANES\_DB: Common maternal-fetal ultrasound images (2020)

5. Chen, L.C., Papandreou, G., Schrof, F., Adam, H.: Rethinking atrous convolution for semantic image segmentation. arXiv preprint arXiv:1706.05587 (2017)

6. Chow, C.: On optimum recognition error and reject tradeof. IEEE Transactions on Information Theory 16(1), 41–46 (1970)

7. Fitzgibbon, A., Pilu, M., Fisher, R.B.: Direct least square fitting of ellipses. IEEE Transactions on Pattern Analysis and Machine Intelligence 21(5), 476–480 (1999)

8. Geifman, Y., El-Yaniv, R.: Selective classification for deep neural networks. In: Advances in Neural Information Processing Systems 30 (NeurIPS). pp. 4878–4887 (2017)

9. Geifman, Y., El-Yaniv, R.: Selectivenet: A deep neural network with an integrated reject option (2019)

10. Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q.: On calibration of modern neural networks (2017)

11. van den Heuvel, T., de Bruijn, D., de Korte, C.L., van Ginneken, B.: Automated measurement of fetal head circumference (HC18 dataset) (2018)

12. Lau, J.J., Gayen, S., Ben Abacha, A., Demner-Fushman, D.: A dataset of clinically generated visual questions and answers about radiology images. Scientific Data 5, 180251 (2018)

13. Lewis, P., Perez, E., Piktus, A., et al.: Retrieval-augmented generation for knowledge-intensive nlp tasks (2020)

14. Li, C., et al.: LLaVA-Med: Training a large language-and-vision assistant for biomedicine in one day. In: Advances in Neural Information Processing Systems 36 (NeurIPS) (2023)

15. Liu, B., Zhan, L.M., Xu, L., et al.: SLAKE: A semantically-labeled knowledgeenhanced dataset for medical visual question answering. In: Proceedings of the 18th IEEE International Symposium on Biomedical Imaging (ISBI). pp. 1650–1654 (2021)

16. Moor, M., Huang, Q., Wu, S., et al.: Med-flamingo: a multimodal medical few-shot learner (2023)

17. Schick, T., et al.: Toolformer: Language models can teach themselves to use tools (2023), arXiv:2302.04761

18. Sinclair, M., Baumgartner, C.F., Matthew, J., et al.: Human-level performance on automatic head biometrics in fetal ultrasound using fully convolutional neural networks. In: 2018 40th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC). pp. 714–717 (2018)

19. Yao, S., Zhao, J., Yu, D., et al.: React: Synergizing reasoning and acting in language models (2022)

20. Yaqub, M., Kelly, B., Papageorghiou, A.T., Noble, J.A.: A deep learning solution for automatic fetal neurosonographic diagnostic plane verification using clinical standard constraints. Ultrasound in Medicine & Biology 43(12), 2925–2933 (2017)

21. Zeng, W., Luo, J., Cheng, J., Lu, Y.: Eficient fetal ultrasound image segmentation for automatic head circumference measurement. Medical Physics 49(8), 5081–5092 (2022)

22. Zhang, X., et al.: PMC-VQA: Visual instruction tuning for medical visual question answering. arXiv preprint arXiv:2305.10415 (2023)