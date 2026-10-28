# Readout and Operating-Point Sensitivity in Event-Stream Candidate Selection

Yifan Shuai

LiAuto

sddsyf@gmail.com

## Abstract

Event-triggered recording needs a useful combination of target capture and false triggers, yet representation comparisons can change with readout and threshold calibration. We evaluate real event accumulations from 520 DailyDVS-200 clips and 47 participants using participant-disjoint training, validation, and testing. The original study varies frozen encoder scale, accumulation duration, and temporal view count; a postreview extension adds two supervised-pretrained convolutional encoders, a 61,586-parameter trained CNN with three seeds, ten repeated participant partitions, and calibration sensitivity. At 50 ms/eight views under the primary handshake-or-fall policy, DINO-S CLS captures 101/132 targets with 40/388 false triggers, versus 78/132 and 35/388 for patch mean. DINO-B CLS has lower selected-threshold capture but higher fold-wise low-FPR partial AUC (0.746 versus 0.732). MobileNetV3-Small captures 97/132 with 53/388 false triggers, demonstrating why lightweight baselines and actual operating points matter. Across ten dependent repartitions, mean normalized partial AUC is 0.738 for DINO-B CLS and 0.674 for MobileNet. The original LeJEPA S-to-B capture diference is 8.3 points, but its multiplicity-adjusted interval includes zero. Duration and view-count efects remain policy dependent. We find no established JEPA-specific advantage. The evidence concerns ofline action-defined candidate selection on previously used development data; it does not establish personal recording value, continuous false alarms per hour, or eyewear energy.

Keywords: event-based vision; candidate selection; frozen representations; readout; calibration; low-FPR evaluation.

## 1 Introduction

A personal recording system may combine audio, inertial sensing, explicit user actions, and vision to decide when to collect richer evidence. For its visual branch, missing a candidate can lose an opportunity to record; excessive triggers increase downstream activity. A high capture rate alone therefore does not establish a good trigger. The same concern applies when comparing models with independently calibrated thresholds.

Event cameras report asynchronous brightness changes [1]. OpenGlass and Helios motivate low-power wearable event sensing [2,3], but their device results do not measure the pipeline studied here. We investigate an earlier, ofline question: how do representation choices and available event evidence afect binary candidate selection under a common grouped protocol?

The initial study focused on encoder size and accumulation budgets. Its readout controls revealed a larger descriptive change for one DINO configuration than the within-family size change. That observation motivates the present post-review analysis, without converting it into a hypothesis tested on fresh data. We show that representation, readout, and operatingpoint calibration interact: a ranking at a selected threshold need not match a ranking over the low-FPR ROC region. Lightweight supervised-pretrained models also provide a competitive alternative to larger frozen Transformers.

The contribution is an empirical assessment with three parts: (1) a common real-event protocol comparing readouts, model families, lightweight baselines, and accumulation budgets; (2) joint reporting of validation-calibrated capture/false triggers and foldwise low-FPR discrimination; and (3) repeated participant grouping, threshold recalibration, and paired uncertainty analyses that make the limits of smalldata conclusions inspectable. We do not propose a new encoder, event representation, training objective, or wearable device.

## 2 Related Work

Event-based vision includes native asynchronous processing and image-like event representations [1]. DailyDVS-200 supplies real event action recordings [4]. SMV-EAR explores multi-view spatiotemporal representations for event action recognition [5]; our independent polarity accumulations are not its alternative-axis projections, and our binary policies are not the complete multiclass benchmark.

I-JEPA and V-JEPA learn visual representations through embedding prediction [6,7]. LeJEPA combines cross-view invariance and isotropic Gaussian regularization [8]; DINO uses self-distillation [9]. Our four main image checkpoints are pinned third-party Open-Knowledge-AI ImageNet-1K releases [10–13]. Shared backbone configurations control some architectural factors, but training and export diferences preclude objective-only causal comparisons. An oficial distilled

V-JEPA 2.1-B image-path reference [14] is evaluated separately from temporal video inference. Supervisedpretrained MobileNetV3-Small [15] and ResNet-18 [16] broaden the baseline beyond self-supervised ViTs.

Wearable event systems demonstrate a broader sensing-and-inference motivation [2,3]. Our novelty claim is confined to the empirical protocol and sensitivity findings. We have neither synchronized wearer modalities nor personal-interest labels to demonstrate multimodal complementarity or subjective memory value.

## 3 Protocol

## 3.1 Task, data, and inputs

The selected DailyDVS-200 subset contains 520 AE-DAT4 recordings from 47 participants: drinking, taking from a bag, phone use, putting into a bag, and handshake contribute 70 clips each; picking up and falling contribute 62 each; door opening contributes 46. P1 targets handshake or fall (132 targets/388 non-targets). P2 additionally targets door opening (178/342). A non-target means outside this fixed policy, not uninteresting to the wearer. Action labels supply deterministic binary supervision; language-model judgments are not labels or predictions.

Native event resolution is 320×240. An event is $e _ { i } =$ $( x _ { i } , y _ { i } , t _ { i } , p _ { i } )$ . We count each polarity in the half-open interval $[ u - W , u )$ , render positive/negative counts as red/blue, and set green to zero:

$$
\begin{array} { r } { R = \mathrm { r o u n d } \{ 2 5 5 [ 1 - \exp ( - h ^ { + } / 3 ) ] \} , } \\ { B = \mathrm { r o u n d } \{ 2 5 5 [ 1 - \exp ( - h ^ { - } / 3 ) ] \} , } \\ { G = 0 . } \end{array}
$$

The whole field of view is bilinearly resized with preserved aspect ratio and centered on a 224×224 gray-127 canvas. All encoders receive ImageNet mean/std normalization. This common event transform difers from the standard supervised ImageNet center-crop evaluation recipe. It is image-like inference on real events, not native asynchronous inference.

Every clip supplies a centered 1.6 s interval; none is shorter. The original grid uses $W \in \{ 2 0 , 3 0 , 5 0 \}$ ms and $N \in \{ 1 , 2 , 4 , 8 , 1 6 , 3 2 \}$ views. For two or more views, right boundaries span 50–1600 ms uniformly, rounded to microseconds; a single view is the final snapshot. Windows do not overlap and can leave unsampled gaps. More views change retained evidence as well as compute. Centering requires the full clip length and does not implement online onset detection.

All new baseline, readout-stability, and calibration comparisons fix 50 ms/eight views. Exact boundaries are 50, 271.429, 492.857, 714.286, 935.714, 1157.143, 1378.571, and 1600 ms. Two original controls additionally vary history (400/800/1600 ms at eight views) or retain 25/50/100% of real events with three fixed thinning seeds. They modify encoder inputs, not measured sensor energy.

## 3.2 Encoders and readout

LeJEPA-S/B contain 21,665,664/85,798,656 encoder parameters, with 12-layer patch-16 ViT-v2 backbones and widths 384/768. DINO-S/B use corresponding publisher configurations. We verify checkpoint hashes, identify excluded training projections, and load encoders strictly. We do not independently establish the DINO export’s teacher/student role from a model-card evaluation statement.

Patch mean and CLS are alternative frozen readouts. Each view’s readout is L2 normalized; the clip feature averages eight view vectors. This aggregation discards order. The V-JEPA 2.1 distilled-B EMA reference (86,833,152 parameters) encodes independent 224-pixel views through its image path with the existing rotary interpolation setting, rather than using temporal attention. It has diferent pretraining and nominal resolution and is a cross-family reference.

MobileNetV3-Small and ResNet-18 use Torchvision 0.24.1 IMAGENET1K V1 weights from oficial hashchecked downloads. We remove their classification layers and spatially average, L2 normalize each view, and average views. Encoder parameter counts are 927,008 and 11,176,512, respectively; full published classification models contain 2,542,856 and 11,689,512. These are frozen supervised-pretrained features, not eventtrained encoders.

TinyCNN has 61,586 parameters: four stride-2 convolutions (5×5, 3→16; 3×3, 16→32→64→64), each with four-group GroupNorm and ReLU, spatial averaging, view averaging, and a 64→2 linear head. Each policy/fold is trained independently from scratch with all three seeds 20261022/23/24. It is a simple trained image-like event baseline, not a state-of-the-art eventnative model or an optimized architecture search.

## 3.3 Fitting and validation

Each person belongs to one of five folds. Test fold $f ,$ validation fold $( f + 1 )$ mod 5, and the remaining training folds are disjoint in people. Original test clip counts are 111/111/95/97/106; people counts are $1 0 / 1 0 / 9 / 9 / 9$ . There is one out-of-fold test prediction per clip per setting.

For frozen features, standardization is fitted on training clips only, followed by two-output ridge regression with alpha in {0.01, 0.1, 1, 10, 100}. The score is positive minus negative output. For each alpha, validation thresholds maximize capture subject to FPR budgets of $5 / 1 0 / 2 0 \%$ , with ties favoring fewer false triggers and then a higher threshold; always-of is allowed. Mean validation capture across budgets selects alpha, ties favoring stronger regularization. Selected thresholds are then applied unchanged to test clips. The 10% validation budget does not enforce 10% test FPR.

TinyCNN uses 30 epochs, AdamW (learning rate 0.001, weight decay 0.0001), clip batch eight, trainingonly class weights, and no augmentation. Each epoch is evaluated only on validation clips. The same threebudget mean capture selects the checkpoint, ties favoring the earliest epoch. A checkpoint and its thresholds are saved before test scoring. All three training seeds are reported without selecting or ensembling them. This training regime is not identical to frozen-feature fitting.

## 4 Evaluation and Analysis Scope

## 4.1 Development exposure and repeated grouping

These clips and all 47 participants had prior experimental use. The original 148 input conditions were fixed before their scoring, but were never an untouched confirmation set. The new baselines, repeated grouping, and calibration analyses are explicitly post-review extensions; they do not retroactively become original predeclared hypotheses.

Ten new fixed grouping seeds (20261101–20261110) repartition the same participants using targetavailability strata and deterministic SHA sorting. Each repeat uses five grouped train/validation/test rotations, the same fitting procedure, and all ten frozen representations (four ViTs × two readouts plus two CNNs): 1,000 new fitted heads. All repeats are retained. Their spread measures partition sensitivity on known development participants, not independent population replications. TinyCNN is evaluated on the original grouping with its three seeds, rather than on this full repeated grid.

## 4.2 Metrics and uncertainty

Capture is TP/(TP+FN); false-trigger rate is FP/(FP+TN), with counts reported. One clip gives one decision. The selected class mixture cannot establish everyday prevalence, precision, recorded duration, or false triggers per hour.

ROC curves are computed within each test fold because diferent fold heads need not share a score scale. We report the unweighted fold mean of AUROC, normalized partial area from FPR 0 to 0.2 (pAUC20 = area/0.2), and linearly interpolated TPR at FPR 0.1 (TPR10). Ties share a ROC step. pAUC20 ranges from zero to one; a random diagonal has 0.1, not 0.5. This is not McClish-standardized partial AUC. Test ROC interpolation is a retrospective matched-FPR diagnostic, not a deployable validation-calibrated threshold or an estimate of calibration reliability. Interpolated points need not be attainable by a deterministic threshold when scores tie.

To examine calibration separately, we hold each original head and alpha fixed, bootstrap validation people 1,000 times within folds, and recalibrate the 10% threshold. Unchanged test scores yield a distribution of aggregate capture/FPR. Its 5th/50th/95th percentiles describe conditional sensitivity; they exclude refitting and pretraining uncertainty and are not confidence limits for generalization.

The three original contrasts (scale, duration, view count) under both policies retain their original definitions. A post-review reanalysis applies 50,000 paired participant resamples and Bonferroni percentile intervals across six contrasts × two metrics (family 12). These conditional, approximate bootstrap intervals do not establish fresh confirmatory significance. Original unadjusted 5,000-resample intervals remain archived, rather than being silently replaced.

## 4.3 Implementation and cost

Raw CRC/content hashes, exact views, pinned weights, fold heads, and selections are retained. Cached legacy features are reused only at matching content, times, preprocessing, and weights. Encoder timing uses M1 Pro/16 GB, MPS FP32, four CPU threads, batch one view, ten warm-ups and 50 measured runs on a fixed training input; CPU return synchronizes inference. CNN references were measured in a separate session before CNN training. These are host-implementation costs, not eyewear energy. Timing excludes ofline decoding and input preparation unless a stage is named explicitly.

## 5 Results

## 5.1 Readout, model scale, and lightweight baselines

DINO-S changes from 59.1% capture/9.0% false triggers with patch mean to 76.5%/10.3% with CLS. Both low-FPR diagnostics also improve (pAUC20 0.635→0.732; TPR10 0.681→0.797). LeJEPA-S changes in the opposite direction for pAUC20 (0.603→0.533), despite identical aggregate selectedthreshold TP/FP counts. All 520 clip features difer between its readouts, 58 P1 clip decisions difer, and independent forwards reproduce each cache; the identical counts are not a fallback to patch mean. CLS is therefore not universally preferable, and readout efects cannot be generalized across families from the DINO-S gain.

DINO-S CLS has higher selected-threshold capture than DINO-B CLS but also more false triggers. DINO-B has higher pAUC20 and TPR10. This reversal demonstrates why a single calibrated operating point does not rank representations by discrimination. LeJEPA and DINO comparisons establish no JEPAspecific advantage; they also do not prove equivalence.

The 0.927M-parameter MobileNet encoder captures 97/132 with 53/388 false triggers, compared with LeJEPA-B’s 89/132 and 35/388. It is competitive in capture but is operating at a diferent FPR; its original-split pAUC20 is lower than both DINO CLS readouts. ResNet-18 is not uniformly stronger despite its larger parameter count. The trained TinyCNN’s P1 capture spans 22.0%–25.8%, with 13.9%–18.6% false triggers across its three seeds. Table 1 reports every seed separately; no best-seed result replaces the others. The limited training recipe and small sample do not establish the best attainable performance of compact CNNs.

Table 1: P1 at 50 ms/eight views. Capture and false triggers use validation-selected thresholds; pAUC and TPR10 are fold-wise test-ROC diagnostics, not calibrated deployment thresholds.
<table><tr><td>Representation</td><td>Targets captured</td><td>Non-target triggers</td><td>pAUC20</td><td>TPR10</td></tr><tr><td>LeJEPA-S patch</td><td>78/132 (59.1%)</td><td>47/388 (12.1%)</td><td>0.603</td><td>0.639</td></tr><tr><td>LeJEPA-B patch</td><td>89/132 (67.4%)</td><td>35/388 (9.0%)</td><td>0.670</td><td>0.709</td></tr><tr><td>DINO-S patch</td><td>78/132 (59.1%)</td><td>35/388 (9.0%)</td><td>0.635</td><td>0.681</td></tr><tr><td>DINO-B patch</td><td>88/132 (66.7%)</td><td>33/388 (8.5%)</td><td>0.682</td><td>0.703</td></tr><tr><td>LeJEPA-S CLS</td><td>78/132 (59.1%)</td><td>47/388 (12.1%)</td><td>0.533</td><td>0.566</td></tr><tr><td>LeJEPA-B CLS</td><td>87/132 (65.9%)</td><td>44/388 (11.3%)</td><td>0.657</td><td>0.682</td></tr><tr><td>DINO-S CLS</td><td>101/132 (76.5%)</td><td>40/388 (10.3%)</td><td>0.732</td><td>0.797</td></tr><tr><td>DINO-B CLS</td><td>98/132 (74.2%)</td><td>37/388 (9.5%)</td><td>0.746</td><td>0.809</td></tr><tr><td>MobileNetV3-Small</td><td>97/132 (73.5%)</td><td>53/388 (13.7%)</td><td>0.613</td><td>0.648</td></tr><tr><td>ResNet-18</td><td>85/132 (64.4%)</td><td>42/388 (10.8%)</td><td>0.569</td><td>0.630</td></tr><tr><td>TinyCNN, seed 20261022</td><td>31/132 (23.5%)</td><td>54/388 (13.9%)</td><td>0.142</td><td>0.143</td></tr><tr><td>TinyCNN, seed 20261023</td><td>34/132 (25.8%)</td><td>72/388 (18.6%)</td><td>0.132</td><td>0.121</td></tr><tr><td>TinyCNN, seed 20261024</td><td>29/132 (22.0%)</td><td>59/388 (15.2%)</td><td>0.145</td><td>0.140</td></tr></table>

## 5.2 Partition and threshold sensitivity

Across ten repartitions, DINO-B CLS has mean P1 pAUC20 0.738, DINO-S CLS 0.696, MobileNet 0.674, and LeJEPA-B patch mean 0.681. MobileNet’s original pAUC20 of 0.613 is below its repeated mean, illustrating that conclusions from one grouping can move. The repeated means summarize the fixed set of partitions; overlapping data prevent treating ten means as ten independent experiments. P2 is retained rather than selecting the more favorable policy. Within DINO, changing patch mean to CLS increases mean P1 pAUC20 by 0.075 for S and 0.080 for B across these repartitions, whereas S→B at patch mean increases it by 0.037. Both readout changes are positive in all ten partitions. This is a within-family descriptive comparison, not a universal dominance of readout over size: LeJEPA patch-mean S→B increases mean pAUC20 by 0.101, while its S readout change averages -0.038. The same participants are reused throughout.

LeJEPA-B patch-mean test FPR ranges from 6.2% to 12.4% between the 5th and 95th calibration percentiles; MobileNet spans 9.3%–15.7%. Only validation participants are resampled. Head scores and alpha stay fixed, so this captures one source of threshold instability rather than total pipeline uncertainty. More calibration data, cross-source backgrounds, and temporal controls are needed before a strict deployment false-alarm budget can be claimed.

## 5.3 Accumulation budgets and multiplicity

LeJEPA’s original P1 S→B capture diference is +8.3 points. Its family-adjusted interval is [-0.8, +19.0], spanning zero; the original marginal interval should not be read as robust multiple-comparison evidence. The 8→32-view P1 capture diference remains positive in its adjusted interval, but its actual false-trigger rate also increases and inputs are not held constant. P2 difers in its duration response. These are conditional development-data observations, not a general scaling law or proof of an optimal 20–50 ms range.

The full original grid remains in the supplement and machine-readable export. Gains are not universally monotone: LeJEPA-B at 50 ms changes from 71.2% capture/9.8% false triggers at 16 views to 70.5%/10.8% at 32. This observation does not identify blur or feature dilution as the cause. With eight 50 ms views, LeJEPA-B capture rises from 54.5% at 400 ms available history to 67.4% at 1600 ms, while DINO-B is higher at 800 ms (68.2%) than 1600 ms (66.7%). History also changes sample locations. Half-event retention gives LeJEPA-S mean capture 58.1%/false triggers 11.3% across three seeds, versus 59.1%/12.1% at full retention. Independently refitted retention conditions measure input-conditioned sensitivity, not robustness of one fixed deployed head or reduced sensor power.

## 5.4 Cost and evidence limits

Parameter count does not predict this host’s runtime ordering: MobileNet is smaller but slower per view than ResNet-18 in the measured setup. Cached aggregation/classification at 32 views takes about 0.014/0.024 ms for LeJEPA S/B, excluding new-view encoding. A streaming implementation need not recompute every historical embedding. Separate CPU medians for event accumulation (about 0.83–0.93 ms) and resize/pad (about 0.44 ms) must not be summed and called measured end-to-end latency. No timing establishes eye-worn power or thermal feasibility.

## 6 Discussion and Limitations

The practical finding is an evaluation dependency: readout and calibrated working points can change the apparent scale comparison. In these releases, DINO CLS is a strong readout, while LeJEPA-S CLS weakens low-FPR discrimination. A compact supervised encoder remains competitive. Any recommendation to compress a particular large JEPA model should first establish its advantage against such controls at an appropriate operating point.

This benchmark addresses an action-defined ofline candidate, not subjective recording value. It uses a selected eight-class, fixed-camera third-person subset with centered, action-containing clips and prior development exposure. Participant isolation addresses identity overlap inside a rotation but does not undo dataset adaptation. Repartitioning the same people cannot establish an independent-source result. Mean pooling also omits temporal order, and we have not compared full video inference or a strong event-native model.

(b) All ten dependent repartitions, P1  
Table 2: Repeated participant grouping: mean [minimum, maximum] normalized pAUC20 over all ten fixed repartitions. These dependent repeats use the same 47 development participants.
<table><tr><td>Representation</td><td>P1 pAUC20</td><td></td><td>P2 pAUC20</td></tr><tr><td>LeJEPA-S patch</td><td>0.580 [0.554, 0.635]</td><td></td><td>0.532 [0.483, 0.585]</td></tr><tr><td>LeJEPA-B patch</td><td>0.681 [0.626, 0.727]</td><td></td><td>0.654 [0.591, 0.718]</td></tr><tr><td>DINO-S patch</td><td>0.621 [0.587, 0.668]</td><td></td><td>0.574 [0.528, 0.636]</td></tr><tr><td>DINO-B patch</td><td>0.658 [0.625, 0.690]</td><td></td><td>0.641 [0.591, 0.676]</td></tr><tr><td>LeJEPA-S CLS</td><td>0.542 [0.491, 0.602]</td><td></td><td>0.531 [0.477, 0.582]</td></tr><tr><td>LeJEPA-B CLS</td><td>0.649 [0.603, 0.696]</td><td></td><td>0.632 [0.587, 0.680]</td></tr><tr><td>DINO-S CLS</td><td>0.696 [0.604, 0.751]</td><td></td><td>0.619 [0.573, 0.666]</td></tr><tr><td>DINO-B CLS</td><td>0.738 [0.686, 0.779]</td><td></td><td>0.706 [0.651, 0.747]</td></tr><tr><td>MobileNetV3-Small</td><td>0.674 [0.631, 0.715]</td><td></td><td>0.617 [0.540, 0.655]</td></tr><tr><td>ResNet-18</td><td>0.592 [0.515, 0.633]</td><td></td><td>0.556 [0.506, 0.614]</td></tr></table>

![](images/71f44ec0a0b08fb8a912c31162ec25c2454321983926bcdb4515c5b52a0dbb87.jpg)

![](images/270731803ee25c33a2c0077715825171e29feae2197d3a1a01fb6c9e9f8f49b4.jpg)  
Figure 1: Readout and representation sensitivity: original fold-wise P1 ROC and all ten repeated-group pAUC20 values. Points are dependent repartitions, not independent confidence samples.

The motivating wearable system remains unvalidated. There are no continuous idle backgrounds, annotated online trigger deadlines, synchronized IMU/audio, RGB wake-up measurements, or device power measurements. Concatenating action clips would impose artificial event prevalence and boundaries; we do not report it as real continuous false alarms per hour. Personal-interest labels, wearer motion, cross-source backgrounds, and hardware measurements are concrete next stages, rather than outcomes of this study.

## 7 Conclusion

Real event accumulations support useful ofline candidate discrimination, but the apparent benefit of model size depends on readout and operating-point reporting. Lightweight baselines, low-FPR curves, repeated participant grouping, and calibration sensitivity materially change the interpretation of one capture score. Our original accumulation-budget results remain policy dependent, with input evidence and compute coupled. The release preserves both original and post-review analyses and their uncertainty boundaries. These findings motivate better-controlled evaluation before extrapolating to personal memory triggers or all-day eyewear.

## References

[1] G. Gallego et al. “Event-based Vision: A Survey.” IEEE TPAMI, arXiv:1904.08405v3, 2020. Source.

[2] P. Bonazzi, J. Moosmann, A. Celik, P. Mayer, and M. Magno. “OpenGlass: Ultra-Low-Power On-Device AI Eyewear with Event-based Vision.” arXiv:2606.07431v2, 2026. Source.

[3] P. Bhattacharyya et al. “Helios: An extremely low power event-based gesture recognition for always-on smart eyewear.” arXiv:2407.05206v4, 2024. Source.

[4] Q. Wang et al. “DailyDVS-200: A Comprehensive Benchmark Dataset for Event-Based Action Recognition.” ECCV, 2024. Source.

Table 3: Validation-threshold sensitivity, P1: 5th / 50th / 95th percentiles (%) from 1,000 validation-participant bootstrap recalibrations, holding original heads and alpha fixed. These are conditional sensitivity summaries.
<table><tr><td>Representation</td><td></td><td>Capture (%) False triggers (%)</td></tr><tr><td> $\mathrm { L e J E P A  – S \ p a t c h }$ </td><td> $5 3 . 8 \ : / \ : 5 9 . 1 \ : / \ : 6 3 . 6$ </td><td> $1 0 . 1 \ / \ 1 2 . 1 \ / \ 1 4 . 7$ </td></tr><tr><td> $\mathrm { L e J E P A  – B \ p a t c h }$ </td><td> $5 1 . 5 \textrm { / } 6 4 . 4 \textrm { / } 7 3 . 5$ </td><td> $6 . 2 \textrm { / } 8 . 8 \textrm { / } 1 2 . 4$ </td></tr><tr><td>DINO-S CLS</td><td> $6 3 . 6 \textrm { / } 7 2 . 0 \textrm { / } 7 8 . 0$ </td><td> $6 . 2 \textrm { / 9 . 8 / 1 4 . 2 }$ </td></tr><tr><td>DINO-B CLS</td><td> $6 8 . 2 \textrm { / } 7 2 . 7 \textrm { / } 7 7 . 3$ </td><td> $6 . 4 { \overset { \cdot } { / } } 9 . 0 { \overset { \cdot } { / } } 1 2 . 4$ </td></tr><tr><td>MobileNetV3-Small</td><td> $6 0 . 6 \ \mathrm { ~ / ~ } 6 9 . 7 \ \mathrm { ~ / ~ } 7 5 . 8$ </td><td> $9 . 3 \ : / \ : 1 2 . 4 \ : \dot { / } \ : 1 5 . 7$ </td></tr></table>

Table 4: Primary-policy results at a validation-selected 10% non-target trigger budget. Test rates are realized rates, not matched-FPR comparisons.
<table><tr><td>Encoder</td><td>Window (ms)</td><td>Views</td><td>Targets captured</td><td>Non-targets triggered</td></tr><tr><td>LeJEPA-S</td><td>20</td><td>8</td><td>76/132 (57.6%)</td><td> $4 9 / 3 8 8 \ ( 1 2 . 6 \% )$ </td></tr><tr><td>LeJEPA-S</td><td>50</td><td>8</td><td>78/132 (59.1%)</td><td>47/388 (12.1%)</td></tr><tr><td>LeJEPA-B</td><td>50</td><td>8</td><td>89/132 (67.4%)</td><td>35/388 (9.0%)</td></tr><tr><td>LeJEPA-S</td><td>50</td><td>32</td><td>95/132 (72.0%)</td><td>56/388 (14.4%)</td></tr></table>

[5] R. Fan and W. Hao. “SMV-EAR: Bring Spatiotemporal Multi-View Representation Learning into Eficient Event-Based Action Recognition.” arXiv:2601.17391v1, 2026. Source.

[6] M. Assran et al. “Self-Supervised Learning From Images With a Joint-Embedding Predictive Architecture.” CVPR, pp. 15619–15629, 2023. Source.

[7] A. Bardes et al. “Revisiting Feature Prediction for Learning Visual Representations from Video.” arXiv:2404.08471, 2024. Source.

[8] R. Balestriero and Y. LeCun. “LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics.” arXiv:2511.08544v3, 2025. Source.

[9] M. Caron et al. “Emerging Properties in Self-Supervised Vision Transformers.” arXiv:2104.14294v2, 2021. Source.

[10] Open-Knowledge-AI. LeJEPA ViT-S/16 ImageNet-1K release, revision cc7022877d51494709ef398d437fb8619349e0f9. Pinned release.

[11] Open-Knowledge-AI. LeJEPA ViT-B/16 ImageNet-1K release, revision 5dd71e2fc9a8e97d84c047f3a8cd5bf19f199fa. Pinned release.

[12] Open-Knowledge-AI. DINO ViT-S/16 ImageNet-1K release, revision fa32cfe8bd41d44462e99336601d9f1ec5bf6ccc. Pinned release.

[13] Open-Knowledge-AI. DINO ViT-B/16 ImageNet-1K release, revision d005b9ab18098bc8af7688f9b66afbe73019bc9. Pinned release.

[14] L. Mur-Labadia et al. “V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning.” arXiv:2603.14482v3, 2026. Source.

[15] A. Howard et al. “Searching for MobileNetV3.” ICCV, 2019. Source.

[16] K. He, X. Zhang, S. Ren, and J. Sun. “Deep Residual Learning for Image Recognition.” CVPR, 2016. Source.

## Reproducibility

The accompanying local release candidate includes the original 148-condition/888-operating-point export, per-clip outputs, the primary frozen features, standalone downstream refitting and verification code, grouped splits, model provenance, training recipes, and post-review analyses. The original matrix comprises 296 policy-condition pairs and 1,480 fold-specific heads; new CNN baselines and repeated grouping are separate additions. Raw DailyDVS recordings and third-party weights are obtained under their source terms and are not redistributed. The release is prepared locally; no public repository or preprint identifier is claimed.

## AI assistance

AI tools assisted implementation, experiment orchestration, checking, and drafting. Binary targets are deterministic mappings of dataset labels. Reported scores are produced by the documented encoders and supervised classifiers, not language-model semantic judgments. The author is responsible for the content.

Table 5: Original paired contrasts, percentage-point diferences with post-review Bonferroni family-12 bootstrap intervals (50,000 participant resamples). All intervals condition on fixed original folds and fitted heads.
<table><tr><td>Policy</td><td>Change</td><td>Capture difference</td><td>False-trigger difference</td><td></td></tr><tr><td>P1</td><td> $\mathrm { S } {  } \mathrm { B } , 5 0 ~ \mathrm { m s } / 8$  views</td><td> $+ 8 . 3 \ [ - 0 . 8 , + 1 9 . 0 ]$ </td><td></td><td> $- 3 . 1 \ [ - 8 . 7 , \ + 2 . 5 ]$ </td></tr><tr><td>P1</td><td> $2 0 {  } 5 0 ~ \mathrm { m s } , \thinspace \mathrm { S } / 8 ~ \mathrm { v i e w s }$ </td><td> $+ 1 . 5 \ [ - 7 . 3 , + 9 . 8 ]$ </td><td></td><td> $- 0 . 5 \ [ - 6 . 9 , \ + 5 . 2 ]$ </td></tr><tr><td>P1</td><td> $8 {  } 3 2 { \mathrm { ~ v i e w s , ~ S / 5 0 ~ m s } }$ </td><td> $+ 1 2 . 9 \ [ + 2 . 1 , + 2 6 . 6 ]$ </td><td></td><td> $+ 2 . 3 \ [ - 1 . 3 , + 6 . 8 ]$ </td></tr><tr><td>P2</td><td> $\mathrm { S {  } B , \ 5 0 \ m s / 8 \ v i e w s }$ </td><td> $+ 1 1 . 2 \ [ + 1 . 3 , + 2 1 . 0 ]$ </td><td></td><td> $- 1 . 8 \ [ - 6 . 6 , \ : + 2 . 8 ]$ </td></tr><tr><td>P2</td><td> $2 0 {  } 5 0 ~ \mathrm { m s } , \mathrm { S } / 8$  views</td><td> $+ 8 . 4 \ [ + 0 . 5 , + 1 7 . 3 ]$ </td><td></td><td> $+ 0 . 9 \ [ - 4 . 7 , \ + 6 . 7 ]$ </td></tr><tr><td>P2</td><td> $8 {  } 3 2 \ \mathrm { v i e w s } , \ \mathrm { S } / 5 0$  ms</td><td> $+ 9 . 0 \ [ - 0 . 6 , + 1 9 . 8 ]$ </td><td></td><td> $+ 1 . 2 \ [ - 3 . 6 , \ : + 6 . 1 ]$ </td></tr></table>

Table 6: Local warm M1 Pro/MPS FP32 median timings. One-view timing includes encoding, normalization of the embedding and CPU return. Recompute includes the classifier, but excludes event rendering and image normalization. CNN timings are a separate-session reference.
<table><tr><td>Encoder</td><td>Encoder parameters</td><td>One view (ms)</td><td>8-view recompute (ms)</td><td>32-view recompute (ms)</td></tr><tr><td>LeJEPA-S</td><td>21,665,664</td><td>18.54</td><td>146.47</td><td>413.98</td></tr><tr><td>LeJEPA-B</td><td>85,798,656</td><td>22.96</td><td>196.09</td><td>768.16</td></tr><tr><td>MobileNetV3-Small</td><td>927,008</td><td>10.65</td><td>not measured</td><td>not measured</td></tr><tr><td>ResNet-18</td><td>11,176,512</td><td>8.23</td><td>not measured</td><td>not measured</td></tr></table>

## Supplementary Material

This supplement separates the original fixed evaluation from explicitly post-review extensions. Rates describe ofline clip decisions, not recording value, online trigger frequency, or device energy.

## S1. Data and grouped splits

Table S1: Participant-disjoint fold composition.
<table><tr><td>Test fold</td><td>Test people</td><td>Test clips</td><td>P1 targets</td><td>P2 targets</td><td>Validation fold</td></tr><tr><td>0</td><td>10</td><td>111</td><td>28</td><td>38</td><td>1</td></tr><tr><td>1</td><td>10</td><td>111</td><td>28</td><td>38</td><td>2</td></tr><tr><td>2</td><td>9</td><td>95</td><td>24</td><td>32</td><td>3</td></tr><tr><td>3</td><td>9</td><td>97</td><td>25</td><td>33</td><td>4</td></tr><tr><td>4</td><td>9</td><td>106</td><td>27</td><td>37</td><td>0</td></tr></table>

Remaining three folds form the training group. No participant crosses the training, validation and test groups within a fold. All people have prior research exposure; these folds are not a fresh confirmation set.

## S2. Complete primary-setting budget points

Each budget is a validation constraint. The realized test non-target rate need not satisfy it. Each entry is capture / non-target trigger rate (%).

Table S2: Primary-setting validation budgets: capture / realized false triggers.
<table><tr><td>Encoder (50 ms, 8 views)</td><td>Policy</td><td>5% validation budget</td><td>10% validation budget</td><td>20% validation budget</td></tr><tr><td>lejepa_s</td><td>P1</td><td>52.3 / 9.8</td><td>59.1 / 12.1</td><td>68.9 / 19.8</td></tr><tr><td>lejepa_s</td><td>P2</td><td>51.7 / 8.8</td><td>57.9 / 11.4</td><td>67.4 / 17.8</td></tr><tr><td>lejepa_b</td><td>P1</td><td>50.8 / 4.9</td><td>67.4 / 9.0</td><td>78.8 / 16.8</td></tr><tr><td>lejepa_b</td><td>P2</td><td>59.6 / 5.6</td><td>69.1 / 9.6</td><td>80.3 / 17.5</td></tr><tr><td>dino_s</td><td>P1</td><td>41.7 / 3.6</td><td>59.1 / 9.0</td><td>71.2 / 18.3</td></tr><tr><td>dino_s</td><td>P2</td><td>42.1 / 5.3</td><td>56.7 / 8.8</td><td>71.3 / 20.2</td></tr><tr><td>dino_b</td><td>P1</td><td>54.5 / 2.8</td><td>66.7 / 8.5</td><td>84.1 / 20.9</td></tr><tr><td>dino_b</td><td>P2</td><td>58.4 / 4.1</td><td>66.3 / 8.8</td><td>79.8 / 21.3</td></tr><tr><td>vjepa_b</td><td>P1</td><td>55.3 / 4.6</td><td>63.6 / 7.0</td><td>72.7 / 14.7</td></tr><tr><td>vjepa_b</td><td>P2</td><td>62.4 / 11.4</td><td>67.4 / 14.3</td><td>72.5 / 17.3</td></tr></table>

## S3. Class-level errors at the predeclared primary setting

P1 target classes are handshake (71) and fall (109); every other class is a non-target by policy definition. Reported counts are selected clips divided by clips available in that class. This is recall for target classes and false triggering for non-target classes.

Table S3: Class-level selections under P1.
<table><tr><td>Action (class ID)</td><td>Clips</td><td>LeJEPA-S selected</td><td>LeJEPA-B selected</td><td>DINO-S selected</td><td>DINO-B selected</td></tr><tr><td>Drink (1)</td><td>70</td><td>8</td><td>7</td><td>4</td><td>5</td></tr><tr><td>Take from bag (8)</td><td>70</td><td>5</td><td>4</td><td>4</td><td>5</td></tr><tr><td>Phone use (31)</td><td>70</td><td></td><td>5</td><td>9</td><td>5</td></tr><tr><td>Put into bag (62)</td><td>70</td><td>4 9</td><td>6</td><td>6</td><td>7</td></tr><tr><td>Handshake (target) (71)</td><td>70</td><td>40</td><td>50</td><td>46</td><td>50</td></tr><tr><td>Pick up object (100)</td><td>62</td><td>18</td><td>12</td><td>9</td><td>11</td></tr><tr><td>Fall (target) (109)</td><td>62</td><td>38</td><td>39</td><td>32</td><td>38</td></tr><tr><td>Open door (158)</td><td>46</td><td>3</td><td>1</td><td>3</td><td>0</td></tr></table>

The picking-up action is retained as a non-target for these fixed policies, rather than excluded after errors are observed. Class counts are descriptive and were not used to change the target policy or select hyperparameters.

## S4. V-JEPA cross-family reference and CLS sensitivity

V-JEPA denotes the oficial V-JEPA 2.1 distilled ViT-B EMA encoder (86,833,152 parameters), evaluated at 224 pixels using its image path: one temporal image dimension, independent per-view encoding, mean patch tokens, per-view L2 normalization, and mean aggregation. Its checkpoint has nominal 384-pixel configuration; this reference uses the same 224-pixel input size as the main study. It is neither a full temporal-video evaluation nor a controlled objective-only comparison. CLS readout is a predeclared sensitivity check, not a replacement selected from test outcomes.

Table S4: V-JEPA image-path reference and CLS sensitivity.
<table><tr><td>Encoder</td><td>Readout</td><td>W (ms)</td><td>Views</td><td>P1 capture / false triggers (%)</td><td>P2 capture / false triggers (%)</td></tr><tr><td>vjepa_b</td><td>patch_mean</td><td>20</td><td>1</td><td>53.8 / 8.8</td><td>49.4 / 9.9</td></tr><tr><td>vjepa_b</td><td>patch_mean</td><td>20</td><td>8</td><td>60.6 / 8.2</td><td>68.0 / 11.7</td></tr><tr><td>vjepa_b</td><td>patch_mean</td><td>20</td><td>32</td><td>66.7 / 9.3</td><td>68.5 / 15.2</td></tr><tr><td>vjepa_b</td><td>patch_mean</td><td>50</td><td>1</td><td>49.2 / 9.8</td><td>38.2 / 9.6</td></tr><tr><td>vjepa_b</td><td>patch_mean</td><td>50</td><td>8</td><td>63.6 / 7.0</td><td>67.4 / 14.3</td></tr><tr><td>vjepa_b</td><td>patch_mean</td><td>50</td><td>32</td><td>70.5 / 10.6</td><td>67.4 / 12.9</td></tr><tr><td>lejepa_s</td><td>cls</td><td>50</td><td>8</td><td>59.1 / 12.1</td><td>53.4 / 9.9</td></tr><tr><td>lejepa_b</td><td>cls</td><td>50</td><td>8</td><td>65.9 / 11.3</td><td>62.9 / 8.5</td></tr><tr><td>dino_s</td><td>cls</td><td>50</td><td>8</td><td>76.5 / 10.3</td><td>66.3 / 10.2</td></tr><tr><td>dino_b</td><td>cls</td><td>50</td><td>8</td><td>74.2 / 9.5</td><td>73.6 / 9.1</td></tr></table>

## S5. Full grid and audit scope

The accompanying complete results.en.csv contains 888 budgeted operating points: 148 distinct input conditions, two target policies, and three validation budgets. These are frozen-encoder evaluations with 1,480 fitted foldspecific classifier heads, not 148 pretrained-model training runs. The original matrix contains 100 main-model inputs and 48 auxiliary inputs; every condition is present.

The full grid includes native 20/30/50 ms accumulations, 1/2/4/8/16/32 views, 400/800/1600 ms available histories, fixed-seed 25/50% real-event retention, simple controls, train-only PCA128, V-JEPA reference, and CLS checks. Retention conditions fit their own classifiers and thresholds; they do not measure corruption of a fixed deployed classifier.

![](images/dfd370445d70e60475a9e90ae3c33aa08e2b2e3bf36ff8c5efd286f495b00d61.jpg)  
Figure S1: Full P2 budget curves.

## S5.1. Original simple and dimension controls

The count baseline averages per-window log(1 + event count), yielding one feature. A 131-dimensional baseline combines that count, positive-polarity proportion, occupied-pixel proportion, and normalized 8×8 spatial counts for each polarity, followed by view averaging. Both baselines use all 18 core input configurations and the same downstream selection procedure.

A dimension control fits PCA to the raw aggregated training embeddings only, retains 128 components, and subsequently fits the same standardized ridge classifier. Validation and test participants are not used to fit PCA. This controls downstream dimension, but also changes feature geometry and is not a complete isolation of encoder capacity.

Table S5: Original simple-statistic and dimension controls.
<table><tr><td>Representation</td><td>P1 capture</td><td>P1 false triggers</td><td>P2 capture</td><td>P2 false triggers</td></tr><tr><td>Event count, 1-D</td><td>12.1%</td><td>10.3%</td><td>8.4%</td><td>10.2%</td></tr><tr><td>Event statistics, 131-D</td><td>25.0%</td><td>14.7%</td><td>20.2%</td><td>11.1%</td></tr><tr><td>LeJEPA-S</td><td>59.1%</td><td>12.1%</td><td>57.9%</td><td>11.4%</td></tr><tr><td>LeJEPA-B</td><td>67.4%</td><td>9.0%</td><td>69.1%</td><td>9.6%</td></tr><tr><td>DINO-S</td><td>59.1%</td><td>9.0%</td><td>56.7%</td><td>8.8%</td></tr><tr><td>DINO-B</td><td>66.7%</td><td>8.5%</td><td>66.3%</td><td>8.8%</td></tr><tr><td>LeJEPA-S, PCA128</td><td>50.8%</td><td>8.5%</td><td>50.6%</td><td>12.9%</td></tr><tr><td>LeJEPA-B, PCA128</td><td>60.6%</td><td>14.2%</td><td>59.6%</td><td>9.9%</td></tr></table>

## S6. Original history and event-retention controls

Table S6: Fixed-eight-view available-history control.
<table><tr><td>Encoder</td><td>400 ms history</td><td>800 ms history</td><td>1600 ms history</td></tr><tr><td>LeJEPA-S</td><td>60.6% 12.9%</td><td>56.8% / 10.8%</td><td>59.1% 12.1%</td></tr><tr><td>LeJEPA-B</td><td>54.5% 10.1%</td><td>60.6% / 9.0%</td><td>67.4% / 9.0%</td></tr><tr><td>DINO-S</td><td>54.5% / 8.5%</td><td>62.9% / 9.5%</td><td>59.1% / 9.0%</td></tr><tr><td>DINO-B</td><td>54.5% / 9.3%</td><td>68.2% / 9.3%</td><td>66.7% / 8.5%</td></tr></table>

At 50 ms, LeJEPA-B capture increases from 54.5% at 400 ms history to 67.4% at 1600 ms, whereas DINO-B reaches 68.2% at 800 ms and 66.7% at 1600 ms. These point estimates do not define an optimal horizon or a statistically established decrease. The complete control includes both 20 and 50 ms windows for all four encoders. Changing available history also changes temporal sampling positions, even at fixed view count. These are ofline interval-prefix comparisons, not measured delays from action onset.

![](images/d3128fefc9c32f5f9f76a01c2c51e459195a2fb5a7d3dcb35789f438eee8eac9.jpg)  
Figure S2: Available-history sensitivity at fixed view count.

Table S7: Actual-event retention with all thinning seeds.
<table><tr><td>Encoder</td><td>Retention</td><td></td><td>Capture: mean [range]</td><td>Non-target triggers: mean [range]</td></tr><tr><td>LeJEPA-S</td><td>25%</td><td></td><td>52.5% [50.8%, 54.5%]</td><td>10.7% [10.3%, 10.8%]</td></tr><tr><td>LeJEPA-S</td><td>50%</td><td>58.1%</td><td>[56.1%, 60.6%]</td><td>11.3% [8.8%, 13.4%]</td></tr><tr><td>LeJEPA-S</td><td>100%</td><td>59.1%</td><td>[59.1%, 59.1%]</td><td>12.1% [12.1%, 12.1%]</td></tr><tr><td>LeJEPA-B</td><td>25%</td><td>60.1%</td><td>[59.8%, 60.6%]</td><td>8.9% [8.8%, 9.0%]</td></tr><tr><td>LeJEPA-B</td><td>50%</td><td>63.9%</td><td>[57.6%, 68.9%]</td><td>9.5% [9.0%, 10.1%]</td></tr><tr><td>LeJEPA-B</td><td>100%</td><td>67.4%</td><td>[67.4%, 67.4%]</td><td>9.0% [9.0%, 9.0%]</td></tr></table>

Half retention gives LeJEPA-S a mean capture of 58.1% and mean non-target trigger rate of 11.3%, compared with 59.1% and 12.1% for all events. LeJEPA-B half-retention capture ranges from 57.6% to 68.9% across seeds, cautioning against a single favorable draw. These descriptive ranges do not establish equivalence with full retention. Each retention/seed setting fits its own training-only head and validation threshold. This measures input-conditioned pipeline sensitivity, not test corruption of a fixed classifier. The reduced fractions retain floor(r\*n) real events per original window. Three seeds are retained without selecting the best or averaging their predictions into a more expensive ensemble. Results for P2 and all 5/20% validation budgets are preserved in the full export.

P1: fixed 50 ms / eight views / 1.6 s history Error bars: range over three seeds; no ensembling  
![](images/26ba911c637fb2265f8959da972a1a9f04586c112d6cfe652ffededbe8b6f5f8.jpg)

![](images/64cd871ea0eba959d41730d3ff7023148d35466a4d3c9b7006e74af472057fdd.jpg)  
Figure S3: Actual-event retention sensitivity.

## S7. Complete new baseline and readout operating points

Table S8: Original grouping: all new baselines and readouts.
<table><tr><td>Representation</td><td>Policy</td><td>5% budget</td><td>10% budget</td><td></td><td>20% budget</td><td>pAUC20</td><td>TPR10</td></tr><tr><td>LeJEPA-S patch</td><td>P1</td><td>52.3%</td><td>/9.8% 59.1%</td><td>/12.1%</td><td>68.9% / 19.8%</td><td>0.603</td><td>0.639</td></tr><tr><td>LeJEPA-S patch</td><td>P2</td><td>51.7%</td><td>8.8% 57.9%</td><td>/ 11.4%</td><td>67.4% / 17.8%</td><td>0.555</td><td>0.601</td></tr><tr><td>LeJEPA-B patch</td><td>P1</td><td>50.8%</td><td>4.9% 67.4%</td><td>9.0%</td><td>78.8% 16.8%</td><td>0.670</td><td>0.709</td></tr><tr><td>LeJEPA-B patch</td><td>P2</td><td>59.6%</td><td>5.6% 69.1%</td><td>9.6%</td><td>80.3% / 17.5%</td><td>0.687</td><td>0.713</td></tr><tr><td>DINO-S patch</td><td>P1</td><td>41.7%</td><td>3.6% 59.1%</td><td>9.0%</td><td>71.2% 18.3%</td><td>0.635</td><td>0.681</td></tr><tr><td>DINO-S patch</td><td>P2</td><td>42.1%</td><td>5.3% 56.7%</td><td>8.8%</td><td>71.3% 20.2%</td><td>0.582</td><td>0.615</td></tr><tr><td>DINO-B patch</td><td>P1</td><td>54.5%</td><td>2.8% 66.7%</td><td>8.5%</td><td>84.1% 20.9%</td><td>0.682</td><td>0.703</td></tr><tr><td>DINO-B patch</td><td>P2</td><td>58.4%</td><td>4.1% 66.3%</td><td>8.8%</td><td>79.8% 21.3%</td><td>0.677</td><td>0.725</td></tr><tr><td>LeJEPA-S CLS</td><td>P1</td><td>35.6%</td><td>4.9% 59.1%</td><td>/ 12.1%</td><td>72.0% 23.5%</td><td>0.533</td><td>0.566</td></tr><tr><td>LeJEPA-S CLS</td><td>P2</td><td>43.3%</td><td>4.4% 53.4%</td><td>/ 9.9%</td><td>65.2% 16.4%</td><td>0.552</td><td>0.592</td></tr><tr><td>LeJEPA-B CLS</td><td>P1</td><td>51.5%</td><td>4.9% 65.9%</td><td>/11.3%</td><td>78.8% 18.8%</td><td>0.657</td><td>0.682</td></tr><tr><td>LeJEPA-B CLS</td><td>P2</td><td>47.8%</td><td>6.1% 62.9%</td><td>/ 8.5%</td><td>72.5% 12.9%</td><td>0.671</td><td>0.689</td></tr><tr><td>DINO-S CLS</td><td>P1</td><td>58.3%</td><td>1.8% 76.5%</td><td>/10.3%</td><td>81.1% /16.0%</td><td>0.732</td><td>0.797</td></tr><tr><td>DINO-S CLS</td><td>P2</td><td>51.7%</td><td>4.4% 66.3%</td><td>10.2%</td><td>77.5% 19.3%</td><td>0.650</td><td>0.675</td></tr><tr><td>DINO-B CLS</td><td>P1</td><td>67.4%</td><td>4.6% 74.2%</td><td>/ 9.5%</td><td>85.6% 17.3%</td><td>0.746</td><td>0.809</td></tr><tr><td>DINO-B CLS</td><td>P2</td><td>65.7%</td><td>7.3% 73.6%</td><td>/ 9.1%</td><td>83.7% 21.9%</td><td>0.720</td><td>0.736</td></tr><tr><td>MobileNetV3-Small</td><td>P1</td><td>59.1%</td><td>8.5% 73.5%</td><td>/13.7%</td><td>81.1% 19.6%</td><td>0.613</td><td>0.648</td></tr><tr><td>MobileNetV3-Small</td><td>P2</td><td>55.1%</td><td>5.0% 65.7%</td><td>/ 11.7%</td><td>81.5% 21.9%</td><td>0.626</td><td>0.718</td></tr><tr><td>ResNet-18</td><td>P1</td><td>47.7%</td><td>5.4% 64.4%</td><td>/10.8%</td><td>75.0% 16.8%</td><td>0.569</td><td>0.630</td></tr><tr><td>ResNet-18</td><td>P2</td><td>49.4%</td><td>8.5% 59.0%</td><td>/12.0%</td><td>66.9% / 17.8%</td><td>0.528</td><td>0.563</td></tr><tr><td>TinyCNN 20261022</td><td>P1</td><td>11.4%</td><td>7.0% 23.5%</td><td>/13.9%</td><td>33.3% 24.7%</td><td>0.142</td><td>0.143</td></tr><tr><td>TinyCNN 20261022</td><td>P2</td><td>7.9%</td><td>4.7% 21.9%</td><td>/13.2%</td><td>28.7% 17.0%</td><td>0.164</td><td>0.191</td></tr><tr><td>TinyCNN 20261023</td><td>P1</td><td>13.6%</td><td>10.3% 25.8%</td><td>/18.6%</td><td>35.6% 25.8%</td><td>0.132</td><td>0.121</td></tr><tr><td>TinyCNN 20261023</td><td>P2</td><td>11.2%</td><td>9.6% 26.4%</td><td>/19.0%</td><td>39.9% 29.2%</td><td>0.129</td><td>0.123</td></tr><tr><td>TinyCNN 20261024</td><td>P1</td><td>9.1%</td><td>/ 7.7% 22.0%</td><td>/15.2%</td><td>29.5% 21.4%</td><td>0.145</td><td>0.140</td></tr><tr><td>TinyCNN 20261024</td><td>P2</td><td>12.9% / 9.9%</td><td>23.6%</td><td>/ 13.2%</td><td>41.0% / 24.0%</td><td>0.163</td><td>0.147</td></tr></table>

## S8. Full threshold-recalibration sensitivity

Table S9: Conditional validation-threshold sensitivity.
<table><tr><td>Representation</td><td>Policy</td><td>Capture</td><td>False triggers</td></tr><tr><td>LeJEPA-S patch</td><td>P1</td><td>53.8 59.1</td><td>63.6 10.1 12.1</td></tr><tr><td>LeJEPA-S patch</td><td>P2</td><td>49.4 57.9</td><td>/14.7 65.2 9.1 / 12.3</td></tr><tr><td>LeJEPA-B patch</td><td>P1</td><td>51.5 64.4</td><td>/ 17.0 73.5 6.2 8.8 / 12.4</td></tr><tr><td>LeJEPA-B patch</td><td>P2</td><td>60.1 66.9</td><td>72.5 5.8 9.1 /13.5</td></tr><tr><td>DINO-S patch</td><td>P1</td><td>44.7 55.7</td><td>65.2 5.2 8.5 /15.2</td></tr><tr><td>DINO-S patch</td><td>P2</td><td>45.5 54.5</td><td>63.5 6.7 10.2 /16.7</td></tr><tr><td>DINO-B patch</td><td>P1</td><td>57.6 67.4 75.8</td><td>4.9 8.8 /13.4</td></tr><tr><td>DINO-B patch</td><td>P2</td><td>58.4 66.3</td><td>73.6 5.6 9.1 / 15.5</td></tr><tr><td>LeJEPA-S CLS</td><td>P1</td><td>43.1 55.3</td><td>66.7 7.0 12.4 /17.8</td></tr><tr><td>LeJEPA-S CLS</td><td>P2</td><td>43.2 52.8</td><td>61.8 5.0 10.5 /18.7</td></tr><tr><td>LeJEPA-B CLS</td><td>P1 53.0</td><td>63.6 74.2</td><td>5.9 10.3 / 14.7</td></tr><tr><td>LeJEPA-B CLS</td><td>P2 55.1</td><td>62.9 71.3</td><td>6.1 9.1 /13.2</td></tr><tr><td>DINO-S CLS</td><td>P1 63.6</td><td>72.0 78.0</td><td>6.2 9.8 / 14.2</td></tr><tr><td>DINO-S CLS</td><td>P2</td><td>54.5 63.5 / 71.9</td><td>5.3 / 9.9 / 15.2</td></tr><tr><td>DINO-B CLS</td><td>P1 68.2</td><td>72.7 77.3</td><td>6.4 9.0 /12.4</td></tr><tr><td>DINO-B CLS</td><td>P2 67.4</td><td>74.2 79.8</td><td>7.6 / 10.8 / 17.3</td></tr><tr><td>MobileNetV3-Small</td><td>P1 60.6</td><td>69.7 /75.8</td><td>9.3 / 12.4 / 15.7</td></tr><tr><td>MobileNetV3-Small</td><td>P2 55.6</td><td>65.7 73.6</td><td>7.0 11.7 / 17.0</td></tr><tr><td>ResNet-18</td><td>P1</td><td>51.5 61.4 68.2</td><td>7.7/ 10.3 / 12.9</td></tr><tr><td>ResNet-18</td><td>P2</td><td>47.8 56.7 / 62.9</td><td>8.5 / 11.7 / 14.9</td></tr></table>

## S9. Training, reproduction, and scope

All TinyCNN seeds, fold-specific validation-selected epochs, class weights, checkpoint hashes, epoch histories, and per-clip scores are retained. Input caching was checked for exact float32 equality without changing training or validation. All ten repartition seeds and their 1,000 fitted heads are retained.

The portable package verifies archived scores, refits primary frozen features, and supplies commoninput/CNN/ViT primary-extraction tools. Authorized raw data, oficial weights, and pinned reviewed Transformer source are required separately. This is not a raw-data reproduction of all 148 historical conditions. Raw recordings and third-party weights/source are excluded from redistribution.