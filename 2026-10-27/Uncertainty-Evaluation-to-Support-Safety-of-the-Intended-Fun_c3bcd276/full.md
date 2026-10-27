# Uncertainty Evaluation to Support Safety of the Intended Functionality Analysis for Determining the Applicability of Performance Targets Across Triggering Conditions in ML-Based LiDAR Object Detection

Milin Patel<sup>1</sup> <sup>[0000−0002−8357−6018]</sup> and Rolf Jung<sup>2</sup> <sup>[0000−0002−0366−4844]</sup>

<sup>1</sup> Institute for Driver Assistance and Connected Mobility,   
Kempten University of Applied Sciences, Benningen, Germany <sup>2</sup> Faculty of Computer Science,   
Kempten University of Applied Sciences, Kempten, Germany {milin.patel, rolf.jung}@hs-kempten.de

Abstract. For safety of the intended functionality (SOTIF) analysis according to ISO 21448, a machine learning object detector for light detection and ranging (LiDAR) point clouds can meet a performance target, a limit on the mean per-frame share of wrong detections among accepted detections, by rejecting detections above an uncertainty threshold. A threshold selected on data of one condition carries a statistical guarantee only for data distributed like its selection data, whereas SOTIF analysis covers every triggering condition. This paper presents an uncertainty evaluation in which the threshold is transferred into a triggering condition, a test decides at a stated error probability whether the target is applicable there, not applicable or not determined, and a lower bound counts the independent units an applicable outcome requires. For PointPillars ensembles on KITTI and nuScenes, counted in recorded sequences, as the dependence between frames requires, neither dataset held enough independent units for any target limit, and no transfer had a not applicable outcome. Counted in frames, the same data gave not applicable outcomes at limits fixed in advance after transfers into far distance, into rain and between two separately trained detectors, thresholds that met the target limit after a transfer rejected most correct detections, and in rain the area under the receiver operating characteristic curve did not decrease. A performance target met under one condition therefore counts as evidence under a triggering condition only after a test there, reported with its share of accepted correct detections and its number of independent recorded sequences.

Keywords: Safety of the intended functionality · Performance targets · Triggering conditions · Uncertainty thresholds · Learn then Test · LiDAR object detection

## 1 Introduction

An automated vehicle perceives its surroundings with object detectors, among them detectors for point clouds from light detection and ranging (LiDAR) sensors. A detector based on machine learning (ML) outputs a three-dimensional box, called a detection, for each object that it recognises in a frame, one point cloud of the sensor. The motion planner, which plans the path and speed of the vehicle, acts on these detections. This paper considers detections of cars, the object class with the most annotations in the KITTI and nuScenes datasets used here [1,2]. A detection is wrong if it does not match a car marked by the annotators of the dataset (an annotated car), for example because no object lies in its box or because the box misses the position or size of the car. A wrong detection can make the vehicle brake or steer for an object that does not exist, or misjudge the position of one that does.

ISO 21448, the standard on the safety of the intended functionality (SOTIF), requires that the residual risk of a driving function meets acceptance criteria at vehicle level [3]. Performance targets for system elements, for example a limit on how often a sensor reports an object that does not exist, can be derived from these criteria. A detector can meet such a target by rejecting uncertain detections. In this paper, the detector is an ensemble of separately trained networks, called members, and two indicators compute the uncertainty of a detection from the outputs of the members. The spatial indicator $u _ { \mathrm { s } }$ grows as the members disagree more on the position and size of the box, and the existence indicator $u _ { \mathrm { e } }$ grows as their confidence that an object is present falls. A detection whose indicator value exceeds the uncertainty threshold � is rejected and does not reach the motion planner. The other detections are accepted. The performance target of this paper bounds thefalse acceptance proportion (FAP), the share of wrong detections among the accepted detections of a frame averaged over frames, by a limit �. A lower threshold rejects more wrong detections but also more correct ones, and every rejected correct detection is a car that the motion planner does not receive.

Learn then Test (LTT) selects a threshold on recorded data so that the FAP does not exceed � with probability at least 1 − �, where � is a chosen error probability [4]. LTT makes no assumption about the form of the data distribution, but it assumes that the recorded units, for example frames, are independent draws from one distribution. The guarantee therefore holds only for data distributed like the data on which the threshold was selected [5,6], with their mix of distances, weather and locations.

ISO 21448 describes a triggering condition (TC) as a specific condition of a scenario that initiates a system reaction contributing to a hazardous behaviour, and it requires the response of the system to TCs to be evaluated. Its example of a statistical argument (Clause C.6.3.2) assumes one environment and does not state what holds in another. ISO/PAS 8800 considers the case in which the safety requirements of an artificial intelligence system are fulfilled only for a limited part of the input space. It requires that such a case is identified and reported to the development of the encompassing system [7]. A SOTIF analysis therefore has to establish under which conditions a performance target is met. Whether a threshold selected under one condition meets the target under another has to be tested.

The conference paper extended here [8] reported the area under the receiver operating characteristic (ROC) curve (AUROC). AUROC measures how well an indicator separates correct from wrong detections (0.5 is chance level, 1 is perfect separation), not whether a threshold meets a limit. It is therefore not known whether AUROC shows a change of condition under which the threshold no longer meets the limit.

Recorded data can show that a limit is met only if they hold enough independent units (frames or recorded sequences). Frames of one recorded sequence, a continuous recording such as a nuScenes scene or a KITTI drive, are not independent of each other. The number of independent frames or recorded sequences needed to show that a limit is met sets the minimum amount of data to record per condition.

In the uncertainty evaluation of this paper, a statistical test, the applicability test, decides whether a performance target met under one condition is also met under another condition. A threshold is transferred when it is selected on recorded data of one condition, the condition of selection, and used unchanged on recorded data of another, the condition ofuse. For example, a threshold selected on the detections at near distance is transferred when it is used unchanged on the detections at far distance. At a stated error probability, the performance target is applicable in the condition of use if the data show that the FAP with the transferred threshold does not exceed the limit there, and not applicable if they show that it exceeds the limit. The applicability is not determined if the data show neither. The test examines whether the target is met under the condition of use, not whether it is required there. This paper takes the target as required under every condition of the operational design domain (ODD). Figure 1 shows, for two transfers, one recorded frame in which the transferred threshold accepts a wrong detection that the threshold selected in the condition of use rejects. Both frames were chosen to show this case. In the KITTI frame, an unannotated object under the box of the wrong detection cannot be excluded.

## 1.1 Research Questions

This paper examines three changes of condition in both directions: between near and far distance, between clear weather and rain, and between the KITTI and nuScenes datasets. A condition is near or far distance, clear weather or rain, or one dataset together with its trained detector, assigned to the car detections by the rules of Sect. 4.2. Far distance and rain are treated as potential TCs, and near distance and clear weather are not. Two datasets difer in the sensor, the location, the mix of objects and, here, the trained detector at once, so a change of dataset does not isolate a TC. It serves as an experimental control, against which the changes of distance and of weather can be compared. This paper answers three research questions (RQs).

RQ1 Does an uncertainty threshold that keeps the false acceptance proportion, the share of wrong detections among the accepted detections averaged over frames, within a required limit under the condition for which it was selected keep it within this limit under another condition, such as the potential triggering conditions far distance and rain?

RQ2 When an uncertainty threshold no longer keeps the false acceptance proportion within the required limit under another condition, which changes in the detections from the condition for which it was selected accompany this, and which of these changes does the area under the receiver operating characteristic curve (AUROC) show?

RQ3 How many independent recorded frames or sequences (continuous recordings, such as nuScenes scenes and KITTI drives) are needed to show that an uncertainty threshold keeps the false acceptance proportion within a required limit under a given condition, such as a potential triggering condition, and which limits can the KITTI and nuScenes datasets show to be met?

![](images/8c54f285df28de6e848869404e8da74aead21971c5c065235679d5e8a55ca00f.jpg)

(a) KITTI, distance, us  
(b) nuScenes, rain, ue  
![](images/9d988a7328ad8adb974a1bfc4cf8d299e8a2aadc13bd5214dbdc3629af0ff919.jpg)

![](images/327eb17f75a76c8eec11f82da35ca72cd7620e9aa8ea74db5459231712afa96a.jpg)

![](images/93e4f401fee98e073c443b3f29cf2b44d5b202336c17d8041db8e0cb2d7f64f0.jpg)

![](images/d4886d788c9618ad4c37c4b772d5d56c7f8239e4931202fd6ca3a2510b1c9c03.jpg)  
correct wrong rejected annotated car camera field of view near/far boundary thin: member boxes  
Fig. 1. A wrong detection that the transferred threshold accepts and the threshold selected in the condition of use rejects. (a) KITTI, far distance, spatial indicator $u _ { \mathrm { s } } ,$ threshold transferred from near distance. (b) nuScenes, rain, existence indicator $u _ { \mathrm { e } } ,$ threshold transferred from clear weather. Bottom strips: indicator value of each detection. Open markers: rejected by the transferred threshold. IoU: intersection over union. Both thresholds were selected at the exploratory limit 0.1<sup>†</sup>, which defines no performance target.

## 1.2 Contributions and Scope

1. The applicability test, which decides at a stated error probability whether a performance target met with an uncertainty threshold under one condition is applicable, not applicable or not determined under another, together with the measure that follows from each outcome in the SOTIF analysis and a lower bound on the number of independent units needed to show that the target is met (Sects. 3 and 6.1).

2. An evaluation of the test with ensembles of PointPillars networks on KITTI and nuScenes after changes of object distance and weather and, as a control, of dataset and detector, with the frame and with the recorded sequence as the independent unit, including the share of correct detections that each threshold with an applicable outcome accepts (RQ1, Sect. 5.1).

3. An identification of the changes in the detections that accompany each not applicable outcome, which shows that AUROC does not reveal the changes found in rain and between the datasets (RQ2, Sect. 5.2).

4. A determination of the limits that the frames and the recorded sequences of KITTI and nuScenes can show to be met (RQ3, Sect. 5.3).

The test combines known statistical elements: the LTT selection, a confidence interval computed on data not used for the selection, and the zero-failure bound on the number of units. The contribution lies in their use for the SOTIF question of a transfer across conditions and in the measured outcomes for a LiDAR detector. The scope is the object detector as one system element. This paper proposes no new uncertainty indicator, detector or threshold selection method. It derives no value of � from an acceptance criterion and does not estimate how often each condition occurs in operation, because both belong to the argument at vehicle level.

Relation to the conference paper. The conference paper [8] evaluated uncertainty indicators of an ensemble LiDAR detector for the analysis activities of ISO 21448 and applied a rule that rejects uncertain detections. It used point clouds from a driving simulator and ensemble outputs simulated by perturbation, not by trained networks, and it selected the thresholds of the rule on all frames, without separate data to test them. This paper uses ensembles trained on two recorded datasets and keeps apart the data that fix the candidate thresholds, select a threshold and test it. It selects thresholds with a finitesample guarantee and tests them in the condition of use. The test, the measurements and the results do not appear in the conference paper.

## 2 Related Work

## 2.1 Performance Targets in SOTIF Processes

ISO 21448 defines the acceptance criterion as a criterion representing the absence of an unreasonable level of risk and the validation target as a value used to argue that the criterion is met [3], and it uses performance targets for system elements without defining the term. Hofmeister and Kinalzyk trace acceptance criteria to performance targets in an ISO 21448 process [9], Madala et al. list the factors behind acceptance criteria and validation targets [10], and Reisgys et al. decompose quantitative risk acceptance criteria into verification and validation targets [11]. Kramer et al. identify and quantify hazardous scenarios [12], and Hoss et al. review the testing of object-based perception [13]. Argument patterns for SOTIF and for ML components [14,15] and the treatment of uncertainty in ML assurance [16] define where evidence for a system element enters a safety argument. None of these works examines whether a target met under one condition is also met under another.

## 2.2 Statistical Risk Control and Its Validity Under Distribution Shift

Selective classification lets a model abstain on some inputs and bounds the risk of the predictions that it makes [17]. Conformal risk control [18] bounds the expected value of a chosen error rate. Risk-controlling prediction sets [19] and LTT [4] keep this expected value below a limit with a stated probability for the number of samples at hand. All three methods select a parameter on calibration data held out for that purpose. Conformal selection controls the false discovery rate over a selected set [20]. These guarantees hold on average over the distribution of the calibration data. Selecting the parameter separately inside each subset of the data gives a guarantee per subset [21,22], which becomes uninformative when a subset holds few data points [23]. Under covariate shift, reweighting the data by the ratio of the two input densities restores the guarantee [5], and when the data are not exchangeable, the loss of the guarantee can still be bounded [6]. Conformal prediction and conformal risk control have been applied to camera-based object detection with calibration and test data from one dataset [24,25]. This paper uses LTT without extending it and tests, for a LiDAR detector, whether a threshold selected with LTT meets its limit after a transfer to another condition.

## 2.3 Uncertainty of LiDAR Object Detectors

The indicators used here are computed from the members of a deep ensemble, several separately trained networks [26]. The regression uncertainty of object detectors has been evaluated with proper scoring rules [27]. The calibration of predictive uncertainty degrades under dataset shift [28], and three-dimensional detectors trained in one country lose accuracy in another [29]. None of these works tests a threshold across conditions.

## 2.4 Condition-Specific Evidence for Perception

Salay et al. decompose a misperception rate over disjoint perception conditions by the law of total probability [30], and CUBICS weights Bayesian performance estimates per operating situation with the expected frequency of each situation [31]. Kumar et al. combine a conformal prediction set with an abstention threshold selected once on one calibration set and report the abstention rate for each corruption severity [32]. Lee et al. adapt the confidence threshold of a LiDAR detector to object distance [33]. Li et al. select the confidence threshold for each synthetic sensor degradation so that the share of wrong detections among the accepted ones stays below a limit [34]. Adee et al. model LiDAR performance limitations as a Bayesian network to search for TCs [35,36]. None of these works, and no other study in LiDAR object detection found in a documented search (supplementary material), tests whether a threshold selected under one condition meets its limit under another (Table 1).

## 3 Test of the Applicability of a Performance Target Across Conditions

The applicability test takes a threshold selected with a guarantee in the condition of selection � and uses it unchanged in the condition of use $c ^ { \prime }$ . From an interval of the FAP in $c ^ { \prime } .$ , it decides whether the performance target is applicable, not applicable or not determined (Fig. 2). It takes as given a trained detector with its uncertainty indicators, the limit � derived from the acceptance criterion at vehicle level and the error probability � chosen in the SOTIF analysis. Table 2 lists the main symbols.

Table 1. Works closest to the question whether a threshold selected under one condition meets its limit under another.
<table><tr><td rowspan=1 colspan=1>work</td><td rowspan=1 colspan=1>quantity that is set or estimated</td><td rowspan=1 colspan=1>set orestimatedpercondition</td><td rowspan=1 colspan=1>limit tested inanothercondition</td><td rowspan=1 colspan=1>finite-sampleguarantee</td></tr><tr><td rowspan=1 colspan=1>Salay [30]</td><td rowspan=1 colspan=1>misperception rate, estimated</td><td rowspan=1 colspan=1>yes</td><td rowspan=1 colspan=1>no</td><td rowspan=1 colspan=1>no</td></tr><tr><td rowspan=1 colspan=1>CUBICS [31]</td><td rowspan=1 colspan=1>performance per situation,Bayesian estimate weighted bysituation frequency</td><td rowspan=1 colspan=1>yes</td><td rowspan=1 colspan=1>no</td><td rowspan=1 colspan=1>no(Bayesian)</td></tr><tr><td rowspan=1 colspan=1>Kumar [32]</td><td rowspan=1 colspan=1>abstention threshold, selectedonce</td><td rowspan=1 colspan=1>no(reportedperseverity)</td><td rowspan=1 colspan=1>no</td><td rowspan=1 colspan=1>no (coverageof thepredictionset only)</td></tr><tr><td rowspan=1 colspan=1>Lee [33]</td><td rowspan=1 colspan=1>confidence threshold, adaptedto distance</td><td rowspan=1 colspan=1>yes</td><td rowspan=1 colspan=1>no</td><td rowspan=1 colspan=1>no</td></tr><tr><td rowspan=1 colspan=1>Li [34]</td><td rowspan=1 colspan=1>confidence threshold, persensor degradation</td><td rowspan=1 colspan=1>yes</td><td rowspan=1 colspan=1>no</td><td rowspan=1 colspan=1>no</td></tr><tr><td rowspan=1 colspan=1>this paper</td><td rowspan=1 colspan=1>uncertainty threshold, selectedwith LTT in each condition</td><td rowspan=1 colspan=1>yes</td><td rowspan=1 colspan=1>yes, tested inthe conditionof use</td><td rowspan=1 colspan=1>yes (LTT,independentunits)</td></tr></table>

## 3.1 Uncertainty Indicators and the False Acceptance Proportion

The detector is an ensemble of � members, networks of one architecture trained from diferent random seeds. A detection � merges the boxes $b _ { k }$ of the $m ( d )$ members that contribute a box to it $( 3 \leq m ( d ) \leq M$ , Sect. 4.1), and $s _ { k }$ is the confidence of member � for its box. The two indicators are taken from the conference paper [8], and Eq. (1) is adapted from it. With the intersection over union (IoU) of two rotated three-dimensional boxes, which is 1 for identical and 0 for disjoint boxes, the spatial indicator is

$$
u _ { \mathrm { s } } ( d ) = 1 - \frac { 2 } { m ( d ) ( m ( d ) - 1 ) } \sum _ { k < l } \mathrm { I o U } ( b _ { k } , b _ { l } ) ,\tag{1}
$$

where � and � run over the contributing members. The existence indicator is $u _ { \mathrm { e } } ( d ) =$ $\begin{array} { r } { 1 - \frac { 1 } { M } \sum _ { k = 1 } ^ { M } s _ { k } } \end{array}$ , with $s _ { k } = 0$ for a member that contributes no box.

A detection is correct $( y ( d ) = 1 )$ if one-to-one matching assigns it to an annotated car with an IoU of at least 0.5, and wrong $( y ( d ) = 0 )$ otherwise. A wrong detection can be a box where no object is, a box on an annotated car that misses the overlap criterion, a second box on a car already matched, or a box on an object of another class or without annotation. A detection is accepted if its indicator value $u ( d )$ , either $u _ { \mathrm { s } } ( d )$ or $u _ { \mathrm { e } } ( d )$ ， does not exceed the threshold $\tau ,$ and rejected otherwise. An accepted wrong detection is called a false acceptance, and a rejected correct detection is called withheld. For a condition �, the share of false acceptances among the accepted detections of � in frame � (the per-frame share) and its expectation over the frames of � (the FAP) are

$$
a _ { f } ( \tau ) = \frac { \big | \{ d \in f _ { c } : u ( d ) \leq \tau , y ( d ) = 0 \} \big | } { \big | \{ d \in f _ { c } : u ( d ) \leq \tau \} \big | } , \qquad A _ { c } ( \tau ) = \mathbb { E } \big [ a _ { f } ( \tau ) \mid f \in c \big ] ,\tag{2}
$$

![](images/8f4c515e2b21c1cd78eafa5b2f623d18379b16a97aa45f8598b99e40e28cab67.jpg)  
Fig. 2. The applicability test. (a) Threshold $\tau _ { c }$ selected in the condition of selection $^ { c , }$ for example near distance. (b) The same threshold in the condition of use $c ^ { \prime }$ , for example far distance. (c) Outcome in $c ^ { \prime }$ from the interval [�, �] of the FAP, or no threshold selected if the lowest candidate threshold fails in �. �: indicator value; $A _ { c } ( \tau )$ : FAP in �. The distributions are schematic.

where $f _ { c }$ denotes the car detections of condition � in frame $f ,$ a frame � belongs to � $( f \in c )$ if $f _ { c }$ is not empty, and a frame without accepted detections in $f _ { c }$ contributes $\boldsymbol { a } _ { f } = 0$ . The performance target is $A _ { c } ( \tau ) \leq \alpha$ , and $\alpha$ is called the limit. The estimate of $A _ { c }$ on a set of frames is written $\hat { A } _ { c }$ . In terms of object detection metrics, $a _ { f }$ is one minus the precision of the accepted detections of a frame, and $A _ { c }$ is the false discovery rate computed per frame and averaged over frames [4]. The FAP difers from the pooled share, the share of wrong detections among all accepted detections of a condition taken together. The per-frame form is used because LTT bounds a mean of per-unit values in [0, 1]. It gives every frame the same weight.

Because a frame of � without accepted detections contributes $\boldsymbol { a } _ { f } = 0$

$$
A _ { c } ( \tau ) = P \big ( N _ { f } ( \tau ) > 0 \big ) \mathbb { E } \big [ a _ { f } ( \tau ) \mid N _ { f } ( \tau ) > 0 \big ] \ \leq \ P \big ( N _ { f } ( \tau ) > 0 \big ) ,\tag{3}
$$

where $N _ { f } ( \tau )$ is the number of accepted detections in $f _ { c }$ , and $P$ refers to the frames of $c .$ If $P ( N _ { f } ( \tau ) > 0 ) < \alpha$ , the limit is met even if every accepted detection is wrong. Every applicable outcome is therefore reported with the retention, the share of the correct detections of a condition that the threshold accepts, estimated on its test part (Sect. 4.3).

## 3.2 Selection of a Threshold per Condition

The frames of each condition are divided into a selection part, used to select thresholds, and a test part, used to test them (Sect. 4.3). A candidate threshold $\tau _ { j }$ , fixed in advance, passes if the Hoefding–Bentkus (HB) �-value of the selection data of � for the null hypothesis $A _ { c } ( \tau _ { j } ) > \alpha$ is at most �. For � selection frames of $c$ with mean $\hat { A } _ { c } ( \tau _ { j } )$ of $a _ { f } ( \tau _ { j } )$ , this �-value is

$$
p = \operatorname* { m i n } \bigl \{ \exp \bigl ( - n \ : h ( \operatorname* { m i n } ( \hat { A } _ { c } ( \tau _ { j } ) , \alpha ) , \alpha ) \bigr ) , \ e \ : P \bigl ( \mathrm { B i n } ( n , \alpha ) \leq \lceil n \hat { A } _ { c } ( \tau _ { j } ) \rceil \bigr ) \bigr \}
$$

Table 2. Symbols.
<table><tr><td> $c , c ^ { \prime }$ </td><td>condition of selection, condition of use</td><td>T</td><td>uncertainty threshold</td></tr><tr><td> $f , f _ { c }$   $d , y ( d )$ </td><td>frame; car detections of c in f detection; 1 if correct, 0 if wrong</td><td> $\tau _ { j }$   $\tau _ { c }$ </td><td>candidate threshold threshold selected in c</td></tr><tr><td> $M ,$ </td><td>number of members of the</td><td> $a _ { f }$ </td><td>per-frame share of false</td></tr><tr><td> $m ( d )$ </td><td>ensemble; number of members that contribute a box to d</td><td></td><td>acceptances among the accepted detections in  $f _ { c }$ </td></tr><tr><td> $b _ { k } , s _ { k }$ </td><td>box of member k; its confidence</td><td> $A _ { c } , \hat { A } _ { c }$ </td><td>FAP in c; its estimate</td></tr><tr><td> $u _ { \mathrm { s } } , u _ { \mathrm { e } }$ </td><td>spatial and existence indicator</td><td> $A , \pi _ { c }$ </td><td>FAP over all frames with a car detection; share of them in c</td></tr><tr><td> $u ( d )$ </td><td>indicator value of d</td><td> $N _ { f }$ </td><td>number of accepted detections in  $f _ { c }$ </td></tr><tr><td> $\alpha$ </td><td>limit</td><td> $n$ </td><td>number of independent units</td></tr><tr><td> $\delta$ </td><td>error probability</td><td> $L , U$ </td><td>lower and upper confidence bound</td></tr><tr><td></td><td></td><td></td><td>of  $A _ { c ^ { \prime } } ( \tau _ { c } )$ </td></tr><tr><td> $\delta ^ { \prime }$ </td><td>δ for selecting,  $\delta / 2$  for testing a threshold</td><td></td><td></td></tr></table>

with a binomial random variable Bin(�, �), Euler’s number � and $h ( v , w ) = v \ln ( v / w ) +$ $( 1 - v ) \ln ( ( 1 - v ) / ( 1 - w ) )$ , where 0 ln $0 = 0 [ 4 ]$ . This �-value is valid for any distribution of the per-frame shares $a _ { f }$ in [0, 1]. The count � includes frames without accepted detections, and the �-value requires independent and identically distributed units.

The candidates are taken in order from the lowest upwards, starting where $A _ { c }$ is small by Eq. (3). The selected threshold $\tau _ { c }$ is the last candidate that passes before the first candidate that fails. If the lowest candidate fails, no threshold is selected in �. The $\mathrm { F A P } A _ { c }$ need not increase monotonically with �, because a lower threshold can remove a correct detection and keep a wrong one. A candidate above the first failure can therefore pass, but it is not selected. If the selection frames of � are independent draws from the frame distribution of $^ { c , }$ this fixed order bounds by � the probability that any candidate passing before the first failure has an FAP above � [4]. The limit therefore holds for $\tau _ { c }$ with probability at least 1 − �, for the distribution of � only.

If every frame with a car detection belongs to exactly one condition and $\pi _ { c }$ is the share of these frames in condition �, then

$$
A ( \tau ) = \sum _ { c } \pi _ { c } A _ { c } ( \tau ) \leq \operatorname* { m a x } _ { c } A _ { c } ( \tau ) .\tag{4}
$$

A bound on the FAP � over all frames with a car detection therefore refers to the mix � of the recording campaign, which need not equal the mix in operation. Weather and dataset assign each frame to exactly one condition. For them, a threshold that meets � in every condition meets it under every mix, provided that the frames within each condition are distributed in operation as in the recorded data. Distance is assigned per detection, and a frame can hold near and far detections, so this premise fails for distance.

## 3.3 Decision on Applicability in the Condition of Use

A threshold $\tau _ { c }$ is tested on the test part of condition $c ^ { \prime }$ . The upper confidence bound � of $A _ { c ^ { \prime } } ( \tau _ { c } )$ is obtained by inverting the HB $p \cdot$ -value at error probability $\delta / 2 ,$ , and the lower confidence bound � is one minus the same bound computed from the values $1 - a _ { f }$ , so $[ L , U ]$ is an interval at confidence level $1 - \delta .$ . The performance target is applicable in $c ^ { \prime }$ if $U \leq \alpha$ , not applicable if $L > \alpha$ , and not determined if $L \leq \alpha < U$ . If no threshold can be selected in �, the outcome is no threshold selected (Fig. 2c). For a given true FAP, only one of the two erroneous outcomes is possible, so the test gives an erroneous applicable or not applicable outcome with probability at most $\delta / 2$ . This holds if the test units are independent draws from $c ^ { \prime }$ and independent of the selection units. This error probability applies to each outcome on its own, so a safety argument that relies on several outcomes at once has to divide $\delta$ by their number. A not determined outcome makes no claim, but it can hide an exceeded limit.

Used in its own condition of selection $( c ^ { \prime } = c )$ with independent frames, a threshold gives the outcome not applicable only after an erroneous selection (probability at most $\delta )$ or an erroneous interval (at most $\delta / 2 )$ . The probability of this outcome is therefore at most $\delta + \delta / 2$ , and its observed frequency serves as a check of the selection (Sect. 5.1).

## 3.4 Changes in the Detections That Can Lead to an Exceeded Limit

A transferred threshold is the same value on the scale of the indicator in both conditions. The pooled share that it gives in a condition depends on the share of wrong detections among all car detections of the condition and on the distributions of the indicator for correct and for wrong detections. The FAP, a mean over frames, depends in addition on how detections are spread over frames. Four changes in the detections from � to $c ^ { \prime }$ can therefore lead to an exceeded limit.

Weaker separation. The two distributions overlap more in $c ^ { \prime } .$

More wrong detections. A larger share of wrong detections in $c ^ { \prime }$ at unchanged distributions.

Shifted values. The values of both kinds of detection lie at another position on the scale, so the same threshold accepts a diferent share of them.

Diferent spread over frames. The wrong detections are distributed diferently over the frames, for example concentrated in frames with few accepted detections.

AUROC measures separation only. It does not depend on the share of wrong detections [37,38] or on the spread of detections over frames, and it is unafected by any rescaling of the indicator that keeps the order of the values. The ROC curve plots, for every threshold, the share of the wrong detections rejected against the share of the correct detections withheld, and a threshold marks its operating point on this curve. A threshold that marks the same operating point in � and $c ^ { \prime }$ rejects the same shares of the correct and of the wrong detections in both conditions. An exceeded limit then comes from more wrong detections or a diferent spread over frames. Weaker separation and more wrong detections lower the retention that any threshold on this indicator can reach at the limit in $c ^ { \prime } { . }$ , whereas shifted values are compensated by a threshold selected in $c ^ { \prime } .$

## 3.5 Minimum Number of Independent Units to Show That a Limit Is Met

A unit is a frame or a recorded sequence that the test treats as one independent draw. Units without any false acceptance are the data most favourable to a passing candidate and to the outcome applicable. If every per-unit value is 0 or 1 and the true FAP lies just above $\alpha ,$ all of � independent units are free of false acceptances with a probability close to $( 1 - \alpha ) ^ { n }$ . Showing that the limit is met at error probability $\delta ^ { \prime }$ therefore requires at least

$$
n \geq { \frac { \ln \delta ^ { \prime } } { \ln ( 1 - \alpha ) } }\tag{5}
$$

independent units per condition and part, with $\delta ^ { \prime } = \delta$ for selecting and $\delta ^ { \prime } = \delta / 2$ for testing a threshold. This lower bound is the zero-failure bound, about $3 / \alpha$ $\delta ^ { \prime } = 0 . 0 5$ (the rule of three). It holds for every test that is valid for any distribution of the per-unit values in [0, 1] and in which fewer false acceptances never weaken the evidence. The HB test is such a test. The bound is necessary, not suficient, and any unit with a false acceptance raises the number further. It does not apply to the outcome not applicable, which a few units with many false acceptances can show.

## 4 Detectors, Datasets and Conditions of the Study

The test is applied to ensembles of PointPillars networks on KITTI and nuScenes for the three changes of condition of Sect. 1.1.

## 4.1 Ensemble Detectors and Rule for Merging Member Boxes

Each ensemble consists of $M = 4$ PointPillars networks [39], trained from diferent random seeds with the OpenPCDet toolbox [40] (version 0.6.0). On KITTI [1], the members use the three-class configuration of the toolbox and are trained for 10 passes (epochs) over the 3,712 training frames. On nuScenes [2], the members use the classbalanced multi-head configuration [41] and are trained for about 10 of its 20 epochs on the 700 training scenes. Each member keeps the car boxes with a confidence of at least 0.1. The boxes of the first member are taken in descending order of confidence, and each other member contributes its unused car box with the highest IoU with that box, if the IoU is at least 0.5. A detection is formed only if at least three of the four members contribute a box, a rule called the vote. The merged box is the mean of the member boxes, and its confidence is the mean confidence of the contributing members.

The vote is itself a rejection rule, applied before any uncertainty threshold. Table 3 lists the share of all annotated cars, without a distance or visibility filter, that a detection after the vote matches (the recall after the vote). Retention therefore refers to the correct detections after the vote, not to all annotated cars. Because a missing member contributes $s _ { k } = 0$ , a detection with three members has $u _ { \mathrm { e } } \geq 0 . 2 5$ , and a threshold on $u _ { \mathrm { e } }$ below 0.25 accepts only detections found by all four members.

## 4.2 Assignment of Distance, Weather and Dataset Conditions

Far distance and rain are treated as potential TCs because both act on the point cloud: rain afects the returns through droplets in the air and water on the road, and an object at greater distance returns fewer points. This paper does not show that they initiate a hazardous behaviour. Weather and dataset labels are assigned per frame by rules that use no output of the detector.

Every detection is assigned the distance of its box centre from the LiDAR origin in the ground plane, because a wrong detection has no annotated car whose distance could be used. Distance therefore describes where the detector places a box: the far condition is defined by the detector output, not by the distance of an annotated car. The near/far boundary is the median distance of the car detections in the grid part (Sect. 4.3). A frame contributes its near detections to the near condition and its far detections to the far condition, so near and far share frames.

A nuScenes scene [2] is a rain scene if its text description contains the word ‘rain’ and a night scene if it contains the word ‘night’. Night scenes are excluded, so that clear and rain frames do not also difer in lighting. The daylight scenes without rain are called clear scenes. The validation set holds 24 rain scenes in daylight. The label gives no rain intensity and does not separate falling rain from a wet road. Clear scenes can include a road wet from earlier rain, and clear and rain scenes also difer in location and trafic.

KITTI marks image regions that are not annotated (DontCare) and treats vans as a neighbouring class of cars in its benchmark [42]. A detection in a DontCare region or on a van is therefore not necessarily wrong. KITTI results are reported with these 2,454 detections excluded (KITTI rule), and Sect. 5.1 reports the efect of including them.

The change of dataset and detector uses two ensembles, the KITTI detector trained on KITTI and the nuScenes detector trained on nuScenes. In the transfer from nuScenes to KITTI, a threshold selected with the nuScenes detector on nuScenes is used with the KITTI detector on KITTI, and the reverse transfer works in the same way. The test therefore examines the transfer of a threshold value between two separately trained detectors, not for one detector into another environment. Distance is examined on both datasets, so the three changes of condition give the four pairs of conditions of Table 3.

## 4.3 Division of the Frames and Choice of the Independent Unit

Within one dataset, a grid part (10 %) fixes the near/far boundary and the grid of candidate thresholds, the quantiles of the indicator at levels 0.01, 0.02, . . . , 1.00 over the car detections of both conditions. A selection part (45 %) selects the thresholds, and a test part (45 %) tests them (Fig. 3). For the change of dataset and detector, the frames of each dataset are divided into halves, and the candidates 0.01, 0.02, . . . , 1.00 are fixed in advance, so that neither dataset influences them. Every division is carried out with five random seeds, and each resulting assignment of the frames to the parts is called a split. The splits divide the same recorded frames in diferent ways. An outcome on all five splits therefore has the same bound on the error probability as an outcome on one split and shows only that it does not depend on the split.

Frames of one recorded sequence, a nuScenes scene of about 20 s or a KITTI drive, share objects, weather and road. Consecutive annotated nuScenes frames are 0.5 s apart and show the same objects from nearly the same position. ISO/PAS 8800 lists such nearly identical frames, one in the training and the next in the test data, as a lack of independence (Table 11-2) [7]. Frames are therefore dependent within one part and, if frames are divided at random, also between the selection and the test part. The results are reported for two evaluations and checked with a third run, all with the same seeds and limits. In the frame-level evaluation, frames are divided at random into the parts (division by frame, Fig. 3), and each frame is one independent unit. In the sequence-level evaluation, whole recorded sequences are divided into the parts (division by recorded sequence), and each sequence is one independent unit. In the sequence-level evaluation, the HB �-value and the bounds use, in place of $a _ { f }$ , one value per sequence, the mean of $a _ { f }$ over its frames in the condition, and � counts sequences. This is valid if sequences are independent draws, although nuScenes scenes cut from one continuous log share the data-collection vehicle, the day and the weather. Each sequence has equal weight, so this evaluation bounds the mean over sequences of these per-sequence means, which equals $A _ { c }$ if long and short sequences have, on average, the same mean of $a _ { f }$ . The third run divides by recorded sequence but counts frames as units, to separate the efect of the division from that of the unit. The 150 nuScenes validation scenes give 15 grid, 68 selection and $^ { 6 7 }$ test scenes, with 9 to 12 rain scenes in each selection and test part. The 45 KITTI validation drives, none of which contributes training frames, give 4 grid, 20 selection and 21 test drives in split 0.

Table 3. Pairs of conditions. Last column: frames per condition in the selection and test parts of split 0 (one of five random assignments of the frames to the parts), first line for the first condition. Boundaries: split 0.
<table><tr><td rowspan=1 colspan=1>pair ofconditions</td><td rowspan=1 colspan=1>recorded frames and cardetections</td><td rowspan=1 colspan=1>conditions</td><td rowspan=1 colspan=1>frames,selection / test</td></tr><tr><td rowspan=1 colspan=1>KITTIdistance</td><td rowspan=1 colspan=1>3,769 validation frames;27,175 car detections (KITTIrule); 0.858 of 14,385annotated cars matched</td><td rowspan=1 colspan=1>near / far, boundary34.8 m</td><td rowspan=1 colspan=1>1,484 / 1,5001,514 /1,545</td></tr><tr><td rowspan=1 colspan=1>nuScenesdistance</td><td rowspan=1 colspan=1>6,019 validation frames from150 scenes; 40,036 cardetections; 0.398 of 65,465annotated cars matched</td><td rowspan=1 colspan=1>near / far, boundary21.4m</td><td rowspan=1 colspan=1>2,207 / 2,2092,270 / 2,276</td></tr><tr><td rowspan=1 colspan=1>nuScenesweather</td><td rowspan=1 colspan=1>4,449 clear and 968 rainframes, all in daylight; 602night frames excluded</td><td rowspan=1 colspan=1>clear / rain</td><td rowspan=1 colspan=1>1,917 / 1,900392 / 414</td></tr><tr><td rowspan=1 colspan=1>change ofdataset anddetector</td><td rowspan=1 colspan=1>all validation frames ofKITTI and of nuScenes</td><td rowspan=1 colspan=1>KITTI with theKITTI detector(KITTI rule) /nuScenes with thenuScenes detector</td><td rowspan=1 colspan=1>1,844 / 1,8522,849 / 2,850</td></tr></table>

![](images/0bd886ede93c2d45f28d1b271afa424b9c606b624890a7c094dbe3561558a6fa.jpg)  
Fig. 3. Division of the recorded frames into parts within one dataset (left) and for the change of dataset and detector (right). Arrows: a threshold selected in � is tested in $c ^ { \prime }$ (one transfer direction) and in � (check of the selection). Bottom: 36 example frames divided by frame or by sequence.

## 4.4 Limits, Error Probability and Predictions Fixed in Advance

Before any outcome was computed, the error probability $\delta = 0 . 0 5 ,$ , the limits $\alpha = 1 0 ^ { - 2 }$ $1 0 ^ { - 3 }$ and $1 0 ^ { - 4 } \cdot$ , the conditions and the FAP in the per-frame form of Eq. (2) were fixed and recorded with a date in version control. These three limits define the performance targets and are called target limits (the required limits of the RQs). The exploratory limits $\alpha = 0 . 0 5$ and 0.1, marked <sup>†</sup>, were fixed at the same time to show at which limits the test reaches the outcome applicable or not applicable. They define no performance target. The sequence-level evaluation and the third run were added afterwards. Predictions of the outcomes at the target limits were recorded together with the outcomes that would count as their failure. The supplementary material lists them with their outcomes.

## 5 Results of the Applicability Test on KITTI and nuScenes

Sections 5.1 to 5.3 answer the research questions in turn: the outcomes of the test (RQ1), the changes in the detections that accompany a not applicable outcome (RQ2) and the number of independent units needed (RQ3).

## 5.1 Applicability of the Performance Target in the Condition of Use (RQ1)

Figure 4 shows all 240 outcomes of the frame-level evaluation, of which 168 were applicable or not applicable. At the target limit $1 0 ^ { - 2 }$ , the outcome was not applicable on all five splits only after the transfer of the threshold for $u _ { \mathrm { s } }$ from nuScenes to KITTI. This transfer gave the same outcome at both exploratory limits. After two distance transfers (near to far on KITTI for $u _ { \mathrm { s } } ,$ far to near on nuScenes for $u _ { \mathrm { e } } )$ , the outcome at $1 0 ^ { - 2 }$ was not applicable on one and on three of five splits. No weather transfer gave this outcome at $1 0 ^ { \hat { - 2 } }$ . At 0.1<sup>†</sup>, the outcome was not applicable on at least four of five splits for at least one indicator after transfers into far distance and into rain. In the opposite directions, it was applicable on all five splits.

![](images/1a4ff590633a3982e7af87ea145e31108fcdf05fe6daeb6e1adbd6a34cf37768.jpg)  
Fig. 4. Outcome of every transfer in the frame-level evaluation (left) and retention of the transferred thresholds with an applicable outcome at $1 0 ^ { - 2 }$ (right). <sup>†</sup>: exploratory limit, which defines no performance target. $\because :$ below 0.01 %. No threshold was selected at $1 0 ^ { - 3 }$ and $1 0 ^ { - 4 }$ . Frames of one recorded sequence are not independent, so the stated error probability is not guaranteed.

$\mathrm { A t } 1 0 ^ { - 2 }$ , transferred thresholds with an applicable outcome had a retention of at most 29 % on every split (Fig. 4, right). Thresholds selected in the same conditions of use (open markers) reached a median retention of at most 33 %. A retention near zero, as for $u _ { \mathrm { s } }$ from KITTI to nuScenes, means that almost no frame holds an accepted detection, so the limit is met by Eq. (3). Every threshold for $u _ { \mathrm { e } }$ at $1 0 ^ { - 2 }$ lay below 0.25 and accepted only detections found by all four members (Sect. 4.1).

Three checks were made. Including the detections excluded by the KITTI rule changed no outcome that held on all five splits. In the third run, every transfer with the outcome not applicable on at least four of five splits in the frame-level evaluation had it again on at least four of five splits. Used in their own condition (Sect. 3.3), thresholds gave the outcome not applicable in none of the 112 outcomes of the nuScenes distance and weather conditions with a selected threshold, and in 5 of 113 in the third run. Both frequencies lie below the bound $\delta + \delta / 2 = 7 . 5 \%$ . With the division by frame, frames of one sequence lie in both parts, so the first frequency is weak evidence.

In the sequence-level evaluation, only 5 of the 240 outcomes were applicable or not applicable, all on nuScenes and applicable at $0 . 1 ^ { \dagger }$ (Sect. 5.3).

Answer to RQ1. In the sequence-level evaluation, which the dependence between frames requires, the recorded data show neither that a transferred threshold meets a target limit nor that it does not. In the frame-level evaluation, the outcome at the target limit $1 0 ^ { - 2 }$ was not applicable after the transfer of the threshold for $u _ { \mathrm { s } }$ from nuScenes to KITTI and, on some splits, after a change of distance. After transfers into far distance and into rain, the outcome was not applicable at the exploratory limit $0 . 1 ^ { \dagger }$ for at least one indicator on at least four of five splits. A transferred threshold with an applicable outcome at $1 0 ^ { - 2 }$ rejected most correct detections.

Table 4. Changes in the detections from the condition of selection to the condition of use (→) for the transfers into a condition with more wrong detections that gave not applicable outcomes, one indicator each. Shares of wrong detections: all validation frames; other values: test part of split 0, FAP and pooled share at the transferred threshold selected at $0 . 1 ^ { \dagger } . - : \mathrm { n o }$ further change reported.
<table><tr><td rowspan=1 colspan=1>transfer (indicator)</td><td rowspan=1 colspan=1>AUROC</td><td rowspan=1 colspan=1>share ofwrongdetections</td><td rowspan=1 colspan=1>further change</td><td rowspan=1 colspan=1>shown byAUROC</td></tr><tr><td rowspan=1 colspan=1>near → far, KITTI $\left( u _ { \mathrm { s } } \right)$ </td><td rowspan=1 colspan=1>0.915 → 0.845</td><td rowspan=1 colspan=1>0.34 → 0.76</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>weakerseparation</td></tr><tr><td rowspan=1 colspan=1>near → far,nuScenes $\left( u _ { \mathrm { s } } \right)$ </td><td rowspan=1 colspan=1>0.729 → 0.655</td><td rowspan=1 colspan=1>0.19 → 0.51</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>weakerseparation</td></tr><tr><td rowspan=1 colspan=1>clear → rain,nuScenes $\left( u _ { \mathrm { e } } \right)$ </td><td rowspan=1 colspan=1>0.844 → 0.856</td><td rowspan=1 colspan=1>0.33 → 0.42</td><td rowspan=1 colspan=1>different spread overframes: FAP 0.09 → 0.17,pooled share 0.12 → 0.17</td><td rowspan=1 colspan=1>none</td></tr><tr><td rowspan=1 colspan=1>nuScenes → KITTI $\left( u _ { \mathrm { s } } \right)$ </td><td rowspan=1 colspan=1>0.749 → 0.905</td><td rowspan=1 colspan=1>0.35 → 0.55</td><td rowspan=1 colspan=1>shifted values: median usof the correct detections0.23 → 0.12</td><td rowspan=1 colspan=1>none</td></tr></table>

## 5.2 Changes in the Detections That Accompany a Not Applicable Outcome (RQ2)

Except in one transfer, not applicable outcomes occurred only in a condition of use with a larger share of wrong detections than the condition of selection. Table 4 lists, for these transfers, the AUROC and the share of wrong detections in both conditions and the further change in the detections. Figure 5 shows three of these transfers. AUROC is computed so that a higher indicator value marks a wrong detection, and operating points refer to the test part of split 0.

With distance, separation weakened and wrong detections became more frequent (Table 4). Among the detections that the threshold selected at near distance accepts, the share of wrong detections grew with distance also below the near/far boundary (supplementary material), so the not applicable outcomes are not specific to this boundary.

In rain, the estimated separation did not weaken. The AUROC of $u _ { \mathrm { e } }$ did not decrease, and the transferred threshold selected at $0 . 1 ^ { \dagger }$ marked nearly the same operating point on both curves (Fig. 5). The medians of $u _ { \mathrm { e } }$ for correct and for wrong detections moved by at most 0.03, so the values did not shift. The FAP rose more than the pooled share, so the spread of wrong detections over frames added to the efect of more wrong detections. Rain and clear scenes also difer in location and trafic (Sect. 4.2), so these changes accompany the not applicable outcomes and are not shown to cause them.

Between the datasets, $u _ { \mathrm { s } }$ took lower values on KITTI (Table 4), so a threshold selected on nuScenes lay high on the KITTI scale. $\mathrm { { A t 0 . 1 } ^ { \dag } }$ , it accepted about 25 % of the

![](images/da60917267b54aa59dc88878222116e4cef6520165a3d4f8d201db440188f691.jpg)  
Fig. 5. ROC curves (top; legend: AUROC) and estimated FAP against the threshold (bottom) on the test part of split 0, in the condition of selection (solid) and the condition of use (dashed). Squares and vertical lines: transferred threshold selected at $0 . 1 { } ^ { \dagger } ;$ circles: transferred threshold selected at $1 0 ^ { - 2 }$ ; dotted horizontal lines: limits $0 . 1 ^ { \dagger }$ and $1 0 ^ { - 2 } \div$ ; bars: interval $[ L , U ]$

KITTI wrong detections, against $8 \%$ on nuScenes. The outcome was not applicable on KITTI although $u _ { \mathrm { s } }$ separated correct from wrong detections better there. The threshold is transferred as a raw value, since rescaling per dataset would hide this shift.

The not applicable outcomes after the one transfer into a condition with a smaller share of wrong detections (far to near distance on nuScenes for $u _ { \mathrm { e } } ,$ at $1 0 ^ { - 2 }$ on three of five splits) were accompanied by shifted values. The threshold selected at far distance lay higher on the scale than the threshold selected at near distance and therefore accepted more of the wrong detections at near distance.

Answer to $R Q 2 .$ . With distance, weaker separation and more wrong detections accompanied the not applicable outcomes. In rain, where not applicable outcomes occurred only at the exploratory limits, more wrong detections and a diferent spread over frames accompanied them at a nearly unchanged operating point. Between the datasets, shifted values and more wrong detections accompanied them, and shifted values also accompanied the one exception. AUROC shows only weaker separation and therefore showed the not applicable outcomes only for transfers into far distance.

## 5.3 Independent Units Needed to Show That a Limit Is Met (RQ3)

Figure 6 places the test units of each condition on the curve of Eq. (5). In the frame-level evaluation, every condition meets the minimum number at $1 0 ^ { - 2 } . \mathrm { A t } 1 0 ^ { - 3 }$ , selecting a threshold requires 2,995 units without a false acceptance, more than the selection part of any condition holds, so no threshold could be selected at $1 0 ^ { - 3 } \ \mathrm { o r } \ 1 0 ^ { - 4 } . \ \mathrm { A t } \ 1 0 ^ { - 2 }$ , no threshold for $u _ { \mathrm { s } }$ could be selected in rain on any of the five splits, nor in clear weather on three of them, because the selection frames held false acceptances, which raise the number needed above that of Eq. (5).

![](images/9741e79e4828e48d758b1f5301e8e683cb0c6f32d05f35157cf5fb445233abd5.jpg)  
Fig. 6. Lowest limit that � independent units without a false acceptance can show to be met (Eq. (5)), with the test units of each condition in split 0. Marker heights are computed from �, not measured. Whole set: the KITTI frames of the change of dataset and detector.

In the sequence-level evaluation, the 150 nuScenes validation scenes are fewer than the 299 that the selection alone requires at $1 0 ^ { - 2 }$ . A selection part of 20 to 22 KITTI drives is too small for any limit used here, so no threshold was selected on KITTI. On nuScenes, the scenes of the distance conditions and of clear weather reach the minimum number only at $0 . 1 ^ { \dagger }$ , and the rain scenes do not reach it.

Answer to RQ3. Showing that a limit � is met requires at least the number of independent units of Eq. (5) per condition and part, and more if any unit holds a false acceptance. At $1 0 ^ { - 2 }$ , these are 299 units for selecting and 368 for testing a threshold. In the frame-level evaluation, KITTI and nuScenes meet this minimum at $1 0 ^ { - 2 }$ in every condition and at $1 0 ^ { - 3 }$ in none. In the sequence-level evaluation, they can show no target limit to be met, and the rain scenes and the KITTI drives no limit used here.

## 6 Discussion

This section states what the results mean for a SOTIF analysis and where they may not hold.

## 6.1 Consequences for the SOTIF Analysis

Use of the outcomes in a safety argument. An outcome is an analysis result for the detector as a system element, not a classification of scenarios, because tracking, fusion and motion planning lie between the detector and a hazardous behaviour. Table 5 lists what each outcome means for the performance target and which measure follows. In the condition of selection, the LTT guarantee is the evidence, and a not determined outcome there does not withdraw it. The vehicle manufacturer supplies how often each condition occurs in operation and the limit � from the risk allocated to the detector.

Table 5. Meaning of each outcome for the performance target in $c ^ { \prime }$ and the measure that follows.
<table><tr><td rowspan=1 colspan=1>outcome</td><td rowspan=1 colspan=1>meaning for the performancetarget</td><td rowspan=1 colspan=1>measure that follows</td></tr><tr><td rowspan=1 colspan=1>applicable</td><td rowspan=1 colspan=1>evidence, at error probability atmost $\delta / 2$ if the units areindependent, that the target is metin $c ^ { \prime }$ with the transferred threshold</td><td rowspan=1 colspan=1>use the threshold in $\overline { { c ^ { \prime } } }$ if the shareof annotated cars that reach themotion planner meets a secondtarget on missed objects</td></tr><tr><td rowspan=1 colspan=1>not applicable</td><td rowspan=1 colspan=1>evidence, at error probability atmost $\delta / 2$ if the units areindependent, that the target is notmet in $c ^ { \prime }$ with the transferredthreshold</td><td rowspan=1 colspan=1>detector supplier: select a thresholdin $c ^ { \prime }$ with a run-time signal for $c ^ { \prime }$ or use a common threshold forseveral conditions; vehiclemanufacturer: restrict the ODD orthe functionality, or add a measurein tracking, fusion or motionplanning</td></tr><tr><td rowspan=1 colspan=1>not determined</td><td rowspan=1 colspan=1>no statement, because the data ofc&#x27; do not suffice or the FAP liesclose to α</td><td rowspan=1 colspan=1>record further independent units(Eq. (5) gives the minimum) andtreat c&#x27; as not verified</td></tr><tr><td rowspan=1 colspan=1>no thresholdselected</td><td rowspan=1 colspan=1>the lowest candidate thresholdfails in c with the available data</td><td rowspan=1 colspan=1>record further independent units inc or request a reallocation of α atvehicle level</td></tr></table>

Independent unit and amount of recorded data. Most transfer outcomes were applicable or not applicable in the frame-level evaluation, almost none in the sequence-level evaluation (Sect. 5.1). What the data can show therefore depends on the independent unit. It is the recorded sequence unless the independence of frames is shown, because with dependent frames the stated error probability is not guaranteed. Equation (5) gives the minimum number of independent recorded sequences per condition before recording. $\mathrm { A t } 1 0 ^ { - 2 }$ , this minimum exceeds the validation sets of KITTI and nuScenes, and it grows linearly with $1 / \alpha$ and only logarithmically with $1 / \delta ^ { \prime }$ . Recording a rare condition more often than it occurs in operation alters the FAP over all frames (Eq. (4)), but it supplies the units that the test per condition requires. ISO/PAS 8800 notes that such a number of samples is needed to estimate the relative frequency of an error [7].

Threshold per condition or common threshold. A threshold per condition requires the condition to be known at run time. The box gives the distance [33], whereas weather requires a rain signal, whose errors have to be included in the allocation of �. A common threshold for several conditions, which has to pass the selection in each of them at error probability � divided by their number, needs no run-time signal. For near and far distance, it always gave an applicable or not determined outcome, with a lower retention at near distance than the threshold selected there (supplementary material).

A second performance target on missed objects. The performance target limits only the wrong detections that are accepted. A missed object can be a more hazardous error than an accepted wrong detection, so a lower threshold exchanges one error for another. A safety argument that uses the threshold therefore needs a second target on missed objects, a lower limit on the share of annotated cars that reach the motion planner. This target can be derived from the same acceptance criterion and tested under the same conditions with the same interval, with Eq. (5) applied to its own limit.

## 6.2 Threats to Validity

Construct validity. A wrong detection is defined by the annotations and the overlap criterion, so boxes on annotated cars that miss the overlap criterion and boxes on unannotated objects count as wrong. On KITTI, of the wrong detections that the threshold for $u _ { \mathrm { s } }$ selected at near distance in Fig. 1a (limit 0.1<sup>†</sup>, split 0) accepts in all validation frames, 1,536 lie inside the camera image, overlap no annotation of another class and overlap no annotated car at an IoU above 0.1. The share of unannotated cars among them was not measured. The near/far boundary is a median of the detector output, so it difers between detectors and splits.

Internal validity. Correctness depends on the matching criterion. This paper uses an IoU of 0.5 for both datasets, whereas the KITTI benchmark uses 0.7 for cars and nuScenes matches by the distance between box centres. Matching by this distance would count some far-distance boxes that miss the overlap criterion as correct (not measured). Every detection is anchored on a box of the first member, so a car that this member misses forms no detection even if the other three members detect it. The heading of a merged box is the arithmetic mean of the member headings. This mean can misplace the box when the member headings lie on both sides of $\pm 1 8 0 ^ { \circ }$ or difer by about 180<sup>◦</sup>, so that the box counts as wrong. The KITTI rule reproduces only part of the ignore logic of the KITTI benchmark, and nuScenes has no equivalent annotation.

Statistical validity. Because frames ofone recorded sequence are not independent, the frame-level evaluation overstates the evidence. The sequence-level evaluation discards the variation within sequences, and the HB bound does not use the variance of sequence means. The number of efectively independent units is at most the number of frames and can lie below the number of sequences, because nuScenes scenes of one log are not independent either. The AUROC values are point estimates on split 0 without an interval, and in rain they rest on frames of 24 recorded scenes. If frames were independent, the expected number of erroneous applicable or not applicable outcomes among the 240 transfer outcomes of Fig. 4 would be at most 6 (240 · �/2). Near and far share most KITTI test frames, so the test data of the two conditions are not independent.

External validity. One detector architecture, one way of estimating uncertainty (a deep ensemble) and one object class were examined. After the vote, the detections match most annotated cars on KITTI but fewer than half on nuScenes (Table 3), on which the members were trained for about half of the configured epochs. The measured FAPs and retentions are therefore specific to these detectors. Evidence for a product requires the test on data from its ODD, which KITTI and nuScenes do not represent.

## 7 Conclusion and Future Work

A detector that meets a performance target by rejecting uncertain detections carries a statistical guarantee only for data like those on which its threshold was selected. A SOTIF analysis, however, has to know whether the target is also met under potential triggering conditions. This paper therefore treats the transfer of a threshold to another condition as a question to be tested. The applicability test answers it at a stated error probability as applicable, not applicable or not determined, and the zero-failure bound gives the minimum number of independent units needed to show that the target is met.

Applied to ensembles of PointPillars networks on KITTI and nuScenes, the test leads to three findings. First, a threshold that keeps the false acceptance proportion within a limit under the condition for which it was selected can exceed this limit under another condition: at the target limit after changes of distance and of dataset, and after a change of weather only at limits above the target limits. AUROC showed this only for transfers into far distance. Second, where a transferred threshold still met the target limit, it did so by rejecting most correct detections, so an applicable outcome is informative only together with its retention, the share of correct detections that the threshold accepts. Third, these findings rest on frames as independent units. Counted in recorded sequences, as the dependence between frames requires, neither validation set is large enough to show a target limit to be met, and no transfer gave a not applicable outcome.

For a SOTIF analysis, a performance target met with an uncertainty threshold therefore counts as evidence under a potential triggering condition only after a test under that condition, reported with the retention and the number of independent units. A not applicable outcome identifies, at the stated error probability and with independent units, a condition in which the detector with this threshold does not meet the target, and Table 5 lists the measures that follow. Because the minimum number of independent units can be computed before recording, it can enter the recording plan of each condition.

Future work will extend the test to further potential triggering conditions, among them occlusion and structures that resemble cars, and to further detector architectures and object classes. It will estimate the dependence between frames of one recorded sequence to state the number of efectively independent units, and it will examine thresholds that adapt to the condition at run time. A further direction is to examine whether a monitor of the indicator values in operation, which needs no annotations, detects changes in the detections of the kind found between the datasets and in rain.

## Statements and Declarations

Funding and competing interests. This research was funded by the Institute for Driver Assistance and Connected Mobility (IFM), Kempten University of Applied Sciences. The authors have no competing interests relevant to the content of this article.

Data and code availability. Analysis scripts, configurations, per-run results and the predictions fixed in advance are available at https://github.com/milinpatel07/ lidar-performance-target-applicability. The KITTI and nuScenes datasets are available from their providers under their own licences.

Use ofgenerative AI. Generative AI tools were used to assist with the literature screening, the analysis and figure scripts, and the writing of the text. The authors reviewed and verified all content and take full responsibility for this article.

## References

1. Geiger, A., Lenz, P., Urtasun, R.: Are we ready for autonomous driving? The KITTI vision benchmark suite. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3354–3361 (2012). https://doi.org/10.1109/CVPR.2012.6248074

2. Caesar, H., Bankiti, V., Lang, A.H., Vora, S., Liong, V.E., Xu, Q., Krishnan, A., Pan, Y., Baldan, G., Beijbom, O.: nuScenes: a multimodal dataset for autonomous driving. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11618–11628 (2020). https://doi.org/10.1109/CVPR42600.2020.01164

3. International Organization for Standardization: ISO 21448:2022, Road vehicles – Safety of the intended functionality. ISO, Geneva (2022)

4. Angelopoulos, A.N., Bates, S., Candès, E.J., Jordan, M.I., Lei, L.: Learn then test: calibrating predictive algorithms to achieve risk control. Ann. Appl. Stat. 19(2), 1641–1662 (2025). https://doi.org/10.1214/24-AOAS1998

5. Tibshirani, R.J., Foygel Barber, R., Candès, E.J., Ramdas, A.: Conformal prediction under covariate shift. In: Advances in Neural Information Processing Systems 32 (NeurIPS) (2019)

6. Foygel Barber, R., Candès, E.J., Ramdas, A., Tibshirani, R.J.: Conformal prediction beyond exchangeability. Ann. Stat. 51(2), 816–845 (2023). https://doi.org/10.1214/ 23-AOS2276

7. International Organization for Standardization: ISO/PAS 8800:2024, Road vehicles – Safety and artificial intelligence. ISO, Geneva (2024)

8. Patel, M., Jung, R.: Uncertainty evaluation to support safety of the intended functionality analysis for identifying performance insuficiencies in ML-based LiDAR object detection. In: 12th International Conference on Vehicle Technology and Intelligent Transport Systems (VEHITS), pp. 154–167. SciTePress (2026). https://doi.org/10.5220/ 0014967200004030

9. Hofmeister, J., Kinalzyk, D.: Acceptance criteria, validation targets and performance targets in an ISO 21448 conform development process. In: Systems, Software and Services Process Improvement (EuroSPI 2023). CCIS, vol. 1891, pp. 139–148. Springer, Cham (2023). https: //doi.org/10.1007/978-3-031-42310-9\_10

10. Madala, K., Krishnamoorthy, J., Avalos Gonzalez, C., Shivkumar, A., Solmaz, M.: Contributing factors to consider while defining acceptance criteria and validation targets for assuring SOTIF in autonomous vehicles. SAE Technical Paper 2022-01-0065 (2022). https://doi.org/10.4271/2022-01-0065

11. Reisgys, A., Joos, S., Fricke, S.: SOTIF-compliant decomposition of quantitative risk acceptance criteria for derivation of V&V targets for automated driving systems. In: IEEE International Automated Vehicle Validation Conference (IAVVC). IEEE (2025). https: //doi.org/10.1109/IAVVC61942.2025.11219564

12. Kramer, B., Neurohr, C., Büker, M., Böde, E., Fränzle, M., Damm, W.: Identification and quantification of hazardous scenarios for automated driving. In: Model-Based Safety and Assessment (IMBSA), pp. 163–178. Springer, Cham (2020). https://doi.org/10.1007/ 978-3-030-58920-2\_11

13. Hoss, M., Scholtes, M., Eckstein, L.: A review of testing object-based environment perception for safe automated driving. Automot. Innov. 5(3), 223–250 (2022). https://doi.org/10. 1007/s42154-021-00172-y

14. Birch, J., Blackburn, D., Botham, J., Habli, I., Higham, D., Monkhouse, H., Price, G., Ratiu, N., Rivett, R.: A structured argument for assuring safety ofthe intended functionality (SOTIF). In: Computer Safety, Reliability, and Security (SAFECOMP 2020 Workshops), pp. 408–414. Springer, Cham (2020). https://doi.org/10.1007/978-3-030-55583-2\_31

15. Hawkins, R., Paterson, C., Picardi, C., Jia, Y., Calinescu, R., Habli, I.: Guidance on the assurance ofmachine learning in autonomous systems (AMLAS). Technical report, University of York, arXiv:2102.01564 (2021)

16. Burton, S., Herd, B.: Addressing uncertainty in the safety assurance of machine-learning. Front. Comput. Sci. 5, 1132580 (2023). https://doi.org/10.3389/fcomp.2023. 1132580

17. Geifman, Y., El-Yaniv, R.: Selective classification for deep neural networks. In: Advances in Neural Information Processing Systems 30 (NeurIPS), pp. 4878–4887 (2017)

18. Angelopoulos, A.N., Bates, S., Fisch, A., Lei, L., Schuster, T.: Conformal risk control. In: International Conference on Learning Representations (ICLR) (2024)

19. Bates, S., Angelopoulos, A.N., Lei, L., Malik, J., Jordan, M.I.: Distribution-free, riskcontrolling prediction sets. J. ACM 68(6), 43:1–43:34 (2021). https://doi.org/10.1145/ 3478535

20. Jin, Y., Candès, E.J.: Selection by prediction with conformal p-values. J. Mach. Learn. Res. 24(244), 1–41 (2023)

21. Vovk, V., Gammerman, A., Shafer, G.: Algorithmic Learning in a Random World. Springer, New York (2005)

22. Vovk, V.: Conditional validity of inductive conformal predictors. In: Asian Conference on Machine Learning (ACML), PMLR, vol. 25, pp. 475–490 (2012)

23. Ding, T., Angelopoulos, A.N., Bates, S., Jordan, M.I., Tibshirani, R.J.: Class-conditional conformal prediction with many classes. In: Advances in Neural Information Processing Systems 36 (NeurIPS) (2023)

24. Andéol, L., Fel, T., de Grancey, F., Mossina, L.: Confident object detection via conformal prediction and conformal risk control: an application to railway signaling. In: 12th Symposium on Conformal and Probabilistic Prediction with Applications (COPA), PMLR, vol. 204, pp. 36–55 (2023)

25. Timans, A., Straehle, C.-N., Sakmann, K., Nalisnick, E.: Adaptive bounding box uncertainties via two-step conformal prediction. In: Computer Vision – ECCV 2024. LNCS, vol. 15146, pp. 363–398. Springer, Cham (2025). https://doi.org/10.1007/978-3-031-73223-2\_21

26. Lakshminarayanan, B., Pritzel, A., Blundell, C.: Simple and scalable predictive uncertainty estimation using deep ensembles. In: Advances in Neural Information Processing Systems 30 (NeurIPS) (2017)

27. Harakeh, A., Waslander, S.L.: Estimating and evaluating regression predictive uncertainty in deep object detectors. In: International Conference on Learning Representations (ICLR) (2021)

28. Ovadia, Y., Fertig, E., Ren, J.J., Nado, Z., Sculley, D., Nowozin, S., Dillon, J.V., Lakshminarayanan, B., Snoek, J.: Can you trust your model’s uncertainty? Evaluating predictive uncertainty under dataset shift. In: Advances in Neural Information Processing Systems 32 (NeurIPS) (2019)

29. Wang, Y., Chen, X., You, Y., Li, L.E., Hariharan, B., Campbell, M., Weinberger, K.Q., Chao, W.-L.: Train in Germany, test in the USA: making 3D object detectors generalize. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11710– 11720 (2020). https://doi.org/10.1109/CVPR42600.2020.01173

30. Salay, R., Czarnecki, K., Kuwajima, H., Yasuoka, H., Abdelzad, V., Huang, C., Kahn, M., Nguyen, V.D., Nakae, T.: The missing link: developing a safety case for perception components in automated driving. SAE Int. J. Adv. Curr. Pract. Mobil. 5(2), 567–579 (2022). https://doi.org/10.4271/2022-01-0818

31. Herd, B., Kelly, J., Trapp, M.: CUBICS: situation-aware performance estimation for safetyrelevant ML components. Preprint, arXiv:2608.16564 (2026)

32. Kumar, D., Tayebati, S., Darabi, N., Hu, V.P.-H., Trivedi, A.R.: Uncertainty-aware LiDAR camera autonomy via conformal prediction and principled abstention. In: IEEE International Conference on Omni-layer Intelligent Systems (COINS). IEEE (2025). https://doi.org/ 10.1109/COINS65080.2025.11125785

33. Lee, E., Jung, M., Kim, A.: Toward robust LiDAR based 3D object detection via density-aware adaptive thresholding. Preprint, arXiv:2404.13852 (2024)

34. Li, X., Sun, X., Li, J., Ji, Y., Fu, W.: Bounded graph conditioning for LiDAR 3D object detection under sensor degradation. Sensors 26(9), 2667 (2026). https://doi.org/10. 3390/s26092667

35. Adee, A., Gansch, R., Liggesmeyer, P.: Systematic modeling approach for environmental perception limitations in automated driving. In: 17th European Dependable Computing Conference (EDCC), pp. 103–110. IEEE (2021). https://doi.org/10.1109/EDCC53658. 2021.00022

36. Adee, A., Gansch, R., Liggesmeyer, P., Glaeser, C., Drews, F.: Discovery of perception performance limiting triggering conditions in automated driving. In: 5th International Conference on System Reliability and Safety (ICSRS), pp. 248–257. IEEE (2021). https://doi.org/10.1109/ICSRS53853.2021.9660641

37. Davis, J., Goadrich, M.: The relationship between precision-recall and ROC curves. In: 23rd International Conference on Machine Learning (ICML), pp. 233–240. ACM (2006). https://doi.org/10.1145/1143844.1143874

38. Saito, T., Rehmsmeier, M.: The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLOS ONE 10(3), e0118432 (2015). https://doi.org/10.1371/journal.pone.0118432

39. Lang, A.H., Vora, S., Caesar, H., Zhou, L., Yang, J., Beijbom, O.: PointPillars: fast encoders for object detection from point clouds. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 12689–12697 (2019). https://doi.org/10.1109/ CVPR.2019.01298

40. OpenPCDet Development Team: OpenPCDet: an open-source toolbox for 3D object detection from point clouds. https://github.com/open-mmlab/OpenPCDet (2020)

41. Zhu, B., Jiang, Z., Zhou, X., Li, Z., Yu, G.: Class-balanced grouping and sampling for point cloud 3D object detection. Preprint, arXiv:1908.09492 (2019)

42. Geiger, A., Lenz, P., Urtasun, R.: KITTI 3D object detection benchmark and object development kit. https://www.cvlibs.net/datasets/kitti/eval\_object.php (2012)