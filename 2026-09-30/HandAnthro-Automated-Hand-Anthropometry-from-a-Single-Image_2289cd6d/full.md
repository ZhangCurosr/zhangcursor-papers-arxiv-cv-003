# HandAnthro: Automated Hand Anthropometry from a Single Image

Fan Zhou<sup>1</sup>, Shuairan Chen<sup>1</sup>, Mengying Zhang<sup>1</sup>, Yulin Wu<sup>1</sup>, Sadegh Jafari<sup>1</sup>, Sixing Yu<sup>2</sup>, Rui Li<sup>1</sup>, Ali Jannesari<sup>1</sup>, Guowen Song<sup>1∗</sup>

<sup>1</sup>Iowa State University

<sup>2</sup>Microsoft

{fanzhou,shuairan,mezhang,yulinw,sadegh,ruili,jannesar,gwsong}@iastate.edu sixingyu@microsoft.com

## Abstract

Hand anthropometry supports protective-glove design, but existing measurement methods often require trained operators, specialized hardware, or manual landmarking. We present HandAnthro, which estimates 44 projected hand dimensions from a smartphone photograph of a palm-up hand on US letter-size paper. The pipeline reconstructs wrist-occluded paper boundaries for rectification, whitens non-hand pixels, and refines 41 anthropometry-specific landmarks from a fine-tuned You Only Look Once (YOLO) pose model using image-specific geometry and contours. Controlled evaluation comprised 720 captures from 45 held-out participants, each contributing 16 images across two smartphones, two backgrounds, two angles, and two nominal illumination settings. HandAnthro produced complete outputs for 704 captures (97.8%); among these, mean absolute error (MAE) was 3.80 mm per dimension against two trained operators’ caliper measurements. Regional MAEs were 2.48 mm for non-thumb fingers, 6.04 mm for thumbs, and 6.17 mm for palm and wrist. In a researcher-assisted mobile-app pilot, automated batch processing returned all 44 dimensions for 260 of 268 retained, researcher-screened firefighter images (97.0%). A descriptive, unpaired comparison with an independent national firefighter reference yielded a mean absolute diference of 2.40 mm across 28 sex-by-dimension group-mean contrasts. These results characterize controlled measurement performance and researcher-assisted field feasibility for future distributed handanthropometry studies.

## 1 Introduction

Acute occupational hand injuries account for more than one million US emergency-department visits annually, and protective gloves reduce their relative risk by about 60% (Sorock et al. 2004a,b). This protection depends on fit: poor fit compounds losses in dexterity, tactile sensitivity, and grip strength and reduces user acceptance (Dianat, Haslegrave, and Stedmon 2012; Grifin et al. 2018b). Yet glove-sizing systems often rely on outdated or unrepresentative hand data (Hsiao et al. 2015, 2026). Large, current anthropometric samples are therefore valuable for glove and hand-tool design, but direct caliper measurement requires trained operators, manual photogrammetry requires work on each image, and three-dimensional (3D) approaches require scanners or

calibrated multi-view hardware (Grifin et al. 2018a; Yu et al.   
2013; Yang et al. 2021).

Prior automated two-dimensional (2D) handmeasurement systems have typically relied on tightly controlled acquisition setups, such as flatbed scanners or calibrated industrial-camera rigs (Han and Park 2016; Nguyen, Le, and La 2025). HandAnthro instead uses a simple smartphone protocol in which a palm-up, fingerspread hand is photographed on a letter-size paper sheet. The workflow uses the sheet geometry to rectify the image and recover metric scale, then localizes the anatomical endpoints needed to compute 44 projected dimensions. By reducing on-site equipment requirements and enabling an automated image-to-measurement workflow without manual involvement, this design targets two major operational barriers to larger, geographically distributed occupational hand-anthropometry studies. First, the system must recover a reliable paper reference despite wrist occlusion and camera viewpoint variation. Second, it must accurately localize measurement endpoints on the true hand contour despite unpredictable shadow patterns around the hand boundary caused by variations in real-world illumination and capture conditions.

HandAnthro addresses the first problem by reconstructing the occluded paper mask before quadrilateral detection. It addresses the second by whitening the non-hand pixels, fine-tuning a You Only Look Once (YOLO) pose architecture (Jocher, Chaurasia, and Qiu 2023; Jocher and Qiu 2024) to predict anthropometry-specific landmarks, and refining those predictions with image-specific finger geometry and contour constraints. These stages automatically produce the protocol-defined projected measurements without participant-specific calibration.

We evaluate HandAnthro in two complementary studies. A controlled repeated-capture study quantifies perspectiverectification and end-to-end completion, caliper-referenced measurement accuracy, and sensitivity across the tested capture conditions. A separate researcher-assisted mobileapp field pilot evaluates completion outside the capture rig and compares sex-specific group means with corresponding hand-anthropometry summaries from an independent national firefighter cohort. Because the cohorts are independent, this comparison assesses population-level plausibility rather than individual-level measurement accuracy.

## 2 Related Work

Classical hand anthropometry measures standardized dimensions directly with contact instruments such as calipers or rulers between predefined anatomical landmarks (Gordon et al. 1989; Yu et al. 2013). The resulting stafing and qualitycontrol burden makes distributed collection operationally demanding, although it does not preclude large centrally organized surveys.

Digital methods measure dimensions from 2D images or 3D surfaces. Controlled 2D studies compared softwarederived hand dimensions with direct measurements, but used fixed camera geometry and interactive placement of calibration or measurement lines (Habibi, Soury, and Zadeh 2013; Patel et al. 2018). 2D and 3D scanning methods have likewise been compared with direct measurements for selected dimensions (Li et al. 2008; Yu et al. 2013). 3D acquisition commonly requires a scanner or calibrated multi-view system and additional landmarking (Grifin et al. 2018b; Yang et al. 2021). Deep learning has also automated measurement extraction from a single 3D hand scan, although the input still depends on 3D sensing hardware (Kaashki et al. 2022).

Prior automated 2D methods vary in acquisition requirements and measurement outputs. A freehand smartphone prototype estimated the length ratio between two fingers without scale (Sandnes 2014). A reference-square workflow automatically estimated finger lengths from four digitalcamera images per participant (Magno and Pabico 2013). Systems reporting more dimensions have used more controlled acquisition: an enclosed flatbed-scanner system automatically derived 18 dimensions from one palmar scan (Han and Park 2016), whereas a calibrated industrial-camera system derived 24 parameters from 12 images covering six postures of each hand in a light-controlled box (Nguyen, Le, and La 2025). These systems demonstrate 2D automation but difer in output breadth, capture count, portability, environmental control, and use of dedicated rigs.

Table 1 summarizes automated dimension extraction, acquisition equipment, study samples, and source-reported mean absolute error (MAE) for hand dimensions. It includes methods that automate anatomical dimension extraction or report hand-dimension MAE against physical reference measurements.

## 3 System Design and Methodology

Capture-to-Measurement Workflow. Figure 1 presents the capture-to-measurement workflow. Panel (a) shows the aisafehand mobile app used to acquire the required hand-onpaper image. Perspective rectification (PR) uses the recovered paper quadrilateral to produce a fronto-parallel view, correcting projective distortion and providing the geometric reference for metric scaling (b). Background whitening (BW) sets non-hand pixels to white, reducing interference from variable shadows around the hand and between the fingers (c). A YOLO landmark predictor produces 41 initial anthropometry-specific landmarks (d); geometryconstrained post-processing (PP) refines them using finger geometry and contour evidence (e); and the app visualizes the resulting 44 predicted dimensions (f).

![](images/e76bde0c727941d754ec4b2f74b498305fa506a0a7f7ebcf0ac1cb114eebb31e.jpg)  
Figure 1: HandAnthro capture-to-measurement workflow.

Occlusion-Aware Perspective Rectification. The paper reference defines the target plane for correcting projective distortion and supplies the known geometry used for metric scaling. Because the wrist interrupts a paper boundary, HandAnthro uses the reconstruct-then-detect sequence illustrated in Fig. 2. Given an input image (Fig. 2 (a)), Segment Anything in High Quality (SAM-HQ) (Ke et al. 2023) segments an incomplete paper mask (Fig. 2 (b)) and a raw hand mask (Fig. 2 (c)).

Fixed prompt coordinates for mask segmentation are unsuitable because the location and scale of the hand within the image, as well as its placement on the paper, vary across captures. In the PR stage, HandAnthro derives image-adaptive SAM-HQ prompts from the 21 hand-pose landmarks returned by MediaPipe Hands (Zhang et al. 2020). These auxiliary hand-pose landmarks provide coarse hand-location and scale cues for prompt placement. Appendix D documents the evaluated adaptive prompt construction.

The raw hand mask is dilated once with a 31 × 31 square kernel so that the inpainting region fully covers the portion of the paper occluded by the hand and wrist (Fig. 2 (d)). The dilated mask is then polarity-converted so that its white foreground identifies the region supplied to LaMa (Suvorov et al. 2022) for inpainting (Fig. 2 (e)). LaMa completes the hand-shaped interruption in the paper-mask representation, producing a recovered paper mask (Fig. 2 (f)).

The completed paper mask (Fig. 2 (f)) is used to detect the paper quadrilateral and its four corners. The detected quadrilateral is overlaid on the original red-green-blue (RGB) image for visualization (Fig. 2 (g)). Matching the four detected source corners to the corresponding corners of an ideal rectangle with the known paper aspect ratio determines the homography H. Applying H to the original RGB image produces the perspective-rectified image (Fig. 2 (h)).

Background Whitening and Anthropometry-Specific Landmarks. HandAnthro reuses the raw, undilated hand mask produced during PR (Fig. 2 (c)). The same homography H warps the original RGB image and the raw hand mask into the rectified frame, preserving their pixel correspondence. In the rectified image (Fig. 2 (h)), BW retains RGB values inside the warped hand mask and sets all pixels outside it to white.

<table><tr><td>Method</td><td>Automation and equipment requirements</td><td>Study sample; dimensions</td><td>Reference</td><td>Hand-dimension MAE (mm)</td></tr><tr><td>Han and Park (2016)</td><td>Automated; required equipment: flatbed scanner and enclosure</td><td>11 people; 17 tabulated dimensions</td><td>Live-hand calipers</td><td></td></tr><tr><td>Kaashki et al. (2022)</td><td>Automated; required equipment: depth sensor and tablet</td><td>20 people; 11 real-hand dimensions</td><td>Anthropometrist; instrument unspecified</td><td>4.5</td></tr><tr><td>Nguyen, Le, and La (2025)</td><td>Automated; required equipment: industrial camera, light box, ring light, and calibration target</td><td>539 people; 24 dimensions</td><td>Gauge-block calibration</td><td></td></tr><tr><td>HandAnthro</td><td>Automated; required equipment: smartphone and known-size paper</td><td>45 people; 44 dimensions</td><td>Mean of two operators&#x27; caliper readings</td><td>3.80 overall; 2.48 non-thumb fingers; 6.04 thumb; 6.17 palm/wrist</td></tr></table>

Table 1: Selected hand-anthropometry methods. MAEs are not directly comparable across studies; see Appendix U for definitions and reporting notes.

![](images/9df56efebdcbe34d41cba95fa29298a6a5c5c1682e9df91a85f6d3ce96453948.jpg)  
Figure 2: Occlusion-aware PR workflow: (a) input image; (b,c) initial paper and hand masks; (d,e) construction of the inpainting region; (f) completed paper mask; (g) detected paper quadrilateral (green frame); and (h) perspective-rectified RGB image.

This removes non-hand pixels, including shadows, strengthening the contour evidence without a second segmentation inference (Appendix F). The background-whitened image is resized to a canonical 720 × 932-pixel frame and passed to the YOLO predictor.

We define and annotate 41 anthropometry-specific landmarks spanning the thumb, four fingers, inter-finger roots, palm, and wrist (Appendix C). The final predictor is a YOLOv11x-pose model (Jocher and Qiu 2024) fine-tuned with augmentation on background-whitened images using this landmark schema; Section 4 describes the evaluated model configurations.

Geometry-Constrained Refinement. PP refines the anatomical placement of measurement endpoints. Initial YOLO finger landmarks can misalign with the finger’s orientation and length. PP uses MediaPipe hand-pose landmarks and the hand contour from the background-whitened image to estimate each finger’s axis. A forward search from the finger centerline along this axis locates a corrected fingertip on the contour.

![](images/bbc3d2f2b45bddc3521fdc541ca9393dd293c3727938b17573c2968a24cfc9a0.jpg)  
Figure 3: Geometry-constrained correction of groupwise rotation-and-scale misalignment.

For each finger, PP holds the YOLO-predicted root fixed and compares the predicted and corrected root-to-tip vectors (Fig. 3). Their angular diference defines the rotation θ, and their length ratio $L _ { c } / L _ { \mathrm { Y O L O } }$ defines the scale factor. PP applies this rotation and scaling to the finger’s configured landmark group, then aligns eligible lateral endpoints with the hand boundary along their measurement directions (Appendix H).

Metric Conversion. After PP, 44 fixed endpoint pairs define the reported dimensions. During PR, the four detected paper corners are mapped to a fronto-parallel rectangle with the US Letter width-to-height ratio of 8.5 : 11. The rectified image is resampled to the canonical 720 × 932-pixel frame, so the sheet’s 11-inch height corresponds to 932 pixels and defines a scale of 932/11 pixels per inch. For dimension $d ,$ $\begin{array} { r } { \widehat { y } _ { d } = \frac { 2 5 . 4 } { 9 3 2 / 1 1 } \| u _ { d } - v _ { d } \| _ { 2 } } \end{array}$ mm, where $u _ { d }$ and $v _ { d }$ are its final endpoints.

## 4 Experiments and Evaluation

YOLO Fine-Tuning and Ablation Design. The modeldevelopment dataset comprised one annotated palmar image from each of 42 independent participants. The participants were randomly assigned to mutually exclusive training and validation sets comprising 33 and 9 images, respectively. The 45 participants in the controlled test set were held out from both sets, with no participant overlap. To isolate the incremental contributions of PR, BW, and PP, we evaluated four progressively constructed pipeline configurations. In ablation case A0, YOLO was fine-tuned on the original captured images. In A1, YOLO was fine-tuned on perspective-rectified images. In A2, YOLO was fine-tuned on images processed by both PR and BW. A3 then applied PP to the corresponding A2 predictions. Each configuration was evaluated using two pose backbones (YOLOv8x-pose (Jocher, Chaurasia, and Qiu 2023) and YOLOv11x-pose) and two training data augmentation settings (with and without augmentation), yielding $4 \times 2 \times 2 = 1 6$ experiments (see Appendices M and G for details).

Controlled Repeated-Capture Evaluation. The controlled test set comprised 45 held-out participants, each contributing 16 palmar captures spanning all combinations of two smartphones, two backgrounds, two capture angles, and two nominal illumination settings (720 images). We evaluated PR and end-to-end completion over all 720 inputs. The two nominal illumination settings produced little visible diference in brightness in the captured images, as the smartphones’ builtin automatic exposure and image processing adjusted image brightness across conditions. We therefore pooled the two settings and did not assess illumination efects separately. Capture examples and rig geometry are shown in Appendices A and B.

Two trained operators independently measured all 44 dimensions of each participant’s hand using calipers; their mean served as the reference for each dimension (Appendix R). Automated measurement accuracy was evaluated on the 704 complete outputs. We first calculated each dimension’s MAE across these images, then averaged the 44 dimension-specific MAEs to obtain overall MAE. Regional MAEs used the same aggregation within the thumb (D1– D5), four non-thumb fingers (D6–D33), and palm and wrist (D34–D44).

For capture-condition accuracy, we calculated each dimension’s MAE separately for each smartphone, background, and angle setting, then averaged across the 44 dimensions (Appendix N). For stability, we calculated the standard deviation (SD) of each participant’s predictions for each dimension across completed captures. For each participant and dimension, we also calculated the absolute diference between the mean predictions at the two levels of each factor, then averaged these diferences across participants and dimensions (Appendix N.1).

For landmark-based evaluation, we selected one image per participant from the 704 complete captures, aiming for balanced representation across the eight smartphone– background–angle combinations. The 41 landmarks were manually annotated on each selected image after PR and BW. We report the mean Euclidean pixel error across the 45 × 41 predicted–reference landmark pairs. From these same annotations, we derived all 44 dimensions, then compared them with the same caliper reference. This annotation-to-caliper diagnostic comprised 1,980 image–dimension comparisons and was reported separately from the 704-image automated evaluation (Appendix Q).

Stage-Level Baseline Comparisons. We restricted stagelevel baselines to alternatives that produced directly comparable outputs at the evaluated pipeline stage. Canny– Hough perspective rectification (CH-PR), our baseline using Canny edge detection and probabilistic Hough line detection (Canny 1986; Matas, Galambos, and Kittler 2000), replaced HandAnthro’s complete PR module and was evaluated for completion over all 720 controlled captures; its measurement-error comparison used the 406 captures completed by both pipelines (Appendix I.1). For the BW baselines, we separately substituted REMBG (Gatis 2025), BackgroundRemover (Nader 2025), or CarveKit (Selin 2024) to process the 704 captures for which HandAnthro completed PR, while holding PR, landmark prediction, and PP fixed. End-to-end measurement error for each BW variant was calculated on that variant’s complete outputs; BW-stage runtime was also measured. Section 5.1 reports these comparisons. Researcher-Assisted Field Pilot. At multiple professional events for firefighters, staf assisted with app operation, hand placement, and photography for the field pilot. The cohort comprised 268 firefighters (204 male, 64 female), each with one retained dominant-hand palmar image. Unsuitable photographs were discarded and retaken. Retained images underwent automated batch processing on a laboratory desktop without human intervention. All completed outputs were included in the analysis without human correction (Appendix S).

Processing completion was defined as returning all 44 dimensions and was evaluated over the 268 retained images. Sex-specific means for 14 dimensions were compared with published summaries from an independent national firefighter cohort (Hsiao et al. 2015), yielding 28 contrasts. This comparison assessed population-level plausibility.

## 5 Results

## 5.1 Component Selection, Ablations, and Stage-Level Baselines

Figure S10 in Appendix K shows representative outputs for ablation configurations A0–A3.

Perspective rectification. Across four matched backbone– augmentation configurations, A1 reduced YOLO validation pose loss (Jocher, Chaurasia, and Qiu 2023; Jocher and Qiu 2024) by 77–94% relative to A0. HandAnthro also completed PR on 704/720 captures, compared with 412/720 for CH-PR. On the 406 captures completed by both pipelines, MAE was 3.66 mm for HandAnthro and 10.68 mm for CH-PR (Fig. 4). Background whitening. Adding BW after PR (A1 to A2) slightly increased validation pose loss for YOLOv8 and decreased it for YOLOv11 under both augmentation settings (Appendix M). We compared HandAnthro’s BW, which reuses the hand mask from PR, with REMBG, BackgroundRemover, and CarveKit, leaving PR, the landmark predictor, and PP unchanged. All four methods received the same 704 rectified images, and each pipeline returned all 44 dimensions for every image. Among these 704 images, MAE against caliper measurements was 3.80 mm with HandAnthro’s BW and 4.26–4.71 mm with the alternatives. Mean processing time for the BW stage alone was 17 ms per image for HandAnthro and 93–379 ms for the alternatives (Appendix I.2).

Mean absolute difference between condition means (mm)  
![](images/a6a6bdde86d2600314b2404d847d554bf7fa91d4a1bd256da0a702432c48565c.jpg)  
(a) HandAnthro

![](images/c776f188114eadf72a64feeb7dbe9de2eaca10991a4f33f8e226dbfdd01a8a4e.jpg)  
(b) CH-PR

![](images/273cc592599fd565abfe349402e3239f122ab66b67d7b8b05eab5b107ba7e154.jpg)  
(c) MAE Results  
Figure 4: Perspective-rectification comparison.

Landmark-model selection. Among the 12 A0–A2 training configurations, the augmented YOLOv11x-pose model had the lowest A2 validation pose loss, which was 1.5% above the overall A1 minimum. We selected it because A3 requires the background-whitened contour for PP; full configuration results appear in Appendix M.

Geometry-constrained refinement. On 45 annotated test images, PP reduced mean 2D landmark error from 11.18 to 7.83 px (29.9% improvement); mean error decreased on 44 of the 45 images. On the 704 captures completed by both configurations (45 participants), caliper-referenced MAE increased slightly from 3.707 mm without PP to 3.804 mm with PP. The caliper reference is also subject to measurement uncertainty, and dimension MAE alone does not fully characterize anatomical landmark placement: diferent endpoint locations can produce similar distances. Despite this 0.097 mm (2.62%) increase in dimension MAE, we retain PP in the final framework to improve anatomical landmark placement and support users’ visual inspection (Fig. 1 (f)).

## 5.2 Controlled Repeated-Capture Evaluation

Pipeline completion. HandAnthro completed PR and all 44 measurements for 704/720 controlled captures (97.8%); no post-PR failures occurred.

Measurement accuracy conditional on completion. On the 704 complete outputs, the mean of the 44 caliperreferenced MAEs was 3.80 mm. Anatomical stratification yielded 2.48 mm for the 28 non-thumb finger dimensions, 6.04 mm for the five thumb dimensions, and 6.17 mm for the 11 palm-and-wrist dimensions (details in Appendix J).

Annotation-derived agreement with calipers. Dimensions computed from the manual image annotations had a caliperreferenced MAE of 3.38 mm across 1,980 hand–dimension pairs (45 images and 44 dimensions; Appendix Q).

Measurement accuracy and stability across capture conditions. Among completed captures, aggregate MAEs were similar between the two tested smartphones and between background settings; top-down captures had higher MAE than oblique captures (Fig. 5 (a)).

(a) Accuracy against calipers  
![](images/1e2a5d41baf646af120580c3fc23ff5fe624bdb9b1448e1b4c0cd51ac018e4b5.jpg)

(b) Within-person prediction changes  
![](images/73fbe02021aa6ec05f397fecd6878de7e2f40ae111c1d032b50fd3e5cf39b2f6.jpg)

Figure 5: Capture-condition accuracy and stability in 704 completed captures from 45 participants. (a) Caliperreferenced MAE averaged across 44 dimensions for each condition. (b) Absolute diferences between the two conditionspecific mean predictions, computed for each participant and dimension and then averaged across participants and dimensions. Both panels summarize completed outputs; seven participants had incomplete capture-condition grids.

Across 1,980 participant–dimension combinations (45 participants, 44 dimensions), within-person SD had a median of 1.25 mm and a 95th percentile of 3.99 mm. Mean absolute diferences between each participant’s conditionspecific mean predictions were largest for angle and smallest for smartphone (Fig. 5 (b)). Per-dimension summaries and caliper-referenced angle contrasts appear in Appendices N.1 and O.

## 5.3 Researcher-Assisted Mobile-App Field Pilot

HandAnthro returned all 44 dimensions for 260/268 retained images (97.0%). Among the complete cases (197 male and 63 female), comparison with an independent national firefighter reference yielded a mean absolute diference of 2.40 mm across 28 sex-by-dimension group-mean contrasts (Hsiao et al. 2015) (Appendix S).

## 5.4 Failure Analysis

All failures occurred during PR (16/720 controlled; 8/268 field), and every PR-complete capture yielded all 44 measurements. Post-hoc review associated every controlled failure with wrist placement at or beyond a paper edge. Five field failures involved hands placed too high or low, with adaptive paper prompts outside the sheet. The remaining field failures involved incomplete paper framing, a hand extending beyond the sheet, or a covered corner (Appendix T).

## 6 Discussion

Principal Findings. HandAnthro achieved 97.8% controlled completion and 3.80 mm caliper-referenced MAE. PR standardized input geometry for landmark prediction, and its addition reduced validation pose loss by 77–94% across matched configurations. BW supplied the hand contour required by PP. Although its efect on validation pose loss varied by backbone, the implementation that reused the PR-stage mask yielded lower caliper-referenced MAE and shorter runtime than the three tested BW alternatives. PP reduced mean 2D landmark localization error by 29.9%, while increasing caliper-referenced MAE by 2.62%. We retain PP to improve anatomical landmark placement and support users’ visual inspection of where each measurement is taken.

Interpreting Measurement Agreement. The overall MAE of 3.80 mm combines dimensions with diferent error levels. MAE was 2.48 mm for the 28 non-thumb finger dimensions, 6.04 mm for the five thumb dimensions, and 6.17 mm for the 11 palm-and-wrist dimensions. The overall average therefore does not represent the same level of agreement across all anatomical regions.

For comparison, Kaashki et al. (2022) reported a mean MAE of 4.5 mm across 11 dimensions from 20 real hand scans. HandAnthro’s 3.80 mm result is on the same broad scale of several millimeters as this published result. However, diferences in dimension definitions, study samples, and reference measurements prevent a direct accuracy ranking. Table 1 provides this measurement context alongside the methods’ automation and equipment requirements.

Dimensions derived from manually annotated landmarks still had an MAE of 3.38 mm against the caliper reference. Thus, using manual annotations did not eliminate discrepancies between image-derived dimensions and caliper measurements. Possible contributors include endpoint-definition diferences, image projection, and uncertainty in annotation, rectification, and caliper measurement.

Practical Implications. The capture-condition analyses describe both agreement with calipers and the stability of predictions for the same person. Across the tested settings, average within-person prediction changes were larger for angle than for smartphone or background (Fig. 5 (b)). This pattern motivates clearer capture-angle guidance and evaluation of whether such guidance improves measurement consistency.

The trained operator estimated that it would take about 15 minutes to measure all 44 dimensions of one hand with calipers. Once a photograph was available, HandAnthro extracted 44 projected dimensions with mean processing times of about 4 seconds on an NVIDIA RTX 4000 Ada graphics processing unit (GPU) and 26 seconds on an Apple M3 processor. These processing times exclude image collection. A smartphone and letter-size paper sufice for image capture, while automated extraction removes the need for manual landmark placement or for a trained operator to measure each dimension with calipers. Together, the simple capture requirements and short processing times ofer potential for collecting hand anthropometric data from larger populations.

Limitations. Evaluation is limited to 45 controlled participants and a researcher-assisted field cohort without matched caliper measurements. Generalization to unassisted capture, additional devices, sites, and occupational domains remains to be evaluated.

Path to Deployment. Building on these results, we will evaluate unassisted app use, retaining all captures to assess attempts per user, guidance adherence, and completion rates overall and among compliant images. Planned AWS GPU inference will return measurements or recapture feedback, with response time and prediction quality evaluated. Pilot participants’ concerns about fingerprint exposure in submitted palmar images motivate future evaluation of dorsal images as an alternative input for hand anthropometry. To address dimension-dependent biases, particularly in palm and wrist measurements, we will investigate ofset and scale corrections using paired image-derived and caliper measurements, comparing shared and occupation-specific calibration on independent participants.

## Ethics Statement and Resource Availability

The institutional review board (IRB) deemed this study exempt under 45 CFR 46.104(d)(3)(i)(B). Potentially identifying participant images and individual-level data are not publicly released. The supplementary material provides the code, runnable sample, schemas, and aggregate results. Access to individual-level data requires prior IRB modification review; qualified researchers may contact the authors.

## Acknowledgments

This work was supported by the Fire Prevention and Safety (FP&S) Research and Development Grant Program, administered by the Federal Emergency Management Agency (FEMA), U.S. Department of Homeland Security, under Award No. EMW-2022-FP-00162.

This work used the Delta system at the National Center for Supercomputing Applications through allocation CIS240389 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, which is supported by National Science Foundation grants #2138259, #2138286, #2138307, #2137603, and #2138296 (Boerner et al. 2023). This research used the Delta advanced computing and data resource which is supported by the National Science Foundation (award OAC 2005572) and the State of Illinois. Delta is a joint efort of the University of Illinois Urbana-Champaign and its National Center for Supercomputing Applications.

## References

Boerner, T. J.; Deems, S.; Furlani, T. R.; Knuth, S. L.; and Towns, J. 2023. ACCESS: Advancing Innovation: NSF’s Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support. In Practice and Experience in Advanced Re-

search Computing 2023: Computingfor the Common Good, 173–176. Association for Computing Machinery.

Canny, J. 1986. A Computational Approach to Edge Detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-8(6): 679–698.

Dianat, I.; Haslegrave, C. M.; and Stedmon, A. W. 2012. Methodology for evaluating gloves in relation to the efects on hand performance capabilities: a literature review. Ergonomics, 55(11): 1429–1451.

Douglas, D. H.; and Peucker, T. K. 1973. Algorithms for the reduction of the number of points required to represent a digitized line or its caricature. Cartographica, 10(2): 112– 122.

Gatis, D. 2025. rembg. GitHub repository, https://github. com/danielgatis/rembg. Version 2.0.66.

Gordon, C. C.; Churchill, T.; Clauser, C. E.; Bradtmiller, B.; and McConville, J. T. 1989. Anthropometric survey of US army personnel: methods and summary statistics 1988. Technical report, United States Army Natick Research, Development and Engineering Center.

Grifin, L.; Kim, N.; Carufel, R.; Sokolowski, S.; Lee, H.; and Seifert, E. 2018a. Dimensions of the dynamic hand: implications for glove design, fit, and sizing. In International Conference on Applied Human Factors and Ergonomics, 38– 48. Springer.

Grifin, L.; Sokolowski, S.; Lee, H.; Seifert, E.; Kim, N.; and Carufel, R. 2018b. Methods and tools for 3D measurement of hands and feet. In International Conference on Applied Human Factors and Ergonomics, 49–58. Springer.

Habibi, E.; Soury, S.; and Zadeh, A. H. 2013. Precise evaluation of anthropometric 2D software processing of hand in comparison with direct method. Journal ofMedical Signals & Sensors, 3(4): 256–261.

Han, H. S.; and Park, C. K. 2016. Automatic hand measurement system from 2D hand image for customized glove production. The Korean Fashion and Textile Research Journal, 18(4): 468–476.

Hsiao, H.; Li, R.; Zhang, M.; and Song, G. 2026. Firefighter gloves sizing: Coverage, practicality, and efectiveness. Applied Ergonomics, 132: 104696.

Hsiao, H.; Whitestone, J.; Kau, T.-Y.; and Hildreth, B. 2015. Firefighter hand anthropometry and structural glove sizing: a new perspective. Humanfactors, 57(8): 1359–1377.

Jocher, G.; Chaurasia, A.; and Qiu, J. 2023. Ultralytics YOLOv8.

Jocher, G.; and Qiu, J. 2024. Ultralytics YOLO11.

Kaashki, N. N.; Dai, X.; Gyarmathy, T.; Hu, P.; Iancu, B.; and Munteanu, A. 2022. Automatic and fast extraction of 3D hand measurements using a deep neural network. In 2022 IEEE International Instrumentation and Measurement Technology Conference (I2MTC), 1–6. IEEE.

Ke, L.; Ye, M.; Danelljan, M.; Liu, Y.; Tai, Y.-W.; Tang, C.-K.; and Yu, F. 2023. Segment anything in high quality. Advances in Neural Information Processing Systems, 36: 29914–29934.

Li, Z.; Chang, C.-C.; Dempsey, P. G.; Ouyang, L.; and Duan, J. 2008. Validation of a three-dimensional hand scanning and dimension extraction method with dimension data. Ergonomics, 51(11): 1672–1692.

Magno, K. J. H.; and Pabico, J. P. 2013. Towards Input Device Satisfaction Through Hand Anthropometry. Philippine Information Technology Journal, 6(1): 17–28. Also available as arXiv:1507.06029.

Matas, J.; Galambos, C.; and Kittler, J. 2000. Robust Detection of Lines Using the Progressive Probabilistic Hough Transform. Computer Vision and Image Understanding, 78(1): 119–137.

Nader, J. 2025. BackgroundRemover. GitHub repository, https://github.com/nadermx/backgroundremover. Version 0.3.8.

Nguyen, C. T. K.; Le, T. T.; and La, T. N. A. 2025. Analyzing a comprehensive hand-size database using automated landmark detection and measurement techniques. Journal of Applied Science and Engineering, 28(11): 2417–2426.

Patel, T.; Ningthoujam, B.; Kumar, P.; and Gurung, S. 2018. Validation of two-dimensional digital photogrammetry measurement for hand anthropometric dimensions. J Ergon, 8(04).

Sandnes, F. E. 2014. Measuring 2D:4D Finger Length Ratios with Smartphone Cameras. In 2014 IEEE International Conference on Systems, Man, and Cybernetics, 1697–1701. IEEE.

Selin, N. 2024. CarveKit: Automated high quality background removal framework. GitHub repository, https://github.com/OPHoperHPO/image-backgroundremove-tool. Version 4.1.2.

Sorock, G. S.; Lombardi, D. A.; Hauser, R.; Eisen, E. A.; Herrick, R. F.; and Mittleman, M. A. 2004a. A case-crossover study of transient risk factors for occupational acute hand injury. Occupational and Environmental Medicine, 61(4): 305–311.

Sorock, G. S.; Lombardi, D. A.; Peng, D. K.; Hauser, R.; Eisen, E. A.; Herrick, R. F.; and Mittleman, M. A. 2004b. Glove Use and the Relative Risk of Acute Hand Injury: A Case-Crossover Study. Journal of Occupational and Environmental Hygiene, 1(3): 182–190.

Suvorov, R.; Logacheva, E.; Mashikhin, A.; Remizova, A.; Ashukha, A.; Silvestrov, A.; Kong, N.; Goka, H.; Park, K.; and Lempitsky, V. 2022. Resolution-robust large mask inpainting with fourier convolutions. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, 2149–2159.

Yang, Y.; Xu, J.; Elkhuizen, W. S.; and Song, Y. 2021. The development of a low-cost photogrammetry-based 3D hand scanner. HardwareX, 10: e00212.

Yu, A.; Yick, K. L.; Ng, S.; and Yip, J. 2013. 2D and 3D anatomical analyses of hand dimensions for custom-made gloves. Applied ergonomics, 44(3): 381–392.

Zhang, F.; Bazarevsky, V.; Vakunov, A.; Tkachenka, A.; Sung, G.; Chang, C.-L.; and Grundmann, M. 2020. Mediapipe hands: On-device real-time hand tracking. arXiv preprint arXiv:2006.10214.

# Supplementary Material

Fan Zhou<sup>1</sup>, Shuairan Chen<sup>1</sup>, Mengying Zhang<sup>1</sup>, Yulin Wu<sup>1</sup>, Sadegh Jafari<sup>1</sup>, Sixing Yu<sup>2</sup>, Rui Li<sup>1</sup>, Ali Jannesari<sup>1</sup>, Guowen Song<sup>1</sup>

<sup>1</sup>Iowa State University <sup>2</sup>Microsoft

## A Test-Set Capture Conditions

Figure S1 illustrates the eight controlled conditions spanning two smartphones, two backgrounds, and two capture angles.

## B Capture Rig Geometry

Figure S2 shows the two-holder capture rig and its measured front-view geometry.

The top-down clamp holds the phone’s bottom edge 28.7 cm above the sheet. For the oblique clamp, the near and far ends are 25 and 28 cm above the sheet across a 7 cm phone width, giving a tilt of arcsin $( 3 / 7 ) \approx 2 5 ^ { \circ }$ from vertical (Fig. S2). Phones were manually re-mounted between shots; one representative top-down mounting was tilted about $5 ^ { \circ }$ and individual image angles were not measured.

## C Anthropometric Landmark Definitions

HandAnthro predicts 41 anthropometry-specific 2D landmarks on each palmar hand image. Fig. S3 shows them on a representative hand, and Table S1 gives their per-region names. From these 41 directly predicted landmarks, the metric-conversion stage deterministically derives 14 midpoint landmarks (Table S2). For landmarks ${ k _ { a } } = ( x _ { a } , y _ { a } )$ and $k _ { b } = ( x _ { b } , y _ { b } )$ , each derived midpoint is

$$
m ( a , b ) = \frac { k _ { a } + k _ { b } } { 2 } = \left( \frac { x _ { a } + x _ { b } } { 2 } , \frac { y _ { a } + y _ { b } } { 2 } \right) .\tag{1}
$$

The submitted implementation contains exactly 44 endpoint pairs, listed with their anatomical names and errors in Table S5.

## D Adaptive SAM-HQ Prompts

The PR stage uses separate SAM-HQ calls to produce the paper mask $M _ { \mathrm { p a p e r } }$ and raw hand mask $M _ { \mathrm { h a n d - r a w } }$ . Positive prompts mark the target region; negative prompts mark regions to exclude from that mask. Let $m _ { j } ~ = ~ ( x _ { j } , y _ { j } )$ $j = 0 , \ldots , 2 0$ , denote the MediaPipe landmarks in image coordinates, with x increasing rightward and y downward. Here $m _ { 0 } , m _ { 9 }$ , and $m _ { 1 2 }$ locate the wrist, middle-finger root, and middle fingertip, respectively. The middle-finger rootto-tip Euclidean distance, $L = \| m _ { 1 2 } - m _ { 9 } \| _ { 2 }$ , sets the scale of the prompt ofsets.

Paper-mask prompts. The four positive prompts are labeled top left (TL), top right (TR), bottom left (BL), and bottom right (BR):

$$
\begin{array} { l } { { p _ { \mathrm { T L / T R } } = \mathrm { c l i p } \displaystyle \left( m _ { 1 2 } + \left( \mp \frac { L } { 1 . 5 } , - \frac { L } { 7 } \right) \right) , } } \\ { { p _ { \mathrm { B L / B R } } = \mathrm { c l i p } \displaystyle \left( m _ { 0 } + \left( \mp \frac { L } { 1 . 5 } , \frac { L } { 1 0 } \right) \right) , } } \end{array}\tag{2}
$$

The upper pair starts at the middle fingertip $m _ { 1 2 } .$ , moves $L / 1 . 5$ left or right, and $L / 7$ upward. The lower pair starts at the wrist $m _ { 0 } .$ , uses the same horizontal ofset, and moves $L / 1 0$ downward. In ∓, the minus sign gives TL/BL and the plus sign gives TR/BR; clip keeps each point within the image bounds. The single negative prompt $n = m _ { 9 }$ marks the hand interior for exclusion from the paper mask. No box prompt is used. Figure S4a shows the paper prompts.

Hand-mask prompts. Let P and H denote the paper and hand prompt sets, with superscripts + and − indicating positive and negative labels. Define $\begin{array} { r l } { \mathcal { P } ^ { + } } & { { } = } \end{array}$ {p<sub>TL</sub>, p<sub>TR</sub>, p<sub>BL</sub>, p<sub>BR</sub>} from Eq. 2. Hand segmentation uses all 21 MediaPipe landmarks as positive prompts and reuses these four paper locations as negative prompts:

$$
\begin{array} { l l } { { \mathcal { P } ^ { - } = \{ m _ { 9 } \} , } } \\ { { \mathcal { H } ^ { + } = \{ m _ { 0 } , \dots , m _ { 2 0 } \} , \qquad \mathcal { H } ^ { - } = \mathcal { P } ^ { + } . } } \end{array}\tag{3}
$$

Before LaMa inpainting, $M _ { \mathrm { h a n d - r a w } }$ is dilated once using an all-ones $3 1 \times 3 1$ square structuring element (15-pixel halfwidth). This expanded mask is used only for the LaMa inpainting region; BW later reuses the undilated hand mask.

## E Quadrilateral Detection and Homography

The PR stage recovers the paper quadrilateral from the inpainting-completed paper mask $M _ { \mathrm { p a p e r - c o m } }$ using the following fixed parameters. The mask is smoothed with an $1 1 \times \bar { 1 } 1$ Gaussian kernel, then processed with Canny edge detection (Canny 1986) using thresholds 50 and 150. Morphological closing uses a $3 \times 3$ rectangular kernel for three dilation and erosion iterations. The five largest contours are approximated using Douglas–Peucker simplification (Douglas and Peucker 1973) with tolerance $\varepsilon = 0 . 0 8 L _ { \mathrm { p e r i } } .$ , where $L _ { \mathrm { p e r i } }$ is the contour perimeter; the first candidate with exactly four vertices and area greater than 30% of the image is accepted.

![](images/bffea82d31e59441cd4be5e4d8272ec230c1f88d5c615270efd020c1145f249d.jpg)  
(a) iPhone, simple background, top-down

![](images/fcd459d873aabd46ac0f482f87766a927b8671743c9970258a234da12c37d8b8.jpg)  
(b) iPhone, complex ground, top-down

![](images/98caccdf7752da8fe70f7b8ae6ade02e9e1a6f8f122da7f51bec063d18a15063.jpg)  
(c) Android, simple background, top-down

![](images/7050c4da85e11f3188db826eebe7c11ce231caf32284dfa2ac0a350691657b29.jpg)  
(d) Android, complex background, top-down

![](images/41154f02d1f2e00b5dfd04294a30c67eae8898b53d1f0a8d4255f1ce00e26353.jpg)  
(e) iPhone, simple background, oblique

![](images/043900cda4e199b33e407d8f2811d1f30cdb4da254a4b16688f439d0287cfbb7.jpg)  
(f) iPhone, complex background, oblique

![](images/3f346866ab15f473f887fec0bcf372d7a55a136154f9d7dafa48d746b60f6971.jpg)  
(g) Android, simple background, oblique

![](images/cbff24cbfd106f2aa2fee6f94fa8b59870973777ce2837f00c0e1e142f88f036.jpg)  
(h) Android, complex background, oblique  
Figure S1: One author’s hand under the eight test-set conditions (two devices, two backgrounds, and two capture angles; illumination pooled). Fingerprints are removed.

<table><tr><td>Region</td><td>Abbreviation range</td><td>Count</td><td>Role</td></tr><tr><td>Thumb</td><td> $T _ { 1 } ~ \mathrm { t o } ~ T _ { 6 }$ </td><td>6</td><td>Thumb-specific landmarks</td></tr><tr><td>Index finger</td><td> $I _ { 1 } \ \mathrm { t o } \ I _ { 7 }$ </td><td>7</td><td>Fingertip to finger root</td></tr><tr><td>Middle finger</td><td> $M _ { 1 } \mathrm { t o } M _ { 7 }$ </td><td>7</td><td>Fingertip to finger root</td></tr><tr><td>Ring finger</td><td> $R _ { 1 } ~ \mathrm { t o } ~ R _ { 7 }$ </td><td>7</td><td>Fingertip to finger root</td></tr><tr><td>Little finger</td><td> $L _ { 1 } \mathrm { t o } L _ { 7 }$ </td><td>7</td><td>Fingertip to finger root</td></tr><tr><td>Inter-finger webbing</td><td> $\mathrm { R O _ { 1 } } ~ \mathrm { t o } ~ \mathrm { R O _ { 3 } }$ </td><td>3</td><td>Root-of-finger webbing points</td></tr><tr><td>Wrist / palm</td><td> $W _ { 1 } , W _ { 2 } , P _ { 1 } , P _ { 2 }$ </td><td>4</td><td>Wrist and palm reference points</td></tr><tr><td>Total</td><td colspan="3">41</td></tr></table>

Table S1: The 41 anthropometry-specific landmarks predicted directly by the YOLO model, grouped by anatomical region.

<table><tr><td>Abbrev.</td><td>Expansion</td><td>Anatomical joint (palmar crease)</td><td>Formula</td></tr><tr><td> $\mathrm { M F K } _ { I }$ </td><td>Mid First Knuckle (Index)</td><td>Index-finger DIP crease</td><td> ${ \mathrm { m i d p o i n t } } ( I _ { 2 } , I _ { 3 } )$ </td></tr><tr><td> $\mathrm { M F K } _ { M }$ </td><td>Mid First Knuckle (Middle)</td><td>Middle-finger DIP crease</td><td> $\mathrm { m i d p o i n t } ( M _ { 2 } , M _ { 3 } )$ </td></tr><tr><td> $\mathrm { M F K } _ { R }$ </td><td>Mid First Knuckle (Ring)</td><td>Ring-finger DIP crease</td><td>midpoint(  $R _ { 2 } , R _ { 3 } )$ </td></tr><tr><td> $\mathrm { M F K } _ { L }$ </td><td>Mid First Knuckle (Little)</td><td>Little-finger DIP crease</td><td> ${ \mathrm { m i d p o i n t } } ( L _ { 2 } , L _ { 3 } )$ </td></tr><tr><td> $\mathrm { M F K } _ { T }$ </td><td>Mid First Knuckle (Thumb)</td><td>Thumb IP crease</td><td> ${ \mathrm { m i d p o i n t } } ( T _ { 2 } , T _ { 3 } )$ </td></tr><tr><td> $\mathrm { M S K } _ { I }$ </td><td>Mid Second Knuckle (Index)</td><td>Index-finger PIP crease</td><td> ${ \mathrm { m i d p o i n t } } ( I _ { 4 } , I _ { 5 } )$ </td></tr><tr><td> $\mathrm { M S K } _ { M }$ </td><td>Mid Second Knuckle (Middle)</td><td>Middle-finger PIP crease</td><td> $\mathrm { m i d p o i n t } ( M _ { 4 } , M _ { 5 } )$ </td></tr><tr><td> $\mathrm { M S K } _ { R }$ </td><td>Mid Second Knuckle (Ring)</td><td>Ring-finger PIP crease</td><td> $\mathrm { m i d p o i n t } ( R _ { 4 } , R _ { 5 } )$ </td></tr><tr><td> $\mathrm { M S K } _ { L }$ </td><td>Mid Second Knuckle (Little)</td><td>Little-finger PIP crease</td><td> $\mathrm { m i d p o i n t } ( L _ { 4 } , L _ { 5 } )$ </td></tr><tr><td> $\mathrm { M T K } _ { I }$ </td><td>Mid Third Knuckle (Index)</td><td>Index-finger MCP crease</td><td> $\mathrm { m i d p o i n t } ( I _ { 6 } , I _ { 7 } )$ </td></tr><tr><td> $\mathrm { M T K } _ { M }$ </td><td>Mid Third Knuckle (Middle)</td><td>Middle-finger MCP crease</td><td> $\mathrm { m i d p o i n t } ( M _ { 6 } , M _ { 7 } )$ </td></tr><tr><td> $\mathrm { M T K } _ { R }$ </td><td>Mid Third Knuckle (Ring)</td><td>Ring-finger MCP crease</td><td> $\mathrm { m i d p o i n t } ( R _ { 6 } , R _ { 7 } )$ </td></tr><tr><td> $\mathrm { M T K } _ { L }$ </td><td>Mid Third Knuckle (Little)</td><td>Little-finger MCP crease</td><td> $\mathrm { m i d p o i n t } ( L _ { 6 } , L _ { 7 } )$ </td></tr><tr><td>MoW</td><td>Midpoint of Wrist</td><td>Midpoint of wrist line</td><td> ${ \mathrm { m i d p o i n t } } ( W _ { 1 } , W _ { 2 } )$ </td></tr></table>

Table S2: The 14 midpoint landmarks derived from the 41 directly predicted anthropometry-specific landmarks.

![](images/05f380164325dca14d4e2eec20e82cede689406afe9ce599d54bea4bc3d41cbd.jpg)  
(a) Physical rig.

![](images/13c82df2ea47b24f12327ef1a5d54a5d9284df0b944e97039c21d2cca617661f.jpg)  
(b) Front-view geometry.

Figure S2: Controlled capture rig: the left clamp provides the top-down view and the right clamp the oblique view. Distances are in centimeters; the measured oblique configuration is approximately $2 5 ^ { \circ }$ from vertical.  
![](images/e519cf7cd041f382305bdf02df9988ec5ea282883ba5f677ee1cb2c59eedcfcb.jpg)  
Figure S3: The 41 anthropometry-specific landmarks on a palmar hand, labeled by abbreviation (Table S1).

![](images/4c08f87000a3068ea8bd8a444e9edaac0a8c79cfe1f08dfa079002720561a996.jpg)

![](images/ddebc27d1f1394252fb81ce07b902f6d5ff86f105f59a32faa29a833a56492c9.jpg)  
(a) Paper-mask prompts.  
(b) Hand-mask prompts.  
Figure S4: Illustrative SAM-HQ prompts for (a) papermask and (b) hand-mask generation. Red points are positive prompts; blue points are negative prompts.

Let the accepted vertices, ordered top-left, top-right, bottom-right, bottom-left, be $q _ { i } ,$ and let $q _ { i } ^ { \prime }$ be the corresponding corners of the destination rectangle. Its width and height are initialized from the larger opposing side lengths; then the smaller adjustment required to enforce the letter-paper ratio $H ^ { \prime } / W ^ { \prime } \stackrel { \sim } { = } 1 1 / 8 . 5$ is applied. The homography H is the projective map satisfying

$$
\begin{array} { r l } { \lambda _ { i } \left[ \boldsymbol { q } _ { i , x } ^ { \prime } \right. } & { \boldsymbol { q } _ { i , y } ^ { \prime } \left. \mathrm {  ~ 1 ~ } \right] ^ { T } = { \bf { \cal { H } } } \left[ \boldsymbol { q } _ { i , x } \quad \boldsymbol { q } _ { i , y } \quad 1 \right] ^ { T } , } \\ & { i = 1 , \ldots , 4 . } \end{array}\tag{4}
$$

The same H warps the original RGB image and the undilated hand mask, preserving their pixel correspondence. Immediately before landmark inference, both rectified outputs are resized to the canonical $7 2 0 \times 9 3 2 { \mathrm { p x } }$ model frame used for metric conversion (Section 3).

## F Background Whitening

Fig. S5 illustrates background whitening (BW). The RGB image and undilated hand mask are warped using the same homography to keep them aligned. The stored mask represents the hand in black; after warping, a pixel is treated as hand foreground if at least one mask channel has a value below 100. BW preserves the rectified image’s RGB values at these foreground pixels and replaces all remaining pixels with white (255 in each RGB channel).

![](images/9ae60ff71f3277168c0b99ff8718c4b0c2d0af77db73fc815e2ed012f9f45a58.jpg)  
(a)

![](images/1b5de5b6d35ae0875336b1b92c1fd8efa3eed87cd0a7bf6d9f6260b137abf708.jpg)  
(b)

![](images/f7337c60b38b45d62413df3815649b3dcb9647fcfba7b2487e62a8f482322842.jpg)  
(c)

![](images/3cb7fcb4bf17e9f4546320741b009dcc9ce4298403844be38c752e194b6a1560.jpg)  
(d)  
Figure S5: Background whitening using the warped hand mask: (a) raw hand mask; (b) detected paper quadrilateral (green); (c) rectified hand mask; (d) background-whitened image.

## G YOLO Fine-Tuning Hyperparameters

Table S3 lists the key YOLO-Pose fine-tuning settings used for all 16 configurations.

<table><tr><td>Item</td><td>Value</td></tr><tr><td>Epochs / batch size</td><td>8000 / 40</td></tr><tr><td>Early stopping patience</td><td>10000</td></tr><tr><td>Compute</td><td>CUDA GPU; dataloader workers: 8 auto (Ultralytics default selection)</td></tr><tr><td>Optimizer Learning rate schedule</td><td>lr0 = 0.001, lrf = 0.01</td></tr><tr><td>Momentum / weight decay</td><td> $0 . 9 3 7 / 5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Warmup</td><td>3 epochs (warmup_epochs=3.0)</td></tr><tr><td>Regularization</td><td>dropout = 0.9</td></tr><tr><td>Loss weights</td><td>box = 0.2, cls = 0.1, pose = 20.0, kobj = 3.0</td></tr><tr><td>Augmentation (selected)</td><td>translate = 0.1, scale = 0.5, fliplr = 0.5, mosaic = 1.0, auto_augment = randaugment, erasing = 0.4</td></tr><tr><td>Reproducibility</td><td>seed = 0, deterministic = True, AMP = True</td></tr></table>

Table S3: Key hyperparameters and settings for YOLO finetuning.

## H Geometry-Constrained Post-Processing

PP runs MediaPipe Hands on $I _ { \mathrm { w h i t e } }$ to obtain 21 auxiliary landmarks ${ \kappa } ^ { M P }$ , then refines the 41 YOLO landmarks $\mathcal { K } ^ { \check { Y } }$ using contour searches on the same image. It constructs separation lines, estimates finger axes and fingertips, applies per-finger similarity transforms, and refines boundary endpoints in that order. A valid boundary pixel is non-white, has at least three non-white pixels in its $3 \times 3$ neighborhood, and has at least one white neighbor. The per-channel white threshold is 250 for finger-axis estimation and 230 for subsequent refinement.

Separation lines and finger axes. Three separation lines, index–middle, middle–ring, and ring–little, connect each neighboring pair’s MediaPipe MCP midpoint to its PIP midpoint. They stop boundary-search rays from crossing into adjacent fingers; the thumb has no separation constraint. For finger $f ,$ let $r _ { f }$ be its MediaPipe root and $c _ { f }$ the midpoint of its fingertip and adjacent tip-side joint. Search from $c _ { f }$ perpendicular to $c _ { f } - r _ { f }$ in both directions for the first boundary pixels $e _ { f } ^ { L } , e _ { f } ^ { R }$ . Their midpoint and the corrected axis are

$$
u _ { f } = \frac { e _ { f } ^ { L } + e _ { f } ^ { R } } { 2 } , \qquad d _ { f } = \frac { u _ { f } - r _ { f } } { \| u _ { f } - r _ { f } \| _ { 2 } } .\tag{5}
$$

A forward search from $u _ { f }$ along $d _ { f }$ finds the corrected fingertip $t _ { f }$ (Fig. S6).

![](images/c0358cda8f0095ed8b52395605ed722c21fbd66bfd1261fe6418619d72cb30c1.jpg)  
(a)

![](images/07a1d266c8a33022b7fce42d5f43987dac52341b2888f83b7912959f71469fbb.jpg)  
(b)  
Figure S6: Finger-axis estimation and fingertip correction for finger $f .$

Per-finger similarity transform. Let $q _ { f }$ and $c _ { f } ^ { y }$ be the YOLO fingertip and fixed root. The signed angle $\theta _ { f }$ rotates $q _ { f } - c _ { f } ^ { y }$ onto $t _ { f } - c _ { f } ^ { y }$ . With $R ( \theta _ { f } )$ the 2D rotation matrix, each member $k _ { i }$ of finger group $G _ { f }$ is updated by

$$
\begin{array} { l } { s _ { f } = \frac { \lvert | t _ { f } - c _ { f } ^ { y } \rvert | _ { 2 } } { \lvert | q _ { f } - c _ { f } ^ { y } \rvert | _ { 2 } } , } \\ { k _ { i } ^ { \prime } = c _ { f } ^ { y } + s _ { f } R ( \theta _ { f } ) ( k _ { i } - c _ { f } ^ { y } ) . } \end{array}\tag{6}
$$

Boundary snapping and special-point refinement. From each endpoint, search bidirectionally along its current configured measurement-pair line for the nearest valid boundary within 0.03W (21.6 px at $W \ = \ 7 2 0 )$ , checking a $3 \times 3$ neighborhood at each sampled position. Exclude fingertips, derived midpoints (including MoW), $T _ { 5 } , T _ { 6 } ,$ , and $\mathrm { R O } _ { 1 : 3 }$ from this bounded snap. Unresolved wrist endpoints use an uncapped search along the wrist line, inward on white pixels and outward on hand pixels. Recompute MoW, then check $\mathrm { R O } _ { 1 : 3 }$ against neighboring finger-root boundaries and move them toward MoW when needed. If $T _ { 5 }$ lies on a white pixel, search from it only toward MoW; subsequently correct an ofboundary $T _ { 4 }$ by uncapped directional search along $T _ { 5 }  T _ { 4 }$ Correct any of-boundary $T _ { 2 } / T _ { 3 }$ by uncapped search along their incoming pair axis, outward from the other endpoint on hand pixels and inward toward it on white pixels. Recompute all midpoint landmarks (Eq. 1) and the 44 metric dimensions (Section 3) after refinement.

## I Stage-Level Comparison Methods

This section documents the alternatives substituted for HandAnthro’s PR and BW stages.

## I.1 Canny–Hough Perspective Rectification (CH-PR)

We implement a fully automatic, learning-free baseline for paper rectification using edge and line geometry. Given an input image, we compute a Canny edge map (Canny 1986) and extract line segments via the probabilistic Hough transform (Matas, Galambos, and Kittler 2000) (OpenCV HoughLinesP). Segments are clustered into two dominant orientation groups and assigned to the four paper sides, and each side line is fitted from its supporting segments. We select the candidate with maximal edge support on the Canny map. Paper corners are computed from intersections of adjacent side lines, and the quadrilateral is required to be convex. The image is then warped to a canonical rectangle via homography.

Failure-mode breakdown. On the 720-image test set, CH-PR returned 412 convex, warpable quadrilaterals (57.2%), with downstream processing completed for all corresponding captures. MAE on the 406 captures completed by both pipelines was 10.68 mm for CH-PR and 3.66 mm for HandAnthro. The 308 non-warpable cases comprised 266 without two orthogonal line groups, 32 with non-convex side-line intersections, and 10 with an unrecoverable missing side.

## I.2 Background Removal Baselines (REMBG, BackgroundRemover, and CarveKit)

To evaluate the accuracy and computational eficiency of our BW module, we compare it with REMBG (Gatis 2025), BackgroundRemover (Nader 2025), and CarveKit (Selin 2024) under identical PR, landmark-prediction, and PP stages. Table S4 reports end-to-end dimension MAE and BW-only runtime, while Fig. S8 shows representative mask behavior under inter-finger shadows. Every variant produced complete outputs for all 704 PR-complete captures.

As an architectural fairness check, forcing our BW to run from scratch on each rectified image without reusing the PR-stage hand mask took 1213 ms per image on the same NVIDIA RTX 4000 Ada Generation GPU, compared with 93 ms for REMBG. The production advantage therefore comes from mask reuse rather than a faster standalone segmentation model.

Figure S7 plots the runtime and MAE values in Table S4.

<table><tr><td>Variant</td><td>(mm)</td><td>Mean Median (mm)</td><td>∆MAE (mm / %)</td><td>Runtime (ms)</td></tr><tr><td>Ours</td><td>3.80</td><td>3.45</td><td></td><td>17</td></tr><tr><td>REMBG</td><td>4.26</td><td>3.79</td><td> $+ 0 . 4 6 / + 1 2 . 0 \%$ </td><td>93</td></tr><tr><td>Background</td><td>4.26</td><td>3.82</td><td> $+ 0 . 4 6 / + 1 2 . 1 \%$ </td><td>188</td></tr><tr><td>Remover CarveKit</td><td>4.71</td><td>4.48</td><td> $+ 0 . 9 1 / + 2 3 . 9 \%$ </td><td>379</td></tr></table>

Table S4: Background-whitening alternatives under identical PR, landmark-prediction, and PP stages. MAE is measured against calipers on the 704 PR-complete captures, all of which yielded complete outputs for every variant; runtime is BW-only mean latency over the same 704 images. Positive ∆MAE indicates degradation relative to mask reuse.

![](images/1a4939778f86a124a92fc59f0144cc4d259e045474aa9e60115b0352bd58b25e.jpg)  
Figure S7: BW-stage runtime versus caliper-referenced dimension MAE on the 704 complete captures. Runtime is measured on an NVIDIA RTX 4000 Ada Generation GPU; the horizontal axis is logarithmic. Values match Table S4.

![](images/8422be5ffec41c04a739dc5bc71e7f7ffd0e93e18cee6e43aeb718d3efb81238.jpg)  
Input

![](images/e55315863b9eea8f443ec53cf761bab98ee881dac26da8660e10635b29266e17.jpg)  
ours

![](images/822460418e3be90ee4384d9aff3190d768014978ef162b37c8c1707b326f3817.jpg)  
REMBG

![](images/11bd50a6facddd3375d432703e42936b9ec965b0d48fb1a2427b1d7d7487c11c.jpg)  
BackgroundRemover

![](images/04a20f949e499edf2161cf4a268c7ec07b57c9850c85fa73d861eb6d83469e51.jpg)  
CarveKit

Figure S8: Our BW versus the three of-the-shelf baselines on representative images. REMBG, BackgroundRemover, and CarveKit often retain inter-finger shadows that blur the hand boundary, whereas our BW gives a clean separation.

## J Per-Dimension Accuracy Breakdown

Table S5 defines all 44 dimensions and reports their absoluteerror statistics on the same 704 complete captures; Fig. S9 shows their distribution and spatial pattern. Regional MAEs equally average the dimension-specific MAEs within D1– D5 (thumb), D6–D33 (non-thumb fingers), and D34–D44 (palm/wrist). The non-thumb group includes breadths, segment lengths, and complete digit lengths. Table S8 uses a separate length/width grouping.

![](images/94030b8b9070517e96fa251e264fab58113e10ab4b639229c5bf94eaf6029eac.jpg)

![](images/d9ed97c3d0025c1fed8deda10c38166e77eafa4a01ef7a4108bb277ac952e51f.jpg)  
Figure S9: Per-dimension MAE for $E _ { 1 6 }$ across 44 dimensions (704 images, 45 participants). (a) Distribution (median 3.45 mm, mean 3.80 mm). (b) Spatial map in which each line connects a dimension’s two endpoints and is colored by its MAE magnitude.

<table><tr><td>ID</td><td>Measurement</td><td>Endpoints</td><td> $\mathbf { M e a n } \pm \mathbf { S D }$ </td><td>Median</td><td> $P _ { 9 5 }$ </td><td>Range</td></tr><tr><td colspan="6">Thumb (5)</td></tr><tr><td>D1</td><td>Thumb IP breadth</td><td> $T _ { 2 }  T _ { 3 }$ </td><td> $3 . 8 4 \pm 1 . 4 6$ </td><td>3.77</td><td>6.57</td><td>0.98-10.01</td></tr><tr><td>D2</td><td>Thumb MCP breadth</td><td> $T _ { 4 }  T _ { 5 }$ </td><td> $6 . 5 1 \pm 3 . 7 1$ </td><td>6.45</td><td>12.63</td><td>0.06-18.17</td></tr><tr><td>D3</td><td>Thumb distal-segment length</td><td> $T _ { 1 }  \mathrm { M F K } _ { T }$ </td><td> $5 . 6 9 \pm 3 . 0 9$ </td><td>5.36</td><td>10.87</td><td>0.00-14.03</td></tr><tr><td>D4</td><td>Thumb proximal-segment length</td><td> $\mathrm { M F K } _ { T }  T _ { 6 }$ </td><td> $7 . 8 2 \pm 4 . 4 3$ </td><td>7.57</td><td>15.65</td><td>0.05–23.46</td></tr><tr><td>D5</td><td>Thumb total length</td><td> $T _ { 1 }  T _ { 6 }$ </td><td> $6 . 3 2 \pm 4 . 0 8$ </td><td>6.01</td><td>13.49</td><td>0.04–20.18</td></tr><tr><td colspan="7">Index finger (7)</td></tr><tr><td>D6</td><td>Index DIP breadth</td><td> $I _ { 2 } \to I _ { 3 }$ </td><td> $2 . 2 8 \pm 1 . 0 0$ </td><td>2.19</td><td>4.13</td><td>0.08-5.25</td></tr><tr><td>D7</td><td>Index PIP breadth</td><td> $I _ { 4 } \to I _ { 5 }$ </td><td> $2 . 5 3 \pm 0 . 9 9$ </td><td>2.41</td><td>4.33</td><td>0.31-5.87</td></tr><tr><td>D8</td><td>Index MCP breadth</td><td> $I _ { 6 } \to I _ { 7 }$ </td><td> $3 . 5 8 \pm 1 . 6 2$ </td><td>3.52</td><td>6.39</td><td>0.04-10.26</td></tr><tr><td>D9</td><td>Index distal-segment length</td><td> $I _ { 1 } \to \mathrm { M F K } _ { I }$ </td><td> $2 . 4 7 \pm 1 . 6 3$ </td><td>2.30</td><td>5.31</td><td>0.01-7.76</td></tr><tr><td>D10</td><td>Index middle-segment length</td><td> $\mathrm { M F K } _ { I } \to \mathrm { M S K } _ { I }$ </td><td> $1 . 7 1 \pm 1 . 2 2$ </td><td>1.57</td><td>3.79</td><td>0.00–5.81</td></tr><tr><td>D11</td><td>Index proximal-segment length</td><td> $\mathrm { M S K } _ { I } \to \mathrm { M T K } _ { I }$ </td><td> $2 . 1 1 \pm 1 . 5 4$ </td><td>1.85</td><td>5.07</td><td>0.00-7.41</td></tr><tr><td>D12</td><td>Index total length</td><td> $I _ { 1 } \to \mathrm { M T K } _ { I }$ </td><td> $3 . 4 4 \pm 2 . 2 1$ </td><td>3.31</td><td>7.14</td><td>0.03-11.81</td></tr><tr><td colspan="7">Middle finger (7)</td></tr><tr><td>D13 Middle DIP breadth</td><td></td><td> $M _ { 2 } \to M _ { 3 }$ </td><td> $1 . 8 9 \pm 0 . 8 4$ </td><td>1.82</td><td>3.31</td><td>0.02-6.28</td></tr><tr><td>D14</td><td>Middle PIP breadth</td><td> $M _ { 4 } \to M _ { 5 }$ </td><td> $2 . 2 0 \pm 0 . 8 8$ </td><td>2.14</td><td>3.77</td><td>0.34–6.27</td></tr><tr><td>D15</td><td>Middle MCP breadth</td><td> $M _ { 6 } \to M _ { 7 }$ </td><td> $2 . 3 6 \pm 1 . 4 2$ </td><td>2.26</td><td>4.96</td><td>0.01-7.19</td></tr><tr><td>D16</td><td>Middle distal-segment length</td><td> $M _ { 1 } \to \mathrm { M F K } _ { M }$ </td><td> $1 . 6 2 \pm 1 . 1 4$ </td><td>1.43</td><td>3.58</td><td>0.00-5.55</td></tr><tr><td>D17</td><td>Middle middle-segment length</td><td> $\mathrm { M F K } _ { M } \to \mathrm { M S K } _ { M }$ </td><td> $1 . 8 4 \pm 1 . 3 8$ </td><td>1.57</td><td>4.43</td><td>0.00-8.20</td></tr><tr><td>D18</td><td>Middle proximal-segment length</td><td> $\mathrm { M S K } _ { M } \to \mathrm { M T K } _ { M }$ </td><td> $2 . 7 0 \pm 1 . 7 4$ </td><td>2.43</td><td>5.90</td><td>0.01-7.88</td></tr><tr><td>D19</td><td>Middle total length</td><td> $M _ { 1 } \to \mathrm { M T K } _ { M }$ </td><td> $3 . 5 4 \pm 2 . 2 9$ </td><td>3.15</td><td>7.42</td><td>0.01-10.19</td></tr><tr><td colspan="7">Ring finger (7)</td></tr><tr><td>D20</td><td>Ring DIP breadth</td><td> $R _ { 2 } \to R _ { 3 }$ </td><td> $1 . 9 3 \pm 0 . 7 3$ </td><td>1.90</td><td>3.09</td><td>0.00-6.97</td></tr><tr><td>D21</td><td>Ring PIP breadth</td><td> $R _ { 4 } \to R _ { 5 }$ </td><td> $2 . 1 1 \pm 0 . 9 0$ </td><td>2.02</td><td>3.75</td><td>0.01-7.14</td></tr><tr><td>D22</td><td>Ring MCP breadth</td><td> $R _ { 6 } \to R _ { 7 }$ </td><td> $3 . 4 6 \pm 1 . 8 5$ </td><td>3.34</td><td>6.63</td><td>0.00-8.82</td></tr><tr><td>D23</td><td>Ring distal-segment length</td><td> $R _ { 1 } \to \mathrm { M F K } _ { R }$ </td><td> $1 . 8 7 \pm 1 . 5 0 $ </td><td>1.58</td><td>4.36</td><td>0.01-8.95</td></tr><tr><td>D24</td><td>Ring middle-segment length</td><td> $\mathrm { M F K } _ { R } \to \mathrm { M S K } _ { R }$ </td><td> $1 . 9 4 \pm 1 . 2 8$ </td><td>1.76</td><td>4.36</td><td>0.01-6.01</td></tr><tr><td>D25</td><td>Ring proximal-segment length</td><td> $\mathrm { M S K } _ { R } \to \mathrm { M T K } _ { R }$ </td><td> $3 . 6 6 \pm 1 . 9 7$ </td><td>3.68</td><td>6.33</td><td>0.03-12.41</td></tr><tr><td>D26</td><td>Ring total length</td><td> $R _ { 1 } \mathrm {  M T K } _ { R }$ </td><td> $4 . 1 4 \pm 2 . 4 7$ </td><td>4.03</td><td>8.57</td><td>0.01-11.20</td></tr><tr><td colspan="7"></td></tr><tr><td>Little finger (7)</td><td>D27 Little DIP breadth</td><td> $L _ { 2 } \to L _ { 3 }$ </td><td> $1 . 8 7 \pm 0 . 7 8$ </td><td>1.84</td><td>3.21</td><td>0.04–4.79</td></tr><tr><td></td><td>D28 Little PIP breadth</td><td> $L _ { 4 } \to L _ { 5 }$ </td><td> $1 . 8 6 \pm 0 . 9 8$ </td><td>1.80</td><td>3.63</td><td>0.01-7.88</td></tr><tr><td>D29</td><td>Little MCP breadth</td><td> $L _ { 6 } \to L _ { 7 }$ </td><td> $4 . 3 2 \pm 2 . 1 9$ </td><td>4.26</td><td>8.19</td><td>0.05–12.75</td></tr><tr><td>D30</td><td>Little distal-segment length</td><td> $L _ { 1 } \to \mathrm { M F K } _ { L }$ </td><td> $1 . 4 9 \pm 1 . 3 0$ </td><td>1.12</td><td>3.80</td><td>0.00-7.09</td></tr><tr><td>D31</td><td>Little middle-segment length</td><td> $\mathrm { M F K } _ { L } \to \mathrm { M S K } _ { L }$ </td><td> $1 . 6 0 \pm 1 . 1 2$ </td><td>1.41</td><td>3.63</td><td>0.00-4.95</td></tr><tr><td>D32</td><td>Little proximal-segment length</td><td> $\mathrm { M S K } _ { L } \to \mathrm { M T K } _ { L }$ </td><td> $2 . 3 1 \pm 1 . 4 3$ </td><td>2.23</td><td>5.05</td><td>0.01-7.09</td></tr><tr><td>D33</td><td>Little total length</td><td> $L _ { 1 } \to \mathrm { M T K } _ { L }$ </td><td> $2 . 5 6 \pm 2 . 0 5$ </td><td>2.09</td><td>6.59</td><td>0.01-10.91</td></tr><tr><td colspan="7">Palm and wrist (11)</td></tr><tr><td>D34</td><td>Metacarpal hand breadth</td><td> $P _ { 1 }  P _ { 2 }$ </td><td> $7 . 6 7 \pm 2 . 4 2$ </td><td>7.77</td><td>11.41</td><td>1.08-15.40</td></tr><tr><td>D35</td><td>Wrist breadth Thumb root-wrist-midpoint length</td><td> $W _ { 1 } \to W _ { 2 }$   $T _ { 6 } \to \mathrm { M o W }$ </td><td> $1 1 . 0 0 \pm 3 . 6 6$ </td><td>10.57</td><td>17.95</td><td>2.79–25.64</td></tr><tr><td>D36 D37</td><td>Thumb-index web-wrist-midpoint length</td><td> $T _ { 5 } \to \mathrm { M o W }$ </td><td> $6 . 3 5 \pm 4 . 3 2$ </td><td>5.95</td><td>14.88</td><td>0.01–20.12</td></tr><tr><td></td><td>Index MCP-wrist-midpoint length</td><td> $\mathrm { M T K } _ { I } \to \mathrm { M o W }$ </td><td> $4 . 1 7 \pm 2 . 8 8$ </td><td>3.68</td><td>9.55</td><td>0.02-12.88</td></tr><tr><td>D38</td><td></td><td></td><td> $4 . 5 6 \pm 3 . 6 8$ </td><td>3.72</td><td>11.75</td><td>0.01–20.35</td></tr><tr><td>D39</td><td>Index-middle web-wrist-midpoint length</td><td> $\mathrm { R O _ { 1 } } \to \mathrm { M o W }$ </td><td> $6 . 0 1 \pm 4 . 3 8$ </td><td>5.13</td><td>14.68</td><td>0.00-18.86</td></tr><tr><td>D40</td><td>Middle MCP-wrist-midpoint length</td><td> $\mathrm { M T K } _ { M } \to \mathrm { M o W }$ </td><td> $5 . 0 0 \pm 3 . 8 6$ </td><td>4.13</td><td>12.94</td><td>0.01-16.52</td></tr><tr><td>D41</td><td>Middle-ring web-wrist-midpoint length</td><td> $\mathrm { R O _ { 2 } } \to \mathrm { M o W }$ </td><td> $5 . 8 1 \pm 4 . 1 9$ </td><td>5.00</td><td>14.48</td><td>0.03-18.62</td></tr><tr><td>D42</td><td>Ring MCP-wrist-midpoint length</td><td> $\mathrm { M T K } _ { R } \to \mathrm { M o W }$ </td><td> $4 . 9 5 \pm 3 . 9 0$ </td><td>4.12</td><td>13.29</td><td>0.00-19.33</td></tr><tr><td>D43</td><td>Ring-little web-wrist-midpoint length</td><td> $\mathrm { R O _ { 3 } } \to \mathrm { M o W }$ </td><td> $6 . 0 9 \pm 4 . 2 9$ </td><td>5.50</td><td>14.50</td><td>0.00-19.15</td></tr><tr><td>D44</td><td>Little MCP-wrist-midpoint length</td><td> $\mathrm { M T K } _ { L } \to \mathrm { M o W }$ </td><td> $6 . 2 1 \pm 4 . 2 5$ </td><td>5.42</td><td>14.48</td><td>0.05–20.16</td></tr></table>

Table S5: Definitions and per-dimension absolute-error statistics for $E _ { 1 6 }$ on the 704 complete outputs. Endpoints use the notation in Tables S1–S2; all statistics are in millimeters, and the Range column reports minimum and maximum errors.

## K Per-Keypoint Pixel Error by Anatomical Group

Tables S6 and S7 compare pooled and regional pixel errors for $E _ { 1 5 } \left( \mathrm { A } 2 \right.$ , without PP) and $E _ { 1 6 } \ ( \mathbf { A } 3$ , with PP) on 45 manually annotated images, one per participant (1,845 keypoint pairs). Mean error decreased on 44 of the 45 images. Figure S10 shows representative A0–A3 outputs.

<table><tr><td>Metric</td><td> $E _ { 1 5 } \ ( \mathrm { A } 2 )$ </td><td> $E _ { 1 6 } \ ( \mathrm { A } 3 )$ </td><td>%∆</td></tr><tr><td>KP Mean (px)</td><td>11.18</td><td>7.83</td><td>-29.94%</td></tr><tr><td>KP Std (px)</td><td>8.73</td><td>6.74</td><td>-22.77%</td></tr><tr><td>Median error (px)</td><td>8.87</td><td>6.12</td><td>-30.98%</td></tr><tr><td>IQR (px)</td><td>9.13</td><td>6.26</td><td>-31.47%</td></tr><tr><td> $P ( { \mathrm { e r r o r } } \leq 5 { \mathrm { p x } } )$ </td><td>21.52%</td><td>39.02%</td><td>+81.3%</td></tr><tr><td> $P ( { \mathrm { e r r o r } } \leq 1 0 { \mathrm { p x } } )$ </td><td>57.56%</td><td>75.77%</td><td>+31.6%</td></tr></table>

Table S6: Pixel-level comparison of $E _ { 1 5 }$ (A2, no PP) and $E _ { 1 6 }$ (A3, with PP) on 45 annotated test images (1,845 keypoint pairs). $\% \Delta = ( E _ { 1 6 } - E _ { 1 5 } ) / E _ { 1 5 }$

<table><tr><td>Group</td><td> $E _ { 1 5 } \ ( \mathrm { A } 2 )$ </td><td> $E _ { 1 6 } \ ( \mathrm { A } 3 )$ </td><td>%∆</td></tr><tr><td>Index</td><td>8.57</td><td>5.59</td><td>-34.8%</td></tr><tr><td>Middle</td><td>8.70</td><td>6.05</td><td>-30.5%</td></tr><tr><td>Ring</td><td>10.81</td><td>6.90</td><td>-36.1%</td></tr><tr><td>Little</td><td>13.13</td><td>7.35</td><td>-44.0%</td></tr><tr><td>Thumb</td><td>17.35</td><td>14.54</td><td>-16.2%</td></tr><tr><td>Root</td><td>5.91</td><td>5.85</td><td>-1.0%</td></tr><tr><td>Palm</td><td>10.78</td><td>7.75</td><td>-28.2%</td></tr><tr><td>Wrist</td><td>13.28</td><td>9.83</td><td>-26.0%</td></tr><tr><td>All</td><td>11.18</td><td>7.83</td><td>-29.9%</td></tr></table>

Table S7: Mean keypoint pixel error (px) by anatomical group for $E _ { 1 5 }$ (A2) and $E _ { 1 6 } \ ( \mathrm { A } 3 )$ on 45 real test images.

## L Paired Millimeter-Space PP Ablation

Paired design. $E _ { 1 5 }$ (A2, no PP) and $E _ { 1 6 } \left( \mathbf { A } \boldsymbol { 3 } \right.$ , full PP; $\mathsf { A p - }$ pendix H) use identical inputs, PR, BW, model weights, raw predictions, midpoint construction, dimension definitions, and metric conversion. Both completed the same 704 of 720 captures from all 45 participants; the remaining 16 failed in both configurations.

Aggregation and uncertainty. Each dimension’s MAE averages absolute errors against the two-operator mean caliper reference across the 704 captures. Overall and group MAEs equally weight their member dimensions (44 overall; 19 finger lengths, 14 finger widths, 11 palm/wrist). Captures have equal weight within dimensions, so participants with more completed captures contribute more observations. Define $\Delta { \mathsf { \bar { \Delta } } } = \mathrm { \ M A { \bar { E } } _ { A 3 } - M A E _ { A 2 } }$ . To obtain paired participant-cluster bootstrap intervals, we resample 45 participants with replacement, retaining their common captures, dimensions, and both configurations together before recomputing MAE. We use 10,000 replicates, NumPy’s default\_rng with seed 20260908, and the 2.5th/97.5th percentiles. Per-dimension intervals are exploratory without

![](images/96b6ae2aa551b6b4a644ab5b50f6b11fdd6ceb7c54dd681dcf95522509d347a2.jpg)

![](images/588cbd9a90ddf18b7c3465b76e379c763581d1ac5b29fa840bdfb028214314eb.jpg)

![](images/aeaadb99070907ec3ff18edd9dd810625b3912e1f7d5b6a539b41ab984460971.jpg)  
(c) A2: +BW

![](images/bab129df1a6dd34ca0bd10a8e28e69a31dbbbcf7258b45f26c9b8c1338c2a64e.jpg)  
(d) A3: +PP  
Figure S10: Representative outputs across the progressive ablation configurations.

multiplicity adjustment. Tables S8 and S9 report all group and dimension results.
<table><tr><td>Group (n)</td><td>A2 A3</td><td>∆ [95% CI]</td><td></td></tr><tr><td>Overall (44)</td><td>3.707</td><td>3.804</td><td>+0.097 [0.002, 0.193]</td></tr><tr><td>Finger length (19)</td><td>3.196 3.097</td><td>-0.100</td><td>[-0.235, 0.038]</td></tr><tr><td>Finger width (14)</td><td>2.722 2.910</td><td>+0.188</td><td>[0.035, 0.340]</td></tr><tr><td>Palm/wrist (11)</td><td>5.842</td><td>6.165+0.323</td><td>[0.167,0.494]</td></tr></table>

Table S8: Paired PP ablation on 704 common completed captures from 45 participants. A2 and A3 columns report MAE; n counts dimensions. All values are in millimeters, and negative $\Delta$ indicates improvement. Confidence intervals use participant-cluster resampling.

<table><tr><td rowspan=1 colspan=11>Mean absolute error                  Signed bias      $P _ { 9 5 }$ absolute error</td></tr><tr><td rowspan=1 colspan=1>ID</td><td rowspan=1 colspan=10>A2    A3       $\Delta$    95% CI for $\Delta$          A2      A3    A2        A3</td></tr><tr><td rowspan=1 colspan=1>D1</td><td rowspan=1 colspan=4>2.709  3.843+1.134   [0.775, 1.504]</td><td rowspan=1 colspan=6>+2.700  +3.809  4.830     6.565</td></tr><tr><td rowspan=1 colspan=1>D2</td><td rowspan=1 colspan=1>8.450</td><td rowspan=1 colspan=1>6.509</td><td rowspan=1 colspan=1>-1.941 [-</td><td rowspan=1 colspan=1>2.389, −1.501]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>+8.426  +6.324</td><td rowspan=1 colspan=3>15.552    12.629</td></tr><tr><td rowspan=1 colspan=1>D3</td><td rowspan=1 colspan=1>3.280</td><td rowspan=1 colspan=1>5.691</td><td rowspan=1 colspan=1>+2.410</td><td rowspan=1 colspan=1>[1.967,2.878]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>-2.600  -5.540</td><td rowspan=1 colspan=3>7.641    10.866</td></tr><tr><td rowspan=1 colspan=1>D4</td><td rowspan=1 colspan=1>11.083</td><td rowspan=1 colspan=1>7.819</td><td rowspan=1 colspan=1>-3.264</td><td rowspan=1 colspan=1>[−4.055, -2.474]</td><td rowspan=1 colspan=2>+10.884</td><td rowspan=1 colspan=1>+7.318</td><td rowspan=1 colspan=3>18.193    15.655</td></tr><tr><td rowspan=1 colspan=1>D5</td><td rowspan=1 colspan=1>9.927</td><td rowspan=1 colspan=1>6.317</td><td rowspan=1 colspan=1>-3.610</td><td rowspan=1 colspan=1>[−4.720, −2.540]</td><td rowspan=1 colspan=2>+9.794</td><td rowspan=1 colspan=1>+3.336</td><td rowspan=1 colspan=3>17.745    13.486</td></tr><tr><td rowspan=1 colspan=1>D6</td><td rowspan=1 colspan=1>2.323</td><td rowspan=1 colspan=1>2.278</td><td rowspan=1 colspan=1>-0.045</td><td rowspan=1 colspan=1>[−0.310,0.207]</td><td rowspan=1 colspan=2>+2.313</td><td rowspan=1 colspan=1>+2.278</td><td rowspan=1 colspan=3>4.473     4.132</td></tr><tr><td rowspan=1 colspan=1>D7</td><td rowspan=1 colspan=1>2.304</td><td rowspan=1 colspan=1>2.527</td><td rowspan=1 colspan=1>+0.223</td><td rowspan=1 colspan=1>-0.015,0.468]</td><td rowspan=1 colspan=2>+2.289</td><td rowspan=1 colspan=1>+2.519</td><td rowspan=1 colspan=3>4.236     4.334</td></tr><tr><td rowspan=1 colspan=1>D8</td><td rowspan=1 colspan=1>3.758</td><td rowspan=1 colspan=1>3.585</td><td rowspan=1 colspan=1>-0.173</td><td rowspan=1 colspan=1>[−0.503, 0.158]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>+3.758</td><td rowspan=1 colspan=1>+3.557</td><td rowspan=1 colspan=3>6.220     6.387</td></tr><tr><td rowspan=1 colspan=1>D9</td><td rowspan=1 colspan=1>1.869</td><td rowspan=1 colspan=1>2.468</td><td rowspan=1 colspan=1>+0.599</td><td rowspan=1 colspan=1>[0.407,0.793]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-1.419</td><td rowspan=1 colspan=1>-2.191</td><td rowspan=1 colspan=3>4.312     5.314</td></tr><tr><td rowspan=1 colspan=1>D10</td><td rowspan=1 colspan=1>1.544</td><td rowspan=1 colspan=1>1.709</td><td rowspan=1 colspan=1>+0.166</td><td rowspan=1 colspan=1>[0.023, 0.301]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>+0.911</td><td rowspan=1 colspan=1>+0.655</td><td rowspan=1 colspan=3>3.571     3.786</td></tr><tr><td rowspan=1 colspan=1>D11</td><td rowspan=1 colspan=1>2.045</td><td rowspan=1 colspan=1>2.110</td><td rowspan=1 colspan=1>+0.064</td><td rowspan=1 colspan=1>[−0.152,0.291]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>+1.672</td><td rowspan=1 colspan=1>+1.512</td><td rowspan=1 colspan=3>4.693     5.070</td></tr><tr><td rowspan=1 colspan=1>D12</td><td rowspan=1 colspan=1>2.547</td><td rowspan=1 colspan=1>3.442</td><td rowspan=1 colspan=1>+0.895</td><td rowspan=1 colspan=1>[0.503, 1.275]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>+0.856</td><td rowspan=1 colspan=1>-0.250</td><td rowspan=1 colspan=3>5.872     7.138</td></tr><tr><td rowspan=1 colspan=1>D13</td><td rowspan=1 colspan=1>1.755</td><td rowspan=1 colspan=1>1.888</td><td rowspan=1 colspan=1>+0.133</td><td rowspan=1 colspan=1>[−0.095, 0.363]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>+1.732</td><td rowspan=1 colspan=1>+1.886</td><td rowspan=1 colspan=3>3.536     3.309</td></tr><tr><td rowspan=1 colspan=1>D14</td><td rowspan=1 colspan=1>1.492</td><td rowspan=1 colspan=1>2.199</td><td rowspan=1 colspan=1>+0.708</td><td rowspan=1 colspan=1>[0.458, 0.957]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>+1.446</td><td rowspan=1 colspan=1>+2.199</td><td rowspan=1 colspan=3>3.228     3.773</td></tr><tr><td rowspan=1 colspan=1>D15</td><td rowspan=1 colspan=1>2.180</td><td rowspan=1 colspan=1>2.358</td><td rowspan=1 colspan=1>+0.178</td><td rowspan=1 colspan=1>[−0.033, 0.384]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>+2.167</td><td rowspan=1 colspan=1>+2.193</td><td rowspan=1 colspan=3>4.264     4.965</td></tr><tr><td rowspan=1 colspan=1>D16</td><td rowspan=1 colspan=1>1.464</td><td rowspan=1 colspan=1>1.625</td><td rowspan=1 colspan=1>+0.160</td><td rowspan=1 colspan=1>-0.010,0.328]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.410</td><td rowspan=1 colspan=1>-1.040</td><td rowspan=1 colspan=3>3.604     3.581</td></tr><tr><td rowspan=1 colspan=1>D17</td><td rowspan=1 colspan=1>1.908</td><td rowspan=1 colspan=1>1.836</td><td rowspan=1 colspan=1>-0.072</td><td rowspan=1 colspan=1>-0.170,0.027]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>+1.445</td><td rowspan=1 colspan=1>+1.363</td><td rowspan=1 colspan=3>4.483     4.428</td></tr><tr><td rowspan=1 colspan=1>D18</td><td rowspan=1 colspan=1>2.437</td><td rowspan=1 colspan=1>2.702</td><td rowspan=1 colspan=1>+0.265</td><td rowspan=1 colspan=1>[0.120,0.403]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>+2.114</td><td rowspan=1 colspan=1>+2.327</td><td rowspan=1 colspan=1>5.704</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>5.905</td></tr><tr><td rowspan=1 colspan=1>D19</td><td rowspan=1 colspan=1>3.768</td><td rowspan=1 colspan=1>3.539</td><td rowspan=1 colspan=1>-0.230</td><td rowspan=1 colspan=1>[−0.542, 0.062]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>+3.198</td><td rowspan=1 colspan=1>+2.705</td><td rowspan=1 colspan=1>8.331</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>7.420</td></tr><tr><td rowspan=1 colspan=1>D20</td><td rowspan=1 colspan=1>1.917</td><td rowspan=1 colspan=1>1.933</td><td rowspan=1 colspan=1>+0.015</td><td rowspan=1 colspan=1>[−0.198,0.227]</td><td rowspan=1 colspan=2>+1.917</td><td rowspan=1 colspan=1>+1.933</td><td rowspan=1 colspan=1>3.563</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3.094</td></tr><tr><td rowspan=1 colspan=1>D21</td><td rowspan=1 colspan=1>1.497</td><td rowspan=1 colspan=1>2.107</td><td rowspan=1 colspan=1>+0.610</td><td rowspan=1 colspan=1>[0.391, 0.823]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>+1.467</td><td rowspan=1 colspan=1>+2.086</td><td rowspan=1 colspan=1>3.069</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3.747</td></tr><tr><td rowspan=1 colspan=1>D22</td><td rowspan=1 colspan=1>3.004</td><td rowspan=1 colspan=1>3.465</td><td rowspan=1 colspan=1>+0.461</td><td rowspan=1 colspan=1>[0.152,0.747]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>+2.842</td><td rowspan=1 colspan=1>+3.313</td><td rowspan=1 colspan=1>5.899</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>6.634</td></tr><tr><td rowspan=1 colspan=1>D23</td><td rowspan=1 colspan=1>1.784</td><td rowspan=1 colspan=1>1.865</td><td rowspan=1 colspan=1>+0.081</td><td rowspan=1 colspan=1>[−0.098,0.276]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.255</td><td rowspan=1 colspan=1>-0.671</td><td rowspan=1 colspan=1>4.617</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>4.360</td></tr><tr><td rowspan=1 colspan=1>D24</td><td rowspan=1 colspan=1>1.807</td><td rowspan=1 colspan=1>1.943</td><td rowspan=1 colspan=1>+0.136</td><td rowspan=1 colspan=1>[−0.022, 0.295]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>+1.056</td><td rowspan=1 colspan=1>+1.271</td><td rowspan=1 colspan=1>4.138</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>4.357</td></tr><tr><td rowspan=1 colspan=1>D25</td><td rowspan=1 colspan=1>3.140</td><td rowspan=1 colspan=1>3.662</td><td rowspan=1 colspan=1>+0.522</td><td rowspan=1 colspan=1>[0.292, 0.747]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>+2.918</td><td rowspan=1 colspan=1>+3.550</td><td rowspan=1 colspan=1>5.949</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>6.329</td></tr><tr><td rowspan=1 colspan=1>D26</td><td rowspan=1 colspan=1>3.779</td><td rowspan=1 colspan=1>4.141</td><td rowspan=1 colspan=1>+0.362</td><td rowspan=1 colspan=1>[−0.157,0.871]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>+3.527</td><td rowspan=1 colspan=1>+3.957</td><td rowspan=1 colspan=1>7.638</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>8.566</td></tr><tr><td rowspan=1 colspan=1>D27</td><td rowspan=1 colspan=1>2.078</td><td rowspan=1 colspan=1>1.870</td><td rowspan=1 colspan=1>-0.208</td><td rowspan=1 colspan=1>-0.418,0.005]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>+2.078</td><td rowspan=1 colspan=1>+1.862</td><td rowspan=1 colspan=1>3.928</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3.207</td></tr><tr><td rowspan=1 colspan=1>D28</td><td rowspan=1 colspan=1>1.781</td><td rowspan=1 colspan=1>1.858</td><td rowspan=1 colspan=1>+0.078</td><td rowspan=1 colspan=1>[−0.156, 0.309]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>+1.722</td><td rowspan=1 colspan=1>+1.816</td><td rowspan=1 colspan=1>3.332</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3.629</td></tr><tr><td rowspan=1 colspan=1>D29</td><td rowspan=1 colspan=1>2.864</td><td rowspan=1 colspan=1>4.318</td><td rowspan=1 colspan=1>+1.454</td><td rowspan=1 colspan=1>[1.091, 1.830]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>+2.851</td><td rowspan=1 colspan=1>+4.286</td><td rowspan=1 colspan=1>5.075</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>8.186</td></tr><tr><td rowspan=1 colspan=1>D30</td><td rowspan=1 colspan=1>1.411</td><td rowspan=1 colspan=1>1.491</td><td rowspan=1 colspan=1>+0.080</td><td rowspan=1 colspan=1>[−0.198, 0.378]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.156</td><td rowspan=1 colspan=1>-0.527</td><td rowspan=1 colspan=1>3.775</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3.804</td></tr><tr><td rowspan=1 colspan=1>D31</td><td rowspan=1 colspan=1>1.690</td><td rowspan=1 colspan=1>1.598</td><td rowspan=1 colspan=1>-0.092</td><td rowspan=1 colspan=1>-0.303,0.124]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>+1.317</td><td rowspan=1 colspan=1>+1.322</td><td rowspan=1 colspan=1>3.935</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3.633</td></tr><tr><td rowspan=1 colspan=1>D32</td><td rowspan=1 colspan=1>2.246</td><td rowspan=1 colspan=1>2.314</td><td rowspan=1 colspan=1>+0.067</td><td rowspan=1 colspan=1>-0.251,0.391]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>+1.506</td><td rowspan=1 colspan=1>+1.765</td><td rowspan=1 colspan=1>4.408</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>5.053</td></tr><tr><td rowspan=1 colspan=1>D33</td><td rowspan=1 colspan=1>3.001</td><td rowspan=1 colspan=1>2.562</td><td rowspan=1 colspan=1>-0.439  [</td><td rowspan=1 colspan=1>-1.147,0.287]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>+2.012</td><td rowspan=1 colspan=1>+2.067</td><td rowspan=1 colspan=1>7.080</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>6.592</td></tr><tr><td rowspan=1 colspan=1>D34</td><td rowspan=1 colspan=1>9.579</td><td rowspan=1 colspan=1>7.672</td><td rowspan=1 colspan=1>-1.907 [−</td><td rowspan=1 colspan=1>2.535, −1.248</td><td rowspan=1 colspan=1>]</td><td rowspan=1 colspan=1>+9.575</td><td rowspan=1 colspan=1>+7.672</td><td rowspan=1 colspan=1>14.912</td><td rowspan=1 colspan=2>11.412</td></tr><tr><td rowspan=1 colspan=1>D35</td><td rowspan=1 colspan=1>8.267</td><td rowspan=1 colspan=1>10.996</td><td rowspan=1 colspan=1>+2.730</td><td rowspan=1 colspan=1>[1.820, 3.687]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>+8.264</td><td rowspan=1 colspan=1>+10.996</td><td rowspan=1 colspan=1>12.708</td><td rowspan=1 colspan=2>17.947</td></tr><tr><td rowspan=1 colspan=1>D36</td><td rowspan=1 colspan=1>5.387</td><td rowspan=1 colspan=1>6.353</td><td rowspan=1 colspan=1>+0.966</td><td rowspan=1 colspan=1>[0.613, 1.328]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-4.250</td><td rowspan=1 colspan=1>-5.412</td><td rowspan=1 colspan=1>12.416</td><td rowspan=1 colspan=2>14.878</td></tr><tr><td rowspan=1 colspan=1>D37</td><td rowspan=1 colspan=1>4.010</td><td rowspan=1 colspan=1>4.174</td><td rowspan=1 colspan=1>+0.164</td><td rowspan=1 colspan=1>-0.469,0.811]</td><td rowspan=1 colspan=2>+2.121</td><td rowspan=1 colspan=1>-0.394</td><td rowspan=1 colspan=1>9.592</td><td rowspan=1 colspan=2>9.548</td></tr><tr><td rowspan=1 colspan=1>D38</td><td rowspan=1 colspan=1>4.626</td><td rowspan=1 colspan=1>4.556</td><td rowspan=1 colspan=1>-0.070</td><td rowspan=1 colspan=1>-0.166,0.024]</td><td rowspan=1 colspan=2>+3.687</td><td rowspan=1 colspan=1>+3.519</td><td rowspan=1 colspan=1>12.172</td><td rowspan=1 colspan=2>11.746</td></tr><tr><td rowspan=1 colspan=1>D39</td><td rowspan=1 colspan=1>5.956</td><td rowspan=1 colspan=1>6.007</td><td rowspan=1 colspan=1>+0.051</td><td rowspan=1 colspan=1>-0.061,0.162]</td><td rowspan=1 colspan=2>+5.054</td><td rowspan=1 colspan=1>+5.077</td><td rowspan=1 colspan=1>14.143</td><td rowspan=1 colspan=2>14.680</td></tr><tr><td rowspan=1 colspan=1>D40</td><td rowspan=1 colspan=1>5.070</td><td rowspan=1 colspan=1>4.999</td><td rowspan=1 colspan=1>-0.071 [−</td><td rowspan=1 colspan=1>0.109, −0.033</td><td rowspan=1 colspan=2>]  +4.233</td><td rowspan=1 colspan=1>+4.182</td><td rowspan=1 colspan=1>13.055</td><td rowspan=1 colspan=2>12.942</td></tr><tr><td rowspan=1 colspan=1>D41</td><td rowspan=1 colspan=2>5.575 5.810</td><td rowspan=1 colspan=1>+0.236</td><td rowspan=1 colspan=1>[0.126,0.349]</td><td rowspan=1 colspan=2>+4.873</td><td rowspan=1 colspan=1>+5.303</td><td rowspan=1 colspan=1>14.199</td><td rowspan=1 colspan=2>14.479</td></tr><tr><td rowspan=1 colspan=1>D42</td><td rowspan=1 colspan=2>4.931 4.952</td><td rowspan=1 colspan=1>+0.021</td><td rowspan=1 colspan=1>[−0.071,0.117]</td><td rowspan=1 colspan=2>+4.060</td><td rowspan=1 colspan=1>+4.173</td><td rowspan=1 colspan=1>13.407</td><td rowspan=1 colspan=2>13.288</td></tr><tr><td rowspan=1 colspan=3>D43 5.358  6.087</td><td rowspan=1 colspan=1>+0.729</td><td rowspan=1 colspan=1>[0.496,0.963]</td><td rowspan=1 colspan=2>+4.445</td><td rowspan=1 colspan=1>+5.339</td><td rowspan=1 colspan=1>13.464</td><td rowspan=1 colspan=2>14.501</td></tr><tr><td rowspan=1 colspan=4>D44 5.505  6.210 +0.705</td><td rowspan=1 colspan=1>[0.496, 0.913]</td><td rowspan=1 colspan=3>+5.036  +5.956</td><td rowspan=1 colspan=3>13.247    14.478</td></tr></table>

Table S9: Complete dimension-level PP comparison on the same 704 captures. IDs follow Table S5, and all values are in millimeters. $\Delta \mathrm { \bar { \Delta } = M A E _ { A 3 } - M A E _ { A 2 } ; }$ negative values indicate improvement. The paired participant-bootstrap intervals are exploratory, without multiplicity adjustment. Signed bias is the mean signed error (prediction minus caliper reference); $P _ { 9 5 }$ is the per-dimension 95th percentile of per-capture absolute errors, using linear interpolation.

## M Full Ablation Configurations

Table S10 reports training and validation pose loss for all 16 configurations.
<table><tr><td>ID</td><td>YOLO Train aug.</td><td>Case</td><td>Pose (Train)</td><td>Pose (Val)</td></tr><tr><td></td><td>Without training data augmentation</td><td></td><td></td><td></td></tr><tr><td>E1</td><td>v8</td><td>no A0</td><td>7.29355</td><td>3.42462</td></tr><tr><td>E2</td><td>v8</td><td>no A1</td><td>0.94881</td><td>0.78883</td></tr><tr><td>E3</td><td>v8</td><td>no A2</td><td>0.93568</td><td>0.83675</td></tr><tr><td>E4</td><td>v8</td><td>no A3</td><td></td><td></td></tr><tr><td>E5</td><td>v11</td><td>no A0</td><td>9.02002</td><td>8.49234</td></tr><tr><td>E6</td><td>v11</td><td>no A1</td><td>1.55365</td><td>0.86025</td></tr><tr><td>E7</td><td>v11</td><td>no A2</td><td>0.85474</td><td>0.79316</td></tr><tr><td>E8</td><td>v11</td><td>no A3</td><td></td><td></td></tr><tr><td></td><td colspan="4">With training data augmentation</td></tr><tr><td>E9</td><td>v8</td><td>yes A0</td><td>8.39867</td><td>10.78510</td></tr><tr><td>E10</td><td>v8</td><td>yes A1</td><td>0.65540</td><td>0.62843</td></tr><tr><td>E11</td><td>v8</td><td>A2</td><td>0.49995</td><td>0.63917</td></tr><tr><td>E12</td><td>v8</td><td>yes A3</td><td></td><td></td></tr><tr><td>E13</td><td></td><td>yes</td><td>8.35649</td><td>6.40620</td></tr><tr><td></td><td>v11</td><td>yes A0</td><td></td><td></td></tr><tr><td>E14</td><td>v11</td><td>yes A1</td><td>0.51637</td><td>0.70538</td></tr><tr><td>E15</td><td>v11</td><td>yes A2</td><td>0.64615</td><td>0.63816</td></tr><tr><td>E16</td><td>v11</td><td>yes A3</td><td></td><td></td></tr></table>

Note: A3 shares the A2 network and therefore has identical pose loss (“–”).  
Table S10: Training and validation pose loss for all 16 configurations.

## N Capture-Condition Sensitivity and Within-Person Variability

Table S11 reports descriptive MAEs and image counts by capture factor; these are not equivalence tests. Nominal illumination settings were pooled because captured brightness difered little.
<table><tr><td>Factor</td><td>A/B</td><td> $N _ { A } /$   $N _ { B }$ </td><td> $\mathrm { M A E } _ { A } /$   $\mathbf { M A E } _ { B }$ </td><td> $\Delta$ </td></tr><tr><td>Device</td><td>Android / iPhone</td><td></td><td>350 / 3543.83 / 3.78+0.05</td><td></td></tr><tr><td></td><td>Background Complex / Simple</td><td></td><td>354 / 3503.80 / 3.80-0.00</td><td></td></tr><tr><td>Angle</td><td>Top-down / Oblique</td><td></td><td>346 / 3584.18 / 3.44+0.74</td><td></td></tr></table>

Table S11: Descriptive accuracy by capture factor among complete outputs. $N _ { A }$ and $N _ { B }$ count images, not independent participants; $\Delta \mathbf { M A E } { = } \mathbf { M A E } _ { A } { - } \mathbf { M A E } _ { B }$ , and all MAE values are in millimeters.

## N.1 Within-Person Capture Variability

Among 45 participants, 38 contributed 16 completed captures and seven contributed 12–15. Within each participant and dimension, we computed the SD and coeficient of variation (CV; SD/mean prediction, expressed as a percentage). Across the 1,980 participant–dimension combinations, median $\prime P _ { 9 5 }$ SD was 1.25/3.99 mm and median $P _ { 9 5 }$ CV was 3.41/8.12%. These repeated summaries are conditional on completion and describe capture variability, not caliper accuracy.

For each participant and dimension, we also computed the absolute diference between mean predictions at the two levels of each factor, then averaged across participants and equally across dimensions: 2.36 mm for angle, 0.46 mm for smartphone, and 0.88 mm for background. Table S12 gives per-dimension results; Appendix O gives caliper-referenced angle contrasts.

<table><tr><td colspan="2"></td><td colspan="2">Within-person CV (%)</td><td colspan="3">Mean absolute condition-mean difference (mm)</td></tr><tr><td>ID</td><td>Measurement</td><td>Median</td><td> $P _ { 9 5 }$ </td><td>Angle</td><td>Smartphone</td><td>Background</td></tr><tr><td>D1</td><td>Thumb IP breadth</td><td>3.16</td><td>7.15</td><td>1.02</td><td>0.37</td><td>0.62</td></tr><tr><td>D2</td><td>Thumb MCP breadth</td><td>4.39</td><td>8.42</td><td>1.88</td><td>0.84</td><td>1.23</td></tr><tr><td>D3</td><td>Thumb distal-segment length</td><td>9.43</td><td>12.36</td><td>3.87</td><td>0.76</td><td>1.42</td></tr><tr><td>D4</td><td>Thumb proximal-segment length</td><td>7.14</td><td>9.80</td><td>3.77</td><td>0.85</td><td>1.52</td></tr><tr><td>D5</td><td>Thumb total length</td><td>7.93</td><td>10.17</td><td>7.73</td><td>1.28</td><td>2.24</td></tr><tr><td>D6</td><td>Index DIP breadth</td><td>3.91</td><td>5.42</td><td>1.06</td><td>0.31</td><td>0.35</td></tr><tr><td>D7</td><td>Index PIP breadth</td><td>3.09</td><td>4.51</td><td>0.81</td><td>0.38</td><td>0.42</td></tr><tr><td>D8</td><td>Index MCP breadth</td><td>4.21</td><td>6.27</td><td>0.89</td><td>0.50</td><td>0.59</td></tr><tr><td>D9</td><td>Index distal-segment length</td><td>5.50</td><td>7.85</td><td>2.27</td><td>0.30</td><td>0.54</td></tr><tr><td>D10</td><td>Index middle-segment length</td><td>6.54</td><td>8.32</td><td>2.38</td><td>0.26</td><td>0.50</td></tr><tr><td>D11</td><td>Index proximal-segment length</td><td>4.70</td><td>5.90</td><td>1.94</td><td>0.30</td><td>0.56</td></tr><tr><td>D12</td><td>Index total length</td><td>5.27</td><td>6.61</td><td>6.50</td><td>0.66</td><td>1.08</td></tr><tr><td>D13</td><td>Middle DIP breadth</td><td>3.11</td><td>4.72</td><td>0.84</td><td>0.23</td><td>0.26</td></tr><tr><td>D14</td><td>Middle PIP breadth</td><td>3.13</td><td>4.78</td><td>0.95</td><td>0.27</td><td>0.32</td></tr><tr><td>D15</td><td>Middle MCP breadth</td><td>5.21</td><td>8.67</td><td>1.17</td><td>0.44</td><td>0.55</td></tr><tr><td>D16</td><td>Middle distal-segment length</td><td>3.66</td><td>5.27</td><td>1.37</td><td>0.30</td><td>0.53</td></tr><tr><td>D17</td><td>Middle middle-segment length</td><td>5.18</td><td>7.43</td><td>2.27</td><td>0.28</td><td>0.53</td></tr><tr><td>D18</td><td>Middle proximal-segment length</td><td>3.13</td><td>4.75</td><td>1.19</td><td>0.27</td><td>0.58</td></tr><tr><td>D19</td><td>Middle total length</td><td>3.55</td><td>4.59</td><td>4.81</td><td>0.59</td><td>0.90</td></tr><tr><td>D20</td><td>Ring DIP breadth</td><td>3.01</td><td>5.43</td><td>0.74</td><td>0.18</td><td>0.26</td></tr><tr><td>D21</td><td>Ring PIP breadth</td><td>2.85</td><td>5.52</td><td>0.78</td><td>0.26</td><td>0.27</td></tr><tr><td>D22</td><td>Ring MCP breadth</td><td>4.58</td><td>8.14</td><td>0.78</td><td>0.43</td><td>0.70</td></tr><tr><td>D23</td><td>Ring distal-segment length</td><td>3.49</td><td>6.47</td><td>1.18</td><td>0.25</td><td>0.77</td></tr><tr><td>D24</td><td>Ring middle-segment length</td><td>4.57</td><td>7.67</td><td>1.75</td><td>0.30</td><td>0.69</td></tr><tr><td>D25</td><td>Ring proximal-segment length</td><td>2.62</td><td>5.33</td><td>0.65</td><td>0.26</td><td>0.67</td></tr><tr><td>D26</td><td>Ring total length</td><td>2.61</td><td>3.93</td><td>3.41</td><td>0.58</td><td>0.87</td></tr><tr><td>D27</td><td>Little DIP breadth</td><td>3.10</td><td>5.61</td><td>0.62</td><td>0.15</td><td>0.17</td></tr><tr><td>D28</td><td>Little PIP breadth</td><td>3.15</td><td>8.14</td><td>0.72</td><td>0.26</td><td>0.31</td></tr><tr><td>D29</td><td>Little MCP breadth</td><td>5.96</td><td>8.73</td><td>1.37</td><td>0.62</td><td>0.79</td></tr><tr><td>D30</td><td>Little distal-segment length</td><td>3.24</td><td>6.93</td><td>0.89</td><td>0.25</td><td>0.73</td></tr><tr><td>D31</td><td>Little middle-segment length</td><td>3.00</td><td>5.36</td><td>0.44</td><td>0.20</td><td>0.46</td></tr><tr><td>D32</td><td>Little proximal-segment length</td><td>4.83</td><td>7.49</td><td>0.88</td><td>0.35</td><td>0.85</td></tr><tr><td>D33</td><td>Little total length</td><td>2.49</td><td>4.45</td><td>1.96</td><td>0.60</td><td>1.13</td></tr><tr><td>D34</td><td>Metacarpal hand breadth</td><td>1.48</td><td>2.51</td><td>1.57</td><td>0.46</td><td>0.66</td></tr><tr><td>D35</td><td>Wrist breadth</td><td>2.97</td><td>5.00</td><td>2.73</td><td>0.57</td><td>1.68</td></tr><tr><td>D36</td><td>Thumb root-wrist-midpoint length</td><td>2.35</td><td>4.88</td><td>1.70</td><td>0.67</td><td>1.30</td></tr><tr><td>D37</td><td>Thumb-index web-wrist-midpoint length</td><td>2.01</td><td>4.32</td><td>1.59</td><td>0.64</td><td>1.35</td></tr><tr><td>D38</td><td>Index MCP-wrist-midpoint length</td><td>2.40</td><td>3.77</td><td>4.68</td><td>0.58</td><td>1.40</td></tr><tr><td>D39</td><td>Index-middle web-wrist-midpoint length</td><td>2.82</td><td>4.04</td><td>5.61</td><td>0.61</td><td>1.48</td></tr><tr><td>D40</td><td>Middle MCP-wrist-midpoint length</td><td>2.28</td><td>3.87</td><td>4.66</td><td>0.54</td><td>1.41</td></tr><tr><td>D41</td><td>Middle-ring web-wrist-midpoint length</td><td>2.31</td><td>3.79</td><td>4.66</td><td>0.59</td><td>1.46</td></tr><tr><td>D42</td><td>Ring MCP-wrist-midpoint length</td><td>2.50</td><td>3.96</td><td>4.93</td><td>0.50</td><td>1.47</td></tr><tr><td>D43</td><td>Ring-little web-wrist-midpoint length</td><td>2.41</td><td>4.84</td><td>4.64</td><td>0.56</td><td>1.51</td></tr><tr><td>D44</td><td>Little MCP-wrist-midpoint length</td><td>2.34</td><td>4.27</td><td>4.27</td><td>0.53</td><td>1.54</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table S12: Within-person capture variability across 44 dimensions. For each dimension, median and $P _ { 9 5 }$ summarize the CVs of 45 participants; each condition-mean diference is the average across participants of the absolute diference between their two condition-specific mean predictions. Summaries use the 704 completed captures. Dimension IDs and endpoints follow Table S5.

## O Dimension-Specific Capture-Angle Patterns

For the 704 complete outputs (346 top-down; 358 oblique), let $\Delta _ { d } = \mathrm { M A E } _ { d , \mathrm { t o p - d o w n } } ^ { - } - \mathrm { M A E } _ { d , \mathrm { o b l i q u e } } ^ { - }$ . The mean across dimensions was +0.74 mm; 26 of 44 contrasts were positive and 18 were negative. Table S13 summarizes the largest and recurrent patterns.

Bland-Altman: Pooled % Differences (N=1980)
<table><tr><td>Pattern</td><td>Dimensions</td><td> $\Delta _ { d }$  (mm)</td></tr><tr><td rowspan="5">Top-down worse</td><td>Wrist-referenced dimensions</td><td>+2.92 to</td></tr><tr><td>(D38-D44)</td><td>+4.28</td></tr><tr><td>Middle/ring total lengths (D19, D26)</td><td>+3.19, +2.97</td></tr><tr><td>Thumb proximal/total lengths (D4,</td><td>+2.89,</td></tr><tr><td>D5)</td><td>+2.22</td></tr><tr><td rowspan="4">Oblique worse</td><td>Thumb distal length (D3)</td><td>-3.74</td></tr><tr><td>Index/middle/ring DIP and PIP</td><td>-1.09 to</td></tr><tr><td>breadths (D6, D7, D13, D14, D20, D21)</td><td>-0.58</td></tr><tr><td>12 of 44 dimensions</td><td> $| \Delta _ { d } | \leq$ </td></tr></table>

<sup>†</sup>At the two-decimal precision shown in the current analysis; this descriptive cutof is not an equivalence margin.

Table S13: Largest and recurrent dimension-specific captureangle contrasts. Positive $\Delta _ { d }$ indicates higher top-down MAE; all contrasts are descriptive. Dimension definitions appear in Table S5.

## P Runtime Analysis

Table S14 reports $E _ { 1 6 } / \mathrm { A } 3$ runtime on an Apple M3 (8-core CPU: 4 performance and 4 eficiency cores; 16 successes from 20 inputs) and an NVIDIA RTX 4000 Ada Generation GPU (20 GB GDDR6; 704 complete test outputs). The workloads difer and do not constitute a matched hardware comparison. The BW runtime includes only cached-mask composition, excluding mask I/O.

<table><tr><td>Stage</td><td>CPU (M3)</td><td>GPU (RTX 4000 Ada)</td></tr><tr><td>Total execution time</td><td> $2 6 . 3 6 \pm 5 . 5 9$ </td><td> $3 . 8 6 \pm 0 . 6 1$ </td></tr><tr><td>Perspective rectification</td><td> $2 4 . 9 8 \pm 5 . 2 5$ </td><td> $2 . 3 0 \pm 0 . 3 8$ </td></tr><tr><td>Background whitening</td><td> $0 . 0 2 \pm 0 . 0 1$ </td><td> $0 . 0 2 \pm 0 . 0 1$ </td></tr><tr><td>Inference and PP</td><td> $1 . 1 6 \pm 0 . 3 7$ </td><td> $1 . 5 1 \pm 0 . 3 1$ </td></tr></table>

Table S14: Per-image stage runtime (mean ± SD, seconds). CPU: Apple M3, 16 successful images from a 20-image batch; GPU: RTX 4000 Ada, 704 complete outputs. BW reports mask-to-white composition only; cached-mask I/O is excluded.

## Q Annotation-to-Caliper Discrepancy Analysis

On one manually annotated rectified image from each of the 45 test participants, we derive all 44 dimensions using

HandAnthro’s midpoint construction and metric conversion. Let $y _ { i , d } ^ { \mathrm { a n n } }$ be the annotation-derived dimension and $\bar { y } _ { i , d } ^ { \mathrm { c a l } }$ the two-operator mean caliper reference. The pooled discrepancy is

$$
\mathrm { M A E _ { a n n o t a t i o n } } = \frac { 1 } { 4 5 \times 4 4 } \sum _ { i = 1 } ^ { 4 5 } \sum _ { d = 1 } ^ { 4 4 } \left| y _ { i , d } ^ { \mathrm { a n n } } - \bar { y } _ { i , d } ^ { \mathrm { c a l } } \right| .\tag{7}
$$

Across 1,980 errors, the mean was 3.38 mm and the sample SD was 3.50 mm. This diagnostic combines endpointdefinition, projection, annotation, rectification, and caliper discrepancies without isolating their contributions. Manual image annotations are not error-free, so this result is not an irreducible lower bound on automated measurement error.

## R Inter-Operator Reliability of Manual Caliper Measurements

An operator estimated that measuring all 44 dimensions of one hand would take approximately 15 minutes. Two trained operators independently measured each of the 45 hands once for all 44 dimensions, yielding $N = 1 { , } 9 8 0$ paired observations. For readings $x _ { i j } ^ { \mathrm { ( i ) } } , x _ { i j } ^ { \mathrm { ( 2 ) } }$ , define the pair mean $\begin{array} { l l } { { m _ { i j } } } & { { = } } \end{array}$ $( x _ { i j } ^ { ( 1 ) } + x _ { i j } ^ { ( 2 ) } ) / 2$ and signed diference $d _ { i j } = x _ { i j } ^ { ( 1 ) } - x _ { i j } ^ { ( 2 ) }$ . The absolute diference is $| d _ { i j }$ | and the absolute percent diference (APD) is $1 0 0 | d _ { i j } | / m _ { i j } .$ The Bland–Altman plot (Fig. S11) uses signed relative diferences $r _ { i j } = 1 0 0 d _ { i j } / m _ { i j }$ against $m _ { i j }$ , with pooled mean bias r¯ and 95% limits of agreement $\bar { r } \pm 1 . 9 6 s _ { r } ,$ , where s is their sample SD.

Mean relative bias was 0.44%, with 95% limits $[ - 9 . 8 0 \% , + 1 0 . 6 9 \% ]$ . Median absolute diference was 0.90 mm (IQR 1.50 mm; $P _ { 9 5 } = 5 . 2 1 $ mm); median APD was 2.86% (IQR 4.18%; $P _ { 9 5 } = 1 1 . 0 0 \% )$

![](images/25956cddcd3d215932f1a88b0d0e852e9f9852e6356e96fe469c9997e9dfa336.jpg)  
Figure S11: Bland–Altman plot of signed relative diferences between the two trained operators $( N = 1 , 9 8 0$ across 45 hands and 44 dimensions). Horizontal lines indicate the pooled mean bias and 95% limits of agreement.

## S Unpaired Population-Level Comparison with the National Firefighter Reference

We compare sex-specific field-cohort summaries with Hsiao et al.’s independent national firefighter reference (Hsiao et al. 2015). This unpaired comparison evaluates group-level plausibility, not individual accuracy or agreement.

Data and review. The field cohort included 268 firefighters with one retained dominant-hand palmar image and a demographic record linked a priori (204 male, 64 female). Unlinked captures were excluded before processing, independently of algorithm outcome. Staf assisted with placement, app operation, and photography; unsuitable images were retaken, but attempt counts were not recorded. Retrospective desktop processing returned all 44 dimensions for 260 cases (97.0%; 197 male, 63 female). Visual review of landmark overlays did not change coordinates, trigger recomputation, or exclude completed outputs. Hsiao et al.’s Table 2 reports means and SDs for 14 dimensions from 855 men and 88 women, measured on right-hand 2D flatbed scans with fingers open.

Dimension mapping. Table S15 maps the reference dimensions to HandAnthro. Hand length sums palm and middle-finger lengths; non-thumb breadths use PIP landmarks. Metacarpal hand breadth is an explicit proxy for palm breadth because the pipeline has no diagonal palm-crease breadth.

<table><tr><td>Reference dimension</td><td>HandAnthro</td><td>Note</td></tr><tr><td>Hand length</td><td>D40+D19</td><td>palm length + middle- finger length</td></tr><tr><td>Hand breadth</td><td>D34</td><td>across the metacarpals</td></tr><tr><td>Palm length</td><td>D40</td><td>wrist-midpoint to middle MCP</td></tr><tr><td>Palm breadth</td><td>D34</td><td>proxy (no palm-crease breadth)</td></tr><tr><td>Thumb length</td><td>D5</td><td>tip to thumb root</td></tr><tr><td>Thumb breadth</td><td>D1</td><td>IP-joint breadth</td></tr><tr><td>Index length</td><td>D12</td><td>tip to MCP base</td></tr><tr><td>Index breadth</td><td>D7</td><td>PIP-joint breadth</td></tr><tr><td>Middle length</td><td>D19</td><td>tip to MCP base</td></tr><tr><td>Middle breadth</td><td>D14</td><td>PIP-joint breadth</td></tr><tr><td>Ring length</td><td>D26</td><td>tip to MCP base</td></tr><tr><td>Ring breadth</td><td>D21</td><td>PIP-joint breadth</td></tr><tr><td>Pinky length</td><td>D33</td><td>tip to MCP base</td></tr><tr><td>Pinky breadth</td><td>D28</td><td>PIP-joint breadth</td></tr></table>

Table S15: Mapping from Hsiao (2015) Table 2 dimensions to HandAnthro outputs.

Method. For each dimension and sex, Table S16 reports mean (SD), signed mean diferences in millimeters and percent (HandAnthro minus reference), Hedges’ g, and Welch two-sample test significance after Benjamini–Hochberg correction across all 28 contrasts. The descriptive count within 5% uses unrounded relative diferences.

Across 28 contrasts, the mean absolute diference was 2.40 mm (median 1.77 mm), and 20/28 were within 5% of the reference. Mean absolute Hedges’ g was 0.56; 18/28 contrasts remained significant after correction. The 5% cutof is descriptive, not an equivalence margin.

(a) Men
<table><tr><td>Dimension</td><td>Ours</td><td>Ref</td><td>∆ ∆%</td><td></td><td>g FDR</td></tr><tr><td>Hand length</td><td></td><td>204.9 (10.6) 197.6 (9.3) +7.33</td><td></td><td>+3.7 +0.77 yes</td><td></td></tr><tr><td>Hand breadth</td><td></td><td>100.8 (6.0) 97.2 (4.6) +3.58</td><td></td><td>+3.7 +0.73 yes</td><td></td></tr><tr><td>Palm length</td><td></td><td>121.5 (6.0) 113.8 (5.8) +7.75</td><td></td><td>+6.8 +1.33 yes</td><td></td></tr><tr><td>Palm breadth</td><td></td><td>100.8 (6.0) 96.0 (4.6) +4.78</td><td></td><td>+5.0 +0.98 yes</td><td></td></tr><tr><td>Thumb length</td><td></td><td>76.0 (5.8) 70.8 (4.3) +5.18 +7.3 +1.12 yes</td><td></td><td></td><td></td></tr><tr><td>Thumb breadth</td><td></td><td>27.7 (7.4) 24.4 (1.6) +3.31 +13.6 +0.94 yes</td><td></td><td></td><td></td></tr><tr><td>Index length</td><td>75.8 (5.6)</td><td>75.8 (4.4) +0.02 +0.0 +0.01 no</td><td></td><td></td><td></td></tr><tr><td>Index breadth</td><td>24.3 (1.8)</td><td>22.7 (1.6) +1.63</td><td></td><td>+7.2 +0.99 yes</td><td></td></tr><tr><td>Middle length</td><td>83.4 (5.8)</td><td>83.8 (4.6) −0.42</td><td></td><td>-0.5 -0.09 no</td><td></td></tr><tr><td>Middle breadth</td><td>24.2 (1.9)</td><td>22.4 (1.7) +1.82</td><td></td><td>+8.1 +1.05 yes</td><td></td></tr><tr><td>Ring length</td><td>77.0 (5.9)</td><td>79.6 (4.5) –2.60</td><td></td><td>-3.3-0.54 yes</td><td></td></tr><tr><td>Ring breadth</td><td>22.0 (1.6)</td><td>21.7 (1.6) +0.30</td><td></td><td>+1.4 +0.19 yes</td><td></td></tr><tr><td>Pinky length</td><td>61.7 (5.6)</td><td>65.2 (4.3) -3.49</td><td></td><td>-5.4 −0.76 yes</td><td></td></tr><tr><td>Pinky breadth</td><td>20.6 (1.7)</td><td>19.8 (1.5) +0.79</td><td></td><td>+4.0 +0.51 yes</td><td></td></tr></table>

(b) Women
<table><tr><td>Dimension</td><td>Ours</td><td>Ref ∆ ∆% gFDR</td></tr><tr><td>Hand length</td><td>184.4 (8.9) 182.7 (8.7) +1.71 +0.9 +0.19 no</td><td></td></tr><tr><td>Hand breadth</td><td></td><td>88.1 (5.4) 87.4 (4.2) +0.72 +0.8 +0.15 no</td></tr><tr><td>Palm length</td><td></td><td>108.0 (5.3) 104.0 (5.7) +4.03 +3.9 +0.72yes</td></tr><tr><td>Palm breadth</td><td></td><td>88.1 (5.4) 85.3 (4.2) +2.82 +3.3 +0.59 yes</td></tr><tr><td>Thumb length</td><td></td><td>65.3 (5.2) 64.8 (4.1) +0.49 +0.8 +0.11 no</td></tr><tr><td>Thumb breadth</td><td></td><td>22.1 (2.1) 21.5 (1.5) +0.65 +3.0 +0.37 no</td></tr><tr><td>Index length</td><td></td><td>69.6 (4.8) 71.3 (4.2) −1.68 −2.4 −0.37 yes</td></tr><tr><td>Index breadth</td><td></td><td>20.2 (1.4) 20.5 (1.2) −0.26 −1.3 −0.20 no</td></tr><tr><td>Middle length</td><td></td><td>76.4 (4.5) 78.6 (4.5) −2.21 −2.8 −0.49 yes</td></tr><tr><td>Middle breadth</td><td></td><td>20.2 (1.2) 20.3 (1.3) −0.14 −0.7 −0.11 no</td></tr><tr><td>Ring length</td><td>70.2 (5.0)</td><td>74.0 (4.5) -3.77 -5.1 −0.80 yes</td></tr><tr><td>Ring breadth</td><td></td><td></td></tr><tr><td></td><td>18.9 (1.7)</td><td>19.4 (1.2) −0.49 −2.5 −0.34 no</td></tr><tr><td>Pinky length Pinky breadth</td><td>55.5 (5.9) 17.3 (1.6)</td><td>60.4 (4.3) -4.88 -8.1 −0.97 yes 17.5 (1.2) −0.24 −1.4 −0.17 no</td></tr></table>

Table S16: Sex-specific comparison of 260 successful dominant-hand palmar cases (197 male and 63 female) with Hsiao et al. Ours and Ref are mean (SD) in millimeters; ∆ = Ours−Ref, g is Hedges’ g, and FDR indicates Benjamini– Hochberg significance at q < 0.05.

Interpretation limits. The independent cohorts difer in recruitment, year, hand laterality (dominant versus right), and measurement protocol; palm breadth also uses a proxy. Their ofsets cannot separate measurement efects from cohort diferences. Reference summaries permit comparison of means and SDs only; statistical significance does not establish agreement.

## T Pipeline Failure-Mode Analysis

Denominators. PR and end-to-end completion were both 704/720 (97.8%): every rectified capture returned all 44 dimensions. Accuracy uses these 704 captures, with no imputed errors for the 16 failures.

Controlled captures. Visual review of all 16 PR failures found the wrist crease on or beyond a paper edge: 15 hands crossed a long edge and one sat low on the bottom edge. One or both lower paper prompts fell outside the sheet, the paper mask included background, and no valid quadrilateral was recovered. The complete sheet was visible in every failed frame. These failures involved seven participants (4, 4, 3, 2, 1, 1, and 1 captures); three of these participants had failures with both smartphones.

Field captures. All eight incomplete field captures failed during PR. Post-hoc review found five hands placed too high or low, with adaptive prompts outside the sheet; one incompletely framed sheet; one hand extending beyond the paper; and one covered paper corner.

## U Notes on the Literature Comparison

These notes accompany Table 1 in the main paper.

Automation refers to dimension extraction after acquisition without per-hand manual landmark annotation or digitization. Equipment lists capture hardware; computing and software are additionally required. Only values explicitly reported as hand-dimension MAE are reproduced. A dash indicates no such value is tabulated: Han and Park report dimension-specific MAD, and Nguyen et al. evaluate gaugeblock calibration. Han and Park claim 18 dimensions but tabulate 17. HandAnthro MAEs use the same 704 complete captures from 45 participants. Regional groups comprise 28 non-thumb, five thumb, and 11 palm/wrist dimensions; each mean weights its member dimensions equally.