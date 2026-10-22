# Open-Vocabulary Models and Label Propagation for Uncertainty-aware Pseudo-Labeling of Object Detection Datasets

1<sup>st</sup> Jonathan Klees

Institute of Computer Science

Osnabruck University ¨

jonathan.klees@uni-osnabrueck.de

2<sup>nd</sup> Timo Heller   
Department of Mathematics   
University of Wuppertal   
timo.heller@uni-wuppertal.de

3<sup>rd</sup> Matthias Rottmann Institute of Computer Science Osnabruck University¨ matthias.rottmann@uni-osnabrueck.de

Abstract—Object detection models rely on large-scale annotated datasets that require substantial manual annotation effort. The labeling costs further increase when annotation uncertainty is captured through soft labels, as multiple annotations must be collected for each object. We address this challenge by introducing a semi-supervised framework for the uncertaintyaware annotation of object detection datasets. Our approach leverages open-vocabulary object detectors to generate region proposals and semantically meaningful object representations. Based on these representations, we construct an object-level similarity graph and propagate probabilistic label information. Our framework assigns soft labels to previously unlabeled objects, quantifying both aleatoric and epistemic uncertainty. We evaluate our method on four object detection benchmarks with softlabel annotations and demonstrate that high-quality annotations can be obtained with only a small fraction of manually labeled objects. Furthermore, we show that detectors trained on semiautomatically annotated data achieve performance comparable to models trained on fully manually annotated data. We achieve state-of-the-art downstream detection performance for the task of semi-supervised object detection on the COCO dataset, with 40.5% AP virtually matching the performance of the model trained on fully manually constructed data (40.9%) at only 1% of labeling costs. These results highlight the potential of open-vocabulary models and graph-based label propagation for efficient, accurate and uncertainty-aware pseudo-labeling.

## I. INTRODUCTION

Modern object detectors rely on large-scale annotated datasets, whose creation requires substantial manual effort. In contrast to image classification, object detection annotations must specify the location and category of every object, making dataset construction particularly time-consuming. Furthermore, existing object detection benchmarks generally treat annotations as deterministic, despite the fact that ambiguity is inherent to many detection tasks due to occlusions, unclear object boundaries, blurring, and semantic uncertainty. Recently, object detection datasets that capture semantic uncertainty were proposed [1], [2]. Annotation uncertainty is captured through multiple independent annotations, often more than ten human evaluations per object. While such approaches provide valuable information about human uncertainty, their annotation costs may limit scalability. This motivates the development of semi-automated methods that can efficiently propagate label information and capture semantic uncertainty. While graphbased semi-supervised techniques proved to be effective in image classification [3]–[5], similar methods for uncertaintyaware annotation in object detection remain unexplored.

![](images/5abed7d0f24f81c7fd914f0bbbf05129b65a3c8094049b92a677b9b0599d3dc7.jpg)  
Fig. 1: Illustration of label propagation for object detection datasets. Initial region proposals are generated with Grounding DINO [6] and form a similarity graph (grey lines) over which sparse label information (solid bounding boxes) is propagated. This results in pseudo-labels with soft label estimates based on which false positives are filtered effectively.

In this work, we develop graph-based semi-supervised learning in the domain of object detection by leveraging powerful openvocabulary object detectors such as Grounding DINO [6] to generate initial region proposals and semantically meaningful object representations. With these object representations, we construct a similarity graph over which label information is propagated (Figure 1). To propagate label information, we rely on probabilistic label spreading [5] which generalizes classical label spreading [3] to the domain of soft labels and accounts for aleatoric and epistemic uncertainty. This way, our method propagates (potentially noisy) label information obtained for few objects to other localized but yet unlabeled objects. This enables the uncertainty-aware, semi-automated annotation of object detection datasets. As a result, all detected objects are equipped with a soft label that reflects the individual class probabilities, including the background class. Soft labels generalize conventional labels and reflect the aleatoric uncertainty associated with label assignments [7]. They allow for uncertainty-aware training [8] and thorough assessment of model uncertainty [2]. Creating soft labels manually requires the repeated evaluation of instances by multiple annotators so that datasets with soft labels are rare [2], [5].

In the domain of object detection, [2] re-annotated the four object detection benchmarks COCO [9], Pascal VOC [10], KITTI [11] and Cityscapes [12] where they equipped objects with soft labels based on at least 11 human annotations. We evaluate our framework on these datasets and quantify achieved data quality in terms of average precision [9] and its soft label variant average similarity [2]. The results demonstrate that high-quality uncertainty-aware annotations can be obtained from only a small fraction of manually labeled objects, substantially reducing annotation effort while maintaining competitive data quality. We can further reduce annotation effort by integrating active querying to steer the selection of objects to annotate next. We also show that our framework can effectively uncover label errors in existing datasets. Notably, we observe that the quality of semi-automatically constructed data is sufficiently high for downstream tasks such as training closed-set object detectors. Consequently, our framework achieves state-of-the art downstream detection performance in semi-supervised object detection [13], [14] on the COCO dataset, with 40.5% AP essentially matching the fully supervised performance of 40.9% at only 1% of labeling costs.

We summarize our main contributions as follows:

We introduce the first graph-based semi-supervised framework for uncertainty-aware annotation of object detection datasets. It leverages open-vocabulary models and probabilistic label spreading to estimate soft labels of objects, capturing both aleatoric and epistemic uncertainty.

We demonstrate the effectiveness of our method on four standard datasets. We obtain accurate annotations at a fraction of annotation effort where our method consistently outperforms baselines in terms of label quality.

We integrate active querying to further reduce the annotation effort required to obtain accurate annotations.

We show that training on semi-automatically constructed annotations yields comparable performance, compared to training on original annotations, setting a new state of the art in semi-supervised object detection on COCO.

## II. RELATED WORK

a) Graph-Based Semi-Supervised Learning.: Semi-supervised learning (SSL) on graphs is an established research field with seminal contributions by [3], [15]. Early work on label propagation [15] combines labeled and unlabeled data in a graph whose edge weights are typically derived from a Gaussian kernel. Class information from labeled samples is then iteratively propagated to neighboring unlabeled nodes while preserving the original seed labels. Building on this idea, label spreading [3] relaxes the hard clamping of initial labels and performs a diffusion process over the graph, where a parameter α governs the balance between propagated and initial label information. Since their introduction, numerous extensions have been proposed, including formulations based on hypergraphs [16], [17], methods designed for improved scalability and computational efficiency [18], [19], and alternative strategies for graph construction. Comprehensive surveys of graph-based semi-supervised learning can be found in [4], [20]. Beyond label propagation methods, annotation-efficient learning paradigms such as active learning and human-in-theloop systems have been introduced to further reduce labeling costs in semi-automated annotation frameworks [21], [22]. While label propagation is well-studied in the domain of image classification, no approaches for object detection datasets exist. We propose the transfer of established graph-based SSL techniques to the domain of object detection by leveraging openvocabulary models for initial region proposals and semantic object representations.

b) Label Propagation on an Object Level.: Semi-automated annotation of objects is well-established for video data where objects are annotated based on only a few labeled frames [23]– [27]. Here, annotated objects are to be re-identified in other frames of the video to propagate initial label information. This is typically achieved by appropriate feature extraction and leveraging motion cues for temporal consistency. The task is different from semi-automated annotation of object detection datasets since the data consists of similar scenes and labels are typically only propagated to the same objects across frames.

Wang et al. [28] conduct label propagation on an instance level. They construct a similarity graph over embeddings of segmented objects to propagate labels across instances using label spreading [3]. For this, the existence of segments is assumed even for unlabeled images. The method is designed for multi instance, single label datasets where an image is associated with multiple objects but only a single label. This work derives image labels through label propagation on the instance level. In contrast, our method introduces label propagation for object detection datasets and leverages open-vocabulary models for region proposals.

Tutevych et al. [29] propose a framework for the semiautomated annotation of instance segmentation datasets. A pretrained model is used for region proposals and proposed segments are embedded with multiple foundation models. The separation of classes in the embedding space is then learned using a contrastive loss. While this framework contains conceptually related ideas, it relies on a segmentation model pretrained on similar data and requires training for accurate label propagation. Instead, we focus on object detection datasets and do not require any training data besides the few labeled examples provided for label propagation.

c) Semi-Supervised Object Detection.: While there is no prior work that applies graph-based SSL techniques in the domain of object detection, the research field of semi-supervised object detection is related to our approach. Methods in this field seek to improve detection performance by leveraging additional unlabeled data during training [13], [14], [30]–[36]. Typically, learning-based approaches are proposed that generate pseudo-labels based on few initial labels. These pseudolabels can then be used as additional supervision during detector training which was shown to improve performance, compared to training with the limited labeled data available. Most existing approaches formulate semi-supervision at the image level and use detector-generated pseudo-labels rather than modeling similarities between object instances across the dataset. In addition, approaches like CSD [31], STAC [13], Humble Teacher [33], Consistent Teacher [36] are learningbased, typically involving a student-teacher structure and data augmentations, which implies training overhead and high computational costs. We develop graph-based SSL in the domain of object detection by leveraging open-vocabulary models which does not require training and offers explainability through appropriate quantification of both aleatoric and epistemic uncertainty of pseudo-labels.

![](images/7dbcb3626eb8d43c705364967d9f1dea0683960f639bf865d7c84ede9ba4bc03.jpg)  
Fig. 2: Method overview: Initial region proposals form a graph based on feature similarity and individual objects are queried for manual inspection. The queried object is assigned either one of the dataset’s classes or the background class. This label information is then propagated over the similarity graph resulting in soft label estimates for each object proposal. The estimate is iteratively refined with each object that is inspected.

d) Open-Vocabulary Object Detection.: Recent advances in open-vocabulary object detection enable localization of arbitrary object categories through text prompts [6], [37], [38]. Models such as Grounding DINO [6] have demonstrated strong zero shot performance and are increasingly used as building blocks for data annotation [39], [40]. In this work, we leverage open-vocabulary detectors to generate region proposals and object representations based on which we conduct graph-based label propagation. We compare our framework to the performance of these models alone.

## III. METHOD

We introduce a graph-based semi-supervised framework for the uncertainty-aware annotation of object detection datasets. Given a small set of annotated objects, label information propagates to unlabeled object instances while both aleatoric and epistemic uncertainty are captured. To this end, we leverage open-vocabulary models to generate object proposals and semantically meaningful object representations. A similarity graph is constructed over these representations and probabilistic label spreading (PLS) [5] is applied to diffuse information throughout the object-level graph. An overview of our method is illustrated in Figure 2 where individual objects are iteratively queried for evaluation and received label information propagates. Importantly, our method also models background probabilities by including a distinct background class, which enables the separation of foreground objects. The complete annotation framework consists of open-vocabulary model inference, post-hoc calibration of raw confidence and PLS to propagate available label information and quantify aleatoric and epistemic uncertainty of soft label estimates.

a) Task Description.: The task at hand is the semi-automated annotation of object detection datasets with soft labels. We assume that a set of given images $I _ { 1 } , \ldots , I _ { n }$ contains a collection of objects $\boldsymbol { B } _ { 1 } , \ldots , \boldsymbol { B } _ { n }$ and each individual object $b _ { j } ~ \in ~ B _ { i }$ is associated with a soft label $p _ { j } \in \Delta ^ { C + 1 }$ , where C is the number of classes in the dataset and $\Delta ^ { C + 1 }$ denotes the C + 1-dimensional probability simplex. The soft label is a categorical probability distribution over all classes including background, i.e., $p _ { j } \stackrel { \cdot } { = } ( p _ { j } ^ { 1 } , . . . , p _ { j } ^ { C } , p _ { j } ^ { b g } ) ^ { \top }$ such that $\textstyle \sum _ { i } p _ { j } ^ { i } = 1$ . It captures the aleatoric uncertainty associated with label assignment for this particular object, which is inherent to object detection datasets due to occlusions, blurring, small object size and ambiguous class affiliations [2]. In this setting, the GT comprises all objects prevalent in the images and the associated soft labels are the underlying semantic distributions, i.e., we assume that manual class assignments for an object follow this distribution. In practice, annotated datasets do not cover all objects and, soft labels, if existing at all, are merely approximations to the underlying distributions based on finitely many annotations collected.

b) Object Proposals.: We first apply an open-vocabulary object detector such as Grounding DINO [6] to all images and obtain a set of object proposals $\hat { d } _ { 1 } , \dots , \hat { d } _ { M }$ . Each proposal $\hat { d } _ { j }$ consists of a bounding box $\hat { b } _ { j }$ , an objectness score $\hat { s } _ { j } \in [ 0 , 1 ]$ and an embedding vector $\hat { v } _ { j } \in \mathbb { R } ^ { d }$ which encodes a representation of the detected object. Open-vocabulary models provide semantically rich representations that generalize across object categories and enable meaningful comparisons between object instances. We capture this information with the hidden representations of detected objects. Based on these embeddings, we construct a similarity graph for label propagation.

c) Post-hoc Calibration.: Obtaining reliable pseudo-labels requires the accurate assessment of proposed objects. One signal we leverage is the detector confidence itself. Since model confidences are often poorly calibrated [41], we conduct post-hoc calibration. Based on the provided annotations, we evaluate a subset of detections and calibrate model confidence using Isotonic Regression [42], [43]. For each evaluated detection, the target for model confidence is 1 if one of the object classes was correctly localized or zero otherwise. We then apply the regression model to all predictions and obtain posthoc calibrated confidences. Ultimately, our framework blends the calibrated detector confidence in a Bayesian setting with independent score estimates obtained through label propagation. The set of pseudo-labels then results from choosing a suitable confidence threshold so that sufficiently many objects are annotated but the number of false positives is low.

d) Graph Construction.: Based on the embeddings of proposed objects for the entire dataset, $\hat { v } _ { 1 } , \dotsc , \hat { v } _ { M }$ , we construct a similarity graph $G = ( V , E , W )$ , where each node corresponds to one object representation. Edges are established between neighboring objects in the embedding space using a k-nearestneighbor graph to ensure computational efficiency. The affinity matrix $\breve { W } \in \mathbf { \bar { \mathbb { R } } } ^ { M \times M }$ is built with a Gaussian kernel as in [3], i.e.,

$$
W _ { i j } = \exp \left( - \frac { 1 } { 2 \sigma ^ { 2 } } \| \hat { v } _ { i } - \hat { v } _ { j } \| ^ { 2 } \right) ,\tag{1}
$$

where σ denotes the kernel’s bandwidth parameter and a nonzero weight is only assigned to edges where at least one node is among the k-NN of the other. Diagonal entries of W are set to zero to avoid self-reinforcement. The affinity matrix captures semantic similarity between object instances across the entire dataset rather than within individual images and enables effective label propagation.

e) Probabilistic Label Spreading.: From a semi-supervised perspective, out of the pool of region proposals $\hat { b } _ { 1 } , \dotsc , \hat { b } _ { M }$ only a subset of objects is manually evaluated. We assume that each object $\hat { b } _ { j }$ is associated with a probability distribution $\hat { p } _ { j } \in \Delta ^ { C + 1 }$ so that querying an annotation for an object $\hat { b } _ { j }$ can be interpreted as sampling a class $i _ { j } \in \{ 1 , \ldots , C , C + 1 \}$ from the underlying $\hat { p } _ { j }$ . To propagate this label information, we employ Probabilistic Label Spreading (PLS) [5], a generalization of classical label spreading [3]. In PLS, label information is iteratively diffused through the graph according to the affinity matrix W and Laplacian regularization. As proposed by [3], we consider the normalized graph Laplacian

$$
S = D ^ { - { \frac { 1 } { 2 } } } W D ^ { - { \frac { 1 } { 2 } } } ,\tag{2}
$$

based on which label information is distributed over the graph using the diffusion kernel $( I - \alpha S ) ^ { - 1 }$ , where α controls the strength of propagation. More precisely, we encode the obtained annotation $i _ { j }$ for object $\hat { p } _ { j }$ in the initial label matrix $Y ~ \in ~ \mathbb { R } ^ { M \times C + 1 }$ by setting $Y _ { j , i _ { j } } ~ = ~ 1$ and keeping all other entries zero. This information then spreads as

$$
F = ( I - \alpha S ) ^ { - 1 } Y .\tag{3}
$$

To approximate (3), we use an efficient implementation of label spreading [44] that constructs a sparse nearest neighbor graph and leverages algebraic multigrid solvers that often achieve near-linear complexity for such large sparse systems due to their multilevel structure [45]. We normalize the propagation scores F by dividing with the score of the labeled node, i.e., $\tilde { F } = ( F _ { j , j } ) ^ { - \mathrm { i } } F$ so that the labeled node receives a score of 1 and neighboring nodes receive partial label information. Hence, we can interpret propagation scores as the number of virtual labels accumulated in each node. The propagation is repeated with each labeled object. Results are summed up and normalization yields estimated soft labels for every object proposal. Denoting $e _ { i _ { j } } \in \mathbb { R } ^ { C + 1 }$ as the unit vector resulting from the j-th query and having evaluated m object proposals $\hat { b } _ { 1 } , \dots , \hat { b } _ { m }$ , the soft label estimate for an object $\hat { b } _ { q }$ is the following weighted average:

$$
\hat { p } _ { q } = \frac { \sum _ { j = 1 } ^ { m } ( I - \alpha S ) _ { q , j } ^ { - 1 } e _ { i _ { j } } } { \sum _ { j = 1 } ^ { m } ( I - \alpha S ) _ { q , j } ^ { - 1 } } \in [ 0 , 1 ] ^ { C + 1 }\tag{4}
$$

PLS propagates probabilistic label information for individual objects and explicitly models uncertainty in the annotation process. As a result, based on few queried object labels, our method equips all proposed objects with estimated soft labels and captures epistemic uncertainty in terms of the absolute number of virtual labels received.

f) Background Modeling.: Open-vocabulary detectors may generate false positive proposals. To account for this, we explicitly model the background class. During propagation, background information is treated analogously to semantic class information and contributes to the final probability estimates. This way, evaluated false positives enable the separation of foreground and background proposals. Ultimately, each object proposal is assigned a soft label estimate $\hat { p } _ { j } ~ = ~ ( \hat { p } _ { j , 1 } , . . . , \hat { p } _ { j , C } , \hat { p } _ { j , \mathrm { b g } } )$ that separates foreground and background objects and captures class ambiguity while epistemic uncertainty can be derived from the number of virtual labels received.

Overall, our framework requires only a small number of manually annotated objects and does not involve any additional model training. Instead, it exploits the semantic structure induced by open-vocabulary object detectors to propagate single annotations across object detection datasets.

## IV. NUMERICAL RESULTS

Our experiments demonstrate the effectiveness of our framework for the efficient and uncertainty-aware annotation of object detection datasets. We compare our method to baselines and similar algorithms and evaluate it on an established benchmark for semi-supervised object detection [13], [14]. We explore the effect of core components through ablation studies. We further reduce annotation costs through active label querying and demonstrate that our framework also uncovers label errors in existing datasets effectively.

In IV-B, we compare our method to baselines in terms of achieved data quality under limited annotation budget.

To assess the effectiveness of our method, we compare it with state-of-the-art semi-supervised object detection methods on an established benchmark in IV-C.

Ablation studies in IV-D provide an assessment of core components of our framework, in particular regarding the OVOD model employed and the feature space considered.

We explore further reduction of annotation costs by leveraging active querying in IV-E.

As another application of our framework, we demonstrate its effectiveness regarding label error detection in IV-F.

## A. Experimental Setup

a) Datasets.: We evaluate our method for uncertainty-aware annotation of object detection datasets on four datasets for which soft labels are available. In [2], common datasets were re-annotated from scratch with a focus on high coverage to also capture ambiguous cases. Identified objects were verified by at least 11 annotators, resulting in a soft label that quantifies human uncertainty. Annotations are available for COCO [9] (validation split), Pascal VOC 2012 [10], KITTI [11] (training split), and Cityscapes [12]. If multiple splits are available, we consider them as a single dataset to be annotated at once.

b) Inference.: We generate initial detections using Grounding DINO [6] in a zero-shot setting. Following the official inference procedure, all categories are concatenated into a single text prompt, enabling simultaneous detection of all classes. Inference results in 300 detections per image where we extract prediction confidence for each class. For YOLO-World [38] and OWL-ViT [37] (considered in an ablation study), we follow the respective inference guidelines.

c) Label Spreading.: To apply Probabilistic Label Spreading (PLS) [5], we set up a k-NN graph in the feature space of the ultimate hidden layer of the model’s decoder. For Grounding DINO, these features have a dimension of 256. We include all predictions with a score of 0.2 or higher in the graph. The bandwidth σ of the Gaussian kernel is set to the average k-NN radius of object features, i.e., the average Euclidean distance of a feature $\hat { v } _ { j }$ to its k-th NN. To ensure efficient computation, as proposed by [44], we rely on Faiss [46] for nearest neighbor search and use Algebraic Multigrid solvers with GPU acceleration [47] to solve the linear system involved.

![](images/a2ff6f415f0e8ffebf05f2ca9b5d61c303751617b5ecc41e3e42e84773b1ff29.jpg)  
Fig. 3: Average Similarity $( \mathrm { A S } _ { 5 0 : 9 5 } )$ [2] as a function of annotation budget for our propagation method (PLS) on the COCO dataset against Gaussian Kernel Regression, k-NN regression, calibrated Grounding DINO confidence, and Hard-Label Injection. The dotted line marks the AS ceiling where all correctly localized detections perfectly align with the corresponding soft labels.

d) Annotation Budget.: Our framework considers initial region proposals as the pool of unlabeled objects. We evaluate a fraction of these predictions regarding their class affiliation and propagate the resulting label information. An annotation budget of 1% signifies that accordingly many predictions are evaluated. Predictions are sampled with replacement as we expect the label assignment to be probabilistic. We simulate the manual evaluation of predictions through IoU matching with the existing ground truth (GT) using a threshold of 0.5. If a prediction does not match with any GT object, the background class is assigned. If a matching bounding box exists, we sample a class from the corresponding soft label which simulates a single annotation.

e) Baseline: Grounding DINO Pseudo-Labels.: We compare our method to the baseline that uses the prediction confidence directly to include detections as pseudo-labels. Here, we keep all detections with a confidence of 0.5 or higher after posthoc calibration. Usually, confidence scores are only returned for the predicted class. To obtain a full soft label prediction, we extract confidence scores for the remaining classes as well and model the background probability as the product of counterprobabilities of all classes, i.e., $\begin{array} { r } { \hat { p } _ { b g } = \prod _ { \kappa = 1 } ^ { C } ( 1 - \hat { p } _ { \kappa } ) } \end{array}$ where $\hat { p } _ { \kappa }$ is the calibrated predicted probability for the κ-th class.

## B. Semi-automated, Uncertainty-aware Annotation of Object Detection Datasets

We compare our method with similar techniques and baselines. All methods equip the initial region proposals with soft label estimates based on the evaluation of sampled objects. We assess the quality of the semi-automatically annotated dataset in terms of Average Similarity (AS) [2], a metric that generalizes Average Precision (AP) to the domain of soft labels by replacing the binary notion of precision with a normalized similarity measure to compare predicted probability distributions with soft labels. We use 1 - total variation distance as similarity measure as proposed by [2], while similarity is zero for false positive predictions. We also evaluate AP performance.

![](images/41f23968955cb6a93158c30eb907f15afa10993cc25920b979cf9dbc36b7bb7e.jpg)  
Fig. 4: Benchmark results for semi-supervised object detection [13]. Average precision of a Faster R-CNN model with ResNet-50 backbone is reported on COCO validation data. The model is trained on pseudo-labels which are constructed using the label information available for a fraction of images. At 100% budget, additional unlabeled COCO images are used.

Comparable methods are Gaussian Kernel Regression (GKR) [48], [49] which propagates label information using a Gaussian Kernel, and k-NN regression where label information is averaged over the k-NN out of the labeled instances. For these parametrized methods, we conduct a grid search for hyperparameters and determine the best AS / AP value for each annotation budget for comparison. See appendix for more details and full parameter sweep results. In addition, we consider two baselines. First, we calibrate model confidence using isotonic regression and infer soft label estimates from calibrated probabilities as described above. Second, we simulate fully manual annotation through hard-label injection. Here, we assign evaluated objects their labels but do not propagate information. This yields a single annotation for evaluated objects and leaves non-evaluated objects at the soft label inferred from Grounding DINO scores.

As displayed in Figure 3, on COCO validation data, our method achieves the highest AS at small annotation budgets (the regime of practical interest) and remains comparable to GKR as the budget grows further. Both clearly outperform calibrated Grounding DINO scores alone and hard-label injection across the full budget range, indicating that posthoc calibration alone is insufficient to approximate soft label probabilities, and that manual hard-label annotation requires substantially more effort to derive high-quality soft labels. At high budgets, semi-supervised methods approach the AS ceiling. This upper bound is mainly due to the detection capabilities of Grounding DINO which does not detect each

GT object and produces false positives. To compute this ceiling, we assign matching predictions the GT soft labels where an IoU threshold of 0.5 is used to match objects.

We evaluate achieved data quality in terms of AP in the appendix and find that our method surpasses Grounding DINO baseline performance with a moderate annotation budget (5– 10%). This amount of label information seems to be needed to obtain accurate score estimates through graph-based label spreading. A Bayesian blending of initial Grounding DINO score and our methods estimate leads to immediate improvements and performs comparably with hard-label injection that resorts predictions according to GT. Absolute gains in AP range from 1 percentage point at 10% to 4 at 100% budget. To further investigate annotation quality, we compute precision and recall of COCO pseudo-labels for Bayesian PLS at 1% annotation budget. We threshold annotations at the value that maximizes the F<sub>1</sub>-score on labeled images. For large objects, this yields a precision of 88% at a recall of 81% meaning that the majority of large objects are correctly annotated.

We obtain similar results on Cityscapes and KITTI where semi-supervised methods outperform calibrated Grounding DINO and hard-label injection regarding AS. In terms of AP, however, hard-label injection is best on both datasets despite its comparatively low AS. This indicates that propagationbased methods better approximate soft labels while relative ranking of predictions is best with injection of available GT. On Pascal VOC, calibrated Grounding DINO is a strong baseline such that our method requires a larger annotation budget of around 25% to overtake other methods in terms of both AS and AP. Interestingly, at a budget of 100%, we even achieve a higher AP than hard-label injection meaning that we improve over the optimal ranking at the fixed IoU threshold of 0.5 by smoothing label information over the graph. We discuss these dataset-specific findings further in the appendix.

## C. Semi-Supervised Object Detection

We compare our method to recent methods in the field of semisupervised object detection (SSOD). Here, label information is provided for a small fraction of images and the task is to generate pseudo-labels for the larger remainder of unlabeled images. Label quality is assessed by downstream detection performance of a Faster R-CNN model [50] with ResNet-50 [51] backbone. We rely on the implementation of Soft Teacher [14] and follow the same training and evaluation protocol, which we describe in the appendix. Results on COCO validation data, averaged over the 5 folds of annotated training data is displayed in Figure 4, the corresponding standard deviations are listed in the appendix.

We significantly outperform existing methods already with minimal training data, achieving 40.5% AP with 1% of labeled images which almost matches the performance of the model trained on the entire training data (40.9%). This strong performance is due to the fact that, in contrast to existing learningbased methods, OVOD-based pseudo-labeling does not depend on training data since the model operates in a zero-shot setting. The Grounding DINO baseline where annotations are used to calibrate model confidences in order to have a default choice of 0.5 for the confidence threshold of pseudo-labels performs very well already (e.g., 40.1% AP with 1% of labeled data). It is further edged out by our Bayesian PLS approach that propagates received label information over the similarity graph to estimate independent scores and blends them with Grounding DINO scores. We optimize hyperparameter on the labeled data available as discussed in the appendix.

TABLE I: Downstream detection performance (AP in %) for Faster R-CNN, YOLOv8-L, and RT-DETR trained with different annotation sets. We compare fully supervised training with Grounding DINO pseudo-labels and our pseudo-labeling approach, distinguishing two annotation budgets (1% / 10% of labeled images provided). Best (bold) and second best (underlined) values per row are highlighted.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Model</td><td rowspan="2">Full 100%</td><td colspan="2">Grounding DINO</td><td colspan="2">Ours</td></tr><tr><td>1%</td><td>10%</td><td>1%</td><td>10%</td></tr><tr><td rowspan="3">COCO</td><td>Faster R-CNN</td><td>38.6</td><td>38.6</td><td>38.6</td><td>38.7</td><td>38.9</td></tr><tr><td>YOLOv8-L</td><td>47.8</td><td>46.8</td><td>46.9</td><td>46.9</td><td>47.9</td></tr><tr><td>RT-DETR</td><td>47.7</td><td>46.7</td><td>46.4</td><td>46.7</td><td>47.0</td></tr><tr><td rowspan="3">VOC</td><td>Faster R-CNN</td><td>44.0</td><td>44.3</td><td>45.0</td><td>44.8</td><td>44.8</td></tr><tr><td>YOLOv8-L</td><td>50.5</td><td>51.5</td><td>52.0</td><td>52.8</td><td>53.1</td></tr><tr><td>RT-DETR</td><td>66.4</td><td>65.3</td><td>65.4</td><td>65.9</td><td>65.9</td></tr><tr><td rowspan="3">Cityscapes</td><td>Faster R-CNN</td><td>33.7</td><td>25.6</td><td>27.1</td><td>19.6</td><td>27.9</td></tr><tr><td>YOLOv8-L</td><td>30.7</td><td>22.4</td><td>23.0</td><td>20.0</td><td>23.7</td></tr><tr><td>RT-DETR</td><td>38.7</td><td>27.7</td><td>30.1</td><td>24.5</td><td>29.7</td></tr><tr><td rowspan="3">KITTI</td><td>Faster R-CNN</td><td>65.7</td><td>21.7</td><td>31.4</td><td>10.9</td><td>32.6</td></tr><tr><td>YOLOv8-L</td><td>77.7</td><td>20.9</td><td>29.4</td><td>12.7</td><td>31.1</td></tr><tr><td>RT-DETR</td><td>73.5</td><td>26.1</td><td>34.6</td><td>15.5</td><td>34.5</td></tr></table>

We assess downstream detection performance of Faster R-CNN [50], YOLOv8-L [52], and RT-DETR [53] on four datasets in Table I. We follow the official training recipes and compare AP on validation data when training on the full set of annotations vs. constructed pseudo-labels. In addition to COCO, we also achieve comparable detection performance on Pascal VOC when training on pseudo-labels. In contrast, manual labels are superior for the driving scene datasets Cityscapes and KITTI where the gap in performance is most severe on KITTI. Our approach consistently performs better than the Grounding DINO baseline and overall, leads to competitive performance across models and datasets with limitations on complex street scene datasets, in particular, KITTI.

Overall, we significantly push the state of the art in SSOD, virtually matching detection performance of the fully supervised model on COCO and Pascal VOC with very little labeled data. These results underline the effectiveness of current openvocabulary models and their usefulness for semi-automated annotation in object detection. Given that our method closely approaches the ’ceiling’ of the fully supervised model, it should be noted that the evaluation protocol is biased towards the original dataset. These annotations are treated as the gold standard even though manual annotations have limitations as well [2]. While it is not possible to directly compare data quality between two sets of annotations (each of them being imperfect), accurate assessment of downstream detection performance requires a third, independent set of annotations.

TABLE II: Ablation Results: Adding additional models to Grounding DINO as an ensemble does not improve recall of COCO pseudo-labels but introduces false positives (left). Embedding detected objects with CLIP [54] for graph construction leads to overall worse results (right).  
(a) OVOD Ensemble where predictions are pooled using NMS.  
(b) Grounding DINO vs. CLIP features for graph construction.
<table><tr><td>Method</td><td>Prec.</td><td>Recall</td><td>F1</td></tr><tr><td>Grounding DINO</td><td>84.9</td><td>61.4 71.3</td><td></td></tr><tr><td>+ YOLO-World</td><td>84.6</td><td>61.5 71.2</td><td></td></tr><tr><td>+ YOLO-World + OWL-ViT</td><td>82.7</td><td>61.5 70.5</td><td></td></tr></table>

<table><tr><td></td><td colspan="3">AP</td><td colspan="3">AS</td></tr><tr><td>Variant</td><td>1%</td><td>10%</td><td>100% |</td><td>1%</td><td>10%</td><td>100%</td></tr><tr><td>GDINO</td><td>44.4</td><td>48.6</td><td>52.7</td><td>27.8</td><td>31.8</td><td>34.9</td></tr><tr><td>CLIP</td><td>29.4</td><td>35.8</td><td>49.9</td><td>10.8</td><td>17.1</td><td>21.5</td></tr></table>

With the refined annotations of [2], we can re-evaluate the performance of models on independent annotations. The refined annotations are more rigorous and include more object annotations than the original ones. For COCO, Pascal VOC, and Cityscapes, a similar pattern as in Table I can be observed. Pseudo-labeling leads to overall competitive performance, where full supervision remains clearly best on Cityscapes, while pseudo-labeling is best on Pascal VOC and results are mixed on COCO. Results on KITTI, however, change significantly. Models trained on pseudo-labels were evaluated as significantly worse on original annotations but actually score better on refined annotations, compared to models trained on the original data. This highlights the dependence on evaluation data, where YOLOv8-L, e.g., drops from 77.7% AP on original labels to only 21.7% when evaluating on the refined annotations. We refer to the appendix for the full performance results on independent annotations and training details.

## D. Ablation Studies

Our framework has two main design choices. For one, we chose Grounding DINO rather than a different or an ensemble of open-vocabulary models. Second, we build the similarity graph on Grounding DINO embeddings instead of using a separate model such as CLIP [54]. These choices are supported by two ablation studies, summarized in Table II.

First, we compare Grounding DINO alone against two OVOD ensembles (adding YOLO-World [38], and additionally OWL-ViT [37]). We first calibrate models individually and retain predictions with calibrated confidence of at least 0.5. To pool predictions, we apply NMS with an IoU threshold of 0.5, where in case of overlap, Grounding DINO boxes are kept. We measure pseudo-label quality in terms of recall and precision on COCO training data at an IoU threshold of 0.5. Recall remains essentially identical indicating that the other models do not detect any GT objects that Grounding DINO missed. However, they introduce false positives so that precision drops slightly and Grounding DINO alone achieves the highest F<sub>1</sub> score. Ensembling with additional OVODs does not prove to be beneficial as it trades away precision for no gain in recall.

![](images/afa9d04e9629e81721261022fc7fbe12161dcbef4cfe13885f217e4862c2282e.jpg)

![](images/a4925af8bd6bb28ac4e4ea5aaaa665ead2a5e7c2c1f848b5243028fd8c30ab5a.jpg)  
Fig. 5: Active querying results: We compare performance in terms of AS (left) and AP (right) on COCO for four querying strategies that choose different nodes to annotate next.

Second, we compare performance for two similarity graphs. One from Grounding DINO’s own decoder embeddings and one built from CLIP embeddings of the same detections. The CLIP graph leads to poorer performance where particularly for small budgets, label information is not sufficiently captured. We therefore build the graph on Grounding DINO’s embeddings directly. It is the simpler design and consistently performs best on both metrics.

## E. Active Querying of Objects to Annotate

From a practical perspective, we aim to further reduce annotation effort by actively querying labels. On COCO validation data, we query one label at a time and evaluate achieved AS and AP after every 100 queries. With each queried label, the soft label predictions are updated and we can assess aleatoric and epistemic uncertainty in every graph node. We compare three strategies to the baseline of random querying. Querying by aleatoric uncertainty which is quantified as the Shannon entropy of current soft label estimates to focus on ambiguous instances. Querying by epistemic uncertainty, quantified by the number of virtual labels received in each node to annotate the node that received the least label information so far. And, an equally weighted mixture of the two sources of uncertainty, where we normalize quantities using the maximal and minimal value occurring. Overall, the results in Figure 5 show that active querying by uncertainty can further reduce the annotation effort required to achieve a certain degree of data quality. Querying by epistemic uncertainty ensures broad coverage of the feature space, improving both metrics over the random baseline. Meanwhile, querying by aleatoric uncertainty may overly focus on boundary regions where ambiguous instances reside, reducing the quality of soft labels through noisy labels. However, the mixture of both sources of uncertainty shows best performance in terms of AP. While AS and AP behave quite differently, it can be concluded that both aleatoric and epistemic uncertainty are relevant for active querying.

## F. Label Error Detection via Label Smoothing

Using the label error evaluation protocol of [2], we assess the effectiveness of our framework for detecting label errors such as missing or poorly localized annotations. Our method smoothens all GT labels over the graph of detected objects such that false positive predictions that are semantically similar to true positives are considered as plausible label error proposals. The pool of label error proposals are NMS-filtered Grounding DINO predictions $\mathrm { ( I o U ) 2 0 . 6 ) }$ with confidence of at least 0.01. We compare ranking quality of these proposals between a random sorting baseline, Grounding DINO and Bayesian PLS. Hyperparameters of Bayesian PLS are optimized towards AP on held-out data using only original annotations. Results in terms of AP and highest $F _ { 1 }$ -score are listed in Table III. Bayesian PLS is best on COCO and Cityscapes while Grounding DINO alone is best on KITTI and Pascal VOC. Notably, our approach is best on Cityscapes where relatively few label errors exist and worst on KITTI, where original and refined annotations are quite different. Since label propagation increases bias towards GT, it can improve label error detection when annotations are mostly complete but inherits a bias when original and refined annotations diverge.

TABLE III: Comparison of label error detection capabilities. We consider originally false positive predictions as label error proposals and evaluate them on actual label errors using the evaluation protocol of [2]. We report AP and the highest $F _ { 1 ^ { - } }$ score of each method.
<table><tr><td></td><td colspan="2">COCO</td><td colspan="2">Cityscapes</td><td colspan="2">KITTI</td><td colspan="2">Pascal VOC</td></tr><tr><td>Method</td><td>AP</td><td>F1</td><td>AP</td><td>F1</td><td>AP</td><td>Fl</td><td>AP</td><td>F1</td></tr><tr><td>Random</td><td>3.50</td><td>7.30</td><td>1.69</td><td>3.36</td><td>3.50</td><td>6.76</td><td>0.65</td><td>1.30</td></tr><tr><td>GDino</td><td>34.51</td><td>40.89</td><td>19.36</td><td>26.99</td><td>58.80</td><td>57.05</td><td>35.91</td><td>41.29</td></tr><tr><td>Bayesian PLS</td><td>34.56</td><td>41.16</td><td>21.03</td><td>26.66</td><td>37.62</td><td>41.96</td><td>33.98</td><td>37.88</td></tr></table>

## V. CONCLUSION

We proposed a framework for graph-based semi-supervised learning in object detection. It enables the semi-automated annotation of object detection datasets with uncertainty-aware soft labels. We demonstrated its effectiveness by comparing it to baselines in terms of achieved soft label quality on multiple soft-labeled datasets. We showed that semi-automatically constructed COCO annotations are of high quality as training on them leads to virtually matching downstream detection performance, compared to training on manually constructed data. This sets a significantly improved new state of the art for semi-supervised object detection in the regime of partially labeled COCO training data. We further demonstrated that data-centric evaluation heavily depends on the evaluation data and treating manual annotations as the gold standard is an implicit assumption that may limit the accurate assessment of pseudo-label quality. We integrated active querying techniques and demonstrated their potential to further reduce annotation effort required to achieve high data quality. Finally, we showed that our framework is able to effectively uncover label errors in existing datasets by smoothing labels over the object-level similarity graph, achieving state-of-the-art performance on two out of four datasets. To support reproducibility and further research, we make our source code publicly available under https://github.com/JonathanKlees/lpod.

J.K. and M.R. acknowledge support by the German Federal Ministry of Research, Technology and Space (BMFTR) within the project RELiABEL (grant no. 16IS24019B).

## REFERENCES

[1] S. Penquitt, J. Klees, R. Cakaj, D. Kondermann, M. Rottmann, and L. Schmarje, “From Label Error Detection to Correction: A Modular Framework and Benchmark for Object Detection Datasets,” in International Conference on Computer Vision Theory and Applications, 2026.

[2] S. Penquitt, J. Klees, A. van Betteray, P. Jashnieh, P. Stehr, M. Rottmann, and L. Schmarje, “Object Detection Benchmarks are Incomplete: The Role of Label Errors and Annotation Uncertainty,” in Accepted at Advances in Neural Information Processing Systems Evaluations & Datasets Track, 2026.

[3] D. Zhou, O. Bousquet, T. Lal, J. Weston, and B. Scholkopf, “Learning¨ with Local and Global Consistency,” in Advances in Neural Information Processing Systems, 2003.

[4] O. Chapelle, B. Scholkopf, and A. Zien, Eds.,¨ Semi-Supervised Learning. MIT Press, 2006.

[5] J. Klees, T. Riedlinger, P. Stehr, B. Boddecker, D. Kondermann, and¨ M. Rottmann, “Probabilistic Label Spreading: Efficient and Consistent Estimation of Soft Labels with Epistemic Uncertainty on Graphs,” in Accepted at Asian Conference on Machine Learning, 2026.

[6] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, Q. Jiang, C. Li, J. Yang, H. Su, J. Zhu, and L. Zhang, “Grounding DINO: Marrying DINO with Grounded Pre-training for Open-Set Object Detection,” in European Conference on Computer Vision, 2024, p. 38–55.

[7] Q. Nguyen, H. Valizadegan, and M. Hauskrecht, “Learning Classification with Auxiliary Probabilistic Information,” in IEEE International Conference on Data Mining, 2011.

[8] P. Peng, R. C.-W. Wong, and P. S. Yu, “Learning on Probabilistic Labels,” in SIAM International Conference on Data Mining, 2014.

[9] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick, “Microsoft COCO: Common Objects in´ Context,” in European Conference on Computer Vision, 2014, pp. 740– 755.

[10] M. Everingham, L. V. Gool, C. K. I. Williams, J. Winn, and A. Zisserman, “The Pascal Visual Object Classes (VOC) Challenge,” International Journal of Computer Vision, vol. 88, no. 2, pp. 303–338, 2010.

[11] A. Geiger, P. Lenz, and R. Urtasun, “Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2012, pp. 3354– 3361.

[12] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele, “The Cityscapes Dataset for Semantic Urban Scene Understanding,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[13] K. Sohn, Z. Zhang, C.-L. Li, H. Zhang, C.-Y. Lee, and T. Pfister, “A Simple Semi-Supervised Learning Framework for Object Detection,” arXiv preprint arXiv:2005.04757, 2020.

[14] M. Xu, Z. Zhang, H. Hu, J. Wang, L. Wang, F. Wei, X. Bai, and Z. Liu, “End-to-End Semi-Supervised Object Detection with Soft Teacher,” in International Conference on Computer Vision, 2021, pp. 3060 – 3069.

[15] X. Zhu and Z. Ghahramani, “Learning from Labeled and Unlabeled Data with Label Propagation,” Carnegie Mellon University, Pittsburgh, Tech. Rep., 2002.

[16] F. Tudisco, A. R. Benson, and K. Prokopchik, “Nonlinear Higher-Order Label Spreading,” in Proceedings of the Web Conference, 2021, pp. 2402–2413.

[17] D. Eswaran, S. Kumar, and C. Faloutsos, “Higher-Order Label Homogeneity and Spreading in Graphs,” in Proceedings of The Web Conference, 2020, pp. 2493–2499.

[18] V. A. Traag and L. Subelj, “Large network community detection by fast<sup>ˇ</sup> label propagation,” Scientific Reports, vol. 13, no. 1, 2023.

[19] Y. Fujiwara and G. Irie, “Efficient Label Propagation,” in International Conference on Machine Learning, 2014.

[20] J. E. Van Engelen and H. H. Hoos, “A survey on semi-supervised learning,” Machine Learning, vol. 109, no. 2, pp. 373–440, 2020.

[21] M. Desmond, E. Duesterwald, K. Brimijoin, M. Brachman, and Q. Pan, “Semi-Automated Data Labeling,” in Advances in Neural Information Processing Systems Competition and Demonstration Track, 2021.

[22] Z. Zhang, E. Coutinho, J. Deng, and B. Schuller, “Cooperative Learning and its Application to Emotion Recognition from Speech,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 23, no. 1, pp. 115–126, 2015.

[23] J.-Y. Kao, D. Tian, H. Mansour, A. Vetro, and A. Ortega, “Geometricguided Label Propagation for Moving Object Detection,” in IEEE International Conference on Acoustics, Speech and Signal Processing, 2016.

[24] U. Waldmann, J. Bamberger, O. Johannsen, O. Deussen, and B. Goldlucke, “Improving Unsupervised Label Propagation for Pose¨ Tracking and Video Object Segmentation,” in DAGM German Conference on Pattern Recognition, 2022.

[25] L.-C. Chen, R. G. Lopes, B. Cheng, M. D. Collins, E. D. Cubuk, B. Zoph, H. Adam, and J. Shlens, “Naive-Student: Leveraging Semi-Supervised Learning in Video Sequences for Urban Scene Segmentation,” in European Conference on Computer Vision, 2020.

[26] Z. Chen, M. Yang, and S. Zhang, “Asymmetric Label Propagation for Video Object Segmentation,” in ACM International Conference on Multimedia in Asia, 2022.

[27] S. Hu, C.-H. Liu, J. Dutta, M.-C. Chang, S. Lyu, and N. Ramakrishnan, “PseudoProp: Robust Pseudo-Label Generation for Semi-Supervised Object Detection in Autonomous Driving Systems,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2022.

[28] Q. Wang, G. Chechik, C. Sun, and B. Shen, “Instance-Level Label Propagation with Multi-Instance Learning,” in International Joint Conference on Artificial Intelligence, 2017.

[29] V. Tutevych, R. Memmesheimer, L. Eichler, D. Pavlichenko, F. Schilke, R. Krudewig, and S. Behnke, “Efficient Image Annotation via Semi-Supervised Object Segmentation with Label Propagation,” arXiv preprint arXiv:2604.22992, 2026.

[30] G. Papandreou, L.-C. Chen, K. Murphy, and A. L. Yuille, “Weakly- and Semi-Supervised Learning of a DCNN for Semantic Image Segmentation,” in International Conference on Computer Vision (ICCV), 2015.

[31] J. Jeong, S. Lee, J. Kim, and N. Kwak, “Consistency-based Semisupervised Learning for Object Detection,” in Advances in Neural Information Processing Systems, 2021.

[32] Y.-C. Liu, C.-Y. Ma, Z. He, C.-W. Kuo, K. Chen, P. Zhang, B. Wu, Z. Kira, and P. Vajda, “Unbiased Teacher for Semi-Supervised Object Detection,” in International Conference on Learning Representations (ICLR), 2021.

[33] Y. Tang, W. Chen, Y. Luo, and Y. Zhang, “Humble Teachers Teach Better Students for Semi-Supervised Object Detection,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.

[34] B. Chen, P. Li, X. Chen, B. Wang, L. Zhang, and X.-S. Hua, “Dense learning based semi-supervised object detection,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[35] H. Li, Z. Wu, A. Shrivastava, and L. Davis, “Rethinking Pseudo Labels for Semi-supervised Object Detection,” AAAI Conference on Artificial Intelligence, 2022.

[36] X. Wang, X. Yang, S. Zhang, Y. Li, L. Feng, S. Fang, C. Lyu, K. Chen, and W. Zhang, “Consistent-Teacher: Towards Reducing Inconsistent Pseudo-targets in Semi-supervised Object Detection,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[37] M. Minderer, A. Gritsenko, A. Stone, M. Neumann, D. Weissenborn, A. Dosovitskiy, A. Mahendran, A. Arnab, M. Dehghani, Z. Shen, X. Wang, X. Zhai, T. Kipf, and N. Houlsby, “Simple Open-Vocabulary Object Detection,” in European Conference on Computer Vision, 2022, p. 728–755.

[38] T. Cheng, L. Song, Y. Ge, W. Liu, X. Wang, and Y. Shan, “YOLO-World: Real-Time Open-Vocabulary Object Detection,” IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 16 901– 16 911, 2024.

[39] X. Zhao, Y. Chen, S. Xu, X. Li, X. Wang, Y. Li, and H. Huang, “An Open and Comprehensive Pipeline for Unified Object Grounding and Detection,” arXiv preprint arXiv:2401.02361, 2024.

[40] C. Xin, A. Hartel, and E. Kasneci, “DART: An Automated End-to-End Object Detection Pipeline with Data Diversification, Open-Vocabulary Bounding Box Annotation, Pseudo-Label Review, and Model Training,” Expert Systems with Applications, vol. 258, 2024.

[41] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On Calibration of Modern Neural Networks,” in International Conference on Machine Learning, 2017, pp. 1321 – 1330.

[42] T. Robertson, F. Wright, and R. Dykstra, Order Restricted Statistical Inference. John Wiley & Sons, 1988.

[43] B. Zadrozny and C. Elkan, “Transforming Classifier Scores into Accurate Multiclass Probability Estimates,” in ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2002, pp. 694 – 699.

[44] A. van Betteray, J. Klees, M. Schafers, and M. Rottmann, “Algebraic¨ Multigrid Acceleration for Efficient Label Spreading,” in Workshop on Curated Data for Efficient Learning, European Conference on Computer Vision Workshops, 2026.

[45] U. Trottenberg, C. W. Oosterlee, and A. Schuller, Multigrid. San Diego: Academic Press (Elsevier Science & Technology), 2000.

[46] M. Douze, A. Guzhva, C. Deng, J. Johnson, G. Szilvasy, P.-E. Mazare,´ M. Lomeli, L. Hosseini, and H. Jegou, “The Faiss library,” ´ arXiv preprint arXiv:2401.08281, 2025.

[47] M. Naumov, M. Arsaev, P. Castonguay, J. Cohen, J. Demouth, J. Eaton, S. Layton, N. Markovskiy, I. Reguly, N. Sakharnykh, V. Sellappan, and R. Strzodka, “AmgX: A library for GPU accelerated algebraic multigrid and preconditioned iterative methods,” SIAM Journal on Scientific Computing, vol. 37, no. 5, pp. 602–626, 2015.

[48] E. A. Nadaraya, “On estimating regression,” Theory of Probability & Its Applications, vol. 9, no. 1, 1964.

[49] G. S. Watson, “Smooth Regression Analysis,” Sankhya: The Indian¯ Journal of Statistics, Series A (1961-2002), vol. 26, no. 4, pp. 359–372, 1964.

[50] S. Ren, K. He, R. Girshick, and J. Sun, “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks,” in Advances in Neural Information Processing Systems, 2015, pp. 91–99.

[51] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015, pp. 770 – 778.

[52] G. Jocher, A. Chaurasia, and J. Qiu, “Ultralytics YOLOv8,” GitHub Repository, 2023. [Online]. Available: https://github.com/ultralytics/ ultralytics

[53] Y. Zhao, W. Lv, S. Xu, J. Wei, G. Wang, Q. Dang, Y. Liu, and J. Chen, “DETRs Beat YOLOs on Real-time Object Detection,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 16 965–16 974.

[54] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, “Learning Transferable Visual Models From Natural Language Supervision,” in International Conference on Machine Learning, 2021.

[55] Q. Zhou, C. Yu, Z. Wang, Q. Qian, and H. Li, “Instant-Teaching: An End-to-End Semi-Supervised Object Detection Framework,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 4081–4090.

[56] F. Zhang, T. Pan, and B. Wang, “Semi-Supervised Object Detection with Adaptive Class-Rebalancing Self-Training,” arXiv preprint arXiv:2107.05031, 2021.

[57] G. Li, X. Li, Y. Wang, S. Zhang, Y. Wu, and D. Liang, “PseCo: Pseudo Labeling and Consistency Training for Semi-Supervised Object Detection,” in European Conference on Computer Vision, 2022, pp. 457– 472.

[58] T.-Y. Lin, P. Dollar, R. Girshick, K. He, B. Hariharan, and S. Belongie,´ “Feature Pyramid Networks for Object Detection,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 2117 – 2125.

[59] K. Chen, J. Wang, J. Pang, Y. Cao, Y. Xiong, X. Li, S. Sun, W. Feng, Z. Liu, J. Xu, Z. Zhang, D. Cheng, C. Zhu, T. Cheng, Q. Zhao, B. Li, X. Lu, R. Zhu, Y. Wu, J. Dai, J. Wang, J. Shi, W. Ouyang, C. C. Loy, and D. Lin, “MMDetection: Open MMLab Detection Toolbox and Benchmark,” arXiv preprint arXiv:1906.07155, 2019.

## A. Supplementary Material Overview

This appendix complements the main manuscript with additional empirical studies, implementation details, and visual examples, which were beyond the scope of the main paper. Below, we give a brief overview to facilitate the navigation of the supplementary material.

Section B lists the full results and hyperparameters for the Semi-supervised Object Detection benchmark. These results are illustrated as a bar chart in the main paper and are now reported with two decimal places and standard deviations with a comprehensive comparison to other methods.

Section C complements the implementation details and results for achieved annotation quality of semi-supervised methods and baselines with average precision as a second metric and performance on other datasets.

Section D studies the effect of hyperparameters such as the spreading intensity, the number of neighbors and the minimal confidence for initial predictions.

Section E provides more details regarding the active querying techniques applied and provides additional insights regarding uncertainty-based querying.

Section G provides visual examples of detected objects and compares soft labels with the estimates of Grounding DINO alone and our framework.

## B. Full Results for Semi-Supervised Object Detection

<table><tr><td>Method</td><td>1%</td><td>5%</td><td>10%</td><td>100%</td></tr><tr><td>Supervised Baseline</td><td>10.00 ± 0.26</td><td>20.92 ± 0.15</td><td>26.94 ± 0.11</td><td>40.90</td></tr><tr><td>CSD [31]</td><td>10.51</td><td>18.63</td><td>22.46</td><td>38.82</td></tr><tr><td>STAC [13]</td><td>13.97 ± 0.35</td><td>24.38 ± 0.12</td><td>28.64 ± 0.21</td><td>39.21</td></tr><tr><td>Instant Teaching [55]</td><td>18.05</td><td>26.75</td><td>30.40</td><td></td></tr><tr><td>Humble Teacher [33]</td><td>16.96</td><td>27.70</td><td>31.61</td><td></td></tr><tr><td>Unbiased Teacher [32]</td><td>20.75</td><td>28.27</td><td>31.50</td><td>41.30</td></tr><tr><td>Soft Teacher [14]</td><td>20.46 ± 0.39</td><td>30.74 ± 0.08</td><td>34.04 ± 0.14</td><td>44.50</td></tr><tr><td>ACRST [56]</td><td>26.07</td><td>31.35</td><td>34.92</td><td>42.79</td></tr><tr><td>PseCO [57]</td><td>22.43</td><td>32.50</td><td>36.06</td><td>46.10</td></tr><tr><td>Grounding DINO [6]</td><td> $4 0 . 1 4 \pm 0 . 1 1$ </td><td> $4 0 . 1 9 \pm 0 . 0 8$ </td><td> $4 0 . 3 0 \pm 0 . 1 0$ </td><td>43.24</td></tr><tr><td>Ours</td><td> ${ \bf 4 0 . 5 0 \pm 0 . 2 3 }$ </td><td> $\pm 0 . 6 5 \pm 0 . 1 1$ </td><td> ${ \bf 4 0 . 7 4 \pm 0 . 0 8 }$ </td><td>43.71</td></tr></table>

TABLE A.1: Comparison of different semi-supervised object detection methods on COCO. The results are reported for different fractions of annotated training data in terms of mean average precision (AP). For annotation budgets below 100%, results are averaged over 5 folds of annotated data and standard deviation is reported, following the evaluation protocol of [13]. We list the numbers for existing methods as reported by [14] and [36] where standard deviations are not always given.

In Table A.1, the full results for the semi-supervised object detection benchmark are collected. We follow the evaluation protocol proposed by [13], which is also implemented in the codebase of [14]. We list the results of existing methods as reported by [14] and [36] and apply the same training schemes with the pseudo-labels we constructed to generate the results for our method and the Grounding DINO baseline. The proposed training scheme for the Faster R-CNN [50] with ResNet-

50 [51] backbone and Feature Pyramid Network [58] uses an initial learning rate of 0.01, which is divided by 10 after 120, 000 and once again after 160, 000 iterations. The training runs for 180, 000 iterations in total on 8 GPUs with a batch size of 5 images per GPU. In contrast, for the budget of 100%, the training schedule is four times longer (720, 000 iterations with the above learning rate decays at 480, 000 and 640, 000 iterations) and the batch size is increased to 8. Further training parameters are based on the MMDetection framework [59] and we refer to the supplementary material of [14] for the full description of hyperparameters and further training details. We computed on 8 NVIDIA RTX6000 Pro Backwell GPUs with 96GB of memory each where one training of 180, 000 iterations takes about 10h and full benchmark evaluation takes about 9 days per method.

We generate pseudo-labels with our Bayesian PLS method that blends the initial Grounding DINO with the PLS derived propagation score. More precisely, if $\hat { p } _ { j } ^ { G D }$ denotes the soft label estimate by Grounding DINO after post-hoc calibration and $F _ { i } \in \mathbb { R } ^ { C + 1 }$ denotes the raw propagation scores derived from PLS in the j-th node, the Bayesian PLS estimate is

$$
\begin{array} { r } { \hat { p } _ { j } = \kappa \ \hat { p } _ { j } ^ { G D } + F _ { j } , } \end{array}\tag{5}
$$

where $\kappa ~ > ~ 0$ is the prior weight and $F _ { j }$ is a vector that corresponds to the virtual number of labels received for each class. We use the following procedure for pseudo-label construction. For the budgets 1%, 5%, 10%, we first build the graph of Grounding DINO embeddings of all predictions with a minimal confidence of 0.2 on the entire COCO training dataset. Based on the images that serve for supervision, we evaluate the corresponding predictions using an IoU threshold of 0.5 and non-greedy matching. We then propagate the binary information (TP / FP) across the graph to obtain independent score estimates for all objects, which are then blended with the calibrated Grounding DINO score. We fit an isotonic regression model to calibrate Grounding DINO confidence using the TP / FP evaluations of the predictions on the labeled images as binary targets. We optimize the diffusion hyperparameters with a grid search of $k ~ \in ~ \{ 1 0 , 2 0 , 5 0 \}$ $\alpha \in \{ 0 . 7 , 0 . 8 5 , 0 . 9 , 0 . 9 5 , 0 . 9 9 \}$ and the Bayesian blend weight $\kappa \in \{ 0 . 0 1 , 0 . 0 3 , 0 . 0 5 , 0 . 1 , 0 . 2 , 0 . 3 , 0 . 5 , 1 . 0 , 2 . 0 , 5 . 0 , 1 0 . 0 \}$ using an 80/20 image-level split of the provided labeled images. We select the parameter combination that maximizes the $F _ { 1 ^ { - } }$ score on the 20% of held-out images by propagating the label information from the remaining 80% of labeled images. Hence, the best operating point in terms of $F _ { \mathrm { 1 } } { \mathrm { - s c o r e } }$ on the respective precision-recall curve is determined and this optimal threshold is then used to include pseudo-labels. To derive the final set of pseudo-labels, the model with the optimal parameter combination is then evaluated using the full set of provided labeled images as supervision for label propagation.

Comparing the training curves in Figure A.1 that display validation AP depending on the number of iterations, we observe that the Faster R-CNN model learns significantly faster due to the additional pseudo-labeled images available, compared to the supervised baseline. Here, we consider a budget of 10% and the first of five folds of labeled images provided for benchmark evaluation [14]. For this budget and fold, we ultimately match the performance of the model that is trained on the full training data, which indicates the high quality of generated pseudo-labels. In particular, since the baseline model leverages a four times longer training schedule with a larger batch size as well, so images are processed more often during training and gradients may be more precise.

![](images/d20850159046b1ad6c4283c62dc6f2f2571fcca9f3c225b00bc1c41712518bd1.jpg)  
Fig. A.1: Training curves for Faster R-CNN with ResNet-50 backbone on COCO for the supervised baseline with 10% of labeled images and our pseudo-labels. We observe a fast increase in validation AP that ultimately approaches the performance of the model trained on the entire set of training data and surpasses other pseudo-labeling approaches (dashed lines).

To assess downstream detection performance on all four datasets, we train Faster R-CNN [50], YOLOv8-L [52], and RT-DETR [53] once on the full training data and in comparison, on pseudo-labeled data. Pseudo-labels are based on either 1% or 10% of annotated images and generated with Grounding DINO alone or Bayesian PLS as described above. All three models are trained with standard hyperparameters regarding optimizer, learning-rate schedule and augmentations and we deviate only as described below. Every model is initialized from ImageNet-pretrained backbone weights not from a COCO-pretrained checkpoint so that the comparison is not skewed by an existing COCO prior. We follow each architecture’s official training recipe where YOLOv8-L is trained for 500 epochs and Faster R-CNN for 26 epochs (the latter distributed across 4 GPUs to match the reference batch size) on all four datasets, while RT-DETR is trained for 100 epochs on Pascal VOC, Cityscapes and KITTI and for 10 epochs on COCO. All models are scored in terms of AP on each dataset’s validation split.

To reduce the dependence on original annotations, we also evaluate all models on independently constructed annotations for all datasets from [2]. The motivation for this is that the assessment of downstream detection performance assumes that original annotations are the gold standard by which model capabilities are to be measured. That way, the model that is trained on the most similar distribution of annotations (i.e, the original annotations if we assume the distribution is the same across training and validation data) is scored the highest in evaluation. But it might be that these annotations are imperfect and a better set of labels may exist. Hence, we compare annotation quality in terms of downstream detection performance on independent data to evaluate how well a model generalizes to these independent annotations based on the training data it was provided. The results in terms of average precision on validation data are displayed in Table A.2 for the three models considered.

TABLE A.2: Downstream detection performance (AP in %) for Faster R-CNN, YOLOv8-L, and RT-DETR, on independently constructed annotations of each dataset’s validation set, which were found to cover more object annotations [2].
<table><tr><td></td><td></td><td>Full</td><td colspan="2">Grounding DINO</td><td colspan="2">Ours</td></tr><tr><td>Dataset</td><td>Model</td><td>100%</td><td>1%</td><td>10%</td><td>1%</td><td>10%</td></tr><tr><td rowspan="3">COCO</td><td>Faster R-CNN</td><td>33.9</td><td>34.1</td><td>34.1</td><td>34.1</td><td>34.4</td></tr><tr><td>YOLOv8-L</td><td>44.5</td><td>42.9</td><td>43.1</td><td>42.9</td><td>43.6</td></tr><tr><td>RT-DETR</td><td>42.7</td><td>42.0</td><td>41.7</td><td>42.1</td><td>42.2</td></tr><tr><td rowspan="3">VOC</td><td>Faster R-CNN</td><td>37.8</td><td>38.2</td><td>38.8</td><td>39.9</td><td>39.0</td></tr><tr><td>YOLOv8-L</td><td>45.4</td><td>46.0</td><td>46.5</td><td>48.2</td><td>46.8</td></tr><tr><td>RT-DETR</td><td>57.4</td><td>57.1</td><td>57.6</td><td>58.8</td><td>57.9</td></tr><tr><td rowspan="3">Cityscapes</td><td>Faster R-CNN</td><td>35.5</td><td>28.4</td><td>30.0</td><td>21.8</td><td>31.5</td></tr><tr><td>YOLOv8-L</td><td>33.1</td><td>25.5</td><td>25.3</td><td>21.7</td><td>27.4</td></tr><tr><td>RT-DETR</td><td>40.7</td><td>31.6</td><td>33.3</td><td>26.8</td><td>33.2</td></tr><tr><td rowspan="3">KITTI</td><td>Faster R-CNN</td><td>22.8</td><td>19.3</td><td>23.0</td><td>10.3</td><td>25.8</td></tr><tr><td>YOLOv8-L</td><td>21.7</td><td>23.3</td><td>24.8</td><td>12.0</td><td>28.2</td></tr><tr><td>RT-DETR</td><td>24.3</td><td>26.1</td><td>29.6</td><td>14.3</td><td>30.1</td></tr></table>

## C. Further Results on Semi-automated, Uncertainty-aware Dataset Annotation

a) Implementation Details.: In the main manuscript we report achieved Average Similarity with GT on the COCO dataset as a function of the annotation budget provided. This budget indicates the fraction of pre-filtered detections evaluated (we keep those with confidence of at least 0.2) and we list the optimal value for each budget after conducting a hyperparameter grid search for parametrized methods. PLS is grid-searched over $k ~ \in ~ \{ 1 0 , 2 0 \}$ , a kernel-bandwidth scale $\in \ \{ 0 . 1 , 0 . 2 , 0 . 5 , 1 . 0 , 2 . 0 \}$ which scales the Gaussian kernel width, relative to the mean squared k-NN distance used by default and α at 15 values spanning 0.05–0.999. The Bayesian PLS builds on the winning parameter combination and sweeps the Grounding DINO prior weight κ (15 values, $0 . 1 \mathrm { - } 1 0 ^ { 5 } )$ . Gaussian kernel regression has a single parameter, the kernel bandwidth γ, which is swept across 60 log-spaced values, $1 0 ^ { - 6 } – 1 0 ^ { 6 } )$ . k-NN regression is evaluated for 15 k values ranging from 2–500), while the calibrated Grounding DINO baseline and hard label injection need no hyperparameter search. The sweep results for parametrized methods are displayed in Figure A.2, where it can be seen that the parameter spaces have been swept properly for all methods to arrive at optimal values.

We measure COCO annotation-quality in both AS and AP. For these metrics, PLS propagates two independent signals jointly over one graph. Since AS focuses on soft labels, it uses the propagated label information directly, i.e., each evaluate detection’s label is propagated resulting in soft label estimates for each object. In contrast, AP focuses on ranking of predictions, which is why in that case, we propagate the binary TP/FP signal obtained by IoU-matching each detection against the soft ground truth. Hence, label propagation rescores the original predictions based on the label information available. This can be interpreted as metaclassification, i.e., learning which predictions were correct and which ones were incorrect by propagating received label information over the similarity graph. AP is then computed with the standard COCO evaluation protocol [9] against the soft ground truth. To derive hard labels from soft labels, they are thresholded such that all objects with a minimal class probability of at least 0.5 for any non-background class are included in the set of annotations as proposed by [2].

b) AP Results.: Performance in terms of AP on COCO is displayed in Figure A.3. The calibrated Grounding DINO baseline is budget-invariant since recalibrating raw confidence scores does not change their ranking with the isotonic regression model applied. It serves as a fixed reference point and does not benefit from more data. PLS alone starts below this reference at low budgets and overtakes it with a budget of 5–10%, then improves further with more supervision. Hence, graph-based label spreading needs a critical mass of confirmed TPs and FPs in the graph before its re-ranking becomes competitive with the raw detector score. Bayesian blending of the PLS estimate with the calibrated Grounding DINO prior removes this weakness entirely. The blend matches or exceeds the baseline at every budget. The Bayesian PLS is closely trailing the ranking based on hard label injection, which is even briefly surpassed by pure PLS. Hard label injection puts confirmed TPs at the top of the confidence ranking and confirmed FPs at the very bottom while preserving original detector ordering between TPs and FPs (we assign TPs a score of $1 - 1 0 ^ { - 6 } ( 1 - \hat { s } )$ , and FPs a score of $1 0 ^ { - 6 } \hat { s }$ where sˆ is the original score). Absolute AP gains of Bayesian PLS over the Grounding DINO baseline grow from roughly 1 percentage point at 10% budget (49.8 vs. 48.9) to about 4 percentage points at 100% (52.6 vs. 48.9).

c) AP and AS Curves.: We take a closer look on the AS and AP curves of each method at a fixed IoU threshold of 0.5 on COCO. Looking at the Average Similarity Curves in Figure A.4a, we observe that alignment with soft labels is significantly improved for propagation-based methods, compared to the Grounding DINO calibration baseline and Hard-Label injection alone. PLS seems to achieve best calibration of soft labels for most confident cases while GKR catches up towards the lower-confidence regime. We used the 10% budget scenario to generate the AS curves. The AP curves at a budget of 100% in Figure A.4b show that the optimal sorting of Hard-Label injection is not reached by the propagationbased methods but they significantly improve the Grounding DINO baseline. Further, we investigate AS and AP at different IoU thresholds in Figure A.5 and notice that improvements over the Grounding DINO baseline are most prominent for low IoU thresholds, while the Grounding DINO score alone cannot be improved significantly in terms of AS and AP for very well-localized objects. Hence, localization quality seems to correlate with calibration of detection confidence.

![](images/64109508266e497b55da665cdc6c405bc71fda476db36493ca0708e873a46b08.jpg)  
Kernel Bandwidth γ  
(a) Gaussian Kernel Regression

![](images/b4c5ed082ba44e11529b6b5cf33f95701b5ef6ff855741b7c511c36b5a62e22d.jpg)  
Number of Neighbors k  
(b) k-NN Regression

![](images/bc41e30d1655c1c8dc64b07db2277cb211beba2c27736fc48961613885cd77fa.jpg)  
Spreading Intensity α  
(c) PLS

Fig. A.2: Performance in terms of Average Similarity for parametrized methods depending on hyperparameter values. Hyperparameters were evaluated over a grid for each annotation budget. For readability, we display the three annotation budgets 1%, 10%, and 100%. For PLS, since there are two hyperparameters, we indicate dependence on the parameter k by shading the minimal and maximal obtained value for each spreading intensity α.  
![](images/4162e80250f5d28376c18917d90803d95676de2b1cb16229f4632130027dfa69.jpg)  
Fig. A.3: Average Precision $( \mathrm { A P } _ { 5 0 : 9 5 } )$ [9] on COCO 2017 validation data with soft labels [2] depending on annotation budget for our propagation method (PLS) against Gaussian Kernel Regression, k-NN regression, calibrated Grounding DINO confidence, and Hard-Label Injection. We also include a Bayesian approach of our method based on the initial Grounding DINO score. Here, the goal is essentially to resort predictions based on obtained labels such that AP increases.

d) Performance on other Datasets.: We displayed Average Similarity (AS) as a function of annotation budget for the COCO dataset [9] in the main manuscript. Here, we additionally discuss performance in terms of Average Precision (AP) on COCO as well as results on other datasets. COCO AP results are displayed in Figure A.3, AS and AP results for Pascal VOC [10], KITTI [11], and Cityscapes [12] are displayed in figures A.6 and A.7 while Table A.3 gives an overview of results.

![](images/3e908ac9d464fc0ebc5b26ffaa5f4b7195f68613e125b290f922e2f22bb9db57.jpg)  
(a)

![](images/6185094ec81f00fff1d5b5fc52e9424c2d4ac24f0b7ebe01590d73713b6425fa.jpg)  
(b)

Fig. A.4: Similarity-Recall Curve (a) and Precision-Recall Curve (b) on COCO for an IoU threshold of 0.5. To determine similarity in terms of 1-TVD and precision, estimated and true soft labels are compared, where predictions are sorted by confidence as usual in COCO-style evaluation. The displayed curves are the average of all 80 class-specific curves.  
![](images/017ce3ab43ad0453e4473d8f2e8ae7378a45ff5bfd1f48d7b56e8330adf4f454.jpg)  
(a)

![](images/2998432efb5d3a57e10920ad66a4086630f5439bffb79f27c386dd4e686589b1.jpg)  
(b)  
Fig. A.5: Average Similarity (a) and Average Precision (b) depending on the IoU threshold used for GT matching of predictions. A common pattern is that improvements over the Grounding DINO baseline through label propagation are most prominent for lower IoU thresholds.

On Cityscapes and KITTI, results are similar to COCO for AS but quite different for AP. Propagation-based methods reach the AS ceiling almost immediately while calibrated Grounding DINO and Hard-Label Injection trail these methods by a wide margin on AS. This indicates that label propagation is very beneficial to approximate the soft labels on these two datasets, compared to post-hoc calibration alone. In terms of AP, however, Hard-Label Injection performs significantly better than propagation-based methods. This disagreement of AS and AP reflects that propagation-based methods are well suited for approximating the class distribution of objects, while the direct use of GT leads to the best ranking of detections that graph-based label spreading and kernel regression cannot fully match through the diffusion of binary TP/FP information.

![](images/ba6fa3927aa7337a088cbcffd49649bb3c7311b571ab3af4b1821c3a9d227fe3.jpg)  
Annotation budget (% of objects)  
(a) Pascal VOC

![](images/bbca5e55a940d4a0961b9237ffdecf7669d435eee2f8db172d40c285e774c4c1.jpg)  
Annotation budget (% of objects)  
(b) KITTI

![](images/87e6beb86441dcab9395f72243d0773df1919bfee05eae473670da2bd45b339f.jpg)  
Annotation budget (% of objects)  
(c) Cityscapes

Fig. A.6: Performance in terms of Average Similarity depending on the provided annotation budget for three datasets. Hyperparameters were swept through a grid search and optimal values for each annotation budget are displayed.  
![](images/1efede3a183941560178896558eeae297b3473e90daf0f1e64335fc4fbfa3e81.jpg)  
Annotation budget (% of objects)  
(a) Pascal VOC

![](images/760620e828d049fba1e3bfd3c20e9426fcdbe4b14161810907c227003a8b5112.jpg)  
Annotation budget (% of objects)  
(b) KITTI

![](images/092fa69b63499d7e617a37cd91efcfd381e0ad7eba43f460b3f588d63bcede56.jpg)  
Annotation budget (% of objects)  
(c) Cityscapes  
Fig. A.7: Performance in terms of Average Precision depending on the provided annotation budget for three datasets. Hyperparameters were swept through a grid search and optimal values for each annotation budget are displayed.

Pascal VOC results are slightly different again. Here, calibrated Grounding DINO scores are already a strong baseline at minimal budget. Because of this strong starting point, propagation-based methods only overtake baselines in both AS and AP once the annotation budget grows to roughly 25%. VOC is also the only dataset where Hard-Label Injection’s AP is non-monotonic in budget, dipping from 69.5% at 1% to 69.1% at 10% before recovering to 71.0% at 100%. One reason for this may be that Hard-Label Injection re-ranks based on a matching computed up front, so a duplicate detection for an already-matched GT object can occasionally be sampled and confirmed as a TP instead of the genuine true-positive box, which may have a better localization. This steal of the GT bounding box then injects a false positive near the top of the ranking. Notably, at 100% budget, both PLS and Bayesian PLS (71.7% and 71.5% AP respectively) surpass Hard-Label Injection (71.0% AP). So smoothing label information across the graph improves on the oracle ranking at an IoU threshold of 0.5, which is the optimal ranking at this threshold but potentially not at higher thresholds.

## D. Effect of Hyperparameters

To assess the sensitivity of our method to hyperparameters, we sweep the number of neighbors k and the spreading intensity α for three annotation budgets. We evaluate performance in terms of achieved Average Similarity (AS) on soft labeled COCO data as discussed in the main paper and display the results in Figure A.8. Overall, AS is relatively robust to both parameters, varying only mildly across most of the grid while degrading for extremely large values of α. The dependency on hyperparameters is budget-conditional. At small budgets, where few labels are available, heavier propagation is preferred, and the heatmap shows poor performance in the regime of low k and low α. In this case, information is presumably not spread sufficiently, resulting in sparsely-labeled regions. At larger budgets, labeled nodes are already dense enough that little propagation is needed, and the optimum shifts to small k and small α, which avoids oversmoothing.

To assess how the confidence threshold applied to raw Grounding DINO detections for pre-filtering affects performance, we swept it jointly with the spreading intensity α (score threshold $t ~ \in ~ [ 0 . 0 1 , 0 . 5 ] , ~ \alpha ~ \in ~ [ 0 . 0 5 , 0 . 9 9 9 ]$ , k = 20 fixed) across three annotation budgets, the results are shown in Figure A.9. The score threshold substantially affects AS, where lower thresholds lead to higher scores in this recall-sensitive metric across every budget and every α. At a lower threshold, more detections are retained, which expands the object graph by almost two orders of magnitude (from 20, 000 at t = 0.5 to well over a million nodes at $t = 0 . 0 1 )$ . The role of α is threshold-dependent. At low thresholds, where the graph is large and densely connected, AS does not vary significantly with α and reduces only marginally as $\alpha \  \ 1$ . At high thresholds, where only the most confident detections form the graph, AS increases from low α and plateaus at around $\alpha \approx 0 . 2 – 0 . 6$ before falling off as α approaches 1. This is consistent across all three budgets, and matches the intuition that a smaller graph offers fewer and less redundant neighbors to spread over. Choosing α too close to 1 leads to oversmoothing labels in that case. Overall, a large spreading intensity can be chosen if a small score threshold is applied such that the underlying graph is sufficiently large whereas at higher thresholds, a moderate α is preferable.

TABLE A.3: Overview of performance in terms of AS and AP (%) across datasets, methods, and three annotation budgets. Best values per budget is highlighted in bold, second best is underlined.  
(a) Average Similarity (AS)
<table><tr><td rowspan="2">Method</td><td rowspan="2">1%</td><td colspan="2">COCO</td><td colspan="3">Cityscapes</td><td colspan="3">KITTI</td><td colspan="3">Pascal VOC</td></tr><tr><td>10%</td><td>100%</td><td>1%</td><td>10%</td><td>100%</td><td>1%</td><td>10%</td><td>100%</td><td>1%</td><td>10%</td><td>100%</td></tr><tr><td>PLS</td><td>28.5</td><td>35.2</td><td>40.6</td><td>39.0</td><td>39.4</td><td>41.3</td><td>42.3</td><td>43.8</td><td>46.3</td><td>58.2</td><td>61.4</td><td>65.7</td></tr><tr><td>Bayesian PLS</td><td>27.5</td><td>34.0</td><td>39.8</td><td>36.0</td><td>38.4</td><td>40.7</td><td>41.6</td><td>43.4</td><td>46.4</td><td>61.7</td><td>63.3</td><td>65.6</td></tr><tr><td>GKR</td><td>26.6</td><td>35.3</td><td>41.6</td><td>38.8</td><td>39.7</td><td>41.1</td><td>43.5</td><td>44.4</td><td>46.5</td><td>54.3</td><td>61.2</td><td>66.1</td></tr><tr><td>k-NN</td><td>23.8</td><td>30.8</td><td>36.0</td><td>37.2</td><td>38.6</td><td>39.4</td><td>42.7</td><td>43.5</td><td>44.3</td><td>46.0</td><td>56.5</td><td>61.4</td></tr><tr><td>Calibrated GDino Score</td><td>26.4</td><td>30.3</td><td>30.7</td><td>32.8</td><td>34.0</td><td>34.2</td><td>41.0</td><td>41.8</td><td>41.9</td><td>61.6</td><td>63.3</td><td>63.3</td></tr><tr><td>Hard-Label Injection</td><td>26.9</td><td>28.9</td><td>38.3</td><td>22.6</td><td>25.5</td><td>36.7</td><td>30.6</td><td>32.6</td><td>42.1</td><td>49.2</td><td>51.7</td><td>63.0</td></tr></table>

(b) Average Precision (AP)
<table><tr><td></td><td colspan="3">COCO</td><td colspan="3">Cityscapes</td><td colspan="3">KITTI</td><td colspan="3">Pascal VOC</td></tr><tr><td>Method</td><td>1%</td><td>10%</td><td>100%</td><td>1%</td><td>10%</td><td>100%</td><td>1%</td><td>10%</td><td>100%</td><td>1%</td><td>10%</td><td>100%</td></tr><tr><td>PLS</td><td>46.0</td><td>49.4</td><td>52.9</td><td>28.6</td><td>30.5</td><td>32.0</td><td>28.3</td><td>31.1</td><td>33.1</td><td>68.6</td><td>69.9</td><td>71.7</td></tr><tr><td>Bayesian PLS</td><td>49.0</td><td>49.8</td><td>52.6</td><td>30.3</td><td>30.4</td><td>31.6</td><td>29.4</td><td>31.2</td><td>33.1</td><td>69.6</td><td>69.8</td><td>71.5</td></tr><tr><td>GKR</td><td>48.5</td><td>49.2</td><td>52.1</td><td>30.0</td><td>30.3</td><td>31.6</td><td>29.7</td><td>30.8</td><td>32.2</td><td>69.5</td><td>69.5</td><td>71.0</td></tr><tr><td>k-NN</td><td>47.9</td><td>48.2</td><td>49.6</td><td>28.2</td><td>29.5</td><td>30.1</td><td>28.5</td><td>29.0</td><td>30.4</td><td>69.0</td><td>69.4</td><td>69.6</td></tr><tr><td>Calibrated GDino Score</td><td>48.9</td><td>49.0</td><td>49.0</td><td>30.3</td><td>30.3</td><td>30.3</td><td>29.4</td><td>29.4</td><td>29.4</td><td>69.6</td><td>69.6</td><td>69.6</td></tr><tr><td>Hard-Label Injection</td><td>49.6</td><td>49.7</td><td>53.3</td><td>31.8</td><td>32.4</td><td>38.1</td><td>32.6</td><td>33.2</td><td>38.4</td><td>69.5</td><td>69.1</td><td>71.0</td></tr></table>

![](images/8a355bbad89f44dd8269b8c84222b24318db56761378d6356887374811d8612e.jpg)  
(a) 1% Budget

![](images/1a6d31ecb5b18da7d0d3924e27a991f08ea426c81f475d656d9def33521cf53f.jpg)  
(b) 10% Budget

![](images/c6b54fe41ec4f9a6aae47bdd8ccc15bea26ef85f3954a42f7e43273622162c21.jpg)  
(c) 100% Budget  
Fig. A.8: Hyperparameter analysis of our method for both spreading intensity α and the number of neighbors k. We evaluate Average Similarity on COCO 2017 validation data with soft labels [2] for each parameter combination.

## E. Active Querying

In terms of active querying, we evaluate four different strategies in terms of AS and AP in the main paper. Here, we give additional details on this experiment and provide more results that give an intuitive explanation of the strategies results.

For this study, we consider the COCO dataset, similar to the main data quality evaluation, we construct the graph on top of Grounding DINO embeddings, thresholded at a minimal confidence of 0.2. We use a fix parameter combination of $k = 2 0$ neighbors and $\alpha = 0 . 9$ for all methods and query one label at a time for each strategy. Based on the current set of labels, PLS with the fixed parameters is applied to estimate soft labels and objectness scores. We evaluate performance in terms of AS and AP every 100 queries as this is the most compute heavy part of the experiment. The random strategy queries a random node for labeling at a time, drawn uniformly from the set of all labels. Epistemic uncertainty querying leverages the normalization we apply to propagation scores such that a labeled node gets a propagation score of 1 and neighboring nodes get partial feedback. Hence, cumulative propagation scores can be interpreted as the number of virtual labels obtained. The node to label next is determined by the cumulative label feedback obtained so far, i.e., the node with minimal received label information (summed over all classes) is queried next. The aleatoric uncertainty strategy focuses on estimated soft labels directly and queries the next label for the node whose soft label estimate has the highest ambiguity in terms of Shannon entropy. Shannon entropy of an estimated soft label $\hat { p _ { j } } \in \Delta ^ { C + 1 }$ is given as

![](images/815a08b1c952a0cf482af4bd66d04d7032097d413f61cb36f5d87333b73146c2.jpg)  
(a) 1% Budget

![](images/44c36ceb0eab69c01863e302ef7ea142137e29ff2546d8ef9165b6a7336d4412.jpg)  
(b) 10% Budget

![](images/30bad94bd3c5808833e437b331f2571134416ff9f2e0d4d5fd08cd7b5c2ccf64.jpg)  
(c) 100% Budget  
Fig. A.9: Hyperparameter analysis of our method for the Grounding DINO score threshold applied as well as the spreading intensity α for a fixed number of neighbors $k = 2 0$ . We evaluate Average Similarity on COCO 2017 validation data with soft labels [2] for each parameter combination.

![](images/9919ba97b61f997eafc358552540ca3df52a5b4ed1ac199f46eb45c0438c9b2c.jpg)  
(a)

![](images/4a4cca3ceb973f95dc1d4065e5bf9e5486c4a94daaf08f0a185db38b60afda2d.jpg)  
(b)  
Fig. A.10: Coverage of labeled nodes during active querying in terms of two metrics where higher indicates better spread of labels across the feature space. In (a), the spread of labeled nodes is captured by the average distance between labeled nodes as a moving average with a window size of 100. Similarly, the redundancy of labeled nodes is captured in (b) by the average distance to any labeled node for all nodes in the graph. We see that querying by epistemic uncertainty leads to better spread of labels whereas this is not the case for aleatoric uncertainty which sits at the random baseline level.

$$
\mathcal { H } ( \hat { p } _ { j } ) = - \sum _ { i = 1 } ^ { C + 1 } \hat { p } _ { j , i } \log _ { 2 } ( \hat { p } _ { j , i } ) ,\tag{6}
$$

where the convention is to set a summand to zero whenever an appearing probability is zero. For the mixture strategy, we first normalize scores such that the node with the highest score is scored with 1 and the lowest with 0 through the corresponding affine transformation of scores. We then weight both uncertainty scores equally and query the node with the highest combined score.

TABLE A.4: Estimated annotation costs for our approach depending on the dataset and annotation budget. The estimate is based on each dataset’s median answer time per object and is compared to the annotation costs of fully manual annotation with soft labels as done by [2].
<table><tr><td></td><td>Median</td><td colspan="3">Cost at budget (h)</td><td>Fully manual</td></tr><tr><td>Dataset</td><td>answer time (s)</td><td>1%</td><td>10%</td><td>100%</td><td>annotation (h)</td></tr><tr><td>COCO</td><td>1.14</td><td>0.3</td><td>2.7</td><td>27.2</td><td>2,005.4</td></tr><tr><td>Cityscapes</td><td>0.82</td><td>0.2</td><td>1.8</td><td>17.7</td><td>992.2</td></tr><tr><td>KITTI</td><td>1.00</td><td>0.3</td><td>2.8</td><td>27.9</td><td>1,306.9</td></tr><tr><td>Pascal VOC</td><td>0.87</td><td>0.1</td><td>1.4</td><td>14.1</td><td>1,016.9</td></tr></table>

## F. Annotation Costs

Here, we compare the annotation costs that can be expected using our approach for the semi-automated, uncertainty-aware annotation of object detection datasets. The results here are based on [2] where annotation costs are reported for the fully manual dataset construction but also for the evaluation of individual objects by human annotators in a crowdsourcing scenario. With this detailed information on how long it takes an annotator to evaluate an object instance regarding its class affiliation, we can determine an estimate for the annotation costs associated with our approach. In particular, we use the median across all answer times per dataset as the expected annotation costs per object instance. This number is then multiplied with the number of objects to evaluate to derive the total annotation costs as listed in Table A.4. Comparing these numbers with the annotation costs of the fully manual approach, we conclude that the semi-automated approach is significantly more efficient in terms of labeling costs. Efficiency is driven by the automated construction of bounding boxes with Grounding DINO but also by avoiding the redundant evaluation of individual object instances to derive soft labels where we rely on probabilistic label spreading instead.

![](images/d3defb9093f4edff0212aca000d129df9bb68dcdac397c67891574df3f01431d.jpg)  
(a) Random Querying

![](images/acc4a1435cd8c74687d81e0e0c2f6abfc24db372c4afd5bdeee3edc9b16ad5e7.jpg)  
(b) Epistemic Uncertainty

![](images/bef08c6dc0f77cf2e82e0bae21f330fcd21c89f7bdb38ffcb194505c63242b6a.jpg)  
(c) Aleatoric Uncertainty

![](images/b56ec6444143f3da8b49b949ee33534c9c0b1232f422d75e0cc889946a535ca7.jpg)  
(d) Epistemic & Aleatoric

![](images/abe0cdbc549183cb94f30f1cee818e68ac9b12c23932e1606ae9b1eae3ef9ce6.jpg)  
Fig. A.11: First 200 Queried nodes in two-dimensional UMAP embedding space of Grounding DINO features of COCO validation data for different querying strategies .

## G. Visual Examples

For qualitative inspection of estimated soft labels, we visualize objects for three distinct cases. Objects where our estimate agrees with Grounding DINO on the class, objects where the two methods predict different classes, and lastly, where the soft ground truth itself is ambiguous (top-class probability below 0.9). For the agreement cases in Figure A.12 our method consistently predicts higher confidences than Grounding DINO’s calibrated confidence, resulting in better alignment with the corresponding soft labels. The disagreement examples in Figure A.13 consistently show that Grounding DINO’s calibrated confidence is overly leaning toward background on genuine, correctly-localized objects. E.g., the apple in a supermarket that Grounding DINO ranks below background but where label propagation recovers a high class probability by borrowing evidence from similar, higher-confidence detections in the embedding graph. On ambiguous ground in Figure A.14, our estimate and Grounding DINO’s calibrated confidence agree with each other on the predicted class and, in most cases, they also reproduce the ambiguity itself. In particular, they assign a reduced confidence similar to the GT annotations that build on multiple human assignments. For the bench in the train where the soft label is split evenly between bench and couch both methods predict a mixture of bench, couch and background as well. For the partially visible knife or other cutlery Grounding DINO and our method accurately express ambiguity by predicting a mixture of knife, spoon and background, similar to the soft GT. The exception is a fenceobscured truck in the ground truth (a), where both methods nonetheless relatively confidently predict the truck class while annotators did not identify it as a truck frequently. Overall, label propagation sharpens confidence for correct predictions, corrects cases where Grounding DINO was under-confident, and preserves ambiguity in cases where human annotators are uncertain as well. We observe a similar pattern for the examples on the other three datasets in figures A.15, A.16, and A.17.

![](images/8059c758e01b00b087ee155b8c1049940e85c9daa86be00be23493c0fadda997.jpg)

![](images/d2f03e74742a948341a644b18619f11c924a9dffde3da35960adf6ec83a1e539.jpg)  
(a)

![](images/cc30de1f1e3af31a49aec9a1458110903ea0b84254f59648ff0be413e6e15a0c.jpg)

![](images/2ebbf69160b89d4365ea1f2f000b70a6d7cb117b32c85c8615e6ad9490df5f82.jpg)  
(b)

![](images/6a574731dd003b5af4bff0ee3390d168e060f69e0f19e7ded2cea0feccf76c3d.jpg)

(c)  
![](images/713b2b6501f44393d598c7eb87e7ccb97ce834fbda9dabe2b5103b04c9566fa7.jpg)

![](images/a32021dc7d4d92c0c9821cf6006a0ba02062a6378f55907f826aa08a648a8996.jpg)

![](images/07643997e3145078806ac68baafe2993e0390ed546d3208657e9cc3556d78e23.jpg)  
(d)  
Fig. A.12: Examples where our method and Grounding DINO agree on the predicted class.

![](images/7dcff4a2f1c3d8de0a3525271a59d2d80ef713d2e506ba9c0082504830354663.jpg)

![](images/b73345a8d1760b735e9aefb41681709274b8c56430f7b1a1f6b4efc9b74a7b14.jpg)

![](images/7f97fea54620abd2db43c35cb0e4f6fd7485e4078e2be227158b3210949bfaee.jpg)

![](images/7a47927149a1aad8d901f518f56501a83025fb19d916c6e70b12fd4b0a6d88be.jpg)

(a)  
![](images/cc057b5580322a3cd8d6a9ad7697b9b762a5c6b72b34ebf9564c06b04e2d6222.jpg)

![](images/2d3c82bd1de63eee5f16bc4dfd6ffb4223bad1da059368dade867701a4cde707.jpg)  
(c)

(b)  
![](images/380e25a80e4e81567c324faaab0a34d86da7136c5b29b5aee239f0a18aa0df40.jpg)

(d)  
![](images/8cd8bedcee3b828cdbfda9dd2b5901b7cb8081e44389a99f782cef4918e886a6.jpg)

Fig. A.13: Examples where our method and Grounding DINO disagree on the predicted class.  
![](images/751c3e6a11188a86cd0a9e4abec34fc0c23eb32601b09ecd53aa1b839e195d1a.jpg)

![](images/da3ae4caef58c852d0dd982506434d3c91c99b43f7e7a02b9809e27a9f2ed922.jpg)

![](images/ed515a48bbc8aec18a09eacfa3e8fb4ed398463f56fb0f7101354423c8fc9b6a.jpg)

(a)  
![](images/e61e3af7d9facfedd86e6d16fcf34cc6520ddf4236df9d3ab323163da76bd211.jpg)

![](images/750975d032e79774c792537f974ff71e629722962be092a730d94b3e5633ff20.jpg)  
(c)

(b)  
![](images/0e0b42464d82b91190c0ce2977136c30b4abeab4f8f7a347f15a8048f669e880.jpg)

![](images/127004ad5491a9d63e1648b2c19be3f302f011b04eec43e36f3c3ac8f3d504cf.jpg)  
(d)

![](images/eb8d7d2d22277902eb6e622846b4df66b06c24c091085a8f763639aa39051a7e.jpg)  
Fig. A.14: Cases where the soft ground truth itself is not fully confident and neither are our method and Grounding DINO.

![](images/449c1ee2a82b98632f1857108e6ee298e82e8ce484827bd59892e264ba18b5be.jpg)

![](images/e0a21ea34fba52f8e6b5b1b3364b8d53a196996b914a15a63f294da284f5391e.jpg)  
(a) Pascal VOC

![](images/32324ba09c1197935d48fcc74406f91f79b05afee71e8172d8f687eaf61a7b9a.jpg)

![](images/08b71421175f01b14e3ef79d2b833b938d3ce602c16ca8f112cfe19072d79ce5.jpg)

![](images/16bc6fddf74375539a77095da3d2155f4971a28e1819736e341906a660fbb493.jpg)

![](images/241238742cc6ab1f803ba50f0c7a8842f4cc6725a8c86c62df3d3079da442c08.jpg)

![](images/5aa533002f6adf87f1e70227252bebed307a00797b49fa6010ff3fec2ccf214a.jpg)

(b) Cityscapes  
(d) Pascal VOC  
![](images/ff9d3974d77c6e81724a213cc0e2e1e5ebd0f29a3d211fe501f2e6deb7b65cbb.jpg)

![](images/851f683bb6bad8f1258610054f727e8baf0e9c5203ffa8230e2dceae4a21ad77.jpg)

![](images/f0076137455550de1d6cea68b1d854176b730249d8fd7a748c8085acb050bdc3.jpg)  
(e) Cityscapes

(c) KITTI  
![](images/83da10362bc34eecb6cb5c1b1731b3e58d2c5a7641bf7191aab1111c818d6007.jpg)

![](images/fcb8c9fba93ce2b36b0303fca8a0135874a725305c51ae9f4c0aa03e35a7fdcd.jpg)  
(f) KITTI

Fig. A.15: Examples on Pascal VOC, Cityscapes and KITTI where our method and Grounding DINO agree on the predicted class.

![](images/891319fb884612b8efca9e0fa5e17ddc986e373a6c469b08295db064262832e7.jpg)

![](images/96d3bce071e4276e15af004b48ee577174c840ca71fe6d552ac32123c2f19582.jpg)  
(a) Pascal VOC

![](images/7cdeb6837fbc51614af8cd7fb09fbcc346cdb27d8fac2d111d9e520a6b1bb51e.jpg)

![](images/7a483fba5f916cefbbab61c3967d41e9184d32421673f8187c8b534a95702a18.jpg)

![](images/b1f7e474075e3e4aa2a11c36d1505fd4413ab56133c3ad0ad5898f62640be16c.jpg)

![](images/4ea75bdfa7f57da7dedee5de05170883e80d5c923f76ca47f04a8d1ac8967aea.jpg)

![](images/5c516c6f1102106699cd5fb8cb3a3815886e1e181cec4d43e6edf3223f6b69d9.jpg)

![](images/5aa12828a9a7b77f39ee6e687509db1ad36fd0afcb7e5118108547da6cd4f112.jpg)  
(d) Pascal VOC

(b) Cityscapes  
![](images/a6fa4b66d71565aa9bff9db67de56e854673975d4565b99beb66587654a721f1.jpg)

![](images/ecb37eebb6559198d9ed2c3fa8c4f6d55550b735a6a42cd720cac5c92be0ed63.jpg)  
(e) Cityscapes

(c) KITTI  
![](images/da9aa3490d9ed15b88a22857587e4010512f7c285f6c0a56372e928f57bd6127.jpg)

![](images/3b4f502a7996f359118dea61e75a6ff2838493452323d1bbfe522f45c511d490.jpg)  
(f) KITTI

Fig. A.16: Examples on Pascal VOC, Cityscapes and KITTI where our method and Grounding DINO do not agree on the predicted class.

![](images/1b6398af0ab17c65e10d1133593b5186fecd871e082a6d1e2a475664485cea80.jpg)

![](images/ab105df25901287f5ae7382f27df18269e1bb449febc85dd77bf390aed3d0276.jpg)  
(a) Pascal VOC

![](images/cd8126f6073c04b3a0bd0fd83c0aad165161cdd7b7a3bb2a2e4e32b1316ddd00.jpg)

![](images/d5e2639210a240f50fd3460050015627b5e89cb840622971eb33d01d73022092.jpg)  
(b) Cityscapes

![](images/a50806293eabbde8068a9c362018e6b2b24be2bae7fcd55f13e5dd2c5f17e97d.jpg)

![](images/bf5c719564d5f966513f622caf04d962ffabe4d40392d9c6f357e82f3a149f4b.jpg)

![](images/8c8dadb5f60fdbe6c77e421ddbf92775d029b05ad9d48de45ae1c4130ec87222.jpg)

![](images/d2975f51f90b13452bc4f992525afd071678645f46ed4d6acb3c3e3f20af04ce.jpg)  
(d) Pascal VOC

![](images/3981adace6fee40f237c7e3c4ba638dc3dd95c983078a372f4a3ca08e6d841a7.jpg)

![](images/794c11759ff60153d91a4fb1bed82550e1745486e7bd23d993431d82a464ac23.jpg)

(c) KITTI  
(e) Cityscapes  
![](images/a5c869802d598458d7ab26672cc5f7ea0f243047aa56e60420f75e6dd076d7cf.jpg)

![](images/65868411befd6b8c66535070f6cfce783ed6c4535f5ecf498984b4b90e7dd11e.jpg)  
(f) KITTI  
Fig. A.17: Uncertain ground truth examples for Pascal VOC, Cityscapes and KITTI.