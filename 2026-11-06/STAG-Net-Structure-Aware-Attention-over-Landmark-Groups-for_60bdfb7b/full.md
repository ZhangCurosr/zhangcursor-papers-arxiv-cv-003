# STAG-Net: Structure-Aware Attention over Landmark Groups for Lightweight Head Pose Estimation

ASIRI GAWESHA LINDAMULAGE<sup>1</sup>, SANKA MOHOTTALA<sup>2</sup>, DHARSHANA KASTHURIRATHNA<sup>3</sup> Member, IEEE, AND NUWAN KODAGODA<sup>4</sup>

<sup>1</sup>Faculty of Engineering, Sri Lanka Institute of Information Technology, Sri Lanka (e-mail: asiri.l@sliit.lk) <sup>2</sup>Faculty of Engineering, University of Sri Jayewardenepura, Sri Lanka (e-mail: sanka@sjp.ac.lk) <sup>3</sup>University of Colombo School of Computing, Colombo 07, Sri Lanka (e-mail: dka@ucsc.cmb.ac.lk)   
<sup>4</sup>Faculty of Computing, Sri Lanka Institute of Information Technology, Sri Lanka (e-mail: nuwan.k@sliit.lk)

ABSTRACT Head pose estimation is important for human computer interaction and driver monitoring, but many image based methods remain too costly for resource constrained deployment. We propose STAG-Net, a structure-aware attention over landmark groups for lightweight head Pose estimation that estimates head pose from facial landmarks using movement aware selection, spatial clustering, compact DeepSet encoders, self attention, and a global geometry descriptor. We develop variants using FaceMesh and Face Alignment Network landmarks and evaluate them under cross dataset and subject independent protocols on 300W-LP, AFLW2000, and BIWI. STAG-Net achieves the lowest mean absolute error among all compared landmark based methods in every evaluated setting. On BIWI frames with valid detections, the FAN variant also records the lowest overall error among all compared methods, reaching 3.51<sup>◦</sup> under cross dataset evaluation and 2.39<sup>◦</sup> under subject independent evaluation. The regressors use only 0.012 to 0.043 million parameters and 24.1 to 83.6 thousand FLOPs, about three orders of magnitude fewer FLOPs than FSA-Net. Despite including landmark extraction and regression, the FaceMesh pipeline reaches 322.9 frames per second on a Tesla T4 GPU, nearly twice the 164.5 frames per second of the fastest measured image based baseline, and 111.0 frames per second on CPU. After normalization by dataset size, STAG-Net processe training samples seven to eight times faster than FSA-Net and more than one hundred times faster than HopeNet and TokenHPE, while complete training finishes within 28 minutes. These results show that STAG-Net combines leading landmark based accuracy with substantially lower training and inference cost.

INDEX TERMS Head pose estimation, facial landmarks, TinyML, lightweight neural networks, Deep Sets, self-attention, edge AI

## 1 Introduction

## 1.1 Background

Head pose estimation (HPE) predicts the orientation of a human head, usually in terms of yaw, pitch, and roll, from an image or a geometric facial representation. It supports applications such as human computer interaction, driver monitoring, augmented and virtual reality, and behavioral analysis [1, 2]. Many accurate HPE systems rely on convolutional or transformer based processing of high resolution facial images. This dense visual processing increases memory use and computational cost, which limits deployment on low power edge devices.

Facial landmarks offer a more compact alternative because they reduce the input to a small set of geometric coordinates while preserving the facial structure needed for pose estimation [3]. A compact landmark based regressor can also support local and privacy preserving inference without sending images to a cloud service. These properties are attractive for wearable systems, IoT devices, and TinyML applications [4]. However, the complete deployment cost also depends on the landmark detector, so both the regression model and the combined pipeline must be evaluated.

Existing landmark based methods do not fully resolve this challenge. Many represent landmarks as a fixed order coordinate vector and apply a simple regressor, which limits explicit modeling of structural relationships [5, 6]. Graph based methods model interactions among facial points more directly [7–9], but lightweight landmark native architectures that capture both local relationships within facial regions and global relationships across regions remain limited [10, 11]. This motivates a compact structure aware model that retains the efficiency of landmark inputs while improving relational modeling and reporting both regressor level and combined pipeline costs.

To address these limitations, we propose STAG-Net (Structure-Aware Attention over Landmark Groups for Lightweight Head Pose Estimation), a landmark based lightweight architecture for head pose estimation. STAG-Net partitions facial landmarks into spatially coherent clusters discovered automatically through connectivity constrained clustering, and applies a movement aware ranking strategy that retains only the most pose informative landmarks from each region. Each cluster is encoded as an unordered set by a compact DeepSet module, and a multihead attention mechanism models dependencies across clusters, allowing the network to capture both local and global structural information. In this way, STAG-Net overcomes the limitations of conventional landmark based methods that rely on fixed order vector representations or simple global aggregation.

Efficiency is treated as a primary design objective. The pose regressor uses permutation invariant set encoding and compact architectural components to balance prediction accuracy with model size and computation. We therefore evaluate not only the regressor, but also the landmark detector and regressor pipeline. This distinction is important because the compact regression stage is compatible with microcontroller class resource budgets, whereas the feasibility of a fully embedded system still depends on the upstream landmark detector and on device implementation.

The main contributions of this work are summarized as follows:

• We propose STAG-Net, a compact landmark based head pose estimation architecture that combines automatically discovered landmark clusters, permutation invariant encoding within each cluster, attention across clusters, and a global geometry descriptor.

• We introduce a movement aware grouping and selection strategy that retains pose sensitive landmarks from every facial region. Through a design analysis, we examine the effects of the retained landmark percentage, connectivity graph construction, architectural hyperparameters, and learning rate scheduling strategies on estimation accuracy.

• Across the three reported evaluation settings, STAG-Net achieves the lowest reported MAE among the compared landmark based methods. On BIWI frames with valid landmark detections, it also obtains the lowest reported overall MAE among all compared methods under both protocols. We further evaluate model size, regression FLOPs, landmark and regression pipeline throughput, and training cost, while using Pareto analysis to study the joint relationship between accuracy and model size.

• We analyze the operating ranges of the two landmark detectors and show that extreme pose coverage is mainly limited by landmark detection rather than pose regression. We also identify 36 physically implausible AFLW2000 annotations and report the cleaned angle ranges used in our experiments.

The remainder of this paper follows the same progression. Section 2 reviews image based, landmark based, and efficient head pose estimation. Section 3 presents the complete STAG-Net method, from landmark preprocessing and movement aware selection to grouped set encoding, attention, geometry fusion, and pose regression. Section 4 defines the datasets, protocols, metrics, and implementation details. Section 5 evaluates accuracy, design choices, detector coverage, and computational performance. Section 6 concludes the paper and discusses the remaining deployment limitations.

## 2 Related Work

Head pose estimation has been studied with a range of sensing modalities, including single camera RGB images, depth sensors, multiple camera systems, and auxiliary sensing setups such as methods assisted by IMUs or infrared sensors [12, 13]. Among these, single camera RGB input remains the most practical setting for low cost and widely deployable systems, since additional sensors or camera setups increase hardware complexity and are less aligned with microcontroller class edge deployment. In this context, prior work is most relevant along two lines: RGB image based methods, which estimate pose directly from a single image and often achieve strong accuracy, and landmark based methods, which operate on sparse facial keypoints and are more suitable for efficient inference. For TinyML, the main challenge is not only predictive accuracy, but also low memory use, low computational cost, and compact input representations.

## 2.1 RGB Image Based Methods

Single camera RGB image based head pose estimation is dominated by CNN and transformer models. Early methods such as Biternion Nets [14], HopeNet [2], and FSA-Net [15] showed that pose can be predicted directly from face images without explicit geometric fitting. Later work improved robustness through stronger backbones and better output representations, including WHENet [16], 6DRepNet [17], HeadPosr [18], and TokenHPE [19]. Most of these methods predict Euler angles directly, while some recent studies use quaternions, vector based rotation cues, or continuous 6D rotation representations to improve robustness for large or unconstrained poses [20–23]. Although these methods benefit from rich visual information and often achieve strong accuracy, they still rely on dense image processing and are generally less suitable for strict TinyML constraints [24, 25].

## 2.2 Landmark Based Methods

Landmark based methods estimate pose from sparse 2D or 3D facial keypoints rather than raw pixels. Classical methods used geometric formulations such as point correspondence and perspective n point style fitting [26], while later methods used learned regressors on top of landmark coordinates [5, 6]. Geometry based approaches such as [27] further showed that reasonable pose estimates can be obtained from only a few facial points at very low cost. This makes landmark based estimation especially attractive for embedded deployment.

However, many landmark methods represent facial points as fixed order vectors and use simple regression models, which limits their ability to capture structural relations among landmarks [5, 6]. More recent graph based methods, including EVA-GCN [7], OsGG-Net [8], and [9], address this by explicitly modeling interactions among facial points. Set based approaches are still rarer. Deep Sets [28], PointNet [29], Point-Net++ [30], and Set Transformer [31] provide the main permutation invariant learning tools, but direct application to head pose remains limited. DS-HPE [10] and GADS [11] are notable examples. Even so, lightweight landmark native models that jointly capture local facial structure and cross region dependencies remain scarce.

![](images/1934eff1c7a2079f27148a3eee2e5cc954617a437fa848ef0e2e6fbecfa82b6a.jpg)  
Figure 1: Overview of the proposed head pose estimation framework. Facial landmarks are first extracted from a cropped facia image and subsequently normalized with respect to the nose point by removing the absolute nose coordinates. The remaining landmarks are then grouped into spatially coherent clusters, ranked according to their motion characteristics, and filtered using a top N% landmark selection strategy. The selected subset of landmarks is finally processed by STAG-Net to estimate the yaw, pitch, and roll angles of the head.

## 2.3 Efficient Models and TinyML Approaches

A parallel line of work has focused on compact and deployable models through lightweight architectures, pruning, and quantization. MobileNet [32] and EfficientNet [33] are standard compact CNN families, and head pose models such as WHENet [16], LwPosr [34], and [35] adopt similar efficiency goals. At the deployment level, TensorFlow Lite Micro [24], CMSIS-NN [36], MCUNet [25], and MCUNetV2 [37] show that TinyML feasibility depends not only on parameter count, but also on SRAM, flash, and operator support. Quantization and pruning methods further improve deployability [38–40]. However, most efficient head pose studies still compress RGB pipelines rather than starting from compact landmark inputs, leaving room for structure aware landmark models that preserve the efficiency of landmark representations while improving relational modeling for real time TinyML deployment. These limitations motivate the grouped landmark representation introduced in the following section.

## 3 Our Method

This section presents STAG-Net and follows the data flow shown in Fig. 1. Facial landmarks are first extracted from a cropped face image and normalized to reduce translation and scale variation. The normalized landmarks are then grouped into spatially coherent regions, after which movement statistics are used to retain the most pose sensitive landmarks from each region. STAG-Net processes every group as an unordered set through an independent DeepSet encoder with a common architecture. The resulting group tokens interact through self attention and are combined with a compact geometry descriptor before the network predicts yaw, pitch, and roll. The method extends the DeepSet [28] representation used in DS-HPE [10] by introducing automatic grouping, movement aware selection, attention across groups, and explicit global geometry. This sequence connects the compact landmark input directly to the accuracy and efficiency objectives established in the introduction.

## 3.1 Problem Formulation

Head pose estimation aims to predict the 3D orientation of a human head from visual or geometric input. In this work, we focus on the estimation of head pose based on landmarks, where the input is a set of detected facial landmarks. Formally, let $X = \{ x _ { 1 } , \ldots , x _ { N } \}$ , where each landmark $x _ { i } \in \mathbb { R } ^ { 3 }$ represents the 3D spatial coordinates of a facial keypoint. The goal is to learn a mapping

$$
\boldsymbol { f } : \mathbb { R } ^ { N \times 3 }  \mathbb { R } ^ { 3 }\tag{1}
$$

that predicts the head pose in terms of yaw, pitch, and roll angles.

A key property of landmark data is that it naturally forms an unordered set, meaning that the prediction should be invariant to permutations of the input ordering. However, many existing approaches treat landmarks as fixed order vectors, which limits their ability to generalize and capture structural relationships.

In this work, we adopt the Euler angle representation for head pose due to its simplicity and interpretability, as well as its widespread use in prior literature. While alternative representations such as quaternions and continuous rotation matrices can provide improved numerical stability, Euler angles remain suitable for lightweight regression tasks and enable direct comparison with existing methods.

Given these considerations, the objective is to design a function $f ( \cdot )$ that is both permutation invariant and computationally efficient, while effectively capturing the structural relationships among facial landmarks for accurate head pose estimation.

![](images/ea9b88dd74927023004ef9624bbb569682020f6ed9426aac35e95c09470bdd80.jpg)

Figure 2: Visualization of 68 3D facial landmarks extracted using the Face Alignment Network (FAN), along with the corresponding Euler angles.  
![](images/846081f80892e034f9cec63b55fd744417bad4adc21e4daee1450ea5316a884d.jpg)  
Figure 3: Visualization of 478 3D facial landmarks extracted using FaceMesh, along with the corresponding Euler angles.

![](images/55232a3bf433c81af72c670d389214a24dd2db51d0ba0535718631563d1c4a91.jpg)  
Figure 4: Movement heatmap for landmarks detected by FaceMesh.

In the following subsections, we describe the proposed STAG-Net pipeline in detail, including input representation, preprocessing, and the architectural components of the model.

![](images/67db56a18d989516bb77f1136400713aed16dfb508832bac0644c0c59c9cb298.jpg)  
Figure 5: Movement heatmap for landmarks detected by FAN.

## 3.2 Input Representation and Preprocessing

Facial landmarks used in this work are obtained using two widely adopted landmark detectors, namely the Face Alignment Network (FAN) [3] and FaceMesh [41], which is distributed as part of Google’s MediaPipe framework. The landmarks are illustrated in the first stage of Fig. 1. These detectors provide reliable 3D landmark estimates under varying pose and lighting conditions, while maintaining a balance between accuracy and

computational efficiency.

Specifically, FAN produces 68 3D facial landmarks, whereas FaceMesh produces 478 3D landmarks. In this work, we utilize the 3D coordinates $( x , y , z )$ extracted from these detectors, enabling the model to capture richer geometric information than purely 2D representations. Figs. 2 and 3 present the landmark distributions generated by the two detectors on randomly selected face images from the three datasets, along with their corresponding ground truth yaw, pitch, and roll annotations.

The detected landmarks are represented as a set $X =$ $\{ x _ { 1 } , \ldots , x _ { N } \}$ , where each $x _ { i } \in \mathbb { R } ^ { 3 }$ . Since the ordering of landmarks does not carry semantic meaning for the learning process, the input is treated as an unordered set, motivating the use of permutation invariant set modeling [28]. To enhance robustness and support effective learning, we apply a normalization step to the input landmarks. Specifically, the landmarks are centered with respect to a reference point, chosen as the nose landmark (i.e., the 30th landmark for FAN and the 0th landmark for FaceMesh). This procedure removes global translation effects and ensures that the model concentrates on the relative geometric configuration.

Following centering, each coordinate dimension is scaled by its range within the sample to reduce variation caused by face size and camera distance. For coordinate dimension $d \in$ $\{ x , y , z \}$ , the normalized value is

$$
\tilde { x } _ { i , d } = \frac { x _ { i , d } - x _ { \mathrm { r e f } , d } } { \operatorname* { m a x } _ { j } x _ { j , d } - \operatorname* { m i n } _ { j } x _ { j , d } + \varepsilon } ,\tag{2}
$$

where $x _ { \mathrm { r e f } , d }$ is the reference coordinate and ε prevents division by zero.

Finally, the reference landmark is removed from the input representation, resulting in 67 processed landmarks for FAN and 477 processed landmarks for FaceMesh. Stage 2 of Fig. 1 illustrates the normalized landmark distributions for both landmark detectors over their corresponding mean face shapes. The nose tip landmark, which serves as the reference point for centering, has been removed from both landmark configurations and is therefore not shown in the visualizations.

## 3.3 Movement Aware Landmark Grouping and Selection

Beyond landmark normalization, the proposed framework performs a movement aware landmark grouping and selection procedure to identify compact yet informative landmark subsets. The procedure consists of three main stages: (i) movement analysis, (ii) landmark clustering, and (iii) cluster wise landmark selection.

In the first stage, landmark movement statistics are computed from the normalized landmarks to quantify the relative sensitivity of each landmark to head pose variations. Let the normalized coordinates of landmark i under pose sample j be denoted by ${ \bf p } _ { i j }$ . The average movement magnitude of landmark i is computed as

$$
m _ { i } = \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \left. \left. { \bf p } _ { i j } - \bar { \bf p } _ { i } \right. \right. _ { 2 } ,\tag{3}
$$

![](images/79554b03966513e0751e1509e837e65340e94e308b9bdb4c26640af5c9553b1e.jpg)  
Figure 6: Spatial landmark clusters obtained from FAN landmarks.

![](images/e6cb03b9a312f1b2c82df3c72c5a04ade600b524a02e948c84990da61fe792c3.jpg)  
Figure 7: Average movement magnitude within each landmark cluster.

where M denotes the number of pose samples and $\bar { \bf p } _ { i }$ is the mean location of landmark i across all samples. Figs. 5 and 4 visualize the resulting landmark movement heatmaps. It can be observed that landmark motion is not uniformly distributed across the face, with facial contour and boundary landmarks generally exhibiting larger pose related displacements than central facial landmarks. These heatmaps provide a qualitative indication of which landmarks are likely to contain richer pose related information, motivating the subsequent grouping and movement based landmark selection procedure. These movement statistics are computed once on the training split only; the resulting cluster assignments and selected landmark indices are then fixed and reused unchanged at test time, avoiding any test set leakage.

![](images/8392b9700f8e6a57f1a2d67579f862bde5b37df2617956e8fa5490bcc1deb1cf.jpg)  
Figure 8: Landmarks retained from each cluster using the top N% selection rule.

In the second stage, the training set mean positions of the normalized landmarks are partitioned into spatially coherent facial regions using connectivity constrained agglomerative hierarchical clustering. The spatial representation used for clustering is

$$
{ \bf X } _ { x y } = \{ \bar { \bf p } _ { 1 } ^ { x y } , \bar { \bf p } _ { 2 } ^ { x y } , \dots , \bar { \bf p } _ { N } ^ { x y } \} ,\tag{4}
$$

where N denotes the number of facial landmarks and $\bar {  { \mathbf { p } } } _ { i } ^ { x y } \in \mathbb { R } ^ { 2 }$ is the frontal (x,y) projection of the mean normalized position of landmark i over the training split. The clustering algorithm groups neighboring landmarks while preserving the underlying facial topology. The optimal number of clusters is determined using silhouette analysis, enabling the grouping strategy to adapt automatically to different facial landmark detectors without requiring manually defined facial regions. Fig. 6 illustrates the resulting landmark clusters obtained using the FAN landmark detector. The effect of varying the number of neighbors in the connectivity graph is investigated in the design analysis.

Following clustering, the movement characteristics of each landmark group are analyzed by computing the average movement magnitude within every cluster,

$$
\mu _ { k } = \frac { 1 } { | C _ { k } | } \sum _ { i \in C _ { k } } m _ { i } ,\tag{5}
$$

where $C _ { k }$ denotes the set of landmarks belonging to cluster k. The corresponding cluster wise movement statistics are presented in Fig. 7. The results show that different facial regions contribute unequally to head pose estimation, with certain clusters exhibiting substantially larger movement magnitudes than others. This observation validates the use of cluster wise analysis for identifying informative facial regions while preserving structural locality.

Finally, landmarks within each cluster are ranked according to their movement magnitude. For a cluster containing |C<sub>k</sub>| landmarks, the number of retained landmarks is determined as

$$
n _ { k } = \left\lceil \frac { N _ { s } } { 1 0 0 } | C _ { k } | \right\rceil ,\tag{6}
$$

where $N _ { s }$ denotes the selected percentage of landmarks. The indices of the $n _ { k }$ landmarks with the highest movement magnitudes are retained,

$$
S _ { k } = { \mathrm { T o p I n d i c e s } } _ { n _ { k } } \left( \{ m _ { i } \} _ { i \in C _ { k } } \right) ,\tag{7}
$$

and the selected landmark group is $\widetilde { C } _ { k } = \{ x _ { i } \mid i \in S _ { k } \}$ . This preserves informative landmarks from every facial region while reducing the input dimensionality.

Fig. 8 highlights the landmarks retained within each discovered cluster after applying the proposed movement based selection strategy. It can be observed that informative landmarks are distributed across all facial regions rather than being concentrated within a single area, demonstrating that the proposed selection method preserves the structural diversity of the face while eliminating less informative landmarks.

The complete landmark grouping and selection procedure is illustrated in Fig. 1. For both FAN and FaceMesh, the figure presents the normalized landmark configuration, the automatically discovered landmark clusters, and the final subset of selected landmarks obtained through movement based ranking. The resulting landmark groups form the input to the proposed STAG-Net architecture.

## 3.4 STAG-Net Architecture

## 3.4.1 STAG-Net Overview

The overall architecture of the proposed Structure Aware TinyML Attention over Groups Network (STAG-Net) is illustrated in Fig. 9. Following the movement aware landmark grouping and selection procedure described in the previous subsection, the selected landmark groups are provided as the input to STAG-Net. Unlike conventional landmark based methods that process all facial landmarks jointly, the proposed framework models each landmark group independently while explicitly learning interactions between different facial regions.

Each landmark group is first encoded by a dedicated DeepSet module into a compact token representation. All modules share the same architecture and hyperparameters but maintain independent weights, keeping the design lightweight while allowing each module to specialize to the geometry of its facial region. The resulting group tokens are subsequently processed using an inter group multihead self attention module, enabling the network to capture dependencies between different facial regions that cannot be modeled by individual landmark groups alone.

In parallel, a compact geometry branch extracts lightweight statistical descriptors from the spatial arrangement of the landmark groups, providing complementary global geometric information. The attention features and geometry descriptors are fused and passed through a shared decoder trunk followed by lightweight per angle regression heads to predict the yaw, pitch, and roll angles.

![](images/5ae28906aa1e63e3139edf626c49a159eff2e33abf3af19d7acbbfcd32fd7bc0.jpg)  
(a) STAG-Net system architecture.

![](images/620f1505cd73cae1b706df70a7657b5963550ebeace111cfeddf5f041c98edb7.jpg)  
(b) Per cluster DeepSet module.  
Figure 9: Overview of the proposed STAG-Net architecture (a) and the per cluster DeepSet module used for cluster tokenization (b). All DeepSet modules share the same architecture and hyperparameters but maintain independent weights.

By combining permutation invariant set encoding, inter group attention, and compact geometric descriptors, STAG-Net effectively captures both local and global facial structure while maintaining the low computational complexity required for TinyML deployment.

The following subsections describe each component of STAG-Net in detail.

## 3.4.2 Per Group DeepSet Token Encoders

Following the movement aware grouping and selection procedure, the input to STAG-Net consists of a set of K landmark groups,

$$
{ \mathcal { G } } = \{ C _ { 1 } , C _ { 2 } , \ldots , C _ { K } \} ,\tag{8}
$$

where each group $C _ { k }$ contains a subset of normalized facial landmarks. Since the ordering of landmarks within a group carries no semantic meaning, each landmark group is modeled as an unordered set. To obtain a compact representation for every group while preserving permutation invariance, STAG-Net employs per group DeepSet encoders with a shared architecture, illustrated in Fig. 9(b).

DeepSets [28] provide a principled framework for learning from unordered sets by decomposing the mapping into an element wise transformation followed by a permutation invariant aggregation. A general DeepSet function can be expressed as

$$
f ( X ) = \rho \left( \operatorname { A g g } \left( \left\{ \phi \left( x _ { i } \right) \mid x _ { i } \in X \right\} \right) \right) ,\tag{9}
$$

where $\phi ( \cdot )$ denotes a shared element wise encoder, $\operatorname { A g g } ( { \mathord { \cdot } } )$ is a permutation invariant aggregation operator, and $\rho ( \cdot )$ represents a decoder that maps the aggregated representation to the desired feature space.

In the proposed architecture, each landmark group is processed by its own DeepSet module. All K modules share an identical architecture and identical hyperparameters, but maintain independent weights, allowing each module to specialize to the geometry of its facial region; consequently, the total parameter count scales linearly with the number of clusters $K .$ For landmark group $k ,$ the corresponding token representation is obtained as

$$
t _ { k } = \rho \left( \operatorname { A g g } \left( \left\{ \phi \left( x _ { i } \right) \vert x _ { i } \in C _ { k } \right\} \right) \right) ,\tag{10}
$$

where $t _ { k } \in \mathbb { R } ^ { F }$ denotes the feature token corresponding to landmark group $C _ { k } .$ , and $F$ is the token embedding dimension.

Within each group, the encoder $\phi ( \cdot )$ first standardizes the raw coordinates with an input batch normalization layer and then projects every landmark independently into a higher dimensional feature space through a linear layer followed by batch normalization, ReLU activation, and dropout. Because the encoder parameters are shared across all landmarks within a group, each DeepSet module remains extremely small, and the per group specialization is obtained without sacrificing the compactness required for TinyML deployment.

To aggregate landmark features within each group, a permutation invariant mean and max pooling strategy is adopted. Let

$$
E _ { k } = \left\{ \phi ( x _ { i } ) \mid x _ { i } \in C _ { k } \right\}\tag{11}
$$

denote the encoded landmark features of group k. The aggregated representation is computed as

$$
\operatorname { A g g } ( E _ { k } ) = \operatorname { C o n c a t } \left( \operatorname { M e a n } ( E _ { k } ) , \operatorname { M a x } ( E _ { k } ) \right) .\tag{12}
$$

The mean pooling operation summarizes the overall geometric structure of the landmark group, whereas max pooling emphasizes the most discriminative local features. Their combination provides a richer representation than either pooling strategy alone while remaining computationally efficient.

Finally, the aggregated feature vector is processed by the decoder $\rho ( \cdot )$ , implemented as a single linear layer, to generate the final group token. Repeating this procedure for every landmark group produces the token set

$$
T = [ t _ { 1 } , t _ { 2 } , \dots , t _ { K } ] ,\tag{13}
$$

which forms the input to the subsequent inter group self attention module.

Unlike conventional landmark based approaches that directly process all facial landmarks jointly, the proposed tokenization strategy first summarizes each facial region independently before modeling interactions between regions. This hierarchical representation improves computational efficiency while preserving the structural information required for accurate head pose estimation.

## 3.4.3 Inter Group Self Attention

Although the per group DeepSet encoders effectively capture local geometric relationships within each landmark group, head pose estimation depends on the coordinated movement of multiple facial regions. For example, pose related variations observed around the jawline, nose, and eyes are often correlated.

To model these global structural dependencies, STAG-Net employs an inter group multihead self attention module that enables information exchange between the group level tokens.

Let

$$
T = [ t _ { 1 } , t _ { 2 } , \ldots , t _ { K } ] \in \mathbb { R } ^ { K \times F } ,\tag{14}
$$

denote the token matrix produced by the per group DeepSet encoders, where each token corresponds to one landmark group and $F$ is the token embedding dimension.

Before computing attention, layer normalization is applied to improve optimization stability,

$$
{ \hat { T } } = \operatorname { L a y e r N o r m } ( T ) .\tag{15}
$$

The normalized tokens are projected into query, key, and value representations,

$$
{ \bf Q } = \hat { T } W _ { Q } , \qquad { \bf K } = \hat { T } W _ { K } , \qquad { \bf V } = \hat { T } W _ { V } ,\tag{16}
$$

where $W _ { Q } , W _ { K }$ , and $W _ { V }$ are learnable projection matrices. The attention weights are computed using scaled dot product attention,

$$
A = \mathrm { S o f t m a x } \left( \frac { \mathbf { Q } \mathbf { K } ^ { \top } } { \sqrt { d } } \right) ,\tag{17}
$$

where d denotes the dimensionality of each attention head. The refined token representations are then obtained as

$$
T _ { \mathrm { a t t } } = A \mathbf { V } .\tag{18}
$$

To facilitate optimization and preserve the original group representations, a residual connection followed by layer normalization is employed,

$$
{ \cal T } ^ { \prime } = \mathrm { L a y e r N o r m } \left( T + T _ { \mathrm { a t t } } \right) .\tag{19}
$$

Multihead attention is adopted to enable the model to jointly capture multiple types of relationships between different facial regions. Each attention head learns complementary interactions among landmark groups, allowing STAG-Net to exploit both local and global structural dependencies that cannot be captured by independently processing each landmark group. Since each head operates on a partition of the token embedding with dimension $d = F / n _ { h }$ , the number of heads $n _ { h }$ controls the granularity of these interactions without affecting the parameter count of the attention module.

During training, token dropout is applied before the attention operation to improve robustness and reduce overfitting. Rather than removing individual landmark features, token dropout randomly suppresses a subset of landmark group tokens while ensuring that at least one token remains available for attention. This encourages the model to learn robust representations that do not rely excessively on any single facial region.

To obtain a fixed length representation that is invariant to group ordering, the refined tokens are aggregated using permutation invariant mean and max pooling,

$$
f _ { \mathrm { a t t } } = \mathrm { C o n c a t } \left( \mathbf { M e a n } ( T ^ { \prime } ) , \mathbf { M a x } ( T ^ { \prime } ) \right) \in \mathbb { R } ^ { 2 F } ,\tag{20}
$$

which serves as the learned structural representation of the face and is subsequently fused with the compact geometry descriptor for the final pose regression stage.

## 3.4.4 Compact Geometry Branch

While the inter group self attention module learns feature interactions between landmark groups, it does not explicitly preserve the global spatial arrangement of the facial geometry. To complement the learned token representations, STAG-Net incorporates a lightweight geometry branch that summarizes the spatial distribution of the landmark groups using compact statistical descriptors. Unlike the attention branch, which learns task specific feature representations, the geometry branch directly encodes the geometric configuration of the face, providing complementary structural information with negligible computational overhead.

For each landmark group $C _ { k }$ , the centroid is computed as

$$
c _ { k } = \frac { 1 } { \vert C _ { k } \vert } \sum _ { x _ { i } \in C _ { k } } x _ { i } ,\tag{21}
$$

where $\left| C _ { k } \right|$ denotes the number of landmarks belonging to group $k .$ The resulting centroid set

$$
\mathcal { C } = \{ c _ { 1 } , c _ { 2 } , . . . , c _ { K } \}\tag{22}
$$

provides a compact representation of the spatial layout of the grouped facial landmarks.

Three statistical descriptors are subsequently extracted from the centroid set. The centroid mean,

$$
\mu _ { c } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } c _ { k } ,\tag{23}
$$

captures the overall spatial location of the landmark groups. The centroid standard deviation,

$$
\sigma _ { c } = \sqrt { \frac { 1 } { K } \sum _ { k = 1 } ^ { K } ( c _ { k } - \mu _ { c } ) ^ { 2 } } ,\tag{24}
$$

quantifies the spatial dispersion of the groups, while the covariance matrix

$$
\Sigma _ { c } = \frac { 1 } { K - 1 } \sum _ { k = 1 } ^ { K } ( c _ { k } - \mu _ { c } ) ( c _ { k } - \mu _ { c } ) ^ { \top }\tag{25}
$$

captures the correlation between the centroid coordinates. For numerical and scale stability, the covariance matrix is normalized by its trace prior to the eigendecomposition, $\tilde { \Sigma } _ { c } =$ $\Sigma _ { c } / ( \mathrm { t r } ( \Sigma _ { c } ) + \varepsilon )$ ), where ε prevents division by zero. The eigenvalues of the normalized covariance matrix,

$$
\lambda = \mathrm { e i g } ( \tilde { \Sigma } _ { c } ) ,\tag{26}
$$

provide a compact description of the principal geometric variation of the grouped facial structure.

The three descriptors are concatenated to form the geometry feature vector

$$
g = \mathrm { C o n c a t } ( \mu _ { c } , \sigma _ { c } , \lambda ) ,\tag{27}
$$

which summarizes the global spatial configuration of the grouped facial landmarks using only a small number of features.

Compared with directly processing all landmark coordinates, the proposed geometry branch provides a compact structural representation while introducing only minimal computational cost. Consequently, it complements the learned token representations obtained from the attention branch and enhances the ability of STAG-Net to model global facial geometry without compromising its suitability for TinyML deployment.

## 3.4.5 Feature Fusion and Multioutput Pose Regression

The refined token representations obtained from the inter group self attention module capture complementary relationships between different facial regions, while the compact geometry branch explicitly encodes the global spatial arrangement of the landmark groups. To exploit the strengths of both representations, STAG-Net combines the learned attention features with the geometry descriptor through feature fusion.

Let $f _ { \mathrm { a t t } } \in \mathbb { R } ^ { 2 F }$ denote the pooled attention representation and let $\boldsymbol { g } \in \mathbb { R } ^ { 9 }$ denote the geometry descriptor. The fused feature representation is obtained by concatenating the two,

$$
z = \mathbf { C o n c a t } ( f _ { \mathrm { a t t } } , g ) \in \mathbb { R } ^ { 2 F + 9 } ,\tag{28}
$$

where z represents the final feature vector used for pose estimation.

The fused representation is subsequently processed by a shared decoder consisting of fully connected layers with layer normalization, ReLU activation, and dropout. The shared decoder learns a compact latent representation that captures information common to all three pose components while maintaining a lightweight architecture suitable for TinyML deployment.

Finally, the shared latent representation is passed to three lightweight per angle regression heads, one each for yaw, pitch, and roll; every head consists of a single hidden layer with layer normalization, ReLU activation, and dropout, followed by a linear output unit. The final pose prediction is given by

$$
\hat { \mathbf { y } } = f _ { \mathrm { r e g } } \left( f _ { \mathrm { s h a r e d } } ( z ) \right) ,\tag{29}
$$

where $f _ { \mathrm { s h a r e d } } ( \cdot )$ denotes the shared decoder, $f _ { \mathrm { r e g } } ( \cdot )$ denotes the three per angle regression heads, and

$$
\hat { \mathbf { y } } = \left[ \hat { y } _ { \mathrm { y a w } } , \hat { y } _ { \mathrm { p i t c h } } , \hat { y } _ { \mathrm { r o l l } } \right] ^ { \top }\tag{30}
$$

contains the predicted yaw, pitch, and roll angles, respectively.

Compared with employing three fully independent prediction networks, the proposed design shares the majority of the learned representation across all pose components in the common trunk, while the lightweight per angle heads retain the flexibility to specialize to each rotation axis. This significantly reduces the number of trainable parameters and computational complexity without sacrificing the ability to jointly estimate the three head pose angles. Consequently, the proposed feature fusion and shared trunk regression strategy contributes to the overall efficiency of STAG-Net, making it well suited for deployment on resource constrained TinyML platforms.

## 4 Experimental Setup

This section presents the experimental setup adopted to evaluate the proposed STAG-Net. We first describe the benchmark datasets and their characteristics, followed by the evaluation metrics and implementation details used throughout this study. The experiments are designed to provide a consistent and reproducible assessment of prediction accuracy and computational efficiency under the two established evaluation protocols.

## 4.1 Datasets and Evaluation Protocols

To comprehensively evaluate the proposed STAG-Net framework, experiments are conducted on three widely used head pose estimation datasets: 300W-LP, AFLW2000, and BIWI. These datasets contain substantial variations in pose, illumination, expression, and acquisition conditions, making them suitable for evaluating both robustness and generalization.

300W-LP [42] is a large scale synthetic head pose estimation dataset derived from the original 300W face alignment dataset [43]. It employs 3D face profiling and image meshing techniques to synthetically generate large out of plane pose variations. The original dataset contains 61,225 samples, which further expands to approximately 122,450 images after horizontal flipping augmentation. The dataset has become the standard training dataset for many head pose estimation methods due to its large pose diversity and scale.

AFLW2000 [44] contains the first 2,000 images from the AFLW dataset annotated with 3D facial landmarks and head pose labels. The dataset includes significant variations in illumination, facial expression, occlusion, and head orientation, making it a widely used benchmark for cross dataset evaluation.

BIWI [45] is a benchmark RGB and depth head pose estimation dataset collected in a controlled indoor environment. The dataset contains $^ { 1 5 , 6 7 8 }$ frames captured from 20 subjects across 24 video sequences. In addition to RGB images, aligned depth frames and accurate pose annotations are also provided. The dataset includes substantial head pose variations and is widely used for evaluating head pose estimation methods under subject independent settings. In this work, only single RGB frames are utilized.

Since the distribution of head pose angles is rarely discussed in the existing head pose estimation literature, we analyzed the Euler angle distributions of the three benchmark datasets used in this study. Fig. 10 presents the distributions of the yaw, pitch, and roll angles for BIWI, 300W-LP, and AFLW2000, where each histogram illustrates the number of samples corresponding to different head pose angles. These distributions reveal the pose diversity of each dataset and highlight the differences in angular coverage among the three benchmarks. Table 1 summarizes the exact minimum and maximum values of the three Euler angles for each dataset. Notably, only 300W-LP spans the full $\pm 9 0 ^ { \circ }$ yaw range, whereas BIWI covers a substantially narrower angular envelope, particularly in yaw and roll; consequently, models trained on 300W-LP under Protocol I must generalize from a wider synthetic pose distribution to the more constrained real world distributions of the evaluation datasets.

![](images/71a640039d3b08e2a43c77ac12ed4ba6721afe4d3d67e61aa0049010a3b748ec.jpg)

(a) 300W-LP  
![](images/c070875799d07c4e387f57858b984987567d15dc9af9a973a0d7cfb476d65a2f.jpg)

(b) BIWI  
![](images/74d4fa1cb80181ca8ee756657c88409251dfa44ace5176f67a81d09e1a3879ac.jpg)  
(c) AFLW2000  
Figure 10: Distribution of yaw, pitch, and roll angles for the three benchmark head pose estimation datasets.

During this analysis, we observed that the AFLW2000 dataset contains 36 samples with extremely large pose annotations that are inconsistent with realistic head rotations. Specifically, these samples exhibit pitch, yaw, and roll values ranging from $- 7 1 3 . 7 5 ^ { \circ } \mathrm { ~ t o ~ } 8 9 3 . 7 5 ^ { \circ } , - 3 5 1 . 2 3 ^ { \circ } \mathrm { ~ t o ~ } 1 8 7 . 7 9 ^ { \circ }$ , and $- 9 7 2 . 2 7 ^ { \circ }$ to 334.00<sup>◦</sup>, respectively. These values are far beyond the physically plausible range of human head motion and are likely the result of annotation errors or inconsistencies in the dataset. Consequently, these anomalous samples were removed as a data cleaning step prior to evaluation. After removal, all AFLW2000 pose angles fall within $\left[ - 8 3 . 6 7 ^ { \circ } , 8 6 . 6 3 ^ { \circ } \right]$ for yaw, $\left[ - 8 2 . 9 6 ^ { \circ } , 8 8 . 2 6 ^ { \circ } \right]$ for pitch, and $\left[ - 8 7 . 0 9 ^ { \circ } , 8 3 . 6 6 ^ { \circ } \right]$ for roll; these cleaned ranges are the ones reported in Table 1, and the histograms in Fig. 10 likewise correspond to the cleaned dataset. To the best of our knowledge, previous studies have not explicitly reported the presence of these anomalous annotations or documented their removal. We believe this observation provides useful guidance for future researchers using the AFLW2000 dataset, as these inconsistencies may influence both model training and evaluation if left unaddressed.

Table 1: Head pose angle ranges of the datasets used in this study. For AFLW2000, the reported ranges correspond to the cleaned dataset after removal of the 36 anomalous samples described in the text.
<table><tr><td>Dataset</td><td>Yaw (°)</td><td>Pitch (°)</td><td>Roll (°)</td></tr><tr><td>BIWI</td><td>–74.89 to 77.97</td><td>-63.83 to 84.37</td><td>-64.70 to 45.02</td></tr><tr><td>300W-LP</td><td>–90.00 to 90.00</td><td>-64.57 to 61.90</td><td>-73.43 to 53.16</td></tr><tr><td>AFLW2000</td><td>-83.67 to 86.63</td><td>-82.96 to 88.26</td><td>6 -87.09 to 83.66</td></tr></table>

To make our experiments consistent, we reconstructed all three datasets from their original sources and used a common pose ordering of yaw, pitch, and roll. This step was necessary because public implementations differ in image resolution and label ordering. For example, the FSA-Net preprocessing pipeline uses $6 4 \times 6 4$ images, while landmark extraction is often performed at higher resolutions. Such differences can alter results when models are rerun. In our STAG-Net experiments, every image is resized to $2 2 4 \times 2 2 4$ before landmark extraction, and the same preprocessing and label conventions are used for both FAN and FaceMesh. Accuracy values quoted for prior methods in the comparison tables are taken from their cited publications under the corresponding protocols, while our runtime measurements use the official implementations on common hardware. For AFLW2000, our values are computed on the cleaned 1,964 sample subset, whereas values quoted from earlier publications may have been computed on the original 2,000 images. The AFLW2000 comparison should therefore be interpreted as contextual rather than as a strictly matched reevaluation of every baseline.

To follow established comparison settings, we adopt two commonly used evaluation protocols in the head pose estimation literature.

Protocol I: Cross Dataset Evaluation. Following FSA-Net [15] and related work, the model is trained on the synthetic 300W-LP dataset and evaluated on the real world AFLW2000 and BIWI datasets. This protocol evaluates the generalizability of the model across different domains and acquisition conditions. For BIWI evaluation under this protocol, no tracking or temporal smoothing techniques are used. The proposed framework performs inference independently on each RGB frame.

Protocol II: Subject Independent BIWI Evaluation. Following TokenHPE [19], BIWI is divided at the sequence level into 16 training sequences and 8 test sequences, corresponding to two thirds of the sequences for training and one third for testing. This protocol evaluates the model’s ability to generalize to the held out subjects and recording sessions defined by this split.

Although several methods evaluated under this protocol utilize additional modalities such as depth information or temporal sequences, the proposed STAG-Net framework operates solely on single RGB images and extracted 3D landmarks.

For both protocols, the STAG-Net regression error is computed on frames for which the selected landmark detector returns a valid landmark set. Frames without valid landmarks cannot be processed by a landmark based regressor. We therefore report the detector angle envelope in Table 5 and interpret MAE together with detector coverage. This distinction is particularly important at extreme poses, where the two detectors cover different subsets of the annotated range.

## 4.2 Evaluation Metrics

Performance is evaluated using the mean absolute error (MAE) of the estimated Euler angles, which is the standard evaluation metric in the head pose estimation literature. Let Θ = {yaw,pitch,roll} denote the set of pose angles, and let $y _ { i } ^ { \theta }$ and $\hat { y } _ { i } ^ { \theta }$ denote the ground truth and predicted values of angle $\theta \in \Theta$ for test sample i. The per angle MAE is computed as

$$
\mathrm { M A E } _ { \theta } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \bigl | y _ { i } ^ { \theta } - \hat { y } _ { i } ^ { \theta } \bigr | ,\tag{31}
$$

where N is the number of test samples with valid landmark detections for the detector being evaluated. The overall MAE is then obtained by averaging over the three pose angles,

$$
\mathbf { M A E } = { \frac { 1 } { | \Theta | } } \sum _ { \theta \in \Theta } \mathbf { M A E } _ { \theta } .\tag{32}
$$

Equation (31) produces the per angle columns reported in Tables 3 and 4, while Equation (32) corresponds to their final MAE column.

Model Complexity Metrics In addition to prediction accuracy, the computational efficiency of the proposed framework is evaluated using the number of parameters and Floating Point Operations (FLOPs), which are commonly used metrics in TinyML and efficient deep learning research [32, 33].

Unlike conventional RGB based head pose estimation methods, the proposed STAG-Net operates on compact landmark representations and lightweight set based processing modules. Therefore, parameter count and FLOPs are particularly relevant for quantifying the efficiency gains achieved through the proposed architecture and assessing its suitability for deployment in resource constrained environments.

The number of trainable parameters provides a measure of the memory required to store the model. For a fully connected layer with $n _ { i n }$ input units and $n _ { o u t }$ output units, the number of parameters is computed as:

$$
P = n _ { i n } \times n _ { o u t } + n _ { o u t } ,\tag{33}
$$

where the first term corresponds to the weights and the second term corresponds to the bias parameters. The total number of parameters of a model is obtained by summing the parameters across all trainable layers:

$$
P _ { \mathrm { t o t a l } } = \sum _ { l = 1 } ^ { L } P _ { l } ,\tag{34}
$$

where L denotes the total number of trainable layers.

The parameter count directly affects the flash memory requirements of embedded systems and is therefore an important consideration for TinyML deployment.

To quantify computational complexity, we report floating point operations (FLOPs), which estimate the number of arith metic operations required during inference. For a fully connected layer, the FLOPs can be approximated as:

$$
F = 2 \times n _ { i n } \times n _ { o u t } ,\tag{35}
$$

where the factor of two accounts for multiplication and accumulation operations.

The total FLOPs of a model are computed as:

$$
F _ { \mathrm { t o t a l } } = \sum _ { l = 1 } ^ { L } F _ { l } .\tag{36}
$$

FLOPs provide a hardware independent estimate of computational cost and are frequently used as a proxy for inference latency and energy consumption in resource constrained systems [46].

Together, MAE, parameter count, and FLOPs provide a comprehensive assessment of both predictive performance and deployment suitability. Since STAG-Net is primarily composed of fully connected layers and attention modules, the reported parameter counts and FLOPs closely reflect the actual memory and computational requirements of the deployed model.

Since TinyML deployment requires optimizing accuracy and resource cost jointly rather than in isolation, we additionally assess the compared methods through a Pareto dominance analysis over the joint (MAE, parameter count) objective. A method is Pareto dominated if another method achieves both a lower or equal MAE and a lower or equal parameter count, with strict improvement in at least one of the two; the methods that are not dominated form the Pareto frontier of the accuracy and efficiency tradeoff. This analysis, presented in Section 5.3, identifies which methods represent genuinely optimal operating points rather than merely favorable positions along a single axis.

## 4.3 Implementation Details

Facial landmarks are extracted using the corresponding pretrained landmark detector, and the resulting coordinates are normalized as described in Section 3.2 before they are provided to the network. The model is trained by minimizing the mean absolute error between the predicted and ground truth Euler angles using the settings summarized in Table 2. The architectural choices examined in the design analysis are reported in Section 5.2.

All experiments were conducted using the Google Colab environment. Performance comparisons were carried out on both the default Google Colab CPU runtime and the NVIDIA Tesla T4 GPU runtime. Model training and inference were evaluated on both platforms, and all reported execution times were measured under these hardware configurations. The proposed model was implemented using the PyTorch deep learning framework, and the primary training hyperparameters are summarized in Table 2.

Table 2: Implementation and training configuration.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Framework</td><td>PyTorch</td></tr><tr><td>Optimizer</td><td>Adam</td></tr><tr><td>Loss function</td><td>Mean absolute error (MAE)</td></tr><tr><td>Batch size</td><td>128</td></tr><tr><td>Training epochs</td><td>150</td></tr><tr><td>Initial learning rate</td><td>5 × 10−4</td></tr><tr><td>Learning rate scheduler</td><td>CosineAnnealingWarmRestarts</td></tr><tr><td>Initial restart period, T0</td><td>10 epochs</td></tr><tr><td>Restart period multiplier, Tmult</td><td>2</td></tr><tr><td>Weight initialization</td><td>Xavier uniform</td></tr><tr><td>Dropout rate</td><td>0.01</td></tr><tr><td>Attention dropout</td><td>0.2</td></tr><tr><td>Token dropout</td><td>0.1</td></tr></table>

Prior to training, all face images were resized to a resolution of 224 × 224 pixels, after which facial landmarks were extracted using the corresponding landmark detector as part of the preprocessing pipeline. The extracted landmark coordinates were stored and subsequently used as the direct input to STAG-Net during training, ensuring a consistent preprocessing procedure across all experiments.

During training, no additional image enhancement techniques were used. Furthermore, early stopping was not utilized. Instead, the model achieving the lowest validation loss was checkpointed throughout training, and the best performing checkpoint was used for the final evaluation. This ensured a consistent model selection strategy while allowing all experiments to be trained for the predefined 150 epochs. The preprocessing, training, and evaluation procedures are kept identical across experiments so that differences can be attributed to the model configuration rather than to changes in the data pipeline. The following section reports the resulting accuracy and computational performance.

To support reproducibility, the implementation of STAG-Net is publicly available at https://github.com/ asirigawesha/STAGNet.

## 5 Experimental Results

This section evaluates STAG-Net in the same order as the claims made in the introduction. We first compare prediction accuracy under the two protocols defined in Section 4.1. We then examine detector coverage, connectivity graph construction, landmark selection, architectural hyperparameters, and learning rate scheduling strategies. Finally, we analyze model size, regression FLOPs, landmark and regression pipeline throughput, training cost, and the joint relationship between accuracy and model size.

## 5.1 Comparison with State of the Art Methods

The proposed STAG-Net is compared with representative state of the art head pose estimation methods under the two evaluation protocols described in Section 4.1. Since both RGB based and landmark based approaches have been reported in the literature, comparisons are conducted against methods from both categories to provide a comprehensive evaluation. Furthermore, as STAG-Net can operate with different facial landmark detectors, results are reported using both FaceMesh and FAN landmarks to investigate the influence of the landmark detector on the overall estimation performance. Tables 3 and 4 group the compared methods into these two categories, together with the corresponding model size in terms of the number of trainable parameters. For prior methods, the accuracy values and parameter counts are taken directly from the cited publications. Runtime values are measured from the official implementations on common hardware without modifying the models. The preprocessing and pose label conventions in Section 4.1 are applied consistently to both STAG-Net variants. Since the STAG-Net errors are conditional on successful landmark detection, the numerical rankings should be interpreted together with the detector envelopes in Table 5.

Table 3: Comparison with state of the art methods under Protocol I, grouped into RGB image based and landmark based methods, where bold and underlined values indicate the lowest and second lowest reported numerical values per column across all methods. N/A indicates a value that could not be obtained.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Params (M)</td><td colspan="4">BIWI</td><td colspan="4">AFLW2000</td></tr><tr><td>Yaw</td><td>Pitch</td><td>Roll</td><td>MAE</td><td>Yaw</td><td>Pitch</td><td>Roll</td><td>MAE</td></tr><tr><td>RGB image based methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>KEPLER (2017) [47]</td><td>N/R</td><td>8.80</td><td>17.30</td><td>16.20</td><td>13.90</td><td>N/R</td><td>N/R</td><td>N/R</td><td>N/R</td></tr><tr><td>3DDFA (2019) [48]</td><td>N/R</td><td>36.20</td><td>12.30</td><td>8.78</td><td>19.10</td><td>5.40</td><td>8.53</td><td>8.25</td><td>7.39</td></tr><tr><td>TokenHPE (2023) [19]</td><td>86.42</td><td>4.51</td><td>3.95</td><td>2.71</td><td>3.72</td><td>5.54</td><td>3.36</td><td>4.08</td><td>4.66</td></tr><tr><td>6DRepNet (2022) [17]</td><td>39.32</td><td>3.23</td><td>5.32</td><td>2.78</td><td>3.78</td><td>3.27</td><td>4.58</td><td>2.98</td><td>3.61</td></tr><tr><td>TriNet (2021) [49]</td><td>26.00</td><td>4.11</td><td>4.76</td><td>3.05</td><td>3.97</td><td>4.04</td><td>5.78</td><td>4.20</td><td>4.67</td></tr><tr><td>Shao (2019) [50]</td><td>24.60</td><td>4.59</td><td>7.25</td><td>6.15</td><td>6.00</td><td>5.07</td><td>6.37</td><td>4.99</td><td>5.48</td></tr><tr><td>HopeNet (α=1) (2018) [51]</td><td>23.90</td><td>4.81</td><td>6.61</td><td>3.27</td><td>4.90</td><td>6.92</td><td>6.64</td><td>5.67</td><td>6.41</td></tr><tr><td>HopeNet (α=2) (2018) [51]</td><td>23.90</td><td>5.12</td><td>6.98</td><td>3.39</td><td>5.18</td><td>6.47</td><td>6.56</td><td>5.44</td><td>6.16</td></tr><tr><td>WHENet (2020) [16]</td><td>4.40</td><td>3.99</td><td>4.39</td><td>3.06</td><td>3.81</td><td>5.11</td><td>6.24</td><td>4.92</td><td>5.42</td></tr><tr><td>FSA-Caps-Fusion (2019) [15]</td><td>1.20</td><td>4.27</td><td>4.96</td><td>2.76</td><td>4.00</td><td>4.50</td><td>6.08</td><td>4.64</td><td>5.07</td></tr><tr><td>SSR-Net-MD (2018) [52]</td><td>0.20</td><td>4.49</td><td>6.31</td><td>3.61</td><td>4.65</td><td>5.14</td><td>7.09</td><td>5.89</td><td>6.01</td></tr><tr><td>LwPosr (2022) [34]</td><td>0.15</td><td>4.11</td><td>4.87</td><td>3.19</td><td>4.05</td><td>4.80</td><td>6.38</td><td>4.88</td><td>5.35</td></tr><tr><td>LwPosr-a (2022) [34]</td><td>0.15</td><td>4.41</td><td>5.11</td><td>3.24</td><td>4.25</td><td>4.44</td><td>6.06</td><td>4.35</td><td>4.95</td></tr><tr><td>Landmark based methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FAN (12 points) (2017) [3]</td><td>36.60</td><td>6.36</td><td>12.30</td><td>8.71</td><td>9.12</td><td>8.53</td><td>7.48</td><td>7.63</td><td>7.88</td></tr><tr><td>Dlib (68 points) (2014) [53]</td><td>6.24</td><td>16.80</td><td>13.80</td><td>6.19</td><td>12.20</td><td>23.10</td><td>13.60</td><td>10.50</td><td>15.80</td></tr><tr><td>EVA-GCN (2021) [7]</td><td>3.30</td><td>4.46</td><td>5.34</td><td>4.11</td><td>4.64</td><td>4.01</td><td>4.78</td><td>2.98</td><td>3.92</td></tr><tr><td>GADS (2025) [11]</td><td>0.02</td><td>3.61</td><td>5.05</td><td>3.04</td><td>3.90</td><td>3.84</td><td>7.06</td><td>5.00</td><td>5.30</td></tr><tr><td>Tourani et al.(2025) [54]</td><td>N/R</td><td>4.60</td><td>5.54</td><td>3.01</td><td>4.47</td><td>4.79</td><td>6.09</td><td>4.88</td><td>5.28</td></tr><tr><td>STAG-Net (FaceMesh)</td><td>0.043</td><td>3.63</td><td>4.96</td><td>2.23</td><td>3.61</td><td>3.28</td><td>4.84</td><td>3.05</td><td>3.72</td></tr><tr><td>STAG-Net (FAN)</td><td>0.034</td><td>2.86</td><td>5.00</td><td>2.68</td><td>3.51</td><td>4.20</td><td>6.86</td><td>5.56</td><td>5.53</td></tr></table>

Table 3 presents the results obtained under Protocol I on the BIWI and AFLW2000 datasets. On the BIWI dataset, the proposed STAG-Net using FAN landmarks achieves an overall mean absolute error (MAE) of 3.51<sup>◦</sup>, which is the lowest reported error among the compared methods while requiring only

0.034 M parameters. Moreover, STAG-Net (FAN) achieves the lowest yaw error among all compared methods, while STAG-Net (FaceMesh) obtains the lowest roll error, demonstrating the effectiveness of the proposed architecture across multiple pose components.

On the AFLW2000 dataset, STAG-Net (FaceMesh) achieves the lowest yaw error (3.28<sup>◦</sup>) and the lowest overall MAE (3.72<sup>◦</sup>) among the compared landmark based methods, while remaining competitive with the strongest RGB based approaches. Although the FAN based variant achieves superior performance on BIWI, the FaceMesh based variant produces a lower error on AFLW2000, suggesting that the choice of landmark detector can influence performance depending on the characteristics of the target dataset. These results indicate that the proposed architecture is robust across different landmark detectors while maintaining consistently competitive performance.

The results under Protocol II are summarized in Table 4. Both STAG-Net variants outperform existing landmark based approaches despite having substantially fewer trainable parameters. In particular, STAG-Net using FAN achieves the lowest overall MAE of 2.39<sup>◦</sup>, together with the lowest yaw and roll errors among all compared methods. STAG-Net using FaceMesh achieves an overall MAE of 2.57<sup>◦</sup> while requiring only 0.012 million trainable parameters. It also obtains the second lowest yaw error of 2.37<sup>◦</sup> among all compared methods and remains more accurate than previously reported landmark based approaches. Together, these results show that STAG-Net remains effective with either landmark detector while providing a favorable balance between accuracy and model size.

Table 4: Comparison with state of the art methods under Protocol II, grouped into RGB image based and landmark based methods, where bold and underlined values indicate the lowest and second lowest reported numerical values per column across all methods. N/A indicates a value that could not be obtained.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Params (M)</td><td colspan="4">BIWI</td></tr><tr><td>Yaw</td><td>Pitch</td><td>Roll</td><td>MAE</td></tr><tr><td>RGB image based methods</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Martin (2014) [55]</td><td>N/R</td><td>3.60</td><td>2.50</td><td>2.60</td><td>2.90</td></tr><tr><td>VGG16-RNN (2017) [56]</td><td>138.00</td><td>3.14</td><td>3.48</td><td>2.60</td><td>3.07</td></tr><tr><td>VGG16(2017) [56]</td><td>138.00</td><td>3.91</td><td>4.03</td><td>3.03</td><td>3.66</td></tr><tr><td>TokenHPE (2023) [19]</td><td>86.42</td><td>3.01</td><td>2.28</td><td>2.01</td><td>2.49</td></tr><tr><td>6DRepNet (2022) [17]</td><td>39.32</td><td>2.39</td><td>2.96</td><td>2.05</td><td>2.47</td></tr><tr><td>TriNet (2021) [49]</td><td>26.00</td><td>2.99</td><td>3.04</td><td>2.44</td><td>2.80</td></tr><tr><td>FSA-Caps-Fusion (2019) [15]</td><td>1.20</td><td>2.89</td><td>4.29</td><td>3.60</td><td>3.60</td></tr><tr><td>SSR-Net-MD (2018) [52]</td><td>0.20</td><td>4.24</td><td>4.35</td><td>4.19</td><td>4.16</td></tr><tr><td>LwPosr (2022) [34]</td><td>0.15</td><td>3.62</td><td>4.65</td><td>3.78</td><td>4.01</td></tr><tr><td>Landmark based methods</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GADS (2025) [11]</td><td>0.02</td><td>3.31</td><td>5.00</td><td>2.94</td><td>3.75</td></tr><tr><td>EVA-GCN (2021) [7]</td><td>3.30</td><td>4.78</td><td>4.01</td><td>2.98</td><td>3.92</td></tr><tr><td>STAG-Net (FaceMesh)</td><td>0.012</td><td>2.37</td><td>3.09</td><td>2.24</td><td>2.57</td></tr><tr><td>STAG-Net (FAN)</td><td>0.034</td><td>2.31</td><td>3.00</td><td>1.87</td><td>2.39</td></tr></table>

An important observation from both evaluation protocols is the computational efficiency of STAG-Net. While several competing methods employ tens or even hundreds of millions of parameters, the proposed models require only 0.012 to 0.043 million parameters depending on the landmark detector and configuration (Table 10). Despite this compact design, STAG-Net consistently achieves the lowest reported or highly competitive accuracy across both benchmark protocols. These results demonstrate that the proposed structure aware grouped representation, combined with per group DeepSet encoding and inter group attention, provides an effective balance between estimation accuracy and computational efficiency, making the STAG-Net regressor a promising candidate for TinyML and other resource constrained deployment scenarios.

Because the aggregate errors are computed only for frames with valid landmark detections, the following analysis examines detector coverage and the variation of error across the pose range.

Per Angle Error Analysis Fig. 11 shows how the absolute error changes with ground truth angle magnitude on BIWI, while Table 5 reports the angle ranges for which each detector returns valid landmarks. Both variants are most accurate near frontal poses and their errors increase as the pose magnitude grows. This pattern is consistent with the concentration of 300W-LP samples around frontal poses, as shown in Fig. 10, and with the increasing self occlusion of facial landmarks at large rotations.

The two variants differ mainly because their detectors have different operating ranges. FaceMesh is generally more accurate over the moderate pose ranges that contain most of the test samples, while FAN covers more of the extreme pose range. This difference is visible in the outermost bins of Fig. 11. In pitch, both detectors cover every bin, since the FaceMesh envelope extends to 70.9<sup>◦</sup>. In yaw and roll, however, the FaceMesh envelope ends at 78.0<sup>◦</sup> and 48.7<sup>◦</sup> respectively, which lies just inside the lower edge of the largest bin in each case; as a result, FaceMesh yields too few detections there to form an error distribution, and only the FAN variant produces predictions across the full annotated range. FAN errors also increase near the limits of its own coverage. The combined system therefore presents a detector level tradeoff. FaceMesh offers higher throughput and strong performance in near frontal and moderate poses, while FAN provides broader angular coverage at a much higher computational cost. This finding explains why detector choice affects both the aggregate errors in Tables 3 and 4 and the practical operating range of the deployed pipeline.

Table 5: Detection envelope of the two landmark detectors: the range of ground truth angles (min, max, in degrees) for which each detector produces valid landmarks, compared with the full annotated range of each dataset (Table 1). The regression MAE for each STAG-Net variant is computed only on frames for which its detector returns valid landmarks.
<table><tr><td>Dataset</td><td>Angle</td><td>Full range</td><td>FaceMesh</td><td>FAN</td></tr><tr><td>BIWI</td><td>Yaw Pitch Roll</td><td>-74.9, 78.0 -63.8, 84.4 -64.7, 45.0</td><td>-74.9, 78.0 -63.1,70.9 -48.7, 43.2</td><td>-74.9, 78.0 -63.8, 82.4 -56.4, 45.0</td></tr><tr><td>300W-LP</td><td>Yaw Pitch Roll</td><td>–90.0, 90.0 -64.6, 61.9 -73.4, 53.2</td><td>-90.0, 89.9 -59.4, 58.3 –68.5, 51.9</td><td>-90.0, 90.0 -63.9,61.1 -73.4, 53.2</td></tr><tr><td>AFLW2000</td><td>Yaw Pitch Roll</td><td>-83.7, 86.6 -83.0, 88.3 -87.1, 83.7</td><td>-80.6, 83.1 –64.6, 62.5 -86.2, 80.3</td><td>-83.7, 86.6 -83.0,88.3 -86.5, 83.7</td></tr></table>

The comparison with existing methods demonstrates that STAG-Net achieves strong estimation accuracy while retaining a compact model size across both evaluation protocols. However, these results alone do not explain how the individual design choices contribute to the final performance. Therefore, the following subsection examines the influence of landmark selection, connectivity graph construction, architectural configuration, and learning rate scheduling to identify the design choices that support the final STAG-Net configuration.

## 5.2 Design Analysis

The following experiments examine STAG-Net using FaceMesh under Protocol II. Unless otherwise stated, only the component under investigation is changed while the remaining clustering, architectural, and training settings are kept fixed.

Effect of the Connectivity Graph Neighborhood The connectivity graph used during agglomerative clustering determines which facial landmarks are allowed to be grouped together. To examine its influence, the FaceMesh connectivity graph is constructed using two, four, and six nearest neighbors while keeping the clustering procedure, movement based landmark selection, model architecture, and training settings unchanged. The number of clusters is not manually fixed for this comparison. Instead, it is selected independently using silhouette analysis. As shown in Table 6, all three neighborhood settings produce five clusters.

![](images/93fd5b2f0fc74389ea19113ff735daf2bf98eea1d54b7d4fb4faffa1500596c7.jpg)  
Figure 11: Distribution of absolute errors of the two STAG-Net variants on BIWI (Protocol I), binned by ground truth angle magnitude. A missing box indicates that the corresponding landmark detector produced too few valid detections in that range to form an error distribution, consistent with the detector envelopes reported in Table 5. Per bin MAE curves are provided in Appendix A.

Table 6: Comparison of connectivity graph neighborhood settings for STAG-Net using FaceMesh under Protocol II. The number of clusters is selected using silhouette analysis.
<table><tr><td>Neighbors</td><td>Clusters</td><td>Yaw</td><td>Pitch</td><td>Roll</td><td>MAE</td></tr><tr><td>2</td><td>5</td><td>2.517</td><td>3.218</td><td>2.367</td><td>2.701</td></tr><tr><td>4</td><td>5</td><td>2.440</td><td>3.235</td><td>2.296</td><td>2.657</td></tr><tr><td>6</td><td>5</td><td>2.373</td><td>3.091</td><td>2.239</td><td>2.568</td></tr></table>

The visualizations in Fig. 12 show that changing the neighborhood size alters the spatial composition of the automatically generated clusters, although silhouette analysis selects five clusters in every case. The four and six neighbor settings produce more spatially continuous groupings than the two neighbor setting. This observation is consistent with the quantitative results in Table 6. Increasing the neighborhood size from two to four reduces the overall MAE from 2.701 to 2.657, while six neighbors further reduce it to 2.568. The six neighbor setting also achieves the lowest yaw, pitch, and roll errors among the evaluated configurations. Therefore, six neighbors are used to construct the connectivity graph in the final STAG-Net configuration.

Changing the graph neighborhood does not alter the trainable parameter count because the model architecture remains unchanged and all three settings produce five clusters. The neighborhood size only affects the connectivity constraints used during clustering and consequently changes the cluster membership and the landmarks retained from each cluster.

Table 7: Effect of movement based landmark selection for STAG-Net using FaceMesh under Protocol II using six connectivity graph neighbors and the resulting five clusters. Boldface identifies the configuration selected according to the lowest overall MAE.
<table><tr><td>Retained</td><td>Yaw</td><td>Pitch</td><td>Roll</td><td>MAE</td></tr><tr><td>25%</td><td>2.408</td><td>3.162</td><td>2.184</td><td>2.584</td></tr><tr><td>20%</td><td>2.334</td><td>3.242</td><td>2.276</td><td>2.617</td></tr><tr><td>15%</td><td>2.373</td><td>3.091</td><td>2.239</td><td>2.568</td></tr><tr><td>10%</td><td>2.428</td><td>3.180</td><td>2.329</td><td>2.646</td></tr><tr><td>5%</td><td>2.572</td><td>2.930</td><td>2.421</td><td>2.641</td></tr></table>

Effect of the Landmark Selection Percentage Table 7 examines the effect of movement based landmark selection for STAG-Net using FaceMesh under Protocol II. Retaining the highest 15% of movement ranked landmarks produces the lowest overall MAE of 2.568 and is therefore used in the main experiments, as indicated by the bolded row. Retaining 25% produces the lowest roll error and the second lowest overall MAE of 2.584, while retaining 20% produces the lowest yaw error. Retaining only 5% produces the lowest pitch error, but its yaw and roll errors are higher. These results show that the optimal retained percentage varies across the individual pose angles. However, retaining 15% provides the lowest average error across yaw, pitch, and roll.

Effect of Architectural Hyperparameters Table 8 examines the sensitivity of STAG-Net using FaceMesh under Protocol II to its main architectural choices. The model remains robust across all evaluated settings, with the overall MAE ranging from 2.57<sup>◦</sup> to 2.72<sup>◦</sup>. The selected configuration, consisting of a dropout rate of 0.01, two attention heads, an encoder width of 32, and one encoder and one decoder layer, achieves the lowest overall MAE of 2.57<sup>◦</sup>. Increasing the number of attention heads, widening the encoder, or deepening the encoder and decoder stack does not improve the overall accuracy. These findings support the use of the smallest evaluated architectural

![](images/ac1a2b66a764cbb21ed5cc908834f76612785c6a0fe13f84e3d6241f65099340.jpg)  
(a) Two neighbors

![](images/4feee1165d373c244cc9c49da33741beac38619601dd79b63da877168eaa67e6.jpg)  
(b) Four neighbors

![](images/f9e30bafdfbb4f2914e94c399b428304a9beb37d3e523c92da7b4229f173659d.jpg)  
(c) Six neighbors  
Figure 12: Automatically generated FaceMesh landmark clusters obtained using connectivity graphs with different neighborhood sizes. Black outlines indicate landmarks whose movement scores are at or above the 85th percentile. Silhouette analysis produces five clusters in all three settings, while the neighborhood size changes their spatial composition.

Table 8: Effect of architectural hyperparameters (STAG-Net (FaceMesh), Protocol II). Defaults marked †: dropout 0.01, $n _ { h } = 2 , h _ { e } = 3 2$ , one encoder and one decoder layer.
<table><tr><td></td><td>Setting</td><td>Yaw</td><td>Pitch</td><td>Roll</td><td>MAE</td></tr><tr><td rowspan="3">Dropout</td><td>0</td><td>2.60</td><td>3.05</td><td>2.31</td><td>2.65</td></tr><tr><td>0.01†</td><td>2.37</td><td>3.09</td><td>2.24</td><td>2.57</td></tr><tr><td>0.05</td><td>2.49</td><td>3.26</td><td>2.42</td><td>2.72</td></tr><tr><td rowspan="3">Heads  $n _ { h }$ </td><td>2†</td><td>2.37</td><td>3.09</td><td>2.24</td><td>2.57</td></tr><tr><td>4</td><td>2.44</td><td>3.26</td><td>2.29</td><td>2.67</td></tr><tr><td>6</td><td>2.40</td><td>3.26</td><td>2.34</td><td>2.66</td></tr><tr><td rowspan="2">Width he</td><td> $3 2 ^ { \dagger }$ </td><td>2.37</td><td>3.09</td><td>2.24</td><td>2.57</td></tr><tr><td>64</td><td>2.46</td><td>3.22</td><td>2.25</td><td>2.64</td></tr><tr><td rowspan="4">Encoder and decoder layers</td><td>1 and 1†</td><td>2.37</td><td>3.09</td><td>2.24</td><td>2.57</td></tr><tr><td>1 and 2</td><td>2.57</td><td>3.04</td><td>2.37</td><td>2.66</td></tr><tr><td>1 and 3</td><td>2.48</td><td>3.05</td><td>2.34</td><td>2.62</td></tr><tr><td>2 and 2</td><td>2.56</td><td>3.09</td><td>2.41</td><td>2.69</td></tr></table>

configuration.

Effect of the Learning Rate Scheduling Strategy To examine the effect of the optimization schedule, four learning rate strategies are compared using STAG-Net with FaceMesh under Protocol II. All remaining training and architectural settings are kept unchanged. CosineAnnealingLR is applied over the complete 150-epoch training period with $T _ { \mathrm { m a x } } = 1 5 0$ , while MultiStepLR reduces the learning rate at epochs 70, 90, and 120. The constant learning rate setting retains the initial learning rate throughout training. CosineAnnealingWarmRestarts uses an initial restart period of $T _ { 0 } = 1 0$ and a restart period multiplier of $T _ { \mathrm { m u l t } } = 2$

Table 9 shows that CosineAnnealingWarmRestarts achieves the lowest overall MAE of 2.568 and is therefore used in the main experiments. The constant learning rate produces the lowest yaw error of 2.365, while CosineAnnealingLR produces the lowest pitch and roll errors of 3.079 and 2.192, respectively. However, CosineAnnealingWarmRestarts provides the most balanced performance across the three pose angles. MultiStepLR performs considerably worse, with an overall MAE of 3.155. The relatively small differences among CosineAnnealingWarmRestarts, CosineAnnealingLR, and the constant learning rate indicate that STAG-Net remains stable across these scheduling strategies, although the warm restart schedule provides the strongest overall result.

Table 9: Comparison of learning rate scheduling strategies for STAG-Net using FaceMesh under Protocol II.
<table><tr><td>Scheduler</td><td>Yaw</td><td>Pitch</td><td>Roll</td><td>MAE</td></tr><tr><td>CosineAnnealingWarmRestarts</td><td>2.373</td><td>3.091</td><td>2.239</td><td>2.568</td></tr><tr><td>CosineAnnealingLR</td><td>2.456</td><td>3.079</td><td>2.192</td><td>2.576</td></tr><tr><td>MultiStepLR</td><td>3.576</td><td>3.538</td><td>2.351</td><td>3.155</td></tr><tr><td>Constant learning rate</td><td>2.365</td><td>3.100</td><td>2.307</td><td>2.591</td></tr></table>

Overall, the design analysis shows that the proposed movement aware landmark selection, connectivity based clustering, compact architecture, and learning rate scheduling strategy each contribute to the final performance of STAG-Net. The results indicate that retaining only the most informative landmarks is sufficient for accurate estimation and that increasing architectural complexity does not consistently improve performance. The clustering and architectural configurations vary according to the landmark detector and evaluation protocol, while all variants retain the same lightweight design principles. The complete detector and protocol specific STAG-Net configurations are summarized in Appendix B. Having established the final design choices, the following subsection evaluates the computational performance of the resulting STAG-Net models.

## 5.3 Computational Performance Analysis

Model Complexity Table 10 compares the trainable parameters and inference FLOPs of STAG-Net with representative image based methods. Parameter counts for prior methods follow their original publications, while FLOPs are measured from the released implementations where available. For STAG-Net, the reported FLOPs cover only the pose regression model. Across the four configurations, the regressor requires between $2 . 4 \times 1 0 ^ { 4 }$ and $8 . 4 \times 1 0 ^ { 4 }$ FLOPs per inference. This is approximately three orders of magnitude below FSA-Net and five to six orders of magnitude below HopeNet, 6DRepNet, and TokenHPE. The largest STAG-Net configuration requires about 170 KB for 32 bit parameters and about 43 KB for 8 bit parameters, before accounting for implementation and activation memory.

Table 10: Model complexity comparison. STAG-Net FLOPs are reported per variant (Protocol I / Protocol II) and cover the pose regression model only, while RGB based baselines process raw images directly.
<table><tr><td>Method</td><td>Params (M)</td><td>FLOPs</td></tr><tr><td>TokenHPE</td><td>86.42</td><td>33.97G</td></tr><tr><td>6DRepNet</td><td>39.32</td><td>9.86G</td></tr><tr><td>HopeNet</td><td>23.90</td><td>4.11G</td></tr><tr><td>FSA-Net</td><td>1.20</td><td>91.58M</td></tr><tr><td>STAG-Net (FaceMesh), Protocol I</td><td>0.043</td><td>83.6K</td></tr><tr><td>STAG-Net (FAN), Protocol I</td><td>0.034</td><td>70.7K</td></tr><tr><td>STAG-Net (FaceMesh), Protocol II</td><td>0.012</td><td>24.1 K</td></tr><tr><td>STAG-Net (FAN), Protocol II</td><td>0.034</td><td>70.5 K</td></tr></table>

These figures describe the compactness of the STAG-Net regressor, not the cost of landmark extraction. The combined pipeline cost depends on the detector. FaceMesh supports real time CPU inference in our measurements, while FAN is considerably more expensive. In applications where landmarks are already produced for face tracking or expression analysis, STAG-Net adds only a small incremental cost. A complete microcontroller deployment, including landmark extraction and measured activation memory, remains outside the scope of the present study.

Pareto Analysis of Accuracy and Model Size Fig. 13 combines the accuracy and parameter results from Tables 3 and 4. Under Protocol II, the Pareto frontier contains only the two STAG-Net variants. The FaceMesh variant provides the smallest model and the FAN variant provides the lowest error. Under Protocol I on BIWI, GADS and STAG-Net using FAN form the frontier. The FAN configuration is both smaller and more accurate than the FaceMesh configuration in this setting. On AFLW2000, 6DRepNet is 0.11<sup>◦</sup> more accurate than STAG-Net using FaceMesh, but it uses roughly 900 times more parameters. The Pareto analysis therefore shows that STAG-Net achieves compactness without simply accepting a large loss in accuracy.

Inference Performance Table 11 reports throughput on the Colab CPU and Tesla T4 GPU described in Section 4.3. For STAG-Net, the measurement starts from a cropped face image and includes image conversion, landmark extraction, normalization, landmark selection, and pose regression. The measurements use the largest configuration of each variant, so the reported throughput is conservative. The reported frame rates therefore represent the landmark extraction and regression pipeline rather than the regressor alone.

Table 11: Inference throughput (frames per second) measured on the Google Colab CPU runtime and an NVIDIA Tesla T4 GPU. For STAG-Net, throughput is reported for the complete pipeline from a cropped face image, covering image conversion, landmark extraction, normalization, and regression, and is measured using the largest configuration of each variant (Protocol I), so the reported values represent the worst case across configurations. RGB based baselines are measured from the same cropped face input as in the original works.
<table><tr><td>Method</td><td>CPU (FPS)</td><td>GPU (FPS)</td></tr><tr><td>HopeNet</td><td>4.4</td><td>147.3</td></tr><tr><td>6DRepNet</td><td>6.9</td><td>62.0</td></tr><tr><td>FSA-Net</td><td>114.0</td><td>164.5</td></tr><tr><td>TokenHPE</td><td>6.6</td><td>69.9</td></tr><tr><td>STAG-Net (FaceMesh)</td><td>111.0</td><td>322.9</td></tr><tr><td>STAG-Net (FAN)</td><td>1.6</td><td>17.7</td></tr></table>

The FaceMesh pipeline reaches 322.9 FPS on the GPU and 111.0 FPS on the CPU. Its GPU throughput is almost twice that of the fastest measured image based baseline, while its CPU throughput is close to the 114.0 FPS measured for FSA-Net on the same runtime. Notably, the CPU throughput of the FaceMesh pipeline also exceeds the GPU throughput of 6DRepNet, which is the closest competitor in accuracy under Protocol II and reaches only 6.9 FPS on the CPU. The FAN pipeline is much slower, reaching 17.7 FPS on the GPU and 1.6 FPS on the CPU. This difference is mainly caused by landmark extraction because the STAG-Net regression stage requires fewer than $8 . 4 \times 1 0 ^ { 4 }$ FLOPs. The two variants therefore provide a practical choice between throughput and angular coverage. FaceMesh is the stronger option for latency constrained applications, while FAN is useful when broader extreme pose coverage is more important.

Training Efficiency Table 12 reports the per epoch training time on the hardware described in Section 4.3. Because STAG-Net operates on previously extracted landmark coordinates rather than raw images, a full epoch over the 300W-LP training set (≈122,450 samples) completes in under 11 seconds on the T4 GPU and under 29 seconds on the CPU; training for all 150 epochs therefore finishes in between 24 and 28 minutes on the GPU, or about 70 minutes on CPU alone. Since the baselines are trained on the much smaller BIWI split (≈11,000 frames), raw per epoch times are not directly comparable; normalizing by dataset size, STAG-Net processes a training sample roughly 7 to 8× faster than FSA-Net and two orders of magnitude faster than HopeNet, 6DRepNet, and TokenHPE on the same GPU. This low training cost makes hyperparameter exploration and retraining for new deployment conditions practical without dedicated hardware.

![](images/d2d4a5d80df11a8200b1bfbadd955a46bab581043ed28bca3a69598367b34052.jpg)

![](images/6b35a9ea43cc5f89656549f86fab62d945ded9aacfbb0d0f264705a2776200b1.jpg)

![](images/e62463b232c6a72ab3effefa636b7d87329dc94da8890bf17446fe50868d3d57.jpg)  
Figure 13: Accuracy and model size trade-off: overall MAE versus trainable parameters (log scale) for all compared methods with published parameter counts; FM denotes the FaceMesh variant. STAG-Net variants are marked with stars; methods toward the lower left are jointly smaller and more accurate.

Table 12: Training time per epoch. The last column normalizes the GPU time by dataset size for direct comparison.
<table><tr><td>Method</td><td>CPU (s)</td><td>GPU (s)</td><td>GPU (ms/sample)</td></tr><tr><td>HopeNet</td><td>8051.9</td><td>127.5</td><td>11.6</td></tr><tr><td>6DRepNet</td><td>5232.2</td><td>253.2</td><td>23.0</td></tr><tr><td>TokenHPE</td><td>6365.5</td><td>410.3</td><td>37.4</td></tr><tr><td>FSA-Net</td><td>203.1</td><td>6.7</td><td>0.61</td></tr><tr><td>STAG-Net (FaceMesh)</td><td>25.9</td><td>9.5</td><td>0.077</td></tr><tr><td>STAG-Net (FAN)</td><td>28.0</td><td>10.9</td><td>0.089</td></tr></table>

Summary of Findings The experiments support four conclusions. First, grouped set processing is effective for landmark based head pose estimation. STAG-Net achieves the lowest reported MAE among the compared landmark based methods across all three evaluation settings and the lowest reported BIWI MAE among all compared methods under both protocols on frames with valid landmark detections. Second, the regression model is compact in terms of parameters and FLOPs, and the FaceMesh configuration provides real time throughput for landmark extraction and regression on the Google Colab CPU runtime. Third, the design analysis shows that performance is stable across the tested architectural settings. It also confirms that a compact landmark subset selected using movement is sufficient and that connectivity graph construction influences the quality of the groups. Fourth, the main variation between the two system configurations comes from the landmark detector. Detector coverage limits estimation at extreme poses and detector cost dominates the latency of the combined pipeline.

## 6 Conclusion

This work introduced STAG-Net, a compact structure aware architecture for landmark based head pose estimation. The method groups facial landmarks into automatically discovered spatial regions, retains the most pose sensitive landmarks in each region, encodes every group as an unordered set, and models relationships among groups through multihead self attention. A compact geometry descriptor complements the learned group representation before the final yaw, pitch, and roll prediction.

The results follow the objectives established in the introduction. Across the three reported evaluation settings, STAG-Net achieves the lowest reported MAE among the landmark based methods included in the comparison. On frames with valid landmark detections, it also obtains the lowest reported overall BIWI MAE among all compared methods under both protocols, with 3.51<sup>◦</sup> under cross dataset evaluation and 2.39<sup>◦</sup> under subject independent evaluation. Depending on the configuration, the pose regressor uses between 0.012 and 0.043 million parameters and requires fewer than $8 . 4 \times 1 0 ^ { 4 }$ FLOPs per inference. The Pareto analysis further shows that the two STAG-Net variants alone form the Protocol II frontier. The FaceMesh configuration provides real time throughput for landmark extraction and regression on the Google Colab CPU runtime, while FAN provides wider coverage at extreme poses. The design analysis supports the movement aware selection strategy and shows that the compact architecture does not depend on a narrowly tuned hyperparameter setting. We also identified and removed 36 physically implausible AFLW2000 annotations and reported the cleaned pose ranges used in the evaluation.

Two limitations remain. First, the complete system inherits the coverage and computational cost of its landmark detector. FaceMesh is fast but does not cover some extreme poses, while FAN covers a wider range but is not suitable for real time CPU operation. Second, the TinyML assessment is based on parameter counts, regression FLOPs, and Google Colab CPU measurements rather than a complete microcontroller implementation. Future work should therefore evaluate quantized STAG-Net models on microcontroller class hardware and report measured latency, flash use, activation memory, and energy. A lightweight on device landmark detector is also required before the entire pipeline can be considered fully embedded. Further work can extend the grouped set representation to continuous rotation outputs and to related landmark based tasks such as gaze and expression estimation.

## Acknowledgment

This study is funded by the Accelerating Higher Education Expansion and Development (AHEAD) Operation of the Ministry of Higher Education of Sri Lanka, funded by the World Bank (https://ahead.lk/result-area-3/) and a research grant funded by the Sri Lanka Institute of Information Technology, Sri Lanka (Grant No. PVC(R&I)/RG/2025/12). .

The authors used Anthropic’s Claude to assist with improving grammar and clarity, generating the model architecture diagram, and checking numerical consistency across tables. All experiments, data, technical content, analyses, results, interpretations, and conclusions were developed and verified by the authors, who take full responsibility for the content of this article.

## References

[1] E. Murphy-Chutorian and M. M. Trivedi, “ Head Pose Estimation in Computer Vision: A Survey ,” IEEE Transactions on Pattern Analysis & Machine Intelligence, vol. 31, no. 04, pp. 607–626, 2009.

[2] N. Ruiz, E. Chong, and J. M. Rehg, “Fine-grained head pose estimation without keypoints,” 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 2155–215 509, 2017. [Online]. Available: https://api.semanticscholar.org/CorpusID: 4880325

[3] A. Bulat and G. Tzimiropoulos, “How far are we from solving the 2d & 3d face alignment problem? (and a dataset of 230,000 3d facial landmarks),” in Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017.

[4] P. Warden and D. Situnayake, TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers. O’Reilly Media, Inc., 2019. [Online]. Available: https://books.google.com/ books/about/TinyML.html?id=sB3mxQEACAAJ

[5] P. Werner, F. Saxen, and A. Al-Hamadi, “Landmark based head pose estimation benchmark and method,” in 2017 IEEE International Conference on Image Processing (ICIP), 2017, pp. 3909–3913.

[6] D. Derkach, A. Ruiz, and F. M. Sukno, “Head pose estimation based on 3-d facial landmarks localization and regression,” in 2017 12th IEEE International Conference on Automatic Face & Gesture Recognition (FG 2017), 2017, pp. 820–827.

[7] M. Xin, S. Mo, and Y. Lin, “Eva-gcn: Head pose estimation based on graph convolutional networks,” in 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2021, pp. 1462–1471.

[8] S. Mo and X. Miao, “Osgg-net: One-step graph generation network for unbiased head pose estimation,”

in Proceedings ofthe 29th ACM International Conference on Multimedia, ser. MM ’21. New York, NY, USA: Association for Computing Machinery, 2021, p. 2465–2473. [Online]. Available: https://doi.org/10.1145/ 3474085.3475417

[9] W.-N. Lie, M. Yim, L. Aing, and J.-C. Chiang, “3d head pose estimation based on graph convolutional network from a single rgb image,” in 2022 IEEE International Conference on Image Processing (ICIP), 2022, pp. 3963– 3967.

[10] V. Menan, A. Gawesha, P. Samarasinghe, and D. Kasthurirathna, “Ds-hpe: Deep set for head pose estimation,” in 2023 IEEE 13th Annual Computing and Communication Workshop and Conference (CCWC), 2023, pp. 1179– 1184.

[11] M. Velayuthan, A. Gawesha, P. Velayuthan, N. Kodagoda, D. Kasthurirathna, and P. Samarasinghe, “Gads: A super lightweight model for head pose estimation,” 2025. [Online]. Available: https://arxiv.org/abs/2504.15751

[12] A. Asperti and D. Filippini, “Deep learning for head pose estimation: A survey,” SN Comput. Sci., vol. 4, no. 4, Apr. 2023. [Online]. Available: https://doi.org/10.1007/s42979-023-01796-z

[13] A. F. Abate, C. Bisogni, A. Castiglione, and M. Nappi, “Head pose estimation: An extensive survey on recent techniques and applications,” Pattern Recognition, vol. 127, p. 108591, 2022. [Online]. Available: https://www.sciencedirect.com/science/article/pii/ S0031320322000723

[14] L. Beyer, A. Hermans, and B. Leibe, “Biternion nets: Continuous head pose regression from discrete training labels,” in Pattern Recognition, J. Gall, P. Gehler, and B. Leibe, Eds. Cham: Springer International Publishing, 2015, pp. 157–168.

[15] T.-Y. Yang, Y.-T. Chen, Y.-Y. Lin, and Y.-Y. Chuang, “Fsanet: Learning fine-grained structure aggregation for head pose estimation from a single image,” in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 1087–1096.

[16] Y. Zhou and J. Gregson, “Whenet: Real-time fine-grained estimation for wide range head pose,” ArXiv, vol. abs/2005.10353, 2020. [Online]. Available: https://api.semanticscholar.org/CorpusID:218763523

[17] T. Hempel, A. A. Abdelrahman, and A. Al-Hamadi, “6d rotation representation for unconstrained head pose estimation,” in 2022 IEEE International Conference on Image Processing (ICIP), 2022, pp. 2496–2500.

[18] N. Dhingra, “Headposr: End-to-end trainable head pose estimation using transformer encoders,” in 2021 16th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2021), 2021, pp. 1–8.

[19] C. Zhang, H. Liu, Y. Deng, B. Xie, and Y. Li, “Tokenhpe: Learning orientation tokens for efficient head pose estimation via transformers,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 8897–8906.

[20] H.-W. Hsu, T.-Y. Wu, S. Wan, W. H. Wong, and C.-Y. Lee, “Quatnet: Quaternion-based head pose estimation with multiregression loss,” IEEE Transactions on Multimedia, vol. 21, no. 4, pp. 1035–1046, 2019.

[21] Z. Cao, Z. Chu, D. Liu, and Y. Chen, “A vector-based representation to enhance head pose estimation,” in 2021 IEEE Winter Conference on Applications of Computer Vision (WACV), 2021, pp. 1187–1196.

[22] T. Hempel, A. A. Abdelrahman, and A. Al-Hamadi, “6d rotation representation for unconstrained head pose estimation,” in 2022 IEEE International Conference on Image Processing (ICIP), 2022, pp. 2496–2500.

[23] ——, “Toward robust and unconstrained full range of rotation head pose estimation,” IEEE Transactions on Image Processing, vol. 33, pp. 2377–2387, 2024.

[24] R. David, J. Duke, A. Jain, V. J. Reddi, N. Jeffries, J. Li, N. Kreeger, I. Nappier, M. Natraj, S. Regev, R. Rhodes, T. Wang, and P. Warden, “Tensorflow lite micro: Embedded machine learning on tinyml systems,” 2021. [Online]. Available: https://arxiv.org/abs/2010.08678

[25] J. Lin, W.-M. Chen, Y. Lin, J. Cohn, C. Gan, and S. Han, “Mcunet: Tiny deep learning on iot devices,” 2020. [Online]. Available: https://arxiv.org/abs/2007.10319

[26] G. Garcia-Mateos, A. Ruiz, P. E. Lopez-de Teruel, A. L. Rodriguez, and L. Fernandez, “Estimating 3d facial pose in video with just three points,” in 2008 IEEE Computer Society Conference on Computer Vision and Pattern Recognition Workshops, 2008, pp. 1–8.

[27] H. Yuan, M. Li, J. Hou, and J. Xiao, “Single image-based head pose estimation with spherical parametrization and 3d morphing,” Pattern Recognition, vol. 103, p. 107316, 2020.

[28] M. Zaheer, S. Kottur, S. Ravanbakhsh, B. Poczos, R. Salakhutdinov, and A. Smola, “Deep sets,” 2018. [Online]. Available: https://arxiv.org/abs/1703.06114

[29] R. Q. Charles, H. Su, M. Kaichun, and L. J. Guibas, “Pointnet: Deep learning on point sets for 3d classification and segmentation,” in 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 77–85.

[30] C. R. Qi, L. Yi, H. Su, and L. J. Guibas, “Pointnet++: Deep hierarchical feature learning on point sets in a metric space,” 2017. [Online]. Available: https: //arxiv.org/abs/1706.02413

[31] J. Lee, Y. Lee, J. Kim, A. R. Kosiorek, S. Choi, and Y. W. Teh, “Set transformer: A framework for attention-based permutation-invariant neural networks,” 2019. [Online]. Available: https://arxiv.org/abs/1810.00825

[32] A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M. Andreetto, and H. Adam, “Mobilenets: Efficient convolutional neural networks for mobile vision applications,” arXiv preprint arXiv:1704.04861, 2017. [Online]. Available: https://arxiv.org/abs/1704.04861

[33] M. Tan and Q. V. Le, “Efficientnet: Rethinking model scaling for convolutional neural networks,” 2020. [Online]. Available: https://arxiv.org/abs/1905.11946

[34] N. Dhingra, “Lwposr: Lightweight efficient fine grained head pose estimation,” in 2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2022, pp. 1204–1214.

[35] X. Li, D. Zhang, M. Li, and D.-J. Lee, “Accurate head pose estimation using image rectification and a lightweight convolutional neural network,” IEEE Transactions on Multimedia, vol. 25, pp. 2239–2251, 2023.

[36] L. Lai, N. Suda, and V. Chandra, “Cmsis-nn: Efficient neural network kernels for arm cortex-m cpus,” 2018. [Online]. Available: https://arxiv.org/abs/1801.06601

[37] J. Lin, W.-M. Chen, H. Cai, C. Gan, and S. Han, “Mcunetv2: Memory-efficient patch-based inference for tiny deep learning,” 2024. [Online]. Available: https://arxiv.org/abs/2110.15352

[38] M. Rusci, A. Capotondi, and L. Benini, “Memory-driven mixed low precision quantization for enabling deep network inference on microcontrollers,” 2019. [Online]. Available: https://arxiv.org/abs/1905.13082

[39] A. Lindamulage, N. Kodagoda, S. Reyal, P. Samarasinghe, and P. Yogarajah, “Comparative study of parameter selection for enhanced edge inference for a multi-output regression model for head pose estimation,” in TENCON 2022 - 2022 IEEE Region 10 Conference (TENCON), 2022, pp. 1–6.

[40] E. Liberis and N. D. Lane, “Differentiable neural network pruning to enable smart applications on microcontrollers,” Proc. ACM Interact. Mob. Wearable Ubiquitous Technol., vol. 6, no. 4, Jan. 2023. [Online]. Available: https://doi.org/10.1145/3569468

[41] Y. Kartynnik, A. Ablavatski, I. Grishchenko, and M. Grundmann, “Real-time facial surface geometry from monocular video on mobile gpus,” 2019. [Online]. Available: https://arxiv.org/abs/1907.06724

[42] X. Zhu, X. Liu, Z. Lei, and S. Z. Li, “Face alignment in full pose range: A 3d total solution,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 41, no. 1, p. 78–92, Jan. 2019. [Online]. Available: http://dx.doi.org/10.1109/TPAMI.2017.2778152

[43] C. Sagonas, G. Tzimiropoulos, S. Zafeiriou, and M. Pantic, “300 faces in-the-wild challenge: The first facial landmark localization challenge,” in Proceedings ofthe IEEE International Conference on Computer Vision Workshops, 2013, pp. 397–403.

[44] X. Zhu, Z. Lei, J. Yan, D. Yi, and S. Z. Li, “High-fidelity pose and expression normalization for face recognition in the wild,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2015, pp. 1087– 1096.

[45] G. Fanelli, T. Weise, J. Gall, and L. Van Gool, “Real time head pose estimation from consumer depth cameras,” in 33rd Annual Symposium ofthe German Associationfor Pattern Recognition (DAGM’11), September 2011.

[46] C. R. Banbury, V. J. Reddi, M. Lam, W. Fu, A. Fazel, J. Holleman, X. Huang, R. Hurtado, D. Kanter, A. Lokhmotov, D. Patterson, D. Pau, J.-s. Seo, J. Sieracki, U. Thakker, M. Verhelst, and P. Yadav, “Benchmarking tinyml systems: Challenges and direction,” in Proceedings ofSysML, 2020, arXiv preprint arXiv:2003.04821.

[47] A. Kumar, A. Alavi, and R. Chellappa, “Kepler: Keypoint and pose estimation of unconstrained faces by learning ef ficient h-cnn regressors,” in 2017 12th IEEE International Conference on Automatic Face & Gesture Recognition (FG 2017), 2017, pp. 258–265.

[48] X. Zhu, X. Liu, Z. Lei, and S. Z. Li, “Face alignment in full pose range: A 3d total solution,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 41, no. 1, pp. 78–92, jan 2019. [Online]. Available: https://doi.org/10.1109%2Ftpami.2017.2778152

[49] Z. Cao, Z. Chu, D. Liu, and Y. Chen, “A vector-based representation to enhance head pose estimation,” in 2021 IEEE Winter Conference on Applications of Computer Vision (WACV), 2021, pp. 1187–1196.

[50] M. Shao, Z. Sun, M. Ozay, and T. Okatani, “Improving head pose estimation with a combined loss and bounding box margin adjustment,” 2019. [Online]. Available: https://arxiv.org/abs/1905.08609

[51] N. Ruiz, E. Chong, and J. M. Rehg, “Fine-grained head pose estimation without keypoints,” in Proceedings ofthe IEEE conference on computer vision and pattern recognition workshops, 2018, pp. 2074–2083.

[52] T.-Y. Yang, Y.-H. Huang, Y.-Y. Lin, P.-C. Hsiu, and Y.-Y. Chuang, “SSR-Net: A Compact Soft Stagewise Regression Network for Age Estimation,” in Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence, IJCAI-18. International Joint Conferences on Artificial Intelligence Organization, 2018, pp. 1078–1084. [Online]. Available: https://doi.org/10.24963/ijcai.2018/150

[53] V. Kazemi and J. Sullivan, “One millisecond face alignment with an ensemble of regression trees,” in 2014 IEEE Conference on Computer Vision and Pattern Recognition, 2014, pp. 1867–1874.

[54] S. Tourani, S. Tourani, A. Mahmood, and M. H. Khan, “Unsupervised discovery of facial landmarks and head pose,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 21 192–21 202.

[55] M. Martin, F. Van De Camp, and R. Stiefelhagen, “Real time head model creation and head pose estimation on consumer depth cameras,” in 2014 2nd International Conference on 3D Vision, vol. 1, 2014, pp. 641–648.

[56] J. Gu, X. Yang, S. De Mello, and J. Kautz, “Dynamic facial analysis: From bayesian filtering to recurrent neural network,” in 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 1531–1540.

## A Per Bin Error Curves

Figs. 14, 15, and 16 complement the box plots in Fig. 11 by showing the mean absolute error of both STAG-Net variants as a function of the ground truth yaw, pitch, and roll angle on BIWI. Gaps in a curve correspond to angle ranges in which the respective landmark detector produced too few valid detections to compute a reliable mean.

![](images/ebcb7e4468b0946b87537de2aca2780f69abee600686a0737a41e369ce565096.jpg)  
Figure 14: MAE versus ground truth yaw angle.

![](images/877ea052312822b5bdc9b1ef2bd5c70879fc31b5ebd25b5ad34eb6b31fd1c0eb.jpg)  
Figure 15: MAE versus ground truth pitch angle.

![](images/3d947b07e5dd55c0dea349d55587debebd27cf24bbdeacf696cfbfe56f30b276.jpg)  
Figure 16: MAE versus ground truth roll angle.

## B STAG-Net Configuration Details

Table 13 summarizes the detector and protocol specific configurations used in the final experiments. Each cluster is processed using an independent DeepSet encoder consisting of input batch normalization, a linear projection to $h _ { e }$ units, batch normalization, ReLU activation, and dropout. Mean and max pooling are then applied to obtain a fixed length representation for each cluster. The decoder projects the pooled representation to F units, while each angle specific regression head maps the trunk output from D to 32 and then to a single pose estimate.

Table 13: Detector and protocol specific STAG-Net architecture configurations.
<table><tr><td rowspan="2">Configuration</td><td colspan="2">Protocol I</td><td colspan="2">Protocol II</td></tr><tr><td>FaceMesh</td><td>FAN</td><td>FaceMesh</td><td>FAN</td></tr><tr><td>Encoder width,  $h _ { e }$ </td><td>64</td><td>64</td><td>32</td><td>64</td></tr><tr><td>Token dimension,  $F$ </td><td>32</td><td>32</td><td>16</td><td>32</td></tr><tr><td>Attention heads,  $n _ { h }$ </td><td>4</td><td>4</td><td>2</td><td>2</td></tr><tr><td>Clusters, K</td><td>7</td><td>5</td><td>5</td><td>5</td></tr><tr><td>Trunk output,  $D$ </td><td>36</td><td>36</td><td>32</td><td>36</td></tr><tr><td>Connectivity graph neighbors</td><td>6</td><td>4</td><td>6</td><td>4</td></tr><tr><td>Retained landmarks, Ns (%)</td><td>15</td><td>40</td><td>15</td><td>40</td></tr><tr><td>Trainable parameters</td><td>42,557</td><td>33,521</td><td>12,209</td><td>33,521</td></tr></table>

For a fixed encoder width, the use of an independent DeepSet encoder for each cluster causes the encoder parameter count to increase with the number of clusters K. However, the total parameter count also depends on the encoder width, token dimension, and trunk output dimension. The number of attention heads partitions the token representation and therefore does not change the parameter count when the token dimension remains fixed.