# Luggage re-identification: A study of strategies towards a real-world application

Quentin Dhote, Romaric Audigierˆ

Universite Paris-Saclay, CEA, List, F-91120 Palaiseau, France´

quentin.dhote@cea.fr, romaric.audigier@cea.fr

Abstract—Luggage re-identification (ReID) is a critical task in video surveillance for forensics, security or border management (suspicious or abandoned luggage recognition), as well as luggage management (lost luggage recovery). It enables the association of observations of the same luggage across disjoint camera views in complex environments such as airports, train stations and border crossing. While existing approaches primarily focus on appearance-based cropped image matching under relatively clean conditions, real-world scenarios are characterized by frequent occlusions, cluttered background and crowded scenes which significantly degrade performance. Yet, strategies leveraging the complementarity of multiple shots and views from different video streams is overlooked. In this paper, we study strategies for an end-to-end pipeline that operates from multiple video streams to detect and re-identify luggage. We evaluate different variants on a benchmark dataset acquired in real-world environments to address the questions of selecting and merging different views and shots of the same luggage as well as training on different environment datasets for an overall ReID performance improvement in practical applications.

Index Terms—Luggage re-identification, luggage retrieval, luggage detection and tracking, surveillance application

## I. INTRODUCTION

Luggage re-identification (ReID) has emerged as a critical capability in modern video surveillance systems, particularly in large-scale environments such as airports and transportation hubs. Existing approaches typically focus on improving feature extraction or metric learning strategies for ReID, often assuming clean detections and fully visible objects. Yet, practical deployment raises several system-level questions that remain largely unexplored: How many cameras are actually required to achieve reliable luggage ReID? How many observations should be gathered for each tracked instance? Does aggregating all available frames improve robustness, or can noisy detections degrade performance? How should multi-camera information be exploited in asymmetric surveillance setups?

To address these questions, we propose an end-to-end pipeline for luggage ReID across multiple scenes in a realistic asymmetric surveillance setting. The gallery’ scene corresponds to the enrollment stage, where luggage items are observed in a controlled environment, e.g., moving alone on a conveyor belt. In contrast, the query’ scene represents unconstrained surveillance areas, where luggage is carried by people and may be partially occluded, viewed from limited angles, or only intermittently visible. Both scenes include multiple cameras providing complementary viewpoints, enabling the proposed approach to leverage multi-view observations rather than relying on single-frame appearance features. This multi-camera setup helps mitigate occlusions and improves the robustness of cross-scene matching. Such a framework is particularly relevant for applications including the retrieval of lost or abandoned luggage, the ReID of suspicious bags previously scanned by security systems, and forensic analysis in large transportation hubs. This practical context further motivates the need for robust multi-camera and multi-view ReID approaches capable of operating under realistic surveillance conditions.

The main contributions of this work are as follows: (1) we propose a practical end-to-end multi-camera pipeline for luggage ReID in realistic surveillance environments, explicitly designed for asymmetric enrollment-vs-monitoring scenarios; (2) we introduce a multi-view and temporal aggregation strategy adapted to the asymmetric nature of our setup, improving robustness to occlusions, missed detections, and fragmented observations; (3) we conduct extensive experimental and ablation studies, including analyses of camera configurations, shot selection strategies, aggregation mechanism, and crossdataset adaptation, providing practical insights for real-world deployment of luggage ReID systems.

## II. RELATED WORK

## A. Object and Luggage Re-Identification

ReID aims at matching instances of the same object across different camera views in non-overlapping surveillance environments. While person, animal and vehicle ReID [1]– [3] have been extensively studied, luggage ReID has received comparatively less attention despite its importance in transportation and security applications [4]–[8]. Existing luggage ReID methods [4]–[8] primarily focus on learning discriminative appearance features from cropped image data. For example, QuadNet [4] introduces a quadruplet loss to learn multi-view representations that are robust to viewpoint variations. However, these methods typically operate on precropped images and assume relatively clean detections, limiting their applicability in real surveillance scenarios.

## B. Whole ReID Pipelines for Real-World Applications

Recent advances in multi-object tracking [9] have incorporated ReID features to improve identity association over time in the same camera. Tracking-by-detection frameworks such as DeepSORT [10] and variants [11], [12] combine motion cues with appearance embeddings to maintain identity consistency across frames. More recent unified approaches jointly optimize detection and ReID within a single framework. While these methods [13], [14] improve tracking robustness, they are primarily designed for single-camera settings and do not explicitly address cross-scene or multi-camera luggage ReID. Also, they often assume relatively continuous object visibility, which is not guaranteed in real-world surveillance environments. Beyond academic research, several industrial solutions have recently explored AI-based luggage ReID systems for realworld airport operations. For instance, a commercial solution automates the ReID of lost or untagged luggage using multiview image acquisition and visual matching techniques in airport baggage handling systems [15]. Yet, unlike our asymmetric scenario, both enrollment and query acquisitions are under controlled environments and luggage is isolated. Such systems further highlight the operational importance of robust luggage ReID pipelines for applications including lost luggage retrieval, baggage tracking, and security investigations.

## C. Occlusion and Multi-View Robust Re-Identification

Occlusion remains one of the most challenging problems in ReID systems. Existing works have proposed part-based representations [16], attention mechanisms [17], and temporal modeling strategies [18] to mitigate the effect of missing or partially visible information. Other approaches specifically target occluded ReID through pose-guided alignment or part discovery strategies [19], [20]. However, most of these approaches focus on person ReID and assume relatively structured motion patterns.

Multi-view learning approaches attempt to leverage complementary camera viewpoints to improve robustness [4], [21], [22]. These methods demonstrate that combining information across views can significantly enhance feature discriminability. Yet, they typically rely on synchronized or fully observed multi-view data, and do not consider real asymmetric surveillance scenarios where non-cooperative behaviors may lead to fragmented or incomplete observations.

## III. METHOD

We consider the problem of luggage ReID across two disjoint scenes with distinct characteristics: a ”query” scene and a ”gallery” scene. Each scene is monitored by multiple cameras providing different viewpoints.

In the gallery scene, luggage is transported on a conveyor belt, resulting in predictable and consistent motion patterns. This controlled setting allows trajectories to be reliably associated across cameras, enabling the extraction of representative observations of each luggage instance. In contrast, the query scene corresponds to a more complex environment where luggage is carried by individuals, leading to irregular motion, partial visibility, and fragmented trajectories that cannot be consistently associated across cameras.

Given a luggage instance enrolled in the gallery scene, the objective is to retrieve the corresponding observations in the query scene across realistic surveillance conditions, enabling practical applications such as lost luggage retrieval, suspicious baggage search, or forensic analysis.

## A. Pipeline Overview

We propose a pipeline for luggage ReID across disjoint scenes, explicitly designed to handle the asymmetric characteristics of setup. An overview of the proposed framework is illustrated in Fig. 1. The pipeline consists of four main stages:

1) Detection and Tracking: Each video stream is processed independently to detect luggage instances and then associated to form tracks over time. In the gallery scene, the structured motion of luggage allows consistent association of tracks across cameras. In contrast, in the query scene, tracking is performed per camera due to the absence of reliable crosscamera associations.

2) Scene-Specific Representation Construction: Due to the asymmetric nature of the considered surveillance setup, several strategies can be considered to construct feature representations for the gallery and query scenes. Depending on the acquisition conditions, a luggage instance may be represented using a single observation, multiple observations from different cameras, or multiple temporal observations extracted from a track. In controlled environments where trajectories are predictable and objects are consistently observed across viewpoints, multi-camera representations can be constructed by selecting one or several observations per camera. In less constrained surveillance areas, where visibility may be fragmented due to occlusions or unconstrained motion, multiple observations can instead be extracted from a track and combined into a track-level representation.

3) Feature Extraction: For each detected instance, visual features are extracted using a deep neural network. These features capture appearance cues that should be invariant to viewpoint and illumination changes, and can then be aggregated depending on the multi-shot aggregation strategy.

4) Cross-Scene Matching: Finally, ReID is performed by matching query representations against gallery representations. Each query track is compared to all gallery tracks using a similarity metric, and gallery candidates are ranked accordingly.

This design leverages the complementary strengths of both scenes: structured multi-view information in the gallery and temporal redundancy in the query. As a result, the proposed pipeline effectively mitigates the impact of occlusions and viewpoint variations in realistic surveillance scenarios.

## B. Detection and Multi-Object Tracking

The first stage of the proposed pipeline, applied independently to each camera stream, consists of detecting luggage instances and associating them over time to form tracks.

1) Detection: Luggage instances are detected in each frame using an object detector that provides bounding boxes (BBox) and confidence scores $s _ { c }$ for each candidate object. Only detections with $s _ { c }$ above a predefined threshold $\theta _ { c }$ are retained to reduce false positives. Then, the asymmetric tracking strategy reflects realistic surveillance conditions, where controlled environments (e.g., conveyor systems) allow multi-view association, while unconstrained environments lead to fragmented and incomplete tracks. This difference motivates the use of scenespecific representation strategies in the subsequent stages of the pipeline.

![](images/6f196c34ba649973346079669bf29bc5e9528939425ce4c462c66114e491f85d.jpg)  
Fig. 1: ReID Pipeline Overview

2) Tracking in the Gallery Scene: The gallery scene exhibits a well-controlled environment in which luggage moves on a conveyor belt, at a controlled and constant speed. This results in predictable and consistent motion across cameras, free from any obstruction, enabling reliable association of trajectories between different views.

Leveraging this property, detections corresponding to the same luggage instance are associated across cameras based on spatio-temporal consistency. This allows the construction of cross-camera tracks, where each luggage instance is observed sequentially by multiple cameras.

3) Tracking in the Query Scene: In contrast, in the query scene, luggage is carried by people, resulting in less predictable motion patterns, frequent occlusions, and partial visibility across cameras. As a consequence, reliable cross-camera association is not feasible.

Tracking is therefore performed independently for each camera using a tracking-by-detection approach, where detections are associated across consecutive frames based on spatial proximity and appearance similarity. This results in a set of per-camera tracks, each corresponding to a sequence of observations of a luggage instance over time. Due to occlusions and missed detections, these tracks may be fragmented.

## C. Feature Extraction and Aggregation

Due to the unconstrained nature of the query scene, individual observations of a luggage instance are often affected by occlusions, motion blur, and imperfect detections. As a result, directly aggregating all observations may introduce noise and degrade the quality of the resulting multi-view and multi-shot representation. For each detected instance, we first extract a visual feature representation using a deep convolutional neural network. More precisely, given a BBox, the corresponding image region is cropped, resized to a fixed resolution, and passed through the network to obtain a feature vector $f \in \mathbb { R } ^ { d }$ where d is the output dimension of the network.

To construct robust query representations, we aggregate features at the track level. Given a track $\tau = x _ { 1 } , x _ { 2 } , \ldots , x _ { T } ,$ where $x _ { i }$ is the $i ^ { t h }$ detected instance of a track and $T$ is the number of instances in this track, we first discard unreliable detections based on their confidence scores so that the remaining satisfy $s _ { c _ { i } } \geq \theta _ { c }$

From the remaining observations, we extract feature vectors $\left( f _ { 1 } , f _ { 2 } , \ldots , f _ { K } \right)$ , where $K \leq T$ . The final multi-shot query representation $F _ { q }$ is then computed as the average of these features: $\begin{array} { r } { F _ { q } = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } f _ { i } } \end{array}$

This simple aggregation strategy leverages temporal redundancy within trajectories while ensuring that only reliable observations contribute to the final representation. By filtering out low-confidence detections prior to aggregation, the method reduces the impact of occlusions and noise without requiring additional weighting mechanisms.

In contrast, the gallery scene provides structured multiview observations with limited occlusion. Therefore, no temporal aggregation is required, and each gallery instance $\gamma$ is composed of one shot from each camera, such as $\gamma =$ $x _ { 1 } , x _ { 2 } , \ldots , x _ { \# G C a m s } ,$ where #GCams is the number of cameras in the gallery scene and $x _ { i }$ is the detected instance of the $i ^ { t h }$ gallery camera.

## D. Cross-Scene Matching

The final stage of the pipeline consists of matching query representations against gallery representations to identify corresponding luggage instances across scenes. Given a query track, we obtain a single aggregated feature vector $F _ { q }$ using the proposed temporal aggregation strategy. In contrast, each gallery instance is represented by a set of multi-view features $( F _ { g } ^ { 1 } , F _ { g } ^ { 2 } , . . . , F _ { g } ^ { \# G C a m s } )$ . During matching, we compute the distance between the query feature $F _ { q }$ and each gallery feature independently. This results in a set of distances between the query and all #GCams gallery features. All gallery features are then ranked in ascending order of their distance to the query. Since multiple gallery features may correspond to the same luggage ID (by multi-view association), the final ranking at the identity level is obtained by considering the best-ranked occurrence of each ID. This formulation allows the matching process to naturally exploit complementary viewpoints in the gallery: even if some observations are less informative or affected by occlusions, other views of the same luggage instance can still yield a low distance and lead to correct identification.

## IV. EXPERIMENTS

## A. Dataset Construction

To evaluate the proposed approach under realistic surveillance conditions, we construct a multi-camera luggage ReID dataset from video footage from a real airport, in the customs area (cf. Fig. 2). The dataset consists of two disjoint scenes, referred to as the gallery and query scenes, each captured by multiple cameras with different viewpoints.

While the gallery scene captures luggage items moving on a conveyor belt immediately after plane landing, the query scene corresponds to passengers leaving the customs area through a corridor while carrying their luggage

The acquisition system consists of a multi-camera configuration with 5 cameras in the gallery scene and 6 cameras in the query scene. In both environments, cameras are positioned around the area of interest and oriented toward a common region to maximize coverage of luggage instances. To establish cross-scene correspondences, we manually annotate matching luggage instances between the query and gallery scenes by selecting consistent trajectories.

The resulting dataset contains a total of 116 bags, with multiple observations of each luggage instance across different cameras and time steps, naturally including variations in viewpoint, illumination, and partial occlusions. Overall, the dataset contains 579 gallery tracks and 745 query tracks, with approximately 18 images available per track on average. This setup enables the evaluation of multi-view aggregation strategies in realistic conditions.

## B. Implementation Details

Object detection is performed using the CoDETR detector [23] with publicly available pretrained weights. This object detector was selected for its strong general object detection performance and its ability to handle complex scenes with multiple overlapping objects.For each luggage instance of the gallery scene, one representative frame is selected per camera by choosing the observation where the luggage instance is closest to the center of the field of view, resulting in a fixed set of 5 images (corresponding to the 5 cameras) that form a structured multi-view representation. For each query track, feature embeddings are extracted and aggregated over time using the proposed aggregation strategy. For each detected instance, visual features are extracted using the TransReID algorithm [24] that is pretrained on the large-scale luggage dataset MVB [22] and fine-tuned on our constructed dataset. Input images are resized to a fixed resolution of 256\*256 pixels, and standard data augmentation techniques are applied during training [25], [26]. During inference, each query track is represented by a single aggregated feature vector, while each gallery instance is represented by a set of multi-view features. Query and gallery representations are compared with Euclidean distance.

## C. Results

We evaluate the proposed approach using standard ReID metrics, including cumulative matching characteristics (CMC) at rank-1, rank-5, and rank-10, as well as mean Average Precision (mAP). To provide a comprehensive assessment, we adopt a bidirectional evaluation protocol and report results for both query-to-gallery (Q→G), for the recognition of every query bag, and gallery-to-query (G→Q) matching, for the recall of every gallery bag.

To assess the contribution of each component of the pipeline, we compare the proposed method with several configurations. First, we consider a random single-frame baseline, where one observation per track is selected uniformly at random. To ensure stability, results for this baseline are averaged over ten runs. Second, we evaluate a score-based singleframe variant, where the observation with the highest detection confidence is selected for each track. Third, we consider a variant that uses all available observations per track without applying any confidence-based filtering, followed by feature aggregation. Lastly, we show the results of the proposed approach, that is when considering only the K most confident detections, based on detection scores, and aggregating them. Here K = 5, an extensive study on this parameter is presented shortly after.

<table><tr><td colspan="3"></td><td colspan="4">Q→G</td></tr><tr><td>Variant</td><td>K</td><td>Conf.</td><td>R@1</td><td>R@5</td><td>R@10</td><td>mAP</td></tr><tr><td>Random frame</td><td>1</td><td>一</td><td>0.742</td><td>0.861</td><td>0.903</td><td>0.732± 0.007</td></tr><tr><td>Most confident</td><td>1</td><td>√</td><td>0.794</td><td>0.899</td><td>0.935</td><td>0.775</td></tr><tr><td>All frames</td><td>all</td><td>-</td><td>0.792</td><td>0.874</td><td>0.914</td><td>0.769</td></tr><tr><td>K-most confident</td><td>5</td><td>√</td><td>0.812</td><td>0.903</td><td>0.931</td><td>0.784</td></tr><tr><td></td><td></td><td></td><td colspan="4">G→Q</td></tr><tr><td>Variant</td><td>K</td><td>Conf.</td><td>R@1</td><td>R@5</td><td>R@10</td><td>mAP</td></tr><tr><td>Random frame</td><td>1</td><td>-</td><td>0.784</td><td>0.904</td><td>0.931</td><td>0.705± 0.005</td></tr><tr><td>Most confident</td><td>1</td><td>√</td><td>0.826</td><td>0.924</td><td>0.963</td><td>0.757</td></tr><tr><td>All frames</td><td>all</td><td>-</td><td>0.831</td><td>0.917</td><td>0.934</td><td>0.756</td></tr><tr><td>K-most confident</td><td>5</td><td>√</td><td>0.837</td><td>0.921</td><td>0.942</td><td>0.767</td></tr></table>

TABLE I: Variants comparison results

As shown in Tab. I, the random single-frame baseline yields the lowest performance, highlighting the limitations of relying on arbitrary observations. Selecting the highest-confidence frame improves results, confirming that detection confidence is a reliable indicator of observation quality even for ReID.

![](images/a43385d4b8ddc7614efff4b6ae1e931e0d38ec9a979b45605df41fd1900031f8.jpg)

![](images/4d8b5054c9aa0bc38e53b0ea138bb7f39388f96bdcb65cdf8a15bbc698aa1627.jpg)  
Fig. 2: Visualization of a query with ReID ranking

Using all frames without filtering further degrades performance compared to the score-based baseline, indicating that low-confidence detections introduce noise into the aggregated representation. In contrast, the proposed approach, which combines confidence-based filtering with feature aggregation, achieves the best performance. This demonstrates that removing unreliable observations prior to aggregation is critical for constructing robust representations.

We observe that the score-based single-frame baseline achieves slightly higher rank-10 performance than the proposed method. This may be because selecting the highestconfidence frame preserves a particularly discriminative observation that still appears among the top-ranked candidates, even if not ranked first. However, the proposed approach consistently outperforms the baseline in rank-1 and mAP, which are more important metrics for practical deployment.

In real-world applications, ReID systems are often integrated into automated pipelines where decisions rely on the top-ranked match rather than a shortlist of candidates. In this context, improving rank-1 accuracy is essential, as it directly impacts the reliability of automatic identification. Similarly, higher mAP reflects better overall ranking quality across all queries. Therefore, despite slightly lower rank-10 performance, the proposed method provides a more robust and practically relevant solution.

Overall, these results validate the design choices of the proposed pipeline and emphasize the importance of both filtering and aggregation for handling noisy and partially occluded observations in realistic surveillance scenarios.

Lastly, all experiments were conducted on a single NVIDIA TITAN RTX GPU. Inference time of detection and tracking is 6.1s for the synchronized images of one scene’s cameras whereas ReID runtime is 30ms for a single query.

## D. Ablation Study

We conduct a series of ablation experiments to evaluate the contribution of each component of the proposed pipeline, including multi-camera configurations, temporal aggregation strategies, and dataset characteristics.

1) Impact of Multi-Camera Configurations: We first investigate the impact of the number and selection of cameras in both query and gallery scenes. To this end, we evaluate different configurations, including single-camera setups, camera pairs, and the full multi-camera setting.

To simplify the presentation of the different camera configurations, we introduce the following notation. Individual cameras are identified based on their relative position in the scene. For instance, $G C a m _ { F R }$ refers to the gallery camera located at the front-right of the scene, while similar notation is used for other viewpoints (e.g., front-left, lateral-left, lateralright, rear-left, rear-right). The same convention applies to the query scene cameras (front-left, front-right, lateral-left, lateral-right, rear). We denote $B _ { Q }$ and $B _ { G }$ the number of bags detected by the query/gallery scenes’ cameras in a specific configuration, while #QCams and #GCams refer to the number of cameras used for an experiment.

Results show that performance varies significantly across cameras, indicating that some viewpoints provide more discriminative information than others. Single-camera configurations generally reduce performance due to limited visibility and missed detections. In particular, Tab. II highlights the asymmetric impact of camera setups in query and gallery scenes. Using a single query camera can still provide relatively strong query-to-gallery performance, as some viewpoints remain discriminative, but it noticeably degrades gallery-toquery results because of missed detections and incomplete luggage coverage. Fig. 3 illustrates ReID errors with a single camera, while combining all cameras yields correct matches. Similarly, restricting the gallery to one camera significantly lowers query-to-gallery performance, especially rank-1 accuracy (Tab. III), since limited gallery views reduce the chances of capturing discriminative observations.

<table><tr><td colspan="3">#GCams = 5</td><td colspan="4">Q→G</td></tr><tr><td>Camera</td><td>BG</td><td>BQ</td><td>R@1</td><td>R@5</td><td>R@10</td><td>mAP</td></tr><tr><td>QCamFL</td><td>116</td><td>100</td><td>0.820</td><td>0.900</td><td>0.930</td><td>0.792</td></tr><tr><td>QCamFR</td><td>116</td><td>104</td><td>0.798</td><td>0.933</td><td>0.962</td><td>0.792</td></tr><tr><td>QCamLL</td><td>116</td><td>106</td><td>0.783</td><td>0.868</td><td>0.906</td><td>0.771</td></tr><tr><td>QCamLR</td><td>116</td><td>110</td><td>0.791</td><td>0.882</td><td>0.945</td><td>0.775</td></tr><tr><td>QCamRL</td><td>116</td><td>98</td><td>0.776</td><td>0.878</td><td>0.908</td><td>0.739</td></tr><tr><td>QCamRR</td><td>116</td><td>97</td><td>0.794</td><td>0.928</td><td>0.938</td><td>0.773</td></tr><tr><td colspan="3">#GCams = 5</td><td colspan="4">G→Q</td></tr><tr><td>Camera</td><td>BG</td><td>BQ</td><td>R@1</td><td>R@5</td><td>R@10</td><td>mAP</td></tr><tr><td>QCamFL</td><td>116</td><td>100</td><td>0.661</td><td>0.789</td><td>0.812</td><td>0.720</td></tr><tr><td>QCamFR</td><td>116</td><td>104</td><td>0.684</td><td>0.839</td><td>0.870</td><td>0.756</td></tr><tr><td>QCamLL</td><td>116</td><td>106</td><td>0.680</td><td>0.822</td><td>0.853</td><td>0.749</td></tr><tr><td>QCamLR</td><td>116</td><td>110</td><td>0.710</td><td>0.879</td><td>0.912</td><td>0.787</td></tr><tr><td>QCamRL</td><td>116</td><td>98</td><td>0.617</td><td>0.737</td><td>0.788</td><td>0.674</td></tr><tr><td>QCamRR</td><td>116</td><td>97</td><td>0.611</td><td>0.765</td><td>0.794</td><td>0.683</td></tr></table>

TABLE II: Single camera results @Query : $\# Q C a m s = 1$
<table><tr><td colspan="3">#QCams = 6</td><td colspan="4">Q→G</td></tr><tr><td>Camera</td><td>BG</td><td> $\overline { { \mathbf { B } _ { Q } } }$ </td><td>R@1</td><td>R@5</td><td>R@10</td><td>mAP</td></tr><tr><td>GCamFL</td><td>116</td><td>114</td><td>0.761</td><td>0.917</td><td>0.948</td><td>0.834</td></tr><tr><td>GCamFR</td><td>116</td><td>114</td><td>0.763</td><td>0.915</td><td>0.953</td><td>0.834</td></tr><tr><td>GCamLL</td><td>116</td><td>114</td><td>0.714</td><td>0.896</td><td>0.933</td><td>0.797</td></tr><tr><td>GCamLR</td><td>116</td><td>114</td><td>0.746</td><td>0.899</td><td>0.943</td><td>0.819</td></tr><tr><td>GCamR</td><td>115</td><td>114</td><td>0.714</td><td>0.899</td><td>0.953</td><td>0.799</td></tr><tr><td colspan="3">#QCams = 6</td><td colspan="4">G→Q</td></tr><tr><td>Camera</td><td>BG</td><td>BQ</td><td>R@1</td><td>R@5</td><td>R@10</td><td>mAP</td></tr><tr><td>GCamFL</td><td>116</td><td>114</td><td>0.828</td><td>0.940</td><td>0.966</td><td>0.770</td></tr><tr><td>GCamFR</td><td>116</td><td>114</td><td>0.853</td><td>0.940</td><td>0.974</td><td>0.771</td></tr><tr><td>GCamLL</td><td>116</td><td>114</td><td>0.793</td><td>0.888</td><td>0.957</td><td>0.737</td></tr><tr><td>GCamLR</td><td>116</td><td>114</td><td>0.828</td><td>0.922</td><td>0.957</td><td>0.753</td></tr><tr><td>GCamR</td><td>115</td><td>114</td><td>0.835</td><td>0.939</td><td>0.965</td><td>0.751</td></tr></table>

TABLE III: Single camera results @Gallery : $\# G C a m s = 1$

![](images/8b80e2a9ea8c9f85cb13af33b16a72dd35f2fa610ec47146ae595360ef82c6dc.jpg)  
Fig. 3: Comparison of recall by one camera vs. all cameras

On a side note, Tab. III also reveals that when considering all available query cameras, some luggage instances remain undetected. This is primarily due to low detection confidence in challenging situations, such as bags piled together on trolleys, which leads to missed or unreliable detections. This observation accentuates the inherent difficulty of the task and the impact of detection quality on overall ReID performance.

These findings highlight the importance of multi-camera setups in realistic surveillance environments, where individual cameras may fail to capture complete object information due to occlusions or limited fields of view.

2) Impact of Temporal Aggregation: We next analyze the effect of temporal aggregation in the query scene. Specifically, we vary the number of observations (K-shot) used to construct track-level representations, and compare the impact of having an aggregation or not. We consider different selection strategies, including choosing the top-K observations based on detection confidence or BBox size.

<table><tr><td colspan="3"></td><td colspan="4">Q→G</td><td colspan="4">G→Q</td></tr><tr><td>K</td><td>Aggreg.</td><td>Selection</td><td>R@1</td><td>R@5</td><td>R@10</td><td>mAP</td><td>R@1</td><td>R@5</td><td>R@10</td><td>mAP</td></tr><tr><td>1</td><td>∅</td><td>score</td><td>0.794</td><td>0.899</td><td>0.935</td><td>0.775</td><td>0.826</td><td>0.924</td><td>0.963</td><td>0.757</td></tr><tr><td>1</td><td>∅</td><td>size</td><td>0.782</td><td>0.883</td><td>0.913</td><td>0.768</td><td>0.812</td><td>0.917</td><td>0.947</td><td>0.748</td></tr><tr><td>2</td><td>no</td><td>score</td><td>0.782</td><td>0.890</td><td>0.924</td><td>0.763</td><td>0.831</td><td>0.902</td><td>0.935</td><td>0.733</td></tr><tr><td>2</td><td>no</td><td>size</td><td>0.783</td><td>0.893</td><td>0.922</td><td>0.771</td><td>0.800</td><td>0.910</td><td>0.944</td><td>0.739</td></tr><tr><td>2</td><td>yes</td><td>score</td><td>0.816</td><td>0.901</td><td>0.936</td><td>0.785</td><td>0.814</td><td>0.924</td><td>0.949</td><td>0.755</td></tr><tr><td>2</td><td>yes</td><td>size</td><td>0.799</td><td>0.896</td><td>0.934</td><td>0.782</td><td>0.812</td><td>0.924</td><td>0.953</td><td>0.763</td></tr><tr><td>3</td><td>no</td><td>score</td><td>0.779</td><td>0.887</td><td>0.925</td><td>0.761</td><td>0.824</td><td>0.893</td><td>0.921</td><td>0.724</td></tr><tr><td>3</td><td>no</td><td>size</td><td>0.791</td><td>0.891</td><td>0.924</td><td>0.778</td><td>0.821</td><td>0.902</td><td>0.930</td><td>0.738</td></tr><tr><td>3</td><td>yes</td><td>score</td><td>0.804</td><td>0.903</td><td>0.937</td><td>0.784</td><td>0.817</td><td>0.921</td><td>0.946</td><td>0.761</td></tr><tr><td>3</td><td>yes</td><td>size</td><td>0.809</td><td>0.906</td><td>0.934</td><td>0.796</td><td>0.822</td><td>0.937</td><td>0.954</td><td>0.773</td></tr><tr><td>4</td><td>no</td><td>score</td><td>0.778</td><td>0.890</td><td>0.925</td><td>0.760</td><td>0.822</td><td>0.905</td><td>0.923</td><td>0.719</td></tr><tr><td>4</td><td>no</td><td>size</td><td>0.789</td><td>0.891</td><td>0.923</td><td>0.776</td><td>0.819</td><td>0.907</td><td>0.926</td><td>0.734</td></tr><tr><td>4</td><td>yes</td><td>score</td><td>0.807</td><td>0.898</td><td>0.930</td><td>0.781</td><td>0.815</td><td>0.926</td><td>0.947</td><td>0.759</td></tr><tr><td>4</td><td>yes</td><td>size</td><td>0.812</td><td>0.906</td><td>0.939</td><td>0.796</td><td>0.814</td><td>0.931</td><td>0.951</td><td>0.772</td></tr><tr><td>5</td><td>no</td><td>score</td><td>0.777</td><td>0.892</td><td>0.926</td><td>0.761</td><td>0.828</td><td>0.896</td><td>0.919</td><td>0.718</td></tr><tr><td>5</td><td>no</td><td>size</td><td>0.788</td><td>0.889</td><td>0.921</td><td>0.773</td><td>0.819</td><td>0.910</td><td>0.924</td><td>0.730</td></tr><tr><td>5</td><td>yes</td><td>score</td><td>0.812</td><td>0.903</td><td>0.931</td><td>0.784</td><td>0.837</td><td>0.921</td><td>0.942</td><td>0.767</td></tr><tr><td>5</td><td>yes</td><td>size</td><td>0.815</td><td>0.908</td><td>0.941</td><td>0.794</td><td>0.798</td><td>0.924</td><td>0.949</td><td>0.771</td></tr><tr><td>all</td><td>no</td><td>all</td><td>0.772</td><td>0.882</td><td>0.920</td><td>0.755</td><td>0.835</td><td>0.903</td><td>0.919</td><td>0.686</td></tr><tr><td>all</td><td>yes</td><td>all</td><td>0.819</td><td>0.897</td><td>0.929</td><td>0.789</td><td>0.819</td><td>0.910</td><td>0.946</td><td>0.761</td></tr></table>

TABLE IV: K-shot study

When observations are selected according to the detection confidence score, Tab. IV shows that increasing K without feature aggregation gradually decreases performance, suggesting that lower-quality observations introduce noise. In contrast, feature aggregation consistently improves performance for the same K, as averaging multiple observations yields more robust descriptors. However, using all available observations significantly increases inference time, highlighting a trade-off between accuracy and computational cost.

The same trends are observed when using BBox size as the selection criterion, although performance at low K is generally lower than with detection confidence. This difference becomes less pronounced for larger K, where size-based selection can achieve comparable or slightly better results. This suggests that detection confidence is better for selecting a few reliable observations, while size-based selection may retain more complementary appearance information for larger subsets. Also, selecting observations based on detection confidence provides a good trade-off between performance and robustness. This validates the design choice of filtering lowconfidence detections and averaging the remaining features in our method.

3) Dataset and Domain Analysis: We analyze the impact of dataset characteristics and domain adaptation using 3 datasets: a large public dataset (MVB), our primary custom dataset (CD-1), and a second dataset acquired in a different location with a similar setup (CD-2). Both datasets are split for train (-tr) and test (-te) purposes. We observe that CD-2 appears to be an easier dataset for the ReID task, likely due to its more uniform background (e.g., consistent floor coloring), which reduces visual clutter and may also correlate with more stable lighting conditions and fewer distractors compared to CD-1.

We first evaluate on CD-1-te a model pretrained on MVB. The results from Tab. V show limited performance, indicating a significant domain gap between the external dataset and the target environment. Fine-tuning the same model on CD-1-tr

![](images/e2f5e344ff0aa4055715f18b87f55ea29cf1cce4aa43afe15379029b3b351f85.jpg)  
Fig. 4: Sample of images from each dataset

leads to substantial improvement, confirming the importance of adapting feature representation to target domain.
<table><tr><td colspan="2">Model used</td><td colspan="4">Q→G</td></tr><tr><td>Training</td><td>Test</td><td>R@1</td><td>R@5</td><td>R@10</td><td>mAP</td></tr><tr><td>MVB</td><td>CD-1-te</td><td>0.611</td><td>0.797</td><td>0.846</td><td>0.559</td></tr><tr><td>MVB + finetuned on CD-1-tr</td><td>CD-1-te</td><td>0.794</td><td>0.899</td><td>0.935</td><td>0.775</td></tr><tr><td>MVB</td><td>CD-2-te</td><td>0.663</td><td>0.848</td><td>0.902</td><td>0.646</td></tr><tr><td>MVB + finetuned on CD-1-tr</td><td>CD-2-te</td><td>0.759</td><td>0.891</td><td>0.938</td><td>0.727</td></tr><tr><td>MVB + finetuned on CD-1-tr &amp; CD-2-tr</td><td>CD-2-te</td><td>0.831</td><td>0.941</td><td>0.985</td><td>0.830</td></tr><tr><td></td><td></td><td colspan="4"></td></tr><tr><td>Training</td><td>Test</td><td>R@1</td><td>R@5</td><td>G→Q R@10</td><td>mAP</td></tr><tr><td>MVB</td><td>CD-1-te</td><td>0.656</td><td>0.843</td><td>0.907</td><td>0.558</td></tr><tr><td>MVB + finetuned on CD-1-tr</td><td>CD-1-te</td><td>0.826</td><td>0.924</td><td>0.963</td><td>0.757</td></tr><tr><td>MVB</td><td>CD-2-te</td><td>0.707</td><td>0.929</td><td>0.962</td><td>0.621</td></tr><tr><td>MVB + finetuned on CD-1-tr</td><td>CD-2-te</td><td>0.801</td><td>0.959</td><td>0.982</td><td>0.712</td></tr><tr><td>MVB + finetuned on CD-1-tr &amp; CD-2-tr</td><td>CD-2-te</td><td>0.819</td><td>0.965</td><td>0.987</td><td>0.866</td></tr></table>

TABLE V: Dataset analysis results

We then extend this analysis to CD-2 to evaluate crossdataset generalization. A model pretrained on MVB and tested directly on CD-2-te again yields relatively low performance, highlighting the difficulty of generalizing across environments. Fine-tuning on CD-1-tr before testing on CD-2-te improves the results, demonstrating partial transferability between datasets with similar acquisition setups. Finally, further fine-tuning on CD-2-tr leads to the best performance, confirming that dataset-specific adaptation remains necessary even when the acquisition conditions are similar. Overall, these experiments highlight the presence of a domain gap not only between large-scale external datasets and real-world surveillance data, but also between datasets collected in different environments, despite looking very similar (cf. Fig. 4). This emphasizes the importance of incremental adaptation when deploying ReID systems in new settings.

## V. CONCLUSION AND FUTURE WORK

This work focused on studying how detection, tracking, feature selection and aggregation strategies can be combined and adapted in a pipeline for multi-camera luggage ReID in realistic real-world scenarios and environments. Through extensive experiments and ablation studies, we analyzed the impact of camera configurations, temporal aggregation, shot selection and dataset adaptation. The results highlight the importance of multi-camera observations and feature aggregation for improving robustness against occlusions and missed detections. Future work will investigate unsupervised domain adaptation to maintain performance while deploying in new environments without any additional annotation cost.

## ACKNOWLEDGMENT

This work was made possible by the use of FactoryIA Supercomputer (funded by the Ile-de-France Regional Council) and partly funded by the European Union (Project 101121309—BAG-INTEL). Views and opinions expressed are, however, those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.

## REFERENCES

[1] L. Zheng, Y. Yang, and A. G. Hauptmann, “Person re-identification: Past, present and future,” arXiv preprint arXiv:1610.02984, 2016.

[2] M. Ye, J. Shen, G. Lin, T. Xiang, L. Shao, and S. C. Hoi, “Deep learning for person re-identification: A survey and outlook,” IEEE transactions on pattern analysis and machine intelligence, vol. 44, no. 6, pp. 2872– 2893, 2021.

[3] R. Kuma, E. Weill, F. Aghdasi, and P. Sriram, “Vehicle re-identification: an efficient baseline using triplet embedding,” in 2019 International Joint Conference on Neural Networks (IJCNN). IEEE, 2019, pp. 1–9.

[4] H. Yang, X. Chu, L. Zhang, Y. Sun, D. Li, and S. J. Maybank, “Quadnet: Quadruplet loss for multi-view learning in baggage re-identification,” Pattern Recognition, vol. 126, p. 108546, 2022.

[5] Y. Liu, H. Zhang, X. Yang, S. Zhao, and J. Zhang, “Cdr-carnet: Baggage re-identification based on cross-domain robust features and cameraaware re-ranking,” Computers & Graphics, p. 104377, 2025.

[6] F. Zhao, G. Xu, K. Xia, Y. Li, Y. Du, and L. Wang, “Luggage reidentification via mask-guided spatial attention network,” in 2025 China Automation Congress (CAC). IEEE, 2025, pp. 7839–7844.

[7] P. L. Mazzeo, C. Libetta, P. Spagnolo, and C. Distante, “A siamese neural network for non-invasive baggage re-identification,” Journal of Imaging, vol. 6, no. 11, p. 126, 2020.

[8] W. Wang, X. Zhang, B. Song, and F. Ahmad, “A model for luggage re-identification based on the network ensemble,” in Proceedings of the 2020 9th International Conference on Computing and Pattern Recognition, 2020, pp. 36–42.

[9] L. Leal-Taixe, A. Milan, I. Reid, S. Roth, and K. Schindler, “Motchal-´ lenge 2015: Towards a benchmark for multi-target tracking,” arXiv preprint arXiv:1504.01942, 2015.

[10] N. Wojke, A. Bewley, and D. Paulus, “Simple online and realtime tracking with a deep association metric,” in 2017 IEEE international conference on image processing (ICIP). IEEE, 2017, pp. 3645–3649.

[11] A. Bewley, Z. Ge, L. Ott, F. Ramos, and B. Upcroft, “Simple online and realtime tracking,” in 2016 IEEE international conference on image processing (ICIP). Ieee, 2016, pp. 3464–3468.

[12] Y. Zhang, P. Sun, Y. Jiang, D. Yu, F. Weng, Z. Yuan, P. Luo, W. Liu, and X. Wang, “Bytetrack: Multi-object tracking by associating every detection box,” in European conference on computer vision. Springer, 2022, pp. 1–21.

[13] Z. Wang, L. Zheng, Y. Liu, Y. Li, and S. Wang, “Towards real-time multi-object tracking,” in European conference on computer vision. Springer, 2020, pp. 107–122.

[14] Y. Zhang, C. Wang, X. Wang, W. Zeng, and W. Liu, “Fairmot: On the fairness of detection and re-identification in multiple object tracking,” International journal of computer vision, vol. 129, no. 11, pp. 3069– 3087, 2021.

[15] IDEMIA, “Alix: Ai-powered luggage identification solution,” https://www.idemia.com/luggage-identification-solution, 2023, online.

[16] Y. Sun, L. Zheng, Y. Yang, Q. Tian, and S. Wang, “Beyond part models: Person retrieval with refined part pooling (and a strong convolutional baseline),” in Proceedings of the European conference on computer vision (ECCV), 2018, pp. 480–496.

[17] J. Xu, R. Zhao, F. Zhu, H. Wang, and W. Ouyang, “Attention-aware compositional network for person re-identification,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 2119–2128.

[18] H. Liu, Z. Jie, K. Jayashree, M. Qi, J. Jiang, S. Yan, and J. Feng, “Videobased person re-identification with accumulative motion context,” IEEE transactions on circuits and systemsfor video technology, vol. 28, no. 10, pp. 2788–2802, 2017.

[19] J. Miao, Y. Wu, P. Liu, Y. Ding, and Y. Yang, “Pose-guided feature alignment for occluded person re-identification,” in Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 542– 551.

[20] Y. Li, J. He, T. Zhang, X. Liu, Y. Zhang, and F. Wu, “Diverse part discovery: Occluded person re-identification with part-aware transformer,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 2898–2907.

[21] W. Lin, Y. Li, X. Yang, P. Peng, and J. Xing, “Multi-view learning for vehicle re-identification,” in 2019 IEEE international conference on multimedia and expo (ICME). IEEE, 2019, pp. 832–837.

[22] Z. Zhang, D. Li, J. Wu, Y. Sun, and L. Zhang, “Mvb: A large-scale dataset for baggage re-identification and merged siamese networks,” in Chinese Conference on Pattern Recognition and Computer Vision (PRCV). Springer, 2019, pp. 84–96.

[23] Z. Zong, G. Song, and Y. Liu, “Detrs with collaborative hybrid assignments training,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 6748–6758.

[24] S. He, H. Luo, P. Wang, F. Wang, H. Li, and W. Jiang, “Transreid: Transformer-based object re-identification,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 15 013–15 022.

[25] H. Luo, Y. Gu, X. Liao, S. Lai, and W. Jiang, “Bag of tricks and a strong baseline for deep person re-identification,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, 2019, pp. 0–0.

[26] N. McLaughlin, J. M. Del Rincon, and P. Miller, “Data-augmentation for reducing dataset bias in person re-identification,” in 2015 12th IEEE International conference on advanced video and signal based surveillance (AVSS). IEEE, 2015, pp. 1–6.