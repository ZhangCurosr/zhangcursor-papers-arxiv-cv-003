# Trainspotting: Learning Object-Centric 3D Reconstruction from Found Photos

Hanyu Chen Noah Snavely Cornell University

![](images/4daab8628b1c5019dc16e7a3c538873a5e1c1f1ddf92a583ef816f5f0b6ee1b3.jpg)  
Figure 1. One object over one century. Wikimedia Commons holds 727 photos of the locomotive Flying Scotsman, taken from the 1920s to 2026 by 187 photographers using 123 camera models. The twelve photos on the left are labeled with their year and camera; our recovered sparse point cloud and camera poses are shown on the right. TRAINSPOTTING contains 12,068 such objects reconstructed from 311,886 photos from Wikimedia Commons. We use this dataset to substantially improve feedforward 3D reconstruction models on in-the-wild object-centric photo collections.

## Abstract

State-of-the-art feedforward reconstruction models perform poorly on in-the-wild object-centric photo collections. They are trained largely on single-session object captures whose images vary in viewpoint while the object and its environment remain nearlyfixed. Public photo archives contain a different form ofmulti-view data. They hold found photos ofthe same object or design: photos taken independently by many people across years, environments, cameras, and sometimes physical units. In this work, we introduce TRAINSPOTTING, a dataset of 12,068 objects and 311,886 photos from Wikimedia Commons. We reconstruct each object to recover camera poses, multi-view stereo depth, and object masks. We categorize the objects into three regimes: objects fixed at one site, objects that move between sites, and designs shared by many physical units. Evaluating recent feedforward reconstruction models on TRAINSPOTTING shows that theyfail most severely in the last two regimes. Fine-tuning VGGT-Ω on TRAINSPOTTING significantly improves camera pose, depth, and 3D geometry

on held-out objects. We observe similar improvements on the independent NAVI in-the-wild benchmark, which contains photos of each object in different environments. Performance on conventional benchmarks remains largely unchanged.

## 1. Introduction

Choose life. Choose a train, a statue, a pipe organ. Choose a 1967 Volkswagen Beetle, a K6 red telephone box in London. We chose not to choose. We reconstructed them all.

Found photos of objects like those listed above accumulate in public archives such as Wikimedia Commons over many years. Feedforward reconstruction models [48, 49, 52], however, obtain most of their object-level training data from datasets that record each object in a single video or capture session [28, 31, 55]. These captures densely sample nearby views while the object and its environment remain nearly fixed.

The photos of an object on Wikimedia Commons follow no shared capture plan. Their viewpoints are unevenly sampled, and their environment may change dramatically from one photo to the next (e.g., a ship on the open seas vs. the same ship in port). An individual object may be repainted or rebuilt over time, and photos of a mass-produced product, like a specific make and model of car, may depict different physical instances. Datasets of landmarks and other places have been built from similar inthe-wild Internet photo collections [24, 38, 42]. However, these scene-level photo collections depict a largely consistent physical space, whereas photos of an object often share only the object itself in many different contexts and times. Figure 1 shows an example of such photos, depicting the locomotive Flying Scotsman captured over a century in many different locations.

![](images/0f7e6d64fbdb80cecb2dbc412b8c2c7eb0565e0ee1ceb161b22e366a6d55edef.jpg)  
Figure 2. VGGT-Ω fails on the Buick Riviera. We visualize the camera poses and point map predicted by VGGT-Ω from sixteen photos of the Buick Riviera (multiple physical instances of the same car design). The views are misregistered and the car is reconstructed as scattered fragments. Figure 6 shows the reconstruction of our fine-tuned model from the same photos.

Standard reconstruction pipelines are unreliable in such inthe-wild settings, and so are feedforward reconstruction models, which are largely trained on their output (Figure 2). Classical structure from motion relies on repeatable correspondences that agree with one rigid scene [35]. A changing environment provides few such correspondences, and the object may occupy only a small part of each photo. Changes to an individual object over time and differences between physical units further weaken geometric consistency. Together, these factors can result in incorrect reconstructions, so building a dataset from these photos requires identifying the photos of the intended object, matching them robustly, and verifying the recovered geometry.

In this work, we propose a dataset and an evaluation protocol for feedforward reconstruction of objects from in-the-wild photo collections. To build this dataset, we collect candidate categories from Wikidata and Wikimedia Commons, such as Cristo Redentor, the Hubble Space Telescope, and the Ford Mustang. We use a large language model (LLM) to retain the categories that depict a rigid physical object or design (Section 3). To address the failures of standard pipelines on these collections, we propose a new object reconstruction pipeline, combining dense learned matching with geometric verification of the recovered camera poses and depth maps (Section 4). The result of our pipeline is TRAINSPOTTING, an in-the-wild object-centric dataset that contains 12,068 objects and 311,886 photos from Wikimedia Commons paired with camera poses, MVS depth, and object masks. An example reconstruction of

Table 1. Real-image datasets for 3D reconstruction tasks. Datasets above the line record each object or scene in a single capture within minutes, while datasets below contain photos of each object or scene taken by many people spanning days to years. NAVI and Pix3D align images to 3D models by hand; all other datasets recover poses with structure from motion.
<table><tr><td>Dataset</td><td>Type</td><td>Count</td><td>Images</td><td>Span</td><td>Depth</td></tr><tr><td>OmniObject3D [53]</td><td>objects</td><td>6k</td><td>1.2M</td><td>minutes</td><td></td></tr><tr><td>CO3D [31]</td><td>objects</td><td>19k</td><td>1.5M</td><td>minutes</td><td>MVS</td></tr><tr><td>MVImgNet2.0 [54]</td><td>objects</td><td>520k</td><td>15.5M</td><td>minutes</td><td></td></tr><tr><td>uCO3D [28]</td><td>objects</td><td>170k</td><td>34M</td><td>minutes</td><td>mono.</td></tr><tr><td>NAVI [19]</td><td>objects</td><td>36</td><td>10.5k</td><td>days</td><td>scan</td></tr><tr><td>Pix3D [40]</td><td>objects</td><td>395</td><td>10k</td><td>days/years</td><td>scan/CAD</td></tr><tr><td>MegaDepth [24]</td><td>scenes</td><td>196</td><td>130k</td><td>years</td><td>MVS</td></tr><tr><td>MegaDepth-X [23]</td><td>scenes</td><td>1.9k</td><td>440k</td><td>years</td><td>MVS</td></tr><tr><td>MegaScenes [42]</td><td>scenes</td><td>80k</td><td>2M</td><td>years</td><td></td></tr><tr><td>TRAINSPOTTING (ours)</td><td>objects</td><td>12k</td><td>312k</td><td>years</td><td>MVS</td></tr></table>

the Flying Scotsman is shown on the right of Figure 1.

We evaluate recent feedforward reconstruction models on TRAINSPOTTING and find that they perform substantially worse on these in-the-wild collections than on conventional benchmarks. Therefore, we fine-tune one such model, VGGT-Ω, on TRAINSPOTTING and show that the resulting model, SPOTTER, significantly improves camera pose, depth, and 3D geometry on these objects, raising camera-pose AUC@30 from 25 to 81 on held-out objects and from 19 to 76 on NAVI in-the-wild [19], an independent benchmark of objects photographed across different environments, while performance on conventional benchmarks is largely retained. Together, these results suggest that TRAINSPOTTING can help feedforward reconstruction models generalize to the real-world diversity of object photos.

## 2. Related Work

3D datasets. 3D datasets have grown rapidly in scale. Synthetic datasets such as Objaverse [9, 10] and ShapeNet [7] provide textured 3D models. Datasets built with 3D scanners include DTU [1], Google Scanned Objects [12], and OmniObject3D [53], and contain at most a few thousand objects. Meanwhile, CO3D [31], MVImgNet [54, 62], and uCO3D [28] reach a larger scale by crowdsourcing videos of objects and annotating them with 3D information via structure from motion. In such datasets, each object is recorded in one coordinated capture [28, 31, 55, 62]. ScanNet [8], ScanNet++ [61], and DL3DV [27] record scenes in the same way. Few datasets photograph objects in different environments. PASCAL3D+ [56], ObjectNet3D [57], and Pix3D [40] align CAD models to internet images of objects, but the models only approximate the depicted objects. NAVI [19] is the closest dataset to our setting. It photographs each object in different environments with different cameras and aligns each image to a 3D scan of that object. In all of these object-scale datasets, each image is aligned to its corresponding 3D model by hand, and per-image geometry such as depth is rendered from that model rather than reconstructed from the photos. In contrast, TRAINSPOTTING consists of multi-view photo collections of objects taken by different people over many years, with camera poses and depth recovered by an automated reconstruction pipeline, and is much larger than NAVI. Table 1 compares TRAINSPOTTING with other datasets of real images used for 3D reconstruction.

![](images/5688cc86310a978b4b70acb298317b72472547cd98844effcf70a6a8b8ab2272.jpg)  
Figure 3. Examples of capture variation. Each column displays photos of example objects, grouped into the three regimes (fixed, roaming, and instance-pooled) and labeled with their year and camera. Fixed objects stay at one site, roaming objects are single instances that move between sites, and instance-pooled objects are different physical instances of one shared design.

Reconstruction from internet photo collections. Photo Tourism [38] and Building Rome in a Day [2] reconstructed landmarks from unordered photos taken by many photographers. Later datasets use such reconstructions as supervision for learned models. MegaDepth [24] reconstructs about 200 landmarks from photos on Flickr with COLMAP [35], filters the resulting multi-view stereo depth, and uses it to supervise single-view depth prediction. MegaScenes [42] reconstructs 80k Wikimedia Commons categories of scenes with COLMAP using SIFT features. MegaDepth-X [23] selects densely photographed MegaScenes scenes and reconstructs them with MASt3R-SfM [13] and doppelganger disambiguation [58]. All three contain scene-level rather than object-centric reconstructions. Of the 12,068 objects in TRAINSPOTTING, only 1,163 (9.6%) are reconstructed as scenes in MegaScenes and 56 (0.5%) in MegaDepth-X, almost all of them objects permanently installed at one site, such as statues and memorials. Objects that move between sites or exist as many physical units are almost entirely absent from these datasets.

Learned matching and structure from motion. Classical structure from motion (SfM) methods like COLMAP [35] match hand-crafted keypoints and register images incrementally. Super-Point [11] and SuperGlue [34] replace traditional keypoint detection and matching with learned components. LoFTR [39] proposes detector-free matching, predicting matches directly with a transformer. Dense matchers such as DKM [14], RoMa [15], and RoMa v2 [16] predict a warp for every pixel of an image pair, aiming to recover correspondences under large changes in viewpoint and illumination. Several SfM systems, such as Detector-Free SfM [18] and Dense-SfM [21], build on such dense matches to form multi-view tracks and refine the tracks together with the geometry. MASt3R-SfM [13] builds on the matches of MASt3R [22] and jointly aligns all images. VGGSfM [47] learns the entire reconstruction pipeline, predicting tracks, cameras, and points and refining them with differentiable bundle adjustment. We keep the incremental reconstruction of COLMAP and supply it with multi-view tracks built from RoMa v2 matches that pass confidence and geometric checks (Section 4).

Feedforward 3D reconstruction. Feedforward models that regress cameras, depth, and point maps directly from a set of images have become a standard approach to 3D reconstruction [20, 22, 25, 41, 48, 49, 51, 52, 59, 64]. Most of their object-level training data come from single-session captures and renderings of synthetic 3D assets. They are likewise evaluated on benchmarks built with such data. Two recent works, MegaDepth-X [23] and MegaUnScene [65] train and evaluate feedforward models on in-the-wild internet photo collections, but only on scene-level, not object-centric, collections. TRAINSPOTTING is an object-level dataset similarly built from internet photos. We fine-tune VGGT-Ω on this dataset and compare the result with recent feedforward models on TRAINSPOTTING and NAVI in-the-wild (Section 5).

## 3. The TRAINSPOTTING Dataset

TRAINSPOTTING is a dataset of objects reconstructed from Wikimedia Commons photos taken by many people over many years. It contains many types of rigid objects (Table 2), including statues, cannons, pipe organs, telephone boxes, fountains, cameras, ships, aircraft, and locomotives. This section describes how our dataset is curated from Internet sources, and Section 4 describes how we reconstruct the individual objects.

![](images/ddd56b1c0b9e838cee1d8df937d6551d723e8925ffff943a3712d99174c0b167.jpg)  
Figure 4. Reconstruction pipeline for the Flying Scotsman. (a) Object gating retains 570 of 727 photos, with red borders marking excluded photos. (b) A retrieved pair is shown with candidate correspondences. Green correspondences pass the confidence, cycle, and epipolar checks, while dashed red correspondences fail. (c) The structure-from-motion reconstruction registers 496 images, merged from 14 partial reconstructions. View selection keeps 481 of them for multi-view stereo. (d) The same pair shows MVS depth before and after MoGe-2 filtering. (e) Four of the 370 photos in the dataset show MVS depth in color and the selected mask in teal over the image.

## 3.1. Objects from Wikimedia Commons

Wikimedia Commons organizes photos into user-curated categories. Categories for physical subjects are often specific to one object or design, such as Category:LNER Class A3 4472 Flying Scotsman. Each object in TRAINSPOTTING corresponds to one such category, and we call the images filed under it a collection. Many of these categories are linked from Wikidata, the structured knowledge base supporting Wikimedia projects, and 68% of the TRAINSPOTTING objects have a Wikidata item.

Objects represent either individuals or designs. An individual is a single physical unit, such as one hull, one statue, or one numbered locomotive. A design is a shape shared by many physical units, such as a car model or a mass-produced telephone box with the same underlying geometry. We pool all units of a design into one single collection, where individual units within the collection may differ in color, trim, and wear.

## 3.2. Curation at scale

We enumerate candidate categories in two ways. Some classes, such as ships and locomotives, are enumerated from their Wikidata items; others are found by traversing the Wikimedia Commons category graph from root categories, such as Cameras and Pipe organs. The traversal records 155,744 categories after excluding branches that do not depict the object itself, such as photos taken with a camera model, drawings, and replicas of an object.

The traversal also reaches subcategories for places, events, and parts related to an object, which we wish to exclude. Title rules cannot reliably separate these categories from the objects themselves. For example, Cannon Street station is a railway station rather than a cannon. Therefore, we use an LLM to classify each of the recorded categories from its title and the categories above it in the category graph. The classifier assigns three labels: (i) whether the category depicts a rigid physical object, which decides whether it is kept; (ii) whether it names an individual or a design; and (iii) whether the object is permanently installed at one site. Labels (ii) and (iii) determine the regime of each object (Section 3.3). Together with the Wikidata-enumerated classes, the categories kept by the classifier yield 33,884 candidate objects, of which we attempt to reconstruct 17,470 after filtering on the number of photos.

Table 2. Dataset composition. Number of objects in each supercategory. An LLM groups the object labels assigned during curation into 53 categories and 13 supercategories.
<table><tr><td>Supercategory</td><td>Objects</td><td>Supercategory</td><td>Objects</td></tr><tr><td>Road vehicles</td><td>3,341</td><td>Household &amp; personal objects</td><td>353</td></tr><tr><td>Statues &amp; sculpture</td><td>2,160</td><td>Weapons &amp; armour</td><td>342</td></tr><tr><td>Monuments &amp; landmarks</td><td>1,773</td><td>Religious objects</td><td>140</td></tr><tr><td>Aircraft &amp; spacecraft</td><td>1,461</td><td>Street furniture</td><td>130</td></tr><tr><td>Rail vehicles</td><td>1,108</td><td>Machinery &amp; equipment</td><td>121</td></tr><tr><td>Watercraft</td><td>701</td><td>Other</td><td>10</td></tr><tr><td>Musical instruments</td><td>428</td><td>Total</td><td>12,068</td></tr></table>

This curation alone does not filter individual images by their visual content. Collections retain cluttered photos, partial views, and images in which the object occupies a small part of the frame. We rely on the robust downstream reconstruction pipeline to either register these images in 3D or remove them when they cannot be registered correctly (Section 4).

## 3.3. Capture variation

Different kinds of objects are characterized by different forms of variation in context and capture. A statue fixed to its plinth yields photos that share a common environment, but differ in viewpoint, time, and lighting. Photos of a locomotive span different stations with corresponding changes in the environment. A telephone-box design represents many physical units whose exact wear and paint may differ. We denote these distinct categories as fixed, roaming, and instance-pooled (Figure 3). Fixed denotes an individual installed at one site, roaming an individual that moves between sites, and instance-pooled a design shared by many physical instances. The labels follow from

LLM classification (Section 3.2) and are therefore approximate.

## 4. Pseudo-Ground Truth via Reconstruction

A Wikimedia Commons category can contain photos of other objects of the same kind, unrelated subjects, and crowded scenes in which the object is small. In this section, we describe how we robustly reconstruct each object to obtain camera poses, MVS depth, and object masks via object gating, learned dense matching, and geometric verification. Figure 4 follows one object through the reconstruction pipeline.

## 4.1. Matching and structure from motion

Object gating. We use an LLM to assign a text prompt to each object, similar to the classification in Section 3.2. We then use SAM3 [6] with this prompt to identify images in which the object is the main subject. We keep an image when the largest detected box covers at least 10% of the image. Common rejected images include crowd photos, close-ups of details, and images of a different subject filed in the same category.

Dense matching. We compute point matches between photos with the dense matcher RoMa v2 [16] due to its robustness to environment and object variation. In the supplement, we compare this choice with structure from motion from SIFT, SuperPoint with LightGlue, and MASt3R matches on the same images. We embed each image with the retrieval model of MASt3R-SfM [13] and match it against the 20 closest images in the embedding space. For each image pair, RoMa v2 returns a dense mapping of every pixel of one image to the other along with a certainty map. We sample candidate correspondences from these outputs. We keep a correspondence between a point x in one image and $W ( { \pmb x } )$ in the other only if

$$
\begin{array} { r l } & { \sigma ( { \pmb x } ) \geq \tau _ { \sigma } , \qquad \| { W } ^ { \prime } \big ( { W } ( { \pmb x } ) \big ) - { \pmb x } \| \leq \tau _ { \mathrm { c y c } } , } \\ & { \qquad d \big ( { W } ( { \pmb x } ) , { \pmb F } { \pmb x } \big ) \leq \tau _ { \mathrm { e p i } } , } \end{array}\tag{1}
$$

where W and $W ^ { \prime }$ are the forward and backward warps, σ is the certainty that RoMa v2 assigns to the match in both directions, and F is a fundamental matrix estimated for the pair with MAGSAC++ [5]. The three conditions only retain confident, cycle-consistent matches that agree with the estimated epipolar geometry. We merge nearby endpoints of the remaining correspondences into shared keypoints, following prior work on structure from motion from dense matches [17, 18, 21]. These shared keypoints link pairwise matches into multi-view tracks.

Structure from motion. We import these keypoints and matches into COLMAP [35] and perform incremental mapping. We exhaustively merge reconstructed components and keep the largest result as the final reconstruction of the object.

## 4.2. MVS depth and masks

View selection. A mis-registered view, or a view of a different object weakly linked to the reconstruction, is unreliable as ground truth and can corrupt the depth of its neighbors during

![](images/fdb59dcaf56935c69c9579b09e2eea884951142ab7372720d8690b08a5efbc49.jpg)

![](images/c1300fce32386dca957b65d091b113f51a3f70dfe11131d28960a58c7fe3a844.jpg)

![](images/53247b5abcde1d3b3f58061b2fbd9ca84ad8e85756e9e31b25218c176c55d64d.jpg)

![](images/7696e41c00b8166511c2f290d15d0dae9016e20be1bb9633d9aea28530cdfd91.jpg)  
Figure 5. Object statistics by regime. Distributions over TRAINSPOT-TING of the number of photos, the span of capture dates, the number of photographers, and the number of camera models.

MVS. Therefore, we keep a view only if enough of its points have a track length of at least three. We then build a covisibility graph in which two views are connected by an edge weighted by the number of 3D points they share. We split this graph by spectral bisection and discard the smaller subgraph when the conductance of the cut is below 0.15.

Multi-view stereo with a monocular check. We run Cas-DiffMVS [44] on the selected views and keep high-confidence depths. Since MVS can be unreliable on textureless and specular surfaces, we verify these depths against a monocular depth prediction from MoGe-2 [50], as in MegaDepth-X [23], using the spherical filtering strategy of MoGe-2 and ViGeo [50, 63].

Object masks. We select the SAM3 mask that covers the largest share of the verified depth as the target object mask. A photo is included only if enough of the selected mask carries verified depth. A final quality check removes objects with erroneous reconstructions and photos with implausible focal lengths or in which the object is not the dominant subject.

## 4.3. Statistics

We pass 17,470 objects with 1,251,986 photos through our reconstruction pipeline. We report the number of objects and photos that remain after each stage in the supplement. Object gating removes the largest share of photos. Structure from motion succeeds for 85% of the objects and registers 72% of the gated photos.

Dataset. Our final dataset, TRAINSPOTTING, contains 12,068 objects and 311,886 photos. Out of all of its objects, 4,338 are fixed, 2,869 roaming, and 4,861 instance-pooled. Capture dates are available for 84% of the photos. The median capture span is 12 years and 90% of the objects have a capture span of at least one year. A median object has photos by 7 different photographers taken with 6 different camera models. Figure 5 shows these distributions by regime. For a coarse view of the composition, we use an LLM to group the 337 object labels of the dataset into 53 categories and 13 supercategories (Table 2). We release TRAINSPOTTING with its camera poses, MVS depth, and object masks. Every photo is distributed with its

TRAINSPOTTING

![](images/61eb25fce0ba73b771513f8dbc4c6a59340a0bc4c64e73216a36fdfb32f96406.jpg)  
NAVIin-the-wild  
Figure 6. Qualitative comparison. We visualize ground truth and predicted camera poses and point maps from $\pi ^ { 3 }$ [52], Depth Anything 3 [25], VGGT-Ω [49], and SPOTTER on held out objects from TRAINSPOTTING and NAVI in-the-wild [19]. We use 4 or 16 input views fo TRAINSPOTTING and 8 for NAVI in-the-wild. Only the first 4 input views are shown. Triangular markers indicate the directions of the cameras outside of the frame. The baselines misregister the input views and produce duplicated or fragmented geometry, whereas SPOTTER recovers a single consistent object and accurate camera poses.

Wikimedia Commons source, author, and licence.

Split. We partition TRAINSPOTTING by object into a training set of 11,068 objects and 279,319 photos and a held-out set of 1,000 objects and 32,567 photos, so that no photo of a held-out object is seen during training.

## 5. Experiments

## 5.1. Setup

Model and training. We obtain SPOTTER by fine-tuning the released 1B-parameter checkpoint of VGGT-Ω [49] on the training objects of TRAINSPOTTING. We update the camera head and all attention blocks of the transformer backbone. Each training sample contains one object and between 2 and 24 of its photos, chosen so that they form a connected set under covisibility. We consider two photos covisible when at least 5% of the object points of one project inside the object mask of the other.

Objective. Following VGGT, we express the cameras and depths of each sample in the coordinate frame of its first view. We normalize the ground truth by the mean distance of its valid 3D points to the origin. We normalize the predictions in the same way, following DUSt3R [51]. The model is trained with a combined camera, depth, and point loss:

$$
\mathcal { L } { = } 5 \mathcal { L } _ { \mathrm { c a m } } { + } \mathcal { L } _ { \mathrm { d e p t h } } { + } \mathcal { L } _ { \mathrm { p o i n t } } ,\tag{2}
$$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { c a m } } { = } \sum _ { i } \lVert \hat { \pmb { g } } _ { i } - \pmb { g } _ { i } \rVert _ { 1 } , } \end{array}\tag{3}
$$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { d e p t h } } { = } \frac { 1 } { | \mathcal { V } | } { \sum _ { \pmb { x } \in \mathcal { V } } } \big ( 1 { + } \frac { 1 } { D ( \pmb { x } ) } \big ) \big | \hat { D } ( \pmb { x } ) { - } D ( \pmb { x } ) \big | , } \end{array}\tag{4}
$$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { p o i n t } } = \frac { 1 } { | \mathcal { V } | } { \sum _ { \pmb { x } \in \mathcal { V } } } \big \| \hat { \pmb { p } } ( \pmb { x } ) - \pmb { p } ( \pmb { x } ) \big \| _ { 2 } , } \end{array}\tag{5}
$$

where $\hat { \mathbf { g } } _ { i }$ is the predicted camera of view $i ,$ encoded as translation, rotation quaternion, and field of view, V is the set of valid pixels, and ${ \hat { \pmb p } } ( { \pmb x } )$ is the 3D point of pixel x given by the predicted depth and camera. For the depth and point losses, we use plain regression loss without confidence weighting and discard the largest 2% of residuals in each batch.

Replay. To limit forgetting, a fixed 15% of every batch contains samples from BlendedMVS [60], Hypersim [32], and uCO3D [28], supervised with the same loss.

Object masks. Since the background differs between the photos of an in-the-wild collection, the model also predicts an object mask for each view, with a small head trained on the dataset masks on top of the frozen features. This allows restricting the point map output to the object, separating it from its background. The head is trained in a separate stage with the rest of the model frozen and does not affect any of the evaluations below. Details are described in the supplement.

Implementation details. We train with AdamW [29] at a peak learning rate of $1 0 ^ { - 5 }$ and a weight decay of 0.05. We use linear warm-up over the first 5% of steps and a cosine decay. Training runs for 7,500 steps, with 768 images in each step, and takes about 12 hours on four NVIDIA B200 GPUs.

Baselines. We compare SPOTTER with the released checkpoints of VGGT [48], $\pi ^ { 3 }$ [52], Depth Anything 3 [25], MapAnything [20], and VGGT-Ω [49].

## 5.2. Evaluation protocol

Samples. On TRAINSPOTTING, we draw one connected 4-view sample from each held-out object and, when the object has enough photos, one connected 16-view sample, for 1,635 samples from 1,000 objects. On each external dataset, we use one sample of 8 views per scene, equally spaced along the capture sequence when one exists and drawn at random otherwise, since published protocols differ in the number and selection of input views. Samples are drawn with a fixed seed.

Camera pose. We compare the predicted and ground truth cameras on every pair of views in a sample. For each pair, we compute the error of the relative rotation and the direction of the relative translation and report the area under the accuracy curve of the larger of the two errors up to $5 ^ { \circ }$ and 30<sup>◦</sup> (AUC@5 and AUC@30), following PoseDiffusion [46]. All metrics are computed per sample and then averaged over samples.

Depth. We evaluate the predicted depth maps on the pixels with ground-truth depth. For models that only predict point maps, we take depth as the z-coordinate of the point map in each camera’s frame. We scale the predicted depth of all views in a sample by a single median ratio to the ground truth. We report the absolute relative error (AbsRel) and the inlier ratio $\delta { < } 1 . 0 3 $ the fraction of pixels whose predicted and ground-truth depths differ by less than a factor of 1.03.

Point maps. For models that only predict depth, we obtain a point map for each view by unprojecting the predicted depth with the predicted camera. For models that predict point maps, we use the predicted point maps directly. We align the predicted points to the ground-truth points with a similarity transform estimated from corresponding pixels and refined by iterative closest point (ICP). We then report accuracy, the mean distance from each predicted point to the nearest ground-truth point, and completeness, the mean distance from each ground-truth point to the nearest predicted point, following Spann3R [45]. On TRAINSPOTTING, whose reconstructions have an arbitrary scale, we express both in percent of the object extent; on the external datasets, we express them in millimeters.

Table 3. Results on TRAINSPOTTING. Point-map accuracy and completeness are given as a percentage of the object extent after a similarity alignment refined by ICP. Depth AbsRel and inlier ratio $\delta _ { 1 }$ .03 are measured after a single median scale per sample. Camera-pose AUC at $5 ^ { \circ }$ and $3 0 ^ { \circ }$ is computed over all pairs of views (Section 5.2).
<table><tr><td>Method</td><td colspan="2">Point map</td><td colspan="2">Depth</td><td colspan="2">Camera</td></tr><tr><td></td><td>Acc↓</td><td>Comp↓</td><td>AbsRel.↓</td><td> $\delta _ { 1 . 0 3 } \uparrow$ </td><td> $\mathrm { \ A U C _ { 5 } \uparrow }$ </td><td> $\mathrm { { A U C } _ { 3 0 } \uparrow }$ </td></tr><tr><td>VGGT [48]</td><td>2.38</td><td>8.08</td><td>0.160</td><td>32.4</td><td>11.5</td><td>44.1</td></tr><tr><td> $\pi ^ { 3 } \left[ 5 2 \right]$ </td><td>2.33</td><td>6.22</td><td>0.162</td><td>34.7</td><td>12.5</td><td>46.4</td></tr><tr><td>DA3 [25]</td><td>2.22</td><td>12.17</td><td>0.189</td><td>32.1</td><td>10.1</td><td>29.4</td></tr><tr><td>MapAny. [20]</td><td>2.20</td><td>7.75</td><td>0.206</td><td>26.6</td><td>5.3</td><td>33.1</td></tr><tr><td>VGGT-Ω [49]</td><td>1.84</td><td>12.68</td><td>0.201</td><td>28.1</td><td>7.2</td><td>25.0</td></tr><tr><td>SPOTTER</td><td>1.22</td><td>1.15</td><td>0.071</td><td>53.3</td><td>44.2</td><td>81.2</td></tr></table>

Table 4. Results by view count and regime. AUC@30 on the held-out TRAINSPOTTING objects. Each TRAINSPOTTING object contributes one 4-view sample and, if it has at least 16 photos, one 16-view sample. The regime columns pool both sample sizes.
<table><tr><td>Method</td><td>4 views (1000)</td><td>16 views (635)</td><td>Fixed (442)</td><td>Roaming (450)</td><td>Pooled (743)</td></tr><tr><td>VGGT-Ω [49]</td><td>29.4</td><td>18.1</td><td>56.7</td><td>14.9</td><td>12.3</td></tr><tr><td>SPOTTER</td><td>79.9</td><td>83.2</td><td>79.0</td><td>81.0</td><td>82.6</td></tr></table>

## 5.3. Results on TRAINSPOTTING

Table 3 reports evaluation results on TRAINSPOTTING. SPOTTER improves on VGGT-Ω in every metric, most strongly in camera pose and completeness, and substantially outperforms the strongest baseline, $\pi ^ { 3 }$ . Table 4 shows that the difficulty is concentrated in roaming and instance-pooled objects, whose photos differ most in environment. VGGT-Ω performs considerably better on fixed objects than on the other two regimes, whereas SPOTTER performs similarly on all three, and performance further improves with more views. Figure 6 shows reconstructions of four TRAINSPOTTING objects and two NAVI in-the-wild objects. The baselines fail to register views taken in different environments. Their point maps therefore contain duplicated or fragmented copies of the object, and their cameras are scattered. SPOTTER registers the same views consistently and reconstructs a single object.

## 5.4. NAVI in-the-wild

The NAVI dataset [19] contains 36 objects, each with a scanned mesh and camera poses aligned to it, photographed both in multiview captures recorded in a single session and in an in-thewild collection across different environments. As the held-out TRAINSPOTTING results are scored against pseudo-ground truth from our own pipeline, evaluating on NAVI in-the-wild provides an independent measure of accuracy. Under our protocol NAVI yields 35 in-the-wild and 281 multiview samples. Tables 5 and 6 report the results. VGGT-Ω reaches an AUC@30 of 91.6 on the multiview captures but only 18.8 on the in-the-wild collections of the same objects. Fine-tuning on TRAINSPOTTING raises the in-the-wild AUC@30 to 76.0 and improves completeness from 20.4 to 1.9 mm. SPOTTER achieves the best in-the-wild result on every metric except accuracy, on which all models lie within 1 mm of one another. We note that when the views of a prediction are misregistered, the similarity alignment scales the prediction down toward the center of the object, so each predicted point stays close to the surface while most of the surface is left uncovered. Pose failures therefore appear as completeness error rather than accuracy error. On the multiview captures, performance is largely unchanged.

Table 5. Geometry on external benchmarks. Point-map accuracy and completeness are given in millimeters after a similarity alignment refined by ICP. Depth inlier ratio $\delta _ { 1 . 0 3 }$ is measured after a single median scale per sample. Every model is given the same 8 views of each scene. NAVI pho tographs each object in one session (NAVI) and across different environments (NAVI (wild)). The other datasets record each scene in a single capture.
<table><tr><td>Method</td><td colspan="3">NAVI (wild)</td><td colspan="3">NAVI</td><td colspan="3">7-Scenes</td><td colspan="3">NRGBD</td><td colspan="3">DTU</td><td colspan="3">ETH3D</td></tr><tr><td></td><td>Acc↓</td><td>Comp↓</td><td> $\delta _ { 1 . 0 3 } \uparrow$ </td><td>Acc↓</td><td>Comp↓</td><td> $\delta _ { 1 . 0 3 } \uparrow$ </td><td>Acc↓</td><td>Comp↓</td><td> $\delta _ { 1 . 0 3 } \uparrow$ </td><td>Acc↓</td><td>Comp↓</td><td> $\delta _ { 1 . 0 3 } \uparrow$ </td><td>Acc↓</td><td>Comp↓</td><td> $\delta _ { 1 . 0 3 } \uparrow$ </td><td>Acc↓</td><td>Comp↓</td><td> $\delta _ { 1 . 0 3 } \uparrow$ </td></tr><tr><td>VGGT [48]</td><td>3.2</td><td>11.5</td><td>35.0</td><td>2.5</td><td>3.4</td><td>71.5</td><td>25.3</td><td>32.6</td><td>67.5</td><td>17.4</td><td>16.9</td><td>90.9</td><td>1.2</td><td>1.8</td><td>97.4</td><td>291.4</td><td>317.5</td><td>52.6</td></tr><tr><td> $\pi ^ { 3 } \left[ 5 2 \right]$ </td><td>3.7</td><td>9.6</td><td>41.2</td><td>2.2</td><td>3.0</td><td>76.5</td><td>20.0</td><td>26.7</td><td>70.5</td><td>17.9</td><td>17.2</td><td>88.6</td><td>1.2</td><td>1.5</td><td>91.2</td><td>154.7</td><td>129.7</td><td>62.4</td></tr><tr><td>DA3 [25]</td><td>3.3</td><td>16.0</td><td>41.1</td><td>3.3</td><td>5.7</td><td>80.0</td><td>25.4</td><td>35.7</td><td>74.1</td><td>16.4</td><td>15.3</td><td>96.7</td><td>1.7</td><td>1.6</td><td>96.5</td><td>274.3</td><td>270.3</td><td>73.2</td></tr><tr><td>MapAny. [20]</td><td>4.1</td><td>11.0</td><td>29.6</td><td>3.8</td><td>5.8</td><td>56.4</td><td>28.2</td><td>36.8</td><td>58.3</td><td>42.8</td><td>30.2</td><td>69.2</td><td>2.4</td><td>1.9</td><td>37.8</td><td>459.7</td><td>542.0</td><td>51.9</td></tr><tr><td>VGGT-Ω [49]</td><td>3.1</td><td>20.4</td><td>29.7</td><td>2.9</td><td>5.3</td><td>76.1</td><td>21.8</td><td>27.4</td><td>71.9</td><td>13.3</td><td>9.0</td><td>92.4</td><td>1.2</td><td>1.7</td><td>97.1</td><td>73.6</td><td>61.1</td><td>83.5</td></tr><tr><td>SPOTTER</td><td>3.2</td><td>1.9</td><td>43.1</td><td>3.1</td><td>4.0</td><td>74.5</td><td>20.2</td><td>27.4</td><td>75.1</td><td>12.9</td><td>8.3</td><td>92.7</td><td>1.2</td><td>1.6</td><td>95.9</td><td>65.5</td><td>52.9</td><td>82.8</td></tr></table>

Table 6. Camera pose on external benchmarks. Camera-pose AUC at $5 ^ { \circ }$ and $3 0 ^ { \circ }$ is computed over all pairs of the same 8 views of each scene.
<table><tr><td>Method</td><td colspan="2">NAVI (wild)</td><td colspan="2">NAVI</td><td colspan="2">7-Scenes</td><td colspan="2">NRGBD</td><td colspan="2">DTU</td><td colspan="2">ETH3D</td></tr><tr><td></td><td>AUC@5↑</td><td>AUC@30↑</td><td>AUC@5↑</td><td>AUC@30↑</td><td>AUC@5↑</td><td>AUC@30↑</td><td>AUC@5↑</td><td>AUC@30↑</td><td>AUC@5↑</td><td>AUC@30↑</td><td>AUC@5↑</td><td>AUC@30↑</td></tr><tr><td>VGGT [48]</td><td>14.4</td><td>45.5</td><td>53.3</td><td>91.2</td><td>26.6</td><td>81.0</td><td>86.4</td><td>97.6</td><td>98.1</td><td>99.7</td><td>37.6</td><td>71.1</td></tr><tr><td> $\pi ^ { 3 } \left[ 5 2 \right]$ </td><td>13.1</td><td>43.7</td><td>55.8</td><td>91.6</td><td>28.8</td><td>83.3</td><td>86.5</td><td>97.2</td><td>80.0</td><td>96.6</td><td>36.4</td><td>72.9</td></tr><tr><td>DA3 [25]</td><td>12.0</td><td>36.9</td><td>57.2</td><td>91.8</td><td>28.5</td><td>81.5</td><td>91.4</td><td>98.6</td><td>91.0</td><td>98.5</td><td>55.4</td><td>75.4</td></tr><tr><td>MapAny. [20]</td><td>3.3</td><td>24.2</td><td>28.4</td><td>84.0</td><td>18.2</td><td>72.4</td><td>56.5</td><td>90.9</td><td>15.0</td><td>73.4</td><td>33.8</td><td>65.6</td></tr><tr><td>VGGT-Ω [49]</td><td>3.8</td><td>18.8</td><td>55.4</td><td>91.6</td><td>40.8</td><td>87.2</td><td>87.8</td><td>97.9</td><td>91.1</td><td>98.5</td><td>66.9</td><td>91.8</td></tr><tr><td>SPOTTER</td><td>30.7</td><td>76.0</td><td>56.7</td><td>91.9</td><td>38.8</td><td>86.1</td><td>86.5</td><td>97.3</td><td>86.1</td><td>97.7</td><td>65.4</td><td>91.8</td></tr></table>

## 5.5. Conventional benchmarks

We evaluate all models on 7-Scenes [37], Neural RGB-D [4], DTU [1], and ETH3D [36] to measure the effect of fine-tuning on conventional data (Tables 5 and 6). Relative to VGGT-Ω, fine-tuning changes AUC@30 by at most 1.1 points and δ <1.03 by at most 3.2 points on these four datasets.

## 5.6. Ablations

Table 7 compares models trained on subsets of the TRAINSPOT-TING training objects. All models in this comparison, including the reference model trained on all objects, are trained for 10 epochs with the same recipe as SPOTTER. We find that the types of photos used for training matter more than their scale. Training only on fixed objects yields lower accuracy than training on a random subset half its size, both on held-out objects and on NAVI in-the-wild. It is also considerably less accurate on roaming and instance-pooled objects, whereas models trained on those regimes remain close to the reference model when evaluated on fixed objects. This shows that photos across different environments, which fixed objects and existing datasets largely lack, are necessary for generalization to the other regimes and to NAVI in-the-wild.

Table 7. Training-data ablations. AUC@30 of models trained for 10 epochs on subsets of the TRAINSPOTTING training objects. The number of training objects is given in parentheses.
<table><tr><td></td><td colspan="4">TRAINSPOTTING held-out</td><td></td></tr><tr><td>Training data</td><td>All</td><td>Fixed</td><td>Roaming</td><td>Pooled</td><td>NAVI (wild)</td></tr><tr><td>All (11,068)</td><td>80.3</td><td>78.2</td><td>79.5</td><td>82.1</td><td>74.6</td></tr><tr><td>Random 20% (2,193)</td><td>79.7</td><td>77.5</td><td>79.2</td><td>81.3</td><td>72.3</td></tr><tr><td>Fixed (4,038)</td><td>68.5</td><td>78.4</td><td>58.0</td><td>68.9</td><td>63.4</td></tr><tr><td>Roaming (2,603)</td><td>79.0</td><td>76.5</td><td>80.3</td><td>79.7</td><td>65.7</td></tr><tr><td>Pooled (4,427)</td><td>78.8</td><td>75.6</td><td>77.0</td><td>81.7</td><td>73.9</td></tr></table>

## 6. Conclusion

Current feedforward reconstruction models perform poorly on in-the-wild object-centric photo collections, even for objects that they reconstruct accurately from a single capture session. Such in-the-wild collections are abundant in public photo archives, and TRAINSPOTTING shows that they can be turned into training data at scale. Fine-tuning VGGT-Ω on TRAINSPOTTING improves camera pose, depth, and 3D geometry on held-out objects and on the independent NAVI in-the-wild benchmark, while performance on conventional benchmarks remains largely unchanged. The base models had already been trained on large datasets of objects recorded in single capture sessions, and we also show that training on fixed objects alone transfers poorly to roaming and instance-pooled objects, so what these models lacked was variation in how an object is photographed. Beyond training, TRAINSPOTTING provides a benchmark for reconstruction under changing environments, and its reconstructions of objects photographed over decades allow studying how objects and their surroundings change over time.

## References

[1] Henrik Aanæs, Rasmus Ramsbøl Jensen, George Vogiatzis, Engin Tola, and Anders Bjorholm Dahl. Large-scale data for multiple-view stereopsis. IJCV, 120:153–168, 2016. 2, 8

[2] Sameer Agarwal, Noah Snavely, Ian Simon, Steven M. Seitz, and Richard Szeliski. Building rome in a day. In ICCV, 2009. 3

[3] Relja Arandjelovic, Petr Gronat, Akihiko Torii, Tomas Pajdla,´ and Josef Sivic. NetVLAD: CNN architecture for weakly supervised place recognition. In CVPR, 2016. 12

[4] Dejan Azinovic, Ricardo Martin-Brualla, Dan B Goldman,´ Matthias Nießner, and Justus Thies. Neural RGB-D surface reconstruction. In CVPR, 2022. 8

[5] Daniel Barath, Jana Noskova, Maksym Ivashechkin, and Jiˇr´ı Matas. MAGSAC++, a fast, reliable and accurate robust estimator. In CVPR, 2020. 5

[6] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, Jie Lei, Tengyu Ma, Baishan Guo, Arpit Kalla, Markus Marks, Joseph Greer, Meng Wang, Peize Sun, Roman Radle, Triantafyllos¨ Afouras, Effrosyni Mavroudi, Katherine Xu, Tsung-Han Wu, Yu Zhou, Liliane Momeni, Rishi Hazra, Shuangrui Ding, Sagar Vaze, Francois Porcher, Feng Li, Siyuan Li, Aishwarya Kamath, Ho Kei Cheng, Piotr Dollar, Nikhila Ravi, Kate Saenko,´ Pengchuan Zhang, and Christoph Feichtenhofer. SAM 3: Segment anything with concepts. In ICLR, 2026. 5

[7] Angel X. Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, Jianxiong Xiao, Li Yi, and Fisher Yu. ShapeNet: An information-rich 3D model repository. arXiv preprint arXiv:1512.03012, 2015. 2

[8] Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richlyannotated 3d reconstructions of indoor scenes. In CVPR, 2017. 2

[9] Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, Eli VanderBilt, Aniruddha Kembhavi, Carl Vondrick, Georgia Gkioxari, Kiana Ehsani, Ludwig Schmidt, and Ali Farhadi. Objaverse-XL: A universe of 10m+ 3D objects. In NeurIPS Datasets and Benchmarks, 2023. 2

[10] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In CVPR, 2023. 2

[11] Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superpoint: Self-supervised interest point detection and description. In CVPR Workshops, 2018. 3, 12

[12] Laura Downs, Anthony Francis, Nate Koenig, Brandon Kinman, Ryan Hickman, Krista Reymann, Thomas B. McHugh, and Vincent Vanhoucke. Google Scanned Objects: A high-quality dataset of 3D scanned household items. In ICRA, 2022. 2

[13] Bardienus Duisterhof, Lojze Zust, Philippe Weinzaepfel, Vincent Leroy, Yohann Cabon, and Jer´ ome Revaud. MASt3R-SfM: aˆ fully-integrated solution for unconstrained structure-from-motion. In 3DV, 2025. 3, 5, 12

[14] Johan Edstedt, Ioannis Athanasiadis, Marten Wadenb˚ ack, and¨

Michael Felsberg. Dkm: Dense kernelized feature matching for geometry estimation. In CVPR, 2023. 3

[15] Johan Edstedt, Qiyu Sun, Georg Bokman, M¨ arten Wadenb˚ ack,¨ and Michael Felsberg. Roma: Robust dense feature matching. In CVPR, 2024. 3

[16] Johan Edstedt, David Nordstrom, Yushan Zhang, Georg ¨ Bokman, Jonathan Astermark, Viktor Larsson, Anders Heyden,¨ Fredrik Kahl, Marten Wadenb˚ ack, and Michael Felsberg.¨ RoMa v2: Harder better faster denser feature matching, 2025. arXiv:2511.15706. 3, 5

[17] Xingyi He, Jiaming Sun, Yuang Wang, Di Huang, Hujun Bao, and Xiaowei Zhou. OnePose++: Keypoint-free one-shot object pose estimation without CAD models. In NeurIPS, 2022. 5

[18] Xingyi He, Jiaming Sun, Yifan Wang, Sida Peng, Qixing Huang, Hujun Bao, and Xiaowei Zhou. Detector-free structure from motion. In CVPR, 2024. 3, 5

[19] Varun Jampani, Kevis-Kokitsi Maninis, Andreas Engelhardt, Arjun Karpur, Karen Truong, Kyle Sargent, Stefan Popov, Andre´ Araujo, Ricardo Martin-Brualla, Kaushal Patel, Daniel Vlasic, Vittorio Ferrari, Ameesh Makadia, Ce Liu, Yuanzhen Li, and Howard Zhou. NAVI: Category-agnostic image collections with high-quality 3D shape and pose annotations. In NeurIPS Datasets and Benchmarks, 2023. 2, 6, 7

[20] Nikhil Keetha, Norman Muller, Johannes Sch¨ onberger, Lorenzo¨ Porzi, Yuchen Zhang, Tobias Fischer, Arno Knapitsch, Duncan Zauss, Ethan Weber, Nelson Antunes, Jonathon Luiten, Manuel Lopez-Antequera, Samuel Rota Bulo, Christian Richardt,\` Deva Ramanan, Sebastian Scherer, and Peter Kontschieder. MapAnything: Universal feed-forward metric 3D reconstruction. In 3DV, 2026. 3, 7, 8, 19

[21] JongMin Lee and Sungjoo Yoo. Dense-SfM: Structure from motion with dense consistent matching. In CVPR, 2025. 3, 5

[22] Vincent Leroy, Yohann Cabon, and Jer´ ome Revaud. Groundingˆ image matching in 3d with MASt3R. In ECCV, 2024. 3, 12

[23] Yuan Li, Yuanbo Xiangli, Hadar Averbuch-Elor, Noah Snavely, and Ruojin Cai. Long-tail internet photo reconstruction. In CVPR, 2026. 2, 3, 5

[24] Zhengqi Li and Noah Snavely. Megadepth: Learning single-view depth prediction from internet photos. In CVPR, 2018. 2, 3

[25] Haotong Lin, Sili Chen, Jun Hao Liew, Donny Y. Chen, Zhenyu Li, Yang Zhao, Sida Peng, Hengkai Guo, Xiaowei Zhou, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth anything 3: Recovering the visual space from any views. In ICLR, 2026. 3, 6, 7, 8, 19

[26] Philipp Lindenberger, Paul-Edouard Sarlin, and Marc Pollefeys. Lightglue: Local feature matching at light speed. In ICCV, 2023. 12

[27] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, Xuanmao Li, Xingpeng Sun, Rohan Ashok, Aniruddha Mukherjee, Hao Kang, Xiangrui Kong, Gang Hua, Tianyi Zhang, Bedrich Benes, and Aniket Bera. DL3DV-10K: A large-scale scene dataset for deep learning-based 3D vision. In CVPR, 2024. 2

[28] Xingchen Liu, Piyush Tayal, Jianyuan Wang, Jesus Zarzar, Tom Monnier, Konstantinos Tertikas, Jiali Duan, Antoine Toisoul, Jason Y. Zhang, Natalia Neverova, Andrea Vedaldi, Roman Shapovalov, and David Novotny. UnCommon objects in 3D. In CVPR, 2025. 1, 2, 7

[29] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2019. 7

[30] David G. Lowe. Distinctive image features from scale-invariant keypoints. IJCV, 60(2):91–110, 2004. 12

[31] Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler, Luca Sbordone, Patrick Labatut, and David Novotny. Common objects in 3d: Large-scale learning and evaluation of real-life 3d category reconstruction. In ICCV, 2021. 1, 2

[32] Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb, and Joshua M Susskind. Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. In ICCV, 2021. 7

[33] Paul-Edouard Sarlin, Cesar Cadena, Roland Siegwart, and Marcin Dymczyk. From coarse to fine: Robust hierarchical localization at large scale. In CVPR, 2019. 12

[34] Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superglue: Learning feature matching with graph neural networks. In CVPR, 2020. 3

[35] Johannes L. Schonberger and Jan-Michael Frahm. Structure-¨ from-motion revisited. In CVPR, 2016. 2, 3, 5, 12

[36] Thomas Schops, Johannes L Sch¨ onberger, Silvano Galliani,¨ Torsten Sattler, Konrad Schindler, Marc Pollefeys, and Andreas Geiger. A multi-view stereo benchmark with high-resolution images and multi-camera videos. In CVPR, 2017. 8

[37] Jamie Shotton, Ben Glocker, Christopher Zach, Shahram Izadi, Antonio Criminisi, and Andrew Fitzgibbon. Scene coordinate regression forests for camera relocalization in RGB-D images. In CVPR, 2013. 8

[38] Noah Snavely, Steven M. Seitz, and Richard Szeliski. Photo tourism: Exploring photo collections in 3D. ACM Transactions on Graphics, 25(3):835–846, 2006. 2, 3

[39] Jiaming Sun, Zehong Shen, Yuang Wang, Hujun Bao, and Xiaowei Zhou. LoFTR: Detector-free local feature matching with transformers. In CVPR, 2021. 3

[40] Xingyuan Sun, Jiajun Wu, Xiuming Zhang, Zhoutong Zhang, Chengkai Zhang, Tianfan Xue, Joshua B. Tenenbaum, and William T. Freeman. Pix3D: Dataset and methods for single-image 3D shape modeling. In CVPR, 2018. 2

[41] Zhenggang Tang, Yuchen Fan, Dilin Wang, Hongyu Xu, Rakesh Ranjan, Alexander Schwing, and Zhicheng Yan. MV-DUSt3R+: Single-stage scene reconstruction from sparse views in 2 seconds. In CVPR, 2025. 3

[42] Joseph Tung, Gene Chou, Ruojin Cai, Guandao Yang, Kai Zhang, Gordon Wetzstein, Bharath Hariharan, and Noah Snavely. Megascenes: Scene-level view synthesis at scale. In ECCV, 2024. 2, 3

[43] Shinji Umeyama. Least-squares estimation of transformation parameters between two point patterns. IEEE Transactions on Pattern Analysis and Machine Intelligence, 13(4):376–380, 1991. 13

[44] Fangjinhua Wang, Qingshan Xu, Yew-Soon Ong, and Marc Pollefeys. Lightweight and accurate multi-view stereo with confidence-aware diffusion model. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(11):10060–10075, 2025. 5

[45] Hengyi Wang and Lourdes Agapito. 3d reconstruction with spatial memory. In 3DV, 2025. 7

[46] Jianyuan Wang, Christian Rupprecht, and David Novotny. Posediffusion: Solving pose estimation via diffusion-aided bundle adjustment. In ICCV, 2023. 7

[47] Jianyuan Wang, Nikita Karaev, Christian Rupprecht, and David Novotny. VGGSfM: Visual geometry grounded deep structure from motion. In CVPR, 2024. 3

[48] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. VGGT: Visual geometry grounded transformer. In CVPR, 2025. 1, 3, 7, 8, 19

[49] Jianyuan Wang, Minghao Chen, Shangzhan Zhang, Nikita Karaev, Johannes Schonberger, Patrick Labatut, Piotr Bojanowski,¨ David Novotny, Andrea Vedaldi, and Christian Rupprecht. VGGT-Ω. In CVPR, 2026. arXiv:2605.15195. 1, 3, 6, 7, 8, 13, 19

[50] Ruicheng Wang, Sicheng Xu, Yue Dong, Yu Deng, Jianfeng Xiang, Zelong Lv, Guangzhong Sun, Xin Tong, and Jiaolong Yang. MoGe-2: Accurate monocular geometry with metric scale and sharp details. In NeurIPS, 2025. 5

[51] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jer´ ome Revaud. DUSt3R: Geometric 3d vision made easy.ˆ In CVPR, 2024. 3, 6

[52] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. $\pi ^ { 3 } \colon$ Permutation-equivariant visual geometry learning. In ICLR, 2026. 1, 3, 6, 7, 8, 19

[53] Tong Wu, Jiarui Zhang, Xiao Fu, Yuxin Wang, Jiawei Ren, Liang Pan, Wayne Wu, Lei Yang, Jiaqi Wang, Chen Qian, Dahua Lin, and Ziwei Liu. OmniObject3D: Large-vocabulary 3D object dataset for realistic perception, reconstruction and generation. In CVPR, 2023. 2

[54] Yushuang Wu, Luyue Shi, Haolin Liu, Hongjie Liao, Lingteng Qiu, Weihao Yuan, Xiaodong Gu, Zilong Dong, Shuguang Cui, and Xiaoguang Han. MVImgNet2.0: A larger-scale dataset of multi-view images. ACM Transactions on Graphics, 43(6): 173:1–173:16, 2024. 2

[55] Hongchi Xia, Yang Fu, Sifei Liu, and Xiaolong Wang. Rgbd objects in the wild: Scaling real-world 3d object learning from rgb-d videos. In CVPR, 2024. 1, 2

[56] Yu Xiang, Roozbeh Mottaghi, and Silvio Savarese. Beyond pascal: A benchmark for 3d object detection in the wild. In WACV, 2014. 2

[57] Yu Xiang, Wonhui Kim, Wei Chen, Jingwei Ji, Christopher Choy, Hao Su, Roozbeh Mottaghi, Leonidas Guibas, and Silvio Savarese. ObjectNet3D: A large scale database for 3D object recognition. In ECCV, 2016. 2

[58] Yuanbo Xiangli, Ruojin Cai, Hanyu Chen, Jeffrey Byrne, and Noah Snavely. Doppelgangers++: Improved visual disambiguation with geometric 3d features. In CVPR, 2025. 3

[59] Jianing Yang, Alexander Sax, Kevin J. Liang, Mikael Henaff, Hao Tang, Ang Cao, Joyce Chai, Franziska Meier, and Matt Feiszli. Fast3R: Towards 3D reconstruction of 1000+ images in one forward pass. In CVPR, 2025. 3

[60] Yao Yao, Zixin Luo, Shiwei Li, Jingyang Zhang, Yufan Ren, Lei Zhou, Tian Fang, and Long Quan. BlendedMVS: A large-scale dataset for generalized multi-view stereo networks. In CVPR, 2020. 7

[61] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. Scannet++: A high-fidelity dataset of 3d indoor scenes. In ICCV, 2023. 2

[62] Xianggang Yu, Mutian Xu, Yidan Zhang, Haolin Liu, Chongjie Ye, Yushuang Wu, Zizheng Yan, Chenming Zhu, Zhangyang

Xiong, Tianyou Liang, Guanying Chen, Shuguang Cui, and Xiaoguang Han. MVImgNet: A large-scale dataset of multi-view images. In CVPR, 2023. 2

[63] Zhu Yu, Jingnan Gao, Runmin Zhang, Lingteng Qiu, Zhengyi Zhao, Rui Peng, Yichao Yan, Kejie Qiu, Siyu Zhu, Zilong Dong, Si-Yuan Cao, and Hui-Liang Shen. Towards consistent video geometry estimation. arXiv preprint arXiv:2605.30060, 2026. 5

[64] Shangzhan Zhang, Jianyuan Wang, Yinghao Xu, Nan Xue, Christian Rupprecht, Xiaowei Zhou, Yujun Shen, and Gordon Wetzstein. FLARE: Feed-forward geometry, appearance and camera estimation from uncalibrated sparse views. In CVPR, 2025. 3

[65] Yiwen Zhang, Joseph Tung, Ruojin Cai, David Fouhey, and Hadar Averbuch-Elor. Emergent extreme-view geometry in 3D foundation models. In CVPR, 2026. 3

# Supplementary Material

## A. Dataset Curation

This section gives the details of the curation summarized in Section 3.2 of the main paper.

Enumeration. Thirteen vehicle classes are enumerated from their Wikidata items: aircraft, ships, locomotives, automobiles, rail vehicles, buses, trams, military vehicles, trucks, motorcycles, tractors, construction equipment, and other vehicles. Each item gives one Wikimedia Commons category, 22,983 in total. Whether a class names individual objects or designs is fixed by the class: aircraft, ships, locomotives, and trams are identified by registration, hull, or fleet number, while the remaining classes are identified by model.

All other objects are found by traversing the Wikimedia Commons category graph from 856 root categories. We chose 211 of them by hand, and the LLM screened the remaining 645 from the subcategories of broad hand-chosen categories. The traversal does not descend into branches whose titles match one of seven kinds of category that do not depict the object itself: photos taken with a camera model, people with an object, media about an object, replicas and scale models, institutions, parts, and wiki maintenance categories. It prunes 37,930 branches this way and records 155,744 categories.

Classification. Of the recorded categories, 49,252 contain at least 25 photos, counting the photos of subcategories that depict the same object. The LLM classifies each of them from its title and the titles of the categories above it in the graph. The prompt is reproduced in Section G. It keeps 40,254 categories as rigid physical objects, rejects 8,956 for the reasons listed in Table S2, and leaves 42 undecided. Of the kept categories, 13,347 name an individual object or a design, and the remaining 26,907 are categories of many objects, such as Statues in Paris, which we do not use. After removing categories already covered by the vehicle classes and merging subcategories that depict the same object into their parent, 10,901 object categories remain. Together with the vehicle classes, they form the 33,884 candidate objects of the main paper.

Table S2. Categories rejected by the classifier. Reasons for the 8,956 categories with at least 25 photos that the LLM does not keep.
<table><tr><td>Reason</td><td>Categories</td></tr><tr><td>Flat object, such as a painting or a sign</td><td>1,802</td></tr><tr><td>Media or two-dimensional depiction of an object</td><td>1,606</td></tr><tr><td>Place or institution</td><td>1,489</td></tr><tr><td>Abstract or maintenance category</td><td>1,066</td></tr><tr><td>Event or activity</td><td>811</td></tr><tr><td>Deformable object</td><td>767</td></tr><tr><td>Part of an object</td><td>614</td></tr><tr><td>People</td><td>275</td></tr><tr><td>Photos taken with or of a product</td><td>246</td></tr><tr><td>Other</td><td>280</td></tr><tr><td>Total</td><td>8,956</td></tr></table>

Categories. Table S1 lists the 53 categories into which the LLM groups the 337 object labels, with the number of objects and photos of each.

Overlap with internet-photo scene datasets. Table S3 gives the overlap of Section 2 of the main paper by regime. A category counts as a MegaScenes scene when its title matches a category collected by MegaScenes, and as reconstructed when MegaScenes releases a reconstruction for it. Nearly all overlapping objects are fixed objects, which MegaScenes and MegaDepth-X treat as scenes.

Table S3. Overlap with internet-photo scene datasets by regime. Number of TRAINSPOTTING objects whose Wikimedia Commons category is a MegaScenes scene, has a released MegaScenes reconstruction, or is a MegaDepth-X scene.
<table><tr><td>Regime</td><td>Objects</td><td>MegaScenes category</td><td>MegaScenes reconstruction</td><td>MegaDepth-X</td></tr><tr><td>Fixed</td><td>4,338</td><td>1,798</td><td>1,128</td><td>56</td></tr><tr><td>Roaming</td><td>2,869</td><td>41</td><td>34</td><td>0</td></tr><tr><td>Instance-pooled</td><td>4,861</td><td>2</td><td>1</td><td>0</td></tr><tr><td>Total</td><td>12,068</td><td>1,841</td><td>1,163</td><td>56</td></tr></table>

## B. Reconstruction Pipeline

Table S4 lists the objects and photos that remain after each stage of the pipeline. Table S5 lists the fixed thresholds of the pipeline of Section 4 of the main paper.

Comparison with other matchers. Figure S1 compares the pipeline of Section 4 of the main paper with four alternatives on the images that pass object gating: COLMAP with SIFT and exhaustive matching [30, 35], SuperPoint with LightGlue [11, 26] in hloc [33] with NetVLAD retrieval [3], MASt3R matches [22] with the COLMAP mapper, and MASt3R-SfM [13]. Each alternative runs with its default settings and, where it retrieves image pairs, with 20 neighbours per image as in our pipeline. On the instance-pooled and roaming objects, the alternatives fail to reconstruct the object. Even on the fixed objects, our pipeline produces a more complete and cleaner reconstruction.

## C. Training Details

This section completes the setup of Section 5.1 of the main paper.

Trainable parameters. We fine-tune the released VGGT-Ω checkpoint with 1B parameters. All 24 frame and global attention layers of the aggregator and the camera head are trained, while the patch embedding, the camera and register tokens, and the depth head are frozen. This leaves 808M of the 1.14B parameters trainable.

Sampling. The number of views per sample is drawn once per batch, uniformly between 2 and 24. Each batch is filled to a fixed budget of 192 images per GPU. Inputs follow the training preprocessing of VGGT-Ω [49]. Depth and points are supervised only on object pixels that have MVS depth. The replay samples from BlendedMVS, Hypersim, and uCO3D are supervised on all pixels with ground truth.

Table S1. Object categories. The 53 categories, grouped by supercategory, with the number of objects and photos of each (12,068 objects and 311,886 photos in total).
<table><tr><td>Category</td><td>Objects</td><td>Photos</td><td>Category</td><td>Objects</td><td>Photos</td></tr><tr><td>Road vehicles</td><td></td><td></td><td>Musical instruments</td><td></td><td></td></tr><tr><td>car</td><td>1,967</td><td>87,193</td><td>pipe organ</td><td>390</td><td>5,733</td></tr><tr><td>bus</td><td>476</td><td>28,932</td><td>musical instrument</td><td>38</td><td>982</td></tr><tr><td>military vehicle</td><td>341</td><td>7,522</td><td>Household &amp; personal objects</td><td></td><td></td></tr><tr><td>truck &amp; van</td><td>269</td><td>8,553</td><td>camera &amp; lens</td><td>142</td><td>2,946</td></tr><tr><td>motorcycle</td><td>187</td><td>2,776</td><td>computer &amp; office machine</td><td>90</td><td>1,364</td></tr><tr><td>tractor &amp; other road vehicle</td><td>71</td><td>522</td><td>vase &amp; vessel</td><td>47</td><td>893</td></tr><tr><td>bicycle</td><td>30</td><td>506</td><td>consumer electronics</td><td>33</td><td>457</td></tr><tr><td>Statues &amp; sculpture</td><td></td><td></td><td>appliance &amp; personal item</td><td>24</td><td>260</td></tr><tr><td>statue</td><td>1,628</td><td>24,940</td><td>furniture &amp; lighting</td><td>17</td><td>178</td></tr><tr><td>sculpture</td><td>432</td><td>7,649</td><td>Weapons &amp; armour</td><td></td><td></td></tr><tr><td>bust &amp; head</td><td>100</td><td>1,576</td><td>artillery</td><td>173</td><td>2,088</td></tr><tr><td>Monuments &amp; landmarks</td><td></td><td></td><td>firearm &amp; hand weapon</td><td>127</td><td>1,550</td></tr><tr><td>war memorial</td><td>553</td><td>8,079</td><td>missile</td><td>29</td><td>286</td></tr><tr><td>fountain</td><td>385</td><td>4,788</td><td>helmet</td><td>13</td><td>144</td></tr><tr><td>monument &amp; memorial</td><td>340</td><td>8,276</td><td>Religious objects</td><td></td><td></td></tr><tr><td>obelisk</td><td>141</td><td>2,205</td><td>altar &amp; altarpiece</td><td>68</td><td>504</td></tr><tr><td>gravestone &amp; tomb</td><td>76</td><td>1,173</td><td>church furnishing &amp; reliquary</td><td>30</td><td>291</td></tr><tr><td>megalith</td><td>73</td><td>1,131</td><td>cross &amp; crucifix</td><td>25</td><td>261</td></tr><tr><td>sarcophagus &amp; coffin door, gate &amp; arch</td><td>46</td><td>685</td><td>bell</td><td>17</td><td>261</td></tr><tr><td></td><td>45</td><td>949</td><td>Street furniture</td><td></td><td></td></tr><tr><td>clock &amp; sundial</td><td>39 33</td><td>1,005</td><td>post box</td><td>85</td><td>3,003</td></tr><tr><td>runestone &amp; carved stone</td><td>31</td><td>657</td><td>bench &amp; other street furniture</td><td>28</td><td>325</td></tr><tr><td>commemorative column &amp; pillar</td><td>11</td><td>302</td><td>telephone box</td><td>17</td><td>733</td></tr><tr><td>tower &amp; small building</td><td></td><td>182</td><td>Machinery &amp; equipment</td><td></td><td></td></tr><tr><td>Aircraft &amp; spacecraft</td><td></td><td></td><td>engine</td><td>62</td><td>618</td></tr><tr><td>aircraft</td><td>1,415</td><td>37,798</td><td>industrial machine &amp; equipment</td><td>24</td><td>385</td></tr><tr><td>spacecraft &amp; rocket</td><td>46</td><td>647</td><td>crane</td><td>14</td><td>135</td></tr><tr><td>Rail vehicles</td><td></td><td></td><td>scientific instrument</td><td>12</td><td>171</td></tr><tr><td>locomotive</td><td>543</td><td>16,838</td><td>robot</td><td>9</td><td>111</td></tr><tr><td>tram</td><td>496</td><td>16,033</td><td>Other</td><td></td><td></td></tr><tr><td>train &amp; railcar</td><td>69</td><td>2,493</td><td>other object</td><td>10</td><td>61</td></tr><tr><td>Watercraft</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ship &amp; boat</td><td>701</td><td>14,736</td><td></td><td></td><td></td></tr></table>

Table S4. Reconstruction yield. Number of objects and photos that remain after each stage of the pipeline in Section 4 of the main paper.
<table><tr><td>Stage</td><td>Objects</td><td>Photos</td></tr><tr><td>Input</td><td>17,470</td><td>1,251,986</td></tr><tr><td>Object gating</td><td>16,141</td><td>775,609</td></tr><tr><td>Structure from motion</td><td>14,818</td><td>558,281</td></tr><tr><td>MVS depth</td><td>14,032</td><td>486,275</td></tr><tr><td>Mask and depth checks</td><td>13,236</td><td>367,245</td></tr><tr><td>Final dataset</td><td>12,068</td><td>311,886</td></tr></table>

Optimization. The cosine schedule decays the learning rate to 10<sup>−8</sup>. Gradients are clipped to a norm of 1. Training uses bfloat16 precision with gradient checkpointing.

Object mask head. After fine-tuning, we freeze the model and train a small convolutional head on the fused features of its depth head to predict, for each pixel, whether it belongs to the object. The target is the object mask of the dataset. On held-out objects, the predicted masks reach an intersection over union of 0.92 with the dataset masks.

## D. Evaluation Details

This section details the protocol of Section 5.2 of the main paper.

Samples on TRAINSPOTTING. The views of a sample form a connected set under the covisibility relation used for training: two photos are connected when at least 5% of the object points of one project inside the object mask of the other. Each view is resized to cover 512 × 512 pixels and cropped around its principal point.

External datasets. For ordered captures, the 8 views are equally spaced over the sequence, including its first and last frame. For unordered collections, they are a fixed random draw. Scenes with fewer than 8 usable frames are skipped. On NAVI, images are center-cropped to a square, and frames that the dataset flags as occluded are excluded. Inputs are resized to each model’s patch multiple, and outputs are resampled to the ground-truth grid.

Metrics. The depth scale of a sample is the median ratio of ground-truth to predicted depth over the valid pixels of all its views. The similarity transform of the point-map metrics is estimated from corresponding valid pixels [43] before the ICP refinement.

Effect of the alignment on misregistered predictions. Section 5.4 of the main paper explains why pose failures appear as completeness error rather than accuracy error. Table S6 quantifies this on NAVI in-the-wild by the extent of the aligned prediction relative to the extent of the ground truth. The alignment scales most predictions of VGGT-Ω to less than half the object’s size, while the predictions of SPOTTER keep their size.

![](images/d9afb9354194d385cb343e1574ca627b967147b1ac920ff096bd998e7848ba03.jpg)  
Figure S1. Structure from motion with other matchers. Each row shows one TRAINSPOTTING object: six of the input photos, then the largest model of COLMAP with SIFT, SuperPoint with LightGlue, MASt3R matches with the COLMAP mapper, MASt3R-SfM, and our pipeline, all run on the same images. The number of registered images out of the input images is given under each panel. The other methods fail on instance-pooled and roaming objects, and on fixed objects our pipeline produces a more complete and cleaner reconstruction. In the bottom three rows, we frame the object to show its details, so not all cameras are visible.

Table S6. Extent of the aligned prediction on NAVI in-the-wild. Ratio of the extent of the aligned prediction to the extent of the ground truth over the 35 objects, and the number of objects with a ratio below 0.7.
<table><tr><td>Method</td><td>Median</td><td>10th pct.</td><td>90th pct.</td><td>Objects below 0.7</td></tr><tr><td>VGGT-Ω</td><td>0.43</td><td>0.15</td><td>0.87</td><td>28</td></tr><tr><td>SPOTTER</td><td>0.92</td><td>0.81</td><td>0.96</td><td>0</td></tr></table>

## E. Additional Qualitative Results

Failure cases. Figure S2 shows four held-out samples on which SPOTTER fails, one for each of the most common failure modes. Views taken with long focal lengths are ambiguous between focal length and depth, so their points can be misaligned with the other views and appear as noise around the airliner. Close-up views can be ambiguous in their placement on the object, and the close-up of a wheel is localized at the wrong position on the car. Symmetric objects produce doppelganger ambiguities, and the cameras of the grave monument are placed on the wrong one of its two identical pillars. Objects far from the training distribution, such as the toy shark, whose appearance changes drastically with its pose and placement across the views, are reconstructed inconsistently.

Additional qualitative comparisons. Figure S3 shows ten held-out samples drawn at random.

## F. Additional Results

Table S7 reports all metrics of every model on TRAINSPOTTING, and Table S8 on the external benchmarks.

## G. LLM Prompts

All three LLM passes of the curation use the same setup. The model is Claude Opus 5, the system prompt is the rubric reproduced below, each user message contains a batch of 50 numbered categories, and the response is JSON constrained to a fixed schema. On a hand-labelled set of 32 categories, the classification pass agreed with the labels on all 32 verdicts and on 14 of 15 granularity labels.

Table S5. Pipeline thresholds. The fixed thresholds of the reconstruction pipeline of Section 4 of the main paper, in the order of its stages. Image-relative values are fractions of the image side.
<table><tr><td>Stage</td><td>Threshold</td><td>Value</td><td>Definition</td></tr><tr><td>Object gating</td><td>minimum box size</td><td>10% of the image area</td><td>Largest SAM3 detection, with detection score above 0.5.</td></tr><tr><td>Retrieval</td><td>neighbours per image</td><td>20</td><td></td></tr><tr><td rowspan="7">Matching</td><td>correspondences per pair</td><td>8,000</td><td>Sampled from the dense warp.</td></tr><tr><td></td><td>0.20</td><td>Certainty threshold.</td></tr><tr><td> $\tau _ { \mathrm { c y c } }$ </td><td>0.4% of the image side</td><td>Cycle-consistency distance.</td></tr><tr><td>Tepi</td><td>0.1% of the image side</td><td>Epipolar distance.</td></tr><tr><td>keypoint merge radius</td><td>0.2% of the image side</td><td>Match endpoints within this radius form one keypoint.</td></tr><tr><td>confirmed points</td><td>100</td><td>Points on tracks of length at least 3.</td></tr><tr><td>confirmed fraction</td><td>0.30</td><td>Confirmed points over all triangulated points of the image.</td></tr><tr><td rowspan="2"></td><td>covisibility cut</td><td>conductance below 0.15</td><td>Spectral bisection of the covisibility graph; the largest cluster is kept.</td></tr><tr><td>source views</td><td>6</td><td></td></tr><tr><td>Multi-view stereo</td><td>confidence</td><td>0.3, 0.5, 0.5</td><td>Minimum confidence at the three stages of CasDiffMVS.</td></tr><tr><td rowspan="2">Mask selection</td><td>confident depth</td><td>200 px</td><td>Minimum confident MVS pixels for a mask to be chosen.</td></tr><tr><td>verified depth</td><td>5% of the mask</td><td>Minimum fraction of the chosen mask with verified depth.</td></tr><tr><td rowspan="3">Final quality check</td><td>dominant subject</td><td>5×</td><td>Chosen mask area relative to every other detection.</td></tr><tr><td>focal length</td><td>0.3 to 10 image widths</td><td></td></tr><tr><td>photos per object</td><td>2 (training), 5 (held-out)</td><td></td></tr></table>

![](images/47c0ddef33268a919be7afc7756ba9df9d5965a5d073f0577cc5c26e0e9cd0e1.jpg)  
Figure S2. Failure cases. Four held-out samples on which SPOTTER fails: airliner (noisy depth at long focal lengths), car (misregistered close-up of a wheel), grave monument (doppelganger ambiguity), and toy shark from NAVI in-the-wild (out of distribution).

Category classification (Section 3.2 of the main paper). The response for each category gives the verdict (keep, reject, or uncertain), the reject reason, the granularity (individual, model, named unknown, or pool), and whether the object is installed at a fixed site.

from the title (plus its parent category and crawl root) whether the category is about physical objects suitable for reconstruction.

Each numbered item gives:

\- the category TITLE (your main evidence),

\- parent: the category it was found under,

\- root: the top-level category a breadth-first crawl started from to reach it. A hint only -- the crawl sometimes wanders, so an item may sit under an unrelated root. An off-root category of genuine physical objects is still a keep.

- files: how many image files the category holds,   
- wikidata: whether a Wikidata item is linked to it. A weak hint that the category names a recognized concept -- but concepts include generic types ("settee"), so it proves nothing about specificity.

The pipeline tolerates noise but not wrongness: a rejected category is gone forever, while a wrongly kept one merely wastes a download before reconstruction discards it. When a title does not clearly establish a reject reason, KEEP it.

For each numbered item return:

verdict:

\- "keep" -- the category depicts physical, rigid objects (whatever family they belong to).

\- "reject" -- the title alone establishes one of the reject reasons below.

"uncertain" -- genuinely cannot tell what the category depicts.

reason (for reject; use "ok" when keep):

- "usage or provenance" -- photos taken WITH the object or sourced from somewhere: "Taken with Nikon D5300", "Photos from X stream", "Media from iNaturalist".   
- "people" -- people using/holding/wearing the object: "Guitarists with...", "People with Canon cameras".   
- "media or 2d" -- depictions rather than the object: logos, icons, SVG, diagrams, drawings, paintings, stamps, advertisements, book titles, close-up detail shots.

\- "place or institution" -- a place, street, building, company, museum, maker, or event venue that merely contains the family noun: "Cannon Street station", "Scientific instrument makers".

\- "event or activity" -- competitions, festivals, ceremonies, activities: "2019 ICF Canoe Slalom World Championships", "Emptying of post boxes".

\- "abstract or meta" -- licenses, wiki maintenance, pronunciation audio, quality-image collections, categories-about-categories.

\- "flat" -- near-planar subjects: manhole covers, coins, plaques, murals, signage, banknotes.

\- "deformable or soft" -- clothing, uniforms, textiles, bags, plush.

\- "parts or accessories" -- components rather than whole objects: "Guitar headstocks", "Camera parts", "Bicycle pannier racks" is a borderline whole object (keep); use this only for clear fragments of a larger object. "other" -- a reject that fits none of the above (say nothing else; the verdict is what matters).

## granularity (for keep; "n a" for reject):

\- "individual" -- one specific physical object: "Penfold pillar box on Otterndorf Green, Sheringham", "USCGC White Pine (WLM-547)".

\- "model" -- one manufactured design or standardized type whose units are geometrically interchangeable: "Fender Stratocaster", "K6 telephone boxes in Derbyshire", "Queen Elizabeth II type A/B red pillar boxes in Merseyside", "IBM Model M Keyboard". A place qualifier does not demote a model.

\- "named unknown" -- the title carries a proper-name or model-number token you do not recognize, modifying the family noun: "Parc Vue benches in New York City", "KX100 telephone boxes". An unfamiliar name is evidence of specificity -- never downgrade these to pool just because you don’t know the name. "pool" -- a generic type, possibly place-scoped, containing many different designs: "Snowplows", "Settees", "Traffic bollards in Wuppertal", "Cannons in Florida", "Wooden benches". Also under-specified family names that span multiple geometries: "Glock" (17/19/26/43 differ), "King George V post boxes in the Republic of Ireland" (mixes wall-mounted and pillar).

fixed location (boolean): true if the objects are permanently installed at fixed sites (street furniture, installed fountains, monuments-like); false if portable/movable; judge from the object family when the title doesn’t say.

object scale (boolean, for keep; use false for reject): a descriptive FLAG, never a reason to reject. True if the subject is hand-to-room scale -- a statue, gravestone, cannon, telescope on a mount, church organ. False if larger. The test: could a person walk around it and photograph the whole thing from a few metres? "Very Large Telescope" (an observatory site) -> false; "Zenit avtomat" (a camera) -> true. Large MOVABLE things -- ships, airliners, locomotives -- are false and that is fine; the flag records size only, and such categories are still kept. Obelisks and triumphal columns -> true; anything with interior rooms -> false.

Different PHYSICAL UNITS of one design count as one object. A category like "Honda Accord Crosstour" contains photos of many distinct physical cars -- different owners, colors, countries -- yet it is ONE object for reconstruction, because every unit shares the same body shape. Likewise fifty K6 telephone kiosks across fifty towns are one "model". A category is never a pool merely because its photos show more than one physical specimen.

The line between "model" and "pool" is SHAPE, not appearance. Reconstruction only uses geometry: units that differ in color, finish, livery, trim, or wear but share one body shape are ONE model ("Fender Stratocaster" stays a model despite countless colors; "Red post boxes" pooled across designs is still a pool despite one color). A category is a pool only when the objects in it have materially different SHAPES -- different products, different size variants (Glock 17 vs 26), or a bare type noun spanning many designs. Minor sub-variant drift (a changed headstock logo, a facelifted grille) does not demote a model; a doubt about shape goes to "model", not "pool".

Rules of thumb:

\- Distinguish "type + place" pools from "named design + place" models by whether the pre-place part names a specific design (code, reign+type, brand+model) or a bare noun.

\- A brand alone ("Zenit cameras") is a pool of that brand’s models -- keep, pool.

- Museum/display context does not reject an object category: "Torpedoes" in museums are still torpedoes. Only reject when the subject is the institution itself.   
- When two readings are plausible, keep it and give your best-guess granularity -- do not resolve doubt by

rejecting.

## User message:

Classify these Wikimedia Commons categories per the   
rubric:   
0. "<title>" | parent: <parent category> | root:   
<crawl root> | files: <n> | wikidata: yes/no   
1. ...

## Segmentation prompt (Section 4.1 of the main paper). The response for each category gives one generic noun phrase, optional alternatives, and a confidence.

You are naming the object class of Wikimedia Commons categories. Each category is about one physical object (or one manufactured design). Your output is a single generic noun phrase per category -- the name of the kind of thing it is. It will later be used as a text prompt for an image segmentation model; you do not need to know anything about that model. Just name the object correctly and consistently.

Each numbered item gives the category TITLE, its parent category, the crawl root it was reached from, and a granularity label (individual / model / named unknown / pool) from an earlier pass.

For each item return:

sam prompt -- ONE short generic noun phrase for the object   
class: "statue", "cannon", "gravestone", "pipe organ",   
"telephone box", "typewriter", "wind turbine". Rules:   
- Generic, never specific. "Bust of Voltaire" -> "bust". "Pentax K10D" -> "camera". "K6 telephone boxes in Derbyshire" -> "telephone box". Names, model numbers, places, and reigns never appear in the output.   
- Name the WHOLE object the category is about, not a component: "pipe organ" not "organ pipes"; "fountain" not "basin"; "cannon" not "cannon barrel".   
- No material, color, pose, or style: "bronze equestrian statue" -> "statue". Those describe one instance; the same word must fit every unit in the category.

\- Two-word compounds are fine when a single word is wrong or too broad: "wayside cross", "baptismal font", "sewing machine", "vending machine".

ANCHOR TO THE CATEGORY TREE. The parent and root tell you the object’s family, and that is authoritative: everything under a "Dolmens" root is a dolmen, everything under "Pipe organs by country" is a pipe organ. Use the family’s noun unless the TITLE clearly names something more specific or different (a "War memorials" root can hold an obelisk, a statue, or a wall -- take the title’s word there). Root and parent names are index-shaped ("Coastal artillery", "War memorials by country") -- give the object noun ("cannon", "war memorial"), not the index name. Be CONSISTENT: every category in the same family should get the same word unless its title says otherwise.

alternatives -- normally an EMPTY list. Fill it ONLY when the title and the tree genuinely disagree about what the object is, or the title alone does not make the object clear (e.g. a battery site that could be read as guns, a bunker, or an installation). A synonym of your chosen word is NOT an uncertainty ("dolmen" is not unsure because "megalithic tomb" exists) -- never list synonyms. At most 2 entries. A non-empty list marks the item for human review, so listing alternatives "to be safe" defeats its purpose.

confidence -- "high" when the object class is clear from title + tree, "low" when you filled alternatives or the title does not make the object clear. Nothing else.

## User message:

Assign segmentation prompts to these categories:   
0. "<title>" | parent: <parent category> | root:   
<crawl root> | granularity: <label>   
1. ...

## Root-category screening (Section A). The response for each category gives a yes or no decision and a short reason.

You are screening Wikimedia Commons category names to   
decide which ones are FAMILIES OF RIGID PHYSICAL OBJECTS   
worth crawling for a 3D object-reconstruction dataset.   
Each name is a direct subcategory of a broad "umbrella"   
category (e.g. "Musical instruments", "Tools", "Furni  
ture", "Containers"); the umbrella is given for context.   
Answer YES when the category is a class of manufactured or   
carved rigid objects whose members would be photographed   
as objects: guitars, pianos, chairs, knives, watches,   
telephones, howitzers, loudspeakers, tins, buckets, radars,   
electric motors.   
Answer NO for anything else:   
- topics, activities, concepts, history ("Ball games",   
"History of synthesizers", "Human-computer interaction",   
"Light art", "Synth DIY")   
- people and occupations ("Clockmakers", "Sculptors")   
- vehicles of any kind -- cars, trucks, ships, aircraft,   
trains, bicycles count as vehicles here ("Vehicles",   
"Anchorhandling, tug and supply ships", "Survey ships")   
-- they are covered by a separate dataset   
- soft / deformable / flat things (textiles, rugs, bags,   
clothing, flags, paper, coins, plaques)   
- media about objects (photos-of, logos, "Numbers on racing   
vehicles", stickers)   
categories that are groupings by attribute rather than   
a family ("Metal sundials", "Black liftbacks", "...by   
color", "...by year")   
ammunition cartridges are borderline small objects   
-- answer YES for guns/weapons families, NO for   
cartridges/bullets   
When the name is a sub-family of something obviously   
object-like ("Bullpup", "Guitars by type", "Submachine   
guns by model", "Lenses by brand", "Watches by brand")   
answer YES -- index suffixes ("by type/brand/model") are   
fine; those are how Commons lists the members.   
Output for each item: index, decision ("yes"/"no"), and a   
reason of at most 6 words.

## User message:

Input views

GT

π<sup>3</sup>

DA3

VGGT-Ω

SPOTTER

![](images/029aaf7a8bf8e60993aef80d4ccaa2cd0920e9eb6d02387797955e47aceddaa3.jpg)  
Figure S3. Additional qualitative comparisons. Ten held-out samples drawn at random: three TRAINSPOTTING objects with 16 input views, five with 4 input views, and two NAVI in-the-wild objects with 8 input views.

Table S7. All metrics on TRAINSPOTTING. Extends Table 3 of the main paper with the median of accuracy and completeness, the inlier ratios at 1.05 and 1.10, and AUC at 15<sup>◦</sup>. Accuracy and completeness are in percent of the object extent.
<table><tr><td>Method</td><td colspan="4">Point map</td><td colspan="4">Depth</td><td colspan="3"> $\mathrm { C a m e r a }$ </td></tr><tr><td></td><td>Acc mean</td><td>Acc med.</td><td>Comp mean</td><td>Comp med.</td><td>AbsRel</td><td>δ1.03</td><td> $\delta _ { 1 . 0 5 }$ </td><td> $\delta _ { 1 . 1 0 }$ </td><td> $\mathrm { \ A U C _ { 5 } }$ </td><td> $\mathrm { { A U C } _ { 1 5 } }$ </td><td> $\mathrm { \bf A U C } _ { 3 0 }$ </td></tr><tr><td>VGGT [48]</td><td>2.38</td><td>1.62</td><td>8.08</td><td>6.48</td><td>0.160</td><td>32.4</td><td>41.8</td><td>56.2</td><td>11.5</td><td>29.9</td><td>44.1</td></tr><tr><td>π3 [52]</td><td>2.33</td><td>1.61</td><td>6.22</td><td>4.76</td><td>0.162</td><td>34.7</td><td>43.8</td><td>56.8</td><td>12.5</td><td>31.9</td><td>46.4</td></tr><tr><td>DA3 [25]</td><td>2.22</td><td>1.60</td><td>12.17</td><td>10.11</td><td>0.189</td><td>32.1</td><td>41.5</td><td>54.6</td><td>10.1</td><td>21.0</td><td>29.4</td></tr><tr><td>MapAny. [20]</td><td>2.20</td><td>1.57</td><td>7.75</td><td>6.18</td><td>0.206</td><td>26.6</td><td>35.3</td><td>49.1</td><td>5.3</td><td>19.3</td><td>33.1</td></tr><tr><td>VGGT-Ω [49]</td><td>1.84</td><td>1.36</td><td>12.68</td><td>10.77</td><td>0.201</td><td>28.1</td><td>37.1</td><td>50.3</td><td>7.2</td><td>16.9</td><td>25.0</td></tr><tr><td>SPOTTER</td><td>1.22</td><td>0.89</td><td>1.15</td><td>0.81</td><td>0.071</td><td>53.3</td><td>65.4</td><td>80.5</td><td>44.2</td><td>70.1</td><td>81.2</td></tr></table>

Table S8. All metrics on the external benchmarks. The metrics of Table S7 and normal consistency (NC) on the benchmarks of Tables 5 and 6 of the main paper. Accuracy and completeness are in millimeters.
<table><tr><td rowspan="2">Method</td><td colspan="5">Point map</td><td colspan="4">Depth</td><td colspan="3">Camera</td></tr><tr><td>Acc mean</td><td>Acc med.</td><td>Comp mean</td><td>Comp med.</td><td>NC</td><td>AbsRel</td><td> $\delta _ { 1 . 0 3 }$ </td><td> $\delta _ { 1 . 0 5 }$ </td><td> $\delta _ { 1 . 1 0 }$ </td><td> $\mathrm { { A U C } _ { 5 } }$ </td><td> $\mathrm { { A U C } _ { 1 5 } }$ </td><td> $\mathrm { \bf A U C } _ { 3 0 }$ </td></tr><tr><td colspan="2">7-Scenes (18 samples)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VGGT [48]</td><td>25.3</td><td>11.6</td><td>32.6</td><td>15.9</td><td>0.735</td><td>0.277</td><td>67.5</td><td>83.4</td><td>92.3</td><td>26.6</td><td>65.2</td><td>81.0</td></tr><tr><td>π3 [52]</td><td>20.0</td><td>10.5 13.1</td><td>26.7</td><td>14.3</td><td>0.765</td><td>0.342</td><td>70.5</td><td>87.1</td><td>96.4</td><td>28.8</td><td>68.5</td><td>83.3</td></tr><tr><td>DA3 [25]</td><td>25.4 28.2</td><td>14.7</td><td>35.7 36.8</td><td>15.9</td><td>0.755</td><td>0.324</td><td>74.1</td><td>88.1</td><td>94.6</td><td>28.5</td><td>66.3</td><td>81.5</td></tr><tr><td>MapAny. [20] VGGT-Ω [49]</td><td>21.8</td><td>11.9</td><td>27.4</td><td>16.0 14.4</td><td>0.739 0.760</td><td>0.332 0.295</td><td>58.3</td><td>77.3</td><td>93.3</td><td>18.2</td><td>54.7 75.4</td><td>72.4 87.2</td></tr><tr><td>SPOTTER</td><td>20.2</td><td>10.2</td><td>27.4</td><td>14.1</td><td>0.760</td><td>0.301</td><td>71.9 75.1</td><td>86.9</td><td>94.3</td><td>40.8 38.8</td><td>73.8</td><td>86.1</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>88.3</td><td>94.5</td><td></td><td></td><td></td></tr><tr><td colspan="2">NRGBD (9 samples)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VGGT [48]</td><td>17.4</td><td>10.1</td><td>16.9</td><td>7.0</td><td>0.914</td><td>0.016</td><td>90.9</td><td>96.2</td><td>98.3</td><td>86.4</td><td>95.1</td><td>97.6</td></tr><tr><td>π3 [52]</td><td>17.9</td><td>9.3</td><td>17.2</td><td>6.8</td><td>0.913</td><td>0.016</td><td>88.6</td><td>94.6</td><td>98.3</td><td>86.5</td><td>94.7</td><td>97.2</td></tr><tr><td>DA3 [25]</td><td>16.4</td><td>7.8</td><td>15.3</td><td>6.3</td><td>0.932</td><td>0.010</td><td>96.7</td><td>98.0</td><td>98.6</td><td>91.4</td><td>97.1</td><td>98.6</td></tr><tr><td>MapAny. [20]</td><td>42.8</td><td>22.2</td><td>30.2</td><td>12.5</td><td>0.856</td><td>0.033</td><td>69.2</td><td>85.9</td><td>95.5</td><td>56.5</td><td>82.4</td><td>90.9</td></tr><tr><td>VGGT-Ω [49] SPOTTER</td><td>13.3 12.9</td><td>9.7 2</td><td>9.0</td><td>5.5</td><td>0.908</td><td>0.013</td><td>92.4</td><td>98.0</td><td>99.1</td><td>87.8</td><td>95.7</td><td>97.9</td></tr><tr><td></td><td></td><td></td><td>8.3</td><td></td><td>0.914</td><td>0.013</td><td>92.7</td><td>97.7</td><td>99.0</td><td>86.5</td><td>95.1</td><td>97.3</td></tr><tr><td colspan="2">DTU (22 samples)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VGGT [48]</td><td>1.2</td><td>0.7</td><td>1.8</td><td>1.0</td><td>0.699</td><td>0.009</td><td>97.4</td><td>98.0</td><td>98.6</td><td>98.1</td><td>99.4</td><td>99.7</td></tr><tr><td>π3 [52]</td><td>1.2</td><td>0.7</td><td>1.5</td><td>0.6</td><td>0.695</td><td>0.014</td><td>91.2</td><td>96.9</td><td>98.7</td><td>80.0</td><td>93.2</td><td>96.6</td></tr><tr><td>DA3 [25]</td><td>1.7</td><td>1.0</td><td>1.6</td><td>0.8</td><td>0.711</td><td>0.009</td><td>96.5</td><td>97.9</td><td>98.6</td><td>91.0</td><td>97.0</td><td>98.5</td></tr><tr><td>MapAny. [20]</td><td>2.4</td><td>1.3</td><td>1.9</td><td>0.8</td><td>0.702</td><td>0.067</td><td>37.8</td><td>55.2</td><td>77.8</td><td>15.0</td><td>53.3</td><td>73.4</td></tr><tr><td>VGGT-Ω [49]</td><td>1.2</td><td>0.7</td><td>1.7</td><td>0.6</td><td>0.701</td><td>0.009</td><td>97.1</td><td>98.0</td><td>98.7</td><td>91.1</td><td>97.0</td><td>98.5</td></tr><tr><td>SPOTTER</td><td>1.2</td><td>0.7</td><td>1.6</td><td>0.6</td><td>0.700</td><td>0.012</td><td>95.9</td><td>97.5</td><td>98.6</td><td>86.1</td><td>95.3</td><td>97.7</td></tr><tr><td colspan="2">ETH3D (13 samples)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VGGT [48]</td><td>291.4</td><td>190.7</td><td>317.5</td><td>204.4</td><td>0.841</td><td>0.045</td><td>52.6</td><td>69.5</td><td>88.4</td><td>37.6</td><td>58.5</td><td>71.1</td></tr><tr><td>π³ [52]</td><td>154.7</td><td>87.2</td><td>129.7</td><td>65.1</td><td>0.890</td><td>0.032</td><td>62.4</td><td>78.4</td><td>93.4</td><td>36.4</td><td>58.7</td><td>72.9</td></tr><tr><td>DA3 [25]</td><td>274.3</td><td>197.3</td><td>270.3</td><td>170.1</td><td>0.844</td><td>0.027</td><td>73.2</td><td>84.8</td><td>95.4</td><td>55.4</td><td>69.5</td><td>75.4</td></tr><tr><td>MapAny. [20]</td><td>459.7</td><td>377.7</td><td>542.0</td><td>413.9</td><td>0.812</td><td>0.161</td><td>51.9</td><td>65.7</td><td>81.3</td><td>33.8</td><td>54.4</td><td>65.6</td></tr><tr><td>VGGT-Ω [49]</td><td>73.6</td><td>43.5</td><td>61.1</td><td>34.5</td><td>0.915</td><td>0.017</td><td>83.5</td><td>94.5</td><td>99.2</td><td>66.9</td><td>84.9</td><td>91.8</td></tr><tr><td>SPOTTER</td><td>65.5</td><td>37.1</td><td>52.9</td><td>30.3</td><td>0.924</td><td>0.017</td><td>82.8</td><td>94.3</td><td>98.7</td><td>65.4</td><td>84.8</td><td>91.8</td></tr><tr><td colspan="2">NAVI (281 samples)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VGGT [48]</td><td>2.5</td><td>1.6</td><td>3.4</td><td>2.7</td><td>0.751</td><td>0.032</td><td>71.5</td><td>87.1</td><td>96.3</td><td>53.3</td><td>82.7</td><td>91.2</td></tr><tr><td>π³ [52]</td><td>2.2 3.3</td><td>1.4 2.0</td><td>3.0</td><td>2.4</td></table>