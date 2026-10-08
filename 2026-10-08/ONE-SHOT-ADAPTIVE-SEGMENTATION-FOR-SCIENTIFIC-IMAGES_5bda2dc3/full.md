# ONE-SHOT ADAPTIVE SEGMENTATION FOR SCIENTIFIC IMAGES

Tejaswi V. Panchagnula, Allison M. Davis and Fengqing Zhu

Elmore Family School of Electrical and Computer Engineering Purdue University West Lafayette, Indiana, USA 47907 {tpanchag, davi1987, zhu0}@purdue.edu

## ABSTRACT

Scientific image segmentation methods rely on extensive annotation and task-specific training, limiting adaptation across imaging modalities and experimental conditions. We present a training-free, one-shot framework that specializes vision foundation models using a single annotated reference image. The framework combines DINOv3 representations with background-adaptive feature orthogonalization to suppress artifact-related feature directions, after which cosine similarity localizes candidate regions for SAM segmentation. We evaluate the framework on red-blood-cell microscopy, structured-illumination pool boiling, and chest radiography. Relative to the strongest baseline, the proposed method improves mean IoU by 5.91% and 78.62% on the microscopy and pool-boiling datasets, respectively, while achieving comparable performance on chest radiographs. These results demonstrate that one-shot reference conditioning can adapt general-purpose vision models to specialized scientific segmentation tasks.

Index Terms— In-Context Segmentation, Scientific Imaging, Vision Foundation Models, Training-Free Adaptation

## 1. INTRODUCTION

Scientific image segmentation enables quantitative analysis in applications such as multiphase flows, blood-cell microscopy, and radiography [1, 2, 3]. Unlike many natural images, scientific images often contain domain-specific structures alongside acquisition-dependent artifacts and background variations, making robust segmentation challenging. Deeplearning-based approaches, including convolutional neural networks (CNNs) and vision transformers (ViTs), have shown success in segmenting scientific images. However, training a supervised model for each experimental setting and segmentation task requires extensive expert annotation.

Vision foundation models offer an alternative by transferring representations learned from large-scale pretraining to new tasks without requiring task-specific training. Among these models, self-supervised feature encoders and promptable segmentation models provide complementary capabilities for feature localization and segmentation. Segment Anything (SAM) [4] generates detailed segmentation masks from spatial prompts, while self-supervised models such as DI-NOv3 [5, 6] provide patch-level representations that support semantic correspondence across images. However, their direct application to scientific images remains unreliable, motivating a generalist-to-specialist framework that adapts to a specific task from one annotated example.

Training-free one-shot segmentation transfers target information from a single annotated reference to localize corresponding structures in unlabeled images. Existing approaches include GF-SAM, which uses foreground and background reference features to guide SAM [7], and INSID3, which mitigates positional bias in DINOv3 representations through orthogonal feature projection [8]. However, these approaches do not adapt their feature correction to the background and imaging artifacts observed in the annotated reference.

We address this limitation with a training-free, backgroundadaptive framework for one-shot scientific-image segmentation, shown in Fig. 1. Our method uses a single annotated reference image and its corresponding mask to localize target structures and produce fine-grained segmentation masks. We evaluate our framework across pool-boiling imaging, red blood cell microscopy, and chest radiography, representing distinct scientific-imaging modalities and conditions.

Our main contributions in this paper are as follows:

• We present a training-free, one-shot framework that adapts frozen foundation vision models to scientific images using only a single annotated reference.

• We introduce a background-adaptive orthogonal projection that uses the annotated reference background to suppress background structures and imaging artifacts.

• We evaluate the framework across three scientific datasets, demonstrating its applicability to distinct structures, artifacts, and modalities.

![](images/c3b82628c878dfc87e35313b01e3de7b228f7bd12e0e667ab14a2aa27d414b6a.jpg)  
Fig. 1: Generalist-to-specialist one-shot segmentation across scientific imaging domains. A single annotated reference defines each target concept, and the same frozen DINOv3–SAM framework localizes and segments corresponding structures without task-specific training.

## 2. METHODOLOGY

The proposed framework combines the semantic representations of DINOv3 with the prompt-based segmentation capabilities of SAM. A single annotated reference example, consisting of an image and its corresponding mask, defines the target structures to be localized in subsequent target images. This formulation enables adaptation to a specific segmentation objective without additional training. A qualitative visualization of the complete framework, including feature matching, candidate-region localization, prompt generation, and final segmentation, is shown in Fig. 2.

## 2.1. Feature Extraction with DINOv3

The proposed method first passes the annotated reference frame and subsequent target frames through a frozen pretrained DINOv3 model [6]. DINOv3 is well suited to this setting because its self-supervised training produces patchlevel representations which are able to group semantically similar regions without predefined class labels. For each image, we extract the normalized patch tokens from the final attention layer, giving us the feature vector $\mathbf { F } \in \mathbb { R } ^ { H ^ { \prime } \times W ^ { \prime } \times D }$ where D is the embedding dimension and $H ^ { \prime }$ and $W ^ { \prime }$ are the dimensions of the patch-wise grid produced by the ViT. The annotated reference image can therefore define a new target class from its feature representation.

## 2.2. Background-Adaptive Feature Orthogonalization

Scientific images can contain acquisition-specific artifacts, including glare, occluded objects, and motion blur in the raw image. During post-processing, these artifacts may not be removed completely. To reduce the effects of these components in the segmentation process, we employ a latent-space orthogonal projection. The reference mask is downsampled to match the spatial dimensions of the DINOv3 patch grid. The resulting Boolean mask partitions the reference tokens into foreground and background sets, with the selected foreground tokens forming the feature set $\mathbf { F } _ { \mathrm { r e f } }$

We obtain a unit background direction, $\mathbf { v } _ { c } ,$ by averaging and L<sub>2</sub>-normalizing the background tokens. A raw foreground prototype, $\mathbf { p } _ { \mathrm { r a w } } \in \mathbb { R } ^ { D }$ , is similarly computed by averaging the foreground tokens. For any feature vector $\mathbf { x } ,$ its normalized projection onto the orthogonal complement of $\mathbf { v } _ { c } ,$ shown in (1) as

$$
\mathcal { P } _ { \perp } ( \mathbf { x } ) = \frac { \mathbf { x } - ( \mathbf { x } ^ { \top } \mathbf { v } _ { c } ) \mathbf { v } _ { c } } { \| \mathbf { x } - ( \mathbf { x } ^ { \top } \mathbf { v } _ { c } ) \mathbf { v } _ { c } \| _ { 2 } } .\tag{1}
$$

The projected reference features are $\mathbf { p } = \mathcal { P } _ { \perp } ( \mathbf { p } _ { \mathrm { r a w } } )$ . During inference, the same operation is applied independently to every target token f , producing $\mathbf { f } _ { i } ^ { \prime } = \mathcal { P } _ { \perp } ( \mathbf { f } _ { i } )$ . The resulting target feature tensor is formed by collecting the projected tokens, $\mathbf { \bar { F } } _ { \mathrm { o u t } } = [ \mathbf { f } _ { 1 } ^ { \prime } , \mathbf { f } _ { 2 } ^ { \prime } , \dots , \mathbf { f } _ { N } ^ { \prime } ] ^ { \top } \in \mathbb { R } ^ { N \times D }$ , where N is the number of image patches and D is the feature dimension. The reference mask need not contain every target instance, provided that the annotated regions adequately represent the target structure.

## 2.3. Localization, Prompt Generation, and Segmentation

To localize candidate target regions, we compute cosine similarity between the projected reference prototype p and every projected target token in $\mathbf { F _ { \mathrm { o u t } } }$ . The resulting scores are reshaped into a two-dimensional similarity map and thresholded to produce a binary candidate map. Because patchlevel similarity can create small gaps within otherwise coherent regions, a morphological hole-filling operation is applied to consolidate the detected structures. A Euclidean distance transform then assigns each foreground pixel its distance to the nearest background pixel. Local maxima of the distance map provide interior point prompts, while the enclosing bounds of each connected region provide bounding-box prompts. These prompts are supplied to SAM, which produces a fine-grained segmentation mask for the target structures represented by the annotated reference.

![](images/9173f065d5a343b6627c391a667750dbf13aec1c91a65d183e00bfc66060bb2d.jpg)  
Fig. 2: End-to-end one-shot segmentation framework. DINOv3 features from the target image are compared with a foreground prototype obtained from the annotated reference. The resulting similarity map is thresholded to localize candidate regions. A distance transform identifies interior point prompts, while connected regions provide bounding-box prompts for SAM2, which produces the final segmentation masks.

## 3. RESULTS

We evaluate the proposed method across red-blood-cell microscopy, structured-illumination pool boiling, and chest radiography. These datasets represent distinct imaging modalities, target structures, and background characteristics, allowing us to assess the adaptability of the same framework under different scientific imaging conditions.

## 3.1. Datasets

Red Blood Cell Microscopy. We first use the 100k-RBC-PathOlOgics dataset [9], which contains 240,790 segmented RBC images across nine classes with varying cell structures. We use a separate annotated reference for each class to assess the framework’s ability to localize and delineate small targets with subtle class-specific differences.

Structured Illumination Pool Boiling. We evaluate 20 poolboiling frames recorded using structured illumination (SI) and reconstructed for optical sectioning. Two different reconstruction methods are evaluated: the Sequence Hilbert Transform (SHT) reconstruction [10] and the root-meansquare (RMS) reconstruction [11, 12]. Residual illumination patterns, spatially varying contrast, and closely spaced features make accurate foreground localization particularly challenging. For each reconstruction method, the reconstructed frames serve as target images for segmentation.

![](images/4250e317a936c6841a42e44d82a6670e34fbac006786d219d54bbea61a854b9b.jpg)  
Fig. 3: Qualitative segmentation comparison on structuredillumination pool-boiling data. The proposed method recovers more of the required annotated features than SAM2 and GF-SAM while accurately reducing the enlarged and merged regions produced by INSID3.

Chest Radiography. Finally, we evaluate one-shot segmentation on the Montgomery County chest X-ray dataset [13], which contains 138 frontal radiographs with corresponding lung masks, including 58 tuberculosis-positive cases. This dataset tests adaptation to projection images with overlapping anatomy and low-contrast boundaries.

## 3.2. Segmentation Results

We evaluate segmentation performance using mean intersection over union (IoU) and Dice score. Table 1 compares the proposed method with SAM2, GF-SAM, and INSID3. On RBC microscopy, the proposed method exceeds GF-SAM, the strongest baseline, by 5.15 percentage points in mean IoU and 2.94 percentage points in Dice. The largest gains occur on pool-boiling images, where the proposed method exceeds INSID3 by 33.36 percentage points in mean IoU and 26.62 percentage points in Dice. On chest radiography, the proposed method achieves performance comparable to SAM2. Fig. 3 shows that SAM2 and GF-SAM recover only a subset of the bubbles in the pool-boiling target image, while INSID3 produces enlarged and merged masks. The proposed method recovers more instances while better preserving their shapes and boundaries. These results indicate that backgroundadaptive feature processing is particularly effective when scientific imaging artifacts dominate the learned feature representations.

Table 1: One-shot segmentation performance across scientific imaging domains. Best results are shown in bold and second-best results are underlined.
<table><tr><td>Dataset</td><td>Imaging Modality</td><td>Method</td><td>Mean Target IoU</td><td>Mean Target Dice (F1)</td></tr><tr><td>PathOlOgics (RBCs)</td><td>Microscopy</td><td>SAM2</td><td>0.8670</td><td>0.9201</td></tr><tr><td>PathOlOgics (RBCs)</td><td>Microscopy</td><td>GF-SAM</td><td>0.8720</td><td>0.9297</td></tr><tr><td>PathOlOgics (RBCs)</td><td>Microscopy</td><td>INSID3</td><td>0.7670</td><td>0.8644</td></tr><tr><td>PathOlOgics (RBCs)</td><td>Microscopy</td><td>Ours</td><td>0.9235</td><td>0.9591</td></tr><tr><td>Pool Boiling</td><td>SI Reconstructed</td><td>SAM2</td><td>0.0600</td><td>0.1127</td></tr><tr><td>Pool Boiling</td><td>SI Reconstructed</td><td>GF-SAM</td><td>0.1646</td><td>0.2761</td></tr><tr><td>Pool Boiling</td><td>SI Reconstructed</td><td>INSID3</td><td>0.4243</td><td>0.5950</td></tr><tr><td>Pool Boiling</td><td>SI Reconstructed</td><td>Ours</td><td>0.7579</td><td>0.8612</td></tr><tr><td>Montgomery Chest X-Ray</td><td>Chest Radiograph</td><td>SAM2</td><td>0.3962</td><td>0.5608</td></tr><tr><td>Montgomery Chest X-Ray</td><td>Chest Radiograph</td><td>GF-SAM</td><td>0.3805</td><td>0.5434</td></tr><tr><td>Montgomery Chest X-Ray</td><td>Chest Radiograph</td><td>INSID3</td><td>0.3044</td><td>0.4637</td></tr><tr><td>Montgomery Chest X-Ray</td><td>Chest Radiograph</td><td>Ours</td><td>0.3960</td><td>0.5610</td></tr></table>

## 3.3. Background Response Suppression

![](images/3d9dd0f9752a202edcbf21859adc53667b1eaa9e280ea24531ff9e669a5de26e.jpg)  
(a) Target Image

![](images/b13068d26c9baae12f129cce0b3657e30ce33a699af2370ee51c21636ec6496f.jpg)  
(b) Unprojected DINOv3

![](images/e3fdf50c1c5a3fa599037da64439d4a5f5f8c0d8bb97c9c45b1127c84b5c8027.jpg)  
(c) Gaussian Noise Projection

![](images/b19521e80d7690e8ccc839cc5fe3922498455ba610d3dca4af21a40ae45b24d9.jpg)  
(d) Background Adaptive  
Fig. 4: Effect of background-adaptive feature orthogonalization on an RMS-reconstructed pool-boiling frame. Compared with unprojected DINOv3 features and the Gaussiannoise projection, the proposed method suppresses background responses while preserving target-bubble activation.

Background-adaptive feature orthogonalization substantially suppresses residual background activation in the cosinesimilarity map. As shown in Fig. 4, the unprojected DINOv3 features and the Gaussian-noise projection used by INSID3 [8] retain elevated responses across the structured background. The proposed method reduces these responses to near zero while preserving strong, localized activation over the target bubbles. Across 20 pool-boiling frames, mean background similarity decreases from 0.3117 for the unprojected features and 0.2847 for the Gaussian-noise projection to −0.0020 with the proposed method. The contrast-to-noise ratio correspondingly increases from 2.74 and 2.64 to 2.97. These results indicate that a background direction estimated from the annotated reference provides stronger feature-space separation than a generic Gaussian-noise reference.

## 4. CONCLUSION

While scientific image segmentation is essential for quantitative analysis, existing approaches often require extensive expert annotation or perform unreliably under specialized imaging conditions. To address these limitations, we present a training-free, one-shot framework that adapts frozen vision foundation models for scientific image segmentation. Experiments on red blood cell microscopy, structured-illumination pool boiling, and chest radiography demonstrate the adaptability of the framework to distinct imaging modalities. On pool-boiling data, the proposed method exceeds the strongest baseline by 33.36 percentage points in mean IoU while substantially suppressing background responses in feature space. By reducing reliance on large, expert-annotated datasets, this background-adaptive framework can make specialized segmentation more accessible across scientific imaging domains.

## 5. REFERENCES

[1] Carlos E. F. do Amaral, Rafael F. Alves, Marco J. da Silva, Lucia V. R. Arruda, Leyza Dorini, Rigoberto´ E. M. Morales, and Daniel R. Pipa, “Image processing techniques for high-speed videometry in horizontal two-phase slug flows,” Flow Measurement and Instrumentation, vol. 33, pp. 257–264, October 2013.

[2] Ahmed Elsafty, Ahmed Soliman, and Yomna Ahmed, “1 million segmented red blood cells with 240k classified in 9 shapes and 47k patches of 25 manual blood smears,” Scientific Data, vol. 11, no. 1, pp. 722, July 2024.

[3] Leilei Zhou, Xindao Yin, Tao Zhang, Yuan Feng, Ying Zhao, Mingxu Jin, Mingyang Peng, Chunhua Xing, Fengfang Li, Ziteng Wang, et al., “Detection and semiquantitative analysis of cardiomegaly, pneumothorax, and pleural effusion on chest radiographs,” Radiology: Artificial Intelligence, vol. 3, no. 4, pp. e200172, July 2021.

[4] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollar, and Ross Girshick, “Segment anything,”´ Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pp. 4015–4026, October 2023, Paris, France.

[5] Mathilde Caron, Hugo Touvron, Ishan Misra, Herve´ Jegou, Julien Mairal, Piotr Bojanowski, and Armand´ Joulin, “Emerging properties in self-supervised vision transformers,” Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 9650–9660, October 2021, Virtual Conference.

[6] Oriane Simeoni, Huy V. Vo, Maximilian Seitzer, Fed-´ erico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michael Ra-¨ mamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timothee Darcet, Th´ eo´ Moutakanni, Leonel Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Herve J´ egou, Patrick Labatut,´ and Piotr Bojanowski, “DINOv3,” arXiv preprint arXiv:2508.10104, August 2025.

[7] Anqi Zhang, Guangyu Gao, Jianbo Jiao, Chi Harold Liu, and Yunchao Wei, “Bridge the points: Graph-based fewshot segment anything semantically,” Advances in Neural Information Processing Systems, vol. 37, pp. 33232– 33261, December 2024, Vancouver, BC, Canada.

[8] Claudia Cuttano, Gabriele Trivigno, Christoph Reich, Daniel Cremers, Carlo Masone, and Stefan Roth, “IN-

SID3: Training-free in-context segmentation with DI-NOv3,” Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21637– 21646, June 2026, Denver, CO, USA.

[9] Mohamed Elmanna, Ahmed Elsafty, Yomna Ahmed, Muhammad Rushdi, and Ahmed Morsy, “Deep learning segmentation and classification of red blood cells using a large multi-scanner dataset,” arXiv preprint arXiv:2403.18468, March 2024.

[10] Xing Zhou, Ming Lei, Dan Dan, Baoli Yao, Jia Qian, Shaohui Yan, Yanlong Yang, Junwei Min, Tong Peng, Tong Ye, and Guangde Chen, “Double-exposure optical sectioning structured illumination microscopy based on hilbert transform reconstruction,” PLOS ONE, vol. 10, no. 3, pp. e0120892, March 2015.

[11] Elias Kristensson, Structured Laser Illumination Planar Imaging: SLIPI Applications for Spray Diagnostics, Ph.D. thesis, Lund University, Lund, Sweden, March 2012.

[12] Mark A. A. Neil, Rimas Juskaitis, and Tony Wilson,ˇ “Method of obtaining optical sectioning by using structured light in a conventional microscope,” Optics Letters, vol. 22, no. 24, pp. 1905–1907, December 1997.

[13] Stefan Jaeger, Alexandros Karargyris, Sameer Antani, and George Thoma, “Detecting tuberculosis in radiographs using combined lung masks,” Proceedings of the Annual International Conference ofthe IEEE Engineering in Medicine and Biology Society, pp. 4978–4981, August 2012, San Diego, CA, USA.