# UI-VISA: U-Net Initialized Vascular Image Segmentation Architecture

Asees Kaur Suzanne S. Sindi Erica M. Rutter

University of California, Merced

akaur101@ucmerced.edu, ssindi@ucmerced.edu, erutter2@ucmerced.edu,

## Abstract

Accurate segmentation ofvascular structures in digital subtraction angiography (DSA) images remains challenging due to the thin, elongated, and branching nature of blood vessels. Pixel-wise deep learning approaches such as U-Net achieve strong general-purpose segmentation performance but often produce fragmented or discontinuous predictions infine vascular regions, since they do not explicitly enforce structural connectivity. Region growing algorithms preserve spatial context and topological continuity, but are highly sensitive to seed point initialization and can be computationally expensive. We propose UI-VISA (U-Net Initialized Vascular Image Segmentation Architecture), a hybrid pipeline that combines the complementary strengths of both approaches. UI-VISA uses U-Net’s foreground predictions as informed seed points for a CNN-guided region growing algorithm, which then iteratively refines the segmentation by enforcing local connectivity and recovering fine vessel details that U-Net alone tends to miss or overpredict. We evaluate UI-VISA against standalone U-Net and a prior region-growing-based method (VISA) using 5- fold cross-validation on 26 DSA images. UI-VISA achieves the highest mean Dice and clDice scores across folds, and a paired Wilcoxon signed-rank test shows the improvement in clDice is statistically significant (p = 0.023), consistent with the method’s design goal ofpreserving vascular connectivity, while the improvement in Dice does not reach significance (p = 0.104).

## 1. Introduction

With advancements in medical imaging techniques such as X-rays, ultrasound, computed tomography (CT), and magnetic resonance (MR) scans, computer-aided image segmentation can extract specific regions of interest (RoIs) for further evaluation, such as organ identification and tumor detection [3]. Biomedical image segmentation, which involves partitioning images into meaningful segments, is particularly significant in both research and clinical settings [16, 18]. By segmenting medical images, clinicians can identify and isolate regions of interest, such as organs or lesions, from complex CT and MRI scans [22].

Cerebrovascular diseases (CVDs) are conditions that slow blood flow to the brain, encompassing aneurysms, strokes, arteriovenous malformations, and arteriovenous fistulas, and are one of the leading causes of death today [28]. While CT and MR imaging techniques are used to detect strokes [9], digital subtraction angiography (DSA) is the standard-of-care imaging method for diagnosing and guiding treatment [19], due to it’s ability to clearly visualize blood vessels. This extraction allows clinicians to make objective, data-driven decisions regarding vascular health. However, as illustrated in Figure 1, manually tracing vessel boundaries is tedious, error prone, and highly susceptible to inter- and intra-observer variability.

![](images/5c42c2a9becd9203166b67026d4b897b4c795ef9021fdee6393af65cfdb76847.jpg)  
(a) DSA Image

![](images/afad40b5dc612be15df5bddac4f510b50119f336103f91a250cf0a1d85bcf202.jpg)  
(b) DSA Mask  
Figure 1. Left (a) is an example of a typical DSA Image, acquired during angiography, and on the right (b) is its corresponding manually traced mask we use as ground truth.

To address these inconsistencies, machine learning methods—specifically convolutional neural networks (CNNs)—have become the standard for biomedical image segmentation tasks. Traditional techniques such as edge detection and thresholding have largely been surpassed by deep learning, which excels at hierarchical feature representation and produces strong segmentation results even under image noise, blur, or low contrast [22, 24, 25]. Architectures such as U-Net [17] are particularly effective in this domain, using an encoder-decoder structure and skip connections to learn localized features while retaining spatial information lost during downsampling. However, because U-Net processes the entire image or tile at once, it struggles with the fine, elongated, and branching structures characteristic of vascular imagery [8], often producing patchy probability maps with discontinuous vessel segments and isolated artifacts [6, 11]. In addition to these architectural issues, medical image analysis often faces a shortage of labeled training data.

Region growing algorithms (RGAs) offer a complementary approach. Unlike U-Net, which classifies each pixel independently based on learned features, region growing makes each pixel’s inclusion explicitly conditional on connectivity to the growing region, enforcing the topological continuity expected of vascular structures. Recent work proposed VISA (Vascular Image Segmentation Architecture), a two-step pipeline of a CNN-based region growing algorithm [8, 10, 13] that showed success in segmenting veinous structures. However, region growing is sensitive to the choice of initial seed points, as poor initialization can cause the algorithm to miss vessel branches or grow into irrelevant background regions [15].

To address these limitations, we propose UI-VISA (U-Net Initialized Vascular Image Segmentation Architecture), a hybrid pipeline that combines the strengths of both approaches. The modified U-Net first performs an initial segmentation and the resulting foreground predictions then serve as seed points for the RGA. UI-VISA refines the initial segmentation by enforcing local connectivity and recovering fine structural details that U-Net may have missed or incorrectly predicted, while also reducing unnecessary computation and eliminating seed bias. Our contributions with UI-VISA are three-fold:

1. UI-VISA replaces random seeding in a CNN-based region growing algorithm with smarter, informed seed selection guided by U-Net predictions. This effectively combines the strengths of U-Net and region growing algorithms.

2. UI-VISA enforces local connectivity by exploiting region growing. Since the region-growing algorithm is applied after the initial U-Net segmentation, we are able to correct patchy or non-contiguous segmentations.

3. UI-VISA achieves statistically significant improvement in clDice scores compared to U-Net alone, demonstrating better preservation of vascular topology.

Figure 2 displays the pipeline of UI-VISA, in which a trained U-Net provides initial seeds (left) to a RGA (middle) to produce a final reconstruction (right).

## 2. Related Work

Our approach draws on two lines of prior work: automated segmentation methods for DSA imagery, and region growing algorithms for preserving vascular connectivity. We

briefly describe both before presenting UI-VISA.

## 2.1. Digital Subtraction Angiography Image Segmentation.

Digital subtraction angiography (DSA) provides dynamic imaging of cerebral vessels with high spatial and temporal resolution and is considered the gold standard for diagnosing cerebrovascular diseases (CVDs) [5, 27]. Extracting vessel structure from DSA images for further analysis has traditionally relied on manual tracing, which motivated the development of automated segmentation methods [27].

Non-deep-learning-based approaches to DSA vessel segmentation use hand-crafted feature enhancement rather than learned representations. One method uses a multi-scale Hessian matrix approach that enhances vessel edges, filters out vessel-like noise, and fuses multiple frames from a DSA sequence to better visualize the overall vascular structure [14]. However, the authors note that this multi-scale enhancement itself introduces blurring at vessel edges [14].

To reduce the dependence on manual and parameterbased methods, more recent work uses deep learning based approaches for vascular segmentation. CNNs are the most widely used, with U-Net [17] being the most common architecture applied to DSA vessel segmentation [28].A U-Net has also been pretrained on large amounts of unlabeled angiograms using an image-to-image translation framework on unannotated angiograms and then fine-tuned on a small set of labeled images to perform vessel segmentation [26]. Furthermore, attention mechanisms, which allow the network to focus on most relevant spatial regions, have also been incorporated into U-Net to better capture the scale variation and fine vessel detail characteristic of cerebral DSA images [2]. However, none of the above methods explicitly account for connectivity of the vessels, often resulting in fragmented or discontinuous segmentations.

## 2.2. Region Growing Algorithms.

It has been recognized that classifying pixels independently - as many segmentation methods do - can yield predictions that are locally plausible but globally inconsistent with underlying structures, particularly through producing fragmented or disconnected regions that fail to respect contiguity of physical and biological objects [4]. Region growing algorithms (RGAs) were developed to address this limitation. Rather than treat each pixel independently, a pixel’s inclusion in a region is made explicitly conditional on it’s similarity to, and connectivity with, a set of neighboring pixels. Specifically, RGAs consider an initial seed and expand iteratively to neighboring pixels until no more similar pixels remain [8]. Because the connectivity constraint is enforced, region growing approaches are especially well suited to vascular segmentation where the underlying biology consists of thin and branching structures.

![](images/b5eff79353f25c168a70a2d5320a62954d7413b13ec94a16a76afd9d76193638.jpg)  
Figure 2. Proposed UI-VISA framework for vessel segmentation. The left panel shows a test image segmented by U-Net, with predicted foreground pixels (red) used as initial seeds. The middle panel illustrates the iterative seed expansion pipeline: 80 × 80 image patches centered on current seeds are fed into a trained CNN to predict 3 × 3 mask patches, with newly predicted foreground pixels added as seeds and the process repeated until convergence. The right panel shows the final reconstructed vessel segmentation.

Non-deep learning RGA variants have been applied directly to vascular data in several contexts. In [1], RGA was applied to vessel segmentation in MR angiography. Other work used frequency-domain information to automatically identify seed points for region growing, rather than placing them manually, by extracting a vessel’s orientation from its spectral characteristics [7]. In a related approach, [15] proposed a coronary artery method that adapts the search area at each growth step to follow vessel geometry.

More recently, region growing techniques have been combined with CNNs to replace the handcrafted similarity criteria with learned features. One method uses CNNs to predict neighboring pixel intensities to decide which pixels need to be added to the growing region [8]. Other work has instead combined a CNN with a graph neural network to explicitly model the connectivity between vessel segments during segmentation [20].

## 3. Methods

## 3.1. Data

The dataset consists of 26 DSA images of size 512 × 512 pixels from distinct patients and their manually traced masks [28]. Images are normalized by dividing pixel intensities by the maximum value in the image. We used 5-fold cross-validation in which the 26 image-mask pairs were randomly split into 5 folds. Since 26 does not divide evenly by 5, one fold contains 6 images for testing while the remaining four folds each contain 5, resulting in 20 or 21 images for training per fold respectively. The training set was further divided into 80% training data and 20% validation set used to select the best model weights and determine the threshold.

## 3.2. Modified U-Net

To establish a baseline and compare with previous results [28], we trained a modified U-Net architecture. Given the limited size of the dataset (26 DSA images of 512 × 512 pixels), patches of size 128 × 128 pixels were extracted as network inputs, following [28], with 200 patches randomly sampled per image to increase the effective training set size. Validation patches were extracted identically but without augmentation. Training patches were augmented via random 90 rotations, horizontal/vertical flips, and random scaling (factor sampled uniformly from [0.8, 1.2]), each applied independently with probability 0.5 and identically to both image and mask; patches were re-cropped or reflection padded to maintain the fixed 128 × 128 input size.

The modified U-Net architecture, illustrated in Figure 3, follows an encoder-decoder design with skip connections. The encoder consists of four double 3 × 3 convolutional blocks (ReLU, dropout 0.2) with channels progressing from 64 to 512, followed by a 1024-channel bottleneck; the decoder mirrors this structure, ending in a 1 × 1 convolution producing a single-channel vessel probability map. The model was trained using SGD (momentum 0.9, initial learning rate 0.01, batch size 32) with binary cross-entropy loss on raw logits. Early stopping (patience 25 epochs, monitored on validation loss) was used, retaining the checkpoint with lowest validation loss as the final model for each fold.

![](images/dce49eaf3a41e046a818f912977805cc65cd9afa897443a98c9015b547b8d8db.jpg)  
Figure 3. Modified U-Net Architecture. The encoder (left) consists of four double convolution blocks with ReLU activations, each followed by max pooling, progressively reducing the spatial dimensions from $1 2 8 \times 1 2 8$ to $8 \times 8$ while increasing the feature channels from 64 to 512. A bottleneck block at the deepest level expands the channels to 1024. The decoder (right) mirrors the encoder, restoring spatial resolution through upsampling and concatenating encoder feature maps via skip connections (horizontal arrows). A final $1 \times 1$ convolution produces the output segmentation mask of size $1 2 8 \times 1 2 8 \times 1$

## 3.3. VISA: Vascular Image Segmentation Architecture

VISA is a two-stage pipeline for CNN-based region rowing algorithm. In Stage 1, a patch-based CNN is trained on a small $8 0 \times 8 0$ pixel image patch to predict foreground locations in a $3 \times 3$ pixel mask. In Stage 2, a RGA uses the trained CNN from stage 1 to iteratively grow foreground pixels. Below, we describe each stage.

Stage 1: CNN Training. To generate the patches for training, images and masks were zero-padded by 40 pixels on all sides (yielding $5 9 2 \times 5 9 2$ images) to ensure that regions centered near the original image boundaries remain fully defined. For each pixel $( i , j )$ in the original image, we extract an $8 0 \times 8 0$ patch from the padded image and a corresponding $3 \times 3$ patch from the padded mask, both centered at $( i + 4 0 , j + 4 0 )$ . This yields 262,144 image–label pairs per image, one for every pixel. To increase training diversity, we applied random $9 0 °$ rotations and horizontal/vertical flips to each training image and mask, each with probability 0.5. Each transformation that was applied produced an additional image-mask pair, meaning each original training image could yield up to four versions (the original plus up to three augmented copies).

Because vessel pixels (foreground) are vastly outnumbered by background pixels, we balanced the training data prior to training. Specifically, for each $3 \times 3$ mask patch, we computed the sum of its binary entries (0 = full background, 1 = full foreground), producing a value from 0 to 9. In each image, we identified the class with the minimum number of samples and then subsampled the remaining classes with replacement to ensure balanced representation.

![](images/8a02ba5ab932e99fdd1628ecb4d3e470952f388e7ead45009d145ea73053a8bb.jpg)  
Figure 4. VISA Network Architecture: Convolutional neural network (CNN) architecture consisting of layers with double convolution and ReLU (purple), max pooling (yellow), and a final single convolution (red). The network takes an $8 0 \times 8 0 \times 1$ grayscale patch as input. The dimensions labeled on each layer reflect the output of that layer, starting from $8 0 \times 8 0 \times 6 4$ after the first double convolution and ending at $3 \times 3 \times 1$

VISA uses a 25-layer CNN resembling the encoder portion of U-Net [17]: five double-convolutional blocks (ReLU-activated), each followed by max pooling, and a final $1 \times 1$ convolution producing a $3 \times 3 \times 1$ output representing foreground/background probability. The architecture is shown in Figure 4. During training, the model was optimized using the Adam optimizer [12] and the binary cross entropy loss function. Training was conducted on a GPU and the process spanned 50 epochs, with a batch size of 50 image-label pairs for the training data. To increase the diversity of training data seen by the model at each epoch, the balanced patch extraction procedure was re-run at the start of every epoch. This means a different random subset of patches was sampled from each category for each epoch, rather than reusing the same fixed set of patches throughout training.To make a binary prediction, the probability of foreground was selected per fold based on validation set experiments, ranging from 0.4 to 0.6, where pixels with a predicted probability greater than the threshold are classified as foreground.

Stage 2: Region Growing Algorithm. Once the CNN is trained, the RGA is used to generate a contiguous vessel segmentation efficiently without processing all pixels in the image. Starting from a set of initial seed pixels (denoted by the yellow pixel in Figure 5a), the algorithm extracts an 80×80 image patch around each seed, passes it through the trained network from Stage 1, and obtains a 3×3 prediction of foreground/background centered around that seed (Fig ure 5b,c). Pixels predicted as foreground are added to the segmented region and the algorithm repeats the process on their neighbors – extracting new patches, predicting, and expanding only in the direction of foreground pixels (Figure 5d). The algorithm terminates when no new foreground pixels are found. In practice, we start with 500 initially selected random pixels in our 512 × 512 image to serve as our seeds.

![](images/d84d81f93afea187a9e4a61d5db9a36f0ce235c08cb495e17810e2a2f7d98ca7.jpg)

![](images/f2a0ec7947c17c7ea2e98d8b155c5134ea8ec5807c5117eeb2a2fe99802fc579.jpg)

(a) Single initial foreground seed (b) Candidate pixels that will be prepixel in yellow dicted by the trained network in red  
![](images/daad0a0221d4077cbe03e4af6d23465aafc2804afc4c3337a4a4608c10a9e2ab.jpg)  
(c) Foreground (yellow) and back ground (purple) predicted pixels

![](images/e5650b01d817dd68add5ab5137da79532674cd34e8d12e6c637b56baaff39d28.jpg)  
(d) Next iteration of candidate pixels (red) to be predicted by network  
Figure 5. Illustration of Stage 2 of VISA. A single initial seed pixel (yellow) is shown in a. Panel b identifies the next candidate pixels (red) that will be evaluated by the trained network. Panel c shows the predicted classification of pixels into foreground (yellow) and background (purple). Panel d highlights the next set of candidate pixels (red) for the subsequent iteration of the network evaluation.

## 3.4. Proposed method: UI-VISA

UI-VISA attempts to mitigate the stochastic nature of VISA predictions by replacing the random selection of seed points. Stage 1 of UI-VISA and VISA are identical. The only difference in Stage 2 of UI-VISA is that we replace the 500 randomly selected initialization pixels with the predicted foreground pixels of U-Net. The subsequent RGA is identical to VISA. As illustrated in Figure 2, the UI-VISA testing pipeline begins with the modified U-Net (see Section 3.2) producing a full segmentation of the test image, as shown in the left panel of the figure. The U-Net output is thresholded at 0.5 to produce a segmentation mask. The resulting foreground pixels, shown in red at bottom of the left panel, are then used as the initial seeds for the RGA, the second stage of VISA. Since each pixel may be predicted multiple times due to the overlapping nature of the patches, the predicted probabilities are averaged across all overlapping predictions before the final threshold is applied. See Algorithm 1 for full details.

It is important to note that the final segmentation is produced entirely by the RGA, not U-Net. The U-Net predictions serve solely as an informed starting point, and the RGA then refines this initialization by expanding around the seed pixels. This refinement recovers fine vessel details and enforces local connectivity in regions where the modified U-Net may have produced fragmented or incomplete predictions. This refinement step is critical for vascular structures, where connectivity and continuity are clinically important. In addition to adding foreground, the refinement can also eliminate false positive foreground pixels segmented by U-Net.

## 3.5. Evaluation Metrics

To quantify the accuracy of segmentation, we use both Dice score and clDice score. While the Dice score measures the overall overlap between the predicted segmentation and the ground truth, it does not explicitly account for the connectivity or topology of thin, elongated structures such as blood vessels. As a result, segmentations with similar Dice scores may differ substantially in terms of vascular continuity, with breaks or disconnections that are clinically relevant. To address this limitation, we also evaluate segmentation performance using centerline Dice (clDice) [21], a metric specifically designed for vascular structures. clDice assesses how well the centerlines of the predicted and ground truth segmentations align with each other, thereby emphasizing connectivity and structural correctness.It is computed by first extracting the centerlines (skeletons) of both the predicted segmentation and the ground truth mask. The clDice score is then defined as:

$$
\mathrm { c l D i c e ( s k e l e t o n ) } = \frac { 2 \times \left| \mathrm { S k e l e t o n _ { s e g } \cap S k e l e t o n _ { g t } } \right| } { \left| \mathrm { S k e l e t o n _ { s e g } } \right| + \left| \mathrm { S k e l e t o n _ { g t } } \right| } .\tag{1}
$$

A higher clDice value indicates better preservation of vessel continuity and topology. Here, $\mathbf { S k e l e t o n } _ { \mathrm { s e g } }$ represents the skeleton of the segmented image, and Skeleto $^ { 1 } \mathrm { g t }$ represents the skeleton of the ground truth mask.

Algorithm 1 UI-VISA Testing Algorithm duces a coherent segmentation. In the second row, we   
Require: Trained U-Net $^ { g , }$ trained VISA CNN $f ,$ test im- present a case in which U-Net achieves the best Dice of   
age I, threshold τ 0.7920, while both VISA (0.5102) and UI-VISA(0.5963)   
Ensure: Predicted segmentation mask $\hat { Y }$ struggle. It appears that UI-VISA introduces extra false   
1: Normalize I by dividing by its maximum pixel value positives by including the skull. In the third row, all three   
2: Pad I with 40 zeros on all sides to obtain $I _ { \mathrm { p a d } }$ methods perform similarly, suggesting that this particular   
3: Initialize $\hat { Y } ( i , j ) \gets 0$ for all pixels $( i , j )$ image is equally challenging for all approaches.   
4: Initialize pixel score dictionary $\mathcal { P }  \{ \}$ Table 1 summarizes the cross-validation results for the   
5: Initialize visited set $S \gets \emptyset$ UI-VISA, U-Net, and VISA models across all five folds,   
6: Obtain U-Net foreground predictions: $\mathcal { C } _ { 0 }  \{ ( i , j )$ with highest values highlighted in bold. UI-VISA achieves   
$g ( I ) ( i , j ) \geq 0 . 5 \}$ the highest Dice and clDice scores in four out of five   
7: $t \gets 0$ folds, demonstrating that replacing random seed initializa-  
8: while $\mathcal { C } _ { t } \neq \emptyset$ do tion with informed U-Net predictions consistently improves   
9: for all $( i , j ) \in \mathcal { C } _ { t }$ do segmentation performance over the random-seed initializa-  
10: Extract $8 0 \times 8 0$ patch $\mathcal { S } ( i , j )$ from $I _ { \mathrm { p a d } }$ centered tion in VISA. Across folds 0–3, the hybrid model outper  
at $( i , j )$ forms both standalone methods in both metrics, with Dice   
11: Obtain $3 \times 3$ prediction mask $P  f ( S ( i , j ) )$ scores ranging from 0.7695 to 0.8227 and clDice scores   
12: for all $( \mathrm { r o w } , \mathrm { c o l } ) \in P$ do ranging from 0.7600 to 0.8084. The only exception is Fold   
13: Append $P [ { \mathrm { r o w } }$ , col] to $\mathcal { P } [ ( i + \mathrm { r o w } - 1 , j +$ $^ { 4 , }$ where U-Net achieves the highest Dice (0.7942) and   
col $- 1 ) ]$ clDice (0.7634), while UI-VISA scores 0.7759 and 0.7587   
14: end for respectively. Even in this challenging case, we note that   
15: $S \gets S \cup \{ ( i , j ) \}$ UI-VISA still outperforms VISA, implying that the more   
16: end for stable seed initialization is an improvement. The last row of   
17: $\mathcal { C } _ { t + 1 } \gets \{ ( i , j ) \notin S : \mathrm { m e a n } ( \mathcal { P } [ ( i , j ) ] ) \geq \tau \}$ the table shows the average of the Dice and clDice scores   
18: $t \gets t + 1$ over all images, since each image is contained in the test   
19: end while dataset exactly once. The results over all images indicate   
20: Reconstruct mask: that UI-VISA is the highest-performing algorithm.   
21: for all $( i , j ) \in \mathscr { P }$ do Table 2 reports the inference time per fold for each Table 2 reports the inference time per fold for each   
22: if mean $( { \mathcal { P } } [ ( i , j ) ] ) \geq \tau$ then method. U-Net is by far the fastest, completing inference in method. U-Net is by far the fastest, completing inference in   
23: $\hat { Y } ( i , j ) \gets 1$ under 40 seconds per fold, since it only requires a forward under 40 seconds per fold, since it only requires a forward   
24: end if pass over extracted patches without any iterative expansion. pass over extracted patches without any iterative expansion.   
25: end for However, speed alone does not reflect segmentation quality However, speed alone does not reflect segmentation quality   
26: Remove padding by cropping rows and columns by 40 for vascular structures, where connectivity and continuity for vascular structures, where connectivity and continuity   
of $\hat { Y }$ to recover the original $5 1 2 \times 5 1 2$ mask are biologically essential properties. As demonstrated in Ta- are biologically essential properties. As demonstrated in Ta-  
27: return $\hat { Y }$ ble 1, U-Net consistently produces lower clDice scores than ble 1, U-Net consistently produces lower clDice scores than   
the hybrid approach across most folds, indicating that its the hybrid approach across most folds, indicating that its   
pixel-wise predictions fail to preserve the topological struc- pixel-wise predictions fail to preserve the topological struc  
ture of the vascular network, a critical limitation for clinica ture of the vascular network, a critical limitation for clinical

## 4. Results

Figure 6 presents a qualitative comparison of the three methods across three representative test images. The images use a color-coded scheme to visualize prediction accuracy: white represents true positives (correctly identified vessel pixels), black represents true negatives (correctly identified background), red represents false positives (background incorrectly classified as vessel), and yellow represents false negatives (vessel pixels incorrectly classified as background). Three different scenarios were selected to show the breadth of results. In the first row, VISA and UI-VISA vastly outperform the modified U-Net, which produces a poor segmentation. UI-VISA recovers well with a Dice of 0.7694, demonstrating that even when U-Net predictions are fragmented, the RGA refinement stage proapplications where vessel continuity directly informs diagnosis and treatment planning. While UI-VISA is significantly slower than U-Net, it is much faster than VISA, with improvement speeds ranging from 3-5 fold. We hypothesize that UI-VISA is quicker as it is only performing refinement instead of processing the entire image.

While the previous results in Figure 6 and Table 1 suggest that UI-VISA outperforms the other methods, we aimed to quantify if the improved performance was statistically significant. To validate the UI-VISA results, we performed a statistical hypothesis test comparing per-image performance against the U-Net baseline:

$H _ { 0 }$ : UI-VISA and U-Net perform equally, $H _ { 1 }$ : UI-VISA outperforms U-Net.

U-Net  
![](images/34baa6abe0f560d177696a935f743b9f69edecbb3362ee1ed989861c10dff532.jpg)  
Figure 6. Qualitative comparison of segmentation outputs from VISA, U-Net, and UI-VISA approaches across three test images. Dice and clDice scores are reported under each prediction.

Table 1. Summary of Cross-Validation Results for VISA, U-Net, and UI-VISA Models
<table><tr><td rowspan="2">Fold</td><td colspan="2">VISA</td><td colspan="2">U-Net</td><td colspan="2">UI-VISA</td></tr><tr><td>Dice</td><td>clDice</td><td>Dice</td><td>clDice</td><td>Dice</td><td>clDice</td></tr><tr><td>0</td><td> $0 . 8 1 2 6 \pm 0 . 0 2 9 0$ </td><td> $0 . 7 9 9 1 \pm 0 . 0 5 9 7$ </td><td> $0 . 8 1 6 5 \pm 0 . 0 5 9 7$ </td><td> $0 . 7 8 9 8 \pm 0 . 0 7 9 6$ </td><td> $\mathbf { 0 . 8 2 2 7 \pm 0 . 0 3 6 3 }$ </td><td> $\mathbf { 0 . 8 0 8 4 \pm 0 . 0 5 1 9 }$ </td></tr><tr><td>1</td><td> $0 . 8 1 3 3 \pm 0 . 0 4 3 1$ </td><td> $0 . 7 9 0 9 \pm 0 . 0 4 9 4$ </td><td> $0 . 8 1 0 6 \pm 0 . 0 5 0 7$ </td><td> $0 . 7 8 9 8 \pm 0 . 0 6 1 2$ </td><td> $\mathbf { 0 . 8 2 2 5 \pm 0 . 0 4 3 8 }$ </td><td> $\mathbf { 0 . 8 0 7 4 \pm 0 . 0 5 6 8 }$ </td></tr><tr><td>2</td><td> $0 . 7 4 3 3 \pm 0 . 0 4 0 6$ </td><td> $0 . 7 1 7 8 \pm 0 . 0 5 6 9$ </td><td> $0 . 6 9 0 5 \pm 0 . 1 9 1 0$ </td><td> $0 . 6 6 5 5 \pm 0 . 1 7 5 4$ </td><td> $\mathbf { 0 . 7 6 9 5 \pm 0 . 0 4 7 1 }$ </td><td> $\mathbf { 0 . 7 6 0 0 \pm 0 . 0 5 6 7 }$ </td></tr><tr><td>3</td><td> $0 . 7 7 9 5 \pm 0 . 0 6 9 2$ </td><td> $0 . 7 7 1 4 \pm 0 . 0 4 6 6$ </td><td> $0 . 7 7 4 7 \pm 0 . 0 6 8 7$ </td><td> $0 . 7 5 9 6 \pm 0 . 0 6 0 0$ </td><td> $\mathbf { 0 . 7 8 2 9 \pm 0 . 0 7 2 4 }$ </td><td> $\mathbf { 0 . 7 8 2 3 \pm 0 . 0 5 2 2 }$ </td></tr><tr><td>4</td><td> $0 . 7 5 8 5 \pm 0 . 1 3 4 7$ </td><td> $0 . 7 3 5 8 \pm 0 . 1 6 0 4$ </td><td> $\mathbf { 0 . 7 9 4 2 \pm 0 . 0 5 3 0 }$ </td><td> $\mathbf { 0 . 7 6 3 4 \pm 0 . 0 5 8 6 }$ </td><td> $0 . 7 7 5 9 \pm 0 . 1 0 4 1$ </td><td> $0 . 7 5 8 7 \pm 0 . 1 1 6 9$ </td></tr><tr><td>All images</td><td>0.7810 ± 0.0725</td><td> $\overline { { 0 . 7 6 1 0 \pm 0 . 0 8 8 7 } }$ </td><td> $\overline { { 0 . 7 7 8 8 \pm 0 . 1 1 1 0 } }$ </td><td> $\overline { { 0 . 7 5 5 0 \pm 0 . 1 0 9 5 } }$ </td><td> $\mathbf { 0 . 7 9 5 8 \pm 0 . 0 7 0 2 }$ </td><td>0.7843 ± 0.0756</td></tr></table>

Table 2. Testing Time per Fold for VISA, U-Net, and UI-VISA Models
<table><tr><td>Fold</td><td>VISA (s)</td><td>U-Net (s)</td><td>UI-VISA (s)</td></tr><tr><td>0</td><td>3074.99</td><td>38.60</td><td>1043.67</td></tr><tr><td>1</td><td>2222.99</td><td>32.70</td><td>695.17</td></tr><tr><td>2</td><td>4420.55</td><td>39.95</td><td>1004.08</td></tr><tr><td>3</td><td>2436.47</td><td>32.53</td><td>592.30</td></tr><tr><td>4</td><td>9341.06</td><td>33.26</td><td>1811.65</td></tr></table>

Because each of the 26 images appears in exactly one test fold, we obtained a single Dice and clDice score per image for both methods. For each metric, we computed the perimage difference between UI-VISA and U-Net and applied the Shapiro-Wilk test to assess whether these differences were normally distributed; for both metrics the p-value was below a significance level of $\alpha = 0 . 0 0 1$ (Dice: $p = 3 \times$ $1 0 ^ { - 6 } ;$ clDice: $p = 7 \times 1 0 ^ { - 6 } )$ , so we used the Wilcoxon signed-rank test rather than a paired t-test.

The Wilcoxon signed-rank test is a nonparametric test for paired data. It ranks the per-image differences between the two methods by size and checks whether they tend to fall on one side of zero, which would indicate that one method consistently outperforms the other [23]. For both Dice and clDice metrics, we computed the per-image difference between UI-VISA and U-Net across all 26 test images and used the one-sided Wilcoxon signed-rank test to evaluate our hypothesis that UI-VISA outperforms U-Net, at a significance level of α = 0.05.

UI-VISA scores higher on 17 of the 26 images for Dice and 18 of 26 for clDice. Each of the differences d was ranked by absolute value, and the ranks with positive sign were summed. Under the null hypothesis this sum would be close to the value expected if the signs were assigned at random. The p-value is the probability of observing a sum at least this large if the two methods performed equally. Using a significance level of α = 0.05, the improvement in clDice was statistically significant $( \mathtt { p } = 0 . 0 2 3 )$ , while the improvement in Dice was not $( \mathtt { p } = 0 . 1 0 4 )$ This is consistent with what UI-VISA is designed to do. clDice measures how well the vessel structure is preserved, while Dice measures pixel overlap, so an improvement in clDice suggests the RGA recovers connectivity that U-Net misses.

## 5. Discussion and Future Work

In this work, we presented UI-VISA (U-Net Initialized Vascular Image Segmentation Architecture), a hybrid pipeline that combines U-Net predictions with the region growing algorithm (RGA) of VISA to improve seed initialization and produce more accurate and connected vessel segmentations. The key distinction between the standalone VISA and UI-

VISA lies in seed initialization. In standalone VISA, 500 seeds are selected randomly from the image, meaning they can fall anywhere in the image including background regions. This results in a large number of initial candidate pixels that must be processed and discarded before the algorithm converges on meaningful vessel structures, driving up computational cost. In UI-VISA, seeds are drawn entirely from U-Net foreground predictions, meaning every seed is already located within a predicted vessel region. This informed initialization eliminates unnecessary expansion into background regions and significantly reduces the total number of iterations required for convergence. As a result, UI-VISA is substantially faster than standalone VISA, ranging from approximately 592 to 1812 seconds per fold compared to 2223 to 9341 seconds for standalone VISA, a reduction of roughly 3 to 5 times.

Taken together, these results demonstrate that UI-VISA gives the most favorable balance among the three methods: it is more accurate and better connected than U-Net alone, more computationally efficient than standalone VISA, and consistently preserves the vascular connectivity that is essential for reliable DSA image segmentation. The reduction in inference time relative to standalone VISA, combined with improved or comparable segmentation accuracy, supports UI-VISA as a practical and effective approach for DSA vessel segmentation.

Despite the computational savings achieved through informed seed initialization, UI-VISA remains substantiall slower than standalone U-Net inference. A promising direction for future work is to further reduce computational cost while preserving the connectivity advantages of the region growing approach. Rather than initializing seeds from all U-Net foreground predictions, a more targeted strategy would be to restrict seed selection to pixels lying on the boundary between predicted foreground and background regions in the U-Net output. Additionally, to avoid unnecessary computation, the RGA could be augmented with an early termination criterion: if the VISA CNN prediction at a given location agrees with the U-Net prediction, expansion in that direction is halted. This boundary-aware initialization strategy has the potential to significantly reduce the number of iterations required for convergence, bringing the computational cost of UI-VISA closer to that of standalone U-Net while retaining the topological correctness that makes region growing essential for vascular segmentation.

## 6. Acknowledgments

Part of this research was conducted using MERCED cluster (NSF-MRI, #1429783) and Pinnacles (NSF MRI, # 2019144) at the Cyberinfrastructure and Research Technologies (CIRT) at University of California, Merced.

## References

[1] Muder M. Almi’ani and Buket D. Barkana. A modified region growing based algorithm to vessel segmentation in magnetic resonance angiography. In 2015 Long Island Systems, Applications and Technology, pages 1–7, 2015. 3

[2] Y. Cui, J. Su, J. Zhu, et al. Spatial multi-scale attention uimproved network for blood vessel segmentation. Signal, Image and Video Processing, 17:2857–2865, 2023. 2

[3] Yanhui Guo and Amira S. Ashour. Neutrosophic sets in dermoscopic medical image segmentation. In Neutrosophic Set in Medical Image Analysis, chapter 11, pages 229–243. Academic Press, 2019. 1

[4] Robert M Haralick and Linda G Shapiro. Image segmentation techniques. Computer vision, graphics, and image processing, 29(1):100–132, 1985. 2

[5] Wolf-Dieter Heiss, Michael Forsting, and Hans-Christoph Diener. Imaging in cerebrovascular disease. Current Opinion in Neurology, 14(1):67–75, 2001. 2

[6] Nabil Ibtehaz and M. Sohel Rahman. Multiresunet: Rethinking the u-net architecture for multimodal biomedical image segmentation. arXiv preprint arXiv:1902.04049, 2020. 2

[7] H. Jiang, B. He, D. Fang, Z. Ma, B. Yang, and L. Zhang. A region growing vessel segmentation algorithm based on spectrum information. Computational and Mathematical Methods in Medicine, 2013:743870, 2013. 3

[8] John H. Lagregren and Erica M. Rutter and Kevin B. Flores. Region growing with convolutional neural networks for biomedical image segmentation. 2020. 2, 3

[9] Pragati Kakkar, Tarun Kakkar, Tufail Patankar, and Sikha Saha. Current approaches and advances in the imaging of stroke. Disease Models & Mechanisms, 14(12):dmm048785, 2021. 1

[10] Asees Kaur. Vascular Image Segmentation Architecture for 2D and 3D Biomedical Images. Ph.d. dissertation, University of California, Merced, Merced, CA, USA, 2026. 2

[11] R. Kimura, A. Teramoto, T. Ohno, K. Saito, and H. Fujita. Virtual digital subtraction angiography using multizone patch-based u-net. Physical and Engineering Sciences in Medicine, 43(4):1305–1315, 2020. 2

[12] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015. 4

[13] John Lagergren, Mirko Pavicic, Hari B Chhetri, Larry M York, Doug Hyatt, David Kainer, Erica M Rutter, Kevin Flores, Jack Bailey-Bale, Marie Klein, et al. Few-shot learning enables population-scale analysis of leaf traits in populus trichocarpa. Plant Phenomics, 5:0072, 2023. 2

[14] Yanping Luo and Linggang Sun. Digital subtraction angiography image segmentation based on multiscale hessian matrix applied to medical diagnosis and clinical nursing of coronary stenting patients. Journal of Radiation Research and Applied Sciences, 16(3):100603, 2023. 2

[15] G. Ma, J. Yang, and H. Zhao. A coronary artery segmentation method based on region growing with variable sector search area. Technology and Health Care, 28(S1):463–472, 2020. 2, 3

[16] George Papanastasiou, Nikos Dikaios, Jia Huang, Chunhao Wang, and Guang Yang. Is attention all you need in medical image analysis? a review. IEEE Journal of Biomedical and Health Informatics, 28(3):1398–1411, 2024. 1

[17] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation, 2015. 1, 2, 4

[18] Han Seo, Mohammadhadi Badiei Khuzani, Vijayaragavan Vasudevan, Chuan Huang, Hongliang Ren, Rui Xiao, Xun Jia, and Lei Xing. Machine learning techniques for biomed ical image segmentation: An overview of technical aspects and introduction to state-of-art applications. Medical physics, 47(5):e148–e167, 2020. 1

[19] Shirin Shaban, Bella Huasen, Abilash Haridas, Murray Killingsworth, John Worthington, Pascal Jabbour, and Sonu Menachem Maimonides Bhaskar. Digital subtraction angiography in cerebrovascular disease: Current practice and perspectives on diagnosis, acute treatment and prognosis. Acta Neurologica Belgica, 122(3):763–780, 2022. 1

[20] Seung Yeon Shin, Soochahn Lee, Il Dong Yun, and Kyoung Mu Lee. Deep vessel segmentation by learning graphi cal connectivity. Medical Image Analysis, 58:101556, 2019. 3

[21] Suprosanna Shit, Johannes C. Paetzold, Anjany Sekuboyina, Ivan Ezhov, Alexander Unger, Andrey Zhylka, Josien P. W. Pluim, Ulrich Bauer, and Bjoern H. Menze. cldice - a novel topology-preserving loss function for tubular structure segmentation. Technical University of Munich and Eindhoven University ofTechnology, 2021. 5

[22] Risheng Wang, Tao Lei, Ruixia Cui, Bingtao Zhang, Hongying Meng, and Asoke K. Nandi. Medical image segmentation using deep learning: A survey. IET Image Processing, 16(5): 1243–1267, 2022. 1

[23] Frank Wilcoxon. Individual comparisons by ranking meth ods. Biometrics Bulletin, 1(6):80–83, 1945. 8

[24] Wenjian Yao, Jiajun Bai, Wei Liao, Yuheng Chen, Mengjuan Liu, and Yao Xie. From cnn to transformer: A review of medical image segmentation models, 2023. 1

[25] Ying Yu, Chunping Wang, Qiang Fu, Renke Kou, Fuyu Huang, Boxiong Yang, Tingting Yang, and Mingliang Gao. Techniques and challenges of image segmentation: A review. 2019. 1

[26] Y. Zeng, H. Liu, J. Hu, et al. Pretrained subtraction and segmentation model for coronary angiograms. Scientific Reports, 14:19888, 2024. 2

[27] Jiong Zhang, Qihang Xie, Lei Mou, Dan Zhang, Da Chen, Caifeng Shan, Yitian Zhao, Ruisheng Su, and Mengguo Guo. Dsca: A digital subtraction angiography sequence dataset and spatio-temporal model for cerebral artery segmentation. IEEE Transactions on Medical Imaging, 44(6):2515–2527, 2025. 2

[28] Min Zhang, Chen Zhang, Xian Wu, Xinhua Cao, Geof frey S Young, Huai Chen, and Xiaoyin Xu. A neural network approach to segment brain blood vessels in digital sub traction angiography. Computer methods and programs in biomedicine, 185:105159, 2020. 1, 2, 3