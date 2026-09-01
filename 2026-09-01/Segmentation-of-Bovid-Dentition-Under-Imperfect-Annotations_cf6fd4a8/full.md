# Segmentation of Bovid Dentition Under Imperfect Annotations: A Comparative Study of Convolutional and Attention Models

Keith G. Mills   
LSU ATHENA Lab   
Baton Rouge, LA   
keith.mills@lsu.edu

Evan B. Sanders<sup>†</sup> LSU ATHENA Lab & EHSBR Baton Rouge, LA SandersE27@ehsbr.org

Gregory J. Matthews   
Loyola Dept. Math & Stats   
Chicago, IL   
gmatthews1@luc.edu

Juliet K. Brophy LSU Dept. Geography & Anthropology Baton Rouge, LA jbrophy@lsu.edu

## Abstract

Semantic segmentation decomposes an image into distinct mask regions corresponding to different object categories, such as people, cars, signs or buildings. Advances in machine learning (ML) have shifted this task away from traditional rule-based heuristics such as edge detection, towards deep neural networks (DNN) that learn to classify pixels directly. However, semantic segmentation DNNs crucially depend on expertly designed mask targets to learn from, and imperfect or misaligned masks can interfere with a model’s ability to learn effectively.

This paper presents a comparative study of segmentation architectures, ranging from convolutional backbones to vision transformers, applied to the B.O.V.I.D. dataset, a corpus ofhigh-resolution bovid dentalphotographspaired with hand-made segmentation masks not originally designedfor ML-based training. We evaluate a range of preprocessing and alignment techniques to mitigate the resulting label imperfections. We find that while these preprocessing choices have limited effect on quantitative metrics such as Dice score and mIoU, their qualitative impact on predicted masks is substantial.

## 1. Introduction

Paleoanthropologists often use the animals associated with human ancestors in order to reconstruct their past environment. Specifically, researchers rely on the teeth of animals in the Family Bovidae because these animals have relatively strict ecological tendencies. Bovid teeth, in particular isolated teeth, are also abundant in the fossil record. In order to help taxonomically identify these teeth, previous studies have relied upon morphometric analyses [4] on the form (shape and size) and shape of the occlusal, or chewing, surface in order to differentiate between the different taxonomic tribes and species [3, 5].

A photograph of the tooth of a known species was manually scaled and converted into a black and white image, binarized, using programs such as GIMP. The teeth were statistically analyzed and shown to differentiate from each other based on tribe, genus, and species. The success of this approach led to the application of this method to the fossil record. Using machine learning, the known teeth were compared to fossil teeth in order to look for fossil representatives of their modern counterparts and fossil relatives. The identifications of the fossil teeth have been used in paleoenvironmental reconstruction. The process of digitizing the teeth to binarize them has been performed by placing points around the occlusal surface of a tooth [3]. More points could be placed in areas where the shape is more complex. Once the outline was complete, the shape was filled in with black and the outside was converted to white. This method, while effective, is quite time consuming and creates imperfections compared to a ML-geared semantic segmentation dataset [6, 19].

The purpose of this paper is to investigate to what kind of machine learning solution is required to adequately recognize and isolate the bovid tooth in a raw, color image, and, with proper training, create a binarized image needed for analysis. As such, this paper performs a comparative study of different segmentation model designs [12] and rule-based processing heuristic techniques [9, 15] in order to combat these issues. Our detailed contributions are as follows:

We first compare a breadth of semantic segmentation architectures, ranging from traditional ResNets [10] to more lightweight MobileNets [16], EfficientNets [17] and Vision Transformer-based [8, 18] backbones to profile the efficacy of different encoder designs across architecture type, model size and technical complexity.

Second, we conduct our experiments on B.O.V.I.D. [2], a small dataset consisting of high-resolution bovid dental photos and black/white masks which we cast as a semantic segmentation task. A key challenge to working with this dataset is that the masks are handmade without the original intention of being used as training data, and thus contain imperfections which make traditional ML-based generalization techniques cumbersomely difficult if applied in a straightforward manner. To combat this, we consider a slew of preprocessing techniques such as downsampling, correction, and histogram equalization [15] alongside a range of different backbone feature extractor types.

Third, we compare and contrast our best methods and techniques on wholly new and out-of-distribution bovid dentation data to demonstrate the generalizability and efficacy of our approach towards labeling future data samples. Moreover, we provide a taxonomic analysis of segmentation performance across different tribes of Bovidae species.

Experimental results demonstrate a surprising finding in the segmentation ability of different encoders. Specifically, we find that while using different preprocessing techniques may not show much difference in terms of quantitative metrics such as Dice score and mIoU, the qualitative impact is much more substantial. Further, we find that lightweight, moderately complex CNN encoders such as MobileNetV2 [16] are surprisingly effective in their ability to handle this task, and are comparable to much newer, attention-drive SegFormer [18] encoders that are several times their computational size. Finally, we provide some insights on segmentation ability across different Bovidae tribes.

## 2. Bovidae Extant Dataset

We introduce and discuss the segmentation dataset considered in this paper. Specifically, we describe the original dataset on a high-level, the challenges imposed performing segmentation on the dataset, as well as how we prune it for suitability training and testing an ML segmentation model.

## 2.1. Dataset Characteristics

In this paper we consider the B.O.V.I.D. dataset [2] which consists of 3592 extant Bovidae teeth images of varying sizes and resolutions from 7 tribes: Alcelaphini, Antilopini, Bovini, Hippotragini, Neotragini, Reduncini and Tragelaphini. The average image resolution is (1288 × 1430) ± (351×492) pixels, with the smallest being (407×238) pixels while the largest image is (5712 × 4288). Each tooth is

![](images/07a8a8ee2667e0980c498fdbea333db9d9aebf08ff94c080c90499bef4fca747.jpg)  
Figure 1. Example (raw, mask) image pairs from the B.O.V.I.D. dataset. Note how for each image, the masked region denotes the specific tooth in question although there may be other teeth shown.

associated with two images:

1. Raw Image: Scaled color image of the occlusal surface of the tooth. These images were primarily sourced from three South African institutions: National Museum, Bloemfontein; Ditsong Museum, Pretoria; and Amathole Museum, King William’s Town. Other images were sourced from the Field Museum in Chicago, Illinois, U.S.A.

2. Mask Image: Black and white image of the tooth’s occlusal surface. These mask images were handcrafted from the original tooth image primarily by student workers using GIMP.

Each image includes metadata pertaining to the taxonomic Family, Tribe, Genus, and Species, as well as the original tooth side and type, i.e., first lower molar (LM1), second upper molar (UM2), tooth’s current location, accession number at institution, origin, etc. Figure 1 provides a sample illustration of several raw image-mask pairs.

In this paper the Raw Images serve as the input to a segmentation model, while the Mask Images serve as the supervised labels. Given the size of the data corpus, we utilize pre-trained off-the-shelf segmentation models and fine-tune them for this task.

## 2.2. Dataset Imperfections

Unlike classical semantic segmentation datasets [6, 19], B.O.V.I.D. was not initially created with ML-driven segmentation in mind. This imposes some unique challenges which make this task easier said than done.

Resolution Mismatch. There are several instances where the resolutions in a given (Raw Image, Mask Image) pair are not the same nor even proportional to each other as the major concern when creating the dataset was DPI matching to match size components. These pairs of images are unsuitable for use in supervised ML-driven segmentation as the segmentation model will produce an output proportional to the input Raw Image, which must then also be proportional to the target Mask Image.

![](images/3620eb97b8c5f3c75fee725fb0ef6e867d455bbdf03bb3c4a0949e24141533b5.jpg)  
Figure 2. Examples of misaligned Raw Image, Mask Image pairs. Left-hand-side: The tooth in the raw image is in the upper third of the picture primarily, while it is centered in the mask. Right-handside: The raw image picture has whitespace padding surrounding it so the tooth is not in the center of the whole image, but this is the case for the mask.

Raw-Mask Misalignment. There are also cases of (Raw Image, Mask Image) pairs where, even if the images have the same resolution, the actual location of the mask in the Mask Image does not visibly align with the actual tooth in the Raw Image. Figure 2 provides some sample example image pairs. This issue is more sinister than if the images are not proportional in resolution, as these (Raw Image, Mask Image) pairs can still be utilized for training, however, the spatial misalignment of the mask to the actual tooth will heavily disrupt the supervised learning process.

The ‘Catch-22’ to these issues is that adequate cropping one image to be proportional to another or to correct misalignment requires us to identify where the tooth and mask overlap and that is the objective we’re aiming to fine-tune a segmentation model on in the first place. However, there are still some non-ML-based techniques we can utilize to alleviate these issues and improve generalizability.

## 3. Augmentations

We discuss the techniques employed to train ML segmentation models on the B.O.V.I.D. dataset. First, we discuss pruning techniques to make the dataset cognizable by an ML model and perform well, before discussing different preprocessing techniques utilized to further increase segmentation performance.

## 3.1. Dataset Pruning & Cropping

The first step in our pruning process is to remove all (Raw Image, Mask Image) pairs where the resolutions are neither the same nor proportional to each other, i.e., they have different aspect ratios. This removes 283 image pairs, reducing the dataset down to 3325 image pairs for a reduction of less than 10%.

Second, to tackle the issue of misalignment in remaining pairs, raw images were spatially aligned to their corresponding binary masks using a foreground-centering procedure. Foreground pixels were identified by grayscale thresholding, a standard image-segmentation operation [9]. Pixels with intensities below 240 were classified as foreground, and their centroid was calculated to estimate the object center. A rectangular crop centered on this centroid was then extracted, following the general principle of translating image content into a common spatial reference frame [21].

The crop dimensions were defined as a scalar multiple α of the corresponding mask dimensions, with $\alpha \ : = \ : 1 . 0$ in this study. Regions extending beyond the original image boundaries were filled with white background pixels. The resulting crop was resized using bilinear interpolation to match the exact height and width of its corresponding mask. This produced pixel-compatible image-mask pairs for downstream segmentation. Image loading, color conversion, resizing, and file writing were implemented using OpenCV [1].

This leaves us with a dataset of 3325 image pairs with an average image resolution is $( 1 3 0 8 \times 1 4 5 8 ) \pm ( 3 3 8 \times 4 7 9 )$ pixels, with the smallest being (407 × 238) pixels while the largest image is $( 5 7 1 2 \times 4 2 8 8 )$ . This pruned and augmented dataset is the basis for all of our experiments. Next, we enumerate the online preprocessing techniques we utilize to enhance ML segmentation performance.

## 3.2. Online Preprocessing and Training

For each (Raw Image, Mask Image) pair, the Raw Image is loaded in RGB while the Mask Image is loaded in grayscale and then inverted so the masked region consists of white pixels, i.e.,

$$
y _ { i j } = { \left\{ \begin{array} { l l } { 1 , } & { { \mathrm { i f ~ t h e ~ g r a y s c a l e ~ m a s k ~ v a l u e ~ a t ~ } } ( i , j ) = 0 , } \\ { 0 , } & { { \mathrm { o t h e r w i s e . } } } \end{array} \right. }
$$

Thus, the resulting masks contain one foreground channel with values in {0, 1}.

Next, we apply one of three contrast-processing modes:

1. Histogram Equalization (HE): We convert the RGB image to the $Y C r C b$ color space apply equalization to the luminance channel.

2. Contrast-Limited Adaptive Histogram Equalization (CLAHE): We also convert the RGB image to $Y C r C b$ and apply an equalization transform to the luminance channel using an $8 \times 8$ tile grid and a configurable clip limit c.

3. No Equalization (NE): Equalization is not performed.

Next, in addition to several boilerplate or required preprocessing techniques, we optionally downsample images and masks by an integer factor d. Downsampling reduces GPU training and inference memory costs while also abating the effect of misalignment. Specifically, we downsample Raw Images using bilinear interpolation and downsample Mask Images by nearest-neighbor interpolation.

Following contrast processing, we pass the image through the encoder-specific ImageNet [7] preprocessing function by the segmentation model library [12] and convert it to a float-point tensor with channel-first ordering. Samples within each minibatch are zero-padded to the largest spatial height and width dimensions within the minibatch. Further, images and masks were optionally downsampled by an integer factor d. This reduces GPU training and inference memory consumption and also helps to abate the affect of misalignment. Specifically, we downsample Raw Images using bilinear interpolation and downsample Mask Images by nearest-neighbor interpolation to preserve binary labels. Finally, if necessary, we pad the height and width to be a multiple of 32 to match spatial requirements of encoderdecoder architectures.

Finally, we train our segmentation models using a combination of the binary cross-entropy and weighted Dice [13] losses as follows

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { B C E } } + \lambda _ { \mathrm { D i c e } } \mathcal { L } _ { \mathrm { D i c e } } , } \end{array}\tag{1}
$$

where $\lambda _ { \mathrm { D i c e } } \in [ 0 , 1 ]$ is a configurable scalar.

## 4. Results and Discussion

We now present our experimental results and findings. Specifically, we first briefly describe our experimental setup, before delving into the effectiveness of different histogram equalization preprocessing techniques, for both Transformers and CNNs. Next, we provide additional pertribe (e.g., like per-class in an ML-sense) segmentation results based on different species of bovine, before finally showing some results on brand new, out of distribution data samples captured in the summer of 2026. Additional ablations exist in the supplementary.

## 4.1. Experimental Setup

We randomly shuffle the B.O.V.I.D. dataset 80%/20% into disjoint training and testing sets. Semantic segmentation models come from the segmentation-models-pytorch [12] library. For transformers we utilize SegFormer MiT-B3 [18] while for CNNs we utilize the U-Net++ [20] model with either MobileNetV2 (MBv2) [16], ResNet-34 [10] or EfficientNet-B0 [17] serving as the image encoder backbone. We train each model for 70 epochs and set $\lambda _ { D i c e } =$ 0.3. Due to space constraints, the most salient hyperparameter choices are part of our methodology in Section 3 while we provide additional details and ablations, i.e., for $\lambda _ { D i c e }$ in the supplementary.

## 4.2. SegFormer Quantitative/Qualitative Results

We train SegFormer MiT-B3 for 70 epochs, varying the type of preprocessing from CLAHE with $c \in [ 1 0 , 2 5 ]$ as the clip limit, HE and NE. Quantitatively, we measure the binary cross entropy (BCE) loss, Dice loss, and Dice metric on the test set after every epoch. Figure 3 plots the results. All four configurations show similar trends: Sharp decrease in the BCE loss followed by a gradual rise, paired with a continuous decrease or increase of the Dice loss or metric, respectively. In terms of Dice, all four configurations are relatively equal and it is hard to discern which is superior, but this is not the case for test BCE loss as CLAHE with $c = 2 5$ consistently achieves a lower loss, even during the upswing after epoch 15 or so. Overall, these results demonstrate the feasibility of fine-tuning a SegFormer model on B.O.V.I.D. despite the segmentation imperfections in the data. However, what is more important is the actual predicted masks.

During training we track the epoch that corresponds to the best Dice loss and save the weights at this epoch as the checkpoint for inference. After training, we generate heatmaps where values range between 0 and 1, rather than being 0 or 1, to visualize model decision making. Figure 4 provides some sample sets of ground truth masks alongside the respective heatmaps generated by each preprocessing technique for SegFormer. For each type of preprocessing, the model predicts a mask in roughly the center of the region even though the ground truth may have the mask location shifted. What is more interesting is the shape of the mask and how well it conforms to the ground truth. Here, we can see that all methods draw a similar region for the first picture, while for the second, HE and NE struggle a bit more with capturing the small masked region shaped like an upside-down $\mathbf { \Delta } ^ { \bullet } \mathbf { Y } ^ { \bullet }$ in terms of heatmap values. For the third image, CLAHE 25 draws over the middle of the mask incorrectly while CLAHE 10 best captures the overall shape. For the final image, most methods except CLAHE 25 and NE tend to destabilize in some way, either missing critical shape details. Overall though, this model is capable of producing competent heatmaps, which can the be thresholded into accurate masks, for the B.O.V.I.D. dataset.

## 4.3. MBv2 Quantitative/Qualitative Results

Next, we study the capacity of older, but more computationally lightweight convolutional neural network architectures, to segment the B.O.V.I.D. dataset. Due to space constraints, we primarily consider MBv2 in the main body of this manuscript but provide ablations to demonstrate its performance over other CNNs in the supplementary. Figure 5 mirrors Fig. 3 for MBv2 by graphing the test metrics across training epochs. We make a few observations:

![](images/512474133b8ac3092615b24c1139e0ae21a3d7f200bb69c8ba4c353cc2f105fd.jpg)  
Raw Image  
Mask Image

![](images/7eb495cca2031afcd73f11706b2468b35be24847e94dd92413a0926e218605f8.jpg)  
CLAHE 25  
CLAHE 10

![](images/1158bd2a1297d9871ffc3e497f38cc9ea13a9b2ce0ca6c351fe27c5ba6c0ebd5.jpg)  
Figure 3. BCE loss, Dice loss and Dice metric for SegFormer MiT-B3. For losses, lower is better, while higher is better for the Dice metric.

HE  
NE  
![](images/d29736bf8ea3ea3761f934f98390751fd903309cf670c97c955df15638cc814b.jpg)  
Figure 4. Example Raw Images (including whitespace) and ground-truth Mask Images as well as inferred heatmaps by SegFormer MiT-B3 using different histogram equalization techniques. Yellow represents regions with high (∼ 1) likelyhood of being part of the tooth; purple represents low likelihood. Best viewed in color.

First, early on, learning is more stochastic on MBv2 than for SegFormer MiT-B3, with larger spikes in the test metrics by epoch, although this eventually smooths out by epoch 50 or so. Second, like SegFormer, the BCE loss drops sharply then gradually increases, though not as quickly as SegFormer, while the Dice loss and Dice metric fall and rise, respectively. Third, the lower-bound of the Dice loss is not as low as it is for SegFormer, nor is the upper bound of the Dice metric. All of these metric observations suggest a model that may also be competent at this task, but less

capable than SegFormer.

Figure 6 mirrors Fig. 4 and largely corroborates this hypothesis. Here, we can see that MBv2 relies on a good preprocessing histogram equalization strategy to generate accurate masks. Specifically, CLAHE c = 10 stands out as best representing the 2nd and 4th images which have the most complex shapes to accurately mask. In contrast, NE best represents the 1st image and does well on the 3rd, which are simpler, but falls apart completely for the more complex 2nd and 4th images, as evidenced by the large gaps in the mask constructed by NE for the 2nd image.

NE  
![](images/ab60aedaf8d1a915ccc63662296bf14cc2a946646a459b7c0deb7b460f956ad9.jpg)  
Raw Image

![](images/eb6d489fd1cb00bcd6c274a3442393a887326bf79439e07e077b7ddc295f914e.jpg)  
Mask Image  
CLAHE 25  
CLAHE 10

![](images/f1735e6b3dbafa19acfcaa45b463ce9750b0e86429af6608be25131d6ac19f54.jpg)  
Figure 5. BCE loss, Dice loss and Dice metric for U-Net++ MBv2. For losses, lower is better, while higher is better for the Dice metric.

HE  
![](images/a2f7e56d6bb6c2f7e6a479b944ad7861f68c14982423f73b82a81652eae75d13.jpg)  
Figure 6. Example Raw Images (including whitespace) and ground-truth Mask Images as well as inferred heatmaps by MBv2 using different histogram equalization techniques. Yellow represents regions with high (∼ 1) likelihood of being part of the tooth; purple represents low likelihood. Best viewed in color.

## 4.4. Stratified Per-Tribe Results

Next, we provide results and comparison on a per-tribe level, i.e., rather than lumping all (Raw Image, Mask Image) pairs together for an ML dataset, we consider the different tribes, e.g., Alcelaphini, Neotragini, etc. [2]. When performing this kind of experiment, we further change the dataloader to stratify the data, ensuring that when we perform an 80%/20% data split, both partitions contain roughly the same proportion of data per tribe as the overall dataset.

Further, in addition to the Dice metric, we also evaluate mean Intersection over Union (mIoU).

Table 1 shows the results on SegFormer MiT-B3. Quantitatively, out of 7 different tribes, CLAHE 25 achieves the best Dice and mIoU in a majority of cases. Specifically, CLAHE c = 25 excels at the Alcelaphini, Antilopini and Hippotragini tribes, all of which tend to carry higher Dice and mIoU scores. Moreover, these results tend not to be correlated with the dataset size, but likely other underlying traits of the tribes. However, like before, these quantitative results are best paired with qualitative visual examples.

<table><tr><td>Metric</td><td colspan="4">Dice</td><td colspan="4">mIoU</td></tr><tr><td>Tribe</td><td>CLAHE 25</td><td>CLAHE 10</td><td>HE</td><td>NE</td><td>CLAHE 25</td><td>CLAHE 10</td><td>HE</td><td>NE</td></tr><tr><td>Alcelaphini (665)</td><td> $\mathbf { 0 . 9 2 7 1 } _ { 0 . 0 0 8 }$ </td><td> $0 . 9 2 3 0 _ { 0 . 0 1 3 }$ </td><td> $0 . 9 2 4 8 _ { 0 . 0 0 5 }$ </td><td> $0 . 9 2 6 9 _ { 0 . 0 1 1 }$ </td><td> $\mathbf { 0 . 8 6 3 9 } _ { 0 . 0 0 8 }$ </td><td> $0 . 8 5 7 2 _ { 0 . 0 2 2 }$ </td><td> $0 . 8 6 0 2 _ { 0 . 0 0 9 }$ </td><td> $0 . 8 6 3 7 _ { 0 . 0 1 9 }$ </td></tr><tr><td>Antilopini (180)</td><td> $\mathbf { 0 . 9 0 2 2 } _ { 0 . 0 0 9 }$ </td><td> $0 . 8 9 6 0 _ { 0 . 0 1 0 }$ </td><td> $0 . 8 9 9 4 _ { 0 . 0 0 2 }$ </td><td> $0 . 9 0 0 6 _ { 0 . 0 0 4 }$ </td><td> $\mathbf { 0 . 8 2 2 0 } _ { 0 . 0 0 9 }$ </td><td> $0 . 8 1 1 7 _ { 0 . 0 1 8 }$ </td><td> $0 . 8 1 7 1 _ { 0 . 0 0 3 }$ </td><td> $0 . 8 1 9 3 _ { 0 . 0 0 6 }$ </td></tr><tr><td>Bovini (138)</td><td> $\mathbf { 0 . 9 4 1 3 } _ { 0 . 0 0 6 }$ </td><td> $0 . 9 3 9 8 _ { 0 . 0 0 6 }$ </td><td> $0 . 9 3 7 0 _ { 0 . 0 0 8 }$ </td><td> $0 . 9 4 0 0 _ { 0 . 0 0 8 }$ </td><td> $0 . 8 8 6 4 _ { 0 . 0 0 6 }$ </td><td> $\mathbf { 0 . 8 8 9 3 } _ { 0 . 0 1 1 }$ </td><td> $0 . 8 8 1 6 _ { 0 . 0 1 3 }$ </td><td> $0 . 8 8 6 6 _ { 0 . 0 1 4 }$ </td></tr><tr><td>Hippotragini (483)</td><td> $\mathbf { 0 . 9 2 0 7 } _ { 0 . 0 0 8 }$ </td><td> $0 . 9 1 6 2 _ { 0 . 0 0 9 }$ </td><td> $0 . 9 1 3 3 _ { 0 . 0 1 0 }$ </td><td> $0 . 9 1 8 3 _ { 0 . 0 0 5 }$ </td><td> $\mathbf { 0 . 8 5 3 1 } _ { 0 . 0 0 8 }$ </td><td> $0 . 8 4 5 4 _ { 0 . 0 1 5 }$ </td><td> $0 . 8 4 0 5 _ { 0 . 0 1 6 }$ </td><td> $0 . 8 4 9 1 _ { 0 . 0 0 8 }$ </td></tr><tr><td>Neotragini (757)</td><td> $0 . 8 6 9 7 _ { 0 . 0 0 3 }$ </td><td> $0 . 8 6 4 5 _ { 0 . 0 1 3 }$ </td><td> $0 . 8 7 0 8 _ { 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 8 6 3 7 } _ { 0 . 0 1 0 }$ </td><td> $0 . 7 6 9 4 _ { 0 . 0 0 3 }$ </td><td> $0 . 7 6 1 4 _ { 0 . 0 2 0 }$ </td><td> $\mathbf { 0 . 7 7 1 2 } _ { 0 . 0 0 1 }$ </td><td> $0 . 7 6 0 1 _ { 0 . 0 1 6 }$ </td></tr><tr><td>Reduncini (489)</td><td> $0 . 9 2 7 5 _ { 0 . 0 0 2 }$ </td><td> $0 . 9 2 6 9 _ { 0 . 0 0 4 }$ </td><td> $\mathbf { 0 . 9 2 9 0 } _ { 0 . 0 0 3 }$ </td><td> $0 . 9 3 1 5 _ { 0 . 0 0 3 }$ </td><td> $0 . 8 6 4 8 _ { 0 . 0 0 2 }$ </td><td> $0 . 8 6 3 8 _ { 0 . 0 0 7 }$ </td><td> $\mathbf { 0 . 8 7 1 8 } _ { 0 . 0 0 6 }$ </td><td> $0 . 8 6 7 4 _ { 0 . 0 0 5 }$ </td></tr><tr><td>Tragelaphini (489)</td><td> $0 . 9 4 2 7 _ { 0 . 0 0 3 }$ </td><td> $0 . 9 3 9 6 _ { 0 . 0 0 4 }$ </td><td> $0 . 9 3 9 0 _ { 0 . 0 0 2 }$ </td><td> $\mathbf { 0 . 9 4 4 0 } _ { 0 . 0 0 4 }$ </td><td> $\mathbf { 0 . 8 9 4 0 } _ { 0 . 0 0 3 }$ </td><td> $0 . 8 8 6 1 _ { 0 . 0 0 8 }$ </td><td> $0 . 8 8 5 0 _ { 0 . 0 0 3 }$ </td><td> $0 . 8 9 1 6 _ { 0 . 0 0 6 }$ </td></tr></table>

Table 1. Stratified per-tribe segmentation Dice metric and mean Intersection over Union (mIoU) scores for SegFormer MiT-B3. We compare CLAHE at c ∈ 10, 25, HE and NE. Tribes annotated with number of images. Best results in bold. Results across 3 seeds.

![](images/171546a374192f3f7902dc8db57781a32853eb0dfb94b3293f8f2c741d30e2d4.jpg)  
Figure 7. Example (Raw Image, Mask Image) samples for different B.O.V.I.D. tribes as well as inference results for SegFormer MiT-B3 CLAHE c = 25 and U-Net++ MBv2 c = 10. Best viewed in color.

Next, Figure 7 provides visual examples of different (Raw Image, Mask Image) pairs for four distinct tribes, as well as the heatmap masks generated by SegFormer MiT-B3 and U-Net++ MBv2. Here we can more directly compare the Transformer and CNN: Alcelaphini teeth are relatively simple and despite the picture showing other teeth flanking the region of interest, both MiT-B3 and MBv2 capture the general area, however, MBv2 fails to properly reconstruct the shape convincingly. For Bovini, MiT-B3 almost perfectly captures the odd shape of the tooth, including the upside-down $\mathbf { \vec { \Delta } Y ^ { \prime } }$ on the bottom of it. For Neotragini, both models capture the general shape of the image, however, only MiT-B3 captures the finer details, such as the protrusions on top of the molars. The same holds true for Tragelaphini, although here, MBv2 falls short by a larger margin.

Model Size Consideration. This type of behaviour, arguably, is expected, as it has been well-documented that Transformers tend to outperform CNNs on visual tasks [8,

![](images/2e18e7f3ab5d4a7caf1fb34e07db8432ab841842cc57dd5c202bc1dfeefe1cda.jpg)  
Figure 8. Inference on out-of-distribution samples captured during a summer 2026 research trip to South Africa using SegFormer MiT-B3 CLAHE c = 25. Best viewed in color.

18]. However, we must also consider that the MiT-B3 Seg-Former is a much larger and more powerful computational model, containing 44.6M parameters to the 6.8M of MBv2, which is expressly designed for mobile deployment. When processing a 512 × 512 RGB image, SegFormer consumes 41.96 GFLOPs [14] to the 17.91 GFLOPs of MBv2, which an increase by a factor of 2.34. Given this, when coupled the dataset imperfections of B.O.V.I.D., the segmentation performance of the model is arguably still impressive.

## 4.5. Generalization to New Data

Finally, we provide some visual results about how our B.O.V.I.D. segmentation model can generalize to new, out-of-distribution (OOD) data. Specifically, we consider several new images, expressly not part of the original B.O.V.I.D. dataset, captured during a summer 2026 research trip to South Africa. Masks have not been drawn out for these images, so we are purely relying on the model, Seg-Former MiT-B3 CLAHE c = 25, to draw accurate masks of the teeth.

Figure 8 provides sample pictures of raw images and estimated masks. In contrast to prior pictures, the background of these photos is different and the lighting can be darker. Despite this, the SegFormer MiT-B3 model with CLAHE does manage to competently mask off the appropriate regions of the image with two caveats: First, the model has a bias towards drawing the mask in the center of the image; second, the 1st and 4th images have a small third lobe (these are pictures of molar teeth) on their right hand side which are not adequately captured by the heatmap. Nevertheless, the model does manage to capture the protrusions on the 5th image’s right hand side. Overall, these results further corroborate the efficacy of segmentation on this data.

## 5. Related Work

Semantic Segmentation Deep Neural Networks. Convolutional encoder-decoder networks remain a staple of pixellevel semantic segmentation, with residual networks such as ResNets [10], MobileNets [16] and EfficientNets [17] serving as a common CNN backbone for mobile and resourceconstrained inference rather than training-data robustness. More recently, Vision Transformers [8] and transformerbased segmentation heads such as SegFormer [18] demonstrate the efficacy of attention-based encoders. Our study draws directly on this architectural lineage by isolating architecture choice as a variable of interest.

Segmentation Benchmarks and Annotation Quality. The design of widely used segmentation datasets such as Cityscapes [6] and ADE20K [19] is geared towards ML training, with annotation protocols designed around pixelaccurate, model-ready masks. This is a meaningful point of departure for the B.O.V.I.D. dataset, whose masks were produced by domain experts for morphometric analysis [4] rather than for supervised learning, and which consequently exhibit unique issues uncommon in ML-geared semantic segmentation datasets. Rather than proposing a new benchmark or new correction algorithm, we contribute a comparative evaluation of how existing, well-established architectures tolerate this brand of annotation noise, a question that is largely orthogonal to, and complements, prior benchmark-driven architecture comparisons.

Preprocessing and Alignment. Contrast-limited adaptive histogram equalization is a standard technique for improving local contrast in domains with inconsistent or not ideal lighting or acquisition conditions, and we adopt it as one of the several preprocesing modes evaluated in this work. To address spatial misalignment specifically, we draw on foreground-centering and common-reference-frame registration principles, applying a lightweight, non-learned centroid-alignment step rather than a full deformable registration pipeline, consistent with the scale of our dataset and the goal of the ablation.

## 6. Conclusion

This paper presents a comparative study of convolutional and transformer-based segmentation architectures as well as preprocessing techniques on B.O.V.I.D., a dentation dataset with handcrafted masks for morphometric analysis instead of ML training. We find that lightweight preprocessing makes the dataset viable for supervised segmentation despite its resolution mismatches and raw-mask misalignment, and that quantative metrics like Dice and mIoU are largely insensitive to the choice of contrast-processing strategy even though qualitative differences in predicted masks are substantial. Finally, we find that a lightweight CNN encoder can achieve segmentation quality broadly comparable to a larger transformer-based model. Results on out-ofdistribution field data demonstrate method generalizability.

## References

[1] Gary Bradski and Adrian Kaehler. Learning OpenCV: Computer vision with the OpenCV library. ” O’Reilly Media, Inc.”, 2008. 3

[2] Juliet K. Brophy and Gregory Matthews. B.O.V.I.D., 2022. 2, 6

[3] Juliet K. Brophy, Darryl J. de Ruiter, Sheela Athreya, and Thomas J. DeWitt. Quantitative morphological analysis of bovid teeth and implications for paleoenvironmental reconstruction of plovers lake, gauteng province, south africa. Journal ofArchaeological Science, 41:376–388, 2014. 1

[4] Juliet K Brophy, Jacopo Moggi-Cecchi, Gregory J Matthews, and Shara E Bailey. Comparative morphometric analyses of the deciduous molars of homo naledi from the dinaledi chamber, south africa. American Journal of Physical Anthropology, 174(2):299–314, 2021. 1, 8

[5] Juliet K. Brophy, Gregory J. Matthews, Nicole Schnitzler, Karthik Bharath, Sebastian Kurtek, and Ofer Harel. Classification of bovidae fossils from gladysvale, south africa using elastic shape analysis. Journal of Archaeological Science, 166:105959, 2024. 1

[6] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 1, 2, 8

[7] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 4

[8] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. 2, 7, 8

[9] Rafael Gonzalez, Richard Woods, and Barry Masters. Digital image processing, third edition. Journal of Biomedical Optics, 14:029901–029901, 2009. 1, 3

[10] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceed ings ofthe IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 2, 4, 8

[11] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Identity mappings in deep residual networks. In European Conference on Computer Vision, pages 630–645. Springer, 2016. 10

[12] Pavel Iakubovskii. Segmentation models pytorch. https: //github.com/qubvel/segmentation\_models. pytorch, 2019. 1, 4, 10

[13] Sota Kato and Kazuhiro Hotta. Adaptive t-vmf dice loss: An effective expansion of dice loss for medical image segmentation. Computers in Biology and Medicine, 168:107695, 2024. 4

[14] Keith G. Mills, Di Niu, Mohammad Salameh, Weichen Qiu, Fred X. Han, Puyuan Liu, Jialin Zhang, Wei Lu, and Shangling Jui. Aio-p: Expanding neural performance predictors beyond image classification. Proceedings of the AAAI Conference on Artificial Intelligence, 37(8):9180– 9189, 2023. 8

[15] S.M. Pizer, R.E. Johnston, J.P. Ericksen, B.C. Yankaskas, and K.E. Muller. Contrast-limited adaptive histogram equalization: speed and effectiveness. In [1990] Proceedings of the First Conference on Visualization in Biomedical Computing, pages 337–345, 1990. 1, 2

[16] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings of the IEEE conference on computer vision and pattern recogni tion, pages 4510–4520, 2018. 2, 4, 8, 10

[17] Mingxing Tan and Quoc Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In International Conference on Machine Learning, pages 6105–6114. PMLR, 2019. 2, 4, 8, 10

[18] Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo. Segformer: Simple and efficient design for semantic segmentation with transformers. Advances in neural information processing systems, 34: 12077–12090, 2021. 2, 4, 8

[19] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 5122–5130, 2017. 1, 2, 8

[20] Zongwei Zhou, Md Mahfuzur Rahman Siddiquee, Nima Tajbakhsh, and Jianming Liang. Unet++: A nested u-net architecture for medical image segmentation. In International workshop on deep learning in medical image analysis, pages 3–11. Springer, 2018. 4

[21] Barbara Zitova and Jan Flusser. Image registration methods: a survey. Image and vision computing, 21(11):977–1000, 2003. 3

## Supplementary Materials

We enumerate the hyperparameters considered in our experiments as well as provide additional ablations and visualizations.

## A. Experimental Hyperparameters

We use the segmentation-models-pytorch [12] library to implement our models and conduct all of our experiments on a single Nvidia DGX Spark unit. We train our models for 70 epochs with a batch size of 8 or 6 (depends on model size, larger if possible) using the AdamW optimizer. For CNNs, we use a default learning rate of $1 e ^ { - 3 }$ and for transformers a default learning rate of $1 e ^ { - 4 }$ is utilized, and in both cases a weight decay factor of $1 e ^ { - 5 }$ is utilized alongside a gradient norm clipping parameter of 1.0. Per Sec. 3 we downsample images and masks by a factor of 4.

We maintain an 80%/20% training/testing split. When per-tribe stratification is enabled, filenames are mapped to tribes using an external CSV file, and then a split is performed using this information to preserve the proportion of data belonging to each tribe. Prior to this, data is shuffled using a random seed which we control. All stratification experiments in Sec. 4.4 are conducted using three distinct random seeds where we take the mean and standard deviation of the results, while mask images are simply pulled from the model produced by the first random seed.

## B. CNN Model Comparison

Due to space constraints, we now provide an comparative study of different CNN backbones, specifically comparing MobileNetV2 (MBv2) [16] to the older, yet larger ResNet-34 (RN34) [11] and more advanced EfficientNet-B0 (EB0) [17]. These experiments are conducted without any preprocessing, i.e., they are all ‘NE’. Figure 9 mirrors Figs. 3 and 5 by plotting the test BCE loss, Dice loss and Dice score metric across model training. The most interesting phenomenon here is that MBv2 lags behind both ResNet-34 and EfficientNet-B0 on all three metrics to a noticeable degree. Specifically, EfficientNet-B0 achieves by far the lowest BCE loss but is roughly tied with ResNet-34 on the Dice loss and score metric, while MBv2 scores noticeably worse in these cases. This finding suggests that both ResNet-34 and EfficientNet-B0 would be better suited to masking B.O.V.I.D., however, we must also consider the qualitative aspect as well.

Figure 10 provides sample images of the mask heatmaps produced by different CNN backbones on this task. As we can see, MBv2 is the only model that can consistently produce a result which vaguely resembles the ground truth Mask Image. ResNet-34 does a decent job on the first image, but then either produces nothing or just a blunch of blotches on the heatmap, while EfficientNet-B0 fails to even visualize the first mask at all.

## C. Ablation Study on $\lambda _ { D i c e }$

We now provide an ablation over the hyperparameter $\lambda _ { D i c e } ,$ primarily on MBv2. Figure 11 plots our findings for the BCE loss, Dice loss and Dice score metric for $\lambda _ { D i c e } ~ \in$ {0.3, 0.5}. Unsurprisingly, the stronger value of $\lambda _ { D i c e } =$ 0.5 provides lower Dice loss and higher dice metric, yet higher BCE loss.

Next, Figure 12 provides sample mask heatmap inferences for MBv2 without histogram equalization for different values of $\lambda _ { D i c e }$ . When we set the Dice coefficient too high, i.e., to 1.0, the model tends to overcompensate and draw excessively large masks that cover more than the region of interest and also have badly defined borders, as evidenced by the amount of green intermittent value pixels in the heatmaps. However, when $\lambda _ { D i c e } = 0 . 5$ , the model will sometimes not even draw a mask heatmap, and when it does, it will usually be inaccurate. That leaves the most optimal value we considered, $\lambda _ { D i c e } = 0 . 3$ which suggests that the BCE loss may actually play a silent, but important role in properly identifying regions of interest for this problem.

## D. Per-Tribe MBv2 Statistics

Table 2 reports stratified, per-class statistics for MBv2. Unlike SegFormer MiT-B3, the results are less competitive as CLAHE c = 25 does not obtain a clear majority for either metric, as both CLAHE c = 10 and NE achieve the best performance on some tribes, while HE is never the best. Moreover, the results are consistent from Dice to mIoU, which was not always the case for SegFormer MiT-B3.

## E. Additional OOD Examples

Finally, Figure 13 provides additional out-of-distribution mask heatmap inferences for Bovidae teeth photos captured in the summer of 2026. The segmentation model utilized is SegFormer MiT-B3 with CLAHE $c \in { 1 0 , 2 5 }$ (see caption). We see that both clip c values are able to competently draw mask heatmaps, however, $c = 2 5$ has a slight edge on certain samples, such as the first, second, and final image, where its heatmap seems more well-defined and a better match to the actual photo.

![](images/7337237c90fad32a0c0cd15dd28c25fa4ffd23c125e77bd63209a29e5fc91394.jpg)

![](images/95d9a23bdf40d7ee22b8944ac318c6167cd8cc18985f1eb165164b69617a2247.jpg)

![](images/3e9d26419ccbf3d4f1ad86c48bde0197c58245dd82145570f1565ca01f6696a1.jpg)  
Figure 9. BCE loss, Dice loss and Dice metric for U-Net++ MBv2 vs. RN34 vs. EB0. For losses, lower is better, while higher is better for the Dice metric.  
Raw Image  
Mask Image

MobileNetV2  
ResNet-34  
EfficientNet-B0  
![](images/688001a05f85a0178908848eb59be2bd7564f7d24257b6348987197035cf2e23.jpg)  
Figure 10. Example Raw Images (including whitespace) and ground-truth Mask Images as well as inferred heatmaps by MBv2, ResNet-34 and EfficientNet-B0 without any histogram equalization. Yellow represents regions with high (∼ 1) likelihood of being part of the tooth; purple represents low likelihood. Best viewed in color.

![](images/75c333b2bce00332ec93941229370dc991cafec5678a8e579f0e18edb373468c.jpg)

![](images/0d0d1ee8a24c773bdf48b3b33d5754ab4f71a93021cc59ca490406cf6af466eb.jpg)

![](images/fc8f2941a44bf22504a2db5aa48676de63478edd2f568cbcec1012212a476f74.jpg)  
Figure 11. BCE loss, Dice loss and Dice metric for U-Net++ MBv2 while varying $\lambda _ { D i c e } .$ For losses, lower is better, while higher is better for the Dice metric. Best viewed in color.

Raw Image

Mask Image

$$
\lambda _ { \mathrm { Ḋ i c e } } = { \bf 1 . 0 }
$$

$$
\lambda _ { \mathrm { Ḋ i c e Ḍ } } = \mathbf { 0 . 5 }
$$

$$
\lambda _ { \mathrm { Ḋ i c e Ḍ } } = \mathbf { 0 . 3 }
$$

![](images/b9cb2da7e7550cdea2635b327cac7ea39008f7335a3ecb3e40498c8e6258483d.jpg)  
Figure 12. Example Raw Images (including whitespace) and ground-truth Mask Images as well as inferred heatmaps by MBv2 withou any histogram equalization while varying $\lambda _ { Ḋ } \mathclose | Ḍ _ { Ḋ } i c e Ḍ$ . Yellow represents regions with high (∼ 1) likelihood of being part of the tooth; purple represents low likelihood. Best viewed in color.

<table><tr><td>Metric</td><td colspan="4">Dice</td><td colspan="4">mIoU</td></tr><tr><td>Tribe</td><td>CLAHE 25</td><td>CLAHE 10</td><td>HE</td><td>NE</td><td>CLAHE 25</td><td>CLAHE 10</td><td>HE</td><td>NE</td></tr><tr><td>Alcelaphini (665)</td><td> $\mathbf { 0 . 8 7 1 0 } _ { 0 . 0 1 9 }$ </td><td> $0 . 8 7 0 8 _ { 0 . 0 1 5 }$ </td><td> $0 . 8 6 5 8 _ { 0 . 0 1 7 }$ </td><td> $0 . 8 6 5 9 _ { 0 . 0 1 5 }$ </td><td> $\mathbf { 0 . 7 7 1 8 } _ { 0 . 0 3 1 }$ </td><td> $0 . 7 7 1 4 _ { 0 . 0 2 3 }$ </td><td> $0 . 7 6 3 7 _ { 0 . 0 2 7 }$ </td><td> $0 . 7 6 3 7 _ { 0 . 0 2 4 }$ </td></tr><tr><td>Antilopini (180)</td><td> $0 . 8 5 7 1 _ { 0 . 0 1 5 }$ </td><td> $\mathbf { 0 . 8 5 9 8 } _ { 0 . 0 1 2 }$ </td><td> $0 . 8 4 2 8 _ { 0 . 0 1 0 }$ </td><td> $0 . 8 5 7 0 _ { 0 . 0 0 3 }$ </td><td> $0 . 7 5 0 0 _ { 0 . 0 2 3 }$ </td><td> $\mathbf { 0 . 7 5 4 2 } _ { 0 . 0 1 9 }$ </td><td> $0 . 7 2 8 4 _ { 0 . 0 1 5 }$ </td><td> $0 . 7 4 9 8 _ { 0 . 0 0 5 }$ </td></tr><tr><td>Bovini (138)</td><td> $0 . 4 0 1 8 _ { 0 . 0 0 8 }$ </td><td> $0 . 9 0 3 0 _ { 0 . 0 0 6 }$ </td><td> $0 . 9 0 0 0 _ { 0 . 0 0 9 }$ </td><td> $\mathbf { 0 . 9 0 3 2 } _ { 0 . 0 0 8 }$ </td><td> $0 . 8 2 1 3 _ { 0 . 0 1 3 }$ </td><td> $0 . 8 2 3 2 _ { 0 . 0 1 0 }$ </td><td> $0 . 8 1 8 2 _ { 0 . 0 1 5 }$ </td><td> $\mathbf { 0 . 8 2 3 5 } _ { 0 . 0 1 3 }$ </td></tr><tr><td>Hippotragini (483)</td><td> $\mathbf { 0 . 8 6 5 7 } _ { 0 . 0 1 5 }$ </td><td> $0 . 8 6 5 1 _ { 0 . 0 1 3 }$ </td><td> $0 . 8 5 7 3 _ { 0 . 0 1 7 }$ </td><td> $0 . 8 6 3 0 _ { 0 . 0 1 0 }$ </td><td> $\mathbf { 0 . 7 6 3 4 } _ { 0 . 0 2 3 }$ </td><td> $0 . 7 6 2 5 _ { 0 . 0 2 1 }$ </td><td> $0 . 7 5 0 5 _ { 0 . 0 2 6 }$ </td><td> $0 . 7 5 9 1 _ { 0 . 0 1 6 }$ </td></tr><tr><td>Neotragini (757)</td><td> $0 . 8 1 3 1 _ { 0 . 0 1 3 }$ </td><td> $\mathbf { 0 . 8 1 6 5 } _ { 0 . 0 0 5 }$ </td><td> $0 . 8 0 2 0 _ { 0 . 0 0 6 }$ </td><td> $0 . 8 0 7 5 _ { 0 . 0 1 3 }$ </td><td> $0 . 6 8 5 2 _ { 0 . 0 1 8 }$ </td><td> $\mathbf { 0 . 6 8 9 9 } _ { 0 . 0 0 7 }$ </td><td> $0 . 6 6 9 6 _ { 0 . 0 0 8 }$ </td><td> $0 . 6 7 7 2 _ { 0 . 0 1 8 }$ </td></tr><tr><td>Reduncini (489)</td><td> $\mathbf { 0 . 8 8 6 9 } _ { 0 . 0 0 8 }$ </td><td> $0 . 8 8 6 3 _ { 0 . 0 0 9 }$ </td><td> $0 . 8 8 5 7 _ { 0 . 0 0 7 }$ </td><td> $0 . 8 8 4 8 _ { 0 . 0 0 5 }$ </td><td> $\mathbf { 0 . 7 9 6 9 } _ { 0 . 0 1 4 }$ </td><td> $0 . 7 9 5 9 _ { 0 . 0 1 5 }$ </td><td> $0 . 7 9 4 8 _ { 0 . 0 1 1 }$ </td><td> $0 . 7 9 3 6 _ { 0 . 0 1 0 }$ </td></tr><tr><td>Tragelaphini (489)</td><td> $0 . 8 9 2 9 _ { 0 . 0 0 3 }$ </td><td> $0 . 8 4 6 2 _ { 0 . 0 0 5 }$ </td><td> $0 . 8 9 1 6 _ { 0 . 0 0 4 }$ </td><td> $\mathbf { 0 . 8 9 7 0 } _ { 0 . 0 0 7 }$ </td><td> $0 . 8 0 6 6 _ { 0 . 0 0 4 }$ </td><td> $0 . 8 1 2 0 _ { 0 . 0 0 8 }$ </td><td> $0 . 8 0 4 5 _ { 0 . 0 0 7 }$ </td><td> $\mathbf { 0 . 8 1 3 4 } _ { 0 . 0 1 1 }$ </td></tr></table>

Table 2. Stratified per-tribe segmentation Dice metric and mean Intersection over Union (mIoU) scores for MBv2. We compare CLAHE $\mathfrak { t } c \in { 1 0 , 2 5 }$ , HE and NE. Tribes annotated with number of images. Best results in bold. Results averaged across 3 random seeds.

![](images/9d1860d01c5214d3c72d0369b5432a555379240cabaed4b75f1f21211af2248c.jpg)  
Figure 13. Additional inference on out-of-distribution samples captured during a summer 2026 research trip to South Africa using Seg Former MiT-B3 CLAHE c = 25 (middle row) and c = 10 (bottom row). Best viewed in color.