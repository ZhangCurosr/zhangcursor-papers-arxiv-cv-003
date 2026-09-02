# Scale-based Approach for Active Wildfire Segmentation on Satellite Imagery

Matheus F. Kovaleski<sup>∗</sup>, Cristiano Premebida<sup>∗</sup>, Joao Ruivo Paulo˜ <sup>∗</sup>

∗ Institute of Systems and Robotics, Department of Electrical and Computer Engineering, University of Coimbra, Portugal Email: {matheus.kovaleski, jpaulo, cpremebida}@isr.uc.pt

Abstract—Active wildfire mapping from satellite imagery is challenging due to the sparse and highly imbalanced nature of fire pixels, especially in early-stage or low-density fire observations. This work investigates the use of multispectral Landsat-8 imagery for active-fire segmentation under multi-scale wildfire size conditions. We propose a data-driven protocol to characterize fireregion size distributions through connected-component analysis and an interquartile range criterion, enabling the evaluation of model robustness across different local fire-region densities. Three segmentation architectures, U-Net, DeepLabV3+, and SegFormer, are evaluated under different SWIR-based spectral configurations. Results show that U-Net achieves the strongest robustness across the evaluated conditions, SegFormer provides competitive performance, and DeepLabV3+ tends to produce conservative predictions with reduced recall. Across architectures, SWIR2 consistently achieves the strongest or near-best results, highlighting its importance for active-fire segmentation in Landsat-8 imagery. These findings suggest that both spectral band selection and architectural design are critical for robust satellite-based active wildfire mapping trained on low active firepixel density images.

Keywords—Wildfire, Land8Fire, Deep Learning, Remote Sensing.

## I. INTRODUCTION

Wildfires are among the most significant natural disturbances affecting terrestrial ecosystems, with increasing frequency and intensity over recent decades due to climate change [1], [2]. Large wildfire events produce severe environmental, economic, and social impacts, including biodiversity loss, carbon emissions, and risks to human life [3]. In addition to detection and monitoring, recent research has also explored computational approaches for wildfire spread prediction and simulation under complex environmental conditions [4], [5].

Satellite remote sensing has become an essential tool for large-scale wildfire monitoring, enabling near-real-time observation of extensive and remote areas using multispectral sensors such as Landsat-8, Sentinel-2, MODIS, and VIIRS [6]. In recent years, deep learning approaches have significantly improved wildfire segmentation performance, particularly through convolutional neural networks (CNNs) such as U-Net and DeepLabV3+ [7], [8] and more recent transformerbased architectures [9].

Despite these advances, wildfire segmentation remains challenging due to severe class imbalance, spectral ambiguity, and large variability in fire morphology and spatial extent [10]. In particular, wildfire events evolve from small ignition regions to large and spatially complex fire fronts, introducing significant changes in spatial patterns and contextual structure.

Most existing studies evaluate wildfire segmentation models using conventional random train-test splits [10]–[12]. Although effective for measuring average performance, such protocols may not properly assess model robustness under distribution shifts caused by fire-scale variability.

In this work, we investigate wildfire segmentation under a scale-based distribution shift scenario, where models trained on small fire regions are evaluated on substantially larger active fire regions. To this end, we implement a datadriven train/test splitting strategy based on the interquartile range (IQR) of connected fire-region sizes extracted from the Land8Fire dataset. Furthermore, we analyze the influence of different SWIR-based spectral configurations on segmentation performance. We compare three segmentation architectures: U-Net, DeepLabV3+, and SegFormer.

The main contributions of this work are summarized as follows:

• A scale-based dataset splitting strategy for evaluating wildfire segmentation robustness under fire-size distribution shifts;

• An analysis of the impact of SWIR spectral configura tions on wildfire segmentation performance;

• A comparative evaluation of CNN-based and transformerbased segmentation architectures under shifted fire-scale distributions.

## II. RELATED WORK

Wildfire detection using remote sensing has been widely studied, benefiting from the increasing availability of multispectral satellite imagery. Early approaches relied on threshold-based methods and spectral indices derived from thermal and shortwave infrared (SWIR) bands, exploiting the strong radiative response of active fires in these wavelengths [6], [13]. While effective in controlled scenarios, these methods are often sensitive to atmospheric effects, noise, and spectral confusion with non-fire elements.

With the emergence of deep learning, convolutional neural networks (CNNs) have become the dominant approach for wildfire detection and segmentation. These models can learn complex spatial and spectral patterns directly from data, significantly outperforming traditional methods [11], [14], [15]. Architectures such as U-Net leverage encoder–decoder structures with skip connections to preserve spatial details, while models like DeepLabV3+ incorporate multi-scale contextual information through atrous convolutions and spatial pyramid pooling [7], [8].

More recent works have further explored deep learning architectures for active fire detection, including specialized models designed for Landsat imagery [12]. In parallel, transformerbased architectures have been introduced such as SegFormer and FireFormer highlight the potential of attention-based representations for wildfire segmentation tasks [9], [16].

The role of spectral information has also been extensively investigated. Multispectral inputs, particularly those including SWIR bands, have been shown to significantly improve fire detection performance due to their sensitivity to hightemperature emissions and reduced interference from smoke and atmospheric effects [17], [18]. Additionally, selecting appropriate band combinations can be more effective than using all available spectral channels [19].

Another key challenge in wildfire segmentation is the severe imbalance between fire and background pixels, combined with large variability in fire morphology, spatial distribution, and scale [20].

Beyond active-fire detection and segmentation, recent studies have also investigated wildfire spread prediction and calibration using optimization and metaheuristic approaches. Pereira et al. [4] explored the calibration of wildfire spread models using evolutionary and stochastic optimization algorithms, while later extending this framework to twodimensional wildfire propagation modeling under real wildfire scenarios [5]. These works highlight the inherent variability and uncertainty of wildfire behavior, reinforcing the importance of robust modeling and evaluation strategies.

Despite these advances, the problem of generalization across different fire scales remains largely unexplored. In real-world scenarios, wildfire events evolve dynamically over time, leading to substantial changes in spatial structure and scale [21]. However, most existing studies rely on random dataset splits, which do not explicitly evaluate model robustness under distribution shifts.

In this work, we address this gap by introducing a scalebased evaluation protocol, rather than a new segmentation architecture, enabling a controlled analysis of model generalization from small-scale to large-scale fire regions.

## III. METHODOLOGY

This section describes the proposed scale-based wildfire segmentation framework, including the preprocessing pipeline, dataset partitioning strategy, spectral configurations, evaluated segmentation models, and experimental setup.

Figure 1 summarizes the overall workflow adopted in this study. The pipeline consists of three main stages: preprocessing and fire-region extraction, statistical fire-size analysis, and scale-based dataset partitioning.

## A. Dataset

The dataset used in this work is the Land8Fire [10], a large-scale, high-resolution multispectral benchmark designed for semantic segmentation of active wildfires using satellite imagery. It was developed to address limitations of existing wildfire datasets, such as limited spectral diversity, noisy annotations, and severe class imbalance.

Land8Fire is derived from multispectral imagery acquired by the Landsat-8 satellite and extends the previously released ActiveFire dataset [6] by incorporating manually validated fire masks and improved sampling strategies. The dataset includes wildfire events from multiple regions worldwide, including the Amazon, Africa, Australia, and the United States, providing significant diversity in vegetation types and fire behavior.

Land8Fire provides two versions of the dataset: a preprocessed version and a raw version containing the original selected Landsat-8 scenes. In this work, we use the raw version of the dataset, and all preprocessing steps, including patch extraction and scale-based splitting, are performed using the pipeline described in Section III-B. This allows full control over the data distribution and enables the proposed scale-based analysis.

## B. Data Preprocessing and Scale-Based Splitting Pipeline

Figure 1 illustrates the proposed pipeline for evaluating wildfire segmentation under a scale-based distribution shift scenario. The pipeline consists of three main stages: patch extraction and connected-component analysis, statistical firesize characterization, and scale-based dataset partitioning.

All Landsat-8 multispectral images were first aligned with their corresponding ground-truth masks through georeferencing to ensure spatial consistency between inputs and labels.

Next, each image was divided into non-overlapping patches of size 256×256 pixels. A connected-component analysis was then applied to the binary fire masks to identify contiguous fire regions within each patch. The size of each connected component was computed as the number of connected fire pixels.

To characterize fire-size variability, the Interquartile Range (IQR) was computed over all detected connected components. The resulting distribution was highly skewed, with most fire regions concentrated in the lower range and a limited number of large outliers. The IQR-based criterion yielded a 16-pixel cutoff, which was used to separate typical small fire components from atypically large connected fire regions. Since each Landsat-8 pixel corresponds to an area of 30 × 30 meters, this threshold represents approximately 1.44 hectares.

In this work, the terms small-scale and large-scale refer to connected fire-region sizes observed within individual image patches rather than to operational wildfire size classes. Figure 2 presents representative examples of patches categorized as small-fire and large-fire samples according to the proposed criterion.

![](images/7208b448171fce9e485a084c4aba4e1c8b88eb1b1f4ba99dfd1e51e1570c67b0.jpg)  
Fig. 1: Overview of the proposed scale-based wildfire segmentation pipeline. The raw Land8Fire scenes are first preprocessed through image-mask alignment, patch extraction, and connected-component analysis. Fire-region sizes are then analyzed using the IQR criterion to define small- and large-fire partitions. The resulting data splits are evaluated using different SWIR-based spectral configurations and segmentation models.

Dataset partitioning was performed based on the size of connected fire regions identified within each patch:

• Patches with largest connected component size $\leq ~ 1 6$ pixels were assigned to the training set;

• Patches with largest connected component size > 16 pixels were assigned to the test set.

This strategy introduces a controlled fire-scale distribution shift, enabling the evaluation of model generalization from small to large fire structures.

Following the Land8Fire curation protocol, patches without fire pixels were excluded from the experiments. The resulting dataset contains approximately 12,000 training patches and 400 testing patches.

## C. Spectral Band Configurations

To evaluate the impact of spectral information on fire detection and generalization under scale-based distribution shifts, three different band combinations were considered. These configurations were designed to analyze how the inclusion or exclusion of shortwave infrared (SWIR) channels affects model performance.

• Dual-SWIR: SWIR1, SWIR2, Red, and NIR. This configuration includes both SWIR bands and it is used to evaluate the complementary contribution of SWIR1 and SWIR2.

• SWIR2-only: SWIR2, Red, and NIR. This configuration focuses on the longer-wavelength SWIR2 band.

• SWIR1-only: SWIR1, Red, and NIR. This configuration focuses on the shorter-wavelength SWIR1 band.

SWIR bands are known to be highly responsive to active fire due to their sensitivity to high-temperature emissions and reduced interference from smoke and atmospheric effects. In particular, SWIR2 is more sensitive to higher temperature ranges, while SWIR1 captures complementary thermal and reflectance properties.

By comparing these configurations, we aim to assess: (i) the individual contribution of each SWIR band, (ii) whether combining both SWIR bands provides complementary information, and (iii) how spectral richness influences the ability of models to generalize from small-scale to large-scale fire events.

![](images/e332c90b3dab97990f2f450a21882c619833870e5aec1b11f349613f468c2cca.jpg)  
Fig. 2: Examples of fire masks from the proposed scale-based partitioning. Rows labeled as small-fire contain patches whose connected fire regions remain below the IQR-derived threshold of 16 pixels, whereas rows labeled as large-fire contain patches with at least one connected fire region exceeding this threshold. For each example, the original 256 × 256 mask and a cropped view around the fire region are shown.

All configurations maintain the Red and Near-Infrared (NIR) bands, which provide contextual information related to vegetation and background conditions. Thermal bands B10 and B11 were not included because their native spatial resolution is 100 m, requiring resampling to 30 m and potentially introducing spatial uncertainty in patch-level segmentation. This study therefore focuses on 30 m OLI bands to isolate the effect of SWIR information under consistent spatial resolution.

## D. Segmentation Models

Three semantic segmentation architectures were evaluated in this work: U-Net, DeepLabV3+, and SegFormer. These models were selected due to their complementary characteristics regarding spatial-detail preservation, contextual aggregation, and global representation learning.

1) U-Net: U-Net [7] is an encoder-decoder convolutional architecture with skip connections that preserve fine-grained spatial information during feature reconstruction. Due to its strong localization capability, U-Net is well suited for detecting sparse and small fire regions.

2) DeepLabV3+: DeepLabV3+ [8] incorporates atrous convolutions and Atrous Spatial Pyramid Pooling (ASPP) to capture multi-scale contextual information. Its large receptive fields enable improved representation of spatially complex fire structures.

3) SegFormer: SegFormer [9] is a transformer-based semantic segmentation architecture that combines hierarchical self-attention encoders with a lightweight multilayer perceptron decoder. Its design enables efficient multi-scale representation learning while preserving spatial structure.

By comparing these three architectures, we analyze how different design trade-offs—local detail preservation (U-Net), contextual multi-scale aggregation (DeepLabV3+), and global attention-based representation learning (SegFormer)—affect generalization from small to large fire events.

a) Multi-channel Input Adaptation: Since the original DeepLabV3+ and SegFormer backbones are designed for three-channel (RGB) inputs, they were adapted to support four-channel multispectral data. Specifically, the first convolutional projection layer of each backbone was modified to accept four input channels.

For DeepLabV3+, the weights corresponding to the original three channels were preserved, while the additional channel was initialized using the mean of the pretrained RGB weights. This initialization strategy allows the model to leverage pretrained features while incorporating additional spectral information.

A similar strategy was adopted for SegFormer, extending its input projection layer to handle four-channel multispectral inputs while preserving pretrained initialization as much as possible.

These modifications enable the use of multispectral inputs without significantly altering the original architectures or training dynamics.

## E. Evaluation Metrics

Model performance was evaluated using five standard segmentation metrics: Precision (P), Recall (R), F1-score (F<sub>1</sub>), Intersection over Union (IoU), and Matthews Correlation Coefficient (MCC).

## IV. EXPERIMENTAL RESULTS

This section presents the experimental evaluation of the proposed scale-based wildfire segmentation framework. The goal of these experiments is to analyze how different data distributions influence model performance, with a particular focus on generalization across fire scales. We compare a conventional random dataset split, which represents a standard training scenario, with the proposed scale-based splitting strategy that introduces a controlled distribution shift between training and testing data. This comparison allows us to assess whether traditional evaluation protocols provide reliable estimates of model performance in realistic wildfire scenarios. Additionally, we evaluate the behavior of models trained exclusively on smallscale fire regions under different testing conditions, including both similar and significantly different data distributions. This enables a detailed analysis of how scale variability impacts segmentation performance. To evaluate the impact of scalebased distribution shifts on wildfire segmentation, we designed a set of experiments comparing a conventional random split with the proposed scale-based splitting strategy.

## A. Training Setup

All models were trained for binary semantic segmentation, where pixels were classified as either fire or background. To ensure a fair comparison, the same learning rate $( 1 \times 1 0 ^ { - 4 } )$ , batch size (8), weight decay $( 5 \times 1 0 ^ { - 4 } )$ , and total number of training iterations (5,000) were adopted across all architectures whenever possible. All reported results correspond to the mean and standard deviation over 5 independent runs with different random seeds, using the preprocessing pipeline described in Section III-B. U-Net was trained using weighted cross-entropy loss to mitigate pixel-level class imbalance. DeepLabV3+ employed the deeplabv3plus mobilenet backbone with output stride 16 and polynomial learning-rate decay. SegFormer was trained using the same dataset partitions while preserving its default architecture-specific hyperparameters. For DeepLabV3+ and SegFormer, the input projection layers were adapted to support four-channel multispectral inputs by extending the original RGB initialization with an additional channel initialized from the mean pretrained weights. Experiments were conducted on GeForce RTX 3090 24GB.

## B. Baseline Setup

As a reference, a baseline dataset split was created using a standard random partitioning strategy, where the dataset was divided into 70% for training, 15% for validation, and 15% for testing. This split follows common practices in deep learning and does not consider fire size or spatial characteristics. The same preprocessing pipeline described in Section III-B was applied, excluding the scale-based filtering step. This baseline setup represents a conventional training scenario where models are exposed to a mixture of fire sizes during training.

## C. Scale-Based Setup

In contrast, the proposed setup follows the scale-based splitting strategy described in Section III-B. In this case, the training set contains only small-scale fire regions, while the test set consists of patches of large fire regions. This setup introduces a controlled distribution shift, enabling the evaluation of model generalization from small-scale to largescale fire events.

## D. Model Training

All three models were trained under the two setups described above (baseline and scale-based). The same spectral band configurations and preprocessing pipeline were used across architectures to ensure a consistent comparison.

## E. Evaluation Protocol

To further analyze generalization behavior, the model trained on small-scale fire data (referred to as the small-fire model) was evaluated under three different testing conditions:

• Baseline: evaluates how the model performs on a standard data distribution containing a mixture of fire sizes.

• Small-fire: evaluates performance on data similar to the training distribution.

• Large-fire: evaluates the model under a strong distribution shift, where fire regions are significantly larger than those seen during training.

This evaluation protocol allows us to analyze how well models generalize across different fire scales and to quantify the impact of scale-based distribution shifts on segmentation performance.

## F. Results

1) U-Net Results: Table I reports the performance of U-Net on the baseline, small-fire, and large-fire test sets, respectively. U-Net achieves the strongest overall performance among the evaluated models. On the baseline and largefire test sets, the Dual-SWIR and SWIR2-only configurations obtain very similar results, with F1-scores above 92% and IoU values above 86%. This indicates that U-Net is highly effective at detecting fire regions when trained on smallscale samples and evaluated on larger fire structures. On the small-fire test set, performance decreases, mainly in precision and IoU, while recall remains consistently high across all spectral configurations. This suggests that small and sparse fire regions are more challenging due to their limited spatial extent and higher sensitivity to false positives. Nevertheless, the model maintains high recall, indicating that it rarely misses fire pixels. Across all U-Net experiments, Dual-SWIR and SWIR2-only provide very similar results, while the SWIR1- only configuration performs consistently worse. This suggests that SWIR2 band contains more discriminative information for active fire segmentation than SWIR1 band.

2) DeepLabV3+ Results: Table II presents the results obtained with DeepLabV3+. Compared with U-Net, DeepLabV3+ shows substantially lower performance, especially in recall. Although the model maintains relatively high precision, recall drops significantly across all test sets, indicating that DeepLabV3+ tends to produce conservative predictions. In practice, this means that when DeepLabV3+ predicts fire, it is often correct, but it misses a large portion of actual fire pixels. This behavior is especially evident in the SWIR1-only configuration, where recall and IoU are considerably lower than in the other spectral configurations. The best DeepLabV3+ results are obtained using Dual-SWIR or SWIR2-only, reinforcing the importance of SWIR2 band for active fire detection. Overall, these results suggest that DeepLabV3+ is more sensitive to the scale-based distribution shift than U-Net. Its stronger reliance on contextual feature aggregation may reduce its ability to detect sparse or spatially fragmented fire patterns when trained on small-fire regions.

3) SegFormer Results: Table III shows the performance of SegFormer. SegFormer achieves intermediate performance between U-Net and DeepLabV3+. On the baseline and largefire test sets, SegFormer performs strongly with Dual-SWIR and SWIR2-only, achieving F1-scores above 85% and IoU values above 74%. However, performance decreases substantially when SWIR1-only is used, particularly in recall and IoU. On the small-fire test set, SegFormer shows a clear reduction in IoU and F1-score compared with the baseline and large-fire sets. This confirms that small fire regions are more difficult to segment, even for transformer-based architectures. However, unlike DeepLabV3+, SegFormer maintains a more balanced precision-recall trade-off, suggesting better robustness to small and sparse fire patterns. The results indicate that SegFormer benefits from the inclusion of both SWIR bands, but SWIR2- only remains highly competitive. This trend is consistent with the U-Net and DeepLabV3+ results.

4) Cross-Model Analysis: The results reveal three main findings. First, U-Net provides the most robust performance across all evaluated conditions, especially in terms of recall, IoU, F1-score, and MCC. This behavior can be attributed to its encoder-decoder structure with skip connections, which helps preserve fine spatial details and supports dense pixellevel localization. Such properties are particularly relevant for active-fire masks, where target regions are sparse, fragmented, and highly imbalanced.

Second, DeepLabV3+ exhibits the weakest generalization behavior, mainly due to low recall. Its predictions are more conservative, leading to fewer false positives but more missed fire pixels. This is undesirable for wildfire monitoring, where missed detections may delay response actions. Third, the spectral configuration strongly affects performance across all architectures. SWIR2-only consistently outperforms SWIR1- only, while the Dual-SWIR configuration generally provides the most stable results. These findings support the importance of SWIR information, particularly SWIR2, for active wildfire mapping. Overall, the experiments show that both architecture choice and spectral band selection are critical for robust wildfire segmentation under scale-based distribution shifts.

TABLE I: U-Net performance across baseline, small-fire, and large-fire test sets (mean ± std)
<table><tr><td>Set</td><td>Input</td><td>Prec.</td><td>Rec.</td><td>IoU</td><td>F1</td><td>MCC</td></tr><tr><td rowspan="3">Baseline</td><td>Dual-SWIR</td><td>90.26±.90</td><td>99.05±.64</td><td>89.49±.81</td><td>94.45±.45</td><td>.944±.004</td></tr><tr><td>SWIR2-only</td><td>89.04±.92</td><td>99.34±.15</td><td>88.51±.80</td><td>93.91±.45</td><td>.939±.004</td></tr><tr><td>SWIR1-only</td><td>80.72±1.87</td><td>95.46±1.29</td><td>77.70±.90</td><td>87.45±.57</td><td>.875±.005</td></tr><tr><td rowspan="3"></td><td>Dual-SWIR</td><td>88.38±1.24</td><td>98.90±.79</td><td>87.51±1.21</td><td>93.34±.69</td><td>.934±.007</td></tr><tr><td>Small-Fire SWIR2-only</td><td>87.00±1.23</td><td>99.23±.21</td><td>86.41±1.08</td><td>92.71±.62</td><td>.928±.006</td></tr><tr><td>SWIR1-only</td><td>78.11±1.84</td><td>98.21±.56</td><td>77.01±1.89</td><td>87.00±1.20</td><td>.874±.011</td></tr><tr><td rowspan="3"></td><td>Dual-SWIR</td><td>88.52±.91</td><td>98.96±.71</td><td>87.70±.98</td><td>93.44±.55</td><td>.935±.005</td></tr><tr><td>Large-Fire SWIR2-only</td><td>87.13±1.25</td><td>99.28±.17</td><td>86.58±1.12</td><td>92.81±.64</td><td>.929±.006</td></tr><tr><td>SWIR1-only</td><td>78.28±1.98</td><td>98.33±.41</td><td>77.25±2.00</td><td>87.16±1.26.875±.012</td><td></td></tr></table>

TABLE II: DeepLabV3+ performance across baseline, smallfire, and large-fire test sets (mean ± std)
<table><tr><td>Set</td><td>Input</td><td>Prec.</td><td>Rec.</td><td>IoU</td><td>F1</td><td>MCC</td></tr><tr><td rowspan="3">Baseline</td><td>Dual-SWIR</td><td>75.72±1.63</td><td> $\overline { { 5 3 . 8 1 \pm 6 . 5 6 } }$ </td><td></td><td>45.89±5.21 62.76±5.02 .631±.045</td><td></td></tr><tr><td>SWIR2-only</td><td> $7 6 . 9 8 { \scriptstyle \pm 1 . 5 7 }$ </td><td> $5 7 . 3 9 { \scriptstyle \pm 2 . 8 5 }$ </td><td> $4 8 . 9 9 { \scriptstyle \pm 2 . 4 9 }$ </td><td></td><td>65.73±2.24.659±.021</td></tr><tr><td>SWIR1-only</td><td> $7 9 . 3 5 { \scriptstyle \pm 2 . 5 4 }$ </td><td> $3 0 . 5 9 { \scriptstyle \pm 2 . 5 7 }$ </td><td> $2 8 . 3 2 { \scriptstyle \pm 2 . 2 6 }$ </td><td> $4 4 . 1 0 { \scriptstyle \pm 2 . 7 9 }$ </td><td>.486±.023</td></tr><tr><td rowspan="3"></td><td>Dual-SWIR</td><td> $7 3 . 2 2 \pm 1 . 6 8$ </td><td> $\overline { { 5 9 . 1 6 { \pm } 5 . 1 0 } }$ </td><td> $\overline { { 4 8 . 6 5 \pm 4 . 0 2 } }$ </td><td>65.38±3.75</td><td> $\overline { { . 6 5 3 \pm . 0 3 5 } }$ </td></tr><tr><td>Small-Fire SWIR2-only</td><td> $7 4 . 5 0 { \scriptstyle \pm 1 . 8 7 }$ </td><td> $5 9 . 8 5 { \scriptstyle \pm 3 . 7 9 }$ </td><td> $4 9 . 6 5 { \scriptstyle \pm 2 . 7 7 }$ </td><td> $6 6 . 3 2 { \scriptstyle \pm 2 . 4 2 }$ </td><td> $. 6 6 3 { \pm } . 0 2 3$ </td></tr><tr><td>SWIR1-only</td><td> $7 7 . 5 6 { \scriptstyle \pm 3 . 7 8 }$ </td><td> $3 3 . 8 6 \pm 3 . 9 6$ </td><td> $3 0 . 7 6 \pm 3 .$  18</td><td> $4 6 . 9 7 { \scriptstyle \pm 3 . 7 8 }$ </td><td> $. 5 0 7 { \scriptstyle \pm . 0 2 9 }$ </td></tr><tr><td rowspan="3"></td><td>Dual-SWIR</td><td> $7 3 . 4 9 \pm 1 . 3 2$ </td><td> $\overline { { 5 8 . 8 4 \pm 5 . 0 0 } }$ </td><td> $\overline { { 4 8 . 5 3 \pm 3 . 7 5 } }$ </td><td>65.27±3.50</td><td> $\overline { { . 6 5 3 \pm . 0 3 2 } }$ </td></tr><tr><td>Large-Fire SWIR2-only</td><td> $7 4 . 2 6 { \pm } 1 . 4 4$ </td><td> $5 9 . 9 0 { \scriptstyle \pm 3 . 1 0 }$ </td><td> $4 9 . 5 8 { \scriptstyle \pm 2 . 2 5 }$ </td><td>66.27±1.98</td><td> $. 6 6 2 { \pm } . 0 1 9$ </td></tr><tr><td>SWIR1-only</td><td> $\lvert 7 7 . 7 8 \pm 2 . 5 8$ </td><td> $3 3 . 5 8 { \scriptstyle \pm 2 . 8 6 }$ </td><td> $3 0 . 6 0 { \scriptstyle \pm 2 . 2 7 }$ </td><td> $4 6 . 8 3 { \scriptstyle \pm 2 . 7 0 }$ </td><td> $. 5 0 6 { \pm } . 0 2 1 $ </td></tr></table>

TABLE III: SegFormer performance across baseline, smallfire, and large-fire test sets (mean ± std)
<table><tr><td>Set</td><td>Input</td><td> $\overline { { \mathrm { P r e c . } } }$ </td><td> $\overline { { \mathrm { R e c . } } }$ </td><td>IoU</td><td>F1</td><td>MCC</td></tr><tr><td rowspan="3">Baseline</td><td>Dual-SWIR</td><td> $\overline { { 8 7 . 6 5 \pm . 9 2 } }$ </td><td> $8 7 . 9 8 { \pm } 1 . 0 9$ </td><td> $7 8 . 2 7 { \scriptstyle \pm . 3 2 }$ </td><td> $8 7 . 8 1 \pm . 2 0 $ </td><td>.876±.002</td></tr><tr><td>SWIR2-only</td><td> $8 7 . 6 5 { \pm } . 3 9$ </td><td> $8 7 . 8 8 \pm . 3 7$ </td><td> $7 8 . 1 9 { \pm } . 3 2$ </td><td></td><td>87.76±.20.875±.002</td></tr><tr><td>SWIR1-only</td><td> $8 7 . 0 4 \pm . 4 7 $ </td><td> $7 1 . 4 5 { \pm } . 7 4$ </td><td></td><td>64.58±.6078.48±.44.785±.004</td><td></td></tr><tr><td rowspan="3"></td><td>Dual-SWIR</td><td> $\overline { { 8 5 . 3 3 \pm 1 . 1 2 } }$ </td><td> $8 5 . 5 3 { \scriptstyle \pm . 9 8 }$ </td><td> $\overline { { 7 4 . 5 5 \pm . 3 5 } }$ </td><td></td><td>85.42±.23.852±.002</td></tr><tr><td>Small-Fire SWIR2-only</td><td> $8 5 . 7 0 { \scriptstyle \pm . 4 6 }$ </td><td> $8 5 . 3 1 { \pm } . 5 0 $ </td><td> $7 4 . 6 8 { \pm } . 4 8 $ </td><td></td><td>85.50±.32 .853±.003</td></tr><tr><td>SWIR1-only</td><td> $8 6 . 2 3 { \scriptstyle \pm . 3 9 }$ </td><td> $7 3 . 5 2 { \pm } . 8 6 $ </td><td> $6 5 . 8 0 { \scriptstyle \pm . 7 7 }$ </td><td> $7 9 . 3 7 { \pm } . 5 6 $ </td><td>.793±.005</td></tr><tr><td rowspan="3"></td><td>Dual-SWIR</td><td>85.66±.95</td><td>85.46±1.11</td><td> $\overline { { 7 4 . 7 5 \pm . 3 1 } }$ </td><td></td><td>85.55±.20.853±.002</td></tr><tr><td>Large-Fire SWIR2-only</td><td>85.93±.48</td><td> $8 5 . 2 7 \pm . 3 2 $ </td><td> $7 4 . 8 2 { \pm } . 2 7 $ </td><td></td><td>85.60±.18.854±.002</td></tr><tr><td> $\mathrm { S W I R 1 - o n l y }$ </td><td> $8 6 . 3 9 { \scriptstyle \pm . 3 3 }$ </td><td> $7 3 . 6 1 \pm . 6 8$ </td><td> $6 5 . 9 6 { \pm } . 4 5 $ </td><td> $7 9 . 4 9 { \pm } . 3 3 $ </td><td>.795±.003</td></tr></table>

## V. CONCLUSION

This work investigated wildfire segmentation under firescale distribution shifts using the Land8Fire dataset. To this end, we proposed a scale-based dataset partitioning strategy derived from the interquartile range of connected fire-region sizes. Experimental results demonstrated that model robustness strongly depends on both architectural design and spectral configuration. Among the evaluated models, U-Net achieved the strongest overall generalization performance, while Seg-Former showed competitive robustness under shifted fire-scale distributions. In contrast, DeepLabV3+ exhibited substantially lower recall, indicating reduced robustness under this evaluation setting.

Across all architectures, SWIR2-only consistently provided the strongest or near-best results, reinforcing the importance of SWIR bands information for active wildfire segmentation in Landsat-8 imagery. Overall, the proposed scale-based evaluation protocol provides a more challenging and informative benchmark than conventional random train-test splits, revealing model behaviors that may remain hidden under standard evaluation settings. Future work includes assessing sensitivity to the scale threshold, incorporating statistical significance testing across repeated runs, and validating the proposed protocol on additional datasets and sensors to evaluate its robustness under broader distribution shifts, including biome variability, seasonal changes, and cross-sensor domain adaptation scenarios.

## REFERENCES

[1] T. Ellis, D. Bowman, P. Jain, M. Flannigan, and G. Williamson, “Global increase in wildfire risk due to climate-driven declines in fuel moisture,” Global Change Biology, vol. 28, pp. 1544–1559, 2021.

[2] M. Senande-Rivera, D. Insua-Costa, and G. Miguez-Macho, “Spatial and temporal expansion of global wildland fire activity in response to climate change,” Nature Communications, vol. 13, 2022.

[3] M. W. Jones, J. Abatzoglou, S. Veraverbeke, N. Andela, G. Lasslop et al., “Global and regional trends and drivers of fire under climate change,” Reviews of Geophysics, vol. 60, 2022.

[4] J. Pereira, J. Mendes, J. S. S. Junior, C. Viegas, and J. R. Paulo, “Wildfire spread prediction model calibration using metaheuristic algorithms,” in IECON 2022 – 48th Annual Conference of the IEEE Industrial Electronics Society, 2022, pp. 1–6.

[5] J. Pereira, J. Mendes, J. S. S. Junior, C. Viegas, and J. R. Paulo, “Meta-´ heuristic algorithms for calibration of two-dimensional wildfire spread prediction model,” Engineering Applications of Artificial Intelligence, vol. 136, p. 108928, 2024.

[6] G. H. d. A. Pereira et al., “Active fire detection in landsat-8 imagery: a large-scale dataset and a deep-learning study,” ISPRS Journal of Photogrammetry and Remote Sensing, 2021.

[7] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in MICCAI, 2015.

[8] L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, and H. Adam, “Encoderdecoder with atrous separable convolution for semantic image segmentation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2018.

[9] E. Xie, W. Wang, Z. Yu, A. Anandkumar, J. M. Alvarez, and P. Luo, “Segformer: Simple and efficient design for semantic segmentation with transformers,” Advances in Neural Information Processing Systems, vol. 34, pp. 12 077–12 090, 2021.

[10] A. Tran et al., “Land8fire: A complete study on wildfire segmentation through comprehensive review, human-annotated multispectral dataset, and extensive benchmarking,” Remote Sensing, vol. 17, no. 16, p. 2776, 2025.

[11] D. Rashkovetsky, F. Mauracher, M. Langer, and M. Schmitt, “Wildfire detection from multisensor satellite imagery using deep semantic segmentation,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 7001–7016, 2021.

[12] X. Gao, W. Shi, M. Zhang, and L. Wang, “Dafdm: A discerning deep learning model for active fire detection based on landsat-8 imagery,” IEEE JSTARS, 2025.

[13] L. Giglio, W. Schroeder, and C. Justice, “The collection 6 modis active fire detection algorithm and fire products,” Remote Sensing of Environment, vol. 217, pp. 72–85, 2018.

[14] X. Hu, Y. Ban, and A. Nascetti, “Uni-temporal multispectral imagery for burned area mapping with deep learning,” Remote Sensing, 2021.

[15] R. Ghali and M. Akhloufi, “Deep learning approaches for wildland fires using satellite remote sensing data,” Fire, 2023.

[16] Y. Zhang, H. Li, and X. Wu, “Fireformer: Transformer-based multimodal wildfire segmentation from sentinel imagery,” Remote Sensing of Environment, 2024.

[17] Z. Wang, P. Yang, H. Liang et al., “Semantic segmentation and analysis on sensitive parameters of forest fire smoke using landsat-8 imagery,” Remote Sensing, vol. 14, no. 1, p. 45, 2021.

[18] Y. Ravan, A. Malek, C. Dolph, and N. Behari, “Real-time wildfire localization on the nasa autonomous modular sensor using deep learning,” arXiv, 2026.

[19] Y. Xu, A. Berg, and L. Haglund, “Sen2fire: A challenging benchmark dataset for wildfire detection using sentinel data,” in IEEE International Geoscience and Remote Sensing Symposium (IGARSS), 2024, arXiv:2403.17884.

[20] Y. Cui, W. Zhang, H. Wang, and T. Li, “A class-balanced loss with adaptive reweighting for imbalanced semantic segmentation,” Biomedical Signal Processing and Control, vol. 96, 2025.

[21] H. Singh, L. Ang, and S. K. Srivastava, “Active wildfire detection via satellite imagery and machine learning: an empirical investigation of australian wildfires,” Natural Hazards, vol. 121, no. 8, pp. 9777–9800, 2025.