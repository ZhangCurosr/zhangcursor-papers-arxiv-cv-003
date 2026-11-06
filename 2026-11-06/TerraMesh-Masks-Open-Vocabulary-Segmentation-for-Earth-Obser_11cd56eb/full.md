# TerraMesh-Masks: Open-Vocabulary Segmentation for Earth Observation

Benedikt Blumenstiel<sup>∗</sup> IBM Research ETH Zurich Zurich, Switzerland benedikt.blumenstiel@ibm.com

Johannes Jakubik IBM Research Zurich, Switzerland johannes.jakubik1@ibm.com

Hugues Devimeux<sup>∗</sup> IBM Research ETH Zurich Zurich, Switzerland hdevimeux@student.ethz.ch

Konrad Schindler ETH Zurich Zurich, Switzerland schindler@ethz.ch

## Abstract

Open-vocabulary segmentation (OVS) enables models to segment a variety of concepts specified in natural language, removing the need for fixed label sets and fine-tuning. While recent OVS approaches have shown strong performance on natural images, they transfer poorly to Earth observation (EO) data due to differing semantics, scales, sensing modality, and geographic context. To date, large-scale EO-specific datasets and benchmarks for OVS remain scarce. We, therefore, introduce TerraMesh-Masks, a large-scale open-vocabulary segmentation dataset for EO, comprising 16 million binary masks aligned with nine million multimodal TerraMesh samples and annotated using OpenStreetMap and Overture-derived semantics. In addition, we provide an expert-reviewed evaluation set of 523 samples covering over 200 classes to enable robust evaluation. To demonstrate the impact of TerraMesh-Masks, we build an open-vocabulary version of TerraMind. Our results demonstrate significantly improved performance over general-purpose stateof-the-art models reaching 35.18% IoU versus 16.29% for DINOv3.txt and 11.33% for SAM-3. We release the dataset publicly under a permissive license at https: //huggingface.co/datasets/ibm-esa-geospatial/TerraMesh-Masks.

## 1 Introduction

Open-vocabulary segmentation (OVS) promises more flexible and scalable interaction with Earth observation (EO) data compared to fine-tuning models for a fixed set of classes manually. Users could query models with flexible language rather than a pre-defined closed label set, and use a single model for a variety of concepts [1]. However, existing OVS models are developed primarily for natural images and transfer poorly to satellite imagery [2], where semantic concepts are expressed at lower spatial resolution, often require multispectral sensing, and vary strongly across geography and land-cover context. While previous works focused on high-resolution aerial imagery [3–5], the field currently lacks both a strong benchmark and a large-scale training resource for open-vocabulary segmentation with satellite data.

We address this gap with TerraMesh-Masks, a planetary-scale semantic-mask dataset, grounded in OpenStreetMap and Overture Maps, built on top of TerraMesh [6]. We additionally provide an expert-reviewed evaluation set called TerraMesh-Masks-Eval for benchmarking future OVS models. To demonstrate the value of this dataset, we adapt TerraMind to open-vocabulary binary segmentation. Our results show that EO-specific multimodal training is substantially more effective than applying a general-purpose baseline off the shelf. As shown in Fig. 1, TerraMind-OVS models outperform the generic models SAM-3 [1] and DINOv3.txt [7] as well as many EO-specific models like GeoPixel [3] in the vast majority of categories.

![](images/e7ba8793694890b1bf492bc92a5c8352274c8c0321550ccab011cc6ac59ae318.jpg)  
Figure 1: Average $\mathrm { I o U } _ { p o s }$ results per group on TerraMesh-Masks-Eval. While other models struggle with most categories, TerraMind-OVS performs particularly well on tasks related to land cover. Building-related categories are generally challenging due to many small masks which are best identified by GeoPixel.

We make the following contributions: (1) We introduce TerraMesh-Masks, a large-scale openvocabulary segmentation dataset for EO comprising 16 million multimodal image-mask samples with broad geographic and semantic diversity; (2) We provide an expert-reviewed evaluation set, enabling grounded benchmarking of OVS models on satellite imagery; and (3) We extend TerraMind to open-vocabulary segmentation as an empirical demonstration of the impact of TerraMesh-Masks, revealing substantially better performance compared to the state-of-the-art.

## 2 Related work

Vision–language models (VLMs), such as CLIP [8], establish robust joint image–text representations that represent the foundation of many open-vocabulary segmentation approaches. Building on these representations, early systems such as CAT-Seg [9] and SAN [10] demonstrated the feasibility of OVS for natural images. More recently, models including SAM-3 [1] and DINOv3.txt [7] have been explicitly designed for spatial segmentation in open-vocabulary settings. Despite their broad training, generalist models often struggle to transfer to domain-specific inputs [2].

Compared to natural images, remote sensing imagery exhibits strong variability in spatial resolution, viewpoint, seasonal effects, and geographic distribution. Moreover, many semantic classes are only weakly expressed in RGB imagery. To address these issues, several works adapt VLMs to the EO domain. MS-CLIP [11] leverages LLM-generated captions for multispectral imagery, demonstrating significant performance gains when incorporating additional spectral bands. SegEarth-OV [12], DINO Soars [13], and other approaches [14, 5] focus on training-free adaptation of CLIP, SAM, or DINO to enable OVS in EO, with strong results on high-resolution imagery. Beyond training-free methods, OVRSISS [15] and AerOSeg [16] introduce high-resolution datasets to fine-tune CLIP- and SAM-based models for aerial imagery. Similarly, MM-OVSeg [17] employs 25k multimodal samples for supervised adaptation. Multimodal LLM-based systems such as GeoPixel [3] and RemoteSAM [4] extend this paradigm to broader vision–language tasks. EarthMind [18] further scales training by combining six public segmentation datasets with labels generated by GPT-4o.

Tab. 1 summarizes existing EO datasets for OVS. Other datasets often focus on high-resolution imagery, cover a limited set of semantic classes, or are proprietary and not publicly available [15,

Table 1: Overview of existing remote sensing datasets for OVS. Most datasets focus on high-resolution imagery. Some dataset licenses might not reflect source dataset restrictions.
<table><tr><td>Dataset</td><td>Number of images</td><td>Vocabulary size</td><td>Modalities</td><td></td><td>Resolution Image source</td><td>Annotation source</td><td>License</td></tr><tr><td>GeoPixelD [3]</td><td>18.7k</td><td>53k</td><td>RGB</td><td>~1m</td><td>iSAID</td><td>Source labels</td><td>CC-BY 4.0</td></tr><tr><td>MM-OVSeg [17]</td><td>25k</td><td>unknown</td><td>RGB, SAR</td><td>0.5-3 m</td><td>SpaceNet6; DFC2023</td><td>unknown</td><td>unknown</td></tr><tr><td>LandDiscover50K [15]</td><td>51k</td><td>40</td><td>RGB</td><td>high-res</td><td>9 datasets</td><td>Source labels</td><td>MIT</td></tr><tr><td>RemoteSAM-270K [4]</td><td>270k</td><td>1000</td><td>RGB</td><td>high-res</td><td>10 datasets</td><td>Source labels</td><td>CC-BY 4.0</td></tr><tr><td>EarthMind [18]</td><td>&gt;500k</td><td>unknown</td><td>Optical, SAR</td><td>various</td><td>7+ datasets</td><td>Source labels</td><td>Apache 2.0</td></tr><tr><td>TerraMesh-Masks (ours)</td><td>9M</td><td>44k</td><td>S1, S2, RGB, DEM</td><td>10m</td><td>TerraMesh</td><td>Overture, OpenStreetMap</td><td>ODbL</td></tr></table>

3, 4, 18, 17, 19]. Additionally, existing models are often evaluated on benchmarks dominated by a small number of land cover categories or high-resolution object classes.

TerraMesh-Masks is designed to address these limitations by providing a large-scale multimodal training dataset and a semantically diverse evaluation benchmark. To the best of our knowledge, TerraMesh-Masks is the first dataset that extends a large-scale unlabeled pretraining dataset to openvocabulary segmentation by grounding images in OpenStreetMap and Overture data and adding annotations from a large language model (LLM), enabling substantially broader semantic coverage and an order of magnitude more training samples compared to prior work.

## 3 TerraMesh-Masks

TerraMesh-Masks forms a planetary-scale, multimodal, analysis-ready dataset for open-vocabulary segmentation in Earth observation, built on top of TerraMesh [6]. TerraMesh itself combines data from SSL4EO-S12 v1.1 [20] and MajorTOM [21]. The dataset provides over nine million globally distributed co-registered patches at 10 m resolution across modalities such as Sentinel-1 SAR, Sentinel-2 optical, DEM, NDVI, and land-cover layers [6]. TerraMesh-Masks complements the original TerraMesh dataset by adding binary semantic masks with language annotations. In total, TerraMesh-Masks contains roughly 16 million masks derived from OpenStreetMap and Overture Maps, making it, to the best of our knowledge, one of the largest resources for segmentation in Earth observation. More importantly, TerraMesh-Masks has been explicitly designed for open-vocabulary learning. TerraMesh-Masks inherits the broad geographic diversity of TerraMesh and covers 174 countries spanning 22 continental regions. Rather than concentrating on a small set of curated benchmark geographies, it exposes models to diverse landscapes, climates, and built environments at planetary scale. Examples from the dataset are shown in Fig. 2.

![](images/21b12f7e24791c146f1d2b0feefa21b662cf624e74ddf0bb8592c8f56da6c1fe.jpg)  
Figure 2: TerraMesh-Masks samples drawn from the manually reviewed Eval set. Each patch is shown with a mask overlay and its open-vocabulary class label, country, and a synonym set.

Dataset creation overview. We start from TerraMesh image patches and associated geospatial metadata. We then obtain semantic-mask candidates and refine them into open-vocabulary segmentation annotations. The resulting dataset expands TerraMesh with pixel-level supervision and language annotations suitable for segmentation. The pipeline integrates two complementary geographic data sources to extract candidates for semantic masks. Overture Maps provides a land-cover-focused base layer with hundreds of standardized semantic annotations that cover all global regions. A challenge is its semi-automated annotation approach, which can lead to imprecise or incorrect annotations. OpenStreetMap (OSM) complements Overture tags and represents a rich collection of volunteercontributed geographic features, from which we extract a broad set of tag categories including natural, highway, building, waterway, amenity, and related structured and semi-structured tags. Compared to Overture, OSM provides rich long-tail semantic coverage, but its schema is heterogeneous and often noisy from the perspective of segmentation supervision. Both sources are spatially filtered against TerraMesh image patches, retaining only geometries intersecting the available satellite observations. We summarize the data pipeline in Fig. 3 and provide additional descriptions of the LLM processing and filtering within this section and the supplementary material.

![](images/1c3e53c11861165abd2a450166ddb8550cb1c964d7606a4b5bef9640367d83b3.jpg)  
Figure 3: Our dataset creation pipeline pairs TerraMesh samples with Overture and OpenStreetMap annotations. For each sample, we compute heuristics based on subtype, location, polygon size, and previous LLM quality scores. The sampled annotation is processed by Mistral-Large-3 which first scores the annotation quality and then generates the class label and synonyms. The annotations are further subsampled with an NDVI-based filtering for pretraining. We tasked experts to manually review a subset of the dataset to obtain a high-quality evaluation dataset.

Preprocessing and geometric filtering. Prior to quality assessment, candidate masks undergo several preprocessing steps. First, area-based filtering retains only candidate masks with sufficient spatial extent, using a default threshold of 1,000 m<sup>2</sup>. This step removes sub-pixel features that are unlikely to be detectable at Sentinel-2 resolution. Second, geometry dissolution merges adjacent or overlapping features with identical semantic labels to create more contiguous regions, reduce fragmentation, and improve spatial coherence. Third, geographic contextualization enriches each mask with country and continent metadata.

Pre-filtering and candidate selection. The pipeline employs a hierarchical filtering strategy that combines heuristic pre-filters with LLM-based semantic validation. At the scale of our raw candidate pool, full LLM validation of every mask is computationally infeasible as the dataset generation starts from roughly 93 million candidate masks. We, therefore, use heuristic pre-filtering not only as a quality-control mechanism, but also as a practical necessity to reduce the candidate set to a tractable subset for large-scale semantic annotation via an LLM. Thus, the filtering stage is important for both efficiency and dataset quality. For example, a naive strategy that submits all raw candidates to an LLM would spend most of the annotation budget on masks that are either clearly implausible, visually unsupported, too small to be useful at Sentinel-2 resolution, or massively overrepresented. Instead, our prefilter concentrates LLM compute on candidates that are more likely to yield valid segmentation supervision while still preserving geographic and semantic diversity. After pre-filtering, we sample up to three candidate masks per image using weighted random sampling. The weights are computed by combining the same three heuristics used in filtering, specifically mask area size, previous LLM quality scores of the same annotation tag, and the number of previously selected masks of this tag. The sampling score is composed as follows: $p = p _ { \mathrm { f r e q } } \cdot p _ { \mathrm { s p e c } } \cdot p _ { \mathrm { c l a s s } } .$ , where $p _ { \mathrm { f r e q } }$ blends a global and an H3-local (around $1 0 0 \mathrm { k m ^ { 2 } }$ areas) quality prior for the annotation tag, $p _ { \mathrm { s p e c } }$ is the NDVI agreement score, and $p _ { \mathrm { c l a s s } }$ downweights already overrepresented tags. The NDVI agreement score measures spectral separability by comparing the NDVI statistics inside and outside the candidate polygon, so a mask whose interior is spectrally distinct from its surroundings scores higher. The heuristics balance compute efficiency, label exploration, and dataset diversity, which we further outline with full definitions of all terms in the supplementary material. Our filtering discards 43.5 million candidate masks (46.4%) and after sample selection reduces the dataset size required to be processed by the LLM to 20.17 million samples (21.5% of the raw pool). Among those candidates, 16.07 million are ultimately accepted by the LLM as we detail in the next section. That corresponds to an acceptance rate of 79.7%. We interpret this high post-filter acceptance rate as an indication that the heuristic stage removes a large fraction of obviously poor candidates while preserving a high-yield subset for downstream semantic annotation.

LLM-based validation and labeling. Following heuristic pre-filtering, candidate masks undergo LLM-based validation to assess semantic consistency. After evaluating the quality and openness of several multimodal LLMs, we selected Mistral-Large-3 [22] for semantic validation and labeling. The LLM evaluates the visual coherence between the satellite imagery and the provided mask, generating both quality assessments and class labels. For each candidate mask, the LLM receives a structured input containing the satellite imagery as RGB with the candidate polygon overlaid, geometric metadata (including area in m<sup>2</sup>, country, and continent), the provisioned subtypes tags from OSM or Overture, and raw heuristic quality scores. The prompt casts the model as a conservative annotator, instructs it to treat subtypes as the primary signal for the concept while using the image to detect clear contradictions, penalizes ambiguity, and rejects concepts that are administrative rather than visually grounded, such as legal borders. The model outputs an explanation of visual coherence between the polygon mask and the image patch alongside a quality score (between 0 and 1) as well as a refined natural-language class label synthesizing tag metadata and visual content. Finally, the model provides synonyms as alternative semantic descriptions. We constrain generation to a fixed JSON schema and accept every mask scoring above a cut-off of 0.5. Validation and labeling required around 4,000 NVIDIA H200 GPU hours for 20M sampled masks. We provide a longer explaination of the approach in the supplementary material.

Scale and diversity. The central value of TerraMesh-Masks is not only its scale, but also the breadth of its vocabulary. We observe more than 44,000 unique class labels for over 16 million labeled masks, together with over 537,000 unique synonym strings. The most frequent labels include broad land-cover concepts such as shrubland,forest, grassland, barren land, and river. The long tail contains more diverse semantic concepts such as coral reef, golfcourse, and airport taxiway. The combination of large-scale supervision and rich lexical variation makes TerraMesh-Masks particularly suitable for open-vocabulary segmentation in EO.

Human-reviewed evaluation set. Beyond the large automatically constructed corpus, we curate a human-reviewed evaluation benchmark from the validation split. Four expert annotators provided over 4200 reviews, with each mask being labeled as valid, unsure, or invalid. Using redundant labeling, this process covered 2,300 candidate masks. For the evaluation set, we select any mask with at least two valid reviews and no invalid review. The pairwise agreement rate between two reviewers is between 62% and 83% (when ignoring unsure). We filter duplicates of the same image by selecting the more infrequent class label, and limit each class label to at most 10 samples to increase diversity. This results in 523 high-quality masks with 207 unique class labels, with examples shown in Fig. 2. We manually group the selected samples into 14 semantic clusters, used also in Fig. 1, to better interpret the strengths and weaknesses of OVS models. For better usability with machine learning models, we center crop the samples to 256 × 256 pixels. To avoid data leakage, we remove the selected masks and all spatially overlapping samples from TerraMesh-Masks.

The manual review provides an upper bound on how noisy the automatically constructed corpus is: Across the 4,211 individual reviews, 44% of verdicts are valid, 29% are invalid, and 28% are unsure. Reasons for rejection include inaccurate or very small masks as well as masks which cover the full image and are, therefore, not well suited for evaluation. We also exclude wrong class labels and and ambiguous classes which are not detectable at 10 m resolution, like government building. The invalid rate should not be read directly as the training-set error rate, since a mask can be correct but not verifiable from the image alone, which is unusable for evaluation yet still an acceptable training target. Furthermore, masks flagged as invalid are often partially wrong with many correctly labeled pixels (see examples in Fig. 13 of the supplementary material).

Evaluation protocol. Using the expert-reviewed validation samples, we propose TerraMesh-Masks-Eval, a subset comprising 523 samples and more than 200 semantic concepts, available at https://huggingface.co/datasets/ibm-esa-geospatial/ TerraMesh-Masks-Eval. For each image–query pair, models must generate a binary mask. We measure performance as Intersection-over-Union and F1 score, computed at sample level and aggregated across all samples. To expose systematic patterns in performance, results are additionally reported by high-level semantic groups such as infrastructure, land cover, and transportation. No restrictions are placed on model architecture or training strategy. The benchmark inherently captures multiple degrees of vocabulary openness. These include queries that are present during training as well as synonyms and lexical variants. We provide the evaluation queries as part of TerraMesh-Masks-Eval, ensuring reproducibility and consistent comparison across methods.

## 4 TerraMind-OVS

We build on top of TerraMind [23] and adapt it to open-vocabulary binary segmentation. We deliberately employ minimal architectural changes in order to demonstrate that performance gains can largely be attributed to our pretraining dataset, TerraMesh-Masks. TerraMind was trained on several modalities during pretraining and previous work demonstrated simple adaptation to new modalities with continuous pretraining. Therefore, we treat open-vocabulary binary masks as an additional modality. TerraMind is trained in a discrete token-prediction paradigm: the decoder predicts token IDs, the encoder-decoder architecture is optimized with via a cross-entropy loss. That structure makes it natural to cast segmentation as predicting a sequence of binary mask tokens, as previously done in literature [24]. We present an overview of the binary mask tokenizer and TerraMind-OVS in Fig. 4 and Fig. 5. We continuously pretrain TerraMind’s existing multimodal encoder-decoder and primarily specialize the language conditioning as well as the decoding to the mask-prediction setting. For language conditioning, we encode the class query with a CLIP text encoder (OpenAI ViT-B-32) and project the resulting embedding into TerraMind’s latent space using a lightweight projection head. The 512-dimensional text embedding is passed through a two-layer MLP with GELU activation and hidden size 1024, and then mapped to 5 × d outputs, where d is TerraMind’s latent token dimension. The resulting vector is reshaped into five learned query tokens that are concatenated to the decoder context, allowing the model to condition its prediction on the prompt.

TerraMind-OVS predicts a sequence of 196 mask tokens as output corresponding to a 14 × 14 spatial token grid. We use a seperately trained binary-mask tokenizer to discretize each segmentation mask into a grid of discrete tokens, following the setting from TerraMind’s Land-Use-Land-Cover tokenizer. In line with the general TerraMind framework, this tokenizer later also allows us to decode predicted tokens back to pixel space.

![](images/8519ea92beca63661d01b2684e555cef1844d804af0ea91e218debfff10f3988.jpg)  
Figure 4: Tokenizer architecture for discretizing binary segmentation masks into tokens. The tokens are later used as training targets of TerraMind-OVS, while the ViT decoder is used during inference.

![](images/4f3950a74521951067d4dd6ae435e18bef1eaf65d5c8814c847c1458c74a989b.jpg)  
Figure 5: Overview of our model architecture. Multimodal inputs are patchified and encoded with TerraMind. The text query is encoded using a frozen CLIP text encoder followed by an MLP. Both embeddings are used as context by the TerraMind decoder to predict the binary mask tokens. During inference, the predicted tokens are de-tokenized to generate binary segmentation masks.

## 5 Experimental setup

Training setup. We continuously pretrain TerraMind variants small, base, and large. Training samples from TerraMesh-Masks are processed through a filtered WebDataset pipeline. To mitigate class imbalance, we apply class-dependent repeat-factor resampling during training. We also filter out masks with invalid geometry, such as masks that are too small or span the full frame. Images are cropped to 224 × 224 pixels.

Across our training setups, models are trained with inputs from Sentinel-1 RTC, Sentinel-1 GRD, Sentinel-2 L2A, Sentinel-2 RGB, DEM, and a text query, and generally only predict tokens of the binary semantic mask on a patch size of 16. We train two model versions, one with all modalities and one limited to S2 L2A and RGB, labeled with S2 in the model name. The text query is built from the class label associated with each mask. We use the MS-CLIP prompt templates [11] to adapt the embeddings to satellite data. When synonyms are available, we sample between the main label and its variants, with the main label sampled with a factor f = 3 relative to any synonym during training. This exposes the model to linguistic variations while keeping training anchored to the dataset taxonomy.

The supervision signal is cross-entropy on the target mask tokens. We train TerraMind-OVS-S with a global batch size of 1024 and a learning rate of 5e-4, while base and large models use a batch size of 512 and a learning rate of 1e-4. All runs use an AdamW optimizer with a cosine decay scheduler and 1 million samples for warmup. Differrent from TerraMind pretraining, we do not apply any target masking and each sample includes 196 target tokens for a full binary mask. With 9M samples in TerraMesh, we contiuously pretrain for 15B tokens (3% of the original training). Training for TerraMind-OVS-L required around 250 NVIDIA A100 80GB GPU hours.

Benchmark setup. While TerraMind-OVS is trained on 224×224 inputs, we pass the full 256×256 images without resizing during evaluation. For multiclass datasets, we run the model per class and stack the logits before appling the argmax step. For benchmark models, we resize the images to the default image resolution such as 224 × 224 pixels for DINO Soars, 512 × 512 for DINOv3.txt, and 1008 × 1008 for SAM-3. We use a confidence threshold of 0.1 for SAM-3 and 0.5 for RemoteSAM, which optimizes the models’ mIoU scores. DINOv3.txt uses the ImageNet prompt templates, while we also tested MS-CLIP templates, which led to lower performance. For DINOv3.txt, DINO Soars, and SegEarth-OV, we use a background label for the negative class after testing several candidates. For GeoPixel, we follow the default settings of reshaping the image to 1680 × 1680 pixel and processing it in 9 patches. We use Could you provide a segmentation mask for label in this image? as a prompt following the authors’ examples.

![](images/fac6d4f6210d257bd13cd4d7ecb767e7ae5b2c240bac62daefb9488d5da7db19.jpg)  
Figure 6: Qualitative comparison on examples from TerraMesh-Masks-Eval. TerraMind performs well on larger land-cover related labels, while GeoPixel is better in spotting objects. Other models lean towards overprediction or no prediction.

## 6 Results

Evaluation performance. We summarize the performance of TerraMind-OVS and baselines in Tab. 2 and report Student-t 95% confidence intervals over the 523 evaluation samples. We compare against generalist open-vocabulary models (SAM-3 [1], DINOv3.txt [7]) as well as four recent EO-specific ones (RemoteSAM [4], DINO Soars [13], SegEarth-OV [12], GeoPixel [3]). Overall, we observe that TerraMind-OVS outperforms all baselines on EO-specific data across most metrics and model sizes. TerraMind-OVS-L improves over SAM-3 and DINOv3.txt by 479% and 53%, respectively, measured in relative mIoU improvement. The EO-specific baselines are considerably stronger, with SegEarth-OV reaching the best $\mathrm { I o U } _ { p o s }$ with 21.59%, yet TerraMind-OVS-L still improves on it by 13.59pp $\mathrm { I o U } _ { p o s }$ . GeoPixel has the second-best mIoU score, 14.29pp below TerraMind. That emphasizes the relevance and impact of TerraMesh-Masks as a domain-specific pretraining dataset. Fig. 6 illustrates zero-shot predictions of TerraMind-OVS models and baselines at inference time. Most baseline models lean towards overprediction, which also results in a high average recall.

We report average inference times on a single A100: SAM-3 requires 310 ms per sample, DINO Soars 319 ms, SegEarth-OV 488 ms, RemoteSAM 524 ms, and DI-NOv3.txt 432 ms, while GeoPixel is significantly slower with 6,298 ms. In comparison, the TerraMind-OVS architecture is more efficient and requires between 52 and 73 ms, mainly limited by data loading speed rather than inference.

Failure cases. Beyond reporting gains over existing models, we showcase a qualitative assessment in Fig. 8 to demonstrate limitations of our model. TerraMind-OVS struggles on classes that are not part of TerraMesh-Masks and, therefore, completely unseen, such as weather patterns (clouds and snow) and specific terminology that is not covered by the synonyms (e.g., terminal).

Effect of multimodal and multispectral inputs. We ablate the effect of multimodal and multispectral inputs to TerraMind-OVS in Tab. 2 and in Fig. 7. Overall, we observe top performances for using multispectral S2 input and a generally strong performance with RGB inputs that is likely related to the fact that the vast majority of objects in OpenStreetMap and Overture Maps are visible in RGB. We observe particular benefits of multispectral inputs on groups like Forest & Wetland as well as Barren terrain. Contrary to our expectations, multimodal input during inference does not further enhance the predictions of TerraMind-OVS. This points to a need for more sophisticated multimodal architectural designs for multimodal OVS.

![](images/c13f35d10f40f211677d6adc24097785d3169486e99d37ced56a3bd6053ed3f7.jpg)  
Figure 7: Eval mIoU results per semantic group for TerraMind-OVS-B with different modality inputs. S2 and RGB perform compareable, while S1 and DEM significalty reduce the performance.

Forest  
Table 2: TerraMesh-Masks-Eval results with Student-t 95% confidence intervals, each metric averaged over the 523 per-sample results. Some TerraMind versions (TM-OVS) are trained on multimodal data that includes S1, S2, RGB, and DEM. Best results are in bold, second-best results are underlined.
<table><tr><td>Model</td><td>Param.</td><td>Training</td><td>Input</td><td> $\mathrm { I o U } _ { p o s }$ </td><td>mIoU</td><td>F1</td><td>Precision</td><td></td><td>Recall</td></tr><tr><td>SAM-3</td><td>841M</td><td>RGB</td><td>RGB</td><td> $1 1 . 3 3 \pm 2 . 3 0$ </td><td> $1 0 . 9 7 \pm 1 . 3 9$ </td><td> $1 3 . 3 2 \pm 2 . 5 4$ </td><td></td><td> $1 2 . 8 6 \pm 2 . 5 8$ </td><td> $2 0 . 9 7 \pm 3 . 3 3$ </td></tr><tr><td>RemoteSAM</td><td>242M</td><td>RGB</td><td>RGB</td><td> $1 4 . 7 9 \pm 2 . 1 3$ </td><td> $3 0 . 4 2 \pm { 1 . 8 1 }$ </td><td> $1 9 . 6 2 \pm 2 . 4 8$ </td><td> $2 0 . 6 0 \pm 2 . 8 4$ </td><td></td><td> $5 0 . 1 9 \pm 3 . 4 9$ </td></tr><tr><td>DINOv3.txt</td><td>867M</td><td>RGB</td><td>RGB</td><td> $1 6 . 2 9 \pm 1 . 8 7$ </td><td> $4 1 . 4 0 \pm 1 . 4 2$ </td><td> $2 3 . 1 4 \pm 2 . 2 8$ </td><td> $2 4 . 7 0 \pm 2 . 7 2$ </td><td></td><td> ${ \bf 5 1 . 5 3 \pm 3 . 5 0 }$ </td></tr><tr><td>DINO Soars</td><td>869M</td><td>RGB</td><td>RGB</td><td> $1 9 . 4 9 \pm 2 . 5 5$ </td><td> $4 4 . 2 9 \pm 1 . 9 3$ </td><td> $2 4 . 5 7 \pm 2 . 8 3$ </td><td> $2 3 . 6 5 \pm 2 . 9 2$ </td><td></td><td> $4 8 . 4 3 \pm 3 . 9 8$ </td></tr><tr><td>SegEarth-OV</td><td>150M</td><td>RGB</td><td>RGB</td><td> $2 1 . 5 9 \pm 2 . 4 7$ </td><td> $4 8 . 3 8 \pm 1 . 5 3$ </td><td> $2 8 . 0 1 \pm 2 . 7 7$ </td><td> $3 0 . 8 1 \pm { 3 . 1 0 }$ </td><td></td><td> $4 5 . 3 4 \pm 3 . 5 1$ </td></tr><tr><td>GeoPixel</td><td>11.3B</td><td>RGB</td><td>RGB</td><td> $1 5 . 0 6 \pm 1 . 9 9$ </td><td> $4 9 . 2 5 \pm 1 . 5 8$ </td><td> $2 0 . 6 2 \pm 2 . 4 1$ </td><td> $4 0 . 2 7 \pm 3 . 6 9$ </td><td></td><td> $2 0 . 1 5 \pm 2 . 5 7$ </td></tr><tr><td>TM-OVS-S</td><td></td><td>248M Multimodal</td><td>RGB</td><td> $2 6 . 5 2 \pm 3 . 2 8$ </td><td> $5 8 . 7 6 \pm 1 . 7 1$ </td><td> $2 9 . 7 5 \pm 3 . 5 0$ </td><td> $3 3 . 8 0 \pm { 3 . 7 2 }$ </td><td></td><td> $3 0 . 1 5 \pm 3 . 5 4$ </td></tr><tr><td>TM-OVS-S</td><td></td><td>247M S2, RGB</td><td>RGB</td><td> $2 7 . 3 7 \pm 3 . 2 7$ </td><td> $5 9 . 0 8 \pm 1 . 7 1$ </td><td> $3 0 . 8 8 \pm { 3 . 5 1 }$ </td><td> $3 4 . 2 4 \pm 3 . 7 2$ </td><td></td><td> $3 2 . 4 2 \pm { 3 . 6 2 }$ </td></tr><tr><td>TM-OVS-B</td><td></td><td>421M Multimodal</td><td>RGB</td><td> $3 2 . 2 8 \pm { 3 . 4 2 }$ </td><td> $6 2 . 0 0 \pm 1 . 7 8$ </td><td> $3 6 . 1 8 \pm 3 . 6 4$ </td><td> $4 0 . 1 2 \pm 3 . 8 0$ </td><td></td><td> $3 7 . 7 9 \pm 3 . 7 4$ </td></tr><tr><td>TM-OVS-B</td><td></td><td>420M S2, RGB</td><td>RGB</td><td> $3 2 . 1 8 \pm 3 . 4 0$ </td><td> $6 1 . 6 7 \pm { 1 . 8 3 }$ </td><td> $3 6 . 2 0 \pm 3 . 6 2$ </td><td> $4 0 . 7 9 \pm 3 . 8 2$ </td><td></td><td> $3 7 . 9 0 \pm 3 . 7 3$ </td></tr><tr><td>TM-OVS-L</td><td></td><td>935M Multimodal</td><td>RGB</td><td> $3 4 . 8 7 \pm 3 . 3 5$ </td><td> $6 2 . 9 0 \pm 1 . 9 0$ </td><td> $3 9 . 7 5 \pm 3 . 5 9$ </td><td> $\underline { { 4 5 . 3 2 } } \pm 3 . 8 0$ </td><td></td><td> $4 0 . 1 5 \pm 3 . 6 6$ </td></tr><tr><td>TM-OVS-L</td><td>934M</td><td>S2, RGB</td><td>RGB</td><td> $3 4 . 9 6 \pm 3 . 3 6$ </td><td> $6 3 . 2 7 \pm 1 . 7 6$ </td><td> $\underline { { 3 9 . 8 3 } } \pm 3 . 5 9$ </td><td> $4 4 . 9 4 \pm 3 . 7 9$ </td><td></td><td> $4 0 . 6 1 \pm 3 . 6 6$ </td></tr><tr><td>TM-OVS-S</td><td></td><td>248M Multimodal</td><td>S2</td><td> $2 6 . 3 4 \pm 3 . 2 6$ </td><td> $5 8 . 4 9 \pm 1 . 7 1 $ </td><td> $2 9 . 6 3 \pm 3 . 4 8$ </td><td> $3 4 . 8 6 \pm 3 . 7 6$ </td><td></td><td></td></tr><tr><td>TM-OVS-S</td><td></td><td>247M S2, RGB</td><td>S2</td><td> $2 7 . 2 8 \pm 3 . 2 8$ </td><td> $5 8 . 9 8 \pm 1 . 7 2 $ </td><td> $3 0 . 7 5 \pm 3 . 5 0$ </td><td> $3 3 . 9 1 \pm { 3 . 7 1 }$ </td><td></td><td> $2 9 . 9 7 \pm 3 . 5 3$   $3 2 . 4 3 \pm 3 . 6 3$ </td></tr><tr><td>TM-OVS-B</td><td></td><td>421M Multimodal</td><td>S2</td><td> $3 2 . 1 9 \pm 3 . 4 2$ </td><td> $6 2 . 0 4 \pm 1 . 7 7$ </td><td> $3 6 . 0 8 \pm 3 . 6 4$ </td><td> $3 9 . 8 6 \pm 3 . 8 1$ </td><td></td><td> $3 7 . 5 4 \pm 3 . 7 4$ </td></tr><tr><td>TM-OVS-B</td><td></td><td>420M S2, RGB</td><td>S2</td><td> $3 2 . 4 4 \pm 3 . 4 1$ </td><td> $6 2 . 0 4 \pm { 1 . 8 2 }$ </td><td> $3 6 . 4 7 \pm 3 . 6 2$ </td><td> $4 0 . 7 3 \pm 3 . 8 1$ </td><td></td><td> $3 8 . 2 5 \pm 3 . 7 3$ </td></tr><tr><td>TM-OVS-L</td><td></td><td>935M Multimodal</td><td>S2</td><td> $3 5 . 0 9 \pm 3 . 3 5$ </td><td> $6 2 . 9 5 \pm 1 . 8 9$ </td><td> ${ \bf 4 0 . 0 4 } \pm 3 . 5 9$ </td><td> $4 5 . 5 7 \pm 3 . 8 0$ </td><td></td><td> $4 0 . 6 5 \pm 3 . 6 6$ </td></tr><tr><td>TM-OVS-L</td><td></td><td>933M S2, RGB</td><td>S2</td><td> $3 5 . 1 8 \pm 3 . 4 1$ </td><td> ${ \bf 6 3 . 5 4 \pm 1 . 8 1 }$ </td><td> $3 9 . 8 0 \pm 3 . 6 3$ </td><td> $4 4 . 7 6 \pm 3 . 8 2$ </td><td></td><td> $4 0 . 9 7 \pm 3 . 7 1$ </td></tr></table>

Cloud  
Vegetation  
Dirt track  
River  
![](images/7e7481576f2f7d0e0988197e533602934bee5f5d2be0db803b220467a18bdf12.jpg)  
Figure 8: These predictions from TerraMind-OVS-L (S2) show that the model does not have an understanding of clouds, snow, and other weather patterns as these are not included in the annotations. The model covers a range of concepts, but can still fail for related terms like dirt track or terminal.

Multiclass segmentation. In Tab. 3, we test TerraMind-OVS in more challenging multi-label segmentation and out-of-distribution scenarios. Our models outperform generalist models by up to

7.78pp mIoU on reBEN-7k [25] and perform competitively on out-of-distribution grayscale SAR imagery from the CaFFe dataset [26]. Since CaFFe does not provide true SAR amplitude data, we replicate the grayscale channel to construct a pseudo-RGB input for all models, ensuring fair comparison (see supplementary material). Interestingly, we generally observe less scaling behavior in model size for the multi-label evaluations compared to the experiments on TerraMesh-Masks-Eval. We largely attribute this to how challenging the underlying task is.

Table 3: Open-vocabulary segmentation performance across two multi-label datasets: CaFFe [26] and reBEN-7k [25, 27]. Thresholds for SAM-3 are optimized over the test set for maximum performance.
<table><tr><td rowspan="2">Model</td><td colspan="2">CaFFe (OOD)</td><td colspan="2">reBEN-7k (Test split)</td><td colspan="2">reBEN-7k (OOD split)</td></tr><tr><td>mIoU (%)</td><td>F1 (%)</td><td>mIoU (%)</td><td>F1 (%)</td><td>mIoU (%)</td><td>F1 (%)</td></tr><tr><td>DINOv3.txt</td><td>15.32</td><td>22.64</td><td>3.52</td><td>5.20</td><td>3.07</td><td>4.60</td></tr><tr><td>SAM-3</td><td>1.00</td><td>1.24</td><td>7.30</td><td>1.00</td><td>8.76</td><td>1.03</td></tr><tr><td>TerraMind-OVS-S</td><td>14.57</td><td>22.82</td><td>10.93</td><td>18.40</td><td>11.26</td><td>17.59</td></tr><tr><td>TerraMind-OVS-S (S2)</td><td>14.35</td><td>22.29</td><td>11.46</td><td>19.25</td><td>14.06</td><td>21.23</td></tr><tr><td>TerraMind-OVS-B</td><td>9.10</td><td>14.14</td><td>13.75</td><td>22.23</td><td>16.54</td><td>24.32</td></tr><tr><td>TerraMind-OVS-B (S2)</td><td>12.10</td><td>19.39</td><td>12.60</td><td>20.72</td><td>14.69</td><td>21.95</td></tr><tr><td>TerraMind-OVS-L</td><td>8.24</td><td>12.40</td><td>13.66</td><td>21.97</td><td>12.17</td><td>19.86</td></tr><tr><td>TerraMind-OVS-L (S2)</td><td>8.26</td><td>12.44</td><td>12.74</td><td>20.87</td><td>14.89</td><td>21.93</td></tr></table>

## 7 Limitations

Scope of the vocabulary. OpenStreetMap and Overture Maps include persistent map features, so TerraMesh-Masks describes static land cover and infrastructure. Dynamic phenomena such as clouds, floods, fire, or seasonal change are absent from the vocabulary, and a model trained only on our dataset cannot segment them. Covering them would require annotations that are time-stamped against the acquisition date of the image.

Annotation noise. The corpus is constructed automatically and therefore introduces noise. Wrong class labels, imprecise boundaries, and concepts that are not resolvable at 10 m all remain in the training data. Our filtering heuristics reduce but do not eliminate them, as the manual review of the evaluation samples demonstrates.

Geographic coverage. Both the training corpus and the evaluation set inherit the geographic bias of OpenStreetMap, which is denser in Europe and North America than in much of Africa, Central Asia, and South America. High latitudes are barely covered, with 13 of the 523 evaluation samples above 60° and none in Antarctica.

Size of the evaluation set. TerraMesh-Masks-Eval contains 523 expert-reviewed masks, which we report with 95% confidence intervals throughout. The intervals are wide enough that it remains chal lenging to distinguish different TerraMind-OVS variants from each other. The per-group breakdowns rest on as few as ten samples for the rarest groups. The benchmark is reliable for the large differences between model families that we discuss, and should not be used to rank closely matched checkpoints.

## 8 Conclusion

In this work, we address a critical gap in open-vocabulary segmentation for Earth observation by introducing TerraMesh-Masks, a large-scale multimodal OVS dataset designed to reflect the richness and complexity of satellite data. By extending TerraMind to open-vocabulary segmentation, we demonstrate the value of TerraMesh-Masks and its potential to unlock substantial performance gains. We hope that our dataset, benchmarks, and strong baselines lay the foundation for scalable, modality-aware OVS in EO and enable broader adoption and future research in this field.

## Acknowledgments and Disclosure of Funding

This work is part of the FAST-EO – CCN2 project funded by the European Space Agency (ESA) Φ-Lab under contract number 4000143501/23/I-DT.

## References

[1] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris Coll-Vinent, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. Sam 3: Segment anything with concepts. In International conference on learning representations, volume 2026, pages 138846–138923, 2026.

[2] Benedikt Blumenstiel, Johannes Jakubik, Hilde Kühne, and Michael Vössing. What a mess: Multi-domain evaluation of zero-shot semantic segmentation. Advances in Neural Information Processing Systems, 36:73299–73311, 2023.

[3] Akashah Shabbir, Mohammed Zumri, Mohammed Bennamoun, Fahad S Khan, and Salman Khan. Geopixel: Pixel grounding large multimodal model in remote sensing. arXiv preprint arXiv:2501.13925, 2025.

[4] Liang Yao, Fan Liu, Delong Chen, Chuanyi Zhang, Yijun Wang, Ziyun Chen, Wei Xu, Shimin Di, and Yuhui Zheng. Remotesam: Towards segment anything for earth observation. In Proceedings ofthe 33rd ACM International Conference on Multimedia, pages 3027–3036, 2025.

[5] Yijie Zheng, Weijie Wu, Qingyun Li, Xuehui Wang, Xu Zhou, Aiai Ren, Jun Shen, Long Zhao, Guoqing Li, and Xue Yang. Instructsam: A training-free framework for instruction-oriented remote sensing object recognition. In Advances in Neural Information Processing Systems, volume 38, pages 110763–110799, 2026.

[6] Benedikt Blumenstiel, Paolo Fraccaro, Valerio Marsocci, Johannes Jakubik, Stefano Maurogiovanni, Mikolaj Czerkawski, Rocco Sedona, Gabriele Cavallaro, Thomas Brunschwiler, Juan Bernabe Moreno, et al. Terramesh: A planetary mosaic of multimodal earth observation data. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2394–2402, 2025.

[7] Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025.

[8] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, pages 8748–8763. PMLR, 2021.

[9] Seokju Cho, Heeseong Shin, Sunghwan Hong, Anurag Arnab, Paul Hongsuck Seo, and Seungryong Kim. Cat-seg: Cost aggregation for open-vocabulary semantic segmentation. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4113–4123. IEEE, 2024.

[10] Mengde Xu, Zheng Zhang, Fangyun Wei, Han Hu, and Xiang Bai. Side adapter network for open-vocabulary semantic segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2945–2954, 2023.

[11] Clive Tinashe Marimo, Benedikt Blumenstiel, Maximilian Nitsche, Johannes Jakubik, and Thomas Brunschwiler. Beyond the visible: Multispectral vision-language learning for earth observation. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pages 359–375. Springer, 2025.

[12] Kaiyu Li, Ruixun Liu, Xiangyong Cao, Xueru Bai, Feng Zhou, Deyu Meng, and Zhi Wang. Segearth-ov: Towards training-free open-vocabulary segmentation for remote sensing images. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10545–10556. IEEE, 2025.

[13] Ryan Faulkenberry and Saurabh Prasad. Dino soars: Dinov3 for open-vocabulary semantic segmentation of remote sensing imagery. arXiv preprint arXiv:2605.03175, 2026.

[14] Jose Sosa, Danila Rukhovich, Anis Kacem, and Djamila Aouada. Enabling training-free text-based remote sensing segmentation. arXiv preprint arXiv:2602.17799, 2026.

[15] Chengyang Ye, Yunzhi Zhuge, and Pingping Zhang. Towards open-vocabulary remote sensing image semantic segmentation. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 39, pages 9436–9444, 2025.

[16] Saikat Dutta, Akhil Vasim, Siddhant Gole, Hamid Rezatofighi, and Biplab Banerjee. Aeroseg: Harnessing sam for open-vocabulary segmentation in remote sensing images. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 2279–2289, 2025.

[17] Yimin Wei, Aoran Xiao, Hongruixuan Chen, Junshi Xia, and Naoto Yokoya. Mm-ovseg: Multimodal optical-sar fusion for open-vocabulary segmentation in remote sensing. arXiv preprint arXiv:2603.17528, 2026.

[18] Yan Shu, Bin Ren, Zhitong Xiong, Danda Pani Paudel, Luc Van Gool, Begüm Demir, Nicu Sebe, and Paolo Rota. Earthmind: Leveraging cross-sensor data for advanced earth observation interpretation with a unified multimodal llm. arXiv preprint arXiv:2506.01667, 2025.

[19] Aviad Barzilai, Yotam Gigi, Amr Helmy, Vered Silverman, Yehonathan Refael, Bolous Jaber, Tomer Shekel, George Leifman, and Genady Beryozkin. A recipe for improving remote sensing vlm zero shot generalization. arXiv preprint arXiv:2503.08722, 2025.

[20] Benedikt Blumenstiel, Nassim Ait Ali Braham, Conrad M Albrecht, Stefano Maurogiovanni, and Paolo Fraccaro. Ssl4eo-s12 v1.1: A multimodal, multiseasonal dataset for pretraining, updated. In IGARSS 2026-2026 IEEE International Geoscience and Remote Sensing Symposium, pages 1964–1968. IEEE, 2026.

[21] Alistair Francis and Mikolaj Czerkawski. Major tom: Expandable datasets for earth observation. In IGARSS 2024-2024 IEEE International Geoscience and Remote Sensing Symposium, pages 2935–2940. IEEE, 2024.

[22] Mistral AI. Mistral large 3 675b instruct. https://huggingface.co/mistralai/ Mistral-Large-3-675B-Instruct-2512, 2025.

[23] Johannes Jakubik, Felix Yang, Benedikt Blumenstiel, Erik Scheurer, Rocco Sedona, Stefano Maurogiovanni, Jente Bosmans, Nikolaos Dionelis, Valerio Marsocci, Niklas Kopp, et al. Terramind: Large-scale generative multimodality for earth observation. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 7383–7394, 2025.

[24] Tao Wang, Changxu Cheng, Lingfeng Wang, Senda Chen, and Wuyue Zhao. Himtok: Learning hierarchical mask tokens for image segmentation with large multimodal model. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 23267–23278, 2025.

[25] Kai Norman Clasen, Leonard Hackel, Tom Burgert, Gencer Sumbul, Begüm Demir, and Volker Markl. reben: Refined bigearthnet dataset for remote sensing image analysis. In IGARSS 2025-2025 IEEE International Geoscience and Remote Sensing Symposium, pages 1264–1268. IEEE, 2025.

[26] Nora Gourmelon, Thorsten Seehaus, Julian Klink, Matthias Braun, Andreas Maier, and Vincent Christlein. Caffe-a benchmark dataset for glacier calving front extraction from synthetic aperture radar imagery. In IGARSS 2023-2023 IEEE International Geoscience and Remote Sensing Symposium, pages 896–898. IEEE, 2023.

[27] Francesc Marti Escofet, Benedikt Blumenstiel, Linus Scheibenreif, Paolo Fraccaro, and Konrad Schindler. Fine-tune smarter, not harder: Parameter-efficient fine-tuning for geospatial foundation models. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pages 516–532. Springer, 2025.

# TerraMesh-Masks Supplementary Material

The supplementary material is organized as follows:

• We provide an overview of the TerraMesh source dataset in Section A.

• More information on the dataset and the captioning pipeline is described in Section B.

• Training details are outlined in Section C.

• Lastly, we provide additional quantitative and qualitative results in Section D.

## A TerraMesh

TerraMesh [6] is a large-scale, analysis-ready multimodal EO dataset comprising over 9 million co-registered samples at 10 m resolution, with a patch size of 264 × 264 pixels, designed explicitly for scalable pretraining workflows. Each sample contains a consistent set of spatially aligned modalities, including Sentinel-2 optical imagery (L1C top-of-atmosphere, L2A surface reflectance, and derived RGB), Sentinel-1 SAR backscatter (GRD or RTC depending on the subset), NDVI, Copernicus DEM, and ESRI land-use/land-cover annotations, all harmonized to a common grid as shown in Fig. 9. The dataset provides global coverage across diverse ecosystems and ecoregions as visualized in Fig. 10. TerraMesh is constructed by combining near-global single-timestamp data from MajorTOM-Core [21] with multi-season urban-centric patches from SSL4EO-S12 v1.1 [20]. Homogeneous regions (e.g., deserts) in MajorTOM are explicitly downsampled. Temporally, acquisitions span eight years (primarily 2019–2023) with S1 and S2 images largely aligned within one week.

![](images/d97aeaa99e68ccf7f0a9fa3b2bdf7086670d59f3dbd887da945e14418469c326.jpg)  
Figure 9: Modality examples in TerraMesh. We exclude NDVI and LULC from training and from the Eval subset as these modalities are related to specific semantic classes and might bias OVS models.

For practical usage, TerraMesh is distributed as shuffled WebDataset tar archives (17 TB) containing compressed Zarr samples with colocated metadata (e.g., geolocation and timestamps), enabling efficient streaming and data loading on large-scale compute systems. TerraMesh-Masks uses a similar data structure and natively integrates with the WebDataset structure of TerraMesh as an additional modality. Our dataset requires an additional 1.2 TB of storage and we adapted the data loading script to image-text pairs.

## B Dataset details

Evaluation set. Fig. 11 shows where the 523 evaluation samples of TerraMesh-Masks-Eval are located. They span 83 countries, and coverage is densest in North America (126 samples) and Europe (101 samples), while Africa contributes 23 samples in total. This mirrors the mapping density of

![](images/6782fa5f5c88a5fbc06970758548928de91e3c95ee2747474e35f351a7e03944.jpg)  
(a) Spatial distribution

![](images/68a515f15da0a941bb58ed3aedf775f24294a2408b98937a6d9ea0c017a0ce82.jpg)  
(b) Temporal distribution

Figure 10: Spatiotemporal distribution of the TerraMesh source dataset with (a) sample count visualized on a one degree map, and (b) monthly temporal distribution of its subsets using the S2 timestamps.  
![](images/bf7d9ae9f23236607f519820527e8994046e07bd1668229b2cd789bfd16376c1.jpg)  
Figure 11: Geographic distribution of the 523 samples in TerraMesh-Masks-Eval, coloured by their source dataset.

OpenStreetMap rather than a sampling choice on our side. Only 13 samples lie above 60° latitude, since TerraMesh excludes the polar regions. We provide further examples drawn from the Eval subset in Fig. 12 to demonstrate the diversity of semantic concepts. An overview of the groups is provided in Tab. 4 with sample counts and labels.

Despite the filtering processes outlined in the following subsections, TerraMesh-Masks still contains several noisy labels as shown in Fig. 13. We, therefore, created and released the evaluation set based on human reviews. In Fig. 14 we show the information that was provided to the reviewers. If the information was not sufficient to accurately label a mask as valid or invalid, reviewers could always label them as unsure.

Table 4: Overview of the assigned label groups based on manual review and grouping of 200+ class labels in TerraMesh-Masks-Eval. We display the most common class names of each group.
<table><tr><td>Id</td><td>Group name</td><td>Sample count</td><td>Unique labels</td><td>Examples</td></tr><tr><td>1</td><td>Inland water</td><td>63</td><td>14</td><td>reservoir, river, stream, canal, lake</td></tr><tr><td>2</td><td>Coastal water</td><td>45</td><td>12</td><td>coastal bay, ocean, bay, sea, coral reef</td></tr><tr><td>3</td><td>Coastal land</td><td>39</td><td>16</td><td>islet, coastal island, island, peninsula</td></tr><tr><td>4</td><td>Forest &amp; wetland</td><td>30</td><td>9</td><td>forest, mangrove forest, wetland, mangrove</td></tr><tr><td>5</td><td>Grass &amp; shrubland</td><td>12</td><td>8</td><td>scrubland, meadow, shrubland, hedge</td></tr><tr><td>6</td><td>Agriculture</td><td>24</td><td>10</td><td>orchard, farmyard, vineyard, agricultural field</td></tr><tr><td>7</td><td>Barren terrain</td><td>40</td><td>15</td><td>desert, cliff, glacier, sand desert, bare rock</td></tr><tr><td>8</td><td>Airport</td><td>22</td><td>9</td><td>airport apron, airport runway, military airfield</td></tr><tr><td>9</td><td>Transport</td><td>31</td><td>15</td><td>bridge, transit platform, parking lot, railway</td></tr><tr><td>10</td><td>Utilities</td><td>32</td><td>17</td><td>dam, pipeline, power plant, power substation</td></tr><tr><td>11</td><td>Industry</td><td>34</td><td>10</td><td>quarry, industrial facility, landfill, industrial area</td></tr><tr><td>12</td><td>Settlements</td><td>32</td><td>13</td><td>residential area, rural hamlet, rural village</td></tr><tr><td>13</td><td>Civic &amp; institutional</td><td>51</td><td>25</td><td>cemetery, school, university campus, prison</td></tr><tr><td>14</td><td>Recreation</td><td>68</td><td>34</td><td>sports pitch, urban park, golf course, marina</td></tr></table>

![](images/c453c3686c0e6c89942df6548ed86bf6afaa3cdb1aa216df9ced20df9b18d50d.jpg)

Urban park   
©France   
City park, public   
green space, urban   
recreational area,   
small urban parkland   
Railway corridor   
© Russia   
Railway track, train   
transportation line,   
paved rail route,   
linear rail   
Coral reef   
Canada   
Reef, shallow coral   
formation, marine   
reef structure,   
coastal reef system

![](images/f6224103e9b3ab656dc1a2aa8f7b75389d2346ab607bf5f8bf1d472312592064.jpg)

![](images/96c3b1401824993baa2104fc4897bb116808a7069622f8fec1973a5d1cfdd6b6.jpg)

![](images/f8bdf8beb2471e2d9741d4cdd372f75c8322eb29b03d6a4e996c94f80132bd0d.jpg)

Hospital complex Austria Medical facility, large healthcare infrastructure, hospital campus

![](images/40e239ef309fe4c27c4146483ac13b9ccff90dc245cd8ce0b66946c0ff5e226a.jpg)

Pond   
© Colombia   
Small water body   
inland water   
reservoir, shallow   
water pond   
Cemetery   
Malaysia   
Burial ground,   
graveyard, compact   
memorial park, urban   
cemetery plot

![](images/f08fc147ed8201a32b6e7ced99f7792f609f230f8bd9c53b0ed2b4b6380eaa0c.jpg)

![](images/a4ca6f69147faf274529405cd751d7153b4fb5a57dc13beff62ccc5423861e1e.jpg)

Bare rock outcrop © Argentina Rocky exposure exposed bedrock, narrow bare rock formation

![](images/9c0bbf5db628f62c44e3af7a94c8f97dfa5fffd1eca95f377f27bebe78201986.jpg)

![](images/eb83ca2ec42b554e5625f4441743711f4272ffe8cc421bf4f3d6ae92123519d8.jpg)

Inland water body © Canada Lake, freshwater lake. small inland lake, shallow water body

![](images/34a2db75ffa46af3f809eda0621fcfeb906d130b18ed18f7ba93d3410620b604.jpg)

![](images/31a66f66e0252a74d1f5d10a32debc6bcc454beefb764a4057d2ff929b3102fe.jpg)

Salt pond   
© India   
Salt evaporation pond   
brine pond, coastal   
salt pan, human-made   
saline water body   
Peninsula   
Canada   
Coastal land   
projection, narrow   
land spit, promontory,   
jutting landmass

Baseball stadium © United States Sports stadium baseball field. athletic complex, large outdoor sports

![](images/dbadacc831f5706681101c64726025172f980c307ac989bdd9bb0ac1c53293a5.jpg)

Canal   
© Egypt   
Irrigation canal,   
narrow water channel   
artificial waterway,   
agricultural canal

![](images/50d0aedf407fb48dd03bebfbaa14a6bd83636b5bf0b62bb63e97e3c1dfde9870.jpg)

![](images/333c6ea77cfa7941711ed18176c2a4fc14cc5d55e21978a50b266cdd1ecb8a03.jpg)

![](images/5425f02c1f93034805c089c00966eeff087bb08b28e90cc169058f37b58a93ae.jpg)

![](images/319972a2811e3dd5da9e13f3a0e1746c377724ed6790e53dd72a1c94c56c62b8.jpg)

University campus © Japan Higher education institution, academic campus, large educational complex

Airport runway   
Italy   
Airstrip, aircraft   
landing strip, paved   
runway, narrow airport   
tarmac   
Tropical forest   
Malaysia   
Dense woodland   
evergreen forest   
cover, tropical   
rainforest   
Farmyarc   
© United Kingdom   
Agricultural yard,   
farm outbuilding area,   
small farmstead   
enclosure Military airport   
© Brunei   
Military airfield,   
defense airstrip.   
armed forces runway complex

![](images/b3d0ad62ecf5845e309c2e03332eaed793013453552461c12ae4a82238ed5725.jpg)

![](images/ebca02b2097a2b530443db5fb009ab24f7efd5f15ec3952a533fe7d6b6bbb0f0.jpg)

![](images/133b70682b970ad6ac45e66079835887260cf2eecb585b82822f58bb2fc46c10.jpg)

![](images/6203b2f4067eaa1c4a5b9f707febd8b51408d1552cc1ac4496dc93aa167d1387.jpg)

![](images/e4a55f4ec7574ff46a004ed103a4b5f6ddcb722ee79c65cc8e5ff4c111026853.jpg)

Rural village © Ivory Coast Small settlement rural habitation cluster, village clearing

Dam   
© China   
Water retention   
structure,   
hydroelectric dam,   
reservoir barrier   
Forest   
© Ukraine   
Woodland, dense tree   
cover, temperate   
forest, forested area

![](images/8608e396d564cf51205a397ca96bec3153f833965428fd3597b1dbce9f16bfec.jpg)

Golf course green © Japan Golf green, putting green, small turf area on golf course, golf course putting surface

Islet   
© Brazil   
Small island, isolated   
landmass, riverine   
islet, compact insular   
feature   
Landfill   
© Chile   
Waste disposal site,   
municipal solid waste   
dump, large landfill   
area

![](images/1c116f1ee6c35bc7d72b4a02a7b307c4db2e9fd5929fe1e56979aa3a82abbb54.jpg)

![](images/b2afd8450523d75c78989fe00cde45f1225756040e967ca72a408405f53a62d6.jpg)

![](images/e356b18671270d6d528b56501997f89a1ce2b64c2425edaac12e2c73bcf70a91.jpg)

![](images/0a72cd443fdbc7a9ac2ab5ddba870c0175fa70ec287bc048f53009943e5157c4.jpg)

Wastewater canal © Uzbekistan Sewage channel, wastewater drainage ditch, narrow sewage waterway

Winery building © Portugal Agricultural facility small winemaking structure, vineyard processing building

![](images/89a4078fe1736b5fb31bfcbede41427901db0fe2e283ac9abe0e0597f3945688.jpg)

![](images/eaa351003eacb613cbe75430c62139d9d8014c19e39ef65ad40474d8813868ca.jpg)

Grandstand   
© Mexico   
Sports grandstand,   
stadium seating   
structure, large   
spectator stand   
Mangrove forest   
© Cambodia   
Coastal mangrove,   
dense mangrove cover,   
tidal mangrove   
woodland Construction site   
© Iran   
Active construction area, earthwork site, building development zone

![](images/d226cabab17ec86307ab9d48ed13d647d66963224a23b508bc05ec2a4498198b.jpg)

![](images/9f3ed5992732fbd795a75c74d531de87b3d3fd9114b17748919469ffe9bc9491.jpg)

Railway platform Greece Train platform, railway station platform, transit boarding area

Bridge   
Iran   
Road bridge,   
transportation bridge,   
narrow river crossing   
elevated roadway   
Pipeline   
©China   
Utility pipeline,   
narrow linear   
infrastructure   
underground pipeline   
Glacier   
© Peru   
Ice field, permanent   
ice cover, mountain   
glacier, high-altitude   
ice mass

![](images/1858a5efac75e426fe4fd1fa5130ad5d48205d34f65c03723f4a82be1fcea3e0.jpg)

![](images/9f9f4a89f28428d7bad55fbbd170c18d912510ad586f6d68d9fd546bb7c9f41f.jpg)

![](images/0fb50e72425782bb3e2f4f6334efd03ba74c16277a6b0c3e0c573bf0936ffa2f.jpg)

![](images/205b8340f93c91d1f8de4c5f1ad11c4d758f4ec8947395cc1f0dd81c98f837ac.jpg)

Figure 12: Additional examples from the TerraMesh-Masks-Eval set. Each patch is shown with a polygon mask overlay and its open-vocabulary class label, country, and a synonym set.

Quarry   
United States   
Open-pit mine,   
extraction site,   
mineral excavation   
area

![](images/6ff9aad1a19bbd7a85caf21dac04dd6071b0825e373e7224471d1d4f24c3f0c9.jpg)

Industrial area © China Industrial complex manufacturing zone, factory district, large industrial

Mountain ridge   
©Tajikistan   
Ridge, narrow elevate   
ridge, linear mountain   
crest, topographic   
ridge line River   
©Argentina   
Stream, waterway, narrow water channel natural river course

![](images/4efe69b5ada6261eade59206fe6fe075763636fec17d928d8d0f501fbc454d06.jpg)

![](images/ab02ab607de59c8e2859e0d700bc4f897ae6fd7b3beead82d8caaaa9d2801ccc.jpg)  
Figure 13: Wrong or challenging annotations. The dataset contains several noisy labels, while water-related classes are often very precise.

![](images/d59959a12099ae7d4d7a72958adc6954417faf7860714c5be0ec775f9deb028a.jpg)  
Figure 14: Example of the provided information to the reviewers when creating the TerraMesh-Masks-Eval dataset. Based on this information, reviewers rated the image-mask-text combination as valid, invalid, or unsure.

## B.1 LLM prompting

We selected the annotation model based on a qualitative experiment. We ran a set of candidate multimodal LLMs on over 100 identical candidate masks and inspected their outputs, judging whether the assigned class label matched the polygon, whether the synonyms were usable as text queries, and whether the model rejected masks that were genuinely uninterpretable at 10 m. Mistral-Large-3 was the most consistent and we used it for the full corpus.

We prompt the model as a conservative remote-sensing annotator whose goal is to decide whether a candidate polygon provides useful supervision at Sentinel-2 resolution and, if so, to provide a class label and synonyms based on weak geospatial metadata. The prompt explicitly discourages hallucination, strongly penalizes ambiguity, and rejects administrative or legally defined concepts that are not visually grounded (e.g., legal borders).

Each prompt combines (i) a rendered Sentinel-2 RGB patch with the candidate polygon overlaid in a purple color, and (ii) structured metadata including geographic context (country/continent), a flag indicating whether the polygon covers the full image, the candidate subtypes tags from Overture or OpenStreetMap, and the polygon area as a percentage of the patch. The prompt instructs the model to treat subtypes as the primary signal for concept definition while using the image mainly to detect clear contradictions and to assess whether the mask is interpretable at the target resolution. Additionally, we add varing example outputs, counter examples, and specific instructions for the main failure cases from our visual review.

We constrain the model to generate JSON only with a fixed output schema. Following a the chain-ofthought approach, the model first returns a short justification, followed by quality scores in [0, 1], and language annotations:

```jsonl
{
"quality_reason": "...",
"sample_quality_mask": 0.0,
"class_label": "...",
"class_synonyms": ["...", "..."]
}
```

Every mask with a quality score larger than 0.5 is accepted and included in the dataset. We share the scores with the dataset, in case researchers want to use a more conservative threshold. The scores are not continously distributed, so we treat them as a bernoulli trial instead. The prompt further enforces output-length limits (e.g., short labels and short synonym phrases) and requires empty class\_label and class\_synonyms whenever the sample is rejected (e.g., both suitability scores are very low). class\_synonyms is constrained to a small set (2–4 items) describing the same physical concept in diverse noun phrases. The full prompt is shared in our code repository https://github.com/IBM/TerraMesh-Masks.

## B.2 Quality filtering

We use several heuristics to filter out unsuited or incorrect source tags and increase the diversity of selected tags. Specifically, we sample up to 3 masks per image using the following sampling probability scores. If the LLM discards a sampled mask, we sample further candidates if available.

Frequency-based scoring. The first heuristic filter leverages previously validated reference masks to estimate spatially aware quality scores. We use the hierarchical, hexagonal geospatial indexing system H3 with resolution 3 (≈110 km in diameter) to group the annotation candidates. For a candidate mask from subtype s in H3 cell h, we compute

$$
p _ { \mathrm { g e o } } ( s , h ) = \frac { \mu \hat { x } _ { \mathrm { g l o b a l } } + n _ { \mathrm { l o c a l } } \hat { x } _ { \mathrm { l o c a l } } } { \mu + n _ { \mathrm { l o c a l } } } ,
$$

where $\hat { x } _ { \mathrm { g l o b a l } }$ is the global score for subtype $s , \hat { x } _ { \mathrm { l o c a l } }$ is the local score computed from reference masks within cell $h ,$ and $n _ { \mathrm { l o c a l } }$ is the local observation count. The parameter $\mu$ controls the strength of the global prior and is set to 5 in our experiments.

We then combine the geographic prior and an area-based prior (using $\omega = 0 . 8 )$ to compute the sampling score for candidate s through

$$
p _ { \mathrm { f r e q } } ( s , h ) = \omega p _ { \mathrm { g e o } } ( s , h ) + ( 1 - \omega ) p _ { \mathrm { a r e a } } ( s ) ,
$$

where $p _ { \mathrm { a r e a } } ( s )$ is an area-based score for subtype s. Candidates with $p _ { \mathrm { g e o } } < 0 . 2$ are discarded: their score is set to $- \infty$ , ensuring that they are never selected for downstream annotation. An exploration bonus is applied to rare subtypes so that underrepresented categories are not eliminated too early.

Spectral quality scoring. The second heuristic filter quantifies the spectral coherence of a candidate mask using the Normalized Difference Vegetation Index (NDVI), calculated using the S2 sample. The approach compares the NDVI statistics inside and outside the polygon to identify potentially incorrect annotations. We distinguish three cases: (1) For small masks covering less than 0.1% of the image, the score is set to 1 by convention, because spectral discrimination is not meaningful at this small scale; (2) Masks covering almost the entire patch with more than 85% are scored by the normalized Shannon entropy of the NDVI histogram inside the mask, as inside-vs-outside contrast is uninformative; and (3) for masks between 0.1% and 85% coverage, we measure spectral separability between inside and outside pixels using

$$
p _ { \mathrm { s p e c } } = 1 - \frac { a b s ( \mu _ { \mathrm { i n } } - \mu _ { \mathrm { o u t } } ) } { \sqrt { \sigma _ { \mathrm { i n } } ^ { 2 } + \sigma _ { \mathrm { o u t } } ^ { 2 } + \varepsilon } } ,
$$

where $\mu _ { \mathrm { i n } } , \mu _ { \mathrm { o u t } } , \sigma _ { \mathrm { i n } } .$ , and $\sigma _ { \mathrm { o u t } }$ are the mean NDVI values and standard deviations inside and outside the mask, respectively. The assumption is that many correct annotations are uniform within the mask while having different scores to classes outside the mask. To account for classes whose spectral signatures do not correctly indicate invalid masks (e.g.,farmland or airport), a manually curated list of classes is downweighted before class selection.

Class-frequency downsampling. In addition to spatial filtering, the pipeline applies a global correction to downsample dominant subtypes like forest or grassland. Let $\pi _ { s }$ denote the number of already selected masks of subtype $s ,$ and let c be a fixed constant (here $c = 0 . 0 2 )$ . We define

$$
p _ { \mathrm { c l a s s } } = \operatorname* { m i n } \biggl ( 1 , \frac { c } { \operatorname* { m a x } ( \pi _ { s } , \varepsilon ) } \biggr ) ,
$$

and use it to reweight the overall score. This regularizer does not force a perfectly uniform distribution, but it reduces sampling heavily overrepresented classes.

Final heuristic score. The final sampling score p is computed as follows

$$
p = p _ { \mathrm { f r e q } } \cdot p _ { \mathrm { s p e c } } \cdot p _ { \mathrm { c l a s s } } .
$$

A hard gate is applied before the final multiplication: if $p _ { \mathrm { g e o } } \ < \ 0 . 2$ , the sample’s probability is discarded. Conceptually, this score is designed to behave as a filter rather than a final label. The geographic prior removes subtype–region combinations that are consistently implausible, the spectral term suppresses masks that are likely misaligned with the actual image, and the classfrequency regularizer prevents single categories form dominating in the overall dataset. The result is a filtered candidate pool that is much smaller, substantially cleaner, and more diverse to support open-vocabulary segmentation at scale.

## C Training details

Multimodal and (S2) models. The two model families differ only in which input modalities are available during pretraining, not in architecture, schedule, or training samples. The multimodal models receive Sentinel-2 L2A, Sentinel-2 RGB, Sentinel-1 GRD, Sentinel-1 RTC, and DEM as input domains. The (S2) models receive only Sentinel-2 L2A and Sentinel-2 RGB. During training, we sample 392 input tokens for the multimodel models, which represents two full input modalities, and 196 tokens for the (S2) models. Both versions predict the same target domain of binary-mask tokens and use the same text query as additional input tokens. We do not sample the target tokens and always use all 196 tokens.

NDVI-basedfiltering during pretraining. Beyond the filtering applied when constructing the corpus, we apply a second filter in the pretraining pipeline. We keep only masks whose NDVI agreement score is at least 0.4, which removes roughly 23% of the released training masks. This threshold is deliberately stricter than what we use for the released dataset, where we keep every mask above the LLM quality score of 0.5 and publish the agreement scores so that others can choose their own operating point. Filtering at pretraining time rather than at release keeps the dataset broad while giving the model a cleaner training signal.

![](images/eaa0d2693e7023f65aa8263b30a9ff10e650ce0494e14a934e4452ae678da437.jpg)  
Figure 15: Label-wise mIoU results of all 207 labels in the eval set, sorted per model.

Class-label reweighting. The distribution over class labels in TerraMesh-Masks is long tailed, since common concepts such as forest or residential area appear far more often than funicular or coral reef. We counteract this with stochastic repetition at the sample level. Each class label c is assigned a repeat factor $\lambda _ { c } ,$ which represents the expected repetition per epoch of a sample with this label. Factors range from 0.1 to a cap of 500, so samples of the most frequent labels are often dropped while the rarest are repeated heavily. This flattens the label distribution in both directions, which matters for open-vocabulary segmentation where the tail of the label distribution is what the model is expected to generalize over.

## D Additional results

Class-wise results. Beyond reporting grouped results, we provide per-class mIoU scores in Fig. 15. The sample performances are grouped by their class labels and then averaged. However, due to the diversity of labels, many scores are based on a single sample. Similar to the aggregated scores, SAM-3 shows the worst performance due to many missing predictions. Many false positive predictions from SAM-3 and DINOv3.txt contribute to a generally lower mIoU. GeoPixel performs quite well on smaller objects which is highlighted by the label examples. These classes are well covered in the GeoPixel training dataset [3]. In contrast, TerraMind-OVS-L performs much better on land-cover-related classes.

Qualitative results. We provide additional predictions in Fig. 16. The samples are selected representatively to highlight different strengths and weaknesses of the models. For example, TerraMind-OVS is very good at detecting land airport related tasks, while failing on small rivers and streams.

## D.1 Per-category performance

Using a grouping of the label space into 14 coarse semantic groups, we observe a clear structure in model behavior (Tab. 5). Across model sizes, broad land-cover and vegetation classes are generally segmented more reliably than thin, fragmented, or highly structured targets. This pattern is consistent with the visual character of the task and with the spatial resolution of Sentinel-2: extended regions with coherent texture are substantially easier than narrow infrastructure elements or boundaries with limited visual support.

Table 5 includes a per-group comparison of TerraMind-OVS-L using RGB and Sentinel-2 inputs. We see large differences between the multimodal and S2 version depending on the group. Within each version, the multispectral inputs slightly improve the performance in most groups, but only within the confidence intervals. In contrast, other results from literature showed significant improvements after

Ground truth  
SAM-3  
RemoteSAM  
DINOv3.txt  
DINO Soars  
SegEarth-OV  
GeoPixel  
TerraMind-OVS-L  
![](images/d02dae375da090ad6b44ccee19b3a0e2d497bc1342eda7a834cb098edff52f64.jpg)  
Figure 16: Qualitative comparison on examples from TerraMesh-Masks-Eval. SAM- and DINOv3- based models often lead towards overprediction or no prediction. TerraMind-OVS performs well on larger land-cover related labels, while GeoPixel is much better in spotting objects.

Table 5: IoU results on TerraMesh-Masks-Eval per group. All models are using RGB inputs, except for the last two rows in each block, which show TerraMind-OVS-L with S2 L2A inputs.
<table><tr><td></td><td>Model</td><td>Inla tr</td><td>Coatl waster</td><td>Coaa Castnd</td><td>oree  rwve.</td><td>Gras s u.</td><td>Aruulie</td><td>Barn errain</td><td>Airpot</td><td>Tranpon</td><td>Uiies</td><td>Indutry</td><td>Setets</td><td>Cic&amp; nst</td><td>Reeaton</td><td>Mean</td></tr><tr><td>SAM-3</td><td></td><td>20.49</td><td>43.58</td><td>22.77</td><td>3.48</td><td>5.84</td><td>5.26</td><td>12.91</td><td>4.66</td><td>0.00</td><td>0.14</td><td>10.90</td><td>11.97</td><td>1.08</td><td>0.77</td><td>10.28</td></tr><tr><td rowspan="15">Ilodos</td><td>RemoteSAM</td><td>6.24</td><td>19.98</td><td>27.29</td><td>32.51</td><td>28.95</td><td>17.89</td><td>36.96</td><td>13.92</td><td>5.04</td><td>4.51</td><td>9.71</td><td>13.76</td><td>2.48</td><td>9.50</td><td>16.34</td></tr><tr><td>DINOv3.txt</td><td>11.22</td><td>42.05</td><td>23.41</td><td>26.12</td><td>8.32</td><td>17.03</td><td>21.80</td><td>12.58</td><td>4.57</td><td>5.09</td><td>19.76</td><td>26.55</td><td>7.90</td><td>4.96</td><td>16.53</td></tr><tr><td></td><td>11.79</td><td>58.89</td><td>21.89</td><td>39.29</td><td>16.10</td><td>13.99</td><td>42.61</td><td>16.97</td><td>1.52</td><td>5.90</td><td>20.56</td><td>26.47</td><td>3.54</td><td>2.96</td><td>20.18</td></tr><tr><td>DINO Soars</td><td></td><td>57.85</td><td>23.32</td><td>39.14</td><td>18.62</td><td>18.89</td><td>46.84</td><td>10.99</td><td>3.89</td><td>5.76</td><td>18.36</td><td>33.13</td><td>7.20</td><td>5.87</td><td></td></tr><tr><td>SegEarth-OV</td><td>16.82</td><td>19.65</td><td>19.51</td><td>20.01</td><td>7.26</td><td>16.59</td><td>13.76</td><td>19.13</td><td>16.56</td><td>4.47</td><td>25.32</td><td>20.26</td><td>12.46</td><td>11.26</td><td>21.91</td></tr><tr><td>GeoPixel</td><td>9.60</td><td>83.82</td><td>50.13</td><td>47.12</td><td>13.65</td><td>25.20</td><td>44.18</td><td>16.02</td><td>2.46</td><td>3.09</td><td>13.87</td><td>41.02</td><td>1.71</td><td>3.61</td><td>15.42</td></tr><tr><td>TerraMind-OVS-S</td><td>24.63 27.07</td><td>84.62</td><td>49.39</td><td>45.67</td><td>19.53</td><td>24.07</td><td>42.99</td><td>16.40</td><td>3.47</td><td>3.13</td><td>19.66</td><td>42.84</td><td>1.26</td><td>4.43</td><td>26.46 27.47</td></tr><tr><td>TerraMind-OVS-S (S2)</td><td>29.52</td><td>82.90</td><td>59.67</td><td>53.67</td><td>21.09</td><td>35.18</td><td>52.76</td><td>24.97</td><td>3.53</td><td>5.38</td><td>26.19</td><td>58.04</td><td>3.32</td><td>5.88</td><td>33.01</td></tr><tr><td>TerraMind-OVS-B</td><td>31.87</td><td>83.02</td><td>58.31</td><td>50.40</td><td>21.85</td><td>35.02</td><td>43.75</td><td>23.17</td><td>6.73</td><td>8.06</td><td>26.67</td><td>56.77</td><td>8.06</td><td>4.96</td><td>32.76</td></tr><tr><td>TerraMind-OVS-B (S2) TerraMind-OVS-L</td><td>31.85</td><td>74.20</td><td>64.08</td><td>50.77</td><td>18.65</td><td>41.87</td><td>44.23</td><td>34.95</td><td>8.34</td><td>14.91</td><td>45.80</td><td>50.96</td><td>11.51</td><td>8.71</td><td>35.77</td></tr><tr><td>TerraMind-OVS-L (S2)</td><td>29.55</td><td>82.56</td><td>62.96</td><td>50.96</td><td>24.08</td><td>34.80</td><td>52.56</td><td>29.00</td><td>11.27</td><td>11.85</td><td>33.40</td><td>61.66</td><td>10.11</td><td>7.46</td><td>35.87</td></tr><tr><td>TM-OVS-L, S2 input</td><td>32.01</td><td>74.07</td><td>63.84</td><td>50.91</td><td>20.23</td><td>42.57</td><td>45.30</td><td>35.26</td><td>8.39</td><td>14.93</td><td>45.89</td><td>51.59</td><td>11.79</td><td>8.61</td><td>36.10</td></tr><tr><td>TM-OVS-L (S2), $2 inp.</td><td>32.07</td><td>82.66</td><td>63.34</td><td>52.06</td><td>24.69</td><td>33.97</td><td>53.02</td><td>29.51</td><td>9.67</td><td>7.83</td><td>31.11</td><td>64.53</td><td>13.04</td><td>6.02</td><td>35.96</td></tr><tr><td>SAM-3 RemoteSAM DINOv3.txt</td><td>11.92</td><td>35.27</td><td>15.02</td><td>17.17</td><td>19.08</td><td>8.63</td><td>24.00</td><td>7.91</td><td>0.46</td><td>0.97</td><td>7.29</td><td>8.40</td><td>1.46</td><td>1.37</td><td>11.35</td></tr><tr><td rowspan="13">u TM-OVS-L, S2 input TM-OVS-L (S2), S2 inp.</td><td></td><td>29.98 22.02</td><td>42.03</td><td></td><td>39.45</td><td>30.71 24.91</td><td>31.38</td><td>32.37</td><td>34.30</td><td>38.42</td><td>26.30</td><td>17.80</td><td>18.29</td><td></td><td></td></tr><tr><td></td><td>39.44 35.64</td><td>31.04</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>37.99</td><td>30.42</td></tr><tr><td></td><td>43.05 46.63</td><td>20.69</td><td>45.05</td><td>31.35</td><td>45.24</td><td>30.60</td><td>30.47</td><td>45.97</td><td>43.96</td><td>51.03</td><td>51.76</td><td>47.44</td><td>44.17</td><td>40.94</td></tr><tr><td>DINO Soars</td><td></td><td></td><td>45.95</td><td>22.74</td><td>33.35</td><td>52.07</td><td>47.04</td><td>47.56</td><td>50.36</td><td>54.68</td><td>44.77</td><td>46.94</td><td>47.10</td><td>43.07</td></tr><tr><td>SegEarth-OV</td><td>49.92 53.58</td><td>36.78</td><td>44.40</td><td>30.79</td><td>42.48</td><td>39.19</td><td>49.32</td><td>49.65</td><td>51.13</td><td>52.96</td><td>59.39</td><td>52.26</td><td>49.92</td><td>47.27</td></tr><tr><td>GeoPixel</td><td>50.54 34.40</td><td>50.35</td><td>44.85</td><td>33.73</td><td>50.59</td><td>32.04</td><td>53.93</td><td>57.67</td><td>50.68</td><td>59.71</td><td>54.04</td><td>55.14</td><td>53.65</td><td>48.67</td></tr><tr><td>TerraMind-OVS-S</td><td>60.79 88.89</td><td></td><td>69.38</td><td>64.02</td><td>40.17 57.72</td><td>52.11</td><td>53.52</td><td>50.68</td><td>50.25</td><td>53.96</td><td>66.44</td><td>49.90</td><td>50.89</td><td>57.76</td></tr><tr><td>TerraMind-OVS-S (S2)</td><td>62.10</td><td>89.40</td><td>68.64</td><td>63.09</td><td>41.61</td><td>56.71</td><td>52.30 51.15</td><td>51.08</td><td>50.50</td><td>57.23</td><td>67.44</td><td>49.67</td><td>51.16</td><td>58.01</td></tr><tr><td>TerraMind-OVS-B</td><td>63.50</td><td>87.87 86.86</td><td>73.36</td><td>69.00</td><td>44.30</td><td>62.92</td><td>58.84</td><td>58.92</td><td>51.26</td><td>51.68 60.49</td><td></td><td>76.41 50.39</td><td>51.98</td><td>61.50</td></tr><tr><td>TerraMind-OVS-B (S2)</td><td>64.69</td><td>79.14</td><td>73.51 78.67</td><td>66.67</td><td>42.71</td><td>62.59</td><td>52.79</td><td>56.68</td><td>52.78</td><td>53.01 60.71</td><td></td><td>75.69 52.97</td><td>51.58</td><td>60.95</td></tr><tr><td>TerraMind-OVS-L</td><td>64.65 63.53</td><td>87.82</td><td>78.18</td><td>67.74 66.62</td><td>41.77 42.74</td><td>67.35 63.15</td><td>51.17 55.44</td><td>64.15 60.73</td><td>53.64 55.13</td><td>56.39 71.72 54.90</td><td>64.63</td><td>71.36 54.28</td><td>53.34</td><td>62.53 62.70</td></tr><tr><td>TerraMind-OVS-L (S2)</td><td>64.74</td><td>79.02</td><td>78.44</td><td>67.82</td><td>41.54</td><td>67.69</td><td>51.71</td><td>64.30 53.65</td><td>56.38</td><td></td><td>71.77 71.70</td><td>78.21</td><td>53.94 52.80</td></table>

## D.2 Effect of mask size

integrating multispectral inputs into a VLM [11]. This is likely due to the nature of many man-made objects and land-cover classes that are clearly visible in RGB.

Since our label space mixes extended land-cover regions with small individual objects, we relate performance to the size of the target mask in Fig. 17. We bin the evaluation samples into log-spaced quintiles of ground-truth mask area, roughly 105 samples per bin, and report Io $\boldsymbol { \mathrm { J } } _ { p o s }$ within each bin. We restrict the comparison to the EO-specific baselines. Mask area is the dominant factor for all methods. TerraMind-OVS-L (S2) improves from $6 . 6 \% \mathrm { I o U } _ { p o s }$ on the smallest bin, below $0 . 0 1 \mathrm { k m ^ { 2 } }$ , to 76.9% on the largest ones. SegEarth-OV, DINO Soars, and RemoteSAM follow the same monotone trend at a lower level. GeoPixel behaves differently and its best results are in the middle bins. It is the strongest model on the two smallest bins, at 7.6% and $1 5 . 8 \% \mathrm { I o U } _ { p o s }$ , consistent with its training data of small annotated objects in high-resolution imagery. It is the only model that degrades on the largest targets of land cover classes, falling to 13.5%. This confirms that the remaining headroom on our benchmark is concentrated on small and thin structures, which are the hardest to resolve at 10 m.

## D.3 Effect of strong distribution shifts

With the evaluation of multi-label datasets, we also investigate the effect of strong distribution shifts on model performance using the example of the CaFFe dataset. CaFFe provides data on glacier calving, derived from diverse SAR images represented as single channel grayscale images. We therefore expect no benefit in using SAR-specific input channels of TerraMind-OVS. Instead, for all models across the benchmark, we stack three identical grayscale images and feed this as RGB representation to the models. All tested models struggle to compensate for this strong distribution shift. In Fig. 18, we provide examples of the predictions of TerraMind-OVS-S on CaFFe test data. We observe significant artefacts and problems in identifying segments, which are further amplified with most other benchmarked models.

![](images/2e573f35635092b96cc4cf60be2621e4176b838b5f65fb0bf59c7c5145e7ff6a.jpg)  
Figure 17: $\mathrm { I o U } _ { p o s }$ against ground-truth mask area on TerraMesh-Masks-Eval for the EO-specific models, binned into log-spaced quintiles. Error bars show 95% confidence intervals.

Overall, we observe strong generalizability across diverse concepts, such as those in TerraMesh-Masks-Eval. However, we find that this generalizability is limited by very strong distribution shifts that the model is not suitably adapted to.

![](images/06719bfa3d2ffb76e859b8f595b8623ac653856d41469d9e2dad30f3465b887d.jpg)  
Figure 18: Predictions on CaFFe with TerraMind-OVS-S.