# Targeted Deep Regression - Initial Applications in Whole-Body Water-Fat MRI in UK Biobank

Filip Marttala<sup>a,b,∗</sup>, Taro Langner<sup>c</sup>, Johan Öfverstedt<sup>a</sup>, Nouman Ahmad<sup>a</sup>, Håkan Ahlström<sup>a,c</sup>, Robin Strand<sup>b</sup>, Joel Kullberg<sup>a,c</sup>

<sup>a</sup>Radiology, Department of Surgical Sciences, Uppsala University, 752 37, Uppsala, Sweden <sup>b</sup>Department of Information Technology, Uppsala University, 751 85, Uppsala, Sweden <sup>c</sup>Antaros Medical AB, BioVenture Hub, 431 53, Mölndal, Sweden

## Abstract

UK Biobank is a British clinical study which, in addition to large amounts of medical data collected, plans to include detailed Magnetic Resonance Imaging (MRI) examinations from 100,000 participants. The image data can be analyzed for example by automated analysis with neural networks for image segmentation or classification to expand the collection of image-derived phenotypes for medical research. In this work, we propose, present and evaluate a methodology that uses image-based regression to estimate numerical properties from the collected neck-to-knee water-fat MRI scans. It applies a ResNet50-style network to 2D multi-channel collages that represent targeted and studyspecific anatomical sub-regions from the MRI scan. The targeted sub-region sampling is achieved by propagating manually defined subregions from a reference scan to the other >30,000 analyzed scans using image registration. The resulting collages can be compact while retaining high-resolution, for example by providing multiple slices of kidney tissue for a kidney volume regression network. This approach also enables group-wise saliency analysis that highlight image and anatomical regions used by the network for the predictions. The method was applied for the regression endpoints of kidney volume, heart volume, and age. Predictive performance was found between $R ^ { 2 } = 0 . 7 3 1 { \ - - 0 . 9 8 4 }$ and the group-wise saliency maps were found to highlight medically relevant regions. We conclude that the method can be used to standardize and restrict the information presented to a deep regression analysis pipeline, increasing information density and opening up a wide range of future medical inference studies.

## 1. Introduction

A popular method of medical imaging is Magnetic Resonance Imaging (MRI) which uses magnetic fields and radio-waves to generate images with good soft tissue contrasts. Its non-invasive nature and the fact that it does not expose the scanned subjects to ionizing radiation makes it very useful both in clinical practice and research. Because of this, it is possible to conduct large-scale imaging studies on healthy individuals, such as UK Biobank [19, 13]. Further, MRI can produce 3D images with detailed anatomical information about organs, vessels and bone at high resolution. While large-scale collection of images is possible, however, large-scale analysis of said images is challenging. Manual analysis of MRI can require minutes or hours per image, making it slow or limited in scope to relatively small sample sizes.

Machine learning techniques can provide major benefits by automating some of the required processing steps for image analysis. In the UK Biobank, proposed use cases for automation via neural networks include semantic segmentation for organ measurements [2, 11, 14, 8] and quality control [3], but also image-based, deep regression [10]. The latter can directly estimate numerical endpoints from image data, enabling a range of diferent predictions [21, 5]. Whereas segmentation networks can perform pixel- or voxelwise predictions, image-wise predictions by regression networks typically require larger quantities of training data [10]. This is especially the case for volumetric MR images with millions of voxel values, each of which could afect the prediction. Although imagebased deep regression has been previously proposed for volumetric MRI [1, 16], images are often compressed or restrict to 2D.

One major downside of neural networks is that it is often not clear what features are relevant for a given prediction [18]. This may be considered problematic since a network prediction does not inherently provide any insights into the logic behind this inference. This may reduce confidence in the prediction as well as making it more dificult to catch faulty predictions. This problem can be partially solved by the use of saliency maps. Saliency maps are a method of highlighting image regions with high impact on the network prediction, typically based on backpropagation of gradients in the network output back to the pixel inputs [18]. This can result in a map highlighting which input pixels had a large impact on the prediction. Individual saliency maps might however be noisy and dificult to interpret. If a large number of saliency maps are averaged, however, it is possible to pick out general patterns of general relevant features of the images[12].

The main contribution of this work consists in combining image-based, deep regression and saliency analysis with a new type of input image format that represents volumetric MRI data in 2D collages showing anatomy that is targeted and specific to the given regression task. The method allows for manually defining specific regions of interest from a UK Biobank neck-to-knee MRI and automatically extracting the anatomically corresponding region in other images by leveraging image registration[6]. These regions are then used to generate a 2D multi-channel image and used as input for a ResNet50- style neural network. The approach was trained and evaluated on several endpoints for more than 30,000 UK Biobank neck-to-knee body MR images.

## 2. Methods

## 2.1. Overview of the targeted deep regression approach

An overview of the targeted deep regression approach is given in Figure 1. By use of image registrations manually selected target image slices and volumes is subsampled for all individuals and scans on a cohort study. The targeted image data is combined into a multi-channel 2D collage that is used as input to a regression network. Saliency analysis is used for interpretability. The image registrations furthermore enable group-wise or cohort saliency analysis.

![](images/bcb3c6b0ed859d075a776b2fa734b14df065e7b6d19aad6446ac9120d3db2a1a.jpg)  
Figure 1: Illustration of the image analysis pipeline used for the targeted deep regression, with associated cohort saliency analysis that enables interpretability. Template scans (one male and one female) from the 3D water-fat MRI images in UK biobank are first selected. The targeted subsampling of the image data is manually defined using these. Image landmarks, subvolumes and projection are selected and used to target specific anatomical regions of the UK Biobank images and to convert the 3D image data into a 2D representation (e.g. axial mean intensity projection of a subvolume centered on the heart). Deformable image registration is used to transfer the manually selected landmarks and subvolumes into all individual scans. This allows targeted subsampling of the 3D data into a 2D representation from all scans. The extracted multi-channel 2D images are combined into a multi-channel collage for every scan. These collages are used as input to a regression network that is trained to predict a reference value of interest from each individual collage (e.g. the age of a subject). The trained regression networks are furthermore analyzed by performing saliency analysis for each performed inference. This gives an individual saliency result, i.e. a heatmap, for each collage, of what regions in a collage important for the performed task. By use of the image registration results these individual saliency results can be combined and aggregated into a cohort saliency map that shows a cohort-wise result of what regions that are important for the predictions (e.g. what regions are found most important for the age prediction task).

## 2.2. Data sets

The MRI data and clinical data used in this study originates from UK Biobank, a large-scale study that collects health information on over half a million volunteers aged 44-82 years [19]. For 100,000 of these participants, further data collection is planned, including, but not limited to, whole-body MRI scans with neck-to-knee coverage. This study included over 40,000 of these whole-body MRI scans. The images were collected using a dual-echo Dixon protocol with a 1.5 T MR Siemens Aera device. These scans allow analysis of images of water signal, fat signal as well as quantitative water fraction as well as fat fraction images[22], with TR = 6.69ms, TE = 2.39/4.77ms and flip angle = 10°. The dimensions of each voxel are 2.23mm × 2.23 mm with the last dimension varying, being 3 mm, 3.5 mm, or 4.5 mm [13]. For the purposes of having consistent sizes, all images were resampled to an efective voxel resolution of 2.23mm × 2.23mm × 3mm $( \mathrm { S A G } \times \mathrm { C O R } \times \mathrm { A X } )$

The numerical values that served as ground truth for the regression targets of age, kidney volume, and heart volume were gathered from several diferent sources. For age, the UK Biobank metadata provided only the year and month of birth (UK Biobank fields 34 and 52). Every subject was therefore assumed to be born in the middle of the month, so that any error in the ground truth data amounts to at most half a month [12]. The values for kidney volume were based on prior work that utilized convolutional neural network for segmentation of the same neck-to-knee body MRI as used here [11]. The measurements of the heart consist of the left ventricular end-diastolic volume (LV-EDV) (UK Biobank field 22421) [20]. It is worth noting that the ground truth dataset for LV-EDV is noisy and UK Biobank includes a warning of non-quality controlled data for this parameter, with the original set of reference measurements containing values from a few 10s of mL to 15,000 mL (see UK Biobank field 22421). To limit the impact of these errors, the data was therefore filtered to exclude any participant with a value higher than 260 ml.

The UK Biobank study and the experiments performed here were approved by the British and Swedish ethics committees, respectively. The subject gave written informed consent, and the study has been performed according to the Declaration of Helsinki[19]. The UK Biobank data was accessed via the UK Biobank website<sup>1</sup> and their associated Access Management System<sup>2</sup>. Data is de-identified/pseudo-anonymized by the UK Biobank organization before it is shared.

![](images/a724df55887e533dba04fcb28c4042482e3e90c35a7fdc4932fcd88085e4949e.jpg)  
(a) Fat image

![](images/5d276a25780c21cffb4c020156d5c33cca423fda55425ab61d7d3a78931de836.jpg)  
(b) Water image

![](images/f0345e354678a8f80394ed6024d4864d3a4e460ace9b6076de6f15cf729ac334.jpg)  
(c) Water Fraction image  
Figure 2: An example of a heart collage illustrating the three image channels used (a, b, and c). In this example three projections are used in each channel. Axial (top left), coronal (bottom left), sagittal (bottom right) as well as one sagittal slice (top right) defined using a landmark in the (Aorta).

## 2.3. Targeted image collages

The input images provided to the neural network were formatted as targeted collages, i.e. the format was designed specifically for a given regression task, based on a research hypothesis. This was done by manually targeting specific regions of interest to be sampled, such that any anatomical area important for the measurement was in the targeted volumes, see Figure 1. A rectangular region of interest was defined and specified by its midpoint and side lengths in the male and female reference subjects respectively. Similar samples were then taken from the remaining images by using deformation fields to map the midpoint of the region of interest to its respective location in the non-reference images. For each image, a region of interest of the same size was then sampled around this new midpoint. Because the network requires the input to be a 2D image, sampled 3D regions were then projected by calculating the mean intensity value along a specified direction.

## 2.4. Image registration

Image registration was used to map manually defined regions of interest from a reference image to corresponding regions in all other images. The image registrations were created by selecting a reference image/subject from each sex and registering all other images to the reference image of the corresponding sex. The reference images were chosen to have an average tissue composition and were quality controlled for artifacts. The image registration used a deformable image registration method based on graph-cuts utilizing the software deform[6] that generated, for each image, deformation field mappings (vector fields) between a sex-specific reference and a given target image. Each deformation field describes how points in the reference images map to the target image and can be used to map manually defined coordinates in the reference persons to corresponding points in all other images.

## 2.5. Multi-channel collage construction

After sampling all desired regions and projecting them onto 2D, the resulting 2D images of targeted anatomical areas were arranged into one collage for each MR image, as shown in Figure 2. These collages were fed to the neural network. Minimizing the size of the collage is important for performance reasons and is a packing problem. This minimization was done using the rectangle-packer package for Python<sup>3</sup>. When a satisfactory packing was been achieved the water signal, fat signal as well as either water fraction or fat fraction images are combined into a 3 channel 256 × 256 image. For storage reasons the images were saved in 8-bit unsigned integer format. This was not found to have any noticeable negative impact on the predictive performance of the network. Because the intensity values range from 0 to about 1500, the intensity values were scaled down such as to fit in the 0-255 range of the uint8 format. For the fat and water image this was done by normalizing the image intensities after clipping the top 1% of the brightest voxel values and any negative values potentially resulting from the spline interpolation.

## 2.6. Network

A ResNet50 [7] architecture was used to perform the regression with pre-training being done on ImageNet [10]. The network was trained using the Adam optimizer [9] for 10,000 iterations with the learning rate decreasing by a factor 10 after 8,000 iterations. The training code is publicly available<sup>4</sup>. We used a batch size of 20 due to the limited memory of the NVidia 1060 GPU which has 6GB of memory. The time to train the network for 10,000 iterations was approximately 1.5 hours.

## 2.7. Evaluation

The evaluation was performed using 10-fold cross validation[17], for a setup of training on 90% of the data and validating on the remaining 10% in a rotating scheme. To evaluate the predictive performance of the network we used the mean absolute error (MAE) and mean absolute percentage error (MAPE) to measure the average accuracy of the predictions. The latter was defined as the mean over all absolute diferences between prediction and ground truth, each divided by the corresponding ground truth. Pearson’s coeficient of correlation (r), the coeficient of determination (R<sup>2</sup>), and the 95% limits of agreement (LoA) were used to compare the model predictions and the known ground truth values. Bland-Altman plots were also generated to evaluate the results.

## 2.8. Group-wise saliency analysis

The saliency analysis was performed for each individual regression task by simple backpropagation using built-in methods in PyTorch. The individual saliency maps were combined from all inferences in the cross-validation into group-wise average saliencies for males and females, respectively. For each image in each individual collage, the saliency map was deformed back using the performed image registration to standardize the geometry. This was done by first mapping each image back to the reference by inverting the deformation fields using simpleitk [15, 23, 4]. This inversion was performed on each sample after projection, thus requiring the inversion of only a 2D field, greatly reducing the computational cost. This also allows us to create a deformation field collage mapped the same as the image collage.

## 2.9. Experiments

To evaluate the feasibility of the method a set of experiments, increasing order of complexity, was performed. The experiments were:

• A - Kidney volume prediction using a collage of the kidneys

• B - Left Ventricular End Diastolic Volume (LV-EDV) prediction using a collage of the heart and surrounding regions

• C - Age prediction using a collage of sagittal slices of the heart and surrounding regions

• D - Age prediction using a collage of axial slices of the heart and surrounding regions

The first two experiments were performed with clear prior knowledge of what results to expect, as the desired information was known to be available in the image data used, even if some of it may have been lost in the projection process. The last two were of more exploratory nature to test the limits of the proposed approach. Here, diferent orientations of the image sampling were used to demonstrate that the experiment design allows for diferent structures to be sampled, such as the aorta diameter as well as its length and curvature, for example.

The collages used in Experiments A-D are shown in figure 3.

![](images/95e8f7eaa07566235ce2fcdee5b7598aed3d53db94d101aa939dc52412f6fdf7.jpg)

![](images/94d368cc7ef66ed30f3401d2a85a6ca7b2586bf57aadb5d1852db3e6d0784970.jpg)

![](images/36a4baa5678beec6a001ca817538f29374c9c66aecf5d492ef93753670666953.jpg)

![](images/ebcb2d83ec36618323042eccbb1c0528bc02b3e2aa69fe737530cb54fc43b09e.jpg)  
Figure 3: Examples of water channels of the collages used in the experiments, for kidney volume (far left), left ventricular end diastolic volume (LV-EDV) of the heart (center left), and age from images targeted on the heart using sagittal slices (center right) and axial slices (far right). Note that the packing algorithm used results in a collage with the diferent image in no particular order. The numbers are not part of the collage image data. They are added to simplify description of the diferent sub images and referred to as (x) in the text.

Detailed description of experiments

## A - Kidney volume prediction

The kidney collage is sampled from a subvolume of size 269.83 × 113.73 × 153.00mm $( \mathrm { S A G } \times \mathrm { C O R } \times \mathrm { A X } )$ , centered between the kidneys. The targeted sampling consisted of three coronal slices, ofset by 10.6mm (1), (7), (8), an axial slice (4), two sagittal slices ofset to capture the center of each kidney (5) and (9). The collage also contains mean projections in the coronal (2), axial (3) and sagittal (6) directions, respectively.

B - Left Ventricular End Diastolic Volume (LV-EDV) prediction

The heart collage is sampled from a subvolume of size 225.23 × 180.63 × 303mm and is made up by a coronal mean projection (1), an axial mean projection (2), a sagittal mean projection (4) as well as a slice in the sagittal direction (3) aimed to try to sample the bent aorta.

## C - Age prediction from sagittal slabs

The sagittal heart slab collage consisted of eight contiguous slabs of thickness 22.3mm that all together span a volume of 180.63 × 167.25 × 195mm centered around the heart. Subsamples 1 through 8 span from right to left. Subsample 9 is a mean projection of the sample volume in the sagittal direction. One voxel from the volume is missing from the slabs in favor of making all slabs the same thickness.

## D - Age prediction from axial slabs

The axial heart slab collage consisted of eight contiguous slabs of thickness 24mm that altogether span a volume of 171.71 × 167.25 × 195mm centered around the heart. Subsamples 1 through 8 span from inferior to superior. Subsample 9 is a mean projection of the sample volume in the axial direction. One voxel from the volume is missing from the slabs in favor of making all slabs the same thickness.

The selection of participants are illustrated in Figure 4.

![](images/e95d710f54bfddcab06f70dcbef3b2f82eb33cd2595380925e9e6df766b8d492.jpg)  
Figure 4: STROBE flowchart showing how the participants for experiments were selected

## 3. Results

Evaluation metrics for all regression targets are summarized in Table 1. Input collages were furthermore visualized for each of the four experiments, overlaid with the corresponding aggregated saliency maps, together with scatter plots and Bland-Altman plots in the Figures 5, 6, 7, and 8, respectively.

For the kidney volume experiment the results were highly accurate, with a mean absolute error of 6.8 mL and a mean absolute percentage error of about 2.8%. Further, the saliency maps showed that the network was able to locate and use the image information in kidneys with rather high accuracy.

The prediction of LV-EDV was not as accurate as for kidney volume, with a mean absolute error of 12.3 mL and a mean absolute percentage error of 9.8%. The saliency maps show that the method is able to localize and use information from region of the left ventricle.

Regarding the age predictions they resulted in similar performance using sagittal and axial slicing, with the latter performing slightly numerically superior. The group-wise saliency maps were also seen to highlight similar regions using both slice orientations. The regions were identified as left atrium, spinal vertebras and a region in the lungs was found highlighted in both males and females using both slice orientations.

(a) Male
<table><tr><td>Target</td><td>N</td><td>Unit</td><td>MAE</td><td>MAPE</td><td>r</td><td> $R ^ { 2 }$ </td><td>LoA</td></tr><tr><td>Kidney volume</td><td>14,182</td><td>mL</td><td>6.761</td><td>0.0277</td><td>0.984</td><td>0.967</td><td>(-18.84 to 19.37)</td></tr><tr><td>Heart LV-EDV</td><td>14,240</td><td>mL</td><td>12.324</td><td>0.0981</td><td>0.856</td><td>0.731</td><td>(-32.41 to 33.15)</td></tr><tr><td>Age sagittal slices</td><td>31,164</td><td>years</td><td>2.832</td><td>0.0448</td><td>0.880</td><td>0.773</td><td>(-6.99 to 7.06)</td></tr><tr><td>Age axial slices</td><td>31,165</td><td>years</td><td>2.759</td><td>0.0437</td><td>0.887</td><td>0.790</td><td>(-6.92 to 6.72)</td></tr></table>

Table 1: Summary of evaluation metrics from all four experiments, listing number of included subjects (N), mean absolute error (MAE), mean absolute percentage error (MAPE), Pearson’s coeficient of correlation (r), the coeficient of determination $\left( R ^ { 2 } \right) ^ { * }$ and the 95% Limits of Agreement (LoA) from the 10-fold cross-validation.

![](images/b7678ea7669dec41f038f5df1c1e2b70fe6df7806c5246bab8dac26c2320a3fb.jpg)  
(a) Male collage saliency

![](images/a18b877d4ee9a1742062e6362bb2fbb315aa283e81c0a2a4d88b40f844be73a2.jpg)  
(b) Female collage and saliency

![](images/ae9364521564912888394986fc91d12258336bf67e3e778de49accb3fce383e1.jpg)

(c) Scatter plot (left) and Bland-Altman plot (right)  
![](images/c3fdd00eea4f2835d9482a4dc814e578bec33e2f1a99e600a828fd2230ba7ee0.jpg)  
Figure 5: Kidney volume prediction results for males 5a and females 5b, with plotted numerical results 5c.

![](images/474b4ddeaab82147d7fa43e83f67bc70d994bc85d7e206293ba6273a98f05570.jpg)  
saliency  
collage

![](images/af10dfe46d5dcc509f06a2ff59a92e832a6bc0f87d99d85a8938e82ca79fc591.jpg)  
(b) Female collage and saliency

![](images/8a1848bfee9be4b2be4122febeea1d18ddecc5de985b36fe5db6dbcf53492471.jpg)

![](images/4abb7da1765c816269c4567e8afc65a2f3cf3b7e84cc3197b4fdaa6aa4aa2a23.jpg)  
(c) Scatter plot (left) and Bland-Altman plot (right)  
Figure 6: Heart LV-EDV prediction results for males 6a and females $^ { \mathrm { 6 b , } }$ with plotted numerical results 6c.

![](images/7cf60caf2cfdd20b4b94ae6f3ef0529df0655dba37cc866b897c79803a4b5ff6.jpg)  
(a) Male collage saliency

![](images/544626e4b1285cdc65ea3855e29d5366eb56fdc6671c45b7c1091806f5f3b819.jpg)  
(b) Female collage and saliency

![](images/6f95acd7e7d1dfe022da02e948492bb01055d13ce1d08a1b43f0a90447fe6240.jpg)

(c) Scatter plot (left) and Bland-Altman plot (right)  
![](images/ac474ea987c6b75c15dbde4a15cd5c3cc4e7b2e4dc4191d5ef5dd6547256a355.jpg)

Figure 7: Age (sagittal) prediction results for males 7a and females 7b, with plotted numerical results 7c.  
![](images/e2f85175219126c0eba885cc405206a53f301fb4cc10ff407371d0cf24e165a6.jpg)  
(a) Male collage saliency

![](images/4755d274dcf4c995ebd1ae2472e1d016772d1f41ca2f0345a00ba756d567bfac.jpg)  
and (b) Female collage and saliency

![](images/25b57df422cbd46e050831f52eabb0b06994002ae7b3e6207790ff6bc03e50a4.jpg)

![](images/9a078af52f97ea472ef0d5d3cb313a38e551593c4740fdae61f6d567db49f909.jpg)  
(c) Scatter plot (left) and Bland-Altman plot (right)  
Figure 8: Age (axial) prediction results for males 8a and females 8b, with plotted numerical results 8c.

## 4. Discussion

We have in this work presented and evaluated a methodology for targeted deep regression using tailored collages that can represent specific morphological information of MR image data and highlight anatomical structures relevant for the prediction with saliency analysis. The proposed technique formats the MRI data into one 2D collage for each subject, based on manually selected regions of interest that are defined in one male and one female template subject and then propagated to all other neck-to-knee body MRI scans with the help of image registration. Predictive performance was found between $R ^ { 2 } = 0 . 7 3 1 { \ - - 0 . 9 8 4 }$ and the group-wise saliency maps were found to highlight medically relevant regions.

The kidney volume regression achieved high accuracy and distinctly highlighted salient kidney tissue in the axial mean intensity projection of the collages, as seen in Fig. 5a and Fig. 5b. The mean absolute percentage error of 2.8% achieved here rivals the intra-operator variability achieved by human annotators cited in the original work that provided the underlying kidney volume measurements based on neural networks for semantic segmentation [11]. It also clearly outperformed a previously proposed approach that reported a relative error of 7% for the same kidney volume measurements, based on a more generic deep regression approach on mean intensity projections of the UK Biobank neck-to-knee body MRI [10]. It is possible that the projection images are on average more useful for a network to focus on as it shows less variation in the presentation compared to a sampled slice determined using image registration as a projection image contains both depth and width information in intensity and in-slice span.

The prediction accuracy of LV-EDV was somewhat lower than when predicting kidney volume. This was expected since LV-EDV is more challenging due to the nature of the left ventricle volume being a subregion of the heart as well as not being constant over the heart beat. This means the network must not only be able to recognize the heart but also its smaller sub-volumes. Further, the UK Biobank reference values were originally acquired using dedicated cardiac MRI scans collected using ECG-triggered scanning [20]. This means that the images used here for predictions are less suitable than the ones used for collecting the ground truth, leading to worse results. As we seen in the saliency maps of Fig. 6a and 6b, however, the network was on average capable of locating the heart as well as recognizing the left ventricle (located in the lower right part of the heart in the images) in the collages.

The age regression experiments yielded predictions with a mean absolute error just below 3 years, using image data from the mediastinal region only. In related work, a volumetric age regression approach on UK Biobank neck-to-knee body MRI yielded a mean absolute error of 3.6 years [16], whereas earlier work on 2D mean intensity projections achieved about 2.5 years [12]. The latter found aggregated saliency analysis highlighting predominantly structures of the aortic arch and knees relevant for age.

In the experiments performed in the current study we note high saliency in the heart (left atrium) in spine and lung tissue, with little emphasis on the aortic arch. These patterns were seen in both females and males and in the experiment using axial and sagittal projections (see Fig. 7a and Fig. 7b).

The targeted sampling can maximize the amount of relevant information in the created collage by targeting specific regions, here kidney and heart for the volume estimation experiments. It can also tailor an experiment for machine-learning based studies of associations between certain organs or tissues and specific biomarkers of interest, such as the thoracic region and age examined here. For both the kidney and heart (LV-EDV) volume experiments the relevant information was readily available in the source image data, limited only by the image quality, e.g. the non-ECG-triggered scans of the whole-body volume. The created collages also aimed to maximize the information in the input format for the neural network, though with the risk that the projections used would result in too much information being lost. Furthermore, these experiments were also made with strong prior knowledge on where the network could be expected to focus its attention to optimize the predictions.

We note that the in the volume predictions illustrated in Figures 4 and 5 the groupswise saliency tends to prefer projections over slices. We hypothesize that this might be due to the fact that the projections contain information from larger volumes compared to slices and thus contain more useful information for volume predictions. It might also be that individual slices show more variability in content and appearance that makes it more dificult for a network to use in the predictions. This illustrates the potential of using diferent targeting or subsampling strategies to compare the "importance" of diferent regions and sampling strategies for the predictions.

One limitation of the image formatting used here consists in the need for two reference images, one for each sex. Due to this, two targeted samplings are necessary which cannot be strictly guaranteed to be identical. This might limit studies on, for example, sexrelated diferences. The sex-specific sampling may lead to the sex being obvious even without learning any anatomical diferences between the sexes. If, as an extreme case, the heart of all male subjects was sampled but the corresponding region of interest for female subjects was placed with a significant ofset, the network may easily predict sex without actually learning anything of use for the intended study. Future work could focus on mitigating this problem, for example by registering one of the references to the other. Then only one image would need to be sampled since the sampling could be automatically transferred to the other reference, similarly to how it is done with the rest of the data set.

In conclusion we have presented and evaluated a methodology for tailored deep regression that can be used to specify and restrict the image data presented to a 2D deep regression analysis pipeline, allowing higher information density and a wide range of future medical inference studies. Beyond just estimating missing measurements, the presented saliency mapping technique provides a visualization of the "black box" analysis performed. The resulting saliency maps enable a biological interpretation and hypothesis generation, opening up a wide range of potential future regression experiments that can integrate imaging and non-imaging information.

## References

[1] K. Armanious, S. Abdulatif, A. R. Bhaktharaguttu, T. Küstner, T. Hepp, S. Gatidis, and B. Yang. Organ-based chronological age estimation based on 3d mri scans, 2021.

[2] W. Bai, M. Sinclair, G. Tarroni, O. Oktay, M. Rajchl, G. Vaillant, A. M. Lee, N. Aung, E. Lukaschuk, M. M. Sanghvi, et al. Automated cardiovascular magnetic resonance image analysis with fully convolutional networks, 2018.

[3] N. Basty, Y. Liu, M. Cule, E. L. Thomas, J. D. Bell, and B. Whitcher. Image processing and quality control for abdominal magnetic resonance imaging in the uk biobank, 2020.

[4] R. Beare, B. Lowekamp, and Z. Yaniv. Image segmentation, registration and characterization in r with simpleitk, 2018.

[5] J. H. Cole, S. J. Ritchie, M. E. Bastin, V. Hernández, S. Muñoz Maniega, N. Royle, J. Corley, A. Pattie, S. E. Harris, Q. Zhang, et al. Brain age predicts mortality, 2018.

[6] S. Ekström, F. Malmberg, H. Ahlström, J. Kullberg, and R. Strand. Fast graph-cut based optimization for practical dense deformable registration of volume images, 2020.

[7] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition, 2016.

[8] T. Kart, M. Fischer, T. Küstner, T. Hepp, F. Bamberg, S. Winzeck, B. Glocker, D. Rueckert, and S. Gatidis. Deep learning-based automated abdominal organ segmentation in the uk biobank and german national cohort magnetic resonance imaging studies, 2021.

[9] D. P. Kingma and J. Ba. Adam: A method for stochastic optimization, 2014.

[10] T. Langner, A. Martínez Mora, R. Strand, H. Ahlström, and J. Kullberg. Mimir: Deep regression for automated analysis of uk biobank mri scans, 2022.

[11] T. Langner, A. Östling, L. Maldonis, A. Karlsson, D. Olmo, D. Lindgren, A. Wallin, L. Lundin, R. Strand, H. Ahlström, et al. Kidney segmentation in neck-to-knee body mri of 40,000 uk biobank participants, 2020.

[12] T. Langner, J. Wikström, T. Bjerner, H. Ahlström, and J. Kullberg. Identifying morphological indicators of aging with neural networks on large-scale whole-body mri, 2019.

[13] T. J. Littlejohns, J. Holliday, L. M. Gibson, S. Garratt, N. Oesingmann, F. Alfaro-Almagro, J. D. Bell, C. Boultwood, R. Collins, M. C. Conroy, et al. The uk biobank imaging enhancement of 100,000 participants: rationale, data collection, management and future directions, 2020.

[14] Y. Liu, N. Basty, B. Whitcher, J. D. Bell, E. P. Sorokin, N. van Bruggen, E. L. Thomas, and M. Cule. Genetic architecture of 11 organ traits derived from abdominal mri using deep learning, 2021.

[15] B. C. Lowekamp, D. T. Chen, L. Ibáñez, and D. Blezek. The design of simpleitk, 2013.

[16] C. Nimhed. Estimation of height, weight, sex and age from magnetic resonance images using 3d convolutional neural networks, 2022.

[17] P. Refaeilzadeh, L. Tang, and H. Liu. Cross-Validation. Springer New York, New York, NY, 2016.

[18] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization, 2017.

[19] C. Sudlow, J. Gallacher, N. Allen, V. Beral, P. Burton, J. Danesh, P. Downey, P. Elliott, J. Green, M. Landray, et al. Uk biobank: an open access resource for identifying the causes of a wide range of complex diseases of middle and old age, 2015.

[20] A. Suinesiaputra, M. M. Sanghvi, N. Aung, J. M. Paiva, F. Zemrak, K. Fung, E. Lukaschuk, A. M. Lee, V. Carapella, Y. J. Kim, et al. Fully-automated left ventricular mass and volume mri analysis in the uk biobank population cohort: evaluation of initial results, 2018.

[21] C. d. Vente, P. Vos, M. Hosseinzadeh, J. Pluim, and M. Veta. Deep learning regression for prostate cancer detection and grading in bi-parametric mri, 2021.

[22] J. West, O. Dahlqvist Leinhard, T. Romu, R. Collins, S. Garratt, J. D. Bell, M. Borga, and L. Thomas. Feasibility of mr-based body composition analysis in large scale population studies, 2016.

[23] Z. Yaniv, B. C. Lowekamp, H. J. Johnson, and R. Beare. Simpleitk image-analysis notebooks: a collaborative environment for education and reproducible research, 2018.

## Acknowledgements

This research has been conducted using the UK Biobank Resource under Application Number 14237 and the Swedish National Infrastructure for Computing (SNIC,

sens2019016). This research was enabled by grants from EXODIAB, the Swedish Heart-Lung Foundation (2020-0492) and the Swedish Research Council (2016-01040, 2019- 04756, 2023-03607 ).

## Author contributions statement

F.M. wrote the main manuscript text and conducted the experiments, N.A. contributed with methodology and code, J.Ö. provided methodological expertise and edited the manuscript, T.L. developed the previous version of the methodology and edited the manuscript, H.A. provided medical expertise., R.S. and J.K. initiated and supervised the project and the data access from UK Biobank. All authors read and approved the final manuscript.

## Competing interests

JK, HA, and TL are or have been employed by Antaros Medical during the project period. Antaros Medical owns the right to patents related to the image registrations used in this work. FM, NA and JÖ report no competing interests.