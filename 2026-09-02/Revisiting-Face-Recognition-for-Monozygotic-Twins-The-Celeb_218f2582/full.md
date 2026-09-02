# Revisiting Face Recognition for Monozygotic Twins: The Celeb Twins Test Set

Michael Zang, Haiyu Wu, Mrinal Sharma, Kevin W. Bowyer Department of Computer Science & Engineering University of Notre Dame

(mzang, hwu, msharma2) @alumni.nd.edu, kwb@nd.edu

## Abstract

Past literature on face recognition for monozygotic (“identical”) twins points to facial marks and mirror asymmetry as possible directions for improved accuracy of twins recognition. The Celeb Twins Test Set (CTTS) contains webscraped image pairs for 80 sets of celebrity twins. It is the only twins test set with meta-data for twins with distinguishing skin marks and possible mirror asymmetry. CTTS is organized in the manner offace verification test sets such as LFW, CALFW, CPLFW, CFP-FP, and AgeDB-30. Current deep CNN matchers can achieve over 76% accuracy in classifying CTTS same-person / different-person image pairs. We show that current matchers do not make use of skin marks, or asymmetry, and discuss reasons for this. Finally, we discuss the feasibility of using generative AI tools such as Grok, ChatGPT and Gemini to create images of imagined monozygotic twins as a means to increase representation oftwins inface recognition training sets.

## 1. Introduction

Monozygotic (MZ) twins occur when a single embryo divides into two. The global incidence of MZ twins is estimated at between 3 to 4 per 1,000 births [31]. A 2022 National Institute of Standards and Technology (NIST) report [16] shows how challenging MZ twins are for face recognition. In this report, a threshold for 1-to-1 face matching is selected based on the impostor distribution for non-twins to give a 0.0001 (1-in-10,000) false match rate (FMR). With this threshold, most algorithms in the report had a 0.98 – 0.99 FMR for one twin being accepted as the other. The lowest FMR for any algorithm in the report was 0.475, meaning that one twin would be accepted as the other about half of the time.

One factor holding back research on automated face recognition for twins is lack of experimental datasets, for both training and testing. The ND-Twins test set was recently introduced to address this [57]. However, image pairs in ND-Twins are drawn from the ND Twins Days dataset [38]; see Figure 2. The ND-Twins test set and the larger ND Twins Days dataset do not have meta-data for the presence of distinguishing skin marks<sup>1</sup> or for “mirror” twin status. This paper analyzes the Celeb Twins Test Set, CTTS-80, which contains 264 image pairs for each of 80 sets of celebrity twins, for a total of 21,120 image pairs. This is over 3.5 times the 6,000 image pairs in ND-twins [57], LFW [19], CALFW [60], CPLFW [59], CFP-FP [43], AgeDB-30 [34], Hadrian [57] and Eclipse [57]. Over half the twins in CTTS-80 have skin marks that can distinguish between the twins; see Figures 1, 5, 7, 11 and 13. In addition, 7 sets of twins are identified as one twin being left-handed and the other right-handed, evidence of them being mirror twins.

![](images/73362f795963d4c0d76aabe122d34f36e02560d8f0c0b0dfdcaf655d47bb177f.jpg)

![](images/f600a0f4c5669801fce6d5d431cf0d4d3366dc19481df7b9b43d4ea7c953a2ba.jpg)

![](images/679405b56fb43cbf793fb8679e4cdc579286fcebee0c422799135f20720b6977.jpg)

![](images/bb36f1556296c5fdbe9edab5c4be0fc279f41fe061ec97a4dc015cb0b2c3b568.jpg)

![](images/16eb0701bbe497eaa06aaa0faecff0c39e2f7c2ca480c15e523114ec7177a268.jpg)

![](images/39eac08fff8f12b01128e030a4118dbda812104ee9d82594e9e4da650551d9ee.jpg)  
Figure 1. Which row contains an image pair of the two MZ twins rather than two images ofthe same twin? Hint: consider freckles. Images shown are 112x112, as input to CNN matchers.

The remainder of this paper is organized as follows. Section 2 reviews selected elements of MZ twin biology. Section 3 reviews research on human ability to distinguish MZ twins. Section 4 reviews research on automated face recognition for MZ twins. Section 5 describes CTTS and presents accuracy of face recognition algorithms on CTTS. Section 6 analyzes whether current matchers make use of visible skin marks and mirror asymmetry. Section 7 looks at whether current commercial Gen AI tools can produce images of non-existent twins that could be used as training data to create matchers with better accuracy for twins. Lastly, Section 8 discusses conclusions and future research directions.

## 2. MZ Twins: Biological Background

The default view of MZ twin biology is that a single embryo divides, resulting in two embryos with “the same” DNA. The default view of this for face recognition is that the two persons have “the same” facial appearance. It should not be surprising that the default view is overly simplistic.

The term “mirror twins” refers to a particular subset of MZ twins, where the “handedness” of some physical features is reversed between the twins [8, 58]. For instance, MZ twins who are mirror twins may have one twin be lefthanded and the other right-handed, or may have a hair whorl that is clockwise for one twin and counter-clockwise for the other, or reversed left-right dental patterns, among other possibilities. What exactly is mirrored varies between sets of mirror twins. A particular set of mirror twins may or may not have an asymmetry in facial appearance that is useful in distinguishing between them.

Most MZ twins occur from the embryo splitting in the range of around four days after conception [8]. For the mirror subset of MZ twins the embryo splits later, in the range of about 7 to 10 days after conception, when the embryo has begun to develop a left and a right side [8]. There is no DNA test to determine whether or not MZ twins are mirror twins. It is estimated that about 1 in 4 MZ twins are mirror twins.

It is sometimes said that moles or freckles are an element of mirror twin appearance; e.g., see [15, 22, 58]. However, there appears to be no basis for this. Mirror twins may or may not have birth marks in mirrored locations, and generally do not have skin marks in mirrored locations. And, non-mirror MZ twins generally do not have skin marks in the same location. Skin marks such as moles and freckles are one source of information to distinguish between MZ twins generally rather than only mirror twins.

![](images/82aef856e24fd639337c682f43e6baff28e6fd794aac756d21d5e97557c42ac0.jpg)  
Figure 2. Example images from the ND Twins Days dataset. Note the coordination of hairstyle, cosmetics, clothing, necklace and earrings. This is common in images from the Twins Days Festival.

A simple way to detect some mirror twins is that one is left-handed and the other right-handed. But not all mirror twins are opposite-handed. One study suggests that something in the range of 50% to 75% of mirror twins are opposite-handed [36]. We use known opposite-handedness to categorize some of the MZ twins in CTTS as mirror twins. It is likely that some CTTS twins not categorized as mirror twins are also mirror twins.

Facial appearance naturally changes over time. Aging effects may differ between MZ twins due to environmental conditions, health habits, or other factors. The practical impact is that an older pair of MZ twins has an increased chance of differences that make them easier to distinguish. The NIST study mentioned earlier [16] noted lower average similarity between MZ twins’ faces with increased age.

Age also impacts DNA. Even if the DNA of the two embryos is identical at the moment the embryo divides, future accumulated mutations can be enough to distinguish between MZ twins based on their DNA. This is done through “next-generation sequencing” or “deep whole genome sequencing” [45]. However, there is no reason to believe that mutations that allow distinguishing between MZ twin DNA cause any change in facial appearance.

In summary, while MZ twins may be called “identical”, this is not to be taken literally. MZ twins generally have some small differences in facial appearance, and the presence of such differences tend to increase with age. One element of this is skin marks such as moles, freckles or scars present on one twin but not the other. Another possible element is mirrored asymmetry in some aspect of face appearance.

## 3. Human Ability to Distinguish Twins

Biswas et al. [9] report on experiments that use images from the ND Twins Days dataset [38]. (See Figure 2.) Images are frontal pose, controlled acquisition, and cropped to the face region based on eye locations. In one experiment, 25 subjects viewed 100 image pairs, with unlimited viewing time, to rate image pairs on a 5-point scale for whether they believe the images are of the same person or of MZ twins. Mean accuracy was 93%, with a high of 100% and a low of 78%. Subjects in this experiment were also asked what features supported their decision on each image pair. The response “moles / scars / freckles” was selected on over 40% of correctly classified image pairs and only 15% of incorrectly classified pairs. In contrast, “eyes”, “nose” and “lips” were selected for over 15% of incorrectly classified pairs, and for less than 10% of correctly classified pairs.

Martini et al. [29] asked twins to categorize the person in a cropped face image shown for 30 ms as either them, their twin, or a close friend. Experiments were done with images from 10 sets of MZ twins. They report that twins are no more accurate at identifying their own image than the image of their twin, and that close friends are no more accurate at identifying one twin than the other. We speculate that the 30 ms viewing time is too short for twins or friends to reliably employ any strategy based on using local facial features or mirror asymmetry of facial features.

Parde et al. [39] compare human and algorithm ability at distinguishing twins. They use image pairs from the ND Twins Days dataset [38], with (frontal, frontal), (frontal, 45<sup>◦)</sup> and (frontal, 90<sup>◦)</sup> image pairs for same-person, twin impostor and general impostor pairs. Each of 87 subjects viewed 120 image pairs. A ResNet101-based matcher [6] was also used to compute similarity for the image pairs. They report that “. . . DCNN accuracy exceeded the accuracy of nearly all human participants tested in all conditions” [39]. Subjects were students; trained face examiners would almost certainly have higher accuracy. The matcher used is also almost certainly not competitive with current ResNet-based matchers.

Srinivas et al. [46] studied “facial marks” as a means to distinguish identical twins. Facial marks were defined broadly to include moles, freckles, light / dark spots, birthmark, scar and other features. Manual annotation and automated detection of facial marks are compared, and bipartite graph matching is used for measuring similarity between sets of facial marks. They report equal error rates of about 24% - 35% for twin recognition based on manually annotated facial marks from different annotators and about 27% for automatically annotated facial marks (see Fig. 27 of [46]). This is a radically different approach to twin recognition than in current deep CNN face matchers, using much higher resolution images. ND Twins Days images [38] have average inter-pupillary distance of 567 pixels.

Wilmer et al. [55] examined whether human face recognition ability is an inherited skill. They recruited 164 MZ twin pairs and 125 same-gender dizygotic (DZ) pairs and had each person take the Cambridge Face Memory Test (CMFT). They compared the correlation between CFMT scores for MZ twins, 0.70, versus for DZ twins, 0.29, and infer that the level of face recognition ability is strongly influenced by genetics. This result may suggest that comparing algorithm and human face recognition accuracy should be based on best-performing human rather than the average of a group of humans.

Stevenage [48] investigated whether training improves ability to discriminate twins. In one experiment, 19 subjects rated similarity of head-and-shoulders images of the same person (e.g., “Elizabeth-Elizabeth”) or the twins (“Elizabeth-Rosie”). This was followed by a training phase in which subjects labeled images by name, progressing until they reached a 90% correct. Subjects then repeated the rating of similarity of image pairs. After training, sameperson image pairs were rated as significantly more similar and twin image pairs as significantly more different. A second experiment involving a fixed amount of training produced a similar result. We speculate that the improvement in rating of similarity of image pairs could result from subjects using the training phase to learn local face features that allow them to distinguish the particular twins.

Lessons from the literature on human ability to distinguish MZ twins include the following.

• Facial marks, broadly defined, appears to be an important means for humans to accurately distinguish twins [9, 46].

• We do not find any explicit mention of the use of asymmetry features in the literature on human ability to distinguish twins.

• Face recognition ability appears to vary substantially across persons [55].

• While face recognition ability is influenced by genetics, it can be improved by training [48, 55].

## 4. Automated Face Recognition of MZ Twins

Research into automated face recognition for twins began around 2010, with a Chinese Academy of Sciences (CA-SIA) report exploring different biometric modalities for twins [50], and results from the ND Twins Days dataset [40, 41] and general availability of that dataset to the research community. The ND Twins Days dataset has been used in publications by a number of researchers [2, 9, 12, 20, 23, 24, 27, 28, 39, 44, 49].

Pruitt et al. [41] reported results for three commercial matchers and a local-region principal components analysis (PCA) algorithm on the ND Twins Days dataset. Phillips et al. [40] expanded on this, including results for matching images acquired across the 2009 and 2010 Twins Days Festivals. Paone et al. [38] expanded further on this with more detailed covariates and additional matchers.

Klare et al. [23] explored the use of different “levels” of features. Level 1 features describe holistic elements of face appearance. (This work is from before the rise of CNNs for face recognition, but CNN embeddings would likely be level 1 features.) Level 2 features are things like local binary patterns (LBP), scale-invariant feature transform (SIFT) and Gabor filters. Level 3 features are facial marks. To abstract from inaccuracies in automated detection of facial marks, facial marks were manually annotated for this work. One interesting observation from this work is that facial marks were relatively more important to increasing accuracy for twins than for non-twins.

Le et al. [24] use a two-step approach to recognize twins. The first step uses Local Fisher Discriminant Analysis (LFDA) to classify an image as from a particular set of twins, and the second step uses Gabor filter features from specific face regions to distinguish between the twins. The features are related to face aging in that, as mentioned earlier, MZ twin faces can acquire unique features as the individuals age. Juefei-Xu and Savvides [20] use facial asymmetry features and Augmented Linear Discriminant Analysis in recognizing MZ twins. Le et al. [25] build on the work of [24] and [20] to use both asymmetry features and local face region features in recognizing MZ twins. These works [20, 25] are the only ones we are aware of to emphasize the use of asymmetry features. One limitation is that the asymmetry features used assume a controlled frontal pose.

Mahalingam and Ricanek [27] analyze accuracy of LBPand HOG-based algorithms for recognizing twins. This is the only paper we found other than [50] that reports results on the CASIA dataset, and it also includes comparison results on the ND Twins Days dataset. Accuracies for the CASIA dataset are lower, possibly due to a number of the subjects being children. They evaluated accuracy across age groups and concluded that “. . . discriminability increases between the identical twins as the age progresses” [27].

Hu et al. [18] assemble the fine-grained face verification (FGFV) dataset and explore using HOG, LBP and SIFT features to classify an image pair as positive / negative. The FGFV dataset is 455 positive image pairs, selected as 2 images of the same person from the LFW dataset, and 455 negative image pairs, selected as images of a pair of twins. Images of a pair of twins are cropped from the same image sourced from the web; FGFV has only one image of a person in a pair of twins. This 100% correlation of image source and category raises the possibility that the classifier learns positive / negative to mean from different / same source image [11, 26]. Their best combination of classifier and features achieves just over 72% accuracy in classifying positive / negative pairs.

Shalin et al. [44] argue that automatically detected facial marks are an effective way to distinguish twins, and claim to use the ND Twins Days dataset in experiments. However, Figure 1 in [44] appears to be a copy of Figure 9 in [46], and Figure 2 in [44] appears to be a copy of Figure 4 in [38]. While [46] deals with twins recognition from facial marks, [38] does not, and [38] is not cited in [44].

Mousavi et al. [35] propose a distinctive landmark-based face recognition (DLFR) approach, using an M-SIFT algorithm to find landmark points on the face. They also create a dataset of 110 pairs of MZ twin images and 100 pairs of non-twin images. Some example images in the paper are of babies and children, and some of adults.

Afaneh et al. [2] explored twins recognition using fusion at feature-, score- and decision-level with PCA, HOG and LBP features. They also report results for an early deep CNN face matcher. Sudhakar and Nithyanandam [49] propose a multi-feature, multi-level fusion approach to distinguishing twins, similar to [2], but using more types of features than [2]. Interestingly, they convert color images to grayscale for further processing. No deep CNN matchers are evaluated in this work.

Ahmad et al. [3] report on training an ArcFace-style face matcher with triplet loss for twin recognition. The experimental dataset apparently consists of 1400 images of each person in 5 sets of twins, for a total of 14,000 images, obtained from Shutterstock.com. The data is split into 7 parts, with 6 used for training and 1 for validation. Apparently all twins are represented in each of the 7 parts, and so there is no estimate of separate test accuracy.

Sundaresan and Shanthi [51] categorize twins face recognition efforts and suggest topics for future research. The categorization of primary types of features includes facial marks, facial aging features and facial asymmetry, which is essentially equivalent to facial marks and asymmetry, with either being from birth or aging. There are no new experimental results in this paper.

Sami et al. [42] explore similarity of facial appearance in MZ twins and in unrelated look-alikes or “doppelgangers”. They use the WV Twins Days dataset, a superset of the ND Twins Days dataset, with additional collections through Twins Days 2018. (The paper states, “The data that support the findings of this paper are available from the corresponding author upon reasonable request.”) Twins discrimination results are shown for a range of deep CNN matchers. An interesting element of this work is that it develops a metric of facial similarity distinct from an embedding and similarity metric used for recognition. Look-alikes are defined in terms of similarity relative to a metric derived from a twins dataset, and an interesting conclusion is that “lookalike pairs are quite rare in the populations contained within the data sets used in this study”.

Hsu and Tsai [17] evaluated multiple deep CNN matchers on their own dataset of 182 images of 54 sets of twins. They fine-tune the pre-trained deep CNN that initially achieves highest accuracy, using a set of 31 twin pairs. However, the fine-tuning step does not increase overall accuracy on twins, which is 61% with and without fine-tuning, and much higher on positive image pairs (77%) than negative image pairs (45%). There is no mention of the twins image dataset being made available to other researchers.

The largest evaluation of face matching for MZ twins is the NIST report [16] mentioned earlier. The evaluation uses a 1-in-10,000 FMR threshold determined from a large set of mugshot-quality, non-twins image pairs. The fraction of the similarity scores for pairs of images of MZ twins that exceeds the 1-in-10,000 threshold is the twins FMR. Twins images come from the dataset used in [42]. One conclusion is – “In the NIST face recognition verification test, algorithms have high false matches and cannot distinguish identical and fraternal same-sex twins as different people.” The report also notes – “Facial landmarks are distinctive and useful, but usually there only few of these features on the face and sometimes they cannot be observed ...”

Brahmbhatt et al. [12] describe an approach to using automatically detected skin marks to distinguish twins. They use a pre-trained deep network to detect various types of skin marks, and build a feature vector using the location and type of skin marks. Using the ND Twins Days dataset, they select the best image of each person for their method. They end up analyzing 319 images from 74 sets of twins, and report an accuracy of 96%.

Observations from previous work on automated face recognition include the following.

• As with human ability to distinguish twins, facial marks appear to be useful [12, 16, 23–25, 35, 51].

• Features related to facial asymmetry may be useful but there is limited work in this area [20, 25].

• The most widely used MZ twins dataset is the ND Twins Days dataset, which is now 15+ years old, and too small for effective deep CNN training.

• It is difficult to compare accuracy across papers, due to lack of a standard face verification test set.

## 5. The Celeb Twins Test Set

Creation of CTTS began with lists of celebrity twins; e.g., [47]. Candidate MZ twins were dropped if a web search did not readily return at least 12 different images of each twin, with the face region being about 112x112 pixels or larger in each image, and with high-confidence identity labeling. The minimum number of images per person and the minimum size for the face region favored twins with greater celebrity status and online presence. High-confidence identity labeling favored twins who played for different sports teams, held different positions in government and / or had distinguishing facial marks visible in many images.

Of the 80 sets of MZ twins in CTTS, 43 have a skin mark that could be used to distinguish between them, and 7 have one twin that is left-handed and the other right-handed, indicating mirror twins. There are 36 male and 44 female sets of twins. There is one set of twins born in the 1940s, 9 in the 1960s, 17 in the 1970s, 26 in the 1980s, 21 in the 1990s, and 6 in the early 2000s.

Web-scraping of publicly accessible data is and has long been legal in the US [1, 7, 32, 54]. CTTS images are versions of face regions cropped from publicly-accessible images, with each face region normalized to standard orientation and 112x112 pixels in size, as appropriate for input to popular deep CNN matchers. Images are organized into a balanced number of pairs, and pairs into folds, to create a test set appropriate to ten-fold cross-validation. CTTS-80 is available to researchers who certify agreement to noncommercial use and that CTTS use is legal in their jurisdiction. See https://github.com/mzang20/CTTS for additional details and access to CTTS.

With 12 images selected for each individual, there are 132 same-person image pairs and 144 different-person pairs for each set of twins. In the context of this type of face verification test set, same-person pairs are “positive” pairs and different-person pairs are “negative” pairs. The number of positive and negative pairs for each set of twins is balanced by randomly selecting 132 of the 144 negative pairs. Each fold contains 2x132= 264 image pairs for each of 8 sets of twins. Thus there are 264x8= 2,112 image pairs per fold and 21,120 image pairs total across the 10 folds. This is over 3.5 times the 6,000 image pairs in other test sets LFW, CALFW, CPLFW, CFP-FP, AgeDB-30, Hadrian, Eclipse or ND-Twins.

There are two main approaches to reporting test accuracy for face verification. One is to report FNMR at a threshold determined by a specific FMR on an impostor distribution. This is the approach used in the NIST report [16], and associated with the IJB benchmarks [30]. The other approach is to report the average accuracy on 10-fold cross-validation of a dataset of positive and negative image pairs. The threshold that gives highest accuracy classifying image pairs on 9 folds is used to classify the image pairs in the 10-th fold.

![](images/84503a72f320932fa6ae60dedcdea01ac02888ff8fb31ca5eb1f9f4a3f64262f.jpg)  
(a) Twelve 112x112 normalized face crops - Ronde Barber

![](images/5c2486041b99c0f5103c9120329ac8e6320677427c46d577a860222feb41a584.jpg)  
(b) Twelve 112x112 normalized face crops - Tiki Barber

![](images/3140f6796fc98699279a4a9becfa8fe367f48dfa476a17e37528ffc446e41a36.jpg)  
(c) Distance Distributions of Same-Person (“positive”) and Different-Person (“negative”) Image Pairs

Figure 3. Example of Well-Separated Distance Score Distributions. Euclidean distance of embeddings from ArcFace ResNet-100 backbone trained on MS1MV2. Threshold selected from other 9 folds in 10-fold cross-val is lower than ideal for these distributions, but accuracy still exceeds 96%. Note image variation in face expression, facial hair, eyeglasses, background and other factors.

![](images/48938d85e6955d9b8e9d6e4e02c4060c058f1d68eec8ab728d7e42719cc41103.jpg)  
(a) Twelve 112x112 normalized face crops - Luke Goss

![](images/f1b02af8ab457255d2663423ffe45a7b0a2e067589a98cc62280e91e457ffeb7.jpg)  
(b) Twelve 112x112 normalized face crops - Matt Goss

![](images/e348c4bd22983781487c0d47c43928dbb8722c5f853bc0859be31ac644d5abcb.jpg)  
(c) Distance Distributions of Same-Person (“positive”) and Different-Person (“negative”) Image Pairs  
Figure 4. Example of Well-Separated Distance Score Distributions. Euclidean distance of embeddings from ArcFace ResNet-100 backbone trained on MS1MV2. Threshold selected from other 9 folds in 10-fold cross-val is lower than ideal for these distributions.

![](images/c433b408579a28079db83ce57ab66504cb6a35e5a4f9737a06bed066f0cd8c21.jpg)  
(b) Twelve 112x112 normalized face crops - Tia Mowry

![](images/14452eafdccffc90888ac7da2e9d4d95ea9d38902aa29dc6522ad319cc027cab.jpg)  
(c) Distance Distributions of Same-Person (“positive”) and Different-Person (“negative”) Image Pairs

Figure 5. Example of Well-Separated Distance Score Distributions. Euclidean distance of embeddings from ArcFace ResNet-100 backbone trained on MS1MV2. Threshold selected from other 9 folds in 10-fold cross-val is lower than ideal for these distributions. Tamera Mowry has a skin mark visible in these images below her left eye that distinguishes the twins.

<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Train set</td><td rowspan=1 colspan=2>Trad 5</td><td rowspan=1 colspan=2>Hadrian</td><td rowspan=1 colspan=1>Eclipse</td><td rowspan=1 colspan=1>ND-Twins</td><td rowspan=1 colspan=1>CTTS-80</td></tr><tr><td rowspan=2 colspan=1>AdaFaceAdaFace</td><td rowspan=2 colspan=1>MS1MV2WebFace4M</td><td rowspan=1 colspan=2>97.17</td><td rowspan=1 colspan=2> $9 2 . 3 5 \pm 1 . 8 5$ </td><td rowspan=1 colspan=1> $8 2 . 3 2 \pm 1 . 6 6$ </td><td rowspan=1 colspan=1> $6 8 . 5 5 \pm 3 . 0 1$ </td><td rowspan=1 colspan=1> $7 6 . 2 0 \pm 3 . 7 3$ </td></tr><tr><td rowspan=1 colspan=2>97.40</td><td rowspan=1 colspan=2> $9 1 . 0 7 \pm 2 . 1 9$ </td><td rowspan=1 colspan=1> $8 2 . 8 0 \pm 1 . 6 3$ </td><td rowspan=1 colspan=1> $7 2 . 4 8 \pm 3 . 0 5$ </td><td rowspan=1 colspan=1> $7 5 . 2 2 \pm 3 . 6 9$ </td></tr><tr><td rowspan=1 colspan=1>AdaFace</td><td rowspan=1 colspan=1>glint360k</td><td rowspan=1 colspan=2>97.69</td><td rowspan=1 colspan=2> $9 5 . 6 3 \pm 1 . 6 4$ </td><td rowspan=1 colspan=1> $8 3 . 8 8 \pm 1 . 5 6$ </td><td rowspan=1 colspan=1> $7 5 . 8 8 \pm 2 . 4 8$ </td><td rowspan=1 colspan=1> $7 5 . 4 6 \pm 4 . 0 9$ </td></tr><tr><td rowspan=4 colspan=1>ArcFaceArcFaceArcFace</td><td rowspan=4 colspan=1>MS1MV2WebFace4Mglint360k</td><td rowspan=1 colspan=2>97.14</td><td rowspan=1 colspan=2> $9 1 . 7 2 \pm 1 . 7 8$ </td><td rowspan=1 colspan=1> $8 1 . 8 0 \pm 1 . 1 9$ </td><td rowspan=1 colspan=1> $6 8 . 1 3 \pm 3 . 2 4$ </td><td rowspan=1 colspan=1> $7 6 . 5 0 \pm 3 . 4 2$ </td></tr><tr><td rowspan=2 colspan=2>97.44</td><td rowspan=1 colspan=1></td><td rowspan=2 colspan=2> $9 1 . 7 8 \pm 2 . 1 4$ </td><td rowspan=2 colspan=1> $8 3 . 2 0 \pm 1 . 6 2$ </td><td rowspan=2 colspan=1> $7 1 . 7 3 \pm 3 . 0 8$ </td><td rowspan=2 colspan=1> $7 4 . 5 3 \pm 4 . 0 7$ </td></tr><tr><td rowspan=1 colspan=1>J</td></tr><tr><td rowspan=1 colspan=2>97.60</td><td rowspan=1 colspan=2> $9 5 . 6 8 \pm 1 . 6 5$ </td><td rowspan=1 colspan=1> $8 4 . 1 2 \pm 1 . 7 8$ </td><td rowspan=1 colspan=1> $8 4 . 6 5 \pm 2 . 8 7$ </td><td rowspan=1 colspan=1> $7 4 . 3 9 \pm 4 . 6 0$ </td></tr><tr><td rowspan=3 colspan=1>UniFaceUniFaceUniFace</td><td rowspan=3 colspan=1>MS1MV2WebFace4Mglint360k</td><td rowspan=1 colspan=2>97.13</td><td rowspan=1 colspan=2> $9 1 . 6 3 \pm 2 . 1 4$ </td><td rowspan=1 colspan=1> $8 2 . 1 8 \pm 1 . 8 3$ </td><td rowspan=1 colspan=1> $6 5 . 7 5 \pm 2 . 7 7$ </td><td rowspan=2 colspan=1> $7 6 . 1 6 \pm 4 . 3 5$  $7 5 . 0 6 \pm 3 . 8 7$ </td></tr><tr><td rowspan=1 colspan=2>97.41</td><td rowspan=1 colspan=2> $9 0 . 6 5 \pm 1 . 9 1 $ </td><td rowspan=1 colspan=1> $8 2 . 2 0 \pm 1 . 5 5$ </td><td rowspan=1 colspan=1> $7 1 . 8 8 \pm 4 . 2 7$ </td></tr><tr><td rowspan=1 colspan=2>97.56</td><td rowspan=1 colspan=2> $9 3 . 5 3 \pm 1 . 9 0$ </td><td rowspan=1 colspan=1> $8 3 . 1 8 \pm 1 . 3 5$ </td><td rowspan=1 colspan=1> $7 1 . 7 2 \pm 2 . 8 2$ </td><td rowspan=1 colspan=1> $7 4 . 9 4 \pm 3 . 5 8$ </td></tr><tr><td rowspan=3 colspan=1>MagFaceMagFaceMagFace</td><td rowspan=1 colspan=1>MS1MV2</td><td rowspan=1 colspan=2>97.00</td><td rowspan=1 colspan=2> $9 2 . 8 2 \pm 2 . 4 9$ </td><td rowspan=1 colspan=1> $8 2 . 6 7 \pm 1 . 7 3$ </td><td rowspan=1 colspan=1> $6 8 . 5 7 \pm 2 . 1 2$ </td><td rowspan=1 colspan=1> $7 5 . 7 1 \pm 3 . 8 5$ </td></tr><tr><td rowspan=2 colspan=1>WebFace4Mglint360k</td><td rowspan=1 colspan=2>97.41</td><td rowspan=1 colspan=2> $9 0 . 9 2 \pm 2 . 1 3$ </td><td rowspan=1 colspan=1> $8 2 . 4 7 \pm 1 . 2 8$ </td><td rowspan=1 colspan=1> $7 1 . 3 2 \pm 3 . 6 8$ </td><td rowspan=1 colspan=1> $7 3 . 9 8 \pm 4 . 2 9$ </td></tr><tr><td rowspan=1 colspan=2>97.56</td><td rowspan=1 colspan=2> $9 5 . 2 0 \pm 1 . 5 3$ </td><td rowspan=1 colspan=1> $8 3 . 7 8 \pm 1 . 3 8$ </td><td rowspan=1 colspan=1> $7 1 . 9 8 \pm 3 . 1 3$ </td><td rowspan=1 colspan=1> $7 3 . 1 4 \pm 4 . 1 9$ </td></tr></table>

Table 1. Accuracy on Trad 5, Hadrian, Eclipse, ND-Twins, and CTTS. ResNet 100 backbone for all matcher models. “Trad 5” is average accuracy across LFW, CALFW, CPLFW, CFP-FP and AgeDB-30. Average and std dev from 10-fold cross-val are listed for Hadrian, Eclipse, ND-Twins and CTTS.

Each fold has accuracy computed in this way, and the average and standard deviation of the accuracies on the 10 folds is reported. This is the approach used with LFW, CALFW, CPLFW, CFP-FP, AgeDB-30, Hadrian, Eclipse, ND-Twins, CTTS and various other face verification test sets.

The threshold used to classify image pairs in CTTS is selected based on classifying other twins image pairs, whereas the threshold used to classify twins image pairs in the NIST report is selected for a 1-in-10,000 FMR on a distribution of non-twins image pairs. Due to the different approaches, it is possible that a face matcher could have a 0.99 FMR in the NIST results and also have 75% accuracy on CTTS.

The CTTS folds are twin-set-disjoint, meaning that all image pairs of a given set of twins appear in the same fold. Thus, when a given set of twins are scored for test accuracy, their image pairs are not used in selecting the threshold by which they are scored. This should give an accuracy estimate that is less optimistically biased than if the image pairs for a given set of twins were distributed across all the folds.

Accuracy achieved on CTTS-80 by 4 matchers (ArcFace [14], AdaFace [21], UniFace [61], MagFace [33]), each trained on 3 different widely-used training sets (MS1MV2 [14], WebFace4M [62], glint360k [5]), for a total of 12 matcher instances, is shown in Table 1. For context, accuracy for the 12 matcher instances on the traditional five (“Trad 5”) face verification test sets LFW, CALFW, CPLFW, CFP-FP and AgeDB-30 is also shown, along with accuracy for newer test sets emphasizing facial hairstyles (“Hadrian”), lighting differences (“Eclipse”) and ND-Twins. Accuracies in Table 1 for the Trad 5 test sets follow the pattern expected from the literature, $\geq 9 7 \%$ for all

12 matchers. The test sets that are focused on challenging facial hair conditions (Hadrian) and illumination conditions (Eclipse) have lower accuracy than Trad 5. CTTS-80 and ND-Twins have still lower accuracies. It is important to note that the standard deviation is higher for the twins test sets, making it difficult to draw firm conclusions about accuracy differences between matchers, training sets, or ND-Twins versus CTTS-80. Accuracy is more stable across matchers and training sets for CTTS than for ND-Twins, likely due to the larger number of image pairs in CTTS.

Available training sets for deep CNN face matchers, including MS1MV2, glint360k and WebFace 4M, are webscraped. CTTS is also web-scraped. MZ twins in CTTS who were established celebrities well before the training sets were created, such as Ronde and Tiki Barber or Aaron and Shawn Ashmore, likely also have images in the training sets. Other CTTS twins, like Keegan and Kris Murray, and Chase and Sydney Brown, who were born in 2000 and have fairly recently become celebrity pro athletes, are almost certainly not in the circa 2019 MS1MV2 and likely also not in the other training sets. It is possible in principle that a matcher could have higher accuracy for CTTS twins that are in the matcher’s training set. However, the overlapped twins between CTTS and a training set would be such a very small percent of the training set that it is unlikely there is any significant effect.

To examine the accuracy across different sets of twins in more detail, we selected the ArcFace matcher trained on MS1MV2 to use in results in the rest of this paper. This matcher gives the highest accuracy on CTTS of the 12 matcher instances in Table 1, though not by a statistically significant margin. This matcher should give a representative idea of the accuracy that can be achieved by current matchers on CTTS.

We first examined how accuracy varies across the sets of twins. The standard deviation listed in Table 1 is for the average accuracy across the 10 folds. The range of accuracy variation across the 80 sets of twins is surprisingly large: 7 sets of twins had accuracy in the range (50%, 60%), 19 in [60%, 70%), 26 in [70%, 80%), 11 in [80%, 90%) and 17 in [90%, 100%).

Accuracy for a set of twins results from two factors. One, different twins can have different levels of separation in the distributions of the distances for their positive and negative image pairs. The separation in their distributions determines the max potential accuracy for a set of twins. Two, the decision threshold selected based on the other 9 folds may or may not generalize well to a given set of twins in the 10- th fold. Twins with high accuracy have good separation of their distributions and a well-placed threshold. Twins with low accuracy have poor separation of their distributions, a poorly-placed threshold, or both.

Examples of well-separated distance distributions appear in Figures 3, 4 and 5. Note that the faces in these examples generally have approximately frontal pose, little or no occlusion by hair, no hats and eyeglasses in only one image. Examples of not-well-separated distance distributions appear in Figures 6, 7 and 8. Multiple images of each twin in Figure 6 contain occlusion by caps, and there are examples of off-frontal pose and atypical facial expressions. Images in Figure 7 show substantial occlusion of the forehead by hairstyle and caps, and substantial pose variation. Images in Figure 8 show substantial variation in pose and in hairstyle. This contrast in the image variation between twins with well-separated and not-well-separated distance distributions suggests that it may be possible to design image acquisition to get well-separated distance distributions.

## 6. Do Current Matchers Encode Skin Marks?

As described earlier, previous research has emphasized skin marks as a means to distinguish identical twins [9, 12, 16, 24, 25, 35, 46, 51]. Over half the twins in CTTS have skin marks that can be used to distinguish them. Of course, skin marks may not be visible in some images, due to pose, cosmetics, image resolution or blur. Figure 9 shows how skin marks that are easily visualized in higher resolution images can become ambiguous at the 112x112 resolution commonly used by face matching models. However, distinguishing skin marks are visible in the 112x112 pixel images of the twins in Figures 5, 7, 11 and 13. This raises the question - Do current deep CNN matchers make use of skin marks?

For an experimental answer to this question, we created skin-mark-erased versions of the Tamera Mowry images, editing only the area of the skin mark. We then matched the skin-mark-erased versions of the Tamera images against the same Tia images as in Figure 5. If the skin-mark-visible images play a role in creating the well-separated distributions in Figure 5, then the distributions with the skin-mark-erased versions of Tamera should be less well separated. However, the distributions do not change significantly using the skinmark-erased images; see Figure 10. This suggests that the visible skin mark has little or no effect on the embedding computed from the image.

To investigate this more directly, we matched the original Tamera images against the skin-mark-erased Tamera images. In this result, the positive pairs are images that either both have a skin mark or both do not, and the negative pairs have one image with a visible skin mark and one without. The distance distributions for positive and negative pairs are nearly totally overlapped. The exception is the 12 negative pairs that are an image with and without the visible skin mark. The distances for these pairs are clustered near zero distance; see Figure 10(c). This is direct evidence that the presence of the visible skin mark has almost no effect on the embedding computed from the image.

The phenomenon of visible skin marks not affecting the embedding is a general property of the model, not something particular to this set of twins. As additional examples, the same phenomenon is evident for two additional sets of twins who have distinguishing skin marks visible in the images; see Figures 11 and 12 for the Hassan twins and Figures 13 and 14 for the Kaczynski twins. (For space reasons, we present just three example sets of twins for this.) The distance distributions for positive and negative image pairs are essentially the same using either original or skinmark-erased versions of images of the twin with the distinguishing skin mark. And, the difference distribution between original and skin-mark-erased images of the one person is the same, with zero difference between original and skin-mark-erased versions of the same image.

Several factors in typical training of deep CNN face matchers may help to explain why visible skin marks have negligible effect on the embedding that is computed. One is the training set. Based on population statistics, it is likely that far less than 1% of identities in the training set represent MZ twins. MZ twins would likely need to be a much larger fraction of the training data to have the network learn to integrate twins-related elements into the embedding. Another factor is augmentation techniques used in training. One common augmentation is a horizontal flip of the training image. With this augmentation, during training the network sometimes sees the distinguishing skin mark on the left side of the face and sometimes on the right. With all identities in training seen sometimes in original orientation and sometimes with horizontal flip, the likely effect is to learn not to use features related to skin mark position or asymmetry.

![](images/535bc38ad18cb9a9b5e510861805b351ffa030da360e1fdb23e2a390d98caf12.jpg)  
(b) Twelve 112x112 normalized face crops - Sydney Brown

![](images/b8f9121308350b7abd7daab7af3be3e1f1f489991fc7199dea6d112804d69162.jpg)  
(c) Distance Distributions of Same-Person (“positive”) and Different-Person (“negative”) Image Pairs  
Figure 6. Example of Not-Well-Separated Distance Score Distributions. Euclidean distance of embeddings from ArcFace ResNet-100 backbone trained on MS1MV2. Chase is left-handed and Sydney is right-handed, indicating mirror twins.

![](images/ecc8240a1d441279288bd5d9f5e5f1f2f41b74d4c79c5fd7fa99ad10fdfbe510.jpg)  
(a) Twelve 112x112 normalized face crops - Lucas Dobre

![](images/14c31fdb62b73ec2627ebce82cb0cd840ff543dbf3d68268c6997675fda383b1.jpg)  
(b) Twelve 112x112 normalized face crops - Marcus Dobre

![](images/a126e4a1d893aeacf5f0e6c854ba3bf517e3dbc725f448d7f7893eb9e2dfa69b.jpg)  
(c) Distance Distributions of Same-Person (“positive”) and Different-Person (“negative”) Image Pairs

Figure 7. Example of Not-Well-Separated Distance Score Distributions for MZ Twins. Euclidean distance of embeddings from ArcFace ResNet-100 backbone trained on MS1MV2. Threshold selected from other 9 folds in 10-fold cross-val is higher than ideal for these distributions. Note that Lucas Dobre has a skin mark on his left cheek, visible in these images, that distinguishes between the twins.

![](images/c5dc36b858066728af6f027b9449c14ade5b82c6c2d15b68e91157f0c805587b.jpg)  
(a) Twelve 112x112 normalized face crops - Amelia Spencer

![](images/8723416430013358037eaae205611acc91c49b8d130e48199cc766a0e62fee1d.jpg)  
(b) Twelve 112x112 normalized face crops - Eliza Spencer

![](images/bfc931b1784ccaffcb75227b4f26e50b63c61e177a9548a0205766960fa67984.jpg)  
(c) Distance Distributions of Same-Person (“positive”) and Different-Person (“negative”) Image Pairs  
Figure 8. Example of Not-Well-Separated Distance Score Distributions. Euclidean distance of embeddings from ArcFace ResNet-100 backbone trained on MS1MV2. Threshold selected from other 9 folds in 10-fold cross-val is lower than ideal for these distributions.

![](images/454a685e2d2f8f9470c17b34ed731b3a4c3f3e4db2135920aea211944c456361.jpg)  
Figure 9. Skin Mark Visibility And Image Resolution. Upper row of images shown at size on the page to illustrate differences in resolution. Lower row of images shown at same size on the page to illustrate changing visibility of skin mark.

![](images/dac409f5333ec393e62e790a51a8290b73282ff02028586a0cc6884da3028950.jpg)  
(a) Skin-mark-erased Versions of Images in Figure 5(a)

![](images/1a8d341bd71377eac863b4238b934ac4a58354ed48df78fff638fc15a2d8356d.jpg)

(b) Distance Distributions Using Skin-mark-erased Tamera Images; compare to Figure 5(c)  
![](images/c8310299a33a2199fe1937b4f4c07a27feb95d02eb2690369ca41f294bd225bb.jpg)  
(c) Distance Distributions from Matching Original and Skin-mark-erased Images  
Figure 10. Exploring the Effect of Skin-Mark-Erased Images. Using skin-mark-erased images produces essentially the same result as images with skin mark present. Matching skin-mark-erased to skin-mark-present images results in distances clustered near zero, indicating that the skin mark has essentially no impact on the embedding computed by the model.

![](images/e0144a393bdea7810f04d12604734d83980aaabfdee72bf0e3f7f6f8de88f296.jpg)  
(b) Twelve 112x112 normalized face crops - Ibrahim Hassan

![](images/895e293df4f627dc58e32c7455fe3223de5c4da08a31fa2d06677f862b91b5be.jpg)  
(c) Distance Distributions of Same-Person (“positive”) and Different-Person (“negative”) Image Pairs  
Figure 11. Example of Distance Score Distributions. Euclidean distance of embeddings from ArcFace ResNet-100 backbone trained on MS1MV2. Ibrahim Hassan has a skin mark visible in these images outside and below the right corner of his lips that distinguishes the twins.

![](images/88992dc4aae1cc498362c2d1573cc10be7388a35d83958acc9cc0f9750250d2d.jpg)  
(a) Skin-mark-erased Versions of Images in Figure 11(b)

![](images/a55b8c28facddbda1bba7c99fed72321fd5984653b3c5ade879dd3595b4a35c7.jpg)  
(b) Distance Distributions Using Skin-mark-erased Ibrahim Hassan Images; compare to Figure 11(c)

![](images/27381a56e74504fa43c29f7b8008879b05d5713e7bf6afec7bebcfe27645603a.jpg)  
(c) Distance Distributions from Matching Original and Skin-mark-erased Images  
Figure 12. Exploring the Effect of Skin-Mark-Erased Images. Using skin-mark-erased images produces essentially the same result as images with skin mark present. Matching skin-mark-erased to skin-mark-present images results in distances clustered near zero, indicating that the skin mark has essentially no impact on the embedding computed by the model.

![](images/28188085f7f6078c59fb4ddece3eb33786db7a7d6aa170d1caa3528520f0f7a3.jpg)  
(a) Twelve 112x112 normalized face crops - Jaroslaw Kaczynski

![](images/c4f64bd63662cc4cbad3585bbd692263e9cff0b874fe585f795df72b51b9e0da.jpg)  
(b) Twelve 112x112 normalized face crops - Lech Kaczynsk

![](images/62bcb76842d96b2bfe44bcded8295eaed1f6f27a530e917218026d6845660539.jpg)  
(c) Distance Distributions of Same-Person (“positive”) and Different-Person (“negative”) Image Pairs  
Figure 13. Example of Distance Score Distributions. Euclidean distance of embeddings from ArcFace ResNet-100 backbone trained on MS1MV2. Lech Kaczynski has a skin mark visible in these images below his left eye that distinguishes the twins.

![](images/aef4ad6c9ff3d2a898e99d58b41fef8403eb3d70c3a81a162ba51da600271195.jpg)  
(a) Skin-mark-erased Versions of Images in Figure 13(b)

![](images/d64101b9b2c967d62501e5de8456b433817c2e0616648e99fe7535f13327eee7.jpg)  
(b) Distance Distributions Using Skin-mark-erased Lech Kaczynski Images; compare to Figure 13(c)

![](images/bdfecf730baefe77b3835c25d612c274c3cb20edec545224109fa00cb481a669.jpg)  
(c) Distance Distributions from Matching Original and Skin-mark-erased Images  
Figure 14. Exploring the Effect of Skin-Mark-Erased Images. Using skin-mark-erased images produces essentially the same result as images with skin mark present. Matching skin-mark-erased to skin-mark-present images results in distances clustered near zero, indicating that the skin mark has essentially no impact on the embedding computed by the model.

Still another factor is a practice commonly used in computing the 10-fold cross-validation accuracy for a face verification test set. This practice is to take the average of two embeddings, one for the original image and one for its horizontal flip, as the embedding to use for the image. This can reduce “noise” in the embedding and increase accuracy. However, it may complicate seeing increased accuracy from using an embedding that encodes location of skin marks.

To explore whether eliminating the horizontal flip augmentation during training could enable better accuracy on twins, we re-trained the ArcFace ResNet 100 matcher with this augmentation turned off, and also computed accuracy on the test set using only the embedding from the original image. This should enable the training process to learn to compute embeddings that exploit asymmetry in the images, either location of skin marks or mirror asymmetry. Overall CTTS accuracy for this “asymmetry-aware” model is 78.58% ±3.7, not statistically significantly different from the original model accuracy. As an example of the asymmetry-aware results, the distance distributions for the Mowry twins using this matcher are shown in Figure 15. The accuracy is slightly lower than for the original model. However, to focus on whether the distance distributions for positive and negative pairs are better separated, we consider the number of pairs of overlap between the distributions. The histograms in Figure 15 have an overlap size of 11 pairs. The corresponding histograms for the original matcher, in Figure 5, have an overlap size of 6 pairs. So the asymmetry-aware matcher is not able to exploit the skin mark to reduce overlap of the distributions.

Table 2 shows how the asymmetry-aware model changes the overlap in distributions for the 7 known sets of mirror twins in CTTS. The overlap increases for 3 sets of mirror twins, stays the same for 1, and decreases for 3. However, the largest magnitude changes by far are decreases for 2 sets of mirror twins, indicating the asymmetry-aware matcher resulted in better separation for them. One possible interpretation of these results is that most mirror twins do not have an asymmetry in facial appearance that can be used to distinguish them, but that a fraction of mirror twins do. Currently, almost nothing is known about the presence of asymmetry-related features in mirror twins facial appearance, and so this topic is deserving of further study.

## 7. Twins Training Images from Generative AI?

What fraction of the training set identities should be MZ twins in order to learn a model that has greater accuracy on MZ twins? This is a speculative question that currently has no firm answer. However, recent works aimed at improving recognition accuracy for persons wearing sunglasses [52] and persons with facial hair [37] found that having 20% or more of the identities in the training set targeted to the particular condition led to a noticeable accuracy increase. WebFace 12M and WebFace 42M [62] are widely-used face recognition training sets in the research community. Commercial face recognition training sets may be much larger. WebFace 12M and WebFace 42M contain images for 600,000 and 2M identities, respectively. Using 20% as a rule of thumb suggests that images for 60,000 to 200,000 sets of twins might be needed to train a “twinsaware” matcher. In the context of images of real MZ twins, this seems an impossible goal.

<table><tr><td rowspan=1 colspan=1>Twins</td><td rowspan=1 colspan=1>originalmatcheroverlap</td><td rowspan=1 colspan=1>asymmetryaware matcheroverlap</td><td rowspan=1 colspan=1>difference</td></tr><tr><td rowspan=1 colspan=1>Boer</td><td rowspan=1 colspan=1>60</td><td rowspan=1 colspan=1>49</td><td rowspan=1 colspan=1>-11</td></tr><tr><td rowspan=1 colspan=1>Brown</td><td rowspan=1 colspan=1>88</td><td rowspan=1 colspan=1>94</td><td rowspan=1 colspan=1>+6</td></tr><tr><td rowspan=1 colspan=1>Bryan</td><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>Howe</td><td rowspan=1 colspan=1>53</td><td rowspan=1 colspan=1>49</td><td rowspan=1 colspan=1>-4</td></tr><tr><td rowspan=1 colspan=1>Lamoureux</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>+6</td></tr><tr><td rowspan=1 colspan=1>Murray</td><td rowspan=1 colspan=1>68</td><td rowspan=1 colspan=1>74</td><td rowspan=1 colspan=1>+6</td></tr><tr><td rowspan=1 colspan=1>Pahde</td><td rowspan=1 colspan=1>36</td><td rowspan=1 colspan=1>23</td><td rowspan=1 colspan=1>-13</td></tr></table>

Table 2. Effect of Asymmetry-aware Matcher on Overlap Between Positive and Negative Pair Distance Distributions. Overlap is simply the number of pairs overlapped in the histograms of positive and negative pair distances. A negative difference indicates the asymmetry-aware matcher resulted in less overlap, increasing the max accuracy that could be achieved with an ideal threshold.

Generating face images of non-existent persons to use as training data for face matchers has recently become a hot research topic [10, 13, 56]. This raises the question - Could generative AI create face images of non-existent MZ twins that could be used to augment other training data?

To explore this possibility, we attempted to obtain images of a set of imagined twins from each of Grok (v3), Chat-GPT (v5.4) and Gemini (3.1 Pro). The hope is that the images of imagined twins would result in distance distributions for positive and negative image pairs that are similar to those shown in this paper for real MZ twins. The same prompt sequence was used with each of Grok, Chat-GPT and Gemini: “Karrie Bowyer and Karla Bowyer are not real persons, but imagine that they are white females about age 35. Also, and this is important, Karrie Bowyer and Karla Bowyer are monozygotic (identical) twins. Generate (... instructions for images with variation in hairstyle, facial expression and age ...)”.

Results for the twins imagined by Grok are shown in Figure 16. While the Grok images show variation in pose, expression, hairstyle and age, they also reveal serious limitations. One, some images do not plausibly appear to be a real human; e.g., image 4 in the top row for Karla Bowyer. Two, the image pairs that are supposed to be of the same person do not exhibit distance scores reflecting that they are the same person. The distance distribution for sameperson image pairs begins at or above the range of sameperson image pairs for real twins. Three, the distributions for same-person and different-person image pairs are effectively wholly overlapped. In effect, the Grok twins are literally identical rather than having the level of difference that real twins have. Also, the supposed multiple images of one Grok person are not identity-preserving at the level of the persons in the real sets of twins.

![](images/b9d08db489e26342f49cf2c2c3654dd0b911400c8733ae1854f77c68f0ae92e0.jpg)  
Figure 15. Exploring the Effect of Training an Asymmetry-Aware model. Distributions for the Mowry twins, from an ArcFace model trained without horizontal flip augmentation, and distances computed using original image embeddings, rather than averaging with embed ding of horizontal flip of image. Note that the distributions for positive and negative image pairs have an overlap of 11 pairs; compare to distributions in Figure 5. The model trained without horizontal flip augmentation is not able to exploit the skin mark to reduce the overlap in the distributions.

Results for the twins as imagined by ChatGPT are shown in Figure 17. The ChatGPT images show variation in hairstyle and expression, but do not show the range of pose variation or age variation that is evident in the Grok images. Subjectively, all of the ChatGPT images appear that they could be of a real person. The distribution of same-person distance scores lies largely in a plausible range compared to distributions for real twins, although a few image pairs run into higher distances. Again, as with Grok, the distributions of same-person distances and different-person distances are effectively wholly overlapped. So, both Grok and ChatGPT fail to “understand” that MZ twins’ face appearance is not literally identical.

The results from Gemini are categorically different, in that Gemini repeatedly stated that it could not produce images to satisfy the prompt. Gemini’s response to one attempt started out with the explanation - “Given the constraints of my generation process, creating 24 highly realistic, distinct images that maintain perfect identical facial geometry across multiple variations in a single generation is not technically possible. ...” The results of a second attempt gave a similar explanation - “I understand that you are conducting an experiment using AI-generated images of fictional identical twins, Karrie and Karla Bowyer ... I cannot generate the specific images you have requested” and also included the recommendation - “To effectively pursue this experiment on positive and negative pair similarity values, I would recommend utilizing an established, labeled dataset of real human twins.”

These initial attempts at getting images of imagined MZ twins from Grok, ChatGPT and Gemini suggest that there is a fundamental problem yet to be solved. The data that Gen AI tools are trained on does not contain explicit information on the normal or typical ways in which the facial appearance of MZ twins can differ. Setting aside issues of facial marks and asymmetry, real MZ twins still exhibit a range of natural differences in facial appearance, as evident in the examples presented. But what features of the face have what range of natural differences between real MZ twins that allows face matchers to produce well-separated distance distributions for positive and negative image pairs? The answer to this question does not yet exist in the face recognition research literature. Additionally, results in the existing literature such as those in the NIST report may lead to a trained generative AI model that “thinks” that MZ twins have literally identical face appearance.

![](images/dea1e82cd3c067ea4634b906006d4cd04bd3b548c58fbe1bcb51f7959c6b4dc8.jpg)  
(a) Images for Grok-imagined Karla Bowyer

![](images/4c67e0f7a85ad714adb186ec1e4c1c2cba108ec09650e37d458dab2b97a50e29.jpg)

![](images/ec6487ddc8937930f529070dbe399bed351137815771341dbaf69f9b2d8d4b3a.jpg)

![](images/6f2dd73f666d8a899d84a18f10835593f08846d52e0cc5ba9247b3a7ba99c6b4.jpg)

![](images/3f207faf43a979376ee9900963d9a46568277c49245d4834f4038da729eb5df1.jpg)

![](images/556678e7bb9d9e8cd93aee5b1996245cf2064f96eb241ce7de07edce436ffb20.jpg)  
(b) Images for Grok-imagined Karrie Bowyer

![](images/1b807ccdccee7c8aa6557804632d8ecdfff1cbe29ca6231800f821eaa5d74b92.jpg)

![](images/2f5c6988d5a91ca462bf8e27345af30e2e96abf1744322f82b8fb89867509c1d.jpg)  
(c) Distance Distributions of Same-Person (“positive”) and Different-Person (“negative”) Image Pairs  
Figure 16. Images and Distance Distributions for Grok-imagined MZ Twins. The distance distributions for the Grok-imagined MZ twins are unlike those for any of the real twins in CTTS.

![](images/32ca35e9b6d627c607916fe70d3e7a62b33bc8584044106beb71727827e29fe0.jpg)

![](images/0ed0cf00b4618b788d71b717d21146d1b0eed6937bf676c92b980b9a3b22b002.jpg)  
(a) Images for ChatGPT-imagined Karla Bowyer

(b) Images for ChatGPT-imagined Karrie Bowyer  
![](images/63460ad2ed557df67a3523782095dc34145da8ff27b0232fa4663b9d11a65d36.jpg)  
(c) Distance Distributions of Same-Person (“positive”) and Different-Person (“negative”) Image Pairs  
Figure 17. Images and Distance Distributions for ChatGPT-imagined MZ Twins. The distance distributions for the ChatGPT-imagined MZ twins are unlike those for any of the real twins in CTTS.

## 8. Conclusions and Discussion

Distinguishing between MZ twins is challenging ... CTTS accuracy is much lower than that for test sets emphasizing factors such as age difference (AgeDB-30, CALFW), pose difference (CFP-FP, CPLFW), challenging facial hairstyles (Hadrian), or challenging illumination combinations (Eclipse). Distinguishing twins is a current open problem in face recognition. CTTS enables research comparing face matchers based on the accuracy of distinguishing MZ twins, with an ability to focus on skin marks and asymmetry.

... but MZ twin face appearance is not identical. If MZ twins face appearance was literally identical, there would be no separation between the distance distributions for positive and negative image pairs, and the expected accuracy in distinguishing positive and negative image pairs would be 50%. All 12 matcher versions in Table 1 achieve accuracy above 70%. The face embeddings computed by these matchers obviously incorporate features that “see” some difference in the faces of MZ twins. This raises the question – What features are current matchers using to distinguish twins? This is particularly interesting in light of the evidence that they are not using facial marks or asymmetry.

What is the role of facial marks? Past research points to skin marks as a means to distinguish MZ twins. But this research does not use deep CNN face matchers, and uses higher-resolution images than currently used with deep CNNs. Some twins in CTTS-80 are readily distinguishable using facial marks, even at the 112x112 resolution of cropped face images typical of input to deep CNN face matchers. Training deep CNN matchers to make use of such information likely requires some significant fraction of such twins in the training data, but should improve accuracy from current levels.

What role can training set composition play? What training set composition and size is needed to increase accuracy on MZ twins? Augmenting training sets with additional data representing MZ twins could potentially improve accuracy on twins as well as non-twins. However, is it possible or practical to acquire large enough datasets of MZ twins images to meaningfully impact deep CNN training?

Could generative AI create images of imagined twins that are useful for training? Current Gen AI tools are not up to this task. But can new research develop knowledge about the natural variation in MZ twins appearance, and can this knowledge be used in new Gen AI tools to create more realistic image sets of imagined MZ twins? Or, as a more grounded instance of the problem, could generative AI tools produce realistic images of an imagined MZ twin of an existing real person? That is, does the problem become more feasible if it is grounded relative to an image set of a real person?

## Acknowledgements

Thanks to participants in the NSF (CNS-2302070) Research Experiences for Teachers program at the University of Notre Dame who contributed to the twins dataset collection and discussion of twins recognition: Molly Bush, Sarah Clark, Lorenzo Davis, Gloria Linkiewicz, Thomas Martin, Missy Cadotte, Stephanie Modlin, Allison Moore, Sarahi Robertson, Chelsea Thompson and Allyson Tobolski.

## References

[1] Is web scraping legal? what you need to know in 2026. https://www.datashake.com/blog/is-webscraping- legal- what- you- need- to- knowin-2026, 2026. Accessed July 20, 2026. 5

[2] Ayman Afaneh, Fatemeh Noroozi, and Onsen Toygar. Recognition of identical twins using fusion of various facial feature extractors. EURASIP Journal on Image and Video Processing, 2017, 2017. 3, 4

[3] Belal Ahmad, Mohd Usama, Jiayi Lu, Wenjing Xiao, Jiafu Wan, and Jun Yang. Deep convolutional neural network using triplet loss to distinguish the identical twins. In IEEE Globecom Workshops (GC Wkshps), 2019. 4

[4] Nexdata AI. Facial skin defects dataset – 21302 images with acne, wrinkles, and dark circles. https://www. nexdata.ai/datasets/computervision/1052, 2026. 1

[5] Xiang An, Xuhan Zhu, Yuan Gao, Yang Xiao, Yongle Zhao, Ziyong Feng, Lan Wu, Bin Qin, Ming Zhang, Debing Zhang, and Ying Fu. Partial FC: training 10 million identities on a single machine. In International Conference on Computer Vision Workshops (ICCVW), 2021. 9

[6] Bansal Ankan, Carlos D. Castillo, Rajeev Ranjan, and Rama Chellappa. The do’s and don’ts for CNN-based face verification. In International Conference on Computer Vision Workshops (ICCVW), 2017. 3

[7] Ansel Barrett. Web scraping 101: 10 myths that everyone should know. https://www.octoparse.com/blog/ 10-myths-about-web-scraping, 2025. 5

[8] Deniz Bingul. Why are only some identical twins consid ered mirror twins? https://www.thetech.org/ ask- a- geneticist/articles/2025/mirrortwins/, 2025. 2

[9] Soma Biswas, Kevin W. Bowyer, and Patrick J. Flynn. A study of face recognition of identical twins by humans. In IEEE International Workshop on Information Forensics and Security, 2011. 3, 10

[10] Fadi Boutros, Vitomir Struc, Julian Fierrez, and Naser Damer. Synthetic data for face recognition: Current state and future prospects. Image and Vision Computing, 135, 2023. 20

[11] Kevin W. Bowyer, Michael C. King, Walter J. Scheirer, and Kushal Vangara. The “criminality from face” illusion. IEEE Transactions on Technology and Society, 1(4):175– 183, 2020. 4

[12] Khush Jay Brahmbhatt, Krishna Prakasha, and Gangothri Sanil. Facial mark based biometric differentiation of identical twins using dynamic feature enhancement. Scientific Reports, 2026. 3, 5, 10

[13] Ivan DeAndres-Tame, Ruben Tolosana, Pietro Melzi, Ruben Vera-Rodriguez, Minchul Kim, Christian Rathgeb, Xiaoming Liu, Aythami Morales, Julian Fierrez, Javier Ortega-Garcia, Zhizhou Zhong, Yuge Huang, Yuxi Mi, Shouhong Ding, Shuigeng Zhou, Shuai He, Lingzhi Fu, Heng Cong, Rongyu Zhang, Zhihong Xiao, Evgeny Smirnov, Anton Pimenov, Aleksei Grigorev, Denis Timoshenko, Kaleb Mesfin Asfaw, Cheng Yaw Low, Hao Liu, Chuyi Wang, Qing Zuo, Zhixiang He, Hatef Otroshi Shahreza, Anjith George, Alexander Unnervik, Parsa Rahimi, Sebastien Marcel, Pe-´ dro C. Neto, Marco Huber, Jan Niklas Kolf, Naser Damer, Fadi Boutros, Jaime S. Cardoso, Ana F. Sequeira, Andrea Atzori, Gianni Fenu, Mirko Marras, Vitomir Struc, Jiang Yu,<sup>ˇ</sup> Zhangjie Li, Jichun Li, Weisong Zhao, Zhen Lei, Xiangyu Zhu, Xiao-Yu Zhang, Bernardo Biesseck, Pedro Vidal, Luiz Coelho, Roger Granada, and David Menotti. Second edition FRCSyn challenge at CVPR 2024: Face recognition challenge in the era of synthetic data. In IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2024. 20

[14] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4690–4699, 2019. 9

[15] Pamela P. Fierro and Alyssa Dweck. What parents need to know about mirror twins. https://www.parents. com/mirror-twins-8665384, 2024. 2

[16] Kayee Hanaoka, Mei Lee Ngan, Patrick J. Grother, and Austin Hom. Ongoing Face Recognition Vendor Test (FRVT) part 9a: Face recognition verification accuracy on distinguishing twins, 2022. 1, 2, 5, 10

[17] Chih-Chung Hsu and Pi-Ju Tsai. Identical twins verification with fine-grained recognition. In 2023 International Conference on Consumer Electronics - Taiwan (ICCE-Taiwan), 2023. 4

[18] Junlin Hu, Jiwen Lu, and Yap-Peng Tan. Fine-grained face verification: Dataset and baseline results. In 2015 International Conference on Biometrics (ICB), 2015. 4

[19] Gary B Huang, Marwan Mattar, Tamara Berg, and Eric Learned-Miller. Labeled faces in the wild: A database for studying face recognition in unconstrained environments. In Workshop on Faces in ’Real-Life’Images: Detection, Alignment, and Recognition, 2008. 2

[20] Felix Juefei-Xu and Marios Savvides. An augmented linear discriminant analysis approach for identifying identical twins with the aid of facial asymmetry features. In IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2013. 3, 4, 5

[21] Minchul Kim, Anil K. Jain, and Xiaoming Liu. Adaface: Quality adaptive margin for face recognition. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18729–18738, 2022. 9

[22] Amar Kirale and Veena Shinde. What are mirror twins - amazing facts through the looking glass. https : / / parentingnmore . com / what - are - mirror - twins-through-the-looking-glass/, 2020. 2

[23] Brendan Klare, Alessandra A. Paulino, and Anil K. Jain. Analysis of facial features in identical twins. In 2011 International Joint Conference on Biometrics (IJCB), 2011. 3, 4, 5

[24] T. Hoang Ngan Le, Khoa Luu, Keshav Seshadri, and Marios Savvides. A facial aging approach to identification of identical twins. In 2012 IEEE Fifth International Conference on Biometrics: Theory, Applications and Systems (BTAS), 2012. 3, 4, 10

[25] T. Hoang Ngan Le, Keshav Seshadri, Khoa Luu, and Marios Savvides. Facial aging and asymmetry decomposition based approaches to identification of twins. Pattern Recognition, 48(12):3843–3856, 2015. 4, 5, 10

[26] Miguel Bordallo Lopez, Elhocine Boutellaa, and Abdenour´ Hadid. Comments on the “kinship face in the wild” data sets. IEEE Transactions on Pattern Analysis and Machine Intelligence, 38(11):2342–2344, 2016. 4

[27] Gayathri Mahalingam and Karl Ricanek. Investigating the effects of gender and age group based differences in identical twins. In Fourth National Conference on Computer Vision, Pattern Recognition, Image Processing and Graphics (NCVPRIPG), 2013. 3, 4

[28] Gayathri Mahalingam and Karl Ricanek. LBP-based peri ocular recognition on challenging face datasets. EURASIP Journal on Image and Video Processing, 2013. 3

[29] Matteo Martini, Ilaria Bufalari, Maria Antonietta Stazi, and Salvatore Maria Aglioti. Is that me or my twin? Lack of selfface recognition advantage in identical twins. PLoS One, 10 (4), 2015. 3

[30] Brianna Maze, Jocelyn Adams, James A. Duncan, Nathan Kalka, Tim Miller, Charles Otto, Anil K. Jain, W. Tyler Niggel, Janet Anderson, Jordan Cheney, and Patrick Grother. Iarpa janus benchmark - c: Face dataset and protocol. In In ternational Conference on Biometrics (ICB), 2018. 5

[31] MedlinePlus. Is the probability of having twins determined by genetics? https://medlineplus.gov/ genetics / understanding / traits / twins/, 2022. 1

[32] Firm Memoranda. The legal landscape of web scraping. https : / / www . quinnemanuel . com / the - firm/publications/the- legal- landscapeof-web-scraping/, 2023. 5

[33] Qiang Meng, Shichao Zhao, Zhida Huang, and Feng Zhou. Magface: A universal representation for face recognition and quality assessment. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14225– 14234, 2021. 9

[34] Stylianos Moschoglou, Athanasios Papaioannou, Christos Sagonas, Jiankang Deng, Irene Kotsia, and Stefanos Zafeiriou. Agedb: The first manually collected, in-the-wild age database. In IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 1997– 2005, 2017. 2

[35] Shokoufeh Mousavi, Mostafa Charmi, and Hossein Hassanpoor. A distinctive landmark-based face recognition system for identical twins by extracting novel weighted features. Computers & Electrical Engineering, 94, 2021. 4, 5, 10

[36] National Organization of Mothers of Twins Club. Handedness in multiples survey. https : / / multiplesofamerica . org / wp - content / uploads / 2014 / 12 / RR12 - Handedness - in - Multiples.pdf, 1987. 2

[37] Kagan Ozturk, Haiyu Wu, and Kevin W. Bowyer. Can the accuracy bias by facial hairstyle be reduced through balancing the training data? In IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2024. 20

[38] Jeffrey R. Paone, Patrick J. Flynn, P. Jonathon Philips, Kevin W. Bowyer, Richard W. Vorder Bruegge, Patrick J. Grother, George W. Quinn, Matthew T. Pruitt, and Jason M. Grant. Double trouble: Differentiating identical twins by face recognition. IEEE Transactions on Information forensics and Security, 2014. 1, 3, 4

[39] Connor J. Parde, Virginia E. Strehle, Vivekjyoti Banerjee, Ying Hu, Jacqueline G. Cavazos, Carlos D. Castillo, and Alice J. O’Toole. Twin identification over viewpoint change: A deep convolutional neural network surpasses humans. ACM Transactions on Applied Perception, 20(3), 2023. 3

[40] P Jonathon Phillips, Patrick J Flynn, Kevin W Bowyer, Richard W Vorder Bruegge, Patrick J Grother, George W Quinn, and Matthew Pruitt. Distinguishing identical twins by face recognition. In 2011 IEEE International Conference on Automatic Face & Gesture Recognition (FG), 2011. 3

[41] Matthew T Pruitt, Jason M Grant, Jeffrey R Paone, Patrick J Flynn, and Richard W Vorder Bruegge. Facial recognition of identical twins. In 2011 International Joint Conference on Biometrics (IJCB). IEEE, 2011. 3

[42] Shoaib Meraj Sami, John McCauley, Sobhan Soleymani, Nasser Nasrabadi, and Jeremy Dawson. Benchmarking human face similarity using identical twins. IET Biometrics, 11 (5):459–484, 2022. 4, 5

[43] Soumyadip Sengupta, Jun-Cheng Chen, Carlos Domingo Castillo, Vishal M. Patel, Rama Chellappa, and David W. Jacobs. Frontal to profile face verification in the wild. In IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2016. 2

[44] Eliabeth S. Shalin, Bino Thomas, and Jubilant J. Kizhakkethottam. Analysis of effective biometric identification on monozygotic twins. In 2015 International Conference on Soft-Computing and Networks Security (ICSNS), 2015. 3, 4

[45] Dorian Smith-Garcia and Carolyn Kay. Do all identical twins have the exact same DNA? https : / / www.healthline.com/health/do-identicaltwins-have-the-same-dna, 2021. 2

[46] Nisha Srinivas, Guarav Aggarwal, Patrick J. Flynn, and Richard W. Vorder Bruegge. Analysis of facial marks to distinguish between identical twins. IEEE Transactions on Information forensics and Security, 7(5):1536–1550, 2012. 3, 4, 10

[47] SI Staff. Notable twins in sports. https://www.si. com/more-sports/2012/10/05/05-0notabletwins-in-sports, 2012. 5

[48] Sarah V. Stevenage. Which twin are you? A demonstra tion of induced categorical perception of identical twin faces. British Journal of Psychology, 89:39–57, 1998. 3

[49] K. Sudhakar and P. Nithyanandam. Facial identification of twins based on fusion score method. Journal of Ambient Intelligence and Humanized Computing, 2021. 3, 4

[50] Zhenan Sun, Alessandra A Paulino, Jianjiang Feng, Zhenhua Chai, Tieniu Tan, and Anil K Jain. A study of multibiometric traits of identical twins. In Biometric Technologyfor Human Identification Vii. SPIE, 2010. 3, 4

[51] Vinusha Sundaresan and S. Amala Shanthi. Monozygotic twin face recognition: An in-depth analysis and plausible improvements. Image and Vision Computing Journal, 116, 2021. 4, 5, 10

[52] Sicong Tian, Haiyu Wu, Michael C. King, and Kevin W. Bowyer. Impact of sunglasses on one-to-many facial identification accuracy. IEEE Transactions on Biometrics, Behavior, and Identity Science, 8(3):461–476, 2026. 20

[53] Unidata. Facial skin condition dataset. https : / / unidata . pro / datasets / facial - skin - condition-image-dataset/, 2026. 1

[54] Zack Whittaker. Web scraping is legal, us appeals court reaf firms. https://techcrunch.com/2022/04/18/ web-scraping-legal-court/, 2022. 5

[55] Jeremy Wilmer, Laura Germine, Christopher Chabris, Garga Chatterjee, Mark Williams, Eric Loken, Ken Nakayama, and Brad Duchaine. Human face recognition ability is specific and highly heritable. Proceedings of the National Academy ofSciences, 107:5238–5241, 2010. 3

[56] Haiyu Wu, Jaskirat Singh, Sicong Tian, Liang Zheng, and Kevin W. Bowyer. Vec2face: Scaling face dataset generation with loosely constrained vectors. In International Confer ence on Learning Representations, 2025. 20

[57] Haiyu Wu, Sicong Tian, Aman Bhatta, Jacob Gutierrez, Grace Bezold, Genesis Argueta, Karl Ricanek, Michael C. King, and Kevin W Bowyer. Goldilocks test sets for face verification. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026. 1, 2

[58] Kimberly Zapata and Carolyn Kay. What are mirror twins? here’s everything you want to know. https://www. healthline.com/health/pregnancy/mirrortwins#identification, 2020. 2

[59] Tianyue Zheng and Weihong Deng. Cross-pose LFW: A database for studying cross-pose face recognition in unconstrained environments. Beijing University of Posts and Telecommunications, Tech. Rep, 5(7), 2018. 2

[60] Tianyue Zheng, Weihong Deng, and Jiani Hu. Crossage LFW: A database for studying cross-age face recog nition in unconstrained environments. arXiv preprint arXiv:1708.08197, 2017. 2

[61] Jiancan Zhou, Xi Jia, Qiufu Li, Linlin Shen, and Jinming Duan. Uniface: Unified cross-entropy loss for deep face recognition. In ICCV, pages 20730–20739, 2023. 9

[62] Zheng Zhu, Guan Huang, Jiankang Deng, Yun Ye, Junjie Huang, Xinze Chen, Jiagang Zhu, Tian Yang, Dalong Du, Jiwen Lu, and Jie Zhou. Webface260m: A benchmark for million-scale deep face recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(2):2627– 2644, 2023. 9, 20