# Towards Zero-Shot Transfer Across Embodiments For Driving VLAs

Caio Azevedo<sup>1,2</sup> Stefano Sabatini<sup>2</sup> Sascha Hornauer<sup>1</sup> Fabien Moutarde<sup>1</sup>

<sup>1</sup>Ecole des Mines de Paris, France <sup>´</sup> <sup>2</sup>Stellantis, France

caio.azevedo@minesparis.psl.eu

## Abstract

Vision-Language-Action models (VLAs) have shown strong potential in autonomous driving by leveraging multimodal pretraining for instruction following, visual reasoning, and scene-level generalization. In robotic manipulation, scaling VLAfine-tuning across multiple robot setups–especially when unifying representations across embodiments–has been shown to improve in-dataset performance and crossembodiment generalization; in autonomous driving, however, VLAs remain largely trained on individual datasets and are rarely evaluated for zero-shot transfer to unseen datasets and camera rigs;furthermore naively adding more datasets to the training data does not necessarily lead to better performance within seen embodiments. To address these problems, we study multi-dataset trainingfor the driving task and BEV-Forcing, an auxiliary objective that transfers ground-plane object-layout informationfrom a specialized Bird’s-Eye-View model into the VLA backbone. By encouraging the model to represent object position through a shared BEV spatial interface, we show that an auxiliary task such as BEV-Forcing can improve both in-distribution and out-of-distribution performance when training on a small number of camera rigs. As the number of training embodiments increases, however, the benefits of the auxiliary task are reduced; we present this as evidence that new techniques in the literature may see their benefits diminish when simply scaling up training diversity, which motivates presenting results taking into account data scaling. Code available in github.com/caiocj1/ad-vla.

## 1. Introduction

Recent progress in the capabilities of vision-language models (VLMs) has opened up a promising new direction of research for end-to-end autonomous driving, namely, how to properly adapt VLM backbones into good driving visionlanguage-action models (VLAs). This problem has piqued the interest of the research community for several reasons:

first, the large-scale multi-modal pretraining of the backbone provides models with semantic priors that can help in correctly interpreting the driving environment based on image content alone, even in rare or unseen long-tail scenes; they also are able to provide reasoning traces that can ground actions on the reality of the traffic scene, which in turn provides explainability and can in principle improve performance in hard scenarios; finally, they allow for human-machine interaction through their acquired understanding of natural language [17].

In practice, the incorporation of pre-trained language backbones into embodied tasks has been shown, in both robotics [1, 18, 47] and driving [15, 28], to lead to both better in-dataset performance and better semantic transfer than previous approaches, i.e. retaining performance over visual or scene-content distribution shifts, such as when novel objects appear in the scene or when a model is trained or finetuned in a certain location and evaluated on another using the same sensor rig. Additionally, modern fine-tuning techniques further improve performance on out-of-distribution tasks on the same embodiment, e.g. by representing actions [10] or by adaptively fine-tuning parameters [14] in such a way as to avoid catastrophic forgetting.

However, when VLAs are adapted for robotic actions such as driving, a problem arises: as the model is fine-tuned to output future actions, it might acquire the ability to spatially interpret the information from the sensor rig of the particular dataset it is being fine-tuned on, but catastrophically fail when asked to output spatially grounded trajectories on another dataset collected with a different camera setup. In other words, VLAs lack geometric transfer: although the backbone may semantically understand the contents of the image from the new dataset, 3D spatial understanding does not necessarily transfer, since it is learned implicitly from a low number of sensor rigs and from much fewer examples than are present in the pretraining of the backbone.

In robotics, it has been shown that pre-training on a wide variety of datasets from different embodiments alleviates this problem, while overall still demanding fine-tuning on the target setup [7, 18, 47]. In autonomous driving, efforts to unify the available open end-to-end driving datasets are still very recent [4, 6]. They also show the potential of training on multiple datasets, however most works remain trained on a single driving dataset and not all training data compositions in terms of embodiments lead to improvements, while further fine-tuning is still required for good performance when transferring to unseen camera rigs. The question of what makes a model good at cross-embodiment zero-shot transfer, in particular for autonomous driving, remains underexplored; this is an important problem since better zero-shot transfer allows for more effective adaptation into new embodiments and therefore facilitates prototype development and iteration.

![](images/58088a88b4520256503130f4b2c39e38332d7f22bb359b444f4409f509a117a1.jpg)  
Figure 1. Illustration of BEV-Forcing, performing supervised fine-tuning (SFT) on a planning sample (analogous for VQA samples). After unifying various end-to-end datasets into a common format, we extract teacher occupancy maps from a specialized BEV model, SimpleBEV [11] in our case. These are used as the ground-truth for BEV-Forcing, which supervises last-layer image hidden states with extracted teacher maps, transferring ground plane object position awareness into the backbone.

Given this context, we wish to investigate the behavior and performance of driving VLAs, both within seen camera rigs and crucially at zero-shot transfer, when we scale the number oftraining embodiments. In particular, we hypothesize that improved zero-shot transfer may arise if the model is able to learn a unified spatial representation from multiview camera images coming from varying rigs. To test this, we also study combining scaling training embodiments with an auxiliary objective related to spatial awareness.

In short, we propose the following contributions:

• BEV-Forcing, an auxiliary task inspired by Spatial Forcing [19], which can improve both in and out of dataset performance when training on a limited number of camera rigs. This incurs no extra cost at inference time since the BEV auxiliary prediction head can simply be removed in a production environment, and is scalable since it does not require annotated maps.

• We provide our observations when scaling up the training of driving VLAs with multiple public datasets, including planning and visual-question-answering (VQA) tasks, evaluating performance in both seen embodiments and at zero-shot transfer with and without BEV-Forcing and several different training configurations. We show how applying BEV-Forcing when training on few embodiments can lead to gains comparable to those obtained by adding extra training embodiments.

• Finally, we provide a series of ablations to better understand the important elements of the auxiliary task and how it interacts with other techniques aimed at increasing robustness such as performing image augmentations or passing calibration parameters as inputs.

## 2. Related Work

Spatial awareness for VLAs. A VLA needs to acquire the knowledge of how to properly navigate a 3D environment, since this knowledge is not sufficiently acquired during the large-scale pretraining of its backbone. For this reason, a variety of approaches try to teach 3D awareness to VLAs. In many works in autonomous driving, this is learned implicitly [32, 38, 46]. This is enough for strong in-dataset performance results, though edge cases still provide challenges which motivate the development of explicit spatial awareness learning; additionally, performance when transferring to a new embodiment is usually not evaluated.

Passing LiDAR-to-camera aligned projection images as an extra input to the backbone has been shown to improve question-answering accuracy on driving scenes [9]. On planning, other approaches also incorporate extra modalities into the model: (i) with 3D data supervision as an auxiliary objective, either of geometry modules as extra parameters [8, 41] or of the backbone itself, but this significantly increases data collection requirements; (ii) by treating the new modalities as extra inputs that contain explicit spatial information that the model learns to use after multi-stage training [39, 45], but this does not take full advantage of the strong vision-language alignment of current natively multi-modal models [31]; (iii) by taking advantage of target features that can instill spatial awareness into the model, whether from 3D foundation models such as VGGT [36] and used as direct supervision targets [19, 25], or from trained dynamicsaware future image predictors [33], or still by conditioning on VGGT features with cross-attention [37].

In our work, we decide to instill spatial awareness using camera inputs only and no 3D data label requirement: first, to fully leverage the vision-language alignment of recent VLM backbones without extra training stages; second, due to the scarceness and cost of collection of 3D data such as LiDAR: important datasets such as WOD-E2E [40] only provide camera images. Our approach aligns most with category (iii), and in particular with Spatial Forcing [19]. We decide however to use as targets for representation supervision occupancy maps output by specialized BEV models, as we hypothesize that these may contain a more uniform and more task aligned spatial interface across embodiments compared to 3D depth-map reconstruction related features.

This decision is justified in Section 4.3.

VLA generalization. Many works, especially in robotic manipulation, have demonstrated and further deepened the capability of VLAs for semantic zero-shot transfer, i.e. evaluating a model on an embodiment it has seen during training but in a new environment or task. In particular, large-scale pretraining on a variety of embodiments is of fundamental importance [1, 18, 27, 43]. Recent techniques such as adapting the constraint on parameter updates depending on model depth [14], casting robotic tasks as pure natural language [10], among many others [22] help retain the knowledge of the backbone’s pretraining leading to better semantic transfer.

With respect to geometric or cross-embodiment zeroshot transfer, there is a relative lack of literature in autonomous driving, though recent efforts are trying to cover this gap [4, 6, 35]. Previous works in autonomous driving have shown that world modeling through video generation aligned with actions unlocks new scaling laws and benefits zero-shot transfer across datasets [23, 24], but this leads to high computational requirements both at training and at inference time. In robotic manipulation, the emphasis is either on learning embodiment specific adapters [43] or on the necessity of unifying observations and action representations across many embodiments and tasks as a means of transferability [30, 42].

Autonomous driving presents its own set of both challenges and advantages compared to robotic manipulation: while there are fewer large-scale public datasets to provide camera setup variability, having a unified task with similar outputs across embodiments (planned trajectories) should in principle facilitate scaling up training to multiple datasets with positive transfer, independently of the particular action representation being used by the backbone. For these reasons, in our work we settle on a simple textual action representation (which is nevertheless argued for in [10]) and focus on investigating a representation learning technique that can improve zero-shot transfer when training on a limited number of camera rigs and with no extra inference cost, as well as investigating how this technique behaves as we train on more embodiments.

## 3. Methodology

In this section we explain in detail the architecture and design choices we made for instilling spatial awareness with BEV-Forcing according to Fig. 1, as well as how we perform multi-dataset training.

## 3.1. Backbone and Action Representation

We start from a pretrained open-source vision-language model, Qwen 3.5 2B [31]. In our work we decide to focus on good spatial representation learning as a means of transferability. Because of this, for action representation we simply use text outputs representing trajectory waypoints at 1 Hz as pairs of 2D coordinates, following [32]. These are then parsed and resampled with cubic splines towards the target sampling rate. Consistent with [10], since the use of text for action representation brings the task closer to the original pretraining of the backbone, we find that applying LoRA [12] to avoid catastrophic forgetting leads to better performance than full fine-tuning of the backbone.

![](images/ee7b9f348510a358f1b4f6cf59687a59ae1b4f78a5f4e9a6b27b78e4da67021c.jpg)  
Figure 2. Illustration of the BEV head. Randomly initialized queries attend to image embeddings to output occupancy maps of vehicles in the scene. The low capacity of the BEV head forces image features that are used for trajectory planning to contain information related to surrounding objects’ position in space.

## 3.2. BEV-Forcing

Similar to [19], which supervises image embeddings with features from a specialized 3D model for more precise robotic manipulation, we aim to instill in the image embeddings within the model an awareness of the spatial representation of the scene, in a way that does not require extra computational costs at inference time. For this, we propose BEV-Forcing as described in Fig. 1: we add a BEV head as shown in Fig. 2, that first initializes queries $q \in \mathbb { R } ^ { ( N _ { X } \cdot N _ { Y } ) \times d }$ , with $N _ { X } \times N _ { Y }$ being the resolution of the spatial grid and d the hidden dimension of the BEV head. We extract image embeddings $f _ { I } ^ { ( \ell ) } \in \mathbb { R } ^ { N _ { I } \times D }$ at a target layer ℓ of the backbone, then project these with an MLP to the query dimension d, $N _ { I }$ being the number of image tokens and D the dimension size of the backbone. Then the queries cross-attend to the projected image embeddings.

Finally, the post-attention features $\hat { b }$ are projected to occupancy logits at each position through a simple linear layer:

$$
\tilde { f } _ { I } = \mathbf { M L P } ( f _ { I } ^ { ( \ell ) } ) ,\tag{1}
$$

$$
\hat { b } = \mathrm { A t t n } ( W _ { Q } \cdot q , W _ { K } \cdot \tilde { f } _ { I } , W _ { V } \cdot \tilde { f } _ { I } ) .\tag{2}
$$

$$
{ \hat { \mathcal { F } } } _ { \mathrm { B E V } } = \operatorname { L i n e a r } ( { \hat { b } } ) .\tag{3}
$$

By limiting the capacity of the BEV auxiliary head that is added on to the backbone (the architecture and parameters in Eq. 1 to 3), by for example simply using a cross-attention layer instead of a full transformer block, we ensure that the representation of the image tokens $f _ { I } ^ { ( \ell ) }$ acquires as much information about the spatial disposition of objects in the scene as possible, which then can help in planning performance as will be shown in Section 4. During inference, the BEV head can simply be removed, leading to no extra computational cost.

## 3.3. Training

The predicted $\hat { \mathcal { F } } _ { \mathrm { B E V } }$ grid is then supervised with a binary cross entropy loss $\mathcal { L } _ { \mathrm { B E V } }$ with targets coming from the occupancy map $\mathcal { F } _ { \mathrm { { S i m p l e B E V } } }$ that is output by a specialized BEV model, SimpleBEV in our case [11]. SimpleBEV’s architecture allows it to output reasonably high-quality occupancy maps even for datasets it was not trained on, and using a BEV expert model gives us a training signal in important datasets which do not have ground-truth maps such as WOD-E2E [40]. For loss calculation, positive grid positions have higher weight $w _ { + }$ to compensate data imbalance.

L<sub>BEV</sub> is then weighted by a factor λ and added to standard next-token prediction (NTP) loss during supervised fine-tuning, where the target sequence is simply the groundtruth future trajectory in text tokens:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { N T P } } + \lambda \cdot \mathcal { L } _ { \mathrm { B E V } }\tag{4}
$$

To make sure that the model does not lose its linguistic capabilities and semantic knowledge, we also do runs with visual-question-answering (VQA) co-training. For VQA samples, the loss is obtained in the same manner: we calculate the next-token prediction loss using the expected answer for each question, while the target layer’s image hidden states are passed to the BEV head to produce L<sub>BEV</sub>.

## 3.4. Dataset Standardization

We load samples from each dataset in a unified contract that easily allows combining multiple datasets. In particular, for training on planning samples we require past trajectories, current timestep images from the three camera images and high-level ego-vehicle intent as inputs, as well as the target future trajectory for supervised fine-tuning; for VQA samples, we pass as inputs all surrounding camera images at the current timestep and supervise on the single-word answer according to the format of nuScenes-QA [29]. For both tasks, a natural language prompt (available in Supplementary Material) specifies the task and output format. Dataset interfaces are available in the code repository.

## 4. Experiments

To recapitulate, we wish to study the behavior of a driving VLA as the number of training embodiments scales, on seen embodiments but more importantly at zero-shot transfer to a new embodiment. Additionally, we investigate the effect of applying BEV-Forcing across training configurations. Here we present and discuss our experiments.

## 4.1. Experimental Setup

Datasets and Metrics. For our experiments with planning samples we use WOD-E2E [40], NAVSIM [3, 5], nuScenes [2], and a subset of six thousand clips of Physical AI [26] due to computational constraints. Each dataset provides camera images at the current timestep, past ego-vehicle states, and for some of the datasets the high-level intent of the ego-vehicle. For VQA we use nuScenes-QA [29].

Our goal with using Physical AI specifically is for zeroshot evaluation and comparison with a model trained directly on Physical AI. For this, we randomly sample five thousand clips for training and one thousand for validation, each clip consisting of 20 seconds of video and other sensor data as well as the features mentioned above. Selected clips are available in code repository for reproducibility. The validation set is used as the target for zero-shot evaluation, while we use the five thousand clip training subset only in Fig. 3, i.e. so we can compare our models in zeroshot transfer with ones trained on Physical AI; there, $5 \mathrm { k } \times 9$ means training on 9 timesteps extracted at regular intervals from each of the five thousand clips, and analogously for 21. We also evaluate zero-shot on the test set KIT LongTail Dataset [35], which was released for a limited time specifically for zero/few shot evaluation with no training set available.

With regards to metrics, we use average and final displacement error (ADE/FDE), as well as the rater feedback score (RFS) on WOD-E2E [40] and the multi-maneuver score (MMS) on the KITScenes LongTail dataset [35]. ADE/FDE simply measure the average and final distance between prediction and recorded ground-truth planned trajectory respectively (lower is better). Both RFS and MMS are calculated based on whether predictions match with human annotated trajectories with assigned scores that take into account the trajectory’s quality. For both RFS and MMS, higher is better.

Implementation Details. We use LoRA with rank 16, a learning rate of $1 0 ^ { - 4 }$ with cosine decay, and train for one epoch on all experiments, using global batch size 64 across 4 A100 GPUs. For each planning dataset, we pass the three front cameras as an input to the model for the planning task; for VQA we use all camera images, always downsizing to lower resolutions to reduce VRAM requirements while keeping the aspect ratio close to the original images; every image transformation is accompanied by its corresponding camera calibration parameters transformation. To extract teacher occupancy maps, we use higher resolution images to maximize the performance of SimpleBEV. Training runs all use 16 past time step trajectories at 4 Hz. We take ℓ as the last layer of the VLM backbone. More architectural details are present in the Supplementary Material.

![](images/5e7dfe750dc2faedb099975d3ac2bc61f41d2a2564bef48bec070bc7e0d4e19b.jpg)  
Figure 3. Performance on PhAI-1k validation set after training on different dataset combinations, with and without BEV task. Training on WOD-E2E only, the BEV task gives large improvements on zero-shot transfer. Introducing NAVSIM and subsequently nuScenesQA into training data also substantially helps zero-shot transfer, with benefits overlapping with those of BEV-Forcing as training variability increases.

## 4.2. Experimental Results

BEV-Forcing helps zero-shot transfer when training on few embodiments. First, we progressively increase the number of embodiments across separate training runs, from WOD-E2E only to then including NAVSIM and finally nuScenes-QA (we discuss the importance of including the VQA task below); we then compare for each training set of camera rigs, the performance with and without the BEV task. We evaluate zero-shot transfer planning performance using a thousand validation clips of Physical AI and on KITScenes LongTail. We observe that the BEV task can serve as a geometry-aware regularizer: it is able to provide large improvements when training on a limited number of camera rigs, with improvements diminishing as training-rig diversity increases. For instance, as seen in Fig. 3, applying the BEV loss while training on only WOD-E2E provides a 10.1% decrease in ADE, an improvement which is seen in qualitative cases such as Fig. 4. Performance however comes closer to the baseline as we add datasets, suggesting that training diversity and BEV supervision may provide partially overlapping benefits.

![](images/efcec660ab50392f20eb252e4c3b22ddb4fb207c3d8acd269600c30a65d075b0.jpg)

![](images/986d89d0e65fc7e9e1899dda03f6fc459af01c148eeedf87120e6a0621437764.jpg)

![](images/5214b381d61094cacc6837e2f29845173499366b3c7d060f2ef0fd637567c770.jpg)  
Figure 4. Qualitative case on a validation sample of Physical AI, of our model trained only on WOD-E2E, without the BEV task above and with it below. BEV-Forcing can adequately provide spatial information that allows the model to perform significantly better at zero-shot transfer.

The same phenomenon is seen on the validation set of KITScenes LongTail, using an internal reproduction of the MMS as described in its paper [35]. Training on WOD-E2E only, we observe an MMS of only 3.90 when not using the BEV task, which improves to 4.64 when adding it. However, as the number of training embodiments increases, data variability overshadows the effect of the BEV task: when training on WOD-E2E + NAVSIM + nuScenesQA, not using the BEV task gives an MMS of 5.13 while using it gives 4.84. For comparison with other methods, we evaluate our models on the test set of KITScenes. As seen in Table 1, our model trained only on WOD-E2E + NAVSIM, a relatively limited amount of variability, achieved an MMS of 5.15 on the test set when using the BEV task, an improvement over the baseline without the BEV task of 4.62 MMS. Our models outperform previous frontier models such as Gemini 3 Pro at zero-shot transfer, showing the need for specializing on domain data for properly navigating the environment; they also outperform solutions based on strong models such as Alpamayo 1.5 [38], which achieved 4.31 MMS. These results indicate that once again, BEV-Forcing offers useful signal for zero-shot transfer when training on a limited number of camera rigs.

BEV-Forcing helps in-dataset performance when training on a single embodiment. Any auxiliary task aimed at improving zero-shot transfer must also not considerably degrade in-dataset performance. Due to constraints on submissions to the larger test set, we first evaluate all training configurations on the RFS-labeled validation split of WOD-E2E. Across these runs, BEV-Forcing consistently maintains or improves planning performance relative to the corresponding baseline. For example, when training on WOD-E2E only, FDE decreases from 5.946 to 5.831 by applying the BEV task, while RFS slightly increases from 7.961 to 7.971. Similar improvements are observed for the more diverse training configurations, with the best validation result using the BEV task reaching 5.687 FDE and 8.119 RFS, compared to 5.851 and 8.063 respectively for the baseline. These results indicate that the auxiliary spatial supervision does not introduce an apparent in-dataset performance trade-off on the validation split. We next evaluate selected configurations on the WOD-E2E test set to assess whether this behavior persists under the more reliable heldout evaluation.

<table><tr><td>Method</td><td>MMS</td><td>L2 (mean)</td></tr><tr><td>UniAD [13] DMAD [34]</td><td>3.24</td><td>10.90</td></tr><tr><td rowspan="3">Alpamayo 1.5 [38]-based† VLA &amp; Kinematics-based†</td><td>3.51</td><td>10.04</td></tr><tr><td>4.31</td><td>3.17</td></tr><tr><td>4.31</td><td>3.57</td></tr><tr><td>Gemini 3 Pro</td><td>4.61</td><td>2.99</td></tr><tr><td>Ours [W+N]</td><td>4.62</td><td>2.63</td></tr><tr><td>Ours [W+N] (+BEV task)</td><td>5.15</td><td>2.48</td></tr></table>

Table 1. Performance of various methods on zero/few-shot evaluation at the test split of the KITScenes LongTail Dataset [35]. “[W+N]” means our models are trained on WOD-E2E + NAVSIM. <sup>†</sup>Descriptions inferred from test set entry titles.

For this, we submit the models trained on WOD-E2E and on WOD-E2E + NAVSIM + nuScenes-QA. Results are in Table 2. First, notice that our overall performance is roughly consistent with the performance of the SFT-only Poutine model, which is expected due to the similar architectures. Next, notice that when training on a single embodiment, i.e. WOD-E2E, the BEV task is able to improve performance, while its effectiveness decreases when we add more diverse camera rigs. With both zero-shot and in-dataset results, an important take away is that techniques aiming at improving end-to-end planning can be useful on single-dataset training (the standard procedure across many papers), while in principle being susceptible to have their effectiveness decreased when training on more embodiments. We stress therefore the importance of evaluating techniques trained with multiple camera rig setups, including at zero-shot.

On the importance of linguistic data in training. We wish to provide further evidence than is present in the literature of autonomous driving of the fact that keeping linguistic supervision is important for good performance when adapting pretrained VLMs. For this, we notice that as we add embodiments to the training set through the addition of new public datasets, we increase the degradation of the backbone in its linguistic and semantic reasoning capabilities: models trained on WOD-E2E and on WOD-E2E + NAVSIM, both without the BEV task, on the validation set of nuScenes-QA [29] achieve 11.1% and 7.0% accuracy respectively, while the base model Qwen 3.5 2B achieves 29.2%. This could be one factor leading to worse in-dataset performance in terms of FDE when adding NAVSIM to the training data: 5.946 to 6.019 FDE on the validation set of WOD-E2E. Then, after also adding nuScenes-QA (around 10% of training samples), we recover on FDE (5.851), and multi-dataset training with the BEV task is capable of achieving 8.119 RFS, a value close to 8.13 which is achieved by ground-truth trajectories [32]. The improved reasoning quality of the resulting model is verified not only through a nuScenes-QA evaluation of 60.2% accuracy, which is expected to be high by training on a specific question format, but also through qualitative checks on traffic scene descriptions, available in the Supplementary Material.

<table><tr><td>Method</td><td>RFS</td><td>ADE</td></tr><tr><td>IRL-VLA [16]</td><td>7.890</td><td>2.823</td></tr><tr><td>Poutine (SFT-only) [32]</td><td>7.909</td><td>2.941</td></tr><tr><td>Poutine [32]</td><td>7.986</td><td>2.742</td></tr><tr><td>NTR [20]</td><td>8.046</td><td>2.638</td></tr><tr><td>DriveMA-2B [44]</td><td>8.075</td><td>2.662</td></tr><tr><td>Ours [W]</td><td>7.873</td><td>2.898</td></tr><tr><td>Ours [W] (+BEV task)</td><td>7.902</td><td>2.891</td></tr><tr><td>Ours [W+N+nSQA]</td><td>7.939</td><td>2.833</td></tr><tr><td>Ours [W+N+nSQA] (+BEV task)</td><td>7.902</td><td>2.938</td></tr></table>

Table 2. Performance of various approaches on the test set of WOD-E2E [40]. “[W]” and “[W+N+nSQA]” mean our models are trained on WOD-E2E alone and on WOD-E2E + NAVSIM + nuScenes-QA respectively. Underlined metrics correspond to our model’s best configuration given the same training datset.

## 4.3. Ablation Studies

In this subsection we study what elements are essential for maximal BEV-Forcing effectiveness and how this auxiliary task interacts with other simple robustness techniques.

How does BEV-Forcing interact with common robustness techniques? Since we train on a limited number of datasets, we hypothesize that the model will be able to better learn a unified spatial interface and thus transfer better if together with the BEV task we are able to introduce at least some degree of variability into the camera images that serve as model inputs. To verify this, we study how the BEV task interacts with common ways to introduce robustness to image variations in the model, such as performing image augmentations or injecting calibration information into the model. For the latter, we do it by embedding camera parameters with a simple MLP and adding the resulting vector to its corresponding image patches, multiplied by parameters α that are initialized as zero to minimize the interference with the vision-language alignment of the backbone, as is done in other works [21].

![](images/9a8030259bd3faf336486b1b5933e666b83415194ccc1376a42df8a59dff358c.jpg)  
Figure 5. Interaction of BEV task with common techniques for dataset robustness, all trainings on WOD-E2E + NAVSIM + nuScenes-QA. As before, runs using the BEV task are able to improve in-dataset WOD-E2E validation results. For zero-shot transfer on Physical AI-1k, common robustness techniques do not lead to improvements unless paired with the BEV task, indicating its usefulness in facilitating learning a unified interface from the increased input signals.

As shown in Fig. 5, combining the BEV task with such robustness techniques retains improvements in-dataset as measured on the validation set of WOD-E2E. More interestingly, the two techniques mentioned—image augmentations and calibration as input—only lead to better zero-shot transfer when paired with BEV-Forcing. Although improvements compared to the base model without either BEV or other techniques are still modest, this indicates that applying small perturbations on the training procedure that aim at introducing robustness can be more effective when there is an invariant unified spatial interface in which the model can interpret the extra signal in the inputs.

<table><tr><td rowspan="2">BEV Head</td><td colspan="2">WOD-E2E</td><td colspan="2">PhAI-1k</td></tr><tr><td>ADE</td><td>FDE</td><td>ADE</td><td>FDE</td></tr><tr><td>Cross-Attention Only Full Transformer Block</td><td>2.376 2.441</td><td>5.806 5.959</td><td>1.725 1.805</td><td>5.073 5.308</td></tr></table>

Table 3. Planning performance using the BEV task with BEV heads with different capacities. Notice that the more capable BEV head degrades both in and out of training dataset performance, consistent with our hypothesis that a higher-capacity head places less pressure on the image embeddings themselves to encode spatial information. Metrics not directly comparable to those in Fig. 3, details about this experiment in the Supplementary Material.

How does the capacity of the BEV head affect performance? Our goal is to instill into the backbone an understanding of the spatial relationship between objects in the scene in a way that affects the planned trajectories. For this, we hypothesize that increasing the capacity of the BEV head, such as by using a full transformer block, leads the target-layer image embeddings to contain sparser spatial information compared to the low capacity version presented in Fig. 2 which consists mostly of just a cross-attention layer with few hidden dimensions compared to the VLM backbone. To test this, we compare two smaller runs: both trained on Waymo and NAVSIM, with our BEV head and a full multi-head attention transformer block. Results are shown in Table 3. Notice how the more capable BEV head (precise architecture in Supplementary Material) leads to degraded planning performance in both WOD-E2E which was seen during training and in Physical AI which was not. This gives evidence that forcing the image features themselves to contain denser spatial-related information, sufficient for good BEV construction after simple transformations, is beneficial for planning.

How does BEV-Forcing compare with Spatial Forcing? Finally, to motivate our choice of supervising image features by reconstructing BEV occupancy maps instead of following the procedure in [19], we compare the two methods as before, on both WOD-E2E (in-dataset) and Physical AI (zero-shot) validation sets. After a variety of configurations and hyperparameter tunings, we report in Table 4 ADE and FDE for the best configurations of each (more details in the Supplementary Material). We find that BEV-Forcing is capable of achieving better metrics in both camera rigs.

## 5. Conclusion

In this work we presented the effect of multi-dataset training on driving VLAs as well as BEV-Forcing, a simple technique that generally achieves improved in-dataset planning performance and allows better zero-shot transfer when trained on limited dataset variability, showing it serves as a geometry-aware regularizer. BEV-Forcing supervises the image hidden states at one target layer of the VLM with BEV occupancy maps output by a specialized BEV model, transferring a unified understanding of multi-view cameras to the backbone. We also show how BEV-Forcing improves the effectiveness of common techniques used for robustness training. Crucially, with this work we intend to encourage evaluating models and techniques aimed at improving planning with varying amounts of camera rigs or embodiments, including zero-shot transfer evaluations, as an extra way of discerning the applicability or usefulness of the increasing number of techniques proposed each year for improving end-to-end driving.

<table><tr><td rowspan="2">Method</td><td colspan="2">WOD-E2E</td><td colspan="2">PhAI-1k</td></tr><tr><td>ADE</td><td>FDE</td><td>ADE</td><td>FDE</td></tr><tr><td>BEV-Forcing</td><td>2.319</td><td>5.687</td><td>1.783</td><td>5.192</td></tr><tr><td>Spatial Forcing [19]</td><td>2.341</td><td>5.766</td><td>1.818</td><td>5.264</td></tr></table>

Table 4. Planning performance of BEV-Forcing and Spatial Forcing, training on WOD-E2E + NAVSIM + nuScenes-QA.

Limitations and future work. In this paper we focus on representation learning with image embeddings. Much more can be done, especially with respect to the reasoning trace, action representation, and architectural choices to further improve spatial awareness and geometric transfer. Another direction of future research is exploring how the model behaves when trained with better quality or more informative BEV teacher occupancy maps, such as maps that include drivable area.

## References

[1] Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Robert Equi, Chelsea Finn, Niccolo Fusai, Manuel Y Galliker, et al. π<sub>0.5</sub> : a vision-language-action model with open-world generaliza tion. In 9th Annual Conference on Robot Learning, 2025. 1, 3

[2] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11621–11631, 2020. 5

[3] Wei Cao, Marcel Hallgarten, Tianyu Li, Daniel Dauner, Xunjiang Gu, Caojun Wang, Yakov Miron, Marco Aiello, Hongyang Li, Igor Gilitschenski, et al. Pseudo-simulation for autonomous driving. arXiv preprint arXiv:2506.04218, 2025. 5

[4] Haohan Chi, Huan-ang Gao, Ziming Liu, Jianing Liu, Chenyu Liu, Jinwei Li, Kaisen Yang, Yangcheng Yu, Zeda Wang, Wenyi Li, et al. Impromptu vla: Open weights and open data for driving vision-language-action models. Advances in Neural Information Processing Systems, 38, 2026. 2, 3

[5] Daniel Dauner, Marcel Hallgarten, Tianyu Li, Xinshuo Weng, Zhiyu Huang, Zetong Yang, Hongyang Li, Igor Gilitschenski, Boris Ivanovic, Marco Pavone, et al. Navsim: Data-driven non-reactive autonomous vehicle simulation and benchmarking. Advances in Neural Information Processing Systems, 37:28706–28719, 2024. 5

[6] Daniel Dauner, Valentin Charraut, Bastian Berle, Tianyu Li, Long Nguyen, Jiabao Wang, Changhui Jing, Maximilian Igl, Holger Caesar, Boris Ivanovic, et al. 123d: Unifying multimodal autonomous driving data at scale. arXiv preprint arXiv:2605.08084, 2026. 2, 3

[7] Ria Doshi, Homer Rich Walke, Oier Mees, Sudeep Dasari, and Sergey Levine. Scaling cross-embodied learning: One policy for manipulation, navigation, locomotion and aviation. In Proceedings ofthe 8th Conference on Robot Learning, pages 496–512. PMLR, 2025. 2

[8] Haoyu Fu, Diankun Zhang, Zongchuang Zhao, Jianfeng Cui, Dingkang Liang, Chong Zhang, Dingyuan Zhang, Hongwei Xie, Bing Wang, and Xiang Bai. Orion: A holistic end-to-end autonomous driving framework by visionlanguage instructed action generation. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 24823–24834. IEEE, 2025. 3

[9] Heesang Han, A. Lynn Abbott, and Abhijit Sarkar. D3vl: Understanding driving scenes from 3d time series data and video with language models. In 2026 IEEE Intelligent Vehicles Symposium (IV). IEEE, 2026. 3

[10] Asher James Hancock, Xindi Wu, Lihan Zha, Olga Russakovsky, and Anirudha Majumdar. Actions as language: Fine-tuning vlms into vlas without catastrophic forgetting. In International Conference on Learning Representations, 2026. 1, 3, 4

[11] Adam W Harley, Zhaoyuan Fang, Jie Li, Rares Ambrus, and Katerina Fragkiadaki. Simple-bev: What really matters for multi-sensor bev perception? In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 2759–2765. IEEE, 2023. 2, 4

[12] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. 4

[13] Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, et al. Planning-oriented autonomous driving. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 17853–17862. IEEE, 2023. 6

[14] Chengyue Huang, Mellon M Zhang, Robert Azarcon, Glen Chou, and Zsolt Kira. Maps: Preserving vision-language representations via module-wise proximity scheduling for

better vision-language-action generalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pat tern Recognition, pages 32451–32462, 2026. 1, 3

[15] Jyh-Jing Hwang, Runsheng Xu, Hubert Lin, Wei-Chih Hung, Jingwei Ji, Kristy Choi, Di Huang, Tong He, Paul Covington, Benjamin Sapp, Yin Zhou, James Guo, Dragomir Anguelov, and Mingxing Tan. Emma: End-to-end multimodal model for autonomous driving. Transactions on Machine Learning Research, 2025. 1

[16] Anqing Jiang, Yu Gao, Yiru Wang, Zhigang Sun, Shuo Wang, Yuwen Heng, Hao Sun, Shichen Tang, Lijuan Zhu, Jinhao Chai, et al. Irl-vla: Training an vision-languageaction policy via reward world model. arXiv preprint arXiv:2508.06571, 2025. 7

[17] Sicong Jiang, Zilin Huang, Kangan Qian, Ziang Luo, Tianze Zhu, Yang Zhong, Yihong Tang, Menglin Kong, Yunlong Wang, Siwen Jiao, et al. A survey on vision-languageaction models for autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4524–4536, 2025. 1

[18] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan P. Foster, Pannag R. Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. Openvla: An opensource vision-language-action model. In Proceedings of the 8th Conference on Robot Learning, pages 2679–2713. PMLR, 2025. 1, 2, 3

[19] Fuhao Li, Wenxuan Song, Han Zhao, Jingbo Wang, Pengxiang Ding, Donglin Wang, Long Zeng, and Haoang Li. Spatial forcing: Implicit spatial representation align ment for vision-language-action model. arXiv preprint arXiv:2510.12276, 2025. 2, 3, 4, 8

[20] Jiahui Li, Jiawei Sun, Zixiang Ren, Ming Liu, Jiamin Shi, Ruiteng Zhao, Zhiyang Liu, Liying Liu, Zuoguan Wang, and Kaidi Yang. Ntr: Neural token reconstruction for scene token bottleneck in end-to-end driving. arXiv preprint arXiv:2605.31116, 2026. 7

[21] Peizheng Li, Zhenghao Zhang, David Holtz, Hang Yu, Yutong Yang, Yuzhi Lai, Rui Song, Andreas Geiger, and Andreas Zell. Spacedrive: Infusing spatial awareness into vlmbased autonomous driving. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 40096–40107, 2026. 7

[22] Weiqi Li, Quande Zhang, Ruifeng Zhai, Liang Lin, and Guangrun Wang. Vla models are more generalizable than you think: Revisiting physical and spatial modeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 35025–35035, 2026. 3

[23] Yingyan Li, Shuyao Shang, Weisong Liu, Bing Zhan, Haochen Wang, Yuqi Wang, Yuntao Chen, Xiaoman Wang, Yasong An, Chufeng Tang, et al. Drivevla-w0: World mod els amplify data scaling law in autonomous driving. In In ternational Conference on Learning Representations, pages 7890–7911, 2026. 3

[24] Mengmeng Liu, Diankun Zhang, Jiuming Liu, Jianfeng Cui, Hongwei Xie, Guang Chen, Hangjun Ye, Michael Ying

Yang, Francesco Nex, and Hao Cheng. Driveva: Video action models are zero-shot drivers. arXiv preprint arXiv:2604.04198, 2026. 3

[25] Yuechen Luo, Fang Li, Shaoqing Xu, Yang Ji, Zehan Zhang, Bing Wang, Yuannan Shen, Jianwei Cui, Long Chen, Guang Chen, et al. Last-vla: Thinking in latent spatio-temporal space for vision-language-action in autonomous driving. arXiv preprint arXiv:2603.01928, 2026. 3

[26] NVIDIA. Physical ai autonomous vehicles dataset. Hugging Face dataset, 2025. 5

[27] Abby O’Neill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 6892–6903. IEEE, 2024. 3

[28] Chenbin Pan, Burhaneddin Yaman, Tommaso Nesti, Abhirup Mallik, Alessandro G Allievi, Senem Velipasalar, and Liu Ren. Vlp: Vision language planning for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14760–14769, 2024. 1

[29] Tianwen Qian, Jingjing Chen, Linhai Zhuo, Yang Jiao, and Yu-Gang Jiang. Nuscenes-qa: A multi-modal visual question answering benchmark for autonomous driving scenario. In Proceedings of the AAAI conference on artificial intelligence, pages 4542–4550, 2024. 5, 7

[30] Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, JiaYuan Gu, Bin Zhao, Dong Wang, et al. Spatialvla: Exploring spatial representations for visual-language-action model. arXiv preprint arXiv:2501.15830, 2025. 3

[31] Qwen Team. Qwen3.5: Towards native multimodal agents, 2026. 3

[32] Luke Rowe, Rodrigue de Schaetzen, Roger Girgis, Christopher Pal, and Liam Paull. Poutine: Vision-languagetrajectory pre-training and reinforcement learning posttraining enable robust end-to-end autonomous driving. arXiv preprint arXiv:2506.11234, 2025. 3, 4, 7

[33] Shuyao Shang, Bing Zhan, Yunfei Yan, Yuqi Wang, Yingyan Li, Yasong An, Xiaoman Wang, Jierui Liu, Lu Hou, Lue Fan, et al. Dynvla: Learning world dynamics for action reasoning in autonomous driving. arXiv preprint arXiv:2603.11041, 2026. 3

[34] Yinzhe Shen, Omer S¸ ahin Tas¸, Kaiwen Wang, Royden Wag-<sup>¨</sup> ner, and Christoph Stiller. Divide and merge: Motion and semantic learning in end-to-end autonomous driving. Transactions on Machine Learning Research, 2025. 6

[35] Royden Wagner, Omer Sahin Tas, Jaime Villa, Felix Hauser, Yinzhe Shen, Marlon Steiner, Dominik Strutz, Carlos Fernandez, Christian Kinzig, Guillermo S Guitierrez-Cabello, et al. Longtail driving scenarios with reasoning traces: The kitscenes longtail dataset. arXiv preprint arXiv:2603.23607, 2026. 3, 5, 6

[36] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the

Computer Vision and Pattern Recognition Conference, pages 5294–5306, 2025. 3

[37] Jie Wang, Guang Li, Zhijian Huang, Chenxu Dang, Hangjun Ye, Yahong Han, and Long Chen. Vggdrive: Empowering vision-language models with cross-view geomet ric grounding for autonomous driving. arXiv preprint arXiv:2602.20794, 2026. 3

[38] Yan Wang, Wenjie Luo, Junjie Bai, Yulong Cao, Tong Che, Ke Chen, Yuxiao Chen, Jenna Diamond, Yifan Ding, Wen hao Ding, et al. Alpamayo-r1: Bridging reasoning and action prediction for generalizable autonomous driving in the long tail. arXiv preprint arXiv:2511.00088, 2025. 3, 6

[39] Katharina Winter, Mark Azer, and Fabian B Flohr. Bevdriver: Leveraging bev maps in llms for robust closed-loop driving. In 2025 IEEE/RSJ International Conference on In telligent Robots and Systems (IROS), pages 20379–20385. IEEE, 2025. 3

[40] Runsheng Xu, Hubert Lin, Wonseok Jeon, Hao Feng, Yuliang Zou, Liting Sun, John Gorman, Kate Tolstaya, Sarah Tang, Brandyn White, et al. Wod-e2e: Waymo open dataset for end-to-end driving in challenging long-tail scenarios. In Proceedings of the IEEE/CVF Conference on Computer Vi sion and Pattern Recognition, pages 3709–3718, 2026. 3, 4, 5, 7

[41] Jin Yao, Dhruva Dixith Kurra, Tom Lampo, Zezhou Cheng, Danhua Guo, and Burhan Yaman. Vlga: Vision-languagegeometry-action models for autonomous driving. arXiv preprint arXiv:2606.12396, 2026. 3

[42] Haoqi Yuan, Zhixuan Liang, Anzhe Chen, Ye Wang, Haoyang Li, Pei Lin, Yiyang Huang, Zixing Lei, Tong Zhang, Jiazhao Zhang, et al. Qwen-robotmanip technical report: Alignment unlocks scale for robotic manipulation foundation models. arXiv preprint arXiv:2606.17846, 2026. 3

[43] Jinliang Zheng, Jianxiong Li, Zhihao Wang, Dongxiu Liu, Xirui Kang, Yuchun Feng, Yinan Zheng, Jiayin Zou, Yilun Chen, Jia Zeng, et al. X-vla: Soft-prompted transformer as scalable cross-embodiment vision-language-action model. In International Conference on Learning Representations, pages 60580–60606, 2026. 3

[44] Weicheng Zheng, Yixin Huang, Qiao Sun, Derun Li, and Hang Zhao. Drivema: Driving vision-languageaction models with verifiable meta-actions. arXiv preprint arXiv:2605.31271, 2026. 7

[45] Xingcheng Zhou, Xuyuan Han, Feng Yang, Yunpu Ma, Volker Tresp, and Alois Knoll. Opendrivevla: Towards endto-end autonomous driving with large vision language action model. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 13782–13790, 2026. 3

[46] Zewei Zhou, Tianhui Cai, Seth Zhao, Yun Zhang, Zhiyu Huang, Bolei Zhou, and Jiaqi Ma. Autovla: A visionlanguage-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. NeurIPS, 38:27920–27956, 2026. 3

[47] Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, et al. Rt-2: Vision-language-action models

transfer web knowledge to robotic control. In Conference on Robot Learning, pages 2165–2183. PMLR, 2023. 1, 2