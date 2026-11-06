# UniGADD: Universal Generator-Agnostic Deepfake Detector

Bernard Tomczyk   
School of Computing and Communications, Lancaster University   
Lancaster, United Kingdom   
Agil Aghasanli   
School of Computing and Communications, Lancaster University   
Lancaster, United Kingdom   
Plamen Angelov   
School of Computing and Communications, Lancaster University   
Lancaster, United Kingdom

b.tomczyk@lancaster.ac.uk

a.aghasanli1@lancaster.ac.uk

p.angelov@lancaster.ac.uk

## Abstract

In this paper, we introduce Universal Generator-Agnostic Deepfake Detector (UniGADD), a novel method that employs supervised learning to achieve high accuracy on known deepfake generators, while maintaining robust performance on previously unseen ones. The proposed approach follows a two-stage optimisation process. In the first stage, a contrastive loss encourages the model to learn discriminative feature embeddings from real and known fake images, resulting in strong performance within the training domain. In the second stage, the embedding space is refined by promoting inter-cluster separation and intra-cluster compact ness, applied exclusively to real samples. This refinement enhances generalisability, enabling the method to exhibit improved robustness against unseen deepfake generation techniques. UniGADD achieves accuracy on par with state-of-the-art methods for known generators, while significantly outperforming them on unseen cases, demonstrating its scalability and practical applicability for adversarial content detection.

Code: https://github.com/Aghasanli/UniGADD-Universal-Generator-Agnostic-Deepfake-Detector

## 1 Introduction

The rapid proliferation of deepfake generators poses an escalating challenge for forensic tools (Rana et al., 2022). Most published detectors are trained with labelled examples from a fixed set of generators, achieving high accuracy on those seen attacks yet faltering when confronted with novel manipulation pipelines. Pellicer et al. (Pellicer et al., 2024) frame the task as unsupervised anomaly detection, showing that generator-agnostic cues exist, although their reliance on nominal-only statistics leaves recent advances in representation learning under-explored

To address the cold-start scenario where no fake examples are available, we refine foundation-model embeddings with a dedicated contrastive loss and derive prototypes that map the real-image distribution. Samples that deviate from these prototypes are flagged as potential anomalies, enabling detection without generatorspecific supervision. This approach sidesteps the need for curated fake datasets and remains robust as synthesis methods evolve.

While such unsupervised strategies excel in generalisability, purely nominal training can sacrifice closedset precision. Conversely, supervised detectors capture fine-grained artefacts but overfit to the generators present in the training set (Wang et al., 2024). We propose the Universal Generator-Agnostic Deepfake Detector (UniGADD), which unifies these complementary strengths. UniGADD first exploits labelled fakes, when available, to carve a discriminative margin between real and manipulated imagery, then tightens the real-image region through the prototype-based refinement introduced above. The result is a detector that preserves high precision on known attacks while degrading gracefully on unseen ones.

Two practical considerations guide our design. Authentic imagery is plentiful, yet collecting labelled samples for every new generator is unrealistic (Naitali et al., 2023). At the same time, detectors must remain reliable as synthesis technology evolves. By combining light supervision with feature-space regularisation, UniGADD meets both requirements without continual retraining.

We benchmark UniGADD on the AIGC datasets (Fan et al., 2024), which contain a diverse mixture of GANand difusion-based generators and reflect varied real-world usage scenarios (Zhu et al., 2023). UniGADD matches or surpasses specialist baselines on their home domains and maintains competitive performance when confronted with previously unseen generators, supporting its suitability for real-world deployment.

Our main contributions are as follows:

• We introduce UniGADD, a contrastive clustering framework that generalizes synthetic image detection to unseen generative models.

• We propose a real-image clustering strategy that enables the model to leverage intrinsic data structure without supervision.

• We establish a leave-one-out protocol covering both GAN- and difusion-based generators, enabling rigorous evaluation of cross-domain transfer.

• We demonstrate competitive performance on the challenging AIGC benchmark, highlighting strong one-to-many generalization.

Terminology. Throughout this paper we use the term “deepfake” in a broad sense to mean AI-generated visual content (AIGC). We adopt this wording because “deepfake” has become the prevalent public term for generated images, but whenever precise distinctions are required we explicitly refer to GAN- or difusionbased synthesis.

## 2 Related Work

Early forensic eforts relied on handcrafted cues such as lighting, facial landmarks, or head pose (Farid, 2016; Sencar & Memon, 2013). As GANs and difusion models advanced, supervised classifiers trained on curated real/fake datasets became standard (Rössler et al., 2019; Yang et al., 2019; Wang et al., 2020a; Bird & Lotfi, 2023), with DE-FAKE (Sha et al., 2023) and the contrastive method of Baraldi et al. (Baraldi et al., 2024) demonstrating strong closed-set accuracy. These systems, however, tend to generalise poorly to unseen generators, as documented by Ojha et al. (Ojha et al., 2023). Frequency-domain studies report similar limitations, with artefacts learnt from GAN families rarely transferring to difusion models (Frank et al., 2020; Wang et al., 2020b).

Generator-agnostic detection aims to mitigate this brittleness. Epstein et al. (Epstein et al., 2023) show substantial accuracy drops when a novel generator appears in a continual-learning setting, and Corvi et al. (Corvi et al., 2023a;b) confirm that detectors trained on GAN artefacts do not automatically extend to difusion artefacts. Prototype-based detectors, exemplified by PUDD (Pellicer et al., 2024), measure similarity to learned class prototypes and flag low-score inputs as potential fakes, improving robustness to unseen attacks. Recent work couples contrastive objectives with large vision encoders to capture both discrimination and generalisation: Larue et al. (Larue et al., 2023) compact the real cluster with a bounded loss, Sun et al. (Sun et al., 2023) use pseudo-labels for open-world attribution, and Shi et al. (Shi et al., 2023) report that linear probes on CLIP features can outperform many bespoke detectors, although fine-tuning still ofers gains.

Visual anomaly-detection methods model the nominal data distribution and treat deviations as outliers. PatchCore (Roth et al., 2022) stores patch features in a memory bank, and PaDiM (Defard et al., 2020) employs pixel-wise Gaussian models. While efective in industrial settings, these designs entail memory overhead or pixel-level supervision that does not transfer cleanly to deepfake tasks. Foundation-model embeddings from CLIP (Radford et al., 2021b) and DINOv2 (Oquab et al., 2024) provide a strong baseline, yet adapting them without sacrificing general utility remains challenging; Han et al. (Han et al., 2024) show that fine-tuned variants can overfit the target domain.

Prototypical networks compute class centroids in the embedding space and classify samples by distance (Snell et al., 2017). Their success in few-shot learning and interpretability has motivated extensions to meta-learning and self-supervised representation learning (Chen et al., 2018; Angelov et al., 2025; Donnelly et al., 2021; Wu et al., 2021). SwAV (Caron et al., 2020) and DINO (Caron et al., 2021) organise unlabelled data into semantically meaningful clusters, although prototype collapse can limit diversity, prompting regularisation strategies such as ReSA (Weng et al., 2025). The contrastive loss used in UniGADD builds on this body of work by enforcing inter-prototype separation and intra-prototype compactness within the real-image manifold, enabling eficient anomaly cues while retaining the general features of the underlying foundation model.

UniGADD integrates supervised contrastive learning, prototype-based refinement, and lightweight anomaly detection, preserving high precision on generators seen during training and improves recall on unseen ones without requiring extensive memory resources or continual retraining.

## 3 Proposed Method

![](images/fe21794fc6c59407e4eaeb5ba2c1db92cf3fa90ae19c676f65f131a7488bff19.jpg)  
Figure 1: Overview of the proposed method, consisting of feature extraction, supervised contrastive prototype refinement and anomaly detection.

We propose a supervised contrastive learning framework for deepfake detection and generator attribution that combines structured pseudo-labelling of real data with explicit supervision from known fake generators. UniGADD follows a two-stage process: the first learns a discriminative embedding space through hybrid labelling, while the second refines cluster structure within the real image distribution to improve robustness and generalisability.

## 3.1 Feature Extraction and Prototype Initialisation

Given a set of real and fake training samples $X = \{ \mathbf { x } _ { i } \} _ { i = 1 } ^ { N } ,$ , we extract feature embeddings using stateof-the-art Vision Transformer architectures, including DINOv2 (Oquab et al., 2024) and CLIP (Radford et al., 2021a). DINOv2 produces 1024-dimensional embeddings, while CLIP generates 768-dimensional representations. These embeddings define a unified representation space for both real and fake data, providing a solid foundation for contrastive learning and structural analysis.

We begin by applying K-means clustering over the real sample embeddings to induce an initial structure in the feature space. These cluster assignments serve as pseudo-labels for real samples, while fake samples retain their known generator labels. The combined labels are used to supervise contrastive learning, which sharpens the representation structure by encouraging alignment within clusters and separation between them. Crucially, this sharpening lays the foundation for the second stage, where we explicitly refine the real clusters through intra-cluster compactness objectives. After contrastive training, we perform prototype selection by identifying the samples closest to each cluster centre in the refined embedding space. This deferred selection ensures that the prototypes reflect the improved geometry of the learned space, leading to enhanced interpretability and robustness in downstream detection and attribution.

## 3.2 Stage 1: Supervised Contrastive Learning

To learn a discriminative embedding space, we optimise a supervised contrastive loss (Khosla et al., 2020) over a composite label set. Let $\mathbf { z } _ { i }$ denote the normalised embedding of sample $i ,$ and let $y _ { i }$ be its composite label. The loss is defined as:

$$
L _ { \mathrm { c o n t r a s t } } = \sum _ { i = 1 } ^ { N } { \frac { - 1 } { | \mathcal { P } ( i ) | } } \sum _ { p \in \mathcal { P } ( i ) } \log { \frac { \exp ( \mathbf { z } _ { i } \cdot \mathbf { z } _ { p } / \tau ) } { \sum _ { a \in { \mathcal { A } } ( i ) } \exp ( \mathbf { z } _ { i } \cdot \mathbf { z } _ { a } / \tau ) } } ,\tag{1}
$$

where ${ \mathcal { P } } ( i ) = \{ j \neq i | y _ { j } = y _ { i } \}$ denotes the set of positives for anchor $i , \mathcal { A } ( i ) = \{ j \neq i \}$ is the full batch excluding the anchor, and τ is a temperature parameter.

This objective encourages embeddings of samples with the same composite label, either real samples sharing a pseudo-cluster label or fake samples from the same generator, to be drawn together, while pushing apart dissimilar instances. Crucially, the pseudo-labels for real samples, derived from K-means clustering, are periodically updated during training to reflect the evolving geometry of the feature space. This dynamic relabelling reinforces the underlying structure and helps sharpen cluster boundaries, setting the stage for more efective intra-cluster refinement.

## 3.3 Stage 2: Intra-Cluster Compactness

Following supervised contrastive learning, we further refine the structure of the real sample distribution by explicitly enforcing intra-cluster compactness. The pseudo-clusters induced in Stage 1 provide an initial structural prior over the real data. However, these clusters may contain internal variation, and loosely aligned samples near the periphery can obscure clear decision boundaries, especially when facing previously unseen deepfakes.

To address this, we define a two-part objective that consolidates cluster structure by tightening the core and softly regulating the boundary. For each real cluster $k ,$ we compute a prototype vector $\pmb { \mu } _ { k }$ as the mean embedding of the assigned samples. Cosine similarity is then used to assess alignment between a sample and its cluster centre:

$$
s _ { k } ( \mathbf { z } _ { i } ) = { \frac { \mathbf { z } _ { i } \cdot { \boldsymbol { \mu } } _ { k } } { \left\| \mathbf { z } _ { i } \right\| \left\| \mu _ { k } \right\| } } ,\tag{2}
$$

$$
X _ { k } ^ { ( s p ) } = \{ \mathbf { z } _ { i } : y _ { i } = k , \ s _ { k } ( \mathbf { z } _ { i } ) \geq \mu _ { k } - 2 \sigma _ { k } \} ,\tag{3}
$$

$$
X _ { k } ^ { ( w p ) } = \{ { \bf z } _ { i } : y _ { i } = k , ~ s _ { k } ( { \bf z } _ { i } ) < \mu _ { k } - 2 \sigma _ { k } \} ,\tag{4}
$$

where $\mu _ { k }$ and $\sigma _ { k }$ denote the mean and standard deviation of similarity scores within cluster k.

The super-positive region $X _ { k } ^ { ( s p ) }$ represents highly aligned, clean real samples. By pulling them closer to the prototype, we reinforce cluster cohesion:

$$
L _ { \mathrm { s u p e r - p o s } } = \sum _ { k = 1 } ^ { K _ { r } } \sum _ { \mathbf { z } _ { i } \in X _ { k } ^ { \left( s p \right) } } \left( 1 - s _ { k } ( \mathbf { z } _ { i } ) \right) ^ { 2 } .\tag{5}
$$

Conversely, the weak-positive region $X _ { k } ^ { ( w p ) }$ captures samples that deviate from the cluster core. These peripheral areas are likely to overlap with novel or unseen fake samples in the test phase. To improve anomaly sensitivity, we gently push such samples outward using:

$$
L _ { \mathrm { w e a k - p o s } } = \sum _ { k = 1 } ^ { K _ { r } } \sum _ { \mathbf { z } _ { i } \in X _ { k } ^ { ( w p ) } } s _ { k } ( \mathbf { z } _ { i } ) .\tag{6}
$$

The total loss is given by:

$$
L _ { \mathrm { r e f i n e } } = \lambda _ { 1 } L _ { \mathrm { s u p e r - p o s } } + \lambda _ { 2 } L _ { \mathrm { w e a k - p o s } } ,\tag{7}
$$

where $\lambda _ { 1 }$ and $\lambda _ { 2 }$ control the relative contribution of each term.

This refinement stage not only improves the clarity and robustness of real clusters but also increases the likelihood that unseen fake samples will fall outside the compacted real distribution, enabling more accurate detection and attribution.

## 3.4 Deepfake Detection and Attribution

Following the two-stage training process, we compute final cluster centroids for both real and fake classes. These centroids summarise the distributional structure of the training data in the embedding space, where real clusters have been compacted through refinement, and fake clusters are formed based on known generator groupings established during contrastive learning.

For a test embedding $\mathbf { z } _ { \mathrm { t e s t } }$ , we calculate its Euclidean distance to each cluster centroid. The sample is classified as real if it is closer to any real cluster than to the closest fake cluster:

$$
\mathrm { l a b e l } = \left\{ \mathrm { R e a l } , \quad \ \mathrm { i f } \ \operatorname* { m i n } _ { k } \ \lVert \mathbf { z } _ { \mathrm { t e s t } } - \boldsymbol { \mu } _ { k } ^ { \mathrm { r e a l } } \rVert < \operatorname* { m i n } _ { j } \ \lVert \mathbf { z } _ { \mathrm { t e s t } } - \boldsymbol { \mu } _ { j } ^ { \mathrm { f a k e } } \rVert , \right.\tag{8}
$$

This nearest-centroid classification mechanism enables simultaneous deepfake detection and generator attribution. The refined structure of real clusters ensures a compact decision region, increasing sensitivity to out-of-distribution manipulations. Meanwhile, fake centroids, which are anchored by generator-specific labels, allow for accurate attribution among known manipulations.

Importantly, since weakly aligned real samples have been softly pushed away from the cluster cores, previously unseen fake content is more likely to fall outside the refined real distribution. This results in improved robustness against generalisation failures and supports open-set detection capabilities in realistic forensic settings.

Algorithm 1: Supervised Contrastive Detection Framework   
Input: Embeddings, ground-truth generator labels, number of real clusters $K _ { r } ,$ learning rate η   
// Stage 1: Contrastive embedding learning   
for epoch = 1 to #Stage 1 epochs do   
Cluster real samples using k-means   
Assign composite labels (real pseudo-labels + fake generator labels)   
Compute contrastive loss L<sub>contrast</sub>   
Update projection model using gradient descent   
end   
// Stage 2: Cluster refinement on real data   
for epoch = 1 to #Stage 2 epochs do   
Compute real cluster prototypes µ<sub>k</sub>   
Identify $X _ { k } ^ { ( s p ) } , X _ { k } ^ { ( w p ) }$ via cosine similarity   
Compute L<sub>super-pos</sub>, L<sub>weak-pos</sub>   
Update model using gradient descent   
end   
// Inference and attribution   
Store final centroids for real and fake clusters   
Classify test samples by nearest-centroid comparison

Visualising Embedding Structure. To qualitatively assess the structure induced by our training procedure, we project the learned embeddings into 2D using t-SNE. Figure 2 shows the resulting layouts when individual generators are held out during training. Real samples form compact, well-separated clusters, while fake samples from unseen generators appear peripheral or distinct, illustrating UniGADD’s ability to isolate anomalous inputs in feature space.

![](images/dc9796dea3d56d2db79c54e432b806bce2dce45cff4d42171baeda341f1d5c49.jpg)  
(a) Held-out: Stable Difusion

![](images/2235b172b0c1ea65777ea766ef36d9d8492f105c8f956a0fa5519dcf7843753e.jpg)  
(b) Held-out: GauGAN

![](images/221014136a13c2d98b2436e29ebd5e7bec476d29fe2f704ba6432e0f9bb773f9.jpg)  
(c) Held-out: Amused  
Figure 2: t-SNE projections of the embedding space after training. Light blue indicates the real samples. Real samples form compact clusters, while unseen generator samples (held out during training) remain distinct or peripheral.

## 4 Experiments

Implementation details. We evaluate UniGADD on the AIGC datasets (Fan et al., 2024), covering diverse synthesis styles and content types. Images are resized to 224 × 224 and normalised with ImageNet statistics. Features are extracted using a frozen DINOv2 ViT-L/14 backbone (Oquab et al., 2024), and a linear projection head maps embeddings to 512 dimensions. The projection is trained for 10 epochs with Adam $( \mathrm { l r } = 1 0 ^ { - 3 } )$ , contrastive batch size 256, and temperature $\tau = 0 . 0 7$

Real images are partitioned into clusters to provide pseudo-labels during training. We fix $n _ { \mathrm { t o t a l } } = 4 0$ clusters, with $n _ { \mathrm { r e a l } } = n _ { \mathrm { t o t a l } } - n _ { \mathrm { f a k e } }$ where $n _ { \mathrm { f a k e } }$ is the number of training generators. After training, class centroids are computed in the learned space and used for nearest-centroid classification at test time. The total number of clusters controls the granularity of the real-image manifold in the embedding space. A smaller number of clusters yields coarse partitions that may under-represent intra-class variation, while a larger number of clusters produces finer partitions that better capture local structure. In our setting, increasing the number of real clusters allows the model to more tightly characterise the nominal data distribution, which in turn improves separation at the boundary where unseen fake samples are likely to appear. Empirically, we observe that performance remains stable across a wide range of cluster counts (see Table 4), indicating that the method is not overl y sensitive to this hyperparameter. We therefore select a moderate value that balances representational fidelity and computational cost.

Evaluation follows a leave-one-out protocol: each generator is excluded in turn and used as an unseen test set, while the remaining generators and all real data form the training pool. Splits are sampled with a fixed random seed for reproducibility. All experiments are implemented in PyTorch and run on NVIDIA Tesla V100 GPUs.

## 5 Results

## 5.1 Evaluation Protocol

We report results using class-averaged accuracy (%) to ensure equal weighting across real and fake classes. To evaluate generalisation under distribution shift, we separate test data into seen (generators included during training) and unseen (generators excluded) subsets. This separation simulates realistic deployment conditions where previously unknown generation techniques may emerge.

![](images/225ca082e21c97992dcacb1fda5fc7884ffa5f9e59adf3e107b022a18d94b963.jpg)

Per-Generator Accuracy Comparison (Unseen Accuracy)  
![](images/a2df32c16acdad9882248f44b8e39cbc80577bdec721fd2de0147075b4b3f611.jpg)  
Figure 3: Average accuracy per generator under leave-one-out evaluation. Subfigure (a) shows accuracy on seen generators; (b) shows accuracy on unseen ones. UniGADD maintains high performance across both, outperforming previous methods in generalisation.

We compare UniGADD against recent state-of-the-art deepfake detection methods, including supervised and unsupervised approaches (Ojha et al., 2023; Wang et al., 2023; Cozzolino et al., 2024). We also benchmark a DINOv2 + Linear classifier baseline to demonstrate that our method improves upon standard frozen-feature detectors by incorporating contrastive structure and cluster refinement. Our experimental design progressively increases the challenge of generalisation. We begin with a Leave-One-Out (LOO) setup, withholding one generator at a time, then extend to a Leave-N-Out (LNO) evaluation where multiple generators are excluded simultaneously. Finally, we examine the efect of changing the backbone encoder to test whether UniGADD’s robustness is architecture-dependent.

<table><tr><td rowspan="2"></td><td rowspan="2">Method → ↓ Generator</td><td colspan="2">Cozzolino et al. ↑</td><td colspan="2">Ohja et al. ↑</td><td colspan="2">Wang et al. ↑</td><td colspan="2">DINOv2 + Linear ↑</td><td colspan="2">Ours ↑</td></tr><tr><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td></tr><tr><td rowspan="9">GANS</td><td>BigGAN</td><td>79.92</td><td>60.47</td><td>91.49</td><td>80.79</td><td>91.03</td><td>66.12</td><td>91.16</td><td>52.50</td><td>94.80</td><td>88.50</td></tr><tr><td>CycleGAN</td><td>79.84</td><td>60.18</td><td>91.55</td><td>73.79</td><td>89.48</td><td>71.86</td><td>89.97</td><td>50.17</td><td>91.58</td><td>89.10</td></tr><tr><td>Faces</td><td>79.77</td><td>59.82</td><td>91.59</td><td>55.05</td><td>86.27</td><td>62.10</td><td>90.20</td><td>48.04</td><td>94.24</td><td>88.00</td></tr><tr><td>GauGAN</td><td>79.96</td><td>60.60</td><td>91.69</td><td>79.74</td><td>90.76</td><td></td><td>97.16</td><td>51.92</td><td>95.03</td><td></td></tr><tr><td></td><td>79.22</td><td>60.60</td><td></td><td></td><td></td><td>74.95</td><td></td><td>45.06</td><td>91.47</td><td>87.06</td></tr><tr><td>ProGAN</td><td>79.64</td><td></td><td>91.39</td><td>67.09</td><td>87.42</td><td>56.47</td><td>90.04</td><td></td><td></td><td>82.50</td></tr><tr><td>StarGAN</td><td></td><td>59.42</td><td>91.51</td><td>78.47</td><td>89.19</td><td>74.88</td><td>90.50</td><td>61.12</td><td>93.88</td><td>86.00</td></tr><tr><td>StyleGAN</td><td>79.93</td><td>60.53</td><td>92.09</td><td>55.05</td><td>89.94</td><td>58.41</td><td>90.89</td><td>64.73</td><td>94.64</td><td>86.20</td></tr><tr><td>StyleGAN2</td><td>79.67</td><td>59.53</td><td>93.13</td><td>56.21</td><td>91.06</td><td>65.83</td><td>92.10</td><td>58.29</td><td>94.87</td><td>82.08</td></tr><tr><td rowspan="7">Diion</td><td>ADM</td><td>81.98</td><td>65.47</td><td>91.67</td><td>79.68</td><td>91.06</td><td>79.63</td><td>90.32</td><td>53.64</td><td></td><td>88.32</td></tr><tr><td>DALL-E 2</td><td>80.14</td><td>61.28</td><td>92.23</td><td>75.05</td><td>91.77</td><td></td><td>90.37</td><td>56.06</td><td>95.30 95.30</td><td></td></tr><tr><td>GLIDE</td><td>79.93</td><td>60.48</td><td>91.55</td><td>79.53</td><td>90.58</td><td>70.85</td><td>89.03</td><td>53.12</td><td>93.42</td><td>82.21</td></tr><tr><td></td><td>81.95</td><td>66.98</td><td>91.69</td><td></td><td></td><td>74.90</td><td></td><td>52.50</td><td></td><td>86.23</td></tr><tr><td>Midjourney</td><td>80.78</td><td></td><td></td><td>78.05</td><td>85.75</td><td>73.50</td><td>90.77</td><td></td><td>94.51</td><td>86.97</td></tr><tr><td>SD XL</td><td>79.94</td><td>63.27</td><td>91.88</td><td>70.37</td><td>87.10</td><td>75.28</td><td>88.10</td><td>55.00</td><td>95.13</td><td>85.51</td></tr><tr><td>SD 1.4 SD</td><td></td><td>60.57 61.28</td><td>91.32 91.55</td><td>79.68 80.05</td><td>86.80 86.96</td><td>75.74 76.64</td><td>89.91</td><td>52.06</td><td>95.13</td><td>74.25</td></tr><tr><td rowspan="3"></td><td>1.5</td><td>80.15 79.80</td><td>59.98</td><td>91.49</td><td>80.26</td><td>86.10</td><td>77.90</td><td>89.58 90.17</td><td>50.75 55.05</td><td>94.61 94.70</td><td>87.75</td></tr><tr><td>VDQM</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>87.75</td></tr><tr><td>AVG Range X</td><td>80.16 2.76</td><td>59.82 22.50</td><td>91.68 0.91</td><td>73.64 25.74</td><td>88.62 5.68</td><td>70.79 21.43</td><td>90.32 9.06</td><td>53.84 19.67</td><td>94.29 3.83</td><td>85.53 14.85</td></tr></table>

Table 1: Detection performance by generator across diferent methods for Leave-One-Out, showing Seen and Unseen accuracy. Generators are grouped by model type (GANs vs Difusion). Our (UniGADD) results are highlighted in green.

## 5.2 Leave-One-Out Generalisation

We first assess UniGADD’s performance in the LOO setting. In each leave-one-out run, one generator is held out in turn, ensuring that every dataset serves once as the unseen test domain; within each split, train/test partitions are sampled randomly on multiple runs and averaged. Results are presented in Table 1 and visualised in Figure 3. It should be noted that the held-out sets include both GAN-based and difusionbased models (e.g., excluding GauGAN while training on difusion models, or vice versa), our protocol naturally covers cross-paradigm transfer between GANs and difusion models.

Across all methods, performance on unseen generators is consistently lower than on seen ones, confirming the dificulty of out-of-distribution detection. However, UniGADD achieves the best unseen accuracy overall (85.53%), significantly outperforming prior methods, including supervised (Ohja: 73.64%) and unsupervised baselines (DINOv2+Linear: 53.84%). UniGADD also exhibits the lowest variance across unseen generators (range: 14.85), suggesting consistent robustness even when exposed to unfamiliar generation styles. Notably, difusion models—often harder to detect—see substantial improvements with UniGADD. For example, on Midjourney and ADM, our method achieves over 86% accuracy, whereas DINOv2+Linear drops below 55%. These findings validate that the contrastive-prototype training and real-cluster refinement stages equip UniGADD with stronger resilience to distribution shift than previous approaches.

![](images/d3303b22728760ca0e80f8b5627eaf333a6812dd305def4f8961dfecfb10042a.jpg)  
Figure 4: Average accuracy as the number of generators excluded from training increases. UniGADD generalises more robustly than prior methods.

## 5.3 Leave-N-Out Generalisation

To further evaluate generalisation under increasingly challenging scenarios, we conduct a Leave-N-Out evaluation, varying N from 3 to 13. For each N, 10 random generator subsets are withheld from training and averaged to ensure stability. Results are summarised in Table 2 and visualised in Figure 4.

As expected, all methods experience performance degradation as N increases. However, UniGADD retains high accuracy on both seen (92.47%) and unseen generators (83.63% on average). Even with 13 generators excluded—nearly 80% of all available training data—UniGADD outperforms others by large margins. Compared to Ohja et al., our method shows a smaller drop in unseen accuracy from N = 3 to N = 13 (15.09pp vs 30.86pp), underscoring UniGADD’s graceful degradation under extreme distributional shifts. The range for unseen accuracy is also lowest for UniGADD (18.51), again demonstrating stability across permutations. This experiment supports our hypothesis that refining the geometry of the real distribution during training leads to broader generalisation across diverse, previously unobserved generative mechanisms.

<table><tr><td rowspan="2">Method → ↓N Unseen</td><td colspan="2">Cozzolino et al. ↑</td><td colspan="2">Ohja et al. ↑</td><td colspan="2">Wang et al. ↑</td><td colspan="2">DINOv2 + Linear ↑</td><td colspan="2">Ours ↑</td></tr><tr><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td></tr><tr><td>3</td><td>79.455</td><td>75.35</td><td>90.92</td><td>86.79</td><td>89.3</td><td>76.5</td><td>90.6</td><td>62.5</td><td>94.20</td><td>86.04</td></tr><tr><td>5</td><td>79.10</td><td>78.86</td><td>90.44</td><td>86.88</td><td>88.5</td><td>75.5</td><td>90.0</td><td>63.0</td><td>93.79</td><td>87.42</td></tr><tr><td>7</td><td>76.32</td><td>73.86</td><td>90.10</td><td>83.68</td><td>87.3</td><td>74.2</td><td>89.4</td><td>62.0</td><td>93.00</td><td>89.46</td></tr><tr><td>9</td><td>78.65</td><td>71.79</td><td>89.17</td><td>83.26</td><td>86.6</td><td>73.0</td><td>88.8</td><td>58.5</td><td>91.94</td><td>83.30</td></tr><tr><td>11</td><td>76.62</td><td>68.35</td><td>90.10</td><td>72.29</td><td>86.0</td><td>72.0</td><td>88.2</td><td>57.0</td><td>90.87</td><td>84.63</td></tr><tr><td>13</td><td>76.27</td><td>55.90</td><td>92.39</td><td>55.93</td><td>85.0</td><td>67.7</td><td>87.0</td><td>49.0</td><td>91.01</td><td>70.95</td></tr><tr><td>AVG</td><td>77.74</td><td>73.73</td><td>90.52</td><td>78.14</td><td>87.12</td><td>73.15</td><td>89.00</td><td>58.67</td><td>92.47</td><td>83.63</td></tr><tr><td>Range ↓</td><td>3.19</td><td>29.20</td><td>3.22</td><td>30.95</td><td>4.30</td><td>8.80</td><td>3.60</td><td>14.00</td><td>3.33</td><td>18.51</td></tr></table>

Table 2: Detection performance by generator across diferent methods for Leave-N-Out, showing Seen and Unseen accuracy. Our (UniGADD) results are highlighted in green.

## 5.4 Backbone Robustness Analysis

To ensure our results are not backbone-specific, we perform ablations using four diferent pretrained vision encoders: CLIP ViT-B/32, CLIP ViT-L/14, DINOv2-S/14, and DINOv2-B/14 (our default). Table 3 reports seen and unseen accuracy for each configuration under the LOO setting.

CLIP-based backbones, while strong on seen data (up to 96.72%), exhibit larger gaps on unseen samples, particularly for the smaller ViT-B/32 (only 70.57% unseen accuracy). In contrast, DINOv2-B/14 achieves better balance with 92.48% seen and 81.47% unseen accuracy, ofering a compelling trade-of between closed- and open-set performance.

These results demonstrate that while UniGADD benefits from stronger backbone representations, its core generalisation ability holds across architectures. The refinement stage appears to amplify the strengths of foundation models while mitigating their overfitting tendencies. Further ablation studies analysing the sensitivity of UniGADD to clustering configuration, decision thresholds, backbone choice, and loss formulation are provided in Appendix A.

<table><tr><td rowspan="2"></td><td rowspan="2">Backbone → ↓ Generator</td><td colspan="2">ViT-B-32 ↑</td><td colspan="2">ViT-L-14 ↑</td><td colspan="2">DINOv2-B/14 ↑</td><td colspan="2">DINOv2-S/14 ↑</td></tr><tr><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td></tr><tr><td rowspan="8">GANS</td><td>BigGAN</td><td>89.54</td><td>62.00</td><td>96.55</td><td>88.50</td><td>91.02</td><td></td><td></td><td></td></tr><tr><td>CycleGAN</td><td>87.57</td><td>71.50</td><td>96.68</td><td>88.50</td><td>92.83</td><td>91.25 84.25</td><td>81.91</td><td>65.75</td></tr><tr><td>Faces</td><td>86.74</td><td></td><td>96.68</td><td>86.50</td><td>89.70</td><td>76.25</td><td>79.93</td><td>69.50</td></tr><tr><td>GauGAN</td><td>90.72</td><td>57.25 67.75</td><td>96.38</td><td>83.50</td><td>91.41</td><td>87.25</td><td>79.11 82.47</td><td>63.25</td></tr><tr><td>ProGAN</td><td>89.12</td><td>76.25</td><td>96.91</td><td>81.25</td><td>94.22</td><td></td><td></td><td>68.25</td></tr><tr><td>StarGAN</td><td></td><td>66.25</td><td></td><td></td><td>90.20</td><td>77.92</td><td>81.56</td><td>76.25</td></tr><tr><td>StyleGAN</td><td>87.96 89.80</td><td>62.75</td><td>96.88 97.01</td><td>88.25 81.25</td><td>94.24</td><td>89.50 70.00</td><td>80.10</td><td>78.50</td></tr><tr><td>StyleGAN2</td><td>87.07</td><td>63.75</td><td>96.32</td><td>87.75</td><td>94.54</td><td>81.75</td><td>82.30 78.98</td><td>58.00 61.00</td></tr><tr><td rowspan="10">Diuion</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ADM</td><td>89.80</td><td>70.00</td><td>96.97</td><td>87.75</td><td>93.12</td><td>87.75</td><td>81.25</td><td>66.00</td></tr><tr><td>DALL-E 2</td><td>89.21</td><td>70.50</td><td>96.58</td><td>87.50</td><td>93.06</td><td>77.75</td><td>86.25</td><td>74.25</td></tr><tr><td>GLIDE</td><td>87.60</td><td>76.25</td><td>96.74</td><td>88.25</td><td>89.51</td><td>74.25</td><td>83.29</td><td>59.50</td></tr><tr><td>Midjourney</td><td>92.57</td><td>78.00</td><td>97.24</td><td>87.25</td><td>93.88</td><td>81.50</td><td>85.10</td><td>76.50</td></tr><tr><td>SD XL</td><td>92.73</td><td>80.25</td><td>96.84</td><td>87.00</td><td>91.68</td><td>79.25</td><td>81.55</td><td>72.50</td></tr><tr><td>SD 1.4</td><td>88.82</td><td>80.25</td><td>96.78</td><td>87.00</td><td>94.38</td><td>86.25</td><td>83.85</td><td>79.25</td></tr><tr><td>SD 1.5</td><td>90.26</td><td>81.25</td><td>96.64</td><td>84.75</td><td>92.57</td><td>83.25</td><td>82.17</td><td>74.50</td></tr><tr><td>VDQM</td><td>89.84</td><td>69.25</td><td>96.81</td><td>85.50</td><td>92.96</td><td>86.50</td><td>82.83</td><td>67.00</td></tr><tr><td>AVG Range ↓</td><td>89.15</td><td>70.57 24.00</td><td>96.72</td><td>86.53 7.25</td><td>92.48 5.03</td><td>81.47 21.25</td><td>81.86 7.29</td><td>68.56 21.50</td></tr></table>

Table 3: Detection accuracy by backbone across GAN and Difusion generators under Leave-One-Out setting.

## 5.5 Limitations

UniGADD shows strong generalisation to unseen generators, but it has several limitations. The method relies on frozen vision transformer backbones such as DINOv2 and CLIP, so any biases or weaknesses in these models are inherited. For example, detection may be less reliable under unusual lighting conditions or non-photographic image styles. The current method only uses single images and does not capture temporal signals that can be useful for spotting subtle video forgeries. Extending UniGADD to video is an important direction for future work.

Our experiments are based on balanced datasets with a reasonable number of samples per generator. The performance of UniGADD in low-shot settings, highly imbalanced data, or in-the-wild content remains untested. Although inference is fast and lightweight, the clustering step used for prototype refinement becomes more expensive as the dataset grows. Large-scale deployments may require streaming or incremental clustering methods.

We do not explicitly evaluate robustness to low-level image perturbations such as compression or blur, which remain important directions for extending our approach.

Finally, we use a fixed threshold for classification, selected on a held-out validation set. In practical deployments, the decision boundary may need to adapt over time as content distributions shift. These points highlight areas where further refinement is needed before UniGADD is suitable for production-level deployment.

## 6 Discussion & Conclusion

We introduced UniGADD, a generator-agnostic deepfake detector that performs well on both seen and unseen manipulations. The method uses a two-stage approach. First, supervised contrastive learning builds a structured embedding space using real and labelled fake examples. Then, a refinement step improves the compactness of real-image clusters, making it easier to detect outliers caused by new manipulations. Results show that UniGADD matches the accuracy of other detectors on known generators and outperforms them when generalising to novel ones. It maintains high accuracy even as more generators are excluded from training. The method also shows low variance across diferent settings, which suggests consistent performance.

We tested UniGADD with diferent backbone models and found that the benefits of refinement hold across architectures. This supports the idea that UniGADD improves generalisation without relying on any one encoder. Because it operates in a frozen feature space, it also avoids the cost of full fine-tuning. Future work will explore adding temporal features for video, testing the method on fewer training samples, and using adversarial training to increase robustness. These changes could make UniGADD more useful in open-world conditions.

UniGADD ofers a simple, efective, and interpretable approach to deepfake detection. Its prototype-based decisions are easy to understand and adapt as new content appears.

## References

Plamen Angelov, Dmitry Kangin, and Ziyang Zhang. Ideal: Interpretable-by-design algorithms for learning from foundation feature spaces. Neurocomputing, 626:129464, 2025. ISSN 0925-2312. doi: https:// doi.org/10.1016/j.neucom.2025.129464. URL https://www.sciencedirect.com/science/article/pii/ S0925231225001365.

Lorenzo Baraldi, Federico Cocchi, Marcella Cornia, Lorenzo Baraldi, Alessandro Nicolosi, and Rita Cucchiara. Contrasting deepfakes difusion via contrastive learning and global-local similarities. In ECCV, 2024.

Jordan J. Bird and Ahmad Lotfi. Cifake: Image classification and explainable identification of ai-generated synthetic images, 2023. URL https://arxiv.org/abs/2303.14126.

Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments. ArXiv, abs/2006.09882, 2020. URL https://api.semanticscholar.org/CorpusID:219721240.

Mathilde Caron, Hugo Touvron, Ishan Misra, Herv’e J’egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 9630–9640, 2021. URL https://api.semanticscholar.org/ CorpusID:233444273.

Chaofan Chen, Oscar Li, Alina Jade Barnett, Jonathan Su, and Cynthia Rudin. This looks like that: deep learning for interpretable image recognition. In Neural Information Processing Systems, 2018. URL https://api.semanticscholar.org/CorpusID:49482223.

Riccardo Corvi, Davide Cozzolino, Giovanni Poggi, Koki Nagano, and Luisa Verdoliva. Intriguing Properties of Synthetic Images: From Generative Adversarial Networks to Difusion Models. In CVPRW, 2023a.

Riccardo Corvi, Davide Cozzolino, Giada Zingarini, Giovanni Poggi, Koki Nagano, and Luisa Verdoliva. On the detection of synthetic images generated by difusion models. In ICASSP, 2023b.

Davide Cozzolino, Giovanni Poggi, Riccardo Corvi, Matthias Nießner, and Luisa Verdoliva. Raising the bar of ai-generated image detection with clip, 2024. URL https://arxiv.org/abs/2312.00195.

Thomas Defard, Aleksandr Setkov, Angelique Loesch, and Romaric Audigier. Padim: a patch distribution modeling framework for anomaly detection and localization, 2020. URL https://arxiv.org/abs/2011. 08785.

Jonathan Donnelly, Alina Jade Barnett, and Chaofan Chen. Deformable protopnet: An interpretable image classifier using deformable prototypes. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10255–10265, 2021. URL https://api.semanticscholar.org/CorpusID: 244729769.

David C Epstein, Ishan Jain, Oliver Wang, and Richard Zhang. Online Detection of AI-Generated Images. In IEEE/CVF International Conference on Computer Vision Workshops, 2023.

Fanda Fan, Chunjie Luo, Wanling Gao, and Jianfeng Zhan. Aigcbench: Comprehensive evaluation of imageto-video content generated by ai. BenchCouncil Transactions on Benchmarks, Standards and Evaluations, pp. 100152, 2024.

Hany Farid. Photo forensics. MIT press, 2016.

Joel Frank, Thorsten Eisenhofer, Lea Schönherr, Asja Fischer, Dorothea Kolossa, and Thorsten Holz. Leveraging frequency analysis for deep fake image recognition. In International Conference on Machine Learning, 2020.

Yue-Hua Han, Tai-Ming Huang, Shu-Tzu Lo, Po-Han Huang, Kai-Lung Hua, and Jun-Cheng Chen. Towards more general video-based deepfake detection through facial feature guided adaptation for foundation model, 2024. URL https://arxiv.org/abs/2404.05583.

Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. Supervised contrastive learning. ArXiv, abs/2004.11362, 2020. URL https: //api.semanticscholar.org/CorpusID:216080787.

Nicolas Larue, Ngoc-Son Vu, Vitomir Struc, Peter Peer, and Vassilis Christophides. Seeable: Soft discrepancies and bounded contrastive learning for exposing deepfakes, 2023. URL https://arxiv.org/abs/ 2211.11296.

Amal Naitali, Mohammed Ridouani, Fatima Salahdine, and Naima Kaabouch. Deepfake attacks: Generation, detection, datasets, challenges, and research directions. Computers, 12(10), 2023. ISSN 2073-431X. URL https://www.mdpi.com/2073-431X/12/10/216.

Utkarsh Ojha, Yuheng Li, and Yong Jae Lee. Towards Universal Fake Image Detectors That Generalize Across Generative Models. In CVPR, 2023.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision, 2024. URL https://arxiv.org/abs/2304.07193.

Alvaro Lopez Pellicer, Yi Li, and Plamen Angelov. Pudd: Towards robust multi-modal prototype-based deepfake detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pp. 3809–3817, June 2024.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021a. URL https://arxiv.org/abs/2103.00020.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021b. URL https://arxiv.org/abs/2103.00020.

Md Shohel Rana, Mohammad Nur Nobi, Beddhu Murali, and Andrew H. Sung. Deepfake detection: A systematic literature review. IEEE Access, 10:25494–25513, 2022. doi: 10.1109/ACCESS.2022.3154404.

Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Schölkopf, Thomas Brox, and Peter Gehler. Towards total recall in industrial anomaly detection, 2022. URL https://arxiv.org/abs/2106.08265.

Andreas Rössler, Davide Cozzolino, Luisa Verdoliva, Christian Riess, Justus Thies, and Matthias Nießner. Faceforensics++: Learning to detect manipulated facial images, 2019. URL https://arxiv.org/abs/ 1901.08971.

Husrev T Sencar and Nasir Memon. Digital image forensics. Springer, 2013.

Zeyang Sha, Zheng Li, Ning Yu, and Yang Zhang. DE-FAKE: Detection and Attribution of Fake Images Generated by Text-to-Image Generation Models. In ACM CCS, 2023.

Liang Shi, Jie Zhang, and Shiguang Shan. Real face foundation representation learning for generalized deepfake detection, 2023. URL https://arxiv.org/abs/2303.08439.

Jake Snell, Kevin Swersky, and Richard S. Zemel. Prototypical networks for few-shot learning. In Neural Information Processing Systems, 2017. URL https://api.semanticscholar.org/CorpusID:309759.

Zhimin Sun, Shen Chen, Taiping Yao, Bangjie Yin, Ran Yi, Shouhong Ding, and Lizhuang Ma. Contrastive pseudo learning for open-world deepfake attribution, 2023. URL https://arxiv.org/abs/2309.11132.

Run Wang, Felix Juefei-Xu, Lei Ma, Xiaofei Xie, Yihao Huang, Jian Wang, and Yang Liu. FakeSpotter: A Simple yet Robust Baseline for Spotting AI-Synthesized Fake Faces. In IJCAI, 2020a.

Sheng-Yu Wang, Oliver Wang, Richard Zhang, Andrew Owens, and Alexei A Efros. CNN-Generated Images Are Surprisingly Easy to Spot... for Now. In CVPR, 2020b.

Tianyi Wang, Xin Liao, Kam Pui Chow, Xiaodong Lin, and Yinglong Wang. Deepfake detection: A comprehensive survey from the reliability perspective. ACM Computing Surveys, 57(3), 2024. ISSN 0360-0300. doi: 10.1145/3699710. URL https://doi.org/10.1145/3699710.

Zhendong Wang, Jianmin Bao, Wengang Zhou, Weilun Wang, Hezhen Hu, Hong Chen, and Houqiang Li. DIRE for Difusion-Generated Image Detection. In ICCV, 2023.

Xi Weng, Jianing An, Xudong Ma, Binhang Qi, Jie Luo, Xi Yang, Jin Song Dong, and Lei Huang. Clustering properties of self-supervised learning. ArXiv, abs/2501.18452, 2025. URL https://api. semanticscholar.org/CorpusID:275993509.

Mike Wu, Noah D. Goodman, Chris Piech, and Chelsea Finn. Prototransformer: A meta-learning approach to providing student feedback. ArXiv, abs/2107.14035, 2021. URL https://api.semanticscholar.org/ CorpusID:236493494.

Xin Yang, Yuezun Li, and Siwei Lyu. Exposing Deep Fakes Using Inconsistent Head Poses. In ICASSP, 2019.

Mingjian Zhu, Hanting Chen, Qiangyu Yan, Xudong Huang, Guanyu Lin, Wei Li, Zhijun Tu, Hailin Hu, Jie Hu, and Yunhe Wang. Genimage: A million-scale benchmark for detecting ai-generated image. Advances in Neural Information Processing Systems, 36:77771–77782, 2023.

## A Additional Ablation Studies

<table><tr><td>Clusters</td><td> $n _ { \mathbf { r e a l } }$ </td><td>Ratio</td><td>Seen</td><td>Unseen</td></tr><tr><td>20</td><td>3</td><td>0.19</td><td>87.6</td><td>45.8</td></tr><tr><td>30</td><td>13</td><td>0.81</td><td>87.0</td><td>54.8</td></tr><tr><td>40</td><td>23</td><td>1.44</td><td>87.4</td><td>54.0</td></tr><tr><td>60</td><td>43</td><td>2.69</td><td>88.4</td><td>54.5</td></tr><tr><td>80</td><td>63</td><td>3.94</td><td>88.3</td><td>52.2</td></tr><tr><td>100</td><td>83</td><td>5.19</td><td>88.6</td><td>56.6</td></tr></table>

Table 4: Cluster ablation showing the efect of varying the total number of clusters.

<table><tr><td>σ</td><td>Seen Acc</td><td>Unseen Acc</td></tr><tr><td>1.0</td><td>91.5</td><td>46.2</td></tr><tr><td>1.5</td><td>90.4</td><td>51.3</td></tr><tr><td>2.0</td><td>89.7</td><td>54.0</td></tr><tr><td>2.5</td><td>89.3</td><td>55.2</td></tr><tr><td>3.0</td><td>89.2</td><td>55.7</td></tr><tr><td>3.5</td><td>89.1</td><td>56.0</td></tr><tr><td>4.0</td><td>89.0</td><td>56.2</td></tr></table>

Table 5: Threshold ablation showing the efect of varying σ on detection accuracy.

<table><tr><td>Backbone</td><td>Weighted Acc</td><td>Silhouette</td></tr><tr><td>ViT-B-32</td><td>80.10</td><td>0.328</td></tr><tr><td>ViT-L-14</td><td>83.42</td><td>0.312</td></tr><tr><td> $\mathrm { { D I N O v 2 – B } / 1 4 }$ </td><td>82.66</td><td>0.293</td></tr><tr><td> $\mathrm { { D I N O v 2 – S } / 1 4 }$ </td><td>81.64</td><td>0.278</td></tr></table>

Table 6: Extended backbone comparison showing weighted accuracy and embedding quality (silhouette score).

<table><tr><td>Loss Configuration</td><td>Seen</td><td>Unseen</td></tr><tr><td> $L _ { \mathrm { c o n t r a s t } } + L _ { \mathrm { s p } }$ </td><td>89.1</td><td>71.4</td></tr><tr><td> $L _ { \mathrm { c o n t r a s t } } + L _ { \mathrm { w p } }$ </td><td>91.4</td><td>73.0</td></tr><tr><td> $L _ { \mathrm { c o n t r a s t } } + L _ { \mathrm { r e f i n e } }$ </td><td>87.9</td><td>61.2</td></tr></table>

Table 7: Loss ablation comparing diferent refinement strategies.