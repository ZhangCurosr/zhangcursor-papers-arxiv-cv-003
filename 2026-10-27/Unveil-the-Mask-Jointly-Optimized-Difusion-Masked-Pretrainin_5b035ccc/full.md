# Unveil the Mask: Jointly-Optimized Difusion Masked Pretraining for Appearance-Agnostic Operator Verification

Katsuya Ogata<sup>1</sup>, Ryoya Yoshiyama<sup>1</sup>, Kawachi Hodaka<sup>1</sup>, and Yuta Nakashima<sup>1</sup>

The University of Osaka

{ogata, yoshiyama, kawachi, n-yuta}@im.sanken.osaka-u.ac.jp

Abstract. Photorealistic avatar videos decouple the appearance of a rendered face from the person who operates it, which makes conventional face recognition unsuitable for identifying the operator. Verifying the operator therefore requires representations of how a face moves that are insensitive to what the avatar looks like. We propose a verification framework built without any external pretrained model. It uses two complementary modalities provided for each sample: facial landmark sequences and video frames. From the landmark modality, it learns motion representations by difusion masked pretraining over grouped landmarks, guided by lightweight auxiliary tasks that predict behavioral cues such as motion intensity and eye openness, and turns them into operator embeddings with a contrastive linear probe. To generalize to unseen avatar generators, the pretraining input is randomly transformed to simulate how a generator may alter the operator’s motion. From the video modality, it captures fine facial motion through diferences between consecutive frames. The embeddings of the two modalities are combined into one. Our approach reaches an overall AUC of 0.769 on the WBA2026 benchmark. Linear probing shows that the video embedding retains little information about the avatar’s identity, whereas the landmark embedding retains some, and fusing the two modalities improves verification in particular across speaking and listening modes.

Keywords: Operator Verification · Difusion Masked Pretraining

## 1 Introduction

The rapid evolution of face reenactment (animating one person’s face with another person’s facial motion) has enabled photorealistic 2D and 3D avatars in telepresence, virtual reality, online education, and AI-mediated communication. However, decoupling facial appearance from the physical operator opens unprecedented security vulnerabilities. Malicious actors can manipulate avatars to impersonate legitimate operators, rendering traditional biometric authentication, such as image-based face recognition, fundamentally obsolete, as the observed texture and facial geometry belong to the synthesized avatar rather than the operator behind it.

The Who is Behind the Avatar (WBA) task introduces the challenge of appearance-agnostic operator identity verification: identifying the operator, called the driver in the task, regardless of the avatar being animated, called the source. This task presents three critical dificulties:

1. Appearance as a Strong Confounder: The static geometry and visual texture of the source identity constitute dominant signals in standard neural encoders, making models prone to shortcut learning where identity verification collapses into avatar matching.

2. Distinct Speaking and Listening Behaviors: Drivers exhibit diferent movement characteristics between active speech (speaking mode, dominated by large-amplitude lip and jaw kinematics) and passive reception (listening mode, dominated by subtle micro-expressions, gaze shifts, and head nodding).

3. Cross-Generator Generalization: Maintaining invariant driver signatures is highly challenging when avatars are generated across distinct rendering engines and unseen generation pipelines.

To address these challenges, we present a self-contained framework that learns appearance-invariant dynamic behavioral signatures without external pretrained weights or distillation. Our framework combines difusion masked pretraining on appearance-normalized landmark tokens and lightweight auxiliary behavioral tasks with a driver-aware contrastive linear probe on the frozen encoder, in which every sample has negatives rendered on the same avatar. To prepare the landmark branch for unseen generators, we randomly transform the landmark motion during pretraining, simulating how a generator may alter it. We fuse this landmark branch with a video branch that learns from diferences between consecutive frames of landmark-guided face crops. Our final system reaches an overall AUC of 0.769 on the WBA2026 test sets.

## 2 Related Work

## 2.1 Behavioral Biometrics and Facial Dynamics

Behavioral biometrics identify individuals by characteristic motion patterns rather than static appearance. Gait recognition is a representative example, with extensive work from Takemura et al. demonstrating person verification under nuisance factors such as viewpoint changes and unconstrained video conditions [11, 19]. More recently, continuous face authentication has extended biometric recognition to streaming facial observations, explicitly suppressing identity-irrelevant variations over time [9,10]. Beyond static facial appearance, prior studies have shown that head and mouth motion, facial-expression dynamics, action-unit statistics, and even micro-expressions contain person-specific behavioral cues [7,14,16,17]. Motivated by these findings, we represent identity through dense landmark trajectories, focusing on dynamic facial behavior while minimizing dependence on avatar appearance.

## 2.2 Self-Supervised Predictive and Masked Modeling

Prediction-based pretraining has become a general mechanism for learning transferable representations without task-specific labels. Autoregressive language models learn from next-token prediction [15], while BERT introduced masked-token prediction to exploit bidirectional context [3]. The same principle was subsequently extended to visual data through masked autoencoding [4], and to temporal and articulated signals through VideoMAE [20], MotionBERT [23], and masked motion prediction [12]. These objectives encourage representations governed by the intrinsic structure of the input rather than a particular downstream decision boundary. Most recently, Difusion Masked Pretraining (DiMP) [22] combines masked modeling with difusion to robustly infer missing spatio-temporal structure. We adapt this principle to facial landmark sequences.

## 3 Task Formulation

The WBA task [21] asks whether two avatar videos are driven by the same person, while the rendered avatars may difer.

## 3.1 Problem Setting

Each sample ${ \pmb x } = ( { \pmb L } , { \pmb V } )$ is an avatar video with two synchronized observations: $- \textrm { a } 2 \mathrm { D }$ facial landmark sequence $\pmb { L } \in \mathbb { R } ^ { T \times K \times 2 }$ of T frames and K landmarks $( K = 1 0 6 )$ , given in the pixel coordinates of the rendered frames; and – the rendered video $V \in \dot { \mathbb { R } } ^ { T \times H \times W \times 3 }$ with $H = W = 5 1 2$

Every driver-source pair contributes two speaking and two listening samples. Test samples carry none of these labels; in particular, the mode of a test sample is unknown at inference time. Furthermore, the test set is divided into two subsets to evaluate cross-generator generalization: Same Gen consists of avatars generated by the same rendering pipelines as the training set, while Dif Gen features avatars rendered by distinct, unseen generation pipelines.

The number T of frames is at most 100, and almost all samples have $T = 1 0 0$ (98.1% of the training samples, 98.5% of Same Gen and all of Dif Gen). The shortest samples have $T = 3 3$ (training) and $T = 5$ (Same Gen).

Table 1 summarizes the dataset. For model selection, we hold out 20 drivers and 12 sources that do not appear in our training split, and use the samples of these identities as the validation set.

The objective of this task is to build an encoder f that maps each sample to a unit-norm embedding

$$
z = f ( \pmb { x } ) \in \mathbb { R } ^ { D } , \qquad \| z \| _ { 2 } = 1 ,\tag{1}
$$

such that the cosine similarity cos $( z _ { i } , z _ { j } ) = z _ { i } z _ { j } ^ { \top }$ is high if $d _ { i } \ = \ d _ { j }$ and low otherwise, regardless of $s _ { i }$ and $s _ { j } ,$ for samples i and $j$ . The challenge protocol accepts only one embedding per test sample, together with the choice of a cosine similarity or Euclidean distance (Eq. (1) is for cosine similarity).

Table 1: Dataset Statistics. The validation set is held out from the oficial training data; its drivers and sources do not occur in our training split.
<table><tr><td colspan="4">Split Samples Drivers Sources</td></tr><tr><td>Official training data</td><td>89,133</td><td>201</td><td>111</td></tr><tr><td>Our training split</td><td>71,577</td><td>181</td><td>99</td></tr><tr><td>Our validation split</td><td>960</td><td>20</td><td>12</td></tr><tr><td>Test, Same Gen (unlabeled)</td><td>40,200</td><td></td><td></td></tr><tr><td>Test, Diff Gen (unlabeled)</td><td>40,200</td><td></td><td></td></tr></table>

## 3.2 Evaluation Metrics

Test samples are divided into probes and galleries, and verification is scored by the area under the ROC curve (AUC) over probe-gallery pairs. AUCs are computed separately along two axes:

Mode combination of probe and gallery: S→S, S→L, L→S and L→L (S: speaking, L: listening).

– Generator condition: pairs rendered by the same generator (Same Gen) and by diferent generators (Dif Gen).

The Same Gen and Dif Gen scores are each the mean of the four mode-combination AUCs, and the overall score is the mean of Same Gen and Dif Gen. On our validation split, which has no generator label, we report a single mean AUC : the average of the S→S AUC, the L→L AUC and one cross-mode AUC computed over the S→L and L→S pairs together.

## 4 Proposed Method

Our framework consists of two branches that are trained independently from scratch (Fig. 1). The landmark branch (Secs. 4.1 to 4.3 and 4.5) learns a motion representation by self-supervised difusion masked pretraining (Stage 1) and turns it into a driver embedding with a linear contrastive probe on the frozen encoder (Stage 2). The video branch (Sec. 4.6) learns driver embeddings directly from inter-frame feature diferences of landmark-guided face crops. Several models are ensembled in each branch, and the two branches are fused by weighted concatenation of their normalized embeddings (Sec. 4.7).

## 4.1 Appearance-Agnostic Landmark Tokenization

Removing the avatar’s static features. The landmark coordinates mix two kinds of information: the static geometry of the rendered face, which belongs to the avatar, and how the face moves over time, which is controlled by the driver. A model can therefore verify identities by recognizing the avatar’s face shape instead of the driver’s motion. We remove the static part with four normalization steps applied to each sample:

![](images/8911c152591de70c6976a6c9f7932b584f4674a87d0a6648abbeab92d05c511d.jpg)  
Fig. 1: Overview of our framework.

1. In every frame, the centroid of the 106 landmarks in that frame is subtracted from the coordinates of each landmark, removing the position of the face in the image.

2. The coordinates in all frames are divided by a single scale, the mean distance of the landmarks to their centroid, removing the size of the face.

3. For each landmark, its mean position over the frames of the sample is subtracted from its position in every frame, removing the avatar’s face shape (e.g., eye spacing and jaw outline). What remains is the displacement of each landmark in each frame from its temporal mean position.

4. The displacements are divided by their mean Euclidean norm over all frames and landmarks of the sample, so that every sample has the same overall motion amplitude while the relative amplitudes across landmarks and frames are kept.

Sequences shorter than 100 frames are padded by repeating their last frame, and during training we add small Gaussian noise as data augmentation.

Spatio-temporal tokens. The 106 landmarks are partitioned into G groups, and each group over l consecutive frames (l = 4) forms one spatio-temporal token (called a tube in DiMP [22]), giving G × T /l tokens per sample. We use three partitioning strategies (Fig. 2):

– Geometric tokens (Geo): 16 spatially compact groups (G = 16) of 6 to 7 landmarks, obtained by recursive median bisection of the mean face shape along its axis of largest spread (Fig. 2a). At each recursion, landmarks are bisected along the coordinate axis of maximum variance. The mean face shape is computed once from the training samples, so the same partition is used for every sample and frame. The resulting clusters (shaded regions in Fig. 2a) are purely spatial and ignore facial anatomy, mixing, e.g., eye and jaw landmarks. This yields 400 tokens per sample.

![](images/6811f1c00c32d9a2aacb4bff31f0a3b49f1ee2b856ccd39cffeeac289bbaf1d1.jpg)  
(a) Geometric Tokens (G = 16)

![](images/12bc10c10cd0c74c5dd76871082c8f398702b3ed9ce6d43aae6b8ace38309301.jpg)  
(b) Anatomical Tokens (G = 10)

![](images/8b79812b74a5bb1b9c8b3bedcabc5a934ed504bf16ab76b3c77d3aee537041dd.jpg)  
(c) Motion Tokens (G = 16)  
Fig. 2: Comparison of landmark tokenization strategies. (a) Geometric tokens (G = 16), obtained by recursive median bisection of the mean face shape along its axis of largest spread. Dashed lines illustrate the hierarchical bisection cuts (red: horizontal, blue: vertical), and shaded regions delineate resulting spatially compact clusters that cross anatomical boundaries. (b) Anatomical tokens (G = 10), partitioned by functional facial units on the canonical 106-point topology. (c) Motion tokens (G = 16), obtained via recursive spectral bisection of the landmark displacement correlation graph, grouping landmarks that exhibit synchronized dynamics.

– Anatomical tokens (Anat): 10 groups (G = 10) defined by a structured partitioning of the 106-point topology into functional facial units (Fig. 2b): the jaw contour split into three contiguous segments (11 landmarks each), the left and right eye (10 contour landmarks plus the pupil each), the left and right eyebrow (9 each), the nose (4 bridge and 9 tip landmarks), the outer lip contour (12), and the inner lip contour (8). This preserves natural kinematic boundaries and yields 250 tokens per sample.

– Motion tokens (Motion): 16 groups (G = 16) of landmarks that move together (Fig. 2c). We build a graph over the landmarks whose edge weights are the positive parts of the correlations between the frame-to-frame displacements of two landmarks over the training samples, and split it by recursive spectral bisection into groups of equal size. Like Geo, this yields 400 tokens per sample.

The embedding of a token is the sum of three terms: (i) a content embedding, a two-layer MLP applied to the flattened coordinates of the token; (ii) a positional embedding, an MLP applied to the token’s centroid and its time index, where the centroid is the mean position of the token’s real landmarks over its four frames, measured relative to the mean of all centroids of the sample; it tells the model where on the face and when the token lies; and (iii) a learnable group embedding, one vector per group, which tells the model which facial part the token belongs to.

## 4.2 Difusion Masked Pretraining (Stage 1)

We adapt Difusion Masked Pretraining (DiMP) [22] to landmark tokens. DiMP combines masked modeling with two difusion processes: one corrupts the masked tokens, whose clean centroids and coordinates must be recovered, and one models their motion.

Masking and corruption. Let T be the set of tokens of a sample, $| \mathcal { T } | = G \times T / l$ Each token $n \in \tau$ consists of $K _ { n }$ landmarks observed over l frames. We denote their normalized 2D positions by $\pmb { P } _ { n } \in \mathbb { R } ^ { l \times K _ { n } \times 2 }$ , their centroid by $\boldsymbol { c } _ { n } \in \mathbb { R } ^ { 2 }$ their displacements between consecutive frames by $D _ { n } \in \mathbb { R } ^ { ( l - 1 ) \times K _ { n } \times 2 } .$ , and by $\phi _ { n } ^ { \mathrm { e n c } } ( { \pmb c } )$ the sum of the positional embedding of a centroid c at the token’s time index and the token’s group embedding (Sec. 4.1). For each sample, a random subset $\mathcal { M } \subset \mathcal { T }$ of 60% of the tokens is masked and corrupted, while the other tokens, the visible tokens $\mathcal { V } = \mathcal { T } \backslash \mathcal { M }$ , remain intact. As in denoising difusion models [5], the amount of corruption is set by a timestep $t = 1 , \ldots , T _ { \mathrm { d } }$ , where $T _ { \mathrm { d } } = 1 0 0 0$ , through a cosine scheduling $\bar { \alpha } _ { t }$ , which decreases from nearly 1 (almost no noise) to nearly 0 (almost pure noise) as t grows. For each sample we draw one such timestep, denoted $t _ { \mathrm { c } } ,$ following DiMP, uniformly from 1 to $T _ { \mathrm { { d } } } .$ , and share it among all masked tokens; their coordinates and centroids are corrupted with two independent noise samples. That is, for $n \in \mathcal { M }$

$$
\begin{array} { r } { P _ { n } ^ { t _ { c } } = \sqrt { \bar { \alpha } _ { t _ { c } } } P _ { n } + \sqrt { 1 - \bar { \alpha } _ { t _ { c } } } \epsilon _ { n } , \qquad c _ { n } ^ { t _ { c } } = \sqrt { \bar { \alpha } _ { t _ { c } } } c _ { n } + \sqrt { 1 - \bar { \alpha } _ { t _ { c } } } \epsilon _ { n } ^ { \prime } , } \end{array}\tag{2}
$$

where $\epsilon _ { n } \in \mathbb { R } ^ { l \times K _ { n } \times 2 }$ and $\epsilon _ { n } ^ { \prime } \in \mathbb { R } ^ { 2 }$ have the same shapes as $\scriptstyle { P _ { n } }$ and $^ { c _ { n } , }$ and each of their elements is drawn independently from the standard Gaussian distribution.

Encoder. The encoder, a Transformer with hidden dimension h, keeps the visible and masked tokens in separate streams. Each visible token is embedded from its clean coordinates and each masked token from its corrupted ones,

$$
\begin{array} { r } { \pmb { e } _ { n } = \left\{ \phi ^ { \mathrm { v } } ( \pmb { P } _ { n } ) + \phi _ { n } ^ { \mathrm { e n c } } ( \pmb { c } _ { n } ) , n \in \mathcal { V } , \right. } \\ { \left. \phi ^ { \mathrm { m } } ( \pmb { P } _ { n } ^ { t _ { \mathrm { c } } } ) + \phi _ { n } ^ { \mathrm { e n c } } ( \pmb { c } _ { n } ^ { t _ { \mathrm { c } } } ) , n \in \mathcal { M } , \right. } \end{array}\tag{3}
$$

where $\phi ^ { \mathrm { v } }$ and $\phi ^ { \mathrm { m } }$ are separate embedding networks for clean and corrupted coordinates; masked tokens are thus placed at their corrupted centroids. Stacking the embeddings $e _ { n }$ as rows gives the inputs of the two streams at layer $\ell = 0 .$ $Z _ { \ell } ^ { \mathrm { v } } \in \mathbb { R } ^ { | \mathcal { V } | \times h }$ for the visible tokens and $Z _ { \ell } ^ { \mathrm { m } } \in \mathbb { R } ^ { | \mathcal { M } | \times h }$ for the masked tokens. In layer $\ell ,$ with $\operatorname { A t t n } ( Q , K , V ) = \operatorname { s o f t m a x } ( Q K ^ { \top } / \sqrt { h } ) V$ , the two streams are updated as

$$
\begin{array} { r } { \pmb { Z } _ { \ell + 1 } ^ { \mathrm { v } } = \pmb { Z } _ { \ell } ^ { \mathrm { v } } + \mathrm { A t t n } \big ( \pmb { Z } _ { \ell } ^ { \mathrm { v } } \pmb { W } _ { \mathrm { Q } } , \pmb { Z } _ { \ell } ^ { \mathrm { v } } \pmb { W } _ { \mathrm { K } } , \ \pmb { Z } _ { \ell } ^ { \mathrm { v } } \pmb { W } _ { \mathrm { V } } \big ) , } \end{array}\tag{4}
$$

$$
\begin{array} { r } { \pmb { Z } _ { \ell + 1 } ^ { \mathrm { m } } = \pmb { Z } _ { \ell } ^ { \mathrm { m } } + \mathrm { A t t n } \big ( \pmb { Z } _ { \ell } ^ { \mathrm { m } } \pmb { W } _ { \mathrm { Q } } ^ { \prime } , \pmb { Z } _ { \ell } ^ { \mathrm { v } } \pmb { W } _ { \mathrm { K } } ^ { \prime } , \pmb { Z } _ { \ell } ^ { \mathrm { v } } \pmb { W } _ { \mathrm { V } } ^ { \prime } \big ) , } \end{array}\tag{5}
$$

where the feed-forward network and layer normalization of each layer are omitted for brevity. The outputs of the last layer, after a final layer normalization, are denoted $Z ^ { \mathrm { v } }$ and $Z ^ { \mathrm { m } }$ , and their rows by $ { \boldsymbol { z } } _ { n } ^ { \mathrm { v } }$ and $z _ { n } ^ { \mathrm { m } }$ . Masked tokens only issue queries: their keys and values come from the visible tokens, so a corrupted token is inferred from the clean context alone and never from other corrupted tokens. Conversely, visible tokens never attend to masked ones, so the visible stream is unafected by the corruption.

Centroid prediction. A small MLP predicts the clean centroid $\hat { \pmb { c } } _ { n }$ of each masked token, $i . e .$ , the mean position of its $K _ { n }$ landmarks over its l frames, from the encoder output $z _ { n } ^ { \mathrm { m } }$ , with the loss

$$
\mathcal { L } _ { \mathrm { c e n } } = \frac { 1 } { \left| \mathcal { M } \right| } \sum _ { n \in \mathcal { M } } \left\| \hat { \pmb { c } } _ { n } - \pmb { c } _ { n } \right\| _ { 2 } ^ { 2 } .\tag{6}
$$

Decoder and reconstruction. The decoder $\psi ,$ a stack of self-attention layers, takes the visible and masked tokens as one sequence,

$$
\begin{array} { r } { \pmb { d } _ { n } ^ { \mathrm { v } } = z _ { n } ^ { \mathrm { v } } + \phi _ { n } ^ { \mathrm { d e c } } ( \pmb { c } _ { n } ) + \pmb { \tau } ( t _ { \mathrm { c } } ) , } \end{array}
$$

$$
n \in \mathcal { V } ,\tag{7}
$$

$$
\begin{array} { r } { \pmb { d } _ { n } ^ { \mathrm { m } } = \phi ^ { \mathrm { m } } ( \pmb { P } _ { n } ^ { t _ { \mathrm { c } } } ) + \phi _ { n } ^ { \mathrm { d e c } } \big ( \mathrm { s g } ( \hat { \pmb { c } } _ { n } ) \big ) + \pmb { \tau } ( t _ { \mathrm { c } } ) , \qquad \qquad n \in \mathcal { M } , } \end{array}\tag{8}
$$

$$
\{ \pmb { u } _ { n } \} _ { n \in \mathcal { T } } = \psi \big ( \{ \pmb { d } _ { n } ^ { \mathrm { v } } \} _ { n \in \mathcal { V } } \cup \{ \pmb { d } _ { n } ^ { \mathrm { m } } \} _ { n \in \mathcal { M } } \big ) ,\tag{9}
$$

where $\phi _ { n } ^ { \mathrm { d e c } }$ is the positional and group embedding of the decoder, of the same form as $\phi _ { n } ^ { \mathrm { e n c } }$ but with its own weights; sg denotes stop-gradient, and $\tau ( t _ { \mathrm { c } } )$ is an embedding of the timestep $t _ { \mathrm { c } } .$ . Visible tokens thus enter as their encoder outputs placed at their true centroids, while masked tokens enter as their corrupted content embeddings placed at the predicted centroids, so that the decoder never sees the true positions of the masked tokens. From the decoder output $\mathbf { \Delta } \pmb { u } _ { n }$ for $n \in { \mathcal { M } } .$ a linear head reconstructs its clean coordinates ${ \hat { P } } _ { n }$ . Unlike ${ \mathcal { L } } _ { \mathrm { c e n } } .$ , the loss $\mathcal { L } _ { \mathrm { g e o } }$ compares the landmarks frame by frame: with $P _ { n } ^ { ( j ) } \in \mathbb { R } ^ { K _ { n } \times 2 }$ denoting the j-th frame of $\scriptstyle { P _ { n } }$

$$
\mathcal { L } _ { \mathrm { g e o } } = \frac { 1 } { | \mathcal { M } | l } \sum _ { n \in \mathcal { M } } \sum _ { j = 1 } ^ { l } \mathrm { C D } \big ( \hat { P } _ { n } ^ { ( j ) } , P _ { n } ^ { ( j ) } \big ) ,\tag{10}
$$

where CD is the Chamfer distance between two sets of $K _ { n }$ points.

Motion difusion. A second, independent difusion process models the displacements $D _ { n }$ of the masked tokens. To cover weak and strong noise evenly, the range $\{ 1 , \dots , T _ { \mathrm { d } } \}$ is split into κ consecutive intervals of equal width $( \kappa = 4 )$ , and one timestep is drawn uniformly from each interval, independently of $t _ { \mathrm { c } }$ . At each of these timesteps, noise is added to $D _ { n }$ as in Eq. (2), and a head conditioned on ${ \pmb u } _ { n }$ predicts the added noise; ${ \mathcal { L } } _ { \mathrm { m o t } }$ is the resulting noise-prediction error, averaged over the κ timesteps and computed over real landmarks. Optionally, as in classifier-free guidance training [6], the condition $\mathbf { \Delta } \pmb { u } _ { n }$ is replaced by a learned null vector for a random subset of the masked tokens during training (condition dropout); no guidance is used, since the motion head is discarded after Stage 1.

Because the head is conditioned on ${ \pmb u } _ { n }$ , this loss reaches the encoder through the decoder.

The pretraining loss is

$$
\mathcal { L } _ { \mathrm { p r e } } = \mathcal { L } _ { \mathrm { g e o } } + \gamma _ { \mathrm { c e n } } \mathcal { L } _ { \mathrm { c e n } } + \lambda _ { \mathrm { m o t } } \mathcal { L } _ { \mathrm { m o t } } ,\tag{11}
$$

where $\gamma _ { \mathrm { c e n } }$ and $\lambda _ { \mathrm { m o t } }$ are loss weights (Sec. 5). As a variant, the reconstruction head can instead predict the displacements $D _ { n }$ of the masked tokens; $\mathcal { L } _ { \mathrm { g e o } }$ is then replaced by the mean squared error of the displacements, divided by their root mean square in the batch (velocity reconstruction).

## 4.3 Auxiliary Behavioral Tasks

To steer the encoder towards behavioral cues, three auxiliary regression tasks are added to Stage 1. Their targets are computed in closed form from the input landmarks, so they require no labels. Each task has a small MLP head on the encoder’s visible-token features, and its gradient reaches the encoder.

– Motion Energy (ME): for every visible token, the mean squared interframe displacement of its landmarks, predicted from that token’s own feature.

– Region Motion (RM): for every annotated region except the jaw contour (both eyes, both pupils, both eyebrows, nose bridge, nose tip, outer and inner lip), the mean squared inter-frame displacement over the sample, predicted from the mean of the visible-token features.

– Eye Shape (ES): for each eye, the mean and the standard deviation over the sample of its openness, measured as the height-to-width ratio of the bounding box of the eye landmarks (a blink proxy), predicted from the mean of the visible-token features.

All three are mean squared errors, added with a small weight:

$$
{ \mathcal { L } } _ { \mathrm { S t a g e 1 } } = { \mathcal { L } } _ { \mathrm { p r e } } + w \left( { \mathcal { L } } _ { \mathrm { M E } } + { \mathcal { L } } _ { \mathrm { R M } } + { \mathcal { L } } _ { \mathrm { E S } } \right) , \qquad w = 0 . 0 1 .\tag{12}
$$

## 4.4 Pseudo-Generator Randomization

The generators of Dif Gen are not seen in training, and a diferent generator may reproduce the driver’s motion with diferent timing, smoothness or tracking noise. We model a generator as an unknown transformation of the driver’s motion and simulate it during Stage 1. Each training sample is transformed by one of six families, chosen uniformly at random: (i) identity; (ii) a smooth global time warp; (iii) a time shift of each facial part; (iv) a frequency response of each facial part, either a low-pass filter or a boost of high frequencies; (v) temporally correlated noise on each landmark, mimicking tracking errors; and (vi) a composition of two of (ii)–(v). The facial parts are those of the Anat tokens, and each transformed sample is normalized again by steps 3 and 4 of Sec. 4.1. Following GroupDRO [18], the motion loss is computed separately for each family k as $\mathcal { L } _ { \mathrm { m o t } , k }$ and replaced by $\begin{array} { r } { \sum _ { k } q _ { k } \mathcal { L } _ { \mathrm { m o t } , k } } \end{array}$ , where the weights are updated at every step as $q _ { k } \gets q _ { k } \exp ( \eta \mathcal { L } _ { \mathrm { m o t } , k } )$ and renormalized to sum to one, so that the families whose motion is predicted worst are emphasized.

## 4.5 Driver-Aware Contrastive Probing (Stage 2)

In Stage 2 the pretrained encoder is frozen, and the masked stream, the decoder and all Stage 1 heads are discarded. All tokens pass through the visible stream, the token features are averaged, and a single linear layer followed by normalization produces the driver embedding. Only this linear layer is trained.

The linear layer is trained with the supervised contrastive (InfoNCE) loss [8, 13] with temperature τ :

$$
\mathcal { L } _ { \mathrm { c o n } } = - \frac { 1 } { | \mathcal { B } | } \sum _ { i \in \mathcal { B } } \frac { 1 } { | \mathcal { Z } ( i ) | } \sum _ { j \in \mathcal { Z } ( i ) } \log \frac { \exp ( z _ { i } z _ { j } ^ { \top } / \tau ) } { \sum _ { \iota \in \mathcal { B } \backslash \{ i \} } \exp ( z _ { i } z _ { \iota } ^ { \top } / \tau ) } ,\tag{13}
$$

where $\boldsymbol { B }$ is a batch, $z _ { i }$ is the embedding of sample i, and ${ \mathcal { T } } ( i ) \subset B \setminus \{ i \}$ is the set of the other samples in B of the same driver as sample i.

Batches are built by crossing drivers with avatars. Each batch contains several drivers and several sources with which all of these drivers have been recorded, and one sample for each (driver, source) combination. Consequently, every positive pair shows the same driver on diferent avatars, and every sample has negatives rendered on the same avatar by diferent drivers, so matching avatar appearance cannot solve the objective. For encoders pretrained with pseudo-generator randomization, the same transformations (Sec. 4.4) are applied to the Stage 2 input.

## 4.6 Video Branch on Landmark-Guided Face Crops

Landmarks capture only the motion of 106 tracked points, whereas the rendered frames also contain the dense motion, such as that of the cheeks and the skin around the eyes and mouth. Diferencing consecutive frames emphasizes such changes between frames and suppresses the static features of each frame, so that the video branch can provide motion cues complementary to the landmarks.

The video branch follows the inter-frame feature diferencing model of Chapariniya et al. [2], trained from scratch on grayscale frames.

Face crop. Shrinking the full $5 1 2 \times 5 1 2$ frame to the network input leaves the face only about 60 pixels wide, at which point the sub-pixel motion of eyelids and lips is largely lost. We therefore crop the face using the challenge landmarks (no external face detector). For each sample, a square box is centered on the bounding box of all landmarks over all frames, with a side proportional to its longer edge. The same box is used for every frame, so head motion is preserved and the crop itself introduces no motion. During training only, the box size and center are randomly perturbed per sample. Regions outside the image are filled with zeros. The crop is resized to $S \times S$ pixels and normalized per sample to zero mean and unit variance. We also train models on the full frames without cropping, which keep the head motion relative to the image.

Network and training. A per-frame convolutional backbone (a strided convolution stack, a meta-kernel convolution block and a nearest-neighbor edge convolution) maps each frame $I _ { f } \in V$ to a feature map $F ( I _ { f } )$ . We use two variants: feature diferencing diferences consecutive feature maps, $F ( I _ { f + 1 } ) - F ( I _ { f } )$ , and pixel diferencing [2] diferences the frames before the backbone, $F ( I _ { f + 1 } - I _ { f } )$ The diferences are stacked over time and aggregated by a 3D-convolutional temporal head into an L2-normalized embedding. The loss is Eq. (13) plus a weighted cross-source alignment term, the mean cosine distance between embeddings of the same driver on diferent sources in the batch. Batches use the same driver–source crossing as Stage 2.

## 4.7 Ensembling and Score-Level Fusion

Because Stage 1 depends on the random seed, and models with diferent tokenizations or input modalities capture diferent cues, we ensemble n landmark models and $n _ { \mathrm { V } }$ video models. Given member embeddings $z _ { 1 } ^ { \mathrm { L } } , \ldots , z _ { n _ { \mathrm { L } } } ^ { \mathrm { L } }$ and $z _ { 1 } ^ { \mathrm { V } } , \ldots , z _ { n _ { \mathrm { V } } } ^ { \mathrm { V } }$ of a sample, the submitted embedding is their concatenation:

$$
\begin{array} { r } { z = \left[ \sqrt { \frac { \alpha } { n _ { \mathrm { L } } } } z _ { 1 } ^ { \mathrm { L } } , \ldots , \sqrt { \frac { \alpha } { n _ { \mathrm { L } } } } z _ { n _ { \mathrm { L } } } ^ { \mathrm { L } } , \sqrt { \frac { 1 - \alpha } { n _ { \mathrm { V } } } } z _ { 1 } ^ { \mathrm { V } } , \ldots , \sqrt { \frac { 1 - \alpha } { n _ { \mathrm { V } } } } z _ { n _ { \mathrm { V } } } ^ { \mathrm { V } } \right] , } \end{array}\tag{14}
$$

which has unit norm, so that the submitted cosine similarity equals

$$
z _ { i } ^ { \top } z _ { j } = \frac { \alpha } { n _ { \mathrm { L } } } \sum _ { a = 1 } ^ { n _ { \mathrm { L } } } z _ { a , i } ^ { \mathrm { L } } z _ { a , j } ^ { \mathrm { L } , \top } + \frac { 1 - \alpha } { n _ { \mathrm { V } } } \sum _ { b = 1 } ^ { n _ { \mathrm { V } } } z _ { b , i } ^ { \mathrm { V } } z _ { b , j } ^ { \mathrm { V } , \top }\tag{15}
$$

a weighted average of the member similarities. The fusion needs neither the mode nor any other label of the test sample.

## 5 Experiments and Results

## 5.1 Experimental Setup

All models are trained on our training split (Tab. 1), and all weights are randomly initialized. The validation split, whose drivers and sources are disjoint from the training split, is used for early stopping of Stage 2 and of the video models, for comparing configurations, and for choosing the fusion weight α.

Implementation details. In Stage 1, the encoder is a Transformer with 6 layers, hidden dimension $h = 2 5 6$ and 8 attention heads, and the decoder has 4 layers of the same width. All models are optimized with AdamW, gradient clipping at norm 1.0 and a learning rate schedule of linear warmup followed by cosine decay. Stage 1 is trained for 25,000 iterations with 192 samples per step, a learning rate of $5 \times 1 0 ^ { - 4 }$ , a weight decay of 0.05 and 1,000 warmup steps. Stage 2 is trained for up to 5,000 iterations with 96 samples per step, a learning rate of $3 \times 1 0 ^ { - 4 }$ a weight decay of 0.05 and 200 warmup steps. The video models are trained for up to 10,000 iterations with 256 samples per step, a learning rate of $1 0 ^ { - 3 }$ a weight decay of $1 0 ^ { - 4 }$ and 200 warmup steps. Stage 2 and the video models are stopped early on the validation mean AUC, and both use a temperature of $\tau = 0 . 0 7$ . The loss weights are $\gamma _ { \mathrm { c e n } } = 0 . 1$ $w = 0 . 0 1$ and $\lambda _ { \mathrm { m o t } } = 1$ , or $\lambda _ { \mathrm { m o t } } = 1 0$ for the models with condition dropout (probability 0.15) or pseudo-generator randomization $( \eta = 0 . 0 1 )$ ; the cross-source alignment term of the video branch has weight 0.1. Both branches produce 256-dimensional embeddings. Further details are available in our public code.<sup>1</sup>

Table 2: Verification performance on the WBA2026 test sets (oficial leaderboard, AUC). Same Gen and Dif Gen are the means of their four mode combinations (probe→gallery; S: speaking, L: listening), and Overall is the mean of Same Gen and Dif Gen. Ensembles combine three seeds of each listed configuration. Ours uses diferent ensembles for Same Gen and Dif Gen (Sec. 5.4).
<table><tr><td rowspan="2">Method</td><td>Same Gen, mode pairs Diff Gen, mode pairs</td><td rowspan="2">Same Gen Diff Gen Overall</td><td rowspan="2"></td><td rowspan="2"></td></tr><tr><td> $\mathrm { S {  } S \ S {  } L \ L {  } S \ L {  } L \ S {  } S {  } L \ L {  } S \ L {  } L }$ </td></tr><tr><td>Landmark baseline (ST-GCN) [1] Video baseline [2]</td><td>0.876 0.723 0.751 0.865 0.542 0.516 0.511 0.547 0.866 0.808 0.822 0.879 0.533 0.508 0.513 0.528</td><td>0.804 0.844</td><td>0.529 0.520</td><td>0.666 0.682</td></tr><tr><td>Landmark branch, single seed DiMP (Geo) + auxiliary tasks</td><td>0.813 0.635 0.653 0.727 0.639 0.561 0.563 0.644 0.841 0.664 0.686 0.7670.6530.5680.566 0.656</td><td>0.707 0.739</td><td>0.602 0.611</td><td>0.654 0.675</td></tr><tr><td>Landmark branch, three seeds MAE (Geo, no auxiliary tasks) DiMP (Geo, no auxiliary tasks)</td><td>0.843 0.684 0.7190.7670.610 0.5430.5380.613</td><td>0.753</td><td>0.576</td><td>0.664</td></tr><tr><td>DiMP (Anat)</td><td>0.8260.6520.6780.7550.6440.5690.5540.654 0.875 0.7080.7390.790 0.6550.5670.570 0.654</td><td>0.728 0.778</td><td>0.605 0.611</td><td>0.667 0.695</td></tr><tr><td></td><td>+ pseudo-generator randomization 0.825 0.659 0.675 0.752 0.682 0.568 0.587 0.670</td><td>0.728</td><td>0.627</td><td>0.677</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Video branch, single seed</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Feature differencing  $[ 2 ] \ ( S = 1 9 2 )$ </td><td>0.862 0.750 0.8000.8470.5780.5380.5340.581</td><td>0.815</td><td>0.558</td><td>0.686</td></tr><tr><td>Pixel differencing [2]  $( S = 1 2 8 )$ </td><td>0.879 0.7710.800 0.8730.5640.5410.526 0.597</td><td>0.831</td><td>0.557</td><td>0.694</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Fusion of the two branches</td><td></td><td></td><td></td><td></td></tr><tr><td>Single models (Anat + pixel diff.)</td><td>0.909 0.7810.8150.8720.622 0.562 0.5580.645</td><td>0.844</td><td>0.596</td><td>0.720</td></tr><tr><td>Ours</td><td>0.946 0.845 0.881 0.9280.711 0.5730.587 0.678</td><td>0.901</td><td>0.637</td><td>0.769</td></tr></table>

## 5.2 Benchmark Performance on WBA2026

Table 2 summarizes the results on the test sets. On landmarks alone, the auxiliary tasks raise the overall AUC of a single DiMP model from 0.654 to 0.675, with gains in all eight mode combinations. Compared with the landmark baseline, our landmark models are lower on Same Gen but clearly higher on Dif Gen (0.60 to 0.63 vs. 0.53, except MAE), i.e., they generalize better to unseen generators. The video branch alone reaches 0.815 to 0.831 on Same Gen but only about 0.56 on Dif Gen, below the landmark models. Fusing a single landmark model with a single video model already gives 0.720, above every single-branch result in Tab. 2. For unseen generators, the difusion objective and pseudo-generator randomization matter: DiMP scores higher than a masked autoencoder (MAE) with the same tokens and encoder on Dif Gen but lower on Same Gen, and pseudo-generator randomization raises Dif Gen from 0.611 to 0.627 at the cost of Same Gen. Our final submission, which uses a fused ensemble for Same Gen and a landmark ensemble trained with pseudo-generator randomization for Dif Gen, reaches 0.769 overall.

Table 3: Ablations on the validation split (mean AUC, Sec. 3.2). (a) Combinations of the auxiliary tasks motion energy (ME), region motion (RM) and eye shape (ES) with $w = 0 . 0 1$ (Geo tokens; ensemble of three seeds); the first row is plain DiMP. (b) Fusion weight α $( \alpha = 1$ : landmarks only, $\alpha = 0 \colon$ video only) for the ensemble with feature-diferencing video models and for the final Same Gen file. (c) Token grouping under an identical recipe (with all three auxiliary tasks; ensembles of three seeds per grouping).  
(a) Auxiliary tasks
<table><tr><td colspan="2">ME RM ES AUC</td><td></td></tr><tr><td colspan="2">√ √ √ V √</td><td>0.811 0.827</td></tr></table>

(b) Fusion weight α  
(c) Token grouping
<table><tr><td>α</td><td>Feat. diff. Final</td></tr><tr><td>1.0 0.856</td><td>0.862</td></tr><tr><td>0.9 0.891</td><td>0.906</td></tr><tr><td>0.8 0.912</td><td>0.933</td></tr><tr><td>0.7 0.922</td><td>0.946</td></tr><tr><td>0.6 0.926</td><td>0.952</td></tr><tr><td>0.5 0.925</td><td>0.952</td></tr><tr><td>0.0 0.901</td><td>0.937</td></tr></table>

<table><tr><td>Tokens</td><td>AUC</td></tr><tr><td>Geo</td><td>0.834</td></tr><tr><td>Anat</td><td>0.850</td></tr><tr><td>Motion</td><td>0.854</td></tr><tr><td>Geo + Anat</td><td>0.856</td></tr><tr><td>Geo + Motion</td><td>0.856</td></tr><tr><td>Anat + Motion 0.862 All three</td><td>0.862</td></tr></table>

## 5.3 Ablation Studies

Auxiliary tasks (Tab. 3a). Each task alone improves on plain DiMP, and using all three tasks gives the best result (0.834, +2.3 points); ME and ES together are close to it.

Fusion weight (Tab. 3b). We select α on the validation split $( \alpha = 0 . 6$ and 0.5, respectively); the fusion is better than either branch alone over a wide range of α.

Token grouping (Tab. 3c). Anat and Motion tokens score 1.6 and 2.0 points above Geo tokens. Combining groupings improves further: every combination is better than any single grouping, and combining all three gives the best result (0.862).

## 5.4 Analysis

Appearance invariance via source probing. To measure how much information about the avatar remains in the embeddings, we train a linear classifier on frozen embeddings to predict the source identity and, separately, the mode of a sample.

Table 4: Linear Probing Accuracy of Avatar Source and Mode. Accuracy of a linear classifier on frozen embeddings, trained and evaluated on disjoint drivers. Chance is the frequency of the most frequent class.
<table><tr><td>Feature Representation</td><td>Source Acc.</td><td>Chance</td><td>Mode Acc.</td></tr><tr><td>Landmark, Geo tokens</td><td>10.2%</td><td>1.3%</td><td>74.7%</td></tr><tr><td>Landmark, Anat tokens</td><td>10.3%</td><td>1.3%</td><td>75.5%</td></tr><tr><td>Video branch</td><td>2.1%</td><td>1.3%</td><td>60.3%</td></tr></table>

The classifier is trained on one half of the drivers and evaluated on the other half (15,000 samples of the training split), so that it cannot rely on the driver identity. We compare single models, whose embeddings all have 256 dimensions.

As summarized in Tab. 4, the video branch predicts the source only slightly above chance (2.1%, chance 1.3%): diferencing consecutive frames removes most of the avatar’s appearance. The landmark branch predicts it at about 10%, well above chance although far from perfect: the normalization removes the avatar’s static face shape, but the motion still carries some properties of the rendered avatar. The landmark embeddings also predict the mode more accurately than the video embeddings. This analysis uses the training generators only and cannot measure cues that are specific to a generator.

Cross-mode and cross-generator verification. Cross-mode pairs are the hardest for the landmark branch: on the validation split, the landmark models of the final Same Gen file reach 0.937 for S→S but 0.817 and 0.850 for S→L and L→S, and their embeddings predict the mode well (Tab. 4), i.e., they retain how the motion of a driver difers between speaking and listening. The video branch depends less on the mode (0.908 and 0.925 for S→L and L→S), and the fusion raises the cross-mode AUCs to 0.925 and 0.938. On Dif Gen, however, video models, including the video baseline, score only 0.52 to 0.56 and fall below our landmark models (0.58 to 0.63, Tab. 2), although video is stronger on the validation split, which is rendered by the training generators; the video branch thus appears to rely partly on cues specific to these generators. We therefore chose to use only landmark models for Dif Gen.

## 6 Conclusion

We presented an operator verification framework combining difusion masked pretraining, auxiliary behavioral tasks, and an inter-frame diferencing video branch, achieving 0.769 AUC on WBA2026. While linear probing shows video retains minimal avatar info and fusion helps cross-mode verification, generalizing to unseen generators remains the primary challenge.

Acknowledgements. This work was partly supported by JSPS KAKENHI Grant Number JP23H00497 and JST FOREST Grant Number JPMJFR216O.

## References

1. Alsherfawi, A., Zhou, J., Shehata, A., Yagi, Y.: Behavioral signature decoding: Facial landmark-based graph learning for cybernetic avatar authentication. In: IEEE IJCB. pp. 1–10 (2025) 12

2. Chapariniya, M., Odobez, J.M., Dellwo, V., Vuković, T.: Micro-expression-aware avatar fingerprinting via inter-frame feature diferencing. In: IEEE FG (2026) 10, 11, 12

3. Devlin, J., Chang, M.W., Lee, K., Toutanova, K.: BERT: Pre-training of deep bidirectional transformers for language understanding. In: NAACL (2019) 3

4. He, K., Chen, X., Xie, S., Li, Y., Dollar, P., Girshick, R.: Masked autoencoders are scalable vision learners. In: CVPR (2022) 3

5. Ho, J., Jain, A., Abbeel, P.: Denoising difusion probabilistic models. In: NeurIPS (2020) 7

6. Ho, J., Salimans, T.: Classifier-free difusion guidance. In: NeurIPS Workshop on Deep Generative Models and Downstream Applications (2021) 8

7. Kashyap, A.L., Tulyakov, S., Govindaraju, V.: Facial behavior as a soft biometric. In: IAPR ICB (2012) 2

8. Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C., Krishnan, D.: Supervised contrastive learning. In: NeurIPS (2020) 10

9. Li, X., Xu, C., Yagi, Y.: Towards robust and eficient continuous face authentication via disentangled representation learning and adaptive identity completion. In: IEEE IJCB (2025) 2

10. Li, X., Xu, C., Yagi, Y.: OU-CA: A face–voice multimodal benchmark for continuous user authentication. IEEE T-BIOM 8, 880–891 (2026) 2

11. MAKIHARA, Y., MURAMATSU, D., YAGI, Y.: Gait recognition and its applications to forensics. IEICE ESS Fundamentals Review 14(4), 318–328 (2021) 2

12. Mao, Y., Deng, J., Zhou, W., Fang, Y., Ouyang, W., Li, H.: Masked motion predictors are strong 3d action representation learners. In: ICCV (2023) 3

13. van den Oord, A., Li, Y., Vinyals, O.: Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748 (2018) 10

14. Paleari, M., Velardo, C., Huet, B., Dugelay, J.L.: Face dynamics for biometric people recognition. In: IEEE MMSP (2009) 2

15. Radford, A., Narasimhan, K., Salimans, T., Sutskever, I.: Improving language understanding by generative pre-training. Tech. rep., OpenAI (2018) 3

16. Saeed, U.: Facial micro-expressions as a soft biometric for person recognition. PRL 143, 95–103 (2021) 2

17. Saeed, U., Matta, F., Dugelay, J.l.: Person recognition based on head and mouth dynamics. In: IEEE MMSP (2006) 2

18. Sagawa, S., Koh, P.W., Hashimoto, T.B., Liang, P.: Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization. In: ICLR (2020) 9

19. Takemura, N., Makihara, Y., Muramatsu, D., Echigo, T., Yagi, Y.: On input/output architectures for convolutional neural network-based cross-view gait recognition. IEEE TCSVT 29(9), 2708–2719 (2019) 2

20. Tong, Z., Song, Y., Wang, J., Wang, L.: VideoMAE: masked autoencoders are data-eficient learners for self-supervised video pre-training. In: NeurIPS (2022) 3

21. WBA Challenge Organizers: Who is behind the avatar (WBA) challenge 2026. https://www.codabench.org/competitions/17680/ (2026), accessed: 2026-10-05 3

22. Zhang, Z., Zhu, J., Fang, C., Liu, J., Mian, A.S.: Difusion masked pretraining for dynamic point cloud. In: NeurIPS (2026) 3, 5, 7

23. Zhu, W., Ma, X., Liu, Z., Liu, L., Wu, W., Wang, Y.: MotionBERT: A unified perspective on learning human motion representations. In: ICCV (2023) 3