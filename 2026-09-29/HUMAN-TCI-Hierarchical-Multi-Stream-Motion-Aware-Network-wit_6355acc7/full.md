# HUMAN-TCI: Hierarchical Multi-Stream Motion-Aware Network with Torso-Centered Interaction for Text-to-Motion Retrieval

Muhammad Islam<sup>a,b</sup>, Euijoon Ahn<sup>a,b</sup>, Usman Naseem<sup>c</sup> and Tao Huang<sup>a,b</sup>

<sup>a</sup>College ofScience and Engineering, James Cook University, Cairns, 4878, QLD, Australia

<sup>b</sup>Centre for AI and Data Science Innovation, James Cook University, Cairns, 4878, QLD, Australia

<sup>c</sup>School ofComputing, Macquarie University, Sydney, 2113, NSW, Australia

## A R T I C L E I N F O

Keywords:   
Text Guided   
Human Motion   
Retrieval   
Multimedia   
Multi-modality   
Temporal Pattern Recognition   
Torso-Centered Interaction   
Hierarchical Multi-Stream

## A BS T RA C T

Accurate retrieval of human motions is a crucial first step in text-guided human motion modeling and synthesis, as it selects semantically relevant sequences from large datasets and provides grounded references for downstream tasks. Retrieving motions from natural language descriptions remains challenging because sentences can describe multiple actions, overlapping movements, and intricate dependencies between body parts. Existing methods often focus on simple, single-action descriptions and typically process body parts independently or by merely concatenating features, without explicitly modeling how torso movements influence other parts. In addition, their processing pipelines often rely on computationally heavy models, introducing considerable overhead, particularly when modeling longer or more complex motion sequences. This limits learning discriminative motion-pattern representations, reducing retrieval accuracy, interpretability, and eficiency in practical applications. To address these limitations, we propose HUMAN-TCI, a Hierarchical Multi-Stream Motion-Aware Network for text-guided human motion retrieval. HUMAN-TCI employs a three-stream architecture that separately models upper-body, lower-body, and torso motions while explicitly capturing their interactions, allowing torso-related movements to influence the positioning and dynamics of other body parts. By incorporating tailored torso attention, our model efectively recognizes complex human motion patterns, captures fine-grained motion relationships and handles complex multiaction descriptions. Our framework supports retrieval for both simple, single-action sentences and long, compositional descriptions containing sequential or overlapping actions without relying on complex models. Evaluation on the KIT Motion-Language Dataset and HumanML3D demonstrates that HUMAN-TCI achieves better performance compared with prior conventional methods under the evaluated settings. Experimental results confirm that our approach retrieves semantically accurate, plausible, and interpretable motion sequences while remaining computationally eficient, making it suitable for large-scale, real-world applications. Project page (including code and visualization videos): GitHub HUMAN-TCI

## 1. Introduction

Retrieving human motion sequences from natural language descriptions is an essential capability for practical artificial intelligence (AI) systems. In real-world applications such as animation generation, robotics, virtual reality, and human-computer interaction, users often describe desired actions in natural language, and systems must translate these descriptions into accurate motion sequences [1, 2]. Without reliable retrieval, it becomes dificult to locate semantically relevant motion data from large-scale motion repositories, limiting the ability to reuse, analyze, or synthesize meaningful motion patterns. However, this task is inherently challenging because natural language is ambiguous and compositional. A single sentence can describe multiple concurrent or sequential actions with variations in duration, intensity, and body-part involvement. For example, a description such as “a person is raising their arms while walking forward and turning their torso” involves coordinated upper-body, lowerbody, and torso movements that must be jointly understood to retrieve the correct motion. Capturing such fine-grained and structured semantics remains a significant challenge for existing methods.

Existing text-to-motion retrieval methods typically embed the entire motion sequence and textual description into a single global feature vector, which often overlooks finegrained spatial and temporal dependencies [3]. Some approaches model body parts independently, treating the upper and lower body separately but simply concatenating torso features without capturing their influence on other joints [4, 5]. Other methods adopt hierarchical [6, 7] or multi-level learning strategies [8] to encode complex motions. Similarly, multi-instance learning approaches attempt to handle sequential or overlapping actions. However, none of these methods explicitly account for how torso movements afect other body parts or capture torso-related interactions, leaving an important gap injoint-level motion understanding and recognition. As a result, existing approaches often fail to distinguish co-occurring actions or capture nuanced correlations between language and joint-level motion, leading to semantically inaccurate or implausible retrieval results.

Another key challenge in motion retrieval is spatial grounding. Spatial grounding refers to the ability to align semantic concepts in text with specific spatial regions of the human body, such as mapping references like “arms,” “legs,” or “torso” to their corresponding joints and motion patterns. Human motion is inherently hierarchical, as the body forms an articulated kinematic structure with parent–child relationships between joints (e.g., torso, upper body, and lower body). While this structure governs how motion propagates across the body, coordinated interactions between these regions jointly define the overall motion semantics. However, many prior works [3, 9] either treat the body as a single unit or rely on simplified upper- and lower-body representations, limiting their ability to model torso-centric interaction (TCI) and cross-part dependencies (CPD).

This limitation is evident in compositional descriptions involving multiple body parts. For example, in a sentence such as “a person is bending forward while stepping sideways,” accurate retrieval requires capturing both the lateral leg movement and the forward torso inclination. Existing approaches often fail to explicitly model these fine-grained dependencies, particularly when multiple coordinated actions are involved [10], limiting their efectiveness in complex real-world scenarios. To address these challenges, we propose HUMAN-TCI (Torso-Centered Interaction), a Hierarchical Multi-Stream Motion-Aware Network for text-guided human motion retrieval. Our model employs a three-stream architecture to separately encode upper-body, lower-body, and torso motions, and a hierarchical attention mechanism to model their interactions. This design better aligns textual descriptions with body-part-specific motion, improving retrieval for both simple and compositional actions. Prior sequential models [3, 10] model motions using separate streams for diferent modalities and capture temporal dependencies sequentially. In contrast, our approach achieves more accurate, plausible, and interpretable retrieval results without requiring excessively complex architectures.

Our results indicate that incorporating hierarchical multistream modeling and torso-aware attention improves semantic alignment between text and motion representations. These gains are especially noticeable in queries involving complex, compositional actions with multiple interacting body parts.

To the best of our knowledge, this work is among early retrieval frameworks to explicitly model torso-centered cross-stream interactions as shown in Figure 1. By introducing a hierarchical multi-stream architecture with explicit torso-aware interaction modeling, HUMAN-TCI enables fine-grained spatial grounding between natural language descriptions and body-part motion dynamics, leading to more accurate and interpretable retrieval of complex multiaction motions. The main contributions of this work are summarized as follows:

• We propose HUMAN-TCI, a hierarchical multi-stream architecture that decomposes human motion into upperbody, torso, and lower-body streams, enabling structured modeling ofbody-part dynamics for fine-grained text-to-motion retrieval.

• We introduce an explicit TCI mechanism that captures the influence of torso movements on both upper- and lower-body actions, allowing the model to better retrieve motions involving torso-driven dynamics such as bending, twisting, and turning.

• Our approach improves the alignment between natural language descriptions and motion representations by explicitly modeling body-part-specific actions through a multi-stream design. This enables accurate, interpretable retrieval for both simple actions and complex compositional descriptions involving sequential and overlapping movements.

• Extensive experiments on the KIT Motion-Language Dataset and HumanML3D demonstrate that HUMAN-TCI achieves consistent improvements in retrieval accuracy while maintaining a computationally eficient architecture compared to more complex retrieval frameworks.

## 2. Related Work

## 2.1. Text-to-Motion Retrieval

Text-to-motion retrieval (TMR) has recently emerged as an important task that aims to retrieve semantically relevant human motion sequences given a natural language description. Compared with traditional image–text or video–text retrieval tasks, TMR presents additional challenges due to the temporal dynamics of motion data [11] and the compositional structure of human actions [3, 10, 12]. Early work in this area formulates TMR by learning a shared embedding space that aligns motion sequences and textual descriptions using contrastive or triplet-based objectives [13].

Subsequent work establishes TMR as a standalone research problem by extending text-to-motion generation frameworks to retrieval settings and leveraging contrastive learning to align motion and language representations [14]. Transformer-based architectures have also been adopted to improve motion representation learning, where motion sequences are encoded using transformer or vision transformerstyle architectures and aligned with language encoders to enable cross-modal retrieval [15, 16].

More recent methods attempt to improve retrieval performance by introducing richer representations or additional supervision. For example, some approaches incorporate auxiliary modalities such as video to construct unified embedding spaces across text, video, and motion modalities [17]. Other works represent motion sequences as motion patches and apply transformer-based modeling to capture motion-language relationships [18]. Event-level modeling has also been explored by decomposing textual descriptions into sequential sub-actions to better capture temporal correspondences between language and motion [6, 8, 19].

Despite these advances, many existing TMR methods focus on aligning global motion representations with textual embeddings [5, 20, 21]. Such global alignment often fails to capture the fine-grained spatial relationships between diferent body parts or the subtle semantics of complex multi-action descriptions, which are common in real-world motion-language datasets.

![](images/0f062ce1d2f24475cf0903fae2546b243b711a924a9bf8eb29c37b940f2856ae.jpg)  
Figure 1: Overview of the proposed 3Tier-Hierarchical Architecture for text-to-motion retrieval. The framework learns hierarchical body-part-aware representations through dedicated upper-body, lower-body, and torso motion streams. Cross-stream dependencies are captured through torso-aware interaction (Figure 3), enabling fine-grained text–motion alignment in a shared embedding space. Triangles and circles represent text (T) and motion (M) embeddings, respectively, in the higher-dimensional embedding space. Details of the motion encoder are shown in Figure 2, while the text encoder is presented in Figure 4.

## 2.2. Cross-Modal Semantic Alignment

Cross-modal semantic alignment aims to learn a shared representation space where semantically related instances from diferent modalities are projected close together [22– 24]. This paradigm has been widely explored in vision language tasks such as image–text and video–text retrieval. A common approach is to learn global embeddings for each modality and perform matching using similarity measures such as cosine similarity [5, 6]. While efective, global alignment strategies may overlook local correspondences that are important for capturing fine-grained semantic relationships.

To address this limitation, several methods introduce fine-grained alignment mechanisms that model interactions between visual regions and textual phrases or between video frames and words [22, 25]. Researchers have also proposed hierarchical and multi-grained alignment strategies to capture semantic relationships at diferent levels of abstraction [7, 10]. In addition, recent approaches leverage knowledge from large pre-trained models to enhance cross-modal repre sentation learning and improve alignment performance [26– 28].

However, directly applying these cross-modal alignment strategies to 3D human motion remains challenging. Human motion involves complex spatial dependencies between body joints and strong temporal correlations across frames. Moreover, natural language descriptions of motion often include multiple actions that occur sequentially or simultaneously across diferent body parts. Existing alignment frameworks rarely consider these structured dependencies, making it dificult to achieve precise correspondence between language and motion at a fine-grained spatial level.

Therefore, a retrieval framework that explicitly models hierarchical body-part interactions and spatial dependencies between torso, upper-body, and lower-body motions is essential for achieving accurate and interpretable text-tomotion retrieval.

## 3. Methodology

## 3.1. Problem Formulation

We study the TMR task, where the goal is to retrieve a semantically consistent human motion sequence � from a natural language description �. Formally, let � = $\{ \mathbf { m } _ { 1 } , \mathbf { m } _ { 2 } , \dots , \mathbf { m } _ { T } \}$ denote a 3D motion sequence of length �, where each frame $\mathbf { m } _ { t } \in \mathbb { R } ^ { J \times 3 }$ encodes the 3D positions of � joints. Similarly, $\mathbf { W } = \{ w _ { 1 } , w _ { 2 } , \dots , w _ { S } \}$ represents a textual description of length �. The objective is to learn embedding functions $f _ { m } ( \cdot )$ and $f _ { t } ( \cdot )$ that map motion sequences and textual descriptions into a shared feature space , such that semantically related pairs (�, �) are assigned higher similarity scores than non-matching pairs:

$$
\sin \bigl ( f _ { t } ( \mathbf { W } ) , f _ { m } ( \mathbf { M } ) \bigr ) > \sin \bigl ( f _ { t } ( \mathbf { W } ) , f _ { m } ( \mathbf { M } ^ { \prime } ) \bigr ) .\tag{1}
$$

Here, � denotes the ground-truth motion corresponding to the text description �, whereas �<sup>′</sup> represents a nonmatching (negative) motion sample. The function sim(⋅, ⋅) denotes cosine similarity, which measures the similarity between text and motion embeddings in the shared embedding space as shown in Figure 2 . The contrastive objective is employed to learn this shared space by increasing the similarity between semantically corresponding text to motion pairs while decreasing the similarity between mismatched pairs, thereby improving text to motion retrieval. Datasets such as HumanML3D and KIT-ML contain multiple textual descriptions for the same motion sequence, resulting in a one-to-many correspondence between language and motion. During training, we treat each text-motion pair as an independent positive instance, while other samples in the batch serve as negatives, following a standard contrastive learning paradigm.

![](images/278a74279fe6eb3db07d3e5bb3e3f0e5c7dd5025b90d0001b26458725c9ac531.jpg)  
Figure 2: Illustration of the HUMAN-TCI motion pipeline. The input 3D skeletal sequence is first preprocessed and reduced from all joints (21-kitml, 22 H-ml3d) to 5 anatomically meaningful body regions, which are then organized into three body-part streams: upper body, torso, and lower body. The three streams are projected into a shared feature space and passed to the Torso-Center Interaction (TCI) module, detailed in Figure 3, where torso-aware attention models cross-stream dependencies between the upper-body, torso, and lower-body representations. The resulting attended features are then independently encoded using GRU layers to capture temporal motion patterns. Finally, the encoded upper-body, torso, and lower-body features are concatenated and passed through a nonlinear projection layer followed by $L _ { 2 }$ normalization to obtain the final motion embedding for text-to-motion retrieval.

## 3.2. Hierarchical Multi-Stream Motion-Aware Network

Our HUMAN decomposes human motion into three higher-level streams: the upper-body motion stream $\mathbf { M } _ { \mathrm { u p p e r } } ,$ the torso motion stream $\mathbf { M } _ { \mathrm { t o r s o } } ,$ and the lower-body motion stream $\mathbf { M } _ { \mathrm { l o w e r } } .$ Each stream is constructed from anatomically defined joint groups and processed through torsocentered cross-stream interaction before temporal encoding. Rather than explicitly fusing independently obtained streamlevel embeddings, the proposed architecture integrates interstream dependencies through torso-guided attention, then combines the temporally encoded streams to form a unified motion representation. The final motion embedding used for text to motion retrieval is expressed as

$$
{ \bf z } _ { \mathrm { m o t i o n } } = f _ { \mathrm { m o t i o n } } \left( { \bf M } _ { \mathrm { u p p e r } } , { \bf M } _ { \mathrm { t o r s o } } , { \bf M } _ { \mathrm { l o w e r } } \right) ,\tag{2}
$$

where $f _ { \mathrm { m o t i o n } } ( \cdot )$ denotes the proposed hierarchical multistream motion encoder with torso-centered cross-stream attention, and $\mathbf { M } _ { \mathrm { u p p e r } } , \mathbf { M } _ { \mathrm { t o r s o } }$ , and $\mathbf { M } _ { \mathrm { l o w e r } }$ represent the upperbody, torso, and lower-body motion streams, respectively.

## 3.3. Multi-Stream Motion Encoding

Each input motion sequence is represented as a sequence of full-body skeletal poses, comprising 21 joints for KIT-ML and 22 joints for HumanML3D. The joints are first organized into five anatomically defined groups: right arm, left arm, right leg, left leg, and mid-body. Each group is independently projected into a part-specific feature space at the frame level. Torso-aware cross-stream interaction is then applied, in which each of the four limb streams (right arm, left arm, right leg, and left leg) attends to the midbody stream. This interaction allows the network to model spatial relationships between the limbs and the torso before temporal encoding. The torso-informed right and left arm features are then concatenated to form the upper-body stream, while the torso-informed right and left leg features are concatenated to form the lower-body stream. Together with the mid-body stream, this reduces the five anatomical groups to three higher-level streams: upper body, lower body, and torso. Each of these three streams is subsequently processed by a dedicated GRU-based temporal encoder, and the resulting representations are concatenated and projected to obtain the final motion representation.

## 3.3.1. Body-Part Motion Decomposition

Let a motion sequence $\mathbf { M } \in \mathbb { R } ^ { T \times J \times 3 }$ consist of � frames and � skeletal joints. The joints are first divided into five anatomically defined groups: left arm, right arm, left leg, right leg, and mid-body. These groups are then organized into three motion streams: upper body, torso, and lower body.

![](images/1498ce5fa894a62fbf9952149e3fd0f58525c88dc912e398bc955c7a02e15b51.jpg)  
Figure 3: HUMAN-TCI module. Motion is decomposed into upper, lower, and torso streams. Upper/lower streams form queries, while the torso provides keys/values for attention. The attended features are encoded with GRU layers and fused into a unified motion embedding.

Specifically, the upper-body stream combines the left- and right-arm joints, the torso stream contains the mid-body joints, and the lower-body stream combines the left- and right-leg joints.

$$
\begin{array} { r l } & { \mathbf { J } _ { \mathrm { u p p e r } } = \mathbf { J } _ { \mathrm { l e f t - a r m } } \cup \mathbf { J } _ { \mathrm { r i g h t - a r m } } , } \\ & { \mathbf { J } _ { \mathrm { t o r s o } } = \mathbf { J } _ { \mathrm { m i d - b o d y } } , } \\ & { \mathbf { J } _ { \mathrm { l o w e r } } = \mathbf { J } _ { \mathrm { l e f t - l e g } } \cup \mathbf { J } _ { \mathrm { r i g h t - l e g } } . } \end{array}\tag{3}
$$

For HumanML3D, the corresponding joint groups are �<sub>upper</sub> $= \{ 1 3 , 1 4 , 1 6 , 1 7 , 1 8 , 1 9 , 2 0 , 2 1 \} , \mathbf { J } _ { \mathrm { t o r s o } } = \{ 0 , 3 , 6 , 9 , 1 2 , 1 \}$ and $\mathbf { J } _ { \mathrm { l o w e r } } ~ = ~ \{ 1 , 2 , 4 , 5 , 7 , 8 , 1 0 , 1 1 \}$ . For KIT-ML, they are $\mathbf { J _ { \mathrm { u p p e r } } } = \{ 5 , 6 , 7 , 8 , 9 , 1 0 \} , \mathbf { J _ { \mathrm { t o r s o } } } = \{ 0 , 1 , 2 , 3 , 4 \}$ , and $\mathbf { J } _ { \mathrm { l o w e r } } \overset { * * } { = } \{ 1 1 , 1 2 , 1 3 , 1 4 , 1 5 , 1 6 , 1 7 , 1 8 , 1 9 , 2 0 \}$ . The indices follow the zero-based joint indexing used in the implementation.

We extract each body-part motion stream individually. Let part ∈ {upper, torso, lower} denote the corresponding body part. The motion representation for each part is obtained by selecting the joints associated with $\mathbf { J } _ { \mathrm { p a r t } }$ along the joint dimension:

$$
\begin{array} { r } { \mathbf { M } _ { \mathrm { p a r t } } = \mathbf { M } [ : , \mathbf { J } _ { \mathrm { p a r t } } , : ] . } \end{array}\tag{4}
$$

This decomposition separates the motion into anatomically meaningful streams while preserving the temporal ordering of the original motion sequence.

## 3.3.2. Torso-Guided Attention Mechanism

Before temporal encoding, each body-part group is projected into a common feature space. Let $\mathbf { X } _ { \mathrm { u p p e r } } , \ \mathbf { X } _ { \mathrm { l o w e r } } ,$ and $\mathbf { X } _ { \mathrm { t o r s o } }$ denote the resulting frame-level feature representations. The upper- and lower-body streams generate query representations, while the torso features serve as keys and values. This design enables the limb streams to selectively incorporate contextual information from torso dynamics while preserving their own motion characteristics. The torso-guided attention operations are defined as

$$
\begin{array} { r l } & { \mathbf { X } _ { \mathrm { u p p e r } } ^ { \prime } = \mathrm { A t t n } \left( \mathbf { Q } _ { \mathrm { u p p e r } } , \mathbf { K } _ { \mathrm { t o r s o } } , \mathbf { V } _ { \mathrm { t o r s o } } \right) , } \\ & { \mathbf { X } _ { \mathrm { l o w e r } } ^ { \prime } = \mathrm { A t t n } \left( \mathbf { Q } _ { \mathrm { l o w e r } } , \mathbf { K } _ { \mathrm { t o r s o } } , \mathbf { V } _ { \mathrm { t o r s o } } \right) . } \end{array}\tag{5}
$$

where the scaled dot-product attention is given by

$$
\mathrm { A t t n } ( Q , K , V ) = \mathrm { S o f t m a x } \left( \frac { Q K ^ { \top } } { \sqrt { d } } \right) V .\tag{6}
$$

The torso stream does not undergo an additional attention update and is therefore retained as

$$
\mathbf { X } _ { \mathrm { t o r s o } } ^ { \prime } = \mathbf { X } _ { \mathrm { t o r s o } } .\tag{7}
$$

Thus, the resulting three torso-aware representations $\mathbf { X } _ { \mathrm { u p p e r } } ^ { \prime } ,$ $\mathbf { X } _ { \mathrm { t o r s o } } ^ { \prime } .$ , and $\mathbf { X } _ { \mathrm { l o w e r } } ^ { \prime }$ are passed to the temporal encoding stage. This asymmetric interaction enables the torso to provide contextual guidance to the upper- and lower-body streams without introducing an additional self-attention update for the torso stream as shown in Figure 3.

## 3.3.3. Temporal Modeling and Fusion

The torso-aware representations are subsequently processed by stream-specific Gated Recurrent Unit (GRU) encoders to model temporal dependencies across the � frames:

$$
\mathbf { H } _ { \mathrm { p a r t } } = \mathbf { G } \mathbf { R } \mathbf { U } _ { \mathrm { p a r t } } \left( \mathbf { X } _ { \mathrm { p a r t } } ^ { \prime } \right) \in \mathbb { R } ^ { T \times d } ,\tag{8}
$$

where part ∈ {upper, torso, lower}. � denotes the number of frames (i.e., the temporal sequence length) and � denotes the GRU feature dimension. The stream-specific GRU encoders capture the temporal evolution of the torso-informed upperbody and lower-body representations while preserving the original torso dynamics.

The resulting stream-level representations are then integrated using a multi-part fusion operation:

$$
\mathbf { Z } _ { \mathrm { m o t i o n } } = \mathrm { F u s e } \left( \mathbf { H } _ { \mathrm { u p p e r } } , \mathbf { H } _ { \mathrm { t o r s o } } , \mathbf { H } _ { \mathrm { l o w e r } } \right) ,\tag{9}
$$

where Fuse(⋅) denotes concatenation followed by a learnable projection layer. The resulting representation $\mathbf { Z _ { \mathrm { m o t i o n } } }$ is subsequently mapped to the shared motion–text embedding space and used for text–motion retrieval. Through this hierarchical design, HUMAN-TCI captures temporal dynamics within individual body parts while explicitly modeling torsoguided interactions between the upper and lower body.

## 3.4. Text Encoding Module

To enable motion retrieval from natural language descriptions, HUMAN-TCI learns a semantic text representation that captures both linguistic meaning and temporal action structure. As illustrated in Figure 4, the text encoding pipeline consists of two stages: 1) contextual word representation using a pretrained language model and 2) sequential modeling using a recurrent encoder.

## 3.4.1. Contextual Word Embeddings

Given a text description $\textbf { W } = \{ w _ { 1 } , w _ { 2 } , \dots , w _ { S } \}$ of length �, we first obtain contextualized token representations using a pretrained BERT-Large-Cased model as the backbone for textual feature extraction. Instead of relying solely on the final layer, we concatenate hidden states from multiple upper layers to capture rich semantic and syntactic information.

![](images/5cbb5fb5889254cee1fe1622669118e3a7f1643d2b26f24ad796bb8444939829.jpg)  
Figure 4: Detail proposed text encoding pipeline in HUMAN-TCI. A natural language description is first tokenized and encoded using BERT to obtain contextualized token-level hidden representations. The resulting sequence of BERT hidden states is then passed to a bidirectional LSTM to model sequential dependencies and capture long-range contextual information. The resulting text representation is subsequently projected into the shared motion–text embedding space, where it is aligned with the corresponding motion representation for fine-grained text-to-motion retrieval.

Formally, let $\mathbf { E } ^ { ( l ) } \in \mathbb { R } ^ { S \times d }$ denote the hidden states from layer �. We select the four layers from 12 to 15 and construct the word embedding matrix as:

$$
\mathbf { E } _ { \mathrm { B E R T } } = \mathrm { C o n c a t } \left( \mathbf { E } ^ { ( L - i ) } \right) _ { i = 1 2 } ^ { 1 5 } \in \mathbb { R } ^ { S \times d ^ { \prime } } ,\tag{10}
$$

where � is the total number of layers and $d ^ { \prime } = 4 d$ . This multi-layer aggregation captures both high-level semantics and intermediate linguistic features, which helps describe complex human actions.

## 3.4.2. Sequential Sentence Encoding

Natural language descriptions of motion often contain temporal cues (e.g., “walking while waving”), requiring sequential modeling beyond static embeddings. Therefore, the contextual token embeddings are processed by a multilayer Long Short-Term Memory (LSTM) network:

$$
\mathbf { h } _ { t } = \mathrm { L S T M } ( \mathbf { E } _ { \mathrm { B E R T } } ( t ) , \mathbf { h } _ { t - 1 } ) .\tag{11}
$$

After processing the full sequence, the final hidden state of the top LSTM layer is taken as the sentence representation:

$$
{ \bf z } _ { \mathrm { t e x t } } ^ { r a w } = { \bf h } _ { S } .\tag{12}
$$

To ensure compatibility with the hierarchical motion representation, the hidden dimension of the LSTM is set to three times the base embedding size, enabling the model to implicitly encode information relevant to upper-body, lowerbody, and torso motions.

## 3.4.3. Projection to the Joint Embedding Space

The raw sentence embedding is further projected into the shared motion and text embedding space using a linear transformation:

$$
\begin{array} { r } { { \bf z } _ { \mathrm { t e x t } } = { \bf F } _ { p } { \bf z } _ { \mathrm { t e x t } } ^ { r a w } + { \bf b } _ { p } , } \end{array}\tag{13}
$$

where $\mathbf { F } _ { p }$ and ${ \bf b } _ { p }$ are learnable parameters.

## 3.4.4. CLIP Text Encoder.

We also experiment with a pretrained Contrastive Language Image Pre-training (CLIP) text encoder to evaluate the impact of large-scale vision–language pretraining on motion retrieval. Specifically, we use the pretrained ViT-B/32 CLIP model, with the text encoder initialized from its publicly released pretrained weights. The CLIP text encoder is kept frozen during training, such that only the subsequent projection layer and the motion-side components are optimized. Input sentences are tokenized using the CLIP tokenizer with the model’s maximum supported context length of 77 tokens; sequences exceeding this limit are truncated, while shorter sequences are padded according to the CLIP implementation. The resulting 512-dimensional CLIP text representation is then passed through the same text projection head used in the baseline configuration and mapped into the shared 256-dimensional motion–text embedding space. This configuration keeps the downstream architecture and embedding dimensionality unchanged, allowing us to evaluate the efect of the CLIP-based textual representation independently. The resulting variant therefore assesses HUMAN-TCI’s robustness to alternative pretrained textual representations while maintaining a consistent retrieval framework.

## 3.5. Model Training and Loss Function

Hierarchical motion embeddings $Z _ { \mathrm { m o t i o n } }$ from the multistream motion encoder and projected text embeddings ${ z } _ { \mathrm { t e x t } }$ are mapped into a shared embedding space $\mathcal { Z }$

$$
z _ { \mathrm { m o t i o n } } , z _ { \mathrm { t e x t } } \in \mathcal { Z } \subset \mathbb { R } ^ { d } .\tag{14}
$$

The similarity between a text query and a candidate motion sequence is computed using cosine similarity:

$$
S ( z _ { \mathrm { t e x t } } , Z _ { \mathrm { m o t i o n } } ) = \frac { z _ { \mathrm { t e x t } } \cdot Z _ { \mathrm { m o t i o n } } } { \| z _ { \mathrm { t e x t } } \| \| Z _ { \mathrm { m o t i o n } } \| } .\tag{15}
$$

Table 1  
Comparison with state-of-the-art methods on KIT-ML [29] and HumanML3D [12].
<table><tr><td rowspan="2">Method</td><td rowspan="2">Pub. Year</td><td colspan="4">KIT-ML</td><td colspan="4">HumanML3D</td></tr><tr><td>R@1 ↑</td><td>R@5 ↑</td><td>R@10 ↑</td><td>MedR↓</td><td>R@1 ↑</td><td>R@5 ↑</td><td>R@10 ↑</td><td>MedR↓</td></tr><tr><td>T2M [12]</td><td>CVPR&#x27;22</td><td>3.37</td><td>16.87</td><td>27.71</td><td>28</td><td>1.80</td><td>7.12</td><td>12.47</td><td>81</td></tr><tr><td>MotionCLIP [30]</td><td>ECCV&#x27;22</td><td>4.87</td><td>20.09</td><td>31.57</td><td>26</td><td>2.33</td><td>12.77</td><td>18.14</td><td>103</td></tr><tr><td>TEMOS [31]</td><td>ECCV&#x27;22</td><td>7.11</td><td>24.10</td><td>35.66</td><td>24</td><td>2.12</td><td>8.26</td><td>13.52</td><td>173</td></tr><tr><td>MoT [3]</td><td>SIGIR&#x27;23</td><td>6.23</td><td>23.92</td><td>37.15</td><td>20</td><td>2.61</td><td>10.66</td><td>17.79</td><td>60</td></tr><tr><td>TMR [14]</td><td>ICCV&#x27;23</td><td>7.23</td><td>28.31</td><td>40.12</td><td>17</td><td>5.68</td><td>20.34</td><td>30.94</td><td>28</td></tr><tr><td>HSA [7]</td><td>SIGIR&#x27;24</td><td>9.29</td><td>29.01</td><td>40.97</td><td>16</td><td>7.14</td><td>24.02</td><td>34.67</td><td>24</td></tr><tr><td>MGSI [8]</td><td>MM&#x27;24</td><td>8.91</td><td>29.64</td><td>40.84</td><td>16</td><td>6.61</td><td>23.91</td><td>34.74</td><td>24</td></tr><tr><td>Messi-B [3]</td><td>SIGIR&#x27;23</td><td>3.20</td><td>15.70</td><td>25.30</td><td>34</td><td>2.40</td><td>10.50</td><td>17.70</td><td>68</td></tr><tr><td>DTL [13]</td><td>MM&#x27;23</td><td>6.77</td><td>23.18</td><td>37.24</td><td>18</td><td>2.30</td><td>10.06</td><td>16.40</td><td>76</td></tr><tr><td>RetNet [15]</td><td>PatR&#x27;26</td><td>9.59</td><td>30.56</td><td>43.07</td><td>15</td><td>7.61</td><td>25.65</td><td>35.04</td><td>24</td></tr><tr><td>HUMA-TCI (Ours)</td><td></td><td>9.96</td><td>32.31</td><td>47.07</td><td>13</td><td>8.21</td><td>27.17</td><td>38.87</td><td>16</td></tr></table>

Arrows indicate whether higher (↑) or lower (↓) values are better. Bold indicates the best performance in each column.

We explore two widely-used metric learning objectives: symmetric triplet loss and InfoNCE loss. Both aim to bring matching text–motion pairs closer while separating nonmatching pairs in the shared embedding space.

Symmetric Triplet Loss Given a batch of size � with paired embeddings $\{ ( z _ { \mathrm { t e x t } , i } , Z _ { \mathrm { m o t i o n } , i } ) \}$ , the symmetric triplet loss is defined as:

$$
\begin{array} { r l } { \mathcal { L } _ { \mathrm { T r i p l e t } } = \displaystyle \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \bigg [ \operatorname* { m a x } _ { j \neq i } \big ( \alpha + S ( z _ { \mathrm { t e x t } , i } , Z _ { \mathrm { m o t i o n } , j } ) } & { } \\ { - S ( z _ { \mathrm { t e x t } , i } , Z _ { \mathrm { m o t i o n } , i } ) \big ) _ { + } } \\ { + \operatorname* { m a x } _ { j \neq i } \big ( \alpha + S ( z _ { \mathrm { t e x t } , j } , Z _ { \mathrm { m o t i o n } , i } ) } & { } \\ { - S ( z _ { \mathrm { t e x t } , i } , Z _ { \mathrm { m o t i o n } , i } ) \big ) \bigg ] , } \end{array}\tag{16}
$$

where $( x ) _ { + } = \operatorname* { m a x } ( 0 , x )$ and � is a margin hyperparameter. The index � corresponds to the hardest negative sample in the batch.

InfoNCE Loss We adopt the InfoNCE loss as our primary training objective due to its superior performance. It is formulated as a symmetric cross-entropy loss:

$$
\begin{array} { r l }   { \mathcal { L } _ { \mathrm { I n f o N C E } } = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } [ \log \frac { \exp ( S ( z _ { \mathrm { t e x t } , i } , Z _ { \mathrm { m o t i o n } , i } ) / \tau ) } { \sum _ { j = 1 } ^ { B } \exp ( S ( z _ { \mathrm { t e x t } , i } , Z _ { \mathrm { m o t i o n } , j } ) / \tau ) }  } \end{array}\tag{17}
$$

where � is a temperature parameter. During inference, we encode a text query and compare it against all motion embeddings using cosine similarity. We retrieve the top-� motions based on similarity scores.

## 4. Experiments

## 4.1. Datasets

We evaluate HUMAN-TCI on two widely used human motion retrieval datasets: HumanML3D [12] and KIT Motion Language (KIT-MoCap) [29], both providing one or more textual descriptions per motion sequence.

Data Representation: In both datasets, each joint is represented with � = 9 features: six for continuous rotation and three for rotation-invariant forward-kinematics joint positions. Preprocessing follows the HumanML3D pipeline [12].

KIT Motion-Language Dataset: KIT-MoCap contains 3,911 full-body motions in Master Motor Map (MMM) format [29] with 6,278 English textual annotations. Each motion may have multiple descriptions (e.g., “A human walks two steps forwards, pivots 180 degrees, and walks two steps back”). For evaluation, 938 textual queries are used to search among 734 motion sequences after removing redundant queries.

HumanML3D Dataset: HumanML3D [12] builds on - AMASS and HumanAct12, with additional textual annotations, totaling 14,616 motion sequences and 44,970 descriptions. For retrieval, 8,401 textual queries search among 4,198 motion sequences. Since descriptions often correspond to motion subsequences, this dataset enables finer-grained retrieval of specific motion segments.

These datasets provide paired text and human motion sequences, covering a diverse set of actions and natural language descriptions. In particular, HumanML3D includes many compositional and multi-action descriptions, making it well-suited for evaluating fine-grained text–motion alignment. Similarly, KIT includes a wide range of motion categories with varying complexity, enabling comprehensive evaluation of retrieval performance.

An analysis of these datasets reveals that over 90% of motion sequences involve multiple action events, and their corresponding textual descriptions often include sequential or overlapping actions [19]. Many descriptions explicitly reference torso-related movements such as bending, twisting, or turning, highlighting the importance of fine-grained spatial modeling in text-guided motion retrieval.

## 4.2. Experimental setup

The proposed HUMAN-TCI model aligns textual descriptions and motion sequences in a shared embedding space. The motion encoder is a hierarchical, multi-stream GRU network (Upper-Lower-Torso GRU), while the text encoder uses either BERT-Large-Cased followed by a multilayer LSTM or CLIP’s pretrained text encoder. We experimented with both InfoNCE and symmetric triplet losses; results reported here use InfoNCE due to superior performance.

## 4.2.1. Implementation Details

Training is performed using the Adam optimizer with a learning rate of $3 \times 1 0 ^ { - 5 }$ and a batch size of 96. For the KIT-ML dataset, the model is trained for 120 epochs using a cosine annealing learning-rate scheduler with $T _ { \mathrm { m a x } } = 1 2 0$ For HumanML3D, the model is trained for 30 epochs using a MultiStepLR scheduler with a milestone at epoch 20 and a decay factor of $\gamma = 0 . 1$ . The random seed is fixed to 42 to ensure reproducibility. A dropout rate of 0.2 is applied to the final projection layers to improve generalization, and the shared embedding space is set to a dimensionality of 256.

The model is trained using a contrastive learning objective based on the InfoNCE loss, with a temperature of 0.07. The loss configuration uses a maximum logit scale of 100 and a margin of 0.01, which is applied to the positive matching pairs. We further apply label smoothing with a factor of 0.1 to both motion-to-text and text-to-motion contrastive objectives. Both motion and text embeddings are $\ell _ { 2 } .$ -normalized prior to similarity computation, enabling cosine-similarity-based retrieval. Bidirectional matching is enforced by computing the contrastive objective in both motion-to-text and text-to-motion directions. We also incorporate a hard-negative penalty (HNP) weighted by 0.05, selecting the highest-scoring non-matching sample in the mini-batch as the hardest negative. A margin-based penalty is applied when its similarity approaches the positive pair, encouraging better separation between positive and challenging negative samples.

For data processing, motion sequences are represented using a continuous 6D rotation representation combined with RIFKE features. For KIT-ML, fixed-length clips of 50 frames are used, whereas HumanML3D uses the original variable sequence lengths. The same data-loading configuration is adopted across the training, validation, and testing splits, with zero worker threads to ensure deterministic and reproducible data loading. The maximum-violation setting is 5 for KIT-ML and 4 for HumanML3D.

Validation is performed periodically during training to monitor retrieval performance. We conduct validation every 100 training iterations. Performance is evaluated using Recall@K (R@1, R@5, and R@10), median rank (MedR), mean rank (MeanR), and semantic relevance measured using SpaCy similarity. The Hydra framework is utilized to provide flexible and modular configuration of model architectures, optimization strategies, and dataset parameters, enabling systematic experimentation across diferent model and dataset configurations.

Table 2  
Ablation study on KIT and HumanML3D datasets.
<table><tr><td>Motion Model</td><td>R01 ↑</td><td>R05 ↑</td><td>R@10 ↑</td><td>MedR↓</td><td>MeanR↓</td></tr><tr><td colspan="6">KIT</td></tr><tr><td>Hier-2TGRU</td><td>3.71</td><td>15.22</td><td>28.57</td><td>31.00</td><td>72.34</td></tr><tr><td>Hier-3TGRU</td><td>5.44</td><td>21.83</td><td>36.20</td><td>21.00</td><td>61.54</td></tr><tr><td>Hier-3TGRU-Att</td><td>7.86</td><td>29.65</td><td>44.48</td><td>14.00</td><td>26.97</td></tr><tr><td>Hier-3TGRU-Att-HNP</td><td>9.96</td><td>32.31</td><td>47.07</td><td>13.00</td><td>18.51</td></tr><tr><td colspan="6">HumanML3D</td></tr><tr><td>Hier-2TGRU</td><td>2.42</td><td>10.56</td><td>21.00</td><td>61.00</td><td>285.7</td></tr><tr><td>Hier-3TGRU</td><td>4.67</td><td>16.43</td><td>28.65</td><td>38.00</td><td>138.0</td></tr><tr><td>Hier-3TGRU-Att</td><td>6.29</td><td>26.37</td><td>33.56</td><td>18.00</td><td>46.27</td></tr><tr><td>Hier-3TGRU-Att-HNP</td><td>8.21</td><td>27.17</td><td>38.87</td><td>16.00</td><td>35.72</td></tr></table>

## 4.2.2. Retrieval and Evaluation

During inference, a textual query is encoded and compared with all precomputed motion embeddings using cosine similarity, and the top-� most similar motions are retrieved. We evaluate performance using standard retrieval metrics, including Recall@� (R@�), Median Rank (MedR), and Mean Rank (MeanR). To further assess semantic alignment, we employ Normalized Discounted Cumulative Gain (NDCG) SPICE, and spaCy-based similarity. Additionally, qualitative visualizations are used to analyze the correspondence between textual descriptions and body-part-specific motion dynamics.

## 4.3. Performance Comparison

Table 1 presents a comprehensive comparison with state-of-the-art methods on the KIT-ML and HumanML3D datasets. Overall, our method achieves consistently strong performance across both datasets, attaining the best results across all reported retrieval metrics while maintaining a lightweight and eficient architecture (Table 4). On the KIT-ML dataset, our method achieves the best performance across all metrics, obtaining an R@1 of 9.96, R@5 of 32.31, an R@10 of 47.08, and a MedR of 13. In particular, the substantial gains in R@5 and R@10 demonstrate improved retrieval consistency at higher recall levels, indicating stronger ranking quality and robustness in capturing diverse motion variations. In comparison, complex transformer-based methods such as TMR [14], RetNet [15], and MoT [3] our method achieve better performance across the reported recall and median-rank metrics.

On the HumanML3D dataset, our method similarly achieves the best performance across all reported metrics, with R@1 of 8.21, R@5 of 27.17, R@10 of 38.87, and MedR of 16. These results demonstrate strong generalization to datasets with more complex and diverse motion descriptions. The improvement in higher-recall metrics further indicates that the proposed model can efectively retrieve

Table 3

relevant motions across a broader range of textual descriptions.

Compared to earlier methods such as T2M [12], MotionCLIP [30], and TEMOS [31], our method shows clear improvements across all reported retrieval metrics, particularly in R@5 and R@10, highlighting its ability to retrieve more relevant candidates within the top-ranked results. Furthermore, compared to sequential baselines such as Messi-B [3], as well as other recent state-of-the-art approaches [12, 13], our method achieves more favorable performance gains across both datasets. These improvements stem from our hierarchical multi-stream design, which explicitly models body-part-specific dynamics while enabling cross-part interactions. In particular, a dedicated torso stream lets the model capture global motion coordination, which prior work often overlooks. This improves alignment between textual descriptions and motion sequences, especially for compositional actions involving multiple body regions.

In contrast, methods that rely on simplified body representations or treat the human body as a single unit struggle to capture fine-grained dependencies, resulting in lower recall performance. Additionally, while transformer-based approaches show strong comparative results, they often incur high computational cost. Our method, based on GRUbased motion encoding and a lightweight text encoding pipeline (BERT-LSTM and CLIP), achieves competitive or superior performance with significantly lower complexity (Table 4). Overall, the results demonstrate that explicitly modeling hierarchical body structure and torso-centric interactions within a multi-stream framework, while employing a lightweight architecture such as GRU instead of more complex transformer-based models, achieves a strong balance between retrieval accuracy, robustness, and computational eficiency.

## 4.4. Ablation Study and Analysis

To validate the efectiveness of the proposed HUMAN-TCI framework, we conduct a comprehensive ablation study analyzing architectural design, body-part decomposition, cross-part interactions, and language representations. We evaluate all experiments on the KIT and HumanML3D datasets using standard text-to-motion retrieval metrics.

## 4.4.1. Impact ofHierarchical Body-Part Modeling

We first evaluate the contribution of hierarchical motion decomposition. The simple GRU models the human body using two streams, whereas our proposed Hier-3TGRU adds a torso stream to explicitly capture interactions between upper- and lower-body movements.

As shown in Table 2, incorporating torso modeling consistently improves retrieval performance across both datasets. This validates our hypothesis that the torso acts as a structural and semantic bridge, enabling better modeling of coordinated and compositional human motions.

## 4.4.2. Efect of Cross Body-Part Attention

To further capture interdependencies between body parts, we introduce a tailored cross-attention mechanism at the body-part level. Unlike prior work that relies on simple concatenation [3], our approach explicitly models the torso’s influence on both upper- and lower-body streams.

Ablation study on KIT and HumanML3D datasets: BERT-LSTM vs CLIP text models. Best results in bold.
<table><tr><td rowspan="2">Text Model</td><td colspan="3">KIT</td><td colspan="3">HumanML3D</td></tr><tr><td>R01 ↑</td><td>R05 ↑</td><td>R010 ↑</td><td>R01 ↑</td><td>R05 ↑</td><td>R@10 ↑</td></tr><tr><td>BERT-Large</td><td>7.82</td><td>29.57</td><td>44.27</td><td>6.84</td><td>25.1</td><td>36.54</td></tr><tr><td>CLIP</td><td>9.96</td><td>32.31</td><td>47.08</td><td>8.21</td><td>27.17</td><td>38.87</td></tr></table>

Computational eficiency comparison of text-to-motion retrieval models on KIT-ML and HumanML3D. Full inference time is reported in minutes.
<table><tr><td>Model</td><td>Body Parts (J)</td><td>Batch Size</td><td colspan="2">Time (m)</td></tr><tr><td></td><td></td><td></td><td>KIT-ML</td><td>HumanML3D</td></tr><tr><td>BERT-LSTM + GRU</td><td>5</td><td>32</td><td>4.47</td><td>4.82</td></tr><tr><td> $\mathsf { C L I P } + \mathsf { G R U }$ </td><td>5</td><td>32</td><td>2.36</td><td>2.51</td></tr><tr><td>Full Joint Model</td><td>22</td><td>32</td><td>6.12</td><td>6.45</td></tr></table>

Although the attention variant shows competitive performance (Table 2), the results indicate that structured hierarchical modeling already captures strong dependencies, while cross-attention further refines inter-part relationships. Notably, torso interactions play a central role in improving compositional motion understanding.

## 4.4.3. Text Encoder Analysis: BERT vs CLIP

The comparison between BERT and CLIP is presented in Table 3. While BERT-based models perform reasonably well, replacing them with CLIP yields substantial improvements across all metrics. This demonstrates that visionlanguage pretraining enables stronger semantic alignment between textual descriptions and motion representations.

## 4.4.4. Computational Eficiency Analysis

To improve computational eficiency, we aggregate skeletal joints into five body-part representations (� = 5), significantly reducing inference time while preserving essential motion dynamics. As shown in Table 4, this design achieves nearly 2x faster inference than processing all joints (� ∼ 22). At the text encoding level, BERT-LSTM models sequential dependencies and captures fine-grained action descriptions, but its sequential nature introduces additional computational overhead. In contrast, CLIP leverages a parallelizable transformer architecture, providing faster embedding computation and more efective cross-modal alignment. Consequently, CLIP-based models ofer a favorable tradeof between speed and retrieval accuracy, making them suitable for real-time or large-scale text-to-motion retrieval applications.

![](images/160de90f87d9ec93e52dc78086c0f08ed9368d26ce5b41fd26452fcc88f75ed4.jpg)  
Figure 5: 3D skeleton visualization depicting the spatiotemporal evolution of human motion across sequential frames. The visualization represents the temporal progression of articulated body joints, highlighting variations in body posture, joint movements, and structural coordination during motion execution. Such skeletal representations provide a compact view of human dynamics for learning discriminative motion features.

## 4.4.5. Semantic Evaluation

Beyond retrieval accuracy, we evaluate semantic alignment using NDCG SPICE, and spaCy similarity. These metrics provide complementary insights into ranking quality, semantic structure, and linguistic similarity. As shown in Table 5, our model achieves improved semantic consistency, particularly for compositional motion descriptions.

Overall, the ablation study demonstrates that: (i) torso modeling plays a critical role in capturing coordinated motion, (ii) hierarchical decomposition enables fine-grained representation learning, (iii) cross-attention enhances interpart relationships, and (iv) CLIP, as a vision–language model, significantly improves both performance and eficiency, while BERT-Large efectively captures semantic information in sequential and compositional long descriptions. These findings validate the efectiveness of HUMAN-TCI as a robust and scalable framework for text-to-motion retrieval.

## 4.5. Visualization Results

We present qualitative results to evaluate the alignment between textual descriptions and retrieved motion sequences. We select a diverse set of queries, including simple, sequential, and compositional actions, to assess the model’s ability to capture fine-grained motion dynamics and coordinated human behavior.

Table 5  
Semantic evaluation of text-to-motion retrieval models on KIT-ML and HumanML3D. Higher values indicate better semantic alignment.
<table><tr><td>Model</td><td>Dataset</td><td colspan="2">nDCG ↑</td></tr><tr><td></td><td></td><td>SPICE</td><td>spaCy</td></tr><tr><td>Upper-Lower GRU</td><td>KIT-ML</td><td>0.271</td><td>0.706</td></tr><tr><td>Hier-3TGRU</td><td>KIT-ML</td><td>0.263</td><td>0.697</td></tr><tr><td>Hier-3TGRU + CLIP</td><td>KIT-ML</td><td>0.339</td><td>0.770</td></tr><tr><td>Upper-Lower GRU</td><td>HumanML3D</td><td>0.318</td><td>0.723</td></tr><tr><td>Hier-3TGRU</td><td>HumanML3D</td><td>0.316</td><td>0.729</td></tr><tr><td>Hier-3TGRU + CLIP</td><td>HumanML3D</td><td>0.355</td><td>0.824</td></tr></table>

The selected queries span simple upper-body actions (e.g., “a person rubs their hands together”), sequential motions involving posture transitions (e.g., “kneeling”, “crawling”, and “standing”), and complex compositional actions combining upper- and lower-body movements (e.g., “stepping” and “kicking”, or repeated “throwing” actions). These examples highlight how textual verbs and action phrases are grounded in corresponding body-part movements, with the torso acting as a central coordinating component.

![](images/14547d7cd7cc7282c79ec79b03631566d13457275247408a873cf9f46ad22ef3.jpg)  
Figure 6: Visualization of human motion using the SMPL body model, showing the reconstructed 3D human mesh over sequential frames. The SMPL-based representation preserves detailed body geometry and pose variations while capturing the spatiotemporal dynamics of articulated human movements, enabling an intuitive interpretation of learned motion patterns.

The visualization results of the skeleton representation and full-body SMPL demonstration, presented in Figures 5 and 6, demonstrate that HUMAN-TCI efectively captures both local and global motion patterns. For simple queries, the model retrieves precise fine-grained movements aligned with the described actions. For longer and compositional descriptions, it preserves temporal ordering and inter-part dependencies, correctly associating multiple verbs with sequential and overlapping motion segments. Notably, torsodriven actions such as crawling, kicking, and throwing are accurately retrieved, reflecting the model’s ability to capture how torso dynamics influence and regulate upper- and lowerbody movements.

To illustrate torso-centered interaction at the level of an individual query, we examine the query “A person takes five slow forward steps” (Figure 7). Each limb attends nonuniformly to the torso sequence, with peak attention occurring at frame 65 for the Right Arm (1.66%), frame 0 for the Left Arm (2.38%), frame 81 for the Right Leg (1.92%), and frame 41 for the Left Leg (1.44%), the four peaks are visually distinct and largely non-overlapping (Figure 7, top row and heatmap), indicating each limb is anchoring to a different torso configuration rather than converging on a single dominant frame. The attention entropy for all four limbs is similar (� ≈ 4.5–4.6), suggesting a comparably difuse-butpeaked distribution across limbs for this query, rather than one limb dominating the torso context while others attend near-uniformly. The cumulative attention curve (Figure 7, bottom-right) shows that 50% of total torso-attention mass is accumulated by frame 47 (� ) and 90% by frame 88 (� ), indicating the model concentrates the majority of its torso reliance within roughly the first half of the sequence, consistent with the “five slow forward steps” description, where the gait pattern is established early and subsequent frames largely reinforce rather than introduce new torso context. Notably, the Right Leg and Left Leg attention peaks (frames 81 and 41) are separated by roughly 40 frames, plausibly corresponding to the alternating left–right footstrike pattern inherent to walking, while the Left Arm’s peak at frame 0 reflects the model anchoring to the initial standing posture before forward motion begins. This per-query breakdown provides direct, frame-level evidence that our proposed torso-centered interaction mechanism dynamically and selectively routes each limb’s representation through kinematically distinct torso states, rather than applying a fixed or redundant attention pattern.

![](images/4c53e8a0049fca7ed3c08d98c0cd0bc2f34186dbd65d3002e131572a5d5e40ab.jpg)

![](images/662acf45e12414889d2f85d6d4f2d8535fc6f02e9f353baee62bbea4b51bde25.jpg)

![](images/027029a86a07fd7da48488690df547cb66dfd1ddd91b268637fcdcf2a2c11dc8.jpg)

![](images/936591957229280a209692cbf4cab759439e40fea38edb08e3c2c69a54753003.jpg)

![](images/543c214fc992cc2c12a2682853eefdde13cfd85aae5233e9349a76159549b4be.jpg)

![](images/92954814f9eda08cc0016cf307e953dddc9c4b84d43312aad164aa49d97e5014.jpg)

![](images/88901040e0b8939ed6dafcff26677b03b431f8c15bd9546df6934b07dbe8131f.jpg)

![](images/872343a6994d786747cd41074c530144f08628afa7ed2b10be63bb9c2a8c9bf2.jpg)  
Figure 7: Detailed torso-centered attention analysis for the query “A person takes five slow forward steps.” Top row: per-limb attention distribution over torso frames for Right Arm, Left Arm, Right Leg, and Left Leg, with peak attention frame (�) and Shannon entropy (�) annotated for each. Middle-left: attention heatmap across all four limbs (rows) and torso frames (columns), row-normalized for visual contrast, with diamond markers indicating each limb’s peak-attention frame. Middle-right: peak (solid) versus mean±std (light) attention magnitude per limb. Bottom-left: all four limb attention curves overlaid for direct comparison. Bottom-right: total attention (summed across limbs) and its cumulative distribution over the sequence, with $F _ { 5 0 }$ and $F _ { 9 0 }$ marking the frames by which 50% and 90% of cumulative attention mass is reached, respectively.

A central design principle of our 3T-hierarchical architecture is torso-centered interaction: rather than encoding each limb’s motion independently, every limb branch (arms, legs) queries the torso branch via AttentionFuse, using the torso’s temporal sequence as the source of Key and Value context. This is motivated by the observation that torso pose and orientation act as the anchor around which limb articulation is organized, the torso provides the frame of reference (e.g., facing direction, balance, weight shift) that gives limb movement its meaning. To verify that the model actually exploits this design rather than treating it as a redundant pathway, we visualize the learned torso-attention weights in Figure 8. Across all examined queries, attention is sharply non-uniform: each limb selectively concentrates its reliance on a small number of torso frames rather than distributing attention uniformly across the sequence, which it is free to do since torso attention is not explicitly supervised. This confirms that the model has learned to treat the torso as an active, frame-selective reference signal, consistent with our architectural hypothesis, rather than as a static or uninformative context. Moreover, the location of peak torso attention aligns with visually salient torso configurations (e.g., a shift in stance mid-stride, or the torso’s compressed pose at the apex of a jump), indicating the model anchors limb representations to kinematically meaningful torso states, not arbitrary frames. This provides direct provides qualitative evidence for the torso-centered interaction our architecture is designed around.

![](images/2161b24f0e42c1e3aa956e59fe15761ca93d8329fb167e01a43570d9ea8812ce.jpg)  
Figure 8: Visualization of Torso-Centered Interaction learned by our proposed AttentionFuse module. In our 3T-hierarchica architecture, each limb branch (arms, legs) queries the torso branch to selectively retrieve temporally-relevant torso context, rather than encoding limb motion in isolation. For nine representative test queries, we plot the resulting torso-attention distribution: the x-axis is the torso frame index and the y-axis is the normalized attention weight (%) each limb assigns to that frame. The bold navy curve is the mean attention across all four limbs (aggregate torso reliance); the thin red/green curves show per-limb (arm/leg) attention. Orange diamonds mark the frames of peak torso reliance, automatically extracted via cumulative-attention sampling.

Overall, these qualitative results confirm that HUMAN-TCI robustly aligns textual semantics with human motion by explicitly modeling torso-centered interactions. This enables efective understanding of action verbs, compositional structures, and coordinated motion patterns, leading to semantically consistent and physically plausible retrieval outcomes.

## 5. Conclusion and Future Work

In this work, we addressed the challenging task of textto-motion retrieval, particularly focusing on the limitations of existing approaches in handling compositional language and complex body-part dependencies. While prior methods often rely on holistic or independently modeled representations, they fail to explicitly capture the hierarchical and interactive nature of human motion, especially the influence of torso dynamics on coordinated body movements. To overcome these limitations, we proposed HUMAN-TCI, a Hierarchical Multi-Stream Motion-Aware Network that explicitly models upper-body, lower-body, and torso interactions through a structured multi-stream architecture. By introducing a torso-aware attention mechanism, our approach enables fine-grained interaction between body parts, allowing the model to better capture complex motion semantics and dependencies. This design makes our framework efective for both simple action descriptions and long, compositional sentences involving sequential or overlapping actions. Extensive experiments on the KIT Motion-Language Dataset and HumanML3D demonstrate that HUMAN-TCI consistently achieves superior performance compared to existing methods across multiple evaluation metrics, including Recall@K, Median Rank, and Mean Rank. The results confirm that our method improves retrieval accuracy, enhances semantic alignment and interpretability, and maintains computational eficiency suitable for large-scale applications.

Despite these promising results, several directions remain for future work. We aim to enhance hierarchical semantic alignment through multi-level reasoning to better capture long-range dependencies in complex textual descriptions. Additionally, leveraging cross-modal pretraining and multimodal foundation models may improve generalization and robustness. Incorporating additional modalities such as video or depth could further enrich motion understanding, while exploring finer skeletal representations and motion decomposition may enable more precise and interpretable retrieval.

## References

[1] M. Islam, T. Huang, E. Ahn, U. Naseem, Multimodal generative ai for human motion understanding and generation: A survey and way forward, Information Fusion (2026) 104435.

[2] R. Zhong, B. Hu, Y. Feng, Z. Liu, Q. Qin, X. V. Wang, L. Wang, J. Tan, Finemld: A fine-grained motion latent difusion for human motion prediction in human–robot collaboration, Advanced Engineering Informatics 70 (2026) 104119.

[3] N. Messina, J. Sedmidubsky, F. Falchi, T. Rebok, Text-to-motion retrieval: Towards joint understanding of human motion data and natural language, in: Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2023, pp. 2420–2425.

[4] D. Xu, T. Zheng, Y. Zhang, X. Yang, W. Fu, Mtr-mse: Motion-text retrieval method based on motion semantics expansion, Neurocomputing 648 (2025) 130632.

[5] H. Chen, G. Lyu, C. Xu, J. Yan, X. Yang, C. Deng, Beyond global alignment: Fine-grained motion-language retrieval via pyramidal shapley-taylor learning, arXiv preprint arXiv:2601.21904 (2026).

[6] S. Yu, Z.-A. Wang, K. Yin, Z. Tian, M. Zhang, W. Si, S. Zou, Multimodal motion retrieval by learning a fine-grained joint embedding space, IEEE Transactions on Multimedia (2026).

[7] Y. Yang, H. Shi, H. Zhang, Hierarchical semantics alignment for 3d human motion retrieval, in: Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2024, pp. 1083–1092.

[8] Y. Yang, L. Cao, H. Shi, H. Zhang, Multi-instance multi-label learning for text-motion retrieval, in: Proceedings of the 32nd ACM International Conference on Multimedia, 2024, pp. 5829–5837.

[9] N. Messina, J. Sedmidubsky, F. Falchi, T. Rebok, Joint-dataset learning and cross-consistent regularization for text-to-motion retrieval, ACM Transactions on Multimedia Computing, Communications and Applications 21 (10) (2025) 1–24.

[10] A. Ghosh, N. Cheema, C. Oguz, C. Theobalt, P. Slusallek, Synthesis of compositional animations from textual descriptions, in: Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 1396–1406.

[11] K. Fujiwara, M. Tanaka, Q. Yu, Chronologically accurate retrieval for temporal grounding of motion-language models, in: European Conference on Computer Vision, Springer, 2024, pp. 323–339.

[12] C. Guo, S. Zou, X. Zuo, S. Wang, W. Ji, X. Li, L. Cheng, Generating diverse and natural 3d human motions from text, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 5152–5161.

[13] S. Yan, Y. Liu, H. Wang, X. Du, M. Liu, H. Liu, Cross-modal retrieval for motion and text via droptriple loss, in: Proceedings of the 5th ACM International Conference on Multimedia in Asia, 2023, pp. 1–7.

[14] M. Petrovich, M. J. Black, G. Varol, Tmr: Text-to-motion retrieval using contrastive 3d human motion synthesis, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 9488–9497.

[15] H. Pan, Q. Wang, G. Zhu, Y. Chen, Text-to-motion retrieval by textto-motion generation, Pattern Recognition 180 (2026) 113983.

[16] S. S. Kalakonda, S. Maheshwari, R. K. Sarvadevabhatla, Moragmulti-fusion retrieval augmented generation for human motion, in: 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), IEEE, 2025, pp. 4564–4573.

[17] K. Yin, S. Zou, Y. Ge, Z. Tian, Tri-modal motion retrieval by learning a joint embedding space, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 1596–1605.

[18] R. Lan, H. Sun, M. Zhu, Text-like motion representation for human motion retrieval, in: International Conference on Intelligent Science and Intelligent Data Engineering, Springer, 2012, pp. 72–81.

[19] H. Shi, H. Zhang, Sequence-event semantic consistent learning for text-to-motion retrieval, in: Proceedings of the 33rd ACM International Conference on Multimedia, 2025, pp. 8740–8749.

[20] B. Dang, L. Wu, X. Yang, Z. Yuan, Z. Chen, Segmo: Segment-aligned text to 3d human motion generation, in: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2026, pp. 6946–6955.

[21] W. Wang, D. Gao, P. He, X. Liu, D. Liu, 3d human motion corpus moment retrieval via multi-granularity semantic alignment, in: 2025 IEEE International Conference on Multimedia and Expo (ICME), IEEE, 2025, pp. 1–6.

[22] H. Shi, H. Zhang, Modal-enhanced semantic modeling for finegrained 3d human motion retrieval, in: Proceedings of the 32nd ACM International Conference on Multimedia, 2024, pp. 10114–10123.

[23] L. Bensabath, M. Petrovich, G. Varol, A cross-dataset study for textbased 3d human motion retrieval, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 1932–1940.

[24] Y. Li, S. Wu, Y. Zhu, W. Sun, Z. Zhang, S. Song, Samr: Symmetric masked multimodal modeling for general multi-modal 3d motion retrieval, Displays 87 (2025) 102987.

[25] S. Yan, Y. Wang, X. Du, H. Jin, M. Liu, Improving fine-grained understanding for retrieval in human motion and text, IEEE Signal Processing Letters (2024).

[26] S. S. Kalakonda, Advancing motion with llms: Leveraging large language models for enhanced text-conditioned motion generation and retrieval, Ph.D. thesis, International Institute of Information Technology, Hyderabad (2025).

[27] Z. Zhou, B. Wang, Ude: A unified driving engine for human motion generation, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 5632–5641.

[28] M. Zhang, X. Guo, L. Pan, Z. Cai, F. Hong, H. Li, L. Yang, Z. Liu, Remodifuse: Retrieval-augmented motion difusion model, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 364–373.

[29] M. Plappert, C. Mandery, T. Asfour, The kit motion-language dataset, Big data 4 (4) (2016) 236–252.

[30] G. Tevet, B. Gordon, A. Hertz, A. H. Bermano, D. Cohen-Or, Motionclip: Exposing human motion generation to clip space, in: European Conference on Computer Vision, Springer, 2022, pp. 358–374.

[31] M. Petrovich, M. J. Black, G. Varol, Temos: Generating diverse human motions from textual descriptions, in: European conference on computer vision, Springer, 2022, pp. 480–497.